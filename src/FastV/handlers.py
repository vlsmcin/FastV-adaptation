"""Model-specific VLM handlers for loading, prompt formatting, and input preparation.

Complements ``adapters.py`` (which binds FastV hooks to decoder internals) by
encapsulating model-family I/O behind a uniform ``BaseVLMHandler`` interface:
  * ``LlavaHandler``: HuggingFace LLaVA-1.5 / LLaVA-NeXT models.
  * ``Qwen2VLHandler``: HuggingFace Qwen2-VL and Qwen2.5-VL models.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

try:
    import torch
except ImportError:  # pragma: no cover - allows pure-Python utilities without torch
    torch = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover
    from .fastv_hf import FastVConfig, FastVController

VICUNA_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)

DEFAULT_LLAVA_MODEL_7B = "llava-hf/llava-1.5-7b-hf"
DEFAULT_LLAVA_MODEL_13B = "llava-hf/llava-1.5-13b-hf"
DEFAULT_QWEN2_VL_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN2_VL_MIN_PIXELS = 256 * 28 * 28
DEFAULT_QWEN2_VL_MAX_PIXELS = 1280 * 28 * 28


class BaseVLMHandler(ABC):
    """Abstract interface for VLM model loading, prompt formatting, and input preparation."""

    name: str = "base"

    @classmethod
    @abstractmethod
    def matches(cls, model_id: Optional[str]) -> bool:
        """Return True if this handler supports ``model_id``."""

    def resolve_model_id(
        self,
        model_id: Optional[str] = None,
        requested_size: Optional[str] = None,
        gpu_memory_gib: Optional[float] = None,
    ) -> str:
        """Resolve a local path, shorthand, or HF model ID into a loadable identifier."""
        return model_id or ""

    @abstractmethod
    def load_model(
        self,
        model_id: Optional[str] = None,
        quantization: str = "4bit",
        dtype: Optional[Any] = None,
        attn_implementation: str = "eager",
        fastv_config: Optional[Union[FastVConfig, Dict[str, Any]]] = None,
        device_map: Optional[Any] = "auto",
        revision: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any, FastVController]:
        """Load the HuggingFace model + processor and attach ``FastVController``."""

    @abstractmethod
    def format_prompt(
        self,
        prompt: str,
        processor: Optional[Any] = None,
        include_system_prompt: bool = False,
        append_output: str = "",
        **kwargs: Any,
    ) -> str:
        """Format a user prompt into the model family's expected conversation template."""

    @abstractmethod
    def prepare_inputs(
        self,
        processor: Any,
        prompt: str,
        image: Any,
        device: Any,
        dtype: Any,
        include_system_prompt: bool = False,
        append_output: str = "",
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Optional[Tuple[int, int]]]:
        """Tokenize and preprocess text + image, returning ``(inputs_dict, image_span)``."""

    def _load_hf_model(
        self,
        model_cls: Any,
        model_id: str,
        quantization: str = "4bit",
        dtype: Optional[Any] = None,
        attn_implementation: str = "eager",
        device_map: Optional[Any] = "auto",
        revision: Optional[str] = None,
    ) -> Any:
        """Shared HF model loader handling quantization, device_map, and transformers 4.45-5.x dtype kwargs."""
        if torch is None:
            raise RuntimeError("PyTorch is required to load VLM models.")
        import transformers
        from transformers import BitsAndBytesConfig

        quant = (quantization or "none").strip().lower()
        if dtype is not None:
            resolved_dtype = dtype
        elif quant == "bf16":
            resolved_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        elif quant == "fp32":
            resolved_dtype = torch.float32
        elif quant == "fp16":
            resolved_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            resolved_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        quant_config = None
        if quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=resolved_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quant == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quant not in ("none", "fp16", "bf16", "fp32"):
            raise ValueError(
                f"Unsupported quantization={quantization!r}; choose '4bit', '8bit', or 'none'"
            )

        load_kwargs: Dict[str, Any] = {
            "low_cpu_mem_usage": True,
            "attn_implementation": attn_implementation,
        }
        if revision:
            load_kwargs["revision"] = revision
        if quant_config is not None:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = device_map if device_map is not None else {"": "cuda:0"}
        elif torch.cuda.is_available():
            load_kwargs["device_map"] = device_map if device_map is not None else {"": "cuda:0"}

        tf_ver = tuple(int(p) for p in re.findall(r"\d+", transformers.__version__)[:2])
        if tf_ver >= (4, 56):
            load_kwargs["dtype"] = resolved_dtype
        else:
            load_kwargs["torch_dtype"] = resolved_dtype

        try:
            model = model_cls.from_pretrained(model_id, **load_kwargs)
        except TypeError:
            if "dtype" in load_kwargs:
                load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            else:
                load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
            model = model_cls.from_pretrained(model_id, **load_kwargs)

        model.eval()
        return model

    @staticmethod
    def _move_inputs_to_device(inputs: Dict[str, Any], device: Any, dtype: Any) -> Dict[str, Any]:
        """Move processor outputs to ``device``, casting only floating-point tensors to ``dtype``."""
        out: Dict[str, Any] = {}
        for k, v in inputs.items():
            if torch is not None and torch.is_tensor(v):
                if v.is_floating_point():
                    out[k] = v.to(device=device, dtype=dtype)
                else:
                    out[k] = v.to(device=device)
            else:
                out[k] = v
        return out


class LlavaHandler(BaseVLMHandler):
    """Handler for HuggingFace LLaVA-1.5 / LLaVA-NeXT models."""

    name: str = "llava"

    @classmethod
    def matches(cls, model_id: Optional[str]) -> bool:
        return bool(model_id and "llava" in model_id.lower())

    def resolve_model_id(
        self,
        model_id: Optional[str] = None,
        requested_size: Optional[str] = None,
        gpu_memory_gib: Optional[float] = None,
    ) -> str:
        """Resolve a local path, HF model id, or legacy LLaVA name to a usable model id."""
        if model_id:
            if os.path.exists(model_id):
                return model_id
            lower = model_id.lower().rstrip("/")
            if "/" in model_id and not model_id.startswith(".") and not model_id.startswith("/"):
                return model_id
            if "13b" in lower:
                return DEFAULT_LLAVA_MODEL_13B
            if "7b" in lower:
                return DEFAULT_LLAVA_MODEL_7B
            return model_id

        size = (requested_size or os.environ.get("FASTV_MODEL_SIZE", "")).strip().lower()
        if size == "7b":
            return DEFAULT_LLAVA_MODEL_7B
        if size == "13b":
            return DEFAULT_LLAVA_MODEL_13B

        if gpu_memory_gib is None and torch is not None and torch.cuda.is_available():
            gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if gpu_memory_gib is not None and gpu_memory_gib >= 24.0:
            return DEFAULT_LLAVA_MODEL_13B
        return DEFAULT_LLAVA_MODEL_7B

    def load_model(
        self,
        model_id: Optional[str] = None,
        quantization: str = "4bit",
        dtype: Optional[Any] = None,
        attn_implementation: str = "eager",
        fastv_config: Optional[Union[FastVConfig, Dict[str, Any]]] = None,
        device_map: Optional[Any] = "auto",
        revision: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any, FastVController]:
        """Load HF ``LlavaForConditionalGeneration`` + ``AutoProcessor`` cleanly across transformers 4.45-5.x."""
        if torch is None:
            raise RuntimeError("PyTorch is required to load LLaVA models.")
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        resolved_id = self.resolve_model_id(model_id)
        model = self._load_hf_model(
            model_cls=LlavaForConditionalGeneration,
            model_id=resolved_id,
            quantization=quantization,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            revision=revision,
        )

        proc_kwargs: Dict[str, Any] = {}
        if revision:
            proc_kwargs["revision"] = revision
        processor = AutoProcessor.from_pretrained(resolved_id, **proc_kwargs)

        from .fastv_hf import attach_fastv

        controller = attach_fastv(model, fastv_config)
        return model, processor, controller

    def format_prompt(
        self,
        prompt: str,
        processor: Optional[Any] = None,
        include_system_prompt: bool = False,
        append_output: str = "",
        **kwargs: Any,
    ) -> str:
        """Format a user prompt into LLaVA-1.5 conversation format."""
        if "<image>" in prompt and "ASSISTANT:" in prompt:
            return prompt + append_output
        prefix = f"{VICUNA_SYSTEM_PROMPT} " if include_system_prompt else ""
        user_turn = prompt if "<image>" in prompt else f"<image>\n{prompt}"
        return f"{prefix}USER: {user_turn}\nASSISTANT:{append_output}"

    def prepare_inputs(
        self,
        processor: Any,
        prompt: str,
        image: Any,
        device: Any,
        dtype: Any,
        include_system_prompt: bool = False,
        append_output: str = "",
        image_token_index: int = 32000,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Optional[Tuple[int, int]]]:
        """Tokenize and preprocess text + image for modern HF LLaVA, returning ``(inputs, image_span)``."""
        formatted = self.format_prompt(
            prompt,
            processor=processor,
            include_system_prompt=include_system_prompt,
            append_output=append_output,
        )
        rgb_image = image.convert("RGB") if hasattr(image, "convert") else image
        inputs = processor(images=rgb_image, text=formatted, return_tensors="pt")
        out = self._move_inputs_to_device(inputs, device=device, dtype=dtype)
        from .fastv_hf import detect_image_token_span

        span = detect_image_token_span(out.get("input_ids"), image_token_index=image_token_index)
        return out, span


class Qwen2VLHandler(BaseVLMHandler):
    """Handler for HuggingFace Qwen2-VL and Qwen2.5-VL models."""

    name: str = "qwen2_vl"

    def __init__(
        self,
        min_pixels: Optional[int] = DEFAULT_QWEN2_VL_MIN_PIXELS,
        max_pixels: Optional[int] = DEFAULT_QWEN2_VL_MAX_PIXELS,
    ):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    @classmethod
    def matches(cls, model_id: Optional[str]) -> bool:
        if not model_id:
            return False
        lower = model_id.lower()
        return any(
            tok in lower
            for tok in ("qwen2-vl", "qwen2_vl", "qwen2.5-vl", "qwen2_5_vl", "qwen2vl")
        )

    def resolve_model_id(
        self,
        model_id: Optional[str] = None,
        requested_size: Optional[str] = None,
        gpu_memory_gib: Optional[float] = None,
    ) -> str:
        """Resolve a local path, HF repo ID, or shorthand to a Qwen2-VL / Qwen2.5-VL checkpoint."""
        if model_id:
            if os.path.exists(model_id):
                return model_id
            if "/" in model_id and not model_id.startswith(".") and not model_id.startswith("/"):
                return model_id
            lower = model_id.lower()
            if ("qwen2-vl" in lower or "qwen2_vl" in lower) and "2.5" not in lower and "2_5" not in lower:
                return "Qwen/Qwen2-VL-7B-Instruct"
            return DEFAULT_QWEN2_VL_MODEL
        return DEFAULT_QWEN2_VL_MODEL

    def load_model(
        self,
        model_id: Optional[str] = None,
        quantization: str = "4bit",
        dtype: Optional[Any] = None,
        attn_implementation: str = "eager",
        fastv_config: Optional[Union[FastVConfig, Dict[str, Any]]] = None,
        device_map: Optional[Any] = "auto",
        revision: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any, FastVController]:
        """Load HF Qwen2-VL / Qwen2.5-VL + ``AutoProcessor`` and attach ``FastVController``."""
        if torch is None:
            raise RuntimeError("PyTorch is required to load Qwen2-VL models.")
        import transformers
        from transformers import AutoProcessor

        resolved_id = self.resolve_model_id(model_id)
        model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_cls is None:
            lower = resolved_id.lower()
            if "2.5" in lower or "2_5" in lower:
                model_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
            if model_cls is None:
                model_cls = getattr(transformers, "Qwen2VLForConditionalGeneration")

        # Qwen2-VL / Qwen2.5-VL are natively trained in bfloat16 and can overflow float16.
        if dtype is None and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            q = (quantization or "none").strip().lower()
            if q not in ("fp16", "fp32"):
                dtype = torch.bfloat16

        model = self._load_hf_model(
            model_cls=model_cls,
            model_id=resolved_id,
            quantization=quantization,
            dtype=dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            revision=revision,
        )

        proc_kwargs: Dict[str, Any] = {}
        if revision:
            proc_kwargs["revision"] = revision
        min_pixels = kwargs.get("min_pixels", self.min_pixels)
        max_pixels = kwargs.get("max_pixels", self.max_pixels)
        tf_ver = tuple(int(p) for p in re.findall(r"\d+", transformers.__version__)[:2])
        if min_pixels is not None or max_pixels is not None:
            if tf_ver >= (5, 0):
                size_cfg: Dict[str, int] = {}
                if min_pixels is not None:
                    size_cfg["shortest_edge"] = int(min_pixels)
                if max_pixels is not None:
                    size_cfg["longest_edge"] = int(max_pixels)
                proc_kwargs["size"] = size_cfg
            else:
                if min_pixels is not None:
                    proc_kwargs["min_pixels"] = int(min_pixels)
                if max_pixels is not None:
                    proc_kwargs["max_pixels"] = int(max_pixels)

        try:
            processor = AutoProcessor.from_pretrained(resolved_id, **proc_kwargs)
        except TypeError:
            proc_kwargs.pop("size", None)
            proc_kwargs.pop("min_pixels", None)
            proc_kwargs.pop("max_pixels", None)
            processor = AutoProcessor.from_pretrained(resolved_id, **proc_kwargs)

        from .fastv_hf import attach_fastv

        controller = attach_fastv(model, fastv_config)
        return model, processor, controller

    def format_prompt(
        self,
        prompt: str,
        processor: Optional[Any] = None,
        include_system_prompt: bool = False,
        append_output: str = "",
        system_prompt: Optional[str] = None,
        num_images: int = 1,
        **kwargs: Any,
    ) -> str:
        """Format a user prompt into Qwen2-VL / Qwen2.5-VL ChatML format."""
        if "<|vision_start|>" in prompt and "<|im_start|>assistant" in prompt:
            return prompt + append_output

        sys_text = system_prompt
        if sys_text is None and include_system_prompt:
            sys_text = "You are a helpful assistant."

        n_imgs = max(0, int(num_images))
        messages: List[Dict[str, Any]] = []
        if sys_text:
            messages.append({"role": "system", "content": sys_text})
        user_content: List[Dict[str, Any]] = [{"type": "image"} for _ in range(n_imgs)]
        user_content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": user_content})

        if processor is not None and hasattr(processor, "apply_chat_template"):
            rendered = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return f"{rendered}{append_output}"

        parts: List[str] = []
        if sys_text:
            parts.append(f"<|im_start|>system\n{sys_text}<|im_end|>\n")
        vision_tokens = "<|vision_start|><|image_pad|><|vision_end|>" * n_imgs
        parts.append(
            f"<|im_start|>user\n{vision_tokens}{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{append_output}"
        )
        return "".join(parts)

    def prepare_inputs(
        self,
        processor: Any,
        prompt: str,
        image: Any,
        device: Any,
        dtype: Any,
        include_system_prompt: bool = False,
        append_output: str = "",
        image_token_index: Optional[int] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Optional[Tuple[int, int]]]:
        """Tokenize and preprocess text + image(s) for Qwen2-VL / Qwen2.5-VL."""
        if image is None:
            images_list: List[Any] = []
        elif isinstance(image, (list, tuple)):
            images_list = [img.convert("RGB") if hasattr(img, "convert") else img for img in image]
        else:
            images_list = [image.convert("RGB") if hasattr(image, "convert") else image]

        formatted = self.format_prompt(
            prompt,
            processor=processor,
            include_system_prompt=include_system_prompt,
            append_output=append_output,
            system_prompt=system_prompt,
            num_images=len(images_list),
        )
        proc_call_kwargs: Dict[str, Any] = {
            "text": [formatted],
            "padding": True,
            "return_tensors": "pt",
        }
        if images_list:
            proc_call_kwargs["images"] = images_list
        inputs = processor(**proc_call_kwargs)
        out = self._move_inputs_to_device(inputs, device=device, dtype=dtype)

        vis_token_id = image_token_index
        if vis_token_id is None:
            tok = getattr(processor, "tokenizer", None)
            if tok is not None and hasattr(tok, "convert_tokens_to_ids"):
                cand = tok.convert_tokens_to_ids("<|image_pad|>")
                if isinstance(cand, int) and cand >= 0:
                    vis_token_id = cand
        if vis_token_id is None:
            vis_token_id = 151655

        from .fastv_hf import detect_image_token_span

        span = detect_image_token_span(
            out.get("input_ids"),
            image_token_index=int(vis_token_id),
            default_image_length=0,
        )
        return out, span


HANDLERS: List[Type[BaseVLMHandler]] = [
    Qwen2VLHandler,
    LlavaHandler,
]


def register_handler(handler_cls: Type[BaseVLMHandler], prepend: bool = True) -> None:
    """Register a custom ``BaseVLMHandler`` subclass."""
    if prepend:
        HANDLERS.insert(0, handler_cls)
    else:
        HANDLERS.append(handler_cls)


def resolve_handler(
    model_id: Optional[str] = None,
    handler: Optional[BaseVLMHandler] = None,
) -> BaseVLMHandler:
    """Resolve an explicit handler instance or auto-detect the best handler for ``model_id``."""
    if handler is not None:
        return handler
    for cls in HANDLERS:
        if cls.matches(model_id):
            return cls()
    return LlavaHandler()


def load_vlm_model(
    model_id: Optional[str] = None,
    handler: Optional[BaseVLMHandler] = None,
    **kwargs: Any,
) -> Tuple[Any, Any, FastVController]:
    """Load any supported VLM using its auto-resolved ``BaseVLMHandler``."""
    resolved_handler = resolve_handler(model_id=model_id, handler=handler)
    return resolved_handler.load_model(model_id=model_id, **kwargs)


def format_vlm_prompt(
    prompt: str,
    processor: Optional[Any] = None,
    model_id: Optional[str] = None,
    handler: Optional[BaseVLMHandler] = None,
    **kwargs: Any,
) -> str:
    """Format a user prompt using the auto-resolved ``BaseVLMHandler``."""
    resolved_handler = resolve_handler(model_id=model_id, handler=handler)
    return resolved_handler.format_prompt(prompt=prompt, processor=processor, **kwargs)


def prepare_vlm_inputs(
    processor: Any,
    prompt: str,
    image: Any,
    device: Any,
    dtype: Any,
    model_id: Optional[str] = None,
    handler: Optional[BaseVLMHandler] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], Optional[Tuple[int, int]]]:
    """Prepare VLM inputs using the auto-resolved ``BaseVLMHandler``."""
    resolved_handler = resolve_handler(model_id=model_id, handler=handler)
    return resolved_handler.prepare_inputs(
        processor=processor,
        prompt=prompt,
        image=image,
        device=device,
        dtype=dtype,
        **kwargs,
    )


__all__ = [
    "BaseVLMHandler",
    "DEFAULT_LLAVA_MODEL_13B",
    "DEFAULT_LLAVA_MODEL_7B",
    "DEFAULT_QWEN2_VL_MAX_PIXELS",
    "DEFAULT_QWEN2_VL_MIN_PIXELS",
    "DEFAULT_QWEN2_VL_MODEL",
    "HANDLERS",
    "LlavaHandler",
    "Qwen2VLHandler",
    "VICUNA_SYSTEM_PROMPT",
    "format_vlm_prompt",
    "load_vlm_model",
    "prepare_vlm_inputs",
    "register_handler",
    "resolve_handler",
]
