"""Modern, version-agnostic FastV implementation for HuggingFace LLaVA / Llama models.

Implements "An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference
Acceleration for Large Vision-Language Models" (Chen et al., ECCV 2024 Oral,
arXiv:2403.06764) using pure PyTorch module hooks so it works directly on
upstream HuggingFace ``transformers`` (>=4.45 through 5.x) without requiring
a vendored ``transformers`` fork.

Supported FastV modes:
  * ``inplace``: Physically drops pruned visual tokens after layer ``K``,
    reducing sequence length, FLOPs, latency, and KV-cache memory in deeper
    layers (compatible with both ``use_cache=True`` and ``use_cache=False``).
  * ``mask``: Masks out pruned visual tokens in the attention mask from layer
    ``K`` onward without changing sequence length (reproduces the Figure 7 /
    OCR-VQA ablation study, including the ``K=0`` random baseline).
"""

from __future__ import annotations

import inspect
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - allows pure-Python utilities without torch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


VICUNA_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)

DEFAULT_HF_MODEL_7B = "llava-hf/llava-1.5-7b-hf"
DEFAULT_HF_MODEL_13B = "llava-hf/llava-1.5-13b-hf"


@dataclass
class FastVConfig:
    """Configuration for FastV visual token pruning."""

    use_fastv: bool = True
    fastv_k: int = 2
    fastv_r: float = 0.5
    mode: str = "inplace"  # "inplace" or "mask"
    image_token_start_index: Optional[int] = None
    image_token_length: Optional[int] = None
    attention_rank: Optional[int] = None  # Explicit number of visual tokens to keep
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.mode not in ("inplace", "mask"):
            raise ValueError(f"FastV mode must be 'inplace' or 'mask', got {self.mode!r}")
        if self.fastv_k < 0:
            raise ValueError(f"fastv_k must be >= 0, got {self.fastv_k}")
        if not (0.0 <= self.fastv_r <= 1.0):
            raise ValueError(f"fastv_r must be in [0.0, 1.0], got {self.fastv_r}")
        if self.attention_rank is not None and self.attention_rank < 0:
            raise ValueError(f"attention_rank must be >= 0, got {self.attention_rank}")

    @classmethod
    def from_any(cls, cfg: Optional[Union["FastVConfig", Dict[str, Any]]]) -> "FastVConfig":
        if cfg is None:
            return cls(use_fastv=False)
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, dict):
            normalized = dict(cfg)
            # Support legacy naming aliases from original repo scripts.
            if "use_fast_v" in normalized and "use_fastv" not in normalized:
                normalized["use_fastv"] = normalized.pop("use_fast_v")
            if "fast_v_agg_layer" in normalized and "fastv_k" not in normalized:
                normalized["fastv_k"] = normalized.pop("fast_v_agg_layer")
            if "fast_v_sys_length" in normalized and "image_token_start_index" not in normalized:
                normalized["image_token_start_index"] = normalized.pop("fast_v_sys_length")
            if "fast_v_image_token_length" in normalized and "image_token_length" not in normalized:
                normalized["image_token_length"] = normalized.pop("fast_v_image_token_length")
            if "fast_v_attention_rank" in normalized and "attention_rank" not in normalized:
                normalized["attention_rank"] = normalized.pop("fast_v_attention_rank")
            if "fast_v_inplace" in normalized and "mode" not in normalized:
                inplace = normalized.pop("fast_v_inplace")
                normalized["mode"] = "inplace" if inplace else "mask"
            valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in normalized.items() if k in valid_fields}
            return cls(**filtered)
        raise TypeError(f"Unsupported FastV config type: {type(cfg)!r}")

    def resolve_keep_count(self, image_token_length: int) -> int:
        """Return number of visual tokens to keep out of ``image_token_length``."""
        if self.attention_rank is not None:
            return max(0, min(int(self.attention_rank), image_token_length))
        keep = int(round(image_token_length * (1.0 - float(self.fastv_r))))
        return max(0, min(keep, image_token_length))

    def effective_pruning_ratio(self, image_token_length: int = 576) -> float:
        if not self.use_fastv or image_token_length <= 0:
            return 0.0
        keep = self.resolve_keep_count(image_token_length)
        return 1.0 - (keep / float(image_token_length))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_fastv_flops_ratio(
    num_layers: int,
    fastv_k: int,
    fastv_r: float,
    n_image_tokens: int = 576,
    n_text_tokens: int = 60,
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    swiglu: bool = False,
) -> Dict[str, float]:
    """Compute theoretical FLOPs and FLOPs reduction ratio (Paper Section 3.3, Eq. 1-2).

    Paper Equation (1):
        FLOPs_layer(n) = 4 * n * d^2 + 2 * n^2 * d + 2 * n * d * m
    (Set ``swiglu=True`` to use ``3 * n * d * m`` for Llama's gated SwiGLU MLP.)

    Paper Equation (2):
        FLOPs_FastV = K * FLOPs_layer(n) + (T - K) * FLOPs_layer(n_hat)
        where n_hat = (1 - R) * n_image_tokens + n_text_tokens.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    k_eff = max(0, min(int(fastv_k), num_layers))
    r_eff = max(0.0, min(float(fastv_r), 1.0))

    n_full = float(n_image_tokens + n_text_tokens)
    n_kept = float(round(n_image_tokens * (1.0 - r_eff)) + n_text_tokens)
    ffn_mult = 3.0 if swiglu else 2.0

    def _layer_flops(n: float) -> float:
        return (
            4.0 * n * (hidden_size ** 2)
            + 2.0 * (n ** 2) * hidden_size
            + ffn_mult * n * hidden_size * intermediate_size
        )

    baseline_flops = num_layers * _layer_flops(n_full)
    fastv_flops = k_eff * _layer_flops(n_full) + (num_layers - k_eff) * _layer_flops(n_kept)
    ratio = fastv_flops / baseline_flops if baseline_flops > 0 else 1.0
    reduction = 1.0 - ratio

    return {
        "baseline_tflops": baseline_flops / 1e12,
        "fastv_tflops": fastv_flops / 1e12,
        "flops_ratio": ratio,
        "flops_ratio_pct": ratio * 100.0,
        "flops_reduction": reduction,
        "flops_reduction_pct": reduction * 100.0,
        "tokens_before": n_full,
        "tokens_after": n_kept,
    }


def detect_image_token_span(
    input_ids: Any,
    image_token_index: int = 32000,
    default_image_length: int = 576,
) -> Optional[Tuple[int, int]]:
    """Detect ``(start_index, length)`` of the visual token span in ``input_ids``.

    Supports both:
      * Modern ``transformers`` (>=4.45) where ``LlavaProcessor`` expands
        ``<image>`` into ``default_image_length`` contiguous ``image_token_index``
        entries directly inside ``input_ids``.
      * Older ``transformers`` (<4.45) where a single ``image_token_index``
        placeholder is expanded inside ``LlavaForConditionalGeneration.forward``.
    """
    if input_ids is None:
        return None
    ids = input_ids
    if hasattr(ids, "detach"):
        ids = ids.detach()
    if hasattr(ids, "ndim") and ids.ndim == 2:
        row = ids[0]
    elif hasattr(ids, "ndim") and ids.ndim == 1:
        row = ids
    elif isinstance(ids, (list, tuple)):
        row = ids[0] if (len(ids) > 0 and isinstance(ids[0], (list, tuple))) else ids
    else:
        return None

    if hasattr(row, "tolist"):
        tokens = row.tolist()
    else:
        tokens = list(row)

    positions = [idx for idx, tok in enumerate(tokens) if int(tok) == int(image_token_index)]
    if not positions:
        return None
    start = positions[0]
    if len(positions) == 1:
        return (start, default_image_length)
    return (start, len(positions))


def resolve_hf_model_id(
    model_path: Optional[str] = None,
    requested_size: Optional[str] = None,
    gpu_memory_gib: Optional[float] = None,
) -> str:
    """Resolve a local path, HF model id, or legacy LLaVA name to a usable model id."""
    if model_path:
        if os.path.exists(model_path):
            return model_path
        lower = model_path.lower().rstrip("/")
        if "/" in model_path and not model_path.startswith(".") and not model_path.startswith("/"):
            # Looks like an explicit HuggingFace repo ID (e.g., llava-hf/llava-1.5-7b-hf)
            return model_path
        if "13b" in lower:
            return DEFAULT_HF_MODEL_13B
        if "7b" in lower:
            return DEFAULT_HF_MODEL_7B
        return model_path

    size = (requested_size or os.environ.get("FASTV_MODEL_SIZE", "")).strip().lower()
    if size == "7b":
        return DEFAULT_HF_MODEL_7B
    if size == "13b":
        return DEFAULT_HF_MODEL_13B

    if gpu_memory_gib is None and torch is not None and torch.cuda.is_available():
        gpu_memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if gpu_memory_gib is not None and gpu_memory_gib >= 24.0:
        return DEFAULT_HF_MODEL_13B
    return DEFAULT_HF_MODEL_7B


def get_llama_backbone(model: Any) -> Any:
    """Locate the underlying ``LlamaModel`` (holding ``.layers``) across transformers versions.

    Works for:
      * ``transformers >= 4.50 / 5.x``: ``model.model.language_model``
      * ``transformers 4.39 - 4.49``: ``model.language_model.model``
      * Standalone ``LlamaForCausalLM``: ``model.model``
      * Bare ``LlamaModel``: ``model``
    """
    candidates = [
        getattr(getattr(model, "model", None), "language_model", None),
        getattr(getattr(model, "language_model", None), "model", None),
        getattr(model, "language_model", None),
        getattr(model, "model", None),
        model,
    ]
    for cand in candidates:
        if cand is not None and hasattr(cand, "layers") and len(cand.layers) > 0:
            return cand
    raise AttributeError(
        f"Could not find a transformer decoder backbone with `.layers` on {type(model).__name__}"
    )


def _infer_default_image_token_length(model: Any) -> int:
    cfg = getattr(model, "config", None)
    vision_cfg = getattr(cfg, "vision_config", None)
    if vision_cfg is not None:
        img_size = getattr(vision_cfg, "image_size", None)
        patch_size = getattr(vision_cfg, "patch_size", None)
        if img_size and patch_size:
            patches = (int(img_size) // int(patch_size)) ** 2
            strategy = getattr(cfg, "vision_feature_select_strategy", "default")
            return patches + 1 if strategy == "full" else patches
    return 576


def _set_eager_attention(model: Any, backbone: Any) -> None:
    """Ensure eager attention is active so layer K-1 materializes attention weights."""
    current = getattr(backbone.config, "_attn_implementation", None)
    if current == "eager":
        return
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
            return
        except Exception:
            pass
    for obj in (model, backbone):
        cfg = getattr(obj, "config", None)
        if cfg is not None:
            cfg._attn_implementation = "eager"
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                text_cfg._attn_implementation = "eager"
    for layer in backbone.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "config"):
            attn.config._attn_implementation = "eager"


def _make_causal_additive_mask(
    seq_len: int,
    kv_len: int,
    dtype: Any,
    device: Any,
    batch_size: int = 1,
) -> Any:
    """Create a 4D causal additive mask ``(B, 1, seq_len, kv_len)`` with 0 / finfo.min."""
    min_val = torch.finfo(dtype).min
    if seq_len == 1:
        return torch.zeros((batch_size, 1, 1, kv_len), dtype=dtype, device=device)
    mask = torch.full((seq_len, kv_len), min_val, dtype=dtype, device=device)
    # Causal upper-triangular masking aligned to the right for kv_len >= seq_len.
    diag = kv_len - seq_len + 1
    mask = torch.triu(mask, diagonal=diag)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, kv_len).clone()


class FastVController:
    """Manages PyTorch hooks that execute FastV pruning on a HuggingFace LLaVA/Llama model."""

    def __init__(self, model: Any, config: Optional[Union[FastVConfig, Dict[str, Any]]] = None):
        if torch is None:
            raise RuntimeError("PyTorch is required to use FastVController.")
        self.model = model
        self.backbone = get_llama_backbone(model)
        self.config = FastVConfig.from_any(config)
        self.default_image_length = _infer_default_image_token_length(model)
        self.image_token_index = int(getattr(getattr(model, "config", None), "image_token_index", 32000))

        self._handles: List[Any] = []
        self._reset_generation_state()
        self._install_hooks()
        if self.config.use_fastv and self.config.fastv_k > 0:
            _set_eager_attention(self.model, self.backbone)

    def _reset_generation_state(self) -> None:
        self._detected_image_span: Optional[Tuple[int, int]] = None
        self._last_prefill_keep_indices: Optional[Any] = None
        self._last_prefill_drop_indices: Optional[Any] = None
        self._last_prefill_orig_len: Optional[int] = None
        self._step_captured_attn: Optional[Any] = None
        self._step_keep_indices: Optional[Any] = None
        self._step_drop_indices: Optional[Any] = None
        self._step_orig_len: Optional[int] = None
        self.last_stats: Dict[str, Any] = {}

    def update_config(self, config: Optional[Union[FastVConfig, Dict[str, Any]]]) -> "FastVController":
        self.config = FastVConfig.from_any(config)
        self._reset_generation_state()
        if self.config.use_fastv and self.config.fastv_k > 0:
            _set_eager_attention(self.model, self.backbone)
        return self

    def disable(self) -> None:
        self.config = FastVConfig(use_fastv=False)
        self._reset_generation_state()

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._reset_generation_state()

    def _resolve_image_span(self, seq_len: int) -> Optional[Tuple[int, int]]:
        start = self.config.image_token_start_index
        length = self.config.image_token_length
        if start is None or length is None:
            if self._detected_image_span is not None:
                det_start, det_len = self._detected_image_span
                if start is None:
                    start = det_start
                if length is None:
                    length = det_len
        if start is None:
            return None
        if length is None:
            length = self.default_image_length
        if start < 0 or length <= 0 or start + length > seq_len:
            return None
        return (int(start), int(length))

    def _select_keep_and_drop_indices(
        self,
        seq_len: int,
        device: Any,
        attn_weights: Optional[Any],
    ) -> Optional[Tuple[Any, Any]]:
        span = self._resolve_image_span(seq_len)
        if span is None:
            return None
        img_start, img_len = span
        keep_n = self.config.resolve_keep_count(img_len)
        if keep_n >= img_len:
            return None

        if self.config.fastv_k == 0 or attn_weights is None:
            # Random visual token selection (reproduces K=0 baseline in Paper Figure 7)
            rng = random.Random(self.config.random_seed)
            rel_indices = sorted(rng.sample(range(img_len), keep_n)) if keep_n > 0 else []
            top_img_indices = torch.tensor(
                [img_start + i for i in rel_indices], dtype=torch.long, device=device
            )
        else:
            # Average attention over heads for sample 0, take last token's attention over image tokens.
            # attn_weights shape: (batch, num_heads, q_len, kv_len)
            last_layer_attn_avg = attn_weights.mean(dim=1)[0]
            last_tok_attn = last_layer_attn_avg[-1]
            img_attn = last_tok_attn[img_start : img_start + img_len]
            if keep_n > 0:
                top_img_indices = img_attn.topk(keep_n).indices + img_start
            else:
                top_img_indices = torch.empty(0, dtype=torch.long, device=device)

        prefix = torch.arange(0, img_start, dtype=torch.long, device=device)
        suffix = torch.arange(img_start + img_len, seq_len, dtype=torch.long, device=device)
        keep_indices = torch.cat([prefix, top_img_indices.to(device=device, dtype=torch.long), suffix])
        keep_indices = keep_indices.sort().values

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[keep_indices] = True
        drop_indices = torch.nonzero(~keep_mask, as_tuple=False).squeeze(-1)

        self.last_stats = {
            "mode": self.config.mode,
            "fastv_k": self.config.fastv_k,
            "fastv_r": self.config.effective_pruning_ratio(img_len),
            "image_token_start_index": img_start,
            "image_token_length": img_len,
            "kept_image_tokens": int(keep_n),
            "dropped_image_tokens": int(img_len - keep_n),
            "orig_seq_len": int(seq_len),
            "pruned_seq_len": int(keep_indices.shape[0]),
        }
        return keep_indices, drop_indices

    def _slice_attention_mask_prefill(self, attn_mask: Any, keep_indices: Any, orig_len: int) -> Any:
        if attn_mask is None:
            return None
        if isinstance(attn_mask, dict):
            return {
                k: self._slice_attention_mask_prefill(v, keep_indices, orig_len)
                for k, v in attn_mask.items()
            }
        if not torch.is_tensor(attn_mask):
            return attn_mask
        if attn_mask.ndim == 4:
            out = attn_mask
            if out.shape[-2] == orig_len:
                out = out.index_select(-2, keep_indices.to(out.device))
            if out.shape[-1] == orig_len:
                out = out.index_select(-1, keep_indices.to(out.device))
            return out
        if attn_mask.ndim == 2 and attn_mask.shape[-1] == orig_len:
            return attn_mask.index_select(-1, keep_indices.to(attn_mask.device))
        return attn_mask

    def _slice_attention_mask_decode(self, attn_mask: Any) -> Any:
        if attn_mask is None or self._last_prefill_keep_indices is None or self._last_prefill_orig_len is None:
            return attn_mask
        if isinstance(attn_mask, dict):
            return {k: self._slice_attention_mask_decode(v) for k, v in attn_mask.items()}
        if not torch.is_tensor(attn_mask):
            return attn_mask
        orig_len = self._last_prefill_orig_len
        keep_idx = self._last_prefill_keep_indices.to(attn_mask.device)
        kv_len = attn_mask.shape[-1]
        if kv_len >= orig_len:
            extra = torch.arange(orig_len, kv_len, dtype=torch.long, device=attn_mask.device)
            col_indices = torch.cat([keep_idx, extra])
            return attn_mask.index_select(-1, col_indices)
        return attn_mask

    def _apply_token_mask(
        self,
        attn_mask: Any,
        drop_indices: Any,
        q_len: int,
        kv_len: int,
        dtype: Any,
        device: Any,
        batch_size: int,
    ) -> Any:
        if drop_indices is None or drop_indices.numel() == 0:
            return attn_mask
        if isinstance(attn_mask, dict):
            return {
                k: self._apply_token_mask(v, drop_indices, q_len, kv_len, dtype, device, batch_size)
                for k, v in attn_mask.items()
            }
        if attn_mask is None:
            mask = _make_causal_additive_mask(q_len, kv_len, dtype=dtype, device=device, batch_size=batch_size)
        else:
            mask = attn_mask.clone()
        min_val = torch.finfo(mask.dtype).min
        valid_drops = drop_indices[drop_indices < mask.shape[-1]].to(mask.device)
        if valid_drops.numel() > 0:
            if mask.ndim == 4:
                mask[:, :, :, valid_drops] = min_val
            elif mask.ndim == 2:
                mask[:, valid_drops] = 0
        return mask

    def _install_hooks(self) -> None:
        # 1. Top-level pre-hook to detect image token span from input_ids on prefill.
        def _top_pre_hook(_module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]) and args[0].dtype in (
                torch.int32,
                torch.int64,
            ):
                input_ids = args[0]
            if input_ids is not None and hasattr(input_ids, "shape") and input_ids.shape[-1] > 1:
                span = detect_image_token_span(
                    input_ids,
                    image_token_index=self.image_token_index,
                    default_image_length=self.default_image_length,
                )
                if span is not None:
                    self._detected_image_span = span

        self._handles.append(self.model.register_forward_pre_hook(_top_pre_hook, with_kwargs=True))

        # 2. Backbone pre-hook to reset per-forward transient state.
        def _backbone_pre_hook(_module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            self._step_captured_attn = None
            self._step_keep_indices = None
            self._step_drop_indices = None
            self._step_orig_len = None

            # Determine if this is a fresh prefill (seq_len > 1) vs single-token decode (seq_len == 1).
            inputs_embeds = kwargs.get("inputs_embeds")
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]):
                input_ids = args[0]
            seq_len = None
            if inputs_embeds is not None:
                seq_len = int(inputs_embeds.shape[1])
            elif input_ids is not None:
                seq_len = int(input_ids.shape[1])

            if seq_len is not None and seq_len > 1:
                self._step_orig_len = seq_len
                self._last_prefill_keep_indices = None
                self._last_prefill_drop_indices = None
                self._last_prefill_orig_len = seq_len

        self._handles.append(self.backbone.register_forward_pre_hook(_backbone_pre_hook, with_kwargs=True))

        # 3. Per-layer attention capture & pruning hooks.
        for layer_idx, decoder_layer in enumerate(self.backbone.layers):
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is not None:
                attn_sig_params = set(inspect.signature(self_attn.forward).parameters.keys())
                has_output_attentions_arg = "output_attentions" in attn_sig_params

                def _make_attn_pre_hook(idx: int, has_oa: bool):
                    def _attn_pre_hook(_mod: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
                        if (
                            self.config.use_fastv
                            and self.config.fastv_k > 0
                            and idx == self.config.fastv_k - 1
                            and self._step_orig_len is not None
                            and self._step_orig_len > 1
                            and has_oa
                        ):
                            new_kwargs = dict(kwargs)
                            new_kwargs["output_attentions"] = True
                            return args, new_kwargs
                        return None

                    return _attn_pre_hook

                def _make_attn_post_hook(idx: int):
                    def _attn_post_hook(_mod: Any, _inputs: Any, output: Any) -> None:
                        if (
                            self.config.use_fastv
                            and self.config.fastv_k > 0
                            and idx == self.config.fastv_k - 1
                            and self._step_orig_len is not None
                            and self._step_orig_len > 1
                        ):
                            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                                self._step_captured_attn = output[1].detach()

                    return _attn_post_hook

                self._handles.append(
                    self_attn.register_forward_pre_hook(
                        _make_attn_pre_hook(layer_idx, has_output_attentions_arg),
                        with_kwargs=True,
                    )
                )
                self._handles.append(self_attn.register_forward_hook(_make_attn_post_hook(layer_idx)))

            def _make_layer_pre_hook(idx: int):
                def _layer_pre_hook(_mod: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
                    if not self.config.use_fastv:
                        return None
                    k = self.config.fastv_k
                    if idx < k:
                        return None

                    # Extract hidden_states (positional arg 0 or kwarg)
                    hs_in_args = len(args) > 0 and torch.is_tensor(args[0])
                    hidden_states = args[0] if hs_in_args else kwargs.get("hidden_states")
                    if hidden_states is None:
                        return None

                    batch_size, cur_seq_len = int(hidden_states.shape[0]), int(hidden_states.shape[1])
                    device = hidden_states.device
                    dtype = hidden_states.dtype

                    # --- Prefill path (seq_len > 1 on original input) ---
                    if self._step_orig_len is not None and self._step_orig_len > 1:
                        orig_len = self._step_orig_len
                        if idx == k:
                            selection = self._select_keep_and_drop_indices(
                                seq_len=orig_len,
                                device=device,
                                attn_weights=self._step_captured_attn,
                            )
                            # Release captured attention tensor immediately to save VRAM.
                            self._step_captured_attn = None
                            if selection is None:
                                return None
                            keep_indices, drop_indices = selection
                            self._step_keep_indices = keep_indices
                            self._step_drop_indices = drop_indices
                            self._last_prefill_keep_indices = keep_indices
                            self._last_prefill_drop_indices = drop_indices
                            self._last_prefill_orig_len = orig_len

                        keep_indices = self._step_keep_indices
                        drop_indices = self._step_drop_indices
                        if keep_indices is None:
                            return None

                        new_args = list(args)
                        new_kwargs = dict(kwargs)

                        if self.config.mode == "inplace":
                            new_seq_len = int(keep_indices.shape[0])
                            # Only slice hidden_states at layer K (subsequent layers already receive pruned hidden_states).
                            if idx == k and cur_seq_len == orig_len:
                                pruned_hs = hidden_states.index_select(1, keep_indices.to(device))
                                if hs_in_args:
                                    new_args[0] = pruned_hs
                                else:
                                    new_kwargs["hidden_states"] = pruned_hs

                            # Slice position_ids if still at original prefill length.
                            pos_ids = new_kwargs.get("position_ids")
                            if pos_ids is not None and torch.is_tensor(pos_ids) and pos_ids.shape[-1] == orig_len:
                                new_kwargs["position_ids"] = pos_ids.index_select(-1, keep_indices.to(pos_ids.device))

                            # Slice precomputed RoPE position_embeddings=(cos, sin) (transformers >= 4.45 / 5.x).
                            pos_emb = new_kwargs.get("position_embeddings")
                            if (
                                isinstance(pos_emb, tuple)
                                and len(pos_emb) == 2
                                and torch.is_tensor(pos_emb[0])
                                and pos_emb[0].shape[1] == orig_len
                            ):
                                cos, sin = pos_emb
                                k_dev = keep_indices.to(cos.device)
                                new_kwargs["position_embeddings"] = (
                                    cos.index_select(1, k_dev),
                                    sin.index_select(1, k_dev),
                                )

                            # Slice causal attention mask.
                            if "attention_mask" in new_kwargs and new_kwargs["attention_mask"] is not None:
                                new_kwargs["attention_mask"] = self._slice_attention_mask_prefill(
                                    new_kwargs["attention_mask"], keep_indices, orig_len
                                )

                            # Slice cache_position.
                            cache_pos = new_kwargs.get("cache_position")
                            if cache_pos is not None and torch.is_tensor(cache_pos) and cache_pos.shape[0] == orig_len:
                                new_kwargs["cache_position"] = cache_pos[:new_seq_len]

                            return tuple(new_args), new_kwargs

                        # Mask mode (no physical token dropping; mask out dropped visual columns).
                        new_kwargs["attention_mask"] = self._apply_token_mask(
                            new_kwargs.get("attention_mask"),
                            drop_indices=drop_indices,
                            q_len=cur_seq_len,
                            kv_len=orig_len,
                            dtype=dtype,
                            device=device,
                            batch_size=batch_size,
                        )
                        return tuple(new_args), new_kwargs

                    # --- Decode path (cur_seq_len == 1 with KV cache) ---
                    if cur_seq_len == 1 and self._last_prefill_keep_indices is not None:
                        new_kwargs = dict(kwargs)
                        if self.config.mode == "inplace":
                            if "attention_mask" in new_kwargs and new_kwargs["attention_mask"] is not None:
                                new_kwargs["attention_mask"] = self._slice_attention_mask_decode(
                                    new_kwargs["attention_mask"]
                                )
                            return args, new_kwargs
                        else:
                            # Determine KV cache length for mask mode during decode.
                            attn_mask = new_kwargs.get("attention_mask")
                            kv_len = None
                            if torch.is_tensor(attn_mask):
                                kv_len = int(attn_mask.shape[-1])
                            else:
                                pkv = (
                                    new_kwargs.get("past_key_values")
                                    or new_kwargs.get("past_key_value")
                                    or new_kwargs.get("cache")
                                )
                                if pkv is not None and hasattr(pkv, "get_seq_length"):
                                    kv_len = int(pkv.get_seq_length(idx)) + 1
                            if kv_len is not None:
                                new_kwargs["attention_mask"] = self._apply_token_mask(
                                    attn_mask,
                                    drop_indices=self._last_prefill_drop_indices,
                                    q_len=1,
                                    kv_len=kv_len,
                                    dtype=dtype,
                                    device=device,
                                    batch_size=batch_size,
                                )
                                return args, new_kwargs
                    return None

                return _layer_pre_hook

            self._handles.append(
                decoder_layer.register_forward_pre_hook(_make_layer_pre_hook(layer_idx), with_kwargs=True)
            )


def attach_fastv(
    model: Any,
    fastv_config: Optional[Union[FastVConfig, Dict[str, Any]]] = None,
) -> FastVController:
    """Attach (or update) a ``FastVController`` on a HuggingFace LLaVA/Llama model."""
    existing = getattr(model, "_fastv_controller", None)
    if isinstance(existing, FastVController):
        existing.update_config(fastv_config)
        return existing
    controller = FastVController(model, fastv_config)
    model._fastv_controller = controller
    return controller


def set_fastv_config(
    model: Any,
    fastv_config: Optional[Union[FastVConfig, Dict[str, Any]]],
) -> FastVController:
    """Convenience alias for attaching or updating FastV configuration on ``model``."""
    return attach_fastv(model, fastv_config)


def disable_fastv(model: Any) -> None:
    """Disable FastV pruning on ``model`` (switches back to vanilla decoding)."""
    existing = getattr(model, "_fastv_controller", None)
    if isinstance(existing, FastVController):
        existing.disable()


def load_llava_model(
    model_id: Optional[str] = None,
    quantization: str = "4bit",
    dtype: Optional[Any] = None,
    attn_implementation: str = "eager",
    fastv_config: Optional[Union[FastVConfig, Dict[str, Any]]] = None,
    device_map: Optional[Any] = "auto",
    revision: Optional[str] = None,
) -> Tuple[Any, Any, FastVController]:
    """Load HF ``LlavaForConditionalGeneration`` + ``AutoProcessor`` cleanly across transformers 4.45-5.x."""
    if torch is None:
        raise RuntimeError("PyTorch is required to load LLaVA models.")
    from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

    resolved_id = resolve_hf_model_id(model_id)
    if dtype is None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    quant = (quantization or "none").strip().lower()
    quant_config = None
    if quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant not in ("none", "fp16", "bf16", "fp32"):
        raise ValueError(f"Unsupported quantization={quantization!r}; choose '4bit', '8bit', or 'none'")

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

    # Handle transformers 5.x (`dtype=`) vs 4.x (`torch_dtype=`) cleanly without warnings/errors.
    import transformers

    tf_ver = tuple(int(p) for p in re.findall(r"\d+", transformers.__version__)[:2])
    if tf_ver >= (4, 56):
        load_kwargs["dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = dtype

    try:
        model = LlavaForConditionalGeneration.from_pretrained(resolved_id, **load_kwargs)
    except TypeError:
        # Fallback if a custom version expects the opposite dtype kwarg name.
        if "dtype" in load_kwargs:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        else:
            load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
        model = LlavaForConditionalGeneration.from_pretrained(resolved_id, **load_kwargs)

    model.eval()

    proc_kwargs: Dict[str, Any] = {}
    if revision:
        proc_kwargs["revision"] = revision
    processor = AutoProcessor.from_pretrained(resolved_id, **proc_kwargs)

    controller = attach_fastv(model, fastv_config)
    return model, processor, controller


def format_llava_prompt(
    prompt: str,
    include_system_prompt: bool = False,
    append_output: str = "",
) -> str:
    """Format a user prompt into LLaVA-1.5 conversation format."""
    if "<image>" in prompt and "ASSISTANT:" in prompt:
        return prompt + append_output
    prefix = f"{VICUNA_SYSTEM_PROMPT} " if include_system_prompt else ""
    user_turn = prompt if "<image>" in prompt else f"<image>\n{prompt}"
    return f"{prefix}USER: {user_turn}\nASSISTANT:{append_output}"


def prepare_llava_inputs(
    processor: Any,
    prompt: str,
    image: Any,
    device: Any,
    dtype: Any,
    include_system_prompt: bool = False,
    append_output: str = "",
    image_token_index: int = 32000,
) -> Tuple[Dict[str, Any], Optional[Tuple[int, int]]]:
    """Tokenize and preprocess text + image for modern HF LLaVA, returning ``(inputs, image_span)``."""
    formatted = format_llava_prompt(
        prompt,
        include_system_prompt=include_system_prompt,
        append_output=append_output,
    )
    rgb_image = image.convert("RGB") if hasattr(image, "convert") else image
    inputs = processor(images=rgb_image, text=formatted, return_tensors="pt")
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        if torch is not None and torch.is_tensor(v):
            if v.is_floating_point():
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    span = detect_image_token_span(out.get("input_ids"), image_token_index=image_token_index)
    return out, span


def compute_layer_attention_efficiency(
    attentions: Sequence[Any],
    image_start: int,
    image_length: int,
    instruction_end: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Compute per-layer attention allocation by token type (Paper Section 2, Figure 3).

    For each layer, averages the attention weights across all heads, then computes
    the mean attention received per token (and total attention share) for:
      * system prompt tokens: ``[0, image_start)``
      * visual / image tokens: ``[image_start, image_start + image_length)``
      * instruction / text tokens: ``[image_start + image_length, instruction_end)``
      * output tokens (if ``instruction_end < seq_len``): ``[instruction_end, seq_len)``
    evaluated from the perspective of the final prompt / generated tokens.
    """
    metrics: Dict[str, List[float]] = {
        "layer": [],
        "system_total": [],
        "image_total": [],
        "text_total": [],
        "output_total": [],
        "system_per_token": [],
        "image_per_token": [],
        "text_per_token": [],
        "output_per_token": [],
        "visual_attention_efficiency": [],
    }

    img_end = image_start + image_length
    for idx, layer_attn in enumerate(attentions):
        if layer_attn is None:
            continue
        # Shape: (batch, heads, seq_len, seq_len) -> (seq_len, seq_len)
        avg = layer_attn.float().mean(dim=1)[0]
        seq_len = int(avg.shape[-1])
        inst_end = min(instruction_end if instruction_end is not None else seq_len, seq_len)

        # Evaluate attention from output tokens (if present) else from the last prompt token.
        if inst_end < seq_len:
            query_attn = avg[inst_end:, :].mean(dim=0)
        else:
            query_attn = avg[-1, :]

        sys_slice = query_attn[:image_start]
        img_slice = query_attn[image_start:min(img_end, seq_len)]
        txt_slice = query_attn[min(img_end, seq_len) : inst_end]
        out_slice = query_attn[inst_end:]

        sys_tot = float(sys_slice.sum().item()) if sys_slice.numel() > 0 else 0.0
        img_tot = float(img_slice.sum().item()) if img_slice.numel() > 0 else 0.0
        txt_tot = float(txt_slice.sum().item()) if txt_slice.numel() > 0 else 0.0
        out_tot = float(out_slice.sum().item()) if out_slice.numel() > 0 else 0.0

        sys_pt = sys_tot / max(1, sys_slice.numel())
        img_pt = img_tot / max(1, img_slice.numel())
        txt_pt = txt_tot / max(1, txt_slice.numel())
        out_pt = out_tot / max(1, out_slice.numel()) if out_slice.numel() > 0 else 0.0

        # Attention efficiency = (image attention share) / (image token fraction)
        img_token_frac = img_slice.numel() / max(1, seq_len)
        vis_eff = img_tot / img_token_frac if img_token_frac > 0 else 0.0

        metrics["layer"].append(idx + 1)
        metrics["system_total"].append(sys_tot)
        metrics["image_total"].append(img_tot)
        metrics["text_total"].append(txt_tot)
        metrics["output_total"].append(out_tot)
        metrics["system_per_token"].append(sys_pt)
        metrics["image_per_token"].append(img_pt)
        metrics["text_per_token"].append(txt_pt)
        metrics["output_per_token"].append(out_pt)
        metrics["visual_attention_efficiency"].append(vis_eff)

    return metrics


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for tok_a in a:
        curr = [0]
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(curr[-1] if curr[-1] >= prev[j] else prev[j])
        prev = curr
    return prev[-1]


def compute_rouge_l(
    references: Sequence[Sequence[str]],
    predictions: Sequence[str],
    beta: float = 1.2,
) -> Tuple[float, List[float]]:
    """Compute ROUGE-L matching ``pycocoevalcap.rouge.rouge.Rouge`` (beta=1.2) in pure Python.

    Avoids fragile native/Java dependencies on Python 3.12/3.13 while producing
    identical scores to ``pycocoevalcap``'s ROUGE-L implementation.
    """
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    scores: List[float] = []
    beta_sq = beta * beta
    for refs, pred in zip(references, predictions):
        pred_tokens = pred.split()
        best_prec = 0.0
        best_rec = 0.0
        for ref in refs:
            ref_tokens = ref.split()
            lcs = _lcs_length(ref_tokens, pred_tokens)
            prec = lcs / float(len(pred_tokens)) if pred_tokens else 0.0
            rec = lcs / float(len(ref_tokens)) if ref_tokens else 0.0
            if rec > best_rec or (rec == best_rec and prec > best_prec):
                best_rec = rec
                best_prec = prec
        if best_prec > 0.0 and best_rec > 0.0:
            score = ((1.0 + beta_sq) * best_prec * best_rec) / (best_rec + beta_sq * best_prec)
        else:
            score = 0.0
        scores.append(score)
    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, scores
