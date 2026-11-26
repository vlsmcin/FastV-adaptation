"""Model-specific VLM adapters decoupling FastV pruning from architecture internals.

Each adapter tells ``FastVController`` how to:
  1. Locate the transformer decoder backbone exposing ``.layers``.
  2. Identify visual token IDs in ``input_ids`` (supporting multi-image / video).
  3. Slice ``position_ids`` and precomputed RoPE ``position_embeddings = (cos, sin)``
     (handling both standard 1D RoPE in LLaVA/Llama and 3D M-RoPE in Qwen2-VL / Qwen2.5-VL).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Type

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class BaseVLMAdapter(ABC):
    """Abstract interface for binding FastV hooks to a Vision-Language Model."""

    name: str = "base"

    @classmethod
    @abstractmethod
    def matches(cls, model: Any) -> bool:
        """Return True if this adapter supports ``model``."""

    @abstractmethod
    def get_decoder_backbone(self, model: Any) -> Any:
        """Return the transformer decoder module exposing ``.layers``."""

    @abstractmethod
    def get_visual_token_ids(self, model: Any) -> Tuple[int, ...]:
        """Return token IDs representing visual tokens in ``input_ids``."""

    def get_default_image_length(self, model: Any) -> int:
        """Fallback visual token count when ``input_ids`` contains a single unexpanded placeholder."""
        return 576

    def get_layer_attention_module(self, decoder_layer: Any) -> Any:
        """Return the self-attention submodule on ``decoder_layer``."""
        return getattr(decoder_layer, "self_attn", None)

    def slice_position_ids(self, position_ids: Any, keep_indices: Any, orig_len: int) -> Any:
        """Slice ``position_ids`` along its sequence axis (``dim=-1``).

        Works for both 2D ``(B, S)`` (Llama / LLaVA) and 3D M-RoPE ``(3, B, S)`` (Qwen2-VL).
        """
        if position_ids is None or torch is None or not torch.is_tensor(position_ids):
            return position_ids
        if position_ids.shape[-1] == orig_len:
            return position_ids.index_select(-1, keep_indices.to(position_ids.device))
        return position_ids

    def slice_position_embeddings(
        self,
        position_embeddings: Any,
        keep_indices: Any,
        orig_len: int,
    ) -> Any:
        """Slice precomputed RoPE ``(cos, sin)`` along the sequence axis (``dim=-2``).

        Works for both standard 3D ``(B, S, head_dim)`` and Qwen2-VL 4D M-RoPE ``(3, B, S, head_dim)``.
        """
        if (
            torch is None
            or not isinstance(position_embeddings, tuple)
            or len(position_embeddings) != 2
        ):
            return position_embeddings
        cos, sin = position_embeddings
        if torch.is_tensor(cos) and cos.ndim >= 2 and cos.shape[-2] == orig_len:
            k_dev = keep_indices.to(cos.device)
            return (
                cos.index_select(-2, k_dev),
                sin.index_select(-2, k_dev),
            )
        return position_embeddings


class LlavaLlamaAdapter(BaseVLMAdapter):
    """Adapter for HuggingFace LLaVA-1.5 / LLaVA-NeXT and standalone Llama models."""

    name: str = "llava_llama"

    @classmethod
    def matches(cls, model: Any) -> bool:
        cls_name = type(model).__name__.lower()
        model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
        return any(tok in cls_name or tok in model_type for tok in ("llava", "llama"))

    def get_decoder_backbone(self, model: Any) -> Any:
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

    def get_visual_token_ids(self, model: Any) -> Tuple[int, ...]:
        cfg = getattr(model, "config", None)
        token_id = getattr(cfg, "image_token_index", None)
        if token_id is None:
            token_id = getattr(cfg, "image_token_id", 32000)
        return (int(token_id),)

    def get_default_image_length(self, model: Any) -> int:
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


class Qwen2VLAdapter(BaseVLMAdapter):
    """Adapter interface for HuggingFace Qwen2-VL and Qwen2.5-VL models (with 3D M-RoPE).

    Handles:
      * Variable-length visual token spans marked by ``config.image_token_id`` (151655)
        and optionally ``config.video_token_id`` (151656).
      * ``Qwen2VLTextModel`` / ``Qwen2_5_VLTextModel`` decoder lookup across
        ``transformers`` 4.45–4.49 (``model.model``) and 4.50–5.x (``model.model.language_model``).
      * 3D M-RoPE ``position_ids`` of shape ``(3, B, S)`` and ``position_embeddings``
        of shape ``((3, B, S, head_dim), (3, B, S, head_dim))``.
    """

    name: str = "qwen2_vl"

    def __init__(self, prune_video_tokens: bool = True):
        self.prune_video_tokens = prune_video_tokens

    @classmethod
    def matches(cls, model: Any) -> bool:
        cls_name = type(model).__name__.lower()
        model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
        return (
            "qwen2vl" in cls_name
            or "qwen2_5_vl" in cls_name
            or model_type in ("qwen2_vl", "qwen2_5_vl")
        )

    def get_decoder_backbone(self, model: Any) -> Any:
        candidates = [
            getattr(getattr(model, "model", None), "language_model", None),
            getattr(model, "language_model", None),
            getattr(model, "model", None),
            model,
        ]
        for cand in candidates:
            if cand is not None and hasattr(cand, "layers") and len(cand.layers) > 0:
                return cand
        raise AttributeError(
            f"Could not find Qwen2-VL decoder backbone with `.layers` on {type(model).__name__}"
        )

    def get_visual_token_ids(self, model: Any) -> Tuple[int, ...]:
        cfg = getattr(model, "config", None)
        ids: List[int] = []
        img_id = getattr(cfg, "image_token_id", 151655)
        if img_id is not None:
            ids.append(int(img_id))
        if self.prune_video_tokens:
            vid_id = getattr(cfg, "video_token_id", None)
            if vid_id is not None:
                ids.append(int(vid_id))
        return tuple(ids)

    def get_default_image_length(self, model: Any) -> int:
        # Qwen2-VL always expands <|image_pad|> tokens inside input_ids according to grid_thw.
        return 0


ADAPTERS: List[Type[BaseVLMAdapter]] = [
    Qwen2VLAdapter,
    LlavaLlamaAdapter,
]


def register_adapter(adapter_cls: Type[BaseVLMAdapter], prepend: bool = True) -> None:
    """Register a custom ``BaseVLMAdapter`` subclass."""
    if prepend:
        ADAPTERS.insert(0, adapter_cls)
    else:
        ADAPTERS.append(adapter_cls)


def resolve_adapter(model: Any, adapter: Optional[BaseVLMAdapter] = None) -> BaseVLMAdapter:
    """Resolve an explicit adapter instance or auto-detect the best adapter for ``model``."""
    if adapter is not None:
        return adapter
    for cls in ADAPTERS:
        if cls.matches(model):
            return cls()
    return LlavaLlamaAdapter()


__all__ = [
    "ADAPTERS",
    "BaseVLMAdapter",
    "LlavaLlamaAdapter",
    "Qwen2VLAdapter",
    "register_adapter",
    "resolve_adapter",
]
