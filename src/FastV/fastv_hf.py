"""Modern, version-agnostic FastV implementation for HuggingFace VLMs (LLaVA & Qwen2-VL ready).

Implements "An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference
Acceleration for Large Vision-Language Models" (Chen et al., ECCV 2024 Oral,
arXiv:2403.06764) using pure PyTorch module hooks so it works directly on
upstream HuggingFace ``transformers`` (>=4.45 through 5.x) without requiring
a vendored ``transformers`` fork.

Key modular components:
  * ``scoring.py``: Pluggable visual token scoring registry (``fastv_mean``, etc.).
  * ``adapters.py``: VLM-specific adapters (``LlavaLlamaAdapter``, ``Qwen2VLAdapter``).
  * ``FastVController``: Architecture-agnostic hook engine supporting both:
      - ``mode="inplace"``: Physically drops pruned visual tokens after layer ``K``,
        reducing sequence length, FLOPs, latency, and KV-cache memory.
      - ``mode="mask"``: Masks out pruned visual tokens in the attention mask from
        layer ``K`` onward without changing sequence length (Paper Fig 7 ablation).
"""

from __future__ import annotations

import inspect
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
except ImportError:  # pragma: no cover - allows pure-Python utilities without torch
    torch = None  # type: ignore[assignment]

from .adapters import BaseVLMAdapter, LlavaLlamaAdapter, resolve_adapter
from .handlers import (
    BaseVLMHandler,
    LlavaHandler,
    Qwen2VLHandler,
    format_vlm_prompt,
    load_vlm_model,
    prepare_vlm_inputs,
    resolve_handler,
)
from .scoring import ScorerFn, resolve_scorer

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
    scorer: Union[str, ScorerFn] = "fastv_mean"
    scorer_kwargs: Dict[str, Any] = field(default_factory=dict)
    image_token_start_index: Optional[int] = None
    image_token_length: Optional[int] = None
    attention_rank: Optional[int] = None  # Explicit number of visual tokens to keep
    random_seed: int = 42
    force_slice_at_r0: bool = False  # Exercise full slicing path even when keep_n == N_vis (for parity tests)

    def __post_init__(self) -> None:
        if self.mode not in ("inplace", "mask"):
            raise ValueError(f"FastV mode must be 'inplace' or 'mask', got {self.mode!r}")
        if self.fastv_k < 0:
            raise ValueError(f"fastv_k must be >= 0, got {self.fastv_k}")
        if not (0.0 <= self.fastv_r <= 1.0):
            raise ValueError(f"fastv_r must be in [0.0, 1.0], got {self.fastv_r}")
        if self.attention_rank is not None and self.attention_rank < 0:
            raise ValueError(f"attention_rank must be >= 0, got {self.attention_rank}")
        # Validate scorer eagerly so invalid names fail fast.
        resolve_scorer(self.scorer)

    @classmethod
    def from_any(cls, cfg: Optional[Union["FastVConfig", Dict[str, Any]]]) -> "FastVConfig":
        if cfg is None:
            return cls(use_fastv=False)
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, dict):
            normalized = dict(cfg)
            # Support legacy naming aliases from original repo scripts & plan.md.
            aliases = {
                "use_fast_v": "use_fastv",
                "enabled": "use_fastv",
                "fast_v_agg_layer": "fastv_k",
                "agg_layer": "fastv_k",
                "prune_ratio": "fastv_r",
                "fast_v_sys_length": "image_token_start_index",
                "fast_v_image_token_length": "image_token_length",
                "fast_v_attention_rank": "attention_rank",
            }
            for old_key, new_key in aliases.items():
                if old_key in normalized and new_key not in normalized:
                    normalized[new_key] = normalized.pop(old_key)
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
        d = asdict(self)
        if callable(self.scorer):
            d["scorer"] = getattr(self.scorer, "__name__", repr(self.scorer))
        return d


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
    """Compute theoretical FLOPs and FLOPs reduction ratio (Paper Section 3.3, Eq. 1-2)."""
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


def detect_visual_token_indices(
    input_ids: Any,
    visual_token_ids: Union[int, Sequence[int]] = 32000,
    default_image_length: int = 576,
) -> Optional[List[int]]:
    """Detect all visual token positions in ``input_ids`` (supports non-contiguous multi-image).

    Supports:
      * Modern ``transformers`` (>=4.45) LLaVA and Qwen2-VL where visual tokens are
        already expanded inside ``input_ids`` (including interleaved multi-image prompts).
      * Older ``transformers`` (<4.45) where a single placeholder token is expanded
        to ``default_image_length`` tokens inside ``forward()``.
    """
    if input_ids is None:
        return None
    ids = input_ids.detach() if hasattr(input_ids, "detach") else input_ids
    if hasattr(ids, "ndim") and ids.ndim == 2:
        row = ids[0]
    elif hasattr(ids, "ndim") and ids.ndim == 1:
        row = ids
    elif isinstance(ids, (list, tuple)):
        row = ids[0] if (len(ids) > 0 and isinstance(ids[0], (list, tuple))) else ids
    else:
        return None

    tokens = row.tolist() if hasattr(row, "tolist") else list(row)
    target_set = (
        {int(visual_token_ids)}
        if isinstance(visual_token_ids, int)
        else {int(t) for t in visual_token_ids}
    )
    positions = [idx for idx, tok in enumerate(tokens) if int(tok) in target_set]
    if not positions:
        return None
    if len(positions) == 1 and default_image_length > 1: # Older versions from transformers save just one placeholder token
        start = positions[0]
        return list(range(start, start + int(default_image_length))) # Depends on the model's configuration 
    return positions


def detect_image_token_span(
    input_ids: Any,
    image_token_index: int = 32000,
    default_image_length: int = 576,
) -> Optional[Tuple[int, int]]:
    """Detect ``(start_index, length)`` for single-image prompts (backwards-compatible helper)."""
    indices = detect_visual_token_indices(
        input_ids=input_ids,
        visual_token_ids=image_token_index,
        default_image_length=default_image_length,
    )
    if not indices:
        return None
    return (indices[0], len(indices)) # Backwards-compatible wrapper dor detect_visual_token_indices


def resolve_hf_model_id(
    model_path: Optional[str] = None,
    requested_size: Optional[str] = None,
    gpu_memory_gib: Optional[float] = None,
) -> str:
    """Resolve a local path, HF model id, or legacy LLaVA name to a usable model id."""
    return LlavaHandler().resolve_model_id(
        model_id=model_path,
        requested_size=requested_size,
        gpu_memory_gib=gpu_memory_gib,
    )


def get_llama_backbone(model: Any) -> Any:
    """Locate the underlying decoder backbone (holding ``.layers``) across transformers versions."""
    return resolve_adapter(model).get_decoder_backbone(model)


def _supports_dynamic_eager_dispatch(self_attn: Any) -> bool:
    """Return True if ``self_attn`` uses unified config-driven dispatch (transformers >= 4.49 / 5.x)."""
    if self_attn is None:
        return False
    cls_name = type(self_attn).__name__.lower()
    if "sdpa" in cls_name or "flash" in cls_name:
        return False
    cfg = getattr(self_attn, "config", None)
    return cfg is not None and hasattr(cfg, "_attn_implementation")


def _set_eager_attention(model: Any, backbone: Any) -> None:
    """Ensure eager attention is active on the text decoder ``backbone`` without touching ``model.visual``."""
    current = getattr(getattr(backbone, "config", None), "_attn_implementation", None)
    if current == "eager":
        return
    if hasattr(backbone, "set_attn_implementation"):
        try:
            backbone.set_attn_implementation("eager")
            return
        except Exception:
            pass
    backbone_cfg = getattr(backbone, "config", None)
    if backbone_cfg is not None:
        backbone_cfg._attn_implementation = "eager"
    model_cfg = getattr(model, "config", None)
    if model_cfg is not None:
        text_cfg = getattr(model_cfg, "text_config", None)
        if text_cfg is not None:
            text_cfg._attn_implementation = "eager"
        elif not hasattr(model, "visual") and not hasattr(getattr(model, "model", None), "visual"):
            model_cfg._attn_implementation = "eager"
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
    diag = kv_len - seq_len + 1
    mask = torch.triu(mask, diagonal=diag)
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, kv_len).clone()


class FastVController:
    """Architecture-agnostic PyTorch hook engine executing FastV pruning on HF VLMs."""

    def __init__(
        self,
        model: Any,
        config: Optional[Union[FastVConfig, Dict[str, Any]]] = None,
        adapter: Optional[BaseVLMAdapter] = None,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required to use FastVController.")
        self.model = model
        self.adapter = resolve_adapter(model, adapter=adapter)
        self.backbone = self.adapter.get_decoder_backbone(model)
        self.config = FastVConfig.from_any(config)
        self.visual_token_ids = self.adapter.get_visual_token_ids(model)
        self.default_image_length = self.adapter.get_default_image_length(model)

        self._handles: List[Any] = []
        self._step_prev_attn_impl: Optional[Tuple[Any, Any]] = None
        self._reset_generation_state()
        self._install_hooks()
        self._ensure_attention_capture_ready()

    def _restore_temp_attn_impl(self) -> None:
        if self._step_prev_attn_impl is not None:
            attn_cfg, prev_impl = self._step_prev_attn_impl
            self._step_prev_attn_impl = None
            try:
                attn_cfg._attn_implementation = prev_impl
            except Exception:
                pass

    def _ensure_attention_capture_ready(self) -> None:
        """Ensure layer K-1 can materialize attention weights without disturbing ``model.visual``."""
        if not (self.config.use_fastv and self.config.fastv_k > 0):
            return
        target_idx = self.config.fastv_k - 1
        if 0 <= target_idx < len(self.backbone.layers):
            target_attn = self.adapter.get_layer_attention_module(self.backbone.layers[target_idx])
            if _supports_dynamic_eager_dispatch(target_attn):
                # Unified attention class (transformers >= 4.49 / 5.x): _attn_pre_hook / _attn_post_hook
                # toggle eager dynamically ONLY on layer K-1 during prefill, keeping all other layers & vision on SDPA.
                return
        _set_eager_attention(self.model, self.backbone)

    def _reset_generation_state(self) -> None:
        self._restore_temp_attn_impl()
        self._detected_vis_indices: Optional[List[int]] = None
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
        self._ensure_attention_capture_ready()
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

    def _resolve_vis_indices(self, seq_len: int, device: Any) -> Optional[Any]:
        """Resolve 1D sorted LongTensor of visual token positions within ``[0, seq_len)``."""
        start = self.config.image_token_start_index
        length = self.config.image_token_length
        if start is not None:
            eff_len = int(length) if length is not None else self.default_image_length
            if start < 0 or eff_len <= 0 or start + eff_len > seq_len:
                return None
            return torch.arange(start, start + eff_len, dtype=torch.long, device=device)

        if self._detected_vis_indices is not None:
            valid = [i for i in self._detected_vis_indices if 0 <= i < seq_len]
            if not valid:
                return None
            if length is not None and length < len(valid):
                valid = valid[: int(length)]
            return torch.tensor(valid, dtype=torch.long, device=device)
        return None

    def _select_keep_and_drop_indices(
        self,
        seq_len: int,
        device: Any,
        attn_weights: Optional[Any],
    ) -> Optional[Tuple[Any, Any]]:
        vis_indices = self._resolve_vis_indices(seq_len=seq_len, device=device)
        if vis_indices is None or vis_indices.numel() == 0:
            return None

        n_vis = int(vis_indices.numel())
        keep_n = self.config.resolve_keep_count(n_vis)
        if keep_n >= n_vis and not self.config.force_slice_at_r0:
            return None

        if keep_n >= n_vis:
            kept_vis_indices = vis_indices
        elif self.config.fastv_k == 0 or attn_weights is None:
            rng = random.Random(self.config.random_seed)
            rel = sorted(rng.sample(range(n_vis), keep_n)) if keep_n > 0 else []
            kept_vis_indices = vis_indices[torch.tensor(rel, dtype=torch.long, device=device)] if rel else torch.empty(0, dtype=torch.long, device=device)
        else:
            scorer_fn = resolve_scorer(self.config.scorer)
            scores = scorer_fn(attn_weights, vis_indices, **self.config.scorer_kwargs)
            if keep_n > 0:
                top_rel = scores.topk(keep_n).indices.to(vis_indices.device)
                kept_vis_indices = vis_indices.index_select(0, top_rel)
            else:
                kept_vis_indices = torch.empty(0, dtype=torch.long, device=device)

        vis_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        vis_mask[vis_indices] = True
        non_vis_indices = torch.nonzero(~vis_mask, as_tuple=False).squeeze(-1)

        keep_indices = torch.cat([non_vis_indices, kept_vis_indices.to(device=device, dtype=torch.long)])
        keep_indices = keep_indices.sort().values

        keep_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        keep_mask[keep_indices] = True
        drop_indices = torch.nonzero(~keep_mask, as_tuple=False).squeeze(-1)

        self.last_stats = {
            "adapter": self.adapter.name,
            "scorer": (
                self.config.scorer
                if isinstance(self.config.scorer, str)
                else getattr(self.config.scorer, "__name__", "custom")
            ),
            "attn_captured": bool(attn_weights is not None),
            "mode": self.config.mode,
            "fastv_k": self.config.fastv_k,
            "fastv_r": self.config.effective_pruning_ratio(n_vis),
            "image_token_start_index": int(vis_indices[0].item()),
            "image_token_length": n_vis,
            "kept_image_tokens": int(kept_vis_indices.numel()),
            "dropped_image_tokens": int(n_vis - kept_vis_indices.numel()),
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
        # 1. Top-level pre-hook to detect all visual token indices from input_ids on prefill.
        def _top_pre_hook(_module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) > 0 and torch.is_tensor(args[0]) and args[0].dtype in (
                torch.int32,
                torch.int64,
            ):
                input_ids = args[0]
            if input_ids is not None and hasattr(input_ids, "shape") and input_ids.shape[-1] > 1:
                indices = detect_visual_token_indices(
                    input_ids,
                    visual_token_ids=self.visual_token_ids,
                    default_image_length=self.default_image_length,
                )
                if indices is not None:
                    self._detected_vis_indices = indices

        self._handles.append(self.model.register_forward_pre_hook(_top_pre_hook, with_kwargs=True))

        # 2. Backbone pre-hook to reset per-forward transient state.
        def _backbone_pre_hook(_module: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            self._restore_temp_attn_impl()
            self._step_captured_attn = None
            self._step_keep_indices = None
            self._step_drop_indices = None
            self._step_orig_len = None

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
            self_attn = self.adapter.get_layer_attention_module(decoder_layer)
            if self_attn is not None:
                attn_sig = inspect.signature(self_attn.forward)
                has_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in attn_sig.parameters.values()
                )
                can_pass_output_attentions = ("output_attentions" in attn_sig.parameters) or has_var_kw
                can_dynamic_eager = _supports_dynamic_eager_dispatch(self_attn)

                def _make_attn_pre_hook(idx: int, can_oa: bool, dynamic_eager: bool):
                    def _attn_pre_hook(mod: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
                        if (
                            self.config.use_fastv
                            and self.config.fastv_k > 0
                            and idx == self.config.fastv_k - 1
                            and self._step_orig_len is not None
                            and self._step_orig_len > 1
                        ):
                            if dynamic_eager:
                                attn_cfg = getattr(mod, "config", None)
                                prev_impl = getattr(attn_cfg, "_attn_implementation", None)
                                if attn_cfg is not None and prev_impl != "eager":
                                    self._step_prev_attn_impl = (attn_cfg, prev_impl)
                                    attn_cfg._attn_implementation = "eager"
                            if can_oa:
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
                            try:
                                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                                    self._step_captured_attn = output[1].detach()
                            finally:
                                self._restore_temp_attn_impl()

                    return _attn_post_hook

                self._handles.append(
                    self_attn.register_forward_pre_hook(
                        _make_attn_pre_hook(layer_idx, can_pass_output_attentions, can_dynamic_eager),
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
                            if idx == k and cur_seq_len == orig_len:
                                pruned_hs = hidden_states.index_select(1, keep_indices.to(device))
                                if hs_in_args:
                                    new_args[0] = pruned_hs
                                else:
                                    new_kwargs["hidden_states"] = pruned_hs

                            # Delegate RoPE & position_ids slicing to adapter (supports 2D Llama & 3D Qwen2-VL M-RoPE).
                            if "position_ids" in new_kwargs and new_kwargs["position_ids"] is not None:
                                new_kwargs["position_ids"] = self.adapter.slice_position_ids(
                                    new_kwargs["position_ids"], keep_indices, orig_len
                                )
                            if "position_embeddings" in new_kwargs and new_kwargs["position_embeddings"] is not None:
                                new_kwargs["position_embeddings"] = self.adapter.slice_position_embeddings(
                                    new_kwargs["position_embeddings"], keep_indices, orig_len
                                )

                            if "attention_mask" in new_kwargs and new_kwargs["attention_mask"] is not None:
                                new_kwargs["attention_mask"] = self._slice_attention_mask_prefill(
                                    new_kwargs["attention_mask"], keep_indices, orig_len
                                )

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
    adapter: Optional[BaseVLMAdapter] = None,
) -> FastVController:
    """Attach (or update) a ``FastVController`` on a HuggingFace VLM/LLM."""
    existing = getattr(model, "_fastv_controller", None)
    if isinstance(existing, FastVController) and adapter is None:
        existing.update_config(fastv_config)
        return existing
    if isinstance(existing, FastVController):
        existing.remove()
    controller = FastVController(model, fastv_config, adapter=adapter)
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


def remove_fastv(model: Any) -> None:
    """Completely detach all FastV hooks from ``model``."""
    existing = getattr(model, "_fastv_controller", None)
    if isinstance(existing, FastVController):
        existing.remove()
        delattr(model, "_fastv_controller")


def load_llava_model(
    model_id: Optional[str] = None,
    quantization: str = "4bit",
    dtype: Optional[Any] = None,
    attn_implementation: str = "eager",
    fastv_config: Optional[Union[FastVConfig, Dict[str, Any]]] = None,
    device_map: Optional[Any] = "auto",
    revision: Optional[str] = None,
) -> Tuple[Any, Any, FastVController]:
    """Load HF ``LlavaForConditionalGeneration`` + ``AutoProcessor`` (delegates to ``LlavaHandler``)."""
    return LlavaHandler().load_model(
        model_id=model_id,
        quantization=quantization,
        dtype=dtype,
        attn_implementation=attn_implementation,
        fastv_config=fastv_config,
        device_map=device_map,
        revision=revision,
    )


def format_llava_prompt(
    prompt: str,
    include_system_prompt: bool = False,
    append_output: str = "",
) -> str:
    """Format a user prompt into LLaVA-1.5 conversation format (delegates to ``LlavaHandler``)."""
    return LlavaHandler().format_prompt(
        prompt,
        include_system_prompt=include_system_prompt,
        append_output=append_output,
    )


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
    """Tokenize and preprocess text + image for modern HF LLaVA (delegates to ``LlavaHandler``)."""
    return LlavaHandler().prepare_inputs(
        processor=processor,
        prompt=prompt,
        image=image,
        device=device,
        dtype=dtype,
        include_system_prompt=include_system_prompt,
        append_output=append_output,
        image_token_index=image_token_index,
    )


def compute_layer_attention_efficiency(
    attentions: Sequence[Any],
    image_start: int,
    image_length: int,
    instruction_end: Optional[int] = None,
) -> Dict[str, List[float]]:
    """Compute per-layer attention allocation by token type (Paper Section 2, Figure 3)."""
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
        avg = layer_attn.float().mean(dim=1)[0]
        seq_len = int(avg.shape[-1])
        inst_end = min(instruction_end if instruction_end is not None else seq_len, seq_len)

        if inst_end < seq_len:
            query_attn = avg[inst_end:, :].mean(dim=0)
        else:
            query_attn = avg[-1, :]

        sys_slice = query_attn[:image_start]
        img_slice = query_attn[image_start : min(img_end, seq_len)]
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
    """Compute ROUGE-L matching ``pycocoevalcap.rouge.rouge.Rouge`` (beta=1.2) in pure Python."""
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
