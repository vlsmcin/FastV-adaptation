"""Pluggable visual token importance scoring and score-combination functions for FastV.

All scorers share the uniform signature:

    def scorer(
        attn_weights: Optional[torch.Tensor],   # (batch, num_heads, q_len, kv_len)
        vis_indices: torch.Tensor,              # 1D LongTensor of visual token indices
        *,
        hidden_states: Optional[torch.Tensor] = None,  # (batch, seq_len, hidden_dim) at layer K
        **kwargs,
    ) -> torch.Tensor:                          # 1D Tensor of importance scores (len == len(vis_indices))

Adding a new research metric (e.g., head-wise entropy weighting, question-guided
multi-query aggregation, hidden-state norms, or normalized linear combinations)
only requires registering a function here without modifying any model adapter.
"""

from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import torch
except ImportError:  # pragma: no cover - allows pure-Python imports without torch
    torch = None  # type: ignore[assignment]

ScorerFn = Callable[..., Any]
VALID_NORMALIZATIONS = ("zscore", "minmax", "rank", "none")


def _require_attn(attn_weights: Any, scorer_name: str) -> None:
    if attn_weights is None:
        raise ValueError(f"{scorer_name} requires attn_weights (set fastv_k >= 1 with eager attention).")


def _require_hidden_states(hidden_states: Any, scorer_name: str) -> None:
    if hidden_states is None:
        raise ValueError(f"{scorer_name} requires hidden_states at the pruning layer.")


# ==============================================================================
# Individual Visual Token Scorers
# ==============================================================================


def fastv_mean_scorer(
    attn_weights: Any,
    vis_indices: Any,
    query_index: int = -1,
    **_kwargs: Any,
) -> Any:
    """Original FastV paper scorer (Chen et al., ECCV 2024): mean attention across heads from last token."""
    _require_attn(attn_weights, "fastv_mean")
    avg_heads = attn_weights.float().mean(dim=1)[0]  # (q_len, kv_len)
    query_attn = avg_heads[query_index]  # (kv_len,)
    return query_attn.index_select(0, vis_indices.to(query_attn.device))


def fastv_max_head_scorer(
    attn_weights: Any,
    vis_indices: Any,
    query_index: int = -1,
    **_kwargs: Any,
) -> Any:
    """Max attention across heads from the target query token (captures head specialization)."""
    _require_attn(attn_weights, "fastv_max_head")
    max_heads = attn_weights.float().max(dim=1).values[0]  # (q_len, kv_len)
    query_attn = max_heads[query_index]
    return query_attn.index_select(0, vis_indices.to(query_attn.device))


def fastv_mean_std_scorer(
    attn_weights: Any,
    vis_indices: Any,
    alpha: float = 0.5,
    query_index: int = -1,
    **_kwargs: Any,
) -> Any:
    """Mean + alpha * std across attention heads (rewards both high mean and strong head focus)."""
    _require_attn(attn_weights, "fastv_mean_std")
    per_head_q = attn_weights.float()[0, :, query_index, :]  # (num_heads, kv_len)
    mean = per_head_q.mean(dim=0)
    std = per_head_q.std(dim=0, unbiased=False)
    combined = mean + float(alpha) * std
    return combined.index_select(0, vis_indices.to(combined.device))


def text_query_mean_scorer(
    attn_weights: Any,
    vis_indices: Any,
    head_agg: str = "mean",
    **_kwargs: Any,
) -> Any:
    """Question-guided cross-attention: mean attention from all instruction/question text tokens.

    Unlike ``fastv_mean`` (which only inspects the final formatting token ``query_index=-1``,
    often biased toward attention sinks), this averages visual attention over every non-visual
    text token following the first visual token (i.e. the user's question/instruction).
    """
    _require_attn(attn_weights, "text_query_mean")
    if head_agg == "mean":
        head_reduced = attn_weights.float().mean(dim=1)[0]  # (q_len, kv_len)
    elif head_agg == "max":
        head_reduced = attn_weights.float().max(dim=1).values[0]
    else:
        raise ValueError(f"head_agg must be 'mean' or 'max', got {head_agg!r}")

    q_len = int(head_reduced.shape[0])
    dev = head_reduced.device
    vis_dev = vis_indices.to(dev)
    if vis_dev.numel() == 0:
        return head_reduced.new_empty((0,))

    vis_mask = torch.zeros(q_len, dtype=torch.bool, device=dev)
    valid_vis = vis_dev[(vis_dev >= 0) & (vis_dev < q_len)]
    if valid_vis.numel() > 0:
        vis_mask[valid_vis] = True
        first_vis = int(valid_vis.min().item())
    else:
        first_vis = 0

    all_positions = torch.arange(q_len, device=dev)
    post_vis_text = all_positions[(~vis_mask) & (all_positions > first_vis)]
    if post_vis_text.numel() > 0:
        query_rows = head_reduced.index_select(0, post_vis_text)
    else:
        non_vis = all_positions[~vis_mask]
        query_rows = head_reduced.index_select(0, non_vis) if non_vis.numel() > 0 else head_reduced[-1:, :]

    agg_attn = query_rows.mean(dim=0)  # (kv_len,)
    return agg_attn.index_select(0, vis_dev)


def entropy_weighted_scorer(
    attn_weights: Any,
    vis_indices: Any,
    query_index: int = -1,
    gamma: float = 1.0,
    eps: float = 1e-12,
    **_kwargs: Any,
) -> Any:
    """Inverse-entropy head-weighted attention from the target query token.

    Computes the normalized Shannon entropy of each attention head's distribution
    over visual tokens and weights sharper (low-entropy, spatially focused) heads
    higher than diffuse/uniform heads:
        w_h proportional to (1 - H_norm(h)) ** gamma
        score(v) = sum_h w_h * A[h, query_index, v]
    """
    _require_attn(attn_weights, "entropy_weighted")
    per_head_q = attn_weights.float()[0, :, query_index, :]  # (num_heads, kv_len)
    vis_dev = vis_indices.to(per_head_q.device)
    n_vis = int(vis_dev.numel())
    if n_vis == 0:
        return per_head_q.new_empty((0,))

    vis_attn = per_head_q.index_select(1, vis_dev).clamp_min(0.0)  # (num_heads, n_vis)
    if n_vis == 1:
        return vis_attn.mean(dim=0)

    head_mass = vis_attn.sum(dim=1, keepdim=True)  # (num_heads, 1)
    probs = vis_attn / (head_mass + eps)
    entropy = -(probs * (probs + eps).log()).sum(dim=1)  # (num_heads,)
    norm_entropy = (entropy / math.log(float(n_vis))).clamp(0.0, 1.0)

    sharpness = (1.0 - norm_entropy).clamp_min(0.0).pow(float(gamma)) + eps
    head_weights = sharpness / sharpness.sum()  # (num_heads,)
    return torch.matmul(head_weights, vis_attn)  # (n_vis,)


def hidden_norm_scorer(
    attn_weights: Any = None,
    vis_indices: Any = None,
    hidden_states: Any = None,
    ord: float = 2.0,
    **_kwargs: Any,
) -> Any:
    """L_p norm (default L2) of visual token hidden-state representations entering layer K."""
    del attn_weights
    _require_hidden_states(hidden_states, "hidden_norm")
    hs = hidden_states.float()[0]  # (seq_len, hidden_dim)
    vis_hs = hs.index_select(0, vis_indices.to(hs.device))  # (n_vis, hidden_dim)
    return torch.linalg.vector_norm(vis_hs, ord=float(ord), dim=-1)


# ==============================================================================
# Score Normalization & Linear Combination Utilities
# ==============================================================================


def normalize_scores(
    scores: Any,
    method: str = "zscore",
    eps: float = 1e-8,
) -> Any:
    """Normalize a 1D visual-token importance vector onto a common scale before fusion.

    Supported methods:
      * ``"zscore"``: ``(s - mean) / (std + eps)`` (0-centered, unit variance; returns zeros if constant)
      * ``"minmax"``: ``(s - min) / (max - min + eps)`` in ``[0, 1]`` (returns zeros if constant)
      * ``"rank"``: percentile rank in ``[0, 1]`` (0 = lowest score, 1 = highest score)
      * ``"none"``: raw float scores unchanged
    """
    norm_key = str(method).strip().lower()
    if norm_key not in VALID_NORMALIZATIONS:
        raise ValueError(
            f"Unsupported score normalization {method!r}; choose from {VALID_NORMALIZATIONS}"
        )
    s = scores.float()
    n = int(s.numel())
    if n <= 1 or norm_key == "none":
        return s if norm_key == "none" else torch.zeros_like(s)

    if norm_key == "zscore":
        mean = s.mean()
        std = s.std(unbiased=False)
        if float(std.item()) <= eps:
            return torch.zeros_like(s)
        return (s - mean) / (std + eps)

    if norm_key == "minmax":
        s_min = s.min()
        s_max = s.max()
        span = s_max - s_min
        if float(span.item()) <= eps:
            return torch.zeros_like(s)
        return (s - s_min) / (span + eps)

    # norm_key == "rank"
    order = torch.argsort(s, stable=True)
    ranks = torch.empty_like(s)
    ranks[order] = torch.arange(n, dtype=s.dtype, device=s.device) / float(n - 1)
    return ranks


def score_visual_tokens(
    scorer: Union[str, ScorerFn],
    attn_weights: Any,
    vis_indices: Any,
    hidden_states: Any = None,
    **kwargs: Any,
) -> Any:
    """Invoke ``scorer`` safely, forwarding ``hidden_states`` when accepted by its signature."""
    scorer_fn = resolve_scorer(scorer)
    sig = inspect.signature(scorer_fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    call_kwargs = dict(kwargs)
    if hidden_states is not None and (accepts_var_kw or "hidden_states" in params):
        call_kwargs["hidden_states"] = hidden_states
    if not accepts_var_kw:
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in params}
    return scorer_fn(attn_weights, vis_indices, **call_kwargs)


def _normalize_component_spec(
    components: Union[
        Mapping[str, float],
        Sequence[Tuple[Union[str, ScorerFn], float]],
        Sequence[Union[str, ScorerFn]],
    ],
) -> List[Tuple[Union[str, ScorerFn], float]]:
    if isinstance(components, Mapping):
        items = [(k, float(v)) for k, v in components.items()]
    elif isinstance(components, Sequence) and not isinstance(components, (str, bytes)):
        items = []
        for elem in components:
            if isinstance(elem, tuple) and len(elem) == 2:
                items.append((elem[0], float(elem[1])))
            else:
                items.append((elem, 1.0))
    else:
        raise TypeError(
            "components must be a dict {scorer: weight} or a sequence of scorers / (scorer, weight) pairs"
        )
    if not items:
        raise ValueError("Linear combination requires at least one component scorer.")
    return items


def make_combined_scorer(
    components: Union[
        Mapping[str, float],
        Sequence[Tuple[Union[str, ScorerFn], float]],
        Sequence[Union[str, ScorerFn]],
    ],
    norm: str = "zscore",
    name: Optional[str] = None,
) -> ScorerFn:
    """Build a normalized linear combination scorer ``sum_i w_i * norm(s_i)``.

    Args:
        components: Mapping ``{scorer_name: weight}``, sequence of ``(scorer, weight)`` tuples,
            or sequence of scorers (weighted equally).
        norm: Default normalization applied to each component before weighting
            (``"zscore"``, ``"minmax"``, ``"rank"``, or ``"none"``). Can be overridden at call time.
        name: Optional function ``__name__`` for controller telemetry.
    """
    if norm.strip().lower() not in VALID_NORMALIZATIONS:
        raise ValueError(f"Unsupported norm={norm!r}; choose from {VALID_NORMALIZATIONS}")
    parsed = _normalize_component_spec(components)

    def _combined(
        attn_weights: Any,
        vis_indices: Any,
        hidden_states: Any = None,
        **kwargs: Any,
    ) -> Any:
        call_kwargs = dict(kwargs)
        active_norm = str(call_kwargs.pop("norm", norm))
        sub_kwargs_map = call_kwargs.pop("component_kwargs", {}) or {}

        dev = (
            attn_weights.device
            if attn_weights is not None and hasattr(attn_weights, "device")
            else (
                hidden_states.device
                if hidden_states is not None and hasattr(hidden_states, "device")
                else vis_indices.device
            )
        )
        total = torch.zeros(int(vis_indices.numel()), dtype=torch.float32, device=dev)
        for sub_scorer, weight in parsed:
            if weight == 0.0:
                continue
            sub_key = sub_scorer if isinstance(sub_scorer, str) else getattr(sub_scorer, "__name__", "")
            merged_kw = dict(call_kwargs)
            if sub_key in sub_kwargs_map:
                merged_kw.update(sub_kwargs_map[sub_key])
            raw = score_visual_tokens(
                sub_scorer,
                attn_weights,
                vis_indices,
                hidden_states=hidden_states,
                **merged_kw,
            ).to(dev)
            normed = normalize_scores(raw, method=active_norm)
            total = total + float(weight) * normed
        return total

    if name:
        _combined.__name__ = str(name)
    return _combined


def linear_combination_scorer(
    attn_weights: Any,
    vis_indices: Any,
    weights: Optional[
        Union[
            Mapping[str, float],
            Sequence[Tuple[Union[str, ScorerFn], float]],
            Sequence[Union[str, ScorerFn]],
        ]
    ] = None,
    norm: str = "zscore",
    hidden_states: Any = None,
    **kwargs: Any,
) -> Any:
    """Configurable linear combination scorer driven via ``FastVConfig.scorer_kwargs``.

    Example::
        FastVConfig(
            scorer="linear_combination",
            scorer_kwargs={
                "weights": {"fastv_mean": 0.5, "text_query_mean": 0.3, "hidden_norm": 0.2},
                "norm": "zscore",
            },
        )
    """
    spec = weights if weights is not None else {"fastv_mean": 0.5, "text_query_mean": 0.5}
    combiner = make_combined_scorer(spec, norm=norm, name="linear_combination")
    return combiner(attn_weights, vis_indices, hidden_states=hidden_states, **kwargs)


# Pre-built research combinations comparing attention-only vs cross-modal + representation fusion
combo_mean_text_scorer = make_combined_scorer(
    {"fastv_mean": 0.50, "text_query_mean": 0.50},
    norm="zscore",
    name="combo_mean_text",
)
combo_attn_ensemble_scorer = make_combined_scorer(
    {"entropy_weighted": 0.40, "text_query_mean": 0.40, "fastv_max_head": 0.20},
    norm="zscore",
    name="combo_attn_ensemble",
)
combo_mean_text_norm_scorer = make_combined_scorer(
    {"fastv_mean": 0.40, "text_query_mean": 0.35, "hidden_norm": 0.25},
    norm="zscore",
    name="combo_mean_text_norm",
)
combo_mean_text_norm_minmax_scorer = make_combined_scorer(
    {"fastv_mean": 0.40, "text_query_mean": 0.35, "hidden_norm": 0.25},
    norm="minmax",
    name="combo_mean_text_norm_minmax",
)


SCORERS: Dict[str, ScorerFn] = {
    # Original FastV paper baseline
    "fastv_mean": fastv_mean_scorer,
    # Alternative individual scorers
    "fastv_max_head": fastv_max_head_scorer,
    "fastv_mean_std": fastv_mean_std_scorer,
    "text_query_mean": text_query_mean_scorer,
    "entropy_weighted": entropy_weighted_scorer,
    "hidden_norm": hidden_norm_scorer,
    # Normalized linear combinations
    "combo_mean_text": combo_mean_text_scorer,
    "combo_attn_ensemble": combo_attn_ensemble_scorer,
    "combo_mean_text_norm": combo_mean_text_norm_scorer,
    "combo_mean_text_norm_minmax": combo_mean_text_norm_minmax_scorer,
    "linear_combination": linear_combination_scorer,
}


def register_scorer(name: str, fn: ScorerFn) -> None:
    """Register a custom visual token scoring function by name."""
    if not callable(fn):
        raise TypeError(f"Scorer must be callable, got {type(fn)!r}")
    SCORERS[str(name)] = fn


def resolve_scorer(scorer: Union[str, ScorerFn]) -> ScorerFn:
    """Resolve a scorer name or callable into a scoring function."""
    if callable(scorer):
        return scorer
    if isinstance(scorer, str):
        if scorer not in SCORERS:
            available = ", ".join(sorted(SCORERS.keys()))
            raise KeyError(f"Unknown FastV scorer {scorer!r}. Available scorers: {available}")
        return SCORERS[scorer]
    raise TypeError(f"scorer must be a str or callable, got {type(scorer)!r}")


__all__ = [
    "SCORERS",
    "ScorerFn",
    "VALID_NORMALIZATIONS",
    "combo_attn_ensemble_scorer",
    "combo_mean_text_norm_minmax_scorer",
    "combo_mean_text_norm_scorer",
    "combo_mean_text_scorer",
    "entropy_weighted_scorer",
    "fastv_max_head_scorer",
    "fastv_mean_scorer",
    "fastv_mean_std_scorer",
    "hidden_norm_scorer",
    "linear_combination_scorer",
    "make_combined_scorer",
    "normalize_scores",
    "register_scorer",
    "resolve_scorer",
    "score_visual_tokens",
    "text_query_mean_scorer",
]
