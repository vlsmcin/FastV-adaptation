"""Pluggable visual token importance scoring functions for FastV.

All scorers share the uniform signature:

    def scorer(
        attn_weights: torch.Tensor,   # (batch, num_heads, q_len, kv_len)
        vis_indices: torch.Tensor,    # 1D LongTensor of visual token indices
        **kwargs,
    ) -> torch.Tensor:                # 1D Tensor of importance scores (len == len(vis_indices))

Adding a new research metric (e.g., head-wise variance, entropy weighting,
multi-query aggregation) only requires registering a function here without
modifying any model adapter or pruning hook.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Union

ScorerFn = Callable[..., Any]


def fastv_mean_scorer(
    attn_weights: Any,
    vis_indices: Any,
    query_index: int = -1,
    **_kwargs: Any,
) -> Any:
    """Original FastV paper scorer: mean attention across heads from the last query token."""
    # attn_weights: (batch, num_heads, q_len, kv_len)
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
    per_head_q = attn_weights.float()[0, :, query_index, :]  # (num_heads, kv_len)
    mean = per_head_q.mean(dim=0)
    std = per_head_q.std(dim=0, unbiased=False)
    combined = mean + float(alpha) * std
    return combined.index_select(0, vis_indices.to(combined.device))


SCORERS: Dict[str, ScorerFn] = {
    "fastv_mean": fastv_mean_scorer,
    "fastv_max_head": fastv_max_head_scorer,
    "fastv_mean_std": fastv_mean_std_scorer,
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
    "fastv_max_head_scorer",
    "fastv_mean_scorer",
    "fastv_mean_std_scorer",
    "register_scorer",
    "resolve_scorer",
]
