"""Modern FastV KV-Cache adapter for lmms-eval / HuggingFace LLaVA models.

Compatible with upstream HuggingFace ``transformers`` (>=4.45 through 5.x)
without requiring manual edits to ``modeling_llama.py`` rotary embeddings.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Union

_FASTV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _FASTV_DIR not in sys.path:
    sys.path.insert(0, _FASTV_DIR)

from fastv_hf import FastVConfig, FastVController, attach_fastv


def enable_fastv_for_lmms_eval(
    model: Any,
    k: int = 3,
    r: float = 0.5,
    mode: str = "inplace",
    image_token_start_index: Optional[int] = None,
    image_token_length: Optional[int] = None,
) -> FastVController:
    """Attach FastV KV-cache pruning to any HF LLaVA/Llama model used inside ``lmms-eval``.

    Args:
        model: Loaded ``LlavaForConditionalGeneration`` or ``LlamaForCausalLM``.
        k: Layer index after which visual tokens are pruned (default ``3``, matching
           the README lmms-eval benchmark table).
        r: Fraction of visual tokens to prune (default ``0.5``, keeping 50%).
        mode: ``"inplace"`` (true KV-cache pruning) or ``"mask"``.
        image_token_start_index: Optional override; auto-detected from ``input_ids`` if None.
        image_token_length: Optional override; auto-detected from ``input_ids`` if None.
    """
    cfg = FastVConfig(
        use_fastv=True,
        fastv_k=k,
        fastv_r=r,
        mode=mode,
        image_token_start_index=image_token_start_index,
        image_token_length=image_token_length,
    )
    return attach_fastv(model, cfg)


def apply_fastv_from_env(model: Any) -> Optional[FastVController]:
    """Attach FastV if ``USE_FASTV=1`` is set in the environment."""
    if os.environ.get("USE_FASTV", "0") != "1":
        return None
    k = int(os.environ.get("FASTV_K", "3"))
    r = float(os.environ.get("FASTV_R", "0.5"))
    mode = os.environ.get("FASTV_MODE", "inplace")
    return enable_fastv_for_lmms_eval(model, k=k, r=r, mode=mode)


__all__ = ["FastVConfig", "FastVController", "enable_fastv_for_lmms_eval", "apply_fastv_from_env"]
