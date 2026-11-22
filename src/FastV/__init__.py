"""FastV: Plug-and-Play Inference Acceleration for Large Vision-Language Models."""

from .fastv_hf import (
    FastVConfig,
    FastVController,
    attach_fastv,
    compute_fastv_flops_ratio,
    compute_layer_attention_efficiency,
    compute_rouge_l,
    detect_image_token_span,
    disable_fastv,
    get_llama_backbone,
    load_llava_model,
    prepare_llava_inputs,
    resolve_hf_model_id,
    set_fastv_config,
)

__all__ = [
    "FastVConfig",
    "FastVController",
    "attach_fastv",
    "compute_fastv_flops_ratio",
    "compute_layer_attention_efficiency",
    "compute_rouge_l",
    "detect_image_token_span",
    "disable_fastv",
    "get_llama_backbone",
    "load_llava_model",
    "prepare_llava_inputs",
    "resolve_hf_model_id",
    "set_fastv_config",
]
