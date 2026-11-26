"""Visualize Inefficient Visual Attention in LLaVA across layers (Paper Figures 2 & 3).

Modernized to run directly on upstream HuggingFace ``transformers`` (>=4.45 / 5.x)
with ``LlavaForConditionalGeneration`` (supports 4-bit/8-bit/fp16 loading).

Produces:
  * ``output.json``: Prompt, image, generated response, and token span metadata
  * ``attn_maps/atten_map_<layer>.png``: Individual layer attention heatmaps
  * ``attn_grid.png``: Multi-layer attention grid (Paper Figure 2 style)
  * ``attention_efficiency.json`` & ``attention_efficiency.png``: Quantitative
    per-layer attention share and visual attention efficiency (Paper Figure 3 style)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from io import BytesIO

import matplotlib

if __name__ == "__main__" and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import torch
from matplotlib.colors import LogNorm
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from FastV import (
    FastVConfig,
    compute_layer_attention_efficiency,
    load_vlm_model,
    prepare_vlm_inputs,
    resolve_handler,
)


def load_image(image_file: str) -> Image.Image:
    if image_file.startswith(("http://", "https://")):
        response = requests.get(image_file, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def visualize_attention(
    multihead_attention: torch.Tensor,
    output_path: str = "atten_map_1.png",
    title: str = "Layer 5",
    image_start: int = 35,
    image_length: int = 576,
    pool_stride: int = 20,
):
    """Plot pooled log-scaled attention heatmap for a single transformer layer."""
    averaged_attention = torch.mean(multihead_attention.float(), dim=1)[0]
    stride = max(1, min(pool_stride, averaged_attention.shape[0] // 2))
    pooled = (
        torch.nn.functional.avg_pool2d(
            averaged_attention.unsqueeze(0).unsqueeze(0), stride, stride=stride
        )
        .squeeze(0)
        .squeeze(0)
    )

    pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1.0, neginf=0.0)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(5, 5), dpi=250)

    raw_max = float(pooled.max().item()) if pooled.numel() > 0 else 0.0
    vmax = raw_max if (raw_max > 1e-3 and raw_max == raw_max) else 1e-3
    vmin = max(min(0.0007, vmax / 10.0), 1e-6)
    log_norm = LogNorm(vmin=vmin, vmax=vmax)

    ax = sns.heatmap(pooled.numpy(), cmap=cmap, norm=log_norm)

    n_bins = pooled.shape[0]
    ticks = [str(i * stride) for i in range(n_bins)]
    ax.set_xticks(range(n_bins))
    ax.set_yticks(range(n_bins))
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    img_end = image_start + image_length
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        pos = int(label.get_text())
        if 0 <= pos < image_start:
            label.set_color("blue")
            label.set_fontweight("bold")
        elif image_start <= pos < img_end:
            label.set_color("red")

    plt.xticks(fontsize=4, rotation=90)
    plt.yticks(fontsize=4, rotation=0)
    plt.title(title)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    top_k = min(10, pooled.shape[-1])
    top_attentions = []
    for row in pooled:
        top_values, top_indices = torch.topk(row, top_k)
        top_attentions.append(list(zip(top_indices.tolist(), top_values.tolist())))

    return top_attentions, pooled


def plot_attention_grid(
    attentions,
    output_path: str,
    image_start: int,
    image_length: int,
    selected_layers: list[int] | None = None,
    pool_stride: int = 20,
) -> None:
    """Save a multi-layer attention grid (reproducing Paper Figure 2)."""
    total_layers = len(attentions)
    if selected_layers is None:
        selected_layers = sorted(
            set(
                idx
                for idx in [0, 1, 2, 3, 7, 15, 23, total_layers - 1]
                if 0 <= idx < total_layers
            )
        )
    ncols = min(4, len(selected_layers))
    nrows = (len(selected_layers) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), dpi=180)
    if nrows * ncols == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flat)

    cmap = plt.get_cmap("viridis")
    for ax, layer_idx in zip(axes_list, selected_layers):
        avg = torch.mean(attentions[layer_idx].detach().cpu().float(), dim=1)[0]
        stride = max(1, min(pool_stride, avg.shape[0] // 2))
        pooled = (
            torch.nn.functional.avg_pool2d(
                avg.unsqueeze(0).unsqueeze(0), stride, stride=stride
            )
            .squeeze(0)
            .squeeze(0)
        )
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1.0, neginf=0.0)
        raw_max = float(pooled.max().item()) if pooled.numel() > 0 else 0.0
        vmax = raw_max if (raw_max > 1e-3 and raw_max == raw_max) else 1e-3
        vmin = max(min(0.0007, vmax / 10.0), 1e-6)
        sns.heatmap(
            pooled.numpy(),
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            ax=ax,
            cbar=False,
        )
        ax.set_title(f"Layer {layer_idx + 1}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Key Token Bin (red=image)", fontsize=8)
        ax.set_ylabel("Query Token Bin", fontsize=8)
        img_start_bin = image_start / stride
        img_end_bin = (image_start + image_length) / stride
        ax.axvline(img_start_bin, color="red", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.axvline(img_end_bin, color="red", linestyle="--", linewidth=0.8, alpha=0.8)

    for ax in axes_list[len(selected_layers) :]:
        ax.axis("off")

    fig.suptitle(
        "Inefficient Visual Attention Across Layers (Red lines mark image tokens)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_attention_efficiency(metrics: dict, output_path: str) -> None:
    """Plot per-layer attention share & visual attention efficiency (reproducing Paper Figure 3)."""
    layers = metrics["layer"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)

    ax1.plot(layers, metrics["system_total"], marker="o", markersize=3, label="System Prompt Tokens")
    ax1.plot(layers, metrics["image_total"], marker="s", markersize=3, label="Image Tokens")
    ax1.plot(layers, metrics["text_total"], marker="^", markersize=3, label="Text / Instruction Tokens")
    if any(v > 0 for v in metrics["output_total"]):
        ax1.plot(layers, metrics["output_total"], marker="d", markersize=3, label="Output Tokens")
    ax1.axvline(2, color="red", linestyle="--", alpha=0.7, label="Layer K=2")
    ax1.set_xlabel("Transformer Layer")
    ax1.set_ylabel("Total Attention Share")
    ax1.set_title("Attention Share by Token Type Across Layers")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(
        layers,
        metrics["visual_attention_efficiency"],
        marker="o",
        markersize=4,
        color="crimson",
        label="Visual Attention Efficiency (Share / Token Fraction)",
    )
    ax2.axhline(1.0, color="gray", linestyle=":", label="Uniform Attention Baseline (1.0)")
    ax2.axvline(2, color="red", linestyle="--", alpha=0.7, label="Layer K=2")
    ax2.set_xlabel("Transformer Layer")
    ax2.set_ylabel("Attention Efficiency")
    ax2.set_title("Visual Token Attention Efficiency Drops After Layer 2")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LLaVA Inefficient Visual Attention")
    parser.add_argument(
        "--model-path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HF model ID or local checkpoint path.",
    )
    parser.add_argument("--image-path", type=str, required=True, help="Path or URL to input image.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt describing what to ask.")
    parser.add_argument("--output-path", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument(
        "--quantization",
        type=str,
        default=os.environ.get("FASTV_QUANTIZATION", "4bit"),
        choices=["4bit", "8bit", "bf16", "fp16", "none"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--include-system-prompt",
        action="store_true",
        default=True,
        help="Include system prompt prefix.",
    )
    parser.add_argument(
        "--skip-individual-maps",
        action="store_true",
        default=False,
        help="Only generate the combined grid and efficiency plots (faster).",
    )
    pargs = parser.parse_args()

    handler = resolve_handler(pargs.model_path)
    model_id = handler.resolve_model_id(pargs.model_path)
    if torch.cuda.is_available():
        if pargs.quantization == "bf16" or (
            handler.name == "qwen2_vl"
            and pargs.quantization != "fp16"
            and torch.cuda.is_bf16_supported()
        ):
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    print(
        f"Loading {model_id} via {handler.name} (quantization={pargs.quantization}, dtype={dtype}) "
        "for attention visualization..."
    )
    model, processor, controller = load_vlm_model(
        model_id=model_id,
        handler=handler,
        quantization=pargs.quantization,
        dtype=dtype,
        attn_implementation="eager",
        fastv_config=FastVConfig(use_fastv=False),
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
    )
    controller.disable()

    device = next(model.parameters()).device
    vis_ids = controller.adapter.get_visual_token_ids(model)
    image_token_id = int(vis_ids[0]) if vis_ids else 32000
    image = load_image(pargs.image_path)

    # 1. Generate response without FastV.
    inputs, span = prepare_vlm_inputs(
        processor=processor,
        prompt=pargs.prompt,
        image=image,
        device=device,
        dtype=dtype,
        handler=handler,
        include_system_prompt=pargs.include_system_prompt,
        image_token_index=image_token_id,
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    img_start, img_len = span if span else (35, 576)

    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=pargs.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
    model_output_ori = processor.tokenizer.decode(
        gen_out.sequences[0, prompt_len:], skip_special_tokens=True
    ).strip()
    print(f"Generated output:\n{model_output_ori}\n")

    # 2. Single teacher-forced forward pass over prompt + generated output to extract full layer attentions.
    output_prefix = "" if handler.name == "qwen2_vl" else " "
    full_inputs, _ = prepare_vlm_inputs(
        processor=processor,
        prompt=pargs.prompt,
        image=image,
        device=device,
        dtype=dtype,
        handler=handler,
        include_system_prompt=pargs.include_system_prompt,
        append_output=output_prefix + model_output_ori,
        image_token_index=image_token_id,
    )
    with torch.inference_mode():
        fwd_out = model(
            **full_inputs,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = [a.detach().cpu() for a in fwd_out.attentions]
    total_layers = len(attentions)

    os.makedirs(pargs.output_path, exist_ok=True)
    attn_dir = os.path.join(pargs.output_path, "attn_maps")
    os.makedirs(attn_dir, exist_ok=True)

    # Compute Figure 3 attention efficiency statistics across all layers.
    efficiency = compute_layer_attention_efficiency(
        attentions=attentions,
        image_start=img_start,
        image_length=img_len,
        instruction_end=prompt_len,
    )

    with open(os.path.join(pargs.output_path, "output.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_id,
                "prompt": pargs.prompt,
                "image": pargs.image_path,
                "output": [model_output_ori],
                "image_token_start_index": img_start,
                "image_token_length": img_len,
                "prompt_length": prompt_len,
                "full_sequence_length": int(full_inputs["input_ids"].shape[1]),
                "num_layers": total_layers,
            },
            f,
            indent=4,
        )

    with open(os.path.join(pargs.output_path, "attention_efficiency.json"), "w", encoding="utf-8") as f:
        json.dump(efficiency, f, indent=2)

    plot_attention_grid(
        attentions=attentions,
        output_path=os.path.join(pargs.output_path, "attn_grid.png"),
        image_start=img_start,
        image_length=img_len,
    )
    plot_attention_efficiency(
        metrics=efficiency,
        output_path=os.path.join(pargs.output_path, "attention_efficiency.png"),
    )

    if not pargs.skip_individual_maps:
        for layer_idx in range(total_layers):
            visualize_attention(
                attentions[layer_idx],
                output_path=os.path.join(attn_dir, f"atten_map_{layer_idx}.png"),
                title=f"Layer {layer_idx + 1}",
                image_start=img_start,
                image_length=img_len,
            )

    print(
        f"Saved attention maps & efficiency plots to {pargs.output_path} "
        f"(Layer 1 visual share: {efficiency['image_total'][0]:.3f} -> "
        f"Layer 3 visual share: {efficiency['image_total'][min(2, total_layers - 1)]:.3f})"
    )


if __name__ == "__main__":
    main()
