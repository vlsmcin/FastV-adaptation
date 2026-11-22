"""Interactive Gradio Demo for FastV (compatible with Gradio 4/5/6 & modern transformers)."""

from __future__ import annotations

import argparse
import base64
import inspect
import os
import sys
import time
import warnings
from io import BytesIO

import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from matplotlib.colors import LogNorm
from PIL import Image

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("GRADIO_TEMP_DIR", "gradio_temp")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from FastV.fastv_hf import (
    FastVConfig,
    compute_fastv_flops_ratio,
    load_llava_model,
    prepare_llava_inputs,
    resolve_hf_model_id,
)


def visualize_attention(
    multihead_attention: torch.Tensor,
    title: str = "Layer 5",
    image_start: int = 5,
    image_length: int = 576,
    sample_style: str = "All layers",
) -> Image.Image:
    averaged_attention = torch.mean(multihead_attention.float(), dim=1)[0]
    pool_size = max(1, min(20, averaged_attention.shape[0] // 2))
    averaged_attention = (
        torch.nn.functional.avg_pool2d(
            averaged_attention.unsqueeze(0).unsqueeze(0), pool_size, stride=pool_size
        )
        .squeeze(0)
        .squeeze(0)
    )

    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(5, 5), dpi=250)
    vmax = max(float(averaged_attention.max().item()), 1e-3)
    vmin = min(0.0007, vmax / 10.0)
    log_norm = LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)

    ax = sns.heatmap(averaged_attention.numpy(), cmap=cmap, norm=log_norm)

    n_bins = averaged_attention.shape[0]
    ticks = [str(i * pool_size) for i in range(n_bins)]
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

    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5, rotation=0)
    plt.title(title, fontsize=16)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf).copy()
    target_size = (768, 768) if sample_style == "All layers" else (1024, 1024)
    image = image.resize(target_size)
    buf.close()
    plt.close(fig)
    return image


def select_numbers(n: int, x: int) -> list[int]:
    if x <= 1:
        return [0]
    return [(i * (n - 1)) // (x - 1) for i in range(x)]


def pil_to_base64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="FastV Interactive Demo")
    parser.add_argument("--model-path", type=str, default=None, help="HF model ID or local path.")
    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
    )
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true", default=False)
    pargs = parser.parse_args()

    model_id = resolve_hf_model_id(pargs.model_path)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Loading model {model_id} (quantization={pargs.quantization})...")
    model, processor, controller = load_llava_model(
        model_id=model_id,
        quantization=pargs.quantization,
        dtype=dtype,
        attn_implementation="eager",
        fastv_config=FastVConfig(use_fastv=False),
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
    )

    text_cfg = getattr(model.config, "text_config", model.config)
    total_layers = int(getattr(text_cfg, "num_hidden_layers", 32))
    hidden_size = int(getattr(text_cfg, "hidden_size", 4096))
    intermediate_size = int(getattr(text_cfg, "intermediate_size", 11008))
    device = next(model.parameters()).device
    image_token_index = int(getattr(model.config, "image_token_index", 32000))

    def inference(image_input, prompt, num_of_layers, use_fastv, fastv_k, fastv_r):
        if image_input is None:
            return [], "Please upload an image.", ""
        prompt = prompt or "Describe the image in detail."

        inputs, span = prepare_llava_inputs(
            processor=processor,
            prompt=prompt,
            image=image_input,
            device=device,
            dtype=dtype,
            include_system_prompt=True,
            image_token_index=image_token_index,
        )
        img_start, img_len = span if span else (35, 576)

        # Step 1: Generate answer with user-selected FastV configuration.
        controller.update_config(
            FastVConfig(
                use_fastv=bool(use_fastv),
                fastv_k=int(fastv_k),
                fastv_r=float(fastv_r),
                mode="inplace",
            )
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=200,
                use_cache=True,
                return_dict_in_generate=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_cost = time.perf_counter() - t0
        prompt_len = int(inputs["input_ids"].shape[1])
        output_text = processor.tokenizer.decode(
            gen.sequences[0, prompt_len:], skip_special_tokens=True
        ).strip()

        # Step 2: Single forward pass with vanilla model to capture layer attention maps.
        controller.disable()
        vis_inputs, _ = prepare_llava_inputs(
            processor=processor,
            prompt=prompt,
            image=image_input,
            device=device,
            dtype=dtype,
            include_system_prompt=True,
            append_output=" " + output_text,
            image_token_index=image_token_index,
        )
        with torch.inference_mode():
            fwd = model(**vis_inputs, output_attentions=True, use_cache=False, return_dict=True)

        if num_of_layers == "All layers":
            show_layers = list(range(total_layers))
        elif num_of_layers == "Sample 3 layers":
            show_layers = select_numbers(total_layers, 3)
        elif num_of_layers == "Sample 5 layers":
            show_layers = select_numbers(total_layers, 5)
        elif num_of_layers == "Sample 10 layers":
            show_layers = select_numbers(total_layers, 10)
        else:
            show_layers = select_numbers(total_layers, 5)

        images_list = []
        if fwd.attentions is not None:
            for layer_idx in show_layers:
                images_list.append(
                    visualize_attention(
                        fwd.attentions[layer_idx].detach().cpu(),
                        title=f"Layer {layer_idx + 1}",
                        image_start=img_start,
                        image_length=img_len,
                        sample_style=num_of_layers,
                    )
                )

        flops = compute_fastv_flops_ratio(
            num_layers=total_layers,
            fastv_k=int(fastv_k),
            fastv_r=float(fastv_r) if use_fastv else 0.0,
            n_image_tokens=img_len,
            n_text_tokens=max(1, prompt_len - img_len),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
        summary = (
            f"Generation Time: {time_cost:.2f}s | "
            f"FastV: {'ON' if use_fastv else 'OFF'} (K={int(fastv_k)}, R={float(fastv_r):.0%}) | "
            f"Theoretical FLOPs Reduction: {flops['flops_reduction_pct']:.1f}%"
        )
        return images_list, output_text, summary

    fastv_tradeoff = pil_to_base64(os.path.join(REPO_ROOT, "figs", "fastv_tradeoff.png"))
    img_banner = (
        f'<center><img src="data:image/png;base64,{fastv_tradeoff}" alt="FastV tradeoff" style="width: 40%;"/></center>'
        if fastv_tradeoff
        else ""
    )

    description = f"""# FastV Interactive Demo
Welcome to the modernized demonstration for [FastV](https://arxiv.org/abs/2403.06764) (*An Image is Worth 1/2 Tokens After Layer 2*).
{img_banner}
The **System Prompt tokens are highlighted in blue**, **Image tokens in red**, and remaining **Text tokens in black**.
"""

    examples = [
        ["figs/example.jpg", "Describe all the objects in the image.", "Sample 5 layers", True, 2, 0.5],
        ["figs/example2.jpg", "Describe the image in detail.", "Sample 5 layers", True, 2, 0.5],
        ["figs/example3.jpg", "Describe the animal in image in detail.", "Sample 5 layers", True, 2, 0.5],
    ]

    interface_kwargs = dict(
        fn=inference,
        inputs=[
            gr.Image(type="pil", label="Image"),
            gr.Textbox(label="Prompt", placeholder="Describe the image in detail.", lines=2),
            gr.Dropdown(
                choices=["Sample 3 layers", "Sample 5 layers", "Sample 10 layers", "All layers"],
                value="Sample 5 layers",
                label="Layer Attention Visualization",
            ),
            gr.Checkbox(value=True, label="Enable FastV Pruning"),
            gr.Slider(minimum=0, maximum=total_layers - 1, step=1, value=2, label="FastV Layer K"),
            gr.Slider(minimum=0.0, maximum=0.9, step=0.05, value=0.5, label="FastV Pruning Ratio R"),
        ],
        outputs=[
            gr.Gallery(label="Attention Maps", columns=5, object_fit="contain", height="auto"),
            gr.Textbox(label="Model Output"),
            gr.Textbox(label="Latency & FLOPs Summary"),
        ],
        description=description,
        examples=examples,
    )

    # Gradio 5/6 uses `flagging_mode="never"` while Gradio 4 uses `allow_flagging="never"`.
    sig = inspect.signature(gr.Interface.__init__).parameters
    if "flagging_mode" in sig:
        interface_kwargs["flagging_mode"] = "never"
    elif "allow_flagging" in sig:
        interface_kwargs["allow_flagging"] = "never"

    demo = gr.Interface(**interface_kwargs)
    demo.launch(share=pargs.share, server_name="0.0.0.0", server_port=pargs.port)


if __name__ == "__main__":
    main()
