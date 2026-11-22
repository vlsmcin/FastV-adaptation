"""FastV HuggingFace LLaVA Demo (modern transformers >=4.45 / 5.x compatible).

Runs LLaVA-1.5 with plug-and-play FastV visual token pruning via PyTorch hooks
(no vendored transformers fork required). Supports 4-bit/8-bit/fp16 loading,
KV-cache pruning, and optional side-by-side Vanilla vs FastV comparison.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import TextStreamer

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


def load_image(image_source: str) -> Image.Image:
    if image_source.startswith(("http://", "https://")):
        resp = requests.get(image_source, timeout=30)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_source).convert("RGB")


def run_generation(
    model,
    processor,
    controller,
    inputs,
    fastv_cfg: FastVConfig,
    max_new_tokens: int,
    min_new_tokens: int,
    use_cache: bool,
    stream: bool = True,
):
    controller.update_config(fastv_cfg)
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True) if stream else None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
            streamer=streamer,
            return_dict_in_generate=True,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_mem_gib = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    prompt_len = int(inputs["input_ids"].shape[1])
    generated_ids = output.sequences[0, prompt_len:]
    num_tokens = int(generated_ids.shape[0])
    time_per_token = elapsed / max(1, num_tokens)
    decoded = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "text": decoded,
        "elapsed_s": elapsed,
        "num_tokens": num_tokens,
        "time_per_token_s": time_per_token,
        "peak_gpu_mem_gib": peak_mem_gib,
        "fastv_stats": dict(controller.last_stats),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastV HuggingFace LLaVA Demo")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HF model ID (defaults to auto-selection via FASTV_MODEL_SIZE / GPU VRAM).",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=os.environ.get("FASTV_QUANTIZATION", "4bit"),
        choices=["4bit", "8bit", "none"],
        help="Quantization mode (default: 4bit).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="http://images.cocodataset.org/val2017/000000039769.jpg",
        help="Image URL or local file path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What are these? Describe the image in details",
        help="User prompt.",
    )
    parser.add_argument(
        "--fastv-k",
        type=int,
        default=int(os.environ.get("FASTV_K", "3")),
        help="Pruning layer K (default: 3).",
    )
    parser.add_argument(
        "--fastv-r",
        type=float,
        default=float(os.environ.get("FASTV_R", "0.75")),
        help="Visual token pruning ratio R in [0, 1] (default: 0.75).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="inplace",
        choices=["inplace", "mask"],
        help="FastV pruning mode (default: inplace).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--min-new-tokens", type=int, default=200)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable KV cache (re-prunes visual tokens at every step like original paper).",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        default=os.environ.get("FASTV_COMPARE_BASELINE", "0") == "1",
        help="Run Vanilla decoding first and compare latency, memory, and FLOPs against FastV.",
    )
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gpu_memory_gib = (
        torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    model_id = resolve_hf_model_id(
        model_path=args.model_id,
        requested_size=os.environ.get("FASTV_MODEL_SIZE", ""),
        gpu_memory_gib=gpu_memory_gib,
    )
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    use_cache = not args.no_cache

    fastv_config = FastVConfig(
        use_fastv=True,
        fastv_k=args.fastv_k,
        fastv_r=args.fastv_r,
        mode=args.mode,
    )

    print(f"Loading model: {model_id} (quantization={args.quantization}, GPU={gpu_memory_gib:.2f} GiB)")
    model, processor, controller = load_llava_model(
        model_id=model_id,
        quantization=args.quantization,
        dtype=dtype,
        attn_implementation="eager",
        fastv_config=fastv_config,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
    )

    raw_image = load_image(args.image)
    device = next(model.parameters()).device
    inputs, image_span = prepare_llava_inputs(
        processor=processor,
        prompt=args.prompt,
        image=raw_image,
        device=device,
        dtype=dtype,
        image_token_index=int(getattr(model.config, "image_token_index", 32000)),
    )

    text_cfg = getattr(model.config, "text_config", model.config)
    num_layers = int(getattr(text_cfg, "num_hidden_layers", 32))
    hidden_size = int(getattr(text_cfg, "hidden_size", 4096))
    intermediate_size = int(getattr(text_cfg, "intermediate_size", 11008))
    img_len = image_span[1] if image_span else 576
    text_len = max(1, int(inputs["input_ids"].shape[1]) - img_len)

    flops_info = compute_fastv_flops_ratio(
        num_layers=num_layers,
        fastv_k=args.fastv_k,
        fastv_r=args.fastv_r,
        n_image_tokens=img_len,
        n_text_tokens=text_len,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    baseline_res = None
    if args.compare_baseline:
        print("\n----------------------------Vanilla Baseline-------------------------------")
        baseline_res = run_generation(
            model=model,
            processor=processor,
            controller=controller,
            inputs=inputs,
            fastv_cfg=FastVConfig(use_fastv=False),
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            use_cache=use_cache,
            stream=not args.no_stream,
        )
        print(f"Baseline Total time: {baseline_res['elapsed_s']:.3f}s")
        print(f"Baseline Time/token: {baseline_res['time_per_token_s']:.4f}s/token")
        print(f"Baseline Peak GPU memory: {baseline_res['peak_gpu_mem_gib']:.2f} GiB")

    print("\n----------------------------FastV-------------------------------")
    print(fastv_config.to_dict())
    print(f"GPU memory: {gpu_memory_gib:.2f} GiB")
    print(f"Model: {model_id}")
    print(f"Detected image span: {image_span}")
    print(
        f"Theoretical FLOPs ratio: {flops_info['flops_ratio_pct']:.1f}% "
        f"(reduction: {flops_info['flops_reduction_pct']:.1f}%)"
    )

    fastv_res = run_generation(
        model=model,
        processor=processor,
        controller=controller,
        inputs=inputs,
        fastv_cfg=fastv_config,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        use_cache=use_cache,
        stream=not args.no_stream,
    )

    print(f"Total time taken: {fastv_res['elapsed_s']:.4f} seconds")
    print(f"Time per token: {fastv_res['time_per_token_s']:.5f} seconds/token")
    print(f"Peak GPU memory: {fastv_res['peak_gpu_mem_gib']:.2f} GiB")
    if fastv_res["fastv_stats"]:
        print(f"FastV stats: {fastv_res['fastv_stats']}")
    if baseline_res is not None:
        speedup = baseline_res["elapsed_s"] / max(1e-6, fastv_res["elapsed_s"])
        mem_saved = baseline_res["peak_gpu_mem_gib"] - fastv_res["peak_gpu_mem_gib"]
        print(f"Speedup vs Vanilla: {speedup:.2f}x | Peak GPU memory delta: {mem_saved:+.2f} GiB")


if __name__ == "__main__":
    main()
