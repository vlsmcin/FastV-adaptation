"""A-OKVQA Accuracy, Latency, and GPU Memory Benchmark for FastV (Paper Table 4).

Modernized to run directly on upstream HuggingFace ``transformers`` (>=4.45 / 5.x)
with ``LlavaForConditionalGeneration`` and hook-based FastV pruning.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from io import BytesIO

import requests
import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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

TEMPLATE = """
Analyse the image and choose the best answer for the following question:
{question}
Options: {options}
Just output the letter of the correct answer."""


def format_choices(choices) -> str:
    return " ".join([f"({chr(ord('A') + i)}) {choice}" for i, choice in enumerate(choices)])


def format_answer(answer_index: int) -> str:
    return f"{chr(ord('A') + int(answer_index))}"


def load_aokvqa_dataset(dataset_path: str):
    if os.path.isdir(dataset_path):
        return load_from_disk(dataset_path)
    print(f"Local dataset {dataset_path!r} not found; loading HuggingFaceM4/A-OKVQA validation split...")
    return load_dataset("HuggingFaceM4/A-OKVQA", split="validation")


def extract_choice_letter(text: str) -> str:
    cleaned = text.strip().upper()
    m = re.search(r"\b([A-D])\b", cleaned)
    if m:
        return m.group(1)
    for ch in cleaned:
        if ch in ("A", "B", "C", "D"):
            return ch
    return cleaned[:1]


def compute_acc(model_outputs: list[str], correct_answers: list[str]) -> float:
    if not model_outputs:
        return 0.0
    correct = 0
    for pred, gt in zip(model_outputs, correct_answers):
        if gt in pred.upper() or extract_choice_letter(pred) == gt:
            correct += 1
    return correct / len(model_outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastV A-OKVQA Latency & Accuracy Evaluation")
    parser.add_argument(
        "--model-path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HF model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./data/aokvqa/validation",
        help="Local HuggingFace dataset directory (falls back to HuggingFaceM4/A-OKVQA).",
    )
    parser.add_argument("--use-fast-v", default=False, action="store_true", help="Enable FastV pruning.")
    parser.add_argument(
        "--fast-v-inplace",
        default=False,
        action="store_true",
        help="Use inplace token dropping (real latency & memory speedup). If omitted, uses token masking.",
    )
    parser.add_argument("--fast-v-sys-length", type=int, default=None, help="Optional override for image start index.")
    parser.add_argument("--fast-v-image-token-length", type=int, default=None, help="Optional override for image tokens.")
    parser.add_argument("--fast-v-attention-rank", type=int, default=None, help="Number of visual tokens to keep.")
    parser.add_argument("--fast-v-r", type=float, default=None, help="Pruning ratio R in [0, 1] (alternative to rank).")
    parser.add_argument("--fast-v-agg-layer", type=int, default=2, help="Pruning layer K (default: 2).")
    parser.add_argument(
        "--scorer",
        type=str,
        default="fastv_mean",
        help="Visual token importance scorer from FastV.scoring.SCORERS (default: fastv_mean).",
    )
    parser.add_argument(
        "--include-system-prompt",
        default=False,
        action="store_true",
        help="Prepend legacy Vicuna v1 system prompt (default: False, matching llava-hf format).",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=os.environ.get("FASTV_QUANTIZATION", "4bit"),
        choices=["4bit", "8bit", "none"],
        help="Model quantization (default: 4bit).",
    )
    parser.add_argument("--use-cache", default=False, action="store_true", help="Enable KV cache during generation.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples (default: all 1,145 validation samples).",
    )
    parser.add_argument("--warmup-samples", type=int, default=3, help="Warmup samples excluded from latency stats.")
    parser.add_argument("--output-path", type=str, required=True, help="Output JSON file path.")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress per-sample output prints.")
    return parser.parse_args()


def main() -> None:
    pargs = parse_args()
    print(pargs)

    dataset = load_aokvqa_dataset(pargs.dataset_path)
    total_available = len(dataset)
    if pargs.max_samples is not None and pargs.max_samples < total_available:
        dataset = dataset.select(range(pargs.max_samples))

    valid_images = dataset["image"]
    valid_questions = dataset["question"]
    valid_choices = dataset["choices"]
    valid_answers = dataset["correct_choice_idx"]

    valid_answer_options = [format_answer(idx) for idx in valid_answers]
    valid_prompts = [
        TEMPLATE.format(question=q, options=format_choices(c))
        for q, c in zip(valid_questions, valid_choices)
    ]

    model_id = resolve_hf_model_id(pargs.model_path)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if pargs.use_fast_v:
        mode = "inplace" if pargs.fast_v_inplace else "mask"
        fastv_r = pargs.fast_v_r if pargs.fast_v_r is not None else 0.5
        fastv_cfg = FastVConfig(
            use_fastv=True,
            fastv_k=pargs.fast_v_agg_layer,
            fastv_r=fastv_r,
            mode=mode,
            scorer=pargs.scorer,
            image_token_start_index=pargs.fast_v_sys_length,
            image_token_length=pargs.fast_v_image_token_length,
            attention_rank=pargs.fast_v_attention_rank,
        )
    else:
        fastv_cfg = FastVConfig(use_fastv=False)

    model, processor, controller = load_llava_model(
        model_id=model_id,
        quantization=pargs.quantization,
        dtype=dtype,
        attn_implementation="eager",
        fastv_config=fastv_cfg,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
    )

    device = next(model.parameters()).device
    image_token_index = int(getattr(model.config, "image_token_index", 32000))

    outputs: list[str] = []
    latencies: list[float] = []
    prompt_lengths: list[int] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for idx, (prompt, image) in enumerate(
        tqdm(zip(valid_prompts, valid_images), total=len(valid_prompts), desc="A-OKVQA")
    ):
        inputs, _ = prepare_llava_inputs(
            processor=processor,
            prompt=prompt,
            image=image,
            device=device,
            dtype=dtype,
            include_system_prompt=pargs.include_system_prompt,
            image_token_index=image_token_index,
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        prompt_lengths.append(prompt_len)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1,
                use_cache=pargs.use_cache,
                return_dict_in_generate=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if idx >= pargs.warmup_samples or len(valid_prompts) <= pargs.warmup_samples:
            latencies.append(dt)

        decoded = processor.tokenizer.decode(
            gen.sequences[0, prompt_len:], skip_special_tokens=True
        ).strip()
        outputs.append(decoded)
        if not pargs.quiet and idx < 5:
            print(f"[sample {idx}] pred={decoded!r} gt={valid_answer_options[idx]!r} latency={dt:.4f}s")

    acc = compute_acc(outputs, valid_answer_options)
    avg_latency = sum(latencies) / max(1, len(latencies))
    peak_mem_gib = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    text_cfg = getattr(model.config, "text_config", model.config)
    img_len = fastv_cfg.image_token_length or 576
    avg_text_len = max(1, int(round((sum(prompt_lengths) / max(1, len(prompt_lengths))) - img_len)))
    flops = compute_fastv_flops_ratio(
        num_layers=int(getattr(text_cfg, "num_hidden_layers", 32)),
        fastv_k=fastv_cfg.fastv_k,
        fastv_r=fastv_cfg.effective_pruning_ratio(img_len) if fastv_cfg.use_fastv else 0.0,
        n_image_tokens=img_len,
        n_text_tokens=avg_text_len,
        hidden_size=int(getattr(text_cfg, "hidden_size", 4096)),
        intermediate_size=int(getattr(text_cfg, "intermediate_size", 11008)),
    )

    out_dir = os.path.dirname(os.path.abspath(pargs.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    result = {
        "model": model_id,
        "quantization": pargs.quantization,
        "num_samples": len(outputs),
        "fastv_config": fastv_cfg.to_dict(),
        "acc": str(acc),
        "acc_pct": round(acc * 100.0, 2),
        "avg_latency_s": round(avg_latency, 5),
        "peak_gpu_mem_gib": round(peak_mem_gib, 3),
        "flops_ratio_pct": round(flops["flops_ratio_pct"], 2),
        "flops_reduction_pct": round(flops["flops_reduction_pct"], 2),
        "output": outputs,
        "labels": valid_answer_options,
    }
    with open(pargs.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(
        f"A-OKVQA Done | acc={acc * 100:.2f}% | avg_latency={avg_latency:.4f}s | "
        f"peak_mem={peak_mem_gib:.2f} GiB | FLOPs={flops['flops_ratio_pct']:.1f}% -> {pargs.output_path}"
    )


if __name__ == "__main__":
    main()
