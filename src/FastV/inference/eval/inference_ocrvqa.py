"""OCR-VQA Hyperparameter Ablation over K and R for FastV (Paper Figure 7).

Modernized to run directly on upstream HuggingFace ``transformers`` (>=4.45 / 5.x)
with ``LlavaForConditionalGeneration`` and hook-based FastV (supporting both
``mask`` mode for Figure 7 ablations including K=0, and ``inplace`` mode for
real latency measurements). Uses a pure-Python ROUGE-L scorer matching
``pycocoevalcap.rouge.rouge.Rouge`` (beta=1.2) so no Java/C build is required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import torch
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from FastV import (
    FastVConfig,
    compute_fastv_flops_ratio,
    compute_rouge_l,
    load_vlm_model,
    prepare_vlm_inputs,
    resolve_handler,
)

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't",
    "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't",
    "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've",
    "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't", "mustve": "must've",
    "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't",
    "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're",
    "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll",
    "y'allll": "y'all'll", "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}
MANUAL_MAP = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
ARTICLES = {"a", "an", "the"}
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
PUNCT = [
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]


def process_punctuation(in_text: str) -> str:
    out_text = in_text
    for p in PUNCT:
        if (p + " " in in_text or " " + p in in_text) or (COMMA_STRIP.search(in_text) is not None):
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    return PERIOD_STRIP.sub("", out_text)


def process_digit_article(in_text: str) -> str:
    out_text = []
    for word in in_text.lower().split():
        word = MANUAL_MAP.get(word, word)
        if word not in ARTICLES:
            out_text.append(CONTRACTIONS.get(word, word))
    return " ".join(out_text)


def clean_text(pred: str) -> str:
    pred = pred.replace("\n", " ").replace("\t", " ").strip()
    pred = process_punctuation(pred)
    return process_digit_article(pred)


def parse_bool_str(val: str) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "y")


def load_ocrvqa_samples(max_examples: int = 500, max_questions: int | None = None):
    """Stream OCR-VQA test split from HuggingFace Hub to avoid downloading the entire archive."""
    images, prompts, labels = [], [], []
    ds = load_dataset("howard-hou/OCR-VQA", split="test", streaming=True)
    for idx, sample in enumerate(ds):
        if idx >= max_examples:
            break
        img = sample["image"]
        for q, a in zip(sample["questions"], sample["answers"]):
            images.append(img)
            prompts.append(q)
            labels.append([a])
            if max_questions is not None and len(prompts) >= max_questions:
                return images, prompts, labels
    return images, prompts, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FastV OCR-VQA Ablation Evaluation")
    parser.add_argument(
        "--model-path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HF model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--use-fast-v",
        type=str,
        default="True",
        help="Whether to use FastV ('True' or 'False').",
    )
    parser.add_argument(
        "--fast-v-mode",
        type=str,
        default="mask",
        choices=["mask", "inplace"],
        help="FastV mode: 'mask' (Paper Fig 7 ablation) or 'inplace' (real speedup).",
    )
    parser.add_argument("--fast-v-sys-length", type=int, default=None, help="Optional image start index override.")
    parser.add_argument("--fast-v-image-token-length", type=int, default=None, help="Optional image length override.")
    parser.add_argument("--fast-v-attention-rank", type=int, default=None, help="Number of visual tokens to keep.")
    parser.add_argument("--fast-v-r", type=float, default=None, help="Pruning ratio R in [0, 1].")
    parser.add_argument("--fast-v-agg-layer", type=int, default=2, help="Pruning layer K (supports K=0 for random).")
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
        choices=["4bit", "8bit", "bf16", "fp16", "none"],
    )
    parser.add_argument("--max-examples", type=int, default=500, help="Number of OCR-VQA images (default: 500).")
    parser.add_argument("--max-questions", type=int, default=None, help="Optional cap on total QA pairs.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--output-path", type=str, required=True, help="Output JSON file path.")
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    pargs = parse_args()
    use_fastv = parse_bool_str(pargs.use_fast_v)

    print(f"Loading OCR-VQA samples (max_examples={pargs.max_examples}, max_questions={pargs.max_questions})...")
    images, prompts, labels = load_ocrvqa_samples(
        max_examples=pargs.max_examples,
        max_questions=pargs.max_questions,
    )
    print(f"Loaded {len(prompts)} OCR-VQA QA pairs.")

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

    if use_fastv:
        fastv_cfg = FastVConfig(
            use_fastv=True,
            fastv_k=pargs.fast_v_agg_layer,
            fastv_r=pargs.fast_v_r if pargs.fast_v_r is not None else 0.5,
            mode=pargs.fast_v_mode,
            scorer=pargs.scorer,
            image_token_start_index=pargs.fast_v_sys_length,
            image_token_length=pargs.fast_v_image_token_length,
            attention_rank=pargs.fast_v_attention_rank,
        )
    else:
        fastv_cfg = FastVConfig(use_fastv=False)

    model, processor, controller = load_vlm_model(
        model_id=model_id,
        handler=handler,
        quantization=pargs.quantization,
        dtype=dtype,
        attn_implementation="eager",
        fastv_config=fastv_cfg,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else None,
    )

    device = next(model.parameters()).device
    vis_ids = controller.adapter.get_visual_token_ids(model)
    image_token_id = int(vis_ids[0]) if vis_ids else 32000

    outputs: list[str] = []
    latencies: list[float] = []
    prompt_lengths: list[int] = []
    image_lengths: list[int] = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for idx, (prompt, image) in enumerate(
        tqdm(zip(prompts, images), total=len(prompts), desc="OCR-VQA")
    ):
        inputs, span = prepare_vlm_inputs(
            processor=processor,
            prompt=prompt,
            image=image,
            device=device,
            dtype=dtype,
            handler=handler,
            include_system_prompt=pargs.include_system_prompt,
            image_token_index=image_token_id,
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        prompt_lengths.append(prompt_len)
        if span is not None:
            image_lengths.append(int(span[1]))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=pargs.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)

        decoded = processor.tokenizer.decode(
            gen.sequences[0, prompt_len:], skip_special_tokens=True
        ).strip()
        outputs.append(decoded)
        if not pargs.quiet and idx < 5:
            print(f"[sample {idx}] Q={prompt!r} -> pred={decoded!r} gt={labels[idx][0]!r}")

    cleaned_refs = [[clean_text(t) for t in ref_list] for ref_list in labels]
    cleaned_preds = [clean_text(p) for p in outputs]
    avg_score, scores = compute_rouge_l(cleaned_refs, cleaned_preds)

    avg_latency = sum(latencies) / max(1, len(latencies))
    peak_mem_gib = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    text_cfg = getattr(model.config, "text_config", model.config)
    img_len = (
        max(1, int(round(sum(image_lengths) / len(image_lengths))))
        if image_lengths
        else (fastv_cfg.image_token_length or 576)
    )
    avg_text_len = max(1, int(round((sum(prompt_lengths) / max(1, len(prompt_lengths))) - img_len)))
    flops = compute_fastv_flops_ratio(
        num_layers=int(getattr(text_cfg, "num_hidden_layers", 32)),
        fastv_k=fastv_cfg.fastv_k,
        fastv_r=fastv_cfg.effective_pruning_ratio(img_len) if fastv_cfg.use_fastv else 0.0,
        n_image_tokens=img_len,
        n_text_tokens=avg_text_len,
        hidden_size=int(getattr(text_cfg, "hidden_size", 4096)),
        intermediate_size=int(getattr(text_cfg, "intermediate_size", 11008)),
        swiglu=(handler.name == "qwen2_vl"),
    )

    out_dir = os.path.dirname(os.path.abspath(pargs.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    result = {
        "model": model_id,
        "handler": handler.name,
        "adapter": controller.adapter.name,
        "quantization": pargs.quantization,
        "num_questions": len(outputs),
        "fastv_config": fastv_cfg.to_dict(),
        "avg": str(avg_score),
        "rouge_l_pct": round(avg_score * 100.0, 2),
        "avg_latency_s": round(avg_latency, 5),
        "peak_gpu_mem_gib": round(peak_mem_gib, 3),
        "flops_ratio_pct": round(flops["flops_ratio_pct"], 2),
        "flops_reduction_pct": round(flops["flops_reduction_pct"], 2),
        "scores": scores,
        "output": outputs,
    }
    with open(pargs.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

    print(
        f"OCR-VQA Done | ROUGE-L={avg_score * 100:.2f}% | avg_latency={avg_latency:.4f}s | "
        f"FLOPs={flops['flops_ratio_pct']:.1f}% -> {pargs.output_path}"
    )


if __name__ == "__main__":
    main()
