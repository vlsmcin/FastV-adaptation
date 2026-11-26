#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${MODEL_PATH:-llava-hf/llava-1.5-7b-hf}"
QUANTIZATION="${FASTV_QUANTIZATION:-4bit}"
OUTPUT_PATH="${OUTPUT_PATH:-ocrvqa_eval_fastv}"
mkdir -p "${OUTPUT_PATH}"

rank_list=(72 144 288 432) # rank = (1 - R) * 576, i.e., R = (87.5%, 75%, 50%, 25%)
Ks=(0 1 2 3 5 10)

for rank in "${rank_list[@]}"; do
    for k in "${Ks[@]}"; do
        python ./src/FastV/inference/eval/inference_ocrvqa.py \
            --model-path "${MODEL_PATH}" \
            --quantization "${QUANTIZATION}" \
            --use-fast-v True \
            --fast-v-mode mask \
            --fast-v-attention-rank "${rank}" \
            --fast-v-agg-layer "${k}" \
            --output-path "${OUTPUT_PATH}/ocrvqa_7b_FASTV_mask_${rank}_${k}.json"
    done
done
