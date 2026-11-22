#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-llava-hf/llava-1.5-7b-hf}"
QUANTIZATION="${FASTV_QUANTIZATION:-4bit}"

python ./src/FastV/inference/plot_inefficient_attention.py \
    --model-path "${MODEL_PATH}" \
    --quantization "${QUANTIZATION}" \
    --image-path "./src/LLaVA/images/llava_logo.png" \
    --prompt "Describe the image in details." \
    --output-path "./output_example"
