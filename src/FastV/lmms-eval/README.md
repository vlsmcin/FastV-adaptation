# Using FastV with `lmms-eval` (Modern Transformers >=4.45 / 5.x)

With the modernized hook-based `FastVController` (`src/FastV/fastv_hf.py`), you **no longer need to patch `modeling_llama.py` rotary embeddings or downgrade `transformers`**.

## Quick Setup

1. Install modern dependencies and `lmms-eval`:
```bash
pip install -r requirements.txt
pip install lmms-eval
```

2. Attach FastV to any HuggingFace LLaVA model inside `lmms-eval` (or your evaluation wrapper):
```python
from src.FastV.lmms-eval.fastv_kvcache import enable_fastv_for_lmms_eval  # or add src/FastV to PYTHONPATH
# After loading LlavaForConditionalGeneration with attn_implementation="eager":
enable_fastv_for_lmms_eval(model, k=3, r=0.5, mode="inplace")
```

3. Run `lmms-eval` benchmarks (e.g., `mme`, `gqa`, `mmmu_val`, `seedbench`, `nocaps`, `flickr30k`, `coco2017_cap_val`):
```bash
accelerate launch --num_processes=1 -m lmms_eval \
    --model llava_hf \
    --model_args pretrained=llava-hf/llava-1.5-7b-hf,attn_implementation=eager \
    --tasks mme,gqa,mmmu_val \
    --batch_size 1 \
    --output_path ./logs/
```

See `executor.ipynb` Section 6 for a ready-to-run Colab evaluation pipeline covering MME (Perception & Cognition), GQA, MMMU, and optional `lmms-eval` execution.

## Acknowledgements

Thanks to [Zhihang Lin](https://github.com/Stardust1956) from Xiamen University for the original KV-cache & `lmms-eval` integration.
