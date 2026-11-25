# FastV Architecture Refactoring TODO

Tracks the planned follow-up work to migrate callers to `BaseVLMHandler` and move from the current pragmatic re-export layout (`fastv_hf.py` re-exporting `.handlers`) to a strictly acyclic package facade via `FastV/__init__.py`.

---

## 1. Migrate Callers to Model-Agnostic `BaseVLMHandler` & `FastV` Facade

- [ ] **Update `executor.ipynb`**:
  - Switch imports from `from FastV.fastv_hf import ...` to `from FastV import ...` (`FastVConfig`, `load_vlm_model`, `prepare_vlm_inputs`, `resolve_handler`, etc.).
  - Replace `load_llava_model` / `prepare_llava_inputs` calls with `handler = resolve_handler(MODEL_ID)` → `handler.load_model(...)` and `handler.prepare_inputs(...)`.
  - Remove LLaVA-specific hardcoded assumptions: `image_token_index=32000` (read from `controller.adapter.get_visual_token_ids(model)`), fixed `576` visual token count (use per-sample detected `span[1]`), and fixed A-OKVQA `attention_rank` (`432, 288, 144, 72` → ratio-based `fastv_r=0.25, 0.50, 0.75, 0.875`).
  - Fix Cell 5 test command to `python -m unittest discover -s tests -p "test_*.py" -v`.
- [ ] **Update CLI scripts** to import from `FastV` and use `resolve_handler(model_id)` + dynamic visual token lengths:
  - `demo-hf.py`
  - `src/FastV/inference/plot_inefficient_attention.py`
  - `src/FastV/inference/eval/inference_aokvqa.py`
  - `src/FastV/inference/eval/inference_ocrvqa.py`

---

## 2. Adopt Strictly Acyclic Module Graph (Textbook Ideal)

Once all callers import from the top-level package (`from FastV import ...`) instead of `FastV.fastv_hf`:

- [ ] **Remove `.handlers` import from `src/FastV/fastv_hf.py`**:
  - Drop `from .handlers import (...)` at the top of `fastv_hf.py` so `fastv_hf.py` depends only on `.adapters` and `.scoring` (zero reference to `.handlers`).
  - Move or deprecate the legacy LLaVA compatibility wrappers (`load_llava_model`, `format_llava_prompt`, `prepare_llava_inputs`, `resolve_hf_model_id`) out of `fastv_hf.py` (or expose them exclusively from `src/FastV/__init__.py` / `src/FastV/handlers.py`).
- [ ] **Restore top-level imports in `src/FastV/handlers.py`**:
  - Replace the local `from .fastv_hf import attach_fastv` / `detect_image_token_span` calls inside `load_model` and `prepare_inputs` with clean top-level imports from `.fastv_hf`.
  - Resulting strict DAG:
    - `adapters.py`, `scoring.py` (leaf modules, zero internal deps)
    - `fastv_hf.py` → `adapters.py`, `scoring.py`
    - `handlers.py` → `fastv_hf.py`
    - `__init__.py` → single public facade re-exporting all four modules
- [ ] **Update `tests/test_fastv_hf.py` and `src/FastV/fastv_hf.md`** to reflect the new facade imports and verify import-order independence.

---

## 3. Real-Checkpoint Qwen2.5-VL GPU Verification

- [ ] Run an `R = 0.0` (`force_slice_at_r0=True`) parity check on `Qwen/Qwen2.5-VL-7B-Instruct` on Colab GPU to verify generated tokens and logits match vanilla decoding exactly (covering `DynamicCache`, 3D M-RoPE `position_ids` / `position_embeddings`, and `rope_deltas`).
- [ ] Run end-to-end A-OKVQA / Demo benchmarks on `Qwen/Qwen2.5-VL-7B-Instruct` at `K=2, R=0.5` and `K=3, R=0.75`.
