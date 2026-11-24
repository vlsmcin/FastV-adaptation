# Implementation Plan: Adapting FastV (LLaVA) to Modern `transformers`

## 1. Objective & Scope

This plan describes how to port **FastV** for **LLaVA** from the repository's legacy vendored stack (`src/LLaVA` + `transformers==4.31.0` / `4.39.0.dev0`, Python 3.10) onto **modern Hugging Face `transformers` (`>= 4.45`, Python 3.11/3.12, Google Colab compatible)** without maintaining a vendored fork of `transformers` or installing `src/LLaVA`.

### Design Goals
1. **Zero vendored forks:** Run directly on top of standard `pip install transformers` and Colab's preinstalled PyTorch.
2. **Modular architecture ready for future research:** Separate the **token-importance scoring algorithm** from the **model-specific forward patch**, so that:
   - Adding new pruning metrics later (e.g., head-wise variance, multi-head aggregation, entropy) only requires adding a scorer function.
   - Extending to **Qwen2-VL** next only requires adding a new model adapter without changing the scoring or evaluation pipeline.
3. **Full feature parity with the repo:** Support both **token dropping (in-place pruning)** with/without KV cache and **attention masking** (for paper ablation studies), plus modernized equivalents of the demo, attention visualization, A-OKVQA, and OCRVQA scripts.

---

## 2. Target Package & File Structure

Instead of editing files inside `src/transformers/` or `src/LLaVA/`, create a clean, standalone package (e.g., `fastv_modern/`) alongside modernized scripts:

```text
fastv_modern/
├── __init__.py
├── config.py                  # FastVConfig dataclass (K, R, mode, scorer, etc.)
├── scoring.py                 # Pluggable visual token scoring functions (vanilla mean + future hooks)
├── adapters/
│   ├── __init__.py
│   └── llava_llama.py         # Modern transformers LlamaModel.forward patch + LLaVA binder
└── scripts/
    ├── demo_llava_hf.py       # Modernized version of demo-hf.py
    ├── eval_aokvqa.py         # Modernized version of src/FastV/inference/eval/inference_aokvqa.py
    ├── eval_ocrvqa.py         # Modernized version of src/FastV/inference/eval/inference_ocrvqa.py
    └── plot_attention.py      # Modernized version of src/FastV/inference/plot_inefficient_attention.py
tests/
└── test_fastv_llava.py        # Verification suite (R=0 parity, KV cache, masking vs inplace, shapes)
requirements-modern.txt        # Clean Colab-compatible dependencies
```

---

## 3. Core Technical Plan: The 5 Modern-`transformers` Adaptations

All FastV pruning logic resides in a single patched `LlamaModel.forward` method (in `fastv_modern/adapters/llava_llama.py`), attached dynamically to a loaded `LlavaForConditionalGeneration` model via `types.MethodType`.

### Change 1: Slice or Recompute Precomputed RoPE `position_embeddings = (cos, sin)`
- **Legacy (`transformers 4.39`):** Each `LlamaAttention` layer called `self.rotary_emb(value_states, position_ids)` internally.
- **Modern (`transformers >= 4.44`):** `LlamaModel.forward` computes `position_embeddings = self.rotary_emb(hidden_states, position_ids)` **once before the decoder layer loop** and passes `(cos, sin)` into every `decoder_layer`.
- **Implementation:**
  - Right after computing `keep_idx` and pruning `hidden_states = hidden_states[:, keep_idx, :]` and `position_ids = position_ids[:, keep_idx]` at layer `K`:
    ```python
    cos, sin = position_embeddings
    position_embeddings = (cos[:, keep_idx, :], sin[:, keep_idx, :])
    ```
  - This preserves the exact original RoPE positional encodings of the surviving visual and text tokens while preventing shape-mismatch crashes in `apply_rotary_pos_emb`.
  - *(Note: Verify tensor layout of `cos`/`sin` — in standard `LlamaRotaryEmbedding`, shape is `(batch, seq_len, head_dim)`, so axis `1` is the sequence axis.)*

### Change 2: Correctly Handle `causal_mask` and `cache_position` (Prefill + Decode)
- **Prefill step (`seq_len > 1`):**
  - After pruning at layer `K`, `hidden_states` shrinks from `seq_len` to `new_seq_len = keep_idx.shape[0]`.
  - Update `cache_position = cache_position[:new_seq_len]`.
  - Update `causal_mask`:
    - In **in-place token-drop mode** (`mode="inplace"`): if `causal_mask` is not `None`, slice it to `causal_mask[:, :, :new_seq_len, :new_seq_len]` (for unpadded single-sample inference) or rebuild the 4D causal mask for the pruned length.
    - In **attention-masking ablation mode** (`mode="mask"`): keep `hidden_states`, `position_ids`, and `position_embeddings` unpruned, and instead set the dropped image token columns in the 4D `causal_mask` to `-inf` (or `torch.finfo(dtype).min`) for all layers `>= K`.
- **Decode steps (`seq_len == 1`, `use_cache=True`):**
  - Because layers `0 .. K-1` store **unpruned** KV caches (`L_full + step`) while layers `K .. N-1` store **pruned** KV caches (`L_pruned + step`), modern `LlamaModel.forward` builds `causal_mask` once using layer 0's KV length (`past_key_values.get_seq_length()`).
  - Passing that unpruned-length 4D mask to layers `>= K` causes a shape mismatch in `attn_weights + causal_mask`.
  - **Fix:** Inside the layer loop during decode (`seq_len == 1`):
    - For single-batch decoding (`batch_size == 1`) with no padding, pass `attention_mask=None` to layers `>= K` (a single query token attends to all previously cached keys causally by construction).
    - For batched/padded decoding, slice the last dimension of `causal_mask` per layer to match `past_key_values.get_seq_length(layer_idx) + 1` (or maintain two masks: `pre_k_mask` and `post_k_mask`).
  - Keep `position_ids` at the logical (unpruned) position during decode so RoPE remains aligned with the original positions retained at prefill.

### Change 3: Modern `DynamicCache` Compatibility (In-Place Mutation)
- **Legacy (`transformers 4.39`):** Used `DynamicCache.from_legacy_cache(...)`, read `next_decoder_cache = layer_outputs[2 if output_attentions else 1]`, and converted back via `.to_legacy_cache()`.
- **Modern (`transformers >= 4.45`):**
  - `past_key_values` is a `DynamicCache` object mutated **in-place** inside `LlamaAttention.forward` (`past_key_value.update(key_states, value_states, self.layer_idx, ...)`).
  - `DynamicCache` natively stores per-layer key/value tensors with **different sequence lengths** across layers (`layer_idx < K` vs `layer_idx >= K`), so static KV-cache pruning works cleanly out of the box once `causal_mask` is handled.
- **Implementation:**
  - Follow the modern `LlamaModel.forward` cache lifecycle (`if use_cache and past_key_values is None: past_key_values = DynamicCache()`).
  - Do **not** call deprecated `from_legacy_cache` / `to_legacy_cache` or try to index `next_decoder_cache` out of `layer_outputs`.

### Change 4: Attention Implementation & Version-Safe Layer Return Handling
- **Eager attention requirement:**
  - Load `LlavaForConditionalGeneration.from_pretrained(..., attn_implementation="eager")` (SDPA and FlashAttention-2 do not materialize attention weights and return `None` when `output_attentions=True`).
  - *(Future optimization note: Once correctness is validated, layer `K - 1` can optionally compute attention weights manually via `Q @ K^T` so that all other layers can run with fast `sdpa` / `flash_attention_2`.)*
- **Capturing layer `K - 1` attention safely across `transformers` versions:**
  - In `transformers` 4.45–4.53, `decoder_layer(..., output_attentions=True)` returns `(hidden_states, self_attn_weights)`.
  - In `transformers >= 4.54`, `LlamaDecoderLayer.forward` was simplified to return a single `hidden_states` tensor directly (moving attention-weight capture into hooks / `output_attentions` wrappers).
  - **Implementation strategy:**
    1. Only enable attention capture at `layer_idx == K - 1` (and only during prefill when `use_cache=True`, or every forward pass when `use_cache=False`) to avoid storing 32 layers of `O(N^2)` attention matrices.
    2. Handle both return conventions cleanly:
       - Check `isinstance(layer_outputs, tuple)` to extract `hidden_states = layer_outputs[0]` and `attn_weights = layer_outputs[1] if len(layer_outputs) > 1 else None`.
       - As a fallback for `transformers >= 4.54` (where `decoder_layer` returns a tensor, not a tuple), register a lightweight forward hook on `self.layers[K - 1].self_attn` (which still returns `(attn_output, attn_weights)`!) to capture `attn_weights` reliably regardless of `LlamaDecoderLayer`'s return signature!
    - *Using a forward hook on `self.layers[K - 1].self_attn` is actually version-resilient across 4.45 through 5.x.*

### Change 5: Zero-Fork Attachment to `LlavaForConditionalGeneration` & Dynamic Visual Token Detection
- **No edits to `modeling_llava.py` or `configuration_llava.py`:**
  - Write a helper `apply_fastv(model, fastv_config)` that locates the underlying `LlamaModel` across `transformers` versions:
    - `transformers >= 4.52`: `model.model.language_model`
    - `transformers <= 4.51`: `model.language_model.model`
    - (Detect dynamically via `isinstance(..., LlamaModel)` rather than hardcoding one attribute path.)
  - Attach `fastv_config` to that `LlamaModel` instance, save the original `forward` (for clean `remove_fastv(model)` unpatching), and bind the FastV `forward` via `types.MethodType`.
- **Dynamic visual token span detection (replaces hardcoded `start=5, len=576`):**
  - Instead of hardcoding `image_token_start_index=5` and `image_token_length=576` (which breaks whenever the system prompt or chat template changes), register a `pre_forward` hook on `LlavaForConditionalGeneration` (or compute from `input_ids` in a wrapper) that inspects:
    ```python
    vis_mask = (input_ids[0] == model.config.image_token_index)  # bool mask of shape [seq_len]
    ```
  - Store the active `vis_mask` (or `vis_indices = vis_mask.nonzero(as_tuple=True)[0]`) on `llama_model._fastv_vis_indices` during the prefill pass (`input_ids.shape[1] > 1`).
  - This automatically works with any LLaVA prompt format **and** directly prepares the interface for Qwen2-VL's variable-length visual tokens!

---

## 4. Module-by-Module Design Details

### 4.1 `fastv_modern/config.py`
Define a clean `FastVConfig` dataclass:
- `enabled: bool = True`
- `agg_layer: int = 2` (`K`: prune right before layer `K`, using attention from layer `K - 1`; must satisfy `1 <= K < num_hidden_layers`)
- `prune_ratio: float = 0.5` (`R`: fraction of visual tokens to discard; keep `round(N_vis * (1 - R))`, or allow explicit `attention_rank` override to match legacy scripts)
- `mode: Literal["inplace", "mask"] = "inplace"` (`inplace` = true token dropping for speedup; `mask` = attention masking ablation matching `src/transformers` default)
- `scorer: str | Callable = "fastv_mean"` (name or callable for visual token importance scoring)
- `scorer_kwargs: dict = field(default_factory=dict)` (extra hyperparameters for future scoring experiments, e.g., `alpha` for `mean + alpha * var`)

### 4.2 `fastv_modern/scoring.py` (Extension Point for Future Research)
Define a uniform scorer signature:
```python
def score_visual_tokens(
    attn_weights: torch.Tensor,   # shape: (batch, num_heads, q_len, kv_len)
    vis_indices: torch.Tensor,    # shape: (num_vis_tokens,)
    **kwargs,
) -> torch.Tensor:                # returns importance scores of shape: (num_vis_tokens,)
```
- **Phase 1 (LLaVA baseline):** Implement `fastv_mean`:
  - Average across heads (`dim=1`), take the last query token (`q_idx = -1`), index `vis_indices`.
  - Exact match to the original paper's ranking formula in [modeling_llama.py](file:///home/lucas/Documents/FastV-adaptation/src/FastV/llava-hf/transformers/src/transformers/models/llama/modeling_llama.py#L1042-L1048).
- **Phase 2 (Ready for your upcoming experiments):** Registry dict `SCORERS = {"fastv_mean": ...}` where you can later plug in head-wise variance (`attn.var(dim=1)`), `mean + alpha * std`, max-head, entropy-weighted head aggregation, or multi-query aggregation without changing a single line in the model adapter.

### 4.3 `fastv_modern/adapters/llava_llama.py`
Contains:
1. `get_llama_backbone(model) -> LlamaModel`: version-safe locator for the `LlamaModel` inside `LlavaForConditionalGeneration`.
2. `register_vis_token_tracker(model)`: pre-forward hook on `LlavaForConditionalGeneration` (or `LlavaModel`) that records `vis_indices = (input_ids[0] == image_token_index).nonzero(as_tuple=True)[0]` whenever `input_ids` contains image tokens.
3. `_make_fastv_forward(...)`: wraps the modern `LlamaModel.forward` layer loop with:
   - Layer `K - 1` attention capture (via `output_attentions=True` / `self_attn` hook).
   - Layer `K` scoring via `scoring.py` and pruning (`inplace` or `mask`), applying Changes 1–4.
   - Clean per-layer mask handling during KV-cached decoding (`seq_len == 1`).
4. `apply_fastv(model, config)` and `remove_fastv(model)`: clean attach/detach lifecycle, plus optional telemetry (`model._fastv_stats`: original tokens, pruned tokens, kept visual indices) for debugging and analysis.

### 4.4 Adapting the Scripts (`fastv_modern/scripts/`)
Keep all benchmark logic (dataset loading, text normalization, accuracy/ROUGE-L scoring, heatmap plotting) identical to the original repo while replacing only the legacy `llava.*` loading/generation block (~20 lines per script):

| Original Script | Modernized Script | Key Changes |
|---|---|---|
| [demo-hf.py](file:///home/lucas/Documents/FastV-adaptation/demo-hf.py) | `demo_llava_hf.py` | Load `llava-hf/llava-1.5-7b-hf` (or `13b-hf`) with standard `LlavaForConditionalGeneration` + `AutoProcessor`, call `apply_fastv(model, cfg)`, measure TTFT + tokens/sec + peak VRAM with both `use_cache=True` and `use_cache=False`. |
| [inference_aokvqa.py](file:///home/lucas/Documents/FastV-adaptation/src/FastV/inference/eval/inference_aokvqa.py) | `eval_aokvqa.py` | Keep `data/aokvqa/validation` loader, multiple-choice `TEMPLATE`, and `compute_acc()`. Replace `llava.model.builder.load_pretrained_model` + `conv_templates` with `LlavaForConditionalGeneration` + `AutoProcessor` chat template (`USER: <image>\n...\nASSISTANT:`). Support optional 4-bit quantization via modern `BitsAndBytesConfig`. |
| [inference_ocrvqa.py](file:///home/lucas/Documents/FastV-adaptation/src/FastV/inference/eval/inference_ocrvqa.py) | `eval_ocrvqa.py` | Keep `clean_text()` and `pycocoevalcap` ROUGE scorer. Replace `llava` model calls with `LlavaForConditionalGeneration` + `apply_fastv(model, cfg)`. Support both `--mode mask` (paper ablation) and `--mode inplace`. |
| [plot_inefficient_attention.py](file:///home/lucas/Documents/FastV-adaptation/src/FastV/inference/plot_inefficient_attention.py) | `plot_attention.py` | Keep `visualize_attention()` seaborn/LogNorm plotting code. Load with `LlavaForConditionalGeneration(..., attn_implementation="eager")` and run `model.generate(..., output_attentions=True, return_dict_in_generate=True)`. |

### 4.5 Modern Dependencies (`requirements-modern.txt`)
Clean, Colab-compatible dependency list (no pinned `torch==2.0.1`, no `src/LLaVA`, no vendored `transformers`):
- `transformers>=4.45.0`
- `accelerate>=0.34.0`
- `bitsandbytes` (optional, for 4-bit/8-bit quantization on Colab T4/L4)
- `pillow`, `requests`, `tqdm`
- `datasets`, `seaborn`, `matplotlib`, `pycocoevalcap` (for evaluation & attention visualization)
- *(Later for Qwen2-VL: `qwen-vl-utils`)*

---

## 5. Step-by-Step Implementation Order

1. **Step 1 — Inspect target `transformers` runtime:**
   - Confirm exact `transformers` version on Colab (and local `.venv` if used) and inspect `LlamaModel.forward`, `LlamaDecoderLayer.forward`, and `LlavaForConditionalGeneration` attribute structure (`model.language_model.model` vs `model.model.language_model`).
2. **Step 2 — Implement `config.py` and `scoring.py`:**
   - Build `FastVConfig` and the `fastv_mean` scorer with unit tests on synthetic attention tensors.
3. **Step 3 — Implement `adapters/llava_llama.py`:**
   - Build `get_llama_backbone`, visual-token index tracking, and the patched `LlamaModel.forward` incorporating all 5 changes (RoPE `position_embeddings`, `causal_mask` prefill & decode, `DynamicCache`, version-safe layer `K-1` attention capture, and `inplace`/`mask` modes).
4. **Step 4 — Run correctness & parity tests (`tests/test_fastv_llava.py`):**
   - Validate before touching any eval scripts (see Section 6).
5. **Step 5 — Port `demo_llava_hf.py` and `eval_aokvqa.py`:**
   - Benchmark on `data/aokvqa/validation` (already present in this repo) to verify accuracy, latency, and VRAM against vanilla decoding.
6. **Step 6 — Port `eval_ocrvqa.py` and `plot_attention.py`:**
   - Complete parity with the repository's original evaluation & visualization workflows.
7. **Step 7 — Prepare handoff to Qwen2-VL & Custom Scorers (Next Phase):**
   - Add `fastv_modern/adapters/qwen2_vl.py` (handling 3D M-RoPE `(3, B, S)` and `config.image_token_id`) and new scorers in `fastv_modern/scoring.py` (head variance, custom multi-head aggregation).

---

## 6. Verification & Testing Plan

Before relying on benchmark numbers, verify the implementation with the following checks:

1. **Exact `R = 0.0` (Zero-Pruning) Parity Test:**
   - Run `LlavaForConditionalGeneration` **without** FastV vs. **with** FastV (`R = 0.0`, keeping 100% of visual tokens) for both `use_cache=True` and `use_cache=False`.
   - Assert generated tokens are **100% identical** and logits match within floating-point tolerance (`atol=1e-4` in fp16).
   - This catches any subtle bugs in RoPE slicing, `causal_mask`, or `DynamicCache`.
2. **Sequence Length & KV-Cache Shape Assertions (`R = 0.5, K = 2`):**
   - Verify that after prefill:
     - `past_key_values.get_seq_length(layer_idx)` equals `L_full` for `layer_idx < K` and `L_full - floor(0.5 * 576)` for `layer_idx >= K`.
     - After generating `T` tokens, cache lengths are `L_full + T - 1` (for `< K`) and `L_pruned + T - 1` (for `>= K`).
3. **Masking vs. In-Place Equivalence Check (`use_cache=False`, 1st generated token):**
   - For the first generated token (prefill), verify that `mode="inplace"` and `mode="mask"` produce matching logits (within fp16 numerical tolerance), confirming that in-place dropping does not corrupt positional embeddings.
4. **End-to-End Sanity & Benchmark Check on Colab:**
   - Run `demo_llava_hf.py` with `llava-hf/llava-1.5-7b-hf` (4-bit on T4 or fp16 on L4/A100) and verify coherent descriptions at `K=2, R=0.5` and `K=3, R=0.75`.
   - Run `eval_aokvqa.py` on a subset (e.g., 100 samples) then the full validation split in `data/aokvqa/validation`, comparing Baseline vs. FastV (`K=2, R=0.25 / 0.50 / 0.75`) on accuracy, latency per token, and peak CUDA memory.

---

## 7. Known Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `transformers` minor version drift between local env and Colab (`4.45`–`4.55+`) | Different `LlamaModel.forward` internals or `LlamaDecoderLayer` return type | Use `self_attn` forward hook for layer `K-1` attention capture; detect `LlamaModel` path dynamically; check `transformers.__version__` / `inspect.signature` for mask helper (`_update_causal_mask` vs `create_causal_mask`). |
| `attn_implementation="eager"` slows down layers `!= K-1` | Latency gains partially masked by eager attention on 31 other layers | First validate accuracy with full `"eager"`, then optionally optimize by computing layer `K-1` attention scores via a lightweight `Q @ K^T` hook while keeping `sdpa` for all layers. |
| Colab T4 (15 GB VRAM) memory limit for 7B/13B fp16 | OOM during inference or attention visualization | Support `BitsAndBytesConfig(load_in_4bit=True)` in all scripts; only store attention weights for layer `K-1` during FastV inference. |
| Prompt format differences (`llava-v1.5` vs `llava-hf`) | Slight score delta vs original paper | Match `llava-hf`'s expected `USER: <image>\n{question}\nASSISTANT:` format consistently across baseline and FastV runs so all comparisons are apples-to-apples. |
