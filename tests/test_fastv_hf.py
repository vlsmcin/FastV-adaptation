"""Unit tests for the modular hook-based FastV implementation (LLaVA & Qwen2-VL interface).

Runs pure-Python tests everywhere, and runs full PyTorch hook, M-RoPE,
DynamicCache, R=0 parity, and inplace-vs-mask equivalence tests whenever
PyTorch is installed.
"""

from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from FastV.adapters import LlavaLlamaAdapter, Qwen2VLAdapter, resolve_adapter
from FastV.fastv_hf import (
    FastVConfig,
    compute_fastv_flops_ratio,
    compute_rouge_l,
    detect_image_token_span,
    detect_visual_token_indices,
    format_llava_prompt,
    resolve_hf_model_id,
)
from FastV.scoring import SCORERS, register_scorer, resolve_scorer

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestFastVPurePython(unittest.TestCase):
    def test_flops_ratio_matches_paper(self) -> None:
        res = compute_fastv_flops_ratio(
            num_layers=32,
            fastv_k=2,
            fastv_r=0.5,
            n_image_tokens=576,
            n_text_tokens=60,
            hidden_size=4096,
            intermediate_size=11008,
        )
        self.assertGreater(res["flops_reduction_pct"], 40.0)
        self.assertLess(res["flops_reduction_pct"], 50.0)
        base = compute_fastv_flops_ratio(num_layers=32, fastv_k=2, fastv_r=0.0)
        self.assertAlmostEqual(base["flops_reduction_pct"], 0.0, places=6)

    def test_detect_visual_token_indices_single_and_multi_image(self) -> None:
        # Expanded single-image tokens
        ids = [[1, 3148, 1001, 29901, 29871] + [32000] * 16 + [13, 500, 501]]
        self.assertEqual(detect_image_token_span(ids, image_token_index=32000), (5, 16))
        self.assertEqual(detect_visual_token_indices(ids, 32000), list(range(5, 21)))

        # Multi-image interleaved visual tokens (e.g., MMMU / Qwen2-VL)
        multi = [[1, 10, 151655, 151655, 20, 21, 151655, 151655, 151655, 30]]
        self.assertEqual(
            detect_visual_token_indices(multi, visual_token_ids=(151655, 151656), default_image_length=0),
            [2, 3, 6, 7, 8],
        )

        # Single placeholder token (legacy transformers)
        ids_single = [[1, 10, 20, 32000, 30]]
        self.assertEqual(
            detect_image_token_span(ids_single, image_token_index=32000, default_image_length=576),
            (3, 576),
        )
        self.assertIsNone(detect_visual_token_indices([[1, 2, 3]], 32000))

    def test_config_legacy_aliases_and_scorer_validation(self) -> None:
        cfg = FastVConfig.from_any(
            {
                "use_fast_v": True,
                "fast_v_agg_layer": 3,
                "fast_v_sys_length": 36,
                "fast_v_image_token_length": 576,
                "fast_v_attention_rank": 144,
                "fast_v_inplace": False,
                "scorer": "fastv_mean_std",
                "scorer_kwargs": {"alpha": 0.25},
            }
        )
        self.assertTrue(cfg.use_fastv)
        self.assertEqual(cfg.fastv_k, 3)
        self.assertEqual(cfg.image_token_start_index, 36)
        self.assertEqual(cfg.image_token_length, 576)
        self.assertEqual(cfg.resolve_keep_count(576), 144)
        self.assertEqual(cfg.mode, "mask")
        self.assertEqual(cfg.scorer, "fastv_mean_std")
        self.assertAlmostEqual(cfg.effective_pruning_ratio(576), 0.75, places=6)

        with self.assertRaises(KeyError):
            FastVConfig(scorer="nonexistent_scorer")

    def test_scorer_registry(self) -> None:
        self.assertIn("fastv_mean", SCORERS)
        self.assertIn("fastv_max_head", SCORERS)
        self.assertIn("fastv_mean_std", SCORERS)
        custom = lambda attn, vis, **kw: vis
        register_scorer("test_dummy_scorer", custom)
        self.assertIs(resolve_scorer("test_dummy_scorer"), custom)
        SCORERS.pop("test_dummy_scorer", None)

    def test_adapter_auto_resolution(self) -> None:
        class _FakeCfg:
            def __init__(self, model_type):
                self.model_type = model_type

        class Qwen2_5_VLForConditionalGeneration:
            def __init__(self):
                self.config = _FakeCfg("qwen2_5_vl")

        class LlavaForConditionalGeneration:
            def __init__(self):
                self.config = _FakeCfg("llava")

        self.assertIsInstance(resolve_adapter(Qwen2_5_VLForConditionalGeneration()), Qwen2VLAdapter)
        self.assertIsInstance(resolve_adapter(LlavaForConditionalGeneration()), LlavaLlamaAdapter)

    def test_rouge_l(self) -> None:
        avg, scores = compute_rouge_l(
            [["the cat sat on the mat"], ["hello world"]],
            ["the cat sat on the mat", "goodbye"],
        )
        self.assertAlmostEqual(scores[0], 1.0, places=6)
        self.assertAlmostEqual(scores[1], 0.0, places=6)
        self.assertAlmostEqual(avg, 0.5, places=6)

    def test_resolve_hf_model_id_and_prompt(self) -> None:
        self.assertEqual(resolve_hf_model_id("./models/llava-v1.5-7b"), "llava-hf/llava-1.5-7b-hf")
        self.assertEqual(resolve_hf_model_id("/cpfs01/models/llava-v1.5-13b"), "llava-hf/llava-1.5-13b-hf")
        self.assertEqual(resolve_hf_model_id("llava-hf/llava-1.5-7b-hf"), "llava-hf/llava-1.5-7b-hf")
        formatted = format_llava_prompt("Describe this.", include_system_prompt=False)
        self.assertIn("USER: <image>\nDescribe this.\nASSISTANT:", formatted)


if HAS_TORCH:
    from FastV.fastv_hf import attach_fastv, compute_layer_attention_efficiency, remove_fastv

    class _MockDynamicCache:
        """Minimal DynamicCache tracking per-layer sequence lengths."""

        def __init__(self):
            self._lengths: dict[int, int] = {}

        def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
            self._lengths[layer_idx] = self._lengths.get(layer_idx, 0) + int(key_states.shape[-2])
            return key_states, value_states

        def get_seq_length(self, layer_idx: int = 0) -> int:
            return self._lengths.get(layer_idx, 0)

    class _DummyConfig:
        def __init__(
            self,
            num_layers: int = 4,
            model_type: str = "llava",
            image_token_index: int = 32000,
            image_token_id: int | None = None,
            video_token_id: int | None = None,
        ):
            self.num_hidden_layers = num_layers
            self.model_type = model_type
            self.image_token_index = image_token_index
            if image_token_id is not None:
                self.image_token_id = image_token_id
            if video_token_id is not None:
                self.video_token_id = video_token_id
            self._attn_implementation = "eager"

    class _DummySelfAttn(nn.Module):
        def __init__(self, layer_idx: int, dim: int, num_heads: int, mrope: bool = False):
            super().__init__()
            self.layer_idx = layer_idx
            self.dim = dim
            self.num_heads = num_heads
            self.mrope = mrope
            self.config = _DummyConfig()
            self.seen_q_lens: list[int] = []
            self.seen_mask_shapes: list[tuple[int, ...] | None] = []
            self.seen_pos_emb_shapes: list[tuple[int, ...] | None] = []

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            past_key_values: _MockDynamicCache | None = None,
            output_attentions: bool = False,
            **kwargs,
        ):
            bsz, q_len, dim = hidden_states.shape
            self.seen_q_lens.append(q_len)
            self.seen_mask_shapes.append(tuple(attention_mask.shape) if attention_mask is not None else None)
            if position_embeddings is not None:
                cos = position_embeddings[0]
                self.seen_pos_emb_shapes.append(tuple(cos.shape))
                assert cos.shape[-2] == q_len
            if position_ids is not None:
                assert position_ids.shape[-1] == q_len

            if past_key_values is not None:
                past_key_values.update(
                    torch.zeros(bsz, self.num_heads, q_len, dim // self.num_heads, device=hidden_states.device),
                    torch.zeros(bsz, self.num_heads, q_len, dim // self.num_heads, device=hidden_states.device),
                    self.layer_idx,
                )

            # Real dot-product attention so inplace vs mask equivalence is mathematically meaningful.
            scores = torch.matmul(hidden_states, hidden_states.transpose(-1, -2)) / (dim ** 0.5)
            scores = scores.unsqueeze(1).expand(bsz, self.num_heads, q_len, q_len)
            if attention_mask is not None and attention_mask.shape[-1] == q_len and attention_mask.shape[-2] == q_len:
                scores = scores + attention_mask
            attn_weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights.mean(dim=1), hidden_states)
            return out, attn_weights

    class _DummyDecoderLayer(nn.Module):
        def __init__(self, layer_idx: int, dim: int, num_heads: int, return_tuple: bool = False, mrope: bool = False):
            super().__init__()
            self.self_attn = _DummySelfAttn(layer_idx, dim, num_heads, mrope=mrope)
            self.return_tuple = return_tuple

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            past_key_values: _MockDynamicCache | None = None,
            cache_position: torch.Tensor | None = None,
            **kwargs,
        ):
            out, attn = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                **kwargs,
            )
            if self.return_tuple:
                return (out, attn)
            return out

    class _DummyBackbone(nn.Module):
        def __init__(
            self,
            num_layers: int = 4,
            dim: int = 16,
            num_heads: int = 2,
            return_tuple: bool = False,
            mrope: bool = False,
        ):
            super().__init__()
            self.config = _DummyConfig(num_layers=num_layers)
            self.mrope = mrope
            self.layers = nn.ModuleList(
                [
                    _DummyDecoderLayer(i, dim, num_heads, return_tuple=return_tuple, mrope=mrope)
                    for i in range(num_layers)
                ]
            )

        def forward(
            self,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: _MockDynamicCache | None = None,
        ):
            bsz, seq_len, dim = inputs_embeds.shape
            device = inputs_embeds.device
            if self.mrope:
                # Qwen2-VL 3D M-RoPE shapes: position_ids (3, B, S), cos/sin (3, B, S, head_dim)
                position_ids = torch.arange(seq_len, device=device).view(1, 1, seq_len).expand(3, bsz, seq_len)
                position_embeddings = (
                    torch.ones((3, bsz, seq_len, dim), device=device),
                    torch.zeros((3, bsz, seq_len, dim), device=device),
                )
            else:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
                position_embeddings = (
                    torch.ones((bsz, seq_len, dim), device=device),
                    torch.zeros((bsz, seq_len, dim), device=device),
                )
            cache_position = torch.arange(seq_len, device=device)
            hidden_states = inputs_embeds
            for layer in self.layers:
                out = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                )
                hidden_states = out[0] if isinstance(out, tuple) else out
            return hidden_states

    class _DummyLlavaWrapper(nn.Module):
        def __init__(self, nested_v5: bool = True):
            super().__init__()
            self.config = _DummyConfig(num_layers=4, model_type="llava", image_token_index=32000)
            backbone = _DummyBackbone(num_layers=4, dim=16, num_heads=2, return_tuple=not nested_v5)
            if nested_v5:
                self.model = nn.Module()
                self.model.language_model = backbone
            else:
                self.language_model = nn.Module()
                self.language_model.model = backbone
            self._backbone_ref = backbone

        def forward(
            self,
            input_ids: torch.Tensor,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: _MockDynamicCache | None = None,
        ):
            return self._backbone_ref(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

    class _DummyQwen2VLWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _DummyConfig(
                num_layers=4,
                model_type="qwen2_5_vl",
                image_token_id=151655,
                video_token_id=151656,
            )
            backbone = _DummyBackbone(num_layers=4, dim=16, num_heads=2, return_tuple=False, mrope=True)
            self.model = nn.Module()
            self.model.language_model = backbone
            self._backbone_ref = backbone

        def forward(
            self,
            input_ids: torch.Tensor,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            past_key_values: _MockDynamicCache | None = None,
        ):
            return self._backbone_ref(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

    class TestFastVTorchHooks(unittest.TestCase):
        def _run_inplace_check(self, nested_v5: bool) -> None:
            model = _DummyLlavaWrapper(nested_v5=nested_v5)
            ctrl = attach_fastv(model, {"use_fastv": True, "fastv_k": 2, "fastv_r": 0.5, "mode": "inplace"})
            cache = _MockDynamicCache()

            input_ids = torch.tensor([[1, 2, 3] + [32000] * 8 + [4, 5, 6]])
            embeds = torch.randn(1, 14, 16)
            causal_mask = torch.zeros(1, 1, 14, 14)

            out = model(input_ids=input_ids, inputs_embeds=embeds, attention_mask=causal_mask, past_key_values=cache)
            self.assertEqual(out.shape, (1, 10, 16))
            self.assertEqual(cache.get_seq_length(0), 14)
            self.assertEqual(cache.get_seq_length(1), 14)
            self.assertEqual(cache.get_seq_length(2), 10)
            self.assertEqual(cache.get_seq_length(3), 10)
            self.assertEqual(ctrl.last_stats["kept_image_tokens"], 4)

            # Decode 2 tokens and verify per-layer DynamicCache lengths (Plan Section 6)
            for step in range(1, 3):
                dec_ids = torch.tensor([[99]])
                dec_embeds = torch.randn(1, 1, 16)
                dec_mask = torch.zeros(1, 1, 1, 14 + step)
                dec_out = model(
                    input_ids=dec_ids,
                    inputs_embeds=dec_embeds,
                    attention_mask=dec_mask,
                    past_key_values=cache,
                )
                self.assertEqual(dec_out.shape, (1, 1, 16))
                self.assertEqual(cache.get_seq_length(0), 14 + step)
                self.assertEqual(cache.get_seq_length(2), 10 + step)
                self.assertEqual(model._backbone_ref.layers[2].self_attn.seen_mask_shapes[-1], (1, 1, 1, 10 + step))

        def test_inplace_transformers_v5_layout(self) -> None:
            self._run_inplace_check(nested_v5=True)

        def test_inplace_transformers_v4_layout(self) -> None:
            self._run_inplace_check(nested_v5=False)

        def test_qwen2_vl_3d_mrope_and_multi_image(self) -> None:
            model = _DummyQwen2VLWrapper()
            ctrl = attach_fastv(
                model,
                FastVConfig(use_fastv=True, fastv_k=2, fastv_r=0.5, mode="inplace", scorer="fastv_mean_std"),
            )
            self.assertIsInstance(ctrl.adapter, Qwen2VLAdapter)

            # Two interleaved images: 4 tokens + 4 tokens = 8 visual tokens, 6 text tokens -> total 14
            input_ids = torch.tensor([[1, 2, 151655, 151655, 151655, 151655, 3, 4, 151655, 151655, 151655, 151655, 5, 6]])
            embeds = torch.randn(1, 14, 16)
            causal_mask = torch.zeros(1, 1, 14, 14)

            out = model(input_ids=input_ids, inputs_embeds=embeds, attention_mask=causal_mask)
            self.assertEqual(out.shape, (1, 10, 16))
            self.assertEqual(model._backbone_ref.layers[0].self_attn.seen_pos_emb_shapes[-1], (3, 1, 14, 16))
            self.assertEqual(model._backbone_ref.layers[2].self_attn.seen_pos_emb_shapes[-1], (3, 1, 10, 16))
            self.assertEqual(ctrl.last_stats["adapter"], "qwen2_vl")
            self.assertEqual(ctrl.last_stats["scorer"], "fastv_mean_std")

        def test_r0_parity_and_inplace_vs_mask_equivalence(self) -> None:
            torch.manual_seed(0)
            model = _DummyLlavaWrapper(nested_v5=True)
            input_ids = torch.tensor([[1, 2, 3] + [32000] * 8 + [4, 5, 6]])
            embeds = torch.randn(1, 14, 16)
            # Causal additive mask
            mask = torch.triu(torch.full((14, 14), torch.finfo(embeds.dtype).min), diagonal=1).view(1, 1, 14, 14)

            # Vanilla reference
            vanilla_out = model(input_ids=input_ids, inputs_embeds=embeds, attention_mask=mask)

            # R = 0.0 with force_slice_at_r0=True must match vanilla exactly
            attach_fastv(
                model,
                FastVConfig(use_fastv=True, fastv_k=2, fastv_r=0.0, mode="inplace", force_slice_at_r0=True),
            )
            r0_out = model(input_ids=input_ids, inputs_embeds=embeds, attention_mask=mask)
            self.assertTrue(torch.allclose(vanilla_out, r0_out, atol=1e-5))

            # Inplace vs Mask must match on the final token representation at K=2, R=0.5
            attach_fastv(model, FastVConfig(use_fastv=True, fastv_k=2, fastv_r=0.5, mode="inplace"))
            inplace_out = model(input_ids=input_ids, inputs_embeds=embeds, attention_mask=mask)

            attach_fastv(model, FastVConfig(use_fastv=True, fastv_k=2, fastv_r=0.5, mode="mask"))
            mask_out = model(input_ids=input_ids, inputs_embeds=embeds, attention_mask=mask)

            self.assertTrue(torch.allclose(inplace_out[:, -1, :], mask_out[:, -1, :], atol=1e-5))
            remove_fastv(model)
            self.assertFalse(hasattr(model, "_fastv_controller"))

        def test_mask_mode_k0_and_k2(self) -> None:
            model = _DummyLlavaWrapper(nested_v5=True)
            input_ids = torch.tensor([[1, 2, 3] + [32000] * 8 + [4, 5, 6]])
            embeds = torch.randn(1, 14, 16)
            causal_mask = torch.zeros(1, 1, 14, 14)

            for k in (0, 2):
                ctrl = attach_fastv(model, {"use_fastv": True, "fastv_k": k, "fastv_r": 0.5, "mode": "mask"})
                out = model(input_ids=input_ids, inputs_embeds=embeds, attention_mask=causal_mask)
                self.assertEqual(out.shape, (1, 14, 16))
                self.assertEqual(ctrl.last_stats["dropped_image_tokens"], 4)

        def test_attention_efficiency_metric(self) -> None:
            attn = torch.full((1, 2, 20, 20), 1.0 / 20.0)
            res = compute_layer_attention_efficiency([attn, attn], image_start=4, image_length=12)
            self.assertEqual(res["layer"], [1, 2])
            self.assertAlmostEqual(res["image_total"][0], 12.0 / 20.0, places=5)
            self.assertAlmostEqual(res["visual_attention_efficiency"][0], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
