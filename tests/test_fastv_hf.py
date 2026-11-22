"""Unit tests for the modern hook-based FastV implementation.

Runs pure-Python tests everywhere, and runs full PyTorch hook/pruning tests
(covering both transformers 4.x tuple-returning decoder layers and
transformers 5.x tensor-returning decoder layers) whenever PyTorch is installed.
"""

from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from FastV.fastv_hf import (
    FastVConfig,
    compute_fastv_flops_ratio,
    compute_rouge_l,
    detect_image_token_span,
    format_llava_prompt,
    resolve_hf_model_id,
)

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestFastVPurePython(unittest.TestCase):
    def test_flops_ratio_matches_paper(self) -> None:
        # LLaVA-1.5-7B: 32 layers, d=4096, m=11008, 576 visual tokens, K=2, R=50%
        res = compute_fastv_flops_ratio(
            num_layers=32,
            fastv_k=2,
            fastv_r=0.5,
            n_image_tokens=576,
            n_text_tokens=60,
            hidden_size=4096,
            intermediate_size=11008,
        )
        # Paper headline: ~45% theoretical FLOPs reduction at K=2, R=50%
        self.assertGreater(res["flops_reduction_pct"], 40.0)
        self.assertLess(res["flops_reduction_pct"], 50.0)

        # Baseline (R=0) must give 0% reduction
        base = compute_fastv_flops_ratio(num_layers=32, fastv_k=2, fastv_r=0.0)
        self.assertAlmostEqual(base["flops_reduction_pct"], 0.0, places=6)

    def test_detect_image_token_span(self) -> None:
        # Expanded tokens (modern transformers >= 4.45)
        ids = [[1, 3148, 1001, 29901, 29871] + [32000] * 16 + [13, 500, 501]]
        self.assertEqual(detect_image_token_span(ids, image_token_index=32000), (5, 16))

        # Single placeholder token (legacy transformers)
        ids_single = [[1, 10, 20, 32000, 30]]
        self.assertEqual(
            detect_image_token_span(ids_single, image_token_index=32000, default_image_length=576),
            (3, 576),
        )

        # No image token
        self.assertIsNone(detect_image_token_span([[1, 2, 3]], image_token_index=32000))

    def test_config_legacy_aliases(self) -> None:
        cfg = FastVConfig.from_any(
            {
                "use_fast_v": True,
                "fast_v_agg_layer": 3,
                "fast_v_sys_length": 36,
                "fast_v_image_token_length": 576,
                "fast_v_attention_rank": 144,
                "fast_v_inplace": False,
            }
        )
        self.assertTrue(cfg.use_fastv)
        self.assertEqual(cfg.fastv_k, 3)
        self.assertEqual(cfg.image_token_start_index, 36)
        self.assertEqual(cfg.image_token_length, 576)
        self.assertEqual(cfg.resolve_keep_count(576), 144)
        self.assertEqual(cfg.mode, "mask")
        self.assertAlmostEqual(cfg.effective_pruning_ratio(576), 0.75, places=6)

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
    from FastV.fastv_hf import attach_fastv, compute_layer_attention_efficiency

    class _DummyConfig:
        def __init__(self, num_layers: int = 4, image_token_index: int = 32000):
            self.num_hidden_layers = num_layers
            self.image_token_index = image_token_index
            self._attn_implementation = "eager"

    class _DummySelfAttn(nn.Module):
        def __init__(self, dim: int, num_heads: int):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.config = _DummyConfig()
            self.seen_q_lens: list[int] = []
            self.seen_mask_shapes: list[tuple[int, ...] | None] = []

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            output_attentions: bool = False,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.shape
            self.seen_q_lens.append(q_len)
            self.seen_mask_shapes.append(tuple(attention_mask.shape) if attention_mask is not None else None)
            if position_embeddings is not None:
                assert position_embeddings[0].shape[1] == q_len
            if position_ids is not None:
                assert position_ids.shape[1] == q_len
            # Construct deterministic attention weights that peak on even image tokens.
            kv_len = attention_mask.shape[-1] if attention_mask is not None else q_len
            attn_weights = torch.linspace(0.01, 1.0, kv_len, device=hidden_states.device).view(1, 1, 1, kv_len)
            attn_weights = attn_weights.expand(bsz, self.num_heads, q_len, kv_len).clone()
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
            return hidden_states, attn_weights

    class _DummyDecoderLayer(nn.Module):
        def __init__(self, dim: int, num_heads: int, return_tuple: bool = False):
            super().__init__()
            self.self_attn = _DummySelfAttn(dim, num_heads)
            self.return_tuple = return_tuple

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            cache_position: torch.Tensor | None = None,
            **kwargs,
        ):
            out, attn = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if self.return_tuple:
                return (out, attn)
            return out

    class _DummyLlamaBackbone(nn.Module):
        def __init__(self, num_layers: int = 4, dim: int = 16, num_heads: int = 2, return_tuple: bool = False):
            super().__init__()
            self.config = _DummyConfig(num_layers=num_layers)
            self.layers = nn.ModuleList(
                [_DummyDecoderLayer(dim, num_heads, return_tuple=return_tuple) for _ in range(num_layers)]
            )

        def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None):
            bsz, seq_len, dim = inputs_embeds.shape
            device = inputs_embeds.device
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
                    cache_position=cache_position,
                )
                hidden_states = out[0] if isinstance(out, tuple) else out
            return hidden_states

    class _DummyLlavaWrapper(nn.Module):
        def __init__(self, nested_v5: bool = True):
            super().__init__()
            self.config = _DummyConfig(num_layers=4, image_token_index=32000)
            backbone = _DummyLlamaBackbone(num_layers=4, dim=16, num_heads=2, return_tuple=not nested_v5)
            if nested_v5:
                self.model = nn.Module()
                self.model.language_model = backbone
            else:
                self.language_model = nn.Module()
                self.language_model.model = backbone
            self._backbone_ref = backbone

        def forward(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None):
            return self._backbone_ref(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

    class TestFastVTorchHooks(unittest.TestCase):
        def _run_inplace_check(self, nested_v5: bool) -> None:
            model = _DummyLlavaWrapper(nested_v5=nested_v5)
            ctrl = attach_fastv(model, {"use_fastv": True, "fastv_k": 2, "fastv_r": 0.5, "mode": "inplace"})

            # 3 prefix + 8 image + 3 suffix = 14 tokens; pruning 50% of 8 image tokens -> 10 tokens after layer 2
            input_ids = torch.tensor([[1, 2, 3] + [32000] * 8 + [4, 5, 6]])
            embeds = torch.randn(1, 14, 16)
            causal_mask = torch.zeros(1, 1, 14, 14)

            out = model(input_ids=input_ids, inputs_embeds=embeds, attention_mask=causal_mask)
            self.assertEqual(out.shape, (1, 10, 16))
            self.assertEqual(model._backbone_ref.layers[0].self_attn.seen_q_lens[-1], 14)
            self.assertEqual(model._backbone_ref.layers[1].self_attn.seen_q_lens[-1], 14)
            self.assertEqual(model._backbone_ref.layers[2].self_attn.seen_q_lens[-1], 10)
            self.assertEqual(model._backbone_ref.layers[3].self_attn.seen_q_lens[-1], 10)
            self.assertEqual(ctrl.last_stats["kept_image_tokens"], 4)

            # Decode step (q_len == 1, KV len on layer 0 = 15 -> pruned layers should see mask width 11)
            dec_ids = torch.tensor([[99]])
            dec_embeds = torch.randn(1, 1, 16)
            dec_mask = torch.zeros(1, 1, 1, 15)
            dec_out = model(input_ids=dec_ids, inputs_embeds=dec_embeds, attention_mask=dec_mask)
            self.assertEqual(dec_out.shape, (1, 1, 16))
            self.assertEqual(model._backbone_ref.layers[0].self_attn.seen_mask_shapes[-1], (1, 1, 1, 15))
            self.assertEqual(model._backbone_ref.layers[2].self_attn.seen_mask_shapes[-1], (1, 1, 1, 11))

        def test_inplace_transformers_v5_layout(self) -> None:
            self._run_inplace_check(nested_v5=True)

        def test_inplace_transformers_v4_layout(self) -> None:
            self._run_inplace_check(nested_v5=False)

        def test_mask_mode_k0_and_k2(self) -> None:
            model = _DummyLlavaWrapper(nested_v5=True)
            input_ids = torch.tensor([[1, 2, 3] + [32000] * 8 + [4, 5, 6]])
            embeds = torch.randn(1, 14, 16)
            causal_mask = torch.zeros(1, 1, 14, 14)

            for k in (0, 2):
                ctrl = attach_fastv(model, {"use_fastv": True, "fastv_k": k, "fastv_r": 0.5, "mode": "mask"})
                out = model(input_ids=input_ids, inputs_embeds=embeds, attention_mask=causal_mask)
                # Mask mode preserves sequence length
                self.assertEqual(out.shape, (1, 14, 16))
                self.assertEqual(ctrl.last_stats["dropped_image_tokens"], 4)

        def test_attention_efficiency_metric(self) -> None:
            # Uniform attention across 20 tokens (4 sys, 12 img, 4 text)
            attn = torch.full((1, 2, 20, 20), 1.0 / 20.0)
            res = compute_layer_attention_efficiency([attn, attn], image_start=4, image_length=12)
            self.assertEqual(res["layer"], [1, 2])
            self.assertAlmostEqual(res["image_total"][0], 12.0 / 20.0, places=5)
            self.assertAlmostEqual(res["visual_attention_efficiency"][0], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
