"""Real tiny-model forward checks; requires the installed Miku ML runtime."""

import unittest

try:
    import torch
    from diffusers import Flux2Transformer2DModel
except ImportError as exc:
    raise unittest.SkipTest("Tiny FLUX.2 checks require the Miku torch/diffusers environment") from exc

from miku.drop_transformer import Flux2DropConditionModel
from utils.miku_multiview_utils import install_miku_block_compatibility


class RuntimeCompatibilityTests(unittest.TestCase):
    def make_models(self):
        torch.manual_seed(31)
        config = dict(in_channels=8, num_layers=2, num_single_layers=2,
                      attention_head_dim=8, num_attention_heads=2,
                      joint_attention_dim=16, timestep_guidance_channels=16,
                      axes_dims_rope=(2, 2, 2, 2), guidance_embeds=False)
        model = Flux2DropConditionModel(**config).eval()
        baseline = Flux2Transformer2DModel(**config).eval()
        baseline.load_state_dict(model.state_dict(), strict=True)
        report = install_miku_block_compatibility(model)
        self.assertEqual(report["double_stream_blocks"], 2)
        self.assertEqual(report["single_stream_blocks"], 2)
        self.assertTrue(report["preserve_single_modulation_batch"])
        return model, baseline

    def inputs(self, batch):
        return dict(hidden_states=torch.randn(batch, 12, 8),
                    encoder_hidden_states=torch.randn(batch, 4, 16),
                    timestep=torch.linspace(0.2, 0.8, batch),
                    img_ids=torch.arange(12).view(-1, 1).expand(-1, 4).float(),
                    txt_ids=torch.zeros(4, 4), return_dict=False)

    @torch.no_grad()
    def test_original_source_matches_unmodified_base_forward(self):
        model, baseline = self.make_models()
        for batch in (1, 2):
            inputs = self.inputs(batch)
            actual = model(**inputs, drop_condition=False)[0]
            expected = baseline(**inputs)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        handles = len(model._miku_runtime_compat_handles)
        install_miku_block_compatibility(model)
        self.assertEqual(len(model._miku_runtime_compat_handles), handles)

    @torch.no_grad()
    def test_drop_condition_runs_with_distinct_batch_timesteps(self):
        model, baseline = self.make_models()
        for batch in (1, 2):
            inputs = self.inputs(batch)
            actual = model(**inputs, drop_condition=True, condition_length=4)[0]
            self.assertEqual(tuple(actual.shape), (batch, 8, 8))
            self.assertTrue(torch.isfinite(actual).all().item())
            full_output = baseline(**inputs)[0]
            self.assertFalse(torch.equal(actual, full_output[:, :8]))


if __name__ == "__main__":
    unittest.main()
