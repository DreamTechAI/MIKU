"""Ensure a relocated release reads weights only from its configured root."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from miku_config import MikuConfig


class WeightPathsTests(unittest.TestCase):
    def test_environment_root_and_explicit_absolute_override(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MIKU_CHECKPOINT_DIR": directory}):
                config = MikuConfig(miku_lora_path="/external/albedo.safetensors").resolve_paths()
            self.assertEqual(config.miku_pretrained_path, str(Path(directory) / "flux2-klein-base-9b"))
            self.assertEqual(config.miku_lora_path, "/external/albedo.safetensors")
            self.assertEqual(config.pbr_checkpoint_path, str(Path(directory) / "pbr"))
            self.assertEqual(config.pbr_dino_weights_path,
                             str(Path(directory) / "dinov2/dinov2_vitl14_reg4_pretrain.pth"))

    def test_default_is_repo_relative_independent_of_current_directory(self):
        import miku_config
        config = MikuConfig(checkpoints_dir="checkpoints").resolve_paths()
        self.assertEqual(Path(config.checkpoints_dir), Path(miku_config.__file__).parent / "checkpoints")
        first = config.miku_lora_path
        config.resolve_paths()
        self.assertEqual(config.miku_lora_path, first)

    def test_missing_assets_fail_before_inference_and_albedo_only_skips_pbr(self):
        with tempfile.TemporaryDirectory() as directory:
            config = MikuConfig(checkpoints_dir=directory)
            with self.assertRaisesRegex(FileNotFoundError, "dino_linear.pt"):
                config.validate_weights()
            base = Path(directory) / "flux2-klein-base-9b"
            for name in ("transformer", "text_encoder", "tokenizer", "vae", "scheduler"):
                (base / name).mkdir(parents=True)
            (base / "model_index.json").write_text("{}")
            albedo = Path(directory) / "albedo/pytorch_lora_weights.safetensors"
            albedo.parent.mkdir()
            albedo.touch()
            config.pbr_enabled = False
            config.validate_weights()
            config.pbr_enabled = True
            with self.assertRaisesRegex(FileNotFoundError, "dinov2_vitl14_reg4_pretrain.pth"):
                config.validate_weights()


if __name__ == "__main__":
    unittest.main()
