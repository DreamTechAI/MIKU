"""Check sequential model lifetime and dispatch without allocating a GPU."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image


class StageTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        events = self.events
        self.image = Image.new("RGB", (32, 32), (20, 30, 40))
        image = self.image
        self.mr_fails = False
        self.expected_masks = None
        test = self

        class MV:
            last_prediction = image
            last_metadata = {"native_canvas_size": [32, 32]}

            def close(self):
                events.append("albedo_close")

        class Base:
            def __init__(self, config, load_models=True):
                self.config, self.models, self.stats_logs = config, {}, {}
                if load_models:
                    self.load_models()

            def load_models(self):
                events.append("albedo_load")
                self.models["multiview_model"] = MV()

            def predict_multiview(self, image_path, conditions, prompt=None):
                events.append("albedo_predict")
                return {"albedo": [image] * 6}

        class MR:
            def __init__(self, config):
                events.append("mr_load")

            def __call__(self, views, reference, **kwargs):
                events.append("mr_predict")
                test.assertIs(kwargs["packed_albedo"], image)
                test.assertEqual(kwargs["packing_metadata"], MV.last_metadata)
                test.assertIs(reference, image)
                test.assertIs(kwargs["masks"], test.expected_masks)
                if test.mr_fails:
                    raise RuntimeError("PBR sampling failed")
                return [image] * 6

            def close(self):
                events.append("mr_close")

        base_module = ModuleType("textureGenPipeline")
        base_module.MaterialMVPPipeline = Base
        mr_module = ModuleType("utils.miku_pbr_utils")
        mr_module.MikuPBRDiffusionNet = MR
        self.patcher = patch.dict("sys.modules", {
            "textureGenPipeline": base_module, "utils.miku_pbr_utils": mr_module,
        })
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        spec = importlib.util.spec_from_file_location(
            "miku_stage_test_module", Path(__file__).resolve().parents[1] / "mikuPipeline.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.config = SimpleNamespace(pbr_enabled=True, pbr_mode="pack",
                                      pbr_checkpoint_path="/checkpoint", miku_debug_dir=None)
        self.pipeline = module.MikuPipeline(self.config, load_models=False)

    def test_models_release_before_next_stage_and_reload_for_next_mesh(self):
        for _ in range(2):
            result = self.pipeline.predict_multiview(self.image, {"masks": None})
            self.assertEqual(set(result), {"albedo", "mr"})
            self.assertNotIn("multiview_model", self.pipeline.models)
        self.assertEqual(self.events, ["albedo_load", "albedo_predict", "albedo_close",
                                      "mr_load", "mr_predict", "mr_close"] * 2)

    def test_albedo_only_does_not_load_pbr(self):
        self.config.pbr_enabled = False
        result = self.pipeline.predict_multiview(self.image, {"masks": None})
        self.assertEqual(set(result), {"albedo"})
        self.assertEqual(self.events, ["albedo_load", "albedo_predict"])

    def test_geometry_masks_are_passed_to_mr_without_reordering(self):
        self.expected_masks = [Image.new("L", (32, 32), i) for i in range(6)]
        self.pipeline.predict_multiview(self.image, {"masks": self.expected_masks})
        self.assertIn("mr_predict", self.events)

    def test_direct_mr_call_releases_an_already_loaded_albedo_model(self):
        self.pipeline.load_models()
        self.pipeline.predict_mr([self.image] * 6, self.image, {"masks": None},
                                 packed_albedo=self.image,
                                 packing_metadata={"native_canvas_size": [32, 32]})
        self.assertEqual(self.events, ["albedo_load", "albedo_close", "mr_load", "mr_predict", "mr_close"])

    def test_mr_failure_propagates_and_closes_model(self):
        self.mr_fails = True
        with self.assertRaisesRegex(RuntimeError, "PBR sampling failed"):
            self.pipeline.predict_multiview(self.image, {"masks": None})
        self.assertEqual(self.events[-1], "mr_close")


if __name__ == "__main__":
    unittest.main()
