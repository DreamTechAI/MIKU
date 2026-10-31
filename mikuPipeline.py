"""Miku albedo -> PBR MR -> MaterialMVP UV baking and inpainting."""

import json
import os
from pathlib import Path
import time

from PIL import Image

from miku_config import MikuConfig
from textureGenPipeline import MaterialMVPPipeline


class MikuPipeline(MaterialMVPPipeline):
    def __init__(self, config=None, load_models=True):
        config = config or MikuConfig()
        if hasattr(config, "resolve_paths"):
            config.resolve_paths()
        super().__init__(config, load_models=load_models)

    def predict_multiview(self, image_path, conditions, prompt=None):
        # PBR runs after albedo and releases both large models between stages.
        # Reloading here makes the same MikuPipeline reusable for another mesh.
        if "multiview_model" not in self.models:
            self.load_models()
        started = time.monotonic()
        print("[Miku] Generating multiview albedo", flush=True)
        views = super().predict_multiview(image_path, conditions, prompt=prompt)
        self.stats_logs["albedo_seconds"] = time.monotonic() - started
        if not self.config.pbr_enabled:
            return views
        model = self.models.pop("multiview_model")
        packed_albedo = model.last_prediction
        metadata = model.last_metadata
        model.close()
        del model
        views["mr"] = self.predict_mr(
            views["albedo"], image_path, conditions,
            packed_albedo=packed_albedo, packing_metadata=metadata,
        )
        return views

    def predict_mr(self, albedo_views, image_path, conditions, *, packed_albedo=None,
                   packing_metadata=None):
        """Predict MR in pack/nonpack mode and restore the same camera frames."""
        from utils.miku_pbr_utils import MikuPBRDiffusionNet

        if isinstance(image_path, (str, os.PathLike)):
            with Image.open(image_path) as source:
                reference = source.copy()
        elif isinstance(image_path, Image.Image):
            reference = image_path
        else:
            references = list(image_path)
            if len(references) != 1:
                raise ValueError("PBR requires exactly one complete reference image")
            reference = references[0]
        # Also support direct MR calls with cached albedo on a freshly created
        # pipeline, whose constructor may already have loaded the MV model.
        previous = self.models.pop("multiview_model", None)
        if previous is not None:
            previous.close()
            del previous
        started = time.monotonic()
        print(f"[Miku] Generating MR ({self.config.pbr_mode}), checkpoint {self.config.pbr_checkpoint_path}",
              flush=True)
        model = MikuPBRDiffusionNet(self.config)
        try:
            views = model(albedo_views, reference, masks=conditions["masks"],
                          packed_albedo=packed_albedo, packing_metadata=packing_metadata)
        finally:
            model.close()
        self.stats_logs["pbr_seconds"] = time.monotonic() - started
        self.stats_logs["pbr_mode"] = self.config.pbr_mode
        self.stats_logs["pbr_checkpoint"] = self.config.pbr_checkpoint_path
        if self.config.miku_debug_dir:
            (Path(self.config.miku_debug_dir) / "stages.json").write_text(
                json.dumps(self.stats_logs, indent=2) + "\n")
        return views

    def __call__(self, mesh_path=None, image_path=None, output_mesh_path=None,
                 use_remesh=False, save_glb=True, prompt=None, mv_only=False):
        return super().__call__(mesh_path, image_path, output_mesh_path,
                                use_remesh, save_glb, prompt, mv_only)
