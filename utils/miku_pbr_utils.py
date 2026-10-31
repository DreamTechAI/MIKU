"""Current albedo-to-MR inference with the Miku camera/atlas inverse mapping.

G is roughness and B is metallic, as in glTF. Foreground predictions retain
continuous values. Geometry masks optionally whiten the predicted background;
the albedo model input and saved raw float prediction remain unchanged.
"""
from __future__ import annotations

import copy
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from utils.miku_multiview_utils import unpack_miku_prediction


def prepare_pbr_reference(reference, size=512):
    """Match eval_direct_pbr.py: composite alpha, fit the entire image, no crop."""
    if size < 16 or size % 16:
        raise ValueError("PBR reference size must be a positive multiple of 16")
    if isinstance(reference, (str, Path)):
        with Image.open(reference) as source:
            reference = source.copy()
    if not isinstance(reference, Image.Image):
        raise TypeError("PBR requires one PIL reference image or its path")
    if "A" in reference.getbands() or "transparency" in reference.info:
        rgba = reference.convert("RGBA")
        reference = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    return ImageOps.pad(reference.convert("RGB"), (size, size),
                        method=Image.Resampling.BILINEAR, color="white", centering=(0.5, 0.5))


def prepare_pbr_atlas(albedo_views, masks=None, *, mode="pack", view_size=1024,
                      packed_albedo=None, packing_metadata=None):
    """Prepare the two user-selected MR layouts without changing RGB backgrounds.

    pack: resize the existing generated albedo atlas by exactly input-scale 0.5
    (round dimensions as eval_direct_pbr.py), then pad right/bottom to 16.
    nonpack: resize each complete camera frame to 512 square and use a 3x2 grid.
    Geometry masks are accepted for compatibility, but never crop or composite
    the actual albedo supplied to this model.
    """
    if not albedo_views:
        raise ValueError("PBR requires at least one albedo view")
    if mode not in ("pack", "nonpack"):
        raise ValueError("pbr_mode must be pack or nonpack")
    if mode == "pack":
        if packed_albedo is None or packing_metadata is None:
            raise ValueError("PBR pack mode requires the original packed_albedo and packing_metadata")
        original = copy.deepcopy(packing_metadata)
        if original.get("format") != "miku_views_v1" or original.get("rotation") != "none":
            raise ValueError("Unsupported PBR albedo atlas mapping")
        if len(original["views"]) != len(albedo_views):
            raise ValueError("PBR atlas view count differs from the albedo views")
        if list(packed_albedo.size) != original["model_canvas_size"]:
            raise ValueError("PBR packed albedo dimensions differ from its mapping")
        for index, (view, entry) in enumerate(zip(albedo_views, original["views"])):
            if entry["view_index"] != index or list(view.size) != entry["source_size"]:
                raise ValueError("PBR albedo views must match their original camera mapping")
        scaled_size = tuple(max(1, round(side * 0.5)) for side in packed_albedo.size)
        condition = packed_albedo.convert("RGB").resize(scaled_size, Image.Resampling.BILINEAR)
        padding = (0, 0, (-condition.width) % 16, (-condition.height) % 16)
        condition = ImageOps.expand(condition, border=padding, fill="white")
        metadata = {"format": "miku_pbr_v1", "mode": mode, "rotation": "none",
                    "native_canvas_size": original["native_canvas_size"],
                    "model_canvas_size": list(condition.size), "views": original["views"],
                    "albedo_packing": original, "scaled_canvas_size": list(scaled_size),
                    "padding_ltrb": list(padding), "output_crop_box": [0, 0, *scaled_size],
                    "input_scale": 0.5, "reused_albedo_atlas": True,
                    "background_edited": False, "rotation_applied": False}
    else:
        if len(albedo_views) != 6:
            raise ValueError("PBR nonpack mode requires exactly six camera views")
        condition = Image.new("RGB", (1536, 1024), "white")
        entries = []
        for index, view in enumerate(albedo_views):
            x, y = index % 3 * 512, index // 3 * 512
            condition.paste(view.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), (x, y))
            entries.append({"view_index": index, "source_size": list(view.size),
                            "view_size": 512, "crop_box": [0, 0, 512, 512],
                            "atlas_box": [x, y, x + 512, y + 512]})
        metadata = {"format": "miku_pbr_v1", "mode": mode, "rotation": "none",
                    "native_canvas_size": [1536, 1024], "model_canvas_size": [1536, 1024],
                    "views": entries, "scaled_canvas_size": [1536, 1024],
                    "padding_ltrb": [0, 0, 0, 0], "output_crop_box": [0, 0, 1536, 1024],
                    "grid_columns": 3, "grid_rows": 2, "tile_size": 512,
                    "reused_albedo_atlas": False, "background_edited": False,
                    "rotation_applied": False}
    return condition, metadata


def unpack_pbr_prediction(prediction, metadata):
    """Remove only recorded padding, then restore original camera frames."""
    if list(prediction.size) != metadata["model_canvas_size"]:
        raise ValueError("PBR output size differs from the model input")
    if metadata["mode"] == "pack":
        original = metadata["albedo_packing"]
        prediction = prediction.crop(metadata["output_crop_box"]).resize(
            tuple(original["model_canvas_size"]), Image.Resampling.BILINEAR)
        return unpack_miku_prediction(prediction, original)
    if metadata["mode"] == "nonpack":
        return unpack_miku_prediction(prediction, metadata)
    raise ValueError("Unknown PBR layout mode")


def prediction_to_image(prediction):
    """Quantize only for PNG/baking, keeping the raw float prediction separately."""
    array = np.asarray(prediction)
    if array.ndim == 3:
        array = array.transpose(1, 2, 0)
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def validate_pbr_masks(masks, metadata):
    """Require one pixel-aligned geometry mask for each original camera frame."""
    if masks is None or len(masks) != len(metadata["views"]):
        raise ValueError("White MR background requires one geometry mask per view")
    binary = []
    for index, (mask, entry) in enumerate(zip(masks, metadata["views"])):
        if not isinstance(mask, Image.Image):
            raise TypeError(f"PBR geometry mask {index} must be a PIL image")
        if entry["view_index"] != index or list(mask.size) != entry["source_size"]:
            raise ValueError(f"PBR geometry mask {index} must match its original camera frame")
        binary.append(Image.fromarray((np.asarray(mask.convert("L")) > 0).astype(np.uint8) * 255))
    return binary


def prepare_pbr_mask(masks, metadata):
    """Map geometry masks to the exact MR canvas, with zero-valued padding."""
    masks = validate_pbr_masks(masks, metadata)
    mapping = metadata["albedo_packing"] if metadata["mode"] == "pack" else metadata
    native = Image.new("L", tuple(mapping["native_canvas_size"]), 0)
    for mask, entry in zip(masks, mapping["views"]):
        patch = mask.resize((entry["view_size"], entry["view_size"]), Image.Resampling.NEAREST)
        patch = patch.crop(entry["crop_box"])
        box = entry["atlas_box"]
        if patch.size != (box[2] - box[0], box[3] - box[1]):
            raise ValueError("PBR mask crop and atlas placement dimensions differ")
        native.paste(patch, box[:2])
    model_mask = native.resize(tuple(mapping["model_canvas_size"]), Image.Resampling.NEAREST)
    if metadata["mode"] == "pack":
        # Preserve both scale operations: nearest-neighbour sampling is not
        # equivalent to a single combined resize for arbitrary canvas sizes.
        half = model_mask.resize(tuple(metadata["scaled_canvas_size"]), Image.Resampling.NEAREST)
        model_mask = Image.new("L", tuple(metadata["model_canvas_size"]), 0)
        model_mask.paste(half, metadata["output_crop_box"][:2])
    elif metadata["mode"] != "nonpack":
        raise ValueError("Unknown PBR layout mode")
    return model_mask


def postprocess_pbr_prediction(prediction, metadata, masks=None, *, background_white=True):
    """Whiten only background, preserving raw inverse-mapped foreground pixels."""
    views = unpack_pbr_prediction(prediction, metadata)
    if not background_white:
        return prediction, views, None
    masks = validate_pbr_masks(masks, metadata)
    model_mask = prepare_pbr_mask(masks, metadata)
    orm = Image.composite(prediction, Image.new("RGB", prediction.size, "white"), model_mask)
    # Restore from RAW first, then apply the original-resolution masks. Resizing
    # a whitened atlas would bleed artificial white into the object's boundary.
    views = [Image.composite(view, Image.new("RGB", view.size, "white"), mask)
             for view, mask in zip(views, masks)]
    return orm, views, model_mask


class MikuPBRDiffusionNet:
    def __init__(self, config):
        import torch
        from miku.pbr.checkpoint import initialize_adapter
        from miku.pbr.dino import FrozenDinoEmbedder_v2
        from miku.pbr.pipeline_pbr import FluxMVKleinPipeline
        from miku.pbr.transformer_orm_compatible import Flux2ORMTransformer2DModel

        self.config = config
        self.device = torch.device(config.device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The current PBR+DINO checkpoint requires CUDA")
        self.steps = int(config.pbr_steps)
        self.seed = int(config.pbr_seed)
        if self.steps < 1:
            raise ValueError("pbr_steps must be positive")
        if config.pbr_dtype not in ("bfloat16", "float16"):
            raise ValueError("pbr_dtype must be bfloat16 or float16")
        self.dtype = getattr(torch, config.pbr_dtype)
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support PBR bfloat16")
        if config.pbr_offload not in ("model", "none"):
            raise ValueError("pbr_offload must be model or none")
        checkpoint = Path(config.pbr_checkpoint_path)
        base_model = Path(config.pbr_pretrained_path or config.miku_pretrained_path)
        dino_weights = Path(config.pbr_dino_weights_path)
        for path in (checkpoint / "pytorch_lora_weights.safetensors", checkpoint / "dino_linear.pt",
                     base_model / "model_index.json", dino_weights):
            if not path.is_file():
                raise FileNotFoundError(path)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        started = time.monotonic()
        with torch.cuda.device(self.device):
            transformer = Flux2ORMTransformer2DModel.from_pretrained(
                str(base_model), subfolder="transformer", torch_dtype=self.dtype, local_files_only=True)
            _, self.checkpoint_report = initialize_adapter(
                transformer, checkpoint, adapter_name="miku_orm", return_report=True)
            transformer.eval().requires_grad_(False)
            self.pipeline = FluxMVKleinPipeline.from_pretrained(
                str(base_model), transformer=transformer, torch_dtype=self.dtype, local_files_only=True)
            self.pipeline.dino_encoder = FrozenDinoEmbedder_v2(
                weights_path=str(dino_weights), device=str(self.device),
                linear_ckpt_path=str(checkpoint / "dino_linear.pt"))
            saved = torch.load(checkpoint / "dino_linear.pt", map_location="cpu", weights_only=True)
            saved = saved.get("state_dict", saved)
            actual = self.pipeline.dino_encoder.linear.state_dict()
            if set(actual) != set(saved):
                raise ValueError("DINO projection checkpoint keys differ")
            for key, value in saved.items():
                loaded = actual[key].detach().cpu()
                if loaded.shape != value.shape or not torch.equal(loaded, value.to(loaded.dtype)):
                    raise ValueError(f"DINO projection was not completely restored: {key}")
            self.pipeline.dino_encoder.eval().requires_grad_(False)
            if config.pbr_vae_tiling:
                self.pipeline.vae.enable_tiling()
            else:
                self.pipeline.vae.disable_tiling()
            if config.pbr_offload == "model":
                self.pipeline.enable_model_cpu_offload(gpu_id=self.device.index or 0, device="cuda")
            else:
                self.pipeline.to(self.device)
        self.load_seconds = time.monotonic() - started
        self.last_metadata = None
        self.last_report = None

    def __call__(self, albedo_views, reference, masks=None, *, packed_albedo=None, packing_metadata=None):
        import torch

        if self.pipeline is None:
            raise RuntimeError("PBR model has been closed")
        started = time.monotonic()
        condition, metadata = prepare_pbr_atlas(
            albedo_views, masks, view_size=self.config.resolution, mode=self.config.pbr_mode,
            packed_albedo=packed_albedo, packing_metadata=packing_metadata)
        background_white = self.config.pbr_background_white
        if background_white:
            validate_pbr_masks(masks, metadata)  # Fail before expensive sampling.
        reference = prepare_pbr_reference(reference, self.config.pbr_reference_size)
        self.last_metadata = metadata
        debug_dir = getattr(self.config, "miku_debug_dir", None)
        if debug_dir:
            debug_dir = Path(debug_dir) / "pbr"
            debug_dir.mkdir(parents=True, exist_ok=True)
            condition.save(debug_dir / "input_condition.png")
            reference.save(debug_dir / "reference_prepared.png")
            (debug_dir / "packing.json").write_text(json.dumps(metadata, indent=2) + "\n")
        finite_steps = []

        def check_step(_pipeline, step, timestep, callback_kwargs):
            if not torch.isfinite(callback_kwargs["latents"]).all().item():
                raise FloatingPointError(f"Nonfinite PBR latent at denoising step {step}")
            finite_steps.append(int(step))
            return callback_kwargs

        with torch.cuda.device(self.device):
            torch.cuda.reset_peak_memory_stats(self.device)
            generator = torch.Generator(device=self.device).manual_seed(self.seed)
            with torch.inference_mode(), torch.autocast("cuda", dtype=self.dtype):
                generated = self.pipeline(
                    [condition, reference], prompt=self.config.pbr_prompt,
                    width=condition.width, height=condition.height, num_inference_steps=self.steps,
                    guidance_scale=1.0, generator=generator, output_type="pt",
                    callback_on_step_end=check_step, callback_on_step_end_tensor_inputs=["latents"],
                ).images
        prediction = generated[0].detach().float().cpu().numpy()
        if prediction.shape != (3, condition.height, condition.width):
            raise ValueError(f"Unexpected decoded PBR shape: {prediction.shape}")
        if not np.isfinite(prediction).all() or len(finite_steps) != self.steps:
            raise FloatingPointError("PBR prediction or denoising execution failed validation")
        raw_orm = prediction_to_image(prediction)
        orm, views, model_mask = postprocess_pbr_prediction(
            raw_orm, metadata, masks, background_white=background_white)
        report = {
            "status": "passed", "checkpoint": str(self.config.pbr_checkpoint_path),
            "base_model": self.config.pbr_pretrained_path or self.config.miku_pretrained_path,
            "checkpoint_load": self.checkpoint_report, "dino_linear_verified": True,
            "seed": self.seed, "steps": self.steps, "prompt": self.config.pbr_prompt,
            "dtype": str(self.dtype), "model_cpu_offload": self.config.pbr_offload == "model",
            "vae_tiling": self.config.pbr_vae_tiling, "reference_size_wh": list(reference.size),
            "input_size_wh": list(condition.size), "prediction_shape_chw": list(prediction.shape),
            "prediction_finite": True, "denoising_finite_steps": len(finite_steps),
            "channel_mapping": {"roughness": "G", "metallic": "B", "red": "unused by glTF MR"},
            "channel_ranges": {name: [float(prediction[i].min()), float(prediction[i].max())]
                               for i, name in enumerate(("red", "roughness", "metallic"))},
            "raw_prediction": "prediction_float32.npy (CHW; no channel overrides)",
            "background_white": background_white,
            "background_mask": "background_mask.png" if background_white else None,
            "background_policy": ("geometry mask > 0; whiten after raw inverse mapping at original resolution"
                                  if background_white else "unchanged"),
            "view_count": len(views), "view_sizes_wh": [list(view.size) for view in views],
            "mode": self.config.pbr_mode,
            "packing_preparation": {key: value for key, value in metadata.items()
                                    if key not in ("views", "albedo_packing")},
            "load_seconds": self.load_seconds,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(self.device) / 1024**2,
        }
        self.last_report = report
        if debug_dir:
            np.save(debug_dir / "prediction_float32.npy", prediction)
            raw_orm.save(debug_dir / "predicted_orm_raw.png")
            if model_mask is not None:
                model_mask.save(debug_dir / "background_mask.png")
            else:
                (debug_dir / "background_mask.png").unlink(missing_ok=True)
            orm.save(debug_dir / "predicted_orm.png")
            orm.getchannel("G").save(debug_dir / "predicted_roughness.png")
            orm.getchannel("B").save(debug_dir / "predicted_metallic.png")
            for index, view in enumerate(views):
                view.save(debug_dir / f"mr_{index:02d}.png")
            (debug_dir / "inference_report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return views

    def close(self):
        """Release CUDA DINO and accelerated modules before the UV bake/next mesh."""
        import torch
        from accelerate.hooks import remove_hook_from_module

        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None:
            if hasattr(pipeline, "remove_all_hooks"):
                pipeline.remove_all_hooks()
            for name in ("transformer", "text_encoder", "vae", "dino_encoder"):
                module = getattr(pipeline, name, None)
                if module is not None:
                    remove_hook_from_module(module, recurse=True)
                    module.to("cpu")
            self.pipeline = None
            del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
