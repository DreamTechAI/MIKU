"""Miku image-conditioned multiview prediction and reversible view packing.

The caller supplies normal/position maps in the MIKU geometry convention.
This module changes no coordinate axes. It mixes their encoded RGB values 1:1,
packs mask bounding boxes without rotation, and restores camera-frame images for
MaterialMVP's existing texture baker. Miku predicts RGB, not metallic/roughness.
"""

import json
import inspect
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


DEFAULT_PROMPT = (
    "Masterpiece, best quality, ultra-detailed, extremely high clarity, sharp focus, "
    "clean and precise structures, high-frequency details preserved, no blur, "
    "no grain, no artifacts, high SNR, accurate color reproduction, professional rendering quality"
)


def _rgb_white(image):
    if image.mode == "RGBA" or "transparency" in image.info:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def prepare_miku_reference(image, mode="fit"):
    """Produce 4096 reference tokens without stretching a new rectangular image.

    Old eval.py reads an already prepared 1024-square reference unchanged. The
    old pack_pipeline_eval.py contains an alpha-crop/fit/pad helper, although its
    active tensor path directly resizes. ``stretch`` reproduces that active path;
    ``fit`` keeps aspect ratio for new inputs and preserves existing square RGB
    references exactly. Transparent input is cropped to its nonempty alpha bbox.
    """
    if mode not in ("fit", "stretch"):
        raise ValueError("miku_reference_resize_mode must be 'fit' or 'stretch'")
    if mode == "stretch":
        return _rgb_white(image).resize((1024, 1024), Image.Resampling.BILINEAR)
    if image.mode == "RGBA" or "transparency" in image.info:
        image = image.convert("RGBA")
        bbox = image.getchannel("A").getbbox()
        if bbox is None:
            raise ValueError("Miku reference has no visible pixels")
        image = image.crop(bbox)
    image = _rgb_white(image)
    resized = ImageOps.contain(image, (1024, 1024), Image.Resampling.BILINEAR)
    reference = Image.new("RGB", (1024, 1024), "white")
    reference.paste(resized, ((1024 - resized.width) // 2, (1024 - resized.height) // 2))
    return reference


def _pack_rectangles(sizes):
    """Find a deterministic compact shelf layout; no view is rotated/mirrored."""
    count = len(sizes)
    widths = {max(w for w, _ in sizes), sum(w for w, _ in sizes)}
    if count <= 12:
        sums = {0}
        for width, _ in sizes:
            sums |= {value + width for value in sums}
        widths |= {width for width in sums if width >= min(widths)}
    else:
        total_area = sum(w * h for w, h in sizes)
        widths |= {max(min(widths), math.ceil(math.sqrt(total_area) * ratio / 16) * 16)
                   for ratio in (0.75, 1, 1.25, 1.5, 2)}
    orders = [list(range(count))]
    for key in (lambda i: sizes[i][1], lambda i: sizes[i][0],
                lambda i: sizes[i][0] * sizes[i][1]):
        orders.append(sorted(range(count), key=key, reverse=True))
    best = None
    for limit in sorted(widths):
        for order in orders:
            shelves = []
            boxes = [None] * count
            bottom = 0
            for index in order:
                width, height = sizes[index]
                if width > limit:
                    break
                shelf = next((item for item in shelves
                              if item[2] >= height and item[0] + width <= limit), None)
                if shelf is None:
                    shelf = [0, bottom, height]
                    shelves.append(shelf)
                    bottom += height
                x, y, _ = shelf
                boxes[index] = [x, y, x + width, y + height]
                shelf[0] += width
            else:
                width = max(box[2] for box in boxes)
                height = bottom
                # A small aspect penalty avoids a very long strip whose later
                # 2048-pixel cap would discard most per-view resolution.
                score = (width * height + 0.1 * max(width, height) ** 2,
                         max(width, height), width)
                if best is None or score < best[0]:
                    best = (score, boxes, (width, height))
    return best[1], best[2]


def pack_miku_conditions(normal_maps, position_maps, masks=None, view_size=1024, max_canvas_size=2048):
    """Return a PIL control atlas and JSON-safe inverse mapping.

    Crops are aligned to the model's 16-pixel token grid. Original background
    colors inside the crops are preserved; newly added atlas space is white,
    matching the old dataset packer. Optional masks must be PIL images with
    foreground > 0. With no mask/alpha, the whole frame is retained.
    """
    if not normal_maps or len(normal_maps) != len(position_maps):
        raise ValueError("Expected equal nonempty lists of normal and position maps")
    if view_size <= 0 or view_size % 16:
        raise ValueError("Miku view_size must be a positive multiple of 16")
    if max_canvas_size < 16 or max_canvas_size % 16:
        raise ValueError("miku_max_canvas_size must be a positive multiple of 16")
    if masks is not None and len(masks) != len(normal_maps):
        raise ValueError("One foreground mask is required for each view")
    patches, entries = [], []
    for index, (normal, position) in enumerate(zip(normal_maps, position_maps)):
        if normal.size != position.size:
            raise ValueError(f"Normal and position sizes differ for view {index}")
        if masks is not None:
            mask = masks[index].convert("L")
        elif normal.mode == "RGBA":
            mask = normal.getchannel("A")
        else:
            mask = Image.new("L", normal.size, 255)
        mask = mask.resize((view_size, view_size), Image.Resampling.NEAREST)
        bbox = mask.getbbox()
        if bbox is None:
            raise ValueError(f"View {index} has no visible mesh pixels")
        x0, y0, x1, y1 = bbox
        bbox = [x0 // 16 * 16, y0 // 16 * 16,
                min(view_size, math.ceil(x1 / 16) * 16),
                min(view_size, math.ceil(y1 / 16) * 16)]
        normal_rgb = np.asarray(normal.convert("RGB").resize(
            (view_size, view_size), Image.Resampling.BILINEAR), dtype=np.float32)
        position_rgb = np.asarray(position.convert("RGB").resize(
            (view_size, view_size), Image.Resampling.BILINEAR), dtype=np.float32)
        # PIL -> pipeline preprocess handles normalization to [-1,1].
        control = Image.fromarray((0.5 * normal_rgb + 0.5 * position_rgb).astype(np.uint8))
        patches.append(control.crop(bbox))
        entries.append({"view_index": index, "source_size": list(normal.size),
                        "view_size": view_size, "crop_box": bbox})
    boxes, native_size = _pack_rectangles([patch.size for patch in patches])
    canvas = Image.new("RGB", native_size, "white")
    for patch, entry, box in zip(patches, entries, boxes):
        canvas.paste(patch, box[:2])
        entry["atlas_box"] = box
    if max(native_size) > max_canvas_size:
        scale = max_canvas_size / max(native_size)
        resized_size = tuple(max(16, int(side * scale) // 16 * 16) for side in native_size)
        canvas = canvas.resize(resized_size, Image.Resampling.BILINEAR)
    metadata = {"format": "miku_views_v1", "native_canvas_size": list(native_size),
                "model_canvas_size": list(canvas.size), "patch_size": 16,
                "rotation": "none", "views": entries}
    return canvas, metadata


def unpack_miku_prediction(prediction, metadata):
    """Invert atlas resizing/crops to the original render camera frames."""
    if tuple(prediction.size) != tuple(metadata["model_canvas_size"]):
        raise ValueError(f"Prediction size {prediction.size} does not match condition "
                         f"{metadata['model_canvas_size']}")
    canvas = prediction.convert("RGB").resize(
        tuple(metadata["native_canvas_size"]), Image.Resampling.BILINEAR)
    views = []
    for entry in metadata["views"]:
        patch = canvas.crop(entry["atlas_box"])
        size = entry["view_size"]
        frame = Image.new("RGB", (size, size), "white")
        frame.paste(patch, entry["crop_box"][:2])
        views.append(frame.resize(tuple(entry["source_size"]), Image.Resampling.BILINEAR))
    return views


def load_miku_adapter_weights(pipeline, source, weight_name, local_files_only=True):
    """Load and verify both LoRA tensors and PEFT modules_to_save parameters.

    The supplied checkpoint includes a trained x_embedder.weight. Diffusers
    0.37's high-level Flux2 loader rejects that valid non-LoRA key, whereas its
    transformer loader supports it through the checkpoint's PEFT metadata.
    Exporting the loaded adapter and checking every key/value makes omissions,
    wrong metadata, random extra adapter parameters, and shape mismatches fatal.
    """
    import torch
    from peft.utils.save_and_load import get_peft_model_state_dict

    state, metadata = pipeline.lora_state_dict(
        source, weight_name=weight_name, local_files_only=local_files_only,
        return_lora_metadata=True)
    if not state or not all(key.startswith("transformer.") for key in state):
        raise ValueError("Miku adapter must contain nonempty transformer-prefixed weights only")
    expected = {key.removeprefix("transformer."): value for key, value in state.items()}
    full_weights = {key: value for key, value in expected.items() if "lora" not in key}
    if full_weights:
        modules_to_save = (metadata or {}).get("transformer.modules_to_save") or []
        base_state = pipeline.transformer.state_dict()
        for key, value in full_weights.items():
            module_name = key.rsplit(".", 1)[0]
            if not any(module_name == name or module_name.endswith("." + name) for name in modules_to_save):
                raise ValueError(f"Non-LoRA tensor {key!r} is not declared by checkpoint modules_to_save")
            if key not in base_state or tuple(base_state[key].shape) != tuple(value.shape):
                raise ValueError(f"Full trained tensor {key!r} is absent from the transformer or has a wrong shape")
    pipeline.load_lora_into_transformer(
        state, transformer=pipeline.transformer, adapter_name="texture",
        metadata=metadata, _pipeline=pipeline, low_cpu_mem_usage=False)
    actual = get_peft_model_state_dict(pipeline.transformer, adapter_name="texture")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(f"Adapter parameter mismatch after loading: missing={missing}, unexpected={unexpected}")
    for key, saved in expected.items():
        loaded = actual[key].detach().cpu()
        if tuple(saved.shape) != tuple(loaded.shape) or not torch.equal(
                loaded, saved.detach().to(device="cpu", dtype=loaded.dtype)):
            raise ValueError(f"Adapter tensor {key!r} was not restored exactly (after runtime dtype conversion)")
    report = {"tensor_count": len(expected), "full_parameter_keys": sorted(full_weights),
              "verified": True}
    print(f"Miku adapter: verified {len(expected)} tensors, full parameters={sorted(full_weights)}")
    return report


def activate_miku_adapter(transformer, scale):
    """Activate the custom transformer without diffusers' class-name registry."""
    from diffusers.utils.peft_utils import set_weights_and_activate_adapters
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils.other import ModulesToSaveWrapper

    if not math.isfinite(scale):
        raise ValueError("miku_lora_scale must be finite")
    set_weights_and_activate_adapters(transformer, ["texture"], [scale])
    lora_count, full_modules = 0, []
    for name, module in transformer.named_modules():
        if isinstance(module, ModulesToSaveWrapper) and "texture" in module.modules_to_save:
            # The generic diffusers helper handles BaseTunerLayer only. Full
            # trained modules such as x_embedder must be enabled explicitly.
            module.enable_adapters(True)
            module.set_adapter("texture")
            if list(module.active_adapters) != ["texture"] or module.disable_adapters:
                raise ValueError(f"Full trained module {name!r} is not active")
            full_modules.append(name)
        if isinstance(module, BaseTunerLayer) and "texture" in module.scaling:
            module.enable_adapters(True)
            rank = module.r["texture"]
            denominator = math.sqrt(rank) if module.use_rslora.get("texture", False) else rank
            expected_scale = scale * module.lora_alpha["texture"] / denominator
            if list(module.active_adapters) != ["texture"] or not math.isclose(
                    module.scaling["texture"], expected_scale, rel_tol=1e-7, abs_tol=1e-12):
                raise ValueError(f"LoRA module {name!r} is inactive or has an incorrect scale")
            lora_count += 1
    if lora_count == 0:
        raise ValueError("No active Miku LoRA layers were found after loading")
    transformer.requires_grad_(False)
    return {"lora_layers": lora_count, "full_modules": full_modules, "scale": scale}


def install_miku_block_compatibility(transformer):
    """Adapt only this model instance to diffusers' packed modulation API.

    The copied source uses the old block kwarg names and indexes the old
    single-modulation tuple. New diffusers returns one packed tensor and moves
    splitting into each block. Keep that complete tensor in a one-item tuple so
    the old ``[0]`` retains every batch element; unwrap it for direct base-model
    calls. No weights, global classes or mathematical operations are changed.
    """
    if hasattr(transformer, "_miku_runtime_compat_report"):
        return transformer._miku_runtime_compat_report
    handles = []
    double_count = single_count = 0

    def rename_double(module, args, kwargs):
        kwargs = dict(kwargs)
        for old, new in (("temb_mod_params_img", "temb_mod_img"),
                         ("temb_mod_params_txt", "temb_mod_txt")):
            if old in kwargs:
                if new in kwargs:
                    raise ValueError(f"Both legacy and current modulation kwargs supplied: {old}, {new}")
                kwargs[new] = kwargs.pop(old)
        return args, kwargs

    def rename_single(module, args, kwargs):
        kwargs = dict(kwargs)
        if "temb_mod_params" in kwargs:
            if "temb_mod" in kwargs:
                raise ValueError("Both legacy and current single modulation kwargs supplied")
            kwargs["temb_mod"] = kwargs.pop("temb_mod_params")
        # A base Flux2 forward does not index single_stream_modulation. Accept
        # its one-item container too, so both forwards see identical tensors.
        if "temb_mod" in kwargs and isinstance(kwargs["temb_mod"], tuple):
            if len(kwargs["temb_mod"]) != 1:
                raise ValueError("Expected exactly one packed single-stream modulation tensor")
            kwargs["temb_mod"] = kwargs["temb_mod"][0]
        elif len(args) > 2 and isinstance(args[2], tuple):
            if len(args[2]) != 1:
                raise ValueError("Expected one packed positional modulation tensor")
            args = (*args[:2], args[2][0], *args[3:])
        return args, kwargs

    for block in transformer.transformer_blocks:
        names = inspect.signature(block.forward).parameters
        if {"temb_mod_params_img", "temb_mod_params_txt"}.issubset(names):
            continue
        if not {"temb_mod_img", "temb_mod_txt"}.issubset(names):
            raise ValueError("Unsupported FLUX.2 double-stream block modulation signature")
        handles.append(block.register_forward_pre_hook(rename_double, with_kwargs=True))
        double_count += 1
    for block in transformer.single_transformer_blocks:
        names = inspect.signature(block.forward).parameters
        if "temb_mod_params" in names:
            continue
        if "temb_mod" not in names:
            raise ValueError("Unsupported FLUX.2 single-stream block modulation signature")
        handles.append(block.register_forward_pre_hook(rename_single, with_kwargs=True))
        single_count += 1
    if single_count:
        if single_count != len(transformer.single_transformer_blocks):
            raise ValueError("Mixed legacy/current single-stream blocks are unsupported")

        def preserve_modulation_batch(module, args, output):
            import torch
            if not torch.is_tensor(output):
                raise TypeError("Current FLUX.2 blocks require packed tensor modulation output")
            return (output,)

        handles.append(transformer.single_stream_modulation.register_forward_hook(preserve_modulation_batch))
    transformer._miku_runtime_compat_handles = handles
    transformer._miku_runtime_compat_report = {
        "double_stream_blocks": double_count, "single_stream_blocks": single_count,
        "preserve_single_modulation_batch": bool(single_count)}
    return transformer._miku_runtime_compat_report


class MikuMultiviewDiffusionNet:
    """The MaterialMVP multiview interface backed by FluxMVKleinPipeline."""

    def __init__(self, config):
        import torch

        self.config = config
        self.device = config.device
        self.view_size = config.resolution
        self.seed = int(getattr(config, "miku_seed", 0))
        self.steps = int(getattr(config, "miku_steps", 10))
        self.guidance_scale = float(getattr(config, "miku_guidance_scale", 2.0))
        if self.steps < 1:
            raise ValueError("miku_steps must be positive")
        dtype_name = getattr(config, "miku_dtype", "float16")
        if dtype_name not in ("float16", "bfloat16", "float32"):
            raise ValueError("miku_dtype must be float16, bfloat16 or float32")
        dtype = getattr(torch, dtype_name)
        try:
            from miku.pipeline_drop import FluxMVKleinPipeline
            from miku.drop_transformer import Flux2DropConditionModel
        except ImportError as exc:
            raise ImportError("Miku requires the FLUX.2 Klein-compatible runtime pinned in requirements.txt") from exc
        pretrained = config.miku_pretrained_path
        if not pretrained:
            raise ValueError("miku_pretrained_path must identify a FLUX.2 Klein base checkpoint")
        local_only = bool(getattr(config, "miku_local_files_only", True))
        transformer = Flux2DropConditionModel.from_pretrained(
            pretrained, subfolder="transformer", torch_dtype=dtype, local_files_only=local_only)
        self.block_compatibility_report = install_miku_block_compatibility(transformer)
        self.pipeline = FluxMVKleinPipeline.from_pretrained(
            pretrained, transformer=transformer, torch_dtype=dtype, local_files_only=local_only)
        if self.pipeline.vae_scale_factor * 2 != 16:
            raise ValueError("The copied Miku pipeline requires a 16-pixel latent token grid "
                             "and 4096 reference tokens; use the FLUX.2 Klein base checkpoint")
        lora_path = getattr(config, "miku_lora_path", None)
        if lora_path:
            lora = Path(lora_path).expanduser()
            if lora.is_file() or (not lora.is_dir() and lora.suffix.lower() in (".safetensors", ".bin")):
                lora_source, weight_name = str(lora.parent), lora.name
            else:
                lora_source = str(lora) if lora.exists() else str(lora_path)
                weight_name = getattr(self.config, "miku_lora_weight_name", "pytorch_lora_weights.safetensors")
            # diffusers' offline loader cannot guess weight_name, including
            # when its first argument points directly to a safetensors file.
            self.adapter_report = load_miku_adapter_weights(
                self.pipeline, lora_source, weight_name=weight_name, local_files_only=local_only)
            self.adapter_activation_report = activate_miku_adapter(
                self.pipeline.transformer, float(getattr(config, "miku_lora_scale", 1.0)))
        # Preserve the pretrained scheduler semantics.
        offload = getattr(config, "miku_offload", "model")
        if offload in (None, "none"):
            self.pipeline.to(self.device)
        elif offload in ("model", "sequential"):
            device = torch.device(self.device)
            if device.type != "cuda":
                raise ValueError("Miku CPU offload requires a CUDA execution device")
            method = (self.pipeline.enable_model_cpu_offload if offload == "model"
                      else self.pipeline.enable_sequential_cpu_offload)
            method(gpu_id=device.index if device.index is not None else 0, device="cuda")
        else:
            raise ValueError("miku_offload must be 'none', 'model' or 'sequential'")
        for name in ("transformer", "vae", "text_encoder"):
            module = getattr(self.pipeline, name, None)
            if module is not None:
                module.eval()
                if hasattr(module, "requires_grad_"):
                    module.requires_grad_(False)
        self.last_metadata = None
        self.last_prediction = None

    def close(self):
        """Release FLUX before the following PBR model is loaded."""
        import gc
        import torch

        pipeline, self.pipeline = self.pipeline, None
        if pipeline is not None and hasattr(pipeline, "remove_all_hooks"):
            pipeline.remove_all_hooks()
        del pipeline
        self.last_prediction = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(self, images, conditions, prompt=None, custom_view_size=None,
                 resize_input=False, masks=None):
        import torch

        if isinstance(images, Image.Image):
            images = [images]
        if len(images) != 1:
            raise ValueError("This Miku checkpoint expects exactly one reference image")
        if not conditions or len(conditions) % 2:
            raise ValueError("conditions must be all normal maps followed by all position maps")
        prompt = prompt if prompt is not None else getattr(self.config, "miku_prompt", DEFAULT_PROMPT)
        if not isinstance(prompt, str):
            raise ValueError("Miku supports one string prompt per single-reference inference")
        count = len(conditions) // 2
        control, metadata = pack_miku_conditions(
            conditions[:count], conditions[count:], masks=masks,
            view_size=custom_view_size or self.view_size,
            max_canvas_size=int(getattr(self.config, "miku_max_canvas_size", 2048)))
        # The copied pipeline hardcodes 4096 reference tokens in its negative
        # branch. 1024x1024 / 16x16 = 4096; keep the eval.py reference size.
        reference = prepare_miku_reference(
            images[0], mode=getattr(self.config, "miku_reference_resize_mode", "fit"))
        self.last_metadata = metadata
        debug_dir = getattr(self.config, "miku_debug_dir", None)
        if debug_dir:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
            reference.save(debug_dir / "reference.png")
            control.save(debug_dir / "control.png")
            (debug_dir / "packing.json").write_text(json.dumps(metadata, indent=2) + "\n")
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        with torch.inference_mode():
            prediction = self.pipeline(
                [reference, control],
                prompt=prompt,
                height=control.height, width=control.width,
                guidance_scale=self.guidance_scale, num_inference_steps=self.steps,
                generator=generator).images[0]
        # Keep the native generated atlas for the downstream MR model so it
        # does not need to unpack and resample these views a second time.
        self.last_prediction = prediction
        views = unpack_miku_prediction(prediction, metadata)
        if debug_dir:
            prediction.save(debug_dir / "generated.png")
            for index, view in enumerate(views):
                view.save(debug_dir / f"view_{index:02d}.png")
        return {"albedo": views}

    forward_one = __call__
