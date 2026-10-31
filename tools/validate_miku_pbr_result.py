"""Check exported Miku PBR geometry, embedded textures, and prediction provenance.

This is an execution/integration check, not a material accuracy metric. It uses
CPU only and writes a six-view albedo/roughness/metallic contact sheet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import trimesh


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def glb_document(path):
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    require(magic == b"glTF" and version == 2 and length == len(data), "Invalid GLB header")
    chunk_length, chunk_type = struct.unpack_from("<I4s", data, 12)
    require(chunk_type == b"JSON", "GLB first chunk is not JSON")
    return json.loads(data[20:20 + chunk_length])


def source_mask(debug, index):
    """Read the renderer's foreground convention independently of the adapter."""
    with Image.open(debug / f"masks_{index:02d}.png") as image:
        binary = (np.asarray(image.convert("L")) > 0).astype(np.uint8) * 255
    return Image.fromarray(binary)


def verify_background_mask(debug, packing):
    """Rebuild the MR model mask from camera masks and recorded atlas geometry."""
    model_size = tuple(packing["model_canvas_size"])
    expected = Image.new("L", model_size, 0)
    if packing["mode"] == "nonpack":
        require(model_size == (1536, 1024), "Nonpack mask has an invalid canvas size")
        for index, entry in enumerate(packing["views"]):
            mask = source_mask(debug, index)
            require(list(mask.size) == entry["source_size"], "Source mask and camera frame sizes differ")
            expected.paste(mask.resize((512, 512), Image.Resampling.NEAREST),
                           (index % 3 * 512, index // 3 * 512))
    elif packing["mode"] == "pack":
        original = packing["albedo_packing"]
        native = Image.new("L", tuple(original["native_canvas_size"]), 0)
        for index, entry in enumerate(original["views"]):
            mask = source_mask(debug, index)
            require(list(mask.size) == entry["source_size"], "Source mask and camera frame sizes differ")
            mask = mask.resize((entry["view_size"], entry["view_size"]), Image.Resampling.NEAREST)
            patch = mask.crop(entry["crop_box"])
            box = entry["atlas_box"]
            require(patch.size == (box[2] - box[0], box[3] - box[1]), "Packed mask crop dimensions differ")
            native.paste(patch, box[:2])
        # Match both recorded scaling steps. Combining them into one resize
        # changes nearest-neighbour samples when either scale is nonintegral.
        original_mask = native.resize(tuple(original["model_canvas_size"]), Image.Resampling.NEAREST)
        half = original_mask.resize(tuple(packing["scaled_canvas_size"]), Image.Resampling.NEAREST)
        expected.paste(half, (0, 0))
    else:
        raise ValueError(f"Unknown PBR mask mode: {packing['mode']}")
    with Image.open(debug / "pbr" / "background_mask.png") as actual_image:
        require(actual_image.size == model_size, "Saved model mask dimensions differ from the MR input")
        actual = np.asarray(actual_image.convert("L"))
    expected_array = np.asarray(expected)
    require(np.array_equal(actual, expected_array), "Saved MR mask differs from the camera-mask mapping")
    require(np.all((actual == 0) | (actual == 255)), "MR background mask must be binary 0/255")
    return expected_array > 0


def verify_view_mapping(debug, packing, quantized, background_white=False):
    """Reconstruct both user-requested layouts from the saved albedo images."""
    mode = packing["mode"]
    model_size = tuple(packing["model_canvas_size"])
    prediction = Image.fromarray(quantized)
    require(prediction.size == model_size, "Raw prediction dimensions disagree with packing")
    expected_views = []
    if mode == "pack":
        albedo_packing = packing["albedo_packing"]
        with Image.open(debug / "generated.png") as source:
            source = source.convert("RGB")
            require(source.size == tuple(albedo_packing["model_canvas_size"]),
                    "PBR input does not refer to the saved albedo atlas")
            half_size = tuple(max(1, round(side * .5)) for side in source.size)
            require(half_size == tuple(packing["scaled_canvas_size"]),
                    "Pack mode must halve both albedo atlas dimensions")
            require(packing["output_crop_box"] == [0, 0, *half_size], "Unexpected pack crop")
            pad_right, pad_bottom = (-half_size[0]) % 16, (-half_size[1]) % 16
            require(packing["padding_ltrb"] == [0, 0, pad_right, pad_bottom],
                    "Only right/bottom token-alignment padding is allowed")
            require(model_size == (half_size[0] + pad_right, half_size[1] + pad_bottom),
                    "Pack model canvas dimensions do not match half-size plus padding")
            prepared = Image.new("RGB", model_size, "white")
            prepared.paste(source.resize(half_size, Image.Resampling.BILINEAR), (0, 0))
        require(packing["views"] == albedo_packing["views"], "Pack mode changed albedo view placement")
        original_atlas = prediction.crop(packing["output_crop_box"]).resize(
            tuple(albedo_packing["model_canvas_size"]), Image.Resampling.BILINEAR)
        native = original_atlas.resize(tuple(albedo_packing["native_canvas_size"]),
                                       Image.Resampling.BILINEAR)
        for entry in albedo_packing["views"]:
            frame = Image.new("RGB", (entry["view_size"], entry["view_size"]), "white")
            frame.paste(native.crop(entry["atlas_box"]), entry["crop_box"][:2])
            expected_views.append(frame.resize(tuple(entry["source_size"]), Image.Resampling.BILINEAR))
    elif mode == "nonpack":
        require(model_size == (1536, 1024), "Nonpack mode must be a 3x2 grid of 512-square views")
        prepared = Image.new("RGB", model_size, "white")
        for i, entry in enumerate(packing["views"]):
            x, y = i % 3 * 512, i // 3 * 512
            require(entry["atlas_box"] == [x, y, x + 512, y + 512], "Nonpack view order changed")
            require(entry["crop_box"] == [0, 0, 512, 512] and entry["view_size"] == 512,
                    "Nonpack mode must retain the whole camera frame")
            with Image.open(debug / f"view_{i:02d}.png") as source:
                prepared.paste(source.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR), (x, y))
            expected_views.append(prediction.crop(entry["atlas_box"]).resize(
                tuple(entry["source_size"]), Image.Resampling.BILINEAR))
    else:
        raise ValueError(f"Unknown PBR mode: {mode}")
    require(np.array_equal(rgb(debug / "pbr" / "input_condition.png"), np.asarray(prepared)),
            "PBR input differs from the requested layout/resize or its background was changed")
    for i, view in enumerate(expected_views):
        expected = np.asarray(view)
        if background_white:
            mask = np.asarray(source_mask(debug, i)) > 0
            require(mask.shape == expected.shape[:2], "Restored view and original mask sizes differ")
            # Whiten after inverse mapping, preserving every interpolated
            # foreground pixel from the raw prediction at original resolution.
            expected = np.where(mask[..., None], expected, 255).astype(np.uint8)
        require(np.array_equal(rgb(debug / "pbr" / f"mr_{i:02d}.png"), expected),
                f"MR inverse mapping changed pixels or orientation in view {i}")


def verify(args):
    source = trimesh.load(args.mesh, force="mesh", process=False)
    exported = trimesh.load(args.output, force="mesh", process=False)
    require(len(source.faces) == len(exported.faces), "Face count changed")
    require(np.isfinite(exported.vertices).all(), "Nonfinite mesh vertices")
    distance, _ = cKDTree(source.vertices).query(exported.vertices)
    bounds_error = float(np.max(np.abs(source.bounds - exported.bounds)))
    require(float(distance.max()) < 1e-5 and bounds_error < 1e-5, "Geometry position or bounds changed")
    uv = np.asarray(exported.visual.uv)
    require(uv.shape == (len(exported.vertices), 2) and np.isfinite(uv).all(), "Invalid exported UVs")
    material = exported.visual.material
    require(material.baseColorTexture is not None, "GLB has no albedo texture")
    require(material.metallicRoughnessTexture is not None, "GLB has no metallic/roughness texture")
    require(float(material.metallicFactor) == 1 and float(material.roughnessFactor) == 1,
            "GLB must use unit metallic and roughness factors")
    embedded_base = np.asarray(material.baseColorTexture.convert("RGB"))
    embedded_mr = np.asarray(material.metallicRoughnessTexture.convert("RGB"))
    require(embedded_base.shape == embedded_mr.shape == (args.texture_size, args.texture_size, 3),
            "Unexpected embedded texture resolution")
    stem = args.output.with_suffix("")
    sidecar_base = rgb(Path(str(stem) + ".png"))
    sidecar_mr = rgb(Path(str(stem) + "_metallic_roughness.png"))
    metallic = rgb(Path(str(stem) + "_metallic.png"))[..., 0]
    roughness = rgb(Path(str(stem) + "_roughness.png"))[..., 0]
    require(np.array_equal(embedded_base, sidecar_base), "Embedded albedo differs from PNG")
    require(np.array_equal(embedded_mr, sidecar_mr), "Embedded MR differs from PNG")
    require(np.array_equal(embedded_mr[..., 1], roughness), "GLB G is not the roughness PNG")
    require(np.array_equal(embedded_mr[..., 2], metallic), "GLB B is not the metallic PNG")

    document = glb_document(args.output)
    require(len(document["materials"]) == 1, "Expected a single baked material")
    gltf_material = document["materials"][0]
    require("occlusionTexture" not in gltf_material, "MR prediction must not be treated as occlusion")
    pbr = gltf_material["pbrMetallicRoughness"]
    for key in ("baseColorTexture", "metallicRoughnessTexture"):
        texture = document["textures"][pbr[key]["index"]]
        image = document["images"][texture["source"]]
        require(image.get("mimeType") == "image/png" and "bufferView" in image,
                f"{key} must be an embedded lossless PNG")

    debug = args.debug_dir or Path(str(stem) + "_debug")
    pbr_debug = debug / "pbr"
    inference = json.loads((pbr_debug / "inference_report.json").read_text())
    require(inference.get("status") == "passed" and inference.get("dino_linear_verified") is True,
            "PBR checkpoint or DINO restoration was not verified")
    require(inference["denoising_finite_steps"] == inference["steps"] > 0,
            "PBR denoising did not finish every step with finite values")
    require(inference["checkpoint_load"]["verified_tensor_count"] > 0,
            "No PBR adapter tensors were verified")
    packing = json.loads((pbr_debug / "packing.json").read_text())
    cameras = json.loads((debug / "cameras.json").read_text())
    prediction = np.load(pbr_debug / "prediction_float32.npy", allow_pickle=False)
    require(prediction.ndim == 3 and prediction.shape[0] == 3 and np.isfinite(prediction).all(),
            "Raw MR prediction has an invalid shape or nonfinite values")
    quantized = np.rint(np.clip(prediction.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
    background_white = inference.get("background_white", False)
    require(isinstance(background_white, bool), "background_white must be a boolean")
    raw_png = pbr_debug / "predicted_orm_raw.png"
    if background_white or raw_png.is_file():
        require(raw_png.is_file(), "White-background mode must preserve the raw ORM PNG")
        require(np.array_equal(rgb(raw_png), quantized), "Raw ORM PNG differs from raw float prediction")
    processed = quantized
    if background_white:
        foreground = verify_background_mask(debug, packing)
        require(foreground.shape == quantized.shape[:2], "MR mask and raw prediction sizes differ")
        processed = np.where(foreground[..., None], quantized, 255).astype(np.uint8)
    require(np.array_equal(rgb(pbr_debug / "predicted_orm.png"), processed),
            "ORM PNG differs from the expected raw prediction and background setting")
    require(np.array_equal(rgb(pbr_debug / "predicted_roughness.png")[..., 0], processed[..., 1]),
            "Roughness prediction channel changed")
    require(np.array_equal(rgb(pbr_debug / "predicted_metallic.png")[..., 0], processed[..., 2]),
            "Metallic prediction channel changed")
    require(cameras["render_size"] == args.render_size, "Unexpected rendering resolution")
    require(cameras["coordinate_system"] == "miku", "Unexpected coordinate system")
    views = packing["views"]
    require(len(views) == len(cameras["azimuths"]) == len(cameras["elevations"]) == 6,
            "Expected six matching cameras and MR views")
    require([v["view_index"] for v in views] == list(range(6)), "MR view order changed")
    require(packing["rotation"] == "none", "MR packing rotated a view")
    verify_view_mapping(debug, packing, quantized, background_white=background_white)

    sheet = Image.new("RGB", (6 * 256, 3 * 292), "white")
    draw = ImageDraw.Draw(sheet)
    foreground_statistics = []
    for i in range(6):
        mask = rgb(debug / f"masks_{i:02d}.png")[..., 0] > 0
        albedo = rgb(debug / f"view_{i:02d}.png")
        mr = rgb(pbr_debug / f"mr_{i:02d}.png")
        require(albedo.shape == mr.shape == (args.render_size, args.render_size, 3),
                f"Unexpected camera frame shape for view {i}")
        require(mask.shape == mr.shape[:2] and mask.any(), f"Invalid camera mask {i}")
        foreground_statistics.append({
            "view_index": i,
            "roughness_min_max": [int(mr[..., 1][mask].min()), int(mr[..., 1][mask].max())],
            "metallic_min_max": [int(mr[..., 2][mask].min()), int(mr[..., 2][mask].max())],
        })
        for row, (label, values) in enumerate((
                ("Albedo", albedo),
                ("Roughness", np.repeat(mr[..., 1:2], 3, axis=-1)),
                ("Metallic", np.repeat(mr[..., 2:3], 3, axis=-1)))):
            display = np.where(mask[..., None], values, 255).astype(np.uint8)
            tile = Image.fromarray(display).resize((256, 256), Image.Resampling.LANCZOS)
            draw.text((i * 256 + 6, row * 292 + 10), f"{label} | view {i}", fill="black")
            sheet.paste(tile, (i * 256, row * 292 + 36))
    report = {
        "status": "passed",
        "scope": "Geometry, foreground prediction preservation, configured background and GLB integration; no material accuracy claim",
        "source_mesh": str(args.mesh), "output_glb": str(args.output),
        "checkpoint": inference.get("checkpoint"),
        "checkpoint_load": inference["checkpoint_load"],
        "denoising_finite_steps": inference["denoising_finite_steps"],
        "pbr_mode": packing["mode"],
        "faces": len(exported.faces), "vertices": len(exported.vertices),
        "max_source_vertex_distance": float(distance.max()), "max_bounds_error": bounds_error,
        "render_size": args.render_size, "texture_size": args.texture_size,
        "raw_prediction_shape_chw": list(prediction.shape),
        "prediction_channels_preserved": not background_white,
        "foreground_prediction_channels_preserved": True,
        "embedded_textures_exactly_match_png": True,
        "input_layout_and_inverse_mapping_exact": True,
        "background_white": background_white,
        "background_mask_and_white_pixels_verified": background_white,
        "metallic_roughness_channels": {"G": "roughness", "B": "metallic"},
        "metallic_factor": float(material.metallicFactor), "roughness_factor": float(material.roughnessFactor),
        "occlusion_texture": False, "views": foreground_statistics,
        "glb_bytes": args.output.stat().st_size,
    }
    destination = args.report_dir or args.output.parent
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "pbr_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    sheet.save(destination / "pbr_predicted_views.png")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Exported GLB")
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--render-size", type=int, default=1024)
    parser.add_argument("--texture-size", type=int, default=2048)
    verify(parser.parse_args())


if __name__ == "__main__":
    main()
