"""Real OBJ/GLB readback checks for linear, lossless metallic/roughness maps."""

import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh

from DifferentiableRenderer.mesh_utils import convert_obj_to_glb, save_mesh


def read_glb(path):
    data = Path(path).read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data)
    assert magic == b"glTF" and version == 2 and length == len(data)
    offset, document, binary = 12, None, None
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8:offset + 8 + chunk_length]
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk)
        elif chunk_type == 0x004E4942:
            binary = chunk
        offset += 8 + chunk_length
    return document, binary


class PBRExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.obj = Path(self.temp.name) / "sample.obj"
        self.glb = self.obj.with_suffix(".glb")
        self.vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        self.faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        self.uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        self.albedo = np.full((3, 4, 3), 0.7, dtype=np.float32)
        self.metallic_bytes = np.array([[0, 17, 67, 255], [11, 87, 201, 3], [233, 19, 101, 29]], dtype=np.uint8)
        self.roughness_bytes = np.array([[240, 97, 29, 1], [31, 211, 13, 71], [7, 161, 47, 189]], dtype=np.uint8)
        self.metallic = np.repeat(self.metallic_bytes[:, :, None], 3, axis=2).astype(np.float32) / 255
        self.roughness = np.repeat(self.roughness_bytes[:, :, None], 3, axis=2).astype(np.float32) / 255

    def save(self, with_mr=True, image_format=".png"):
        save_mesh(str(self.obj), self.vertices, self.faces, self.uv, self.faces, self.albedo,
                  metallic=self.metallic if with_mr else None,
                  roughness=self.roughness if with_mr else None, image_format=image_format)
        self.assertTrue(convert_obj_to_glb(str(self.obj), str(self.glb), backend="trimesh"))

    def test_png_channels_survive_glb_embedding_without_gamma_or_flip(self):
        self.save()
        np.testing.assert_array_equal(np.asarray(Image.open(self.obj.with_name("sample_metallic.png"))),
                                      self.metallic_bytes)
        np.testing.assert_array_equal(np.asarray(Image.open(self.obj.with_name("sample_roughness.png"))),
                                      self.roughness_bytes)
        document, binary = read_glb(self.glb)
        material = document["materials"][0]
        pbr = material["pbrMetallicRoughness"]
        self.assertEqual(pbr["metallicFactor"], 1.0)
        self.assertEqual(pbr["roughnessFactor"], 1.0)
        self.assertNotIn("occlusionTexture", material)
        texture = document["textures"][pbr["metallicRoughnessTexture"]["index"]]
        image_info = document["images"][texture["source"]]
        self.assertEqual(image_info["mimeType"], "image/png")
        view = document["bufferViews"][image_info["bufferView"]]
        start = view.get("byteOffset", 0)
        packed = np.asarray(Image.open(io.BytesIO(binary[start:start + view["byteLength"]])))
        np.testing.assert_array_equal(packed[:, :, 0], 255)
        np.testing.assert_array_equal(packed[:, :, 1], self.roughness_bytes)
        np.testing.assert_array_equal(packed[:, :, 2], self.metallic_bytes)
        loaded = trimesh.load(self.glb, force="mesh", process=False)
        source = trimesh.load(self.obj, force="mesh", process=False)
        np.testing.assert_allclose(loaded.vertices, source.vertices)
        np.testing.assert_allclose(loaded.visual.uv, source.visual.uv)

    def test_albedo_only_rerun_ignores_stale_mr_sidecars(self):
        self.save()
        self.save(with_mr=False)
        self.assertTrue(self.obj.with_name("sample_metallic.png").exists())
        document, _ = read_glb(self.glb)
        pbr = document["materials"][0]["pbrMetallicRoughness"]
        self.assertEqual(pbr["metallicFactor"], 0.0)
        self.assertEqual(pbr["roughnessFactor"], 1.0)
        self.assertNotIn("metallicRoughnessTexture", pbr)

    def test_legacy_obj_still_uses_jpg_maps(self):
        self.save(image_format=".jpg")
        self.assertTrue(self.obj.with_suffix(".jpg").exists())
        mtl = self.obj.with_suffix(".mtl").read_text()
        self.assertIn("map_Pm sample_metallic.jpg", mtl)
        self.assertIn("map_Pr sample_roughness.jpg", mtl)


if __name__ == "__main__":
    unittest.main()
