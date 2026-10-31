"""CPU-only checks for packing/inversion; no model weights or CUDA required.

Run from the repository root: python -m unittest discover -s miku/tests
"""

import unittest

import numpy as np
from PIL import Image

from utils.miku_multiview_utils import pack_miku_conditions, unpack_miku_prediction


class PackingTests(unittest.TestCase):
    def make_views(self, count=6, size=96):
        normals, positions, masks = [], [], []
        y, x = np.mgrid[:size, :size]
        for index in range(count):
            # Distinct colors and asymmetric xy ramps reveal both view swaps
            # and any accidental rotation or mirroring during atlas inversion.
            normal = np.stack([x * 2, y, np.full_like(x, 20 + index * 30)], axis=-1).astype(np.uint8)
            position = np.stack([x, y * 2, np.full_like(x, 10 + index * 20)], axis=-1).astype(np.uint8)
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[5 + index:70 - index * 2, 17:90 - index * 3] = 255
            normals.append(Image.fromarray(normal))
            positions.append(Image.fromarray(position))
            masks.append(Image.fromarray(mask))
        return normals, positions, masks

    def test_mix_and_inverse_preserve_view_order_and_orientation(self):
        normals, positions, masks = self.make_views()
        control, metadata = pack_miku_conditions(normals, positions, masks, view_size=96)
        self.assertEqual(control.width % 16, 0)
        self.assertEqual(control.height % 16, 0)
        self.assertEqual(metadata["rotation"], "none")
        restored = unpack_miku_prediction(control, metadata)
        for normal, position, output, entry in zip(normals, positions, restored, metadata["views"]):
            expected = ((np.asarray(normal, dtype=np.float32) +
                         np.asarray(position, dtype=np.float32)) * 0.5).astype(np.uint8)
            x0, y0, x1, y1 = entry["crop_box"]
            actual = np.asarray(output)
            np.testing.assert_array_equal(actual[y0:y1, x0:x1], expected[y0:y1, x0:x1])
            self.assertEqual(output.size, normal.size)
            outside = np.ones(actual.shape[:2], dtype=bool)
            outside[y0:y1, x0:x1] = False
            self.assertTrue(np.all(actual[outside] == 255))

    def test_downscaled_canvas_restores_original_render_size(self):
        normals, positions, masks = self.make_views()
        control, metadata = pack_miku_conditions(normals, positions, masks, view_size=192,
                                                 max_canvas_size=128)
        self.assertLessEqual(max(control.size), 128)
        self.assertEqual(control.width % 16, 0)
        self.assertEqual(control.height % 16, 0)
        restored = unpack_miku_prediction(control, metadata)
        self.assertEqual(len(restored), len(normals))
        self.assertTrue(all(image.size == (96, 96) for image in restored))

    def test_no_mask_keeps_full_frame_and_rgba_uses_alpha(self):
        normals, positions, masks = self.make_views(count=1)
        _, whole = pack_miku_conditions(normals, positions, view_size=96)
        self.assertEqual(whole["views"][0]["crop_box"], [0, 0, 96, 96])
        rgba = normals[0].convert("RGBA")
        rgba.putalpha(masks[0])
        _, cropped = pack_miku_conditions([rgba], positions, view_size=96)
        self.assertNotEqual(cropped["views"][0]["crop_box"], [0, 0, 96, 96])

    def test_bad_inputs_fail_before_inference(self):
        normals, positions, masks = self.make_views(count=1)
        with self.assertRaisesRegex(ValueError, "equal nonempty"):
            pack_miku_conditions(normals, [])
        with self.assertRaisesRegex(ValueError, "multiple of 16"):
            pack_miku_conditions(normals, positions, view_size=95)
        with self.assertRaisesRegex(ValueError, "no visible mesh"):
            pack_miku_conditions(normals, positions, [Image.new("L", (96, 96))], view_size=96)
        control, metadata = pack_miku_conditions(normals, positions, masks, view_size=96)
        with self.assertRaisesRegex(ValueError, "Prediction size"):
            unpack_miku_prediction(control.resize((16, 16)), metadata)


if __name__ == "__main__":
    unittest.main()
