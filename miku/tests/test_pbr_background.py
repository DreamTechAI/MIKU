"""Check geometry-mask mapping and exact foreground preservation on both layouts."""
import unittest

import numpy as np
from PIL import Image

from utils.miku_pbr_utils import (
    postprocess_pbr_prediction, prepare_pbr_atlas, prepare_pbr_mask,
    unpack_pbr_prediction,
)


class PBRBackgroundTests(unittest.TestCase):
    @staticmethod
    def masks(count, size=(64, 48)):
        masks = []
        for i in range(count):
            mask = Image.new("L", size, 0)
            mask.paste(1 if i % 2 else 255, (7 + i, 11, 34 + i, 37))
            mask.paste(255, (43, 17 + i, 49, 20 + i))
            masks.append(mask)
        return masks

    @staticmethod
    def raw(size):
        x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        return Image.fromarray(np.stack((x % 211, y % 197, (x + y) % 239), -1).astype(np.uint8))

    def check_foreground_and_background(self, metadata, masks):
        raw = self.raw(tuple(metadata["model_canvas_size"]))
        original = np.asarray(raw).copy()
        baseline = unpack_pbr_prediction(raw, metadata)
        orm, views, model_mask = postprocess_pbr_prediction(raw, metadata, masks)
        foreground = np.asarray(model_mask) > 0
        np.testing.assert_array_equal(np.asarray(orm)[foreground], original[foreground])
        self.assertTrue(np.all(np.asarray(orm)[~foreground] == 255))
        for view, expected, mask in zip(views, baseline, masks):
            fg = np.asarray(mask) > 0
            np.testing.assert_array_equal(np.asarray(view)[fg], np.asarray(expected)[fg])
            self.assertTrue(np.all(np.asarray(view)[~fg] == 255))
        np.testing.assert_array_equal(np.asarray(raw), original)
        disabled, unmasked_views, disabled_mask = postprocess_pbr_prediction(
            raw, metadata, background_white=False)
        np.testing.assert_array_equal(np.asarray(disabled), original)
        self.assertIsNone(disabled_mask)
        for view, expected in zip(unmasked_views, baseline):
            np.testing.assert_array_equal(np.asarray(view), np.asarray(expected))
        return model_mask

    def test_nonpack_mask_grid_and_final_foreground_are_exact(self):
        masks = self.masks(6)
        _, metadata = prepare_pbr_atlas([Image.new("RGB", mask.size) for mask in masks], mode="nonpack")
        model_mask = self.check_foreground_and_background(metadata, masks)
        for i, mask in enumerate(masks):
            binary = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
            expected = binary.resize((512, 512), Image.Resampling.NEAREST)
            x, y = i % 3 * 512, i // 3 * 512
            np.testing.assert_array_equal(np.asarray(model_mask)[y:y + 512, x:x + 512], expected)

    def test_pack_mask_crop_two_resizes_padding_and_original_frame(self):
        masks = self.masks(2)
        views = [Image.new("RGB", mask.size) for mask in masks]
        original = {
            "format": "miku_views_v1", "rotation": "none",
            "native_canvas_size": [104, 64], "model_canvas_size": [272, 304],
            "views": [
                {"view_index": 0, "source_size": [64, 48], "view_size": 64,
                 "crop_box": [8, 16, 48, 56], "atlas_box": [3, 7, 43, 47]},
                {"view_index": 1, "source_size": [64, 48], "view_size": 64,
                 "crop_box": [4, 8, 52, 56], "atlas_box": [49, 11, 97, 59]},
            ],
        }
        _, metadata = prepare_pbr_atlas(views, mode="pack", packed_albedo=Image.new("RGB", (272, 304)),
                                        packing_metadata=original)
        actual = self.check_foreground_and_background(metadata, masks)
        native = Image.new("L", (104, 64), 0)
        for mask, crop, offset in zip(masks, [(8, 16, 48, 56), (4, 8, 52, 56)], [(3, 7), (49, 11)]):
            binary = Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255)
            native.paste(binary.resize((64, 64), Image.Resampling.NEAREST).crop(crop), offset)
        expected = native.resize((272, 304), Image.Resampling.NEAREST).resize((136, 152), Image.Resampling.NEAREST)
        self.assertEqual(actual.size, (144, 160))
        np.testing.assert_array_equal(np.asarray(actual)[:152, :136], expected)
        self.assertTrue(np.all(np.asarray(actual)[152:, :] == 0))
        self.assertTrue(np.all(np.asarray(actual)[:, 136:] == 0))

    def test_missing_or_misaligned_masks_fail_and_empty_foreground_is_white(self):
        views = [Image.new("RGB", (64, 48)) for _ in range(6)]
        _, metadata = prepare_pbr_atlas(views, mode="nonpack")
        for bad in (None, self.masks(5), [Image.new("L", (32, 32))] * 6):
            with self.assertRaises(ValueError):
                prepare_pbr_mask(bad, metadata)
        raw = self.raw((1536, 1024))
        orm, restored, mask = postprocess_pbr_prediction(raw, metadata, [Image.new("L", (64, 48))] * 6)
        self.assertTrue(np.all(np.asarray(mask) == 0))
        for image in [orm, *restored]:
            self.assertTrue(np.all(np.asarray(image) == 255))


if __name__ == "__main__":
    unittest.main()
