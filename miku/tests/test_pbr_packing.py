"""CPU geometry checks for both MR layouts and the no-crop reference policy."""
import unittest

import numpy as np
from PIL import Image

from utils.miku_multiview_utils import pack_miku_conditions
from utils.miku_pbr_utils import prepare_pbr_atlas, prepare_pbr_reference, unpack_pbr_prediction


class PBRPackingTests(unittest.TestCase):
    @staticmethod
    def views(size=1024):
        result = []
        for index in range(6):
            # Corner colors deliberately distinguish horizontal/vertical flips
            # and view swaps, independent of packing implementation details.
            image = Image.new("RGB", (size, size), (30 * index, 17, 31))
            image.paste((30 * index, 51, 91), (0, 0, size // 2, size // 2))
            image.paste((30 * index, 151, 191), (size // 2, size // 2, size, size))
            result.append(image)
        return result

    def test_nonpack_is_six_full_frames_in_row_major_grid(self):
        views = self.views()
        atlas, metadata = prepare_pbr_atlas(views, mode="nonpack")
        self.assertEqual(atlas.size, (1536, 1024))
        restored = unpack_pbr_prediction(atlas, metadata)
        self.assertEqual([view.size for view in restored], [(1024, 1024)] * 6)
        for index, view in enumerate(restored):
            self.assertEqual(atlas.getpixel((index % 3 * 512 + 64, index // 3 * 512 + 64)),
                             (30 * index, 51, 91))
            self.assertEqual(view.getpixel((128, 128)), (30 * index, 51, 91))
            self.assertEqual(view.getpixel((896, 896)), (30 * index, 151, 191))
            self.assertEqual(view.getpixel((896, 128)), (30 * index, 17, 31))

    def test_pack_uses_half_original_atlas_without_background_edit(self):
        views = self.views(size=64)
        packed, original = pack_miku_conditions(views, views, view_size=64)
        # An actual generated atlas can have colored background: preserve it,
        # even if the caller supplies all-zero geometry masks.
        packed = packed.resize((272, 304), Image.Resampling.BILINEAR)
        original["model_canvas_size"] = [272, 304]
        masks = [Image.new("L", view.size) for view in views]
        atlas, metadata = prepare_pbr_atlas(views, masks=masks, mode="pack",
                                           packed_albedo=packed, packing_metadata=original)
        self.assertEqual(atlas.size, (144, 160))
        self.assertEqual(metadata["scaled_canvas_size"], [136, 152])
        self.assertEqual(metadata["padding_ltrb"], [0, 0, 8, 8])
        np.testing.assert_array_equal(np.asarray(atlas)[:152, :136],
                                      np.asarray(packed.resize((136, 152), Image.Resampling.BILINEAR)))
        self.assertTrue(np.all(np.asarray(atlas)[152:, :] == 255))
        self.assertTrue(np.all(np.asarray(atlas)[:, 136:] == 255))
        restored = unpack_pbr_prediction(atlas, metadata)
        for index, view in enumerate(restored):
            self.assertEqual(view.size, (64, 64))
            self.assertEqual(view.getpixel((12, 12)), (30 * index, 51, 91))
            self.assertEqual(view.getpixel((52, 52)), (30 * index, 151, 191))
        self.assertNotIn("albedo_packing", original)  # metadata not mutated

    def test_nonpack_never_uses_mask_to_crop_white_or_dark_material(self):
        views = self.views(size=64)
        masks = [Image.new("L", view.size) for view in views]
        actual, _ = prepare_pbr_atlas(views, masks, mode="nonpack")
        expected, _ = prepare_pbr_atlas(views, mode="nonpack")
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_reference_preserves_transparent_margin_and_whole_image(self):
        reference = Image.new("RGBA", (80, 40), (255, 255, 255, 0))
        reference.paste((12, 34, 56, 255), (20, 10, 40, 30))
        prepared = prepare_pbr_reference(reference, 512)
        self.assertEqual(prepared.size, (512, 512))
        self.assertEqual(prepared.getpixel((190, 256)), (12, 34, 56))
        self.assertEqual(prepared.getpixel((64, 256)), (255, 255, 255))
        self.assertEqual(prepared.getpixel((400, 256)), (255, 255, 255))

    def test_pack_requires_original_atlas_and_ordered_mapping(self):
        views = self.views(size=64)
        with self.assertRaisesRegex(ValueError, "original packed_albedo"):
            prepare_pbr_atlas(views, mode="pack")
        packed, metadata = pack_miku_conditions(views, views, view_size=64)
        metadata["views"][0]["view_index"] = 5
        with self.assertRaisesRegex(ValueError, "original camera mapping"):
            prepare_pbr_atlas(views, packed_albedo=packed, packing_metadata=metadata)
        with self.assertRaisesRegex(ValueError, "exactly six"):
            prepare_pbr_atlas(views[:5], mode="nonpack")

    def test_output_size_mismatch_fails_before_uv_mapping(self):
        atlas, metadata = prepare_pbr_atlas(self.views(), mode="nonpack")
        with self.assertRaisesRegex(ValueError, "output size"):
            unpack_pbr_prediction(atlas.resize((512, 512)), metadata)


if __name__ == "__main__":
    unittest.main()
