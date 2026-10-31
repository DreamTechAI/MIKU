"""CPU geometry checks; run with the Miku runtime (torch/numpy/trimesh)."""
import unittest
import numpy as np
import torch
import trimesh

from DifferentiableRenderer.camera_utils import get_miku_mv_matrix, get_perspective_projection_matrix
from DifferentiableRenderer.MeshRender import MeshRender


class MikuOrientationTests(unittest.TestCase):
    def test_legacy_camera_axes_including_pole_roll(self):
        # Physical right/up vectors from the old float32 camera generator.
        cases = [
            (0, 270, [1, 0, 0], [0, 0, 1]),
            (0, 90, [-1, 0, 0], [0, 0, 1]),
            (0, 0, [0, 1, 0], [0, 0, 1]),
            (0, 180, [0, -1, 0], [0, 0, 1]),
            (90, 0, [0, -1, 0], [1, 0, 0]),
            (-90, 0, [0, -1, 0], [-1, 0, 0]),
        ]
        for elev, azim, right, up in cases:
            with self.subTest(elev=elev, azim=azim):
                view = get_miku_mv_matrix(elev, azim, 4.5)
                np.testing.assert_allclose(view[0, :3], right, atol=2e-6)
                np.testing.assert_allclose(view[1, :3], up, atol=2e-6)
                np.testing.assert_allclose(view[:3, :3] @ view[:3, :3].T, np.eye(3), atol=2e-6)
                self.assertAlmostEqual(float(np.linalg.det(view[:3, :3])), 1, places=6)

    def test_original_y_up_projects_to_smaller_image_row(self):
        view = get_miku_mv_matrix(0, 270, 4.5)
        proj = get_perspective_projection_matrix(30, 1, .01, 100)
        proj[1, 1] *= -1
        # Original glTF +Y becomes Miku world +Z, and must appear higher.
        points = np.array([[0, 0, 0, 1], [0, 0, .4, 1]], dtype=np.float32)
        clip = points @ view.T @ proj.T
        rows = (clip[:, 1] / clip[:, 3] * .5 + .5) * 1023
        self.assertLess(rows[1], rows[0])

    def test_mesh_normalization_and_export_round_trip(self):
        mesh = trimesh.creation.box(extents=[2, 4, 6])
        original = mesh.vertices.copy() + [3, -4, 5]
        render = MeshRender.__new__(MeshRender)
        render.device = 'cpu'
        render.coordinate_system = 'miku'
        render.set_mesh(original, mesh.faces)
        self.assertAlmostEqual(float((render.vtx_pos.max(0).values - render.vtx_pos.min(0).values).max()), 2)
        np.testing.assert_array_equal(original, mesh.vertices + [3, -4, 5])
        # get_mesh expects UVs; this test needs no rasterizer or UV baking.
        render.vtx_uv = torch.zeros((len(original), 2))
        render.uv_idx = torch.tensor(mesh.faces)
        saved_vertices = render.vtx_pos.clone()
        exported, *_ = render.get_mesh(normalize=False)
        np.testing.assert_allclose(exported, original, atol=1e-6)
        torch.testing.assert_close(render.vtx_pos, saved_vertices)

    def test_condition_channels_and_white_background(self):
        render = MeshRender.__new__(MeshRender)
        render.device = 'cpu'
        render.coordinate_system = 'miku'
        render.pos_idx = torch.tensor([[0, 1, 2]])
        # Original normal +Y and normalized original point (.2, .4, .6).
        render.miku_source_normals = torch.tensor([[0., 1., 0.]]).repeat(3, 1)
        render.vtx_pos = torch.tensor([[.2, -.6, .4]]).repeat(3, 1)
        render._create_view_state = lambda config: type('State', (), {
            'pos_clip': None, 'resolution': (1, 2),
        })()
        raster = torch.tensor([[[[1., 0., 0., 1.], [0., 0., 0., 0.]]]])
        render.raster_rasterize = lambda *args: (raster, None)
        render.raster_interpolate = lambda attrs, *args: (attrs[:, :1].reshape(1, 1, 1, 3).repeat(1, 1, 2, 1), None)
        normal, position, mask = render.render_miku_conditions(0, 270)
        torch.testing.assert_close(normal[0, 0], torch.tensor([.5, .5, 0.]))
        torch.testing.assert_close(position[0, 0], torch.tensor([.6, .8, .3]))
        torch.testing.assert_close(normal[0, 1], torch.ones(3))
        torch.testing.assert_close(position[0, 1], torch.ones(3))
        self.assertEqual(mask.flatten().tolist(), [True, False])


if __name__ == '__main__':
    unittest.main()
