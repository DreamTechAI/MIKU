"""Exercise example defaults and custom-input validation without model imports."""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import infer_miku


class CliDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.example = self.root / "assets/axe"
        self.example.mkdir(parents=True)
        (self.example / "mesh.glb").touch()
        (self.example / "image.png").touch()
        self.root_patch = patch.object(infer_miku, "REPO_ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def assert_cli_error(self, argv, message):
        error = io.StringIO()
        with contextlib.redirect_stderr(error), self.assertRaises(SystemExit) as raised:
            infer_miku.parse_args(argv)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(message, error.getvalue())

    def test_default_paths_do_not_depend_on_working_directory(self):
        previous = Path.cwd()
        try:
            os.chdir(self.example)
            args = infer_miku.parse_args([])
        finally:
            os.chdir(previous)
        self.assertEqual(args.mesh, str(self.example / "mesh.glb"))
        self.assertEqual(args.image, str(self.example / "image.png"))
        self.assertEqual(args.output, str(self.root / "outputs/axe/textured_mesh.obj"))
        self.assertEqual(args.config, str(self.root / "cfgs/miku.yaml"))

    def test_custom_prediction_inputs_must_be_paired(self):
        self.assert_cli_error(["--mesh", str(self.example / "mesh.glb")],
                              "--image is required with a custom --mesh")
        self.assert_cli_error(["--image", str(self.example / "image.png")],
                              "--mesh is required when --image is supplied")

    def test_explicit_arguments_still_override_defaults(self):
        mesh = self.root / "custom.obj"
        image = self.root / "custom.png"
        mesh.touch()
        image.touch()
        args = infer_miku.parse_args([
            "--mesh", str(mesh), "--image", str(image),
            "--output", "custom/result.obj", "--config", "custom.yaml",
            "--steps", "5", "--mv-only", "--no-pbr-background-white",
        ])
        self.assertEqual(args.mesh, str(mesh))
        self.assertEqual(args.image, str(image))
        self.assertEqual(args.output, "custom/result.obj")
        self.assertEqual(args.config, "custom.yaml")
        self.assertEqual(args.steps, 5)
        self.assertTrue(args.mv_only)
        self.assertFalse(args.pbr_background_white)

    def test_conditions_only_does_not_require_reference_image(self):
        (self.example / "image.png").unlink()
        custom_mesh = self.root / "custom.obj"
        custom_mesh.touch()
        for extra in ([], ["--mesh", str(custom_mesh)]):
            with self.subTest(extra=extra):
                args = infer_miku.parse_args(["--conditions-only", *extra])
                self.assertTrue(Path(args.mesh).is_file())
                self.assertIsNone(args.image)

    def test_input_and_output_validation_is_preserved(self):
        self.assert_cli_error(["--output", "result.glb"], "--output must be an .obj path")
        (self.example / "image.png").unlink()
        self.assert_cli_error([], "Reference image does not exist:")
        (self.example / "mesh.glb").unlink()
        self.assert_cli_error([], "Mesh does not exist:")


if __name__ == "__main__":
    unittest.main()
