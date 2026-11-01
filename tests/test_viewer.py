"""Check the viewer's file boundary, export flags and headless launch behavior."""

import contextlib
import http.client
import io
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

import infer_miku
from tools import preview_glb


class ViewerServerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        (self.dist / "index.html").write_text("<title>MIKU viewer</title>")
        (self.dist / "app.js").write_text("// viewer app")
        self.model = self.root / "private models" / "custom axe.glb"
        self.model.parent.mkdir()
        self.model.write_bytes(b"glTF-test")
        (self.model.parent / "private.txt").write_text("must not be served")
        (self.dist / "outside.txt").symlink_to(self.model.parent / "private.txt")
        self.server, self.url = preview_glb.create_server(self.model, port=0, root=self.dist)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, method="GET", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        try:
            connection.request(method, path, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_serves_only_static_assets_and_selected_model(self):
        status, _headers, data = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"MIKU viewer", data)
        status, headers, data = self.request("/__model__.glb")
        self.assertEqual((status, data), (200, b"glTF-test"))
        self.assertEqual(headers["Content-Type"], "model/gltf-binary")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        self.assertIn("name=custom+axe.glb", self.url)
        self.assertNotIn(str(self.model.parent), self.url)

    def test_blocks_escape_directory_listing_and_arbitrary_host(self):
        for path in ("/../private%20models/private.txt", "/%2e%2e/private%20models/private.txt",
                     "/outside.txt", "/private.txt", "/.git/config"):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 404)
        (self.dist / "subdir").mkdir()
        self.assertEqual(self.request("/subdir/")[0], 404)
        self.assertEqual(self.request("/", headers={"Host": "untrusted.example"})[0], 403)

    def test_head_matches_model_headers_without_body(self):
        status, headers, data = self.request("/__model__.glb", method="HEAD")
        self.assertEqual((status, data), (200, b""))
        self.assertEqual(int(headers["Content-Length"]), len(b"glTF-test"))

    def test_detached_launch_serves_selected_model_after_parent_returns(self):
        helper = self.root / "tools" / "preview_glb.py"
        helper.parent.mkdir()
        shutil.copyfile(preview_glb.__file__, helper)
        built = self.root / "viewer" / "dist"
        built.parent.mkdir()
        shutil.copytree(self.dist, built)
        children = []
        popen = subprocess.Popen

        def remember_child(*args, **kwargs):
            child = popen(*args, **kwargs)
            children.append(child)
            return child

        try:
            with patch.object(preview_glb, "__file__", str(helper)), \
                    patch.object(preview_glb, "VIEWER_ROOT", built), \
                    patch.object(preview_glb.subprocess, "Popen", side_effect=remember_child), \
                    contextlib.redirect_stdout(io.StringIO()):
                url = preview_glb.launch_viewer(self.model, port=0, open_browser=False)
            from urllib.request import urlopen
            from urllib.parse import urlsplit
            parsed = urlsplit(url)
            with urlopen(f"{parsed.scheme}://{parsed.netloc}/__model__.glb", timeout=2) as response:
                self.assertEqual(response.read(), b"glTF-test")
            self.assertIsNone(children[0].poll())
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=5)


class PreviewCliTests(unittest.TestCase):
    def test_export_only_preview_and_port_validation(self):
        for argv in (["--preview", "--no-glb"], ["--preview", "--conditions-only"],
                     ["--preview", "--mv-only"], ["--preview-port", "65536"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                infer_miku.parse_args(argv)
            self.assertEqual(raised.exception.code, 2)
        args = infer_miku.parse_args(["--preview", "--preview-port", "0"])
        self.assertTrue(args.preview)
        self.assertEqual(args.preview_port, 0)

    def test_missing_viewer_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(FileNotFoundError, "viewer/dist/index.html"):
                preview_glb.validate_inputs(root=path)

    def test_remote_session_does_not_open_server_browser(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"SSH_CONNECTION": "present"}), \
                patch.object(preview_glb.webbrowser, "open") as browser, \
                contextlib.redirect_stdout(output):
            preview_glb.announce("http://127.0.0.1:8765/", 8765)
        browser.assert_not_called()
        self.assertIn("ssh -L 8765:127.0.0.1:8765", output.getvalue())

    def test_explicit_no_open_never_opens_browser(self):
        with patch.object(preview_glb, "is_headless", return_value=False), \
                patch.object(preview_glb.webbrowser, "open") as browser, \
                contextlib.redirect_stdout(io.StringIO()):
            preview_glb.announce("http://127.0.0.1:8765/", 8765, open_browser=False)
        browser.assert_not_called()


if __name__ == "__main__":
    unittest.main()
