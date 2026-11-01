"""Open an exported GLB in the local MIKU material viewer (standard library only)."""

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
from urllib.parse import unquote, urlencode, urlsplit
import webbrowser


REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWER_ROOT = REPO_ROOT / "viewer" / "dist"


class ViewerHandler(BaseHTTPRequestHandler):
    """Serve built viewer files and one explicitly selected model, never its folder."""

    def __init__(self, *args, root, model, **kwargs):
        self.root = Path(root).resolve()
        self.model = Path(model).resolve() if model else None
        super().__init__(*args, **kwargs)

    def log_message(self, *_args):
        pass

    def do_HEAD(self):
        self._serve(body=False)

    def do_GET(self):
        self._serve(body=True)

    def _serve(self, body):
        # A browser may reach localhost via a hostile DNS name. Keep the server
        # usable only at loopback hostnames, without enabling CORS or uploads.
        host = self.headers.get("Host", "").split(":", 1)[0].lower()
        if host not in ("127.0.0.1", "localhost"):
            self.send_error(403, "Use the localhost viewer URL")
            return
        path = unquote(urlsplit(self.path).path)
        if path == "/__model__.glb":
            target = self.model
        else:
            relative = Path(path.lstrip("/") or "index.html")
            if any(part.startswith(".") for part in relative.parts):
                self.send_error(404)
                return
            try:
                target = (self.root / relative).resolve()
                target.relative_to(self.root)
            except (ValueError, OSError):
                self.send_error(404)
                return
        if target is None or not target.is_file():
            self.send_error(404)
            return
        try:
            stream = target.open("rb")
        except OSError:
            self.send_error(404)
            return
        with stream:
            self.send_response(200)
            content_type = "model/gltf-binary" if target.suffix.lower() == ".glb" else (
                mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            )
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(os.fstat(stream.fileno()).st_size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if body:
                try:
                    shutil.copyfileobj(stream, self.wfile)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # Closing a tab during a model load is harmless.


def validate_inputs(model=None, root=None):
    root = Path(root or VIEWER_ROOT).resolve()
    if not (root / "index.html").is_file():
        raise FileNotFoundError("Viewer is missing: viewer/dist/index.html. Include the built viewer directory.")
    if model is not None:
        model = Path(model).expanduser().resolve()
        if model.suffix.lower() != ".glb" or not model.is_file():
            raise FileNotFoundError(f"Select an existing .glb file: {model}")
    return model, root


def create_server(model=None, port=8765, root=None):
    model, root = validate_inputs(model, root)
    handler = partial(ViewerHandler, root=root, model=model)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    url = f"http://127.0.0.1:{server.server_port}/"
    if model is not None:
        url += "?" + urlencode({"model": "/__model__.glb", "name": model.name})
    return server, url


def is_headless():
    return bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY")) or (
        sys.platform.startswith("linux")
        and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    )


def announce(url, port, open_browser=True, pid=None):
    print(f"MIKU viewer: {url}", flush=True)
    if pid is not None:
        print(f"Viewer server PID: {pid}; stop it with: kill {pid}", flush=True)
    if is_headless():
        print(f"From your desktop, forward this port: ssh -L {port}:127.0.0.1:{port} <user>@<server>", flush=True)
        print("Then open the viewer URL in your desktop browser, or copy the GLB locally and drag it into the viewer.", flush=True)
    elif open_browser:
        if not webbrowser.open(url):
            print("Open the viewer URL in a browser with WebGL 2 support.", flush=True)


def launch_viewer(model, port=8765, open_browser=True):
    """Launch a persistent local server after inference and return its URL."""
    model, _root = validate_inputs(model)
    command = [sys.executable, str(Path(__file__).resolve()), "--model", str(model),
               "--port", str(port), "--serve-child"]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=True, text=True)
    try:
        readable, _, _ = select.select([process.stdout], [], [], 10)
        if not readable:
            raise RuntimeError("Viewer server did not start within 10 seconds")
        payload = json.loads(process.stdout.readline())
        if "error" in payload:
            raise RuntimeError(payload["error"])
    except (OSError, ValueError, RuntimeError) as exc:
        process.terminate()
        process.wait(timeout=5)
        raise RuntimeError(f"Could not start the viewer: {exc}") from exc
    finally:
        process.stdout.close()
    announce(payload["url"], payload["port"], open_browser=open_browser, pid=process.pid)
    return payload["url"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="GLB to display; otherwise show the bundled axe result")
    parser.add_argument("--port", type=int, default=8765, help="Local port (default: 8765; 0 selects a free port)")
    parser.add_argument("--no-open", action="store_true", help="Print the URL without opening a browser")
    parser.add_argument("--serve-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        server, url = create_server(args.model, args.port)
    except OSError as exc:
        message = str(exc)
        if isinstance(exc, OSError) and exc.errno in (48, 98):
            message = f"Port {args.port} is in use. Choose another --port, or use --port 0."
        if args.serve_child:
            print(json.dumps({"error": message}), flush=True)
        else:
            print(f"MIKU viewer: {message}", file=sys.stderr)
        return 1
    if args.serve_child:
        print(json.dumps({"url": url, "port": server.server_port}), flush=True)
    else:
        announce(url, server.server_port, open_browser=not args.no_open)
        print("Press Ctrl+C to stop the viewer server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
