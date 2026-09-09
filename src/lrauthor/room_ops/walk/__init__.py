"""The walkable viewer: orbit, point-and-go, and walk, as three modes over one room.

This viewer is a reusable room operation. It reads a glb and serves it. It never touches `Room.py`
or pipeline state, so it stays under `room_ops`.

Served rather than written to a file: a room runs to tens of megabytes, and a self-contained page
would have to carry it base64'd a third larger again and parse the whole thing before drawing
anything. Over localhost the browser just streams it, behind the progress bar the viewer draws.

Takes a glb PATH, not a run — finding a run's published room is `publish`'s business, and knowing
nothing about run layout is what makes this reusable on any room folder.

    python -m lrauthor.room_ops.walk <Room.glb> [--port 8780]

`app.js` / `app.css` are vendored from the published site (see the header in `app.js`). The page
below is the same skeleton those pages use, minus the site's navigation and the banana easter egg
that its `app.js` does not implement anyway.
"""

from __future__ import annotations

import html
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from lrauthor.room_ops.serve import bind

# The bundle is written against 0.168; the live viewer's page pins 0.160 for its own. Kept
# separate on purpose — matching the version each bundle was tested against matters more than
# having one number, and the two pages never share a document.
THREE = "0.168.0"

ASSETS = Path(__file__).parent
MIME = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


def asset(name: str) -> tuple[bytes, str] | None:
    """One of the vendored assets by bare filename, or None. Never takes a path: the name is
    whatever a browser asked for, so anything with a separator in it is rejected outright."""
    if name not in {"app.js", "app.css"}:
        return None
    path = ASSETS / name
    if not path.is_file():
        return None
    return path.read_bytes(), MIME[path.suffix]


_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="theme-color" content="#bcd2d3">
<title>__LABEL__ — LiteReality</title>
<link rel="stylesheet" href="app.css">
</head>
<body>

<div class="loading-screen">
  <div class="loading-inner">
    <div class="loading-text">Loading Scene…</div>
    <div class="loading-sub">__LABEL__</div>
    <div class="progress-container">
      <div class="progress-bar"></div>
      <div class="progress-text">0%</div>
    </div>
  </div>
</div>

<canvas class="webgl"></canvas>

<div class="control-panel">
  <div class="panel-title">__LABEL__<span>__SUB__</span></div>
  <div class="mode-buttons"></div>
</div>

<div class="fps-info"></div>
<div class="crosshair"></div>

<div class="cam-strip" id="camStrip" hidden>
  <span class="cam-strip-label">Camera views</span>
  <div class="cam-strip-rail" id="camRail"></div>
</div>

<div class="cmp-card" id="cmpCard" hidden>
  <img class="cmp-img" id="cmpImg" alt="render vs real comparison" title="Click to enlarge">
  <div class="cmp-bar">
    <button id="cmpPrev" title="Previous view">◀</button>
    <span class="cmp-cap" id="cmpCap"></span>
    <button id="cmpNext" title="Next view">▶</button>
    <button id="cmpClose" title="Back to orbit">✕</button>
  </div>
</div>

<div class="cmp-lightbox" id="cmpLightbox" hidden><img id="cmpLightImg" alt="render vs real comparison"></div>

<div class="objects-panel" id="objPanel" hidden>
  <div class="op-head" id="opHead"><b>Objects</b><span class="op-count" id="opCount"></span><span class="op-toggle">▾</span></div>
  <div class="op-actions">
    <button id="opReset" title="Back to the dollhouse view">Reset</button>
    <button id="opWire" title="Show the reconstructed mesh as wireframe">Wireframe</button>
  </div>
  <div class="op-body" id="opBody"></div>
</div>

<script>window.SCENE = __SCENE__;</script>
<script type="importmap">{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@__THREE__/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@__THREE__/examples/jsm/"
}}</script>
<script type="module" src="app.js"></script>
</body></html>
"""


def render(label: str, glb_url: str = "room.glb", *, subtitle: str = "",
           cloud_url: str | None = None, cameras: list | None = None) -> str:
    """The page for one room.

    `cloud_url` is left unset by a plain run: the viewer's Compare button is created only when a
    scan cloud is actually there to compare against, so omitting it hides the feature rather than
    shipping a button that fails. Same for `cameras` and the camera strip.
    """
    scene = {"url": glb_url, "label": label, "cameras": cameras or []}
    if cloud_url:
        scene["cloud"] = cloud_url
    return (
        _PAGE.replace("__SCENE__", json.dumps(scene))
        .replace("__THREE__", THREE)
        .replace("__LABEL__", html.escape(label))
        .replace("__SUB__", html.escape(subtitle))
    )


def handler(glb: Path, label: str, *, subtitle: str = ""):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:  # noqa: D102 — one line per asset is pure noise
            pass

        def _send(self, body: bytes, ctype: str, cache: str = "no-store") -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/":
                return self._send(render(label, "room.glb", subtitle=subtitle).encode(),
                                  "text/html; charset=utf-8")
            if path == "/room.glb":
                # Tens of megabytes that never change while the server is up, so let the browser
                # keep it — otherwise every reload re-streams the whole room.
                return self._send(glb.read_bytes(), "model/gltf-binary", "public, max-age=3600")
            found = asset(path.lstrip("/"))
            if found:
                body, ctype = found
                return self._send(body, ctype)
            self.send_error(404)

    return Handler


def start(glb: Path, label: str, *, port: int = 8780, host: str = "127.0.0.1",
          subtitle: str = ""):
    """Serve `glb` on a background thread and return `(server, url)`. Caller owns shutdown."""
    server = bind(handler(Path(glb), label, subtitle=subtitle), host, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://{host}:{server.server_port}/"


def serve(glb: Path, label: str, *, port: int = 8780, host: str = "127.0.0.1",
          subtitle: str = "", open_browser: bool = True) -> int:
    """Serve one room until interrupted."""
    glb = Path(glb)
    server, url = start(glb, label, port=port, host=host, subtitle=subtitle)
    print(f"walkable viewer  {url}")
    print(f"  room   {glb}  ({glb.stat().st_size / 1e6:.0f} MB)")
    print("  orbit · point-and-go · walk (WASD) · ctrl-c to stop")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — headless or no handler; the url is printed either way
            pass
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.shutdown()
        server.server_close()
    return 0
