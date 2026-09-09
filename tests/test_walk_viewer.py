"""The walkable viewer and the local server that feeds it.

No browser and no Blender here: what is checked is the page's contract with the vendored `app.js`
(which is what silently breaks when the bundle is re-vendored) and the routes the server answers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from litereality_agent.pipeline.context import RunContext
from litereality_agent.pipeline.room_qc import publish
from litereality_agent.pipeline.room_qc.publish import WEB_GLB_NAME
from litereality_agent.room_ops import walk


@pytest.fixture
def published(tmp_path: Path) -> Path:
    scene = tmp_path / "run" / "Scan-1"
    preview = scene / "realism_authoring" / "room_preview"
    preview.mkdir(parents=True)
    (preview / "Room.glb").write_bytes(b"glTF-published")
    return scene


def _context(scene: Path) -> RunContext:
    return RunContext(
        scan="Scan-1", capture_dir=scene, scene_dir=scene, output_root=scene.parent,
    )


# --------------------------------------------------------------------------- #
# the page's contract with the vendored bundle
# --------------------------------------------------------------------------- #
def test_page_provides_every_element_the_bundle_reaches_for() -> None:
    """`app.js` is vendored from the site generator, and the skeleton here is hand-trimmed — so a
    re-vendor that starts using a new element would fail at runtime, in a browser, silently. This
    reads the ids straight out of the bundle instead of trusting a list written down once."""
    page = walk.render("Scan-1")
    source = (walk.ASSETS / "app.js").read_text()

    # Only what is looked up on the DOCUMENT. `el.querySelector('.op-eye')` and friends run against
    # rows app.js built itself moments earlier, so they are not the page's job to provide.
    ids = set(re.findall(r"document\.getElementById\('([^']+)'\)", source))
    ids |= set(re.findall(r"(?<![\w.])getElementById\('([^']+)'\)", source))
    classes = set(re.findall(r"document\.querySelector\('\.?([a-zA-Z.-]+)'\)", source))
    classes = {c.split(".")[-1] for c in classes}

    missing_ids = sorted(i for i in ids if f'id="{i}"' not in page)
    missing_classes = sorted(c for c in classes if f'class="{c}' not in page and f' {c}"' not in page)
    assert not missing_ids, f"app.js reaches for ids the page never renders: {missing_ids}"
    assert not missing_classes, f"app.js reaches for classes the page never renders: {missing_classes}"


def test_no_scan_cloud_means_no_compare_button() -> None:
    """The Compare button is built only when `cloud` is set, so a run with no point cloud must
    leave it out entirely rather than ship a control that fails when pressed."""
    scene = json.loads(re.search(r"window\.SCENE = (\{.*?\});", walk.render("Scan-1")).group(1))
    assert "cloud" not in scene
    assert scene["cameras"] == []      # and no camera strip without pose data

    with_cloud = json.loads(
        re.search(r"window\.SCENE = (\{.*?\});", walk.render("Scan-1", cloud_url="p.ply")).group(1))
    assert with_cloud["cloud"] == "p.ply"


def test_the_bundle_never_reaches_the_r2_worker() -> None:
    """Vendoring copies a page built to stream from the site's bucket. Anything still pointing there
    is an asset this viewer cannot serve, and it fails only once someone is offline from that host."""
    source = (walk.ASSETS / "app.js").read_text()
    assert "workers.dev" not in source
    assert "cdn.jsdelivr.net" in source     # the decoder it does need, from a public CDN


def test_assets_are_served_by_name_only() -> None:
    assert walk.asset("app.js") is not None
    assert walk.asset("app.css") is not None
    for attack in ("../__init__.py", "/etc/passwd", "walk/app.js", "", "__init__.py"):
        assert walk.asset(attack) is None, attack


# --------------------------------------------------------------------------- #
# the server
# --------------------------------------------------------------------------- #
def test_serves_the_page_the_bundle_and_the_room(published: Path) -> None:
    context = _context(published)
    server, url = walk.start(publish.require_published(context), context.scan, port=0)
    try:
        page = urlopen(url).read().decode()
        assert "window.SCENE" in page and "Scan-1" in page
        assert urlopen(url + "app.js").read().startswith(b"/* Vendored")
        assert urlopen(url + "room.glb").read() == b"glTF-published"
        with pytest.raises(HTTPError):
            urlopen(url + "nope.js")
    finally:
        server.shutdown()
        server.server_close()


def test_prefers_the_compressed_body_when_it_is_current(published: Path) -> None:
    """A Draco copy sits beside the build once someone has viewed it. It is the same room at
    roughly a quarter the size, so it is what the browser should be handed — but only while it
    matches the build."""
    context = _context(published)
    preview = context.preview_dir
    (preview / WEB_GLB_NAME).write_bytes(b"draco-small")

    import os

    os.utime(preview / WEB_GLB_NAME, (2_000_000, 2_000_000))
    os.utime(preview / "Room.glb", (1_000_000, 1_000_000))
    assert publish.published_room(context) == preview / WEB_GLB_NAME

    # a rebuilt room outdates the cache, and stale geometry must never win
    os.utime(preview / "Room.glb", (3_000_000, 3_000_000))
    assert publish.published_room(context) == preview / "Room.glb"


def test_an_unpublished_run_says_how_to_publish_it(tmp_path: Path) -> None:
    scene = tmp_path / "run" / "Scan-1"
    scene.mkdir(parents=True)
    with pytest.raises(SystemExit, match="stage publish"):
        publish.require_published(_context(scene))


# --------------------------------------------------------------------------- #
# the web copy, which the viewer makes rather than the publish stage
# --------------------------------------------------------------------------- #
@pytest.fixture
def compressor(monkeypatch):
    """Stands in for Blender: records what it was asked to compress and writes the cache."""
    calls: list[Path] = []

    def fake_compressed(glb: Path, cache: Path) -> Path:
        calls.append(Path(glb))
        cache.write_bytes(b"draco:" + Path(glb).read_bytes())
        return cache

    monkeypatch.setattr("litereality_agent.room_ops.compress.compressed", fake_compressed)
    return calls


def test_the_first_view_of_a_build_compresses_it(published: Path, compressor) -> None:
    """The whole point of moving this off the publish stage: a run that is never viewed never pays
    for the web copy, and viewing one that has no web copy makes it rather than refusing."""
    context = _context(published)

    room = publish.viewable_room(context)

    assert room == context.preview_dir / WEB_GLB_NAME
    assert room.read_bytes() == b"draco:glTF-published"
    assert compressor == [context.preview_dir / "Room.glb"], "compresses the current build"


def test_a_later_view_reuses_the_web_copy(published: Path, compressor) -> None:
    """`compressed` caches, so the Blender launch is once per build and not once per view."""
    context = _context(published)
    publish.viewable_room(context)
    compressor.clear()

    assert publish.viewable_room(context) == context.preview_dir / WEB_GLB_NAME
    assert compressor == [], "a current web copy is served without touching Blender"


def test_a_rebuilt_room_is_recompressed_before_it_is_served(published: Path, compressor) -> None:
    """Serving the stale cache would show last build's geometry — the failure `published_room`'s
    mtime test exists to prevent, and it has to survive the lazy path too."""
    import os

    context = _context(published)
    publish.viewable_room(context)
    compressor.clear()

    (context.preview_dir / "Room.glb").write_bytes(b"glTF-rebuilt")
    os.utime(context.preview_dir / WEB_GLB_NAME, (1_000_000, 1_000_000))
    os.utime(context.preview_dir / "Room.glb", (3_000_000, 3_000_000))

    room = publish.viewable_room(context)
    assert compressor == [context.preview_dir / "Room.glb"]
    assert room.read_bytes() == b"draco:glTF-rebuilt"


def test_no_blender_serves_the_full_size_room(published: Path, monkeypatch) -> None:
    """`compress` hands back the ORIGINAL when Blender is missing. A slower page is not a reason to
    refuse to show the room, so the viewer takes whatever comes back."""
    context = _context(published)
    monkeypatch.setattr("litereality_agent.room_ops.compress.compressed", lambda glb, cache: glb)

    assert publish.viewable_room(context) == context.preview_dir / "Room.glb"


def test_viewing_an_unpublished_run_still_says_how_to_publish_it(tmp_path: Path) -> None:
    """Compressing nothing is not a fallback — with no compiled room there is nothing to view, and
    the message has to stay the one that says how to get one."""
    scene = tmp_path / "run" / "Scan-1"
    scene.mkdir(parents=True)
    with pytest.raises(SystemExit, match="stage publish"):
        publish.viewable_room(_context(scene))
