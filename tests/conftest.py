"""Shared fixtures — chiefly a REAL stage tree, because the layout is where tools break.

The bug these tests exist to prevent (`render` dead for a whole authoring run with "could not
infer scan") was not a logic error inside a tool. It was the *shape of the paths* the tools are
handed: on a tagged run `the CLI` writes stage data into `run/<scan><RUN_TAG>/` and symlinks
`run/<scan>/scene_init/{obj_stage,scene_stage}` at it, so every room path has two legitimate spellings and
`Path.resolve()` silently moves you from one to the other. A fixture that just makes a directory
called `room` cannot catch that. `stage_tree` reproduces the symlink.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

SCAN = "test-scan-Room"

# A Room.py stripped to what the surface/scan helpers actually parse: the SHELL dict. Wall2 is a
# deliberate sub-0.35m RoomPlan corner artifact; Wall10 is there so numeric (not lexicographic)
# ordering is exercised; the opening key proves `Wall1_Door_0` does not mint a phantom wall.
ROOM_PY = '''\
"""Test room — minimal but structurally honest."""

SHELL = {
    "walls": {
        "Wall0":  {"start": [0.00, 0.00], "end": [4.00, 0.00], "height": 2.60},
        "Wall1":  {"start": [4.00, 0.00], "end": [4.00, 3.00], "height": 2.60},
        "Wall2":  {"start": [4.00, 3.00], "end": [4.20, 3.05], "height": 2.60},
        "Wall3":  {"start": [4.20, 3.05], "end": [0.00, 3.05], "height": 2.60},
        "Wall10": {"start": [0.00, 3.05], "end": [0.00, 0.00], "height": 2.60},
    },
    "openings": {
        "Wall1_Door_0": {"offset": 1.20, "width": 0.90, "height": 2.05, "sill": 0.00},
    },
    "floor_height": 0.0,
    "ceiling_height": 2.60,
}


def build():
    return SHELL
'''


@dataclass(frozen=True)
class StageTree:
    """The two spellings of one room, plus the pieces tools resolve around them."""

    root: Path
    scan: str
    physical_room: Path  # <deliverables>/run/<scan>/scene_init/scene_stage/_oneshot/room
    symlinked_room: Path  # <staging>/run/<scan>/scene_init/scene_stage/_oneshot/room
    surface_ref: Path


@pytest.fixture
def stage_tree(tmp_path: Path) -> StageTree:
    """Build two `run/<scan>/...` trees — one physical, one symlinked — as the CLI can produce.

    `$LITEREALITY_OUTPUT` and `$LITEREALITY_FINAL` are separate roots that merely DEFAULT to the
    same `run/`. When they differ, the CLI links the stage dirs across, and one room again has two
    legitimate spellings whose `.resolve()` differ. Both roots are named `run` — that matters:
    the scan name is inferred from the directory under the stage root, so a fixture that renamed
    the scan dir (a RUN_TAG suffix, say) would assert the tag as the scan.

    Mirrors the CLI: `ln -sfn "$FINAL_OUT/<stage>" "$OUT/<stage>"`, per stage directory (not one
    link at the scan level) — the asymmetry is what makes the resolved path differ from the
    caller's while both name the same tree.
    """
    final = tmp_path / "deliverables" / "run" / SCAN
    out = tmp_path / "staging" / "run" / SCAN
    room = final / "scene_init" / "scene_stage" / "_oneshot" / "room"
    room.mkdir(parents=True)
    (room / "Room.py").write_text(ROOM_PY, encoding="utf-8")

    surface_ref = final / "scene_init" / "scene_stage" / "_harness" / "surface_ref"
    surface_ref.mkdir(parents=True)
    for name in ("Wall0", "Wall1", "Wall3", "Wall10", "Floor0", "Ceiling0"):
        (surface_ref / f"{name}_stitched.jpg").write_bytes(b"")

    (final / "scene_init" / "obj_stage" / "traces").mkdir(parents=True)
    out.mkdir(parents=True)
    # Both spellings occur in the wild: the CLI writes ABSOLUTE targets, while trees made by hand
    # use RELATIVE ones. Cover each, and route the room under test through the relative link,
    # since that is what a live run actually has.
    (out / "scene_init").mkdir(parents=True)
    (out / "scene_init" / "obj_stage").symlink_to(final / "scene_init" / "obj_stage")
    (out / "scene_init" / "scene_stage").symlink_to(
        Path("..") / ".." / ".." / ".." / "deliverables" / "run" / SCAN / "scene_init" / "scene_stage"
    )

    return StageTree(
        root=tmp_path,
        scan=SCAN,
        physical_room=room,
        symlinked_room=out / "scene_init" / "scene_stage" / "_oneshot" / "room",
        surface_ref=surface_ref,
    )


@pytest.fixture(autouse=True)
def _isolate_data_roots(tmp_path_factory, monkeypatch: pytest.MonkeyPatch):
    """Point every data root at a temp dir, for every test.

    The suite's fixture scan is `test-scan-Room`, and it turned up as a real directory in the
    user's `run/` tree: a config module was mkdir-ing under $LITEREALITY_OUTPUT at import time.
    That import-time side effect is fixed, but the roots are pinned here too — a test has no
    business writing next to someone's scans, and this is the cheap way to guarantee it.
    """
    root = tmp_path_factory.mktemp("data-roots")
    # WRITE roots only. $LR_SCANS_DIR points at read-only captures, and several tests reason
    # about whether it is set at all — pinning it here would make them assert the fixture.
    for var in ("LITEREALITY_OUTPUT", "LITEREALITY_FINAL", "OBJECT_INIT_OUTPUT"):
        monkeypatch.setenv(var, str(root / var.lower()))
    monkeypatch.delenv("LR_SCENE", raising=False)


@dataclass(frozen=True)
class ExampleScan:
    """A real capture plus the seed room built from it — what a `-m scan` tool test binds to."""

    name: str
    capture: Path  # <scans>/<name>/  — frame_*.jpg, frame_*.json, roomplan/room.usdz
    room: Path  # run/<name>/scene_init/scene_stage/room_init/room  — the seed Room.py

    @property
    def frames(self) -> list[Path]:
        return sorted(self.capture.glob("frame_*.jpg"))


def _scan_roots() -> list[Path]:
    """Where a capture may live, in the order we prefer.

    `$LR_SCANS_DIR` is resolved through settings rather than `os.environ` on purpose: pytest does
    not load `.env`, so a raw env read finds nothing and every scan test skips while looking like
    it ran. That is exactly how the previous scan test died unnoticed.
    """
    from lrauthor.settings import load_settings

    repo = Path(__file__).resolve().parents[1]
    roots = [load_settings().resolved_scans_dir(), repo / "example-scans", repo / "scans_uploaded"]
    seen, out = set(), []
    for root in roots:
        if root not in seen:
            seen.add(root)
            out.append(root)
    return out


def _looks_like_capture(path: Path) -> bool:
    return (path / "roomplan" / "room.usdz").is_file() and bool(next(path.glob("frame_*.jpg"), None))


@pytest.fixture(scope="session")
def example_scan() -> ExampleScan:
    """The first real capture that also has a seed room built, else skip with how to get one.

    Opt in with `-m scan`. Point `$LR_SCANS_DIR` at your captures, or clone the published
    examples:  git clone https://github.com/LiteReality/example-scans
    """
    repo = Path(__file__).resolve().parents[1]
    output = Path(os.environ.get("LITEREALITY_SCAN_OUTPUT") or repo / "run")
    tried = []
    for root in _scan_roots():
        if not root.is_dir():
            continue
        for capture in sorted(p for p in root.iterdir() if p.is_dir()):
            if not _looks_like_capture(capture):
                continue
            room = output / capture.name / "scene_init" / "scene_stage" / "room_init" / "room"
            if (room / "Room.py").is_file():
                return ExampleScan(name=capture.name, capture=capture, room=room)
            tried.append(f"{capture.name} (capture ok, no seed room at {room})")
    pytest.skip(
        "no example scan with a seed room. Clone "
        "https://github.com/LiteReality/example-scans and run `litereality run <scan> "
        "--through seed`, or point $LR_SCANS_DIR at your captures."
        + (f" Tried: {'; '.join(tried[:3])}" if tried else "")
    )


@pytest.fixture(autouse=True)
def _isolate_scan_env(monkeypatch: pytest.MonkeyPatch):
    """`$LITEREALITY_SCAN` is ambient input AND output here — `compose._config_for` writes it as a
    side effect of importing the harness config. Left alone, one test's write becomes the next
    test's fallback and the suite passes for the wrong reason.
    """
    monkeypatch.delenv("LITEREALITY_SCAN", raising=False)
    before = dict(os.environ)
    yield
    for key in set(os.environ) - set(before):
        os.environ.pop(key, None)
