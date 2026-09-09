"""Every module named as a STRING must resolve.

A cross-package call that goes through a subprocess names its target in a string — `-m
lrauthor.models.object_generation.generate` — and a string is invisible to every import
check, every linter and every rename. This has already failed twice in exactly the same way: a
package moved, the imports were rewritten, and the `-m` strings quietly kept pointing at the old
name. The symptom is never an ImportError in the parent; it is a stage that "completes" with a
subprocess exit code nobody reads, and a missing GLB fifteen minutes later.

So: find every module path spelled as text in package and script sources and check it exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "src" / "lrauthor"

# The top-level names that mean "a module in this repo". Anything else in a `-m` is a stdlib or
# third-party module (`json.tool`, `pytest`) and not ours to check.
OURS = ("lrauthor",)
# Spellings that USED to be importable and must never come back — the exact regression above.
RETIRED = ("authoring", "scene_builder", "scene_package", "init", "object_init",
           "backends", "integration", "scene_init", "realism_authoring", "services", "adapters")

# `-m foo.bar` on a command line, and `NAME_MODULE = "foo.bar"` constants.
DASH_M = re.compile(r"""-m["'\s,\]]+\s*["']?([A-Za-z_][\w.]*\.[\w.]+)""")
MODULE_CONST = re.compile(r"""^\s*\w*MODULE\w*\s*=\s*["']([A-Za-z_][\w.]*\.[\w.]+)["']""", re.M)

SOURCES = sorted(
    [p for p in PKG.rglob("*.py")]
    + [p for p in PKG.rglob("*.sh")]
    + [p for p in (REPO / "scripts").rglob("*.py")]
    + [p for p in (REPO / "scripts").rglob("*.sh")]
    + [REPO / "sanity.py"]
)


def module_file(dotted: str) -> Path | None:
    """`lrauthor.a.b` -> the .py (or package dir) it names, or None."""
    parts = dotted.split(".")
    if parts[0] != "lrauthor":
        return None
    base = PKG.joinpath(*parts[1:])
    for candidate in (base.with_suffix(".py"), base / "__init__.py", base):
        if candidate.exists():
            return candidate
    return None


def collect() -> list[tuple[Path, int, str]]:
    found = []
    for src in SOURCES:
        if not src.is_file():
            continue
        for i, line in enumerate(src.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for m in DASH_M.finditer(line):
                found.append((src, i, m.group(1)))
        for m in MODULE_CONST.finditer(src.read_text(encoding="utf-8", errors="replace")):
            found.append((src, 0, m.group(1)))
    return found


def test_some_module_strings_exist():
    """Guard the guard: if the patterns stop matching, everything below passes vacuously."""
    ours = [d for _, _, d in collect() if d.startswith(OURS)]
    assert len(ours) >= 8, f"only found {len(ours)} module strings — the scan is not working"


def test_every_module_string_resolves():
    broken = [
        f"{src.relative_to(REPO)}:{line or '?'}: -m {dotted}"
        for src, line, dotted in collect()
        if dotted.startswith(OURS) and module_file(dotted) is None
    ]
    assert not broken, "module path(s) named in a string do not exist:\n  " + "\n  ".join(broken)


def test_no_retired_top_level_spelling():
    """`-m backends.procedural...` resolved before the src layout and silently stopped after.
    Nothing in the package is importable by a bare top-level name any more."""
    stale = [
        f"{src.relative_to(REPO)}:{line or '?'}: -m {dotted}"
        for src, line, dotted in collect()
        if dotted.split(".")[0] in RETIRED
    ]
    assert not stale, (
        "module path(s) missing the `lrauthor.` prefix — importable before the src layout, "
        "silently dead after:\n  " + "\n  ".join(stale)
    )


@pytest.mark.parametrize("dotted", [
    "lrauthor.models.object_generation.generate",  # subprocess target used by the pipeline
    "lrauthor.room_ops.compile.build_from_room",
    "lrauthor.room_ops.manifest",
    "lrauthor.pipeline.measure.flow",
    "lrauthor.models.grounding_dino.worker",
    "lrauthor.pipeline.author.realism.evidence",
    "lrauthor.pipeline.author.realism.entrypoint",
    "lrauthor.pipeline.author.realism.refine_objects",
    "lrauthor.pipeline.author.realism.materials",
    "lrauthor.pipeline.author.realism.quality",
])
def test_known_subprocess_targets(dotted: str):
    """The specific modules some other process launches by name, pinned individually so a rename
    that misses one fails here rather than mid-run."""
    assert module_file(dotted) is not None, f"{dotted} is launched by name but does not exist"


def test_executable_helpers_survived_the_layout_move():
    from lrauthor.agent.tools.shared import config
    from lrauthor.pipeline.author.reconstruct import flow

    paths = (
        flow.LAUNCHER,
        config.RENDER_TOOL,
        config.SELECT_TOOL,
        config.STITCH_TOOL,
        PKG / "room_ops" / "rendering" / "object_turntable.py",
    )
    assert all(path.is_file() for path in paths), "missing helper(s): " + ", ".join(
        str(path) for path in paths if not path.is_file()
    )


def test_reconstruct_resolves_python_from_canonical_repo_root(monkeypatch):
    from lrauthor.pipeline.author.reconstruct import flow

    monkeypatch.delenv("LITEREALITY_TRELLIS_PYTHON", raising=False)
    assert Path(flow.resolve_python(None)) == REPO / ".venv" / "bin" / "python"


def test_agent_render_tool_uses_the_engine_under_its_own_source():
    """The engine is the render tool's own source now; nothing may point back at room_ops."""
    source = (PKG / "agent" / "tools" / "render" / "tool.py").read_text(encoding="utf-8")
    assert "lrauthor.agent.tools.render.source" in source
    assert "room_ops.rendering.engine" not in source
    assert not (PKG / "room_ops" / "rendering" / "engine").exists(), (
        "an engine reappeared under room_ops — the render tool owns exactly one"
    )


def test_blender_render_worker_finds_the_camera_renderer():
    """The worker runs inside Blender and reaches package code by path, so the path must resolve.

    It used to join `_HERE/../room_render`, which broke silently when the tools' source moved under
    `agent/`: Blender exits 0 on a raised script, so the failure surfaced far away as a missing
    render_manifest.json. It anchors on the package directory now.
    """
    worker = PKG / "agent" / "tools" / "render" / "source" / "_blender_frames.py"
    assert worker.is_file()
    assert (PKG / "room_ops" / "rendering" / "room_render" / "render_room_cameras.py").is_file()

    source = worker.read_text(encoding="utf-8")
    assert '"room_ops", "rendering", "room_render"' in source
    assert 'os.path.join(_HERE, "..", "room_render")' not in source, (
        "back to the sibling assumption that the move already broke once"
    )


def test_chair_repair_uses_hosted_trellis_when_configured(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from lrauthor.pipeline.author.reconstruct.mesh_quality_check import chair_repair

    ref = tmp_path / "chair.png"
    ref.write_bytes(b"image")
    out = tmp_path / "Chair0.glb"
    settings = SimpleNamespace(modal_trellis_app="litereality-trellis")
    monkeypatch.setattr("lrauthor.settings.load_settings", lambda: settings)

    class Hosted:
        def reconstruct(self, images, *, out_dir, asset_id, **options):
            assert images == [ref]
            assert asset_id == "Chair0"
            path = Path(out_dir) / f"{asset_id}.glb"
            path.write_bytes(b"glb")
            return str(path)

    monkeypatch.setattr(
        "lrauthor.models.registry.gen3d_from_settings", lambda configured: Hosted()
    )
    assert chair_repair._trellis_one("scan", ref, out, python=None, seed=42, decimation=50_000) == 0
