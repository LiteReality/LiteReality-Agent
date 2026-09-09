"""The scene PACKAGE — a scan's init output as a SELF-DESCRIBING folder.

`scene_init` used to leave its results as a pile of directories whose meaning lived in `the CLI`:
the caller had to know that the seed room is `run/<scan>/scene_init/scene_stage/room_init/room`, that the
object references are `run/<scan>/scene_init/obj_stage/object_init`, that the capture is
`$LR_SCANS_DIR/<scan>`, and that four separate `$LITEREALITY_*` env vars have to agree for any of
it to resolve. Stage 2 therefore could not be launched without re-supplying all of that.

A scene package fixes that by writing **one manifest at the root of what init produced**::

    run/<scan>/
        scene.json            <- THE MANIFEST: every path + root + provenance stage 2 needs
        obj_stage/            <- object_init work tree, references, reconstructed GLBs
        scene_stage/          <- room_init (seed Room.py + Room.glb), _harness, _scene_assets
        capture/<scan>/       <- the RoomPlan capture (symlinked by default, copied on request)
        realism_authoring/    <- STAGE 2's root: room/ (the Room.py it edits), surface_ref/,
                                 logs/, and room_preview/ (the build, the web copy, trace.html)

Given that folder — and nothing else — stage 2 can rebuild the whole environment::

    pkg = manifest.require("run/<scan>")  # or None: discovered from $LR_SCENE / cwd
    manifest.apply_env(pkg)                         # exports the $LITEREALITY_* contract
    ...                                             # lrauthor.agent.tools.shared.config now resolves

From a shell, `python -m lrauthor.room_ops.manifest env <dir>` prints those exports for
`eval` — that is how `the CLI --scene <dir>` rebuilds its environment.

Deliberately **stdlib only**, importing nothing else in this package: it is the seam between the
deterministic and agentic halves, so it must stay cheap enough for either side to depend on —
including a future split into two uv environments, where it would be the shared piece.

Paths inside the package are stored RELATIVE to it, so the folder can be moved or copied to
another machine and still resolve; anything outside is stored absolute.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "SCHEMA",
    "MANIFEST_NAME",
    "ScenePackage",
    "apply_env",
    "bootstrap",
    "discover",
    "embed_capture",
    "read",
    "require",
    "write",
]

SCHEMA = "litereality.scene/1"
MANIFEST_NAME = "scene.json"

# Paths the manifest carries. `required` ones are what stage 2 cannot start without — `check()`
# reports them separately from the merely-nice-to-have. Keys are stable API: the CLI and the
# authoring CLIs address them by name.
PATH_KEYS = (
    "obj_stage",
    "refroot",
    "reconstructed_objs",
    "traces",
    "scene_stage",
    "room_init",
    "room_init_preview",
    "room_py",
    "room_glb",
    "assets",
    "harness",
    "surface_ref",
    "authoring",       # stage 2's own root: the room it edits, its renders, its deliverables
    "authoring_room",
    "authoring_preview",
    "authoring_scratch",  # working images the model makes to LOOK at — kept, see scratch.py
    "capture",
)
REQUIRED_PATHS = ("room_init", "room_py", "capture", "refroot")
ROOT_KEYS = ("output", "final", "scans")


# ── the package ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ScenePackage:
    """A parsed `scene.json` bound to the directory it was actually found in.

    `root` is the real location on disk, not whatever was recorded at write time — a package
    that gets moved still resolves, because every in-package path is stored relative.
    """

    root: Path
    data: dict[str, Any]

    # --- identity ---
    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def scan(self) -> str:
        return str(self.data.get("scan") or "")

    @property
    def schema(self) -> str:
        return str(self.data.get("schema") or "")

    # --- paths ---
    def path(self, key: str) -> Path | None:
        """Resolve one recorded path. Relative entries hang off the package root."""
        raw = (self.data.get("paths") or {}).get(key)
        if not raw:
            return None
        p = Path(raw)
        return p if p.is_absolute() else (self.root / p)

    def root_path(self, key: str) -> Path | None:
        raw = (self.data.get("roots") or {}).get(key)
        return Path(raw) if raw else None

    def __getattr__(self, name: str) -> Any:  # pkg.room_py, pkg.surface_ref, …
        if name in PATH_KEYS:
            return self.path(name)
        raise AttributeError(name)

    @property
    def scans_root(self) -> Path | None:
        """The directory that must be `$LR_SCANS_DIR` for `<root>/<scan>` to be this capture."""
        recorded = self.root_path("scans")
        capture = self.path("capture")
        # The capture's PARENT is authoritative: every consumer computes SCAN_DIR as
        # $LR_SCANS_DIR/<scan>, so an embedded capture (capture/<scan>/) only resolves if the
        # scans root points at the embedding dir rather than at wherever the scan came from.
        if capture is not None and capture.name == self.scan:
            return capture.parent
        return recorded

    # --- the environment contract ---
    def env(self) -> dict[str, str]:
        """The `$LITEREALITY_*` variables every downstream config module reads.

        Deliberately excludes `$LR_REPO_ROOT`: that locates the CODE, not the data, and baking a
        writer's checkout path into a portable package is how a moved package silently reads a
        stale repo.
        """
        env: dict[str, str] = {"LR_SCENE": str(self.root), "LITEREALITY_SCAN": self.scan}
        for key, var in (("output", "LITEREALITY_OUTPUT"), ("final", "LITEREALITY_FINAL")):
            value = self.root_path(key)
            if value is not None:
                env[var] = str(value)
        scans = self.scans_root
        if scans is not None:
            env["LR_SCANS_DIR"] = str(scans)
        return env

    def stage_env(self) -> dict[str, str]:
        """`LR_*` conveniences for shell callers (the CLI) — one per stage-2 argument."""
        out: dict[str, str] = {}
        for key, var in (
            ("room_init", "LR_ROOM_INIT"),
            ("room_init_preview", "LR_ROOM_PREVIEW"),
            ("surface_ref", "LR_SURFACE_REF"),
            ("authoring", "LR_AUTHORING"),
            ("authoring_room", "LR_ROOM"),
            ("authoring_preview", "LR_ROOM_PREVIEW"),
            ("authoring_scratch", "LR_SCRATCH"),
            ("refroot", "LR_REFROOT"),
            ("capture", "LR_SCAN_DIR"),
            ("obj_stage", "LR_OBJ_STAGE"),
            ("scene_stage", "LR_SCENE_STAGE"),
        ):
            value = self.path(key)
            if value is not None:
                out[var] = str(value)
        return out

    # --- health ---
    def check(self) -> tuple[list[str], list[str]]:
        """`(missing_required, missing_optional)` — recorded paths that are not on disk."""
        required, optional = [], []
        for key in PATH_KEYS:
            p = self.path(key)
            if p is None or p.exists():
                continue
            (required if key in REQUIRED_PATHS else optional).append(f"{key}={p}")
        for key in REQUIRED_PATHS:
            if self.path(key) is None:
                required.append(f"{key}=(not recorded)")
        return required, optional

    def summary(self) -> str:
        lines = [f"scene package  {self.root}", f"  scan     {self.scan}   schema {self.schema}"]
        created = self.data.get("created_utc")
        if created:
            lines.append(f"  created  {created}")
        for key in PATH_KEYS:
            p = self.path(key)
            if p is None:
                continue
            mark = "✓" if p.exists() else "✗"
            lines.append(f"  {mark} {key:<19} {p}")
        cap = self.data.get("capture") or {}
        if cap:
            lines.append(
                f"  capture: mode={cap.get('mode', '?')} frames={cap.get('frames', '?')} "
                f"usdz={'yes' if cap.get('has_usdz') else 'no'}"
            )
        init = self.data.get("init") or {}
        if init:
            lines.append(
                f"  init: rc={init.get('object_init_rc', '?')} glb={init.get('n_glb', '?')} "
                f"objects={len(init.get('objects') or [])}"
            )
        return "\n".join(lines)


# ── writing ──────────────────────────────────────────────────────────────────
def _rel(p: Path | str | None, root: Path) -> str | None:
    """Store `p` relative to `root` when it lives inside it, else absolute.

    Tries the LEXICAL spelling first and only then the resolved one. Both matter here: the
    lexical pass keeps an in-package symlink (`capture/<scan>` -> the scans dir) recorded as the
    in-package path instead of chasing it back out, while the resolved pass catches the reverse
    case — a caller handing us `run/<scan>/scene_init/obj_stage`, which is itself a symlink INTO the
    package.
    """
    if p is None:
        return None
    p = Path(p)
    abs_p = Path(os.path.abspath(p))
    abs_root = Path(os.path.abspath(root))
    for cand, base in ((abs_p, abs_root), (_resolve(p), _resolve(root))):
        try:
            return cand.relative_to(base).as_posix() or "."
        except ValueError:
            continue
    return str(abs_p)


def _resolve(p: Path) -> Path:
    try:
        return p.resolve()
    except OSError:
        return Path(os.path.abspath(p))


def write(
    root: Path | str,
    *,
    scan: str,
    paths: Mapping[str, Path | str | None],
    roots: Mapping[str, Path | str | None] | None = None,
    **sections: Any,
) -> Path:
    """Write `<root>/scene.json`. Returns the manifest path.

    `paths` keys should come from `PATH_KEYS`; unknown keys are kept (forward compatibility) but
    nothing downstream will look for them. Extra `sections` (capture=…, init=…, providers=…) are
    stored verbatim alongside.
    """
    from datetime import datetime, timezone

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "scan": scan,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "paths": {k: v for k, v in ((k, _rel(v, root)) for k, v in paths.items()) if v},
        "roots": {
            k: str(Path(os.path.abspath(v)))
            for k, v in (roots or {}).items()
            if v is not None
        },
    }
    for key, value in sections.items():
        if value is not None:
            payload[key] = value
    manifest = root / MANIFEST_NAME
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return manifest


# ── reading + discovery ──────────────────────────────────────────────────────
def read(path: Path | str) -> ScenePackage:
    """Load a package from its root, its `scene.json`, or any directory inside it."""
    found = _manifest_at(Path(path))
    if found is None:
        raise FileNotFoundError(f"no {MANIFEST_NAME} at or above {path}")
    data = json.loads(found.read_text(encoding="utf-8"))
    return ScenePackage(root=found.parent, data=data)


def _manifest_at(path: Path) -> Path | None:
    """Find the manifest for `path`: itself, inside it, or in a parent."""
    if path.is_file() and path.name == MANIFEST_NAME:
        return path
    for start in _spellings(path):
        for candidate in (start, *start.parents):
            manifest = candidate / MANIFEST_NAME
            if manifest.is_file():
                return manifest
    return None


def _spellings(path: Path) -> list[Path]:
    """A path and its realpath — `run/<scan>/…` and `run/<scan>/…` are the same tree.

    the CLI symlinks `run/<scan>/{obj_stage,scene_stage}` at `run/<scan>/…`, so a cwd
    under the `output/` spelling walks up to `run/<scan>/`, which holds no manifest. Searching
    the resolved spelling as well is what makes discovery work from either side of that link.
    """
    lexical = Path(os.path.abspath(path))
    real = _resolve(path)
    return [lexical] if lexical == real else [lexical, real]


def _repo_root() -> Path:
    from lrauthor import repo_root

    return repo_root()


def _default_roots() -> list[Path]:
    """Where packages live when nothing points at one — mirrors object_init.config."""
    repo = _repo_root()
    final = os.environ.get("LITEREALITY_FINAL")
    output = os.environ.get("LITEREALITY_OUTPUT") or os.environ.get("OBJECT_INIT_OUTPUT")
    return [
        Path(final).expanduser() if final else (repo / "run"),
        Path(output).expanduser() if output else (repo / "run"),
    ]


def discover(hint: Path | str | None = None, *, deep: bool = True) -> ScenePackage | None:
    """Locate a scene package. Returns None rather than raising — see `require`.

    Order, most explicit first:
      1. `hint` (a package root, a `scene.json`, or anything inside a package)
      2. `$LR_SCENE`
      3. the current directory, walking up
      4. `$LITEREALITY_SCAN` under the default roots (`run/<scan>`, `run/<scan>`)
      5. `deep` only — the only package under the default roots

    Step 5 is convenient at a prompt ("just run the thing I built") and wrong inside a library:
    it is the one rule that can bind to a package nobody named. `deep=False` stops before it, and
    is what the passive `bootstrap()` uses.
    """
    for candidate in (hint, os.environ.get("LR_SCENE") or None, Path.cwd()):
        if candidate is None or not str(candidate).strip():
            continue  # an empty hint means "no hint" — shell callers pass "" for that
        manifest = _manifest_at(Path(candidate))
        if manifest is not None:
            return read(manifest)

    scan = (os.environ.get("LITEREALITY_SCAN") or "").strip()
    roots = _default_roots()
    if scan:
        for base in roots:
            manifest = base / scan / MANIFEST_NAME
            if manifest.is_file():
                return read(manifest)
    if not deep:
        return None

    found = list(_packages_under(roots))
    return read(found[0]) if len(found) == 1 else None


def list_packages(roots: Iterable[Path] | None = None) -> list[Path]:
    """Every package root under the default (or given) roots, newest manifest first."""
    found = list(_packages_under(list(roots) if roots is not None else _default_roots()))
    found.sort(key=lambda m: m.stat().st_mtime, reverse=True)
    return [m.parent for m in found]


def _packages_under(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for base in roots:
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            manifest = child / MANIFEST_NAME
            if not manifest.is_file():
                continue
            key = _resolve(manifest)
            if key in seen:
                continue
            seen.add(key)
            yield manifest


def require(hint: Path | str | None = None) -> ScenePackage:
    """`discover`, but fail loudly and usefully — the stage-2 entry points call this."""
    pkg = discover(hint)
    if pkg is not None:
        return pkg
    known = list_packages()
    detail = (
        "\n  candidates: " + ", ".join(str(p) for p in known[:8])
        if known
        else "\n  (no scene.json found under " + ", ".join(str(r) for r in _default_roots()) + ")"
    )
    raise SystemExit(
        f"no scene package found — pass --scene <dir>, set $LR_SCENE, or run from inside one.{detail}\n"
        "  A package is written by `uv run litereality run <scan> --through seed`; an output tree from before packages existed "
        "can be upgraded in place with `uv run litereality scene --adopt <scan>` (nothing is rebuilt)."
    )


# ── applying it ──────────────────────────────────────────────────────────────
def apply_env(pkg: ScenePackage, *, override: bool = True) -> dict[str, str]:
    """Export the package's environment contract. Returns what was set.

    `override=False` lets an explicit shell environment win, which is the right default when the
    package was only *discovered* — a user who exported `$LITEREALITY_OUTPUT` by hand meant it.
    """
    env = pkg.env()
    for key, value in env.items():
        if override or not os.environ.get(key):
            os.environ[key] = value
    return env


def bootstrap(argv: list[str] | None = None, *, hint: Path | str | None = None) -> ScenePackage | None:
    """Resolve a package from `--scene`/`$LR_SCENE`/cwd and apply its env. Never raises.

    Call this BEFORE importing anything that reads `$LITEREALITY_*` at import time (chiefly
    `lrauthor.agent.tools.shared.config`, which resolves every harness path the moment it is imported).

    `--scene` is PEEKED at, not consumed: the entry point's own argparse still declares it, so
    `--help` stays honest and a typo is still an argparse error rather than a silent miss.

    An explicitly named package overrides the environment; a merely *discovered* one only fills
    the gaps, so a legacy invocation that exported `$LITEREALITY_*` by hand is never retargeted
    underneath the caller.
    """
    explicit = hint if hint is not None else _peek_scene(argv)
    pkg = discover(explicit, deep=explicit is not None)
    if pkg is not None:
        apply_env(pkg, override=explicit is not None)
    return pkg


def _peek_scene(argv: list[str] | None = None) -> str | None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    for i, arg in enumerate(args):
        if arg == "--scene" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--scene="):
            return arg.split("=", 1)[1]
    return None


# ── capture embedding ────────────────────────────────────────────────────────
def embed_capture(root: Path | str, source: Path | str, scan: str, mode: str = "link") -> Path:
    """Put the capture inside the package as `capture/<scan>/` and return that path.

    The `<scan>` level is not decoration: every consumer computes the capture as
    `$LR_SCANS_DIR/<scan>`, so embedding it one level down is what lets `$LR_SCANS_DIR` simply
    point at `<package>/capture` with no code change anywhere.

    modes: `link` (symlink — cheap, stays in step with the source), `copy` (real bytes — the
    package becomes portable), `reference` (nothing embedded; the manifest records the source).
    """
    root, source = Path(root), Path(source)
    if mode == "reference":
        return source
    dest = root / "capture" / scan
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        else:
            shutil.rmtree(dest)
    if mode == "link":
        try:
            dest.symlink_to(Path(os.path.abspath(source)), target_is_directory=True)
            return dest
        except OSError:
            mode = "copy"  # filesystem without symlinks — fall through rather than fail init
    if mode == "copy":
        shutil.copytree(source, dest, symlinks=True)
        return dest
    raise ValueError(f"unknown capture mode: {mode!r} (link|copy|reference)")


def capture_stats(capture: Path | str) -> dict[str, Any]:
    """What the manifest records about the capture, so a consumer can sanity-check it cheaply."""
    capture = Path(capture)
    if not capture.is_dir():
        return {"exists": False}
    frames = sorted(capture.glob("frame_*.jpg"))
    poses = sorted(capture.glob("frame_*.json"))
    usdz = (capture / "room.usdz").is_file() or (capture / "roomplan" / "room.usdz").is_file()
    return {
        "exists": True,
        "frames": len(frames),
        "poses": len(poses),
        "has_usdz": usdz,
        "has_depth": bool(next(capture.glob("depth_*.png"), None)),
    }


# ── the shell interface ──────────────────────────────────────────────────────
# `python -m lrauthor.room_ops.manifest <cmd> [<dir>]`
#
#     env    eval-able exports that rebuild the stage-2 environment — the CLI sources this
#     show   human summary + which recorded paths are actually on disk
#     check  exit 1 if a required path is missing
#     list   every package under run/
def _cli(argv: list[str] | None = None) -> int:
    import shlex
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args and not args[0].startswith("-") else "show"
    hint = next((a for a in args[1:] if a.strip() and not a.startswith("-")), None)

    if cmd == "list":
        found = list_packages()
        for root in found:
            print(root)
        return 0 if found else 1

    if cmd == "env":
        # Quiet on failure: the CLI guards this and falls back to the legacy <scan>-derived paths,
        # so a noisy stderr here would be alarming rather than useful.
        pkg = discover(hint)
        if pkg is None:
            return 1
        for key, value in {**pkg.env(), **pkg.stage_env()}.items():
            print(f"export {key}={shlex.quote(value)}")
        return 0

    pkg = require(hint)
    if cmd == "show":
        print(pkg.summary())
        missing_required, missing_optional = pkg.check()
        if missing_optional:
            print("  ~ missing (optional): " + ", ".join(missing_optional))
        if missing_required:
            print("  ! MISSING (required): " + ", ".join(missing_required))
        return 0

    if cmd == "check":
        missing_required, _ = pkg.check()
        for item in missing_required:
            print(f"missing: {item}", file=sys.stderr)
        return 1 if missing_required else 0

    print(f"unknown command: {cmd} (env|show|check|list)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
