"""Public ``litereality`` command line.

The command surface deliberately mirrors the core architecture: run a pipeline,
run one stage, inspect a scene package, or invoke one model directly.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import threading
from pathlib import Path

from litereality_agent import console
from litereality_agent.pipeline import PipelineRunner, RunContext
from litereality_agent.pipeline.stages import STAGES
from litereality_agent.settings import load_settings

SCAN_MARKERS = ("room.usdz", "roomplan/room.usdz")

# Stages that CONSUME the scene package rather than producing it. Starting work here means the
# package must already exist — and then which package is the entire question.
AUTHORING_STAGES = frozenset({"author", "publish"})


def looks_like_scan_dir(path: str | os.PathLike[str]) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (
        any((candidate / marker).is_file() for marker in SCAN_MARKERS)
        or bool(next(candidate.glob("frame_*.jpg"), None))
    )


def resolve_scan(value: str) -> str:
    """Resolve a scan folder for small standalone callers retained in the package."""
    raw = value.rstrip("/")
    candidate = Path(raw).expanduser()
    if os.sep not in raw and not raw.startswith("."):
        return raw
    if not candidate.is_dir():
        raise SystemExit(f"no such scan folder: {candidate}")
    if not looks_like_scan_dir(candidate):
        raise SystemExit(f"{candidate} does not look like a RoomPlan capture")
    candidate = candidate.resolve()
    os.environ["LR_SCANS_DIR"] = str(candidate.parent)
    return candidate.name


def _print_results(results) -> int:
    failed = False
    marks = {"completed": "✓", "reused": "↻", "skipped": "⊘", "failed": "✗"}
    for result in results:
        detail = result.error or "; ".join(result.warnings)
        suffix = f" — {detail}" if detail else ""
        print(
            f"{marks[result.status.value]} {result.stage:<12} "
            f"{result.status.value:<9} {result.duration_seconds:7.1f}s{suffix}"
        )
        failed |= result.status.value == "failed" and result.details.get("fatal", True)
    return int(failed)


def _context(args) -> RunContext:
    return RunContext.resolve(args.target, output_root=getattr(args, "output_root", None))


def _require_scene_package(target: str, context: RunContext, stage: str) -> None:
    """Insist that an authoring stage is handed the scene package, not the capture it came from.

    A capture folder resolves to a perfectly valid context — same scan name, same default output
    root — so `stage author scans/<name>` appears to work. It stops appearing to work the moment
    `--output-root` differs from the default, at which point the stage authors one tree while you
    sit watching another, and nothing says so. The package path cannot be wrong in that way.

    The two failures want different answers, so they get them: a package that does not exist yet
    means scene init has not run, and saying so here beats the stage failing several steps later
    on a missing seed room.
    """
    package = context.scene_dir / "scene.json"
    try:
        given_is_package = Path(target).expanduser().resolve() == context.scene_dir.resolve()
    except OSError:
        given_is_package = False
    if package.is_file() and given_is_package:
        return

    red, bold, dim, off = (console.colour(c) for c in ("red", "bold", "dim", "off"))
    scene, capture = console.short(context.scene_dir), console.short(context.capture_dir)
    lines = [
        f"\n{red}✗ {stage} runs on the scene package, not the capture{off}\n",
        f"   {dim}given   {off} {target}",
        f"   {dim}expected{off} {scene}{dim}  — the tree scene init produces{off}\n",
    ]
    if package.is_file():
        lines += ["Point it at the package instead:\n",
                  f"   {bold}uv run litereality stage {stage} {scene}{off}\n"]
    else:
        lines += [f"{dim}{scene}/scene.json does not exist — scene init has not produced it yet.{off}\n",
                  f"   {bold}uv run litereality run {capture} --through seed{off}",
                  f"   {bold}uv run litereality stage {stage} {scene}{off}\n"]
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit(2)


def _author_options(args) -> dict:
    polish = getattr(args, "polish", False)
    return {
        "refine_objects": polish or getattr(args, "refine_objects", False),
        "materials": polish or getattr(args, "materials", False),
        "quality_pass": polish or getattr(args, "quality_pass", False),
    }


def _add_author_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--polish",
        action="store_true",
        help="run object refinement, materials, and model-driven QC after authoring",
    )
    parser.add_argument("--refine-objects", action="store_true")
    parser.add_argument("--materials", action="store_true")
    parser.add_argument("--quality-pass", action="store_true")


def _add_publish_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compare-frames", type=int, default=None,
        help="real-vs-render comparison pairs to build at the end of publish; 0 to skip (default 6)",
    )


def _publish_options(args) -> dict:
    return {"compare_frames": getattr(args, "compare_frames", None)}


def _add_live_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--live",
        action="store_true",
        help="watch the room and the agent's trace in a browser while this runs; the viewer keeps "
             "serving after it finishes (ctrl-c to stop)",
    )
    parser.add_argument(
        "--live-port", type=int, default=8770,
        help="where to start looking for a free port; taken ones are skipped",
    )
    parser.add_argument(
        "--live-no-bake",
        action="store_true",
        help="live viewer rebuilds geometry only — keeps up with rapid saves, but procedural "
             "SHELL materials render flat",
    )


def _block_until_interrupt() -> None:
    """Park the main thread so the daemon HTTP server keeps serving. Named so tests can replace it —
    otherwise every `--live` test would hang here waiting for a ctrl-c that never comes."""
    threading.Event().wait()


@contextlib.contextmanager
def _live_viewer(args, context):
    """Serve the room beside the work when `--live` is passed, and keep serving after it ends.

    Yields a callback the caller uses to say how the run turned out. Sharing one `RunContext` with
    the stage is the whole point — a separately launched viewer can be pointed at a different
    `--output-root` than the run it is meant to be watching, and then silently shows nothing.

    Deliberately does NOT require the room to exist first. The stage this wraps is the thing that
    creates it — `author` copies the seed room in as its first act — so demanding it up front made
    `--live` work only on the SECOND authoring run of a scene, which is the run you least need to
    watch. Every layer below already copes with a room that is not there yet: the watcher ignores a
    missing `Room.py` and fires the moment one appears, the server answers `no build yet`, and the
    page says "waiting for a build". Standalone `litereality live` keeps the hard check, because
    nothing in that command is going to produce a room.
    """
    if not getattr(args, "live", False):
        yield lambda _note: None
        return

    from litereality_agent.pipeline.realism_authoring import live

    bake = not args.live_no_bake
    room, server, url = live.start(context, port=args.live_port, bake=bake)
    live.describe(context, url, bake)
    print()
    try:
        yield room.finish
    finally:
        bold, cyan, off = (console.colour(c) for c in ("bold", "cyan", "off"))
        # Same burial problem as the opening banner, at the other end: this lands under whatever
        # the stage printed last, and it is the line you need in order to go and look at the room.
        print(f"\n   {bold}{cyan}▶ viewer still serving at {url}{off}"
              f"   {console.colour('dim')}ctrl-c to stop{off}\n")
        try:
            _block_until_interrupt()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            room.stop()
            server.shutdown()
            server.server_close()


def _run(args) -> int:
    context = _context(args)
    # Only when the run STARTS at authoring. A full `run scans/<name>` legitimately begins with a
    # capture and produces the package on the way through.
    if args.from_stage in AUTHORING_STAGES:
        _require_scene_package(args.target, context, args.from_stage)
    with _live_viewer(args, context) as finished:
        results = PipelineRunner().run(
            context,
            start=args.from_stage,
            through=args.through,
            force=set(args.force or ()),
            strict=args.strict,
            options={"author": _author_options(args), "publish": _publish_options(args)},
        )
        rc = _print_results(results)
        finished("run finished" if not rc else "run failed")
    return rc


def _stage(args) -> int:
    context = _context(args)
    if args.stage in AUTHORING_STAGES:
        _require_scene_package(args.target, context, args.stage)
    with _live_viewer(args, context) as finished:
        result = PipelineRunner().run_stage(
            context,
            args.stage,
            force=args.force,
            strict=args.strict,
            options={
                "skip_image_generation": args.skip_image_generation,
                **_author_options(args),
                **_publish_options(args),
            },
        )
        rc = _print_results([result])
        finished(f"{args.stage} finished" if not rc else f"{args.stage} failed")
    return rc


def _live(args) -> int:
    from litereality_agent.pipeline.realism_authoring import live

    return live.serve(
        args.target, port=args.port, host=args.host, poll=args.poll,
        output_root=args.output_root, bake=args.bake, bake_resolution=args.bake_resolution,
    )


def _view(args) -> int:
    """Walk a published room. `publish` knows which file it wrote; `room_ops.walk` serves any glb —
    resolving one run's tree to the other's argument is this layer's whole job."""
    from litereality_agent.pipeline.room_qc import publish
    from litereality_agent.room_ops import walk

    context = _context(args)
    return walk.serve(
        publish.viewable_room(context), context.scan,
        port=args.port, host=args.host, open_browser=args.open_browser,
        subtitle="published reconstruction",
    )


def _inspect(args) -> int:
    from litereality_agent.room_ops import manifest

    package = manifest.require(args.target)
    print(package.summary())
    required, optional = package.check()
    for path in optional:
        print(f"  ~ missing optional: {path}")
    for path in required:
        print(f"  ! missing required: {path}")
    return int(bool(required))


def _setup(args) -> int:
    from litereality_agent import repo_root
    from litereality_agent.runtimes import setup

    settings = load_settings()
    return setup.run(
        repo_root(),
        environment=args.env or settings.modal_environment,
        profile=args.profile or settings.modal_profile,
        credentials=None if args.profile else settings.modal_credentials(),
        skip_deploy=args.skip_deploy,
    )


def _generate_3d(args) -> int:
    from litereality_agent.models.registry import gen3d_from_settings

    service = gen3d_from_settings(load_settings())
    output = args.out or str(Path(args.image).with_suffix(".glb"))
    result = service.reconstruct(
        [args.image], out_dir=str(Path(output).resolve().parent),
        asset_id=Path(output).stem, seed=args.seed, simplify=args.simplify,
    )
    print(result)
    return int(not result or not Path(result).is_file())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="litereality", description="LiteReality-Agent")
    commands = parser.add_subparsers(dest="command", required=True)
    stages = [stage.name for stage in STAGES]

    run = commands.add_parser("run", help="run the reconstruction pipeline")
    run.add_argument("target", metavar="SCAN_OR_SCENE")
    run.add_argument("--from", dest="from_stage", choices=stages)
    run.add_argument("--through", choices=stages)
    run.add_argument("--force", action="append", choices=stages)
    run.add_argument("--strict", action="store_true")
    run.add_argument("--output-root")
    _add_author_options(run)
    _add_live_options(run)
    _add_publish_options(run)
    run.set_defaults(handler=_run)

    stage = commands.add_parser("stage", help="run exactly one pipeline stage")
    stage.add_argument("stage", choices=stages)
    stage.add_argument("target", metavar="SCAN_OR_SCENE")
    stage.add_argument("--force", action="store_true")
    stage.add_argument("--strict", action="store_true")
    stage.add_argument("--skip-image-generation", action="store_true")
    stage.add_argument("--output-root")
    _add_author_options(stage)
    _add_live_options(stage)
    _add_publish_options(stage)
    stage.set_defaults(handler=_stage)

    view = commands.add_parser("view", help="walk a published room in the browser")
    view.add_argument("target", metavar="SCAN_OR_SCENE")
    view.add_argument(
        "--port", type=int, default=8780,
        help="where to start looking for a free port; taken ones are skipped",
    )
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--output-root")
    view.add_argument("--no-open", dest="open_browser", action="store_false")
    view.set_defaults(handler=_view)

    live = commands.add_parser("live", help="watch a room being authored, with the agent's trace")
    live.add_argument("target", metavar="SCAN_OR_SCENE")
    live.add_argument(
        "--port", type=int, default=8770,
        help="where to start looking for a free port; taken ones are skipped",
    )
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--poll", type=float, default=1.0, help="seconds between Room.py checks")
    live.add_argument("--output-root")
    live.add_argument(
        "--no-bake", dest="bake", action="store_false",
        help="geometry only — faster, but procedural SHELL materials render flat",
    )
    live.add_argument("--bake-resolution", type=int, default=1024)
    live.set_defaults(handler=_live)

    setup = commands.add_parser("setup", help="authenticate Modal and deploy the model apps")
    setup.add_argument("--profile", help="Modal profile to use instead of the active one")
    setup.add_argument("--env", help="Modal environment (default: MODAL_ENVIRONMENT)")
    setup.add_argument(
        "--skip-deploy", action="store_true", help="authenticate and record the profile only"
    )
    setup.set_defaults(handler=_setup)

    scene = commands.add_parser("scene", help="inspect a generated scene package")
    scene_subcommands = scene.add_subparsers(dest="scene_command", required=True)
    inspect = scene_subcommands.add_parser("inspect")
    inspect.add_argument("target", nargs="?", default=None)
    inspect.set_defaults(handler=_inspect)

    model = commands.add_parser("model", help="invoke one configured model")
    model_subcommands = model.add_subparsers(dest="model_command", required=True)
    generate = model_subcommands.add_parser("generate-3d")
    generate.add_argument("image")
    generate.add_argument("--out")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--simplify", type=float, default=0.95)
    generate.set_defaults(handler=_generate_3d)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_settings().apply_environment()
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
