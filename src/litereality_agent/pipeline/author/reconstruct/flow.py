"""Reconstruction stage — object-init references to textured GLBs via TRELLIS.2.

This is the bridge from this package's clean object-only references to the local
canonical TRELLIS.2 inference module under ``models/trellis``. It gathers, for one scan:

    object_refs/<scan>/<Object>/reference_1024.png
    chair_clusters/<scan>/references/<ChairCluster>.png
    opening_refs/<scan>/<Wall_*_Door/Window>/reference_1024.png

stages them (one PNG per asset, named by asset id) under
``run/<scan>/reconstruct/_refs/``, and runs the launcher once in batch mode so
every GLB lands together in ``run/<scan>/reconstruct/<asset>.glb``.

The launcher needs the heavy TRELLIS.2 environment (torch + o_voxel + natten). The
interpreter is resolved as: ``--trellis-python`` arg -> ``$LITEREALITY_TRELLIS_PYTHON``
-> the repo ``.venv`` -> the ``trellis2`` conda env. object_init itself runs fine on
``.venv``; only this stage needs the GPU env.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from litereality_agent import PACKAGE_ROOT, REPO_ROOT, telemetry
from litereality_agent.pipeline.measure import paths as config

LAUNCHER = PACKAGE_ROOT / "models" / "trellis" / "inference.py"
PROC_MODULE = "litereality_agent.models.object_generation.generate"


def resolve_python(explicit: str | None) -> str:
    """Pick the interpreter that runs the TRELLIS launcher."""
    for cand in (explicit, os.environ.get("LITEREALITY_TRELLIS_PYTHON")):
        if cand and Path(cand).exists():
            return str(cand)
    venv = REPO_ROOT / ".venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def resolve_procedural_python(explicit: str | None) -> str:
    """Interpreter for the articulated agent (needs ``claude_agent_sdk``)."""
    for cand in (explicit, os.environ.get("LITEREALITY_PROCEDURAL_PYTHON")):
        if cand and Path(cand).exists():
            return str(cand)
    return sys.executable  # caller's env; must have claude_agent_sdk


def _blender_env(blender_path: str | None) -> dict:
    """Process env with Blender prepended to PATH (the agent shells out to blender)."""
    env = dict(os.environ)
    bdir = blender_path or os.environ.get("LITEREALITY_BLENDER")
    if bdir:
        env["PATH"] = f"{bdir}:{env.get('PATH', '')}"
    # so the subprocess resolves the same per-scan output tree
    env["LITEREALITY_OUTPUT"] = str(config.output_root())
    return env



# When several builders run at once their subprocess output interleaves into an unreadable
# braid, so the parallel phase points each branch at its own log and shows a progress bar
# instead. Unset (the serial path, and every existing caller) streams to the terminal as before.
_LOG: dict[str, "object"] = {}


def log_to(path):
    """Send this thread's builder subprocess output to `path`. None restores the terminal."""
    import threading

    key = str(threading.get_ident())
    if path is None:
        _LOG.pop(key, None)
    else:
        _LOG[key] = Path(path)
    return path


def _redirect():
    """`stdout=`/`stderr=` kwargs for subprocess.run, honouring this thread's log."""
    import threading

    path = _LOG.get(str(threading.get_ident()))
    if path is None:
        return {}
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115 — closed below
    return {"stdout": handle, "stderr": subprocess.STDOUT, "_handle": handle}


def _run(cmd, **kwargs):
    """subprocess.run with this thread's output redirection applied.

    Forces the child unbuffered when redirecting. Python block-buffers stdout (4-8 KB) the
    moment it is not a terminal, so a long-running worker's output sits in that buffer until it
    exits — the log reads as dead for ten minutes while the work is fine. On a tty it was
    line-buffered and this never showed up. Everything downstream (the live status line, the
    branch board) tails these logs, so an unflushed child is an invisible run.
    """
    redirect = _redirect()
    handle = redirect.pop("_handle", None)
    if handle is not None:
        env = dict(kwargs.pop("env", None) or os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        kwargs["env"] = env
    try:
        return subprocess.run(cmd, **kwargs, **redirect)
    finally:
        if handle is not None:
            handle.close()


def run_procedural_for_scan(
    scan: str,
    *,
    openings: bool = False,
    python: str | None = None,
    blender_path: str | None = None,
    concurrency: int = 2,
    retries: int = 1,
    budget: float = 5.0,
    model: str | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> dict:
    """Build the route==procedural objects (or the doors/windows) of ``scan`` into
    ``run/<scan>/reconstruct/`` via the image-to-articulated-glb agent.

    Runs in a subprocess so it uses its own env (``claude_agent_sdk`` + Blender on
    PATH) rather than the light object_init env.
    """
    config.set_scan(scan)
    interpreter = resolve_procedural_python(python)
    cmd = [
        interpreter,
        "-m",
        PROC_MODULE,
        "--work",
        str(config.work_root()),
        "--scan",
        scan,
        "--out",
        str(config.reconstruct_dir(scan)),
        "--concurrency",
        str(concurrency),
        "--retries",
        str(retries),
        "--max-budget-usd",
        str(budget),
    ]
    if openings:
        cmd.append("--openings")
    if skip_existing:
        cmd.append("--skip-existing")
    if model:
        cmd += ["--model", model]
    if dry_run:
        cmd.append("--dry-run")

    label = "openings" if openings else "procedural"
    plan = {"scan": scan, "stage": label, "python": interpreter, "cmd": " ".join(cmd)}
    telemetry.event(f"{label}_launch", scan=scan, python=interpreter, cmd=plan["cmd"])
    print(f"[{label}] {scan}\n  $ {plan['cmd']}", flush=True)
    proc = _run(cmd, cwd=str(REPO_ROOT), env=_blender_env(blender_path))
    plan["status"] = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
    glbs = sorted(p.name for p in config.reconstruct_dir(scan).glob("*/*.glb"))
    plan["generated"] = len(glbs)

    # Physics, while the recipe is still the thing to fix. An object that leaves this stage with
    # no mass, no inertia and no collider is one the room export has to invent all three for.
    if not dry_run and glbs:
        from litereality_agent.models.object_generation.sim.pipeline import compile_directory

        sim = compile_directory(config.reconstruct_dir(scan))
        plan["sim"] = {k: v for k, v in sim.items() if k != "rows"}
        print(f"[{label}] physics: {sim['ok']}/{sim['objects']} compiled, "
              f"{sim['passed']} passed the gate, {sim['failed']} failed, {sim['errors']} errored",
              flush=True)
        telemetry.event(f"{label}_physics", scan=scan, **plan["sim"])

    telemetry.event(f"{label}_done", scan=scan, status=plan["status"], generated=len(glbs))
    return plan


def _trellis_route(scan: str) -> set[str] | None:
    """Names the complexity router assigned to the **trellis** route, or None if no
    routing has been run yet (then reconstruct everything as a fallback)."""
    manifest = config.work_root() / "routing" / "routing_manifest.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text())
    return {r["name"] for r in data.get("scans", {}).get(scan, []) if r.get("route") == "trellis"}


def collect_references(scan: str) -> list[tuple[str, Path]]:
    """Return (asset_id, reference_png) for every TRELLIS-reconstructable asset.

    When routing exists, only route==trellis objects/chairs are returned and openings
    are left to the procedural openings stage. With no routing, every reference
    (objects + chairs + openings) is reconstructed as a fallback.
    """
    refs: list[tuple[str, Path]] = []
    want = _trellis_route(scan)  # None -> no routing, take everything

    obj_root = config.object_refs_root() / scan
    if obj_root.is_dir():
        for obj_dir in sorted(p for p in obj_root.iterdir() if p.is_dir()):
            ref = obj_dir / "reference_1024.png"
            if ref.exists() and (want is None or obj_dir.name in want):
                refs.append((obj_dir.name, ref))

    chair_root = config.chair_clusters_root() / scan
    chair_norm = chair_root / "references"
    if not chair_norm.is_dir():
        chair_norm = chair_root / "nano_banana_normalized_1024"
    if chair_norm.is_dir():
        for png in sorted(chair_norm.glob("*.png")):
            if want is None or png.stem in want:
                refs.append((png.stem, png))

    # Openings are reconstructed by the procedural openings stage; only fall back to
    # TRELLIS for them when there is no routing at all.
    if want is None:
        op_root = config.opening_refs_root() / scan
        if op_root.is_dir():
            for op_dir in sorted(p for p in op_root.iterdir() if p.is_dir()):
                ref = op_dir / "reference_1024.png"
                if ref.exists():
                    refs.append((op_dir.name, ref))

    return refs


def run_for_scan(
    scan: str,
    *,
    python: str | None = None,
    seed: int = 42,
    decimation: int = 50000,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> dict:
    """Reconstruct the TRELLIS-route references of ``scan`` into ``run/<scan>/reconstruct/``."""
    config.set_scan(scan)
    recon_dir = config.reconstruct_dir(scan)
    refs = collect_references(scan)
    interpreter = resolve_python(python)

    plan = {
        "scan": scan,
        "launcher": str(LAUNCHER),
        "python": interpreter,
        "assets": [aid for aid, _ in refs],
        "n_assets": len(refs),
        "reconstruct_dir": str(recon_dir),
        "generated": 0,
    }
    if not refs:
        plan["status"] = "no_references"
        print(f"[reconstruct] no references found for {scan}", flush=True)
        return plan

    # Route to a configured backend. Never infer that the pipeline Python should run TRELLIS:
    # local execution is allowed only when the caller supplied an explicit interpreter.
    from litereality_agent.pipeline.author.reconstruct import service as reconstructor

    if not reconstructor.using_service() and python is None:
        # Auto-install the configured backend regardless of entry point.
        try:
            from litereality_agent.models.registry import gen3d_from_settings

            svc = gen3d_from_settings()
            if svc is not None:
                reconstructor.set_service(svc)
                print(
                    f"[reconstruct] gen3d backend: {getattr(svc, 'name', svc.__class__.__name__)}",
                    flush=True,
                )
        except Exception as exc:
            plan["status"] = "backend_unconfigured"
            plan["error"] = str(exc)
            print(f"[reconstruct] {exc}", flush=True)
            return plan
    if reconstructor.using_service():
        service_name = getattr(
            reconstructor.service(), "name", reconstructor.service().__class__.__name__
        )
        refs_dir = recon_dir / "_refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[str, Path]] = []
        for asset_id, ref in refs:
            dst = refs_dir / f"{asset_id}.png"
            if not dst.exists() or dst.stat().st_mtime < ref.stat().st_mtime:
                shutil.copy(ref, dst)
            staged.append((asset_id, dst))
        plan["backend"] = service_name
        plan["python"] = None
        plan["launcher"] = None
        telemetry.event("reconstruct_launch", scan=scan, n_assets=len(refs), backend=service_name)
        print(
            f"[reconstruct] {len(refs)} asset(s) -> {recon_dir}  (gen3d tool: {service_name})",
            flush=True,
        )
        if dry_run:
            plan["status"] = "dry_run"
            return plan
        results = reconstructor.reconstruct_refs(
            staged, recon_dir, seed=seed
        )  # simplify ratio default
        results = results or {}
        failed = [
            asset_id
            for asset_id, _ in staged
            if not results.get(asset_id) or not Path(results[asset_id]).is_file()
        ]
        generated = sorted(
            Path(path).name for path in results.values() if path and Path(path).is_file()
        )
        plan["status"] = "ok" if not failed else "failed"
        if failed:
            plan["failed_assets"] = failed
        plan["generated"] = len(generated)
        plan["glb"] = generated
        rep = getattr(reconstructor.service(), "last_report", None)
        if rep is not None:
            _g = (
                (lambda k: rep.get(k))
                if isinstance(rep, dict)
                else (lambda k: getattr(rep, k, None))
            )
            if _g("wall_s") is not None:
                plan["wall_s"] = round(_g("wall_s"), 1)
            if _g("total_exec_s") is not None:
                plan["exec_s"] = round(_g("total_exec_s"), 1)
            plan["cost_usd"] = _g("total_cost_usd")
        telemetry.event(
            "reconstruct_done",
            scan=scan,
            status=plan["status"],
            generated=len(generated),
            backend=service_name,
            cost_usd=plan.get("cost_usd"),
            wall_s=plan.get("wall_s"),
        )
        return plan

    plan["backend"] = "local-launcher"
    if not LAUNCHER.exists():
        plan["status"] = "launcher_missing"
        print(f"[reconstruct] launcher not found at {LAUNCHER}", flush=True)
        return plan

    # Stage one PNG per asset under reconstruct/_refs/<asset>.png for batch mode.
    refs_dir = recon_dir / "_refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    for asset_id, ref in refs:
        dst = refs_dir / f"{asset_id}.png"
        if not dst.exists() or dst.stat().st_mtime < ref.stat().st_mtime:
            shutil.copy(ref, dst)

    cmd = [
        interpreter,
        str(LAUNCHER),
        "-i",
        str(refs_dir),
        "-o",
        str(recon_dir),
        "--seed",
        str(seed),
        "--decimation",
        str(decimation),
    ]
    if skip_existing:
        cmd.append("--skip-existing")

    plan["cmd"] = " ".join(cmd)
    telemetry.event(
        "reconstruct_launch", scan=scan, n_assets=len(refs), python=interpreter, cmd=plan["cmd"]
    )
    print(f"[reconstruct] {len(refs)} asset(s) -> {recon_dir}\n  $ {plan['cmd']}", flush=True)

    if dry_run:
        plan["status"] = "dry_run"
        return plan

    proc = _run(cmd, cwd=str(REPO_ROOT))
    generated = sorted(p.name for p in recon_dir.glob("*.glb"))
    failed = sorted(aid for aid, _ in refs if not (recon_dir / f"{aid}.glb").is_file())
    if proc.returncode:
        plan["status"] = f"launcher_exit_{proc.returncode}"
    else:
        plan["status"] = "failed" if failed else "ok"
    plan["generated"] = len(generated)
    plan["glb"] = generated
    if failed:
        plan["failed_assets"] = failed
    telemetry.event("reconstruct_done", scan=scan, status=plan["status"], generated=len(generated))
    return plan
