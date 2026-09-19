"""stage.py — the layout pass as the pipeline calls it.

One function, `run_layout(scan)`, run from `scene_init/flow.py` in the window immediately after the
box merge and before crops. This low-level repair helper returns failures as structured summaries.
Pipeline callers must then call `require_valid_layout`: the required gate checks saved geometry,
writes its verdict, and stops the pipeline on remaining errors or an unsuccessful repair.

    LR_LAYOUT=0        skip the pass entirely
    LR_LAYOUT_AGENT=0  disable photographic agent repair (enabled by default when needed;
                       the mandatory validation gate still blocks unresolved layouts)
    The production fidelity gate disallows dropping measured objects; splitting a merged unit
    back into all its original members is allowed.
    LR_LAYOUT_VIZ=0    do not draw the before/after plan (on by default — see report.py; it is
                       string building, costs milliseconds, and a run that repairs a room without
                       leaving a picture of the repair cannot be checked afterwards)
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any

__all__ = ["run_layout"]


def _enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "no", "")


def _visualize(scan: str, before: dict[str, Any], after: dict[str, Any],
               scene_data_dir: Path, **kwargs) -> str | None:
    """Draw the pass and save it. Returns the path, or None if it was off or could not be drawn.

    The canonical copy sits beside ``layout_report.json``, tied to the data it describes. A second
    goes under ``traces/``, which is where a person browsing a finished run actually looks — the
    stage log for this pass is already there, and a plan filed three levels deeper inside the
    preprocessing work tree is one nobody opens.

    Its own try/except, deliberately narrow: the repair has already been written to disk by the
    time this runs, and failing to draw a picture of a completed repair must not report the repair
    as failed.
    """
    if not _enabled("LR_LAYOUT_VIZ"):
        return None
    try:
        from . import report

        path = report.write(scene_data_dir / "layout.html", scan, before, after, **kwargs)
        try:
            from litereality_agent.pipeline.scene_init import paths as config

            traces = config.traces_dir(scan) / "layout.html"
            if (config.scene_data_dir(scan).resolve() == scene_data_dir.resolve()
                    and traces.resolve() != path.resolve()):
                traces.parent.mkdir(parents=True, exist_ok=True)
                traces.write_bytes(path.read_bytes())
                path = traces
        except Exception:       # noqa: BLE001 — a standalone run has no traces tree; keep the first
            pass
        print(f"  [layout] plan written to {path}", flush=True)
        return str(path)
    except Exception as exc:    # noqa: BLE001
        print(f"  [layout] visualization unavailable (non-fatal): "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None


def _publish_shell(shell: dict[str, Any], scene_data_dir: Path, dropped: list[str]) -> None:
    """Write the boxes the object stage was built against, for the room export to place into.

    The scene stage does NOT read objects.pkl. `export_room` re-extracts the room from `room.usdz`
    under Blender and embeds THAT as the SHELL in `Room.py`, which is what places every GLB — so a
    repair made here reached the crops, the references and the generated extents, and then the
    assembler put the asset back in the unrepaired box. A counter trimmed to 63 cm was generated at
    63 cm and scaled into the 87 cm box the scan measured, which is the "generated at the wrong
    extent and then squashed to fit" failure this stage exists to prevent, reintroduced one stage
    later.

    Objects only, and deliberately so. Walls, openings and the floor stay the export's, because the
    layout pass never touches them and has no business being their source of truth.
    """
    # Unlike the visualization, this is the required hand-off to the room assembler.
    _write_shell(shell, scene_data_dir, dropped)


def _write_shell(shell: dict[str, Any], scene_data_dir: Path, dropped: list[str]) -> None:
    payload = {
        "scan": scene_data_dir.name,
        "source": "scene_init/layout",
        "note": "Validated output placement and extents, overlaid by room_ops export. "
                "Photographic evidence retains the original scan geometry.",
        "objects": {oid: {"category": o.get("category"), "center": list(o["center"]),
                          "size": list(o["size"]), "yaw": o.get("yaw", 0.0)}
                    for oid, o in (shell.get("objects") or {}).items()},
        "dropped": list(dropped),
    }
    (scene_data_dir / "layout_shell.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def run_layout(scan: str, *, scene_data_dir: str | Path | None = None,
               use_agent: bool | None = None) -> dict[str, Any]:
    """Repair one scan's layout in place. Returns a summary; never raises."""
    if not _enabled("LR_LAYOUT"):
        print("  [layout] disabled ($LR_LAYOUT=0)", flush=True)
        return {"disabled": True}

    try:
        from . import adapter
        from .adjust import check

        if scene_data_dir is None:
            from litereality_agent.pipeline.scene_init import paths as config
            scene_data_dir = config.scene_data_dir(scan)
        scene_data_dir = Path(scene_data_dir)
        pkl = scene_data_dir / "objects.pkl"
        if not pkl.is_file():
            print(f"  [layout] no objects.pkl at {pkl} — skipping", flush=True)
            return {"skipped": "no objects.pkl"}

        shell = adapter.shell_from_scene_data(scene_data_dir)
        baseline_path = scene_data_dir / "layout_baseline.json"
        source = scene_data_dir / "objects.extracted.pkl"
        source_key = hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None
        if baseline_path.exists():
            saved = json.loads(baseline_path.read_text())
            if saved.get("source_key") != source_key:
                raise ValueError("Extraction baseline changed; re-extract before layout repair")
            baseline = saved["shell"]
        else:
            baseline = shell
            baseline_path.write_text(json.dumps({"source_key": source_key, "shell": shell}, indent=2))
        if use_agent is None:
            use_agent = _enabled("LR_LAYOUT_AGENT", "1")
        if use_agent:
            from .evidence import prepare

            prepare(scan, scene_data_dir, shell)
        found = check(shell)
        before = [v for v in found if v.severity == "error"]
        from .converge import fidelity_errors, solve
        if not before and not fidelity_errors(shell, baseline):
            # Still draw it. "Already sound" is a claim about the room, and the plan is what lets
            # someone see that the room it is a claim about is the room they scanned — an empty
            # object set and a correctly placed one both report zero violations.
            print(f"  [layout] {len(shell['objects'])} objects, already sound", flush=True)
            summary = {"before": 0, "after": 0, "moved": [], "resized": [], "dropped": []}
            _publish_shell(shell, scene_data_dir, [])
            drawn = _visualize(scan, shell, shell, scene_data_dir, violations_before=found,
                               violations_after=found, actions=[], mode="nothing to repair")
            if drawn:
                summary["visualization"] = drawn
            _save_verdict(scene_data_dir, summary)
            return summary

        shell.setdefault("meta", {})["batch_dir"] = str(scene_data_dir)
        repaired, _moves, actions, progress = solve(shell, baseline, use_agent=use_agent)

        settled = check(repaired)
        after = [v for v in settled if v.severity == "error"]
        entries = pickle.load(open(pkl, "rb"))
        changed = adapter.apply_to_objects(entries, repaired)

        backup = pkl.with_suffix(".pre_layout.pkl")
        if not backup.exists():
            backup.write_bytes(pkl.read_bytes())
        with open(pkl, "wb") as handle:
            pickle.dump(entries, handle)

        report = {"before": len(before), "after": len(after),
                  "actions": [a.get("action") for a in actions], **changed,
                  "remaining": [str(v) for v in after], **progress, "decisions": actions}
        if progress["fidelity_errors"]:
            report["error"] = "Cumulative fidelity limits exceeded: " + ", ".join(progress["fidelity_errors"])
        _publish_shell(repaired, scene_data_dir, changed["dropped"])
        drawn = _visualize(scan, shell, repaired, scene_data_dir, violations_before=found,
                           violations_after=settled, actions=actions,
                           mode="agent-assisted" if use_agent else "deterministic")
        if drawn:
            report["visualization"] = drawn
        _save_verdict(scene_data_dir, report)

        print(f"  [layout] {len(before)} violations -> {len(after)}   "
              f"moved {len(changed['moved'])}, resized {len(changed['resized'])}, "
              f"dropped {len(changed['dropped'])}", flush=True)
        for line in report["remaining"]:
            print(f"  [layout]   unresolved: {line}", flush=True)
        return report
    except Exception as exc:                    # noqa: BLE001 — never break init over the layout
        print(f"  [layout] repair FAILED (pipeline validation will reject this result): "
              f"{type(exc).__name__}: {exc}", flush=True)
        result = {"error": str(exc)}
        if scene_data_dir is not None and Path(scene_data_dir).is_dir():
            try:
                _save_verdict(Path(scene_data_dir), result)
            except OSError:
                pass  # the caller still rejects the returned error when disk writes fail
        return result


def _save_verdict(scene_data_dir, result):
    from .validation import LayoutValidationError, require_valid_layout

    try:
        require_valid_layout(scene_data_dir.name, result, scene_data_dir=scene_data_dir)
    except LayoutValidationError:
        pass  # callers still enforce the required gate; standalone runs get the same verdict
    (scene_data_dir / "layout_report.json").write_text(json.dumps(result, indent=2))
