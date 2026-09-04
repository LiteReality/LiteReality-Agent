"""stage.py — the layout pass as the pipeline calls it.

One function, `run_layout(scan)`, run from `scene_init/flow.py` in the window immediately after the
box merge and before crops. It is written to the same contract as the merge it follows: it never
raises into the caller. A layout pass that fails must leave the scan exactly as it found it and let
the run continue unrepaired, because the alternative — taking down an otherwise good reconstruction
over a geometry edge case — is worse than the misplacement it was trying to fix.

    LR_LAYOUT=0        skip the pass entirely
    LR_LAYOUT_AGENT=1  let the agent look at the reference photographs for what geometry cannot
                       settle (off by default: it costs model calls, and the deterministic pass
                       already clears 19 of 21 captures on its own)
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

__all__ = ["run_layout"]


def _enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) not in ("0", "false", "no", "")


def run_layout(scan: str, *, scene_data_dir: str | Path | None = None,
               use_agent: bool | None = None) -> dict[str, Any]:
    """Repair one scan's layout in place. Returns a summary; never raises."""
    if not _enabled("LR_LAYOUT"):
        print("  [layout] disabled ($LR_LAYOUT=0)", flush=True)
        return {"disabled": True}

    try:
        from . import adapter
        from .adjust import check
        from .repair import repair

        if scene_data_dir is None:
            from litereality_agent.pipeline.scene_init import paths as config
            scene_data_dir = config.scene_data_dir(scan)
        scene_data_dir = Path(scene_data_dir)
        pkl = scene_data_dir / "objects.pkl"
        if not pkl.is_file():
            print(f"  [layout] no objects.pkl at {pkl} — skipping", flush=True)
            return {"skipped": "no objects.pkl"}

        shell = adapter.shell_from_scene_data(scene_data_dir)
        before = [v for v in check(shell) if v.severity == "error"]
        if not before:
            print(f"  [layout] {len(shell['objects'])} objects, already sound", flush=True)
            return {"before": 0, "after": 0, "moved": [], "resized": [], "dropped": []}

        if use_agent is None:
            use_agent = _enabled("LR_LAYOUT_AGENT", "0")
        if use_agent:
            shell.setdefault("meta", {})["batch_dir"] = str(scene_data_dir)
            from .repair import solve_v13
            repaired, _moves, actions = solve_v13(shell, detail=True)
        else:
            repaired, _moves, actions = repair(shell)

        after = [v for v in check(repaired) if v.severity == "error"]
        entries = pickle.load(open(pkl, "rb"))
        changed = adapter.apply_to_objects(entries, repaired)

        backup = pkl.with_suffix(".pre_layout.pkl")
        if not backup.exists():
            backup.write_bytes(pkl.read_bytes())
        with open(pkl, "wb") as handle:
            pickle.dump(entries, handle)

        report = {"before": len(before), "after": len(after),
                  "actions": [a.get("action") for a in actions], **changed,
                  "remaining": [str(v) for v in after]}
        (scene_data_dir / "layout_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")

        print(f"  [layout] {len(before)} violations -> {len(after)}   "
              f"moved {len(changed['moved'])}, resized {len(changed['resized'])}, "
              f"dropped {len(changed['dropped'])}", flush=True)
        for line in report["remaining"]:
            print(f"  [layout]   unresolved: {line}", flush=True)
        return report
    except Exception as exc:                    # noqa: BLE001 — never break init over the layout
        print(f"  [layout] FAILED (non-fatal, continuing unrepaired): "
              f"{type(exc).__name__}: {exc}", flush=True)
        return {"error": str(exc)}
