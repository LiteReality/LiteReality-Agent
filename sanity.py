#!/usr/bin/env python3
"""Pre-run sanity check — verify the full pipeline stack BEFORE a run.

The point is to catch SILENT DEGRADATION: components that "work" by quietly falling back to a
weaker path (e.g. DINOv2 chair grouping falling back to hand-crafted CV features because
torchvision is missing) instead of failing. Those cost you quality without an error. This asserts
the intended path is actually available, and exits non-zero if any critical check fails.

    uv run python sanity.py [<scan>]                 # fast: imports + config
    SANITY_DEEP=1 uv run python sanity.py [<scan>]   # also loads each model + runs one inference

Every failure prints how to fix it, and the end prints a copy-paste block (a `.env` append + any
commands to run) so you can just follow it.

Exit 0 = safe to run · Exit 1 = a critical check failed (do not run until fixed).
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAILS: list[str] = []
WARNS: list[str] = []
ENV_FIXES: list[tuple[str, str, str]] = []  # (KEY, value, comment) — lines to add to .env
CMD_FIXES: list[tuple[str, str]] = []       # (command, comment) — commands to run
NOTE_FIXES: list[str] = []                  # freeform manual steps


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


def _register(env, cmd, note) -> None:
    """Print the concrete fix inline and stash it for the copy-paste block at the end."""
    if env:
        key, val, comment = env
        ENV_FIXES.append((key, val, comment))
        print(f"      {_c('2', '→ fix:')} add to .env → {_c('36', f'{key}={val}')}"
              + (f"   {_c('2', '(' + comment + ')')}" if comment else ""))
    if cmd:
        command, comment = cmd
        CMD_FIXES.append((command, comment))
        print(f"      {_c('2', '→ fix: run')} {_c('36', command)}"
              + (f"   {_c('2', '(' + comment + ')')}" if comment else ""))
    if note:
        NOTE_FIXES.append(note)
        print(f"      {_c('2', '→ fix:')} {_c('36', note)}")


def ok(msg: str) -> None:
    print(f"  {_c('32', '✓')} {msg}")


def fail(msg: str, *, env=None, cmd=None, note=None) -> None:
    print(f"  {_c('31', '✗')} {msg}")
    _register(env, cmd, note)
    FAILS.append(msg)


def warn(msg: str, *, env=None, cmd=None, note=None) -> None:
    print(f"  {_c('33', '!')} {msg}")
    _register(env, cmd, note)
    WARNS.append(msg)


def _load_dotenv() -> bool:
    """Populate os.environ from ./.env AND ./models.env so a bare `python sanity.py` sees the SAME
    config as the CLI (which sources both). Without .env, a key sitting right there reads as
    "unset" here — confusing, because the real run would have it. Without models.env, every model
    choice reads as the CODE default instead of the one the run will use, so this script reports a
    checkpoint (`LR_DINO_MODEL`, `LR_DINO_EMBED_MODEL`) or image model the pipeline will not touch
    — the opposite of its job. Existing shell env wins. Returns True if a .env file was found.

    The package's own loader is preferred: it is what `uv run -m litereality_agent` calls, so
    sanity cannot drift from the real run, and it already parses models.env's
    `export K="${K:-default}"   # comment` grammar (getting the trailing comment right is fiddly —
    see models/config.py._value). The inline parse below is the fallback for a half-installed tree,
    where the package is unimportable and only .env's paths and keys still mean anything."""
    envp = ROOT / ".env"
    try:
        from litereality_agent.settings import load_settings

        load_settings(ROOT).apply_environment()
        return envp.is_file()
    except Exception:  # noqa: BLE001 — sanity must run even from a half-installed tree
        pass
    if not envp.is_file():
        return False
    for raw in envp.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]  # strip surrounding quotes
        os.environ.setdefault(key, val)
    return True


def _import(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def _deep() -> bool:
    """SANITY_DEEP=1 actually loads each model and runs one inference (slow — downloads
    weights on first run) instead of only checking that imports succeed."""
    return os.environ.get("SANITY_DEEP", "0") == "1"


def _hosted() -> bool:
    """True when detection, embedding, and gen3d run on Modal rather than in this process.

    Asked of `registry`, not of an env var, so this can never disagree with what a run will
    actually do. It matters because the local torch stack is several GB that the DEFAULT install
    never uses: `registry.detection_from_settings()` returns a Modal service whenever credentials
    are configured, and `detector.embed_available()` then routes embeddings through it too. Before
    this check existed, sanity failed the recommended install for missing torch and told the user
    to `uv pip install torch` — the one thing hosting exists to avoid.
    """
    try:
        from litereality_agent.settings import load_settings

        return bool(load_settings(ROOT).modal_configured())
    except Exception:  # noqa: BLE001 — a half-installed tree just means "not hosted"
        return False


def _tiny_png() -> str:
    """Write a small RGB image to the temp dir and return its path (for smoke tests)."""
    import tempfile

    from PIL import Image

    p = Path(tempfile.gettempdir()) / "lr_sanity_probe.png"
    Image.new("RGB", (64, 64), (127, 127, 127)).save(p)
    return str(p)


# What each stage actually touches. Checking everything on every run gates a stage-2 run on a
# torch install or hosted endpoint it will never call, and a failure there aborts a pipeline
# that would have run perfectly well.
STAGE_CHECKS = {
    "scene_init": {"deps", "dinov2", "detect", "blender", "providers", "gen3d", "agent_cli", "scan"},
    # "collision" belongs to stage 2 only: qc_collision runs in the QC stage, and stage 1 never
    # touches FCL.
    "realism_authoring": {"blender", "agent_cli", "scan", "collision"},
}
STAGE_CHECKS["all"] = set().union(*STAGE_CHECKS.values())


def _check_local_dinov2() -> None:
    """Chair grouping, running in this process. Silent-degradation risk: unavailable DINOv2 does
    not raise — it falls back to hand-crafted CV features and quietly loses grouping quality."""
    try:
        from litereality_agent.models.device import pick_device

        dev = pick_device()
        label = {"cuda": " (NVIDIA GPU)", "mps": " (Apple Silicon GPU)",
                 "cpu": " (CPU — works, just slower; fine on a laptop)"}.get(dev, "")
        ok(f"torch compute device: {dev}{label}")
    except Exception as e:  # noqa: BLE001
        fail(f"could not select a torch device: {type(e).__name__}: {e}")
    try:
        from litereality_agent.models.dinov2 import inference as dino_embed

        mid = dino_embed.default_model_id()
        if not dino_embed.available():
            fail("DINOv2 NOT available — chair grouping would SILENTLY fall back to weaker CV features.",
                 cmd=("uv pip install torchvision", "match your torch build / CUDA version"))
        elif _deep():
            vec = dino_embed.embed_paths([_tiny_png()])[0]
            if vec:
                ok(f"DINOv2 loaded + inferred: {mid} ({len(vec)}-d embedding)")
            else:
                fail(f"DINOv2 {mid} returned an empty embedding — model ran but produced nothing.",
                     note="rm -rf ~/.cache/huggingface and re-run to re-download, or check GPU/RAM")
        else:
            ok(f"DINOv2 grouping available ({mid})  [import check only — SANITY_DEEP=1 to load+infer]")
    except Exception as e:  # noqa: BLE001
        fail(f"DINOv2 load/inference failed: {type(e).__name__}: {e}",
             cmd=("uv pip install torch torchvision transformers",
                  "and allow network to huggingface.co so facebook/dinov2-small can download"))


def _check_local_detect() -> None:
    """Object detection, running in this process."""
    try:
        from litereality_agent.models.grounding_dino import inference as dino_detect

        mid = dino_detect.default_model_id()
        if not dino_detect.available():
            fail("GroundingDINO NOT available — object detection would fail.",
                 cmd=("uv pip install torch transformers", "HF-transformers backend, NOT groundingdino-py"))
        elif _deep():
            from PIL import Image

            dino_detect.detect(Image.open(_tiny_png()), "chair")  # runs, may return 0 detections — fine
            ok(f"GroundingDINO loaded + ran a detection: {mid}")
        else:
            ok(f"GroundingDINO detector available ({mid})  [import check only — SANITY_DEEP=1 to load+infer]")
    except Exception as e:  # noqa: BLE001
        fail(f"GroundingDINO load/inference failed: {type(e).__name__}: {e}",
             cmd=("uv pip install torch transformers",
                  "and allow network to huggingface.co so IDEA-Research/grounding-dino-tiny can download"))


def _check_hosted_dino(*, embed: bool) -> None:
    """The same two capabilities when they run on Modal.

    Fast mode confirms a service can be constructed at all — that is what `registry` will do at
    run time, so a missing app name or half a token pair surfaces here rather than mid-run. Deep
    mode actually calls it, which is the only way to catch an app that was configured but never
    deployed.
    """
    what = "DINOv2 embedding" if embed else "GroundingDINO detection"
    try:
        from litereality_agent.models.registry import detection_from_settings

        service = detection_from_settings()
    except Exception as e:  # noqa: BLE001
        fail(f"{what}: could not build the Modal service: {type(e).__name__}: {e}",
             cmd=("uv run litereality setup", "deploys both model apps into your workspace"))
        return
    if service is None:
        fail(f"{what}: no runtime configured — neither Modal credentials nor GROUNDING_DINO_PYTHON.",
             env=("MODAL_TOKEN_ID", "ak-<your-token-id>",
                  "with MODAL_TOKEN_SECRET; create at modal.com/settings/tokens"))
        return
    app = os.environ.get("MODAL_DINO_APP") or "litereality"
    if not _deep():
        ok(f"{what} hosted on Modal ({app})  [config check only — SANITY_DEEP=1 to call it]")
        return
    try:
        from PIL import Image

        image = Image.open(_tiny_png())
        if embed:
            vec = service.embed([image])[0]
            if vec:
                ok(f"{what} ran on Modal ({app}, {len(vec)}-d embedding)")
            else:
                fail(f"{what} returned an empty embedding — the app ran but produced nothing.",
                     note=f"check the {app} logs at modal.com/apps")
        else:
            service.detect(image, "chair")  # may return 0 detections — fine
            ok(f"{what} ran on Modal ({app})")
    except Exception as e:  # noqa: BLE001
        fail(f"{what} failed on Modal: {type(e).__name__}: {e}",
             cmd=("uv run litereality setup", f"redeploys {app}; check modal.com/apps for its logs"))
    finally:
        close = getattr(service, "close", None)
        if close:
            close()


def main() -> int:
    args = sys.argv[1:]
    stage = "all"
    for i, a in enumerate(args):
        if a == "--stage" and i + 1 < len(args):
            stage = args[i + 1]
        elif a.startswith("--stage="):
            stage = a.split("=", 1)[1]
    wanted = STAGE_CHECKS.get(stage, STAGE_CHECKS["all"])
    positional = [a for a in args if not a.startswith("--")]
    if stage in positional:
        positional.remove(stage)
    scan = positional[0] if positional else None
    sys.path.insert(0, str(ROOT))

    # Load the same dotenv file as the application.
    loaded = _load_dotenv()
    os.environ.setdefault("LR_CLASSIFY_PROVIDER", "claude")
    os.environ.setdefault("HARNESS_VLM", "claude")
    # Honor the documented alias names (BLENDER_PATH / GROUNDING_DINO_PYTHON / TRELLIS_PYTHON)
    # as in the CLI + agent/config.py; otherwise the check below falls back to `blender` on PATH.
    # No machine-specific path is baked in.
    for _alias, _canon in (("BLENDER_PATH", "LITEREALITY_BLENDER"),
                           ("BLENDER", "LITEREALITY_BLENDER"),
                           ("GROUNDING_DINO_PYTHON", "LR_DINO_PYTHON"),
                           ("TRELLIS_PYTHON", "LITEREALITY_TRELLIS_PYTHON")):
        if os.environ.get(_alias, "").strip() and not os.environ.get(_canon, "").strip():
            os.environ[_canon] = os.environ[_alias]
    envp = ROOT / ".env"
    print(_c("2", f"config: loaded {envp}" if loaded else f"config: no .env at {envp} (using shell env only)"))
    print()

    hosted = _hosted()

    if "deps" in wanted:
        print("── python dependencies ──")
        pip_pkg = {"PIL": "pillow"}  # import name → pip package name where they differ
        for m in ("openai", "PIL", "numpy", "trimesh"):
            if _import(m):
                ok(f"import {m}")
            else:
                fail(f"import {m}", cmd=(f"uv pip install {pip_pkg.get(m, m)}", ""))
        # The torch stack is only imported when the models run IN THIS PROCESS. Hosted, it is
        # several GB of nothing, so its absence is the expected state rather than a failure.
        for m in ("torch", "torchvision", "transformers"):
            if _import(m):
                ok(f"import {m}")
            elif hosted:
                ok(f"{m} not installed — not needed, models run on Modal")
            else:
                fail(f"import {m}", cmd=(f"uv pip install {pip_pkg.get(m, m)}", ""))

    if "dinov2" in wanted:
        print("── enhanced chair grouping (DINOv2) ──")
        if hosted:
            # `detector.embed_available()` routes embeddings through the Modal DINO service when
            # one is configured, so there is no local device to pick and no local weights to load.
            _check_hosted_dino(embed=True)
        else:
            _check_local_dinov2()

    if "detect" in wanted:
        print("── object detection (GroundingDINO, HF transformers) ──")
        if hosted:
            _check_hosted_dino(embed=False)
        else:
            _check_local_detect()


    if "blender" in wanted:
        print("── Blender ──")
        # THE resolver, not another copy of the probe. sanity had its own, which knew only
        # $LITEREALITY_BLENDER and `blender` on PATH — so on a stock macOS install (the binary lives
        # inside /Applications/Blender.app/Contents/MacOS/) it reported "not found" and aborted a run
        # the pipeline itself would have completed.
        try:
            from litereality_agent.room_ops.paths import find_blender

            binp = find_blender()
        except SystemExit:
            binp = None
        except Exception:  # noqa: BLE001 — sanity must run even from a half-installed tree
            bl = os.environ.get("LITEREALITY_BLENDER", "")
            binp = (Path(bl) / "blender") if bl else shutil.which("blender")
        if binp and Path(binp).exists():
            try:
                v = subprocess.run([str(binp), "--version"], capture_output=True, text=True, timeout=30)
                ok(f"Blender runs: {v.stdout.splitlines()[0] if v.stdout else '(no version line)'}")
            except Exception as e:  # noqa: BLE001
                fail(f"Blender present but won't run: {type(e).__name__}: {e}",
                     cmd=(f"chmod +x {bl or '<blender-dir>'}/blender", "or install libs: libxi libxrender libgl"))
        else:
            fail("Blender not found.",
                 env=("BLENDER_PATH", "/path/to/blender-4.5-dir",
                      "the dir with the 'blender' binary. macOS: /Applications/Blender.app/Contents/MacOS"))

    if "collision" in wanted:
        print("── QC true-mesh collision (python-fcl) ──")
        # The QC stage runs qc_collision on EVERY default run, and the CLI guards it with `|| true`,
        # so a missing FCL does not fail the run — it just prints a traceback into the stage log and
        # the summary still ticks the stage green. That is exactly the silent degradation this
        # script exists to catch: the deterministic clash gate quietly stops running.
        try:
            import trimesh.collision

            trimesh.collision.CollisionManager()
            ok("python-fcl available (true-mesh clash resolver)")
        except Exception:  # noqa: BLE001
            fail("python-fcl NOT available — QC's true-mesh clash resolver will abort and the "
                 "run will SILENTLY skip it. It is a core dependency, so this means the env is "
                 "stale, not that an extra is missing.",
                 cmd=("uv sync --frozen --extra modal --group dev",
                      "or: uv pip install python-fcl networkx"))

    if "agent_cli" in wanted:
        print("── agent CLI (drives every authoring session) ──")
        # Which CLI is REQUIRED depends on the configured harness per role, so check the ones
        # actually selected. Checking `claude` unconditionally told a codex user their setup was
        # broken, and said nothing about the CLI their run would really need.
        roles = {"author": "authoring", "materials": "materials", "quality": "qc",
                 "refine": "refine", "procedural": "procedural objects"}
        binaries = {"claude": ("claude", "npm install -g @anthropic-ai/claude-code"),
                    "codex": ("codex", "npm install -g @openai/codex")}

        def harness_for(role: str) -> str:
            return (os.environ.get(f"LR_{role.upper()}_PROVIDER")
                    or os.environ.get("LR_AGENT_PROVIDER") or "claude").strip().lower()

        picked = {role: harness_for(role) for role in roles}
        for name in sorted(set(picked.values())):
            using = [roles[r] for r, n in picked.items() if n == name]
            binary, install = binaries.get(name, (name, f"install the {name} CLI"))
            if shutil.which(binary):
                ok(f"{binary} CLI on PATH ({' · '.join(using)})")
            else:
                fail(f"{binary} CLI not on PATH — it is the configured harness for "
                     f"{', '.join(using)}.",
                     cmd=(install, "then re-open the shell so PATH updates"))
        if "codex" in picked.values():
            warn("codex harness selected: no step-budget wind-down, no tool allowlist "
                 "(shell always available), no cost reporting.",
                 env=("LR_AGENT_PROVIDER", "claude", "set this to get the full-capability harness"))
        if picked["refine"] == "codex":
            fail("refine is set to codex, but its per-object `render_object` tool cannot be "
                 "bridged over stdio MCP — the pass aborts rather than run without it.",
                 env=("LR_REFINE_PROVIDER", "claude", "other roles can stay on codex"))

    if "providers" in wanted:
        # CHECK THE PROVIDER THAT IS ACTUALLY SELECTED. This asked for an OpenAI key unconditionally,
        # so a correctly configured Gemini setup failed sanity while the run itself would have been
        # fine — and, worse, an OpenAI key present alongside `LR_IMAGE_PROVIDER=gemini` passed here
        # and then failed on a missing Gemini key at the first object.
        provider = (os.environ.get("LR_IMAGE_PROVIDER") or "openai").strip().lower()
        model = os.environ.get("LR_OPENAI_IMAGE_MODEL") or "gpt-image-2"
        if provider not in ("openai", "gemini"):
            provider = "gemini" if model.lower().startswith("gemini") else "openai"
        print(f"── hosted models ({provider} images · Claude reasoning) ──")

        if provider == "gemini":
            if os.environ.get("GEMINI_API_KEY", ""):
                ok(f"GEMINI_API_KEY set (image provider is gemini, "
                   f"{os.environ.get('LR_GEMINI_IMAGE_MODEL') or 'gemini-2.5-flash-image'})")
            else:
                fail("LR_IMAGE_PROVIDER=gemini but GEMINI_API_KEY is unset — every reference "
                     "image will fall back to the raw evidence sheet.",
                     env=("GEMINI_API_KEY", "<your-gemini-key>",
                          "create at https://aistudio.google.com/apikey"))
        key = os.environ.get("OPENAI_API_KEY", "")
        if provider != "openai":
            ok("OPENAI_API_KEY not required — images come from gemini")
        elif not key:
            fail(f"OPENAI_API_KEY unset — reference image-gen ({model}) will fail.",
                 env=("OPENAI_API_KEY", "sk-<your-openai-key>", "create at https://platform.openai.com/api-keys"))
        else:
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"}
                )
                ok(f"OpenAI key valid (HTTP {urllib.request.urlopen(req, timeout=15).getcode()})")
            except urllib.error.HTTPError as e:
                fail(f"OpenAI key REJECTED (HTTP {e.code}).",
                     env=("OPENAI_API_KEY", "sk-<a-valid-key>",
                          "current key rejected — replace it and confirm billing at platform.openai.com/account/billing"))
            except Exception as e:  # noqa: BLE001
                warn(f"OpenAI key not verified (network): {type(e).__name__}",
                     note="check network / proxy, then re-run to confirm the key")
        if os.environ.get("LR_CLASSIFY_PROVIDER", "claude").lower() != "claude":
            warn("LR_CLASSIFY_PROVIDER is not 'claude' (policy is claude).",
                 env=("LR_CLASSIFY_PROVIDER", "claude", ""))
        else:
            ok("LR_CLASSIFY_PROVIDER=claude")
        ok("HARNESS_VLM=claude")

    if "gen3d" in wanted:
        print("── gen3d (TRELLIS) ──")
        if os.environ.get("MODAL_TRELLIS_APP"):
            ok("Modal TRELLIS app set (cloud gen3d)")
        else:
            warn("no MODAL_TRELLIS_APP — reconstruction has no cloud gen3d and needs a local GPU env.",
                 env=("MODAL_TRELLIS_APP", "litereality",
                      "deploy deploy/modal/app.py from the shared workspace; or use a local GPU"))

    if scan:
        print(f"── scan: {scan} ──")
        sd = Path(os.environ.get("LR_SCANS_DIR") or (ROOT / "scans_uploaded")) / scan
        if not sd.is_dir():
            fail(f"scan not found: {sd}",
                 env=("LR_SCANS_DIR", str(sd.parent), f"dir that contains the '{scan}' scan folder"))
        else:
            n = len(list(sd.glob("frame_*.jpg")))
            if n:
                ok(f"{n} capture frames")
            else:
                fail(f"no capture frames in {sd}",
                     note=f"the scan needs frame_*.jpg captures — re-export the RoomPlan scan into {sd}")
            if list(sd.glob('*.usdz')):
                ok("RoomPlan .usdz present")
            else:
                fail(f"no RoomPlan .usdz in {sd}", note=f"copy the RoomPlan .usdz export into {sd}")

    print()
    if FAILS:
        print(_c("31", f"SANITY FAILED — {len(FAILS)} critical issue(s). Fix before running."))
    else:
        print(_c("32", "SANITY OK") + (f"  ({len(WARNS)} warning(s))" if WARNS else ""))

    _print_fix_block()
    return 1 if FAILS else 0


def _print_fix_block() -> None:
    """Emit a copy-paste-ready to-do list: a single .env append, then commands, then manual steps."""
    if not (ENV_FIXES or CMD_FIXES or NOTE_FIXES):
        return
    print(_c("1", "\n═══ HOW TO FIX (copy-paste) ═══"))
    step = 0
    if ENV_FIXES:
        step += 1
        print(_c("1", f"\n{step}) Add these to .env  (replace every <...> with a real value first):\n"))
        print("cat >> .env <<'EOF'")
        for key, val, comment in ENV_FIXES:
            print(f"{key}={val}" + (f"    # {comment}" if comment else ""))
        print("EOF")
    if CMD_FIXES:
        step += 1
        print(_c("1", f"\n{step}) Run these commands:\n"))
        for command, comment in CMD_FIXES:
            print(f"{command}" + (f"    # {comment}" if comment else ""))
    if NOTE_FIXES:
        step += 1
        print(_c("1", f"\n{step}) Then handle manually:\n"))
        for note in NOTE_FIXES:
            print(f"- {note}")
    print(_c("2", "\nThen re-run: uv run python sanity.py   "
                  "(the CLI loads .env automatically, so env fixes apply on the next run)"))


if __name__ == "__main__":
    sys.exit(main())
