"""Per-object refinement.

For each procedural object (one with editable `object.py` build code), a self-paced model session
tightens its shape and proportions against the clean reference image:
  read object.py / object.md + the reference -> render a 4-view turntable beside the reference ->
  view it -> edit object.py to close the gap -> repeat.

Neural (TRELLIS) objects are baked meshes and are not refinable this way, so they are skipped.

    python -m lrauthor.pipeline.author.realism.refine_objects --scene <scene dir>

`--scene` is the scene package the seed stage wrote; it supplies the room, references, and
capture, and the object list defaults to every procedural object in that room. The explicit
spelling still works: `--room <room dir> --refroot <object_refs root> --scan <scan dir>`.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from lrauthor import PACKAGE_ROOT, REPO_ROOT
from lrauthor.agent.tool_narration import hint_for
from lrauthor.fonts import font as _shared_font

# Blender execs this by path, so it must be absolute: the CWD of a stage is not the repo.
OBJVIEW = str(PACKAGE_ROOT / "room_ops" / "rendering" / "object_turntable.py")

OBJECTS = ["Table0", "Table1", "Wall1_Door_0", "Wall5_Window_0"]
RESULTS = Path("output/_results/obj_refine")
_SEM = None       # concurrency cap over object sessions (set in main)
_RETRIES = 3      # per-object SDK retry attempts
_BUDGET = 8.0     # per-object USD budget cap

# Max render->look->fix rounds the agent is told to do. Kept deliberately small: these
# stages are the long pole, and the first round or two lands most of the win.
ROUNDS = int(os.environ.get("LR_REFINE_ROUNDS", "2"))

PROMPT = """\
You are refining a PROCEDURALLY-modelled 3D {cat} named `{name}` so it matches the REAL SCANNED
CAPTURE of it.

Your working directory holds `object.py` (the Blender code that BUILDS this object), `object.md`
(its spec) and `textures/`. FIRST Read `object.md` and `object.py` IN FULL to understand exactly how
the shape is generated (the boxes/loops/parameters). Then Read the REAL SCAN CAPTURE FRAMES below —
these are full room photos (chosen by the view-picker as the frames where this {cat} is best visible),
so the {cat} is somewhere IN each frame, at a real angle and in context. LOCATE the {cat} in them and
use it as your ground-truth TARGET. Do NOT trust any cleaned-up/AI reference — only these real frames:
  {targets}

Use the `render_object` tool: it builds your CURRENT `object.py` and returns a montage — 4 turntable
views of YOUR build next to the real capture frame(s). READ that montage, find the {cat} in the real
frame, and compare SHAPE, PROPORTIONS and visible DETAIL.

**First decide whether anything actually needs changing.** If the build already reads as the same
{cat} as the reference — proportions roughly right, no structure obviously missing — then say so
and STOP. "No change needed" is a perfectly good, and common, result. Do not invent work: a
speculative edit to an object that was already fine usually makes it worse, and costs you the
build.

**If something IS clearly wrong, make small, high-value change — do not rebuild the object.**
Pick the max 3 most obvious mismatches and fix just them. In rough priority order:
- Overall structure
- PBR Materails
- proportions (height/width/depth ratios, leg thickness, top overhang, panel divisions);
- ONE piece of missing structure the reference clearly shows (a table's apron or stretchers; a
  door's panels or handle; a window's mullion or sill) added as a procedural part;
- the silhouette, so it reads like the reference from every angle.

Only fix what you can actually SEE is wrong in the reference. If the frames are too small, blurry
or occluded to judge some detail, leave that detail alone — guessing from bad evidence is worse
than leaving the current build. Do NOT rewrite `object.py`, do NOT refactor it, and do NOT chase
small details.

BUDGET: do not over-inspect. Read `object.md`, `object.py` and the frames ONCE, render once to see
where you stand, then either make the one edit or conclude no edit is warranted. Avoid re-cropping
or re-reading images you have already seen. Do AT MOST {rounds} render->look->fix rounds, and stop
as soon as you have either landed the change or decided none is needed.

Constraints: edit ONLY `object.py`; it MUST stay valid and build (render_object reports build
errors — fix them). Keep it PROCEDURAL (boxes/loops, no external mesh). Keep real-world metre scale
and roughly the same overall bounding size (it must still fit its slot in the room).
{artic}
When done, summarise what you changed and how close the build now is to the reference — or, if you
made no edit, say so and why the existing build was already an acceptable match.
"""

_ARTIC_CLAUSE = """
OPENING (articulation): this {cat} OPENS — `object.py` calls `set_articulation(...)` and
`animate_revolute/animate_prismatic(...)` to make the movable leaf swing/slide. render_object shows a
CLOSED and an OPENED panel. You MUST keep the opening working: the `set_articulation` + `animate_*`
calls must stay, the movable part must keep its own origin on the hinge line / slide axis, and the
OPENED panel must actually differ from CLOSED (swing on the correct edge, right direction and angle,
no clipping through the frame). Match the hinge side and swing to the real capture. If your edits
break the movement (opened looks identical to closed), fix the articulation wiring."""


def _quad(views_dir: Path) -> Image.Image:
    grid = Image.new("RGB", (640, 640), (30, 30, 30))
    for i in range(4):
        p = views_dir / f"view_{i}.png"
        if p.is_file():
            grid.paste(Image.open(p).convert("RGB").resize((320, 320)), ((i % 2) * 320, (i // 2) * 320))
    return grid


def _targets_grid(targets: list) -> Image.Image:
    """Real capture target(s) tiled (up to 2, letterboxed to preserve aspect)."""
    tgrid = Image.new("RGB", (640, 640), (30, 30, 30))
    ts = targets[:2]
    ch = 640 // max(1, len(ts))
    for i, tp in enumerate(ts):
        im = Image.open(tp).convert("RGB")
        im.thumbnail((640, ch - 4))
        tgrid.paste(im, ((640 - im.width) // 2, i * ch + (ch - im.height) // 2))
    return tgrid


def _font(s):
    try:
        return _shared_font(s)
    except Exception:
        return ImageFont.load_default()


def _montage(views_dir: Path, targets: list, out: Path, open_views: Path | None = None):
    """REAL capture | CURRENT build (closed). If `open_views` is given (articulated object), append
    an OPENED band so the swing/slide is visible in the same sheet."""
    panels = [(_targets_grid(targets), "REAL CAPTURE (target)"),
              (_quad(views_dir), "CURRENT build — closed")]
    if open_views is not None:
        panels.append((_quad(open_views), "CURRENT build — OPENED"))
    sheet = Image.new("RGB", (10 + 650 * len(panels), 690), (80, 80, 80))
    for i, (img, label) in enumerate(panels):
        x = 10 + 650 * i
        b = Image.new("RGB", (640, 30), (15, 15, 15))
        ImageDraw.Draw(b).text((8, 6), label, font=_font(18), fill=(255, 255, 255))
        sheet.paste(b, (x, 0))
        sheet.paste(img, (x, 35))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def _snapshot_views(obj_dir: Path, blender: str, views_dir: Path, open_frac: float = 0.0) -> bool:
    """Build the CURRENT object.py and render 4 turntable views into `views_dir` — isolated from the
    agent's live `_refine_views` dir so a before/after snapshot never clobbers an in-flight render.
    `open_frac` poses the articulation (0=closed, 1=most open) so the opening can be checked.
    Returns True on success. Never raises (a failed snapshot must not abort refinement) — but it
    does SAY WHY it failed. Staying silent is what let a broken Blender path go unnoticed: the only
    symptom was a missing before/after image, with nothing in the log connecting the two."""
    why = ""
    try:
        from lrauthor.room_ops.compile.fetch_textures import materialize
        tj = obj_dir / "textures.json"
        if tj.is_file():
            try:
                materialize(json.loads(tj.read_text()), obj_dir / "textures")
            except Exception:  # noqa: BLE001
                pass
        views_dir.mkdir(parents=True, exist_ok=True)
        built = views_dir.parent / f"{views_dir.name}_built.glb"
        b = subprocess.run([blender, "-b", "--python", str(obj_dir / "object.py"), "--",
                            str(obj_dir / "textures"), str(built)], capture_output=True, text=True)
        if b.returncode != 0 or not built.is_file():
            why = f"build rc={b.returncode}: {(b.stderr or b.stdout or '').strip()[-300:]}"
        else:
            r = subprocess.run([blender, "-b", "--python", str(OBJVIEW), "--",
                                str(built), str(views_dir), "4", str(open_frac)],
                               capture_output=True, text=True)
            if r.returncode == 0 and (views_dir / "view_0.png").is_file():
                return True
            why = f"render rc={r.returncode}: {(r.stderr or r.stdout or '').strip()[-300:]}"
    except Exception as e:  # noqa: BLE001
        why = f"{type(e).__name__}: {e}"
    print(f"  [snapshot] {views_dir.name} FAILED — no before/after image will be written. {why}",
          flush=True)
    return False


def _is_articulated(obj_dir: Path) -> bool:
    """True if this object declares an opening (a door/window/drawer swing or slide)."""
    try:
        return "set_articulation(" in (obj_dir / "object.py").read_text()
    except Exception:  # noqa: BLE001
        return False


def _missing_quad(label: str) -> Image.Image:
    g = Image.new("RGB", (640, 640), (30, 30, 30))
    ImageDraw.Draw(g).text((180, 300), label, font=_font(20), fill=(160, 160, 160))
    return g


def _beforeafter_sheet(name: str, targets: list, before: Path | None, after: Path, out: Path,
                       after_open: Path | None = None):
    """Comparison sheet: REAL capture | BEFORE-refine build | AFTER-refine build. For an articulated
    object an extra AFTER-OPENED panel is appended, so the effect of the editing — and that the
    opening still works — is visible at a glance."""
    panels = [(_targets_grid(targets), f"{name} — REAL capture"),
              (_quad(before) if before else _missing_quad("(no BEFORE build)"), "BEFORE refine"),
              (_quad(after), "AFTER refine")]
    if after_open is not None:
        panels.append((_quad(after_open), "AFTER — OPENED"))
    sheet = Image.new("RGB", (10 + 650 * len(panels), 690), (80, 80, 80))
    for i, (img, label) in enumerate(panels):
        x = 10 + 650 * i
        b = Image.new("RGB", (640, 30), (15, 15, 15))
        ImageDraw.Draw(b).text((8, 6), label, font=_font(18), fill=(255, 255, 255))
        sheet.paste(b, (x, 0))
        sheet.paste(img, (x, 35))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def _render_object_sync(obj_dir: Path, targets: list, blender: str, tag: str, counter: dict) -> dict:
    """Materialize textures -> build object.py -> 4-view render -> montage vs reference. Returns paths."""
    from lrauthor.room_ops.compile.fetch_textures import materialize
    tj = obj_dir / "textures.json"
    if tj.is_file():
        try:
            materialize(json.loads(tj.read_text()), obj_dir / "textures")
        except Exception:  # noqa: BLE001
            pass
    built = obj_dir / "_refine_built.glb"
    b = subprocess.run([blender, "-b", "--python", str(obj_dir / "object.py"), "--",
                        str(obj_dir / "textures"), str(built)], capture_output=True, text=True)
    if b.returncode != 0 or not built.is_file():
        tail = (b.stdout + b.stderr).splitlines()[-6:]
        return {"error": "object.py failed to BUILD — fix it:\n" + "\n".join(tail)}
    views = obj_dir / "_refine_views"
    r = subprocess.run([blender, "-b", "--python", str(OBJVIEW), "--",
                        str(built), str(views), "4", "0.0"], capture_output=True, text=True)
    if r.returncode != 0 or not (views / "view_0.png").is_file():
        return {"error": "render failed:\n" + (r.stdout + r.stderr)[-500:]}
    # articulated object (door/window/drawer): also render the OPEN pose so the swing is checkable
    open_views = None
    note_open = ""
    if _is_articulated(obj_dir):
        ov = obj_dir / "_refine_views_open"
        ro = subprocess.run([blender, "-b", "--python", str(OBJVIEW), "--",
                             str(built), str(ov), "4", "1.0"], capture_output=True, text=True)
        if ro.returncode == 0 and (ov / "view_0.png").is_file():
            open_views = ov
            note_open = (" The THIRD panel shows your build fully OPENED — check the opening: hinge on "
                         "the correct edge, swing/slide direction and angle right, no clipping, and the "
                         "articulation actually moves (if it looks identical to closed, the "
                         "set_articulation/animate_* wiring is broken — fix it).")
    counter[tag] = counter.get(tag, 0) + 1
    out = RESULTS / f"{tag}_round{counter[tag]:02d}.png"
    _montage(views, targets, out, open_views=open_views)
    return {"montage": str(os.path.abspath(out)),
            "note": f"round {counter[tag]}: LEFT = the REAL capture photo(s) of this object (target), "
                    "then 4 turntable views of your current build. Read it and compare "
                    "shape/proportion/detail to the real photo." + note_open}


def build_server(obj_dir: Path, targets: list, blender: str, tag: str, counter: dict):
    from claude_agent_sdk import create_sdk_mcp_server
    from claude_agent_sdk import tool as sdk_tool

    @sdk_tool("render_object", "Build the current object.py and return a montage: the REAL capture "
                              "photo(s) next to 4 turntable views of your build. Read it to compare.", {})
    async def render_object(args):
        res = await asyncio.to_thread(_render_object_sync, obj_dir, targets, blender, tag, counter)
        return {"content": [{"type": "text", "text": json.dumps(res)}], "is_error": "error" in res}

    server = create_sdk_mcp_server(name="obj", version="1.0.0", tools=[render_object])
    return server, ["mcp__obj__render_object"]


async def refine_one(name: str, room: Path, refroot: Path, scan: Path, blender: str, model: str,
                     max_turns: int, counter: dict, provider: str | None = None):
    from lrauthor.agent import providers

    # `render_object` is built HERE, per object, closing over this session's targets and round
    # counter — it is not a registry tool, so there is nothing for the stdio MCP bridge to serve.
    # Without it the model cannot see its own build, and the whole prompt (N render rounds) is
    # meaningless. Fail loudly rather than run a paid session that was silently lobotomised.
    harness = providers.resolve("refine", provider)
    if "inproc_tools" not in harness.supports:
        raise RuntimeError(
            f"object refinement needs an in-process tool host; the {harness.name} harness has none. "
            f"Its `render_object` tool is per-session, not a registry tool, so it cannot be bridged "
            f"over stdio MCP yet. Set LR_REFINE_PROVIDER=claude (other roles can stay on "
            f"{harness.name}), or skip the refine pass."
        )
    obj_dir = room / "Objects" / "Procedural" / name
    if not (obj_dir / "object.py").is_file():
        return name, {"error": "no object.py"}
    # FRAME-SELECTION: the frames the object-init view-picker chose (best object visibility). Then use
    # the RAW FULL capture PHOTOS for those frames — NOT crops, NOT the nano reference. (Per request:
    # pick the right frames, optimise against the frames themselves.)
    stems = (glob.glob(str(refroot / "**" / name / "images" / "frame_*.jpg"), recursive=True)
             or glob.glob(str(refroot / "**" / name / "crops" / "frame_*.jpg"), recursive=True)
             or glob.glob(str(refroot / "**" / name / "frame_*_ranking_*.jpg"), recursive=True))
    nums = sorted({int(re.search(r"frame_(\d+)", os.path.basename(p)).group(1)) for p in stems})
    raw = [scan / f"frame_{n:05d}.jpg" for n in nums if (scan / f"frame_{n:05d}.jpg").is_file()]
    if not raw:
        return name, {"error": "no selected capture frames"}
    # upright the raw frames (the capture is stored 90 deg rotated) so the model sees them the right way up
    tdir = RESULTS / "_targets" / name
    tdir.mkdir(parents=True, exist_ok=True)
    targets = []
    for p in raw[:3]:
        up = tdir / p.name
        Image.open(p).convert("RGB").transpose(Image.Transpose.ROTATE_270).save(up)
        targets.append(up)
    # generic category label from the object name (e.g. Wall1_Door_0->door, Sofa0->sofa, Storage2->storage)
    base = re.sub(r"^Wall\d+_", "", name)
    cat = (re.sub(r"_?\d+$", "", base) or "object").lower()
    server, allowed = build_server(obj_dir, targets, blender, name, counter)

    # BEFORE snapshot: render the untouched build + stash its object.py, so the effect of this
    # refinement session is auditable afterwards (visual before/after + a code diff).
    badir = RESULTS / "_beforeafter"
    before_views = badir / f"{name}_before_views"
    have_before = _snapshot_views(obj_dir, blender, before_views)
    code_before = (obj_dir / "object.py").read_text()

    artic = _ARTIC_CLAUSE.format(cat=cat) if _is_articulated(obj_dir) else ""
    prompt = PROMPT.format(name=name, cat=cat, artic=artic, rounds=ROUNDS,
                           targets="\n".join(f"  - {os.path.abspath(p)}" for p in targets))
    dirs = sorted({str(REPO_ROOT), str(os.path.realpath(obj_dir)),
                   str(RESULTS.resolve())} | {str(os.path.realpath(p.parent)) for p in targets})
    spec = providers.SessionSpec(
        prompt=prompt,
        cwd=obj_dir,
        read_roots=tuple(Path(d) for d in dirs),
        model=model,
        max_turns=max_turns,
        budget_usd=_BUDGET,  # budget guardrail
        extra_mcp={"obj": server},
        extra_allowed=tuple(allowed),
    )
    t0 = time.monotonic()
    calls = 0
    counts: dict = {}
    summary = ""
    cost = None
    # RETRY guardrail: the SDK occasionally dies mid-session ("returned an error result: success")
    # under load. Retry the object up to _RETRIES times (its edits so far persist on disk).
    last_err = None
    for attempt in range(1, _RETRIES + 1):
        try:
            async with _SEM:
                async for m in harness.run(spec):
                    for block in getattr(m, "content", []) or []:
                        if type(block).__name__ == "ToolUseBlock":
                            calls += 1
                            nm = (getattr(block, "name", "?") or "?").split("__")[-1]
                            counts[nm] = counts.get(nm, 0) + 1
                            # Objects refine concurrently, so the line leads with WHICH object.
                            hint = hint_for(nm, getattr(block, "input", {}) or {}, 34)
                            print(f"  [{name}] {nm:<14} {hint}".rstrip(), flush=True)
                    if isinstance(m, providers.SessionResult):
                        summary = m.result or ""
                        cost = m.total_cost_usd
            last_err = None
            break  # session completed
        except Exception as e:  # noqa: BLE001 — a failed object must not kill the scene
            last_err = e
            print(f"  [{name}] attempt {attempt}/{_RETRIES} error: {type(e).__name__}: {e}", flush=True)
            if attempt < _RETRIES:
                await asyncio.sleep(5 * attempt)
    # NOTE: a failed session is NOT an empty one. The agent edits `object.py` in place as it goes,
    # so a run that dies on "Reached maximum number of turns" (or the budget cap) usually leaves
    # real edits behind. Returning here — as this did — threw away the AFTER render, the
    # before/after sheet and the code diff for exactly the objects whose result you most need to
    # look at, and left `refine_beforeafter.png` unwritten whenever any object errored. So fall
    # through and produce the evidence either way; the error is reported alongside it below.
    dt = round(time.monotonic() - t0, 1)
    # AFTER snapshot: render the refined build, write the before/after visual + a unified code diff.
    after_views = badir / f"{name}_after_views"
    have_after = _snapshot_views(obj_dir, blender, after_views)
    after_open = None
    if _is_articulated(obj_dir):  # also capture the OPEN pose so the opening is shown before/after
        aov = badir / f"{name}_after_open_views"
        if _snapshot_views(obj_dir, blender, aov, open_frac=1.0):
            after_open = aov
    ba_png = badir / f"{name}_beforeafter.png"
    try:
        if have_after:
            _beforeafter_sheet(name, targets, before_views if have_before else None, after_views,
                               ba_png, after_open=after_open)
    except Exception:  # noqa: BLE001
        pass
    code_after = (obj_dir / "object.py").read_text()
    diff = "".join(difflib.unified_diff(
        code_before.splitlines(True), code_after.splitlines(True),
        fromfile=f"{name}/object.py (before)", tofile=f"{name}/object.py (after)"))
    (badir / f"{name}.diff").write_text(diff)
    added = sum(1 for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in diff.splitlines() if ln.startswith("-") and not ln.startswith("---"))
    # An errored session that still changed the code is a PARTIAL result, not a failure with
    # nothing to show — say which it was, and keep the sheet/diff either way.
    status = "done" if last_err is None else ("partial" if (added or removed) else "no-edit")
    print(f"== {name} {status} {dt}s calls={calls} {counts} cost=${cost} "
          f"code Δ +{added}/-{removed}"
          + (f" ({type(last_err).__name__}: {last_err})" if last_err is not None else "")
          + " ==", flush=True)
    res = {"seconds": dt, "calls": calls, "counts": counts, "cost": cost,
           "rounds": counter.get(name, 0), "summary": summary[:1200],
           "beforeafter": str(ba_png) if ba_png.is_file() else None,
           "diff": str(badir / f"{name}.diff"), "code_delta": {"added": added, "removed": removed}}
    if last_err is not None:
        res["error"] = f"{type(last_err).__name__}: {last_err}"
        res["partial"] = bool(added or removed)
    return name, res


def _blender() -> str:
    """The Blender BINARY — always via THE resolver, never `$LITEREALITY_BLENDER` raw.

    This is load-bearing, not defensive. The README and .env.example both document `BLENDER_PATH`
    as the install DIRECTORY ("the folder containing the `blender` binary, not the binary itself"),
    and run.sh forwards it verbatim. Passing that straight to `subprocess.run` tries to execute a
    directory — `PermissionError: Permission denied: /Applications/Blender.app/Contents/MacOS` —
    and `_snapshot_views` swallows the exception and returns False. The visible symptom is not an
    error at all: every before/after comparison image silently goes missing, because the sheet is
    only written when the snapshot succeeds. `find_blender` accepts either spelling and returns the
    binary. (A blank BLENDER_PATH is the same story: `os.environ[...]` raised KeyError, or handed
    over an empty string.)"""
    from lrauthor.room_ops.paths import find_blender

    return find_blender()


def _procedural_objects(room: Path) -> list[str]:
    """Every refinable object in the room — the ones with editable `object.py` build code.

    Deriving this from the room lets the pass run from a scene package alone. The module-level
    `OBJECTS` list is a last-resort fallback for a
    room whose Objects/Procedural dir is missing.
    """
    procedural = room / "Objects" / "Procedural"
    if not procedural.is_dir():
        return list(OBJECTS)
    return sorted(p.name for p in procedural.iterdir() if p.is_dir())


async def main():
    global RESULTS, _SEM, _RETRIES, _BUDGET
    from lrauthor.pipeline.author import arguments as stage_args

    ap = argparse.ArgumentParser(description=__doc__)
    stage_args.add_scene_arg(ap)
    ap.add_argument("--room", default=None)
    ap.add_argument("--refroot", default=None)
    ap.add_argument("--scan", default=None)
    ap.add_argument("--model", default=os.environ.get("HARNESS_MODEL", "claude-opus-5"))
    # 100, not 25: at 25 an object typically spent every turn just READING (object.md, object.py,
    # 3 capture frames, then crops of them) and died on "Reached maximum number of turns" before
    # landing an edit — 7 of 8 objects on a real scene. The prompt now asks for ONE focused change
    # rather than an exhaustive rebuild, so the extra headroom goes into finishing, not wandering.
    ap.add_argument("--max-turns", type=int, default=int(os.environ.get("LR_REFINE_MAX_TURNS", "100")))
    ap.add_argument("--rounds", type=int, default=int(os.environ.get("LR_REFINE_ROUNDS", "2")),
                    help="max render->look->fix rounds the agent is told to do (default 2)")
    ap.add_argument("--objects", default=None,
                    help="comma-separated (default: every procedural object in the room)")
    ap.add_argument("--results", default=None, help="per-scene results dir")
    ap.add_argument("--concurrency", type=int, default=2, help="max object sessions at once")
    ap.add_argument("--retries", type=int, default=3, help="per-object SDK retry attempts")
    ap.add_argument("--budget", type=float, default=8.0, help="per-object USD budget cap")
    a = stage_args.bind(ap.parse_args(), need=("room", "refroot", "scan"))
    global ROUNDS
    ROUNDS = a.rounds
    _RETRIES = a.retries
    _BUDGET = a.budget
    RESULTS = Path(a.results) if a.results else RESULTS
    RESULTS.mkdir(parents=True, exist_ok=True)
    _SEM = asyncio.Semaphore(a.concurrency)
    objs = a.objects.split(",") if a.objects else _procedural_objects(Path(a.room))
    if not objs:
        print("no procedural objects to refine", flush=True)
        return
    counter: dict = {}
    print(f"== refine {objs} (<= {a.concurrency} at once) ==", flush=True)
    async def settle(coro):
        """Keep one provider failure local while allowing cancellation to stop the run."""
        try:
            return await coro
        except Exception as exc:  # noqa: BLE001 — reported with the object that failed
            return exc

    settled = await asyncio.gather(*(
        settle(refine_one(n, Path(a.room), Path(a.refroot), Path(a.scan), _blender(),
                          a.model, a.max_turns, counter)) for n in objs))
    results = []
    for n, r in zip(objs, settled):
        results.append((n, {"error": repr(r)}) if isinstance(r, Exception) else r)
    out = dict(results)
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2))

    # stack every object's before/after sheet into one contact sheet for a whole-scene glance
    rows = [Path(r["beforeafter"]) for r in out.values()
            if isinstance(r, dict) and r.get("beforeafter") and Path(r["beforeafter"]).is_file()]
    combined = None
    if rows:
        ims = [Image.open(p).convert("RGB") for p in rows]
        W = max(i.width for i in ims)
        H = sum(i.height for i in ims) + 4 * (len(ims) - 1)
        sheet = Image.new("RGB", (W, H), (40, 40, 40))
        y = 0
        for im in ims:
            sheet.paste(im, (0, y))
            y += im.height + 4
        combined = RESULTS / "refine_beforeafter.png"
        sheet.save(combined)

    print("\n== ALL DONE ==")
    n_err = 0
    for n, r in out.items():
        d = r.get("code_delta") or {}
        # A session can error AND still have landed edits (it writes object.py as it goes). Report
        # those as partial, with their delta — calling them a bare ERROR hides real work and makes
        # a tuning problem (turn cap) look like a broken stage.
        if r.get("error") and r.get("partial"):
            print(f"  {n}: ~ PARTIAL — code Δ +{d.get('added', '?')}/-{d.get('removed', '?')} "
                  f"then {str(r['error'])[:160]}")
            continue
        if r.get("error"):  # surface failures instead of printing rounds=None calls=None (silent no-op)
            n_err += 1
            print(f"  {n}: ✗ ERROR — {str(r['error'])[:300]}")
            continue
        # A clean session that changed nothing is a real verdict — the build already matched the
        # reference — not a no-op. Say that, so it is not mistaken for a silent failure.
        if not (d.get("added") or d.get("removed")):
            print(f"  {n}: ✓ no change needed — calls={r.get('calls')} cost=${r.get('cost')}")
            continue
        print(f"  {n}: rounds={r.get('rounds')} calls={r.get('calls')} "
              f"code Δ +{d.get('added', '?')}/-{d.get('removed', '?')} cost=${r.get('cost')}")
    if n_err:
        print(f"  ⚠ {n_err}/{len(out)} object(s) failed to refine (see errors above)")
    if combined:
        print(f"\n== before/after: {combined}  (per-object sheets + .diff in {RESULTS / '_beforeafter'}) ==")


if __name__ == "__main__":
    asyncio.run(main())
