"""Record what an agent pass did as structured events and a raw transcript.

Each authoring agent records its own events here while it runs:

    <traces>/authoring_trace.<pass>.jsonl   one JSON object per line, oldest first

One file PER PASS. They used to share a single path, and since each pass truncates its
file on start, the materials pass destroyed the authoring pass's trace mid-run.

Events (all carry `seq`, `t` epoch seconds, `dt` seconds since the session started):

    session_start  pass, model, room, profile, max_turns, scratch, prompt (verbatim)
    think          text        — the model's reasoning
    tool           tool, hint, id, args (the FULL input), and for edits `file`/`old`/`new`
    result         id, tool, secs, is_error, chars, output — what that call ANSWERED
    images         images[], saved[] — paths seen, plus durable copies under <traces>/img/
    session_end    calls, seconds, cost_usd, counts{tool: n}, summary

`tool` and `result` share an `id`, so a call and its answer can be paired; `secs` is how long
that one call took. Set `LR_TRACE_DETAIL=0` for the old slim trace (hints only, no outputs).

Alongside it, a VERBATIM sidecar — every SDK message exactly as it arrived:

    <traces>/authoring_trace.<pass>.raw.jsonl   `LR_TRACE_RAW=0` to disable

The curated events are a reading of the run and can only answer questions we anticipated; the
sidecar is the source they were derived from, for the ones we did not.

`trace_report.py` reads this back into a readable timeline, so a default run is
documented without any extra step.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import fields, is_dataclass
from pathlib import Path

# Tool-input keys worth keeping, in the order we prefer them for the one-line hint.
_HINT_KEYS = ("query", "target", "file_path", "path", "image", "pattern", "prompt", "command")
# Which tools mean "an image was produced or looked at" — the viewer surfaces these.
_IMAGE_TOOLS = {"render", "read_image", "critic", "select_views", "render_object", "render_wall"}
_EDIT_TOOLS = {"Edit", "Write", "NotebookEdit"}

# How much of each call to keep. The trace used to store a 200-character HINT of the input and
# NOTHING of the result, which loses the two things that actually explain a run: the analysis
# the model wrote (a 30-line PIL/numpy script measuring a dado band) and what that analysis
# ANSWERED. A `Bash` reading "Detect whiteboard columns on Wall0" is a label; the columns it
# found are the finding. jsonl on disk is cheap next to a paid authoring session, so the
# default is generous. `LR_TRACE_DETAIL=0` restores a slim trace.
_DETAIL = (os.environ.get("LR_TRACE_DETAIL", "1") or "1").strip().lower() not in ("0", "false", "no")
_CAP_INPUT = 20000 if _DETAIL else 400      # per tool-input value
_CAP_OUTPUT = 20000 if _DETAIL else 0       # tool result text
_CAP_THINK = 8000 if _DETAIL else 600       # one reasoning block
_CAP_EDIT = 12000                           # each side of a diff (unchanged: already generous)

# The curated trace above is SHAPED — it decides what is worth keeping, and every such decision
# can be wrong later ("why did it choose that wall?" is answerable only if the thing you need
# was on the keep-list). The raw sidecar makes that unrecoverable case impossible: every SDK
# message verbatim, in arrival order, in `<traces>/authoring_trace.<pass>.raw.jsonl`.
# `LR_TRACE_RAW=0` turns it off.
_RAW = (os.environ.get("LR_TRACE_RAW", "1") or "1").strip().lower() not in ("0", "false", "no")
# Base64 image payloads arrive inline and are worth nothing here — the images are saved as files.
_CAP_RAW_STR = 60000


def _plain(obj, depth: int = 0):
    """Any SDK object as JSON-able plain data, structure preserved.

    The SDK hands back dataclasses holding dataclasses (`AssistantMessage` → `[ToolUseBlock,
    TextBlock]`). `str()` on those is a lossy one-liner, so walk them instead — the point of a
    raw dump is that the shape survives too.
    """
    if obj is None or isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= _CAP_RAW_STR else obj[:_CAP_RAW_STR] + f"…<+{len(obj) - _CAP_RAW_STR} chars>"
    if depth > 8:  # cycles and pathological nesting: stop, do not recurse forever
        return str(obj)[:2000]
    if isinstance(obj, (list, tuple, set)):
        return [_plain(o, depth + 1) for o in obj]
    if isinstance(obj, dict):
        return {str(k): _plain(v, depth + 1) for k, v in obj.items()}
    if is_dataclass(obj) and not isinstance(obj, type):
        return {"__type__": type(obj).__name__,
                **{f.name: _plain(getattr(obj, f.name, None), depth + 1) for f in fields(obj)}}
    if hasattr(obj, "__dict__"):
        return {"__type__": type(obj).__name__,
                **{k: _plain(v, depth + 1) for k, v in vars(obj).items() if not k.startswith("_")}}
    return str(obj)[:2000]


def trace_path(room: Path | None = None, scan: str | None = None,
               pass_name: str = "authoring") -> Path:
    """Where to write. Prefers the scan's traces/ dir (so it sits beside the init trace), else the
    room's own directory, so a pass always records something even outside a full run."""
    for base in (os.environ.get("LITEREALITY_FINAL"), os.environ.get("LITEREALITY_OUTPUT")):
        if base and scan:
            d = Path(base) / scan / "scene_init" / "obj_stage" / "traces"
            if d.is_dir():
                return d / f"authoring_trace.{pass_name}.jsonl"
    if scan:
        for base in (os.environ.get("LITEREALITY_OUTPUT"), os.environ.get("LITEREALITY_FINAL")):
            if base:
                d = Path(base) / scan / "traces"
                d.mkdir(parents=True, exist_ok=True)
                return d / f"authoring_trace.{pass_name}.jsonl"
    if room:
        return Path(room) / f"authoring_trace.{pass_name}.jsonl"
    return Path(f"authoring_trace.{pass_name}.jsonl")


def _run_id_of(path: Path) -> str:
    """The `run_NNN` a trace belongs to, from the scratch dir its own session_start recorded.

    The raw sidecar carries no session_start of its own, so it borrows the curated trace's —
    they are written by the same session and must archive under one id.
    """
    for candidate in (path, Path(str(path).replace(".raw.jsonl", ".jsonl"))):
        try:
            with candidate.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("kind") == "session_start":
                        name = Path(str(rec.get("scratch") or "")).name
                        return name if re.fullmatch(r"run_\d+", name) else ""
                    break  # session_start is the first record or it is not there
        except (OSError, ValueError):
            continue
    return ""


def _archive(path: Path, tag: str = "") -> Path | None:
    """Move a previous run's trace into `<traces>/history/` before this run overwrites it.

    Tagged with the `run_NNN` of the run that WROTE it, so an archived trace and the scratch
    images it describes carry one identifier. That id is read from the file's own
    `session_start.scratch` — NOT from `$LR_RUN_ID`, which by this point names the run doing the
    archiving, one ahead of the run being archived. Falls back to the next free number for
    traces written before scratch directories were per-run.
    """
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        hist = path.parent / "history"
        hist.mkdir(parents=True, exist_ok=True)
        tag = tag or _run_id_of(path)
        if not tag:
            n = sum(1 for _ in hist.glob(f"{path.stem}.*{path.suffix}")) + 1
            tag = f"run_{n:03d}"
        dest = hist / f"{path.stem}.{tag}{path.suffix}"
        i = 2
        while dest.exists() and i < 1000:  # same id twice: keep both rather than clobber
            dest = hist / f"{path.stem}.{tag}_{i}{path.suffix}"
            i += 1
        path.replace(dest)
        return dest
    except OSError:
        return None


class AgentTrace:
    """Append-only recorder. Never raises into the caller — a broken trace must not fail a run."""

    def __init__(self, pass_name: str, room: Path | None = None, scan: str | None = None):
        self.pass_name = pass_name
        self.t0 = time.time()
        self.seq = 0
        self.counts: dict[str, int] = {}
        self.room = Path(room) if room else None  # a root for resolving relative image names
        self._calls: dict[str, tuple[str, float]] = {}  # tool_use_id -> (name, started)
        self._fh = None       # open handle, so a rotation underneath us cannot cross-contaminate
        self._raw_fh = None
        try:
            self.path = trace_path(room, scan, pass_name)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate only THIS pass's own file. Every pass used to share one path, so
            # starting the materials pass wiped the authoring pass's record mid-run.
            # ARCHIVE first: truncating outright meant only the most recent run was ever
            # traced, and re-running to check something destroyed the record of the run that
            # raised the question. The live path keeps its name so existing readers (the
            # viewer, the report) are unaffected.
            # Read the outgoing run's id BEFORE anything is truncated. The raw sidecar has no
            # session_start of its own and borrows the curated trace's — so resolving it after
            # the curated file was blanked silently fell back to counting, and the two halves
            # of one session archived under different ids.
            prev = _run_id_of(self.path)
            _archive(self.path, prev)
            self.path.write_text("", encoding="utf-8")
            # Hold the HANDLE, not the name. Every event used to re-open `self.path`, so when a
            # newer run archived (moved) that path mid-session, this session's remaining events
            # were appended into the NEW run's file — silently interleaving two sessions in one
            # trace and corrupting both. Writing to the inode means a rotation underneath us
            # leaves this session writing to its own file, wherever that file now lives.
            self._fh = self.path.open("a", encoding="utf-8")
            # The raw sidecar is append-only during a run, so it must be reset here too —
            # otherwise a re-run reads as one impossibly long session spliced onto the last.
            if _RAW:
                raw_path = self.path.with_suffix(".raw.jsonl")
                _archive(raw_path, prev)
                raw_path.write_text("", encoding="utf-8")
                self._raw_fh = raw_path.open("a", encoding="utf-8")
            self.ok = True
        except Exception:  # noqa: BLE001
            self.ok = False

    def _write(self, kind: str, **fields) -> None:
        if not self.ok:
            return
        self.seq += 1
        rec = {"seq": self.seq, "t": round(time.time(), 3),
               "dt": round(time.time() - self.t0, 1), "kind": kind, "pass": self.pass_name}
        rec.update({k: v for k, v in fields.items() if v not in (None, "", {}, [])})
        try:
            self._fh.write(json.dumps(rec, default=str) + "\n")
            self._fh.flush()  # a killed run must still leave everything up to the kill
        except Exception:  # noqa: BLE001
            self.ok = False

    def start(self, **fields) -> None:
        self._write("session_start", **fields)

    def raw(self, message) -> None:
        """Append one SDK message VERBATIM to the raw sidecar. Never raises.

        Deliberately unshaped: `tool`/`result` records are a reading of the run, and a reading
        made in advance cannot answer a question nobody had yet. This is the primary source the
        curated trace was derived from.
        """
        if not (_RAW and self.ok):
            return
        try:
            self._raw_fh.write(json.dumps({"t": round(time.time(), 3),
                                           "dt": round(time.time() - self.t0, 1),
                                           "msg": _plain(message)}, default=str) + "\n")
            self._raw_fh.flush()
        except Exception:  # noqa: BLE001 — a raw dump must never break a paid session
            pass

    def think(self, text: str) -> None:
        t = (text or "").strip()
        if t:
            self._write("think", text=t[:_CAP_THINK])

    def tool(self, name: str, inp: dict | None, tool_id: str = "") -> None:
        """One tool call: the hint a reader skims, plus the FULL input they need to reconstruct it.

        `tool_id` links this call to the result that answers it, and starts its clock.
        """
        name = (name or "?").split("__")[-1]
        self.counts[name] = self.counts.get(name, 0) + 1
        inp = inp or {}
        hint = next((str(inp[k]) for k in _HINT_KEYS if inp.get(k)), "")
        extra: dict = {}
        if tool_id:
            extra["id"] = tool_id
            self._calls[tool_id] = (name, time.time())
        # The whole input, not a 200-char prefix of one key: for `Bash` that prefix is the
        # description while the command is the method, and a run you cannot re-derive is an
        # anecdote rather than a record.
        args = {k: (v if isinstance(v, (int, float, bool)) else str(v)[:_CAP_INPUT])
                for k, v in inp.items() if v not in (None, "", [], {})}
        if args:
            extra["args"] = args
        if name in _EDIT_TOOLS and inp.get("file_path"):
            extra["file"] = Path(str(inp["file_path"])).name
            old = str(inp.get("old_string", ""))
            new = str(inp.get("new_string", inp.get("content", "")))
            if old or new:
                extra["delta_lines"] = new.count("\n") - old.count("\n")
                # Keep the actual before/after so the viewer can render a real per-edit diff
                # (not just a line count). Capped so a huge Write can't bloat the log.
                extra["old"] = old[:_CAP_EDIT]
                extra["new"] = new[:_CAP_EDIT]
        if name in _IMAGE_TOOLS:
            img = next((str(inp[k]) for k in ("image", "path", "file_path") if inp.get(k)), "")
            if img:
                extra["image"] = img
        self._write("tool", tool=name, n=self.counts[name], hint=hint[:200], **extra)

    def result(self, block) -> None:
        """A tool RESULT: what the call ANSWERED, plus any image it revealed.

        Two separate records, deliberately. The `result` record carries the output text — the
        measured numbers, the error, the render's report — which is the half of every tool call
        the trace used to discard. The `images` record stays as it was, because a render only
        names its output here (never in the input), and image paths need their own handling to
        be copied before they vanish.
        """
        try:
            content = getattr(block, "content", None)
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = " ".join(str(c.get("text", "")) if isinstance(c, dict) else str(c)
                                for c in content)
            else:
                text = str(content or "")

            tool_id = getattr(block, "tool_use_id", "") or ""
            name, started = self._calls.pop(tool_id, ("", None))
            if _CAP_OUTPUT:
                self._write("result", id=tool_id, tool=name,
                            secs=round(time.time() - started, 2) if started else None,
                            is_error=bool(getattr(block, "is_error", False)) or None,
                            chars=len(text) or None, output=text[:_CAP_OUTPUT])

            imgs = re.findall(r"[\w./~-]+\.(?:png|jpg|jpeg)", text)
            if imgs:
                # de-dup, keep order, cap so a chatty result can't flood the timeline
                seen, keep = set(), []
                for i in imgs:
                    if i not in seen:
                        seen.add(i)
                        keep.append(i)
                # Copy each image into the trace's own img/ dir NOW, while it still exists — many
                # renders live in /tmp and are gone by report time. `saved` are durable, trace-relative.
                saved = [s for s in (self._save_image(p) for p in keep[:6]) if s]
                self._write("images", images=keep[:6], saved=saved, n_images=len(seen))
        except Exception:  # noqa: BLE001
            pass

    def _locate(self, src: str) -> Path | None:
        """Resolve an image path scraped from tool output to a file on disk, or None.

        Results name images both ways. Absolute paths just work; BARE names (`conf_00000.png`,
        `frame_00000.jpg`) were previously checked with `Path(name).is_file()`, which resolves
        against the trace process's CWD — never where the file actually is. Every relative name
        therefore failed silently: one of seven image events in a real run saved anything. Try
        the roots this stage genuinely works out of before giving up.
        """
        p = Path(src).expanduser()
        if p.is_absolute():
            return p if p.is_file() else None
        if p.is_file():  # genuinely relative to CWD
            return p
        roots: list[Path] = [self.room] if self.room else []
        for var in ("LR_SURFACE_REF", "LR_SCAN_DIR", "LR_ROOM", "LR_AUTHORING", "LR_SCRATCH"):
            value = os.environ.get(var)
            if value:
                roots.append(Path(value))
        for root in roots:
            hit = root / p
            if hit.is_file():
                return hit
        return None

    def _save_image(self, src: str) -> str | None:
        """Copy (downscaled) an image the agent produced into <traces>/img/, return its path
        relative to the trace file so the record survives the original being deleted."""
        if not self.ok:
            return None
        try:
            p = self._locate(src)
            if p is None:
                return None
            imgdir = self.path.parent / "img"
            imgdir.mkdir(parents=True, exist_ok=True)
            self.seq  # noqa: B018  (ordering only)
            dst = imgdir / f"{self.pass_name}_{self.seq:04d}_{p.name}"
            try:
                from PIL import Image  # noqa: PLC0415
                im = Image.open(p).convert("RGB")
                if im.width > 1024:
                    im = im.resize((1024, round(im.height * 1024 / im.width)))
                dst = dst.with_suffix(".jpg")
                im.save(dst, format="JPEG", quality=84)
            except Exception:  # noqa: BLE001 — PIL missing/unreadable: fall back to a raw copy
                import shutil
                shutil.copyfile(p, dst)
            return str(dst.relative_to(self.path.parent))
        except Exception:  # noqa: BLE001
            return None

    def end(self, calls: int = 0, cost_usd=None, summary: str = "") -> None:
        self._write("session_end", calls=calls, seconds=round(time.time() - self.t0, 1),
                    cost_usd=cost_usd, counts=dict(self.counts), summary=(summary or "")[:3000])
        self.close()
        self.render_report()

    def render_report(self) -> Path | None:
        """Rebuild the run's HTML report — the readable form of everything recorded above.

        Rendered HERE, at the end of every pass, rather than only at publish: the report is most
        wanted for a run that DIED, and a run that dies never reaches publish. Each pass rewrites
        it, and the report merges every pass's jsonl, so it grows as the run proceeds.

        `LR_TRACE_HTML=0` disables it. Never raises — the same rule as the rest of this file: a
        trace must not take down the run it is describing.
        """
        if (os.environ.get("LR_TRACE_HTML", "1") or "1").strip().lower() in ("0", "false", "no"):
            return None
        try:
            from lrauthor.agent import trace_report

            # <scene>/scene_init/obj_stage/traces/<file> -> <scene>. Verified by structure rather
            # than by counting: a traces dir somewhere else must not write a report into a
            # directory that merely sits four levels up.
            scene = self.path.parents[3]
            if not (scene / "scene_init").is_dir():
                return None
            # Alongside the viewer in room_preview/, so one folder holds everything there is to
            # LOOK at. Created here rather than relied upon: publish builds room_preview/, and the
            # runs that most need a report are the ones that never got there.
            out = scene / "realism_authoring" / "room_preview" / "trace.html"
            out.parent.mkdir(parents=True, exist_ok=True)
            trace_report.build(scene, out)
            return out
        except Exception:  # noqa: BLE001 — a report is a convenience, never a failure mode
            return None

    def close(self) -> None:
        """Release the handles. Safe to call twice; a trace must never raise into a run."""
        for attr in ("_fh", "_raw_fh"):
            fh = getattr(self, attr, None)
            if fh is not None:
                try:
                    fh.close()
                except Exception:  # noqa: BLE001
                    pass
                setattr(self, attr, None)
        self.ok = False
