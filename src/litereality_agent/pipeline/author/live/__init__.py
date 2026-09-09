"""Watch an authoring run and stream it to a browser: the room as it is built, beside the trace.

`publish` produces the artifact you keep. This produces the one you *watch* — it points at a run
that is still going, recompiles `Room.py` whenever the agent saves it, and hands the browser both
the fresh geometry and the agent's own event log. Nothing here is on the pipeline's critical path:
the server only reads the run tree and writes into `<authoring_root>/.live/`, so it can be started,
killed, and restarted at any point in a run without disturbing it.

It lives under `pipeline/` because finding those things IS run-tree layout knowledge — where the
authored room sits, where each pass writes its trace — which `room_ops` deliberately does not have
(see the note in `room_ops/walk`, which walks a FINISHED room and therefore lives there).
The compile itself is a room-ops capability and is called as one, so the dependency points inward.

    python -m litereality_agent.pipeline.author.live <scan> [--port 8770]
    litereality live <scan>

Rebuilds land in two phases, because the two halves of a room have very different costs. Assembling
`Room.py` takes seconds; baking the SHELL's node-graph materials takes the better part of a minute
and is dominated by loading the .blend, so turning the bake resolution down barely helps. Waiting
for it would make every save feel slow, but skipping it is worse: a procedural wall/floor material
has no glTF representation and exports with neither a texture nor a base colour, so an unbaked room
renders as flat grey and the materials the agent just chose are invisible.

So the page gets the geometry the moment it compiles, and the baked room replaces it in place a
little later — two swaps, no reload, and the camera never moves. A bake whose geometry has already
been superseded by a newer save is dropped rather than published stale.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from litereality_agent import console
from litereality_agent.pipeline.author.live import page
from litereality_agent.pipeline.context import RunContext
from litereality_agent.room_ops.serve import bind

# Every pass writes its own trace file, and a run has several. Globbed rather than listed so a new
# pass shows up in the feed without editing this module.
TRACE_GLOBS = ("trace.jsonl", "authoring_trace*.jsonl")

# Only what a tracer actually saves beside a run. Anything else 404s rather than being guessed at.
MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif",
}


def trace_dirs(context: RunContext) -> list[Path]:
    """Where this run's tracers write, most interesting first."""
    return [context.object_root / "traces", context.scene_dir / "traces"]


@dataclass
class _Trace:
    """Merged view of every trace file in a run, tailed by byte offset.

    Offsets matter more than they look: an authoring trace grows to megabytes (one event carries the
    whole system prompt), and re-reading every file on every poll would burn more time than the
    poll interval. We keep what we have parsed and only read the tail each pass.
    """

    dirs: list[Path]
    events: list[dict] = field(default_factory=list)
    # Wall clock of the newest event ever seen — the page's evidence that the agent is still alive.
    # Tracked here rather than read off `events[-1]` because the merge only sorts each batch within
    # itself, so the tail of the list is not reliably the latest thing that happened.
    last_t: float = 0.0
    _offsets: dict[Path, int] = field(default_factory=dict)

    def refresh(self) -> None:
        fresh = []
        for d in self.dirs:
            if not d.is_dir():
                continue
            for pattern in TRACE_GLOBS:
                for path in sorted(d.glob(pattern)):
                    if path.name.endswith(".raw.jsonl"):
                        continue  # the normalised sibling says the same thing, smaller
                    fresh += self._tail(path)
        if fresh:
            # Passes run concurrently and each file is only ordered within itself, so the merged
            # feed is sorted by wall clock. Events already delivered keep their place.
            self.events += sorted(fresh, key=lambda e: e.get("t") or 0.0)
            self.last_t = max([self.last_t] + [e.get("t") or 0.0 for e in fresh])

    def _tail(self, path: Path) -> list[dict]:
        out: list[dict] = []
        try:
            size = path.stat().st_size
            seen = self._offsets.get(path, 0)
            if size <= seen:
                # Truncated or rewritten (a re-run of the same pass) — start over rather than
                # slicing into the middle of a line.
                if size < seen:
                    self._offsets[path] = 0
                    seen = 0
                else:
                    return out
            with path.open("rb") as fh:
                fh.seek(seen)
                blob = fh.read()
            # Only consume through the last complete line; a tracer may be mid-write.
            cut = blob.rfind(b"\n")
            if cut < 0:
                return out
            self._offsets[path] = seen + cut + 1
            for line in blob[:cut].splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
        except OSError:
            pass
        return out

    def since(self, cursor: int) -> list[dict]:
        return self.events[max(0, cursor):]


class LiveRoom:
    """Compiles on change and holds what the page needs to ask for."""

    def __init__(
        self,
        context: RunContext,
        poll: float = 1.0,
        bake: bool = True,
        bake_resolution: int = 1024,
    ) -> None:
        self.context = context
        self.poll = poll
        self.bake = bake
        self.bake_resolution = bake_resolution
        self.room = context.authored_room
        self.source = self.room / "Room.py"
        self.serve_dir = context.authoring_root / ".live"
        self.serve_dir.mkdir(parents=True, exist_ok=True)
        self.glb = self.serve_dir / "room.glb"

        self.build = 0
        self.status = "idle"
        self.phase = "none"
        self.error = ""
        # Set by `start`. The banner goes out before the stage runs, when the page is still empty;
        # the first build is when there is finally something to look at, and by then the banner is
        # hundreds of lines of stage output further up. So the url is said once more, there.
        self.url = ""
        # Set once the run that this viewer was started alongside has ended. The viewer deliberately
        # keeps serving afterwards — the finished room is the thing you most want to look at — so the
        # page needs to say so, otherwise a run that ended is indistinguishable from one gone quiet.
        self.finished = ""
        self.compile_s = 0.0
        self.bake_s = 0.0
        self.baking = False
        self.trace = _Trace(trace_dirs(context))
        # Bumped on every source change. A bake carries the generation it started from and is
        # dropped if a newer save has landed meanwhile, so the page never gets materials baked
        # onto geometry that no longer exists.
        self._gen = 0
        self._stamp: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    # -- publishing --------------------------------------------------------------------
    def _publish(self, built: Path, phase: str) -> None:
        """Swap `built` in as the served room. Renamed rather than copied in place so a poll
        landing mid-write never sees half a glb."""
        tmp = self.serve_dir / "room.glb.tmp"
        shutil.copy2(built, tmp)
        tmp.replace(self.glb)
        with self._lock:
            self.build += 1
            first = self.build == 1
            self.phase = phase
        if first and self.url:
            cyan, bold, off = (console.colour(c) for c in ("cyan", "bold", "off"))
            print(f"\n   {bold}{cyan}▶ room is live at {self.url}{off}   "
                  f"{console.colour('dim')}first build ready{off}\n", flush=True)

    # -- compilation -------------------------------------------------------------------
    def _syntax_error(self) -> str:
        """`"SyntaxError: … (line N)"` when Room.py cannot be parsed, else `""`.

        Compiling on top of the read is deliberately not done: this only has to reject a file the
        compiler would mis-handle, and parsing 100 KB costs a few milliseconds against a build of
        several seconds.
        """
        try:
            source = self.source.read_text(encoding="utf-8")
        except OSError as exc:
            return f"cannot read Room.py: {exc}"
        try:
            ast.parse(source, filename=str(self.source))
        except SyntaxError as exc:
            return f"SyntaxError: {exc.msg} (line {exc.lineno})"
        return ""

    def rebuild(self) -> bool:
        """Compile `Room.py` and publish the geometry. Returns True when the page has new geometry.

        The bake, when enabled, continues on its own thread and publishes a second time.
        """
        from litereality_agent.room_ops import api

        with self._lock:
            self.status, self.error = "building", ""
            gen = self._gen
        started = time.time()

        # Parse before compiling. `build_from_room` runs Room.py inside Blender, and a Room.py that
        # fails to parse there is only *logged* — the process still exits 0 and the previous
        # Room.glb is left on disk, so compiling a half-saved file returns the last good room and
        # the page reports a clean new build. Watching an agent is exactly when that lies worst:
        # you see "build N", the geometry never changed, and nothing says why. An agent saves
        # partial files often enough that this is the common case, not the exotic one.
        broken = self._syntax_error()
        if broken:
            with self._lock:
                self.status, self.error = "failed", broken
            return False

        try:
            built = api.compile_room(self.room, self.context.preview_dir, bake=False)
        except Exception as exc:  # noqa: BLE001 — a broken Room.py must not kill the server
            with self._lock:
                self.status, self.error = "failed", f"{type(exc).__name__}: {exc}"
            return False
        if not built or not Path(built).is_file():
            with self._lock:
                self.status, self.error = "failed", "compile produced no Room.glb"
            return False

        self._publish(Path(built), "geometry")
        with self._lock:
            self.compile_s = time.time() - started
            self.status = "idle"

        if self.bake:
            threading.Thread(target=self._bake, args=(gen,), daemon=True).start()
        return True

    def _bake(self, gen: int) -> None:
        """Flatten the SHELL's node materials into a copy of the room, then publish it.

        Baked into a sibling file, never the served one: `bake_room` re-exports to the path it is
        given, so pointing it at the live glb would blank the page for the length of the bake.
        """
        from litereality_agent.room_ops import api

        with self._lock:
            self.baking = True
        started = time.time()
        out = self.serve_dir / "baked.glb"
        try:
            rc = api.bake_room(
                self.context.preview_dir / "Room.blend", out, resolution=self.bake_resolution
            )
            superseded = gen != self._gen
            if rc == 0 and out.is_file() and not superseded:
                self._publish(out, "baked")
                with self._lock:
                    self.bake_s = time.time() - started
        except Exception:  # noqa: BLE001 — a failed bake leaves the geometry build standing
            pass
        finally:
            with self._lock:
                self.baking = False

    # -- watching ----------------------------------------------------------------------
    def _changed(self) -> bool:
        try:
            stamp = self.source.stat().st_mtime
        except OSError:
            return False
        if self._stamp is None or stamp != self._stamp:
            self._stamp = stamp
            with self._lock:
                self._gen += 1
            return True
        return False

    def watch(self) -> None:
        while not self._stop.is_set():
            try:
                self.trace.refresh()
                if self._changed():
                    self.rebuild()
            except Exception:  # noqa: BLE001 — the watcher outlives any single bad iteration
                pass
            self._stop.wait(self.poll)

    def stop(self) -> None:
        self._stop.set()

    def finish(self, note: str) -> None:
        """Record that the run this viewer accompanies has ended. Does not stop the watcher: a
        `Room.py` edited by hand after the run should still rebuild."""
        with self._lock:
            self.finished = note

    # -- what the page asks for --------------------------------------------------------
    def image(self, rel: str) -> Path | None:
        """Resolve one of a trace event's `saved` paths to a file on disk, or None.

        `saved` entries are relative to the trace directory that recorded them (`img/author_0048_
        Wall0_stitched.jpg`). Resolution is containment-checked rather than merely prefix-checked:
        `resolve()` collapses `..` and follows symlinks first, so a crafted path cannot walk out of
        the run and turn this into an arbitrary-file endpoint. That matters because `--host` can put
        this server on a network interface.
        """
        if not rel:
            return None
        for base in self.trace.dirs + [self.context.scene_dir]:
            try:
                root = base.resolve(strict=True)
            except OSError:
                continue
            candidate = (root / rel).resolve()
            if candidate.is_file() and candidate.is_relative_to(root):
                return candidate
        return None

    def state(self, cursor: int) -> dict:
        with self._lock:
            build, status, error, secs = self.build, self.status, self.error, self.compile_s
            phase, baking, bake_s = self.phase, self.baking, self.bake_s
            finished = self.finished
        events = self.trace.since(cursor)
        # Seconds since the agent last did anything, or None before it has done anything at all —
        # a distinction the page needs, since "no events yet" and "gone quiet" look the same in a
        # number but mean opposite things while a run is starting up.
        last_t = self.trace.last_t
        return {
            "build": build,
            "status": status,
            "phase": phase,
            "finished": finished,
            "idle_s": round(max(0.0, time.time() - last_t), 1) if last_t else None,
            "baking": baking,
            "bake_s": round(bake_s, 2),
            "error": error,
            "compile_s": round(secs, 2),
            "mb": round(self.glb.stat().st_size / 1e6, 1) if self.glb.is_file() else 0,
            "events": events,
            "cursor": cursor + len(events),
            "total": len(self.trace.events),
        }


def _handler(room: LiveRoom, label: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:  # noqa: D102 — one line per poll is pure noise
            pass

        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
            route = urlparse(self.path)
            path = route.path.rstrip("/") or "/"
            if path == "/":
                return self._send(page.render(label).encode(), "text/html; charset=utf-8")
            if path == "/state":
                since = int((parse_qs(route.query).get("since") or ["0"])[0] or 0)
                return self._send(json.dumps(room.state(since)).encode(), "application/json")
            if path == "/room.glb":
                if not room.glb.is_file():
                    self.send_error(404, "no build yet")
                    return
                return self._send(room.glb.read_bytes(), "model/gltf-binary")
            if path == "/img":
                rel = (parse_qs(route.query).get("p") or [""])[0]
                found = room.image(rel)
                if not found:
                    self.send_error(404, "no such trace image")
                    return
                kind = MIME.get(found.suffix.lower(), "application/octet-stream")
                # A trace image never changes once written, so let the browser keep it — the feed
                # re-renders on every poll and would otherwise refetch every thumbnail.
                self.send_response(200)
                self.send_header("Content-Type", kind)
                body = found.read_bytes()
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                return self.wfile.write(body)
            self.send_error(404)

    return Handler


def start(
    context: RunContext,
    *,
    port: int = 8770,
    host: str = "127.0.0.1",
    poll: float = 1.0,
    bake: bool = True,
    bake_resolution: int = 1024,
) -> tuple[LiveRoom, ThreadingHTTPServer, str]:
    """Start the watcher and the HTTP server on background threads and return straight away.

    Split out of `serve` so a stage can run *alongside* the viewer in one process (`--live`). That
    sharing is the point: both halves take the same `RunContext`, so the room being edited and the
    traces being tailed cannot end up under different roots — which is exactly what happens when a
    viewer and a stage are launched separately with mismatched `--output-root`.

    `port` is where to start looking, not a requirement — see `room_ops.serve.bind`. Read it off the
    returned url (or `server.server_port`); it is not necessarily the one asked for.

    The caller owns shutdown: `room.stop()` then `server.shutdown()`.
    """
    room = LiveRoom(context, poll=poll, bake=bake, bake_resolution=bake_resolution)
    # Bind before the watcher starts, so exhausting the port range leaves nothing running behind.
    server = bind(_handler(room, context.scan), host, port)
    room.url = f"http://{host}:{server.server_port}/"  # so the first build can say it again
    room.trace.refresh()
    threading.Thread(target=room.watch, daemon=True).start()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return room, server, room.url


def require_room(context: RunContext) -> None:
    """The viewer has nothing to show without a compiled-able room.

    For standalone `litereality live` only. `--live` alongside a stage must NOT call this: there
    the room is about to be written by the very stage the viewer is accompanying, and waiting for
    it is the correct behaviour rather than an error.
    """
    if not (context.authored_room / "Room.py").is_file():
        raise SystemExit(
            f"no authored room for {context.scan} — expected {context.authored_room / 'Room.py'}"
        )


def describe(context: RunContext, url: str, bake: bool) -> None:
    """The banner both entry points print, so `--live` says the same things `live` does.

    Set apart with a rule and given the url a line of its own, because with `--live` this prints
    BEFORE the stage does and the stage prints hundreds of lines. One url buried in the third
    column of a wall of `cov=97% -> plain #9e9486` is a url nobody finds again.
    """
    room_py = context.authored_room / "Room.py"
    cyan, bold, dim, off = (console.colour(c) for c in ("cyan", "bold", "dim", "off"))
    print(console.rule("live viewer"))
    print(f"\n   {bold}{cyan}{url}{off}\n")
    # An empty page is alarming when you do not know it is expected. Say so before it happens.
    pending = f"  {dim}(not written yet — the page waits for it){off}" if not room_py.is_file() else ""
    print(f"   {dim}room  {off} {console.short(room_py)}{pending}")
    traces = ", ".join(console.short(d) for d in trace_dirs(context) if d.is_dir()) or "(none yet)"
    print(f"   {dim}traces{off} {traces}")
    print(
        f"   {dim}"
        + ("geometry first, materials baked right after" if bake else "geometry only (--no-bake)")
        + f" · rebuilds on every save · ctrl-c to stop{off}"
    )
    print(f"{dim}{'─' * 72}{off}")


def serve(
    target: str,
    *,
    port: int = 8770,
    host: str = "127.0.0.1",
    poll: float = 1.0,
    output_root: str | None = None,
    bake: bool = True,
    bake_resolution: int = 1024,
) -> int:
    """Run the live viewer for `target` until interrupted."""
    context = RunContext.resolve(target, output_root=output_root)
    require_room(context)
    room, server, url = start(
        context, port=port, host=host, poll=poll, bake=bake, bake_resolution=bake_resolution,
    )
    describe(context, url, bake)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        room.stop()
        server.shutdown()
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="live authoring viewer")
    ap.add_argument("target", metavar="SCAN_OR_SCENE")
    ap.add_argument(
        "--port", type=int, default=8770,
        help="where to start looking for a free port; taken ones are skipped",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between Room.py checks")
    ap.add_argument("--output-root")
    ap.add_argument(
        "--no-bake", dest="bake", action="store_false",
        help="geometry only — faster, but procedural SHELL materials render flat",
    )
    ap.add_argument("--bake-resolution", type=int, default=1024)
    args = ap.parse_args(argv)
    return serve(
        args.target, port=args.port, host=args.host, poll=args.poll,
        output_root=args.output_root, bake=args.bake, bake_resolution=args.bake_resolution,
    )


if __name__ == "__main__":
    raise SystemExit(main())
