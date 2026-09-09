"""The live viewer's non-Blender halves: trace tailing, state, and the HTTP surface.

Compilation is deliberately not exercised — it spawns Blender, which the unit suite never does.
`LiveRoom.rebuild` is stubbed where a build is needed so the parts that run on every poll are
still covered.
"""

from __future__ import annotations

import errno
import json
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

import pytest

from litereality_agent.pipeline.author import live
from litereality_agent.pipeline.context import RunContext
from litereality_agent.room_ops import serve


def _event(seq: int, t: float, **extra) -> str:
    return json.dumps({"seq": seq, "t": t, "dt": t - 100.0, **extra})


@pytest.fixture
def run_tree(tmp_path: Path) -> Path:
    scene = tmp_path / "run" / "Scan-1"
    (scene / "realism_authoring" / "room").mkdir(parents=True)
    (scene / "realism_authoring" / "room" / "Room.py").write_text("# room\n")
    (scene / "scene_init" / "obj_stage" / "traces").mkdir(parents=True)
    return scene


def _context(run_tree: Path) -> RunContext:
    return RunContext(
        scan="Scan-1",
        capture_dir=run_tree,
        scene_dir=run_tree,
        output_root=run_tree.parent,
    )


def test_trace_merges_files_in_wall_clock_order(run_tree: Path) -> None:
    traces = run_tree / "scene_init" / "obj_stage" / "traces"
    (traces / "trace.jsonl").write_text(_event(1, 100.0, kind="stage") + "\n")
    (traces / "authoring_trace.author.jsonl").write_text(
        _event(1, 101.0, kind="think", text="a") + "\n" + _event(2, 103.0, kind="tool") + "\n"
    )

    trace = live._Trace(live.trace_dirs(_context(run_tree)))
    trace.refresh()

    assert [e["t"] for e in trace.events] == [100.0, 101.0, 103.0]


def test_trace_tails_only_new_lines(run_tree: Path) -> None:
    path = run_tree / "scene_init" / "obj_stage" / "traces" / "trace.jsonl"
    path.write_text(_event(1, 100.0, kind="stage") + "\n")

    trace = live._Trace(live.trace_dirs(_context(run_tree)))
    trace.refresh()
    trace.refresh()  # nothing new — must not duplicate
    assert len(trace.events) == 1

    with path.open("a") as fh:
        fh.write(_event(2, 102.0, kind="tool") + "\n")
    trace.refresh()
    assert len(trace.events) == 2


def test_trace_ignores_a_partial_final_line(run_tree: Path) -> None:
    """A tracer mid-write leaves a line with no newline; it must be picked up on the next pass,
    not dropped and not parsed as truncated JSON."""
    path = run_tree / "scene_init" / "obj_stage" / "traces" / "trace.jsonl"
    path.write_text(_event(1, 100.0, kind="stage") + "\n" + '{"seq": 2, "t": 101.0, "kin')

    trace = live._Trace(live.trace_dirs(_context(run_tree)))
    trace.refresh()
    assert len(trace.events) == 1

    with path.open("a") as fh:
        fh.write('d": "tool"}\n')
    trace.refresh()
    assert [e["kind"] for e in trace.events] == ["stage", "tool"]


def test_trace_skips_raw_sibling(run_tree: Path) -> None:
    traces = run_tree / "scene_init" / "obj_stage" / "traces"
    (traces / "authoring_trace.author.jsonl").write_text(_event(1, 100.0, kind="tool") + "\n")
    (traces / "authoring_trace.author.raw.jsonl").write_text(_event(1, 100.0, kind="tool") + "\n")

    trace = live._Trace(live.trace_dirs(_context(run_tree)))
    trace.refresh()
    assert len(trace.events) == 1


def test_state_cursor_only_returns_new_events(run_tree: Path) -> None:
    path = run_tree / "scene_init" / "obj_stage" / "traces" / "trace.jsonl"
    path.write_text(_event(1, 100.0, kind="stage") + "\n")

    room = live.LiveRoom(_context(run_tree))
    room.trace.refresh()

    first = room.state(0)
    assert len(first["events"]) == 1 and first["cursor"] == 1
    assert room.state(first["cursor"])["events"] == []


def test_idle_s_reports_silence_since_the_last_event(run_tree: Path) -> None:
    """The page's evidence that the agent is still working. Absent this, an idle-between-builds run
    and a run that ended an hour ago report exactly the same state."""
    path = run_tree / "scene_init" / "obj_stage" / "traces" / "trace.jsonl"
    room = live.LiveRoom(_context(run_tree))

    room.trace.refresh()
    assert room.state(0)["idle_s"] is None, "nothing has happened yet — not the same as gone quiet"

    path.write_text(_event(1, time.time() - 30.0, kind="tool") + "\n")
    room.trace.refresh()
    assert 29.0 <= room.state(0)["idle_s"] <= 35.0

    with path.open("a") as fh:              # tracers append; they never rewrite an earlier line
        fh.write(_event(2, time.time(), kind="think") + "\n")
    room.trace.refresh()
    assert room.state(0)["idle_s"] < 5.0, "a fresh event must reset the clock"


def test_idle_s_tracks_the_newest_event_not_the_last_appended(run_tree: Path) -> None:
    """Each batch is sorted only within itself, so a slow pass can deliver stale lines after a fast
    one. Reading `events[-1]` would then age the run backwards and stop the dot on a live agent."""
    traces = run_tree / "scene_init" / "obj_stage" / "traces"
    now = time.time()
    room = live.LiveRoom(_context(run_tree))

    (traces / "trace.jsonl").write_text(_event(1, now, kind="stage") + "\n")
    room.trace.refresh()

    # A second pass's trace file appears, carrying events from ten minutes ago.
    (traces / "authoring_trace.author.jsonl").write_text(_event(1, now - 600.0, kind="tool") + "\n")
    room.trace.refresh()

    assert room.trace.events[-1]["t"] == now - 600.0   # the naive read this guards against
    assert room.state(0)["idle_s"] < 5.0


def test_changed_fires_once_per_save(run_tree: Path) -> None:
    room = live.LiveRoom(_context(run_tree))
    assert room._changed() is True   # first look always builds
    assert room._changed() is False

    source = run_tree / "realism_authoring" / "room" / "Room.py"
    source.write_text("# edited\n")
    import os

    os.utime(source, (1_000_000, 1_000_000))
    assert room._changed() is True


def test_rebuild_records_failure_without_raising(run_tree: Path, monkeypatch) -> None:
    """A Room.py the agent has half-saved must leave the server up and say so on the page."""
    room = live.LiveRoom(_context(run_tree))

    def boom(*_a, **_k):
        raise RuntimeError("SyntaxError: unexpected EOF")

    monkeypatch.setattr("litereality_agent.room_ops.api.compile_room", boom)
    assert room.rebuild() is False
    assert room.state(0)["status"] == "failed"
    assert "unexpected EOF" in room.state(0)["error"]
    assert room.build == 0


def test_unparseable_room_is_rejected_before_compiling(run_tree: Path, monkeypatch) -> None:
    """`build_from_room` runs Room.py inside Blender, logs a SyntaxError, and still exits 0 leaving
    the previous Room.glb in place — so compiling a half-saved file republishes the LAST good room
    as a clean new build. The parse has to happen before the compiler is ever reached."""
    room = live.LiveRoom(_context(run_tree))
    (run_tree / "realism_authoring" / "room" / "Room.py").write_text("def broken(:\n")

    def never(*_a, **_k):
        raise AssertionError("compile_room must not run on a Room.py that does not parse")

    monkeypatch.setattr("litereality_agent.room_ops.api.compile_room", never)

    assert room.rebuild() is False
    state = room.state(0)
    assert state["status"] == "failed"
    assert state["error"].startswith("SyntaxError:") and "line 1" in state["error"]
    assert room.build == 0          # nothing published, so the page cannot claim a fresh build


def test_parseable_room_still_compiles(run_tree: Path, monkeypatch) -> None:
    room = live.LiveRoom(_context(run_tree))
    built = run_tree / "realism_authoring" / "room_preview" / "Room.glb"
    built.parent.mkdir(parents=True, exist_ok=True)
    built.write_bytes(b"geometry")
    monkeypatch.setattr("litereality_agent.room_ops.api.compile_room", lambda *a, **k: built)

    assert room.rebuild() is True
    assert room.state(0)["error"] == ""


def test_recovers_after_the_syntax_error_is_fixed(run_tree: Path, monkeypatch) -> None:
    """The failure is per-build state, not sticky: the next good save must clear it."""
    room = live.LiveRoom(_context(run_tree))
    source = run_tree / "realism_authoring" / "room" / "Room.py"
    built = run_tree / "realism_authoring" / "room_preview" / "Room.glb"
    built.parent.mkdir(parents=True, exist_ok=True)
    built.write_bytes(b"geometry")
    monkeypatch.setattr("litereality_agent.room_ops.api.compile_room", lambda *a, **k: built)

    source.write_text("def broken(:\n")
    assert room.rebuild() is False
    assert room.state(0)["status"] == "failed"

    source.write_text("# fixed\n")
    assert room.rebuild() is True
    assert room.state(0)["status"] == "idle" and room.state(0)["error"] == ""


def test_geometry_publishes_before_the_bake(run_tree: Path, monkeypatch) -> None:
    """The point of the two-phase build: the page must get geometry without waiting on the bake."""
    room = live.LiveRoom(_context(run_tree), bake=True)
    built = run_tree / "realism_authoring" / "room_preview" / "Room.glb"
    built.parent.mkdir(parents=True, exist_ok=True)
    built.write_bytes(b"geometry")

    released = threading.Event()

    def slow_bake(_blend, out, **_k):
        released.wait(5)
        Path(out).write_bytes(b"baked")
        return 0

    monkeypatch.setattr("litereality_agent.room_ops.api.compile_room", lambda *a, **k: built)
    monkeypatch.setattr("litereality_agent.room_ops.api.bake_room", slow_bake)

    assert room.rebuild() is True
    assert room.state(0)["phase"] == "geometry"      # published while the bake is still running
    assert room.glb.read_bytes() == b"geometry"

    released.set()
    for _ in range(100):
        if room.state(0)["phase"] == "baked":
            break
        time.sleep(0.05)
    assert room.state(0)["phase"] == "baked"
    assert room.glb.read_bytes() == b"baked"
    assert room.build == 2                            # one swap for each phase


def test_superseded_bake_is_dropped(run_tree: Path, monkeypatch) -> None:
    """A bake that finishes after a newer save must not paint stale materials over new geometry."""
    room = live.LiveRoom(_context(run_tree), bake=True)
    built = run_tree / "realism_authoring" / "room_preview" / "Room.glb"
    built.parent.mkdir(parents=True, exist_ok=True)
    built.write_bytes(b"geometry")

    def bake(_blend, out, **_k):
        room._gen += 1          # a newer save lands while this bake is running
        Path(out).write_bytes(b"stale-bake")
        return 0

    monkeypatch.setattr("litereality_agent.room_ops.api.compile_room", lambda *a, **k: built)
    monkeypatch.setattr("litereality_agent.room_ops.api.bake_room", bake)

    room.rebuild()
    for _ in range(100):
        if not room.state(0)["baking"]:
            break
        time.sleep(0.05)
    assert room.glb.read_bytes() == b"geometry"
    assert room.state(0)["phase"] == "geometry"


def test_no_bake_stays_single_phase(run_tree: Path, monkeypatch) -> None:
    room = live.LiveRoom(_context(run_tree), bake=False)
    built = run_tree / "realism_authoring" / "room_preview" / "Room.glb"
    built.parent.mkdir(parents=True, exist_ok=True)
    built.write_bytes(b"geometry")

    def refuse(*_a, **_k):
        raise AssertionError("bake_room must not run with bake=False")

    monkeypatch.setattr("litereality_agent.room_ops.api.compile_room", lambda *a, **k: built)
    monkeypatch.setattr("litereality_agent.room_ops.api.bake_room", refuse)

    assert room.rebuild() is True
    assert room.build == 1 and room.state(0)["phase"] == "geometry"


def test_trace_image_resolves_relative_to_its_trace_dir(run_tree: Path) -> None:
    img = run_tree / "scene_init" / "obj_stage" / "traces" / "img"
    img.mkdir(parents=True)
    (img / "author_0048_Wall0.jpg").write_bytes(b"jpegbytes")

    room = live.LiveRoom(_context(run_tree))
    found = room.image("img/author_0048_Wall0.jpg")
    assert found is not None and found.read_bytes() == b"jpegbytes"
    assert room.image("img/nope.jpg") is None
    assert room.image("") is None


def test_trace_image_refuses_to_escape_the_run(run_tree: Path, tmp_path: Path) -> None:
    """`--host` can put this server on a real interface, so `p` is untrusted input."""
    secret = tmp_path / "secret.png"
    secret.write_bytes(b"private")
    (run_tree / "scene_init" / "obj_stage" / "traces" / "img").mkdir(parents=True)

    room = live.LiveRoom(_context(run_tree))
    for attack in (
        "../../../../../../etc/passwd",
        "img/../../../../secret.png",
        f"../../../{secret.name}",
        str(secret),                      # absolute path outside the run
    ):
        assert room.image(attack) is None, attack


def test_http_serves_a_trace_image(run_tree: Path) -> None:
    img = run_tree / "scene_init" / "obj_stage" / "traces" / "img"
    img.mkdir(parents=True)
    (img / "shot.png").write_bytes(b"\x89PNG-ish")
    room = live.LiveRoom(_context(run_tree))

    server = ThreadingHTTPServer(("127.0.0.1", 0), live._handler(room, "Scan-1"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        got = urlopen(base + "/img?p=img/shot.png")
        assert got.read() == b"\x89PNG-ish"
        assert got.headers["Content-Type"] == "image/png"
        with pytest.raises(Exception):
            urlopen(base + "/img?p=../../../../etc/passwd")
    finally:
        server.shutdown()
        server.server_close()


def test_http_surface(run_tree: Path) -> None:
    room = live.LiveRoom(_context(run_tree))
    room.glb.write_bytes(b"glTF-not-really")
    room.build = 1

    server = ThreadingHTTPServer(("127.0.0.1", 0), live._handler(room, "Scan-1"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        home = urlopen(base + "/").read().decode()
        assert "Agent trace" in home and "Scan-1" in home

        state = json.loads(urlopen(base + "/state?since=0").read())
        assert state["build"] == 1

        assert urlopen(base + "/room.glb?b=1").read() == b"glTF-not-really"
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- #
# `--live`: the viewer running alongside the stage that is editing the room
# --------------------------------------------------------------------------- #
def test_finish_marks_the_run_done_without_stopping_the_watcher(run_tree: Path) -> None:
    """The viewer outlives the run on purpose — a finished room is the thing you most want to look
    at, and a Room.py edited by hand afterwards must still rebuild."""
    room = live.LiveRoom(_context(run_tree))
    assert room.state(0)["finished"] == ""

    room.finish("author finished")

    assert room.state(0)["finished"] == "author finished"
    assert not room._stop.is_set(), "finishing the run must not stop the file watcher"


def test_start_serves_immediately_and_shuts_down_cleanly(run_tree: Path) -> None:
    """`start` is the non-blocking half `--live` needs: it must serve before the caller does more."""
    room, server, url = live.start(_context(run_tree), port=0, bake=False)
    port = server.server_address[1]
    try:
        assert url == f"http://127.0.0.1:{port}/"  # the bound port, so the banner is reachable
        state = json.loads(urlopen(f"http://127.0.0.1:{port}/state?since=0").read())
        assert state["finished"] == ""
    finally:
        room.stop()
        server.shutdown()
        server.server_close()


def test_start_skips_a_taken_port(run_tree: Path) -> None:
    """A viewer left running from an earlier run must not stop the next one from starting — the
    port is incidental to the room, so it walks up instead of dying on EADDRINUSE."""
    squatter = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    taken = squatter.server_address[1]
    try:
        room, server, url = live.start(_context(run_tree), port=taken, bake=False)
        try:
            assert server.server_port != taken
            assert server.server_port in range(taken, taken + serve.PORT_TRIES)
            assert url == f"http://127.0.0.1:{server.server_port}/"
            urlopen(f"{url}state?since=0").read()  # the one that answers is the one announced
        finally:
            room.stop()
            server.shutdown()
            server.server_close()
    finally:
        squatter.server_close()


def test_start_gives_up_when_the_whole_range_is_taken(run_tree: Path, monkeypatch) -> None:
    """Walking up is bounded — an exhausted range is a real failure, not an endless search."""
    def refuse(*_args, **_kwargs):
        raise OSError(errno.EADDRINUSE, "Address already in use")

    monkeypatch.setattr(serve, "ThreadingHTTPServer", refuse)
    with pytest.raises(SystemExit, match="no free port"):
        live.start(_context(run_tree), port=8770, bake=False)


def test_start_does_not_swallow_other_bind_errors(run_tree: Path, monkeypatch) -> None:
    """Only a taken port is worth retrying; a bad host is a mistake the caller needs to see."""
    def refuse(*_args, **_kwargs):
        raise OSError(errno.EADDRNOTAVAIL, "Can't assign requested address")

    monkeypatch.setattr(serve, "ThreadingHTTPServer", refuse)
    with pytest.raises(OSError) as caught:
        live.start(_context(run_tree), port=8770, bake=False)
    assert caught.value.errno == errno.EADDRNOTAVAIL


def test_live_flag_shares_one_run_context_with_the_stage(run_tree: Path, monkeypatch) -> None:
    """The regression this flag exists to prevent.

    Launching `live` and `stage` separately lets them resolve different `--output-root`s: the stage
    then writes its trace under one root while the viewer tails another, and the page shows a stale
    trace forever with nothing reporting an error. Sharing the context makes that unrepresentable.
    """
    from litereality_agent import cli

    context = _context(run_tree)
    seen = {}

    def fake_start(ctx, **kw):
        seen["context"] = ctx
        seen["bake"] = kw["bake"]
        room = live.LiveRoom(ctx)
        return room, _StubServer(), "http://127.0.0.1:8770/"

    monkeypatch.setattr(live, "start", fake_start)
    monkeypatch.setattr(cli, "_block_until_interrupt", lambda: None)

    args = _LiveArgs(live=True, live_port=8770, live_no_bake=True)
    with cli._live_viewer(args, context) as finished:
        finished("author finished")

    assert seen["context"] is context, "the viewer must watch the very context the stage runs in"
    assert seen["bake"] is False, "--live-no-bake must reach the viewer"


def test_live_starts_before_the_room_exists(run_tree: Path, monkeypatch, capsys) -> None:
    """The chicken-and-egg `--live` used to lose to.

    `author` CREATES the authored room — it copies the seed room in as its first act — so on a
    scene's first authoring run there is no `Room.py` when the viewer starts. Requiring one up
    front made `--live` usable only on the second run, which is the run you least need to watch.
    """
    from litereality_agent import cli

    context = _context(run_tree)
    shutil.rmtree(context.authored_room)  # a scene that has been seeded but never authored
    started = {}

    def fake_start(ctx, **kw):
        started["yes"] = True
        return live.LiveRoom(ctx), _StubServer(), "http://127.0.0.1:8770/"

    monkeypatch.setattr(live, "start", fake_start)
    monkeypatch.setattr(cli, "_block_until_interrupt", lambda: None)

    with cli._live_viewer(_LiveArgs(live=True, live_port=8770, live_no_bake=True), context) as done:
        done("author finished")

    assert started.get("yes"), "the viewer must start and wait, not abort the run"
    assert "not written yet" in capsys.readouterr().out, "an empty page needs explaining"


def test_standalone_live_still_refuses_a_scene_with_no_room(run_tree: Path) -> None:
    """Nothing in `litereality live` will ever create a room, so waiting there is a hang."""
    context = _context(run_tree)
    shutil.rmtree(context.authored_room)

    with pytest.raises(SystemExit, match="no authored room"):
        live.require_room(context)


def test_the_viewer_picks_the_room_up_when_the_stage_writes_it(run_tree: Path, monkeypatch) -> None:
    """Starting without a room is only useful if the watcher then notices one arriving."""
    context = _context(run_tree)
    source = context.authored_room / "Room.py"
    body = source.read_text(encoding="utf-8")
    shutil.rmtree(context.authored_room)

    room = live.LiveRoom(context, poll=0.01)
    builds = []
    monkeypatch.setattr(room, "rebuild", lambda: builds.append(True) or True)

    assert room._changed() is False, "a missing Room.py is not a change, and must not raise"

    context.authored_room.mkdir(parents=True, exist_ok=True)  # the stage's copytree lands
    source.write_text(body, encoding="utf-8")

    assert room._changed() is True, "the room appearing must register as a change"


def test_without_live_flag_nothing_is_served(run_tree: Path, monkeypatch) -> None:
    from litereality_agent import cli

    def refuse(*_a, **_k):
        raise AssertionError("no viewer may start without --live")

    monkeypatch.setattr(live, "start", refuse)
    with cli._live_viewer(_LiveArgs(live=False), _context(run_tree)) as finished:
        finished("author finished")  # must be a no-op, not a crash


class _StubServer:
    def shutdown(self) -> None:
        pass

    def server_close(self) -> None:
        pass


class _LiveArgs:
    def __init__(self, *, live: bool, live_port: int = 8770, live_no_bake: bool = False) -> None:
        self.live = live
        self.live_port = live_port
        self.live_no_bake = live_no_bake
