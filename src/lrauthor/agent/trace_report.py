"""trace_report.py — a self-contained HTML report of an authoring run: the trace timeline WITH
the rendered images embedded inline and the actual Room.py code changes shown as a diff.

The run's trace (authoring_trace.*.jsonl) records reasoning + tool calls + image PATHS + edit
line-deltas, but it does not embed the images or the code. This rebuilds that: it reads the
trace, embeds every referenced image that still exists on disk (as a downscaled data-URI), and
renders the net seed->final Room.py as a coloured unified diff.

    python -m lrauthor.agent.trace_report <scene_dir> [out.html]

<scene_dir> = run/<scan>/ (holds obj_stage/traces/ + scene_stage/{room_init,_oneshot}).
"""

from __future__ import annotations

import base64
import difflib
import html
import io
import json
import sys
from pathlib import Path

try:
    from PIL import Image
except Exception:  # noqa: BLE001
    Image = None

THUMB_W = 460  # embedded thumbnail width (px)


def data_uri(path: Path, max_w: int = THUMB_W) -> str | None:
    """Downscale to max_w and return a data: URI, or None if unreadable."""
    try:
        if Image is None:
            raw = path.read_bytes()
            mime = "jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "png"
            return f"data:image/{mime};base64," + base64.b64encode(raw).decode()
        im = Image.open(path).convert("RGB")
        if im.width > max_w:
            im = im.resize((max_w, round(im.height * max_w / im.width)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def load_trace(traces: Path) -> list[dict]:
    """Every pass's events, oldest first.

    Globbed rather than listed by name: passes were being missed because the list named only
    `author` and `materials`, so a `qc` or `refine` run rendered an empty report. `.raw.jsonl`
    is excluded — it is the verbatim sidecar, read separately by `load_raw`.
    """
    events: list[dict] = []
    files = [traces / "trace.jsonl"] + sorted(
        p for p in traces.glob("authoring_trace.*.jsonl") if not p.name.endswith(".raw.jsonl"))
    for f in files:
        if not f.is_file():
            continue
        for line in f.read_text(errors="replace").splitlines():
            try:
                r = json.loads(line)
                r["_src"] = f.name
                events.append(r)
            except Exception:  # noqa: BLE001
                pass
    return events


def load_raw(traces: Path) -> list[dict]:
    """The verbatim SDK sidecar, if it was written."""
    out: list[dict] = []
    for f in sorted(traces.glob("authoring_trace.*.raw.jsonl")):
        for line in f.read_text(errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                pass
    return out


def block(label: str, body: str, *, cls: str = "", open_: bool = False) -> str:
    """A collapsible chunk. The full command and the full output belong in the report, but
    inline they would bury the timeline — so they are one click away, not one file away."""
    if not body.strip():
        return ""
    o = " open" if open_ else ""
    return (f'<details class="blk {cls}"{o}><summary>{html.escape(label)}</summary>'
            f'<pre>{html.escape(body)}</pre></details>')


def diff_html(seed: Path, final: Path) -> str:
    a = seed.read_text().splitlines() if seed.is_file() else []
    b = final.read_text().splitlines() if final.is_file() else []
    out = []
    add = rem = 0
    for line in difflib.unified_diff(a, b, "seed Room.py", "authored Room.py", lineterm=""):
        cls = ""
        if line.startswith("+") and not line.startswith("+++"):
            cls, add = "add", add + 1
        elif line.startswith("-") and not line.startswith("---"):
            cls, rem = "del", rem + 1
        elif line.startswith("@@"):
            cls = "hunk"
        out.append(f'<div class="dl {cls}">{html.escape(line) or "&nbsp;"}</div>')
    header = f'<p class="muted">seed → authored: <b>+{add}</b> / <b>−{rem}</b> lines</p>'
    return header + '<div class="diff">' + "".join(out) + "</div>"


def build(scene_dir: Path, out: Path) -> None:
    traces = scene_dir / "scene_init" / "obj_stage" / "traces"
    events = load_trace(traces)
    raw = load_raw(traces)
    seed = scene_dir / "scene_init" / "scene_stage" / "room_init" / "room" / "Room.py"
    # The authored room moved to realism_authoring/ when stage 2 got its own output tree; older
    # runs still have it under scene_stage/_oneshot. Prefer the current spelling, fall back.
    final = next((p for p in (scene_dir / "realism_authoring" / "room" / "Room.py",
                              scene_dir / "scene_init" / "scene_stage" / "_oneshot" / "room" / "Room.py")
                  if p.is_file()), scene_dir / "realism_authoring" / "room" / "Room.py")

    # gather every image referenced, in first-seen order. Three sources: paths named in the
    # trace, the durable copies under traces/img/, and the scratch dir the model wrote into —
    # the last is where its own working images live, and they are the evidence for its calls.
    img_paths: list[str] = []
    for r in events:
        named = ([r.get("image")] if r.get("image") else []) + (r.get("images") or [])
        saved = [str(traces / s) for s in (r.get("saved") or [])]
        for p in saved + named:  # prefer the durable copy: the original may be long gone
            if p not in img_paths:
                img_paths.append(p)
    for extra in sorted((scene_dir / "realism_authoring" / "_scratch").rglob("*")):
        if extra.is_file() and extra.suffix.lower() in (".png", ".jpg", ".jpeg"):
            img_paths.append(str(extra))
    embedded: dict[str, str] = {}
    for p in img_paths:
        pp = Path(p)
        if p not in embedded and pp.is_file():
            uri = data_uri(pp)
            if uri:
                embedded[p] = uri

    # Results carry the answer to the call above them; pair by id so one block shows both.
    results = {r["id"]: r for r in events if r.get("kind") == "result" and r.get("id")}

    # ---- timeline rows ----
    rows = []
    for r in events:
        k = r.get("kind")
        dt = r.get("dt", "")
        badge = f'<span class="dt">+{dt}s</span>'
        if k == "result":
            continue  # rendered inside its tool call
        if k in ("session_start",):
            head = (f'<div class="ev start">{badge} <b>▶ {r.get("pass","")}</b> '
                    f'· model {html.escape(str(r.get("model","")))}'
                    f' · profile {html.escape(str(r.get("profile","")))}'
                    f' · max_turns {html.escape(str(r.get("max_turns","")))}</div>')
            # The prompt is what every later decision was a response to.
            rows.append(head + block("the prompt this session was given", str(r.get("prompt", ""))))
        elif k == "think":
            rows.append(f'<div class="ev think">{badge} 💭 {html.escape(r.get("text",""))}</div>')
        elif k == "tool":
            extra = ""
            if r.get("file"):
                extra = f' <span class="file">{html.escape(r["file"])}</span>'
                if r.get("delta_lines") is not None:
                    extra += f' <span class="delta">Δ{r["delta_lines"]:+} ln</span>'
            res = results.get(r.get("id", ""), {})
            if res.get("secs") is not None:
                extra += f' <span class="secs">{res["secs"]}s</span>'
            if res.get("is_error"):
                extra += ' <span class="err">error</span>'
            body = ""
            # The FULL input: for Bash the hint is the label and the command is the method.
            args = r.get("args") or {}
            for key in ("command", "prompt", "content", "new_string", "old_string"):
                if args.get(key):
                    body += block(f"{key} ({len(str(args[key]))} chars)", str(args[key]))
            rest = {k2: v for k2, v in args.items()
                    if k2 not in ("command", "prompt", "content", "new_string", "old_string")}
            if rest:
                body += block("arguments", json.dumps(rest, indent=2))
            if res.get("output"):
                label = f'output ({res.get("chars", len(res["output"]))} chars)'
                body += block(label, str(res["output"]), cls="out")
            rows.append(f'<div class="ev tool">{badge} 🔧 <b>{html.escape(str(r.get("tool","")))}</b>'
                        f'{extra} <span class="hint">{html.escape(str(r.get("hint",""))[:120])}</span>'
                        f'{body}</div>')
        elif k in ("images",) or r.get("image"):
            # Prefer the durable copy under traces/img/ — the original is often a render or a
            # /tmp crop that no longer exists by the time anyone reads the report.
            imgs = [str(traces / s) for s in (r.get("saved") or [])]
            imgs += [p for p in (([r.get("image")] if r.get("image") else [])
                                 + (r.get("images") or [])) if p not in imgs]
            thumbs = ""
            for p in imgs:
                if p in embedded:
                    thumbs += (f'<a href="{embedded[p]}" target="_blank">'
                               f'<img src="{embedded[p]}" title="{html.escape(p)}"></a>')
                else:
                    thumbs += f'<span class="missing" title="{html.escape(p)}">▧ (not on disk)</span>'
            rows.append(f'<div class="ev img">{badge} 🖼 {thumbs}</div>')
        elif k == "session_end":
            rows.append(f'<div class="ev end">{badge} ■ {r.get("calls","?")} calls · '
                        f'{r.get("seconds","?")}s · ${r.get("cost_usd",0) or 0:.2f}'
                        f'<div class="summary">{html.escape(str(r.get("summary","")))}</div></div>')

    # ---- image gallery (grouped by parent dir) ----
    groups: dict[str, list[str]] = {}
    for p, uri in embedded.items():
        groups.setdefault(Path(p).parent.name, []).append(uri)
    gallery = ""
    for g, uris in sorted(groups.items()):
        cells = "".join(f'<a href="{u}" target="_blank"><img src="{u}"></a>' for u in uris)
        gallery += f'<h3>{html.escape(g)} <span class="muted">({len(uris)})</span></h3><div class="grid">{cells}</div>'

    n_tools = sum(1 for r in events if r.get("kind") == "tool")

    # Earlier runs are kept, not overwritten — say so, and name them, or the retention is
    # invisible and the report reads as though this were the only run there has ever been.
    hist = sorted((traces / "history").glob("authoring_trace.*.jsonl"))
    if hist:
        items = "".join(
            f'<li><code>{html.escape(p.name)}</code> '
            f'<span class="muted">{p.stat().st_size / 1024:.0f} KB</span></li>' for p in hist)
        history_html = (f'<p class="muted">{len(hist)} archived trace file(s) from earlier runs, '
                        f'in <code>{html.escape(str(traces / "history"))}</code>. Each is tagged '
                        f'with the <code>run_NNN</code> of its scratch images.</p><ul>{items}</ul>')
    else:
        history_html = '<p class="muted">No earlier runs archived yet.</p>'

    # ---- raw sidecar ----
    # One collapsible per message rather than one 1 MB blob: the point of the raw dump is that
    # you can find the one message the curated timeline summarised away, and a wall of JSON is
    # no more searchable than the file itself.
    if raw:
        raw_rows = []
        for i, m in enumerate(raw, 1):
            msg = m.get("msg") or {}
            kind = msg.get("__type__", "?")
            preview = ", ".join(
                str(b.get("__type__", "?")) for b in (msg.get("content") or [])
                if isinstance(b, dict)) or ""
            raw_rows.append(block(f"[{i}] +{m.get('dt', '')}s  {kind}  {preview}".rstrip(),
                                  json.dumps(msg, indent=2, ensure_ascii=False)))
        raw_html = (f'<p class="muted">Every SDK message exactly as it arrived — the source the '
                    f'timeline above was derived from.</p>{"".join(raw_rows)}')
    else:
        raw_html = ('<p class="muted">No raw sidecar for this run. It is written by default; '
                    '<code>LR_TRACE_RAW=0</code> disables it.</p>')

    title = f"{scene_dir.name} — authoring trace report"
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:#0e0f12; color:#e6e6e6; }}
@media (prefers-color-scheme: light) {{ body {{ background:#fafafa; color:#1a1a1a; }} }}
header {{ padding:18px 22px; border-bottom:1px solid #333; position:sticky; top:0; background:inherit; z-index:5; }}
h1 {{ font-size:17px; margin:0 0 4px; }} .muted {{ opacity:.6; }}
nav button {{ font:inherit; background:#1d2027; color:inherit; border:1px solid #333; border-radius:7px; padding:5px 12px; margin-right:6px; cursor:pointer; }}
nav button.on {{ background:#2d6cdf; border-color:#2d6cdf; color:#fff; }}
main {{ padding:18px 22px; max-width:1100px; }}
section {{ display:none; }} section.on {{ display:block; }}
.ev {{ padding:5px 8px; border-left:3px solid #333; margin:3px 0; border-radius:3px; }}
.ev.think {{ border-color:#8a6; font-style:italic; opacity:.92; }}
.ev.tool {{ border-color:#59f; font-family:ui-monospace,monospace; font-size:12.5px; }}
.ev.img {{ border-color:#c7a; }} .ev.start {{ border-color:#2d6cdf; }} .ev.end {{ border-color:#c55; margin-top:10px; }}
.dt {{ display:inline-block; min-width:60px; opacity:.5; font-size:11px; }}
.file {{ color:#e9a; }} .delta {{ color:#7c7; }} .hint {{ opacity:.65; }}
.ev.img img {{ height:150px; margin:4px 6px 0 0; border-radius:5px; vertical-align:top; border:1px solid #333; }}
.grid {{ display:flex; flex-wrap:wrap; gap:8px; }} .grid img {{ height:170px; border-radius:6px; border:1px solid #333; }}
.summary {{ white-space:pre-wrap; opacity:.8; margin-top:6px; font-size:12.5px; }}
.diff {{ font-family:ui-monospace,monospace; font-size:12px; overflow-x:auto; border:1px solid #333; border-radius:8px; }}
.dl {{ padding:0 8px; white-space:pre; }}
.dl.add {{ background:rgba(80,200,120,.14); }} .dl.del {{ background:rgba(220,80,80,.14); }}
.dl.hunk {{ background:#1d2430; color:#8bf; }}
.missing {{ opacity:.5; font-size:12px; }}
.secs {{ opacity:.55; font-size:11px; margin-left:6px; }}
.err {{ color:#f88; font-size:11px; border:1px solid #a44; border-radius:4px; padding:0 5px; margin-left:6px; }}
.blk {{ margin:5px 0 2px 66px; }}
.blk summary {{ cursor:pointer; opacity:.62; font-size:11.5px; user-select:none; }}
.blk summary:hover {{ opacity:1; }}
.blk pre {{ white-space:pre-wrap; word-break:break-word; margin:5px 0 0; padding:9px 11px;
  background:rgba(128,128,128,.10); border-radius:6px; max-height:460px; overflow:auto;
  font-family:ui-monospace,monospace; font-size:11.5px; line-height:1.45; }}
.blk.out pre {{ border-left:3px solid #4a8; }}
#raw pre {{ white-space:pre-wrap; word-break:break-all; font-size:11px; }}
</style></head><body>
<header>
  <h1>{html.escape(scene_dir.name)} — authoring trace report</h1>
  <div class="muted">{len(events)} trace events · {n_tools} tool calls · {len(embedded)} images embedded ·
   {len(raw)} raw SDK messages · code diff seed→authored</div>
  <nav style="margin-top:10px">
    <button data-t="timeline" class="on">Timeline</button>
    <button data-t="images">Rendered images ({len(embedded)})</button>
    <button data-t="code">Code changes</button>
    <button data-t="raw">Raw messages ({len(raw)})</button>
    <button data-t="history">Earlier runs ({len(hist)})</button>
  </nav>
</header>
<main>
  <section id="timeline" class="on">{''.join(rows)}</section>
  <section id="images">{gallery or '<p class="muted">no images recovered</p>'}</section>
  <section id="code"><h3>Room.py — what the agent changed</h3>{diff_html(seed, final)}</section>
  <section id="raw">{raw_html}</section>
  <section id="history"><h3>Retained runs</h3>{history_html}</section>
</main>
<script>
document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));
  document.querySelectorAll('section').forEach(s=>s.classList.remove('on'));
  b.classList.add('on'); document.getElementById(b.dataset.t).classList.add('on');
}});
</script></body></html>"""
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1e6:.1f} MB, {len(embedded)} images, {len(events)} events)")


if __name__ == "__main__":
    scene = Path(sys.argv[1]).resolve()
    out = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else scene / "realism_authoring" / "room_preview" / "trace.html"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    build(scene, out)
