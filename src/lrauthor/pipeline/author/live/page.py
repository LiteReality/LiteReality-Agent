"""The live page: one HTML string, no build step and no sidecar files.

This is the *streaming* sibling of `room_ops/walk`. That one walks a finished room; this one
stays connected to a run in progress. The difference that matters is the swap: a finished viewer
loads its glb once, while this page replaces the geometry in place on every rebuild and never
touches the camera. Reloading the page instead would be far less code, but it throws away where the
viewer was standing — and watching an agent work is exactly when you want to keep looking at the
wall it is working on.

Three.js comes from a CDN, as both other viewers do, so nothing here adds a dependency to the repo.
"""

from __future__ import annotations

import html
import json

THREE = "0.160.0"

_PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__LABEL__ — LiteReality live</title>
<style>
  :root{--bg:#14161a;--panel:#1b1e24;--line:#2c313a;--ink:#e9eaee;--dim:#9aa2ae;--accent:#5ab0ff;
        --ok:#49c26a;--warn:#f0a53c;--bad:#ef6a6a}
  *{box-sizing:border-box}
  html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);
    font:13px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;overflow:hidden}
  #wrap{display:grid;grid-template-columns:1fr 380px;height:100%}
  #stage{position:relative;min-width:0}
  canvas{display:block;width:100%;height:100%}

  #bar{position:absolute;left:14px;top:12px;right:14px;display:flex;gap:10px;align-items:center;
       pointer-events:none;z-index:3}
  #bar .t{font-weight:600;font-size:14px}
  .pill{display:inline-flex;align-items:center;gap:7px;background:rgba(27,30,36,.9);
        border:1px solid var(--line);border-radius:20px;padding:5px 12px;backdrop-filter:blur(6px)}
  button.pill{pointer-events:auto;color:var(--ink);font:inherit;font-size:12px;cursor:pointer}
  button.pill:hover{border-color:var(--accent);color:var(--accent)}
  #art{pointer-events:auto;font-size:12px} #art label{color:var(--dim)}
  #art input{width:104px;accent-color:var(--accent)} #art #pct{color:var(--dim);min-width:44px}

  /* trace thumbnails — what the agent actually looked at, inline with the call that produced it */
  .ev .shots{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px}
  .ev .shots a{display:block;line-height:0}
  .ev .shots img{width:58px;height:44px;object-fit:cover;border-radius:4px;border:1px solid var(--line);
                 background:#0d0f12}
  .ev .shots img:hover{border-color:var(--accent)}
  .ev .more{color:var(--dim);font-size:10px;align-self:center;padding-left:2px}
  .ev .args{color:#8b93a0;font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
            overflow-wrap:anywhere}
  #lightbox{position:fixed;inset:0;background:rgba(8,9,11,.9);z-index:50;display:none;
            align-items:center;justify-content:center;padding:28px;cursor:zoom-out}
  #lightbox.on{display:flex}
  #lightbox img{max-width:100%;max-height:100%;border-radius:6px}
  /* Liveness. A solid green dot means "nothing wrong", which is also what it means after the run
     has ended — so on its own it never answers the question you actually have while watching:
     is this thing still working? The pulse is that answer, and it is driven by trace activity
     rather than by builds, because the long gaps are exactly when the agent is thinking or
     mid-edit and no build is in flight. */
  .dot{width:7px;height:7px;border-radius:50%;background:var(--ok);flex:0 0 auto}
  .dot.live{animation:pulse 1.6s ease-in-out infinite}
  .dot.building{background:var(--warn);animation:pulse 1s ease-in-out infinite}
  .dot.failed{background:var(--bad);animation:none}
  .dot.done{background:var(--dim);animation:none}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
  #hint{position:absolute;left:14px;bottom:12px;color:var(--dim);font-size:12px;z-index:3;
        pointer-events:none;text-shadow:0 1px 3px rgba(0,0,0,.6)}

  #side{background:var(--panel);border-left:1px solid var(--line);display:flex;flex-direction:column;
        min-height:0}
  #shead{padding:12px 14px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:7px}
  #shead b{font-size:13px} #shead span{color:var(--dim);font-size:11px}
  #shead #tstate{margin-left:1px} #shead #tcount{margin-left:auto}
  #feed{flex:1;overflow:auto;padding:6px 0;min-height:0}
  .ev{display:grid;grid-template-columns:52px 1fr;gap:9px;padding:5px 14px;border-left:2px solid transparent}
  .ev:hover{background:rgba(255,255,255,.035)}
  .ev .dt{color:#69707c;font-variant-numeric:tabular-nums;font-size:11px;text-align:right;padding-top:1px}
  .ev .bd{min-width:0}
  .ev .k{display:inline-block;font-size:10px;letter-spacing:.4px;text-transform:uppercase;
         color:var(--dim);margin-right:6px}
  .ev .d{overflow-wrap:anywhere;color:#cfd4dc}
  .ev.tool{border-left-color:var(--accent)} .ev.tool .k{color:var(--accent)}
  .ev.think{border-left-color:#7d7fe0} .ev.think .d{color:#b9bcf0;font-style:italic}
  .ev.err{border-left-color:var(--bad)} .ev.err .d{color:#ffb3b3}
  .ev.session{border-left-color:var(--ok)} .ev.session .k{color:var(--ok)}
  .ev .sub{color:#7c838f;font-size:11px}
  #empty{color:var(--dim);padding:18px 14px;font-size:12px}
  #foot{border-top:1px solid var(--line);padding:8px 14px;color:var(--dim);font-size:11px;
        display:flex;gap:10px}
  #foot b{color:var(--ink);font-weight:600}
</style>
</head><body>

<div id="wrap">
  <div id="stage">
    <canvas id="c"></canvas>
    <div id="bar">
      <span class="pill t">__LABEL__</span>
      <span class="pill"><i class="dot" id="dot"></i><span id="status">waiting for a build</span></span>
      <span class="pill" id="meta"></span>
      <button class="pill" id="reset" title="Frame the whole room again (R)">reset view</button>
      <span class="pill" id="art" hidden><label for="open">articulate</label>
        <input id="open" type="range" min="0" max="100" value="100"><span id="pct">open</span></span>
    </div>
    <div id="hint">drag to orbit · scroll to zoom · right-drag to pan · R to reset the view</div>
  </div>

  <div id="side">
    <div id="shead"><b>Agent trace</b><i class="dot" id="tdot"></i><span id="tstate">—</span>
      <span id="tcount">—</span></div>
    <div id="feed"><div id="empty">No trace yet. Events appear here as the agent works.</div></div>
    <div id="foot"><span>build <b id="fbuild">—</b></span><span>compile <b id="fsecs">—</b></span>
      <span>objects <b id="fobj">—</b></span></div>
  </div>
</div>

<div id="lightbox"><img id="lightboxImg" alt=""></div>

<script type="importmap">{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@__THREE__/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@__THREE__/examples/jsm/"
}}</script>
<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const POLL = __POLL__;

const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 500);
camera.position.set(6, 5, 8);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }
}
(function tick() { resize(); controls.update(); renderer.render(scene, camera); requestAnimationFrame(tick); })();

const draco = new DRACOLoader();
draco.setDecoderPath('https://cdn.jsdelivr.net/npm/three@__THREE__/examples/jsm/libs/draco/');
const loader = new GLTFLoader().setDRACOLoader(draco);

/* Free a swapped-out room. Three keeps GPU buffers alive until they are explicitly disposed, so
   without this a long authoring session leaks a full room per rebuild and eventually loses the
   WebGL context — the viewer would die exactly during the long runs it exists to watch. */
function dispose(root) {
  root.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    for (const m of [].concat(o.material || [])) {
      if (!m) continue;
      for (const k of Object.keys(m)) { const v = m[k]; if (v && v.isTexture) v.dispose(); }
      m.dispose();
    }
  });
}

let current = null, framed = false;

function frame(root) {
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) return;
  const size = box.getSize(new THREE.Vector3()), mid = box.getCenter(new THREE.Vector3());
  const span = Math.max(size.x, size.y, size.z) || 5;
  controls.target.copy(mid);
  camera.position.copy(mid).add(new THREE.Vector3(span * 0.9, span * 0.75, span * 0.9));
  camera.near = span / 500; camera.far = span * 40; camera.updateProjectionMatrix();
}

/* Articulation. The compiled room carries one baked clip per moving part (Door_*_swing,
   Drawer_*_slide). Every action is played but paused, so the slider scrubs them together as a
   position rather than animating anything.

   Rebuilt on every swap because the mixer is bound to the glTF that was replaced — but the slider
   VALUE is deliberately not reset, for the same reason the camera isn't: if you opened the doors to
   look through a doorway, the next rebuild should not close them on you. */
let mixer = null, actions = [], openAt = [];
const artBox = document.getElementById('art');
const artSlider = document.getElementById('open');

/* Where in a clip the part is furthest from its rest pose.

   The baked clips are round TRIPS — a door swings open and shuts again inside one clip, so the
   first and last keyframes are both the closed pose (the swing peaks around a third of the way
   in). Scrubbing to `duration` therefore shows a CLOSED door while claiming to be open, which is
   why the slider looked inert: both ends of its travel were the same pose. Mapping the slider onto
   each clip's own peak instead makes 0 mean shut and 1 mean fully open, per part. */
function peakTime(clip) {
  let best = 0, bestDev = 0;
  for (const tr of clip.tracks) {
    const v = tr.values, t = tr.times, stride = v.length / t.length;
    for (let i = 0; i < t.length; i++) {
      let dev = 0;
      for (let k = 0; k < stride; k++) dev = Math.max(dev, Math.abs(v[i * stride + k] - v[k]));
      if (dev > bestDev) { bestDev = dev; best = t[i]; }
    }
  }
  return bestDev > 1e-4 ? best : clip.duration;   // a clip that never moves: nothing to aim at
}

function articulate(frac) {
  for (let i = 0; i < actions.length; i++) actions[i].time = frac * openAt[i];
  if (mixer) mixer.update(0);
  document.getElementById('pct').textContent =
    frac <= 0.02 ? 'shut' : frac >= 0.98 ? 'open' : Math.round(frac * 100) + '%';
}

function bindClips(gltf, root) {
  mixer = null; actions = []; openAt = [];
  const clips = gltf.animations || [];
  artBox.hidden = clips.length === 0;
  if (!clips.length) return;
  mixer = new THREE.AnimationMixer(root);
  actions = clips.map((c) => { const a = mixer.clipAction(c); a.play(); a.paused = true; return a; });
  openAt = clips.map(peakTime);
  articulate(artSlider.value / 100);
}
artSlider.addEventListener('input', (e) => articulate(e.target.value / 100));

async function swap(build) {
  const gltf = await loader.loadAsync('room.glb?b=' + build);
  const root = gltf.scene;
  // Blender is Z-up, three is Y-up. The exporter already rotates, so nothing to do here — but
  // keep the room centred on its own origin so orbiting feels the same across rebuilds.
  if (current) { scene.remove(current); dispose(current); }
  current = root; scene.add(root);
  bindClips(gltf, root);
  if (!framed) { frame(root); framed = true; }   // never re-frame: the camera is the user's
  let n = 0; root.traverse((o) => { if (o.isMesh) n++; });
  document.getElementById('fobj').textContent = n;
}

/* ---- state polling ---------------------------------------------------------------------- */
const feed = document.getElementById('feed');
const dot = document.getElementById('dot');
const tdot = document.getElementById('tdot');
const statusEl = document.getElementById('status');
let seenBuild = -1, cursor = 0, loading = false, lastBuild = '';

/* How long the trace may go quiet before the page stops claiming the agent is working. Generous
   on purpose: a single render or compile tool call blocks the trace for the better part of a
   minute, and a dot that drops out mid-call reads as "it died" every time the agent does the slow
   thing. Erring the other way costs at most one stale minute after a run is killed. */
const LIVE_S = 120;
const TITLE = document.title;
let titled = null;

const ago = (s) => s < 60 ? Math.round(s) + 's'
                 : s < 3600 ? Math.round(s / 60) + 'm' : Math.round(s / 3600) + 'h';

const KIND = {
  tool: 'tool', think: 'think', session_start: 'session', session_end: 'session',
  trace_start: 'session', result: 'result', images: 'images', build_progress: 'result',
};

/* The one argument that says what a call was FOR. A tool line reading just "Bash" is noise; the
   description/command/query is the thing you scan the feed for. */
function subject(e) {
  const a = e.args || {};
  const pick = a.description || a.command || a.query || a.target || a.file_path
            || a.pattern || a.name || a.prompt || e.hint;
  if (!pick) return '';
  const s = String(pick);
  // Absolute run paths are almost all shared prefix; the tail is the only part that identifies
  // anything. Applied to whatever won above, since file_path and hint are both usually absolute.
  return s.startsWith('/') && s.includes('/') ? s.split('/').slice(-2).join('/') : s;
}

function shots(e) {
  // `saved` is the durable copy the tracer wrote beside the run; `images` may be scratch paths
  // that no longer exist, so only `saved` is offered as a thumbnail.
  const paths = Array.isArray(e.saved) ? e.saved : [];
  if (!paths.length) return null;
  const wrap = document.createElement('div');
  wrap.className = 'shots';
  for (const p of paths.slice(0, 6)) {
    const a = document.createElement('a');
    a.href = 'img?p=' + encodeURIComponent(p);
    const img = document.createElement('img');
    img.src = a.href; img.loading = 'lazy'; img.alt = p; img.title = p;
    img.addEventListener('click', (ev) => {
      ev.preventDefault();
      document.getElementById('lightboxImg').src = a.href;
      document.getElementById('lightbox').classList.add('on');
    });
    a.appendChild(img); wrap.appendChild(a);
  }
  if (paths.length > 6) {
    const more = document.createElement('span');
    more.className = 'more'; more.textContent = '+' + (paths.length - 6);
    wrap.appendChild(more);
  }
  return wrap;
}

function line(e) {
  const cls = e.is_error ? 'err' : (KIND[e.kind] || '');
  const el = document.createElement('div');
  el.className = 'ev ' + cls;
  const dt = (e.dt == null) ? '' : (e.dt < 60 ? e.dt.toFixed(1) + 's'
            : Math.floor(e.dt / 60) + 'm' + String(Math.round(e.dt % 60)).padStart(2, '0'));
  let kind = e.kind || 'log', detail = '', sub = '';
  if (e.kind === 'tool') { kind = e.tool || 'tool'; detail = subject(e); }
  else if (e.kind === 'think') { detail = e.text || ''; }
  else if (e.kind === 'result') { kind = 'ok'; detail = (e.secs != null ? e.secs.toFixed(1) + 's' : '');
                                  if (e.chars) sub = e.chars + ' chars'; }
  else if (e.kind === 'session_start') { detail = (e.model || '') + ' · ' + (e.pass || ''); }
  else if (e.kind === 'session_end') { detail = (e.calls || 0) + ' calls'
                                       + (e.cost_usd ? ' · $' + Number(e.cost_usd).toFixed(2) : ''); }
  else if (e.kind === 'images') { detail = (e.n_images || 0) + ' image(s)'; }
  else if (e.kind === 'build_progress') {
    kind = 'build';
    detail = (e.built || 0) + '/' + (e.total || 0) + ' built';
    const per = e.branches ? Object.keys(e.branches).map((k) => k + ' ' + e.branches[k]) : [];
    /* Which objects are mid-build matters more than the tally during a 20-minute pass, so it
       leads; the per-branch split follows for the ones with nothing running. */
    sub = (e.active && e.active.length ? 'building ' + e.active.join(', ') + '  ·  ' : '')
        + per.join('  ·  ');
  }
  else { detail = e.msg || e.name || e.stage || e.status || ''; }

  el.innerHTML = '<div class="dt"></div><div class="bd"><span class="k"></span>'
               + '<span class="d"></span><div class="sub"></div></div>';
  el.querySelector('.dt').textContent = dt;
  el.querySelector('.k').textContent = kind;
  el.querySelector('.d').textContent = String(detail).slice(0, 400);
  el.querySelector('.sub').textContent = sub;

  // an Edit says how big it was; a path-y tool says which file — both without re-reading the hint
  const body = el.querySelector('.bd');
  if (e.delta_lines != null || (e.kind === 'tool' && e.hint && e.hint !== detail)) {
    const args = document.createElement('div');
    args.className = 'args';
    args.textContent = e.delta_lines != null
      ? `${e.file || 'Room.py'}  ${e.delta_lines > 0 ? '+' : ''}${e.delta_lines} lines`
      : String(e.hint).split('/').slice(-2).join('/');
    body.appendChild(args);
  }
  const thumbs = shots(e);
  if (thumbs) body.appendChild(thumbs);
  return el;
}

async function poll() {
  try {
    const s = await fetch('state?since=' + cursor, { cache: 'no-store' }).then((r) => r.json());

    const busy = s.status === 'building' || s.baking;
    // `idle_s` is null until the first trace event lands, which is not the same as a quiet agent.
    const quiet = s.idle_s == null || s.idle_s > LIVE_S;
    const working = !s.finished && (busy || !quiet);

    dot.className = 'dot ' + (busy ? 'building' : s.status === 'failed' ? 'failed'
                            : working ? 'live' : s.finished ? 'done' : '');
    dot.title = working ? 'the agent is still working on this room'
      : s.finished ? 'the run has ended — the room stays here to look at'
      : 'no agent activity for ' + ago(s.idle_s == null ? 0 : s.idle_s);

    tdot.className = 'dot ' + (working ? 'live' : s.finished ? 'done' : '');
    document.getElementById('tstate').textContent = working ? 'working'
      : s.finished ? 'ended' : s.idle_s == null ? 'no events yet' : 'quiet ' + ago(s.idle_s);

    /* A live run is usually watched from another tab. The bullet is the only part of this page
       visible from there, so it carries the same answer the dot does. */
    if (working !== titled) { document.title = (working ? '● ' : '') + TITLE; titled = working; }

    /* `finished` outranks the idle build line but not a build in flight: a run that has ended may
       still be baking its last save, and claiming it is done while the page is about to swap
       geometry underneath the viewer reads as a glitch. */
    statusEl.textContent = s.status === 'building' ? 'compiling…'
      : s.status === 'failed' ? (s.error || 'build failed')
      : s.baking ? 'baking materials…'
      : s.finished ? s.finished + (s.build ? ' · build ' + s.build : '')
      : s.build > 0 ? 'build ' + s.build + ' · ' + (s.phase === 'baked' ? 'materials' : 'geometry')
      : 'waiting for a build';
    document.getElementById('meta').textContent = s.mb ? s.mb.toFixed(0) + ' MB' : '';
    document.getElementById('fbuild').textContent = s.build || '—';
    document.getElementById('fsecs').textContent = (s.compile_s ? s.compile_s.toFixed(1) + 's' : '—')
      + (s.bake_s ? ' +' + s.bake_s.toFixed(0) + 's bake' : '');

    if (s.events && s.events.length) {
      const stick = feed.scrollTop + feed.clientHeight > feed.scrollHeight - 60;
      const first = document.getElementById('empty');
      if (first) first.remove();
      for (const e of s.events) {
        /* Heartbeats exist to keep the liveness dot honest through a 20-minute object build; every
           one of them reaching the feed would bury the agent's actual steps. Keep the ones that
           moved the count or changed what is building, drop the rest — they have already done
           their job by refreshing `idle_s` server-side. */
        if (e.kind === 'build_progress') {
          const sig = e.built + '|' + ((e.active || []).join(','));
          if (sig === lastBuild) continue;
          lastBuild = sig;
        }
        feed.appendChild(line(e));
      }
      cursor = s.cursor;
      document.getElementById('tcount').textContent = s.total + ' events';
      if (stick) feed.scrollTop = feed.scrollHeight;
    }

    if (s.build > 0 && s.build !== seenBuild && !loading) {
      loading = true;
      const b = s.build;
      try { await swap(b); seenBuild = b; } catch (err) { console.error('swap failed', err); }
      loading = false;
    }
  } catch (err) { /* server restarting — keep polling */ }
  setTimeout(poll, POLL);
}
poll();
/* Re-frame on demand. The swap deliberately never moves the camera, which is right while you are
   watching one wall — but it means a viewer that has wandered inside the geometry has no way back
   short of reloading, and reloading costs a full glb parse. */
document.getElementById('reset').addEventListener('click', () => { if (current) frame(current); });
document.getElementById('lightbox').addEventListener('click', (e) =>
  e.currentTarget.classList.remove('on'));
addEventListener('keydown', (e) => {
  if (e.key === 'Escape') document.getElementById('lightbox').classList.remove('on');
});
addEventListener('keydown', (e) => {
  if (e.key === 'r' && !e.metaKey && !e.ctrlKey && current) frame(current);
});
</script>
</body></html>
"""


def render(label: str, poll_ms: int = 700) -> str:
    """The full page for one live room. `label` is shown in the title bar and browser tab."""
    return (
        _PAGE.replace("__LABEL__", html.escape(label))
        .replace("__THREE__", THREE)
        .replace("__POLL__", json.dumps(int(poll_ms)))
    )
