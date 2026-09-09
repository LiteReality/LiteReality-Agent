/* Vendored from Litereality-agent-site/vr/app.js, which `make_vr_pages.py` generates in the
   litereality-agent-web working area. Kept byte-identical apart from the Draco decoder URL
   below, which points at the R2 Worker on the site and has to reach a public CDN here.
   Upstream fixes belong in that generator; re-vendor rather than editing this copy. */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const CFG      = window.SCENE;
const IS_TOUCH = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
const REDUCED  = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* Tuning. The movement constants are per-frame-at-60Hz, as in the page this is modelled on; every
   use below is scaled by `n` (elapsed 60Hz frames) so a 120 Hz display walks at the same speed. */
const FOV_ORBIT = 45, FOV_FP = 75;
const EYE = 1.4, PITCH_LIMIT = Math.PI / 3, LOOK_SENS = 0.0013;
const MOVE = { maxSpeed:0.004, accel:0.004, decel:0.0035, damp:0.92,
               jump:0.1, gravity:0.008, maxJumpHeight:1.8, duckDepth:0.8, duckForce:0.08 };
const TWEEN_MS = 1000;

/* ------------------------------------------------------------------ loading screen */
const LS = {
  el:  document.querySelector('.loading-screen'),
  bar: document.querySelector('.progress-bar'),
  pct: document.querySelector('.progress-text'),
  txt: document.querySelector('.loading-text'),
  set(f, label){
    const p = Math.max(0, Math.min(100, Math.round(f * 100)));
    this.bar.style.width = p + '%'; this.pct.textContent = p + '%';
    if (label) this.txt.textContent = label;
  },
  hide(){ this.el.style.opacity = '0'; setTimeout(() => this.el.style.display = 'none', 500); },
};

/* ------------------------------------------------------------------ scene */
const canvas = document.querySelector('canvas.webgl');
const scene  = new THREE.Scene();
scene.background = new THREE.Color('#bcd2d3');

const sizes = { width: innerWidth, height: innerHeight };
const camera = new THREE.PerspectiveCamera(FOV_ORBIT, sizes.width / sizes.height, 0.05, 500);
scene.add(camera);

const renderer = new THREE.WebGLRenderer({
  canvas, antialias: true, powerPreference: IS_TOUCH ? 'low-power' : 'high-performance',
});
renderer.setSize(sizes.width, sizes.height);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.setClearColor('#bcd2d3');

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const orbit = new OrbitControls(camera, canvas);
orbit.enableDamping = true; orbit.dampingFactor = 0.08; orbit.enabled = false;

addEventListener('resize', () => {
  sizes.width = innerWidth; sizes.height = innerHeight;
  camera.aspect = sizes.width / sizes.height; camera.updateProjectionMatrix();
  renderer.setSize(sizes.width, sizes.height);
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
});

/* ------------------------------------------------------------------ point-and-go marker */
const marker = new THREE.Group();
marker.add(new THREE.Mesh(
  new THREE.CircleGeometry(0.15, 32),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5,
                                side: THREE.DoubleSide })));
marker.add(new THREE.Mesh(
  new THREE.RingGeometry(0.15, 0.175, 32),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9,
                                side: THREE.DoubleSide })));
marker.visible = false;
scene.add(marker);

/* ------------------------------------------------------------------ articulation
   Every baked clip drives exactly one named node that carries its own mesh (Door_Leaf, Drawer_L,
   Sash_Right, …). We index each animated node name → its clip so a ray-hit can be traced up to the
   part it belongs to, and a click toggles that part open or shut. The clip plays in real time at
   its own duration, eased, and reverses cleanly if clicked mid-swing. */
const ARTIC = { list: [] };
const raycaster = new THREE.Raycaster();

function clipLabel(name){ return name.replace(/_(swing|slide|open|move)$/i, '').replace(/_/g, ' '); }

/* Tag each clip's target node (resolved the same way the mixer binds it, so names/uuids can't
   drift) with its entry. A ray-hit then walks up to the first tagged ancestor. */
function findArticulated(obj){
  for (let o = obj; o; o = o.parent){
    if (o.userData && o.userData.__artic) return o.userData.__artic;
  }
  return null;
}

function toggleArticulation(e){
  e.open = !e.open;
  e.target = e.open ? 1 : 0;
  e.moving = true;
}

function tryArticulate(cx, cy){
  if (!ARTIC.list.length) return null;
  raycaster.setFromCamera(new THREE.Vector2((cx / innerWidth) * 2 - 1, -(cy / innerHeight) * 2 + 1), camera);
  const hits = raycaster.intersectObjects(scene.children.filter(o => o !== marker), true);
  for (const h of hits){
    const e = findArticulated(h.object);
    if (e){ toggleArticulation(e); return e; }
  }
  return null;
}

function stepArticulation(dt){
  if (!mixer) return;
  let changed = false;
  for (const e of ARTIC.list){
    if (!e.moving) continue;
    // Travel time is decoupled from the baked clip length: a 4 s door swing would feel sluggish, so
    // every part opens in a snappy 0.7–1.6 s while `phase` still scrubs the whole clip.
    const rate = 1 / Math.min(Math.max(e.dur, 0.7), 1.6);
    e.phase = e.phase < e.target ? Math.min(e.target, e.phase + rate * dt)
                                 : Math.max(e.target, e.phase - rate * dt);
    const k = e.phase * e.phase * (3 - 2 * e.phase);   // smoothstep, so the swing eases at both ends
    e.action.time = k * e.dur;
    if (e.phase === e.target) e.moving = false;
    changed = true;
  }
  if (changed) mixer.update(0);
}

/* ------------------------------------------------------------------ room measurements */
const ROOM = { box: null, center: new THREE.Vector3(), floorY: 0, ceilings: [] };
const VIEW = { ORBIT: null, FP: null };

function topLevel(root){
  /* Some exporters wrap everything in a single unnamed `Scene`/`RootNode`. Step through it so the
     ceiling lookup and the door lookup see the real room nodes. */
  let kids = root.children;
  while (kids.length === 1 && kids[0].children.length && /^(scene|rootnode|root|)$/i.test(kids[0].name || ''))
    kids = kids[0].children;
  return kids;
}

function measure(root){
  /* Bounds come from the SHELL, not the whole scene: the export also carries helpers such as
     `daylight_card` that sit outside the walls, which would inflate the box until the walk clamp
     stopped containing anything and could drag the floor plane below the actual floor. */
  const shell = new THREE.Box3(); let got = false, fy = null;
  root.traverse(o => {
    if (!o.isMesh) return;
    const n = o.name || '';
    if (/^(wall|floor|ceiling)/i.test(n)) { shell.expandByObject(o); got = true; }
    if (/^floor/i.test(n)) {
      const top = new THREE.Box3().setFromObject(o).max.y;
      fy = (fy === null) ? top : Math.min(fy, top);
    }
  });
  ROOM.box = got ? shell : new THREE.Box3().setFromObject(root);
  ROOM.box.getCenter(ROOM.center);
  ROOM.floorY = (fy === null) ? ROOM.box.min.y : fy;

  const nodes = topLevel(root);

  /* Ceiling (and the ceiling light grid) hide in orbit, so it reads as a dollhouse. */
  for (const child of nodes) if (/^ceiling/i.test(child.name || '')) ROOM.ceilings.push(child);

  const size = ROOM.box.getSize(new THREE.Vector3());
  const r = Math.max(size.x, size.y, size.z) * 0.5;
  const d = r / Math.sin(THREE.MathUtils.degToRad(FOV_ORBIT * 0.5));
  VIEW.ORBIT = {
    pos: new THREE.Vector3(ROOM.center.x + d * 0.8, ROOM.center.y + d * 0.55, ROOM.center.z + d * 0.8),
    target: ROOM.center.clone(),
  };

  /* Enter where a person would: just inside the door, facing the room. Falls back to standing at
     one end of the longer axis looking down it. */
  const eye = ROOM.floorY + EYE;
  let door = null;
  for (const child of nodes) if (/^door\d*$/i.test(child.name || '')) { door = child; break; }
  let pos;
  if (door) {
    const dc = new THREE.Box3().setFromObject(door).getCenter(new THREE.Vector3());
    const inward = new THREE.Vector3(ROOM.center.x - dc.x, 0, ROOM.center.z - dc.z);
    if (inward.lengthSq() < 1e-6) inward.set(0, 0, 1);
    inward.normalize();
    pos = new THREE.Vector3(dc.x, eye, dc.z).addScaledVector(inward, 1.8);
  } else {
    const long = size.x >= size.z ? 'x' : 'z';
    pos = new THREE.Vector3(ROOM.center.x, eye, ROOM.center.z);
    pos[long] = ROOM.box.min[long] + size[long] * 0.25;
  }
  clampToRoom(pos);
  VIEW.FP = { pos, target: new THREE.Vector3(ROOM.center.x, eye, ROOM.center.z) };
}

function clampToRoom(v){
  if (!ROOM.box) return v;
  const m = 0.35;
  v.x = Math.min(Math.max(v.x, ROOM.box.min.x + m), ROOM.box.max.x - m);
  v.z = Math.min(Math.max(v.z, ROOM.box.min.z + m), ROOM.box.max.z - m);
  return v;
}

/* ------------------------------------------------------------------ load */
const draco = new DRACOLoader().setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/libs/draco/');
const loader = new GLTFLoader().setDRACOLoader(draco);
let mixer = null, roomRoot = null;

loader.load(CFG.url, gltf => {
  LS.set(0.95, 'Preparing Scene…');
  const root = gltf.scene;
  roomRoot = root;                      // the banana physics bounces off this, and nothing else
  scene.add(root);
  measure(root);
  buildObjects(root);

  /* Everything starts shut, so the room reads as real — clicking a door, drawer or window is what
     articulates it (see the ARTIC index above). Each action is parked, paused, at time 0 and driven
     by hand from stepArticulation. */
  const clips = gltf.animations || [];
  if (clips.length){
    mixer = new THREE.AnimationMixer(root);
    for (const clip of clips){
      const action = mixer.clipAction(clip);
      action.play(); action.paused = true; action.time = 0;
      const entry = { action, dur: clip.duration || 1, phase: 0, target: 0, moving: false,
                      open: false, label: clipLabel(clip.name) };
      ARTIC.list.push(entry);
      for (const track of clip.tracks){
        const name = THREE.PropertyBinding.parseTrackName(track.name).nodeName;
        const node = name && THREE.PropertyBinding.findNode(root, name);
        if (node) node.userData.__artic = entry;
      }
    }
    mixer.update(0);
    attachArticulation();
  }
  /* After the articulation handler, so a click that opens a door is already flagged by the time the
     launcher sees it — doors keep priority, and you don't open a drawer *and* fire into it. */
  /* banana easter-egg not included in this template */

  camera.position.copy(VIEW.ORBIT.pos);
  camera.lookAt(VIEW.ORBIT.target);
  orbit.target.copy(VIEW.ORBIT.target);
  LS.set(1, 'Ready');
  LS.hide();
  (window.__vrDefault ? window.__vrDefault() : setMode(MODES.POINTER));  // default: side-by-side compare
}, ev => {
  if (ev.total) LS.set((ev.loaded / ev.total) * 0.92, 'Loading Model…');
  else LS.set(Math.min(0.9, ev.loaded / 12e6), 'Loading Model…');
}, err => {
  console.error(err);
  LS.txt.textContent = 'Could not load the scene';
  LS.el.querySelector('.loading-sub').textContent =
    'These pages stream the model over HTTP — open the site through serve.sh, not as a file:// path.';
});

/* ------------------------------------------------------------------ modes */
const MODES = { ORBIT:'ORBIT', POINTER:'POINTER', FPS:'FPS' };
const MODE_LIST = [
  { id: MODES.ORBIT,   label: 'Orbit',        hint: 'Drag to orbit · scroll to zoom · right-drag to pan' },
  { id: MODES.POINTER, label: 'Point and Go', hint: 'Click the floor to walk · click a door or drawer to open it · drag to look' },
  { id: MODES.FPS,     label: 'Walk',         hint: 'WASD to Move | SPACE Jump | SHIFT Duck | ESC to Exit' },
];

const panel     = document.querySelector('.mode-buttons');
const hintPill  = document.querySelector('.fps-info');
const crosshair = document.querySelector('.crosshair');
const objPanel  = document.getElementById('objPanel');

let mode = MODES.POINTER, busy = false, locked = false;
let pitch = 0, hintTimer = null;

for (const m of MODE_LIST){
  const b = document.createElement('button');
  b.className = 'mode-button'; b.textContent = m.label; b.dataset.id = m.id;
  b.addEventListener('click', () => setMode(m.id));
  panel.appendChild(b);
}

function showHint(text, sticky){
  clearTimeout(hintTimer);
  hintPill.textContent = text;
  hintPill.style.opacity = '1';
  if (!sticky) hintTimer = setTimeout(() => { hintPill.style.opacity = '0'; }, 2800);
}

function setButtons(disabled){
  panel.querySelectorAll('.mode-button').forEach(b => {
    b.disabled = disabled;
    b.classList.toggle('active', !disabled && b.dataset.id === mode);
  });
}

function setMode(next){
  if (busy || !VIEW.ORBIT) return;
  busy = true;
  mode = next;
  panel.querySelectorAll('.mode-button')
       .forEach(b => b.classList.toggle('active', b.dataset.id === next));
  setButtons(true);

  // tear down whatever the previous mode installed
  canvas.removeEventListener('mousedown', onMouseDown);
  canvas.removeEventListener('mousemove', onMouseMove);
  canvas.removeEventListener('mouseup',   onMouseUp);
  canvas.removeEventListener('click',     onClick);
  removeEventListener('keydown', onKeyDown);
  removeEventListener('keyup',   onKeyUp);
  detachTouch(); detachPointerLock();
  orbit.enabled = false;
  marker.visible = false;
  canvas.style.cursor = '';
  if (document.pointerLockElement) document.exitPointerLock();
  locked = false;
  resetKeys();

  // the objects panel and camera strip only make sense from the dollhouse view
  objPanel.hidden = next !== MODES.ORBIT || !OBJS.list.length;
  camStrip.hidden = !(next === MODES.ORBIT && CAMV.list.length);
  if (next !== MODES.ORBIT){
    if (OBJS.wire) setWire(false);       // don't carry wireframe into walking
    closeCamera(true);                    // drop any open comparison without a tween-back
  }
  applyVisibility(next);
  crosshair.style.opacity = next === MODES.FPS ? '1' : '0';

  const view = next === MODES.ORBIT ? VIEW.ORBIT : VIEW.FP;
  camera.fov = next === MODES.ORBIT ? FOV_ORBIT : FOV_FP;
  camera.updateProjectionMatrix();

  if (next === MODES.ORBIT){
    orbit.target.copy(view.target);
    tweenTo(view.pos, view.target, () => {
      orbit.enabled = true; orbit.update();
      busy = false; setButtons(false);
      if (!CAMV.list.length) showHint(MODE_LIST[0].hint);   // strip occupies the top otherwise
    });
    return;
  }

  tweenTo(view.pos, view.target, () => {
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup',   onMouseUp);
    canvas.addEventListener('click',     onClick);
    addEventListener('keydown', onKeyDown);
    addEventListener('keyup',   onKeyUp);
    if (IS_TOUCH) attachTouch();
    if (next === MODES.FPS){
      attachPointerLock();
      if (!IS_TOUCH) requestAnimationFrame(() => canvas.requestPointerLock());
    }
    busy = false; setButtons(false);
    showHint(next === MODES.FPS ? MODE_LIST[2].hint : MODE_LIST[1].hint, next === MODES.FPS);
  });
}

function tweenTo(pos, target, done){
  if (REDUCED){
    camera.position.copy(pos); camera.lookAt(target);
    pitch = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ').x;
    done && done(); return;
  }
  const fromP = camera.position.clone(), fromQ = camera.quaternion.clone();
  const aim = new THREE.PerspectiveCamera();
  aim.position.copy(pos); aim.lookAt(target);
  const toQ = aim.quaternion.clone();
  const t0 = performance.now();
  (function step(){
    const k = Math.min((performance.now() - t0) / TWEEN_MS, 1);
    const e = 1 - Math.pow(1 - k, 4);
    camera.position.lerpVectors(fromP, pos, e);
    camera.quaternion.copy(fromQ).slerp(toQ, e);
    if (k < 1) return requestAnimationFrame(step);
    camera.position.copy(pos); camera.quaternion.copy(toQ);
    pitch = new THREE.Euler().setFromQuaternion(camera.quaternion, 'YXZ').x;
    done && done();
  })();
}

/* ------------------------------------------------------------------ pointer lock (Walk) */
function attachPointerLock(){
  canvas.addEventListener('click', requestLock);
  document.addEventListener('pointerlockchange', onLockChange);
}
function detachPointerLock(){
  canvas.removeEventListener('click', requestLock);
  document.removeEventListener('pointerlockchange', onLockChange);
}
function requestLock(){ if (mode === MODES.FPS && !locked) canvas.requestPointerLock(); }
function onLockChange(){
  locked = document.pointerLockElement === canvas;
  if (locked) showHint(MODE_LIST[2].hint, true);
  else if (mode === MODES.FPS) setMode(MODES.POINTER);   // esc drops you back to Point and Go
}

/* ------------------------------------------------------------------ mouse look / point / go */
let dragging = false, dragged = false, lastX = 0, lastY = 0;

function onMouseDown(e){
  if (busy) return;
  if (mode === MODES.POINTER){ dragging = true; dragged = false; lastX = e.clientX; lastY = e.clientY; }
}
function onMouseUp(){ dragging = false; }

function onMouseMove(e){
  if (busy) return;
  if (mode === MODES.POINTER){
    castMarker(e.clientX, e.clientY);
    if (dragging){
      look(e.clientX - lastX, e.clientY - lastY, false);
      lastX = e.clientX; lastY = e.clientY;
    }
  } else if (mode === MODES.FPS && locked){
    look(e.movementX, e.movementY, true);
  }
}

function look(dx, dy, inverted){
  dragged = true;
  const euler = new THREE.Euler(0, 0, 0, 'YXZ').setFromQuaternion(camera.quaternion);
  euler.y += (inverted ? -dx : dx) * LOOK_SENS;
  const next = pitch + (inverted ? -dy : dy) * LOOK_SENS;
  if (Math.abs(next) < PITCH_LIMIT){ pitch = next; euler.x = pitch; }
  camera.quaternion.setFromEuler(euler);
}

function castMarker(cx, cy){
  const ndc = new THREE.Vector2((cx / innerWidth) * 2 - 1, -(cy / innerHeight) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  const hits = raycaster.intersectObjects(scene.children.filter(o => o !== marker), true);
  if (!hits.length || camera.position.distanceTo(hits[0].point) < 1){ marker.visible = false; canvas.style.cursor = ''; return; }
  // Pointing at a door/drawer/window: swap the "walk here" disc for a click-to-open cursor, so it
  // is clear the click opens the part rather than walking you to it.
  if (findArticulated(hits[0].object)){ marker.visible = false; canvas.style.cursor = 'pointer'; return; }
  canvas.style.cursor = '';
  const hit = hits[0];
  marker.position.copy(hit.point);
  const nrm = hit.face
    ? hit.face.normal.clone().transformDirection(hit.object.matrixWorld)
    : new THREE.Vector3(0, 1, 0);
  const up = new THREE.Vector3(0, 1, 0);
  const basis = new THREE.Matrix4();
  if (Math.abs(nrm.dot(up)) > 0.9999){
    // a floor or ceiling hit: `lookAt` needs an up vector that is not parallel to the normal
    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    fwd.y = 0; fwd.normalize();
    basis.lookAt(new THREE.Vector3(), nrm, fwd);
  } else {
    basis.lookAt(new THREE.Vector3(), nrm, up);
  }
  marker.quaternion.setFromRotationMatrix(basis);
  marker.position.addScaledVector(nrm, 0.01);
  marker.visible = true;
}

/* A click on an articulated part opens/shuts it in ANY mode, and takes priority over walking. These
   handlers are added once, for the life of the page; a tap is a press that neither moved far nor
   lasted long, which is what separates "open this door" from an orbit-drag or a look-drag. */
let tapX = 0, tapY = 0, tapT = 0, suppressWalkUntil = 0;
function attachArticulation(){
  canvas.addEventListener('pointerdown', e => { tapX = e.clientX; tapY = e.clientY; tapT = performance.now(); });
  canvas.addEventListener('pointerup', e => {
    if (busy) return;
    if (Math.hypot(e.clientX - tapX, e.clientY - tapY) > 6 || performance.now() - tapT > 500) return;
    // when pointer-locked in Walk mode the cursor is pinned; aim from the crosshair at screen centre
    const x = (mode === MODES.FPS && locked) ? innerWidth / 2 : e.clientX;
    const y = (mode === MODES.FPS && locked) ? innerHeight / 2 : e.clientY;
    const hit = tryArticulate(x, y);
    if (hit){ suppressWalkUntil = performance.now() + 400; showHint((hit.open ? 'Opening ' : 'Closing ') + hit.label); }
  });
}

function onClick(){
  if (busy || mode !== MODES.POINTER || dragged || !marker.visible) return;
  if (performance.now() < suppressWalkUntil) return;   // this click just opened a part — don't also walk
  goTo(marker.position.clone());
}

function goTo(point){
  const dest = new THREE.Vector3(point.x, ROOM.floorY + EYE, point.z);
  const away = new THREE.Vector3().subVectors(dest, camera.position);
  away.y = 0;
  if (away.length() > 0.8) dest.sub(away.normalize().multiplyScalar(0.8));   // stop short of the wall
  clampToRoom(dest);
  if (REDUCED){ camera.position.copy(dest); return; }
  const from = camera.position.clone(), t0 = performance.now();
  (function step(){
    const k = Math.min((performance.now() - t0) / TWEEN_MS, 1);
    camera.position.lerpVectors(from, dest, 1 - Math.pow(1 - k, 3));
    if (k < 1) requestAnimationFrame(step);
  })();
}

/* ------------------------------------------------------------------ keyboard walk */
const keys = { f:false, b:false, l:false, r:false };
const body = { velocity: new THREE.Vector3(), speed: 0,
               jumping:false, ducking:false, vy:0, jumpFrom:0 };
function resetKeys(){ keys.f = keys.b = keys.l = keys.r = false; body.velocity.set(0,0,0); body.speed = 0; }

function onKeyDown(e){
  if (e.target && /^(INPUT|TEXTAREA)$/.test(e.target.tagName)) return;
  if (mode === MODES.FPS && /^(Space|ControlLeft|ControlRight)$/.test(e.code)) e.preventDefault();
  switch (e.code){
    case 'ArrowUp': case 'KeyW': keys.f = true; break;
    case 'ArrowDown': case 'KeyS': keys.b = true; break;
    case 'ArrowLeft': case 'KeyA': keys.l = true; break;
    case 'ArrowRight': case 'KeyD': keys.r = true; break;
    case 'Space':
      if (mode === MODES.FPS && locked && !body.jumping){
        body.jumping = true; body.vy = MOVE.jump; body.jumpFrom = camera.position.y;
      }
      break;
    case 'ShiftLeft': case 'ShiftRight':
      if (mode === MODES.FPS && !body.jumping && !body.ducking){ body.ducking = true; body.vy = -MOVE.duckForce; }
      break;
    case 'Escape': if (mode === MODES.FPS) setMode(MODES.POINTER); break;
  }
}
function onKeyUp(e){
  switch (e.code){
    case 'ArrowUp': case 'KeyW': keys.f = false; break;
    case 'ArrowDown': case 'KeyS': keys.b = false; break;
    case 'ArrowLeft': case 'KeyA': keys.l = false; break;
    case 'ArrowRight': case 'KeyD': keys.r = false; break;
    case 'ShiftLeft': case 'ShiftRight': if (body.ducking) body.vy = MOVE.duckForce; break;
  }
}

const _dir = new THREE.Vector3(), _side = new THREE.Vector3(), _flat = new THREE.Vector3();
function stepMove(n){
  if (mode === MODES.ORBIT) return;
  _dir.set(0, 0, 0);
  if (keys.f || keys.b || keys.l || keys.r){
    if (keys.f || keys.b){
      _dir.setFromMatrixColumn(camera.matrix, 2); _dir.y = 0; _dir.normalize();
      if (keys.f) _dir.multiplyScalar(-1);
      if (keys.f && keys.b) _dir.set(0, 0, 0);
    }
    if (keys.l || keys.r){
      _side.setFromMatrixColumn(camera.matrix, 0); _side.y = 0; _side.normalize();
      if (keys.l) _side.multiplyScalar(-1);
      if (!(keys.l && keys.r)) _dir.add(_side);
    }
    if (_dir.lengthSq() > 0){
      _dir.normalize();
      body.speed = Math.min(body.speed + MOVE.accel * n, MOVE.maxSpeed);
      body.velocity.addScaledVector(_dir, body.speed * n);
    }
  } else {
    body.speed = Math.max(body.speed - MOVE.decel * n, 0);
  }
  body.velocity.multiplyScalar(Math.pow(MOVE.damp, n));

  const heldY = camera.position.y;
  _flat.copy(body.velocity); _flat.y = 0;
  camera.position.addScaledVector(_flat, n);
  clampToRoom(camera.position);

  const ground = ROOM.floorY + EYE;
  if (mode === MODES.FPS){
    if (body.jumping){
      body.vy -= MOVE.gravity * n;
      camera.position.y += body.vy * n;
      if (camera.position.y - body.jumpFrom >= MOVE.maxJumpHeight) body.vy = Math.min(body.vy, 0);
      if (camera.position.y <= ground){ camera.position.y = ground; body.jumping = false; body.vy = 0; }
    } else if (body.ducking){
      const floorLimit = ground - MOVE.duckDepth;
      camera.position.y += body.vy * n;
      if (camera.position.y <= floorLimit){ camera.position.y = floorLimit; body.vy = 0; }
      else if (camera.position.y >= ground){ camera.position.y = ground; body.ducking = false; body.vy = 0; }
    } else {
      camera.position.y = ground;
    }
  } else {
    camera.position.y = heldY;      // Point and Go keeps whatever height the glide left you at
  }
}

/* ------------------------------------------------------------------ touch */
const TOUCH = { startX:0, startY:0, dragging:false, tapAt:0, moveId:null, lookId:null,
                mx:0, my:0, lx:0, ly:0 };
const TOUCH_ROT = 0.004, TOUCH_THRESHOLD = 10;
const smooth = { factor:0.15, cur:new THREE.Vector2(), target:new THREE.Vector2() };

function attachTouch(){
  canvas.addEventListener('touchstart', onTouchStart, { passive:false });
  canvas.addEventListener('touchmove',  onTouchMove,  { passive:false });
  canvas.addEventListener('touchend',   onTouchEnd);
}
function detachTouch(){
  canvas.removeEventListener('touchstart', onTouchStart);
  canvas.removeEventListener('touchmove',  onTouchMove);
  canvas.removeEventListener('touchend',   onTouchEnd);
}
function onTouchStart(e){
  if (busy) return;
  e.preventDefault();
  if (mode === MODES.FPS){
    // left half of the screen drives movement, right half looks around
    for (const t of e.changedTouches){
      const left = t.clientX < innerWidth / 2;
      if (left && TOUCH.moveId === null){ TOUCH.moveId = t.identifier; TOUCH.mx = t.clientX; TOUCH.my = t.clientY; }
      else if (!left && TOUCH.lookId === null){ TOUCH.lookId = t.identifier; TOUCH.lx = t.clientX; TOUCH.ly = t.clientY; }
    }
  } else {
    const t = e.touches[0];
    TOUCH.startX = t.clientX; TOUCH.startY = t.clientY;
    TOUCH.dragging = false; TOUCH.tapAt = performance.now();
    castMarker(t.clientX, t.clientY);
  }
}
function onTouchMove(e){
  if (busy) return;
  e.preventDefault();
  if (mode === MODES.FPS){
    for (const t of e.touches){
      if (t.identifier === TOUCH.moveId){
        const dx = t.clientX - TOUCH.mx, dy = t.clientY - TOUCH.my;
        keys.f = dy < -TOUCH_THRESHOLD; keys.b = dy > TOUCH_THRESHOLD;
        keys.l = dx < -TOUCH_THRESHOLD; keys.r = dx > TOUCH_THRESHOLD;
        body.speed = MOVE.maxSpeed * Math.min(Math.hypot(dx, dy) / 50, 1);
      } else if (t.identifier === TOUCH.lookId){
        smooth.target.x -= (t.clientX - TOUCH.lx) * TOUCH_ROT;
        smooth.target.y -= (t.clientY - TOUCH.ly) * TOUCH_ROT;
        TOUCH.lx = t.clientX; TOUCH.ly = t.clientY;
      }
    }
    return;
  }
  const t = e.touches[0];
  const dx = t.clientX - TOUCH.startX, dy = t.clientY - TOUCH.startY;
  if (!TOUCH.dragging && (Math.abs(dx) > TOUCH_THRESHOLD || Math.abs(dy) > TOUCH_THRESHOLD)) TOUCH.dragging = true;
  if (TOUCH.dragging){
    const euler = new THREE.Euler(0, 0, 0, 'YXZ').setFromQuaternion(camera.quaternion);
    euler.y += dx * TOUCH_ROT;
    const next = pitch + dy * TOUCH_ROT;
    if (Math.abs(next) < PITCH_LIMIT){ pitch = next; euler.x = pitch; }
    camera.quaternion.setFromEuler(euler);
  } else {
    castMarker(t.clientX, t.clientY);
  }
  TOUCH.startX = t.clientX; TOUCH.startY = t.clientY;
}
function onTouchEnd(e){
  if (mode === MODES.FPS){
    for (const t of e.changedTouches){
      if (t.identifier === TOUCH.moveId){ TOUCH.moveId = null; keys.f = keys.b = keys.l = keys.r = false; }
      if (t.identifier === TOUCH.lookId) TOUCH.lookId = null;
    }
    return;
  }
  if (!TOUCH.dragging && performance.now() - TOUCH.tapAt < 300 && marker.visible) goTo(marker.position.clone());
  TOUCH.dragging = false;
}
function stepTouchLook(n){
  if (mode !== MODES.FPS) return;
  smooth.cur.x += (smooth.target.x - smooth.cur.x) * smooth.factor;
  smooth.cur.y += (smooth.target.y - smooth.cur.y) * smooth.factor;
  const euler = new THREE.Euler(0, 0, 0, 'YXZ').setFromQuaternion(camera.quaternion);
  euler.y += smooth.cur.x * n;
  const next = pitch + smooth.cur.y * n;
  if (Math.abs(next) < PITCH_LIMIT){ pitch = next; euler.x = pitch; }
  camera.quaternion.setFromEuler(euler);
  smooth.target.set(0, 0);
  smooth.cur.multiplyScalar(0.85);
}

/* ------------------------------------------------------------------ objects panel (orbit)
   Every top-level part sorted into three layers — Structure, Furniture, Fixtures. Each layer has a
   header you can collapse and an eye that hides/shows the whole type; each part can be hidden or
   isolated on its own. It only touches visibility in Orbit; the walking modes see the full room. */
const OBJS = { list: [], hidden: new Set(), iso: null, wire: false, collapsed: new Set() };

const CAT_COLOR = {
  table:'#e0a866', desk:'#e0a866', chair:'#5b8def', stool:'#5b8def', sofa:'#e07b9a', couch:'#e07b9a',
  bed:'#e6c25b', storage:'#54c08c', cabinet:'#54c08c', wardrobe:'#54c08c', shelf:'#54c08c', sink:'#54c08c',
  dishwasher:'#3fb6a8', fridge:'#3fb6a8', oven:'#3fb6a8', appliance:'#3fb6a8',
  door:'#ff7a3c', window:'#27c7e0', sash:'#27c7e0',
  whiteboard:'#8a7bd8', noticeboard:'#8a7bd8', board:'#8a7bd8', screen:'#8a7bd8', television:'#8a7bd8',
  panel:'#8a7bd8', radiator:'#c98b6b', trunking:'#b0b6be', conduit:'#b0b6be', socket:'#b0b6be',
  sensor:'#b0b6be', dispenser:'#b0b6be', wall:'#6b7480', floor:'#6b7480', ceiling:'#6b7480' };

/* Three layers, tested in order — Structure matches the shell by prefix, Furniture matches the
   free-standing pieces anywhere in the name, and anything left (doors, windows, sinks, appliances,
   boards, sockets, trunking …) falls through to Fixtures. */
const LAYERS = [
  { key:'structure', label:'Structure',
    re:/^(wall|floor|ceiling|room|shell|column|beam|stair|slab)/i },
  { key:'furniture', label:'Furniture',
    re:/(table|desk|chair|stool|sofa|couch|bench|bed|wardrobe|dresser|shelf|bookcase|storage|cabinet|drawer|nightstand|ottoman)/i },
  { key:'fixture',   label:'Fixtures', re:null },
];
const SKIP_RE = /^(daylight|_)/i;      // helpers the pipeline adds outside the walls, not real parts

function layerOf(name){
  for (const L of LAYERS) if (L.re && L.re.test(name)) return L.key;
  return 'fixture';
}
function catColor(name){
  const s = name.toLowerCase();
  for (const k in CAT_COLOR) if (s.includes(k)) return CAT_COLOR[k];
  return '#9aa7b0';
}
function isCeiling(name){ return /^ceiling/i.test(name); }
function fmtTris(n){ return n >= 1000 ? (n / 1000).toFixed(n >= 9500 ? 0 : 1) + 'k' : String(n); }
function layerItems(key){ return OBJS.list.filter(it => it.layer === key); }

function buildObjects(root){
  for (const node of topLevel(root)){
    const name = node.name || '';
    if (!name || SKIP_RE.test(name)) continue;
    let meshes = 0, tris = 0;
    node.traverse(o => {
      if (!o.isMesh || !o.geometry) return;
      meshes++;
      const g = o.geometry;
      tris += g.index ? g.index.count / 3 : (g.attributes.position ? g.attributes.position.count / 3 : 0);
    });
    if (!meshes) continue;
    OBJS.list.push({ node, name, tris: Math.round(tris), layer: layerOf(name) });
  }
  const order = { structure: 0, furniture: 1, fixture: 2 };
  OBJS.list.sort((a, b) => (order[a.layer] - order[b.layer])
                          || a.name.localeCompare(b.name, undefined, { numeric: true }));
  // dollhouse default: ceilings start hidden
  for (const it of OBJS.list) if (isCeiling(it.name)) OBJS.hidden.add(it.node);
  renderObjects();
}

function ceilingNodes(){ return OBJS.list.filter(it => isCeiling(it.name)).map(it => it.node); }

function applyVisibility(m = mode){
  const orbit = m === MODES.ORBIT;
  for (const it of OBJS.list){
    it.node.visible = !orbit ? true
                     : OBJS.iso ? (it.node === OBJS.iso)
                     : !OBJS.hidden.has(it.node);
  }
  syncRows();
}

function setWire(on){
  OBJS.wire = on;
  root_wire(on);
  document.getElementById('opWire').classList.toggle('on', on);
}
/* toggle wireframe on every material in the loaded scene (materials may be shared — harmless) */
function root_wire(on){
  scene.traverse(o => {
    if (!o.isMesh || o === marker || marker.children.includes(o)) return;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for (const mm of mats) if (mm) mm.wireframe = on;
  });
}

function renderObjects(){
  const body = document.getElementById('opBody');
  document.getElementById('opCount').textContent = OBJS.list.length + ' parts';
  const rowHTML = it => {
    const i = OBJS.list.indexOf(it);
    return `<div class="op-row" data-i="${i}"><span class="op-sw" style="background:${catColor(it.name)}"></span>`
         + `<span class="op-nm">${it.name}</span><span class="op-ct">${fmtTris(it.tris)}</span>`
         + `<span class="op-eye">◉</span></div>`;
  };
  let html = '';
  for (const L of LAYERS){
    const items = layerItems(L.key);
    if (!items.length) continue;
    const collapsed = OBJS.collapsed.has(L.key);
    html += `<div class="op-grp${collapsed ? ' collapsed' : ''}" data-g="${L.key}">`
          + `<span class="op-car">▾</span><span class="op-gname">${L.label}</span>`
          + `<span class="op-gct">${items.length}</span><span class="op-geye">◉</span></div>`;
    if (!collapsed) html += items.map(rowHTML).join('');
  }
  body.innerHTML = html;

  body.querySelectorAll('.op-row').forEach(el => {
    const it = OBJS.list[+el.dataset.i];
    el.querySelector('.op-eye').addEventListener('click', e => {
      e.stopPropagation();
      OBJS.iso = null;
      if (OBJS.hidden.has(it.node)) OBJS.hidden.delete(it.node); else OBJS.hidden.add(it.node);
      applyVisibility();
    });
    el.addEventListener('click', () => {           // isolate this part (click again to release)
      OBJS.iso = OBJS.iso === it.node ? null : it.node;
      applyVisibility();
    });
  });

  body.querySelectorAll('.op-grp').forEach(el => {
    const key = el.dataset.g;
    el.addEventListener('click', () => {           // collapse / expand this layer
      if (OBJS.collapsed.has(key)) OBJS.collapsed.delete(key); else OBJS.collapsed.add(key);
      renderObjects();
    });
    el.querySelector('.op-geye').addEventListener('click', e => {   // hide / show the whole type
      e.stopPropagation();
      OBJS.iso = null;
      const items = layerItems(key);
      const allHidden = items.every(it => OBJS.hidden.has(it.node));
      for (const it of items){ if (allHidden) OBJS.hidden.delete(it.node); else OBJS.hidden.add(it.node); }
      applyVisibility();
    });
  });
  syncRows();
}

function syncRows(){
  document.querySelectorAll('#opBody .op-row').forEach(el => {
    const it = OBJS.list[+el.dataset.i];
    const hiddenNow = OBJS.iso ? it.node !== OBJS.iso : OBJS.hidden.has(it.node);
    el.classList.toggle('off', hiddenNow);
    el.classList.toggle('iso', OBJS.iso === it.node);
    el.querySelector('.op-eye').textContent = hiddenNow ? '○' : '◉';
  });
  document.querySelectorAll('#opBody .op-grp').forEach(el => {
    const items = layerItems(el.dataset.g);
    const allHidden = items.length && items.every(it => OBJS.hidden.has(it.node));
    el.classList.toggle('ghidden', !!allHidden);
    el.querySelector('.op-geye').textContent = allHidden ? '○' : '◉';
  });
}

document.getElementById('opHead').addEventListener('click', () => objPanel.classList.toggle('collapsed'));
document.getElementById('opReset').addEventListener('click', () => {
  OBJS.iso = null; OBJS.hidden = new Set(ceilingNodes()); OBJS.collapsed = new Set();
  if (OBJS.wire) setWire(false);
  renderObjects(); applyVisibility();
});
document.getElementById('opWire').addEventListener('click', () => setWire(!OBJS.wire));

/* ------------------------------------------------------------------ camera views (orbit)
   A strip of the scene's capture viewpoints. Click one and the 3D camera flies to that exact pose
   (orbit stays usable from there), while a render-vs-real swipe of that frame drops in — the
   reconstruction and the original photo overlaid, drag to wipe between them. */
const CAMV = { list: (CFG.cameras || []), active: null, busy: false };
const camStrip = document.getElementById('camStrip');
const camRail  = document.getElementById('camRail');
const cmpCard  = document.getElementById('cmpCard');
const cmpImg   = document.getElementById('cmpImg');

function buildCameras(){
  if (!CAMV.list.length) return;
  camRail.innerHTML = CAMV.list.map((c, i) =>
    `<div class="cam-chip" data-i="${i}" style="background-image:url('${c.thumb}')"><span>${c.label}</span></div>`
  ).join('');
  camRail.querySelectorAll('.cam-chip').forEach(el =>
    el.addEventListener('click', () => activateCamera(+el.dataset.i)));
}

function activateCamera(i){
  const c = CAMV.list[i];
  if (!c || CAMV.busy || mode !== MODES.ORBIT) return;
  CAMV.active = i;
  camRail.querySelectorAll('.cam-chip').forEach((el, j) => el.classList.toggle('active', j === i));

  cmpImg.src = c.pair;
  document.getElementById('cmpCap').textContent = c.label + ' · render vs real';
  cmpCard.hidden = false;

  const pos = new THREE.Vector3().fromArray(c.pos);
  const fwd = new THREE.Vector3().fromArray(c.fwd);
  if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, -1);
  fwd.normalize();
  const target = pos.clone().addScaledVector(fwd, Math.max(1.5, pos.distanceTo(ROOM.center)));
  camera.up.set(0, 1, 0);
  camera.fov = c.fov || FOV_ORBIT; camera.updateProjectionMatrix();
  CAMV.busy = true; orbit.enabled = false;
  tweenTo(pos, pos.clone().add(fwd), () => {
    orbit.target.copy(target); orbit.enabled = true; orbit.update();
    CAMV.busy = false;
  });
}

function closeCamera(silent){
  const wasActive = CAMV.active !== null;
  CAMV.active = null;
  cmpCard.hidden = true;
  document.getElementById('cmpLightbox').hidden = true;
  camRail.querySelectorAll('.cam-chip').forEach(el => el.classList.remove('active'));
  if (!wasActive) return;
  camera.fov = FOV_ORBIT; camera.updateProjectionMatrix();
  if (silent || !VIEW.ORBIT){ orbit.target.copy(ROOM.center); return; }
  CAMV.busy = true; orbit.enabled = false;
  tweenTo(VIEW.ORBIT.pos, VIEW.ORBIT.target, () => {
    orbit.target.copy(VIEW.ORBIT.target); orbit.enabled = true; orbit.update();
    CAMV.busy = false;
  });
}

/* click the side-by-side image to see it full-size */
const cmpLightbox = document.getElementById('cmpLightbox');
cmpImg.addEventListener('click', () => {
  document.getElementById('cmpLightImg').src = cmpImg.src;
  cmpLightbox.hidden = false;
});
cmpLightbox.addEventListener('click', () => { cmpLightbox.hidden = true; });

document.getElementById('cmpPrev').addEventListener('click', () => CAMV.active !== null && activateCamera((CAMV.active - 1 + CAMV.list.length) % CAMV.list.length));
document.getElementById('cmpNext').addEventListener('click', () => CAMV.active !== null && activateCamera((CAMV.active + 1) % CAMV.list.length));
document.getElementById('cmpClose').addEventListener('click', () => closeCamera(false));

buildCameras();

/* ------------------------------------------------------------------ frame loop */
let prev = performance.now();
(function tick(){
  requestAnimationFrame(tick);
  const now = performance.now();
  const dt = Math.min((now - prev) / 1000, 0.05);       // real seconds, clamped
  const n = Math.min((now - prev) / (1000 / 60), 3);    // elapsed 60 Hz frames, clamped
  prev = now;
  stepTouchLook(n);
  stepMove(n);
  stepArticulation(dt);
  if (mode === MODES.ORBIT && orbit.enabled) orbit.update();
  (window.__vrDraw ? window.__vrDraw() : renderer.render(scene, camera));
})();


/* ===== side-by-side scan comparison (added by make_vr_pages.py) ===== */
(async () => {
  const CLOUD_URL = (window.SCENE && window.SCENE.cloud) || null;
  if (!CLOUD_URL) return;
  let PLYLoader;
  try { ({ PLYLoader } = await import('three/addons/loaders/PLYLoader.js')); }
  catch (e) { console.error('PLYLoader unavailable', e); return; }

  const host = document.querySelector('.mode-buttons');
  if (!host) return;
  const cmpBtn = document.createElement('button');
  cmpBtn.className = 'mode-button'; cmpBtn.style.marginTop = '4px'; cmpBtn.textContent = 'Compare ⇄';
  cmpBtn.title = 'Side by side: reconstruction | scan cloud, one synced camera';
  host.appendChild(cmpBtn);

  // density (point size) — shown only while comparing; default reads solid ("dense points")
  const dens = document.createElement('input');
  dens.type = 'range'; dens.min = '1'; dens.max = '100'; dens.value = '40';
  dens.title = 'Point size (density)';
  dens.style.cssText = 'width:120px;margin-top:8px;accent-color:#0b0d0e;display:none';
  host.appendChild(dens);

  let cloud = null, loading = false, split = false, baseSize = 0.01;
  const pointSize = () => baseSize * (+dens.value / 40);   // slider 40 = 1x

  async function ensureCloud() {
    if (cloud || loading) return;
    loading = true; cmpBtn.textContent = 'loading…';
    try {
      const geo = await new PLYLoader().loadAsync(CLOUD_URL);
      geo.computeBoundingBox();
      const s = geo.boundingBox.getSize(new THREE.Vector3());
      baseSize = (Math.max(s.x, s.y, s.z) || 3) * 0.004;   // dense-ish by default
      const hasColor = !!geo.attributes.color;
      cloud = new THREE.Points(geo, new THREE.PointsMaterial({
        size: pointSize(), sizeAttenuation: true,
        vertexColors: hasColor, color: hasColor ? 0xffffff : 0x9fd0d6 }));
      cloud.raycast = () => {};   // never intercept point-and-go / door clicks or collision rays
      cloud.visible = false;
      scene.add(cloud);
    } catch (e) { console.error('scan cloud load failed', e); }
    loading = false; cmpBtn.textContent = split ? 'Exit compare' : 'Compare ⇄';
  }
  dens.addEventListener('input', () => { if (cloud) cloud.material.size = pointSize(); });

  // pane labels for split
  const mkLabel = (txt, leftPct) => { const d = document.createElement('div'); d.textContent = txt;
    d.style.cssText = 'position:fixed;top:16px;left:' + leftPct + '%;transform:translateX(-50%);' +
      'background:rgba(255,255,255,.9);color:#0b0d0e;font:600 12px system-ui;padding:6px 14px;' +
      'border-radius:20px;z-index:1000;pointer-events:none;display:none;' +
      'box-shadow:0 2px 10px rgba(0,0,0,.12);backdrop-filter:blur(6px)';
    document.body.appendChild(d); return d; };
  const lblL = mkLabel('Reconstruction', 25), lblR = mkLabel('Scan cloud', 75);
  const findMode = re => [...document.querySelectorAll('.mode-button')].find(b => re.test(b.textContent.trim()));

  // center reticle for EACH pane (both panes look down the same camera axis, so a mark at 25% / 75%
  // is the same world point in the reconstruction and the scan — an alignment aid + the walk aim
  // point, which the built-in crosshair can't show because it sits on the seam at 50%).
  const mkMark = leftPct => { const d = document.createElement('div');
    d.style.cssText = 'position:fixed;left:' + leftPct + '%;top:50%;width:14px;height:14px;' +
      'margin:-7px 0 0 -7px;border-radius:50%;border:2px solid #fff;box-sizing:border-box;' +
      'box-shadow:0 0 0 1.5px rgba(0,0,0,.7),0 0 5px rgba(0,0,0,.5);z-index:1001;' +
      'pointer-events:none;display:none'; document.body.appendChild(d); return d; };
  const markL = mkMark(25), markR = mkMark(75);
  const divider = document.createElement('div');
  divider.style.cssText = 'position:fixed;left:50%;top:0;bottom:0;width:2px;margin-left:-1px;' +
    'background:rgba(255,255,255,.9);box-shadow:0 0 0 1px rgba(0,0,0,.4);z-index:1001;' +
    'pointer-events:none;display:none'; document.body.appendChild(divider);
  const appCross = document.querySelector('.crosshair');   // the built-in walk crosshair (at the seam)

  function setSplit(v){
    split = v;
    cmpBtn.classList.toggle('active', split);
    cmpBtn.textContent = split ? 'Exit compare' : 'Compare ⇄';
    dens.style.display = split ? 'block' : 'none';
    const show = split ? 'block' : 'none';
    lblL.style.display = lblR.style.display = markL.style.display = markR.style.display = divider.style.display = show;
    if (appCross) appCross.style.display = split ? 'none' : '';   // hide the seam crosshair while comparing
    if (split){ const o = findMode(/^Orbit$/); if (o) o.click(); }   // best as a synced dollhouse
    else {
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, innerWidth, innerHeight);
      if (cloud) cloud.visible = false;
      if (roomRoot) roomRoot.visible = true;
      camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    }
  }
  cmpBtn.addEventListener('click', async () => { await ensureCloud(); if (cloud) setSplit(!split); });

  // split render hook — reconstruction LEFT, scan cloud RIGHT, one synced camera
  window.__vrDraw = () => {
    if (!split || !cloud){
      renderer.setScissorTest(false);
      renderer.setViewport(0, 0, innerWidth, innerHeight);
      camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
      renderer.render(scene, camera); return;
    }
    const w = innerWidth, h = innerHeight, lw = Math.floor(w / 2), rw = w - lw;
    const scv = cloud.visible, srv = roomRoot ? roomRoot.visible : true;
    renderer.setScissorTest(true);
    cloud.visible = false; if (roomRoot) roomRoot.visible = true;                 // left = reconstruction
    renderer.setViewport(0, 0, lw, h); renderer.setScissor(0, 0, lw, h);
    camera.aspect = lw / h; camera.updateProjectionMatrix(); renderer.render(scene, camera);
    cloud.visible = true;  if (roomRoot) roomRoot.visible = false;                // right = scan cloud
    renderer.setViewport(lw, 0, rw, h); renderer.setScissor(lw, 0, rw, h);
    camera.aspect = rw / h; camera.updateProjectionMatrix(); renderer.render(scene, camera);
    cloud.visible = scv; if (roomRoot) roomRoot.visible = srv;
    renderer.setScissorTest(false);
  };

  // DEFAULT: open in the synced side-by-side, framed in Orbit so both rooms are visible at once.
  // Orbit / Point-and-Go / Walk stay one click away and keep working while comparing. ?compare=0 opts out.
  window.__vrDefault = () => {
    const o = findMode(/^Orbit$/); if (o) o.click();               // activate orbit immediately (interactive)
    if (new URLSearchParams(location.search).get('compare') !== '0') ensureCloud().then(() => setSplit(true));
  };
})();
