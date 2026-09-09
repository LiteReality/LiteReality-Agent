#!/usr/bin/env python3
"""Semantic QC probe for an articulated GLB — the real gate (beyond structure).

Where `validate_glb.py` checks that the GLB *parses* (nodes/animations/materials/extras),
this checks that the articulated object is *plausible*: parts are connected, nothing grossly
interpenetrates at rest, moving parts CLEAR the body when opened, joints are sane, and — the
big one — parts open toward the canonical FRONT (-Y). A joint that opens into +Y means the
whole object was built 180deg backwards; this replaces the prose CANONICAL_ORIENTATION plea
(and the hard-coded chair flip) with a gate.

Design: AABB-first (fast, pure stdlib — no Blender, no numpy). glTF POSITION accessors carry
min/max, so per-mesh local AABBs are free; we compose node world transforms and check AABBs
in world space. Approximate on purpose — HARD checks are the ones AABBs get right (connectivity,
joint sanity, open-direction), SOFT checks (rest overlap, swing clearance) are reported to guide
the agent but do not fail the gate, so we don't block legitimate objects on AABB coarseness.

Usage:
    python3 probe_glb.py <model.glb> [--json] [--tol 0.005]

Exit code 0 = pass (no HARD violations), 1 = HARD violations, 2 = bad file/usage.
With --json, prints a structured report {pass, checks[], violations[]} to stdout.
"""

from __future__ import annotations

import json
import math
import struct
import sys

TOL = 0.005  # metres — AABB touch/overlap tolerance (5 mm)
# Frame: the GLB is exported Y-up (export_yup=True), so in GLB space UP=+Y and the canonical
# FRONT (Blender -Y) maps to +Z. The articulation_axis EXTRA, however, is written raw in the
# BUILD (Blender Z-up) frame, so it must be converted before use (blender_to_gltf_axis).
FRONT_AXIS = 2  # GLB +Z is the canonical front
OPEN_TOL = 0.02  # metres of centroid travel toward the BACK before we call it "opens into body"
SMALL_PART_FRAC = 0.03  # a part < 3% of the biggest part's volume is attachment HARDWARE
#   (hinge / handle / knob / hook / latch) — it legitimately overlaps what it attaches to, so it
#   is exempt from the overlap/clearance SOFT flags. Flagging it wrongly pushes the agent to DROP
#   the detail (measured: the probe step suppressed door hinges), which is the opposite of the goal.


def blender_to_gltf_axis(a):
    """Blender Z-up (x, y, z) → glTF Y-up (x, z, -y), matching export_yup on the geometry."""
    return [a[0], a[2], -a[1]]


# ----------------------------- GLB / glTF parsing ----------------------------- #
def load_glb_json(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20:
        sys.exit(f"ERROR: not a GLB file: {path}")
    magic, _version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        sys.exit(f"ERROR: not a GLB file: {path}")
    clen, ctype = struct.unpack_from("<I4s", data, 12)
    if ctype != b"JSON":
        sys.exit("ERROR: first chunk is not JSON")
    return json.loads(data[20 : 20 + clen])


# ----------------------------- tiny 4x4 math (row-major) ---------------------- #
def mat_ident():
    return [1.0 if i % 5 == 0 else 0.0 for i in range(16)]


def mat_mul(a, b):
    r = [0.0] * 16
    for i in range(4):
        for j in range(4):
            r[i * 4 + j] = sum(a[i * 4 + k] * b[k * 4 + j] for k in range(4))
    return r


def mat_from_gltf_colmajor(c):
    # glTF node.matrix is column-major; convert to our row-major.
    return [c[j * 4 + i] for i in range(4) for j in range(4)]


def quat_to_mat3(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ]


def mat_from_trs(t, q, s):
    r3 = quat_to_mat3(*q)
    return [
        r3[0] * s[0], r3[1] * s[1], r3[2] * s[2], t[0],
        r3[3] * s[0], r3[4] * s[1], r3[5] * s[2], t[1],
        r3[6] * s[0], r3[7] * s[1], r3[8] * s[2], t[2],
        0.0, 0.0, 0.0, 1.0,
    ]


def node_local_matrix(node):
    if "matrix" in node:
        return mat_from_gltf_colmajor(node["matrix"])
    t = node.get("translation", [0.0, 0.0, 0.0])
    q = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    s = node.get("scale", [1.0, 1.0, 1.0])
    return mat_from_trs(t, q, s)


def xform_point(m, p):
    x, y, z = p
    return [
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
    ]


def rot3(m):  # rotation/scale part applied to a direction (no translation)
    def f(p):
        x, y, z = p
        return [m[0] * x + m[1] * y + m[2] * z,
                m[4] * x + m[5] * y + m[6] * z,
                m[8] * x + m[9] * y + m[10] * z]
    return f


def rodrigues(p, origin, axis, angle):
    """Rotate point p about the line through `origin` with unit direction `axis` by `angle` rad."""
    ax, ay, az = axis
    n = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
    ax, ay, az = ax / n, ay / n, az / n
    vx, vy, vz = p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]
    c, s = math.cos(angle), math.sin(angle)
    dot = ax * vx + ay * vy + az * vz
    cx, cy, cz = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx  # axis x v
    rx = vx * c + cx * s + ax * dot * (1 - c)
    ry = vy * c + cy * s + ay * dot * (1 - c)
    rz = vz * c + cz * s + az * dot * (1 - c)
    return [origin[0] + rx, origin[1] + ry, origin[2] + rz]


# ----------------------------- AABB helpers ----------------------------------- #
def corners(aabb):
    (x0, y0, z0), (x1, y1, z1) = aabb
    return [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]


def aabb_of(points):
    xs, ys, zs = zip(*points)
    return [[min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]]


def centroid(aabb):
    return [(aabb[0][i] + aabb[1][i]) / 2 for i in range(3)]


def overlap_extents(a, b, tol=0.0):
    return [min(a[1][i], b[1][i]) - max(a[0][i], b[0][i]) + tol for i in range(3)]


def aabbs_overlap(a, b, tol=0.0):
    return all(e > 0 for e in overlap_extents(a, b, tol))


def penetration_vol(a, b):
    e = overlap_extents(a, b, 0.0)
    return e[0] * e[1] * e[2] if all(x > 0 for x in e) else 0.0


def contained(inner, outer, tol):
    return all(inner[0][i] >= outer[0][i] - tol and inner[1][i] <= outer[1][i] + tol
               for i in range(3))


def embed_axes(a, b, tol):
    """How many axes have a's extent inside b's (or vice-versa). >=2 ⇒ a is inset/seated in b
    (a door leaf in its frame, a lift column in its base, a panel in a wall) — legitimate, not a
    cross-through. Full 3-axis containment is the strict case; 2 axes catches proud-in-depth insets."""
    fwd = sum(1 for i in range(3) if a[0][i] >= b[0][i] - tol and a[1][i] <= b[1][i] + tol)
    rev = sum(1 for i in range(3) if b[0][i] >= a[0][i] - tol and b[1][i] <= a[1][i] + tol)
    return max(fwd, rev)


def vol(aabb):
    return max(0.0, (aabb[1][0] - aabb[0][0])) * max(0.0, (aabb[1][1] - aabb[0][1])) * \
        max(0.0, (aabb[1][2] - aabb[0][2]))


# ----------------------------- scene extraction ------------------------------- #
def mesh_local_aabb(gltf, mesh_idx):
    """Union of POSITION accessor min/max over a mesh's primitives (local space)."""
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    got = False
    for prim in gltf.get("meshes", [])[mesh_idx].get("primitives", []):
        acc_i = prim.get("attributes", {}).get("POSITION")
        if acc_i is None:
            continue
        acc = gltf.get("accessors", [])[acc_i]
        mn, mx = acc.get("min"), acc.get("max")
        if not mn or not mx:
            continue
        got = True
        for i in range(3):
            lo[i] = min(lo[i], mn[i])
            hi[i] = max(hi[i], mx[i])
    return ([lo, hi] if got else None)


def collect_parts(gltf):
    """Walk the scene graph; return one entry per mesh-bearing node with world AABB + articulation."""
    nodes = gltf.get("nodes", [])
    scene = gltf.get("scene", 0)
    roots = gltf.get("scenes", [{}])[scene].get("nodes", []) if gltf.get("scenes") else []
    parts = []
    reachable = set()

    def walk(idx, parent_m):
        reachable.add(idx)
        node = nodes[idx]
        wm = mat_mul(parent_m, node_local_matrix(node))
        if "mesh" in node:
            local = mesh_local_aabb(gltf, node["mesh"])
            if local is not None:
                world = aabb_of([xform_point(wm, c) for c in corners(local)])
                ex = node.get("extras", {}) or {}
                parts.append({
                    "idx": idx,
                    "name": node.get("name", f"node{idx}"),
                    "aabb": world,
                    "wm": wm,
                    "origin": [wm[3], wm[7], wm[11]],
                    "art_type": ex.get("articulation_type"),
                    "axis": ex.get("articulation_axis"),
                    "lmin": ex.get("limit_min"),
                    "lmax": ex.get("limit_max"),
                })
        for ch in node.get("children", []):
            walk(ch, wm)

    for r in roots:
        walk(r, mat_ident())
    return parts, reachable, len(nodes)


def open_pose_aabb(part):
    """AABB of the part transformed to its OPEN pose, plus the centroid travel vector."""
    aabb, art = part["aabb"], part["art_type"]
    raw = part.get("axis") or [0, -1, 0]
    world_axis = blender_to_gltf_axis(raw)  # extra is in the build (Blender Z-up) frame
    lmax = float(part.get("lmax") or 0.0)
    lmin = float(part.get("lmin") or 0.0)
    c0 = centroid(aabb)
    if art == "prismatic":
        n = math.sqrt(sum(a * a for a in world_axis)) or 1.0
        d = [world_axis[i] / n * lmax for i in range(3)]
        opened = [[aabb[0][i] + d[i] for i in range(3)], [aabb[1][i] + d[i] for i in range(3)]]
    elif art == "revolute":
        pts = [rodrigues(c, part["origin"], world_axis, lmax) for c in corners(aabb)]
        opened = aabb_of(pts)
    else:
        return aabb, [0.0, 0.0, 0.0]
    travel = [centroid(opened)[i] - c0[i] for i in range(3)]
    _ = lmin
    return opened, travel


# ----------------------------- the checks ------------------------------------- #
def probe(path, tol=TOL):
    gltf = load_glb_json(path)
    parts, reachable, n_nodes = collect_parts(gltf)
    movable = [p for p in parts if p["art_type"] in ("revolute", "prismatic")]
    violations = []

    def add(sev, check, part, detail):
        violations.append({"severity": sev, "check": check, "part": part, "detail": detail})

    part_vol = {id(p): vol(p["aabb"]) for p in parts}
    max_vol = max(part_vol.values(), default=1.0) or 1.0

    def is_hardware(p):  # small attachment/decoration part (hinge/handle/knob/hook/latch)
        return part_vol[id(p)] < SMALL_PART_FRAC * max_vol

    # 1) CONNECTIVITY (HARD): every mesh node reachable from a scene root.
    all_node_idxs = {i for i, n in enumerate(gltf.get("nodes", [])) if "mesh" in n}
    for i in sorted(all_node_idxs - reachable):
        nm = gltf["nodes"][i].get("name", f"node{i}")
        add("hard", "connectivity", nm, "mesh node not reachable from the scene root (floating/orphaned)")

    # 1b) ISOLATED PART (SOFT): a part whose AABB doesn't touch any other part.
    for p in parts:
        if len(parts) > 1 and not any(q is not p and aabbs_overlap(p["aabb"], q["aabb"], tol)
                                      for q in parts):
            add("soft", "isolated_part", p["name"],
                "AABB has a gap from every other part — likely detached in space")

    # 2) ARTICULATION SANITY (HARD): declared movers must have a real joint.
    for p in movable:
        ax = p.get("axis") or []
        amag = math.sqrt(sum(a * a for a in ax)) if ax else 0.0
        lmin, lmax = float(p.get("lmin") or 0.0), float(p.get("lmax") or 0.0)
        if amag < 1e-6:
            add("hard", "articulation_sanity", p["name"], "articulation_axis is zero-length")
        if abs(lmax - lmin) < 1e-6:
            add("hard", "articulation_sanity", p["name"],
                f"degenerate joint limits (min={lmin}, max={lmax}) — part cannot move")

    # 3) ORIENTATION (HARD): parts must open toward the front (+Z in the GLB), not into the body.
    for p in movable:
        _, travel = open_pose_aabb(p)
        if travel[FRONT_AXIS] < -OPEN_TOL:
            add("hard", "orientation", p["name"],
                f"opens toward the BACK (-Z) by {-travel[FRONT_AXIS]:.3f} m — the object is likely "
                "built 180deg about the up-axis (parts should open toward the front, +Z)")

    # 4) REST OVERLAP (SOFT): non-containment interpenetration at the closed pose.
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            a, b = parts[i], parts[j]
            if not aabbs_overlap(a["aabb"], b["aabb"]):
                continue
            if embed_axes(a["aabb"], b["aabb"], tol) >= 2:
                continue  # inset/seated (drawer-in-carcass, leaf-in-frame, panel-in-wall) is legit
            if is_hardware(a) or is_hardware(b):
                continue  # a hinge/handle/knob overlapping what it attaches to is expected
            pv = penetration_vol(a["aabb"], b["aabb"])
            frac = pv / (min(vol(a["aabb"]), vol(b["aabb"])) or 1.0)
            if frac > 0.10:  # >10% of the smaller part's volume interpenetrates
                add("soft", "rest_overlap", f"{a['name']} / {b['name']}",
                    f"parts interpenetrate at rest ({frac * 100:.0f}% of the smaller part)")

    # 5) SWING/SLIDE CLEARANCE (SOFT): a mover collides with a static body when opened.
    static = [p for p in parts if p["art_type"] not in ("revolute", "prismatic")]
    for p in movable:
        opened, _ = open_pose_aabb(p)
        for s in static:
            if embed_axes(p["aabb"], s["aabb"], tol) >= 2 or is_hardware(s):
                continue  # seated in its own carcass, or small hardware — expected to touch
            rest_pen = penetration_vol(p["aabb"], s["aabb"])
            open_pen = penetration_vol(opened, s["aabb"])
            if open_pen > rest_pen + 1e-6 and open_pen > 0.05 * (vol(p["aabb"]) or 1.0):
                add("soft", "clearance", f"{p['name']} vs {s['name']}",
                    "moving part passes into the body when opened (swing/slide clearance risk)")

    hard = [v for v in violations if v["severity"] == "hard"]
    checks = [
        {"name": "connectivity", "detail": "all mesh parts reachable / connected"},
        {"name": "articulation_sanity", "detail": "movers have valid axis + non-degenerate limits"},
        {"name": "orientation", "detail": "parts open toward the front (-Y)"},
        {"name": "rest_overlap", "detail": "no gross interpenetration at rest (soft)"},
        {"name": "clearance", "detail": "movers clear the body when opened (soft)"},
    ]
    return {
        "file": path,
        "pass": len(hard) == 0,
        "n_parts": len(parts),
        "n_movable": len(movable),
        "n_nodes": n_nodes,
        "checks": checks,
        "violations": violations,
    }


def _fmt(report):
    lines = [f"probe: {report['file']}",
             f"  parts={report['n_parts']}  movable={report['n_movable']}  "
             f"=> {'PASS' if report['pass'] else 'FAIL'}"]
    if not report["violations"]:
        lines.append("  no issues")
    for v in report["violations"]:
        mark = "✗" if v["severity"] == "hard" else "!"
        lines.append(f"  {mark} [{v['severity']}] {v['check']}: {v['part']} — {v['detail']}")
    return "\n".join(lines)


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    as_json = "--json" in argv
    tol = TOL
    if "--tol" in argv:
        tol = float(argv[argv.index("--tol") + 1])
    if len(args) != 1:
        sys.exit(__doc__)
    report = probe(args[0], tol)
    print(json.dumps(report, indent=2) if as_json else _fmt(report))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
