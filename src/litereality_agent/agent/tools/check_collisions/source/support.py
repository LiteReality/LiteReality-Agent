"""support.py — nothing floats: what holds every object up, from the real meshes in `Room.glb`.

The collision check asks whether two things occupy the same volume. This asks the opposite question:
does every object *touch* the thing that holds it up. In an indoor scene there are only four
answers — the floor, a wall, the top of another object, or the ceiling — and an object that matches
none of them is floating in mid-air.

The support relation is DERIVED, never declared, so the scene graph falls out of the same pass: the
surface an object rests on IS its parent. That makes the last rule ("everything traces back to the
floor, a wall or the ceiling") a plain graph check rather than a separate piece of geometry.

WHY RAYS, NOT BOXES. An object's bounding-box top is not its support surface. A table's box top is
the tabletop, which is fine — but a chair's box top is the BACKREST, and nothing rests on a
backrest. Casting rays at the real triangles is what tells the two apart. The rays start from the
object's own extreme vertices rather than a grid over its footprint: a grid sends most of its rays
through the empty air between a chair's legs, whereas the extreme vertices ARE the four feet.

WHICH FACE PROBES WHICH SUPPORT. The bottom face only answers the DOWNWARD question. A picture on a
wall has nothing at all beneath it, and a pendant lamp's lowest vertex is its bulb — for those the
contact patch is the back face and the top face respectively. So each hypothesis probes the face
that points at its own candidate support:

    floor / on another object   bottom face, rays cast down       (which body is hit decides the parent)
    wall hanging                back face  vs the wall plane      (planes — no rays needed)
    ceiling hanging             top face   vs the ceiling plane

Walls and the ceiling are planes we already know from the SHELL, so those two are exact arithmetic
and cost nothing. Only the downward case needs rays, because only there is the identity of the
supporting body in question.

Everything is computed in the `Room.glb` frame, which is Y-up: height is y, the plan is (x, z), and
a SHELL point maps in as SHELL(x, y) -> glb(x, -y).

Consumes the same `bodies` dict as `collision_mesh.build_bodies`, so the two checks share one
decomposition of the room and cannot disagree about what an object is.
"""

from __future__ import annotations

import numpy as np

from .geometry import _expected

UP = np.array([0.0, 1.0, 0.0])   # Room.glb is Y-up
DOWN = -UP

# metres
GAP = 0.015        # a surface this close to the contact face counts as touching
EMBED = 0.015      # deeper than this into the support is embedded, not resting on
# Wall mounts get their own, much looser tolerance: a bracket, batten or backplate legitimately
# holds a thing off the wall. Measured on Elliott-Studio, a wall-hung TV and a wall cabinet both
# stand 6 cm proud, and at GAP they were both reported as floating in mid-air. Loosening this
# cannot swallow floor-standing furniture, because the DOWNWARD test runs first and wins.
WALL_GAP = 0.10
# Vertices within this of the extreme are the contact face. Not razor-thin on purpose: a mesh whose
# base is a millimetre out of level would otherwise offer a hairline strip of feet. Widening is
# safe — a candidate that turns out not to reach its support is dropped by its ray gap anyway.
FACE_BAND = 0.02
BACK_OFF = 0.10    # rays start this far back along the face normal, so an embedded face still
                   # sees the surface it is buried in (deeper than this is a placement error,
                   # not an embedding, and gets reported as such)
MAX_ORIGINS = 32   # rays per object — the feet of a chair are 4 points; 32 is generous
SPAN_TOL = 0.05    # fraction-free slack for "does this object lie along that wall's span"


# ---------------------------------------------------------------------------------------------
# contact faces and rays
# ---------------------------------------------------------------------------------------------

def _contact_face(mesh, direction, band: float = FACE_BAND, cap: int = MAX_ORIGINS):
    """The vertices on the mesh's extreme face along `direction` — its contact patch that way.

    `direction` is where the support is expected to be: DOWN gives the feet, UP the top, a wall's
    outward normal the back face. Downsampled evenly (and deterministically — same glb in, same
    rays out) so a 50k-triangle sofa costs the same as a stool.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    if len(V) == 0:
        return V
    d = np.asarray(direction, dtype=float)
    proj = V @ d
    keep = V[proj >= proj.max() - band]
    if len(keep) > cap:
        order = np.lexsort((keep[:, 2], keep[:, 1], keep[:, 0]))
        keep = keep[order][np.linspace(0, len(keep) - 1, cap).astype(int)]
    return keep


def _first_hits(origins, direction, targets: dict):
    """Cast one ray per origin and keep, PER RAY, the nearest surface across all targets.

    Nearest per ray is the point: a laptop on a table in a room hits both the tabletop (0 m) and the
    floor (0.75 m), and only the first one is holding it up. Rays start BACK_OFF behind the face so
    a face already buried in its support still sees that surface ahead of it — which is what makes
    the gap signed: positive = a gap (floating), ~0 = touching, negative = embedded.

    Returns (gap per ray, parent name per ray, hit point per ray); gap is +inf where nothing was hit.
    """
    n = len(origins)
    gaps = np.full(n, np.inf)
    parent: list[str | None] = [None] * n
    points = np.zeros((n, 3))
    if n == 0:
        return gaps, parent, points

    d = np.asarray(direction, dtype=float)
    start = origins - d * BACK_OFF
    dirs = np.tile(d, (n, 1))
    for name, mesh in targets.items():
        if mesh is None or len(mesh.faces) == 0:
            continue
        loc, idx_ray, _ = mesh.ray.intersects_location(start, dirs, multiple_hits=False)
        if len(loc) == 0:
            continue
        t = (loc - start[idx_ray]) @ d - BACK_OFF   # distance from the ORIGINAL face, signed
        for k, r in enumerate(idx_ray):
            if t[k] < gaps[r]:
                gaps[r], parent[r], points[r] = t[k], name, loc[k]
    return gaps, parent, points


def _plan_overlaps(a, b, slack: float = GAP) -> bool:
    """Do two meshes' footprints overlap in plan (glb x/z)? Cheap prefilter before any ray work."""
    amin, amax = a.bounds
    bmin, bmax = b.bounds
    return (amin[0] - slack <= bmax[0] and bmin[0] - slack <= amax[0]
            and amin[2] - slack <= bmax[2] and bmin[2] - slack <= amax[2])


# ---------------------------------------------------------------------------------------------
# wall / ceiling planes (no rays — we know these surfaces exactly from the SHELL)
# ---------------------------------------------------------------------------------------------

def _wall_planes(shell: dict):
    """SHELL walls as glb-plan planes: (wall_id, origin, inward normal, along unit, length)."""
    walls = shell.get("walls") or {}
    pts = [(p[0], -p[1]) for w in walls.values() for p in (w.get("start"), w.get("end")) if p]
    if not pts:
        return []
    fcx = sum(p[0] for p in pts) / len(pts)
    fcz = sum(p[1] for p in pts) / len(pts)
    out = []
    for wid, w in walls.items():
        s, e = w.get("start"), w.get("end")
        if not s or not e:
            continue
        sx, sz, ex, ez = s[0], -s[1], e[0], -e[1]
        dx, dz = ex - sx, ez - sz
        L = float(np.hypot(dx, dz))
        if L < 1e-6:
            continue
        ux, uz = dx / L, dz / L
        nx, nz = -uz, ux
        if (fcx - sx) * nx + (fcz - sz) * nz < 0:     # orient toward the room interior
            nx, nz = -nx, -nz
        out.append((wid, (sx, sz), (nx, nz), (ux, uz), L))
    return out


def _wall_support(mesh, planes):
    """The wall this object is mounted on, or None.

    Mounted means its BACK face sits on the wall plane: project every vertex onto the inward normal
    and take the minimum — that IS the back face, whatever the object's shape — then require it to
    be within touching distance of the plane, and the object to lie along the wall's span.
    """
    V = np.asarray(mesh.vertices, dtype=float)
    best = None
    for wid, (sx, sz), (nx, nz), (ux, uz), L in planes:
        rx, rz = V[:, 0] - sx, V[:, 2] - sz
        d = rx * nx + rz * nz                      # >0 interior, so the back face is the minimum
        t = (rx * ux + rz * uz) / L
        if not ((t >= -SPAN_TOL) & (t <= 1 + SPAN_TOL)).any():
            continue                               # object does not lie along this wall
        gap = float(d.min())
        if -EMBED <= gap <= WALL_GAP and (best is None or abs(gap) < abs(best[1])):
            best = (wid, gap)
    return best


# ---------------------------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------------------------

def _hull2d(pts):
    """Convex hull of plan points (monotone chain). Hand-rolled to keep scipy out of the deps."""
    P = sorted({(round(float(x), 6), round(float(z), 6)) for x, z in pts})
    if len(P) < 3:
        return P

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h

    return half(P)[:-1] + half(P[::-1])[:-1]


def _inside(hull, p, tol: float = 1e-9) -> bool:
    """Is a plan point inside the convex support polygon? Degenerate hulls (a line, a single
    contact point) are never 'inside' — an object balanced on one point is exactly the unstable
    case this is here to catch."""
    if len(hull) < 3:
        return False
    sign = 0
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(c) <= tol:
            continue
        s = 1 if c > 0 else -1
        if sign and s != sign:
            return False
        sign = s
    return True


# ---------------------------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------------------------

def find_supports(bodies: dict, shell: dict) -> dict:
    """What holds every object up → (scene graph, findings).

    Hypotheses are tried in the order they win ties in the real world: something standing ON the
    floor is floor-supported even though its back also touches a wall, so DOWN is tested first, then
    the wall, then the ceiling. An object that matches none of them is floating.

    Returns {'supports': {id: {...}}, 'graph': {child: parent}, 'findings': [...]}.
    """
    furniture = bodies.get("furniture") or {}
    floor = bodies.get("floor")
    planes = _wall_planes(shell)
    ceil_y = shell.get("ceiling_z")

    supports: dict[str, dict] = {}
    graph: dict[str, str] = {}
    findings: list[dict] = []
    embedded: list[tuple] = []   # (object, what it is buried in, depth) — resolved after the loop

    for oid, mesh in furniture.items():
        # --- 1. downward: the floor, or the top of another object -----------------------------
        feet = _contact_face(mesh, DOWN)
        face_y = float(feet[:, 1].min()) if len(feet) else float(mesh.bounds[0][1])
        targets = {}
        if floor is not None and _plan_overlaps(mesh, floor):
            targets["Floor"] = floor
        for other, om in furniture.items():
            # only bodies with geometry in the band the rays actually travel through
            if (other != oid and _plan_overlaps(mesh, om)
                    and om.bounds[1][1] >= face_y - GAP and om.bounds[0][1] <= face_y + BACK_OFF):
                targets[other] = om
        gaps, parent, points = _first_hits(feet, DOWN, targets)

        touching = [i for i, g in enumerate(gaps) if -EMBED <= g <= GAP]
        buried = [i for i, g in enumerate(gaps) if g < -EMBED]
        contact = touching + buried
        if contact:
            # An object may rest on SEVERAL bodies at once — a plank across two boxes, a tray
            # bridging two shelf levels. Taking only the body most rays hit and discarding the rest
            # would both name the wrong parent on a near-tie and, worse, judge stability against
            # half the contact patch: a plank supported at both ends has its centre in the gap
            # BETWEEN the two patches, and would read as balanced on nothing.
            names = [parent[i] for i in contact]
            per = {n: [i for i in contact if parent[i] == n] for n in set(names)}
            who = max(per, key=lambda n: (len(per[n]), -min(abs(gaps[i]) for i in per[n])))
            also = sorted(n for n in per if n != who)
            gap = float(min(gaps[i] for i in per[who]))
            supports[oid] = {"kind": "floor" if who == "Floor" else "on_object",
                             "parent": who, "gap_m": round(gap, 4), "also_on": also}
            graph[oid] = who

            for n, idx in per.items():
                deep = float(min(gaps[i] for i in idx))
                if deep < -EMBED:
                    embedded.append((oid, n, -deep))
            # Stability: the centre must fall inside everything it touches, taken together. Buried
            # rays count as contact here — a face sunk into its support is still HELD UP by it, and
            # its burial is already a finding of its own. Judging stability on the unburied rays
            # alone turns one defect into two: a built-in fridge whose base is partly over its
            # cabinet plinth would be called both embedded AND balanced on the strip beside it.
            if contact:
                hull = _hull2d([(points[i][0], points[i][2]) for i in contact])
                c = mesh.centroid
                if not _inside(hull, (c[0], c[2])):
                    on = " + ".join([who, *also])
                    findings.append({"id": oid, "kind": "unstable", "parent": who,
                                     "detail": f"centre falls outside the patch it rests on ({on})"})
            continue

        # --- 2. wall hanging: the BACK face against a wall plane ------------------------------
        # Deliberately not the bottom face — a picture on a wall has nothing at all beneath it.
        hit = _wall_support(mesh, planes)
        if hit:
            wid, gap = hit
            supports[oid] = {"kind": "wall", "parent": wid, "gap_m": round(gap, 4)}
            graph[oid] = wid
            if gap < -EMBED:
                embedded.append((oid, wid, -gap))
            continue

        # --- 3. ceiling hanging: the TOP face against the ceiling plane -----------------------
        if ceil_y is not None:
            gap = float(ceil_y) - float(mesh.bounds[1][1])
            if -EMBED <= gap <= GAP:
                supports[oid] = {"kind": "ceiling", "parent": "Ceiling", "gap_m": round(gap, 4)}
                graph[oid] = "Ceiling"
                continue

        # --- 4. nothing holds it up -----------------------------------------------------------
        floor_y = shell.get("floor_z")
        how_high = f"{face_y - float(floor_y):.2f} m above the floor" if floor_y is not None else "no support found"
        supports[oid] = {"kind": "floating", "parent": None, "gap_m": None}
        findings.append({"id": oid, "kind": "floating",
                         "detail": f"nothing under, behind or over it — {how_high}"})

    findings.extend(_embedded_findings(embedded, shell))
    findings.extend(_graph_issues(graph, set(furniture)))
    return {"supports": supports, "graph": graph, "findings": findings}


def _embedded_findings(embedded: list, shell: dict) -> list[dict]:
    """Turn raw burial measurements into findings — once per pair, and only where it is a defect.

    Two things have to be filtered out, both seen on the first real room this ran against:

    * A pair buried in each other is reported from BOTH sides (each one's contact face finds the
      other's geometry above it). It is one defect, so keep the deeper measurement and drop the
      mirror image.
    * Some burial is CORRECT. A built-in fridge sits inside its cabinet run, an undermount sink
      inside its counter — the same containment the clash check already allows by category, so it
      is allowed here from the same table rather than a second opinion that contradicts it.
    """
    cats = {i: (o.get("category") or "") for i, o in (shell.get("objects") or {}).items()}
    deepest: dict[frozenset, tuple] = {}
    for oid, host, depth in embedded:
        if _expected(cats.get(oid, ""), cats.get(host, "")):
            continue
        key = frozenset((oid, host))
        if key not in deepest or depth > deepest[key][2]:
            deepest[key] = (oid, host, depth)
    return [{"id": oid, "kind": "embedded", "parent": host,
             "detail": f"{depth:.3f} m into {host} — it should sit on it, not inside it"}
            for oid, host, depth in sorted(deepest.values(), key=lambda e: -e[2])]


def support_pairs(graph: dict, objects: set) -> set:
    """Support edges as unordered object pairs — the contacts the CLASH check should expect.

    Resting on something IS touching it, and FCL counts touching as contact, so a book on a shelf
    and a lamp on a table are reported as `object_clash` by `collision_mesh` today. The support
    graph answers that generically: if B holds A up, their contact is not a clash. That is the same
    job the hand-written `EXPECTED_CONTAINMENT` category table does for a chair under a table or a
    sink in its counter, except derived from the geometry instead of enumerated by category — so it
    covers pairs nobody thought to list.
    """
    return {frozenset((c, p)) for c, p in graph.items() if p in objects}


ROOTS = ("Floor", "Ceiling")


def _graph_issues(graph: dict, objects: set) -> list[dict]:
    """The last rule as a graph check: every chain ends at the floor, a wall or the ceiling.

    A parent outside `objects` is structure (Floor / Ceiling / a wall id) and terminates the chain.
    Anything else must climb to one, so the only two failures are a cycle (two objects resting on
    each other) and a chain that runs off into something with no support of its own.
    """
    out = []
    for oid in graph:
        seen, cur = [], oid
        while cur in graph:
            if cur in seen:
                out.append({"id": oid, "kind": "support_cycle",
                            "detail": "support chain loops: " + " → ".join(seen + [cur])})
                break
            seen.append(cur)
            cur = graph[cur]
        else:
            if cur in objects or not (cur.startswith(ROOTS) or cur.startswith("Wall")):
                out.append({"id": oid, "kind": "unsupported_chain",
                            "detail": f"chain ends at {cur}, which nothing holds up"})
    return out
