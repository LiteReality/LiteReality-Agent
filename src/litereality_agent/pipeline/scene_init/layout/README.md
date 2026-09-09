# Layout

Make the room's layout sound *before* anything expensive is built from it.

```text
capture → scene extraction → merged boxes → LAYOUT → crops → DINO refinement → references
```

The window is deliberate, and it is the same one the box merge occupies. By the time an object has
been cropped, clustered and reconstructed, every mistake in its box has been paid for several times
over: a duplicate detection becomes two generated assets, a counter run recorded half a metre too
deep becomes an asset generated at the wrong extent and then squashed to fit, and a box that moves
*after* its crop was cut leaves a reference image describing geometry that no longer exists.

Settled here, on boxes, a correction is free.

The entry point is `stage.run_layout(scan)`, called from `scene_init/flow.py`. It never raises into
the caller: a layout pass that fails leaves the scan exactly as it found it and lets the run
continue unrepaired, because taking down an otherwise good reconstruction over a geometry edge case
is worse than the misplacement it was trying to fix.

## What it does

![before and after the layout pass](../../../../../assets/layout/layout-before-after.png)

A real scan, before and after. Two violations, both invisible in a violation count and obvious in a
picture: `Chair0` was recorded outside the room, poking through the wall on the right, and
`Sink_Storage0` sat off the wall it is installed against. Dashed outlines on the right show where
each box was.

## The four parts

**`check`** — the success condition, and what the repair optimises against. Nothing overlapping that
should not, nothing inside a wall, nothing off the floor plate, nothing floating or sunk. Pure
arithmetic: no Blender, no compiled glb.

**`repair`** — a search over legal *actions*: seat a unit against its wall, fit one into its corner,
separate two along the wall they share, trim a little off each of two that no longer fit. Every
action is applied to a copy and kept only if the room provably improves, so **a repair can never
return a worse room than it was given**.

It also never deletes. Dropping a duplicate detection is available behind `$LR_LAYOUT_DROP=1` and is
off by default, because a wrong deletion is the one outcome here that leaves no trace to notice.

**`report`** — every run draws what it did: the scanned room and the repaired one side by side, the
scanned positions ghosted underneath, each correction an arrow. A violation count cannot tell a
counter put back against its wall from a table shoved half a metre across the room — both read as
zero. The picture can, which is why it is on by default.

**`agent`** — where geometry runs out. Reads the reference photograph and answers the question a
person finds easy and a solver cannot: is this box the wrong *size*, does it belong in that corner,
are these two the same object detected twice. Nothing it proposes is trusted — every edit goes
through the same gate as everything else, so the model chooses what to try and the geometry decides
what is true. Off by default; it costs model calls.

## Upstream: the merge wall guard

The layout pass is downstream of `ingest/merge_boxes.py`, and some of what looks like a layout
problem is really a merge problem.

RoomPlan sometimes detects one unit twice and records the second copy far too deep — deep enough to
punch through the wall behind it. That smeared copy genuinely overlaps the counter run on its own
side *and* a unit on the far side, so union-find welds both runs into a single box spanning the
wall.

![the merge wall guard](../../../../../assets/layout/merge-wall-guard.png)

On the left, a cooktop belonging to the far-side run has disappeared into a counter run in the next
space: one crop, one reference image and one generated asset for two units in two different places.
No later stage can recover from that — the layout pass reads the result as a counter recorded too
deep, and every repair it can make is the wrong one.

The guard is a single rule: if the line between two boxes' centres crosses a wall **segment**,
refuse the edge. Not "which side of the wall's infinite line" — a room is full of walls whose
infinite lines cut everything in half, and the only question that matters is whether a wall
physically stands in the way. It can only ever *remove* a merge.

### Why that is a separate rule from the vertical one

`auto_groups` already defends against the same shape of bug one axis over, and it is worth seeing
why that defence cannot cover this case.

![stacked, not duplicated](../../../../../assets/layout/merge-stacking.png)

Seen from above these two boxes are almost the same box — 0.98 footprint overlap. In elevation they
are an oven with its wall cabinet above it, touching but distinct. The existing gate catches that by
refusing groups taller than a single unit.

It cannot catch the wall case, on two counts: the group there is no taller than one cabinet, and
every edge in it is a genuine 3D overlap rather than a weak abutment. So the wall guard is its
horizontal twin, applied before the strong/weak split rather than after clustering.

## Switches

| variable | default | effect |
| --- | --- | --- |
| `LR_LAYOUT` | `1` | `0` skips the pass entirely |
| `LR_LAYOUT_AGENT` | `0` | `1` lets the agent read reference photographs for what geometry cannot settle |
| `LR_LAYOUT_DROP` | `0` | `1` allows the pass to delete a duplicate detection |
| `LR_LAYOUT_VIZ` | `1` | `0` skips the before/after plan |
| `LR_BOX_MERGE_WALL_GUARD` | `1` | `0` turns off the wall guard above |

With the guard off, or with no wall geometry on disk, `merge_boxes` behaves exactly as it did before
the guard existed.

## Not included

This is not collision QC. Small box-on-box overlap is a physics concern, and a chair overlapping the
table it is tucked under is correct and left alone. What is settled here — duplicates, units buried
in walls, furniture outside the room, boxes recorded at the wrong size — is visible in any render
and expensive to discover later.

The plans above are drawn from `run/Kitchen-Xiaoyang_Lyu`; regenerate them from any scan's
`scene_data` with `adapter.shell_from_scene_data`.
