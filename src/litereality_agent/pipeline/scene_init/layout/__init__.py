"""layout — make the room's LAYOUT sound before anything expensive is built from it.

Runs inside ingest, immediately after ``merge_boxes`` and before crops, references and
reconstruction. That window is deliberate and it is the same one the merge itself occupies: by the
time an object has been cropped, clustered and reconstructed, every mistake in its box has been
paid for several times over. A duplicate detection becomes two generated assets. A counter run
recorded half a metre too deep becomes an asset generated at the wrong extent and then squashed to
fit. A box that moves after its crop was cut leaves the reference image describing geometry that no
longer exists.

So the layout is settled first, on boxes, where a correction is free:

``check``     the success condition — nothing overlapping that should not, nothing inside a wall,
              nothing off the floor plate. Also what the repair optimises against.
``repair``    a search over legal ACTIONS (drop a duplicate, seat a unit against its wall, fit one
              into its corner, separate two along the wall they share, trim a little off each of
              two that no longer fit) where every action is applied to a copy and kept only if the
              room provably improves. A repair can therefore never return a worse room.
``agent``     where geometry runs out. Reads the reference photograph and answers the question a
              person would find easy and a solver cannot: is this box the wrong SIZE, does it
              belong in that corner, or are these two the same object detected twice.

This is not collision QC. Small box-on-box overlap is a physics concern; what is settled here —
duplicates, units buried in walls, furniture outside the room, boxes recorded at the wrong size —
is visible in any render and expensive to discover later.
"""

from .adjust import check
from .repair import repair, score
from .stage import run_layout

__all__ = ["check", "repair", "score", "run_layout"]
