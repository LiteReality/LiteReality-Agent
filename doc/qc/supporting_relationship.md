# Supporting relationship


So the two are a pair. Collision is about space that must be *empty*; support is about space that
must be *touched*.

We have the following entities:

1. **The floor.** The root of everything. It supports, and is supported by nothing.
2. **Floor-standing furniture.** Tables, chairs, sofas, storage, beds. Supported by the floor.
3. **Objects on top of other objects.** A lamp on a desk, a monitor on a table, a box on a shelf.
   Supported by the top face of whatever they sit on.
4. **Objects inside other objects.** A bowl in a cabinet, a book on a shelf inside a unit.
   Supported by the shelf, not by the floor — even though the floor is what carries the load.
5. **Wall and ceiling fixtures.** Whiteboards, radiators, sockets, pendant lights. *Anchored*, not
   supported: gravity is carried by the mount, so they are exempt from everything below except
   needing a mount that exists.

Rules:

- Every object has exactly one supporter, and the supporter is one of: the floor, the top face of
  another object, or a wall/ceiling anchor.
- Support follows the chain to the floor. No cycles, and no object supported by something that is
  itself unsupported.
- The contact gap is zero. Not a gap you can see, and not a penetration.

# The deterministic function

Same shape as the collision check: no model, reads only, one answer.

```python
check_support("…/room_preview/Room.glb")   ->  [findings]
```

```bash
python -m litereality_agent.pipeline.room_qc.support --glb <Room.glb>
# exit 0 = pass, 1 = fail
```

For each object it finds the supporter — the highest top face under the footprint, else the floor,
else the wall anchor — and measures the gap to it. The gap decides everything:

| finding | means | fixed by |
|---|---|---|
| `no_contact` / `embedded` | a supporter is there, the gap isn't zero | the snap |
| `unsupported` | nothing under the footprint at all | the agent |
| `orphan_anchor` | a wall fixture on no wall | the agent |
| `support_cycle` | A on B on A | nobody — the layout is wrong, report it |

`checks.py` today only measures against the **floor** (`FLOAT_TOL` 0.08 m, `SUNK_TOL` 0.05 m).
Nothing asks what an object sits *on*, which is why `floating` is reported and not failed.

# Two ways to fix

**Snap.** The supporter exists, so translate in Z by the gap. XY never moves — that is the collision
resolver's axis. A support chain moves root-first, so dropping a desk carries its monitor. Capped
around 0.15 m: past that it is the wrong supporter, not a bad contact, and it escalates.

**Author the missing support.** `unsupported` is not a placement error — the thing that holds it was
never reconstructed (a wall shelf, a plinth, a bracket). The agent gets the geometry already solved,
`top_z` and the footprint to cover, plus the reference frames that see that spot. It only decides
*what* the piece is, and writes it into `Room.py`. It never picks the height, or it would resolve a
floating monitor by dropping it to the floor. If the frames show nothing there, say so.

Then rebuild and run the gate again.
