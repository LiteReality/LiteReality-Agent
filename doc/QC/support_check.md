# Support check

Nothing floats. In an indoor environment every object rests on something, so once the room is built
each object has to be traced back to what holds it up.

There are only four ways an object can be supported:

1. **On the floor.** Standing on the ground.
2. **Wall hanging.** Mounted on a wall.
3. **On top of another object.** Sitting on a table, a shelf, a counter.
4. **Ceiling hanging.** Hung from the roof.

Rules:

- Every object has exactly one of these four supports.
- It touches its support — no gap, no floating.
- It is not embedded into its support either. It sits on it, not inside it.
- What holds it up must itself be supported, so everything traces back to the floor, a wall, or the
  ceiling.


We can also build a scene graph from the room.

# The deterministic function

For each object, cast rays straight down from its bottom face and see what they hit first.

- hits the floor at zero gap → on the floor
- hits another object at zero gap → on top of that object
- hits nothing, or the gap is too big → floating
- the hit is above the bottom face → embedded

Wall hanging is the same test with the rays going backwards into the wall, ceiling hanging with the
rays going up.

The first hit *is* the supporter, so the scene graph falls out of the same pass — one edge per
object, child → supporter. Then it is just a graph check: no cycles, and every chain ends at the
floor, a wall, or the ceiling.

Rays, not bounding boxes. An object's box top is not its support surface — a table's box top is the
tabletop, fine, but a chair's box top is the backrest, and nothing rests on a backrest. Rays hit the
real surface.

Two numbers to set: how big a gap still counts as touching, and how much penetration still counts as
sitting on rather than embedded. Both around a centimetre.

Walls need a third, looser one. A bracket or a batten legitimately holds something a few
centimetres off the wall — on Elliott-Studio a wall-hung TV and a wall cabinet both stand 6 cm
proud, and at the resting tolerance they read as floating in mid-air. Loosening it is safe because
the downward test runs first, so nothing standing on the floor can be captured by it.

One more thing the same rays give us for free: the object should not just touch its support, it
should be **stable on** it. Take the points where the rays land and check the object's centre sits
inside them. That catches the mug placed half off the edge of the table.

# The scene graph

Whatever the rays hit is the parent, so the graph comes out of the same pass:

```
Floor
├── Table0
│   ├── Book0
│   └── Cup0
└── Sofa0
Wall0
└── Shelf0
    └── Mug0
Ceiling
└── Lamp0
```

The shelf is the case that makes this worth having. The shelf hangs on the wall, the mug sits on
the shelf, so the mug's chain is `Mug0 → Shelf0 → Wall0`. Nothing in that chain ever touches the
floor and it is still correct — which is why the rule is "the chain ends at the floor, a wall or
the ceiling", not "the object is above the floor".

Two things fall out of having the graph:

- Objects can rest on **more than one** thing — a plank across two boxes, a tray bridging two shelf
  levels. Keep all of them, or the stability test looks at half the contact patch and calls a
  perfectly steady plank unstable.
- **Resting on something is touching it**, so the collision check would call a book on a shelf a
  clash. Every edge in this graph is contact we expect. The graph is the allow-list.
