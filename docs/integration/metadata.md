# What an asset must carry to be sim-ready

A `.glb` is geometry. A simulator additionally needs to know what each part weighs, how it
resists sliding, what shape to collide it as, and how its moving parts are hinged. Every
generated object carries that alongside its mesh, in a sidecar named `<Object>.physics.json`,
written by `models/object_generation/sim/properties.py`.

The asset carries its own physics so that every exporter is a pure function of the sidecar.
`to_urdf` and `to_mjcf` read it and add nothing of their own; a consumer that wants a third
format writes a third reader.

## The sidecar

`schema_version` is `1`. Lengths are metres, masses kilograms, and **everything is Z-up** —
the glTF Y-up frame is converted once on the way in, so no exporter converts again.

| field | meaning |
|---|---|
| `name` | object name, matching the `.glb` stem |
| `category` | token used to pick density and friction defaults, `""` if unrecognised |
| `root` | name of the root link |
| `links` | one per rigid part |
| `joints` | one per moving connection, empty for a rigid object |
| `bbox` | axis-aligned bounds of the whole object |
| `total_mass` | sum over links |
| `notes` | anything derivation had to assume, for a human reading a bad result |

### Links

| field | meaning |
|---|---|
| `name` | link name, matching its node in the `.glb` |
| `mass` | kilograms |
| `com` | centre of mass in the link frame |
| `inertia` | `[ixx, iyy, izz, ixy, ixz, iyz]` about the centre of mass |
| `friction`, `restitution` | contact parameters |
| `material` | the glTF material this part was assigned, if any |
| `visual` | mesh file for rendering, in the link frame |
| `colliders` | convex pieces, each a `file` and its `volume` |
| `mass_source` | `authored` if the build agent supplied it, else `derived` |
| `inertia_source` | how the inertia tensor was arrived at |

`mass_source` and `inertia_source` exist because authored values always win and derivation only
fills gaps — the field records which happened, so a wrong number can be traced to whoever
produced it.

### Joints

| field | meaning |
|---|---|
| `name` | joint name |
| `type` | `revolute`, `prismatic`, `continuous` or `fixed` |
| `parent`, `child` | link names |
| `origin` | the pivot, in the **parent** link frame |
| `axis` | axis of motion |
| `limit_lower`, `limit_upper` | travel limits |
| `damping`, `friction`, `effort`, `velocity` | actuation defaults |
| `origin_source` | how the pivot was located |

The pivot is stated explicitly rather than implied by link placement, so a door that opens
around the wrong edge is a wrong number in one field instead of a mystery in the geometry.

## Two things worth knowing before reading a number

**Colliders are convex.** Every engine here collides a mesh as its hull, so concavity that
matters — the space under a table, the inside of a cupboard — must be spelled out as several
convex pieces or it is silently filled in. That is what the collider list is for, and why a
single-piece collider on a hollow object is a bug rather than an optimisation.

**Density is per cubic metre of the bounding box, not of the mesh.** A real object is mostly
air: a chair is four thin legs and a thin seat, so material density over mesh volume puts it at
about 1.5 kg and it behaves like cardboard. The `OCCUPANCY_DENSITY` table is calibrated against
real furniture over its own bounding box — a chair is 23–27 kg/m³ whatever kind it is. The
total is then split across links by mesh volume, which is what gives a drawer front a
believable mass of its own.

## Not covered yet

How an object is fixed to a wall. A painting resting on a hook and a socket set into the wall
are different attachments, and neither is described here — a wall-mounted object currently
arrives as an unattached rigid body.
