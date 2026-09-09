"""Per-RoomPlan-category procedural build specs — geometry + correct articulation.

RoomPlan only ever detects a FIXED set of object categories, so we can hand-author
a detailed spec per category: how to build it from primitives, what materials it
usually has, and — critically — exactly how its moving parts should ARTICULATE
(which part, joint type, hinge edge / slide axis, open direction, limit). The
procedural generator injects the matching spec into the agent prompt so the asset
both looks right and "comes alive" correctly (the dishwasher door drops down, the
drawer pulls out, the cabinet door swings on its outer edge).

Axes are in the object's own frame: X = width (left-right), Y = depth (front-back,
the FRONT faces -Y — the Blender Front view face), Z = height (up). Every object is
built in the same resting pose — see CANONICAL_ORIENTATION below.
"""

from __future__ import annotations

# Injected into EVERY generation prompt. Inconsistent orientation is the #1 cause
# of objects placed backwards in the scene (the repo even hard-codes a 180-deg
# chair flip to compensate), so this is mandatory and matches the skill's own
# Blender frame + its `--view front` render so the agent can self-verify.
CANONICAL_ORIENTATION = (
    "CANONICAL ORIENTATION — MANDATORY. Build the object in this exact resting pose "
    "(Blender Z-up metres; the glTF exporter handles the Y-up conversion). Do NOT bake any "
    "correcting rotation into the export — the GLB's local axes must BE this frame:\n"
    "- UP = +Z: stands upright, base/feet/bottom flat on the ground at Z=0 (lowest point Z=0). "
    "Never on its side, upside down, or tilted.\n"
    "- FRONT = -Y: the natural FRONT (the face a person faces and uses) points toward -Y — the "
    "face shown by the Blender Front view (the render script's `--view front`). Concretely: a "
    "cabinet / dresser / dishwasher / oven / fridge / washer has its DOORS, DRAWERS, HANDLES and "
    "CONTROLS on the -Y side; a TV/monitor has its SCREEN facing -Y; a chair / sofa / bench has its "
    "SEAT opening toward -Y (you sit facing -Y, the backrest is on the +Y side); a sink / stove "
    "shows its basin / cooktop / faucet toward -Y; a desk / table presents its main usable long "
    "side toward -Y; a bed's headboard is on the +Y side; a door / window shows its face (handle / "
    "operable side) toward -Y.\n"
    "- CENTRED: width runs along X centred at the origin, depth along Y; footprint centred at X=0, Y=0.\n"
    "- All articulation below is in THIS frame: doors swing open toward -Y, drawers and appliance "
    "doors pull/drop toward -Y. If a joint would open INTO the body, the front is on the wrong side "
    "(you are 180 deg about Z wrong).\n"
    "- VERIFY before exporting: the `--view front` render MUST show the real front (screen / doors / "
    "seat-front / handle). If it shows the BACK, rotate the whole object 180 deg about Z and rebuild "
    "— never deliver a reversed object."
)

# Each spec: geometry (primitive build), materials, articulation (moving parts +
# joints), and static (True if no moving parts by default).
SPECS: dict[str, dict] = {
    "storage": {
        "geometry": (
            "A rectangular carcass (box) with a thin recessed back panel and either a toe-kick "
            "plinth or short legs at the bottom. The front is divided into the EXACT layout of "
            "doors, drawers, and open shelves visible in the reference — count them and match their "
            "proportions and split lines. Add handles/knobs or recessed finger-pulls where shown."
        ),
        "materials": "Laminate or wood carcass; the fronts may be a different colour/material; metal or wood handles.",
        "articulation": (
            "For EACH hinged door: a REVOLUTE joint whose hinge axis is that door's OUTER VERTICAL "
            "edge (a left-side door hinges on its left edge, a right-side door on its right edge), "
            "opening outward, limit_min 0, limit_max ~100 degrees. "
            "For EACH drawer: a PRISMATIC joint sliding along -Y (straight out the front), travel "
            "≈ 0.9 x the drawer's depth. Open shelves and fixed panels stay static. "
            "Open-state preview: doors ~90 deg open, drawers pulled ~80% out."
        ),
        "static": False,
    },
    "dishwasher": {
        "geometry": (
            "A built-in rectangular box (standard ~600 mm wide). A full-height front door panel with "
            "a handle or a recessed grip near the top, a thin control strip along the top edge, and a "
            "toe-kick at the bottom. A shallow interior cavity is enough; optionally one wire rack."
        ),
        "materials": "Stainless-steel or white front; dark interior cavity.",
        "articulation": (
            "The front door is a REVOLUTE joint hinged on its BOTTOM HORIZONTAL front edge; it drops "
            "forward/down from vertical (closed) to horizontal (open), limit_min 0, limit_max ~90 deg. "
            "Optionally the lower rack is a PRISMATIC pull-out along -Y. Open-state: door down ~90 deg."
        ),
        "static": False,
    },
    "oven": {
        "geometry": "A rectangular appliance body with a front door that has a window, a handle bar across the top, and a control panel with knobs (built-in or under a cooktop).",
        "materials": "Stainless or black/white; dark glass door window.",
        "articulation": "The oven door is a REVOLUTE joint hinged on its BOTTOM front edge, dropping down from vertical to horizontal, limit ~90 deg. Open-state: door down.",
        "static": False,
    },
    "stove": {
        "geometry": "A range: a cooktop on top with burners/elements (static) over an oven box below with a windowed front door, handle, and control knobs.",
        "materials": "Stainless/enamel body; black glass cooktop or metal grates.",
        "articulation": "The oven door below is a REVOLUTE joint hinged on its BOTTOM front edge, opens down ~90 deg. Burners/knobs are static.",
        "static": False,
    },
    "refrigerator": {
        "geometry": "A tall rectangular body with the door split shown in the reference (single; side-by-side double; or fridge-over-freezer). Vertical handles, a toe grille at the bottom.",
        "materials": "Stainless, white, or black finish; subtle panel gaps.",
        "articulation": (
            "Each upper/main door is a REVOLUTE joint hinged on its OUTER VERTICAL edge, opening ~110 deg "
            "(a double-door fridge has two doors hinging on opposite outer edges). A bottom freezer is a "
            "PRISMATIC pull-out drawer along -Y (or a revolute door if shown). Open-state: door(s) ~90 deg."
        ),
        "static": False,
    },
    "washer": {
        "geometry": "A rectangular appliance body. Front-load: a round door on the front with a control panel and a dispenser drawer above. Top-load: a flat lid on top with a rear control console.",
        "materials": "White or grey body; glass or metal door.",
        "articulation": (
            "Front-load: the round door is a REVOLUTE joint hinged on one VERTICAL side edge, opening ~150 deg. "
            "Top-load: the lid is a REVOLUTE joint hinged on its REAR top edge, opening up ~80 deg. "
            "A detergent drawer is a PRISMATIC pull-out. Open-state: door/lid open."
        ),
        "static": False,
    },
    "toilet": {
        "geometry": "A ceramic bowl + tank (or one-piece). Seat and lid rings on top.",
        "materials": "White ceramic, matte or glossy.",
        "articulation": "The lid and the seat are each REVOLUTE joints hinged on their REAR edge, rotating up ~90 deg. Open-state: lid up.",
        "static": False,
    },
    "table": {
        "geometry": "A top (round/oval/rectangular — match the reference) of realistic thickness, on the EXACT support shown: single pedestal, 3–4 legs, trestle, or panel ends. Add an apron/stretchers if visible. Remove anything resting on top.",
        "materials": "Wood/laminate/glass top; wood or metal base. Match top vs base materials separately.",
        "articulation": "Static — no moving parts. (Exception: an obvious sit-stand desk frame is a PRISMATIC vertical lift of the top.) Render iso + front previews.",
        "static": True,
    },
    "desk": {
        "geometry": "A worktop slab on a leg/pedestal frame; optional drawer pedestal and modesty panel; cable grommets. Match the reference.",
        "materials": "Laminate/wood top; metal or wood frame.",
        "articulation": "Usually static. If it is clearly a sit-stand desk, the top is a PRISMATIC vertical lift. Drawer pedestals: PRISMATIC pull-out along -Y.",
        "static": True,
    },
    "sink": {
        "geometry": "One or two basins set in a counter slab or as a unit; the faucet/tap and any drainer board, only if part of the fixture. Match basin count and faucet form.",
        "materials": "Stainless steel or ceramic basin; metal faucet.",
        "articulation": "Static — no moving parts. (A faucet handle/spout swivel is optional and minor.) Render iso + front previews.",
        "static": True,
    },
    "counter": {
        "geometry": "A countertop slab of realistic thickness on supporting cabinets or legs that belong to the unit; match the edge profile.",
        "materials": "Stone/laminate/wood top; cabinet base below.",
        "articulation": "The countertop is static; any base cabinet doors/drawers articulate as in 'storage'.",
        "static": True,
    },
    "television": {
        "geometry": "A thin screen rectangle (match the aspect ratio) with a slim bezel, on either a central stand/feet or a wall mount. Remove any reflected room content on the screen.",
        "materials": "Black/glossy panel; dark bezel; metal/plastic stand.",
        "articulation": "Static — no moving parts. Render iso + front previews.",
        "static": True,
    },
    "sofa": {
        "geometry": "A base/frame with the EXACT seat-cushion count and back cushions, armrest shape, and either legs or a skirted base; match piping/seams.",
        "materials": "Fabric or leather upholstery; wood/metal legs.",
        "articulation": "Static (unless a clear recliner — then the footrest is a REVOLUTE extend and the back a REVOLUTE recline). Render iso + front previews.",
        "static": True,
    },
    "chair": {
        "geometry": "Backrest + seat + the EXACT base: 4 legs / sled / cantilever / 5-star caster base / pedestal; armrests if present.",
        "materials": "Upholstered or hard shell; metal/wood/plastic frame.",
        "articulation": "A simple chair is static. An office chair: the seat assembly is a PRISMATIC vertical gas-lift and a REVOLUTE swivel about the vertical axis; casters optional.",
        "static": True,
    },
    "bed": {
        "geometry": "A mattress + base block and a headboard (match shape/height); a footboard and frame/legs if shown. Keep the basic mattress form; remove loose bedding clutter.",
        "materials": "Upholstered or wood headboard; fabric mattress.",
        "articulation": "Static. (Storage-bed drawers in the base: PRISMATIC pull-out.) Render iso + front previews.",
        "static": True,
    },
    "bathtub": {
        "geometry": "A tub shell (rectangular/oval/freestanding) with an inner basin and a rim; a faucet only if part of the fixture.",
        "materials": "Acrylic/enamel; metal faucet.",
        "articulation": "Static. Render iso + front previews.",
        "static": True,
    },
    "fireplace": {
        "geometry": "A surround/mantel box with a firebox opening; a hearth slab. Match the surround profile.",
        "materials": "Stone/brick/metal surround; dark firebox.",
        "articulation": "Static (a glass door, if shown, is a REVOLUTE side hinge). Render iso + front previews.",
        "static": True,
    },
    "stairs": {
        "geometry": "A run of equal treads and risers (match the count) with stringers and any railing.",
        "materials": "Wood/concrete/metal.",
        "articulation": "Static. Render iso + front previews.",
        "static": True,
    },
    "door": {
        "geometry": (
            "A door LEAF (flush, panelled, or glazed — match the reference exactly) set inside a door "
            "FRAME/lining with architrave trim and a threshold at the bottom. Reproduce the real leaf "
            "design: panel/glazing layout and proportions, a vision panel or glass lights if shown, a "
            "handle/lever or knob at its real height and on the correct side, hinges on the opposite "
            "vertical side, and any kick plate, lock/escutcheon, or overhead door closer. Single or "
            "double leaf as shown. The leaf is thin (~40 mm) within the frame depth (the wall thickness)."
        ),
        "materials": "Painted/veneer/laminate wood leaf (match colour); metal lever/handle, hinges, and closer; clear or frosted glass for any vision panel.",
        "articulation": (
            "The door leaf is a REVOLUTE joint whose hinge axis is the leaf's HINGE-SIDE vertical edge "
            "(the side OPPOSITE the handle — read the hinge/handle sides from the reference), swinging "
            "open about that vertical edge, limit_min 0, limit_max ~90 degrees. The frame, architrave, "
            "and threshold stay STATIC. For a double door, two leaves hinge on opposite outer vertical "
            "edges. Open-state preview: leaf ~90 degrees open."
        ),
        "static": False,
    },
    "window": {
        "geometry": (
            "A window FRAME with the EXACT glazing layout from the reference — the number of sashes/"
            "lights, the mullions and glazing bars, and the real proportion of glass to frame — plus the "
            "SILL/ledge and surrounding frame. Render the glass as clean, empty, transparent glazing "
            "(no scenery or content behind it). Include any blind/curtain ONLY as it actually appears in "
            "the reference (same type and position), as a static element; if there is none, add none."
        ),
        "materials": "Wood / uPVC / metal frame (match colour); clean transparent glass; real blind/curtain material if one is present.",
        "articulation": (
            "Match the opening mechanism shown in the reference and animate ONLY the operable sash: "
            "casement/side-hung = REVOLUTE about one VERTICAL edge (~limit 60 deg); top-hung/awning = "
            "REVOLUTE about the TOP horizontal edge (~limit 30 deg); horizontal sliding = PRISMATIC along "
            "X with travel ~0.9 x sash width; vertical sash = PRISMATIC along Z. A fixed/picture window is "
            "STATIC. The frame, sill, mullions, and any blind/curtain stay static. Open-state preview: the "
            "operable sash opened per its mechanism (or iso + front if fixed)."
        ),
        "static": False,
    },
}

# Aliases -> canonical key.
ALIASES = {
    "cabinet": "storage",
    "cupboard": "storage",
    "shelf": "storage",
    "bookshelf": "storage",
    "wardrobe": "storage",
    "nightstand": "storage",
    "sideboard": "storage",
    "dresser": "storage",
    "washerdryer": "washer",
    "washer dryer": "washer",
    "dryer": "washer",
    "washing machine": "washer",
    "fridge": "refrigerator",
    "freezer": "refrigerator",
    "range": "stove",
    "cooktop": "stove",
    "microwave": "oven",
    "tv": "television",
    "monitor": "television",
    "couch": "sofa",
    "armchair": "chair",
    "stool": "chair",
    "bench": "table",
    "toilet bowl": "toilet",
    "wc": "toilet",
}

DEFAULT_SPEC = {
    "geometry": "Build it from simple primitives matching the reference's overall silhouette, parts, and proportions.",
    "materials": "Match the dominant materials and colours in the reference.",
    "articulation": "Add a REVOLUTE/PRISMATIC joint for any part that clearly opens, slides, or moves in the reference; otherwise static. Render iso + front previews if static.",
    "static": True,
}


def get_spec(category: str) -> dict:
    key = (category or "").strip().lower()
    key = ALIASES.get(key, key)
    return SPECS.get(key, DEFAULT_SPEC)
