"""LOD 350 Mary Elizabeth Hospital — the WHOLE building, every plan step in one file.

This is the plan-level reference: a whole building, closed.

A radiating clinical building modelled on 3XN's BørneRiget in Copenhagen: five FINGERS
from one centre around a top-lit atrium, reading in plan as interlocked hands.

WHY THE PLAN SAYS TO BUILD IT ALL AND THEN LOOK. Almost everything here is
``placement=``d, because a pinwheel is one wing authored at bearing 0 and rotated five
times. A placed subtree leaves the inferred-carve pass, so NOTHING carves anything, and
therefore nothing reports a junction that is wrong. Zero warnings, zero carves and CSG 0
are all perfectly true of a building whose facade is buried inside its own slab.

So this file asserts its adjacencies in world coordinates, and the two it asserts
hardest are the two the plan records as having actually happened, each of them living in
the SEAM between two systems that were each correct alone:

  1. **The roof deck capped the atrium.** Massing owns the deck and has no void; the
     atrium system owns the void and had no deck. A top-lit atrium with a lid is a
     shaft, and it renders as a perfectly good roof.
  2. **Fire barriers stood 8250 mm proud of their own roof.** The barrier skill has no
     roof decks to be proud of.

``assert_the_atriums_are_open()`` and ``assert_nothing_stands_proud_of_its_roof()`` are
those two, made mechanical.

THE FOUR THINGS THIS FILE ADDS that the examples/ copy never had:

  1. **A roof.** There was none — the top storey's slab was its FLOOR and the building
     was open to the sky. Which is also why defect 1 above could not be reproduced from
     the examples/ copy: there was no deck to cap anything with.
  2. **Atriums, central and at every fingertip**, obeying the plan's three rules: the
     void has a ROOF that is open, it has a FLOOR (the ground plate is NOT bitten, or
     the atrium is open to the terrain and renders as grass indoors), and a plate with
     holes and a plate without are TWO Products, which is what makes the first two
     expressible at all.
  3. **Spaces and departments.** A declared ``Space`` bound is one of the few things
     that WILL report a piece in the wrong place in a model where nothing carves.
  4. **Clinical partitions** — the lead-lined imaging suite, at real BR/IEC wall
     thicknesses, in the service band where it belongs.

The cross-section closes exactly, and ``assert_the_section_closes()`` refuses the model
if it stops:

    2 x (ROOM_DEPTH + CORRIDOR_W) + BAND_W == ARM_WIDTH
    2 x (7800     + 2400      ) + 3600   == 24000

Change one and another must give, or the rooms stop meeting the facade.

All coordinates are int millimetres, Z-up; angles are centidegrees.
"""
# ─────────────────────────────────────────────────────────────────────────────
# `python examples/<file>.py` puts examples/ on sys.path, not the repo root, so
# `import lite_step` would bind to some other copy of the DSL and never say so.
# This block is why running a file directly works; `python -m lite_step <file>`
# does not need it.
# ─────────────────────────────────────────────────────────────────────────────
from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import math

from lite_step import compile_main
from lite_step.materials import MaterialDef, dk
from lite_step.materials.dk import Render, SheetGeometry
from lite_step.models import (
    Box,
    Column,
    Door,
    Element,
    Material,
    Mesh,
    Pipe,
    Point,
    Point2D,
    Product,
    Project,
    Revolve,
    Site,
    Slab,
    Space,
    Storey,
    Transform,
    Wall,
)

I = lambda x: int(round(x))  # noqa: E731,E741

# ── Materials Setup ────────────────────────────────────────────────────────
STRUCT_CONCRETE = "Concrete_C30-37"
STEEL = "Steel_S355"
ALUMINIUM = "Aluminium_6063_T6"
GLAZING = "Glass_VIG"
SPANDREL_PANEL = "Steel_S250GD_Z275"
GYPSUM = "Gypsum_Standard"
WOOD = "Timber_C24"

if STEEL not in dk.MATERIALS:
    dk.MATERIALS[STEEL] = MaterialDef(
        category="metal",
        function="Structure",
        psets={"Pset_MaterialCommon": {"MassDensity": 7850}},
        render=Render(rgb="#6F7377", alpha=1.0, finish="satin"),
        geometry=SheetGeometry(form="sheet", thicknesses_mm=[3, 5, 8]),
    )

if ALUMINIUM not in dk.MATERIALS:
    dk.MATERIALS[ALUMINIUM] = MaterialDef(
        category="metal",
        function="Finish",
        psets={"Pset_MaterialCommon": {"MassDensity": 2700}},
        render=Render(rgb="#B0B5B3", alpha=1.0, finish="satin"),
        geometry=SheetGeometry(form="sheet", thicknesses_mm=[3, 4, 6]),
    )

# ── Finger & Grid Geometry ─────────────────────────────────────────────────
WING_COUNT = 5
ARM_WIDTH = 24000
ARM_REACH = 60000
ARM_HALF = ARM_WIDTH // 2
CAP_X = ARM_REACH - ARM_HALF  # 48000
PITCH = 36000 // WING_COUNT  # 7200 centidegrees = 72.0 deg
STOREY_HEIGHT = 4200
SLAB_T = 275
WING_STOREYS = [6, 5, 4, 5, 4]
ATRIUM_RADIUS = 9000
#: A SECOND atrium on every fingertip, on the tip's own rounding circle — the one
#: part of this plan that is genuinely circular.
TIP_ATRIUM_RADIUS = 6000
#: The atrium void starts at the FIRST floor. The ground plate is what you stand on
#: to look up; bite it too and the atrium opens onto the terrain and renders as grass
#: indoors. This is why a plate with holes and a plate without are two Products.
ATRIUM_FIRST_LEVEL = 1
SITE_BUFFER = 30000
SITE_DEPTH = 15000

# ── Ward Cross-Section ─────────────────────────────────────────────────────
ROOM_DEPTH = 7800
CORRIDOR_W = 2400
BAND_W = 3600
MODULE = 4000
BAND_HALF = BAND_W // 2  # 1800
CORRIDOR_OUTER = BAND_HALF + CORRIDOR_W  # 4200
ROOM_OUTER = CORRIDOR_OUTER + ROOM_DEPTH  # 12000

ROOM_H = 2800
PARTITION_T = 120
DOOR_W, DOOR_H = 1200, 2100
WARD_X0 = 12000
ROOMS_PER_SIDE = (CAP_X - WARD_X0) // MODULE  # 9 rooms per side = 18 per finger

# ── Vertical Circulation ───────────────────────────────────────────────────
RISER = 175
GOING = 280
FLIGHT_W = 1500
WELL_W = 600
STEPS = STOREY_HEIGHT // (2 * RISER)  # 12
FLIGHT_RUN = STEPS * GOING  # 3360
LANDING_L = 1500
BAY_L = LANDING_L + FLIGHT_RUN + LANDING_L  # 6360
LANDING_T = 200
RAIL_H, RAIL_D = 1000, 40

STAIR_BAYS = (
    (13000, 13000 + BAY_L, BAND_HALF),
    (32000, 32000 + BAY_L, BAND_HALF),
)

# ── Curtain Wall Envelope ──────────────────────────────────────────────────
CW_MODULE = 1500
MULLION_D, MULLION_W = 180, 60
TRANSOM_H, SPANDREL_H = 60, 900


# ── 1. Site Block ──────────────────────────────────────────────────────────
def build_site_block():
    half = I(ARM_REACH + SITE_BUFFER)
    return Mesh(
        heightmap=[[0, 0], [0, 0]],
        depth=SITE_DEPTH,
        corner_min=Point2D(x=-half, y=-half),
        corner_max=Point2D(x=half, y=half),
        source_type="synthetic",
    )


# ── 0. The section, checked before any concrete ────────────────────────────
def assert_the_section_closes():
    """The arm's cross-section is a chain of clearances that has to add up.

    Change one and another must give, or the rooms stop meeting the facade — and
    nothing else in this model would say so, because every wing is ``placement=``d
    and a placed subtree carves nothing.
    """
    across = 2 * (ROOM_DEPTH + CORRIDOR_W) + BAND_W
    if across != ARM_WIDTH:
        raise AssertionError(
            f"the ward cross-section sums to {across} mm across a {ARM_WIDTH} mm arm.\n"
            f"  2 x (ROOM_DEPTH {ROOM_DEPTH} + CORRIDOR_W {CORRIDOR_W}) + BAND_W {BAND_W}"
            f" = {across}\n"
            f"{'rooms would overhang the facade' if across > ARM_WIDTH else 'a ' + str(ARM_WIDTH - across) + ' mm strip of nothing would run the length of every finger'}."
        )
    if ROOM_OUTER != ARM_HALF:
        raise AssertionError(
            f"the room's outer face is at y={ROOM_OUTER} but the arm's is at "
            f"y={ARM_HALF} — the curtain wall and the rooms are on different lines."
        )
    if STEPS * 2 * RISER != STOREY_HEIGHT:
        raise AssertionError(
            f"{STEPS} x 2 x {RISER} = {STEPS * 2 * RISER} does not climb a "
            f"{STOREY_HEIGHT} mm storey; a dogleg would arrive off the floor."
        )
    if ATRIUM_RADIUS >= ARM_HALF:
        raise AssertionError(
            f"a {ATRIUM_RADIUS} mm atrium radius is wider than the {ARM_HALF} mm arm "
            f"half-width — the void would cut the fingers off the centre."
        )
    if TIP_ATRIUM_RADIUS >= ARM_HALF:
        raise AssertionError(
            f"a {TIP_ATRIUM_RADIUS} mm tip atrium does not fit the {ARM_HALF} mm "
            f"rounding circle it sits on — the fingertip would be a ring, not a tip."
        )
    print(
        f"INFO section: {ARM_WIDTH} mm arm = 2 x (room {ROOM_DEPTH} + corridor "
        f"{CORRIDOR_W}) + service band {BAND_W}; storey {STOREY_HEIGHT} = {STEPS} x 2 x "
        f"{RISER}; atrium r={ATRIUM_RADIUS}, tip r={TIP_ATRIUM_RADIUS} inside a "
        f"{ARM_HALF} mm half-arm"
    )


#: Every atrium bite that was actually cut, as (plate tag, which void). Written by
#: `_atrium_bites`, read by `assert_the_atriums_are_open` — so the check is that the
#: code RAN for the roof, not that someone remembered to pass a flag.
ATRIUM_BITES: list = []


# ── 2. Structural Slabs ────────────────────────────────────────────────────
def _atrium_bites(slab, tag):
    """The central void and the fingertip void, as `.void()`s on one plate.

    Both are stated in the WING's local frame, and the central one is at local
    (0, 0) — which is the same point for all five wings, because the pinwheel is one
    arm authored at bearing 0 and rotated about the origin. One statement, one hole,
    five plates through it.

    `.void()` emits `IfcRelVoidsElement`, not a boolean, so the whole atrium costs
    nothing against the CSG budget of 10.
    """
    slab.void(
        Pipe(
            name=f"atrium_core_{tag}",
            radius=I(ATRIUM_RADIUS),
            path=[Point(x=0, y=0, z=-500), Point(x=0, y=0, z=I(SLAB_T + 500))],
        ),
        name=f"atrium_core_bite_{tag}",
    )
    slab.void(
        Pipe(
            name=f"atrium_tip_{tag}",
            radius=I(TIP_ATRIUM_RADIUS),
            path=[Point(x=I(CAP_X), y=0, z=-500), Point(x=I(CAP_X), y=0, z=I(SLAB_T + 500))],
        ),
        name=f"atrium_tip_bite_{tag}",
    )
    ATRIUM_BITES.append((tag, "core"))
    ATRIUM_BITES.append((tag, "tip"))


def build_finger_slab(name="ward_slab", ground=False, atrium=False):
    slab = Slab(name=name, props={"Pset_SlabCommon": {"LoadBearing": True}})
    straight = Box(
        name="body",
        material=STRUCT_CONCRETE,
        start=Point(x=0, y=I(-ARM_HALF), z=0),
        end=Point(x=I(CAP_X), y=I(ARM_HALF), z=SLAB_T),
    )
    tip = Revolve(
        name="body_tip",
        angle=36000,
        profile=[
            Point2D(x=0, y=0),
            Point2D(x=I(ARM_HALF), y=0),
            Point2D(x=I(ARM_HALF), y=SLAB_T),
            Point2D(x=0, y=SLAB_T),
        ],
        path=[Point(x=I(CAP_X), y=0, z=0), Point(x=I(CAP_X), y=0, z=SLAB_T)],
        material=Material(key=STRUCT_CONCRETE),
    )
    slab.add(straight.union(tip))

    if not ground:
        for i, (x0, x1, half_w) in enumerate(STAIR_BAYS):
            slab.void(
                Box(
                    name=f"stairwell_{i}",
                    start=Point(x=I(x0), y=I(-half_w), z=-500),
                    end=Point(x=I(x1), y=I(half_w), z=I(SLAB_T + 500)),
                ),
                name=f"stairwell_bite_{i}",
            )
    if atrium:
        _atrium_bites(slab, name)
    return slab


def build_roof_deck(name="ward_roof", atrium=True):
    """The deck that closes the top of a finger — WITH the atrium bites.

    This is the seam the plan records as having been got wrong: massing owns this
    deck and has no void; the atrium system owns the void and has no deck. A deck
    without the bites caps the void, and a top-lit atrium with a lid is a shaft. It
    is valid IFC, it breaks no rule, it warns about nothing, and it is obvious in the
    first render.
    """
    deck = build_finger_slab(name=name, ground=True, atrium=atrium)
    deck.props = {"Pset_SlabCommon": {"LoadBearing": True, "IsExternal": True}}
    return deck


# ── 3. Ward Room Module (Catalog Product) ──────────────────────────────────
def build_ward_room(name="ward_room", module=MODULE, depth=ROOM_DEPTH, t=PARTITION_T, height=ROOM_H):
    room = Element(ifc_class="IfcBuildingElementProxy", name=name)
    hx = module // 2
    y0 = CORRIDOR_OUTER
    y1 = ROOM_OUTER
    zb, zt = SLAB_T, I(SLAB_T + height)

    faces = [
        ("corridor", Point(x=I(-hx), y=I(y0), z=zb), Point(x=I(hx), y=I(y0 + t), z=zt)),
        ("facade", Point(x=I(-hx), y=I(y1 - t), z=zb), Point(x=I(hx), y=I(y1), z=zt)),
        ("west", Point(x=I(-hx), y=I(y0), z=zb), Point(x=I(-hx + t), y=I(y1), z=zt)),
        ("east", Point(x=I(hx - t), y=I(y0), z=zb), Point(x=I(hx), y=I(y1), z=zt)),
    ]
    walls = {}
    for tag, s, e in faces:
        w = Wall(
            name=f"{name}_{tag}",
            props={"Pset_WallCommon": {"LoadBearing": False, "IsExternal": tag == "facade"}},
        )
        w.add(Box(name="body", material=GYPSUM, start=s, end=e))
        room.add(w)
        walls[tag] = w

    door = Door(
        name="door_corridor",
        width=DOOR_W,
        height=DOOR_H,
        style="hinged",
        props={"Pset_DoorCommon": {"IsExternal": False}},
    )
    walls["corridor"].opening(door, along_center=I(module / 2), up=0)
    return room


# ── 4. Stair Flights & Landings (Catalog Products) ─────────────────────────
def build_flight(name="stair_flight"):
    flight = Element(
        ifc_class="IfcStairFlight",
        predefined_type="STRAIGHT",
        name=name,
        props={"Pset_StairFlightCommon": {"NumberOfRiser": STEPS, "RiserHeight": RISER, "TreadLength": GOING}},
    )
    hw = FLIGHT_W // 2
    for i in range(STEPS):
        flight.add(
            Box(
                name=f"step_{i:02d}",
                material=STRUCT_CONCRETE,
                start=Point(x=I(i * GOING), y=I(-hw), z=0),
                end=Point(x=I((i + 1) * GOING), y=I(hw), z=I((i + 1) * RISER)),
            )
        )
    for tag, y in (("left", -hw), ("right", hw)):
        rail = Element(ifc_class="IfcRailing", predefined_type="HANDRAIL", name=f"handrail_{tag}")
        rail.add(
            Pipe(
                name="body",
                radius=I(RAIL_D / 2),
                material=STEEL,
                path=[Point(x=0, y=I(y), z=RAIL_H), Point(x=I(FLIGHT_RUN), y=I(y), z=I(STEPS * RISER + RAIL_H))],
            )
        )
        flight.add(rail)
    return flight


def build_landing(length, width, name="landing"):
    lp = Element(ifc_class="IfcSlab", predefined_type="LANDING", name=name)
    lp.add(
        Box(
            name="body",
            material=STRUCT_CONCRETE,
            start=Point(x=0, y=I(-width / 2), z=I(-LANDING_T)),
            end=Point(x=I(length), y=I(width / 2), z=0),
        )
    )
    return lp


# ── 5. Unitized Curtain Wall Panel (Catalog Product) ───────────────────────
def build_curtain_wall_unit():
    panel = Element(ifc_class="IfcCurtainWall", name="cw_unit")
    w, h = CW_MODULE, STOREY_HEIGHT
    hw = w // 2

    # Vertical mullions
    m_left = Box(
        name="mullion_left",
        material=ALUMINIUM,
        start=Point(x=-hw, y=-MULLION_D // 2, z=0),
        end=Point(x=-hw + MULLION_W, y=MULLION_D // 2, z=h),
    )
    m_right = Box(
        name="mullion_right",
        material=ALUMINIUM,
        start=Point(x=hw - MULLION_W, y=-MULLION_D // 2, z=0),
        end=Point(x=hw, y=MULLION_D // 2, z=h),
    )

    # Transoms
    t_bottom = Box(
        name="transom_bottom",
        material=ALUMINIUM,
        start=Point(x=-hw + MULLION_W, y=-MULLION_D // 2, z=0),
        end=Point(x=hw - MULLION_W, y=MULLION_D // 2, z=TRANSOM_H),
    )
    t_mid = Box(
        name="transom_mid",
        material=ALUMINIUM,
        start=Point(x=-hw + MULLION_W, y=-MULLION_D // 2, z=SPANDREL_H),
        end=Point(x=hw - MULLION_W, y=MULLION_D // 2, z=SPANDREL_H + TRANSOM_H),
    )
    t_top = Box(
        name="transom_top",
        material=ALUMINIUM,
        start=Point(x=-hw + MULLION_W, y=-MULLION_D // 2, z=h - TRANSOM_H),
        end=Point(x=hw - MULLION_W, y=MULLION_D // 2, z=h),
    )

    # Spandrel panel
    spandrel = Box(
        name="spandrel_body",
        material=SPANDREL_PANEL,
        start=Point(x=-hw + MULLION_W, y=-10, z=TRANSOM_H),
        end=Point(x=hw - MULLION_W, y=10, z=SPANDREL_H),
    )

    # Vision glass unit
    vision_glass = Box(
        name="vision_glass",
        material=GLAZING,
        start=Point(x=-hw + MULLION_W, y=-6, z=SPANDREL_H + TRANSOM_H),
        end=Point(x=hw - MULLION_W, y=6, z=h - TRANSOM_H),
    )

    panel.add(m_left, m_right, t_bottom, t_mid, t_top, spandrel, vision_glass)
    return panel


# ── 6. Structural Column (Catalog Product) ─────────────────────────────────
def build_column():
    col = Column(name="column_500x500", props={"Pset_ColumnCommon": {"LoadBearing": True}})
    col.add(
        Box(
            name="body",
            material=STRUCT_CONCRETE,
            start=Point(x=-250, y=-250, z=0),
            end=Point(x=250, y=250, z=STOREY_HEIGHT),
        )
    )
    return col


# ── 9. Clinical partitions (systems/clinical-partition) ────────────────────
#: Imaging is the one clinical wall type whose THICKNESS is a shielding
#: calculation, not a stud choice. 2 mm of lead against the corridor and the
#: adjacent rooms, and the wall gets thick enough to say so.
LEAD_T = 2
LEAD_SHIELD_HEIGHT = 2134  # the plan's pinned variable — lining height, not wall height
IMAGING_WALL_T = 250
IMAGING_ROOM = (6000, 5400)  # along the arm x, across y — fits inside the service band


def build_imaging_suite(name="imaging_suite"):
    """A lead-lined imaging room, in the service band.

    The plan's whitelist has no MEP, so the scanner itself is not modellable. What
    IS modellable is the enclosure it needs, and that is the part that constrains
    the building: four walls at 250 mm carrying a 2 mm lead layer, sized to the band
    rather than to the ward module, because a scanner does not fit a ward module.
    """
    suite = Element(ifc_class="IfcBuildingElementProxy", name=name)
    lx, ly = IMAGING_ROOM
    hx, hy = lx // 2, ly // 2
    zb, zt = SLAB_T, I(SLAB_T + ROOM_H)
    walls = {}
    faces = (
        ("south", (-hx, -hy), (hx, -hy + IMAGING_WALL_T)),
        ("north", (-hx, hy - IMAGING_WALL_T), (hx, hy)),
        ("west", (-hx, -hy), (-hx + IMAGING_WALL_T, hy)),
        ("east", (hx - IMAGING_WALL_T, -hy), (hx, hy)),
    )
    for tag, (x0, y0), (x1, y1) in faces:
        w = Wall(
            name=f"{name}_{tag}",
            props={
                "Pset_WallCommon": {
                    "LoadBearing": False,
                    "IsExternal": False,
                    "FireRating": "EI60",
                    "LeadLiningThickness": LEAD_T,
                    "LeadLiningHeight": LEAD_SHIELD_HEIGHT,
                }
            },
        )
        w.add(
            Box(
                name="core",
                material=STRUCT_CONCRETE,
                start=Point(x=I(x0), y=I(y0), z=zb),
                end=Point(x=I(x1), y=I(y1), z=zt),
            )
        )
        suite.add(w)
        walls[tag] = w

    # The one door is a lead-lined leaf, cut into the corridor side. `along_center`
    # is in the HOST WALL's own frame, not world — the trap the plan calls out.
    walls["south"].opening(
        Door(
            name="door_imaging",
            width=DOOR_W,
            height=DOOR_H,
            style="hinged",
            props={"Pset_DoorCommon": {"IsExternal": False, "FireRating": "EI60"}},
        ),
        along_center=I(lx / 2),
        up=0,
    )
    print(
        f"INFO partition: imaging suite {lx} x {ly} mm, {IMAGING_WALL_T} mm walls with "
        f"{LEAD_T} mm lead to {LEAD_SHIELD_HEIGHT} mm — it fits the {BAND_W} mm service "
        f"band, not the {MODULE} mm ward module, which is the whole reason it is its own "
        f"system"
    )
    return suite


# ── 10. The two seam checks the plan records as having failed ──────────────
def assert_the_atriums_are_open(deck_tag, plate_tag):
    """A top-lit atrium with a lid is a shaft, and it renders as a perfectly good roof.

    Checked against what the BUILDER did rather than against a flag someone passed:
    `_atrium_bites` logs every hole it cuts, so this fails if the deck was built by a
    path that skips them — which is exactly how it failed the first time, when the
    deck came from the massing system and the void came from the atrium system and
    neither knew about the other.
    """
    cut = {t for t, _ in ATRIUM_BITES}
    for tag, what in ((plate_tag, "upper plate"), (deck_tag, "roof deck")):
        kinds = {k for t, k in ATRIUM_BITES if t == tag}
        if kinds != {"core", "tip"}:
            raise AssertionError(
                f"the {what} ('{tag}') was cut for {sorted(kinds) or 'nothing'}, not "
                f"for both atriums. Plates that were cut: {sorted(cut)}. A deck without "
                f"the bite caps the void — valid IFC, no warning, and the atrium is a "
                f"shaft."
            )
    print(
        f"INFO atrium: core r={ATRIUM_RADIUS} and tip r={TIP_ATRIUM_RADIUS} cut through "
        f"the upper plate AND the roof deck; the ground plate is left solid, so the void "
        f"has a floor to look up from"
    )


def assert_nothing_stands_proud_of_its_roof(tops, roof_top):
    """Four of five fire barriers once stood 8250 mm above their own roof.

    The barrier skill has no roof decks to be proud of, and the massing skill has no
    barriers, so neither could see it. Here both exist, so it is checkable: per wing,
    the highest thing placed against the highest thing covering it.
    """
    for wing, (what, top) in sorted(tops.items()):
        roof = roof_top[wing]
        if top > roof + 1:
            raise AssertionError(
                f"wing {wing}: '{what}' tops out at z={top} but its roof deck tops out "
                f"at z={roof} — {top - roof} mm of it stands in the open air above the "
                f"building it belongs to."
            )
    print(
        "INFO massing: nothing stands proud of its own roof -- "
        + ", ".join(
            f"w{w}: {what} {top} under deck {roof_top[w]}" for w, (what, top) in sorted(tops.items())
        )
    )


# ── Placement Helper ───────────────────────────────────────────────────────
#: wing -> (what, world top z) of the tallest thing placed in it. `place()` keeps it,
#: because after a Transform there is nothing left to ask: a placed subtree's own
#: `authored_aabb()` is in its own frame, and the compiler will not tell you either.
WING_TOPS: dict = {}


def place(product, storey, name, *, wing, z=0, local=(0, 0), heading=0.0, top=None):
    a = math.radians(wing * PITCH / 100.0)
    lx, ly = local
    holder = Element(ifc_class="IfcBuildingElementProxy", name=name)
    holder.add(product.occurrence(name="unit"))
    holder.placement = Transform(
        origin=Point(
            x=I(lx * math.cos(a) - ly * math.sin(a)),
            y=I(lx * math.sin(a) + ly * math.cos(a)),
            z=I(z),
        ),
        rotations=[("z", I((wing * PITCH / 100.0 + heading) * 100))],
    )
    storey.add(holder)
    if top is not None:
        world_top = I(z + top)
        if wing not in WING_TOPS or world_top > WING_TOPS[wing][1]:
            WING_TOPS[wing] = (name, world_top)
    return holder


def local_top(source) -> int:
    """The top of a catalog item in its OWN frame, before it is placed anywhere."""
    return I(source.authored_aabb().max.z)


def place_space(proj, storey, name, *, wing, x0, x1, half_w, z0, z1, kind="residential"):
    """A room declared by its BOUNDS, in a model where nothing carves.

    The plan is blunt about why this is worth the lines: a declared `Space` bound is
    one of the very few things that WILL report a piece in the wrong place here, and
    it caught two rooms sitting entirely off the floorplate in a model that was
    otherwise clean. Added `carve="none"` — an authored Space is a solid in the
    displacement pass, and one drawn to the room faces would carve the walls holding
    it up.
    """
    a = wing * PITCH / 100.0
    sp = Space(name=name, space_type=kind)
    sp.add(
        Box(
            start=Point(x=I(x0), y=I(-half_w), z=I(z0)),
            end=Point(x=I(x1), y=I(half_w), z=I(z1)),
        )
    )
    sp.placement = Transform(origin=Point(x=0, y=0, z=0), rotations=[("z", I(a * 100))])
    storey.add(sp, carve="none")
    return sp


def core_stations(bay, level_z):
    x0, _x1, _hw = bay
    base = level_z + SLAB_T
    y_off = I(WELL_W / 2 + FLIGHT_W / 2)
    turn_x = x0 + LANDING_L + FLIGHT_RUN
    return [
        ("flight", "flight_up", x0 + LANDING_L, -y_off, base, 0.0),
        ("half_landing", "half_landing", turn_x, 0, base + STEPS * RISER, 0.0),
        ("flight", "flight_return", turn_x, y_off, base + STEPS * RISER, 180.0),
        ("arrival", "arrival", x0 + LANDING_L, 0, base + STOREY_HEIGHT, 180.0),
    ]


# ── 7. Project Assembly ───────────────────────────────────────────────────
def generate_project():
    proj = Project(name="Mary Elizabeth Hospital (BørneRiget) — LOD 350")

    assert_the_section_closes()

    site = Site(name="site")
    terrain = Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN", name="terrain")
    terrain.add(build_site_block())
    site.add(terrain)
    proj.add(site)

    ground = Storey(name="ground", elevation=0)
    proj.add_storey(ground)

    # Register Products. THREE plates, not two: the ground plate is solid (it is the
    # atrium's floor), the upper plate is bitten, and the roof deck is bitten too.
    # A plate with holes and a plate without are two Products — that is what makes
    # "the void has a roof that is open AND a floor to stand on" expressible at all.
    ground_src = build_finger_slab(name="ward_slab_ground", ground=True)
    upper_src = build_finger_slab(name="ward_slab_upper", atrium=True)
    roof_src = build_roof_deck(name="ward_roof", atrium=True)
    assert_the_atriums_are_open(deck_tag="ward_roof", plate_tag="ward_slab_upper")

    GROUND_SLAB = Product(ground_src, name="ward_slab_ground")
    UPPER_SLAB = Product(upper_src, name="ward_slab_upper")
    ROOF_DECK = Product(roof_src, name="ward_roof")

    ward_src = build_ward_room()
    cw_src = build_curtain_wall_unit()
    col_src = build_column()
    imaging_src = build_imaging_suite()

    WARD_ROOM = Product(ward_src, name=f"ward_room_{MODULE}x{ROOM_DEPTH}")
    CW_UNIT = Product(cw_src, name="cw_unit_1500x4200")
    COLUMN = Product(col_src, name="col_500x500")
    IMAGING = Product(imaging_src, name=f"imaging_suite_{IMAGING_ROOM[0]}x{IMAGING_ROOM[1]}")

    TOP = {
        "slab": local_top(ground_src),
        "roof": local_top(roof_src),
        "room": local_top(ward_src),
        "cw": local_top(cw_src),
        "col": local_top(col_src),
        "imaging": local_top(imaging_src),
    }

    FLIGHT = Product(build_flight(), name=f"stair_flight_{FLIGHT_W}x{STEPS}")
    HALF_LAND = Product(build_landing(LANDING_L, 2 * BAND_HALF, "half_landing"), name=f"stair_landing_{LANDING_L}")
    ARRIVE_LAND = Product(build_landing(LANDING_L, 2 * BAND_HALF, "arrival_landing"), name=f"stair_arrival_{LANDING_L}")
    STAIR_PRODUCTS = {"flight": FLIGHT, "half_landing": HALF_LAND, "arrival": ARRIVE_LAND}

    # Room station coordinates along the arm
    room_stations = [
        (I(WARD_X0 + MODULE * (k + 0.5)), head)
        for k in range(ROOMS_PER_SIDE)
        for head in (0.0, 180.0)
    ]

    # Grid columns along the arm (at corridor lines y = ±1800, ±4200)
    col_x_grid = [10000, 18000, 26000, 34000, 42000]
    col_y_grid = [-CORRIDOR_OUTER, -BAND_HALF, BAND_HALF, CORRIDOR_OUTER]

    # Curtain wall perimeter stations
    cw_flank_stations = []
    # North & South straight flanks
    n_cw = (CAP_X - ATRIUM_RADIUS) // CW_MODULE
    for k in range(n_cw):
        x = ATRIUM_RADIUS + I((k + 0.5) * CW_MODULE)
        cw_flank_stations.append((x, ARM_HALF, 0.0))  # North flank (heading 0)
        cw_flank_stations.append((x, -ARM_HALF, 180.0))  # South flank (heading 180)

    # Place building components per wing & storey
    for w, storeys in enumerate(WING_STOREYS):
        for level in range(storeys):
            z = level * STOREY_HEIGHT

            # Slabs — level 0 is the atrium's FLOOR and is deliberately not bitten
            place(
                GROUND_SLAB if level < ATRIUM_FIRST_LEVEL else UPPER_SLAB,
                ground,
                f"slab_w{w}_l{level}",
                wing=w,
                z=z,
                top=TOP["slab"],
            )

            # Columns
            for ci, cx in enumerate(col_x_grid):
                for cj, cy in enumerate(col_y_grid):
                    place(COLUMN, ground, f"col_w{w}_l{level}_{ci}_{cj}", wing=w, z=z,
                          local=(cx, cy), top=TOP["col"])

            # Ward room modules
            for ri, (cx, head) in enumerate(room_stations):
                place(WARD_ROOM, ground, f"room_w{w}_l{level}_{ri:02d}", wing=w, z=z,
                      local=(cx, 0), heading=head, top=TOP["room"])

            # Curtain Wall Facade
            for f_i, (fx, fy, f_head) in enumerate(cw_flank_stations):
                place(CW_UNIT, ground, f"cw_w{w}_l{level}_{f_i:02d}", wing=w, z=z,
                      local=(fx, fy), heading=f_head, top=TOP["cw"])

            # The ward volume, declared by its bounds
            place_space(
                proj, ground, f"ward_w{w}_l{level}",
                wing=w, x0=WARD_X0, x1=CAP_X, half_w=ARM_HALF,
                z0=z + SLAB_T, z1=z + STOREY_HEIGHT,
            )

        # The roof deck closes the finger at the top of its own stack.
        roof_z = storeys * STOREY_HEIGHT
        place(ROOF_DECK, ground, f"roof_w{w}", wing=w, z=roof_z, top=TOP["roof"])

        # Imaging suite — one per finger, in the service band on the ground floor,
        # clear of the central atrium.
        place(IMAGING, ground, f"imaging_w{w}", wing=w, z=0,
              local=(ATRIUM_RADIUS + IMAGING_ROOM[0], 0), top=TOP["imaging"])

        # Dogleg Stair Cores
        for c, bay in enumerate(STAIR_BAYS):
            for level in range(storeys - 1):
                for kind, tag, x, y, sz, head in core_stations(bay, level * STOREY_HEIGHT):
                    place(
                        STAIR_PRODUCTS[kind],
                        ground,
                        f"stair_w{w}_c{c}_l{level}_{tag}",
                        wing=w,
                        z=sz,
                        local=(x, y),
                        heading=head,
                    )

    # The two atrium volumes, and the departments.
    for w, storeys in enumerate(WING_STOREYS):
        place_space(
            proj, ground, f"atrium_tip_w{w}",
            wing=w, x0=CAP_X - TIP_ATRIUM_RADIUS, x1=CAP_X + TIP_ATRIUM_RADIUS,
            half_w=TIP_ATRIUM_RADIUS,
            z0=SLAB_T, z1=storeys * STOREY_HEIGHT, kind="atrium",
        )
    place_space(
        proj, ground, "atrium_core",
        wing=0, x0=-ATRIUM_RADIUS, x1=ATRIUM_RADIUS, half_w=ATRIUM_RADIUS,
        z0=SLAB_T, z1=max(WING_STOREYS) * STOREY_HEIGHT, kind="atrium",
    )
    proj.zone(
        "inpatient",
        members=[
            f"space:ward_w{w}_l{level}:storey:ground"
            for w, storeys in enumerate(WING_STOREYS)
            for level in range(storeys)
        ],
    )
    proj.zone(
        "atriums",
        members=["space:atrium_core:storey:ground"]
        + [f"space:atrium_tip_w{w}:storey:ground" for w in range(WING_COUNT)],
    )

    roof_top = {w: I(s * STOREY_HEIGHT + TOP["roof"]) for w, s in enumerate(WING_STOREYS)}
    assert_nothing_stands_proud_of_its_roof(WING_TOPS, roof_top)
    print(
        f"INFO massing: {WING_COUNT} fingers, storeys {WING_STOREYS}, "
        f"{sum(WING_STOREYS)} floor plates + {WING_COUNT} roof decks"
    )
    return proj


result = generate_project()


if __name__ == "__main__":
    compile_main()
