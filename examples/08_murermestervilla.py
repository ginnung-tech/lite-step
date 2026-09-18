"""LOD 350 murermestervilla — the WHOLE 1.5-storey house, every plan step in one file.

This is the plan-level reference: a whole house, closed. Read it to copy the
numbers that only exist where the steps meet.

WHAT MAKES IT 1.5 STOREYS, and why every level below is derived rather than typed:

    z  ...   ridge                     wall_top + (hy + sandwich)·tan(pitch)
    z  ...   wall_top                  loft_level + LOFT_WALL_HEIGHT   (the *skunk*)
    z  ...   loft_level                floor_level + GROUND_STOREY_HEIGHT
    z  ...   floor_level               terrain at the origin + clearance

    GROUND_STOREY_HEIGHT = STAIR_RISERS x STAIR_RISER            <- the STAIR sets the storey
    LOFT_WALL_HEIGHT     = 700                                   <- and 700 is what puts a
                                                                    dormer cill and a gable
                                                                    cill ~1 m over the floor

A 1.5-storey house is not a 2-storey house with a roof on it. Its upper floor is the loft
INSIDE the roof, standing on the bjælkelag, with only ``LOFT_WALL_HEIGHT`` of masonry above
its own floor at the eaves. Every daylight decision follows from that one number: at 700 the
gable triangle starts 678 mm above the loft floor, so a gable window cills at ~930 — a window
you look out of, not at. Push the eaves wall to a full storey and the same window cills at
1550, which compiles identically and is a window you can only see sky through.

THE SIX THINGS THIS FILE ADDS that no single system example has:

  1. **A real ridge** (rygning) — an extruded clay ridge cover whose section is DERIVED from
     ``roof_pitch``: its two wings lie flat on the two tile fields, so changing the pitch
     re-cuts the ridge instead of leaving it hovering. The gable skill's ridge *cap* was a
     placeholder tent; this is the covering detail it stood in for.
  2. **Dormers** (kviste) — one per slope, with a HIPPED roof and 2-3 windows each, which is
     the only move in the whole house that adds usable loft floor area. The main roof is
     interrupted around each dormer by SPLITTING the courses, battens and underlay in X —
     no booleans — and closed at the back with a zinc gutter.
  3. **Gable windows** — cut straight through the gable sandwich with ``gable.opening()``.
     The gable is an ``Element(IfcWall)`` container, so one call distributes the hole to the
     brick, the insulation, the drywall AND every stud standing in it.
  4. **A stair** from the ground floor to the loft — quarter-turn, 17 risers, and the loft
     deck's stairwell derived from the HEADROOM rule rather than stated twice.
  5. **A ground floor that is planned, not drawn** — ``build_ground_plan`` derives the rooms,
     the partitions, the internal doors and the window positions from a room PROGRAM and a
     handful of rules, then proves the result is walkable (every room reachable from the
     front door) and livable (every room has daylight).
  6. **Internal doors** in those partitions, from the same three-leaf catalog as the front
     and garden doors.

All coordinates are int millimetres, Z-up. X = East and carries the RIDGE; Y = North and
carries the SPAN, so the gables land at ±X and the eaves at ±Y — the default massing the
DSL reference describes, which is what makes "the south facade" and "the gable window"
resolve to one wall each.
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
from lite_step.models import (Project, Storey, Space, Point, Point2D, Mesh, Site, Box, Slab,
                              Wall, Window, Door, Element, Extrude, Sweep, Sheet, Roof, Pipe,
                              Material, LayerSet, miter, Bar, Product)

I = lambda x: int(round(x))  # noqa: E731

# ══ FOOTPRINT ════════════════════════════════════════════════════════════════
FOUNDATION_FOOTPRINT_X = 12000   # mm, E–W OUTER footprint — the RIDGE direction; GABLES at ±hx
FOUNDATION_FOOTPRINT_Y = 9000    # mm, N–S OUTER footprint — the SPAN; EAVES at ±hy
SITE_BLOCK_BUFFER = 10000
SITE_BLOCK_DEPTH = 15000

# ══ FOUNDATION (systems/foundation, systems/stenkant) ════════════════════════
FOUNDATION_SLAB_THICKNESS = 200
FOUNDATION_TERRAIN_CLEARANCE = 50
FOUNDATION_CLEARING_HEIGHT = 20000
FOUNDATION_CONCRETE = "Concrete_C30-37"
FOUNDATION_STRIP_WIDTH = 400
FOUNDATION_STRIP_DEPTH = 900          # rules/frost-depth: ≥ 900 mm below the slab underside
FOUNDATION_STRIP_EDGE_INSET = 100
FOUNDATION_STRIP_GAP_MIN = 800
FOUNDATION_STRIP_GAP_MAX = 1400
STENKANT_WIDTH = 500
STENKANT_DEPTH = 500
STENKANT_SUBDIV = 8
STENKANT_CUT_OVERSHOOT = 400
STENKANT_GRAVEL = "Gravel_16-32"

# ══ THE 1.5-STOREY SECTION — every level below is DERIVED from these four ════
STAIR_RISER = 175            # stair_riser_height
STAIR_GOING = 260            # 2·175 + 260 = 610, inside the 600…640 comfort band
STAIR_RISERS = 17            # ⇒ a 2975 mm ground storey. See assert_the_section_closes().
STAIR_WIDTH = 1000           # clear flight width — a villa stair, not the 1500 escape stair
GROUND_STOREY_HEIGHT = STAIR_RISERS * STAIR_RISER          # 2975
LOFT_WALL_HEIGHT = 700       # the *skunk*: masonry above the LOFT floor at the eaves
STAIR_HEADROOM = 2100        # mm clear over any walking surface (BR18 minimum)
STAIR_LOWER_STEPS = 9        # 9 up, quarter landing, 8 up
STAIR_LANDING = STAIR_WIDTH  # a quarter landing is square, one flight wide
STAIR_STEP_NOSING = 25
HALL_CLEAR = 300             # clear zone the hall keeps around the stair on each side

WALL_CORE_THICKNESS = 180
STRUCT_CONCRETE = "Concrete_C30-37"

# ══ EXTERIOR WALL SANDWICH (systems/exterior-wall) ═══════════════════════════
EXT_BRICK_MM = 108           # outermost — inset 0 is its OUTER face
EXT_CAVITY_MM = 30
EXT_INSULATION_MM = 190
EXT_SANDWICH_MM = EXT_BRICK_MM + EXT_CAVITY_MM + EXT_INSULATION_MM      # 328

# ══ ROOF FRAME (systems/roof) ════════════════════════════════════════════════
ROOF_PITCH_DEG = 45
RAFTER_SPACING = 1000
TIMBER = "Timber_C24"
GLULAM = "Glulam_GL28h"
FLOOR_BEAM_GLULAM = "Glulam_GL24h"
RAFTER_PROFILE = (45, 195)
COLLAR_PROFILE = (45, 145)
PLATE_PROFILE = (195, 45)
POST_PROFILE = (45, 145)
RIDGE_BEAM_PROFILE = (140, 360)
FLOOR_BEAM_PROFILE = (140, 360)
BEAM_EMBED = 50
BEAM_MAX_FREE_SPAN = 8000
INNER_WALL_OFFSET = 3000
LOFT_HEADROOM = 2400         # the collar sits where the loft gets this much flat ceiling

# ══ ROOF OVERHANG (systems/gable-detail) ═════════════════════════════════════
EAVE_OVERHANG = 80
VERGE_OVERHANG = 80
EAVE_HALF_Y = FOUNDATION_FOOTPRINT_Y // 2 + EXT_SANDWICH_MM + EAVE_OVERHANG    # 4908
VERGE_HALF_X = FOUNDATION_FOOTPRINT_X // 2 + EXT_SANDWICH_MM + VERGE_OVERHANG  # 6408
GABLE_HALF_Y = FOUNDATION_FOOTPRINT_Y // 2 + EXT_SANDWICH_MM                   # 4828

# ══ ROOF COVERING (systems/roof-covering) ════════════════════════════════════
TILE = "ClayTile_Black_Engobed"
TILE_COVER_WIDTH = 300
TILE_GAUGE = 330
TILE_HEAD_LAP = 90
TILE_LAP_STEP = 18
TILE_MIN_PITCH_DEG = 25      # DS/EN 1304 — the registry carries this as min_roof_pitch_centideg
BATTEN = "Timber_C24"
BATTEN_W = 38
BATTEN_H = 25
BATTEN_VERGE_INSET = 120
UNDERLAY = "Underlay_Membrane"
UNDERLAY_T = 2
ROOF_INSULATION = "MineralWool_Roof38"
ROOF_INSULATION_T = 180

ROOF_TILE_PROFILE = [
    (0, 40), (19, 45), (37, 49), (55, 52), (73, 53), (90, 54), (106, 53),
    (121, 51), (135, 48), (148, 43), (160, 37),
    (170, 30),
    (176, 21), (185, 14), (197, 9), (212, 6),
    (227, 5),
    (243, 6), (258, 8), (273, 13), (285, 20), (294, 29), (300, 40),
    (300, 55), (294, 44), (285, 35), (273, 28), (258, 23), (243, 21),
    (227, 20),
    (212, 21), (197, 24), (185, 29), (176, 36),
    (170, 45),
    (160, 52), (148, 58), (135, 63), (121, 66), (106, 68),
    (90, 69),
    (73, 68), (55, 67), (37, 64), (19, 60), (0, 55),
]

# ── The RIDGE (rygning). Its section is derived from the pitch, not drawn. ────
RIDGE_WING = 190             # mm each wing reaches DOWN THE SLOPE from the apex
RIDGE_RISE = 60              # mm the crown stands above where the two roof planes cross
RIDGE_SHELL = 20             # mm the cover's own gauge at its wing tips
RIDGE_BED = 40               # mm the top course stops short of the apex, for the cover to bed on

# ══ GABLE ENDS + CORBELS (systems/gable-detail) ══════════════════════════════
GABLE_BRICK = "Brick_Red_DK"
GABLE_BRICK_MM = EXT_BRICK_MM
GABLE_CAVITY_MM = EXT_CAVITY_MM
GABLE_INSUL = "MineralWool_Facade34"
GABLE_INSUL_MM = EXT_INSULATION_MM
GABLE_DRYWALL = "Gypsum_Standard"
GABLE_DRYWALL_MM = 25
GABLE_STUD_PROFILE = (45, 195)
GABLE_STUD_SPACING = 600
GABLE_APEX_CLAMP = 140
CORBEL_BRICK = "Brick_Red_DK"
CORBEL_COURSE_H = 62
CORBEL_PROUDS = (40, 70, 100)

# ══ DORMERS (systems/dormer) ═════════════════════════════════════════════════
DORMER_SETBACK = 0           # mm the dormer front face sits BEHIND the brick face. ZERO is
                             # the murermestervilla move — the kvist springs straight off the
                             # facade line. Set it back and the roof between the eave and the
                             # dormer front has to be tiled, cut and flashed for a strip a few
                             # hundred mm deep; at 0 the eave overhang passes in front of the
                             # dormer's foot and one apron closes the joint.
DORMER_BASE_DROP = 300       # mm the front wall is buried below the roof surface at its foot
DORMER_APRON = 250           # mm from the roof surface at the front face up to the cill
DORMER_HEAD = 250            # mm from the window head up to the front wall top
DORMER_PIER = 200            # mm of front wall between (and outside) the windows
DORMER_ZINC_T = 1            # the standing-seam skin, the only Zinc_Titanium stock gauge
DORMER_LINING_T = 13         # Gypsum_Standard, the only stock
DORMER_INSUL = "MineralWool_Facade34"
DORMER_INSUL_T = 190         # stock: 125 / 150 / 190 / 220 / 250 / 300
# The front wall is its LAYER SUM, never a round number chosen first: a buildup is
# either layers or authored geometry, and the body has to be sized to the layers.
DORMER_FRONT_T = DORMER_ZINC_T + DORMER_INSUL_T + DORMER_LINING_T      # 204
DORMER_CHEEK_T = DORMER_FRONT_T
DORMER_PITCH_DEG = 18        # a dormer roof on a 45° main roof MUST be shallow — see
                             # dormer_depth(): the depth blows up as the pitch nears 45°
DORMER_ROOF_T = 22           # boarding (Chipboard_P6 is 22 mm stock, the only one)
DORMER_BOARD = "Chipboard_P6"
DORMER_METAL = "Zinc_Titanium_EN988"   # 18° is BELOW the clay tile's 25° minimum
DORMER_METAL_T = 1
DORMER_WIN_W = 900
DORMER_WIN_H = 1100
DORMER_EAVE = 90             # mm the dormer roof projects past its front and cheeks
# ── The tile field ON the dormer roof, and why there is metal UNDER it. ──────
# At DORMER_PITCH_DEG the clay tile is below its own minimum pitch, so on this roof the
# TILE IS NOT THE WATER BARRIER — the zinc under it is. Laying tile over a sealed metal
# deck is how a shallow dormer is finished when the house is tiled: the tiles match the
# main roof, the metal keeps the water out, and neither is asked to do the other's job.
DORMER_TILE_BATTEN_H = 25    # the tiles hang on battens over the zinc, not on the zinc
DORMER_TILE_BATTEN_W = 38
DORMER_HIP_ROLL_HALF = 90    # half-width of the clay roll capping a hip or the ridge
DORMER_HIP_ROLL_RISE = 70
DORMER_VALLEY_HALF = 160     # half-width of the zinc valley flashing on the MAIN roof

# ══ JOINERY — windows (systems/window) ═══════════════════════════════════════
JOINERY = "Timber_C24"
KARM_DEPTH = 120
KARM_FACE = 45
FIT_GAP = 10
POST_DEPTH = 120
POST_FACE = 45
SASH_DEPTH = 70
SASH_FACE = 45
SASH_LAP = 12
BAR_WIDTH = 24
BAR_PROUD = 12
PANE_ROWS = 3
GLAZING = "Glass_Monolithic"
PANE_MM = 4
GAS_GAP_MM = 16
IGU_MM = 3 * PANE_MM + 2 * GAS_GAP_MM
GLASS_BITE = 10

SILL_OUTER = [
    (122.44, 49.41), (72.44, 49.41), (0.00, 30.00), (0.00, 2.50),
    (5.00, 2.50), (5.00, 10.00),
]
SILL_HEM_CENTRE = (2.50, 2.50)
SILL_HEM_RADIUS = 2.50
SILL_SHEET = "Zinc_Titanium_EN988"
SILL_THICKNESS = 1
SILL_ARC_SEGMENTS = 8
SILL_PROJECTION = 30
SILL_OVERRUN = 30
SILL_BACK_RISE = 49

# ══ JOINERY — doors (systems/door) ═══════════════════════════════════════════
WOOD = "Timber_C24"
GLASS = "Glass_Monolithic"
FRAME_FACE = 70
EXT_FRAME_DEPTH = 140
DOOR_FIT_GAP = 8
LEAF_T = 45
FRONT_LEAF_T = 55
FRONT_INSET = 45
LEAF_REVEAL = 3
STILE_W = 110
MUNTIN_W = 90
GRID_PROUD = 22
PANEL_PROUD = 12
PANEL_MARGIN = 18
FRONT_ROWS = 3
GARDEN_SPLIT = 0.52
GARDEN_RAIL = 90
GLASS_T = 4
ROSE_RADIUS = 25
ROSE_DEPTH = 5
LEVER_PROJECTION = 50
LEVER_LENGTH = 130
LEVER_ARC_RADIUS = 20
LEVER_DIAMETER = 18
IRONMONGERY = "Steel_Stainless_304"
HANDLE_Z = 1050

# ══ INTERIOR PARTITIONS + FINISHES ═══════════════════════════════════════════
PARTITION_BOARD = "Gypsum_Standard"
PARTITION_BOARD_T = 13
PARTITION_BATT = "MineralWool_Acoustic37"
PARTITION_BATT_T = 70                                   # stock: 45 / 70 / 95 / 120
PARTITION_T = 2 * PARTITION_BOARD_T + PARTITION_BATT_T  # 96 — the body is the layer sum
PARTITION_TIMBER = "Timber_C24"

FLOOR_PANEL = "Chipboard_P6"
FLOOR_PANEL_T = 22
DRYWALL = "Gypsum_Standard"
DRYWALL_T = 13
LOFT_DECK = "Chipboard_P6"
LOFT_DECK_T = 22
LOFT_INSULATION = "MineralWool_Acoustic37"
LOFT_INSULATION_T = 120
LOFT_CEILING = "Gypsum_Standard"
LOFT_CEILING_T = 13

# ══ THE GROUND-FLOOR PROGRAM — the input to build_ground_plan ════════════════
# Weights are RELATIVE demands, not sizes: each band divides what is left after the
# hall takes its (stair-derived) width, in proportion to them. Change a weight and the
# plan re-proportions; change the stair and the hall — and therefore every south room —
# re-sizes with it. Nothing here is a coordinate.
SOUTH_BAND_FRAC = 0.55       # of the internal depth; the rest is corridor + service band
CORRIDOR_WIDTH = 1200        # the *fordelingsgang* along the north face of the spine

GROUND_PROGRAM = (
    # (name,         band,    place,  weight, space_type,    long name,     unit)
    ("stue",         "south", "east", 1.00,   "residential", "Living room", "main"),
    ("spisestue",    "south", "west", 1.00,   "residential", "Dining room", "main"),
    ("koekken",      "north", 0,      1.00,   "residential", "Kitchen",     "main"),
    ("badevaerelse", "north", 1,      0.45,   "sanitary",    "Bathroom",    "small"),
    ("vaerelse",     "north", 2,      0.75,   "residential", "Bedroom",     "main"),
)

WIN_W, WIN_H, WIN_SILL = 1400, 1600, 900        # the main ground-floor unit
WIN_SMALL_W, WIN_SMALL_H = 800, 1000            # the bathroom / hall unit
GABLE_WIN_W, GABLE_WIN_H = 1100, 1300           # the loft's gavlvindue
GABLE_WIN_UP = 250                              # above wall_top — the gable triangle's foot
#: unit tag -> (width, height, cill above the floor). A bathroom cills high for privacy,
#: which is a property of the UNIT the room asked for, not of the room.
WIN_UNITS = {"main": (WIN_W, WIN_H, WIN_SILL), "small": (WIN_SMALL_W, WIN_SMALL_H, 1400)}
WIN_PIER_MIN = 350           # the least brick a window leaves at a jamb or a partition
MAX_WIN_PER_WALL = 3
DOOR_W, DOOR_H = 900, 2050                      # the internal leaf
FRONT_DOOR_W, FRONT_DOOR_H = 1100, 2200
GARDEN_DOOR_W, GARDEN_DOOR_H = 1000, 2100
DOOR_WIN_CLEAR = 300         # the gap an exterior door keeps clear of any window


# ══ 0. SHARED — reinforcement (rules/reinforcement, copied verbatim) ═════════
REBAR_COVER = 50
REBAR_PITCH_MAX = 400
REBAR_DIAMETER = 12
_AXIS = ("x", "y", "z")


def _bar_positions(span, cover=REBAR_COVER, pitch_max=REBAR_PITCH_MAX):
    """Bar centres across ``span``, inset ``cover`` from both ends, pitch ≤ pitch_max."""
    usable = span - 2 * cover
    if usable <= 0:
        return [I(span / 2)]
    n = max(1, -(-usable // pitch_max))
    return [I(cover + usable * k / n) for k in range(int(n) + 1)]


def build_reinforcement(element, cover=REBAR_COVER, pitch_max=REBAR_PITCH_MAX,
                        diameter=REBAR_DIAMETER, grade="B500B"):
    """Two orthogonal sets of bars for one solid concrete element.

    The THINNEST extent is the cover direction; each of the other two axes gets one set.
    Call it AFTER the body and BEFORE any ``.opening()`` — bars later than the body (or
    the body annihilates them, with only a warning) and earlier than the opening (which
    then notches the bars standing in it)."""
    b = element.authored_aabb()
    lo = (b.min.x, b.min.y, b.min.z)
    hi = (b.max.x, b.max.y, b.max.z)
    size = [hi[i] - lo[i] for i in range(3)]
    thin = size.index(min(size))
    mid = (lo[thin] + hi[thin]) / 2
    u, v = [i for i in range(3) if i != thin]

    bars = []
    for k, (run, dist) in enumerate(((u, v), (v, u))):
        ends = _bar_positions(size[run], cover, pitch_max)
        a, z = I(lo[run] + ends[0]), I(lo[run] + ends[-1])
        off = I(mid + (k - 0.5) * diameter)
        for i, p in enumerate(_bar_positions(size[dist], cover, pitch_max)):
            fixed = {_AXIS[thin]: off, _AXIS[dist]: I(lo[dist] + p)}
            bars.append(Bar(name=f"{_AXIS[run]}_{i:02d}",
                            path=[Point(**fixed, **{_AXIS[run]: a}),
                                  Point(**fixed, **{_AXIS[run]: z})],
                            diameter=diameter, grade=grade,
                            mark=f"R{_AXIS[run].upper()}"))
    return bars


def _ccw(pts):
    """Winding guard — force positive signed area (a mirrored slope reverses it)."""
    a = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
    return pts if a > 0 else list(reversed(pts))


def _member(name, profile, p0, p1, material=TIMBER):
    """One extruded-profile timber member — a Sweep whose Material carries the section."""
    return Sweep(name=name, path=[p0, p1],
                 material=Material(key=material, profile_mm=profile))


# ══ THE SECTION — assert it closes before a single solid is built ════════════

def assert_the_section_closes():
    """The four numbers that decide whether this is a 1.5-storey house or a drawing of one.

    Nothing in the DSL reports any of them. Each one compiles, renders and validates
    exactly as well when it is wrong.

      1. The stair divides the ground storey EXACTLY. A rise that does not leaves one odd
         step, which is the commonest real stair defect there is and a trip hazard rather
         than a drafting nicety.
      2. The two flights add up to the storey — 9 up, quarter turn, 8 up.
      3. The dormer cill lands within reach of the loft floor. This is what
         ``LOFT_WALL_HEIGHT`` is FOR; at a full storey the same dormer cills at chest
         height and the window is scenery.
      4. The dormer roof pitch is below the clay tile's own minimum, which is WHY it is
         zinc. Reach for tile at 18° and the material's data sheet says no.
    """
    if STAIR_RISERS * STAIR_RISER != GROUND_STOREY_HEIGHT:
        raise AssertionError(
            f"{STAIR_RISERS} risers of {STAIR_RISER} = {STAIR_RISERS * STAIR_RISER}, "
            f"not the {GROUND_STOREY_HEIGHT} storey — one step would be a different height")
    upper = STAIR_RISERS - STAIR_LOWER_STEPS
    if STAIR_LOWER_STEPS + upper != STAIR_RISERS or upper < 2:
        raise AssertionError(f"a {STAIR_LOWER_STEPS} + {upper} split is not a quarter-turn stair")
    # The dormer cill, measured from the loft WALKING surface, before any geometry exists.
    cill_over_floor = LOFT_WALL_HEIGHT + DORMER_SETBACK + DORMER_APRON - LOFT_DECK_T
    if not 700 <= cill_over_floor <= 1250:
        raise AssertionError(
            f"the dormer cill would sit {cill_over_floor} mm over the loft floor. At "
            f"LOFT_WALL_HEIGHT={LOFT_WALL_HEIGHT} this house is not 1.5 storeys — the "
            f"loft floor has dropped away from the eaves and the dormer became a clerestory")
    if DORMER_PITCH_DEG >= TILE_MIN_PITCH_DEG:
        raise AssertionError(
            f"a {DORMER_PITCH_DEG}° dormer roof is at or above the {TILE_MIN_PITCH_DEG}° "
            f"clay-tile minimum — then it should be TILED, not zinc; pick one")
    print(f"INFO section: {STAIR_RISERS} x {STAIR_RISER} = {GROUND_STOREY_HEIGHT} mm ground "
          f"storey, + {LOFT_WALL_HEIGHT} mm skunk -> dormer cill {cill_over_floor} mm over "
          f"the loft floor")
    return True


# ══ 1. SITE BLOCK (systems/site-block) ═══════════════════════════════════════

def build_site_block(footprint_x, footprint_y,
                     buffer=SITE_BLOCK_BUFFER, depth=SITE_BLOCK_DEPTH, heightmap=None):
    """Terrain GEOMETRY — a heightmap ``Mesh``, lowest point normalised to z = 0."""
    half_x = I(footprint_x / 2) + buffer
    half_y = I(footprint_y / 2) + buffer
    if heightmap is None:
        heightmap = [[0, 0], [0, 0]]
    floor = min(h for row in heightmap for h in row)
    grid = [[I(h - floor) for h in row] for row in heightmap]
    return Mesh(heightmap=grid, depth=depth,
                corner_min=Point2D(x=-half_x, y=-half_y),
                corner_max=Point2D(x=half_x, y=half_y),
                source_type="synthetic")


def _terrain_mesh(site):
    for child in site.elements:
        if (isinstance(child, Element)
                and child.ifc_class == "IfcGeographicElement"
                and child.predefined_type == "TERRAIN"):
            for gc in child.elements:
                if isinstance(gc, Mesh):
                    return gc
    return None


# ══ 2. STRIP FOUNDATION + GROUND SLAB (systems/foundation) ═══════════════════

def build_strip_foundation(footprint_x, footprint_y, site,
                           slab_thickness=FOUNDATION_SLAB_THICKNESS,
                           terrain_clearance=FOUNDATION_TERRAIN_CLEARANCE,
                           strip_width=FOUNDATION_STRIP_WIDTH,
                           strip_depth=FOUNDATION_STRIP_DEPTH,
                           strip_gap_max=FOUNDATION_STRIP_GAP_MAX,
                           strip_edge_inset=FOUNDATION_STRIP_EDGE_INSET,
                           clearing_height=FOUNDATION_CLEARING_HEIGHT):
    """``(slab, strips, floor_level, clearing_void)``. Real buried solids: the displacement
    engine carves them out of the terrain, so nothing here calls ``.difference()``. The one
    explicit cut is the clearing void ABOVE the slab, which the CALLER passes to
    ``site.void()`` so the terrain mutation stays visible at the call site."""
    mesh = _terrain_mesh(site)
    z0 = mesh.height_at(x=0, y=0, round=True) if mesh is not None else 0
    slab_top = z0 + terrain_clearance
    slab_bottom = slab_top - slab_thickness
    hx, hy = I(footprint_x / 2), I(footprint_y / 2)

    slab = Slab(name="foundation")
    slab.add(Box(start=Point(x=-hx, y=-hy, z=slab_bottom),
                 end=Point(x=hx, y=hy, z=slab_top),
                 type="foundation", material=FOUNDATION_CONCRETE))
    slab.add(*build_reinforcement(slab))

    # Strips run ∥ the SHORT side, distributed along the LONG axis.
    long_dim, short_dim, long_is_x = ((footprint_x, footprint_y, True)
                                      if footprint_x >= footprint_y
                                      else (footprint_y, footprint_x, False))
    centre_span = long_dim - strip_width - 2 * strip_edge_inset
    if centre_span <= 0:
        centres = [0]
    else:
        n = max(1, -(-centre_span // (strip_gap_max + strip_width)))
        centres = [I(-centre_span / 2 + centre_span * k / n) for k in range(int(n) + 1)]
    half_len = I(short_dim / 2) - strip_edge_inset
    half_w = I(strip_width / 2)

    strips = []
    for i, c in enumerate(centres):
        if long_is_x:
            start = Point(x=c - half_w, y=-half_len, z=slab_bottom - strip_depth)
            end = Point(x=c + half_w, y=half_len, z=slab_bottom)
        else:
            start = Point(x=-half_len, y=c - half_w, z=slab_bottom - strip_depth)
            end = Point(x=half_len, y=c + half_w, z=slab_bottom)
        footing = Element(ifc_class="IfcFooting", name=f"strip_{i}")
        footing.add(Box(start=start, end=end, material=FOUNDATION_CONCRETE))
        footing.add(*build_reinforcement(footing))
        strips.append(footing)

    clearing_void = Box(start=Point(x=-hx, y=-hy, z=slab_top),
                        end=Point(x=hx, y=hy, z=slab_top + clearing_height))
    print(f"INFO foundation: slab top z={slab_top} ({terrain_clearance} mm over the terrain "
          f"at the origin), {len(strips)} strips to z={slab_bottom - strip_depth} "
          f"({slab_top - (slab_bottom - strip_depth)} mm below the slab top, frost-free)")
    return slab, strips, slab_top, clearing_void


# ══ 2.2 STENKANT — the perimeter gravel apron (systems/stenkant) ═════════════

def build_stenkant(footprint_x, footprint_y, site,
                   width=STENKANT_WIDTH, depth=STENKANT_DEPTH, subdiv=STENKANT_SUBDIV,
                   cut_overshoot=STENKANT_CUT_OVERSHOOT, material=STENKANT_GRAVEL):
    """``(bed, voids)`` — one ``IfcEarthworksFill`` holding 4 heightmap patches whose TOP
    follows the terrain and whose BOTTOM is one level plane. A Mesh never carves, so the
    trench is an EXPLICIT ``site.void()``, not an inferred displacement."""
    terr = _terrain_mesh(site)
    if terr is None:
        raise ValueError("build_stenkant needs the site block first — no TERRAIN mesh found")
    hx, hy = I(footprint_x / 2), I(footprint_y / 2)
    ox, oy = hx + width, hy + width
    patches = [
        (-ox, -oy, ox, -hy, subdiv + 1, 2),
        (-ox, hy, ox, oy, subdiv + 1, 2),
        (-ox, -hy, -hx, hy, 2, subdiv + 1),
        (hx, -hy, ox, hy, 2, subdiv + 1),
    ]
    grids = [terr.heightmap_at(corner_min=Point2D(x=x0, y=y0),
                               corner_max=Point2D(x=x1, y=y1),
                               rows=rows, cols=cols, round=True)
             for (x0, y0, x1, y1, cols, rows) in patches]
    all_heights = [h for g in grids for row in g for h in row]
    bottom = I(min(all_heights) - depth)
    cut_top = I(max(all_heights) + cut_overshoot)

    bed = Element(ifc_class="IfcEarthworksFill", predefined_type="BACKFILL",
                  name="stenkant", material=material)
    voids = []
    for (x0, y0, x1, y1, _c, _r), grid in zip(patches, grids):
        bed.add(Mesh(heightmap=grid, bottom_z=bottom,
                     corner_min=Point2D(x=x0, y=y0), corner_max=Point2D(x=x1, y=y1)))
        voids.append(Box(start=Point(x=x0, y=y0, z=bottom),
                         end=Point(x=x1, y=y1, z=cut_top)))
    print(f"INFO stenkant: {width} mm {material} apron, level bottom z={bottom}, "
          f"terrain-following top")
    return bed, voids


# ══ 3 + 4. THE WALL — core + cladding leaf, as ONE body-less assembly ════════

CORE_PROPS = {
    "Pset_WallCommon": {"LoadBearing": True, "IsExternal": True,
                        "FireRating": "REI 60", "Combustible": False},
    "Pset_ConcreteElementGeneral": {"ExposureClass": "XC1"},
}
_SANDWICH_OUTER_TO_INNER = [
    ("Brick_Red_DK", EXT_BRICK_MM),
    ("Cavity_Ventilated", EXT_CAVITY_MM),
    ("MineralWool_Facade34", EXT_INSULATION_MM),
]


def build_walls(footprint_x, footprint_y, base_z, top_z,
                core_thickness=WALL_CORE_THICKNESS, concrete=STRUCT_CONCRETE,
                t=EXT_SANDWICH_MM):
    """The 4 walls of one storey, each an ASSEMBLY with no body of its own::

        wall:south                    (no body)
          wall:core                   concrete, load-bearing, reinforced
          wall:facade_south           brick / cavity / mineral wool, mitered

    That is what lets ONE ``.opening()`` cut the whole thickness — brick, cavity,
    insulation, the concrete core AND the core's rebar. Never cut the core separately:
    two calls that have to agree fail silently in one direction, and the facade opens
    while you walk into 180 mm of concrete.

    Returned per side as ``(wall, start, end, run, outward)``; ``start``/``end`` describe
    the CLADDING leaf, whose outer face is the zero every ``inset=`` measures from."""
    hx, hy = I(footprint_x / 2), I(footprint_y / 2)
    inx, iny = hx - core_thickness, hy - core_thickness
    ox, oy = hx + t, hy + t

    cfg = [
        ("south", Point(x=-ox, y=-oy, z=base_z), Point(x=ox, y=-hy, z=top_z),
         (1, 0, 0), (0, -1, 0),
         Point(x=-hx, y=-hy, z=base_z), Point(x=hx, y=-iny, z=top_z)),
        ("north", Point(x=-ox, y=hy, z=base_z), Point(x=ox, y=oy, z=top_z),
         (1, 0, 0), (0, 1, 0),
         Point(x=-hx, y=iny, z=base_z), Point(x=hx, y=hy, z=top_z)),
        ("west", Point(x=-ox, y=-oy, z=base_z), Point(x=-hx, y=oy, z=top_z),
         (0, 1, 0), (-1, 0, 0),
         Point(x=-hx, y=-iny, z=base_z), Point(x=-inx, y=iny, z=top_z)),
        ("east", Point(x=hx, y=-oy, z=base_z), Point(x=ox, y=oy, z=top_z),
         (0, 1, 0), (1, 0, 0),
         Point(x=inx, y=-iny, z=base_z), Point(x=hx, y=iny, z=top_z)),
    ]
    bodies = {n: Box(name="body", start=s, end=e) for n, s, e, _r, _o, _cs, _ce in cfg}
    for ns, y in (("south", -oy), ("north", oy)):
        for ew, x in (("west", -ox), ("east", ox)):
            miter(bodies[ns], bodies[ew], at=Point(x=x, y=y, z=base_z), edge=(0, 0, 1))

    leaf_props = {"Pset_WallCommon": {"LoadBearing": False, "IsExternal": True}}
    walls = {}
    for name, start, end, run, outward, core_start, core_end in cfg:
        core = Wall(name="core", props=dict(CORE_PROPS))
        core.add(Box(start=core_start, end=core_end, material=concrete))
        core.add(*build_reinforcement(core))

        leaf = Wall(name=f"facade_{name}", props=dict(leaf_props))
        leaf.add(bodies[name])
        leaf.layers = LayerSet(name="ext_wall_type_a", outward=outward,
                               layers=[Material(key=k, thickness_mm=t_)
                                       for k, t_ in _SANDWICH_OUTER_TO_INNER])

        wall = Wall(name=name)              # the assembly — deliberately body-less
        wall.add(core, leaf)
        walls[name] = (wall, start, end, run, outward)
    return walls


def _wall_along_centre(start, end, run, outward, centre):
    """A WORLD coordinate on the run axis, expressed in the wall's own +x frame.

    A wall's local +x is *right as seen from outside*, so it agrees with the world run
    axis on the leaves whose outward normal lies on the −(Z × run) side and opposes it on
    the others. Derived from the leaf's own outward normal, never hand-cased per side."""
    axis = 0 if run[0] else 1
    lo = min(start.x, end.x) if axis == 0 else min(start.y, end.y)
    hi = max(start.x, end.x) if axis == 0 else max(start.y, end.y)
    left = (-run[1], run[0], 0)
    if sum(a * b for a, b in zip(outward, left)) > 0:
        return hi - centre
    return centre - lo


# ══ THE DORMER, AS PURE NUMBERS ══════════════════════════════════════════════
# Three systems need the same dormer envelope and must not each re-derive it: the roof
# FRAME trims its rafters to it, the COVERING splits its courses around it, and the
# dormer itself is built in it. So the geometry is one function returning one dict, and
# the three consumers read fields off it. Let them drift and the failure is silent — a
# rafter through a dormer window renders perfectly.

#: (tag, slope sign, window count, centre x). "2-3 windows" is a per-dormer decision:
#: the show side (south) takes three, the north takes two.
DORMERS = (("south", -1, 3, 0), ("north", 1, 2, 0))


def dormer_geometry(sign, n_win, centre_x, wall_top, ridge_z,
                    pitch_deg=ROOF_PITCH_DEG, dormer_pitch_deg=DORMER_PITCH_DEG):
    """Every number the dormer, the roof FRAME and the COVERING share.

    **THE RIDGE RUNS FRONT-TO-BACK, and that is what makes the hip possible.** A hip roof's
    ridge is equidistant from the eaves it is hipped between, and this dormer has a front
    eave and two SIDE eaves — no back eave, because the back dies into the main roof. So
    the ridge sits at the dormer's x-centre and runs BACKWARD, and it stops where the main
    roof rises to meet it. Reach for a side-to-side ridge instead and you are asking two
    equal-pitch hips to close over a fixed back line: on this footprint they cannot, and
    the three "faces" come out COPLANAR — a flat roof with two cosmetic seams that reads as
    a hip in the code and not in the render.

    So the whole form comes off three level eaves at ``ze`` and one pitch::

        z_ridge       = ze + (W/2)·tan p          W/2 from each side eave, equal pitch
        u_ridge_start = u_front − W/2             where the front hip end reaches it
        u_ridge_die   = ridge_z − z_ridge         where the MAIN roof rises to meet it
        u_valley      = ridge_z − ze              where a side eave meets the main roof

    and the hips fall out at 45° in plan, because equal pitches always bisect.

    The DEPTH still decides whether any of it fits: ``(head−foot)/(tan P − tan p)`` blows
    up as the dormer pitch approaches the main pitch — 9.9 m at 40° on a 45° roof. That is
    why a dormer on a steep roof is FLAT-ish, and why 18° is a structural fact rather than
    a style choice. It is also below the clay tile's 25° minimum, which is why there is
    ZINC under the tiles."""
    tp, td = math.tan(math.radians(pitch_deg)), math.tan(math.radians(dormer_pitch_deg))
    e = DORMER_EAVE
    y_f = GABLE_HALF_Y - DORMER_SETBACK
    foot_z = ridge_z - I(y_f * tp)
    cill_z = foot_z + DORMER_APRON
    head_z = cill_z + DORMER_WIN_H + DORMER_HEAD
    depth = I((head_z - foot_z) / (tp - td))
    width = n_win * DORMER_WIN_W + (n_win + 1) * DORMER_PIER

    x0, x1 = centre_x - width // 2, centre_x + width // 2
    X0, X1 = x0 - e, x1 + e                 # the OVERHANG outline
    half_w = (X1 - X0) / 2.0
    u_front = y_f + e                       # |y| of the front eave
    ze = head_z - e * td                    # every eave — front and both sides — is level
    z_ridge = ze + half_w * td
    u_ridge_start = u_front - half_w        # the front hip end reaches the ridge here
    u_ridge_die = ridge_z / tp - z_ridge / tp    # …and the main roof rises to meet it here
    u_valley = ridge_z / tp - ze / tp       # a side eave meets the main roof here
    u_cheek_back = ridge_z / tp - head_z / tp    # the cheek's level top meets the main roof
    cos_P = math.cos(math.radians(pitch_deg))
    return {
        "tag": "south" if sign < 0 else "north", "sign": sign, "n_win": n_win,
        "centre_x": centre_x, "width": width, "x0": x0, "x1": x1,
        "X0": X0, "X1": X1, "xc": (X0 + X1) / 2.0, "half_w": half_w,
        "y_front": sign * y_f, "abs_y_front": y_f, "depth": depth,
        "foot_z": foot_z, "base_z": foot_z - DORMER_BASE_DROP,
        "cill_z": cill_z, "head_z": head_z,
        "ze": ze, "z_ridge": z_ridge, "u_front": u_front,
        "u_ridge_start": u_ridge_start, "u_ridge_die": u_ridge_die,
        "u_valley": u_valley, "u_cheek_back": u_cheek_back,
        "slope_len": half_w / math.cos(math.radians(dormer_pitch_deg)),
        "s_front": (EAVE_HALF_Y - u_front) / cos_P,
        "s_back": (EAVE_HALF_Y - u_ridge_die) / cos_P,   # deepest reach into the main roof
    }


def dormer_x_span_at(g, abs_y):
    """The x range the dormer occupies on the MAIN roof at ``|y|``, or ``None``.

    Its footprint is NOT a rectangle: behind ``u_valley`` the two valleys pinch in, and at
    ``u_ridge_die`` they meet. Splitting the main roof on the bounding rectangle instead
    over-removes ~2 m² of tile at the back corners — courses that should run right up to
    the valley simply are not there, and nothing reports a tile that was never built."""
    if abs_y > g["u_front"] or abs_y <= g["u_ridge_die"] + 0.5:
        return None
    if abs_y >= g["u_valley"]:
        return g["X0"], g["X1"]
    inset = (g["u_valley"] - abs_y) / math.tan(math.radians(DORMER_PITCH_DEG))
    return g["X0"] + inset, g["X1"] - inset


def build_dormer_geometries(wall_top, ridge_z):
    return [dormer_geometry(sign, n, cx, wall_top, ridge_z)
            for _tag, sign, n, cx in DORMERS]


def dormer_roof_faces(g):
    """``[(name, contour)]`` — the THREE planes of the hipped roof, as finished surfaces.

    * ``roof_front`` — the hip END: a triangle off the front eave, rising to the ridge's
      front point. Its plane depends only on ``y``.
    * ``roof_side_left`` / ``roof_side_right`` — quadrilaterals off the two level side
      eaves, rising to the ridge. Their planes depend only on ``x``.

    Depending on different coordinates is exactly what makes them non-coplanar, and
    ``assert_the_roof_is_really_hipped`` measures it rather than trusting this comment."""
    s = g["sign"]
    X0, X1, xc = g["X0"], g["X1"], g["xc"]
    ze, zr = g["ze"], g["z_ridge"]
    r0 = (xc, s * g["u_ridge_start"], zr)          # ridge front point
    r1 = (xc, s * g["u_ridge_die"], zr)            # …and where the main roof kills it
    return [
        ("roof_front", [(X0, s * g["u_front"], ze), (X1, s * g["u_front"], ze), r0]),
        ("roof_side_left", [(X0, s * g["u_front"], ze), r0, r1,
                            (X0, s * g["u_valley"], ze)]),
        ("roof_side_right", [(X1, s * g["u_front"], ze), (X1, s * g["u_valley"], ze),
                             r1, r0]),
    ]


def assert_the_roof_is_really_hipped(g, tol=0.01):
    """The three roof planes must have three DIFFERENT normals.

    This is the check the first version of this file did not have, and it is the reason
    it shipped a flat roof described as a hip. Coplanar faces render as one surface, carry
    correct materials, pass `check_renderability.mjs` and validate — the only thing wrong
    with them is that they are not the building."""
    normals = {name: _face_up(pts) for name, pts in dormer_roof_faces(g)}
    front, left, right = normals.values()
    for tag, other in (("front/left", left), ("front/right", right)):
        if all(abs(a - b) < tol for a, b in zip(front, other)):
            raise AssertionError(
                f"dormer '{g['tag']}': the {tag} roof faces are COPLANAR "
                f"({tuple(round(c, 3) for c in front)}) — that is a flat roof with a "
                f"decorative seam, not a hip. The ridge has to run FRONT-TO-BACK; see "
                f"dormer_geometry")
    if all(abs(a - b) < tol for a, b in zip(left, right)):
        raise AssertionError(f"dormer '{g['tag']}': the two side planes are coplanar")
    return True


def _dormer_cut_planes(g, face, dormer_pitch_deg=DORMER_PITCH_DEG):
    """The VERTICAL planes a tile band on ``face`` is scribed to — hips, and the valleys.

    Each is written as an inequality in world coordinates and then read off as
    ``(origin, normal)``, because ``clip()`` removes the half-space its normal points at.
    Deriving them rather than drawing them is what keeps the two sides of a hip agreeing:
    the front face removes ``x + s·y < C`` and the side face removes ``x + s·y > C`` —
    the same plane, opposite sides, one constant."""
    s, tp = g["sign"], math.tan(math.radians(dormer_pitch_deg))
    X0, X1, ze = g["X0"], g["X1"], g["ze"]
    # left hip:  x + s·y = X0 + u_front      right hip: x − s·y = X1 − u_front
    p_left = Point(x=I(X0), y=I(s * g["u_front"]), z=I(ze))
    p_right = Point(x=I(X1), y=I(s * g["u_front"]), z=I(ze))
    if face == "front":                                  # keep what is BETWEEN the hips
        return [(p_left, (-1.0, -float(s), 0.0)), (p_right, (1.0, -float(s), 0.0))]
    if face == "side_left":                              # hip above, valley below
        return [(p_left, (1.0, float(s), 0.0)),
                (Point(x=I(X0), y=I(s * g["u_valley"]), z=I(ze)),
                 (-tp, -float(s), 0.0))]
    return [(p_right, (-1.0, float(s), 0.0)),
            (Point(x=I(X1), y=I(s * g["u_valley"]), z=I(ze)), (tp, -float(s), 0.0))]


def build_dormer_tiles(g, dormer_pitch_deg=DORMER_PITCH_DEG):
    """Clay tile courses on all three roof planes, plus the clay rolls over the arrises.

    **ONE course set-out serves all three faces**, and that is not a convenience — it is
    what a hip IS. Every face rises ``half_w · tan p`` over a plan run of ``half_w``, so
    every face has the same slope length, and courses laid at one gauge meet at the hips
    instead of stepping past each other.

    Each face gets a local frame: ``up`` (up the slope, in the plane) and a WIDTH function
    giving how far the face reaches sideways at a given plan depth. Both narrow linearly —
    the hip end because the hips close at 45° in plan, the side planes because the front
    hip and the valley converge — so a course is one `Sweep` whose profile is the pantile
    wave repeated across that width."""
    s, tag = g["sign"], g["tag"]
    tp = math.tan(math.radians(dormer_pitch_deg))
    cp = math.cos(math.radians(dormer_pitch_deg))
    sp = math.sin(math.radians(dormer_pitch_deg))
    X0, X1, xc = g["X0"], g["X1"], g["xc"]
    ze = g["ze"]
    ref = DORMER_TILE_BATTEN_H + max(y for _, y in ROOF_TILE_PROFILE)
    courses, gauge = _course_layout(g["slope_len"])

    # (name, up-direction, eave anchor, width(d) -> (lo, hi) on the LEVEL axis)
    def front_width(d):
        return X0 + d, X1 - d                                  # 45 deg hips, in plan
    def side_width(d):
        return (s * (g["u_valley"] - d * tp), s * (g["u_front"] - d))

    # (name, up-direction, eave anchor, width fn, axis the width runs on, outward normal,
    #  does the sweep's local X point the POSITIVE way along that axis?)
    #
    # Local X of a Sweep along `up` is `unit(Z x up)`. Worked through per face rather than
    # guessed: on the hip end it comes out −x when the slope faces −y and +x when it faces
    # +y; on the left side plane +y; on the right −y. A course started at the wrong end
    # sweeps its whole width off the dormer and across the roof — which is what the first
    # attempt did, because `side_width` returns its pair in DESCENDING order on the south
    # dormer (both values negative) and the code assumed ascending. Sort, then pick the end
    # from the direction; never assume a coordinate pair is ordered.
    faces = [
        ("front", (0.0, -s * cp, sp), (0.0, s * g["u_front"], ze), front_width, 0,
         (0.0, s * sp, cp), s > 0),
        ("side_left", (cp, 0.0, sp), (X0, 0.0, ze), side_width, 1,
         (-sp, 0.0, cp), True),
        ("side_right", (-cp, 0.0, sp), (X1, 0.0, ze), side_width, 1,
         (sp, 0.0, cp), False),
    ]
    out = []
    roof = Roof(name=f"dormer_tiles_{tag}")
    for fname, up, anchor, width_at, axis, nrm, localx_pos in faces:
        for k in range(courses):
            s0, s1 = k * gauge, k * gauge + gauge + TILE_HEAD_LAP
            lo, hi = sorted(width_at((s0 + s1) / 2.0 * cp))
            span = hi - lo
            if span < TILE_COVER_WIDTH:
                continue                       # the last course, up at the hip point
            # Run the band to the course's WIDEST end and CUT it back, exactly as a roofer
            # does: a tile at a hip is scribed to the hip line, not chosen from a bin of
            # pre-narrowed tiles. Sizing each band to its midpoint width instead leaves a
            # sawtooth of bare zinc half a tile deep along every arris — every course a
            # little short at its tail and a little proud at its head.
            wide = max(width_at(s0 * cp), width_at(s1 * cp), key=lambda w: abs(w[1] - w[0]))
            wlo, whi = sorted(wide)
            prof = _tile_band_profile(whi - wlo, ref)
            pts = []
            for sd, lift in ((s0, DORMER_TILE_BATTEN_H + TILE_LAP_STEP),
                             (s1, DORMER_TILE_BATTEN_H)):
                base = [anchor[i] + up[i] * sd + nrm[i] * lift for i in range(3)]
                base[axis] = wlo if localx_pos else whi
                pts.append(base)
            band = Sweep(name=f"tile_{fname}_{k}", profile=prof,
                         path=[Point(x=I(p[0]), y=I(p[1]), z=I(p[2])) for p in pts],
                         material=Material(key=TILE))
            for origin, normal in _dormer_cut_planes(g, fname):
                band.clip(origin=origin, normal=normal)
            roof.add(band)
    out.append(roof)

    # The clay rolls: two hips (front eave corner -> ridge front point) and the ridge
    # itself (ridge front point -> where the main roof kills it).
    r0 = (xc, s * g["u_ridge_start"], g["z_ridge"])
    r1 = (xc, s * g["u_ridge_die"], g["z_ridge"])
    lift = DORMER_TILE_BATTEN_H + max(y for _, y in ROOF_TILE_PROFILE)
    hw, rise = DORMER_HIP_ROLL_HALF, DORMER_HIP_ROLL_RISE
    tri = [Point2D(x=I(a), y=I(b)) for a, b in _ccw([(-hw, 0), (hw, 0), (0, rise)])]
    for rname, a, b in (
        ("hip_left", (X0, s * g["u_front"], ze), r0),
        ("hip_right", (X1, s * g["u_front"], ze), r0),
        ("ridge", r0, r1),
    ):
        out.append(Sweep(
            name=f"dormer_{rname}_{tag}", profile=tri,
            path=[Point(x=I(a[0]), y=I(a[1]), z=I(a[2] + lift)),
                  Point(x=I(b[0]), y=I(b[1]), z=I(b[2] + lift))],
            material=Material(key=TILE)))
    print(f"INFO dormer: {tag} roof tiled — {courses} courses at {gauge:.0f} mm on each of "
          f"3 planes ({g['slope_len']:.0f} mm slope), 2 hip rolls + a "
          f"{g['u_ridge_start'] - g['u_ridge_die']:.0f} mm ridge roll, all clay over zinc")
    return out


def build_dormer_valley_flashings(g, ridge_z, pitch_deg=ROOF_PITCH_DEG,
                                  half=DORMER_VALLEY_HALF):
    """A zinc strip up each valley, laid in the MAIN roof plane.

    The valley runs from where the side eave meets the main roof to where the ridge dies —
    a straight line in plan, because both surfaces are planes. Authored as a quad IN the
    main roof plane and lifted to the tile surface, so it needs no sweep-frame reasoning:
    the plane's height depends only on ``|y|``, so every corner's z is read off that.

    Anchor points (world coordinates, mm, Z-up):
    * South dormer (3 windows):
      - Ridge connection point: (0, -3078, 6288) mm (where the dormer ridge dies into the main roof)
      - Left valley point:      (-920, -2958, 6071) mm (where the left side eave meets the main roof)
      - Right valley point:     (920, -2958, 6071) mm (where the right side eave meets the main roof)
    * North dormer (2 windows):
      - Ridge connection point: (0, 2838, 6109) mm (where the dormer ridge dies into the main roof)
      - Left valley flashing:   (-645, 3048, 5982) mm
      - Right valley flashing:  (645, 3048, 5982) mm"""
    s = g["sign"]
    tP = math.tan(math.radians(pitch_deg))
    lift = (RAFTER_PROFILE[1] / 2.0 + UNDERLAY_T + BATTEN_H)
    out = []
    for tag, xa, xb in (("left", g["X0"], g["xc"]), ("right", g["X1"], g["xc"])):
        quad = []
        for x, u in ((xa, g["u_valley"]), (xb, g["u_ridge_die"])):
            for du in (-half, half):
                uu = u + du
                quad.append((x, s * uu, ridge_z - uu * tP + lift / math.cos(
                    math.radians(pitch_deg))))
        quad = [quad[0], quad[1], quad[3], quad[2]]
        out.append(Extrude(name=f"valley_{tag}_{g['tag']}",
                           contour=[Point(x=I(p[0]), y=I(p[1]), z=I(p[2])) for p in quad],
                           material=Material(key=DORMER_METAL,
                                             thickness_mm=DORMER_METAL_T)))
    return out


# ══ 6. THE ROOF FRAME (systems/roof) ═════════════════════════════════════════

def _ridge_z(footprint_y, wall_top, pitch_deg=ROOF_PITCH_DEG):
    """Apex height. The eave pivots at the OUTER BRICK FACE, not the concrete edge, so the
    pitched roof reaches ``wall_top`` at the brick face and sits ON TOP of the brick. Pivot
    at the concrete edge instead and the roof drops to ``wall_top − sandwich`` there, and
    the brick facade and the eaves cornice poke straight through it."""
    return wall_top + I(GABLE_HALF_Y * math.tan(math.radians(pitch_deg)))


def _bay_xs(footprint_x, spacing=RAFTER_SPACING):
    """Rafter-pair x positions along the ridge, ``spacing`` c/c with a small eave inset."""
    hx = I(footprint_x / 2)
    n_gaps = max(1, round((footprint_x - 200) / spacing))
    return [I(-hx + 100 + (footprint_x - 200) * k / n_gaps) for k in range(n_gaps + 1)]


def collar_level(loft_level, headroom=LOFT_HEADROOM, deck_t=LOFT_DECK_T):
    """The collar sits where the loft gets ``headroom`` of FLAT ceiling — a derivation,
    not the fraction-of-the-rise the trimmed roof example uses. On a 1.5-storey house the
    collar height is the room's headroom; on a bare frame with no floor under it there is
    nothing to measure from, which is why the roof skill had to guess."""
    return loft_level + deck_t + headroom + COLLAR_PROFILE[1] // 2


def build_roof_frame(footprint_x, footprint_y, wall_top, loft_level, dormers,
                     pitch_deg=ROOF_PITCH_DEG, spacing=RAFTER_SPACING):
    """``(rafters, collars, plates, trimming)`` — the *hanebåndsspær*, opened for the dormers.

    Rafter pairs are ``miter()``ed at the ridge (an authored joint, so neither is carved)
    and CANTILEVER past the eave to the fascia. Where a rafter runs through a dormer it is
    **not** built to the eave: it starts at the dormer's back line, which is what a trimmed
    rafter is on site. The opening is then closed the way a real one is —

      * a **header** (skiftespær) across the back of each dormer, carrying the cut rafters;
      * a **doubled trimming rafter** on each cheek line, carrying the header.

    Leave the rafters whole instead and they run straight through the dormer's windows: it
    compiles, it validates, and it is visible only from inside the dormer."""
    hy = I(footprint_y / 2)
    hx = I(footprint_x / 2)
    ridge_z = _ridge_z(footprint_y, wall_top, pitch_deg)
    th = math.radians(pitch_deg)
    eave_z = ridge_z - I(EAVE_HALF_Y * math.tan(th))

    def _cut(sign, x):
        """Where a rafter on slope ``sign`` at ``x`` must start — the eave, or a dormer back."""
        for d in dormers:
            if d["sign"] == sign and d["x0"] < x < d["x1"]:
                u = d["u_ridge_die"]
                return (sign * u, ridge_z - I(u * math.tan(math.radians(pitch_deg)))), d
        return (sign * EAVE_HALF_Y, eave_z), None

    xs = _bay_xs(footprint_x, spacing)
    rafters, trimmed = [], {d["tag"]: 0 for d in dormers}
    for i, x in enumerate(xs):
        ends = {}
        for sign, tag in ((-1, "s"), (1, "n")):
            (y_end, z_end), d = _cut(sign, x)
            ends[tag] = _member(f"rafter_{tag}_{i}", RAFTER_PROFILE,
                                Point(x=x, y=I(y_end), z=I(z_end)),
                                Point(x=x, y=0, z=ridge_z))
            if d is not None:
                trimmed[d["tag"]] += 1
        miter(ends["s"], ends["n"], at=Point(x=x, y=0, z=ridge_z), edge=(1, 0, 0))
        rafters.extend((ends["s"], ends["n"]))

    # HANEBÅND — the collar, lapped on the OUTSIDE face of each rafter pair (bolted through
    # the lap, never let in), so it takes no boolean against a rafter at all. Its height is
    # DERIVED from the headroom the loft is meant to have, not from a fraction of the rise.
    collar_z = collar_level(loft_level)
    cw = COLLAR_PROFILE[0]
    x_off = (RAFTER_PROFILE[0] + cw) // 2      # halve the SUM, never sum the halves
    y_at_collar = I(GABLE_HALF_Y * (1 - (collar_z - wall_top) / (ridge_z - wall_top)))
    y_long = y_at_collar + 400
    collars = []
    for i, x in enumerate(xs):
        for side, sx in (("a", -1), ("b", 1)):
            c = _member(f"collar_{i}{side}", COLLAR_PROFILE,
                        Point(x=x + sx * x_off, y=-y_long, z=collar_z),
                        Point(x=x + sx * x_off, y=y_long, z=collar_z))
            for cut in _rafter_face_voids(x + sx * x_off, cw, wall_top, ridge_z, pitch_deg):
                c.difference(cut)
            collars.append(c)

    # REM — a wall plate laid flat on each eave wall.
    plates = [_member(name, PLATE_PROFILE,
                      Point(x=-hx, y=y, z=wall_top), Point(x=hx, y=y, z=wall_top))
              for name, y in (("plate_south", -hy), ("plate_north", hy))]

    # The dormer trimming: a doubled rafter on each cheek + a header across the back.
    trimming = []
    for d in dormers:
        sign = d["sign"]
        for tag, x in (("left", d["x0"]), ("right", d["x1"])):
            for k, dx in ((0, -RAFTER_PROFILE[0]), (1, 0)):
                trimming.append(_member(
                    f"trimmer_{d['tag']}_{tag}_{k}", RAFTER_PROFILE,
                    Point(x=I(x + dx), y=I(sign * EAVE_HALF_Y), z=I(eave_z)),
                    Point(x=I(x + dx), y=0, z=ridge_z)))
        trimming.append(_member(
            f"header_{d['tag']}", RAFTER_PROFILE,
            Point(x=I(d["x0"]), y=I(d["sign"] * d["u_ridge_die"]),
                  z=I(ridge_z - d["u_ridge_die"] * math.tan(math.radians(pitch_deg)))),
            Point(x=I(d["x1"]), y=I(d["sign"] * d["u_ridge_die"]),
                  z=I(ridge_z - d["u_ridge_die"] * math.tan(math.radians(pitch_deg))))))
        print(f"INFO roof: dormer '{d['tag']}' trims {trimmed[d['tag']]} rafters — "
              f"1 header at |y|={d['u_ridge_die']:.0f} on 2 doubled trimming rafters")
    return rafters, collars, plates, trimming


def _rafter_face_voids(x_centre, width, wall_top, ridge_z, pitch_deg):
    """Two void slabs, one per slope, lying just OUTSIDE the rafters' upper face. Difference
    them off a collar and its ends come back trimmed flush with the roof plane."""
    th = math.radians(pitch_deg)
    half = RAFTER_PROFILE[1] / 2.0
    reach, run, w = 2000.0, 1500.0, 4 * width
    rect = [Point2D(x=I(-w / 2), y=I(-reach / 2)), Point2D(x=I(w / 2), y=I(-reach / 2)),
            Point2D(x=I(w / 2), y=I(reach / 2)), Point2D(x=I(-w / 2), y=I(reach / 2))]
    voids = []
    for sign in (-1, 1):
        ny, nz = sign * math.sin(th), math.cos(th)
        off = half + reach / 2.0
        dy, dz = -sign * GABLE_HALF_Y, ridge_z - wall_top
        ln = math.hypot(dy, dz)
        uy, uz = dy / ln, dz / ln
        a = (sign * GABLE_HALF_Y + off * ny - run * uy, wall_top + off * nz - run * uz)
        b = (0 + off * ny + run * uy, ridge_z + off * nz + run * uz)
        voids.append(Sweep(profile=rect,
                           path=[Point(x=I(x_centre), y=I(a[0]), z=I(a[1])),
                                 Point(x=I(x_centre), y=I(b[0]), z=I(b[1]))]))
    return voids


def _floor_beam_xs(footprint_x, spacing=RAFTER_SPACING, core_thickness=WALL_CORE_THICKNESS):
    """Bays that actually carry a floor beam — the ones nearest the gables would sit
    inside the wall rather than spanning across, so they are dropped."""
    inner_x = I(footprint_x / 2) - core_thickness
    return [x for x in _bay_xs(footprint_x, spacing)
            if abs(x) + FLOOR_BEAM_PROFILE[0] // 2 <= inner_x]


def build_floor_beams(footprint_x, footprint_y, loft_level,
                      spacing=RAFTER_SPACING, core_thickness=WALL_CORE_THICKNESS,
                      inner_wall_offset=INNER_WALL_OFFSET):
    """The BJÆLKELAG that goes ACROSS the building, at the LOFT floor level.

    This is where a 1.5-storey house differs from the trimmed system examples: they put
    the bjælkelag at ``wall_top`` because they had no loft below it. Here the loft floor
    IS the upper floor, so the beams bear on the GROUND storey's walls and the skunk wall
    sits on top of them. Each end runs ``BEAM_EMBED`` past the inner face, so — added
    after the walls — it cuts its own bearing pocket."""
    hy = I(footprint_y / 2)
    inner_y = hy - core_thickness
    y_end = inner_y + BEAM_EMBED
    _bw, bd = FLOOR_BEAM_PROFILE
    z_mid = loft_level - bd // 2                       # top flush with the loft level

    xs = _floor_beam_xs(footprint_x, spacing, core_thickness)
    beams = [Sweep(name=f"floor_beam_{i}",
                   path=[Point(x=x, y=-y_end, z=z_mid), Point(x=x, y=y_end, z=z_mid)],
                   material=Material(key=FLOOR_BEAM_GLULAM, profile_mm=FLOOR_BEAM_PROFILE))
             for i, x in enumerate(xs)]

    supports = sorted([-inner_y, -inner_wall_offset, inner_wall_offset, inner_y])
    longest = max(b - a for a, b in zip(supports, supports[1:]))
    if longest > BEAM_MAX_FREE_SPAN:
        raise NotImplementedError(
            f"floor-beam span {longest} mm exceeds beam_max_free_span {BEAM_MAX_FREE_SPAN} "
            f"— re-site the inner walls or add a mid-span column line")
    print(f"INFO roof: {len(beams)} bjælkelag beams at z_top={loft_level}, supports at "
          f"y={supports}, longest span {longest} mm <= {BEAM_MAX_FREE_SPAN}")
    return beams


def build_inner_wall_posts(footprint_x, wall_top, loft_level,
                           offset=INNER_WALL_OFFSET, spacing=RAFTER_SPACING,
                           pitch_deg=ROOF_PITCH_DEG, core_thickness=WALL_CORE_THICKNESS):
    """The posts standing ON the bjælkelag that carry the rafters, on the lines where the
    loft's own partitions go. Each is housed ``BEAM_EMBED`` into its rafter — added after
    the rafters, it cuts that seat itself, so it is fastened rather than merely touching."""
    th = math.radians(pitch_deg)
    ridge_z = _ridge_z(FOUNDATION_FOOTPRINT_Y, wall_top, pitch_deg)
    centre_z = ridge_z - (offset) * math.tan(th)
    under = centre_z - (RAFTER_PROFILE[1] / 2.0) / math.cos(th)
    posts = []
    for i, x in enumerate(_floor_beam_xs(footprint_x, spacing, core_thickness)):
        for side, sy in (("s", -1), ("n", 1)):
            posts.append(_member(f"post_{i}{side}", POST_PROFILE,
                                 Point(x=x, y=sy * offset, z=loft_level),
                                 Point(x=x, y=sy * offset, z=I(under + BEAM_EMBED))))
    return posts


def build_ridge_beam(footprint_x, ridge_z,
                     rafter_depth=RAFTER_PROFILE[1], pitch_deg=ROOF_PITCH_DEG):
    """The GL28h *rygås*, carried by the rafter pairs and nothing else — no column under
    it, no wall for it to land on. Its top is set ``BEAM_EMBED`` ABOVE the rafters'
    underside at the apex, so it is housed that far into every pair and is FASTENED;
    exactly tangent would be touching, jointless, held by nothing."""
    hx = I(footprint_x / 2)
    under = ridge_z - rafter_depth / math.cos(math.radians(pitch_deg))
    _bw, bd = RIDGE_BEAM_PROFILE
    z_mid = I(under + BEAM_EMBED) - bd // 2
    return Sweep(name="ridge_beam",
                 path=[Point(x=-hx, y=0, z=z_mid), Point(x=hx, y=0, z=z_mid)],
                 material=Material(key=GLULAM, profile_mm=RIDGE_BEAM_PROFILE))


# ══ 6b. THE ROOF COVERING (systems/roof-covering), OPENED FOR THE DORMERS ════
# Everything here is added ``carve="none"``: the covering sits ON the frame and has no
# boolean relationship with it. Left in the inferred-carve pass a Sweep's AABB pad (its
# profile's max reach, ~6.4 m on a full-slope prism) swallows the building and the CSG
# depth goes past 30 — where the viewer silently draws the un-subtracted base solid.
#
# The dormers are handled by SPLITTING, not by cutting. A course that runs into a dormer
# is emitted as two shorter courses either side of it; the underlay becomes three prisms;
# the battens become two laths. That is how a roof is tiled around a dormer on site, and
# it costs zero booleans — the alternative (one full course, then a .void()) would put a
# boolean on every one of the 20 courses it crosses.


def _slope(eave_half, ridge_z, pitch_deg, sign):
    """One roof slope in the (Y, Z) plane: ``(eave_point, up, out, run)``."""
    th = math.radians(pitch_deg)
    eave_z = ridge_z - eave_half * math.tan(th)
    up = (-sign * math.cos(th), math.sin(th))     # eave → ridge
    out = (sign * math.sin(th), math.cos(th))     # away from the roof, upward
    return (sign * eave_half, eave_z), up, out, math.hypot(eave_half, ridge_z - eave_z)


def _course_layout(run, gauge_max=TILE_GAUGE, head_lap=TILE_HEAD_LAP, bed=RIDGE_BED):
    """``(course count, ACTUAL gauge)`` — the batten set-out (*lægtefordeling*).

    **``tile_gauge`` is a MAXIMUM, not a spacing.** A roofer divides the slope into whole
    courses and sets the battens at the resulting gauge; nobody lays 20 courses at exactly
    330 and leaves whatever is left over at the ridge. Fixing the gauge instead leaves a
    random remainder — 251 mm on this roof — and something has to cover it. Mine did not:
    the ridge wings reached 190 mm down the slope, the tile field stopped 251 mm short,
    and 61 mm of bare batten ran the whole length of the ridge. It renders as a gold line
    under the ridge cover, warns about nothing, and is a leak.

    Setting the gauge out closes the roof to within ``bed``, which is the seat the ridge
    cover is supposed to have anyway."""
    usable = run - head_lap - bed
    n = max(1, math.ceil(usable / gauge_max))
    return n, usable / n


def _tile_band_profile(width_mm, ref):
    """One course's whole section: the pantile wave REPEATED across ``width_mm``.

    1762 points per course sounds heavy and is not — the generator's entity cache shares
    it, so every course of the same width references ONE profile entity."""
    n = max(1, int(round(width_mm / TILE_COVER_WIDTH)))
    top, bot = ROOF_TILE_PROFILE[:23], ROOF_TILE_PROFILE[23:]
    pts = []
    for k in range(n):
        run = top if k == 0 else top[1:]
        pts += [(px + k * TILE_COVER_WIDTH, py) for px, py in run]
    for idx, k in enumerate(reversed(range(n))):
        run = bot if idx == 0 else bot[1:]
        pts += [(px + k * TILE_COVER_WIDTH, py) for px, py in run]
    return [Point2D(x=I(px), y=I(ref - py)) for px, py in pts]


def slope_x_segments(sign, s_lo, s_hi, dormers, verge=VERGE_HALF_X,
                     pitch_deg=ROOF_PITCH_DEG):
    """``[(lo, hi, clips)]`` — the x runs a covering band from ``s_lo`` to ``s_hi`` may
    occupy on slope ``sign``, and the planes each run is then scribed to.

    ONE function, so the tiles and the battens cannot hold two different opinions about
    where the dormers are — which is how a course ends up running through a dormer cheek
    while its batten stops correctly short of it.

    In the valley (between ``u_valley`` and ``u_ridge_die``), the segments extend all the
    way to the dormer centerline (``xc``) before clipping. The 3D ``valley_cut_plane`` then
    slices the tile course and batten cleanly, maintaining an exact 50 mm drainage gap to
    the valley centerline with zero wave scalloping."""
    cosP = math.cos(math.radians(pitch_deg))
    y_tail = EAVE_HALF_Y - s_lo * cosP           # nearer the eave: larger |y|
    y_head = EAVE_HALF_Y - s_hi * cosP
    blocked = []
    for d in dormers:
        if d["sign"] != sign:
            continue
        spans = [sp for sp in (dormer_x_span_at(d, y_tail), dormer_x_span_at(d, y_head))
                 if sp is not None]
        # When a course reaches the ridge die point, the dormer footprint tapers to (xc, xc).
        # Include the apex so courses crossing u_ridge_die meet at the ridge rather than
        # leaving an un-tiled gap above the ridge roll.
        if y_tail > d["u_ridge_die"] + 0.5 >= y_head:
            spans.append((d["xc"], d["xc"]))
        if not spans:
            continue
        # Only courses in the valley receive the diagonal cut plane.
        # Below the valley (along the cheek wall), the blocked span is the cheek overhangs.
        # In the valley, courses extend to the centerline (xc) and are sliced by valley_cut_plane.
        has_cut = (y_tail <= d["u_valley"] + 0.5 and y_head < d["u_valley"])
        if not has_cut:
            blocked.append((d["X0"], d["X1"], d))
        else:
            blocked.append((d["xc"], d["xc"], d))
    if not blocked:
        return [(-verge, verge, [])]
    spans_out, cursor, pending = [], -verge, []
    for lo, hi, d in sorted(blocked):
        cuts = [valley_cut_plane(d, "left", pitch_deg=pitch_deg)] if (y_tail <= d["u_valley"] + 0.5 and y_head < d["u_valley"]) else []
        if lo - cursor > TILE_COVER_WIDTH:
            spans_out.append((cursor, lo, pending + cuts))
        cursor = max(cursor, hi)
        pending = [valley_cut_plane(d, "right", pitch_deg=pitch_deg)] if (y_tail <= d["u_valley"] + 0.5 and y_head < d["u_valley"]) else []
    if verge - cursor > TILE_COVER_WIDTH:
        spans_out.append((cursor, verge, pending))
    return spans_out


def valley_cut_plane(g, side, pitch_deg=ROOF_PITCH_DEG, drainage_gap=50.0):
    """``(origin, normal)`` scribing the MAIN roof covering to one of a dormer's valleys.

    The cutting plane is PERPENDICULAR to the main roof surface and parallel to the
    3D valley line, offset by ``drainage_gap`` (50 mm) into the tile field.
    Because its normal lies strictly IN the roof plane (normal · n_roof == 0), the cut
    slices through both the crests and troughs of the corrugated pantiles at the exact
    same line — eliminating wave scalloping and maintaining a uniform 50 mm clearance
    along the entire zinc valley flashing."""
    s = g["sign"]
    P = math.radians(pitch_deg)
    # Main roof outward unit normal:
    # south slope (s = -1): n_roof = (0, -sin P, cos P)
    # north slope (s = 1):  n_roof = (0, sin P, cos P)
    n_roof = (0.0, s * math.sin(P), math.cos(P))

    # 3D valley line: from side eave intersection to ridge die apex
    x_eave = g["X0"] if side == "left" else g["X1"]
    p_eave = (float(x_eave), float(s * g["u_valley"]), float(g["ze"]))
    p_apex = (float(g["xc"]), float(s * g["u_ridge_die"]), float(g["z_ridge"]))

    # Direction vector along valley from eave up to apex:
    v = (p_apex[0] - p_eave[0], p_apex[1] - p_eave[1], p_apex[2] - p_eave[2])
    len_v = math.hypot(v[0], v[1], v[2])
    d_val = (v[0] / len_v, v[1] / len_v, v[2] / len_v)

    # In-plane normal perpendicular to valley line: cross(d_val, n_roof)
    cx = d_val[1] * n_roof[2] - d_val[2] * n_roof[1]
    cy = d_val[2] * n_roof[0] - d_val[0] * n_roof[2]
    cz = d_val[0] * n_roof[1] - d_val[1] * n_roof[0]
    len_c = math.hypot(cx, cy, cz)
    unx, uny, unz = cx / len_c, cy / len_c, cz / len_c

    # Ensure normal points INTO the dormer (the removed half-space):
    # Left valley: dormer is towards +X (xc > X0), so nx must be positive.
    # Right valley: dormer is towards -X (xc < X1), so nx must be negative.
    if (side == "left" and unx < 0) or (side == "right" and unx > 0):
        unx, uny, unz = -unx, -uny, -unz

    # Origin: offset from valley line by drainage_gap away from the dormer into the tile field
    ox = p_eave[0] - drainage_gap * unx
    oy = p_eave[1] - drainage_gap * uny
    oz = p_eave[2] - drainage_gap * unz

    return (Point(x=I(ox), y=I(oy), z=I(oz)), (unx, uny, unz))


def build_roof_tiles(footprint_y, wall_top, dormers,
                     pitch_deg=ROOF_PITCH_DEG, rafter_depth=RAFTER_PROFILE[1],
                     verge=VERGE_HALF_X):
    """ONE ``Roof`` aggregate holding every tile course.

    Courses sweep UP THE SLOPE (the wave repeats ACROSS), each ``tile_gauge +
    tile_head_lap`` long so it laps the course below, and each lifted ``tile_lap_step`` at
    its TAIL so the upper course rests ON TOP of the lower rather than being coincident
    with it. The lift is the SAME on every course, so it does not accumulate up the slope.

    Every segment across split courses is phase-aligned to the master roof tile grid
    (anchored to the gable verge, e.g. ``tile_south_7_1``), ensuring continuous pantile
    waves across the entire roof with zero horizontal or vertical offset."""
    ridge_z = _ridge_z(footprint_y, wall_top, pitch_deg)
    half = rafter_depth / 2.0
    ref = UNDERLAY_T + BATTEN_H + max(y for _, y in ROOF_TILE_PROFILE)
    roof = Roof(name="tiles")
    counts = {}
    cosP = math.cos(math.radians(pitch_deg))

    for sign, tag in ((-1, "south"), (1, "north")):
        (ey, ez), up, out, run = _slope(EAVE_HALF_Y, ridge_z, pitch_deg, sign)
        courses, gauge = _course_layout(run)
        n = 0
        slope_dormers = [d for d in dormers if d["sign"] == sign]

        for k in range(courses):
            s0 = k * gauge
            s1 = s0 + gauge + TILE_HEAD_LAP

            # Find split stations crossing valley start or ridge die point
            split_pts = []
            for d in slope_dormers:
                s_val = (EAVE_HALF_Y - d["u_valley"]) / cosP
                s_rd = (EAVE_HALF_Y - d["u_ridge_die"]) / cosP
                if s0 < s_val < s1 and s0 + gauge > s_val:
                    split_pts.append(s_val)
                if s0 < s_rd < s1 and s0 + gauge > s_rd:
                    split_pts.append(s_rd)

            pts = sorted(set([s0] + split_pts + [s1]))
            sub_ranges = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

            for sub_idx, (sa, sb) in enumerate(sub_ranges):
                ta = (sa - s0) / (s1 - s0)
                tb = (sb - s0) / (s1 - s0)
                d_tail = half + TILE_LAP_STEP * (1.0 - ta)
                d_head = half + TILE_LAP_STEP * (1.0 - tb)
                a = (ey + sa * up[0] + d_tail * out[0], ez + sa * up[1] + d_tail * out[1])
                b = (ey + sb * up[0] + d_head * out[0], ez + sb * up[1] + d_head * out[1])

                for j, (xlo, xhi, cuts) in enumerate(
                        slope_x_segments(sign, sa, sb, dormers, verge=verge, pitch_deg=pitch_deg)):
                    # Master pantile grid phase alignment:
                    # Snap the Sweep starting edge to the global tile grid (set out from the verge)
                    # so that every course and split segment shares the exact wave crest positions.
                    if sign < 0:
                        delta = verge - xhi
                        m = math.floor(delta / TILE_COVER_WIDTH)
                        x_start = verge - m * TILE_COVER_WIDTH
                        w_span = x_start - xlo
                    else:
                        delta = xlo - (-verge)
                        m = math.floor(delta / TILE_COVER_WIDTH)
                        x_start = -verge + m * TILE_COVER_WIDTH
                        w_span = xhi - x_start

                    b_tag = f"tile_{tag}_{k}_{sub_idx}_{j}" if len(sub_ranges) > 1 else f"tile_{tag}_{k}_{j}"
                    band = Sweep(name=b_tag,
                                 profile=_tile_band_profile(w_span, ref),
                                 path=[Point(x=I(x_start), y=I(a[0]), z=I(a[1])),
                                       Point(x=I(x_start), y=I(b[0]), z=I(b[1]))],
                                 material=Material(key=TILE))

                    # If x_start extended beyond segment bounds to lock to the grid,
                    # clip the excess along the cheek wall if no valley cut plane is active:
                    if sign < 0 and x_start > xhi and not cuts:
                        band.clip(origin=Point(x=I(xhi), y=I(a[0]), z=I(a[1])),
                                  normal=(1.0, 0.0, 0.0))
                    elif sign > 0 and x_start < xlo and not cuts:
                        band.clip(origin=Point(x=I(xlo), y=I(a[0]), z=I(a[1])),
                                  normal=(-1.0, 0.0, 0.0))

                    for origin, normal in cuts:
                        band.clip(origin=origin, normal=normal)
                    roof.add(band)
                    n += 1

            # Clay hanging hook (ophængsknast / hage) resting on and catching behind the cross spanner:
            # Each course k hangs on Batten k+1 at s_batten = (k + 1) * gauge. The hook sits on the up-slope
            # face of the batten (s_batten + 19 mm), projecting 18 mm downward towards the underlay with 7 mm
            # clearance. The batten sits 2/3 down the tile span from the head.
            s_batten = (k + 1) * gauge
            s_up = s_batten + BATTEN_W / 2.0
            t_hook = 15.0   # mm thick along the slope (clay lug thickness)
            h_hook = 18.0   # mm deep normal to the roof (leaves 7 mm clearance to underlay)
            s_mid = s_up + t_hook / 2.0
            d_mid = half + UNDERLAY_T + BATTEN_H - h_hook / 2.0
            hy = ey + s_mid * up[0] + d_mid * out[0]
            hz = ez + s_mid * up[1] + d_mid * out[1]
            hw_hook, hh_hook = t_hook / 2.0, h_hook / 2.0
            hook_prof = [Point2D(x=I(a), y=I(b)) for a, b in _ccw(
                [(su * hw_hook * up[0] + sh * hh_hook * out[0],
                  su * hw_hook * up[1] + sh * hh_hook * out[1])
                 for su, sh in ((-1, -1), (1, -1), (1, 1), (-1, 1))])]

            for j, (xlo, xhi, cuts) in enumerate(
                    slope_x_segments(sign, s_batten, s_batten, dormers, verge=verge, pitch_deg=pitch_deg)):
                x0 = xlo + BATTEN_VERGE_INSET if xlo <= -verge else xlo
                x1 = xhi - BATTEN_VERGE_INSET if xhi >= verge else xhi
                hook = Sweep(name=f"hook_{tag}_{k}_{j}",
                             profile=hook_prof,
                             path=[Point(x=I(x0), y=I(hy), z=I(hz)),
                                   Point(x=I(x1), y=I(hy), z=I(hz))],
                             material=Material(key=TILE))
                for origin, normal in cuts:
                    hook.clip(origin=origin, normal=normal)
                roof.add(hook)

        counts[tag] = (courses, n)
        print(f"INFO covering: {tag} slope run {run:.0f} mm -> {courses} courses set out "
              f"at {gauge:.0f} mm (max {TILE_GAUGE}), emitted as {n} bands (split around "
              f"the dormer), top course {RIDGE_BED} mm short of the apex")
    return roof, counts


def build_roof_battens(footprint_y, wall_top, dormers,
                       pitch_deg=ROOF_PITCH_DEG, rafter_depth=RAFTER_PROFILE[1]):
    """One 38x25 lægte per course, on the underlay — the rung the tiles actually hang on,
    and the only thing ``tile_gauge`` really is. Off the Timber_C24 stock list (all 45
    wide) on purpose: a batten is a sawn lath, so it takes an explicit ``profile=``."""
    ridge_z = _ridge_z(footprint_y, wall_top, pitch_deg)
    half = rafter_depth / 2.0
    battens = []
    for sign, tag in ((-1, "south"), (1, "north")):
        (ey, ez), up, out, run = _slope(EAVE_HALF_Y, ridge_z, pitch_deg, sign)
        d = half + UNDERLAY_T + BATTEN_H / 2.0
        hw, hh = BATTEN_W / 2.0, BATTEN_H / 2.0
        prof = [Point2D(x=I(a), y=I(b)) for a, b in _ccw(
            [(su * hw * up[0] + sh * hh * out[0], su * hw * up[1] + sh * hh * out[1])
             for su, sh in ((-1, -1), (1, -1), (1, 1), (-1, 1))])]
        courses, gauge = _course_layout(run)
        for k in range(1, courses + 1):
            s = k * gauge
            cy, cz = ey + s * up[0] + d * out[0], ez + s * up[1] + d * out[1]
            for j, (xlo, xhi, cuts) in enumerate(
                    slope_x_segments(sign, s, s, dormers)):
                # Hold the ends inboard of the raking cornice so corbelled brick, not raw
                # batten ends, reads as the verge edge.
                x0 = xlo + BATTEN_VERGE_INSET if xlo <= -VERGE_HALF_X else xlo
                x1 = xhi - BATTEN_VERGE_INSET if xhi >= VERGE_HALF_X else xhi
                lath = Sweep(name=f"batten_{tag}_{k}_{j}", profile=prof,
                             path=[Point(x=I(x0), y=I(cy), z=I(cz)),
                                   Point(x=I(x1), y=I(cy), z=I(cz))],
                             material=Material(key=BATTEN))
                for origin, normal in cuts:
                    lath.clip(origin=origin, normal=normal)
                battens.append(lath)
    print(f"INFO covering: {len(battens)} battens ({BATTEN_W}x{BATTEN_H} {BATTEN} lægter) "
          f"at {TILE_GAUGE} mm gauge")
    return battens


def _slope_prism(ridge_z, pitch_deg, sign, d_inner, d_outer, name,
                 material, x0, x1, s_lo=0.0, s_hi=None):
    """One placed band of the buildup on a single slope, between two up-slope stations."""
    (ey, ez), up, out, run = _slope(EAVE_HALF_Y, ridge_z, pitch_deg, sign)
    a, b = s_lo, run if s_hi is None else min(run, s_hi)
    quad = [(s * up[0] + d * out[0], s * up[1] + d * out[1])
            for s, d in ((a, d_inner), (b, d_inner), (b, d_outer), (a, d_outer))]
    return Sweep(name=name,
                 profile=[Point2D(x=I(p), y=I(q)) for p, q in _ccw(quad)],
                 path=[Point(x=I(x0), y=I(ey), z=I(ez)), Point(x=I(x1), y=I(ey), z=I(ez))],
                 material=material)


def build_roof_underlay(footprint_y, wall_top, dormers,
                        pitch_deg=ROOF_PITCH_DEG, rafter_depth=RAFTER_PROFILE[1]):
    """The wet barrier (undertag) on the rafters' upper face, stopped exactly where the top
    tile course reaches so no membrane pokes into the bare ridge gap, and split around each
    dormer the same way the courses are.

    Includes a ≥ 150 mm vertical upstand (opkant) along the dormer cheek walls on both sides
    to provide a watertight seal against the dormer sides."""
    ridge_z = _ridge_z(footprint_y, wall_top, pitch_deg)
    half = rafter_depth / 2.0
    tP = math.tan(math.radians(pitch_deg))
    cosP = math.cos(math.radians(pitch_deg))
    lift_underlay = (half + UNDERLAY_T) / cosP
    out = []
    for sign, tag in ((-1, "south"), (1, "north")):
        (_ep, _up, _out, run) = _slope(EAVE_HALF_Y, ridge_z, pitch_deg, sign)
        courses, gauge = _course_layout(run)
        top_reach = courses * gauge + TILE_HEAD_LAP
        blockers = [d for d in dormers if d["sign"] == sign]
        if not blockers:
            out.append(_slope_prism(ridge_z, pitch_deg, sign, half, half + UNDERLAY_T,
                                    f"underlay_{tag}_0", Material(key=UNDERLAY),
                                    -VERGE_HALF_X, VERGE_HALF_X, s_hi=top_reach))
        else:
            cursor = -VERGE_HALF_X
            for idx, d in enumerate(sorted(blockers, key=lambda b: b["x0"])):
                out.append(_slope_prism(ridge_z, pitch_deg, sign, half, half + UNDERLAY_T,
                                        f"underlay_{tag}_{idx}", Material(key=UNDERLAY),
                                        cursor, d["x0"], s_hi=top_reach))
                cursor = max(cursor, d["x1"])
                out.append(_slope_prism(ridge_z, pitch_deg, sign, half, half + UNDERLAY_T,
                                        f"underlay_{tag}_over_{d['tag']}", Material(key=UNDERLAY),
                                        d["x0"], d["x1"], s_lo=d["s_back"], s_hi=top_reach))
                # 150 mm vertical upstand (opkant) tight along cheek walls
                upstand_h = 150
                u0, u1 = d["abs_y_front"], d["u_cheek_back"]
                z0 = ridge_z - u0 * tP + lift_underlay
                z1 = ridge_z - u1 * tP + lift_underlay
                s = d["sign"]
                # Left cheek upstand at x = d["x0"] (outward normal -X)
                p0_l = Point(x=I(d["x0"]), y=I(s * u0), z=I(z0))
                p1_l = Point(x=I(d["x0"]), y=I(s * u1), z=I(z1))
                p2_l = Point(x=I(d["x0"]), y=I(s * u1), z=I(z1 + upstand_h))
                p3_l = Point(x=I(d["x0"]), y=I(s * u0), z=I(z0 + upstand_h))
                out.append(Extrude(name=f"underlay_upstand_left_{d['tag']}",
                                   contour=[p0_l, p3_l, p2_l, p1_l],
                                   material=Material(key=UNDERLAY, thickness_mm=UNDERLAY_T)))
                # Right cheek upstand at x = d["x1"] (outward normal +X)
                p0_r = Point(x=I(d["x1"]), y=I(s * u0), z=I(z0))
                p1_r = Point(x=I(d["x1"]), y=I(s * u1), z=I(z1))
                p2_r = Point(x=I(d["x1"]), y=I(s * u1), z=I(z1 + upstand_h))
                p3_r = Point(x=I(d["x1"]), y=I(s * u0), z=I(z0 + upstand_h))
                out.append(Extrude(name=f"underlay_upstand_right_{d['tag']}",
                                   contour=[p0_r, p1_r, p2_r, p3_r],
                                   material=Material(key=UNDERLAY, thickness_mm=UNDERLAY_T)))
            if VERGE_HALF_X - cursor > 0:
                out.append(_slope_prism(ridge_z, pitch_deg, sign, half, half + UNDERLAY_T,
                                        f"underlay_{tag}_{len(blockers)}", Material(key=UNDERLAY),
                                        cursor, VERGE_HALF_X, s_hi=top_reach))
    return out


def build_roof_insulation(footprint_x, footprint_y, wall_top, dormers,
                          spacing=RAFTER_SPACING, pitch_deg=ROOF_PITCH_DEG,
                          rafter_depth=RAFTER_PROFILE[1]):
    """Mineral wool between the rafters, ONE slab per bay per slope — never one slab per
    slope. A full-slope slab's AABB encloses the entire frame, so displacement carves it
    with every collar, post, beam, plate and wall in that volume: 81 booleans on one
    element, measured. Per-bay slabs need no carve at all, and are how batts are cut."""
    ridge_z = _ridge_z(footprint_y, wall_top, pitch_deg)
    half = rafter_depth / 2.0
    xs = _bay_xs(footprint_x, spacing)
    out = []
    for sign, tag in ((-1, "south"), (1, "north")):
        (_ep, _up, _out, run) = _slope(EAVE_HALF_Y, ridge_z, pitch_deg, sign)
        for i, (xa, xb) in enumerate(zip(xs, xs[1:])):
            x0 = xa + RAFTER_PROFILE[0] // 2
            x1 = xb - RAFTER_PROFILE[0] // 2
            s_lo = 0.0
            for d in dormers:                    # a bay inside a dormer starts at its back
                if d["sign"] == sign and d["x0"] <= x0 and x1 <= d["x1"]:
                    s_lo = d["s_back"]
            out.append(_slope_prism(ridge_z, pitch_deg, sign,
                                    half - ROOF_INSULATION_T, half,
                                    f"roof_insulation_{tag}_{i}",
                                    Material(key=ROOF_INSULATION), x0, x1, s_lo=s_lo))
    print(f"INFO covering: {len(out)} insulation batts ({ROOF_INSULATION} "
          f"{ROOF_INSULATION_T} mm, {rafter_depth - ROOF_INSULATION_T} mm vent gap)")
    return out


# ── THE RIDGE (rygning) — a section CUT BY THE PITCH, not drawn to look right ──

def build_ridge(footprint_y, wall_top, pitch_deg=ROOF_PITCH_DEG,
                rafter_depth=RAFTER_PROFILE[1]):
    """The ridge cover: one clay prism along the ridge whose two wings lie FLAT on the two
    tile fields.

    Everything in the section is derived from ``pitch_deg`` and from where the tile
    surface actually is (``t_out``, the perpendicular reach from the rafter centreline out
    past underlay, batten and tile), so changing the pitch RE-CUTS the ridge. That is the
    whole difference between this and the placeholder tent the gable skill shipped: a tent
    with hand-picked half-width and drop looks right at 45° and hovers, or bites, at any
    other pitch.

    For a path along +X the compiler maps profile x to world +Y and profile y to world +Z,
    so the section below is literally (Δy, Δz) from the ridge line — no rotation maths.

    The one thing that must hold and is not geometry: the wing tips have to reach PAST the
    top tile course's head, or the cover seats on underlay and the ridge leaks. The tile
    field always stops short of the apex (a course is a whole ``tile_gauge``), so the
    overlap is asserted rather than assumed."""
    ridge_z = _ridge_z(footprint_y, wall_top, pitch_deg)
    th = math.radians(pitch_deg)
    half = rafter_depth / 2.0
    t_out = half + UNDERLAY_T + BATTEN_H + max(y for _, y in ROOF_TILE_PROFILE) \
        - min(y for _, y in ROOF_TILE_PROFILE)          # rafter centreline → tile surface

    # A wing tip: RIDGE_WING down the slope from the apex, sitting on the tile surface.
    w_y = I(RIDGE_WING * math.cos(th) + t_out * math.sin(th))
    w_z = I(-RIDGE_WING * math.sin(th) + t_out * math.cos(th))
    apex_in = I(t_out / math.cos(th))                   # where the two tile planes cross
    apex_out = apex_in + RIDGE_RISE

    # THE CHECK, and it is measured ALONG THE SLOPE — which is the whole trap. Compare a
    # wing tip's Y (which carries the covering's own 188 mm perpendicular reach) against a
    # shortfall measured up the slope and the numbers look comfortable while the cover
    # hangs 89 mm clear of the tile it is supposed to lap. Both sides must be the same
    # quantity: how far down the slope from the apex.
    (_ep, _up, _out, run) = _slope(EAVE_HALF_Y, ridge_z, pitch_deg, -1)
    courses, gauge = _course_layout(run)
    short_s = run - (courses * gauge + TILE_HEAD_LAP)
    if RIDGE_WING <= short_s + 100:
        raise AssertionError(
            f"the ridge wings reach {RIDGE_WING} mm down the slope but the tile field "
            f"stops {short_s:.0f} mm short of the apex — that leaves under 100 mm of lap, "
            f"and the cover would seat on bare batten. Raise RIDGE_WING or lower RIDGE_BED")

    contour = [(-w_y, w_z + RIDGE_SHELL), (0, apex_out), (w_y, w_z + RIDGE_SHELL),
               (w_y, w_z), (0, apex_in), (-w_y, w_z)]
    ridge = Sweep(name="ridge",
                  profile=[Point2D(x=I(a), y=I(b)) for a, b in _ccw(contour)],
                  path=[Point(x=-VERGE_HALF_X, y=0, z=ridge_z),
                        Point(x=VERGE_HALF_X, y=0, z=ridge_z)],
                  material=Material(key=TILE))
    print(f"INFO covering: ridge (rygning) wings {RIDGE_WING} mm down each slope at "
          f"{pitch_deg}° (±{w_y} mm in Y), crown {apex_out} mm over the ridge line, "
          f"lapping the top course by {RIDGE_WING - short_s:.0f} mm")
    return ridge


# ══ 6c. GABLE ENDS + MASONRY CORBELS (systems/gable-detail) ══════════════════

def _gable_triangle(sx, x_out, thickness, gy, wall_top, apex_z, material, name):
    """One triangular gable layer: an ``Extrude`` of the gable triangle in the YZ plane at
    the layer's OUTER face, extruded ``thickness`` INBOARD. Newell's normal for the forward
    winding is +X, so the winding reverses for the east gable to run both inboard."""
    p_left = Point(x=x_out, y=-gy, z=wall_top)
    p_right = Point(x=x_out, y=gy, z=wall_top)
    p_apex = Point(x=x_out, y=0, z=apex_z)
    contour = [p_left, p_right, p_apex] if sx < 0 else [p_left, p_apex, p_right]
    return Extrude(name=name, contour=contour, thickness=thickness, material=material)


def build_gable_ends(footprint_x, footprint_y, wall_top, pitch_deg=ROOF_PITCH_DEG):
    """The two gable-end containers (west x=−hx, east x=+hx), each holding its own sandwich.

    At roof level the E/W walls are NOT load-bearing, so the concrete core stops at
    ``wall_top`` and the triangle is a light brick + 30 mm cavity + insulation + stud +
    drywall wall instead.

    **Why a container, and why ``ifc_class="IfcWall"``.** A gable is a wall, and saying so
    gives it the WALL frame: local +x runs the gable span, +y goes INTO the sandwich from
    the outer brick face, +z is up from ``wall_top``. That frame is what lets the corbels
    be authored once and ANCHORED, and — new here — what lets a gable WINDOW be cut with
    one ``gable.opening()`` in exactly the same opening-local coordinates every other
    window in this house uses.

    The internal layers stop ``GABLE_APEX_CLAMP`` below the brick's apex so the brick
    MASKS them from outside; nothing green or tan pokes over the rake."""
    hx, hy = I(footprint_x / 2), I(footprint_y / 2)
    gy = hy + EXT_SANDWICH_MM
    ridge_z = _ridge_z(footprint_y, wall_top, pitch_deg)
    gables = []
    for sx, tag in ((-1, "west"), (1, "east")):
        brick_out = sx * (hx + EXT_SANDWICH_MM)
        insul_out = brick_out - sx * (GABLE_BRICK_MM + GABLE_CAVITY_MM)
        dry_out = insul_out - sx * GABLE_INSUL_MM
        inner_apex = ridge_z - GABLE_APEX_CLAMP

        gable = Element(ifc_class="IfcWall", name=f"gable_{tag}")
        gable.add(
            _gable_triangle(sx, brick_out, GABLE_BRICK_MM, gy, wall_top, ridge_z,
                            Material(key=GABLE_BRICK), "brick"),
            _gable_triangle(sx, insul_out, GABLE_INSUL_MM, gy, wall_top, inner_apex,
                            Material(key=GABLE_INSUL), "insul"),
            _gable_triangle(sx, dry_out, GABLE_DRYWALL_MM, gy, wall_top, inner_apex,
                            Material(key=GABLE_DRYWALL), "drywall"),
        )
        x_stud = insul_out - sx * (GABLE_INSUL_MM / 2.0)
        n = max(1, int((2 * gy) // GABLE_STUD_SPACING))
        for j in range(n + 1):
            y = -gy + I(2 * gy * j / n)
            top = ridge_z - abs(y) - GABLE_APEX_CLAMP
            if top - wall_top < 200:
                continue
            gable.add(Sweep(name=f"stud_{j}",
                            path=[Point(x=I(x_stud), y=y, z=wall_top),
                                  Point(x=I(x_stud), y=y, z=top)],
                            material=Material(key=TIMBER, profile_mm=GABLE_STUD_PROFILE)))
        print(f"INFO gable: {tag} end — brick {GABLE_BRICK_MM} + cavity {GABLE_CAVITY_MM} + "
              f"insul {GABLE_INSUL_MM} + studs + drywall {GABLE_DRYWALL_MM}, "
              f"outer face x={brick_out}")
        gables.append(gable)
    return gables


def _rake_height_at(along, gy, rise):
    """How tall the gable triangle is at ``along`` mm from the left rake foot."""
    return rise - abs(along - gy)


def place_gable_windows(gables, product, footprint_y, wall_top, offsets,
                        pitch_deg=ROOF_PITCH_DEG, up=GABLE_WIN_UP,
                        width=GABLE_WIN_W, height=GABLE_WIN_H):
    """Cut the loft's gavlvinduer straight through the gable sandwich.

    ONE call per window. The gable is a container whose geometry lives in its parts, so the
    hole is DISTRIBUTED — brick triangle, insulation triangle, drywall triangle and every
    stud standing in it each get an ``IfcOpeningElement``, and ONE ``IfcWindow`` fills the
    outermost. This is the follow-up the gable skill left open ("no gable window ... mount
    it through the stud frame with the window opening machinery").

    Each opening is checked against the RAKE, because the gable is a triangle and a window
    that fits at mid-span pokes out through the sloping edge nearer a foot. The check is
    the reason ``offsets`` are offsets from the gable's middle rather than absolute
    positions: move the middle and the clearance moves with it."""
    gy = I(footprint_y / 2) + EXT_SANDWICH_MM
    rise = _ridge_z(footprint_y, wall_top, pitch_deg) - wall_top
    placed = 0
    for gable in gables:
        tag = gable.name.rsplit("_", 1)[-1]
        for k, off in enumerate(offsets):
            along = gy + off
            head = up + height
            clear = _rake_height_at(along + width // 2, gy, rise)
            clear = min(clear, _rake_height_at(along - width // 2, gy, rise))
            if head + 200 > clear:
                raise AssertionError(
                    f"gable {tag} window {k}: head at {head} mm over wall_top but the rake "
                    f"is only {clear} mm tall at that end of the opening — it would break "
                    f"through the sloping gable edge")
            gable.opening(product.occurrence(name=f"gavl_{k}"),
                          along_center=I(along), up=up)
            placed += 1
    print(f"INFO window: {placed} gable windows ({width}x{height}) cut through the gable "
          f"sandwich, cill {up} mm over wall_top")
    return gables


def _raking_cornice(name, reach, gy, rise):
    """One gavlgesims — a 3-course stepped brick band up one rake, in the gable's own
    frame: the band sits at y=0 and steps PROUD into negative y, so the gable's outward
    direction never appears as a number. ``reach`` is the one genuine handedness — a
    gable's two rakes climb TOWARD each other."""
    foot_x, apex_x = (0, gy) if reach > 0 else (2 * gy, gy)
    cornice = Element(ifc_class="IfcBuildingElementProxy", name=name)
    for i, proud in enumerate(CORBEL_PROUDS):
        y_lo, y_hi = (i - 1.5) * CORBEL_COURSE_H, (i - 0.5) * CORBEL_COURSE_H
        quad = _ccw([(0, y_lo), (-proud, y_lo), (-proud, y_hi), (0, y_hi)])
        cornice.add(Sweep(name=f"course_{i}",
                          profile=[Point2D(x=I(a), y=I(b)) for a, b in quad],
                          path=[Point(x=I(foot_x), y=0, z=0),
                                Point(x=I(apex_x), y=0, z=I(rise))],
                          material=Material(key=CORBEL_BRICK)))
    return cornice


def _kneestone(name, reach):
    """One 2-over-1 knægt at a rake foot, in the gable's own frame: proud is simply
    negative y, and the two courses stack about z=0."""
    H, u = CORBEL_COURSE_H, 115
    rows = [(-H, 0, 100, 0, I(1.4 * u)), (0, H, 95, I(0.9 * u), I(3.0 * u))]
    bracket = Element(ifc_class="IfcBuildingElementProxy", name=name)
    for i, (z0, z1, proud, near, far) in enumerate(rows):
        x0, x1 = (near, far) if reach > 0 else (-far, -near)
        bracket.add(Box(name=f"course_{i}", start=Point(x=x0, y=-proud, z=z0),
                        end=Point(x=x1, y=0, z=z1), material=CORBEL_BRICK))
    return bracket


def build_corbels(gables, facade, footprint_x, footprint_y, base_z, wall_top,
                  pitch_deg=ROOF_PITCH_DEG):
    """The three red-brick corbel families, all anchored into a frame rather than authored
    in world coordinates: 4 raking cornices and 4 kneestones into the two gables, 2 eaves
    cornices into the S/N facade leaves. Named ``_left``/``_right`` as seen from outside —
    ``along=0`` is the left-hand foot on EVERY host, so a world-direction name would be
    right on one gable and wrong on the other."""
    hy = I(footprint_y / 2)
    gy = hy + EXT_SANDWICH_MM
    rise = _ridge_z(footprint_y, wall_top, pitch_deg) - wall_top
    for gable in gables:
        tag = gable.name.rsplit("_", 1)[-1]
        gable.anchor(_raking_cornice(f"gavlgesims_{tag}_left", 1, gy, rise), along=0)
        gable.anchor(_raking_cornice(f"gavlgesims_{tag}_right", -1, gy, rise), along=0)
        gable.anchor(_kneestone(f"knaegt_{tag}_left", reach=1), along=0)
        gable.anchor(_kneestone(f"knaegt_{tag}_right", reach=-1), along=2 * gy)

    th = math.radians(pitch_deg)
    face_y = hy + EXT_SANDWICH_MM
    underside = _ridge_z(footprint_y, wall_top, pitch_deg) - face_y \
        - I((RAFTER_PROFILE[1] / 2.0) / math.cos(th))
    span = footprint_x + 2 * EXT_SANDWICH_MM
    for side in ("south", "north"):
        leaf = facade[side]
        for i, proud in enumerate(CORBEL_PROUDS):
            top = underside - 60 - i * CORBEL_COURSE_H
            leaf.anchor(Box(name=f"murgesims_{i}", start=Point(x=0, y=-proud, z=0),
                            end=Point(x=span, y=0, z=CORBEL_COURSE_H),
                            material=CORBEL_BRICK),
                        along=0, up=top - CORBEL_COURSE_H - base_z)
    print(f"INFO corbel: 4 gavlgesims + 4 knægte anchored into the gable frames, "
          f"2 murgesims anchored into the S/N facade leaves, {len(CORBEL_PROUDS)} "
          f"courses each stepping {CORBEL_PROUDS} mm proud")
    return gables


# ══ 6d. THE DORMERS (kviste) — the only move that ADDS loft floor area ═══════
#
# A hip REMOVES volume; a dormer adds it. This is the one that makes the ½ storey a
# storey: it lifts the ceiling over a band of the loft from the 45° rake to a flat soffit,
# and it puts a window at eye level where the rake would otherwise be at knee level.
#
# Everything is derived from ``dormer_geometry`` — the frame trimmed its rafters to the
# same dict, and the covering split its courses on it, so there is one dormer envelope in
# this file and three consumers of it, not three descriptions that must agree.


def _yz_normal_x(pts):
    """Newell's X-component for a contour drawn in a plane of constant x, as (y, z)."""
    return sum((a[0] - b[0]) * (a[1] + b[1]) for a, b in zip(pts, pts[1:] + pts[:1]))


def _face_up(pts):
    """The UPWARD unit normal of a planar contour, and the fact that decides this file.

    **A sloped ``Extrude`` always thickens UPWARD, whatever its winding.** Probed, not
    assumed: reverse the contour of a pitched face and ``extrude_offset`` returns the same
    ``(0, −54.6, +182)``; reverse a VERTICAL face and it flips clean from ``+200`` to
    ``−200``. So the winding rule the gable triangles rely on (they are vertical) does NOT
    carry to a roof plane — the compiler canonicalises the non-vertical case.

    That is why a layered roof face is stacked by OFFSETTING each layer's contour DOWN by
    what sits above it, rather than by turning the winding over. Do it the other way and
    the 22 mm boarding extrudes up THROUGH the 1 mm zinc: the roof renders as bare
    chipboard, no warning anywhere, and every dimension is still exactly right."""
    nx = ny = nz = 0.0
    for a, b in zip(pts, pts[1:] + pts[:1]):
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    n = (nx / ln, ny / ln, nz / ln)
    return n if n[2] >= 0 else (-n[0], -n[1], -n[2])


def _offset(pts, n, d):
    return [(p[0] + n[0] * d, p[1] + n[1] * d, p[2] + n[2] * d) for p in pts]


def build_dormer(g, window_product, footprint_y, wall_top,
                 pitch_deg=ROOF_PITCH_DEG, dormer_pitch_deg=DORMER_PITCH_DEG):
    """One *valmet kvist* — front wall with 2-3 windows, two cheeks, a hipped roof.

    THREE GENUINELY DIFFERENT PLANES, and checking that is not paranoia. The first version
    of this file built a "hip" as a front trapezoid plus two end triangles sharing a level
    back line — and every one of the three came out with normal ``(0, −0.309, 0.951)``.
    They were COPLANAR: one flat roof with two decorative seams, which renders as a shed
    dormer, validates perfectly, and is described in prose as hipped. ``dormer_geometry``
    explains the fix (the ridge runs front-to-back, not side-to-side); this function's job
    is to build it, and ``assert_the_roof_is_really_hipped`` is what stops it regressing.

    THE BUILDUP, water side in: clay tiles on battens, over the zinc, over the boarding.
    At ``dormer_pitch`` the tile is below its own minimum pitch, so here the tile is the
    FINISH and the zinc under it is the WATER BARRIER. That is not a compromise — it is
    how a shallow dormer on a tiled house is detailed, and it is why the metal stays."""
    s, tag = g["sign"], g["tag"]
    y_f = g["abs_y_front"]
    x0, x1 = g["x0"], g["x1"]
    dormer = Element(ifc_class="IfcBuildingElementProxy", name=f"dormer_{tag}")

    # ── front wall: outer face flush with the dormer's front plane ───────────
    fy0, fy1 = sorted((s * y_f, s * (y_f - DORMER_FRONT_T)))
    front = Wall(name="front",
                 props={"Pset_WallCommon": {"LoadBearing": False, "IsExternal": True}})
    front.add(Box(name="body", start=Point(x=I(x0), y=I(fy0), z=I(g["base_z"])),
                  end=Point(x=I(x1), y=I(fy1), z=I(g["head_z"]))))
    front.layers = LayerSet(name="dormer_front", outward=(0, s, 0),
                            layers=[Material(key=DORMER_METAL, thickness_mm=DORMER_ZINC_T),
                                    Material(key=DORMER_INSUL, thickness_mm=DORMER_INSUL_T),
                                    Material(key=GABLE_DRYWALL,
                                             thickness_mm=DORMER_LINING_T)])
    dormer.add(front)

    # ── cheeks: a TRIANGLE each. Its two long edges ARE the main roof line and the
    #    dormer roof line, so a cheek cannot drift from either. A cheek is VERTICAL,
    #    so here the winding DOES decide the extrusion direction — see _face_up.
    for side, xc, into in (("left", x0, 1), ("right", x1, -1)):
        # Front face foot (on the main roof) → front wall top → back along the LEVEL side
        # eave until the main roof rises to meet it. All three corners are on something
        # that already exists, so a cheek cannot drift from the roof or from the wall.
        tri = [(s * y_f, g["foot_z"]), (s * y_f, g["head_z"]),
               (s * g["u_cheek_back"], g["head_z"])]
        if (_yz_normal_x(tri) > 0) != (into > 0):
            tri = list(reversed(tri))
        dormer.add(Extrude(name=f"cheek_{side}",
                           contour=[Point(x=I(xc), y=I(a), z=I(b)) for a, b in tri],
                           thickness=DORMER_CHEEK_T,
                           material=Material(key=DORMER_INSUL)))
        dormer.add(Extrude(name=f"cheek_{side}_skin",
                           contour=[Point(x=I(xc - into * DORMER_ZINC_T), y=I(a), z=I(b))
                                    for a, b in tri],
                           thickness=DORMER_ZINC_T, material=Material(key=DORMER_METAL)))

    # ── the hipped roof: a front HIP END + two side planes flanking the ridge ──
    for name, pts in dormer_roof_faces(g):
        # `pts` is the FINISHED outer surface — the plane the tiles' battens sit on.
        # Every layer thickens UPWARD whatever its winding (see _face_up), so each is
        # dropped by what covers it: the zinc by its own gauge, the boarding by both.
        n = _face_up(pts)
        for layer, mat, t, drop in (
                ("metal", DORMER_METAL, DORMER_METAL_T, DORMER_METAL_T),
                ("board", DORMER_BOARD, DORMER_ROOF_T, DORMER_METAL_T + DORMER_ROOF_T)):
            dormer.add(Extrude(name=f"{name}_{layer}",
                               contour=[Point(x=I(p[0]), y=I(p[1]), z=I(p[2]))
                                        for p in _offset(pts, n, -drop)],
                               material=Material(key=mat, thickness_mm=t)))

    # ── the tiles OVER that zinc, plus the clay rolls that cap the arrises ──
    for e_ in build_dormer_tiles(g):
        dormer.add(e_)

    # ── the flashings the split main roof needs ────────────────────────────
    ridge_z = _ridge_z(footprint_y, wall_top, pitch_deg)
    (_ep, _up, _out, run) = _slope(EAVE_HALF_Y, ridge_z, pitch_deg, s)
    half = RAFTER_PROFILE[1] / 2.0
    d_in = half + UNDERLAY_T + BATTEN_H
    # FRONT apron (inddækning): the main-roof courses are split from the EAVE up, so the
    # eave overhang below the dormer's foot would otherwise show bare trimming-rafter feet.
    dormer.add(_slope_prism(ridge_z, pitch_deg, s, d_in, d_in + 4, "front_apron",
                            Material(key=DORMER_METAL), g["X0"], g["X1"],
                            s_lo=0.0, s_hi=g["s_front"] + 120))
    # VALLEY flashings: with the split following the dormer's real (pinched) footprint,
    # the bare strip is not a level band at the back — it is the two valley lines.
    for e_ in build_dormer_valley_flashings(g, ridge_z, pitch_deg):
        dormer.add(e_)

    # ── the windows, evenly spaced in the front wall ───────────────────────
    for k in range(g["n_win"]):
        centre = x0 + DORMER_PIER + DORMER_WIN_W // 2 + k * (DORMER_WIN_W + DORMER_PIER)
        along = _wall_along_centre(Point(x=I(x0), y=I(fy0), z=0),
                                   Point(x=I(x1), y=I(fy1), z=0),
                                   (1, 0, 0), (0, s, 0), centre)
        front.opening(window_product.occurrence(name=f"kvist_{k}"),
                      along_center=I(along), up=I(g["cill_z"] - g["base_z"]))
    print(f"INFO dormer: {tag} — {g['n_win']} windows, {g['width']} mm wide, "
          f"{g['depth']} mm deep, cill z={g['cill_z']}")
    return dormer


#: One dormer window type, promoted once and placed five times.
# ══ 7. WINDOWS (systems/window) ══════════════════════════════════════════════
#
# OPENING-LOCAL COORDINATES — every wall is the south wall. Stand outside and look at it:
# +x is right (0..width), +y goes INTO the wall from the OUTER face (0..thickness), +z is
# up from the cill. The frame is a pure ROTATION of the south wall's on all four facades,
# nothing mirrored, so one body of code builds a correct window on every leaf — including
# a gable triangle and a dormer front, which is what lets this file reuse it three times.

WINDOW_PROPS = {
    "Pset_WindowCommon": {"IsExternal": True, "ThermalTransmittance": 0.8,
                          "Infiltration": 0.1, "SecurityRating": "RC2"},
    "Pset_DoorWindowGlazingType": {"GlassLayers": 3, "FillGas": "Argon",
                                   "GlassThickness1": PANE_MM, "GlassThickness2": PANE_MM,
                                   "GlassThickness3": PANE_MM, "IsExternal": True},
}


def _karm_band():
    """``(y_out, y_in)`` of the karm as INSETS from the leaf's OUTER face.

    The mount plane is a REGULATION, not a preference: the unit bridges the rigid
    insulation and the brick, and mounting it against the concrete core is forbidden — a
    cold bridge straight to mould and wood rot. Centring the karm on the ventilation
    cavity satisfies it by construction, and the raises below keep it satisfied when
    someone edits a layer thickness. Summed from the layers, never read off
    ``EXT_SANDWICH_MM``: this function exists to FAIL when a layer changes, and a stale
    total would keep passing."""
    total = EXT_BRICK_MM + EXT_CAVITY_MM + EXT_INSULATION_MM
    y_brick_in = EXT_BRICK_MM
    y_insul_out = y_brick_in + EXT_CAVITY_MM
    y_mid = (y_brick_in + y_insul_out) // 2
    y_out, y_in = y_mid - KARM_DEPTH // 2, y_mid + KARM_DEPTH // 2
    if not (y_out < y_brick_in and y_in > y_insul_out):
        raise ValueError(
            f"window mount violates the fenestration rule: a {KARM_DEPTH} mm karm at "
            f"y={y_mid} spans [{y_out}, {y_in}] and does not bridge the brick "
            f"(< {y_brick_in}) and the insulation (> {y_insul_out})")
    if y_in >= total:
        raise ValueError(f"window frame reaches the concrete core (y_in={y_in}) — "
                         f"forbidden: a cold bridge to mould and wood rot")
    return y_out, y_in


def _light_band(wall_t):
    """``(y_out, y_in)`` for a window in a LIGHT wall — a dormer front, a partition.

    There is no cavity to centre on and no concrete to keep away from, so the rule the
    brick sandwich obeys does not apply and pretending it does would refuse a perfectly
    good dormer window. What still has to hold is that the frame lives INSIDE the wall."""
    y_out = (wall_t - KARM_DEPTH) // 2
    if y_out < 0:
        raise ValueError(f"a {KARM_DEPTH} mm karm does not fit a {wall_t} mm wall")
    return y_out, y_out + KARM_DEPTH


def _frame_loop(x0, x1, z0, z1, y):
    """A CLOSED centreline rectangle for a frame member sweep.

    Closing the path (first point == last) is what makes this a frame rather than four
    sticks: the compiler recognises a planar closed loop with a rectangular section and
    emits one mitered PRISM per side, ZERO booleans. Four 2-point sweeps give four
    overlapping bars with fouled corners; an explicit ``profile=`` instead of
    ``Material(profile_mm=)`` makes the section non-rectangular and every corner falls
    back to a pair of half-space clips.

    **Never let the frame's outer face land ON the opening boundary.** A karm inset
    exactly half its face width puts its outer offset polygon on the plane of the
    IfcOpeningElement, and the frame tessellates to four sub-millimetre corner slivers —
    measured 0.000360 m³ against a true 0.031406, 1/87 of the timber, with the bounding
    box still exactly right. Only VOLUME catches it. ``FIT_GAP`` = 10 mm is the fugebredde
    a real window is fitted with anyway, so the fix and the correct detail are one number."""
    return [Point(x=I(x0), y=I(y), z=I(z0)), Point(x=I(x1), y=I(y), z=I(z0)),
            Point(x=I(x1), y=I(y), z=I(z1)), Point(x=I(x0), y=I(y), z=I(z1)),
            Point(x=I(x0), y=I(y), z=I(z0))]


def _bar_profile():
    hx, hy = BAR_PROUD // 2, BAR_WIDTH // 2
    return [Point2D(x=-hx, y=-hy), Point2D(x=hx, y=-hy),
            Point2D(x=hx, y=hy), Point2D(x=-hx, y=hy)]


def build_window(width, height, name, band=None):
    """A two-sash outward-opening wooden window, subdivided 2 x 3.

    ``band`` is the one thing this file adds to the system version: the brick sandwich
    computes it from the fenestration rule, a dormer front from its own thickness. The
    JOINERY is identical either way, which is the point — a window is a window, and only
    where it is allowed to sit changes with the wall."""
    if FIT_GAP < 1:
        raise ValueError("FIT_GAP must be at least 1 mm — see _frame_loop")
    y_out, y_in = band if band is not None else _karm_band()
    kh, ph, sh = KARM_FACE // 2, POST_FACE // 2, SASH_FACE // 2

    win = Window(name=name, width=I(width), height=I(height),
                 style="casement", panes=2, props=dict(WINDOW_PROPS))
    y_karm = (y_out + y_in) // 2
    k0 = FIT_GAP + kh
    k1 = FIT_GAP + 2 * kh
    win.add(Sweep(name="karm",
                  material=Material(key=JOINERY, profile_mm=(KARM_DEPTH, KARM_FACE)),
                  path=_frame_loop(k0, width - k0, k0, height - k0, y_karm)))
    xc = width // 2
    win.add(Sweep(name="lodpost",
                  material=Material(key=JOINERY, profile_mm=(POST_DEPTH, POST_FACE)),
                  path=[Point(x=xc, y=y_karm, z=k1),
                        Point(x=xc, y=y_karm, z=height - k1)]))

    y_sash = (y_out + y_out + SASH_DEPTH) // 2
    sashes = [("left", k1 - SASH_LAP, xc - ph + SASH_LAP),
              ("right", xc + ph - SASH_LAP, width - k1 + SASH_LAP)]
    z0_s, z1_s = k1 - SASH_LAP, height - k1 + SASH_LAP

    day_w = min(x1_s - x0_s for _t, x0_s, x1_s in sashes) - 4 * sh
    day_h = (z1_s - z0_s) - 4 * sh
    if day_w < 1 or day_h < 1:
        raise ValueError(
            f"window {name}: {width} x {height} leaves a sash daylight of {day_w} x "
            f"{day_h} mm — the frames overlap; below that it is a single-sash window, "
            f"which this system does not build")

    for tag, x0_s, x1_s in sashes:
        win.add(Sweep(name=f"ramme_{tag}",
                      material=Material(key=JOINERY, profile_mm=(SASH_DEPTH, SASH_FACE)),
                      path=_frame_loop(x0_s + sh, x1_s - sh, z0_s + sh, z1_s - sh, y_sash)))
        gx0, gx1 = x0_s + 2 * sh - GLASS_BITE, x1_s - 2 * sh + GLASS_BITE
        gz0, gz1 = z0_s + 2 * sh - GLASS_BITE, z1_s - 2 * sh + GLASS_BITE
        y_igu_out = y_sash - IGU_MM // 2
        for k in range(3):
            y_pane = y_igu_out + k * (PANE_MM + GAS_GAP_MM) + PANE_MM // 2
            win.add(Extrude(name=f"glass_{tag}_{k}",
                            contour=[Point(x=gx0, y=y_pane, z=gz0),
                                     Point(x=gx1, y=y_pane, z=gz0),
                                     Point(x=gx1, y=y_pane, z=gz1),
                                     Point(x=gx0, y=y_pane, z=gz1)],
                            material=Material(key=GLAZING, thickness_mm=PANE_MM)))
        faces = [("out", y_igu_out - BAR_PROUD // 2),
                 ("in", y_igu_out + IGU_MM + BAR_PROUD // 2)]
        for r in range(1, PANE_ROWS):
            z_bar = gz0 + I((gz1 - gz0) * r / PANE_ROWS)
            for side, y_bar in faces:
                win.add(Sweep(name=f"sprosse_{tag}_{side}_{r}", profile=_bar_profile(),
                              path=[Point(x=gx0, y=y_bar, z=z_bar),
                                    Point(x=gx1, y=y_bar, z=z_bar)],
                              material=Material(key=JOINERY)))
    return win


def _sill_outer_contour(segments=SILL_ARC_SEGMENTS):
    cx, cy = SILL_HEM_CENTRE
    pts = list(SILL_OUTER[:4])
    for i in range(1, segments):
        th = math.radians(180 + 180 * i / segments)
        pts.append((cx + SILL_HEM_RADIUS * math.cos(th), cy + SILL_HEM_RADIUS * math.sin(th)))
    pts.extend(SILL_OUTER[4:])
    return pts


def build_window_sill(width, name):
    """The sålbænk as a ``Sheet``, authored in the WALL'S OWN FRAME — the path runs along
    local +x and the section stands where ``up=``/``inset=`` put it. ``.anchor()`` never
    mirrors, it ROTATES, so standing the section up once is correct on all four sides."""
    prof = [Point2D(x=I(px), y=I(py)) for px, py in _sill_outer_contour()]
    return Sheet(name=name, profile=prof, thickness=SILL_THICKNESS,
                 path=[Point(x=0, y=0, z=0), Point(x=width + 2 * SILL_OVERRUN, y=0, z=0)],
                 material=Material(key=SILL_SHEET))


def place_window(side, product, centre, sill_z, width, name):
    """Cut one opening, fill it, and hang the sålbænk under it.

    ONE call, on the WALL — not one per leaf. The wall is an assembly with no body of its
    own, so ``.opening()`` distributes down to every leaf the hole passes through: the
    cladding gets the void the ``Window`` fills, the core behind it gets the matching
    penetration, and the core's rebar loses its steel where the hole is. Never pair this
    with a second hole through the core — two calls that must agree fail silently in one
    direction, and the wall behind the perfectly rendered unit stays shut."""
    wall, start, end, run, outward = side
    along = I(_wall_along_centre(start, end, run, outward, centre))
    wall.opening(product.occurrence(name=name), along_center=along,
                 up=I(sill_z - start.z))
    wall.anchor(build_window_sill(width, name=f"sill_{name}"), along_center=along,
                up=I(sill_z - SILL_BACK_RISE - start.z), inset=-SILL_PROJECTION)


# ══ 7b. DOORS (systems/door) ═════════════════════════════════════════════════

DOOR_PROPS_EXT = {"Pset_DoorCommon": {"IsExternal": True, "FireRating": "EI2 30",
                                      "ThermalTransmittance": 1.2, "SecurityRating": "RC3"}}


def _door_band(depth, exterior):
    """``(y_out, y_in)`` of a door frame as insets. Exterior doors obey the same
    fenestration rule the windows do; interior frames span the partition."""
    if not exterior:
        return 0, depth
    y_brick_in = EXT_BRICK_MM
    y_insul_out = y_brick_in + EXT_CAVITY_MM
    y_mid = (y_brick_in + y_insul_out) // 2
    y_out, y_in = y_mid - depth // 2, y_mid + depth // 2
    if not (y_out < y_brick_in and y_in > y_insul_out):
        raise ValueError(f"door mount violates the fenestration rule: [{y_out}, {y_in}]")
    if y_in >= EXT_SANDWICH_MM:
        raise ValueError(f"door frame reaches the concrete core (y_in={y_in}) — forbidden")
    return y_out, y_in


def _obox(door, x0, x1, y0, y1, z0, z1, material, name):
    """One opening-local Box: x along the wall, y = INSET from the outer face, z up."""
    door.add(Box(name=name, material=material,
                 start=Point(x=I(min(x0, x1)), y=I(min(y0, y1)), z=I(min(z0, z1))),
                 end=Point(x=I(max(x0, x1)), y=I(max(y0, y1)), z=I(max(z0, z1)))))


def _circle(r, segments=16):
    return [(r * math.cos(2 * math.pi * i / segments),
             r * math.sin(2 * math.pi * i / segments)) for i in range(segments)]


def _add_handle(door, width, y_face, name):
    """The details lever handle on the latch stile. Proud is toward the outside — a
    SMALLER inset — so the rose stands off the leaf and the lever projects further out
    before turning back toward the hinge."""
    xh, zc = width - 90, HANDLE_Z
    y_rose = y_face - ROSE_DEPTH
    door.add(Extrude(name=f"{name}_rose", thickness=ROSE_DEPTH,
                     contour=[Point(x=I(xh + rx), y=I(y_rose), z=I(zc + rz))
                              for rx, rz in _circle(ROSE_RADIUS)],
                     material=Material(key=IRONMONGERY)))
    door.add(Sweep(name=f"{name}_lever", fillet_radius=LEVER_ARC_RADIUS,
                   profile=[Point2D(x=I(px), y=I(py))
                            for px, py in _circle(LEVER_DIAMETER / 2.0)],
                   path=[Point(x=I(xh), y=I(y_rose), z=I(zc)),
                         Point(x=I(xh), y=I(y_rose - LEVER_PROJECTION), z=I(zc)),
                         Point(x=I(xh - LEVER_LENGTH), y=I(y_rose - LEVER_PROJECTION),
                               z=I(zc))],
                   material=Material(key=IRONMONGERY)))


def _add_door_frame(door, width, height, y_out, y_in):
    _obox(door, 0, FRAME_FACE, y_out, y_in, 0, height, WOOD, "jamb_left")
    _obox(door, width - FRAME_FACE, width, y_out, y_in, 0, height, WOOD, "jamb_right")
    _obox(door, 0, width, y_out, y_in, height - FRAME_FACE, height, WOOD, "head")


def build_front_door(width, height, name):
    """Heavy framed-and-panelled front door: 2x3 raised panels, monumental recess. The
    leaf sits BEHIND the frame's outer face, so the frame stands proud of it."""
    door = Door(name=name, width=I(width), height=I(height), style="hinged",
                props=dict(DOOR_PROPS_EXT))
    y_out, y_in = _door_band(EXT_FRAME_DEPTH, exterior=True)
    _add_door_frame(door, width, height, y_out, y_in)
    x0, x1 = FRAME_FACE + LEAF_REVEAL, width - FRAME_FACE - LEAF_REVEAL
    z1 = height - LEAF_REVEAL
    lf_out = y_out + FRONT_INSET
    _obox(door, x0, x1, lf_out, lf_out + FRONT_LEAF_T, 0, z1, WOOD, "leaf")

    g0, g1 = lf_out - GRID_PROUD, lf_out
    xc = width // 2
    xs = [x0, xc - MUNTIN_W // 2, xc + MUNTIN_W // 2, x1]
    zs = [I(z1 * r / FRONT_ROWS) for r in range(FRONT_ROWS + 1)]
    _obox(door, x0, x1, g0, g1, 0, STILE_W, WOOD, "rail_bottom")
    _obox(door, x0, x1, g0, g1, z1 - STILE_W, z1, WOOD, "rail_top")
    _obox(door, x0, x0 + STILE_W, g0, g1, 0, z1, WOOD, "stile_left")
    _obox(door, x1 - STILE_W, x1, g0, g1, 0, z1, WOOD, "stile_right")
    _obox(door, xc - MUNTIN_W // 2, xc + MUNTIN_W // 2, g0, g1, 0, z1, WOOD, "muntin")
    for r in range(1, FRONT_ROWS):
        _obox(door, x0, x1, g0, g1, zs[r] - STILE_W // 2, zs[r] + STILE_W // 2,
              WOOD, f"rail_{r}")
    p0, p1 = lf_out - PANEL_PROUD, lf_out
    for ci, (ax0, ax1) in enumerate([(xs[0] + STILE_W, xs[1]), (xs[2], xs[3] - STILE_W)]):
        for r in range(FRONT_ROWS):
            z0 = zs[r] + (STILE_W // 2 if r else STILE_W) + PANEL_MARGIN
            zt = zs[r + 1] - (STILE_W // 2 if r < FRONT_ROWS - 1 else STILE_W) - PANEL_MARGIN
            if zt - z0 < 60 or (ax1 - PANEL_MARGIN) - (ax0 + PANEL_MARGIN) < 60:
                continue
            _obox(door, ax0 + PANEL_MARGIN, ax1 - PANEL_MARGIN, p0, p1, z0, zt,
                  WOOD, f"panel_{ci}_{r}")
    _add_handle(door, width, lf_out - GRID_PROUD, f"{name}_handle")
    return door


def build_garden_door(width, height, name):
    """One wood sheet flush with the frame, two glazed lights in the upper half."""
    door = Door(name=name, width=I(width), height=I(height), style="hinged",
                props={"Pset_DoorCommon": {"IsExternal": True, "ThermalTransmittance": 1.4}})
    y_out, y_in = _door_band(EXT_FRAME_DEPTH, exterior=True)
    _add_door_frame(door, width, height, y_out, y_in)
    x0, x1 = FRAME_FACE + LEAF_REVEAL, width - FRAME_FACE - LEAF_REVEAL
    z1 = height - LEAF_REVEAL
    lf_out, lf_in = y_out, y_out + LEAF_T
    z_split = I(z1 * GARDEN_SPLIT)
    _obox(door, x0, x1, lf_out, lf_in, 0, z_split, WOOD, "panel_lower")
    _obox(door, x0, x1, lf_out, lf_in, z_split, z_split + GARDEN_RAIL, WOOD, "rail_mid")
    _obox(door, x0, x1, lf_out, lf_in, z1 - GARDEN_RAIL, z1, WOOD, "rail_top")
    _obox(door, x0, x0 + GARDEN_RAIL, lf_out, lf_in, z_split, z1, WOOD, "stile_left")
    _obox(door, x1 - GARDEN_RAIL, x1, lf_out, lf_in, z_split, z1, WOOD, "stile_right")
    xc = width // 2
    _obox(door, xc - GARDEN_RAIL // 2, xc + GARDEN_RAIL // 2, lf_out, lf_in,
          z_split, z1, WOOD, "mullion")
    gt = (lf_out + lf_in) // 2
    gz0, gz1 = z_split + GARDEN_RAIL, z1 - GARDEN_RAIL
    _obox(door, x0 + GARDEN_RAIL, xc - GARDEN_RAIL // 2, gt - GLASS_T // 2,
          gt + GLASS_T // 2 + 1, gz0, gz1, GLASS, "glass_left")
    _obox(door, xc + GARDEN_RAIL // 2, x1 - GARDEN_RAIL, gt - GLASS_T // 2,
          gt + GLASS_T // 2 + 1, gz0, gz1, GLASS, "glass_right")
    _add_handle(door, width, lf_out, f"{name}_handle")
    return door


def build_inner_door(width, height, name, frame_depth):
    """The plain case: one flush wood sheet in a light frame spanning the partition."""
    door = Door(name=name, width=I(width), height=I(height), style="hinged",
                props={"Pset_DoorCommon": {"IsExternal": False}})
    y_out, y_in = _door_band(frame_depth, exterior=False)
    _add_door_frame(door, width, height, y_out, y_in)
    x0, x1 = FRAME_FACE + LEAF_REVEAL, width - FRAME_FACE - LEAF_REVEAL
    _obox(door, x0, x1, y_out, y_out + LEAF_T, 0, height - LEAF_REVEAL, WOOD, "leaf")
    _add_handle(door, width, y_out, f"{name}_handle")
    return door


# ══ 5. THE GROUND FLOOR — PLANNED, NOT DRAWN ═════════════════════════════════
#
# Nothing below is a coordinate. The plan falls out of a room PROGRAM (relative demands,
# not sizes), four rules, and the stair — and it is then PROVED walkable and livable
# rather than eyeballed. Change a weight and the plan re-proportions; change the stair and
# the hall, and therefore every south room, re-sizes with it.
#
#   RULE 1 — a SPINE at ``SOUTH_BAND_FRAC`` of the depth splits daylight rooms (south)
#            from service rooms (north). South-facing living space is the whole reason
#            this house is turned with its long axis east-west.
#   RULE 2 — the HALL is a full-depth band whose width is the STAIR plus a clear zone. The
#            hall is not sized and then checked against the stair; it IS the stair plus
#            ``HALL_CLEAR``, so the two cannot disagree.
#   RULE 3 — the north band carries a *fordelingsgang* along the spine, so every service
#            room opens onto circulation instead of through its neighbour. A bedroom you
#            reach through the bathroom is a plan that compiles and that nobody would live
#            in — and no rule in this DSL would ever mention it.
#   RULE 4 — every room takes the widest window unit its own wall can carry, in the free
#            runs left after the exterior doors have taken theirs.
#
# Then: a BFS from the front door must reach every room (walkable), and every habitable
# room must end up with at least one window (livable). Both raise.


def _free_runs(lo, hi, reserved):
    """``[lo, hi]`` minus every reserved interval — the runs a window may still use."""
    runs, cursor = [], lo
    for a, b in sorted(reserved):
        if a - cursor > 0:
            runs.append((cursor, min(a, hi)))
        cursor = max(cursor, b)
    if hi - cursor > 0:
        runs.append((cursor, hi))
    return [(a, b) for a, b in runs if b > a]


def _fit_windows(lo, hi, unit_w, max_n=MAX_WIN_PER_WALL, pier=WIN_PIER_MIN):
    """Centres for the MOST units of ``unit_w`` that fit ``[lo, hi]`` with equal piers of
    at least ``pier``. Returns ``[]`` when not even one fits — which is the correct answer
    for a corridor end or a wall a door already owns, and is why this returns a list
    rather than forcing a window into a wall that has no room for one."""
    span = hi - lo
    for n in range(min(max_n, 4), 0, -1):
        gap = (span - n * unit_w) / (n + 1.0)
        if gap >= pier:
            return [I(lo + gap * (k + 1) + unit_w * (k + 0.5)) for k in range(n)]
    return []


def build_ground_plan(footprint_x, footprint_y, stair_extent_x,
                      core_t=WALL_CORE_THICKNESS, t=PARTITION_T,
                      south_frac=SOUTH_BAND_FRAC, corridor_w=CORRIDOR_WIDTH,
                      program=GROUND_PROGRAM):
    """Derive the whole ground floor: rooms, partitions, internal doors, exterior doors
    and every window position. Geometry-free — it returns numbers, and the builders below
    turn them into elements."""
    hx, hy = I(footprint_x / 2), I(footprint_y / 2)
    x0, x1 = -hx + core_t, hx - core_t          # the plate, at the core's INNER faces
    y0, y1 = -hy + core_t, hy - core_t
    half = t // 2

    spine_y = y0 + I((y1 - y0) * south_frac)                    # RULE 1
    hall_w = stair_extent_x + 2 * HALL_CLEAR + t                # RULE 2 (centre to centre)
    hall_x0, hall_x1 = -I(hall_w / 2), I(hall_w / 2)
    corridor_y1 = spine_y + corridor_w                          # RULE 3

    rooms, walls = [], []

    def _room(name, long_name, stype, unit, rx0, rx1, ry0, ry1):
        rooms.append({"name": name, "long_name": long_name, "space_type": stype,
                      "unit": unit, "x0": rx0, "x1": rx1, "y0": ry0, "y1": ry1,
                      "windows": []})

    # ── the south band: one room each side of the hall ───────────────────────
    for name, band, place, _w, stype, long_name, unit in program:
        if band != "south":
            continue
        rx0, rx1 = ((x0, hall_x0 - half) if place == "west"
                    else (hall_x1 + half, x1))
        _room(name, long_name, stype, unit, rx0, rx1, y0, spine_y - half)

    # ── the north band: rooms in program order, split by relative weight ─────
    north = sorted([p for p in program if p[1] == "north"], key=lambda p: p[2])
    total = sum(p[3] for p in north)
    edges, acc = [x0], 0.0
    for p in north[:-1]:
        acc += p[3]
        edges.append(x0 + I((x1 - x0) * acc / total))
    edges.append(x1)
    for i, (name, _b, _p, _w, stype, long_name, unit) in enumerate(north):
        _room(name, long_name, stype, unit,
              edges[i] + (half if i else 0),
              edges[i + 1] - (half if i < len(north) - 1 else 0),
              corridor_y1 + half, y1)

    # ── circulation. Derived, never in the program: it is what is LEFT. ──────
    _room("entre", "Entrance hall", "circulation", "small",
          hall_x0 + half, hall_x1 - half, y0, spine_y - half)
    _room("fordelingsgang", "Distribution corridor", "circulation", "small",
          x0, x1, spine_y + half, corridor_y1 - half)

    # ── the partitions. The spine has a GAP where the hall crosses it: that gap
    #    IS the connection between the two circulation spaces, so it is authored
    #    as two wall segments rather than one wall with a doorway in it.
    for tag, wx0, wx1 in (("spine_west", x0, hall_x0), ("spine_east", hall_x1, x1)):
        walls.append({"name": tag, "x0": wx0, "x1": wx1,
                      "y0": spine_y - half, "y1": spine_y + half,
                      "run": (1, 0, 0), "outward": (0, -1, 0)})
    for tag, wx in (("hall_west", hall_x0), ("hall_east", hall_x1)):
        walls.append({"name": tag, "x0": wx - half, "x1": wx + half,
                      "y0": y0, "y1": spine_y + half,
                      "run": (0, 1, 0), "outward": (1 if wx > 0 else -1, 0, 0)})
    walls.append({"name": "corridor_north", "x0": x0, "x1": x1,
                  "y0": corridor_y1 - half, "y1": corridor_y1 + half,
                  "run": (1, 0, 0), "outward": (0, -1, 0)})
    for i in range(1, len(north)):
        walls.append({"name": f"north_split_{i}", "x0": edges[i] - half,
                      "x1": edges[i] + half, "y0": corridor_y1 - half, "y1": y1,
                      "run": (0, 1, 0), "outward": (1, 0, 0)})

    by_name = {r["name"]: r for r in rooms}

    # ── internal doors: one per room, onto circulation ───────────────────────
    doors = []
    for name, band, place, _w, _s, _l, _u in program:
        r = by_name[name]
        if band == "south":
            doors.append({"wall": f"hall_{place}", "centre": spine_y - 1100,
                          "name": name, "connects": ("entre", name)})
        else:
            doors.append({"wall": "corridor_north", "centre": (r["x0"] + r["x1"]) // 2,
                          "name": name, "connects": ("fordelingsgang", name)})

    # ── exterior doors, and the wall runs they take out of the window budget ─
    stue = by_name["stue"]
    ext_doors = [
        {"side": "south", "centre": hall_x0 + 1000, "kind": "front", "name": "hoveddoer",
         "width": FRONT_DOOR_W, "room": "entre"},
        {"side": "east", "centre": (stue["y0"] + stue["y1"]) // 2, "kind": "garden",
         "name": "havedoer", "width": GARDEN_DOOR_W, "room": "stue"},
    ]

    # ── RULE 4: windows, in what the doors leave ─────────────────────────────
    touches = (("south", "y0", y0, "x0", "x1"), ("north", "y1", y1, "x0", "x1"),
               ("west", "x0", x0, "y0", "y1"), ("east", "x1", x1, "y0", "y1"))
    windows = []
    for r in rooms:
        if r["space_type"] == "circulation" and r["name"] == "fordelingsgang":
            continue                       # an internal corridor; its light is borrowed
        uw, uh, us = WIN_UNITS[r["unit"]]
        for side, key, edge, lo_k, hi_k in touches:
            if r[key] != edge:
                continue
            reserved = [(d["centre"] - d["width"] // 2 - DOOR_WIN_CLEAR,
                         d["centre"] + d["width"] // 2 + DOOR_WIN_CLEAR)
                        for d in ext_doors if d["side"] == side]
            for lo, hi in _free_runs(r[lo_k], r[hi_k], reserved):
                for c in _fit_windows(lo, hi, uw):
                    tag = f"{r['name']}_{side}_{len(windows)}"
                    windows.append({"side": side, "centre": c, "width": uw, "height": uh,
                                    "sill": us, "unit": r["unit"], "room": r["name"],
                                    "name": tag})
                    r["windows"].append(tag)

    plan = {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "spine_y": spine_y,
            "hall_x0": hall_x0, "hall_x1": hall_x1, "corridor_y1": corridor_y1,
            "rooms": rooms, "by_name": by_name, "walls": walls, "doors": doors,
            "ext_doors": ext_doors, "windows": windows, "t": t}
    assert_the_plan_works(plan)
    return plan


def assert_the_plan_works(plan):
    """Walkable and livable — the two properties a floor plan has or silently does not.

    A plan where the bedroom is only reachable through the bathroom compiles, validates,
    renders and prices exactly like one where it is not. So the connectivity is a BFS from
    the FRONT DOOR over the doors that were actually placed, and the daylight check is run
    over the windows that were actually derived — neither reads the intent, both read the
    output."""
    graph = {r["name"]: set() for r in plan["rooms"]}
    for d in plan["doors"]:
        a, b = d["connects"]
        graph[a].add(b)
        graph[b].add(a)
    graph["entre"].add("fordelingsgang")        # the gap in the spine, not a door
    graph["fordelingsgang"].add("entre")

    start = plan["ext_doors"][0]["room"]
    seen, queue = {start}, [start]
    while queue:
        for nxt in graph[queue.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    missing = sorted(set(graph) - seen)
    if missing:
        raise AssertionError(
            f"the ground floor is not walkable: {missing} cannot be reached from the "
            f"front door. Every room needs a door onto circulation — a room reached only "
            f"through another room is a plan nobody would live in, and nothing in this "
            f"DSL would ever mention it")

    dark = [r["name"] for r in plan["rooms"]
            if r["space_type"] != "circulation" and not r["windows"]]
    if dark:
        raise AssertionError(
            f"the ground floor is not livable: {dark} ended up with no window. Either the "
            f"program asks for more rooms than this footprint can daylight, or the unit is "
            f"too wide for that wall — try the 'small' unit before shrinking the room")

    total = sum((r["x1"] - r["x0"]) * (r["y1"] - r["y0"]) for r in plan["rooms"])
    for r in plan["rooms"]:
        a = (r["x1"] - r["x0"]) * (r["y1"] - r["y0"]) / 1e6
        print(f"INFO plan:   {r['long_name']:<22s} {(r['x1'] - r['x0']) / 1000:5.2f} x "
              f"{(r['y1'] - r['y0']) / 1000:4.2f} m = {a:5.1f} m²  "
              f"{len(r['windows'])} window(s)")
    print(f"INFO plan: {len(plan['rooms'])} rooms totalling {total / 1e6:.1f} m², "
          f"{len(plan['walls'])} partitions, {len(plan['doors'])} internal doors, "
          f"{len(plan['windows'])} windows — walkable from the front door, all lit")
    return True


def build_partitions(plan, base_z, top_z):
    """The internal walls as ``Wall``s WITH a body, so each carries its own hole rather
    than distributing it — the other side of the line the exterior assemblies sit on, and
    the same ``.opening()`` call either way.

    The buildup is a ``LayerSet`` over that one body — board / acoustic batt / board — so
    the partitions need no separate drywall pass; ``interior-finishes`` skips them by the
    same rule that skips the concrete core, because they arrive already faced."""
    layers = LayerSet(name="partition_type_a", outward=(0, -1, 0),
                      layers=[Material(key=PARTITION_BOARD, thickness_mm=PARTITION_BOARD_T),
                              Material(key=PARTITION_BATT, thickness_mm=PARTITION_BATT_T),
                              Material(key=PARTITION_BOARD, thickness_mm=PARTITION_BOARD_T)])
    out = {}
    for w in plan["walls"]:
        wall = Wall(name=w["name"],
                    props={"Pset_WallCommon": {"LoadBearing": False, "IsExternal": False,
                                               "AcousticRating": "R'w 40"}})
        wall.add(Box(name="body",
                     start=Point(x=I(w["x0"]), y=I(w["y0"]), z=I(base_z)),
                     end=Point(x=I(w["x1"]), y=I(w["y1"]), z=I(top_z))))
        wall.layers = LayerSet(name="partition_type_a", outward=w["outward"],
                               layers=list(layers.layers))
        out[w["name"]] = (wall, Point(x=I(w["x0"]), y=I(w["y0"]), z=I(base_z)),
                          Point(x=I(w["x1"]), y=I(w["y1"]), z=I(top_z)),
                          w["run"], w["outward"])
    print(f"INFO partition: {len(out)} partitions, {PARTITION_T} mm "
          f"({PARTITION_BOARD_T}+{PARTITION_BATT_T}+{PARTITION_BOARD_T} = the layer sum)")
    return out


def place_inner_doors(plan, partitions, product):
    """One leaf per internal door, cut into the partition it belongs to. Each partition
    has a body, so ``.opening()`` lands the hole on its own ``IfcWall``."""
    for d in plan["doors"]:
        wall, start, end, run, outward = partitions[d["wall"]]
        along = _wall_along_centre(start, end, run, outward, d["centre"])
        wall.opening(product.occurrence(name=d["name"]), along_center=I(along), up=0)
    print(f"INFO door: {len(plan['doors'])} internal doors "
          f"({DOOR_W}x{DOOR_H}), one per room onto circulation")


# ══ 8b. THE STAIR — ground floor to loft ═════════════════════════════════════
#
# A quarter-turn (*kvartsvingstrappe*): nine risers up, a square quarter landing, eight
# more. It is the reason ``GROUND_STOREY_HEIGHT`` is 2975 and not a round 3000 — the rise
# divides the storey EXACTLY, and a rise that does not leaves one odd step at one end,
# which is the commonest real stair defect there is and which nothing here would report.
#
# Every piece is a solid box from the flight's base to that step's nosing, which is what a
# built flight is: treads and risers are faces of one solid, not separate parts.


def stair_layout(plan, floor_level, loft_level):
    """Where the stair stands, and what the loft floor therefore has to open for.

    Sited in the hall's far corner so the front door faces the flight rather than its
    side. ``extent_x`` is fed BACK into ``build_ground_plan`` to size the hall, which is
    why this function takes a plan and returns numbers instead of building anything."""
    lower = STAIR_LOWER_STEPS
    upper = STAIR_RISERS - lower
    lower_run, upper_run = lower * STAIR_GOING, upper * STAIR_GOING
    half = plan["t"] // 2
    ax = plan["hall_x0"] + half + HALL_CLEAR
    by = plan["spine_y"] - half - HALL_CLEAR - (lower_run + STAIR_LANDING)
    turn_z = floor_level + lower * STAIR_RISER
    return {"lower": lower, "upper": upper, "lower_run": lower_run, "upper_run": upper_run,
            "ax": ax, "by": by, "turn_y": by + lower_run, "turn_z": turn_z,
            "x1": ax + STAIR_WIDTH + upper_run, "y1": by + lower_run + STAIR_LANDING,
            "floor_level": floor_level, "loft_level": loft_level}


def stair_extent_x():
    """The stair's own footprint along X, which the hall is sized from. Stated here and
    nowhere else — the hall must not carry its own copy of this number."""
    return STAIR_WIDTH + (STAIR_RISERS - STAIR_LOWER_STEPS) * STAIR_GOING


def build_stair(s):
    """One ``IfcStair`` holding two flights, a quarter landing and two handrails."""
    stair = Element(ifc_class="IfcStair", predefined_type="QUARTER_TURN_STAIR",
                    name="trappe",
                    props={"Pset_StairCommon": {
                        "NumberOfRiser": STAIR_RISERS,
                        "NumberOfTreads": STAIR_RISERS - 1,
                        "RiserHeight": STAIR_RISER, "TreadLength": STAIR_GOING,
                        "RequiredHeadroom": STAIR_HEADROOM, "HandicapAccessible": False}})
    ax, by, f = s["ax"], s["by"], s["floor_level"]

    lower = Element(ifc_class="IfcStairFlight", predefined_type="STRAIGHT",
                    name="flight_lower",
                    props={"Pset_StairFlightCommon": {"NumberOfRiser": s["lower"],
                                                      "RiserHeight": STAIR_RISER,
                                                      "TreadLength": STAIR_GOING}})
    for i in range(s["lower"]):
        lower.add(Box(name=f"step_{i:02d}", material=WOOD,
                      start=Point(x=I(ax), y=I(by + i * STAIR_GOING), z=I(f)),
                      end=Point(x=I(ax + STAIR_WIDTH),
                                y=I(by + (i + 1) * STAIR_GOING + STAIR_STEP_NOSING),
                                z=I(f + (i + 1) * STAIR_RISER))))
    stair.add(lower)

    landing = Element(ifc_class="IfcSlab", predefined_type="LANDING", name="repos")
    landing.add(Box(name="body", material=WOOD,
                    start=Point(x=I(ax), y=I(s["turn_y"]), z=I(s["turn_z"] - 200)),
                    end=Point(x=I(ax + STAIR_WIDTH), y=I(s["y1"]), z=I(s["turn_z"]))))
    stair.add(landing)

    upper = Element(ifc_class="IfcStairFlight", predefined_type="STRAIGHT",
                    name="flight_upper",
                    props={"Pset_StairFlightCommon": {"NumberOfRiser": s["upper"],
                                                      "RiserHeight": STAIR_RISER,
                                                      "TreadLength": STAIR_GOING}})
    for j in range(s["upper"]):
        upper.add(Box(name=f"step_{j:02d}", material=WOOD,
                      start=Point(x=I(ax + STAIR_WIDTH + j * STAIR_GOING),
                                  y=I(s["turn_y"]), z=I(s["turn_z"])),
                      end=Point(x=I(ax + STAIR_WIDTH + (j + 1) * STAIR_GOING
                                    + STAIR_STEP_NOSING),
                                y=I(s["y1"]),
                                z=I(s["turn_z"] + (j + 1) * STAIR_RISER))))
    stair.add(upper)

    for tag, path in (
        ("lower", [Point(x=I(ax + STAIR_WIDTH - 60), y=I(by), z=I(f + 900)),
                   Point(x=I(ax + STAIR_WIDTH - 60), y=I(s["turn_y"]),
                         z=I(s["turn_z"] + 900))]),
        ("upper", [Point(x=I(ax + STAIR_WIDTH), y=I(s["y1"] - 60), z=I(s["turn_z"] + 900)),
                   Point(x=I(s["x1"]), y=I(s["y1"] - 60), z=I(s["loft_level"] + 900))]),
    ):
        rail = Element(ifc_class="IfcRailing", predefined_type="HANDRAIL",
                       name=f"handrail_{tag}")
        rail.add(Pipe(name="body", radius=20, material=IRONMONGERY, path=path))
        stair.add(rail)

    print(f"INFO stair: quarter-turn, {s['lower']} + {s['upper']} = {STAIR_RISERS} risers "
          f"of {STAIR_RISER} mm, {STAIR_GOING} mm going (2R+G = "
          f"{2 * STAIR_RISER + STAIR_GOING}), {STAIR_WIDTH} mm clear")
    return stair


def stairwells(s, loft_level, joist_depth=FLOOR_BEAM_PROFILE[1],
               ceiling_t=LOFT_CEILING_T, headroom=STAIR_HEADROOM):
    """The two boxes the loft floor opens for the stair — DERIVED from the headroom rule.

    The vertical-circulation skill lists "no headroom check" as a gap, and this is what
    filling it looks like: walk up the lower flight, find the first step whose head would
    hit the ceiling under the joists, and start the well THERE. State the well by hand
    instead and it is one of two numbers that must agree, which is the shape of bug this
    corpus keeps finding — the flight simply passes through the floor, silently.

    Two boxes, not one bounding box: the stair is an L, and a rectangular well would throw
    away ~3.8 m² of the loft floor it is the whole point of reaching."""
    soffit = loft_level - joist_depth - ceiling_t
    k = 0
    while k < s["lower"] and soffit - (s["floor_level"] + (k + 1) * STAIR_RISER) >= headroom:
        k += 1
    y_open = s["by"] + k * STAIR_GOING
    lo, hi = loft_level - joist_depth - 400, loft_level + 400
    print(f"INFO stair: soffit at z={soffit}; the first {k} steps clear {headroom} mm under "
          f"it, so the well opens at y={I(y_open)} — derived, not stated")
    return [Box(name="well_flight", start=Point(x=I(s["ax"]), y=I(y_open), z=I(lo)),
                end=Point(x=I(s["ax"] + STAIR_WIDTH), y=I(s["y1"]), z=I(hi))),
            Box(name="well_turn", start=Point(x=I(s["ax"] + STAIR_WIDTH),
                                              y=I(s["turn_y"]), z=I(lo)),
                end=Point(x=I(s["x1"]), y=I(s["y1"]), z=I(hi)))]


# ══ 8. THE LOFT FLOOR (systems/loft-floor) ═══════════════════════════════════

def _overlaps(box, x0, x1, y0, y1):
    return (min(box.start.x, box.end.x) < x1 and max(box.start.x, box.end.x) > x0
            and min(box.start.y, box.end.y) < y1 and max(box.start.y, box.end.y) > y0)


def build_loft_floor(footprint_x, footprint_y, loft_level, wells,
                     core_t=WALL_CORE_THICKNESS, spacing=RAFTER_SPACING):
    """Deck on top of the bjælkelag, acoustic batts bay by bay, and — the layer everyone
    forgets — a drywall CEILING under the joists.

    Every other ceiling in this house is a cast concrete soffit and the concrete-stands-
    alone rule leaves those bare. Here the ground floor looks up at timber, so it needs a
    board. Miss it and the loft is the one storey with a structural ceiling and no finish.

    All three layers are added ``carve="none"`` — a finish lying on joists has no boolean
    relationship with them — and each takes the STAIRWELL as an explicit ``.void()``,
    which is a hole, not a carve."""
    inx, iny = I(footprint_x / 2) - core_t, I(footprint_y / 2) - core_t
    bottom = loft_level - FLOOR_BEAM_PROFILE[1]
    out = []

    deck = Element(ifc_class="IfcCovering", predefined_type="FLOORING", name="loft_deck")
    deck.add(Box(start=Point(x=-inx, y=-iny, z=loft_level),
                 end=Point(x=inx, y=iny, z=loft_level + LOFT_DECK_T), material=LOFT_DECK))
    ceiling = Element(ifc_class="IfcCovering", predefined_type="CEILING",
                      name="loft_underside")
    ceiling.add(Box(start=Point(x=-inx, y=-iny, z=bottom - LOFT_CEILING_T),
                    end=Point(x=inx, y=iny, z=bottom), material=LOFT_CEILING))
    for e in (deck, ceiling):
        for i, w in enumerate(wells):
            e.void(w(name=f"{w.name}_{e.name}"), name=f"stairwell_{i}_{e.name}")
        out.append(e)

    xs = _floor_beam_xs(footprint_x, spacing, core_t)
    for i, (xa, xb) in enumerate(zip(xs, xs[1:])):
        x0 = xa + FLOOR_BEAM_PROFILE[0] // 2
        x1 = xb - FLOOR_BEAM_PROFILE[0] // 2
        batt = Element(ifc_class="IfcCovering", predefined_type="INSULATION",
                       name=f"loft_insulation_{i}")
        batt.add(Box(start=Point(x=I(x0), y=-iny, z=bottom),
                     end=Point(x=I(x1), y=iny, z=bottom + LOFT_INSULATION_T),
                     material=LOFT_INSULATION))
        for j, w in enumerate(wells):
            if _overlaps(w, x0, x1, -iny, iny):
                batt.void(w(name=f"{w.name}_batt_{i}"), name=f"stairwell_{j}_batt_{i}")
        out.append(batt)
    print(f"INFO loft: deck z={loft_level}, {LOFT_INSULATION_T} mm batts in "
          f"{len(xs) - 1} joist bays, ceiling under the joists at z={bottom}, "
          f"{len(wells)} stairwell openings")
    return out


# ══ 9. INTERIOR FINISHES (systems/interior-finishes) ═════════════════════════
# THE RULE: **concrete stands alone.** A wall or soffit that is already concrete is a
# finished surface and gets no drywall — drywall faces a framed surface, not a cast one.
# In this house that leaves exactly two jobs, because the partitions arrive already faced
# (their LayerSet is board/batt/board) and every core wall is concrete.


def build_ground_finishes(plan, floor_level):
    """The ground floor's walking surface, laid room by room rather than as one slab-wide
    panel — a floor stops at a partition, and laying it per room is also what makes a
    per-room material change (tile in the bathroom) a one-line edit later."""
    out = []
    for r in plan["rooms"]:
        cov = Element(ifc_class="IfcCovering", predefined_type="FLOORING",
                      name=f"floor_{r['name']}")
        cov.add(Box(start=Point(x=I(r["x0"]), y=I(r["y0"]), z=I(floor_level)),
                    end=Point(x=I(r["x1"]), y=I(r["y1"]), z=I(floor_level + FLOOR_PANEL_T)),
                    material=FLOOR_PANEL))
        out.append(cov)
    print(f"INFO finishes: {len(out)} floor panels ({FLOOR_PANEL} {FLOOR_PANEL_T} mm), "
          f"one per room; every core wall is concrete and stands alone, and the "
          f"partitions arrive faced, so no wall cladding pass is needed")
    return out


def build_loft_ceilings(footprint_x, footprint_y, wall_top, loft_level,
                        pitch_deg=ROOF_PITCH_DEG, core_t=WALL_CORE_THICKNESS):
    """The loft's own ceiling: flat under the collars, sloping under the rafters down to
    the skunk walls. The shape comes from the frame's geometry — ``roof_pitch``, the
    collar height, the rafter depth — not from a guess."""
    inx = I(footprint_x / 2) - core_t
    th = math.radians(pitch_deg)
    collar_z = collar_level(loft_level)
    flat_z = collar_z - COLLAR_PROFILE[1] // 2
    yc = I(GABLE_HALF_Y - (collar_z - wall_top) / math.tan(th))
    out = []
    flat = Element(ifc_class="IfcCovering", predefined_type="CEILING", name="loft_flat")
    flat.add(Box(start=Point(x=-inx, y=-yc, z=flat_z - DRYWALL_T),
                 end=Point(x=inx, y=yc, z=flat_z), material=DRYWALL))
    out.append(flat)

    half = RAFTER_PROFILE[1] / 2.0
    run = (GABLE_HALF_Y - yc) / math.cos(th)
    for sign, tag in ((-1, "south"), (1, "north")):
        up = (-sign * math.cos(th), math.sin(th))
        nrm = (sign * math.sin(th), math.cos(th))
        quad = [(s * up[0] + d * nrm[0], s * up[1] + d * nrm[1])
                for s, d in ((0, -half - DRYWALL_T), (run, -half - DRYWALL_T),
                             (run, -half), (0, -half))]
        cov = Element(ifc_class="IfcCovering", predefined_type="CEILING",
                      name=f"loft_slope_{tag}")
        cov.add(Sweep(profile=[Point2D(x=I(a), y=I(b)) for a, b in _ccw(quad)],
                      path=[Point(x=-inx, y=I(sign * GABLE_HALF_Y), z=wall_top),
                            Point(x=inx, y=I(sign * GABLE_HALF_Y), z=wall_top)],
                      material=Material(key=DRYWALL)))
        out.append(cov)
    print(f"INFO finishes: loft ceiling — flat under the collars at z={flat_z} "
          f"({flat_z - DRYWALL_T - (loft_level + LOFT_DECK_T)} mm of headroom over "
          f"{2 * yc} mm of the span), sloping under the rafters to the skunk walls")
    # The SOFFIT, not the board's top: a Space drawn to the top swallows the ceiling and
    # the compiler reports that the board bounds nothing.
    return out, yc, flat_z - DRYWALL_T


# ══ THE CATALOG — every repeated unit promoted once ══════════════════════════
# A ``Product`` IS finished physical dimensions: every parameter is consumed BEFORE
# promotion, and restating one on ``.occurrence()`` is an error. Two windows differing by
# 200 mm are two Products — two cut-list lines, two order codes. What it buys beyond
# authoring the joinery once: an ``IfcWindowType`` / ``IfcDoorType`` each occurrence is
# joined to, so a schedule reads "9 x nordic_1400x1600" instead of nine unrelated units
# that happen to share dimensions.

NORDIC = Product(build_window(WIN_W, WIN_H, "nordic"), name=f"nordic_{WIN_W}x{WIN_H}")
SMALL = Product(build_window(WIN_SMALL_W, WIN_SMALL_H, "small"),
                name=f"nordic_{WIN_SMALL_W}x{WIN_SMALL_H}")
GAVL = Product(build_window(GABLE_WIN_W, GABLE_WIN_H, "gavl"),
               name=f"gavl_{GABLE_WIN_W}x{GABLE_WIN_H}")
KVIST = Product(build_window(DORMER_WIN_W, DORMER_WIN_H, "kvist",
                             band=_light_band(DORMER_FRONT_T)),
                name=f"kvist_{DORMER_WIN_W}x{DORMER_WIN_H}")
WINDOW_PRODUCTS = {"main": NORDIC, "small": SMALL}

FRONT = Product(build_front_door(FRONT_DOOR_W, FRONT_DOOR_H, "front"),
                name=f"front_panelled_{FRONT_DOOR_W}x{FRONT_DOOR_H}")
GARDEN = Product(build_garden_door(GARDEN_DOOR_W, GARDEN_DOOR_H, "garden"),
                 name=f"garden_glazed_{GARDEN_DOOR_W}x{GARDEN_DOOR_H}")
INNER = Product(build_inner_door(DOOR_W, DOOR_H, "inner", PARTITION_T),
                name=f"inner_flush_{DOOR_W}x{DOOR_H}")
DOOR_PRODUCTS = {"front": FRONT, "garden": GARDEN}

#: A gently sloped plot, rising to the north — the foundation samples it at the origin
#: and the stenkant follows it, so nothing here assumes the ground is flat.
SITE_HEIGHTMAP = [
    [-300, -300, -300, -300, -300],
    [-150, -100, -150, -100, -150],
    [0, 50, 0, 50, 0],
    [200, 300, 200, 300, 200],
    [450, 500, 450, 500, 450],
]


def generate_project():
    proj = Project(name="Murermestervilla — the whole 1.5-storey house")
    assert_the_section_closes()

    # ── STEP 1: the site block. Lowest ground point normalised to z = 0. ─────
    site = Site(name="site")
    terrain = Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN",
                      name="terrain")
    terrain.add(build_site_block(FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y,
                                 heightmap=SITE_HEIGHTMAP))
    site.add(terrain)

    # ── STEP 2 + 2.2: foundation and the perimeter gravel apron. ────────────
    slab, strips, floor_level, clearing_void = build_strip_foundation(
        FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y, site)
    bed, stenkant_voids = build_stenkant(FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y,
                                         site)

    # ── THE SECTION. Everything above the slab is derived from these. ───────
    loft_level = floor_level + GROUND_STOREY_HEIGHT
    wall_top = loft_level + LOFT_WALL_HEIGHT
    ridge_z = _ridge_z(FOUNDATION_FOOTPRINT_Y, wall_top)
    ceiling_z = loft_level - FLOOR_BEAM_PROFILE[1]        # the joists' underside
    print(f"INFO section: slab {floor_level} / loft {loft_level} / wall top {wall_top} / "
          f"ridge {ridge_z} — {ridge_z - loft_level - LOFT_DECK_T} mm from the loft floor "
          f"to the apex")

    ground = Storey(name="ground", elevation=floor_level)
    loft = Storey(name="loft", elevation=loft_level)
    proj.add_storey(ground)
    proj.add_storey(loft)

    # ── STEP 3 + 4: the concrete core and the sandwich outside it, per storey.
    #    The upper ring is only LOFT_WALL_HEIGHT tall — the skunk — which is what
    #    makes this 1.5 storeys rather than 2.
    walls_g = build_walls(FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y,
                          floor_level, loft_level)
    walls_l = build_walls(FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y,
                          loft_level, wall_top)

    # ── STEP 5: the ground floor, PLANNED. The stair sizes the hall, so it is
    #    measured before the plan exists and built after it.
    plan = build_ground_plan(FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y,
                             stair_extent_x())
    s = stair_layout(plan, floor_level, loft_level)
    stair = build_stair(s)
    wells = stairwells(s, loft_level)
    partitions = build_partitions(plan, floor_level, ceiling_z)
    place_inner_doors(plan, partitions, INNER)

    # ── STEP 7: the ground-floor windows the plan derived. ─────────────────
    for w in plan["windows"]:
        place_window(walls_g[w["side"]], WINDOW_PRODUCTS[w["unit"]], w["centre"],
                     floor_level + w["sill"], w["width"], w["name"])
    print(f"INFO window: {len(plan['windows'])} ground-floor units placed from the plan's "
          f"own room geometry — no hand-written schedule")

    # ── STEP 7b: the two exterior doors. ONE call each, on the ASSEMBLY. ────
    for d in plan["ext_doors"]:
        wall, start, end, run, outward = walls_g[d["side"]]
        wall.opening(DOOR_PRODUCTS[d["kind"]].occurrence(name=d["name"]),
                     along_center=I(_wall_along_centre(start, end, run, outward,
                                                       d["centre"])), up=0)
    print(f"INFO door: front door on the {plan['ext_doors'][0]['side']} facade into the "
          f"hall, garden door on the {plan['ext_doors'][1]['side']} facade out of the stue")

    # ── STEP 6: the roof frame, opened for the dormers. ────────────────────
    dormer_geo = build_dormer_geometries(wall_top, ridge_z)
    rafters, collars, plates, trimming = build_roof_frame(
        FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y, wall_top, loft_level, dormer_geo)
    floor_beams = build_floor_beams(FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y,
                                    loft_level)
    posts = build_inner_wall_posts(FOUNDATION_FOOTPRINT_X, wall_top, loft_level)
    ridge_beam = build_ridge_beam(FOUNDATION_FOOTPRINT_X, ridge_z)

    # ── STEP 6b: the covering, split around the dormers, and the RIDGE. ────
    underlay = build_roof_underlay(FOUNDATION_FOOTPRINT_Y, wall_top, dormer_geo)
    battens = build_roof_battens(FOUNDATION_FOOTPRINT_Y, wall_top, dormer_geo)
    tiles, _counts = build_roof_tiles(FOUNDATION_FOOTPRINT_Y, wall_top, dormer_geo)
    roof_insul = build_roof_insulation(FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y,
                                       wall_top, dormer_geo)
    ridge = build_ridge(FOUNDATION_FOOTPRINT_Y, wall_top)

    # ── STEP 6c: gables, corbels, and the loft's gable windows. ────────────
    gables = build_gable_ends(FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y, wall_top)
    build_corbels(gables, {k: v[0].elements[1] for k, v in walls_l.items()},
                  FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y, loft_level, wall_top)
    place_gable_windows(gables, GAVL, FOUNDATION_FOOTPRINT_Y, wall_top, (-1200, 1200))

    # ── STEP 6d: the dormers. ──────────────────────────────────────────────
    dormers = [build_dormer(g, KVIST, FOUNDATION_FOOTPRINT_Y, wall_top)
               for g in dormer_geo]

    # ── STEP 8 + 9: the loft floor and the finishes. ───────────────────────
    loft_layers = build_loft_floor(FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y,
                                   loft_level, wells)
    ground_floors = build_ground_finishes(plan, floor_level)
    loft_ceilings, loft_flat_y, loft_flat_z = build_loft_ceilings(
        FOUNDATION_FOOTPRINT_X, FOUNDATION_FOOTPRINT_Y, wall_top, loft_level)

    # ══ WIRE THE ASSEMBLY ═══════════════════════════════════════════════════
    # ORDER IS TREE ORDER, NOT CALL ORDER: displacement walks storey by storey, then
    # add-order within each, and a later solid carves the earlier one it overlaps.
    # Two joints depend on it and would be silent if it were wrong:
    #   * the bjælkelag is in the LOFT storey and the walls it bears on are in GROUND,
    #     so each beam cuts its own 50 mm bearing pocket in the ground ring's top;
    #   * the covering goes in BEFORE the rafters and the rafters before the skunk
    #     walls, so the walls birdsmouth each rafter FOOT — one shallow notch each —
    #     instead of 26 feet each carving the wall into a CSG chain the viewer drops.
    proj.add(site)                      # a Site attaches to the project, not to a storey
    site.void(clearing_void)
    site.add(bed)
    site.void(*stenkant_voids)

    ground.add(slab, *strips)
    ground.add(*[w[0] for w in walls_g.values()])
    ground.add(*[p[0] for p in partitions.values()])
    ground.add(stair)
    ground.add(*ground_floors, carve="none")

    loft.add(*underlay, carve="none")
    loft.add(*battens, carve="none")
    loft.add(tiles, carve="none")
    loft.add(ridge, carve="none")
    loft.add(*roof_insul, carve="none")
    loft.add(*rafters, *collars)
    loft.add(*[w[0] for w in walls_l.values()])
    loft.add(*floor_beams)
    loft.add(*plates)
    loft.add(*trimming)
    loft.add(*posts)
    loft.add(ridge_beam)
    loft.add(*gables, carve="none")
    loft.add(*dormers, carve="none")
    loft.add(*loft_layers, carve="none")
    loft.add(*loft_ceilings, carve="none")

    for side in ("south", "north", "west", "east"):
        proj.aggregate(f"wall_{side}", members=[f"wall:{side}:storey:ground",
                                                f"wall:{side}:storey:loft"])

    # ══ SPACES ══════════════════════════════════════════════════════════════
    # Authored, not derived. Every partition here is shared FULL LENGTH by the two rooms
    # either side of it, and derived mode takes the wrong side for the second of them —
    # so on a real floor plan the honest answer is to draw each room to the inner faces
    # the plan already knows. Bounds are then INFERRED from what the box touches.
    #
    # **AN AUTHORED SPACE CARVES.** It is a solid in the displacement pass like any other,
    # and it is added LAST, so it claims every piece of building standing inside it: seven
    # stair treads, the quarter landing, a handrail and all 22 inner-wall posts came back
    # from `check_renderability` as products the renderer draws nothing for. The compile
    # said so too — "wholly inside <anonymous Box>, carved to nothing" — and both are easy
    # to read past, because the model still validates and the render still looks like a
    # house. A room is AIR: `carve="none"` is not a workaround, it is the truth.
    z0 = floor_level + FLOOR_PANEL_T
    z1 = ceiling_z - LOFT_CEILING_T
    rooms = []
    for r in plan["rooms"]:
        sp = Space(name=r["name"], space_type=r["space_type"])
        sp.add(Box(start=Point(x=I(r["x0"]), y=I(r["y0"]), z=I(z0)),
                   end=Point(x=I(r["x1"]), y=I(r["y1"]), z=I(z1))))
        rooms.append(sp)
    ground.add(*rooms, carve="none")

    # The loft is ONE room, drawn to the STANDABLE volume — under the flat ceiling,
    # between the two points where the rake drops below head height. Drawing it wall to
    # wall would claim the *skunk* behind the knee walls as living space, which is the
    # oldest lie in a 1.5-storey floor area.
    inx = I(FOUNDATION_FOOTPRINT_X / 2) - WALL_CORE_THICKNESS
    loftroom = Space(name="loftrum", space_type="residential")
    loftroom.add(Box(start=Point(x=-inx, y=-loft_flat_y, z=loft_level + LOFT_DECK_T),
                     end=Point(x=inx, y=loft_flat_y, z=loft_flat_z)))
    loft.add(loftroom, carve="none")

    proj.zone("occupied", members=[f"space:{r['name']}:storey:ground"
                                   for r in plan["rooms"]] + ["space:loftrum:storey:loft"])
    return proj


result = generate_project()


if __name__ == "__main__":
    compile_main()
