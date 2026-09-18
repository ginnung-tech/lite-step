"""LOD 350 steel truss deck arch — the WHOLE bridge, every plan step in one file.

This is the plan-level reference: a whole crossing, closed. Read it to copy the
numbers that only exist where the steps meet.

Modelled on the High Steel Bridge (South Fork Skokomish, 1929): a single-span steel deck
arch carrying a one-lane forest road about 106 m above the canyon river.

THE SECTION, top down, every level DERIVED rather than typed:

    z ...  deck top       z_deck(u)          linear between the two rim markers
    z ...  deck soffit    z_deck(u) - DECK_DEPTH
    z ...  arch crown     deck soffit at midspan - DECK_ARCH_CLEARANCE
    z ...  arch springing SOLVED against the canyon, see below
           ARCH_SPAN     = the canyon's width at that level      <- the GROUND sets the span
           ARCH_RISE     = Z_CROWN - Z_SPRING
           spandrel post = deck soffit - arch_z(u)               <- and the rise sets every post
           approach bent = deck soffit - ground_at(u)            <- outside the arch

A deck arch is not an arch with a road on top, and it is not a shape you type either.
The road level comes from the terrain at the two rims and is fixed. The arch has to fit
UNDER it AND land on the canyon walls, and only one springing level does both: the level
at which the gorge happens to be as wide as an arch of the chosen rise ratio wants to be.
``_solve_the_springing()`` bisects the DEM for it. The spandrel bents are whatever is
left between the two curves; the trestle bents are whatever is left between the deck and
the ground outside the arch.

THIS FILE EXISTS TO AVOID THAT ERROR. Typing the springing
(``Z_SPRING = 45000``) alongside the prototype's ``ARCH_SPAN = 209000`` without
consulting the DEM under them is the mistake. This canyon is 88 m wide at that level, not 209 m — so
the outer 36 m of arch at each end was INSIDE the canyon wall, each skewback was cast
53 m below the ground surface, and 62 m of deck at each end had nothing underneath it at
all. It compiled clean, validated as IFC 4.3, held CSG depth 0, passed the renderability
gate, and looked plausible in five of the six corpus views.

THE THREE THINGS THIS FILE ADDS that no single system example has:

  1. **One frame, shared by every system.** ``W(u, v, z)`` maps local along-span /
     cross-span / height to world. The crossing is diagonal (the rim markers are 143 m
     apart in X and 195 m in Y) and NOTHING here handles that diagonal by hand -- it is
     handled once, in ``W``, and the arch, the deck, the roadway and the abutments are all
     authored as though the bridge ran along a straight axis.
  2. **Asserts, because nothing carves.** Every steel piece is added ``carve="none"``
     (rules/extruded-steel-member), so the compiler reports no junction as wrong -- a
     model whose deck is buried in its own arch compiles clean with zero warnings. So the
     adjacencies are asserted directly, in world coordinates, BEFORE any solid is built.
     ``assert_the_span_closes()`` samples 201 stations along the arch against the DEM,
     between the panel nodes as well as at them: an arch can clear both ends of a bay and
     still be buried halfway along it.
  3. **The bridge is checked against the GROUND, everywhere it touches it.**
     ``assert_the_abutments_reach_rock()`` samples the DEM under every concrete block --
     two skewbacks, two deck seats, six bent pads -- with ``mesh.height_at()``, and
     refuses one that hangs in the air or is buried whole. This is the seam a terrain
     skill and a steel skill cannot each see alone, and it is the whole reason the plan
     says to build every system on one datum and then look.

Five ``IfcBridgePart``s, not four: SUPERSTRUCTURE (arch + spandrels), DECK, SURFACESTRUCTURE,
ABUTMENT, and PIER for the approach trestles.

All coordinates are int millimetres, Z-up, and the model is z-normalised so the canyon
floor is z=0.
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
from lite_step.materials.dk import MemberGeometry, Render, SheetGeometry
from lite_step.models import (
    Bar,
    Element,
    Extrude,
    Material,
    Mesh,
    Point,
    Point2D,
    Project,
    Site,
    SpatialElement,
    Sweep,
)

I = lambda x: int(round(x))  # noqa: E731,E741

# ── Inline Materials (create-material rule) ────────────────────────────────
dk.MATERIALS["Steel_S355"] = MaterialDef(
    category="metal",
    function="LoadBearing",
    psets={
        "Pset_MaterialCommon": {"MassDensity": 7850},
        "Pset_MaterialSteel": {"YieldStress": 355},
    },
    render=Render(rgb="#7d8a6a", alpha=1.0, finish="satin"),
    geometry=MemberGeometry(form="member", stock_profiles_mm=[]),
)
dk.MATERIALS["Asphalt_Pavement"] = MaterialDef(
    category="aggregate",
    function="Finish",
    psets={"Pset_MaterialCommon": {"MassDensity": 2300}},
    render=Render(rgb="#2b2b2f", alpha=1.0, finish="matt"),
    geometry=SheetGeometry(form="sheet", thicknesses_mm=[40, 50, 60]),
)
dk.MATERIALS["RoadMarking_White"] = MaterialDef(
    category="board",
    function="Finish",
    psets={"Pset_MaterialCommon": {"MassDensity": 1500}},
    render=Render(rgb="#eef0ec", alpha=1.0, finish="matt"),
    geometry=SheetGeometry(form="sheet", thicknesses_mm=[3]),
)

STEEL, ASPHALT, PAINT, CONCRETE = (
    "Steel_S355",
    "Asphalt_Pavement",
    "RoadMarking_White",
    "Concrete_C30-37",
)

# ── The Local Bridge Frame (LOD 100 markers, recentred + z-normalised) ─────
START = (-73005, -97546)  # SW abutment (deck START marker), on the canyon rim
END = (73006, 97545)  # NE abutment (deck END marker)
Z_START, Z_END = 107770, 106657  # deck top at each end (nearly level)
_DX, _DY = END[0] - START[0], END[1] - START[1]
LH = math.hypot(_DX, _DY)  # 243678 — horizontal span rim to rim
_AX, _AY = _DX / LH, _DY / LH  # unit along-span (horizontal)
_CX, _CY = -_AY, _AX  # unit cross-span (horizontal, left)

# ── Reinforcement Parameters ───────────────────────────────────────────────
REBAR_COVER = 50
REBAR_PITCH_MAX = 400
REBAR_DIAMETER = 12
_AXIS = ("x", "y", "z")


def _bar_positions(span, cover=REBAR_COVER, pitch_max=REBAR_PITCH_MAX):
    usable = span - 2 * cover
    if usable <= 0:
        return [I(span / 2)]
    n = max(1, -(-usable // pitch_max))
    return [I(cover + usable * k / n) for k in range(int(n) + 1)]


def build_reinforcement(
    element,
    cover=REBAR_COVER,
    pitch_max=REBAR_PITCH_MAX,
    diameter=REBAR_DIAMETER,
    grade="B500B",
):
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
            bars.append(
                Bar(
                    name=f"{_AXIS[run]}_{i:02d}",
                    path=[
                        Point(**fixed, **{_AXIS[run]: a}),
                        Point(**fixed, **{_AXIS[run]: z}),
                    ],
                    diameter=diameter,
                    grade=grade,
                    mark=f"R{_AXIS[run].upper()}",
                )
            )
    return bars


def W(u, v, z):
    """Local (along u, cross v, height z) -> world Point."""
    return Point(
        x=I(START[0] + u * _AX + v * _CX),
        y=I(START[1] + u * _AY + v * _CY),
        z=I(z),
    )


def z_deck(u):
    """Deck-top height at along-position u (linear between rim levels)."""
    return Z_START + (Z_END - Z_START) * (u / LH)


def steel(name, profile, pts):
    """One extruded-steel member."""
    return Sweep(name=name, path=pts, material=Material(key=STEEL, profile_mm=profile))


# ── 1. The canyon (bridge-terrain) — built FIRST, because the arch is derived ──
# from it. Real-world DEM elevation grid: subdivide for resolution, gently smooth
# (strength=5) to soften elevation quantisation stepping, simplify to prune flat cells.
TERRAIN_HEIGHTMAP = [
    [82172, 94504, 108137, 110219, 106742, 94887, 77043, 82242, 69110],
    [78035, 86852, 105770, 100043, 94309, 83516, 69305, 61996, 55664],
    [76188, 89078, 104321, 94403, 88215, 67614, 39918, 32594, 31852],
    [76383, 83235, 80856, 77387, 62414, 16004, 6598, 12, 0],
    [84950, 63418, 31285, 34078, 16543, 1250, 17137, 40446, 38047],
    [72149, 53188, 28418, 9180, 1289, 22200, 41200, 61379, 74778],
    [83672, 51211, 26235, 2965, 8778, 34008, 55387, 81082, 91180],
    [71434, 29301, 1575, 15492, 36930, 51075, 78926, 96856, 102043],
    [35989, 5645, 4614, 49391, 65801, 79403, 84739, 97602, 106336],
    [12707, 8293, 30461, 68250, 79571, 85684, 94797, 101723, 104481],
    [3438, 33895, 61825, 83719, 91961, 99121, 101871, 102860, 109692],
]

CANYON = (
    Mesh(
        heightmap=TERRAIN_HEIGHTMAP,
        depth=20000,
        corner_min=Point2D(x=-114616, y=-133115),
        corner_max=Point2D(x=97718, y=132303),
        source_type="synthetic",
    )
    .subdivide(height_map_only=True)
    .smooth(strength=5, height_map_only=True)
    .simplify(strength=10, height_map_only=True)
)


def ground_at(u, v=0):
    """Canyon surface height on the bridge axis at along-position ``u``.

    A ``Mesh`` with ``heightmap=`` and no ``placement=`` answers ``height_at``
    off its RETAINED heightmap — exact bilinear interpolation, not a raycast
    against the tessellation. That is why the terrain is a module-level object
    here rather than something fished back out of the site later.
    """
    p = W(u, v, 0)
    return CANYON.height_at(x=p.x, y=p.y)


# ── Geometry Parameters ────────────────────────────────────────────────────
DECK_WIDTH = 7000
DECK_HALF = DECK_WIDTH // 2
RIB_HALF = 3100
DECK_DEPTH = 2600
UC = LH / 2

#: Clear air between the deck soffit and the top of the arch at midspan. Named,
#: because it is the number that makes the arch a DECK arch: shrink it to zero and
#: the crown grows up into the deck truss, which compiles perfectly and is wrong.
DECK_ARCH_CLEARANCE = 4000
Z_CROWN = z_deck(UC) - DECK_DEPTH - DECK_ARCH_CLEARANCE
ARCH_DEPTH = 6000
N_PANEL = 12

#: Rise / span of the arch. The High Steel Bridge is close to 1:3.7, which is the
#: usual band for a riveted steel truss deck arch: flatter and the thrust at the
#: skewback grows past what a rock face takes, steeper and the deck-level posts
#: near the springing get long enough to need their own bracing system.
ARCH_RISE_RATIO = 0.27

#: A spandrel post shorter than this is not built — near the crown the deck soffit
#: and the arch have nearly met, and a 40 mm "post" is a modelling artefact.
MIN_POST = 300
#: X-bracing needs a bay tall enough to brace. Same reasoning, one level up.
MIN_BRACE = 2000

#: How far a skewback is cast into the rock face behind the springing.
SKEWBACK_EMBEDMENT = 6000
#: How far a deck seat or a bent pad is founded below the surface it stands on.
SEAT_EMBEDMENT = 1500
#: How far the arch is allowed to graze the ground before it counts as buried. The
#: springings sit ON the surface by construction and the solver finds them to within
#: its scan step, so the two ends legitimately touch.
ARCH_SPRING_TOL = 600

ARCH_CHORD = (900, 600)
ARCH_WEB = (350, 350)
POST = (450, 450)
SPAN_BRACE = (300, 300)
LAT_BRACE = (320, 320)
DECK_CHORD = (400, 400)
DECK_WEB = (220, 220)
FLOOR_BEAM = (400, 700)
RAIL_POST = (140, 140)
RAIL = (160, 120)
N_DECK = 16


# ── 2. The arch, SOLVED against the canyon (bridge-arch) ───────────────────
def _canyon_width_at(level, step=500.0):
    """Half-width of the canyon on the bridge axis at ``level``, symmetric about
    midspan. Walks in from each rim to the first station at or below the level;
    the arch is symmetric, so the tighter of the two sides governs."""
    def crossing(direction):
        u = UC
        while 0 <= u <= LH:
            if ground_at(u) >= level:
                return u
            u += direction * step
        return None

    u0, u1 = crossing(-1), crossing(+1)
    if u0 is None or u1 is None:
        return None
    return min(UC - u0, u1 - UC)


def _solve_the_springing():
    """Find the level where the canyon is exactly as wide as the arch wants to be.

    THE SPRINGING IS DERIVED, NOT TYPED. Typing it —
    ``Z_SPRING = 45000``, with ``ARCH_SPAN = 209000`` taken from the prototype —
    leaves the DEM under it unconsulted. The canyon in this heightmap is
    88 m wide at that level, not 209 m, so the outer 36 m of arch at each end
    was INSIDE the canyon wall and each skewback was cast 53 m below the ground
    surface. It compiled clean, validated, held CSG depth, passed the
    renderability gate, and looked plausible in five of six views.

    So the span is not typed any more. The rise ratio is (it is the engineering
    choice), and the level that delivers it against this ground is bisected for:
    raise the springing and the canyon widens, so the ratio falls, monotonically.
    """
    lo, hi = ground_at(UC), Z_CROWN  # river bed .. crown; the ratio spans it
    for _ in range(60):
        mid = (lo + hi) / 2
        half = _canyon_width_at(mid)
        if half is None or half <= 0:
            lo = mid
            continue
        # rise/span falls as the springing rises
        if (Z_CROWN - mid) / (2 * half) > ARCH_RISE_RATIO:
            lo = mid
        else:
            hi = mid
    level = (lo + hi) / 2
    half = _canyon_width_at(level)
    if half is None or half <= 0:
        raise AssertionError(
            "no level between the river bed and the crown gives the arch a canyon to "
            "span. Either the deck axis does not cross the gorge or the DEM is flat."
        )
    return level, 2 * half


Z_SPRING, ARCH_SPAN = _solve_the_springing()
ARCH_U0, ARCH_U1 = UC - ARCH_SPAN / 2, UC + ARCH_SPAN / 2
ARCH_RISE = Z_CROWN - Z_SPRING


def arch_z(u):
    """Upper-arch-chord height parabola."""
    return Z_CROWN - 4 * ARCH_RISE * ((u - UC) / ARCH_SPAN) ** 2


def _arch_nodes():
    return [ARCH_U0 + i * (ARCH_SPAN / N_PANEL) for i in range(N_PANEL + 1)]


# ── The section, checked before any steel is cut ────────────────────────────
def assert_the_span_closes():
    """Every steel piece is added ``carve="none"``, so NOTHING here is checked by
    the compiler. A deck buried in its own arch produces zero warnings, zero carves
    and CSG 0 — all perfectly true of a bridge that cannot be built.

    So the section is asserted directly, in world coordinates, before the first
    ``Sweep`` exists. Each check is a way the derivation can stop closing, not a
    restatement of a constant.
    """
    if ARCH_SPAN >= LH:
        raise AssertionError(
            f"the arch spans {ARCH_SPAN} mm between rims {LH:.0f} mm apart — it would "
            f"spring outside the canyon. ARCH_SPAN must stay under the rim-to-rim run."
        )
    if ARCH_RISE <= 0:
        raise AssertionError(
            f"arch rise is {ARCH_RISE:.0f} mm. The crown is derived DOWN from the deck "
            f"soffit ({z_deck(UC) - DECK_DEPTH:.0f}) and the springing is typed "
            f"({Z_SPRING}); a non-positive rise means the springing is above the crown "
            f"and the parabola is inside out."
        )
    if ARCH_RISE < ARCH_DEPTH:
        raise AssertionError(
            f"the arch rises {ARCH_RISE:.0f} mm but its truss is {ARCH_DEPTH} mm deep — "
            f"the lower chord would pass above the springing."
        )

    # The crown is the tightest point between the two curves, by construction. If it
    # is not, the parabola is not the one the clearance was derived against.
    gaps = [(z_deck(u) - DECK_DEPTH) - arch_z(u) for u in _arch_nodes()]
    tightest = min(gaps)
    if abs(tightest - DECK_ARCH_CLEARANCE) > 1.0:
        raise AssertionError(
            f"the deck soffit comes within {tightest:.0f} mm of the arch, not the "
            f"{DECK_ARCH_CLEARANCE} mm it was derived for. The deck grade and the "
            f"parabola have stopped agreeing."
        )
    if tightest <= 0:
        raise AssertionError("the deck passes THROUGH the arch")

    # Springings: the parabola must actually arrive at the level the skewbacks are cast at.
    for tag, u in (("SW", ARCH_U0), ("NE", ARCH_U1)):
        drop = abs(arch_z(u) - Z_SPRING)
        if drop > 1.0:
            raise AssertionError(
                f"the {tag} springing lands at z={arch_z(u):.0f}, but its skewback is "
                f"cast at z={Z_SPRING}. A {drop:.0f} mm gap is an arch resting on air."
            )

    built = [g for g in gaps if g > MIN_POST]
    if len(built) < N_PANEL // 2:
        raise AssertionError(
            f"only {len(built)} of {len(gaps)} spandrel stations clear {MIN_POST} mm — "
            f"the deck is riding the arch, not standing off it."
        )

    # THE ONE THAT CAUGHT THE SHIPPED DEFECT. The arch has to be in the air, not in
    # the hillside, and nothing else in the model can tell. Sampled between the
    # panel nodes too: an arch can clear both nodes of a bay and still be buried
    # halfway along it, which is exactly how a coarse check would pass this.
    buried = []
    for k in range(201):
        u = ARCH_U0 + (ARCH_U1 - ARCH_U0) * k / 200
        clear = arch_z(u) - ground_at(u)
        # ARCH_SPRING_TOL: the springings sit ON the surface by construction, and the
        # solver locates them to within its scan step, so the two ends are allowed to
        # graze it. Anything deeper is rock.
        if clear < -ARCH_SPRING_TOL:
            buried.append((u, clear))
    if buried:
        u, clear = min(buried, key=lambda t: t[1])
        raise AssertionError(
            f"{len(buried)} of 201 stations along the arch are INSIDE the canyon wall — "
            f"worst {-clear:.0f} mm below the surface at u={u:.0f}. The springing is "
            f"solved against the DEM by _solve_the_springing(); if this fires, the "
            f"terrain and the arch have stopped agreeing."
        )
    print(
        f"INFO section: deck top {z_deck(0):.0f} -> {z_deck(LH):.0f} over a {LH:.0f} mm "
        f"diagonal run; soffit -{DECK_DEPTH}; crown {Z_CROWN:.0f}; springing {Z_SPRING} "
        f"-> rise {ARCH_RISE:.0f} mm over a {ARCH_SPAN} mm arch span"
    )
    print(
        f"INFO section: {len(built)} of {len(gaps)} spandrel bents carry a post; "
        f"tallest {max(gaps):.0f} mm at the springing, tightest {tightest:.0f} mm at the crown"
    )


def assert_the_abutments_reach_rock(terrain, blocks):
    """The seam no single system can see: steel-on-terrain.

    ``bridge-terrain`` knows the canyon and nothing about the bridge;
    ``bridge-roadway`` casts the abutments and has no ground to cast them into.
    Between them a skewback can hang in mid-air, or a deck seat can be buried in
    the hillside, and both compile.

    So this samples the DEM directly under every concrete block and asserts the
    block straddles the surface: founded below it, reaching above it.
    """
    for block in blocks:
        b = block.authored_aabb()
        cx, cy = (b.min.x + b.max.x) / 2, (b.min.y + b.max.y) / 2
        ground = terrain.height_at(x=cx, y=cy)
        if b.min.z > ground:
            raise AssertionError(
                f"'{block.name}' has its underside at z={b.min.z} but the canyon "
                f"surface under it is z={ground:.0f} — it is founded on {b.min.z - ground:.0f} "
                f"mm of air."
            )
        if b.max.z < ground:
            raise AssertionError(
                f"'{block.name}' tops out at z={b.max.z}, {ground - b.max.z:.0f} mm BELOW "
                f"the canyon surface (z={ground:.0f}) — it is buried, and whatever it was "
                f"meant to seat is bearing on hillside."
            )
        print(
            f"INFO abutment: '{block.name}' founded {ground - b.min.z:.0f} mm into rock "
            f"(surface z={ground:.0f}), reaching {b.max.z - ground:.0f} mm proud"
        )


# ── Arch Ribs ──────────────────────────────────────────────────────────────
def build_arch():
    nodes = _arch_nodes()
    els = []
    for side, v in (("l", -RIB_HALF), ("r", RIB_HALF)):
        upper = [W(u, v, arch_z(u)) for u in nodes]
        lower = [W(u, v, arch_z(u) - ARCH_DEPTH) for u in nodes]
        els.append(steel(f"arch_chord_up_{side}", ARCH_CHORD, upper))
        els.append(steel(f"arch_chord_lo_{side}", ARCH_CHORD, lower))
        for i, u in enumerate(nodes):
            els.append(
                steel(
                    f"arch_vert_{side}_{i}",
                    ARCH_WEB,
                    [W(u, v, arch_z(u)), W(u, v, arch_z(u) - ARCH_DEPTH)],
                )
            )
        for i in range(N_PANEL):
            u0, u1 = nodes[i], nodes[i + 1]
            els.append(
                steel(
                    f"arch_diag_{side}_{i}",
                    ARCH_WEB,
                    [W(u0, v, arch_z(u0) - ARCH_DEPTH), W(u1, v, arch_z(u1))],
                )
            )
    for i in range(0, N_PANEL, 2):
        u0, u1 = nodes[i], nodes[i + 1]
        za, zb = arch_z(u0) - ARCH_DEPTH, arch_z(u1) - ARCH_DEPTH
        els.append(
            steel(
                f"arch_lat_a_{i}",
                LAT_BRACE,
                [W(u0, -RIB_HALF, za), W(u1, RIB_HALF, zb)],
            )
        )
        els.append(
            steel(
                f"arch_lat_b_{i}",
                LAT_BRACE,
                [W(u0, RIB_HALF, za), W(u1, -RIB_HALF, zb)],
            )
        )
    return els


# ── Spandrel Bents ─────────────────────────────────────────────────────────
def build_spandrel():
    nodes = _arch_nodes()
    deck_bot = lambda u: z_deck(u) - DECK_DEPTH  # noqa: E731
    els = []
    for side, v in (("l", -RIB_HALF), ("r", RIB_HALF)):
        tops = []
        for i, u in enumerate(nodes):
            top, bot = deck_bot(u), arch_z(u)
            tops.append(top)
            if top - bot > MIN_POST:
                els.append(steel(f"post_{side}_{i}", POST, [W(u, v, bot), W(u, v, top)]))
        for i in range(N_PANEL):
            u0, u1 = nodes[i], nodes[i + 1]
            a0, a1 = arch_z(u0), arch_z(u1)
            if min(tops[i] - a0, tops[i + 1] - a1) < MIN_BRACE:
                continue
            els.append(
                steel(
                    f"span_x_a_{side}_{i}",
                    SPAN_BRACE,
                    [W(u0, v, a0), W(u1, v, tops[i + 1])],
                )
            )
            els.append(
                steel(
                    f"span_x_b_{side}_{i}",
                    SPAN_BRACE,
                    [W(u1, v, a1), W(u0, v, tops[i])],
                )
            )
    return els


# ── Deck Truss + Floor Beams ───────────────────────────────────────────────
def _deck_nodes():
    return [i * (LH / N_DECK) for i in range(N_DECK + 1)]


def build_deck_truss():
    nodes = _deck_nodes()
    els = []
    for side, v in (("l", -DECK_HALF), ("r", DECK_HALF)):
        top = [W(u, v, z_deck(u)) for u in nodes]
        bot = [W(u, v, z_deck(u) - DECK_DEPTH) for u in nodes]
        els.append(steel(f"deck_chord_top_{side}", DECK_CHORD, top))
        els.append(steel(f"deck_chord_bot_{side}", DECK_CHORD, bot))
        for i, u in enumerate(nodes):
            els.append(
                steel(
                    f"deck_vert_{side}_{i}",
                    DECK_WEB,
                    [W(u, v, z_deck(u)), W(u, v, z_deck(u) - DECK_DEPTH)],
                )
            )
        for i in range(N_DECK):
            u0, u1 = nodes[i], nodes[i + 1]
            els.append(
                steel(
                    f"deck_diag_{side}_{i}",
                    DECK_WEB,
                    [W(u0, v, z_deck(u0) - DECK_DEPTH), W(u1, v, z_deck(u1))],
                )
            )
    for i, u in enumerate(nodes):
        els.append(
            steel(
                f"floor_beam_{i}",
                FLOOR_BEAM,
                [
                    W(u, -DECK_HALF, z_deck(u) - DECK_DEPTH),
                    W(u, DECK_HALF, z_deck(u) - DECK_DEPTH),
                ],
            )
        )
    return els


# ── Roadway ────────────────────────────────────────────────────────────────
def build_roadway():
    els = []
    PAVE_T, PLATE_T, CURB_H, RAIL_H = 60, 120, 200, 1100
    road_half = DECK_HALF - 400
    top = z_deck(0)

    plate = [
        Point2D(x=-DECK_HALF, y=-PAVE_T - PLATE_T),
        Point2D(x=DECK_HALF, y=-PAVE_T - PLATE_T),
        Point2D(x=DECK_HALF, y=-PAVE_T),
        Point2D(x=-DECK_HALF, y=-PAVE_T),
    ]
    els.append(
        Sweep(
            name="deck_plate",
            profile=plate,
            path=[W(0, 0, top), W(LH, 0, top)],
            material=Material(key=STEEL),
        )
    )

    asph = [
        Point2D(x=-road_half, y=-PAVE_T),
        Point2D(x=road_half, y=-PAVE_T),
        Point2D(x=road_half, y=0),
        Point2D(x=-road_half, y=0),
    ]
    els.append(
        Sweep(
            name="asphalt",
            profile=asph,
            path=[W(0, 0, top), W(LH, 0, top)],
            material=Material(key=ASPHALT),
        )
    )

    dash, gap, w = 4000, 6000, 150
    u = 6000
    k = 0
    while u < LH - 6000:
        bar = [
            Point2D(x=-w // 2, y=0),
            Point2D(x=w // 2, y=0),
            Point2D(x=w // 2, y=6),
            Point2D(x=-w // 2, y=6),
        ]
        els.append(
            Sweep(
                name=f"centreline_{k}",
                profile=bar,
                path=[W(u, 0, top), W(u + dash, 0, top)],
                material=Material(key=PAINT),
            )
        )
        u += dash + gap
        k += 1

    for side, v in (("l", -1), ("r", 1)):
        cv = v * road_half
        curb = [
            Point2D(x=cv - v * 200, y=0),
            Point2D(x=cv, y=0),
            Point2D(x=cv, y=CURB_H),
            Point2D(x=cv - v * 200, y=CURB_H),
        ]
        els.append(
            Sweep(
                name=f"curb_{side}",
                profile=curb,
                path=[W(0, 0, top), W(LH, 0, top)],
                material=Material(key=CONCRETE),
            )
        )
        ev = v * DECK_HALF
        els.append(
            steel(
                f"rail_top_{side}",
                RAIL,
                [W(0, ev, top + RAIL_H), W(LH, ev, top + RAIL_H)],
            )
        )
        for i in range(N_DECK + 1):
            u_node = i * (LH / N_DECK)
            els.append(
                steel(
                    f"rail_post_{side}_{i}",
                    RAIL_POST,
                    [W(u_node, ev, top), W(u_node, ev, top + RAIL_H)],
                )
            )
    return els


# ── Abutments ──────────────────────────────────────────────────────────────
def _block(name, u_half, v_half, u, z_bottom, height):
    corners = [
        W(u - u_half, -v_half, z_bottom),
        W(u + u_half, -v_half, z_bottom),
        W(u + u_half, v_half, z_bottom),
        W(u - u_half, v_half, z_bottom),
    ]
    block = Element(ifc_class="IfcFooting", name=name)
    # I(): the seat and pad heights are DERIVED off the DEM now, so they arrive as
    # floats. Extrude thickness is int mm.
    block.add(Extrude(contour=corners, thickness=I(height), material=CONCRETE))
    block.add(*build_reinforcement(block))
    return block


def build_abutments():
    """Skewbacks at the springings, deck seats on the rims — both sized to the GROUND.

    The seat height is derived, not typed. A 5000 mm seat is right at one rim and
    2.5 m short at the other, because the two rims are not the same height and the
    deck is nearly level; typed, it leaves the seat hanging in air on whichever side
    happens to be lower, and no gate in the repo would say so.
    """
    els = []
    # The springing sits ON the surface by construction (that is what _solve_the_
    # springing found), so the skewback is cast symmetrically about it.
    for tag, u in (("sw_arch", ARCH_U0), ("ne_arch", ARCH_U1)):
        els.append(
            _block(
                f"skewback_{tag}",
                3000,
                RIB_HALF + 1500,
                u,
                Z_SPRING - SKEWBACK_EMBEDMENT,
                SKEWBACK_EMBEDMENT + 1500,
            )
        )
    for tag, u in (("sw_deck", 0), ("ne_deck", LH)):
        zt = z_deck(u) - DECK_DEPTH
        founding = ground_at(u) - SEAT_EMBEDMENT
        els.append(_block(f"deckseat_{tag}", 2000, DECK_HALF + 800, u, founding, zt - founding))
    return els


# ── Approach bents (bridge-deck-truss, off the arch) ───────────────────────
def build_approach_bents():
    """Carry the deck where the arch does not reach.

    The arch spans the gorge; the deck spans rim to rim, and the two are not the
    same length. Between the springing and each rim the deck can end up with
    nothing under it — a 60 m unsupported run that reads as fine because a bridge
    authored ``carve="none"`` never reports anything.

    One bent per deck-truss node outside the arch: two posts from a pad footing on
    the canyon wall up to the deck soffit, a cap beam across, and X-bracing once the
    bent is tall enough to need it.
    """
    els, built = [], []
    for i in range(N_DECK + 1):
        u = i * (LH / N_DECK)
        if ARCH_U0 <= u <= ARCH_U1:
            continue
        if min(abs(u), abs(LH - u)) < 1.0:
            continue  # the rims carry deck SEATS, not bents
        soffit = z_deck(u) - DECK_DEPTH
        base = ground_at(u)
        height = soffit - base
        if height < MIN_POST:
            continue
        built.append((u, height))
        els.append(_block(f"bent_pad_{i:02d}", 2400, RIB_HALF + 1200, u, base - SEAT_EMBEDMENT,
                          SEAT_EMBEDMENT + 800))
        for side, v in (("l", -RIB_HALF), ("r", RIB_HALF)):
            els.append(steel(f"bent_post_{i:02d}_{side}", POST,
                             [W(u, v, base + 800), W(u, v, soffit)]))
        els.append(steel(f"bent_cap_{i:02d}", FLOOR_BEAM,
                         [W(u, -RIB_HALF, soffit), W(u, RIB_HALF, soffit)]))
        if height >= MIN_BRACE:
            els.append(steel(f"bent_x_a_{i:02d}", SPAN_BRACE,
                             [W(u, -RIB_HALF, base + 800), W(u, RIB_HALF, soffit)]))
            els.append(steel(f"bent_x_b_{i:02d}", SPAN_BRACE,
                             [W(u, RIB_HALF, base + 800), W(u, -RIB_HALF, soffit)]))
    print(
        f"INFO approach: {len(built)} trestle bents carry the deck outside the arch -- "
        + ", ".join(f"u={u:.0f} h={h:.0f}" for u, h in built)
    )
    return els


def build_bridge_site():
    """Wrap the canyon in the spatial structure. The mesh itself is built at module
    scope, above, because the ARCH is derived from it — by the time the site exists
    the springing level has already been solved against the ground."""
    site = Site(name="site")
    terrain = Element(
        ifc_class="IfcGeographicElement", predefined_type="TERRAIN", name="terrain"
    )
    terrain.add(CANYON)
    site.add(terrain)
    return site


# ── Project Assembly ───────────────────────────────────────────────────────
def generate_project():
    # Before anything is built. A section that does not close is cheaper to fail
    # here than to find in a render of 900 steel members.
    assert_the_span_closes()

    proj = Project(name="High Steel Bridge — Steel Truss Deck Arch")
    site = build_bridge_site()

    bridge = SpatialElement(
        ifc_class="IfcBridge", name="high_steel", predefined_type="ARCHED"
    )
    arch = SpatialElement(
        ifc_class="IfcBridgePart",
        name="arch",
        usage="VERTICAL",
        predefined_type="SUPERSTRUCTURE",
    )
    arch_steel = build_arch() + build_spandrel()
    arch.add(*arch_steel)

    deck = SpatialElement(
        ifc_class="IfcBridgePart",
        name="deck",
        usage="VERTICAL",
        predefined_type="DECK",
    )
    deck_steel = build_deck_truss()
    deck.add(*deck_steel)

    surfacing = SpatialElement(
        ifc_class="IfcBridgePart",
        name="surfacing",
        usage="VERTICAL",
        predefined_type="SURFACESTRUCTURE",
    )
    road = build_roadway()
    surfacing.add(*road)

    abutments = SpatialElement(
        ifc_class="IfcBridgePart",
        name="abutments",
        usage="VERTICAL",
        predefined_type="ABUTMENT",
    )
    blocks = build_abutments()
    abutments.add(*blocks)

    approach = SpatialElement(
        ifc_class="IfcBridgePart",
        name="approach",
        usage="VERTICAL",
        predefined_type="PIER",
    )
    bents = build_approach_bents()
    approach.add(*bents)

    # Every concrete block in the model, checked against the DEM in one place.
    assert_the_abutments_reach_rock(
        CANYON, [e for e in blocks + bents if isinstance(e, Element)]
    )

    parts = (
        (arch, arch_steel),
        (deck, deck_steel),
        (surfacing, road),
        (abutments, blocks),
        (approach, bents),
    )
    bridge.add(*[p for p, _ in parts])
    site.add(bridge, carve="none")
    proj.add(site)

    print(
        f"INFO bridge: one IfcBridge, {len(parts)} IfcBridgeParts -- "
        + ", ".join(f"{p.name} {len(members)}" for p, members in parts)
    )
    return proj


result = generate_project()


if __name__ == "__main__":
    compile_main()
