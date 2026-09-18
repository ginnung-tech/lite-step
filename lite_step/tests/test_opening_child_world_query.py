"""A world query INSIDE a Window/Door resolves through the opening's frame.

The bug this suite inverts into a pass: ``frames._bake_walk`` skips
``Window``/``Door`` subtrees ENTIRELY — deliberately, because an opening's
``_anchor_spec`` is consumed as SCALARS the void is built from and its
children are opening-LOCAL Z-up. But those children ARE reachable from the
``Project`` through ``_parent``, so every accessor answered for them, in
opening-local coordinates, with no error. Two windows anchored 600 mm apart
on the same wall reported the IDENTICAL ``world_aabb()`` for their own frame
boxes while the compiled ``IfcWindow`` placements were genuinely 600 mm
apart. A silent wrong number is the exact failure class the bounding-query
family exists to prevent.

The fix composes the opening's own frame in the QUERY path only
(``elements._world_matrix_for``); the bake still skips openings, so the void,
the zero-boolean-delta pins and both backends are untouched. What makes the
answer trustworthy is that the composition and the emission read ONE
derivation — ``frames.opening_host_frame`` + ``frames.opening_placement`` —
so this suite asserts against ``get_local_placement`` on the compiled
product, never against its own arithmetic. Two derivations that merely agree
today are what created the bug.

**On the "mirrored" positions.** A window anchored ``along=1000`` on a wall
running 0..6000 in X whose exterior resolves to ``+Y`` reports its frame at
world x 4000..5000, not 1000..2000. That is not a sign error: the v21 frame
is the wall's own — stand OUTSIDE, look at the facade, ``+x`` is your right —
so with the exterior on ``+Y`` the viewer faces ``-Y`` and their right is
``-X``. The compiled IFC says the same thing, which is what settles it.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.models import Box, Point, Project, Transform, Wall, Window
from lite_step.models.bounds import BoundsError

_H = 4500
_IN = _H - 300

#: The same 4-fold PINWHEEL ring ``test_opening_local_frame`` uses: each leaf
#: is the previous one turned 90° about Z, so "the four walls agree" is a
#: statement about the FRAME and not about an accidentally symmetric plan.
_LEAVES = {
    "south": ((-_H, -_H), (_IN, -_IN)),
    "east":  ((_IN, -_H), (_H, _IN)),
    "north": ((-_IN, _IN), (_H, _H)),
    "west":  ((-_H, -_IN), (-_IN, _H)),
}

_WIDTH, _HEIGHT = 1200, 1400


def _frame_box(name="frame"):
    """The joinery: a Box in OPENING-LOCAL coordinates, starting at the
    opening's own origin so a wrong composition shows up as the raw local
    numbers rather than as a plausible offset."""
    return Box(name=name, start=Point(x=0, y=0, z=0),
               end=Point(x=_WIDTH, y=90, z=_HEIGHT))


def _window(name, inset=0):
    return Window(width=_WIDTH, height=_HEIGHT, name=name).add(_frame_box())


def _one_wall(alongs=(1000, 1600), inset=0, placement=None):
    """One 6 m wall running +X, with a window per entry in ``alongs``."""
    wall = Wall(name="south")
    wall.add(Box(name="body", type="wall",
                 start=Point(x=0, y=0, z=0),
                 end=Point(x=6000, y=300, z=3000)))
    wins = []
    for i, along in enumerate(alongs):
        win = _window(f"w{i}", inset=inset)
        wall.anchor(win, along=along, up=900, inset=inset)
        wins.append(win)
    if placement is not None:
        wall.placement = placement
    proj = Project(name="opening-query")
    proj.add(wall)
    return proj, wall, wins


def _ring(along=1000, up=900, inset=0):
    proj = Project(name="opening-query-ring")
    wins = {}
    for name, (start, end) in _LEAVES.items():
        wall = Wall(name=name)
        wall.add(Box(name="body", type="wall",
                     start=Point(x=start[0], y=start[1], z=0),
                     end=Point(x=end[0], y=end[1], z=2700)))
        win = _window(f"w_{name}")
        wall.anchor(win, along=along, up=up, inset=inset)
        wins[name] = win
        proj.add(wall)
    return proj, wins


# ---------------------------------------------------------------------------
# Compile helpers — the cross-check reads the REAL placement, not our math
# ---------------------------------------------------------------------------


def _compiled(proj: Project, backend: str):
    pytest.importorskip("ifcopenshell")
    import ifcopenshell

    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    with tempfile.NamedTemporaryFile("w", suffix=".ifc", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(result.ifc_content)
        path = fh.name
    try:
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _placement_mm(product) -> tuple:
    """The product's world placement origin, converted back to authoring mm.

    ``ifcopenshell.util.placement.get_local_placement`` composes the whole
    ``IfcLocalPlacement`` chain, so this is what a viewer actually draws — the
    independent answer the query has to match.
    """
    import ifcopenshell.util.placement as plc

    m = plc.get_local_placement(product.ObjectPlacement)
    return (float(m[0][3]) * 1000.0, float(m[1][3]) * 1000.0,
            float(m[2][3]) * 1000.0)


def _proxies_by_name(model) -> dict:
    return {p.Name: p for p in model.by_type("IfcBuildingElementProxy")
            if p.Name}


# ---------------------------------------------------------------------------
# 1. The repro, inverted
# ---------------------------------------------------------------------------


def test_two_windows_at_different_along_report_different_positions():
    """The Finding-1 repro. Before the fix both frames answered ``min.x =
    0.0`` — the raw opening-local coordinate, identical and both wrong."""
    proj, wall, (wa, wb) = _one_wall(alongs=(1000, 1600))
    wall.world_aabb()                       # trigger the bake

    a = wa._elements[0].world_aabb()
    b = wb._elements[0].world_aabb()

    assert a.min.x != b.min.x, "two windows 600 mm apart still answer alike"
    assert abs(b.min.x - a.min.x) == 600, "the delta is the anchor delta"
    # And neither is the un-composed local coordinate.
    assert a.min.x != 0 and b.min.x != 0


def test_the_sill_is_the_anchor_up_exactly():
    """``up=900`` puts the joinery's own z=0 at world z=900 — no rounding,
    no half-thickness fudge."""
    proj, wall, (win,) = _one_wall(alongs=(900,))
    wall.world_aabb()
    assert win._elements[0].world_aabb().min.z == 900


def test_inset_moves_the_child_into_the_wall():
    """``inset=`` shifts the FILL and everything aggregated to it. The void
    is untouched — that is emission, not this query — but a child's world
    position must follow the fill it belongs to."""
    proj0, wall0, (flush,) = _one_wall(alongs=(1000,), inset=0)
    wall0.world_aabb()
    proj1, wall1, (recessed,) = _one_wall(alongs=(1000,), inset=50)
    wall1.world_aabb()

    a = flush._elements[0].world_aabb()
    b = recessed._elements[0].world_aabb()
    # The wall runs +X with its exterior on +Y, so "into the wall" is -Y.
    assert b.min.y - a.min.y == -50
    assert b.min.x == a.min.x and b.min.z == a.min.z


# ---------------------------------------------------------------------------
# 2. The query answer AGREES with the compiled IFC — the whole point
# ---------------------------------------------------------------------------


def test_query_matches_the_compiled_placement_on_one_wall():
    """Assert against ``get_local_placement``, not against our own
    arithmetic: the two disagreeing IS the bug, so only the real placement
    can falsify the fix."""
    proj, wall, wins = _one_wall(alongs=(1000, 1600))
    wall.world_aabb()
    answers = {w.name: w._elements[0].world_aabb() for w in wins}

    # A fresh tree: normalization mutates a copy, but the query above already
    # baked this one, and the compile must start from the authored numbers.
    fresh, _wall, _wins = _one_wall(alongs=(1000, 1600))
    proxies = _proxies_by_name(_compiled(fresh, "ifcopenshell"))

    for leaf, box in answers.items():
        name = f"box:frame:window:{leaf}:wall:south"
        assert name in proxies, sorted(proxies)
        emitted = _placement_mm(proxies[name])
        # A box child is emitted as a rect profile CENTRED on its placement,
        # so the placement origin is the world centre of the same box.
        centre = box.center_mm
        for got, want, axis in zip(emitted, centre, "xyz"):
            assert abs(got - want) < 1e-6, (
                f"{leaf} {axis}: IFC says {got}, the query says {want}")


def test_query_matches_the_compiled_placement_on_all_four_facades():
    """One authoring, four walls — including the two whose ``+x`` runs the
    other way. A facing sign that only worked on the south wall would land
    here."""
    proj, wins = _ring()
    next(iter(wins.values()))._parent.world_aabb()
    answers = {leaf: w._elements[0].world_aabb() for leaf, w in wins.items()}

    fresh, _wins = _ring()
    proxies = _proxies_by_name(_compiled(fresh, "ifcopenshell"))

    for leaf, box in answers.items():
        name = f"box:frame:window:w_{leaf}:wall:{leaf}"
        assert name in proxies, sorted(proxies)
        emitted = _placement_mm(proxies[name])
        for got, want, axis in zip(emitted, box.center_mm, "xyz"):
            assert abs(got - want) < 1e-6, (
                f"{leaf} {axis}: IFC says {got}, the query says {want}")

    # ... and the four are genuinely in four different places, so the check
    # above cannot pass by every wall resolving to the same frame.
    centres = {tuple(round(c, 3) for c in b.center_mm) for b in answers.values()}
    assert len(centres) == 4


def test_the_composition_is_not_the_identity():
    """Falsification guard. Every assertion above would also pass if the
    composed matrix were the identity AND the generator happened to emit the
    local numbers — so pin that the matrix actually moves the child."""
    import numpy as np

    from lite_step.models.elements import _opening_child_matrix

    proj, wall, (win,) = _one_wall(alongs=(1000,))
    wall.world_aabb()
    m = _opening_child_matrix(win)
    assert not np.allclose(m, np.eye(4)), (
        "the opening frame composed to the identity — the query is back to "
        "reading opening-local coordinates as world ones"
    )


# ---------------------------------------------------------------------------
# 3. Composition with the rest of the query family
# ---------------------------------------------------------------------------


def test_an_ancestor_placement_composes_on_top_of_the_opening_frame():
    """A ``placement=`` on the HOST moves the whole wall, opening and all —
    so it multiplies ONTO the opening's local frame, not under it."""
    proj0, wall0, (plain,) = _one_wall(alongs=(1000,))
    wall0.world_aabb()
    base = plain._elements[0].world_aabb()

    proj1, wall1, (moved,) = _one_wall(
        alongs=(1000,),
        placement=Transform(origin=Point(x=1000, y=2000, z=3000)))
    wall1.world_aabb()
    shifted = moved._elements[0].world_aabb()

    assert shifted.min.x - base.min.x == 1000
    assert shifted.min.y - base.min.y == 2000
    assert shifted.min.z - base.min.z == 3000


def test_obb_inside_an_opening_resolves_through_the_same_frame():
    """``obb()`` and ``world_aabb()`` share ``_world_matrix_for``, so the
    oriented answer must move with the axis-aligned one."""
    proj, wall, (wa, wb) = _one_wall(alongs=(1000, 1600))
    wall.world_aabb()
    ca = wa._elements[0].obb().center_mm
    cb = wb._elements[0].obb().center_mm
    assert abs(cb[0] - ca[0]) == 600


def test_authored_aabb_inside_an_opening_stays_opening_local():
    """``authored_aabb()`` is the PRE-placement stage by definition, so it
    keeps reporting the coordinates the author wrote. The asymmetry is the
    contract, not an oversight."""
    proj, wall, (win,) = _one_wall(alongs=(1000,))
    wall.world_aabb()
    assert win._elements[0].authored_aabb().min.x == 0


# ---------------------------------------------------------------------------
# 4. What did NOT change
# ---------------------------------------------------------------------------


def test_the_opening_itself_still_raises_and_points_at_the_host():
    """Only an opening's CHILDREN gained a world position. The opening
    carries a size and no position of its own, and saying so is correct,
    documented and separately tested."""
    proj, wall, (win,) = _one_wall(alongs=(1000,))
    wall.world_aabb()
    with pytest.raises(BoundsError, match="HOST decides"):
        win.world_aabb()
    with pytest.raises(BoundsError, match="HOST decides"):
        win.authored_aabb()


def test_the_opening_is_still_not_baked():
    """The fix is in the query path only — ``frames._bake_walk`` must still
    skip the opening, spec and subtree both, or the void loses the scalars it
    is built from."""
    proj, wall, (win,) = _one_wall(alongs=(1000,))
    wall.world_aabb()
    assert win._anchor_resolved is False
    assert win._anchor_spec is not None
    child = win._elements[0]
    assert child.start.x == 0, "the child's authored coordinates were moved"


def test_a_host_with_no_usable_body_raises_instead_of_guessing():
    """Loud failure over silent degradation: with no frame to compose, the
    honest answer is a named error, not the opening-local coordinate dressed
    up as a world one."""
    from lite_step.models.project import Storey

    wall = Wall(name="south")
    wall.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=6000, y=300, z=3000)))
    win = _window("w0")
    wall.anchor(win, along=1000, up=900)
    proj = Project(name="t")
    proj.add(wall)
    wall.world_aabb()

    # Strip the body AFTER the tree is built: the host now bears no frame.
    wall._elements = [c for c in wall._elements if c is not wall._elements[0]]
    with pytest.raises(BoundsError, match="no usable host frame"):
        win._elements[0].world_aabb()
    assert isinstance(proj.storeys[0], Storey)
