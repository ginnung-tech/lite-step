"""A bounding query answers in millimetres, whatever unit the tree is in.

``Bounds`` promises **integer millimetres** — ``size`` / ``min`` / ``max`` /
``center`` all round through ``round_half_up``. That promise is exact for the
domain a model is AUTHORED in, and it was silently void one stage later:
handed the metre-domain tree ``normalize_project_to_meters`` returns, every
sub-metre extent rounded to nothing and the box came back a lie.

Measured on ``5e3030d``, one wall whose body box is authored
``x 1000..2000, y 100..300, z 500..900`` mm::

    wall.world_aabb().size   ->  Point(x=1.0, y=0.0, z=0.0)
    wall.world_aabb().min    ->  Point(x=1.0, y=0.0, z=1.0)

``x`` survived only because 1000 mm is a whole metre. 200 mm and 400 mm became
ZERO, and a 508 mm wall leaf reported a 1 mm thickness — while
``frames.element_aabb``, reading the same geometry, was exact
.

**The cause is not the rounding.** ``bounds._pt`` already suspends the strict
int-mm gate and rounds deliberately, and in the authoring domain that is
correct to the millimetre — which is why the corpus never moved and why every
control here passes before the fix as well as after. The cause is that the
three accessors could not be TOLD which domain they were reading. That is the
same defect ``lite_step.compiler.extent`` fixed for the two frozen Material
facts, and its module docstring already states the rule
this file pins one layer out: *nothing about a coordinate discloses its unit*,
so the domain cannot be derived — only stated.

Three things are pinned, and the split matters:

1.  **The authoring domain is untouched.** Every number an author sees today
    is the number they see after. These tests pass on both sides of the fix;
    without them the suite could not tell a fix from a coincidence.
2.  **The metre domain, STATED, answers in the same millimetres.** These fail
    before the fix, quoting the rounded number.
3.  **The metre domain, UNSTATED, raises instead of answering zero** — and the
    guard is partial, which is asserted rather than implied. It sees an extent
    collapse to nothing; it cannot see a 6 m wall arriving as "6 mm", because
    6 mm is an ordinary millimetre extent.

Numbers, never agreement: a test that only checked ``world_aabb()`` against
``frames.element_aabb`` would have passed while both were rounded.
"""

from __future__ import annotations

import math

import pytest

from lite_step.compiler import frames
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.compiler.extent import MM_PER_METER
from lite_step.models import (
    Box,
    Extrude,
    Point,
    Project,
    Storey,
    Transform,
    Wall,
)
from lite_step.models.bounds import Bounds, BoundsError

# ---------------------------------------------------------------------------
# Fixtures. Two walls, and the difference between them is the whole point:
# ``_SUB_METRE`` has extents that do not survive being read as metres, and
# ``_WHOLE_METRE`` has extents that do. The second is the control.
# ---------------------------------------------------------------------------

#: The issue's own fixture, in millimetres.
_SUB_METRE = dict(start=(1000, 100, 500), end=(2000, 300, 900))
_SUB_METRE_MIN = (1000, 100, 500)
_SUB_METRE_MAX = (2000, 300, 900)
_SUB_METRE_CENTER = (1500, 200, 700)
_SUB_METRE_SIZE = (1000, 200, 400)

#: Every extent a whole number of metres. Nothing here is destroyed by being
#: read as metres, so it reads plausibly in BOTH domains — which is exactly
#: why it is the control and exactly why no guard can catch it. Longest
#: horizontal axis on X so that the wall frame's (along, across, up) is the
#: same triple as the AABB's (x, y, z) and one constant serves both.
_WHOLE_METRE = dict(start=(0, 0, 0), end=(3000, 1000, 2000))
_WHOLE_METRE_SIZE = (3000, 1000, 2000)

#: The layered-wall leaf : a 508 mm buildup. It rounds to 1 mm rather
#: than to 0, so it clears the collapse guard while still being wrong.
_LEAF_THICKNESS_MM = 508


def _wall(corners, name: str = "north") -> Wall:
    wall = Wall(name=name)
    wall.add(Box(name="body",
                 start=Point(x=corners["start"][0], y=corners["start"][1],
                             z=corners["start"][2]),
                 end=Point(x=corners["end"][0], y=corners["end"][1],
                           z=corners["end"][2]),
                 type="wall"))
    return wall


def _rooted(elem):
    """The element, in a Project, in the AUTHORING (millimetre) domain."""
    proj = Project(name="unit-domain")
    storey = Storey(name="ground", elevation=0)
    storey.add(elem)
    proj.add_storey(storey)
    return proj, elem


def _in_metres(elem):
    """The same element on the tree ``normalize_project_to_meters`` returns.

    It deep-copies, so the returned element is a DIFFERENT object standing in
    metres — the stage no author ever holds and every compiler pass does.
    """
    proj, _ = _rooted(elem)
    assert validate_project_report(proj).errors == []
    normalized = normalize_project_to_meters(proj) or proj
    return normalized, normalized.storeys[0].elements[0]


def _xyz(point):
    return (point.x, point.y, point.z)


# ---------------------------------------------------------------------------
# 1. The authoring domain is untouched — the control half of the suite
# ---------------------------------------------------------------------------


def test_an_authored_sub_metre_wall_already_reported_every_millimetre() -> None:
    """The accessors were never broken where authors use them.

    This is the assertion that keeps the fix honest: it passes on
    ``5e3030d`` and after, so a change that "fixed" the metre domain by
    disturbing the authoring one fails here.
    """
    _, wall = _rooted(_wall(_SUB_METRE))
    b = wall.world_aabb()

    assert _xyz(b.min) == _SUB_METRE_MIN
    assert _xyz(b.max) == _SUB_METRE_MAX
    assert _xyz(b.center) == _SUB_METRE_CENTER
    assert _xyz(b.size) == _SUB_METRE_SIZE


def test_an_authored_whole_metre_wall_reports_millimetres_not_metres() -> None:
    """The control element, in the control domain. ``(3000, 1000, 2000)`` and
    not ``(3, 1, 2)`` — so a fix that started answering in metres would be
    caught here rather than passing as "the numbers agree"."""
    _, wall = _rooted(_wall(_WHOLE_METRE))
    assert _xyz(wall.world_aabb().size) == _WHOLE_METRE_SIZE
    assert _xyz(wall.obb().size) == _WHOLE_METRE_SIZE


def test_the_authored_and_world_stages_agree_on_an_unplaced_wall() -> None:
    """``authored_aabb`` takes the same ``divisor`` and must not have been
    left behind — with nothing moving the wall the two stages are one box."""
    _, wall = _rooted(_wall(_SUB_METRE))
    assert _xyz(wall.authored_aabb().size) == _SUB_METRE_SIZE
    assert _xyz(wall.authored_aabb().min) == _SUB_METRE_MIN


# ---------------------------------------------------------------------------
# 2. The metre domain, stated — the numbers the issue reported destroyed
# ---------------------------------------------------------------------------


def test_a_metre_domain_wall_states_its_domain_and_gets_millimetres_back() -> None:
    """THE test. Falsify by reverting ``divisor`` on ``world_aabb``: ``size``
    comes back ``(1, 0, 0)`` and ``min`` ``(1, 0, 1)``."""
    _, wall = _in_metres(_wall(_SUB_METRE))
    b = wall.world_aabb(MM_PER_METER)

    assert _xyz(b.min) == _SUB_METRE_MIN
    assert _xyz(b.max) == _SUB_METRE_MAX
    assert _xyz(b.center) == _SUB_METRE_CENTER
    assert _xyz(b.size) == _SUB_METRE_SIZE


def test_the_metre_domain_answer_is_exact_and_not_merely_rounded_right() -> None:
    """``size`` is an int Point by contract, so the int assertions above cannot
    tell 200.0 from 200.4. The unrounded floats can, and 1e-9 is the tolerance
    the issue asked for."""
    _, wall = _in_metres(_wall(_SUB_METRE))
    b = wall.world_aabb(MM_PER_METER)

    for got, want in zip(b.size_mm, _SUB_METRE_SIZE):
        assert got == pytest.approx(want, abs=1e-9)
    for got, want in zip(b.center_mm, _SUB_METRE_CENTER):
        assert got == pytest.approx(want, abs=1e-9)


def test_obb_states_its_domain_the_same_way_world_aabb_does() -> None:
    """``obb`` reached ``Bounds`` down a different path (``from_frame``, via
    ``oriented_extent``), so it needs its own measurement — the issue reported
    the two identical, and identical-and-wrong is the easiest thing to
    preserve by fixing only one of them."""
    _, wall = _in_metres(_wall(_SUB_METRE))
    b = wall.obb(MM_PER_METER)

    assert _xyz(b.min) == _SUB_METRE_MIN
    assert _xyz(b.max) == _SUB_METRE_MAX
    assert _xyz(b.center) == _SUB_METRE_CENTER
    assert _xyz(b.size) == _SUB_METRE_SIZE


def test_a_508_mm_wall_leaf_does_not_report_a_1_mm_thickness() -> None:
    """the layered leaf. It rounds to 1 rather than to 0, so it is the case
    the collapse guard is blind to — and therefore the case that proves the
    fix is the stated domain and not the guard."""
    leaf = _wall(dict(start=(0, 0, 0), end=(6000, _LEAF_THICKNESS_MM, 3000)),
                 name="leaf")
    _, placed = _in_metres(leaf)

    assert _xyz(placed.world_aabb(MM_PER_METER).size) == (
        6000, _LEAF_THICKNESS_MM, 3000)
    assert placed.world_aabb(MM_PER_METER).size_mm[1] == pytest.approx(
        _LEAF_THICKNESS_MM, abs=1e-9)
    # The centre the issue reported as z=2.0 for a wall standing 0..3 m.
    assert _xyz(placed.world_aabb(MM_PER_METER).center) == (
        3000, _LEAF_THICKNESS_MM // 2, 1500)


def test_the_whole_metre_control_survives_the_metre_domain_too() -> None:
    """The control element in the OTHER domain. It reported ``(3, 1, 2)``
    before — three plausible numbers — and reports the same millimetres as the
    authoring domain after."""
    _, wall = _in_metres(_wall(_WHOLE_METRE))
    assert _xyz(wall.world_aabb(MM_PER_METER).size) == _WHOLE_METRE_SIZE
    assert _xyz(wall.obb(MM_PER_METER).size) == _WHOLE_METRE_SIZE


def test_point_at_addresses_a_face_in_millimetres_in_both_domains() -> None:
    """``point_at`` is what an author places joinery against, and it rounds at
    its own output — so a metre-domain box reached it already destroyed."""
    _, authored = _rooted(_wall(_SUB_METRE))
    _, placed = _in_metres(_wall(_SUB_METRE))

    for b in (authored.world_aabb(), placed.world_aabb(MM_PER_METER)):
        assert _xyz(b.point_at(0, -100, 0)) == (1500, 100, 700)
        assert _xyz(b.point_at(0, 100, 0)) == (1500, 300, 700)
        assert _xyz(b.point_at(-100, -100, -100)) == _SUB_METRE_MIN


# ---------------------------------------------------------------------------
# 3. A rotated element, where obb() and world_aabb() genuinely differ
# ---------------------------------------------------------------------------


def _slanted() -> Extrude:
    """``test_obb``'s diagonal wall: 3000x3000 in plan, 300 thick, 2500 up.

    Its OBB is 4243 x 300 x 2500 while its AABB is 3212 x 3212 x 2500, so the
    two accessors cannot borrow each other's answer — and the 300 mm thickness
    is the number the metre domain destroyed.
    """
    return Extrude(
        contour=[Point(x=0, y=0, z=0), Point(x=3000, y=3000, z=0),
                 Point(x=3000, y=3000, z=2500), Point(x=0, y=0, z=2500)],
        thickness=300, name="slanted")


def test_a_slanted_wall_keeps_its_300_mm_thickness_in_the_metre_domain() -> None:
    _, authored = _rooted(_slanted())
    _, placed = _in_metres(_slanted())

    want = (round(3000 * math.sqrt(2)), 300, 2500)
    assert _xyz(authored.obb().size) == want
    assert _xyz(placed.obb(MM_PER_METER).size) == want
    assert placed.obb(MM_PER_METER).rule == "extrude-normal"
    # The AABB really is the other box — so this pair measures two derivations.
    assert _xyz(placed.world_aabb(MM_PER_METER).size) == (3212, 3212, 2500)


def test_a_z_rotated_box_reports_the_same_bound_in_both_domains() -> None:
    """A ``Transform`` rotation puts a world matrix in the path, which is a
    third route into ``Bounds`` (``_transform_aabb``). 2000 x 200 turned 30
    degrees spans 1832 x 1173 — and the 200 mm depth that produces the 1173 is
    sub-metre."""
    def rotated() -> Wall:
        wall = Wall(name="rot").add(
            Box(start=Point(x=0, y=0, z=0), end=Point(x=2000, y=200, z=2000),
                name="b"))
        wall.placement = Transform(origin=Point(x=0, y=0, z=0),
                                   rotations=[("z", 3000)])
        return wall

    _, authored = _rooted(rotated())
    _, placed = _in_metres(rotated())

    assert (authored.world_aabb().size.x, authored.world_aabb().size.y) == (1832, 1173)
    b = placed.world_aabb(MM_PER_METER)
    assert (b.size.x, b.size.y) == (1832, 1173)
    assert b.exact is False


# ---------------------------------------------------------------------------
# 4. The containers that answer for a whole floor or a whole model
# ---------------------------------------------------------------------------


def test_a_storey_and_a_project_take_the_same_divisor() -> None:
    """``Storey`` is not a ``BimElement`` and carries its own copy of the
    query, and ``Project`` overrides it again — three implementations of one
    idea, which is how two of them keep a defect the third one lost."""
    proj, _ = _rooted(_wall(_SUB_METRE))
    assert _xyz(proj.authored_aabb().size) == _SUB_METRE_SIZE
    assert _xyz(proj.storeys[0].authored_aabb().size) == _SUB_METRE_SIZE

    normalized, _ = _in_metres(_wall(_SUB_METRE))
    assert _xyz(normalized.authored_aabb(MM_PER_METER).size) == _SUB_METRE_SIZE
    assert _xyz(
        normalized.storeys[0].authored_aabb(MM_PER_METER).size) == _SUB_METRE_SIZE


# ---------------------------------------------------------------------------
# 5. The guard — what an unstated metre-domain caller gets instead of zeros
# ---------------------------------------------------------------------------


def test_an_unstated_metre_domain_query_raises_instead_of_answering_zero() -> None:
    """The failure this issue was: ``size.y == 0.0`` for a 200 mm leaf, no
    error, no warning. Loud beats plausible."""
    _, wall = _in_metres(_wall(_SUB_METRE))

    with pytest.raises(BoundsError) as excinfo:
        wall.world_aabb()
    message = str(excinfo.value)
    assert "rounds to 0 mm" in message
    assert "divisor=1000.0" in message
    # It names the number it refused, so the reader can tell 0.2 m from 0.2 mm.
    assert "0.19999999999999998" in message or "0.2" in message


def test_the_unstated_refusal_covers_obb_and_authored_aabb_too() -> None:
    """Written the long way round so that FAILING here prints the number that
    came back — which is the whole evidence for the bug, and which
    ``pytest.raises`` would swallow into "DID NOT RAISE"."""
    _, wall = _in_metres(_wall(_SUB_METRE))
    for name in ("obb", "authored_aabb", "world_aabb"):
        try:
            got = getattr(wall, name)()
        except BoundsError:
            continue
        raise AssertionError(
            f"{name}() answered instead of refusing: size={_xyz(got.size)}, "
            f"min={_xyz(got.min)} — for a wall authored {_SUB_METRE_SIZE} mm "
            f"at {_SUB_METRE_MIN} mm.")


def test_the_guard_is_partial_and_the_whole_metre_control_walks_past_it() -> None:
    """Asserted rather than implied, because a guard sold as complete is worse
    than one documented as partial.

    Nothing distinguishes a 1 m extent read as 1 mm from a real 1 mm extent —
    that is the unit invariant, and it is why the fix is the stated domain and
    the guard is only the part of it that happens to be visible. This
    element's extents are whole metres, so every one of them survives the
    rounding and the query answers ``(1, 2, 3)`` with no complaint.
    """
    _, wall = _in_metres(_wall(_WHOLE_METRE))
    assert _xyz(wall.world_aabb().size) == (3, 1, 2)      # wrong, and silent


def test_a_genuinely_flat_box_is_not_mistaken_for_a_lost_extent() -> None:
    """Exactly zero is a real answer — a contour is a zero-width slab — and the
    guard must not confuse "no extent" with "an extent I destroyed"."""
    assert Bounds(center_mm=(0.0, 0.0, 0.0), size_mm=(1000.0, 0.0, 400.0)).size.y == 0


def test_float_residue_from_a_rotation_is_not_mistaken_for_a_lost_extent() -> None:
    """A rotation leaves ULP-scale residue where an extent is truly zero. At
    ~1e5 mm a float64 ULP is ~1e-11 mm, so the 1e-3 floor sits eight orders
    clear — pinned so a future tightening cannot creep into the noise."""
    b = Bounds(center_mm=(0.0, 0.0, 0.0), size_mm=(1000.0, 1e-11, 400.0))
    assert b.size.y == 0

    with pytest.raises(BoundsError):
        Bounds(center_mm=(0.0, 0.0, 0.0), size_mm=(1000.0, 1e-3, 400.0))


# ---------------------------------------------------------------------------
# 6. The constructors, directly
# ---------------------------------------------------------------------------


def test_from_aabb_multiplies_the_callers_units_up_into_millimetres() -> None:
    """``extent`` DIVIDES frozen mm facts down into the caller's units; this
    MULTIPLIES the caller's coordinates back up. One number, two directions —
    written down here because getting the direction wrong is silent and off by
    a factor of a million."""
    metres = Bounds.from_aabb(((1.0, 0.1, 0.5), (2.0, 0.3, 0.9)),
                              divisor=MM_PER_METER)
    millimetres = Bounds.from_aabb(((1000, 100, 500), (2000, 300, 900)))

    assert metres.size_mm == pytest.approx(millimetres.size_mm, abs=1e-9)
    assert metres.center_mm == pytest.approx(millimetres.center_mm, abs=1e-9)
    assert _xyz(metres.min) == _SUB_METRE_MIN


def test_from_frame_scales_the_extents_and_leaves_the_axes_alone() -> None:
    """``axes`` are unit vectors and carry no length. Scaling them would keep
    every extent right and put the box in the wrong place."""
    axes = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    b = Bounds.from_frame((1.5, 0.2, 0.7), (1.0, 0.2, 0.4), axes,
                          divisor=MM_PER_METER)

    assert b.size_mm == pytest.approx((1000.0, 200.0, 400.0), abs=1e-9)
    assert b.center_mm == pytest.approx((1500.0, 200.0, 700.0), abs=1e-9)
    assert b.axes == axes


# ---------------------------------------------------------------------------
# 7. Against the derivation the compiler actually reads
# ---------------------------------------------------------------------------


def test_the_query_and_the_compilers_own_aabb_describe_one_box() -> None:
    """Last, and never alone: ``frames.element_aabb`` was exact throughout, so
    agreement with it is worth checking only once the numbers above have been
    nailed down independently. Agreement on its own would have passed with
    both sides rounded."""
    _, placed = _in_metres(_wall(_SUB_METRE))
    (x0, y0, z0), (x1, y1, z1) = frames.element_aabb(placed)
    b = placed.world_aabb(MM_PER_METER)

    assert b.size_mm == pytest.approx(
        ((x1 - x0) * MM_PER_METER, (y1 - y0) * MM_PER_METER,
         (z1 - z0) * MM_PER_METER), abs=1e-9)
    # ...and against the authored millimetres, which is the number that has
    # not passed through either derivation.
    assert b.size_mm == pytest.approx(_SUB_METRE_SIZE, abs=1e-9)
