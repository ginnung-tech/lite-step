"""``world_aabb()`` — the extent AS PLACED, and the errors that protect it.

The stage rule in one line: ``authored_aabb()`` never changes meaning, and
``world_aabb()`` RAISES until the answer is decided. A helper called too early
fails at dev time instead of returning numbers that are about to move.

Most of these assert that the two agree, which is the point rather than a
weak test: this DSL has no local coordinates, so a Box authored at x=3170 IS
at x=3170. They diverge only where something MOVES the element — and each of
those cases is pinned separately.
"""
import pytest

from lite_step.models import (
    Anchor,
    Box,
    Element,
    Point,
    Project,
    Transform,
    Wall,
    Window,
)
from lite_step.models.bounds import BoundsError
from lite_step.models.project import Storey

_X, _Y, _Z = 3170, -2410, 1130


def _box(name="body", dx=1210, dy=347, dz=2703, x=_X, y=_Y, z=_Z):
    return Box(start=Point(x=x, y=y, z=z),
               end=Point(x=x + dx, y=y + dy, z=z + dz), name=name)


def _wall(name="south"):
    return Wall(name=name).add(_box())


# ---------------------------------------------------------------------------
# The gate: no root, no answer
# ---------------------------------------------------------------------------


def test_unrooted_element_raises_and_says_what_is_missing():
    wall = _wall()
    with pytest.raises(BoundsError, match="not reachable from a Project"):
        wall.world_aabb()


def test_the_error_offers_the_accessor_that_DOES_work():
    """An error that only says no costs a round trip. This one names
    ``authored_aabb()``, which is answerable right now."""
    with pytest.raises(BoundsError, match="authored_aabb"):
        _wall().world_aabb()


def test_parented_but_not_rooted_still_raises():
    """The case an earlier spec draft allowed, wrongly.

    A wall inside a Storey that is not in a Project can still move — answering
    hands out a world coordinate that is going to change, which is the defect
    the stage split exists to remove, one level up.
    """
    wall = _wall()
    Storey(name="ground").add(wall)          # storey never added to a project
    with pytest.raises(BoundsError, match="not reachable from a Project"):
        wall.world_aabb()


def test_authored_aabb_answers_where_world_aabb_refuses():
    wall = _wall()
    assert wall.authored_aabb().min.x == _X   # no raise
    with pytest.raises(BoundsError):
        wall.world_aabb()


def test_rooting_it_makes_the_query_legal():
    wall = _wall()
    Project(name="t").add(wall)
    assert wall.world_aabb().min.x == _X


# ---------------------------------------------------------------------------
# With no placement, world IS authored
# ---------------------------------------------------------------------------


def test_unplaced_element_reports_the_same_box_in_both_stages():
    wall = _wall()
    Project(name="t").add(wall)
    a, w = wall.authored_aabb(), wall.world_aabb()
    assert (a.min, a.max) == (w.min, w.max)
    assert w.rule == "world"


def test_storey_elevation_does_not_shift_the_world_box():
    """Authored coordinates are absolute; a storey elevation is an IFC
    structural convenience, not a translation.

    The generator SUBTRACTS the elevation from each element's placement and
    re-adds it on the IfcBuildingStorey (``_relative_placements``), so the net
    world position is what the author wrote. If that ever changes, this test
    is where it surfaces rather than in a building 3 m underground.
    """
    wall = _wall()
    proj = Project(name="t")
    proj.add_storey(Storey(name="first", elevation=3000))
    proj.storeys[0].add(wall)
    assert wall.world_aabb().min.z == _Z


def test_nested_child_resolves_through_its_ancestor():
    """Placement matrices are keyed on TOP-LEVEL elements; a nested child
    inherits the nearest ancestor that has one — which is only reachable
    because ``.parent`` exists. §A and §B are one feature for this reason."""
    body = _box()
    wall = Wall(name="south").add(body)
    Project(name="t").add(wall)
    assert body.world_aabb().min.x == _X


# ---------------------------------------------------------------------------
# ...and where they diverge
# ---------------------------------------------------------------------------


def test_translation_moves_the_world_box_and_stays_exact():
    wall = Wall(name="moved").add(_box(x=0, y=0, z=0))
    wall.placement = Transform(origin=Point(x=5000, y=7000, z=0))
    Project(name="t").add(wall)

    assert wall.authored_aabb().min.x == 0
    w = wall.world_aabb()
    assert (w.min.x, w.min.y) == (5000, 7000)
    assert w.exact is True, "a pure translation preserves the extent"


def test_rotation_makes_the_answer_a_BOUND_and_says_so():
    """The axis-aligned box around a rotated box is strictly larger.

    A 2000 x 200 box turned 30 degrees spans 2000*cos30 + 200*sin30 = 1832
    across and 2000*sin30 + 200*cos30 = 1173 deep. Both are correct as a
    bound and wrong as an extent, so ``exact`` is False and ``obb()`` is the
    tight answer.
    """
    wall = Wall(name="rot").add(
        Box(start=Point(x=0, y=0, z=0), end=Point(x=2000, y=200, z=2000), name="b"))
    wall.placement = Transform(origin=Point(x=0, y=0, z=0), rotations=[("z", 3000)])
    Project(name="t").add(wall)

    w = wall.world_aabb()
    assert (w.size.x, w.size.y) == (1832, 1173)
    assert w.exact is False


def test_all_eight_corners_are_transformed_not_just_two():
    """Under a rotation the transformed MIN corner is not the min of the
    transformed box. Taking two corners would return a box that fails to
    contain the solid — silently, and only when rotated."""
    wall = Wall(name="rot").add(
        Box(start=Point(x=1000, y=0, z=0), end=Point(x=3000, y=400, z=100), name="b"))
    wall.placement = Transform(origin=Point(x=0, y=0, z=0), rotations=[("z", 9000)])
    Project(name="t").add(wall)

    w = wall.world_aabb()
    # 90 degrees about Z: x-extent and y-extent swap.
    assert (w.size.x, w.size.y) == (400, 2000)


# ---------------------------------------------------------------------------
# The documented side effect
# ---------------------------------------------------------------------------


def test_world_query_bakes_anchors_and_that_is_visible():
    """A world query triggers ``resolve_child_anchors``, which rewrites an
    anchored child's coordinates into parent space.

    Documented rather than hidden: after the first world query, an anchored
    child's ``authored_aabb()`` reports the BAKED coordinates. The pass is
    idempotent and the compile runs it anyway, so this changes when the
    numbers move, not whether they are right.
    """
    wall = _wall()
    shelf = Box(start=Point(x=0, y=0, z=0), end=Point(x=300, y=200, z=40),
                name="shelf")
    wall.anchor(shelf, along=507, up=903)
    proj = Project(name="t")
    proj.add(wall)

    assert shelf._anchor_spec is not None, "not baked yet"
    wall.world_aabb()
    assert shelf._anchor_resolved is True, "the world query baked it"


def test_openings_are_NOT_baked_by_a_world_query():
    """``frames._bake_walk`` skips Window/Door entirely — spec and subtree.

    An opening is not a solid the parent frame translates: its ``_anchor_spec``
    is consumed as SCALARS by the opening machinery at emission, and its
    children live in opening-local Z-up coordinates the wall's frame must
    never drag into world space. So a world query on the host leaves the
    opening untouched, and ``world_aabb()`` on the opening itself raises for
    a different reason entirely (it has no extent of its own).
    """
    wall = _wall()
    win = Window(width=1210, height=1403, name="w0")
    wall.anchor(win, along=507, up=903)
    proj = Project(name="t")
    proj.add(wall)

    wall.world_aabb()
    assert win._anchor_resolved is False, "openings are deliberately skipped"
    assert win._anchor_spec is not None
    with pytest.raises(BoundsError, match="HOST decides"):
        win.world_aabb()


def test_re_entrancy_raises_a_named_error_not_a_recursionerror():
    """If an internal pass reached back through the public accessor mid-bake,
    the bake would call itself and the author would see a ``RecursionError``
    from deep inside the compiler. Cheap insurance for a class of bug that is
    otherwise very hard to read."""
    import lite_step.models.elements as el

    wall = _wall()
    proj = Project(name="t")
    proj.add(wall)

    el._RESOLVING_WORLD = True
    try:
        with pytest.raises(BoundsError, match="re-entered during resolution"):
            wall.world_aabb()
    finally:
        el._RESOLVING_WORLD = False


# ---------------------------------------------------------------------------
# Staleness by ordering
# ---------------------------------------------------------------------------


def test_a_world_read_before_a_later_mutation_is_reported():
    """The one hazard root-reachability does NOT cover: a legitimate read,
    then an ancestor moves. It cannot be fixed retroactively, so it is made
    loud at compile — a message naming exactly which reads to move."""
    from lite_step.compiler.executor import validate_project_report

    wall = _wall()
    proj = Project(name="t")
    proj.add(wall)
    wall.world_aabb()                       # legitimate read...
    proj.add(_wall("north"))                # ...and now the model changed

    report = validate_project_report(proj)
    assert any("read before the model finished changing" in w
               for w in report.warnings)


def test_no_warning_when_reads_come_after_the_last_mutation():
    """Place parents before you read them — the authoring rule this enforces.
    It must stay silent when followed, or it is noise."""
    from lite_step.compiler.executor import validate_project_report

    proj = Project(name="t")
    proj.add(_wall("south"), _wall("north"))
    proj.storeys[0].elements[0].world_aabb()     # read AFTER the last add

    report = validate_project_report(proj)
    assert not any("read before the model finished changing" in w
                   for w in report.warnings)
