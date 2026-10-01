"""Tests for ``parent.anchor(child)`` — placement in the parent's local frame.

``.add()`` is pure containment (world coordinates, child untouched);
``.anchor()`` authors the child around a local origin and lets the parent
place it. The bake (``frames.resolve_child_anchors``) runs on the normalized
meters project before displacement, so an anchored child holds true world
coordinates by the time carves are inferred.
"""
from __future__ import annotations

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.compiler.frames import (
    ChildAnchorError,
    resolve_child_anchors,
    stamp_frames,
)
from lite_step.models import (
    Box,
    Extrude,
    Point,
    Project,
    Storey,
    Transform,
    Wall,
)


def _south_wall(name="south"):
    return Wall(name=name).add(
        Box(start=Point(x=-6000, y=-4500, z=0), end=Point(x=6000, y=-4200, z=2700)))


def _north_wall():
    return Wall(name="north").add(
        Box(start=Point(x=-6000, y=4200, z=0), end=Point(x=6000, y=4500, z=2700)))


def _payload(name="shelf"):
    return Box(name=name, start=Point(x=0, y=0, z=0), end=Point(x=400, y=300, z=600))


def _resolved(*elements):
    """Normalize + stamp + bake — the generation-time order."""
    storey = Storey()
    storey.add(*elements)
    proj = Project(name="anchor-test")
    proj.storeys = [storey]
    normalized = normalize_project_to_meters(proj)
    stamp_frames(normalized)
    resolve_child_anchors(normalized)
    return normalized


def _child(project, wall_index, name):
    wall = project.storeys[0].elements[wall_index]
    for c in wall._elements:
        if getattr(c, "name", None) == name:
            return c
    raise AssertionError(f"no child {name!r}")


# ---------------------------------------------------------------------------
# .add() stays pure containment — the counterpart contract
# ---------------------------------------------------------------------------


def test_add_never_moves_the_child():
    """PINNED: ``.add()`` is containment only. Coordinates and rotations come
    out bit-identical to what was authored."""
    wall = _south_wall()
    box = _payload()
    before = (box.start.x, box.start.y, box.start.z,
              box.end.x, box.end.y, box.end.z)
    wall.add(box)
    after = (box.start.x, box.start.y, box.start.z,
             box.end.x, box.end.y, box.end.z)
    assert before == after
    assert box._anchor_spec is None


# ---------------------------------------------------------------------------
# The bake
# ---------------------------------------------------------------------------


def test_anchor_places_along_up_from_the_corner_origin():
    wall = _south_wall()
    wall.anchor(_payload(), along=2000, up=900)
    proj = _resolved(wall)
    shelf = _child(proj, 0, "shelf")
    # wall x-min = -6.0 -> along 2.0 lands at -4.0; z-min 0 -> up 0.9.
    assert shelf.start.x == pytest.approx(-4.0)
    assert shelf.end.x == pytest.approx(-3.6)
    assert shelf.start.z == pytest.approx(0.9)
    assert shelf.end.z == pytest.approx(1.5)


def test_out_is_symmetric_on_opposite_walls():
    """THE parameter's contract: ``out=thickness`` flushes a child with the
    OUTSIDE of its host on both the south and the north wall — in opposite
    world directions."""
    south, north = _south_wall(), _north_wall()
    south.anchor(_payload("s"), along=6000, up=2000, inset=-300)
    north.anchor(_payload("n"), along=6000, up=2000, inset=-300)
    proj = _resolved(south, north)
    s, n = _child(proj, 0, "s"), _child(proj, 1, "n")
    assert min(s.start.y, s.end.y) == pytest.approx(-4.8)
    assert max(s.start.y, s.end.y) == pytest.approx(-4.5)
    assert min(n.start.y, n.end.y) == pytest.approx(4.5)
    assert max(n.start.y, n.end.y) == pytest.approx(4.8)


def test_out_zero_sits_against_the_inside():
    south, north = _south_wall(), _north_wall()
    south.anchor(_payload("s"), along=0, up=0, inset=0)
    north.anchor(_payload("n"), along=0, up=0, inset=0)
    proj = _resolved(south, north)
    s, n = _child(proj, 0, "s"), _child(proj, 1, "n")
    assert max(s.start.y, s.end.y) == pytest.approx(-4.2)
    assert min(n.start.y, n.end.y) == pytest.approx(4.2)


def test_anchored_child_does_not_corrupt_its_parents_frame():
    """Regression: an unbaked anchored child's coordinates are PARENT-LOCAL.
    Counting them in the parent's AABB dragged the frame origin toward the
    local origin, which then placed the child against a frame its own
    presence had corrupted (a north-wall child landed near y=0)."""
    north = _north_wall()
    north.anchor(_payload("n"), along=0, up=0, inset=0)
    proj = _resolved(north)
    frame = proj.storeys[0].elements[0]._frame
    assert frame.origin[1] == pytest.approx(4.2), "parent frame pulled to the local origin"


def test_quarter_turn_on_a_box_is_baked():
    wall = _south_wall()
    wall.anchor(_payload(), along=1000, up=0, rotations=[("z", 9000)])
    proj = _resolved(wall)
    shelf = _child(proj, 0, "shelf")
    # A 400x300 footprint rotated 90 degrees becomes 300x400 in plan.
    assert abs(shelf.end.x - shelf.start.x) == pytest.approx(0.3)
    assert abs(shelf.end.y - shelf.start.y) == pytest.approx(0.4)


def test_extrude_takes_any_z_rotation():
    wall = _south_wall()
    plate = Extrude(
        name="plate",
        contour=[Point(x=0, y=0, z=0), Point(x=600, y=0, z=0),
                 Point(x=600, y=0, z=400), Point(x=0, y=0, z=400)],
        thickness=20,
    )
    wall.anchor(plate, along=1000, up=1000, rotations=[("z", 3000)])
    proj = _resolved(wall)
    baked = _child(proj, 0, "plate")
    assert len(baked.contour) == 4
    # Rotated 30 degrees in plan: the first edge is no longer axis-aligned.
    assert baked.contour[1].y != pytest.approx(baked.contour[0].y)


def test_bake_is_idempotent():
    wall = _south_wall()
    wall.anchor(_payload(), along=2000, up=900)
    proj = _resolved(wall)
    once = _child(proj, 0, "shelf").start.x
    resolve_child_anchors(proj)          # second run must be a no-op
    assert _child(proj, 0, "shelf").start.x == pytest.approx(once)


def test_anchored_child_rides_a_placed_parent():
    """A wall with placement= moves its anchored children rigidly: the bake
    lands them in the wall's AUTHORING space, and the placement matrix is
    applied to the whole family at IFC level."""
    wall = _south_wall()
    wall.placement = Transform(origin=Point(x=10000, y=0, z=0))
    wall.anchor(_payload(), along=0, up=0)
    proj = _resolved(wall)
    shelf = _child(proj, 0, "shelf")
    assert shelf.start.x == pytest.approx(-6.0)


# ---------------------------------------------------------------------------
# Loud failures
# ---------------------------------------------------------------------------


def test_placement_and_anchor_are_mutually_exclusive():
    wall = _south_wall()
    box = _payload()
    box.placement = Transform(origin=Point(x=0, y=0, z=0))
    with pytest.raises(ValueError, match="mutually exclusive"):
        wall.anchor(box)


def test_a_child_has_one_parent_frame():
    box = _payload()
    _south_wall().anchor(box)
    with pytest.raises(ValueError, match="already anchored"):
        _north_wall().anchor(box)


def test_anchor_has_no_type_table_at_all():
    """``.anchor()`` does not validate against ``.add()``'s table — a Wall
        takes a nested Wall, and everything else.

    The gate it replaced prevented nothing: ``Element(ifc_class="IfcWall")``
    was already anchorable and already held every geometry type, so a refused
    author wrapped the child and carried on. What replaces the gate is the
    emitter guarantee — ``test_anchor_matrix`` asserts every pairing puts
    geometry in the file. ``.add()`` keeps its own table (below)."""
    nested = Wall(name="nested")
    nested.add(Box(name="body", start=Point(x=0, y=0, z=0),
                   end=Point(x=300, y=120, z=200)))
    wall = _south_wall()
    wall.anchor(nested)
    assert nested in wall._elements
    assert nested._anchor_spec is not None


def test_add_takes_the_same_child_and_does_not_anchor_it():
    """``.add()`` generalised too, and the verbs stayed distinct anyway.

    A nested ``Wall`` is ordinary containment now. What ``.add()`` does NOT do
    is stamp an anchor — its claim is that the child's coordinates are already
    world coordinates, so nothing is baked. That is the whole remaining
    difference, and it is a claim about coordinates, never about types."""
    wall = _south_wall()
    nested = Wall(name="nested")
    nested.add(Box(name="body", start=Point(x=0, y=0, z=0),
                   end=Point(x=300, y=120, z=200)))
    wall.add(nested)
    assert nested in wall._elements
    assert nested._anchor_spec is None


def test_non_z_rotation_is_a_loud_error():
    """At the ``.anchor()`` line, not at compile depth."""
    wall = _south_wall()
    with pytest.raises(ChildAnchorError, match="up axis"):
        wall.anchor(_payload(), rotations=[("x", 9000)])


def test_box_off_quarter_turn_is_a_loud_error():
    wall = _south_wall()
    with pytest.raises(ChildAnchorError, match="quarter turns") as exc:
        wall.anchor(_payload(), rotations=[("z", 3000)])
    assert "Box" in str(exc.value), "the error must name the refusing leaf"
    assert "Extrude" in str(exc.value), "…and the way out"


# ---------------------------------------------------------------------------
# Rotation legality is decided at the ASSEMBLY CALL SITE
# ---------------------------------------------------------------------------
# A legal assembly must not become illegal because of a leaf's type
# discovered at compile depth: a Box refusing 17 degrees inside an anchor
# chain, with the traceback pointing at the bake. The rules are
# unchanged; where they are DISCOVERED is not.


def _assembly(name, leaf):
    """A container holding one leaf — the shape an anchored assembly takes."""
    from lite_step.models import Element

    return Element(ifc_class="IfcBuildingElementProxy", name=name).add(leaf)


def test_seventeen_degrees_on_a_nested_box_fails_at_the_anchor_line():
    """The reviewer's case: the Box that refuses is INSIDE the thing being
    anchored, and the error still arrives on the line that asked for the
    rotation rather than deep inside the bake."""
    wall = _south_wall()
    assembly = _assembly("bracket", _payload("corbel"))

    with pytest.raises(ChildAnchorError, match="quarter turns") as exc:
        wall.anchor(assembly, along=1000, up=1500, rotations=[("z", 1700)])
    msg = str(exc.value)
    assert "'corbel'" in msg, "names the leaf that refused"
    assert "'bracket'" in msg, "…and the child that was being anchored"
    assert "1700" in msg, "…and the angle it refused"
    assert "Extrude" in msg, "…and the way out"


def test_the_same_seventeen_degrees_is_accepted_on_an_Extrude_leaf():
    """The check must not refuse what the bake accepts — an Extrude rotates
    fine, so the early error has to know that too."""
    wall = _south_wall()
    assembly = _assembly("panel", Extrude(name="fin", thickness=50, contour=[
        Point(x=0, y=0, z=0), Point(x=600, y=0, z=0),
        Point(x=600, y=0, z=400), Point(x=0, y=0, z=400)]))
    wall.anchor(assembly, along=1000, up=1500, rotations=[("z", 1700)])
    assert assembly._anchor_spec is not None


def test_quarter_turns_and_zero_rotation_still_pass_the_early_check():
    """The other direction of the same falsification: a rule that refuses
    everything is not a rule."""
    _south_wall().anchor(_payload("a"), rotations=[("z", 9000)])
    _south_wall().anchor(_payload("b"))
    _south_wall().anchor(_payload("c"), rotations=[("z", -18000)])


def test_a_refused_anchor_leaves_the_container_untouched():
    """A raise mid-``.anchor()`` must not half-attach the child — otherwise the
    author's next compile reports a second, unrelated problem."""
    wall = _south_wall()
    before = len(wall._elements)
    payload = _payload("shelf")
    with pytest.raises(ChildAnchorError):
        wall.anchor(payload, rotations=[("z", 3000)])
    assert len(wall._elements) == before
    assert payload._anchor_spec is None
    assert payload._parent is None


def test_the_early_check_and_the_bake_read_ONE_rule():
    """Two copies of one rule agreeing *usually* is.

    ``check_anchor_rotation`` and ``_bake_geometry`` must both resolve through
    ``rotation_refusal`` — pinned by construction, so a future edit that
    reintroduces an inline copy in either place fails here.
    """
    import inspect

    from lite_step.compiler import frames

    for fn in (frames.check_anchor_rotation, frames._bake_geometry):
        assert "rotation_refusal" in inspect.getsource(fn), (
            f"{fn.__name__} must read the shared rule, not restate it"
        )


def test_anchoring_to_a_geometryless_container_is_a_loud_error():
    wall = Wall(name="empty")          # no body child -> no frame
    wall.anchor(_payload())
    with pytest.raises(ChildAnchorError, match="geometry"):
        _resolved(wall)


def test_strict_int_mm_applies_to_anchor_offsets():
    with pytest.raises(Exception):
        _south_wall().anchor(_payload(), along=1500.5)


# ---------------------------------------------------------------------------
# Backend gate
# ---------------------------------------------------------------------------


def test_anchored_solid_on_a_wall_is_emitted_not_refused():
    """There is no refusal to pin: a Wall aggregates every child that is not
        its body or an opening (``_aggregate_wall_details``).

    Both halves are asserted, because either alone would be satisfied by the
    old bug — validation passing means nothing if the solid then vanishes,
    and this exact pairing is what made ``gable-detail`` author its gable ends
    as ``Element(ifc_class="IfcWall")`` instead of ``Wall``."""
    from lite_step.compiler.executor import validate_project_report
    from lite_step.ifc.generator import generate_ifc

    wall = _south_wall()
    wall.anchor(_payload("sign"), along=1000, up=1000, inset=-320)
    storey = Storey()
    storey.add(wall)
    proj = Project(name="p")
    proj.storeys = [storey]

    assert validate_project_report(proj).errors == []

    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    assert "sign" in result.ifc_content, "the Wall dropped its anchored child"


@pytest.mark.parametrize("container_factory,body,mark", [
    ("Slab", (0, 0, 0, 4000, 3000, 200), "slab_mark"),
    ("Column", (0, 0, 0, 400, 400, 3000), "col_mark"),
    ("Beam", (0, 0, 2400, 5000, 200, 2700), "beam_mark"),
    ("Roof", (0, 0, 3000, 4000, 3000, 3200), "roof_mark"),
])
def test_aggregating_containers_emit_their_anchored_children(
        container_factory, body, mark):
    """Slab/Column/Beam/Roof aggregate their children, so an anchored solid
    survives all the way into the IFC (verified end-to-end by compiling).

    The full ``{every container} x {every type}`` sweep is
    ``test_anchor_matrix.py``; this one stays as the narrow named case."""
    import lite_step.models as m
    from lite_step.ifc.generator import generate_ifc

    cls = getattr(m, container_factory)
    x0, y0, z0, x1, y1, z1 = body
    host = cls(name="host")
    host.add(Box(name="body", start=Point(x=x0, y=y0, z=z0),
                 end=Point(x=x1, y=y1, z=z1)))
    host.anchor(Box(name=mark, start=Point(x=0, y=0, z=0),
                    end=Point(x=200, y=200, z=200)), along=1000, up=0)
    storey = Storey()
    storey.add(host)
    proj = Project(name="p")
    proj.storeys = [storey]

    result = generate_ifc(normalize_project_to_meters(proj))
    assert mark in result.ifc_content, (
        f"{container_factory} dropped its anchored child")
