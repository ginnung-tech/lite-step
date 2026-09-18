"""Tests for the authoritative frame pass (``lite_step/compiler/frames.py``).

The frame is the single derivation of "which way does this element run, and
which side is out", replacing six independent AABB re-derivations. These
tests pin:

* the **corner origin** — local ``(0,0,0)`` is the AABB min corner in frame
  axes ("lower left back") for EVERY rule, including beam and slab whose
  legacy anchor lines sat at the centerline / top surface,
* the derived attach points still matching the legacy ``_host_frame`` table
  (baseline / centerline / top surface / exterior face), so the migration in
  WS1 is behaviour-preserving,
* the **extrusion-direction normal** — an authored contour's Newell normal
  wins over the AABB,
* **facing** pointing away from the nearest containment scope
  (Space -> Storey -> Building), including the off-origin case the legacy
  "farther from the world origin" rule got wrong,
* unit-following through normalize (the deepcopied-mm-frame trap).
"""
from __future__ import annotations

import pytest

from lite_step.compiler.frames import (
    ElementFrame,
    contour_normal,
    element_aabb,
    stamp_frames,
)
from lite_step.models import (
    Beam,
    Box,
    Extrude,
    Point,
    Project,
    Slab,
    Space,
    Storey,
    Wall,
)


def _project(*elements) -> Project:
    storey = Storey()
    storey.add(*elements)
    proj = Project(name="t")
    proj.storeys = [storey]
    return proj


def _wall(name, x0, y0, x1, y1, z0=0, z1=2700) -> Wall:
    return Wall(name=name).add(
        Box(start=Point(x=x0, y=y0, z=z0), end=Point(x=x1, y=y1, z=z1))
    )


def _south_north():
    """The spec's canonical fixture: 12 m walls, building centered at origin."""
    return _wall("south", -6000, -4500, 6000, -4200), _wall("north", -6000, 4200, 6000, 4500)


# ---------------------------------------------------------------------------
# Corner origin — the authoritative local (0,0,0)
# ---------------------------------------------------------------------------


def test_origin_is_the_min_corner_for_a_wall():
    """PINNED: local (0,0,0) is the AABB min corner ("lower left back"),
    NOT the wall baseline (which sits centered in the thickness)."""
    south, north = _south_north()
    stamp_frames(_project(south, north))
    f = south._frame
    assert f.origin == pytest.approx((-6000.0, -4500.0, 0.0))
    assert (f.extent_along, f.thickness, f.height) == pytest.approx((12000.0, 300.0, 2700.0))


def test_origin_is_the_min_corner_for_beam_and_slab_too():
    """The legacy anchor LINE differed per rule (beam centerline, slab top);
    the ORIGIN does not — it is the min corner for every rule."""
    beam = Beam(name="b").add(
        Box(start=Point(x=0, y=0, z=2400), end=Point(x=5000, y=200, z=2700))
    )
    slab = Slab(name="s").add(
        Box(start=Point(x=0, y=0, z=0), end=Point(x=4000, y=3000, z=200))
    )
    stamp_frames(_project(beam, slab))
    assert beam._frame.origin == pytest.approx((0.0, 0.0, 2400.0))
    assert slab._frame.origin == pytest.approx((0.0, 0.0, 0.0))
    assert beam._frame.rule == "beam"
    assert slab._frame.rule == "slab"


# ---------------------------------------------------------------------------
# Derived attach points reproduce the legacy _host_frame table
# ---------------------------------------------------------------------------


def test_wall_attach_points_match_the_legacy_table():
    """Baseline at the wall base, centered in thickness; face = exterior."""
    south, north = _south_north()
    stamp_frames(_project(south, north))
    f = south._frame
    assert f.point("start") == pytest.approx((-6000.0, -4350.0, 0.0))
    assert f.point("end") == pytest.approx((6000.0, -4350.0, 0.0))
    assert f.point("center") == pytest.approx((0.0, -4350.0, 1350.0))
    # Exterior of the south wall = the far side, y = -4500 (the pin in
    # test_placement_engine.test_attach_face_is_exterior_side).
    assert f.point("face") == pytest.approx((0.0, -4500.0, 1350.0))


def test_beam_line_is_the_centerline_and_face_is_the_top():
    beam = Beam(name="b").add(
        Box(start=Point(x=0, y=0, z=2400), end=Point(x=5000, y=200, z=2700))
    )
    stamp_frames(_project(beam))
    f = beam._frame
    assert f.point("start")[2] == pytest.approx(2550.0)   # centerline, not base
    assert f.point("face")[2] == pytest.approx(2700.0)    # top face


def test_slab_line_is_the_top_surface_and_axes_are_literal_world():
    slab = Slab(name="s").add(
        Box(start=Point(x=0, y=0, z=0), end=Point(x=4000, y=3000, z=200))
    )
    stamp_frames(_project(slab))
    f = slab._frame
    assert f.along == pytest.approx((1.0, 0.0, 0.0))
    assert f.across == pytest.approx((0.0, 1.0, 0.0))
    assert f.point("start")[2] == pytest.approx(200.0)    # top surface


def test_wall_running_along_y_swaps_the_axes():
    """along = the longer horizontal span; across = up x along."""
    east = _wall("east", 5700, -4000, 6000, 4000)
    stamp_frames(_project(east))
    f = east._frame
    assert f.along == pytest.approx((0.0, 1.0, 0.0))
    assert f.across == pytest.approx((-1.0, 0.0, 0.0))
    assert f.extent_along == pytest.approx(8000.0)
    assert f.thickness == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Extrusion direction wins over the AABB
# ---------------------------------------------------------------------------


def test_contour_wall_frame_comes_from_the_newell_normal():
    """An authored Extrude carries a real direction — use it, not the AABB."""
    contour = [
        Point(x=-4000, y=0, z=0),
        Point(x=4000, y=0, z=0),
        Point(x=4000, y=0, z=2500),
        Point(x=-4000, y=0, z=2500),
    ]
    wall = Wall(name="gable").add(Extrude(contour=contour, thickness=300))
    stamp_frames(_project(wall))
    f = wall._frame
    assert f.rule == "wall-contour"
    assert f.along == pytest.approx((1.0, 0.0, 0.0))
    assert f.across == pytest.approx((0.0, 1.0, 0.0))


def test_contour_run_axis_is_winding_independent():
    """Reversed winding flips the Newell normal but NOT the run axis — the
    canonicalization toward +X (ties +Y) is what keeps offsets stable."""
    pts = [
        Point(x=-4000, y=0, z=0),
        Point(x=4000, y=0, z=0),
        Point(x=4000, y=0, z=2500),
        Point(x=-4000, y=0, z=2500),
    ]
    fwd = Wall(name="a").add(Extrude(contour=list(pts), thickness=300))
    rev = Wall(name="b").add(Extrude(contour=list(reversed(pts)), thickness=300))
    stamp_frames(_project(fwd))
    stamp_frames(_project(rev))
    assert contour_normal(fwd._elements[0].contour) == pytest.approx((0.0, -1.0, 0.0))
    assert contour_normal(rev._elements[0].contour) == pytest.approx((0.0, 1.0, 0.0))
    assert fwd._frame.along == pytest.approx(rev._frame.along, abs=1e-9)


def test_box_wall_uses_the_aabb_rule():
    """No authored direction on a Box body -> the legacy longer-axis rule."""
    south, _ = _south_north()
    stamp_frames(_project(south))
    assert south._frame.rule == "wall-box"


# ---------------------------------------------------------------------------
# Facing — away from the nearest containment scope
# ---------------------------------------------------------------------------


def test_facing_points_away_from_the_building_center():
    south, north = _south_north()
    stamp_frames(_project(south, north))
    assert south._frame.facing == -1          # exterior on -across (-y)
    assert north._frame.facing == +1
    assert south._frame.point("face")[1] == pytest.approx(-4500.0)
    assert north._frame.point("face")[1] == pytest.approx(4500.0)


def test_facing_is_correct_for_an_off_origin_building():
    """THE behaviour fix. The legacy rule was "the across-face farther from
    the WORLD ORIGIN", which inverts once a building sits off-origin: for a
    wall at y[15500, 15800] it picks 15800 (the INTERIOR face) purely because
    |15800| > |15500|. The scope-center rule gets it right."""
    south = _wall("south", -6000, 15500, 6000, 15800)
    north = _wall("north", -6000, 24200, 6000, 24500)
    stamp_frames(_project(south, north))
    assert south._frame.point("face")[1] == pytest.approx(15500.0)
    assert north._frame.point("face")[1] == pytest.approx(24500.0)
    # The legacy rule would have said 15800 for the south wall:
    (_, y0, _z0), (_, y1, _z1) = element_aabb(south)
    legacy = y0 if abs(y0) > abs(y1) else y1
    assert legacy == pytest.approx(15800.0)


def test_an_enclosing_space_is_a_nearer_scope_than_the_building():
    """A Space accepts only Box children, so "the room this wall is in" is a
    SPATIAL question. A wall inside a room faces away from the ROOM."""
    inner = _wall("inner", -2000, 2000, 2000, 2300, z1=2500)
    room = Space(name="room")
    room.add(Box(start=Point(x=-2500, y=1800, z=0), end=Point(x=2500, y=8000, z=2500)))
    outer = _wall("outer", -9000, -10000, 9000, -9700, z1=2500)
    stamp_frames(_project(inner, room, outer))
    # Room center y = 4900 -> exterior of the inner wall is its y-min face.
    assert inner._frame.facing == -1
    assert inner._frame.point("face")[1] == pytest.approx(2000.0)
    # Outside every room -> building center governs.
    assert outer._frame.point("face")[1] == pytest.approx(-10000.0)


def test_authored_layerset_outward_wins_over_the_scope_center():
    from lite_step.models.material import LayerSet, Material

    # Point outward at +y — the OPPOSITE of what the building center implies
    # for a south wall (whose scope-derived exterior is -y).
    buildup = LayerSet(
        name="ext",
        outward=(0, 1, 0),
        layers=[Material(key="Brick_Red_DK", thickness_mm=108)],
    )
    wall = Wall(name="south", layers=buildup).add(
        Box(start=Point(x=-6000, y=-4500, z=0), end=Point(x=6000, y=-4392, z=2700))
    )
    stamp_frames(_project(wall))
    assert wall._frame.facing == +1


def test_no_geometry_no_frame():
    empty = Wall(name="empty")
    stamp_frames(_project(empty))
    assert empty._frame is None


# ---------------------------------------------------------------------------
# Pass mechanics
# ---------------------------------------------------------------------------


def test_stamp_is_idempotent():
    south, north = _south_north()
    proj = _project(south, north)
    stamp_frames(proj)
    first = south._frame
    stamp_frames(proj)
    assert south._frame == first


def test_local_to_world_is_the_south_wall_view_on_every_wall():
    """THE mental model: stand where the camera stands and look at the south
    facade — that view is the local frame of every wall.

    Origin on the OUTER face, ``inset`` measuring INTO the element. So one
    value means the same physical depth on every facade, which is what lets
    a window be authored once and placed anywhere."""
    south, north = _south_north()   # south y[-4500,-4200], north y[4200,4500]
    stamp_frames(_project(south, north))

    # inset=0 -> flush with the OUTSIDE of each wall.
    assert south._frame.local_to_world(0.0, 0.0, 0.0)[1] == pytest.approx(-4500.0)
    assert north._frame.local_to_world(0.0, 0.0, 0.0)[1] == pytest.approx(4500.0)

    # inset = thickness -> flush with the INSIDE.
    for wall, inner in ((south, -4200.0), (north, 4200.0)):
        f = wall._frame
        assert f.local_to_world(0.0, f.thickness, 0.0)[1] == pytest.approx(inner)

    # Negative -> proud of the facade, in opposite world directions.
    assert south._frame.local_to_world(0.0, -500.0, 0.0)[1] < -4500.0
    assert north._frame.local_to_world(0.0, -500.0, 0.0)[1] > 4500.0


def test_outward_normal_is_local_minus_y_on_every_wall():
    """The corollary an author has to hold: outward is local ``(0,-1,0)``,
    so all authored geometry sits in the positive octant."""
    south, north = _south_north()
    stamp_frames(_project(south, north))
    for wall in (south, north):
        f = wall._frame
        at_face = f.local_to_world(0.0, 0.0, 0.0)
        deeper = f.local_to_world(0.0, 100.0, 0.0)
        outward_world = tuple(a - b for a, b in zip(at_face, deeper))
        # moving to -y (out of the element) is moving toward the weather
        assert abs(outward_world[1]) == pytest.approx(100.0)
        expected = -1 if wall is south else 1      # south faces -Y, north +Y
        assert (outward_world[1] > 0) == (expected > 0)


def test_frame_follows_units_through_normalize():
    """A frame stamped in mm must NOT survive normalize verbatim — the pass
    re-runs at the end of normalization so the stamp is in meters."""
    from lite_step.compiler.executor import normalize_project_to_meters

    south, north = _south_north()
    proj = _project(south, north)
    stamp_frames(proj)
    assert south._frame.extent_along == pytest.approx(12000.0)     # mm
    normalized = normalize_project_to_meters(proj)
    elem = normalized.storeys[0].elements[0]
    assert elem._frame is not None
    assert elem._frame.extent_along == pytest.approx(12.0)         # meters
    assert elem._frame.thickness == pytest.approx(0.3)


def test_frame_is_a_frozen_dataclass():
    south, _ = _south_north()
    stamp_frames(_project(south))
    assert isinstance(south._frame, ElementFrame)
    with pytest.raises(Exception):
        south._frame.facing = -south._frame.facing


# ---------------------------------------------------------------------------
# opening_axes — the single run-axis derivation (WS2)
# ---------------------------------------------------------------------------


def test_opening_axes_box_picks_the_longer_horizontal_span():
    """X-running body -> along +X, thickness walked along +Y (the POSITIVE
    perpendicular, which is what the generator adds half a thickness of)."""
    from lite_step.compiler.frames import opening_axes

    body = Box(start=Point(x=-6000, y=-4500, z=0), end=Point(x=6000, y=-4200, z=2700))
    along, thickness_dir = opening_axes(body)
    assert along == pytest.approx((1.0, 0.0, 0.0))
    assert thickness_dir == pytest.approx((0.0, 1.0, 0.0))


def test_opening_axes_y_running_box_walks_thickness_along_positive_x():
    """PINNED: for a Y-running body the thickness direction is +X, NOT the
    right-handed ``up x along`` (-X). The generator walks the positive
    perpendicular; returning the cross product here would silently place
    every opening on the wrong side of a Y-running wall."""
    from lite_step.compiler.frames import opening_axes

    body = Box(start=Point(x=5700, y=-4000, z=0), end=Point(x=6000, y=4000, z=2700))
    along, thickness_dir = opening_axes(body)
    assert along == pytest.approx((0.0, 1.0, 0.0))
    assert thickness_dir == pytest.approx((1.0, 0.0, 0.0))


def test_opening_axes_ties_go_to_x():
    """A square body has no longer span — the tie-break is +X, matching the
    convention every consumer inherited from the box-mode rule."""
    from lite_step.compiler.frames import opening_axes

    body = Box(start=Point(x=0, y=0, z=0), end=Point(x=3000, y=3000, z=2700))
    along, _ = opening_axes(body)
    assert along == pytest.approx((1.0, 0.0, 0.0))


def test_opening_axes_snapping_is_applied_before_the_comparison():
    """The ifcopenshell backend snapped spans before comparing; the streaming
    copy did not. A body whose spans differ only below the snap threshold must
    resolve the SAME way for both — which is only true if the snapper runs
    before the comparison."""
    from lite_step.compiler.frames import opening_axes
    from lite_step.ifc.entity_cache import snap_to_precision

    # y is a hair longer than x, below snapping precision.
    body = Box(start=Point(x=0, y=0, z=0), end=Point(x=3000, y=3000, z=2700))
    body.end.__dict__["y"] = 3000.0000001
    unsnapped = opening_axes(body)
    snapped = opening_axes(body, snap=snap_to_precision)
    assert snapped[0] == pytest.approx((1.0, 0.0, 0.0)), (
        "snapped spans tie -> +X")
    assert unsnapped is not None


def test_opening_axes_contour_returns_the_plane_normal():
    """A contour body's thickness runs along the AUTHORED plane normal (the
    contour is one wall FACE), not along ``up x along``."""
    from lite_step.compiler.frames import opening_axes

    body = Extrude(
        contour=[
            Point(x=-4000, y=0, z=0),
            Point(x=4000, y=0, z=0),
            Point(x=4000, y=0, z=2500),
            Point(x=-4000, y=0, z=2500),
        ],
        thickness=300,
    )
    along, normal = opening_axes(body)
    assert along == pytest.approx((1.0, 0.0, 0.0))
    assert abs(normal[1]) == pytest.approx(1.0)   # horizontal plane normal
    assert normal[2] == pytest.approx(0.0)


def test_opening_axes_rejects_a_non_vertical_contour():
    """A horizontal (floor-like) contour has no horizontal run axis — the
    caller turns this None into a loud compile error."""
    from lite_step.compiler.frames import opening_axes

    body = Extrude(
        contour=[
            Point(x=0, y=0, z=0),
            Point(x=4000, y=0, z=0),
            Point(x=4000, y=3000, z=0),
            Point(x=0, y=3000, z=0),
        ],
        thickness=200,
    )
    assert opening_axes(body) is None


