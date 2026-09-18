"""Placement engine tests (DSL v1.5, WS1 PR-F): Transform + Anchor consumption.

Every geometry test compiles a Project through the REAL pipeline
(validate -> normalize_project_to_meters -> generate_ifc) and asserts
MEASURED world-space geometry from tessellation (ifcopenshell.geom,
USE_WORLD_COORDS) — placements must move actual vertices, not just decorate
the model.

Pinned conventions proven here (see ``lite_step/compiler/placement.py``):

* Transform: ``world = T(origin) @ R_1 @ R_2 @ ... @ local`` — rotations
  about the element's LOCAL ORIGIN, composed left-to-right in list order
  (intrinsic sequence; same order as the legacy box ``rotations=``).
* Anchor: host frame from the host's authoring AABB per host type
  (Wall: baseline/exterior-face; Beam: centerline/top; Slab: top surface),
  offsets along the unrotated frame, anchored-frame rotations, recursive
  host chains, cycle ERRORs.
* Application point: IFC local placement composition — the whole product
  family (openings, fills, aggregated children) moves rigidly, without
  regressing the PR-E / window-children chaining.
"""

from __future__ import annotations

import math

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
np = pytest.importorskip("numpy")
geom = pytest.importorskip("ifcopenshell.geom")

import ifcopenshell.util.placement as plc

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.compiler.placement import resolve_placement_matrices
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Anchor,
    Beam,
    Project,
    Element,
    Material,
    Point,
    Sweep,
    Site,
    Slab,
    Box,
    Extrude,
    Transform,
    Wall,
    Window,
)
from lite_step.strict import disable_strict_int_mm, enable_strict_int_mm


# ---------------------------------------------------------------------------
# Helpers (same harness as test_primitives_ifc.py)
# ---------------------------------------------------------------------------


TOL = 2e-3  # writer rounds to 0.1 mm; direction cosines to 4 decimals


def compile_building(proj: Project):
    """validate -> normalize -> generate; returns the parsed ifcopenshell file."""
    report = validate_project_report(proj)
    assert report.errors == [], report.errors
    normalized = normalize_project_to_meters(proj)
    result = generate_ifc(normalized)
    assert result.success, result.error
    return ifcopenshell.file.from_string(result.ifc_content), result


def _own_leaf(nm):
    """The element's OWN leaf from a canonical name (DSL v2.1, leaf-first).

    Canonical is ``type:leaf[:ancestor:pairs...]`` so the element's own leaf is
    segment[1]. Lets these geometry tests keep looking a product up by the leaf
    they authored while the emitted Name is the full canonical path.
    """
    if not nm:
        return None
    segs = nm.split(":")
    return segs[1] if len(segs) >= 2 else segs[0]


def product_by_name(f, name: str):
    matches = [p for p in f.by_type("IfcProduct")
               if p.Name == name or _own_leaf(p.Name) == name]
    assert matches, f"no IfcProduct named {name!r}"
    return matches[0]


_SETTINGS = geom.settings()
_SETTINGS.set(_SETTINGS.USE_WORLD_COORDS, True)

# See test_primitives_ifc.py: window children legitimately live inside the
# opening; disable ancestor-opening subtraction when measuring their OWN
# geometry (the-adjacent quirk).
_SETTINGS_OWN = geom.settings()
_SETTINGS_OWN.set(_SETTINGS_OWN.USE_WORLD_COORDS, True)
_SETTINGS_OWN.set(_SETTINGS_OWN.DISABLE_OPENING_SUBTRACTIONS, True)


def tessellate(product, settings=None):
    """(vertices Nx3, |volume|) of a product's world-space triangulation."""
    shape = geom.create_shape(settings or _SETTINGS, product)
    vs = np.array(shape.geometry.verts).reshape(-1, 3)
    fs = np.array(shape.geometry.faces).reshape(-1, 3)
    vol = 0.0
    for tri in fs:
        a, b, c = vs[tri[0]], vs[tri[1]], vs[tri[2]]
        vol += float(np.dot(a, np.cross(b, c))) / 6.0
    return vs, abs(vol)


def assert_aabb(vs, expect_min, expect_max, tol=TOL):
    got_min = vs.min(axis=0)
    got_max = vs.max(axis=0)
    assert np.allclose(got_min, expect_min, atol=tol), (got_min, expect_min)
    assert np.allclose(got_max, expect_max, atol=tol), (got_max, expect_max)


def measured_aabb(proj: Project, name: str):
    f, _ = compile_building(proj)
    vs, _vol = tessellate(product_by_name(f, name))
    return vs


def _building(*elements) -> Project:
    b = Project(name="placement-test")
    for e in elements:
        b.add(e)
    return b


def _wall_south() -> Wall:
    """The spec's canonical south wall: 12 m along X, 300 mm thick,
    exterior at y = -4500 (farther from the origin)."""
    wall = Wall(name="wall_south")
    wall.add(Box(start=Point(x=-6000, y=-4500, z=0),
                   end=Point(x=6000, y=-4200, z=2700)))
    return wall


def _shelf(**kwargs) -> "BimElement":
    """Anchored payload: a 400 x 300 x 600 box authored local to the
    placement origin."""
    return Box(name="shelf", start=Point(x=0, y=0, z=0),
                 end=Point(x=400, y=300, z=600), **kwargs)


# ---------------------------------------------------------------------------
# Transform — translation
# ---------------------------------------------------------------------------


class TestTransformTranslation:
    def test_translation_moves_measured_geometry(self):
        b = _building(Box(
            name="box", start=Point(x=0, y=0, z=0),
            end=Point(x=1000, y=2000, z=500),
            placement=Transform(origin=Point(x=3000, y=2000, z=1000)),
        ))
        vs = measured_aabb(b, "box")
        assert_aabb(vs, (3.0, 2.0, 1.0), (4.0, 4.0, 1.5))

    def test_translation_equals_world_authoring(self):
        """A placed local box and a world-authored box are the same geometry."""
        placed = _building(Box(
            name="box", start=Point(x=0, y=0, z=0),
            end=Point(x=1000, y=2000, z=500),
            placement=Transform(origin=Point(x=3000, y=2000, z=1000)),
        ))
        authored = _building(Box(
            name="box", start=Point(x=3000, y=2000, z=1000),
            end=Point(x=4000, y=4000, z=1500),
        ))
        vs_placed = measured_aabb(placed, "box")
        vs_authored = measured_aabb(authored, "box")
        assert np.allclose(vs_placed.min(axis=0), vs_authored.min(axis=0), atol=TOL)
        assert np.allclose(vs_placed.max(axis=0), vs_authored.max(axis=0), atol=TOL)

    def test_negative_origin(self):
        b = _building(Box(
            name="box", start=Point(x=0, y=0, z=0),
            end=Point(x=1000, y=1000, z=1000),
            placement=Transform(origin=Point(x=-2000, y=-3000, z=-500)),
        ))
        vs = measured_aabb(b, "box")
        assert_aabb(vs, (-2.0, -3.0, -0.5), (-1.0, -2.0, 0.5))


# ---------------------------------------------------------------------------
# Transform — rotation (measured vertex positions)
# ---------------------------------------------------------------------------


class TestTransformRotation:
    def test_rotate_box_9000_about_z(self):
        """PINNED: rotation is about the element's LOCAL ORIGIN — the box
        authored in +X/+Y swings into the -X/+Y quadrant, right-hand rule."""
        b = _building(Box(
            name="box", start=Point(x=0, y=0, z=0),
            end=Point(x=2000, y=1000, z=500),
            placement=Transform(origin=Point(x=0, y=0, z=0),
                                rotations=[("z", 9000)]),
        ))
        vs = measured_aabb(b, "box")
        assert_aabb(vs, (-1.0, 0.0, 0.0), (0.0, 2.0, 0.5))

    def test_rotation_then_translation(self):
        """world = T(origin) @ R @ local — origin applies AFTER rotation."""
        b = _building(Box(
            name="box", start=Point(x=0, y=0, z=0),
            end=Point(x=2000, y=1000, z=500),
            placement=Transform(origin=Point(x=5000, y=0, z=0),
                                rotations=[("z", 9000)]),
        ))
        vs = measured_aabb(b, "box")
        assert_aabb(vs, (4.0, 0.0, 0.0), (5.0, 2.0, 0.5))

    def test_rotate_about_x(self):
        """Right-hand rule about +X: +Y swings toward +Z."""
        b = _building(Box(
            name="box", start=Point(x=0, y=0, z=0),
            end=Point(x=2000, y=1000, z=500),
            placement=Transform(origin=Point(x=0, y=0, z=0),
                                rotations=[("x", 9000)]),
        ))
        vs = measured_aabb(b, "box")
        assert_aabb(vs, (0.0, -0.5, 0.0), (2.0, 0.0, 1.0))

    def test_rotate_45_about_z(self):
        """Non-axis-aligned check: a 45-degree rotation puts the box's far
        corner at (sqrt(2)/2 * (dx - dy) ..) — measured, not just placed."""
        b = _building(Box(
            name="box", start=Point(x=0, y=0, z=0),
            end=Point(x=1000, y=1000, z=1000),
            placement=Transform(origin=Point(x=0, y=0, z=0),
                                rotations=[("z", 4500)]),
        ))
        vs = measured_aabb(b, "box")
        s = math.sqrt(2.0) / 2.0
        assert_aabb(vs, (-s, 0.0, 0.0), (s, 2 * s, 1.0), tol=5e-3)


class TestTransformRotationSequence:
    """PINNED: the rotation list composes left-to-right in list order —
    R_total = R_1 @ R_2 (intrinsic sequence). [z, x] and [x, z] differ."""

    def test_z_then_x_sequence(self):
        # R = Rz(90) @ Rx(90) maps (x, y, z) -> (z, x, y).
        b = _building(Box(
            name="box", start=Point(x=0, y=0, z=0),
            end=Point(x=2000, y=1000, z=500),
            placement=Transform(origin=Point(x=0, y=0, z=0),
                                rotations=[("z", 9000), ("x", 9000)]),
        ))
        vs = measured_aabb(b, "box")
        assert_aabb(vs, (0.0, 0.0, 0.0), (0.5, 2.0, 1.0))

    def test_x_then_z_sequence_differs(self):
        # R = Rx(90) @ Rz(90) maps (x, y, z) -> (-y, -z, x).
        b = _building(Box(
            name="box", start=Point(x=0, y=0, z=0),
            end=Point(x=2000, y=1000, z=500),
            placement=Transform(origin=Point(x=0, y=0, z=0),
                                rotations=[("x", 9000), ("z", 9000)]),
        ))
        vs = measured_aabb(b, "box")
        assert_aabb(vs, (-1.0, -0.5, 0.0), (0.0, 0.0, 2.0))

    def test_full_turn_is_identity(self):
        b = _building(Box(
            name="box", start=Point(x=1000, y=1000, z=0),
            end=Point(x=2000, y=2000, z=500),
            placement=Transform(origin=Point(x=0, y=0, z=0),
                                rotations=[("z", 36000)]),
        ))
        vs = measured_aabb(b, "box")
        assert_aabb(vs, (1.0, 1.0, 0.0), (2.0, 2.0, 0.5))


# ---------------------------------------------------------------------------
# Anchor — Wall host (baseline frame, all attach_to points, offsets)
# ---------------------------------------------------------------------------


class TestAnchorWall:
    def test_attach_start_with_offsets(self):
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="start",
                             offset_along=500, offset_up=700)))
        vs = measured_aabb(b, "shelf")
        # baseline start (-6, -4.35, 0) + 0.5 along +X + 0.7 up
        assert_aabb(vs, (-5.5, -4.35, 0.7), (-5.1, -4.05, 1.3))

    def test_attach_end(self):
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="end")))
        vs = measured_aabb(b, "shelf")
        assert_aabb(vs, (6.0, -4.35, 0.0), (6.4, -4.05, 0.6))

    def test_attach_center_is_aabb_center(self):
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="center")))
        vs = measured_aabb(b, "shelf")
        assert_aabb(vs, (0.0, -4.35, 1.35), (0.4, -4.05, 1.95))

    def test_attach_face_is_exterior_side(self):
        """PINNED: wall 'face' = the across-side facing AWAY from the
        nearest containment scope — the south wall's exterior at y = -4.5.

        (This fixture is RULE-4 compliant — the building is centered on the
        origin — so the scope-center rule and the legacy "farther from the
        world origin" rule agree here. ``test_frames`` covers the off-origin
        case where they diverge.)"""
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="face")))
        vs = measured_aabb(b, "shelf")
        assert_aabb(vs, (0.0, -4.5, 1.35), (0.4, -4.2, 1.95))

    def test_offset_inset_moves_inward_through_thickness(self):
        """PINNED: ``offset_inset`` is FACING-signed — positive goes INTO the
        host, away from the weather.

        The south wall's exterior is -Y, so a NEGATIVE inset (proud of the
        facade) moves the shelf to -Y — not along the raw ``across`` axis,
        which is +Y here, being ``up x along``."""
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="center",
                             offset_inset=-1000)))
        vs = measured_aabb(b, "shelf")
        assert_aabb(vs, (0.0, -5.35, 1.35), (0.4, -5.05, 1.95))

    def test_offset_inset_means_the_same_depth_on_both_sides(self):
        """THE reason for the rename: one value, one meaning, every wall.

        A north and a south wall have OPPOSITE raw ``across`` signs, so the
        original ``offset_across`` needed a different sign on each to mean
        the same physical thing. ``offset_inset=-500`` now stands both
        shelves 500 mm proud of their own facade."""
        north = Wall(name="wall_north")
        north.add(Box(start=Point(x=-6000, y=4200, z=0),
                      end=Point(x=6000, y=4500, z=2700)))
        def shelf(name, host):
            return Box(name=name, start=Point(x=0, y=0, z=0),
                       end=Point(x=400, y=300, z=600),
                       placement=Anchor(host=host, attach_to="center",
                                        offset_inset=-500))

        b = _building(
            _wall_south(), north,
            shelf("shelf_s", "wall:wall_south"),
            shelf("shelf_n", "wall:wall_north"),
        )
        south_y = float(np.asarray(measured_aabb(b, "shelf_s"))[:, 1].mean())
        north_y = float(np.asarray(measured_aabb(b, "shelf_n"))[:, 1].mean())
        # South wall centre is y=-4.35, north's is y=+4.35: a negative inset
        # stands both 0.5 m proud of their host, in OPPOSITE world directions.
        assert south_y < -4.35, f"south shelf should move -Y, got {south_y}"
        assert north_y > 4.35, f"north shelf should move +Y, got {north_y}"

    def test_wall_running_along_y_rotates_the_anchored_frame(self):
        """The anchored element's local axes align to the host frame:
        local X = along (+Y here), local Y = across (-X here), local Z = up."""
        wall = Wall(name="wall_east")
        wall.add(Box(start=Point(x=2000, y=-3000, z=0),
                       end=Point(x=2300, y=3000, z=2700)))
        b = _building(wall, _shelf(
            placement=Anchor(host="wall:wall_east", attach_to="start")))
        vs = measured_aabb(b, "shelf")
        # start = (2.15, -3, 0); local (lx, ly, lz) -> (2.15 - ly, -3 + lx, lz)
        assert_aabb(vs, (1.85, -3.0, 0.0), (2.15, -2.6, 0.6))

    def test_anchor_to_named_wall_body_child_uses_wall_frame(self):
        """Hosts resolve against container children; the frame rule comes from
        the enclosing Wall container. DSL v2.1: anchoring to the wall
        (host="wall:wall_south", anonymous body) and to a NAMED body child
        (host=its full canonical path — a shorter suffix would collide with the
        wall's tail) yield the same frame."""
        wall_named = Wall(name="wall_south")
        wall_named.add(Box(name="ws_body", start=Point(x=-6000, y=-4500, z=0),
                           end=Point(x=6000, y=-4200, z=2700)))
        b_wall = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="start")))
        b_body = _building(wall_named, _shelf(
            placement=Anchor(host="box:ws_body:wall:wall_south",
                             attach_to="start")))
        vs_wall = measured_aabb(b_wall, "shelf")
        vs_body = measured_aabb(b_body, "shelf")
        assert np.allclose(vs_wall.min(axis=0), vs_body.min(axis=0), atol=TOL)
        assert np.allclose(vs_wall.max(axis=0), vs_body.max(axis=0), atol=TOL)


# ---------------------------------------------------------------------------
# Anchor — Beam + Slab host frames
# ---------------------------------------------------------------------------


class TestAnchorBeamSlab:
    def _beam(self) -> Beam:
        beam = Beam(name="beam_a")
        beam.add(Box(start=Point(x=-2000, y=-150, z=2400),
                       end=Point(x=2000, y=150, z=2700)))
        return beam

    def test_beam_attach_start_is_centerline(self):
        b = _building(self._beam(), Box(
            name="clip", start=Point(x=0, y=0, z=0),
            end=Point(x=200, y=200, z=200),
            placement=Anchor(host="beam:beam_a", attach_to="start", offset_up=100)))
        vs = measured_aabb(b, "clip")
        # centerline start (-2, 0, 2.55) + 0.1 up
        assert_aabb(vs, (-2.0, 0.0, 2.65), (-1.8, 0.2, 2.85))

    def test_beam_attach_face_is_top(self):
        b = _building(self._beam(), Box(
            name="clip", start=Point(x=0, y=0, z=0),
            end=Point(x=200, y=200, z=200),
            placement=Anchor(host="beam:beam_a", attach_to="face")))
        vs = measured_aabb(b, "clip")
        assert_aabb(vs, (0.0, 0.0, 2.7), (0.2, 0.2, 2.9))

    def test_slab_attach_face_is_top_surface(self):
        slab = Slab(name="foundation")
        slab.add(Box(start=Point(x=-5000, y=-4000, z=-300),
                     end=Point(x=5000, y=4000, z=0)))
        b = _building(slab, Box(
            name="post", start=Point(x=0, y=0, z=0),
            end=Point(x=150, y=150, z=2400),
            placement=Anchor(host="slab:foundation", attach_to="face",
                             offset_along=1000, offset_inset=-2000)))
        vs = measured_aabb(b, "post")
        # slab frame: along=+X, across=+Y, up=+Z; face = top surface center.
        # A lone slab centered on the origin has facing=+1, so out=+2000
        # runs along +across (+Y) — same result as the legacy across offset.
        assert_aabb(vs, (1.0, 2.0, 0.0), (1.15, 2.15, 2.4))

    def test_slab_start_is_top_surface_x_line(self):
        slab = Slab(name="foundation")
        slab.add(Box(start=Point(x=-5000, y=-4000, z=-300),
                     end=Point(x=5000, y=4000, z=0)))
        b = _building(slab, Box(
            name="post", start=Point(x=0, y=0, z=0),
            end=Point(x=150, y=150, z=2400),
            placement=Anchor(host="slab:foundation", attach_to="start")))
        vs = measured_aabb(b, "post")
        assert_aabb(vs, (-5.0, 0.0, 0.0), (-4.85, 0.15, 2.4))


# ---------------------------------------------------------------------------
# Anchor — rotations about the anchored frame
# ---------------------------------------------------------------------------


class TestAnchorRotations:
    def test_rotation_about_up_at_attach_point(self):
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="center",
                             rotations=[("z", 9000)])))
        vs = measured_aabb(b, "shelf")
        # about (0, -4.35, 1.35): local (lx, ly) -> (-ly, lx)
        assert_aabb(vs, (-0.3, -4.35, 1.35), (0.0, -3.95, 1.95))

    def test_rotation_axes_follow_the_anchored_frame(self):
        """On a Y-running wall, "z" still means the anchored frame's up
        axis: basis @ Rz — resolved at matrix level for exactness."""
        wall = Wall(name="wall_east")
        wall.add(Box(start=Point(x=2000, y=-3000, z=0),
                       end=Point(x=2300, y=3000, z=2700)))
        b = _building(wall, _shelf(
            placement=Anchor(host="wall:wall_east", attach_to="start",
                             rotations=[("z", 9000)])))
        normalized = normalize_project_to_meters(b)
        shelf = normalized.storeys[0].elements[1]
        m = resolve_placement_matrices(normalized)[id(shelf)]
        # basis: along=+Y, across=-X, up=+Z; then Rz(90) intrinsic:
        # local +X -> across (-X world), local +Y -> -along (-Y world)
        assert np.allclose(m[:3, 0], (-1.0, 0.0, 0.0), atol=1e-9)
        assert np.allclose(m[:3, 1], (0.0, -1.0, 0.0), atol=1e-9)
        assert np.allclose(m[:3, 3], (2.15, -3.0, 0.0), atol=1e-9)


# ---------------------------------------------------------------------------
# Anchor — host resolution (name-based; DSL v1.5)
# ---------------------------------------------------------------------------


class TestAnchorHostResolution:
    def test_host_resolves_by_name(self):
        target = Box(name="legacy_wall", start=Point(x=1000, y=1000, z=0),
                       end=Point(x=3000, y=1300, z=2000))
        b = _building(target, _shelf(
            placement=Anchor(host="box:legacy_wall", attach_to="center")))
        vs = measured_aabb(b, "shelf")
        # bare Solid host: center = AABB center (2, 1.15, 1)
        assert_aabb(vs, (2.0, 1.15, 1.0), (2.4, 1.45, 1.6))

    def test_host_resolves_against_names_not_ids(self):
        """DSL v1.5: host= resolves against name= only — the pre-v1.5
        legacy-id fallback was removed, so an element's auto id never
        satisfies a host reference."""
        named = Box(name="target", start=Point(x=0, y=0, z=0),
                      end=Point(x=1000, y=1000, z=1000))
        b = _building(named, _shelf(
            placement=Anchor(host="box:target", attach_to="center")))
        normalized = normalize_project_to_meters(b)
        shelf = normalized.storeys[0].elements[1]
        m = resolve_placement_matrices(normalized)[id(shelf)]
        assert np.allclose(m[:3, 3], (0.5, 0.5, 0.5), atol=1e-9)

    def test_unresolvable_host_fails_generation_loudly(self):
        """Generation without prior validation still fails LOUD."""
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall_ghost")))
        normalized = normalize_project_to_meters(b)
        result = generate_ifc(normalized)
        assert not result.success
        assert "wall_ghost" in result.error
        assert "does not resolve" in result.error

    def test_unresolvable_host_is_still_a_compile_error(self):
        """PR-D behavior extended: exactly one host-existence error (the
        resolver's duplicate kind="host" failure is filtered)."""
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall_ghost")))
        report = validate_project_report(b)
        errs = [e for e in report.errors if "wall_ghost" in e]
        assert len(errs) == 1

    def test_host_without_geometry_is_compile_error(self):
        stair = Element(ifc_class="IfcStair", name="stair_main")
        b = _building(stair, _shelf(placement=Anchor(host="stair:stair_main")))
        report = validate_project_report(b)
        assert any("no point-bearing geometry" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Anchor — chained placements + cycles
# ---------------------------------------------------------------------------


class TestChainedPlacements:
    def test_anchor_to_transform_placed_wall(self):
        wall = _wall_south()
        wall.placement = Transform(origin=Point(x=1000, y=0, z=0))
        b = _building(wall, _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="start")))
        vs = measured_aabb(b, "shelf")
        # host frame start (-6, -4.35, 0) shifted by the wall's +1 m X
        assert_aabb(vs, (-5.0, -4.35, 0.0), (-4.6, -4.05, 0.6))

    def test_anchor_to_rotated_wall_rotates_the_anchor(self):
        wall = _wall_south()
        wall.placement = Transform(origin=Point(x=0, y=0, z=0),
                                   rotations=[("z", 9000)])
        b = _building(wall, _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="start")))
        vs = measured_aabb(b, "shelf")
        # M_wall = Rz90; authored start (-6, -4.35, 0) -> (4.35, -6, 0);
        # anchored local (lx, ly, lz) -> (-ly, lx, lz) + that
        assert_aabb(vs, (4.05, -6.0, 0.0), (4.35, -5.6, 0.6))

    def test_two_level_anchor_chain(self):
        shelf1 = _shelf(placement=Anchor(host="wall:wall_south", attach_to="start"))
        shelf1.name = "shelf1"
        shelf2 = Box(name="shelf2", start=Point(x=0, y=0, z=0),
                       end=Point(x=100, y=100, z=100),
                       placement=Anchor(host="box:shelf1", attach_to="face"))
        b = _building(_wall_south(), shelf1, shelf2)
        vs = measured_aabb(b, "shelf2")
        # shelf1's world matrix = T(-6, -4.35, 0); its authoring-AABB top
        # face center (fallback rule) = (0.2, 0.15, 0.6)
        assert_aabb(vs, (-5.8, -4.2, 0.6), (-5.7, -4.1, 0.7))

    def test_anchor_cycle_is_compile_error(self):
        a = Box(name="a", start=Point(x=0, y=0, z=0),
                  end=Point(x=1000, y=1000, z=1000),
                  placement=Anchor(host="box:b"))
        bb = Box(name="b", start=Point(x=0, y=0, z=0),
                   end=Point(x=1000, y=1000, z=1000),
                   placement=Anchor(host="box:a"))
        report = validate_project_report(_building(a, bb))
        assert any("cycle" in e.lower() for e in report.errors)

    def test_self_anchor_is_compile_error(self):
        a = Box(name="a", start=Point(x=0, y=0, z=0),
                  end=Point(x=1000, y=1000, z=1000),
                  placement=Anchor(host="box:a"))
        report = validate_project_report(_building(a))
        assert any("cycle" in e.lower() for e in report.errors)

    def test_cycle_fails_generation_loudly(self):
        a = Box(name="a", start=Point(x=0, y=0, z=0),
                  end=Point(x=1000, y=1000, z=1000),
                  placement=Anchor(host="box:b"))
        bb = Box(name="b", start=Point(x=0, y=0, z=0),
                   end=Point(x=1000, y=1000, z=1000),
                   placement=Anchor(host="box:a"))
        normalized = normalize_project_to_meters(_building(a, bb))
        result = generate_ifc(normalized)
        assert not result.success
        assert "cycle" in result.error.lower()


# ---------------------------------------------------------------------------
# Strict int-mm interplay
# ---------------------------------------------------------------------------


@pytest.fixture
def strict_mode():
    enable_strict_int_mm()
    try:
        yield
    finally:
        enable_strict_int_mm()  # post-cutover default stays ON


class TestStrictFlag:
    def test_float_origin_rejected_when_strict(self, strict_mode):
        with pytest.raises(Exception, match="float"):
            Transform(origin=Point(x=1.5, y=0, z=0))

    def test_float_offset_rejected_when_strict(self, strict_mode):
        with pytest.raises(Exception, match="float"):
            Anchor(host="wall:wall_south", offset_along=500.0)

    def test_float_rotation_angle_rejected_when_strict(self, strict_mode):
        with pytest.raises(Exception, match="centidegrees"):
            Transform(origin=Point(x=0, y=0, z=0), rotations=[("z", 4500.0)])

    def test_int_placement_accepted_when_strict(self, strict_mode):
        a = Anchor(host="wall:wall_south", offset_along=500,
                   rotations=[("z", 4500)])
        assert a.offset_along == 500

    def test_normalization_runs_inside_suspend_strict(self, strict_mode):
        """The mm->m placement rebuild is internal math — the strict gate
        must never fire on it (compile end-to-end with the flag ON)."""
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="start")))
        vs = measured_aabb(b, "shelf")
        assert_aabb(vs, (-6.0, -4.35, 0.0), (-5.6, -4.05, 0.6))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestPlacementNormalization:
    def test_transform_origin_divides_by_1000(self):
        b = _building(Box(
            name="box", start=Point(x=0, y=0, z=0),
            end=Point(x=1000, y=1000, z=1000),
            placement=Transform(origin=Point(x=3000, y=-2000, z=500),
                                rotations=[("z", 4500)]),
        ))
        n = normalize_project_to_meters(b)
        p = n.storeys[0].elements[0].placement
        assert (p.origin.x, p.origin.y, p.origin.z) == (3.0, -2.0, 0.5)
        # rotations are centidegrees — NOT normalized
        assert p.rotations == [("z", 4500)]

    def test_anchor_offsets_divide_by_1000(self):
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south", offset_along=1500,
                             offset_inset=250, offset_up=700,
                             rotations=[("x", 9000)])))
        n = normalize_project_to_meters(b)
        p = n.storeys[0].elements[1].placement
        assert p.offset_along == 1.5
        assert p.offset_inset == 0.25
        assert p.offset_up == 0.7
        assert p.host == "wall:wall_south"
        assert p.attach_to == "center"
        assert p.rotations == [("x", 9000)]

    def test_original_building_is_not_mutated(self):
        placement = Transform(origin=Point(x=3000, y=2000, z=1000))
        b = _building(Box(
            name="box", start=Point(x=0, y=0, z=0),
            end=Point(x=1000, y=1000, z=1000), placement=placement))
        normalize_project_to_meters(b)
        assert b.storeys[0].elements[0].placement.origin.x == 3000

    def test_end_to_end_compile_main_with_placement(self, tmp_path):
        """Full pipeline: compile_main -> output.ifc -> measured geometry."""
        from lite_step.compiler.executor import compile_main

        src = tmp_path / "model.py"
        src.write_text("# placement engine e2e fixture\n", encoding="utf-8")
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="face",
                             offset_along=2000, offset_up=-500)))
        out = compile_main(result=b, source_path=str(src))
        f = ifcopenshell.open(str(out))
        vs, _ = tessellate(product_by_name(f, "shelf"))
        # face (0, -4.5, 1.35) + 2.0 along + (-0.5) up
        assert_aabb(vs, (2.0, -4.5, 0.85), (2.4, -4.2, 1.45))


# ---------------------------------------------------------------------------
# Validation — construction + report
# ---------------------------------------------------------------------------


class TestPlacementValidation:
    def test_unknown_rotation_axis_fails_at_construction(self):
        with pytest.raises(Exception, match="unknown rotation axis"):
            Transform(origin=Point(x=0, y=0, z=0), rotations=[("w", 9000)])

    def test_malformed_rotation_tuple_fails_at_construction(self):
        with pytest.raises(Exception, match="tuple"):
            Transform(origin=Point(x=0, y=0, z=0), rotations=[("z",)])

    def test_rotations_must_be_a_list(self):
        with pytest.raises(Exception, match="list"):
            Transform(origin=Point(x=0, y=0, z=0), rotations="z9000")

    def test_anchor_rotation_axis_validated_too(self):
        with pytest.raises(Exception, match="unknown rotation axis"):
            Anchor(host="wall:wall_south", rotations=[("q", 100)])

    def test_off_enum_attach_to_fails_at_construction(self):
        with pytest.raises(Exception):
            Anchor(host="wall:wall_south", attach_to="top")

    def test_nested_placement_is_compile_error(self):
        wall = Wall(name="wall_x")
        wall.add(Box(name="body_x", start=Point(x=0, y=0, z=0),
                       end=Point(x=5000, y=300, z=2700),
                       placement=Transform(origin=Point(x=0, y=0, z=0))))
        report = validate_project_report(_building(wall))
        assert any("nested element" in e for e in report.errors)

    def test_site_placement_is_compile_error(self):
        site = Site(name="terrain",
                    placement=Transform(origin=Point(x=0, y=0, z=0)))
        site.add(Box(start=Point(x=-10000, y=-10000, z=-500),
                       end=Point(x=10000, y=10000, z=0), type="site"))
        report = validate_project_report(_building(site))
        assert any("Site container" in e for e in report.errors)

    def test_boolean_operand_placement_is_compile_error(self):
        void = Box(start=Point(x=0, y=0, z=0),
                     end=Point(x=500, y=400, z=2000),
                     placement=Transform(origin=Point(x=1000, y=0, z=0)))
        body = Box(name="body", start=Point(x=-3000, y=0, z=0),
                     end=Point(x=3000, y=300, z=2700)).difference(void)
        report = validate_project_report(_building(body))
        assert any("boolean operand" in e for e in report.errors)

    def test_clean_placed_building_validates(self):
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south", attach_to="face")))
        report = validate_project_report(b)
        assert report.errors == []

    def test_placement_matrices_only_for_placed_top_elements(self):
        b = _building(_wall_south(), _shelf(
            placement=Anchor(host="wall:wall_south")))
        n = normalize_project_to_meters(b)
        matrices = resolve_placement_matrices(n)
        assert len(matrices) == 1
        assert id(n.storeys[0].elements[1]) in matrices


# ---------------------------------------------------------------------------
# Wall placement + window children (#374-adjacent chaining guard)
# ---------------------------------------------------------------------------


class TestPlacedWallWithWindow:
    def _placed_wall_with_window(self) -> Project:
        wall = _wall_south()
        win = Window(width=1200, height=1400, name="win_s0")
        win.add(Extrude(name="pane_s0",
                      contour=[Point(x=50, y=100, z=50),
                               Point(x=1150, y=100, z=50),
                               Point(x=1150, y=100, z=1350),
                               Point(x=50, y=100, z=1350)],
                      thickness=6, color="glass"))
        fw_c = 22
        win.add(Sweep(name="frame_s0",
                        material=Material(key="Timber_C24", profile_mm=(95, 45)),
                        path=[Point(x=fw_c, y=100, z=fw_c),
                              Point(x=1200 - fw_c, y=100, z=fw_c),
                              Point(x=1200 - fw_c, y=100, z=1400 - fw_c),
                              Point(x=fw_c, y=100, z=1400 - fw_c),
                              Point(x=fw_c, y=100, z=fw_c)]))
        wall.anchor(win, along=1500, up=900)
        wall.placement = Transform(origin=Point(x=0, y=0, z=0),
                                   rotations=[("z", 9000)])
        return _building(wall)

    def test_wall_body_rotates(self):
        f, _ = compile_building(self._placed_wall_with_window())
        vs, _ = tessellate(product_by_name(f, "wall_south"))
        # authored (x, y) -> (-y, x): x in [4.2, 4.5], y in [-6, 6]
        assert_aabb(vs, (4.2, -6.0, 0.0), (4.5, 6.0, 2.7))

    def test_window_fill_placement_follows_the_wall(self):
        f, _ = compile_building(self._placed_wall_with_window())
        win = product_by_name(f, "win_s0")
        m = plc.get_local_placement(win.ObjectPlacement)
        # v21: the fill sits FLUSH with the wall's outer face (y = -4.5), not
        # on the thickness midline. Exterior is -Y here (the storey scope is
        # empty — the wall carries a placement — so facing falls back to the
        # world-origin rule), so local +x still runs +X and along= does not
        # reverse. Authored fill centre (-3.9, -4.5, 0.9) -> Rz90.
        assert np.allclose(m[:3, 3], (4.5, -3.9, 0.9), atol=TOL)

    def test_frame_member_chains_and_rotates_rigidly(self):
        """The PR-E identity-relative chained frame member must follow the
        rotated window with its measured volume intact (no regression)."""
        f, _ = compile_building(self._placed_wall_with_window())
        frame = product_by_name(f, "frame_s0")
        # still chained (identity-relative) to the window's placement
        assert frame.ObjectPlacement.PlacementRelTo is not None
        vs, vol = tessellate(frame, settings=_SETTINGS_OWN)
        # picture-frame volume: section 95 x 45; loop 1156 x 1356 centerline
        expected_vol = 0.095 * 0.045 * 2 * (1.156 + 1.356)
        assert vol == pytest.approx(expected_vol, rel=0.02)
        # authored world frame AABB (measured, unplaced reference): the
        # member is authored at opening-local y=100 mm, which v21 measures
        # from the OUTER face (-4.5) instead of the midline (-4.35), so the
        # 95 mm section straddles y=-4.4 rather than y=-4.25 —
        # x [-4.5005, -3.2995], y [-4.4475, -4.3525], z [0.8995, 2.3005];
        # the placement maps (x, y) -> (-y, x) rigidly.
        assert_aabb(vs,
                    (4.3525, -4.5005, 0.8995),
                    (4.4475, -3.2995, 2.3005), tol=5e-3)

    def test_glass_pane_rotates_with_the_wall(self):
        f, _ = compile_building(self._placed_wall_with_window())
        pane = product_by_name(f, "pane_s0")
        vs, _ = tessellate(pane, settings=_SETTINGS_OWN)
        # authored world x [-4.45, -3.35], y ~ [-4.25, -4.244], z [0.95, 2.25]
        # (pane at y=100mm through wall from corner -4.35, thickness 6 along
        # the contour normal) -> rotated: x = -y, y = x
        got_min, got_max = vs.min(axis=0), vs.max(axis=0)
        assert got_min[1] == pytest.approx(-4.45, abs=TOL)
        assert got_max[1] == pytest.approx(-3.35, abs=TOL)
        assert got_min[2] == pytest.approx(0.95, abs=TOL)
        assert got_max[2] == pytest.approx(2.25, abs=TOL)
        # pane sits inside the rotated wall slice x in [4.2, 4.5]
        assert got_min[0] > 4.2 - TOL
        assert got_max[0] < 4.5 + TOL

    def test_unplaced_window_wall_still_has_stable_geometry(self):
        """Control: the same wall WITHOUT placement keeps its authored
        coordinates — the hook composes only on placed elements."""
        b = self._placed_wall_with_window()
        b.storeys[0].elements[0].placement = None
        f, _ = compile_building(b)
        vs, _ = tessellate(product_by_name(f, "wall_south"))
        assert_aabb(vs, (-6.0, -4.5, 0.0), (6.0, -4.2, 2.7))
