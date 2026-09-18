"""Geometry-level IFC tests for the DSL v1.5 primitives (WS1 PR-E).

Every test compiles a Project through the REAL pipeline
(validate -> normalize_project_to_meters -> generate_ifc) and asserts on
the emitted IFC: entity types, attributes, and MEASURED dimensions from
tessellated geometry (ifcopenshell.geom) — including the two canonical
spec examples:

* the arched doorway (Revolve as a merged void: door_void.union(arch_turn);
  wall_body.difference(door_void)) — proving the pinned Revolve conventions
  (right-hand winding, z-cross-axis start plane -> the 18000-centidegree
  sweep about a horizontal axis is the UPPER half);
* the stirruped beam (stirrup_path + Bar grade auto-registry).

Plus: mitered closed-loop frame (4 segments, exact volume — no
double-solid corner overlap), cross-primitive booleans, Bar quantities,
and streaming-gate routing.
"""

from __future__ import annotations

import math

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
np = pytest.importorskip("numpy")
geom = pytest.importorskip("ifcopenshell.geom")

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.helpers import stirrup_path
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Bar,
    Beam,
    Project,
    Element,
    Material,
    Point,
    Point2D,
    Sweep,
    Pipe,
    Box,
    Extrude,
    Revolve,
    Wall,
    Window,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compile_building(proj: Project):
    """validate -> normalize -> generate; returns the parsed ifcopenshell file."""
    report = validate_project_report(proj)
    assert report.errors == [], report.errors
    normalized = normalize_project_to_meters(proj)
    result = generate_ifc(normalized)
    assert result.success, result.error
    f = ifcopenshell.file.from_string(result.ifc_content)
    return f, result


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

# Member-level measurement settings: ifcopenshell SUBTRACTS the wall's
# opening void from products aggregated under the filling IfcWindow/IfcDoor
# — but window parts (frame members, glass panes) legitimately LIVE inside
# the opening, so with subtraction on, create_shape erases them down to
# whatever pokes outside the reveal (the mitered frame measured its 0.5 mm
# overhang ring, ~1/87 of its true volume). Disable subtraction when
# measuring a window child's OWN geometry.
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


def tessellate_own(product):
    """tessellate() without ancestor-opening subtraction — for window/door
    children whose geometry lives inside the opening (see _SETTINGS_OWN)."""
    return tessellate(product, settings=_SETTINGS_OWN)


ARCH_SECTION = [Point2D(x=0, y=0), Point2D(x=500, y=0),
                Point2D(x=500, y=360), Point2D(x=0, y=360)]
ARCH_AXIS = [Point(x=0, y=4170, z=2000), Point(x=0, y=4530, z=2000)]


# ---------------------------------------------------------------------------
# Pipe
# ---------------------------------------------------------------------------


class TestRodIfc:
    def test_straight_rod_swept_disk(self):
        proj = Project(name="rod")
        proj.add(Pipe(path=[Point(x=0, y=0, z=0), Point(x=2000, y=0, z=0)],
                     radius=25, name="rod_1"))
        f, _ = compile_building(proj)
        solids = f.by_type("IfcSweptDiskSolid")
        assert len(solids) == 1
        assert solids[0].Radius == pytest.approx(0.025)
        assert solids[0].Directrix.is_a("IfcPolyline")
        # Measured: cylinder r=25mm, l=2m.
        vs, vol = tessellate(product_by_name(f, "rod_1"))
        assert vol == pytest.approx(math.pi * 0.025 ** 2 * 2.0, rel=0.02)
        assert vs[:, 0].min() == pytest.approx(0.0, abs=1e-6)
        assert vs[:, 0].max() == pytest.approx(2.0, abs=1e-6)

    def test_filleted_rod_true_arc(self):
        proj = Project(name="rod fillet")
        proj.add(Pipe(path=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0),
                           Point(x=1000, y=1000, z=0)],
                     radius=25, fillet_radius=200, name="rail_1"))
        f, _ = compile_building(proj)
        # True arc: IfcIndexedPolyCurve directrix with an IfcArcIndex segment.
        curves = f.by_type("IfcIndexedPolyCurve")
        assert len(curves) == 1
        segment_types = [s.is_a() for s in curves[0].Segments]
        assert "IfcArcIndex" in segment_types
        assert "IfcLineIndex" in segment_types
        # Measured volume: true centerline length = 2000 - 2t + r*theta.
        length = 2.0 - 2 * 0.2 + 0.2 * math.pi / 2
        vs, vol = tessellate(product_by_name(f, "rail_1"))
        assert vol == pytest.approx(math.pi * 0.025 ** 2 * length, rel=0.05)

    def test_sharp_rod_polyline_directrix(self):
        proj = Project(name="rod sharp")
        proj.add(Pipe(path=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0),
                           Point(x=1000, y=1000, z=0)],
                     radius=25, name="rod_sharp"))
        f, _ = compile_building(proj)
        assert len(f.by_type("IfcIndexedPolyCurve")) == 0
        assert f.by_type("IfcSweptDiskSolid")[0].Directrix.is_a("IfcPolyline")


# ---------------------------------------------------------------------------
# Revolve — pinned conventions, proven by measurement
# ---------------------------------------------------------------------------


class TestTurnIfc:
    def test_revolve_never_emits_an_ifcrevolvedareasolid(self):
        """web-ifc draws no IfcRevolvedAreaSolid, so we must not emit one.

        This is the renderability contract, not a style preference: measured
        against web-ifc 0.0.77, EVERY IfcRevolvedAreaSolid returns zero
        vertices — silently, with no error — so an element emitted that way
        is absent from the app while compiling, validating and tessellating
        perfectly here..
        """
        proj = Project(name="turn")
        proj.add(Revolve(profile=ARCH_SECTION, path=ARCH_AXIS, angle=18000,
                      name="arch_1"))
        f, _ = compile_building(proj)
        assert f.by_type("IfcRevolvedAreaSolid") == []
        # …and what replaced it is still a SOLID, so a Revolve stays a legal
        # IfcBooleanOperand — the reason this is not a tessellation.
        item = product_by_name(f, "arch_1").Representation.Representations[0].Items[0]
        assert item.is_a("IfcSolidModel"), item.is_a()
        # A rectangle profile takes the extrusion path.
        assert item.is_a("IfcExtrudedAreaSolid")
        # RADIAN plane-angle unit is still on the assignment (kept so a file
        # whose only change should be the solid does not also lose a unit).
        units = f.by_type("IfcUnitAssignment")[0].Units
        assert any(getattr(u, "UnitType", None) == "PLANEANGLEUNIT"
                   and getattr(u, "Name", None) == "RADIAN" for u in units)

    # ---- the three emission paths --------------------------------
    #
    # A Revolve is spelled as one of three IFC solids depending on its
    # profile and angle. All three are IfcSolidModel, so all three stay
    # legal boolean operands; two of them are EXACT.

    @staticmethod
    def _one_revolve(profile, angle, axis_len=1000):
        proj = Project(name="paths")
        proj.add(Revolve(profile=profile, angle=angle, name="r1",
                         path=[Point(x=0, y=0, z=0),
                               Point(x=0, y=0, z=axis_len)]))
        f, _ = compile_building(proj)
        return f, product_by_name(f, "r1").Representation.Representations[0].Items[0]

    def test_full_turn_touching_the_axis_is_an_exact_circle(self):
        """No faceting at all — the arc IS a circle, so emit one.

        This is the door rose's shape (systems/details), i.e. the only
        Revolve in the committed corpus: a rectangle from the axis out,
        swept 360 degrees. It is a cylinder, and saying so costs one
        entity and zero approximation.
        """
        f, item = self._one_revolve(
            [Point2D(x=0, y=0), Point2D(x=2000, y=0),
             Point2D(x=2000, y=500), Point2D(x=0, y=500)], 36000)
        assert item.is_a("IfcExtrudedAreaSolid")
        assert item.SweptArea.is_a("IfcCircleProfileDef")
        assert item.SweptArea.Radius == pytest.approx(2.0)
        assert item.Depth == pytest.approx(0.5)
        _vs, vol = tessellate(product_by_name(f, "r1"))
        assert vol == pytest.approx(math.pi * 2.0 ** 2 * 0.5, rel=0.005)

    def test_full_turn_clear_of_the_axis_is_an_exact_hollow_circle(self):
        """An annulus is exact too — IfcCircleHollowProfileDef carries it."""
        f, item = self._one_revolve(
            [Point2D(x=3000, y=0), Point2D(x=5000, y=0),
             Point2D(x=5000, y=400), Point2D(x=3000, y=400)], 36000)
        assert item.SweptArea.is_a("IfcCircleHollowProfileDef")
        assert item.SweptArea.Radius == pytest.approx(5.0)
        # inner radius = Radius - WallThickness
        assert item.SweptArea.Radius - item.SweptArea.WallThickness \
            == pytest.approx(3.0)
        _vs, vol = tessellate(product_by_name(f, "r1"))
        assert vol == pytest.approx(math.pi * (5.0 ** 2 - 3.0 ** 2) * 0.4,
                                    rel=0.005)

    def test_a_sector_facets_within_the_stated_tolerance(self):
        """A partial turn is the ONLY approximating path — and it is bounded.

        Measured on the emitted profile rather than on a volume: every
        vertex sits exactly on the true radius, so the whole error is the
        sagitta at the chord midpoints, which must not exceed
        ``_REVOLVE_CHORD_TOL_M``.
        """
        from lite_step.ifc.generator import _REVOLVE_CHORD_TOL_M
        r_outer = 30.0
        # STEP writes coordinates at finite precision (~1e-4 m at this
        # magnitude), so both the arc filter and the bound carry that slack
        # explicitly rather than pretending the round-trip is exact.
        write_eps = 1e-3
        _f, item = self._one_revolve(
            [Point2D(x=12000, y=0), Point2D(x=30000, y=0),
             Point2D(x=30000, y=2000), Point2D(x=12000, y=2000)], 6000)
        assert item.SweptArea.is_a("IfcArbitraryClosedProfileDef")
        pts = [p.Coordinates for p in item.SweptArea.OuterCurve.Points][:-1]
        # …[:-1] drops the explicit closing point, which repeats the arc's
        # START. Left in, it pairs with the arc's END and reports the whole
        # 60-degree span as one chord.
        outer = [(x, y) for x, y in pts
                 if abs(math.hypot(x, y) - r_outer) < write_eps]
        assert len(outer) >= 3, "expected a faceted outer arc"
        worst = max(r_outer - math.hypot((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                    for a, b in zip(outer, outer[1:]))
        assert worst <= _REVOLVE_CHORD_TOL_M + write_eps, f"sagitta {worst} m"
        # …and it is not trivially satisfied by refusing to facet: a 5 mm
        # budget at r=30 over 60 degrees genuinely needs ~29 chords.
        assert worst > _REVOLVE_CHORD_TOL_M / 2, (
            f"sagitta {worst} m is suspiciously fine — is the arc faceted "
            f"far beyond the budget, at needless byte cost?")

    def test_a_non_rectangle_profile_becomes_a_solid_brep(self):
        """An L-shaped profile cannot be an extrusion — it becomes a brep.

        Positive volume is the assertion that matters: the shell's winding
        is decided by a measured signed volume, so a negative one here
        would mean every face points inward and the solid renders
        inside-out.
        """
        f, item = self._one_revolve(
            [Point2D(x=2000, y=0), Point2D(x=4000, y=0),
             Point2D(x=4000, y=300), Point2D(x=2600, y=300),
             Point2D(x=2600, y=900), Point2D(x=2000, y=900)], 18000)
        assert item.is_a("IfcFacetedBrep")
        assert item.is_a("IfcSolidModel")     # still boolean-legal
        _vs, vol = tessellate(product_by_name(f, "r1"))
        # Pappus: V = theta * r_centroid * area, area 0.96, r_c 2.7375.
        assert vol > 0
        assert vol == pytest.approx(math.pi * 2.7375 * 0.96, rel=0.01)

    def test_semicircle_about_horizontal_axis_is_upper_half(self):
        """The MEASURED pinned-convention proof (spec arched doorway).

        Axis horizontal (+Y) at z=2000, radius 500, angle=18000: the
        revolution must produce the UPPER half-cylinder — z in
        [2000, 2500], symmetric x in [-500, 500], y = the axis extent.
        """
        proj = Project(name="arch")
        proj.add(Revolve(profile=ARCH_SECTION, path=ARCH_AXIS, angle=18000,
                      name="arch_1"))
        f, _ = compile_building(proj)
        vs, vol = tessellate(product_by_name(f, "arch_1"))
        assert vs[:, 2].min() == pytest.approx(2.0, abs=0.005)   # never below spring
        assert vs[:, 2].max() == pytest.approx(2.5, abs=0.005)   # apex = spring + r
        assert vs[:, 0].min() == pytest.approx(-0.5, abs=0.005)
        assert vs[:, 0].max() == pytest.approx(0.5, abs=0.005)
        assert vs[:, 1].min() == pytest.approx(4.17, abs=0.005)
        assert vs[:, 1].max() == pytest.approx(4.53, abs=0.005)
        assert vol == pytest.approx(math.pi * 0.5 ** 2 * 0.36 / 2, rel=0.02)

    def test_axis_direction_flip_still_sweeps_up(self):
        """Reversing the axis points must still give the upper half (the
        start plane r = unit(z x a) flips with the axis, so the right-hand
        sweep goes up either way — pinned convention robustness)."""
        proj = Project(name="arch flipped")
        proj.add(Revolve(profile=ARCH_SECTION,
                      path=[ARCH_AXIS[1], ARCH_AXIS[0]], angle=18000,
                      name="arch_flip"))
        f, _ = compile_building(proj)
        vs, _ = tessellate(product_by_name(f, "arch_flip"))
        assert vs[:, 2].min() == pytest.approx(2.0, abs=0.005)
        assert vs[:, 2].max() == pytest.approx(2.5, abs=0.005)

    def test_full_revolution_vertical_axis(self):
        """Turned column base: full 36000 about a vertical axis."""
        proj = Project(name="col base")
        proj.add(Revolve(profile=[Point2D(x=0, y=0), Point2D(x=400, y=0),
                               Point2D(x=250, y=600), Point2D(x=0, y=600)],
                      path=[Point(x=3000, y=2000, z=0), Point(x=3000, y=2000, z=1000)],
                      name="col_base"))
        f, _ = compile_building(proj)
        vs, vol = tessellate(product_by_name(f, "col_base"))
        # Truncated cone-ish: radius 0.4 -> 0.25 over height 0.6, centered (3, 2).
        assert vs[:, 2].min() == pytest.approx(0.0, abs=0.005)
        assert vs[:, 2].max() == pytest.approx(0.6, abs=0.005)
        assert vs[:, 0].min() == pytest.approx(3.0 - 0.4, abs=0.01)
        assert vs[:, 0].max() == pytest.approx(3.0 + 0.4, abs=0.01)
        # Conical frustum volume: pi*h/3 * (R^2 + R*r + r^2)
        expected = math.pi * 0.6 / 3 * (0.4 ** 2 + 0.4 * 0.25 + 0.25 ** 2)
        assert vol == pytest.approx(expected, rel=0.02)


# ---------------------------------------------------------------------------
# Bar
# ---------------------------------------------------------------------------


class TestBarIfc:
    def _quantities(self, f, product):
        for rel in f.by_type("IfcRelDefinesByProperties"):
            if product in rel.RelatedObjects and \
                    rel.RelatingPropertyDefinition.is_a("IfcElementQuantity"):
                eq = rel.RelatingPropertyDefinition
                assert eq.Name == "Qto_ReinforcingElementBaseQuantities"
                return {q.Name: (q.LengthValue if q.is_a("IfcQuantityLength")
                                 else q.WeightValue)
                        for q in eq.Quantities}
        raise AssertionError("no IfcElementQuantity attached")

    def test_reinforcing_bar_entity_and_attributes(self):
        proj = Project(name="bar")
        proj.add(Bar(path=[Point(x=0, y=-1000, z=100), Point(x=0, y=1000, z=100)],
                     diameter=8, mark="M1", name="bar_1"))
        f, _ = compile_building(proj)
        bars = f.by_type("IfcReinforcingBar")
        assert len(bars) == 1
        bar = bars[0]
        assert bar.NominalDiameter == pytest.approx(0.008)
        assert bar.Tag == "M1"
        assert bar.PredefinedType == "MAIN"
        assert f.by_type("IfcSweptDiskSolid")[0].Radius == pytest.approx(0.004)

    def test_length_and_mass_quantities(self):
        """Length = true centerline; Weight = length * area * density
        (density pinned to the registry Pset_MaterialCommon.MassDensity)."""
        proj = Project(name="bar qto")
        proj.add(Bar(path=[Point(x=0, y=-1000, z=100), Point(x=0, y=1000, z=100)],
                     diameter=8, name="bar_1"))
        f, _ = compile_building(proj)
        q = self._quantities(f, product_by_name(f, "bar_1"))
        assert q["Length"] == pytest.approx(2.0)
        expected_mass = 2.0 * math.pi * 0.004 ** 2 * 7850  # ~0.789 kg
        assert q["Weight"] == pytest.approx(expected_mass, rel=1e-6)
        # kg mass unit declared
        units = f.by_type("IfcUnitAssignment")[0].Units
        assert any(getattr(u, "UnitType", None) == "MASSUNIT" for u in units)

    def test_bent_bar_length_is_arc_corrected(self):
        # L-bend, phi 8 -> auto centerline bend radius 20 mm.
        proj = Project(name="bar bent")
        proj.add(Bar(path=[Point(x=0, y=0, z=0), Point(x=0, y=1000, z=0),
                           Point(x=0, y=1000, z=1000)],
                     diameter=8, name="bar_l"))
        f, _ = compile_building(proj)
        q = self._quantities(f, product_by_name(f, "bar_l"))
        r = 0.020
        expected = 2.0 - 2 * r + r * math.pi / 2
        assert q["Length"] == pytest.approx(expected, rel=1e-6)

    def test_stirrup_predefined_type_is_ligature(self):
        proj = Project(name="stirrup")
        proj.add(Bar(path=stirrup_path(width=230, height=530,
                                       origin=Point(x=0, y=0, z=35)),
                     diameter=8, bar_type="stirrup", mark="S1", name="st_1"))
        f, _ = compile_building(proj)
        bar = f.by_type("IfcReinforcingBar")[0]
        assert bar.PredefinedType == "LIGATURE"

    def test_steel_material_associated(self):
        proj = Project(name="bar mat")
        proj.add(Bar(path=[Point(x=0, y=0, z=0), Point(x=0, y=2000, z=0)],
                     diameter=8, name="bar_1"))
        f, _ = compile_building(proj)
        assert any(m.Name == "Steel_B500B" for m in f.by_type("IfcMaterial"))


# ---------------------------------------------------------------------------
# Path Sweep
# ---------------------------------------------------------------------------

SEC = [Point2D(x=-45, y=-20), Point2D(x=45, y=-20),
       Point2D(x=45, y=20), Point2D(x=-45, y=20)]


class TestProfilePathIfc:
    def test_straight_two_point_path_is_plain_extrusion(self):
        proj = Project(name="straight")
        proj.add(Sweep(profile=SEC,
                         path=[Point(x=0, y=0, z=0), Point(x=2000, y=0, z=0)],
                         name="p_1"))
        f, _ = compile_building(proj)
        member = product_by_name(f, "p_1")
        rep = member.Representation.Representations[0]
        assert rep.RepresentationType == "SweptSolid"
        assert len(rep.Items) == 1
        assert rep.Items[0].is_a("IfcExtrudedAreaSolid")
        assert rep.Items[0].Depth == pytest.approx(2.0)
        vs, vol = tessellate(member)
        assert vol == pytest.approx(2.0 * 0.09 * 0.04, rel=1e-3)

    def test_material_section_rafter(self):
        """The spec's rafter: member Material carries the section."""
        proj = Project(name="rafter")
        proj.add(Sweep(material=Material(key="Timber_C24", profile_mm=(45, 195)),
                         path=[Point(x=0, y=-4800, z=2700), Point(x=0, y=0, z=4500)],
                         name="rafter_0"))
        f, _ = compile_building(proj)
        member = product_by_name(f, "rafter_0")
        vs, vol = tessellate(member)
        length = math.hypot(4.8, 1.8)
        assert vol == pytest.approx(length * 0.045 * 0.195, rel=1e-3)
        assert any(m.Name == "Timber_C24" for m in f.by_type("IfcMaterial"))

    def test_mitered_closed_loop_frame_exact_volume(self):
        """Closed rectangular loop = frame with MITERED corners.

        4 segments, 4 mitered corners: exact volume 2t(w+h)*d — any corner
        double-solid overlap (or gap) breaks this equality.
        """
        proj = Project(name="frame")
        loop = [Point(x=0, y=100, z=0), Point(x=1000, y=100, z=0),
                Point(x=1000, y=100, z=1000), Point(x=0, y=100, z=1000),
                Point(x=0, y=100, z=0)]
        proj.add(Sweep(profile=SEC, path=loop, name="frame_1"))
        f, _ = compile_building(proj)
        member = product_by_name(f, "frame_1")
        rep = member.Representation.Representations[0]
        assert rep.RepresentationType == "CSG"
        # One item: the union chain over 4 mitered segments (single-item
        # CSG representation per IFC4). 4 base extrusions, each trimmed by
        # half-space planes at both corner joints.
        assert len(rep.Items) == 1
        assert rep.Items[0].is_a("IfcBooleanResult")
        assert len(f.by_type("IfcExtrudedAreaSolid")) == 4
        assert len(f.by_type("IfcHalfSpaceSolid")) == 8  # 4 corners x 2 trims
        vs, vol = tessellate(member)
        # section: 90 through wall (X), 40 in-plane (Y); centerline 1m square
        expected = 2 * 0.04 * (1.0 + 1.0) * 0.09
        assert vol == pytest.approx(expected, rel=1e-4), \
            "mitered frame volume off — corner overlap or gap"
        # through-wall extent: 90 mm centered on y=0.1
        assert vs[:, 1].min() == pytest.approx(0.055, abs=1e-4)
        assert vs[:, 1].max() == pytest.approx(0.145, abs=1e-4)

    def test_window_frame_material_section_through_wall(self):
        """The spec's mitered window-frame example: closed-loop Sweep with
        Material(profile_mm=(95, 45)) inside a Window — 95 through the wall
        on ALL four members (pinned orientation convention)."""
        proj = Project(name="window frame")
        wall = Wall(name="wall_west")
        wall.add(Box(start=Point(x=-4500, y=-4500, z=0),
                       end=Point(x=-4200, y=4500, z=2700),
                       material="Brick_Red_DK"))
        win = Window(width=1200, height=1400, name="win_w0")
        fw = 45
        c = int(round(fw / 2))
        frame = [Point(x=c, y=100, z=c), Point(x=1200 - c, y=100, z=c),
                 Point(x=1200 - c, y=100, z=1400 - c), Point(x=c, y=100, z=1400 - c),
                 Point(x=c, y=100, z=c)]
        win.add(Sweep(material=Material(key="Timber_C24", profile_mm=(95, 45)),
                        path=frame, name="frame_w0"))
        wall.anchor(win, along=1500, up=900)
        proj.add(wall)
        f, _ = compile_building(proj)
        member = product_by_name(f, "frame_w0")
        # tessellate_own: the frame lives INSIDE the opening — default
        # settings would subtract the void and measure only the 0.5 mm
        # overhang ring (see _SETTINGS_OWN).
        vs, vol = tessellate_own(member)
        # Frame centerline: 1156 x 1356 loop; 45 in-plane, 95 through wall.
        w, h, t, d = 1.156, 1.356, 0.045, 0.095
        assert vol == pytest.approx(2 * t * (w + h) * d, rel=1e-4)
        # The wall runs along Y (west wall): opening-local Y (through wall)
        # is world X here. 95 mm through-wall extent, centered 100 mm into
        # the wall from the interior face at x=-4500.
        assert vs[:, 0].max() - vs[:, 0].min() == pytest.approx(d, abs=1e-4)
        # Aggregated under the IfcWindow.
        window = product_by_name(f, "win_w0")
        rels = [r for r in f.by_type("IfcRelAggregates")
                if r.RelatingObject == window]
        assert rels and member in rels[0].RelatedObjects

    def test_open_multi_point_path_miters_joint(self):
        proj = Project(name="L")
        proj.add(Sweep(profile=SEC,
                         path=[Point(x=0, y=100, z=0), Point(x=1000, y=100, z=0),
                               Point(x=1000, y=100, z=1000)],
                         name="l1"))
        f, _ = compile_building(proj)
        member = product_by_name(f, "l1")
        vs, vol = tessellate(member)
        # Two 1m legs mitered at the corner: union area = 2*1.02*0.04 - 0.04^2
        expected = (2 * 1.02 * 0.04 - 0.04 ** 2) * 0.09
        assert vol == pytest.approx(expected, rel=1e-3)

    def test_filleted_open_path_facets_corner(self):
        proj = Project(name="fillet profile")
        proj.add(Sweep(profile=SEC,
                         path=[Point(x=0, y=100, z=0), Point(x=1000, y=100, z=0),
                               Point(x=1000, y=100, z=1000)],
                         fillet_radius=200, name="pf1"))
        f, _ = compile_building(proj)
        member = product_by_name(f, "pf1")
        rep = member.Representation.Representations[0]
        # Single-item union chain; 90 deg corner at <=15 deg facet steps
        # -> at least 6 facet segments + 2 legs = 8+ base extrusions.
        assert len(rep.Items) == 1
        assert len(f.by_type("IfcExtrudedAreaSolid")) >= 8
        vs, vol = tessellate(member)
        # Volume ~ section area * arc-corrected centerline length.
        length = 2.0 - 2 * 0.2 + 0.2 * math.pi / 2
        assert vol == pytest.approx(0.09 * 0.04 * length, rel=0.02)


# ---------------------------------------------------------------------------
# Generic Element
# ---------------------------------------------------------------------------


class TestElementIfc:
    def test_whitelisted_class_emitted(self):
        proj = Project(name="stair")
        proj.add(Element(ifc_class="IfcStair", name="stair_main",
                         props={"Pset_StairCommon": {"NumberOfRiser": 16}}))
        f, _ = compile_building(proj)
        stairs = f.by_type("IfcStair")
        assert len(stairs) == 1
        assert stairs[0].Name == "stair:stair_main"
        # props= emitted on the product.
        psets = [rel.RelatingPropertyDefinition
                 for rel in f.by_type("IfcRelDefinesByProperties")
                 if stairs[0] in rel.RelatedObjects
                 and rel.RelatingPropertyDefinition.is_a("IfcPropertySet")]
        assert any(p.Name == "Pset_StairCommon" for p in psets)

    def test_children_aggregate_under_element(self):
        proj = Project(name="railing")
        railing = Element(ifc_class="IfcRailing", name="railing_1")
        railing.add(
            Box(start=Point(x=0, y=0, z=0), end=Point(x=100, y=2000, z=100)),
            Pipe(path=[Point(x=50, y=0, z=100), Point(x=50, y=2000, z=100)],
                radius=20),
            Revolve(profile=[Point2D(x=0, y=0), Point2D(x=40, y=0), Point2D(x=40, y=100),
                          Point2D(x=0, y=100)],
                 path=[Point(x=50, y=1000, z=100), Point(x=50, y=1000, z=200)]),
        )
        proj.add(railing)
        f, _ = compile_building(proj)
        rail = f.by_type("IfcRailing")[0]
        rels = [r for r in f.by_type("IfcRelAggregates") if r.RelatingObject == rail]
        assert rels and len(rels[0].RelatedObjects) == 3

    def test_empty_element_emits_product_without_geometry(self):
        proj = Project(name="empty stair")
        proj.add(Element(ifc_class="IfcStair", name="stair_x"))
        f, _ = compile_building(proj)
        stair = f.by_type("IfcStair")[0]
        assert stair.Representation is None

    def test_semantic_containers_unchanged(self):
        # Wall/Beam etc. stay their own classes — Element is additive.
        proj = Project(name="wall stays")
        wall = Wall(name="w1")
        wall.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=6000, y=300, z=2700)))
        proj.add(wall)
        f, _ = compile_building(proj)
        assert len(f.by_type("IfcWall")) == 1


# ---------------------------------------------------------------------------
# Cross-primitive booleans — the canonical arched doorway
# ---------------------------------------------------------------------------


class TestCrossPrimitiveBooleans:
    def _arched_wall_building(self):
        proj = Project(name="arched doorway")
        wall = Wall(name="wall_north")
        n_body = Box(start=Point(x=-6000, y=4200, z=0),
                       end=Point(x=6000, y=4500, z=2700), name="wall_n_body")
        door = Box(start=Point(x=-500, y=4170, z=0),
                     end=Point(x=500, y=4530, z=2000))
        arch = Revolve(profile=ARCH_SECTION, path=ARCH_AXIS, angle=18000)
        door.union(arch)      # merge the two voids into one
        n_body.difference(door)    # pierce the wall
        wall.add(n_body)
        proj.add(wall)
        return proj

    def test_arched_doorway_compiles_to_nested_boolean(self):
        f, _ = compile_building(self._arched_wall_building())
        wall = product_by_name(f, "wall_north")
        rep = wall.Representation.Representations[0]
        assert rep.RepresentationType == "CSG"
        outer = rep.Items[0]
        assert outer.is_a("IfcBooleanResult")
        assert outer.Operator == "DIFFERENCE"
        # The void operand is itself a UNION of box + revolved arch.
        void = outer.SecondOperand
        assert void.is_a("IfcBooleanResult")
        assert void.Operator == "UNION"
        # The arch is not an IfcRevolvedAreaSolid (web-ifc draws none), but it
    # must still be an IfcSolidModel or this very UNION
        # would be an illegal IfcBooleanOperand. That is the property worth
        # pinning here; the resulting VOLUME is pinned by the sibling test.
        assert void.SecondOperand.is_a("IfcSolidModel")
        assert not void.SecondOperand.is_a("IfcRevolvedAreaSolid")

    def test_arched_doorway_measured_volume(self):
        """Wall volume = box - door void - arch half-cylinder (both clipped
        to the 300 mm wall thickness)."""
        f, _ = compile_building(self._arched_wall_building())
        wall = product_by_name(f, "wall_north")
        vs, vol = tessellate(wall)
        box = 12.0 * 0.3 * 2.7
        door = 1.0 * 0.3 * 2.0
        arch = math.pi * 0.5 ** 2 / 2 * 0.3
        assert vol == pytest.approx(box - door - arch, rel=0.005)

    def test_solid_cuts_rod(self):
        proj = Project(name="rod cut")
        slab = Box(start=Point(x=-1000, y=-1000, z=0),
                     end=Point(x=1000, y=1000, z=200), name="slab_1")
        duct = Pipe(path=[Point(x=0, y=-1100, z=100), Point(x=0, y=1100, z=100)],
                   radius=50)
        slab.difference(duct)
        proj.add(slab)
        f, _ = compile_building(proj)
        vs, vol = tessellate(product_by_name(f, "slab_1"))
        box = 2.0 * 2.0 * 0.2
        duct_vol = math.pi * 0.05 ** 2 * 2.0  # clipped to the 2 m box
        assert vol == pytest.approx(box - duct_vol, rel=0.005)

    def test_rod_cuts_solid(self):
        proj = Project(name="rod cut by solid")
        rail = Pipe(path=[Point(x=0, y=0, z=0), Point(x=2000, y=0, z=0)],
                   radius=50, name="rail_1")
        notch = Box(start=Point(x=900, y=-60, z=-60),
                      end=Point(x=1100, y=60, z=60))
        rail.difference(notch)
        proj.add(rail)
        f, _ = compile_building(proj)
        vs, vol = tessellate(product_by_name(f, "rail_1"))
        full = math.pi * 0.05 ** 2 * 2.0
        cut = math.pi * 0.05 ** 2 * 0.2
        assert vol == pytest.approx(full - cut, rel=0.02)

    def test_consumed_operand_never_renders_standalone(self):
        """An element consumed as a void AND added to the project renders
        exactly once (as the void) — the standalone render is skipped."""
        proj = Project(name="double add")
        body = Box(start=Point(x=0, y=0, z=0),
                     end=Point(x=6000, y=300, z=2700), name="body_1")
        void = Box(start=Point(x=1000, y=-30, z=0),
                     end=Point(x=2000, y=330, z=2100), name="void_1")
        body.difference(void)
        proj.add(body)
        proj.add(void)  # mistake: also added standalone
        report = validate_project_report(proj)
        assert any("consumed as a boolean operand" in w for w in report.warnings)
        normalized = normalize_project_to_meters(proj)
        result = generate_ifc(normalized)
        assert result.success, result.error
        f = ifcopenshell.file.from_string(result.ifc_content)
        # No standalone product named void_1 — it exists only as an operand.
        assert not [p for p in f.by_type("IfcProduct") if p.Name == "void_1"]
        assert result.stats["boxes"] == 1


# ---------------------------------------------------------------------------
# Integration: the spec's stirruped beam
# ---------------------------------------------------------------------------


class TestStirrupedBeamIntegration:
    def test_beam_with_stirrups(self):
        """Canonical spec example: concrete beam + stirrup_path Bars with
        grade auto-registry."""
        proj = Project(name="stirruped beam")
        beam = Beam(name="beam_a1")
        beam.add(Box(start=Point(x=-150, y=-2000, z=0),
                       end=Point(x=150, y=2000, z=600),
                       material="Concrete_C25-30"))
        beam.add(*[Bar(path=stirrup_path(width=230, height=530,
                                         origin=Point(x=0, y=int(y), z=35)),
                       diameter=8, grade="B500B", bar_type="stirrup", mark="S1")
                   for y in range(-1900, 1901, 150)])
        proj.add(beam)
        f, result = compile_building(proj)
        bars = f.by_type("IfcReinforcingBar")
        assert len(bars) == 26  # -1900..1900 step 150
        assert all(b.PredefinedType == "LIGATURE" for b in bars)
        assert all(b.Tag == "S1" for b in bars)
        assert all(b.NominalDiameter == pytest.approx(0.008) for b in bars)
        # Bars aggregate under the beam.
        ifc_beam = f.by_type("IfcBeam")[0]
        rels = [r for r in f.by_type("IfcRelAggregates")
                if r.RelatingObject == ifc_beam]
        assert rels and len(rels[0].RelatedObjects) == 27  # body + 26 stirrups
        # Registry materials present for both concrete and steel.
        material_names = {m.Name for m in f.by_type("IfcMaterial")}
        assert {"Concrete_C25-30", "Steel_B500B"} <= material_names
        # Stirrup geometry sits inside the beam section minus cover.
        vs, _ = tessellate(bars[0])
        assert vs[:, 0].min() >= -0.150 + 0.030
        assert vs[:, 0].max() <= 0.150 - 0.030
        assert vs[:, 2].min() >= 0.030
        assert vs[:, 2].max() <= 0.600 - 0.030
