"""Contour-mode Wall bodies (gable-end walls) — DSL v1.5.

A Wall's single body Solid may be authored in contour mode
(``Extrude(contour=[...], thickness=)``) when the contour is a planar
VERTICAL polygon — the gable-end / sloped-top wall case. This suite pins:

1. the ifcopenshell backend emits an IfcWall with the extruded-contour
   representation (``_create_wall_contour``);
2. Window/Door children still become IfcOpeningElement voids +
   IfcRelVoidsElement + fills, with box-mode opening semantics preserved
   (``.anchor(along=)`` along the run axis from the wall start, ``up=`` above
   the contour's min z, void through the full thickness) — including a
   void authored partially OUTSIDE the gable slope (emitted as authored,
   no clipping, no silent repositioning);
3. tilted / non-planar / degenerate contours and a missing thickness
   source are compile ERRORs in ``validate_project_report`` — loud,
   never a generator drop;
4. ``can_stream`` routes any project holding a contour-bodied Wall to
   the ifcopenshell backend (``_uses_contour_wall_body`` — streaming
   parity by routing, same precedent as contour glass in openings);
5. box-mode wall bodies are untouched by the branch (regression control).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from lite_step.models import Project, Point, Box, Extrude, Wall, Window
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)


# ---------------------------------------------------------------------------
# Fixtures (int-mm, strict-friendly)
# ---------------------------------------------------------------------------


def _gable_body(**kw):
    """5-point gable-end contour: 6 m rectangle + apex, in the y=0 plane.

    Winding gives a Newell normal of (0, -1, 0); the extrusion (300 mm)
    runs toward -Y, so the wall occupies y ∈ [-300, 0] mm.
    """
    return Extrude(
        contour=[
            Point(x=0, y=0, z=0),
            Point(x=6000, y=0, z=0),
            Point(x=6000, y=0, z=2400),
            Point(x=3000, y=0, z=3600),
            Point(x=0, y=0, z=2400),
        ],
        thickness=300,
        **kw,
    )


def _box_body(**kw):
    return Box(
        start=Point(x=0, y=-300, z=0),
        end=Point(x=6000, y=0, z=2400),
        **kw,
    )


def _gable_building(*wall_children, wall_name="wall_gable",
                    along=0, up=0) -> Project:
    """``wall_children`` that are openings are ANCHORED (v20: an opening has
    no position of its own); anything else is plain containment."""
    from lite_step.models import Door, Window
    wall = Wall(name=wall_name)
    wall.add(_gable_body())
    for child in wall_children:
        if isinstance(child, (Window, Door)):
            wall.anchor(child, along=along, up=up)
        else:
            wall.add(child)
    b = Project(name="gable-house")
    b.add(wall)
    return b


def _generate_ifcopenshell(project: Project):
    """Normalize + generate on the ifcopenshell backend (skip if missing)."""
    pytest.importorskip("ifcopenshell")
    from lite_step.ifc.generator import generate_ifc

    normalized = normalize_project_to_meters(project)
    result = generate_ifc(normalized)
    assert result.success, result.error
    assert result.ifc_content
    return result


def _open_model(ifc_text: str):
    import ifcopenshell

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ifc", delete=False, encoding="utf-8"
    ) as f:
        f.write(ifc_text)
        path = f.name
    try:
        return ifcopenshell.open(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _world_origin(product) -> tuple:
    """Product's world-space placement origin (feature.add_feature re-parents
    opening placements relative to the wall, so raw Location is wall-local)."""
    import ifcopenshell.util.placement as plc

    matrix = plc.get_local_placement(product.ObjectPlacement)
    return (float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3]))


# ---------------------------------------------------------------------------
# 1. Gable wall body → IfcWall with extruded-contour representation
# ---------------------------------------------------------------------------


class TestGableWallBody:
    def test_gable_wall_emits_ifcwall_with_contour_extrusion(self):
        result = _generate_ifcopenshell(_gable_building())
        content = result.ifc_content

        assert "IFCWALL(" in content
        assert "IFCEXTRUDEDAREASOLID" in content
        assert "IFCARBITRARYCLOSEDPROFILEDEF" in content

        model = _open_model(content)
        walls = model.by_type("IfcWall")
        assert len(walls) == 1
        assert walls[0].Name == "wall:wall_gable"

        rep_items = walls[0].Representation.Representations[0].Items
        assert len(rep_items) == 1
        solid = rep_items[0]
        assert solid.is_a("IfcExtrudedAreaSolid")
        assert solid.Depth == pytest.approx(0.3)
        assert solid.SweptArea.is_a("IfcArbitraryClosedProfileDef")
        # 5-point contour closes back onto its first point → 6 polyline points
        assert len(solid.SweptArea.OuterCurve.Points) == 6

    def test_gable_wall_placement_frame(self):
        result = _generate_ifcopenshell(_gable_building())
        model = _open_model(result.ifc_content)
        wall = model.by_type("IfcWall")[0]

        ax2 = wall.ObjectPlacement.RelativePlacement
        # Placement origin = contour centroid (meters)
        assert tuple(ax2.Location.Coordinates) == pytest.approx(
            (3.0, 0.0, 1.68), abs=1e-6
        )
        # Extrusion axis = plane normal (winding gives -Y)
        assert tuple(ax2.Axis.DirectionRatios) == pytest.approx(
            (0.0, -1.0, 0.0), abs=1e-6
        )

    def test_gable_wall_counts_as_wall_in_stats(self):
        result = _generate_ifcopenshell(_gable_building())
        assert result.stats["walls"] == 1

    def test_validation_accepts_vertical_gable(self):
        report = validate_project_report(_gable_building())
        assert report.errors == []


# ---------------------------------------------------------------------------
# 2. Openings on a gable wall
# ---------------------------------------------------------------------------


class TestGableWallOpenings:
    def test_window_emits_opening_void_and_fill(self):
        win = Window(width=1200, height=1000, name="win_g0")
        result = _generate_ifcopenshell(
            _gable_building(win, along=1500, up=800))
        content = result.ifc_content

        assert "IFCOPENINGELEMENT" in content
        assert "IFCRELVOIDSELEMENT" in content
        assert "IFCRELFILLSELEMENT" in content
        assert "IFCWINDOW" in content

        model = _open_model(content)
        wall = model.by_type("IfcWall")[0]
        voids = wall.HasOpenings
        assert len(voids) == 1
        opening = voids[0].RelatedOpeningElement
        assert opening.is_a("IfcOpeningElement")
        fills = opening.HasFillings
        assert len(fills) == 1
        assert fills[0].RelatedBuildingElement.is_a("IfcWindow")

    def test_window_position_follows_run_axis_and_sill(self):
        """along= walks the wall's own +x (right as seen from OUTSIDE) from
        its outer-face origin; up= is height above the contour's min z; the
        void still centers through the thickness.

        Gable plane y=0, thickness 300 toward -Y, exterior +Y → outer face
        y=0, mid-plane y=-0.15 m, and local +x runs -X from x=6. Opening
        placement = (6 - along - width/2, -0.15, up) in meters. Pre-v21 the
        x read 2.1: ``along`` walked a +X-canonicalized run that ignored
        which side of the wall was outside.
        """
        win = Window(width=1200, height=1000, name="win_g0")
        result = _generate_ifcopenshell(
            _gable_building(win, along=1500, up=800))
        model = _open_model(result.ifc_content)

        opening = model.by_type("IfcOpeningElement")[0]
        assert _world_origin(opening) == pytest.approx((3.9, -0.15, 0.8),
                                                       abs=1e-6)

        # Void box: width × wall thickness, extruded to the window height —
        # cuts through the full 300 mm thickness.
        solid = opening.Representation.Representations[0].Items[0]
        assert solid.is_a("IfcExtrudedAreaSolid")
        assert solid.SweptArea.XDim == pytest.approx(1.2)
        assert solid.SweptArea.YDim == pytest.approx(0.3)
        assert solid.Depth == pytest.approx(1.0)

    def test_window_in_gable_slope_is_emitted_as_authored(self):
        """A void reaching past the gable roofline is NOT clipped or
        repositioned — IFC allows a void partially outside the body."""
        # up=2600 (the sill) + height 1200 tops out at z=3800 > apex 3600.
        win = Window(width=800, height=1200, name="win_attic")
        result = _generate_ifcopenshell(
            _gable_building(win, along=2600, up=2600))
        model = _open_model(result.ifc_content)

        openings = model.by_type("IfcOpeningElement")
        assert len(openings) == 1
        assert _world_origin(openings[0]) == pytest.approx((3.0, -0.15, 2.6),
                                                           abs=1e-6)
        solid = openings[0].Representation.Representations[0].Items[0]
        assert solid.Depth == pytest.approx(1.2)  # full authored height


# ---------------------------------------------------------------------------
# 3. Compile-time validation: planar VERTICAL polygon or ERROR
# ---------------------------------------------------------------------------

_CONSTRAINT = "contour body must be a planar vertical polygon"


def _wall_with_contour(points, thickness=300):
    wall = Wall(name="wall_bad")
    wall.add(Extrude(contour=points, thickness=thickness))
    b = Project(name="bad-house")
    b.add(wall)
    return b


class TestContourWallValidation:
    def test_tilted_plane_is_a_compile_error(self):
        # Planar parallelogram leaning 2 m inward over 2.4 m of rise.
        b = _wall_with_contour([
            Point(x=0, y=0, z=0),
            Point(x=6000, y=0, z=0),
            Point(x=6000, y=2000, z=2400),
            Point(x=0, y=2000, z=2400),
        ])
        report = validate_project_report(b)
        assert any(_CONSTRAINT in e for e in report.errors)
        assert any("out of vertical" in e for e in report.errors)

    def test_non_planar_contour_is_a_compile_error(self):
        # Fourth point twists 500 mm out of the y=0 plane.
        b = _wall_with_contour([
            Point(x=0, y=0, z=0),
            Point(x=6000, y=0, z=0),
            Point(x=6000, y=0, z=2400),
            Point(x=0, y=500, z=2400),
        ])
        report = validate_project_report(b)
        assert any(_CONSTRAINT in e for e in report.errors)
        assert any("deviate" in e for e in report.errors)

    def test_degenerate_collinear_contour_is_a_compile_error(self):
        b = _wall_with_contour([
            Point(x=0, y=0, z=0),
            Point(x=3000, y=0, z=0),
            Point(x=6000, y=0, z=0),
        ])
        report = validate_project_report(b)
        assert any(_CONSTRAINT in e for e in report.errors)
        assert any("degenerate" in e for e in report.errors)

    def test_missing_thickness_source_is_a_compile_error(self):
        wall = Wall(name="wall_no_thickness")
        wall.add(Extrude(contour=[
            Point(x=0, y=0, z=0),
            Point(x=6000, y=0, z=0),
            Point(x=6000, y=0, z=2400),
        ]))
        b = Project(name="no-thickness")
        b.add(wall)
        report = validate_project_report(b)
        assert any("needs a thickness source" in e for e in report.errors)

    def test_error_names_the_wall_and_the_escape_hatch(self):
        b = _wall_with_contour([
            Point(x=0, y=0, z=0),
            Point(x=6000, y=0, z=0),
            Point(x=6000, y=2000, z=2400),
            Point(x=0, y=2000, z=2400),
        ])
        report = validate_project_report(b)
        msg = next(e for e in report.errors if _CONSTRAINT in e)
        assert "Element(ifc_class='IfcWall')" in msg


# ---------------------------------------------------------------------------
# 4. A contour-bodied Wall compiles end to end
# ---------------------------------------------------------------------------


class TestContourWallCompiles:
    def test_a_gable_project_compiles_end_to_end(self):
        """A contour-bodied (gable) Wall must reach the file as one IfcWall."""
        pytest.importorskip("ifcopenshell")
        from lite_step.ifc.generator import generate_ifc

        normalized = normalize_project_to_meters(_gable_building())
        result = generate_ifc(normalized)  # default backend
        assert result.success, result.error
        assert "IFCWALL(" in result.ifc_content
        assert result.stats["walls"] == 1


# ---------------------------------------------------------------------------
# 5. Box-mode regression control
# ---------------------------------------------------------------------------


class TestBoxWallRegression:
    def test_box_wall_with_window_unchanged(self):
        wall = Wall(name="wall_box")
        wall.add(_box_body())
        wall.anchor(Window(width=1200, height=1000, name="win_b0"),
                    along=1500, up=800)
        b = Project(name="box-house")
        b.add(wall)
        assert validate_project_report(b).errors == []

        result = _generate_ifcopenshell(b)
        model = _open_model(result.ifc_content)
        wall_e = model.by_type("IfcWall")[0]
        assert wall_e.Name == "wall:wall_box"
        assert len(wall_e.HasOpenings) == 1
        opening = wall_e.HasOpenings[0].RelatedOpeningElement
        # Box wall along X in y ∈ [-0.3, 0] → same outer-face frame and
        # mid-thickness void centering as the contour case above (semantics
        # parity: the box and contour derivations must not diverge).
        assert _world_origin(opening) == pytest.approx((3.9, -0.15, 0.8),
                                                       abs=1e-6)


# ---------------------------------------------------------------------------
# An unsupported boolean operand is skipped, not fatal
# ---------------------------------------------------------------------------


def test_contour_cut_skips_an_unsupported_operand_instead_of_crashing(caplog):
    """`_create_contour_cut_solid` must answer the way its sibling already does.

    `_create_cut_solid` guards the identical condition — an operand with no
    `start`/`end` is not a box-mode Solid — warns, and returns None, which
    `apply_boolean_chain` treats as "skip this operand". The contour twin,
    ~400 lines away, read `operand.start.x` unconditionally and died with
    `AttributeError: 'Extrude' object has no attribute 'start'`.

    Two code paths, one condition, opposite outcomes. Reached in practice
    whenever two contour-mode solids overlap and displacement infers a carve
    between them — and an anchored child cannot use the usual identity
    `placement=` escape hatch, because `.anchor()` and `placement=` are
    mutually exclusive by construction. Found by the 5-link anchor chain in
    scenario_b2_real_anchor_chain.py.
    """
    import logging

    import numpy as np

    from lite_step.ifc import generator as gen

    cutter = Extrude(
        contour=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0),
                 Point(x=1000, y=0, z=1000)],
        thickness=200, name="wedge",
    )
    assert getattr(cutter, "start", None) is None, (
        "fixture is wrong — an Extrude must have no start/end for this to "
        "exercise the guard"
    )

    with caplog.at_level(logging.WARNING, logger=gen.logger.name):
        out = gen._create_contour_cut_solid(
            None, cutter, (0.0, 0.0, 0.0),
            np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]), None,
        )

    # `model` and `cache` are None on purpose: the guard must answer before
    # touching either, which is what makes it a guard rather than a late catch.
    assert out is None, "an unsupported operand must be skipped, not built"
    assert any("unsupported boolean operand" in r.getMessage()
               for r in caplog.records), "the skip must be announced"
