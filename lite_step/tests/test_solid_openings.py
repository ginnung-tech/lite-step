"""Openings on a standalone ``Solid`` body — box AND contour.

``Box(...).opening(Window/Door)`` voids the Solid's own geometry (the same
IfcOpeningElement + IfcRelVoidsElement + fill machinery a Wall assembly uses).
Before that the contour-mode path crashed at compile (a NameError on
``min_pt``/``center`` that the box path shadowed), and even patched it would
have silently no-op'd because the shared opening routine re-derived the frame
from ``start``/``end`` only. This suite pins the consolidated one-frame
derivation (``_derive_opening_frame``):

1. A contour-mode standalone Solid (the repro) compiles and emits the void;
2. box-mode standalone Solid still voids, now thickness-CENTERED;
3. a contour body rotated ~30° in plan places the void along the run axis
   (the case an axis-aligned AABB cannot express);
4. loud compile failure for a tilted contour body with openings and for a
   body with neither box nor contour + openings — never a silent drop;
5. ``can_stream`` routes any standalone Solid opening to the ifcopenshell
   backend, and the default (streaming→fallback) path succeeds end-to-end.
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest

from lite_step.models import Project, Point, Box, Extrude, Window
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)


# ---------------------------------------------------------------------------
# Helpers (int-mm authoring, mirrors test_contour_wall.py assertion style)
# ---------------------------------------------------------------------------


def _gable_contour(**kw):
    """5-point gable contour in the y=0 plane (6 m wide, 2.4 m eaves, apex 3.6 m).

    Winding gives a Newell normal of (0, -1, 0); the 300 mm extrusion runs
    toward -Y, so the body occupies y in [-0.3, 0] m and the opening frame's
    thickness midline sits at y = -0.15 m.
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


def _box(**kw):
    """Box body along +X, 300 mm thick toward -Y (y in [-0.3, 0])."""
    return Box(start=Point(x=0, y=-300, z=0), end=Point(x=6000, y=0, z=2400), **kw)


def _building(elem) -> Project:
    b = Project(name="solid-openings")
    b.add(elem)
    return b


def _generate_ifcopenshell(project: Project):
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
    """Product's world-space placement origin. ``feature.add_feature``
    re-expresses the opening placement so its world origin is preserved
    regardless of where the host body sits."""
    import ifcopenshell.util.placement as plc

    matrix = plc.get_local_placement(product.ObjectPlacement)
    return (float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3]))


# ---------------------------------------------------------------------------
# 1. Repro — contour-mode standalone Solid with a Window opening
# ---------------------------------------------------------------------------


class TestContourSolidOpening:
    def test_contour_solid_opening_compiles_and_voids(self):
        """A contour Solid + Window must emit the void + fill, not raise a
            NameError at compile."""
        win = Window(width=1200, height=1000, name="win_c0")
        result = _generate_ifcopenshell(_building(
            _gable_contour(name="gable_solid").opening(win, along=1500, up=800)))
        content = result.ifc_content

        assert "IFCOPENINGELEMENT" in content
        assert "IFCRELVOIDSELEMENT" in content
        assert "IFCWINDOW" in content

        model = _open_model(content)
        openings = model.by_type("IfcOpeningElement")
        assert len(openings) == 1
        voids = model.by_type("IfcRelVoidsElement")
        assert len(voids) == 1

    def test_contour_solid_void_position_and_full_thickness(self):
        """along= walks the body's own +x from its outer-face origin; up= is
        height above the contour min z; the void centers through the 300 mm
        depth."""
        win = Window(width=1200, height=1000, name="win_c1")
        result = _generate_ifcopenshell(_building(
            _gable_contour(name="gable_solid").opening(win, along=1500, up=800)))
        model = _open_model(result.ifc_content)

        opening = model.by_type("IfcOpeningElement")[0]
        # Exterior +Y, so local +x runs -X from x=6: 6 - 1.5 - 0.6 = 3.9.
        # Mid-thickness y=-0.15 is unchanged — the hole did not move through
        # the wall, only along it (v21 outer-face frame).
        assert _world_origin(opening) == pytest.approx((3.9, -0.15, 0.8), abs=1e-6)

        solid = opening.Representation.Representations[0].Items[0]
        assert solid.is_a("IfcExtrudedAreaSolid")
        assert solid.SweptArea.XDim == pytest.approx(1.2)   # width
        assert solid.SweptArea.YDim == pytest.approx(0.3)   # full thickness
        assert solid.Depth == pytest.approx(1.0)            # height


# ---------------------------------------------------------------------------
# 2. Box-mode regression + the corrected thickness centering
# ---------------------------------------------------------------------------


class TestBoxSolidOpening:
    def test_box_solid_opening_is_thickness_centered(self):
        """Box-mode standalone openings still void, and the void now centers
        through the full thickness (y=-0.15 mid-plane) — the fix aligned
        this with the wall-assembly and contour semantics. An un-centered min
            corner (y=0) is the wrong place for the frame."""
        win = Window(width=1200, height=1000, name="win_b0")
        result = _generate_ifcopenshell(_building(
            _box(type="wall", name="box_solid").opening(win, along=1500, up=800)))
        model = _open_model(result.ifc_content)

        opening = model.by_type("IfcOpeningElement")[0]
        assert _world_origin(opening) == pytest.approx((3.9, -0.15, 0.8), abs=1e-6)
        solid = opening.Representation.Representations[0].Items[0]
        assert solid.SweptArea.YDim == pytest.approx(0.3)  # cuts full thickness


# ---------------------------------------------------------------------------
# 3. Angle / winding — a contour body rotated ~30° in plan
# ---------------------------------------------------------------------------


class TestAngledContourOpening:
    def test_void_follows_run_axis_at_plan_angle(self):
        """A vertical contour whose horizontal run is rotated 30° in plan: the
        void must be placed along (cos30, sin30), which an axis-aligned AABB
            cannot express (it produces only (1,0) or (0,1))."""
        c30, s30 = math.cos(math.radians(30)), math.sin(math.radians(30))
        length = 6000
        p1 = (round(length * c30), round(length * s30))  # (5196, 3000)
        solid = Extrude(
            contour=[
                Point(x=0, y=0, z=0),
                Point(x=p1[0], y=p1[1], z=0),
                Point(x=p1[0], y=p1[1], z=2400),
                Point(x=0, y=0, z=2400),
            ],
            thickness=300, type="wall", name="angled",
        ).opening(Window(width=1000, height=1200, name="win_a0"),
                  along=2000, up=800)

        from lite_step.ifc.generator import _derive_opening_frame

        normalized = normalize_project_to_meters(_building(solid))
        frame = _derive_opening_frame(normalized.storeys[0].elements[0])
        assert frame is not None
        # The frame's +x is right-as-seen-from-outside, so it is the run axis
        # possibly REVERSED (v21) — what must hold is that it stays on the
        # 30° line and never snaps back to a world axis, which is the whole
        # point of this case.
        assert (abs(frame.dir_x), abs(frame.dir_y)) == pytest.approx(
            (c30, s30), abs=1e-3)
        assert frame.dir_x * s30 - frame.dir_y * c30 == pytest.approx(0.0, abs=1e-3)

        result = _generate_ifcopenshell(_building(solid))
        model = _open_model(result.ifc_content)
        opening = model.by_type("IfcOpeningElement")[0]

        # wall_start (on the OUTER face) + (offset + width/2) along the
        # frame's +x, then half a thickness INWARD so the void still spans
        # the full 300 mm, sill in z.
        run_m = 2.0 + 0.5
        in_x, in_y = -frame.dir_y, frame.dir_x
        half_t = frame.thickness / 2.0
        expected = (
            frame.wall_start[0] + frame.dir_x * run_m + in_x * half_t,
            frame.wall_start[1] + frame.dir_y * run_m + in_y * half_t,
            frame.wall_start[2] + 0.8,
        )
        # 1e-4 m (0.1 mm), not 1e-6: the opening's world origin is recovered
        # through the contour proxy's genuinely-rotated axis frame, so the
        # feature.add_feature inverse/multiply round-trip carries ~1e-5 float
        # noise (the axis-aligned box/gable cases above round-trip exactly).
        assert _world_origin(opening) == pytest.approx(expected, abs=1e-4)
        # The void genuinely leaves the axis-aligned world planes.
        assert abs(_world_origin(opening)[1]) > 0.5


# ---------------------------------------------------------------------------
# 4. Loud failure — never a silent drop
# ---------------------------------------------------------------------------

_CONSTRAINT = "contour body must be a planar vertical polygon"


class TestOpeningLoudFailure:
    def test_tilted_contour_with_openings_is_compile_error(self):
        """A tilted (non-vertical) contour body that carries openings is a
        compile ERROR — the generator's opening frame does not hold for it."""
        tilted = Extrude(
            contour=[
                Point(x=0, y=0, z=0),
                Point(x=6000, y=0, z=0),
                Point(x=6000, y=2000, z=2400),
                Point(x=0, y=2000, z=2400),
            ],
            thickness=300, type="wall", name="tilted",
        ).opening(Window(width=1000, height=1000, name="win_t"),
                  along=1500, up=800)

        report = validate_project_report(_building(tilted))
        assert any(_CONSTRAINT in e for e in report.errors)
        assert any("out of vertical" in e for e in report.errors)

    def test_tilted_contour_without_openings_is_not_gated(self):
        """The frame check is gated on openings: a tilted contour Solid with NO
        openings is a legitimate sloped face (e.g. a roof) and must not error."""
        tilted = Extrude(
            contour=[
                Point(x=0, y=0, z=0),
                Point(x=6000, y=0, z=0),
                Point(x=6000, y=2000, z=2400),
                Point(x=0, y=2000, z=2400),
            ],
            thickness=300, type="roof", name="roof_face",
        )
        report = validate_project_report(_building(tilted))
        assert not any(_CONSTRAINT in e for e in report.errors)

    def test_body_with_neither_mode_and_openings_is_compile_error(self):
        """A body with neither a box nor a contour but carrying openings errors
        loudly (defensive path for hand-mutated / hand-built elements)."""
        solid = _box(type="wall", name="degen").opening(
            Window(width=800, height=1000, name="win_d"), along=1000, up=500
        )
        # Strip the box mode post-construction (validate_mode only runs at
        # construction; assignment is not re-validated).
        solid.start = None
        solid.end = None

        report = validate_project_report(_building(solid))
        assert any(
            "openings require a box body" in e and "vertical planar contour" in e
            for e in report.errors
        )

    def test_compile_does_not_crash_on_bad_opening_body(self):
        """Neither loud-failure case raises — they surface as validation
        errors, and the generator raise routes into a failed result rather
        than an uncaught exception."""
        pytest.importorskip("ifcopenshell")
        from lite_step.ifc.generator import generate_ifc

        tilted = Extrude(
            contour=[
                Point(x=0, y=0, z=0),
                Point(x=6000, y=0, z=0),
                Point(x=6000, y=2000, z=2400),
                Point(x=0, y=2000, z=2400),
            ],
            thickness=300, type="wall", name="tilted2",
        ).opening(Window(width=1000, height=1000, name="win_t2"),
                  along=1500, up=800)
        # generate_ifc never raises: a bad frame becomes success=False.
        result = generate_ifc(normalize_project_to_meters(_building(tilted)))
        assert result.success is False
        assert "openings require a box or vertical planar contour body" in result.error


# ---------------------------------------------------------------------------
# 5. Streaming gate + end-to-end default path
# ---------------------------------------------------------------------------


class TestStreamingGate:
    def test_default_backend_falls_back_end_to_end(self):
        """generate_ifc default (streaming→fallback) routes a standalone Solid
        opening through the ifcopenshell backend and still succeeds."""
        pytest.importorskip("ifcopenshell")
        from lite_step.ifc.generator import generate_ifc

        win = Window(width=1200, height=1000, name="win_e2e")
        normalized = normalize_project_to_meters(
            _building(_gable_contour(name="gable_e2e")
                      .opening(win, along=1500, up=800))
        )
        result = generate_ifc(normalized)  # default backend
        assert result.success, result.error
        assert "IFCOPENINGELEMENT" in result.ifc_content
