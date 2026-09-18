"""The ``Sheet`` primitive — an open face line + a gauge, closed at compile.

``Sheet`` lowers to a path-form ``Sweep`` carrying an explicitly CLOSED
profile. Four things are pinned here and none of them are cosmetic:

* the emitted profile is an ``IfcArbitraryClosedProfileDef`` and there is no
  ``IfcCenterLineProfileDef`` anywhere. That entity looks like the right one
  for a folded sheet and is a trap — against ifcopenshell 0.8.5 a
  ``Thickness`` of 2 and of 20 give byte-identical geometry (14 verts, bbox
  exactly the bare centreline). The kernel accepts it, raises nothing, and
  builds a zero-thickness ribbon;
* the material lands on the side the polyline's DIRECTION OF TRAVEL chooses
  (``n = (-dy, dx)`` — 90 degrees LEFT), and both faces stay exactly
  ``thickness`` apart through a corner (the miter correction);
* the two failure modes the hand-written skill version passed over in silence
  — points collapsing into duplicates under int rounding, and an offset loop
  that crosses itself once the gauge exceeds the inner turn radius — are a
  dedupe and a LOUD compile error respectively;
* a placed sheet costs NO booleans, so it does not eat the CSG depth budget.

The fixture is the real sålbænk section: a 7-point outer contour whose 2.5 mm
U-bend hem is tessellated into ``segments`` chords. Note that the profile
reaching the primitive is INT MILLIMETRES (RULE 2), so the hem quantises to
the mm grid — a 2.5 mm radius has very little room there, and the gauge that
clears a finely tessellated version of it is 1 mm (which is exactly what
``Zinc_Titanium_EN988`` ships).
"""

from __future__ import annotations

import math

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.compiler.csg_depth import predict_csg_depth
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.ifc.generator import generate_ifc
from lite_step.ifc.geometry import dedupe_consecutive_2d
from lite_step.materials import registry_definition
from lite_step.models import Material, Point, Point2D, Project, Sheet, Sweep
from lite_step.models.project import Storey
from lite_step.strict import suspend_strict


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

#: The real sålbænk outer contour ("farvesiden"), origin at the bottom of the
#: U-bend hem. x runs from the drip edge INTO the building, y is up. Travel is
#: from under the window OUT to the drip edge, which puts the material below
#: the visible face — reverse the list and it would sit above it.
SILL_OUTER = [
    (122, 49),   # P1  under the window
    (72, 49),    # P2  corner D, start of the fall
    (0, 30),     # P3  corner C, top of the vertical drip edge
    (0, 3),      # P4  bottom of the drip edge, start of the U-bend
    # P5 (2.5, 0) is the lowest point — it arrives from the hem arc
    (5, 3),      # P6  end of the U-bend, start of the return hook
    (5, 10),     # P7  top of the return hook
]
SILL_HEM_CENTRE = (2.5, 2.5)
SILL_HEM_RADIUS = 2.5


def sill_profile(segments: int = 8):
    """The sill contour with its hem tessellated into ``segments`` chords.

    The hem runs from P4 at 180 degrees through (2.5, 0) at 270 to P6 at 360:
    a half circle about (2.5, 2.5) swept the short way UNDER the profile, so
    the hook returns upward. Points are int millimetres, as the DSL requires.
    """
    cx, cy = SILL_HEM_CENTRE
    pts = list(SILL_OUTER[:4])
    for i in range(1, segments):
        th = math.radians(180 + 180 * i / segments)
        pts.append((cx + SILL_HEM_RADIUS * math.cos(th),
                    cy + SILL_HEM_RADIUS * math.sin(th)))
    pts.extend(SILL_OUTER[4:])
    return [Point2D(x=int(round(x)), y=int(round(y))) for x, y in pts]


def p2d(pairs):
    return [Point2D(x=x, y=y) for x, y in pairs]


def xy(sweep: Sweep):
    """The lowered profile as int ``(x, y)`` pairs."""
    return [(int(p.x), int(p.y)) for p in sweep.profile]


def distinct_outer(profile) -> int:
    """How many points the face line has once exact duplicates are dropped."""
    return len(dedupe_consecutive_2d([(p.x, p.y) for p in profile], tol=0.0))


def project_with(*elements) -> Project:
    proj = Project(name="sheet_test")
    storey = Storey(elevation=0)
    for e in elements:
        storey.add(e)
    proj.add_storey(storey)
    return proj


def compile_project(proj: Project):
    report = validate_project_report(proj)
    assert report.errors == [], report.errors
    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    return ifcopenshell.file.from_string(result.ifc_content), result.ifc_content


PATH = [Point(x=-600, y=0, z=900), Point(x=600, y=0, z=900)]


# ---------------------------------------------------------------------------
# 1. What reaches the IFC
# ---------------------------------------------------------------------------


def test_sheet_emits_one_extrusion_over_a_closed_profile():
    """One straight run → exactly one ``IfcExtrudedAreaSolid``, and the profile
    is a CLOSED arbitrary profile — never ``IfcCenterLineProfileDef``."""
    sheet = Sheet(name="sill", profile=sill_profile(segments=2), thickness=2,
                  path=PATH, material=Material(key="Steel_S250GD_Z275"))
    model, raw = compile_project(project_with(sheet))

    solids = model.by_type("IfcExtrudedAreaSolid")
    assert len(solids) == 1, [s.id() for s in solids]
    profile = solids[0].SweptArea
    assert profile.is_a("IfcArbitraryClosedProfileDef"), profile.is_a()
    assert profile.OuterCurve.is_a("IfcPolyline")

    assert model.by_type("IfcCenterLineProfileDef") == []
    assert "IFCCENTERLINEPROFILEDEF" not in raw.upper()


def test_sheet_lowers_to_a_path_form_sweep():
    """``Sheet`` is a constructor, not an element type: what lands in the tree
    is an ordinary path-form ``Sweep``, so miters, fillets, displacement bounds
    and the streaming gate all apply without a second emission path."""
    sheet = Sheet(profile=p2d([(0, 0), (100, 0)]), thickness=5, path=PATH)
    assert isinstance(sheet, Sweep)
    assert type(sheet).__name__ == "Sweep"
    assert sheet.path is not None and sheet.profile is not None


def test_real_sill_fixture_compiles():
    """The sålbænk with a finely tessellated hem, at the 1 mm zinc gauge that
    exists for it, survives the whole validate → normalize → generate path."""
    outer = sill_profile(segments=8)
    assert distinct_outer(outer) == 12
    sheet = Sheet(name="sill", profile=outer, thickness=1, path=PATH,
                  material=Material(key="Zinc_Titanium_EN988"))
    assert len(sheet.profile) == 24        # 2 x 12, nothing collapsed at 1 mm
    model, _ = compile_project(project_with(sheet))
    assert len(model.by_type("IfcExtrudedAreaSolid")) == 1


def test_sheet_inside_a_named_container_gets_a_canonical_name():
    """Because it lowers to a Sweep, a sheet part inherits ordinary DSL v2.1
    containment naming rather than needing its own identity rules."""
    from lite_step.models import Element

    host = Element(ifc_class="IfcCovering", name="reveal")
    host.add(Sheet(name="sill", profile=sill_profile(segments=2), thickness=2,
                   path=PATH, material=Material(key="Steel_S250GD_Z275")))
    model, _ = compile_project(project_with(host))
    names = {e.Name for e in model.by_type("IfcProduct") if e.Name}
    assert "sweep:sill:covering:reveal" in names, sorted(names)


# ---------------------------------------------------------------------------
# 2. Which side gets the material
# ---------------------------------------------------------------------------


def test_material_side_follows_the_direction_of_travel():
    """``n = (-dy, dx)`` — material 90 degrees LEFT of travel. Reversing the
    same face line puts the gauge on the other side of it."""
    forward = Sheet(profile=p2d([(0, 0), (100, 0)]), thickness=5, path=PATH)
    reverse = Sheet(profile=p2d([(100, 0), (0, 0)]), thickness=5, path=PATH)

    fy = [y for _x, y in xy(forward)]
    ry = [y for _x, y in xy(reverse)]
    assert (min(fy), max(fy)) == (0, 5), xy(forward)
    assert (min(ry), max(ry)) == (-5, 0), xy(reverse)


def test_sill_winding_puts_the_metal_under_the_visible_face():
    """The shipped point order is not arbitrary — travelling from under the
    window out to the drip edge is what keeps the gauge BELOW the weather
    face instead of growing up into the reveal."""
    sheet = Sheet(profile=sill_profile(segments=2), thickness=2, path=PATH)
    tail = [p for p in xy(sheet) if p[0] > 100]     # the run under the window
    assert tail == [(122, 49), (122, 47)], tail


def test_miter_correction_keeps_both_faces_exactly_thickness_apart():
    """Through a 90 degree bend the corner offset rides the angle BISECTOR at
    ``t / (u · n)``. A plain averaged normal would pinch the corner to
    ``t / sqrt(2)``; here every face pair stays exactly ``t`` apart."""
    sheet = Sheet(profile=p2d([(0, 0), (100, 0), (100, 100)]), thickness=5,
                  path=PATH)
    assert xy(sheet) == [
        (0, 0), (100, 0), (100, 100),      # outer, forward
        (95, 100), (95, 5), (0, 5),        # inner, reversed
    ]
    # horizontal run: y = 0 against y = 5; vertical run: x = 100 against 95.
    assert abs(0 - 5) == 5
    assert abs(100 - 95) == 5


# ---------------------------------------------------------------------------
# 3. The two validations the hand-written version lacked
# ---------------------------------------------------------------------------


def test_rounding_duplicates_are_deduped_not_rejected():
    """A finely tessellated bend collapses points onto each other once the
    offset run is rounded to int mm. That is normal for sheet metal, so it
    dedupes silently and the profile stays free of repeated vertices."""
    outer = sill_profile(segments=6)
    sheet = Sheet(profile=outer, thickness=1, path=PATH)
    assert len(sheet.profile) < 2 * distinct_outer(outer)
    pts = xy(sheet)
    assert len(set(pts)) == len(pts), pts
    assert pts[0] != pts[-1]


def test_dedupe_helper_closes_the_wrap_around_pair():
    """The loop is a closed polygon, so it must never repeat its closing
    vertex — the wrap-around pair is deduped like any other."""
    assert dedupe_consecutive_2d(
        [(0, 0), (0, 0), (10, 0), (10, 10), (0, 0)], tol=0.0, wrap=True
    ) == [(0, 0), (10, 0), (10, 10)]


def test_profile_collapsing_below_three_points_is_an_error():
    """A part smaller than the millimetre grid rounds away entirely. That is a
    compile error, not a degenerate profile handed to the kernel. (Only
    reachable with strictness suspended — under RULE 2 the face line is
    already int mm.)"""
    with suspend_strict():
        with pytest.raises(ValueError, match="distinct point"):
            Sheet(profile=p2d([(0.0, 0.0), (0.4, 0.0)]), thickness=0.4,
                  path=PATH)


def test_self_intersecting_offset_is_a_loud_error():
    """At 3 mm the sill's hem is tighter than the gauge and the offset face
    crosses itself. The message names the thickness and the segment pair."""
    with pytest.raises(ValueError) as excinfo:
        Sheet(name="sill", profile=sill_profile(segments=2), thickness=3,
              path=PATH)
    msg = str(excinfo.value)
    assert "Sheet 'sill'" in msg
    assert "thickness 3 mm" in msg
    assert "self-intersects between segments 6 and 10" in msg
    assert "Reduce thickness" in msg


def test_a_finer_hem_has_less_room_for_the_gauge():
    """Tessellating the same 2.5 mm hem into 8 chords quantises it onto the mm
    grid, and 2 mm does not fit inside it — caught, not emitted knotted."""
    fine = sill_profile(segments=8)
    assert isinstance(Sheet(profile=fine, thickness=1, path=PATH), Sweep)
    with pytest.raises(ValueError, match="self-intersects"):
        Sheet(profile=fine, thickness=2, path=PATH)


def test_thicknesses_below_the_turn_radius_are_accepted():
    """The check is a real geometric test, not a blanket ban on bends."""
    for t in (1, 2):
        assert isinstance(Sheet(profile=sill_profile(segments=2), thickness=t,
                                path=PATH), Sweep)


def test_closed_profile_is_a_construction_error():
    with pytest.raises(ValueError, match="closed sheet is meaningless"):
        Sheet(profile=p2d([(0, 0), (100, 0), (100, 50), (0, 0)]),
              thickness=2, path=PATH)


def test_profile_needs_two_points_and_a_positive_gauge():
    with pytest.raises(ValueError, match="at least 2 points"):
        Sheet(profile=p2d([(0, 0)]), thickness=2, path=PATH)
    with pytest.raises(ValueError, match="thickness must be > 0"):
        Sheet(profile=p2d([(0, 0), (100, 0)]), thickness=0, path=PATH)


def test_float_thickness_is_rejected_under_strict_int_mm():
    with pytest.raises(ValueError, match="int millimeters"):
        Sheet(profile=p2d([(0, 0), (100, 0)]), thickness=1.5, path=PATH)


def test_unknown_kwarg_is_rejected_by_the_sweep_it_lowers_to():
    with pytest.raises(ValueError, match="extra"):
        Sheet(profile=p2d([(0, 0), (100, 0)]), thickness=2, path=PATH,
              thicknes=2)


# ---------------------------------------------------------------------------
# 4. Registry coupling
# ---------------------------------------------------------------------------


def test_registry_stock_gauge_is_enforced():
    """Zinc ships in one modelled gauge; a part drawn at another is a
    fabrication error, and the message says what IS available."""
    zinc = Material(key="Zinc_Titanium_EN988")
    assert isinstance(Sheet(profile=sill_profile(segments=8), thickness=1,
                            path=PATH, material=zinc), Sweep)
    with pytest.raises(ValueError) as excinfo:
        Sheet(name="sill", profile=sill_profile(segments=2), thickness=2,
              path=PATH, material=zinc)
    msg = str(excinfo.value)
    assert "not a stock gauge" in msg
    assert "'Zinc_Titanium_EN988'" in msg
    assert "1 mm" in msg


def test_bare_string_material_resolves_to_the_registry_too():
    with pytest.raises(ValueError, match="not a stock gauge"):
        Sheet(profile=sill_profile(segments=2), thickness=2, path=PATH,
              material="Zinc_Titanium_EN988")


def test_non_sheet_and_unknown_materials_do_not_gate_thickness():
    """The gauge check is a SHEET-form check. A member material, or a key that
    is not in any registry, leaves it unconstrained (the compile-time material
    walk reports the unknown key separately, as a warning)."""
    assert isinstance(Sheet(profile=p2d([(0, 0), (100, 0)]), thickness=7,
                            path=PATH,
                            material=Material(key="Steel_Stainless_304")), Sweep)
    assert isinstance(Sheet(profile=p2d([(0, 0), (100, 0)]), thickness=7,
                            path=PATH, material="Aluminium_Made_Up"), Sweep)


def test_zinc_titanium_registry_entry():
    mdef = registry_definition("Zinc_Titanium_EN988")
    assert mdef is not None
    assert mdef.function == "Finish"
    assert mdef.geometry.form == "sheet"
    assert mdef.geometry.thicknesses_mm == [1]
    assert mdef.geometry.min_bend_radius_factor_t == 1.75
    assert mdef.psets["Pset_MaterialCommon"]["MassDensity"] == 7200
    assert mdef.psets["DK_Product"]["MinFormingTemperature_C"] == 7
    assert mdef.psets["DK_EPD"]["GWP_A1A3_kgCO2e_per_kg"] == 3.9
    assert mdef.render.rgb == "#7D8A90"


def test_stainless_304_registry_entry():
    mdef = registry_definition("Steel_Stainless_304")
    assert mdef is not None
    assert mdef.function == "Finish"
    assert mdef.geometry.form == "member"
    # Made-to-order ironmongery: an explicitly EMPTY catalog, so any turned or
    # cast section is in stock — but the field is still declared, so a registry
    # entry that merely FORGOT its stock list still fails at import.
    assert mdef.geometry.stock_profiles_mm == []
    assert mdef.geometry.has_profile(18, 18)
    assert Material(key="Steel_Stainless_304", profile_mm=(18, 18)).profile_mm \
        == (18, 18)
    assert mdef.psets["Pset_MaterialSteel"]["YieldStress"] == 210
    assert mdef.psets["DK_Product"]["Designation"] == "EN 1.4301 (AISI 304)"
    assert mdef.render.rgb == "#C7CCD1"


def test_bracket_steel_keeps_its_own_stock_catalog():
    """Adding ironmongery must not loosen the CE-marked connector grade."""
    bracket = registry_definition("Steel_S250GD_Z275")
    assert bracket.geometry.thicknesses_mm == [2, 3]
    with pytest.raises(ValueError, match="not in .* stock catalog"):
        Material(key="Timber_C24", profile_mm=(47, 197))


# ---------------------------------------------------------------------------
# 5. The CSG depth budget
# ---------------------------------------------------------------------------


def test_sheet_costs_no_booleans():
    """A placed sheet part is a plain extrusion. It must not arrive already
    several ``IfcBooleanResult`` levels deep — the budget is spent on the
    building, and depth is the one geometry property that fails invisibly."""
    sheet = Sheet(name="sill", profile=sill_profile(segments=8), thickness=1,
                  path=PATH)
    proj = project_with(sheet)

    report = predict_csg_depth(proj)
    assert report.max_depth == 0, report.stats_line()

    model, _ = compile_project(proj)
    assert model.by_type("IfcBooleanResult") == []


class TestSheetIsNotATypeAndSaysSo:
    """``Sheet`` returns a ``Sweep``, so type tests against it cannot work.

    Answering ``False`` would be a silently wrong answer to a reasonable
    question — and ``isinstance`` is the idiom the DSL reference points survey
    authors at for classifying elements. So the type test raises with the fix
    in the message instead.
    """

    def _sill(self, **kw):
        return Sheet(
            profile=[Point2D(x=122, y=49), Point2D(x=72, y=49),
                     Point2D(x=0, y=30), Point2D(x=0, y=3)],
            thickness=1,
            path=[Point(x=-600, y=100, z=900), Point(x=600, y=100, z=900)],
            **kw,
        )

    def test_isinstance_raises_and_names_the_alternative(self):
        sill = self._sill()
        with pytest.raises(TypeError) as exc:
            isinstance(sill, Sheet)
        msg = str(exc.value)
        assert "constructor" in msg
        assert "Sweep" in msg, "the message must name what to test instead"

    def test_issubclass_raises_too(self):
        with pytest.raises(TypeError):
            issubclass(Sweep, Sheet)

    def test_the_sanctioned_test_actually_works(self):
        """The alternative the error message recommends must be true."""
        sill = self._sill(material=Material(key="Zinc_Titanium_EN988"))
        assert isinstance(sill, Sweep)
        assert sill.material.key == "Zinc_Titanium_EN988"
