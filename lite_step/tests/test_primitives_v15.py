"""Construction-validation tests for the DSL v1.5 primitives (WS1 PR-E).

Covers Pipe, Revolve, Bar, generic Element, and the two-form Sweep:

* per-primitive construction validation (Revolve X >= 0, Bar off-stock
  diameter, Sweep form discrimination, Element whitelist, path checks);
* grade -> registry resolution and EC2 auto bend radius on Bar;
* strict int-mm flag interaction on the new scalar fields;
* the universal BimElement machinery each new class inherits for free:
  derivation-by-call, name/ifc_name identity, props, material.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lite_step.models import (
    Bar,
    BAR_TYPE_TO_IFC,
    Project,
    Element,
    ELEMENT_IFC_CLASS_WHITELIST,
    Material,
    Point,
    Point2D,
    Sweep,
    Pipe,
    Revolve,
    collect_consumed_operand_ids,
    Box,
    Extrude,
)
from lite_step.strict import disable_strict_int_mm, enable_strict_int_mm


@pytest.fixture(autouse=True)
def _strict_off():
    """These tests construct float geometry deliberately — run with the
    strict gate off, then restore the post-cutover default (ON)."""
    disable_strict_int_mm()
    yield
    enable_strict_int_mm()


def _pts(*coords):
    return [Point(x=x, y=y, z=z) for (x, y, z) in coords]


# ---------------------------------------------------------------------------
# Pipe
# ---------------------------------------------------------------------------


class TestRodConstruction:
    def test_minimal_rod(self):
        rod = Pipe(path=_pts((0, 0, 0), (2000, 0, 0)), radius=25)
        assert rod.fillet_radius == 0
        # DSL v2.1: ifc_name is the canonical, None until the naming pass stamps it
        assert rod.ifc_name is None
        assert rod.name is None

    def test_rod_with_fillet(self):
        rod = Pipe(path=_pts((0, 0, 0), (1000, 0, 0), (1000, 1000, 0)),
                  radius=25, fillet_radius=200, name="rail")
        # DSL v2.1: name= is the leaf; ifc_name (canonical) is stamped by the pass
        assert rod.name == "rail"

    def test_single_point_path_rejected(self):
        with pytest.raises(ValidationError, match="at least 2 points"):
            Pipe(path=_pts((0, 0, 0)), radius=25)

    def test_zero_radius_rejected(self):
        with pytest.raises(ValidationError, match="radius must be > 0"):
            Pipe(path=_pts((0, 0, 0), (1000, 0, 0)), radius=0)

    def test_negative_fillet_rejected(self):
        with pytest.raises(ValidationError, match="fillet_radius must be >= 0"):
            Pipe(path=_pts((0, 0, 0), (1000, 0, 0)), radius=25, fillet_radius=-1)

    def test_duplicate_consecutive_points_rejected(self):
        with pytest.raises(ValidationError, match="coincide"):
            Pipe(path=_pts((0, 0, 0), (0, 0, 0), (1000, 0, 0)), radius=25)

    def test_reversing_path_rejected(self):
        # 180-degree turn: out and straight back.
        with pytest.raises(ValidationError, match="nearly reverse"):
            Pipe(path=_pts((0, 0, 0), (1000, 0, 0), (0, 0, 0)), radius=25)

    def test_fillet_too_large_for_segment_rejected(self):
        # 90-degree joint, tangent trim = r; r=600 > 500 segment.
        with pytest.raises(ValidationError, match="does not fit"):
            Pipe(path=_pts((0, 0, 0), (500, 0, 0), (500, 5000, 0)),
                radius=10, fillet_radius=600)

    def test_adjacent_fillets_may_not_overlap(self):
        # Middle segment 500 long with two 90-degree joints: 2 * 300 > 500.
        with pytest.raises(ValidationError, match="does not fit"):
            Pipe(path=_pts((0, 0, 0), (1000, 0, 0), (1000, 500, 0), (2000, 500, 0)),
                radius=10, fillet_radius=300)


# ---------------------------------------------------------------------------
# Revolve
# ---------------------------------------------------------------------------

ARCH_SECTION = [Point2D(x=0, y=0), Point2D(x=500, y=0),
                Point2D(x=500, y=360), Point2D(x=0, y=360)]
ARCH_AXIS = _pts((0, 4170, 2000), (0, 4530, 2000))


class TestTurnConstruction:
    def test_arch_turn(self):
        turn = Revolve(profile=ARCH_SECTION, path=ARCH_AXIS, angle=18000)
        assert turn.angle == 18000

    def test_default_angle_full_revolution(self):
        turn = Revolve(profile=ARCH_SECTION, path=ARCH_AXIS)
        assert turn.angle == 36000

    def test_negative_section_x_rejected(self):
        with pytest.raises(ValidationError, match="radial distance.*>= 0"):
            Revolve(profile=[Point2D(x=-1, y=0), Point2D(x=500, y=0),
                          Point2D(x=500, y=360)],
                 path=ARCH_AXIS)

    def test_section_needs_three_points(self):
        with pytest.raises(ValidationError, match="at least 3 points"):
            Revolve(profile=[Point2D(x=0, y=0), Point2D(x=500, y=0)], path=ARCH_AXIS)

    def test_path_is_a_two_point_axis(self):
        with pytest.raises(ValidationError, match="2-point AXIS"):
            Revolve(profile=ARCH_SECTION,
                 path=_pts((0, 0, 0), (0, 100, 0), (0, 200, 0)))

    def test_degenerate_axis_rejected(self):
        with pytest.raises(ValidationError, match="axis points must differ"):
            Revolve(profile=ARCH_SECTION, path=_pts((0, 0, 0), (0, 0, 0)))

    @pytest.mark.parametrize("angle", [0, -18000, 36001])
    def test_angle_bounds(self, angle):
        with pytest.raises(ValidationError, match=r"\(0, 36000\]"):
            Revolve(profile=ARCH_SECTION, path=ARCH_AXIS, angle=angle)


# ---------------------------------------------------------------------------
# Bar
# ---------------------------------------------------------------------------


class TestBarConstruction:
    def test_grade_resolves_to_registry_key(self):
        bar = Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8)
        assert bar.grade == "B500B"
        assert bar.material == "Steel_B500B"

    def test_exact_registry_key_grade_accepted(self):
        bar = Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8,
                  grade="Steel_B500B")
        assert bar.material == "Steel_B500B"

    def test_off_stock_diameter_rejected(self):
        with pytest.raises(ValidationError, match="not in 'Steel_B500B' stock"):
            Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=9)

    def test_unknown_grade_rejected(self):
        with pytest.raises(ValidationError, match="unknown grade"):
            Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8, grade="X999")

    def test_non_bar_material_grade_rejected(self):
        with pytest.raises(ValidationError, match="not reinforcement bar stock"):
            Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8, grade="Timber_C24")

    @pytest.mark.parametrize("diameter,expected_radius", [
        # EC2 8.1N: mandrel 4*phi (phi <= 16) / 7*phi (phi > 16);
        # centerline bend radius = (mandrel + phi) / 2.
        (8, 20), (16, 40), (20, 80), (32, 128),
    ])
    def test_ec2_auto_bend_radius(self, diameter, expected_radius):
        bar = Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=diameter)
        assert bar.bend_radius == expected_radius

    def test_explicit_bend_radius_kept(self):
        bar = Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8, bend_radius=50)
        assert bar.bend_radius == 50

    def test_zero_bend_radius_rejected(self):
        with pytest.raises(ValidationError, match="bend_radius must be > 0"):
            Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8, bend_radius=0)

    def test_bar_type_values(self):
        for bar_type, ifc in BAR_TYPE_TO_IFC.items():
            bar = Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8,
                      bar_type=bar_type)
            assert bar.bar_type == bar_type
            assert ifc in ("MAIN", "SHEAR", "LIGATURE", "EDGE")

    def test_invalid_bar_type_rejected(self):
        with pytest.raises(ValidationError):
            Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8, bar_type="hoop")

    def test_mark_is_shared_not_identity(self):
        b1 = Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8, mark="S1")
        b2 = Bar(path=_pts((0, 150, 0), (0, 2150, 0)), diameter=8, mark="S1")
        proj = Project(name="marks")
        proj.add(b1)
        proj.add(b2)
        from lite_step.compiler.executor import validate_project_report
        report = validate_project_report(proj)
        assert report.errors == []  # shared mark is fine; names are None

    def test_explicit_material_not_overridden(self):
        bar = Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8,
                  material="Steel_B500B")
        assert bar.material == "Steel_B500B"


# ---------------------------------------------------------------------------
# Sweep — form discrimination
# ---------------------------------------------------------------------------

SEC = [Point2D(x=-45, y=-20), Point2D(x=45, y=-20),
       Point2D(x=45, y=20), Point2D(x=-45, y=20)]


class TestProfileForms:
    def test_legacy_form_still_works(self):
        # DSL v17: shape= removed — a bare legacy start/end Sweep is valid
        # (section falls back to the 89x38 stud default at generation).
        p = Sweep(start=Point(x=0, y=0, z=0),
                    end=Point(x=0, y=0, z=2400))
        assert p.path is None

    def test_legacy_material_section_form_still_works(self):
        p = Sweep(material=Material(key="Timber_C24", profile_mm=(45, 195)),
                    start=Point(x=0, y=0, z=2700), end=Point(x=0, y=4500, z=2700))
        assert p.path is None

    def test_path_form_with_explicit_section(self):
        p = Sweep(profile=SEC, path=_pts((0, 0, 0), (1000, 0, 0)))
        assert p.start is None and p.end is None

    def test_path_form_with_material_section(self):
        p = Sweep(material=Material(key="Timber_C24", profile_mm=(45, 195)),
                    path=_pts((0, 0, 0), (1000, 0, 0)))
        assert p.profile is None

    def test_both_forms_in_one_call_rejected(self):
        with pytest.raises(ValidationError, match="not both"):
            Sweep(start=Point(x=0, y=0, z=0),
                    end=Point(x=0, y=0, z=2400),
                    path=_pts((0, 0, 0), (1000, 0, 0)))

    def test_section_with_legacy_form_rejected(self):
        with pytest.raises(ValidationError, match="not both"):
            Sweep(start=Point(x=0, y=0, z=0),
                    end=Point(x=0, y=0, z=2400), profile=SEC)

    def test_neither_form_rejected(self):
        with pytest.raises(ValidationError, match="geometry source"):
            Sweep()

    def test_legacy_form_requires_both_endpoints(self):
        with pytest.raises(ValidationError, match="both start= and end="):
            Sweep(start=Point(x=0, y=0, z=0))

    def test_shape_kwarg_is_forbidden(self):
        # DSL v17: the legacy catalog shape= key was removed and is now a
        # forbidden extra field on Sweep.
        with pytest.raises(ValidationError):
            Sweep(shape="Stud_2x4", start=Point(x=0, y=0, z=0),
                    end=Point(x=0, y=0, z=2400))

    def test_wide_axis_with_path_form_rejected(self):
        with pytest.raises(ValidationError, match="wide_axis= is legacy-form"):
            Sweep(profile=SEC, path=_pts((0, 0, 0), (1000, 0, 0)),
                    wide_axis="y")

    def test_fillet_radius_with_legacy_form_rejected(self):
        with pytest.raises(ValidationError, match="path-form fields"):
            Sweep(start=Point(x=0, y=0, z=0),
                    end=Point(x=0, y=0, z=2400), fillet_radius=50)

    def test_section_needs_three_points(self):
        with pytest.raises(ValidationError, match="at least 3 points"):
            Sweep(profile=SEC[:2], path=_pts((0, 0, 0), (1000, 0, 0)))

    def test_closed_loop_accepted(self):
        loop = _pts((0, 0, 0), (1000, 0, 0), (1000, 0, 1000), (0, 0, 1000),
                    (0, 0, 0))
        p = Sweep(profile=SEC, path=loop)
        assert p.path[0].x == p.path[-1].x

    def test_closed_loop_needs_three_vertices(self):
        with pytest.raises(ValidationError, match="at least 4 points"):
            Sweep(profile=SEC, path=_pts((0, 0, 0), (1000, 0, 0), (0, 0, 0)))

    def test_closed_loop_with_fillet_rejected(self):
        loop = _pts((0, 0, 0), (1000, 0, 0), (1000, 0, 1000), (0, 0, 1000),
                    (0, 0, 0))
        with pytest.raises(ValidationError, match="closed-loop"):
            Sweep(profile=SEC, path=loop, fillet_radius=50)

    def test_one_source_rule_section_and_material_section(self):
        # Both profile= and Material(profile_mm=) — rejected at compile.
        p = Sweep(profile=SEC,
                    material=Material(key="Timber_C24", profile_mm=(45, 195)),
                    path=_pts((0, 0, 0), (1000, 0, 0)))
        proj = Project(name="one source")
        proj.add(p)
        from lite_step.compiler.executor import validate_project_report
        report = validate_project_report(proj)
        assert any("one-source rule" in e and "profile=" in e
                   for e in report.errors)

    def test_path_form_without_section_source_rejected_at_compile(self):
        p = Sweep(path=_pts((0, 0, 0), (1000, 0, 0)))
        proj = Project(name="no section")
        proj.add(p)
        from lite_step.compiler.executor import validate_project_report
        report = validate_project_report(proj)
        assert any("needs a profile source" in e for e in report.errors)


# ---------------------------------------------------------------------------
# Element (generic container)
# ---------------------------------------------------------------------------


class TestElementConstruction:
    def test_whitelisted_class(self):
        el = Element(ifc_class="IfcStair", name="stair_main")
        assert el.ifc_class == "IfcStair"

    def test_spec_examples_are_whitelisted(self):
        for cls in ("IfcStair", "IfcRailing", "IfcPlate", "IfcFooting"):
            assert cls in ELEMENT_IFC_CLASS_WHITELIST
            Element(ifc_class=cls)

    def test_unknown_class_rejected(self):
        with pytest.raises(ValidationError, match="unknown ifc_class"):
            Element(ifc_class="IfcSpaceship")

    def test_non_whitelisted_real_class_rejected(self):
        # Real IFC class, but not a whitelisted project element.
        with pytest.raises(ValidationError, match="unknown ifc_class"):
            Element(ifc_class="IfcWallStandardCase")

    def test_children_add_chain(self):
        el = Element(ifc_class="IfcRailing")
        result = el.add(Box(start=Point(x=0, y=0, z=0),
                              end=Point(x=100, y=100, z=1000)))
        assert result is el
        assert len(el.elements) == 1

    def test_empty_element_is_legal(self):
        el = Element(ifc_class="IfcStair", name="stair_main",
                     props={"Pset_StairCommon": {"NumberOfRiser": 16}})
        assert el.elements == []


# ---------------------------------------------------------------------------
# Strict int-mm interaction
# ---------------------------------------------------------------------------


class TestStrictIntMm:
    def test_rod_radius_float_rejected_when_strict(self):
        enable_strict_int_mm()
        with pytest.raises(ValidationError, match="int millimeters"):
            Pipe(path=_pts((0, 0, 0), (1000, 0, 0)), radius=25.0)

    def test_turn_angle_float_rejected_when_strict(self):
        enable_strict_int_mm()
        with pytest.raises(ValidationError, match="int millimeters"):
            Revolve(profile=ARCH_SECTION, path=ARCH_AXIS, angle=18000.0)

    def test_bar_diameter_float_rejected_when_strict(self):
        enable_strict_int_mm()
        with pytest.raises(ValidationError, match="int millimeters"):
            Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8.0)

    def test_profile_fillet_float_rejected_when_strict(self):
        enable_strict_int_mm()
        with pytest.raises(ValidationError, match="int millimeters"):
            Sweep(profile=SEC, path=_pts((0, 0, 0), (1000, 0, 0), (1000, 1000, 0)),
                    fillet_radius=50.0)

    def test_integral_ints_pass_when_strict(self):
        enable_strict_int_mm()
        Pipe(path=_pts((0, 0, 0), (1000, 0, 0)), radius=25)
        Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8)
        Revolve(profile=ARCH_SECTION, path=ARCH_AXIS, angle=18000)

    def test_floats_allowed_when_flag_off(self):
        rod = Pipe(path=_pts((0, 0, 0), (1000, 0, 0)), radius=25.0)
        assert rod.radius == 25


# ---------------------------------------------------------------------------
# BimElement machinery smoke: derivation, name/ifc_name, props, material
# ---------------------------------------------------------------------------


class TestUniversalMachinery:
    def test_rod_derivation(self):
        rod = Pipe(path=_pts((0, 0, 0), (1000, 0, 0)), radius=25, name="r1")
        rod2 = rod(name="r2", radius=30)
        assert rod2.radius == 30 and rod2.name == "r2"
        assert rod2.id != rod.id
        assert rod.radius == 25  # template untouched

    def test_turn_derivation(self):
        turn = Revolve(profile=ARCH_SECTION, path=ARCH_AXIS, angle=18000, name="t1")
        turn2 = turn(name="t2", angle=9000)
        assert turn2.angle == 9000 and turn.angle == 18000

    def test_bar_derivation_carries_grade(self):
        bar = Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8, mark="S1")
        bar2 = bar(path=_pts((0, 150, 0), (0, 2150, 0)))
        assert bar2.material == "Steel_B500B"
        assert bar2.mark == "S1"
        assert bar2.id != bar.id

    def test_element_derivation_unknown_field_raises(self):
        el = Element(ifc_class="IfcStair")
        with pytest.raises(TypeError, match="unknown field"):
            el(riser_count=16)

    def test_props_accepted_on_all_new_classes(self):
        props = {"Pset_Custom": {"Key": 1.5}}
        for elem in (
            Pipe(path=_pts((0, 0, 0), (1000, 0, 0)), radius=25, props=props),
            Revolve(profile=ARCH_SECTION, path=ARCH_AXIS, props=props),
            Bar(path=_pts((0, 0, 0), (0, 2000, 0)), diameter=8, props=props),
            Element(ifc_class="IfcStair", props=props),
            Sweep(profile=SEC, path=_pts((0, 0, 0), (1000, 0, 0)), props=props),
        ):
            assert elem.props == props

    def test_ifc_name_prefers_name(self):
        rod = Pipe(path=_pts((0, 0, 0), (1000, 0, 0)), radius=25, name="rail_x")
        # DSL v2.1: the canonical is derived from containment by the naming pass
        from lite_step.compiler.naming import stamp_canonical_names
        from lite_step.models.project import Project, Storey
        b = Project(name="t"); st = Storey(elevation=0)  # anonymous single storey
        st.add(rod); b.add_storey(st); stamp_canonical_names(b)
        assert rod.ifc_name == "pipe:rail_x"  # no storey segment

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            Pipe(path=_pts((0, 0, 0), (1000, 0, 0)), radius=25, diameter=8)


# ---------------------------------------------------------------------------
# Boolean-operand consumption
# ---------------------------------------------------------------------------


class TestConsumedOperands:
    def test_collect_consumed_operand_ids(self):
        wall_body = Box(start=Point(x=0, y=0, z=0),
                          end=Point(x=6000, y=300, z=2700))
        void = Box(start=Point(x=1000, y=-30, z=0),
                     end=Point(x=2000, y=330, z=2100))
        arch = Revolve(profile=ARCH_SECTION, path=ARCH_AXIS, angle=18000)
        void.union(arch)
        wall_body.difference(void)
        proj = Project(name="consume")
        proj.add(wall_body)
        consumed = collect_consumed_operand_ids(proj)
        assert id(void) in consumed
        assert id(arch) in consumed  # nested operand collected too
        assert id(wall_body) not in consumed

    def test_double_use_is_refused_at_compile(self):
        body = Box(start=Point(x=0, y=0, z=0),
                     end=Point(x=6000, y=300, z=2700), name="body")
        void = Box(start=Point(x=1000, y=-30, z=0),
                     end=Point(x=2000, y=330, z=2100), name="void_x")
        body.difference(void)
        proj = Project(name="double use")
        proj.add(body)
        proj.add(void)  # ALSO added standalone — must warn
        from lite_step.compiler.executor import validate_project_report
        report = validate_project_report(proj)
        # An operand that is ALSO added standalone is one object on two paths,
        # so the v10 shared-element refusal blocks it. The two messages are
        # complementary rather than duplicate: the error names the two paths,
        # the warning names the fix (drop the proj.add()).
        assert any("attached in 2 places" in e for e in report.errors), report.errors
        assert any("consumed as a boolean operand" in w for w in report.warnings)
