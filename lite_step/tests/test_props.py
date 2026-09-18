"""Universal element-level ``props={}`` (DSL v1.5 §SIGNATURES, WS1 PR-D).

``props={"Pset_X": {"Key": value}}`` → one IfcPropertySet per top-level key,
attached to the element's IFC product via IfcRelDefinesByProperties.

Pinned behavior:

1.  Universal field, default empty, per-instance dict.
2.  Floats are LEGAL inside props under strict int-mm (the gate covers
    geometry only) — the strict-mode test PR-C deferred.
3.  Generator emission (ifcopenshell backend): entity-level assertions in
    the PR-A style; consumed geometry (Wall body Solid) attaches to the
    nearest ancestor product; Window/Door attach to their own product.
4.  Invalid shapes are compile ERRORS; props on boolean-operand geometry is
    a compile WARNING (would be silently dropped otherwise).
5.  Streaming gate: props route to the ifcopenshell backend (registry-
    materials precedent) until streaming parity lands.
"""

from __future__ import annotations

import re

import pytest

from lite_step import disable_strict_int_mm, enable_strict_int_mm
from lite_step.compiler.executor import (
    compile_main,
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.models import Project, Point, Slab, Box, Extrude, Wall, Window

pytest.importorskip("ifcopenshell")


@pytest.fixture(autouse=True)
def _reset_strict_flag():
    yield
    enable_strict_int_mm()  # post-cutover default


def _box(name=None, **kwargs) -> "BimElement":
    return Box(
        start=Point(x=-6000, y=-4500, z=0),
        end=Point(x=6000, y=-4200, z=2700),
        name=name,
        **kwargs,
    )


def _building(*elements) -> Project:
    b = Project(name="props-test")
    for e in elements:
        b.add(e)
    return b


def _generate(project: Project) -> str:
    from lite_step.ifc.generator import generate_ifc

    normalized = normalize_project_to_meters(project)
    result = generate_ifc(normalized)
    assert result.success, result.error
    return result.ifc_content


# ---------------------------------------------------------------------------
# 1. Model surface
# ---------------------------------------------------------------------------


class TestPropsField:
    def test_default_is_empty_dict(self):
        assert _box().props == {}

    def test_default_dicts_are_per_instance(self):
        a, b = _box(), _box()
        a.props["Pset_X"] = {"K": 1}
        assert b.props == {}

    def test_props_on_every_element_kind(self):
        p = {"Pset_Common": {"Reference": "R1"}}
        assert _box(props=p).props == p
        assert Wall(name="w", props=p).props == p
        assert Window(width=1200, height=1400, props=p).props == p
        assert Slab(props=p).props == p

    def test_floats_legal_in_props_under_strict(self):
        # The strict int-mm gate is a geometry gate. Floats inside props are
        # spec-legal (RULE 2: "Floats are legal only inside props={}").
        enable_strict_int_mm()
        elem = Box(
            start=Point(x=0, y=0, z=0),
            end=Point(x=1200, y=100, z=1400),
            props={"Pset_WindowCommon": {"ThermalTransmittance": 0.9}},
        )
        assert elem.props["Pset_WindowCommon"]["ThermalTransmittance"] == 0.9

    def test_floats_in_props_survive_compile_under_strict(self, tmp_path):
        enable_strict_int_mm()
        b = _building(_box(name="s", props={"Pset_X": {"U": 0.9}}))
        src = tmp_path / "model.py"
        src.write_text("# strict props fixture\n", encoding="utf-8")
        out = compile_main(result=b, source_path=str(src))
        assert out is not None and out.is_file()
        assert "IFCPROPERTYSET" in out.read_text()


# ---------------------------------------------------------------------------
# 2. Shape validation (compile errors / warnings)
# ---------------------------------------------------------------------------


class TestPropsShapeValidation:
    def test_well_formed_props_no_errors(self):
        b = _building(_box(props={"Pset_X": {"S": "v", "I": 1, "F": 0.5,
                                             "B": True}}))
        report = validate_project_report(b)
        assert [e for e in report.errors if "props" in e] == []

    def test_non_dict_top_level_value_is_error(self):
        b = _building(_box(name="s1", props={"FireRating": "R30"}))
        report = validate_project_report(b)
        errs = [e for e in report.errors if "props" in e]
        assert len(errs) == 1
        assert "'FireRating'" in errs[0]
        assert "Pset" in errs[0]  # names the expected shape

    def test_non_scalar_property_value_is_error(self):
        b = _building(_box(name="s1", props={"Pset_X": {"K": [1, 2]}}))
        report = validate_project_report(b)
        errs = [e for e in report.errors if "props" in e]
        assert len(errs) == 1
        assert "'K'" in errs[0] and "scalar" in errs[0]

    def test_nested_dict_value_is_error(self):
        b = _building(_box(props={"Pset_X": {"K": {"nested": 1}}}))
        report = validate_project_report(b)
        assert any("scalar" in e for e in report.errors)

    def test_shape_errors_found_on_container_children(self):
        wall = Wall(name="w")
        wall.add(_box(props={"Bad": "not-a-dict"}))
        report = validate_project_report(_building(wall))
        assert any("props" in e and "'Bad'" in e for e in report.errors)

    def test_props_on_boolean_operand_warns(self):
        void = Box(start=Point(x=0, y=0, z=0),
                     end=Point(x=900, y=400, z=2100),
                     props={"Pset_X": {"K": 1}})
        host = _box(name="host").difference(void)
        report = validate_project_report(_building(host))
        warns = [w for w in report.warnings if "boolean operand" in w]
        assert len(warns) == 1
        assert "never" in warns[0] and "emitted" in warns[0]

    def test_malformed_props_on_boolean_operand_still_errors(self):
        void = Box(start=Point(x=0, y=0, z=0),
                     end=Point(x=900, y=400, z=2100),
                     props={"Bad": "not-a-dict"})
        host = _box(name="host").difference(void)
        report = validate_project_report(_building(host))
        assert any("props" in e and "'Bad'" in e for e in report.errors)

    def test_empty_props_is_silent(self):
        report = validate_project_report(_building(_box()))
        assert [e for e in report.errors if "props" in e] == []
        assert [w for w in report.warnings if "props" in w] == []


# ---------------------------------------------------------------------------
# 3. Generator emission (ifcopenshell backend) — PR-A-style entity assertions
# ---------------------------------------------------------------------------


def _element_psets(content: str) -> list[str]:
    """The IFCPROPERTYSET lines that came from ``props=``.

    Every model also carries 2D drawing wiring — ``BBIM_Documentation`` on
    IfcProject and one ``EPset_Drawing`` per plan — which is unrelated to
    element props. A bare ``"IFCPROPERTYSET" not in content`` would sweep
    those up and assert something this module does not own.
    """
    return [
        line for line in content.splitlines()
        if "IFCPROPERTYSET(" in line
        and "BBIM_Documentation" not in line and "EPset_Drawing" not in line
    ]


def _drawing_pset_count(content: str) -> int:
    """How many psets belong to the 2D drawing wiring, not to ``props=``.

    Each one carries exactly one IFCRELDEFINESBYPROPERTIES, so this is also
    the baseline rel count for a model with no element props.
    """
    return sum(
        1 for line in content.splitlines()
        if "IFCPROPERTYSET(" in line
        and ("BBIM_Documentation" in line or "EPset_Drawing" in line)
    )


class TestGeneratorEmission:
    def test_props_emit_pset_and_rel(self):
        content = _generate(_building(_box(
            name="foundation", type="sketch", color="foundation",
            props={"Pset_ConcreteElementGeneral": {"ExposureClass": "XC2"}})))
        assert "IFCPROPERTYSET" in content
        assert "'Pset_ConcreteElementGeneral'" in content
        assert "'ExposureClass'" in content
        assert "IFCLABEL('XC2')" in content
        assert "IFCRELDEFINESBYPROPERTIES" in content

    def test_one_pset_per_top_level_key(self):
        content = _generate(_building(_box(name="s", props={
            "Pset_A": {"K1": 1},
            "Pset_B": {"K2": 2},
        })))
        assert len(_element_psets(content)) == 2, _element_psets(content)
        assert "'Pset_A'" in content and "'Pset_B'" in content

    def test_scalar_types_wrap_to_matching_ifc_types(self):
        content = _generate(_building(_box(name="s", props={
            "Pset_Mix": {"B": True, "I": 3, "F": 0.9, "S": "label"},
        })))
        assert "IFCBOOLEAN(.T.)" in content
        assert "IFCINTEGER(3)" in content
        assert "IFCREAL(0.9" in content  # float serialization may pad digits
        assert "IFCLABEL('label')" in content

    def test_no_props_no_pset_entities(self):
        # Force the ifcopenshell backend (props-free buildings normally
        # stream) to prove the post-pass is a strict no-op without props.
        from lite_step.ifc.generator import generate_ifc

        normalized = normalize_project_to_meters(_building(_box(name="s")))
        result = generate_ifc(normalized)
        assert result.success, result.error
        assert _element_psets(result.ifc_content) == []
        assert (result.ifc_content.count("IFCRELDEFINESBYPROPERTIES(")
                == _drawing_pset_count(result.ifc_content))

    def test_wall_body_props_attach_to_wall_product(self):
        """Consumed geometry (Wall body Solid) → nearest ancestor product,
        the same attribution PR-A uses for the body's registry material."""
        wall = Wall(name="wall_w")
        wall.add(Box(start=Point(x=-4500, y=-4500, z=0),
                       end=Point(x=-4200, y=4500, z=2700),
                       props={"Pset_WallCommon": {"FireRating": "REI60"}}))
        content = _generate(_building(wall))
        wall_id = re.search(r"#(\d+)=IFCWALL\(", content).group(1)
        rel = re.search(
            r"IFCRELDEFINESBYPROPERTIES\('[^']*',[^,]*,[^,]*,[^,]*,"
            r"\(#(\d+)\),#(\d+)\)", content)
        assert rel is not None, "no IfcRelDefinesByProperties emitted"
        assert rel.group(1) == wall_id
        pset_line = re.search(
            rf"#{rel.group(2)}=IFCPROPERTYSET\('[^']*',[^,]*,'([^']*)'",
            content)
        assert pset_line.group(1) == "Pset_WallCommon"

    def test_window_props_attach_to_window_product(self):
        wall = Wall(name="wall_w")
        wall.add(Box(start=Point(x=-4500, y=-4500, z=0),
                       end=Point(x=-4200, y=4500, z=2700)))
        wall.anchor(Window(width=1200, height=1400, name="win_w0",
                           props={"Pset_WindowCommon":
                                  {"ThermalTransmittance": 0.9}}),
                    along=1500, up=900)
        content = _generate(_building(wall))
        win_id = re.search(r"#(\d+)=IFCWINDOW\(", content).group(1)
        rels = re.findall(
            r"IFCRELDEFINESBYPROPERTIES\('[^']*',[^,]*,[^,]*,[^,]*,"
            r"\(#(\d+)\),#\d+\)", content)
        assert win_id in rels

    def test_empty_pset_dict_emits_nothing(self):
        content = _generate(_building(_box(name="s",
                                           props={"Pset_Empty": {}})))
        assert _element_psets(content) == []

    def test_compile_main_end_to_end(self, tmp_path):
        b = _building(_box(name="foundation",
                           props={"Pset_ConcreteElementGeneral":
                                  {"ExposureClass": "XC2"}}))
        src = tmp_path / "model.py"
        src.write_text("# props e2e fixture\n", encoding="utf-8")
        out = compile_main(result=b, source_path=str(src))
        content = out.read_text()
        assert "'Pset_ConcreteElementGeneral'" in content
        assert "IFCLABEL('XC2')" in content
