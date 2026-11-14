"""IFC 4.3 express (where-) rules hold on the compiler's output.

``ifcopenshell.validate`` checks schema TYPES by default; the where-rules only
run with ``express_rules=True``. Without that flag three rule families were
invisible and present on most real models (#15, #16):

* ``IfcBooleanClippingResult.FirstOperandType`` — a ``.clip()`` wrapped a
  boolean tree or a faceted brep, which a clipping result may not take;
* ``IfcBuildingElementProxy.HasObjectName`` — anonymous solids (a footing's
  body, a covering's panel) emitted with ``Name = $``;
* ``IfcGridAxis.WR1`` — 3D grid axis curves;

plus ``IfcShapeRepresentationTypes`` on a brep Revolve labelled
``SweptSolid``. The gate below compiles every example at budget 0, and is
first shown to fail on a planted violation.
"""
from __future__ import annotations

import os
import runpy
import tempfile
from pathlib import Path

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.validate  # noqa: E402,F811

from lite_step.compiler.executor import normalize_project_to_meters  # noqa: E402
from lite_step.ifc.generator import generate_ifc  # noqa: E402
from lite_step.models import (  # noqa: E402
    Box, Element, GuideLine, Point, Point2D, Project, Revolve,
)
from lite_step.models.project import Storey  # noqa: E402

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
#: Clean when measured (0 issues each, 2026-11-14) but 8-40 s apiece to
#: validate: mesh- and brep-heavy models whose emitters the remaining examples
#: also reach. Kept out so the gate stays cheap enough to run on every change.
_SLOW = {"05_log_cabin.py", "10_mary_elizabeth_hospital.py",
         "12_stellarator.py", "13_stellarator.py", "14_stellarator.py"}


def _open(ifc_text: str):
    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(ifc_text)
    try:
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _issues(model) -> list:
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(model, logger, express_rules=True)
    return [str(s.get("message", "")).splitlines()[0][:120] for s in logger.statements]


def _compile(*elements, project=None):
    if project is None:
        project = Project(name="rules")
        storey = Storey(name="ground")
        for e in elements:
            storey.add(e)
        project.add_storey(storey)
    res = generate_ifc(normalize_project_to_meters(project), source_code="x = 1")
    assert res.success, res.error
    return res


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_gate_sees_a_planted_violation():
    """The control: a gate that cannot fail proves nothing. Strip one proxy's
    Name and the validator must report HasObjectName."""
    model = _open(_compile(Box(name="b", start=Point(x=0, y=0, z=0),
                               end=Point(x=1000, y=1000, z=1000))).ifc_content)
    assert _issues(model) == []
    model.by_type("IfcBuildingElementProxy")[0].Name = None
    assert any("Name" in i for i in _issues(model)), _issues(model)


@pytest.mark.parametrize("example", sorted(
    p.name for p in _EXAMPLES.glob("*.py") if p.name not in _SLOW))
def test_every_example_meets_the_express_rules(example, monkeypatch):
    from lite_step.materials import dk

    monkeypatch.setenv("LITESTEP_TREE_REPORT", "0")
    monkeypatch.setenv("LITESTEP_CARVE_REPORT", "0")
    # An example may register its own materials (09_steel_bridge does) into
    # the process-wide registry; restore it so no later test sees them.
    saved = dict(dk.MATERIALS)
    try:
        ns = runpy.run_path(str(_EXAMPLES / example), run_name="lite_step_rules_gate")
        issues = _issues(_open(_compile(project=ns["result"]).ifc_content))
    finally:
        dk.MATERIALS.clear()
        dk.MATERIALS.update(saved)
    assert issues == [], f"{example}: {len(issues)} where-rule issue(s): {issues[:5]}"


# ---------------------------------------------------------------------------
# One test per fix
# ---------------------------------------------------------------------------


def _clip_on_a_boolean_tree():
    host = Box(name="host", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=1000, z=1000))
    host.difference(Box(start=Point(x=500, y=-100, z=500), end=Point(x=900, y=1100, z=1100)))
    host.clip(origin=Point(x=1500, y=0, z=0), normal=(1, 0, 0))
    return host


def test_a_clip_on_a_boolean_tree_is_a_boolean_result_not_a_clipping_result():
    model = _open(_compile(_clip_on_a_boolean_tree()).ifc_content)
    for clip in model.by_type("IfcBooleanClippingResult"):
        assert clip.FirstOperand.is_a() in (
            "IfcExtrudedAreaSolid", "IfcBooleanClippingResult") or \
            clip.FirstOperand.is_a("IfcSweptAreaSolid")
    halves = [b for b in model.by_type("IfcBooleanResult")
              if b.SecondOperand.is_a("IfcHalfSpaceSolid")]
    assert halves, "the clip must still be emitted"
    assert _issues(model) == []


def test_a_clip_on_a_swept_solid_stays_a_clipping_result():
    """The falsification partner: the swap is only for operands the clipping
    result cannot take."""
    b = Box(name="b", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=1000, z=1000))
    b.clip(origin=Point(x=1500, y=0, z=0), normal=(1, 0, 0))
    model = _open(_compile(b).ifc_content)
    assert len(model.by_type("IfcBooleanClippingResult")) == 1


def test_anonymous_bodies_are_named_after_their_owner_and_stay_out_of_the_manifest():
    footing = Element(ifc_class="IfcFooting", name="strip")
    footing.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=400, z=300)))
    footing.add(Box(start=Point(x=0, y=600, z=0), end=Point(x=1000, y=1000, z=300)))
    res = _compile(footing)
    model = _open(res.ifc_content)
    names = sorted(p.Name for p in model.by_type("IfcBuildingElementProxy"))
    assert names == ["proxy#1:footing:strip:storey:ground",
                     "proxy#2:footing:strip:storey:ground"]
    assert "proxy#" not in _manifest(model)
    assert _issues(model) == []


def _manifest(model) -> str:
    for p in model.by_type("IfcPropertySingleValue"):
        if p.Name == "MANIFEST":
            return p.NominalValue.wrappedValue
    raise AssertionError("no MANIFEST")


def _grid(z0=0, z1=0):
    p = Project(name="grid")
    p.add(GuideLine(points=[Point(x=0, y=0, z=z0), Point(x=10000, y=0, z=z1)],
                    line_type="grid", label="A"))
    p.add(GuideLine(points=[Point(x=0, y=0, z=z0), Point(x=0, y=8000, z=z0)],
                    line_type="grid", label="1"))
    return p


def test_grid_axis_curves_are_2d_and_the_grid_keeps_its_height():
    model = _open(_compile(project=_grid(z0=3000, z1=3000)).ifc_content)
    grid = model.by_type("IfcGrid")[0]
    for axis in list(grid.UAxes) + list(grid.VAxes):
        assert axis.AxisCurve.Dim == 2
    loc = grid.ObjectPlacement.RelativePlacement.Location.Coordinates
    assert loc[2] == pytest.approx(3.0)
    assert _issues(model) == []


def test_a_sloping_grid_axis_is_projected_loudly(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _compile(project=_grid(z0=0, z1=3000))
    assert any("projected onto the grid plane" in r.getMessage() and "A" in r.getMessage()
               for r in caplog.records)


def test_a_brep_revolve_is_labelled_brep():
    r = Revolve(name="r", angle=36000,
                profile=[Point2D(x=100, y=0), Point2D(x=400, y=0), Point2D(x=100, y=300)],
                path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=500)])
    model = _open(_compile(r).ifc_content)
    reps = [rep for rep in model.by_type("IfcShapeRepresentation")
            if rep.Items and rep.Items[0].is_a("IfcFacetedBrep")]
    assert reps and all(rep.RepresentationType == "Brep" for rep in reps)
    assert _issues(model) == []


def test_every_rectangle_profile_has_a_position():
    """#14: web-ifc logs an error per IfcRectangleProfileDef without one."""
    model = _open(_compile(_clip_on_a_boolean_tree()).ifc_content)
    profiles = model.by_type("IfcRectangleProfileDef")
    assert profiles and all(p.Position is not None for p in profiles)
