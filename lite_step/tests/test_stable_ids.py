"""GlobalIds are the same every time the same model is compiled (lite-step#9).

Other programs keep hold of an element by its GlobalId: a Lapin link reloads by
it, a BCF topic points at it. What is pinned here is what they see: the same
model compiles to the same ids, an edit to one element leaves every other
element's id alone, and two models never share one.
"""
from __future__ import annotations

import json

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.compiler.executor import normalize_project_to_meters  # noqa: E402
from lite_step.ifc.embedder import LITESTEP_META_PSET_NAME  # noqa: E402
from lite_step.ifc.generator import generate_ifc  # noqa: E402
from lite_step.ifc.stable_ids import stabilise_global_ids  # noqa: E402
from lite_step.models import Box, Point, Project, Wall, Window  # noqa: E402
from lite_step.models.project import Storey  # noqa: E402

P = Point


def _model(name="house", post_height=2000, with_shed=False):
    proj = Project(name=name)
    ground = Storey(name="ground")
    wall = Wall(name="north")
    wall.add(Box(name="body", start=P(x=0, y=0, z=0), end=P(x=4000, y=200, z=2400)))
    wall.anchor(Window(width=1200, height=1400, name="w1"), along=1500, up=900)
    ground.add(wall)
    ground.add(Box(name="post", start=P(x=5000, y=0, z=0), end=P(x=5200, y=200, z=post_height)))
    if with_shed:
        ground.add(Box(name="shed", start=P(x=9000, y=0, z=0), end=P(x=11000, y=2000, z=2000)))
    proj.add_storey(ground)
    return proj


def _compile(proj, source=None):
    res = generate_ifc(normalize_project_to_meters(proj), source_code=source)
    assert res.success, res.error
    return ifcopenshell.file.from_string(res.ifc_content)


def _ids(model, cls="IfcRoot"):
    return {(e.is_a(), e.Name): e.GlobalId for e in model.by_type(cls) if e.Name}


def test_the_same_model_compiles_to_the_same_ids_everywhere():
    a, b = _compile(_model()), _compile(_model())
    ga = sorted(e.GlobalId for e in a.by_type("IfcRoot"))
    gb = sorted(e.GlobalId for e in b.by_type("IfcRoot"))
    assert ga == gb, "every IfcRoot, relationships and property sets included"
    assert len(set(ga)) == len(ga), "and no two the same"


def test_an_edited_element_keeps_its_id_and_so_does_everything_else():
    before = _ids(_compile(_model(post_height=2000)), "IfcProduct")
    after = _ids(_compile(_model(post_height=3000, with_shed=True)), "IfcProduct")
    assert set(before) <= set(after)
    assert {k: after[k] for k in before} == before, "the taller post and every other element keep their ids"
    assert ("IfcBuildingElementProxy", "box:shed:storey:ground") in after


def test_two_models_never_share_an_id():
    """Two models linked into one Lapin project may have the same element names."""
    a = {e.GlobalId for e in _compile(_model(name="house")).by_type("IfcRoot")}
    b = {e.GlobalId for e in _compile(_model(name="carport")).by_type("IfcRoot")}
    assert a.isdisjoint(b)


def test_the_manifest_names_the_ids_the_file_has():
    model = _compile(_model(), source="result = None\n")
    pset = next(p for p in model.by_type("IfcPropertySet") if p.Name == LITESTEP_META_PSET_NAME)
    manifest = json.loads(next(p for p in pset.HasProperties if p.Name == "MANIFEST").NominalValue.wrappedValue)
    assert manifest, "the manifest lists the named products"
    for name, gid in manifest.items():
        assert model.by_guid(gid).Name == name


def test_entities_that_would_share_an_id_get_their_own():
    model = _compile(_model())
    post = next(e for e in model.by_type("IfcProduct") if e.Name == "box:post:storey:ground")
    twin = model.create_entity(post.is_a(), GlobalId=ifcopenshell.guid.new(), Name=post.Name,
                               ObjectPlacement=post.ObjectPlacement)
    stabilise_global_ids(model)
    assert twin.GlobalId != post.GlobalId
    ids = [e.GlobalId for e in model.by_type("IfcRoot")]
    assert len(set(ids)) == len(ids)


def test_ids_are_valid_ifc_globalids():
    for e in _compile(_model()).by_type("IfcRoot"):
        assert len(e.GlobalId) == 22
        ifcopenshell.guid.expand(e.GlobalId)  # raises on a malformed id
