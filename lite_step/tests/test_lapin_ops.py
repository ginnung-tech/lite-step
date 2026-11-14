"""A compiled model as Lapin operations (``lite_step.lapin.ops``).

What is pinned here is the contract a Lapin user sees: a plain box arrives as
an editable box of the right size in the right place, a carve arrives as a
subtract between two elements, and whatever cannot be said that way arrives
as a body with the carve already in it. Never a wrong shape.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.compiler.executor import normalize_project_to_meters  # noqa: E402
from lite_step.ifc.generator import generate_ifc  # noqa: E402
from lite_step.lapin import ops as lapin_ops  # noqa: E402
from lite_step.lapin import ops_from_ifc, ops_from_json  # noqa: E402
from lite_step.models import Box, Extrude, Point, Project  # noqa: E402
from lite_step.models.project import Storey  # noqa: E402

P = Point


def _ops(*solids, storey="ground", elevation=None):
    proj = Project(name="lapin")
    s = Storey(name=storey) if elevation is None else Storey(name=storey, elevation=elevation)
    for solid in solids:
        s.add(solid)
    proj.add_storey(s)
    res = generate_ifc(normalize_project_to_meters(proj), source_code=None)
    assert res.success, res.error
    model = ifcopenshell.file.from_string(res.ifc_content)
    return ops_from_ifc(model), model


def _by_leaf(result):
    return {o.path.split(":")[1]: o for o in result.ops}


def _volume(args):
    v = np.array(args["vertices"])
    return abs(sum(np.dot(v[a], np.cross(v[b], v[c])) for a, b, c in args["faces"]) / 6.0)


def _slab():
    return Box(name="slab", start=P(x=0, y=0, z=0), end=P(x=2000, y=1000, z=300))


def _post():
    return Box(name="post", start=P(x=900, y=400, z=-200), end=P(x=1100, y=600, z=2000))


def _l_post():
    """A post no primitive can say: an L in plan, cutting the slab like ``_post``."""
    return Extrude(name="lpost", thickness=2200, contour=[
        P(x=900, y=400, z=-200), P(x=1100, y=400, z=-200), P(x=1100, y=500, z=-200),
        P(x=1000, y=500, z=-200), P(x=1000, y=600, z=-200), P(x=900, y=600, z=-200)])


# --------------------------------------------------------------------------- primitives


def test_a_box_is_a_primitive_at_the_middle_of_its_base_in_metres():
    result, _ = _ops(Box(name="a", start=P(x=1000, y=2000, z=500), end=P(x=3000, y=2500, z=3500)))
    (op,) = result.ops
    assert (op.kind, op.ifc_class, op.tools) == ("primitive", "IfcBuildingElementProxy", ())
    assert op.path == "box:a:storey:ground"
    assert op.args == {"shape": "box", "at": [2.0, 2.25, 0.5], "rotation": 0.0,
                       "width": 2.0, "length": 0.5, "height": 3.0}
    assert result.summary() == {"elements": 1, "primitive": 1, "primitive_carved": 0, "body": 0, "skipped": 0}


def test_the_storey_is_named_with_its_elevation_in_metres():
    result, _ = _ops(_slab(), storey="loft", elevation=3200)
    assert result.storeys == [("storey:loft", 3.2)], "the name is the canonical one the IFC carries"
    assert (result.ops[0].storey, result.ops[0].elevation) == ("storey:loft", 3.2)


def test_an_inferred_carve_is_a_subtract_between_two_primitives():
    result, _ = _ops(_slab(), _post())
    ops = _by_leaf(result)
    assert ops["slab"].kind == "primitive" and ops["post"].kind == "primitive"
    assert ops["slab"].tools == ("box:post:storey:ground",), "the slab loses to the post added after it"
    assert ops["post"].tools == (), "the cutter stays whole, as it does in the IFC"
    # The slab is sent uncarved: Lapin does the cut, live.
    assert ops["slab"].args["width"] == 2.0 and ops["slab"].args["height"] == 0.3
    assert result.summary()["primitive_carved"] == 1


def test_a_cutter_that_is_no_primitive_sends_the_host_as_a_carved_body():
    result, _ = _ops(_slab(), _l_post())
    ops = _by_leaf(result)
    assert ops["lpost"].kind == "body", "an L in plan is not a box"
    assert ops["slab"].kind == "body" and ops["slab"].tools == ()
    whole = 2.0 * 1.0 * 0.3
    hole = (0.2 * 0.1 + 0.1 * 0.1) * 0.3
    assert _volume(ops["slab"].args) == pytest.approx(whole - hole, abs=1e-6), "the carve is in the mesh"
    assert any("box:slab:storey:ground: sent as a body because a cutter" in n for n in result.notes)


def test_a_cutter_that_is_itself_carved_is_not_used_as_a_tool():
    """Lapin would subtract the tool as it stands, pin-hole and all; the IFC subtracts it whole."""
    pin = Box(name="pin", start=P(x=950, y=450, z=1000), end=P(x=1050, y=550, z=1500))
    result, _ = _ops(_slab(), _post(), pin)
    ops = _by_leaf(result)
    assert ops["post"].tools == ("box:pin:storey:ground",)
    assert ops["slab"].kind == "body" and ops["slab"].tools == ()
    assert _volume(ops["slab"].args) == pytest.approx(0.6 - 0.2 * 0.2 * 0.3, abs=1e-6)


def test_an_element_with_a_void_is_a_body_with_the_hole_in_it():
    slab = _slab()
    slab.void(Box(start=P(x=500, y=250, z=-100), end=P(x=700, y=450, z=400)))
    result, _ = _ops(slab)
    (op,) = result.ops
    assert op.kind == "body", "an opening is not an element Lapin could subtract"
    assert _volume(op.args) == pytest.approx(0.6 - 0.2 * 0.2 * 0.3, abs=1e-6)


def test_a_body_is_a_closed_mesh_of_the_right_size():
    result, _ = _ops(_l_post())
    (op,) = result.ops
    assert op.kind == "body"
    assert _volume(op.args) == pytest.approx(0.03 * 2.2, abs=1e-6)
    v = np.array(op.args["vertices"])
    assert v.min(0).tolist() == [0.9, 0.4, -0.2] and v.max(0).tolist() == [1.1, 0.6, 2.0]


# --------------------------------------------------------------------------- identity


def test_hashes_are_the_same_for_two_compiles_of_the_same_model():
    """GlobalIds are random per compile; the hash must not see them."""
    first, m1 = _ops(_slab(), _post(), _l_post())
    second, m2 = _ops(_slab(), _post(), _l_post())
    assert {p.GlobalId for p in m1.by_type("IfcElement")}.isdisjoint(p.GlobalId for p in m2.by_type("IfcElement"))
    assert [(o.path, o.hash) for o in first.ops] == [(o.path, o.hash) for o in second.ops]


def test_a_changed_element_changes_its_hash_and_no_other():
    before, _ = _ops(_slab(), Box(name="far", start=P(x=5000, y=0, z=0), end=P(x=6000, y=1000, z=1000)))
    after, _ = _ops(_slab(), Box(name="far", start=P(x=5000, y=0, z=0), end=P(x=6000, y=1000, z=1001)))
    b, a = _by_leaf(before), _by_leaf(after)
    assert b["slab"].hash == a["slab"].hash
    assert b["far"].hash != a["far"].hash


def test_an_unnamed_element_and_a_repeated_name_are_still_tracked_and_said(monkeypatch):
    result, model = _ops(_slab(), _post())
    slab, post = (p for p in model.by_type("IfcElement"))
    slab.Name = None
    post.Name = "twin"
    extra = model.create_entity("IfcBuildingElementProxy", GlobalId=ifcopenshell.guid.new(), Name="twin",
                                ObjectPlacement=post.ObjectPlacement, Representation=post.Representation)
    assert extra.Name == "twin"
    result = ops_from_ifc(model)
    assert sorted(o.path for o in result.ops) == ["twin", "twin#2", "unnamed#1:IfcBuildingElementProxy"]
    assert any("has no name" in n for n in result.notes)
    assert any("2 elements share this name" in n for n in result.notes)


# --------------------------------------------------------------------------- the box itself


def _edges(angle, sizes=(4.0, 1.0, 3.0)):
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return [np.array([c, s, 0.0]) * sizes[0], np.array([-s, c, 0.0]) * sizes[1], np.array([0.0, 0.0, sizes[2]])]


@pytest.mark.parametrize("angle, rotation, width, length", [
    (0, 0.0, 4.0, 1.0),
    (30, 30.0, 4.0, 1.0),
    (120, 30.0, 1.0, 4.0),   # the same box, said from its other side
    (210, 30.0, 4.0, 1.0),
    (90, 0.0, 1.0, 4.0),
    (-45, 45.0, 1.0, 4.0),
])
def test_a_turned_box_has_one_way_of_being_said(angle, rotation, width, length):
    solid = lapin_ops._box(np.array([10.0, 20.0, 5.0]), _edges(angle))
    assert solid.rotation == pytest.approx(rotation, abs=1e-9)
    assert solid.sizes == pytest.approx((width, length, 3.0))
    assert solid.at[2] == pytest.approx(5.0)


def test_the_extrusion_axis_does_not_matter_only_that_one_edge_is_vertical():
    """A beam is a rectangle pushed along X: still a box standing on Z."""
    lying = [np.array([0.0, 0.0, 0.3]), np.array([0.0, 0.2, 0.0]), np.array([5.0, 0.0, 0.0])]
    solid = lapin_ops._box(np.zeros(3), lying)
    assert solid.sizes == pytest.approx((5.0, 0.2, 0.3)) and solid.rotation == 0.0
    assert solid.at == pytest.approx((2.5, 0.1, 0.0))


def test_a_box_leaning_off_z_or_sheared_is_not_a_primitive():
    tilt = np.radians(10)
    leaning = [np.array([1.0, 0, 0]), np.array([0, np.cos(tilt), np.sin(tilt)]), np.array([0, -np.sin(tilt), np.cos(tilt)])]
    assert lapin_ops._box(np.zeros(3), leaning) is None
    sheared = [np.array([1.0, 0, 0]), np.array([0.2, 1.0, 0]), np.array([0, 0, 1.0])]
    assert lapin_ops._box(np.zeros(3), sheared) is None


def test_two_solids_match_within_a_millimetre_and_not_beyond():
    a = lapin_ops._Solid("box", (1.0, 2.0, 0.0), (4.0, 1.0, 3.0), 30.0)
    assert a.same_as(lapin_ops._Solid("box", (1.0004, 2.0, 0.0), (4.0, 1.0004, 3.0), 30.0))
    assert not a.same_as(lapin_ops._Solid("box", (1.002, 2.0, 0.0), (4.0, 1.0, 3.0), 30.0))
    assert not a.same_as(lapin_ops._Solid("box", (1.0, 2.0, 0.0), (4.0, 1.0, 3.0), 30.1)), "0.1° over 4 m is 7 mm"
    assert not a.same_as(lapin_ops._Solid("cylinder", (1.0, 2.0, 0.0), (4.0, 1.0), 0.0))


# --------------------------------------------------------------------------- the JSON


def test_the_json_round_trips_and_says_what_it_is():
    result, _ = _ops(_slab(), _post(), _l_post())
    text = result.to_json()
    data = json.loads(text)
    assert (data["format"], data["version"]) == ("lite-step.lapin-ops", 1)
    back = ops_from_json(text)
    assert [o.to_dict() for o in back.ops] == [o.to_dict() for o in result.ops]
    assert (back.storeys, back.skipped, back.notes) == (result.storeys, result.skipped, result.notes)


def test_another_format_a_newer_version_and_an_edited_op_are_refused():
    result, _ = _ops(_slab())
    data = json.loads(result.to_json())
    with pytest.raises(ValueError, match="not a lite-step.lapin-ops document"):
        ops_from_json(json.dumps({**data, "format": "something-else"}))
    with pytest.raises(ValueError, match="version 2 is not version 1"):
        ops_from_json(json.dumps({**data, "version": 2}))
    data["ops"][0]["args"]["width"] = 9.0
    with pytest.raises(ValueError, match="the hash does not match"):
        ops_from_json(json.dumps(data))
