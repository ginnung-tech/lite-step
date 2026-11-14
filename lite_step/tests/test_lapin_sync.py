"""The delta a push applies to Lapin (``tools/lapin/lapin_sync.py``).

Run against a stand-in for Lapin that keeps elements, one-step undo and the
calls it was sent. What is pinned is what a Lapin user would notice: a second
push does nothing, a change touches one element and keeps its id, nothing the
sync did not make is ever touched, and a failure leaves the project as it was.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lite_step.lapin import LapinOps, Op

_PATH = Path(__file__).resolve().parents[2] / "tools" / "lapin" / "lapin_sync.py"
_spec = importlib.util.spec_from_file_location("lapin_sync", _PATH)
sync = importlib.util.module_from_spec(_spec)
sys.modules["lapin_sync"] = sync  # dataclasses look their module up by name
_spec.loader.exec_module(sync)


class FakeLapin:
    """Enough of Lapin's engine for the sync: elements, subtracts, and a step that rolls back."""

    def __init__(self):
        self.elements: dict = {}
        self.subtracts: set = set()  # (target id, tool id)
        self.calls: list = []
        self._n = 0
        self.fail_on = None
        self.steps: list = []
        self._add("IfcBuildingStorey", "Ground floor", {"elevation": 0.0})

    def _add(self, type_, name, attrs, id_=None):
        self._n += 1
        id_ = id_ or f"e{self._n}"
        assert id_ not in self.elements, f"GlobalId {id_} is taken"
        self.elements[id_] = SimpleNamespace(id=id_, type=type_, name=name, attrs=attrs)
        return id_

    @property
    def model(self):
        return list(self.elements.values())

    def step(self, label):
        fake = self

        class Step:
            def __enter__(self):
                self.before = (copy.deepcopy(fake.elements), set(fake.subtracts))
                fake.steps.append(label)

            def __exit__(self, kind, value, tb):
                if kind is not None:
                    fake.elements, fake.subtracts = self.before
                return False

        return Step()

    def execute(self, command, args):
        self.calls.append((command, args))
        if command == self.fail_on:
            raise RuntimeError(f"{command} refused")
        e = self.elements
        if command == "storey.create":
            return {"id": self._add("IfcBuildingStorey", args["name"], {"elevation": args["elevation"]})}
        if command == "studio.primitive":
            attrs = {"operation": "primitive", **{k: v for k, v in args.items() if k not in ("storeyId", "class", "name")},
                     "exportAs": args["class"], "storeyId": args["storeyId"]}
            return {"id": self._add("IfcBuildingElementProxy", args["name"], attrs)}
        if command == "element.createBody":
            attrs = {"owner": args["owner"], "scriptOwned": True, "body": args["body"], "storeyId": args["storey"]}
            return {"id": self._add(args["ifcClass"], args["name"], attrs, args["globalId"])}
        if command == "element.setBody":
            e[args["id"]].attrs["body"] = args["body"]
        elif command == "element.update":
            e[args["id"]].attrs.update(args["attrs"])
        elif command == "studio.place":
            e[args["id"]].attrs.update(at=args["at"], rotation=args["rotation"])
        elif command == "element.setParam":
            params = [p for p in e[args["id"]].attrs.setdefault("params", []) if p["name"] != args["name"]]
            params.append({"name": args["name"], "kind": args["kind"], "value": args["value"], "pset": args["pset"]})
            e[args["id"]].attrs["params"] = params
        elif command == "element.delete":
            for i in args["ids"]:
                del e[i]
            self.subtracts = {(a, b) for a, b in self.subtracts if a in e and b in e}
        elif command == "studio.boolean":
            assert args["op"] == "subtract"
            self.subtracts |= {(args["target"], t) for t in args["tools"]}
        elif command == "studio.removeBoolean":
            self.subtracts = {(a, b) for a, b in self.subtracts if a not in args["ids"] and b not in args["ids"]}
        else:
            raise AssertionError(f"the sync sent a command the stand-in does not know: {command}")
        return None

    def by_name(self, name):
        return next(x for x in self.elements.values() if x.name == name)

    def names(self):
        return sorted(x.name for x in self.elements.values() if x.type != "IfcBuildingStorey")


def _box(path, width=2.0, x=1.0, tools=(), storey="storey:ground", elevation=0.0, cls="IfcWall"):
    args = {"shape": "box", "at": [x, 0.5, 0.0], "rotation": 0.0, "width": width, "length": 1.0, "height": 3.0}
    return Op(path, "primitive", cls, storey, elevation, args, tuple(tools))


def _mesh(path, size=1.0):
    return Op(path, "body", "IfcBeam", "storey:ground", 0.0,
              {"vertices": [[0, 0, 0], [size, 0, 0], [0, size, 0], [0, 0, size]], "faces": [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]})


def _push(lapin, *ops, model="house"):
    result = LapinOps(ops=list(ops))
    the_plan = sync.plan(result, sync.read_owned(lapin, model))
    counts = sync.apply(lapin, result, the_plan, model, lambda name, ns: f"gid:{ns}:{name}")
    return the_plan, counts


def _mutating(lapin):
    return [c for c, _ in lapin.calls]


# --------------------------------------------------------------------------- first push, second push


def test_the_first_push_makes_every_element_marks_it_and_cuts():
    lapin = FakeLapin()
    _, counts = _push(lapin, _box("slab", tools=["post"]), _box("post", x=5.0), _mesh("beam"))
    assert counts["create"] == 3 and lapin.names() == ["beam", "post", "slab"]
    assert lapin.steps == ["Lite-STEP sync: house"], "one undo step"
    slab, post, beam = lapin.by_name("slab"), lapin.by_name("post"), lapin.by_name("beam")
    assert lapin.subtracts == {(slab.id, post.id)}
    assert (slab.attrs["exportAs"], slab.attrs["width"]) == ("IfcWall", 2.0)
    assert beam.id == "gid:lite-step:house:beam" and beam.attrs["owner"] == "lite-step:house"
    params = {p["name"]: p["value"] for p in slab.attrs["params"] if p["pset"] == "LiteStep"}
    assert params["Path"] == "slab" and json.loads(params["Sync"])["model"] == "house"


def test_a_second_push_of_the_same_model_sends_nothing():
    lapin = FakeLapin()
    ops = (_box("slab", tools=["post"]), _box("post", x=5.0), _mesh("beam"))
    _push(lapin, *ops)
    lapin.calls.clear()
    lapin.steps.clear()
    the_plan, counts = _push(lapin, *ops)
    assert the_plan.empty and counts["keep"] == 3
    assert lapin.calls == [] and lapin.steps == [], "not even an empty undo step"


# --------------------------------------------------------------------------- deltas


def test_a_resized_primitive_is_changed_in_place_and_keeps_its_id():
    lapin = FakeLapin()
    _push(lapin, _box("slab"), _box("post", x=5.0))
    before = {n: lapin.by_name(n).id for n in ("slab", "post")}
    lapin.calls.clear()
    the_plan, _ = _push(lapin, _box("slab", width=4.0, x=2.0), _box("post", x=5.0))
    assert [o.path for o, _ in the_plan.update] == ["slab"] and len(the_plan.keep) == 1
    assert {n: lapin.by_name(n).id for n in ("slab", "post")} == before
    assert (lapin.by_name("slab").attrs["width"], lapin.by_name("slab").attrs["at"]) == (4.0, [2.0, 0.5, 0.0])
    assert {a.get("id") for _, a in lapin.calls} == {before["slab"]}, "the post was not sent a single call"


def test_a_changed_body_gets_a_new_body_on_the_same_element():
    lapin = FakeLapin()
    _push(lapin, _mesh("beam"))
    lapin.calls.clear()
    _push(lapin, _mesh("beam", size=2.0))
    assert "element.setBody" in _mutating(lapin) and "element.createBody" not in _mutating(lapin)
    assert lapin.by_name("beam").attrs["body"]["vertices"][1] == [2.0, 0, 0]


def test_what_cannot_change_in_place_is_made_again():
    """A box that became a mesh, another class, another storey: a new element."""
    for changed in (_mesh("thing"), _box("thing", cls="IfcColumn"), _box("thing", storey="storey:loft", elevation=3.0)):
        lapin = FakeLapin()
        _push(lapin, _box("thing"))
        the_plan, _ = _push(lapin, changed)
        assert len(the_plan.replace) == 1 and lapin.names() == ["thing"]


def test_an_element_gone_from_the_script_is_deleted_with_its_cut():
    lapin = FakeLapin()
    _push(lapin, _box("slab", tools=["post"]), _box("post", x=5.0))
    the_plan, _ = _push(lapin, _box("slab"))
    assert [h.path for h in the_plan.delete] == ["post"] and lapin.names() == ["slab"]
    assert lapin.subtracts == set()


def test_a_changed_cutter_list_is_removed_and_made_again():
    lapin = FakeLapin()
    _push(lapin, _box("slab", tools=["a"]), _box("a", x=5.0), _box("b", x=9.0))
    lapin.calls.clear()
    _push(lapin, _box("slab", tools=["b"]), _box("a", x=5.0), _box("b", x=9.0))
    slab, b = lapin.by_name("slab"), lapin.by_name("b")
    assert lapin.subtracts == {(slab.id, b.id)}
    assert _mutating(lapin).index("studio.removeBoolean") < _mutating(lapin).index("studio.boolean")


def test_a_host_made_again_keeps_cutting_and_being_cut():
    """Deleting an element drops its subtracts on both sides; both come back."""
    lapin = FakeLapin()
    ops = lambda cls: (_box("top", tools=["mid"]), _box("mid", tools=["low"], cls=cls), _box("low", x=9.0))  # noqa: E731
    _push(lapin, *ops("IfcWall"))
    _push(lapin, *ops("IfcColumn"))
    top, mid, low = (lapin.by_name(n).id for n in ("top", "mid", "low"))
    assert lapin.subtracts == {(top, mid), (mid, low)}


# --------------------------------------------------------------------------- ownership


def test_nothing_the_sync_did_not_make_is_touched_or_deleted():
    lapin = FakeLapin()
    wall = lapin._add("IfcWall", "the user's wall", {"operation": "primitive", "shape": "box", "width": 3.0})
    _push(lapin, _box("slab"))
    _push(lapin, model="carport")  # an empty other model
    _push(lapin)                   # the house emptied
    assert lapin.names() == ["the user's wall"]
    assert all(a.get("id") != wall and wall not in a.get("ids", []) for _, a in lapin.calls)


def test_two_models_in_one_project_do_not_see_each_other():
    lapin = FakeLapin()
    _push(lapin, _box("slab"), _mesh("beam"), model="house")
    _push(lapin, _box("slab"), _mesh("beam"), model="carport")
    assert len(lapin.names()) == 4, "the same names, twice: a body's GlobalId carries the model"
    _push(lapin, model="carport")
    assert lapin.names() == ["beam", "slab"]
    assert set(sync.marks(lapin)) == {"house"}


def test_an_existing_storey_is_used_and_none_is_ever_deleted():
    lapin = FakeLapin()
    _push(lapin, _box("a"), _box("b", storey="storey:loft", elevation=3.0), _box("c", storey=None, elevation=None))
    storeys = {s.name: s for s in lapin.model if s.type == "IfcBuildingStorey"}
    assert sorted(storeys) == ["Ground floor", "loft"], "ground matched by its elevation; loft made, without the prefix"
    assert lapin.by_name("a").attrs["storeyId"] == lapin.by_name("c").attrs["storeyId"] == storeys["Ground floor"].id
    _push(lapin)
    assert len([s for s in lapin.model if s.type == "IfcBuildingStorey"]) == 2


def test_a_mark_someone_edited_gives_the_element_up():
    lapin = FakeLapin()
    _push(lapin, _box("slab"))
    slab = lapin.by_name("slab")
    next(p for p in slab.attrs["params"] if p["name"] == "Sync")["value"] = "not json"
    assert sync.read_owned(lapin, "house") == {}
    assert sync.remove(lapin, "house") == 0 and lapin.names() == ["slab"]


def test_remove_deletes_this_models_elements_and_no_others():
    lapin = FakeLapin()
    lapin._add("IfcWall", "the user's wall", {})
    _push(lapin, _box("slab"), _mesh("beam"), model="house")
    _push(lapin, _box("slab"), model="carport")
    assert sync.remove(lapin, "house") == 2
    assert lapin.names() == ["slab", "the user's wall"]
    assert lapin.steps[-1] == "Lite-STEP remove: house"


# --------------------------------------------------------------------------- hand edits, failure


def test_a_primitive_changed_in_lapin_is_named_and_put_back():
    lapin = FakeLapin()
    _push(lapin, _box("slab"), _box("post", x=5.0))
    lapin.by_name("slab").attrs["width"] = 7.0  # the user dragged a face
    the_plan, _ = _push(lapin, _box("slab"), _box("post", x=5.0))
    assert the_plan.hand_edited == ["slab"] and [o.path for o, _ in the_plan.update] == ["slab"]
    assert lapin.by_name("slab").attrs["width"] == 2.0
    assert sync.plan(LapinOps(ops=[_box("slab"), _box("post", x=5.0)]), sync.read_owned(lapin, "house")).empty


def test_float_noise_is_not_a_hand_edit():
    lapin = FakeLapin()
    _push(lapin, _box("slab"))
    lapin.by_name("slab").attrs["width"] = 2.0 + 1e-9
    assert _push(lapin, _box("slab"))[0].empty


def test_a_failure_part_way_leaves_lapin_as_it_was():
    lapin = FakeLapin()
    _push(lapin, _box("slab"), _box("post", x=5.0))
    before = copy.deepcopy(lapin.elements)
    lapin.fail_on = "element.createBody"
    with pytest.raises(RuntimeError, match="createBody refused"):
        _push(lapin, _box("slab", width=9.0), _mesh("beam"))
    assert lapin.elements == before, "the delete and the resize that came first were rolled back"


def test_a_cutter_that_is_not_in_lapin_stops_the_push():
    lapin = FakeLapin()
    with pytest.raises(RuntimeError, match="slab is cut by ghost, which is not in Lapin"):
        _push(lapin, _box("slab", tools=["ghost"]))
    assert lapin.names() == []
