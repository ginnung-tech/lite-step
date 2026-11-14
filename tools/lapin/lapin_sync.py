"""Lapin operations applied to a running Lapin, as a delta.

``lite_step/lapin/ops.py`` says what each element of a model should be in
Lapin. This module looks at what Lapin holds for that model, works out the
difference, and applies it as one undo step.

**Ownership.** The sync touches only elements it made: those carrying its
mark for this model id. Everything else in the project (what the user drew,
what another model pushed, every storey) is left alone. The mark is two
custom parameters in the set ``LiteStep``: ``Path``, the element's canonical
name, there for people to read; and ``Sync``, a small JSON text the next run
reads. Lapin saves both with the project, so the delta still works after the
project is closed and opened again.

**The client.** Any object with ``execute(command, args)``, ``step(label)``
(a context manager that is one undo step and rolls back on an exception) and
``model`` (iterable elements with ``id``, ``type``, ``name``, ``attrs``).
That is ``lapinbim.Lapin``; the tests use a small stand-in.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

PSET = "LiteStep"
_SIZES = {"box": ("width", "length", "height"), "cylinder": ("radius", "height")}
#: A hand edit smaller than this is not one (metres; Lapin keeps its own float noise).
_EDIT_TOLERANCE = 1e-4


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def frame_of(op) -> str:
    """What cannot be changed on an element in place: when this differs, it is made again."""
    return _digest([op.kind, op.ifc_class, op.storey, op.elevation, op.args.get("shape")])


def owner_of(model_id: str) -> str:
    return f"lite-step:{model_id}"


@dataclass
class Owned:
    """An element in Lapin that an earlier sync of this model made."""

    id: str
    path: str
    hash: str
    frame: str
    tools: list
    #: For a primitive: its shape as the last sync set it, and as Lapin holds it now.
    set_shape: Optional[dict] = None
    now_shape: Optional[dict] = None

    @property
    def hand_edited(self) -> bool:
        """A primitive the user resized, moved or turned in Lapin since the last sync."""
        if self.set_shape is None or self.now_shape is None:
            return False
        return not _same_shape(self.set_shape, self.now_shape)


def _same_shape(a: dict, b: dict) -> bool:
    if a.get("shape") != b.get("shape"):
        return False
    flat = lambda s: [*s.get("at", []), s.get("rotation", 0.0), *(s.get(k, 0.0) for k in _SIZES.get(s.get("shape"), ()))]  # noqa: E731
    x, y = flat(a), flat(b)
    return len(x) == len(y) and all(abs(float(p) - float(q)) < _EDIT_TOLERANCE for p, q in zip(x, y))


def _params(element) -> dict:
    return {p.get("name"): p.get("value") for p in (element.attrs.get("params") or []) if p.get("pset") == PSET}


def _shape_now(element) -> Optional[dict]:
    a = element.attrs
    if a.get("operation") != "primitive":
        return None
    out = {"shape": a.get("shape"), "at": list(a.get("at") or []), "rotation": a.get("rotation", 0.0)}
    for key in _SIZES.get(a.get("shape"), ()):
        out[key] = a.get(key)
    return out


def marks(client) -> dict:
    """``{model id: {path: Owned}}`` for every element in Lapin that carries a Lite-STEP mark."""
    out: dict = {}
    for element in client.model:
        params = _params(element)
        if "Sync" not in params:
            continue
        try:
            mark = json.loads(params["Sync"])
            owned = Owned(element.id, mark["path"], mark["hash"], mark["frame"], list(mark.get("tools", [])),
                          mark.get("shape"), _shape_now(element))
            out.setdefault(mark["model"], {})[owned.path] = owned
        except (ValueError, KeyError, TypeError):
            # A mark someone edited: the element is no longer ours to change or delete.
            continue
    return out


def read_owned(client, model_id: str) -> dict:
    return marks(client).get(model_id, {})


@dataclass
class Plan:
    """What a sync would do. Built without changing anything."""

    create: list = field(default_factory=list)   # Op
    update: list = field(default_factory=list)   # (Op, Owned): changed in place, id kept
    replace: list = field(default_factory=list)  # (Op, Owned): deleted and made again
    delete: list = field(default_factory=list)   # Owned: gone from the script
    keep: list = field(default_factory=list)     # (Op, Owned): untouched
    #: Paths of primitives the user changed in Lapin; the script's version is put back.
    hand_edited: list = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.create or self.update or self.replace or self.delete)

    def counts(self) -> dict:
        return {"create": len(self.create), "update": len(self.update), "replace": len(self.replace),
                "delete": len(self.delete), "keep": len(self.keep), "hand_edited": len(self.hand_edited)}


def plan(ops, owned: dict) -> Plan:
    out = Plan()
    wanted = set()
    for op in ops.ops:
        wanted.add(op.path)
        have = owned.get(op.path)
        if have is None:
            out.create.append(op)
            continue
        if have.hand_edited:
            out.hand_edited.append(op.path)
        if have.frame != frame_of(op):
            out.replace.append((op, have))
        elif have.hash != op.hash or have.hand_edited:
            out.update.append((op, have))
        else:
            out.keep.append((op, have))
    out.delete = [have for path, have in owned.items() if path not in wanted]
    return out


# --------------------------------------------------------------------------- applying


def _storey_label(name: Optional[str], elevation: Optional[float]) -> str:
    if name:
        return name.split(":", 1)[1] if name.startswith("storey:") else name
    return f"Level {elevation or 0.0:g}"


def _storey_id(client, cache: dict, name: Optional[str], elevation: Optional[float]) -> str:
    """Lapin's storey for an operation: by name, else by elevation, else made. Never deleted."""
    key = (name, elevation)
    if key in cache:
        return cache[key]
    storeys = [e for e in client.model if e.type == "IfcBuildingStorey"]
    label = _storey_label(name, elevation)
    found = None
    if name:
        found = next((s for s in storeys if s.name.lower() in (name.lower(), label.lower())), None)
    if found is None and elevation is not None:
        found = next((s for s in storeys if abs(float(s.attrs.get("elevation") or 0.0) - elevation) < 1e-3), None)
    if found is None and name is None and elevation is None:
        found = min(storeys, key=lambda s: abs(float(s.attrs.get("elevation") or 0.0)), default=None)
    if found is not None:
        cache[key] = found.id
    else:
        cache[key] = client.execute("storey.create", {"name": label, "elevation": elevation or 0.0})["id"]
    return cache[key]


def _mark(client, element_id: str, model_id: str, op) -> None:
    mark = {"model": model_id, "path": op.path, "hash": op.hash, "frame": frame_of(op), "tools": list(op.tools)}
    if op.kind == "primitive":
        mark["shape"] = op.args
    client.execute("element.setParam", {"id": element_id, "pset": PSET, "name": "Path", "kind": "text", "value": op.path})
    client.execute("element.setParam", {"id": element_id, "pset": PSET, "name": "Sync", "kind": "text",
                                        "value": json.dumps(mark, sort_keys=True, separators=(",", ":"))})


def _body(op) -> dict:
    return {"vertices": op.args["vertices"], "faces": op.args["faces"]}


def _create(client, op, model_id: str, storey_id: str, make_id: Callable) -> str:
    if op.kind == "primitive":
        args = {**op.args, "storeyId": storey_id, "class": op.ifc_class, "name": op.path}
        element_id = client.execute("studio.primitive", args)["id"]
    else:
        owner = owner_of(model_id)
        element_id = client.execute("element.createBody", {
            "ifcClass": op.ifc_class, "body": _body(op), "storey": storey_id, "name": op.path,
            "globalId": make_id(op.path, owner), "owner": owner,
        })["id"]
    _mark(client, element_id, model_id, op)
    return element_id


def _update(client, op, have: Owned, model_id: str) -> None:
    if op.kind == "primitive":
        sizes = {k: op.args[k] for k in _SIZES[op.args["shape"]]}
        client.execute("element.update", {"id": have.id, "attrs": sizes})
        client.execute("studio.place", {"id": have.id, "at": op.args["at"], "rotation": op.args["rotation"]})
    else:
        client.execute("element.setBody", {"id": have.id, "body": _body(op)})
    _mark(client, have.id, model_id, op)


def apply(client, ops, the_plan: Plan, model_id: str, make_id: Callable) -> dict:
    """The plan carried out as one undo step. An exception leaves Lapin as it was.

    ``make_id(name, namespace)`` gives a body's GlobalId (``lapinbim.global_id``): the
    same for the same name, so a body made again is the same element to Lapin and to IFC.
    """
    if the_plan.empty:
        return the_plan.counts()
    ids = {op.path: have.id for op, have in the_plan.keep}
    with client.step(f"Lite-STEP sync: {model_id}"):
        storeys: dict = {}
        gone = [have.id for have in the_plan.delete] + [have.id for _, have in the_plan.replace]
        if gone:
            client.execute("element.delete", {"ids": gone})
        touched = set()  # elements whose subtracts must be (re)made
        for op in [*the_plan.create, *(op for op, _ in the_plan.replace)]:
            ids[op.path] = _create(client, op, model_id, _storey_id(client, storeys, op.storey, op.elevation), make_id)
            touched.add(op.path)
        for op, have in the_plan.update:
            ids[op.path] = have.id
            if list(op.tools) != have.tools:
                # Lapin removes every operation the element takes part in, as target or as tool.
                client.execute("studio.removeBoolean", {"ids": [have.id]})
                touched.add(op.path)
            _update(client, op, have, model_id)
        for op in ops.ops:
            if not op.tools or op.path not in ids:
                continue
            tools = [t for t in op.tools if op.path in touched or t in touched]
            missing = [t for t in tools if t not in ids]
            if missing:
                raise RuntimeError(f"{op.path} is cut by {missing[0]}, which is not in Lapin")
            if tools:
                client.execute("studio.boolean", {"op": "subtract", "target": ids[op.path], "tools": [ids[t] for t in tools]})
    return the_plan.counts()


def remove(client, model_id: str) -> int:
    """Everything this model owns in Lapin deleted, as one undo step; nothing else."""
    owned = read_owned(client, model_id)
    if owned:
        with client.step(f"Lite-STEP remove: {model_id}"):
            client.execute("element.delete", {"ids": [o.id for o in owned.values()]})
    return len(owned)
