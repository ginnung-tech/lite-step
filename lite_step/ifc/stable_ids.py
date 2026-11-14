"""GlobalIds that are the same every time the same model is compiled.

An IFC GlobalId is how other programs keep hold of an element: Lapin's links
reload by it, BCF topics and clash results point at it, a schedule exported
yesterday names it. Random ids made every compile a different building to all
of them. Here every ``IfcRoot`` gets an id derived from what it is:

- an object (element, storey, type, the project) from its class and its
  canonical name, which the compiler derives from the element's place in the
  tree, so it survives edits elsewhere in the model;
- an object without a name from the named objects it is related to;
- a property set from its name and the objects it describes;
- a relationship from its class and the ids of what it relates.

All of them in a namespace made from the project's name, so two models linked
into one project do not collide on a shared element name. When two entities
would get the same id (two elements sharing a name), the later ones in the
file take ``#2``, ``#3``…: still stable while their order is.

The id is a UUID 5 of that text, compressed as IFC writes GlobalIds.
"""

from __future__ import annotations

import json
import uuid

import ifcopenshell
import ifcopenshell.guid

#: The namespace every Lite-STEP model's own namespace is made in. Never change it:
#: every id in every file compiled from then on would change with it.
NAMESPACE = uuid.UUID("5c0f8a52-3b1e-4d6a-9e2f-7a41c8d0b913")


def _sides(entity) -> list:
    """The IfcRoot entities an entity refers to, one list per attribute (a relationship's sides)."""
    out = []
    for value in entity:
        items = value if isinstance(value, (tuple, list)) else (value,)
        side = [v for v in items if isinstance(v, ifcopenshell.entity_instance) and v.is_a("IfcRoot")]
        if side:
            out.append(side)
    return out


def _refs(entity) -> list:
    """The IfcRoot entities an entity refers to, attribute by attribute."""
    return [v for side in _sides(entity) for v in side]


def stabilise_global_ids(model) -> dict:
    """Give every IfcRoot in ``model`` its stable GlobalId; ``{old: new}`` for what changed."""
    projects = model.by_type("IfcProject")
    namespace = uuid.uuid5(NAMESPACE, (projects[0].Name or "") if projects else "")

    roots = sorted(model.by_type("IfcRoot"), key=lambda e: e.id())
    objects = [e for e in roots if e.is_a("IfcObjectDefinition")]
    properties = [e for e in roots if e.is_a("IfcPropertyDefinition")]
    relations = [e for e in roots if e.is_a("IfcRelationship")]

    # Who is related to whom, read once from the relationships: across a relationship's
    # sides only (an element and the storey containing it), never among the members of
    # one side. Siblings say nothing about an entity, and a storey's 4,000 elements
    # would each get the 3,999 others: 16 million entries and half a minute.
    neighbours: dict = {}
    for rel in relations:
        sides = [[v.id() for v in side] for side in _sides(rel)]
        for i, side in enumerate(sides):
            across = [x for j, other in enumerate(sides) if j != i for x in other]
            for a in side:
                neighbours.setdefault(a, set()).update(across)

    key: dict = {}

    def settle(entity, text: str) -> None:
        key[entity.id()] = text

    named = [e for e in objects if e.Name]
    for e in named:
        settle(e, f"{e.is_a()}|{e.Name}")

    def anchored(entity, prefix: str) -> str:
        around = sorted(key[n] for n in neighbours.get(entity.id(), ()) if n in key)
        return f"{prefix}|{entity.is_a()}|{entity.Name or ''}|" + "|".join(around)

    for e in objects:
        if not e.Name:
            settle(e, anchored(e, "object"))
    for e in properties:
        settle(e, anchored(e, "properties"))
    for rel in relations:
        ends = [key.get(x.id(), x.is_a()) for x in _refs(rel)]
        settle(rel, f"relation|{rel.is_a()}|{rel.Name or ''}|" + "|".join(sorted(ends)))

    seen: dict = {}
    changed: dict = {}
    for e in roots:
        text = key[e.id()]
        n = seen[text] = seen.get(text, 0) + 1
        if n > 1:
            text = f"{text}#{n}"
        new = ifcopenshell.guid.compress(uuid.uuid5(namespace, text).hex)
        if e.GlobalId != new:
            changed[e.GlobalId] = new
            e.GlobalId = new
    return changed


def remap_manifest(model, changed: dict, pset_name: str) -> None:
    """The LITESTEP_META manifest (``{name: GlobalId}``) brought up to the new ids."""
    if not changed:
        return
    for pset in model.by_type("IfcPropertySet"):
        if pset.Name != pset_name:
            continue
        for prop in pset.HasProperties or ():
            if prop.Name == "MANIFEST" and prop.NominalValue is not None:
                manifest = json.loads(prop.NominalValue.wrappedValue)
                remapped = {name: changed.get(gid, gid) for name, gid in manifest.items()}
                prop.NominalValue.wrappedValue = json.dumps(remapped, separators=(",", ":"))
