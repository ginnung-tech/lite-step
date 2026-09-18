"""A product does not go inside AIR, by ``.add()`` OR by ``.anchor()``.

``.add()`` has refused a product into a ``Space`` for a long time, and its
message gives a reason about what a ``Space`` **is** rather than about how the
child's coordinates were expressed::

    Space 'k2': invalid child Wall 'n' — a Space is AIR, and .add() into one
    takes only the Box that is the volume it occupies. A physical element does
    not go INSIDE a space; it BOUNDS one, and IFC's relation for that is
    IfcRelSpaceBoundary …

Nothing in that reason is about ``.add()``. It applied to ``.anchor()`` word
for word, and only ``.add()`` enforced it. Measured on ``main`` @
``a7f41e9``::

    Space.anchor(Wall)  -> ACCEPTED at the line
    compile success: True
    AGG:      IfcSpace space:kitchen:storey:ground -> ['IfcWall']
    CONTAINS: IfcBuildingStorey storey:ground      -> ['IfcSpace']

That ``IfcRelAggregates`` is not "unhelpful". It is a DECOMPOSITION claiming
the wall is a **part of the room's air** — a statement IFC has a different,
correct relation for (``IfcRelSpaceBoundary``, which this repo already derives
from the geometry in ``lite_step/ifc/space_boundaries.py``). So an author who
hit the ``.add()`` refusal could route around it with one word and get a wrong
relation in the file, with no diagnostic anywhere.

**Why the assertions here are about the RELATION and never about a count.**
The wall IS in the pre-fix file — ``IFCWALL(`` is present, one of them, at the
anchored coordinates. Every count-shaped assertion available (products, walls,
aggregations, validator errors) passes on the broken version. Only the SHAPE of
the relation distinguishes them, which is exactly how this survived.

What stays legal, deliberately, and pinned below:

1.  ``space.anchor(<geometry primitive>)``. A ``Box``/``Sweep``/``Mesh`` is a
    SHAPE, not a building element. (NOT for the obvious reason: a Space's own
    volume comes from ``.add(Box)``, which becomes the ``IfcSpace``'s
    *representation*. An ``.anchor()``-ed Box is a separate
    ``IfcBuildingElementProxy`` child — measured in
    ``test_an_anchored_box_is_not_the_spaces_volume``, because a comment
    claiming otherwise would send the next reader to the wrong fix.)
2.  ``space.anchor(space)``. IFC decomposes an ``IfcSpace`` into ``IfcSpace``
    through ``IfcRelAggregates``, so this is the one aggregation from a Space
    that says something true.
3.  Every physical pairing that does not cross the boundary —
    ``test_anchor_matrix.py`` walks the whole product and
    ``test_container_children.py`` pins the cost of this refusal at six cells.

And the second half, unrelated to Spaces: ``Storey.add(Window)``
and ``Project.add(Window)`` were accepted at the authoring line, because
neither routes through ``_validate_container_children`` where the opening refusal lives. The compile-time backstop caught it loudly, so nothing
misbuilt — what changes is only WHERE the author learns it.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Beam, Box, Column, Door, Element, Extrude, Mesh, Point, Point2D, Project,
    Roof, Site, Slab, Space, Storey, Sweep, Wall, Window,
)
from lite_step.models import taxonomy as tx


# ---------------------------------------------------------------------------
# Fixtures — one 4 x 3 x 2.5 m room on a named storey, and the wall that
# bounds it. Both are authored the way the refusal's own remedy says to.
# ---------------------------------------------------------------------------

def _room() -> Space:
    """A Space with its volume, which is what ``.add(Box)`` is for."""
    space = Space(name="kitchen")
    space.add(Box(name="vol", start=Point(x=0, y=0, z=0),
                  end=Point(x=4000, y=3000, z=2500)))
    return space


def _wall(name: str = "north") -> Wall:
    wall = Wall(name=name)
    wall.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=4000, y=200, z=2500)))
    return wall


def _bodied(element):
    element.add(Box(name="body", start=Point(x=0, y=0, z=0),
                    end=Point(x=300, y=120, z=200)))
    return element


#: The six PHYSICAL children the refusal takes off the ``Space`` column, each
#: built valid so a ``pytest.raises`` here cannot be catching pydantic instead.
PHYSICAL_CHILDREN = {
    "Wall": lambda: _bodied(Wall(name="det")),
    "Column": lambda: _bodied(Column(name="det")),
    "Beam": lambda: _bodied(Beam(name="det")),
    "Slab": lambda: _bodied(Slab(name="det")),
    "Roof": lambda: _bodied(Roof(name="det")),
    "Element": lambda: _bodied(
        Element(ifc_class="IfcBuildingElementProxy", name="det")),
}

GEOMETRY_CHILDREN = {
    "Box": lambda: Box(name="det", start=Point(x=0, y=0, z=0),
                       end=Point(x=200, y=100, z=60)),
    "Extrude": lambda: Extrude(name="det", thickness=100, contour=[
        Point(x=0, y=0, z=0), Point(x=200, y=0, z=0),
        Point(x=200, y=0, z=60), Point(x=0, y=0, z=60)]),
    "Sweep": lambda: Sweep(
        name="det",
        profile=[Point2D(x=-20, y=-20), Point2D(x=20, y=-20),
                 Point2D(x=20, y=20), Point2D(x=-20, y=20)],
        path=[Point(x=0, y=0, z=0), Point(x=400, y=0, z=0)]),
    "Mesh": lambda: Mesh(
        name="det",
        vertices=[Point(x=0, y=0, z=0), Point(x=200, y=0, z=0),
                  Point(x=200, y=200, z=0), Point(x=0, y=0, z=100)],
        faces=[(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]),
}


def _project(*elements) -> Project:
    proj = Project(name="spatial-boundary")
    storey = Storey(name="ground")
    storey.add(*elements)
    proj.add_storey(storey)
    return proj


def _emit(proj: Project):
    """Compile and reopen with ifcopenshell, so the assertions read RELATIONS
    off the artefact rather than off the model tree — a relation written from
    the wrong side is invisible in the tree that produced it."""
    ios = pytest.importorskip("ifcopenshell")
    with contextlib.redirect_stdout(io.StringIO()):
        result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    path = os.path.join(tempfile.mkdtemp(), "m.ifc")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(result.ifc_content)
    return ios.open(path)


def _space_aggregated_building_elements(model):
    """Every ``(space name, related entity)`` where an ``IfcSpace`` DECOMPOSES
    into a building element — the exact wrong relation, and nothing else.

    ``IfcBuildingElement`` and not ``IfcProduct``: a sub-``IfcSpace`` under a
    space is correct IFC and must not be reported, and neither must the
    ``IfcBuildingStorey`` above it.
    """
    hits = []
    for rel in model.by_type("IfcRelAggregates"):
        if not rel.RelatingObject.is_a("IfcSpace"):
            continue
        for obj in rel.RelatedObjects:
            if obj.is_a("IfcBuildingElement") or obj.is_a("IfcElement"):
                hits.append((rel.RelatingObject.Name, obj.is_a()))
    return hits


# ---------------------------------------------------------------------------
# 1. The refusal, at the authoring line, for every physical child
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("child_name", sorted(PHYSICAL_CHILDREN))
def test_a_space_refuses_a_physical_child_through_anchor_too(child_name):
    """``space.anchor(<product>)`` raises, and names the relation IFC has.

    The message has to carry ``IfcRelSpaceBoundary`` for the same reason
    ``.add()``'s does: an author refused with "no" and no alternative goes
    looking for a spelling that works, and there was one.
    """
    space = _room()
    with pytest.raises(ValueError) as exc:
        space.anchor(PHYSICAL_CHILDREN[child_name](), along=1000, up=0)
    message = str(exc.value)
    assert "IfcRelSpaceBoundary" in message, message
    assert "IfcRelAggregates" in message, message
    assert child_name in message, message
    # A refused .anchor() never partially mutates the container: the Box that
    # is the room's volume is still the only child.
    assert [type(c).__name__ for c in space._elements] == ["Box"]


def test_both_verbs_refuse_the_same_pairing_for_the_same_reason():
    """One rule, two verbs. The two messages must agree on the RELATION, which
    is the part an author acts on — they differ only in which verb they name.

    A rule stated in two places is the shape; this is the assertion that
    they have not drifted into two rules.
    """
    added, anchored = None, None
    with pytest.raises(ValueError) as exc:
        _room().add(_bodied(Wall(name="det")))
    added = str(exc.value)
    with pytest.raises(ValueError) as exc:
        _room().anchor(_bodied(Wall(name="det")), along=1000, up=0)
    anchored = str(exc.value)

    for message in (added, anchored):
        assert "IfcRelSpaceBoundary" in message, message
        assert "space_boundaries.py" in message, message
        assert "Space is AIR" in message, message
    assert ".add()" in added
    assert ".anchor()" in anchored


def test_an_opening_into_a_space_names_the_wall_not_the_storey():
    """A ``Window`` is physical too, so the boundary catches it — but "add it
    to the storey as a peer" is the WRONG remedy for one: an opening carries no
    coordinates, and a Space is not voidable, so nothing here could host it.

    Pre-fix this pairing was worse than the wall: ``space.anchor(window)``
    compiled clean and emitted NO ``IfcWindow`` at all — a silent DROP rather
    than a wrong relation.
    """
    for opening in (Window(name="w", width=800, height=1200),
                    Door(name="d", width=900, height=2100)):
        with pytest.raises(ValueError) as exc:
            _room().anchor(opening, along=1000, up=0)
        message = str(exc.value)
        assert "wall.anchor(" in message, message
        assert "as a PEER" not in message, message
        # …and the evidence clause states the DROP, not a wrong relation the
        # opening path never wrote.
        assert "silent drop" in message, message
        assert "IfcRelAggregates" not in message, message


def test_a_space_is_the_only_spatial_container_that_can_reach_the_refusal():
    """The refusal is written on ``is_spatial(container)``, which is
    ``(Site, Space)`` — but a ``Site`` has no ``.anchor()`` at all
    (``has_local_frame(Site)`` is False, refused by the line), so
    ``Space`` is the only spatial type that reaches ``_anchor_child`` as a
    container. Asserted rather than assumed: if a ``Site`` ever grew a frame,
    the branch above would start firing on it, and this is where that shows up.
    """
    assert sorted(t.__name__ for t in tx.SPATIAL) == ["Site", "Space"]
    assert tx.has_local_frame(Space) and not tx.has_local_frame(Site)
    assert hasattr(Space, "anchor") and not hasattr(Site, "anchor")
    # Not a refusal inside ``_anchor_child`` — the method does not exist on a
    # Site at all, so the call never gets that far.
    with pytest.raises(AttributeError, match="anchor"):
        Site(name="plot").anchor(_bodied(Wall(name="det")))
    # ``.opening()`` is the verb that DOES exist and refuses, naming ``.add()``.
    with pytest.raises(TypeError, match="local frame"):
        Site(name="plot").opening(Window(name="w", width=800, height=1200),
                                  along=1000, up=900)


# ---------------------------------------------------------------------------
# 2. The relation, in the emitted file — the regression assertion
# ---------------------------------------------------------------------------

def test_no_ifc_space_ever_decomposes_into_a_building_element():
    """**The regression assertion**, and it is about the RELATION.

    Either the DSL refuses the authoring line, or the compiled file carries no
    ``IfcRelAggregates(IfcSpace -> IfcElement)``. Pre-fix it did NEITHER: the
    call was accepted, the compile succeeded, and the file carried exactly that
    relation with an ``IfcWall`` on the wrong side of it.

    Written as "refuses OR emits nothing" on purpose. It is the statement the
    issue makes, and it survives either resolution of the issue's open question
    — a refusal in ``_anchor_child`` (what shipped) or a corrected emitter.
    """
    space = _room()
    try:
        space.anchor(_wall(), along=1000, up=0)
    except ValueError as exc:
        assert "IfcRelSpaceBoundary" in str(exc)
        return
    pytest.fail(  # pragma: no cover - reached only on a reverted fix
        "space.anchor(wall) was ACCEPTED; emitted relation: "
        f"{_space_aggregated_building_elements(_emit(_project(space)))}")


def test_the_peer_model_the_refusal_recommends_compiles_with_the_wall_in_it():
    """The remedy has to WORK — the probe finding was a refusal
    whose advice hit a second refusal one line later.

    So: follow the message (wall and space as peers on the storey), compile,
    and assert three things about the file — no space decomposes into a
    building element, the wall IS reachable through
    ``IfcRelContainedInSpatialStructure``, and the space is reachable from the
    storey too. Without the last two, "no bad relation" would also hold for an
    empty file.

    The space's own link is found by scanning BOTH spatial relations for the
    one that names it, rather than by naming one of them. WHICH relation
    carries a ``Space`` is a live question elsewhere —
    ``IfcRelContainedInSpatialStructure``'s WR31 forbids a spatial element in
    it — and this test has no stake in the answer. It has a stake in the space
    being REACHABLE from its storey, which is the property a wrong relation
    destroys.
    """
    model = _emit(_project(_room(), _wall()))

    assert _space_aggregated_building_elements(model) == []

    contained = {rel.RelatingStructure.is_a(): [o.is_a() for o in
                                               rel.RelatedElements]
                 for rel in model.by_type("IfcRelContainedInSpatialStructure")}
    assert "IfcWall" in contained.get("IfcBuildingStorey", []), contained

    space = model.by_type("IfcSpace")[0]
    parents = [rel.RelatingObject.is_a()
               for rel in model.by_type("IfcRelAggregates")
               if space in rel.RelatedObjects]
    parents += [rel.RelatingStructure.is_a()
                for rel in model.by_type("IfcRelContainedInSpatialStructure")
                if space in rel.RelatedElements]
    assert parents == ["IfcBuildingStorey"], parents
    assert len(model.by_type("IfcSpace")) == 1
    assert len(model.by_type("IfcWall")) == 1


def test_a_product_count_cannot_tell_the_two_models_apart():
    """Why the assertion above is shaped the way it is.

    The peer model — the CORRECT one — carries exactly one ``IfcWall`` and one
    ``IfcSpace``, and every product in it is reachable. Those are the same
    numbers the broken model produced, which is the whole reason a count-based
    test would have shipped this bug. Pinned as numbers so the claim is
    checkable rather than asserted in a comment.

    Deliberately NOT a total ``IfcRelAggregates`` count: how many aggregations
    a well-formed file carries is a property of the spatial scaffold, not of
    this refusal, and pinning it here would make an unrelated change to that
    scaffold fail in a test about Spaces and walls. The count that belongs to
    this refusal is the last line, and it is zero.
    """
    model = _emit(_project(_room(), _wall()))
    assert len(model.by_type("IfcWall")) == 1
    assert len(model.by_type("IfcSpace")) == 1
    assert _space_aggregated_building_elements(model) == []


# ---------------------------------------------------------------------------
# 3. What stays legal — decided, not left over
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("child_name", sorted(GEOMETRY_CHILDREN))
def test_a_geometry_primitive_still_anchors_into_a_space(child_name):
    """A shape is not a building element. The refusal is on the CHILD being a
    PRODUCT, so every geometry primitive still goes in — which is what keeps
    this a boundary question rather than a container blacklist."""
    space = _room()
    assert not tx.is_physical(GEOMETRY_CHILDREN[child_name]())
    space.anchor(GEOMETRY_CHILDREN[child_name](), along=1000, up=0)
    assert space._elements[-1]._anchor_spec is not None


def test_an_anchored_box_is_not_the_spaces_volume():
    """The measurement behind the paragraph in ``_anchor_child``'s docstring.

    It would be natural to justify keeping ``space.anchor(Box)`` legal with
    "a Space's body is geometry, so refusing it would break the Space's
    shape". That is FALSE, and a future reader acting on it would look for the
    volume in the wrong place: the volume comes from ``.add(Box)``, which
    becomes the ``IfcSpace``'s own representation and emits no aggregation at
    all. An ``.anchor()``-ed Box is a separate ``IfcBuildingElementProxy``.

    Both halves asserted, because the contrast IS the finding.
    """
    volume_only = _emit(_project(_room()))
    assert volume_only.by_type("IfcSpace")[0].Representation is not None
    assert [r for r in volume_only.by_type("IfcRelAggregates")
            if r.RelatingObject.is_a("IfcSpace")] == []

    space = _room()
    space.anchor(GEOMETRY_CHILDREN["Box"](), along=1000, up=0)
    with_detail = _emit(_project(space))
    assert with_detail.by_type("IfcSpace")[0].Representation is not None
    proxies = [o.is_a() for r in with_detail.by_type("IfcRelAggregates")
               if r.RelatingObject.is_a("IfcSpace") for o in r.RelatedObjects]
    assert proxies == ["IfcBuildingElementProxy"], proxies


def test_a_space_still_decomposes_into_a_space():
    """Kept legal, and this is the reason: IFC decomposes an ``IfcSpace`` into
    ``IfcSpace`` through ``IfcRelAggregates``, so a sub-space is the one
    aggregation from a Space that states something true. It also falls out of
    the predicate rather than being special-cased — ``is_physical(Space)`` is
    False, so place-inside-place never crosses the boundary at all.
    """
    assert not tx.is_physical(Space) and tx.is_spatial(Space)
    space = _room()
    space.anchor(_bodied(Space(name="pantry")), along=1000, up=0)
    model = _emit(_project(space))

    related = [o.is_a() for r in model.by_type("IfcRelAggregates")
               if r.RelatingObject.is_a("IfcSpace") for o in r.RelatedObjects]
    assert related == ["IfcSpace"], related
    assert _space_aggregated_building_elements(model) == []


def test_a_wall_still_takes_a_space_child():
    """The boundary is asked of the PAIR, not of the ``Space`` type. A ``Space``
    anchored into a physical container is untouched — ``test_space_bounds.py``
    authors exactly that — and only the container-is-air direction changed."""
    wall = _wall()
    wall.anchor(_bodied(Space(name="pantry")), along=1000, up=0)
    assert wall._elements[-1]._anchor_spec is not None


# ---------------------------------------------------------------------------
# 4. Part 2 — Storey.add(Window) / Project.add(Window)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("opening_name,make", [
    ("Window", lambda: Window(name="w", width=800, height=1200)),
    ("Door", lambda: Door(name="d", width=900, height=2100)),
])
def test_a_storey_refuses_an_opening_at_the_authoring_line(opening_name, make):
    """``Storey.add`` has no type table, so the refusal — whose
    message says an opening "cannot be ADDED", full stop — never fired here.

    The same function every other container calls, so the message is the same
    message, naming ``Storey.add()``.
    """
    storey = Storey(name="ground")
    with pytest.raises(ValueError) as exc:
        storey.add(make())
    message = str(exc.value)
    assert message.startswith(f"Storey.add(): {opening_name} "), message
    assert "wall.anchor(" in message and "body.opening(" in message, message
    assert storey.elements == [], "a refused .add() must not have appended"


@pytest.mark.parametrize("opening_name,make", [
    ("Window", lambda: Window(name="w", width=800, height=1200)),
    ("Door", lambda: Door(name="d", width=900, height=2100)),
])
def test_a_project_refuses_an_opening_before_minting_a_storey(opening_name,
                                                              make):
    """``Project.add`` forwards non-Site elements to the default storey, so it
    would now reach ``Storey.add``'s refusal anyway — but only AFTER
    auto-creating that storey.

    The number is what makes this test worth having: ``len(proj.storeys)`` must
    be **0** afterwards. A refused ``.add()`` leaves the project exactly as it
    was rather than minting a phantom storey on the way to an error, which is
    the contract the storey refusal already keeps here.
    """
    proj = Project(name="p")
    with pytest.raises(ValueError) as exc:
        proj.add(make())
    # The NUMBER first, because it is the half the forward cannot give: with
    # only ``Storey.add``'s copy of the refusal in place this is 1, and the
    # message says "Storey.add()" rather than the method the author called.
    assert len(proj.storeys) == 0, (
        "the refusal fired after the default storey was minted")
    message = str(exc.value)
    assert message.startswith(f"Project.add(): {opening_name} "), message


def test_the_opening_refusal_is_one_function_not_two_messages():
    """Reused, not restated. ``Wall.add``, ``Storey.add`` and ``Project.add``
    must produce the SAME sentence about openings, differing only in the verb
    owner — a second copy of a message is how two refusals drift into two
    rules."""
    messages = []
    for container, call in (
        ("Wall", lambda: _wall().add(Window(name="w", width=800, height=1200))),
        ("Storey", lambda: Storey(name="g").add(
            Window(name="w", width=800, height=1200))),
        ("Project", lambda: Project(name="p").add(
            Window(name="w", width=800, height=1200))),
    ):
        with pytest.raises(ValueError) as exc:
            call()
        message = str(exc.value)
        assert message.startswith(f"{container}.add(): "), message
        messages.append(message[len(f"{container}.add(): "):])

    assert len(set(messages)) == 1, messages


def test_the_compile_time_backstop_is_still_there():
    """The early check is IN ADDITION to the backstop, never instead of it —
    the same pairing ``frames.check_anchor_rotation`` uses.

    A hand-built tree that bypasses ``.add()`` still fails the compile, naming
    the element and the fix. This is also why part 2 was the LOW-severity half: nothing ever misbuilt, only the place the author learned it
    moved.
    """
    from lite_step.compiler.executor import validate_project_report

    proj = Project(name="p")
    storey = Storey(name="ground")
    proj.add_storey(storey)
    # Past the guard, the way a patcher or a hand-built tree gets there.
    storey.elements.append(Window(name="w", width=800, height=1200))

    errors = validate_project_report(proj).errors
    assert any("needs a position" in e and "wall.anchor" in e for e in errors), \
        errors
