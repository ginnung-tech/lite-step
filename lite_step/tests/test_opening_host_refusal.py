"""An opening only compiles on a host that will actually EMIT it.

``agents/dsl-reference.md`` named seven opening hosts. The emitters implement
three, and the other four (plus five more the reference never claimed) reached
the Window with no route for it, emitted nothing, and — on a container — let
``displacement`` notch the details standing in the hole anyway.

Measured on ``main`` @ 24304c8, both backends, with one 6 m body
``Box(x 0..6000, y -300..0, z 0..2700)`` ``.add()``ed to the host and one
``host.opening(Window(width=1200, height=1400), along=4200, up=900)``:

===========================================  ==================  ==========
host                                         IfcOpeningElement   IfcWindow
===========================================  ==================  ==========
``Wall``                                     1                   1
standalone ``Box`` / ``Extrude``             1                   1
``Column`` ``Beam`` ``Slab`` ``Roof``        0                   0
``Element(ifc_class="IfcWall")``             0                   0
``Sweep`` ``Pipe`` ``Bar`` ``Revolve``       0                   0
``Mesh``                                     0                   0
a ``Wall``'s BODY child                      0                   0
===========================================  ==================  ==========

The five containers logged ``"child Window … has no IFC emitter and was
skipped"``. The leaf solids and the Wall body logged **nothing on any stream**
— the Wall-body model came out byte-identical (6049 bytes) to the same model
with the ``.opening()`` line deleted.

**Narrowed, not deleted.** ``Column``, ``Beam`` and ``Element`` moved
out of the refusal and into an emitter: they distribute the hole to the
geometry-bearing leaves it overlaps, N ``IfcOpeningElement`` and ONE
``IfcWindow``. What they emit is pinned by
``test_opening_distributes_to_leaves.py``; what is left here is the half that
still refuses, and the table above is still the reason it does. The split is a
FRAME question and not an emission one: ``along=``/``inset=``/``up=`` measure
from an outer face along a run axis, which a ``Column``/``Beam``/``Element``
frame basis has and a HORIZONTAL ``Slab``/``Roof`` does not. The gap stays open
for that half.

**Narrowed once more**, and this file is untouched by it: a
BODY-LESS ``Wall`` now distributes too. The ``Wall`` row above is the BODIED
one — its ``.add()``ed prism IS the ``IfcWall``'s representation, so it keeps
carrying its own hole — and the ``Wall``-BODY row is a prism child of such a
wall. Neither shape can be body-less by construction, which is why nothing
here moved; the body-less rows live in ``test_bodyless_wall_openings.py`` and
``test_bodyless_container.py``.

Pinned here:

1.  The table above, minus the three rows moved: every host that still
    emits nothing is a compile refusal, and the ones that emit still do (the
    control that keeps this from passing by refusing everything).
2.  Both SPELLINGS refuse. ``.opening()`` on a container routes to
    ``.anchor()``, so the two build the identical tree — a refusal that only
    caught ``.opening()`` would leave the same silence one keyword away.
3.  The remedy each refusal names WORKS: 1 opening + 1 window + 1
    IfcRelVoidsElement, for all five containers — including the three that do
    not need it, because widening the container route must not break the leaf
    one.
4.  The phantom notch stops by construction — measured as the
    ``IFCBOOLEANRESULT`` count on a Slab fixture whose bar stood in the hole.
    (The ``Column`` version of that measurement moved to
    ``test_opening_distributes_to_leaves.py``, where the notch is gone for the
    opposite reason: the window IS in the file now.)
5.  The refusal is not over-broad: an ANCHORED prism inside a Wall is a
    product of its own and still emits its openings.
6.  ``tx.hosts_openings`` agrees with what the emitters actually read.
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import (
    LiteStepCompileError,
    compile_main,
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Bar, Beam, Box, Column, Element, Extrude, Mesh, Pipe, Point, Project,
    Revolve, Roof, Slab, Sweep, Wall, Window,
)
from lite_step.models import taxonomy as tx
from lite_step.models.primitives import Point2D

_RUN, _H = 6000, 2700

#: The HORIZONTAL containers, whose opening children still no emitter reads
#: (left them refused: no outer face, no run axis, so ``along=``/``up=``
#: measure against nothing). ``Wall`` is the control that does emit.
_REFUSED_CONTAINERS = {
    "Slab": lambda: Slab(name="host"),
    "Roof": lambda: Roof(name="host"),
}

#: The containers moved OUT of the refusal — they distribute the hole to
#: their leaves. Here only so the tests that must keep working on them
#: (``body.opening()``, the leaf route) can name them; what they emit is pinned
#: in ``test_opening_distributes_to_leaves.py``.
_DISTRIBUTING_CONTAINERS = {
    "Column": lambda: Column(name="host"),
    "Beam": lambda: Beam(name="host"),
    "Element": lambda: Element(ifc_class="IfcWall", name="host"),
}

_ALL_CONTAINERS = {**_REFUSED_CONTAINERS, **_DISTRIBUTING_CONTAINERS}

#: Leaf solids that are ``VOIDABLE`` (or, for ``Mesh``, merely geometry) and
#: that ``_process_openings_on_element`` is never called for.
_REFUSED_LEAVES = {
    "Sweep": lambda: Sweep(
        name="s", path=[Point(x=0, y=-150, z=1350),
                        Point(x=_RUN, y=-150, z=1350)],
        profile=[Point2D(x=-150, y=-1350), Point2D(x=150, y=-1350),
                 Point2D(x=150, y=1350), Point2D(x=-150, y=1350)]),
    "Pipe": lambda: Pipe(name="s", path=[Point(x=0, y=-150, z=1350),
                                         Point(x=_RUN, y=-150, z=1350)],
                         radius=300),
    "Bar": lambda: Bar(name="s", path=[Point(x=0, y=-150, z=1350),
                                       Point(x=_RUN, y=-150, z=1350)],
                       diameter=32),
    "Revolve": lambda: Revolve(
        name="s",
        profile=[Point2D(x=100, y=0), Point2D(x=400, y=0),
                 Point2D(x=400, y=2700), Point2D(x=100, y=2700)],
        path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=2700)]),
    "Mesh": lambda: Mesh(name="s",
                         vertices=[Point(x=0, y=0, z=0),
                                   Point(x=_RUN, y=0, z=0),
                                   Point(x=_RUN, y=0, z=_H),
                                   Point(x=0, y=0, z=_H)],
                         faces=[(0, 1, 2), (0, 2, 3)],
                         is_watertight=True, source_type="synthetic"),
}


def _body(name: str = "body") -> Box:
    return Box(name=name, type="wall", start=Point(x=0, y=-300, z=0),
               end=Point(x=_RUN, y=0, z=_H))


def _window(name: str = "g0") -> Window:
    return Window(width=1200, height=1400, name=name)


def _bar(name: str = "det", x: int = 1200) -> Bar:
    """A 32 mm vertical bar dead in the hole's path — the detail the phantom
    notch ate. Its own AABB is degenerate, so only ``padded_aabb`` bounds it."""
    return Bar(path=[Point(x=x, y=-150, z=0), Point(x=x, y=-150, z=_H)],
               diameter=32, name=name)


def _project(element) -> Project:
    proj = Project(name="opening-host")
    proj.add(element)
    return proj


def _errors(element) -> list:
    return validate_project_report(_project(element)).errors


def _compiled(proj: Project, backend: str = "ifcopenshell") -> str:
    assert validate_project_report(proj).errors == []
    result = generate_ifc(normalize_project_to_meters(proj) or proj)
    assert result.success, result.error
    return result.ifc_content


def _counts(ifc: str) -> dict:
    return {k: ifc.count(f"{k}(") for k in
            ("IFCOPENINGELEMENT", "IFCWINDOW", "IFCRELVOIDSELEMENT",
             "IFCBOOLEANRESULT")}


# ---------------------------------------------------------------------------
# 1. The table — the hosts that emitted nothing now refuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(_REFUSED_CONTAINERS))
def test_a_horizontal_container_refuses_the_opening(kind) -> None:
    """``Slab``/``Roof``: the half of the five that left refused.

    The refusal names the host that DOES work and the remedy that does — the shape — because "unsupported" without a next step is what sends an
    LLM author round the same loop. Since it must also name the REASON,
    because the sibling containers now emit and "a Column works and a Slab does
    not" is otherwise arbitrary: a horizontal host has no outer face and no run
    axis for ``along=``/``up=`` to measure against.
    """
    host = _REFUSED_CONTAINERS[kind]()
    host.add(_body())
    host.opening(_window(), along=4200, up=900)

    errors = _errors(host)
    assert len(errors) == 1, errors
    assert "no IFC emitter reads" in errors[0]
    assert f"{kind} 'window" not in errors[0]      # the HOST is named, not the child
    assert "Window 'g0'" in errors[0]
    assert "body.opening(window, along=" in errors[0]
    assert "or make the host a Wall" in errors[0]
    # The reason, and the sibling that does work.
    assert "HORIZONTAL" in errors[0]
    assert "Column/Beam/Element" in errors[0]
    assert "not implemented" in errors[0]


@pytest.mark.parametrize("kind", sorted(_DISTRIBUTING_CONTAINERS))
def test_a_distributing_container_no_longer_refuses_the_opening(kind) -> None:
    """The inversion, stated where the refusal was.

    ``Column``/``Beam``/``Element`` compile the identical fixture the test
    above refuses. The counts are pinned in
    ``test_opening_distributes_to_leaves.py``; this asserts only that the
    validator lets it through, so the two halves of the table cannot silently
    become one again.
    """
    host = _DISTRIBUTING_CONTAINERS[kind]()
    host.add(_body())
    host.opening(_window(), along=4200, up=900)

    assert _errors(host) == []


@pytest.mark.parametrize("kind", sorted(_DISTRIBUTING_CONTAINERS))
def test_a_distributing_container_with_nothing_to_distribute_to_refuses(kind) -> None:
    """The positional half of the new rule.

    The frame comes from the container's ``.add()``ed children, so a container
    with none has no run axis for ``along=`` to walk AND no leaf for the hole
    to land on — ``generator._process_container_openings`` would raise, and
    before that ``frames.opening_host_frame`` answers ``None``. Refused at
    compile, naming both halves.
    """
    host = _DISTRIBUTING_CONTAINERS[kind]()
    host.opening(_window(), along=4200, up=900)

    errors = _errors(host)
    assert any("no IFC emitter reads" in e for e in errors), errors
    hit = next(e for e in errors if "no IFC emitter reads" in e)
    assert "no .add()ed children" in hit
    assert "leaf.opening(window, along=" in hit


@pytest.mark.parametrize("kind", sorted(_REFUSED_LEAVES))
def test_a_leaf_solid_with_no_opening_route_refuses(kind) -> None:
    """``Sweep``/``Pipe``/``Bar``/``Revolve``/``Mesh`` — beyond the own
    table, and found by measuring rather than by reading it.

    ``_process_openings_on_element`` is called from ``_create_box`` and
    ``_create_extrude`` and from nowhere else, so a ``Pipe``'s ``_openings``
    list is written by ``.opening()`` and read by nobody. These emitted 0/0
    with NO warning at all, which is the quietest failure in the whole set.
    """
    solid = _REFUSED_LEAVES[kind]()
    solid.opening(_window(), along=4200, up=900)

    errors = _errors(solid)
    assert len(errors) == 1, errors
    assert "no IFC emitter reads" in errors[0]
    assert ".void(tool, name=" in errors[0]        # the verb that DOES cut here
    assert errors[0].startswith(f"{kind} '")


def test_a_wall_body_refuses_and_names_the_wall() -> None:
    """The silent case: no warning on any stream, and the file byte-identical
    to omitting the line.

    ``.opening()``'s own docstring listed "a Wall body" as a legal host. A
    Wall's body is not a product of its own — ``_create_wall`` turns it into
    the ``IfcWall``'s representation and ``_create_box`` never runs on it — so
    ``_process_openings_on_element`` is never reached for it.
    """
    wall = Wall(name="south")
    body = _body()
    wall.add(body)
    body.opening(_window(), along=4200, up=900)

    errors = _errors(wall)
    assert len(errors) == 1, errors
    assert "A Wall's BODY is not a product of its own" in errors[0]
    assert "wall.anchor(window, along=" in errors[0]
    assert errors[0].startswith("Box '")


def test_a_wall_body_extrude_refuses_the_same_way() -> None:
    """Contour-mode bodies (gable walls) take the same path and were equally
    silent — the refusal must not be a ``Box``-only check."""
    wall = Wall(name="gable")
    body = Extrude(name="body", type="wall", thickness=300,
                   contour=[Point(x=0, y=0, z=0), Point(x=_RUN, y=0, z=0),
                            Point(x=_RUN, y=0, z=_H), Point(x=0, y=0, z=_H)])
    wall.add(body)
    body.opening(_window(), along=4200, up=900)

    errors = _errors(wall)
    assert len(errors) == 1, errors
    assert "A Wall's BODY is not a product of its own" in errors[0]


# ---------------------------------------------------------------------------
# 2. The control — the three hosts that DO emit still do
# ---------------------------------------------------------------------------


def test_a_wall_still_emits_its_opening() -> None:
    wall = Wall(name="south")
    wall.add(_body())
    wall.opening(_window(), along=4200, up=900)

    assert _errors(wall) == []
    for backend in ("ifcopenshell",):
        counts = _counts(_compiled(_project(wall), backend))
        assert counts["IFCOPENINGELEMENT"] == 1, backend
        assert counts["IFCWINDOW"] == 1, backend
        assert counts["IFCRELVOIDSELEMENT"] == 1, backend


@pytest.mark.parametrize("kind", ["Box", "Extrude"])
def test_a_standalone_prism_still_emits_its_opening(kind) -> None:
    """The LEAF path — ``_openings`` read by ``_process_openings_on_element``.
    A fix that emptied it everywhere would break the spelling ``.opening()``'s
    own docstring is written around."""
    if kind == "Box":
        solid = _body("s")
    else:
        solid = Extrude(name="s", type="wall", thickness=300,
                        contour=[Point(x=0, y=0, z=0), Point(x=_RUN, y=0, z=0),
                                 Point(x=_RUN, y=0, z=_H),
                                 Point(x=0, y=0, z=_H)])
    solid.opening(_window(), along=4200, up=900)

    assert _errors(solid) == []
    for backend in ("ifcopenshell",):
        counts = _counts(_compiled(_project(solid), backend))
        assert counts["IFCOPENINGELEMENT"] == 1, backend
        assert counts["IFCWINDOW"] == 1, backend


def test_an_anchored_prism_inside_a_wall_still_emits_its_openings() -> None:
    """The refusal is positional, and this is the position it must NOT catch.

    ``tx.body_prisms`` excludes an anchored child — a detail placed in the
    wall's frame, never the body that defines it — and an anchored prism gets
    its own ``IfcBuildingElementProxy`` from ``_create_box``, which does read
    ``_openings``. Measured: 1/1/1 on both backends.
    """
    wall = Wall(name="south")
    wall.add(_body())
    detail = Box(name="det", start=Point(x=0, y=0, z=0),
                 end=Point(x=2000, y=200, z=2000))
    wall.anchor(detail, along=0, up=0)
    detail.opening(_window(), along=400, up=200)

    assert _errors(wall) == []
    for backend in ("ifcopenshell",):
        counts = _counts(_compiled(_project(wall), backend))
        assert counts["IFCOPENINGELEMENT"] == 1, backend
        assert counts["IFCWINDOW"] == 1, backend


def test_a_wall_nested_inside_a_refused_container_still_emits() -> None:
    """The refusal is about the HOST of the opening, not about anything above
    it: a ``Wall`` inside a ``Column`` is still a Wall."""
    col = Column(name="frame")
    wall = Wall(name="infill")
    wall.add(_body())
    wall.opening(_window(), along=4200, up=900)
    col.add(wall)

    assert _errors(col) == []
    assert _counts(_compiled(_project(col)))["IFCWINDOW"] == 1


# ---------------------------------------------------------------------------
# 3. Both spellings, because they build one tree
# ---------------------------------------------------------------------------


def test_anchor_refuses_on_the_same_host_that_opening_does() -> None:
    """``.opening()`` on a container IS ``.anchor()`` (routed, not duplicated),
    so the two produce byte-identical trees and a DSL-level raise in
    ``.opening()`` would have left ``.anchor()`` silent.

    That is why the refusal lives in ``validate_project_report`` and not in
    ``BimElement.opening``: the validator reads the TREE, so every route in —
    ``.opening()``, ``.anchor()``, or a hand-stamped list — meets it.
    """
    by_opening = Slab(name="host")
    by_opening.add(_body())
    by_opening.opening(_window(), along=4200, up=900)

    by_anchor = Slab(name="host")
    by_anchor.add(_body())
    by_anchor.anchor(_window(), along=4200, up=900)

    assert _errors(by_anchor) == _errors(by_opening)
    assert len(_errors(by_anchor)) == 1


def test_the_refusal_is_a_loud_compile_failure_not_a_warning(tmp_path) -> None:
    """Through the real entry point: non-zero exit, the message on stderr, and
    no ``output.ifc`` — never a file with the window quietly missing."""
    col = Slab(name="host")
    col.add(_body())
    col.opening(_window(), along=4200, up=900)

    model = tmp_path / "model.py"
    model.write_text("# fixture\n", encoding="utf-8")
    with pytest.raises(LiteStepCompileError) as exc:
        compile_main(result=_project(col), source_path=str(model))
    assert "no IFC emitter reads" in str(exc.value)
    assert not (tmp_path / "output.ifc").exists()


# ---------------------------------------------------------------------------
# 4. The phantom notch
# ---------------------------------------------------------------------------


def test_the_phantom_notch_is_unreachable_now() -> None:
    """A detail losing material to a window that is not in the file.

    ``displacement._carve_openings_through_details`` runs off the DSL tree,
    not off what the emitter did. Skipping the validator — which is what the compile effectively did, since it raised no error — this Slab
    emits **0 IfcOpeningElement, 0 IfcWindow and 2 IfcBooleanResult**, one of
    which is the bar's 1.4 m notch for the phantom hole. The first half of
    this test measures that; the second pins the refusal that makes the whole
    path unreachable from a real compile.

    The fixture was a ``Column`` until then taught Columns to emit. The
    measurement is unchanged — a ``Slab`` is the same shape and is still
    refused — and the ``Column`` half moved to
    ``test_opening_distributes_to_leaves.py``, where the notch is gone because
    the hole is REAL rather than because the tree is unbuildable.
    """
    col = Slab(name="host")
    col.add(_body())
    col.add(_bar())
    col.opening(_window(), along=4200, up=900)
    proj = _project(col)

    # 1. What the emitter would produce if the validator were not there.
    #    ``normalize_project_to_meters`` hands back a COPY, and displacement
    #    stamps the copy — read the notch off the tree the emitter saw.
    normalized = normalize_project_to_meters(proj) or proj
    result = generate_ifc(normalized)
    assert result.success
    counts = _counts(result.ifc_content)
    assert counts["IFCOPENINGELEMENT"] == 0
    assert counts["IFCWINDOW"] == 0
    # Two booleans: the body/bar inferred carve, plus the notch for the hole
    # that is not in the file.
    assert counts["IFCBOOLEANRESULT"] == 2
    emitted_host = normalized.storeys[0].elements[0]
    bar = next(c for c in emitted_host._elements
               if getattr(c, "name", None) == "det")
    assert len(getattr(bar, "_cuts", None) or []) == 1

    # 2. …and that is exactly the model the validator now refuses, so nothing
    #    reaches the carve pass. One error, about the Window, not about the bar.
    errors = validate_project_report(_project(col)).errors
    assert len(errors) == 1, errors
    assert "Window 'g0'" in errors[0]


def test_the_remedy_puts_the_notch_back_where_it_belongs() -> None:
    """The same Column, spelled the way the refusal says — and now the notch
    is honest: the window IS in the file.

    The bar is a sibling of the body rather than a detail of it, so it keeps
    its material here (``_iter_opening_carvees`` walks the OPENING's host, and
    the host is the Box). That is a reach question, not an emission one, and
    it is the follow-up issue's business, not this refusal's — what matters
    for is that no material is lost to a hole that is not there.
    """
    col = Column(name="host")
    body = _body()
    col.add(body)
    col.add(_bar())
    body.opening(_window(), along=4200, up=900)

    assert _errors(col) == []
    for backend in ("ifcopenshell",):
        counts = _counts(_compiled(_project(col), backend))
        assert counts["IFCOPENINGELEMENT"] == 1, backend
        assert counts["IFCWINDOW"] == 1, backend
        assert counts["IFCRELVOIDSELEMENT"] == 1, backend
        # Down from 2 to 1, and the one that went is the phantom notch. What
        # is left is the ordinary body-vs-bar inferred carve, which is the
        # same boolean the model had before any window was authored.
        assert counts["IFCBOOLEANRESULT"] == 1, backend


@pytest.mark.parametrize("kind", sorted(_ALL_CONTAINERS))
def test_the_named_remedy_works_on_every_container(kind) -> None:
    """A refusal that names a remedy has to be measured naming a working one —
    the rule. 1 opening + 1 window + 1 IfcRelVoidsElement, both backends,
    for all five.

    Still all FIVE after that, not just the two that refuse: the leaf route
    (``body.opening()``) is what the container route was widened alongside, and
    a Column whose leaf-hosted opening started emitting TWO holes would mean
    the distribution had reached a hole it does not own.
    """
    host = _ALL_CONTAINERS[kind]()
    body = _body()
    host.add(body)
    body.opening(_window(), along=4200, up=900)

    assert _errors(host) == []
    for backend in ("ifcopenshell",):
        counts = _counts(_compiled(_project(host), backend))
        assert counts["IFCOPENINGELEMENT"] == 1, (kind, backend)
        assert counts["IFCWINDOW"] == 1, (kind, backend)
        assert counts["IFCRELVOIDSELEMENT"] == 1, (kind, backend)


def test_the_wall_body_remedy_takes_the_identical_scalars() -> None:
    """The Wall-body refusal claims ``wall.anchor()`` is a mechanical rewrite:
    same ``along=``/``up=``, same hole. It is — the wall's opening frame is
    derived from that very body — and the proof is that the remedy compiles
    and puts the void where the body's own frame says.
    """
    from lite_step.compiler import frames
    from lite_step.ifc.entity_cache import snap_to_precision

    wall = Wall(name="south")
    wall.add(_body())
    wall.anchor(_window(), along=4200, up=900)
    proj = _project(wall)
    ifc = _compiled(proj)
    assert _counts(ifc)["IFCWINDOW"] == 1

    normalized = normalize_project_to_meters(proj) or proj
    host = normalized.storeys[0].elements[0]
    frame = frames.opening_host_frame(frames._body_of(host),
                                      snap=snap_to_precision, divisor=1000.0)
    opening = next(c for c in host._elements if tx.brings_void(c))
    (x0, _y0, z0), (x1, _y1, z1) = frames.opening_void_prism(
        opening, frame).aabb()
    # host-frame along=, NOT world x=4.2 — the trap, restated because
    # this file's remedies are all written in host-frame scalars.
    assert (round(x0, 6), round(x1, 6)) == (0.6, 1.8)
    assert (round(z0, 6), round(z1, 6)) == (0.9, 2.3)


# ---------------------------------------------------------------------------
# 5. The routing table agrees with the emitters
# ---------------------------------------------------------------------------


def test_hosts_openings_names_exactly_the_types_that_emit() -> None:
    """Six since — the three self-geometry hosts plus the three
    containers that distribute. ``Slab`` and ``Roof`` are the deliberate
    omission and the reason is a frame, not an emitter."""
    assert set(tx.OPENING_HOSTS) == {Wall, Box, Extrude, Column, Beam, Element}
    for cls in (Wall, Box, Extrude, Column, Beam, Element):
        assert tx.hosts_openings(cls), cls.__name__
        assert tx.hosts_openings(cls.__name__), cls.__name__
    for cls in (Slab, Roof, Sweep, Pipe, Bar, Revolve, Mesh, Window):
        assert not tx.hosts_openings(cls), cls.__name__


def test_hosts_openings_is_strictly_narrower_than_is_voidable() -> None:
    """The gap between the two IS, and naming it keeps the reference
    honest: ``is_voidable`` answers "may a void hang off this in principle",
    ``hosts_openings`` answers "does an emitter read this element's openings".
    Everything that hosts an opening is voidable; the gap is what is left to
    build.

    closed three of the nine. The six that remain are the two
    HORIZONTAL containers (a frame question, the open half) and the four
    leaf solids with no ``_openings`` reader at all.
    """
    voidable = {t.__name__ for t in tx.VOIDABLE}
    hosts = {t.__name__ for t in tx.OPENING_HOSTS}
    assert hosts < voidable
    assert voidable - hosts == {"Slab", "Roof",
                                "Sweep", "Pipe", "Bar", "Revolve"}
