"""An ``IfcSpace`` DECOMPOSES its container; it is never CONTAINED in it.

``IfcRelContainedInSpatialStructure`` is the spatial-to-product relation, and
its **WR31** says outright:

    The relationship object shall not be used to include other spatial
    structure elements into a spatial structure element.

``IfcSpace`` *is* an ``IfcSpatialStructureElement``. The IFC 4.3 relation for a
space in a storey is ``IfcRelAggregates`` — the same one
``generator._emit_spatial_node`` already used for a facility under a site.

What was measured on ``main`` @ a7f41e9, in every container that can
hold a ``Space``, on both backends:

=================================  ================================  ==========
container                          relation emitted                  backends
=================================  ================================  ==========
``Storey``                         ``…ContainedInSpatialStructure``  both
``Site``                           ``…ContainedInSpatialStructure``  both
``SpatialElement`` facility part   ``…ContainedInSpatialStructure``  ifcopenshell
=================================  ================================  ==========

The facility-part row is not in the issue's scope list — it was found while
scoping this and is the same defect at a fifth call site. (Only the
ifcopenshell backend emits facilities; ``can_stream`` routes a model with a
``SpatialElement`` to it, so there is no streaming row to measure.)

Why nothing caught it
---------------------

``ifcopenshell.validate`` does not report WHERE rules — measured, with
controls, in ``tests/test_ifc43_conformance.py`` — so every one of those files
passed the conformance gate at **zero errors**. The corpus hash saw no drift
because the corpus was compiled by the same broken compiler on both sides. And
the symptom downstream is not an error but an absence, the shape: an
entity in the wrong relation is simply somewhere else in the inspector tree.

What is pinned here
-------------------

1.  The relation BY NAME, in all three containers, on both backends
    (:func:`test_a_space_is_aggregated_by_its_container`,
    :func:`test_a_space_is_not_in_any_containment_relation`). Not an entity
    count and not "the compile succeeded": both pass unchanged on the broken
    version, which is exactly why this bug stood.
2.  A CONTROL in the same file — an ordinary product under the same storey is
    still CONTAINED (:func:`test_an_ordinary_product_is_still_contained`), so
    a fix that stopped emitting containment altogether cannot read as green.
3.  Reachability from ``ifcopenshell.util.element.get_decomposition``
    (:func:`test_a_space_is_reachable_from_its_container`) — the lesson: the expensive failure of this class is invisibility in the spatial
    tree, not an error.
4.  The two backends land on the same relation
    (:func:`test_both_backends_put_the_space_in_the_same_relation`), asserted
    as a comparison. The other half was two backends disagreeing about
    what a container holds with nothing gating it.
5.  A ``Space`` inside a ``Space`` still aggregates
    (:func:`test_a_space_inside_a_space_aggregates_too`) — legal in IFC 4.3,
    and the one nesting that is already right.

Falsification is :func:`test_the_relation_assertions_catch_the_old_spelling`,
which stubs the shared predicate back to its pre-fix answer (everything is a
product) and asserts the relation assertions go red naming
``IfcRelContainedInSpatialStructure``.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc import facilities as fac
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Box, Point, Project, Site, Space, SpatialElement, Storey, Wall,
)

BACKENDS = ("ifcopenshell",)

#: Every container that can hold a ``Space`` and the backends that emit it.
#: ``facility_part`` is ifcopenshell-only by routing, not by omission:
#: ``streaming_generator._uses_spatial_elements`` sends any model carrying a
#: ``SpatialElement`` to the other backend, so a "streaming" run of that
#: fixture measures the ifcopenshell emitter and would assert nothing new.
CONTAINERS = {
    "storey": BACKENDS,
    "site": BACKENDS,
    "facility_part": ("ifcopenshell",),
}

#: The IFC class of the entity each container becomes.
CONTAINER_IFC_CLASS = {
    "storey": "IfcBuildingStorey",
    "site": "IfcSite",
    "facility_part": "IfcBridgePart",
}


def _body(name: str = "body") -> Box:
    return Box(name=name, start=Point(x=0, y=0, z=0),
               end=Point(x=3000, y=300, z=2400))


def _space(name: str = "kitchen") -> Space:
    return Space(name=name).add(_body())


def _wall() -> Wall:
    """The CONTROL — an ordinary product, in the same file as the space.

    Spelled as a ``Wall`` holding its own ``Box`` because that is the shape
    both backends emit natively into their storey loop; the point is only that
    it is not an ``IfcSpatialElement``.
    """
    return Wall(name="north").add(Box(name="body",
                                      start=Point(x=0, y=1000, z=0),
                                      end=Point(x=3000, y=1300, z=2400)))


def _project(container: str, *, with_control: bool = False) -> Project:
    proj = Project(name="spaceagg")
    if container == "storey":
        storey = Storey(name="ground")
        storey.add(_space())
        if with_control:
            storey.add(_wall())
        proj.add_storey(storey)
        return proj

    site = Site(name="plot")
    if container == "site":
        site.add(_space())
        if with_control:
            site.add(_wall())
    elif container == "facility_part":
        part = SpatialElement(ifc_class="IfcBridgePart", name="deck",
                              usage="VERTICAL")
        part.add(_space())
        if with_control:
            part.add(_wall())
        site.add(SpatialElement(ifc_class="IfcBridge", name="span").add(part))
    else:  # pragma: no cover - guards a typo in a parametrize list
        raise AssertionError(f"unknown container {container!r}")
    proj.add(site)
    return proj


def _model_of(proj: Project, backend: str, label: str):
    """Compile ``proj`` and hand back the opened model."""
    ifcopenshell = pytest.importorskip("ifcopenshell")
    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, (
        f"{label} on {backend} failed to compile: {result.error}")
    fd, path = tempfile.mkstemp(suffix=".ifc")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(result.ifc_content)
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _open_model(container: str, backend: str, *, with_control: bool = False):
    return _model_of(_project(container, with_control=with_control),
                     backend, container)


def _container_entity(model, container: str):
    entities = model.by_type(CONTAINER_IFC_CLASS[container])
    assert len(entities) == 1, (
        f"expected exactly one {CONTAINER_IFC_CLASS[container]}, got "
        f"{len(entities)} — this fixture is not measuring what it claims")
    return entities[0]


def _aggregated_classes(model, structure) -> set:
    return {o.is_a() for rel in model.by_type("IfcRelAggregates")
            if rel.RelatingObject == structure
            for o in (rel.RelatedObjects or [])}


def _contained_classes(model, structure) -> set:
    return {e.is_a()
            for rel in model.by_type("IfcRelContainedInSpatialStructure")
            if rel.RelatingStructure == structure
            for e in (rel.RelatedElements or [])}


# ---------------------------------------------------------------------------
# 1. The relation, by name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "container,backend",
    [(c, b) for c, backends in CONTAINERS.items() for b in backends])
def test_a_space_is_aggregated_by_its_container(container, backend):
    """``IfcRelAggregates`` holds it — asserted by relation name, not by count.

    An entity count passes on the broken version: the ``IfcSpace`` was always
    in the file, just wired to the wrong relation. So was ``result.success``,
    and so was ``ifcopenshell.validate`` at zero errors. Only the relation
    name can tell the two files apart.
    """
    model = _open_model(container, backend)
    structure = _container_entity(model, container)
    assert "IfcSpace" in _aggregated_classes(model, structure), (
        f"{container} on {backend}: the IfcSpace is not in any "
        f"IfcRelAggregates whose RelatingObject is the "
        f"{CONTAINER_IFC_CLASS[container]}. Aggregated classes: "
        f"{sorted(_aggregated_classes(model, structure))}")


@pytest.mark.parametrize(
    "container,backend",
    [(c, b) for c, backends in CONTAINERS.items() for b in backends])
def test_a_space_is_not_in_any_containment_relation(container, backend):
    """WR31, stated as the assertion it is.

    Scanned over EVERY ``IfcRelContainedInSpatialStructure`` in the file
    rather than only the container's own, because "aggregated here AND
    contained somewhere else" is a legal-looking file that gives one entity
    two parents — and the reader that honours the second one is undefined.
    """
    model = _open_model(container, backend)
    offenders = [
        (rel.RelatingStructure.is_a(), e.Name)
        for rel in model.by_type("IfcRelContainedInSpatialStructure")
        for e in (rel.RelatedElements or [])
        if e.is_a("IfcSpace")]
    assert not offenders, (
        f"{container} on {backend}: {offenders} — an IfcSpace is an "
        f"IfcSpatialStructureElement, and IfcRelContainedInSpatialStructure's "
        f"WR31 forbids one in RelatedElements. ifcopenshell.validate does not "
        f"report WHERE rules, so this file passes the conformance gate at "
        f"zero errors.")


@pytest.mark.parametrize(
    "container,backend",
    [(c, b) for c, backends in CONTAINERS.items() for b in backends])
def test_an_ordinary_product_is_still_contained(container, backend):
    """The CONTROL. Without it, "no space is contained" has a cheap fix.

    A change that stopped emitting ``IfcRelContainedInSpatialStructure``
    entirely would satisfy every assertion above. The wall in the same file
    is what makes them discriminating: it must still be CONTAINED, in the
    same container, in the same compile.
    """
    model = _open_model(container, backend, with_control=True)
    structure = _container_entity(model, container)
    contained = _contained_classes(model, structure)
    assert "IfcWall" in contained, (
        f"{container} on {backend}: the control wall is not contained "
        f"in the {CONTAINER_IFC_CLASS[container]} — contained classes are "
        f"{sorted(contained)}. A product under a spatial structure element "
        f"joins by containment; only a spatial element does not.")
    assert "IfcSpace" not in contained


# ---------------------------------------------------------------------------
# 2. Reachable in the spatial tree — the lesson
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "container,backend",
    [(c, b) for c, backends in CONTAINERS.items() for b in backends])
def test_a_space_is_reachable_from_its_container(container, backend):
    """Right relation is not enough; it has to be FINDABLE.

    ``get_decomposition`` follows both relations, so this passes on the broken
    version too — deliberately. It is here as the guard against a fix that
    moved the space into a relation nothing walks: the failure this class of
    bug actually costs is absence from the inspector tree, not an error, and
    an assertion on the relation name alone cannot see that.
    """
    _el = pytest.importorskip("ifcopenshell.util.element")
    model = _open_model(container, backend)
    structure = _container_entity(model, container)
    reachable = {e.is_a() for e in _el.get_decomposition(structure)}
    assert "IfcSpace" in reachable, (
        f"{container} on {backend}: the IfcSpace is not reachable from the "
        f"{CONTAINER_IFC_CLASS[container]} via get_decomposition — it is in "
        f"no relation the spatial walk follows, which is invisible to every "
        f"other gate we have. Reachable: {sorted(reachable)}")
# ---------------------------------------------------------------------------
# 4. Space inside Space — legal, and already right
# ---------------------------------------------------------------------------


def test_a_space_inside_a_space_aggregates_too():
    """IFC 4.3 composes an ``IfcSpace`` out of further ``IfcSpace`` s.

    Reached through ``.anchor()`` — ``Space.add()`` takes only the ``Box``
    that is the volume the space occupies. This nesting was already emitted as
    an aggregate before that (``_create_space`` aggregates its child
    products), and it is asserted here so the change above cannot quietly
    reroute it.
    """
    proj, storey = Project(name="nested"), Storey(name="ground")
    outer = Space(name="apartment").add(_body())
    outer.anchor(_space("kitchen"), along=500)
    storey.add(outer)
    proj.add_storey(storey)

    model = _model_of(proj, "ifcopenshell", "space-in-space")
    spaces = {s.Name: s for s in model.by_type("IfcSpace")}
    assert len(spaces) == 2, f"expected two IfcSpaces, got {sorted(spaces)}"
    outer_ent = next(s for n, s in spaces.items() if "apartment" in (n or ""))
    assert "IfcSpace" in _aggregated_classes(model, outer_ent), (
        "the inner space does not decompose the outer one")
    assert not [e for rel in model.by_type("IfcRelContainedInSpatialStructure")
                for e in (rel.RelatedElements or []) if e.is_a("IfcSpace")]


# ---------------------------------------------------------------------------
# 5. The predicate is asked of the EMITTED class, not the DSL type
# ---------------------------------------------------------------------------


def test_the_predicate_reads_both_backends_spelling_of_a_class():
    """``is_a()`` returns ``"IfcSpace"``; the streaming IR carries ``"IFCSPACE"``.

    One predicate serves both only if it folds the case, and it folds it in
    ``facilities`` rather than at each call site — a caller normalizing on its
    own is the second implementation that agrees *usually*.
    """
    for spelling in ("IfcSpace", "IFCSPACE", "ifcspace"):
        assert fac.is_spatial_element(spelling), spelling
    for spelling in ("IfcWall", "IFCWALL", "IfcGeographicElement",
                     "IFCBUILDINGELEMENTPROXY", "IfcProject"):
        assert not fac.is_spatial_element(spelling), spelling


def test_every_dedicated_spatial_class_has_a_container_that_names_it():
    """``DEDICATED_SPATIAL_CLASSES`` is read off ``DEDICATED_CONTAINERS``.

    Two hand-written lists of "the spatial classes with their own DSL
    container" would agree until one of them changed. ``IfcProject`` is the
    one key deliberately absent: it is an ``IfcContext``, not an
    ``IfcSpatialElement``, and calling it one would put the whole project into
    an aggregate branch it has always taken for its own reasons.
    """
    assert fac.DEDICATED_SPATIAL_CLASSES == (
        set(fac.DEDICATED_CONTAINERS) - {"IfcProject"})
    assert not fac.is_spatial_element("IfcProject")


# ---------------------------------------------------------------------------
# Falsification
# ---------------------------------------------------------------------------


def test_the_relation_assertions_catch_the_old_spelling(monkeypatch):
    """Stub the predicate back to "everything is a product" — the answer.

    Both backends read ``facilities.is_spatial_element`` by attribute at call
    time, so one patch restores the exact behaviour measured on ``main``. The
    compile still SUCCEEDS and the file still validates at zero errors, which
    is the whole reason the assertions above are written on relation names: if
    they were written on entity counts or on ``result.success`` they would
    fail to fail here.
    """
    monkeypatch.setattr(fac, "is_spatial_element", lambda ifc_class: False)

    model = _open_model("storey", "ifcopenshell")
    structure = _container_entity(model, "storey")
    assert "IfcSpace" not in _aggregated_classes(model, structure), (
        "the stub did not reach the emitter — this falsification is not "
        "measuring what it claims")
    assert "IfcSpace" in _contained_classes(model, structure), (
        "with the predicate stubbed the space should be back in "
        "IfcRelContainedInSpatialStructure, the relation WR31 forbids")
