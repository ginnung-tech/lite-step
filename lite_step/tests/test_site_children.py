"""Every product type under a ``Site`` reaches the file, on both backends.

``generate_ifc``'s project-level site pass was the ONE container in this
compiler with no generic child fallback. It carried its own branch
list — prism / ``Mesh`` / ``Element`` / ``SpatialElement`` — and its ``else``
warned and dropped, while every physical container ends in
``_emit_offtable_children`` -> ``_create_child_product``. Measured on ``main``,
with the ``.add()`` table bypassed so the EMITTER was what answered:

======================================  ==============  =============
child under a Site                      ifcopenshell    streaming
======================================  ==============  =============
Wall / Column / Beam / Slab / Roof      nothing         emitted
Space                                   COMPILE ERROR   emitted
Sweep / Revolve / Pipe / Bar            nothing         nothing
Box / Extrude / Mesh / Element          emitted         emitted
======================================  ==============  =============

Two defects, one cause. The drop is the first. The second is that the two
backends **disagreed** about what a ``Site`` holds, and nothing gated it —
which is why test 2 below is written as a COMPARISON of the two files rather
than as two independent counts. Half of this bug is only visible from that
angle.

The ``Space`` row is worth its own line: it did not drop quietly, it raised
``SpaceBoundaryError`` ("no emitted product is named ... for RelatingSpace").
The space-boundary walk reads the DSL tree and saw the Space; the product walk
read the file and did not. A model that put a Space on its site did not
compile at all.

What is pinned here:

1.  Every type on ``IfcSite``'s ``.add()`` row puts geometry in the file
    (:func:`test_every_site_child_emits_geometry`). The children come from
    ``test_anchor_matrix.CHILDREN`` — imported, not re-derived, because that
    module is already the answer to "one authored instance of every type" and
    a second list here would agree with it only until one of them changed.
2.  The two backends produce the same products for the same model
    (:func:`test_both_backends_agree_on_what_a_site_holds`) — asserted as an
    equality between the two entity histograms.
3.  A Site child is REACHABLE from ``ifcopenshell.util.element.
    get_decomposition`` (:func:`test_every_site_child_is_in_the_spatial_tree`).
    The lesson: the expensive failure is invisibility in the spatial
    tree, not an error — a product emitted into no relation at all validates
    at zero errors and is simply absent from every inspector. An entity count
    passes on that version; a reachability walk does not.
4.  What the ``.add()`` table PERMITS is exactly what emits
    (:func:`test_the_add_table_promises_nothing_the_emitter_drops`) — read off
    ``allowed_children_for(Site(...))`` rather than restated, so widening the
    row without the emitter behind it fails here.

Falsification is :func:`test_the_matrix_catches_the_dropped_child`, which
stubs the generic path back to its pre-fix behaviour and asserts the whole
matrix goes red. Without it, "geometry is in the file" would also be satisfied
by the terrain mesh the fixture puts there.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc import facilities as fac
from lite_step.ifc.generator import generate_ifc
from lite_step.models import Mesh, Point, Project, Site
from lite_step.models.elements import allowed_children_for

# The authored-instance-per-type table. Imported rather than rebuilt: it is
# the same question ``test_anchor_matrix`` asks of every container, and a
# private copy here would drift silently the first time a primitive's
# constructor changed.
from lite_step.tests.test_anchor_matrix import CHILDREN

#: Every product type the ``.add()`` table lets a ``Site`` hold. Derived from
#: the table, minus ``SpatialElement`` — a facility is a spatial NODE with its
#: own suite (``test_spatial_element.py``), joins the tree through
#: ``IfcRelAggregates`` rather than containment, and holds no geometry itself.
SITE_CHILDREN = tuple(sorted(
    name for name in allowed_children_for(Site(name="probe"))
    if name != "SpatialElement"
))


def _emitted_ifc_class(child_name: str) -> str:
    from lite_step.ifc.product_types import element_ifc_class

    return element_ifc_class(CHILDREN[child_name]())


#: The rows that emit an ``IfcSpatialElement`` of their own — today, ``Space``.
#: They join their site by ``IfcRelAggregates`` rather than containment (WR31,
#:), which is the ONE thing that differs between the two relation
#: tests below. Split by the emitter's own predicate rather than by a name
#: written here, so a new spatial row lands in the right test without an edit.
SPATIAL_SITE_CHILDREN = tuple(
    n for n in SITE_CHILDREN if fac.is_spatial_element(_emitted_ifc_class(n)))
PRODUCT_SITE_CHILDREN = tuple(
    n for n in SITE_CHILDREN if n not in SPATIAL_SITE_CHILDREN)


def _terrain() -> Mesh:
    """The terrain every real site carries — and the control for this file.

    Its geometry is in the file whatever the emitter does with the CHILD, so
    every assertion below is a DELTA against a site holding only this. Without
    it a broken emitter would still leave an ``IfcSite``, an ``IfcProject`` and
    a units chain to count.
    """
    return Mesh(
        name="terrain",
        vertices=[Point(x=0, y=0, z=0), Point(x=20000, y=0, z=0),
                  Point(x=20000, y=20000, z=0), Point(x=0, y=0, z=2000)],
        faces=[(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)],
    )


def _site_project(child_name: str | None) -> Project:
    proj, site = Project(name="sitechildren"), Site(name="plot")
    site.add(_terrain())
    if child_name is not None:
        site.add(CHILDREN[child_name]())
    proj.add(site)
    return proj


def _compile(child_name: str | None, backend: str):
    """The raw ``IFCGenerationResult``, success or not.

    Unasserted because a failure is a legitimate outcome for exactly one
    caller — the falsification below, where the emitter does not merely drop a ``Space`` but takes the whole
    compile down with it (``SpaceBoundaryError``: the boundary walk reads the
    DSL tree and finds the Space, the product walk reads the file and does
    not). Every other caller asserts success.
    """
    return generate_ifc(normalize_project_to_meters(_site_project(child_name)))


def _step(child_name: str | None, backend: str) -> str:
    result = _compile(child_name, backend)
    assert result.success, (
        f"Site.add({child_name}) failed to compile on {backend}: "
        f"{result.error}")
    return result.ifc_content


def _open_model(step: str):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    fd, path = tempfile.mkstemp(suffix=".ifc")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(step)
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


#: The schema SUPERTYPES that mean "this entity is a body". Written as
#: supertypes and resolved through ifcopenshell rather than as a list of leaf
#: class names, because a list is a second implementation of the schema that
#: agrees with it only until an emitter reaches for a class nobody listed. A
#: ``Mesh`` is the reason a naive scan is not enough: it emits an
#: ``IfcTriangulatedFaceSet`` and no ``IfcCartesianPoint`` at all (its vertices
#: ride a point LIST), so a scan that knew only swept solids would read
#: "dropped" for the Mesh row — the trap ``test_anchor_matrix`` documents.
_BODY_SUPERTYPES = ("IfcSolidModel", "IfcTessellatedItem", "IfcBooleanResult",
                    "IfcCsgPrimitive3D")


def _shape(step: str) -> dict:
    """``{entity class: count}`` for the PRODUCTS a file holds and their bodies.

    Deliberately not the whole entity histogram. The two backends differ in
    style and representation plumbing for reasons that predate this issue and
    have nothing to do with it — the streaming path emits an ``IfcStyledItem``
    for a terrain-reassigned prism where the ifcopenshell path's deduplicator
    does not — so a full-histogram comparison would fail on noise and say
    nothing about what the site CONTAINS. Products and bodies are exactly the
    claim is about.
    """
    model = _open_model(step)
    counts: dict = {}
    for product in model.by_type("IfcProduct"):
        counts[product.is_a()] = counts.get(product.is_a(), 0) + 1
    for supertype in _BODY_SUPERTYPES:
        for item in model.by_type(supertype):
            counts[item.is_a()] = counts.get(item.is_a(), 0) + 1
    return counts


def _delta(child_name: str, backend: str) -> dict:
    """What the CHILD added, over the same site holding only its terrain."""
    base = _shape(_step(None, backend))
    with_child = _shape(_step(child_name, backend))
    return {k: with_child.get(k, 0) - base.get(k, 0)
            for k in set(base) | set(with_child)
            if with_child.get(k, 0) - base.get(k, 0) > 0}


def _bodies(delta: dict) -> dict:
    """The body half of a delta — what actually renders."""
    schema = pytest.importorskip("ifcopenshell.ifcopenshell_wrapper").schema_by_name(
        "IFC4X3_ADD2")
    out = {}
    for name, count in delta.items():
        decl = schema.declaration_by_name(name)
        parents = set()
        while decl is not None:
            parents.add(decl.name())
            decl = decl.supertype()
        if parents & set(_BODY_SUPERTYPES):
            out[name] = count
    return out


# ---------------------------------------------------------------------------
# 1. Every type emits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("child_name", SITE_CHILDREN)
def test_every_site_child_emits_geometry(child_name):
    """The matrix. Not "it compiled" — a dropped child compiles perfectly."""
    delta = _delta(child_name, "ifcopenshell")
    bodies = _bodies(delta)
    assert bodies, (
        f"Site.add({child_name}) on {"ifcopenshell"}: the compiled IFC gained no "
        f"geometry entity at all over the same site without it — the child "
        f"was DROPPED. Full delta: {delta}. A Site whose pass has no "
        f"_emit_offtable_children fallback compiles clean and silent; that is "
        f"what this asserts against.")


# ---------------------------------------------------------------------------
# 2. The two backends agree — as a comparison, not two counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("child_name", SITE_CHILDREN)
def test_every_site_child_is_in_the_spatial_tree(child_name):
    """Emitted is not enough; it has to be FINDABLE from the IfcSite.

    A product minted into the wrong relation — or into none — still validates
    at zero errors (``ifcopenshell.validate`` does not report WHERE rules,
    measured in ``tests/test_ifc43_conformance.py``) and still counts in a
    histogram. What it does is vanish from ``getSpatialStructure()``: absent
    from the inspector tree, unselectable, unschedulable. That is,
    and it is why this walks rather than counts.
    """
    _el = pytest.importorskip("ifcopenshell.util.element")
    model = _open_model(_step(child_name, "ifcopenshell"))
    site = model.by_type("IfcSite")[0]
    reachable = {e.Name for e in _el.get_decomposition(site)}
    assert "mesh:terrain:site:plot" in reachable, (
        "the terrain control is not reachable either — this fixture is not "
        "measuring what it claims")
    others = reachable - {"mesh:terrain:site:plot", None}
    assert others, (
        f"Site.add({child_name}): nothing but the terrain is reachable from "
        f"the IfcSite via get_decomposition. The child is either in no "
        f"relation at all or in one the spatial walk does not follow.")


@pytest.mark.parametrize("child_name", PRODUCT_SITE_CHILDREN)
def test_a_site_child_is_CONTAINED_not_aggregated(child_name):
    """A Site is an ``IfcSpatialStructureElement``; its products are CONTAINED.

    ``IfcRelContainedInSpatialStructure`` is the spatial-to-product relation;
    ``IfcRelAggregates`` is spatial-to-spatial (and, one level down, a
    product's own parts). Getting it backwards is invisible to every gate we
    have — the file validates at zero errors — so the relation is asserted by
    name here rather than inferred from reachability, which both spellings
    satisfy.

    ``Space`` is not a row of this test: an ``IfcSpace`` is itself an
    ``IfcSpatialStructureElement``, so WR31 puts it on the OTHER side of the
    same rule. Its row is in
    :func:`test_a_spatial_site_child_is_AGGREGATED_not_contained`,
    with every other row here untouched.
    """
    model = _open_model(_step(child_name, "ifcopenshell"))
    site = model.by_type("IfcSite")[0]
    contained = {e.Name
                 for rel in model.by_type("IfcRelContainedInSpatialStructure")
                 if rel.RelatingStructure == site
                 for e in rel.RelatedElements}
    aggregated = {e.Name for rel in model.by_type("IfcRelAggregates")
                  if rel.RelatingObject == site for e in rel.RelatedObjects}
    assert len(contained) >= 2, (
        f"Site.add({child_name}): only {contained} is contained under the "
        f"IfcSite — the child did not join through the spatial-containment "
        f"relation.")
    assert not aggregated, (
        f"Site.add({child_name}): {aggregated} was AGGREGATED under the "
        f"IfcSite. A product under a spatial structure element is contained; "
        f"aggregation here is the shape — zero validation errors and "
        f"absent from getSpatialStructure().")


@pytest.mark.parametrize("child_name", SPATIAL_SITE_CHILDREN)
def test_a_spatial_site_child_is_AGGREGATED_not_contained(child_name):
    """The other side of the same rule — the ``Space`` row, inverted.

    ``IfcRelContainedInSpatialStructure``'s **WR31**: the relation "shall not
    be used to include other spatial structure elements into a spatial
    structure element". ``IfcSpace`` is one, so a ``Space`` on a ``Site``
    decomposes it. Emitting containment instead leaves
    ``ifcopenshell.validate`` reporting zero errors over that file — the conformance gate does not read WHERE rules, so the
    relation name here is the only thing that can tell the two files apart.

    The terrain control still has to be CONTAINED in the same file, which is
    what stops "the site aggregates everything" from reading as green.
    """
    model = _open_model(_step(child_name, "ifcopenshell"))
    site = model.by_type("IfcSite")[0]
    contained = {e.Name
                 for rel in model.by_type("IfcRelContainedInSpatialStructure")
                 if rel.RelatingStructure == site
                 for e in rel.RelatedElements}
    aggregated = {e.Name for rel in model.by_type("IfcRelAggregates")
                  if rel.RelatingObject == site for e in rel.RelatedObjects}
    assert aggregated, (
        f"Site.add({child_name}) emits an IfcSpatialElement, which joins its "
        f"site by IfcRelAggregates — and nothing is aggregated under the "
        f"IfcSite. Contained instead: {sorted(contained)}.")
    assert "mesh:terrain:site:plot" in contained, (
        "the terrain control is not CONTAINED — this fixture is not "
        "measuring what it claims, and 'aggregate everything' would pass")
    assert aggregated.isdisjoint(contained), (
        f"Site.add({child_name}): {sorted(aggregated & contained)} is in BOTH "
        f"relations — two parents, and which one a reader honours is "
        f"undefined.")


@pytest.mark.parametrize("child_name", SITE_CHILDREN)
def test_a_site_child_validates(child_name):
    """``ifcopenshell.validate`` at zero errors on every new pairing.

    ``tests/test_ifc43_conformance.py`` is the corpus gate, and the corpus
    reaches none of this: not one committed model puts an off-table child on
    its site (measured — the whole corpus is byte-identical across this
    change), so a new emission path here would ship unvalidated. This carries
    its own proof, the ``test_product_geometry_sharing.py`` precedent.

    A floor, not a ceiling: the validator does NOT report WHERE rules, so it
    would pass an aggregated-under-a-site product too. That is what
    :func:`test_a_site_child_is_CONTAINED_not_aggregated` is for.
    """
    validate = pytest.importorskip("ifcopenshell.validate")
    model = _open_model(_step(child_name, "ifcopenshell"))
    logger = validate.json_logger()
    validate.validate(model, logger)
    errors = [s for s in logger.statements if s.get("level") == "error"]
    assert not errors, (
        f"Site.add({child_name}) on {"ifcopenshell"}: {len(errors)} IFC 4.3 "
        f"validation error(s), first: {errors[0]}")


# ---------------------------------------------------------------------------
# 4. The table promises exactly what the emitter delivers
# ---------------------------------------------------------------------------


def test_the_add_table_promises_nothing_the_emitter_drops():
    """``IfcSite``'s ``.add()`` row and the emitter's reach are one claim.

    The row was deliberately narrow until the emitter existed, because a table
    that accepts a child the generator drops trades a clear compile-time
    refusal for a silent hole — the failure this repo ranks below a crash.
    Read off ``allowed_children_for`` rather than restated, so widening the
    row on its own turns the matrix above red instead of shipping green.
    """
    row = allowed_children_for(Site(name="probe"))
    assert set(SITE_CHILDREN) | {"SpatialElement"} == set(row)
    # Every product type is on it — the §1.2 asymmetry (Element(ifc_class=
    # "IfcWall") renders under a Site, Site.add(Wall) is refused) is gone.
    for name in ("Wall", "Column", "Beam", "Slab", "Roof", "Space",
                 "Sweep", "Revolve", "Pipe", "Bar", "Box", "Extrude",
                 "Mesh", "Element"):
        assert name in row, f"{name} is missing from IfcSite's .add() row"


def test_the_escape_hatch_and_the_product_agree():
    """§1.2: ``Element(ifc_class="IfcWall")`` and ``Wall`` under a Site are
    the same file.

    This asymmetry is the reason was filed as a bug rather than a
    missing feature — the escape hatch already rendered, so the refusal was
    protecting nothing an author could not route around by spelling it
    differently.
    """
    from lite_step.models import Box, Element, Wall

    def _body():
        return Box(name="body", start=Point(x=0, y=0, z=0),
                   end=Point(x=3000, y=300, z=2400))

    proj_a, site_a = Project(name="hatch"), Site(name="plot")
    site_a.add(_terrain())
    site_a.add(Element(ifc_class="IfcWall", name="w").add(_body()))
    proj_a.add(site_a)

    proj_b, site_b = Project(name="hatch"), Site(name="plot")
    site_b.add(_terrain())
    site_b.add(Wall(name="w").add(_body()))
    proj_b.add(site_b)

    hist_a = _shape(generate_ifc(normalize_project_to_meters(proj_a)).ifc_content)
    hist_b = _shape(generate_ifc(normalize_project_to_meters(proj_b)).ifc_content)
    assert hist_a.get("IfcWall") == hist_b.get("IfcWall") == 1
    assert (hist_a.get("IfcExtrudedAreaSolid")
            == hist_b.get("IfcExtrudedAreaSolid") == 1)


# ---------------------------------------------------------------------------
# Falsification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("child_name", ["Wall", "Column", "Sweep", "Pipe",
                                        "Bar", "Revolve", "Space"])
def test_the_matrix_catches_the_dropped_child(monkeypatch, child_name):
    """Stub the generic path back to the earlier behaviour; the matrix reddens.

    ``_create_child_product`` returning ``None`` is exactly the drop shape to
    guard against: ``_emit_offtable_children`` logs a warning and carries on,
    and the compile still SUCCEEDS. If
    :func:`test_every_site_child_emits_geometry` asserted "no exception" it
    would fail to fail, which is the whole reason this exists.

    The terrain control is what makes it discriminating in the other
    direction: it keeps geometry in the file, so a green assertion up there
    cannot be explained by "there is a solid somewhere".
    """
    from lite_step.ifc import generator

    assert generator._create_child_product.__module__ == generator.__name__
    monkeypatch.setattr(generator, "_create_child_product",
                        lambda *a, **k: None)

    stubbed = _compile(child_name, "ifcopenshell")
    if not stubbed.success:
        # The LOUDEST form of the drop, and the one main actually shipped for
        # a Space: the space-boundary walk finds the Space in the DSL tree,
        # the product walk does not find it in the file, and the compile dies
        # rather than emitting. Nothing rendered, which is the claim.
        assert "no emitted product is named" in (stubbed.error or "")
        return
    base = _shape(_step(None, "ifcopenshell"))
    with_child = _shape(stubbed.ifc_content)
    delta = {k: with_child.get(k, 0) - base.get(k, 0)
             for k in set(base) | set(with_child)
             if with_child.get(k, 0) - base.get(k, 0) > 0}
    bodies = _bodies(delta)
    assert not bodies, (
        f"Site.add({child_name}): the stub did not drop the child ({bodies}), "
        f"so this row's assertion is not exercising the path it claims to.")
