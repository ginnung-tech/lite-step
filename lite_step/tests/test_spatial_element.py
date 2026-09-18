"""``SpatialElement`` — the IFC 4.3 spatial vocabulary as a DSL container.

Spec: ``infrastructure/specs/ifc43-facilities.md``.

The first test is the important one. ``lite_step`` compiles without
ifcopenshell at runtime, so ``ifc/facilities.py``'s sets are enumerated by
hand — and a hand-maintained mirror of a schema is exactly the thing that goes
stale silently. ``test_the_sets_match_the_schema`` derives every one of them
from ``IFC4X3_ADD2`` and fails when they disagree, so the rule stays derived
even though the data is static.
"""

from __future__ import annotations

import pytest

from lite_step.ifc import facilities as fac
from lite_step.models.elements import Site, SpatialElement, Box, Point, Material
from lite_step.models.project import Project


# ── the mirror stays true ──────────────────────────────────────────────────


def _schema():
    ios = pytest.importorskip("ifcopenshell")
    from ifcopenshell import ifcopenshell_wrapper as w
    return w.schema_by_name("IFC4X3_ADD2")


def _concrete_subtypes_of(schema, root: str) -> set:
    out = set()
    for d in schema.declarations():
        if not hasattr(d, "supertype"):
            continue
        try:
            if d.is_abstract():
                continue
            chain, x = [], d
            while x:
                chain.append(x.name())
                x = x.supertype()
            if root in chain[1:]:
                out.add(d.name())
        except Exception:  # noqa: BLE001 - non-entity declarations
            continue
    return out


def test_the_sets_match_the_schema():
    """`facilities.py` is a hand-written mirror; this is what keeps it honest.

    Every set is re-derived from IFC4X3_ADD2 here. If IFC gains a facility
    subtype (or we typo one), this fails with the difference named — which is
    the only reason it is acceptable to enumerate them in a package that must
    not import ifcopenshell at runtime.
    """
    s = _schema()

    facilities = _concrete_subtypes_of(s, "IfcFacility") | {"IfcFacility"}
    parts = _concrete_subtypes_of(s, "IfcFacilityPart")

    # IfcBuilding is a facility in the schema but is DERIVED in the DSL (from
    # having storeys), so it is deliberately not in FACILITY_CLASSES.
    assert facilities - {"IfcBuilding"} == fac.FACILITY_CLASSES, (
        "FACILITY_CLASSES has drifted from IFC4X3_ADD2: "
        f"schema-only={sorted(facilities - {'IfcBuilding'} - fac.FACILITY_CLASSES)}, "
        f"ours-only={sorted(fac.FACILITY_CLASSES - facilities)}")

    assert parts == fac.FACILITY_PART_CLASSES, (
        "FACILITY_PART_CLASSES has drifted: "
        f"schema-only={sorted(parts - fac.FACILITY_PART_CLASSES)}, "
        f"ours-only={sorted(fac.FACILITY_PART_CLASSES - parts)}")

    # UsageType required on exactly the parts — the rule requires_usage encodes.
    for cls in sorted(fac.SPATIAL_IFC_CLASS_WHITELIST):
        d = s.declaration_by_name(cls)
        required = any(a.name() == "UsageType" and not a.optional()
                       for a in d.all_attributes())
        assert required == fac.requires_usage(cls), (
            f"{cls}: schema says UsageType required={required}, "
            f"requires_usage() says {fac.requires_usage(cls)}")

    enum_vals = set(s.declaration_by_name("IfcFacilityUsageEnum").enumeration_items())
    assert enum_vals == set(fac.FACILITY_USAGE_VALUES), (
        f"IfcFacilityUsageEnum drifted: schema={sorted(enum_vals)}, "
        f"ours={sorted(fac.FACILITY_USAGE_VALUES)}")

    # SPATIAL_ELEMENT_CLASSES is the set ``is_spatial_element`` answers on, and
    # it decides the RELATION a child joins its container by. A
    # class missing from it is emitted into IfcRelContainedInSpatialStructure
    # in violation of WR31, which no gate we have reports — so it is derived
    # here from the schema like every other set in this file, from the
    # ``IfcSpatialElement`` root rather than by listing leaves.
    spatial = _concrete_subtypes_of(s, "IfcSpatialElement")
    assert spatial == fac.SPATIAL_ELEMENT_CLASSES, (
        "SPATIAL_ELEMENT_CLASSES has drifted from IFC4X3_ADD2: "
        f"schema-only={sorted(spatial - fac.SPATIAL_ELEMENT_CLASSES)}, "
        f"ours-only={sorted(fac.SPATIAL_ELEMENT_CLASSES - spatial)}")
    for cls in sorted(spatial):
        assert fac.is_spatial_element(cls), (
            f"{cls} is an IfcSpatialElement the emitter would CONTAIN rather "
            f"than aggregate — WR31, and no validator reports it")


def test_every_whitelisted_class_is_spatial_and_concrete():
    """A physical class in the spatial whitelist would emit WR31 silently."""
    s = _schema()
    for cls in sorted(fac.SPATIAL_IFC_CLASS_WHITELIST):
        d = s.declaration_by_name(cls)
        assert not d.is_abstract(), f"{cls} is ABSTRACT and cannot be instantiated"
        chain, x = [], d
        while x:
            chain.append(x.name())
            x = x.supertype()
        assert "IfcSpatialElement" in chain, (
            f"{cls} is not an IfcSpatialElement — it would be emitted into a "
            f"containment relation and violate WR31 with no validator to say so")


def test_spatial_and_physical_whitelists_are_disjoint():
    from lite_step.models.elements import ELEMENT_IFC_CLASS_WHITELIST
    overlap = fac.SPATIAL_IFC_CLASS_WHITELIST & ELEMENT_IFC_CLASS_WHITELIST
    assert not overlap, (
        f"{sorted(overlap)} is in BOTH whitelists — one class cannot be joined "
        f"to the tree by two different relations")


# ── construction ───────────────────────────────────────────────────────────


def test_a_facility_and_a_part_construct():
    bridge = SpatialElement(ifc_class="IfcBridge", name="storstroem",
                            predefined_type="ARCHED")
    deck = SpatialElement(ifc_class="IfcBridgePart", name="deck",
                          usage="LONGITUDINAL", predefined_type="DECK")
    assert bridge.ifc_class == "IfcBridge"
    assert deck.usage == "LONGITUDINAL"


def test_usage_is_required_on_a_part_and_refused_elsewhere():
    with pytest.raises(Exception, match="usage= is required"):
        SpatialElement(ifc_class="IfcBridgePart", name="deck")
    with pytest.raises(Exception, match="applies only to facility"):
        SpatialElement(ifc_class="IfcBridge", name="b", usage="LONGITUDINAL")


def test_a_bad_usage_value_is_refused():
    with pytest.raises(Exception, match="IfcFacilityUsageEnum"):
        SpatialElement(ifc_class="IfcBridgePart", name="d", usage="SIDEWAYS")


def test_a_dedicated_container_class_names_the_alternative():
    for cls, hint in (("IfcSite", "Site"), ("IfcBuildingStorey", "Storey"),
                      ("IfcSpace", "Space")):
        with pytest.raises(Exception, match=hint):
            SpatialElement(ifc_class=cls, name="x")


def test_a_physical_class_is_refused_and_points_at_Element():
    with pytest.raises(Exception, match="Element"):
        SpatialElement(ifc_class="IfcWall", name="w")


# ── nesting: the ONLY defence against WR31 ─────────────────────────────────


def test_a_part_outside_a_facility_is_a_compile_error():
    """The validator will not catch this — see the module docstring."""
    site = Site(name="site")
    part = SpatialElement(ifc_class="IfcBridgePart", name="deck",
                          usage="LONGITUDINAL")
    with pytest.raises(Exception, match="must sit inside a facility"):
        site.add(part)


def test_a_facility_inside_a_part_is_a_compile_error():
    part = SpatialElement(ifc_class="IfcBridgePart", name="deck",
                          usage="LONGITUDINAL")
    with pytest.raises(Exception, match="cannot sit inside"):
        part.add(SpatialElement(ifc_class="IfcBridge", name="b"))


def test_a_part_inside_a_facility_is_accepted():
    bridge = SpatialElement(ifc_class="IfcBridge", name="b")
    bridge.add(SpatialElement(ifc_class="IfcBridgePart", name="deck",
                              usage="LONGITUDINAL"))
    assert len(bridge.elements) == 1


def test_elements_go_inside_a_part():
    part = SpatialElement(ifc_class="IfcBridgePart", name="deck",
                          usage="LONGITUDINAL")
    part.add(Box(name="girder", start=Point(x=0, y=0, z=0),
                 end=Point(x=1000, y=200, z=400),
                 material=Material(key="Steel_S250GD_Z275")))
    assert len(part.elements) == 1


# ── emission: the chain, and that it is NAVIGABLE ──────────────────────────


def _bridge_project():
    proj = Project(name="bridgetest")
    site = Site(name="site")
    bridge = SpatialElement(ifc_class="IfcBridge", name="storstroem",
                            predefined_type="ARCHED")
    deck = SpatialElement(ifc_class="IfcBridgePart", name="deck",
                          usage="LONGITUDINAL", predefined_type="DECK")
    deck.add(Box(name="girder", start=Point(x=0, y=0, z=0),
                 end=Point(x=10000, y=500, z=800),
                 material=Material(key="Steel_S250GD_Z275")))
    bridge.add(deck)
    site.add(bridge)
    proj.add(site)
    return proj


def _emit(proj):
    import contextlib
    import io as _io
    import os
    import tempfile

    from lite_step.compiler.executor import normalize_project_to_meters
    from lite_step.ifc.generator import generate_ifc

    ios = pytest.importorskip("ifcopenshell")
    with contextlib.redirect_stdout(_io.StringIO()):
        res = generate_ifc(normalize_project_to_meters(proj))
    path = os.path.join(tempfile.mkdtemp(), "m.ifc")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(res.ifc_content)
    return ios.open(path)


def test_the_facility_chain_is_emitted():
    f = _emit(_bridge_project())
    assert len(f.by_type("IfcBridge")) == 1
    assert len(f.by_type("IfcBridgePart")) == 1


def test_the_facility_is_REACHABLE_from_the_spatial_tree():
    """The whole point — the failure was a node in no relation.

    Walks Project -> Site -> facility -> part -> product using only the
    inverse attributes an inspector uses, so a node joined by the wrong
    relation simply does not appear.
    """
    f = _emit(_bridge_project())

    def decomposed(e):
        return [o for rel in (getattr(e, "IsDecomposedBy", []) or [])
                for o in rel.RelatedObjects]

    site = decomposed(f.by_type("IfcProject")[0])
    assert [e.is_a() for e in site] == ["IfcSite"]
    facility = decomposed(site[0])
    assert [e.is_a() for e in facility] == ["IfcBridge"], (
        "the bridge is not aggregated under the site — it would be invisible "
        "in every inspector, and the IFC validator does not check this")
    part = decomposed(facility[0])
    assert [e.is_a() for e in part] == ["IfcBridgePart"]

    contained = [o for rel in (getattr(part[0], "ContainsElements", []) or [])
                 for o in rel.RelatedElements]
    assert len(contained) == 1, "the girder is not contained in the part"


def test_products_are_CONTAINED_and_spatial_nodes_are_AGGREGATED():
    """WR31 in assertion form: no spatial element in a containment relation."""
    f = _emit(_bridge_project())
    for rel in f.by_type("IfcRelContainedInSpatialStructure"):
        for e in rel.RelatedElements:
            assert not e.is_a("IfcSpatialElement"), (
                f"{e.is_a()} is in IfcRelContainedInSpatialStructure."
                "RelatedElements — that is WR31, and ifcopenshell.validate "
                "does NOT report it, so this assertion is the only check")


def test_usage_and_predefined_types_reach_the_file():
    """A dropped OPTIONAL is invisible to the conformance gate."""
    f = _emit(_bridge_project())
    part = f.by_type("IfcBridgePart")[0]
    assert part.UsageType == "LONGITUDINAL"
    assert part.PredefinedType == "DECK"
    assert f.by_type("IfcBridge")[0].PredefinedType == "ARCHED"


def test_the_emitted_file_is_ifc43_conformant():
    import contextlib
    import io as _io

    import ifcopenshell.validate
    f = _emit(_bridge_project())
    logger = ifcopenshell.validate.json_logger()
    with contextlib.redirect_stderr(_io.StringIO()):
        ifcopenshell.validate.validate(f, logger)
    errors = [s for s in logger.statements if s.get("level") == "error"]
    assert not errors, f"{len(errors)} schema error(s): {errors[:3]}"


# ── a part RENDERS every child its .add() accepts ──────────────────────────


def _part_with(child):
    proj = Project(name="partmatrix")
    site = Site(name="site")
    bridge = SpatialElement(ifc_class="IfcBridge", name="b")
    part = SpatialElement(ifc_class="IfcBridgePart", name="deck",
                          usage="LONGITUDINAL")
    part.add(child)
    bridge.add(part)
    site.add(bridge)
    proj.add(site)
    return proj


def test_the_add_table_and_the_emitter_agree():
    """The table is the promise; this is the only thing that keeps it one.

    ``_FACILITY_PART_CHILDREN`` is what a part's ``.add()`` ACCEPTS, so every
    entry must reach the file. The emitter originally hand-rolled three
    branches (prism / Mesh / Element) covering under half of it, and the gap
    was the whole of a steel bridge: ``Sweep`` members were accepted at
    ``.add()`` and dropped at emission, taking ~250 of them out of four
    models. Derived from the table rather than a written-out list, so a type
    added to one side cannot quietly skip the other.

    **The ``Space`` row is INVERTED, and deliberately**. It read
    ``ContainsElements`` like every other row until an ``IfcSpace`` was noticed
    to be an ``IfcSpatialStructureElement``, which
    ``IfcRelContainedInSpatialStructure``'s WR31 forbids in
    ``RelatedElements``. It joins the part by ``IfcRelAggregates`` now. Which
    relation a row takes is read off ``facilities.is_spatial_element`` rather
    than from a name list here, so the test and the emitter cannot disagree
    about it — and every other row is untouched, which is what makes this a
    row inversion and not a weakened assertion.
    """
    from lite_step.models.elements import _FACILITY_PART_CHILDREN
    from lite_step.ifc.product_types import element_ifc_class
    from lite_step.tests.test_anchor_matrix import CHILDREN

    # SpatialElement is the nested-part row, not a product — covered by the
    # chain tests above; everything else must render, under one relation or
    # the other.
    expected = set(_FACILITY_PART_CHILDREN) - {"SpatialElement"}
    assert expected <= set(CHILDREN), (
        f"no factory for {sorted(expected - set(CHILDREN))} — the .add() table "
        f"grew a type this test cannot build, so it is silently uncovered")

    for name in sorted(expected):
        child = CHILDREN[name]()
        spatial = fac.is_spatial_element(element_ifc_class(child))
        f = _emit(_part_with(child))
        part = f.by_type("IfcBridgePart")[0]
        if spatial:
            joined = [o for rel in (getattr(part, "IsDecomposedBy", []) or [])
                      for o in rel.RelatedObjects]
            assert joined, (
                f"{name} emits a spatial element, so it decomposes the part "
                f"(WR31) — and nothing does. Dropped.")
            assert not [o for rel in (getattr(part, "ContainsElements", []) or [])
                        for o in rel.RelatedElements], (
                f"{name} is in IfcRelContainedInSpatialStructure, which WR31 "
                f"forbids for a spatial element")
        else:
            joined = [o for rel in (getattr(part, "ContainsElements", []) or [])
                      for o in rel.RelatedElements]
            assert joined, f"{name}: nothing contained in the part — dropped"
        with_geometry = [p for p in joined if getattr(p, "Representation", None)
                         or (getattr(p, "IsDecomposedBy", []) or [])]
        assert with_geometry, (
            f"{name}: the product reached the part but carries no "
            f"representation — an empty node the viewer renders as nothing")


def _three_level_bridge_project():
    """``IfcBridge -> IfcBridgePart(substructure) -> IfcBridgePart(pier)``.

    The girder sits in the DEEPEST part, so nothing about it is provable
    without the recursion.
    """
    proj = Project(name="deepbridge")
    site = Site(name="site")
    bridge = SpatialElement(ifc_class="IfcBridge", name="storstroem",
                            predefined_type="ARCHED")
    sub = SpatialElement(ifc_class="IfcBridgePart", name="substructure",
                         usage="VERTICAL", predefined_type="SUBSTRUCTURE")
    pier = SpatialElement(ifc_class="IfcBridgePart", name="pier",
                          usage="VERTICAL", predefined_type="PIER")
    pier.add(Box(name="shaft", start=Point(x=0, y=0, z=0),
                 end=Point(x=1200, y=1200, z=8000),
                 material=Material(key="Steel_S250GD_Z275")))
    sub.add(pier)
    bridge.add(sub)
    site.add(bridge)
    proj.add(site)
    return proj


def test_a_part_inside_a_part_is_accepted():
    """The construction gate, widened for exactly this pairing.

    Without it the fixture above cannot be built at all, and the emitter's
    recursion is a branch no author can reach.
    """
    sub = SpatialElement(ifc_class="IfcBridgePart", name="substructure",
                         usage="VERTICAL")
    sub.add(SpatialElement(ifc_class="IfcBridgePart", name="pier",
                           usage="VERTICAL"))
    assert len(sub.elements) == 1


def test_the_widening_is_exactly_one_pairing():
    """`validate_nesting` is the ONLY defence — everything else stays refused.

    ``ifcopenshell.validate`` does not report WR31, so a pairing let through
    here has nothing behind it. Each refusal below would otherwise emit a file
    that passes the conformance gate with the node missing from the tree.
    """
    # A part inside a part — the one thing that changed.
    fac.validate_nesting("IfcBridgePart", "IfcBridgePart")
    fac.validate_nesting("IfcRoadPart", "IfcFacilityPartCommon")

    # A part still needs a facility — or another part — above it.
    with pytest.raises(ValueError, match="must sit inside a facility"):
        fac.validate_nesting("IfcSite", "IfcBridgePart")
    with pytest.raises(ValueError, match="must sit inside a facility"):
        fac.validate_nesting("IfcSpatialZone", "IfcBridgePart")

    # A facility still cannot sit inside a part.
    with pytest.raises(ValueError, match="cannot sit inside"):
        fac.validate_nesting("IfcBridgePart", "IfcBridge")


def test_three_levels_emit_three_aggregations():
    """One ``IfcRelAggregates`` per spatial link, and containment at the LEAF.

    Two links would mean the pier was dropped; a containment relation on the
    substructure would mean its products were hoisted out of the part they
    were authored in — legal-looking, and wrong about where the shaft stands.
    """
    f = _emit(_three_level_bridge_project())

    site = f.by_type("IfcSite")[0]
    bridge = f.by_type("IfcBridge")[0]
    parts = {p.Name: p for p in f.by_type("IfcBridgePart")}
    assert len(parts) == 2, f"expected substructure + pier, got {sorted(parts)}"
    sub = next(p for n, p in parts.items() if "substructure" in n)
    pier = next(p for n, p in parts.items() if "pier" in n)

    links = {(r.RelatingObject.id(), o.id())
             for r in f.by_type("IfcRelAggregates")
             for o in r.RelatedObjects
             if r.RelatingObject.is_a("IfcSpatialElement")}
    assert links == {(site.id(), bridge.id()),
                     (bridge.id(), sub.id()),
                     (sub.id(), pier.id())}, (
        "the spatial chain is not three aggregations deep — a level was "
        "dropped, and the file still validates at zero errors")

    def contained(e):
        return [o for rel in (getattr(e, "ContainsElements", []) or [])
                for o in rel.RelatedElements]

    assert not contained(bridge), "a facility holds no products"
    assert not contained(sub), (
        "the shaft was contained in the SUBSTRUCTURE — containment fires at "
        "the leaf where the products actually live, not at the first part")
    assert len(contained(pier)) == 1, (
        f"the shaft is not in the pier: {[p.is_a() for p in contained(pier)]}")


def test_the_deepest_product_is_REACHABLE_from_the_facility():
    """The one that catches the regression — the failure was silence.

    Revert the emitter's recursion and the sub-part is dropped with a log
    warning, while the file it writes still validates at ZERO errors. Entity
    counts, conformance and ``IfcBridge`` being present all stay green; what
    changes is that the pier and its shaft are unreachable from the tree, so
    every inspector shows an empty bridge. This walks the same inverse
    attributes an inspector walks.
    """
    import ifcopenshell.util.element as _el

    f = _emit(_three_level_bridge_project())
    bridge = f.by_type("IfcBridge")[0]

    reachable = {e.id() for e in _el.get_decomposition(bridge)}
    piers = [p for p in f.by_type("IfcBridgePart") if "pier" in p.Name]
    assert piers, "the deepest part is not in the file at all — it was dropped"
    pier = piers[0]
    shaft = [o for rel in (getattr(pier, "ContainsElements", []) or [])
             for o in rel.RelatedElements]
    assert shaft, "the pier contains nothing"

    assert pier.id() in reachable, (
        "the pier is not in the bridge's decomposition — it exists in the "
        "file but hangs off nothing, which no validator reports")
    assert shaft[0].id() in reachable, (
        "the shaft is unreachable from the bridge — this is exactly: "
        "a file at zero errors with the geometry missing from the tree")


def test_the_three_level_file_is_ifc43_conformant():
    """0 errors here is necessary and NOT sufficient — see the test above.

    It is also the control for the falsification: dropping the sub-part keeps
    THIS test green, which is why reachability is asserted separately.
    """
    import contextlib
    import io as _io

    import ifcopenshell.validate
    f = _emit(_three_level_bridge_project())
    logger = ifcopenshell.validate.json_logger()
    with contextlib.redirect_stderr(_io.StringIO()):
        ifcopenshell.validate.validate(f, logger)
    errors = [s for s in logger.statements if s.get("level") == "error"]
    assert not errors, f"{len(errors)} schema error(s): {errors[:3]}"


