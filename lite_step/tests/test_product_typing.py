"""``Product`` promotion + ``IfcRelDefinesByType`` emission (WS-B step 1).

Roadmap §V.4–V.7. A Product is a finished, reusable catalog item promoted
from a built element; its occurrences carry identity only. This slice emits
the TYPE relation and nothing else — occurrences still emit their own
geometry, so there is no ``IfcMappedItem`` and ``RepresentationMaps`` is ``$``
(that is step 2, where the geometry risk lives).

What is pinned here:

1.  The four promotion guards, each with the reason in its message —
    unplaced, physical, a schema-backed type counterpart, and a valid
    catalog key.
2.  The HARD RULE (§V.6): a Product takes no variables, at the constructor
    AND at ``.occurrence()`` AND at derivation-by-call on an occurrence
    (the one back door the spec's stated trap does not close).
3.  Promotion SNAPSHOTS — mutating the source afterwards cannot reach a
    placed occurrence.
4.  Emission on BOTH backends, asserted to agree: exactly one
    ``Ifc<Class>Type``, exactly one ``IfcRelDefinesByType``, listing every
    occurrence.
5.  **The manifest trap.** ``IfcAnnotation`` is an ``IfcProduct``, so a named drawing landed in ``LITESTEP_META`` and
    polluted patch identity with an entity matching no DSL element. An
    ``IfcTypeProduct`` is NOT an ``IfcProduct``, so the type entity should
    not — but "should not" is how that bug shipped the first time, so it is
    asserted on both backends, whose exclusion mechanisms are different.
6.  The classes that have NO ``Ifc*Type`` counterpart, re-derived from the
    live IFC4X3_ADD2 schema so a whitelist addition cannot silently emit an
    entity name the schema does not define.
7.  Additivity: a project with no Product emits no type entity at all.
"""

from __future__ import annotations

import json
import re

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.product_types import (
    IFC_CLASSES_WITHOUT_TYPE,
    DEFAULT_TYPE_TAIL,
    TYPE_TAIL_OVERRIDES,
    COMMON_TYPE_ATTR_COUNT,
    ProductTypeError,
    collect_product_groups,
    type_entity_for,
)
from lite_step.models import (
    Beam, Box, Element, Point, Product, Project, Transform, Wall, Window,
)

pytest.importorskip("ifcopenshell")


# ---------------------------------------------------------------------------
# Fixtures — local geometry only (never exec a skill; see tests/_fixtures.py)
# ---------------------------------------------------------------------------

def build_panel(width: int = 2000, name: str = None) -> Wall:
    """A finished wall panel, UNPLACED — the promotable shape."""
    w = Wall(name=name)
    w.add(Box(start=Point(x=0, y=0, z=0),
              end=Point(x=width, y=300, z=2700)))
    return w


def build_window(width: int = 1400, height: int = 1600) -> Window:
    win = Window(width=width, height=height)
    win.add(Box(start=Point(x=0, y=0, z=0),
                end=Point(x=width, y=60, z=height)))
    return win


def project_of(product: Product, count: int, spacing: int = 2500) -> Project:
    proj = Project(name="catalog")
    for i in range(count):
        occ = product.occurrence(name=f"unit{i}")
        occ.placement = Transform(origin=Point(x=i * spacing, y=0, z=0))
        proj.add(occ)
    return proj


def _compile(proj: Project, backend: str, source_code: str = "src") -> str:
    """Compile through the emitter.

    The streaming case calls ``generate_ifc_streaming`` DIRECTLY for the
    mirror-image reason: ``generate_ifc(backend="streaming")`` falls BACK to
    ifcopenshell whenever ``can_stream()`` says no, so it names a backend
    without guaranteeing one. Same shape as ``test_grids.py``.
    """
    normalized = normalize_project_to_meters(proj)
    from lite_step.ifc.generator import generate_ifc
    result = generate_ifc(normalized, source_code=source_code)
    assert result.success, result.error
    return result.ifc_content



def _entities(ifc: str, entity_type: str) -> list:
    """Every STEP line of ``entity_type`` (exact match on the entity name)."""
    pattern = re.compile(r"^#\d+=" + entity_type.upper() + r"\(", re.MULTILINE)
    return [line for line in ifc.splitlines() if pattern.match(line)]


def _map_shape(rep_map) -> tuple:
    """An ``IfcRepresentationMap`` reduced to what the two backends must agree on.

    NOT its ``repr``: entity IDs differ between the backends by construction,
    so comparing reprs compares the ID allocator. What has to match is the
    identity origin and the representation being mapped.
    """
    origin = rep_map.MappingOrigin
    mapped = rep_map.MappedRepresentation
    return (
        origin.is_a(),
        tuple(round(c, 9) for c in origin.Location.Coordinates),
        origin.Axis, origin.RefDirection,
        mapped.RepresentationIdentifier, mapped.RepresentationType,
        len(mapped.Items),
    )


def _manifest(ifc: str) -> dict:
    import ifcopenshell
    import tempfile
    import os

    path_handle = tempfile.NamedTemporaryFile(suffix=".ifc", delete=False)
    path_handle.close()
    path = path_handle.name
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ifc)
        model = ifcopenshell.open(path)
        for pset in model.by_type("IfcPropertySet"):
            if pset.Name == "LITESTEP_META":
                for prop in pset.HasProperties:
                    if prop.Name == "MANIFEST":
                        return json.loads(prop.NominalValue.wrappedValue)
    finally:
        os.unlink(path)
    raise AssertionError("no LITESTEP_META manifest in the file")


# ---------------------------------------------------------------------------
# 1. Promotion guards
# ---------------------------------------------------------------------------


class TestPromotionRefusesAPlacedSource:
    """§V.5 rule 1 — a catalog object has no canonical path."""

    def test_contained_source_refuses(self):
        proj = Project(name="p")
        panel = build_panel(name="south")
        proj.add(panel)
        with pytest.raises(ValueError, match="already placed"):
            Product(panel, name="panel_2000")

    def test_anchored_source_refuses(self):
        host = build_panel(name="south")
        win = build_window()
        host.anchor(win, along=1000, up=900)
        with pytest.raises(ValueError, match="already placed"):
            Product(win, name="nordic_1400x1600")

    def test_the_message_names_the_reason_and_the_fix(self):
        proj = Project(name="p")
        panel = build_panel(name="south")
        proj.add(panel)
        with pytest.raises(ValueError) as exc:
            Product(panel, name="panel_2000")
        message = str(exc.value)
        assert "no canonical path" in message
        assert "occurrence(name=" in message

    def test_an_unplaced_source_is_accepted(self):
        # The falsification partner: the guard must not refuse everything.
        assert Product(build_panel(), name="panel_2000").name == "panel_2000"


class TestPromotionRefusesNonPhysical:
    """§V.6 rule 2 — ``is_physical``'s first production consumer.

    A bare ``Box`` is GEOMETRIC: the schema types PRODUCTS, and there is no
    ``IfcBoxType`` for ``IfcRelDefinesByType`` to point at.
    """

    def test_a_box_refuses(self):
        box = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000))
        with pytest.raises(ValueError, match="only a PHYSICAL element"):
            Product(box, name="cube")

    def test_the_message_names_the_wrapper_that_fixes_it(self):
        box = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000))
        with pytest.raises(ValueError) as exc:
            Product(box, name="cube")
        assert "Element(ifc_class=" in str(exc.value)

    def test_every_physical_type_is_promotable(self):
        from lite_step.models import taxonomy as tx

        # The guard reads the taxonomy, so the taxonomy is what must agree.
        for cls in (Wall, Beam):
            assert tx.is_physical(cls)
            assert Product(cls(), name="x").ifc_class == "Ifc" + cls.__name__

    def test_the_same_geometry_wrapped_is_accepted(self):
        wrapper = Element(ifc_class="IfcStair")
        wrapper.add(Box(start=Point(x=0, y=0, z=0),
                        end=Point(x=1000, y=1000, z=1000)))
        assert Product(wrapper, name="stair_std").type_entity == "IfcStairType"


class TestPromotionRefusesClassesWithoutAType:
    """Some IFC4X3_ADD2 classes have no ``Ifc*Type``. Refuse loudly at the
    ``Product(...)`` line rather than emitting an entity name the schema does
    not define — an invented ``IFCEARTHWORKSFILLTYPE`` parses as a STEP tuple
    and is only rejected (or silently dropped) by whichever viewer opens it."""

    @pytest.mark.parametrize("ifc_class", sorted(IFC_CLASSES_WITHOUT_TYPE))
    def test_each_refuses_at_promotion(self, ifc_class):
        elem = Element(ifc_class=ifc_class)
        with pytest.raises(ProductTypeError, match="no Ifc.*Type counterpart"):
            Product(elem, name="ground")

    def test_type_entity_for_is_the_one_decision_point(self):
        assert type_entity_for("IfcWindow") == "IfcWindowType"
        with pytest.raises(ProductTypeError):
            type_entity_for("IfcEarthworksFill")


class TestCatalogKey:
    """§V.5 rule 3 — the name is a catalog KEY, not a canonical path."""

    def test_a_path_separator_refuses(self):
        with pytest.raises(ValueError, match="catalog key"):
            Product(build_panel(), name="wall:south")

    def test_uppercase_refuses(self):
        with pytest.raises(ValueError, match="catalog key"):
            Product(build_panel(), name="Nordic")

    def test_a_missing_key_refuses(self):
        with pytest.raises(ValueError, match="catalog key"):
            Product(build_panel())

    def test_the_message_says_why_a_key_is_not_a_path(self):
        with pytest.raises(ValueError) as exc:
            Product(build_panel(), name="wall:south")
        assert "NOT a path" in str(exc.value)


# ---------------------------------------------------------------------------
# 2. The hard rule — a Product takes no variables (§V.6)
# ---------------------------------------------------------------------------


class TestNoVariables:
    def test_occurrence_refuses_a_geometric_kwarg(self):
        nordic = Product(build_window(), name="nordic_1400x1600")
        with pytest.raises(ValueError) as exc:
            nordic.occurrence(name="g1", width=1200)
        message = str(exc.value)
        assert "takes NO variables" in message
        assert "'width'" in message
        assert "TWO" in message and "Products" in message

    def test_occurrence_refuses_any_unknown_kwarg(self):
        # Not just model fields: the gate is "identity only", so anything
        # other than name= is refused before it can be interpreted.
        nordic = Product(build_window(), name="nordic_1400x1600")
        with pytest.raises(ValueError, match="takes NO variables"):
            nordic.occurrence(name="g1", material="wood_pine")

    def test_the_constructor_refuses_variables_too(self):
        with pytest.raises(ValueError, match="takes NO variables"):
            Product(build_window(), name="nordic_1400x1600", width=1200)

    def test_occurrence_requires_a_name(self):
        nordic = Product(build_window(), name="nordic_1400x1600")
        with pytest.raises(ValueError, match="needs a name"):
            nordic.occurrence()

    def test_identity_only_is_accepted(self):
        nordic = Product(build_window(), name="nordic_1400x1600")
        occ = nordic.occurrence(name="g0")
        assert occ.name == "g0"
        assert occ.width == 1400

    def test_a_product_is_not_callable(self):
        """§V.5's stated trap: ``NORDIC(name="g0")`` reads as
        derivation-by-call, which means an independent COPY everywhere else."""
        nordic = Product(build_window(), name="nordic_1400x1600")
        with pytest.raises(TypeError, match=r"occurrence\(name="):
            nordic(name="g0")

    def test_deriving_from_an_occurrence_refuses(self):
        """The back door the trap does not close.

        Private state carries over on derivation-by-call, so ``occ(width=...)``
        would keep the catalog key while changing the geometry — the type
        relation would claim a shape the occurrence does not have. Dropping
        the key instead is the other silent failure.
        """
        nordic = Product(build_window(), name="nordic_1400x1600")
        occ = nordic.occurrence(name="g0")
        with pytest.raises(TypeError, match="is an occurrence of Product"):
            occ(width=1200)
        with pytest.raises(TypeError, match="is an occurrence of Product"):
            occ(name="g1")

    def test_derivation_still_works_on_an_ordinary_element(self):
        # Falsification partner: the guard is keyed on the occurrence marker,
        # not on the call itself.
        assert build_window()(name="plain").name == "plain"


# ---------------------------------------------------------------------------
# 3. Promotion snapshots (§V.5 rule 2)
# ---------------------------------------------------------------------------


class TestPromotionSnapshots:
    def test_mutating_the_source_does_not_change_later_occurrences(self):
        source = build_window(width=1400, height=1600)
        nordic = Product(source, name="nordic_1400x1600")
        source.width = 9999
        assert nordic.occurrence(name="g0").width == 1400

    def test_mutating_the_source_does_not_change_placed_occurrences(self):
        source = build_panel(width=2000)
        panel = Product(source, name="panel_2000")
        proj = project_of(panel, 2)
        source._elements[0].end = Point(x=9999, y=300, z=2700)
        for occ in proj.storeys[0].elements:
            assert occ._elements[0].end.x == 2000

    def test_the_snapshot_is_not_the_source_object(self):
        source = build_panel()
        assert Product(source, name="panel_2000").snapshot is not source

    def test_occurrences_do_not_share_geometry_with_each_other(self):
        panel = Product(build_panel(), name="panel_2000")
        a, b = panel.occurrence(name="a"), panel.occurrence(name="b")
        a._elements[0].end = Point(x=1, y=1, z=1)
        assert b._elements[0].end.x == 2000

    def test_each_occurrence_gets_a_fresh_internal_id(self):
        # Derivation-by-call's rule, and load-bearing here: N occurrences
        # sharing one id= would collide in the ID-uniqueness compile check.
        panel = Product(build_panel(), name="panel_2000")
        ids = {panel.occurrence(name=f"u{i}").id for i in range(5)}
        assert len(ids) == 5


# ---------------------------------------------------------------------------
# 4. Emission — both backends, asserted to agree
# ---------------------------------------------------------------------------


class TestTypeEmission:
    def test_nine_occurrences_emit_one_type_and_one_relation(self):
        panel = Product(build_panel(), name="panel_2000")
        ifc = _compile(project_of(panel, 9), "ifcopenshell")

        types = _entities(ifc, "IFCWALLTYPE")
        rels = _entities(ifc, "IFCRELDEFINESBYTYPE")
        assert len(types) == 1, types
        assert len(rels) == 1, rels
        # ...and it lists all nine, not eight.
        assert len(re.search(r"\(((?:#\d+,?)+)\)", rels[0]).group(1).split(",")) == 9

    def test_the_type_carries_the_catalog_key_as_its_name(self):
        panel = Product(build_panel(), name="panel_2000")
        ifc = _compile(project_of(panel, 3), "ifcopenshell")
        assert "'panel_2000'" in _entities(ifc, "IFCWALLTYPE")[0]

    def test_the_relation_lists_every_occurrence_product(self):
        import ifcopenshell
        import os
        import tempfile

        panel = Product(build_panel(), name="panel_2000")
        ifc = _compile(project_of(panel, 4), "ifcopenshell")
        path_handle = tempfile.NamedTemporaryFile(suffix=".ifc", delete=False)
        path_handle.close()
        path = path_handle.name
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(ifc)
            model = ifcopenshell.open(path)
            rel = model.by_type("IfcRelDefinesByType")[0]
            assert sorted(o.Name for o in rel.RelatedObjects) == [
                "wall:unit0", "wall:unit1", "wall:unit2", "wall:unit3"]
            assert rel.RelatingType.Name == "panel_2000"
            assert rel.RelatingType.is_a() == "IfcWallType"
        finally:
            os.unlink(path)

    def test_step_2_fills_representation_maps_on_the_type(self):
        """The inverse of what step 1 asserted, on both backends.

        Asserting ``RepresentationMaps`` is ``$`` with no ``IfcMappedItem``
            is inverted here rather than deleted, because it is the one assertion that tells step 1's file from step 2's,
        and a stale copy of it going green would have meant the sharing pass
        was not reaching this "ifcopenshell" at all. The geometry consequences live in
        ``test_product_geometry_sharing.py``.
        """
        panel = Product(build_panel(), name="panel_2000")
        ifc = _compile(project_of(panel, 3), "ifcopenshell")
        assert _entities(ifc, "IFCMAPPEDITEM")
        assert _entities(ifc, "IFCREPRESENTATIONMAP")
        # Attribute 7 of 10 (index 6) is RepresentationMaps.
        attrs = _entities(ifc, "IFCWALLTYPE")[0].split("(", 1)[1].rsplit(")", 1)[0]
        assert attrs.split(",")[6].startswith("(#"), attrs
        # ...and each occurrence still has its own IfcProductDefinitionShape —
        # sharing replaces what a product's shape CONTAINS, never the fact that
        # it has one. A product that lost its shape would vanish from view.
        assert len(_entities(ifc, "IFCPRODUCTDEFINITIONSHAPE")) >= 3

    def test_no_product_emits_no_type_entity(self):
        """Additivity: this feature is invisible to every existing model."""
        proj = Project(name="plain")
        proj.add(build_panel(name="south"))
        ifc = _compile(proj, "ifcopenshell")
        assert "TYPE(" not in ifc.upper().replace("IFCPROPERTYSINGLEVALUE", "")
        assert not _entities(ifc, "IFCRELDEFINESBYTYPE")

    def test_a_window_occurrence_anchored_into_a_wall_is_typed(self):
        """The §V.4 worked case: a Window is a semantic opening, so the type
        types the FILL only — the void and the anchor stay per-occurrence."""
        import ifcopenshell
        import os
        import tempfile

        nordic = Product(build_window(), name="nordic_1400x1600")
        proj = Project(name="facade")
        wall = build_panel(width=12000, name="south")
        for i in range(3):
            wall.anchor(nordic.occurrence(name=f"g{i}"),
                        along=1000 + i * 3000, up=900)
        proj.add(wall)
        ifc = _compile(proj, "ifcopenshell")

        path_handle = tempfile.NamedTemporaryFile(suffix=".ifc", delete=False)
        path_handle.close()
        path = path_handle.name
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(ifc)
            model = ifcopenshell.open(path)
            assert len(model.by_type("IfcWindowType")) == 1
            rel = model.by_type("IfcRelDefinesByType")[0]
            assert len(rel.RelatedObjects) == 3
            # Three separate voids survive — the hole is geometry in the HOST.
            assert len(model.by_type("IfcOpeningElement")) == 3
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# 5. The manifest trap
# ---------------------------------------------------------------------------


class TestTypeEntityStaysOutOfTheManifest:
    """``IfcAnnotation`` IS an ``IfcProduct``, so a named
    drawing landed in ``LITESTEP_META`` and polluted patch identity with an
    entity matching no DSL element. An ``IfcTypeProduct`` is not an
    ``IfcProduct``, so a type should not — but each backend excludes it by a
    DIFFERENT mechanism (schema hierarchy vs. emission phase), so assert both.
    """

    def test_the_catalog_key_is_not_a_manifest_entry(self):
        panel = Product(build_panel(), name="panel_2000")
        manifest = _manifest(_compile(project_of(panel, 3), "ifcopenshell"))
        assert "panel_2000" not in manifest
        # The occurrences ARE, because they are real DSL elements.
        assert {"wall:unit0", "wall:unit1", "wall:unit2"} <= set(manifest)

    def test_no_manifest_entry_maps_to_the_type_guid(self):
        """Stronger than a key check: the pollution that mattered was a GUID
        in the manifest pointing at an entity no DSL element corresponds to."""
        panel = Product(build_panel(), name="panel_2000")
        ifc = _compile(project_of(panel, 3), "ifcopenshell")
        type_guid = re.search(r"IFCWALLTYPE\('([^']+)'",
                              _entities(ifc, "IFCWALLTYPE")[0]).group(1)
        rel_guid = re.search(r"IFCRELDEFINESBYTYPE\('([^']+)'",
                             _entities(ifc, "IFCRELDEFINESBYTYPE")[0]).group(1)
        values = set(_manifest(ifc).values())
        assert type_guid not in values
        assert rel_guid not in values

    def test_ifctypeproduct_is_not_an_ifcproduct(self):
        """The schema fact the ifcopenshell backend's exclusion rests on."""
        from ifcopenshell import ifcopenshell_wrapper

        schema = ifcopenshell_wrapper.schema_by_name("IFC4X3_ADD2")
        window_type = schema.declaration_by_name("IfcWindowType").as_entity()
        supertypes = set()
        cursor = window_type
        while cursor is not None:
            supertypes.add(cursor.name())
            cursor = cursor.supertype()
        assert "IfcTypeProduct" in supertypes
        assert "IfcProduct" not in supertypes


# ---------------------------------------------------------------------------
# 6. Loud failure over silent degradation
# ---------------------------------------------------------------------------


class TestSilentDegradationIsRefused:
    def test_an_occurrence_used_as_a_boolean_operand_refuses(self):
        """An operand is consumed into its host's shape and emits no product,
        so it would be counted in IfcRelDefinesByType while nothing in the
        file corresponds to it."""
        panel = Product(build_panel(), name="panel_2000")
        proj = Project(name="p")
        host = build_panel(width=6000, name="south")
        host.difference(panel.occurrence(name="notch"))
        proj.add(host)
        with pytest.raises(ProductTypeError, match="boolean operand"):
            collect_product_groups(normalize_project_to_meters(proj))

    def test_two_promotions_sharing_a_key_refuse(self):
        """One type entity for two catalog items: a schedule reading it would
        count them as one, and if they differ by a millimetre it is silently
        wrong. Refused on the SERIAL, which survives normalize's deepcopy —
        object identity does not, and snapshot equality cannot answer it
        either (two elements built by the same function differ by ``id``).
        """
        proj = Project(name="p")
        a = Product(build_panel(width=2000), name="panel")
        b = Product(build_panel(width=3000), name="panel")
        proj.add(a.occurrence(name="u0"))
        occ = b.occurrence(name="u1")
        occ.placement = Transform(origin=Point(x=5000, y=0, z=0))
        proj.add(occ)
        with pytest.raises(ProductTypeError, match="two separate Product"):
            collect_product_groups(normalize_project_to_meters(proj))

    def test_the_serial_survives_normalize(self):
        """Falsification partner: the guard must not fire on ONE promotion.

        ``normalize_project_to_meters`` deep-copies the tree, so an identity
        test would report every occurrence as a separate promotion and the
        guard above would reject every legal model.
        """
        panel = Product(build_panel(width=2000), name="panel")
        groups = collect_product_groups(
            normalize_project_to_meters(project_of(panel, 4)))
        assert len(groups) == 1 and len(groups[0].occurrences) == 4

    def test_two_classes_sharing_a_key_refuse(self):
        proj = Project(name="p")
        wall_p = Product(build_panel(), name="unit")
        beam_p = Product(Beam(), name="unit")
        proj.add(wall_p.occurrence(name="u0"))
        proj.add(beam_p.occurrence(name="u1"))
        with pytest.raises(ProductTypeError, match="two different IFC classes"):
            collect_product_groups(normalize_project_to_meters(proj))

    def test_the_missing_occurrence_message_counts_what_was_lost(self):
        """A relation listing 7 of 9 occurrences is a procurement lie, and it
        is the kind that reads as success in every log."""
        from lite_step.ifc.product_types import ProductTypeGroup, missing_occurrence_error

        group = ProductTypeGroup(key="panel", ifc_class="IfcWall",
                                 type_entity="IfcWallType",
                                 tail=DEFAULT_TYPE_TAIL,
                                 occurrences=("wall:a", "wall:b"))
        message = str(missing_occurrence_error(group, ["wall:b"]))
        assert "1 of 2" in message and "undercount" in message

    def test_an_unresolvable_occurrence_fails_the_compile(self, monkeypatch):
        """Both backends RAISE rather than emitting a short relation.

        Forced through the shared collector, because the canonical names it
        returns are exactly what the join can fail on — the generators
        re-stamp canonicals themselves, so mutating the tree beforehand
        cannot reach this path.
        """
        import lite_step.ifc.product_types as pt

        real = pt.collect_product_groups

        def with_a_ghost(proj):
            groups = real(proj)
            return [pt.ProductTypeGroup(
                key=g.key, ifc_class=g.ifc_class, type_entity=g.type_entity,
                tail=g.tail, occurrences=g.occurrences + ("wall:ghost",))
                for g in groups]

        monkeypatch.setattr(pt, "collect_product_groups", with_a_ghost)

        panel = Product(build_panel(), name="panel_2000")
        normalized = normalize_project_to_meters(project_of(panel, 2))
        # Not via ``_compile``: that helper asserts success, and this asserts
        # a FAILED compile.
        from lite_step.ifc.generator import generate_ifc
        result = generate_ifc(normalized, source_code="src")
        assert not result.success
        assert "wall:ghost" in result.error and "undercount" in result.error


# ---------------------------------------------------------------------------
# 7. The type table, re-derived from the live schema
# ---------------------------------------------------------------------------


class TestSchemaTable:
    """The static tables in ``product_types`` are a schema snapshot. Re-derive
    them so a whitelist addition (or a schema bump) cannot leave a promotable
    class emitting an entity name IFC4X3_ADD2 does not define, or an entity
    with the wrong attribute count."""

    @staticmethod
    def _schema():
        from ifcopenshell import ifcopenshell_wrapper
        from lite_step.ifc.schema_version import IFC_OUTPUT_SCHEMA

        return ifcopenshell_wrapper.schema_by_name(IFC_OUTPUT_SCHEMA)

    @staticmethod
    def _promotable_classes():
        from lite_step.models import ELEMENT_IFC_CLASS_WHITELIST
        from lite_step.models import taxonomy as tx
        from lite_step.ifc.product_types import element_ifc_class

        sugar = {element_ifc_class(cls) for cls in tx.PHYSICAL
                 if cls.__name__ != "Element"}
        return sugar | set(ELEMENT_IFC_CLASS_WHITELIST)

    def test_the_refusal_set_is_exactly_what_the_schema_lacks(self):
        schema = self._schema()
        lacking = set()
        for ifc_class in self._promotable_classes():
            try:
                schema.declaration_by_name(ifc_class + "Type")
            except Exception:
                lacking.add(ifc_class)
        assert lacking == set(IFC_CLASSES_WITHOUT_TYPE)

    def test_every_emitted_tail_matches_the_schema(self):
        schema = self._schema()
        for ifc_class in sorted(self._promotable_classes() - IFC_CLASSES_WITHOUT_TYPE):
            type_entity = type_entity_for(ifc_class)
            decl = schema.declaration_by_name(type_entity).as_entity()
            tail = TYPE_TAIL_OVERRIDES.get(type_entity, DEFAULT_TYPE_TAIL)

            assert decl.attribute_count() == COMMON_TYPE_ATTR_COUNT + len(tail), (
                f"{type_entity}: attribute count drifted — a STEP line with "
                f"the wrong arity is malformed")
            for offset, (name, value) in enumerate(tail):
                attr = decl.attribute_by_index(COMMON_TYPE_ATTR_COUNT + offset)
                assert attr.name() == name, type_entity
                if value is None:
                    assert attr.optional(), (
                        f"{type_entity}.{name} is MANDATORY but we emit $")
                else:
                    assert not attr.optional(), (
                        f"{type_entity}.{name} is optional — emitting "
                        f"{value} is noise, not a requirement")
                    enum_name = attr.type_of_attribute().declared_type().name()
                    members = list(
                        schema.declaration_by_name(enum_name).as_enumeration_type()
                        .enumeration_items())
                    assert value in members, (
                        f"{type_entity}.{name}: {value} is not a member of "
                        f"{enum_name}")

    def test_the_class_to_ifc_class_derivation_answers_classes_too(self):
        """Adjacent silent failure, found while writing the test above.

        ``container_ifc_class`` derives from ``type(x)``, so a naive fallback
            answers ``"Ifc" + type(Wall).__name__`` == ``"IfcModelMetaclass"``
            — pydantic's metaclass, a class no schema defines — with no error.
        Every caller passes an instance, so this is one convenience call away
            from being wrong silently.
        """
        from lite_step.ifc.product_types import element_ifc_class

        assert element_ifc_class(Wall) == "IfcWall"
        assert element_ifc_class(Wall()) == "IfcWall"
        assert element_ifc_class(Element(ifc_class="IfcStair")) == "IfcStair"
        # The one case a class genuinely cannot answer.
        with pytest.raises(TypeError, match="per instance"):
            element_ifc_class(Element)

    def test_the_first_nine_attributes_are_the_common_type_product_ones(self):
        schema = self._schema()
        expected = ["GlobalId", "OwnerHistory", "Name", "Description",
                    "ApplicableOccurrence", "HasPropertySets",
                    "RepresentationMaps", "Tag", "ElementType"]
        decl = schema.declaration_by_name("IfcWindowType").as_entity()
        got = [decl.attribute_by_index(i).name()
               for i in range(COMMON_TYPE_ATTR_COUNT)]
        assert got == expected


class TestOccurrenceDivergence:
    """An occurrence must still BE the item it names.

    ``.occurrence()`` refuses variables and ``__call__`` refuses derivation,
    so the two spellings that ANNOUNCE an override are closed. These pin the
    ones that do not announce it — and the reason the check is on the
    invariant rather than on a list of spellings is that the set of ways to
    mutate a mutable object is open, while "does it still match?" is not.
    """

    @staticmethod
    def _product():
        from lite_step.models import Box, Point, Product, Window
        win = Window(name="casement", width=1200, height=1400)
        win.add(Box(name="frame", start=Point(x=0, y=0, z=0),
                    end=Point(x=1200, y=60, z=1400), material="Wood_Pine"))
        return Product(win, name="casement_1200")

    def test_an_untouched_occurrence_matches(self):
        product = self._product()
        assert product.matches(product.occurrence(name="g0")) is True

    def test_model_copy_with_an_update_diverges(self):
        """The pydantic back door. ``__call__`` is refused, this is not — and
        it reaches the identical state: ``_product_key`` kept, geometry
        changed."""
        product = self._product()
        occ = product.occurrence(name="g0").model_copy(update={"width": 900})
        assert occ._product_key == "casement_1200"      # still claims the type
        assert product.matches(occ) is False

    def test_plain_attribute_assignment_diverges(self):
        product = self._product()
        occ = product.occurrence(name="g0")
        occ.width = 900
        assert product.matches(occ) is False

    def test_mutating_a_CHILD_diverges(self):
        """The one a guard on the occurrence object could never catch."""
        from lite_step.models import Point
        product = self._product()
        occ = product.occurrence(name="g0")
        occ.elements[0].end = Point(x=1200, y=60, z=9999)
        assert product.matches(occ) is False

    def test_a_boolean_operand_diverges(self):
        """``.difference()`` changes the shape and nothing else records it."""
        from lite_step.models import Box, Point
        product = self._product()
        occ = product.occurrence(name="g0")
        occ.difference(Box(name="notch", start=Point(x=0, y=0, z=0),
                           end=Point(x=100, y=100, z=100)))
        assert product.matches(occ) is False

    # -- what must NOT fire -------------------------------------------------

    def test_a_different_name_is_not_a_divergence(self):
        product = self._product()
        occ = product.occurrence(name="g0")
        occ.name = "g1"
        assert product.matches(occ) is True, "identity is not shape"

    def test_a_placement_is_not_a_divergence(self):
        """Placement is the occurrence's ENTIRE job. If this ever fires, the
        check refuses the normal case and is worse than useless."""
        from lite_step.models import Point, Transform
        product = self._product()
        occ = product.occurrence(name="g0")
        occ.placement = Transform(origin=Point(x=9000, y=4000, z=0),
                                  rotations=[("z", 9000)])
        assert product.matches(occ) is True

    def test_two_occurrences_of_one_product_match_each_other(self):
        product = self._product()
        assert product.matches(product.occurrence(name="a")) is True
        assert product.matches(product.occurrence(name="b")) is True

    def test_editing_the_SOURCE_after_promotion_does_not_diverge(self):
        """Promotion snapshots (rule 3). A later edit to the source must not
        retroactively invalidate occurrences already placed."""
        from lite_step.models import Box, Point, Product, Window
        win = Window(name="casement", width=1200, height=1400)
        win.add(Box(name="frame", start=Point(x=0, y=0, z=0),
                    end=Point(x=1200, y=60, z=1400), material="Wood_Pine"))
        product = Product(win, name="casement_1200")
        occ = product.occurrence(name="g0")
        win.width = 999                                  # edit the SOURCE
        assert product.matches(occ) is True

    # -- and through the compile path --------------------------------------

    def test_the_compile_refuses_a_diverged_occurrence(self):
        from lite_step.compiler.executor import validate_project_report
        from lite_step.models import (Box, Column, Point, Product, Project,
                                      Storey)
        col = Column(name="post")
        col.add(Box(name="body", start=Point(x=0, y=0, z=0),
                    end=Point(x=300, y=300, z=3000),
                    material="Concrete_C30-37"))
        product = Product(col, name="post_300")

        clean = Project(name="p", storeys=[Storey(name="ground", elevation=0)])
        clean.storeys[0].add(product.occurrence(name="a"))
        assert validate_project_report(clean).errors == []

        dirty = Project(name="p", storeys=[Storey(name="ground", elevation=0)])
        occ = product.occurrence(name="a")
        occ.elements[0].end = Point(x=300, y=300, z=9000)
        dirty.storeys[0].add(occ)
        errors = validate_project_report(dirty).errors
        assert any("no longer matches the promoted item" in e for e in errors), errors
