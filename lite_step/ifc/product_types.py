"""``Product`` typing — the ONE place the emitter reads (WS-B step 1).

A :class:`~lite_step.models.product.Product` is a finished, reusable catalog
item promoted from a built element (roadmap §V.5). Its occurrences are ordinary
tree members that emit their own geometry exactly as any other element does;
what this module adds is the **type relation**:

* one ``Ifc<Class>Type`` entity per Product — the catalog object, which lives
  OUTSIDE the spatial tree and has no canonical path (§V.2.3);
* one ``IfcRelDefinesByType`` linking that type to every occurrence.

WS-B **step 2** adds geometry SHARING on top, in the second half of this
module (see "Geometry sharing" below):

* one ``IfcRepresentationMap`` per shared body, held in the type's
  ``RepresentationMaps``;
* every occurrence's body becomes an ``IfcShapeRepresentation`` of
  ``RepresentationType='MappedRepresentation'`` carrying one
  ``IfcMappedItem`` that points back at the map.

Step 1 shipped separately, and on purpose: it buys the procurement value (a
scheduler can count "9 × window type nordic_1400x1600") with ZERO change to
the representation pipeline, so a geometry regression in step 2 can never be
confused with a typing bug in step 1.

**Why this module exists rather than two emitters.** Two implementations of
one idea agreeing *usually* is what produces the bug. The
class→type mapping, the refusal set, the STEP attribute tail and the
occurrence walk are all decided HERE; ``generator.py`` contributes only the
four lines that mint an entity.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Attributes an ``IfcTypeProduct`` carries before its subtype's own, in
#: schema order: GlobalId, OwnerHistory, Name, Description,
#: ApplicableOccurrence, HasPropertySets (IfcTypeObject), RepresentationMaps
#: (IfcTypeProduct), Tag, ElementType (IfcElementType). We fill GlobalId and
#: Name; the rest are optional and emit ``$``.
COMMON_TYPE_ATTR_COUNT = 9

#: Everything AFTER :data:`COMMON_TYPE_ATTR_COUNT`, per type entity, in schema
#: order — ``(attribute_name, value_or_None)``. ``None`` emits ``$``; a string
#: is an IFC enum member.
#:
#: The shape is uniform across the schema: every promotable element type ends
#: in a MANDATORY ``PredefinedType`` enum (so it cannot be ``$``), and exactly
#: two of them (Door, Window) carry a second mandatory enum plus two optional
#: attributes. ``NOTDEFINED`` is a member of every one of these enums —
#: verified against IFC4X3_ADD2, and re-derived from the live schema by
#: ``test_product_typing.py::TestSchemaTable`` so a whitelist addition cannot
#: silently emit a malformed entity.
DEFAULT_TYPE_TAIL: Tuple[Tuple[str, Optional[str]], ...] = (
    ("PredefinedType", "NOTDEFINED"),
)

TYPE_TAIL_OVERRIDES: Dict[str, Tuple[Tuple[str, Optional[str]], ...]] = {
    "IfcDoorType": (
        ("PredefinedType", "NOTDEFINED"),
        ("OperationType", "NOTDEFINED"),
        ("ParameterTakesPrecedence", None),
        ("UserDefinedOperationType", None),
    ),
    "IfcWindowType": (
        ("PredefinedType", "NOTDEFINED"),
        ("PartitioningType", "NOTDEFINED"),
        ("ParameterTakesPrecedence", None),
        ("UserDefinedPartitioningType", None),
    ),
}

#: IFC classes a DSL element can emit as that have **no ``Ifc<Class>Type``
#: counterpart in IFC4X3_ADD2**. Promotion refuses these at the ``Product(...)``
#: call rather than inventing an entity name the schema does not define — a
#: fabricated ``IFCEARTHWORKSFILLTYPE`` line parses as a STEP tuple and is
#: rejected (or silently dropped) only much later, by whichever viewer opens
#: the file.
#:
#: All seven are IFC4.3 geotechnical/earthworks classes, which is coherent:
#: they describe a mass of ground, and a mass of ground is not a catalog item
#: you order nine of. Pinned against the live schema by ``TestSchemaTable``.
IFC_CLASSES_WITHOUT_TYPE = frozenset({
    "IfcBorehole",
    "IfcEarthworksCut",
    "IfcEarthworksFill",
    "IfcGeomodel",
    "IfcGeoslice",
    "IfcGeotechnicalStratum",
    "IfcReinforcedSoil",
})


class ProductTypeError(ValueError):
    """A Product cannot be typed, or its occurrences cannot be resolved."""


def type_entity_for(ifc_class: str) -> str:
    """``"IfcWindow"`` -> ``"IfcWindowType"``, or raise naming the class.

    Raises :class:`ProductTypeError` for a class in
    :data:`IFC_CLASSES_WITHOUT_TYPE`. Called at PROMOTION time (loud, at the
    line that wrote ``Product(...)``), not at emission.
    """
    if ifc_class in IFC_CLASSES_WITHOUT_TYPE:
        raise ProductTypeError(
            f"{ifc_class} has no Ifc{ifc_class[3:]}Type counterpart in "
            f"IFC4X3_ADD2, so it cannot be promoted to a Product — there is "
            f"no catalog entity for the schema to point IfcRelDefinesByType "
            f"at. Classes without a type: "
            f"{', '.join(sorted(IFC_CLASSES_WITHOUT_TYPE))}."
        )
    return ifc_class + "Type"


def type_tail_attrs(type_entity: str) -> Tuple[Tuple[str, Optional[str]], ...]:
    """Attributes after the 9 common ones, in schema order."""
    return TYPE_TAIL_OVERRIDES.get(type_entity, DEFAULT_TYPE_TAIL)


def element_ifc_class(elem) -> str:
    """The IFC class an element emits as.

    Reuses :func:`lite_step.models.elements.container_ifc_class` — the sugar
    containers ARE their class with the ``Ifc`` prefix and ``Element`` states
    it outright, which is exactly the derivation needed here. A second dict
    would be the shape.
    """
    from lite_step.models.elements import container_ifc_class

    return container_ifc_class(elem)


@dataclass(frozen=True)
class ProductTypeGroup:
    """One ``Ifc<Class>Type`` + the occurrences it defines."""

    key: str                       #: catalog key — the Product's ``name=``
    ifc_class: str                 #: e.g. ``"IfcWindow"``
    type_entity: str               #: e.g. ``"IfcWindowType"``
    tail: Tuple[Tuple[str, Optional[str]], ...]
    occurrences: Tuple[str, ...]   #: occurrence canonical names, authoring order


#: Containment attributes the occurrence walk descends. Mirrors
#: ``compiler.naming._CONTAINMENT_ATTRS`` — an occurrence is a tree member, so
#: it is reached exactly the way a canonical name is derived.
_CONTAINMENT_ATTRS = ("_elements", "_openings")

#: Boolean-operand attributes. An occurrence found HERE is consumed into its
#: host's shape and emits no product, so it would silently vanish from the
#: type relation and undercount the schedule. Walked only to REFUSE.
_OPERAND_ATTRS = ("_cuts", "_adds", "_intersects", "_fills", "_voids")


def _walk(elem, containment: bool, out: List, seen: set) -> None:
    if id(elem) in seen:
        return
    seen.add(id(elem))

    key = getattr(elem, "_product_key", None)
    if key is not None:
        if not containment:
            raise ProductTypeError(
                f"the {type(elem).__name__} occurrence {key!r} is a boolean "
                f"operand (.difference()/.union()/.intersection()/.fills()/"
                f".void()). An operand is consumed into its host's shape and "
                f"emits no IFC product, so it would be counted in the "
                f"IfcRelDefinesByType of Product {key!r} while nothing in the "
                f"file corresponds to it. Place occurrences with .add() / "
                f".anchor(); build cutting tools from plain geometry."
            )
        out.append(elem)

    for attr in _CONTAINMENT_ATTRS:
        for child in getattr(elem, attr, None) or []:
            _walk(child, True, out, seen)
    for attr in _OPERAND_ATTRS:
        for child in getattr(elem, attr, None) or []:
            _walk(child, False, out, seen)


def collect_occurrences(proj) -> List[Any]:
    """Every placed Product occurrence in ``proj``, in authoring order.

    Walks storeys and sites through containment (``_elements`` /
    ``_openings``) — the same edges ``compiler.naming`` walks to derive a
    canonical name, because an occurrence's canonical IS how the emitters find
    the product it belongs to. Boolean operands are walked only so an
    occurrence hidden in one can be REFUSED (see :func:`_walk`).
    """
    found: List[Any] = []
    seen: set = set()
    for storey in getattr(proj, "storeys", None) or []:
        for elem in storey.elements:
            _walk(elem, True, found, seen)
    for site in getattr(proj, "sites", None) or []:
        _walk(site, True, found, seen)
    return found


def _serial_of(occ) -> Optional[int]:
    """The promotion serial an occurrence carries, or ``None``.

    Reads through ``_product`` rather than storing a second copy on the
    element: one fact, one home. ``None`` only if the element was marked as an
    occurrence without a Product, which no supported path does.
    """
    product = getattr(occ, "_product", None)
    return getattr(product, "serial", None) if product is not None else None


def collect_product_groups(proj) -> List[ProductTypeGroup]:
    """Group the project's occurrences into one entry per catalog key.

    Raises :class:`ProductTypeError` — never warns — when:

    * an occurrence carries no canonical name (it would be unreachable from
      the emitted products and drop out of the count);
    * two occurrences share a catalog key but come from different
      ``Product(...)`` promotions, or disagree on IFC class. One type entity
      would then stand for two catalog items, and a schedule reading it would
      count them as one — "9 × nordic" for two different windows.

    Both are the silent-degradation class this chapter refuses: a Product's
    entire value is that its occurrence count is TRUE.
    """
    order: List[str] = []
    by_key: Dict[str, Dict[str, Any]] = {}

    for occ in collect_occurrences(proj):
        key = occ._product_key
        canonical = getattr(occ, "_canonical_name", None)
        if not canonical:
            raise ProductTypeError(
                f"a {type(occ).__name__} occurrence of Product {key!r} has no "
                f"canonical name. Every occurrence is named at creation "
                f"(.occurrence(name=...)), so this means the naming pass did "
                f"not run over it — the occurrence would be invisible to "
                f"IfcRelDefinesByType and the type would undercount."
            )
        ifc_class = element_ifc_class(occ)
        entry = by_key.get(key)
        if entry is None:
            order.append(key)
            by_key[key] = {
                "ifc_class": ifc_class,
                "serial": _serial_of(occ),
                "first": canonical,
                "names": [canonical],
            }
            continue
        if entry["ifc_class"] != ifc_class:
            raise ProductTypeError(
                f"catalog key {key!r} is claimed by two different IFC classes "
                f"— {entry['ifc_class']} ({entry['first']}) and {ifc_class} "
                f"({canonical}). A Product is one catalog item; give the "
                f"second promotion its own name=."
            )
        serial = _serial_of(occ)
        if entry["serial"] is not None and serial is not None and serial != entry["serial"]:
            raise ProductTypeError(
                f"catalog key {key!r} is claimed by two separate Product(...) "
                f"promotions ({entry['first']} and {canonical}). Only ONE "
                f"{entry['ifc_class']}Type would be emitted, so a schedule "
                f"reading this file would count two catalog items as one — and "
                f"if the two promotions differ by so much as a millimetre the "
                f"count is silently wrong. The key IS the item's identity: "
                f"promote once and reuse the Product, or give the second "
                f"promotion its own name=."
            )
        entry["names"].append(canonical)

    groups: List[ProductTypeGroup] = []
    for key in order:
        entry = by_key[key]
        type_entity = type_entity_for(entry["ifc_class"])
        groups.append(ProductTypeGroup(
            key=key,
            ifc_class=entry["ifc_class"],
            type_entity=type_entity,
            tail=type_tail_attrs(type_entity),
            occurrences=tuple(entry["names"]),
        ))
    return groups


def missing_occurrence_error(group: "ProductTypeGroup", missing: List[str]) -> ProductTypeError:
    """The shared error the emitter raises when an occurrence emitted no product.

    Loud rather than a dropped name: an ``IfcRelDefinesByType`` listing 7 of 9
    occurrences is a procurement lie, and it is exactly the kind that reads as
    success in every log.
    """
    return ProductTypeError(
        f"Product {group.key!r}: no IFC product was emitted for "
        f"{len(missing)} of {len(group.occurrences)} occurrence(s) "
        f"({', '.join(missing)}). {group.type_entity} would define fewer "
        f"occurrences than were authored, so any schedule read off this file "
        f"would undercount. An occurrence must be placed somewhere that emits "
        f"its own product."
    )


# =============================================================================
# GEOMETRY SHARING — IfcRepresentationMap / IfcMappedItem (WS-B step 2)
# =============================================================================
#
# THE PLACEMENT DECISION, and why it is not the MappingTarget
# -----------------------------------------------------------
# Read the IFC 4.3 CONCEPT pages, not only the entity pages (that omission is
# how shipped a backwards aggregate):
#
# * *Product Local Placement* — "Product occurrences can be placed in 3D space
#   relative to where they are contained"; ``IfcProduct.ObjectPlacement``
#   establishes the occurrence's object coordinate system, and the shape
#   representation is read IN that system.
# * *Mapped Geometry* — the occurrence's representation is an
#   ``IfcShapeRepresentation`` with ``RepresentationType='MappedRepresentation'``
#   whose single item is an ``IfcMappedItem``. Nothing in that concept assigns
#   the occurrence's POSITION to the mapped item; ``MappingTarget`` is defined
#   on ``IfcMappedItem`` as what inserts the source definition into the
#   representation it is an item of — an intra-representation transform, the
#   block-insert of ISO 10303-43.
# * *Product Type Geometric Representation* — the maps live on
#   ``IfcTypeProduct.RepresentationMaps``, and "in order to utilize the
#   representation map at each occurrence of the product type, the product
#   occurrence has to use the concept 'Mapped Geometry'".
#
# So: **the occurrence's placement stays in its ``ObjectPlacement``, exactly
# where step 1 left it, and both ``MappingOrigin`` and ``MappingTarget`` are
# the IDENTITY.** Three reasons, in order of how expensive getting them wrong
# would be:
#
# 1. It is the only choice that cannot move anything. This compiler already
#    authors every body in the element's own local frame and puts the whole
#    world transform — anchor rotation included — in the ``IfcLocalPlacement``
#    (``generator._apply_dsl_placement``, which states outright that placements
#    are NOT baked into geometry). The bodies of two occurrences are therefore
#    already identical and the placements already correct; identity mapping
#    keeps every world coordinate bit-for-bit what step 1 emitted, which is
#    what ``test_product_geometry_sharing.py`` asserts through
#    ``ifcopenshell.geom``.
# 2. Moving the placement into ``MappingTarget`` would have to EMPTY
#    ``ObjectPlacement``, and that attribute is load-bearing far beyond this
#    element: ``PlacementRelTo`` chains to the storey (and the storey elevation
#    is subtracted through it), openings chain to their wall, fills chain to
#    their opening. An occurrence with no placement of its own detaches from
#    all of that. Leaving BOTH set double-applies the transform — every
#    occurrence at twice its offset, in a file that still opens.
# 3. ``MappingOrigin`` and ``MappingTarget`` compose differently in different
#    toolkits (origin, or origin inverse). At identity that ambiguity does not
#    exist. A non-identity pair would make our world coordinates a function of
#    the reader — the worst possible property for a catalog item.
#
# WHERE THE GEOMETRY ACTUALLY IS
# ------------------------------
# A Product occurrence is NOT one product with one body. Measured on this
# compiler: a ``Wall`` with one ``Box`` carries its own Body; a ``Wall`` with
# three carries a Body plus two aggregated ``IfcBuildingElementProxy`` children;
# a ``Column`` / ``Beam`` / ``Element(ifc_class=…)`` carries NO body at all and
# every part is an aggregated child. Sharing only "the occurrence's shape"
# would therefore share nothing whatsoever for the very case the roadmap
# measures (9 units × 18 joinery parts). So the unit of sharing is a SLOT:
# every representation-bearing product in the occurrence's aggregation subtree,
# matched between occurrences by its path through that subtree.

#: ``RepresentationType`` of an occurrence's mapped shape. From the *Mapped
#: Geometry* concept; ``RepresentationIdentifier`` is carried over from the
#: representation being shared ('Body' in every case this compiler emits).
MAPPED_REPRESENTATION_TYPE = "MappedRepresentation"

#: Depth cap on the structural walk of a representation. A representation is a
#: DAG in practice; a cycle would hang the compile, and hanging is the one
#: failure mode with no error message.
_FINGERPRINT_MAX_DEPTH = 64

#: The one entity that points AT a geometry item instead of being pointed at
#: by it. Named because :func:`purgeable_after_sharing` has to treat it as a
#: special case in both directions — see that docstring.
_STYLED_ITEM = "IFCSTYLEDITEM"


class ShapeAdapter:
    """What geometry sharing needs to know about one backend's entities.

    The emitter holds entirely different things — ``ifcopenshell`` C++
    instances on one side, :class:`~lite_step.ifc.ir.EntityDescriptor` on the
    other — but every *decision* below (which representations are slots, when
    two of them are the same geometry, what may be purged) has to be ONE
    decision. The adapter is the seam: a backend answers these seven
    questions, this module does all the reasoning.
    """

    def children(self, product) -> Sequence[Any]:
        """Aggregated child products, in emission order."""
        raise NotImplementedError

    def representations(self, product) -> Sequence[Tuple[Optional[str], Optional[str], Any]]:
        """``(RepresentationIdentifier, RepresentationType, handle)`` per shape."""
        raise NotImplementedError

    def name(self, product) -> Optional[str]:
        raise NotImplementedError

    def is_entity(self, value) -> bool:
        """True for a handle to another entity, as opposed to a STEP scalar."""
        raise NotImplementedError

    def key(self, handle):
        """A hashable identity for ``handle``, stable across lookups.

        Python's ``id()`` is NOT that identity, and assuming it was cost this
        module a silent half-purge on first run: ``ifcopenshell`` mints a FRESH
        wrapper object on every attribute read and every ``get_inverse`` call,
        so two wrappers for one STEP entity compare unequal by ``id`` while
        being the same thing. Every fixpoint below then decided "this referrer
        is outside the doomed set", kept everything alive, and left the
        duplicate bodies in the file — with the correct number of maps sitting
        next to them, which is what made it look like it had worked.
        """
        raise NotImplementedError

    def entity_type(self, handle) -> str:
        raise NotImplementedError

    def entity_attrs(self, handle) -> Sequence[Any]:
        """Forward attributes, in schema order. Inverses are never walked."""
        raise NotImplementedError

    def referrers(self, handle) -> Sequence[Any]:
        """Every entity holding a forward reference to ``handle``."""
        raise NotImplementedError


@dataclass(frozen=True)
class ShapeSlot:
    """One shareable representation, addressed by where it sits.

    ``path`` is the route through the occurrence's aggregation subtree — ``()``
    is the occurrence product itself, ``(2,)`` its third aggregated child — and
    ``index`` the position within that product's ``Representations`` list.
    Together they are the identity that makes "the same body on another
    occurrence" answerable; matching by flat list position alone would pair a
    window's sill with another window's glass the moment one part is reordered,
    and the file would still open.
    """

    path: Tuple[int, ...]
    index: int
    identifier: Optional[str]
    rep_type: Optional[str]
    owner: Any
    owner_name: Optional[str]
    rep: Any

    @property
    def address(self) -> Tuple[Tuple[int, ...], int, Optional[str], Optional[str]]:
        return (self.path, self.index, self.identifier, self.rep_type)

    def describe(self) -> str:
        where = ("the occurrence itself" if not self.path else
                 "aggregated child " + ".".join(str(i) for i in self.path))
        return (f"{self.identifier or '<no identifier>'}/"
                f"{self.rep_type or '<no type>'} #{self.index} on {where}"
                f" ({self.owner_name or '<unnamed>'})")


def occurrence_shape_slots(occurrence, adapter: ShapeAdapter) -> List[ShapeSlot]:
    """Every shareable representation under ``occurrence``, deterministically.

    Pre-order: a product's own representations before its children's, children
    in emission order. An already-mapped representation is skipped rather than
    re-wrapped — nesting a mapped item inside a mapped item is legal IFC, but
    here it could only mean this pass ran twice, and a second wrap would move
    nothing while doubling the entity count.
    """
    slots: List[ShapeSlot] = []
    seen: set = set()

    def walk(product, path: Tuple[int, ...]) -> None:
        marker = adapter.key(product)
        if marker in seen:
            return
        seen.add(marker)
        for index, (identifier, rep_type, rep) in enumerate(adapter.representations(product)):
            if rep_type == MAPPED_REPRESENTATION_TYPE:
                logger.warning(
                    "product geometry sharing: %s already carries a %s at "
                    "index %d — skipping it rather than nesting a mapped item "
                    "inside a mapped item",
                    adapter.name(product) or "<unnamed>",
                    MAPPED_REPRESENTATION_TYPE, index)
                continue
            slots.append(ShapeSlot(
                path=path, index=index, identifier=identifier,
                rep_type=rep_type, owner=product,
                owner_name=adapter.name(product), rep=rep))
        for child_index, child in enumerate(adapter.children(product)):
            walk(child, path + (child_index,))

    walk(occurrence, ())
    return slots


def _scalar_token(value) -> str:
    """A STEP scalar, spelled the same way every time.

    Floats are rounded before they are printed. Without that, two bodies that
    ARE the same geometry could fingerprint differently because one arrived by
    a different arithmetic order — a false geometry-mismatch refusal, which is
    the worst kind: it names the right suspect for the wrong reason.
    """
    if value is None:
        return "$"
    if isinstance(value, bool):
        return ".T." if value else ".F."
    if isinstance(value, float):
        return repr(round(value, 9) + 0.0)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return "'" + value + "'"
    # StepEnum / StepTyped / STEP_DERIVED — frozen dataclasses and sentinels
    # whose repr is stable and distinct.
    return repr(value)


def geometry_fingerprint(handle, adapter: ShapeAdapter) -> str:
    """A structural digest of a representation, ignoring entity IDs.

    Two representations share a fingerprint exactly when they are the same
    geometry expressed in the same local frame. That is what licenses sharing:
    an occurrence's emitted world geometry is ``ObjectPlacement ∘ body``, and
    the placements are untouched, so equal bodies ⇒ equal world geometry.

    It is a REFUSAL instrument, not an optimisation. A Product whose
    occurrences carry different bodies is a Product whose count is a lie — the
    same argument :func:`collect_product_groups` already makes about two
    promotions under one key — and unchecked it would show up as nine windows
    silently wearing the first one's shape.
    """
    parts: List[str] = []

    def walk(value, depth: int) -> None:
        if depth > _FINGERPRINT_MAX_DEPTH:
            raise ProductTypeError(
                f"a shared representation nests deeper than "
                f"{_FINGERPRINT_MAX_DEPTH} levels. Either the geometry is "
                f"pathological or the representation graph has a cycle; "
                f"either way the compile would hang instead of failing.")
        if adapter.is_entity(value):
            parts.append("(" + adapter.entity_type(value).upper())
            for attr in adapter.entity_attrs(value):
                walk(attr, depth + 1)
            parts.append(")")
        elif isinstance(value, (list, tuple)):
            parts.append("[")
            for item in value:
                walk(item, depth + 1)
            parts.append("]")
        else:
            parts.append(_scalar_token(value))

    walk(handle, 0)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SharedShape:
    """One ``IfcRepresentationMap``-to-be: a source slot and its followers."""

    source: ShapeSlot
    followers: Tuple[ShapeSlot, ...]

    @property
    def slots(self) -> Tuple[ShapeSlot, ...]:
        """Every slot that ends up carrying a ``MappedRepresentation``.

        The SOURCE is in here too. ``IfcShapeModel`` WR11 requires a shape
        representation to be used by an ``IfcProductRepresentation``, an
        ``IfcRepresentationMap`` or an ``IfcShapeAspect`` — so the moment the
        first occurrence's body becomes the map's ``MappedRepresentation`` it
        must STOP being that occurrence's own representation. Leaving it in
        both places is a schema violation that still renders correctly in every
        viewer, i.e. exactly the silent kind.
        """
        return (self.source,) + self.followers


def plan_shared_geometry(
    group: "ProductTypeGroup", occurrences: Sequence[Any], adapter: ShapeAdapter,
) -> List[SharedShape]:
    """Which representations become maps, and which slots point at them.

    ``occurrences`` are the emitted products, in ``group.occurrences`` order.

    **Occurrences are grouped into geometry VARIANTS, not assumed identical.**
    That is not defensive coding, it is a measured property of this compiler:
    an anchored child is baked into WORLD-AXIS-ALIGNED coordinates, so a window
    on a wall that runs along +Y emits ``IfcRectangleProfileDef(0.06, 1.4)``
    where the same window on a wall running along +X emits
    ``IfcRectangleProfileDef(1.4, 0.06)`` — with an IDENTITY rotation in the
    placement on both. The host's 90° is in the profile, not in the
    ``ObjectPlacement``. Nine windows spread over four facades are therefore
    genuinely four bodies, and a design that assumed one would have given three
    of the four facades the first facade's window, rotated wrong, in a file
    that opens.

    So each variant gets its own ``IfcRepresentationMap`` and the type carries
    several — which is exactly what ``IfcTypeProduct.RepresentationMaps`` being
    a LIST is for. Refusing instead was considered and rejected: sharing is an
    optimisation, and turning a missed optimisation into a compile error would
    make ``Product`` unusable for the roadmap's own motivating example (one
    window type, four facades).

    A variant of ONE still gets its map. It costs three entities and makes the
    catalog entry self-describing: a type whose ``RepresentationMaps`` is empty
    tells a downstream library nothing about the item it names.
    """
    if not occurrences:
        return []

    variants: Dict[Any, List[Tuple[str, List[ShapeSlot]]]] = {}
    order: List[Any] = []
    bodiless: List[str] = []

    for name, occurrence in zip(group.occurrences, occurrences):
        slots = occurrence_shape_slots(occurrence, adapter)
        if not slots:
            bodiless.append(name)
            continue
        signature = tuple((slot.address, geometry_fingerprint(slot.rep, adapter))
                          for slot in slots)
        if signature not in variants:
            variants[signature] = []
            order.append(signature)
        variants[signature].append((name, slots))

    if bodiless:
        logger.warning(
            "Product %r: %d of %d occurrence(s) carry no shape representation "
            "at all (%s) — nothing of theirs is shared. This is a Product of a "
            "geometry-less element, NOT a lookup that came back empty.",
            group.key, len(bodiless), len(group.occurrences),
            ", ".join(bodiless))
    if not order:
        logger.warning(
            "Product %r: no occurrence carries geometry, so %s."
            "RepresentationMaps stays unset.",
            group.key, group.type_entity)
        return []

    if len(order) > 1:
        _report_variants(group, variants, order)

    plan: List[SharedShape] = []
    for signature in order:
        members = variants[signature]
        first_slots = members[0][1]
        for index, source in enumerate(first_slots):
            plan.append(SharedShape(
                source=source,
                followers=tuple(slots[index] for _, slots in members[1:])))
    return plan


def _report_variants(group, variants, order) -> None:
    """Say — every compile — that this Product did not collapse to one body.

    Two levels, because the two causes are not equally suspicious:

    * same structure, different geometry — the host-orientation bake described
      on :func:`plan_shared_geometry`. Expected, so INFO; warning on every
      compile of a four-facade building is noise nobody reads.
    * different STRUCTURE (a different number of bodies, or a body in a
      different place in the subtree) — orientation cannot explain that.
      Something reached one occurrence and not another, and it is worth a
      WARNING naming both, because the alternative is that the type quietly
      speaks for a shape only some of its occurrences have.
    """
    shapes = {tuple(address for address, _ in signature) for signature in order}
    summary = "; ".join(
        f"{len(variants[signature])}×[{', '.join(n for n, _ in variants[signature])}]"
        for signature in order)
    if len(shapes) == 1:
        logger.info(
            "Product %r: its %d occurrences fall into %d geometry variants — "
            "%s. %s carries %d representation maps rather than one. The usual "
            "cause is host ORIENTATION: this compiler bakes an anchored "
            "child's rotation into its profile, not into its placement, so the "
            "same catalog item on two differently-aligned hosts is two bodies.",
            group.key, len(group.occurrences), len(order), summary,
            group.type_entity, len(order))
        return
    logger.warning(
        "Product %r: its occurrences do not even agree on the SHAPE of their "
        "representation subtree — %s. Occurrences are copies of one snapshot, "
        "so host orientation cannot explain this; something between promotion "
        "and emission (a carve, an opening, a consumed operand) reached some "
        "of them and not others. Each variant gets its own representation map, "
        "so nothing renders wrong — but the catalog item is not one item.",
        group.key, summary)


def purgeable_after_sharing(roots: Sequence[Any], adapter: ShapeAdapter) -> List[Any]:
    """Entities that become dead once ``roots`` stop being referenced.

    ``roots`` are the follower occurrences' now-replaced representations. Their
    bodies, and everything only those bodies reached, have to GO: an
    ``IfcShapeRepresentation`` used by neither a product shape, a representation
    map nor a shape aspect violates ``IfcShapeModel`` WR11, so leaving the husks
    behind would trade a duplicated body for a schema error. The conformance
    gate catches exactly that, which is why this function is not optional.

    Two traps it exists to avoid:

    * **Shared leaves must survive.** Profiles, points and directions are
      cached across the whole model; the fixpoint below drops a candidate the
      moment anything OUTSIDE the candidate set still refers to it.
    * **``IfcStyledItem`` is a referrer, not a referee.** It points AT the solid
      and nothing points at it, so a purge that only follows forward references
      leaves a styled item pointing into deleted space, while one that collects
      everything unreferenced deletes every style in the model. Styles are
      pulled in only when the thing they style is already doomed.
    """
    #: ``adapter.key`` throughout, never ``id()`` and never ``==`` — see
    #: :meth:`ShapeAdapter.key`. ``list.remove`` is avoided for the second half
    #: of that reason: ``EntityDescriptor`` is a plain dataclass, so two
    #: DIFFERENT entities that happen to carry equal attributes compare equal
    #: and ``remove`` would drop whichever came first.
    candidates: Dict[Any, Any] = {}

    def collect(handle, depth: int) -> None:
        marker = adapter.key(handle)
        if marker in candidates:
            return
        if depth > _FINGERPRINT_MAX_DEPTH:
            raise ProductTypeError(
                "the representation being purged nests deeper than "
                f"{_FINGERPRINT_MAX_DEPTH} levels — refusing to walk a graph "
                "that may be cyclic rather than hanging the compile.")
        candidates[marker] = handle
        for attr in adapter.entity_attrs(handle):
            for sub in _entity_values(attr, adapter):
                collect(sub, depth + 1)

    for root in roots:
        collect(root, 0)

    # Styles attach to a doomed item from the outside; take them with it.
    for handle in list(candidates.values()):
        for referrer in adapter.referrers(handle):
            if adapter.entity_type(referrer).upper() == _STYLED_ITEM:
                candidates.setdefault(adapter.key(referrer), referrer)

    changed = True
    while changed:
        changed = False
        for marker, handle in list(candidates.items()):
            if any(adapter.key(r) not in candidates
                   for r in adapter.referrers(handle)):
                del candidates[marker]
                changed = True

    return list(candidates.values())


def _entity_values(value, adapter: ShapeAdapter):
    if adapter.is_entity(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _entity_values(item, adapter)
