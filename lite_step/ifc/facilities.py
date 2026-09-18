"""IFC 4.3 SPATIAL structure — which classes exist, and what each one requires.

The ONE place the emitter and the DSL model read, mirroring
:mod:`lite_step.ifc.grids`, :mod:`lite_step.ifc.groupings`,
:mod:`lite_step.ifc.storeys` and :mod:`lite_step.ifc.product_types`. Two implementations of one idea agreeing *usually* is what produces the bug,
so every policy decision about the spatial hierarchy is decided HERE.

Spec: ``infrastructure/specs/ifc43-facilities.md``.

Why a separate mechanism from ``Element(ifc_class=)``
-----------------------------------------------------

``Element`` is the PHYSICAL path: whatever it emits is joined to the tree with
``IfcRelContainedInSpatialStructure``. A facility is a SPATIAL node, joined
with ``IfcRelAggregates``, and that relation's **WR31** says outright:

    The relationship object shall not be used to include other spatial
    structure elements into a spatial structure element.

So routing a facility through ``Element`` would not be merely clumsy — it
emits a schema violation. The DSL encodes the split independently: **zero** of ``ELEMENT_IFC_CLASS_WHITELIST``'s 37 entries are spatial
classes, and every spatial container we support has its own container.

**And the violation would be SILENT.** ``ifcopenshell.validate`` does not
report WHERE rules — measured with controls in
``tests/test_ifc43_conformance.py::test_where_rules_are_not_enforced_by_the_validator``
(and already known for ``IfcShapeModel``'s WR11). So the conformance gate
would stay green over an invalid file, and the downstream symptom is the shape: a node unreachable from ``getSpatialStructure()`` — missing from the
inspector rather than loud.

That measurement is why :func:`validate_nesting` is load-bearing rather than
belt-and-braces. It is the only thing standing between an author and that
failure.

Why the sets are STATIC, and how they stay true
------------------------------------------------

``lite_step`` compiles to IFC4X3 STEP **without ifcopenshell at runtime**:
every ``import ifcopenshell`` in the package is function-local. A ``SpatialElement`` validates at CONSTRUCTION
time, so it cannot ask the schema anything.

The sets below are therefore enumerated by hand and **verified against the real
schema by a test** (``test_spatial_element.py::test_the_sets_match_the_schema``),
which derives each one from ``IFC4X3_ADD2`` and fails if this file has drifted.
The rule is derived; the data is static; the test is what keeps those the same
thing.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------

#: ``IfcFacility`` subtypes, minus ``IfcBuilding`` (which the DSL derives from
#: storeys — see DEDICATED_CONTAINERS). ``IfcFacility`` itself is CONCRETE in
#: IFC 4.3 and is the honest choice for a built asset that is none of the
#: specialised kinds.
FACILITY_CLASSES = frozenset({
    "IfcFacility",
    "IfcBridge",
    "IfcRoad",
    "IfcRailway",
    "IfcMarineFacility",
})

#: ``IfcFacilityPart`` subtypes. ``IfcFacilityPart`` itself is ABSTRACT and so
#: is not here; ``IfcFacilityPartCommon`` is its concrete generic.
#:
#: Every one of these REQUIRES ``UsageType`` — that is the whole reason this
#: set exists separately from FACILITY_CLASSES.
FACILITY_PART_CLASSES = frozenset({
    "IfcFacilityPartCommon",
    "IfcBridgePart",
    "IfcRoadPart",
    "IfcRailwayPart",
    "IfcMarinePart",
})

#: Concrete spatial classes that are neither facility nor part. Kept because
#: excluding a concrete spatial class would be arbitrary; nothing in the
#: corpus uses them yet.
OTHER_SPATIAL_CLASSES = frozenset({
    "IfcSpatialZone",
    "IfcExternalSpatialElement",
})

#: Everything ``SpatialElement(ifc_class=)`` accepts.
SPATIAL_IFC_CLASS_WHITELIST = (
    FACILITY_CLASSES | FACILITY_PART_CLASSES | OTHER_SPATIAL_CLASSES
)

#: Spatial classes that already have a DSL container, mapped to the thing to
#: use instead. Refused rather than allowed, because a second spelling of one
#: idea is free to drift from the first — and because these carry
#: attributes ``SpatialElement`` does not model (``IfcBuildingStorey.Elevation``
#: is ``Storey(elevation=)``).
DEDICATED_CONTAINERS = {
    "IfcProject": "Project(...)",
    "IfcSite": "Site(...)",
    "IfcSpace": "Space(...)",
    "IfcBuildingStorey": "Storey(...)",
    "IfcBuilding": "Storey(...) — the IfcBuilding is derived from having storeys",
}

#: ``IfcFacilityUsageEnum``. Required on every part; see requires_usage.
FACILITY_USAGE_VALUES = frozenset({
    "LATERAL", "LONGITUDINAL", "REGION", "VERTICAL", "USERDEFINED", "NOTDEFINED",
})

#: The concrete ``IfcSpatialElement`` subtypes the DSL reaches through a
#: DEDICATED container rather than through ``SpatialElement`` — the keys of
#: :data:`DEDICATED_CONTAINERS`, minus ``IfcProject``, which is an
#: ``IfcContext`` and not an ``IfcSpatialElement`` at all.
DEDICATED_SPATIAL_CLASSES = frozenset({
    "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace",
})

#: **Every** concrete ``IfcSpatialElement`` this compiler can emit, whichever
#: DSL verb produced it. The union is what :func:`is_spatial_element` answers
#: on, and it is derived from IFC4X3_ADD2 by
#: ``test_spatial_element.py::test_the_sets_match_the_schema`` like every other
#: set here.
SPATIAL_ELEMENT_CLASSES = (
    SPATIAL_IFC_CLASS_WHITELIST | DEDICATED_SPATIAL_CLASSES
)

#: Uppercased, because an entity's class is spelled differently depending on
#: where it is read from — ``ifcopenshell`` returns ``"IfcSpace"`` from
#: ``is_a()``, a STEP line carries ``"IFCSPACE"``. Folding the case HERE is
#: what lets one predicate serve both; a caller normalizing on its own would
#: be the second implementation this module exists to prevent.
_SPATIAL_ELEMENT_CLASSES_UPPER = frozenset(
    c.upper() for c in SPATIAL_ELEMENT_CLASSES
)


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------


def is_facility(ifc_class: str) -> bool:
    """True for an ``IfcFacility`` subtype — the node that sits under the site."""
    return ifc_class in FACILITY_CLASSES


def is_facility_part(ifc_class: str) -> bool:
    """True for an ``IfcFacilityPart`` subtype — the node that holds elements."""
    return ifc_class in FACILITY_PART_CLASSES


def is_spatial_element(ifc_class: str) -> bool:
    """Whether an EMITTED entity joins its container by aggregation.

    The one question the emitter asks before wiring a child into a spatial
    container, and the reason it is asked of the emitted IFC CLASS rather than
    of the DSL type: ``IfcRelContainedInSpatialStructure``'s **WR31** is a
    statement about entities in a file —

        The relationship object shall not be used to include other spatial
        structure elements into a spatial structure element.

    — so whatever spelling produced an ``IfcSpace``, it is aggregated. A
    predicate over DSL classes (``isinstance(elem, Space)``) answers the same
    question for today's vocabulary and stops answering it the moment a second
    spelling reaches the same class.

    ``lite_step/models/taxonomy.py::is_spatial`` looks like this and is NOT
    it: that one classifies the DSL node (``Space``/``Site`` True,
    ``SpatialElement`` False, because a facility is not authored as content),
    which is the right answer to a different question and the wrong one here.
    Classifying on the DSL node instead lets an ``IfcSpace`` under a storey and
    an ``IfcSpace`` under a site emit ``IfcRelContainedInSpatialStructure``,
    and a ``Space`` inside an ``IfcFacilityPart`` do the same. Every one of
    those files passed ``ifcopenshell.validate`` at zero errors, because it
    does not report WHERE rules (measured, with controls, in
    ``tests/test_ifc43_conformance.py``). Nothing else was going to catch it.
    """
    return ifc_class.upper() in _SPATIAL_ELEMENT_CLASSES_UPPER


def requires_usage(ifc_class: str) -> bool:
    """Whether ``UsageType`` must be stated.

    Exactly the ``IfcFacilityPart`` subtypes: the schema marks ``UsageType``
    non-optional on ``IfcFacilityPart``, and optional-or-absent everywhere
    else. Note this is inverted from ``PredefinedType``, which is OPTIONAL on
    every spatial class including the facilities.
    """
    return is_facility_part(ifc_class)


# ---------------------------------------------------------------------------
# Validation — the loud half
# ---------------------------------------------------------------------------


def validate_ifc_class(ifc_class: str) -> str:
    """Refuse an unknown or dedicated-container class, naming the fix."""
    if ifc_class in DEDICATED_CONTAINERS:
        raise ValueError(
            f"SpatialElement: {ifc_class!r} already has a container — use "
            f"{DEDICATED_CONTAINERS[ifc_class]} instead. Two spellings of one "
            f"container drift apart; the dedicated one also carries attributes "
            f"SpatialElement does not model."
        )
    if ifc_class not in SPATIAL_IFC_CLASS_WHITELIST:
        raise ValueError(
            f"SpatialElement: unknown ifc_class {ifc_class!r}. Spatial classes "
            f"are {', '.join(sorted(SPATIAL_IFC_CLASS_WHITELIST))}. For a "
            f"PHYSICAL element (a wall, a beam, a pavement) use "
            f"Element(ifc_class=…) — a spatial node and a product join the "
            f"tree through different relations (IfcRelAggregates vs "
            f"IfcRelContainedInSpatialStructure), and IfcRel"
            f"ContainedInSpatialStructure's WR31 forbids the mix."
        )
    return ifc_class


def validate_usage(ifc_class: str, usage: Optional[str]) -> Optional[str]:
    """Require ``usage`` on parts, refuse it elsewhere, and check the value.

    Not defaulted on purpose. ``LONGITUDINAL`` vs ``LATERAL`` is a real claim
    about how the asset subdivides, and inventing one writes an assertion no
    author made — the ``Material(profile_mm=)`` lesson, where a plausible
    default is worse than a refusal because nothing downstream can tell it from
    a decision.

    The conformance gate DOES catch a missing mandatory attribute (measured),
    so this refusal has CI behind it rather than only convention — but failing
    at construction names the author's line instead of a STEP entity id.
    """
    if requires_usage(ifc_class):
        if usage is None:
            raise ValueError(
                f"SpatialElement({ifc_class}): usage= is required — IFC 4.3 "
                f"marks IfcFacilityPart.UsageType mandatory, and it is a real "
                f"statement about how the asset is subdivided, so there is no "
                f"safe default. One of: "
                f"{', '.join(sorted(FACILITY_USAGE_VALUES))}."
            )
    elif usage is not None:
        raise ValueError(
            f"SpatialElement({ifc_class}): usage= applies only to facility "
            f"PARTS ({', '.join(sorted(FACILITY_PART_CLASSES))}); "
            f"{ifc_class} has no UsageType attribute."
        )
    if usage is not None and usage not in FACILITY_USAGE_VALUES:
        raise ValueError(
            f"SpatialElement({ifc_class}): usage={usage!r} is not an "
            f"IfcFacilityUsageEnum value. One of: "
            f"{', '.join(sorted(FACILITY_USAGE_VALUES))}."
        )
    return usage


def validate_nesting(parent_class: Optional[str], child_class: str) -> None:
    """Refuse a spatial node in a container that cannot hold it.

    **This is the only defence.** ``ifcopenshell.validate`` does not report
    WR31 (measured), so a facility emitted in the wrong relation produces a
    file that passes the conformance gate and is invisible in the inspector.
    There is no second net under this one.

    ``parent_class`` is ``None`` for a node not yet added to anything.

    A part inside ANOTHER PART is legal and deliberate. IFC 4.3 composes an
    ``IfcFacilityPart`` out of further ``IfcFacilityPart`` s through the same
    ``IfcRelAggregates`` that joins the first one to its facility — a bridge's
    SUBSTRUCTURE holding its piers is the canonical shape — so the chain has
    no fixed depth. This pairing and the emitter's recursion
    (``generator._emit_spatial_node``) are one change: allowing the nesting
    without the recursion accepts a model the file then drops, and recursing
    without allowing it writes a branch nothing can reach.
    """
    if parent_class is None:
        return
    if (is_facility_part(child_class)
            and not (is_facility(parent_class) or is_facility_part(parent_class))):
        raise ValueError(
            f"SpatialElement({child_class}) must sit inside a facility "
            f"({', '.join(sorted(FACILITY_CLASSES))}) or another facility "
            f"part ({', '.join(sorted(FACILITY_PART_CLASSES))}), not "
            f"{parent_class}. A part outside its facility has nothing to "
            f"aggregate into, and the IFC validator will NOT tell you — it "
            f"does not check WHERE rules."
        )
    if is_facility(child_class) and is_facility_part(parent_class):
        raise ValueError(
            f"SpatialElement({child_class}) is a facility and cannot sit "
            f"inside {parent_class}, which is a part of one."
        )
