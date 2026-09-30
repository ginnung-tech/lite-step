"""
Main IFC Generator - Convert Project to IFC4 using ifcopenshell API.

This is the primary entry point for IFC generation.

Coordinate Convention:
- DSL uses Z-up (X=right, Y=forward, Z=up) — matches IFC
- IFC uses Z-up (X=right, Y=forward, Z=up)
- No axis swap needed (identity transform)

Unit Convention:
- Project object dimensions are in METERS (normalized by executor.py)
- Sweep dimensions from catalog are in MILLIMETERS (converted here)
- IFC output uses SI units (meters)

Geometry Rules:
- All meshes must be closed 3D solids — no single-sided surfaces
- Every mesh needs top, bottom, and side faces for correct rendering from all angles

Sweep Geometry:
- Sweep section centered in local XY plane
- Extrusion along local Z axis
- Placement at midpoint between start/end
- Orientation via lookAt: local Z points from start toward end
"""

import logging
import math
import sys
import tempfile
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

logger = logging.getLogger(__name__)

# Type checking imports for Pylance
if TYPE_CHECKING:
    pass

# Runtime imports with fallback
try:
    import ifcopenshell
    from ifcopenshell.api import run
    import numpy as np
    IFC_AVAILABLE = True
except ImportError:
    IFC_AVAILABLE = False
    ifcopenshell = None  # type: ignore[assignment]
    run = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

from lite_step.models import Project, Wall, Site, Door, Window, Box, Extrude, Sweep, Pipe, Revolve, Bar, Element, Mesh, Column, Beam, Slab, Roof, ReferencePoint, GuideLine, Space
from lite_step.models import BAR_TYPE_TO_IFC, collect_consumed_operand_ids
from lite_step.models import taxonomy as tx
from lite_step.compiler import frames
from lite_step.ifc.entity_cache import EntityCache, snap_to_precision
from lite_step.ifc.schema_version import IFC_OUTPUT_SCHEMA
from lite_step.ifc.geometry import (
    dsl_to_ifc_point as _dsl_to_ifc_point,
    axis_rotation_matrix_3x3 as _axis_rotation_matrix_3x3,
    extend_cut_for_boolean as _extend_cut_for_boolean,
    newell_normal as _newell_normal,
    apply_orientation_rules as _apply_orientation_rules,
    calculate_centroid as _calculate_centroid,
    segment_axes as _segment_axes,
    fillet_corner as _fillet_corner,
    facet_fillet_path as _facet_fillet_path,
    path_length_with_fillets as _path_length_with_fillets,
    orient_faces_consistently,
    weld_coincident_vertices,
    drop_coincident_faces,
    mesh_boundary_edge_count,
)
from lite_step.ifc import boolean_tree as _bt
from lite_step.ifc import grids
from lite_step.ifc import storeys


# ---------------------------------------------------------------------------
# Balanced boolean trees (see lite_step/ifc/boolean_tree.py for the why)
# ---------------------------------------------------------------------------


def _bool_factory(model):
    """``create_entity(operator, first, second)`` bound to an ifcopenshell model."""
    def _create(operator, first, second):
        return model.create_entity("IfcBooleanResult", Operator=operator,
                                   FirstOperand=first, SecondOperand=second)
    return _create


def _balanced_union_ios(model, operands):
    """Balanced ``IfcBooleanResult UNION`` over already-built solids.

    Single operand → returned unchanged, no entity created."""
    return _bt.balanced_union(_bool_factory(model), operands)


def _apply_boolean_chain_ios(model, base, *, build, cuts=None, adds=None,
                             intersects=None):
    """``boolean_tree.apply_boolean_chain`` on the ifcopenshell backend."""
    return _bt.apply_boolean_chain(_bool_factory(model), base, build=build,
                                   cuts=cuts, adds=adds, intersects=intersects)


@dataclass
class IFCGenerationResult:
    """
    Result of IFC generation.

    Attributes:
        success: True if IFC was generated successfully
        ifc_content: The IFC STEP file content as string
        stats: Statistics about generated elements
        error: Error message if generation failed
    """
    success: bool
    ifc_content: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# =============================================================================
# FAST-PATH HELPERS (bypass ifcopenshell.api.run overhead)
# =============================================================================

def _warn_child_dropped(container_kind: str, container_name, child):
    """Loud breadcrumb when a container child produced no IFC geometry.

    Distinguishes "child skipped, no geometry emitted" from silent loss —
    every drop path routes through here so the reason is identifiable in
    logs (loud-failure > silent-degradation).
    """
    logger.warning(
        "%s %r: child %s %r produced no geometry and was dropped",
        container_kind,
        container_name or "<anonymous>",
        type(child).__name__,
        getattr(child, "ifc_name", None) or "<anonymous>",
    )


def _create_child_product(model, context, child, cache: EntityCache):
    """Emit ONE container child as its own IFC product — whatever its type.

    **The single generic child path.** Every container's emitter keeps its own
    branches for the types it has always styled itself (they carry the
    container auto-color and the layer slicing), and routes everything else
    here. Any type can be anchored into any container, so a drop here would
    trade a clear compile-time refusal for a silent hole in the IFC, the
    failure this codebase ranks below a crash.

    Returns ``None`` only for a type with no emitter at all; the caller warns.
    ``Window``/``Door`` are deliberately NOT handled: an opening is not a
    placed child but a size-only void cut into its host at full thickness
    (``_create_wall_openings_from_assembly`` / ``_process_opening_children``),
    and folding it in here would erase that.

    Pinned by ``lite_step/tests/test_anchor_matrix.py``, which asserts every
    container x type pairing puts geometry in the file — not merely that it
    compiles, which a silent drop also does.
    """
    if tx.is_prism(child):
        return _create_solid(model, context, child, cache)
    if isinstance(child, Sweep):
        return _create_sweep(model, context, child, cache)
    if isinstance(child, Revolve):
        return _create_revolve(model, context, child, cache)
    if isinstance(child, Pipe):
        return _create_pipe(model, context, child, cache)
    if isinstance(child, Bar):
        return _create_bar(model, context, child, cache)
    if isinstance(child, Mesh):
        return _create_mesh(model, context, child, cache)
    if isinstance(child, Element):
        return _create_element_generic(model, context, child, cache)
    if isinstance(child, Wall):
        return _create_wall(model, context, child, cache)
    if isinstance(child, Column):
        return _create_column(model, context, child, cache)
    if isinstance(child, Beam):
        return _create_beam(model, context, child, cache)
    if isinstance(child, Slab):
        return _create_slab(model, context, child, cache)
    if isinstance(child, Roof):
        return _create_roof(model, context, child, cache)
    if isinstance(child, Space):
        return _create_space(model, context, child, cache)
    return None


def _emit_offtable_children(model, context, container, children, cache: EntityCache,
                            *, kind: str, handled) -> list:
    """Products for the children a container's own branches did not take.

    ``handled`` is the predicate naming what the caller already emitted, so the
    two never double-emit. Anything else goes through
    :func:`_create_child_product`; a type with no emitter at all is the one
    case that still warns.
    """
    products = []
    for child in children:
        if handled(child):
            continue
        product = _create_child_product(model, context, child, cache)
        if product is None:
            logger.warning(
                "%s %r: child %s %r has no IFC emitter and was skipped",
                kind, container.ifc_name or "<anonymous>",
                type(child).__name__,
                getattr(child, "ifc_name", None) or "<anonymous>",
            )
            continue
        cache.record_child_product(container, child, product)
        products.append(product)
    return products


#: Statistics key per DSL type, for the Site children that reach the file
#: through :func:`_emit_offtable_children`. Same key names the storey loop
#: uses for the same types, because ``stats`` is one namespace and
#: ``total_elements`` sums it — a Site child counted under a new key would be
#: emitted and still missing from the count the user is shown.
_SITE_OFFTABLE_STAT_KEY = {
    "Wall": "walls", "Column": "columns", "Beam": "beams", "Slab": "slabs",
    "Roof": "roofs", "Space": "spaces", "Sweep": "sweeps", "Pipe": "pipes",
    "Revolve": "revolves", "Bar": "bars",
}


def _count_site_offtable_children(site_container, cache: EntityCache, stats) -> None:
    """Count the Site children :func:`_emit_offtable_children` just emitted.

    Read back off the cache rather than counted in the loop because
    ``_emit_offtable_children`` returns products, not ``(child, product)``
    pairs — and it is the only writer of ``record_child_product`` for a Site,
    since the four on-table branches record nothing.

    They must stay counted: ``total_elements`` is read as the number of things
    in the model, and an off-table child is one of them.
    """
    for child, _product in cache.child_products(site_container):
        key = _SITE_OFFTABLE_STAT_KEY.get(type(child).__name__)
        if key is None:
            continue
        stats.setdefault(key, 0)
        stats[key] += 1


def _drawings_enabled() -> bool:
    """Whether to emit the 2D drawing wiring (default ON).

    On by default because the whole point is that a downstream BIM tool needs
    no setup — an opt-in flag nobody knows about would defeat it. The
    kill-switch exists for byte-comparison runs (corpus hash gates) and for
    any consumer that turns out to choke on annotation products.
    """
    return os.environ.get("LITESTEP_EMIT_DRAWINGS", "1") not in ("0", "false", "False")


def _create_ifc_entity(model, ifc_class: str, name: str = None, predefined_type: str = None):
    """
    Create a rooted IFC entity with GlobalId, bypassing run("root.create_entity").

    The high-level API adds: GlobalId, OwnerHistory (None in IFC4), Name,
    PredefinedType handling, and IFC2X3 defaults (unused — we use IFC4).
    """
    element = model.create_entity(ifc_class, GlobalId=ifcopenshell.guid.new())
    if name:
        element.Name = name
    if predefined_type:
        try:
            element.PredefinedType = predefined_type
        except Exception:
            element.PredefinedType = "USERDEFINED"
            if hasattr(element, "ObjectType"):
                element.ObjectType = predefined_type
    return element


def _axis2_placement_from_matrix(model, matrix):
    """
    Build a fresh IfcAxis2Placement3D from a 4x4 matrix.

    Note: Direction and point entities are created fresh (not cached) for placements because
    ifcopenshell's C++ layer can segfault when the same Direction entity is shared across
    multiple IfcAxis2Placement3D entities. The post-processing deduplicator will merge these.
    """
    pos = (float(matrix[0][3]), float(matrix[1][3]), float(matrix[2][3]))
    x_ax = (float(matrix[0][0]), float(matrix[1][0]), float(matrix[2][0]))
    z_ax = (float(matrix[0][2]), float(matrix[1][2]), float(matrix[2][2]))

    # Create fresh entities for placements (not cached - see docstring)
    origin = model.create_entity("IfcCartesianPoint", Coordinates=list(pos))

    is_identity_rot = (
        abs(x_ax[0] - 1.0) < 1e-9 and abs(x_ax[1]) < 1e-9 and abs(x_ax[2]) < 1e-9
        and abs(z_ax[0]) < 1e-9 and abs(z_ax[1]) < 1e-9 and abs(z_ax[2] - 1.0) < 1e-9
    )

    if is_identity_rot:
        return model.create_entity("IfcAxis2Placement3D", Location=origin)
    axis = model.create_entity("IfcDirection", DirectionRatios=list(z_ax))
    ref_dir = model.create_entity("IfcDirection", DirectionRatios=list(x_ax))
    return model.create_entity(
        "IfcAxis2Placement3D", Location=origin,
        Axis=axis,
        RefDirection=ref_dir
    )


def _set_placement(model, product, matrix, cache: EntityCache):
    """
    Set object placement from a 4x4 numpy matrix, bypassing run("geometry.edit_object_placement").

    The high-level API does: unit scale (1.0 for METRE), ShapeBuilder instantiation,
    inverse-relation walking for parent placement, matrix inversion for relative coords,
    old placement removal. All unnecessary since we position absolutely and call once.
    """
    product.ObjectPlacement = model.create_entity(
        "IfcLocalPlacement", RelativePlacement=_axis2_placement_from_matrix(model, matrix)
    )


def _collect_placement_family(product):
    """The product + every product rigidly carried by it (WS1 PR-F).

    Walks aggregated children (IfcRelAggregates), opening voids
    (IfcRelVoidsElement) and their fills (IfcRelFillsElement) — the whole
    subgraph a DSL placement must move as one rigid body. Members whose
    placements CHAIN (PlacementRelTo set — e.g. window frame members after
    the PR-E / identity-relative chaining) are returned too; the
    caller skips them so they follow their parent automatically.
    """
    seen: set = set()
    order = []

    def walk(p):
        if p.id() in seen:
            return
        seen.add(p.id())
        order.append(p)
        for rel in (getattr(p, "IsDecomposedBy", None) or []):
            for child in (rel.RelatedObjects or []):
                walk(child)
        for rel in (getattr(p, "HasOpenings", None) or []):
            walk(rel.RelatedOpeningElement)
        for rel in (getattr(p, "HasFillings", None) or []):
            walk(rel.RelatedBuildingElement)

    walk(product)
    return order


def _apply_dsl_placement(model, product, matrix, cache: EntityCache):
    """Compose a resolved DSL placement (4x4 world matrix) ON TOP of the
    product's own object placement (WS1 PR-F placement engine).

    APPLICATION-POINT DECISION (pinned): placements are applied as IFC
    local placements, NOT baked into DSL geometry, because baking cannot
    express a rigid transform on this generator:

    * a rotated box-mode Solid is unrepresentable in ``start``/``end``,
      and ``_create_wall`` / ``_create_wall_openings_from_assembly`` derive
      the opening math from an AXIS-ALIGNED body box;
    * ``_segment_axes`` (path Sweep) and the Revolve start plane re-derive
      section orientation from WORLD-frame references — baked path points
      would silently change the section roll / revolve start plane instead
      of rotating the element rigidly.

    Composition is ``world = M_place @ M_own``: the product's authored
    coordinates are the element's LOCAL frame (geometry authored local to
    the placement origin, per the spec composition rule). Each family
    member's existing IfcLocalPlacement is MUTATED in place (fresh
    IfcAxis2Placement3D, same IfcLocalPlacement entity) so PlacementRelTo
    chains — window frame members,-adjacent — keep pointing at the
    right entity and follow automatically; chained members are skipped.
    ``_set_relative_placements`` later re-parents the top product to its
    storey and adjusts Location z, which stays correct because the
    composed translation lives in Location.
    """
    import ifcopenshell.util.placement as _plc
    assert np is not None

    m_place = np.asarray(matrix, dtype=np.float64)
    for member in _collect_placement_family(product):
        placement = getattr(member, "ObjectPlacement", None)
        if placement is None:
            # No own placement: authored coordinates ARE the local frame —
            # the resolved matrix is the placement.
            _set_placement(model, member, m_place, cache)
            continue
        if getattr(placement, "PlacementRelTo", None) is not None:
            continue  # chained within the family — follows its parent
        m_own = _plc.get_local_placement(placement)
        composed = m_place @ m_own
        placement.RelativePlacement = _axis2_placement_from_matrix(model, composed)


def _emit_facility(model, body_context, ifc_site, facility, cache, stats):
    """Emit a ``SpatialElement`` facility and its parts under ``IfcSite``.

    The chain, from ``infrastructure/specs/ifc43-facilities.md``::

        IfcSite --aggregate--> FACILITY --aggregate--> PART --contain--> products

    Two relations, and the boundary is a formal rule rather than a
    convention: spatial-to-spatial is ``IfcRelAggregates``; spatial-to-product
    is ``IfcRelContainedInSpatialStructure``, whose **WR31** states that it
    "shall not be used to include other spatial structure elements into a
    spatial structure element".

    Getting that backwards is the failure this function exists to avoid, and
    it is INVISIBLE to every gate we have: ``ifcopenshell.validate`` does not
    report WHERE rules (measured with controls in
    ``tests/test_ifc43_conformance.py``), so the file would pass conformance
    at zero errors and the facility would simply not appear in
    ``getSpatialStructure()`` — the symptom, missing from the
    inspector rather than loud.

    The PART row is not one level deep. IFC 4.3 composes an
    ``IfcFacilityPart`` out of further ``IfcFacilityPart`` s through the same
    ``IfcRelAggregates`` — a bridge's SUBSTRUCTURE holding its piers, a road's
    section holding its carriageways — so the chain above is the SHORTEST
    legal shape, not the only one. :func:`_emit_spatial_node` therefore
    recurses instead of running a fixed two-level loop, and containment fires
    only at the node where products actually sit.

    Emitted AFTER the manifest build for the v21.2 ``IfcAnnotation`` reason:
    a facility is an ``IfcProduct``, so a named one would otherwise land in
    ``LITESTEP_META`` as an entity owning no DSL element. (This runs inside
    the site-children pass, which is already after that build.)
    """
    _emit_spatial_node(model, body_context, ifc_site, facility, cache, stats,
                       stat_key="facilities")


def _emit_spatial_node(model, body_context, parent_ent, node, cache, stats,
                       *, stat_key):
    """Emit ONE spatial node under ``parent_ent``, then recurse into its own.

    ``parent_ent`` is the ``IfcSite`` for a facility and the enclosing spatial
    entity for anything below it; ``stat_key`` is ``"facilities"`` for that
    top row and ``"facility_parts"`` for every node under it, which keeps the
    two counters meaning what they meant when this was a two-level loop.

    Why recursion rather than a third hand-written level: a fixed depth is a
    limit nothing announces. Dropping a part-inside-a-part leaves a warning as
    the ONLY trace — the emitted
    file still validates at zero errors (``ifcopenshell.validate`` does not
    report WHERE rules, measured), so the sub-part and everything standing in
    it simply vanish from ``getSpatialStructure()``. That is again,
    and the reason ``test_spatial_element.py`` asserts REACHABILITY through
    ``get_decomposition`` rather than counting entities: a count passes on the
    broken version too.

    Spatial children AGGREGATE (``IfcRelAggregates``); product children are
    CONTAINED, and only where they actually are. A node whose children are all
    spatial gets no ``IfcRelContainedInSpatialStructure`` at all, which is
    what WR31 requires — hoisting a leaf's products to the facility would put
    a product in the wrong storey-equivalent and read as legal.
    """
    from lite_step.ifc import facilities as fac

    # predefined_type rides the constructor: it already falls back to
    # USERDEFINED + ObjectType for a value the class's enum does not know,
    # which is the sanctioned mechanism and never a silent drop.
    ent = _create_ifc_entity(model, node.ifc_class, name=node.ifc_name,
                             predefined_type=node.predefined_type)
    _set_identity_placement(model, ent)
    ent.ObjectPlacement.PlacementRelTo = parent_ent.ObjectPlacement
    if node.long_name is not None:
        ent.LongName = node.long_name
    # UsageType is MANDATORY on IfcFacilityPart — the model layer refuses to
    # construct a part without it, so this cannot be None here. False for a
    # facility and for the non-facility spatial classes, which have no such
    # attribute at all.
    if fac.requires_usage(node.ifc_class):
        ent.UsageType = node.usage
    _assign_aggregate(model, parent_ent, [ent])
    stats.setdefault(stat_key, 0)
    stats[stat_key] += 1

    # A FACILITY decomposes and holds no products; a PART (or any other
    # spatial node) holds them. That is the model layer's own split
    # (``_FACILITY_CHILDREN`` vs ``_FACILITY_PART_CHILDREN``), read the same
    # way here so the two cannot drift: a product handed to a facility is
    # off-table at ``.add()`` and stays a loud drop rather than becoming a
    # containment relation the author never asked for.
    holds_products = not fac.is_facility(node.ifc_class)
    kind = "facility" if not holds_products else "facility part"

    products = []
    for child in node.elements:
        if type(child).__name__ == "SpatialElement":
            _emit_spatial_node(model, body_context, ent, child, cache, stats,
                               stat_key="facility_parts")
            continue
        if not holds_products:
            _warn_child_dropped(kind, node.ifc_name, child)
            continue
        # The generic path, not a hand-rolled subset: a part's ``.add()``
        # table is ``_FACILITY_PART_CHILDREN``, and every type in it must
        # reach the file. Three branches (prism/Mesh/Element) covered less
        # than half of that table, so a Sweep/Revolve/Pipe/Bar member —
        # what a steel bridge is ENTIRELY made of — was accepted at
        # ``.add()`` and dropped here.
        p = _create_child_product(model, body_context, child, cache)
        if p:
            products.append(p)
        else:
            _warn_child_dropped(kind, node.ifc_name, child)
    # Joined here, and only here. Mostly products, so mostly containment —
    # but a part's ``.add()`` row includes ``Space`` (``_PRODUCT_CHILDREN``),
    # and an IfcSpace in a containment relation is the WR31 violation this
    # function's docstring is about. Measurement found a Space under an
    # IfcBridgePart doing exactly that; the split lives in
    # :func:`_assign_spatial_children` so this site cannot drift from the
    # storey and site passes. No-op when the node only decomposes.
    _assign_spatial_children(model, ent, products)


def _assign_aggregate(model, relating_object, products):
    """
    Create IfcRelAggregates directly, bypassing run("aggregate.assign_object").

    Skips: previous aggregate check, container unassignment, placement re-localization.
    Safe because products are freshly created and not yet assigned anywhere.

    **The bypass is now load-bearing for DETERMINISM, not only for speed.**
    ``aggregate.assign_object`` builds its members as ``list(set(...) | products)``
    (assign_object.py:136), and ``entity_instance.__hash__`` keys on the model's
    heap address — so routing through the API would make ``RelatedObjects`` order
    vary per compile. Measured on a real 6-member set: **10 distinct orders over
    200 fresh files**; a 2-member group is a coin flip. That is element order in
    the spatial tree, which schedules and inspector trees read. See
    :func:`_canonical_unit_order` for the same defect where we could not bypass.
    """
    if not products:
        return
    model.create_entity(
        "IfcRelAggregates",
        GlobalId=ifcopenshell.guid.new(),
        RelatingObject=relating_object,
        RelatedObjects=list(products)
    )


def _assign_container(model, relating_structure, products):
    """
    Create IfcRelContainedInSpatialStructure directly, bypassing run("spatial.assign_container").

    Skips: previous container check, aggregate unassignment, placement re-localization.
    Safe because products are freshly created and not yet assigned anywhere.

    **The bypass is now load-bearing for DETERMINISM, not only for speed** —
    ``spatial.assign_container`` carries the identical ``list(set(...))`` pattern
    (assign_container.py:152). See :func:`_assign_aggregate` for the measurement.
    """
    if not products:
        return
    model.create_entity(
        "IfcRelContainedInSpatialStructure",
        GlobalId=ifcopenshell.guid.new(),
        RelatingStructure=relating_structure,
        RelatedElements=list(products)
    )


def _canonical_unit_order(model) -> None:
    """Pin ``IfcUnitAssignment.Units`` to unit CREATION order.

    ``ifcopenshell.api.unit.assign_unit`` ends with
    ``unit_assignment.Units = list(units)`` over a Python **set** of
    ``entity_instance``, and ``entity_instance.__hash__`` is
    ``hash((step_id, wrapped_data.file_pointer()))`` — the C++ ``IfcFile``'s
    heap ADDRESS. The step ids are fixed (``#2``/``#3``/``#4`` =
    LENGTH/AREA/VOLUME), so the address is the only varying term, and every
    ``ifcopenshell.file()`` lands somewhere new. The SAME model compiled twice
    in ONE process therefore emits ``(#2,#3,#4)`` or ``(#3,#4,#2)`` or
    ``(#4,#2,#3)``.

    **No hash seed pins this.** Measured: 12 compiles in one process at
    ``PYTHONHASHSEED=0`` still gave 3 orderings, and 6 separate subprocesses
    gave 3 — so it is address-derived, not string-hash randomisation. (Positive
    proof: rebuilding ``list(set(...))`` from pure-Python stand-ins whose
    ``__hash__`` is ``hash((step_id, file_pointer))`` predicted the real emitted
    order 20/20.)

    ``Units`` is an IFC SET, so every order is equally VALID — this makes it
    REPRODUCIBLE, which is what a committed build product and a byte diff need.
    Sorting on ``id()`` keys on precisely the term the shuffle does not touch,
    and creation order is (LENGTH, AREA, VOLUME), so ``_ensure_si_unit``'s
    later RADIAN/MASSUNIT (higher ids) stay at the tail with no second rule.
    """
    for unit_assignment in model.by_type("IfcUnitAssignment"):
        if unit_assignment.Units:
            unit_assignment.Units = sorted(unit_assignment.Units,
                                           key=lambda u: u.id())


def _assign_spatial_children(model, structure, children) -> None:
    """Join ``children`` to a SPATIAL ``structure`` by the relation each needs.

    The ONE place this backend decides it. A product is CONTAINED
    (``IfcRelContainedInSpatialStructure``); a child that is itself an
    ``IfcSpatialElement`` — an ``IfcSpace``, a facility — is AGGREGATED,
    because WR31 on the containment relation states that it "shall not be used
    to include other spatial structure elements into a spatial structure
    element".

    Split on the EMITTED class through
    :func:`lite_step.ifc.facilities.is_spatial_element`, not on the DSL type:
    the rule is about what is in the file, so any spelling that reaches
    ``IfcSpace`` follows without a second branch. The predicate is shared, which
    is the point: four hand-written ``if`` statements in four call sites is the
    failure mode it exists to prevent.

    Order matters for neither relation, but containment is emitted first so
    document order is unchanged for every model with no spatial child.
    """
    from lite_step.ifc import facilities as _fac

    contained = [c for c in children if not _fac.is_spatial_element(c.is_a())]
    spatial = [c for c in children if _fac.is_spatial_element(c.is_a())]
    _assign_container(model, structure, contained)
    _assign_aggregate(model, structure, spatial)


def _set_identity_placement(model, product):
    """Create an identity placement (origin, no rotation) for spatial structure elements."""
    origin = model.create_entity("IfcCartesianPoint", Coordinates=[0.0, 0.0, 0.0])
    ax2 = model.create_entity("IfcAxis2Placement3D", Location=origin)
    product.ObjectPlacement = model.create_entity("IfcLocalPlacement", RelativePlacement=ax2)


def _set_relative_placements(products, parent_placement, parent_elevation=0.0):
    """
    Set PlacementRelTo on products for OJP001 compliance and adjust Z for storey elevation.

    IFC interprets coordinates relative to PlacementRelTo. Since we use absolute coords,
    we subtract the parent's elevation to keep world-space positions correct.
    """
    for product in products:
        placement = getattr(product, 'ObjectPlacement', None)
        if placement:
            placement.PlacementRelTo = parent_placement
            if parent_elevation != 0.0:
                rel = placement.RelativePlacement
                if rel and rel.Location:
                    coords = list(rel.Location.Coordinates)
                    coords[2] -= parent_elevation
                    rel.Location.Coordinates = coords


def _create_extruded_body(model, context, profile, depth, cache: EntityCache, z_offset: float = 0.0):
    """
    Create extruded body representation directly, bypassing run("geometry.add_profile_representation").

    The high-level API creates: IfcAxis2Placement3D + 2x uncached IfcDirection +
    IfcExtrudedAreaSolid + IfcShapeRepresentation. We do the same but use cached entities.

    Returns:
        Tuple of (IfcShapeRepresentation, IfcExtrudedAreaSolid) for coloring the solid.
    """
    if z_offset != 0.0:
        prof_origin = cache.get_or_create_point([0.0, 0.0, z_offset])
    else:
        prof_origin = cache.get_or_create_point([0.0, 0.0, 0.0])
    prof_placement = model.create_entity("IfcAxis2Placement3D", Location=prof_origin)

    solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=prof_placement,
        ExtrudedDirection=cache.get_z_up_direction(),
        Depth=float(depth)
    )

    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[solid]
    )

    return shape_rep, solid


def _assign_body_representation(model, product, shape_rep):
    """
    Assign representation to product, bypassing run("geometry.assign_representation").

    Creates IfcProductDefinitionShape and assigns it. For IfcProduct instances only
    (not IfcTypeProduct which needs IfcRepresentationMap — not used by us).
    """
    product.Representation = model.create_entity(
        "IfcProductDefinitionShape",
        Representations=[shape_rep]
    )


def _tm_utm(lat: float, lng: float, central_meridian: float) -> tuple[float, float]:
    """Transverse-Mercator (Redfearn series) easting/northing on GRS80/WGS84.

    UTM scale factor 0.9996, 500 km false easting, northern-hemisphere
    northing (the 10 000 km southern false northing is the caller's job).
    Millimetre-accurate within ~3 degrees of ``central_meridian`` and still
    sub-centimetre at the 6 degree zone edge.
    """
    import math as _math
    a = 6_378_137.0
    f = 1 / 298.257_223_563
    k0 = 0.9996
    e2 = 2 * f - f * f
    ep2 = e2 / (1 - e2)

    phi = _math.radians(lat)
    lam = _math.radians(lng - central_meridian)
    sin_phi, cos_phi, tan_phi = _math.sin(phi), _math.cos(phi), _math.tan(phi)

    n_rad = a / _math.sqrt(1 - e2 * sin_phi * sin_phi)
    t = tan_phi * tan_phi
    c = ep2 * cos_phi * cos_phi
    aa = lam * cos_phi
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * _math.sin(2 * phi)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * _math.sin(4 * phi)
        - (35 * e2**3 / 3072) * _math.sin(6 * phi)
    )
    easting = 500_000 + k0 * n_rad * (
        aa
        + (1 - t + c) * aa**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * aa**5 / 120
    )
    northing = k0 * (
        m + n_rad * tan_phi * (
            aa**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720
        )
    )
    return easting, northing


#: Denmark (incl. Bornholm) lat/lng box. Inside it the national CRS applies:
#: ETRS89 / UTM zone 32N (EPSG:25832) for ALL of it, even east of 12 E where
#: the longitude-derived zone would be 33 — one CRS per country is what
#: Danish surveyors and the site pipeline (DHM, LER) use.
_DK_LAT = (54.5, 57.8)
_DK_LNG = (8.0, 15.3)


def _utm_crs_for(lat: float, lng: float) -> tuple[int, int, str, str]:
    """``(zone, epsg, datum, description)`` for a WGS84 point.

    Rule: inside the Danish box -> ETRS89 / UTM 32N (EPSG:25832); elsewhere
    the UTM zone from longitude, WGS 84 / UTM zone N (EPSG:326zz) or S (327zz).
    """
    if _DK_LAT[0] <= lat <= _DK_LAT[1] and _DK_LNG[0] <= lng <= _DK_LNG[1]:
        return 32, 25832, "ETRS89", "ETRS89 / UTM zone 32N"
    zone = min(60, max(1, int((lng + 180) // 6) + 1))
    hemisphere = "N" if lat >= 0 else "S"
    epsg = (32600 if lat >= 0 else 32700) + zone
    return zone, epsg, "WGS 84", f"WGS 84 / UTM zone {zone}{hemisphere}"


def _wgs84_to_utm(lat: float, lng: float) -> tuple[float, float, int]:
    """Convert WGS84 lat/lng to ``(easting, northing, epsg)`` with Redfearn TM math.

    CRS rule (see ``_utm_crs_for``): Danish coordinates -> EPSG:25832, else the
    longitude's WGS 84 UTM zone. ETRS89 and WGS84 agree to ~1 m, below the
    precision of a site latitude/longitude.
    """
    zone, epsg, _datum, _desc = _utm_crs_for(lat, lng)
    easting, northing = _tm_utm(lat, lng, zone * 6 - 183)
    if lat < 0:
        northing += 10_000_000  # southern hemisphere false northing
    return easting, northing, epsg


def _add_georeferencing(model, proj: Project) -> None:
    """Add IfcMapConversion + IfcProjectedCRS to place the model in real-world coordinates.

    Maps the local origin (0,0,0) to UTM coordinates derived from the proj's
    WGS84 site_latitude/site_longitude. The CRS follows ``_utm_crs_for``: EPSG:25832
    for Denmark, otherwise the UTM zone of the longitude.
    """
    import math as _math

    lat = proj.site_latitude
    lng = proj.site_longitude
    elev = proj.site_elevation or 0.0
    true_north = proj.site_true_north or 0.0

    easting, northing, epsg = _wgs84_to_utm(lat, lng)

    # Axis orientation: XAxisAbscissa/XAxisOrdinate define the direction of local X
    # relative to CRS grid east. Default (1,0) = local X points East.
    # true_north is degrees from local Y to CRS north; convert to axis rotation.
    x_abscissa = _math.cos(_math.radians(true_north))
    x_ordinate = _math.sin(_math.radians(true_north))

    # Create IfcProjectedCRS — names the coordinate reference system
    zone, _epsg, datum, description = _utm_crs_for(lat, lng)
    crs = model.create_entity(
        "IfcProjectedCRS",
        Name=f"EPSG:{epsg}",
        Description=description,
        GeodeticDatum=datum,
        MapProjection="Transverse Mercator",
        MapZone=str(zone),
    )

    # Create IfcMapConversion — maps local (0,0,0) to CRS coordinates
    # Must reference the top-level geometric representation context (not subcontexts).
    # Top-level contexts are IfcGeometricRepresentationContext (no ParentContext attr);
    # subcontexts are IfcGeometricRepresentationSubContext (has ParentContext).
    source_context = None
    for ctx in model.by_type("IfcGeometricRepresentationContext"):
        if not ctx.is_a("IfcGeometricRepresentationSubContext"):
            source_context = ctx
            break

    if source_context:
        model.create_entity(
            "IfcMapConversion",
            SourceCRS=source_context,
            TargetCRS=crs,
            Eastings=easting,
            Northings=northing,
            OrthogonalHeight=elev,
            XAxisAbscissa=x_abscissa,
            XAxisOrdinate=x_ordinate,
            Scale=1.0,
        )


def _report_csg_depth(proj) -> None:
    """Print the compile's CSG-depth stats line, plus a ``warning:`` line per
    element over the budget.

    Same loud-but-non-fatal channel ``compile_main`` uses for
    ``ValidationReport.warnings``: ``warning:`` on stderr, compile continues.
    The metric itself must never be able to fail a compile — a bug in the
    predictor would otherwise take down a model whose geometry is fine — so it
    is wrapped, and a failure is reported as a warning rather than swallowed.
    """
    try:
        from lite_step.compiler.csg_depth import predict_csg_depth
        report = predict_csg_depth(proj)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"warning: csg depth metric failed ({type(exc).__name__}: {exc}) "
              f"— boolean depth NOT checked this compile", file=sys.stderr)
        return
    print(report.stats_line(), file=sys.stderr)
    for w in report.warnings():
        print(f"warning: {w}", file=sys.stderr)


def _report_carve_pairs(proj) -> None:
    """Report what carved what, beside ``csg depth:``.

    Three settings, and the DEFAULT is the summary:

    ==============================  =====================================
    ``LITESTEP_CARVE_REPORT``       output
    ==============================  =====================================
    unset / anything but the below  ``carves: N pairs`` — one line
    ``all``                         one ``carve: <loser> <- <winner>``
                                    per pair, sorted
    ``0``                           silent (byte-comparison runs)
    ==============================  =====================================

    WS-0 shipped this uncapped, and flagged the default as a decision to
    revisit. Measured on the corpus: 16 of 20 models emit 2 lines or fewer,
    but the four villa-chain models emit **96-265 lines, about 12 KB of
    stderr per compile** — on the models the author builds most, on every recompile,
    straight into the agent's bounded context. That is the httpx failure
    again: volume pushing the signal that matters out of a fixed buffer.

    The visibility §1.9 requires survives the change, because the COUNT is
    always there: a reordering that changes the joinery changes the number, and
    the detail is one env var away. What a summary cannot do is show WHICH
    joint moved when the count happens to stay equal — hence ``=all``, which is
    what a diff of two compiles wants and what a human debugging joinery
    reaches for. Two audiences, and only one of them is paying for context.

    Wrapped like ``_report_csg_depth`` and for the same reason: a bug in a
    REPORT must never take down a model whose geometry is fine.
    """
    setting = os.environ.get("LITESTEP_CARVE_REPORT", "")
    if setting == "0":
        return
    try:
        from lite_step.compiler.displacement import (carve_report_lines,
                                                     inert_authored_report_lines)
        lines = carve_report_lines(proj)
        inert_authored = inert_authored_report_lines(proj)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"warning: carve report failed ({type(exc).__name__}: {exc}) "
              f"— carve pairs NOT listed this compile", file=sys.stderr)
        return
    # BEFORE the carves, and at every verbosity: this is the one line in the
    # report that says something the author asked for did not happen, and it
    # must not sit below a list they may not scroll through.
    for line in inert_authored:
        print(line, file=sys.stderr)
    if setting == "all":
        from lite_step.compiler.displacement import near_miss_report_lines
        for line in lines:
            print(line, file=sys.stderr)
        # After the carves, so the count above still lines up with the list.
        for line in near_miss_report_lines(proj):
            print(line, file=sys.stderr)
        return
    # The UNDECLARED count rides this line rather than becoming a warning per
    # pair. The spec asked for a warning naming both elements of every
    # order-decided overlap; on the hospital that is ~4,000 of them, which is
    # the 12 KB-of-stderr failure described above pointing at a different
    # target. A number on a line nobody has to scroll past is the same signal:
    # it is visible on every compile and it trends to zero as a model migrates
    # to explicit carve=, and LITESTEP_CARVE_REPORT=all still names each pair.
    undeclared = getattr(proj, "_carve_undeclared", 0) or 0
    # Pairs the padded-AABB broad phase proposed and the narrow phase measured
    # as removing nothing. Reported for the same reason ``undeclared`` is: it
    # is the CSG the trigger's over-approximation was manufacturing, and a
    # model whose depth budget is climbing wants to see it.
    inert = getattr(proj, "_carve_inert", 0) or 0
    print(f"carves: {len(lines)} pairs"
          f"{f'  ({undeclared} by tree order alone)' if undeclared else ''}"
          f"{f'  ({inert} near-miss dropped)' if inert else ''}"
          f"{'' if not lines else '  (LITESTEP_CARVE_REPORT=all to list them)'}",
          file=sys.stderr)


def _report_container_tree(proj) -> None:
    """Print the container tree this compile built.

    Three settings, named and behaving exactly like ``LITESTEP_CARVE_REPORT``
    above — see :mod:`lite_step.compiler.tree_report` for what each prints and
    why the default is capped. ``LITESTEP_TREE_REPORT=0`` is the escape a batch
    compile needs, for the same reason.

    Wrapped like ``_report_csg_depth`` and ``_report_carve_pairs``, and for the
    same reason: a bug in a REPORT must never take down a model whose geometry
    is fine. That matters more here than for either sibling — this one reads
    ``Bounds``, which RAISES on an extent it cannot state honestly, so it has a
    live raising path rather than a hypothetical one.
    """
    setting = os.environ.get("LITESTEP_TREE_REPORT", "")
    if setting == "0":
        return
    try:
        from lite_step.compiler.tree_report import tree_report_lines
        lines = tree_report_lines(proj, full=(setting == "all"))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"warning: container tree report failed "
              f"({type(exc).__name__}: {exc}) — the tree is NOT shown this "
              f"compile", file=sys.stderr)
        return
    for line in lines:
        print(line, file=sys.stderr)


def _refuse_unresolved_miters(proj) -> None:
    """Refuse a model still carrying a joint nobody derived.

    ``miter()`` RECORDS; ``compiler.composite_clip`` derives, and it runs
    inside ``normalize_project_to_meters``. So a caller that reaches this
    function without normalising would emit a file with the joint simply
    missing — the model compiles, validates and renders, and the corner is
    just not cut.

    That is not hypothetical: it is exactly how three
    ``test_displacement_clips`` cases failed while this was being built, and
    the only reason it was caught is that they measure volume through the
    kernel. Nothing in the IFC records that a cut was asked for, so no gate
    downstream can find it — which is why the check is here, at the one door
    every backend goes through, rather than in a test.

    Every production caller normalises today (``execute_lite_step_script``
    returns an already-normalized project; the overlay's site-context path
    calls it explicitly), so this costs nothing and closes the door before
    someone adds a path that does not.
    """
    stranded = []

    def walk(elem):
        if getattr(elem, "_pending_miters", None):
            stranded.append(
                getattr(elem, "ifc_name", None)
                or getattr(elem, "name", None)
                or type(elem).__name__
            )
        for attr in ("_elements", "_openings", "_cuts", "_adds",
                     "_intersects", "_fills", "_voids"):
            for child in (getattr(elem, attr, None) or []):
                walk(child)

    for storey in getattr(proj, "storeys", []) or []:
        for elem in getattr(storey, "elements", []) or []:
            walk(elem)
    for site in getattr(proj, "sites", []) or []:
        walk(site)

    if stranded:
        raise ValueError(
            "generate_ifc: "
            + ", ".join(sorted(set(stranded))[:5])
            + (" and others" if len(set(stranded)) > 5 else "")
            + " still carry a miter() joint that was never derived. The angle "
            "is derived by normalize_project_to_meters, so this model reached "
            "the generator without it — call it first (or use compile_main, "
            "which does). Emitting now would drop the corner cut in silence."
        )


def generate_ifc(
    proj: Project,
    source_code: Optional[str] = None,
    quality: str = "preview",
) -> IFCGenerationResult:
    """
    Generate IFC4 file from Project model.

    Creates the full IFC spatial hierarchy:
        IfcProject
          └── IfcSite
                └── IfcBuilding
                      └── IfcBuildingStorey (per storey)
                            └── Elements (IfcWall, IfcMember, etc.)

    Args:
        proj: Validated Project object from Lite-STEP script
        source_code: Optional Lite-STEP source to embed for round-trip support.
                     When provided, embeds LITESTEP_META PropertySet with source,
                     element manifest, version, and hash.
        quality: "preview" (fast) or "download" (IfcOpenShell round-trip for Revit/Solibri)

    Returns:
        IFCGenerationResult with IFC content or error
    """
    # DSL v2.1: ensure canonical names are stamped before either backend
    # reads ``ifc_name``. Idempotent — normalize_project_to_meters already
    # stamps on the compile path; this covers direct generate_ifc callers
    # (tests) that skip normalization.
    from lite_step.compiler.naming import stamp_canonical_names
    stamp_canonical_names(proj)

    _refuse_unresolved_miters(proj)
    # Authoritative frames (meters here) — stamped before displacement so
    # every downstream consumer reads one derivation instead of its own.
    from lite_step.compiler.frames import resolve_child_anchors, stamp_frames
    stamp_frames(proj)
    # Bake parent.anchor(child) into world coordinates while the parent's
    # frame still describes ONLY its own authored geometry. The bake stamps
    # ``_anchor_resolved``, which the displacement pass reads to EXEMPT the
    # child from the inferred carve: ``.add()`` is the carver, ``.anchor()``
    # is a placement relationship (see displacement._collect._walk).
    resolve_child_anchors(proj)
    # The bake moved geometry, so the stamps it was derived from are stale
    # for anything downstream that reads extents (a container's AABB now
    # includes its anchored children). Recompute; the pass is idempotent.
    stamp_frames(proj)

    # Declared space bounds (WS-F): resolve every Space(bounds=[...]), derive
    # its volume and materialize it as the space's body — BEFORE displacement
    # (the injected body is carve="none"-marked air, but its boolean operands
    # must exist where the consumed-operand walk and the CSG-depth report
    # look). Idempotent, like apply_displacement.
    from lite_step.compiler.space_volume import (
        SpaceVolumeError,
        materialize_bound_spaces,
    )
    try:
        materialize_bound_spaces(proj)
    except SpaceVolumeError as e:
        return IFCGenerationResult(success=False, error=str(e))

    # Universal geometric displacement: any solid ENCLOSED within another
    # solid/mesh carves its host (occupant still renders). Runs once, before
    # either backend serializes, so both emit the already-carved result.
    # Idempotent via Project._displacement_applied. A carve that can't be
    # realised (untessellable/placed operand, manifold failure) raises
    # DisplacementError → a LOUD failed result (never a silent uncarved mesh).
    from lite_step.compiler.displacement import (
        CarveAssertionError,
        apply_displacement,
        check_carve_assertions,
        DisplacementError,
    )
    try:
        apply_displacement(proj)
    except DisplacementError as e:
        return IFCGenerationResult(success=False, error=str(e))
    # Who carved whom, on every compile. Displacement direction is authoring
    # order, and this line is the only place a diff or a reviewer can see it.
    _report_carve_pairs(proj)
    # …and the assertions that PIN it. Fatal, like a DisplacementError: a
    # reordering that silently changes the joinery is the failure this exists
    # to stop being silent.
    try:
        check_carve_assertions(proj)
    except CarveAssertionError as e:
        return IFCGenerationResult(success=False, error=str(e))

    # CSG depth budget. Runs HERE — after displacement, before either backend —
    # because most boolean depth comes from inferred carves, and because the
    # prediction is then true for the emitter. Always prints one stats line;
    # over-budget elements additionally print `warning:` lines (loud, non-fatal:
    # a deep model must still compile and download). See
    # lite_step/compiler/csg_depth.py for what depth costs.
    _report_csg_depth(proj)

    # …and the SHAPE both of those describe. Beside its siblings, and for the
    # same reason they are here: this is after the anchor bake and after
    # displacement, so the tree reported is the one the emitter serializes —
    # an anchored child stands in world coordinates and carries
    # ``_anchor_resolved``, which is what makes the ``@`` marker true.
    #
    # Deliberately BEFORE the backend dispatch rather than on the success
    # path: a compile that dies inside an emitter is exactly when an author
    # wants to see what the compiler thought it was building.
    _report_container_tree(proj)

    if not IFC_AVAILABLE:
        return IFCGenerationResult(
            success=False,
            error="ifcopenshell library not available. Install with: uv pip install ifcopenshell"
        )

    # Assert imports are available after IFC_AVAILABLE check (for type checker)
    assert ifcopenshell is not None
    assert run is not None
    assert np is not None

    try:
        # Create the output model on the pinned schema (IFC4X3_ADD2 —
        # see lite_step/ifc/schema_version.py; 4.3 semantic classes emit natively)
        model = ifcopenshell.file(schema=IFC_OUTPUT_SCHEMA)
        
        # Create entity cache for deduplication
        cache = EntityCache(model)

        # Create project first (required before adding context)
        project = _create_ifc_entity(model, "IfcProject", name=proj.name)

        # Assign SI units with METRE as length unit (no prefix = meters)
        run("unit.assign_unit", model, length={"is_metric": True, "raw": "METRE"})
        # assign_unit hands its members back through a Python set keyed on
        # the model's heap address, so canonicalise before anything reads or
        # serialises them. See _canonical_unit_order.
        _canonical_unit_order(model)

        # Create geometric representation context hierarchy
        # First the parent Model context, then subcontexts for different views
        model_context = run("context.add_context", model, context_type="Model")

        # Body context for semantic types (wall, floor, roof, etc.)
        body_context = run("context.add_context", model,
                          context_type="Model",
                          context_identifier="Body",
                          target_view="MODEL_VIEW",
                          parent=model_context)

        # Sketch context for ideation elements (type="sketch")
        sketch_context = run("context.add_context", model,
                            context_type="Model",
                            context_identifier="Sketch",
                            target_view="SKETCH_VIEW",
                            parent=model_context)

        # IfcSite Name comes from the DSL Site's canonical name when authored
        # (``Site(name="site")`` -> ``site:site``); else the default.
        site_name = proj.sites[0]._canonical_name if proj.sites else "Site"
        site = _create_ifc_entity(model, "IfcSite", name=site_name)
        _set_identity_placement(model, site)

        # Link spatial hierarchy: Project -> Site is ALWAYS present.
        _assign_aggregate(model, project, [site])

        # P1: IfcBuilding (+ the Site -> Project aggregate) is emitted ONLY when
        # there is building content — a site-only model (terrain, no building
        # elements) carries NO IfcBuilding and NO IfcBuildingStorey.
        has_building_content = any(storey.elements for storey in proj.storeys)
        ifc_building = None
        if has_building_content:
            ifc_building = _create_ifc_entity(model, "IfcBuilding", name=proj.name)
            _set_identity_placement(model, ifc_building)
            ifc_building.ObjectPlacement.PlacementRelTo = site.ObjectPlacement
            _assign_aggregate(model, site, [ifc_building])

        # Georeferencing: map local (0,0,0) to real-world CRS
        if proj.site_latitude is not None and proj.site_longitude is not None:
            _add_georeferencing(model, proj)

        # Statistics tracking
        stats = {
            "storeys": 0,
            "walls": 0,
            "openings": 0,
            "doors": 0,
            "windows": 0,
            "boxes": 0,
            "extrudes": 0,
            "sweeps": 0,
            # Semantic element classes
            "columns": 0,
            "beams": 0,
            "slabs": 0,
            "roofs": 0,
            "total_elements": 0
        }

        # Boolean-operand consumption (v1.5 WS1 PR-E): an element consumed
        # by .cuts()/.adds()/.fills() is merged into its consumer's shape.
        # If the same object was ALSO added to the proj, skip its
        # standalone render (loudly) — it must not appear twice.
        consumed_operands = collect_consumed_operand_ids(proj)

        # Placement engine (v1.5 WS1 PR-F): resolve Transform/Anchor world
        # matrices up front (identity-keyed on this proj's elements);
        # each is composed onto its product after creation below. An
        # unresolvable placement (cycle, dangling host, nested placement on
        # an unvalidated path) raises PlacementResolutionError here — caught
        # by this function's except into a LOUD failed result, never a
        # silently unplaced element.
        from lite_step.compiler.placement import resolve_placement_matrices
        placement_matrices = resolve_placement_matrices(proj)

        # (ifc_storey, label) for the 2D drawing pass at the end.
        drawing_storeys: list = []

        # Process each storey. P1: only storeys that actually have elements get
        # an IfcBuildingStorey; a site-only model has none (and no building).
        for storey_index, storey_data in enumerate(proj.storeys):
            if not storey_data.elements:
                continue
            # Create storey entity
            # DSL v2.1: IfcBuildingStorey Name is the storey's canonical name
            # (``storey:<leaf>`` when named, ``None`` when anonymous — the
            # single auto-created storey).
            #
            #: Elevation is written too. It is OPTIONAL in the schema, so
                # nothing validates it and nothing renders wrong when it is
                # missing — the data is simply dropped. All four
            # attribute values (Name / LongName / CompositionType / Elevation)
            # are decided in ``lite_step.ifc.storeys`` so the emitter
            # cannot disagree about them.
            storey_attrs = storeys.attributes_of(storey_data)
            ifc_storey = _create_ifc_entity(
                model, "IfcBuildingStorey", name=storey_attrs.name)
            ifc_storey.LongName = storey_attrs.long_name
            ifc_storey.CompositionType = storey_attrs.composition_type
            ifc_storey.Elevation = storey_attrs.elevation

            # Aggregate storey to building
            _assign_aggregate(model, ifc_building, [ifc_storey])

            # Set storey elevation using transformation matrix. The SAME float
            # that went into ``Elevation`` above — one derivation, so a file
            # whose stated elevation contradicts where its storey stands is
            # impossible.
            matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, storey_attrs.elevation],
                [0, 0, 0, 1]
            ], dtype=np.float64)
            _set_placement(model, ifc_storey, matrix, cache)
            ifc_storey.ObjectPlacement.PlacementRelTo = ifc_building.ObjectPlacement

            stats["storeys"] += 1
            drawing_storeys.append(
                (ifc_storey, storey_data._canonical_name or f"L{stats['storeys']}"))

            # Process elements in this storey
            products = []
            site_products = []  # Site terrain → IfcGeographicElement(TERRAIN) under IfcSite
            site_context_products = []  # Non-terrain site context (existing buildings, trees) → under IfcSite, unreassigned

            # One IfcGrid per storey, holding ALL its axes. Resolved
            # up front — every refusal in ``grids.collect_grid`` is about the
            # SET of axes, so it must be raised before any of them is emitted,
            # not discovered halfway through the loop. Emitted in place of the
            # FIRST grid line so document order still follows authoring order.
            storey_grid = grids.collect_grid(
                storey_data.elements,
                grids.container_label("storey", storey_data._canonical_name,
                                      storey_index))
            storey_grid_emitted = False

            for elem in storey_data.elements:
                if id(elem) in consumed_operands:
                    logger.warning(
                        "skipping standalone render of %s %r — it is "
                        "consumed as a boolean operand "
                        "(.cuts()/.adds()/.fills()) elsewhere in the "
                        "proj",
                        type(elem).__name__, elem.ifc_name,
                    )
                    continue

                # Reset per iteration so the placement hook below never
                # sees a stale product from a previous element.
                ifc_elem = None

                if isinstance(elem, Wall):
                    # Wall assemblies create IfcWall (semantic), use body_context
                    ifc_elem = _create_wall(model, body_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats["walls"] += 1
                        # Count Window/Door children
                        for child in elem._elements:
                            if isinstance(child, (Window, Door)):
                                stats["openings"] += 1
                                if isinstance(child, Door):
                                    stats["doors"] += 1
                                elif isinstance(child, Window):
                                    stats["windows"] += 1

                elif isinstance(elem, Site):
                    # DSL Site containers are routed to ``proj.sites`` and
                    # handled in the proj-level site pass below (their
                    # children land under IfcSite, not a storey). A Site should
                    # never appear in a storey now — skip defensively.
                    continue

                # --- New primitives ---
                elif tx.is_prism(elem):
                    # Route to sketch context for type="sketch", body_context otherwise
                    elem_type = getattr(elem, 'type', '')
                    ctx = sketch_context if elem_type == 'sketch' else body_context
                    ifc_elem = _create_solid(model, ctx, elem, cache)
                    if ifc_elem:
                        # Site solids go under IfcSite for proper BIM filtering
                        if elem_type == 'site':
                            site_products.append(ifc_elem)
                        else:
                            products.append(ifc_elem)
                        stats["boxes" if isinstance(elem, Box) else "extrudes"] += 1

                elif isinstance(elem, Sweep):
                    # Sweeps can also use sketch context
                    elem_type = getattr(elem, 'type', '')
                    ctx = sketch_context if elem_type == 'sketch' else body_context
                    ifc_elem = _create_sweep(model, ctx, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats["sweeps"] += 1

                # --- Semantic element classes (architecture sketching) ---
                elif isinstance(elem, Column):
                    ifc_elem = _create_column(model, sketch_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats["columns"] += 1

                elif isinstance(elem, Beam):
                    ifc_elem = _create_beam(model, sketch_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats["beams"] += 1

                elif isinstance(elem, Slab):
                    ifc_elem = _create_slab(model, sketch_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats["slabs"] += 1

                elif isinstance(elem, Roof):
                    ifc_elem = _create_roof(model, sketch_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats["roofs"] += 1

                elif isinstance(elem, ReferencePoint):
                    if elem.point_type in ("boundary", "foundation"):
                        ifc_elem = _create_marker_peg(model, body_context, elem, cache)
                    else:
                        ifc_elem = _create_reference_point(model, body_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats.setdefault("reference_points", 0)
                        stats["reference_points"] += 1

                elif isinstance(elem, GuideLine):
                    if grids.is_grid_line(elem):
                        # All of this storey's grid lines are ONE IfcGrid,
                        # minted at the first of them. The rest
                        # contribute an axis to that entity and no product of
                        # their own — but they are still AUTHORED elements, so
                        # every one of them counts in the statistics.
                        if not storey_grid_emitted:
                            ifc_elem = _create_grid(
                                model, body_context, storey_grid, cache)
                            storey_grid_emitted = True
                            if ifc_elem:
                                products.append(ifc_elem)
                        # Never route the merged grid through the placement
                        # hook below: it stands for N elements, so ``elem``'s
                        # matrix is not its matrix. ``collect_grid`` refuses a
                        # grid line carrying placement= outright, so there is
                        # no matrix to lose here.
                        ifc_elem = None
                        stats.setdefault("guide_lines", 0)
                        stats["guide_lines"] += 1
                    else:
                        ifc_elem = _create_guide_line(model, body_context, elem, cache)
                        if ifc_elem:
                            products.append(ifc_elem)
                            stats.setdefault("guide_lines", 0)
                            stats["guide_lines"] += 1

                elif isinstance(elem, Space):
                    ifc_elem = _create_space(model, sketch_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats.setdefault("spaces", 0)
                        stats["spaces"] += 1

                elif isinstance(elem, Mesh):
                    # A bare mesh is pure geometry → a storey product (proxy).
                    # Terrain routing by mesh_type is GONE — terrain is the
                    # semantic wrapper under a Site.
                    ifc_elem = _create_mesh(model, body_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats.setdefault("meshes", 0)
                        stats["meshes"] += 1

                # --- DSL v1.5 primitives (WS1 PR-E) ---
                elif isinstance(elem, Pipe):
                    ifc_elem = _create_pipe(model, body_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats.setdefault("pipes", 0)
                        stats["pipes"] += 1

                elif isinstance(elem, Revolve):
                    ifc_elem = _create_revolve(model, body_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats.setdefault("revolves", 0)
                        stats["revolves"] += 1

                elif isinstance(elem, Bar):
                    ifc_elem = _create_bar(model, body_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats.setdefault("bars", 0)
                        stats["bars"] += 1

                elif isinstance(elem, Element):
                    ifc_elem = _create_element_generic(model, body_context, elem, cache)
                    if ifc_elem:
                        products.append(ifc_elem)
                        stats.setdefault("generic_elements", 0)
                        stats["generic_elements"] += 1

                # Placement engine (v1.5 WS1 PR-F): compose the resolved
                # world matrix ON TOP of the product's own placement —
                # rigid-body semantics for the whole product family
                # (aggregated children, opening voids, fills).
                if ifc_elem is not None and id(elem) in placement_matrices:
                    _apply_dsl_placement(
                        model, ifc_elem, placement_matrices[id(elem)], cache
                    )

            # Assign elements to storey — products CONTAINED, an IfcSpace (or
            # any other emitted IfcSpatialElement) AGGREGATED. See
            # :func:`_assign_spatial_children`;.
            _assign_spatial_children(model, ifc_storey, products)
            # OJP001: set PlacementRelTo on the storey's children, contained
            # and aggregated alike (adjust Z for storey elevation). The
            # relation changed ; the placement chain did not, and a
            # space that stopped following its storey's elevation would move.
            _set_relative_placements(products, ifc_storey.ObjectPlacement, storey_attrs.elevation)

            # Reassign site elements from IfcBuildingElementProxy to IfcGeographicElement(TERRAIN)
            # This is centralized here so creation functions remain pure primitive creators.
            # IMPORTANT: reassign_class returns a NEW entity reference — the old one becomes
            # a dangling pointer that segfaults on access. Must capture the return value.
            reassigned_site_products = []
            for site_elem in site_products:
                new_elem = run("root.reassign_class", model,
                    product=site_elem,
                    ifc_class="IfcGeographicElement",
                    predefined_type="TERRAIN")
                reassigned_site_products.append(new_elem)

            # Assign site elements to IfcSite (not storey) for BIM category filtering
            _assign_container(model, site, reassigned_site_products)
            # OJP001: set PlacementRelTo on site-contained elements
            _set_relative_placements(reassigned_site_products, site.ObjectPlacement)

            # Non-terrain site context (existing buildings, trees) also lives under
            # IfcSite, but is NOT reassigned to IfcGeographicElement — it is not terrain.
            _assign_spatial_children(model, site, site_context_products)
            _set_relative_placements(site_context_products, site.ObjectPlacement)

        # Project-level site pass (P2): DSL Site containers live in
        # proj.sites (routed out of storeys). Every child lands under
        # IfcSite — Box/Extrude and terrain Mesh become
        # IfcGeographicElement(TERRAIN); non-terrain context meshes (existing
        # buildings, trees) stay proxies under IfcSite. Runs once, even with
        # zero storeys — LOD-100 site-context (terrain + OSM context) is
        # site-only and flows through here.
        building_site_products = []
        building_site_context_products = []
        for site_container in proj.sites:
            for child in site_container._elements:
                if tx.is_prism(child):
                    # Site children use body_context (not sketch)
                    ifc_elem = _create_solid(model, body_context, child, cache)
                    if ifc_elem:
                        building_site_products.append(ifc_elem)
                        stats["boxes" if isinstance(child, Box) else "extrudes"] += 1
                    else:
                        _warn_child_dropped("Site", site_container.ifc_name, child)
                elif isinstance(child, Mesh):
                    # A bare Site-child mesh is site CONTEXT (existing
                    # buildings, scans) — a proxy under IfcSite. Terrain is
                    # the semantic wrapper (Element branch below); mesh_type
                    # routing is GONE.
                    ifc_elem = _create_mesh(model, body_context, child, cache)
                    if ifc_elem:
                        building_site_context_products.append(ifc_elem)
                        stats.setdefault("meshes", 0)
                        stats["meshes"] += 1
                    else:
                        _warn_child_dropped("Site", site_container.ifc_name, child)
                elif isinstance(child, Element):
                    # Semantic site feature (geometry ⊥ semantics ⊥ relations):
                    # the wrapper says what it IS (IfcGeographicElement TERRAIN,
                    # IfcEarthworksFill, …), the Site says it belongs under
                    # IfcSite. The creator emits the class natively (IFC4X3
                    # schema) — no proxy + reassign dance, so it goes in the
                    # context list which is assigned to IfcSite verbatim.
                    ifc_elem = _create_element_generic(
                        model, body_context, child, cache)
                    if ifc_elem:
                        building_site_context_products.append(ifc_elem)
                        stats.setdefault("generic_elements", 0)
                        stats["generic_elements"] += 1
                    else:
                        _warn_child_dropped("Site", site_container.ifc_name, child)
                elif type(child).__name__ == "SpatialElement":
                    # A FACILITY (IfcBridge/Road/Railway/...) — a spatial node,
                    # not a product. It joins the tree with IfcRelAggregates,
                    # never with IfcRelContainedInSpatialStructure: WR31 on
                    # that relation forbids a spatial element in its
                    # RelatedElements, and ifcopenshell.validate does NOT
                    # report WHERE rules (measured), so the wrong relation
                    # would pass the conformance gate and simply vanish from
                    # getSpatialStructure(). See lite_step/ifc/facilities.py.
                    _emit_facility(model, body_context, site, child, cache, stats)

            # Everything the four branches above did not take. Until then
            # this was an `else: _warn_child_dropped` — the ONE container in
            # the compiler with no generic fallback, so a Wall/Column/Beam/
            # Slab/Roof/Space/Sweep/Revolve/Pipe/Bar under a Site emitted
            # nothing at all. See :func:`_emit_offtable_children`.
            offtable = _emit_offtable_children(
                model, body_context, site_container, site_container._elements,
                cache, kind="Site",
                handled=lambda c: (tx.is_prism(c)
                                   or isinstance(c, (Mesh, Element))
                                   or type(c).__name__ == "SpatialElement"))
            # A Site is an IfcSpatialStructureElement, so a PRODUCT joins the
            # tree with IfcRelContainedInSpatialStructure — the same relation
            # the four branches above use — and aggregating one instead would
            # be the shape: valid at 0 errors, absent from
            # getSpatialStructure(). An emitted IfcSpatialElement is the
            # opposite case and takes the opposite relation (WR31);
            # :func:`_assign_spatial_children` below decides which. No terrain
            # reassign either: only a bare prism is terrain.
            building_site_context_products.extend(offtable)
            _count_site_offtable_children(site_container, cache, stats)

        # Reassign terrain products to IfcGeographicElement(TERRAIN) under IfcSite.
        # reassign_class returns a NEW reference — the old one dangles.
        reassigned_building_site_products = []
        for site_elem in building_site_products:
            new_elem = run("root.reassign_class", model,
                product=site_elem,
                ifc_class="IfcGeographicElement",
                predefined_type="TERRAIN")
            reassigned_building_site_products.append(new_elem)
        _assign_container(model, site, reassigned_building_site_products)
        _set_relative_placements(reassigned_building_site_products, site.ObjectPlacement)
        _assign_spatial_children(model, site, building_site_context_products)
        _set_relative_placements(building_site_context_products, site.ObjectPlacement)

        # total_elements = every AUTHORED element counted exactly once:
        # top-level products + Window/Door opening children (as doors/windows).
        # "openings" stays as the informational void count (== doors+windows)
        # but is EXCLUDED from the sum — including both double-counted every
        # window and door.
        stats["total_elements"] = (stats["walls"] + stats["boxes"] +
                                   stats["extrudes"] + stats["sweeps"] +
                                   stats["doors"] + stats["windows"] +
                                   stats["columns"] + stats["beams"] +
                                   stats["slabs"] + stats["roofs"] +
                                   stats.get("spaces", 0) + stats.get("meshes", 0) +
                                   stats.get("reference_points", 0) +
                                   stats.get("guide_lines", 0) +
                                   stats.get("pipes", 0) + stats.get("revolves", 0) +
                                   stats.get("bars", 0) + stats.get("generic_elements", 0))

        # Element-level props= → IfcPropertySet (v1.5 WS1 PR-D). Post-pass
        # after all products exist; no-op for buildings without props.
        _attach_element_psets(model, proj, cache)

        # Product typing (WS-B step 1). BEFORE the manifest build, and that is
        # safe here for a schema reason rather than an ordering one: an
        # Ifc*Type is an IfcTypeProduct, which is NOT an IfcProduct, so
        # ``model.by_type("IfcProduct")`` below cannot see it. Contrast the
        # IfcAnnotation drawings below, which ARE IfcProducts and therefore
        # have to be emitted after. Pinned by test_product_typing.py.
        _emit_product_types(model, proj, cache)

        # Embed round-trip metadata if source code provided
        if source_code:
            from lite_step.ifc.embedder import embed_litestep_meta
            # Build manifest from all named products: {element_id: ifc_global_id}
            manifest = {}
            for product in model.by_type("IfcProduct"):
                if product.Name:
                    manifest[product.Name] = product.GlobalId
            # #100031: re-home LITESTEP_META to the IfcSite (always present) —
            # ifc_building is None for a site-only model.
            embed_litestep_meta(model, site, source_code, manifest)

        # Aggregates + zones (WS-D / WS-C). Deliberately AFTER the manifest
        # build, for the reason the drawings below are: an IfcRelAggregates
        # parent IS an IfcProduct, so a named facade would otherwise land in
        # LITESTEP_META and pollute patch identity with an entity that
        # corresponds to no DSL element — the v21.2 IfcAnnotation bug. (An
        # IfcZone is an IfcGroup and would be excluded by schema anyway; it
        # rides here so one call site owns both.) Also after every placement
        # has been applied, so the member re-chaining below cannot be undone
        # by _apply_dsl_placement's family walk.
        _emit_groupings(model, proj, cache)

        # Inferred space boundaries (WS-C). AFTER the groupings for
        # one reason: an aggregate parent is a product with a canonical name,
        # and a boundary joins products BY name — so running first would relate
        # a space to a facade that does not exist yet, or to a member the
        # re-chain is about to move. It is also after the manifest build, like
        # everything else here: ``IfcRelSpaceBoundary1stLevel`` carries
        # ``Name = "1stLevel"`` on EVERY instance, and the manifest predicate is
        # purely structural (22-char GlobalId at 0, Name at 2), so
        # one shared literal name would otherwise collapse every boundary into
        # a single bogus manifest key. Not a risk on this backend (a relation
        # is not an IfcProduct) — kept in lockstep anyway, and pinned on both.
        _emit_space_boundaries(model, proj)

        # 2D drawing wiring (v21.2). Deliberately AFTER the manifest build:
        # IfcAnnotation is an IfcProduct, so a named drawing would otherwise
        # land in the manifest and pollute patch identity with an entity that
        # corresponds to no DSL element.
        if _drawings_enabled():
            from lite_step.ifc.drawings import emit_drawings
            emit_drawings(model, project, drawing_storeys,
                          body_context=body_context, proj=proj)

        # Material associations accumulated their members in Python (see
        # EntityCache.associate_material — appending them entity-side is
        # O(N^2)). Write them out here: after every emitter that can add one,
        # and before anything reads the model back.
        cache.flush_material_associations()

        # Serialize to string
        ifc_content = _model_to_string(model)

        # Post-process: deduplicate entities created by ifcopenshell internals,
        # and drop the Product geometry purge's condemned ids in the same pass
        # (#789 — they were recorded rather than removed, see
        # `_share_product_geometry`).
        from lite_step.ifc.deduplicator import deduplicate_ifc_step
        ifc_content, dedup_stats = deduplicate_ifc_step(
            ifc_content, drop_ids=cache.purged_ids)
        stats["dedup"] = dedup_stats

        return IFCGenerationResult(
            success=True,
            ifc_content=ifc_content,
            stats=stats
        )

    except Exception as e:
        import traceback
        return IFCGenerationResult(
            success=False,
            error=f"IFC generation failed: {e}\n{traceback.format_exc()}"
        )


def _create_wall(model, context, wall: Wall, cache: EntityCache):
    """
    Create IfcWall from Wall assembly.

    Finds body Solid in wall._elements and creates IfcWall from it: box
    geometry for a box-mode body (start=/end=), or a planar vertical
    contour extrusion for a contour-mode body (gable-end walls — see
    _create_wall_contour). Then processes Window/Door children as IFC
    openings with child elements (glass panels, frames) properly
    aggregated to their parent windows/doors.

    Body geometry uses the same approach as _create_solid_box (box mode)
    / _create_solid_contour (contour mode) but creates IfcWall instead of
    IfcBuildingElementProxy.
    """
    assert run is not None
    assert np is not None

    # Find body primitive in wall._elements. Only the FIRST unanchored
    # Box/Extrude is the wall body — validate_project_report ERRORs on >1 in a
    # Wall, and reads the SAME ``tx.body_prisms`` for that count, so the two
    # cannot disagree about which child is the body. They did: this loop took
    # the first prism whatever it was, so an ``.anchor()``ed detail was emitted
    # as the wall's own geometry and its canonical name never reached the file
    #.
    body_solids = tx.body_prisms(wall)
    body = body_solids[0] if body_solids else None

    if body is None:
        # No prism body — but the wall is NOT skipped. Since §1.2 keyed the
        # child table on the IFC class, a Wall may legitimately be built from
        # a Sweep or a Revolve (a curved wall), exactly as
        # ``Element(ifc_class="IfcWall")`` always could. The prism branches
        # below cannot render that, so the wall is emitted as an identity +
        # placement carrying its children as aggregated products — the same
        # shape ``_aggregate_wall_details`` already uses for a Wall's
        # non-body children, and the same result the escape hatch produces.
        #
        # Returning `None` behind a `logger.warning` would make the whole
        # wall — identity, Psets, geometry — vanish from the file with only
        # a stderr line. Making the table symmetric without this would have
        # moved §1.2's asymmetry one layer down instead of removing it: the
        # sugar accepted the child and dropped the product, the escape hatch
        # emitted it.
        #
        # Since this is also the AGGREGATOR path: a Wall that emits no
        # solid of its own and derives its frame from the ``.add()``ed leaves
        # it aggregates. That is the same emission — a body-less
        # ``RelatingObject`` over placed members — so it needed no new branch,
        # only the validator to stop demanding a body.
        #
        # Its HOLES are the one thing that could not stay on the bodied path
        #. The ``IfcWall`` below carries ``Representation=None``, so an
        # ``IfcRelVoidsElement`` on it carves nothing in a renderer that builds
        # each element's mesh from its own representation — the exact reason
        # ``_process_container_voids`` exists — and until then the author had
        # to add an envelope solid spanning the whole buildup purely to give
        # the hole somewhere to land. That envelope is not inert: measured, it
        # reports 2.12 m3 of PHANTOM wall volume when the assembly is
        # under-tiled, and tessellates to zero vertices behind 29 booleans when
        # it is tiled correctly. So this wall's holes are DISTRIBUTED, through
        # the same two functions a Column's go through — the routing decision
        # is ``voids.void_reaches_children``'s, which answers per instance here
        # and reads ``tx.body_prisms`` to do it.
        #
        # Both calls run AFTER the aggregation, because the distribution walks
        # the ``(dsl child, ifc product)`` pairs ``_emit_offtable_children``
        # records as it emits — same order as ``_create_column``.
        ifc_wall = _create_ifc_entity(model, "IfcWall", name=wall.ifc_name)
        _set_placement(model, ifc_wall, np.eye(4), cache)
        _aggregate_wall_details(model, context, ifc_wall, wall, None, cache)
        _process_container_voids(model, context, wall, ifc_wall, cache)
        _process_container_openings(model, context, wall, ifc_wall, cache)
        return ifc_wall

    if isinstance(body, Extrude):
        # Contour-mode body: planar VERTICAL polygon extruded by thickness
        # (gable-end / sloped-top walls). validate_project_report rejects
        # tilted/non-planar contours at compile time.
        return _create_wall_contour(model, context, wall, body, cache)

    if body.start is None or body.end is None:
        # Defensive: a Box always carries start/end, so this fires only for
        # hand-built _elements lists.
        logger.warning(
            "Wall %r: body Box has neither box (start=/end=) nor contour "
            "geometry — wall skipped entirely",
            wall.ifc_name,
        )
        return None

    # Transform DSL coordinates to IFC coordinates (both Z-up, identity)
    start_ifc = _dsl_to_ifc_point(body.start.x, body.start.y, body.start.z)
    end_ifc = _dsl_to_ifc_point(body.end.x, body.end.y, body.end.z)

    # Normalize corners
    min_pt = (min(start_ifc[0], end_ifc[0]), min(start_ifc[1], end_ifc[1]),
              min(start_ifc[2], end_ifc[2]))
    max_pt = (max(start_ifc[0], end_ifc[0]), max(start_ifc[1], end_ifc[1]),
              max(start_ifc[2], end_ifc[2]))

    width = snap_to_precision(max_pt[0] - min_pt[0])   # X dimension in IFC
    depth = snap_to_precision(max_pt[1] - min_pt[1])   # Y dimension in IFC (forward)
    height = snap_to_precision(max_pt[2] - min_pt[2])  # Z dimension in IFC (up)

    if width < 0.001 or height < 0.001:
        return None

    center = ((min_pt[0]+max_pt[0])/2.0, (min_pt[1]+max_pt[1])/2.0,
              (min_pt[2]+max_pt[2])/2.0)

    # Shared by the base solid AND the per-layer slices below.
    z_offset = -height / 2.0
    extrusion_dir = cache.get_z_up_direction()

    def _make_base_solid():
        """The whole-body extrusion, centred at origin.

        Built LAZILY because a layered wall never uses it — that branch emits
        one sliced solid per material layer instead. Creating it eagerly left
        one unreferenced IfcExtrudedAreaSolid per layered wall in the file
        (4 in the villa, one per facade leaf): invisible to every product walk,
        but a consumer that iterates geometry entities renders it as an
        UNCLIPPED box, whose square end-caps read as stray lines cutting across
        the layers at mitered corners. See the reachability gate in
        tests/test_no_orphan_geometry.py.
        """
        profile = cache.get_or_create_rect_profile(x_dim=width, y_dim=depth)
        origin = cache.get_or_create_point([0.0, 0.0, z_offset])
        placement = model.create_entity("IfcAxis2Placement3D", Location=origin)
        return model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile, Position=placement,
            ExtrudedDirection=extrusion_dir, Depth=height
        )

    # Place at center
    matrix = np.array([
        [1, 0, 0, center[0]],
        [0, 1, 0, center[1]],
        [0, 0, 1, center[2]],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    ifc_wall = _create_ifc_entity(model, "IfcWall", name=wall.ifc_name)
    _set_placement(model, ifc_wall, matrix, cache)

    # Body booleans (v1.5 WS1 PR-E): the wall body Solid's .cuts()/.adds()
    # apply in the wall's local (centered) frame — same semantics as a
    # standalone box Solid. This is the spec's arched-doorway path:
    # door_void.adds(arch_turn); wall_body.cuts(door_void).
    cuts = body._cuts if body._cuts else []
    adds = body._adds if body._adds else []
    intersects = body._intersects if getattr(body, "_intersects", None) else []

    def _apply_boolean_chain(solid):
        """The body's boolean/clip chain, applied to one base solid — shared
        between the single-body path and the per-layer slices (a layered
        wall's carves apply to every slice; a non-intersecting operand is a
        harmless no-op, per displacement doctrine).

        Balanced within each class (``lite_step.ifc.boolean_tree``): a 13-void
        wall nests 5 deep instead of 13, which is what keeps the viewer from
        silently falling back to the un-carved block."""
        result = _apply_boolean_chain_ios(
            model, solid,
            build=lambda e, is_cut=True: _create_hole_solid_local(model, e, min_pt, center, cache, is_cut=is_cut),
            cuts=cuts, adds=adds, intersects=intersects)
        return _apply_clips(model, result, body._clips, center=center)

    rep_type = "CSG" if (cuts or adds or intersects or body._clips) else "SweptSolid"

    if wall.layers is not None:
        # Layered wall: one IfcWall whose Body carries one solid per non-cavity
        # layer, sliced along the through axis. The buildup list is outer→inner;
        # LayerSet.outward (inner→outer normal) places it — layer_axis_and_dir
        # resolves the axis + which face the first-listed (outer) layer sits on.
        # Cavity layers leave a literal air gap. Each item is styled from its
        # layer material's registry render (explicit body color wins for ALL
        # slices); the material data rides the IfcMaterialLayerSetUsage chain.
        from lite_step.compiler.layers import (
            layer_spans, is_cavity, layer_axis_and_dir)
        axis, outer_at_max = layer_axis_and_dir(wall, body)
        axis_i = "xyz".index(axis)  # IFC X or Y = through-thickness
        items = []
        for mat, s_lo, s_hi in layer_spans(
                wall.layers, min_pt[axis_i], max_pt[axis_i], reverse=outer_at_max):
            if is_cavity(mat):
                continue
            t = snap_to_precision(s_hi - s_lo)
            mid_local = (s_lo + s_hi) / 2.0 - center[axis_i]
            # Per-wall profile name + UNIQUE DIMS: a mirror pair (west/east
            # facade leaf) has identical slice dims → identical vertex data →
            # the SPA's multi-threaded fragments render worker groups them into
            # one instanced mesh, and a race in that worker sometimes drops an
            # instance (the missing-cladding bug; nondeterministic — usually
            # one wall of the two mirror pairs, not one per pair). The name
            # split alone (v18.0.4) was NOT enough — the worker keys on
            # geometry content, not ProfileName — so unique_dims adds a
            # deterministic sub-tolerance (10 nm) nudge that makes each wall's
            # slice dims byte-distinct: no shared geometry, nothing to drop.
            wall_salt = wall.ifc_name or f"wall-{id(wall)}"
            if axis_i == 0:
                # XDim is the layer THICKNESS on this axis, so the salt has to
                # land in YDim (the wall length) instead — see
                # EntityCache._claim_unique_dims. Salting the thickness made a
                # 43 mm cavity measure 42.98 mm and left each slice 0.01 mm
                # proud of the miter plane it was clipped to.
                profile_i = cache.get_or_create_rect_profile(
                    x_dim=t, y_dim=depth, name=wall_salt, unique_dims=True,
                    salt="y")
                origin_i = cache.get_or_create_point([snap_to_precision(mid_local), 0.0, z_offset])
            else:
                # XDim is the wall LENGTH here, which is the safe default.
                profile_i = cache.get_or_create_rect_profile(
                    x_dim=width, y_dim=t, name=wall_salt, unique_dims=True,
                    salt="x")
                origin_i = cache.get_or_create_point([0.0, snap_to_precision(mid_local), z_offset])
            placement_i = model.create_entity("IfcAxis2Placement3D", Location=origin_i)
            solid_i = model.create_entity(
                "IfcExtrudedAreaSolid",
                SweptArea=profile_i, Position=placement_i,
                ExtrudedDirection=extrusion_dir, Depth=height)
            item = _apply_boolean_chain(solid_i)
            _style_layer_item(model, item, mat, body, cache)
            items.append(item)
        shape_rep = model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=context, RepresentationIdentifier="Body",
            RepresentationType=rep_type, Items=items)
        ifc_wall.Representation = model.create_entity(
            "IfcProductDefinitionShape", Representations=[shape_rep])
        cache.associate_layer_usage(
            ifc_wall, wall.layers, _layer_axis_from_slicing(axis),
            _layer_offset_from_reference_line(
                min_pt[axis_i], max_pt[axis_i], center[axis_i], outer_at_max),
            _layer_thickness_scale(wall.layers, min_pt[axis_i], max_pt[axis_i]),
            outer_at_max=outer_at_max)
        result_solid = items[0] if items else None
    else:
        result_solid = _apply_boolean_chain(_make_base_solid())
        shape_rep = model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=context, RepresentationIdentifier="Body",
            RepresentationType=rep_type, Items=[result_solid]
        )
        product_shape = model.create_entity(
            "IfcProductDefinitionShape", Representations=[shape_rep]
        )
        ifc_wall.Representation = product_shape

        # Registry material on the body Solid (v1.5) attaches to the wall product.
        # Precedence: explicit color > material render > body type > "wall".
        styled = _apply_registry_material(model, ifc_wall, result_solid, body, cache)
        if not styled:
            # Apply wall color: explicit color > body type > container default "wall"
            color_key = body.color if body.color else (body.type if body.type in ELEMENT_TYPE_COLORS else "wall")
            _apply_typed_color(model, result_solid, color_key, cache)

    # Process Window/Door children as openings
    _create_wall_openings_from_assembly(model, context, ifc_wall, wall, body, cache)

    _aggregate_wall_details(model, context, ifc_wall, wall, body, cache)

    # A Wall carries its body ON the IfcWall, so voids attach to the product —
    # the wall's OWN ``.void()`` operands and the BODY's alike. The body's were
    # dropped in silence until WS-A: a Wall's body is rendered here, never by
    # ``_create_box``, so nothing ever read its ``_voids`` and
    # ``body.void(tool)`` inside a Wall emitted no IfcOpeningElement, no
    # IfcRelVoidsElement and no boolean, while the identical
    # ``wall.void(tool)`` worked. Same shape as the ``.opening()`` drop §1.1
    # exists to close, and found the same way.
    _process_voids_on_element(model, context, ifc_wall, wall, cache)
    if body is not None and body is not wall:
        _process_voids_on_element(model, context, ifc_wall, body, cache)
    return ifc_wall


def _aggregate_wall_details(model, context, ifc_wall, wall: Wall, body,
                            cache: EntityCache) -> None:
    """Emit and aggregate every Wall child that is not the body or an opening.

    The body itself is excluded by IDENTITY, not by type: a second Box in a
    Wall is a detail (a corbel, a sign, a cladding piece), and only the first
    prism is the wall's own geometry.
    """
    details = _emit_offtable_children(
        model, context, wall, wall._elements, cache, kind="Wall",
        handled=lambda c: c is body or tx.is_opening(c))
    if details:
        _assign_aggregate(model, ifc_wall, details)


def _create_wall_contour(model, context, wall: Wall, body, cache: EntityCache):
    """
    Create IfcWall from a contour-mode body Solid (gable-end / sloped-top
    walls — ``Wall(...).add(Extrude(contour=[...], thickness=))``).

    Same Newell-plane extrusion machinery as _create_solid_contour, so a
    gable wall body and a standalone contour Solid with the same points
    produce the same extrusion — but the product is the semantic IfcWall,
    the appearance follows the box-wall precedence (registry material,
    else explicit color > body type > container default "wall"), and
    Window/Door children still become IfcOpeningElement voids (see
    _create_wall_openings_from_assembly, which derives the run axis from
    the contour plane for this mode).

    validate_project_report guarantees the contour is a planar vertical
    polygon with a thickness source; the checks below stay defensive for
    hand-built buildings and warn loudly instead of silently reshaping.
    """
    assert run is not None
    assert np is not None

    thickness_m = body.thickness
    if thickness_m is None:
        # v1.5 one-source rule: a sheet Material carries the thickness.
        thickness_m = _material_thickness_m(body)
    if thickness_m is None:
        logger.warning(
            "Wall %r: contour body has no thickness source (thickness= or "
            "Material thickness_mm) — wall skipped entirely "
            "(validate_project_report rejects this at compile)",
            wall.ifc_name,
        )
        return None

    contour_3d = [(pt.x, pt.y, pt.z) for pt in body.contour]
    if len(contour_3d) < 3:
        logger.warning(
            "Wall %r: contour body has fewer than 3 points — wall skipped "
            "entirely",
            wall.ifc_name,
        )
        return None

    # Plane frame via Newell's method; orientation rules never flip a
    # vertical surface, so the extrusion side is the author's winding.
    normal = _newell_normal(contour_3d)
    normal = _apply_orientation_rules(normal)
    centroid = _calculate_centroid(contour_3d)

    # Local coordinate system from the normal (Up-Preferred strategy —
    # identical to _create_solid_contour). For a vertical plane this makes
    # x_axis the in-plane horizontal direction and y_axis point up.
    z_axis = np.array(normal)
    if abs(z_axis[2]) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(ref, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 0.001:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)

    # Project 3D contour points onto the local 2D plane
    contour_2d = []
    for pt in contour_3d:
        v = np.array([pt[0] - centroid[0], pt[1] - centroid[1], pt[2] - centroid[2]])
        contour_2d.append((float(np.dot(v, x_axis)), float(np.dot(v, y_axis))))

    ifc_origin = cache.get_or_create_point((centroid[0], centroid[1], centroid[2]))
    axis = cache.get_or_create_direction(
        float(z_axis[0]), float(z_axis[1]), float(z_axis[2]))
    ref_direction = cache.get_or_create_direction(
        float(x_axis[0]), float(x_axis[1]), float(x_axis[2]))
    axis2_placement = model.create_entity(
        "IfcAxis2Placement3D", Location=ifc_origin,
        Axis=axis, RefDirection=ref_direction)
    local_placement = model.create_entity(
        "IfcLocalPlacement", RelativePlacement=axis2_placement)

    ifc_wall = _create_ifc_entity(model, "IfcWall", name=wall.ifc_name)
    ifc_wall.ObjectPlacement = local_placement

    # Sweep from the local 2D contour
    points = [model.create_entity("IfcCartesianPoint", Coordinates=[p[0], p[1]])
              for p in contour_2d]
    points.append(points[0])
    polyline = model.create_entity("IfcPolyline", Points=points)
    profile = model.create_entity(
        "IfcArbitraryClosedProfileDef", ProfileType="AREA", OuterCurve=polyline)

    # Body booleans: same semantics as the box wall body — the Solid's
    # .cuts()/.adds() operands apply in the contour's local frame.
    cuts = body._cuts if body._cuts else []
    adds = body._adds if body._adds else []
    intersects = body._intersects if getattr(body, "_intersects", None) else []
    clips = body._clips if getattr(body, "_clips", None) else []
    if cuts or adds or intersects or clips:
        local_origin = cache.get_origin_3d()
        solid_placement = model.create_entity(
            "IfcAxis2Placement3D", Location=local_origin)
        base_solid = model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile, Position=solid_placement,
            ExtrudedDirection=cache.get_z_up_direction(), Depth=thickness_m,
        )
        result_solid = _apply_contour_boolean_operations(
            model, base_solid, cuts, adds,
            centroid, x_axis, y_axis, z_axis, thickness_m, cache,
            intersects=intersects, clips=clips,
        )
        ifc_wall.Representation = _create_body_representation(
            model, context, result_solid)
    else:
        shape_rep, result_solid = _create_extruded_body(
            model, context, profile, thickness_m, cache)
        _assign_body_representation(model, ifc_wall, shape_rep)

    # Appearance: SAME path as the box wall body (§6 precedence chain is
    # documented on _apply_container_color).
    styled = _apply_registry_material(model, ifc_wall, result_solid, body, cache)
    if not styled:
        color_key = body.color if body.color else (
            body.type if body.type in ELEMENT_TYPE_COLORS else "wall")
        _apply_typed_color(model, result_solid, color_key, cache)

    # Process Window/Door children as openings
    _create_wall_openings_from_assembly(model, context, ifc_wall, wall, body, cache)

    _aggregate_wall_details(model, context, ifc_wall, wall, body, cache)

    # A Wall carries its body ON the IfcWall, so voids attach to the product —
    # the wall's OWN ``.void()`` operands and the BODY's alike. The body's were
    # dropped in silence until WS-A: a Wall's body is rendered here, never by
    # ``_create_box``, so nothing ever read its ``_voids`` and
    # ``body.void(tool)`` inside a Wall emitted no IfcOpeningElement, no
    # IfcRelVoidsElement and no boolean, while the identical
    # ``wall.void(tool)`` worked. Same shape as the ``.opening()`` drop §1.1
    # exists to close, and found the same way.
    _process_voids_on_element(model, context, ifc_wall, wall, cache)
    if body is not None and body is not wall:
        _process_voids_on_element(model, context, ifc_wall, body, cache)
    return ifc_wall


#: The opening frame + placement derivation lives in ``compiler.frames``,
#: read here and by ``BimElement._world_matrix_for`` (the authoring-query
#: composer). Before
#: that third caller existed, a query on a Box inside an anchored Window
#: answered in opening-LOCAL coordinates — silently, metres from where this
#: generator actually placed it.
OpeningFrame = frames.OpeningFrame


def _derive_opening_frame(elem) -> Optional[frames.OpeningFrame]:
    """The SINGLE opening-frame derivation for any voidable body Solid.

    A thin unit adapter over ``frames.opening_host_frame``: this backend runs
    on the normalized (meters) project, so it hands in the meters snapper and
    the mm->m divisor a Material's ``thickness_mm`` needs. Everything else —
    box vs contour dispatch, the outer-face origin, the run axis — is the
    shared derivation.

    ``None`` when the body bears no usable frame (degenerate or non-vertical
    contour, no thickness source, neither mode), all of which
    ``validate_project_report`` rejects at compile; the callers turn it into
    a loud error rather than a silent drop.
    """
    return frames.opening_host_frame(elem, snap=snap_to_precision,
                                     divisor=1000.0)


def _create_wall_openings_from_assembly(model, context, ifc_wall, wall: Wall, body, cache: EntityCache):
    """
    Create IfcOpeningElements for Window/Door children in a Wall assembly.

    Derives wall direction from the body Solid: from the axis-aligned box
    for a box-mode body, or from the contour plane (in-plane horizontal
    run axis, extrusion depth as thickness) for a contour-mode body — see
    _derive_contour_wall_frame. Opening semantics are identical in both
    modes: offset= along the run from the wall start, sill= above the
    wall's minimum z, void cut through the full thickness.

    For each Window/Door child:
    1. Create IfcOpeningElement with box geometry (from offset/sill/width/height)
    2. Link to wall via IfcRelVoidsElement
    3. Create IfcDoor/IfcWindow fill entity and link via IfcRelFillsElement
    4. Aggregate fill entities to parent wall (for hierarchy tree visibility)

    An opening whose box extends outside a contour body (e.g. a window up
    in the gable slope past the roofline) is emitted as authored — IFC
    allows a void partially outside the body; no clipping, no silent
    repositioning.
    """
    assert run is not None
    assert np is not None

    # Collect Window/Door children
    # The carve trigger is a property of the CHILD (``brings_void``), not a
    # type test on the verb's argument — see ``taxonomy.brings_void``.
    children = [e for e in wall._elements if tx.brings_void(e)]
    if not children:
        return

    # One derivation for box AND contour bodies, at any plan angle.
    frame = _derive_opening_frame(body)
    if frame is None:
        # validate_project_report rejects degenerate/non-vertical/no-
        # thickness contour bodies at compile; reaching here means the
        # proj skipped validation — fail LOUD, never a silent drop.
        raise ValueError(
            f"Wall {wall.ifc_name!r}: openings require a box or vertical "
            f"planar contour body — validate_project_report should have "
            f"rejected this"
        )

    fill_entities = []
    for child in children:
        ifc_fill = _create_opening_from_assembly(model, context, ifc_wall,
                                                 child, frame, cache)
        if ifc_fill:
            fill_entities.append(ifc_fill)

    # Aggregate Windows/Doors to parent Wall for hierarchy tree visibility
    if fill_entities:
        _assign_aggregate(model, ifc_wall, fill_entities)


#: Opening placement reads the ONE derivation in ``compiler.frames`` — the
#: scalar spec (``along``/``up``/``inset``) and the frame axes — so both IFC
#: backends place a Window/Door identically.
_opening_spec = frames.opening_spec
_opening_out_sign = frames.opening_out_sign


def _create_opening_from_assembly(model, context, ifc_wall, child,
                                   frame: frames.OpeningFrame,
                                   cache: EntityCache):
    """
    Create IfcOpeningElement + fill for a Window/Door assembly child.

    Opening size comes from ``child.width``/``child.height``; the POSITION
    comes from the host's anchor spec (``along`` = the old ``offset=``, ``up``
    = the old ``sill=``, ``inset`` = how far the FILL sits back from the outer
    face). Dimensions are in meters (normalized by executor).

    ``wall_start`` is on the OUTER face (v21), so the VOID — which cuts the
    full thickness and is centred on its own placement — steps half a
    thickness INWARD from there. That step is the only reason the void and
    the fill do not share an origin.

    An INFERRED size brings an origin offset with it
    (``frames.opening_origin_offset``): the hole is the joinery's bounding
    box, so it starts at the joinery's min corner rather than at the anchor.
    The offset moves the void and the fill together and leaves the CHILDREN
    where they were authored — which is what makes off-wall joinery carve
    off-wall instead of opening a hole between the anchor and itself.

    The one-host spelling of :func:`_create_opening_on_hosts`, which is the
    implementation. A ``Wall`` and a standalone ``Box``/``Extrude`` carry their
    own representation, so the hole lands on exactly one product; only a
    representation-less container needs the plural form.
    """
    return _create_opening_on_hosts(model, context, [ifc_wall], child, frame,
                                    cache)


def _create_opening_on_hosts(model, context, host_products, child,
                             frame: frames.OpeningFrame, cache: EntityCache,
                             fill_index: int = 0):
    """One authored hole, ``len(host_products)`` ``IfcOpeningElement``s, ONE
    fill. Returns the fill product (or ``None`` for a non-Window/Door child).

    **Why N openings and one fill, measured against IFC4X3_ADD2**::

        IfcElement.FillsVoids          bound = 0 .. 1
        IfcElement.HasOpenings         bound = 0 .. *
        IfcOpeningElement.HasFillings  bound = 0 .. *

    ``FillsVoids`` is [0:1] — an element fills AT MOST ONE opening — so a hole
    distributed across N leaves cannot be filled N times by one ``IfcWindow``.
    The schema-clean alternative, one ``IfcWindow`` per leaf, **double-counts
    every window in every schedule**, which is the objection that blocked
    spelling a layered wall's core with ``.opening()`` in the first place
. So the other N−1 leaves carry unfilled holes — which is also
    physically honest: a window sits in ONE layer of a buildup and the rest
    simply have holes through them.

    ``fill_index`` picks which one. It is an index and not a search so that
    the CHOICE lives with the caller who knows the geometry
    (``_process_container_openings`` picks the outermost leaf; see
    :func:`_outermost_host_index` for the rule and why) while the emission
    stays one path.

    Every opening is emitted at the SAME placement off the SAME frame, so the
    N holes are one hole; only the ``IfcRelVoidsElement`` differs. The name is
    shared too, exactly as ``_add_one_void`` shares ``_void_canonical`` across
    a distributed ``.void()``.
    """
    assert run is not None
    assert np is not None

    if not host_products:
        return None

    # Void / fill / child-frame origins, from the SINGLE derivation both
    # backends and the authoring-query composer read.
    place = frames.opening_placement(child, frame)
    cos_a, sin_a = place.cos_a, place.sin_a
    width_m, height_m = place.width, place.height
    depth_m = place.thickness            # Opening cuts through full wall

    # ``place.void`` is centred on the hole in plan and stepped half a
    # thickness inward, because the void's rect profile is centred on its
    # placement and must still span the FULL wall depth.
    matrix = np.array([
        [cos_a, -sin_a, 0, place.void[0]],
        [sin_a, cos_a, 0, place.void[1]],
        [0, 0, 1, place.void[2]],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    filled_opening = None
    for index, host_product in enumerate(host_products):
        # Emission ORDER inside this loop is the single-host order verbatim —
        # opening, placement, profile, body, feature — because the corpus hash
        # is byte-level: hoisting the (cached) profile out of the loop changed
        # nothing semantically and moved three committed models.
        #
        # Create opening entity. DSL v2.1: derived names track the canonical —
        # a NAMED opening's void is ``<canonical>_void``; an ANONYMOUS opening
        # (ifc_name None) emits Name=None, never the literal "None_void" (which
        # is now reachable, reversing the pre-v2.1 note). The void inherits
        # the opening's anonymity: it belongs to the same named ancestor.
        ifc_opening = _create_ifc_entity(
            model, "IfcOpeningElement",
            name=f"{child.ifc_name}_void" if child.ifc_name else None,
        )
        _set_placement(model, ifc_opening, matrix, cache)

        # Opening box geometry (width x depth profile, extruded to height)
        profile = cache.get_or_create_rect_profile(
            x_dim=width_m, y_dim=depth_m,
            name=f"{child.ifc_name}_opening_profile" if child.ifc_name else None
        )
        shape_rep, _ = _create_extruded_body(model, context, profile, height_m,
                                             cache)
        _assign_body_representation(model, ifc_opening, shape_rep)

        # Link opening to its host via IfcRelVoidsElement
        run("feature.add_feature", model, feature=ifc_opening,
            element=host_product)
        if index == fill_index:
            filled_opening = ifc_opening

    if filled_opening is None:
        # LOUD. ``fill_index`` out of range means the caller's choice rule and
        # its host list disagree, and the quiet outcome is a hole with no
        # window in it — the silence refused.
        raise ValueError(
            f"opening {child.ifc_name!r}: fill_index {fill_index} is outside "
            f"the {len(host_products)} host product(s) the hole was "
            f"distributed to"
        )

    # Create fill element (IfcDoor or IfcWindow). ``inset=`` moves the FILL
    # only — the void above already cuts the full depth.
    ifc_fill = _create_fill_from_assembly(model, context, filled_opening, child,
                                          place, cache)
    return ifc_fill


def _create_fill_from_assembly(model, context, ifc_opening, child,
                                place: frames.OpeningPlacement,
                                cache: EntityCache):
    """
    Create IfcDoor or IfcWindow to fill an opening from an assembly child.

    ``place.fill`` is the void's own origin moved back by ``inset=`` — the
    depth the fill sits behind the OUTER face, along the fill's local +Y. It
    needs no facing sign of its own: the frame it was built from is already
    the wall's own (``+x`` right-as-seen-from-outside, ``+y`` inward), so one
    value means the same thing on every wall. It moves the fill product and,
    with it, the children aggregated to it — the reveal depth of a recessed
    window. The void is untouched: a Door/Window is a SEMANTIC opening that
    always cuts the full wall thickness, so a recess must never turn into a
    shallower hole. ``inset=0`` puts the fill FLUSH with the facade — the
    same zero the children are authored from, which is the whole point of
    moving the datum off the invisible midline.

    ``place.corner`` is where those children start. It is NOT ``place.fill``
    shifted by half a width: an inferred opening's hole has slid onto the
    joinery, and the corner steps that offset back out so the joinery stays
    exactly where the author drew it while the hole moves onto it.
    """
    assert run is not None
    assert np is not None

    cos_a, sin_a = place.cos_a, place.sin_a
    fill_x, fill_y, fill_z = place.fill

    if isinstance(child, Door):
        ifc_fill = _create_ifc_entity(model, "IfcDoor", name=child.ifc_name, predefined_type="DOOR")
    elif isinstance(child, Window):
        ifc_fill = _create_ifc_entity(model, "IfcWindow", name=child.ifc_name, predefined_type="WINDOW")
    else:
        return None

    # Place fill at opening position
    matrix = np.array([
        [cos_a, -sin_a, 0, fill_x],
        [sin_a, cos_a, 0, fill_y],
        [0, 0, 1, fill_z],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    _set_placement(model, ifc_fill, matrix, cache)

    # NEITHER a Window nor a Door carries its own body: an opening's geometry
    # is its children (glass + frame for a window, the leaf for a door).
    #
    # No forced slab: the child processing below emits the leaf exactly as it
    # emits a window's frame, so an opening's size stays inferred from its
    # children. A Door with no children is a clean hole, symmetric with a bare
    # Window.

    # Link fill to opening via IfcRelFillsElement
    run("feature.add_filling", model, opening=ifc_opening, element=ifc_fill)

    # Process child elements (Sweep frames, glass, etc.)
    # Children use opening-local coords: origin at bottom-left corner of opening
    # fill_x/fill_y are at the CENTER of the opening, so shift back to corner
    if hasattr(child, '_elements') and child._elements:
        _process_opening_children(model, context, child, ifc_fill,
                                   place.corner[0], place.corner[1],
                                   place.corner[2], cos_a, sin_a, cache)

    return ifc_fill


def _process_opening_children(model, context, opening_elem, parent_fill,
                                origin_x, origin_y, origin_z,
                                cos_a, sin_a, cache: EntityCache):
    """
    Create IFC geometry for child elements (Sweep, Solid) of a Window/Door.

    Aggregates child elements to the parent IfcWindow/IfcDoor using IfcRelAggregates,
    creating proper nested hierarchy instead of flat storey containment.

    Child coordinates are in opening-local space — **the wall's own frame,
    identical on every wall of the building** (Z-up, meters):
      Origin: the opening's lower-left corner on the wall's OUTER face
      X: 0 to width — right, AS SEEN FROM OUTSIDE
      Y: 0 at the outer face, positive INTO the wall (``inset=`` measures the
         same way, from the same zero)
      Z: 0 to height (up from sill)

    Pre-v21, Y was measured from the wall MIDLINE and its world direction
    followed a run axis canonicalized toward +X — so the identical local
    coordinate landed inward on a south wall and outward on a north one, and
    every window/door skill carried a per-facade ``local_y_sign`` to undo it.
    ``cos_a``/``sin_a`` now carry the facing, so that sign is gone.

    Transform to global IFC coords:
      ifc_x = origin_x + x_local * cos_a + y_local * (-sin_a)
      ifc_y = origin_y + x_local * sin_a + y_local * cos_a
      ifc_z = origin_z + z_local  (local Z = IFC Z = up)

    Args:
        model: ifcopenshell model
        context: IFC geometric representation context
        opening_elem: the Window/Door DSL element with _elements
        parent_fill: the IfcWindow or IfcDoor parent element
        origin_x, origin_y, origin_z: IFC global coords of opening origin
        cos_a, sin_a: wall direction rotation components
        cache: EntityCache for deduplication
    """
    assert run is not None
    assert np is not None
    from lite_step.models import Sweep as SweepElem, Box as BoxElem, Extrude as ExtrudeElem

    child_products = []

    opening_type = type(opening_elem).__name__  # "Window" / "Door"
    for child in opening_elem._elements:
        if isinstance(child, SweepElem):
            ifc_product = _create_opening_profile(model, context, child,
                                                    origin_x, origin_y, origin_z,
                                                    cos_a, sin_a, cache,
                                                    parent_fill=parent_fill)
            if ifc_product:
                child_products.append(ifc_product)
            else:
                _warn_child_dropped(opening_type, opening_elem.ifc_name, child)
        elif isinstance(child, BoxElem) and child.start is not None and child.end is not None:
            ifc_product = _create_opening_solid(model, context, child,
                                                  origin_x, origin_y, origin_z,
                                                  cos_a, sin_a, cache)
            if ifc_product:
                child_products.append(ifc_product)
            else:
                _warn_child_dropped(opening_type, opening_elem.ifc_name, child)
        elif isinstance(child, ExtrudeElem) and child.contour and len(child.contour) >= 3:
            ifc_product = _create_opening_contour_solid(model, context, child,
                                                          origin_x, origin_y, origin_z,
                                                          cos_a, sin_a, cache)
            if ifc_product:
                child_products.append(ifc_product)
            else:
                _warn_child_dropped(opening_type, opening_elem.ifc_name, child)
        else:
            # Defensive: .add() validation rejects off-table children at
            # construction; reachable only via hand-built _elements lists.
            logger.warning(
                "%s %r: unsupported child %s %r skipped (%s takes Box, "
                "Extrude, Sweep)",
                opening_type, opening_elem.ifc_name, type(child).__name__,
                getattr(child, 'ifc_name', None) or '<anonymous>',
                opening_type,
            )

    # Aggregate child elements to parent Window/Door
    if child_products:
        _assign_aggregate(model, parent_fill, child_products)


def _create_opening_profile(model, context, profile_elem,
                              origin_x, origin_y, origin_z,
                              cos_a, sin_a, cache: EntityCache,
                              parent_fill=None):
    """
    Create an IfcMember for a Sweep child of a Window/Door.

    Sweep start/end are in opening-local space (meters).
    We transform to global IFC coords and create an extruded profile.

    v1.5 (WS1 PR-E): the path form (mitered window frames) routes to
    _create_opening_profile_path — path points are opening-local and get
    the same wall-rotation transform.
    """
    assert run is not None
    assert np is not None

    if getattr(profile_elem, 'path', None) is not None:
        return _create_opening_profile_path(model, context, profile_elem,
                                            origin_x, origin_y, origin_z,
                                            cos_a, sin_a, cache,
                                            parent_fill=parent_fill)

    if profile_elem.start is None or profile_elem.end is None:
        return

    # Transform local coords to IFC global
    # Local (Z-up, matching world): x=along wall, y=depth through wall, z=up
    s = profile_elem.start
    e = profile_elem.end

    start_global = (
        origin_x + s.x * cos_a + s.y * (-sin_a),
        origin_y + s.x * sin_a + s.y * cos_a,
        origin_z + s.z
    )
    end_global = (
        origin_x + e.x * cos_a + e.y * (-sin_a),
        origin_y + e.x * sin_a + e.y * cos_a,
        origin_z + e.z
    )

    # Section source (v17): a member Material carries profile_mm. When the
    # frame has no material profile, fall back to a standard 89x38 stud
    # section (the legacy bim_catalog "Stud_2x4" default; catalog retired).
    material_selection = _registry_material_selection(profile_elem)
    if material_selection is not None and material_selection.profile_mm is not None:
        profile_mm = material_selection.profile_mm
    else:
        profile_mm = (89, 38)

    # Calculate direction and length
    start_ifc = np.array(start_global)
    end_ifc = np.array(end_global)
    direction = end_ifc - start_ifc
    length = np.linalg.norm(direction)

    if length < 0.001:
        logger.warning(
            "dropped degenerate Sweep %r: opening-local start/end collapse "
            "to a point in world coords (length=%.6f m). Common cause: "
            "opening-local was authored Y-up (legacy) but the generator "
            "expects Z-up — height should be in pt.z, not pt.y.",
            profile_elem.ifc_name or "<unnamed>",
            length,
        )
        return

    # Create IfcMember
    ifc_member = _create_ifc_entity(model, "IfcMember", name=profile_elem.ifc_name)

    # Calculate placement: midpoint, looking from start to end
    midpoint = (start_ifc + end_ifc) / 2.0
    dir_norm = direction / length

    # Build rotation matrix: local Z → dir_norm
    local_z = dir_norm
    # Choose an up vector that's not parallel to dir_norm
    if abs(local_z[2]) < 0.99:
        up = np.array([0, 0, 1], dtype=np.float64)
    else:
        up = np.array([1, 0, 0], dtype=np.float64)

    local_x = np.cross(up, local_z)
    local_x_norm = np.linalg.norm(local_x)
    if local_x_norm < 1e-6:
        local_x = np.array([1, 0, 0], dtype=np.float64)
    else:
        local_x = local_x / local_x_norm
    local_y = np.cross(local_z, local_x)

    matrix = np.array([
        [local_x[0], local_y[0], local_z[0], midpoint[0]],
        [local_x[1], local_y[1], local_z[1], midpoint[1]],
        [local_x[2], local_y[2], local_z[2], midpoint[2]],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    _set_placement(model, ifc_member, matrix, cache)

    # Create profile geometry (section dims are mm, convert to meters)
    w = profile_mm[0] / 1000.0
    d = profile_mm[1] / 1000.0
    profile = cache.get_or_create_rect_profile(
        x_dim=w, y_dim=d,
        name=f"{profile_elem.ifc_name}_profile" if profile_elem.ifc_name else None)

    # Offset so profile is centered along length
    z_offset = -length / 2.0
    prof_origin = cache.get_or_create_point([0.0, 0.0, z_offset])
    prof_placement = model.create_entity("IfcAxis2Placement3D", Location=prof_origin)

    solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=prof_placement,
        ExtrudedDirection=cache.get_z_up_direction(),
        Depth=length
    )

    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[solid]
    )
    product_shape = model.create_entity(
        "IfcProductDefinitionShape",
        Representations=[shape_rep]
    )
    ifc_member.Representation = product_shape

    # Registry material (v1.5): IfcMaterial + Psets + render style.
    styled = _apply_registry_material(model, ifc_member, solid, profile_elem, cache)
    if not styled:
        # Apply color based on type
        element_type = getattr(profile_elem, 'type', '') or ''
        _apply_typed_color(model, solid, element_type, cache)

    return ifc_member


def _create_opening_solid(model, context, solid_elem,
                            origin_x, origin_y, origin_z,
                            cos_a, sin_a, cache: EntityCache):
    """
    Create an IfcBuildingElementProxy for a Solid child of a Window/Door (e.g. door body).

    Solid start/end are in opening-local space (meters).
    We transform to global IFC coords.
    """
    assert run is not None
    assert np is not None

    if solid_elem.start is None or solid_elem.end is None:
        return

    s = solid_elem.start
    e = solid_elem.end

    # Transform local box corners to IFC global
    # Local (Z-up, matching world): x=along wall, y=depth through wall, z=up
    start_global = (
        origin_x + s.x * cos_a + s.y * (-sin_a),
        origin_y + s.x * sin_a + s.y * cos_a,
        origin_z + s.z
    )
    end_global = (
        origin_x + e.x * cos_a + e.y * (-sin_a),
        origin_y + e.x * sin_a + e.y * cos_a,
        origin_z + e.z
    )

    min_pt = (
        min(start_global[0], end_global[0]),
        min(start_global[1], end_global[1]),
        min(start_global[2], end_global[2])
    )
    max_pt = (
        max(start_global[0], end_global[0]),
        max(start_global[1], end_global[1]),
        max(start_global[2], end_global[2])
    )

    width = snap_to_precision(max_pt[0] - min_pt[0])
    depth = snap_to_precision(max_pt[1] - min_pt[1])
    height = snap_to_precision(max_pt[2] - min_pt[2])

    if width < 0.001 or depth < 0.001 or height < 0.001:
        return

    center = (
        (min_pt[0] + max_pt[0]) / 2.0,
        (min_pt[1] + max_pt[1]) / 2.0,
        (min_pt[2] + max_pt[2]) / 2.0
    )

    ifc_proxy = _create_ifc_entity(model, "IfcBuildingElementProxy", name=solid_elem.ifc_name)

    matrix = np.array([
        [1, 0, 0, center[0]],
        [0, 1, 0, center[1]],
        [0, 0, 1, center[2]],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    _set_placement(model, ifc_proxy, matrix, cache)

    profile = cache.get_or_create_rect_profile(x_dim=width, y_dim=depth)
    z_offset = -height / 2.0
    origin_pt = cache.get_or_create_point([0.0, 0.0, z_offset])
    placement = model.create_entity("IfcAxis2Placement3D", Location=origin_pt)

    solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=placement,
        ExtrudedDirection=cache.get_z_up_direction(),
        Depth=height
    )

    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[solid]
    )
    product_shape = model.create_entity(
        "IfcProductDefinitionShape",
        Representations=[shape_rep]
    )
    ifc_proxy.Representation = product_shape

    # Registry material (v1.5): IfcMaterial + Psets + render style.
    styled = _apply_registry_material(model, ifc_proxy, solid, solid_elem, cache)
    if not styled:
        element_type = solid_elem.type or ''
        color_key = solid_elem.color if solid_elem.color else element_type
        _apply_typed_color(model, solid, color_key, cache)

    return ifc_proxy


def _create_opening_contour_solid(model, context, solid_elem,
                                    origin_x, origin_y, origin_z,
                                    cos_a, sin_a, cache: EntityCache):
    """
    Create an IfcBuildingElementProxy for a contour-based Solid child of a
    Window/Door (e.g. door leaf, window/door frame pieces).

    Same geometry logic as _create_opening_panel but produces IfcBuildingElementProxy
    The contour-Solid path (v1.5 glass panes + sheet children); the legacy Panel primitive was removed at the cutover.
    """
    assert run is not None
    assert np is not None

    if not solid_elem.contour or len(solid_elem.contour) < 3:
        return

    # Transform contour points from opening-local to IFC global
    # Local (Z-up, matching world): x=along wall, y=depth through wall, z=up
    ifc_points = []
    for pt in solid_elem.contour:
        gx = origin_x + pt.x * cos_a + pt.y * (-sin_a)
        gy = origin_y + pt.x * sin_a + pt.y * cos_a
        gz = origin_z + pt.z
        ifc_points.append((gx, gy, gz))

    # Newell's method for face normal
    normal = [0.0, 0.0, 0.0]
    n = len(ifc_points)
    for i in range(n):
        curr = ifc_points[i]
        nxt = ifc_points[(i + 1) % n]
        normal[0] += (curr[1] - nxt[1]) * (curr[2] + nxt[2])
        normal[1] += (curr[2] - nxt[2]) * (curr[0] + nxt[0])
        normal[2] += (curr[0] - nxt[0]) * (curr[1] + nxt[1])

    norm_len = (normal[0]**2 + normal[1]**2 + normal[2]**2) ** 0.5
    if norm_len < 1e-9:
        logger.warning(
            "dropped degenerate Solid contour %r in opening: Newell normal "
            "is zero — contour points collapsed to a line in world coords "
            "(norm_len=%.3e). Common cause: opening-local was authored Y-up "
            "(legacy) but the generator expects Z-up — height should be in "
            "pt.z, not pt.y.",
            solid_elem.ifc_name or "<unnamed>",
            norm_len,
        )
        return

    normal = [c / norm_len for c in normal]

    cx = sum(p[0] for p in ifc_points) / n
    cy = sum(p[1] for p in ifc_points) / n
    cz = sum(p[2] for p in ifc_points) / n

    local_z = np.array(normal, dtype=np.float64)

    edge = np.array([
        ifc_points[1][0] - ifc_points[0][0],
        ifc_points[1][1] - ifc_points[0][1],
        ifc_points[1][2] - ifc_points[0][2]
    ], dtype=np.float64)
    edge_len = np.linalg.norm(edge)
    if edge_len < 1e-9:
        logger.warning(
            "dropped Solid contour %r in opening: first contour edge has "
            "zero length (edge_len=%.3e). Check for duplicate contour points.",
            solid_elem.ifc_name or "<unnamed>",
            edge_len,
        )
        return
    local_x = edge / edge_len
    local_y = np.cross(local_z, local_x)

    pts_2d = []
    for p in ifc_points:
        dx = p[0] - cx
        dy = p[1] - cy
        dz = p[2] - cz
        u = dx * local_x[0] + dy * local_x[1] + dz * local_x[2]
        v = dx * local_y[0] + dy * local_y[1] + dz * local_y[2]
        pts_2d.append((u, v))

    ifc_pts_2d = [model.create_entity("IfcCartesianPoint", Coordinates=[float(u), float(v)]) for u, v in pts_2d]
    ifc_pts_2d.append(ifc_pts_2d[0])
    polyline = model.create_entity("IfcPolyline", Points=ifc_pts_2d)
    profile = model.create_entity("IfcArbitraryClosedProfileDef",
                                   ProfileType="AREA", OuterCurve=polyline)

    # Thickness from the element, a sheet Material (v1.5 one-source rule),
    # or the 50 mm frame default.
    thickness = solid_elem.thickness or _material_thickness_m(solid_elem) or 0.050
    origin_pt = cache.get_or_create_point([0.0, 0.0, -thickness / 2.0])
    placement = model.create_entity("IfcAxis2Placement3D", Location=origin_pt)

    ifc_solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=placement,
        ExtrudedDirection=cache.get_z_up_direction(),
        Depth=thickness
    )

    ifc_proxy = _create_ifc_entity(model, "IfcBuildingElementProxy",
                                    name=solid_elem.ifc_name)

    matrix = np.array([
        [local_x[0], local_y[0], local_z[0], cx],
        [local_x[1], local_y[1], local_z[1], cy],
        [local_x[2], local_y[2], local_z[2], cz],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    _set_placement(model, ifc_proxy, matrix, cache)

    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[ifc_solid]
    )
    product_shape = model.create_entity(
        "IfcProductDefinitionShape",
        Representations=[shape_rep]
    )
    ifc_proxy.Representation = product_shape

    # Registry material (v1.5): IfcMaterial + Psets + render style.
    styled = _apply_registry_material(model, ifc_proxy, ifc_solid, solid_elem, cache)
    if not styled:
        element_type = solid_elem.type or ''
        color_key = solid_elem.color if solid_elem.color else element_type
        _apply_typed_color(model, ifc_solid, color_key, cache)

    return ifc_proxy


def _process_openings_on_element(model, context, ifc_product, elem, cache: EntityCache):
    """
    Create IfcOpeningElements for ``_openings`` on any voidable product.

    Derives the opening frame from the element's OWN geometry via
    ``_derive_opening_frame`` — a box AABB (start=/end=) or a vertical planar
    contour (contour=) — then reuses the same opening creation logic as wall
    assemblies. The frame's origin is on the OUTER face in both modes and the
    void steps half a thickness inward, so it still cuts through the full
    body depth. Contour bodies are supported (the earlier version silently
    no-op'd them because it only read start/end).

    Args:
        model: ifcopenshell model
        context: IFC geometric representation context
        ifc_product: the IfcProduct to void (e.g. IfcBuildingElementProxy)
        elem: the DSL element with _openings list
        cache: EntityCache for deduplication
    """
    assert run is not None
    assert np is not None

    openings = elem._openings if hasattr(elem, '_openings') else []
    if not openings:
        return

    # One derivation for box AND contour bodies, at any plan angle.
    frame = _derive_opening_frame(elem)
    if frame is None:
        # validate_project_report rejects opening-bearing bodies with no
        # box and no vertical planar contour at compile; reaching here means
        # the proj skipped validation — fail LOUD, never a silent drop.
        raise ValueError(
            f"{type(elem).__name__} {elem.ifc_name!r}: openings require a box "
            f"or vertical planar contour body — validate_project_report "
            f"should have rejected this"
        )

    for child in openings:
        _create_opening_from_assembly(model, context, ifc_product, child,
                                      frame, cache)


def _add_one_void(model, context, ifc_product, operand, cache: EntityCache, owner: str) -> bool:
    """Attach ONE ``.void()`` operand to ``ifc_product`` as an
    ``IfcOpeningElement`` + ``IfcRelVoidsElement``. Returns True if it emitted.

    The opening carries the name ``.void(name=)`` asked for — the canonical
    ``compiler.naming`` stamped on ``_void_canonical`` — and ``Name=None``
    when the author named nothing. Naming is the ONLY difference between the
    two; the geometry path below is one path.

    The void solid is built with the SAME operand machinery as ``.difference()``
    (``_create_hole_solid_local``) — so ``.void(x)`` and ``.difference(x)`` carve
    geometrically identical holes — but in world-absolute coords (a huge symmetric
    parent AABB so no operand face is treated as coplanar → no cut-face extension)
    and hung on an identity-placed opening; the host's own placement resolves the
    subtraction to world, exactly as the Window/Door opening path does.
    """
    assert run is not None
    assert np is not None

    world_center = (0.0, 0.0, 0.0)          # parent_center=origin → operand stays in world coords
    world_min = (-1e9, -1e9, -1e9)          # symmetric huge bounds → no coplanar-face extension
    world_max = (1e9, 1e9, 1e9)
    void_solid = _create_hole_solid_local(
        model, operand, world_min, world_center, cache, parent_max=world_max)
    if void_solid is None:
        logger.warning(
            "%s: .void() operand %s %r produced no solid — skipped",
            owner, type(operand).__name__,
            getattr(operand, "ifc_name", None) or "<anonymous>",
        )
        return False
    opening = _create_ifc_entity(
        model, "IfcOpeningElement",
        name=getattr(operand, "_void_canonical", None),
    )
    _set_placement(model, opening, np.eye(4, dtype=np.float64), cache)
    opening.Representation = _create_body_representation(model, context, void_solid)
    run("feature.add_feature", model, feature=opening, element=ifc_product)
    return True


def _process_voids_on_element(model, context, ifc_product, elem, cache: EntityCache):
    """Create ANONYMOUS IfcOpeningElements for ``.void()`` operands on a
    SELF-GEOMETRY element (a Box/Extrude primitive, or a Wall whose body is on
    the product itself). Every void attaches to this one product.

    Unlike ``.difference()`` — which folds the operand into the host's
    ``IfcBooleanResult`` (a hole invisible to schedules) — a ``.void()`` operand
    becomes a first-class ``IfcOpeningElement`` linked by ``IfcRelVoidsElement``
    (net/gross quantities and opening schedules see it), but ANONYMOUS
    (``Name=None``) and with no fill: the middle rung between ``.difference()``
    and the named ``.opening()``.
    """
    owner = f"{type(elem).__name__} {elem.ifc_name!r}"
    for operand in (getattr(elem, "_voids", None) or []):
        _add_one_void(model, context, ifc_product, operand, cache, owner)


def _process_container_voids(model, context, container, container_product,
                             cache: EntityCache):
    """Distribute a container's ``.void()`` operands to the geometry-bearing
    products in its aggregation subtree (Slab/Column/Beam/Roof/Element, and a
    body-less Wall since).

    An aggregate container (IfcSlab/IfcColumn/…) carries no representation of
    its own — the geometry is on the aggregated leaves — so an
    ``IfcRelVoidsElement`` on the container carves nothing in a renderer that
    builds each element's mesh from its OWN representation (web-ifc / ThatOpen,
    our production path). Each void therefore attaches to the leaves it
    OVERLAPS.

    Every decision is :mod:`lite_step.ifc.voids`' — which descendants are legal
    hosts (they own a shape slot, so a ``Representation=None`` aggregate parent
    is ineligible), at what depth (any), and which of them one operand reaches.
    This function contributes only the lines that mint the opening.

    The pairing between a DSL child and its product is READ, not inferred:
    ``cache.record_child_product`` stated it as each emitter ran. The
    ``zip(geom_leaves, child_products)`` this replaced could not see a nested
    container (no AABB of its own), a ``Mesh`` child (an AABB but no product of
    its own) or a layered container's slices (not in ``.elements`` at all), and
    its count-mismatch fallback then attached the hole to EVERY child product
    — including ones whose ``Representation`` is ``None`` and ones it does not
    overlap.
    """
    voids = getattr(container, "_voids", None) or []
    if not voids:
        return
    from lite_step.ifc.voids import void_targets

    owner = f"{type(container).__name__} {container.ifc_name!r}"
    adapter = _ModelShapeAdapter(model)

    for operand in voids:
        targets = void_targets(container, container_product, adapter,
                               cache.child_products, operand)
        if not targets:
            # LOUD. A hole that reaches nothing is either an authoring mistake
            # or a leaf this pass cannot see; both would otherwise end as an
                # opening on every child product — visible in the entity count, invisible in
            # the render.
            logger.warning(
                "%s: .void() operand %s %r overlaps no representation-bearing "
                "product in the container's subtree — no opening emitted",
                owner, type(operand).__name__,
                getattr(operand, "_void_canonical", None)
                or getattr(operand, "ifc_name", None) or "<anonymous>",
            )
            continue
        for host in targets:
            _add_one_void(model, context, host.product, operand, cache, owner)


def _outermost_host_index(hosts, frame: frames.OpeningFrame) -> int:
    """Which of ``hosts`` carries the ONE fill — the OUTERMOST leaf.

    A distributed opening yields N holes and exactly one ``IfcWindow``
    (``IfcElement.FillsVoids`` is [0:1]; see
    :func:`_create_opening_on_hosts`). This is the rule that says which leaf
    gets it, and it is stated here because a reader will otherwise assume "the
    first one".

    **The outermost leaf, measured along the host frame's OUTWARD normal.**
    Not the first, and not the leaf the frame was derived from — the frame is
    derived from the container's whole basis, so no single leaf owns it. Three
    reasons for outermost:

    * it is where the window physically IS. A buildup is cladding / core /
      lining; the joinery sits in the outer layer and the rest have holes
      through them, which is exactly what N-openings-one-fill models;
    * ``inset=`` is measured from the OUTER FACE (v21), so the fill's own
      datum and the leaf it is attached to are then the same surface. Attach
      it to the lining and ``inset=0`` puts a flush window on a product two
      layers behind the facade it is flush with;
    * it is stable under editing. Adding an inner leaf does not move the
      window; adding one in FRONT does, and should — the facade moved.

    Ties (leaves flush at the outer face, which is the ordinary case for a
    single-leaf container) resolve to emission order, i.e. the ``>`` below is
    strict and the first recorded host wins. Emission order is the order
    ``cache.record_child_product`` stated, so it is deterministic for one
    tree.
    """
    from lite_step.compiler.extent import MM_PER_METER, padded_aabb

    out_x, out_y = -frame.inward[0], -frame.inward[1]
    best, best_reach = 0, None
    for index, host in enumerate(hosts):
        box = padded_aabb(host.node, MM_PER_METER)
        if box is None:
            continue
        (x0, y0, _z0), (x1, y1, _z1) = box
        # The furthest this leaf reaches along the outward normal — a max over
        # the four plan corners, which for an axis-aligned normal reduces to
        # one face and stays correct for a skew one.
        reach = max(x * out_x + y * out_y
                    for x in (x0, x1) for y in (y0, y1))
        if best_reach is None or reach > best_reach:
            best, best_reach = index, reach
    return best


def _process_container_openings(model, context, container, container_product,
                                cache: EntityCache):
    """Distribute a container's ``.opening()`` holes to the geometry-bearing
    products in its aggregation subtree — for ``Column``/``Beam``/
    ``Element``, for a BODY-LESS ``Wall``.

    The twin of :func:`_process_container_voids`, and it exists for that
    function's reason verbatim: an aggregate container carries no
    representation of its own, so an ``IfcRelVoidsElement`` on it carves
    nothing in a renderer that builds each element's mesh from its OWN
    representation (web-ifc / ThatOpen, our production path). A ``Column``'s window
    would emit zero ``IfcOpeningElement`` and zero ``IfcWindow``, so it is
    refused loudly instead.

    Where it differs from the void twin, and only here:

    * the hole's extent is not the operand's. An ``.opening()`` carries
      HOST-FRAME scalars (``along=``/``up=``), so the world box comes from
      ``frames.opening_host_frame`` + ``opening_void_prism`` — the same two
      calls ``displacement`` makes — and ``voids.hole_targets`` takes that
      bound. ``along=4200`` on a 6 m host does NOT land at world ``x=4.2``;
      see ``test_opening_reach_matches_void.py``.
    * there is a FILL, and there is exactly one of it for N holes. See
      :func:`_create_opening_on_hosts` for the schema bound that forces it and
      :func:`_outermost_host_index` for which leaf carries it.

    ``Slab`` and ``Roof`` never reach here: ``along``/``inset``/``up`` assume
    an outer face and a run axis a horizontal host does not have, so
    ``taxonomy.OPENING_HOSTS`` still excludes them and
    ``executor._opening_host_errors`` still refuses. The gap stays open for that
    half.

    **A body-less ``Wall`` needs nothing else from this function**. Its
    route into it differs only in the CALLER: ``_create_wall``'s body-less
    branch, where ``_create_wall_openings_from_assembly`` cannot run because
    there is no body to derive a frame from and no ``Representation`` on the
    ``IfcWall`` to carry the hole. The frame comes from ``frame_basis_aabb``
    exactly as a ``Column``'s does (``frames.opening_container_aabb`` has been
    gated on ``tx.hosts_openings``, which already included ``Wall``, since), and a LAYERED leaf under such an assembly keeps working unchanged:
    its per-layer slices live on ONE product with one shape slot, so the leaf
    is a single distribution target and its ``IfcRelVoidsElement`` carves every
    slice.
    """
    openings = [c for c in (getattr(container, "_elements", None) or [])
                if tx.brings_void(c)]
    openings += [c for c in (getattr(container, "_openings", None) or [])
                 if tx.brings_void(c)]
    if not openings:
        return []
    from lite_step.ifc.voids import hole_targets

    owner = f"{type(container).__name__} {container.ifc_name!r}"
    frame = _derive_opening_frame(container)
    if frame is None:
        # validate_project_report rejects a container with no frame basis at
        # compile; reaching here means the proj skipped validation — fail
        # LOUD, never a silent drop.
        raise ValueError(
            f"{owner}: openings require a frame derived from the container's "
            f".add()ed children — validate_project_report should have "
            f"rejected this"
        )

    adapter = _ModelShapeAdapter(model)
    fill_entities = []
    for child in openings:
        prism = frames.opening_void_prism(child, frame)
        targets = hole_targets(container, container_product, adapter,
                               cache.child_products,
                               prism.aabb() if prism is not None else None)
        if not targets:
            # LOUD, exactly as the void twin is. A hole that reaches nothing is
            # an authoring mistake or a leaf this pass cannot see; either way
            # the window is not in the file and the author has to be told.
            logger.warning(
                "%s: opening %s %r overlaps no representation-bearing product "
                "in the container's subtree — no IfcOpeningElement and no "
                "fill emitted. along=/up= are measured in the HOST's frame, "
                "whose +x routinely opposes world +x.",
                owner, type(child).__name__,
                child.ifc_name or "<anonymous>",
            )
            continue
        ifc_fill = _create_opening_on_hosts(
            model, context, [t.product for t in targets], child, frame, cache,
            fill_index=_outermost_host_index(targets, frame))
        if ifc_fill:
            fill_entities.append(ifc_fill)

    # Aggregate Windows/Doors to the CONTAINER for hierarchy tree visibility —
    # the same line the Wall route ends on. The fill hangs off one leaf's
    # opening and belongs, in the tree an author reads, to the assembly.
    if fill_entities:
        _assign_aggregate(model, container_product, fill_entities)
    return fill_entities


def _create_column(model, context, column: Column, cache: EntityCache):
    """
    Create IfcColumn container with aggregated child face elements.

    The Column container:
    - Creates IfcColumn as the parent element
    - Processes child Solid/Sweep/Revolve elements as column geometry
    - Aggregates children under the IfcColumn

    Child appearance follows the precedence chain documented on
    ``_apply_container_color`` (container auto-color key: "column").
    """
    assert run is not None
    assert np is not None

    # Create IfcColumn container
    ifc_column = _create_ifc_entity(model, "IfcColumn", name=column.ifc_name)

    # Identity placement at origin
    matrix = np.eye(4)
    _set_placement(model, ifc_column, matrix, cache)

    # Process child elements as column geometry (spec: Solid, Sweep, Revolve)
    child_products = []
    for child in column.elements:
        if tx.is_prism(child):
            # Create the solid geometry
            face = _create_solid(model, context, child, cache)
            if face:
                child_products.append(face)
                cache.record_child_product(column, child, face)
                # Container-driven coloring: apply "column" color unless the
                # child has an explicit color or a registry material render
                if not child.color and _registry_material_selection(child) is None:
                    _apply_container_color(model, face, "column", cache)
            else:
                _warn_child_dropped("Column", column.ifc_name, child)
        elif isinstance(child, Revolve):
            face = _create_revolve(model, context, child, cache)
            if face:
                child_products.append(face)
                cache.record_child_product(column, child, face)
                if not child.color and _registry_material_selection(child) is None:
                    _apply_container_color(model, face, "column", cache)
            else:
                _warn_child_dropped("Column", column.ifc_name, child)
        elif isinstance(child, Sweep):
            face = _create_sweep(model, context, child, cache)
            if face:
                child_products.append(face)
                cache.record_child_product(column, child, face)
                if not child.color and _registry_material_selection(child) is None:
                    _apply_container_color(model, face, "column", cache)
            else:
                _warn_child_dropped("Column", column.ifc_name, child)
    # Everything else the branches above did not take — any type is
    # anchorable into any container, so these render instead of dropping.
    child_products.extend(_emit_offtable_children(
        model, context, column, column.elements, cache, kind="Column",
        handled=lambda c: (tx.is_prism(c) or tx.is_opening(c)
                           or isinstance(c, (Revolve, Sweep)))))

    # Aggregate children under IfcColumn
    if child_products:
        _assign_aggregate(model, ifc_column, child_products)

    _process_container_voids(model, context, column, ifc_column, cache)
    _process_container_openings(model, context, column, ifc_column, cache)
    return ifc_column


def _create_beam(model, context, beam: Beam, cache: EntityCache):
    """
    Create IfcBeam container with aggregated child face elements.

    The Beam container:
    - Creates IfcBeam as the parent element
    - Processes child Solid/Sweep elements as beam geometry and Bar
      children as reinforcement
    - Aggregates children under the IfcBeam

    Child appearance follows the precedence chain documented on
    ``_apply_container_color`` (container auto-color key: "beam"); Bar
    children are the one deliberate exception (see the Bar branch below).
    """
    assert run is not None
    assert np is not None

    # Create IfcBeam container
    ifc_beam = _create_ifc_entity(model, "IfcBeam", name=beam.ifc_name)

    # Identity placement at origin
    matrix = np.eye(4)
    _set_placement(model, ifc_beam, matrix, cache)

    # Process child elements as beam geometry (spec: Solid, Sweep, Bar)
    child_products = []
    for child in beam.elements:
        if tx.is_prism(child):
            # Create the solid geometry
            face = _create_solid(model, context, child, cache)
            if face:
                child_products.append(face)
                cache.record_child_product(beam, child, face)
                # Container-driven coloring: apply "beam" color unless the
                # child has an explicit color or a registry material render
                if not child.color and _registry_material_selection(child) is None:
                    _apply_container_color(model, face, "beam", cache)
            else:
                _warn_child_dropped("Beam", beam.ifc_name, child)
        elif isinstance(child, Bar):
            face = _create_bar(model, context, child, cache)
            if face:
                child_products.append(face)
                cache.record_child_product(beam, child, face)
                # INTENTIONAL exception to the auto-color rule: container
                # auto-color never applies to Bar — reinforcement keeps its
                # own appearance (registry render of its steel grade).
            else:
                _warn_child_dropped("Beam", beam.ifc_name, child)
        elif isinstance(child, Sweep):
            face = _create_sweep(model, context, child, cache)
            if face:
                child_products.append(face)
                cache.record_child_product(beam, child, face)
                if not child.color and _registry_material_selection(child) is None:
                    _apply_container_color(model, face, "beam", cache)
            else:
                _warn_child_dropped("Beam", beam.ifc_name, child)

    child_products.extend(_emit_offtable_children(
        model, context, beam, beam.elements, cache, kind="Beam",
        handled=lambda c: (tx.is_prism(c) or tx.is_opening(c)
                           or isinstance(c, (Bar, Sweep)))))

    # Aggregate children under IfcBeam
    if child_products:
        _assign_aggregate(model, ifc_beam, child_products)

    _process_container_voids(model, context, beam, ifc_beam, cache)
    _process_container_openings(model, context, beam, ifc_beam, cache)
    return ifc_beam


def _layered_children(elem) -> Optional[list]:
    """Effective geometry children for a layered container (v15.2), or None.

    When ``elem.layers`` is set and the element has exactly ONE Box/Extrude
    body child, that body is replaced by its per-layer slices (anonymous,
    ``material=layer.key`` — the existing aggregated-children emission then
    gives each slice its registry render + material for free); other
    children (openings etc.) pass through. Returns None when not layered or
    when slicing can't apply (zero/multiple bodies, degenerate contour) —
    validate_project_report errors on those shapes; here we warn and let the
    caller emit unsliced (loud, not silent)."""
    if getattr(elem, "layers", None) is None:
        return None
    from lite_step.compiler.layers import (
        layered_body_child, box_layer_slices, extrude_layer_slices)
    body = layered_body_child(elem)
    if body is None:
        logger.warning(
            "layers: %r needs exactly ONE Box/Extrude body child — emitted "
            "unsliced (validate_project reports this as an error)",
            elem.ifc_name or type(elem).__name__)
        return None
    if isinstance(body, Extrude):
        slices = extrude_layer_slices(elem, body)
        if slices is None:
            logger.warning("layers: %r contour is degenerate — emitted unsliced",
                           elem.ifc_name or type(elem).__name__)
            return None
    else:
        slices = box_layer_slices(elem, body)
    others = [c for c in elem._elements if c is not body]
    return slices + others


def _layer_direction_axis(elem) -> str:
    """LayerSetDirection for the usage chain, by ARCHETYPE.

    Walls (incl. Element with a wall ifc_class) are AXIS2 (through-thickness
    perpendicular to the wall axis); every other planar archetype is AXIS3
    (vertical).

    Prefer :func:`_layer_axis_from_slicing` wherever the real slicing axis is
    known — see its docstring for why the archetype answer is not always the
    true one.
    """
    ifc_class = getattr(elem, "ifc_class", None)
    if type(elem).__name__ == "Wall" or ifc_class in ("IfcWall", "IfcCurtainWall"):
        return "axis2"
    return "axis3"


#: World slicing axis -> IFC LayerSetDirection. Valid because our products carry
#: IDENTITY-ROTATION placements, so an element's local axes ARE the world axes.
#: If placements ever gain rotation this mapping has to compose with it.
_SLICE_AXIS_TO_IFC = {"x": "axis1", "y": "axis2", "z": "axis3"}


def _layer_thickness_scale(layer_set, min_v: float, max_v: float) -> float:
    """Factor ``layer_spans`` applies to the authored thicknesses, or 1.0.

    ``layers.layer_spans`` fits the buildup PROPORTIONALLY onto the body's
    actual extent (tolerated to 5% by validation), but the emitted
    ``IfcMaterialLayer.LayerThickness`` was always the NOMINAL value. Where the
    two disagree the chain describes a buildup we did not draw, and every band
    after the first drifts progressively — the same "first band right, later
    bands wrong" shape as the offset bug, and independent of it.

    Returns exactly ``1.0`` when the body matches its buildup (every corpus
    model, and the normal authoring case), so a file that was already honest
    stays byte-identical.
    """
    nominal = sum(m.thickness_mm for m in layer_set.layers) / 1000.0
    if nominal <= 0:
        return 1.0
    scale = abs(max_v - min_v) / nominal
    return 1.0 if abs(scale - 1.0) < 1e-9 else scale


def _layer_offset_from_reference_line(
        min_v: float, max_v: float, placement_v: float, outer_at_max: bool) -> float:
    """``OffsetFromReferenceLine`` for a sliced buildup, in metres.

    IFC measures a buildup from the product's ``ObjectPlacement`` along the
    through axis: it starts at the FIRST-listed layer's outer face and runs in
    ``DirectionSense``. So the offset is just where that face sits relative to
    the placement.

    Derived from the real extent and the real placement rather than assumed to
    be ``±T/2``: our products happen to be placed at their AABB centre today,
    but the moment one is not, a constant is silently wrong again — which is
    exactly the shape of the bug this replaces (a hardcoded ``0.0``, correct
    only for a product whose placement sits on its outer face, i.e. none of
    ours)..
    """
    first_face = max_v if outer_at_max else min_v
    return first_face - placement_v


#: IFC LayerSetDirection -> local axis index. Direct because our placements are
#: identity-rotation (same caveat as ``_SLICE_AXIS_TO_IFC``).
_IFC_AXIS_INDEX = {"axis1": 0, "axis2": 1, "axis3": 2}


def _layer_datum_for_container(elem, direction_axis: str,
                               outer_at_max: bool) -> Tuple[float, float]:
    """``(OffsetFromReferenceLine, thickness_scale)`` for a slab / roof /
    planar-Element buildup.

    Unlike a wall, these containers carry an IDENTITY placement at the world
    origin — the slices themselves hold world coordinates — so the reference
    plane IS the world origin and the offset is just the first layer's face in
    world coordinates.

    Falls back to ``0.0`` with a warning when the body extent cannot be
    recovered. That is the value, and the only honest answer without
    an extent; it is warned rather than swallowed because a silent 0.0 here
    reproduces exactly the bug being fixed.
    """
    from lite_step.compiler.frames import element_aabb
    from lite_step.compiler.layers import layered_body_child

    body = layered_body_child(elem)
    box = element_aabb(body) if body is not None else None
    if box is None:
        logger.warning(
            "layers: %r has no recoverable body extent — emitting "
            "OffsetFromReferenceLine=0.0, so a consumer will re-derive this "
            "buildup at the wrong datum",
            getattr(elem, "ifc_name", None) or type(elem).__name__)
        return 0.0, 1.0
    i = _IFC_AXIS_INDEX.get(direction_axis, 2)
    lo, hi = box[0][i], box[1][i]
    return (_layer_offset_from_reference_line(lo, hi, 0.0, outer_at_max),
            _layer_thickness_scale(elem.layers, lo, hi))


def _layer_axis_from_slicing(axis: str) -> str:
    """The LayerSetDirection matching the axis the compiler ACTUALLY slices on.

    Declaring ``AXIS2`` unconditionally while ``layer_axis_and_dir`` slices
    along whichever horizontal axis is thin makes the east/west leaves of a
    conventional ring slice along world X while telling every consumer they
    slice along local Y. Anything that walks a buildup from the usage chain
    (a plan view hatching per layer does exactly this) then read half the walls
    wrong, with no error anywhere..

    IFC's own intent is that a wall's local X runs ALONG the wall, which would
    make AXIS2 right by construction. That is the better long-term shape — it
    needs rotated ObjectPlacements — but until then the honest answer is the
    axis we really used.
    """
    return _SLICE_AXIS_TO_IFC.get(axis, "axis3")


def _create_slab(model, context, slab: Slab, cache: EntityCache):
    """
    Create IfcSlab as a container aggregating Box/Extrude geometry children.

    Slab is a CONTAINER (DSL v8): it holds Box/Extrude children (reversing
    the pre-v8 geometry-direct form). Children auto-color by their own type
    (or "slab") unless they carry an explicit color or registry material.
    Mirrors _create_roof.

    Predefined type is derived from the children (type/color do not live
    on the container): a "foundation"-typed child -> BASESLAB, else FLOOR.
    """
    assert run is not None
    assert np is not None

    child_types = {getattr(c, 'type', '') for c in slab.elements}
    predefined_type = "BASESLAB" if "foundation" in child_types else "FLOOR"
    ifc_slab = _create_ifc_entity(model, "IfcSlab", name=slab.ifc_name, predefined_type=predefined_type)

    # Identity placement at origin (children carry world coordinates).
    matrix = np.eye(4)
    _set_placement(model, ifc_slab, matrix, cache)

    # Layered slab (v15.2): the single body child is replaced by per-layer
    # slices; the buildup data rides the IfcMaterialLayerSetUsage chain.
    sliced = _layered_children(slab)
    if slab.layers is not None:
        from lite_step.compiler.layers import layer_outer_at_max
        _axis = _layer_direction_axis(slab)
        _outer_at_max = layer_outer_at_max(slab)
        _offset, _scale = _layer_datum_for_container(slab, _axis, _outer_at_max)
        cache.associate_layer_usage(
            ifc_slab, slab.layers, _axis, _offset, _scale,
            outer_at_max=_outer_at_max)

    child_products = []
    for child in (sliced if sliced is not None else slab.elements):
        if tx.is_prism(child):
            face = _create_solid(model, context, child, cache)
            if face:
                child_products.append(face)
                cache.record_child_product(slab, child, face)
                # Container-driven coloring: apply the child's own type color
                # (or "slab") unless it has an explicit color or registry material.
                if not child.color and _registry_material_selection(child) is None:
                    _apply_container_color(model, face, getattr(child, 'type', '') or "slab", cache)
            else:
                _warn_child_dropped("Slab", slab.ifc_name, child)

    child_products.extend(_emit_offtable_children(
        model, context, slab, (sliced if sliced is not None else slab.elements),
        cache, kind="Slab", handled=tx.is_prism))

    if child_products:
        _assign_aggregate(model, ifc_slab, child_products)

    _process_container_voids(model, context, slab, ifc_slab, cache)
    return ifc_slab



# =============================================================================
# PLANNING PRIMITIVES — IFC GENERATION
# =============================================================================


# ── Default marker colors ──
_MARKER_COLORS = {
    "boundary": "#f32ca5",   # pink
    "foundation": "#e8e8e8", # white
}
_MARKER_TOP_COLOR = "#111111"  # black measurement point


def _create_marker_peg(model, context, elem: ReferencePoint, cache: EntityCache):
    """
    Create IfcBuildingElementProxy with visible 3D peg mesh.

    Hexagonal prism (6-sided stake) with colored body and black top cap.
    Used for boundary corners and foundation footprint markers.
    """
    assert np is not None
    import math as _math

    # Peg dimensions (in meters — post-normalization)
    radius = 0.05    # 50mm radius
    height = 0.4     # 400mm tall
    cap_h = 0.02     # 20mm black top cap
    sides = 6

    loc = _dsl_to_ifc_point(elem.location.x, elem.location.y, elem.location.z)
    color_hex = elem.marker_color or _MARKER_COLORS.get(elem.point_type, "#f32ca5")

    # Generate hexagonal prism vertices.
    # The location point is the cardinal measurement point (black top center).
    # The peg mesh extends DOWNWARD from there.
    # y=0 is the top (location), y=-height is the bottom of the stake.
    verts = []
    for i in range(sides):
        angle = i * 2 * _math.pi / sides
        x = radius * _math.cos(angle)
        z = radius * _math.sin(angle)
        verts.append((x, -height, z))                  # bottom ring (stake tip)
    for i in range(sides):
        angle = i * 2 * _math.pi / sides
        x = radius * _math.cos(angle)
        z = radius * _math.sin(angle)
        verts.append((x, -cap_h, z))                   # body top ring (below cap)
    for i in range(sides):
        angle = i * 2 * _math.pi / sides
        x = radius * _math.cos(angle)
        z = radius * _math.sin(angle)
        verts.append((x, 0.0, z))                      # cap top ring (at location)
    verts.append((0.0, 0.0, 0.0))                      # center-top (cardinal point)
    verts.append((0.0, -height, 0.0))                  # center-bottom (stake tip)
    # indices: bottom=0..5, body_top=6..11, cap_top=12..17,
    #          center_top=18, center_bottom=19

    # Faces (0-based)
    faces = []
    # Bottom cap — fanned from a CENTRE vertex, mirroring the top cap below.
    # Fanning from ring vertex 0 over the full `range(sides)` is wrong: a fan
    # over an n-gon is n-2 triangles, not n, and the `% sides` wrap makes i=0
    # emit (0,1,0) and i=sides-1 emit (0,0,sides-1) — a repeated index, zero area,
    # not 2-manifold. Invisible in a render (nothing to see) and invisible to
    # ifcopenshell.validate (no schema rule is broken), which is how it
    # survived. Using a centre vertex rather than the shorter range(1, sides-1)
    # is what keeps both caps built the same way; building them differently
    # emitted two different pegs.
    center_bottom = 3 * sides + 1
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((j, i, center_bottom))  # reversed winding for bottom-facing
    # Body sides
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((i, j, j + sides))
        faces.append((i, j + sides, i + sides))
    # Cap sides (body_top to cap_top)
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((i + sides, j + sides, j + 2 * sides))
        faces.append((i + sides, j + 2 * sides, i + 2 * sides))
    # Top cap (fan from center)
    center_idx = 3 * sides
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((2 * sides + i, 2 * sides + j, center_idx))

    # Create IFC entity
    ifc_proxy = _create_ifc_entity(model, "IfcBuildingElementProxy", name=elem.ifc_name)
    ifc_proxy.ObjectType = "MarkerPeg"

    # Placement at point location
    matrix = np.array([
        [1, 0, 0, loc[0]],
        [0, 1, 0, loc[1]],
        [0, 0, 1, loc[2]],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    _set_placement(model, ifc_proxy, matrix, cache)

    # Geometry: IfcTriangulatedFaceSet
    coords = model.create_entity(
        "IfcCartesianPointList3D",
        CoordList=[list(v) for v in verts]
    )
    # Convert to 1-based indices for IFC
    ifc_faces = [[f[0] + 1, f[1] + 1, f[2] + 1] for f in faces]
    # Closed=True is a MEASURED claim: test_marker_peg.py checks the
    # emitted CoordIndex is 2-manifold with consistent winding — every
    # undirected edge used exactly twice, every directed edge exactly once —
    # rather than trusting the loops above. It also asserts the emitter emits
    # the SAME peg, which is what neither of them did before.
    face_set = model.create_entity(
        "IfcTriangulatedFaceSet",
        Coordinates=coords,
        Closed=True,
        CoordIndex=ifc_faces,
    )
    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="Tessellation",
        Items=[face_set]
    )
    ifc_proxy.Representation = model.create_entity(
        "IfcProductDefinitionShape", Representations=[shape_rep]
    )

    # Apply body color (hex → RGB tuple)
    h = color_hex.lstrip("#")
    rgb = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    _apply_surface_color(model, face_set, rgb, cache)

    # Property set
    props = [
        model.create_entity("IfcPropertySingleValue",
            Name="PointType",
            NominalValue=model.create_entity("IfcLabel", wrappedValue=elem.point_type)),
        model.create_entity("IfcPropertySingleValue",
            Name="MarkerColor",
            NominalValue=model.create_entity("IfcLabel", wrappedValue=color_hex)),
    ]
    pset = model.create_entity("IfcPropertySet",
        GlobalId=ifcopenshell.guid.new(), Name="Pset_MarkerPeg", HasProperties=props)
    model.create_entity("IfcRelDefinesByProperties",
        GlobalId=ifcopenshell.guid.new(), RelatedObjects=[ifc_proxy], RelatingPropertyDefinition=pset)

    return ifc_proxy


def _create_reference_point(model, context, elem: ReferencePoint, cache: EntityCache):
    """
    Create IfcAnnotation with ObjectType='ReferencePoint'.

    Representation: IfcGeometricSet containing IfcCartesianPoint.
    Properties: Pset_SurveyPoint with survey metadata.
    """
    assert np is not None

    ifc_annotation = _create_ifc_entity(model, "IfcAnnotation", name=elem.ifc_name)
    ifc_annotation.ObjectType = "ReferencePoint"

    # Placement at point location
    loc = _dsl_to_ifc_point(elem.location.x, elem.location.y, elem.location.z)
    matrix = np.array([
        [1, 0, 0, loc[0]],
        [0, 1, 0, loc[1]],
        [0, 0, 1, loc[2]],
        [0, 0, 0, 1]
    ], dtype=np.float64)
    _set_placement(model, ifc_annotation, matrix, cache)

    # Representation: IfcGeometricSet with a single point
    cart_pt = model.create_entity("IfcCartesianPoint", Coordinates=list(loc))
    geom_set = model.create_entity("IfcGeometricSet", Elements=[cart_pt])
    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Annotation",
        RepresentationType="GeometricSet",
        Items=[geom_set]
    )
    product_shape = model.create_entity("IfcProductDefinitionShape", Representations=[shape_rep])
    ifc_annotation.Representation = product_shape

    # Property set: Pset_SurveyPoint
    props = [
        model.create_entity("IfcPropertySingleValue",
            Name="PointType",
            NominalValue=model.create_entity("IfcLabel", wrappedValue=elem.point_type)),
    ]
    if elem.accuracy_mm is not None:
        props.append(model.create_entity("IfcPropertySingleValue",
            Name="HorizontalAccuracy_mm",
            NominalValue=model.create_entity("IfcInteger", wrappedValue=elem.accuracy_mm)))
    if elem.survey_method:
        props.append(model.create_entity("IfcPropertySingleValue",
            Name="SurveyMethod",
            NominalValue=model.create_entity("IfcLabel", wrappedValue=elem.survey_method)))
    if elem.crs:
        props.append(model.create_entity("IfcPropertySingleValue",
            Name="CRS",
            NominalValue=model.create_entity("IfcLabel", wrappedValue=elem.crs)))

    pset = model.create_entity("IfcPropertySet",
        GlobalId=ifcopenshell.guid.new(), Name="Pset_SurveyPoint", HasProperties=props)
    model.create_entity("IfcRelDefinesByProperties",
        GlobalId=ifcopenshell.guid.new(), RelatedObjects=[ifc_annotation], RelatingPropertyDefinition=pset)

    return ifc_annotation


def _create_guide_line(model, context, elem: GuideLine, cache: EntityCache):
    """
    Create IfcAnnotation with IfcPolyline representation for freeform guide lines.

    ObjectType set based on line_type: PropertyBoundary, SetbackLine, EasementLine, UtilityCorridor.
    """
    assert np is not None

    object_type_map = {
        "boundary": "PropertyBoundary",
        "setback": "SetbackLine",
        "easement": "EasementLine",
        "corridor": "UtilityCorridor",
    }
    obj_type = object_type_map.get(elem.line_type, "GuideLine")

    ifc_annotation = _create_ifc_entity(model, "IfcAnnotation", name=elem.label or elem.ifc_name)
    ifc_annotation.ObjectType = obj_type

    # Identity placement
    matrix = np.eye(4, dtype=np.float64)
    _set_placement(model, ifc_annotation, matrix, cache)

    # Create polyline from points
    ifc_points = []
    for pt in elem.points:
        loc = _dsl_to_ifc_point(pt.x, pt.y, pt.z)
        ifc_points.append(model.create_entity("IfcCartesianPoint", Coordinates=list(loc)))

    polyline = model.create_entity("IfcPolyline", Points=ifc_points)

    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Annotation",
        RepresentationType="Curve",
        Items=[polyline]
    )
    product_shape = model.create_entity("IfcProductDefinitionShape", Representations=[shape_rep])
    ifc_annotation.Representation = product_shape

    # Property set: Pset_GuideLine
    props = []
    if elem.source:
        props.append(model.create_entity("IfcPropertySingleValue",
            Name="Source",
            NominalValue=model.create_entity("IfcLabel", wrappedValue=elem.source)))
    if elem.constraint:
        props.append(model.create_entity("IfcPropertySingleValue",
            Name="Constraint",
            NominalValue=model.create_entity("IfcLabel", wrappedValue=elem.constraint)))
    if elem.buffer_width is not None:
        props.append(model.create_entity("IfcPropertySingleValue",
            Name="BufferWidth",
            NominalValue=model.create_entity("IfcLengthMeasure", wrappedValue=float(elem.buffer_width))))
    if props:
        pset = model.create_entity("IfcPropertySet",
            GlobalId=ifcopenshell.guid.new(), Name="Pset_GuideLine", HasProperties=props)
        model.create_entity("IfcRelDefinesByProperties",
            GlobalId=ifcopenshell.guid.new(), RelatedObjects=[ifc_annotation], RelatingPropertyDefinition=pset)

    return ifc_annotation


def _create_grid(model, context, grid: "grids.ResolvedGrid", cache: EntityCache):
    """Create the ONE ``IfcGrid`` a storey's grid lines resolve to.

    Takes the :class:`lite_step.ifc.grids.ResolvedGrid` — every decision about
    which axis is U and which is V, and every refusal, is already made there,
    so this function is pure emission and decides none of it.
    """
    assert np is not None

    def _axis_entity(axis: "grids.GridAxis"):
        ifc_points = [
            model.create_entity("IfcCartesianPoint",
                                Coordinates=list(_dsl_to_ifc_point(pt.x, pt.y, pt.z)))
            for pt in axis.element.points
        ]
        polyline = model.create_entity("IfcPolyline", Points=ifc_points)
        # SameSense=True: the curve is emitted in its AUTHORED direction. The
        # half-plane canonicalisation grids.py does is for classification only
        # and never touches the geometry.
        return model.create_entity("IfcGridAxis", AxisTag=axis.tag,
                                   AxisCurve=polyline, SameSense=True)

    u_axes = [_axis_entity(axis) for axis in grid.u_axes]
    v_axes = [_axis_entity(axis) for axis in grid.v_axes]

    # Name is None by policy (grids.GRID_NAME) — the grid corresponds to no
    # single DSL element, and a synthetic name would enter LITESTEP_META.
    ifc_grid = _create_ifc_entity(model, "IfcGrid", name=grids.GRID_NAME)
    ifc_grid.UAxes = u_axes
    ifc_grid.VAxes = v_axes
    if grid.predefined_type:
        ifc_grid.PredefinedType = grid.predefined_type

    # Identity placement
    matrix = np.eye(4, dtype=np.float64)
    _set_placement(model, ifc_grid, matrix, cache)

    return ifc_grid


def _create_space(model, context, space: Space, cache: EntityCache):
    """
    Create IfcSpace container with child Solid geometry.

    Unlike the aggregate-only containers (_create_roof et al.), the FIRST
    child Solid's representation is transferred onto the IfcSpace itself
    (so the space carries its own geometry); any additional child Solids
    are aggregated underneath.

    Child appearance follows the precedence chain documented on
    ``_apply_container_color``; the Space container auto-color is the
    70%-transparent light blue (173, 216, 230) rather than a palette key.
    """
    assert np is not None

    ifc_space = _create_ifc_entity(model, "IfcSpace", name=space.ifc_name)

    # Identity placement
    matrix = np.eye(4, dtype=np.float64)
    _set_placement(model, ifc_space, matrix, cache)

    # Process first child Solid as the space's own representation
    # Additional children are aggregated underneath
    children = space.elements
    child_products = []
    # The representation donor is the first PRISM child, not the first child:
    # any type can be anchored into a Space now, and a leading Sweep must not
    # silently take the slot the space's own body is meant to fill.
    donated = False
    for child in children:
        if tx.is_prism(child):
            face = _create_solid(model, context, child, cache)
            if face:
                # Appearance precedence (see _apply_container_color):
                # explicit color= and registry material render are applied
                # by _create_solid itself; the transparent-blue container
                # auto-color applies only when neither is present.
                auto_color = (
                    not child.color
                    and _registry_material_selection(child) is None
                )
                if not donated and face.Representation:
                    donated = True
                    # First child: transfer its representation to the IfcSpace directly
                    ifc_space.Representation = face.Representation
                    if auto_color:
                        solid_item = _get_solid_item(ifc_space)
                        if solid_item:
                            _apply_surface_color(model, solid_item, (173, 216, 230), cache, transparency=0.7)
                    # The donor proxy's ONLY purpose was to build that shape;
                    # it is not aggregated and not spatially contained, so
                    # leaving it in the model is an orphan product that
                    # viewers double-draw over the IfcSpace. Detach the shared
                    # shape (so removal doesn't take it — the IfcSpace holds it
                    # now) then drop the donor product.
                    face.Representation = None
                    # **The donor's PLACEMENT comes with the shape.**
                    # ``_create_box`` builds the solid centred on LOCAL (0,0,0)
                    # and puts the box's world centre in the placement, so the
                    # representation is only half the geometry. Dropping the
                    # placement (as this did) left the IfcSpace on the identity
                    # it was created with, and every space rendered at world
                    # origin instead of in its room — measured: a living room
                    # authored at x[-4000, 450] came out shifted +1775 mm in x
                    # and -1400 in z.
                    #
                    # Nothing catches that downstream: the file is valid, the
                    # space has a Representation, and it renders — just in the
                    # wrong place. Same shape as the IfcBuildingStorey.Elevation
                    # bug, where a passing validator proved nothing.
                    donor_placement = getattr(face, "ObjectPlacement", None)
                    if donor_placement is not None:
                        stale = ifc_space.ObjectPlacement
                        ifc_space.ObjectPlacement = donor_placement
                        # The identity placement the space was created with is
                        # now unused — but only drop it if nothing chained to
                        # it, or an aggregated child is left pointing at a
                        # deleted entity.
                        if (stale is not None and stale != donor_placement
                                and not stale.PlacesObject
                                and not stale.ReferencedByPlacements):
                            model.remove(stale)
                    model.remove(face)
                else:
                    child_products.append(face)
                    if auto_color:
                        solid_item = _get_solid_item(face)
                        if solid_item:
                            _apply_surface_color(model, solid_item, (173, 216, 230), cache, transparency=0.7)
            else:
                _warn_child_dropped("Space", space.ifc_name, child)

    child_products.extend(_emit_offtable_children(
        model, context, space, children, cache, kind="Space",
        handled=tx.is_prism))

    # Aggregate additional children under IfcSpace
    if child_products:
        _assign_aggregate(model, ifc_space, child_products)

    # Property set: Pset_SpacePlanning
    props = []
    if space.space_type:
        props.append(model.create_entity("IfcPropertySingleValue",
            Name="SpaceType",
            NominalValue=model.create_entity("IfcLabel", wrappedValue=space.space_type)))
    if props:
        pset = model.create_entity("IfcPropertySet",
            GlobalId=ifcopenshell.guid.new(), Name="Pset_SpacePlanning", HasProperties=props)
        model.create_entity("IfcRelDefinesByProperties",
            GlobalId=ifcopenshell.guid.new(), RelatedObjects=[ifc_space], RelatingPropertyDefinition=pset)

    return ifc_space


def _get_solid_item(product):
    """Extract the solid geometry item from a product's representation for styling."""
    if product.Representation:
        for rep in product.Representation.Representations:
            if rep.Items:
                return rep.Items[0]
    return None


def _parse_hex_color(value):
    """``"#RRGGBB"`` or ``"#RRGGBBAA"`` → ``((r, g, b), alpha)``, else None.

    The AA byte carries alpha (255 = opaque) — e.g. ``"#9E9E9E80"`` is
    50%-transparent ghost massing, the replacement for the retired
    ``mesh_type="building_ghost"`` shortcut."""
    if not (isinstance(value, str) and value.startswith("#") and len(value) in (7, 9)):
        return None
    try:
        rgb = tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
        alpha = int(value[7:9], 16) / 255.0 if len(value) == 9 else 1.0
    except ValueError:
        return None
    return rgb, alpha


def _mesh_face_set(model, elem: Mesh):
    """Tessellate one Mesh into an ``IfcTriangulatedFaceSet`` (0-based DSL
    faces → 1-based IFC). Shared by the standalone proxy path and the
    semantic-wrapper merged-representation path.

    ``Closed`` says whether the tessellation bounds a SOLID, and it is the
    only channel a consumer reads to tell a ground slab from a draped
    surface. We already know the answer — heightmap mode forces
    ``is_watertight`` True when it builds the closed prism (top + bottom + 4
    skirts), explicit mode takes the author's word — but until now the fact
    was published ONLY as ``LiteStep_MeshVolume.IsWatertight``, a
    proprietary Pset no IFC consumer looks at, while the schema attribute
    that means exactly the same thing went out as ``$``.

    Note this is deliberately NOT ``IfcTriangulatedIrregularNetwork``, whose
    name makes it the obvious candidate for terrain. A TIN represents a
    horizontal SURFACE, single-valued in z; our terrain is a closed prism
    because displacement subtracts foundations from it and manifold3d needs
    a solid. Emitting the prism as a TIN would claim "sample ground level
    here" and hand the caller a bottom plane or a skirt.
    """
    coords = [list(_dsl_to_ifc_point(v.x, v.y, v.z)) for v in elem.vertices]
    coord_list = model.create_entity("IfcCartesianPointList3D", CoordList=coords)
    ifc_faces = [[f[0] + 1, f[1] + 1, f[2] + 1] for f in elem.faces]
    return model.create_entity(
        "IfcTriangulatedFaceSet",
        Coordinates=coord_list,
        Closed=elem.is_watertight,
        CoordIndex=ifc_faces,
    )


def _style_mesh_item(model, product, face_set, elem: Mesh, cache: EntityCache,
                     terrain_style: bool = False, registry_render=None):
    """Appearance chain for one mesh face set (spec §6 /):

        own color= ("#RRGGBB"/"#RRGGBBAA") > own material hex > own registry
        grade (IfcMaterial + Psets on ``product``) > inherited wrapper
        registry render > terrain default (``terrain_style``).

    ``building_ghost``'s 50% alpha survives in the material-hex branch until
    the pack migrates to ``color="#RRGGBBAA"``; then the field dies.
    """
    # The SAME vocabulary a solid gets: a palette name here resolves rather
    # than falling through to the terrain default.
    resolved = resolve_color(getattr(elem, "color", None))
    if resolved is not None:
        rgb, transparency = resolved
        _apply_surface_color(model, face_set, rgb, cache, transparency=transparency)
        return
    if elem.material and isinstance(elem.material, str) and elem.material.startswith("#") and len(elem.material) >= 7:
        hex_color = elem.material.lstrip("#")
        rgb = (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16))
        # Ghost massing is ``color="#RRGGBBAA"`` now — handled by the
        # _parse_hex_color branch above, which reads the alpha byte. A
        # material hex is RGB only and therefore opaque.
        transparency = 0.0
        # Terrain gets higher transparency (legacy site-context naming)
        if getattr(elem, "name", None) == "site_terrain":
            transparency = 0.8
        _apply_surface_color(model, face_set, rgb, cache, transparency=transparency)
        return
    if _apply_registry_material(model, product, face_set, elem, cache):
        return
    if registry_render is not None:
        rgb_hex = registry_render.rgb.lstrip("#")
        rgb = (int(rgb_hex[0:2], 16), int(rgb_hex[2:4], 16), int(rgb_hex[4:6], 16))
        _apply_surface_color(model, face_set, rgb, cache,
                             transparency=1.0 - registry_render.alpha)
        return
    if terrain_style:
        # Terrain is OPAQUE ground — a translucent style let viewers see
        # straight through it. Muted green-grey, surface style 'Terrain'.
        # Triggered by the canonical semantic wrapper only
        # (Element(IfcGeographicElement, TERRAIN) parent).
        _apply_surface_color(model, face_set, (100, 120, 90), cache,
                             transparency=0.0, name="Terrain")


def _create_mesh(model, context, elem: Mesh, cache: EntityCache):
    """
    Create IfcBuildingElementProxy with IfcTriangulatedFaceSet — the
    STANDALONE mesh path (a bare mesh has no semantics, so a proxy is the
    honest class). A mesh inside a semantic ``Element`` wrapper does NOT come
    through here — its geometry merges into the wrapper's own representation
    (``_create_element_generic``), so no semantically-blank proxy rides along.
    """
    assert np is not None

    ifc_elem = _create_ifc_entity(model, "IfcBuildingElementProxy", name=elem.ifc_name)

    # Identity placement
    matrix = np.eye(4, dtype=np.float64)
    _set_placement(model, ifc_elem, matrix, cache)

    tri_face_set = _mesh_face_set(model, elem)

    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="Tessellation",
        Items=[tri_face_set]
    )
    product_shape = model.create_entity("IfcProductDefinitionShape", Representations=[shape_rep])
    ifc_elem.Representation = product_shape

    # Property set: LiteStep_MeshVolume
    props = [
        model.create_entity("IfcPropertySingleValue",
            Name="VertexCount",
            NominalValue=model.create_entity("IfcInteger", wrappedValue=len(elem.vertices))),
        model.create_entity("IfcPropertySingleValue",
            Name="FaceCount",
            NominalValue=model.create_entity("IfcInteger", wrappedValue=len(elem.faces))),
        model.create_entity("IfcPropertySingleValue",
            Name="IsWatertight",
            NominalValue=model.create_entity("IfcBoolean", wrappedValue=elem.is_watertight)),
    ]
    if elem.source_type:
        props.append(model.create_entity("IfcPropertySingleValue",
            Name="SourceType",
            NominalValue=model.create_entity("IfcLabel", wrappedValue=elem.source_type)))

    # LiteStep_MeshVolume (not Pset_MeshVolume): buildingSMART reserves the
    # ``Pset_`` prefix for standard property sets.
    pset = model.create_entity("IfcPropertySet",
        GlobalId=ifcopenshell.guid.new(), Name="LiteStep_MeshVolume", HasProperties=props)
    model.create_entity("IfcRelDefinesByProperties",
        GlobalId=ifcopenshell.guid.new(), RelatedObjects=[ifc_elem], RelatingPropertyDefinition=pset)

    # Appearance chain (color= > material hex > own grade > terrain default)
    # via the shared item styler. No wrapper inheritance on the standalone
    # path — a bare mesh has no wrapper.
    _style_mesh_item(model, ifc_elem, tri_face_set, elem, cache)

    return ifc_elem


def _create_roof(model, context, roof: Roof, cache: EntityCache):
    """
    Create IfcRoof container with aggregated child face elements.

    The Roof container:
    - Creates IfcRoof as the parent element
    - Processes child Solid elements as roof faces and Sweep children
      as rafters/purlins
    - Aggregates children under the IfcRoof

    Child appearance follows the precedence chain documented on
    ``_apply_container_color`` (container auto-color key: "roof").
    """
    assert run is not None
    assert np is not None

    # Create IfcRoof container
    ifc_roof = _create_ifc_entity(model, "IfcRoof", name=roof.ifc_name)

    # Identity placement at origin
    matrix = np.eye(4)
    _set_placement(model, ifc_roof, matrix, cache)

    # Layered roof (v15.2): single body child → per-layer slices + usage chain.
    sliced = _layered_children(roof)
    if roof.layers is not None:
        from lite_step.compiler.layers import layer_outer_at_max
        _axis = _layer_direction_axis(roof)
        _outer_at_max = layer_outer_at_max(roof)
        _offset, _scale = _layer_datum_for_container(roof, _axis, _outer_at_max)
        cache.associate_layer_usage(
            ifc_roof, roof.layers, _axis, _offset, _scale,
            outer_at_max=_outer_at_max)

    # Process child elements as roof faces (spec: Solid, Sweep)
    child_products = []
    for child in (sliced if sliced is not None else roof.elements):
        if tx.is_prism(child):
            # Create the solid geometry
            face = _create_solid(model, context, child, cache)
            if face:
                child_products.append(face)
                cache.record_child_product(roof, child, face)
                # Container-driven coloring: apply "roof" color unless the
                # child has an explicit color or a registry material render
                if not child.color and _registry_material_selection(child) is None:
                    _apply_container_color(model, face, "roof", cache)
            else:
                _warn_child_dropped("Roof", roof.ifc_name, child)
        elif isinstance(child, Sweep):
            face = _create_sweep(model, context, child, cache)
            if face:
                child_products.append(face)
                cache.record_child_product(roof, child, face)
                if not child.color and _registry_material_selection(child) is None:
                    _apply_container_color(model, face, "roof", cache)
            else:
                _warn_child_dropped("Roof", roof.ifc_name, child)
    child_products.extend(_emit_offtable_children(
        model, context, roof, (sliced if sliced is not None else roof.elements),
        cache, kind="Roof",
        handled=lambda c: tx.is_prism(c) or isinstance(c, Sweep)))

    # Aggregate children under IfcRoof
    if child_products:
        _assign_aggregate(model, ifc_roof, child_products)

    _process_container_voids(model, context, roof, ifc_roof, cache)
    return ifc_roof


def _model_to_string(model) -> str:
    """
    Serialize ifcopenshell model to STEP string.

    Uses a temp file since ifcopenshell.file.to_string() may not be available.
    IfcOpenShell's C++ serializer emits raw full-precision doubles, so the
    result is passed through ``normalize_step_floats`` —
    bit-exact re-encoding to shortest STEP reals, geometry unchanged.
    """
    from lite_step.ifc.step_writer import normalize_step_floats

    # Try direct serialization first (newer ifcopenshell versions)
    if hasattr(model, 'to_string'):
        return normalize_step_floats(model.to_string())

    # Fallback to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ifc', delete=False) as f:
        temp_path = f.name

    try:
        model.write(temp_path)
        with open(temp_path, 'r', encoding='utf-8') as f:
            return normalize_step_floats(f.read())
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# =============================================================================
# ELEMENT TYPE COLOR MAPPINGS
# =============================================================================

# Color mappings for element types (RGB 0-255)
# The palette, the transparency table and the ``color=`` vocabulary live in
# ``lite_step.ifc.colors`` so the emitter reads one decision — re-exported
# here because callers (and tests) have imported these names from the
# generator since before the split.
from lite_step.ifc.colors import (  # noqa: E402
    ELEMENT_TYPE_COLORS,
    TRANSPARENT_TYPES,
    resolve_color,
)



def _apply_surface_color(model, solid, color: tuple, cache, transparency: float = 0.0,
                         name: Optional[str] = None):
    """
    Apply IfcSurfaceStyle with color (and optional transparency) to a solid geometry.

    Uses EntityCache to share IfcSurfaceStyle across elements with the same color.
    Only the IfcStyledItem is created per-solid.

    IFC4: IfcSurfaceStyle assigned directly to IfcStyledItem.Styles
    (IfcPresentationStyleAssignment is deprecated).

    Args:
        model: ifcopenshell model
        solid: The IfcRepresentationItem (solid geometry)
        color: RGB tuple with integers 0-255, e.g., (128, 200, 50)
        cache: EntityCache for style chain deduplication
        transparency: 0.0 = fully opaque, 1.0 = fully transparent
        name: optional IfcSurfaceStyle Name override (e.g. "Terrain" for opaque ground)
    """
    r = color[0] / 255.0
    g = color[1] / 255.0
    b = color[2] / 255.0

    # Get cached IfcSurfaceStyle (IFC4 — no IfcPresentationStyleAssignment wrapper)
    surface_style = cache.get_or_create_style_chain(r, g, b, transparency, name=name)

    # IfcStyledItem is per-solid, always create new
    model.create_entity(
        "IfcStyledItem",
        Item=solid,
        Styles=[surface_style],
        Name=None
    )


def _apply_typed_color(model, solid, color_key: str, cache):
    """Apply a ``color=`` value — a palette name OR a hex string — to a solid.

    The vocabulary is ``colors.resolve_color`` and covers both spellings; an
    unreadable value is refused by ``executor._validate_color`` long before
    here, so silence at this point means "no colour stated"."""
    resolved = resolve_color(color_key)
    if resolved is None:
        return
    rgb, transparency = resolved
    _apply_surface_color(model, solid, rgb, cache, transparency)


# =============================================================================
# ELEMENT-LEVEL PROPERTY SETS (v1.5 — props={})
# =============================================================================

class _ModelShapeAdapter:
    """:class:`~lite_step.ifc.product_types.ShapeAdapter` over an ifcopenshell model.

    Answers only; every decision lives in ``product_types``. ``referrers`` is
    ``model.get_inverse``, which is the whole reason the purge can tell a
    cached profile (still referenced by a surviving solid) from a body nothing
    points at any more.
    """

    def __init__(self, model):
        self._model = model

    def children(self, product):
        found = []
        for rel in (getattr(product, "IsDecomposedBy", None) or []):
            found.extend(rel.RelatedObjects or [])
        return found

    def representations(self, product):
        shape = getattr(product, "Representation", None)
        if shape is None:
            return []
        return [(r.RepresentationIdentifier, r.RepresentationType, r)
                for r in (shape.Representations or [])]

    def name(self, product):
        return getattr(product, "Name", None)

    def is_entity(self, value) -> bool:
        """An ADDRESSABLE entity — a `#123=` line, not a wrapped scalar.

        ``ifcopenshell`` returns a defined TYPE (``IfcLineIndex``,
        ``IfcArcIndex``, every measure type) as an ``entity_instance`` too, so
        ``isinstance`` alone answers True for things that are STEP scalars —
        exactly what this predicate must exclude. They are told apart by
        ``id()``: a real entity has its STEP line number, a wrapped scalar has
        ``0``.

        Both consequences were live bugs. A multi-point
        ``Bar`` emits an ``IfcIndexedPolyCurve`` whose ``Segments`` are
        ``IfcLineIndex``, and the purge walk then:

        * keyed every one of them at ``0`` — so N distinct segments collided
          into ONE candidate slot, and the fixpoint reasoned about a set that
          did not describe the model; and
        * called ``get_inverse`` on them, which raises
          ``Only entities with ids are supported for get_inverse``.

        The raise is the visible half and the collision is the dangerous one:
        it is a silent mis-purge of the kind ``key``'s own docstring exists to
        warn about.
        """
        return (isinstance(value, ifcopenshell.entity_instance)
                and value.id() != 0)

    def key(self, handle) -> int:
        """The STEP line number — the only stable identity here.

        ``ifcopenshell`` hands back a NEW Python wrapper on every attribute
        read and every ``get_inverse``, so ``id(wrapper)`` identifies the read,
        not the entity.
        """
        return handle.id()

    def entity_type(self, handle) -> str:
        return handle.is_a()

    def entity_attrs(self, handle):
        return [handle[i] for i in range(len(handle))]

    def referrers(self, handle):
        return list(self._model.get_inverse(handle))


def _identity_transformation_operator(model, cache: EntityCache):
    """The identity ``IfcCartesianTransformationOperator3D``.

    Every optional attribute is ``$``: ``Axis1``/``Axis2``/``Axis3`` unset
    means the derived axes are the coordinate axes, and ``Scale`` unset derives
    to 1.0. ``LocalOrigin`` is MANDATORY, so it is stated, at the origin.

    One instance serves every mapped item, and that is not merely tidy: the
    post-pass deduplicator merges only the types in
    ``deduplicator.DEDUP_SAFE_TYPES``, which this is NOT one of. Minting one
    per mapped item would put N identical operator lines in the file and leave
    them there.
    """
    for existing in model.by_type("IfcCartesianTransformationOperator3D"):
        # Reuse only one that IS the identity. ``by_type(...)[0]`` would be
        # enough today — nothing else in this compiler emits one — but the day
        # something does, a non-identity operator would be picked up here and
        # move every occurrence of every Product, silently.
        if (existing.Axis1 is None and existing.Axis2 is None
                and existing.Axis3 is None and existing.Scale is None
                and tuple(existing.LocalOrigin.Coordinates) == (0.0, 0.0, 0.0)):
            return existing
    return model.create_entity(
        "IfcCartesianTransformationOperator3D",
        LocalOrigin=cache.get_or_create_point([0.0, 0.0, 0.0]),
    )


def _share_product_geometry(model, group, type_entity, occurrences, cache: EntityCache) -> None:
    """WS-B step 2 — one body per Product instead of one per occurrence.

    See the "Geometry sharing" header in :mod:`lite_step.ifc.product_types` for
    the placement decision and the concept pages behind it. In one line: the
    occurrence keeps its ``ObjectPlacement``, the map and the mapped item are
    both identity, so nothing moves.
    """
    from lite_step.ifc.product_types import (
        MAPPED_REPRESENTATION_TYPE, plan_shared_geometry, purgeable_after_sharing,
    )

    adapter = _ModelShapeAdapter(model)
    plan = plan_shared_geometry(group, occurrences, adapter)
    if not plan:
        return

    operator = _identity_transformation_operator(model, cache)
    origin = cache.get_or_create_point([0.0, 0.0, 0.0])
    maps, orphaned = [], []

    for shared in plan:
        source = shared.source
        rep_map = model.create_entity(
            "IfcRepresentationMap",
            MappingOrigin=model.create_entity("IfcAxis2Placement3D", Location=origin),
            MappedRepresentation=source.rep,
        )
        maps.append(rep_map)

        for slot in shared.slots:
            mapped = model.create_entity(
                "IfcShapeRepresentation",
                ContextOfItems=source.rep.ContextOfItems,
                RepresentationIdentifier=source.identifier,
                RepresentationType=MAPPED_REPRESENTATION_TYPE,
                Items=[model.create_entity(
                    "IfcMappedItem", MappingSource=rep_map, MappingTarget=operator)],
            )
            shape = slot.owner.Representation
            representations = list(shape.Representations)
            representations[slot.index] = mapped
            shape.Representations = representations

        orphaned.extend(slot.rep for slot in shared.followers)

    # RECORDED, not removed. `file.remove` nulls out every
    # inverse reference to the entity it deletes, which costs a scan of the
    # file — so removing here was O(removals x file size): measured on
    # unitized-curtain-wall (149,370 entities), 60,189 removals at 6.3 ms =
    # 381.8 s, about 72% of that model's compile.
    #
    # `model.batch()` / `unbatch()` looks like the answer and is NOT: measured,
    # it moves the entire cost from `file_remove` (381.8 s) to `file_unbatch`
    # (384.4 s). It defers the bookkeeping rather than avoiding it. Do not
    # re-try it without new evidence.
    #
    # The ids are dropped in ONE pass by the deduplicator, which has the file
    # parsed into a dict already — see `deduplicator._drop_purged`, including
    # why deleting outright is equivalent to `file.remove` here (the purge set
    # is reference-closed) and the assertion that keeps it so.
    #
    # These entities therefore stay in the in-memory model for the rest of
    # generation. That is sound because they are unreachable: the followers'
    # `IfcProductDefinitionShape`s were rewired to the mapped representations
    # above, and nothing else refers to them. No later pass walks
    # representations by anything but a product's `Representation`.
    cache.record_purged(dead.id() for dead in
                        purgeable_after_sharing(orphaned, adapter))

    type_entity.RepresentationMaps = maps


def _emit_product_types(model, proj, cache: EntityCache) -> None:
    """Emit one ``Ifc<Class>Type`` + one ``IfcRelDefinesByType`` per Product,
    and share the occurrences' geometry through it.

    Joins occurrence → product on ``elem.ifc_name``, the same identity join
    :func:`_attach_element_psets`, the LITESTEP_META manifest and the
    patch-mode differ use. The policy (which class gets which type entity,
    which attributes that entity carries, which occurrences belong to which
    key, which representations may be shared and when two of them are the same
    geometry) lives in :mod:`lite_step.ifc.product_types`. This function only
    mints entities.
    """
    from lite_step.ifc.product_types import (
        collect_product_groups, missing_occurrence_error,
    )

    groups = collect_product_groups(proj)
    if not groups:
        return

    assert ifcopenshell is not None
    by_name: dict = {}
    for product in model.by_type("IfcProduct"):
        if product.Name and product.Name not in by_name:
            by_name[product.Name] = product

    for group in groups:
        occurrences, missing = [], []
        for canonical in group.occurrences:
            found = by_name.get(canonical)
            if found is None:
                missing.append(canonical)
            else:
                occurrences.append(found)
        if missing:
            raise missing_occurrence_error(group, missing)

        attrs = {"GlobalId": ifcopenshell.guid.new(), "Name": group.key}
        # Mandatory PredefinedType (and Door/Window's second enum) come from
        # the shared tail; the optional trailing attributes are left unset.
        attrs.update({n: v for n, v in group.tail if v is not None})
        type_entity = model.create_entity(group.type_entity, **attrs)
        model.create_entity(
            "IfcRelDefinesByType",
            GlobalId=ifcopenshell.guid.new(),
            RelatedObjects=occurrences,
            RelatingType=type_entity,
        )
        # WS-B step 2 — RepresentationMaps on the type, MappedRepresentation
        # on every occurrence. Runs AFTER the type exists because the maps hang
        # off it, and after IfcRelDefinesByType because a Product whose
        # occurrences cannot all be resolved must fail on the count, which is
        # the older and more specific error.
        _share_product_geometry(model, group, type_entity, occurrences, cache)


def _emit_groupings(model, proj, cache: EntityCache) -> None:
    """Emit one ``IfcRelAggregates`` / ``IfcZone`` per grouping statement.

    Every decision — resolution, the derived class, the refusals, the
    placement rule and the missing-member message — lives in
    :mod:`lite_step.ifc.groupings`. This function only mints entities.

    Joins member → product on ``product.Name`` (the stamped canonical), the
    same identity join the manifest, the patch differ and
    :func:`_emit_product_types` use.
    """
    from lite_step.ifc.groupings import (
        AGGREGATE, collect_groupings, missing_member_error, parent_container,
    )

    groups = collect_groupings(proj)
    if not groups:
        return

    assert ifcopenshell is not None
    assert np is not None

    by_name: dict = {}
    for product in model.by_type("IfcProduct"):
        if product.Name and product.Name not in by_name:
            by_name[product.Name] = product

    storeys = model.by_type("IfcBuildingStorey")
    sites = model.by_type("IfcSite")

    for group in groups:
        members, missing = [], []
        for canonical in group.members:
            found = by_name.get(canonical)
            if found is None:
                missing.append(canonical)
            else:
                members.append(found)
        if missing:
            raise missing_member_error(group, missing)

        if group.kind != AGGREGATE:
            zone = model.create_entity(
                "IfcZone", GlobalId=ifcopenshell.guid.new(), Name=group.name)
            model.create_entity(
                "IfcRelAssignsToGroup",
                GlobalId=ifcopenshell.guid.new(),
                RelatedObjects=members,
                RelatingGroup=zone,
            )
            continue

        container = parent_container(proj, group)
        structure = _grouping_structure(model, group, container, storeys, sites)

        parent = model.create_entity(
            group.ifc_class, GlobalId=ifcopenshell.guid.new(), Name=group.name)
        # No Representation, by rule: geometry lives on the members only, or
        # quantities double-count and a viewer draws the facade twice.
        _set_placement(model, parent, np.eye(4), cache)
        parent.ObjectPlacement.PlacementRelTo = structure.ObjectPlacement
        _rechain_members_to(model, parent, members, cache)
        model.create_entity(
            "IfcRelAggregates",
            GlobalId=ifcopenshell.guid.new(),
            RelatingObject=parent,
            RelatedObjects=members,
        )
        # IFC 4.3 Spatial Containment: the WHOLE carries the
        # containment, the PARTS have ContainedInStructure NIL. Members are
        # un-contained FIRST so a storey relation left empty by that removal
        # can be deleted before the parent is added to (possibly) the same one.
        _uncontain(model, members)
        _contain_in(model, structure, parent)


def _emit_space_boundaries(model, proj) -> None:
    """Emit one ``IfcRelSpaceBoundary1stLevel`` per inferred adjacency.

    Every decision — which elements are candidates, what counts as contact,
    both enum values, the entity type and the ``"1stLevel"`` naming — lives in
    :mod:`lite_step.ifc.space_boundaries`. This function only mints entities.

    Joins both products on ``product.Name`` (the stamped canonical), the same
    identity join the manifest, the patch differ, :func:`_emit_groupings` and
    :func:`_emit_product_types` use — and CHECKS the IFC type of each, because
    ``RelatingSpace`` is an ``IfcSpaceBoundarySelect`` and
    ``RelatedBuildingElement`` is an ``IfcElement``: a canonical name resolving
    to the wrong kind of entity would build a schema-invalid relation that
    ifcopenshell writes out without complaint.
    """
    from lite_step.ifc.space_boundaries import (
        BOUNDARY_ENTITY, collect_boundaries, missing_product_error,
        report_lines, wrong_type_error,
    )

    boundaries = collect_boundaries(proj)
    if not boundaries:
        return

    assert ifcopenshell is not None

    by_name: dict = {}
    for product in model.by_type("IfcProduct"):
        if product.Name and product.Name not in by_name:
            by_name[product.Name] = product

    minted: list = []
    for boundary in boundaries:
        space = by_name.get(boundary.space)
        element = by_name.get(boundary.element)
        if space is None:
            raise missing_product_error(boundary, "RelatingSpace",
                                        boundary.space)
        if element is None:
            raise missing_product_error(boundary, "RelatedBuildingElement",
                                        boundary.element)
        if not space.is_a("IfcSpace"):
            raise wrong_type_error(boundary, "RelatingSpace", boundary.space,
                                   "IfcSpace", space.is_a())
        if not element.is_a("IfcElement"):
            raise wrong_type_error(boundary, "RelatedBuildingElement",
                                   boundary.element, "IfcElement",
                                   element.is_a())
        minted.append(model.create_entity(
            BOUNDARY_ENTITY,
            GlobalId=ifcopenshell.guid.new(),
            Name=boundary.name,
            Description=boundary.description,
            RelatingSpace=space,
            RelatedBuildingElement=element,
            # ConnectionGeometry stays NULL — OPTIONAL in IFC4X3_ADD2, and the
            # schema says an omitted one describes the boundary LOGICALLY.
            ConnectionGeometry=None,
            PhysicalOrVirtualBoundary=boundary.physical_or_virtual,
            InternalOrExternalBoundary=boundary.internal_or_external,
            ParentBoundary=(minted[boundary.parent_index]
                            if boundary.parent_index is not None else None),
        ))

    for line in report_lines(boundaries):
        print(line)


def _grouping_structure(model, group, container, storeys, sites):
    """The ``IfcBuildingStorey`` / ``IfcSite`` entity ``container`` names.

    Looked up by ``emitted_index`` — the emitter skips empty storeys, so the
    DSL index and the file index diverge as soon as a model has one — and
    then CHECKED against the canonical name. Containing a facade one floor off
    moves no geometry and raises nothing, so the cheap assertion is the only
    thing standing between that and a silent wrong answer.
    """
    from lite_step.ifc.groupings import GroupingError

    pool = storeys if container.kind == "storey" else sites
    if container.emitted_index >= len(pool):
        raise GroupingError(
            f"aggregate {group.name!r}: its members stand in {container.kind} "
            f"#{container.emitted_index} ({container.name!r}), but only "
            f"{len(pool)} were emitted. The parent would be contained "
            f"nowhere.")
    structure = pool[container.emitted_index]
    if container.name is not None and structure.Name != container.name:
        raise GroupingError(
            f"aggregate {group.name!r}: expected to be contained in "
            f"{container.name!r} but {container.kind} #{container.emitted_index} "
            f"in the file is {structure.Name!r}. The emitted spatial order no "
            f"longer matches the DSL order, so the facade would be filed under "
            f"the wrong storey with no geometry moved and nothing raised.")
    return structure


def _uncontain(model, products) -> None:
    """Drop ``products`` from every ``IfcRelContainedInSpatialStructure``.

    IFC 4.3: a decomposition part's ``ContainedInStructure`` is NIL. A
    relation emptied by the removal is DELETED rather than left with an empty
    ``RelatedElements`` — the attribute's cardinality is ``SET [1:?]``, so an
    empty one is a schema violation that most readers accept silently.
    """
    for product in products:
        for rel in list(getattr(product, "ContainedInStructure", None) or []):
            remaining = [e for e in rel.RelatedElements if e != product]
            if remaining:
                rel.RelatedElements = remaining
            else:
                model.remove(rel)


def _contain_in(model, structure, product) -> None:
    """Add ``product`` to ``structure``'s containment relation, making one if
    the structure has none (a storey whose only element was the member)."""
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        if rel.RelatingStructure == structure:
            rel.RelatedElements = list(rel.RelatedElements) + [product]
            return
    model.create_entity(
        "IfcRelContainedInSpatialStructure",
        GlobalId=ifcopenshell.guid.new(),
        RelatedElements=[product],
        RelatingStructure=structure,
    )


def _rechain_members_to(model, parent, members, cache: EntityCache) -> None:
    """Re-root each member's ``ObjectPlacement`` on the aggregate ``parent``.

    Constraint 1 of roadmap §3, and the measured shape: a world-coordinate
    child with an UNCHAINED placement inside an ``IfcRelAggregates`` whose
    ``RelatingObject`` has no ``Representation`` tessellated at ~1/87 of its
    true volume — silently. Chaining removes that configuration, and it is
    also what IFC asks for (a decomposed element's placement is relative to
    the whole, not to its spatial container).

    The rewrite is exact by construction: the new ``RelativePlacement`` is
    ``inv(parent_world) @ member_world``, so the resolved world matrix is
    unchanged to the bit. The ``IfcLocalPlacement`` ENTITY is mutated in
    place rather than replaced — a member's own children (window frame
    members,-adjacent) chain to that entity and must keep following it.

    Spatial containment is handled separately, by :func:`_uncontain` — under
    IFC 4.3 the member's ``ContainedInStructure`` becomes NIL and the parent
    carries the containment instead.
    """
    import ifcopenshell.util.placement as _plc

    parent_world = _plc.get_local_placement(parent.ObjectPlacement)
    inverse = np.linalg.inv(parent_world)
    for member in members:
        placement = getattr(member, "ObjectPlacement", None)
        if placement is None:
            # Authored coordinates ARE the local frame — same reading
            # ``_apply_dsl_placement`` takes for a product without one.
            _set_placement(model, member, np.eye(4), cache)
            placement = member.ObjectPlacement
        member_world = _plc.get_local_placement(placement)
        placement.RelativePlacement = _axis2_placement_from_matrix(
            model, inverse @ member_world)
        placement.PlacementRelTo = parent.ObjectPlacement


def _attach_element_psets(model, proj, cache: EntityCache) -> None:
    """Emit ``props={}`` as IfcPropertySet + IfcRelDefinesByProperties.

    One IfcPropertySet per top-level props key (spec §SIGNATURES). Runs as a
    post-pass once every product exists, joining element → product on the
    emitted IFC Name (``elem.ifc_name``) — the same identity join the
    LITESTEP_META manifest and the patch-mode differ use. Elements whose
    geometry is consumed into a parent product (e.g. a Wall body Solid)
    attach their props to the nearest ancestor that emitted a product —
    the same attribution PR-A uses for a body Solid's registry material.

    Boolean-operand geometry (``_cuts``/``_adds``/``_fills``) never emits
    (``validate_project_report`` warns about props there). Any other
    unattachable props log a loud warning rather than dropping silently.
    """
    by_name: dict = {}
    for product in model.by_type("IfcProduct"):
        if product.Name and product.Name not in by_name:
            by_name[product.Name] = product

    for storey in proj.storeys:
        for top in storey.elements:
            _attach_psets_for_tree(model, cache, by_name, top, None)
    # Site groundwork (Element wrappers / meshes under proj.sites) can
    # carry props too — the storeys-only walk dropped them silently (the
    #/walk-coverage class).
    for site in (getattr(proj, "sites", None) or []):
        _attach_psets_for_tree(model, cache, by_name, site, None)


def _attach_psets_for_tree(model, cache: EntityCache, by_name: dict,
                           elem, ancestor_product) -> None:
    """Depth-first props emission for ``elem`` and its identity children."""
    assert ifcopenshell is not None

    product = by_name.get(elem.ifc_name) or ancestor_product
    elem_props = getattr(elem, "props", None) or {}
    if elem_props:
        if product is None:
            logger.warning(
                "props on %s %r dropped — no IFC product was emitted for it "
                "or any ancestor",
                type(elem).__name__, elem.ifc_name,
            )
        else:
            for pset_name, pset_props in elem_props.items():
                if not isinstance(pset_props, dict):
                    # validate_project_report rejects this shape at compile;
                    # guard here for direct generate_ifc callers.
                    logger.warning(
                        "props[%r] on %s %r skipped — expected a dict of "
                        "property values, got %s",
                        pset_name, type(elem).__name__, elem.ifc_name,
                        type(pset_props).__name__,
                    )
                    continue
                prop_entities = [
                    model.create_entity(
                        "IfcPropertySingleValue",
                        Name=prop_name,
                        NominalValue=cache._wrap_pset_value(value),
                    )
                    for prop_name, value in pset_props.items()
                ]
                if not prop_entities:
                    continue
                pset = model.create_entity(
                    "IfcPropertySet",
                    GlobalId=ifcopenshell.guid.new(),
                    Name=pset_name,
                    HasProperties=prop_entities,
                )
                model.create_entity(
                    "IfcRelDefinesByProperties",
                    GlobalId=ifcopenshell.guid.new(),
                    RelatedObjects=[product],
                    RelatingPropertyDefinition=pset,
                )

    for child in (getattr(elem, "_elements", None) or []):
        _attach_psets_for_tree(model, cache, by_name, child, product)
    for child in (getattr(elem, "_openings", None) or []):
        _attach_psets_for_tree(model, cache, by_name, child, product)
    # _cuts/_adds/_fills operands are consumed into the parent's shape and
    # never become products — validate_project_report warns about props
    # there; nothing to emit.


# =============================================================================
# REGISTRY MATERIALS (v1.5)
# =============================================================================

def _registry_material_selection(elem):
    """Resolve an element's material= to a registry Material selection.

    Returns None for the sketch stage and every legacy channel (legacy
    vocabulary strings, "Void", hex colors, the Mesh hex-color field) so
    those paths stay byte-for-byte identical to pre-v1.5 output.
    """
    from lite_step.materials import element_registry_material
    return element_registry_material(elem)


def _apply_registry_material(model, product, solid_item, elem, cache: EntityCache) -> bool:
    """Emit IfcMaterial + IfcRelAssociatesMaterial (+ registry render style).

    Registry keys get material Psets attached (via EntityCache) and the
    registry render color emitted as an IfcSurfaceStyle — including alpha
    for the glass keys (transparency = 1 - alpha). Unknown keys get a plain
    named IfcMaterial only (open vocabulary; validate_project warns).

    Appearance precedence (spec §MATERIALS): explicit ``color=`` > material
    render > container auto-color. The style is therefore skipped when the
    element sets ``color=``.

    Returns:
        True when a surface style was applied (callers skip their palette
        color path in that case), False otherwise.
    """
    mat = _registry_material_selection(elem)
    if mat is None:
        return False

    from lite_step.materials import registry_definition
    mdef = registry_definition(mat.key)

    ifc_material = cache.get_or_create_ifc_material(mat.key, mdef)
    cache.associate_material(product, ifc_material)

    if mdef is None or solid_item is None:
        return False
    if getattr(elem, "color", None):
        return False  # explicit color= overrides the material render

    rgb_hex = mdef.render.rgb.lstrip("#")
    rgb = (int(rgb_hex[0:2], 16), int(rgb_hex[2:4], 16), int(rgb_hex[4:6], 16))
    transparency = 1.0 - mdef.render.alpha
    _apply_surface_color(model, solid_item, rgb, cache, transparency)
    return True


def _style_layer_item(model, item, layer_material, body, cache: EntityCache) -> None:
    """Style ONE layered-wall body item (v15.2).

    Precedence per the one appearance chain: explicit body ``color=`` wins
    for EVERY slice; else the layer material's registry render (rgb + alpha);
    else the body type / container default "wall" palette color.
    """
    if body.color:
        # Pass the authored value THROUGH: filtering it against
        # ELEMENT_TYPE_COLORS would render every hex colour on a layered wall
        # as "wall".
        _apply_typed_color(model, item, body.color, cache)
        return
    from lite_step.materials import registry_definition
    mdef = registry_definition(layer_material.key)
    if mdef is not None:
        rgb_hex = mdef.render.rgb.lstrip("#")
        rgb = (int(rgb_hex[0:2], 16), int(rgb_hex[2:4], 16), int(rgb_hex[4:6], 16))
        _apply_surface_color(model, item, rgb, cache, 1.0 - mdef.render.alpha)
        return
    color_key = body.type if body.type in ELEMENT_TYPE_COLORS else "wall"
    _apply_typed_color(model, item, color_key, cache)


def _material_thickness_m(elem) -> Optional[float]:
    """Thickness in meters carried by a registry Material (thickness_mm), or None.

    Material dimensions stay in mm (the frozen Material is never normalized);
    the generator receives an already-normalized (meters) proj, so the
    conversion happens here.
    """
    mat = _registry_material_selection(elem)
    if mat is not None and mat.thickness_mm is not None:
        return mat.thickness_mm / 1000.0
    return None


def _apply_container_color(model, product, container_type: str, cache):
    """
    Apply color to a product based on its container type (Site, etc.).

    CANONICAL appearance-precedence statement (spec §6 — one chain, no
    exceptions): explicit ``color=`` on element > material ``render`` >
    container auto-color. This helper is the last link of that chain —
    callers gate on ``not child.color and _registry_material_selection(child)
    is None`` before invoking it. The single deliberate carve-out: container
    auto-color never applies to Bar children (reinforcement keeps its own
    appearance; see the Bar branch in ``_create_beam``).

    Container-driven coloring: the container determines the color,
    not the child element's type field.

    Args:
        model: ifcopenshell model
        product: The IFC product (e.g., IfcBuildingElementProxy)
        container_type: Container semantic type ("site", etc.)
        cache: EntityCache for style chain deduplication
    """
    if container_type not in ELEMENT_TYPE_COLORS:
        return
    color = ELEMENT_TYPE_COLORS[container_type]
    if product.Representation:
        # Try "Body" first, then fall back to first representation with items
        target_rep = None
        for rep in product.Representation.Representations:
            if rep.RepresentationIdentifier == "Body":
                target_rep = rep
                break
        if target_rep is None:
            for rep in product.Representation.Representations:
                if rep.Items:
                    target_rep = rep
                    break
        if target_rep and target_rep.Items:
            _apply_surface_color(model, target_rep.Items[0], color, cache)





def _create_hole_solid_local(model, hole: 'BimElement', parent_min: tuple, parent_center: tuple, cache: EntityCache, parent_max: tuple = None, is_cut: bool = True):
    """
    Create a boolean-operand solid positioned in local coordinates
    (relative to the parent's center).

    Dispatches on the operand type (box Solid / rotated cutter / Pipe / Bar /
    Revolve), then recursively applies the operand's OWN ``_cuts``/``_adds`` in
    the same parent frame — so merged voids
    (``door_void.adds(arch_turn); wall.cuts(door_void)``) compile to a
    nested IfcBooleanResult operand.

    Returns None (with a loud warning) for unsupported operand shapes.

    ``parent_max`` defaults to the mirror of ``parent_min`` about
    ``parent_center`` (the historical centered-box assumption); world-frame
    callers (new v1.5 primitives, parent_center = origin) pass their real
    AABB max explicitly.
    """
    base = _create_operand_base_solid(model, hole, parent_min, parent_center, cache,
                                      parent_max=parent_max, is_cut=is_cut)
    if base is None:
        return None

    # Operand clips (spec-displacement-miter-clip, option b): a `.clip()`ed
    # element used as a boolean operand removes only its CLIPPED volume — the
    # half-spaces apply to the operand exactly as they would on a receiver.
    # This is what makes miter ∘ displacement compose: the inferred carve of a
    # mitered leaf subtracts the leaf's KEPT half only, so the neighbouring
    # leaf keeps its own half and the corner stays flush (no 45° hole).
    # Applied to the BASE before sub-cuts/adds — IfcBooleanClippingResult's
    # FirstOperand must be a swept solid or another clipping result (WR1), and
    # (X∩H)−C ≡ (X−C)∩H so the order is volume-neutral. The one shape that
    # can't be clip-wrapped is a half-space base (large cutters): unbounded ∩
    # unbounded — keep the loud warning for that corner.
    clips = getattr(hole, "_clips", None)
    if clips:
        if base.is_a("IfcHalfSpaceSolid"):
            logger.warning(
                "operand %r carries .clip() half-spaces but its base emitted "
                "as an (unbounded) IfcHalfSpaceSolid cutter — clips skipped; "
                "use a bounded operand to combine with .clip()",
                getattr(hole, "ifc_name", None) or type(hole).__name__)
        else:
            base = _apply_clips(model, base, clips, center=parent_center)

    return _apply_boolean_chain_ios(
        model, base,
        build=lambda sub, is_cut=True: _create_hole_solid_local(
            model, sub, parent_min, parent_center, cache, parent_max=parent_max, is_cut=is_cut),
        cuts=(getattr(hole, '_cuts', None) or []),
        adds=(getattr(hole, '_adds', None) or []))


def _create_operand_base_solid(model, hole: 'BimElement', parent_min: tuple, parent_center: tuple, cache: EntityCache, parent_max: tuple = None, is_cut: bool = True):
    """
    Create the base solid for a boolean operand, positioned in local
    coordinates (centered at parent origin).

    The hole is defined in DSL coordinates relative to the parent's min corner.
    We transform it to be relative to the parent's center (for rotation).

    Supports operand rotations - if the operand has rotations, the solid is created
    with IfcAxis2Placement3D that includes the rotation directly.

    v1.5 (WS1 PR-E): Revolve operands become IfcRevolvedAreaSolid, Pipe/Bar
    operands become IfcSweptDiskSolid — all positioned in the same
    parent-center-relative frame.

    IMPORTANT: Returns IfcExtrudedAreaSolid (or IfcHalfSpaceSolid for large cutters),
    NOT IfcMappedItem, because IfcBooleanResult requires solid operands.

    Args:
        model: ifcopenshell model
        hole: BimElement model with start/end coordinates in DSL space (may have rotations)
        parent_min: (x, y, z) of parent's minimum corner in IFC coordinates
        parent_center: (x, y, z) of parent's center in IFC coordinates
        cache: EntityCache for deduplication

    Returns:
        A solid operand positioned relative to parent center, or None (with
        a warning) if the operand shape is unsupported.
    """
    import math

    # v1.5 primitives as operands (WS1 PR-E). Their geometry is authored in
    # world coords; shift by -parent_center into the parent's local frame.
    offset = (-parent_center[0], -parent_center[1], -parent_center[2])
    if isinstance(hole, Revolve):
        return _create_revolved_solid(model, hole, cache, offset=offset)
    if isinstance(hole, (Pipe, Bar)):
        # Boolean-safe cylinders, NOT IfcSweptDiskSolid — the geometry
        # kernel mis-tessellates boolean trees with swept-disk operands
        # (see _create_disk_solid_boolean_safe).
        radius = hole.radius if isinstance(hole, Pipe) else hole.diameter / 2.0
        fillet = hole.fillet_radius if isinstance(hole, Pipe) else hole.bend_radius
        return _create_disk_solid_boolean_safe(model, hole.path, radius, fillet,
                                               cache, offset=offset)

    if isinstance(hole, Sweep) and getattr(hole, 'path', None):
        # Path-form Sweep operand (a profile member as a carver): build the swept
        # solid in the parent's local frame by offsetting the path centerline,
        # then return the single solid (a multi-item composite section unions
        # into one). The section (Material(profile_mm=)) comes off the
        # element, same as the standalone member. Without this a member could
        # never be a `.add()`-with-carve operand.
        ox, oy, oz = offset
        local_path = [p.model_copy(update={'x': p.x + ox, 'y': p.y + oy, 'z': p.z + oz})
                      for p in hole.path]
        items, _hb = _profile_path_items(model, local_path, hole, cache)
        if not items:
            return None
        return _balanced_union_ios(model, items)

    # Guard: contour-mode elements don't have start/end
    if not hasattr(hole, 'start') or hole.start is None or not hasattr(hole, 'end') or hole.end is None:
        logger.warning(
            "unsupported boolean operand %s %r skipped — operands must be "
            "box-mode Solids (start/end), Revolve, Pipe, or Bar",
            type(hole).__name__, getattr(hole, 'ifc_name', None) or '<anonymous>',
        )
        return None

    # Transform hole coordinates to IFC
    start_ifc = _dsl_to_ifc_point(hole.start.x, hole.start.y, hole.start.z)
    end_ifc = _dsl_to_ifc_point(hole.end.x, hole.end.y, hole.end.z)

    # Normalize corners
    cut_min = [
        min(start_ifc[0], end_ifc[0]),
        min(start_ifc[1], end_ifc[1]),
        min(start_ifc[2], end_ifc[2])
    ]
    cut_max = [
        max(start_ifc[0], end_ifc[0]),
        max(start_ifc[1], end_ifc[1]),
        max(start_ifc[2], end_ifc[2])
    ]

    # Extend cut faces that are coplanar with the parent to avoid boolean failures (cuts only)
    if is_cut:
        if parent_max is None:
            # Historical centered-box assumption: max mirrors min about center.
            parent_max = tuple(2.0 * c - m for c, m in zip(parent_center, parent_min))
        _extend_cut_for_boolean(cut_min, cut_max, parent_min, parent_max)

    min_pt = tuple(cut_min)
    max_pt = tuple(cut_max)

    # Calculate hole dimensions
    width = snap_to_precision(max_pt[0] - min_pt[0])
    depth = snap_to_precision(max_pt[1] - min_pt[1])
    height = snap_to_precision(max_pt[2] - min_pt[2])

    if width < 0.001 or depth < 0.001 or height < 0.001:
        return None

    # Calculate hole center relative to parent center
    hole_center = (
        (min_pt[0] + max_pt[0]) / 2.0 - parent_center[0],
        (min_pt[1] + max_pt[1]) / 2.0 - parent_center[1],
        (min_pt[2] + max_pt[2]) / 2.0 - parent_center[2]
    )

    # Check if hole has rotations
    rotations = getattr(hole, 'rotations', []) or []

    if not rotations:
        # No rotation - simple placement
        # Create profile (cached)
        profile = cache.get_or_create_rect_profile(x_dim=width, y_dim=depth)

        origin = cache.get_or_create_point(
            [hole_center[0], hole_center[1], hole_center[2] - height / 2.0]
        )
        placement = model.create_entity("IfcAxis2Placement3D", Location=origin)

        extrusion_dir = cache.get_z_up_direction()

        solid = model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile,
            Position=placement,
            ExtrudedDirection=extrusion_dir,
            Depth=height
        )
        return solid

    # Has rotations - use IfcHalfSpaceSolid for "sky void" type cuts
    # This is much simpler and more reliable for boolean operations
    assert np is not None  # numpy available when IFC_AVAILABLE is True

    # Build rotation matrix from sequential rotations
    # DSL and IFC both use Z-up — direct axis mapping (identity)
    rotation_matrix = np.eye(3, dtype=np.float64)
    for axis, angle_centideg in rotations:
        ifc_axis = axis.lower()
        angle_sign = 1.0

        angle_rad = math.radians(angle_centideg / 100.0) * angle_sign
        rot = _axis_rotation_matrix_3x3(ifc_axis, angle_rad)
        rotation_matrix = rotation_matrix @ rot

    # For sky void cutters, use IfcHalfSpaceSolid which is a semi-infinite solid
    # bounded by a plane. This is much more reliable for boolean operations.
    #
    # The plane is defined by:
    # - A point on the plane (the bottom center of what would be the box)
    # - A normal direction (perpendicular to the plane surface)

    # The plane normal is the Z-axis after rotation (pointing "up" in rotated space)
    plane_normal = np.array([
        rotation_matrix[0, 2],
        rotation_matrix[1, 2],
        rotation_matrix[2, 2]
    ])

    # Determine cutting direction based on hole type:
    # - "sky_cutter" or "sky" holes: cut ABOVE the plane (remove sky)
    # - "dirt_top_cutter" or "bottom" holes: cut BELOW the plane (remove ground)
    #
    # For grass cap to work as a thin layer between two planes:
    # - sky_void cuts above → keeps below
    # - dirt_top_void cuts below → keeps above
    # Result: the intersection (thin layer between planes)
    hole_id = getattr(hole, 'id', '') or ''
    cut_below = 'dirt_top' in hole_id or 'bottom' in hole_id or 'lower' in hole_id

    # Position on the plane: start from hole_center, move down by half height in rotated Z direction
    plane_point = np.array([
        hole_center[0] - plane_normal[0] * height / 2.0,
        hole_center[1] - plane_normal[1] * height / 2.0,
        hole_center[2] - plane_normal[2] * height / 2.0
    ])

    # Create the plane surface
    plane_location = model.create_entity(
        "IfcCartesianPoint",
        Coordinates=(float(plane_point[0]), float(plane_point[1]), float(plane_point[2]))
    )

    # For cutting below, we flip the normal direction so the half-space
    # extends downward instead of upward
    if cut_below:
        final_normal = -plane_normal
    else:
        final_normal = plane_normal

    # Normal direction - points into the half-space to remove
    plane_axis = model.create_entity(
        "IfcDirection",
        DirectionRatios=(float(final_normal[0]), float(final_normal[1]), float(final_normal[2]))
    )

    # Reference direction (X-axis of plane) - use first column of rotation matrix
    plane_ref = model.create_entity(
        "IfcDirection",
        DirectionRatios=(
            float(rotation_matrix[0, 0]),
            float(rotation_matrix[1, 0]),
            float(rotation_matrix[2, 0])
        )
    )

    plane_placement = model.create_entity(
        "IfcAxis2Placement3D",
        Location=plane_location,
        Axis=plane_axis,
        RefDirection=plane_ref
    )

    plane_surface = model.create_entity(
        "IfcPlane",
        Position=plane_placement
    )

    # Create IfcHalfSpaceSolid
    # AgreementFlag=True means the solid is in the direction of the surface normal
    # With the normal pointing in the direction we want to remove,
    # DIFFERENCE will subtract that half-space from the base solid
    half_space = model.create_entity(
        "IfcHalfSpaceSolid",
        BaseSurface=plane_surface,
        AgreementFlag=True
    )

    return half_space


def _create_body_representation(model, context, solid):
    """
    Create an IfcShapeRepresentation for a solid item.

    Args:
        model: ifcopenshell model
        context: Body representation context
        solid: IfcRepresentationItem (solid geometry)

    Returns:
        IfcProductDefinitionShape with the solid
    """
    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid" if solid.is_a("IfcExtrudedAreaSolid") else "CSG",
        Items=[solid]
    )

    product_shape = model.create_entity(
        "IfcProductDefinitionShape",
        Representations=[shape_rep]
    )

    return product_shape


#: Curved/swept operand types → their exact product builder. Prisms (Box/Extrude)
#: are NOT here — they keep the pure-manifold3d fast path in ``_occupant_mesh``.
def _curved_solid_builders():
    """Curved/swept solid type-name → its exact product builder. The *membership*
    is the taxonomy's CURVED kind (single source of truth); only the per-type
    builder mapping lives here (the assert catches drift)."""
    builders = {
        "Revolve": _create_revolve,
        "Sweep": _create_sweep,
        "Pipe": _create_pipe,
        "Bar": _create_bar,
    }
    assert set(builders) == {t.__name__ for t in tx.TESSELLATION_CURVED}, \
        "curved builders out of sync with taxonomy.TESSELLATION_CURVED"
    return builders


def _authoring_scale(elem) -> float:
    """Largest absolute coordinate in the element's authored geometry.

    Used to make mesher deflection SIZE-RELATIVE: this tessellator is called
    with raw authoring mm at query time and with normalized meters at carve
    time, and the kernel's deflection is expressed in model units — a fixed
    value that is right for one domain is 1000x off in the other (a donut arc in the mm domain
    tessellates to 443k verts / ~50s per call)."""
    m = 0.0
    for field in ("profile", "path", "vertices"):
        for p in (getattr(elem, field, None) or []):
            for a in ("x", "y", "z"):
                v = getattr(p, a, None)
                if v is not None:
                    m = max(m, abs(float(v)))
    for field in ("start", "end"):
        p = getattr(elem, field, None)
        if p is not None:
            for a in ("x", "y", "z"):
                m = max(m, abs(float(getattr(p, a))))
    return m


def tessellate_solid_to_mesh(elem):
    """Tessellate a curved/swept solid operand (Revolve/Sweep/Pipe/Bar) into
    ``(verts Nx3 float64, faces Mx3 int64)`` in **authoring-space** via the
    ifcopenshell geometry kernel.

    Reuses the element's own EXACT product builder (``_create_revolve`` etc.) — the
    same geometry the element renders — rather than re-deriving revolve/sweep/fillet
    tessellation in numpy, so all four curved/swept types are covered at once. The
    operand's ``placement=`` is NOT applied here (the caller resolves it). Returns
    ``None`` if the type is unsupported or the kernel yields no geometry (the caller
    fails loudly).
    """
    import ifcopenshell.geom

    builder = _curved_solid_builders().get(type(elem).__name__)
    if builder is None:
        return None

    model = ifcopenshell.file(schema=IFC_OUTPUT_SCHEMA)
    cache = EntityCache(model)
    _create_ifc_entity(model, "IfcProject", name="_tessellation")
    run("unit.assign_unit", model, length={"is_metric": True, "raw": "METRE"})
    # This scratch model is never serialised, so this call is drift-
    # prevention only — but leaving one call site un-canonicalised is the
    # shape that lets two spellings of one rule diverge.
    _canonical_unit_order(model)
    ctx = run("context.add_context", model, context_type="Model")
    body = run("context.add_context", model, context_type="Model",
               context_identifier="Body", target_view="MODEL_VIEW", parent=ctx)

    try:
        product = builder(model, body, elem, cache)
    except Exception:
        return None
    if product is None or getattr(product, "Representation", None) is None:
        return None

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    # Size-relative resolution: chord error <= ~0.1% of the element's extent
    # regardless of authoring units (see _authoring_scale). Angular deflection
    # caps curved-segment chords at ~23 deg — matching the density the render
    # path gets from the kernel defaults at meter scale.
    scale = _authoring_scale(elem)
    if scale > 0:
        settings.set("mesher-linear-deflection", max(scale * 1e-3, 1e-4))
        settings.set("mesher-angular-deflection", 0.4)
    try:
        shape = ifcopenshell.geom.create_shape(settings, product)
    except Exception:
        return None
    verts = np.asarray(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(shape.geometry.faces, dtype=np.int64).reshape(-1, 3)
    if verts.size == 0 or faces.size == 0:
        return None
    if not _mesh_is_where_it_should_be(elem, verts):
        return None
    # WELD before returning. OCC evaluates every face
    # independently, so a seam shared by two faces comes back as two vertex
    # entries a few float64 bits apart — geometrically one point, topologically
    # two. The surface is closed; the INDEX TOPOLOGY is not, and every mesh
    # boolean downstream reads the topology. ifcopenshell's own weld-vertices
    # pass is already on and does not reach 1e-14 m. Unwelded, manifold3d
    # answers Error.NotManifold without raising and then poisons every boolean
    # it touches — a bridge deck that never intersected the terrain deleted the
    # whole terrain. Welding here rather than in the carve so the query methods
    # (world_aabb / raycast) read the same repaired topology.
    before = len(verts)
    verts, faces = weld_coincident_vertices(verts, faces)
    # Same root cause, THIRD defect, and it must run HERE — after the weld,
    # before the orient. A swept solid built from abutting
    # segments emits the shared internal cap once per segment. Before welding
    # those are distinct vertex indices and nothing looks wrong; the weld
    # merges them into two triangles on the SAME three corners, so every edge
    # of that cap is traversed by four faces. That is a NON-MANIFOLD edge, not
    # a hole — and ``orient_faces_consistently`` says in its own docstring
    # that non-manifold input has no consistent orientation to find, so
    # running it first propagates winding across a wall that should not exist
    # and returns a mesh that is both non-manifold AND mis-wound. Measured:
    # dedup alone still answers Error.NotManifold; dedup THEN re-orient gives
    # Error.NoError.
    faces_before_dedup = len(faces)
    verts, faces = drop_coincident_faces(verts, faces)
    if len(faces) != faces_before_dedup:
        logger.info(
            "%s '%s': dropped %d coincident interior-wall face(s) (%d → %d)",
            type(elem).__name__,
            getattr(elem, "ifc_name", None) or getattr(elem, "id", "?"),
            faces_before_dedup - len(faces), faces_before_dedup, len(faces),
        )
    # Same root cause, second defect: OCC evaluates each FACE independently,
    # so nothing makes one face's winding agree with the next one's. A Bar
    # comes back topologically perfect and still fails, because its low cap is
    # wound backwards relative to the sides — manifold3d answers
    # Error.NotManifold WITHOUT raising and poisons every boolean downstream,
    # the signature exactly. Repair here, beside the weld, so the query
    # methods read the same repaired topology.
    verts, faces = orient_faces_consistently(verts, faces)
    if len(verts) != before:
        logger.info(
            "%s '%s': welded %d coincident tessellation vertices (%d → %d); "
            "boundary edges %d",
            type(elem).__name__,
            getattr(elem, "ifc_name", None) or getattr(elem, "id", "?"),
            before - len(verts), before, len(verts),
            mesh_boundary_edge_count(faces),
        )
    return verts, faces


def _mesh_is_where_it_should_be(elem, verts) -> bool:
    """Does the tessellation stand where the AUTHORED geometry stands?

    A position check on the kernel's answer, because a mesh in the wrong place
    is worse than no mesh: this function feeds the displacement carve, so a
    misplaced mesh cuts a hole somewhere the author never asked for, and
    nothing downstream can tell.

    Not hypothetical. ``Pipe`` and ``Bar`` come back
    POINT-REFLECTED THROUGH THE ORIGIN here (a pipe authored at x=1000..3000
    tessellates to x=-3000..-1000), while ``Sweep`` and ``Revolve`` through the
    same helper are correct and the SHIPPED IFC is correct for all four. The
    entity this function writes is verifiably right — the directrix carries the
    authored points — so the fault is in how the kernel evaluates it in this
    single-element model. Rather than guess at OCC, refuse the answer.

    Nothing caught it for months because the only existing guard on this
    function asserts vertex COUNT (`len(verts) < 20000`, the 443k-vert donut
    regression) and a reflected mesh has exactly the right count.

    Returning ``None`` is the established "could not tessellate" signal:
    ``displacement`` turns it into a loud ``DisplacementError`` and the query
    methods into a ``ValueError``. Loud failure over silent degradation.
    """
    try:
        from lite_step.compiler.extent import MM_PER_METER, element_extent
        # MM_PER_METER: the generator runs on the normalized (metres) project,
        # so a Material's mm facts have to be converted into model units. At
        # the authoring default the pad would come out 1000x too large and
        # this guard would accept any reflection it exists to refuse.
        authored = element_extent(elem, MM_PER_METER)
    except Exception:
        return True                     # cannot check — do not block the caller
    if authored is None:
        return True

    (ax0, ay0, az0), (ax1, ay1, az1) = authored
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    # Scale-relative slack: the extent is a BOUND for these types (isotropic
    # radius pads), so the mesh may sit well inside it, but its CENTRE cannot
    # wander outside the authored box.
    span = max(ax1 - ax0, ay1 - ay0, az1 - az0, 1e-9)
    tol = max(span * 0.05, 1e-6)
    centre = (lo + hi) / 2.0
    inside = (
        ax0 - tol <= centre[0] <= ax1 + tol
        and ay0 - tol <= centre[1] <= ay1 + tol
        and az0 - tol <= centre[2] <= az1 + tol
    )
    if inside:
        return True

    authored_centre = ((ax0 + ax1) / 2.0, (ay0 + ay1) / 2.0, (az0 + az1) / 2.0)
    reflected = all(
        abs(c + a) <= tol for c, a in zip(centre, authored_centre)
    )
    logger.warning(
        "%s '%s': tessellation landed outside its authored extent — "
        "mesh centre %s vs authored %s%s. Refusing the mesh.",
        type(elem).__name__, getattr(elem, "ifc_name", None) or getattr(elem, "id", "?"),
        tuple(round(float(c), 3) for c in centre),
        tuple(round(float(c), 3) for c in authored_centre),
        " (POINT-REFLECTED THROUGH THE ORIGIN — the signature)"
        if reflected else "",
    )
    return False


# =============================================================================
# BOOLEAN OPERATIONS
# =============================================================================

def _apply_contour_boolean_operations(model, base_solid, cuts: list, adds: list,
                                      centroid: tuple, x_axis, y_axis, z_axis,
                                      thickness_m: float, cache: EntityCache,
                                      intersects: list = None,
                                      clips: list = None):
    """
    Apply boolean operations to a contour-extruded solid, with cuts/adds
    transformed to the contour's local coords.

    The contour's local coordinate system:
    - Origin: centroid of the contour
    - X-axis: computed from normal using Up-Preferred strategy
    - Y-axis: cross(Z, X) with Y-negation for winding orientation
    - Z-axis: normal (extrusion direction)

    Args:
        model: ifcopenshell model
        base_solid: The base IfcExtrudedAreaSolid for the contour element
        cuts: List of box-mode Solid operands to subtract
        adds: List of box-mode Solid operands to add
        centroid: (x, y, z) centroid in DSL coordinates
        x_axis: numpy array, local X axis in DSL coords
        y_axis: numpy array, local Y axis in DSL coords
        z_axis: numpy array, local Z axis (normal) in DSL coords
        thickness_m: Contour extrusion thickness in meters
        cache: EntityCache for deduplication

    Returns:
        Final solid (IfcBooleanResult or original)
    """
    assert np is not None

    result = _apply_boolean_chain_ios(
        model, base_solid,
        build=lambda op: _create_contour_cut_solid(
            model, op, centroid, x_axis, y_axis, z_axis, cache),
        cuts=cuts, adds=adds, intersects=intersects)

    # .clip() half-spaces — projected into the same contour-local frame.
    result = _apply_clips(model, result, clips or [],
                          center=centroid, axes=(x_axis, y_axis, z_axis))

    return result


def _create_contour_cut_solid(model, operand, centroid: tuple,
                              x_axis, y_axis, z_axis, cache: EntityCache):
    """
    Create a boolean-operand box solid in the contour's local coordinate system.

    Transforms a box-mode Solid operand (start/end corners) from global DSL
    coordinates to the contour element's local coordinate system for use in
    boolean operations.

    Args:
        model: ifcopenshell model
        operand: box-mode Solid operand, coordinates in meters (after normalization)
        centroid: (x, y, z) contour centroid in DSL coordinates
        x_axis, y_axis, z_axis: numpy arrays defining the contour's local axes in DSL
        cache: EntityCache for deduplication

    Returns:
        IfcExtrudedAreaSolid in the contour's local coords, or None if degenerate
    """
    assert np is not None

    # Same condition, same answer as the box-mode path (`_create_cut_solid`'s
    # guard ~400 lines up): an operand without start/end is not a box-mode
    # Solid and cannot be read as one. That path warns and returns None, which
    # `apply_boolean_chain` treats as "skip this operand"; this one read
    # `operand.start.x` unconditionally and died with
    # `AttributeError: 'Extrude' object has no attribute 'start'`.
    #
    # Reached whenever two contour-mode solids end up overlapping and
    # displacement infers a carve between them — and an anchored child cannot
    # dodge it the usual way, because `.anchor()` and `placement=` are mutually
    # exclusive by construction, so the "give it an identity placement" escape
    # hatch is unavailable. Found by the 5-link anchor chain in
    # scenario_b2_real_anchor_chain.py.
    if (getattr(operand, "start", None) is None
            or getattr(operand, "end", None) is None):
        logger.warning(
            "unsupported boolean operand %s %r skipped — operands must be "
            "box-mode Solids (start/end), Revolve, Pipe, or Bar",
            type(operand).__name__,
            getattr(operand, 'ifc_name', None) or '<anonymous>',
        )
        return None

    # Get operand corners in DSL space (already in meters)
    start_dsl = np.array([operand.start.x, operand.start.y, operand.start.z])
    end_dsl = np.array([operand.end.x, operand.end.y, operand.end.z])

    # Normalize corners in global space first
    min_dsl = np.minimum(start_dsl, end_dsl)
    max_dsl = np.maximum(start_dsl, end_dsl)

    # Transform corners to the contour's local coordinate system
    centroid_arr = np.array(centroid)

    def to_local(pt_dsl):
        """Transform a point from global DSL to the contour's local coords."""
        v = pt_dsl - centroid_arr
        local_x = float(np.dot(v, x_axis))
        # +y_axis, kept in lockstep with the contour projection above so the
        # cut solid lands in the same (un-mirrored) local frame as the profile.
        local_y = float(np.dot(v, y_axis))
        local_z = float(np.dot(v, z_axis))
        return (local_x, local_y, local_z)

    min_local = to_local(min_dsl)
    max_local = to_local(max_dsl)

    # Re-normalize in local space (transformation may swap min/max)
    final_min = (
        min(min_local[0], max_local[0]),
        min(min_local[1], max_local[1]),
        min(min_local[2], max_local[2])
    )
    final_max = (
        max(min_local[0], max_local[0]),
        max(min_local[1], max_local[1]),
        max(min_local[2], max_local[2])
    )

    # Calculate dimensions in local space
    width = snap_to_precision(final_max[0] - final_min[0])
    depth = snap_to_precision(final_max[1] - final_min[1])
    height = snap_to_precision(final_max[2] - final_min[2])

    if width < 0.001 or depth < 0.001 or height < 0.001:
        return None

    # Create solid at local position, offset by half dimensions to account for centered profile
    # IfcRectangleProfileDef centers the rectangle at its position
    position = cache.get_or_create_point(
        [final_min[0] + width / 2.0, final_min[1] + depth / 2.0, final_min[2]]
    )

    placement = model.create_entity(
        "IfcAxis2Placement3D",
        Location=position
    )

    profile = cache.get_or_create_rect_profile(x_dim=width, y_dim=depth)

    extrusion_dir = cache.get_z_up_direction()

    solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=placement,
        ExtrudedDirection=extrusion_dir,
        Depth=height
    )

    return solid


# =============================================================================
# NEW PRIMITIVE CREATION FUNCTIONS
# =============================================================================

def _create_solid(model, context, elem, cache: EntityCache):
    """Dispatch a Box or Extrude primitive to its geometry handler."""
    if isinstance(elem, Box):
        return _create_box(model, context, elem, cache)
    elif isinstance(elem, Extrude):
        return _create_extrude(model, context, elem, cache)
    return None


def _create_box(model, context, elem: 'Box', cache: EntityCache):
    """
    Create IfcBuildingElementProxy with box geometry for a Box primitive.

    Supports:
    - Sequential rotations around the box center
    - Boolean _cuts (DIFFERENCE) and _adds (UNION) via PrivateAttr
    - type-based automatic coloring

    DSL convention (Z-up):
    - start/end define opposite corners of base box
    - rotations: List of (axis, angle_centidegrees) tuples
    - _cuts: List of Solid elements for boolean cuts
    - _adds: List of Solid elements for boolean adds
    - type: Determines color ("site", "wall", "roof", etc.)
    """
    import math
    assert run is not None
    assert np is not None

    # Transform DSL coordinates to IFC coordinates (both Z-up, identity)
    start_ifc = _dsl_to_ifc_point(elem.start.x, elem.start.y, elem.start.z)
    end_ifc = _dsl_to_ifc_point(elem.end.x, elem.end.y, elem.end.z)

    # Normalize corners (ensure min < max on all axes)
    min_pt = (
        min(start_ifc[0], end_ifc[0]),
        min(start_ifc[1], end_ifc[1]),
        min(start_ifc[2], end_ifc[2])
    )
    max_pt = (
        max(start_ifc[0], end_ifc[0]),
        max(start_ifc[1], end_ifc[1]),
        max(start_ifc[2], end_ifc[2])
    )

    # Calculate dimensions
    width = snap_to_precision(max_pt[0] - min_pt[0])   # X dimension
    depth = snap_to_precision(max_pt[1] - min_pt[1])   # Y dimension (forward in IFC)
    height = snap_to_precision(max_pt[2] - min_pt[2])  # Z dimension (up in IFC)

    if width < 0.001 or depth < 0.001 or height < 0.001:
        return None

    # Calculate center point (pivot for rotations)
    center = (
        (min_pt[0] + max_pt[0]) / 2.0,
        (min_pt[1] + max_pt[1]) / 2.0,
        (min_pt[2] + max_pt[2]) / 2.0
    )

    element_type = elem.type or ''

    # Create base solid centered at origin (for rotation around center)
    profile = cache.get_or_create_rect_profile(
        x_dim=width,
        y_dim=depth
    )

    # Position the extrusion so the box is centered at origin
    z_offset = -height / 2.0
    origin = cache.get_or_create_point([0.0, 0.0, z_offset])
    placement = model.create_entity("IfcAxis2Placement3D", Location=origin)
    extrusion_dir = cache.get_z_up_direction()

    base_solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=placement,
        ExtrudedDirection=extrusion_dir,
        Depth=height
    )

    # _cuts (DIFFERENCE) → _adds (UNION) → _intersects (INTERSECTION), each
    # class balanced internally, in local coordinates before rotation.
    cuts = elem._cuts if elem._cuts else []
    adds = elem._adds if elem._adds else []
    intersects = elem._intersects if getattr(elem, "_intersects", None) else []
    result_solid = _apply_boolean_chain_ios(
        model, base_solid,
        build=lambda e, is_cut=True: _create_hole_solid_local(model, e, min_pt, center, cache, is_cut=is_cut),
        cuts=cuts, adds=adds, intersects=intersects)

    # Apply _clips as IfcBooleanClippingResult (local frame, before rotation —
    # same convention as .difference() on a rotated Box: the plane is
    # interpreted in the box's UNROTATED local frame).
    result_solid = _apply_clips(model, result_solid, elem._clips, center=center)

    # Build rotation matrix from sequential rotations
    # DSL and IFC both use Z-up — direct axis mapping (identity)
    rotation_matrix = np.eye(3, dtype=np.float64)
    rotations = elem.rotations if elem.rotations else []
    for axis, angle_centideg in rotations:
        ifc_axis = axis.lower()
        angle_sign = 1.0

        angle_rad = math.radians(angle_centideg / 100.0) * angle_sign
        rot = _axis_rotation_matrix_3x3(ifc_axis, angle_rad)
        rotation_matrix = rotation_matrix @ rot

    # Build 4x4 transformation matrix: rotation + translation to center
    matrix = np.array([
        [rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], center[0]],
        [rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], center[1]],
        [rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], center[2]],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    # Create IfcBuildingElementProxy (IFC class reassignment happens in main loop for site elements)
    proxy = _create_ifc_entity(model, "IfcBuildingElementProxy",
                              name=elem.ifc_name,
                              predefined_type="NOTDEFINED")

    # Set placement with rotation
    _set_placement(model, proxy, matrix, cache)

    # Collect geometry items
    geometry_items = [result_solid]
    has_booleans = bool(cuts) or bool(adds) or bool(intersects) or bool(elem._clips)
    rep_type = "CSG" if has_booleans else "SweptSolid"

    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType=rep_type,
        Items=geometry_items
    )

    product_shape = model.create_entity(
        "IfcProductDefinitionShape",
        Representations=[shape_rep]
    )
    proxy.Representation = product_shape

    # Registry material (v1.5): IfcMaterial + Psets + render style.
    # Precedence: explicit color= > material render > type palette.
    styled = _apply_registry_material(model, proxy, result_solid, elem, cache)
    if not styled:
        # Apply surface style based on color parameter or type
        color_key = elem.color if elem.color else element_type
        _apply_typed_color(model, result_solid, color_key, cache)

    # Process openings on box-mode Solids
    if hasattr(elem, '_openings') and elem._openings:
        _process_openings_on_element(model, context, proxy, elem, cache)
    if getattr(elem, '_voids', None):
        _process_voids_on_element(model, context, proxy, elem, cache)

    return proxy


def _create_extrude(model, context, elem: 'Extrude', cache: EntityCache):
    """
    Create IfcBuildingElementProxy with extruded polygon geometry for an Extrude primitive.

    Uses Newell's Method for
    plane/normal calculation from 3D contour points.

    - contour: List of 3D Points defining polygon vertices in world space
    - thickness: Extrusion depth in meters (already normalized)
    - _cuts/_adds: Boolean operations via PrivateAttr
    - type: Determines color
    """
    assert run is not None
    assert np is not None

    thickness_m = elem.thickness
    if thickness_m is None:
        # v1.5 one-source rule: a sheet Material carries the thickness
        # (Material(key=..., thickness_mm=...) replaces thickness=).
        thickness_m = _material_thickness_m(elem)

    # Extract 3D contour points (already in meters from executor)
    contour_3d = [(pt.x, pt.y, pt.z) for pt in elem.contour]

    if len(contour_3d) < 3:
        return None

    # Calculate plane normal using Newell's Method; orientation-rule keeps
    # sloped/horizontal surfaces pointing up (Z-up convention) so an
    # accidentally-CW-wound roof slope doesn't extrude downward through
    # the ground. Vertical surfaces (|nz| < epsilon) are not flipped.
    normal = _newell_normal(contour_3d)
    normal = _apply_orientation_rules(normal)

    # Calculate centroid for placement origin
    centroid = _calculate_centroid(contour_3d)

    # Build local coordinate system from normal
    z_axis = np.array(normal)

    # Choose reference vector using "Up-Preferred" strategy (Z-up)
    if abs(z_axis[2]) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])

    x_axis = np.cross(ref, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 0.001:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)

    # Project 3D contour points onto local 2D plane
    contour_2d = []
    for pt in contour_3d:
        v = np.array([pt[0] - centroid[0], pt[1] - centroid[1], pt[2] - centroid[2]])
        local_x = float(np.dot(v, x_axis))
        # +y_axis matches the placement's reconstructed local Y; see
        # _create_solid_contour_ir for the mirror-on-negation explanation.
        local_y = float(np.dot(v, y_axis))
        contour_2d.append((local_x, local_y))

    # Transform centroid to IFC coordinates (identity, both Z-up)
    ifc_origin = cache.get_or_create_point((centroid[0], centroid[1], centroid[2]))

    # Transform axes to IFC coordinates (identity, both Z-up)
    axis_vec = (float(z_axis[0]), float(z_axis[1]), float(z_axis[2]))
    ref_vec = (float(x_axis[0]), float(x_axis[1]), float(x_axis[2]))

    axis = cache.get_or_create_direction(*axis_vec)
    ref_direction = cache.get_or_create_direction(*ref_vec)

    axis2_placement = model.create_entity("IfcAxis2Placement3D",
                                           Location=ifc_origin,
                                           Axis=axis,
                                           RefDirection=ref_direction)

    local_placement = model.create_entity("IfcLocalPlacement",
                                           RelativePlacement=axis2_placement)

    # Create IfcBuildingElementProxy (IFC class reassignment happens in main loop for site elements)
    ifc_proxy = _create_ifc_entity(model, "IfcBuildingElementProxy", name=elem.ifc_name, predefined_type="NOTDEFINED")

    ifc_proxy.ObjectPlacement = local_placement

    # Create profile from local 2D contour
    points = [model.create_entity("IfcCartesianPoint", Coordinates=[p[0], p[1]])
              for p in contour_2d]
    points.append(points[0])

    polyline = model.create_entity("IfcPolyline", Points=points)

    profile = model.create_entity(
        "IfcArbitraryClosedProfileDef",
        ProfileType="AREA",
        OuterCurve=polyline
    )

    # Check for boolean operations
    cuts = elem._cuts if elem._cuts else []
    adds = elem._adds if elem._adds else []
    intersects = elem._intersects if getattr(elem, "_intersects", None) else []
    clips = elem._clips if getattr(elem, "_clips", None) else []
    has_booleans = bool(cuts) or bool(adds) or bool(intersects) or bool(clips)

    result_solid = None
    if has_booleans:
        local_origin = cache.get_origin_3d()
        solid_placement = model.create_entity("IfcAxis2Placement3D", Location=local_origin)
        extrusion_dir = cache.get_z_up_direction()

        base_solid = model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile,
            Position=solid_placement,
            ExtrudedDirection=extrusion_dir,
            Depth=thickness_m
        )

        final_solid = _apply_contour_boolean_operations(
            model, base_solid, cuts, adds,
            centroid, x_axis, y_axis, z_axis, thickness_m, cache,
            intersects=intersects, clips=clips,
        )

        result_solid = final_solid
        ifc_proxy.Representation = _create_body_representation(model, context, final_solid)
    else:
        shape_rep, result_solid = _create_extruded_body(model, context, profile, thickness_m, cache)
        _assign_body_representation(model, ifc_proxy, shape_rep)

    # Registry material (v1.5): IfcMaterial + Psets + render style.
    # Precedence: explicit color= > material render > type palette.
    styled = _apply_registry_material(model, ifc_proxy, result_solid, elem, cache)
    if not styled:
        # Apply surface style based on color parameter or type
        element_type = elem.type or ''
        color_key = elem.color if elem.color else element_type
        if result_solid:
            _apply_typed_color(model, result_solid, color_key, cache)

    # Process openings on box-mode Solids
    if hasattr(elem, '_openings') and elem._openings:
        _process_openings_on_element(model, context, ifc_proxy, elem, cache)
    if getattr(elem, '_voids', None):
        _process_voids_on_element(model, context, ifc_proxy, elem, cache)

    return ifc_proxy


def _create_sweep(model, context, elem: 'Sweep', cache: EntityCache):
    """
    Create IfcMember/IfcBeam/IfcColumn with extruded profile for Sweep primitive.

    Uses catalog profile lookup for dimensions and IFC class.

    Three.js standard geometry pipeline:
    1. Sweep centered in local XY plane (w=X, d=Y)
    2. Extrusion along local Z axis
    3. Placement at midpoint between start/end
    4. LookAt rotation: local Z points from start toward end

    v1.5 (WS1 PR-E): the path form (section= travels along path=) routes to
    _create_profile_path; the legacy start/end form keeps this body.
    """
    assert run is not None
    assert np is not None

    if getattr(elem, 'path', None) is not None:
        return _create_profile_path(model, context, elem, cache)

    # Section source (v17 one-source rule): a member Material carries the
    # section (Material(key=..., profile_mm=(w, h))). When the legacy start/end
    # Sweep has no material profile, fall back to a standard 89x38 stud member
    # (the legacy bim_catalog "Stud_2x4" default; catalog retired).
    material_selection = _registry_material_selection(elem)
    if material_selection is not None and material_selection.profile_mm is not None:
        section_w_mm, section_d_mm = material_selection.profile_mm
        ifc_class = "IfcMember"
        predefined_type = None
        profile_name = material_selection.key
    else:
        section_w_mm, section_d_mm = 89, 38
        ifc_class = "IfcMember"
        predefined_type = "STUD"
        profile_name = None

    # Transform DSL coordinates to IFC coordinates (both Z-up, identity)
    start_ifc = np.array(_dsl_to_ifc_point(elem.start.x, elem.start.y, elem.start.z))
    end_ifc = np.array(_dsl_to_ifc_point(elem.end.x, elem.end.y, elem.end.z))

    # Calculate member direction and length
    direction = end_ifc - start_ifc
    length = np.linalg.norm(direction)

    if length < 0.001:
        return None

    z_axis = direction / length

    # Calculate midpoint for placement
    midpoint = (start_ifc + end_ifc) / 2.0

    # Build lookAt rotation matrix
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(z_axis, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])

    x_axis = np.cross(world_up, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 0.001:
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        x_axis = x_axis / x_norm

    y_axis = np.cross(z_axis, x_axis)

    # Handle wide_axis
    wide_axis = getattr(elem, 'wide_axis', 'x')
    if wide_axis == 'x':
        x_axis, y_axis = y_axis, -x_axis

    # Build 4x4 transformation matrix
    matrix = np.array([
        [x_axis[0], y_axis[0], z_axis[0], midpoint[0]],
        [x_axis[1], y_axis[1], z_axis[1], midpoint[1]],
        [x_axis[2], y_axis[2], z_axis[2], midpoint[2]],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    # Create member with correct IFC class (catalog or material-derived)
    ifc_member = _create_ifc_entity(model, ifc_class, predefined_type=predefined_type)

    _set_placement(model, ifc_member, matrix, cache)

    # Create rectangular profile
    profile = cache.get_or_create_rect_profile(
        x_dim=float(section_w_mm) / 1000.0,
        y_dim=float(section_d_mm) / 1000.0,
        name=profile_name
    )

    # Create centered extrusion: from -length/2 to +length/2 on local Z
    shape_rep, solid = _create_extruded_body(model, context, profile, length, cache, z_offset=-length / 2.0)
    _assign_body_representation(model, ifc_member, shape_rep)

    # Registry material (v1.5): IfcMaterial + Psets + render style.
    # (Profiles have no palette color path — legacy output is unchanged.)
    _apply_registry_material(model, ifc_member, solid, elem, cache)

    if getattr(elem, "_voids", None):
        _process_voids_on_element(model, context, ifc_member, elem, cache)
    return ifc_member


# =============================================================================
# DSL v1.5 PRIMITIVES (WS1 PR-E): Pipe, Revolve, Bar, generic Element, path Sweep
# =============================================================================
# Conventions shared by these creators:
# * All geometry arrives in METERS (normalize_project_to_meters converted
#   path/section points and the scalar mm fields: radius, diameter,
#   bend_radius, fillet_radius).
# * Products carry an IDENTITY ObjectPlacement; the world position lives in
#   the geometry items' own Position/Directrix — this keeps multi-item
#   representations (mitered profile segments) and boolean operand frames
#   simple, and _set_relative_placements' storey-elevation adjustment
#   applies to the identity placement exactly like the container creators.
# * Boolean operands (_cuts/_adds) are resolved in the same frame via
#   _create_hole_solid_local with parent_center = world origin.


def _ensure_si_unit(model, unit_type: str, name: str, prefix: str = None) -> None:
    """Append an IfcSIUnit to the IfcUnitAssignment when missing.

    generate_ifc assigns only length/area/volume units; a Revolve needs a
    PLANEANGLEUNIT (RADIAN — IfcRevolvedAreaSolid.Angle is emitted in
    radians) and Bar quantities need a MASSUNIT (kg). Added lazily so
    buildings without these primitives stay byte-identical.
    """
    for ua in model.by_type("IfcUnitAssignment"):
        units = list(ua.Units)
        if any(getattr(u, "UnitType", None) == unit_type for u in units):
            continue
        kwargs = {"UnitType": unit_type, "Name": name}
        if prefix:
            kwargs["Prefix"] = prefix
        units.append(model.create_entity("IfcSIUnit", **kwargs))
        ua.Units = units


def _offset_pt(pt, offset):
    return (pt.x + offset[0], pt.y + offset[1], pt.z + offset[2])


def _create_directrix(model, pts: list, fillet_radius: float):
    """Directrix curve for a swept disk: IfcPolyline (sharp) or
    IfcIndexedPolyCurve with true IfcArcIndex arcs at filleted joints.

    ``pts`` are (x, y, z) tuples in meters, already offset into the target
    frame. ``fillet_radius`` in meters; 0 = sharp polyline.
    """
    has_fillets = fillet_radius and fillet_radius > 0 and len(pts) >= 3
    if not has_fillets:
        points = [model.create_entity("IfcCartesianPoint", Coordinates=list(p))
                  for p in pts]
        return model.create_entity("IfcPolyline", Points=points)

    coords: list = [tuple(float(c) for c in pts[0])]
    segments: list = []
    run = [1]  # 1-based indices of the current straight run
    for j in range(1, len(pts) - 1):
        fc = _fillet_corner(pts[j - 1], pts[j], pts[j + 1], fillet_radius)
        if fc is None:
            # Collinear joint — the straight run continues through it.
            coords.append(tuple(float(c) for c in pts[j]))
            run.append(len(coords))
            continue
        p_start, p_mid, p_end, _t, _theta = fc
        coords.append(tuple(float(c) for c in p_start))
        run.append(len(coords))
        segments.append(model.create_entity("IfcLineIndex", tuple(run)))
        coords.append(tuple(float(c) for c in p_mid))
        coords.append(tuple(float(c) for c in p_end))
        arc_start = run[-1]
        segments.append(model.create_entity(
            "IfcArcIndex", (arc_start, arc_start + 1, arc_start + 2)))
        run = [arc_start + 2]  # next run starts at the arc's end point
    coords.append(tuple(float(c) for c in pts[-1]))
    run.append(len(coords))
    segments.append(model.create_entity("IfcLineIndex", tuple(run)))
    point_list = model.create_entity("IfcCartesianPointList3D", CoordList=coords)
    return model.create_entity("IfcIndexedPolyCurve", Points=point_list,
                               Segments=segments)


def _create_swept_disk_solid(model, path, radius: float, fillet_radius: float,
                             cache: EntityCache, offset=(0.0, 0.0, 0.0)):
    """IfcSweptDiskSolid along a path (meters), shifted by ``offset``."""
    pts = [_offset_pt(p, offset) for p in path]
    directrix = _create_directrix(model, pts, fillet_radius)
    return model.create_entity(
        "IfcSweptDiskSolid",
        Directrix=directrix,
        Radius=float(radius),
    )


def _cylinder_solid(model, a, b, radius: float, cache: EntityCache,
                    ext_a: float = 0.0, ext_b: float = 0.0):
    """Extruded-circle cylinder from ``a`` to ``b`` (tuples, meters), with
    optional overshoot at each end."""
    assert np is not None
    av = np.array(a, dtype=np.float64)
    bv = np.array(b, dtype=np.float64)
    d = bv - av
    length = float(np.linalg.norm(d))
    d = d / length
    if abs(d[2]) < 0.99:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(ref, d)
    x_axis = x_axis / np.linalg.norm(x_axis)
    start = av - d * ext_a
    loc = model.create_entity("IfcCartesianPoint",
                              Coordinates=[float(v) for v in start])
    axis = model.create_entity("IfcDirection",
                               DirectionRatios=[float(v) for v in d])
    ref_dir = model.create_entity("IfcDirection",
                                  DirectionRatios=[float(v) for v in x_axis])
    position = model.create_entity("IfcAxis2Placement3D", Location=loc,
                                   Axis=axis, RefDirection=ref_dir)
    profile = model.create_entity("IfcCircleProfileDef", ProfileType="AREA",
                                  Radius=float(radius))
    return model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=position,
        ExtrudedDirection=cache.get_z_up_direction(),
        Depth=float(length + ext_a + ext_b),
    )


def _create_disk_solid_boolean_safe(model, path, radius: float,
                                    fillet_radius: float, cache: EntityCache,
                                    offset=(0.0, 0.0, 0.0)):
    """Boolean-operand-safe equivalent of a swept disk (pinned WS1 PR-E).

    ifcopenshell's geometry kernel mis-tessellates IfcBooleanResult trees
    containing IfcSweptDiskSolid operands (measured: a box minus a swept
    disk loses ~30% volume; extruded-circle cylinders are exact). So
    whenever a Pipe/Bar participates in a boolean — as a void/union operand
    or as a base carrying its own cuts — its geometry is built as extruded
    IfcCircleProfileDef cylinders instead:

    * straight 2-point path -> one cylinder (geometrically identical);
    * bent path -> fillets faceted (<=15 deg chords), then one cylinder per
      segment, mitered at the joints' bisector planes and unioned into a
      single boolean chain — the same treatment the path Sweep gets.

    The pure render path (no booleans) keeps the true IfcSweptDiskSolid
    with real arcs.
    """
    assert np is not None
    pts = [_offset_pt(p, offset) for p in path]
    if fillet_radius and fillet_radius > 0:
        pts = _facet_fillet_path(pts, fillet_radius)
    if len(pts) == 2:
        return _cylinder_solid(model, pts[0], pts[1], radius, cache)

    dirs = []
    for i in range(len(pts) - 1):
        d = np.array(pts[i + 1], dtype=np.float64) - np.array(pts[i], dtype=np.float64)
        dirs.append(d / np.linalg.norm(d))

    def joint(i_prev, i_next, vertex):
        m = dirs[i_prev] + dirs[i_next]
        norm = np.linalg.norm(m)
        if norm < 1e-9:
            return None
        m = m / norm
        cos_t = max(-1.0, min(1.0, float(np.dot(dirs[i_prev], dirs[i_next]))))
        theta = math.acos(cos_t)
        overshoot = radius * math.tan(theta / 2.0) * 1.05 + 0.001 if theta > 1e-9 else 0.0
        return m, overshoot, vertex

    solids = []
    n_seg = len(pts) - 1
    for i in range(n_seg):
        start_joint = joint(i - 1, i, pts[i]) if i > 0 else None
        end_joint = joint(i, i + 1, pts[i + 1]) if i < n_seg - 1 else None
        ext_a = start_joint[1] if start_joint else 0.0
        ext_b = end_joint[1] if end_joint else 0.0
        solid = _cylinder_solid(model, pts[i], pts[i + 1], radius, cache,
                                ext_a=ext_a, ext_b=ext_b)
        if start_joint is not None:
            m, _o, vertex = start_joint
            solid = model.create_entity(
                "IfcBooleanResult", Operator="DIFFERENCE",
                FirstOperand=solid,
                SecondOperand=_miter_half_space(model, vertex, m))
        if end_joint is not None:
            m, _o, vertex = end_joint
            solid = model.create_entity(
                "IfcBooleanResult", Operator="DIFFERENCE",
                FirstOperand=solid,
                SecondOperand=_miter_half_space(model, vertex, -m))
        solids.append(solid)

    # Balanced union: a many-segment bend otherwise nests one boolean per
    # segment on top of the per-segment miter cuts.
    return _balanced_union_ios(model, solids)


#: Maximum distance between a faceted chord and the true arc it replaces, in
#: METRES. Only spent where an arc is genuinely approximated — a full 360°
#: revolve of a rectangle profile emits an exact circle and reads this not at
#: all. 5 mm at any radius is inside LOD 350 tolerance and costs ~29 segments
#: on the largest arc in the corpus-adjacent hospital model (r = 30 m, 60°).
_REVOLVE_CHORD_TOL_M = 0.005
#: A closed loop needs 3 sides before it bounds anything; 180 caps a
#: degenerate-radius blowup (the segment count grows as 1/sqrt(tol/r)).
_REVOLVE_MIN_SEGMENTS = 3
_REVOLVE_MAX_SEGMENTS = 180


def _revolve_segments(radius: float, angle_rad: float) -> int:
    """Segments needed to hold ``_REVOLVE_CHORD_TOL_M`` at ``radius``.

    The sagitta of a chord subtending 2h is r(1 - cos h), so the half-step
    that meets the tolerance is acos(1 - tol/r). Deterministic: pure float
    arithmetic on values that are already in the model, no iteration.
    """
    if radius <= _REVOLVE_CHORD_TOL_M:
        return _REVOLVE_MIN_SEGMENTS
    half = math.acos(max(-1.0, min(1.0, 1.0 - _REVOLVE_CHORD_TOL_M / radius)))
    if half <= 0.0:
        return _REVOLVE_MAX_SEGMENTS
    return max(_REVOLVE_MIN_SEGMENTS,
               min(_REVOLVE_MAX_SEGMENTS, int(math.ceil(angle_rad / (2.0 * half)))))


def _revolve_rect_bounds(profile):
    """``(r0, r1, h0, h1)`` when ``profile`` is a rectangle aligned to the axis.

    A Revolve profile lives in the (radial, axial) plane — x is the distance
    from the axis, y the offset along it. When that polygon is exactly the
    four corners of its own bounding box, the swept solid is an annular
    sector PRISM, which an extrusion along the axis describes exactly (and
    for a full turn, with a real circle — no faceting at all).

    Returns ``None`` for anything else, which routes to the brep path.
    Deliberately strict: a rectangle traversed as a bowtie is not a
    rectangle, and a degenerate one (zero radial or axial extent) has no
    volume to extrude.
    """
    pts = [(float(p.x), float(p.y)) for p in profile]
    if len(pts) > 1 and pts[0] == pts[-1]:      # tolerate an explicit closer
        pts = pts[:-1]
    if len(pts) != 4:
        return None
    r0 = min(x for x, _ in pts); r1 = max(x for x, _ in pts)
    h0 = min(y for _, y in pts); h1 = max(y for _, y in pts)
    if r1 - r0 <= 0.0 or h1 - h0 <= 0.0:
        return None
    if {(x, y) for x, y in pts} != {(r0, h0), (r1, h0), (r1, h1), (r0, h1)}:
        return None
    # Consecutive corners of a rectangle differ in exactly ONE coordinate;
    # a bowtie (which has the same point SET) differs in both on two edges.
    for i in range(4):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % 4]
        if (ax != bx) == (ay != by):
            return None
    return r0, r1, h0, h1


def _revolve_frame(elem: 'Revolve', offset):
    """``(p0, a, r_dir, s_dir)`` — the pinned revolve frame, in world metres.

    ``a`` is the axis (path[0] -> path[1]); ``r_dir`` the radial start
    direction; ``s_dir = a x r_dir`` completes a right-handed basis, so a
    right-handed rotation by t about ``a`` sends ``r_dir`` to
    ``cos t * r_dir + sin t * s_dir``. That identity is what lets the sector
    be written as an ordinary counter-clockwise 2D profile.
    """
    assert np is not None
    p0 = _offset_pt(elem.path[0], offset)
    p1 = _offset_pt(elem.path[1], offset)
    a = np.array([p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]], dtype=np.float64)
    a = a / np.linalg.norm(a)
    zxa = np.cross(np.array([0.0, 0.0, 1.0]), a)
    n_zxa = np.linalg.norm(zxa)
    r_dir = np.array([1.0, 0.0, 0.0]) if n_zxa < 1e-9 else zxa / n_zxa
    return np.asarray(p0, dtype=np.float64), a, r_dir, np.cross(a, r_dir)


def _revolve_as_extrusion(model, elem, rect, offset):
    """Rectangle-profile Revolve -> IfcExtrudedAreaSolid about the same axis.

    EXACT for a full turn (a circle or a hollow circle — no facets at all);
    sagitta-bounded for a sector, where the arc becomes a polygon.
    """
    r0, r1, h0, h1 = rect
    p0, a, r_dir, s_dir = _revolve_frame(elem, offset)
    angle_rad = math.radians(elem.angle / 100.0)
    base = p0 + a * h0                       # the profile plane sits at h0

    position = model.create_entity(
        "IfcAxis2Placement3D",
        Location=model.create_entity(
            "IfcCartesianPoint",
            Coordinates=[float(base[0]), float(base[1]), float(base[2])]),
        Axis=model.create_entity(
            "IfcDirection", DirectionRatios=[float(v) for v in a]),
        RefDirection=model.create_entity(
            "IfcDirection", DirectionRatios=[float(v) for v in r_dir]),
    )
    origin2d = model.create_entity(
        "IfcAxis2Placement2D",
        Location=model.create_entity("IfcCartesianPoint", Coordinates=[0.0, 0.0]),
        RefDirection=model.create_entity("IfcDirection", DirectionRatios=[1.0, 0.0]),
    )

    full_turn = elem.angle == 36000
    if full_turn and r0 <= 0.0:
        profile = model.create_entity("IfcCircleProfileDef", ProfileType="AREA",
                                      Position=origin2d, Radius=float(r1))
    elif full_turn:
        profile = model.create_entity("IfcCircleHollowProfileDef",
                                      ProfileType="AREA", Position=origin2d,
                                      Radius=float(r1),
                                      WallThickness=float(r1 - r0))
    else:
        n = _revolve_segments(r1, angle_rad)
        ring = [(math.cos(angle_rad * i / n), math.sin(angle_rad * i / n))
                for i in range(n + 1)]
        pts = [(r1 * c, r1 * s) for c, s in ring]
        # A sector reaching the axis is a pie (one apex); an annular one
        # returns along the inner arc. Both traverse counter-clockwise, which
        # is the direction a positive Angle sweeps in this basis.
        pts += [(0.0, 0.0)] if r0 <= 0.0 else [(r0 * c, r0 * s)
                                               for c, s in reversed(ring)]
        nodes = [model.create_entity("IfcCartesianPoint",
                                     Coordinates=[float(x), float(y)])
                 for x, y in pts]
        nodes.append(nodes[0])
        profile = model.create_entity(
            "IfcArbitraryClosedProfileDef", ProfileType="AREA",
            OuterCurve=model.create_entity("IfcPolyline", Points=nodes))

    return model.create_entity(
        "IfcExtrudedAreaSolid", SweptArea=profile, Position=position,
        ExtrudedDirection=model.create_entity("IfcDirection",
                                              DirectionRatios=[0.0, 0.0, 1.0]),
        Depth=float(h1 - h0),
    )


def _revolve_as_brep(model, elem, offset):
    """General-profile Revolve -> IfcFacetedBrep (a solid, so still boolean-able).

    Faceted only in the sweep direction; every profile vertex is exact. The
    caps stay single polygonal faces rather than fans, because a fan over a
    non-convex profile self-overlaps.

    Orientation is MEASURED, not assumed: the shell is built, its signed
    volume taken by the divergence theorem, and every loop reversed when it
    comes out negative. That is exact for any simple polygon (the fan's
    signed areas cancel), so an author's profile winding cannot leak out as
    inside-out geometry.
    """
    p0, a, r_dir, s_dir = _revolve_frame(elem, offset)
    angle_rad = math.radians(elem.angle / 100.0)
    prof = [(float(p.x), float(p.y)) for p in elem.profile]
    if len(prof) > 1 and prof[0] == prof[-1]:
        prof = prof[:-1]
    full_turn = elem.angle == 36000
    n = _revolve_segments(max(x for x, _ in prof) or 0.0, angle_rad)

    rings, verts = [], []
    for j in range(n if full_turn else n + 1):
        t = angle_rad * j / n
        radial = math.cos(t) * r_dir + math.sin(t) * s_dir
        ring = []
        for x, y in prof:
            v = p0 + radial * x + a * y
            ring.append(len(verts)); verts.append(v)
        rings.append(ring)

    loops: List[List[int]] = []
    m = len(prof)
    for j in range(len(rings) if full_turn else len(rings) - 1):
        cur, nxt = rings[j], rings[(j + 1) % len(rings)]
        for i in range(m):
            k = (i + 1) % m
            loops.append([cur[i], cur[k], nxt[k]])
            loops.append([cur[i], nxt[k], nxt[i]])
    if not full_turn:                       # the two end caps
        loops.append(list(rings[0]))
        loops.append(list(reversed(rings[-1])))

    vol = 0.0
    for loop in loops:                      # divergence theorem, fan per loop
        for i in range(1, len(loop) - 1):
            v0, v1, v2 = verts[loop[0]], verts[loop[i]], verts[loop[i + 1]]
            vol += float(np.dot(v0, np.cross(v1, v2)))
    if vol < 0.0:
        loops = [list(reversed(loop)) for loop in loops]

    nodes = [model.create_entity(
        "IfcCartesianPoint",
        Coordinates=[float(v[0]), float(v[1]), float(v[2])]) for v in verts]
    faces = [model.create_entity(
        "IfcFace",
        Bounds=[model.create_entity(
            "IfcFaceOuterBound",
            Bound=model.create_entity("IfcPolyLoop",
                                      Polygon=[nodes[i] for i in loop]),
            Orientation=True)]) for loop in loops]
    return model.create_entity(
        "IfcFacetedBrep",
        Outer=model.create_entity("IfcClosedShell", CfsFaces=faces))


def _create_revolved_solid(model, elem: 'Revolve', cache: EntityCache,
                           offset=(0.0, 0.0, 0.0)):
    """The solid for a Revolve (meters), shifted by ``offset``.

    NOT an ``IfcRevolvedAreaSolid``, and that is the whole point of this
    function. **web-ifc — the production renderer, via @thatopen/fragments —
    draws no IfcRevolvedAreaSolid at all**, and draws it silently: no error,
    no warning, an empty mesh. Measured against web-ifc 0.0.77 (the latest
    published) with a two-solid falsifier and a five-case matrix: arbitrary
    profile, rectangle profile, 60 degrees, 360 degrees and both position
    frames all return ZERO vertices, while the SAME profile entity extruded
    returns geometry. So the analytic form was correct and invisible — every
    Revolve ever authored, including ``systems/details``' door rose, has been
    missing from the app since the primitive shipped. See for
    the falsifier and the upstream report.

    Both replacements are ``IfcSolidModel`` subtypes, so a Revolve remains a
    legal ``IfcBooleanOperand`` and ``.void()`` / ``.difference()`` / the
    inferred carve keep working on it. That is why neither is a tessellation:
    ``IfcTriangulatedFaceSet`` renders too, but IFC does not admit it as a
    boolean operand, so choosing it would have traded an invisible floorplate
    for an uncarvable one.

    The pinned conventions are UNCHANGED — the axis, the radial start plane
    and the winding are the same frame the arch test proved:

    * Axis a = path[0] -> path[1]; radial start r = unit(world_Z x a), with a
      fallback to world +X for a (near-)vertical axis.
    * Positive Angle revolves right-handed about a, so a horizontal axis
      sweeps UPWARD first and angle=18000 is the upper half.

    What changes is only the IFC spelling of that sweep:

    * Rectangle profile (the axis-aligned box case) -> ``IfcExtrudedAreaSolid``
      along a. A full turn is EXACT — ``IfcCircleProfileDef`` when the profile
      reaches the axis, ``IfcCircleHollowProfileDef`` when it does not. A
      sector facets the arc within ``_REVOLVE_CHORD_TOL_M``.
    * Anything else -> ``IfcFacetedBrep``, faceted in the sweep direction only.
    """
    assert np is not None
    # Kept although no Angle is emitted any more: a plane-angle unit is
    # legitimate in any model, and dropping it here would move bytes in
    # files whose only change should be the solid.
    _ensure_si_unit(model, "PLANEANGLEUNIT", "RADIAN")
    rect = _revolve_rect_bounds(elem.profile)
    if rect is not None:
        return _revolve_as_extrusion(model, elem, rect, offset)
    return _revolve_as_brep(model, elem, offset)


def _path_aabb(path, radius: float):
    """Axis-aligned bounds of a swept path (meters), padded by radius."""
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    zs = [p.z for p in path]
    return (
        (min(xs) - radius, min(ys) - radius, min(zs) - radius),
        (max(xs) + radius, max(ys) + radius, max(zs) + radius),
    )


def _apply_world_frame_booleans(model, base_solid, elem, aabb_min, aabb_max,
                                cache: EntityCache):
    """Apply an element's _cuts/_adds to a world-frame base solid.

    The new primitives keep their geometry in world coordinates (identity
    product placement), so operands resolve with parent_center = origin;
    aabb bounds feed the coplanar-face extension only.
    """
    origin = (0.0, 0.0, 0.0)
    result = _apply_boolean_chain_ios(
        model, base_solid,
        build=lambda e, is_cut=True: _create_hole_solid_local(model, e, aabb_min, origin, cache,
                                                             parent_max=aabb_max, is_cut=is_cut),
        cuts=(elem._cuts or []),
        adds=(elem._adds or []),
        intersects=(getattr(elem, "_intersects", None) or []))
    # .clip() half-spaces — world frame (identity placement), no transform.
    result = _apply_clips(model, result, getattr(elem, "_clips", None) or [])
    return result


def _v15_rep_type(solid) -> str:
    """RepresentationType for a v1.5 primitive's final solid."""
    if solid.is_a("IfcBooleanResult"):
        return "CSG"
    if solid.is_a("IfcSweptDiskSolid"):
        return "AdvancedSweptSolid"
    return "SweptSolid"


def _finish_v15_product(model, context, product, elem, result_solid,
                        rep_type: str, cache: EntityCache):
    """Identity placement + Body representation + material/color styling."""
    assert np is not None
    _set_placement(model, product, np.eye(4), cache)
    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType=rep_type,
        Items=[result_solid],
    )
    _assign_body_representation(model, product, shape_rep)
    styled = _apply_registry_material(model, product, result_solid, elem, cache)
    if not styled:
        color_key = getattr(elem, 'color', None) or ''
        if color_key:
            _apply_typed_color(model, result_solid, color_key, cache)
    # A v1.5 primitive carries its body ON its own product, so a ``.void()``
    # attaches here — exactly as it does for a Box in ``_create_solid``. It
    # did not, and the drop was silent in the WS-A ``body.void()`` shape: the
    # model validated, ``apply_displacement`` carved the details standing in
    # the hole, the compile reported success, and the file came out with no
    # IfcOpeningElement, no IfcRelVoidsElement and no boolean — a Pipe with a
    # hole in the DSL and no hole in the IFC. Every curved primitive finishes
    # through this function (Pipe, Revolve, Bar), so this is the one place they
    # all needed it.
    if getattr(elem, "_voids", None):
        _process_voids_on_element(model, context, product, elem, cache)
    return product


def _create_pipe(model, context, elem: 'Pipe', cache: EntityCache):
    """IfcBuildingElementProxy with IfcSweptDiskSolid geometry for Pipe.

    Bends between path segments are TRUE ARCS (IfcArcIndex segments on an
    IfcIndexedPolyCurve directrix) of radius fillet_radius; 0 = sharp.

    When the Pipe carries booleans (.cuts()/.adds()) its base switches to
    the boolean-safe cylinder form — the kernel mis-tessellates boolean
    trees over IfcSweptDiskSolid (see _create_disk_solid_boolean_safe).
    """
    has_own_booleans = bool(elem._cuts) or bool(elem._adds) or bool(getattr(elem, "_intersects", None))
    if has_own_booleans:
        base = _create_disk_solid_boolean_safe(model, elem.path, elem.radius,
                                               elem.fillet_radius, cache)
    else:
        base = _create_swept_disk_solid(model, elem.path, elem.radius,
                                        elem.fillet_radius, cache)
    aabb_min, aabb_max = _path_aabb(elem.path, elem.radius)
    result = _apply_world_frame_booleans(model, base, elem, aabb_min, aabb_max, cache)
    proxy = _create_ifc_entity(model, "IfcBuildingElementProxy",
                               name=elem.ifc_name,
                               predefined_type="NOTDEFINED")
    rep_type = _v15_rep_type(result)
    return _finish_v15_product(model, context, proxy, elem, result, rep_type, cache)


def _create_revolve(model, context, elem: 'Revolve', cache: EntityCache):
    """IfcBuildingElementProxy with IfcRevolvedAreaSolid geometry for Revolve."""
    _ensure_si_unit(model, "PLANEANGLEUNIT", "RADIAN")
    base = _create_revolved_solid(model, elem, cache)
    # Conservative AABB: axis endpoints padded by the max radial extent.
    max_r = max((p.x for p in elem.profile), default=0.0)
    aabb_min, aabb_max = _path_aabb(elem.path, max_r)
    result = _apply_world_frame_booleans(model, base, elem, aabb_min, aabb_max, cache)
    proxy = _create_ifc_entity(model, "IfcBuildingElementProxy",
                               name=elem.ifc_name,
                               predefined_type="NOTDEFINED")
    return _finish_v15_product(model, context, proxy, elem, result,
                               _v15_rep_type(result), cache)


def _create_bar(model, context, elem: 'Bar', cache: EntityCache):
    """IfcReinforcingBar with swept-disk geometry + derived quantities.

    * PredefinedType from bar_type (stirrup -> LIGATURE; IFC4 has no
      STIRRUP literal).
    * NominalDiameter = the nominal diameter (meters, file length unit).
    * Tag = mark (schedule mark — shared across bars, not identity).
    * Qto_ReinforcingElementBaseQuantities: Length = true arc-corrected
      centerline length; Weight = Length * area * density. Density source
      (pinned WS1 PR-E): the registry material's
      psets["Pset_MaterialCommon"]["MassDensity"] (kg/m3) — 7850 for
      Steel_B500B. IfcElementQuantity is used (not a Pset) because these
      are measured quantities; a KILO-GRAM MASSUNIT is ensured lazily.
    """
    assert ifcopenshell is not None
    radius = elem.diameter / 2.0
    if elem._cuts or elem._adds or getattr(elem, "_intersects", None):
        # Boolean-safe cylinder form (see _create_disk_solid_boolean_safe).
        base = _create_disk_solid_boolean_safe(model, elem.path, radius,
                                               elem.bend_radius, cache)
    else:
        base = _create_swept_disk_solid(model, elem.path, radius,
                                        elem.bend_radius, cache)
    aabb_min, aabb_max = _path_aabb(elem.path, radius)
    result = _apply_world_frame_booleans(model, base, elem, aabb_min, aabb_max, cache)

    ifc_bar = _create_ifc_entity(
        model, "IfcReinforcingBar",
        name=elem.ifc_name,
        predefined_type=BAR_TYPE_TO_IFC[elem.bar_type],
    )
    ifc_bar.NominalDiameter = float(elem.diameter)
    if elem.mark:
        ifc_bar.Tag = elem.mark

    _finish_v15_product(model, context, ifc_bar, elem, result,
                        _v15_rep_type(result), cache)

    # Derived quantities: length from the (possibly bent) centerline path,
    # mass from the registry density (see docstring for the pinned source).
    pts = [(p.x, p.y, p.z) for p in elem.path]
    length_m = _path_length_with_fillets(pts, elem.bend_radius)
    quantities = [model.create_entity(
        "IfcQuantityLength", Name="Length", LengthValue=float(length_m))]
    density = None
    from lite_step.materials import registry_definition
    mdef = registry_definition(elem.material) if isinstance(elem.material, str) else None
    if mdef is not None:
        density = mdef.psets.get("Pset_MaterialCommon", {}).get("MassDensity")
    if density is not None:
        _ensure_si_unit(model, "MASSUNIT", "GRAM", prefix="KILO")
        area = math.pi * radius * radius
        mass_kg = length_m * area * float(density)
        quantities.append(model.create_entity(
            "IfcQuantityWeight", Name="Weight", WeightValue=float(mass_kg)))
    else:
        logger.warning(
            "Bar %r: no MassDensity in registry Pset_MaterialCommon for "
            "%r — Weight quantity skipped",
            elem.ifc_name, elem.material,
        )
    element_quantity = model.create_entity(
        "IfcElementQuantity",
        GlobalId=ifcopenshell.guid.new(),
        Name="Qto_ReinforcingElementBaseQuantities",
        Quantities=quantities,
    )
    model.create_entity(
        "IfcRelDefinesByProperties",
        GlobalId=ifcopenshell.guid.new(),
        RelatedObjects=[ifc_bar],
        RelatingPropertyDefinition=element_quantity,
    )
    return ifc_bar


def _create_element_generic(model, context, elem: 'Element', cache: EntityCache):
    """Whitelisted-class generic SEMANTIC wrapper (v1.5 Element).

    Children (Solid, Sweep, Revolve, Pipe, Bar, Mesh) aggregate under the
    product, same pattern as the semantic containers. No children is legal:
    the product carries identity + props with no representation.

    Emission (output schema is IFC4X3_ADD2 — schema_version.py): EVERY
    whitelisted class emits natively, including the IFC4.3 groundworks/
    landscape vocabulary (IfcEarthworksFill, IfcCourse, IfcKerb, …).
    ``predefined_type`` is set natively when the class's enum knows the
    value, else ``USERDEFINED`` + ``ObjectType`` (the ``_create_ifc_entity``
    fallback — the sanctioned mechanism, never a silent drop). The canonical
    terrain wrapper (IfcGeographicElement + TERRAIN) styles its un-materialed
    Mesh children as opaque terrain.

    Registry material on the WRAPPER (— the stenkant form:
    ``Element(IfcEarthworksFill, material="Gravel_16-32")`` holding Mesh
    patches): the IfcMaterial + Psets associate to the wrapper product, and
    the grade's render colour is passed down to un-materialed Mesh children
    (solids keep their own material= channel untouched).

    MESH children merge INTO the wrapper's own representation: each mesh
    becomes an ``IfcTriangulatedFaceSet`` item of ONE Body/Tessellation
    shape on the wrapper product. The terrain wrapper therefore carries its
    geometry DIRECTLY on the ``IfcGeographicElement`` — no semantically
    blank ``IfcBuildingElementProxy`` riding along. Solid children keep
    the aggregation pattern — they have their own boolean/placement
    machinery.
    """
    assert np is not None
    ifc_el = _create_ifc_entity(model, elem.ifc_class, name=elem.ifc_name,
                                predefined_type=elem.predefined_type)
    _set_placement(model, ifc_el, np.eye(4), cache)

    # Wrapper-level registry material: associate IfcMaterial + Psets to the
    # wrapper product; keep the render for mesh-child colour inheritance.
    registry_render = None
    wrapper_mat = _registry_material_selection(elem)
    if wrapper_mat is not None:
        from lite_step.materials import registry_definition
        wrapper_mdef = registry_definition(wrapper_mat.key)
        ifc_material = cache.get_or_create_ifc_material(wrapper_mat.key, wrapper_mdef)
        cache.associate_material(ifc_el, ifc_material)
        if wrapper_mdef is not None and not getattr(elem, "color", None):
            registry_render = wrapper_mdef.render

    # Layered planar Element (v15.2): single body child → per-layer slices +
    # usage chain (planar ifc_class enforced at construction).
    sliced = _layered_children(elem)
    if elem.layers is not None:
        from lite_step.compiler.layers import layer_outer_at_max
        _axis = _layer_direction_axis(elem)
        _outer_at_max = layer_outer_at_max(elem)
        _offset, _scale = _layer_datum_for_container(elem, _axis, _outer_at_max)
        cache.associate_layer_usage(
            ifc_el, elem.layers, _axis, _offset, _scale,
            outer_at_max=_outer_at_max)

    terrain_style = tx.is_terrain_wrapper(elem)
    child_products = []
    mesh_items = []
    mesh_children = []
    for child in (sliced if sliced is not None else elem.elements):
        if tx.is_opening(child):
            # A Window/Door is not a child product of this wrapper — it is a
            # HOLE through the leaves, emitted by
            # ``_process_container_openings`` below. Reaching
            # ``_create_child_product`` with one produced the "no IFC emitter
            # and was skipped" warning that turned into a refusal.
            continue
        if isinstance(child, Mesh):
            face_set = _mesh_face_set(model, child)
            _style_mesh_item(model, ifc_el, face_set, child, cache,
                             terrain_style=terrain_style,
                             registry_render=registry_render)
            mesh_items.append(face_set)
            mesh_children.append(child)
            continue
        product = None
        if tx.is_prism(child):
            product = _create_solid(model, context, child, cache)
        elif isinstance(child, Sweep):
            product = _create_sweep(model, context, child, cache)
        elif isinstance(child, Revolve):
            product = _create_revolve(model, context, child, cache)
        elif isinstance(child, Pipe):
            product = _create_pipe(model, context, child, cache)
        elif isinstance(child, Element):
            # A nested Element is an ANCHORED SUB-ASSEMBLY — the one child type
            # ``.anchor()`` accepts that ``.add()`` does not
            # (``Element._ALLOWED_ANCHOR_CHILDREN``). It aggregates like any
            # other child product, so the IFC tree mirrors the authoring tree:
            # the assembly stays one addressable product with its own identity,
            # material and props instead of dissolving into loose solids.
            product = _create_element_generic(model, context, child, cache)
        else:
            # Any other type — a semantic container anchored into this one.
            product = _create_child_product(model, context, child, cache)
            if product is None:
                logger.warning(
                    "Element %r (%s): child %s %r has no IFC emitter and was "
                    "skipped",
                    elem.ifc_name, elem.ifc_class, type(child).__name__,
                    getattr(child, 'ifc_name', None) or '<anonymous>',
                )
                continue
        if product is not None:
            child_products.append(product)
            cache.record_child_product(elem, child, product)
        else:
            _warn_child_dropped(f"Element({elem.ifc_class})", elem.ifc_name, child)

    if mesh_items:
        shape_rep = model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=context,
            RepresentationIdentifier="Body",
            RepresentationType="Tessellation",
            Items=mesh_items,
        )
        ifc_el.Representation = model.create_entity(
            "IfcProductDefinitionShape", Representations=[shape_rep])
        # One combined LiteStep_MeshVolume on the wrapper (per-mesh psets
        # would collide on the pset name; the counts are informational).
        props = [
            model.create_entity("IfcPropertySingleValue",
                Name="VertexCount",
                NominalValue=model.create_entity("IfcInteger",
                    wrappedValue=sum(len(c.vertices) for c in mesh_children))),
            model.create_entity("IfcPropertySingleValue",
                Name="FaceCount",
                NominalValue=model.create_entity("IfcInteger",
                    wrappedValue=sum(len(c.faces) for c in mesh_children))),
            model.create_entity("IfcPropertySingleValue",
                Name="IsWatertight",
                NominalValue=model.create_entity("IfcBoolean",
                    wrappedValue=all(c.is_watertight for c in mesh_children))),
        ]
        pset = model.create_entity("IfcPropertySet",
            GlobalId=ifcopenshell.guid.new(), Name="LiteStep_MeshVolume",
            HasProperties=props)
        model.create_entity("IfcRelDefinesByProperties",
            GlobalId=ifcopenshell.guid.new(), RelatedObjects=[ifc_el],
            RelatingPropertyDefinition=pset)

    if child_products:
        _assign_aggregate(model, ifc_el, child_products)
    _process_container_voids(model, context, elem, ifc_el, cache)
    _process_container_openings(model, context, elem, ifc_el, cache)
    return ifc_el


# ---------------------------------------------------------------- path Sweep

def _resolve_profile_section(elem: 'Sweep'):
    """Section source for a path-form Sweep, in meters.

    Returns ``("poly", [(x, y)])`` for an explicit section= (points are
    int-mm by RULE 2, so they survive the writer's mm rounding), or
    ``("rect", w, h, key)`` for a member Material carrying profile_mm —
    emitted as IfcRectangleProfileDef because a centered polygon would put
    HALF-mm coordinates (e.g. 45/2 = 22.5 mm) into IfcCartesianPoints,
    which the post-pass deduplicator rounds to whole mm; XDim/YDim are
    whole-mm dimensions and round-trip exactly. Returns None when no
    section source exists (validate_project_report rejects that).

    ``profile_rotation`` is NOT applied here — the placement axes rotate
    instead (see _profile_path_items), for the same mm-rounding reason.
    """
    if elem.profile is not None:
        return ("poly", [(float(p.x), float(p.y)) for p in elem.profile])
    mat = _registry_material_selection(elem)
    if mat is not None and mat.profile_mm is not None:
        return ("rect", mat.profile_mm[0] / 1000.0, mat.profile_mm[1] / 1000.0,
                mat.key)
    return None


def _create_section_profile_def(model, section_2d):
    """IfcArbitraryClosedProfileDef from [(x, y)] meter tuples."""
    points = [model.create_entity("IfcCartesianPoint",
                                  Coordinates=[float(x), float(y)])
              for (x, y) in section_2d]
    points.append(points[0])
    polyline = model.create_entity("IfcPolyline", Points=points)
    return model.create_entity("IfcArbitraryClosedProfileDef",
                               ProfileType="AREA", OuterCurve=polyline)


def _miter_half_space(model, vertex, normal):
    """IfcHalfSpaceSolid removing the side of ``vertex`` the ``normal``
    points AWAY from (AgreementFlag=True keeps material below the plane —
    measured behavior, matching the sky-void cutter usage).

    An explicit in-plane ``RefDirection`` is supplied: without it OCC derives
    the plane's local X from the axis, and for a DIAGONAL normal (a mitered
    joint, a corner lop) that derivation composes with the product placement
    into a degenerate frame that tessellates to EMPTY geometry. Deriving a
    stable RefDirection perpendicular to the normal makes diagonal clips
    robust (axis-aligned normals were unaffected)."""
    n = np.asarray(normal, dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn > 0:
        n = n / nn
    # in-plane reference: cross the normal with whichever world axis it is
    # least parallel to (avoids the degenerate cross when n ~ that axis).
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    ref = np.cross(n, helper)
    ref = ref / np.linalg.norm(ref)
    loc = model.create_entity("IfcCartesianPoint",
                              Coordinates=[float(v) for v in vertex])
    axis = model.create_entity("IfcDirection",
                               DirectionRatios=[float(v) for v in n])
    ref_dir = model.create_entity("IfcDirection",
                                  DirectionRatios=[float(v) for v in ref])
    placement = model.create_entity("IfcAxis2Placement3D", Location=loc,
                                    Axis=axis, RefDirection=ref_dir)
    plane = model.create_entity("IfcPlane", Position=placement)
    return model.create_entity("IfcHalfSpaceSolid", BaseSurface=plane,
                               AgreementFlag=True)


def _apply_clips(model, solid, clips, *, center=(0.0, 0.0, 0.0), axes=None):
    """Wrap ``solid`` in one ``IfcBooleanClippingResult`` per ``.clip()``
    ``HalfSpace`` in ``clips`` — the half-space cut.

    ``center`` translates the plane origin into the receiver's local frame
    (world minus center); ``axes`` — an optional ``(x_axis, y_axis, z_axis)``
    triple of unit rows — additionally rotates origin AND normal into a
    contour-local frame. The DSL normal points at the REMOVED side;
    ``_miter_half_space`` removes the side its normal points AWAY from, so the
    NEGATED normal is passed (pinned by ``test_clips`` sign tests)."""
    if not clips:
        return solid
    result = solid
    cx, cy, cz = center
    for hs in clips:
        o = [hs.origin.x - cx, hs.origin.y - cy, hs.origin.z - cz]
        n = list(hs.normal)
        if axes is not None:
            o = [float(np.dot(o, ax)) for ax in axes]
            n = [float(np.dot(n, ax)) for ax in axes]
        # OCC needs a UNIT plane normal or it tessellates the clip to empty
        # geometry; the DSL normal is unit-free, so normalize here.
        mag = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
        n = [c / mag for c in n]
        half = _miter_half_space(model, o, [-c for c in n])
        result = model.create_entity(
            "IfcBooleanClippingResult", Operator="DIFFERENCE",
            FirstOperand=result, SecondOperand=half)
    return result


def _planar_frame_prisms(model, pts, section, cache, vertical_ref):
    """Boolean-free member prisms for a planar closed rect-section loop.

    Each member of a planar mitered frame is a PRISM: a quad in the frame
    plane — its corners are the offset-polygon points at ±half the in-plane
    section width, which IS the mitered joint — extruded through the plane
    by the section's through-plane dimension. The union of the members
    tiles the frame exactly (picture-frame identity), with zero booleans.

    Returns a list of IfcExtrudedAreaSolid, or ``None`` when the loop is
    not planar / the section is not plane-aligned — the caller then falls
    back to the CSG miter path.
    """
    assert np is not None
    _kind, sec_w, sec_h, _sec_name = section
    vertices = [np.array(p, dtype=np.float64) for p in pts[:-1]]
    n = len(vertices)
    if n < 3:
        return None

    # Loop plane via Newell's method (robust for any simple polygon).
    normal = np.zeros(3)
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        normal[0] += (a[1] - b[1]) * (a[2] + b[2])
        normal[1] += (a[2] - b[2]) * (a[0] + b[0])
        normal[2] += (a[0] - b[0]) * (a[1] + b[1])
    nrm = np.linalg.norm(normal)
    if nrm < 1e-12:
        return None  # degenerate (collinear) loop
    normal /= nrm
    for v in vertices[1:]:
        if abs(float(np.dot(v - vertices[0], normal))) > 1e-6:
            return None  # not planar

    # Segment directions + section orientation. The extrusion axes
    # convention (_segment_axes) maps section X (sec_w) along x_axis; the
    # prism needs one section dim exactly along the plane normal on EVERY
    # segment — mixed or tilted orientations fall back to CSG.
    dirs = []
    for i in range(n):
        d = vertices[(i + 1) % n] - vertices[i]
        ln = np.linalg.norm(d)
        if ln < 1e-12:
            return None
        dirs.append(d / ln)

    through_dim = None
    for d in dirs:
        x_axis, y_axis = _segment_axes(d, vertical_ref)
        ax_n = abs(float(np.dot(np.asarray(x_axis, dtype=np.float64), normal)))
        ay_n = abs(float(np.dot(np.asarray(y_axis, dtype=np.float64), normal)))
        if ax_n > 1.0 - 1e-9 and ay_n < 1e-9:
            seg_through = sec_w
        elif ay_n > 1.0 - 1e-9 and ax_n < 1e-9:
            seg_through = sec_h
        else:
            return None  # section tilted vs the plane
        if through_dim is None:
            through_dim = seg_through
        elif abs(seg_through - through_dim) > 1e-12:
            return None  # inconsistent orientation across members
    if through_dim is None:
        return None  # unreachable (n >= 3) — keeps the type-checker honest
    half_in = (sec_w + sec_h - through_dim) / 2.0  # the in-plane dim / 2

    # In-plane edge normals + standard miter offset-polygon corners.
    q = [np.cross(normal, d) for d in dirs]
    plus_c, minus_c = [], []
    for i in range(n):
        q_prev, q_cur = q[(i - 1) % n], q[i]
        denom = 1.0 + float(np.dot(q_prev, q_cur))
        if abs(denom) < 1e-9:
            return None  # 180-degree reversal
        m = (q_prev + q_cur) / denom
        plus_c.append(vertices[i] + m * half_in)
        minus_c.append(vertices[i] - m * half_in)

    # Shared placement basis: origin on the loop plane shifted back by half
    # the through-plane depth; Axis = plane normal; RefDirection = first
    # segment direction. Sweep coordinates are 2D projections onto
    # (e1, e2) = (dirs[0], normal x dirs[0]).
    e1 = dirs[0]
    e2 = np.cross(normal, e1)
    base = vertices[0] - normal * (through_dim / 2.0)

    def to2d(p3):
        rel = p3 - base
        return (float(np.dot(rel, e1)), float(np.dot(rel, e2)))

    loc = model.create_entity(
        "IfcCartesianPoint", Coordinates=[float(v) for v in base])
    axis_dir = model.create_entity(
        "IfcDirection", DirectionRatios=[float(v) for v in normal])
    ref_dir = model.create_entity(
        "IfcDirection", DirectionRatios=[float(v) for v in e1])
    position = model.create_entity(
        "IfcAxis2Placement3D", Location=loc, Axis=axis_dir,
        RefDirection=ref_dir)

    items = []
    for i in range(n):
        j = (i + 1) % n
        quad = [to2d(plus_c[i]), to2d(plus_c[j]),
                to2d(minus_c[j]), to2d(minus_c[i])]
        # Normalize winding to CCW (positive shoelace area): consumers do
        # NOT normalize IfcArbitraryClosedProfileDef winding, and a CW
        # profile extrudes to an inside-out solid whose signed volume
        # cancels its CCW siblings at tessellation (found: 2-of-4 frame
        # members inverted -> near-zero measured volume).
        area2 = sum(quad[k][0] * quad[(k + 1) % 4][1]
                    - quad[(k + 1) % 4][0] * quad[k][1] for k in range(4))
        if area2 < 0:
            quad.reverse()
        poly_pts = [model.create_entity("IfcCartesianPoint",
                                        Coordinates=[x, y])
                    for (x, y) in quad]
        poly = model.create_entity(
            "IfcPolyline", Points=poly_pts + [poly_pts[0]])
        prof = model.create_entity(
            "IfcArbitraryClosedProfileDef", ProfileType="AREA",
            OuterCurve=poly)
        items.append(model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=prof,
            Position=position,
            ExtrudedDirection=cache.get_z_up_direction(),
            Depth=float(through_dim),
        ))
    return items


def _profile_path_items(model, path_pts, elem: 'Sweep', cache: EntityCache,
                        vertical_ref=(1.0, 0.0, 0.0)):
    """Geometry items for a path-form Sweep (world/meters path points).

    * Straight 2-point path -> one plain IfcExtrudedAreaSolid.
    * Multi-point path -> one extrusion per segment, MITERED at each joint:
      the segment overshoots the joint, then an IfcHalfSpaceSolid at the
      joint's angle-bisector plane trims it — the two segments meet exactly
      at the plane (no overlap, no gap). A closed loop (first == last
      point) miters the wrap joint too.
    * fillet_radius > 0 (open paths; construction rejects closed+fillet):
      corners are faceted into <=15 deg arc chords first, then mitered.

    Returns (items, has_booleans).
    """
    assert np is not None
    section = _resolve_profile_section(elem)
    if section is None:
        logger.warning(
            "Sweep %r: path form without a section source — skipped "
            "(validate_project_report rejects this at compile)",
            elem.ifc_name,
        )
        return [], False

    pts = [(float(p[0]), float(p[1]), float(p[2])) if isinstance(p, tuple)
           else (float(p.x), float(p.y), float(p.z))
           for p in path_pts]
    closed = len(pts) >= 4 and pts[0] == pts[-1]
    fillet_m = getattr(elem, 'fillet_radius', 0) or 0
    if fillet_m and not closed:
        pts = _facet_fillet_path(pts, fillet_m)

    # Planar closed rect-section loop (the mitered frame): use the
    # boolean-free prism construction. Each member of a planar frame is a
    # PRISM — a trapezoid in the frame plane (long outer edge, short inner
    # edge, mitered ends at the offset-polygon corners) extruded through
    # the plane by the section's through-plane dimension. CSG half-space
    # mitering is exact on paper but OpenCASCADE's evaluation of
    # exactly-meeting trimmed solids is fragile (sub-mm coordinate
    # variations flip it between clean and degenerate — found via the
    # window-frame path producing sliver volumes from identical-looking
    # entities), and the mm-rounding dedup pass compounds it. The prism
    # form has no booleans at all: simple, deterministic, consumer-friendly.
    if closed and not fillet_m and section[0] == "rect" \
            and not (getattr(elem, 'profile_rotation', 0) or 0):
        planar_items = _planar_frame_prisms(
            model, pts, section, cache, vertical_ref)
        if planar_items is not None:
            return planar_items, False

    # Max radial extent of the section: bounds the miter overshoot.
    if section[0] == "rect":
        _kind, sec_w, sec_h, sec_name = section
        s_max = math.hypot(sec_w / 2.0, sec_h / 2.0)
        profile_def = cache.get_or_create_rect_profile(
            x_dim=sec_w, y_dim=sec_h, name=sec_name)
    else:
        section_2d = section[1]
        s_max = max(math.hypot(x, y) for (x, y) in section_2d)
        profile_def = _create_section_profile_def(model, section_2d)

    if closed:
        vertices = pts[:-1]
        n_seg = len(vertices)
        seg_of = lambda i: (vertices[i], vertices[(i + 1) % n_seg])  # noqa: E731
    else:
        n_seg = len(pts) - 1
        seg_of = lambda i: (pts[i], pts[i + 1])  # noqa: E731

    dirs = []
    for i in range(n_seg):
        a, b = seg_of(i)
        d = np.array([b[0] - a[0], b[1] - a[1], b[2] - a[2]], dtype=np.float64)
        dirs.append(d / np.linalg.norm(d))

    def joint_data(i_prev, i_next, vertex):
        """Miter plane normal + overshoot for the joint between segments."""
        d_in, d_out = dirs[i_prev], dirs[i_next]
        m = d_in + d_out
        norm = np.linalg.norm(m)
        if norm < 1e-9:
            return None  # collinear reversal — construction already rejects
        m = m / norm
        cos_t = max(-1.0, min(1.0, float(np.dot(d_in, d_out))))
        theta = math.acos(cos_t)
        overshoot = s_max * math.tan(theta / 2.0) * 1.05 + 0.001 if theta > 1e-9 else 0.0
        return m, overshoot, vertex

    items = []
    has_booleans = False
    for i in range(n_seg):
        a, b = seg_of(i)
        d = dirs[i]
        seg_len = float(np.linalg.norm(np.array(b) - np.array(a)))
        # Joints at this segment's start/end (None at open-path free ends).
        start_joint = None
        end_joint = None
        if closed:
            start_joint = joint_data((i - 1) % n_seg, i, a)
            end_joint = joint_data(i, (i + 1) % n_seg, b)
        else:
            if i > 0:
                start_joint = joint_data(i - 1, i, a)
            if i < n_seg - 1:
                end_joint = joint_data(i, i + 1, b)

        ext_a = start_joint[1] if start_joint else 0.0
        ext_b = end_joint[1] if end_joint else 0.0
        seg_start = (a[0] - d[0] * ext_a, a[1] - d[1] * ext_a, a[2] - d[2] * ext_a)

        # profile_rotation rotates the PLACEMENT axes about the path axis
        # (not the section points) — keeps profile coordinates whole-mm exact.
        x_axis, y_axis = _segment_axes(
            d, vertical_ref, getattr(elem, 'profile_rotation', 0) or 0)
        loc = model.create_entity("IfcCartesianPoint",
                                  Coordinates=[float(v) for v in seg_start])
        axis = model.create_entity("IfcDirection",
                                   DirectionRatios=[float(v) for v in d])
        ref_dir = model.create_entity("IfcDirection",
                                      DirectionRatios=[float(v) for v in x_axis])
        position = model.create_entity("IfcAxis2Placement3D", Location=loc,
                                       Axis=axis, RefDirection=ref_dir)
        solid = model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile_def,
            Position=position,
            ExtrudedDirection=cache.get_z_up_direction(),
            Depth=float(seg_len + ext_a + ext_b),
        )

        # Joint-overlap epsilon (meters): each segment keeps a 0.5 mm band
        # PAST the exact bisector plane, so neighbouring segments genuinely
        # OVERLAP at every joint instead of meeting exactly on a shared
        # plane. OpenCASCADE's fuse of exactly-coplanar solids is fragile —
        # it flips between clean and broken (open shells whose signed
        # volumes cancel at tessellation) on 1-ULP coordinate differences,
        # which is why the identical frame compiled correctly from
        # /1000-rounded proj coordinates but collapsed to slivers from
        # origin+offset opening coordinates. Overlapping operands make the
        # union robust; the band lies inside the joint material, so the
        # outer surface and the union volume are unchanged.
        _JOINT_EPS = 5e-4
        result = solid
        if start_joint is not None:
            # Remove the back side: {(p - vertex) . m < 0} belongs to the
            # previous segment -> half-space with normal +m, its plane
            # shifted EPS into the previous segment (this segment keeps
            # the overlap band).
            m, _o, vertex = start_joint
            v_eps = (vertex[0] - m[0] * _JOINT_EPS,
                     vertex[1] - m[1] * _JOINT_EPS,
                     vertex[2] - m[2] * _JOINT_EPS)
            result = model.create_entity(
                "IfcBooleanResult", Operator="DIFFERENCE",
                FirstOperand=result,
                SecondOperand=_miter_half_space(model, v_eps, m))
            has_booleans = True
        if end_joint is not None:
            # Remove the overshoot: {(p - vertex) . m > 0} belongs to the
            # next segment -> half-space with normal -m, its plane shifted
            # EPS into the next segment.
            m, _o, vertex = end_joint
            v_eps = (vertex[0] + m[0] * _JOINT_EPS,
                     vertex[1] + m[1] * _JOINT_EPS,
                     vertex[2] + m[2] * _JOINT_EPS)
            result = model.create_entity(
                "IfcBooleanResult", Operator="DIFFERENCE",
                FirstOperand=result,
                SecondOperand=_miter_half_space(model, v_eps, -m))
            has_booleans = True
        items.append(result)

    # Union the mitered segments into ONE boolean chain: the IFC4 'CSG'
    # representation carries a single item, and multi-item CSG reps
    # tessellate inconsistently (product-level vs item-level) in consumers.
    # The segments meet exactly at the shared miter planes, so the union is
    # well-defined.
    # BALANCED union: a 20-segment or filleted path would stack 20 nested
    # UNIONs on top of the per-segment miter cuts, which alone can blow the
    # viewer's boolean-depth tolerance. ceil(log2(n)) instead of n.
    if len(items) > 1:
        items = [_balanced_union_ios(model, items)]
        has_booleans = True
    return items, has_booleans


def _create_profile_path(model, context, elem: 'Sweep', cache: EntityCache):
    """IfcMember with path-form Sweep geometry (world coordinates)."""
    items, has_booleans = _profile_path_items(model, elem.path, elem, cache)
    if not items:
        return None
    ifc_member = _create_ifc_entity(model, "IfcMember", name=elem.ifc_name)
    assert np is not None
    _set_placement(model, ifc_member, np.eye(4), cache)

    # Element-level booleans (cuts/adds) on multi-item profiles: cut every
    # item; union into the first item only (a union into each would repeat
    # the operand's shape once per item).
    if (elem._cuts or elem._adds or getattr(elem, "_intersects", None)
            or getattr(elem, "_clips", None)):
        aabb_min, aabb_max = _path_aabb(elem.path, 0.0)
        new_items = []
        _build = lambda e, is_cut=True: _create_hole_solid_local(  # noqa: E731
            model, e, aabb_min, (0.0, 0.0, 0.0), cache, parent_max=aabb_max, is_cut=is_cut)
        for idx, item in enumerate(items):
            result = _apply_boolean_chain_ios(
                model, item, build=_build,
                cuts=(elem._cuts or []),
                # union into the FIRST item only — a union into each would
                # repeat the operand's shape once per item.
                adds=((elem._adds or []) if idx == 0 else []),
                intersects=(getattr(elem, "_intersects", None) or []))
            if result is not item:
                has_booleans = True
            clips = getattr(elem, "_clips", None) or []
            if clips:
                result = _apply_clips(model, result, clips)   # world frame
                has_booleans = True
            new_items.append(result)
        items = new_items

    rep_type = "CSG" if has_booleans else "SweptSolid"
    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType=rep_type,
        Items=items,
    )
    _assign_body_representation(model, ifc_member, shape_rep)
    _style_profile_items(model, ifc_member, items, elem, cache)
    if getattr(elem, "_voids", None):
        _process_voids_on_element(model, context, ifc_member, elem, cache)
    return ifc_member


def _style_profile_items(model, product, items, elem, cache: EntityCache):
    """Material association + per-item styling for multi-item profiles."""
    styled = _apply_registry_material(model, product, items[0], elem, cache)
    if styled and len(items) > 1:
        mat = _registry_material_selection(elem)
        from lite_step.materials import registry_definition
        mdef = registry_definition(mat.key) if mat is not None else None
        if mdef is not None:
            rgb_hex = mdef.render.rgb.lstrip("#")
            rgb = (int(rgb_hex[0:2], 16), int(rgb_hex[2:4], 16), int(rgb_hex[4:6], 16))
            transparency = 1.0 - mdef.render.alpha
            for item in items[1:]:
                _apply_surface_color(model, item, rgb, cache, transparency)
    elif not styled:
        color_key = getattr(elem, 'color', None) or ''
        if color_key:
            for item in items:
                _apply_typed_color(model, item, color_key, cache)


def _create_opening_profile_path(model, context, profile_elem,
                                 origin_x, origin_y, origin_z,
                                 cos_a, sin_a, cache: EntityCache,
                                 parent_fill=None):
    """Path-form Sweep child of a Window/Door (the mitered frame).

    Path points are opening-local (meters, Z-up: X across the opening,
    Y through the wall, Z up from the sill).

    GEOMETRY LIVES IN THE PARENT WINDOW'S LOCAL FRAME, with the member's
    ObjectPlacement chained (identity-relative) to the window's placement.
    This is the standard IFC decomposition pattern — and the ONLY shape
    ifcopenshell resolves correctly here: a world-coordinate child with an
    unchained placement inside an IfcRelAggregates whose RelatingObject has
    no Representation tessellates to garbage (the mitered frame measured
    ~1/87 of its true volume from a bit-identical entity subgraph that was
    exact outside the aggregation), and non-identity RelativePlacements
    re-trigger the same mis-resolution. Identity-relative chaining resolves
    natively on every wall rotation.
    """
    assert np is not None
    world_pts = []
    for p in profile_elem.path:
        world_pts.append((
            origin_x + p.x * cos_a + p.y * (-sin_a),
            origin_y + p.x * sin_a + p.y * cos_a,
            origin_z + p.z,
        ))

    parent_placement = getattr(parent_fill, "ObjectPlacement", None)         if parent_fill is not None else None
    if parent_placement is not None:
        import ifcopenshell.util.placement as _plc
        parent_abs = _plc.get_local_placement(parent_placement)
        inv = np.linalg.inv(parent_abs)
        local_pts = []
        for w in world_pts:
            v = inv @ np.array([w[0], w[1], w[2], 1.0])
            local_pts.append((float(v[0]), float(v[1]), float(v[2])))
        # In the window frame the wall runs along local +X — the section
        # orientation reference is constant.
        build_pts, vref = local_pts, (1.0, 0.0, 0.0)
    else:
        # ``cos_a``/``sin_a`` ARE the wall's run axis, carried down from the
        # OpeningFrame — i.e. from ``compiler.frames.opening_axes``, the same
        # single derivation every other consumer reads. Passing it as the
        # section's orientation reference is what keeps
        # ``Material(profile_mm=(95, 45))`` meaning "95 = through the wall"
        # on all four frame members of a rotated wall.
        build_pts, vref = world_pts, (float(cos_a), float(sin_a), 0.0)

    items, has_booleans = _profile_path_items(
        model, build_pts, profile_elem, cache, vertical_ref=vref)
    if not items:
        return None
    ifc_member = _create_ifc_entity(model, "IfcMember",
                                    name=profile_elem.ifc_name)
    _set_placement(model, ifc_member, np.eye(4), cache)
    if parent_placement is not None:
        ifc_member.ObjectPlacement.PlacementRelTo = parent_placement
    rep_type = "CSG" if has_booleans else "SweptSolid"
    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType=rep_type,
        Items=items,
    )
    _assign_body_representation(model, ifc_member, shape_rep)
    _style_profile_items(model, ifc_member, items, profile_elem, cache)
    return ifc_member
