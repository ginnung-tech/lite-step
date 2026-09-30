"""SPACE BOUNDARIES — ``IfcRelSpaceBoundary1stLevel``, one per
(space, bounding element) pair. DECLARED on a bounds-mode space (WS-F),
INFERRED on a geometry-mode one (WS-C).

The ONE place the emitter reads, mirroring :mod:`lite_step.ifc.grids`,
:mod:`lite_step.ifc.groupings`, :mod:`lite_step.ifc.product_types` and
:mod:`lite_step.ifc.storeys`. Two implementations of one idea agreeing *usually* is what produces the bug, so every policy decision — which
elements are candidates, what counts as adjacency, both enum classifications,
which entity is minted and what it is called — is decided HERE; ``generator.py``
contributes only the lines that mint an entity.

The two modes (WS-F, roadmap §WS-F)
-----------------------------------

``Space(name="kitchen", bounds=["wall:north", "slab:floor", ...])`` declares
its boundary set outright: the declared members ARE the
``IfcRelSpaceBoundary1stLevel`` set — authoritative, not inferred, each one
emitted wherever the member sits (a chimney breast poking into the room is a
FEATURE, not a dropped boundary). The volume such a space carries is DERIVED
from the same declaration (:mod:`lite_step.compiler.space_volume`), and only
three things here stay computed, because they are not author statements:

* ``InternalOrExternalBoundary`` — what lies BEYOND an element
  (:func:`_classify`, unchanged);
* Window/Door fillings of declared hosts — their inner boundaries derive
  exactly as in geometry mode (fillings are never declared);
* the CHECKER — the WS-C contact engine demoted to a warnings channel on
  bounds-mode spaces (:func:`_check_declared_set`): a declared bound the
  derived volume stops touching (a boolean carved the room away from it)
  warns, and an undeclared element that passes the contact test warns as a
  candidate member.

Geometry-mode spaces (``.add(Box(...))`` — LOD 100 zones with no walls to
cite) keep the full WS-C inference below, unchanged. ``collect_boundaries``
remains the one entry point; only the collection half forks on the mode.

Why this needs its own gate
---------------------------

``IfcRelSpaceBoundary`` has no mandatory inverse, so a model that emits none
is schema-valid; a conformance gate cannot miss an entity that was never meant
to exist, and a byte-hash sees no drift in output that was never produced. An
unwritten relation is the same silent class as a dropped OPTIONAL
(:mod:`lite_step.ifc.storeys`): the only gate that can exist is a test that
asserts the relation is there.

Reading the schema before writing the solver
--------------------------------------------

IFC 4.3.2 (IFC4X3_ADD2), 5.4.3.59 / 5.4.3.60. ``ConnectionGeometry`` is
**OPTIONAL** on ``IfcRelSpaceBoundary`` and on both subtypes, and the concept
text is explicit about what omitting it means:

    "The attribute ConnectionGeometry may be inserted, in this case it
    describes the physical space boundary geometrically, or it may be omitted;
    in that case it describes a physical space boundary logically."

So a conformant boundary needs ``RelatingSpace``, ``RelatedBuildingElement``,
``PhysicalOrVirtualBoundary`` and ``InternalOrExternalBoundary`` — an adjacency
plus two classifications. **We emit the logical form.** No surface solver, no
``IfcConnectionSurfaceGeometry``, and the four surface representations the
schema lists for 1st level (``IfcSurfaceOfLinearExtrusion``,
``IfcCurveBoundedPlane``, ``IfcCurveBoundedSurface``,
``IfcFaceBasedSurfaceModel``) are not reachable from here at all.

**Why the 1st-level SUBTYPE and not the base entity.** Both are instantiable —
``IfcRelSpaceBoundary`` is ``SUPERTYPE OF (ONEOF(IfcRelSpaceBoundary1stLevel))``,
not ABSTRACT — so this is a decision, and it has three reasons:

* The concept the schema defines for this data is *Space Boundaries 1st Level*,
  and its applicable relationship is ``IfcRelSpaceBoundary1stLevel``. What we
  compute IS the 1st-level definition, verbatim: "boundaries of the space, not
  taking into account any change in building element or spaces on the other
  side", "defined totally from inside the space", the architectural/FM view.
* The subtype is where ``ParentBoundary`` / ``InnerBoundaries`` live, and the
  concept requires opening boundaries to be linked to their host's boundary
  through it (see the Window/Door section below). On the base entity that link
  cannot be expressed, so the base entity would make the opening boundaries
  free-floating.
* ``IfcRelSpaceBoundary1stLevel`` is New-in-IFC4. Emitting the IFC2x-era base
  entity in an IFC4X3_ADD2 file, where a more specific type exists and applies,
  is the same class of choice as emitting a deprecated attribute.

**And the Name is still stated.** The schema distinguishes the levels by
``IfcRoot.Name``, independently of the entity type::

    1st level: IfcRoot.Name = "1stLevel"   IfcRoot.Description = NIL
    2nd level: IfcRoot.Name = "2ndLevel"   IfcRoot.Description = "2a" or "2b"

That convention predates the subtypes and consumers still read it, so
:data:`BOUNDARY_NAME` is written on every boundary and
:data:`BOUNDARY_DESCRIPTION` is deliberately ``None`` — ``NIL`` is the stated
1st-level value, not an omission. A ``Description`` of ``"2a"`` on a
1st-level relation would claim second-level decomposition we did not do.

PhysicalOrVirtualBoundary — always PHYSICAL, and VIRTUAL is UNREACHABLE
-----------------------------------------------------------------------

Not a hedge and not a default. The WHERE rule ``CorrectPhysOrVirt`` binds the
enum to the RELATED ELEMENT'S TYPE:

    PHYSICAL  -> RelatedBuildingElement must NOT be an IfcVirtualElement
    VIRTUAL   -> RelatedBuildingElement must BE an IfcVirtualElement or an
                 IfcOpeningElement

``IfcVirtualElement`` is not in ``ELEMENT_IFC_CLASS_WHITELIST`` (neither the
IFC4 nor the IFC4.3 half), there is no DSL container that maps to it, and the
string appears nowhere in this repository. So **no Lite-STEP model can produce
a virtual divider**, and the VIRTUAL branch of that WHERE rule is dead code
here rather than an unimplemented case.

The other VIRTUAL-eligible type, ``IfcOpeningElement``, IS emitted — every
``Window``/``Door`` mints one, as does ``.void()``. It is deliberately never a
``RelatedBuildingElement``: an opening is a void, and the thing that bounds the
space at that location is the FILLING (the ``IfcWindow``/``IfcDoor``), which is
physical. Relating the void instead would emit VIRTUAL for a boundary a person
can walk into.

Emitting NOTDEFINED "to be safe" would be strictly worse than either: the
schema says NOTDEFINED means "no information available", and we have the
information.

InternalOrExternalBoundary — four values, and none of them is a guess
---------------------------------------------------------------------

The enum's own definitions are about what lies BEYOND the bounding element:

    EXTERNAL        an external space on the other side
    EXTERNAL_EARTH  earth (or terrain) on the other side
    INTERNAL        an internal space on the other side
    NOTDEFINED      no information available

so this is answered by looking past the element, along the boundary normal,
and asking what covers the contact area (:func:`_classify`):

* **The element IS terrain** (the canonical
  ``Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN")``
  wrapper, or a Mesh inside one) -> ``EXTERNAL_EARTH``. There is earth on the
  far side of an earth surface by definition. Plain ``EXTERNAL`` here would be
  a false statement — it claims an external SPACE — which is why terrain is not
  simply lumped in with "nothing beyond".
* **Another Space covers the whole contact area** -> ``INTERNAL``. This is the
  "space pair across a wall" the reference already promised, and it is computed
  from the SAME adjacency relation: the far space is one that bounds the same
  element from the opposite side.
* **Terrain covers the whole contact area** (and no space does) ->
  ``EXTERNAL_EARTH`` — a basement wall with soil behind it.
* **Nothing covers it** -> ``EXTERNAL``.
* **Partly covered, or covered by both kinds** -> ``NOTDEFINED``. This is not a
  give-up: the 1st-level concept names exactly this case —

      "1st level space boundaries are differentiated in two ways: virtual or
      physical and internal, external, or undefined (internal and external)
      e.g. for a wall that is partially inside and outside."

  A wall bounding a room along its full length while another room sits behind
  only half of it IS partly internal and partly external, and 1st-level
  boundaries are not decomposed (that is what 2nd level is for), so
  ``NOTDEFINED`` is the accurate answer rather than a coin flip between two
  wrong ones. Coverage is exact, not sampled — see :func:`_coverage`.

Adjacency — reused, never re-implemented
-----------------------------------------

Three facts feed the test, and all three already existed:

* :func:`lite_step.compiler.extent.element_extent` — the ONE authoring-extent
  derivation, which ``BimElement.authored_aabb()`` also reads. Containers union
  per child (a ``Wall``'s box comes from its ``Box`` body), openings return
  ``None`` and are handled separately below.
* :func:`lite_step.compiler.extent.overlap_depths` — the per-axis overlap
  primitive the displacement engine's ``_overlaps`` is now built out of. That
  extraction is the whole point: a second overlap implementation that agrees
  with displacement *usually* is the shape, and space boundaries need the
  case displacement deliberately excludes (a FLUSH face, zero-depth contact),
  so a copy would have differed by exactly one comparison operator.
* :func:`lite_step.compiler.placement.resolve_placement_matrices` — the world
  matrix of a ``placement=`` element, keyed by ``id()`` over exactly the
  top-level storey/site elements this module walks.

**The contact rule** (:func:`_contact`), in three conditions:

1. **No gap on any axis.** Touching counts; a model with a 1 mm construction
   gap between wall and room has no boundary, deliberately — inferring one
   would mean inventing a tolerance for "close enough" that no author stated.
2. **At most one axis may be flush.** The other two must overlap positively, so
   the contact has AREA. Two flush axes is an EDGE touch (a wall meeting the
   room's corner diagonally) and three is a POINT touch; neither delimits
   anything, and both would otherwise pass a naive "overlaps-or-touches on
   every axis" test.
3. **At least one boundary axis** — an axis on which the element sits past
   exactly ONE of the space's two faces. This is what "delimits" means: a wall
   abutting the north face reaches past the space's ``y`` maximum and not past
   its minimum. An element that spans BOTH faces on every axis reaches past
   nothing (it merely contains or is contained), and an element that spans
   NEITHER is floating strictly inside the room. Both are excluded.

Condition 3 is also where the LIMIT of an AABB reading sits, and it is stated
rather than hidden: a free-standing column in the middle of a room, whose
surfaces DO carry finishes, produces no boundary here because its box reaches
no face of the space. Finding it needs surface adjacency, not box adjacency —
the same machinery 2nd-level boundaries need. See the closing section.

Window and Door — the inner boundaries
---------------------------------------

The 1st-level concept requires them:

    "1st level space boundaries form a closed shell around the space … and
    include overlapping boundaries representing openings (filled or not) in the
    building elements"
    "the attribute ParentBoundary with inverse InnerBoundaries is provided to
    link the space boundaries of doors, windows, and openings to the parent
    boundary, such as of a wall or slab"
    "The space boundary of the parent is not cut by the inner boundary — both
    overlap."

So a ``Window``/``Door`` in a wall that bounds a space gets its OWN boundary,
carrying ``ParentBoundary`` -> the wall's boundary for that same space, and the
wall's boundary is left whole.

An opening has no extent of its own (``element_extent`` returns ``None`` —
since v20.0.0 a Window carries a size and no position; the HOST decides where
it lands), so its world box is composed from the same
``frames.opening_host_frame`` + ``frames.opening_placement`` pair the emitter
emits the void and the fill from. It is therefore the same rectangle the file
already contains, not a second guess at it — and it means the "window at the
far end of a wall that only partly bounds the room" case answers correctly
instead of being inherited from the wall.

``ParentBoundary`` is OPTIONAL, so a filling whose host wall bounds the space
somewhere the filling does not is emitted with a NULL parent rather than being
dropped or attached to an unrelated boundary — and it says so on the warning
channel, because a parentless inner boundary is a fact about the model, not a
detail to swallow.

What 2nd level would need, and why it is NOT here
--------------------------------------------------

``IfcRelSpaceBoundary2ndLevel`` decomposes each 1st-level boundary wherever the
material of the bounding element or the space behind it changes, splits into
type 2a (a space on the other side, with ``CorrespondingBoundary`` pointing at
its twin) and 2b (a building element on the other side), and — unlike 1st level
— is only useful WITH ``ConnectionGeometry``, since thermal analysis consumes
the surfaces. That needs three things this module deliberately does not have:
real surface geometry rather than boxes (an ``IfcCurveBoundedPlane`` with
polygonal boundaries, or an ``IfcFaceBasedSurfaceModel``, in the SPACE's local
placement); the material-layer set of every bounding element resolved to
per-layer subdivisions; and a paired-boundary solver so each 2a boundary can
cite its opposite. It is a surface solver, which is exactly what the logical
1st-level form let us skip.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lite_step.compiler.extent import MM_PER_METER, Aabb, overlap_depths

logger = logging.getLogger(__name__)


class SpaceBoundaryError(ValueError):
    """A boundary this module derived that its backend could not emit.

    Always a compiler defect rather than an authoring mistake: the policy said
    "relate this space to this element" and the file has no such product. Loud,
    because the alternative is a boundary silently dropped from a relation set
    whose absence nothing else can detect.
    """


#: ``IfcPhysicalOrVirtualEnum``. VIRTUAL is unreachable from this DSL — see the
#: module docstring — so PHYSICAL is a derivation, not a default.
PHYSICAL = "PHYSICAL"

#: ``IfcInternalOrExternalEnum``. All four values this module can produce.
INTERNAL = "INTERNAL"
EXTERNAL = "EXTERNAL"
EXTERNAL_EARTH = "EXTERNAL_EARTH"
NOTDEFINED = "NOTDEFINED"

#: ``IfcRoot.Name`` / ``IfcRoot.Description`` on every emitted boundary. The
#: schema's own level discriminator (5.4.3.59.1); ``None`` IS the stated
#: 1st-level Description, not a dropped value.
BOUNDARY_NAME = "1stLevel"
BOUNDARY_DESCRIPTION: Optional[str] = None

#: The entity the emitter mints. See the module docstring for why the subtype
#: rather than the base ``IfcRelSpaceBoundary``.
BOUNDARY_ENTITY = "IfcRelSpaceBoundary1stLevel"

#: Contact tolerance, in the MODEL's units (metres — this runs post-normalize,
#: like the displacement engine). Same magnitude as
#: ``displacement._EPS`` and for the same reason: authoring is int mm, so two
#: faces an author placed at the same coordinate divide to the same double and
#: the difference is float noise, never a real gap. 1e-6 m is a micron; the
#: smallest authorable distance is a thousand times larger.
CONTACT_TOL = 1e-6


@dataclass(frozen=True)
class SpaceBoundary:
    """One ``IfcRelSpaceBoundary1stLevel``, as POLICY rather than as entities.

    Both products are cited by CANONICAL NAME — the same identity join the
    manifest, the patch differ, ``groupings`` and ``product_types`` use — so
    this carries no backend's notion of an entity and the emitter cannot
    disagree about what a boundary says.
    """

    space: str                       #: RelatingSpace, canonical name
    element: str                     #: RelatedBuildingElement, canonical name
    physical_or_virtual: str         #: IfcPhysicalOrVirtualEnum
    internal_or_external: str        #: IfcInternalOrExternalEnum
    name: str                        #: IfcRoot.Name — always "1stLevel"
    description: Optional[str]       #: IfcRoot.Description — always None
    parent_index: Optional[int]      #: index into the SAME list, or None
    #: True when the boundary comes from a bounds-mode space's DECLARED
    #: member set (WS-F) — including its fillings' inner boundaries — rather
    #: than from WS-C geometric inference. Report-channel honesty only: no
    #: backend reads it, the emitted entity is identical either way.
    declared: bool = False

    @property
    def is_inner(self) -> bool:
        """True for a Window/Door boundary that found its host's boundary."""
        return self.parent_index is not None


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Box:
    """A candidate and its world box. ``terrain`` drives EXTERNAL_EARTH.

    The three optional fields carry a bounds-mode SPACE's declaration
    (stamped by ``space_volume.materialize_bound_spaces``): the declared
    member set in authoring order, the per-member inner-face axes (the
    classification fallback for a declared bound the volume does not
    geometrically touch) and the checker's effective volume (the derived box
    with boolean end-slices trimmed). All ``None`` on geometry-mode spaces
    and on element candidates.
    """

    name: str
    aabb: Aabb
    terrain: bool = False
    host: Optional[str] = None       #: canonical name of an opening's host
    declared: Optional[Tuple[str, ...]] = None
    face_axes: Optional[Dict[str, Tuple[Tuple[int, int], ...]]] = None
    effective: Optional[Aabb] = None


def collect_boundaries(project) -> List[SpaceBoundary]:
    """Every 1st-level space boundary in ``project``, ordered.

    DECLARED on a bounds-mode space (the members, in authoring order —
    authoritative, see the module docstring), INFERRED on a geometry-mode
    one. Every HOST boundary comes before the inner boundaries that cite it,
    so ``parent_index`` always points BACKWARD and a backend can mint
    entities in one forward pass. Deterministic by construction — two
    compiles of one model emit the boundaries in the same order, which is
    what makes the corpus hash a gate rather than noise.

    Runs in the MODEL's units. The emitter calls it after
    ``normalize_project_to_meters``, so that is metres; the extent derivation
    is told so via :data:`~lite_step.compiler.extent.MM_PER_METER`, exactly as
    the displacement engine tells it.
    """
    # Bounds-mode spaces carry no authored geometry until their declared
    # volume is materialized. Idempotent — the backends' shared preamble has
    # usually run it already; this covers direct policy-level callers.
    from lite_step.compiler.space_volume import materialize_bound_spaces

    materialize_bound_spaces(project)

    spaces = _collect_spaces(project)
    if not spaces:
        return []
    elements, openings = _collect_elements(project)
    if not elements:
        return []

    hosts: List[SpaceBoundary] = []
    # (space, host element) -> index in ``hosts``; the ParentBoundary join.
    host_index: Dict[Tuple[str, str], int] = {}
    elements_by_name = {e.name: e for e in elements}

    for space in spaces:
        if space.declared is not None:
            for member in space.declared:
                elem = elements_by_name.get(member)
                if elem is None:
                    # The member resolved at materialization against the SAME
                    # candidate walk; missing here means the two walks have
                    # drifted — a compiler defect, never an authoring one.
                    raise SpaceBoundaryError(
                        f"space bounds: {space.name} declares {member!r}, "
                        f"which resolved at volume derivation but names no "
                        f"boundary candidate — the bounds walk and the "
                        f"boundary walk disagree about this model.")
                host_index[(space.name, elem.name)] = len(hosts)
                hosts.append(SpaceBoundary(
                    space=space.name,
                    element=elem.name,
                    physical_or_virtual=PHYSICAL,
                    internal_or_external=_classify_declared(
                        space, elem, member, spaces, elements),
                    name=BOUNDARY_NAME,
                    description=BOUNDARY_DESCRIPTION,
                    parent_index=None,
                    declared=True,
                ))
            _check_declared_set(space, elements)
            continue
        for elem in elements:
            contact = _contact(space.aabb, elem.aabb)
            if contact is None:
                _warn_if_flush_inside(space, elem)
                continue
            host_index[(space.name, elem.name)] = len(hosts)
            hosts.append(SpaceBoundary(
                space=space.name,
                element=elem.name,
                physical_or_virtual=PHYSICAL,
                internal_or_external=_classify(space, elem, contact,
                                               spaces, elements),
                name=BOUNDARY_NAME,
                description=BOUNDARY_DESCRIPTION,
                parent_index=None,
            ))

    inner: List[SpaceBoundary] = []
    for space in spaces:
        for opening in openings:
            if space.declared is not None and opening.host not in space.declared:
                # The declared members ARE the boundary set: a filling only
                # derives an inner boundary when its host is declared. The
                # geometric test below still decides whether THIS filling
                # stands over THIS room — exactly as in geometry mode.
                continue
            contact = _contact(space.aabb, opening.aabb)
            if contact is None:
                continue
            parent = host_index.get((space.name, opening.host))
            if parent is None:
                # Loud-failure doctrine: this is a real geometric statement —
                # the filling bounds the room at a place its host wall does
                # not — and ParentBoundary is OPTIONAL, so the boundary is
                # still emitted. Swallowing it would hide a host whose extent
                # disagrees with the opening the compiler placed in it.
                logger.warning(
                    "space boundary: %s bounds %s but its host %s does not — "
                    "emitting the inner boundary with ParentBoundary NULL",
                    opening.name, space.name, opening.host)
            inner.append(SpaceBoundary(
                space=space.name,
                element=opening.name,
                physical_or_virtual=PHYSICAL,
                internal_or_external=_classify(space, opening, contact,
                                               spaces, elements),
                name=BOUNDARY_NAME,
                description=BOUNDARY_DESCRIPTION,
                parent_index=parent,
                declared=space.declared is not None,
            ))

    return hosts + inner


def _collect_spaces(project) -> List[_Box]:
    """Every named ``Space`` with a world box, in emission order."""
    from lite_step.models import taxonomy as tx

    matrices = _matrices(project)
    out: List[_Box] = []
    for elem in _top_level(project):
        if type(elem).__name__ != "Space":
            continue
        if not tx.is_spatial(elem):                       # pragma: no cover
            continue
        name = getattr(elem, "_canonical_name", None)
        if not name:
            # An anonymous Space emits ``Name=None`` and cannot be joined to
            # by any name-keyed relation, here or in groupings. Warned rather
            # than dropped in silence — the author's room simply has no
            # boundaries and nothing else would ever say why.
            logger.warning(
                "space boundary: an anonymous Space was skipped — a boundary "
                "cites its space by canonical name, so name it to get "
                "IfcRelSpaceBoundary1stLevel relations for it")
            continue
        aabb = _world_box(elem, matrices)
        if aabb is None:
            logger.warning(
                "space boundary: Space %s has no extent (no geometry), so "
                "nothing can bound it", name)
            continue
        declared = getattr(elem, "_bounds_resolved", None)
        out.append(_Box(
            name=name, aabb=aabb,
            declared=tuple(declared) if declared is not None else None,
            face_axes=getattr(elem, "_bounds_face_axes", None),
            effective=getattr(elem, "_effective_volume", None),
        ))
    return out


def _collect_elements(project) -> Tuple[List[_Box], List[_Box]]:
    """``(bounding elements, openings)`` — every candidate ``IfcElement``.

    Candidates are exactly the elements the backends turn into TOP-LEVEL
    products: a storey's own elements and a site's own children. Their
    geometry children are not products (a ``Wall``'s ``Box`` body is merged
    into the wall's representation), so walking deeper would relate spaces to
    entities that do not exist in the file.

    Openings are separated because they are the only candidates whose box is
    not an extent — and because they become INNER boundaries, which must be
    minted after the hosts they cite.
    """
    from lite_step.models import taxonomy as tx

    matrices = _matrices(project)
    elements: List[_Box] = []
    openings: List[_Box] = []

    for elem in _top_level(project):
        if tx.is_spatial(elem) or tx.is_annotation(elem):
            continue                       # a place, or a planning primitive
        name = getattr(elem, "_canonical_name", None)
        aabb = _world_box(elem, matrices)
        if name and aabb is not None:
            elements.append(_Box(name=name, aabb=aabb,
                                 terrain=_is_terrain(elem)))
        elif aabb is not None:
            logger.warning(
                "space boundary: an anonymous %s was skipped — a boundary "
                "cites its element by canonical name",
                type(elem).__name__)
        if name is not None:
            openings.extend(_openings_of(elem, name, matrices))

    return elements, openings


def _top_level(project) -> List[Any]:
    """Every element that becomes a top-level product, storeys then sites."""
    out: List[Any] = []
    for storey in getattr(project, "storeys", None) or []:
        out.extend(storey.elements)
    for site in getattr(project, "sites", None) or []:
        out.extend(getattr(site, "_elements", None) or [])
    return out


def _matrices(project) -> Dict[int, Any]:
    """``{id(element): 4x4}`` for every ``placement=`` top-level element.

    The compiler-facing resolver, not ``BimElement.world_aabb()``: that public
    accessor raises through the ``_RESOLVING_WORLD`` re-entrancy guard when a
    compiler pass calls it, and its docstring says so ("internal compiler
    passes must use frames.* / placement.* directly"). Idempotent and keyed by
    identity, so calling it twice per compile costs a dict lookup.
    """
    from lite_step.compiler.placement import resolve_placement_matrices

    return resolve_placement_matrices(project)


def _world_box(elem, matrices: Dict[int, Any]) -> Optional[Aabb]:
    """``elem``'s extent in world coordinates, or ``None`` when it has none.

    ``MM_PER_METER`` states the unit domain (see ``extent``'s module
    docstring): a caller in metres that forgets it over-approximates every
    ``Material``-derived pad by 1000x.
    """
    from lite_step.compiler.extent import element_extent

    box = element_extent(elem, MM_PER_METER)
    if box is None:
        return None
    matrix = matrices.get(id(elem))
    if matrix is None:
        return box
    return _transform(box, matrix)


def _transform(box: Aabb, matrix) -> Aabb:
    """The world-axis-aligned bound of ``box`` under ``matrix``.

    A rotated box's AABB is larger than the box, exactly as
    ``BimElement.world_aabb()`` documents. Over-approximating is the safe
    direction here — it can add a boundary against a neighbour a rotated wall
    does not quite reach, which is visible in the relation set, where
    under-approximating would silently drop a real one.
    """
    import numpy as np

    m = np.asarray(matrix, dtype=float)
    (x0, y0, z0), (x1, y1, z1) = box
    corners = np.array([[x, y, z]
                        for x in (x0, x1)
                        for y in (y0, y1)
                        for z in (z0, z1)], dtype=float)
    placed = corners @ m[:3, :3].T + m[:3, 3]
    lo = placed.min(axis=0)
    hi = placed.max(axis=0)
    return (tuple(float(v) for v in lo), tuple(float(v) for v in hi))


def _is_terrain(elem) -> bool:
    """The canonical terrain semantic — the wrapper, or a Mesh inside one."""
    from lite_step.models import taxonomy as tx

    if tx.is_terrain_wrapper(elem):
        return True
    return any(True for _ in tx.iter_terrain_meshes(elem))


def _openings_of(host, host_name: str,
                 matrices: Dict[int, Any]) -> List[_Box]:
    """Every ``Window``/``Door`` in ``host``, with its WORLD box.

    Composed from ``frames.opening_host_frame`` + ``frames.opening_placement``
    — the one derivation the emitter emits the void and the fill placements
    from — so the rectangle tested here is the rectangle the file contains.
    An opening has no extent of its own; ``element_extent`` returns ``None``
    for one on purpose.
    """
    from lite_step.compiler import frames
    from lite_step.models import taxonomy as tx

    out: List[_Box] = []
    children = getattr(host, "_elements", None) or []
    if not any(tx.brings_void(c) for c in children):
        return out

    frame = frames.host_opening_frame(host, divisor=MM_PER_METER)
    if frame is None:
        logger.warning(
            "space boundary: %s holds openings but bears no opening frame, "
            "so their boundaries cannot be placed — the wall's own boundary "
            "still stands", host_name)
        return out

    matrix = matrices.get(id(host))
    for child in children:
        if not tx.brings_void(child):
            continue
        name = getattr(child, "_canonical_name", None)
        if not name:
            logger.warning(
                "space boundary: an anonymous %s in %s was skipped — a "
                "boundary cites its element by canonical name",
                type(child).__name__, host_name)
            continue
        box = _opening_box(child, frame, frames)
        if box is None:
            continue
        if matrix is not None:
            box = _transform(box, matrix)
        out.append(_Box(name=name, aabb=box, host=host_name))
    return out


def _opening_box(opening, frame, frames) -> Optional[Aabb]:
    """The world AABB of one opening's FILLING.

    ``corner`` is the filling's lower-left on the host's outer face, ``+x``
    right as seen from outside and ``+y`` into the body — the same rigid
    transform ``_world_matrix_for`` composes for a query inside a window. The
    box is the fill's own width x host thickness x height: a Window bounds the
    space over the hole it fills, not over the void's centre point.
    """
    placed = frames.opening_placement(opening, frame)
    cos_a, sin_a = placed.cos_a, placed.sin_a
    cx, cy, cz = placed.corner
    xs, ys = [], []
    for along in (0.0, placed.width):
        for through in (0.0, placed.thickness):
            xs.append(cx + cos_a * along - sin_a * through)
            ys.append(cy + sin_a * along + cos_a * through)
    return ((min(xs), min(ys), cz),
            (max(xs), max(ys), cz + placed.height))


# ---------------------------------------------------------------------------
# Adjacency
# ---------------------------------------------------------------------------


#: WHERE an element meets a space: the boundary axes with their outward sign
#: (``+1`` = the element sits past the space's maximum on that axis). One entry
#: is the common case; a column in a room's corner delimits on two.
Contact = Tuple[Tuple[int, int], ...]


def _contact(space: Aabb, elem: Aabb,
             tol: float = CONTACT_TOL) -> Optional[Contact]:
    """Does ``elem`` bound ``space``, and along which axes?

    Both conditions are spelled out in the module docstring. They are ONE
    function because they are one question — split across the caller they would
    become a rule nobody reads as a whole, which is how "overlaps on every
    axis" starts counting a corner touch.
    """
    depths = overlap_depths(space, elem)

    # 1. A gap on any axis: the element is simply not there.
    if any(d < -tol for d in depths):
        return None

    # 2. At least one boundary axis: the element reaches past exactly ONE of
    #    the space's two faces on it, AND the contact it makes on the plane
    #    perpendicular to that axis has AREA.
    #
    #    The area clause carries two jobs, and BOTH were measured rather than
    #    reasoned:
    #
    #    * It excludes edge and point touches outright. Two flush axes is an
    #      edge (a wall meeting the room's corner diagonally), three is a
    #      vertex; on either, every candidate axis has a flush axis in its
    #      perpendicular plane, so none survives and the pair produces no
    #      boundary. An earlier draft ALSO carried a separate "at most one
    #      flush axis" condition for this. It was redundant — a mutation that
    #      disabled it left all 66 tests green — so it is gone rather than
    #      standing as a guard that cannot fail.
    #    * It stops a real boundary manufacturing a second, empty one. A wall
    #      flush against a room's north face also reaches past that room's
    #      east face whenever it runs longer than the room — but the two boxes
    #      share ZERO area on the plane perpendicular to east, because they
    #      are flush in north. Counting it asked "what lies beyond" over an
    #      empty rectangle, and the answer (nothing there -> EXTERNAL) then
    #      disagreed with the real axis and downgraded a correct INTERNAL to
    #      NOTDEFINED.
    (s0, s1), (e0, e1) = space, elem
    axes: List[Tuple[int, int]] = []
    for k in range(3):
        if any(depths[j] <= tol for j in range(3) if j != k):
            continue                        # the contact on this plane is a line
        low = e0[k] < s0[k] - tol           # reaches past the space's minimum
        high = e1[k] > s1[k] + tol          # ...and/or past its maximum
        if low != high:
            axes.append((k, 1 if high else -1))
    if not axes:
        return None
    return tuple(axes)


def _warn_if_flush_inside(space: _Box, elem: _Box,
                          tol: float = CONTACT_TOL) -> None:
    """Say so when an element lies INSIDE a space and is flush with one of its
    faces — the one omission a reader could reasonably call a miss.

    ``_contact`` requires the element to reach PAST a face of the space. An
    element that only reaches the face from inside (a room authored to its
    walls' OUTER faces, so its volume swallows them; a partition standing
    against the boundary plane) therefore gets no boundary, and that is
    deliberate rather than an oversight: the classification asks what lies
    BEYOND the element, and for something inside the room the honest answer is
    "the room" — which no ``IfcInternalOrExternalEnum`` value states.
    ``EXTERNAL`` would claim an external space beyond a wall standing in the
    middle of a room, which is the same false statement that keeps terrain out
    of ``EXTERNAL``.

    Silence would make that indistinguishable from "the compiler did not see
    this element". It is one warning naming both, so the omission is a fact in
    the log rather than a gap in the relation set.

    Deliberately narrow. A free-floating object touching NOTHING is not warned
    about — nobody expects a chair to bound a room, and warning on every
    interior element would bury the case that is genuinely arguable.
    """
    (s0, s1), (e0, e1) = space.aabb, elem.aabb
    inside = all(e0[k] >= s0[k] - tol and e1[k] <= s1[k] + tol for k in range(3))
    if not inside:
        return
    flush = [k for k in range(3)
             if abs(e0[k] - s0[k]) <= tol or abs(e1[k] - s1[k]) <= tol]
    if not flush:
        return
    logger.warning(
        "space boundary: %s lies inside %s and is flush with %d of its faces, "
        "so no boundary is inferred — a bounding element has to reach PAST a "
        "face of the space. Author the space to the INNER faces of its walls "
        "if you meant these to bound it.",
        elem.name, space.name, len(flush))


def _contact_rect(space: Aabb, elem: Aabb,
                  axis: int) -> Tuple[Tuple[float, float], ...]:
    """The shared area on the plane perpendicular to ``axis``, as two spans.

    Spans are in ascending axis order (axis 0 -> (y, z); 1 -> (x, z);
    2 -> (x, y)), which is the order :func:`_coverage` compresses in.
    """
    (s0, s1), (e0, e1) = space, elem
    return tuple((max(s0[k], e0[k]), min(s1[k], e1[k]))
                 for k in range(3) if k != axis)


# ---------------------------------------------------------------------------
# InternalOrExternalBoundary
# ---------------------------------------------------------------------------


def _classify(space: _Box, elem: _Box, contact: Contact,
              spaces: Sequence[_Box], elements: Sequence[_Box]) -> str:
    """``IfcInternalOrExternalEnum`` for one boundary — see the docstring.

    Every boundary axis is answered independently and the answers are then
    reconciled: agreement is the answer, disagreement is ``NOTDEFINED``, which
    is the same reading of "internal and external" the per-axis partial case
    produces. A corner column delimiting into a corridor on one axis and the
    outdoors on the other genuinely is both.
    """
    if elem.terrain:
        # There is earth on the far side of an earth surface. ``EXTERNAL``
        # would assert an external SPACE, which the enum's own definition
        # rules out — this is not a synonym.
        return EXTERNAL_EARTH

    verdicts = {_classify_axis(space, elem, axis, sign, spaces, elements)
                for axis, sign in contact}
    if len(verdicts) == 1:
        return verdicts.pop()
    return NOTDEFINED


def _classify_declared(space: _Box, elem: _Box, member: str,
                       spaces: Sequence[_Box],
                       elements: Sequence[_Box]) -> str:
    """``IfcInternalOrExternalEnum`` for a DECLARED boundary.

    Classification stays COMPUTED in bounds-mode — what lies beyond an
    element is not an author statement — and the computation is
    :func:`_classify`, unchanged. What a declared boundary needs extra is the
    CONTACT: a declared bound is emitted wherever it sits, including where
    the geometric contact test fails (a projection pulled the derived volume
    short of the wall; a feature poking into the room usually still passes).
    Three rungs, most-informed first:

    1. the geometric contact against the derived volume (the normal case);
    2. the inner-face axes the volume derivation recorded for this member —
       a bound that closed a side of the room is asked "what lies beyond"
       along exactly that side;
    3. neither (a declared bound that spans the room, or contributed only
       projections and stands clear) → ``NOTDEFINED``, the enum's own
       "no information" value — a guess in either direction would be a
       false statement.
    """
    if elem.terrain:
        return EXTERNAL_EARTH
    contact = _contact(space.aabb, elem.aabb)
    if contact is None:
        contact = (space.face_axes or {}).get(member) or None
    if contact is None:
        return NOTDEFINED
    return _classify(space, elem, contact, spaces, elements)


def _check_declared_set(space: _Box, elements: Sequence[_Box],
                        tol: float = CONTACT_TOL) -> None:
    """The WS-C contact engine, demoted to a CHECKER on a bounds-mode space.

    Warnings, never errors — declared intent is honored at face value, and
    both shapes here are legal models that are probably not what the author
    meant:

    * a declared bound the derived volume stops touching — typically a
      boolean carved the room away from it. Judged against the EFFECTIVE
      volume (the derived box with boolean end-slices trimmed —
      ``space_volume.effective_volume``), because the derived box itself
      cannot see a carve;
    * an undeclared element that passes the WS-C contact test against the
      derived volume — a candidate member the declaration missed.
    """
    volume = space.effective or space.aabb
    declared = set(space.declared or ())
    by_name = {e.name: e for e in elements}
    for member in space.declared or ():
        elem = by_name.get(member)
        if elem is None:                                  # pragma: no cover
            continue                 # already refused loudly by the caller
        if any(d < -tol for d in overlap_depths(volume, elem.aabb)):
            logger.warning(
                "space bounds: %s's derived volume does not touch its "
                "declared bound %s — either a boolean carved the room away "
                "from it, or the bound never enclosed it. The declared "
                "boundary is still emitted (declared means declared); drop "
                "it from bounds= if it is stale.", space.name, member)
    for elem in elements:
        if elem.name in declared:
            continue
        if _contact(volume, elem.aabb) is not None:
            logger.warning(
                "space bounds: %s touches %s, which is not in its bounds= — "
                "candidate member (no boundary is emitted for it).",
                space.name, elem.name)


def _classify_axis(space: _Box, elem: _Box, axis: int, sign: int,
                   spaces: Sequence[_Box],
                   elements: Sequence[_Box]) -> str:
    """What lies beyond ``elem`` on ``axis``, over the contact area."""
    rect = _contact_rect(space.aabb, elem.aabb, axis)

    space_rects = [_contact_rect(other.aabb, elem.aabb, axis)
                   for other in spaces
                   if other.name != space.name
                   and _is_beyond(other.aabb, elem.aabb, axis, sign)]
    earth_rects = [_contact_rect(other.aabb, elem.aabb, axis)
                   for other in elements
                   if other.terrain
                   and _is_beyond(other.aabb, elem.aabb, axis, sign)]

    by_space = _coverage(rect, space_rects)
    by_earth = _coverage(rect, earth_rects)

    if by_space == _FULL and by_earth == _NONE:
        return INTERNAL
    if by_earth == _FULL and by_space == _NONE:
        return EXTERNAL_EARTH
    if by_space == _NONE and by_earth == _NONE:
        return EXTERNAL
    # Partly one, partly the other, or partly nothing: the concept's
    # "undefined (internal and external)" case, verbatim.
    return NOTDEFINED


def _is_beyond(other: Aabb, elem: Aabb, axis: int, sign: int,
               tol: float = CONTACT_TOL) -> bool:
    """Is ``other`` on the FAR side of ``elem`` along ``axis``, AGAINST it?

    Position only — HOW MUCH of the contact rectangle ``other`` covers is
    :func:`_coverage`'s question, and asking it in both places is the shape. (An earlier draft screened for shared area here too; a mutation
    disabling that screen left all 77 tests green, because a box overlapping
    the element on a LINE yields a degenerate contact rectangle that covers
    nothing anyway. Removed rather than kept as a guard that cannot fail.)

    Two conditions, both measured:

    * **No GAP along the axis.** ``InternalOrExternalBoundary`` says what is on
      the OTHER SIDE of the element, not what is roughly over there, so a room
      standing 100 mm clear of the wall is not behind it — the same strictness
      the contact test applies on the near side, where a 1 mm gap is no
      boundary at all. Without this a terrain block pulled back from a basement
      wall still reported ``EXTERNAL_EARTH``, because "somewhere in the -x
      half-space" was the whole test.
    * **Past the FAR face, and starting past the NEAR one.** A box that
      STRADDLES the element is not behind it — the element does not separate
      the two. Two spaces on the same side of one wall (an open-plan room and
      a circulation zone drawn inside it) would otherwise make that wall
      INTERNAL, claiming a room across a wall that has nothing across it.
    """
    (o0, o1), (e0, e1) = other, elem
    if overlap_depths(other, elem)[axis] < -tol:
        return False                       # a gap: nothing is against it
    if sign > 0:
        return o1[axis] > e1[axis] + tol and o0[axis] > e0[axis] + tol
    return o0[axis] < e0[axis] - tol and o1[axis] < e1[axis] - tol


#: Coverage verdicts. Three, not a fraction: the enum has three outcomes and a
#: ratio would invite a threshold nobody could justify.
_NONE = "none"
_PARTIAL = "partial"
_FULL = "full"


def _coverage(rect: Tuple[Tuple[float, float], ...],
              others: Sequence[Tuple[Tuple[float, float], ...]],
              tol: float = CONTACT_TOL) -> str:
    """How much of ``rect`` the union of ``others`` covers — EXACTLY.

    Coordinate compression, not sampling: every edge of ``rect`` and of every
    coverer becomes a grid line, and each resulting cell is either wholly
    inside some coverer or wholly outside all of them (a cell cannot be split
    — there is no edge inside it, by construction). So "all cells covered" IS
    full coverage, including the case two rooms behind one wall cover it
    together and neither covers it alone. A ratio test with a threshold would
    have called that partial or full depending on a number nobody could
    defend.
    """
    if not others:
        return _NONE
    spans = tuple(hi - lo for lo, hi in rect)
    if any(s <= tol for s in spans):                      # pragma: no cover
        return _NONE                       # degenerate contact covers nothing

    grids = []
    for k, (lo, hi) in enumerate(rect):
        cuts = {lo, hi}
        for other in others:
            for value in other[k]:
                if lo + tol < value < hi - tol:
                    cuts.add(value)
        grids.append(sorted(cuts))

    covered = uncovered = 0
    for u0, u1 in zip(grids[0], grids[0][1:]):
        for v0, v1 in zip(grids[1], grids[1][1:]):
            centre = ((u0 + u1) / 2.0, (v0 + v1) / 2.0)
            if any(o[0][0] - tol <= centre[0] <= o[0][1] + tol
                   and o[1][0] - tol <= centre[1] <= o[1][1] + tol
                   for o in others):
                covered += 1
            else:
                uncovered += 1
    if uncovered == 0:
        return _FULL
    if covered == 0:
        return _NONE
    return _PARTIAL


# ---------------------------------------------------------------------------
# Emission support
# ---------------------------------------------------------------------------


def missing_product_error(boundary: SpaceBoundary, attribute: str,
                          canonical: str) -> SpaceBoundaryError:
    """The refusal when a backend cannot join a boundary to a product.

    This module walks the DSL tree and the backends walk their own emitted
    entities; a name that resolves in one and not the other means those two
    walks have drifted. Raised rather than skipped, because a dropped boundary
    leaves a relation set that looks merely incomplete — and an incomplete
    boundary set is indistinguishable from a model that genuinely has fewer
    adjacencies.
    """
    return SpaceBoundaryError(
        f"space boundary {boundary.space!r} -> {boundary.element!r}: no "
        f"emitted product is named {canonical!r} for {attribute}. The "
        f"boundary walk and the product walk disagree about what this model "
        f"contains, so the relation cannot be minted."
    )


def wrong_type_error(boundary: SpaceBoundary, attribute: str, canonical: str,
                     expected: str, found: str) -> SpaceBoundaryError:
    """The refusal when a name resolves to the wrong KIND of entity.

    ``RelatingSpace`` is an ``IfcSpaceBoundarySelect`` (``IfcSpace`` or
    ``IfcExternalSpatialElement``) and ``RelatedBuildingElement`` is an
    ``IfcElement``. Neither backend's writer enforces a SELECT or a supertype
    at create time, so a canonical name that resolved to, say, an
    ``IfcBuildingStorey`` would produce a relation that serialises perfectly
    and violates the schema — the exact shape of failure this repository
    refuses to ship.
    """
    return SpaceBoundaryError(
        f"space boundary {boundary.space!r} -> {boundary.element!r}: "
        f"{attribute} resolved {canonical!r} to a {found}, but the schema "
        f"requires a {expected}. The relation would serialise and be invalid."
    )


def report_lines(boundaries: Sequence[SpaceBoundary]) -> List[str]:
    """One line per boundary for the compile log — what was emitted and why.

    Front-loaded observability: a derived relation nobody authored is
    invisible in the source, so the compile output is the only place a
    reviewer can see that the classification changed.

    The header counts by MODE, honestly: a bounds-mode space's boundaries
    are DECLARED (the author's statement, expanded), a geometry-mode
    space's are inferred — calling a declared set "inferred" was the one
    word the doctrine reformulation left behind. A single-mode model prints
    one number (``6 declared`` / ``6 inferred``, the historical form); a
    mixed model prints both.
    """
    if not boundaries:
        return []
    declared = sum(1 for b in boundaries if b.declared)
    inferred = len(boundaries) - declared
    parts = ([f"{declared} declared"] if declared else []) \
        + ([f"{inferred} inferred"] if inferred else [])
    lines = [f"space boundaries: {', '.join(parts)}"]
    for b in boundaries:
        parent = " (inner)" if b.is_inner else ""
        lines.append(f"  {b.space} <- {b.element}: "
                     f"{b.physical_or_virtual}/{b.internal_or_external}{parent}")
    return lines
