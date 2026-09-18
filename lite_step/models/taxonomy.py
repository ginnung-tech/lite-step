"""Compiler routing table — NOT an axis taxonomy.

Each tuple below names a routing DECISION the compiler makes, grouped by the axis
it routes on. **The axes are independent**: a ``Box`` is on the representation axis
(prism-tessellated) AND the opening-eligibility axis (voidable). That cross-axis
overlap is deliberate — this is a routing table, not a partition.

Conceptually a Lite-STEP element holds one or more **aspects** — it may carry
geometry, denote a building element (physical), and/or be a spatial container. The
element-nature predicates (``is_geometric`` / ``is_physical`` / ``is_spatial`` /
``is_annotation``) partition the element TYPES; the routing tuples
(``TESSELLATION_*`` / ``MESH`` / ``VOIDABLE`` / ``SOLIDS`` / ``OPENINGS``) cross-cut
them (``VOIDABLE`` spans geometric solids AND building elements). The conceptual
aspects are described in the DSL reference; they are not what the tuples encode.

**Scope:** this enumerates the ``BimElement`` PRODUCT types the compiler routes. It
is not the whole type system — material/module specs (not products), and the
``Project``/``Storey`` tree scaffold (spatial structure, but never a void host or a
tessellation operand — never routed here), are intentionally absent.

Every predicate accepts an element INSTANCE, a class, or a type-name string.
"""
from __future__ import annotations

from lite_step.models import elements as _elements_module
from lite_step.models.elements import (
    BimElement,
    Box, Extrude, Revolve, Sweep, Pipe, Bar, Mesh,
    Wall, Column, Beam, Slab, Roof, Window, Door, Element,
    Site, Space, SpatialElement, ReferencePoint, GuideLine,
)

#: Every ``BimElement`` subclass the models module defines, by reflection.
#: The tuples below that are DERIVED rather than written out scan this, so a
#: new element class cannot be silently forgotten by one of them. Scanning the
#: module namespace rather than ``BimElement.__subclasses__()`` keeps the set
#: deterministic — a subclass defined in a test or a skill is not routing.
_ALL_ELEMENT_TYPES = tuple(
    obj for obj in vars(_elements_module).values()
    if isinstance(obj, type)
    and issubclass(obj, BimElement)
    and obj is not BimElement
)

# ── Representation axis — how a mesh is produced (geometric only, exclusive) ──
TESSELLATION_PRISM = (Box, Extrude)                 # numpy fast-path
TESSELLATION_CURVED = (Revolve, Sweep, Pipe, Bar)   # ifcopenshell-kernel
MESH = (Mesh,)                                      # already a mesh (passthrough)
SOLIDS = TESSELLATION_PRISM + TESSELLATION_CURVED
GEOMETRIC = SOLIDS + MESH

# ── Opening-eligibility axis — can host an IfcRelVoidsElement? ────────────────
#
# Every PRODUCT type this module routes. It exists so ``OPENINGS`` can be
# DERIVED from ``BimElement.brings_void`` rather than restated: the class
# attribute is the single source of truth for "does placing this cut a void",
# and a tuple written out by hand beside it is the second implementation that
# agrees *usually* — the shape. Adding a void-bringing type means setting
# its ``brings_void`` and importing it here; nothing else to remember.
_PRODUCT_TYPES = (
    Box, Extrude, Revolve, Sweep, Pipe, Bar, Mesh,
    Wall, Column, Beam, Slab, Roof, Window, Door, Element,
    Site, Space, ReferencePoint, GuideLine,
)

OPENINGS = tuple(t for t in _PRODUCT_TYPES if t.brings_void)   # == (Window, Door)
VOIDABLE = SOLIDS + (Wall, Column, Beam, Slab, Roof, Element)  # solids + non-fill building elements
SPATIAL = (Site, Space)                             # IfcSpace/IfcSite — NOT voidable via an opening

# ── Opening-EMISSION axis — which host puts the void in the FILE? ────────────
#
# NOT ``VOIDABLE``, and the gap between the two is. ``VOIDABLE`` answers
# "may an ``IfcOpeningElement`` hang off this in principle" and it is the tuple
# ``agents/dsl-reference.md`` was written from; ``OPENING_HOSTS`` answers "does
# an emitter READ this element's openings", which is the question every author
# is actually asking. Three emitters do:
#
# * ``generator._create_wall_openings_from_assembly`` — reads a ``Wall``'s
#   ``_elements`` for ``brings_void`` children;
# * ``generator._process_openings_on_element`` — reads ``_openings`` off a
#   ``Box``/``Extrude``, called from ``_create_box`` / ``_create_extrude`` and
#   from nowhere else;
# * ``generator._process_container_openings`` — reads BOTH lists off a
#   ``Column``/``Beam``/``Element`` and DISTRIBUTES the hole to the
#   geometry-bearing leaves it overlaps, exactly as
#   ``_process_container_voids`` has distributed ``.void()`` since. A
#   representation-less container cannot carry the hole itself: web-ifc /
#   ThatOpen (our production path) builds each element's mesh from its OWN
#   representation, so an ``IfcRelVoidsElement`` on the aggregate parent carves
#   nothing. See ``lite_step.ifc.voids``.
#
# Every other product type reaches the opening with no route for it. Measured
# on ``main`` @ 24304c8, the emitter, one 6 m body and one
# ``host.opening(Window(1200x1400), along=4200, up=900)``: ``Wall`` and a
# standalone ``Box``/``Extrude`` emit 1 ``IfcOpeningElement`` + 1 ``IfcWindow``;
# ``Column``, ``Beam``, ``Slab``, ``Roof``, ``Element``, ``Sweep``, ``Pipe``,
# ``Bar``, ``Revolve`` and ``Mesh`` emit **zero of each** — the containers with
# a "has no IFC emitter and was skipped" warning, the leaf solids with no
# warning at all. ``executor._opening_host_errors`` turns that silence into a
# compile refusal for the ones still missing an emitter.
#
# **``Slab`` and ``Roof`` are deliberately still out** (#732 stays open for
# them). ``along=``/``inset=``/``up=`` measure from an outer FACE along a run
# axis, and the distribution answers *where the hole goes*, not *what a
# HORIZONTAL host's opening measures against*: a slab's "outer face" is the
# soffit or the topside depending on which way you look, and its run axis is a
# plan direction with no canonical choice between the two in-plane axes.
# ``Column``/``Beam``/``Element`` are the tractable subset because their frame
# basis IS wall-shaped — an outer face and a run axis — so the existing scalars
# already mean something on them. A slab penetration probably wants a different
# spelling, and guessing one here would be a signature nobody can hold.
#
# **Type is necessary and not sufficient.** A ``Box`` that is a ``Wall``'s BODY
# is not emitted by ``_create_box`` at all — ``_create_wall`` builds the
# ``IfcWall`` out of it and never reads its ``_openings`` — so that one
# position is refused too, by the validator and not by this tuple, which
# answers about a TYPE. See ``executor._opening_host_errors``.
OPENING_HOSTS = ((Wall,) + TESSELLATION_PRISM       # Wall, Box, Extrude
                 + (Column, Beam, Element))         # ...distributing

# ── Physical domain — building elements (IfcElement, aggregate children) ──────
PHYSICAL = (Wall, Column, Beam, Slab, Roof, Element) + OPENINGS

# ── Containment axis — holds no points of its OWN; its children carry them ────
#
# The routing decision this answers: "does normalizing / baking / emitting this
# element mean touching its own fields, or only recursing?" It became a decision
# worth naming when ``.anchor()`` stopped gating child types (v22.3.0): any
# container can now hold any type, so a pass that dispatched on a per-container
# child table would silently skip a newly-legal pairing. Openings are NOT here —
# a Window/Door carries its own size and a void, and its children are
# opening-local, so it is a special case in every pass that meets it.
#
# ``SpatialElement`` IS one — it is ``_elements``-bearing and holds no points
# of its own, exactly like the rest. It was absent because the tuple was
# written for the physical containers, so a pass asking "is this a container?"
# got ``False`` for the one node type whose entire job is holding other things
# (a facility's parts, a part's products).
#
# DERIVED, not hand-written, for exactly that reason: "declares ``_elements``"
# is the rule the docstring above states, so it is the rule the tuple runs.
# A new container class joins by declaring the attribute; a hand-written tuple
# beside the rule is the second implementation that agrees *usually*,
# and ``SpatialElement`` is what "usually" cost.
_ELEMENTS_BEARING = tuple(
    t for t in _ALL_ELEMENT_TYPES if "_elements" in t.__private_attributes__
)
#: Openings stay OUT, per the reasoning above: a Window/Door is
#: ``_elements``-bearing but carries its own size and a void, and its children
#: are opening-local — a special case in every pass, never a container.
CONTAINERS = tuple(t for t in _ELEMENTS_BEARING if not t.brings_void)

# ── Local-frame axis — can host a child in ITS OWN frame? ─────────────────────
#
# The SECOND container question, and keeping it separate from ``is_container``
# is the entire point of naming it. THREE definitions of "container" were live
# at once and disagreed pairwise: ``is_container`` (holds no points of its
# own), "declares ``_elements``" (Window/Door do, and are not containers), and
# "has ``.anchor()``" (``Site`` and ``SpatialElement`` do not). ``.opening()``
# routed on the first AND-ed with an inline ``hasattr(self, "anchor")``, so the
# classes where those two disagreed fell through to the LEAF path and appended
# to ``_openings`` — a list nothing reads for a spatial node. No
# ``IfcOpeningElement``, no ``IfcWindow``, no warning: the doorway that
# silently is not there, on the very method whose comment names that failure.
#
# ``along=`` / ``inset=`` / ``up=`` measure from an outer face along a run
# axis. A Wall has both; a ``Site`` is a PLACE and terrain has neither, and a
# ``SpatialElement`` is a facility node with no body at all. Neither gets an
# invented ``.anchor()`` — they get a refusal naming the verb that is correct.
LOCAL_FRAME = tuple(t for t in CONTAINERS if hasattr(t, "anchor"))

# ── Residual ─────────────────────────────────────────────────────────────────
ANNOTATION = (ReferencePoint, GuideLine)            # planning primitives


def _name(x) -> str:
    """Type-name of an element instance, a class, or a string (identity)."""
    if isinstance(x, str):
        return x
    if isinstance(x, type):
        return x.__name__
    return type(x).__name__


def _names(types) -> frozenset:
    return frozenset(t.__name__ for t in types)


_PRISM_N = _names(TESSELLATION_PRISM)
_CURVED_N = _names(TESSELLATION_CURVED)
_SOLID_N = _names(SOLIDS)
_MESH_N = _names(MESH)
_GEOMETRIC_N = _names(GEOMETRIC)
_PHYSICAL_N = _names(PHYSICAL)
_OPENING_N = _names(OPENINGS)
_VOIDABLE_N = _names(VOIDABLE)
_OPENING_HOST_N = _names(OPENING_HOSTS)
_SPATIAL_N = _names(SPATIAL)
_CONTAINER_N = _names(CONTAINERS)
_LOCAL_FRAME_N = _names(LOCAL_FRAME)
_ANNOTATION_N = _names(ANNOTATION)


# ── Representation axis ───────────────────────────────────────────────────────
def tessellation_kind(x):
    """``"prism"`` | ``"curved"`` | ``"mesh"`` | ``None`` — how a mesh is produced
    from this element (``None`` = not a geometric primitive)."""
    n = _name(x)
    if n in _PRISM_N:
        return "prism"
    if n in _CURVED_N:
        return "curved"
    if n in _MESH_N:
        return "mesh"
    return None


def is_geometric(x) -> bool:
    return _name(x) in _GEOMETRIC_N


def is_solid(x) -> bool:
    return _name(x) in _SOLID_N


def is_prism(x) -> bool:
    return _name(x) in _PRISM_N


def is_curved(x) -> bool:
    return _name(x) in _CURVED_N


def is_mesh(x) -> bool:
    return _name(x) in _MESH_N


def body_prisms(elem) -> list:
    """The container's OWN body solids — its unanchored ``Box``/``Extrude``
    children, in authoring order.

    THE definition of "the body", read by everything that has to agree on it:
    ``executor.validate_project_report`` (which refuses a second one) and the
    wall handlers (which emit the first as the wall's geometry). Spelled inline
    as "the first ``is_prism`` child", the copies disagreed with the validator
    on exactly one thing — **anchoring**.

    An ANCHORED prism is a detail placed in the container's frame (a corbel, a
    sign, a cladding piece), never the body that DEFINES that frame; the
    validator has excluded one since ``.anchor()`` on a Wall became legal.
    The backends did not, so the first prism in ``_elements`` won whatever it
    was — order-dependent while a body was mandatory, and systematic once a
    body-less container became legal: the anchored child was emitted AS
    the container's body, so its canonical name vanished from the file and the
    container claimed geometry it never had. Measured: a ``ledge`` anchored
    into a body-less aggregator produced no ``box:ledge:…`` product at all.
    """
    from lite_step.compiler.frames import is_anchored_child

    return [c for c in (getattr(elem, "_elements", None) or [])
            if is_prism(c) and not is_anchored_child(c)]


# ── Opening-eligibility + container-aggregation ───────────────────────────────
def brings_void(x) -> bool:
    """Does placing this child cut a VOID in its host?

    **The carve trigger, read off the CHILD** (WS-A §1.1) — the one question
    the opening machinery asks, and the only place it is asked. The emitter select a body's opening children with this, so neither
    type-dispatches on ``isinstance(c, (Window, Door))`` any more: adding a
    void-bringing type is a ``brings_void = True`` on the class, not an edit
    to two emitters that must agree.

    Accepts an instance or a class (the ``ClassVar`` reads the same on both)
    and a type-NAME string (the routing-table convention).
    """
    if isinstance(x, str):
        return x in _OPENING_N
    return bool(getattr(x, "brings_void", False))


def is_opening(x) -> bool:
    """Alias of :func:`brings_void` kept for the routing-table vocabulary.

    "Is an opening" and "brings a void" are the same fact; this spelling reads
    better where the question is about the element's KIND (``is_voidable``
    excludes these) and ``brings_void`` reads better where the question is
    about the carve. One implementation either way."""
    return brings_void(x)


def is_voidable(x) -> bool:
    """Can host an ``IfcOpeningElement``: a geometric SOLID or a (non-fill) building
    element. NOT a Mesh (carved by a mesh boolean), a SPATIAL container, or a
    Window/Door (which ARE openings)."""
    return _name(x) in _VOIDABLE_N


def hosts_openings(x) -> bool:
    """Is this element TYPE one whose ``Window``/``Door`` children an emitter
    actually reads — ``Wall``, ``Box``, ``Extrude``, ``Column``, ``Beam``,
    ``Element``?

    The narrower, honest half of :func:`is_voidable`. ``is_voidable`` says a
    ``Slab`` may carry an ``IfcOpeningElement`` in principle; this says whether
    anything in ``lite_step.ifc`` will put one there today. Where they
    disagree, the author gets a compile refusal from
    ``executor._opening_host_errors`` instead of a model with the window
    missing.

    ``Slab`` and ``Roof`` are the disagreement that is left, and it is a frame
    question rather than an emission one — see the ``OPENING_HOSTS`` comment
    and.

    Type only: a ``Box`` that is a ``Wall``'s body answers True here and is
    still refused, because ``_create_wall`` — not ``_create_box`` — emits it.
    """
    return _name(x) in _OPENING_HOST_N


def is_physical(x) -> bool:
    """A building element — an ``IfcElement`` product with aggregated children."""
    return _name(x) in _PHYSICAL_N


def is_container(x) -> bool:
    """Holds no point-bearing fields of its own — its children carry the
    geometry. A pass that reads points from an element must recurse past one
    of these rather than dispatch on a per-container child table."""
    return _name(x) in _CONTAINER_N


def has_local_frame(x) -> bool:
    """Can host a child in ITS OWN frame — the ``.anchor()`` / ``.opening()``
    question, and NOT the same set as :func:`is_container`.

    ``along=`` / ``inset=`` / ``up=`` need an outer face and a run axis to
    measure from. A ``Wall`` or a ``Column`` has both. A ``Site`` and a
    ``SpatialElement`` are containers that have neither — a place and a
    facility node, no body of their own — so a local-frame verb on one has
    nothing to resolve against and MUST refuse rather than park the child.

    Strictly narrower than :func:`is_container`; every local-frame type is a
    container, and ``test_container_children`` pins that inclusion.
    """
    return _name(x) in _LOCAL_FRAME_N


def is_spatial(x) -> bool:
    """An ``IfcSpatialStructureElement`` authored as content (Site/Space): a spatial
    container, not voidable via an opening."""
    return _name(x) in _SPATIAL_N


def is_annotation(x) -> bool:
    return _name(x) in _ANNOTATION_N


# ── Terrain identity (geometry ⊥ physical ⊥ spatial) ─────────────────────────
#
# A Mesh is pure GEOMETRY; "this is terrain" is PHYSICAL, expressed by the
# wrapper ``Element(ifc_class="IfcGeographicElement", predefined_type=
# "TERRAIN")``; "this belongs to the site" is CONTAINMENT into the SPATIAL
# domain (the ``Site`` container). Terrain-ness drives two behaviors —
# carve-host candidacy in the displacement engine and ``site.void()`` folding
# — and BOTH read these predicates. The wrapper is the ONLY terrain marker: the
# legacy ``mesh_type="terrain"`` shortcut (and its ``is_legacy_terrain_mesh``
# shim) was removed once the site-context pack migrated — the value is now a
# construction error naming the wrapper form.

TERRAIN_WRAPPER_CLASS = "IfcGeographicElement"
TERRAIN_PREDEFINED_TYPE = "TERRAIN"


def is_terrain_wrapper(x) -> bool:
    """True for the canonical terrain semantic wrapper:
    ``Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN")``."""
    return (
        _name(x) == "Element"
        and getattr(x, "ifc_class", None) == TERRAIN_WRAPPER_CLASS
        and getattr(x, "predefined_type", None) == TERRAIN_PREDEFINED_TYPE
    )


def iter_terrain_meshes(container):
    """Yield the TERRAIN meshes of a spatial container (``Site``): every Mesh
    child of a terrain wrapper. This is the ``site.void()`` folding target —
    a void on the site clears the GROUND, never sibling groundwork (a gravel
    IfcEarthworksFill bed must not be eaten by the clearing void)."""
    for child in (getattr(container, "_elements", None) or []):
        if is_terrain_wrapper(child):
            for sub in (getattr(child, "_elements", None) or []):
                if is_mesh(sub):
                    yield sub
