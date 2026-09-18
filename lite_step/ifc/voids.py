"""Where a HOLE lands — which products receive the ``IfcOpeningElement``.

Both verbs, one rule. ``.void()`` and ``.opening()`` are two rows of one table
(see ``BimElement.void``): both mint an ``IfcOpeningElement`` linked by
``IfcRelVoidsElement``, and they differ only in whether the hole is NAMED and
whether joinery FILLS it. Neither difference is about which product carries the
hole, so since they share this module — ``.void()`` through
:func:`void_targets`, ``.opening()`` through :func:`hole_targets`, which is the
same function with the frame resolution lifted out. The names below still say
"void" because that is what an ``IfcOpeningElement`` is.


The ONE place the emitter and the displacement pass read, mirroring
:mod:`lite_step.ifc.grids`, :mod:`lite_step.ifc.groupings`,
:mod:`lite_step.ifc.storeys` and :mod:`lite_step.ifc.product_types`. Two implementations of one idea agreeing *usually* is what produces the bug,
so the routing decision is made HERE; ``generator.py`` contributes only the
lines that mint an ``IfcOpeningElement``, and
``compiler.displacement`` only the booleans it must NOT emit as a result.

Two routes, one question
------------------------

A ``.void()`` on a SELF-GEOMETRY host (a ``Box``, an ``Extrude``, a ``Pipe``,
a ``Wall`` whose body sits on the ``IfcWall``) attaches to that host's own
product and reaches nothing else — ``generator._process_voids_on_element``.

A hole on a CONTAINER (``Column``/``Beam``/``Slab``/``Roof``/``Element``, and
since a BODY-LESS ``Wall``) has nowhere to attach: the container carries
no representation of its own, the geometry is on the aggregated leaves, and an
``IfcRelVoidsElement`` on a representation-less product carves nothing in a
renderer that builds each element's mesh from its OWN representation (web-ifc /
ThatOpen, our production path). So the hole is handed DOWN, to the
representation-bearing descendants it overlaps.

The ``Wall`` row is the one decided per INSTANCE, and that is not a special
case bolted on — it is the same question asked of a container whose class
answers it both ways. See :func:`void_reaches_children`.

:func:`void_reaches_children` is that fork, and it exists as a function
because five callers must agree on it:

* ``generator._process_container_voids`` takes the "hand it down" branch;
* ``generator._process_container_openings`` takes the same branch for
  ``.opening()``;
* ``displacement._carve_voids_through_details`` and
  ``displacement._carve_openings_through_details`` must SKIP exactly those
  hosts, because their details already receive an ``IfcRelVoidsElement`` for
  the hole and a boolean would carve a second time for nothing — one redundant
  operand per detail per hole, paid for in CSG depth (see
  :mod:`lite_step.compiler.csg_depth`), which is the budget the primary web
  viewer silently runs out of.

The fork is spelled on the CONTAINER and not on the verb for that last reason:
``Slab`` and ``Roof`` are in the frozenset because their ``.void()``
distributes, and their ``.opening()`` is refused at compile (a horizontal host
has no run axis for ``along=`` to walk stays open for it), so the skip
is unreachable for them through the opening pass rather than wrong.

That agreement must not be a hardcoded ``frozenset`` in ``displacement.py``
mirroring the generator's routing table by hand. Two spellings of one rule is
the shape names, and here the failure is invisible in both directions: a
container newly routed to the child-distribution path but missing from the
copy gets its details carved twice, and one dropped from the routing table but
left in the copy gets its details not carved at all.

Depth, and why the walk is not a ``zip``
----------------------------------------

Pairing ONE level — ``zip(geom_leaves, child_products)``, where
``geom_leaves`` is ``[c for c in container.elements if _element_aabb(c) is not
None]`` — is wrong in three separate
ways, and all three are silent:

* a nested container child has NO geometry of its own, so
  ``_element_aabb`` answers ``None`` and it is missing from ``geom_leaves``
  while its product IS in ``child_products`` — the counts disagree, the code
  falls back to "attach the void to every child product", and the hole lands
  on a product whose ``Representation`` is ``None`` (carving nothing) and on
  products it does not overlap (carving nothing, and costing entities);
* a ``Mesh`` child of an ``Element`` is the mirror image — it HAS an AABB but
  emits no product of its own (its face set merges into the wrapper's
  representation), so it shifts every pairing after it by one;
* a LAYERED container emits per-layer SLICES in place of its body child, so
  ``container.elements`` is not the list that produced ``child_products`` at
  all.

So the pairing is not inferred here. The generator RECORDS ``(dsl child, ifc
product)`` as it emits — :meth:`~lite_step.ifc.entity_cache.EntityCache
.record_child_product` — and this module walks that, which makes it
depth-agnostic for free: a nested container's own children were recorded by
the same call when it was emitted.

What is eligible is decided by
:func:`~lite_step.ifc.product_types.occurrence_shape_slots`, the depth-
agnostic walk the emitter already reads for geometry sharing: a product is a
legal void host exactly when it owns a shape slot, i.e. when it bears a
representation of its own. A ``Representation=None`` aggregate parent is
therefore ineligible by construction rather than by a check someone has to
remember to write.

Overlap is measured with :func:`~lite_step.compiler.extent.padded_aabb`, the
same bound ``displacement._carve_one_void_host`` uses on the non-container
route — and for the same reason. ``_element_aabb`` tracks path POINTS only, so
a ``Pipe`` or ``Bar`` leaf standing in the hole has a DEGENERATE box that a
positive-overlap test can never match; with an exact pairing (which does not
fall back to "attach to everything") that miss would be a silently un-voided
leaf. The pad is a bound, so it can only over-select, and an over-selected
target is an opening that intersects nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

from lite_step.ifc.product_types import ShapeAdapter, occurrence_shape_slots

#: Container kinds whose HOLES — ``.void()`` and, since, ``.opening()`` —
#: are handed to their geometry-bearing descendants instead of attaching to the
#: container's own product, WHATEVER they contain. Read through
#: :func:`void_reaches_children` — never copied.
#:
#: A ``Slab``'s ``.add()``ed ``Box`` is not a counter-example: it is emitted as
#: its OWN product and aggregated, so the ``IfcSlab`` still bears no
#: representation. ``Wall`` is the one container where that is not true, which
#: is why it is not in here — see :data:`BODY_BEARING_CONTAINERS`.
VOID_REACHES_CHILDREN = frozenset({"Column", "Beam", "Slab", "Roof", "Element"})

#: Container kinds whose answer is per-INSTANCE, because their first
#: ``.add()``ed prism becomes the container's OWN representation rather than a
#: product of its own (``generator._create_wall``, ``tx.body_prisms``). Such a
#: container carries the hole itself when it has a body and distributes when it
#: does not — one rule, read off the geometry rather than off the class.
BODY_BEARING_CONTAINERS = frozenset({"Wall"})


def void_reaches_children(container) -> bool:
    """True when ``container``'s holes are distributed to its descendants.

    The single fork described in the module docstring, and the question it
    actually asks is **does this container bear a representation of its own**.
    A product that does can carry the ``IfcRelVoidsElement`` and be carved by
    it; a product that does not cannot, in the renderer that matters.

    For :data:`VOID_REACHES_CHILDREN` the answer is a property of the CLASS —
    an ``IfcColumn``/``IfcSlab``/``IfcElementAssembly`` aggregating leaves never
    has a body of its own. For a ``Wall`` it is a property of the INSTANCE:
    ``_create_wall`` turns the first unanchored prism into the ``IfcWall``'s
    representation, so a bodied wall keeps its holes and a body-less
    aggregator has nothing to keep them on — the ``IfcWall`` it emits
    has ``Representation=None``. Measurement showed what the workaround for that
    cost: an authored envelope spanning the whole buildup reports **2.12 m3 of
    phantom wall** when the assembly is under-tiled and tessellates to
    **verts = 0** behind 29 booleans when it is tiled correctly, and the better
    the model the more degenerate the artefact.

    ``tx.body_prisms`` is THE definition of "the body", shared with the
    validator and the emitter, so this cannot become a fourth opinion about
    what a Wall's geometry is.

    Keyed on the DSL class NAME rather than on ``isinstance`` because
    ``displacement`` reaches this with the models package already imported
    under a different alias in some call paths, and because the routing table
    in ``generator.py`` it mirrors is itself a set of container emitters, not a
    class hierarchy.
    """
    kind = type(container).__name__
    if kind in VOID_REACHES_CHILDREN:
        return True
    if kind in BODY_BEARING_CONTAINERS:
        from lite_step.models import taxonomy as tx

        return not tx.body_prisms(container)
    return False


def opening_reaches_children(container) -> bool:
    """True when ``container``'s ``.opening()`` is distributed to its
    descendants — the OPENING fork, and NOT the same set as
    :func:`void_reaches_children`.

    ``void_reaches_children`` AND-ed with "an emitter reads this type's
    openings at all" (``taxonomy.hosts_openings``). The two differ on ``Slab``
    and ``Roof``: their ``.void()`` distributes, and their ``.opening()`` is
    refused at compile because ``along=``/``inset=``/``up=`` have no outer face
    or run axis to measure against on a horizontal host.

    That difference is load-bearing and it is why this is not spelled inline.
    ``displacement._carve_openings_through_details`` skips a host whose leaves
    will each receive a real ``IfcRelVoidsElement``, because a boolean would
    then carve a second time for nothing. On a ``Slab`` they will NOT — no
    emitter distributes a slab's opening — so skipping it there would silently
    change what the unvalidated ``generate_ifc`` path emits for a model the
    validator refuses, which is a measurement three tests make.
    """
    from lite_step.models import taxonomy as tx

    return void_reaches_children(container) and tx.hosts_openings(container)


@dataclass(frozen=True)
class VoidHost:
    """One legal landing place for a container's ``.void()``.

    ``node`` is the DSL element (the thing with an AABB); ``product`` is the
    backend handle to the IFC product that carries its representation. The
    pair is what the distribution needs, and keeping it a pair is what makes
    "the void overlaps THIS leaf" and "the opening attaches to THAT product"
    the same statement instead of two that have to be kept aligned.
    """

    node: Any
    product: Any


def void_hosts(container, container_product, adapter: ShapeAdapter,
               children_of: Callable[[Any], Sequence[Tuple[Any, Any]]]
               ) -> List[VoidHost]:
    """Every representation-bearing descendant of ``container``, at any depth.

    ``children_of(node)`` yields the recorded ``(dsl child, ifc product)``
    pairs for one container — see the module docstring on why the pairing is
    recorded rather than inferred.

    The container's OWN product is deliberately not a candidate even when it
    bears a representation (an ``Element`` wrapper whose ``Mesh`` children
    merge into one tessellation does). A void on such a wrapper has never
    attached there, and a mesh is not a body an ``IfcOpeningElement`` carves —
    ``executor`` refuses ``.void()`` on a ``Mesh`` outright, and a ``Site``'s
    void is realised as terrain CSG instead. Reaching it here would contradict
    both.
    """
    eligible = {adapter.key(slot.owner)
                for slot in occurrence_shape_slots(container_product, adapter)}
    hosts: List[VoidHost] = []
    seen: set = set()

    def walk(node) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        for child, product in children_of(node):
            if product is not None and adapter.key(product) in eligible:
                hosts.append(VoidHost(child, product))
            walk(child)

    walk(container)
    return hosts


def void_targets(container, container_product, adapter: ShapeAdapter,
                 children_of: Callable[[Any], Sequence[Tuple[Any, Any]]],
                 operand) -> List[VoidHost]:
    """The hosts ONE ``.void()`` operand carves, in emission order.

    Returns the overlapping subset of :func:`void_hosts`, or an EMPTY list
    when the hole reaches nothing — the caller warns. Attaching a
    non-overlapping opening would be the fallback for an unreliable pairing;
    with the pairing recorded there is nothing left for it to rescue, and it
    would put an ``IfcOpeningElement`` in the file for every leaf a hole
    misses.
    """
    from lite_step.compiler.extent import MM_PER_METER, padded_aabb

    return hole_targets(container, container_product, adapter, children_of,
                        padded_aabb(operand, MM_PER_METER))


def hole_targets(container, container_product, adapter: ShapeAdapter,
                 children_of: Callable[[Any], Sequence[Tuple[Any, Any]]],
                 hole_aabb) -> List[VoidHost]:
    """The hosts ONE hole lands on, in emission order — the shared half.

    ``hole_aabb`` is the hole's WORLD ``(min, max)``, and taking it as a bound
    rather than as a DSL operand is what lets both verbs share this. A
    ``.void()`` operand is absolute world geometry, so
    :func:`void_targets` measures it with ``padded_aabb``; an ``.opening()``
    carries HOST-FRAME scalars, so ``generator._process_container_openings``
    resolves them through ``frames.opening_void_prism`` first and hands in
    ``prism.aabb()``. Two frames, one landing rule — which is the whole
    point, because ``displacement`` skips the boolean carve on exactly the
    containers this returns targets for, and a second landing rule would make
    that skip right for one verb and wrong for the other.

    ``None`` means the hole has no recoverable extent (an empty ``Mesh``
    operand, say). It cannot be tested, and refusing to place it would be the
    silent drop this whole module exists to remove — so it goes to every host.
    """
    from lite_step.compiler.displacement import _overlaps
    from lite_step.compiler.extent import MM_PER_METER, padded_aabb

    hosts = void_hosts(container, container_product, adapter, children_of)
    if hole_aabb is None:
        return hosts
    hit: List[VoidHost] = []
    for host in hosts:
        h_aabb = padded_aabb(host.node, MM_PER_METER)
        if h_aabb is not None and _overlaps(hole_aabb, h_aabb):
            hit.append(host)
    return hit
