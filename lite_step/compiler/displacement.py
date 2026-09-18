"""Universal geometric displacement (place + carve).

Doctrine — pure geometry, **no type/label checks** in the trigger:

    A solid spatially OVERLAPPING another solid/volume displaces it — the
    host is carved by the occupant (only the shared volume is removed), and
    the occupant still renders. Hiding the occupant reveals the void.

That one rule covers window-in-wall's cousins without a single label check:
rebar buried in a beam → the beam gains a rebar-shaped void; a footing sunk
into the ground → the terrain gains a footing-shaped pit; a deck THROUGH a
wall → the wall gains a deck-shaped through-void. It is *visually idempotent*
— the occupant fills its own void, so nothing looks different until you hide
the occupant — which is why "carve every overlapped host" is safe. For
solid-vs-solid pairs the carve is one-directional and TREE POSITION decides:
the solid LATER in ``_collect``'s depth-first walk is the carver, the earlier
one the carvee (construction order — new construction displaces what already
stands). NOT ``.add()`` order: the two diverge once the model spans
containers (per-storey walls), where a helper called FIRST can land LATER.
The rendered union is preserved; abutting/flush faces never trigger.

**Two phases, because "overlapping" has to mean overlapping.** The trigger is a
padded AABB (``extent.padded_aabb``) and the pad is deliberately generous — a
member is padded by its section reach, without which a member would not take
part in displacement at all. That makes it a BROAD phase: it answers "these
might touch", and it says yes to every pair that merely comes close. Behind it
runs a NARROW phase (``_removes_nothing``) that measures the real solids. For
the shapes it can answer — a ``Box``, a straight rectangular-section member —
the oriented box IS the solid, so a separating-axis test is exact rather than a
bound, and it costs fifteen dot products with no tessellation and no mesh
boolean. Anything it cannot answer exactly (curved solid, mesh, bent member,
shaped profile) keeps its carve.

This matters beyond tidiness. A boolean that removes nothing still adds
vertices, still spends one level of the host's CSG depth budget, and is still a
nested ``IfcBooleanResult`` for a viewer to get wrong — web-ifc drops the
deepest chains outright. On the reference villa the broad phase alone proposed
117 such booleans and drove the rafters to 9 of a budget of 10; dropping them
changed no product's volume by a single cubic millimetre.

An undecidable child inherits its parent's proof — ``_sheltered_by_parent``.

"Keeps its carve" is the safe default, and on its own it produced a real
contradiction. A rebar ``Bar`` is a swept disk, so ``convex_solid_obb`` returns
``None`` for it and the pair is kept. Its host wall's body IS a ``Box``, so the
SAME pair against the same distant member is decided and dropped. The result:
the wall correctly stopped carving a member 168 mm away while 23 bars *inside
that wall* carried on carving it — 300 such booleans across five corpus
systems, every one removing nothing.

Containment is transitive through disjointness: a child wholly inside a solid
that is provably disjoint from the host is provably disjoint too, whatever
shape the child is. Only the wholly-inside case qualifies — a child that pokes
out may legitimately touch what its parent does not, because the miter leaves
room for children so they miter individually.

Two things a maintainer needs and neither is obvious from the call site:

* **A container is not a solid.** A ``Wall`` holds its shape in a ``Box``
  child, so ``convex_solid_obb`` on the container returns ``None`` and decides
  nothing. The parent must be resolved to the solids it owns. A first version
  asked the container directly and could never fire — it reported zero, which
  reads as "no problem".
* **The over-stating extents are load-bearing and must stay.**
  ``authored_aabb()`` is pre-carve by contract, ``world_aabb()`` is as-placed,
  and neither reflects a miter clip. That is deliberate: the miter extends
  inner layers to close a corner, so an extent that shrank to the clipped body
  would open the corners. The fix therefore had to be additive — inherit a
  proof — rather than a smaller box for reinforcement to be laid out against.
  Authoring-side placement (``rules/reinforcement``) is correct as written and
  is not the lever here.

That direction rule is solid-vs-solid ONLY. A MESH never carves; it is only
ever carved, by every overlapping un-exempt solid regardless of tree position
(the ``tx.is_mesh(host)`` branch takes no index) — so a mesh is defended by
exemption (``placement=`` / ``carve="none"``), never by order. EVERY mesh hosts,
not only terrain: a mesh is matter, and what a solid displaces does not depend
on what the matter is called.

Mechanism (the ONE representation branch, forced by IFC geometry, not a label):

* **Solid host** (beam/wall/slab body, standalone box) → a copy of the
  occupant's *real* parametric solid is appended to ``host._cuts`` and becomes a
  native ``IfcBooleanResult`` via the existing boolean machinery — exact
  tube/sweep/box hole, no tessellation. The copy is anonymous and consumed
  (``collect_consumed_operand_ids`` keys on object identity); the *original*
  occupant is a different object, so it still renders.
* **Mesh host** (terrain — a tessellated ``Mesh`` can't be an IFC boolean
  operand) → an exact 3D mesh boolean (``manifold3d``): ``terrain − occupants``,
  watertight by construction. The occupant still renders as its own element.

Three kinds of element are excluded, and that list mentions no type:

* ``placement=`` subtrees — the AABB is in pre-placement coordinates, so
  overlap cannot be judged where the geometry actually stands.
* ``.anchor()``ed children — an anchored child's position is a RELATIONSHIP to
  its host, so abutting or entering that host is what anchoring MEANS. This is
  the row a Window/Door falls under too: an opening has no world coordinates of
  its own, so it is always anchored, and it DECLARES its void rather than
  inferring one. There is no "Window/Door subtrees" row.
* ``carve="none"`` subtrees — the author saying so outright (§1.8).

``.add()`` is the carver.

The pass runs once, at the model level, inside ``generate_ifc`` (so the emitter serializes the already-carved result); it is idempotent via
``Project._displacement_applied``.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

# The ONE routing decision about where a ``.void()`` lands, shared with
# ``generator._process_container_voids``. A container whose void the generator
# hands to its geometry-bearing descendants must be SKIPPED here: those
# descendants already receive an ``IfcRelVoidsElement`` for the hole, so a
# boolean would carve a second time for nothing — one redundant operand per
# detail per void, paid for in CSG depth. Restating the container list here is
# what this import replaces; see ``lite_step/ifc/voids.py`` for why one rule
# spelled twice fails invisibly in both directions.
from lite_step.ifc.voids import opening_reaches_children, void_reaches_children

# The ONE padded-AABB derivation, shared with ``BimElement.authored_aabb()``.
# See ``extent``'s docstring for why it is not ``frames.element_aabb``.
from lite_step.compiler.extent import (
    MM_PER_METER,
    Aabb,
    _material_thickness,
    _mesh_aabb,
    convex_solid_obb,
    obbs_are_disjoint,
    overlap_depths,
    padded_aabb,
)

logger = logging.getLogger(__name__)


class CarveAssertionError(Exception):
    """A ``proj.assert_carved(target, by=cutter)`` the compile did not satisfy.

    Loud and fatal, like ``DisplacementError``: the whole point of the
    assertion is that a reordering which silently changes the joinery must
    stop being silent."""


class DisplacementError(Exception):
    """A requested carve could not be realised — an untessellable or unsupported
    void operand (bad shape, ``placement=Anchor``, degenerate geometry) or a
    ``manifold3d`` boolean failure. Raised loudly so ``generate_ifc`` returns a
    failed result instead of silently emitting an uncarved mesh (loud-failure
    doctrine). Caught in ``generate_ifc`` → surfaced as a compile error."""



#: Positive-overlap threshold — abutting/flush faces (zero-depth overlap) must
#: NOT trigger a carve. Model units are meters at carve time.
_EPS = 1e-6

#: Fraction of a mesh host's ORIGINAL volume that must survive its carve before
#: the result counts as annihilated. A terrain left with a thousandth of itself
#: is gone for every practical purpose — it renders as nothing — so "all but
#: annihilated" is warned about on the same terms as an exactly-empty result.
#: The band is deliberately narrow: a real excavation removes a pit, not the
#: ground. See ``_carve_mesh_host``.
_MESH_ANNIHILATION_RESIDUE = 1e-3


# ---------------------------------------------------------------------------
# Collection + enclosure
# ---------------------------------------------------------------------------


def _collect(project, _roots_out: Optional[List] = None,
             _root: Optional[object] = None) -> Tuple[
    List[Tuple[object, Aabb]], List[Tuple[object, Aabb]]
]:
    """Return ``(solids, meshes)`` — every RENDERED solid primitive and every
    ``Mesh`` in the model, each with its world AABB. **Every mesh is a carve
    host**: a solid sunk into a mesh displaces it, whatever the mesh is
    semantically. Terrain, a gravel ``IfcEarthworksFill`` bed and a bare scan
    mesh are all matter, and a footing driven through all three displaces all
    three.

    Host candidacy is GEOMETRIC, not semantic. Restricting it to a mesh inside
    the canonical ``Element(IfcGeographicElement, TERRAIN)`` wrapper would stop
    a gravel bed enclosing a footing from taking a pit the terrain should have,
    and would make a mesh inert unless it is terrain — wrong the moment a model
    has two kinds of ground. Both hosts carve, which is what a footing does.

    ``_root`` re-enters this SAME walk rooted at one placed element instead of
    at the project, which is how ``_collect_frames`` collects a placed
    subtree's own frame. It is a parameter rather than a second function on
    purpose: a copy of this walk would carry its own copy of the four
    exemptions below, and the two would diverge the first time one gained a
    fifth — a solid silently ceasing to carve inside placed subtrees, which is
    the exact bug class exists to close.

    Recurses container ``_elements`` only (never ``_cuts``/``_openings`` —
    those aren't rendered as independent occupants), and skips ``placement=``
    subtrees, ``carve="none"`` subtrees, and ``.anchor()``ed children — which
    is the one row every Window/Door falls under too (see ``_walk``)."""
    from lite_step.models import taxonomy as tx

    solids: List[Tuple[object, Aabb]] = []
    meshes: List[Tuple[object, Aabb]] = []

    def _walk(elem, is_frame_root: bool = False) -> None:
        # ── ONE RULE, three spellings ────────────────────────────────────────
        #
        # This pass resolves ACCIDENTAL overlap: two solids authored in raw
        # world coordinates that happen to occupy the same space, where nothing
        # in the source says which of them should lose material. An element
        # whose position the author STATED is not in that pass, subtree and
        # all — because overlap somebody placed is legal and stays overlap.
        #
        # Three ways to state it, and the three clauses below are the same rule
        # each time:
        #
        #   placement=Transform   its own frame — the exemption WITHOUT binding
        #                         to a container, since a transform is a claim
        #                         about coordinates rather than about a parent
        #   .anchor(child)        a relationship to a host frame
        #   carve="none"          said outright, on the .add()/.anchor() that
        #                         attached it
        #
        # They differ in what ELSE they do (a Transform also transforms; an
        # anchor also resolves against a host), and each clause carries only
        # that difference. What they do not differ in is this rule.
        #
        # ``carve="none"`` additionally reaches the OPENING/VOID carve, which
        # this pass is not — a hole removes matter regardless of how the matter
        # got there, and ``carve="none"`` is the author saying THIS matter is
        # meant to be in it. That is why stating it on an .anchor() is not
        # redundant even though anchoring already grants the clause below.
        #
        # 1. PLACED — the AABBs (and their children's) are in pre-placement
        # authoring coords, so overlap can't be judged where the geometry
        # actually stands; recursing into children would judge them at the
        # UN-transformed position, a spurious carve against whatever stands
        # near the authoring origin.
        #
        # ``is_frame_root`` is the ONE exception, and it is not a weakening:
        # when this walk is re-entered rooted AT a placed element we are
        # already inside that element's frame, so its own ``placement`` is the
        # thing that got us here rather than a reason to stop. Its geometry
        # sits in the same local frame as its children, and excluding it would
        # leave a placed ``Wall`` uncarved by the pipe its author ran through
        # it. Children recurse with the flag unset, so a NESTED placement still
        # ends this frame and opens its own.
        if not is_frame_root and getattr(elem, "placement", None) is not None:
            # Not discarded — HANDED ON. Everything inside a placed subtree
            # shares that subtree's pre-placement frame, so those AABBs are
            # mutually comparable even though they are not comparable with
            # anything outside it. ``_collect_frames`` runs the pairing over
            # each such subtree separately; see ``apply_displacement``.
            if _roots_out is not None:
                _roots_out.append(elem)
            return
        # 2. SAID OUTRIGHT. Whole subtree, like clause 1 — it replaces the
        # ``placement=Transform(origin=Point(x=0, y=0, z=0))`` idiom that
        # bought this exemption by accident, and an exemption that stopped at
        # the container would let an exempt assembly's own parts eat the thing
        # the assembly deliberately overlaps. Nothing to read through
        # ancestors here: the walk prunes at the first flag it meets, so the
        # flag on the element the author named already governs everything
        # under it, including parts added after that line.
        if getattr(elem, "_no_carve", False):
            return
        # 3. ANCHORED. An anchored child's position is authored as a
        # RELATIONSHIP to its host
        # ("1000 along, 2000 up, on that wall"), so sitting inside or against
        # that host is the NORMAL case, not the accidental overlap the inferred
        # carve exists to resolve: a corbel anchored to a wall would otherwise
        # eat the wall it hangs off. ``.add()`` is the carver — world-coordinate
        # containment is the author saying "this occupies that space"; anchoring
        # is the author saying "this belongs to that element". ``placement=``
        # subtrees have the same exemption one line up, and the two are mutually
        # exclusive by construction, so without this an anchored detail has NO
        # way out of the pass at all.
        #
        # ``_anchor_resolved``, not ``_anchor_spec``, is the durable marker:
        # ``frames._bake_child`` CONSUMES the spec (sets it to ``None``) once it
        # has baked world coordinates, and displacement runs post-bake — keying
        # on the spec alone would skip nothing where it matters. The spec is
        # still checked for the pre-bake state (a caller that collects before
        # ``resolve_child_anchors``), where the child's AABB is in parent-local
        # coords and is unjudgeable for exactly the reason ``placement=`` is.
        #
        # The WHOLE subtree is skipped, like ``placement=``. When only geometry
        # primitives could be anchored this was a distinction without a
        # difference — a Box has no ``_elements`` — but a container can now be
        # anchored, and then its children are the anchored ASSEMBLY: the bake
        # moved them by the same relationship that moved their container, so a
        # corbel's two bricks abut the gable for exactly the reason the corbel
        # does. Recursing would have exempted the bracket and let the bricks
        # inside it eat the wall it hangs off.
        #
        # WS-A §1.1 folded the former ``if tx.is_opening(elem): return`` row
        # INTO this one. A Window/Door was never exempt for being a
        # Window/Door — it is exempt for the reason every anchored child is:
        # it has no world coordinates of its own (v20 removed offset=/sill=),
        # so it can only reach ``_elements`` through ``.anchor()``, and
        # ``executor._opening_placement_errors`` makes ``.add()``-ing one a
        # compile error. ``brings_void`` stays in the condition as a NAMED
        # backstop for internal callers that reach ``generate_ifc`` without
        # ``validate_project_report`` (tests, tooling): an opening's children
        # are opening-LOCAL, so walking into them would judge them as world
        # coordinates and manufacture phantom carves. One rule, one branch —
        # and the reference's exemption table has one row for it, not two.
        if (getattr(elem, "_anchor_resolved", False)
                or getattr(elem, "_anchor_spec", None) is not None
                or tx.brings_void(elem)):
            return
        if tx.is_mesh(elem):
            aabb = _mesh_aabb(elem)
            if aabb is not None:
                meshes.append((elem, aabb))
        elif tx.is_solid(elem):
            # MM_PER_METER: this pass runs POST-normalize, so a Material's mm
            # facts have to be converted into the model's metres. See the unit
            # invariant in ``extent``'s module docstring.
            aabb = padded_aabb(elem, MM_PER_METER)
            if aabb is not None:
                solids.append((elem, aabb))
        for child in (getattr(elem, "_elements", None) or []):
            _walk(child)

    if _root is not None:
        _walk(_root, is_frame_root=True)
        return solids, meshes

    for storey in project.storeys:
        for elem in storey.elements:
            _walk(elem)
    for site in getattr(project, "sites", None) or []:
        _walk(site)
    return solids, meshes


def _collect_frames(project) -> List[Tuple[object, List, List]]:
    """``[(frame_root, solids, meshes)]`` — one entry per COMPARABLE frame.

    The first entry is the world frame (``frame_root`` is ``None``) and is
    collected by one walk, in its order — so a model with no ``placement=`` produces one group
    identical to before, down to the append order that
    ``boolean_tree.balanced_union`` pairs positionally.

    Every later entry is one ``placement=`` subtree.

    **Why a subtree is its own frame rather than an exemption.** A placed
    element's AABB is in PRE-placement authoring coordinates, so it cannot be
    compared with anything outside. But
    everything INSIDE it is in those same coordinates, so those comparisons
    are exactly as sound as any unplaced pair, and need no matrix at all. The
    old code threw that away with the ambiguous case, which is why an
    ``.add()``ed solid inside a placed wing carved nothing: no error, no
    warning, and a solid block where the hole should be.

    Cross-frame carving stays out, and that is the honest limit: judging a
    placed solid against an unplaced one needs the resolved matrix, and
    ``resolve_placement_matrices`` runs later than this pass.
    """
    roots: List = []
    solids, meshes = _collect(project, roots)
    frames: List[Tuple[object, List, List]] = [(None, solids, meshes)]

    # Breadth-first over the roots the world walk handed on. A nested
    # ``placement=`` is already a placement-engine error, so in practice this
    # queue drains one level deep; draining it properly costs nothing and
    # keeps the pass from depending on that being true.
    seen: set = set()
    while roots:
        root = roots.pop(0)
        if id(root) in seen:
            continue
        seen.add(id(root))
        nested: List = []
        # The SAME walk, re-entered at this root. See ``_collect``'s ``_root``.
        sub_solids, sub_meshes = _collect(None, nested, _root=root)
        if sub_solids or sub_meshes:
            frames.append((root, sub_solids, sub_meshes))
        roots.extend(nested)
    return frames


def _overlaps(a: Aabb, b: Aabb, eps: float = _EPS) -> bool:
    """Positive AABB overlap on every axis (touching faces don't count).

    The comparison is the whole of this function's content; the per-axis
    depths come from ``extent.overlap_depths``, which
    ``ifc.space_boundaries`` also reads — with the OPPOSITE verdict about a
    flush face, which is exactly why the arithmetic is shared and only the
    comparison is not. See that function's docstring."""
    return all(d > eps for d in overlap_depths(a, b))


def _contains(outer: Aabb, inner: Aabb, eps: float = _EPS) -> bool:
    """True when ``inner``'s AABB lies wholly inside ``outer``'s."""
    (o0, o1), (i0, i1) = outer, inner
    return all(i0[k] >= o0[k] - eps and i1[k] <= o1[k] + eps for k in range(3))


def _label(e) -> str:
    """Best available identity for a warning — canonical if the naming pass has
    run, else the leaf, else the class."""
    return (getattr(e, "_canonical_name", None)
            or getattr(e, "name", None)
            or f"<anonymous {type(e).__name__}>")


def _carve_label(e) -> str:
    """The CANONICAL NAME a carve is reported and asserted against.

    Most bodies in a real model are anonymous — ``Wall(name="north")`` holding
    an unnamed ``Box`` — and by the naming rules an anonymous element has no
    canonical name of its own: it BELONGS TO its nearest named ancestor. So
    that ancestor is what a carve on it is reported as.

    This is not a cosmetic choice. It is what makes ``assert_carved`` writable:
    an author names walls and rafters, never wall bodies, so
    ``assert_carved("wall:north", by="sweep:rafter_s_0")`` has to resolve even
    though the thing physically carved is a ``Box`` two levels down. Reporting
    ``<anonymous Box>`` (or ``box in wall:north``) would give the author a name
    they cannot write, which is a name for nothing.

    Two anonymous bodies of one named wall therefore collapse to one line. That
    is the same resolution the naming pass gives the author, so nothing is lost
    that they could have said.
    """
    node, seen = e, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        canonical = (getattr(node, "_canonical_name", None)
                     or getattr(node, "name", None))
        if canonical:
            return canonical
        node = getattr(node, "_parent", None)
    return f"<anonymous {type(e).__name__}>"


def _joint_ids(elem) -> set:
    """Every ``miter()`` joint ``elem`` takes part in, INCLUDING through its
    ancestors.

    The id is recorded on whatever ``miter()`` was called on, and since
    that may be a CONTAINER — while this pass pairs the leaf SOLIDS underneath
    it. Reading the element alone therefore found nothing on exactly the models
    the container miter was added for, and the auto-carve fired on a joint the
    author had already resolved: measured, one redundant boolean per mitered
    pair, which is CSG depth spent for a near-no-op on the models that have
    least of it to spare.

    Walked at DISPLACEMENT time rather than stamped down the subtree at
    ``miter()`` time, for the reason the derivation itself is deferred: parts
    added after the ``miter()`` line would miss a stamp, and "it depends where
    in the file you wrote it" is the property this feature exists to remove.
    """
    ids = set()
    seen = set()
    node = elem
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        ids.update(getattr(node, "_miter_joints", None) or ())
        node = getattr(node, "_parent", None)
    return ids


def _carve_direction(elem):
    """The ``add(carve=)`` direction governing ``elem``, read through ancestors,
    NEAREST DECLARATION WINS. ``None`` when nobody declared one.

    Same failure as ``_joint_ids`` above, in the other subtree-scoped author
    statement. Copying the direction onto every descendant at ``.add()`` time
    breaks
    twice:

    * **a child added LATER fell out of scope.** ``st.add(asm, carve="self")``
      followed by ``asm.add(part)`` left ``part`` unstamped, so the inversion
      silently reverted to the default for it. Measured: the pair comes back
      the other way round. ``carve="none"`` was immune only by accident —
      ``_walk`` prunes that subtree at walk time, so it never reads the
      descendants at all — and ``carve="other"`` was masked because it names
      the default anyway. Only ``"self"``, the one direction that CHANGES
      anything, actually lost its meaning.
    * **an outer declaration CLOBBERED an inner one.**
      ``asm.add(child, carve="other")`` then ``st.add(asm, carve="self")``
      rewrote the child to ``"self"`` — the enclosing add overruling a
      statement the author wrote explicitly, one line deeper.

    Reading rather than stamping fixes both, and inverts the precedence to the
    one that matches how the DSL reads everywhere else: the innermost thing the
    author said about an element is the thing that governs it.
    """
    node = elem
    seen = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        d = getattr(node, "_carve", None)
        if d is not None:
            return d
        node = getattr(node, "_parent", None)
    return None


def _are_mitered(a, b) -> bool:
    """True when ``a`` and ``b`` are an authored ``miter()`` joint — they share a
    joint id (``BimElement._miter_joints``), on themselves or on an ancestor.
    Mitered pairs must NOT auto-carve: the miter clips are the corner
    resolution, so the carve is a redundant no-op that only deepens the CSG.
    Empty on either side ⇒ not a joint (the common case), so non-mitered
    elements are unaffected."""
    ja = _joint_ids(a)
    jb = _joint_ids(b)
    return bool(ja) and bool(jb) and not ja.isdisjoint(jb)


# ---------------------------------------------------------------------------
# Narrow phase — does this carve remove anything at all?
# ---------------------------------------------------------------------------

def _obb(elem, cache: dict):
    """``elem``'s exact oriented box, or ``None`` if it is not a right-angled
    volume.

    Memoised in a caller-owned dict rather than stamped onto the element: one
    occupant is tested against many hosts, so the box is worth keeping, but it
    is derived from geometry that a later pass may move. A cache that lives
    exactly as long as one ``apply_displacement`` call cannot go stale; an
    attribute on the model can, and would do it silently.
    """
    key = id(elem)
    hit = cache.get(key)
    if hit is not None:
        return hit[1]
    try:
        box = convex_solid_obb(elem, MM_PER_METER)
    except Exception as exc:            # pragma: no cover - defensive
        logger.warning("displacement: oriented box for %s failed (%s) — the "
                       "carve trigger falls back to the padded AABB for it",
                       _label(elem), exc)
        box = None
    # The element is stored beside its box, not discarded: ``id()`` is unique
    # only among LIVE objects, so without a reference a collected element could
    # hand its address to a new one and with it the wrong geometry.
    cache[key] = (elem, box)
    return box


def _removes_nothing(host, occ, cache: dict) -> bool:
    """True only when the two solids PROVABLY do not intersect.

    The padded AABB that got this pair here is a broad phase, loose on purpose:
    it pads a member by its section reach so a member takes part in
    displacement at all, and the price is a boolean handed to every pair that
    merely comes close. Those booleans remove nothing, and a boolean that
    removes nothing is not free — it adds vertices, it adds one level to the
    host's CSG chain, and web-ifc drops the deepest chains outright.

    So the broad phase proposes and this disposes, on the solid rather than on
    a box around it. For the shapes it answers for, that distinction does not
    exist: a ``Box`` and a straight rectangular-section member ARE oriented
    boxes, so the separating-axis test is exact, not a bound.

    ``False`` is the safe answer and is what everything unknown returns — a
    curved solid, a mesh, a bent member, a shaped profile. Those pairs carve
    exactly as they do today.
    """
    a = _obb(host, cache)
    if a is None:
        return False
    b = _obb(occ, cache)
    if b is None:
        return False
    return obbs_are_disjoint(a, b)


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------


def apply_displacement(project):
    """Carve every host by the solids overlapping it (see module docstring).

    Idempotent — guarded by ``project._displacement_applied``. Mutates in place
    and returns the project.
    """
    if getattr(project, "_displacement_applied", False):
        return project

    from lite_step.models import taxonomy as tx

    # WHO CARVED WHOM. Recorded HERE because this is the only place both
    # identities still exist: a solid host is handed an anonymous
    # ``model_copy`` of the occupant, so nothing downstream can read
    # ``host._cuts`` and say who put it there. Read by the compile-time carve
    # report and by ``Project.assert_carved`` — see ``carve_pairs``.
    pairs: List[Tuple[str, str, str]] = []

    frames = _collect_frames(project)
    # Every mesh hosts. A mesh is matter, and a solid sunk into matter
    # displaces it — there is no second rule for which matter (see _collect).
    # The pairing loop below reads `solids` and `meshes` directly rather than a
    # concatenation, so that it can slice the solid half to the earlier hosts.

    # For each occupant, carve EVERY host it overlaps. The trigger is the same
    # for both host representations — positive AABB overlap (abutting faces
    # don't count) — because both carve mechanisms remove exactly the
    # intersection and are no-ops outside it:
    #   * solid host → native IFC boolean (a DIFFERENCE operand may extend
    #     beyond the host — CSG removes only the shared volume). A partial or
    #     through penetration (column through a slab, deck through a wall)
    #     carves automatically; `.void()`/`.opening()` remain the explicit
    #     rungs when the penetration must be schedulable.
    #   * mesh host (terrain) → manifold3d boolean: only the buried part is
    #     removed, the above-grade excess is free.
    # DIRECTION RULE (solid hosts only): ``.add()`` is directional — the ADDED
    # geometry is the carver, the acted-upon element's existing geometry is the
    # carvee. Generalised over siblings this is AUTHORING ORDER (the tree
    # traversal order ``_collect`` yields): the LATER solid carves every
    # EARLIER solid it overlaps, never the reverse. Authoring order is
    # construction order — new construction displaces what already stands
    # (body first, then the bars that carve it; the wall first, then the deck
    # punched through it). One-directional carve keeps the rendered union
    # intact ((earlier − later) ∪ later = earlier ∪ later — seamless at
    # junctions); mutual carve would render the symmetric difference (a hole
    # where two wall bodies meet). Consequence, deliberate: an earlier solid
    # fully inside a later one is carved to nothing — the later add claimed
    # that space.
    # Every overlapped earlier host is carved (not just the tightest): a pipe
    # through two walls must carve both. Nesting (C in B in A) stays correct —
    # the extra direct A−C operand is a geometric no-op inside the A−B cut.
    # id(mesh) -> (mesh, [occupants]). Occupants arrive by three routes but are
    # carved identically — each is subtracted; a non-overlapping one is a harmless
    # no-op (never an error: a void may be precautionary, and the foundation pad is
    # a deliberate no-op on flat sites):
    #   * inferred — a buried solid the enclosure trigger paired with this mesh;
    #   * explicit .difference() operands in the mesh's own _cuts (e.g. the pad);
    #   * .void() operands on a SPATIAL container, routed to its mesh children.
    mesh_carves: dict = {}
    # A one-cell list so the pairing loops below can increment it without
    # a nonlocal — they are nested functions in this scope.
    undeclared = [0]
    # Pairs the broad phase proposed and the narrow phase measured as
    # removing nothing. Reported, never silent: the number is how much
    # CSG the padded AABB was manufacturing, and it is the thing to watch
    # if a model's depth budget starts climbing.
    inert = [0]
    # The dropped pairs themselves, for the verbose carve report. A count
    # says how much noise the pad made; the list says WHICH joint is not
    # there, which is the question an author actually has when a member
    # looks attached in the render and is holding nothing.
    inert_pairs: List[Tuple[str, str]] = []
    # Oriented boxes for the narrow phase, memoised for this call only.
    obb_cache: dict = {}
    # Realised (host, occupant) ELEMENT pairs, kept alongside the label pairs
    # so the escape check below can ask geometric questions the labels cannot
    # answer. Same lifetime as `obb_cache`.
    realised: List[Tuple[Any, Any]] = []

    def _mc(mesh):
        return mesh_carves.setdefault(id(mesh), (mesh, []))

    # ONE PAIRING PASS PER COMPARABLE FRAME. The world frame first, in the
    # order it always had — so a model with no ``placement=`` runs exactly
    # the loop it ran before, including the ``host._cuts`` append order that
    # ``boolean_tree.balanced_union`` pairs POSITIONALLY. Then one pass per
    # placed subtree, over its own children in its own frame.
    for _frame_root, solids, meshes in frames:
        for j, (occ, occ_aabb) in enumerate(solids):
            # SOLID hosts, earlier ones only. The direction rule is "later carves
            # earlier", so a host at index >= j never acts — scanning all of them
            # discards exactly half AFTER paying for `host is occ`,
            # `_overlaps` and `tx.is_mesh`. Slicing to `range(j)` makes the index
            # test structurally true, and `host is occ` structurally false, so both
            # guards go away with the wasted iterations.
            #
            # ORDER IS UNCHANGED, and that is load-bearing rather than incidental:
            # `hosts = solids + meshes` already visited every solid before every
            # mesh for a given `j`, so splitting the loop preserves `host._cuts`
            # append order, `mesh_carves` insertion order and the `pairs` list. The
            # first of those feeds `boolean_tree.balanced_union`, which pairs
            # operands POSITIONALLY — reordering it would move bytes and CSG depth.
            for i in range(j):
                host, host_aabb = solids[i]
                if not _overlaps(host_aabb, occ_aabb):
                    continue
                if _are_mitered(host, occ):
                    # Authored joint (miter()): the miter clips already resolve
                    # the overlap into a flush corner, so the auto-carve is
                    # redundant — a near-no-op that only deepens the CSG (web-ifc
                    # drops the deepest chain). Skip it; corner geometry is
                    # byte-identical. Non-mitered overlaps still carve.
                    continue
                if (_removes_nothing(host, occ, obb_cache)
                        or _sheltered_by_parent(host, occ, obb_cache)):
                    # NARROW PHASE. The padded AABB above is a broad phase and
                    # says "these might touch"; this says they measurably do
                    # not, on the real solids. A carve here would subtract
                    # empty space — no geometry change, one more level of CSG.
                    #
                    # Silent for an INFERRED pair: nobody asked for it, tree
                    # order alone proposed it, and dropping it is the whole
                    # point. Loud for a DECLARED one, because there the author
                    # said these two meet and the geometry says they do not —
                    # a joint that is not there is exactly the failure that is
                    # invisible in a render (the element still draws; it is
                    # just structurally attached to nothing).
                    if _carve_direction(occ) is not None or \
                            _carve_direction(host) is not None:
                        logger.warning(
                            "displacement: %s and %s were declared a carve "
                            "pair but their solids do not intersect — nothing "
                            "is removed and no joint is made. Check the "
                            "placement: members that only touch are not "
                            "joined.", _label(host), _label(occ))
                    inert[0] += 1
                    inert_pairs.append((_carve_label(host), _carve_label(occ)))
                    continue
                # ANNIHILATION (loud). The direction rule is "later carves
                # earlier", and its documented consequence is that an earlier
                # solid wholly inside a later one is carved to NOTHING — the
                # later add claimed that space. That is deliberate and stays.
                # The hazard is that it happens in total silence: author
                # the bars before the beam body instead of after and the
                # reinforcement simply disappears from the model, with no
                # error, no warning, and geometry that still compiles.
                #
                # AABB containment, not volume: the solid path is deliberately
                # declarative (IfcBooleanResult trees, never evaluated), so
                # there is no cheap true volume here. Containment is a
                # NECESSARY condition for annihilation, which is why this is a
                # warning rather than an error — full displacement can be
                # intended.
                #
                # It cannot fire on correct code: the check reads the EARLIER
                # element, and correctly-ordered rebar is the LATER add.
                # ``add(occ, carve="self")`` INVERTS this pair: the newcomer
                # yields to what is already there instead of claiming its
                # space. Read off the newcomer because that is the element
                # whose ``.add()`` carried the argument — the incumbent's own
                # declaration governed the adds IT was the newcomer in, and
                # has no say here.
                if _carve_direction(occ) == "self":
                    if _contains(host_aabb, occ_aabb):
                        # The mirror of the annihilation warning below, and the
                        # reason it is a WARNING and not an error: the compiler
                        # did exactly what it was told and the answer happens
                        # to be empty. An impossible state is user error, not
                        # code error. It still has to SAY so — the failure mode
                        # is an element that is simply not there.
                        logger.warning(
                            "displacement: %s was added with carve=\"self\" and "
                            "lies wholly inside %s, so it yields ALL of its "
                            "geometry — %s is EMPTIED and will not appear in "
                            "the model. carve=\"self\" is for an element that "
                            "pokes out of what it is trimmed against.",
                            _label(occ), _label(host), _label(occ))
                    _carve_solid_host(occ, host)
                    # Mechanism stays "inferred": this IS an inferred carve,
                    # and the inversion is already carried by the loser/winner
                    # order. Encoding it in the mechanism string instead would
                    # break every consumer that compares it by equality — the
                    # carve report and four tests do.
                    pairs.append((_carve_label(occ), _carve_label(host),
                                  "inferred"))
                    realised.append((occ, host))
                    continue
                if _contains(occ_aabb, host_aabb):
                    logger.warning(
                        "displacement: %s is wholly inside %s, which was added "
                        "AFTER it — the later add claims that space, so %s is "
                        "carved to nothing and will not appear in the model. "
                        "If you meant it to survive, add it after %s "
                        "(authoring order is construction order).",
                        _label(host), _label(occ), _label(host), _label(occ))
                # the direction rule above: later carves earlier only.
                _carve_solid_host(host, occ)
                pairs.append((_carve_label(host), _carve_label(occ), "inferred"))
                realised.append((host, occ))
                # An UNDECLARED pair is decided by tree order alone and flips
                # if the two adds are swapped. Counted on the side rather than
                # tagged into the mechanism string, which consumers compare by
                # equality; the residual trends to zero as a model migrates to
                # explicit carve=.
                if (_carve_direction(occ) is None
                        and _carve_direction(host) is None):
                    undeclared[0] += 1

            # MESH hosts. Every mesh carves, regardless of index — a mesh is not
            # in `solids`, so the direction rule does not apply to it. Visited
            # AFTER the solids for this `j`, exactly as `solids + meshes` did.
            for host, host_aabb in meshes:
                if _overlaps(host_aabb, occ_aabb):
                    _mc(host)[1].append(occ)

    # A ``.void()`` on a SPATIAL container (an IfcSpatialStructureElement — Site/
    # Space) can't host an IfcOpeningElement, so it is realised by carving the
    # container's TERRAIN meshes (``mesh_boolean(child − operand)``). The operands
    # are consumed via the container's ``_voids``. Terrain-wrapper targeting
    # (taxonomy.iter_terrain_meshes: meshes inside terrain wrappers): a site
    # void clears the GROUND — it must
    # never eat sibling groundwork (the gravel IfcEarthworksFill bed) or scanned
    # context meshes. (A Space has no terrain mesh and its void is a validator
    # error, so in practice only Site voids route here — but the rule stays
    # kind-based, not Site-specific.)
    spatial = [c for c in (getattr(project, "sites", None) or []) if tx.is_spatial(c)]
    spatial += [e for st in project.storeys for e in st.elements if tx.is_spatial(e)]
    for container in spatial:
        voids = list(getattr(container, "_voids", None) or [])
        if voids:
            for child in tx.iter_terrain_meshes(container):
                _mc(child)[1].extend(voids)

    # Explicit ``.difference()`` on a Mesh (``mesh.difference(shape)``): a tessellated
    # mesh can't be an IFC boolean operand, so it is realised here as mesh CSG. The
    # operands live in the mesh's ``_cuts`` and are consumed
    # (``collect_consumed_operand_ids`` skips ``_cuts``) → they never render.
    # EVERY mesh folds its own cuts — an explicit boolean is a geometry feature,
    # independent of terrain semantics (a gravel bed may be sculpted too).
    for mesh_elem, _aabb in meshes:
        cuts = list(getattr(mesh_elem, "_cuts", None) or [])
        if cuts:
            _mc(mesh_elem)[1].extend(cuts)
        # A ``.clip()`` on a Mesh is realised here too (manifold trim_by_plane);
        # register the mesh even when it has no occupant/cut so a clip-only mesh
        # still gets carved.
        if getattr(mesh_elem, "_clips", None):
            _mc(mesh_elem)

    for mesh_elem, occupants in mesh_carves.values():
        occupants, subsumed = _drop_subsumed_operands(occupants)
        for occ, container in subsumed:
            # Loud, because a dropped operand is a carve the author can see in
            # the model and will not see in the file. Says WHICH solid absorbed
            # it, so "why is there no notch" has an answer without a rebuild.
            logger.info(
                "displacement: %s does not carve %s — it is wholly inside %s, "
                "which carves the same host, so the subtraction is already "
                "made (no geometry change, one boolean saved)",
                _label(occ), _label(mesh_elem), _label(container))
        _carve_mesh_host(mesh_elem, occupants)
        for occ in occupants:
            pairs.append((_carve_label(mesh_elem), _carve_label(occ), "mesh"))

    # Explicit named operands, so the record answers "does this pair carve"
    # rather than "did the inference produce it" (§6.5). ``.difference()`` and
    # ``.void()`` are subtractions the author WROTE, and an assertion must not
    # break when a later workstream moves a pair from one mechanism to the
    # other. Anonymous operands are skipped: the displacement copies above are
    # anonymous by construction and are already in the list.
    # An opening cuts through everything in it, its host's own anchored
    # details included — the ONE exception to the anchored exemption, argued
    # in ``_iter_opening_carvees``. It runs last because it must see the tree
    # the sibling carves left behind, and it appends to ``_cuts`` like they do.
    pairs.extend(_carve_openings_through_details(project))
    # ...and so does a ``.void()`` hole, for the identical reason. Same list,
    # same moment, different frame — see ``_carve_voids_through_details``.
    pairs.extend(_carve_voids_through_details(project))

    inert_authored: List[Tuple[str, str, str]] = []
    pairs.extend(_explicit_carve_pairs(project, inert_authored))


    try:
        project._carve_inert_authored = inert_authored
        project._carve_pairs = pairs
        # Read by the carve report. Stashed rather than recomputed: the
        # declaration lives on the DSL elements, and by report time the
        # report only has the (loser, winner, mechanism) label triples.
        project._carve_undeclared = undeclared[0]
        project._carve_inert = inert[0]
        project._carve_inert_pairs = inert_pairs
        project._displacement_applied = True
    except Exception:  # pragma: no cover - non-pydantic stand-ins in tests
        pass
    return project


#: Authored mechanisms whose operand coordinates are WORLD-space, so the same
#: oriented box that answers for a host answers for them.
#:
#: ``_openings`` is deliberately absent. ``.opening(along=…)`` is expressed in
#: the HOST's frame, not the world's, so testing an opening with a world-space
#: box would compare two different coordinate systems and report a miss for
#: every correctly-placed opening in the corpus. Cf. workspace memory
#: `feedback_opening_along_is_host_frame_not_world`, where exactly that
#: confusion produced two wrong issues in one day.
_WORLD_FRAME_AUTHORED = (("_cuts", "difference"), ("_voids", "void"))


def _sheltered_by_parent(host, occ, cache: dict) -> bool:
    """True when ``occ`` cannot reach ``host`` because its PARENT cannot.

    Containment is transitive through disjointness: if ``occ`` lies wholly
    inside a solid that is provably disjoint from ``host``, then ``occ`` is
    provably disjoint from ``host`` too — **whatever shape occ is**. That is
    what makes this sound for the shapes the narrow phase cannot decide.

    It exists because of an asymmetry that was producing real, useless CSG. A
    rebar ``Bar`` is a swept disk, not a right-angled volume, so
    ``convex_solid_obb`` returns ``None`` for it and ``_removes_nothing``
    answers ``False`` — keep the carve, the safe default. Its host wall's body
    IS a ``Box``, so the SAME pair against the wall is decided and dropped.
    The result: the wall correctly stops carving a rafter it does not touch,
    while 23 bars *inside that wall* keep carving it. Measured on the corpus:
    300 such booleans across five systems, every one of them removing nothing.

    The parent is resolved to the solids it actually owns. A ``Wall`` is a
    container, not a solid — it holds its shape in a ``Box`` child — so asking
    the container for an oriented box returns ``None`` and decides nothing. A
    first version did exactly that and could never fire.

    Only case (1) — child WHOLLY INSIDE parent — is eligible. A child that
    pokes out, or sits outside entirely, may legitimately touch what its parent
    does not: the miter leaves room for children so they miter individually, so
    reaching past the parent's own solid is designed behaviour. Those keep
    their carves, which is why this cannot suppress a real joint.
    """
    parent = getattr(occ, "_parent", None)
    if parent is None or parent is host:
        return False

    solids = []
    if _obb(parent, cache) is not None:
        solids.append(parent)
    else:
        for child in (getattr(parent, "_elements", None) or []):
            if child is not occ and _obb(child, cache) is not None:
                solids.append(child)
    if not solids:
        return False

    occ_box = padded_aabb(occ, MM_PER_METER)
    for solid in solids:
        if not _removes_nothing(host, solid, cache):
            return False          # part of the parent DOES reach host
    return any(_contains(padded_aabb(solid, MM_PER_METER), occ_box)
               for solid in solids)


def _explicit_carve_pairs(project, inert_authored=None) -> List[Tuple[str, str, str]]:
    """``(loser, winner, mechanism)`` for every AUTHORED subtraction in the
    model — ``.difference()`` / ``.void()`` operands and ``.opening()`` fills.

    The reference already defines a carve as ANY subtraction realised on a
    host, so the record has to cover all of them or ``assert_carved`` would
    quietly mean "the inference produced this", which §6.5 forbids.

    **An authored subtraction that removes nothing is not recorded as a
    carve.** Recording one makes the two mechanisms built to make carves
    observable both confirm a subtraction that never happened: the
    ``carve:`` report printed the pair, and ``assert_carved`` was satisfied by
    it, while the host's volume did not move by a cubic millimetre. Measured on
    a collar with one deliberately misplaced ``.difference()``: the compile
    output is byte-identical to the same model with the operand moved onto the
    collar, and the host measures 0.026100 m³ either way.

    That equality is the whole problem. An author — a person or an agent —
    whose cut silently did nothing has no signal to correct from: exit 0, the
    same report, the same depth line, and no way to see the geometry. The
    observed cost is a build that spent six compile-and-look passes on one
    rafter void cut, which is what this makes impossible: the first pass now
    says the cut removed nothing, and names both sides.

    Loud rather than fatal. ``.difference()`` on a disjoint solid is
    well-formed and answerable — the answer is "unchanged" — so the compiler
    warns and keeps going, per the DSL's standing rule that a degenerate but
    well-formed request is the author's error to see, not the compiler's to
    raise on.
    """
    out: List[Tuple[str, str, str]] = []
    seen: set = set()
    cache: dict = {}

    def walk(elem):
        if id(elem) in seen:
            return
        seen.add(id(elem))
        host = _carve_label(elem)
        for attr, mechanism in _WORLD_FRAME_AUTHORED:
            for operand in (getattr(elem, attr, None) or []):
                named = bool(getattr(operand, "_canonical_name", None)
                             or getattr(operand, "name", None))
                if _removes_nothing(elem, operand, cache):
                    # Warned for anonymous operands too, and that is the case
                    # that most needed it: an unnamed operand never reached the
                    # carve report at all, so a missed cut left NO trace in the
                    # compile output — not even a wrong one.
                    logger.warning(
                        "%s: .%s() operand %s does not touch it — the "
                        "subtraction removes nothing and the element is "
                        "emitted whole. The two solids are provably disjoint, "
                        "so this is a placement error in the operand, not a "
                        "tolerance. Nothing was carved.",
                        host, mechanism,
                        _carve_label(operand) if named else "<anonymous>")
                    if inert_authored is not None:
                        inert_authored.append(
                            (host,
                             _carve_label(operand) if named else "<anonymous>",
                             mechanism))
                    continue
                if named:
                    out.append((host, _carve_label(operand), mechanism))
        for operand in (getattr(elem, "_openings", None) or []):
            # Host-frame; see `_WORLD_FRAME_AUTHORED`. Recorded exactly as
            # before, untested.
            if getattr(operand, "_canonical_name", None) or getattr(
                    operand, "name", None):
                out.append((host, _carve_label(operand), "opening"))
        for attr in ("_elements", "_openings"):
            for child in (getattr(elem, attr, None) or []):
                walk(child)

    for storey in getattr(project, "storeys", None) or []:
        for elem in storey.elements:
            walk(elem)
    for site in getattr(project, "sites", None) or []:
        walk(site)
    return out


def carve_pairs(project) -> List[Tuple[str, str, str]]:
    """Every ``(loser, winner, mechanism)`` this compile realised, or ``[]``.

    ``[]`` before ``apply_displacement`` has run — which is a real answer, not
    a missing one, because nothing has carved anything yet.
    """
    return list(getattr(project, "_carve_pairs", None) or [])


def carve_report_lines(project) -> List[str]:
    """One ``carve: <loser> <- <winner>`` line per realised pair, sorted.

    Printed on EVERY compile beside ``csg depth:``. Displacement direction is
    authoring order, and until this line existed the only way to observe it
    was to render the model and look.
    """
    lines = {f"carve: {loser} <- {winner}"
             for loser, winner, _mech in carve_pairs(project)}

    # Deduped on the LINE, not on the (loser, winner, mechanism) triple: the
    # mechanism is diagnostic only (§6.5) and must not double-print a pair, and
    # two anonymous bodies of one named wall are one pair as far as any name an
    # author can write is concerned.
    return sorted(lines)


def near_miss_report_lines(project) -> List[str]:
    """One ``near-miss:`` line per pair the narrow phase dropped, sorted.

    Deliberately NOT part of :func:`carve_report_lines`. That function's length
    IS the reported carve count, so folding these in would silently inflate the
    number the whole report exists to make trustworthy — a near miss is the
    absence of a carve, and counting it as one would say the opposite of what
    happened.

    Printed only under ``LITESTEP_CARVE_REPORT=all``, where they answer the
    question a render cannot: which intended joint is not there. The element
    still draws in full; it is simply attached to nothing.
    """
    return sorted(
        f"near-miss: {loser} x {winner} (no intersection, no carve)"
        for loser, winner in getattr(project, "_carve_inert_pairs", None) or ())


def inert_authored_report_lines(project) -> List[str]:
    """One ``inert-cut:`` line per AUTHORED subtraction that removed nothing.

    Unlike :func:`near_miss_report_lines` these print on EVERY compile, not
    only under ``LITESTEP_CARVE_REPORT=all``. A near miss is the inference
    declining a pair the author never asked for — routine, and noise at the
    default verbosity. This is an instruction the author WROTE that did not
    happen, which is never routine and is exactly what the author is unable to
    see any other way.
    """
    return sorted(
        f"inert-cut: {host} .{mech}() {operand} removed nothing"
        for host, operand, mech
        in getattr(project, "_carve_inert_authored", None) or ())


def check_carve_assertions(project) -> None:
    """Resolve every ``proj.assert_carved()`` against what actually carved.

    Runs AFTER ``apply_displacement`` — the records do not exist before it —
    and raises :class:`CarveAssertionError` on the first unsatisfied pair.

    Resolution is the reference grammar ``Anchor(host=)`` already uses
    (``naming.resolve_host``), against the canonical names that DID carve, so
    an author writes the same fragment in both places.
    """
    from lite_step.compiler.naming import resolve_host

    assertions = list(getattr(project, "_carve_assertions", None) or [])
    if not assertions:
        return
    pairs = carve_pairs(project)
    losers = {loser for loser, _w, _m in pairs}
    winners = {winner for _l, winner, _m in pairs}
    realised = {(loser, winner) for loser, winner, _m in pairs}

    for target, cutter in assertions:
        t_status, t_name, t_cands = resolve_host(target, losers)
        c_status, c_name, c_cands = resolve_host(cutter, winners)
        if t_status == "ok" and c_status == "ok" and (t_name, c_name) in realised:
            continue
        raise CarveAssertionError(_assertion_message(
            target, cutter, t_status, t_name, t_cands,
            c_status, c_name, c_cands, pairs,
            getattr(project, "_carve_inert_authored", None) or []))


def _assertion_message(target, cutter, t_status, t_name, t_cands,
                       c_status, c_name, c_cands, pairs,
                       inert_authored=()) -> str:
    """Say which half failed and what DID happen — a bare 'assertion failed'
    on a carve sends the author back to rendering the model, which is the
    thing the assertion exists to replace.

    When the author DID write the subtraction and it simply removed nothing,
    that is the answer and it goes first. Otherwise the message reads "nothing
    carved X", which is true and sends the author looking for a missing
    ``.difference()`` line that is right there in front of them.
    """
    head = (f"assert_carved({target!r}, by={cutter!r}) is not satisfied: ")
    for host, operand, mech in inert_authored:
        if target in host or (operand != "<anonymous>" and cutter in operand):
            return (head + f"you wrote {host}.{mech}({operand}), and it removed "
                    f"nothing — the two solids are provably disjoint, so no "
                    f"carve was realised. Move the operand onto the host. "
                    + _what_did_happen(pairs))
    if t_status == "ambiguous":
        return (head + f"{target!r} matches several carved elements "
                f"({', '.join(t_cands)}) — write more leaf-first pairs.")
    if c_status == "ambiguous":
        return (head + f"{cutter!r} matches several carving elements "
                f"({', '.join(c_cands)}) — write more leaf-first pairs.")
    if t_status == "miss":
        return (head + f"nothing carved {target!r} in this compile. "
                + _what_did_happen(pairs))
    if c_status == "miss":
        return (head + f"{cutter!r} carved nothing in this compile. "
                + _what_did_happen(pairs))
    carvers = sorted({w for lo, w, _m in pairs if lo == t_name})
    return (head + f"{t_name} was carved, but by {', '.join(carvers)} — not by "
            f"{c_name}. Displacement direction is authoring order: the LATER "
            f"proj.add()/.add() carves what already stands, so if you meant "
            f"{c_name} to win, it has to be added after {t_name}.")


def _what_did_happen(pairs) -> str:
    if not pairs:
        return "Nothing carved anything at all this compile."
    shown = sorted({f"{lo} <- {w}" for lo, w, _m in pairs})
    head = shown[:8]
    tail = "" if len(shown) <= 8 else f" (+{len(shown) - 8} more)"
    return "What carved what: " + "; ".join(head) + tail + "."


def _carve_solid_host(host, occupant) -> None:
    """Append an anonymous copy of the occupant's real solid to ``host._cuts``.

    The copy is consumed as a boolean operand (native ``IfcBooleanResult`` in the
    host's local frame, via the generator's existing ``_create_hole_solid_local``
    machinery); the original occupant is a distinct object and still renders."""
    # NON-validating deep copy: a post-normalize occupant holds float-meter
    # geometry in int-typed fields (Bar.diameter/bend_radius), which the
    # re-validating derivation constructor would reject. ``model_copy`` copies
    # state verbatim (fresh identity → consumed as operand while the original,
    # a distinct object, still renders) and clears the name so the cut tool is
    # anonymous.
    cut = occupant.model_copy(deep=True, update={"name": None})
    host._cuts.append(cut)
    logger.info("displacement: solid host %r carved by %r",
                getattr(host, "ifc_name", None) or type(host).__name__,
                getattr(occupant, "ifc_name", None) or type(occupant).__name__)


# ---------------------------------------------------------------------------
# Openings carve the host's OWN details (the one exception to the anchored
# exemption)
# ---------------------------------------------------------------------------


def _openings_of(elem) -> List[object]:
    """Every opening this element cuts through itself, in emission order.

    Two lists, one meaning. ``.anchor()`` puts a Window/Door in ``_elements``
    (what a container's opening pass reads) and ``.opening()`` puts it in
    ``_openings`` (what a standalone solid's reads); both stamp the same
    ``ChildAnchor`` and both become the same ``IfcOpeningElement``. The trigger
    is ``brings_void``, never a type test — see ``taxonomy.brings_void``.
    """
    from lite_step.models import taxonomy as tx

    return ([c for c in (getattr(elem, "_elements", None) or [])
             if tx.brings_void(c)]
            + [c for c in (getattr(elem, "_openings", None) or [])
               if tx.brings_void(c)])


def _iter_opening_carvees(host, body, openings):
    """The host's own detail solids — the things standing in its holes.

    **This is the one place an ``.anchor()``ed child is NOT exempt, and the
    exemption is otherwise untouched.** ``_collect``'s reasoning still holds
    for every sibling-to-sibling pair: an anchored child's position is a
    RELATIONSHIP to its host, so entering that host is what anchoring MEANS
    and a corbel must not eat the wall it hangs off. An OPENING is not a
    sibling. It is a hole THROUGH the host, declared on the host, and matter
    inside a hole is not matter — a window does not stop at the reinforcement
    it happens to cross. So the opening carves, and nothing else about
    anchoring changes.

    It has to be the anchored children or the feature is empty: a Wall refuses
    a second unanchored prism ("a Wall takes exactly one body element"), so
    every stirrup, tie and bar in a wall reaches the tree through
    ``.anchor()`` and through nothing else.

    Excluded, and each for a reason that is not "it is a detail":

    * the **body** — the host product already carries the
      ``IfcRelVoidsElement``; carving it again would be one redundant boolean
      per opening, paid for in CSG depth (see ``csg_depth``);
    * the **openings themselves and their subtrees** — an opening's joinery
      lives in the hole by definition, and its children are opening-LOCAL;
    * ``placement=`` subtrees — pre-placement coordinates, unjudgeable, same
      as everywhere else in this module;
    * ``carve="none"`` subtrees — the author saying so outright, and the escape
      hatch for a sill or lintel meant to sit in the reveal.
    """
    from lite_step.models import taxonomy as tx

    skip = {id(o) for o in openings}
    if body is not None:
        skip.add(id(body))

    def walk(elem):
        if id(elem) in skip:
            return
        if getattr(elem, "placement", None) is not None:
            return
        if getattr(elem, "_no_carve", False):
            return
        if tx.brings_void(elem):
            return
        if tx.is_solid(elem):
            yield elem
        for child in (getattr(elem, "_elements", None) or []):
            yield from walk(child)

    for child in (getattr(host, "_elements", None) or []):
        yield from walk(child)


#: A frame counts as axis-aligned when its run axis is within this of a world
#: axis. Openings are placed from a snapped frame, so a genuinely orthogonal
#: wall lands on an exact 0/±1 and the tolerance only absorbs float noise.
_AXIS_EPS = 1e-9


def _void_operand(prism):
    """A boolean operand shaped exactly like ``prism``, or ``None``.

    ``None`` on a SKEW host — see :func:`_carve_one_host`, which turns it into
    a named warning. The boolean-operand vocabulary is closed
    (``generator._create_hole_solid_local``: box-mode ``Box``, ``Revolve``,
    ``Pipe``/``Bar``, path-form ``Sweep``), and none of those expresses a
    rotated rectangular prism. A ``Box`` around a skew void would carve the
    void's BOUNDING box — a hole wider than the window, in the one direction
    nobody looks at — so the skew case is refused out loud rather than
    approximated. Every wall in the corpus is orthogonal; a contour-``Extrude``
    operand is the fix when one is not.

    Built under ``suspend_strict``: this runs post-normalize, so the
    coordinates are float METERS and int-mm validation would reject every one
    of them. Anonymous, so it renders nowhere and is consumed as an operand.
    """
    from lite_step.models.elements import Box, Point
    from lite_step.strict import suspend_strict

    (x0, y0, z0), (x1, y1, z1) = prism.aabb()
    if not prism.is_axis_aligned(_AXIS_EPS):
        return None

    with suspend_strict():
        return Box(start=Point(x=x0, y=y0, z=z0),
                   end=Point(x=x1, y=y1, z=z1))


def _carve_openings_through_details(project) -> List[Tuple[str, str, str]]:
    """Cut every opening's void out of the details standing in it.

    Returns the ``(loser, winner, mechanism)`` rows for the carve report.

    The frame comes from ``frames.opening_host_frame`` and the box from
    ``frames.opening_void_prism`` — the same two calls the generator makes to
    place the void it emits — so the notch in a bar is the hole in the wall by
    construction rather than by agreement.

    A CONTAINER whose hole the generator hands to its leaves is skipped, for
    the reason ``_carve_voids_through_details`` skips its own set: since
    those leaves each receive a real ``IfcRelVoidsElement``, so a boolean would
    carve a second time for nothing — one redundant operand per detail per
    hole, paid for in CSG depth. Before that this pass ran on such a container
    and notched its details for a window that was **not in the file** (#728
    measured ``IFCOPENINGELEMENT: 0  IFCWINDOW: 0  IFCBOOLEANRESULT: 2``); the
    refusal stopped it by making the tree unbuildable, and this skip is what
    keeps it stopped now that the tree compiles.

    The predicate is ``opening_reaches_children`` and NOT
    ``void_reaches_children``: a ``Slab``/``Roof`` distributes its ``.void()``
    and does not distribute its ``.opening()``, so skipping it here would drop
    a carve nothing replaces.
    """
    from lite_step.compiler import frames
    from lite_step.ifc.entity_cache import snap_to_precision

    out: List[Tuple[str, str, str]] = []
    seen: set = set()

    def walk(elem):
        if id(elem) in seen:
            return
        seen.add(id(elem))

        if not opening_reaches_children(elem):
            openings = _openings_of(elem)
            if openings:
                _carve_one_host(elem, openings, out, snap_to_precision, frames)

        for attr in ("_elements", "_openings"):
            for child in (getattr(elem, attr, None) or []):
                walk(child)

    for storey in getattr(project, "storeys", None) or []:
        for elem in storey.elements:
            walk(elem)
    for site in getattr(project, "sites", None) or []:
        walk(site)
    return out


def _carve_one_host(host, openings, out, snap, frames) -> None:
    """One host's openings against one host's details."""
    body = frames._body_of(host)
    frame = frames.opening_host_frame(body if body is not None else host,
                                      snap=snap, divisor=MM_PER_METER)
    if frame is None:
        # No usable frame — the generator raises on exactly this case a moment
        # later, and raising twice for one defect names the second reason.
        # Silence here is a deferral, not a swallow.
        return

    carvees = [(c, padded_aabb(c, MM_PER_METER))
               for c in _iter_opening_carvees(host, body, openings)]
    carvees = [(c, a) for c, a in carvees if a is not None]
    if not carvees:
        return

    for opening in openings:
        prism = frames.opening_void_prism(opening, frame)
        if prism is None:
            continue
        void_aabb = prism.aabb()
        for carvee, carvee_aabb in carvees:
            if not _overlaps(void_aabb, carvee_aabb):
                continue
            operand = _void_operand(prism)
            if operand is None:
                logger.warning(
                    "displacement: the void of %s crosses %s but the host is "
                    "not axis-aligned, so the detail keeps its material where "
                    "the opening cuts. A skew opening has no boolean-operand "
                    "shape today (box-mode Box / Revolve / Pipe / Bar / "
                    "path-Sweep only) and a bounding box would carve wider "
                    "than the window.",
                    _label(opening), _label(carvee))
                continue
            carvee._cuts.append(operand)
            # LOUD by default. A detail losing a notch it did not ask for is
            # the kind of change an author has to be able to find without
            # re-rendering, and this is the only line that records it.
            logger.info(
                "displacement: %s carved by the void of %s "
                "(an opening cuts through the details in it — attach the detail "
                "with carve=\"none\" to let it stand in the reveal)",
                _label(carvee), _label(opening))
            out.append((_carve_label(carvee), _carve_label(opening),
                        "opening-void"))


def _voids_of(elem) -> List[object]:
    """Every ``.void()`` hole cut through THIS element's product.

    Two lists again, and for the reason the generator reads two: a ``Wall``
    carries its body ON the ``IfcWall``, so ``wall.void(tool)`` and
    ``body.void(tool)`` land on one product and cut one hole
    (``generator._create_wall`` calls ``_process_voids_on_element`` for both).
    Reading only the host's own list would carve the details for the first
    spelling and not for the second, and they are the same hole.
    """
    from lite_step.compiler import frames

    out = list(getattr(elem, "_voids", None) or [])
    body = frames._body_of(elem)
    if body is not None and body is not elem:
        out += list(getattr(body, "_voids", None) or [])
    return out


def _void_label(operand) -> str:
    """A ``.void()`` hole's identity for the carve report.

    ``.void(tool, name=…)`` names the HOLE, not the tool — the tool is consumed
    and never emitted — so ``compiler.naming`` stamps the canonical on the
    operand as ``_void_canonical``, and that is the name an author can write in
    an ``assert_carved``. An anonymous void has none, and a consumed operand
    has no place in the tree for ``_carve_label`` to walk up from, so it falls
    back to the operand's own leaf or its class.
    """
    return getattr(operand, "_void_canonical", None) or _carve_label(operand)


def _carve_voids_through_details(project) -> List[Tuple[str, str, str]]:
    """Cut every ``.void()`` hole out of the details standing in it.

    Returns the ``(loser, winner, mechanism)`` rows for the carve report.

    The twin of :func:`_carve_openings_through_details`, and the argument is
    the one that pass already makes: **matter inside a hole is not matter.**
    ``.void()`` and ``.opening()`` are two rows of ONE table (see
    ``BimElement.void``) — both emit an ``IfcOpeningElement`` linked by
    ``IfcRelVoidsElement``, and they differ only in whether the hole is NAMED
    and whether joinery FILLS it. Neither difference is about the detail
    standing in the hole, so a rebar cage loses its material for either. It did
    not, and the failure was silent: a wall's reinforcement ran straight
    through every ``.void()`` doorway with no warning and no error, in a corpus
    fixture. Skills could not route around it — spelling the hole
    ``.opening()`` instead would emit a second ``IfcWindow`` inside the wall
    and double-count every window in the schedule.

    Where the two passes genuinely differ is the FRAME, and it is why this is
    not a third argument to the opening pass. An ``.opening()`` child is
    authored in the host's opening-local frame, so that pass derives a world
    prism (``frames.opening_host_frame`` + ``opening_void_prism``) and then
    reduces it to an axis-aligned ``Box``, refusing a skew host out loud
    because a bounding box would carve wider than the window. A ``.void()``
    operand is ALREADY absolute world geometry: there is no frame to derive and
    no reduction to make, so the cut is a copy of the operand itself — exact
    for a rotated or curved tool, and with no skew case to refuse.

    Everything else is shared with the opening pass on purpose, via
    ``_iter_opening_carvees``: the body is exempt (the host product already
    carries the ``IfcRelVoidsElement``), so are ``placement=`` subtrees, so are
    the operands themselves, and ``carve="none"`` is still the author's way to
    leave a sill standing in the reveal.

    Two host kinds are skipped outright, each because its void is ALREADY
    realised somewhere this pass would only duplicate — see
    ``ifc.voids.void_reaches_children`` for containers, and ``tx.is_spatial`` for
    Site/Space, whose void is not an ``IfcOpeningElement`` at all: a spatial
    container cannot host one, so ``apply_displacement`` realises a site void
    by carving the site's TERRAIN meshes. That void is a clearing volume the
    size of a plot, and letting it reach solids would take a retaining wall
    out with the ground.
    """
    from lite_step.models import taxonomy as tx

    out: List[Tuple[str, str, str]] = []
    seen: set = set()

    def walk(elem):
        if id(elem) in seen:
            return
        seen.add(id(elem))

        if not tx.is_spatial(elem) and not void_reaches_children(elem):
            voids = _voids_of(elem)
            if voids:
                _carve_one_void_host(elem, voids, out)

        for attr in ("_elements", "_openings"):
            for child in (getattr(elem, attr, None) or []):
                walk(child)

    for storey in getattr(project, "storeys", None) or []:
        for elem in storey.elements:
            walk(elem)
    for site in getattr(project, "sites", None) or []:
        walk(site)
    return out


def _carve_one_void_host(host, voids, out) -> None:
    """One host's ``.void()`` holes against one host's details."""
    from lite_step.compiler import frames

    body = frames._body_of(host)
    carvees = [(c, padded_aabb(c, MM_PER_METER))
               for c in _iter_opening_carvees(host, body, voids)]
    carvees = [(c, a) for c, a in carvees if a is not None]
    if not carvees:
        return

    for void in voids:
        # ``padded_aabb`` rather than the raw AABB, for the reason it exists:
        # a curved tool with a straight axis-aligned path (a ``Bar``, a
        # ``Pipe`` run) has a DEGENERATE point AABB, and a positive-overlap
        # test against one can never pass. The pad is a bound, so it can only
        # over-select — and an over-selected pair costs one boolean against a
        # solid the operand misses, which is a geometric no-op.
        void_aabb = padded_aabb(void, MM_PER_METER)
        if void_aabb is None:
            # Not a shape this compiler can bound, so it cannot be tested for
            # overlap. LOUD, because the alternative is details keeping their
            # material inside a hole with nothing said — the exact silence
            # this pass exists to end.
            logger.warning(
                "displacement: the void %s has no measurable extent, so the "
                "details standing in it keep their material. A void operand "
                "must be a solid the compiler can bound (Box / Extrude / "
                "Sweep / Revolve / Pipe / Bar).",
                _void_label(void))
            continue
        for carvee, carvee_aabb in carvees:
            if not _overlaps(void_aabb, carvee_aabb):
                continue
            # The operand ITSELF, copied — not a bounding box. It is already
            # absolute world geometry in the same frame the carvee is, and it
            # is already a shape the boolean-operand vocabulary accepts (the
            # host is cutting with it). Anonymous, so the copy renders nowhere
            # and is consumed as an operand, exactly as in
            # ``_carve_solid_host``; the original is a distinct object and
            # still cuts the host.
            carvee._cuts.append(
                void.model_copy(deep=True, update={"name": None}))
            # LOUD by default, same as the opening pass: a detail losing a
            # notch it did not ask for has to be findable without a re-render,
            # and this is the only line that records it.
            logger.info(
                "displacement: %s carved by the void %s "
                "(a void cuts through the details in it — attach the detail with "
                "carve=\"none\" to let it stand in the reveal)",
                _label(carvee), _void_label(void))
            out.append((_carve_label(carvee), _void_label(void), "void-hole"))


def _manifold_status(m) -> str:
    """``manifold3d``'s error status as a readable name, never raising.

    The status is the ONLY channel manifold3d uses to report an invalid input —
    the constructor does not throw — so a diagnostic that omits it cannot
    distinguish "genuinely empty" from "rejected as non-manifold", which are
    opposite bugs."""
    try:
        return str(m.status())
    except Exception:  # pragma: no cover - older manifold3d without .status()
        return "unavailable"


def _edge_defects(faces) -> Tuple[int, int]:
    """``(hole_edges, non_manifold_edges)`` — reported separately on purpose.

    Summing them under one name is what misdirected: five edges each
    shared by FOUR faces were reported as "boundary edges", so the hunt went
    looking for a hole that did not exist."""
    try:
        from lite_step.ifc.geometry import mesh_edge_defects
        return mesh_edge_defects(faces)
    except Exception:  # pragma: no cover - diagnostics must never mask the error
        return -1, -1


def _exact_box_bounds(elem):
    """The world box a solid occupies EXACTLY, or ``None``.

    Only an axis-aligned ``Box`` with nothing removed from it qualifies, and
    that restriction is the whole reason this is sound. For every other solid
    the AABB is a strict OVER-approximation — a rotated ``Extrude``, a
    ``Revolve``, a diagonal ``Sweep`` all leave space inside their bounding
    box that is not inside the solid. Treating those as containers would drop
    a carve that mattered, which is the one failure this optimisation must
    never have.

    Disqualifiers, each because it removes material the box still covers:
    ``_cuts`` (a hole an operand could sit in), ``_clips`` (a half-space trim
    — ``_carve_mesh_host`` applies these to the operand, so the kept volume is
    smaller than the box), and ``placement=`` (coordinates are pre-placement,
    so the box is not where the geometry stands).
    """
    from lite_step.models.elements import Box

    if not isinstance(elem, Box):
        return None
    if getattr(elem, "placement", None) is not None:
        return None
    if getattr(elem, "_cuts", None) or getattr(elem, "_clips", None):
        return None
    s_, e_ = elem.start, elem.end
    if s_ is None or e_ is None:
        return None
    return ((min(s_.x, e_.x), min(s_.y, e_.y), min(s_.z, e_.z)),
            (max(s_.x, e_.x), max(s_.y, e_.y), max(s_.z, e_.z)))


def _contains_aabb(outer, inner, eps: float = 1e-9) -> bool:
    """Is ``inner`` wholly inside ``outer``? Both are ``(min3, max3)``."""
    (ox0, oy0, oz0), (ox1, oy1, oz1) = outer
    (ix0, iy0, iz0), (ix1, iy1, iz1) = inner
    return (ix0 >= ox0 - eps and iy0 >= oy0 - eps and iz0 >= oz0 - eps
            and ix1 <= ox1 + eps and iy1 <= oy1 + eps and iz1 <= oz1 + eps)


def _drop_subsumed_operands(occupants):
    """Drop operands whose carve another operand on the same host already makes.

    ``H - A - B == H - A`` whenever ``B`` is wholly inside ``A``. Reinforcement
    is the case that motivates it: `loadbearing-structure` puts **171 bar
    operands on the terrain mesh**, and every bar sits at least its cover
    distance inside a slab or wall body that carves the same terrain. All 171
    booleans are exact no-ops — same result, 171 extra mesh booleans, each one
    a full manifold3d subtraction against a terrain that can run to hundreds of
    thousands of triangles.

    **Soundness rests entirely on ``_exact_box_bounds``.** The container's AABB
    has to BE the container, or "inside the box" would not mean "inside the
    solid" and this would delete a real carve. The contained operand's bound
    may be an over-approximation in the other direction — that only makes the
    test stricter, so a subsumed operand is occasionally kept, never a
    load-bearing one dropped.

    Deliberately NOT keyed on type. "A Bar does not carve a Mesh" was the first
    shape proposed for this and it is narrower and less true: it would miss a
    ``Pipe`` sleeve inside a slab, a ``Sweep`` nosing inside a stair, a small
    ``Box`` detail inside a wall — and it would wrongly drop a bar that is
    genuinely exposed, standing in the ground with nothing around it.
    """
    if len(occupants) < 2:
        return list(occupants), []

    bounds = [(occ, padded_aabb(occ, MM_PER_METER)) for occ in occupants]
    containers = [(occ, box) for occ, box in
                  ((o, _exact_box_bounds(o)) for o in occupants) if box]
    if not containers:
        return list(occupants), []

    kept, dropped = [], []
    for occ, aabb in bounds:
        host_box = None
        if aabb is not None:
            for other, box in containers:
                if other is occ:
                    continue
                if _contains_aabb(box, aabb):
                    host_box = other
                    break
        if host_box is None:
            kept.append(occ)
        else:
            dropped.append((occ, host_box))
    return kept, dropped


def _carve_mesh_host(mesh_elem, occupants) -> None:
    """Carve a ``Mesh`` host by its occupants with an exact, watertight 3D boolean
    (``manifold3d``). Rewrites ``mesh_elem.vertices`` / ``.faces`` in place; the emitter then serializes the carved mesh.

    A non-overlapping occupant is a harmless no-op — never an error (a void may be
    precautionary; the foundation pad is a deliberate no-op on flat sites). The loud
    ``DisplacementError`` is reserved for real failures: an operand that can't be
    tessellated (``_occupant_mesh``), an operand ``manifold3d`` REFUSES (by status,
    not by raising — see the guard inside), or a subtraction that throws."""
    from lite_step.models.primitives import Point

    label = getattr(mesh_elem, "ifc_name", None) or type(mesh_elem).__name__
    try:
        from manifold3d import Manifold, Mesh as MMesh
    except ImportError:  # pragma: no cover - dep declared, present in prod/CI
        logger.warning("displacement: manifold3d unavailable — mesh carve skipped "
                       "for %r", label)
        return

    m_verts = np.array([[p.x, p.y, p.z] for p in mesh_elem.vertices], dtype=np.float32)
    m_faces = np.array(mesh_elem.faces, dtype=np.uint32)
    try:
        host = Manifold(MMesh(vert_properties=m_verts, tri_verts=m_faces))
    except Exception as exc:
        logger.warning("displacement: mesh host %r is not a valid manifold (%s) — "
                       "carve skipped", label, exc)
        return
    if host.is_empty():
        logger.warning("displacement: mesh host %r produced an empty manifold "
                       "(manifold3d status %s) — carve skipped",
                       label, _manifold_status(host))
        return

    orig_vol = host.volume()
    carved = host
    for occ in occupants:
        ov, of = _occupant_mesh(occ)   # exact tessellation; raises DisplacementError on real failure
        occ_label = getattr(occ, "ifc_name", None) or type(occ).__name__
        try:
            operand = Manifold(MMesh(
                vert_properties=np.asarray(ov, dtype=np.float32),
                tri_verts=np.asarray(of, dtype=np.uint32)))
        except Exception as exc:
            raise DisplacementError(
                f"mesh carve of {label!r} by {occ_label!r} ({type(occ).__name__}) "
                f"failed: {exc}") from exc
        # THE OPERAND MUST BE A VALID MANIFOLD. ``Manifold(…)``
        # does NOT raise on a bad mesh — the ``except`` above cannot fire for
        # this — it returns a Manifold carrying an error STATUS, and manifold3d
        # then propagates that status through every boolean the value touches.
        # The result is not a wrong subtraction, it is not a subtraction at all:
        # even ``host + operand`` (a UNION, which cannot reduce volume) comes
        # back empty. So a single unwelded operand silently deleted an entire
        # terrain mesh — the bridge deck, which post-repair removes exactly
        # 0 m³ because it never intersected the ground.
        #
        # This is the loud twin of the host check above. It WARNS AND SKIPS
        # rather than raising. Refusing the carve is still
        # mandatory — an error-status operand poisons the boolean and would
        # erase the host — but refusing the whole COMPILE is a different and
        # much larger claim, and the wrong one: one bad chord out of ~250
        # members took an entire bridge build down, and the host surviving
        # un-carved is strictly better than no model at all. The degradation
        # is visible (a notch that should be there is not) and named in the
        # warning, so it is not the silent-wrongness case the loud-failure
        # rule exists for. Note ``is_empty()`` alone is the right test only
        # BEFORE the clip trim below — a fully-trimmed-away operand is a
        # legitimate no-op, and a plain empty Manifold subtracts cleanly (it
        # is the error status, not the emptiness, that poisons).
        if operand.is_empty():
            holes, non_manifold = _edge_defects(of)
            if holes and not non_manifold:
                why = ("The surface is OPEN — those edges are traversed by one "
                       "face each, so a face is missing.")
            elif non_manifold and not holes:
                why = ("The surface is CLOSED but non-manifold — those edges are "
                       "traversed by three or more faces, usually a coincident "
                       "interior wall that drop_coincident_faces should have "
                       "removed.")
            elif holes and non_manifold:
                why = ("Both an open patch and a non-manifold edge — repair the "
                       "non-manifold one first; it defeats the orientation pass "
                       "and can manufacture the rest.")
            else:
                why = ("Every edge is traversed by exactly two faces, so the "
                       "topology is sound and the winding or the geometry is "
                       "not — look for a self-intersection.")
            logger.warning(
                "displacement: %r does NOT carve %r — manifold3d rejected the "
                "operand (status %s; %d verts, %d faces, %d hole edges, "
                "%d non-manifold edges). An invalid operand would erase the "
                "host instead of carving it, so this one "
                "carve is skipped and the rest of the model is unaffected: "
                "%s expect a notch that is not in the file. %s "
                "Three repairs run before this point — weld_coincident_vertices, "
                "drop_coincident_faces, orient_faces_consistently, in that "
                "order — so this is a cause none of them covers "
                ".",
                occ_label, label, _manifold_status(operand), len(ov), len(of),
                holes, non_manifold, label, why)
            continue
        # Operand clips (spec-displacement-miter-clip, option b): a clipped
        # occupant excavates only its KEPT volume — trim the operand by its
        # own half-spaces before subtracting (same world-frame trim as the
        # receiver path below; a fully-trimmed-away operand is a no-op).
        operand = _trim_by_clips(operand, getattr(occ, "_clips", None),
                                 f"operand {occ_label!r}")
        try:
            carved = carved - operand
        except Exception as exc:
            raise DisplacementError(
                f"mesh carve of {label!r} by {occ_label!r} ({type(occ).__name__}) "
                f"failed: {exc}") from exc
    # .clip() half-spaces on the RECEIVER mesh. Origins are already meters
    # here (displacement runs post-normalize).
    carved = _trim_by_clips(carved, getattr(mesh_elem, "_clips", None), label)
    carved_vol = carved.volume()
    if abs(orig_vol - carved_vol) < 1e-9:
        return   # nothing removed (every occupant/clip was a no-op) — leave as-is

    # ANNIHILATION (loud) — the OUTPUT side. Every other guard in this function
    # reads an INPUT: the host manifold before any subtraction (``is_empty()``
    # above), or a degenerate operand (``_occupant_mesh``). None of them can see
    # a host that was valid going in and is gone coming out, and the early
    # return just above is the OPPOSITE case — it fires when NOTHING was
    # removed. So a carve that eats the whole terrain sailed straight through to
    # the assignment below, which set ``vertices = []`` / ``faces = []``: the
    # ground silently vanished from the model, and the only trace was an
    # ``info``-level "→ 0 verts".
    #
    # True VOLUME, not AABB containment: unlike the solid path (see the sibling
    # annihilation warning in ``apply_displacement``, which has no cheap volume
    # because its booleans are declarative and never evaluated), the mesh path
    # has already evaluated the boolean here — ``orig_vol`` and ``carved_vol``
    # are exact. So this is a SUFFICIENT condition, not a necessary one: when it
    # fires, the host really is gone.
    #
    # WARNING, not error, deliberately — same severity as the solid-side twin:
    #   * full displacement can be authored on purpose (a ``.clip()`` that trims
    #     a mesh entirely away, a site void that clears all of a small ground
    #     patch), and models that ship today rely on compiling;
    #   * a mesh carved to nothing still produces a valid, downloadable IFC —
    #     an empty ``IfcTriangulatedFaceSet``, not a broken file;
    #   * the failure mode this exists for is SILENCE, and a warning naming the
    #     host, the occupants and the volume lost ends the silence.
    if carved.is_empty() or (orig_vol > 0
                             and carved_vol <= orig_vol * _MESH_ANNIHILATION_RESIDUE):
        occ_names = ", ".join(_label(o) for o in occupants)
        by = occ_names or "its own .clip() half-space(s)"
        logger.warning(
            "displacement: mesh host %s is wholly consumed by %s — the carve "
            "removed %.6g m3 of %.6g m3, so it is carved to nothing and will "
            "not appear in the model. If it was meant to survive, check that "
            "the carving geometry covers only the part you meant to remove "
            "(an occupant with no placement= participates in the inferred "
            "carve; a placed one does not).",
            _label(mesh_elem), by, orig_vol - carved_vol, orig_vol)

    out = carved.to_mesh()
    # manifold3d is internally float32, so the carved verts carry single-precision
    # noise (-15.2 → -15.199999809265137). Snap to GEOMETRY_DECIMALS (micrometre)
    # — same 6-decimal cutoff the generator applies to computed dimensions
    #; mesh verts otherwise escape both the serializer bit-exact pass
    # (#408 keeps the noisy value) and the dimension snap. Lossless at
    # project scale, and topology (faces) is untouched so it stays watertight.
    from lite_step.ifc.entity_cache import GEOMETRY_DECIMALS
    verts = np.round(np.asarray(out.vert_properties, dtype=np.float64)[:, :3],
                     GEOMETRY_DECIMALS)
    faces = np.asarray(out.tri_verts)
    # Snapped float-meter Points — suspend the strict int-mm construction gate,
    # exactly like the meters-normalizer does for its own internal Points
    # ([[feedback_suspend_strict_for_internal_geometry]]).
    from lite_step.strict import suspend_strict
    with suspend_strict():
        mesh_elem.vertices = [Point(x=float(v[0]), y=float(v[1]), z=float(v[2]))
                              for v in verts]
    mesh_elem.faces = [(int(f[0]), int(f[1]), int(f[2])) for f in faces]
    logger.info("displacement: mesh host %r carved by %d occupant(s) → %d verts",
                label, len(occupants), len(verts))


def _trim_by_clips(manifold, clips, label: str):
    """Trim a manifold by ``.clip()`` half-spaces. ``trim_by_plane`` KEEPS the
    side the normal points at; the DSL normal points at the REMOVED side, so
    the negated normal is the keep-normal (pinned by test_clips). Used for
    both the receiver mesh and (spec-displacement-miter-clip) clipped carve
    OPERANDS."""
    if not clips:
        return manifold
    trimmed = manifold
    for hs in clips:
        n = np.asarray(hs.normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        keep = -n
        offset = float(np.dot(keep, [hs.origin.x, hs.origin.y, hs.origin.z]))
        try:
            trimmed = trimmed.trim_by_plane(normal=tuple(keep), origin_offset=offset)
        except Exception as exc:
            raise DisplacementError(
                f"mesh clip of {label} by plane normal={hs.normal} failed: "
                f"{exc}") from exc
    return trimmed


# ---------------------------------------------------------------------------
# Occupant tessellation (for the mesh-host boolean)
# ---------------------------------------------------------------------------


def _occupant_mesh(elem) -> Tuple[np.ndarray, np.ndarray]:
    """Tessellate an occupant into ``(verts Nx3 float64, faces Mx3 uint32)`` in
    WORLD space — its EXACT shape, no approximation:

    * ``Box``/``Extrude`` (prisms) → pure-numpy fast path (no ifcopenshell).
    * ``Revolve``/``Sweep``/``Pipe``/``Bar`` → exact ifcopenshell tessellation of
      the element's real product (``tessellate_solid_to_mesh``).

    Then ``placement=Transform`` (if any) is applied. Anything that can't be
    realised — unsupported shape, degenerate/empty tessellation, ``placement=Anchor``
    — raises ``DisplacementError`` (loud), never a silent AABB approximation."""
    from lite_step.models import taxonomy as tx
    from lite_step.models.elements import Box

    verts = faces = None
    kind = tx.tessellation_kind(elem)          # "prism" | "curved" | "mesh" | None
    if kind == "prism":
        if isinstance(elem, Box):
            verts, faces = _box_mesh(elem.start.x, elem.start.y, elem.start.z,
                                     elem.end.x, elem.end.y, elem.end.z)
        else:                                # Extrude
            m = _extrude_mesh(elem)
            if m is not None:
                verts, faces = m
    elif kind == "curved":
        from lite_step.ifc.generator import tessellate_solid_to_mesh
        m = tessellate_solid_to_mesh(elem)
        if m is not None:
            verts, faces = m
    elif kind == "mesh" and elem.vertices and elem.faces:
        verts = np.array([[p.x, p.y, p.z] for p in elem.vertices], dtype=np.float64)
        faces = np.array(elem.faces, dtype=np.uint32)
    # kind is None (not a geometric operand) → verts/faces stay None → loud below

    label = getattr(elem, "ifc_name", None) or type(elem).__name__
    if verts is None or len(verts) == 0 or len(faces) == 0:
        raise DisplacementError(
            f"mesh carve operand {label!r} ({type(elem).__name__}) could not be "
            f"tessellated into a solid — an unsupported or degenerate void shape")

    verts = _apply_operand_placement(elem, np.asarray(verts, dtype=np.float64), label)
    return verts, np.asarray(faces, dtype=np.uint32)


def _apply_operand_placement(elem, verts: np.ndarray, label: str) -> np.ndarray:
    """Apply a void operand's ``placement=Transform`` to its world-space verts.

    ``Transform`` is a self-contained affine (origin + rotations) → reuse
    ``transform_matrix``. ``Anchor`` needs a host frame the operand-carve path does
    not resolve → loud ``DisplacementError`` (author the void in absolute coords or
    with ``Transform``)."""
    placement = getattr(elem, "placement", None)
    if placement is None:
        return verts
    from lite_step.models import Transform
    if not isinstance(placement, Transform):
        raise DisplacementError(
            f"mesh carve operand {label!r} uses placement={type(placement).__name__} "
            f"— only Transform is supported on a void/difference operand (author the "
            f"shape in absolute coords, or orient it with placement=Transform)")
    from lite_step.compiler.placement import transform_matrix
    matrix = transform_matrix(placement)                    # 4x4, meters
    hom = np.hstack([verts, np.ones((len(verts), 1))])      # Nx4
    return (hom @ matrix.T)[:, :3]


# 12 triangles of an axis-aligned box, CCW outward.
_BOX_TRIS = np.array([
    [0, 2, 1], [0, 3, 2],  # -z (bottom)
    [4, 5, 6], [4, 6, 7],  # +z (top)
    [0, 1, 5], [0, 5, 4],  # -y
    [2, 3, 7], [2, 7, 6],  # +y
    [1, 2, 6], [1, 6, 5],  # +x
    [0, 4, 7], [0, 7, 3],  # -x
], dtype=np.uint32)


def _box_mesh(x0, y0, z0, x1, y1, z1) -> Tuple[np.ndarray, np.ndarray]:
    lo = (min(x0, x1), min(y0, y1), min(z0, z1))
    hi = (max(x0, x1), max(y0, y1), max(z0, z1))
    v = np.array([
        [lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]],
    ], dtype=np.float32)
    return v, _BOX_TRIS.copy()


def _extrude_mesh(ext) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Prism from a planar contour extruded by ``thickness`` along its plane
    normal. Contour is triangulated with shapely (already a dep)."""
    from shapely.geometry import Polygon
    from shapely.ops import triangulate

    pts = [(p.x, p.y, p.z) for p in (ext.contour or [])]
    if len(pts) < 3:
        return None
    contour = np.array(pts, dtype=np.float64)
    # Plane normal via Newell's method.
    normal = np.zeros(3)
    for i in range(len(contour)):
        a, b = contour[i], contour[(i + 1) % len(contour)]
        normal[0] += (a[1] - b[1]) * (a[2] + b[2])
        normal[1] += (a[2] - b[2]) * (a[0] + b[0])
        normal[2] += (a[0] - b[0]) * (a[1] + b[1])
    ln = np.linalg.norm(normal)
    if ln < 1e-12:
        return None
    normal /= ln
    thickness = float(getattr(ext, "thickness", 0.0)
                      or _material_thickness(ext, MM_PER_METER))
    if thickness <= 0:
        return None

    # Local 2D basis on the contour plane.
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[2]) > 0.9 else np.array([0.0, 0.0, 1.0])
    u = np.cross(ref, normal); u /= np.linalg.norm(u)
    w = np.cross(normal, u)
    origin = contour[0]
    uv = [((pt - origin) @ u, (pt - origin) @ w) for pt in contour]
    poly = Polygon(uv)
    if not poly.is_valid or poly.area <= 0:
        return None

    def to3d(u2, v2, off):
        return origin + u2 * u + v2 * w + off * normal

    verts: List[np.ndarray] = []
    faces: List[Tuple[int, int, int]] = []
    index: dict = {}

    def vid(u2, v2, off):
        key = (round(u2, 6), round(v2, 6), off)
        if key not in index:
            index[key] = len(verts)
            verts.append(to3d(u2, v2, off))
        return index[key]

    # Two caps (bottom off=0, top off=thickness) from shapely's triangulation,
    # constrained to points inside the polygon.
    for tri in triangulate(poly):
        (ax, ay), (bx, by), (cx, cy) = list(tri.exterior.coords)[:3]
        if not poly.contains(tri.representative_point()):
            continue
        b0, b1, b2 = vid(ax, ay, 0.0), vid(bx, by, 0.0), vid(cx, cy, 0.0)
        t0, t1, t2 = vid(ax, ay, thickness), vid(bx, by, thickness), vid(cx, cy, thickness)
        faces.append((b0, b2, b1))   # bottom (reversed for outward −normal)
        faces.append((t0, t1, t2))   # top
    # Side walls around the contour boundary.
    for i in range(len(uv)):
        a, b = uv[i], uv[(i + 1) % len(uv)]
        a0, a1 = vid(a[0], a[1], 0.0), vid(a[0], a[1], thickness)
        c0, c1 = vid(b[0], b[1], 0.0), vid(b[0], b[1], thickness)
        faces.append((a0, c0, c1))
        faces.append((a0, c1, a1))
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.uint32)


