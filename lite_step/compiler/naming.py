"""DSL v2.1 canonical-naming pass (hierarchical identity).

``name=`` is an optional lowercase LEAF (``^[a-z0-9_]+$``). The element's
**canonical name** — its ``IfcRoot.Name``, its manifest key, its patch-mode
identity — is derived from containment as ``type:leaf`` pairs, **LEAF-FIRST**:

    Window(name="left") inside Wall(name="north")  ->  window:left:wall:north

The canonical name is not a bottom-up property: an element's own ``name`` is
only a leaf, and the path in front of it is whatever containment happens to be.
This module is the **top-down tree walk** that computes it: starting
at each storey's top-level elements with an empty ancestor stack, it descends
through container children (``_elements``), ``.opening()`` attachments
(``_openings``) and boolean operands (``_cuts``/``_adds``/``_intersects``/
``_fills``), carrying the leaf-first stack of named-ancestor ``type:leaf``
pairs. Each element is stamped with a private ``_canonical_name``:

* **named** (``name`` set) -> its own ``type:leaf`` pair FIRST, then the
  named-ancestor pairs, joined by ``:``. Its pair is pushed onto the stack the
  children see.
* **anonymous** (``name is None``) -> ``_canonical_name = None`` (no uuid8
  fallback), and it contributes NO segment to the ancestor stack — anonymous
  geometry belongs to its nearest NAMED ancestor.

The stamp is read by the emitter (``ifc_name`` returns it) and by the
validator (canonical-name uniqueness + Anchor host resolution), so the pass
must run BEFORE generation. It is idempotent — re-running recomputes from the
tree — so it is safe to stamp at every entry point (validate, normalize,
generate).

``type`` segment = the lowercase class name; a generic ``Element`` uses its
``ifc_class`` with the ``Ifc`` prefix stripped and lowercased
(``IfcStair`` -> ``stair``).

This walk also STAMPS the ``_parent`` back-reference (``BimElement.parent``)
on every contained child, and — on request — counts how many times it reaches
each object, which is how :func:`find_shared_elements` detects one object
attached in two places. It also records every containment CYCLE it meets
(:func:`find_containment_cycles`). All three ride this walk rather than adding
their own, because it already visits exactly the right set of nodes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from lite_step.compiler.walkguard import WalkGuard

#: Private-attr lists the walk descends into. Container children and opening
#: attachments carry identity; boolean operands are usually anonymous, but a
#: NAMED operand roots its canonical at the host (spec §15 — walked here too).
_WALK_ATTRS = ("_elements", "_openings", "_cuts", "_adds", "_intersects", "_fills")

#: The subset of the walk that is CONTAINMENT, and therefore the subset that
#: stamps ``_parent``. A boolean operand is consumed into its host's shape
#: rather than contained by it, so it is walked for naming (a NAMED operand
#: roots its canonical at the host) but never parented — see
#: ``BimElement.parent``.
_CONTAINMENT_ATTRS = ("_elements", "_openings")


def type_segment(elem) -> str:
    """The lowercase ``type`` segment of an element's canonical pair.

    Generic ``Element`` maps by its ``ifc_class`` (``IfcStair`` -> ``stair``);
    every sugar container / primitive uses its class name lowercased.
    """
    # Lazy import — models import this module's package; avoid a cycle.
    from lite_step.models.elements import Element

    if isinstance(elem, Element):
        cls = elem.ifc_class
        if cls.startswith("Ifc"):
            cls = cls[3:]
        return cls.lower()
    return type(elem).__name__.lower()


def _walk(elem, ancestor_pairs: List[str], counts: Optional[dict] = None,
          guard: Optional[WalkGuard] = None,
          cycles: Optional[Dict[int, str]] = None) -> None:
    """Stamp ``elem`` and recurse, carrying the leaf-first named-ancestor stack.

    ``ancestor_pairs`` is ordered nearest-ancestor-first (index 0 = the
    immediate named ancestor's ``type:leaf`` pair), so a named element's
    canonical is ``[own_pair, *ancestor_pairs]`` joined — leaf-first.

    ``counts`` (when given) is bumped per visited object, which is how
    :func:`find_shared_elements` gets its answer for the price of one dict
    write per node instead of a second full traversal. Compile time is a
    first-class constraint here — the CSG-depth machinery exists for the
    same reason — and a standalone counting walk measured
    49 ms on a 5 000-element model purely to report nothing.

    ``guard`` is the ON-PATH re-entrancy guard (:mod:`compiler.walkguard`).
    Not a visited set: an element is legitimately reached once per path, and
    ``counts`` above depends on it being reached every one of those times —
    a visited set would silently un-report every shared element. Re-entry
    while the element is still on the current path is a CYCLE, and it is
    recorded into ``cycles`` (keyed by ``id()`` so one cycle is reported
    once, in insertion order) and NOT recursed into. Recording rather than
    raising, because the authoritative report is a list of compile errors
    naming every cycle — see :func:`find_containment_cycles`.
    """
    if guard is None:
        guard = WalkGuard()
    if not guard.enter(elem):
        if cycles is not None and id(elem) not in cycles:
            cycles[id(elem)] = _cycle_message(elem, guard.cycle_path(elem))
        return
    try:
        if counts is not None:
            counts[id(elem)] = counts.get(id(elem), 0) + 1
        leaf = getattr(elem, "name", None)
        if leaf:
            own_pair = f"{type_segment(elem)}:{leaf}"
            canonical = ":".join([own_pair, *ancestor_pairs])
            elem._canonical_name = canonical
            child_stack = [own_pair, *ancestor_pairs]
        else:
            elem._canonical_name = None
            # Anonymous contributes no segment — children root at the nearest
            # NAMED ancestor (the stack is passed through unchanged).
            child_stack = ancestor_pairs

        for attr in _WALK_ATTRS:
            containment = attr in _CONTAINMENT_ATTRS
            for child in getattr(elem, attr, None) or []:
                if containment:
                    child._parent = elem
                _walk(child, child_stack, counts, guard, cycles)

        # A NAMED `.void()` (``.void(tool, name="doorway")``) names the HOLE,
        # not the tool: the tool is consumed and never emitted, so the
        # canonical is built here from the leaf and the HOST's stack —
        # leaf-first, same grammar as every other canonical, with the
        # `opening` type segment the emitted IfcOpeningElement actually is.
        # Done in this pass rather than in the emitter because only the pass
        # knows the named-ancestor stack, and doing it twice is how two
        # spellings of one name start to disagree.
        for operand in getattr(elem, "_voids", None) or []:
            leaf = getattr(operand, "_void_leaf", None)
            operand._void_canonical = (
                ":".join([f"opening:{leaf}", *child_stack]) if leaf else None
            )
    finally:
        guard.leave(elem)


def stamp_canonical_names(project, count_visits: bool = False) -> "object":
    """Stamp ``_canonical_name`` on every storey + element (top-down).

    DSL v2.1: a Storey obeys the SAME segment rule as an element — a NAMED
    storey contributes ``storey:<leaf>`` to the ancestor stack every element
    under it carries (leaf-first: the element's own pairs come first, the
    storey pair is the outermost tail), while an ANONYMOUS storey contributes
    nothing. ``Project`` is never a segment — canonical paths stop at the
    storey. The storey's own ``_canonical_name`` (``storey:<leaf>`` or None) is
    stamped for the IfcBuildingStorey Name.

    Also re-stamps ``_parent`` on every contained element (NOT on boolean
    operands — see ``_CONTAINMENT_ATTRS``). The attach helpers set it at
    authoring time so it is available before any pass runs, but it cannot stay
    authoring-only: ``normalize_project_to_meters`` deep-copies the project and
    then REBUILDS its elements in metres, so a pointer written at authoring
    time survives into the copy still aimed at a pre-rebuild object — a
    back-reference that has escaped its own tree, and a world query resolving
    against it would read millimetre coordinates after normalization. Making
    it a derived property of the same top-down walk that computes the
    canonical name means the two describe one tree by construction.

    Returns the same project (stamped in place). Idempotent.

    With ``count_visits=True``, also publishes ``project._visit_counts`` — how
    many times the walk reached each object, keyed by ``id()``. Any value above
    1 is a shared element (see :func:`find_shared_elements`). Computed here
    rather than by its own pass because this walk already visits exactly the
    right set of nodes.

    Off by default because this pass runs at SEVERAL pipeline entry points
    (validate, generate, the placement resolver) while the count is wanted
    once. The dict write per node measured ~6 ms on a 5 000-element model —
    small, but there is no reason to pay it four times, and compile time is a
    stated constraint. ``validate_project_report`` is the one caller that asks.

    When counting is off the previous counts are CLEARED rather than left in
    place: a stale count dict describes a tree that no longer exists, which is
    the same escaped-reference hazard ``_parent`` has.

    Always publishes ``project._containment_cycles`` — the list of cycle
    messages :func:`find_containment_cycles` returns. Unlike the visit counts
    this is NOT opt-in: the on-path guard runs on every walk whether anyone
    asks or not (without it this function does not terminate on a cyclic
    tree), so the messages are already paid for. They are STRINGS rather than
    element references on purpose — a list of live objects hung off the
    project is the same escaped-reference hazard ``_parent`` has, and
    ``normalize_project_to_meters`` deep-copies the project.
    """
    counts: dict = {}
    cycles: Dict[int, str] = {}
    guard = WalkGuard()
    for storey in project.storeys:
        storey._parent = project
        for elem in storey.elements:
            elem._parent = storey
        leaf = getattr(storey, "name", None)
        if leaf:
            storey_pair = f"storey:{leaf}"
            storey._canonical_name = storey_pair
            ancestor = [storey_pair]
        else:
            storey._canonical_name = None
            ancestor = []
        for elem in storey.elements:
            _walk(elem, ancestor, counts if count_visits else None,
                  guard, cycles)

    # DSL Site containers live in ``project.sites`` (routed out of storeys so a
    # site-only model carries no phantom project). A Site is a named element in
    # its own right — ``Site(name="site")`` → ``site:site`` — and its Mesh
    # children root at it: ``Mesh(name="site_block")`` → ``mesh:site_block:site:site``.
    for site in getattr(project, "sites", []):
        site._parent = project
        _walk(site, [], counts if count_visits else None, guard, cycles)
    project._visit_counts = counts if count_visits else None
    project._containment_cycles = list(cycles.values())
    return project


# ---------------------------------------------------------------------------
# Tree-shape diagnostics — one object in two places, or one object inside
# itself
#
# Both read the SAME walk and the same :func:`_path_label`, because they are
# two readings of one question: how many times, and from where, does the walk
# reach a given object. A CYCLE is the case where one of those "wheres" is the
# object itself, and it is the more severe of the two — a shared element gets
# a wrong single-valued stamp, a cyclic one gets no stamp at all because the
# walk that computes it does not terminate.
# ---------------------------------------------------------------------------


def _path_label(elem) -> str:
    """Short, stable descriptor of one step in a containment path.

    The local ``name`` leaf, not ``_canonical_name`` — the canonical of a
    SHARED element is itself last-write-wins, so quoting it inside the message
    that reports the sharing would print the same string for every path and
    hide the very thing being reported.
    """
    leaf = getattr(elem, "name", None)
    return f"{type_segment(elem)}:{leaf}" if leaf else type_segment(elem)


def _cycle_message(elem, path: Tuple) -> str:
    """One containment cycle, as the compile-error string the author reads.

    ``path`` is the loop CLOSED — ``elem`` at both ends — so it reads as a
    loop rather than as a list that happens to start and end nearby. Labels
    are :func:`_path_label`, the same local-leaf spelling
    :func:`find_shared_elements` uses and for the same reason: the canonical
    name of an element inside a cycle is whatever truncated string the walk
    got to before it turned back, so quoting it would print the defect as if
    it were the identity.
    """
    return (
        f"{type(elem).__name__} '{_path_label(elem)}': containment CYCLE — "
        + " > ".join(_path_label(node) for node in path)
        + ". An element cannot contain itself: its canonical name, its frame "
        f"and its extent are all derived by descending containment, so on a "
        f"cyclic tree none of them has a finite answer and the walks that "
        f"compute them do not terminate. Detach one of the two attachments — "
        f"a container may hold a child, or be held by it, never both."
    )


def find_containment_cycles(project) -> List[str]:
    """Containment cycles in ``project``, as compile-error strings.

    A cycle is one object reachable from INSIDE itself:
    ``w1.anchor(w2); w2.anchor(w1)`` is two ordinary authoring lines, and
    ``a.add(b); b.add(a)`` between two facility ``SpatialElement``s is two
    more. Both were accepted at the authoring line and both took the compile
    down with a bare ``RecursionError`` raised inside :func:`_walk`, naming no
    element — before validation had produced a single message.

    An ERROR, never a warning, and unlike :func:`find_shared_elements` it did
    not need the shim-first sequence: there is no cyclic model to break. A
    shared element at least compiles to something (wrongly — the last write
    wins); a cyclic one compiles to nothing at all, so refusing it removes no
    working behaviour.

    Shaped exactly like :func:`find_shared_elements`: the answer is a
    by-product of the naming walk, which the compile path has already run, so
    the common case costs one attribute read. ``id()`` identity — two
    structurally identical elements are two objects — and insertion-ordered,
    so cycles come out in authoring order rather than in whatever order
    ``id()`` hashes to.
    """
    cycles = getattr(project, "_containment_cycles", None)
    if cycles is None:
        # A direct call from a test or a ``--q`` query, before any pass has
        # run. Same self-sufficiency :func:`find_shared_elements` has.
        stamp_canonical_names(project)
        cycles = project._containment_cycles
    return list(cycles)


def _iter_walk_paths(elem, path: List[str], out: List,
                     guard: Optional[WalkGuard] = None) -> None:
    """Every (element, containment path) pair reachable from ``elem``.

    Same descent as :func:`_walk` — the SAME ``_WALK_ATTRS`` — because the
    hazard reported below is precisely that this walk visits one object twice.
    A separate, subtly different walk would report a different set of elements
    from the one that actually gets double-stamped.

    ``path`` names the CONTAINERS traversed, not the element itself: two
    different attachment sites of one object must render as two different
    strings, or the warning cannot say where to look.

    Carries the same on-path guard as :func:`_walk`, for the same reason and
    with the same identity — this is the diagnosis walk, and a diagnosis that
    blows the stack reports nothing. Re-entry is PRUNED here rather than
    recorded: the cycle already has its own error from the walk above, and
    listing an element's infinite self-containing paths as "attached in N
    places" would report the cycle a second time under the wrong name.
    """
    if guard is None:
        guard = WalkGuard()
    if not guard.enter(elem):
        return
    try:
        here = [*path, _path_label(elem)]
        out.append((elem, " > ".join(here)))
        for attr in _WALK_ATTRS:
            for i, child in enumerate(getattr(elem, attr, None) or []):
                _iter_walk_paths(child, [*here, f"{attr.lstrip('_')}[{i}]"],
                                 out, guard)
    finally:
        guard.leave(elem)


def find_shared_elements(project) -> List[str]:
    """Elements reachable through MORE THAN ONE path, as warning strings.

    The compiler stamps ``_canonical_name``, ``_frame`` and ``_anchor_spec`` as
    **single-valued per object**. Reach one object through eight containment
    paths and it is stamped eight times, last write wins — and because
    uniqueness is validated per SCOPE, eight windows in eight walls never
    collide and nothing raises today. The model assumes a tree and nothing
    enforces it.

    This is what makes a bounding query on a shared element **undefined**
    rather than merely ambiguous: eight placements, one slot to record them in.
    It is a pre-existing silent bug independent of that feature.

    Returned as ERROR strings. The caller is
    ``executor.validate_project_report``, which does
    ``errors.extend(find_shared_elements(project))`` — so a shared element
    fails the compile.

    It began as a warning, on the shim-first sequence:
    refusing a second attachment is a breaking DSL change, so it warned until
    the skills corpus was clean. The corpus was measured clean (zero of 20
    models) and v10 promoted it to the refusal it is now. This paragraph said
    "warning" for some time after that was false, which is worth a moment:
    the function name, the return type and the call site are all neutral about
    severity, so nothing here contradicted the stale claim.

    Identity is ``id()``: two structurally identical elements built by two
    constructor calls are two objects and are fine; the same object appended
    twice is the defect. That is also why the fix is derivation-by-call —
    ``win(name=f"w{i}")`` yields a new object per placement.
    """
    # The answer is a by-product of the naming walk, which the compile path
    # has already run — so the common case (nothing shared, every model we
    # ship) costs one dict scan and stops here. Stamping when the counts are
    # absent covers a direct call from a test or a `--q` query, and is the
    # same self-sufficiency ``placement.resolve_placement_matrices`` has.
    counts = getattr(project, "_visit_counts", None)
    if counts is None:
        stamp_canonical_names(project, count_visits=True)
        counts = project._visit_counts
    if not any(c > 1 for c in counts.values()):
        return []

    # Only reached when something IS shared, so the diagnosis can be as
    # expensive as it needs to be. Re-stamp so the surviving canonical quoted
    # below is the one this tree would actually compile with.
    stamp_canonical_names(project)

    pairs: List = []
    for i, storey in enumerate(project.storeys):
        root = [f"storey:{storey.name}" if storey.name else f"storey[{i}]"]
        for elem in storey.elements:
            _iter_walk_paths(elem, root, pairs)
    for site in getattr(project, "sites", []):
        _iter_walk_paths(site, ["sites"], pairs)

    # Insertion-ordered so the warnings come out in authoring order rather
    # than in whatever order id() hashes to.
    paths_by_obj: dict = {}
    for elem, path in pairs:
        _, paths = paths_by_obj.setdefault(id(elem), (elem, []))
        paths.append(path)

    out: List[str] = []
    for elem, paths in paths_by_obj.values():
        if len(paths) < 2:
            continue
        surviving = getattr(elem, "_canonical_name", None)
        out.append(
            f"{type(elem).__name__} '{_path_label(elem)}': the SAME object is "
            f"attached in {len(paths)} places — "
            + "; ".join(paths)
            + f". The canonical name, frame and anchor are stamped once per "
            f"OBJECT, so all but the last are silently lost"
            + (f" (surviving canonical: {surviving})" if surviving else "")
            + ". Derive a new object per placement instead — "
            f"win(name=\"w0\"), win(name=\"w1\"), … — since "
            f"derivation-by-call copies the template."
        )
    return out


# ---------------------------------------------------------------------------
# Anchor host resolution — segment-aligned suffix match ( ruling)
# ---------------------------------------------------------------------------


def host_matches(fragment: str, canonical: str) -> bool:
    """True when ``fragment`` is a segment-aligned suffix of ``canonical``.

    The match is on whole ``type:leaf`` pairs from the TAIL (leaf-first, so
    the tail is the outermost container). ``fragment`` must be an even number
    of segments (whole pairs) and equal the last ``len(fragment)`` segments of
    ``canonical``. Example: ``wall:north`` matches both ``wall:north`` and
    ``box:body:wall:north``; ``wall:nor`` (not a segment) and
    ``body:wall:north`` (odd, not whole pairs) match neither.
    """
    frag_segs = fragment.split(":")
    if not fragment or len(frag_segs) % 2 != 0:
        return False
    canon_segs = canonical.split(":")
    if len(frag_segs) > len(canon_segs):
        return False
    return canon_segs[-len(frag_segs):] == frag_segs


def resolve_host(
    fragment: str, canonical_names
) -> Tuple[str, Optional[str], List[str]]:
    """Resolve an ``Anchor(host=...)`` fragment against known canonical names.

    ``canonical_names`` is any iterable of the project's canonical names
    (anonymous elements — ``None`` — are ignored by the caller). Returns a
    ``(status, matched, candidates)`` triple:

    * ``("ok", canonical, [canonical])`` — exactly one segment-aligned suffix
      match.
    * ``("miss", None, [])`` — zero matches; the author's fragment names no
      element.
    * ``("ambiguous", None, sorted_candidates)`` — 2+ matches; the author must
      write the shortest unambiguous fragment (more leaf-first pairs).
    """
    candidates = sorted(
        {c for c in canonical_names if c and host_matches(fragment, c)}
    )
    # Exact-canonical match wins the tiebreak: a fragment that IS a full
    # canonical name resolves to THAT element even when longer names carry
    # it as a trailing pair-suffix. Without this, naming any child makes its
    # container unreferenceable — host="wall:north" would be ambiguous with
    # the (common) named child "window:left:wall:north". An author who writes
    # the element's own full name means that element.
    if fragment in candidates:
        return "ok", fragment, [fragment]
    if len(candidates) == 1:
        return "ok", candidates[0], candidates
    if not candidates:
        return "miss", None, []
    return "ambiguous", None, candidates
