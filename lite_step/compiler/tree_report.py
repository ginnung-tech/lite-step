"""The container tree the compiler actually built, printed on every compile.

``carves: N pairs`` and ``csg depth: max=N`` report what the compiler DID.
Neither reports the SHAPE it did it to. An author who mis-nested a leaf, or
whose ``.add()`` landed on the wrong container, learned it from geometry that
looked wrong — or did not learn it at all.

Three settings, and the DEFAULT is the reduced tree:

==============================  =====================================
``LITESTEP_TREE_REPORT``        output
==============================  =====================================
unset / anything but the below  the tree, sibling runs collapsed,
                                capped — with an explicit ``+N more``
                                line naming everything it dropped
``all``                         the same tree, uncapped
``0``                           silent (byte-comparison runs)
==============================  =====================================

Same three settings, same names and the same stderr channel as
``LITESTEP_CARVE_REPORT`` (``ifc.generator._report_carve_pairs``), because a
second convention for one idea is a thing to remember rather than a thing to
know. ``0`` in particular is not decoration: a batch compile reads stderr.

**The bound is the design, not a trimming afterwards.** A container tree is
strictly LARGER than the carve list, and the carve list already had to be
capped: measured on the corpus, four villa-chain models emitted 96-265 lines
of carve detail — about 12 KB of stderr per compile, on the models the author builds
most, straight into an agent's bounded context. So the default here is capped
twice over, and both caps announce themselves:

* **runs of identical siblings collapse** to one row plus ``xN`` — a
  ``column:stud_00 .. stud_09  x10``. Identity is the whole SUBTREE (type,
  verb, extents, openings, children — everything this report prints except the
  name), so two rows never merge into a claim that is false about either of
  them. This applies to ``all`` as well: it is a lossless rewrite, not a cap.
* **the default caps depth and total rows**, and states the count it dropped.
  A report that hides its truncation is the failure this repo keeps fixing, so
  the ``+N more`` line is emitted from the same walk that did the dropping and
  counts nodes, not rows.

**Units.** Every extent here is integer millimetres, and this module runs on
the POST-``normalize_project_to_meters`` tree, whose coordinates are METRES.
So both derivations are told so — :func:`~lite_step.compiler.extent.element_extent`
divides the frozen ``Material`` millimetre facts down into metres, and
:meth:`~lite_step.models.bounds.Bounds.from_aabb` multiplies the metre
coordinates back up into millimetres. Omitting the second one RAISES
(``BoundsError``: a 306 mm leaf arrives as ``size_mm=0.306``, real but rounding
to zero) — but that guard is PARTIAL by construction and is not sold as more:
a container whose extents are all whole metres rounds cleanly and nothing can
see the error. ``lite_step/tests/test_tree_report.py``'s fixture therefore
carries deliberately non-whole-metre extents.

**What a row says.** Indentation is containment. The character immediately
before the label is the VERB — ``@`` for ``.anchor()``, blank for ``.add()`` —
because that distinction decides where a child ends up and is invisible in the
emitted file (both become ``IfcRelAggregates``). Then the label, which is the
element's own ``type:leaf`` canonical pair, so a row can be grepped against
``LITESTEP_META`` and against ``Anchor(host=)``. Then, in order: extents,
``IfcClass`` (only for the generic ``Element(ifc_class=…)`` escape hatch,
where the DSL type does not say), ``layers=``, ``openings=``, ``voids=``, and
``(no body)``.

``(no body)`` is exactly ``taxonomy.body_prisms(elem) == []`` on a container —
THE definition of "the body", shared with the validator and the emitter. It
means the container emits no unanchored ``Box``/``Extrude`` of its own and its
extents come entirely from its children: the body-less aggregator, and also
a ``Wall`` built from a ``Sweep``. It does NOT try to predict whether the
emitted product carries a ``Representation``; that is a different question,
answered by ``ifc.product_types.occurrence_shape_slots`` after emission, and
guessing at it here would put a second opinion beside a decided one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: Rows the DEFAULT setting will print before it starts counting instead.
#: MEASURED, not picked. At this number 13 of the 20 corpus models print their
#: WHOLE tree and cost 0.2-1.7 KB of stderr; the other 7 are capped at 26 rows
#: / ~1.3 KB. The comparison that decides it is the one ``_report_carve_pairs``
#: had to make: its uncapped default cost the villa-chain models 96-265 lines
#: and ~12 KB per compile, straight into an agent's bounded context. A tree is
#: strictly larger than that carve list, so uncapped is not an option here
#: either — ``=all`` reaches 185 rows / 9.4 KB on ``bridge-roadway``.
_MAX_ROWS = 24

#: Depth the DEFAULT setting descends to, counting a site/storey as 1. Both
#: caps exist because each misses what the other catches: a depth cap alone
#: lets one 300-child storey through, and a row cap alone spends the entire
#: budget inside the first wall and never reaches the second storey — and
#: "my .add() landed on the wrong container" is a question about the top of
#: the tree, so the top is what the default must always reach.
_MAX_DEPTH = 4


@dataclass
class _Node:
    """One element, with everything the renderer needs already resolved.

    Built in a first pass so the collapse can compare whole SUBTREES: a
    signature computed from a half-rendered row would merge two studs that
    differ only in what hangs off them.
    """

    label: str
    anchored: bool
    extras: Tuple[str, ...]
    leaf_name: Optional[str]
    children: List["_Node"] = field(default_factory=list)
    #: Set by :func:`_stamp_signatures`. Equal signatures ⇒ interchangeable
    #: rows, so a run of them may be reported as one row plus a count.
    sig: Tuple = ()
    #: Label of a lone body child folded into this row — see :func:`_fold`.
    folded: Optional[str] = None

    def count(self) -> int:
        """Nodes in this subtree, including this one — what ``+N more`` counts."""
        return 1 + (1 if self.folded else 0) + sum(c.count() for c in self.children)


def _type_pair(elem) -> Tuple[str, Optional[str]]:
    """``("wall", "south")`` — the element's own canonical pair, split.

    Reads :func:`lite_step.compiler.naming.type_segment`, the same derivation
    that builds ``_canonical_name``, so a row's label is a prefix of the name
    the file carries rather than a second spelling free to drift.
    """
    from lite_step.compiler.naming import type_segment

    return type_segment(elem), (getattr(elem, "name", None) or None)


def _size_mm(elem) -> Optional[Tuple[int, int, int, bool]]:
    """``(x, y, z, exact)`` extents in int mm, or ``None`` when it has none.

    Both divisors are stated here — see the module docstring's Units section.
    ``exact=False`` means this is a BOUND (a padded curve, a rotated box), and
    the renderer says so rather than presenting it as a measurement.
    """
    from lite_step.compiler.extent import MM_PER_METER, element_extent, extent_is_exact
    from lite_step.models.bounds import Bounds

    box = element_extent(elem, MM_PER_METER)
    if box is None:
        return None
    bounds = Bounds.from_aabb(box, exact=extent_is_exact(elem),
                              rule="tree-report", divisor=MM_PER_METER)
    size = bounds.size
    return int(size.x), int(size.y), int(size.z), bounds.exact


def _extras_for(elem) -> Tuple[str, ...]:
    """The trailing tokens of one row, in a fixed order.

    Fixed because the row is read by an LLM as much as by a person: a stable
    column order is what makes two compiles diffable.
    """
    from lite_step.models import taxonomy as tx
    from lite_step.models.elements import Element

    out: List[str] = []

    if tx.is_opening(elem):
        # An opening carries a SIZE and no position (v20.0.0); the HOST states
        # where it goes. Reporting an extent for one would be reporting the
        # host's frame as if it were the opening's own geometry — and
        # ``_authored_box`` refuses the question for that reason. The anchor
        # scalars are what the author wrote, and ``_bake_walk`` deliberately
        # never consumes them on a Window/Door, so they are still here.
        #
        # ``_mm`` on every one of them: ``width``/``height`` and the
        # ``ChildAnchor`` scalars are ordinary mm dimensions, so the meters
        # normalizer scaled them with everything else. Reading them raw
        # printed a 1400x2000 window as "1x2 mm opening" at "up=0.9".
        out.append(f"{_mm(getattr(elem, 'width', 0))}x"
                   f"{_mm(getattr(elem, 'height', 0))} mm opening")
        spec = getattr(elem, "_anchor_spec", None)
        if spec is not None:
            along = (f"along_center={_mm(spec.along_center)}"
                     if spec.along_center is not None
                     else f"along={_mm(spec.along)}")
            out.append(f"{along} up={_mm(spec.up)} mm")
    else:
        size = _size_mm(elem)
        if size is None:
            out.append("(no extent)")
        else:
            x, y, z, exact = size
            out.append(f"{'~' if not exact else ''}{x}x{y}x{z} mm")

    if isinstance(elem, Element):
        # The one case where the DSL type does not name the emitted class.
        # Worth a column of its own: the escape hatch being reached for by
        # default is how a whole production model came out as
        # IfcBuildingElementProxy.
        out.append(str(elem.ifc_class))

    layers = getattr(elem, "layers", None)
    if layers is not None:
        out.append(f"layers={len(getattr(layers, 'layers', None) or [])}")

    openings = _opening_count(elem)
    if openings:
        out.append(f"openings={openings}")

    voids = len(getattr(elem, "_voids", None) or [])
    if voids:
        out.append(f"voids={voids}")

    if tx.is_physical(elem) and tx.is_container(elem) and not tx.body_prisms(elem):
        out.append("(no body)")

    return tuple(out)


def _mm(value) -> int:
    """A scalar off the METRE tree, as the integer millimetres a row prints.

    THE one place this report states the domain for a plain scalar — the same
    statement :func:`_size_mm` makes for a box, and made once so the two cannot
    disagree. Everything here runs after ``normalize_project_to_meters``, which
    scales every mm dimension in the tree: a storey ``elevation``, a
    ``Window.width``, a ``ChildAnchor.along``. Reading one of them raw prints
    the number 1400 as ``1``.
    """
    from lite_step.compiler.extent import MM_PER_METER
    from lite_step.models.bounds import round_half_up

    try:
        return round_half_up(float(value) * MM_PER_METER)
    except (TypeError, ValueError):                  # pragma: no cover - defensive
        return 0


def _opening_count(elem) -> int:
    """Openings hosted BY this element, from both places one can live.

    A leaf host parks them in ``_openings`` (``body.opening(win)``); a
    container's are ordinary ``_elements`` children (``wall.anchor(win)``, which
    ``.opening()`` routes to). Counting only one list reports zero openings for
    half the models in the corpus.
    """
    from lite_step.models import taxonomy as tx

    n = len(getattr(elem, "_openings", None) or [])
    n += sum(1 for c in (getattr(elem, "_elements", None) or [])
             if tx.is_opening(c))
    return n


def _build(elem, on_path: set) -> _Node:
    """One element and its subtree, resolved.

    ``on_path`` is an ON-PATH guard, not a visited set — the same distinction
    ``naming._walk`` documents. A containment cycle is a compile ERROR caught
    upstream (``naming.find_containment_cycles``), so reaching one here means a
    caller skipped validation; recursing forever on it would turn a report into
    the reason a compile hangs.
    """
    seg, leaf = _type_pair(elem)
    node = _Node(
        label=f"{seg}:{leaf}" if leaf else seg,
        anchored=_is_anchored(elem),
        extras=_extras_for(elem),
        leaf_name=leaf,
    )
    if id(elem) in on_path:                          # pragma: no cover - defensive
        node.extras = node.extras + ("(cycle)",)
        return node
    on_path.add(id(elem))
    try:
        for attr in ("_elements", "_openings"):
            for child in getattr(elem, attr, None) or []:
                node.children.append(_build(child, on_path))
    finally:
        on_path.discard(id(elem))
    return node


def _fold(node: _Node) -> None:
    """Fold a container's lone body child into the container's own row.

        wall:south   6000x306x3000 mm          <- was two rows, and the second
          box:body   6000x306x3000 mm             stated nothing the first did not

    becomes ``wall:south  6000x306x3000 mm  +box:body``. Applied depth-first
    to every node, in ``all`` as well as the default, because a row whose every
    token repeats its parent's is not detail — it is the same fact twice, and
    this shape is roughly half the rows in the corpus (a ``Wall``/``Slab``/
    ``Roof`` whose whole geometry is one authored ``Box``).

    The conditions are deliberately narrow, and each removes a way the fold
    could hide something:

    * **exactly one child**, so no sibling disappears into the parent;
    * **the child is a leaf**, so no subtree goes with it;
    * **the child is** ``.add()``\\ **ed**, because an ``.anchor()``ed child is
      the one distinction this report exists to make visible;
    * **the child's extras are exactly the parent's size token** — same
      extents, and no openings/voids/layers/class of its own to lose.

    The folded child keeps its NAME in the ``+`` suffix (``box:body`` is a
    manifest key and an ``Anchor(host=)`` target, so dropping it would make a
    row ungreppable) and keeps its place in the header's node count.
    """
    for child in node.children:
        _fold(child)
    if len(node.children) != 1:
        return
    child = node.children[0]
    if child.children or child.anchored or child.folded:
        return
    if not node.extras or child.extras != (node.extras[0],):
        return
    node.folded = child.label
    node.children = []


def _is_anchored(elem) -> bool:
    """``.anchor()``ed rather than ``.add()``ed.

    ``frames.is_anchored_child`` reads BOTH ``_anchor_spec`` (pending) and
    ``_anchor_resolved`` (baked) for a reason this report would otherwise walk
    straight into: the bake consumes the spec, so a report reading only the
    spec would say ``.add()`` about every anchored child on the compile path
    and ``.anchor()`` about the same child in a unit test.
    """
    from lite_step.compiler import frames

    return frames.is_anchored_child(elem)


def _stamp_signatures(node: _Node) -> Tuple:
    """Bottom-up identity: everything a row states except the name.

    The name is excluded on purpose — a run of ``stud_00 .. stud_09`` is
    exactly a set of siblings that differ ONLY in name — and everything else is
    included, so a collapsed row is true of every sibling it stands for.
    """
    sig = (node.label.split(":", 1)[0], node.anchored, node.extras,
           (node.folded or "").split(":", 1)[0],
           tuple(_stamp_signatures(c) for c in node.children))
    node.sig = sig
    return sig


def _runs(children: List[_Node]) -> List[Tuple[_Node, List[_Node]]]:
    """Consecutive siblings grouped by signature.

    CONSECUTIVE, not sorted-and-grouped: authoring order is information (it is
    what decides displacement direction), and reordering a report to make its
    runs longer would hide an interleaving the author did not intend.
    """
    out: List[Tuple[_Node, List[_Node]]] = []
    for child in children:
        if out and out[-1][0].sig == child.sig:
            out[-1][1].append(child)
        else:
            out.append((child, [child]))
    return out


def _row(node: _Node, run: List[_Node], depth: int) -> str:
    """``  @column:stud_00 .. stud_09  x10  45x145x2700 mm``.

    Two spaces per level of containment, then the verb marker in the column
    immediately left of the label, then the label, then the extras. The marker
    occupies a column that exists on every row, so ``.add()`` and ``.anchor()``
    are distinguishable without counting spaces.
    """
    prefix = "  " * max(depth - 1, 0) + ("@" if node.anchored else " ")
    label = node.label
    if len(run) > 1:
        last = run[-1].leaf_name
        if node.leaf_name and last:
            label = f"{node.label} .. {last}"
        label = f"{label}  x{len(run)}"
    tail = list(node.extras)
    if node.folded:
        tail.append(f"+{node.folded}")
    return f"{prefix}{label}  {'  '.join(tail)}".rstrip()


def _plan(roots: List[_Node], max_depth: Optional[int],
          max_rows: Optional[int]) -> Tuple[set, int]:
    """Choose which rows the budget buys; return ``(included ids, nodes dropped)``.

    **Breadth-first, and that is the whole point of separating this from the
    rendering.** Spending the budget depth-first is what the first draft did,
    and ``bridge-roadway`` showed what it costs: all 24 rows went into one
    ``spatialelement:arch``'s diagonals and the report never mentioned that the
    model has storeys at all. The question this report exists to answer — "did
    my ``.add()`` land on the wrong container" — is a question about the TOP of
    the tree, so the top is what a bounded default must always reach. Level 1
    is therefore complete before level 2 starts, and the rows that get dropped
    are the tail of the deepest level the budget reached.

    Rendering stays depth-first (a tree read out breadth-first is unreadable);
    this pass only decides membership, keyed on ``id`` of each run's
    representative. Both passes group with the same :func:`_runs`, which is a
    pure function of the sibling list, so they cannot disagree about what a run
    is.

    The dropped count counts NODES, not rows, and is produced by the pass that
    does the dropping — a report that under-states its own truncation is worse
    than one that prints nothing. It counts only what the CAPS cut. A collapsed
    run's other members are not dropped: ``x10`` states that they exist and the
    signature guarantees the printed row is true of each of them, so counting
    them as missing would make the notice cry wolf on every model with a stud
    wall in it.
    """
    from collections import deque

    included: set = set()
    dropped = 0
    queue = deque((1, head, run) for head, run in _runs(roots))
    while queue:
        depth, head, run = queue.popleft()
        if ((max_depth is not None and depth > max_depth)
                or (max_rows is not None and len(included) >= max_rows)):
            dropped += sum(m.count() for m in run)
            continue
        included.add(id(head))
        for child_head, child_run in _runs(head.children):
            queue.append((depth + 1, child_head, child_run))
    return included, dropped


def _emit(nodes: List[_Node], depth: int, out: List[str], included: set) -> None:
    """Depth-first rows for everything :func:`_plan` kept."""
    for head, run in _runs(nodes):
        if id(head) not in included:
            continue
        out.append(_row(head, run, depth))
        _emit(head.children, depth + 1, out, included)


def tree_report_lines(proj, *, full: bool = False) -> List[str]:
    """The report, as stderr lines. ``full=True`` is ``LITESTEP_TREE_REPORT=all``.

    Collapse happens either way; ``full`` only removes the depth and row caps.
    """
    from lite_step.compiler.extent import memoized_extents

    # One memo for the whole report. Every row asks its node for an extent,
    # and a container answers by recursing into each child's own — so without
    # this, a subtree is recomputed once for every ancestor above it and the
    # walk costs O(nodes x depth). The report is the safe place for it and
    # ``memoized_extents`` states why: it runs after every mutating pass, it
    # only reads, and the cache dies with this call.
    with memoized_extents():
        return _tree_report_lines(proj, full=full)


def _tree_report_lines(proj, *, full: bool = False) -> List[str]:
    roots: List[_Node] = []
    on_path: set = set()
    # Sites BEFORE storeys, which is the order ``Project.add`` routes into and
    # the order the file nests (IfcSite then IfcBuilding).
    for site in getattr(proj, "sites", None) or []:
        roots.append(_build(site, on_path))
    for storey in getattr(proj, "storeys", None) or []:
        node = _Node(
            label=(f"storey:{storey.name}" if storey.name else "storey"),
            anchored=False,
            extras=(f"elev={_mm(storey.elevation or 0)} mm",),
            leaf_name=storey.name,
        )
        for elem in storey.elements or []:
            node.children.append(_build(elem, on_path))
        roots.append(node)

    for node in roots:
        _fold(node)
        _stamp_signatures(node)

    total = sum(n.count() for n in roots)
    included, dropped = _plan(roots,
                              None if full else _MAX_DEPTH,
                              None if full else _MAX_ROWS)
    rows: List[str] = []
    _emit(roots, 1, rows, included)

    head = (f"tree: {getattr(proj, 'name', None) or '(unnamed)'}  "
            f"{total} nodes  (@ = .anchor()ed, extents in mm)")
    if dropped:
        rows.append(f"  +{dropped} more nodes not shown  "
                    f"(LITESTEP_TREE_REPORT=all for the whole tree)")
    return [head] + rows
