"""Balanced CSG trees — one shared shape for every boolean emission site.

**Why.** LEFT-DEEP boolean chains are the naive shape: subtracting V voids
from a base solid produces V nested ``IfcBooleanResult`` entities, so
nesting depth == operand count. Downstream tessellators are not indifferent to
that depth — the primary web viewer stops evaluating a deeply nested chain and
silently falls back to rendering the *un-subtracted* base solid (measured: a
foundation wall with 13 sequential DIFFERENCE nestings rendered as a solid block
that swallowed the floor beams; a roof covering at 30+ booleans was dropped
outright). No error is raised anywhere, so the failure reads as "the geometry is
wrong" rather than "the chain is too deep".

**The fix** is a provable set identity, not an approximation:

    (((A − c1) − c2) − c3)  ≡  A − (c1 ∪ c2 ∪ c3)

and ∪ / ∩ are associative, so the operand group can be combined as a BALANCED
tree. Depth drops from ``V`` to ``ceil(log2(V)) + 1``: the 13-void wall goes
13 → 5.

**What must NOT change.** The class order cuts → adds → intersects is
semantically load-bearing (``(A − c) ∪ a`` ≠ ``(A ∪ a) − c``) and is preserved
exactly. Only the shape WITHIN each class changes. ``.clip()`` half-spaces stay
a chained ``IfcBooleanClippingResult`` sequence applied by the caller — unioning
unbounded half-spaces into the void group is not proven safe in the viewer, so
it is deliberately out of scope.

**Emitter-neutral.** Callers pass a ``create_entity(operator, first, second)``
callable, so the tree shape lives here and nothing else re-derives it.

**V == 1 is a no-op.** With a single operand in a class, the emitted entities —
and the order in which they are created — are byte-identical to the pre-balancing
compiler. That is pinned by a test; balancing must never churn a simple model.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

#: Maximum boolean nesting depth the primary web viewer evaluates reliably.
#: Measured tolerance is ~13 before silent fallback; the compiler budget is set
#: below that with headroom, and exceeding it is a loud warning (never a compile
#: error — a deep model must still compile and download).
CSG_DEPTH_BUDGET = 10

#: Boolean operation COUNT, per element representation, above which the
#: compiler warns.
#:
#: A different hazard from depth, with the opposite signature. Depth breaks
#: HARD and INVISIBLY: past ~13 levels the web viewer stops evaluating and
#: renders the uncarved base solid, fast and wrong. Width breaks SOFT: the
#: geometry stays correct and the evaluation time grows, measured at roughly
#: N^1.33 for a mitered sweep, with no cliff anywhere.
#:
#: Because there is no cliff there is no defensible pass/fail line, so this is
#: a REPORTING threshold, not a limit — it decides when the number is worth
#: saying out loud, and the number itself is what the author acts on. 99 is
#: chosen so an ordinary building member never trips it (a mitered window
#: frame is 11, a filleted handrail well under 50) while the counts that make
#: a file practically unopenable always do.
#:
#: Measured (square profile, ifcopenshell 0.8.5 + OCC):
#:     4 segs /  11 booleans → 0.25 s
#:    32 segs /  95 booleans → 4.18 s
#:    96 segs / 287 booleans → 17.19 s
#:   220 segs / 659 booleans → no shape produced in 420 s
#: The last one is a real model (a stellarator helical coil). It compiled
#: clean, validated, hashed, rendered correctly in our own viewer, and could
#: not be imported by Blender Bonsai at all.
CSG_WIDTH_BUDGET = 99


def balanced_tree(create_entity: Callable, operator: str, operands: Sequence):
    """Combine ``operands`` with ``operator`` as a balanced binary tree.

    Pairs adjacent operands level by level (an odd tail element carries to the
    next level unchanged), so depth is ``ceil(log2(len(operands)))``. Returns
    the lone operand UNCHANGED when there is exactly one — no entity is created.
    Empty input is a programming error (callers guard on emptiness because the
    "no operands" case means "emit no boolean at all").
    """
    level: List = list(operands)
    if not level:
        raise ValueError("balanced_tree() needs at least one operand")
    while len(level) > 1:
        nxt: List = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(create_entity(operator, level[i], level[i + 1]))
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def balanced_union(create_entity: Callable, operands: Sequence):
    """Balanced pairwise UNION over ``operands`` (see :func:`balanced_tree`)."""
    return balanced_tree(create_entity, "UNION", operands)


def _built(build: Optional[Callable], operands: Optional[Iterable], is_cut: bool = True) -> List:
    """Realise each model operand into a solid entity, dropping the ones the
    backend could not build (``None`` — an unsupported operand shape, already
    warned about at the build site)."""
    if not operands:
        return []
    if build is None:
        return [o for o in operands if o is not None]
    res = []
    for o in operands:
        try:
            s = build(o, is_cut=is_cut)
        except TypeError:
            s = build(o)
        if s is not None:
            res.append(s)
    return res


def apply_boolean_chain(create_entity: Callable, base, *,
                        cuts: Optional[Iterable] = None,
                        adds: Optional[Iterable] = None,
                        intersects: Optional[Iterable] = None,
                        build: Optional[Callable] = None):
    """Apply one element's boolean classes to ``base``, balanced within each class.

    ``build`` turns a model operand into a backend solid (or ``None`` when the
    operand shape is unsupported); omit it when the caller already holds entities.
    Operand solids are realised class by class, immediately before that class's
    boolean is emitted — that ordering is what keeps a single-operand model
    byte-identical to the pre-balancing output.

    Class order (semantically load-bearing, unchanged):

    1. ``cuts``      → ONE ``DIFFERENCE`` against the balanced UNION of the cuts
    2. ``adds``      → balanced ``UNION`` over ``[result] + adds``
    3. ``intersects``→ balanced ``INTERSECTION`` over ``[result] + intersects``

    ``.clip()`` half-spaces are NOT handled here — the caller applies them after,
    as today.
    """
    result = base

    cut_solids = _built(build, cuts, is_cut=True)
    if cut_solids:
        result = create_entity("DIFFERENCE", result,
                               balanced_union(create_entity, cut_solids))

    add_solids = _built(build, adds, is_cut=False)
    if add_solids:
        result = balanced_tree(create_entity, "UNION", [result] + add_solids)

    int_solids = _built(build, intersects, is_cut=False)
    if int_solids:
        result = balanced_tree(create_entity, "INTERSECTION",
                               [result] + int_solids)

    return result


# ---------------------------------------------------------------------------
# Depth arithmetic — shared by the emitters above and the model-level predictor
# (``lite_step.compiler.csg_depth``) so the prediction cannot drift from what is
# actually emitted.
# ---------------------------------------------------------------------------


def balanced_depth(depths: Sequence[int]) -> int:
    """Nesting depth of :func:`balanced_tree` over operands with these depths.

    Simulates the exact same pairing, so it is not an estimate.
    """
    level = list(depths)
    if not level:
        return 0
    while len(level) > 1:
        nxt: List[int] = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(max(level[i], level[i + 1]) + 1)
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


def chain_depth(base_depth: int,
                cut_depths: Sequence[int] = (),
                add_depths: Sequence[int] = (),
                intersect_depths: Sequence[int] = ()) -> int:
    """Depth of :func:`apply_boolean_chain`'s result, given operand depths."""
    d = base_depth
    if cut_depths:
        d = max(d, balanced_depth(list(cut_depths))) + 1
    if add_depths:
        d = balanced_depth([d] + list(add_depths))
    if intersect_depths:
        d = balanced_depth([d] + list(intersect_depths))
    return d
