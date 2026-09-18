"""The ONE on-path re-entrancy guard every containment walk reads.

The DSL tree is a tree by convention and by nothing else: ``.add()`` and
``.anchor()`` append to ``_elements``, so ``w1.anchor(w2); w2.anchor(w1)``
builds a cycle out of two ordinary authoring lines. Eight recursive walks
descend that structure — canonical naming, frame stamping, the anchor bake,
the anchor-liveness probe, the anchor subtree map, geometry-point collection,
CSG-depth prediction, the grouping index and the extent union — and every one
of them recursed until the stack ran out. What the author saw was a bare
``RecursionError`` from inside the compiler, with no element named.

**On-path, not visited.** A global visited set would be wrong, not merely
conservative: an element legitimately appears once per PATH — the same
canonical-naming walk deliberately reaches a boolean operand from its host and
would then skip a second, legitimate host — and `find_shared_elements` exists
precisely because reaching one object twice through two different paths is a
separate, already-reported condition. Only re-entry while the element is still
on the CURRENT path is a cycle. So the guard pushes on the way down and pops on
the way back up.

**One helper, eight call sites, one mechanism — but the POLICY stays local.**
Each walk has a different right answer on re-entry and they are not
interchangeable: naming RECORDS the cycle (it is the pass the validator reads),
`extent.element_extent` RAISES (a truncated extent is a plausible wrong number,
which is the one thing `Bounds` refuses to hand out), and the analysis walks
PRUNE (they must terminate; the validator has already refused the model). Eight
copies of the guard itself is how they would start to disagree about what a
cycle *is*; one copy with the handling at the call site is honest about where
they genuinely differ.

The guard also carries the current path, because the only walk that needs to
NAME a cycle would otherwise have to thread a parallel list through its hot
recursion — one list allocation per node, on the pass whose own docstring
budgets in single milliseconds. A push and a pop on a list already being
maintained costs nothing measurable and makes :meth:`WalkGuard.cycle_path`
free.
"""

from __future__ import annotations

from typing import List, Tuple


class ContainmentCycleError(ValueError):
    """A containment walk re-entered an element already on its own path.

    Raised by the walks that cannot answer at all on a cyclic tree — an
    extent query has no finite answer when an element contains itself, and a
    truncated one is a plausible wrong number. The walks that only need to
    TERMINATE prune instead; the authoritative report is
    ``compiler.naming.find_containment_cycles``, surfaced as a compile error
    by ``validate_project_report``.
    """


class WalkGuard:
    """Tracks which elements are on the CURRENT branch of a recursive walk.

    Usage is the same three lines everywhere, and the ``try``/``finally`` is
    load-bearing — without it a walk that raises mid-descent leaves its
    ancestors on the path, and a guard reused for a second root would then
    report a cycle that is not there::

        def _walk(elem, ..., _on_path=None):
            guard = _on_path if _on_path is not None else WalkGuard()
            if not guard.enter(elem):
                ...                      # the walk's own policy
                return
            try:
                ...                      # recurse, passing _on_path=guard
            finally:
                guard.leave(elem)

    Identity is ``id()``, the same identity ``find_shared_elements`` uses:
    two structurally identical elements built by two constructor calls are two
    objects and neither is a cycle; the same object reached from inside itself
    is.
    """

    __slots__ = ("_ids", "_path")

    def __init__(self) -> None:
        self._ids: set = set()
        self._path: List = []

    def enter(self, elem) -> bool:
        """Push ``elem`` onto the path; ``False`` when it is already on it.

        A ``False`` return means the caller must NOT recurse and must NOT
        call :meth:`leave` — nothing was pushed.
        """
        key = id(elem)
        if key in self._ids:
            return False
        self._ids.add(key)
        self._path.append(elem)
        return True

    def leave(self, elem) -> None:
        """Pop ``elem`` back off the path."""
        self._ids.discard(id(elem))
        if self._path:
            self._path.pop()

    def cycle_path(self, elem) -> Tuple:
        """The elements forming the cycle ``elem`` just closed, in order.

        From ``elem``'s first appearance on the current path through to the
        re-entry, with ``elem`` repeated at the end so the loop reads as a
        loop (``wall:a > wall:b > wall:a``) rather than as a list that
        happens to start and end nearby.

        Only meaningful immediately after :meth:`enter` returned ``False``.
        """
        key = id(elem)
        for i, node in enumerate(self._path):
            if id(node) == key:
                return (*self._path[i:], elem)
        return (elem,)                              # pragma: no cover - unreachable


class MeasurementCycleError(ValueError):
    """An element is positioned from a measurement of itself, transitively.

    Beside :class:`ContainmentCycleError` because it is the same family of
    refusal — an element defined in terms of itself — but a different
    mechanism. Containment cycles are structural and the walk cannot
    terminate; a MEASUREMENT cycle terminates fine and emits a building.
    That is what makes it worse: three meshes each placed from the next
    compiled exit 0 and produced an IFC 1000 mm out of position, with no
    warning (defect report).

    **This refusal is about intent, not solvability.** The compiler could
    pick an answer — today it silently does, whichever the last write left
    behind. It refuses because a cycle does not describe one design. When a
    window's frame is placed from its sash and the sash from its frame, the
    model expresses several buildings and statement order decides which one
    you get. Choosing between them is not the compiler's to do.

    Raised at the moment the closing edge is authored, so the traceback
    points at the line that completed the loop rather than at a compile pass
    a thousand statements later.
    """
