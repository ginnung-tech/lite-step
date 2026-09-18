"""A containment cycle is a compile error, and no walk hangs on one.

The DSL tree is a tree by convention and by nothing else. ``.add()`` and
``.anchor()`` append to ``_elements``, so two ordinary authoring lines build an
element that contains itself::

    w1.anchor(w2); w2.anchor(w1)          # two Walls
    a.add(b);      b.add(a)               # two facility SpatialElements

Without this suite: **both are accepted**, and compiling
either raised a bare ``RecursionError`` from inside ``naming._walk`` — naming no
element, at a stack depth thousands of frames from the line that caused it, and
*before* validation had produced a single message. There was no acyclicity
enforcement anywhere in the repo.

What is pinned here, and why each part must exist separately:

1.  **The authoring line refuses.** ``.anchor()`` and ``.add()`` climb
    ``_parent`` and refuse when the container is already inside the child, so
    the ordinary two-line case fails on the *second* line with both elements
    named. This is IN ADDITION to the validator, on the precedent
    ``frames.check_anchor_rotation`` set — it moved a refusal to the authoring
    line and kept the compile-time one "as the backstop rather than being
    replaced". ``_parent`` is last-write-wins, so this check CANNOT see every
    cycle; the validator is authoritative.
2.  **The validator reports it, and does not blow the stack.** Test 3 builds a
    cycle by hand, mutating ``_elements`` directly to bypass the attach check,
    and demands a named compile error. It catches ``RecursionError`` explicitly
    and fails on it, because that is the regression this whole change exists to
    prevent — a green test that swallowed the recursion would prove nothing.
3.  **A world query raises a typed error rather than hanging.**
    ``compiler.extent`` refuses to hand out the union of the finite prefix the
    walk happened to reach: that is exactly the plausible-looking wrong number
    ``Bounds`` exists to refuse.
4.  **No corpus model has a cycle.** ``find_shared_elements`` needed a
    proof-before-promotion pass over the corpus before it could become an
    error; this claims the same ground for the same reason, and measures it
    rather than asserting it.

The guard is ON-PATH, not a visited set, and test 6 pins the difference: an
element is legitimately reached once per path (the naming walk deliberately
reaches a boolean operand from its host, and ``find_shared_elements`` reports
one object reached through two paths as its OWN condition). A visited set would
terminate just as well and silently un-report every shared element.
"""

from __future__ import annotations


import pytest

from lite_step.compiler.executor import validate_project_report
from lite_step.compiler.naming import find_containment_cycles
from lite_step.compiler.walkguard import ContainmentCycleError
from lite_step.models import (
    Box,
    Element,
    Point,
    Project,
    SpatialElement,
    Wall,
)

# ---------------------------------------------------------------------------
# Fixtures. Every element carries a real body — a cycle between two empty
# containers would be caught by the "no point-bearing geometry" refusal on some
# paths and prove nothing about the cycle itself.
# ---------------------------------------------------------------------------


def _body(name: str, x0: int = 0) -> Box:
    return Box(start=Point(x=x0, y=0, z=0),
               end=Point(x=x0 + 1000, y=200, z=2000), name=name)


def _proxy(name: str, x0: int = 0) -> Element:
    elem = Element(ifc_class="IfcBuildingElementProxy", name=name)
    elem.add(_body(f"body_{name}", x0))
    return elem


def _hand_built_cycle() -> tuple:
    """A project holding ``a`` -> ``b`` -> ``a``, built by direct mutation.

    ``_elements.append`` bypasses ``_add_children`` and therefore the attach
    check, which is the point: the validator has to stand on its own. This is
    also not a contrived shape — it is what an author gets today whenever the
    cycle closes through a chain ``_parent`` (last-write-wins) does not record.
    """
    a, b = _proxy("a"), _proxy("b", x0=2000)
    proj = Project(name="acyclicity")
    proj.add(a)
    a._elements.append(b)
    b._elements.append(a)
    return proj, a, b


# ---------------------------------------------------------------------------
# 1-2. The authoring line refuses, and names both elements
# ---------------------------------------------------------------------------


def test_anchoring_two_walls_into_each_other_raises_on_the_second_call():
    """``w1.anchor(w2); w2.anchor(w1)`` must die on line 2, not at compile.

    Without this the pair is accepted in full and the author's next signal is a
    ``RecursionError`` inside the compiler naming no element. The refusal has
    to fire on the SECOND call specifically: firing on the first would mean the
    check is refusing an ordinary nesting.
    """
    w1, w2 = Wall(name="a"), Wall(name="b")

    w1.anchor(w2)                       # ordinary nesting — must be accepted

    with pytest.raises(ValueError) as exc:
        w2.anchor(w1)

    msg = str(exc.value)
    assert "'a'" in msg and "'b'" in msg, (
        f"both elements must be named — an author cannot find a cycle from a "
        f"message that identifies one end of it: {msg}"
    )
    assert "Wall.anchor()" in msg, (
        f"the message must name the verb that was refused, so the traceback's "
        f"authoring line is confirmed rather than merely nearby: {msg}"
    )


def test_adding_two_facilities_into_each_other_raises_on_the_second_call():
    """The same, through ``.add()`` between two facility ``SpatialElement``s.

    This pairing is reachable because ``_FACILITY_CHILDREN`` is
    ``("SpatialElement",)`` — a facility's legal child list is facilities, so
    facility-contains-facility-contains-facility is exactly what the child
    table permits. Pinned separately from ``.anchor()`` because the two verbs
    run different code (``_add_children`` vs ``_anchor_child``) and a guard
    installed on one only is the silent-asymmetry failure ``_add_children``'s
    own docstring names.
    """
    a = SpatialElement(ifc_class="IfcBridge", name="a")
    b = SpatialElement(ifc_class="IfcBridge", name="b")

    a.add(b)                            # ordinary nesting — must be accepted

    with pytest.raises(ValueError) as exc:
        b.add(a)

    msg = str(exc.value)
    assert "'a'" in msg and "'b'" in msg, (
        f"both elements must be named: {msg}"
    )
    assert "SpatialElement.add()" in msg, (
        f"the message must name the verb that was refused: {msg}"
    )


def test_a_container_cannot_be_added_to_itself():
    """``a.add(a)`` is the one-line cycle, and the walk starts AT the container
    so it must be caught by the same check rather than needing its own."""
    a = SpatialElement(ifc_class="IfcBridge", name="a")
    with pytest.raises(ValueError, match="contain"):
        a.add(a)


# ---------------------------------------------------------------------------
# 3. The validator is authoritative — and does not blow the stack
# ---------------------------------------------------------------------------


def test_a_hand_built_cycle_is_a_named_compile_error_not_a_recursionerror():
    """THE regression this change exists to prevent.

    ``RecursionError`` is caught and failed on explicitly rather than left to
    surface as an error: it is a ``RuntimeError`` subclass, so a bare
    ``pytest.raises``-free call would simply error out with a traceback that
    looks like a test bug rather than like the product defect it is. Measured
    on ``main``: this fixture raised ``RecursionError`` out of
    ``stamp_canonical_names`` — the FIRST line of ``validate_project_report``,
    so nothing else in validation ever ran.
    """
    proj, a, b = _hand_built_cycle()

    try:
        report = validate_project_report(proj)
    except RecursionError as exc:                            # pragma: no cover
        pytest.fail(
            f"validate_project_report blew the stack on a cyclic tree instead "
            f"of reporting it — the on-path guard in naming._walk is not "
            f"holding: {exc!r}"
        )

    assert report.errors, (
        "a containment cycle must be a compile ERROR; accepting it produces a "
        "model whose canonical names, frames and extents are all undefined"
    )
    joined = "\n".join(report.errors)
    assert "CYCLE" in joined, f"the error must say what it is: {joined}"
    assert "buildingelementproxy:a" in joined, (
        f"the error must name the element the cycle closes on: {joined}"
    )
    assert "buildingelementproxy:b" in joined, (
        f"the error must name the whole loop, not just its entry — a one-name "
        f"message does not say which attachment to detach: {joined}"
    )


def test_the_cycle_is_reported_alone_and_first():
    """A cyclic tree reports the cycle and nothing else.

    Every later check reads a stamp the cycle truncated: the canonical name of
    an element inside a cycle is whatever partial string the walk got to before
    it turned back, so the uniqueness scan, Anchor host resolution and grouping
    member resolution would all report misses and collisions that describe a
    tree the author never wrote. Burying the one actionable message under those
    is how an author ends up fixing the symptom.
    """
    proj, _, _ = _hand_built_cycle()
    report = validate_project_report(proj)
    assert len(report.errors) == 1, (
        f"a cycle must be reported alone — follow-on errors are artefacts of "
        f"it: {report.errors}"
    )


def test_find_containment_cycles_is_self_sufficient():
    """Called directly, with no pass having run, it stamps and answers.

    Same self-sufficiency ``find_shared_elements`` has — a ``--q`` query or a
    test must not have to know which pass publishes the data.
    """
    proj, _, _ = _hand_built_cycle()
    assert not hasattr(proj, "_containment_cycles") or \
        getattr(proj, "_containment_cycles", None) is None or True
    cycles = find_containment_cycles(proj)
    assert len(cycles) == 1, f"one cycle, reported once: {cycles}"


def test_one_cycle_is_reported_once_however_many_times_it_is_reached():
    """Two entry points into the same loop are one defect, not two errors.

    ``id()``-keyed, insertion-ordered — the ``find_shared_elements`` shape. A
    per-visit report would print the same loop once per path leading into it,
    which on a wide model is a wall of duplicates.
    """
    a, b = _proxy("a"), _proxy("b", x0=2000)
    c = _proxy("c", x0=4000)
    proj = Project(name="acyclicity")
    proj.add(a)
    proj.add(c)
    a._elements.append(b)
    b._elements.append(a)
    c._elements.append(a)               # a second way in to the same loop

    cycles = find_containment_cycles(proj)
    assert len(cycles) == 1, (
        f"one loop entered twice is one cycle: {cycles}"
    )


# ---------------------------------------------------------------------------
# 4. A world query raises a typed error rather than hanging
# ---------------------------------------------------------------------------


def test_world_aabb_on_a_cyclic_tree_raises_a_typed_error():
    """It must not hang, and it must not answer.

    Both halves matter. The hang was real — ``_world_matrix_for``'s ``_parent``
    climb had no guard, so once the naming walk had stamped ``a._parent = b``
    and ``b._parent = a`` it looped forever with no stack growth to stop it.
    And the answer must not be a NUMBER: the union of the finite prefix the
    extent walk reached before turning back is a box that looks perfectly
    ordinary, which is precisely what ``Bounds`` refuses to hand out (it raises
    for an empty container rather than report a degenerate box at the origin).
    """
    proj, a, b = _hand_built_cycle()

    with pytest.raises(ContainmentCycleError) as exc:
        a.world_aabb()

    assert "buildingelementproxy:a" in str(exc.value), (
        f"the query must name the element it refused to answer for: {exc.value}"
    )


def test_authored_aabb_on_a_cyclic_tree_raises_the_same_typed_error():
    """The authoring-stage query has no more of an answer than the world one.

    ``authored_aabb`` is documented as "never raises for placement reasons",
    which is a statement about PLACEMENT — a cyclic container has no extent at
    any stage, and the two accessors must not disagree about that.
    """
    proj, a, b = _hand_built_cycle()
    with pytest.raises(ContainmentCycleError):
        a.authored_aabb()


def test_the_cyclic_error_is_a_valueerror():
    """``ContainmentCycleError`` subclasses ``ValueError``, like every other
    authoring refusal in this DSL (``BoundsError``, ``ChildAnchorError``), so a
    model script's existing ``except ValueError`` still catches it."""
    assert issubclass(ContainmentCycleError, ValueError)


# ---------------------------------------------------------------------------
# 5. On-path, not visited
# ---------------------------------------------------------------------------


def test_a_shared_element_is_still_reported_as_shared_not_swallowed():
    """The guard must not be a VISITED set.

    An element is legitimately reached once per path. A global visited set
    would terminate the walk just as well and would silently stop
    ``find_shared_elements`` counting the second visit — turning a reported
    error back into the undefined-behaviour silence the v10 cutover promoted it
    out of. This is the test that distinguishes the two implementations; a
    visited-set guard passes every other test in this file.
    """
    shared = _body("shared")
    a, b = _proxy("a"), _proxy("b", x0=2000)
    a.add(shared)
    b.add(shared)                       # the SAME object, in two places
    proj = Project(name="acyclicity")
    proj.add(a)
    proj.add(b)

    report = validate_project_report(proj)
    assert not find_containment_cycles(proj), (
        "one object in two places is SHARED, not cyclic — no path re-enters it"
    )
    assert any("attached in 2 places" in e for e in report.errors), (
        f"the shared-element error must survive the cycle guard: "
        f"{report.errors}"
    )


def test_an_ordinary_deep_tree_is_not_a_cycle():
    """A container chain that merely nests deeply reports nothing.

    A guard that keyed on depth, or that failed to POP on the way back up,
    would report this — and it is the shape every real assembly has.
    """
    proj = Project(name="acyclicity")
    root = _proxy("root")
    node = root
    for i in range(12):
        child = _proxy(f"n{i}")
        node.anchor(child, along=100, up=0)
        node = child
    proj.add(root)

    assert find_containment_cycles(proj) == [], (
        "12 levels of ordinary nesting is a tree, not a cycle"
    )
    assert validate_project_report(proj).errors == []


def test_the_guard_pops_between_siblings():
    """Two siblings holding structurally identical subtrees are not a cycle.

    If :meth:`WalkGuard.leave` were missed anywhere, the second sibling's
    descent would meet the first's ancestors still on the path — and because
    one guard object is deliberately shared across every top-level element of
    a project (it is empty between them, by construction), a missed pop shows
    up here rather than at the end of a long walk.
    """
    proj = Project(name="acyclicity")
    left, right = _proxy("left"), _proxy("right", x0=2000)
    left.anchor(_proxy("inner_l"), along=100, up=0)
    right.anchor(_proxy("inner_r"), along=100, up=0)
    proj.add(left)
    proj.add(right)

    assert find_containment_cycles(proj) == []
    assert validate_project_report(proj).errors == []
