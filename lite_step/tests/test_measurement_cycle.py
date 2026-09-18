"""An element positioned from a measurement of itself is refused.

The compiler has always refused CONTAINMENT cycles — an element that contains
itself has no finite extent, so the walk cannot answer. A MEASUREMENT cycle is
different and, until now, worse: it terminates fine and emits a building.

The defect report of placed three meshes in a ring, each from a
``world_aabb()`` of the next. It compiled exit 0, no warning, and wrote an IFC
in which ``mesh1`` sat 1000 mm from where it believed ``mesh3`` was — because
each read answered against a position the others had not taken yet.

**The refusal is about intent, not solvability.** The compiler could pick an
answer; today it silently does, whichever the last write left behind. It
refuses because a ring does not describe one design. A window whose frame is
placed from its sash while the sash is placed from its frame expresses several
buildings, and statement order decides which one you get.

Why the existing stale-read warning does not cover this: it compares a global
mutation counter bumped by ``add`` / ``anchor`` / ``opening``, and a plain
``elem.placement = …`` assignment bumps nothing. Every read in the repro is
recorded at the current generation and nothing moves afterwards, so the check
sees no staleness at all. ``BimElement.__setattr__`` is what makes the write
observable.
"""

import pytest

from lite_step.compiler.walkguard import MeasurementCycleError
from lite_step.models import (
    Box,
    Mesh,
    Point,
    Point2D,
    Project,
    Storey,
    Transform,
)


def _mesh(name):
    return Mesh(
        name=name,
        heightmap=[[0, 50], [50, 0]],
        depth=100,
        corner_min=Point2D(x=-500, y=-500),
        corner_max=Point2D(x=500, y=500),
        source_type="synthetic",
    )


def _rooted(*names):
    """``(project, [elements…])`` — every element rooted, none placed."""
    proj = Project(name="cycle test")
    storey = Storey(name="ground")
    elems = [_mesh(n) for n in names]
    storey.add(*elems)
    proj.add_storey(storey)
    return proj, elems


def _at(z):
    return Transform(origin=Point(x=0, y=0, z=int(round(z))))


# ── the cycle is refused ────────────────────────────────────────────────────


def test_report_repro_raises_and_names_the_whole_ring():
    """The defect report's exact three-mesh ring.

    Delete the ``placement`` branch of ``BimElement.__setattr__`` and this
    goes green again — that hook is the entire fix, and a test that passes
    without it is testing nothing.
    """
    _, (m1, m2, m3) = _rooted("mesh1", "mesh2", "mesh3")

    m1.placement = _at(m3.world_aabb().max.z + 300)
    m2.placement = _at(m1.world_aabb().max.z + 300)

    with pytest.raises(MeasurementCycleError) as exc:
        m3.placement = _at(m2.world_aabb().max.z + 300)

    msg = str(exc.value)
    for name in ("mesh1", "mesh2", "mesh3"):
        assert name in msg, f"the refusal must name {name}: {msg}"
    # Intent, not solvability — an author told "no solution" goes looking for
    # a solver, and there isn't one to find.
    assert "more than one design" in msg
    assert "Anchor ONE of them" in msg


def test_two_element_cycle_raises():
    _, (a, b) = _rooted("a", "b")
    a.placement = _at(b.world_aabb().max.z + 100)
    with pytest.raises(MeasurementCycleError):
        b.placement = _at(a.world_aabb().max.z + 100)


def test_self_measurement_raises():
    """The degenerate ring: one element placed from its own position."""
    _, (a,) = _rooted("a")
    with pytest.raises(MeasurementCycleError):
        a.placement = _at(a.world_aabb().max.z + 100)


def test_four_element_cycle_raises():
    """Pins that the walk is transitive, not depth-1."""
    _, (a, b, c, d) = _rooted("a", "b", "c", "d")
    a.placement = _at(b.world_aabb().max.z + 100)
    b.placement = _at(c.world_aabb().max.z + 100)
    c.placement = _at(d.world_aabb().max.z + 100)
    with pytest.raises(MeasurementCycleError):
        d.placement = _at(a.world_aabb().max.z + 100)


def test_realistic_assembly_cycle_raises():
    """A frame placed from its sash while the sash is placed from the frame.

    The shape the feature exists for, and the one an author actually hits.
    Kept distinct from the bare-mesh cases because real nesting goes through
    container placement rather than a flat sibling list, and a synthetic
    three-mesh repro can pass while this path does not.
    """
    proj = Project(name="window")
    storey = Storey(name="ground")
    frame = Box(name="frame", start=Point(x=0, y=0, z=0), end=Point(x=1200, y=100, z=1400))
    sash = Box(name="sash", start=Point(x=60, y=10, z=60), end=Point(x=1140, y=90, z=1340))
    storey.add(frame, sash)
    proj.add_storey(storey)

    frame.placement = _at(sash.world_aabb().max.z + 10)
    with pytest.raises(MeasurementCycleError):
        sash.placement = _at(frame.world_aabb().max.z + 10)


# ── what must NOT be refused ────────────────────────────────────────────────
#
# The boundary is structural, not a judgement about whether a read "mattered":
# a WORLD read followed by a placement makes an edge, and a ring of edges is
# refused. Everything below is outside that boundary for a reason you can state
# without evaluating any arithmetic — no ring closes (chain, diamond), or the
# read was never a world read at all (authored_aabb, height_at).
#
# What is NOT a reason to land here: "the value did not really affect the
# result". See ``test_reading_and_discarding_the_value_still_refuses``.


def test_chain_is_not_a_cycle():
    """a→b→c and stop. Three edges, no ring."""
    _, (a, b, c) = _rooted("a", "b", "c")
    a.placement = _at(b.world_aabb().max.z + 100)
    b.placement = _at(c.world_aabb().max.z + 100)  # must not raise


def test_place_then_read_is_the_correct_order():
    """The documented pattern: fix one element, measure outward from it."""
    _, (a, b) = _rooted("a", "b")
    a.placement = _at(500)                          # fixed coordinate
    b.placement = _at(a.world_aabb().max.z + 100)   # measured from it


def test_diamond_is_not_a_cycle():
    """Two elements measured from the same third. Shared target, no ring."""
    _, (base, left, right) = _rooted("base", "left", "right")
    base.placement = _at(500)
    left.placement = _at(base.world_aabb().max.z + 100)
    right.placement = _at(base.world_aabb().max.z + 200)


def test_authored_aabb_is_not_a_measurement_dependency():
    """``authored_aabb`` is the extent AS WRITTEN, before any container places
    the element. It carries no placement dependency by definition, and its
    pre-carve extent is load-bearing for miter extension."""
    _, (a, b) = _rooted("a", "b")
    a.placement = _at(b.authored_aabb().max.z + 100)
    b.placement = _at(a.authored_aabb().max.z + 100)  # must not raise


def test_unplaced_heightmap_height_at_stays_free():
    """Pins the villa pattern verbatim.

    ``Mesh.height_at`` answers from the local heightmap while ``placement is
    None``, and ``08_murermestervilla`` derives its foundation elevation that
    way — on a mesh not yet added to the project. A future tightening that
    swept ``height_at`` into this feature would break ``main`` and the pattern
    documented with the site-block corpus system.
    """
    terrain = _mesh("site_mesh")
    smoothed = terrain.smooth(strength=25, height_map_only=True)
    surface_z = smoothed.height_at(x=0, y=0, round=True)
    assert isinstance(surface_z, int)

    proj = Project(name="villa")
    storey = Storey(name="ground")
    slab = Box(
        name="slab",
        start=Point(x=-100, y=-100, z=surface_z - 200),
        end=Point(x=100, y=100, z=surface_z),
    )
    storey.add(slab)
    proj.add_storey(storey)


def test_deep_chain_does_not_recurse():
    """200 links. Iterative walk, or a RecursionError from inside the
    compiler reports the wrong problem entirely."""
    names = [f"m{i}" for i in range(200)]
    _, elems = _rooted(*names)
    elems[-1].placement = _at(500)
    for i in range(len(elems) - 2, -1, -1):
        elems[i].placement = _at(elems[i + 1].world_aabb().max.z + 10)

    # And the ring that closes it is still caught at depth 200.
    with pytest.raises(MeasurementCycleError):
        elems[-1].placement = _at(elems[0].world_aabb().max.z + 10)


def test_reading_and_discarding_the_value_still_refuses():
    """A read you do not use is still a read. Refuse it anyway.

    ``* 0`` makes the measurement have no arithmetic effect on the result,
    and the ring is still refused. This is deliberate
    and is NOT a false positive to be engineered away:

    The check is on what the author *declared*, not on what the arithmetic
    happens to survive. An author who reaches for ``b.world_aabb()`` while
    positioning ``a`` has said the two are related; whether the expression
    then multiplies it away is an accident of the formula, not a statement
    of independence. Tracing which reads actually reach the assigned value
    would mean evaluating author arithmetic to decide whether to refuse —
    the compiler guessing at intent, which is the thing this feature exists
    to stop.

    So: do not add use-tracking, taint analysis, or an "unused read" escape
    hatch. The loud failure is the product.
    """
    _, (a, b) = _rooted("a", "b")
    a.placement = _at(500 + b.world_aabb().max.z * 0)
    with pytest.raises(MeasurementCycleError):
        b.placement = _at(500 + a.world_aabb().max.z * 0)
