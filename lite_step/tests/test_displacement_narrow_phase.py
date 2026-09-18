"""The displacement NARROW PHASE — a carve is emitted only if it removes material.

The trigger is a padded AABB, and the pad is a deliberate over-approximation:
a member's section reach is added on every axis so that a member takes part in
displacement at all (``extent.padded_aabb``). The price is a boolean handed to
every pair that merely comes CLOSE. Those booleans remove nothing, and a
boolean that removes nothing is not free — it adds vertices, it adds a level to
the host's CSG chain, and web-ifc drops the deepest chains outright.

So the padded AABB proposes and an exact separating-axis test disposes. For the
shapes it answers for the distinction between box and solid does not exist: a
``Box`` and a straight rectangular-section member ARE oriented boxes, so SAT is
exact rather than a bound.

The invariant these tests defend is absolute and is what makes the change safe:
**a dropped carve removes exactly zero material.** Fewer booleans, identical
solids. Anything the test cannot decide exactly keeps its carve.
"""

from __future__ import annotations

import numpy as np
import pytest

from lite_step.compiler.displacement import apply_displacement, carve_pairs
from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.compiler.extent import (
    convex_solid_obb,
    obbs_are_disjoint,
)
from lite_step.models import Box, Material, Point, Project, Sweep
from lite_step.models.project import Storey

TIMBER = "Timber_C24"
RAFTER = (45, 195)
COLLAR = (45, 145)


def _member(name, profile, p0, p1):
    return Sweep(name=name, path=[p0, p1],
                 material=Material(key=TIMBER, profile_mm=profile))


def _compile(*elements):
    """Run the pass the way a real compile does — normalize to metres FIRST.

    Skipping the normalize is not a shortcut, it is a different test: the pad
    divides ``Material(profile_mm=)`` by the unit domain, so a project still in
    millimetres gets a pad 1000x too small and NOTHING overlaps. That mistake
    reads as "the narrow phase dropped everything" and is why this helper
    exists instead of calling ``apply_displacement`` directly.
    """
    proj = Project(name="narrow phase")
    storey = Storey(elevation=0)
    proj.add_storey(storey)
    storey.add(*elements)
    # RETURNS a deep copy; it does not mutate. Dropping the return value
    # leaves the project in millimetres while the pad is computed in metres,
    # so every AABB comes out 1000x too small and nothing overlaps at all.
    proj = normalize_project_to_meters(proj)
    apply_displacement(proj)
    return proj


def _pair_names(proj):
    return {(loser.split(":")[-1], winner.split(":")[-1])
            for loser, winner, _ in carve_pairs(proj)}


# ---------------------------------------------------------------------------
# The behaviour
# ---------------------------------------------------------------------------


def test_face_to_face_members_do_not_carve():
    """Two members butted on a shared plane touch; they are not joined.

    The padded AABB says they overlap — each is padded by its section
    half-diagonal, ~76 mm, against a 0 mm gap. The solids say otherwise.
    """
    a = _member("a", COLLAR, Point(x=0, y=-1000, z=0), Point(x=0, y=1000, z=0))
    b = _member("b", COLLAR, Point(x=45, y=-1000, z=0), Point(x=45, y=1000, z=0))
    proj = _compile(a, b)
    assert _pair_names(proj) == set()
    assert proj._carve_inert == 1


def test_one_millimetre_overlap_still_carves():
    """The counter-test, and the one that must fail if the threshold slips.

    Same pair moved 1 mm closer, so they really do share 1 mm of material.
    That is a real joint and has to survive: it is the difference between
    "dropped a boolean that did nothing" and "deleted the joinery".
    """
    a = _member("a", COLLAR, Point(x=0, y=-1000, z=0), Point(x=0, y=1000, z=0))
    b = _member("b", COLLAR, Point(x=44, y=-1000, z=0), Point(x=44, y=1000, z=0))
    proj = _compile(a, b)
    assert _pair_names(proj) == {("a", "b")}
    assert proj._carve_inert == 0


def test_near_miss_inside_the_pad_does_not_carve():
    """A gap smaller than the pad but larger than zero — the case the pad
    manufactures a boolean for and the solids reject."""
    a = _member("a", COLLAR, Point(x=0, y=-1000, z=0), Point(x=0, y=1000, z=0))
    b = _member("b", COLLAR, Point(x=70, y=-1000, z=0), Point(x=70, y=1000, z=0))
    proj = _compile(a, b)
    assert _pair_names(proj) == set()
    assert proj._carve_inert == 1


def test_a_real_overlap_is_untouched():
    """The ordinary case: two members genuinely interpenetrating still carve,
    in tree order, exactly as before."""
    a = _member("a", RAFTER, Point(x=0, y=-1000, z=0), Point(x=0, y=1000, z=0))
    b = _member("b", RAFTER, Point(x=0, y=0, z=-1000), Point(x=0, y=0, z=1000))
    proj = _compile(a, b)
    assert _pair_names(proj) == {("a", "b")}
    assert proj._carve_inert == 0


def test_boxes_touching_on_a_face_do_not_carve():
    """Same rule for plain ``Box`` solids, whose OBB is axis-aligned."""
    a = Box(name="a", start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000),
            material="Concrete")
    b = Box(name="b", start=Point(x=1000, y=0, z=0), end=Point(x=2000, y=1000, z=1000),
            material="Concrete")
    proj = _compile(a, b)
    assert _pair_names(proj) == set()


def test_boxes_overlapping_by_one_millimetre_still_carve():
    a = Box(name="a", start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000),
            material="Concrete")
    b = Box(name="b", start=Point(x=999, y=0, z=0), end=Point(x=2000, y=1000, z=1000),
            material="Concrete")
    proj = _compile(a, b)
    assert _pair_names(proj) == {("a", "b")}


def test_the_report_names_the_joint_that_is_not_there():
    """A count says how much noise the pad made; the list says WHICH joint is
    missing. That is the question an author has when a member looks attached in
    the render and is in fact holding nothing up — and it is unanswerable from
    a render, because the element still draws."""
    from lite_step.compiler.displacement import (
        carve_report_lines,
        near_miss_report_lines,
    )
    a = _member("a", COLLAR, Point(x=0, y=-1000, z=0), Point(x=0, y=1000, z=0))
    b = _member("b", COLLAR, Point(x=70, y=-1000, z=0), Point(x=70, y=1000, z=0))
    proj = _compile(a, b)
    assert any("a" in line and "b" in line
               for line in near_miss_report_lines(proj))
    # And it must NOT reach the carve report, whose LENGTH is the reported
    # carve count — a near miss counted as a carve would say the opposite of
    # what happened.
    assert carve_report_lines(proj) == []


def test_a_declared_but_empty_carve_is_loud(caplog):
    """Dropping an INFERRED near-miss is silent — nobody asked for it.

    A DECLARED one is different: the author said these two meet, and the
    geometry says they do not. That is a joint which is not there, and it is
    invisible in a render because the element still draws. It has to say so.
    """
    a = _member("a", COLLAR, Point(x=0, y=-1000, z=0), Point(x=0, y=1000, z=0))
    b = _member("b", COLLAR, Point(x=70, y=-1000, z=0), Point(x=70, y=1000, z=0))
    proj = Project(name="declared")
    storey = Storey(elevation=0)
    proj.add_storey(storey)
    storey.add(a)
    storey.add(b, carve="other")
    proj = normalize_project_to_meters(proj)
    with caplog.at_level("WARNING"):
        apply_displacement(proj)
    assert carve_pairs(proj) == []
    assert "do not intersect" in caplog.text


# ---------------------------------------------------------------------------
# What it refuses to decide — every one of these must KEEP its carve
# ---------------------------------------------------------------------------


def test_a_bent_member_is_not_decided():
    """A three-point sweep's oriented box fills the inside of the bend, which
    is sound but so loose it would decide nothing. Excluded on purpose, so the
    pair keeps whatever the padded AABB said."""
    bent = Sweep(name="bent",
                 path=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0),
                       Point(x=1000, y=1000, z=0)],
                 material=Material(key=TIMBER, profile_mm=COLLAR))
    assert convex_solid_obb(bent) is None


def test_a_flat_box_is_refused_rather_than_answered():
    """The one failure mode that would DELETE joinery instead of noise.

    A box with no thickness on some axis is a plane, and a zero-extent box
    separates from everything by construction — so answering for it would drop
    every carve it takes part in, silently. It must come back ``None``, which
    routes the pair to the padded AABB exactly as today.
    """
    flat = Box(name="flat", start=Point(x=0, y=0, z=0),
               end=Point(x=1000, y=1000, z=0), material="Concrete")
    assert convex_solid_obb(flat) is None


def test_a_zero_width_section_is_refused():
    """Same failure mode via the section rather than the path."""
    # A ``Material(profile_mm=)`` section always has both dimensions, so the
    # guard is reached through an explicit degenerate profile instead.
    from lite_step.models import Point2D
    degenerate = Sweep(
        name="d",
        profile=[Point2D(x=0, y=-50), Point2D(x=0, y=50),
                 Point2D(x=0, y=50), Point2D(x=0, y=-50)],
        path=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0)],
        material="Timber_C24")
    assert convex_solid_obb(degenerate) is None
    # The healthy control, so this cannot pass by refusing everything.
    healthy = _member("healthy", COLLAR, Point(x=0, y=0, z=0),
                      Point(x=1000, y=0, z=0))
    assert convex_solid_obb(healthy) is not None


def test_a_curved_solid_is_not_decided():
    from lite_step.models import Pipe
    pipe = Pipe(name="p", path=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0)],
                radius=50, material="Steel_S355")
    assert convex_solid_obb(pipe) is None


def test_the_dsl_refuses_a_zero_length_member_outright():
    """The degenerate-path guard in ``convex_solid_obb`` can never be reached
    from authored source — the constructor refuses first. Pinned so the guard
    is understood as belt-and-braces rather than a live path."""
    with pytest.raises(Exception):
        Sweep(name="d", path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=0)],
              material=Material(key=TIMBER, profile_mm=COLLAR))


# ---------------------------------------------------------------------------
# The separating-axis test itself
# ---------------------------------------------------------------------------


def _box(centre, half, axes=None):
    return (np.array(centre, float),
            np.eye(3) if axes is None else np.asarray(axes, float),
            np.array(half, float))


@pytest.mark.parametrize("dx,disjoint", [
    (2.0, True),        # clear gap
    (1.0, True),        # exact face contact -> touching, not joined
    (0.999, False),     # 1 per-mille overlap
    (0.0, False),       # concentric
])
def test_sat_contact_lands_on_the_disjoint_side(dx, disjoint):
    assert obbs_are_disjoint(_box((0, 0, 0), (0.5, 0.5, 0.5)),
                             _box((dx, 0, 0), (0.5, 0.5, 0.5))) is disjoint


def test_sat_agrees_with_an_exact_mesh_boolean_on_random_pairs():
    """SAT is only worth trusting if it matches a real intersection.

    Cross-checked against manifold3d — already a dependency, used by the mesh
    carve path — over randomly oriented pairs. An UNSAFE answer is the one that
    matters: claiming disjoint when the solids really overlap would delete
    joinery, so it is asserted at zero rather than at a rate.
    """
    manifold3d = pytest.importorskip("manifold3d")
    Manifold = manifold3d.Manifold
    rng = np.random.default_rng(20260820)

    def rand():
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[0] = -q[0]
        return rng.uniform(-3, 3, 3), q, rng.uniform(0.2, 1.5, 3)

    def mesh(obb):
        c, ax, e = obb
        pts = np.array([[sx * e[0], sy * e[1], sz * e[2]]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return Manifold.hull_points((pts @ ax + c).astype(np.float32))

    unsafe = overlapping = 0
    for _ in range(600):
        a, b = rand(), rand()
        said_disjoint = obbs_are_disjoint(a, b)
        volume = (mesh(a) ^ mesh(b)).volume()
        if volume > 1e-6:
            overlapping += 1
            if said_disjoint:
                unsafe += 1
    # The fixture has to actually exercise both answers, or a test that always
    # returns "disjoint" would pass it.
    assert overlapping > 50, "fixture produced almost no overlapping pairs"
    assert unsafe == 0


def test_sat_separates_along_an_edge_axis():
    """The nine edge cross-products are not decoration: two boxes can be
    separated by an axis that is a face normal of NEITHER. Without them SAT
    reports a false overlap, which is safe but useless — this pins that the
    complete axis set is in play."""
    tilt = np.radians(45)
    rot = np.array([[np.cos(tilt), -np.sin(tilt), 0],
                    [np.sin(tilt), np.cos(tilt), 0],
                    [0, 0, 1]])
    a = _box((0, 0, 0), (1.0, 0.05, 0.05))
    b = _box((0.9, 0.9, 0), (1.0, 0.05, 0.05), rot)
    assert obbs_are_disjoint(a, b) is True


def test_obb_of_a_straight_member_is_its_real_section():
    """The oriented box must BE the member, not a box around it — that is the
    whole reason the test is exact. A 45x145 collar 2 m long measures exactly
    that, and a half-diagonal pad would answer 76 mm on both cross axes."""
    collar = _member("c", COLLAR, Point(x=0, y=-1000, z=0), Point(x=0, y=1000, z=0))
    centre, axes, half = convex_solid_obb(collar)
    assert np.allclose(sorted(2 * half), [45.0, 145.0, 2000.0])
    assert np.allclose(centre, [0, 0, 0])
    assert np.allclose(np.abs(axes @ axes.T), np.eye(3), atol=1e-9)
