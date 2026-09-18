"""``obb()`` — the placed extent, oriented to how the shape was WRITTEN.

The design constraint that shapes every test here: the frame comes from the
element's own authored construction direction — a contour normal, a path, a
revolve axis, a stamped wall frame — and NEVER from fitting a minimal box to
the geometry.

Two failure modes that distinction avoids, both pinned below:

* a minimal fit swaps ``along`` and ``up`` on a near-cube, and a 1 mm edit
  flips them back, so a helper reading ``point_at(100, 0, 0)`` silently
  changes which face it means;
* a Newell normal flips with the contour winding, so the same shape wound the
  other way would return swapped corners.

And where there is genuinely no single direction, ``rule`` says ``"aabb"``
rather than picking one. That is the honest answer to "which way does this
point".
"""
import math

import pytest

from lite_step.models import (
    Box,
    Extrude,
    Mesh,
    Pipe,
    Point,
    Point2D,
    Project,
    Revolve,
    Sweep,
    Wall,
)
from lite_step.models.bounds import WORLD_AXES, BoundsError

_X, _Y, _Z = 3170, -2410, 1130


def _rooted(elem):
    Project(name="t").add(elem)
    return elem


def _size(b):
    return (b.size.x, b.size.y, b.size.z)


# ---------------------------------------------------------------------------
# The case this feature exists for
# ---------------------------------------------------------------------------


def test_slanted_extrude_gets_a_box_that_hugs_it():
    """A 3000x3000 diagonal wall, 300 thick.

    Axis-aligned, it reports 3212 x 3212 — two numbers that describe the
    bounding box of the slant rather than the wall. Oriented, it reports its
    real dimensions: 4243 (= 3000*sqrt(2)) along the run, 300 through, 2500 up.
    """
    e = _rooted(Extrude(
        contour=[Point(x=0, y=0, z=0), Point(x=3000, y=3000, z=0),
                 Point(x=3000, y=3000, z=2500), Point(x=0, y=0, z=2500)],
        thickness=300, name="slanted"))

    assert _size(e.world_aabb()) == (3212, 3212, 2500)
    assert _size(e.obb()) == (round(3000 * math.sqrt(2)), 300, 2500)
    assert e.obb().rule == "extrude-normal"
    assert e.obb().exact is True


def test_the_two_faces_of_a_slanted_wall_are_one_thickness_apart():
    """What the OBB is FOR: addressing a face you could not name otherwise."""
    e = _rooted(Extrude(
        contour=[Point(x=0, y=0, z=0), Point(x=3000, y=3000, z=0),
                 Point(x=3000, y=3000, z=2500), Point(x=0, y=0, z=2500)],
        thickness=300, name="slanted"))
    b = e.obb()
    a, c = b.point_at(0, -100, 0), b.point_at(0, 100, 0)
    dist = math.dist((a.x, a.y, a.z), (c.x, c.y, c.z))
    assert round(dist) == 300


def test_diagonal_pipe_reports_length_by_diameter_by_diameter():
    p = _rooted(Pipe(path=[Point(x=0, y=0, z=0), Point(x=2000, y=2000, z=0)],
                     radius=100, name="diag"))
    assert _size(p.world_aabb()) == (2200, 2200, 200)
    assert _size(p.obb()) == (round(2000 * math.sqrt(2)), 200, 200)
    assert p.obb().rule == "path-direction"


def test_revolve_axis_becomes_up():
    """The revolve axis is the one direction carrying no shape information —
    the body is radially symmetric about it — so it is ``up``, not ``along``."""
    r = _rooted(Revolve(
        profile=[Point2D(x=100, y=0), Point2D(x=300, y=0), Point2D(x=300, y=200)],
        path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X, y=_Y, z=_Z + 900)],
        angle=360, name="r"))
    b = r.obb()
    assert b.rule == "revolve-axis"
    assert b.size.z == 900, "the axis span is the up extent"
    assert b.size.x == b.size.y == 600, (
        "2 x the profile's radial max — max(p.x), the generator's own bound. "
        "The displacement pad uses the full 2D norm (sqrt(x^2+y^2)), which "
        "over-approximates: correct for a carve trigger, wrong as an extent.")


# ---------------------------------------------------------------------------
# Where there is no single direction, say so
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("elem", [
    Box(start=Point(x=_X, y=_Y, z=_Z),
        end=Point(x=_X + 1000, y=_Y + 400, z=_Z + 2000), name="bx"),
    Mesh(vertices=[Point(x=_X, y=_Y, z=_Z), Point(x=_X + 900, y=_Y, z=_Z),
                   Point(x=_X, y=_Y + 700, z=_Z + 300)],
         faces=[[0, 1, 2]], name="tri"),
    Sweep(profile=[Point2D(x=-90, y=-45), Point2D(x=90, y=-45),
                   Point2D(x=90, y=45), Point2D(x=-90, y=45)],
          path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X + 1900, y=_Y + 700, z=_Z),
                Point(x=_X + 3100, y=_Y + 700, z=_Z + 800)], name="bent"),
], ids=["box", "mesh", "multi-segment-sweep"])
def test_no_authored_direction_reports_rule_aabb(elem):
    """A Box is axis-aligned by construction, a Mesh is a point cloud, and a
    multi-segment sweep has several directions — none has ONE frame.

    Returning the axis-aligned answer and SAYING so beats picking the first
    segment and hoping: an author can branch on ``rule``, but cannot detect a
    guess.
    """
    _rooted(elem)
    b = elem.obb()
    assert b.rule == "aabb"
    assert b.axes == WORLD_AXES
    assert _size(b) == _size(elem.world_aabb())


# ---------------------------------------------------------------------------
# The two stability properties
# ---------------------------------------------------------------------------


def test_reversed_contour_winding_keeps_the_AXES_stable():
    """A Newell normal flips with the winding, and with it which corner is
    ``min``. The wall rule solves that by canonicalizing toward +X (ties +Y),
    and the raw-primitive frame reuses that convention rather than inventing
    a second one — so ``point_at(100, 0, 0)`` addresses the same face
    whichever way the author happened to wind the contour.

    The CORNERS are a different matter and deliberately not asserted equal:
    on a VERTICAL contour ``apply_orientation_rules`` trusts the winding (it
    only flips a normal with a Z component), so the two windings extrude to
    OPPOSITE SIDES of the plane and are genuinely different solids, offset by
    one thickness. The frame is winding-independent; the geometry is not, and
    that is the generator's documented behaviour rather than something this
    accessor should paper over.
    """
    forward = [Point(x=0, y=0, z=0), Point(x=2400, y=900, z=0),
               Point(x=2400, y=900, z=1700), Point(x=0, y=0, z=1700)]
    a = _rooted(Extrude(contour=list(forward), thickness=220, name="fwd"))
    b = _rooted(Extrude(contour=list(reversed(forward)), thickness=220, name="rev"))

    assert a.obb().axes == b.obb().axes, "the FRAME must not depend on winding"
    assert _size(a.obb()) == _size(b.obb())
    # ...and the two solids sit one thickness apart, on opposite faces.
    ca, cb = a.obb().center, b.obb().center
    assert round(math.dist((ca.x, ca.y, ca.z), (cb.x, cb.y, cb.z))) == 220


def test_a_near_cube_survives_a_one_millimetre_edit():
    """The failure a MINIMAL-FIT box has and an authored frame does not.

    Fit a box to a 2000 x 2001 x 2000 solid and ``along`` is the 2001 axis;
    shave one millimetre and it jumps elsewhere — so a helper reading
    ``point_at(100, 0, 0)`` addresses a different face after an edit that
    changed nothing about how the shape was written.
    """
    def cube(dy):
        return _rooted(Extrude(
            contour=[Point(x=0, y=0, z=0), Point(x=2000, y=0, z=0),
                     Point(x=2000, y=0, z=dy), Point(x=0, y=0, z=dy)],
            thickness=2000, name=f"c{dy}"))

    before, after = cube(2001).obb(), cube(2000).obb()
    assert before.axes == after.axes, "a 1 mm edit must not re-orient the box"
    assert before.rule == after.rule == "extrude-normal"


def test_obb_axes_are_orthonormal_and_right_handed():
    import numpy as np

    e = _rooted(Extrude(
        contour=[Point(x=0, y=0, z=0), Point(x=1700, y=1100, z=0),
                 Point(x=1700, y=1100, z=2300), Point(x=0, y=0, z=2300)],
        thickness=180, name="e"))
    a0, a1, a2 = (np.array(a) for a in e.obb().axes)
    for a in (a0, a1, a2):
        assert abs(np.linalg.norm(a) - 1.0) < 1e-9
    assert abs(float(np.dot(a0, a1))) < 1e-9
    assert abs(float(np.dot(a1, a2))) < 1e-9
    assert float(np.dot(np.cross(a0, a1), a2)) > 0, "right-handed"


# ---------------------------------------------------------------------------
# Shared contract with the other accessors
# ---------------------------------------------------------------------------


def test_obb_raises_until_rooted_like_world_aabb():
    e = Extrude(contour=[Point(x=0, y=0, z=0), Point(x=1000, y=1000, z=0),
                         Point(x=1000, y=1000, z=800), Point(x=0, y=0, z=800)],
                thickness=100, name="e")
    with pytest.raises(BoundsError, match="not reachable from a Project"):
        e.obb()


def test_min_and_max_are_still_the_two_diagonal_corners():
    """Uniform across all three accessors: the same two corners, in world mm.
    What changes is only the directions they are separated along."""
    e = _rooted(Extrude(
        contour=[Point(x=0, y=0, z=0), Point(x=2000, y=2000, z=0),
                 Point(x=2000, y=2000, z=1500), Point(x=0, y=0, z=1500)],
        thickness=250, name="e"))
    b = e.obb()
    assert b.min == b.point_at(-100, -100, -100)
    assert b.max == b.point_at(100, 100, 100)
    assert b.is_axis_aligned is False, "an oriented box is not axis-aligned"


def test_size_is_not_max_minus_min_on_an_oriented_box():
    """The trap the docstring warns about, made concrete.

    On a rotated OBB the two corners are a diagonal, so subtracting them is
    meaningless — and can even be negative. ``.size`` is the accessor.
    """
    e = _rooted(Extrude(
        contour=[Point(x=0, y=0, z=0), Point(x=3000, y=3000, z=0),
                 Point(x=3000, y=3000, z=2000), Point(x=0, y=0, z=2000)],
        thickness=300, name="e"))
    b = e.obb()
    assert b.max.y - b.min.y != b.size.y
    assert b.size.y == 300
