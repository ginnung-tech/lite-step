"""A ``Revolve``'s bound is the solid it occupies, not a cube around it.

Padding the axis endpoints by the profile's radial maximum ISOTROPICALLY — in all three axes — and said so in its own comment ("the
extent overshoots along the revolve AXIS"). Conservative against a missed
carve; it manufactures phantom ones instead.

Measured on a 5-wing 6-storey plan of annular floorplates, bare plates and
columns only: **1335 carve pairs of which ~30 are genuine**, CSG depth 7 of 10
before a single wall exists, 232 KB. A 275 mm plate claimed a 60 m cube, so
every storey "overlapped" every other one 4 m away in Z.

The geometry is exactly knowable — ``generator._create_revolve`` fixes the
frame: profile local Y is the axis, local X is radial, and the sweep angle
clips the disc to a sector.
"""
from __future__ import annotations

import math

import pytest

from lite_step.compiler.extent import padded_aabb
from lite_step.models import Point, Point2D, Revolve


def _plate(angle_centideg: int, axis=(0, 0, 1000)):
    """A 60-degree annular floorplate: r = 12000..30000, 275 thick."""
    return Revolve(
        name="plate",
        profile=[Point2D(x=12000, y=0), Point2D(x=30000, y=0),
                 Point2D(x=30000, y=275), Point2D(x=12000, y=275)],
        path=[Point(x=0, y=0, z=0),
              Point(x=axis[0], y=axis[1], z=axis[2])],
        angle=angle_centideg,
    )


def _size(elem):
    lo, hi = padded_aabb(elem, 1.0)
    return tuple(round(hi[i] - lo[i]) for i in range(3))


def test_the_axis_extent_is_the_profile_thickness_not_the_radius():
    """The headline. 275, not ~60000 — this is the phantom-carve source."""
    assert _size(_plate(6000))[2] == 275


def test_a_sector_is_bounded_by_the_wedge_it_occupies():
    """24000 x 25981: 25981 is 30000*sin(60) to the millimetre, and 24000 is
    30000 - 12000*cos(60) — the near radius bounds the other side."""
    assert _size(_plate(6000)) == (24000, 25981, 275)
    assert round(30000 * math.sin(math.radians(60))) == 25981


def test_a_full_revolution_is_the_whole_disc():
    """The falsifier for the angular clipping: with nothing to clip, the
    perpendicular extent must be the full diameter — but the AXIS stays
    tight, which is the part that mattered."""
    assert _size(_plate(36000)) == (60000, 60000, 275)


def test_the_old_isotropic_bound_would_have_failed_all_of_these():
    """Guard against a revert: the replaced rule padded by the radial max in
    every axis, so the box was ~60003 on a side. Nothing above can pass under
    it."""
    isotropic = round((30000 ** 2 + 275 ** 2) ** 0.5) * 2
    assert isotropic > 59000
    assert _size(_plate(6000))[2] < isotropic / 100


@pytest.mark.parametrize("axis", [(0, 0, 1000), (1000, 0, 0), (0, 1000, 0)])
def test_the_axis_stays_tight_whichever_way_it_points(axis):
    """The thin dimension must follow the AXIS, not a fixed world axis."""
    size = _size(_plate(36000, axis=axis))
    assert min(size) == 275, f"the 275 mm thickness moved: {size}"


def test_a_degenerate_axis_never_reaches_the_bound_at_all():
    """The fallback branch is unreachable, and that is the better answer.

    A zero-length axis is refused by the model at CONSTRUCTION, so
    ``_revolve_aabb`` never has to decide what a revolve with no axis means.
    The ``return None`` guard stays as a belt for a hand-built tree, but this
    records why it is not exercised.
    """
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        Revolve(name="d", profile=[Point2D(x=100, y=0), Point2D(x=200, y=50)],
                path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=0)], angle=9000)
