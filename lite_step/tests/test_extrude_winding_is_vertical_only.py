"""Reversing an ``Extrude``'s contour flips it only when the contour is VERTICAL.

``agents/dsl-reference.md`` said "the contour winding decides the sign", flat. That is
half the rule. ``apply_orientation_rules`` forces the normal to point UP the moment
``|nz| > 0.001``, so a face one degree off vertical is already canonicalised and
reversing its points changes nothing.

Half a rule is worse than none here, because the failure is silent and dimensionally
perfect. Building a layered sloped buildup — a dormer roof of 1 mm zinc over 22 mm
boarding — an author who trusts the flat rule reverses the winding to send the boarding
downward, gets no flip, and the board extrudes UP through the zinc. The roof then renders
as bare chipboard while every thickness, every contour and every bounding box is exactly
as authored, nothing warns, and the schema validates. The right move is to offset each
layer's contour DOWN by what covers it, which is only obvious once you know the flat rule
is false.

So this pins both halves, from the AUTHOR's side (``extrude_offset``, the query the
reference tells you to ask) rather than from the orientation helper's:

* an exactly vertical face FLIPS on reversal — the rule as written;
* a face tilted off vertical does NOT — including one degree off, which is where the
  documented rule and the compiler part company;
* and the boundary is the epsilon in ``apply_orientation_rules``, not "roughly vertical".
"""
import math

import pytest

from lite_step.compiler.extent import extrude_offset
from lite_step.models import Extrude, Point


def _face(tilt_deg: float, reverse: bool = False, thickness: int = 100):
    """A 1 m square face tilted ``tilt_deg`` FROM VERTICAL (0 = wall, 90 = flat roof)."""
    t = math.radians(tilt_deg)
    dy, dz = math.sin(t) * 1000.0, math.cos(t) * 1000.0
    pts = [(0, 0, 0), (1000, 0, 0), (1000, dy, dz), (0, dy, dz)]
    if reverse:
        pts = list(reversed(pts))
    return Extrude(name="face", thickness=thickness,
                   contour=[Point(x=int(round(a)), y=int(round(b)), z=int(round(c)))
                            for a, b, c in pts])


def test_a_vertical_contour_flips_with_its_winding():
    """The rule as the DSL reference states it — true, on a vertical face."""
    forward = extrude_offset(_face(0))
    reversed_ = extrude_offset(_face(0, reverse=True))
    assert forward == pytest.approx((0.0, -100.0, 0.0), abs=1e-6)
    assert reversed_ == pytest.approx((0.0, 100.0, 0.0), abs=1e-6)


@pytest.mark.parametrize("tilt", [1, 5, 15, 30, 45, 60, 89, 90])
def test_a_tilted_contour_does_not(tilt):
    """…and false on everything else, including ONE DEGREE off vertical.

    The extrusion is canonicalised upward, so the two windings return the same offset."""
    forward = extrude_offset(_face(tilt))
    reversed_ = extrude_offset(_face(tilt, reverse=True))
    assert forward == pytest.approx(reversed_, abs=1e-6)
    assert forward[2] >= 0.0, "a canonicalised extrusion never points down"


def test_the_boundary_is_the_orientation_epsilon():
    """It is `|nz| > 0.001` in ``apply_orientation_rules``, not a vague 'near vertical'.

    A face tilted far enough for ``nz`` to clear that epsilon is canonicalised; below it
    the winding is still trusted. Pinning the threshold rather than a comfortable margin
    is what stops someone 'tidying' the epsilon and quietly moving the line between the
    two behaviours."""
    from lite_step.ifc.geometry import apply_orientation_rules

    assert apply_orientation_rules((0.0, -1.0, -0.0005)) == (0.0, -1.0, -0.0005), \
        "inside the epsilon: still treated as vertical, winding trusted"
    assert apply_orientation_rules((0.0, -1.0, -0.002)) == (-0.0, 1.0, 0.002), \
        "outside it: forced to point up, winding ignored"


def test_a_layered_sloped_buildup_stacks_by_offsetting_down():
    """The reason any of this matters, stated as geometry.

    Two layers on one 45 degree plane: 1 mm of metal whose outer face IS the design
    surface, and 22 mm of board under it. Both extrude UP, so the board's contour is
    dropped by metal + board and the metal's by metal — and the two solids then stack
    without overlapping. Author them at the same contour, or offset the board the other
    way, and the board's far face rises past the metal's."""
    n = (0.0, -math.sin(math.radians(45)), math.cos(math.radians(45)))

    def layer(drop, thickness):
        base = [(0, 0, 2000), (1000, 0, 2000), (1000, 1000, 3000), (0, 1000, 3000)]
        pts = [(x - n[0] * drop, y - n[1] * drop, z - n[2] * drop) for x, y, z in base]
        return _face_from(pts, thickness)

    def _face_from(pts, thickness):
        return Extrude(name="layer", thickness=thickness,
                       contour=[Point(x=int(round(a)), y=int(round(b)), z=int(round(c)))
                                for a, b, c in pts])

    metal = layer(1, 1)
    board = layer(1 + 22, 22)
    metal_far = metal.authored_aabb().max.z
    board_far = board.authored_aabb().max.z
    assert board_far <= metal_far, (
        "the boarding's far face must not rise past the metal's — if it does, the layer "
        "order is inverted and the roof renders as bare board")
