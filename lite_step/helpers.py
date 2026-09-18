"""Path helpers for Lite-STEP model scripts (DSL v1.5, WS1 PR-E).

The v1.5 file template imports these directly::

    from lite_step.helpers import arc_path, stirrup_path

``arc_path`` / ``stirrup_path`` generate ``[Point]`` lists for the path-based
primitives (Pipe, Bar, path-form Sweep). All outputs are int millimeters
(``round()``-rounded), so they construct cleanly under the strict int-mm
flag. All angles are centidegrees (DSL RULE 3).

Site/terrain geometry is authored as a ``Mesh`` under a ``Site`` container.
"""

from __future__ import annotations

import math
from typing import List, Optional

from lite_step.models import Point

__all__ = [
    "arc_path",
    "stirrup_path",
]

_PLANES = {
    # plane -> (unit vector for local u, unit vector for local v) as
    # (x, y, z) component selectors. u = cos direction, v = sin direction.
    "xy": ((1, 0, 0), (0, 1, 0)),
    "xz": ((1, 0, 0), (0, 0, 1)),
    "yz": ((0, 1, 0), (0, 0, 1)),
}


def _plane_point(center: Point, plane: str, u: float, v: float) -> Point:
    """Point at center + u*U + v*V in the given plane, int-mm rounded."""
    du, dv = _PLANES[plane]
    return Point(
        x=int(round(center.x + du[0] * u + dv[0] * v)),
        y=int(round(center.y + du[1] * u + dv[1] * v)),
        z=int(round(center.z + du[2] * u + dv[2] * v)),
    )


def arc_path(
    center: Point,
    radius: int,
    start_angle: int,
    end_angle: int,
    segments: int = 16,
    plane: str = "xy",
) -> List[Point]:
    """Polyline approximation of a circular arc, as ``segments + 1`` Points.

    Angles are centidegrees, measured in the given plane from its first
    axis toward its second (right-hand): ``plane="xy"`` -> 0 = +X,
    9000 = +Y; ``"xz"`` -> 0 = +X, 9000 = +Z; ``"yz"`` -> 0 = +Y,
    9000 = +Z. ``end_angle`` may be smaller than ``start_angle`` for a
    clockwise arc. Coordinates are int mm (``round()``), per DSL RULE 2.

    Example — semicircle over a doorway, in the XZ (wall) plane:
        arc_path(center=Point(x=0, y=4350, z=2000), radius=500,
                 start_angle=0, end_angle=18000, plane="xz")
    """
    if plane not in _PLANES:
        raise ValueError(
            f"arc_path: plane must be one of {sorted(_PLANES)} (got {plane!r})"
        )
    if radius <= 0:
        raise ValueError(f"arc_path: radius must be > 0 (got {radius})")
    if segments < 1:
        raise ValueError(f"arc_path: segments must be >= 1 (got {segments})")
    if start_angle == end_angle:
        raise ValueError("arc_path: start_angle and end_angle must differ")
    pts: List[Point] = []
    for i in range(segments + 1):
        ang_cd = start_angle + (end_angle - start_angle) * i / segments
        ang = math.radians(ang_cd / 100.0)
        pts.append(
            _plane_point(center, plane, radius * math.cos(ang), radius * math.sin(ang))
        )
    return pts


#: Diameter-independent default hook extension (mm) for ``stirrup_path``
#: when ``hook_length=None``. Pinned to 80 mm = the EC2 10*phi extension for
#: the smallest stock stirrup diameter (phi 8). For larger bars pass
#: ``hook_length`` explicitly (10*phi) — the helper is deliberately
#: diameter-agnostic so its output is a pure geometry path.
DEFAULT_HOOK_LENGTH_MM = 80


def stirrup_path(
    width: int,
    height: int,
    origin: Point,
    hook_angle: int = 13500,
    hook_length: Optional[int] = None,
    plane: str = "xz",
) -> List[Point]:
    """Closed rectangular stirrup path with EC2-style end hooks.

    Centerline rectangle ``width x height`` in the given plane, with
    ``origin`` at the MIDPOINT OF THE BOTTOM LEG (matching the spec's
    beam example: ``origin=Point(x=0, y=..., z=cover)`` centers the
    stirrup on the beam axis). The closure sits at the top-left corner:
    both hook legs bend ``hook_angle`` centidegrees (default 13500 = the
    EC2 135-degree seismic hook) off the perimeter direction and extend
    ``hook_length`` mm into the section core.

    ``hook_length=None`` -> :data:`DEFAULT_HOOK_LENGTH_MM` (80 mm = 10*phi
    for phi 8, the diameter-independent default — pass 10*phi explicitly
    for larger bars).

    NOTE: the two hook legs are coincident in centerline geometry (the
    physical offset between them is one bar diameter, which a
    diameter-agnostic path cannot know). Feed the result to
    ``Bar(path=stirrup_path(...), diameter=..., bar_type="stirrup")``.

    Returns the path as ``[Point]``: hook1_end -> corner -> around the
    rectangle -> corner -> hook2_end.
    """
    if plane not in _PLANES:
        raise ValueError(
            f"stirrup_path: plane must be one of {sorted(_PLANES)} (got {plane!r})"
        )
    if width <= 0 or height <= 0:
        raise ValueError(
            f"stirrup_path: width and height must be > 0 (got {width}x{height})"
        )
    if not (3000 <= hook_angle <= 16000):
        raise ValueError(
            f"stirrup_path: hook_angle must be within [3000, 16000] "
            f"centidegrees (got {hook_angle})"
        )
    if hook_length is None:
        hook_length = DEFAULT_HOOK_LENGTH_MM
    if hook_length <= 0:
        raise ValueError(
            f"stirrup_path: hook_length must be > 0 (got {hook_length})"
        )

    half_w = width / 2.0
    # Rectangle corners in local (u, v), origin at bottom-leg midpoint:
    #   D(-w/2, h) ---- C(+w/2, h)
    #   A(-w/2, 0) ---- B(+w/2, 0)
    # Loop D -> A -> B -> C -> D with the closure (both hooks) at D.
    corner_a = (-half_w, 0.0)
    corner_b = (half_w, 0.0)
    corner_c = (half_w, float(height))
    corner_d = (-half_w, float(height))

    def rot(vec, ang_cd):
        ang = math.radians(ang_cd / 100.0)
        c, s = math.cos(ang), math.sin(ang)
        return (vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c)

    # Start hook: travels dir_in, bends +hook_angle at D onto the first
    # leg D->A (direction (0, -1)). So dir_in = rotate((0,-1), -hook_angle).
    dir_in = rot((0.0, -1.0), -hook_angle)
    h1 = (corner_d[0] - dir_in[0] * hook_length, corner_d[1] - dir_in[1] * hook_length)
    # End hook: arrives along C->D (direction (-1, 0)), bends +hook_angle.
    dir_out = rot((-1.0, 0.0), hook_angle)
    h2 = (corner_d[0] + dir_out[0] * hook_length, corner_d[1] + dir_out[1] * hook_length)

    local = [h1, corner_d, corner_a, corner_b, corner_c, corner_d, h2]
    return [_plane_point(origin, plane, u, v) for (u, v) in local]
