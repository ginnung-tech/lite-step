"""
Pure math and geometry utilities for IFC generation.

No IFC/ifcopenshell dependencies. Functions for coordinate transforms,
rotation matrices, polygon normals, centroids, and boolean extensions.

Coordinate convention: DSL and IFC both use Z-up, right-handed.
DSL units are millimeters; IFC units are meters. The only transform
needed is mm→m scaling (handled by the caller, not here).
"""

import math
import numpy as np


def dsl_to_ifc_point(x: float, y: float, z: float) -> tuple:
    """
    Transform DSL coordinates to IFC coordinates.

    Both DSL and IFC use Z-up, right-handed coordinates.
    DSL: X=East, Y=North, Z=Up (mm)
    IFC: X=East, Y=North, Z=Up (meters)

    Identity mapping — no axis swap needed.
    Caller is responsible for mm→m scaling before calling this.
    """
    return (x, y, z)


def axis_rotation_matrix_3x3(axis: str, angle_rad: float):
    """
    Create 3x3 rotation matrix around specified axis.

    Args:
        axis: "x", "y", or "z" (lowercase)
        angle_rad: Rotation angle in radians

    Returns:
        3x3 numpy rotation matrix
    """
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)

    if axis == "x":
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=np.float64)
    elif axis == "y":
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=np.float64)
    elif axis == "z":
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=np.float64)
    else:
        return np.eye(3, dtype=np.float64)


def extend_cut_for_boolean(cut_min: list, cut_max: list, parent_min: tuple, parent_max: tuple) -> None:
    """
    Extend a cut solid's bounds where its faces are coplanar with the parent.

    When a boolean DIFFERENCE operand has a face exactly coplanar with the
    parent solid, the boolean engine may fail or produce Z-fighting artifacts.
    This function detects such faces and extends the cut by 10% of the parent
    dimension on that axis, ensuring the cut cleanly penetrates the surface.

    Modifies cut_min and cut_max lists in place.
    """
    for axis in range(3):
        parent_dim = parent_max[axis] - parent_min[axis]
        if parent_dim < 0.001:
            continue
        extend_amount = parent_dim * 0.1
        tolerance = max(0.005, parent_dim * 0.05)  # 5% or 5mm minimum

        # If cut's min face is near parent's min face, push it outward
        if abs(cut_min[axis] - parent_min[axis]) < tolerance:
            cut_min[axis] -= extend_amount
        # If cut's max face is near parent's max face, push it outward
        if abs(cut_max[axis] - parent_max[axis]) < tolerance:
            cut_max[axis] += extend_amount


def newell_normal(contour_3d: list) -> tuple:
    """
    Calculate the surface normal using Newell's Method.

    Newell's method computes the normal of a polygon from its vertices
    by summing cross products of consecutive edge pairs.

    Args:
        contour_3d: List of (x, y, z) tuples representing polygon vertices

    Returns:
        Normalized (nx, ny, nz) tuple representing the surface normal
    """
    n = len(contour_3d)
    nx = ny = nz = 0.0

    for i in range(n):
        curr = contour_3d[i]
        next_pt = contour_3d[(i + 1) % n]

        nx += (curr[1] - next_pt[1]) * (curr[2] + next_pt[2])
        ny += (curr[2] - next_pt[2]) * (curr[0] + next_pt[0])
        nz += (curr[0] - next_pt[0]) * (curr[1] + next_pt[1])

    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-10:
        return (0.0, 0.0, 1.0)

    return (nx / length, ny / length, nz / length)


def apply_orientation_rules(normal: tuple) -> tuple:
    """
    Apply orientation rules to ensure consistent extrusion direction.

    Rules (Z-up convention):
    - Horizontal/sloped surfaces (Z != 0): Normal should point up (Z > 0)
    - Vertical surfaces (Z ~ 0): Trust user's winding order (no flip)

    Args:
        normal: (nx, ny, nz) tuple from Newell's method

    Returns:
        Possibly flipped (nx, ny, nz) tuple
    """
    nx, ny, nz = normal
    epsilon = 0.001

    if abs(nz) > epsilon:
        if nz < 0:
            return (-nx, -ny, -nz)

    return normal


def calculate_centroid(contour_3d: list) -> tuple:
    """
    Calculate the centroid (average) of 3D polygon vertices.

    Args:
        contour_3d: List of (x, y, z) tuples

    Returns:
        (cx, cy, cz) centroid tuple
    """
    n = len(contour_3d)
    if n == 0:
        return (0.0, 0.0, 0.0)

    cx = sum(pt[0] for pt in contour_3d) / n
    cy = sum(pt[1] for pt in contour_3d) / n
    cz = sum(pt[2] for pt in contour_3d) / n

    return (cx, cy, cz)


#: A segment is treated as VERTICAL (its section frame ambiguous around the
#: path axis) past this |d.z|. Shared by the generator's emission and by the
#: displacement AABB so the two never disagree about which rule applies.
VERTICAL_SEGMENT_DZ = 0.99


def segment_axes(d, vertical_ref=(1.0, 0.0, 0.0), profile_rotation_cd: float = 0.0):
    """Section-plane axes for a path segment (pinned WS1 PR-E convention).

    THE single definition of the path-Sweep section frame. The generator emits
    geometry with it and the displacement pass bounds geometry with it; a second
    copy of this rule is how an AABB silently stops matching the solid it is
    supposed to bound, so there is exactly one.

    ``d`` = unit segment direction. Returns ``(x_axis, y_axis)`` with
    ``x = unit(up_ref x d)``, ``y = d x x``. ``up_ref`` is world Z for
    non-vertical segments (section X = horizontal "across", Y = "up", per spec)
    and ``vertical_ref`` for near-vertical ones (``|d.z| > VERTICAL_SEGMENT_DZ``).
    The default vertical_ref (world X) lands section X on the Y-ish axis;
    window-frame profiles pass the wall direction so section X is
    through-the-wall on vertical members too — keeping
    ``Material(profile_mm=(95, 45))`` "95 = through wall" on all four frame
    members even in rotated walls. Rect sections are symmetric about both axes
    so axis signs are geometry-neutral.

    ``profile_rotation_cd`` (centidegrees) rotates the frame about the path
    axis. Rotating the PLACEMENT axes rather than the section points is what
    keeps profile coordinates whole-mm exact.
    """
    d = np.asarray(d, dtype=np.float64)
    if abs(d[2]) > VERTICAL_SEGMENT_DZ:
        up_ref = np.array(vertical_ref, dtype=np.float64)
    else:
        up_ref = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(up_ref, d)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(d, x_axis)
    if profile_rotation_cd:
        ang = math.radians(profile_rotation_cd / 100.0)
        c_r, s_r = math.cos(ang), math.sin(ang)
        x_axis, y_axis = (x_axis * c_r + y_axis * s_r,
                          y_axis * c_r - x_axis * s_r)
    return x_axis, y_axis


# =============================================================================
# PATH MATH — DSL v1.5 primitives (WS1 PR-E: Rod, Turn, Bar, path Profile)
# =============================================================================
# Pure functions over point tuples. Unit-agnostic: callers pass mm ints at
# construction-validation time and meter floats at generation time — the
# math only relies on ratios and Euclidean geometry.

#: Maximum turn angle (radians) at a path joint. Beyond this the segments
#: nearly reverse and both the fillet tangent length (r*tan(theta/2)) and the
#: miter overshoot diverge — construction rejects such paths loudly instead
#: of emitting broken geometry.
MAX_JOINT_TURN_RAD = math.radians(160.0)


def _vec(a, b):
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _norm(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _unit(v):
    n = _norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector")
    return (v[0] / n, v[1] / n, v[2] / n)


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _add_scaled(p, d, t):
    return (p[0] + d[0] * t, p[1] + d[1] * t, p[2] + d[2] * t)


def joint_turn_angle(prev, vertex, nxt) -> float:
    """Turn angle (radians) at ``vertex``: 0 = collinear continuation,
    pi = full reversal."""
    u = _unit(_vec(prev, vertex))
    w = _unit(_vec(vertex, nxt))
    d = max(-1.0, min(1.0, _dot(u, w)))
    return math.acos(d)


def fillet_corner(prev, vertex, nxt, radius):
    """Fillet geometry at a path joint (circular arc tangent to both segments).

    Returns ``(p_start, p_mid, p_end, tangent_len, turn_angle)`` where
    ``p_start``/``p_end`` are the tangent points on the incoming/outgoing
    segments, ``p_mid`` is the arc midpoint (for 3-point IfcArcIndex arcs),
    ``tangent_len`` is the trimmed length ``r*tan(theta/2)`` on each side and
    ``turn_angle`` the joint's turn angle in radians. Returns ``None`` for a
    (near-)collinear joint — no arc needed.
    """
    u = _unit(_vec(prev, vertex))
    w = _unit(_vec(vertex, nxt))
    d = max(-1.0, min(1.0, _dot(u, w)))
    theta = math.acos(d)
    if theta < 1e-6:
        return None
    t = radius * math.tan(theta / 2.0)
    p_start = _add_scaled(vertex, u, -t)
    p_end = _add_scaled(vertex, w, t)
    # Internal bisector from the vertex: between (-u) and (+w).
    bis = _unit((-u[0] + w[0], -u[1] + w[1], -u[2] + w[2]))
    # |V->C| = r / sin(phi/2) with phi = pi - theta the interior angle,
    # so sin(phi/2) = cos(theta/2); arc midpoint sits r short of the center.
    dist_vc = radius / math.cos(theta / 2.0)
    p_mid = _add_scaled(vertex, bis, dist_vc - radius)
    return (p_start, p_mid, p_end, t, theta)


def validate_fillet_path(points, fillet_radius, label: str) -> list:
    """Construction-time checks shared by Rod / Bar / path Profile.

    ``points`` are (x, y, z) tuples; ``fillet_radius`` >= 0 in the same unit.
    Returns a list of error strings (empty = valid):

    * fewer than 2 points, or consecutive duplicate points;
    * a joint turning more than ``MAX_JOINT_TURN_RAD`` (near-reversal);
    * with ``fillet_radius > 0``: tangent trims ``r*tan(theta/2)`` that do
      not fit inside the adjacent segments (adjacent fillets may not overlap).
    """
    errors: list = []
    if len(points) < 2:
        errors.append(f"{label}: path needs at least 2 points (got {len(points)})")
        return errors
    seg_lens = []
    for i in range(len(points) - 1):
        seg = _vec(points[i], points[i + 1])
        seg_len = _norm(seg)
        if seg_len < 1e-9:
            errors.append(
                f"{label}: path points {i} and {i + 1} coincide — "
                f"degenerate zero-length segment"
            )
            return errors
        seg_lens.append(seg_len)
    tangents = [0.0] * len(seg_lens)  # trim already claimed at segment start
    for j in range(1, len(points) - 1):
        theta = joint_turn_angle(points[j - 1], points[j], points[j + 1])
        if theta > MAX_JOINT_TURN_RAD:
            errors.append(
                f"{label}: path joint {j} turns {math.degrees(theta):.1f} deg "
                f"— segments nearly reverse; maximum supported turn is "
                f"{math.degrees(MAX_JOINT_TURN_RAD):.0f} deg"
            )
            continue
        if fillet_radius and fillet_radius > 0 and theta > 1e-6:
            t = fillet_radius * math.tan(theta / 2.0)
            # incoming segment j-1: start-claimed + this joint's trim
            if tangents[j - 1] + t > seg_lens[j - 1] + 1e-9:
                errors.append(
                    f"{label}: fillet_radius={fillet_radius} does not fit at "
                    f"path joint {j} — needs {t:.1f} tangent length on a "
                    f"segment of {seg_lens[j - 1]:.1f} (adjacent fillets "
                    f"may not overlap)"
                )
            if t > seg_lens[j] + 1e-9:
                errors.append(
                    f"{label}: fillet_radius={fillet_radius} does not fit at "
                    f"path joint {j} — needs {t:.1f} tangent length on a "
                    f"segment of {seg_lens[j]:.1f}"
                )
            tangents[j] = t
    return errors


def path_length_with_fillets(points, fillet_radius) -> float:
    """True centerline length of a filleted path.

    Straight polyline length, with each filleted joint's two tangent trims
    (2 * r*tan(theta/2)) replaced by the arc length (r * theta). With
    ``fillet_radius == 0`` this is the plain polyline length. Used for the
    Bar derived Length/Weight quantities.
    """
    total = 0.0
    for i in range(len(points) - 1):
        total += _norm(_vec(points[i], points[i + 1]))
    if fillet_radius and fillet_radius > 0:
        for j in range(1, len(points) - 1):
            theta = joint_turn_angle(points[j - 1], points[j], points[j + 1])
            if theta < 1e-6:
                continue
            t = fillet_radius * math.tan(theta / 2.0)
            total += fillet_radius * theta - 2.0 * t
    return total


def facet_fillet_path(points, fillet_radius, max_step_rad=math.radians(15.0)):
    """Replace each filleted joint of an OPEN path with arc facet points.

    Used by the path-form Profile: arbitrary cross-sections cannot ride an
    IfcSweptDiskSolid arc, so rounded corners are approximated by faceting
    the corner arc into chords no wider than ``max_step_rad`` and mitering
    every sub-joint. Returns a new list of (x, y, z) tuples.
    """
    if not fillet_radius or fillet_radius <= 0 or len(points) < 3:
        return list(points)
    out = [points[0]]
    for j in range(1, len(points) - 1):
        fc = fillet_corner(points[j - 1], points[j], points[j + 1], fillet_radius)
        if fc is None:
            out.append(points[j])
            continue
        p_start, p_mid, p_end, _t, theta = fc
        steps = max(2, int(math.ceil(theta / max_step_rad)))
        # Sample the arc through p_start -> p_mid -> p_end by rotating the
        # start point about the arc center.
        u = _unit(_vec(points[j - 1], points[j]))
        w = _unit(_vec(points[j], points[j + 1]))
        bis = _unit((-u[0] + w[0], -u[1] + w[1], -u[2] + w[2]))
        center = _add_scaled(points[j], bis, fillet_radius / math.cos(theta / 2.0))
        axis = _unit(_cross(u, w))  # rotation axis of the turn
        for k in range(steps + 1):
            ang = theta * k / steps
            # Rodrigues rotation of (p_start - center) about axis by ang
            v = _vec(center, p_start)
            c, s = math.cos(ang), math.sin(ang)
            kxv = _cross(axis, v)
            kdv = _dot(axis, v)
            rot = (
                v[0] * c + kxv[0] * s + axis[0] * kdv * (1 - c),
                v[1] * c + kxv[1] * s + axis[1] * kdv * (1 - c),
                v[2] * c + kxv[2] * s + axis[2] * kdv * (1 - c),
            )
            out.append((center[0] + rot[0], center[1] + rot[1], center[2] + rot[2]))
    out.append(points[-1])
    return out


# ---------------------------------------------------------------------------
# Sheet metal — closing an OPEN face line into an extrudable profile
# ---------------------------------------------------------------------------
#
# A folded sheet part (sålbænk, flashing, gutter, capping, cold-formed section)
# is specified the way the fabricator specifies it: the OUTER/visible face as an
# open polyline, plus a stock thickness. A line has no area, so it cannot be
# extruded — it has to be closed into a loop first. These three helpers are that
# operation, kept here (with the other profile/path math) so both the model
# layer and any future consumer read the SAME implementation.


def offset_open_polyline(points, thickness):
    """Offset an OPEN polyline by ``thickness`` onto its material side.

    ``points`` are ``(x, y)`` tuples in profile coordinates; the result has the
    same length and the same ordering.

    **The material side is decided by the direction of travel.** For a segment
    direction ``d = (dx, dy)`` the material lies along ``n = (-dy, dx)`` — i.e.
    90 degrees to the LEFT of travel. Reversing the point order therefore puts
    the material on the other side of the same line.

    At each interior vertex the offset follows the angle BISECTOR, scaled by
    ``1 / (u · n)`` so both faces end up exactly ``thickness`` apart (a plain
    averaged normal pinches the corners — the miter correction is what keeps a
    folded part its nominal gauge through every bend).
    """
    def seg_n(p, q):
        dx, dy = q[0] - p[0], q[1] - p[1]
        L = math.hypot(dx, dy) or 1.0
        return (-dy / L, dx / L)

    out = []
    for i, p in enumerate(points):
        ns = []
        if i > 0:
            ns.append(seg_n(points[i - 1], p))
        if i < len(points) - 1:
            ns.append(seg_n(p, points[i + 1]))
        bx = sum(n[0] for n in ns) / len(ns)
        by = sum(n[1] for n in ns) / len(ns)
        L = math.hypot(bx, by)
        if L < 1e-9:
            bx, by, L = ns[0][0], ns[0][1], 1.0
        ux, uy = bx / L, by / L
        dot = ux * ns[0][0] + uy * ns[0][1]          # miter correction
        k = thickness / (dot if abs(dot) > 1e-6 else 1.0)
        out.append((p[0] + ux * k, p[1] + uy * k))
    return out


def dedupe_consecutive_2d(points, tol=1e-9, wrap=False):
    """Drop consecutive duplicate ``(x, y)`` points, optionally wrap-around.

    A finely tessellated bend collapses several of its arc points onto one
    another once profile coordinates are rounded to int millimeters — normal,
    and not an error. What IS an error is a profile that has fewer than three
    distinct points left, and the caller checks that.

    ``wrap=True`` additionally drops the last point when it duplicates the
    first (a CLOSED loop repeats no vertex).
    """
    out = []
    for p in points:
        if out and abs(p[0] - out[-1][0]) <= tol and abs(p[1] - out[-1][1]) <= tol:
            continue
        out.append(p)
    if wrap and len(out) > 1:
        if (abs(out[-1][0] - out[0][0]) <= tol
                and abs(out[-1][1] - out[0][1]) <= tol):
            out.pop()
    return out


#: Weld tolerance as a fraction of the mesh's own longest span. See
#: ``weld_coincident_vertices`` for why this number has seven orders of
#: headroom on both sides.
_WELD_EPS_REL = 1e-9
_WELD_EPS_MIN = 1e-12


def weld_coincident_vertices(verts, faces, eps: float | None = None):
    """Merge vertices that are the SAME POINT but distinct INDICES.

    Returns ``(verts Nx3 float64, faces Mx3 int64)`` — the identical surface
    with a single index per geometric position, and any triangle that
    collapsed to a line or a point dropped.

    **Why this exists.** The OCC kernel behind
    ``ifcopenshell.geom.create_shape`` evaluates each FACE independently, so a
    seam shared by two faces comes back as two vertex entries whose float64
    coordinates differ in the last few bits. Measured on the bridge-deck
    ``Sweep``: four such pairs, **1.4e-14 m apart** (14 femtometres), against a
    smallest legitimate feature distance of **1e-3 m** in the same mesh.
    ifcopenshell's own ``weld-vertices`` setting is already ON by default and
    collapses 56 raw entries to 20 — but its threshold does not reach 1e-14, so
    the seam survives as duplicate indices.

    Geometrically that mesh is closed. **Topologically it is not**: the two
    triangles either side of a doubled seam reference different indices, so the
    shared edge appears with multiplicity 1 twice instead of 2, and every
    watertightness test fails. ``manifold3d`` then rejects the mesh with
    ``Error.NotManifold`` — WITHOUT raising — and propagates that error state
    through every subsequent boolean, which is how a bridge deck that never
    touches the ground erased an entire terrain. See ``_carve_mesh_host``.

    ``eps`` defaults to ``max(span) * 1e-9`` floored at ``1e-12``. It is a
    repair of a representation defect, never a simplification: at this
    tolerance no vertex the author placed can merge with another.

    **Why 1e-9 and not wider.** Measured distance spectra of the ribbon,
    as fractions of its own longest span:

    ======================  ==================  ====================
    tessellated at          duplicate seam gap  smallest real feature
    ======================  ==================  ====================
    meters (post-normalize)  7e-17  (1.4e-14 m)  5e-6   (1 mm)
    millimeters (authoring)  5e-9   (1 um)       2.5e-3 (500 mm)
    ======================  ==================  ====================

    The two rows disagree because OCC's internal confusion tolerances are
    ABSOLUTE while the mesher deflection we set is relative, so the same solid
    tessellates to a different artefact size at each scale. No single relative
    tolerance has a large margin on both sides of both rows. 1e-9 is chosen for
    the METER row — 1.4e8x above its round-off, 5000x below its smallest real
    feature — because that is the only row production reaches: the sole
    non-test caller is ``displacement._occupant_mesh``, and displacement runs
    after ``normalize_project_to_meters``. Widening to 1e-8 would close the
    millimeter row too, but with only a 2x margin over its duplicate gap, which
    is not a margin.

    Whatever this misses is caught LOUDLY one layer up: ``_carve_mesh_host``
    refuses an operand ``manifold3d`` rejects and raises, rather than letting
    the error state erase the host. Repair first, loud failure second — never
    silence, which is what was.

    Cell hashing, not plain quantisation. Rounding coordinates into buckets is
    six lines and would work here, but two points a hair either side of a
    bucket boundary stay unmerged however close they are — reintroducing this
    exact bug, non-deterministically, for a subset of geometry. So each vertex
    checks its own cell and all 26 neighbours, and the FIRST vertex (in index
    order) to claim a cell owns it: boundary-free and deterministic, which
    matters because the committed corpus IFC is compared byte for byte.
    """
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0:
        return v, f

    if eps is None:
        span = float(np.max(v.max(axis=0) - v.min(axis=0))) if len(v) > 1 else 0.0
        eps = max(span * _WELD_EPS_REL, _WELD_EPS_MIN)

    cells = np.floor(v / eps).astype(np.int64)
    owner: dict[tuple[int, int, int], list[int]] = {}
    remap = np.empty(len(v), dtype=np.int64)
    keep_order: list[int] = []

    neighbourhood = [(dx, dy, dz)
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    for i in range(len(v)):
        cx, cy, cz = int(cells[i, 0]), int(cells[i, 1]), int(cells[i, 2])
        hit = -1
        for dx, dy, dz in neighbourhood:
            for j in owner.get((cx + dx, cy + dy, cz + dz), ()):
                if np.all(np.abs(v[j] - v[i]) <= eps):
                    hit = j
                    break
            if hit >= 0:
                break
        if hit >= 0:
            remap[i] = remap[hit]
        else:
            owner.setdefault((cx, cy, cz), []).append(i)
            remap[i] = len(keep_order)
            keep_order.append(i)

    if len(keep_order) == len(v):
        return v, f                      # nothing coincident — untouched

    welded_faces = remap[f]
    non_degenerate = (
        (welded_faces[:, 0] != welded_faces[:, 1])
        & (welded_faces[:, 1] != welded_faces[:, 2])
        & (welded_faces[:, 0] != welded_faces[:, 2])
    )
    return v[np.asarray(keep_order, dtype=np.int64)], welded_faces[non_degenerate]


def mesh_boundary_edge_count(faces) -> int:
    """Number of edges NOT shared by exactly two triangles.

    Zero on a closed, orientable, index-consistent surface — the precondition
    every mesh boolean has. Non-zero is the ``manifold3d``
    ``Error.NotManifold`` signature, and it is what ``weld_coincident_vertices``
    exists to drive to zero. Kept here rather than in a test
    helper so production diagnostics and tests read the SAME number.
    """
    from collections import Counter

    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    counts: Counter = Counter()
    for a, b, c in f:
        for e in ((a, b), (b, c), (c, a)):
            counts[(int(min(e)), int(max(e)))] += 1
    return sum(1 for n in counts.values() if n != 2)


def mesh_edge_defects(faces) -> tuple[int, int]:
    """``(hole_edges, non_manifold_edges)`` — the two ways a surface fails.

    ``mesh_boundary_edge_count`` above sums BOTH into one number, which is
    right for "is this closed?" and wrong for "why isn't it?". A diagnostic
    that reports the sum under the name *boundary edges* reads as a hole even
    when every offending edge is shared by four faces, and that is not a
    cosmetic difference — the two have opposite repairs. It sent the diagnosis the wrong way: the reporter saw "5 boundary edges", looked for a
    hole, and the mesh actually had zero holes and five non-manifold edges
    from a doubled interior wall (see ``drop_coincident_faces``).

    * ``hole_edges`` — traversed by ONE face. The surface is open.
    * ``non_manifold_edges`` — traversed by THREE OR MORE. Usually a
      coincident interior wall; occasionally three shells meeting on an edge.
    """
    from collections import Counter

    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    counts: Counter = Counter()
    for a, b, c in f:
        for e in ((a, b), (b, c), (c, a)):
            counts[(int(min(e)), int(max(e)))] += 1
    holes = sum(1 for n in counts.values() if n == 1)
    non_manifold = sum(1 for n in counts.values() if n > 2)
    return holes, non_manifold


def _orient2d(a, b, c) -> float:
    """Twice the signed area of triangle abc — >0 left turn, <0 right turn."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _within_bbox(a, b, c, eps: float) -> bool:
    """``c`` (already known collinear with ab) lies inside ab's bounding box."""
    return (min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
            and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps)


def segments_intersect_2d(p1, p2, p3, p4, eps: float = 1e-9) -> bool:
    """True when segment ``p1p2`` and segment ``p3p4`` cross or touch.

    Standard four-orientation test with the collinear-overlap cases handled
    explicitly, so a segment that merely GRAZES another (endpoint landing on
    an edge, or a collinear overlap) still reports — for a profile loop those
    are just as fatal as a clean crossing.
    """
    d1 = _orient2d(p3, p4, p1)
    d2 = _orient2d(p3, p4, p2)
    d3 = _orient2d(p1, p2, p3)
    d4 = _orient2d(p1, p2, p4)
    if (((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps))
            and ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps))):
        return True
    if abs(d1) <= eps and _within_bbox(p3, p4, p1, eps):
        return True
    if abs(d2) <= eps and _within_bbox(p3, p4, p2, eps):
        return True
    if abs(d3) <= eps and _within_bbox(p1, p2, p3, eps):
        return True
    if abs(d4) <= eps and _within_bbox(p1, p2, p4, eps):
        return True
    return False


def find_loop_self_intersection(loop, eps: float = 1e-9):
    """First self-intersecting segment pair of a CLOSED loop, or ``None``.

    ``loop`` is the vertex list of a closed polygon with NO repeated closing
    point; segment ``i`` runs from ``loop[i]`` to ``loop[i + 1 mod n]``.
    Adjacent pairs (which legitimately share a vertex), the wrap pair
    ``(0, n - 1)``, and zero-length segments are excluded. Returns ``(i, j)``
    with ``i < j``.
    """
    n = len(loop)
    if n < 4:
        return None

    def seg(i):
        return loop[i], loop[(i + 1) % n]

    for i in range(n):
        a1, a2 = seg(i)
        if abs(a1[0] - a2[0]) <= eps and abs(a1[1] - a2[1]) <= eps:
            continue
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue          # wrap-adjacent — shares loop[0]
            b1, b2 = seg(j)
            if abs(b1[0] - b2[0]) <= eps and abs(b1[1] - b2[1]) <= eps:
                continue
            if segments_intersect_2d(a1, a2, b1, b2, eps):
                return (i, j)
    return None


def _signed_volume(v: "np.ndarray", f: "np.ndarray") -> float:
    """Six times the signed volume of a closed triangle soup.

    Positive when the faces wind counter-clockwise seen from outside, which
    is the outward convention ``manifold3d`` and IFC both use.
    """
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum())


def drop_coincident_faces(verts, faces):
    """Remove interior walls — triangles that occupy the same place twice.

    Returns ``(verts, faces)`` with the SAME vertices; only faces are dropped.

    **The third defect from the same root cause**, after
    ``weld_coincident_vertices`` and ``orient_faces_consistently``.
    ``ifcopenshell.geom.create_shape`` evaluates each OCC face independently,
    and a swept solid built from abutting segments emits the shared internal
    cap ONCE PER SEGMENT. Before welding those are two triangle pairs on
    distinct vertex indices, so nothing looks wrong; welding merges the
    vertices and leaves two triangles indexing the SAME three corners. Every
    edge of that cap is then traversed by four faces instead of two.

    That is a NON-MANIFOLD edge, not a hole, and the distinction is the whole
    reason this function exists: ``orient_faces_consistently`` says so in its
    own docstring — non-manifold input "has no consistent orientation to
    find". Running it against the doubled surface propagates winding across a
    wall that should not be there, so the mesh comes back both non-manifold
    AND mis-wound. **This must therefore run BETWEEN the weld and the
    orient**, and the order is load-bearing: measured on the chord that found
    it, dedup alone leaves ``Error.NotManifold``; dedup then re-orient gives
    ``Error.NoError``.

    A triangle appearing an EVEN number of times is an interior wall and every
    copy is dropped — keeping one would leave a membrane sealing the solid's
    own interior. An ODD count keeps exactly one, so a surplus pair cancels
    without deleting a face the surface genuinely needs. Winding is ignored
    (the key is the sorted index triple) because the two copies of an interior
    wall face opposite ways by construction — that is what makes them
    cancel.

    Measured on ``bridge-deck-truss``'s ``deck_chord_top_r``: 124 faces with 5
    non-manifold edges and 0 boundary edges, ``Error.NotManifold``, volume 0.
    Dropping the 4 duplicated cap triangles and re-orienting gives 120 faces,
    0 non-manifold edges, ``Error.NoError``, volume **38.988962 m³** against
    an analytic 0.4 x 0.4 x 243.6 m path = 38.98 m³ — so the tessellation was
    always right and only the doubled cap was wrong.
    """
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return v, f

    # Winding-insensitive identity: the two copies of an interior wall are
    # wound oppositely, so only the SET of corners identifies them as a pair.
    keys = np.sort(f, axis=1)
    _uniq, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)

    # Keep the first occurrence of each odd-multiplicity triangle; drop every
    # copy of an even-multiplicity one.
    order = np.argsort(inverse, kind="stable")
    is_first = np.zeros(len(f), dtype=bool)
    sorted_inv = inverse[order]
    firsts = np.ones(len(order), dtype=bool)
    firsts[1:] = sorted_inv[1:] != sorted_inv[:-1]
    is_first[order[firsts]] = True

    keep = is_first & (counts[inverse] % 2 == 1)
    return v, f[keep]


def orient_faces_consistently(verts, faces):
    """Make every triangle agree with its neighbours on which side is out.

    Returns ``(verts, faces)`` with the SAME vertices and the SAME set of
    faces — only the vertex ORDER within a triangle changes.

    **Why this exists.** ``weld_coincident_vertices`` above repairs the duplicate-index defect and nothing else. It has a sibling defect from the
    same root cause: ``ifcopenshell.geom.create_shape`` evaluates each OCC
    FACE independently, so nothing makes one face's winding agree with the
    next one's. A tessellated ``Bar`` comes back topologically perfect —
    Euler characteristic 2, zero boundary edges, zero non-manifold edges, no
    degenerate triangles, one index per position — and still fails, because
    its low end cap is wound backwards relative to the sides. In directed-edge
    terms: 32 directed edges appear twice in the SAME direction and 32 have no
    opposite at all.

    ``manifold3d`` requires consistent orientation, so it answers
    ``Error.NotManifold`` — **without raising** — and then propagates that
    error state through every boolean it touches. Identical failure signature
    , and identical blast radius: a carve operand silently voids the
    host it was meant to notch.

    Measured on the bar that found this: re-orienting by BFS over face
    adjacency, changing nothing but triangle vertex order, turns
    ``Error.NotManifold`` into ``Error.NoError`` and yields volume
    0.00133724 m³ against an analytic ``pi*r^2*L`` of 0.00134586. The 0.64%
    deficit is exactly a 32-gon's area ratio (0.99359) — i.e. the tessellation
    was always right and only the winding was wrong.

    Method: BFS the face-adjacency graph, flipping any neighbour that shares
    an edge in the same direction (two correctly-oriented neighbours traverse
    their shared edge in OPPOSITE directions); then flip a whole connected
    component if its signed volume comes out negative, so the result is
    outward-facing rather than merely self-consistent. Components are handled
    independently — a multi-shell solid orients each shell on its own.

    Non-manifold input (an edge shared by more than two faces) has no
    consistent orientation to find; BFS still terminates and the mesh is
    returned as improved as it can be, for the layer above to reject loudly.
    """
    from collections import defaultdict, deque

    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3).copy()
    if len(f) == 0:
        return v, f

    edge_faces: dict = defaultdict(list)
    for i, tri in enumerate(f):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, w in ((a, b), (b, c), (c, a)):
            edge_faces[(u, w) if u < w else (w, u)].append(i)

    seen = np.zeros(len(f), dtype=bool)
    for start in range(len(f)):
        if seen[start]:
            continue
        seen[start] = True
        component = [start]
        queue = deque([start])
        while queue:
            i = queue.popleft()
            a, b, c = int(f[i][0]), int(f[i][1]), int(f[i][2])
            for u, w in ((a, b), (b, c), (c, a)):
                for j in edge_faces[(u, w) if u < w else (w, u)]:
                    if j == i or seen[j]:
                        continue
                    ja, jb, jc = int(f[j][0]), int(f[j][1]), int(f[j][2])
                    # Same DIRECTION on the shared edge means j disagrees.
                    if (u, w) in ((ja, jb), (jb, jc), (jc, ja)):
                        f[j] = f[j][::-1]
                    seen[j] = True
                    component.append(j)
                    queue.append(j)
        idx = np.asarray(component, dtype=np.int64)
        if _signed_volume(v, f[idx]) < 0.0:
            f[idx] = f[idx][:, ::-1]

    return v, f
