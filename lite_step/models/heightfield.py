"""Heightmap → watertight prism tessellation (the Mesh heightmap mode).

A heightmap is a compact, first-class way to specify a ``Mesh``: a
``rows × cols`` grid of top-surface heights plus a positive depth below the
lowest point. The compiler owns the representation plumbing — grid → vertex
placement, flat bottom, four perimeter skirts, outward-facing 0-based
windings — so a model script states the *design* (the grid of heights) and
never reproduces stitching boilerplate.

Semantics (kept deliberately dumb — design conventions live in skills):

* **Heights are literal.** No auto-shift: ``heightmap[r][c]`` becomes the top
  vertex z exactly as authored (mm at script time). The site-block skill's
  "lowest point at z=0" carve datum is *that skill's* rule, applied by the
  author, not baked in here.
* **Bottom is one flat plane** at ``min(heights) - depth``. ``depth`` is a
  positive mm distance below the LOWEST top vertex, so the prism always has
  positive thickness everywhere.
* **Grid orientation:** ``heightmap[0][0]`` sits at ``corner_min`` (the SW
  corner: min x, min y); rows advance +y (south → north), columns advance +x
  (west → east); ``heightmap[-1][-1]`` sits at ``corner_max``. Interpolated
  grid x/y coordinates are rounded to int mm (they are *computed*, not
  authored — same convention as the ``I()`` helper).
* **Watertight by construction:** top grid + flat bottom grid + 4 skirts,
  outward normals, 0-based triangles. The winding is the proven site-block
  stitching, ported verbatim.

The tessellation runs at Mesh **construction** (``model_validator``), so every
downstream consumer — normalizer, the emitter, the displacement engine,
validators — sees an ordinary vertices+faces mesh. The heightmap fields are
*retained* on the model (compact ``output.py``, meaningful edit-mode diffs,
and the substrate for future surface queries).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .primitives import Point, Point2D


def _validate_grid(heightmap: List[List[float]],
                   corner_min: Point2D, corner_max: Point2D) -> Tuple[int, int]:
    """Shared grid validation → (rows, cols). At least 2×2, rectangular,
    ``corner_min`` strictly below ``corner_max`` on both axes."""
    rows = len(heightmap)
    cols = len(heightmap[0]) if rows else 0
    if rows < 2 or cols < 2:
        raise ValueError(
            f"heightmap must be at least 2×2 (got {rows}×{cols})"
        )
    if any(len(row) != cols for row in heightmap):
        raise ValueError(
            "heightmap must be rectangular — every row needs "
            f"{cols} entries (row lengths: {[len(r) for r in heightmap]})"
        )
    if not (corner_min.x < corner_max.x and corner_min.y < corner_max.y):
        raise ValueError(
            "corner_min must be strictly below corner_max on both axes "
            f"(got corner_min=({corner_min.x}, {corner_min.y}), "
            f"corner_max=({corner_max.x}, {corner_max.y}))"
        )
    return rows, cols


def surface_z_at(
    heightmap: List[List[float]],
    corner_min: Point2D,
    corner_max: Point2D,
    x: float,
    y: float,
) -> float:
    """Bilinear-sample the heightfield's TOP surface at ``(x, y)``.

    Stage 2 of the heightmap design: the retained grid is a FUNCTION
    z = f(x, y), so terrain-conforming construction never re-implements
    the sampler skill-side (the old ``_terrain_z_at`` lockstep problem —
    a copy of the grid convention that had to track the builder exactly).
    Points OUTSIDE the rectangle clamp to the nearest edge (same behavior
    the skill sampler had). Returns a float — a computed value; callers in
    the mm authoring domain wrap it in ``I()`` per the convention. Reads
    the AUTHORED grid — declarative surface, independent of later carves.
    """
    rows, cols = _validate_grid(heightmap, corner_min, corner_max)
    fx = (x - corner_min.x) / (corner_max.x - corner_min.x) * (cols - 1)
    fy = (y - corner_min.y) / (corner_max.y - corner_min.y) * (rows - 1)
    fc = min(max(fx, 0.0), cols - 1)
    fr = min(max(fy, 0.0), rows - 1)
    c0 = int(fc); c1 = min(c0 + 1, cols - 1); tc = fc - c0
    r0 = int(fr); r1 = min(r0 + 1, rows - 1); tr = fr - r0
    z0 = heightmap[r0][c0] * (1 - tc) + heightmap[r0][c1] * tc
    z1 = heightmap[r1][c0] * (1 - tc) + heightmap[r1][c1] * tc
    return z0 * (1 - tr) + z1 * tr


def sample_grid(
    heightmap: List[List[float]],
    src_corner_min: Point2D,
    src_corner_max: Point2D,
    corner_min: Point2D,
    corner_max: Point2D,
    rows: int,
    cols: int,
    round: bool = False,
) -> List[List[float]]:
    """Sample a ``rows × cols`` heightmap over ``corner_min``→``corner_max``
    from a source heightfield — the terrain-CONFORMING top in one call (the
    stenkant patch grids: gravel following grade). Heights are floats;
    ``round=True`` rounds to int mm (the ``I()`` convention) so the result
    feeds straight back into ``Mesh(heightmap=...)`` under strict int-mm.
    """
    import builtins

    if rows < 2 or cols < 2:
        raise ValueError(f"heightmap_at needs rows/cols >= 2 (got {rows}×{cols})")
    out: List[List[float]] = []
    for r in range(rows):
        y = corner_min.y + (corner_max.y - corner_min.y) * r / (rows - 1)
        row: List[float] = []
        for c in range(cols):
            x = corner_min.x + (corner_max.x - corner_min.x) * c / (cols - 1)
            z = surface_z_at(heightmap, src_corner_min, src_corner_max, x, y)
            row.append(int(builtins.round(z)) if round else z)
        out.append(row)
    return out


def _as_arrays(vertices, faces):
    """Accept Point-lists (model fields) or ndarrays (tessellator output)."""
    import numpy as np

    if isinstance(vertices, np.ndarray):
        V = vertices.astype(np.float64)
    else:
        V = np.array([[p.x, p.y, p.z] for p in vertices], dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    return V, F


def mesh_raycast(vertices, faces, origin, direction):
    """General ray query (Möller–Trumbore, vectorized): the NEAREST
    intersection of the ray ``origin + t*direction`` (t ≥ 0) with the mesh's
    triangles, or ``None`` on a miss.

    ``direction`` is the full 3D measurement vector — cast down a slope's
    normal, horizontally against a face, whatever the measurement needs.
    ``surface_z_at`` is the vertical special case. Queries are in the mesh's
    authoring frame (``placement=`` applies later).
    """
    import numpy as np

    o = np.array(origin, dtype=np.float64)
    d = np.array(direction, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-12:
        raise ValueError("raycast direction must be a non-zero vector")
    d = d / n

    V, F = _as_arrays(vertices, faces)
    a = V[F[:, 0]]
    e1 = V[F[:, 1]] - a
    e2 = V[F[:, 2]] - a
    pvec = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > 1e-12                      # skip parallel/degenerate
    if not ok.any():
        return None
    a, e1, e2, pvec, det = a[ok], e1[ok], e2[ok], pvec[ok], det[ok]
    tvec = o - a
    u = np.einsum("ij,ij->i", tvec, pvec) / det
    qvec = np.cross(tvec, e1)
    v = np.einsum("j,ij->i", d, qvec) / det
    t = np.einsum("ij,ij->i", e2, qvec) / det
    eps = 1e-9
    hit = (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t >= -eps)
    if not hit.any():
        return None
    t_min = float(t[hit].min())
    return tuple(o + t_min * d)


def mesh_surface_z_at(vertices, faces, x: float, y: float):
    """The GENERAL surface query: the highest intersection of the vertical
    line through ``(x, y)`` with the mesh's triangles, or ``None`` when the
    line misses the mesh footprint entirely.

    Works on ANY triangle mesh — explicit vertices, baked-in rotations,
    scanned terrain, carved results — because it reads the actual geometry,
    not an authoring grid. Vertical faces (the prism skirts) project to a
    degenerate 2D triangle and are skipped; for a closed prism the max picks
    the top face over the bottom. Queries are in the mesh's AUTHORING frame
    (``placement=`` is applied later in the pipeline, not here).
    """
    import numpy as np

    V, F = _as_arrays(vertices, faces)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    d = b[:, :2] - a[:, :2]
    e = c[:, :2] - a[:, :2]
    det = d[:, 0] * e[:, 1] - d[:, 1] * e[:, 0]
    ok = np.abs(det) > 1e-12                      # skip vertical/degenerate tris
    if not ok.any():
        return None
    a, b, c, d, e, det = a[ok], b[ok], c[ok], d[ok], e[ok], det[ok]
    qx, qy = x - a[:, 0], y - a[:, 1]
    u = (qx * e[:, 1] - qy * e[:, 0]) / det
    v = (d[:, 0] * qy - d[:, 1] * qx) / det
    eps = 1e-9
    inside = (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps)
    if not inside.any():
        return None
    z = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
    return float(z[inside].max())


def tessellate_heightmap(
    heightmap: List[List[float]],
    depth: Optional[float],
    corner_min: Point2D,
    corner_max: Point2D,
    bottom_z: Optional[float] = None,
) -> Tuple[List[Point], List[Tuple[int, int, int]]]:
    """Tessellate a heightmap into a closed prism (vertices, faces).

    The flat bottom plane comes from exactly ONE of:

    * ``depth`` — a positive mm distance below the LOWEST height (relative);
    * ``bottom_z`` — an ABSOLUTE plane, strictly below the lowest height.
      This is how sibling prisms share one level bottom (the stenkant's four
      patches) without per-patch depth arithmetic.

    Raises ``ValueError`` on an invalid grid — at least 2×2, rectangular,
    ``corner_min`` strictly below ``corner_max`` on both axes.
    Strict int-mm on heights/depth/bottom_z is enforced at the ``Mesh`` FIELD
    boundary (``mode="before"`` validators — pydantic's float-typed storage
    coerces ints before any after-validator could check them), so by the time
    this runs the values are spec-conforming and float-stored; Points are
    built with ``model_construct`` accordingly (the ``Point.__add__``
    rationale).
    """
    rows, cols = _validate_grid(heightmap, corner_min, corner_max)
    lowest = min(h for row in heightmap for h in row)
    if (depth is None) == (bottom_z is None):
        raise ValueError(
            "heightmap mode needs exactly ONE of depth= (relative, below the "
            "lowest height) or bottom_z= (absolute level plane)."
        )
    if depth is not None and depth <= 0:
        raise ValueError(
            f"depth must be a positive mm distance below the lowest "
            f"height (got {depth!r})"
        )
    if bottom_z is not None and bottom_z >= lowest:
        raise ValueError(
            f"bottom_z must lie strictly below the lowest height "
            f"(bottom_z={bottom_z!r}, lowest height={lowest!r}) — the prism "
            f"needs positive thickness everywhere."
        )

    x0, y0 = corner_min.x, corner_min.y
    x1, y1 = corner_max.x, corner_max.y

    # Top surface: heights literal; interpolated grid x/y rounded to int mm
    # (computed values, not authored — the I() convention). model_construct:
    # the values passed field-boundary validation already; re-validating the
    # float-stored heights under strict would wrongly reject them.
    top: List[Point] = []
    for r in range(rows):
        for c in range(cols):
            x = int(round(x0 + (x1 - x0) * c / (cols - 1)))
            y = int(round(y0 + (y1 - y0) * r / (rows - 1)))
            top.append(Point.model_construct(x=x, y=y, z=heightmap[r][c]))

    # One flat bottom plane: absolute (bottom_z) or `depth` below the lowest.
    bottom_z = bottom_z if bottom_z is not None else min(p.z for p in top) - depth
    bottom = [
        Point.model_construct(x=p.x, y=p.y, z=bottom_z) for p in top
    ]

    n = rows * cols
    verts = top + bottom  # top 0..n-1, bottom n..2n-1

    def t(r: int, c: int) -> int:
        return r * cols + c

    def b(r: int, c: int) -> int:
        return n + r * cols + c

    faces: List[Tuple[int, int, int]] = []
    # Top surface (outward normal +Z): two triangles per grid cell.
    for r in range(rows - 1):
        for c in range(cols - 1):
            faces.append((t(r, c), t(r, c + 1), t(r + 1, c + 1)))
            faces.append((t(r, c), t(r + 1, c + 1), t(r + 1, c)))
    # Flat bottom (outward normal -Z): reversed winding.
    for r in range(rows - 1):
        for c in range(cols - 1):
            faces.append((b(r, c), b(r + 1, c + 1), b(r, c + 1)))
            faces.append((b(r, c), b(r + 1, c), b(r + 1, c + 1)))
    # South skirt (r=0, outward -Y).
    for c in range(cols - 1):
        faces.append((t(0, c), b(0, c), b(0, c + 1)))
        faces.append((t(0, c), b(0, c + 1), t(0, c + 1)))
    # North skirt (r=rows-1, outward +Y).
    R = rows - 1
    for c in range(cols - 1):
        faces.append((t(R, c), t(R, c + 1), b(R, c + 1)))
        faces.append((t(R, c), b(R, c + 1), b(R, c)))
    # West skirt (c=0, outward -X).
    for r in range(rows - 1):
        faces.append((t(r, 0), t(r + 1, 0), b(r + 1, 0)))
        faces.append((t(r, 0), b(r + 1, 0), b(r, 0)))
    # East skirt (c=cols-1, outward +X).
    C = cols - 1
    for r in range(rows - 1):
        faces.append((t(r, C), b(r, C), b(r + 1, C)))
        faces.append((t(r, C), b(r + 1, C), t(r + 1, C)))

    return verts, faces
