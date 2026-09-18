"""
Geometric and Topological Mesh Operations for Lite-STEP.

Provides chainable, non-mutating mesh processing operations:
- Subdivision (1 -> 4 triangle midpoint splitting)
- Laplacian Smoothing (strength 1..100)
- Add Random Perturbation (with neighbor walk correlation 0..9 and strength in mm)
- Simplification / Decimation (strength 1..100)
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, List, Tuple
import numpy as np

from .heightfield import sample_grid, tessellate_heightmap
from .primitives import Point

if TYPE_CHECKING:
    from .elements import Mesh


def _round_int(val: float | int) -> int:
    return int(round(val))


def _validate_strength(strength: int) -> None:
    if not isinstance(strength, int) or isinstance(strength, bool):
        raise TypeError(f"strength must be an integer (got {type(strength).__name__})")
    if not (1 <= strength <= 100):
        raise ValueError(f"strength must be between 1 and 100 (got {strength})")


def _validate_add_random_strength(strength: int) -> None:
    if not isinstance(strength, int) or isinstance(strength, bool):
        raise TypeError(f"strength must be an integer (got {type(strength).__name__})")
    if not (1 <= strength <= 10_000_000):
        raise ValueError(f"strength must be between 1 mm and 10,000,000 mm (got {strength})")


def _validate_walk(walk: int) -> None:
    if not isinstance(walk, int) or isinstance(walk, bool):
        raise TypeError(f"walk must be an integer (got {type(walk).__name__})")
    if not (0 <= walk <= 9):
        raise ValueError(f"walk must be between 0 and 9 (got {walk})")


def _build_adjacency(faces: List[Tuple[int, int, int]], n_verts: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n_verts)]
    for v0, v1, v2 in faces:
        adj[v0].add(v1)
        adj[v0].add(v2)
        adj[v1].add(v0)
        adj[v1].add(v2)
        adj[v2].add(v0)
        adj[v2].add(v1)
    return adj


def _compute_vertex_normals(vertices: List[Point], faces: List[Tuple[int, int, int]]) -> np.ndarray:
    n_verts = len(vertices)
    V = np.array([[p.x, p.y, p.z] for p in vertices], dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)

    normals = np.zeros((n_verts, 3), dtype=np.float64)
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    for i in range(3):
        np.add.at(normals, F[:, i], face_normals)

    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms


# =============================================================================
# 1. SUBDIVISION (1 -> 4 TRIANGLES)
# =============================================================================

def mesh_subdivide(mesh: "Mesh", *, height_map_only: bool = False) -> "Mesh":
    """Subdivides mesh triangles (1 -> 4 midpoint split).

    Returns a new Mesh instance without mutating the input mesh.
    """
    out = mesh.model_copy(deep=True)

    if height_map_only:
        if mesh.heightmap is None:
            raise ValueError("height_map_only=True is only supported on meshes generated from a heightmap")

        rows = len(mesh.heightmap)
        cols = len(mesh.heightmap[0])
        new_rows = 2 * rows - 1
        new_cols = 2 * cols - 1

        new_hm = sample_grid(
            mesh.heightmap,
            mesh.corner_min,
            mesh.corner_max,
            mesh.corner_min,
            mesh.corner_max,
            rows=new_rows,
            cols=new_cols,
            round=True,
        )
        out.heightmap = new_hm
        out.vertices, out.faces = tessellate_heightmap(
            out.heightmap,
            out.depth,
            out.corner_min,
            out.corner_max,
            bottom_z=out.bottom_z,
        )
        return out

    # General explicit triangle subdivision
    verts = list(mesh.vertices)
    faces = list(mesh.faces)
    edge_midpoints: dict[tuple[int, int], int] = {}

    def get_midpoint(i: int, j: int) -> int:
        edge = (min(i, j), max(i, j))
        if edge in edge_midpoints:
            return edge_midpoints[edge]
        p1, p2 = verts[i], verts[j]
        m = Point(
            x=_round_int((p1.x + p2.x) / 2.0),
            y=_round_int((p1.y + p2.y) / 2.0),
            z=_round_int((p1.z + p2.z) / 2.0),
        )
        idx = len(verts)
        verts.append(m)
        edge_midpoints[edge] = idx
        return idx

    new_faces = []
    for v0, v1, v2 in faces:
        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)
        new_faces.append((v0, m01, m20))
        new_faces.append((v1, m12, m01))
        new_faces.append((v2, m20, m12))
        new_faces.append((m01, m12, m20))

    out.vertices = verts
    out.faces = new_faces

    # If this mesh was a heightmap, also upsample its grid
    if mesh.heightmap is not None:
        rows = len(mesh.heightmap)
        cols = len(mesh.heightmap[0])
        out.heightmap = sample_grid(
            mesh.heightmap,
            mesh.corner_min,
            mesh.corner_max,
            mesh.corner_min,
            mesh.corner_max,
            rows=2 * rows - 1,
            cols=2 * cols - 1,
            round=True,
        )
    return out


# =============================================================================
# 2. SMOOTHING (LAPLACIAN)
# =============================================================================

def mesh_smooth(mesh: "Mesh", strength: int = 50, *, height_map_only: bool = False) -> "Mesh":
    """Applies Laplacian smoothing to the mesh.

    Returns a new Mesh instance without mutating the input mesh.
    """
    _validate_strength(strength)
    out = mesh.model_copy(deep=True)

    if height_map_only:
        if mesh.heightmap is None:
            raise ValueError("height_map_only=True is only supported on meshes generated from a heightmap")

        hm = np.array(mesh.heightmap, dtype=np.float64)
        rows, cols = hm.shape
        iterations = max(1, strength // 10)
        factor = (strength / 100.0) * 0.5

        for _ in range(iterations):
            new_hm = hm.copy()
            for r in range(rows):
                for c in range(cols):
                    neighbors = []
                    if r > 0:
                        neighbors.append(hm[r - 1, c])
                    if r < rows - 1:
                        neighbors.append(hm[r + 1, c])
                    if c > 0:
                        neighbors.append(hm[r, c - 1])
                    if c < cols - 1:
                        neighbors.append(hm[r, c + 1])
                    if neighbors:
                        avg = sum(neighbors) / len(neighbors)
                        new_hm[r, c] = hm[r, c] + factor * (avg - hm[r, c])
            hm = new_hm

        out.heightmap = [[_round_int(val) for val in row] for row in hm]
        out.vertices, out.faces = tessellate_heightmap(
            out.heightmap,
            out.depth,
            out.corner_min,
            out.corner_max,
            bottom_z=out.bottom_z,
        )
        return out

    # General mesh Laplacian smoothing
    verts = mesh.vertices
    faces = mesh.faces
    n_verts = len(verts)
    adj = _build_adjacency(faces, n_verts)

    V = np.array([[p.x, p.y, p.z] for p in verts], dtype=np.float64)
    iterations = max(1, strength // 10)
    factor = (strength / 100.0) * 0.5

    for _ in range(iterations):
        V_new = V.copy()
        for i in range(n_verts):
            nbrs = list(adj[i])
            if nbrs:
                avg_pos = np.mean(V[nbrs], axis=0)
                V_new[i] = V[i] + factor * (avg_pos - V[i])
        V = V_new

    out.vertices = [Point(x=_round_int(p[0]), y=_round_int(p[1]), z=_round_int(p[2])) for p in V]

    if mesh.heightmap is not None:
        rows = len(mesh.heightmap)
        cols = len(mesh.heightmap[0])
        out.heightmap = [[0] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                out.heightmap[r][c] = out.vertices[r * cols + c].z
    return out


# =============================================================================
# 3. ADD RANDOM PERTURBATION WITH WALK CORRELATION
# =============================================================================

def mesh_add_random(
    mesh: "Mesh",
    seed: int = 42,
    strength: int = 500,
    walk: int = 0,
    *,
    height_map_only: bool = False,
) -> "Mesh":
    """Adds random perturbation with spatial neighbor correlation (walk).

    Returns a new Mesh instance without mutating the input mesh.
    Uses np.random.RandomState for cross-version deterministic reproducibility.
    """
    _validate_add_random_strength(strength)
    _validate_walk(walk)
    out = mesh.model_copy(deep=True)

    rng = np.random.RandomState(seed)
    amplitude = float(strength)  # Direct millimeter amplitude

    if height_map_only or mesh.heightmap is not None:
        if height_map_only and mesh.heightmap is None:
            raise ValueError("height_map_only=True is only supported on meshes generated from a heightmap")

        hm = np.array(mesh.heightmap, dtype=np.float64)
        rows, cols = hm.shape
        noise = rng.uniform(-1.0, 1.0, size=(rows, cols))

        if walk > 0:
            diffuse_passes = walk
            w_blend = walk / 10.0
            for _ in range(diffuse_passes):
                smoothed = noise.copy()
                for r in range(rows):
                    for c in range(cols):
                        nbrs = []
                        if r > 0:
                            nbrs.append(noise[r - 1, c])
                        if r < rows - 1:
                            nbrs.append(noise[r + 1, c])
                        if c > 0:
                            nbrs.append(noise[r, c - 1])
                        if c < cols - 1:
                            nbrs.append(noise[r, c + 1])
                        if nbrs:
                            smoothed[r, c] = (1.0 - w_blend) * noise[r, c] + w_blend * (sum(nbrs) / len(nbrs))
                noise = smoothed

        hm += noise * amplitude
        out.heightmap = [[_round_int(val) for val in row] for row in hm]
        out.vertices, out.faces = tessellate_heightmap(
            out.heightmap,
            out.depth,
            out.corner_min,
            out.corner_max,
            bottom_z=out.bottom_z,
        )
        return out

    # General mesh 3D normal displacement
    verts = mesh.vertices
    faces = mesh.faces
    n_verts = len(verts)
    adj = _build_adjacency(faces, n_verts)
    normals = _compute_vertex_normals(verts, faces)

    noise = rng.uniform(-1.0, 1.0, size=n_verts)
    if walk > 0:
        diffuse_passes = walk
        w_blend = walk / 10.0
        for _ in range(diffuse_passes):
            smoothed = noise.copy()
            for i in range(n_verts):
                nbrs = list(adj[i])
                if nbrs:
                    avg = np.mean(noise[nbrs])
                    smoothed[i] = (1.0 - w_blend) * noise[i] + w_blend * avg
            noise = smoothed

    V = np.array([[p.x, p.y, p.z] for p in verts], dtype=np.float64)
    V += normals * (noise[:, np.newaxis] * amplitude)

    out.vertices = [Point(x=_round_int(p[0]), y=_round_int(p[1]), z=_round_int(p[2])) for p in V]
    return out


# =============================================================================
# 4. SIMPLIFICATION (DECIMATION)
# =============================================================================

def mesh_simplify(mesh: "Mesh", strength: int = 50, *, height_map_only: bool = False) -> "Mesh":
    """Simplifies the mesh by reducing triangle and vertex count.

    Returns a new Mesh instance without mutating the input mesh.
    """
    _validate_strength(strength)
    out = mesh.model_copy(deep=True)

    if height_map_only or mesh.heightmap is not None:
        if height_map_only and mesh.heightmap is None:
            raise ValueError("height_map_only=True is only supported on meshes generated from a heightmap")

        rows = len(mesh.heightmap)
        cols = len(mesh.heightmap[0])
        # Downsampling ratio based on strength
        ratio = 1.0 - 0.75 * (strength / 100.0)
        target_rows = max(2, int(round(rows * ratio)))
        target_cols = max(2, int(round(cols * ratio)))

        if target_rows == rows and target_cols == cols:
            target_rows = max(2, rows - 1)
            target_cols = max(2, cols - 1)

        new_hm = sample_grid(
            mesh.heightmap,
            mesh.corner_min,
            mesh.corner_max,
            mesh.corner_min,
            mesh.corner_max,
            rows=target_rows,
            cols=target_cols,
            round=True,
        )
        out.heightmap = new_hm
        out.vertices, out.faces = tessellate_heightmap(
            out.heightmap,
            out.depth,
            out.corner_min,
            out.corner_max,
            bottom_z=out.bottom_z,
        )
        return out

    # Explicit mesh decimation: grid vertex clustering decimation
    verts = mesh.vertices
    faces = mesh.faces
    V = np.array([[p.x, p.y, p.z] for p in verts], dtype=np.float64)

    bbox_min = V.min(axis=0)
    bbox_max = V.max(axis=0)
    bbox_span = np.maximum(bbox_max - bbox_min, 1.0)

    # Cluster grid resolution from strength
    divisions = max(2, int(round(20.0 * (1.0 - 0.85 * (strength / 100.0)))))
    cell_size = bbox_span / divisions

    cluster_map: dict[tuple[int, int, int], list[int]] = collections.defaultdict(list)
    for i, p in enumerate(V):
        coord = tuple(np.floor((p - bbox_min) / cell_size).astype(int))
        cluster_map[coord].append(i)

    new_verts: list[Point] = []
    old_to_new = np.zeros(len(V), dtype=np.int64)

    for cluster_verts in cluster_map.values():
        new_idx = len(new_verts)
        center = np.mean(V[cluster_verts], axis=0)
        new_verts.append(Point(x=_round_int(center[0]), y=_round_int(center[1]), z=_round_int(center[2])))
        for old_idx in cluster_verts:
            old_to_new[old_idx] = new_idx

    new_faces: list[Tuple[int, int, int]] = []
    seen_faces = set()
    for v0, v1, v2 in faces:
        nv0 = int(old_to_new[v0])
        nv1 = int(old_to_new[v1])
        nv2 = int(old_to_new[v2])
        if nv0 != nv1 and nv1 != nv2 and nv2 != nv0:
            face_key = (nv0, nv1, nv2)
            if face_key not in seen_faces:
                seen_faces.add(face_key)
                new_faces.append(face_key)

    if len(new_faces) >= 4:
        out.vertices = new_verts
        out.faces = new_faces

    return out
