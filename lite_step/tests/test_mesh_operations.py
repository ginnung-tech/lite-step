"""
Unit tests for Mesh modification operations:
- .subdivide()
- .smooth()
- .add_random()
- .simplify()

Verifies:
1. Pure/immutable behavior (every op returns a new Mesh copy, never mutating in-place).
2. Manifold watertightness (directed edge matching on closed geometry).
3. RandomState determinism across seeds.
4. Input validation on strength/walk and height_map_only gates.
"""

from collections import Counter
import numpy as np
import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc
from lite_step.models import Mesh, Point, Point2D, Project, Storey


def _make_sample_explicit_mesh() -> Mesh:
    # A simple octahedron-like closed 2-manifold explicit mesh (8 faces, 6 vertices)
    vertices = [
        Point(x=0, y=0, z=1000),     # Top apex (0)
        Point(x=-500, y=-500, z=0),  # Base SW (1)
        Point(x=500, y=-500, z=0),   # Base SE (2)
        Point(x=500, y=500, z=0),    # Base NE (3)
        Point(x=-500, y=500, z=0),   # Base NW (4)
        Point(x=0, y=0, z=-1000),    # Bottom apex (5)
    ]
    faces = [
        # Top 4 faces
        (0, 1, 2),
        (0, 2, 3),
        (0, 3, 4),
        (0, 4, 1),
        # Bottom 4 faces
        (5, 2, 1),
        (5, 3, 2),
        (5, 4, 3),
        (5, 1, 4),
    ]
    return Mesh(vertices=vertices, faces=faces, is_watertight=True)


def _make_sample_heightmap_mesh() -> Mesh:
    return Mesh(
        heightmap=[[0, 500, 200], [300, 800, 400], [100, 400, 0]],
        depth=5000,
        corner_min=Point2D(x=-5000, y=-5000),
        corner_max=Point2D(x=5000, y=5000),
    )


def _assert_is_closed_manifold(mesh: Mesh) -> None:
    """Verifies that a triangle mesh is a closed 2-manifold.

    Every directed edge (u, v) must appear exactly once, and its reverse (v, u)
    must appear exactly once in another face.
    """
    assert mesh.faces is not None and len(mesh.faces) > 0
    half_edges = Counter()
    for v0, v1, v2 in mesh.faces:
        assert len({v0, v1, v2}) == 3, f"Degenerate face: ({v0}, {v1}, {v2})"
        half_edges[(v0, v1)] += 1
        half_edges[(v1, v2)] += 1
        half_edges[(v2, v0)] += 1

    for (u, v), count in half_edges.items():
        assert count == 1, f"Non-manifold edge: ({u}, {v}) appears {count} times"
        assert half_edges[(v, u)] == 1, f"Boundary/unmatched edge: ({u}, {v}) lacks reverse ({v}, {u})"


# =============================================================================
# 1. IMMUTABILITY & CHAINING TESTS
# =============================================================================

def test_mesh_operations_are_non_mutating():
    base = _make_sample_explicit_mesh()
    orig_verts = len(base.vertices)
    orig_faces = len(base.faces)

    coarse = base.subdivide()
    fine = coarse.subdivide()

    # Verify distinct instances
    assert coarse is not base
    assert fine is not coarse
    assert fine is not base

    # Verify input mesh was unmutated
    assert len(base.vertices) == orig_verts
    assert len(base.faces) == orig_faces

    # Verify correct progression
    assert len(coarse.faces) == orig_faces * 4
    assert len(fine.faces) == orig_faces * 16


def test_heightmap_mesh_operations_are_non_mutating():
    base = _make_sample_heightmap_mesh()
    assert len(base.heightmap) == 3

    sub = base.subdivide(height_map_only=True)
    assert sub is not base
    assert len(base.heightmap) == 3
    assert len(sub.heightmap) == 5

    smoothed = sub.smooth(strength=50, height_map_only=True)
    assert smoothed is not sub
    assert smoothed.heightmap != sub.heightmap


# =============================================================================
# 2. SUBDIVIDE & TOPOLOGICAL MANIFOLD TESTS
# =============================================================================

def test_subdivide_manifold_preservation():
    mesh = _make_sample_explicit_mesh()
    _assert_is_closed_manifold(mesh)

    sub1 = mesh.subdivide()
    _assert_is_closed_manifold(sub1)

    sub2 = sub1.subdivide()
    _assert_is_closed_manifold(sub2)


def test_subdivide_heightmap_only_error_on_explicit():
    mesh = _make_sample_explicit_mesh()
    with pytest.raises(ValueError, match="height_map_only=True is only supported on meshes generated from a heightmap"):
        mesh.subdivide(height_map_only=True)


# =============================================================================
# 3. SMOOTH & MANIFOLD TESTS
# =============================================================================

def test_smooth_validation():
    mesh = _make_sample_explicit_mesh()
    with pytest.raises(ValueError):
        mesh.smooth(0)
    with pytest.raises(ValueError):
        mesh.smooth(101)
    with pytest.raises(TypeError):
        mesh.smooth(3.5)  # type: ignore


def test_smooth_manifold_preservation():
    mesh = _make_sample_explicit_mesh().subdivide()
    smoothed = mesh.smooth(strength=75)
    _assert_is_closed_manifold(smoothed)


def test_smooth_reduces_height_variance():
    hm = [[0, 1000, 0], [1000, 0, 1000], [0, 1000, 0]]
    mesh = Mesh(
        heightmap=hm,
        depth=5000,
        corner_min=Point2D(x=-5000, y=-5000),
        corner_max=Point2D(x=5000, y=5000),
    )
    orig_variance = np.var(mesh.heightmap)

    smoothed = mesh.smooth(strength=80, height_map_only=True)
    smoothed_variance = np.var(smoothed.heightmap)
    assert smoothed_variance < orig_variance


# =============================================================================
# 4. ADD_RANDOM & DETERMINISM TESTS
# =============================================================================

def test_add_random_validation():
    mesh = _make_sample_explicit_mesh()
    with pytest.raises(ValueError):
        mesh.add_random(strength=0)
    with pytest.raises(ValueError):
        mesh.add_random(strength=10_000_001)
    with pytest.raises(ValueError):
        mesh.add_random(walk=-1)
    with pytest.raises(ValueError):
        mesh.add_random(walk=10)


def test_add_random_deterministic_seeding():
    mesh1 = _make_sample_heightmap_mesh().add_random(seed=123, strength=400, walk=2)
    mesh2 = _make_sample_heightmap_mesh().add_random(seed=123, strength=400, walk=2)
    assert mesh1.heightmap == mesh2.heightmap

    mesh3 = _make_sample_heightmap_mesh().add_random(seed=999, strength=400, walk=2)
    assert mesh1.heightmap != mesh3.heightmap


def test_add_random_manifold_preservation():
    mesh = _make_sample_explicit_mesh().subdivide()
    perturbed = mesh.add_random(seed=42, strength=100, walk=5)
    _assert_is_closed_manifold(perturbed)


def test_add_random_walk_correlation():
    mesh_walk0 = Mesh(
        heightmap=[[0] * 10 for _ in range(10)],
        depth=5000,
        corner_min=Point2D(x=-5000, y=-5000),
        corner_max=Point2D(x=5000, y=5000),
    ).add_random(seed=42, strength=500, walk=0)

    mesh_walk9 = Mesh(
        heightmap=[[0] * 10 for _ in range(10)],
        depth=5000,
        corner_min=Point2D(x=-5000, y=-5000),
        corner_max=Point2D(x=5000, y=5000),
    ).add_random(seed=42, strength=500, walk=9)

    hm0 = np.array(mesh_walk0.heightmap)
    hm9 = np.array(mesh_walk9.heightmap)

    diff0 = np.abs(np.diff(hm0, axis=0)).mean()
    diff9 = np.abs(np.diff(hm9, axis=0)).mean()
    assert diff9 < diff0


# =============================================================================
# 5. SIMPLIFY TESTS
# =============================================================================

def test_simplify_validation():
    mesh = _make_sample_explicit_mesh()
    with pytest.raises(ValueError):
        mesh.simplify(0)
    with pytest.raises(ValueError):
        mesh.simplify(101)


def test_simplify_heightmap():
    mesh = Mesh(
        heightmap=[[100 * (r + c) for c in range(10)] for r in range(10)],
        depth=5000,
        corner_min=Point2D(x=-5000, y=-5000),
        corner_max=Point2D(x=5000, y=5000),
    )
    assert len(mesh.heightmap) == 10
    simplified = mesh.simplify(strength=80, height_map_only=True)
    assert len(simplified.heightmap) < 10
    assert len(simplified.heightmap) >= 2


def test_simplify_explicit_reduces_faces():
    # Start with a high-poly subdivided octahedron (8 * 16 = 128 faces)
    high_poly = _make_sample_explicit_mesh().subdivide().subdivide()
    assert len(high_poly.faces) == 128

    simplified = high_poly.simplify(strength=80)
    assert len(simplified.faces) < len(high_poly.faces)
    assert len(simplified.faces) >= 4


# =============================================================================
# 6. FULL PIPELINE & COMPILATION INTEGRATION
# =============================================================================

def test_chainable_pipeline_and_ifc_generation():
    proj = Project(name="mesh_ops_test")
    ground = Storey(name="ground", elevation=0)

    terrain = Mesh(
        heightmap=[[0, 100], [100, 200]],
        depth=5000,
        corner_min=Point2D(x=-10000, y=-10000),
        corner_max=Point2D(x=10000, y=10000),
    )
    processed = (
        terrain.subdivide(height_map_only=True)
        .add_random(seed=42, strength=200, walk=5)
        .smooth(40)
        .simplify(30)
    )

    ground.add(processed)
    proj.add_storey(ground)

    norm_proj = normalize_project_to_meters(proj)
    ifc_res = generate_ifc(norm_proj)
    assert ifc_res.success
    assert ifc_res.ifc_content
