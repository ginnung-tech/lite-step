"""Tests for mesh watertightness, coincident vertex welding, and Gauss Divergence signed volume validation."""

from pathlib import Path
import numpy as np
import pytest

from lite_step.compiler.executor import (
    check_mesh_signed_volume,
    check_mesh_watertightness,
    validate_project_report,
)
from lite_step.models import Box, Mesh, Point, Point2D, Project, Roof, Slab, Storey, Wall


def _make_roof_mesh() -> Mesh:
    """Generate an open canopy roof mesh (open 2-manifold with boundary edges)."""
    rx_w, rx_e, ry, ridge_z = -3500, 3000, 0, 8800
    ox_w, ox_e, oy_s, oy_n, eave_z = -7300, 7300, -6300, 6300, 6450
    cx, cy, r_eave = 4500, -3500, 2800

    verts = [
        Point(x=rx_w, y=ry, z=ridge_z),
        Point(x=rx_e, y=ry, z=ridge_z),
        Point(x=ox_w, y=oy_n, z=eave_z),
        Point(x=ox_e, y=oy_n, z=eave_z),
        Point(x=ox_e, y=cy, z=eave_z),
    ]
    n_arc = 12
    arc_start_idx = len(verts)
    for i in range(n_arc + 1):
        ang = -np.pi / 2 + (np.pi / 2) * (i / n_arc)
        verts.append(Point(x=int(round(cx + r_eave * np.cos(ang))),
                           y=int(round(cy + r_eave * np.sin(ang))),
                           z=eave_z))
    arc_end_idx = len(verts) - 1
    sw_idx = len(verts)
    verts.append(Point(x=ox_w, y=oy_s, z=eave_z))

    faces = [
        (2, 3, 1), (2, 1, 0), (2, 0, sw_idx), (sw_idx, 0, 1), (sw_idx, 1, arc_start_idx),
    ]
    for i in range(n_arc):
        faces.append((arc_start_idx + i, 1, arc_start_idx + i + 1))
    faces.append((arc_end_idx, 1, 3))

    return Mesh(name="test_roof", vertices=verts, faces=faces)


def _make_cube_mesh(min_pt=(0, 0, 0), size: int = 1000, inverted: bool = False) -> Mesh:
    """Generate a closed watertight cube mesh."""
    x0, y0, z0 = int(min_pt[0]), int(min_pt[1]), int(min_pt[2])
    s = int(size)
    corners = [
        Point(x=x0, y=y0, z=z0), Point(x=x0 + s, y=y0, z=z0),
        Point(x=x0 + s, y=y0 + s, z=z0), Point(x=x0, y=y0 + s, z=z0),
        Point(x=x0, y=y0, z=z0 + s), Point(x=x0 + s, y=y0, z=z0 + s),
        Point(x=x0 + s, y=y0 + s, z=z0 + s), Point(x=x0, y=y0 + s, z=z0 + s),
    ]
    # CCW outward faces
    faces_ccw = [
        (0, 3, 2), (0, 2, 1),  # bottom (-Z)
        (4, 5, 6), (4, 6, 7),  # top (+Z)
        (0, 1, 5), (0, 5, 4),  # front (-Y)
        (2, 3, 7), (2, 7, 6),  # back (+Y)
        (0, 4, 7), (0, 7, 3),  # left (-X)
        (1, 2, 6), (1, 6, 5),  # right (+X)
    ]
    if inverted:
        faces = [(a, c, b) for a, b, c in faces_ccw]
    else:
        faces = faces_ccw

    return Mesh(name="test_cube", vertices=corners, faces=faces)


def test_synthetic_terrain_mesh_not_warned():
    proj = Project(name="terrain_proj")
    s = Storey(elevation=0, name="storey")
    terrain = Mesh(
        heightmap=[[0, 100], [100, 200]],
        corner_min=Point2D(x=-5000, y=-5000),
        corner_max=Point2D(x=5000, y=5000),
        bottom_z=-2000,
        source_type="synthetic",
    )
    s.add(terrain)
    proj.add_storey(s)
    report = validate_project_report(proj)
    assert report.errors == []
    mesh_warnings = [w for w in report.warnings if "mesh(es)" in w]
    assert len(mesh_warnings) == 0


def test_compiler_emits_warning_for_open_mesh():
    proj = Project(name="test_open_warning_proj")
    s = Storey(elevation=0, name="storey")
    roof = Roof(name="roof")
    roof.add(_make_roof_mesh())
    s.add(roof)
    proj.add_storey(s)

    report = validate_project_report(proj)
    assert report.errors == []
    matching = [w for w in report.warnings if "are open surfaces" in w]
    assert len(matching) == 1
    assert "test_roof" in matching[0]


def test_bad_building_fixture_emits_open_roof_warning():
    fixture_py = Path(__file__).parent / "fixtures" / "inverted_roof_building.py"
    assert fixture_py.exists(), f"Fixture {fixture_py} missing"

    import importlib.util
    spec = importlib.util.spec_from_file_location("inverted_roof_building", str(fixture_py))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    proj = module.generate_project()
    report = validate_project_report(proj)

    # Open roof mesh must warn loudly about open boundaries
    mesh_warnings = [w for w in report.warnings if "are open surfaces" in w]
    assert len(mesh_warnings) >= 1
    assert any("roof_tiles" in w for w in mesh_warnings)
    assert report.errors == []


def test_mesh_watertightness_and_signed_volume_direct():
    open_roof = _make_roof_mesh()
    is_closed, open_edges = check_mesh_watertightness(open_roof)
    assert is_closed is False
    assert open_edges > 0

    good_box = _make_cube_mesh(inverted=False)
    bad_box = _make_cube_mesh(inverted=True)

    assert check_mesh_watertightness(good_box) == (True, 0)
    assert check_mesh_watertightness(bad_box) == (True, 0)

    assert check_mesh_signed_volume(good_box) > 0.0
    assert check_mesh_signed_volume(bad_box) < 0.0


def test_open_mesh_loud_warning_with_downsides():
    proj = Project(name="test_open_mesh_downsides")
    s = Storey(elevation=0, name="storey")
    roof = Roof(name="roof")
    roof.add(_make_roof_mesh())
    s.add(roof)
    proj.add_storey(s)

    report = validate_project_report(proj)
    assert report.errors == []

    open_warnings = [w for w in report.warnings if "are open surfaces" in w]
    assert len(open_warnings) == 1
    w = open_warnings[0]
    assert "not watertight solid volume" in w
    assert "boolean cuts/openings (.difference())" in w
    assert "backface culling" in w
    assert "zero physical BIM volume" in w


def test_closed_mesh_negative_volume_warning():
    proj = Project(name="test_closed_mesh_negative_volume")
    s = Storey(elevation=0, name="storey")
    bad_box = _make_cube_mesh(inverted=True)
    s.add(bad_box)
    proj.add_storey(s)

    report = validate_project_report(proj)
    assert report.errors == []
    vol_warnings = [w for w in report.warnings if "have negative signed volume" in w]
    assert len(vol_warnings) == 1
    assert "cannot hold water" in vol_warnings[0]
    assert "inverted inside-out winding" in vol_warnings[0]


def test_non_mesh_elements_not_flagged_by_mesh_checks():
    proj = Project(name="test_standard_primitives")
    s = Storey(elevation=0, name="storey")
    w = Wall(name="south")
    w.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=3000, y=300, z=2600), name="body"))
    s.add(w)
    s.add(Slab(name="floor"))
    proj.add_storey(s)

    report = validate_project_report(proj)
    assert report.errors == []
    mesh_warnings = [w for w in report.warnings if "mesh(es)" in w]
    assert len(mesh_warnings) == 0


def test_duplicate_vertex_closed_solid_watertight_c2():
    """C2: Closed solid authored with unshared duplicate vertices (4 per face = 24 verts)
    must weld coincident vertices and pass watertightness with 0 open boundary edges."""
    verts = []
    faces = []
    # 6 quad faces with outward-facing CCW windings, each quad has 4 unique vertex instances
    quad_defs = [
        # bottom (-Z)
        [(0, 0, 0), (0, 1000, 0), (1000, 1000, 0), (1000, 0, 0)],
        # top (+Z)
        [(0, 0, 1000), (1000, 0, 1000), (1000, 1000, 1000), (0, 1000, 1000)],
        # front (-Y)
        [(0, 0, 0), (1000, 0, 0), (1000, 0, 1000), (0, 0, 1000)],
        # back (+Y)
        [(0, 1000, 0), (0, 1000, 1000), (1000, 1000, 1000), (1000, 1000, 0)],
        # left (-X)
        [(0, 0, 0), (0, 0, 1000), (0, 1000, 1000), (0, 1000, 0)],
        # right (+X)
        [(1000, 0, 0), (1000, 1000, 0), (1000, 1000, 1000), (1000, 0, 1000)],
    ]
    for quad in quad_defs:
        base_idx = len(verts)
        for x, y, z in quad:
            verts.append(Point(x=x, y=y, z=z))
        faces.append((base_idx, base_idx + 1, base_idx + 2))
        faces.append((base_idx, base_idx + 2, base_idx + 3))

    assert len(verts) == 24
    assert len(faces) == 12

    unshared_cube = Mesh(name="unshared_cube", vertices=verts, faces=faces)
    is_closed, open_edges = check_mesh_watertightness(unshared_cube)
    assert is_closed is True, f"Expected watertight solid via vertex welding, got {open_edges} open edges"
    assert open_edges == 0

    vol = check_mesh_signed_volume(unshared_cube)
    assert abs(vol - 1_000_000_000.0) < 1.0


def test_out_of_range_and_negative_index_c3():
    """C3: Out-of-range index (>len(v)) and negative index (<0) must be caught upfront."""
    verts = [Point(x=0, y=0, z=0), Point(x=100, y=0, z=0), Point(x=0, y=100, z=0), Point(x=0, y=0, z=100)]
    # Out of range index (index 10 is >= 4)
    bad_f_high = Mesh(name="bad_high", vertices=verts, faces=[(0, 1, 10), (0, 1, 2), (0, 2, 3), (1, 2, 3)])
    # Negative index (-1 wraps in python)
    bad_f_neg = Mesh(name="bad_neg", vertices=verts, faces=[(0, 1, -1), (0, 1, 2), (0, 2, 3), (1, 2, 3)])

    with pytest.raises(ValueError, match="out of range"):
        check_mesh_watertightness(bad_f_high)

    with pytest.raises(ValueError, match="out of range"):
        check_mesh_signed_volume(bad_f_high)

    with pytest.raises(ValueError, match="out of range"):
        check_mesh_watertightness(bad_f_neg)

    with pytest.raises(ValueError, match="out of range"):
        check_mesh_signed_volume(bad_f_neg)

    # In validate_project_report it must record a compile error
    proj = Project(name="bad_indices_proj")
    s = Storey(elevation=0, name="storey")
    s.add(bad_f_high)
    s.add(bad_f_neg)
    proj.add_storey(s)
    report = validate_project_report(proj)
    assert len(report.errors) == 2
    assert any("out of range" in err for err in report.errors)


def test_non_manifold_edge_detected_open():
    """Two boxes meeting along exactly one edge create a non-manifold edge (shared by 4 faces)."""
    b1 = _make_cube_mesh(min_pt=(0, 0, 0), size=1000)
    b2 = _make_cube_mesh(min_pt=(1000, 1000, 0), size=1000)
    verts = list(b1.vertices) + list(b2.vertices)
    faces = list(b1.faces) + [(a + len(b1.vertices), b + len(b1.vertices), c + len(b1.vertices)) for a, b, c in b2.faces]
    compound = Mesh(name="edge_touch", vertices=verts, faces=faces)

    is_closed, open_edges = check_mesh_watertightness(compound)
    assert is_closed is False
    assert open_edges > 0


def test_compound_multi_solid_mesh():
    """Two completely disjoint closed cubes in one mesh pass watertightness and sum volumes correctly (+16,000,000)."""
    b1 = _make_cube_mesh(min_pt=(0, 0, 0), size=200)      # vol = 200^3 = 8,000,000
    b2 = _make_cube_mesh(min_pt=(1000, 0, 0), size=200)   # vol = 200^3 = 8,000,000
    verts = list(b1.vertices) + list(b2.vertices)
    faces = list(b1.faces) + [(a + len(b1.vertices), b + len(b1.vertices), c + len(b1.vertices)) for a, b, c in b2.faces]
    multi = Mesh(name="two_cubes", vertices=verts, faces=faces)

    is_closed, open_edges = check_mesh_watertightness(multi)
    assert is_closed is True
    assert open_edges == 0

    vol = check_mesh_signed_volume(multi)
    assert abs(vol - 16_000_000.0) < 1.0


def test_hollow_solid_with_cavity():
    """Hollow solid: outer cube (+27,000,000) with reversed inner cube (-8,000,000) -> net vol = +19,000,000 mm3."""
    outer = _make_cube_mesh(min_pt=(0, 0, 0), size=300, inverted=False)   # vol = 27,000,000
    inner = _make_cube_mesh(min_pt=(50, 50, 50), size=200, inverted=True) # vol = -8,000,000
    verts = list(outer.vertices) + list(inner.vertices)
    faces = list(outer.faces) + [(a + len(outer.vertices), b + len(outer.vertices), c + len(outer.vertices)) for a, b, c in inner.faces]
    hollow = Mesh(name="hollow_box", vertices=verts, faces=faces)

    is_closed, open_edges = check_mesh_watertightness(hollow)
    assert is_closed is True
    assert open_edges == 0

    vol = check_mesh_signed_volume(hollow)
    assert abs(vol - 19_000_000.0) < 1.0


def test_complicated_random_walk_mesh_signed_volume():
    """Verify signed volume evaluation correctly determines normal vs inverted on tortuous 744-face solid."""
    verts, faces_norm, _ = _generate_random_walk_mesh()
    faces_inv = [(a, c, b) for a, b, c in faces_norm]

    mesh_norm = Mesh(name="chain_normal", vertices=verts, faces=faces_norm)
    mesh_inv = Mesh(name="chain_inverted", vertices=verts, faces=faces_inv)

    vol_norm = check_mesh_signed_volume(mesh_norm)
    vol_inv = check_mesh_signed_volume(mesh_inv)

    assert vol_norm > 0.0
    assert vol_inv < 0.0
    assert abs(vol_norm + vol_inv) < 1e-3
    assert check_mesh_watertightness(mesh_norm) == (True, 0)
    assert check_mesh_watertightness(mesh_inv) == (True, 0)


def _generate_random_walk_mesh(num_steps=30, step_len=500.0, radius=150.0, n_sides=12, seed=42):
    """Generate a tortuous 3D tube mesh following a Monte Carlo random walk."""
    import math
    rng = np.random.default_rng(seed)
    pos = np.array([0.0, 0.0, 0.0])
    direction = np.array([step_len, 0.0, 0.0])
    path_pts = [pos.copy()]
    tangents = []

    for _ in range(num_steps):
        next_pos = pos + direction
        path_pts.append(next_pos.copy())
        tangents.append(direction / np.linalg.norm(direction))
        ax, ay, az = np.radians(rng.uniform(-35, 35, size=3))
        rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
        ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
        rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
        rot = rz @ ry @ rx
        pos = next_pos
        direction = rot @ direction
    tangents.append(tangents[-1])

    rings = []
    up_ref = np.array([0.0, 0.0, 1.0])
    for pt, t in zip(path_pts, tangents):
        if abs(np.dot(t, up_ref)) > 0.9:
            u_axis = np.cross(t, np.array([0.0, 1.0, 0.0]))
        else:
            u_axis = np.cross(t, up_ref)
        u_axis /= np.linalg.norm(u_axis)
        v_axis = np.cross(t, u_axis)
        v_axis /= np.linalg.norm(v_axis)
        ring = []
        for s in range(n_sides):
            angle = 2.0 * math.pi * s / n_sides
            ring_pt = pt + radius * (math.cos(angle) * u_axis + math.sin(angle) * v_axis)
            ring.append(ring_pt)
        rings.append(ring)

    verts = [Point(x=int(round(path_pts[0][0])), y=int(round(path_pts[0][1])), z=int(round(path_pts[0][2])))]
    ring_indices = []
    for ring in rings:
        r_idxs = []
        for pt in ring:
            r_idxs.append(len(verts))
            verts.append(Point(x=int(round(pt[0])), y=int(round(pt[1])), z=int(round(pt[2]))))
        ring_indices.append(r_idxs)
    end_center_idx = len(verts)
    verts.append(Point(x=int(round(path_pts[-1][0])), y=int(round(path_pts[-1][1])), z=int(round(path_pts[-1][2]))))

    faces = []
    r0 = ring_indices[0]
    for s in range(n_sides):
        s_next = (s + 1) % n_sides
        faces.append((0, r0[s], r0[s_next]))
    for i in range(len(rings) - 1):
        r_curr = ring_indices[i]
        r_next = ring_indices[i + 1]
        for s in range(n_sides):
            s_next = (s + 1) % n_sides
            c_curr, c_next = r_curr[s], r_curr[s_next]
            n_curr, n_next = r_next[s], r_next[s_next]
            faces.append((c_curr, n_next, n_curr))
            faces.append((c_curr, c_next, n_next))
    r_last = ring_indices[-1]
    for s in range(n_sides):
        s_next = (s + 1) % n_sides
        faces.append((end_center_idx, r_last[s_next], r_last[s]))

    n_pts = len(path_pts)
    spine_probes = [
        path_pts[int(n_pts * 0.15)],
        path_pts[int(n_pts * 0.35)],
        path_pts[int(n_pts * 0.50)],
        path_pts[int(n_pts * 0.65)],
        path_pts[int(n_pts * 0.85)],
    ]
    return verts, faces, spine_probes
