#!/usr/bin/env python3
"""Fuzzer for IfcOpenShell geometry and boolean operations.

Systematically tests combinations of geometry (Box, Extrude, Bar, Pipe, Mesh)
and operations (CSG Difference, CSG Union, CSG Intersection, IfcRelVoidsElement,
Clips) to determine whether the tessellated result "holds water".

Validates:
1. Generation: Project compiles to valid IFC.
2. Tessellation: `ifcopenshell.geom.create_shape()` succeeds without crashing.
3. Watertightness: Evaluated via `check_mesh_watertightness()` (0 open boundary edges).
4. Signed Volume ("Holds Water"): Evaluated via `check_mesh_signed_volume()`:
   - Must be strictly positive (> 0).
   - Negative volume indicates inverted face normals (inside-out).
   - Zero volume indicates collapsed / vanished geometry.
5. Invariant Bounds: Measured signed volume matches analytical conservation bounds
   (catches material-dropping bugs such as IfcOpenShell issue).

Usage:
    python -m lite_step.tests.tools.fuzz_ifcopenshell
    python -m lite_step.tests.tools.fuzz_ifcopenshell --suite swept_disk
    python -m lite_step.tests.tools.fuzz_ifcopenshell --suite coplanar
    python -m lite_step.tests.tools.fuzz_ifcopenshell --iterations 50 --seed 42
    python -m lite_step.tests.tools.fuzz_ifcopenshell --out-dir failures/
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
import sys
import time
from typing import Callable, Optional

import numpy as np

# Ensure repo root is on sys.path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# ruff: noqa: E402
import ifcopenshell
import ifcopenshell.geom

from lite_step.compiler.executor import (
    check_mesh_signed_volume,
    check_mesh_watertightness,
    normalize_project_to_meters,
)
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Bar,
    Box,
    Column,
    Extrude,
    Mesh,
    Pipe,
    Point,
    Project,
    Storey,
)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FuzzCase:
    name: str
    suite: str
    description: str
    builder: Callable[[Storey], None]
    expected_vol_min: Optional[float] = None  # in mm³
    expected_vol_max: Optional[float] = None  # in mm³
    is_known_bug: bool = False
    known_bug_ref: Optional[str] = None


@dataclasses.dataclass
class ProductResult:
    product_name: str
    product_type: str
    vertex_count: int
    face_count: int
    is_watertight: bool
    open_edges: int
    signed_volume: float  # mm³
    status: str  # PASS, LEAK, INVERTED, EMPTY, DISCARDED_MATERIAL, CRASH
    error_message: str = ""


@dataclasses.dataclass
class CaseResult:
    case: FuzzCase
    duration_ms: float
    ifc_content: str = ""
    products: list[ProductResult] = dataclasses.field(default_factory=list)
    status: str = "PASS"
    error: str = ""


class MeshDuckView:
    """Lightweight duck-type wrapping raw numpy vertices and faces for executor checks."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vertices = vertices
        self.faces = faces
        self._np_vertices = vertices
        self._np_faces = faces


# ---------------------------------------------------------------------------
# Test Case Suites
# ---------------------------------------------------------------------------

def _make_watertight_cube_mesh(name: str = "cube", size: float = 1000.0, center: tuple[float, float, float] = (0, 0, 0)) -> Mesh:
    hs = int(round(size / 2.0))
    cx, cy, cz = int(round(center[0])), int(round(center[1])), int(round(center[2]))
    corners = [
        Point(x=cx - hs, y=cy - hs, z=cz - hs), Point(x=cx + hs, y=cy - hs, z=cz - hs),
        Point(x=cx + hs, y=cy + hs, z=cz - hs), Point(x=cx - hs, y=cy + hs, z=cz - hs),
        Point(x=cx - hs, y=cy - hs, z=cz + hs), Point(x=cx + hs, y=cy - hs, z=cz + hs),
        Point(x=cx + hs, y=cy + hs, z=cz + hs), Point(x=cx - hs, y=cy + hs, z=cz + hs),
    ]
    faces = [
        (0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
        (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
    ]
    return Mesh(name=name, vertices=corners, faces=faces)


def build_suite_swept_disk() -> list[FuzzCase]:
    """Test suite covering IfcSweptDiskSolid (Bar, Pipe) with voids and booleans.

    Specifically targets IfcOpenShell Issue #9256:
    https://github.com/IfcOpenShell/IfcOpenShell/issues/9256
    """
    cases = []

    # 1. Bar middle void (direct repro)
    # 32mm bar, 2700mm tall, hole at Z[900, 2300].
    # Expected: Keep Z[0, 900] + Z[2300, 2700] (1300mm length, vol ~ 1,045,522 mm³)
    # Bug: ifcopenshell trims directrix and discards Z[0, 900], keeping only 400mm (~318,577 mm³).
    def case_bar_void_middle(s: Storey):
        col = Column(name="post")
        col.add(Bar(path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=2700)], diameter=32, name="bar"))
        col.void(Box(start=Point(x=-100, y=-100, z=900), end=Point(x=100, y=100, z=2300), name="void"))
        s.add(col)

    bar_r = 16.0
    bar_vol_1300 = math.pi * (bar_r ** 2) * 1300.0  # ~1,045,522 mm³
    cases.append(FuzzCase(
        name="bar_void_middle",
        suite="swept_disk",
        description="Bar with middle void: tests whether material on BOTH sides of void is kept",
        builder=case_bar_void_middle,
        expected_vol_min=bar_vol_1300 * 0.95,
        expected_vol_max=bar_vol_1300 * 1.05,
        is_known_bug=True,
        known_bug_ref="https://github.com/IfcOpenShell/IfcOpenShell/issues/9256",
    ))

    # 2. Bar top void (Issue #9256 edge condition)
    # Void overlaps the end of the directrix: Z[1800, 2800].
    # Expected: Keep Z[0, 1800] (~1,447,646 mm³).
    # Bug: ifcopenshell trims directrix past opening and the whole solid vanishes (0 verts).
    def case_bar_void_top(s: Storey):
        col = Column(name="post")
        col.add(Bar(path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=2700)], diameter=32, name="bar"))
        col.void(Box(start=Point(x=-100, y=-100, z=1800), end=Point(x=100, y=100, z=2800), name="void"))
        s.add(col)

    bar_vol_1800 = math.pi * (bar_r ** 2) * 1800.0
    cases.append(FuzzCase(
        name="bar_void_top",
        suite="swept_disk",
        description="Bar with void at top end: tests whether bar survives or vanishes to empty",
        builder=case_bar_void_top,
        expected_vol_min=bar_vol_1800 * 0.95,
        expected_vol_max=bar_vol_1800 * 1.05,
        is_known_bug=True,
        known_bug_ref="https://github.com/IfcOpenShell/IfcOpenShell/issues/9256",
    ))

    # 3. Bar CSG Difference (Counterpart to void)
    # CSG difference on Bar uses IfcBooleanResult, which should correctly preserve both ends.
    def case_bar_csg_diff(s: Storey):
        col = Column(name="post")
        b = Bar(path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=2700)], diameter=32, name="bar")
        b.difference(Box(start=Point(x=-100, y=-100, z=900), end=Point(x=100, y=100, z=2300), name="tool"))
        col.add(b)
        s.add(col)

    cases.append(FuzzCase(
        name="bar_csg_diff_middle",
        suite="swept_disk",
        description="Bar with CSG difference in middle: tests solid boolean subtraction path",
        builder=case_bar_csg_diff,
        expected_vol_min=bar_vol_1300 * 0.95,
        expected_vol_max=bar_vol_1300 * 1.05,
        is_known_bug=False,
    ))

    # 4. Pipe with middle void
    pipe_r = 50.0
    pipe_vol_1300 = math.pi * (pipe_r ** 2) * 1300.0
    def case_pipe_void(s: Storey):
        col = Column(name="post")
        col.add(Pipe(path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=2700)], radius=50, name="pipe"))
        col.void(Box(start=Point(x=-100, y=-100, z=900), end=Point(x=100, y=100, z=2300), name="void"))
        s.add(col)

    cases.append(FuzzCase(
        name="pipe_void_middle",
        suite="swept_disk",
        description="Pipe with middle void: tests IfcSweptDiskSolid voiding on larger diameter",
        builder=case_pipe_void,
        expected_vol_min=pipe_vol_1300 * 0.95,
        expected_vol_max=pipe_vol_1300 * 1.05,
        is_known_bug=True,
        known_bug_ref="https://github.com/IfcOpenShell/IfcOpenShell/issues/9256",
    ))

    # 5. Angled 3D Bar with horizontal void
    def case_bar_angled(s: Storey):
        col = Column(name="post")
        col.add(Bar(path=[Point(x=0, y=0, z=0), Point(x=2000, y=2000, z=2000)], diameter=32, name="bar"))
        col.void(Box(start=Point(x=600, y=600, z=600), end=Point(x=1400, y=1400, z=1400), name="void"))
        s.add(col)

    cases.append(FuzzCase(
        name="bar_angled_void",
        suite="swept_disk",
        description="3D diagonal Bar with void: tests swept disk subtraction on non-axial paths",
        builder=case_bar_angled,
    ))

    # 6. L-shaped multi-segment Bar with corner void
    def case_bar_l_shape(s: Storey):
        col = Column(name="post")
        col.add(Bar(path=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0), Point(x=1000, y=1000, z=0)], diameter=32, name="bar"))
        col.void(Box(start=Point(x=900, y=-50, z=-50), end=Point(x=1050, y=100, z=50), name="void"))
        s.add(col)

    cases.append(FuzzCase(
        name="bar_l_shape_corner_void",
        suite="swept_disk",
        description="L-shaped Bar with void over elbow corner",
        builder=case_bar_l_shape,
    ))

    return cases


def build_suite_coplanar() -> list[FuzzCase]:
    """Test suite covering coplanar and grazing cuts (boundary alignment edge cases)."""
    cases = []

    # 1. Coplanar 1-face flush cut
    # Base: [0, 1000]^3 (1,000,000,000 mm³). Cut: [0, 500] x [0, 1000] x [0, 1000].
    # Expected: exactly 500,000,000 mm³.
    def case_coplanar_1_face(s: Storey):
        col = Column(name="col")
        b = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000), name="base")
        b.difference(Box(start=Point(x=0, y=0, z=0), end=Point(x=500, y=1000, z=1000), name="tool"))
        col.add(b)
        s.add(col)

    cases.append(FuzzCase(
        name="coplanar_single_face_flush",
        suite="coplanar",
        description="Box cut sharing 1 coplanar face with host",
        builder=case_coplanar_1_face,
        expected_vol_min=499_999_000.0,
        expected_vol_max=500_001_000.0,
    ))

    # 2. Coplanar corner flush (sharing 2 faces: X=0 and Y=0)
    def case_coplanar_corner(s: Storey):
        col = Column(name="col")
        b = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000), name="base")
        b.difference(Box(start=Point(x=0, y=0, z=0), end=Point(x=400, y=400, z=1000), name="tool"))
        col.add(b)
        s.add(col)

    cases.append(FuzzCase(
        name="coplanar_two_faces_corner_flush",
        suite="coplanar",
        description="Box cut sharing 2 coplanar faces (corner notch flush on X and Y)",
        builder=case_coplanar_corner,
        expected_vol_min=839_999_000.0,
        expected_vol_max=840_001_000.0,
    ))

    # 3. Near-zero grazing cut (0.1 mm sliver)
    def case_grazing_sliver(s: Storey):
        col = Column(name="col")
        b = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000), name="base")
        b.difference(Box(start=Point(x=999, y=0, z=0), end=Point(x=1500, y=1000, z=1000), name="tool"))
        col.add(b)
        s.add(col)

    cases.append(FuzzCase(
        name="coplanar_grazing_sliver",
        suite="coplanar",
        description="Near-zero grazing cut (1mm sliver cut at boundary)",
        builder=case_grazing_sliver,
        expected_vol_min=998_999_000.0,
        expected_vol_max=999_001_000.0,
    ))

    # 4. Coplanar touching union
    def case_touching_union(s: Storey):
        col = Column(name="col")
        b1 = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000), name="left")
        b1.union(Box(start=Point(x=1000, y=0, z=0), end=Point(x=2000, y=1000, z=1000), name="right"))
        col.add(b1)
        s.add(col)

    cases.append(FuzzCase(
        name="coplanar_touching_union",
        suite="coplanar",
        description="Union of two adjacent boxes touching at X=1000 coplanar face",
        builder=case_touching_union,
        expected_vol_min=1_999_990_000.0,
        expected_vol_max=2_000_010_000.0,
    ))

    return cases


def build_suite_csg() -> list[FuzzCase]:
    """Test suite covering CSG difference, union, intersection across primitives."""
    cases = []

    # 1. Box through-hole cut
    # Host: 1000x1000x1000. Hole: 400x400x1200 centered at Z.
    def case_through_hole(s: Storey):
        col = Column(name="col")
        b = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000), name="host")
        b.difference(Box(start=Point(x=300, y=300, z=-100), end=Point(x=700, y=700, z=1100), name="hole"))
        col.add(b)
        s.add(col)

    cases.append(FuzzCase(
        name="csg_diff_through_hole",
        suite="csg",
        description="Centered rectangular through-hole penetrating top and bottom",
        builder=case_through_hole,
        expected_vol_min=839_999_000.0,
        expected_vol_max=840_001_000.0,
    ))

    # 2. Box blind pocket cut
    def case_blind_pocket(s: Storey):
        col = Column(name="col")
        b = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000), name="host")
        b.difference(Box(start=Point(x=200, y=200, z=500), end=Point(x=800, y=800, z=1100), name="pocket"))
        col.add(b)
        s.add(col)

    cases.append(FuzzCase(
        name="csg_diff_blind_pocket",
        suite="csg",
        description="Blind pocket cut into top face without penetrating bottom",
        builder=case_blind_pocket,
        expected_vol_min=819_999_000.0,
        expected_vol_max=820_001_000.0,
    ))

    # 3. Extrude L-column with notch
    def case_extrude_notch(s: Storey):
        col = Column(name="col")
        contour = [
            Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0),
            Point(x=1000, y=400, z=0), Point(x=400, y=400, z=0),
            Point(x=400, y=1000, z=0), Point(x=0, y=1000, z=0),
        ]
        ext = Extrude(contour=contour, thickness=2000, name="l_col")
        ext.difference(Box(start=Point(x=100, y=-100, z=500), end=Point(x=300, y=500, z=1500), name="notch"))
        col.add(ext)
        s.add(col)

    cases.append(FuzzCase(
        name="csg_extrude_polygon_notch",
        suite="csg",
        description="L-shaped polygon Extrude with CSG box notch cut",
        builder=case_extrude_notch,
        expected_vol_min=1_179_990_000.0,
        expected_vol_max=1_180_010_000.0,
    ))

    # 4. CSG Union cross junction
    def case_union_cross(s: Storey):
        col = Column(name="col")
        b1 = Box(start=Point(x=-1000, y=-200, z=0), end=Point(x=1000, y=200, z=400), name="arm_x")
        b1.union(Box(start=Point(x=-200, y=-1000, z=0), end=Point(x=200, y=1000, z=400), name="arm_y"))
        col.add(b1)
        s.add(col)

    # Vol = 2000*400*400 + 2000*400*400 - 400*400*400 = 320M + 320M - 64M = 576M
    cases.append(FuzzCase(
        name="csg_union_cross_junction",
        suite="csg",
        description="Orthogonal cross-junction union of two beams",
        builder=case_union_cross,
        expected_vol_min=575_990_000.0,
        expected_vol_max=576_010_000.0,
    ))

    # 5. CSG Intersection partial overlap
    def case_intersection(s: Storey):
        col = Column(name="col")
        b1 = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000), name="box1")
        b1.intersection(Box(start=Point(x=500, y=500, z=500), end=Point(x=1500, y=1500, z=1500), name="box2"))
        col.add(b1)
        s.add(col)

    cases.append(FuzzCase(
        name="csg_intersection_partial",
        suite="csg",
        description="Intersection of two partially overlapping boxes",
        builder=case_intersection,
        expected_vol_min=124_999_000.0,
        expected_vol_max=125_001_000.0,
    ))

    return cases


def build_suite_mesh() -> list[FuzzCase]:
    """Test suite covering Mesh boolean cuts and composite polyhedra."""
    cases = []

    # 1. Watertight cube Mesh cut by Box
    def case_mesh_cut_box(s: Storey):
        col = Column(name="col")
        m = _make_watertight_cube_mesh(name="cube", size=1000.0, center=(500, 500, 500))
        m.difference(Box(start=Point(x=0, y=0, z=0), end=Point(x=500, y=1000, z=1000), name="half_cut"))
        col.add(m)
        s.add(col)

    cases.append(FuzzCase(
        name="mesh_cube_cut_box",
        suite="mesh",
        description="Closed watertight Mesh cube cut in half by Box",
        builder=case_mesh_cut_box,
        expected_vol_min=499_990_000.0,
        expected_vol_max=500_010_000.0,
    ))

    # 2. Triangular prism Mesh cut by Box
    def case_prism_cut_box(s: Storey):
        col = Column(name="col")
        verts = [
            Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0), Point(x=0, y=1000, z=0),
            Point(x=0, y=0, z=1000), Point(x=1000, y=0, z=1000), Point(x=0, y=1000, z=1000),
        ]
        faces = [
            (0, 2, 1),  # bottom
            (3, 4, 5),  # top
            (0, 1, 4), (0, 4, 3),  # front
            (1, 2, 5), (1, 5, 4),  # hypotenuse
            (2, 0, 3), (2, 3, 5),  # side
        ]
        prism = Mesh(name="prism", vertices=verts, faces=faces)
        prism.difference(Box(start=Point(x=-100, y=-100, z=500), end=Point(x=200, y=200, z=1100), name="notch"))
        col.add(prism)
        s.add(col)

    # Prism area = 0.5 * 1000 * 1000 = 500,000; depth = 1000 -> 500,000,000 mm3
    # Notch = 200 * 200 * 500 = 20,000,000 mm3 -> Expected: 480,000,000 mm3
    cases.append(FuzzCase(
        name="mesh_triangular_prism_cut",
        suite="mesh",
        description="Watertight triangular prism Mesh with corner notch cut",
        builder=case_prism_cut_box,
        expected_vol_min=479_990_000.0,
        expected_vol_max=480_010_000.0,
    ))

    # 3. Watertight cube Mesh with void opening
    def case_mesh_void(s: Storey):
        col = Column(name="col")
        m = _make_watertight_cube_mesh(name="cube", size=1000.0, center=(500, 500, 500))
        col.add(m)
        col.void(Box(start=Point(x=300, y=300, z=-100), end=Point(x=700, y=700, z=1100), name="void"))
        s.add(col)

    # 1000^3 - 400 * 400 * 1000 = 1,000,000,000 - 160,000,000 = 840,000,000 mm3
    cases.append(FuzzCase(
        name="mesh_cube_void_penetration",
        suite="mesh",
        description="Watertight Mesh cube with through-opening void",
        builder=case_mesh_void,
        expected_vol_min=839_990_000.0,
        expected_vol_max=840_010_000.0,
    ))

    # 4. Inverted normals detection (reversed face winding)
    def case_inverted_mesh(s: Storey):
        col = Column(name="col")
        corners = [
            Point(x=-500, y=-500, z=-500), Point(x=500, y=-500, z=-500),
            Point(x=500, y=500, z=-500), Point(x=-500, y=500, z=-500),
            Point(x=-500, y=-500, z=500), Point(x=500, y=-500, z=500),
            Point(x=500, y=500, z=500), Point(x=-500, y=500, z=500),
        ]
        # Inverted faces (reversed winding)
        faces = [
            (0, 2, 3), (0, 1, 2), (4, 6, 5), (4, 7, 6),
            (0, 5, 1), (0, 4, 5), (2, 7, 3), (2, 6, 7),
            (0, 7, 4), (0, 3, 7), (1, 6, 2), (1, 5, 6),
        ]
        m = Mesh(name="inverted_cube", vertices=corners, faces=faces)
        col.add(m)
        s.add(col)

    cases.append(FuzzCase(
        name="mesh_inverted_cube_detection",
        suite="mesh",
        description="Mesh with inside-out face winding: asserts fuzzer flags INVERTED via check_mesh_signed_volume",
        builder=case_inverted_mesh,
        is_known_bug=True,  # Expected to report INVERTED
        known_bug_ref="Intentional inverted input to verify divergence test detection",
    ))

    return cases


def build_suite_multi_cut() -> list[FuzzCase]:
    """Test suite covering deep chained CSG operations (depth 3 to 5)."""
    cases = []

    # 1. 3 cascading corner cuts
    def case_chain_3(s: Storey):
        col = Column(name="col")
        b = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000), name="box")
        b.difference(Box(start=Point(x=-100, y=-100, z=600), end=Point(x=300, y=300, z=1100), name="cut1"))
        b.difference(Box(start=Point(x=700, y=-100, z=600), end=Point(x=1100, y=300, z=1100), name="cut2"))
        b.difference(Box(start=Point(x=300, y=700, z=600), end=Point(x=700, y=1100, z=1100), name="cut3"))
        col.add(b)
        s.add(col)

    cases.append(FuzzCase(
        name="chain_3_corner_cuts",
        suite="multi_cut",
        description="3 successive corner cuts on top face",
        builder=case_chain_3,
    ))

    # 2. 5 stepped cuts along edge
    def case_chain_5(s: Storey):
        col = Column(name="col")
        b = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000), name="box")
        for i in range(5):
            z_step = 200 * (i + 1)
            b.difference(Box(start=Point(x=200 * i, y=-100, z=z_step), end=Point(x=200 * (i + 1), y=500, z=1100), name=f"step_{i}"))
        col.add(b)
        s.add(col)

    cases.append(FuzzCase(
        name="chain_5_stepped_cuts",
        suite="multi_cut",
        description="5 sequential stepped cuts along an edge",
        builder=case_chain_5,
    ))

    return cases


def build_random_permutation_cases(num_cases: int = 20, seed: int = 42) -> list[FuzzCase]:
    """Generate randomized pseudo-random geometry and boolean combinations."""
    rng = np.random.default_rng(seed)
    cases = []

    for idx in range(num_cases):
        op_choice = rng.choice(["diff_box", "diff_corner", "void_middle", "touching_union"])
        size_x = int(rng.integers(800, 2000))
        size_y = int(rng.integers(800, 2000))
        size_z = int(rng.integers(1000, 3000))

        if op_choice == "diff_box":
            hole_w = int(rng.integers(200, size_x // 2))
            hole_d = int(rng.integers(200, size_y // 2))
            cx = size_x // 2
            cy = size_y // 2

            def builder(s: Storey, sx=size_x, sy=size_y, sz=size_z, hw=hole_w, hd=hole_d, cx=cx, cy=cy):
                col = Column(name="rand_col")
                b = Box(start=Point(x=0, y=0, z=0), end=Point(x=sx, y=sy, z=sz), name="base")
                b.difference(Box(start=Point(x=cx - hw // 2, y=cy - hd // 2, z=-100),
                                 end=Point(x=cx + hw // 2, y=cy + hd // 2, z=sz + 100), name="tool"))
                col.add(b)
                s.add(col)

            exp_vol = float(size_x * size_y * size_z - hole_w * hole_d * size_z)
            cases.append(FuzzCase(
                name=f"rand_diff_through_{idx:02d}",
                suite="random",
                description=f"Random through-hole cut: Box({size_x}x{size_y}x{size_z}) with hole({hole_w}x{hole_d})",
                builder=builder,
                expected_vol_min=exp_vol * 0.99,
                expected_vol_max=exp_vol * 1.01,
            ))

        elif op_choice == "diff_corner":
            cut_x = int(rng.integers(100, size_x // 2))
            cut_y = int(rng.integers(100, size_y // 2))
            cut_z = int(rng.integers(100, size_z // 2))

            def builder(s: Storey, sx=size_x, sy=size_y, sz=size_z, cx=cut_x, cy=cut_y, cz=cut_z):
                col = Column(name="rand_col")
                b = Box(start=Point(x=0, y=0, z=0), end=Point(x=sx, y=sy, z=sz), name="base")
                b.difference(Box(start=Point(x=-50, y=-50, z=sz - cz), end=Point(x=cx, y=cy, z=sz + 50), name="tool"))
                col.add(b)
                s.add(col)

            exp_vol = float(size_x * size_y * size_z - cut_x * cut_y * cut_z)
            cases.append(FuzzCase(
                name=f"rand_diff_corner_{idx:02d}",
                suite="random",
                description=f"Random corner notch: Box({size_x}x{size_y}x{size_z}) cut ({cut_x}x{cut_y}x{cut_z})",
                builder=builder,
                expected_vol_min=exp_vol * 0.99,
                expected_vol_max=exp_vol * 1.01,
            ))

        elif op_choice == "void_middle":
            # Test bar void
            diam = int(rng.choice([16, 20, 25, 32]))
            h_len = int(rng.integers(1500, 3000))
            void_z0 = int(rng.integers(400, h_len // 2))
            void_z1 = int(rng.integers(void_z0 + 200, h_len - 200))

            def builder(s: Storey, diam=diam, h_len=h_len, vz0=void_z0, vz1=void_z1):
                col = Column(name="rand_col")
                col.add(Bar(path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=h_len)], diameter=diam, name="bar"))
                col.void(Box(start=Point(x=-100, y=-100, z=vz0), end=Point(x=100, y=100, z=vz1), name="void"))
                s.add(col)

            surviving_len = float(void_z0 + (h_len - void_z1))
            exp_vol = math.pi * ((diam / 2.0) ** 2) * surviving_len
            cases.append(FuzzCase(
                name=f"rand_bar_void_{idx:02d}",
                suite="random",
                description=f"Random Bar(d={diam}, L={h_len}) with void Z[{void_z0}, {void_z1}]",
                builder=builder,
                expected_vol_min=exp_vol * 0.95,
                expected_vol_max=exp_vol * 1.05,
                is_known_bug=True,
                known_bug_ref="https://github.com/IfcOpenShell/IfcOpenShell/issues/9256",
            ))

        else:
            # Touching union
            w2 = int(rng.integers(300, 1000))

            def builder(s: Storey, sx=size_x, sy=size_y, sz=size_z, w2=w2):
                col = Column(name="rand_col")
                b1 = Box(start=Point(x=0, y=0, z=0), end=Point(x=sx, y=sy, z=sz), name="base")
                b1.union(Box(start=Point(x=sx, y=0, z=0), end=Point(x=sx + w2, y=sy, z=sz), name="tool"))
                col.add(b1)
                s.add(col)

            exp_vol = float((size_x + w2) * size_y * size_z)
            cases.append(FuzzCase(
                name=f"rand_union_touching_{idx:02d}",
                suite="random",
                description=f"Random coplanar touching union: ({size_x}+{w2})x{size_y}x{size_z}",
                builder=builder,
                expected_vol_min=exp_vol * 0.99,
                expected_vol_max=exp_vol * 1.01,
            ))

    return cases


# ---------------------------------------------------------------------------
# Fuzz Execution Engine
# ---------------------------------------------------------------------------

def run_fuzz_case(case: FuzzCase) -> CaseResult:
    """Execute a single fuzz case and perform all hold-water assertions."""
    t0 = time.perf_counter()
    res = CaseResult(case=case, duration_ms=0.0)

    # 1. Build Project
    proj = Project(name=f"fuzz_{case.name}")
    storey = Storey(name="ground", elevation=0)
    try:
        case.builder(storey)
    except Exception as exc:
        res.status = "ERROR_BUILD"
        res.error = f"Builder raised: {exc}"
        res.duration_ms = (time.perf_counter() - t0) * 1000.0
        return res
    proj.add_storey(storey)

    # 2. Compile to IFC
    try:
        normalized = normalize_project_to_meters(proj)
        ifc_res = generate_ifc(normalized)
        if not ifc_res.success:
            res.status = "FAIL_GEN"
            res.error = f"generate_ifc failed: {ifc_res.error}"
            res.duration_ms = (time.perf_counter() - t0) * 1000.0
            return res
        res.ifc_content = ifc_res.ifc_content
    except Exception as exc:
        res.status = "CRASH_GEN"
        res.error = f"generate_ifc crashed: {exc}"
        res.duration_ms = (time.perf_counter() - t0) * 1000.0
        return res

    # 3. Tessellate with IfcOpenShell
    try:
        model = ifcopenshell.file.from_string(res.ifc_content)
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)

        products = [
            p for p in model.by_type("IfcProduct")
            if getattr(p, "Representation", None)
            and not p.is_a("IfcAnnotation")
            and not p.is_a("IfcOpeningElement")
        ]

        if not products:
            res.status = "FAIL_NO_PRODUCTS"
            res.error = "IFC model contained no products with representation"
            res.duration_ms = (time.perf_counter() - t0) * 1000.0
            return res

        for prod in products:
            prod_name = prod.Name or getattr(prod, "GlobalId", "prod")
            prod_type = prod.is_a()

            try:
                shape = ifcopenshell.geom.create_shape(settings, prod)
            except Exception as exc:
                res.products.append(ProductResult(
                    product_name=prod_name,
                    product_type=prod_type,
                    vertex_count=0,
                    face_count=0,
                    is_watertight=False,
                    open_edges=-1,
                    signed_volume=0.0,
                    status="CRASH_TESSELLATION",
                    error_message=str(exc),
                ))
                continue

            raw_verts = getattr(shape.geometry, "verts", None) or []
            raw_faces = getattr(shape.geometry, "faces", None) or []

            # Vertices in meters -> convert to mm
            v_arr = np.array(raw_verts, dtype=np.float64).reshape(-1, 3) * 1000.0
            f_arr = np.array(raw_faces, dtype=np.int32).reshape(-1, 3)
            num_v = len(v_arr)
            num_f = len(f_arr)

            # Check 1: Non-emptiness
            if num_v == 0 or num_f == 0:
                res.products.append(ProductResult(
                    product_name=prod_name,
                    product_type=prod_type,
                    vertex_count=0,
                    face_count=0,
                    is_watertight=False,
                    open_edges=0,
                    signed_volume=0.0,
                    status="EMPTY_SHAPE",
                    error_message="Shape has 0 vertices/faces (geometry vanished)",
                ))
                continue

            # Check 2: Watertightness (0 boundary edges)
            mesh_view = MeshDuckView(vertices=v_arr, faces=f_arr)
            is_wt, open_edges = check_mesh_watertightness(mesh_view)

            # Check 3: Signed Volume (Gauss Divergence Theorem)
            vol = check_mesh_signed_volume(mesh_view)

            # Determine product status
            prod_status = "PASS"
            err_msg = ""

            if not is_wt:
                prod_status = "LEAK"
                err_msg = f"Mesh is not topologically closed ({open_edges} open boundary edges)"
            elif vol < -1e-4:
                prod_status = "INVERTED"
                err_msg = f"Mesh has negative signed volume ({vol:,.0f} mm3) — inverted face normals"
            elif abs(vol) < 1e-4:
                prod_status = "DEGENERATE"
                err_msg = "Mesh has zero volume (collapsed solid)"
            elif case.expected_vol_min is not None and vol < case.expected_vol_min:
                prod_status = "DISCARDED_MATERIAL"
                err_msg = f"Measured volume ({vol:,.0f} mm3) < expected min ({case.expected_vol_min:,.0f} mm3)"
            elif case.expected_vol_max is not None and vol > case.expected_vol_max:
                prod_status = "PHANTOM_MATERIAL"
                err_msg = f"Measured volume ({vol:,.0f} mm3) > expected max ({case.expected_vol_max:,.0f} mm3)"

            res.products.append(ProductResult(
                product_name=prod_name,
                product_type=prod_type,
                vertex_count=num_v,
                face_count=num_f,
                is_watertight=is_wt,
                open_edges=open_edges,
                signed_volume=vol,
                status=prod_status,
                error_message=err_msg,
            ))

    except Exception as exc:
        res.status = "CRASH_IFCOPENSHELL"
        res.error = f"IfcOpenShell evaluation crashed: {exc}"
        res.duration_ms = (time.perf_counter() - t0) * 1000.0
        return res

    res.duration_ms = (time.perf_counter() - t0) * 1000.0

    # Aggregate case status from products
    statuses = [p.status for p in res.products if not p.product_name.startswith("opening:")]
    if not statuses:
        statuses = [p.status for p in res.products]

    if any(s in ("CRASH_TESSELLATION", "EMPTY_SHAPE") for s in statuses):
        res.status = "EMPTY_OR_CRASH"
    elif any(s == "DISCARDED_MATERIAL" for s in statuses):
        res.status = "DISCARDED_MATERIAL"
    elif any(s == "LEAK" for s in statuses):
        res.status = "LEAK"
    elif any(s == "INVERTED" for s in statuses):
        res.status = "INVERTED"
    elif all(s == "PASS" for s in statuses):
        res.status = "PASS"
    else:
        res.status = statuses[0]

    return res


# ---------------------------------------------------------------------------
# CLI & Reporter
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fuzz IfcOpenShell geometry and boolean operations using check_mesh_signed_volume")
    parser.add_argument("--suite", choices=["all", "swept_disk", "coplanar", "csg", "mesh", "multi_cut", "random"], default="all",
                        help="Suite to run (default: all)")
    parser.add_argument("--iterations", type=int, default=15, help="Number of random permutation cases (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to save failing IFC and repro files")
    parser.add_argument("--strict", action="store_true", help="Fail non-zero even on known upstream bugs like #9256")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed product breakdown for every case")
    args = parser.parse_args()

    # Collect test cases
    all_cases: list[FuzzCase] = []
    if args.suite in ("all", "swept_disk"):
        all_cases.extend(build_suite_swept_disk())
    if args.suite in ("all", "coplanar"):
        all_cases.extend(build_suite_coplanar())
    if args.suite in ("all", "csg"):
        all_cases.extend(build_suite_csg())
    if args.suite in ("all", "mesh"):
        all_cases.extend(build_suite_mesh())
    if args.suite in ("all", "multi_cut"):
        all_cases.extend(build_suite_multi_cut())
    if args.suite in ("all", "random"):
        all_cases.extend(build_random_permutation_cases(num_cases=args.iterations, seed=args.seed))

    print(f"\n{'=' * 80}")
    print(" IfcOpenShell 'Hold Water' Fuzzing Test (Gauss Divergence + Watertightness)")
    print(f" Cases: {len(all_cases)} | Suite: {args.suite} | Seed: {args.seed}")
    print(f"{'=' * 80}\n")

    out_path = Path(args.out_dir) if args.out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    passed_count = 0
    known_bug_count = 0
    new_bug_count = 0
    total_time = 0.0

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    header = f"{'Case Name':<32} {'Suite':<12} {'Watertight':<11} {'Signed Vol (mm3)':<18} {'Status':<18} {'Time'}"
    print(header)
    print("-" * len(header))

    for case in all_cases:
        res = run_fuzz_case(case)
        total_time += res.duration_ms

        primary_prod = next((p for p in res.products if not p.product_name.startswith("opening:")), None)
        if primary_prod is None and res.products:
            primary_prod = res.products[0]

        wt_str = "YES" if primary_prod and primary_prod.is_watertight else "NO"
        vol_str = f"{primary_prod.signed_volume:,.0f}" if primary_prod else "N/A"

        if res.status == "PASS":
            status_display = "[PASS]"
            passed_count += 1
        elif case.is_known_bug:
            status_display = f"[KNOWN_BUG:{res.status}]"
            known_bug_count += 1
        else:
            status_display = f"[{res.status}]"
            new_bug_count += 1

        print(f"{case.name:<32} {case.suite:<12} {wt_str:<11} {vol_str:<18} {status_display:<18} {res.duration_ms:5.1f}ms")

        if args.verbose or (res.status != "PASS" and not case.is_known_bug):
            for p in res.products:
                prefix = "  |-" if p.status == "PASS" else "  [!]"
                print(f"{prefix} {p.product_name} ({p.product_type}): verts={p.vertex_count}, faces={p.face_count}, wt={p.is_watertight}, vol={p.signed_volume:,.0f} mm3 | {p.error_message or 'OK'}")

        # Dump reproducer files for failures
        if out_path and res.status != "PASS":
            case_dir = out_path / case.name
            case_dir.mkdir(exist_ok=True)
            if res.ifc_content:
                (case_dir / f"{case.name}.ifc").write_text(res.ifc_content)

    print("\n" + "=" * 80)
    print(" Fuzzing Summary")
    print(f" Total Cases:          {len(all_cases)}")
    print(f" Passed (Holds Water): {passed_count}")
    print(f" Known Bugs:           {known_bug_count}  (e.g. Issue #9256: IfcSweptDiskSolid voids)")
    print(f" New Failures:         {new_bug_count}")
    print(f" Total Wall Time:      {total_time:,.1f} ms  (avg {total_time / len(all_cases):.1f} ms/case)")
    print("=" * 80 + "\n")

    if new_bug_count > 0 or (args.strict and known_bug_count > 0):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
