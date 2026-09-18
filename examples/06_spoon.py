"""Continuous Watertight Silver Spoon Mesh Example.

Demonstrates:
- Continuous parametric surface design using custom mathematical manifold loops.
- Seamless Hermite smoothstep blending from a thin concave scoop bowl into a solid ergonomic handle.
- Watertight vertex-face topology with correct outward surface normals.
- Micrometer-precision scaling within the integer-millimeter STEP coordinate space.

Compile with::

    lite-step examples/06_spoon.py
    # or:
    python examples/06_spoon.py
"""
import math
from pathlib import Path
import sys
import numpy as np

# Ensure repo root is in sys.path when run directly
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from lite_step import compile_main  # noqa: E402
from lite_step.models import (  # noqa: E402
    Element,
    Mesh,
    Point,
    Project,
)

I = lambda x: int(round(x))  # noqa: E731,E741


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite smoothstep interpolation."""
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return float(t * t * (3.0 - 2.0 * t))


def build_spoon_mesh(Nu: int = 160, Nv: int = 64) -> Mesh:
    """Generate a smooth, continuous, watertight 3D triangle mesh of a silver spoon.

    Scaled 1000x to give micrometer-level integer precision (1 unit = 1 um in real scale):
    - Total length: 380,000 mm along X (380 mm scaled 1000x)
    - Bowl: 130,000 mm long, 84,000 mm wide, 32,000 mm deep concave scoop
    - Neck: slender transition waist (14,000 mm wide, 7,000 mm thick)
    - Handle: ergonomic flared grip (31,000 mm wide) with longitudinal S-curvature
    """
    SCALE = 1000.0
    L_bowl = 130.0 * SCALE
    L_total = 380.0 * SCALE

    def evaluate_point(u: float, v: float) -> tuple[int, int, int]:
        x = u * L_total

        # Smooth transition from bowl shell into handle solid
        w_handle = smoothstep(105.0 * SCALE, 140.0 * SCALE, x)
        w_bowl = 1.0 - w_handle

        # ── 1. Bowl Surface (True Continuous Concave U-Loop) ──
        tb = np.clip(x / L_bowl, 0.001, 0.999)
        bw_max = 42.0 * SCALE * (math.sin(tb * math.pi) ** 0.42) * (1.12 - 0.22 * tb)
        z_rim = (8.0 * (1.0 - tb) ** 1.8 - 4.0 * math.sin(tb * math.pi)) * SCALE
        b_depth = 32.0 * SCALE * (math.sin(tb * math.pi) ** 0.65)
        thickness = 2.8 * SCALE

        # Continuous closed cross-section loop around v in [0, 2*pi]:
        # y(v) = bw_max * cos(v) continuously walks:
        #   v in [0, pi]: outer bottom hull from right (+w) to left (-w)
        #   v in [pi, 2*pi]: inner top scoop from left (-w) to right (+w)
        by = bw_max * math.cos(v)

        if v <= math.pi:
            # Outer convex bottom hull
            bz = z_rim - b_depth * (math.sin(v) ** 1.3)
        else:
            # Inner concave top scoop cavity
            inner_depth = max(0.0, b_depth - thickness)
            bz = z_rim - inner_depth * ((-math.sin(v)) ** 1.3)

        # ── 2. Handle & Neck Surface (Ergonomic Flared Bar) ──
        th = np.clip((x - 105.0 * SCALE) / (L_total - 105.0 * SCALE), 0.0, 1.0)
        hz_spine = (1.0 + 18.0 * math.sin(th * math.pi * 0.9) - 6.0 * th) * SCALE

        if th < 0.15:
            tn = th / 0.15
            hw = (14.0 * (1.0 - tn) + 7.0 * tn) * SCALE
        elif th < 0.85:
            tf = (th - 0.15) / 0.70
            hw = (7.0 + 8.5 * (math.sin(tf * math.pi * 0.5) ** 1.3)) * SCALE
        else:
            te = (th - 0.85) / 0.15
            hw = (15.5 * math.sqrt(max(0.0, 1.0 - te**2))) * SCALE

        h_thick = (3.5 - 1.2 * th) * SCALE
        hy = hw * math.cos(v)
        hz = hz_spine + h_thick * math.sin(v)

        # ── 3. Smooth Surface Blend ──
        fx = x
        fy = by * w_bowl + hy * w_handle
        fz = bz * w_bowl + hz * w_handle

        return I(fx), I(fy), I(fz)

    u_vals = np.linspace(0.001, 0.999, Nu)
    v_vals = np.linspace(0, 2 * math.pi, Nv, endpoint=False)

    verts: list[Point] = []

    # Front tip vertex (u = 0)
    p0_x, p0_y, p0_z = evaluate_point(0.0, 0.0)
    verts.append(Point(x=p0_x, y=p0_y, z=p0_z))
    front_idx = 0

    # Intermediate ring grid
    grid: list[list[int]] = []
    for u in u_vals:
        row: list[int] = []
        for v in v_vals:
            px, py, pz = evaluate_point(u, v)
            row.append(len(verts))
            verts.append(Point(x=px, y=py, z=pz))
        grid.append(row)

    # Rear tip vertex (u = 1)
    p1_x, p1_y, p1_z = evaluate_point(1.0, 0.0)
    verts.append(Point(x=p1_x, y=p1_y, z=p1_z))
    rear_idx = len(verts) - 1

    faces: list[tuple[int, int, int]] = []

    # Front cap triangle fan
    for j in range(Nv):
        j_next = (j + 1) % Nv
        faces.append((front_idx, grid[0][j], grid[0][j_next]))

    # Body surface quads (consistent outward normal winding)
    for i in range(Nu - 1):
        for j in range(Nv):
            j_next = (j + 1) % Nv
            v00 = grid[i][j]
            v01 = grid[i][j_next]
            v10 = grid[i + 1][j]
            v11 = grid[i + 1][j_next]
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))

    # Rear cap triangle fan
    for j in range(Nv):
        j_next = (j + 1) % Nv
        faces.append((rear_idx, grid[-1][j_next], grid[-1][j]))

    return Mesh(
        name="silver_spoon_body",
        vertices=verts,
        faces=faces,
        is_watertight=True,
        color="#D0D4D8",  # Polished silver / stainless steel
    )


def generate_project() -> Project:
    proj = Project(name="Monumental Smooth Silver Spoon")

    spoon = Element(ifc_class="IfcBuildingElementProxy", name="silver_spoon")
    spoon.add(build_spoon_mesh(Nu=160, Nv=64))

    proj.add(spoon)
    return proj


result = generate_project()


if __name__ == "__main__":
    compile_main()
