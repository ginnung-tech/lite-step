"""Lite-STEP project script."""
# ruff: noqa: F401 — the import block below is the full DSL menu, kept intact
# by design: unused names are a feature (later tasks use them), never lint
# errors, and never to be pruned.
# Resolve ``lite_step`` from THIS checkout, not from whatever is installed.
# ``python examples/x.py`` puts ``examples/`` on sys.path — NOT the repo root —
# so without these two lines the import falls through to any other copy on the
# machine and compiles against a different version of the DSL, silently.
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))

import math
import numpy as np
from lite_step import compile_main
from lite_step.models import (Project, Storey, Point, Point2D,
    Wall, Window, Door, Slab, Roof, Column, Beam, Space, Site, Element,
    Box, Extrude, Sweep, Sheet, Pipe, Revolve, Mesh, Bar, Material, LayerSet,
    ReferencePoint, GuideLine, Transform, Anchor)
from lite_step.helpers import arc_path, stirrup_path
from lite_step.compiler.placement import compose_rotations
from lite_step.materials.dk import MATERIALS   # noqa: F401 — stock introspection, e.g. MATERIALS["Timber_C24"].geometry.stock_profiles_mm

I = lambda x: int(round(x))  # noqa: E731,E741


def generate_project():
    proj = Project(name="Concrete Column Chain")
    
    np.random.seed(42)  # Monte Carlo random walk seed
    pos = np.array([0.0, 0.0, 0.0])
    direction = np.array([700.0, 0.0, 0.0])
    
    # 20 cm x 20 cm cross-section profile centered on the 70 cm path
    prof = [
        Point2D(x=-100, y=-100),
        Point2D(x=100, y=-100),
        Point2D(x=100, y=100),
        Point2D(x=-100, y=100),
    ]

    cols = []
    for i in range(20):
        next_pos = pos + direction
        p_start = Point(x=I(pos[0]), y=I(pos[1]), z=I(pos[2]))
        p_end = Point(x=I(next_pos[0]), y=I(next_pos[1]), z=I(next_pos[2]))
        
        col = Column(name=f"block_{i:02d}")
        col.add(Sweep(
            name="body",
            profile=prof,
            path=[p_start, p_end],
            material=Material(key="Concrete_C25-30")
        ))
        cols.append(col)
        
        # Monte Carlo random walk: 0-35 deg rotation in 3D for the next block
        angles_rad = np.radians(np.random.uniform(-35, 35, size=3))
        ax, ay, az = angles_rad
        
        # 3D Euler rotation matrix
        rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
        ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
        rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
        rot = rz @ ry @ rx
        
        pos = next_pos
        direction = rot @ direction

    # Sequentially chain-add each column to the previous one so they carve each other
    for i in range(len(cols) - 1):
        cols[i].add(cols[i + 1])

    proj.add(cols[0])
    return proj


result = generate_project()


if __name__ == "__main__":
    compile_main()
