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
    ReferencePoint, GuideLine, Transform, Anchor, Product)
from lite_step.helpers import arc_path, stirrup_path
from lite_step.compiler.placement import compose_rotations
from lite_step.materials.dk import MATERIALS   # noqa: F401 — stock introspection, e.g. MATERIALS["Timber_C24"].geometry.stock_profiles_mm

I = lambda x: int(round(x))  # noqa: E731,E741


# Define unplaced source element and promote to Product catalog item
_source_col = Column()
_source_col.add(Box(
    name="body",
    start=Point(x=0, y=0, z=0),
    end=Point(x=700, y=200, z=200),
    material=Material(key="Concrete_C25-30")
))
CONCRETE_BLOCK = Product(_source_col, name="concrete_700x200x200")


def generate_project():
    proj = Project(name="Concrete Column Product Chain")
    
    np.random.seed(42)  # Monte Carlo random walk seed

    # First occurrence placed at origin
    col_0 = CONCRETE_BLOCK.occurrence(name="block_00")
    proj.add(col_0)

    # Subsequent occurrences anchored to preceding block in local coordinates
    for i in range(1, 20):
        # 0-35 deg random rotation in all 3 local axes (centidegrees)
        rx, ry, rz = [I(np.random.uniform(-35, 35) * 100) for _ in range(3)]
        
        col = CONCRETE_BLOCK.occurrence(name=f"block_{i:02d}")
        col.placement = Anchor(
            host=f"column:block_{i-1:02d}",
            attach_to="end",
            rotations=[("x", rx), ("y", ry), ("z", rz)]
        )
        proj.add(col)

    return proj


result = generate_project()


if __name__ == "__main__":
    compile_main()

