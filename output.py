"""Lite-STEP project script."""
# ruff: noqa: F401 — the import block below is the full DSL menu, kept intact
# by design: unused names are a feature (later tasks use them), never lint
# errors, and never to be pruned.
import math
import numpy as np
from lite_step import compile_main
from lite_step.models import (Project, Storey, Point, Point2D,
    Wall, Window, Door, Slab, Roof, Column, Beam, Space, Site, Element,
    Box, Extrude, Sweep, Sheet, Pipe, Revolve, Mesh, Bar, Material, LayerSet,
    ReferencePoint, GuideLine, Transform, Anchor)
from lite_step.helpers import arc_path, stirrup_path
from lite_step.materials.dk import MATERIALS   # noqa: F401 — stock introspection, e.g. MATERIALS["Timber_C24"].geometry.stock_profiles_mm

I = lambda x: int(round(x))  # noqa: E731,E741


def generate_project():
    proj = Project(name="New Project")
    # YOUR CODE HERE
    return proj


result = generate_project()


if __name__ == "__main__":
    compile_main()
