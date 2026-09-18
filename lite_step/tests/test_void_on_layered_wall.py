"""A gable cut with `.void()` on a wall carrying `layers=`.

Void cuts are the chosen gable route (decided), which puts this
interaction on the shipping path. Slicing (compiler/layers.py) and voids
(displacement) are separate passes and neither mentions the other, so whether
a void reaches all N layer slices or only the pre-slice body was never
established. This test records the answer instead of assuming it.

Measured as a VOLUME RATIO rather than an entity count: counting
IfcOpeningElement tells you a void exists, not that it reached the
insulation. The ratio is also unit-free, so it doesn't depend on whether the
kernel reports mm3 or m3.
"""
from __future__ import annotations

import numpy as np
import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.geom  # noqa: E402

from lite_step.ifc.generator import generate_ifc  # noqa: E402
from lite_step.models import (Box, LayerSet, Material, Point,  # noqa: E402
                              Project, Wall)
from lite_step.models.project import Storey  # noqa: E402

# Wall 6000 x 300 x 3600. The gable void spans the FULL depth, so every layer
# is in its path: x 3000..6000 (3000) * y 0..300 (300) * z 2400..3600 (1200).
WALL = 6000 * 300 * 3600
CUT_ALL_LAYERS = 3000 * 300 * 1200          # the void meets all three slices
CUT_OUTER_ONLY = 3000 * 108 * 1200          # ...only the 108 mm brick leaf


def _wall(with_void: bool) -> Project:
    proj = Project(name="gable")
    wall = Wall(name="south")
    wall.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=6000, y=300, z=3600)))
    wall.layers = LayerSet(name="t", outward=(0, -1, 0), layers=[
        Material(key="Brick_Red_DK", thickness_mm=108),
        Material(key="MineralWool_Facade34", thickness_mm=150),
        Material(key="Gypsum_Standard", thickness_mm=42),
    ])
    if with_void:
        wall.void(Box(start=Point(x=3000, y=-100, z=2400),
                      end=Point(x=6100, y=400, z=3700)))
    s = Storey(elevation=0)
    s.add(wall)
    proj.add_storey(s)
    return proj


def _wall_volume(proj) -> float:
    res = generate_ifc(proj, source_code=None)
    assert res.success, res.error
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        with open(path, "w") as f:
            f.write(res.ifc_content)
        model = ifcopenshell.open(path)
        settings = ifcopenshell.geom.settings()
        total = 0.0
        for prod in model.by_type("IfcWall"):
            sh = ifcopenshell.geom.create_shape(settings, prod)
            v = np.array(sh.geometry.verts).reshape(-1, 3)
            fcs = np.array(sh.geometry.faces).reshape(-1, 3)
            t = v[fcs]
            total += abs(np.einsum('ij,ij->i', t[:, 0],
                                   np.cross(t[:, 1], t[:, 2])).sum() / 6.0)
        return total
    finally:
        os.unlink(path)


def test_void_reaches_every_layer_slice():
    full = _wall_volume(_wall(with_void=False))
    cut = _wall_volume(_wall(with_void=True))
    assert full > 0, "the layered wall emitted no volume at all"
    removed = (full - cut) / full

    expected_all = CUT_ALL_LAYERS / WALL      # 0.1667
    expected_outer = CUT_OUTER_ONLY / WALL    # 0.0600

    assert abs(removed - expected_all) < 0.01, (
        f"a gable void removed {removed:.4f} of the wall; all-layers would be "
        f"{expected_all:.4f} and outer-leaf-only {expected_outer:.4f}. "
        f"If this is near the outer-leaf figure, the void carved the brick and "
        f"left insulation and gypsum standing full height — a gable that looks "
        f"right in elevation and is wrong in section. Void cuts are the "
        f"chosen gable route, so this path has to hold."
    )
