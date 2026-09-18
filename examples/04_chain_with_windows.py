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


# ── Murermestervilla Window Specification ──────────────────────────────────
KARM_DEPTH = 120
KARM_FACE = 45
POST_DEPTH = 120
POST_FACE = 45
SASH_DEPTH = 70
SASH_FACE = 45
SASH_LAP = 12
FIT_GAP = 10
BAR_WIDTH = 24
BAR_PROUD = 12
PANE_ROWS = 3
GLAZING = "Glass_Monolithic"
PANE_MM = 4
GAS_GAP_MM = 16
IGU_MM = 3 * PANE_MM + 2 * GAS_GAP_MM
GLASS_BITE = 10
JOINERY = "Timber_C24"


def _frame_loop(x0, x1, z0, z1, y):
    return [Point(x=I(x0), y=I(y), z=I(z0)), Point(x=I(x1), y=I(y), z=I(z0)),
            Point(x=I(x1), y=I(y), z=I(z1)), Point(x=I(x0), y=I(y), z=I(z1)),
            Point(x=I(x0), y=I(y), z=I(z0))]


def _bar_profile():
    hx, hy = BAR_PROUD // 2, BAR_WIDTH // 2
    return [Point2D(x=-hx, y=-hy), Point2D(x=hx, y=-hy),
            Point2D(x=hx, y=hy), Point2D(x=-hx, y=hy)]


def build_window(width, height, name):
    kh, ph, sh = KARM_FACE // 2, POST_FACE // 2, SASH_FACE // 2
    win = Window(name=name, width=I(width), height=I(height), style="casement", panes=2)
    y_karm = 0
    k0 = FIT_GAP + kh
    k1 = FIT_GAP + 2 * kh
    win.add(Sweep(name="karm",
                  material=Material(key=JOINERY, profile_mm=(KARM_DEPTH, KARM_FACE)),
                  path=_frame_loop(k0, width - k0, k0, height - k0, y_karm)))
    xc = width // 2
    win.add(Sweep(name="lodpost",
                  material=Material(key=JOINERY, profile_mm=(POST_DEPTH, POST_FACE)),
                  path=[Point(x=xc, y=y_karm, z=k1), Point(x=xc, y=y_karm, z=height - k1)]))
    y_sash = -(KARM_DEPTH - SASH_DEPTH) // 2
    sashes = [("left", k1 - SASH_LAP, xc - ph + SASH_LAP),
              ("right", xc + ph - SASH_LAP, width - k1 + SASH_LAP)]
    z0_s, z1_s = k1 - SASH_LAP, height - k1 + SASH_LAP
    for tag, x0_s, x1_s in sashes:
        win.add(Sweep(name=f"ramme_{tag}",
                      material=Material(key=JOINERY, profile_mm=(SASH_DEPTH, SASH_FACE)),
                      path=_frame_loop(x0_s + sh, x1_s - sh, z0_s + sh, z1_s - sh, y_sash)))
        gx0, gx1 = x0_s + 2 * sh - GLASS_BITE, x1_s - 2 * sh + GLASS_BITE
        gz0, gz1 = z0_s + 2 * sh - GLASS_BITE, z1_s - 2 * sh + GLASS_BITE
        y_igu_out = y_sash - IGU_MM // 2
        for k in range(3):
            y_pane = y_igu_out + k * (PANE_MM + GAS_GAP_MM) + PANE_MM // 2
            win.add(Extrude(name=f"glass_{tag}_{k}",
                            contour=[Point(x=gx0, y=y_pane, z=gz0), Point(x=gx1, y=y_pane, z=gz0),
                                     Point(x=gx1, y=y_pane, z=gz1), Point(x=gx0, y=y_pane, z=gz1)],
                            material=Material(key=GLAZING, thickness_mm=PANE_MM)))
        y_igu_in = y_igu_out + IGU_MM
        faces = [("out", y_igu_out - BAR_PROUD // 2), ("in", y_igu_in + BAR_PROUD // 2)]
        span = gz1 - gz0
        for r in range(1, PANE_ROWS):
            z_bar = gz0 + I(span * r / PANE_ROWS)
            for side, y_bar in faces:
                win.add(Sweep(name=f"sprosse_{tag}_{side}_{r}",
                              profile=_bar_profile(),
                              path=[Point(x=gx0, y=y_bar, z=z_bar), Point(x=gx1, y=y_bar, z=z_bar)],
                              material=Material(key=JOINERY)))
    return win


# Promote detailed window to Product catalog item
WIN_W, WIN_H = 1000, 1200
NORDIC_WINDOW = Product(build_window(WIN_W, WIN_H, "nordic"), name=f"nordic_{WIN_W}x{WIN_H}")


def generate_project():
    proj = Project(name="Murermestervilla Window Chain")
    np.random.seed(42)

    # First block at origin with a window opening
    wall_0 = Wall(name="block_00")
    wall_0.add(Box(name="body", start=Point(x=0, y=0, z=0), end=Point(x=1600, y=300, z=1600), material=Material(key="Concrete_C25-30")))
    wall_0.opening(NORDIC_WINDOW.occurrence(name="win_00"), along_center=800, up=200)
    proj.add(wall_0)

    # 19 subsequent blocks anchored to the preceding block's end face, each hosting a window opening
    for i in range(1, 20):
        rx, ry, rz = [I(np.random.uniform(-35, 35) * 100) for _ in range(3)]
        wall = Wall(
            name=f"block_{i:02d}",
            placement=Anchor(
                host=f"wall:block_{i-1:02d}",
                attach_to="end",
                rotations=[("x", rx), ("y", ry), ("z", rz)]
            )
        )
        wall.add(Box(name="body", start=Point(x=0, y=0, z=0), end=Point(x=1600, y=300, z=1600), material=Material(key="Concrete_C25-30")))
        wall.opening(NORDIC_WINDOW.occurrence(name=f"win_{i:02d}"), along_center=800, up=200)
        proj.add(wall)

    return proj


result = generate_project()


if __name__ == "__main__":
    compile_main()

