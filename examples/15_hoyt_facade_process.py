"""11 Hoyt Street Structural Core & 8-Step Creative Facade Process.

How to Handle Complex Geometric & Creative Ideas (Step-by-Step Guide)
=====================================================================

When a user describes a sculptural or non-standard architectural form in plain
language (e.g. "a concrete block carved by two rotated rectangle-to-oval tubes
meeting oval-to-oval at an offset peak"), follow this workflow to stay aligned
with their creative intent:

1. **Understand the user's physical mental model before optimizing algebra.**
   - Listen for sculptural verbs ("press inward", "rotate in the horizontal
     plane", "subtract oval to oval", "extend past the modules").
   - Do not silently replace a requested 3D form with a sheared wedge or
     single-face surface hack just because the boolean result looks similar.
     If an exact cutting surface differs from the idealized 3D form, show both
     explicitly so the user can see the relationship.

2. **Guide the user step-by-step with visual, inspectable intermediate steps.**
   - Lay out every stage side-by-side in front of the main model (``-Y`` side)
     so the user can orbit and inspect the design progression left-to-right:
     - **Step 1 (`step1_block`):** Raw concrete stock block.
     - **Step 2 (`step2_rect_to_oval_tube`):** Un-rotated prototype 3D form
       (stretched rectangle-to-oval tube).
     - **Step 3 (`step3_rotated_tube_left/right`):** Two copies rotated in the
       horizontal (XY) plane—never sheared—meeting oval-to-oval at the peak.
     - **Step 4 (`step4_tube_*` + `step4_block`):** Extended 10 m along the
       linear morph (revealing the concave 4-lobed "flower" cross-section) with
       a transparent red block to verify internal alignment.
     - **Step 5 (`step5_*`):** Inverted transparency (solid concrete block +
       transparent red extended tubes) to inspect the cut from the outside.
     - **Step 6 (`step6_cutters`):** Exact cutting wedges used for the boolean.
     - **Step 7 (`step7_block_and_cutters`):** Block + cutting wedges overlaid.
     - **Step 8 (`step8_block_minus_cutters`):** Final carved precast facade
       with punched openings, thin black frames, and tinted blue glazing.
   - Wrap each half and block in its own ``IfcBuildingElementProxy`` with
     ``carve="none"`` so every piece can be selected and toggled independently
     in any IFC viewer.

3. **Bridge creative intuition and exact engineering with real measurements.**
   - When fitting an idealized 3D form (like a linear ``rectangle -> oval``
     tube) to an analytic target surface, measure the actual mesh deviation
     (``Mean |dy|``, ``RMSE``, ``Max |dy|``) and explain the tradeoff honestly
     (here: stretching the tube taller than the 3-module block drops mean
     error from ``47.2 mm`` to ``10.7 mm`` with linear ``s``).
   - Consult the user before changing the underlying mathematical form.

Compile with::

    python -m lite_step examples/15_hoyt_facade_process.py
"""
# ruff: noqa: F401 — the import block below is the full DSL menu, kept intact
# by design: unused names are a feature (later tasks use them), never lint
# errors, and never to be pruned.
import math
import numpy as np
from lite_step import compile_main, param
from lite_step.models import (Project, Storey, Point, Point2D,
    Wall, Window, Door, Slab, Roof, Column, Beam, Space, Site, Element,
    Box, Extrude, Sweep, Sheet, Pipe, Revolve, Mesh, Bar, Material, LayerSet,
    ReferencePoint, GuideLine, Transform, Anchor, Product)
from lite_step.helpers import arc_path, stirrup_path
from lite_step.materials.dk import MATERIALS   # noqa: F401 — stock introspection, e.g. MATERIALS["Timber_C24"].geometry.stock_profiles_mm

I = lambda x: int(round(x))  # noqa: E731,E741

# ── Datums & Dimensions (all int mm, Z-up) ────────────────────────────────────
CONCRETE_MAT = "Concrete_C30-37"

# Tower structural floor plate (10 bays E-W x 18 bays N-S @ 3500 mm)
TOWER_W_X = param(35000, "mm", "Tower floor plate width (E-W)")
TOWER_L_Y = param(63000, "mm", "Tower floor plate length (N-S)")
HALF_X = TOWER_W_X // 2   # 17500
HALF_Y = TOWER_L_Y // 2   # 31500

SLAB_THICK = param(200, "mm", "Typical floor slab thickness")          # 8-in. cast-in-place flat-plate concrete slab
CELLAR_SLAB_THICK = param(300, "mm", "Cellar base slab thickness")
ROOF_SLAB_THICK = param(250, "mm", "Roof slab thickness")

# Structural column grid (centred at origin)
GRID_X = [-17000, -9000, -3000, 3000, 9000, 17000]
GRID_Y = [-31000, -24000, -17000, -10000, -4000, 4000, 10000, 17000, 24000, 31000]

# Central core (2 elevator shafts + 2 flanking stairwells)
CORE_WALL_T = param(300, "mm", "Core shear wall thickness")
ELEV_W = param(2500, "mm", "Elevator shaft clear width")             # clear width per elevator shaft
ELEV_D = param(2500, "mm", "Elevator shaft clear depth")             # clear depth per elevator shaft
STAIR_W = param(3000, "mm", "Stairwell clear width")                 # clear width per stairwell
STAIR_L = param(5600, "mm", "Stairwell clear length")                # clear length per stairwell (y: -2800..2800)

# Core X coordinates
X_ELEV_W_R = -(CORE_WALL_T // 2)        # -150
X_ELEV_W_L = X_ELEV_W_R - ELEV_W        # -2650
X_ELEV_E_L = CORE_WALL_T // 2           # 150
X_ELEV_E_R = X_ELEV_E_L + ELEV_W        # 2650
X_STAIR_W_R = X_ELEV_W_L - CORE_WALL_T  # -2950
X_STAIR_W_L = X_STAIR_W_R - STAIR_W     # -5950
X_STAIR_W_OUT = X_STAIR_W_L - CORE_WALL_T   # -6250
X_STAIR_E_L = X_ELEV_E_R + CORE_WALL_T  # 2950
X_STAIR_E_R = X_STAIR_E_L + STAIR_W     # 5950
X_STAIR_E_OUT = X_STAIR_E_R + CORE_WALL_T   # 6250

# Core Y coordinates
Y_ELEV_S_IN = -(ELEV_D // 2)            # -1250
Y_ELEV_N_IN = ELEV_D // 2               # 1250
Y_ELEV_S_OUT = Y_ELEV_S_IN - CORE_WALL_T    # -1550
Y_ELEV_N_OUT = Y_ELEV_N_IN + CORE_WALL_T    # 1550
Y_CORE_S_IN = -(STAIR_L // 2)           # -2800
Y_CORE_N_IN = STAIR_L // 2              # 2800
Y_CORE_S_OUT = Y_CORE_S_IN - CORE_WALL_T    # -3100
Y_CORE_N_OUT = Y_CORE_N_IN + CORE_WALL_T    # 3100

# Stair geometry inside each stairwell
STAIR_GAP_W = 300
STAIR_FLIGHT_W = (STAIR_W - STAIR_GAP_W) // 2   # 1350
STAIR_LANDING_D = 1400    # South floor landing (-2800..-1400) & North mid-landing (+1400..+2800)
STAIR_RUN_S = Y_CORE_S_IN + STAIR_LANDING_D     # -1400
STAIR_RUN_N = Y_CORE_N_IN - STAIR_LANDING_D     # 1400
STAIR_WAIST_T = param(180, "mm", "Stair waist slab thickness")

# Door / opening sizes in core
ELEV_DOOR_W = param(1100, "mm", "Elevator door opening width")
ELEV_DOOR_H = param(2200, "mm", "Elevator door opening height")
STAIR_DOOR_W = param(1000, "mm", "Stairwell door opening width")
STAIR_DOOR_H = param(2100, "mm", "Stairwell door opening height")
ELEV_TOWER_H = param(4500, "mm", "Roof elevator overrun tower height")

# ── 3-Module Rectangle->Oval Subtraction Steps Parameters ─────────────────────
FACADE_BAY_W = 3500
FACADE_MOD_H = 3450
FACADE_W = 2 * FACADE_BAY_W   # 7000
FACADE_H3 = 3 * FACADE_MOD_H  # 10350
FACADE_D = 1360               # solid concrete block depth (220 base + 1140 projection)
FACADE_BASE_T = 220
FACADE_X_PEAK = 4340          # offset peak in X (longer left side, shorter right side)
FACADE_OVAL_PRESS = 450       # inward press of the ovals in the center
PRECAST_COLOR = "#EAE6DF"
CUTTER_COLOR = "#E07A5F99"


def _signed_volume_ok(verts, faces):
    """Return True if closed mesh has positive signed volume."""
    pts = np.array([[p.x, p.y, p.z] for p in verts], dtype=np.float64)
    tris = np.array(faces, dtype=np.int64)
    v0 = pts[tris[:, 0]]
    v1 = pts[tris[:, 1]]
    v2 = pts[tris[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2)))) > 0.0


def _cutter_y_at(x, z, side):
    """Return cutting-surface Y (more positive = deeper into block) for left or right rectangle->oval cutter."""
    if side == "left":
        x0, y0 = -600.0, -FACADE_BASE_T + 140.0
        x1, y1 = float(FACADE_X_PEAK), -float(FACADE_D)
        s = (x - x0) / (x1 - x0)
    else:
        x0, y0 = float(FACADE_W + 600), -FACADE_BASE_T + 140.0
        x1, y1 = float(FACADE_X_PEAK), -float(FACADE_D)
        s = (x0 - x) / (x0 - x1)

    if s < 0.0 or s > 1.05:
        return -9999.0
    s_clamped = max(0.0, min(1.0, s))
    y_plane = y0 + s * (y1 - y0)
    zn = (z - 0.5 * FACADE_H3) / (0.5 * FACADE_H3)
    oval = math.sqrt(max(0.0, 1.0 - zn * zn))
    # Morph from flat rectangle (s=0) to curved oval pressing inward (+Y) at s=1
    press = (s_clamped ** 1.2) * FACADE_OVAL_PRESS * oval
    return y_plane + press


def _build_rect_to_oval_cutter_mesh(side):
    """Build a watertight lofted rectangle->oval cutter Mesh rotated in XY."""
    ns, nz = 24, 24
    y_front = -FACADE_D - 800
    zs = [I(FACADE_H3 * k / nz) for k in range(nz + 1)]
    if side == "left":
        xs = [I(-600 + (FACADE_X_PEAK + 600) * i / ns) for i in range(ns + 1)]
    else:
        xs = [I(FACADE_X_PEAK + (FACADE_W + 600 - FACADE_X_PEAK) * i / ns) for i in range(ns + 1)]

    verts = []
    f_idx = np.zeros((ns + 1, nz + 1), dtype=int)
    b_idx = np.zeros((ns + 1, nz + 1), dtype=int)
    for ix, x in enumerate(xs):
        for iz, z in enumerate(zs):
            yb = I(_cutter_y_at(x, z, side))
            f_idx[ix, iz] = len(verts)
            verts.append(Point(x=x, y=y_front, z=z))
            b_idx[ix, iz] = len(verts)
            verts.append(Point(x=x, y=yb, z=z))

    faces = []
    for ix in range(ns):
        for iz in range(nz):
            f00, f10 = int(f_idx[ix, iz]), int(f_idx[ix + 1, iz])
            f11, f01 = int(f_idx[ix + 1, iz + 1]), int(f_idx[ix, iz + 1])
            b00, b10 = int(b_idx[ix, iz]), int(b_idx[ix + 1, iz])
            b11, b01 = int(b_idx[ix + 1, iz + 1]), int(b_idx[ix, iz + 1])
            faces.extend([(f00, f10, f11), (f00, f11, f01)])  # front (-Y)
            faces.extend([(b00, b11, b10), (b00, b01, b11)])  # back/cutting (+Y)
            if ix == 0:
                faces.extend([(f00, f01, b01), (f00, b01, b00)])
            if ix == ns - 1:
                faces.extend([(f10, b10, b11), (f10, b11, f11)])
            if iz == 0:
                faces.extend([(f00, b00, b10), (f00, b10, f10)])
            if iz == nz - 1:
                faces.extend([(f01, f11, b11), (f01, b11, b01)])

    assert _signed_volume_ok(verts, faces), f"Non-positive volume in cutter {side}"
    return Mesh(vertices=verts, faces=faces, color=CUTTER_COLOR, is_watertight=True)


def _block_front_y_at(x, z):
    """Return front Y coordinate (<= -FACADE_BASE_T) of the subtracted concrete block."""
    y_cut = max(-float(FACADE_D), _cutter_y_at(x, z, "left"), _cutter_y_at(x, z, "right"))
    return I(min(-float(FACADE_BASE_T), y_cut))


def _quad_prism_mesh(p00, p10, p11, p01, y_depth, color):
    """Build a watertight 8-vertex hexahedron from 4 outer (x,y,z) points extruded +y."""
    verts = [
        Point(x=p00[0], y=p00[1], z=p00[2]),
        Point(x=p10[0], y=p10[1], z=p10[2]),
        Point(x=p11[0], y=p11[1], z=p11[2]),
        Point(x=p01[0], y=p01[1], z=p01[2]),
        Point(x=p00[0], y=p00[1] + y_depth, z=p00[2]),
        Point(x=p10[0], y=p10[1] + y_depth, z=p10[2]),
        Point(x=p11[0], y=p11[1] + y_depth, z=p11[2]),
        Point(x=p01[0], y=p01[1] + y_depth, z=p01[2]),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 3, 7), (0, 7, 4),
        (1, 5, 6), (1, 6, 2),
        (0, 4, 5), (0, 5, 1),
        (3, 2, 6), (3, 6, 7),
    ]
    return Mesh(vertices=verts, faces=faces, color=color, is_watertight=True)


# Window X ranges per module: left window offset away from peak (4340), right window on shorter side
_WIN_X_RANGES = [(350, 2850), (4700, 6650)]
_WIN_SILL_H = 550
_WIN_HEAD_H = 450


def _build_subtracted_block_mesh():
    """Build watertight Mesh of the concrete block minus the 2 cutters, with 6 punched window openings."""
    # Grid aligned to exact window boundaries and peak
    xs = sorted({I(FACADE_W * i / 28) for i in range(29)} | {350, 2850, FACADE_X_PEAK, 4700, 6650})
    zs_set = {I(FACADE_H3 * k / 28) for k in range(29)}
    win_z_ranges = []
    for m in range(3):
        z0 = m * FACADE_MOD_H + _WIN_SILL_H
        z1 = (m + 1) * FACADE_MOD_H - _WIN_HEAD_H
        win_z_ranges.append((z0, z1))
        zs_set.update((m * FACADE_MOD_H, z0, z1, (m + 1) * FACADE_MOD_H))
    zs = sorted(zs_set)

    nx, nz = len(xs) - 1, len(zs) - 1
    solid = np.ones((nx, nz), dtype=bool)
    for ix in range(nx):
        xm = 0.5 * (xs[ix] + xs[ix + 1])
        if any(xa < xm < xb for xa, xb in _WIN_X_RANGES):
            for iz in range(nz):
                zm = 0.5 * (zs[iz] + zs[iz + 1])
                if any(za < zm < zb for za, zb in win_z_ranges):
                    solid[ix, iz] = False

    verts = []
    o_idx = np.zeros((nx + 1, nz + 1), dtype=int)
    n_idx = np.zeros((nx + 1, nz + 1), dtype=int)
    for ix, x in enumerate(xs):
        for iz, z in enumerate(zs):
            y_front = _block_front_y_at(x, z)
            o_idx[ix, iz] = len(verts)
            verts.append(Point(x=x, y=y_front, z=z))
            n_idx[ix, iz] = len(verts)
            verts.append(Point(x=x, y=0, z=z))

    faces = []
    for ix in range(nx):
        for iz in range(nz):
            if not solid[ix, iz]:
                continue
            o00, o10 = int(o_idx[ix, iz]), int(o_idx[ix + 1, iz])
            o11, o01 = int(o_idx[ix + 1, iz + 1]), int(o_idx[ix, iz + 1])
            n00, n10 = int(n_idx[ix, iz]), int(n_idx[ix + 1, iz])
            n11, n01 = int(n_idx[ix + 1, iz + 1]), int(n_idx[ix, iz + 1])
            faces.extend([(o00, o10, o11), (o00, o11, o01)])
            faces.extend([(n00, n11, n10), (n00, n01, n11)])
            if ix == 0 or not solid[ix - 1, iz]:
                faces.extend([(o00, o01, n01), (o00, n01, n00)])
            if ix == nx - 1 or not solid[ix + 1, iz]:
                faces.extend([(o10, n10, n11), (o10, n11, o11)])
            if iz == 0 or not solid[ix, iz - 1]:
                faces.extend([(o00, n00, n10), (o00, n10, o10)])
            if iz == nz - 1 or not solid[ix, iz + 1]:
                faces.extend([(o01, o11, n11), (o01, n11, n01)])

    assert _signed_volume_ok(verts, faces), "Non-positive volume in subtracted block"
    return Mesh(vertices=verts, faces=faces, color=PRECAST_COLOR, is_watertight=True)


def _add_windows_to_final_block(parent):
    """Add 6 windows (thin black metal frames + slightly tinted blue glass) to the final block."""
    frame_w, frame_d, glass_t, recess = 45, 80, 24, 90
    frame_color = "#1A1A1A"
    glass_color = "#4F86B899"

    for m in range(3):
        z0 = m * FACADE_MOD_H + _WIN_SILL_H
        z1 = (m + 1) * FACADE_MOD_H - _WIN_HEAD_H
        for b, (xa, xb) in enumerate(_WIN_X_RANGES):
            def _y_at(x, z):
                return _block_front_y_at(x, z) + recess

            def _y_interp(x, z):
                t = (x - xa) / float(xb - xa)
                return I(_y_at(xa, z) * (1.0 - t) + _y_at(xb, z) * t)

            frame = Element(ifc_class="IfcMember", predefined_type="MULLION", name=f"win_frame_{m}_{b}")
            frame.add(_quad_prism_mesh(
                (xa, _y_at(xa, z0), z0), (xb, _y_at(xb, z0), z0),
                (xb, _y_at(xb, z0 + frame_w), z0 + frame_w), (xa, _y_at(xa, z0 + frame_w), z0 + frame_w),
                frame_d, frame_color,
            ))
            frame.add(_quad_prism_mesh(
                (xa, _y_at(xa, z1 - frame_w), z1 - frame_w), (xb, _y_at(xb, z1 - frame_w), z1 - frame_w),
                (xb, _y_at(xb, z1), z1), (xa, _y_at(xa, z1), z1),
                frame_d, frame_color,
            ))
            frame.add(_quad_prism_mesh(
                (xa, _y_at(xa, z0 + frame_w), z0 + frame_w), (xa + frame_w, _y_interp(xa + frame_w, z0 + frame_w), z0 + frame_w),
                (xa + frame_w, _y_interp(xa + frame_w, z1 - frame_w), z1 - frame_w), (xa, _y_at(xa, z1 - frame_w), z1 - frame_w),
                frame_d, frame_color,
            ))
            frame.add(_quad_prism_mesh(
                (xb - frame_w, _y_interp(xb - frame_w, z0 + frame_w), z0 + frame_w), (xb, _y_at(xb, z0 + frame_w), z0 + frame_w),
                (xb, _y_at(xb, z1 - frame_w), z1 - frame_w), (xb - frame_w, _y_interp(xb - frame_w, z1 - frame_w), z1 - frame_w),
                frame_d, frame_color,
            ))
            parent.add(frame, carve="none")

            glass = Element(ifc_class="IfcWindow", name=f"win_glass_{m}_{b}")
            gx0, gx1 = xa + frame_w, xb - frame_w
            gz0, gz1 = z0 + frame_w, z1 - frame_w
            gy_off = (frame_d - glass_t) // 2
            glass.add(_quad_prism_mesh(
                (gx0, _y_interp(gx0, gz0) + gy_off, gz0),
                (gx1, _y_interp(gx1, gz0) + gy_off, gz0),
                (gx1, _y_interp(gx1, gz1) + gy_off, gz1),
                (gx0, _y_interp(gx0, gz1) + gy_off, gz1),
                glass_t, glass_color,
            ))
            parent.add(glass, carve="none")


def _add_facade_step_showcase(storey):
    """Place the 8-step facade showcase cleanly in front (-Y) of the building on level_01."""
    y_pos = -50000
    step_x = [-52000, -41000, -26500, -7000, 19000, 39500, 51500, 63500]

    # Step 1: Raw 3-module precast concrete block before any carving
    s1 = Element(ifc_class="IfcBuildingElementProxy", name="step1_block")
    s1.add(Box(name="body", start=Point(x=0, y=-FACADE_D, z=0), end=Point(x=FACADE_W, y=0, z=FACADE_H3), material=CONCRETE_MAT, color=PRECAST_COLOR))
    s1.placement = Transform(origin=Point(x=step_x[0], y=y_pos, z=0), rotations=[])
    storey.add(s1)

    # Step 2: Un-rotated stretched rectangle->oval concrete prototype tube
    s2 = Element(ifc_class="IfcBuildingElementProxy", name="step2_rect_to_oval_tube")
    s2.add(_build_free_rect_to_oval_tube_mesh())
    s2.placement = Transform(origin=Point(x=step_x[1], y=y_pos, z=0), rotations=[])
    storey.add(s2)

    # Step 3: Two copies rotated in the horizontal plane (perpendicular slices, not sheared)
    # placed with carve="none" so left and right halves are independently selectable
    for side in ("left", "right"):
        s3 = Element(ifc_class="IfcBuildingElementProxy", name=f"step3_rotated_tube_{side}")
        s3.add(_build_rotated_rect_to_oval_tube_mesh(side))
        s3.placement = Transform(origin=Point(x=step_x[2], y=y_pos, z=0), rotations=[])
        storey.add(s3, carve="none")

    # Step 4: Tubes extended 10 m past the oval end (revealing the concave flower cross-section)
    # overlaid with a transparent red block so internal cut alignment is visible
    _add_extended_tubes_and_block_step(
        storey, "step4", step_x[3], y_pos, tube_color=PRECAST_COLOR, block_color=CUTTER_COLOR
    )

    # Step 5: Same geometry with inverted transparency (solid concrete block + transparent red tubes)
    _add_extended_tubes_and_block_step(
        storey, "step5", step_x[4], y_pos, tube_color=CUTTER_COLOR, block_color=PRECAST_COLOR
    )

    # Step 6: Analytic cutting wedges matching the exact s^1.2 inward oval press
    s6 = Element(ifc_class="IfcBuildingElementProxy", name="step6_cutters")
    s6.add(_build_rect_to_oval_cutter_mesh("left"))
    s6.add(_build_rect_to_oval_cutter_mesh("right"))
    s6.placement = Transform(origin=Point(x=step_x[5], y=y_pos, z=0), rotations=[])
    storey.add(s6)

    # Step 7: Concrete block + analytic cutting wedges overlaid together (carve="none")
    s7 = Element(ifc_class="IfcBuildingElementProxy", name="step7_block_and_cutters")
    s7.add(Box(name="body", start=Point(x=0, y=-FACADE_D, z=0), end=Point(x=FACADE_W, y=0, z=FACADE_H3), material=CONCRETE_MAT, color=PRECAST_COLOR), carve="none")
    s7.add(_build_rect_to_oval_cutter_mesh("left"), carve="none")
    s7.add(_build_rect_to_oval_cutter_mesh("right"), carve="none")
    s7.placement = Transform(origin=Point(x=step_x[6], y=y_pos, z=0), rotations=[])
    storey.add(s7)

    # Step 8: Final sculpted 3-module precast panel with 6 punched windows, black frames, and blue glass
    s8 = Element(ifc_class="IfcBuildingElementProxy", name="step8_block_minus_cutters")
    s8.add(_build_subtracted_block_mesh())
    _add_windows_to_final_block(s8)
    s8.placement = Transform(origin=Point(x=step_x[7], y=y_pos, z=0), rotations=[])
    storey.add(s8)


def _build_free_rect_to_oval_tube_mesh():
    """Build a watertight concrete lofted Mesh matching the stretched rect->oval prototype."""
    w_rect = 2097
    h_rect = 11521
    a_y = 1072
    b_z = 6201
    length_x = 7953
    z_c = 5175
    n_slices = 24
    n_theta = 48

    verts = []
    rings = []
    for i in range(n_slices + 1):
        s = i / float(n_slices)
        x = I(s * length_x)
        ring = []
        for j in range(n_theta):
            theta = 2.0 * math.pi * j / float(n_theta)
            cy = math.cos(theta)
            cz = math.sin(theta)
            m = max(abs(cy), abs(cz))
            ry = cy / m
            rz = cz / m
            y = I((1.0 - s) * (0.5 * w_rect * ry) + s * (a_y * cy))
            z = I(z_c + (1.0 - s) * (0.5 * h_rect * rz) + s * (b_z * cz))
            ring.append(len(verts))
            verts.append(Point(x=x, y=y, z=z))
        rings.append(ring)

    faces = []
    for i in range(n_slices):
        r0 = rings[i]
        r1 = rings[i + 1]
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            v00, v01 = r0[j], r0[j_next]
            v10, v11 = r1[j], r1[j_next]
            faces.append((v00, v01, v11))
            faces.append((v00, v11, v10))

    start_center_idx = len(verts)
    verts.append(Point(x=0, y=0, z=z_c))
    end_center_idx = len(verts)
    verts.append(Point(x=length_x, y=0, z=z_c))

    r_start = rings[0]
    r_end = rings[-1]
    for j in range(n_theta):
        j_next = (j + 1) % n_theta
        faces.append((start_center_idx, r_start[j_next], r_start[j]))
        faces.append((end_center_idx, r_end[j], r_end[j_next]))

    if not _signed_volume_ok(verts, faces):
        faces = [(a, c, b) for (a, b, c) in faces]
    assert _signed_volume_ok(verts, faces), "Non-positive volume in free rect->oval tube"
    return Mesh(
        name="body",
        vertices=verts,
        faces=faces,
        material=CONCRETE_MAT,
        color=PRECAST_COLOR,
        is_watertight=True,
    )


def _build_rotated_rect_to_oval_tube_mesh(side, extend_mm=0, color=PRECAST_COLOR):
    """Build a true rotated rectangle->oval tube aligned to the left or right cut plane."""
    if side == "left":
        p0 = (-600.0, -float(FACADE_BASE_T) + 140.0)
    else:
        p0 = (float(FACADE_W + 600), -float(FACADE_BASE_T) + 140.0)
    p1 = (float(FACADE_X_PEAK), -float(FACADE_D))

    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    l_plane = math.hypot(dx, dy)
    ux, uy = dx / l_plane, dy / l_plane
    # Inward (+Y-ish) unit normal perpendicular to tube axis (ux, uy)
    if side == "left":
        nx, ny = -uy, ux
    else:
        nx, ny = uy, -ux

    # Stretched rect->oval cross-section fitted with linear s to minimize cut-surface error (~10-11 mm mean)
    w_rect = 2097.0
    h_rect = 11521.0
    a_w = 1072.0
    b_z = 6201.0
    z_c = 0.5 * FACADE_H3
    if side == "left":
        t0, l_tube, off0, off1 = -1282.0, 7953.0, -1168.0, -492.0
    else:
        t0, l_tube, off0, off1 = -921.0, 5641.0, -1187.0, -475.0
    l_total = l_tube + float(extend_mm)
    n_slices = 24 + int(extend_mm // 500)
    n_theta = 48

    verts = []
    rings = []
    centers = []
    for i in range(n_slices + 1):
        t = (i / float(n_slices)) * l_total
        s = t / l_tube
        t_ax = t0 + t
        w_off = (1.0 - s) * off0 + s * off1
        cx = p0[0] + t_ax * ux + w_off * nx
        cy = p0[1] + t_ax * uy + w_off * ny
        centers.append((cx, cy))

        ring = []
        for j in range(n_theta):
            theta = 2.0 * math.pi * j / float(n_theta)
            ct = math.cos(theta)
            st = math.sin(theta)
            m = max(abs(ct), abs(st))
            rt_w = ct / m
            rt_z = st / m
            w_local = (1.0 - s) * (0.5 * w_rect * rt_w) + s * (a_w * ct)
            z_local = (1.0 - s) * (0.5 * h_rect * rt_z) + s * (b_z * st)
            px = I(cx + w_local * nx)
            py = I(cy + w_local * ny)
            pz = I(z_c + z_local)
            ring.append(len(verts))
            verts.append(Point(x=px, y=py, z=pz))
        rings.append(ring)

    faces = []
    for i in range(n_slices):
        r0 = rings[i]
        r1 = rings[i + 1]
        for j in range(n_theta):
            j_next = (j + 1) % n_theta
            v00, v01 = r0[j], r0[j_next]
            v10, v11 = r1[j], r1[j_next]
            faces.append((v00, v01, v11))
            faces.append((v00, v11, v10))

    start_center_idx = len(verts)
    verts.append(Point(x=I(centers[0][0]), y=I(centers[0][1]), z=I(z_c)))
    end_center_idx = len(verts)
    verts.append(Point(x=I(centers[-1][0]), y=I(centers[-1][1]), z=I(z_c)))

    r_start = rings[0]
    r_end = rings[-1]
    for j in range(n_theta):
        j_next = (j + 1) % n_theta
        faces.append((start_center_idx, r_start[j_next], r_start[j]))
        faces.append((end_center_idx, r_end[j], r_end[j_next]))

    if not _signed_volume_ok(verts, faces):
        faces = [(a, c, b) for (a, b, c) in faces]
    assert _signed_volume_ok(verts, faces), f"Non-positive volume in rotated rect->oval tube {side}"
    return Mesh(
        name=f"body_{side}",
        vertices=verts,
        faces=faces,
        material=CONCRETE_MAT,
        color=color,
        is_watertight=True,
    )


def _add_extended_tubes_and_block_step(storey, prefix, x_pos, y_pos, tube_color, block_color):
    """Place left tube, right tube, and aligned block as separate selectable no-carve elements."""
    for side in ("left", "right"):
        elem = Element(ifc_class="IfcBuildingElementProxy", name=f"{prefix}_tube_{side}")
        elem.add(_build_rotated_rect_to_oval_tube_mesh(side, extend_mm=10000, color=tube_color))
        elem.placement = Transform(origin=Point(x=x_pos, y=y_pos, z=0), rotations=[])
        storey.add(elem, carve="none")

    block = Element(ifc_class="IfcBuildingElementProxy", name=f"{prefix}_block")
    block.add(
        Box(
            name="body",
            start=Point(x=0, y=-FACADE_D, z=0),
            end=Point(x=FACADE_W, y=0, z=FACADE_H3),
            material=CONCRETE_MAT,
            color=block_color,
        )
    )
    block.placement = Transform(origin=Point(x=x_pos, y=y_pos, z=0), rotations=[])
    storey.add(block, carve="none")


def _build_storey_schedule():
    """Return list of (name, elevation_mm, height_mm, slab_thick_mm, has_voids) for all levels."""
    levels = [
        ("cellar", -4200, 4200, CELLAR_SLAB_THICK, False),
        ("level_01", 0, 4800, SLAB_THICK, True),
        ("level_02", 4800, 4200, SLAB_THICK, True),
        ("level_03", 9000, 4800, SLAB_THICK, True),
    ]
    z = 13800
    for n in range(4, 32):
        levels.append((f"level_{n:02d}", z, 3450, SLAB_THICK, True))
        z += 3450
    levels.append(("level_32", z, 5000, SLAB_THICK, True))
    z += 5000
    for n in range(33, 52):
        levels.append((f"level_{n:02d}", z, 3450, SLAB_THICK, True))
        z += 3450
    levels.append(("level_52", z, 7050, SLAB_THICK, True))
    z += 7050
    levels.append(("roof", z, 0, ROOF_SLAB_THICK, False))
    return levels


def _col_size_for_level(idx):
    """Return square concrete column width (mm) tapered by height tier."""
    if idx <= 15:
        return 900
    if idx <= 35:
        return 750
    return 600


def _make_slab(storey_name, z_top, thick, has_voids, elev_only=False):
    """Create a concrete floor slab with elevator shaft and stairwell openings."""
    slab = Slab(name=f"slab_{storey_name}")
    slab.add(
        Box(
            name="body",
            start=Point(x=-HALF_X, y=-HALF_Y, z=z_top - thick),
            end=Point(x=HALF_X, y=HALF_Y, z=z_top),
            material=CONCRETE_MAT,
            color="slab",
        )
    )
    if has_voids:
        z0 = z_top - thick - 50
        z1 = z_top + 50
        slab.void(
            Box(name="void_elev_w", start=Point(x=X_ELEV_W_L, y=Y_ELEV_S_IN, z=z0), end=Point(x=X_ELEV_W_R, y=Y_ELEV_N_IN, z=z1)),
            name="elev_shaft_west",
        )
        slab.void(
            Box(name="void_elev_e", start=Point(x=X_ELEV_E_L, y=Y_ELEV_S_IN, z=z0), end=Point(x=X_ELEV_E_R, y=Y_ELEV_N_IN, z=z1)),
            name="elev_shaft_east",
        )
        if not elev_only:
            slab.void(
                Box(name="void_stair_w", start=Point(x=X_STAIR_W_L, y=STAIR_RUN_S, z=z0), end=Point(x=X_STAIR_W_R, y=Y_CORE_N_IN, z=z1)),
                name="stairwell_west",
            )
            slab.void(
                Box(name="void_stair_e", start=Point(x=X_STAIR_E_L, y=STAIR_RUN_S, z=z0), end=Point(x=X_STAIR_E_R, y=Y_CORE_N_IN, z=z1)),
                name="stairwell_east",
            )
    return slab


def _make_columns(storey_name, z_bot, z_top, col_w):
    """Create structural concrete columns on the grid, omitting core zone."""
    half_c = col_w // 2
    cols = []
    for ix, cx in enumerate(GRID_X):
        for iy, cy in enumerate(GRID_Y):
            # Omit interior columns inside/adjacent to the central core & elevator lobby
            if abs(cx) <= 3500 and abs(cy) <= 4500:
                continue
            col = Column(name=f"col_{storey_name}_{ix}_{iy}")
            col.add(
                Box(
                    name="body",
                    start=Point(x=cx - half_c, y=cy - half_c, z=z_bot),
                    end=Point(x=cx + half_c, y=cy + half_c, z=z_top),
                    material=CONCRETE_MAT,
                    color="column",
                )
            )
            cols.append(col)
    return cols


def _make_core_wall(name, x0, y0, x1, y1, z0, z1):
    """Create a disjoint concrete core shear wall segment."""
    wall = Wall(name=name)
    wall.add(
        Box(
            name="body",
            start=Point(x=x0, y=y0, z=z0),
            end=Point(x=x1, y=y1, z=z1),
            material=CONCRETE_MAT,
            color="wall",
        )
    )
    return wall


def _make_core(storey_name, z_bot, z_top):
    """Create the 2 central elevator shafts and 2 flanking stairwell enclosures."""
    walls = [
        # Full-length N-S core walls (disjoint)
        _make_core_wall(f"core_w_out_{storey_name}", X_STAIR_W_OUT, Y_CORE_S_OUT, X_STAIR_W_L, Y_CORE_N_OUT, z_bot, z_top),
        _make_core_wall(f"core_w_in_{storey_name}", X_STAIR_W_R, Y_CORE_S_OUT, X_ELEV_W_L, Y_CORE_N_OUT, z_bot, z_top),
        _make_core_wall(f"core_e_in_{storey_name}", X_ELEV_E_R, Y_CORE_S_OUT, X_STAIR_E_L, Y_CORE_N_OUT, z_bot, z_top),
        _make_core_wall(f"core_e_out_{storey_name}", X_STAIR_E_R, Y_CORE_S_OUT, X_STAIR_E_OUT, Y_CORE_N_OUT, z_bot, z_top),
        # Center divider between the 2 elevator shafts
        _make_core_wall(f"elev_div_{storey_name}", X_ELEV_W_R, Y_ELEV_S_IN, X_ELEV_E_L, Y_ELEV_N_IN, z_bot, z_top),
        # North walls (no openings)
        _make_core_wall(f"stair_w_n_{storey_name}", X_STAIR_W_L, Y_CORE_N_IN, X_STAIR_W_R, Y_CORE_N_OUT, z_bot, z_top),
        _make_core_wall(f"stair_e_n_{storey_name}", X_STAIR_E_L, Y_CORE_N_IN, X_STAIR_E_R, Y_CORE_N_OUT, z_bot, z_top),
        _make_core_wall(f"elev_n_{storey_name}", X_ELEV_W_L, Y_ELEV_N_IN, X_ELEV_E_R, Y_ELEV_N_OUT, z_bot, z_top),
    ]

    # South walls with door / elevator openings
    stair_w_s = _make_core_wall(f"stair_w_s_{storey_name}", X_STAIR_W_L, Y_CORE_S_OUT, X_STAIR_W_R, Y_CORE_S_IN, z_bot, z_top)
    stair_w_s.opening(Door(name="stair_door_w", width=STAIR_DOOR_W, height=STAIR_DOOR_H), along_center=STAIR_W // 2, up=0)

    stair_e_s = _make_core_wall(f"stair_e_s_{storey_name}", X_STAIR_E_L, Y_CORE_S_OUT, X_STAIR_E_R, Y_CORE_S_IN, z_bot, z_top)
    stair_e_s.opening(Door(name="stair_door_e", width=STAIR_DOOR_W, height=STAIR_DOOR_H), along_center=STAIR_W // 2, up=0)

    elev_s = _make_core_wall(f"elev_s_{storey_name}", X_ELEV_W_L, Y_ELEV_S_OUT, X_ELEV_E_R, Y_ELEV_S_IN, z_bot, z_top)
    elev_s.opening(Door(name="elev_door_w", width=ELEV_DOOR_W, height=ELEV_DOOR_H), along_center=ELEV_W // 2, up=0)
    elev_s.opening(Door(name="elev_door_e", width=ELEV_DOOR_W, height=ELEV_DOOR_H), along_center=ELEV_W + CORE_WALL_T + ELEV_W // 2, up=0)

    walls.extend([stair_w_s, stair_e_s, elev_s])
    return walls


def _stair_flight_contour(x_pos, y_start, y_end, z_start, z_end, n_risers):
    """Build a closed CCW Y-Z contour at x=x_pos (Newell normal = +X) for a monolithic stair flight."""
    dy = (y_end - y_start) / n_risers
    dz = (z_end - z_start) / n_risers
    pts = [Point(x=x_pos, y=y_start, z=z_start - STAIR_WAIST_T)]
    pts.append(Point(x=x_pos, y=y_end, z=z_end - STAIR_WAIST_T))
    pts.append(Point(x=x_pos, y=y_end, z=z_end))
    for i in range(n_risers - 1, -1, -1):
        yi = I(y_start + i * dy)
        zi_top = I(z_start + (i + 1) * dz)
        zi_bot = I(z_start + i * dz)
        pts.append(Point(x=x_pos, y=yi, z=zi_top))
        if i > 0:
            pts.append(Point(x=x_pos, y=yi, z=zi_bot))
    pts.append(Point(x=x_pos, y=y_start, z=z_start))
    # Reverse if going -Y so Newell normal always points +X
    if y_end < y_start:
        pts.reverse()
    return pts


def _make_stair(name, x_min, z_bot, z_next):
    """Create a dog-leg concrete stair (2 flights + mid-landing) inside a 3000 mm wide stairwell."""
    stair = Element(ifc_class="IfcStair", name=name)
    z_mid = I((z_bot + z_next) / 2)
    n_risers = 10 if (z_next - z_bot) <= 3600 else 12

    # Mid-landing at North end (+1400..+2800)
    landing = Element(ifc_class="IfcSlab", predefined_type="LANDING", name="mid_landing")
    landing.add(
        Box(
            name="body",
            start=Point(x=x_min, y=STAIR_RUN_N, z=z_mid - SLAB_THICK),
            end=Point(x=x_min + STAIR_W, y=Y_CORE_N_IN, z=z_mid),
            material=CONCRETE_MAT,
            color="slab",
        )
    )
    stair.add(landing)

    # Flight 1: West side, South (-1400) -> North (+1400), z_bot -> z_mid
    f1 = Element(ifc_class="IfcStairFlight", name="flight_up_1")
    f1.add(
        Extrude(
            name="body",
            contour=_stair_flight_contour(x_min, STAIR_RUN_S, STAIR_RUN_N, z_bot, z_mid, n_risers),
            thickness=STAIR_FLIGHT_W,
            material=CONCRETE_MAT,
            color="slab",
        )
    )
    stair.add(f1)

    # Flight 2: East side, North (+1400) -> South (-1400), z_mid -> z_next
    f2_x = x_min + STAIR_FLIGHT_W + STAIR_GAP_W
    f2 = Element(ifc_class="IfcStairFlight", name="flight_up_2")
    f2.add(
        Extrude(
            name="body",
            contour=_stair_flight_contour(f2_x, STAIR_RUN_N, STAIR_RUN_S, z_mid, z_next, n_risers),
            thickness=STAIR_FLIGHT_W,
            material=CONCRETE_MAT,
            color="slab",
        )
    )
    stair.add(f2)

    return stair


def _make_elevator_towers(z_base):
    """Create two small concrete elevator overrun towers on top of the roof."""
    z_wall_top = z_base + ELEV_TOWER_H - SLAB_THICK
    z_roof_top = z_base + ELEV_TOWER_H
    elems = [
        # West elevator tower walls
        _make_core_wall("elev_tower_w_w", X_STAIR_W_R, Y_ELEV_S_OUT, X_ELEV_W_L, Y_ELEV_N_OUT, z_base, z_wall_top),
        _make_core_wall("elev_tower_w_e", X_ELEV_W_R, Y_ELEV_S_OUT, 0, Y_ELEV_N_OUT, z_base, z_wall_top),
        _make_core_wall("elev_tower_w_s", X_ELEV_W_L, Y_ELEV_S_OUT, X_ELEV_W_R, Y_ELEV_S_IN, z_base, z_wall_top),
        _make_core_wall("elev_tower_w_n", X_ELEV_W_L, Y_ELEV_N_IN, X_ELEV_W_R, Y_ELEV_N_OUT, z_base, z_wall_top),
        # East elevator tower walls
        _make_core_wall("elev_tower_e_w", 0, Y_ELEV_S_OUT, X_ELEV_E_L, Y_ELEV_N_OUT, z_base, z_wall_top),
        _make_core_wall("elev_tower_e_e", X_ELEV_E_R, Y_ELEV_S_OUT, X_STAIR_E_L, Y_ELEV_N_OUT, z_base, z_wall_top),
        _make_core_wall("elev_tower_e_s", X_ELEV_E_L, Y_ELEV_S_OUT, X_ELEV_E_R, Y_ELEV_S_IN, z_base, z_wall_top),
        _make_core_wall("elev_tower_e_n", X_ELEV_E_L, Y_ELEV_N_IN, X_ELEV_E_R, Y_ELEV_N_OUT, z_base, z_wall_top),
    ]
    for side, x0, x1 in (("w", X_STAIR_W_R, 0), ("e", 0, X_STAIR_E_L)):
        cap = Roof(name=f"elev_tower_roof_{side}")
        cap.add(
            Box(
                name="body",
                start=Point(x=x0, y=Y_ELEV_S_OUT, z=z_wall_top),
                end=Point(x=x1, y=Y_ELEV_N_OUT, z=z_roof_top),
                material=CONCRETE_MAT,
                color="roof",
            )
        )
        elems.append(cap)
    return elems


def generate_project():
    proj = Project(name="11 Hoyt Street - Structural Core & Frame")
    schedule = _build_storey_schedule()

    for idx, (s_name, z_elev, height, slab_t, has_voids) in enumerate(schedule):
        storey = Storey(name=s_name, elevation=z_elev)
        proj.add_storey(storey)

        # Concrete floor slab (roof slab keeps elevator shaft voids open into roof towers)
        is_roof = idx == len(schedule) - 1
        storey.add(_make_slab(s_name, z_elev, slab_t, has_voids or is_roof, elev_only=is_roof))

        # Structural elements rising to next level's slab soffit
        if idx < len(schedule) - 1:
            _, z_next, _, next_slab_t, _ = schedule[idx + 1]
            z_soffit = z_next - next_slab_t
            col_w = _col_size_for_level(idx)

            storey.add(*_make_columns(s_name, z_elev, z_soffit, col_w))
            storey.add(*_make_core(s_name, z_elev, z_soffit))
            storey.add(_make_stair(f"stair_w_{s_name}", X_STAIR_W_L, z_elev, z_next))
            storey.add(_make_stair(f"stair_e_{s_name}", X_STAIR_E_L, z_elev, z_next))
        else:
            storey.add(*_make_elevator_towers(z_elev))

    # 8-step facade construction showcase placed cleanly in front (-Y) of the building on level_01
    _add_facade_step_showcase(proj.storeys[1])

    return proj


result = generate_project()


if __name__ == "__main__":
    compile_main()
