"""Wendelstein 7-X / Helias modular advanced stellarator.

Demonstrates:
  - 50 discrete non-planar modular coils (10 coils per field period, N_p=5)
    with 3D toroidal saddle warping and poloidal harmonic shaping.
  - Rectangular copper/superconducting bar coil geometry (160mm x 280mm)
    extruded along closed non-planar loops using rotation-minimising frames.
  - 5-period rotating bean-shaped plasma column in vivid yellow (#FFD700).
  - Non-planar 3D magnetic axis tube in emerald green (#00E676).
  - Zero CSG overhead: compiled directly as watertight vertex/face meshes
    under IfcBuildingElementProxy with carve="none".

Physics / Geometry notes:
  - N_p = 5 field periods on a major radius R_0 = 5500 mm (W7-X scale).
  - 50 discrete modular coils replace the continuous helical windings of 13_.
  - An open cutout window (omitting ~12 front coils) reveals the inner
    twisted yellow plasma core and magnetic axis, matching the iconic
    Max Planck IPP / Wikipedia W7-X render.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

# Ensure repo root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from lite_step import compile_main  # noqa: E402
from lite_step.models import (  # noqa: E402
    Element,
    Mesh,
    Point,
    Project,
    Site,
)

I = lambda x: int(round(x))  # noqa: E731,E741

# ── Global Stellarator Parameters (W7-X Scale) ───────────────────────────

R0 = 5_500           # Major radius of the torus (mm)
N_P = 5              # Number of toroidal field periods (5-fold symmetry)
TOTAL_COILS = 50     # Total modular coils in full 360-degree torus (10 per period)

# Plasma core geometry
PLASMA_A = 480       # Minor semi-axis 1 (mm)
PLASMA_B = 820       # Minor semi-axis 2 (mm)
PLASMA_BEAN = 160    # Higher-order bean/triangular shaping parameter (mm)
AXIS_R_RIPPLE = 180  # Non-planar magnetic axis radial ripple (mm)
AXIS_Z_RIPPLE = 180  # Non-planar magnetic axis vertical ripple (mm)

# Modular coil winding parameters (W7-X Non-Planar Shaping)
COIL_A = 1_280       # Coil minor semi-axis 1 (mm)
COIL_B = 1_580       # Coil minor semi-axis 2 (mm)
COIL_BEAN = 360      # Strong bean/crescent shaping harmonic (mm)
COIL_TRI = 180       # Triangular shaping harmonic (mm)
COIL_WARP_PHI = 0.12 # Strong toroidal non-planar deflection amplitude (~7 deg)

# Coil bar cross-section (Tall radial sandwich band, slim circumferentially)
COIL_BAR_RADIAL_DEPTH = 380.0   # Radial sandwich stack depth (mm)
COIL_BAR_TOROIDAL_WIDTH = 110.0 # Circumferential width along plasma stream (mm)

# Meshing resolution
COIL_PTS = 120       # Points around each closed coil loop
PLASMA_PHI_SEGS = 240 # Toroidal segments for plasma column
PLASMA_TH_SEGS = 64  # Poloidal segments for plasma column

# Cutout window configuration (set to True to omit coils in a sector to show plasma interior)
ENABLE_CUTOUT = False
CUTOUT_START = 27    # Starting coil index to omit if ENABLE_CUTOUT is True
CUTOUT_END = 41      # Ending coil index to omit (inclusive)


# ── Vector Utilities ──────────────────────────────────────────────────────

def _norm(v: tuple[float, float, float]) -> tuple[float, float, float]:
    m = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) or 1.0
    return (v[0] / m, v[1] / m, v[2] / m)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


# ── 1. Magnetic Axis (Emerald Green) ──────────────────────────────────────

def magnetic_axis_point(phi: float) -> tuple[float, float, float]:
    """Analytical 3D space curve of the non-planar magnetic axis."""
    r = R0 + AXIS_R_RIPPLE * math.cos(N_P * phi)
    z = AXIS_Z_RIPPLE * math.sin(N_P * phi)
    return (r * math.cos(phi), r * math.sin(phi), z)


def magnetic_axis_mesh(radius: float = 24.0, segments: int = 180, n_sides: int = 8) -> Mesh:
    """Generate a smooth non-planar closed tube mesh for the magnetic axis."""
    stations = [magnetic_axis_point(2 * math.pi * i / segments) for i in range(segments)]
    tangents: list[tuple[float, float, float]] = []
    for i in range(segments):
        nxt = stations[(i + 1) % segments]
        prv = stations[(i - 1) % segments]
        tangents.append(_norm(_sub(nxt, prv)))

    # Bishop parallel-transport frame
    v_init = (0.0, 0.0, 1.0)
    if abs(_dot(v_init, tangents[0])) > 0.9:
        v_init = (1.0, 0.0, 0.0)
    u0 = _norm(_cross(tangents[0], v_init))
    normals: list[tuple[float, float, float]] = [u0]
    for i in range(1, segments):
        t0, t1 = tangents[i - 1], tangents[i]
        axis = _cross(t0, t1)
        axis_len = math.sqrt(_dot(axis, axis))
        if axis_len > 1e-7:
            axis_unit = (axis[0] / axis_len, axis[1] / axis_len, axis[2] / axis_len)
            dot_val = max(-1.0, min(1.0, _dot(t0, t1)))
            angle = math.acos(dot_val)
            prev_u = normals[-1]
            k_cross_u = _cross(axis_unit, prev_u)
            k_dot_u = _dot(axis_unit, prev_u)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            u_next = (
                prev_u[0] * cos_a + k_cross_u[0] * sin_a + axis_unit[0] * k_dot_u * (1.0 - cos_a),
                prev_u[1] * cos_a + k_cross_u[1] * sin_a + axis_unit[1] * k_dot_u * (1.0 - cos_a),
                prev_u[2] * cos_a + k_cross_u[2] * sin_a + axis_unit[2] * k_dot_u * (1.0 - cos_a),
            )
            normals.append(_norm(u_next))
        else:
            normals.append(normals[-1])

    verts: list[Point] = []
    faces: list[tuple[int, int, int]] = []
    for i in range(segments):
        c = stations[i]
        t = tangents[i]
        u = normals[i]
        w = _cross(t, u)
        for s in range(n_sides):
            th = 2 * math.pi * s / n_sides
            px = c[0] + radius * (math.cos(th) * u[0] + math.sin(th) * w[0])
            py = c[1] + radius * (math.cos(th) * u[1] + math.sin(th) * w[1])
            pz = c[2] + radius * (math.cos(th) * u[2] + math.sin(th) * w[2])
            verts.append(Point(x=I(px), y=I(py), z=I(pz)))

    for i in range(segments):
        i_next = (i + 1) % segments
        for s in range(n_sides):
            s_next = (s + 1) % n_sides
            v0 = i * n_sides + s
            v1 = i * n_sides + s_next
            v2 = i_next * n_sides + s
            v3 = i_next * n_sides + s_next
            faces.append((v0, v3, v1))
            faces.append((v0, v2, v3))

    return Mesh(
        name="magnetic_axis",
        vertices=verts,
        faces=faces,
        is_watertight=True,
        color="#00E676",
        source_type="synthetic",
    )


def plasma_surface_stripe_mesh(width_mm: float = 40.0, segments: int = 240) -> Mesh:
    """Generate the emerald green midplane stripe running along the outer ridge of the plasma column."""
    stations = [plasma_surface_point(2 * math.pi * i / segments, theta=0.0) for i in range(segments + 1)]
    verts: list[Point] = []
    faces: list[tuple[int, int, int]] = []
    half_w = width_mm / 2.0

    for i in range(segments + 1):
        p = stations[i]
        nxt = stations[(i + 1) % segments]
        prv = stations[(i - 1) % segments]
        t = _norm(_sub(nxt, prv))
        r_torus = _norm((p[0], p[1], 0.0))
        nrm = _norm(_cross(t, r_torus))

        # Push outward slightly (+12mm) from plasma surface
        p_ctr = (p[0] + r_torus[0] * 12.0, p[1] + r_torus[1] * 12.0, p[2])
        p_left = (p_ctr[0] + nrm[0] * half_w, p_ctr[1] + nrm[1] * half_w, p_ctr[2] + nrm[2] * half_w)
        p_right = (p_ctr[0] - nrm[0] * half_w, p_ctr[1] - nrm[1] * half_w, p_ctr[2] - nrm[2] * half_w)

        verts.append(Point(x=I(p_left[0]), y=I(p_left[1]), z=I(p_left[2])))
        verts.append(Point(x=I(p_right[0]), y=I(p_right[1]), z=I(p_right[2])))

    for i in range(segments):
        v0, v1 = 2 * i, 2 * i + 1
        v2, v3 = 2 * i + 2, 2 * i + 3
        faces.extend([(v0, v1, v3), (v0, v3, v2), (v0, v3, v1), (v0, v2, v3)])

    return Mesh(
        name="plasma_surface_stripe",
        vertices=verts,
        faces=faces,
        is_watertight=False,
        color="#00C853",
        source_type="synthetic",
    )


# ── 2. Rotating Bean-Shaped Plasma Core (Vivid Yellow) ───────────────────

def plasma_surface_point(phi: float, theta: float) -> tuple[float, float, float]:
    """Parametric surface of a 5-period rotating bean-shaped plasma column."""
    alpha = (N_P * phi) / 2.0

    # Local bean/crescent cross section
    r_local = (
        PLASMA_A * math.cos(theta) * math.cos(alpha)
        - PLASMA_B * math.sin(theta) * math.sin(alpha)
        + PLASMA_BEAN * math.cos(2 * theta) * math.cos(alpha)
    )
    z_local = (
        PLASMA_A * math.cos(theta) * math.sin(alpha)
        + PLASMA_B * math.sin(theta) * math.cos(alpha)
        + PLASMA_BEAN * math.sin(2 * theta) * math.sin(alpha)
    )

    r = R0 + AXIS_R_RIPPLE * math.cos(N_P * phi) + r_local
    z = AXIS_Z_RIPPLE * math.sin(N_P * phi) + z_local

    return (r * math.cos(phi), r * math.sin(phi), z)


def plasma_mesh() -> Mesh:
    """Generate the watertight rotating-ellipse/bean plasma mesh."""
    n_phi = PLASMA_PHI_SEGS
    n_th = PLASMA_TH_SEGS

    verts: list[Point] = []
    faces: list[tuple[int, int, int]] = []

    for i in range(n_phi):
        phi = 2 * math.pi * i / n_phi
        for j in range(n_th):
            theta = 2 * math.pi * j / n_th
            p = plasma_surface_point(phi, theta)
            verts.append(Point(x=I(p[0]), y=I(p[1]), z=I(p[2])))

    # For odd N_p, crossing phi = 2pi introduces a half-turn (alpha rotates by N_p * pi).
    # Shift poloidal index by n_th // 2 to ensure watertight seam closure.
    seam_shift = n_th // 2

    for i in range(n_phi):
        i_next = (i + 1) % n_phi
        shift = seam_shift if (i + 1) == n_phi else 0
        for j in range(n_th):
            j_curr = j
            j_curr_next = (j + 1) % n_th
            j_next = (j + shift) % n_th
            j_next_next = (j + 1 + shift) % n_th

            v0 = i * n_th + j_curr
            v1 = i * n_th + j_curr_next
            v2 = i_next * n_th + j_next
            v3 = i_next * n_th + j_next_next

            faces.append((v0, v3, v1))
            faces.append((v0, v2, v3))

    return Mesh(
        name="plasma_boundary",
        vertices=verts,
        faces=faces,
        is_watertight=True,
        color="#FFD700",
        source_type="synthetic",
    )


# ── 3. Non-Planar Modular Coils (Cobalt Blue) ─────────────────────────────

def modular_coil_point(coil_index: int, theta: float) -> tuple[float, float, float]:
    """Calculate 3D space curve of modular coil `coil_index` at poloidal angle `theta`."""
    nominal_phi = 2 * math.pi * coil_index / TOTAL_COILS
    alpha = (N_P * nominal_phi) / 2.0
    field_phase = N_P * nominal_phi

    # 1. Non-planar 3D toroidal deflection (strong S-curve saddle twist)
    phi_deflection = COIL_WARP_PHI * (
        math.sin(theta) * math.sin(field_phase)
        + 0.55 * math.sin(2 * theta) * math.cos(field_phase)
        + 0.25 * math.sin(3 * theta) * math.sin(field_phase)
    )
    actual_phi = nominal_phi + phi_deflection

    # 2. Modulated minor radii per period
    ca = COIL_A + 140.0 * math.cos(field_phase)
    cb = COIL_B - 140.0 * math.cos(field_phase)

    # 3. Local cross-section with bean & triangular shaping harmonics
    r_local = (
        ca * math.cos(theta) * math.cos(alpha)
        - cb * math.sin(theta) * math.sin(alpha)
        + COIL_BEAN * math.cos(2 * theta) * math.cos(alpha)
        + COIL_TRI * math.cos(3 * theta) * math.sin(alpha)
    )
    z_local = (
        ca * math.cos(theta) * math.sin(alpha)
        + cb * math.sin(theta) * math.cos(alpha)
        + COIL_BEAN * math.sin(2 * theta) * math.sin(alpha)
        + COIL_TRI * math.sin(3 * theta) * math.cos(alpha)
    )

    r = R0 + AXIS_R_RIPPLE * math.cos(N_P * actual_phi) + r_local
    z = AXIS_Z_RIPPLE * math.sin(N_P * actual_phi) + z_local

    return (r * math.cos(actual_phi), r * math.sin(actual_phi), z)


def modular_coil_mesh(coil_index: int) -> Mesh:
    """Generate a watertight rectangular bar mesh (tall radial sandwich, slim circumferentially)."""
    n_pts = COIL_PTS
    stations = [modular_coil_point(coil_index, 2 * math.pi * s / n_pts) for s in range(n_pts)]
    nominal_phi = 2 * math.pi * coil_index / TOTAL_COILS

    # Compute tangents along the closed loop
    tangents: list[tuple[float, float, float]] = []
    for s in range(n_pts):
        nxt = stations[(s + 1) % n_pts]
        prv = stations[(s - 1) % n_pts]
        tangents.append(_norm(_sub(nxt, prv)))

    # Frame orientation:
    # u = Outward radial direction from the magnetic axis
    # w = Circumferential / toroidal direction (t x u)
    half_radial = COIL_BAR_RADIAL_DEPTH / 2.0
    half_toroidal = COIL_BAR_TOROIDAL_WIDTH / 2.0
    profile_offsets = [
        (-half_radial, -half_toroidal),
        (half_radial, -half_toroidal),
        (half_radial, half_toroidal),
        (-half_radial, half_toroidal),
    ]

    verts: list[Point] = []
    faces: list[tuple[int, int, int]] = []

    for s in range(n_pts):
        c = stations[s]
        t = tangents[s]

        # Outward radial direction from the magnetic axis at this angle
        axis_pt = magnetic_axis_point(nominal_phi)
        v_rad = _sub(c, axis_pt)
        t_dot_v = _dot(t, v_rad)
        u_unnorm = (v_rad[0] - t_dot_v * t[0], v_rad[1] - t_dot_v * t[1], v_rad[2] - t_dot_v * t[2])
        u = _norm(u_unnorm)
        w = _cross(t, u)

        for du, dw in profile_offsets:
            px = c[0] + du * u[0] + dw * w[0]
            py = c[1] + du * u[1] + dw * w[1]
            pz = c[2] + du * u[2] + dw * w[2]
            verts.append(Point(x=I(px), y=I(py), z=I(pz)))

    for s in range(n_pts):
        s_next = (s + 1) % n_pts
        for k in range(4):
            k_next = (k + 1) % 4
            v0 = s * 4 + k
            v1 = s * 4 + k_next
            v2 = s_next * 4 + k
            v3 = s_next * 4 + k_next

            faces.append((v0, v3, v1))
            faces.append((v0, v2, v3))

    return Mesh(
        name=f"modular_coil_{coil_index}",
        vertices=verts,
        faces=faces,
        is_watertight=True,
        color="#1E88E5",
        source_type="synthetic",
    )


# ── 4. High-Field Hotspot Volumes (80% Transparent Red) ───────────────────

def hotspot_obloid_mesh(
    name: str,
    center: tuple[float, float, float],
    rx: float,
    ry: float,
    rz: float,
    phi_rad: float,
    n_lat: int = 16,
    n_lon: int = 24,
    color: str = "#FF174433",  # 80% transparent red (0x33 alpha / 255 -> 80% transparency)
) -> Mesh:
    """Generate a smooth obloid volume marking a localized coil density / magnetic pinch spot."""
    # Orientation unit vectors:
    u_rad = (math.cos(phi_rad), math.sin(phi_rad), 0.0)
    u_tor = (-math.sin(phi_rad), math.cos(phi_rad), 0.0)
    u_vert = (0.0, 0.0, 1.0)

    verts: list[Point] = []
    faces: list[tuple[int, int, int]] = []

    for i in range(n_lat + 1):
        lat = -math.pi / 2.0 + math.pi * i / n_lat
        cos_lat = math.cos(lat)
        sin_lat = math.sin(lat)
        for j in range(n_lon):
            lon = 2 * math.pi * j / n_lon
            cos_lon = math.cos(lon)
            sin_lon = math.sin(lon)

            dx = rx * cos_lat * cos_lon
            dy = ry * cos_lat * sin_lon
            dz = rz * sin_lat

            px = center[0] + dx * u_rad[0] + dy * u_tor[0] + dz * u_vert[0]
            py = center[1] + dx * u_rad[1] + dy * u_tor[1] + dz * u_vert[1]
            pz = center[2] + dx * u_rad[2] + dy * u_tor[2] + dz * u_vert[2]

            verts.append(Point(x=I(px), y=I(py), z=I(pz)))

    for i in range(n_lat):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            v0 = i * n_lon + j
            v1 = i * n_lon + j_next
            v2 = (i + 1) * n_lon + j
            v3 = (i + 1) * n_lon + j_next

            faces.append((v0, v1, v3))
            faces.append((v0, v3, v2))
            faces.append((v0, v3, v1))
            faces.append((v0, v2, v3))

    return Mesh(
        name=name,
        vertices=verts,
        faces=faces,
        is_watertight=False,
        color=color,
        source_type="synthetic",
    )


# ── Assembly ─────────────────────────────────────────────────────────────

def generate_project() -> Project:
    proj = Project(name="Wendelstein 7-X Modular Stellarator")
    site = Site(name="machine_hall")

    # 1. Emerald Green Magnetic Axis
    axis_elem = Element(name="magnetic_axis", ifc_class="IfcBuildingElementProxy")
    axis_elem.add(magnetic_axis_mesh())
    site.add(axis_elem, carve="none")

    # 2. Vivid Yellow Rotating-Bean Plasma Core + Outer Midplane Stripe
    plasma_elem = Element(name="plasma", ifc_class="IfcBuildingElementProxy")
    plasma_elem.add(plasma_mesh())
    site.add(plasma_elem, carve="none")

    stripe_elem = Element(name="plasma_stripe", ifc_class="IfcBuildingElementProxy")
    stripe_elem.add(plasma_surface_stripe_mesh())
    site.add(stripe_elem, carve="none")

    # 3. 14 Localized High-Field Hotspots (80% Transparent Red Obloids)
    hotspot_count = 0

    # A. 5 Inboard High-Field Pinches (inner compression ring, R ~ 4250mm)
    for p in range(N_P):
        phi_in = 2 * math.pi * p / N_P + math.pi / N_P
        r_in = R0 - 1_250.0 + AXIS_R_RIPPLE * math.cos(N_P * phi_in)
        z_in = AXIS_Z_RIPPLE * math.sin(N_P * phi_in)
        ctr_in = (r_in * math.cos(phi_in), r_in * math.sin(phi_in), z_in)

        hs_mesh = hotspot_obloid_mesh(
            name=f"hotspot_inboard_{p}",
            center=ctr_in,
            rx=420.0,
            ry=350.0,
            rz=750.0,
            phi_rad=phi_in,
        )
        hs_elem = Element(name=f"hotspot_in_{p}", ifc_class="IfcBuildingElementProxy")
        hs_elem.add(hs_mesh)
        site.add(hs_elem, carve="none")
        hotspot_count += 1

    # B. 5 Outboard Saddle Apex Pinches (outer flare ring, R ~ 6850mm)
    for p in range(N_P):
        phi_out = 2 * math.pi * p / N_P
        r_out = R0 + 1_350.0 + AXIS_R_RIPPLE * math.cos(N_P * phi_out)
        z_out = AXIS_Z_RIPPLE * math.sin(N_P * phi_out)
        ctr_out = (r_out * math.cos(phi_out), r_out * math.sin(phi_out), z_out)

        hs_mesh = hotspot_obloid_mesh(
            name=f"hotspot_outboard_{p}",
            center=ctr_out,
            rx=460.0,
            ry=380.0,
            rz=850.0,
            phi_rad=phi_out,
        )
        hs_elem = Element(name=f"hotspot_out_{p}", ifc_class="IfcBuildingElementProxy")
        hs_elem.add(hs_mesh)
        site.add(hs_elem, carve="none")
        hotspot_count += 1

    # C. 4 Intermediate Transition Inflection Pinches (top/bottom S-curve nodes)
    trans_angles = [0.45, 1.85, 3.45, 4.95]
    for k, phi_tr in enumerate(trans_angles):
        r_tr = R0 + 400.0 * math.cos(N_P * phi_tr)
        z_tr = 750.0 if (k % 2 == 0) else -750.0
        ctr_tr = (r_tr * math.cos(phi_tr), r_tr * math.sin(phi_tr), z_tr)

        hs_mesh = hotspot_obloid_mesh(
            name=f"hotspot_trans_{k}",
            center=ctr_tr,
            rx=380.0,
            ry=340.0,
            rz=550.0,
            phi_rad=phi_tr,
        )
        hs_elem = Element(name=f"hotspot_tr_{k}", ifc_class="IfcBuildingElementProxy")
        hs_elem.add(hs_mesh)
        site.add(hs_elem, carve="none")
        hotspot_count += 1

    # 4. Discrete Non-Planar Modular Coils (Cobalt Blue)
    coils_built = 0
    for idx in range(TOTAL_COILS):
        if ENABLE_CUTOUT and (CUTOUT_START <= idx <= CUTOUT_END):
            continue  # Open viewing window to reveal inner plasma and magnetic axis

        coil_mesh_obj = modular_coil_mesh(idx)
        coil_elem = Element(name=f"coil_{idx:02d}", ifc_class="IfcBuildingElementProxy")
        coil_elem.add(coil_mesh_obj)
        site.add(coil_elem, carve="none")
        coils_built += 1

    print(f"[stellarator] Built {coils_built}/{TOTAL_COILS} modular non-planar coils + {hotspot_count} red pinch hotspots")
    proj.add(site)
    return proj


result = generate_project()

if __name__ == "__main__":
    compile_main()
