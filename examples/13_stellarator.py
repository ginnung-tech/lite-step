"""Classical helical stellarator (torsatron / heliotron) with magnetic fieldlines.

Closed-form analytical geometry without an MHD equilibrium solver:

1. **Helical winding law** — 2l thin-filament conductors wrapped on a torus
   of major radius R0, minor radius a_coil, with N_p helical turns:

       x_k(phi) = (R0 + a_coil*cos(N_p/l*phi + pi*k/l)) * cos(phi)
       y_k(phi) = (R0 + a_coil*cos(N_p/l*phi + pi*k/l)) * sin(phi)
       z_k(phi) =       a_coil*sin(N_p/l*phi + pi*k/l)

2. **Rotating-ellipse plasma boundary** — an ellipse (a, b) rotated by
   alpha = N_p*phi/2 as it travels the torus. Reproduces the 3D rotating
   topology of stellarator magnetic surfaces.

3. **Magnetic fieldline ribbons** — transparent blue ribbons tracing magnetic
   fieldlines at intermediate flux surfaces between the core plasma and the
   helical coils. Fieldlines follow rotational transform iota with N_p helical
   ripple modulation.

Compile with::

    python examples/13_stellarator.py
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

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

# ── Machine parameters (W7-X / LHD scale, mm) ────────────────────────────
R0 = 5_500          # major radius
A_COIL = 1_600      # helical winding minor radius
N_P = 5             # field periods / helical turns
L = 2               # l pairs -> 2l conductors

PLASMA_A = 400      # core plasma ellipse semi-minor axis (mm)
PLASMA_B = 900      # core plasma ellipse semi-major axis (mm, kappa = 2.25)
CONDUCTOR_R = 170   # coil conductor tube radius
CONDUCTOR_SIDES = 12

N_PHI_PLASMA = 120  # toroidal grid resolution for plasma
N_THETA_PLASMA = 48  # poloidal grid resolution for plasma
SEG_PER_LAP = 120   # coil polyline resolution per lap


# ── 1. Helical Coils ──────────────────────────────────────────────────────

def coil_laps() -> int:
    """Toroidal circuits before a filament closes on itself."""
    return L // math.gcd(N_P, L)


def distinct_coils() -> int:
    """Filaments that are geometrically distinct curves."""
    return math.gcd(2 * N_P, 2 * L)


def coil_point(k: int, phi: float) -> tuple[float, float, float]:
    phase = (N_P / L) * phi + math.pi * k / L
    r = R0 + A_COIL * math.cos(phase)
    return (r * math.cos(phi), r * math.sin(phi), A_COIL * math.sin(phase))


def assert_coil_closes(k: int) -> None:
    """The filament must return to its start after `laps` circuits."""
    laps = coil_laps()
    p0 = coil_point(k, 0.0)
    p1 = coil_point(k, 2 * math.pi * laps)
    gap = math.dist(p0, p1)
    assert gap < 1.0, (
        f"coil {k} does not close after {laps} lap(s): {gap:.1f} mm gap. "
        f"laps must be l/gcd(N_p, l) = {L}/{math.gcd(N_P, L)}"
    )
    if laps > 1:
        near = math.dist(p0, coil_point(k, 2 * math.pi))
        assert near > 1.0, f"coil {k} already closed after 1 lap; laps={laps} is wrong"


def _tube_frames(stations: list[tuple[float, float, float]]):
    """Rotation-minimising (parallel-transport) frames along the filament."""
    n = len(stations)

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def norm(v):
        m = math.sqrt(sum(c * c for c in v)) or 1.0
        return (v[0] / m, v[1] / m, v[2] / m)

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    tang = [norm(sub(stations[(i + 1) % n], stations[i - 1])) for i in range(n)]
    seed = (0.0, 0.0, 1.0) if abs(tang[0][2]) < 0.9 else (1.0, 0.0, 0.0)
    nrm = [norm(cross(seed, tang[0]))]
    for i in range(1, n):
        axis = cross(tang[i - 1], tang[i])
        sn = math.sqrt(sum(c * c for c in axis))
        if sn < 1e-9:
            nrm.append(nrm[-1])
            continue
        axis = (axis[0] / sn, axis[1] / sn, axis[2] / sn)
        ang = math.atan2(sn, dot(tang[i - 1], tang[i]))
        pv, ca, sa = nrm[-1], math.cos(ang), math.sin(ang)
        cr = cross(axis, pv)
        dt = dot(axis, pv)
        nrm.append(norm(tuple(pv[m] * ca + cr[m] * sa + axis[m] * dt * (1 - ca) for m in range(3))))
    return tang, nrm


def coil_mesh(k: int) -> Mesh:
    """The filament as a closed tube MESH."""
    laps = coil_laps()
    n = SEG_PER_LAP * laps
    stations = [coil_point(k, 2 * math.pi * laps * i / n) for i in range(n)]
    tang, nrm = _tube_frames(stations)

    verts, faces = [], []
    for i in range(n):
        t, u = tang[i], nrm[i]
        w = (
            t[1] * u[2] - t[2] * u[1],
            t[2] * u[0] - t[0] * u[2],
            t[0] * u[1] - t[1] * u[0],
        )
        c = stations[i]
        for m in range(CONDUCTOR_SIDES):
            a = 2 * math.pi * m / CONDUCTOR_SIDES
            ca, sa = math.cos(a) * CONDUCTOR_R, math.sin(a) * CONDUCTOR_R
            verts.append(
                Point(
                    x=I(c[0] + u[0] * ca + w[0] * sa),
                    y=I(c[1] + u[1] * ca + w[1] * sa),
                    z=I(c[2] + u[2] * ca + w[2] * sa),
                )
            )
    for i in range(n):
        nx = (i + 1) % n
        for m in range(CONDUCTOR_SIDES):
            mm = (m + 1) % CONDUCTOR_SIDES
            a, b = i * CONDUCTOR_SIDES + m, i * CONDUCTOR_SIDES + mm
            c_idx, d_idx = nx * CONDUCTOR_SIDES + m, nx * CONDUCTOR_SIDES + mm
            faces.append((a, b, d_idx))
            faces.append((a, d_idx, c_idx))

    return Mesh(
        name=f"helical_coil_{k}",
        vertices=verts,
        faces=faces,
        is_watertight=True,
        color="#C9752B",
        source_type="synthetic",
    )


# ── 2. Rotating-Ellipse Plasma Boundary ───────────────────────────────────

def plasma_point(theta: float, phi: float) -> tuple[float, float, float]:
    alpha = N_P * phi / 2.0
    r_local = PLASMA_A * math.cos(theta) * math.cos(alpha) - PLASMA_B * math.sin(theta) * math.sin(alpha)
    z = PLASMA_A * math.cos(theta) * math.sin(alpha) + PLASMA_B * math.sin(theta) * math.cos(alpha)
    r = R0 + r_local
    return (r * math.cos(phi), r * math.sin(phi), z)


def plasma_seam_shift() -> int:
    """Poloidal index offset needed to close the toroidal seam."""
    return (N_THETA_PLASMA // 2) if (N_P % 2) else 0


def plasma_mesh() -> Mesh:
    shift = plasma_seam_shift()
    verts, faces = [], []
    for i in range(N_PHI_PLASMA):
        phi = 2 * math.pi * i / N_PHI_PLASMA
        for j in range(N_THETA_PLASMA):
            x, y, z = plasma_point(2 * math.pi * j / N_THETA_PLASMA, phi)
            verts.append(Point(x=I(x), y=I(y), z=I(z)))

    def v(i: int, j: int) -> int:
        return i * N_THETA_PLASMA + (j % N_THETA_PLASMA)

    for i in range(N_PHI_PLASMA):
        nxt = (i + 1) % N_PHI_PLASMA
        off = shift if i == N_PHI_PLASMA - 1 else 0
        for j in range(N_THETA_PLASMA):
            a, b = v(i, j), v(i, j + 1)
            c, d = v(nxt, j + off), v(nxt, j + 1 + off)
            faces.append((a, b, d))
            faces.append((a, d, c))

    return Mesh(
        name="plasma_boundary",
        vertices=verts,
        faces=faces,
        is_watertight=True,
        color="#FF6B9D80",
        source_type="synthetic",
    )


# ── 3. Magnetic Fieldlines: Near-Field Coil Flux & Core Confinement ─────────

def coil_near_field_ribbon_mesh(
    coil_k: int,
    ribbon_id: int,
    offset_angle: float,
    spiral_turns_per_lap: float = 48.0,
    sheath_radius: float = 270.0,
    width_mm: float = 40.0,
    color: str = "#8A2BE285",
) -> Mesh:
    """Generate a translucent purple ribbon spiraling tightly around a helical coil conductor.

    Visualizes the near-field Ampèrian magnetic vortex induced around the conductor tube.
    """
    laps = coil_laps()
    n = SEG_PER_LAP * laps * 6  # High resolution for tight corkscrew winding
    stations = [coil_point(coil_k, 2 * math.pi * laps * i / n) for i in range(n + 1)]
    tang, nrm = _tube_frames(stations[:n])
    # Extend last frame for the endpoint
    tang.append(tang[0])
    nrm.append(nrm[0])

    verts: list[Point] = []
    faces: list[tuple[int, int, int]] = []
    half_w = width_mm / 2.0
    total_phi = 2 * math.pi * laps

    for i in range(n + 1):
        phi = total_phi * i / n
        t, u = tang[i], nrm[i]
        # Binormal w = t x u
        w = (
            t[1] * u[2] - t[2] * u[1],
            t[2] * u[0] - t[0] * u[2],
            t[0] * u[1] - t[1] * u[0],
        )
        # Poloidal angle spiraling tightly around the coil tube
        psi = offset_angle + (spiral_turns_per_lap * (phi / (2 * math.pi))) * 2 * math.pi
        cos_psi, sin_psi = math.cos(psi), math.sin(psi)

        # Unit radial outward vector from coil centerline to fieldline
        n_rad = (
            u[0] * cos_psi + w[0] * sin_psi,
            u[1] * cos_psi + w[1] * sin_psi,
            u[2] * cos_psi + w[2] * sin_psi,
        )
        # Ribbon width vector tangent to the cylinder around the coil: t x n_rad
        binorm_ribbon = (
            t[1] * n_rad[2] - t[2] * n_rad[1],
            t[2] * n_rad[0] - t[0] * n_rad[2],
            t[0] * n_rad[1] - t[1] * n_rad[0],
        )

        c = stations[i]
        # Center of fieldline
        p_ctr = (
            c[0] + n_rad[0] * sheath_radius,
            c[1] + n_rad[1] * sheath_radius,
            c[2] + n_rad[2] * sheath_radius,
        )

        p_left = (
            p_ctr[0] + binorm_ribbon[0] * half_w,
            p_ctr[1] + binorm_ribbon[1] * half_w,
            p_ctr[2] + binorm_ribbon[2] * half_w,
        )
        p_right = (
            p_ctr[0] - binorm_ribbon[0] * half_w,
            p_ctr[1] - binorm_ribbon[1] * half_w,
            p_ctr[2] - binorm_ribbon[2] * half_w,
        )

        verts.append(Point(x=I(p_left[0]), y=I(p_left[1]), z=I(p_left[2])))
        verts.append(Point(x=I(p_right[0]), y=I(p_right[1]), z=I(p_right[2])))

    for i in range(n):
        v0, v1 = 2 * i, 2 * i + 1
        v2, v3 = 2 * i + 2, 2 * i + 3
        # Double-sided quad strip
        faces.append((v0, v1, v3))
        faces.append((v0, v3, v2))
        faces.append((v0, v3, v1))
        faces.append((v0, v2, v3))

    return Mesh(
        name=f"coil_{coil_k}_near_field_{ribbon_id}",
        vertices=verts,
        faces=faces,
        is_watertight=False,
        color=color,
        source_type="synthetic",
    )


def fieldline_point(
    surface_r: float,
    theta_0: float,
    phi: float,
    iota: float = 1.2,
    ripple_amp: float = 0.15,
) -> tuple[float, float, float]:
    """Analytical fieldline trajectory on a rotating flux surface.

    Parameters:
    - surface_r: effective minor radius of the flux surface
    - theta_0: starting poloidal phase at phi=0
    - phi: toroidal angle
    - iota: rotational transform (poloidal pitch per toroidal revolution)
    - ripple_amp: field period ripple modulation depth
    """
    alpha = (N_P * phi) / 2.0
    # Fieldline poloidal angle advances with iota plus helical ripple
    theta = theta_0 + iota * phi + ripple_amp * math.sin(N_P * phi)

    # Elliptical flux surface scaling at this radius
    aspect = PLASMA_B / PLASMA_A
    a_surf = surface_r
    b_surf = surface_r * (1.0 + (aspect - 1.0) * (PLASMA_A / surface_r))

    r_local = a_surf * math.cos(theta) * math.cos(alpha) - b_surf * math.sin(theta) * math.sin(alpha)
    z = a_surf * math.cos(theta) * math.sin(alpha) + b_surf * math.sin(theta) * math.cos(alpha)
    r = R0 + r_local
    return (r * math.cos(phi), r * math.sin(phi), z)


def fieldline_ribbon_mesh(
    ribbon_id: int,
    surface_r: float,
    theta_0: float,
    iota: float = 1.2,
    laps: int = 5,
    pts_per_lap: int = 180,
    width_mm: float = 50.0,
    color: str = "#00BFFF44",
) -> Mesh:
    """Generate a continuous, twist-free translucent ribbon following a magnetic fieldline.

    Constructed as a double-sided quad strip (2 triangles per segment) oriented
    along the local flux surface normal.
    """
    n_pts = laps * pts_per_lap
    total_phi = 2 * math.pi * laps

    stations = []
    for i in range(n_pts + 1):
        phi = total_phi * i / n_pts
        stations.append(fieldline_point(surface_r, theta_0, phi, iota=iota))

    # Parallel transport frames to prevent artificial ribbon twisting
    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def norm(v):
        m = math.sqrt(sum(c * c for c in v)) or 1.0
        return (v[0] / m, v[1] / m, v[2] / m)

    def cross(a, b):
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )

    # Compute ribbon lateral direction (normal to tangent and major radius vector)
    verts: list[Point] = []
    faces: list[tuple[int, int, int]] = []
    half_w = width_mm / 2.0

    for i in range(n_pts + 1):
        p = stations[i]
        # Tangent vector
        if i == 0:
            t = norm(sub(stations[1], stations[0]))
        elif i == n_pts:
            t = norm(sub(stations[n_pts], stations[n_pts - 1]))
        else:
            t = norm(sub(stations[i + 1], stations[i - 1]))

        # Major radius radial unit vector from torus center (0,0,z)
        r_unit = norm((p[0], p[1], 0.0))
        # Normal to the flux surface
        surf_normal = norm(cross(t, r_unit))
        if abs(surf_normal[0]) + abs(surf_normal[1]) + abs(surf_normal[2]) < 1e-6:
            surf_normal = (0.0, 0.0, 1.0)

        # Left and right ribbon edges
        p_left = (
            p[0] + surf_normal[0] * half_w,
            p[1] + surf_normal[1] * half_w,
            p[2] + surf_normal[2] * half_w,
        )
        p_right = (
            p[0] - surf_normal[0] * half_w,
            p[1] - surf_normal[1] * half_w,
            p[2] - surf_normal[2] * half_w,
        )

        verts.append(Point(x=I(p_left[0]), y=I(p_left[1]), z=I(p_left[2])))
        verts.append(Point(x=I(p_right[0]), y=I(p_right[1]), z=I(p_right[2])))

    for i in range(n_pts):
        # Current pair: 2*i, 2*i+1; Next pair: 2*i+2, 2*i+3
        v0, v1 = 2 * i, 2 * i + 1
        v2, v3 = 2 * i + 2, 2 * i + 3
        # Front face
        faces.append((v0, v1, v3))
        faces.append((v0, v3, v2))
        # Back face (for visible double-sided translucent rendering)
        faces.append((v0, v3, v1))
        faces.append((v0, v2, v3))

    return Mesh(
        name=f"fieldline_ribbon_{ribbon_id}",
        vertices=verts,
        faces=faces,
        is_watertight=False,
        color=color,
        source_type="synthetic",
    )


# ── Assembly ─────────────────────────────────────────────────────────────

def generate_project() -> Project:
    laps = coil_laps()
    n_coils = distinct_coils()
    print(f"[stellarator] l={L} N_p={N_P} -> {n_coils} distinct filament(s), "
          f"{laps} lap(s) each ({2 * L} k-values collapse to {n_coils})")

    proj = Project(name="Classical Helical Stellarator")
    site = Site(name="machine_hall")

    # 1. Helical Coils (Copper/Bronze) + Purple Near-Field Flux Sheaths
    for k in range(n_coils):
        assert_coil_closes(k)
        coil = coil_mesh(k)
        wrapper = Element(name=f"coil_{k}", ifc_class="IfcBuildingElementProxy")
        wrapper.add(coil)
        site.add(wrapper, carve="none")

        # Ampèrian near-field flux ribbons looping directly around this coil tube
        vivid_purple = ["#8A2BE290", "#9400D390", "#BA55D390"]
        for j, (offset_ang, r_sheath) in enumerate([(0.0, 270.0), (math.pi, 270.0), (math.pi / 2.0, 310.0)]):
            near_ribbon = coil_near_field_ribbon_mesh(
                coil_k=k,
                ribbon_id=j,
                offset_angle=offset_ang,
                spiral_turns_per_lap=48.0,
                sheath_radius=r_sheath,
                width_mm=45.0,
                color=vivid_purple[j % len(vivid_purple)],
            )
            near_elem = Element(
                name=f"coil_{k}_near_field_{j}",
                ifc_class="IfcBuildingElementProxy",
            )
            near_elem.add(near_ribbon)
            site.add(near_elem, carve="none")

    print(f"[stellarator] Added {n_coils * 3} near-field coil flux ribbons in transparent purple")

    # 2. Core Plasma Boundary (Translucent Magenta/Pink)
    plasma = Element(name="plasma", ifc_class="IfcBuildingElementProxy")
    plasma.add(plasma_mesh())
    site.add(plasma, carve="none")

    # 3. Core Confinement Magnetic Fieldline Ribbons (Transparent Blue)
    # Generate nested magnetic flux ribbons at different radii and starting poloidal angles
    flux_radii = [550, 750, 1050, 1350]  # mm from magnetic axis
    phase_offsets = [0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0]
    colors = ["#00BFFF44", "#1E90FF4D", "#00A2FF55", "#4169E150"]

    ribbon_count = 0
    for r_idx, r_surf in enumerate(flux_radii):
        for th_idx, th_0 in enumerate(phase_offsets[:2]):  # 2 ribbons per flux shell
            ribbon_mesh = fieldline_ribbon_mesh(
                ribbon_id=ribbon_count,
                surface_r=r_surf,
                theta_0=th_0 + (r_idx * 0.2),
                iota=1.25 + 0.1 * (r_idx / len(flux_radii)),  # shear: iota increases with radius
                laps=4,
                width_mm=45.0,
                color=colors[r_idx % len(colors)],
            )
            ribbon_elem = Element(
                name=f"magnetic_fieldline_{ribbon_count}",
                ifc_class="IfcBuildingElementProxy",
            )
            ribbon_elem.add(ribbon_mesh)
            site.add(ribbon_elem, carve="none")
            ribbon_count += 1

    print(f"[stellarator] Added {ribbon_count} core magnetic fieldline ribbons in transparent blue")

    proj.add(site)
    return proj


result = generate_project()

if __name__ == "__main__":
    compile_main()
