"""Classical helical stellarator (torsatron / heliotron), analytically.

Two closed-form models, no MHD solver:

1. **Helical winding law** — 2l thin-filament conductors wrapped on a torus
   of major radius R0, minor radius a_coil, with N_p helical turns:

       x_k(phi) = (R0 + a_coil*cos(N_p/l*phi + pi*k/l)) * cos(phi)
       y_k(phi) = (R0 + a_coil*cos(N_p/l*phi + pi*k/l)) * sin(phi)
       z_k(phi) =       a_coil*sin(N_p/l*phi + pi*k/l)

2. **Rotating-ellipse plasma boundary** — an ellipse (a, b) rotated by
   alpha = N_p*phi/2 as it travels the torus. This is the cheapest surface
   that reproduces the W7-X/HSX *topology* (rotating, elongated cross-
   section) without solving the equilibrium.

Both are exact; the idealisation is in the physics, not the geometry.

Two facts about the parameters decide the model, and both are asserted below
rather than assumed — get either wrong and the output is quietly incorrect
rather than loud:

**The coil does not close after one lap.** The helical phase advances by
2*pi*N_p/l per toroidal circuit. That is a multiple of 2*pi only when l
divides N_p. For the canonical l=2, N_p=5 it advances 5*pi — half a turn out
— so the filament closes only after TWO laps. Sweeping phi over [0, 2*pi)
yields an open C, and closing the tube over it would weld a seam between two
points that are not the same point.

**There are fewer distinct coils than values of k.** Advancing k by
2*N_p (mod 2l) lands on the same space curve, so the 2l starting phases
collapse into gcd(2*N_p, 2l) distinct filaments. At l=2, N_p=5 that is 2, not
4 — which is right, and is why LHD (l=2) has two continuous helical coils.
Emitting all 2l would place two coincident solids per coil.

Compile with::

    python 12_stellarator.py
"""

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

# ── Machine parameters (W7-X scale, mm) ──────────────────────────────────
R0 = 5_500          # major radius
A_COIL = 1_600      # helical winding minor radius
N_P = 5             # field periods / helical turns
L = 2               # l pairs -> 2l conductors

PLASMA_A = 400      # ellipse semi-axis (mm)
PLASMA_B = 900      # elongation kappa = b/a = 2.25
CONDUCTOR_R = 170   # coil conductor radius
CONDUCTOR_SIDES = 12

N_PHI_PLASMA = 120  # toroidal grid
N_THETA_PLASMA = 48  # poloidal grid
SEG_PER_LAP = 110   # coil polyline resolution


# ── 1. Helical coils ─────────────────────────────────────────────────────

def coil_laps() -> int:
    """Toroidal circuits before a filament closes on itself.

    Phase advance per lap is 2*pi*N_p/l; it returns to a multiple of 2*pi
    after l/gcd(N_p, l) laps.
    """
    return L // math.gcd(N_P, L)


def distinct_coils() -> int:
    """Filaments that are actually different curves.

    k and k + 2*N_p (mod 2l) trace the same curve, so the orbit count of
    that shift on Z_2l is gcd(2*N_p, 2l).
    """
    return math.gcd(2 * N_P, 2 * L)


def coil_point(k: int, phi: float) -> tuple:
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
    # And it must NOT close after fewer, or `laps` is overstated and the
    # filament is traced twice.
    if laps > 1:
        near = math.dist(p0, coil_point(k, 2 * math.pi))
        assert near > 1.0, f"coil {k} already closed after 1 lap; laps={laps} is wrong"


def _tube_frames(stations):
    """Rotation-minimising (parallel-transport) frames along the filament.

    Not the same as what ``Sweep`` would do: that derives profile orientation
    from a FIXED world-Z up-reference, which is right for building members
    and merely acceptable here. Transporting the frame costs nothing and is
    correct for any path.
    """
    n = len(stations)

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def norm(v):
        m = math.sqrt(sum(c * c for c in v)) or 1.0
        return (v[0] / m, v[1] / m, v[2] / m)

    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

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
        nrm.append(norm(tuple(pv[m] * ca + cr[m] * sa + axis[m] * dt * (1 - ca)
                              for m in range(3))))
    return tang, nrm


def coil_mesh(k: int) -> Mesh:
    """The filament as a closed tube MESH, not a Sweep.

    Why not ``Sweep``: a multi-point path emits one IfcExtrudedAreaSolid per
    segment, mitred at every joint by half-space clipping. Correct and cheap
    for a four-segment window frame; pathological at 220 segments. The first
    version of this file did exactly that and produced 440 extrusions carved
    by 880 IfcHalfSpaceSolid in a boolean tree 10 deep — right on the
    compiler's own `csg depth budget 10` ceiling.

    An IfcHalfSpaceSolid is an INFINITE solid, so every consumer has to bound
    and cut it. ifcopenshell's iterator (which is what Blender Bonsai uses to
    import) produced no shape at all for a single coil in over two minutes,
    while the plasma mesh beside it loaded in 3.4 s. The file was valid and
    unopenable, which is the worst pair.

    A tube mesh has zero booleans, imports instantly, and gets a
    rotation-minimising frame for free.
    """
    laps = coil_laps()
    n = SEG_PER_LAP * laps
    stations = [coil_point(k, 2 * math.pi * laps * i / n) for i in range(n)]
    tang, nrm = _tube_frames(stations)

    verts, faces = [], []
    for i in range(n):
        t, u = tang[i], nrm[i]
        w = (t[1] * u[2] - t[2] * u[1], t[2] * u[0] - t[0] * u[2],
             t[0] * u[1] - t[1] * u[0])
        c = stations[i]
        for m in range(CONDUCTOR_SIDES):
            a = 2 * math.pi * m / CONDUCTOR_SIDES
            ca, sa = math.cos(a) * CONDUCTOR_R, math.sin(a) * CONDUCTOR_R
            verts.append(Point(x=I(c[0] + u[0] * ca + w[0] * sa),
                               y=I(c[1] + u[1] * ca + w[1] * sa),
                               z=I(c[2] + u[2] * ca + w[2] * sa)))
    for i in range(n):
        nx = (i + 1) % n            # closed loop: last ring wraps to the first
        for m in range(CONDUCTOR_SIDES):
            mm = (m + 1) % CONDUCTOR_SIDES
            a, b = i * CONDUCTOR_SIDES + m, i * CONDUCTOR_SIDES + mm
            c, d = nx * CONDUCTOR_SIDES + m, nx * CONDUCTOR_SIDES + mm
            faces.append((a, b, d))
            faces.append((a, d, c))
    # A tube closed on both loops has no boundary — every edge in two faces.
    return Mesh(name=f"helical_coil_{k}", vertices=verts, faces=faces,
                is_watertight=True, color="#C9752B", source_type="synthetic")


# ── 2. Rotating-ellipse plasma boundary ──────────────────────────────────

def plasma_point(theta: float, phi: float) -> tuple:
    alpha = N_P * phi / 2.0
    r_local = PLASMA_A * math.cos(theta) * math.cos(alpha) - PLASMA_B * math.sin(theta) * math.sin(alpha)
    z = PLASMA_A * math.cos(theta) * math.sin(alpha) + PLASMA_B * math.sin(theta) * math.cos(alpha)
    r = R0 + r_local
    return (r * math.cos(phi), r * math.sin(phi), z)


def plasma_seam_shift() -> int:
    """Poloidal index offset needed to close the toroidal seam.

    Over one full circuit alpha advances by N_p*pi. An ellipse maps onto
    itself under a pi rotation with theta -> theta + pi, so for ODD N_p the
    ring at phi=2*pi is ring 0 REPARAMETERISED, not ring 0. Wrapping the
    grid without this offset stitches the seam to the wrong vertices and
    produces a twisted, non-watertight band.
    """
    shift = (N_THETA_PLASMA // 2) if (N_P % 2) else 0
    # Verify rather than trust the parity argument.
    for j in (0, N_THETA_PLASMA // 3, N_THETA_PLASMA // 2):
        theta = 2 * math.pi * j / N_THETA_PLASMA
        end = plasma_point(theta, 2 * math.pi)
        jj = (j + shift) % N_THETA_PLASMA
        start = plasma_point(2 * math.pi * jj / N_THETA_PLASMA, 0.0)
        assert math.dist(end, start) < 1.0, (
            f"plasma seam shift {shift} is wrong at j={j}: "
            f"{math.dist(end, start):.1f} mm apart"
        )
    return shift


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
        # Only the wrap row carries the reparameterisation.
        off = shift if i == N_PHI_PLASMA - 1 else 0
        for j in range(N_THETA_PLASMA):
            a, b = v(i, j), v(i, j + 1)
            c, d = v(nxt, j + off), v(nxt, j + 1 + off)
            faces.append((a, b, d))
            faces.append((a, d, c))

    # A torus grid closed on both loops has every edge in exactly two
    # triangles — no boundary, so it is watertight by construction.
    return Mesh(
        name="plasma_boundary",
        vertices=verts,
        faces=faces,
        is_watertight=True,
        color="#FF6B9D80",
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

    for k in range(n_coils):
        assert_coil_closes(k)
        coil = coil_mesh(k)
        # Geometry and semantics are separate layers: the Sweep says what
        # SHAPE it is, the Element says what it IS. A fusion magnet has no
        # IFC class — this is an honest Proxy, not a mislabelled Beam.
        wrapper = Element(name=f"coil_{k}", ifc_class="IfcBuildingElementProxy")
        wrapper.add(coil)
        site.add(wrapper, carve="none")

    plasma = Element(name="plasma", ifc_class="IfcBuildingElementProxy")
    plasma.add(plasma_mesh())
    site.add(plasma, carve="none")

    proj.add(site)
    return proj


result = generate_project()

if __name__ == "__main__":
    compile_main()
