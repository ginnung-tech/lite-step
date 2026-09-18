"""Regression: ``normalize_project_to_meters`` scales MESH boolean operands.

Bug (same class as — a normalization-coverage gap): the
mm→m normalizer scaled a terrain ``Mesh``'s own ``vertices`` but NOT the
boolean-operand solids attached to a mesh host, nor ``Mesh``-typed operands
anywhere. So after normalization the host was in meters and the cut tool
still in mm — off by ~1000× — and ``apply_displacement``'s manifold3d
subtraction silently removed nothing. No error, no warning.

Two routes were affected:

* ``terrain.difference(tool)`` / ``.union()`` / ``.intersection()`` →
  operand in the mesh's ``_cuts`` / ``_adds`` / ``_intersects``. The mesh
  host never had ``_normalize_private_modifiers`` called on it, so NEITHER a
  ``Box`` NOR a ``Mesh`` operand was scaled.
* ``site.void(Mesh)`` → operand in ``site._voids``. The fix scaled
  ``Box`` void operands but a ``Mesh`` operand's ``vertices`` were untouched.

The fix (each operand list scaled in exactly one place — no double-scaling):
``_normalize_sub_element`` now scales a ``Mesh`` operand's ``vertices``;
``_normalize_private_modifiers`` now also covers ``_intersects`` and is
invoked on mesh hosts (via ``_normalize_mesh``). ``_voids`` stays owned by
``_normalize_voids_recursive``.

Tests assert the operand landed in METERS after normalization AND — for the
carve path — that geometry actually changed (not a silent no-op), mirroring
the position-asserting style.
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc
from lite_step.models import Project, Site, Mesh, Box, Point


def _box_mesh(x0, y0, z0, x1, y1, z1) -> Mesh:
    """A watertight axis-aligned box Mesh (mm) usable as a boolean operand.

    Vertex index = 4*ix + 2*iy + iz for ix,iy,iz in {0,1}."""
    verts = [Point(x=x, y=y, z=z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    faces = [
        (0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5),
        (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3),
        (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
    ]
    return Mesh(vertices=verts, faces=faces, is_watertight=True)


def _terrain_site():
    """A bare heightmap terrain mesh under a Site (local fixture — no corpus/
    coupling). Bare is fine here: explicit ``.difference()`` folding is a
    geometry feature open to every mesh, independent of terrain semantics."""
    from lite_step.tests._fixtures import terrain_mesh
    b = Project(name="t")
    site = Site(name="site")
    terr = terrain_mesh(20000, 20000)  # flat top z=0, mm
    site.add(terr)
    return b, site, terr


# ── mesh._cuts operand scaling ──────────────────────────────────────────────

def test_mesh_difference_box_operand_scaled_to_meters():
    """The spec repro: ``terrain.difference(Box)`` — the Box operand in the mesh's
    ``_cuts`` must be in meters after normalization (was mm → carve was inert)."""
    b, site, terr = _terrain_site()
    terr.difference(Box(start=Point(x=-6500, y=-5000, z=-500),
                        end=Point(x=6500, y=-4500, z=50)))
    b.add(site)

    out = normalize_project_to_meters(b)
    mesh = [m for m in out.sites[0]._elements if isinstance(m, Mesh)][0]
    cut = mesh._cuts[0]
    assert abs(cut.start.x - (-6.5)) < 1e-9, f"operand still in mm: {cut.start.x}"
    assert abs(cut.end.z - 0.05) < 1e-9, f"operand end not scaled: {cut.end.z}"


def test_mesh_difference_box_actually_carves():
    """End-to-end: the scaled operand makes the manifold3d carve add geometry to
    the terrain mesh — proof the difference is not a silent no-op."""
    pytest.importorskip("manifold3d")
    b, site, terr = _terrain_site()
    n_before = len(terr.vertices)
    terr.difference(Box(start=Point(x=-6500, y=-5000, z=-500),
                        end=Point(x=6500, y=-4500, z=50)))
    b.add(site)

    out = normalize_project_to_meters(b)
    assert generate_ifc(out).success
    mesh = [m for m in out.sites[0]._elements if isinstance(m, Mesh)][0]
    assert len(mesh.vertices) > n_before, "carve added no geometry — still a no-op"


def test_mesh_difference_mesh_operand_scaled_to_meters():
    """A ``Mesh``-typed operand of ``terrain.difference(mesh)`` has its ``vertices``
    scaled to meters (Fix 1) — the stenkant/gravel-bed use case needs a mesh cut
    to follow a varying surface, which a Box cannot."""
    b, site, terr = _terrain_site()
    tool = _box_mesh(-6500, -5000, -500, 6500, -4500, 50)
    terr.difference(tool)
    b.add(site)

    out = normalize_project_to_meters(b)
    mesh = [m for m in out.sites[0]._elements if isinstance(m, Mesh)][0]
    op = mesh._cuts[0]
    assert isinstance(op, Mesh)
    assert abs(op.vertices[0].x - (-6.5)) < 1e-9, f"mesh operand still mm: {op.vertices[0].x}"
    assert generate_ifc(out).success


# ── Mesh operand in the void route (parity with site.void(Box)) ───────────────

def test_site_void_mesh_operand_scaled_to_meters():
    """``site.void(Mesh)`` scales the mesh operand's vertices — parity with the
    already-working ``site.void(Box)`` path (#428 scaled Box void operands only)."""
    b, site, _terr = _terrain_site()
    vtool = _box_mesh(-6500, -5000, -500, 6500, -4500, 50)
    site.void(vtool)
    b.add(site)

    out = normalize_project_to_meters(b)
    void_op = out.sites[0]._voids[0]
    assert isinstance(void_op, Mesh)
    assert abs(void_op.vertices[0].x - (-6.5)) < 1e-9, f"void mesh operand still mm: {void_op.vertices[0].x}"


def test_site_void_box_operand_unchanged():
    """Guard: the already-working ``site.void(Box)`` path still scales its operand
    to meters (no regression from routing mesh hosts through _normalize_mesh)."""
    b, site, _terr = _terrain_site()
    clear = Box(start=Point(x=-6500, y=-5000, z=-500), end=Point(x=6500, y=-4500, z=50))
    site.void(clear)
    b.add(site)

    out = normalize_project_to_meters(b)
    void_op = out.sites[0]._voids[0]
    assert abs(void_op.start.x - (-6.5)) < 1e-9, f"box void operand: {void_op.start.x}"


# ── _intersects was scaled nowhere before (Fix 2) ─────────────────────────────

def test_intersects_operand_scaled_to_meters():
    """``.intersection()`` operands live in ``_intersects``. A Box
    intersection operand on a mesh host must land in meters."""
    b, site, terr = _terrain_site()
    terr.intersection(Box(start=Point(x=-3000, y=-3000, z=-1000),
                          end=Point(x=3000, y=3000, z=100)))
    b.add(site)

    out = normalize_project_to_meters(b)
    mesh = [m for m in out.sites[0]._elements if isinstance(m, Mesh)][0]
    inter = mesh._intersects[0]
    assert abs(inter.start.x - (-3.0)) < 1e-9, f"_intersects operand still mm: {inter.start.x}"
