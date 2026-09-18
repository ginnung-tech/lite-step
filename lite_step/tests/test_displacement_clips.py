"""Displacement + clips (spec-displacement-miter-clip).

Two composing rules:

* **miter() is an authored joint (option a).** A mitered pair records a shared
  joint id and is SKIPPED by the displacement auto-carve — the miter clips alone
  resolve the corner (each leaf keeps its half, flush, no hole/double). The old
  redundant carve only deepened the CSG (web-ifc dropped the deepest chain).

* **Clips ride along on NON-mitered operands (option b).** A ``.clip()``ed
  element used as a carve operand by ``.difference()``/``.void()``, or by the
  displacement carve of a non-mitered overlap, removes only its CLIPPED (kept)
  volume — so a clipped void/occupant excavates only its kept half.

Pins:
1. Two mitered leaves → each wall's kernel volume is EXACTLY its kept half; the
   pair sums to the union (miter clips do this — no auto-carve between them).
2. The same with LAYERED leaves (multi-item bodies) — volumes still exact,
   per-slice IfcBooleanClippingResult still present, no operand clips on top.
3. A mitered PAIR records no auto-carve (0 cuts); a mitered RING emits 0
   IfcBooleanResult (acceptance — all four walls present).
4. A clipped occupant on TERRAIN excavates only its kept half.
5. A clipped `.void()`/`.difference()` operand removes only its kept half.
"""

from __future__ import annotations

import numpy as np
import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.geom  # noqa: E402

from lite_step.compiler.displacement import apply_displacement
from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Project, Point, Box, Wall, Site, Mesh, Element, Material, LayerSet, miter,
)
from lite_step.models.project import Storey

#: mm³ -> m³. These tests NORMALIZE (see ``_proj``), so the emitted IFC is in
#: metres and every authored-millimetre volume below converts by this. The
#: figures stay written as mm products because that is how the fixtures are
#: authored — converting at the end keeps the arithmetic checkable by eye.
MM3 = 1e-9


def _proj(*elements, site=None) -> Project:
    """Build the fixture and normalize it, as every production caller does.

    Handing an un-normalized mm project straight to ``generate_ifc`` drops the
    joint: the angle is derived by ``normalize_project_to_meters``, so the leaf
    comes out at its full uncut volume with ``csg depth: max=0`` and no error
    anywhere. ``generate_ifc`` refuses such a model outright, and it exercises a
    path no production caller takes.
    """
    p = Project(name="t")
    s = Storey(elevation=0)
    for e in elements:
        s.add(e)
    p.add_storey(s)
    if site is not None:
        p.add(site)
    return normalize_project_to_meters(p)


def _find(proj, predicate):
    """The first element under ``proj`` matching ``predicate``.

    ``_proj`` deep-copies (normalize does), so a test that inspects an element
    AFTER displacement must reach the copy — the object it constructed is no
    longer the one the compiler touched. Reading the original does not fail
    loudly, it just reports the un-displaced state, and an assertion of the
    ``assert not x._cuts`` shape passes on it whatever the code does.
    """
    def walk(elem):
        if predicate(elem):
            return elem
        for attr in ("_elements", "_openings", "_cuts", "_adds",
                     "_intersects", "_fills", "_voids"):
            for child in (getattr(elem, attr, None) or []):
                hit = walk(child)
                if hit is not None:
                    return hit
        return None

    for storey in proj.storeys:
        for elem in storey.elements:
            hit = walk(elem)
            if hit is not None:
                return hit
    for site in getattr(proj, "sites", []) or []:
        hit = walk(site)
        if hit is not None:
            return hit
    raise AssertionError("no element matched — the fixture changed shape")


def _kernel_volumes_by_type(ifc_text: str, ifc_type: str) -> list:
    import os
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        with open(path, "w") as f:
            f.write(ifc_text)
        model = ifcopenshell.open(path)
        settings = ifcopenshell.geom.settings()
        vols = []
        for prod in model.by_type(ifc_type):
            sh = ifcopenshell.geom.create_shape(settings, prod)
            v = np.array(sh.geometry.verts).reshape(-1, 3)
            fcs = np.array(sh.geometry.faces).reshape(-1, 3)
            t = v[fcs]
            vols.append(abs(np.einsum('ij,ij->i', t[:, 0],
                                      np.cross(t[:, 1], t[:, 2])).sum() / 6.0))
        return vols
    finally:
        os.unlink(path)


def _two_mitered_leaves(layers=None):
    """Two perpendicular wall leaves overlapping in the 300×300 corner column
    at the origin, mitered on the 45° bisector through the OUTER corner."""
    wa = Wall(name="south")
    body_a = Box(name="body", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=300, z=2700))
    wa.add(body_a)
    wb = Wall(name="west")
    body_b = Box(name="body", start=Point(x=0, y=0, z=0), end=Point(x=300, y=4000, z=2700))
    wb.add(body_b)
    miter(body_a, body_b, at=Point(x=0, y=0, z=0), edge=(0, 0, 1))
    if layers is not None:
        wa.layers = layers                             # thin in y → outward along y
        # wb is thin in x → its outward points along x (same buildup, its own placement)
        wb.layers = LayerSet(name=layers.name, outward=(-1, 0, 0), layers=list(layers.layers))
    return wa, wb


FULL = 4000 * 300 * 2700 * MM3    # one full leaf
CORNER = 300 * 300 * 2700 * MM3   # the overlap column
KEPT = FULL - CORNER / 2          # each leaf keeps its half of the corner
UNION = 2 * FULL - CORNER


def test_mitered_leaves_compile_to_flush_corner():
    wa, wb = _two_mitered_leaves()
    res = generate_ifc(_proj(wa, wb), source_code=None)
    assert res.success, res.error
    vols = _kernel_volumes_by_type(res.ifc_content, "IfcWall")
    assert len(vols) == 2
    for v in vols:
        assert abs(v - KEPT) / KEPT < 1e-3, f"leaf volume {v} != kept half {KEPT}"
    assert abs(sum(vols) - UNION) / UNION < 1e-3   # no hole, no double


def test_mitered_layered_leaves_keep_volumes_and_slice_clips():
    buildup = LayerSet(name="miter_test", outward=(0, -1, 0), layers=[
        Material(key="Brick_Red_DK", thickness_mm=108),
        Material(key="Gypsum_Standard", thickness_mm=192)])
    wa, wb = _two_mitered_leaves(layers=buildup)
    res = generate_ifc(_proj(wa, wb), source_code=None)
    assert res.success, res.error
    vols = _kernel_volumes_by_type(res.ifc_content, "IfcWall")
    assert len(vols) == 2
    for v in vols:
        assert abs(v - KEPT) / KEPT < 1e-3
    # per-slice miter clips (2 walls × 2 slices, each clipped). The mitered pair
    # does not auto-carve (authored joint), so there are NO operand clipping
    # results on top — only the miter clips remain.
    assert res.ifc_content.count("IFCBOOLEANCLIPPINGRESULT") >= 4


def test_mitered_pair_does_not_auto_carve():
    """miter() is an authored joint — the pair records mutual no-auto-carve, so
    apply_displacement skips the (redundant) solid-host carve between them. The
    miter clips alone resolve the corner; the carve only inflated the CSG, which
    is what web-ifc dropped. Non-mitered clipped operands still carve — that's, pinned by the void/terrain tests below."""
    wa, wb = _two_mitered_leaves()
    proj = _proj(wa, wb)
    apply_displacement(proj)
    # Read the NORMALIZED copies — the originals never see the carve pass, and
    # `assert not x._cuts` on them passes whatever the code does.
    for leaf in ("south", "west"):
        wall = _find(proj, lambda e, n=leaf: getattr(e, "name", None) == n)
        assert not wall._elements[0]._cuts      # authored joint → no auto-carve


def _ring_leaf(name, start, end):
    w = Wall(name=name)
    w.add(Box(name="body", start=start, end=end))
    return w


def test_mitered_ring_emits_no_boolean_result_from_leaf_carves():
    """Acceptance: a 4-leaf mitered ring emits ZERO IfcBooleanResult from leaves
    carving each other — only the miter IfcBooleanClippingResults remain, and all
    four walls are present. This is the redundant-CSG depth web-ifc choked on."""
    t, hx, hy = 330, 6000, 4500
    ox, oy = hx + t, hy + t
    s = _ring_leaf("south", Point(x=-ox, y=-oy, z=0), Point(x=ox, y=-hy, z=3000))
    n = _ring_leaf("north", Point(x=-ox, y=hy, z=0),  Point(x=ox, y=oy, z=3000))
    w = _ring_leaf("west",  Point(x=-ox, y=-oy, z=0), Point(x=-hx, y=oy, z=3000))
    e = _ring_leaf("east",  Point(x=hx, y=-oy, z=0),  Point(x=ox, y=oy, z=3000))
    sb, nb, wb, eb = (x._elements[0] for x in (s, n, w, e))
    for A, B, at in [(sb, wb, (-ox, -oy)), (sb, eb, (ox, -oy)),
                     (nb, wb, (-ox, oy)), (nb, eb, (ox, oy))]:
        miter(A, B, at=Point(x=at[0], y=at[1], z=0), edge=(0, 0, 1))
    res = generate_ifc(_proj(s, n, w, e), source_code=None)
    assert res.success, res.error
    assert res.ifc_content.count("IFCBOOLEANRESULT(") == 0          # no leaf-carves
    assert res.ifc_content.count("IFCBOOLEANCLIPPINGRESULT(") > 0   # miter clips remain
    assert res.ifc_content.count("IFCWALL(") == 4                   # no dropped side


def _terrain(half=10000, depth=15000, top=0) -> Mesh:
    lo, hi, bot = -half, half, top - depth
    V = [(lo, lo, bot), (hi, lo, bot), (hi, hi, bot), (lo, hi, bot),
         (lo, lo, top), (hi, lo, top), (hi, hi, top), (lo, hi, top)]
    F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]
    return Mesh(name="ground", vertices=[Point(x=a, y=b, z=c) for a, b, c in V],
                faces=F, is_watertight=True)


def test_clipped_occupant_excavates_only_kept_half_of_terrain():
    from manifold3d import Manifold, Mesh as MMesh
    terrain = _terrain()
    wrapper = Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN",
                      name="terrain")
    wrapper.add(terrain)
    site = Site(name="site")
    site.add(wrapper)
    foot = Box(name="strip", start=Point(x=-3000, y=-3000, z=-900),
               end=Point(x=3000, y=3000, z=0))
    # clip away the +x half of the footing → only the -x half excavates
    foot.clip(origin=Point(x=0, y=0, z=-450), normal=(1, 0, 0))
    proj = _proj(foot, site=site)
    apply_displacement(proj)
    # The carve lands on the NORMALIZED copy of the terrain mesh, not on the
    # `terrain` object this test built.
    carved = _find(proj, lambda e: getattr(e, "vertices", None) is not None)
    v = np.array([[p.x, p.y, p.z] for p in carved.vertices], dtype=np.float32)
    f = np.array(carved.faces, dtype=np.uint32)
    vol = Manifold(MMesh(vert_properties=v, tri_verts=f)).volume()
    block = 20000 * 20000 * 15000 * MM3
    half_pit = 3000 * 6000 * 900 * MM3    # only x∈[-3000,0] excavated
    assert abs(vol - (block - half_pit)) / block < 1e-4


def test_clipped_void_operand_removes_only_kept_half():
    host = Wall(name="host")
    body = Box(name="body", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=300, z=2700))
    host.add(body)
    tool = Box(start=Point(x=1000, y=0, z=1000), end=Point(x=2000, y=300, z=2000))
    tool.clip(origin=Point(x=1500, y=150, z=1500), normal=(0, 0, 1))  # remove top half
    body.difference(tool)
    res = generate_ifc(_proj(host), source_code=None)
    assert res.success, res.error
    vols = _kernel_volumes_by_type(res.ifc_content, "IfcWall")
    removed = 1000 * 300 * 500 * MM3      # bottom half of the tool only
    expect = FULL - removed
    assert abs(vols[0] - expect) / expect < 1e-3


def test_no_operand_clip_warning_for_bounded_operands(caplog):
    import logging
    wa, wb = _two_mitered_leaves()
    with caplog.at_level(logging.WARNING):
        res = generate_ifc(_proj(wa, wb), source_code=None)
    assert res.success, res.error
    assert not [r for r in caplog.records if "half-space" in r.getMessage()
                and "ignored" in r.getMessage()]
