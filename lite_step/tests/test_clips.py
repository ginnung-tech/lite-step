"""``.clip()`` — a half-space cut, and ``miter()`` — the derived joint cut.

    body.clip(origin=Point, normal=(dx,dy,dz))   # remove the normal-side half-space
    miter(a, b, at=corner, edge=(0,0,1))          # two elements → two bisector clips

``.clip()`` bakes an ``IfcBooleanClippingResult`` over an ``IfcHalfSpaceSolid``
into the body CSG (like ``.difference()`` — invisible to schedules; queries read
the AUTHORED shape). The DSL normal points at the REMOVED side. ``miter()`` takes
the joint corner ``at`` and the joint-edge vector ``edge`` (both required), then
derives the cut angle from where each element's centroid sits relative to ``at`` in
the plane ⟂ ``edge``: the bisector splits the inner angle, so a 90° corner yields
two 45° faces, and a vertical ``edge`` keeps the plane vertical.

Model tests assert the plumbing; end-to-end tests assert the emitted IFC and, via
kernel tessellation, WHICH side survived (the two sign traps this feature turns on).
"""
from __future__ import annotations

import numpy as np
import pytest

from lite_step.compiler.executor import (
    execute_lite_step_script,
    normalize_project_to_meters,
)
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Project, Box, Mesh, Point, Point2D, Element, Site, HalfSpace, miter,
    collect_consumed_operand_ids,
)
from lite_step.models.project import Storey
from lite_step.strict import enable_strict_int_mm, disable_strict_int_mm


@pytest.fixture()
def strict():
    """Run a test under strict int-mm (the production default for scripts)."""
    enable_strict_int_mm()
    try:
        yield
    finally:
        disable_strict_int_mm()


# ── model layer ───────────────────────────────────────────────────────────────
def test_clip_is_chainable_and_stores_halfspace():
    b = Box(name="block", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=2000, z=1000))
    ret = b.clip(origin=Point(x=1000, y=0, z=0), normal=(0, -1, 0))
    assert ret is b                                   # chainable
    assert len(b._clips) == 1
    hs = b._clips[0]
    assert isinstance(hs, HalfSpace)
    assert hs.normal == (0.0, -1.0, 0.0)
    # chainable composition
    b.clip(origin=Point(x=0, y=0, z=500), normal=(0, 0, 1))
    assert len(b._clips) == 2


def test_halfspace_is_frozen_and_rejects_zero_normal():
    hs = HalfSpace(origin=Point(x=0, y=0, z=0), normal=(1, 0, 0))
    with pytest.raises(Exception):
        hs.normal = (0, 1, 0)                          # frozen
    with pytest.raises(Exception, match="non-zero"):
        HalfSpace(origin=Point(x=0, y=0, z=0), normal=(0, 0, 0))


def test_clip_origin_strict_normal_float(strict):
    """Under strict int-mm the plane ORIGIN is a Point (int mm gate applies);
    the NORMAL is a unit-free float tuple and is always accepted."""
    b = Box(name="x", start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000))
    b.clip(origin=Point(x=500, y=0, z=0), normal=(0.7071, -0.7071, 0.0))   # floats OK
    with pytest.raises(Exception):
        b.clip(origin=Point(x=500.5, y=0, z=0), normal=(1, 0, 0))          # float origin rejected


def test_clip_not_a_consumed_operand():
    """A clip is a plane on the receiver, not a consumed element — it must not
    appear in the consumed-operand set, and collecting must not crash on it."""
    host = Box(name="block", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=2000, z=1000))
    host.clip(origin=Point(x=1000, y=0, z=0), normal=(0, -1, 0))
    b = Project(name="t")
    s = Storey(elevation=0)
    s.add(host)
    b.add_storey(s)
    consumed = collect_consumed_operand_ids(b)     # no crash
    assert consumed == set()


def test_derivation_deep_copies_clips():
    tmpl = Box(name="a", start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000))
    tmpl.clip(origin=Point(x=500, y=0, z=0), normal=(0, -1, 0))
    copy = tmpl(name="b")
    assert len(copy._clips) == 1
    tmpl.clip(origin=Point(x=0, y=0, z=500), normal=(0, 0, 1))   # mutate template
    assert len(copy._clips) == 1, "derived copy must not share the _clips list"


# ── normalizer: origin mm→m, normal untouched (the bug class) ────────────
def _clipped_building(on_site: bool):
    b = Project(name="t")
    host = Box(name="block", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=4000, z=2000))
    host.clip(origin=Point(x=2000, y=1000, z=0), normal=(0.0, -1.0, 0.0))
    if on_site:
        site = Site(name="site")
        wrap = Element(ifc_class="IfcBuildingElementProxy", name="w")
        wrap.add(host)
        site.add(wrap)
        b.add(site)
    else:
        s = Storey(elevation=0)
        s.add(host)
        b.add_storey(s)
    return b, host


def _find_first_clip(elem):
    if getattr(elem, "_clips", None):
        return elem._clips[0]
    for attr in ("_elements", "_openings", "_cuts", "_adds", "_intersects",
                 "_fills", "_voids"):
        for child in (getattr(elem, attr, None) or []):
            hit = _find_first_clip(child)
            if hit is not None:
                return hit
    return None


@pytest.mark.parametrize("on_site", [False, True], ids=["storey", "site"])
def test_clip_origin_scaled_to_meters(on_site):
    b, _ = _clipped_building(on_site)
    out = normalize_project_to_meters(b)
    root = (out.sites[0] if on_site else out.storeys[0].elements[0])
    hs = _find_first_clip(root)
    assert hs is not None, "clip survived normalization"
    assert abs(hs.origin.x - 2.0) < 1e-9 and abs(hs.origin.y - 1.0) < 1e-9
    assert hs.normal == (0.0, -1.0, 0.0), "normal is unit-free — never scaled"


# ── end-to-end IFC (ifcopenshell backend) ─────────────────────────────────────
def _compile(script: str):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    r = execute_lite_step_script(script)
    assert r.success, r.error
    ifc = generate_ifc(r.project)
    assert ifc.success, ifc.error
    return ifcopenshell.file.from_string(ifc.ifc_content)


def _world_verts(model, product):
    ios = pytest.importorskip("ifcopenshell.geom")
    s = ios.settings()
    s.set(s.USE_WORLD_COORDS, True)
    shp = ios.create_shape(s, product)
    return np.array(shp.geometry.verts).reshape(-1, 3)


_CLIP_SCRIPT = """
from lite_step.models import Project, Box, Point
def generate_project():
    b = Project(name="clip")
    host = Box(name="block", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=2000, z=1000))
    # remove the +y half beyond y=1000 (normal points at the removed side)
    host.clip(origin=Point(x=0, y=1000, z=0), normal=(0, 1, 0))
    b.add(host)
    return b
result = generate_project()
"""


def test_clip_emits_clipping_result_and_halfspace():
    model = _compile(_CLIP_SCRIPT)
    assert model.by_type("IfcBooleanClippingResult"), "clip → IfcBooleanClippingResult"
    assert model.by_type("IfcHalfSpaceSolid"), "clip → IfcHalfSpaceSolid"


def test_clip_removes_the_normal_side():
    """SIGN PIN (solid): normal=(0,1,0) at y=1 m removes everything with y>1 m.
    Every surviving world vertex must satisfy y <= 1 m (+ kernel eps)."""
    model = _compile(_CLIP_SCRIPT)
    proxy = model.by_type("IfcBuildingElementProxy")[0]
    verts = _world_verts(model, proxy)
    assert verts[:, 1].max() <= 1.0 + 1e-4, verts[:, 1].max()
    assert verts[:, 1].min() < 0.1, "the kept y<1 m half is still there"


# ── mesh receiver (trim_by_plane) ─────────────────────────────────────────────
_MESH_CLIP_SCRIPT = """
from lite_step.models import Project, Site, Element, Mesh, Point, Point2D
def generate_project():
    b = Project(name="mc")
    site = Site(name="site")
    terr = Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN", name="terrain")
    m = Mesh(heightmap=[[0,0],[0,0]], depth=2000,
             corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=4000, y=4000))
    m.clip(origin=Point(x=0, y=2000, z=0), normal=(0, 1, 0))   # remove y>2 m
    terr.add(m)
    site.add(terr)
    b.add(site)
    return b
result = generate_project()
"""


def test_mesh_clip_trims_plane_side():
    """SIGN PIN (mesh): trim_by_plane keeps the side opposite the DSL normal.
    normal=(0,1,0) at y=2 m → surviving mesh verts all have y <= 2 m."""
    ios = pytest.importorskip("ifcopenshell")
    from lite_step.compiler.displacement import apply_displacement
    r = execute_lite_step_script(_MESH_CLIP_SCRIPT)
    assert r.success, r.error
    # trim_by_plane runs in displacement (normally inside generate_ifc) —
    # normalize + displace so the carved vertices are observable.
    nb = normalize_project_to_meters(r.project)
    apply_displacement(nb)
    from lite_step.models import Mesh as _M
    def _meshes(elem, out):
        if isinstance(elem, _M):
            out.append(elem)
        for a in ("_elements",):
            for c in (getattr(elem, a, None) or []):
                _meshes(c, out)
    got = []
    for site in nb.sites:
        _meshes(site, got)
    assert got, "found the terrain mesh"
    ys = [p.y for m in got for p in m.vertices]
    assert max(ys) <= 2.0 + 1e-4, f"clip kept y<=2 m; got max {max(ys)}"
    assert min(ys) < 0.1


# ── miter (derived joint) ─────────────────────────────────────────────────────
def _mitered(a, b, **kw):
    """``miter()`` + the derivation the compiler runs, so a unit test can read
    the plane without a full compile.

    The call RECORDS the joint and the angle is derived later, from both
    assemblies as they finally stand — that is what makes the result
    independent of where the line sits in the file (measured: eager derivation
    gave 45 degrees with a return wing added before the call and 33.8 with it
    added after). These tests therefore assert on the derived plane, not on
    what ``miter()`` returns.

    No normalisation here: ``at`` and the geometry only have to agree with each
    other, so the authored mm domain answers the same normal as metres.
    """
    from lite_step.compiler.composite_clip import derive_pending_miters

    ret = miter(a, b, **kw)
    derive_pending_miters(a)
    derive_pending_miters(b)
    return ret


def test_miter_perpendicular_boxes_two_45_faces():
    """Two bars meeting at the origin corner (one along +x, one along +y) get
    one clip each, on the SAME bisector plane with opposite normals → two 45°
    faces."""
    a = Box(name="a", start=Point(x=0, y=-100, z=0), end=Point(x=2000, y=100, z=300))
    c = Box(name="c", start=Point(x=-100, y=0, z=0), end=Point(x=100, y=2000, z=300))
    ra, rc = _mitered(a, c, at=Point(x=0, y=0, z=0), edge=(0, 0, 1))
    assert ra is a and rc is c
    assert len(a._clips) == 1 and len(c._clips) == 1
    na = np.array(a._clips[0].normal)
    nc = np.array(c._clips[0].normal)
    # opposite normals on the same plane
    assert np.allclose(na, -nc, atol=1e-9), (na, nc)
    # the bisector of +x and +y is the plane x=y; its normal is ±(1,-1,0)/√2
    assert abs(abs(na[0]) - 0.70710678) < 1e-6 and abs(na[2]) < 1e-9


def test_miter_parallel_raises():
    # collinear centroids along +x → no joint edge
    p1 = Box(name="p1", start=Point(x=100, y=-50, z=-50), end=Point(x=900, y=50, z=50))
    p2 = Box(name="p2", start=Point(x=1100, y=-50, z=-50), end=Point(x=1900, y=50, z=50))
    # Raised by the DERIVATION, not by the call: whether two elements run
    # parallel depends on geometry that is still being authored when miter()
    # returns.
    with pytest.raises(ValueError, match="parallel"):
        _mitered(p1, p2, at=Point(x=0, y=0, z=0), edge=(0, 0, 1))


def test_miter_centroid_on_corner_raises():
    a = Box(name="a", start=Point(x=-100, y=-100, z=-100), end=Point(x=100, y=100, z=100))  # centroid = origin
    c = Box(name="c", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=100, z=100))
    with pytest.raises(ValueError, match="joint edge"):
        _mitered(a, c, at=Point(x=0, y=0, z=0), edge=(0, 0, 1))


def test_miter_at_base_of_tall_walls_stays_vertical():
    """REGRESSION (vertical-edge bug): when ``at`` sits at the BASE of two
    tall walls, the joint edge is vertical and the miter plane must be too.
    v1 derived the edge from the raw 3D ``at``→centroid directions, so the
    ``at``-at-base vs centroid-at-mid-height offset tilted the plane out of
    vertical (z-component ≈ 0.09) and skewed the angle off 45° — the two
    faces then met on a skew line, one leaf showing a spurious extra cut.
    The tight 45° tolerance also pins the OBB run-axis derivation: the corner
    sits at the END of each long wall, so a corner→centroid direction would
    skew ~1° off (0.704) — only the oriented long axis gives exactly 45°."""
    # two 15 m-tall perpendicular leaves meeting at the SW outer corner,
    # ``at`` at the base (z=0) — the failing configuration from the villa.
    s = Box(name="s", start=Point(x=-6330, y=-4830, z=0), end=Point(x=6330, y=-4500, z=15000))
    w = Box(name="w", start=Point(x=-6330, y=-4830, z=0), end=Point(x=-6000, y=4830, z=15000))
    _mitered(s, w, at=Point(x=-6330, y=-4830, z=0), edge=(0, 0, 1))
    for leaf, label in ((s, "south"), (w, "west")):
        n = np.array(leaf._clips[0].normal)
        assert abs(n[2]) < 1e-9, f"{label} miter plane tilted out of vertical: {n}"
        # a 90° corner → exactly 45°: |nx| == |ny| == 1/√2 (OBB run-axis, not centroid)
        assert abs(abs(n[0]) - 0.70710678) < 1e-4, f"{label} not exactly 45°: {n}"
        assert abs(abs(n[1]) - 0.70710678) < 1e-4, f"{label} not exactly 45°: {n}"


def test_miter_explicit_edge_for_raked_joint():
    """``edge=`` fixes a non-vertical joint: two members meeting along a
    horizontal edge get a miter plane containing that edge."""
    a = Box(name="a", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=200, z=100))
    c = Box(name="c", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=100, z=2000))
    # joint runs along +x; the miter plane must contain the x-axis → nx ≈ 0
    _mitered(a, c, at=Point(x=0, y=0, z=0), edge=(1, 0, 0))
    for leaf in (a, c):
        n = np.array(leaf._clips[0].normal)
        assert abs(n[0]) < 1e-9, f"raked miter plane does not contain the edge: {n}"


def test_miter_zero_edge_raises():
    a = Box(name="a", start=Point(x=0, y=-100, z=0), end=Point(x=2000, y=100, z=300))
    c = Box(name="c", start=Point(x=-100, y=0, z=0), end=Point(x=100, y=2000, z=300))
    with pytest.raises(ValueError, match="zero vector"):
        miter(a, c, at=Point(x=0, y=0, z=0), edge=(0, 0, 0))


_MITER_SCRIPT = """
from lite_step.models import Project, Box, Point, miter
def generate_project():
    b = Project(name="miter")
    a = Box(name="a", start=Point(x=0, y=-100, z=0), end=Point(x=2000, y=100, z=300))
    c = Box(name="c", start=Point(x=-100, y=0, z=0), end=Point(x=100, y=2000, z=300))
    miter(a, c, at=Point(x=0, y=0, z=0), edge=(0, 0, 1))
    b.add(a); b.add(c)
    return b
result = generate_project()
"""


def test_miter_compiles_to_two_clipping_results():
    model = _compile(_MITER_SCRIPT)
    # two mitered members → one clipping result per RECEIVER, each over a
    # half-space. The mitered pair is an authored joint, so displacement does
    # NOT auto-carve between them (no operand clip on top) — exactly the two
    # receiver-side miter clips, guaranteed by the volume pins in
    # test_displacement_clips.py.
    assert len(model.by_type("IfcBooleanClippingResult")) >= 2
    assert len(model.by_type("IfcHalfSpaceSolid")) >= 2


def test_diagonal_clip_tessellates_nonempty():
    """REGRESSION GUARD: a DIAGONAL clip normal (every miter uses one) must
    tessellate to real geometry. Without an explicit in-plane RefDirection on
    the half-space plane, OCC composes a degenerate frame under the product
    placement and the clip silently yields EMPTY geometry — the whole feature
    would look like it 'lost' the mitered members."""
    script = """
from lite_step.models import Project, Box, Point
def generate_project():
    b = Project(name="d")
    slab = Box(name="slab", start=Point(x=5000, y=0, z=0), end=Point(x=8000, y=3000, z=200))
    slab.clip(origin=Point(x=8000, y=2000, z=0), normal=(1, 1, 0))
    b.add(slab)
    return b
result = generate_project()
"""
    model = _compile(script)
    proxy = model.by_type("IfcBuildingElementProxy")[0]
    verts = _world_verts(model, proxy)
    assert len(verts) > 0, "diagonal clip tessellated to EMPTY (RefDirection regression)"
    # the removed corner is beyond the x+y = 10 m plane; kept verts stay <= it
    assert (verts[:, 0] + verts[:, 1]).max() <= 10.0 + 1e-3


def test_miter_faces_meet_on_the_bisector():
    """The two mitered members abut on the shared bisector plane (x=y): each
    member's tessellated verts stay on its own side (no overlap, no gap)."""
    model = _compile(_MITER_SCRIPT)
    proxies = {p.Name.split(":")[-1]: p
               for p in model.by_type("IfcBuildingElementProxy")}
    vx = _world_verts(model, proxies["a"])
    vy = _world_verts(model, proxies["c"])
    assert len(vx) and len(vy)
    # member a (runs +x) keeps x >= y; member c (runs +y) keeps y >= x
    assert (vx[:, 0] >= vx[:, 1] - 1e-3).all(), "member a crossed the bisector"
    assert (vy[:, 1] >= vy[:, 0] - 1e-3).all(), "member c crossed the bisector"
