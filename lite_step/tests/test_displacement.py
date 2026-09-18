"""Universal geometric displacement — an overlapping solid carves its host
(only the shared volume is removed), and the occupant still renders (hide it →
void). Pure-geometry trigger, no labels. Solid-vs-solid pairs carve
one-directionally (smaller carves larger — the direction rule), so junctions
render seamlessly and equal twins stay ambiguous.

Model-level tests (fast) assert the pass wiring; end-to-end tests assert the
emitted IFC (native boolean for solid hosts, watertight carved mesh for the
terrain host)."""

from __future__ import annotations

import numpy as np

from lite_step.compiler.displacement import apply_displacement
# The padded-AABB derivation moved to compiler/extent.py and became public:
# ``BimElement.authored_aabb()`` reads the same function, so a second copy
# would let the carve trigger and the authoring query disagree.
from lite_step.compiler.extent import padded_aabb
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Project, Point, Box, Extrude, Site, Mesh, Beam, Bar, Wall, Window, Sweep, Material,
)
from lite_step.models.project import Storey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bldg(*elements, site=None) -> Project:
    b = Project(name="t")
    s = Storey(elevation=0)
    for e in elements:
        s.add(e)
    b.add_storey(s)
    if site is not None:
        b.add(site)
    return b


def _terrain(half=10000, depth=15000, top=0, name="ground") -> Mesh:
    """A watertight terrain block: top surface at ``top``, bottom at ``top-depth``."""
    lo, hi, bot = -half, half, top - depth
    V = [(lo, lo, bot), (hi, lo, bot), (hi, hi, bot), (lo, hi, bot),
         (lo, lo, top), (hi, lo, top), (hi, hi, top), (lo, hi, top)]
    F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]
    return Mesh(name=name, vertices=[Point(x=a, y=b, z=c) for a, b, c in V],
                faces=F, is_watertight=True)


def _wrap(mesh: Mesh):
    """The canonical terrain form — host candidacy is semantic, so terrain
    HOST fixtures wrap the mesh in Element(IfcGeographicElement, TERRAIN)."""
    from lite_step.models import Element
    w = Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN",
                name="terrain")
    w.add(mesh)
    return w


def _mesh_is_watertight(mesh: Mesh) -> bool:
    from manifold3d import Manifold, Mesh as MMesh
    v = np.array([[p.x, p.y, p.z] for p in mesh.vertices], dtype=np.float32)
    f = np.array(mesh.faces, dtype=np.uint32)
    m = Manifold(MMesh(vert_properties=v, tri_verts=f))
    return (not m.is_empty()) and m.genus() == 0


# ---------------------------------------------------------------------------
# Solid host
# ---------------------------------------------------------------------------


def test_solid_host_gains_cut_and_occupant_still_renders():
    big = Box(name="host", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=4000, z=4000))
    small = Box(name="occ", start=Point(x=1000, y=1000, z=1000), end=Point(x=2000, y=2000, z=2000))
    b = _bldg(big, small)
    apply_displacement(b)
    assert len(big._cuts) == 1                      # host carved
    assert big._cuts[0] is not small               # by a COPY, not the original
    assert big._cuts[0].name is None               # anonymous cut tool
    assert small in b.storeys[0].elements          # occupant still renders


def test_solid_host_emits_native_boolean_result():
    big = Box(name="host", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=4000, z=4000),
              material="Concrete_C25-30")
    small = Box(name="occ", start=Point(x=1000, y=1000, z=1000), end=Point(x=2000, y=2000, z=2000),
                material="Concrete_C25-30")
    res = generate_ifc(_bldg(big, small), source_code=None)
    assert res.success, res.error
    assert "IFCBOOLEANRESULT" in res.ifc_content    # host is carved


def test_idempotent():
    big = Box(name="host", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=4000, z=4000))
    small = Box(name="occ", start=Point(x=1000, y=1000, z=1000), end=Point(x=2000, y=2000, z=2000))
    b = _bldg(big, small)
    apply_displacement(b)
    apply_displacement(b)
    assert len(big._cuts) == 1                      # not appended twice


def test_rebar_carves_beam_body_and_bars_render():
    """The defining example: bars enclosed in a beam body carve it; the bars
    still render (hide them → tube-shaped voids)."""
    beam = Beam(name="a1")
    body = Box(name="body", start=Point(x=-150, y=-2000, z=0), end=Point(x=150, y=2000, z=600))
    beam.add(body)
    bars = [Bar(path=[Point(x=0, y=-1800, z=35), Point(x=0, y=1800, z=35)],
                diameter=16, grade="B500B") for _ in range(1)]
    beam.add(*bars)
    b = _bldg(beam)
    apply_displacement(b)
    assert len(body._cuts) == 1                      # beam body carved by the bar
    # the real bar (not the copy) still lives under the beam → renders
    assert bars[0] in beam._elements


# ---------------------------------------------------------------------------
# Mesh host (terrain)
# ---------------------------------------------------------------------------


def test_mesh_host_watertight_pit_and_foundation_renders():
    terrain = _terrain()
    site = Site(name="site"); site.add(_wrap(terrain))
    foot = Box(name="strip", start=Point(x=-3000, y=-3000, z=-900), end=Point(x=3000, y=3000, z=0))
    b = _bldg(foot, site=site)
    before = len(terrain.vertices)
    apply_displacement(b)
    assert len(terrain.vertices) > before           # terrain was carved
    assert _mesh_is_watertight(terrain)             # still a closed volume
    assert foot in b.storeys[0].elements            # foundation still renders


def test_mesh_pit_has_correct_volume():
    terrain = _terrain(half=10000, depth=15000, top=0)   # 20000 x 20000 x 15000
    site = Site(name="site"); site.add(_wrap(terrain))
    foot = Box(name="strip", start=Point(x=-3000, y=-3000, z=-900), end=Point(x=3000, y=3000, z=0))
    apply_displacement(_bldg(foot, site=site))
    from manifold3d import Manifold, Mesh as MMesh
    v = np.array([[p.x, p.y, p.z] for p in terrain.vertices], dtype=np.float32)
    f = np.array(terrain.faces, dtype=np.uint32)
    vol = Manifold(MMesh(vert_properties=v, tri_verts=f)).volume()
    expected = 20000 * 20000 * 15000 - 6000 * 6000 * 900   # block minus pit
    assert abs(vol - expected) / expected < 1e-4


def test_partially_buried_slab_carves_buried_part_on_flat_ground():
    """A ground slab whose top pokes +50 mm above grade must still carve the
    terrain on a FLAT block (regression: the fully-enclosed trigger missed it,
    making excavation terrain-shape-dependent). The manifold boolean removes
    only the buried part."""
    terrain = _terrain(half=10000, depth=15000, top=0)   # AABB top = z=0
    site = Site(name="site"); site.add(_wrap(terrain))
    slab = Box(name="slab", start=Point(x=-4000, y=-4000, z=-150), end=Point(x=4000, y=4000, z=50))
    apply_displacement(_bldg(slab, site=site))
    from manifold3d import Manifold, Mesh as MMesh
    v = np.array([[p.x, p.y, p.z] for p in terrain.vertices], dtype=np.float32)
    f = np.array(terrain.faces, dtype=np.uint32)
    vol = Manifold(MMesh(vert_properties=v, tri_verts=f)).volume()
    # only the 150 mm BELOW grade is removed — the +50 above grade removes nothing
    expected = 20000 * 20000 * 15000 - 8000 * 8000 * 150
    assert abs(vol - expected) / expected < 1e-4
    assert _mesh_is_watertight(terrain)


def test_explicit_terrain_difference_carves_mesh_and_consumes_operand():
    """``.difference()`` on a terrain ``Mesh`` (e.g. a foundation pad excavation)
    is realised as a mesh boolean: the operand lives in the mesh's ``_cuts``, is
    consumed (never in the render tree), and levels the terrain down to the pad
    bottom. Replaces the retired ``_cut_only`` flag path."""
    terrain = _terrain(half=10000, depth=15000, top=500)   # top surface at +500
    site = Site(name="site"); site.add(_wrap(terrain))
    pad = Box(name="pad", start=Point(x=-4000, y=-4000, z=50), end=Point(x=4000, y=4000, z=20050))
    terrain.difference(pad)                           # explicit terrain cut
    assert pad in terrain._cuts                       # consumed operand, not a tree element
    b = _bldg(site=site)                              # pad is NOT added to the storey
    apply_displacement(b)
    assert _mesh_is_watertight(terrain)              # carved, still closed
    from manifold3d import Manifold, Mesh as MMesh
    v = np.array([[p.x, p.y, p.z] for p in terrain.vertices], dtype=np.float32)
    f = np.array(terrain.faces, dtype=np.uint32)
    vol = Manifold(MMesh(vert_properties=v, tri_verts=f)).volume()
    # mm³ (this terrain isn't normalized). Block 20000 x 20000 x 15000 minus the
    # pad footprint 8000 x 8000 levelled from the terrain top (+500) down to the
    # pad bottom (+50) = 8000 * 8000 * 450 removed.
    expected = 20000 * 20000 * 15000 - 8000 * 8000 * 450
    assert abs(vol - expected) / expected < 1e-4


def test_carved_mesh_verts_snapped_to_6_decimals():
    """manifold3d is internally float32, so raw carved verts carry noise
    (-15.2 → -15.199999809265137). They must be snapped to 6 decimals,
    since mesh verts escape both the serializer bit-exact pass and the dimension
    snap."""
    terrain = _terrain(half=16000, depth=15000, top=0)
    site = Site(name="site"); site.add(_wrap(terrain))
    slab = Box(name="slab", start=Point(x=-6000, y=-4500, z=-150), end=Point(x=6000, y=4500, z=50))
    apply_displacement(_bldg(slab, site=site))
    for p in terrain.vertices:
        for c in (p.x, p.y, p.z):
            assert abs(c * 1e6 - round(c * 1e6)) < 1e-3, f"unsnapped vert coord {c!r}"


def test_staircase_nested_footprints():
    terrain = _terrain()
    site = Site(name="site"); site.add(_wrap(terrain))
    slab = Box(name="slab", start=Point(x=-4000, y=-4000, z=-150), end=Point(x=4000, y=4000, z=0))
    strip = Box(name="strip", start=Point(x=-3500, y=-3500, z=-900), end=Point(x=-3000, y=3500, z=0))
    apply_displacement(_bldg(slab, strip, site=site))
    assert _mesh_is_watertight(terrain)             # nested carves stay watertight


# ---------------------------------------------------------------------------
# The overlap trigger + direction rule
# ---------------------------------------------------------------------------


def test_partial_penetration_carves_solid_host():
    """A solid poking OUT of its host still carves it — the boolean removes
    only the shared volume (was: strict enclosure, no carve)."""
    body = Box(name="body", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=300, z=3000))
    deck = Box(name="deck", start=Point(x=1000, y=50, z=1000), end=Point(x=1400, y=600, z=1400))
    b = _bldg(body, deck)
    apply_displacement(b)
    assert len(body._cuts) == 1                     # partial overlap carves
    assert not deck._cuts                           # one-directional


def test_through_penetration_carves_solid_host():
    """A solid passing clean through its host (out both faces) carves it."""
    body = Box(name="body", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=300, z=3000))
    duct = Box(name="duct", start=Point(x=1000, y=-200, z=1000), end=Point(x=1400, y=600, z=1400))
    b = _bldg(body, duct)
    apply_displacement(b)
    assert len(body._cuts) == 1
    assert not duct._cuts


def test_penetrant_carves_every_overlapped_host():
    """A duct through TWO walls carves both — every overlapped host, not just
    the tightest."""
    w1 = Box(name="w1", start=Point(x=0, y=0, z=0), end=Point(x=300, y=4000, z=3000))
    w2 = Box(name="w2", start=Point(x=3000, y=0, z=0), end=Point(x=3300, y=4000, z=3000))
    duct = Box(name="duct", start=Point(x=-200, y=1900, z=1400), end=Point(x=3500, y=2100, z=1600))
    apply_displacement(_bldg(w1, w2, duct))
    assert len(w1._cuts) == 1 and len(w2._cuts) == 1
    assert not duct._cuts


def test_corner_junction_carves_one_directionally():
    """Two wall bodies overlapping at a corner: the LATER-added carves the
    earlier, NEVER both ways — mutual carve would render the symmetric
    difference (a hole at the junction)."""
    a = Box(name="a", start=Point(x=0, y=0, z=0), end=Point(x=6000, y=300, z=3000))
    b_ = Box(name="b", start=Point(x=0, y=0, z=0), end=Point(x=300, y=4000, z=3000))
    apply_displacement(_bldg(a, b_))
    assert len(a._cuts) == 1                        # earlier is carved…
    assert not b_._cuts                             # …later renders the junction


def test_add_order_decides_direction():
    """``.add()`` is directional — the added solid is the carver, the
    already-standing geometry the carvee. Swapping authoring order flips the
    carve direction."""
    t1 = Box(name="t1", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=2000, z=2000))
    t2 = Box(name="t2", start=Point(x=1000, y=0, z=0), end=Point(x=3000, y=2000, z=2000))
    apply_displacement(_bldg(t1, t2))
    assert len(t1._cuts) == 1 and not t2._cuts      # t2 (later) carves t1

    r1 = Box(name="r1", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=2000, z=2000))
    r2 = Box(name="r2", start=Point(x=1000, y=0, z=0), end=Point(x=3000, y=2000, z=2000))
    apply_displacement(_bldg(r2, r1))               # reversed authoring order
    assert len(r2._cuts) == 1 and not r1._cuts      # r1 (later) carves r2


def test_later_add_claims_enclosed_earlier_space():
    """Deliberate consequence of the direction rule: an earlier solid fully
    inside a later one is carved away entirely — the later add claimed that
    space. (The normal flow — host body first, occupants after — never hits
    this; classic displacement below.)"""
    inner = Box(name="inner", start=Point(x=1000, y=1000, z=1000), end=Point(x=2000, y=2000, z=2000))
    outer = Box(name="outer", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=4000, z=4000))
    apply_displacement(_bldg(inner, outer))
    assert len(inner._cuts) == 1                    # outer (later) carves inner away
    assert not outer._cuts
    # classic flow: host authored first, occupant after → occupant carves host
    host = Box(name="host", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=4000, z=4000))
    occ = Box(name="occ", start=Point(x=1000, y=1000, z=1000), end=Point(x=2000, y=2000, z=2000))
    apply_displacement(_bldg(host, occ))
    assert len(host._cuts) == 1 and not occ._cuts


def test_partial_overlap_boolean_removes_exact_shared_volume():
    """End-to-end: the emitted native boolean subtracts exactly the
    intersection — the poke-out excess removes nothing."""
    import ifcopenshell
    import ifcopenshell.geom
    wall = Wall(name="host")
    body = Box(name="body", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=300, z=3000),
               material="Concrete_C25-30")
    wall.add(body)
    deck = Box(name="deck", start=Point(x=1000, y=50, z=1000), end=Point(x=1400, y=600, z=1400),
               material="Concrete_C25-30")
    wall.add(deck)
    res = generate_ifc(_bldg(wall), source_code=None)
    assert res.success, res.error
    assert "IFCBOOLEANRESULT" in res.ifc_content
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".ifc"); os.close(fd)
    try:
        with open(path, "w") as f:
            f.write(res.ifc_content)
        model = ifcopenshell.open(path)
        settings = ifcopenshell.geom.settings()
        walls = model.by_type("IfcWall")
        vol = 0.0
        for w in walls:
            sh = ifcopenshell.geom.create_shape(settings, w)
            v = np.array(sh.geometry.verts).reshape(-1, 3)
            fcs = np.array(sh.geometry.faces).reshape(-1, 3)
            t = v[fcs]
            vol += abs(np.einsum('ij,ij->i', t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)
    finally:
        os.unlink(path)
    # overlap slice is y=50..300 (250 deep) — the y=300..600 poke-out is free.
    # This model bypasses normalize (unit test), so the IFC is in mm → mm³.
    expected = (4000 * 300 * 3000) - (400 * 250 * 400)
    assert abs(vol - expected) / expected < 1e-3


def test_above_grade_solid_does_not_carve_terrain():
    terrain = _terrain(top=0, depth=15000)
    site = Site(name="site"); site.add(_wrap(terrain))
    wall = Box(name="wall", start=Point(x=-3000, y=-3000, z=0), end=Point(x=3000, y=3000, z=3000))
    b = _bldg(wall, site=site)
    before = [(p.x, p.y, p.z) for p in terrain.vertices]
    apply_displacement(b)
    after = [(p.x, p.y, p.z) for p in terrain.vertices]
    assert after == before                          # wall sits ON grade → not enclosed → no carve


def test_abutting_solids_do_not_carve():
    a = Box(name="a", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=2000, z=2000))
    b_ = Box(name="b", start=Point(x=2000, y=0, z=0), end=Point(x=4000, y=2000, z=2000))
    b = _bldg(a, b_)
    apply_displacement(b)
    assert not a._cuts and not b_._cuts             # flush faces → zero-depth overlap → no carve


def test_placed_subtree_is_exempt():
    """Geometry under ``placement=`` does not participate — in EITHER role.
    A placed container's children are authored in pre-placement coords; judging
    them where they were authored would carve against whatever stands near the
    origin (spurious). Placed overlap is legal and stays overlap."""
    from lite_step.models import Transform
    standing = Box(name="standing", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=4000, z=4000))
    moved = Wall(name="moved", placement=Transform(origin=Point(x=50000, y=0, z=0)))
    # child authored around the origin — overlaps `standing` in authoring
    # coords but stands 50 m away after placement
    moved.add(Box(name="body", start=Point(x=1000, y=1000, z=1000), end=Point(x=2000, y=2000, z=2000)))
    b = _bldg(standing, moved)
    apply_displacement(b)
    assert not standing._cuts                       # no spurious carve at authoring coords
    assert not moved._elements[0]._cuts             # and the placed child isn't carved either


def test_no_occupants_no_change():
    wall = Wall(name="w")
    wall.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=5000, y=200, z=3000)))
    b = _bldg(wall)
    apply_displacement(b)
    body = wall._elements[0]
    assert not body._cuts                           # nothing overlapping → no carve


# ---------------------------------------------------------------------------
# Backend parity + opening-identity regression
# ---------------------------------------------------------------------------


def test_a_footing_through_terrain_emits_the_carved_mesh():
    terrain = _terrain()
    site = Site(name="site"); site.add(_wrap(terrain))
    foot = Box(name="strip", start=Point(x=-3000, y=-3000, z=-900), end=Point(x=3000, y=3000, z=0))
    res = generate_ifc(_bldg(foot, site=site), source_code=None)
    assert res.success, res.error
    assert "IFCTRIANGULATEDFACESET" in res.ifc_content


def test_window_still_produces_opening_element():
    wall = Wall(name="north")
    wall.add(Box(start=Point(x=-3000, y=-100, z=0), end=Point(x=3000, y=100, z=2700),
                 material="Brick_Red_DK"))
    wall.anchor(Window(width=1200, height=1400, name="w0"), along=1400, up=900)
    res = generate_ifc(_bldg(wall), source_code=None)
    assert res.success, res.error
    assert "IFCOPENINGELEMENT" in res.ifc_content   # opening identity path untouched


# ── Profile members (Material(profile_mm=)) participate in displacement ──────
# Two gaps this pins (both reproduced on main before the fix):
#  1. `padded_aabb` padded a Sweep only from `elem.profile`; a member whose
#     section is `Material(profile_mm=(w,h))` has `profile=None`, so the pad was
#     0 and the AABB collapsed to a zero-width line — the overlap trigger could
#     never fire, so the member never participated in displacement.
#  2. A path-form Sweep was rejected as a boolean operand ("must be box-mode
#     Solids…, Revolve, Pipe, or Bar"), so a member could never be a carver.


def test_material_profile_sweep_aabb_is_padded_not_degenerate():
    # A rafter whose section is Material(profile_mm=) — no explicit contour.
    rafter = Sweep(
        name="rafter",
        path=[Point(x=0, y=0, z=3000), Point(x=4000, y=0, z=5000)],
        material=Material(key="Timber_C24", profile_mm=(45, 195)),
    )
    (x0, y0, z0), (x1, y1, z1) = padded_aabb(rafter)
    # in-plane extents are padded by the section half-diagonal (was 0 → line)
    assert (y1 - y0) > 0.05, f"AABB collapsed on y: {(y1 - y0)}"
    assert (x1 - x0) > 4.0   # spans the path


def test_material_profile_sweep_carves_as_operand():
    # Box host added FIRST, material-profile Sweep carver added SECOND and
    # overlapping → the Sweep auto-carves the host (later carves earlier). Needs
    # BOTH fixes: the padded AABB (trigger) + the Sweep accepted as an operand.
    plate = Beam(name="wallplate")
    plate.add(Box(name="body", start=Point(x=-200, y=-200, z=2900),
                  end=Point(x=4200, y=200, z=3100), material="Concrete_C30-37"))
    rafter = Beam(name="rafter")
    rafter.add(Sweep(name="rbody",
                     path=[Point(x=0, y=0, z=2950), Point(x=4000, y=0, z=5000)],
                     material=Material(key="Timber_C24", profile_mm=(45, 195))))
    res = generate_ifc(_bldg(plate, rafter), source_code=None)
    assert res.success, res.error
    # the Sweep member became a DIFFERENCE operand on the plate
    assert res.ifc_content.count("IFCBOOLEANRESULT(") >= 1


def test_non_overlapping_material_profile_sweep_is_a_noop():
    # A padded AABB is a BOUND, not exact — but a member that doesn't overlap
    # must still not carve (no spurious boolean).
    plate = Beam(name="plate")
    plate.add(Box(name="body", start=Point(x=0, y=0, z=0),
                  end=Point(x=1000, y=200, z=200), material="Concrete_C30-37"))
    rafter = Beam(name="rafter")
    rafter.add(Sweep(name="rbody",
                     path=[Point(x=0, y=0, z=5000), Point(x=4000, y=0, z=7000)],
                     material=Material(key="Timber_C24", profile_mm=(45, 195))))
    res = generate_ifc(_bldg(plate, rafter), source_code=None)
    assert res.success, res.error
    assert res.ifc_content.count("IFCBOOLEANRESULT(") == 0


# ── Tight AABBs: the phantom-carve and the silent-non-carve ──────────────────
# Both are pure trigger bugs — the geometry is fine, the BOUND is wrong — and
# both were found through boolean depth: an over-inflated AABB manufactures
# carves against elements the solid never touches, and every phantom carve
# pushes the host's CSG chain toward the depth at which web viewers silently
# render the UNCARVED base solid.


def test_wide_profile_sweep_does_not_carve_what_it_merely_encloses():
    # A roof sweep whose profile runs eave-to-ridge reaches ~6.3 m ACROSS the
    # sweep and 2.8 m up — but only a few hundred mm along it. Padding the path
    # AABB by the profile's RADIAL max (6.9 m) in every axis put the whole
    # building inside the roof's box, so every element under the roof became a
    # spurious carve operand. The pad is directional now: taken in the real
    # per-segment section frame.
    from lite_step.models import Point2D

    interior = Box(name="partition", start=Point(x=-1000, y=-1000, z=0),
                   end=Point(x=1000, y=1000, z=2500))
    roof = Sweep(
        name="roofplane",
        profile=[Point2D(x=-6300, y=0), Point2D(x=6300, y=0),
                 Point2D(x=0, y=2800)],
        path=[Point(x=0, y=-5000, z=3000), Point(x=0, y=5000, z=3000)],
    )
    (_x0, _y0, z0), (_x1, _y1, z1) = padded_aabb(roof)
    # the sweep sits at z=3000 and rises 2800 — it must NOT reach down to z=0
    assert z0 > 2000, f"sweep AABB still floods downward: z0={z0}"
    assert z1 < 6500, f"sweep AABB still floods upward: z1={z1}"

    b = _bldg(interior, roof)          # roof authored LAST → it would be the carver
    apply_displacement(b)
    assert interior._cuts == [], "roof carved an element it only encloses by AABB"


def test_wide_profile_sweep_still_carves_what_it_really_overlaps():
    # The tightened pad must not become an UNDER-approximation: a solid the
    # sweep genuinely passes through still carves.
    from lite_step.models import Point2D

    post = Box(name="post", start=Point(x=-100, y=-100, z=2000),
               end=Point(x=100, y=100, z=4000))
    roof = Sweep(
        name="roofplane",
        profile=[Point2D(x=-6300, y=0), Point2D(x=6300, y=0),
                 Point2D(x=0, y=2800)],
        path=[Point(x=0, y=-5000, z=3000), Point(x=0, y=5000, z=3000)],
    )
    b = _bldg(post, roof)
    apply_displacement(b)
    assert len(post._cuts) == 1


def test_contour_extrude_is_padded_by_its_thickness():
    # A contour Extrude's AABB came from the contour POINTS only, so it was a
    # zero-width slab in the extrusion direction — and _overlaps demands
    # POSITIVE overlap on every axis, so it could NEVER register overlap along
    # that axis. Silent non-carve.
    slab = Extrude(name="insulation", thickness=200, contour=[
        Point(x=0, y=0, z=3000), Point(x=4000, y=0, z=3000),
        Point(x=4000, y=2000, z=3000), Point(x=0, y=2000, z=3000)])
    (_x0, _y0, z0), (_x1, _y1, z1) = padded_aabb(slab)
    assert (z1 - z0) >= 200, f"contour Extrude AABB is still flat: {(z1 - z0)}"


def test_contour_extrude_is_carved_by_an_overlapping_rafter():
    # The real failure: an insulation slab authored as a contour Extrude sat in
    # the rafter zone and was silently never carved by the rafters.
    slab = Extrude(name="insulation", thickness=200, contour=[
        Point(x=0, y=0, z=3000), Point(x=4000, y=0, z=3000),
        Point(x=4000, y=2000, z=3000), Point(x=0, y=2000, z=3000)])
    rafter = Box(name="rafter", start=Point(x=1000, y=500, z=2950),
                 end=Point(x=1050, y=1500, z=3050))
    b = _bldg(slab, rafter)             # rafter authored LAST → it is the carver
    apply_displacement(b)
    assert len(slab._cuts) == 1, "contour Extrude still not carved by the rafter"
    res = generate_ifc(b, source_code=None)
    assert res.success, res.error
    assert res.ifc_content.count("IFCBOOLEANRESULT(") >= 1


def _corbel_wall(anchored: bool):
    """A 4000x300x3000 wall body plus a 200x100x60 corbel at the SAME world
    position either way — anchored (``along=1000, up=2000``, which bakes to
    x 1000..1200, y 200..300, z 2000..2060) or ``.add()``ed in those exact
    world coordinates. Returns ``(project, wall, body, corbel)``."""
    wall = Wall(name="w")
    body = Box(name="body", start=Point(x=0, y=0, z=0),
               end=Point(x=4000, y=300, z=3000))
    wall.add(body)
    if anchored:
        corbel = Box(name="knaegt", start=Point(x=0, y=0, z=0),
                     end=Point(x=200, y=100, z=60))
        wall.anchor(corbel, along=1000, up=2000)
    else:
        corbel = Box(name="knaegt", start=Point(x=1000, y=200, z=2000),
                     end=Point(x=1200, y=300, z=2060))
        wall.add(corbel)
    return _bldg(wall), wall, body, corbel


def _bake(project):
    """The pre-displacement half of ``generate_ifc``'s pipeline."""
    from lite_step.compiler.frames import resolve_child_anchors, stamp_frames
    from lite_step.compiler.naming import stamp_canonical_names

    stamp_canonical_names(project)
    stamp_frames(project)
    resolve_child_anchors(project)
    stamp_frames(project)
    return project


def test_anchored_child_is_not_an_occupant_after_the_bake():
    from lite_step.compiler.displacement import _collect

    project, _wall, _body, corbel = _corbel_wall(anchored=True)
    _bake(project)
    # Guard the fixture: an unbaked corbel proves nothing, and a corbel that
    # does not overlap the body would pass a no-op implementation.
    assert corbel._anchor_resolved is True, "fixture is not baked — test is vacuous"
    assert corbel._anchor_spec is None, (
        "the bake is supposed to CONSUME the spec — if it stops doing so, "
        "_anchor_resolved may not be the durable marker any more")
    assert (corbel.start.y, corbel.end.y) == (200.0, 300.0), (
        f"corbel does not overlap the wall body in y: {corbel.start}..{corbel.end}")

    names = [getattr(e, "name", None) for e, _aabb in _collect(project)[0]]
    assert names == ["body"], (
        f"an .anchor()ed child must not be collected as an occupant, got {names}")


def test_anchored_child_does_not_carve_its_host():
    project, _wall, body, _corbel = _corbel_wall(anchored=True)
    _bake(project)
    apply_displacement(project)
    assert body._cuts == [], (
        f"the wall body was carved by its anchored corbel: {len(body._cuts)} cut(s)")


def test_added_child_in_the_same_position_still_carves():
    # The rule that SURVIVES: world-coordinate containment via .add() is the
    # author saying "this occupies that space", and it still carves. Same
    # geometry, same position, different verb.
    project, _wall, body, _corbel = _corbel_wall(anchored=False)
    _bake(project)
    apply_displacement(project)
    assert len(body._cuts) == 1, (
        f"an .add()ed corbel must still carve its host, got {len(body._cuts)} cut(s)")


def test_compiled_model_has_no_boolean_where_the_anchored_child_overlaps():
    # End-to-end through generate_ifc (which runs the bake itself): the
    # anchored model emits exactly the booleans of a bare wall, the .add()ed
    # twin emits one more.
    bare_wall = Wall(name="w")
    bare_wall.add(Box(name="body", start=Point(x=0, y=0, z=0),
                      end=Point(x=4000, y=300, z=3000)))
    res_bare = generate_ifc(_bldg(bare_wall), source_code=None)
    assert res_bare.success, res_bare.error
    baseline = res_bare.ifc_content.count("IFCBOOLEANRESULT(")

    res_anchored = generate_ifc(_corbel_wall(anchored=True)[0], source_code=None)
    assert res_anchored.success, res_anchored.error
    assert res_anchored.ifc_content.count("IFCBOOLEANRESULT(") == baseline, (
        "anchoring a corbel to a wall manufactured a boolean in the IFC")

    res_added = generate_ifc(_corbel_wall(anchored=False)[0], source_code=None)
    assert res_added.success, res_added.error
    assert res_added.ifc_content.count("IFCBOOLEANRESULT(") == baseline + 1, (
        "the .add()ed twin should still emit its carve — if it does not, the "
        "anchored assertion above is vacuous")


# ---------------------------------------------------------------------------
# Carve assertability (roadmap §1.9): the report and proj.assert_carved
# ---------------------------------------------------------------------------
# Displacement direction is authoring order — the later-added solid carves what
# already stands — and that is a FEATURE: a model file's assembly tail reads as
# a construction narrative. What it was not is observable. Reordering two
# `proj.add()` lines silently changed the geometry and nothing in a diff, a
# type or a test could see it.


def _pair_model(rafter_first: bool):
    """A rafter and a wall that overlap. Whichever is added LAST carves.

    The rafter's body is NAMED and the wall's is not, on purpose: that is what
    real models look like, and it exercises both label paths at once.
    """
    rafter = Beam(name="rafter")
    rafter.add(Box(name="rbody", start=Point(x=0, y=-100, z=2400),
                   end=Point(x=4000, y=100, z=2600)))
    wall = Wall(name="eave")
    wall.add(Box(start=Point(x=1000, y=-200, z=0),
                 end=Point(x=1200, y=200, z=2500)))
    order = (rafter, wall) if rafter_first else (wall, rafter)
    return _bldg(*order), rafter, wall


def _carved(project):
    """Displacement, with the naming pass ``generate_ifc`` runs first — the
    report is in CANONICAL names, so it needs them stamped."""
    from lite_step.compiler.displacement import carve_report_lines
    from lite_step.compiler.naming import stamp_canonical_names

    stamp_canonical_names(project)
    apply_displacement(project)
    return carve_report_lines(project)


def test_the_compile_reports_who_carved_whom():
    project, _rafter, _wall = _pair_model(rafter_first=True)
    assert "carve: box:rbody:beam:rafter <- wall:eave" in _carved(project)


def test_the_report_names_an_anonymous_body_by_its_named_owner():
    """A wall body is anonymous and the author cannot name it, so the report —
    and therefore the assertion — speaks of the wall."""
    project, _rafter, _wall = _pair_model(rafter_first=False)
    lines = _carved(project)
    assert "carve: wall:eave <- box:rbody:beam:rafter" in lines
    assert not any("anonymous" in line for line in lines)


def test_the_report_follows_the_authoring_order_it_describes():
    """Reverse the two adds and the arrow reverses. If it does not, the report
    is decoration rather than a record of what happened."""
    first, _r, _w = _pair_model(rafter_first=True)
    second, _r2, _w2 = _pair_model(rafter_first=False)
    assert "carve: box:rbody:beam:rafter <- wall:eave" in _carved(first)
    assert "carve: wall:eave <- box:rbody:beam:rafter" in _carved(second)


def test_a_satisfied_assertion_compiles():
    """…and it is written with the names the AUTHOR wrote — `beam:rafter`,
    resolved onto the body that was physically carved by the same
    segment-aligned suffix match `Anchor(host=)` uses."""
    project, _rafter, _wall = _pair_model(rafter_first=True)
    project.assert_carved("beam:rafter", by="wall:eave")
    res = generate_ifc(project, source_code=None)
    assert res.success, res.error


def test_reordering_the_two_adds_breaks_the_assertion():
    """THE falsification. The same assertion and the same two elements; the
    only difference is which `proj.add()` line comes first."""
    kept, _r, _w = _pair_model(rafter_first=True)
    kept.assert_carved("beam:rafter", by="wall:eave")
    assert generate_ifc(kept, source_code=None).success

    reordered, _r2, _w2 = _pair_model(rafter_first=False)
    reordered.assert_carved("beam:rafter", by="wall:eave")
    res = generate_ifc(reordered, source_code=None)
    assert not res.success, (
        "swapping the two adds must fail the named assertion — otherwise "
        "nothing pins the joinery and §1.9's order-semantics stay invisible"
    )


def _three_solids():
    """A, then B (carves A), then C (carves B only — it never reaches A)."""
    a = Beam(name="a")
    a.add(Box(name="abody", start=Point(x=0, y=0, z=0),
              end=Point(x=1000, y=300, z=300)))
    b = Beam(name="b")
    b.add(Box(name="bbody", start=Point(x=900, y=0, z=0),
              end=Point(x=2000, y=300, z=300)))
    c = Beam(name="c")
    c.add(Box(name="cbody", start=Point(x=1900, y=0, z=0),
              end=Point(x=3000, y=300, z=300)))
    return _bldg(a, b, c)


def test_an_unsatisfied_assertion_names_both_elements_and_says_why():
    project = _three_solids()
    project.assert_carved("beam:a", by="beam:c")     # c never reaches a
    res = generate_ifc(project, source_code=None)
    assert not res.success
    assert "beam:a" in res.error and "beam:c" in res.error, \
        "the message must name BOTH elements"
    assert "beam:b" in res.error, "…and what actually carved the target"
    assert "authoring order" in res.error, (
        "the message must say WHY, or it sends the author back to rendering "
        "the model — which is what the assertion replaces")


def test_an_assertion_naming_nothing_is_an_error_not_a_pass():
    """A vacuous assertion is worse than none: it reads as coverage."""
    project, _rafter, _wall = _pair_model(rafter_first=True)
    project.assert_carved("wall:nosuchwall", by="beam:rafter")
    res = generate_ifc(project, source_code=None)
    assert not res.success
    assert "nothing carved" in res.error


def test_the_assertion_is_about_the_PAIR_not_the_MECHANISM():
    """§6.5's version-skew contract, pinned.

    A later workstream moves exemption boundaries between carve mechanisms, so
    an assertion must be satisfied by an AUTHORED `.difference()` exactly as by
    an inferred carve. If this ever starts failing, every assertion written
    against today's inference breaks the day the boundary moves.
    """
    host = Beam(name="girder")
    body = Box(name="gbody", start=Point(x=0, y=0, z=0),
               end=Point(x=4000, y=300, z=400))
    tool = Box(name="notch", start=Point(x=1000, y=-50, z=300),
               end=Point(x=1200, y=350, z=500))
    body.difference(tool)                    # authored, never inferred
    host.add(body)
    project = _bldg(host)
    project.assert_carved("beam:girder", by="box:notch:box:gbody:beam:girder")
    res = generate_ifc(project, source_code=None)
    assert res.success, res.error


# ── the carve report's DEFAULT verbosity (WS-A commit 5) ──────────────────────
#
# WS-0 shipped the report uncapped and flagged the default as a decision to
# revisit. Measured on the corpus: 16 of 20 models emit <=2 lines, but the four
# villa-chain models emit 96-265 (~12 KB of stderr per compile) — on the models
# the author builds most, on every recompile, into a bounded agent context.

def _report_stderr(project, monkeypatch, capsys, setting=None):
    from lite_step.compiler.naming import stamp_canonical_names
    from lite_step.ifc.generator import _report_carve_pairs

    if setting is None:
        monkeypatch.delenv("LITESTEP_CARVE_REPORT", raising=False)
    else:
        monkeypatch.setenv("LITESTEP_CARVE_REPORT", setting)
    stamp_canonical_names(project)
    apply_displacement(project)
    capsys.readouterr()
    _report_carve_pairs(project)
    return capsys.readouterr().err


def test_the_report_defaults_to_one_summary_line(monkeypatch, capsys):
    project, _r, _w = _pair_model(rafter_first=True)
    err = _report_stderr(project, monkeypatch, capsys)
    assert err.splitlines()[0].startswith("carves: 1 pairs")
    assert "carve: box:rbody" not in err          # the DETAIL is not printed
    assert "LITESTEP_CARVE_REPORT=all" in err     # …and it says how to get it


def test_the_count_is_always_there(monkeypatch, capsys):
    """What makes the summary an acceptable default: §1.9's visibility survives
    because a reordering that breaks the joinery changes the NUMBER.

    Falsification: if both orderings reported the same thing, the summary would
    be decoration and the cap would have cost the license for order-semantics.
    """
    project, _r, _w = _pair_model(rafter_first=True)
    assert "carves: 1 pairs" in _report_stderr(project, monkeypatch, capsys)
    empty = _bldg(Wall(name="lonely").add(
        Box(start=Point(x=0, y=0, z=0), end=Point(x=100, y=100, z=100))))
    assert "carves: 0 pairs" in _report_stderr(empty, monkeypatch, capsys)


def test_all_restores_the_full_list(monkeypatch, capsys):
    project, _r, _w = _pair_model(rafter_first=True)
    err = _report_stderr(project, monkeypatch, capsys, setting="all")
    assert "carve: box:rbody:beam:rafter <- wall:eave" in err
    assert "carves: 1 pairs" not in err


def test_zero_is_still_the_kill_switch(monkeypatch, capsys):
    """Byte-comparison runs want the two compiles' stderr to match exactly."""
    project, _r, _w = _pair_model(rafter_first=True)
    assert _report_stderr(project, monkeypatch, capsys, setting="0") == ""
