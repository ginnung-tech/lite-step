"""Carve annihilation must be LOUD.

The `.add()` direction rule is "later carves earlier", and its documented
consequence is that an earlier solid wholly inside a later one is carved to
nothing. That is deliberate and stays. The hazard is the silence: author
the bars before the beam body instead of after and the reinforcement vanishes
with no error, no warning, and geometry that still compiles.

Two hosts, two mechanisms, one doctrine:

* **solid host** — the direction rule above, detected on the INPUT side by AABB
  containment (the solid path's booleans are declarative and never evaluated,
  so there is no volume to read).
* **mesh host** (terrain) — the manifold3d boolean IS evaluated, so the check is
  on the OUTPUT side and reads true volume. An unplaced solid over a terrain
  Mesh could subtract the whole ground away — `IfcTriangulatedFaceSet` from 198
  coordinates to 0 — and the model still compiled clean.

That mesh-host annihilation is ROOT-CAUSED: it is never a subtraction, it is
`manifold3d`'s error status propagating out of a seam-duplicated `Sweep`
tessellation. The fixture is kept as the regression pin, inverted — see
`test_bridge_deck_does_not_annihilate_the_terrain`. The WARNING stays pinned on
a cutter that genuinely swallows the terrain, so loudness is still tested
without keeping the bug. Mechanism-level coverage is in
`test_tessellation_weld.py`.
"""
import logging
import pytest
from lite_step.models import (Project, Beam, Box, Point, Point2D, Bar, Mesh,
                              Site, Element, Sweep, Material)
from lite_step.compiler.displacement import apply_displacement
from lite_step.compiler.executor import normalize_project_to_meters


def _beam_with_bars(bars_first: bool):
    proj = Project(name="t")
    beam = Beam(name="a1")
    body = Box(name="body", start=Point(x=-150, y=-2000, z=0),
               end=Point(x=150, y=2000, z=600))
    bar = Bar(name="b1", path=[Point(x=0, y=-1800, z=60), Point(x=0, y=1800, z=60)],
              diameter=16)
    if bars_first:
        beam.add(bar)     # WRONG ORDER — the body will claim this space
        beam.add(body)
    else:
        beam.add(body)    # right order — the bar carves the body and survives
        beam.add(bar)
    proj.add(beam)
    return proj


def test_annihilated_element_warns(caplog):
    with caplog.at_level(logging.WARNING):
        apply_displacement(_beam_with_bars(bars_first=True))
    msgs = [r.getMessage() for r in caplog.records]
    hits = [m for m in msgs if "carved to nothing" in m]
    assert hits, (
        "a solid carved out of existence must say so — this is the "
        "bars-before-body footgun, and silence is the failure.\n"
        f"got: {msgs}")
    assert "b1" in hits[0], f"the warning must name the victim: {hits[0]}"
    assert "body" in hits[0], f"the warning must name what claimed it: {hits[0]}"


def test_correct_order_is_silent(caplog):
    """Falsification: the check must NOT fire on correct code, or it is noise
    and gets muted — at which point the loud path stops being read."""
    with caplog.at_level(logging.WARNING):
        apply_displacement(_beam_with_bars(bars_first=False))
    msgs = [r.getMessage() for r in caplog.records]
    assert not [m for m in msgs if "carved to nothing" in m], (
        f"correctly-ordered rebar must not warn: {msgs}")


# ── mesh host (terrain) ───────────────────────────────────────────────────────

#: The real bridge terrain that reproduces the annihilation, verbatim. Smaller
#: synthetic patches do NOT reproduce it: `_mesh_aabb` is the true vertex hull,
#: and a large irregular open shell like this one is far easier for `_overlaps`
#: to satisfy than a small flat grid. Keep the numbers — they are the repro.
_BRIDGE_HEIGHTMAP = [
    [82172, 94504, 108137, 110219, 106742, 94887, 77043, 82242, 69110],
    [78035, 86852, 105770, 100043, 94309, 83516, 69305, 61996, 55664],
    [76188, 89078, 104321, 94403, 88215, 67614, 39918, 32594, 31852],
    [76383, 83235, 80856, 77387, 62414, 16004, 6598, 12, 0],
    [84950, 63418, 31285, 34078, 16543, 1250, 17137, 40446, 38047],
    [72149, 53188, 28418, 9180, 1289, 22200, 41200, 61379, 74778],
    [83672, 51211, 26235, 2965, 8778, 34008, 55387, 81082, 91180],
    [71434, 29301, 1575, 15492, 36930, 51075, 78926, 96856, 102043],
    [35989, 5645, 4614, 49391, 65801, 79403, 84739, 97602, 106336],
    [12707, 8293, 30461, 68250, 79571, 85684, 94797, 101723, 104481],
    [3438, 33895, 61825, 83719, 91961, 99121, 101871, 102860, 109692],
]


def _bridge_terrain():
    """A Project whose Site carries the bridge terrain inside the canonical
    wrapper. The wrapper is not decoration: mesh carve-host candidacy is
    SEMANTIC (`taxonomy.is_terrain_wrapper`), so a bare Mesh under Site is
    inert and would silently prove nothing."""
    proj = Project(name="t")
    site = Site(name="site")
    wrapper = Element(ifc_class="IfcGeographicElement",
                      predefined_type="TERRAIN", name="terrain")
    ground = Mesh(name="ground", heightmap=_BRIDGE_HEIGHTMAP, depth=20000,
                  corner_min=Point2D(x=-114616, y=-133115),
                  corner_max=Point2D(x=97718, y=132303),
                  source_type="synthetic")
    wrapper.add(ground)
    site.add(wrapper)
    proj.add(site)
    return proj, ground


def _deck_ribbon():
    """The unplaced deck-axis ribbon that eats the ground. UNPLACED is
    load-bearing: `_collect._walk` returns on any element carrying
    `placement=`, skipping its whole subtree, so a placed cutter never enters
    the inferred-carve pass and the terrain survives."""
    return Sweep(name="deck_axis_preview",
                 profile=[Point2D(x=-3500, y=-250), Point2D(x=3500, y=-250),
                          Point2D(x=3500, y=250), Point2D(x=-3500, y=250)],
                 path=[Point(x=-73005, y=-97546, z=107770),
                       Point(x=0, y=0, z=107213),
                       Point(x=73006, y=97545, z=106657)],
                 material=Material(key="Concrete_C30-37"))


def test_bridge_deck_does_not_annihilate_the_terrain(caplog):
    """ROOT CAUSE — the fixture that can take the terrain from 198 verts to 0
    must leave it standing, untouched.

    This test's ancestor asserted `len(mesh.vertices) == 0` and told whoever
    fixed the boolean to retire it deliberately. This is that retirement,
    inverted into the regression pin.

    The annihilation was never a subtraction. OCC evaluates each face of the
    swept solid independently, so the ribbon's segment seam came back as two
    vertex entries 1.4e-14 m apart — one point geometrically, two indices
    topologically, 8 edges with multiplicity 1. `manifold3d` answers
    `Error.NotManifold` to that WITHOUT raising (so `_carve_mesh_host`'s
    `except` around the constructor could never fire) and then propagates the
    error status through every boolean: `host - operand` empty, and decisively
    `host + operand` — a UNION — empty too, which no geometry can explain.

    Welded (`ifc.geometry.weld_coincident_vertices`, applied at the kernel
    boundary in `tessellate_solid_to_mesh`) the operand is a valid 852.9 m³
    manifold and the true answer appears: the carve removes **exactly 0 m³**,
    because a bridge deck at z≈107 m never intersected the ground it flies
    over. The AABB overlap that triggered the carve was a BOUND overlap only.
    """
    pytest.importorskip("manifold3d")
    proj, ground = _bridge_terrain()
    proj.add(_deck_ribbon())
    out = normalize_project_to_meters(proj)
    mesh = out.sites[0]._elements[0]._elements[0]
    assert len(mesh.vertices) == 198, "fixture drifted — this is the repro shape"
    faces_before = list(mesh.faces)

    with caplog.at_level(logging.WARNING):
        apply_displacement(out)

    assert len(mesh.vertices) == 198, (
        "the terrain was annihilated again — an operand manifold3d refuses "
        "erases the host instead of carving it")
    assert list(mesh.faces) == faces_before, (
        "the deck removes 0 m3 from this terrain, so the mesh must come out "
        "byte-identical — any rewrite means a phantom subtraction happened")
    msgs = [r.getMessage() for r in caplog.records]
    assert not [m for m in msgs if "carved to nothing" in m], (
        f"nothing was annihilated, so nothing may claim it was: {msgs}")


def test_operand_manifold3d_refuses_is_skipped_loudly(caplog):
    """The guard that makes the above impossible to regress SILENTLY.

    A non-closed operand is not a wrong carve, it is an erasure: manifold3d's
    error status propagates and the host comes back empty. So the CARVE must
    be refused — that part is unchanged and is what the vertex assertion below
    pins.

    **Refusing the carve does not refuse the whole compile.** The original reasoning ("an operand manifold3d refuses
    erases the host") argues for skipping the operand, not for killing the
    model — and the two came apart the moment a real build hit it: ONE bad
    chord out of ~250 members took an entire bridge down. A host that survives
    un-carved is strictly better than no model, the degradation is visible (a
    notch that should be there is not), and the warning names the operand, so
    this is not the silent-wrongness case the loud-failure rule exists for.

    Uses an open shell (a cube missing one face) as a `Mesh` operand, which
    reaches the same code path as a bad tessellation without depending on OCC.
    """
    pytest.importorskip("manifold3d")

    proj, ground = _bridge_terrain()
    # Unit cube spanning the terrain centre, with the top face LEFT OFF.
    v = [Point(x=x, y=y, z=z)
         for z in (-30000, 130000) for x, y in ((-9000, -9000), (9000, -9000),
                                               (9000, 9000), (-9000, 9000))]
    open_shell = Mesh(name="open_cutter", vertices=v, source_type="synthetic",
                      faces=[(0, 2, 1), (0, 3, 2),                  # bottom
                             (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                             (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])
    ground.difference(open_shell)
    out = normalize_project_to_meters(proj)
    mesh = out.sites[0]._elements[0]._elements[0]

    with caplog.at_level(logging.WARNING):
        apply_displacement(out)              # must NOT raise

    msgs = [r.getMessage() for r in caplog.records]
    rejected = [m for m in msgs if "rejected the operand" in m]
    assert rejected, f"the refusal must still be LOUD: {msgs}"
    msg = rejected[0]
    assert "open_cutter" in msg, f"the warning must name the operand: {msg}"
    # The diagnosis must say WHICH failure this is. This fixture is a genuine
    # hole (one face left off), so it must not be reported as non-manifold —
    # conflating the two is what misdirected in the first place.
    assert "hole edges" in msg, f"the diagnosis must be in the warning: {msg}"
    assert "The surface is OPEN" in msg, (
        f"an open shell must be diagnosed as open, not as non-manifold: {msg}")

    assert len(mesh.vertices) == 198, (
        "the host must be left ALONE when its operand is refused — the whole "
        "point is that an invalid operand does not erase the terrain")


def test_real_annihilation_still_warns(caplog):
    """The warning stays under test — on a GENUINE annihilation.

    With the bridge fixture not annihilating, the loudness needs a case
    that really does remove everything: a cutter that swallows the terrain's
    whole hull. This keeps the warning honest without keeping the bug.
    """
    pytest.importorskip("manifold3d")
    proj, ground = _bridge_terrain()
    ground.difference(Box(name="swallow",
                          start=Point(x=-200000, y=-200000, z=-50000),
                          end=Point(x=200000, y=200000, z=200000)))
    out = normalize_project_to_meters(proj)
    mesh = out.sites[0]._elements[0]._elements[0]

    with caplog.at_level(logging.WARNING):
        apply_displacement(out)

    assert len(mesh.vertices) == 0, "bad fixture — this must really annihilate"
    msgs = [r.getMessage() for r in caplog.records]
    hits = [m for m in msgs if "carved to nothing" in m]
    assert hits, (
        "a terrain mesh subtracted out of existence must say so — an "
        f"info-level '→ 0 verts' is not enough.\ngot: {msgs}")
    assert "ground" in hits[0], f"the warning must name the host: {hits[0]}"
    assert "swallow" in hits[0], (
        f"the warning must name what consumed it: {hits[0]}")


def test_partial_mesh_carve_is_silent(caplog):
    """Falsification, the noise side: a real excavation removes a pit, not the
    ground. A carve that leaves the terrain standing must stay quiet, or the
    warning gets muted and stops being read."""
    pytest.importorskip("manifold3d")
    proj, ground = _bridge_terrain()
    # A ~10m x 10m footing pit in a ~212m x 265m terrain.
    ground.difference(Box(start=Point(x=-5000, y=-5000, z=-20000),
                          end=Point(x=5000, y=5000, z=120000)))
    out = normalize_project_to_meters(proj)
    mesh = out.sites[0]._elements[0]._elements[0]

    with caplog.at_level(logging.WARNING):
        apply_displacement(out)

    assert len(mesh.vertices) > 0, "the pit ate the whole terrain — bad fixture"
    msgs = [r.getMessage() for r in caplog.records]
    assert not [m for m in msgs if "carved to nothing" in m], (
        f"a partial excavation must not warn: {msgs}")


def _pier_through_the_ground():
    """A solid that genuinely intersects the terrain (unlike `_deck_ribbon`,
    which flies over it). Spans the full height band of the heightmap, so the
    overlap is real volume, not just an AABB touch."""
    return Box(name="pier",
               start=Point(x=-20000, y=-20000, z=0),
               end=Point(x=20000, y=20000, z=120000),
               material=Material(key="Concrete_C30-37"))


@pytest.mark.parametrize("cutter_first", [True, False])
def test_terrain_carve_ignores_tree_order(cutter_first):
    """A MESH host is carved wherever the cutter sits in the tree.

    The solid-vs-solid direction rule ("later in the tree carves earlier") is
    what `apply_displacement`'s `solid_index[id(host)] < j` implements — and
    that branch is reached ONLY for solid hosts. The `tx.is_mesh(host)` branch
    above it appends the occupant unconditionally, with no index comparison.

    So terrain has no ordering defence, and this test pins both directions:
    move the cutter before or after the Site and the ground is carved either
    way. The consequence for authors is the one the reference now states —
    reordering never saves the terrain, only `placement=` / `.no_carve()`
    does — and it is why the "un-exempted solid over the terrain" trap cannot
    be worked around by shuffling `.add()` lines.
    """
    pytest.importorskip("manifold3d")
    proj, _ = _bridge_terrain()
    pier = _pier_through_the_ground()
    if cutter_first:
        # Rebuild so the cutter precedes the Site in the tree walk.
        proj2 = Project(name="t")
        proj2.add(pier)
        for s in proj.sites:
            proj2.add(s)
        proj = proj2
    else:
        proj.add(pier)

    out = normalize_project_to_meters(proj)
    mesh = out.sites[0]._elements[0]._elements[0]
    faces_before = list(mesh.faces)

    apply_displacement(out)

    assert list(mesh.faces) != faces_before, (
        "the terrain was NOT carved with the cutter "
        f"{'before' if cutter_first else 'after'} the Site — if this passes "
        "only in one order, the mesh branch has grown an ordering test and "
        "the reference's 'terrain ignores tree order' is now wrong")
