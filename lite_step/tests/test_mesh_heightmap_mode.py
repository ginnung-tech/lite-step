"""Mesh heightmap mode — a first-class, compact way to specify a mesh.

``Mesh(heightmap=…, depth=…, corner_min=…, corner_max=…)`` is the second
specification mode next to explicit ``vertices``+``faces``: a grid of literal
top-surface heights (mm) over an XY rectangle, extruded down to one flat
bottom plane ``depth`` mm below the LOWEST height. The compiler owns the
stitching (top + bottom + 4 skirts, outward 0-based windings — the proven
site-block prism, moved out of skill boilerplate); the model retains the
heightmap fields so ``output.py`` stays compact and future surface queries
can read the grid.

Pinned here:
* construction invariants (vertex/face counts, literal heights, corner
  mapping, auto-watertight) — under strict int-mm, the production default;
* strict rejection of float heights/depth at the FIELD boundary (before
  pydantic's float coercion — the gate placement that makes ints pass);
* mode exclusivity + grid validation errors;
* parity with the hand-built site-block stitching (identical geometry);
* mm→m normalization of the RETAINED fields on BOTH routes — host and
  boolean operand (the coverage lesson: every new mm field
  gets a scaling test);
* carve end-to-end: heightmap terrain − heightmap cut actually removes
  volume via manifold3d and generates IFC (the stenkant shape).
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc
from lite_step.models import Box, Project, Site, Mesh, Point, Point2D, Revolve
from lite_step.models.heightfield import tessellate_heightmap
from lite_step.strict import enable_strict_int_mm, disable_strict_int_mm


@pytest.fixture()
def strict():
    """Run a test under strict int-mm (the production default for scripts)."""
    enable_strict_int_mm()
    try:
        yield
    finally:
        disable_strict_int_mm()


def _terrain(heightmap=None, depth=15000):
    return Mesh(
        heightmap=heightmap if heightmap is not None else [[0, 0], [0, 0]],
        depth=depth,
        corner_min=Point2D(x=-16000, y=-14500),
        corner_max=Point2D(x=16000, y=14500),
    )


# ── construction ──────────────────────────────────────────────────────────────

def test_heightmap_mode_constructs_closed_prism(strict):
    m = _terrain(heightmap=[[0, 200], [100, 400]])
    # 2x2 grid → 8 verts (top+bottom), 12 faces (a box).
    assert len(m.vertices) == 8
    assert len(m.faces) == 12
    assert m.is_watertight is True  # forced — closed by construction
    # Retained authoring fields survive expansion.
    assert m.heightmap == [[0, 200], [100, 400]]
    assert m.depth == 15000


def test_heights_are_literal_and_bottom_is_min_minus_depth(strict):
    m = _terrain(heightmap=[[0, 200], [100, 400]], depth=15000)
    zs = sorted(p.z for p in m.vertices)
    assert zs[-1] == 400.0, "heights are literal — no auto-shift"
    assert zs[0] == -15000.0, "bottom = min(heights) - depth = 0 - 15000"


def test_grid_orientation_row0_col0_at_corner_min(strict):
    m = _terrain()
    # vertex 0 = heightmap[0][0] = SW corner (corner_min).
    assert m.vertices[0].x == -16000 and m.vertices[0].y == -14500
    # last top vertex = heightmap[-1][-1] = NE corner (corner_max).
    top_last = m.vertices[3]  # rows*cols - 1 = 3 for a 2x2 top
    assert top_last.x == 16000 and top_last.y == 14500


def test_face_count_formula_3x3(strict):
    m = Mesh(
        heightmap=[[0, 100, 0], [50, 400, 50], [0, 100, 0]],
        depth=2000,
        corner_min=Point2D(x=0, y=0),
        corner_max=Point2D(x=9000, y=9000),
    )
    rows = cols = 3
    assert len(m.vertices) == 2 * rows * cols
    # top + bottom: 2·2(r-1)(c-1); skirts: 4(r-1) + 4(c-1).
    assert len(m.faces) == 4 * (rows - 1) * (cols - 1) + 4 * (rows - 1) + 4 * (cols - 1)


def test_interpolated_grid_coords_rounded_to_int_mm(strict):
    # Extent 1000 over 4 cols → interior cols at x=333.33…/666.67… must round
    # to int mm (computed values — the I() convention).
    m = Mesh(
        heightmap=[[0, 0, 0, 0], [0, 0, 0, 0]],
        depth=500,
        corner_min=Point2D(x=0, y=0),
        corner_max=Point2D(x=1000, y=1000),
    )
    xs = sorted({p.x for p in m.vertices})
    assert xs == [0, 333, 667, 1000]


def test_explicit_mode_unchanged(strict):
    """Guard: vertices+faces meshes behave exactly as before; heightmap
    fields default to None."""
    m = Mesh(
        vertices=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0), Point(x=0, y=1000, z=0)],
        faces=[(0, 1, 2)],
    )
    assert m.heightmap is None and m.depth is None
    assert m.is_watertight is False  # not forced in explicit mode


# ── strict int-mm at the field boundary ───────────────────────────────────────

def test_strict_rejects_float_height_with_grid_message(strict):
    with pytest.raises(Exception, match=r"heightmap\[0\]\[1\]"):
        Mesh(heightmap=[[0, 100.5], [0, 0]], depth=500,
             corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1000, y=1000))


def test_strict_rejects_float_depth(strict):
    with pytest.raises(Exception, match="depth"):
        Mesh(heightmap=[[0, 0], [0, 0]], depth=500.5,
             corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1000, y=1000))


def test_non_strict_accepts_float_heights():
    """Internal (non-LLM) builders run with strict suspended — float heights
    pass, same as float Points do."""
    m = Mesh(heightmap=[[0.5, 0.5], [0.5, 0.5]], depth=500.0,
             corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1000, y=1000))
    assert len(m.vertices) == 8


# ── mode exclusivity + grid validation ────────────────────────────────────────

@pytest.mark.parametrize("kwargs, fragment", [
    # both modes at once
    (dict(vertices=[Point(x=0, y=0, z=0)] * 3, faces=[(0, 1, 2)],
          heightmap=[[0, 0], [0, 0]], depth=5,
          corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1, y=1)), "not both"),
    # incomplete heightmap mode
    (dict(heightmap=[[0, 0], [0, 0]], depth=5), "missing"),
    # neither mode
    (dict(), "explicit mode"),
    # grid too small / ragged / bad depth / inverted corners
    (dict(heightmap=[[0, 0]], depth=5,
          corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1, y=1)), "2×2"),
    (dict(heightmap=[[0, 0], [0, 0, 0]], depth=5,
          corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1, y=1)), "rectangular"),
    (dict(heightmap=[[0, 0], [0, 0]], depth=0,
          corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1, y=1)), "positive"),
    (dict(heightmap=[[0, 0], [0, 0]], depth=5,
          corner_min=Point2D(x=1, y=1), corner_max=Point2D(x=0, y=0)), "strictly below"),
])
def test_invalid_specifications_rejected(kwargs, fragment):
    with pytest.raises(Exception, match=fragment):
        Mesh(**kwargs)


# ── parity with the hand-built site-block stitching ───────────────────────────

def test_parity_with_handbuilt_prism(strict):
    """The compiler tessellation must produce the same closed prism the
    site-block boilerplate built by hand (same vertex set, same face count) —
    it IS that code, moved into the engine."""
    hm = [[0, 200], [100, 400]]
    m = Mesh(heightmap=hm, depth=15000,
             corner_min=Point2D(x=-16000, y=-14500),
             corner_max=Point2D(x=16000, y=14500))

    # Hand-build per the moremester/site-block convention (heights literal).
    top = []
    for r in range(2):
        for c in range(2):
            x = -16000 + 32000 * c
            y = -14500 + 29000 * r
            top.append((x, y, hm[r][c]))
    bottom_z = min(z for _, _, z in top) - 15000
    expect = top + [(x, y, bottom_z) for x, y, _ in top]

    got = [(p.x, p.y, p.z) for p in m.vertices]
    assert got == [(float(x), float(y), float(z)) for x, y, z in expect]
    assert len(m.faces) == 12


def test_tessellate_function_directly():
    """The pure function is usable standalone (it is the single home of the
    stitching — no skill should reimplement it)."""
    verts, faces = tessellate_heightmap(
        [[0, 0], [0, 0]], 1000,
        Point2D(x=0, y=0), Point2D(x=2000, y=2000),
    )
    assert len(verts) == 8 and len(faces) == 12


# ── mm→m normalization of retained fields (both routes) ───────────────────────

def _building_with_heightmap_terrain():
    b = Project(name="t")
    site = Site(name="site")
    terr = _terrain()
    site.add(terr)
    return b, site, terr


def test_normalizer_scales_retained_fields_on_host():
    b, site, _terr = _building_with_heightmap_terrain()
    b.add(site)
    out = normalize_project_to_meters(b)
    mesh = [c for c in out.sites[0]._elements if isinstance(c, Mesh)][0]
    assert abs(mesh.depth - 15.0) < 1e-9
    assert abs(mesh.corner_min.x - (-16.0)) < 1e-9
    assert abs(mesh.corner_max.y - 14.5) < 1e-9
    assert mesh.heightmap == [[0.0, 0.0], [0.0, 0.0]]
    # And the tessellated vertices scaled as usual.
    assert abs(min(p.z for p in mesh.vertices) - (-15.0)) < 1e-9


def test_normalizer_scales_retained_fields_on_cut_operand():
    b, site, terr = _building_with_heightmap_terrain()
    cut = Mesh(heightmap=[[50, 50], [50, 50]], depth=550,
               corner_min=Point2D(x=6000, y=-4500), corner_max=Point2D(x=6500, y=4500))
    terr.difference(cut)
    b.add(site)
    out = normalize_project_to_meters(b)
    mesh = [c for c in out.sites[0]._elements if isinstance(c, Mesh)][0]
    op = mesh._cuts[0]
    assert abs(op.depth - 0.55) < 1e-9
    assert abs(op.corner_min.x - 6.0) < 1e-9
    assert abs(op.vertices[0].z - 0.05) < 1e-9


def test_normalizer_scales_retained_fields_on_void_operand():
    b, site, _terr = _building_with_heightmap_terrain()
    vtool = Mesh(heightmap=[[50, 50], [50, 50]], depth=550,
                 corner_min=Point2D(x=6000, y=-4500), corner_max=Point2D(x=6500, y=4500))
    site.void(vtool)
    b.add(site)
    out = normalize_project_to_meters(b)
    op = out.sites[0]._voids[0]
    assert abs(op.depth - 0.55) < 1e-9
    assert abs(op.vertices[0].x - 6.0) < 1e-9


# ── carve end-to-end (the stenkant shape) ─────────────────────────────────────

def test_heightmap_terrain_minus_heightmap_cut_carves():
    """A heightmap-mode terrain carved by a heightmap-mode gravel cut — the
    exact stenkant pattern, now two constructor calls instead of two copies
    of stitching boilerplate. Must remove volume (verts grow) and generate."""
    pytest.importorskip("manifold3d")
    b, site, terr = _building_with_heightmap_terrain()
    n_before = len(terr.vertices)
    cut = Mesh(heightmap=[[50, 50], [50, 50]], depth=550,
               corner_min=Point2D(x=6000, y=-4500), corner_max=Point2D(x=6500, y=4500))
    terr.difference(cut)
    b.add(site)

    out = normalize_project_to_meters(b)
    assert generate_ifc(out).success
    mesh = [c for c in out.sites[0]._elements if isinstance(c, Mesh)][0]
    assert len(mesh.vertices) > n_before, "carve added no geometry — silent no-op"


# ── stage 2: surface queries + absolute level bottom ──────────────────────────

def test_height_at_exact_bilinear_and_clamped(strict):
    m = Mesh(heightmap=[[0, 100], [200, 400]], depth=15000,
             corner_min=Point2D(x=-1000, y=-1000), corner_max=Point2D(x=1000, y=1000))
    assert m.height_at(x=-1000, y=-1000) == 0        # grid corner exact
    assert m.height_at(x=1000, y=1000) == 400
    assert abs(m.height_at(x=0, y=0) - 175.0) < 1e-9  # centre bilinear mean
    assert m.height_at(x=-9999, y=-9999) == 0         # clamps to nearest edge


def test_height_at_degenerate_mesh_is_loud(strict):
    """A mesh with no projectable area (all-degenerate triangles) has no
    queryable surface — loud error. (Explicit meshes in general ARE
    queryable now — see the general-path tests below.)"""
    m = Mesh(vertices=[Point(x=0, y=0, z=0)] * 3, faces=[(0, 1, 2)])
    with pytest.raises(ValueError, match="misses"):
        m.height_at(x=0, y=0)


def test_heightmap_at_round_true_feeds_strict_authoring(strict):
    """The terrain-conforming top in one call — ``round=True`` gives int
    heights that feed straight back into Mesh(heightmap=...) under strict
    int-mm (the stenkant patch)."""
    terr = Mesh(heightmap=[[0, 100], [200, 400]], depth=15000,
                corner_min=Point2D(x=-1000, y=-1000), corner_max=Point2D(x=1000, y=1000))
    g = terr.heightmap_at(corner_min=Point2D(x=-1000, y=-1000),
                         corner_max=Point2D(x=0, y=0), rows=2, cols=3, round=True)
    assert all(isinstance(h, int) for row in g for h in row), g
    assert g[0][0] == 0
    # And the grid is valid strict-mode authoring input:
    patch = Mesh(heightmap=g, bottom_z=-500,
                 corner_min=Point2D(x=-1000, y=-1000), corner_max=Point2D(x=0, y=0))
    assert patch.is_watertight is True


def test_bottom_z_absolute_level_plane(strict):
    """bottom_z= is the absolute alternative to depth= — sibling prisms share
    one level bottom with no per-patch depth arithmetic."""
    m = Mesh(heightmap=[[100, 100], [300, 300]], bottom_z=-500,
             corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1000, y=1000))
    assert min(p.z for p in m.vertices) == -500.0
    assert max(p.z for p in m.vertices) == 300.0


@pytest.mark.parametrize("kwargs, fragment", [
    # both depth and bottom_z
    (dict(heightmap=[[0, 0], [0, 0]], depth=500, bottom_z=-500,
          corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1, y=1)), "exactly ONE"),
    # bottom_z not strictly below the lowest height
    (dict(heightmap=[[100, 100], [100, 100]], bottom_z=100,
          corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1, y=1)), "strictly below the lowest"),
])
def test_bottom_z_validation(kwargs, fragment):
    with pytest.raises(Exception, match=fragment):
        Mesh(**kwargs)


def test_strict_rejects_float_bottom_z(strict):
    with pytest.raises(Exception, match="bottom_z"):
        Mesh(heightmap=[[0, 0], [0, 0]], bottom_z=-500.5,
             corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1000, y=1000))


def test_normalizer_scales_bottom_z():
    """The/rule for the NEW mm field: bottom_z scales mm→m on
    the wrapper-child route."""
    from lite_step.models import Element
    b = Project(name="t")
    site = Site(name="site")
    tw = Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN",
                 name="terrain")
    tw.add(Mesh(heightmap=[[0, 0], [0, 0]], depth=15000,
                corner_min=Point2D(x=-16000, y=-14500),
                corner_max=Point2D(x=16000, y=14500)))
    site.add(tw)
    bed = Element(ifc_class="IfcEarthworksFill", name="bed")
    bed.add(Mesh(heightmap=[[50, 50], [50, 50]], bottom_z=-450,
                 corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=500, y=500)))
    site.add(bed)
    b.add(site)
    out = normalize_project_to_meters(b)
    bm = [s for s in out.sites[0]._elements
          if type(s).__name__ == "Element"
          and s.ifc_class == "IfcEarthworksFill"][0]._elements[0]
    assert abs(bm.bottom_z - (-0.45)) < 1e-9
    # And the query surface works in meters post-normalize:
    assert abs(bm.height_at(x=0.25, y=0.25) - 0.05) < 1e-9


# ── general surface queries — ANY mesh, not just heightmap mode ───────────────

def _box_mesh_explicit(z_top=300, z_bot=-500):
    V = [Point(x=x, y=y, z=z) for x in (0, 2000) for y in (0, 2000)
         for z in (z_bot, z_top)]
    F = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 2, 6), (0, 6, 4),
         (1, 5, 7), (1, 7, 3), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6)]
    return Mesh(vertices=V, faces=F, is_watertight=True)


def test_height_at_general_path_explicit_mesh(strict):
    """An explicit mesh is queryable too: the highest vertical intersection
    with its triangles (top face of a closed prism, never the bottom)."""
    m = _box_mesh_explicit()
    assert abs(m.height_at(x=1000, y=1000) - 300.0) < 1e-9


def test_height_at_rotated_terrain(strict):
    """The user's case: a terrain with a baked-in rotation — the ray-cast
    reads actual triangles, so orientation is irrelevant."""
    import math
    th = math.radians(30)

    def rot(x, y):
        return (x * math.cos(th) - y * math.sin(th),
                x * math.sin(th) + y * math.cos(th))

    from lite_step.strict import suspend_strict
    verts = []
    for (x, y, z) in [(0, 0, 0), (2000, 0, 100), (2000, 2000, 200),
                      (0, 2000, 100), (0, 0, -500), (2000, 0, -500),
                      (2000, 2000, -500), (0, 2000, -500)]:
        rx, ry = rot(x, y)
        verts.append(Point.model_construct(x=rx, y=ry, z=z))
    faces = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
             (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]
    with suspend_strict():
        m = Mesh(vertices=verts, faces=faces, is_watertight=True)
    cx, cy = rot(1000, 1000)
    assert abs(m.height_at(x=cx, y=cy) - 100.0) < 5.0  # tilted-plane centre


def test_height_at_general_miss_is_loud(strict):
    """Off a general mesh's footprint there IS no surface — loud error, not
    a silent clamp (clamping only applies to the heightmap fast path)."""
    m = _box_mesh_explicit()
    with pytest.raises(ValueError, match="misses"):
        m.height_at(x=99999, y=99999)


def test_heightmap_at_general_path(strict):
    """heightmap_at works on explicit meshes too — rows along Y, cols along X,
    result[r][c] with row 0 at corner_min.y (the heightmap convention).
    Default return is floats; round=True flips to int mm."""
    m = _box_mesh_explicit(z_top=250)
    g = m.heightmap_at(corner_min=Point2D(x=0, y=0),
                      corner_max=Point2D(x=2000, y=2000), rows=2, cols=3)
    assert g == [[250, 250, 250], [250, 250, 250]]
    assert all(isinstance(h, float) for row in g for h in row)
    gi = m.heightmap_at(corner_min=Point2D(x=0, y=0),
                        corner_max=Point2D(x=2000, y=2000), rows=2, cols=3,
                        round=True)
    assert all(isinstance(h, int) for row in gi for h in row)


def test_skirt_faces_never_answer_the_query(strict):
    """Vertical faces (prism skirts) project to degenerate 2D triangles and
    are skipped — a query on the footprint edge reads the top, not a wall."""
    m = _box_mesh_explicit(z_top=300)
    assert abs(m.height_at(x=0, y=1000) - 300.0) < 1e-9  # on the west edge


# ── queries on ALL geometry + explicit-direction raycast ──────────────────────

def test_box_surface_and_heightmap_at(strict):
    """Queries are universal geometry features — a Box answers too (prism
    numpy tessellation under the hood)."""
    b = Box(start=Point(x=0, y=0, z=-500), end=Point(x=2000, y=2000, z=300))
    assert abs(b.height_at(x=1000, y=1000) - 300.0) < 1e-9
    g = b.heightmap_at(corner_min=Point2D(x=0, y=0),
                      corner_max=Point2D(x=2000, y=2000), rows=2, cols=2)
    assert g == [[300, 300], [300, 300]]


def test_raycast_explicit_direction_vector(strict):
    """raycast(origin=, direction=) is the general 3D measurement: any
    direction, nearest hit as a Point, None on a miss."""
    b = Box(start=Point(x=0, y=0, z=-500), end=Point(x=2000, y=2000, z=300))
    west = b.raycast(origin=Point(x=-1000, y=1000, z=0), direction=(1, 0, 0))
    assert west is not None and abs(west.x - 0.0) < 1e-9
    down = b.raycast(origin=Point(x=1000, y=1000, z=99999), direction=(0, 0, -1))
    assert down is not None and abs(down.z - 300.0) < 1e-9
    assert b.raycast(origin=Point(x=-1000, y=99999, z=0),
                     direction=(1, 0, 0)) is None
    with pytest.raises(ValueError, match="non-zero"):
        b.raycast(origin=Point(x=0, y=0, z=0), direction=(0, 0, 0))


def test_placed_element_queried_in_world_space():
    """placement=Transform is applied — the query answers where the element
    actually IS."""
    from lite_step.models import Transform
    from lite_step.strict import suspend_strict
    with suspend_strict():
        pb = Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=500),
                 placement=Transform(origin=Point(x=5000, y=0, z=0)))
    assert abs(pb.height_at(x=5500, y=500) - 500.0) < 1e-9


def test_container_queries_reject_loudly(strict):
    from lite_step.models import Wall
    with pytest.raises(ValueError, match="not a geometric primitive"):
        Wall(name="w").height_at(x=0, y=0)


def test_heightmap_raycast_agrees_on_planar_cells(strict):
    """On a PLANAR heightmap cell the bilinear fast path and the triangle
    ray-cast agree exactly. (On a non-planar cell they legitimately differ:
    height_at reads the AUTHORED bilinear surface, raycast reads the
    tessellated triangles — the documented two-representation nuance.)"""
    hm = Mesh(heightmap=[[0, 100], [100, 200]], depth=15000,
              corner_min=Point2D(x=-1000, y=-1000),
              corner_max=Point2D(x=1000, y=1000))
    r = hm.raycast(origin=Point(x=0, y=0, z=99999), direction=(0, 0, -1))
    assert r is not None
    assert abs(r.z - hm.height_at(x=0, y=0)) < 1e-6


# ── round= on every query + inside-origin raycast semantics ───────────────────

def test_raycast_returns_int_mm_by_default(strict):
    """``raycast`` returns int mm unless asked otherwise. ``height_at`` and
    ``heightmap_at`` still default to floats — and the asymmetry is the
    whole point, not an oversight.

    raycast is the ONLY one of the three that can smuggle a float into
    geometry. It builds its result with ``Point.model_construct``, which
    SKIPS the int-mm field validator, so a float-bearing Point flows into
    ``Box(start=hit)`` unchallenged even though Rule 2 forbids floats —
    while the honest spelling ``Point(x=hit.x)`` fails loudly. Same-shaped
    code, opposite outcomes.

    The other two cannot do that. Misusing their floats hits a validator
    that fires: ``Point(x=<float>)`` raises, and
    ``Mesh(heightmap=[[<float>]])`` raises with a grid-specific message. So
    their float default is safe — and it is also REQUIRED, because after
    ``normalize_project_to_meters`` the model is in METRES and a terrain
    height of 0.05 m rounds to 0 (see
    ``test_normalizer_scales_bottom_z``, which is what caught this).

    Blanket-rounding all three is wrong."""
    b = Box(start=Point(x=0, y=0, z=-500), end=Point(x=2000, y=2000, z=300))

    hit = b.raycast(origin=Point(x=-1000, y=1000, z=0), direction=(1, 0, 0))
    assert isinstance(hit.x, int) and isinstance(hit.y, int) and isinstance(hit.z, int), (
        "a bare raycast must return int mm — a float-bearing Point bypasses "
        "the int-mm validator and lands in geometry silently"
    )
    hitf = b.raycast(origin=Point(x=-1000, y=1000, z=0), direction=(1, 0, 0),
                     round=False)
    assert isinstance(hitf.x, float), "round=False stays available for measurement math"

    # The safe two keep their float default, and round=True still works.
    z = b.height_at(x=1000, y=1000)
    assert isinstance(z, float)
    assert b.height_at(x=1000, y=1000, round=True) == 300

    hm = Mesh(heightmap=[[0, 100], [100, 201]], depth=15000,
              corner_min=Point2D(x=-1000, y=-1000),
              corner_max=Point2D(x=1000, y=1000))
    zc = hm.height_at(x=0, y=0)
    assert isinstance(zc, float) and abs(zc - 100.25) < 1e-9
    assert hm.height_at(x=0, y=0, round=True) == 100

    g = hm.heightmap_at(corner_min=Point2D(x=-1000, y=-1000),
                        corner_max=Point2D(x=1000, y=1000), rows=2, cols=2)
    assert all(isinstance(h, float) for row in g for h in row)
    gi = hm.heightmap_at(corner_min=Point2D(x=-1000, y=-1000),
                         corner_max=Point2D(x=1000, y=1000), rows=2, cols=2,
                         round=True)
    assert all(isinstance(h, int) for row in gi for h in row)


def test_the_two_safe_queries_fail_loudly_on_misuse(strict):
    """Why height_at/heightmap_at may keep float defaults: the validators
    that raycast bypasses DO fire for them. If either of these ever stops
    raising, their float default becomes as unsafe as raycast's was."""
    b = Box(start=Point(x=0, y=0, z=-500), end=Point(x=2000, y=2000, z=300))
    zf = b.height_at(x=1000, y=1000)          # a float
    with pytest.raises(Exception):
        Point(x=zf, y=0, z=0)                  # ...cannot become geometry
    with pytest.raises(Exception):
        Mesh(heightmap=[[0.5, 1.5], [2.5, 3.5]], depth=1000,
             corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=100, y=100))


def test_curved_query_tessellation_is_size_relative(strict):
    """The query tessellator's deflection scales with the element's
    extent, so a mm-authored donut arc tessellates to ~1k verts (was 443k —
    the kernel default deflection is in MODEL units and the query path feeds
    raw mm). Accuracy stays query-grade: a ray from the hole centre hits the
    inner tube wall at R - r within chord tolerance."""
    import math
    I = lambda x: int(round(x))  # noqa: E731
    prof = [Point2D(x=I(1500 + 300 * math.cos(t)), y=I(300 * math.sin(t)))
            for t in [k * 2 * math.pi / 16 for k in range(16)]]
    d = Revolve(profile=prof,
                path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=1000)],
                angle=28800)
    verts, _ = d._query_triangles()
    assert len(verts) < 20000, f"tessellation blew up: {len(verts)} verts"
    hit = d.raycast(origin=Point(x=0, y=0, z=0), direction=(1, 0, 0))
    assert hit is not None and abs(hit.x - 1200.0) < 30.0, hit


def test_raycast_from_inside_reports_exit_point(strict):
    """PINNED semantics: an origin INSIDE the geometry reports the EXIT
    point — the first surface crossed along the direction (no backface
    culling). That's how interior clearance is measured. An origin exactly
    ON the surface casting outward returns itself (t=0 hit)."""
    b = Box(start=Point(x=0, y=0, z=0), end=Point(x=2000, y=2000, z=1000))
    east = b.raycast(origin=Point(x=1000, y=1000, z=500), direction=(1, 0, 0))
    assert (abs(east.x - 2000.0) < 1e-9 and abs(east.y - 1000.0) < 1e-9
            and abs(east.z - 500.0) < 1e-9)
    floor = b.raycast(origin=Point(x=1000, y=1000, z=500), direction=(0, 0, -1))
    assert abs(floor.z - 0.0) < 1e-9
    on_surface = b.raycast(origin=Point(x=1000, y=1000, z=1000),
                           direction=(0, 0, 1))
    assert on_surface is not None and abs(on_surface.z - 1000.0) < 1e-9
