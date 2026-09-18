"""The tessellation weld — the root cause, at the unit level.

`test_displacement_annihilation.py` pins the SYMPTOM (a terrain that must not
vanish). This file pins the MECHANISM, so a regression says which of the three
links broke rather than only that the ground disappeared:

1. an index-duplicated seam makes a geometrically closed surface topologically
   open (`mesh_boundary_edge_count` > 0);
2. `manifold3d` answers that with `Error.NotManifold` and **does not raise**,
   then propagates the error state through every boolean — including a UNION,
   which is the proof it is not geometry;
3. `weld_coincident_vertices` repairs the topology without moving any surface,
   and `tessellate_solid_to_mesh` applies it at the kernel boundary.
"""
import numpy as np
import pytest

from lite_step.ifc.geometry import (weld_coincident_vertices,
                                    mesh_boundary_edge_count)
from lite_step.models import Point, Point2D, Sweep, Material


# A unit cube. `_SEAM_SPLIT` is the same cube with the two vertices of one top
# edge duplicated a femtometre away and the top face rewired onto the copies —
# exactly what OCC hands back for a swept solid's segment seam.
_CUBE_V = [(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.),
           (0., 0., 1.), (1., 0., 1.), (1., 1., 1.), (0., 1., 1.)]
_CUBE_F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
           (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
           (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
_EPSILON = 1.4e-14                                  # the measured gap


def _seam_split_cube():
    verts = list(_CUBE_V) + [(1. + _EPSILON, 1., 1.), (0., 1. + _EPSILON, 1.)]
    faces = [(8, 9, 7) if f == (4, 6, 7) else
             (4, 5, 8) if f == (4, 5, 6) else f for f in _CUBE_F]
    faces = [(4, 5, 8), (4, 8, 9)] + [f for f in _CUBE_F if f not in
                                      ((4, 5, 6), (4, 6, 7))]
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int64)


def test_closed_cube_has_no_boundary_edges():
    """Baseline, so a broken counter cannot make the next test pass."""
    assert mesh_boundary_edge_count(_CUBE_F) == 0


def test_a_duplicated_seam_opens_a_closed_surface():
    """The defect itself: the SURFACE is closed, the INDEX TOPOLOGY is not."""
    v, f = _seam_split_cube()
    assert len(v) == 10, "10 entries for 8 distinct points — that is the defect"
    assert mesh_boundary_edge_count(f) > 0, (
        "if this is 0 the fixture stopped reproducing the shape")


def test_weld_closes_it_without_moving_the_surface():
    v, f = _seam_split_cube()
    wv, wf = weld_coincident_vertices(v, f)
    assert len(wv) == 8, f"expected the 8 real corners, got {len(wv)}"
    assert mesh_boundary_edge_count(wf) == 0, "still not watertight"
    # No surface moved: the welded corners are the cube's corners, exactly.
    assert sorted(map(tuple, np.round(wv, 12))) == sorted(_CUBE_V)


def test_weld_never_merges_real_features():
    """Falsification, the destructive side. A weld that eats authored detail
    would be a far worse bug than the one it fixes — it would shrink geometry
    silently. The default tolerance is span * 1e-9; 1 mm of a 1 m cube is six
    orders above it and must survive untouched."""
    v = np.array(_CUBE_V + [(0.001, 0., 0.)], dtype=np.float64)
    f = np.array(_CUBE_F, dtype=np.int64)
    wv, wf = weld_coincident_vertices(v, f)
    assert len(wv) == 9, (
        "a 1 mm feature on a 1 m body was welded away — the tolerance is wrong")
    assert np.array_equal(wf, f), "faces must be untouched when nothing merged"


def test_weld_is_a_noop_when_nothing_is_coincident():
    v = np.array(_CUBE_V, dtype=np.float64)
    f = np.array(_CUBE_F, dtype=np.int64)
    wv, wf = weld_coincident_vertices(v, f)
    assert np.array_equal(wv, v) and np.array_equal(wf, f)


def test_weld_drops_triangles_that_collapse():
    """A triangle whose two corners merge is a line and must not be emitted —
    a zero-area face is another way to make a mesh unusable as an operand."""
    v = np.array([(0., 0., 0.), (1., 0., 0.), (1. + _EPSILON, 0., 0.)],
                 dtype=np.float64)
    wv, wf = weld_coincident_vertices(v, np.array([(0, 1, 2)], dtype=np.int64))
    assert len(wv) == 2 and len(wf) == 0


# ── manifold3d's contract: it reports by STATUS, never by raising ────────────

def test_manifold3d_refuses_silently_and_poisons_every_boolean():
    """The step that made invisible, asserted rather than remembered.

    `_carve_mesh_host` wraps `Manifold(MMesh(...))` in a `try/except`. That
    except CANNOT fire for a non-manifold mesh, and the resulting value is not
    merely wrong — it is contagious. The union is the tell: a union can never
    reduce volume, so an empty union is proof the library stopped doing
    geometry and started propagating an error."""
    Manifold = pytest.importorskip("manifold3d").Manifold
    MMesh = pytest.importorskip("manifold3d").Mesh

    def mk(v, f):
        return Manifold(MMesh(vert_properties=np.asarray(v, dtype=np.float32),
                              tri_verts=np.asarray(f, dtype=np.uint32)))

    good = mk(_CUBE_V, _CUBE_F)
    assert not good.is_empty() and good.volume() == pytest.approx(1.0)

    v, f = _seam_split_cube()
    bad = mk(v, f)                                  # no exception raised
    assert bad.is_empty(), "manifold3d accepted a non-manifold mesh"
    assert (good - bad).is_empty(), "difference should be poisoned"
    assert (good + bad).is_empty(), (
        "a UNION came back non-empty — then the error stops propagating and "
        "the mechanism has changed; re-derive it before trusting the fix")

    wv, wf = weld_coincident_vertices(v, f)
    welded = mk(wv, wf)
    assert not welded.is_empty()
    assert welded.volume() == pytest.approx(1.0, rel=1e-4)


# ── the kernel boundary: tessellate_solid_to_mesh welds ─────────────────────

def _deck_ribbon():
    """The bridge deck, verbatim — a 3-point path whose middle vertex
    reverses direction. That joint is where OCC duplicates the seam."""
    return Sweep(name="deck_axis_preview",
                 profile=[Point2D(x=-3500, y=-250), Point2D(x=3500, y=-250),
                          Point2D(x=3500, y=250), Point2D(x=-3500, y=250)],
                 path=[Point(x=-73005, y=-97546, z=107770),
                       Point(x=0, y=0, z=107213),
                       Point(x=73006, y=97545, z=106657)],
                 material=Material(key="Concrete_C30-37"))


def _normalized_ribbon():
    """The ribbon as the carve sees it — POST-normalize, in meters.

    Not a detail. `displacement` runs after `normalize_project_to_meters`, and
    OCC's internal confusion tolerances are absolute, so the same Sweep
    tessellates to a materially different artefact size at each scale (see the
    table in `weld_coincident_vertices`). Testing the authoring-millimeter
    answer would pin a mesh no production code path ever asks for.
    """
    from lite_step.compiler.executor import normalize_project_to_meters
    from lite_step.compiler.displacement import _collect
    from lite_step.models import Project

    proj = Project(name="t")
    proj.add(_deck_ribbon())
    solids, _meshes = _collect(normalize_project_to_meters(proj))
    return solids[0][0]


def test_tessellate_solid_to_mesh_returns_a_watertight_operand():
    """The fix at the layer that owns the defect. Unwelded this mesh had 20
    vertices for 16 points and 8 boundary edges; ifcopenshell's own
    `weld-vertices` pass is already ON (it collapses 56 raw entries to 20) and
    does not reach 1.4e-14 m."""
    pytest.importorskip("ifcopenshell")
    pytest.importorskip("manifold3d")
    from lite_step.ifc.generator import tessellate_solid_to_mesh
    from manifold3d import Manifold, Mesh as MMesh

    out = tessellate_solid_to_mesh(_normalized_ribbon())
    assert out is not None, "the kernel refused the fixture — it is the repro"
    verts, faces = out
    assert len(verts) == 16, (
        f"expected the 16 distinct section corners, got {len(verts)} — an "
        "unwelded seam is the defect")
    assert mesh_boundary_edge_count(faces) == 0, (
        "the swept solid is not closed — manifold3d will refuse it and then "
        "erase whatever it is subtracted from")
    m = Manifold(MMesh(vert_properties=np.asarray(verts, dtype=np.float32),
                       tri_verts=np.asarray(faces, dtype=np.uint32)))
    assert not m.is_empty(), f"manifold3d still refuses it: {m.status()}"
    # ~244 m of 7.0 m x 0.5 m ribbon.
    assert m.volume() == pytest.approx(853.0, rel=0.02)


def test_the_carve_operand_reaches_displacement_watertight():
    """End of the chain: what `_occupant_mesh` actually hands the boolean.

    `tessellate_solid_to_mesh` being clean is worth nothing if the operand path
    re-derives or post-processes the mesh, so assert the value at the point of
    use rather than one function earlier.
    """
    pytest.importorskip("ifcopenshell")
    from lite_step.compiler.displacement import _occupant_mesh

    verts, faces = _occupant_mesh(_normalized_ribbon())
    assert mesh_boundary_edge_count(faces) == 0, (
        "the carve operand is not a closed surface — it will erase its host")
