"""A swept solid built from abutting segments tessellates without interior walls.

The third defect from the ``create_shape``-evaluates-each-face-independently
root cause, after ``weld_coincident_vertices`` and
``orient_faces_consistently``. A sweep built from abutting segments emits the
shared internal cap ONCE PER SEGMENT; before welding those are distinct vertex
indices and nothing looks wrong, and after welding they are two triangles on
the SAME corners, so every edge of that cap is traversed by four faces.

That is a NON-MANIFOLD edge, not a hole — and the two are not interchangeable:
``orient_faces_consistently`` states in its own docstring that non-manifold
input "has no consistent orientation to find". Running it against the doubled
surface propagates winding across a wall that should not exist, so the mesh
comes back non-manifold AND mis-wound. Hence the ordering assertion below..
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from lite_step.ifc.geometry import (
    drop_coincident_faces,
    mesh_boundary_edge_count,
    mesh_edge_defects,
    orient_faces_consistently,
)


def _unit_cube():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)
    return v, f


def _two_segments_sharing_a_doubled_cap():
    """Two stacked boxes whose shared cap is emitted by BOTH — the real shape.

    Vertices are already welded (one index per position), which is exactly the
    state the generator hands over: the cap's two copies index the same four
    corners, so the doubling is invisible in the vertex array and shows only
    in the edge census.
    """
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],      # z=0
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],      # z=1  (the shared cap)
        [0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2],      # z=2
    ], dtype=float)
    tris = [
        [0, 2, 1], [0, 3, 2],                            # bottom of lower box
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],      # lower sides
        [4, 5, 6], [4, 6, 7],                            # lower box's TOP cap
        [4, 5, 6], [4, 6, 7],                            # upper box's BOTTOM cap — SAME
        #   winding, which is the part that matters. OCC evaluates each face
        #   independently, so nothing makes the second copy oppose the first;
        #   an opposing pair would still be an interior wall but manifold3d's
        #   halfedge pairing happens to accept it, and then the defect is
        #   invisible until some later boolean. Same-wound is the case that
        #   actually reaches Error.NotManifold, and it is what the chord had.
        [4, 5, 9], [4, 9, 8], [5, 6, 10], [5, 10, 9],
        [6, 7, 11], [6, 11, 10], [7, 4, 8], [7, 8, 11],  # upper sides
        [8, 9, 10], [8, 10, 11],                         # top
    ]
    return v, np.array(tris, dtype=np.int64)


def _edge_uses(f):
    c: Counter = Counter()
    for a, b, cc in np.asarray(f).reshape(-1, 3):
        for e in ((a, b), (b, cc), (cc, a)):
            c[(int(min(e)), int(max(e)))] += 1
    return c


def test_the_fixture_really_carries_the_defect():
    """A fixture that is already clean would make every assertion below pass
    for the wrong reason."""
    v, f = _two_segments_sharing_a_doubled_cap()
    holes, non_manifold = mesh_edge_defects(f)
    assert holes == 0, "the doubled cap is not a hole"
    assert non_manifold > 0, "fixture does not reproduce the interior wall"


def test_dropping_the_interior_wall_closes_the_topology():
    v, f = _two_segments_sharing_a_doubled_cap()
    v2, f2 = drop_coincident_faces(v, f)
    holes, non_manifold = mesh_edge_defects(f2)
    assert (holes, non_manifold) == (0, 0)
    assert mesh_boundary_edge_count(f2) == 0
    # BOTH copies go, not one: keeping one would leave a membrane sealing the
    # solid's own interior.
    assert len(f2) == len(f) - 4


def test_a_clean_mesh_is_untouched():
    v, f = _unit_cube()
    v2, f2 = drop_coincident_faces(v, f)
    assert len(f2) == len(f)
    assert np.array_equal(np.sort(f2, axis=1), np.sort(f, axis=1))


def test_winding_is_ignored_when_pairing():
    """The two copies of an interior wall are wound oppositely by construction
    — that is what makes them cancel — so the pairing key must be the SET of
    corners, not the ordered triple."""
    v, f = _unit_cube()
    flipped = f[0][::-1]                      # same triangle, opposite winding
    f_dup = np.vstack([f, flipped[None, :]])
    _v2, f2 = drop_coincident_faces(v, f_dup)
    remaining = [tuple(sorted(t)) for t in f2]
    assert tuple(sorted(f[0])) not in remaining, "opposite winding must still pair"


def test_an_odd_multiplicity_keeps_exactly_one():
    """A surplus PAIR cancels; the face the surface genuinely needs stays."""
    v, f = _unit_cube()
    f_tripled = np.vstack([f, f[0][None, :], f[0][None, :]])
    _v2, f2 = drop_coincident_faces(v, f_tripled)
    keys = Counter(tuple(sorted(t)) for t in f2)
    assert keys[tuple(sorted(f[0]))] == 1
    assert len(f2) == len(f)


def test_the_order_is_load_bearing_dedup_then_orient():
    """Dedup alone is not enough, and orient alone cannot help.

    This is the assertion that pins the pipeline position. On the chord that
    found this, dedup alone still answered Error.NotManifold; dedup THEN
    re-orient answered Error.NoError.
    """
    pytest.importorskip("manifold3d")
    from manifold3d import Manifold, Mesh as MMesh

    def status(verts, faces):
        m = Manifold(MMesh(vert_properties=np.asarray(verts, dtype=np.float32),
                           tri_verts=np.asarray(faces, dtype=np.uint32)))
        return str(m.status())

    v, f = _two_segments_sharing_a_doubled_cap()
    assert "NoError" not in status(v, f), "fixture should start rejected"

    # orient FIRST (the order) cannot repair a non-manifold surface
    v_o, f_o = orient_faces_consistently(v, f)
    assert "NoError" not in status(v_o, f_o)

    # dedup THEN orient does
    v_d, f_d = drop_coincident_faces(v, f)
    v_f, f_f = orient_faces_consistently(v_d, f_d)
    assert "NoError" in status(v_f, f_f)

    m = Manifold(MMesh(vert_properties=np.asarray(v_f, dtype=np.float32),
                       tri_verts=np.asarray(f_f, dtype=np.uint32)))
    assert m.volume() == pytest.approx(2.0, rel=1e-6), "two stacked unit cubes"


def test_edge_defects_separates_holes_from_non_manifold():
    """The mislabel that misdirected: one number for two opposite causes.

    ``mesh_boundary_edge_count`` sums them, which is right for "is this
    closed?" and wrong for "why isn't it?".
    """
    v, f = _unit_cube()
    holed = f[:-1]                                        # remove one triangle
    assert mesh_edge_defects(holed) == (3, 0)

    v2, f2 = _two_segments_sharing_a_doubled_cap()
    holes, non_manifold = mesh_edge_defects(f2)
    assert holes == 0 and non_manifold > 0

    # And the summing helper cannot tell them apart — the reason for the split.
    assert mesh_boundary_edge_count(holed) == 3
    assert mesh_boundary_edge_count(f2) == non_manifold
