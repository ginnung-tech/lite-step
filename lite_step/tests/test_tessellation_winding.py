"""A curved solid tessellates into a mesh manifold3d will accept.

``weld_coincident_vertices`` repairs duplicate INDICES. This suite
covers its sibling defect from the same root cause: ``create_shape``
evaluates each OCC face independently, so nothing makes one face's winding
agree with the next one's.

The failure is silent by construction — ``manifold3d`` returns
``Error.NotManifold`` rather than raising, then propagates that error state
through every boolean it touches. A `Bar` used as a carve operand therefore
voids its host instead of notching it, with nothing in any log.
"""
from __future__ import annotations

import numpy as np
import pytest

from lite_step.ifc.geometry import _signed_volume, orient_faces_consistently


def _cube():
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int64)
    return v, f


def _faces_as_sets(f):
    return sorted(tuple(sorted(int(i) for i in t)) for t in f)


def test_a_reversed_face_is_brought_back_into_agreement() -> None:
    v, f = _cube()
    broken = f.copy()
    broken[3] = broken[3][::-1]
    _v, fixed = orient_faces_consistently(v, broken)
    assert _signed_volume(v, fixed) / 6.0 == pytest.approx(1.0)


def test_the_repair_changes_order_only_never_the_face_set() -> None:
    """Same vertices, same triangles — only the vertex order within each."""
    v, f = _cube()
    broken = f.copy()
    broken[0] = broken[0][::-1]
    broken[7] = broken[7][::-1]
    v2, fixed = orient_faces_consistently(v, broken)
    assert np.array_equal(v, v2)
    assert _faces_as_sets(fixed) == _faces_as_sets(f)


def test_an_inside_out_shell_is_flipped_whole() -> None:
    """Self-consistent but inward-facing is still wrong. Signed volume decides."""
    v, f = _cube()
    inside_out = f[:, ::-1].copy()
    assert _signed_volume(v, inside_out) < 0
    _v, fixed = orient_faces_consistently(v, inside_out)
    assert _signed_volume(v, fixed) / 6.0 == pytest.approx(1.0)


def test_an_already_correct_mesh_is_untouched() -> None:
    v, f = _cube()
    _v, out = orient_faces_consistently(v, f)
    assert np.array_equal(out, f)


def test_a_bar_tessellates_into_a_mesh_manifold3d_accepts() -> None:
    """The end-to-end case, and the one that found this.

    Before the repair: ``Error.NotManifold`` and volume 0.0, from a mesh with
    a perfect Euler characteristic and zero boundary edges — the low end cap
    wound backwards relative to the sides.
    """
    manifold3d = pytest.importorskip("manifold3d")
    pytest.importorskip("ifcopenshell.geom")

    from lite_step.ifc.generator import tessellate_solid_to_mesh
    from lite_step.models import Bar, Point
    from lite_step.strict import suspend_strict

    with suspend_strict():
        bar = Bar(path=[Point(x=0.0, y=0.0, z=0.0), Point(x=11.9, y=0.0, z=0.0)],
                  diameter=12, grade="B500B")
    verts, faces = tessellate_solid_to_mesh(bar)

    solid = manifold3d.Manifold(manifold3d.Mesh(
        np.asarray(verts, np.float32), np.asarray(faces, np.uint32)))
    assert solid.status() == manifold3d.Error.NoError, solid.status()

    # A polygonal approximation of a cylinder INSCRIBES it, so the volume
    # lands just under pi*r^2*L — never over, and never zero.
    analytic = np.pi * 6.0 ** 2 * 11.9
    assert 0.97 * analytic < solid.volume() <= analytic
