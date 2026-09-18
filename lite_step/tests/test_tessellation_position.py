"""The tessellation helper must not hand back geometry in the wrong PLACE.

``generator.tessellate_solid_to_mesh`` builds a throwaway single-element IFC
model and asks the kernel for a mesh. That mesh feeds the displacement carve,
``raycast`` / ``height_at`` / ``heightmap_at``, and ``miter()`` — so a mesh in
the wrong place is worse than no mesh at all: it cuts a hole somewhere the
author never asked for, and nothing downstream can tell.: ``Pipe`` and ``Bar`` come back POINT-REFLECTED THROUGH THE
ORIGIN from that helper — a pipe authored at x=1000..3000 tessellates to
x=-3000..-1000 — while ``Sweep`` and ``Revolve`` through the same helper are
correct, and the SHIPPED IFC is correct for all four. The entity the helper
writes is verifiably right (its directrix carries the authored points), so the
fault is in how the kernel evaluates it in that single-element model.

Nothing caught it for months because the only prior guard on this function
asserts vertex COUNT (``len(verts) < 20000``, the 443k-vert donut regression)
— and a reflected mesh has exactly the right count. Position was never
checked. That is the real lesson: a metric that cannot distinguish right from
mirrored is not a guard.
"""
import numpy as np
import pytest

from lite_step.ifc.generator import tessellate_solid_to_mesh
from lite_step.models import Bar, Pipe, Point, Point2D, Revolve, Sweep

pytest.importorskip("ifcopenshell.geom")

_X, _Y, _Z = 1000, 2000, 3000


def _pipe():
    return Pipe(path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X + 2000, y=_Y, z=_Z)],
                radius=100, name="p")


def _bar():
    return Bar(path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X, y=_Y, z=_Z + 1000)],
               diameter=32, name="b")


def _revolve():
    return Revolve(
        profile=[Point2D(x=100, y=0), Point2D(x=300, y=0), Point2D(x=300, y=200)],
        path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X, y=_Y, z=_Z + 1000)],
        angle=360, name="r")


def _sweep():
    return Sweep(
        profile=[Point2D(x=-50, y=-50), Point2D(x=50, y=-50),
                 Point2D(x=50, y=50), Point2D(x=-50, y=50)],
        path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X + 2000, y=_Y, z=_Z)],
        name="s")


@pytest.mark.parametrize("factory", [_sweep, _revolve], ids=["sweep", "revolve"])
def test_correct_tessellations_are_accepted(factory):
    """The guard must not cost us the shapes that DO work.

    A position check that refuses everything is not a guard, it is an outage.
    """
    elem = factory()
    mesh = tessellate_solid_to_mesh(elem)
    assert mesh is not None, "a correctly-placed mesh must still come through"
    verts = np.asarray(mesh[0])
    assert verts.min(axis=0)[0] > 0, "and it must be on the positive side"


@pytest.mark.parametrize("factory", [_pipe, _bar], ids=["pipe", "bar"])
def test_reflected_tessellations_are_refused_not_returned(factory):
    """Rather than hand back a mirrored mesh, return the established
    "could not tessellate" signal.

    ``None`` is what every caller already handles —
    ``displacement._occupant_mesh`` turns it into a loud ``DisplacementError``
    and ``BimElement._query_triangles`` into a ``ValueError``. Loud failure
    over silent degradation: a carve that fails to happen is visible in the
    render, a carve in the wrong place is not.

    If this test starts FAILING, the upstream evaluation has been fixed —
    delete the guard and this file rather than loosening either.
    """
    assert tessellate_solid_to_mesh(factory()) is None


def test_the_refusal_is_logged_with_the_diagnosis(caplog):
    """A silent ``None`` would just move the mystery. The warning names the
    mesh centre, the authored centre, and the reflection signature."""
    import logging

    with caplog.at_level(logging.WARNING, logger="lite_step.ifc.generator"):
        tessellate_solid_to_mesh(_pipe())
    text = caplog.text
    assert "outside its authored extent" in text
    assert "POINT-REFLECTED THROUGH THE ORIGIN" in text
    assert "Refusing the mesh" in text


def test_a_vertex_count_check_could_not_have_caught_this():
    """Why the existing guard missed it, pinned so nobody re-derives it.

    The reflected mesh is a perfectly good mesh — right vertex count, right
    topology, right size. Only its position is wrong. Any assertion on count
    or extent-magnitude passes.
    """
    import lite_step.ifc.generator as gen

    # Reach past the guard to see what the kernel actually returned.
    real = gen._mesh_is_where_it_should_be
    gen._mesh_is_where_it_should_be = lambda elem, verts: True
    try:
        mesh = tessellate_solid_to_mesh(_pipe())
    finally:
        gen._mesh_is_where_it_should_be = real

    assert mesh is not None
    verts = np.asarray(mesh[0])
    assert len(verts) < 20_000, "a bare count assertion passes on it"
    size = verts.max(axis=0) - verts.min(axis=0)
    assert round(float(size[0])) == 2000, "so does its SIZE — 2 m of pipe"
    assert round(float(size[1])) == 200, "and its diameter"
    # ...and yet it is on the wrong side of the origin entirely.
    assert verts.max(axis=0)[0] < 0
