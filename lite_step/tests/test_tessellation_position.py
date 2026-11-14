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

ifcopenshell 0.9 fixed the evaluation: Pipe and Bar now come back where they
were authored. The guard is kept and tested directly on a reflected mesh.
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


@pytest.mark.parametrize("factory", [_sweep, _revolve, _pipe, _bar],
                         ids=["sweep", "revolve", "pipe", "bar"])
def test_correct_tessellations_are_accepted(factory):
    """The guard must not cost us the shapes that DO work.

    A position check that refuses everything is not a guard, it is an outage.
    """
    elem = factory()
    mesh = tessellate_solid_to_mesh(elem)
    assert mesh is not None, "a correctly-placed mesh must still come through"
    verts = np.asarray(mesh[0])
    assert verts.min(axis=0)[0] > 0, "and it must be on the positive side"


# ifcopenshell 0.9 (now the floor in requirements.txt) evaluates Pipe and Bar
# in the right place, so the kernel no longer produces the reflected mesh this
# guard was written for — Pipe and Bar are in the accepted set above. The
# guard stays: a position check costs nothing and is the only thing between a
# misplaced tessellation and a carve in the wrong place. It is now exercised
# directly, on the reflection the 0.8 kernel used to produce.


def _reflected(verts):
    return -np.asarray(verts)


@pytest.mark.parametrize("factory", [_pipe, _bar], ids=["pipe", "bar"])
def test_a_reflected_mesh_is_refused(factory):
    import lite_step.ifc.generator as gen

    elem = factory()
    verts = np.asarray(tessellate_solid_to_mesh(elem)[0])
    assert gen._mesh_is_where_it_should_be(elem, verts), "fixture: the real mesh passes"
    assert not gen._mesh_is_where_it_should_be(elem, _reflected(verts))


def test_the_refusal_is_logged_with_the_diagnosis(caplog):
    """A silent refusal would just move the mystery. The warning names the
    mesh centre, the authored centre, and the reflection signature."""
    import logging

    import lite_step.ifc.generator as gen

    elem = _pipe()
    verts = np.asarray(tessellate_solid_to_mesh(elem)[0])
    with caplog.at_level(logging.WARNING, logger="lite_step.ifc.generator"):
        gen._mesh_is_where_it_should_be(elem, _reflected(verts))
    text = caplog.text
    assert "outside its authored extent" in text
    assert "POINT-REFLECTED THROUGH THE ORIGIN" in text
    assert "Refusing the mesh" in text


def test_a_vertex_count_check_could_not_have_caught_a_reflection():
    """Why the old count-only guard missed it, pinned so nobody re-derives it:
    a reflected mesh has the right vertex count and the right size. Only its
    position is wrong."""
    verts = np.asarray(tessellate_solid_to_mesh(_pipe())[0])
    flipped = _reflected(verts)
    assert len(flipped) == len(verts) < 20_000
    size = flipped.max(axis=0) - flipped.min(axis=0)
    assert round(float(size[0])) == 2000, "2 m of pipe"
    assert round(float(size[1])) == 200, "and its diameter"
    assert flipped.max(axis=0)[0] < 0, "on the wrong side of the origin"
