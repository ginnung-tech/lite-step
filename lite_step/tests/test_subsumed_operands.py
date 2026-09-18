"""An operand wholly inside another operand's solid is dropped.

``H - A - B == H - A`` whenever ``B`` is inside ``A``. `loadbearing-structure`
puts 171 bar operands on the terrain mesh and every one sits its cover
distance inside a slab or wall that carves the same terrain — 171 exact
no-ops, each a full manifold3d subtraction against a terrain that can run to
hundreds of thousands of triangles.

The half that matters more is the REFUSAL. Soundness rests on the container's
AABB *being* the container, so only an axis-aligned `Box` with nothing removed
from it qualifies. Anything else over-approximates, and dropping against an
over-approximation deletes a real carve.
"""
from __future__ import annotations

import pytest

from lite_step.compiler.displacement import (
    _contains_aabb,
    _drop_subsumed_operands,
    _exact_box_bounds,
)
from lite_step.models import Bar, Box, Extrude, Point


def _box(x0, y0, z0, x1, y1, z1, name=None):
    return Box(start=Point(x=x0, y=y0, z=z0), end=Point(x=x1, y=y1, z=z1),
               name=name)


def _bar_inside():
    """A D12 bar 50 mm inside the wall box below — real cover geometry."""
    return Bar(path=[Point(x=100, y=90, z=100), Point(x=1900, y=90, z=100)],
               diameter=12, grade="B500B", name="bar_in")


_WALL = lambda: _box(0, 0, 0, 2000, 180, 3000, name="body")   # noqa: E731


def test_a_bar_inside_a_wall_box_is_dropped() -> None:
    wall, bar = _WALL(), _bar_inside()
    kept, dropped = _drop_subsumed_operands([wall, bar])
    assert kept == [wall]
    assert [o for o, _c in dropped] == [bar]
    assert dropped[0][1] is wall


def test_a_bar_outside_the_box_is_kept() -> None:
    """The exposed case — a bar standing in the ground with nothing around it.

    A type rule ("a Bar does not carve a Mesh") would drop this one too.
    """
    wall = _WALL()
    exposed = Bar(path=[Point(x=100, y=5000, z=100), Point(x=1900, y=5000, z=100)],
                  diameter=12, grade="B500B", name="bar_out")
    kept, dropped = _drop_subsumed_operands([wall, exposed])
    assert set(map(id, kept)) == {id(wall), id(exposed)}
    assert dropped == []


def test_a_bar_crossing_the_box_face_is_kept() -> None:
    """Partly inside is not inside. It carves the part that sticks out."""
    wall = _WALL()
    through = Bar(path=[Point(x=100, y=90, z=100), Point(x=4000, y=90, z=100)],
                  diameter=12, grade="B500B", name="bar_through")
    kept, _dropped = _drop_subsumed_operands([wall, through])
    assert id(through) in set(map(id, kept))


# ---------------------------------------------------------------------------
# The refusals. Each container below LOOKS like it encloses the bar and does
# not, so treating its AABB as its solid would delete a real carve.
# ---------------------------------------------------------------------------


def test_a_box_with_a_hole_in_it_is_not_a_container() -> None:
    """The bar could be sitting in the hole."""
    wall = _WALL()
    wall.difference(_box(50, 40, 50, 1950, 140, 2950))
    assert _exact_box_bounds(wall) is None
    kept, dropped = _drop_subsumed_operands([wall, _bar_inside()])
    assert dropped == []
    assert len(kept) == 2


def test_a_clipped_box_is_not_a_container() -> None:
    """`_carve_mesh_host` trims an operand by its own half-spaces, so the kept
    volume is smaller than the box."""
    wall = _WALL()
    wall._clips = [object()]          # shape irrelevant; presence disqualifies
    assert _exact_box_bounds(wall) is None
    _kept, dropped = _drop_subsumed_operands([wall, _bar_inside()])
    assert dropped == []


def test_a_contour_extrude_is_not_a_container() -> None:
    """A contour's AABB is a strict over-approximation of its solid — the
    triangle below leaves most of its bounding box empty."""
    tri = Extrude(contour=[Point(x=0, y=0, z=0), Point(x=2000, y=0, z=0),
                           Point(x=0, y=0, z=3000)], thickness=180, name="gable")
    assert _exact_box_bounds(tri) is None
    _kept, dropped = _drop_subsumed_operands([tri, _bar_inside()])
    assert dropped == []


def test_a_placed_box_is_not_a_container() -> None:
    """Pre-placement coordinates: the box is not where the geometry stands."""
    from lite_step.models.elements import Transform

    wall = _WALL()
    wall.placement = Transform(origin=Point(x=9000, y=0, z=0))
    assert _exact_box_bounds(wall) is None


def test_containment_is_not_mere_overlap() -> None:
    outer = ((0.0, 0.0, 0.0), (10.0, 10.0, 10.0))
    assert _contains_aabb(outer, ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)))
    assert not _contains_aabb(outer, ((-1.0, 1.0, 1.0), (2.0, 2.0, 2.0)))
    assert not _contains_aabb(outer, ((1.0, 1.0, 1.0), (11.0, 2.0, 2.0)))


def test_a_single_operand_is_never_dropped() -> None:
    bar = _bar_inside()
    kept, dropped = _drop_subsumed_operands([bar])
    assert kept == [bar] and dropped == []
