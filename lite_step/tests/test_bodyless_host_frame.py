"""A body-less wall's opening frame is DERIVED, and every consumer must read it.

A ``Wall`` that only aggregates leaf walls has no body of its own; its opening
frame comes from the bounds of what it aggregates (``frames.opening_container_aabb``)
and its distributed window compiles correctly. Two consumers still read
``frames._body_of`` and so saw "no frame": ``--q``/``--check`` claimed
"compiling would reject the same model", and the space-boundary pass warned
"bears no opening frame" and dropped the window's boundary (6 of 7).
``frames.host_opening_frame`` is the one host-frame answer both now share.
"""
from __future__ import annotations

import logging

import pytest

from lite_step.compiler import frames
from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.space_boundaries import collect_boundaries
from lite_step.models import Box, Point, Project, Space, Storey, Wall, Window

pytest.importorskip("ifcopenshell")

# Off-origin, non-square room, walls 300 thick on every side.
_X0, _Y0, _X1, _Y1 = 1700, -3100, 5900, 1300
_H = 2700


def _bodied(storey, name, x0, y0, x1, y1):
    w = Wall(name=name)
    w.add(Box(start=Point(x=x0, y=y0, z=0), end=Point(x=x1, y=y1, z=_H)))
    storey.add(w)
    return w


def _leaf(name, x0, y0, x1, y1):
    leaf = Wall(name=name)
    leaf.add(Box(start=Point(x=x0, y=y0, z=0), end=Point(x=x1, y=y1, z=_H)))
    return leaf


def _room(*, north_bodyless: bool):
    """Four walls round a room; the north one is either ONE bodied wall or a
    body-less aggregator of an inner and an outer leaf. Either way it carries
    the room's one window, placed by its host."""
    proj = Project(name="host-frame")
    storey = Storey(name="ground", elevation=0)
    proj.add_storey(storey)
    _bodied(storey, "south", _X0 - 300, _Y0 - 300, _X1 + 300, _Y0)
    _bodied(storey, "west", _X0 - 300, _Y0, _X0, _Y1)
    _bodied(storey, "east", _X1, _Y0, _X1 + 300, _Y1)
    if north_bodyless:
        north = Wall(name="north")
        north.add(_leaf("inner", _X0 - 300, _Y1, _X1 + 300, _Y1 + 100))
        north.add(_leaf("outer", _X0 - 300, _Y1 + 100, _X1 + 300, _Y1 + 300))
        storey.add(north)
    else:
        north = _bodied(storey, "north", _X0 - 300, _Y1, _X1 + 300, _Y1 + 300)
    north.anchor(Window(name="pane", width=1200, height=1500), along=1500, up=900)
    room = Space(name="room")
    room.add(Box(start=Point(x=_X0, y=_Y0, z=0), end=Point(x=_X1, y=_Y1, z=_H)))
    storey.add(room)
    return proj, north


def _boundaries(proj):
    return collect_boundaries(normalize_project_to_meters(proj))


def test_bodyless_wall_window_keeps_its_space_boundary(caplog):
    """The issue's repro: every boundary is emitted, the window's included."""
    control, _ = _room(north_bodyless=False)
    proj, _ = _room(north_bodyless=True)
    want = _boundaries(control)
    with caplog.at_level(logging.WARNING):
        got = _boundaries(proj)
    # Same boundary set as the bodied control: 4 walls + the window, 5 of 5.
    assert [b.element for b in got] == [b.element for b in want]
    assert len(got) == 5
    assert len([b for b in got if b.element.startswith("window:")]) == 1
    assert "bears no opening frame" not in caplog.text


def test_query_inside_a_bodyless_walls_window_answers():
    """``--q`` / ``--check``: the pane's world box, not "no usable host frame"."""
    proj, north = _room(north_bodyless=True)
    win = next(c for c in north._elements if c.name == "pane")
    win.add(Box(name="glass", start=Point(x=0, y=0, z=0),
                end=Point(x=1200, y=40, z=1500)))
    box = win._elements[0].world_aabb()
    # 1200 x 1500 pane, inside the aggregated wall bounds.
    agg = north.world_aabb()
    assert agg.min.x <= box.min.x and box.max.x <= agg.max.x
    assert agg.min.z <= box.min.z and box.max.z <= agg.max.z
    assert box.max.z - box.min.z == pytest.approx(1500)


def test_host_frame_helper_routes():
    bodied, _ = _room(north_bodyless=False)
    bodyless, north = _room(north_bodyless=True)
    # Bodied container -> its body's frame; body-less -> the aggregated bounds.
    bodied_north = next(e for e in bodied.storeys[0].elements
                        if getattr(e, "name", None) == "north")
    assert frames.host_opening_frame(bodied_north) is not None
    assert frames.host_opening_frame(north) is not None
    assert frames._body_of(north) is None         # the old reading: no frame
    # A host with nothing to derive a frame from still answers None (loud path).
    assert frames.host_opening_frame(Wall(name="empty")) is None
