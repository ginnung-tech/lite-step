"""``along_center=`` positions a child by its CENTRE, not its near edge.

Centring an opening in a bay is the common case, and it was costing every
author the same ``along = centre - width // 2`` they could get wrong. This is
a distinct parameter rather than an ``attach_to="center"`` mode flag: a mode
would make ``along`` mean two different things depending on a second
argument, which is the pattern this DSL keeps deleting (``offset_across``'s
context-dependent sign, ``offset=`` on the wrong object). The NAME says which
reference point it uses; nothing is contextual.
"""
from __future__ import annotations

import pytest

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.compiler.frames import resolve_along, stamp_frames
from lite_step.models import (
    Box,
    Door,
    Element,
    Point,
    Project,
    Storey,
    Wall,
    Window,
)

try:
    import ifcopenshell
    import ifcopenshell.util.placement as plc
    HAVE_IFC = True
except ImportError:                                   # pragma: no cover
    HAVE_IFC = False


def _wall():
    """12 m wall from x=-6000 to x=+6000 — its centre is along=6000."""
    return Wall(name="s").add(
        Box(start=Point(x=-6000, y=-4500, z=0), end=Point(x=6000, y=-4200, z=2700)))


def _proj(*elements):
    storey = Storey()
    storey.add(*elements)
    proj = Project(name="t")
    proj.storeys = [storey]
    return proj


def _world_x(project, name):
    from lite_step.ifc.generator import generate_ifc
    import tempfile
    import os

    content = generate_ifc(normalize_project_to_meters(project)).ifc_content
    fh = tempfile.NamedTemporaryFile("w", suffix=".ifc", delete=False)
    fh.write(content)
    fh.close()
    try:
        model = ifcopenshell.open(fh.name)
        for prod in model.by_type("IfcProduct"):
            nm = prod.Name or ""
            if name in nm and "_void" not in nm and prod.ObjectPlacement:
                return plc.get_local_placement(prod.ObjectPlacement)[0, 3]
    finally:
        os.unlink(fh.name)
    raise AssertionError(f"no product matching {name!r}")


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------


def test_resolver_subtracts_half_the_opening_width():
    wall = _wall()
    door = Door(name="d", width=900, height=2100)
    wall.anchor(door, along_center=6000)
    assert resolve_along(door._anchor_spec, door) == pytest.approx(6000 - 450)


def test_resolver_uses_an_inferred_width_when_the_size_is_omitted():
    """The opening's size may itself be inferred from its children; centring
    has to see the same number the hole is cut with."""
    wall = _wall()
    win = Window(name="w")
    win.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=1200, y=60, z=1400)))
    wall.anchor(win, along_center=6000)
    assert resolve_along(win._anchor_spec, win) == pytest.approx(6000 - 600)


def test_resolver_measures_a_plain_solid_by_its_own_extent():
    """A non-opening child has no ``width`` — its span along the run axis is
    what gets centred, measured in the child's OWN coordinates (the host's
    world frame is not an input; see test_along_center_run_directions.py)."""
    host = Element(name="pylon", ifc_class="IfcBuildingElementProxy")
    host.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=4000, y=400, z=3000)))
    bracket = Box(name="bracket", start=Point(x=0, y=0, z=0),
                  end=Point(x=300, y=100, z=200))
    host.anchor(bracket, along_center=2000)
    stamp_frames(_proj(host))
    assert resolve_along(bracket._anchor_spec, bracket) == pytest.approx(2000 - 150)


def test_plain_along_is_returned_untouched():
    """The common path costs nothing — no width lookup, no arithmetic."""
    wall = _wall()
    door = Door(name="d", width=900, height=2100)
    wall.anchor(door, along=1500)
    assert resolve_along(door._anchor_spec, door) == 1500


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_IFC, reason="ifcopenshell not installed")
def test_a_centred_door_lands_on_the_wall_centre():
    """The whole point: no ``- width // 2`` in the authoring code."""
    wall = _wall()
    wall.anchor(Door(name="centred", width=900, height=2100), along_center=6000)
    assert _world_x(_proj(wall), "centred") == pytest.approx(0.0, abs=1e-6)


@pytest.mark.skipif(not HAVE_IFC, reason="ifcopenshell not installed")
def test_along_and_along_center_agree_when_the_arithmetic_is_done_by_hand():
    """``along_center=C`` must equal ``along=C - width/2`` exactly — it is
    sugar over the same placement, not a second code path."""
    auto = _wall()
    auto.anchor(Door(name="auto", width=900, height=2100), along_center=4000)
    manual = _wall()
    manual.anchor(Door(name="manual", width=900, height=2100), along=4000 - 450)
    assert _world_x(_proj(auto), "auto") == pytest.approx(
        _world_x(_proj(manual), "manual"), abs=1e-9)


# ---------------------------------------------------------------------------
# Loud failures
# ---------------------------------------------------------------------------


def test_along_and_along_center_together_are_refused():
    """Two ways to say the same thing; honouring both would need a rule about
    which wins, and a rule like that is what a mode flag smuggles in."""
    with pytest.raises(ValueError, match="not both"):
        _wall().anchor(Door(name="d", width=900, height=2100),
                       along=1000, along_center=6000)


def test_strict_int_mm_applies_to_along_center():
    with pytest.raises(Exception):
        _wall().anchor(Door(name="d", width=900, height=2100),
                       along_center=6000.5)


def test_along_center_survives_normalization():
    wall = _wall()
    door = Door(name="d", width=900, height=2100)
    wall.anchor(door, along_center=6000)
    assert validate_project_report(_proj(wall)).errors == []
    normalized = normalize_project_to_meters(_proj(wall))
    opening = [c for c in normalized.storeys[0].elements[0]._elements
               if isinstance(c, Door)][0]
    assert opening._anchor_spec.along_center == pytest.approx(6.0)
