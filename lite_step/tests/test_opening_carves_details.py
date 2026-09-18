"""An opening cuts through the details standing in it.

Framing and reinforcement inside a wall have to lose their material where a
window goes. They could not, and the reason was a single deliberate policy:
``displacement._collect`` exempts every ``.anchor()``ed child, because an
anchored child's position is a RELATIONSHIP to its host and entering that host
is what anchoring MEANS. That is right for siblings — a corbel must not eat the
wall it hangs off — and it is what blocked this, because a ``Wall`` refuses a
second unanchored prism ("a Wall takes exactly one body element"), so every
stirrup, tie and bar in a wall reaches the tree through ``.anchor()`` and
through nothing else.

The exception, and its exact width: **an opening is not a sibling.** It is a
hole THROUGH the host, declared on the host, and matter inside a hole is not
matter. So an opening carves the host's details; sibling-to-sibling anchoring
stays exempt, and this suite pins BOTH halves — a change that widened the
exception into "anchored children carve each other" would pass half of it.

Pinned here:

1.  A detail standing in the opening is carved, MEASURED as volume: the notch
    is exactly the intersection, not a bounding box and not the whole detail.
2.  A detail clear of the opening is untouched, and so is the wall body (the
    host product already carries the ``IfcRelVoidsElement`` — carving it again
    would be one redundant boolean per opening, paid for in CSG depth).
3.  The opening's own joinery is never carved. It lives in the hole.
4.  ``.no_carve()`` exempts a detail meant to stand in the reveal.
5.  Sibling-to-sibling anchoring is STILL exempt.
6.  ``.opening()`` on a container emits the window rather than parking the
    child in ``_openings``, which a container's opening pass never reads — no
    ``IfcOpeningElement``, no ``IfcWindow``, no warning.
7.  A skew host refuses OUT LOUD rather than carving a bounding box.
"""

from __future__ import annotations

import logging
import math

import pytest

from lite_step.compiler.displacement import carve_pairs
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.models import Box, Point, Project, Wall, Window

# ---------------------------------------------------------------------------
# One 6 m wall running +X, 300 mm thick, y in [-300, 0], z in [0, 2700].
#
# It is the storey's only element, so the exterior is +Y (``_facing_for``'s tie
# resolves to the MAX side) and local +x runs -X — which is why ``along=1500``
# on a 1200-wide window puts the hole at x in [3.3, 4.5]. Measured, not
# assumed; ``test_the_fixture_puts_the_window_where_this_suite_says`` pins it,
# so a frame change surfaces here as one failure rather than as eight.
# ---------------------------------------------------------------------------

_WIN_X0, _WIN_X1 = 3.3, 4.5
_WIN_Z0, _WIN_Z1 = 0.9, 2.3          # up=900, height=1400
_POST = 0.05                          # a 50 x 50 post, full storey height
_POST_H = 2.7


def _post(name: str) -> Box:
    return Box(start=Point(x=0, y=100, z=0),
               end=Point(x=50, y=150, z=2700), name=name)


def _wall(*, verb: str = "anchor", no_carve: bool = False,
          posts=((2100, "post_in"), (200, "post_out"))) -> Wall:
    wall = Wall(name="south")
    wall.add(Box(start=Point(x=0, y=-300, z=0),
                 end=Point(x=6000, y=0, z=2700), type="wall", name="body"))
    for along, name in posts:
        p = _post(name)
        exempt = no_carve and name == "post_in"
        wall.anchor(p, along=along, up=0,
                    carve="none" if exempt else None)
    win = Window(width=1200, height=1400, name="w0")
    getattr(wall, verb)(win, along=1500, up=900)
    return wall


def _project(wall: Wall) -> Project:
    proj = Project(name="opening-carves-details")
    proj.add(wall)
    assert validate_project_report(proj).errors == []
    return normalize_project_to_meters(proj) or proj


def _compiled(proj: Project) -> str:
    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(proj)
    assert result.success, result.error
    return result.ifc_content


def _child(wall_or_proj, name: str):
    wall = (wall_or_proj.storeys[0].elements[0]
            if isinstance(wall_or_proj, Project) else wall_or_proj)
    for child in wall._elements:
        if getattr(child, "name", None) == name:
            return child
    raise AssertionError(f"no child named {name!r}")


def _cuts(proj: Project, name: str) -> int:
    return len(getattr(_child(proj, name), "_cuts", None) or [])


def _volume(ifc_text: str, product_name: str) -> float:
    """Tessellated world volume of one named product's OWN representation.

    Volume rather than a vertex scan: a boolean that carved the WRONG box
    still produces vertices at plausible places, and only the number says by
    how much.

    ``disable-opening-subtractions`` is load-bearing, and finding out why is
    what this helper is for. ``ifcopenshell.geom`` applies an
    ``IfcRelVoidsElement`` on a parent to the products AGGREGATED under it, so
    a post inside this wall measures 0.00325 whether or not it carries a
    boolean of its own — the measurement could not fail. Turning the
    subtraction off leaves only what is in the element's own representation,
    which is also exactly what web-ifc / ThatOpen renders (our production
    path) and what any consumer reading one product in isolation sees. That
    difference is the whole reason this feature exists.
    """
    import tempfile
    import os

    geom = pytest.importorskip("ifcopenshell.geom")
    import ifcopenshell
    import ifcopenshell.util.shape

    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ifc_text)
        model = ifcopenshell.open(path)
        settings = geom.settings()
        settings.set("use-world-coords", True)
        settings.set("disable-opening-subtractions", True)
        matches = [p for p in model.by_type("IfcProduct")
                   if (p.Name or "").startswith(product_name)
                   and p.Representation is not None]
        assert matches, f"no product named {product_name!r} with geometry"
        assert len(matches) == 1, [p.Name for p in matches]
        shape = geom.create_shape(settings, matches[0])
        return ifcopenshell.util.shape.get_volume(shape.geometry)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 0. The fixture itself
# ---------------------------------------------------------------------------


def test_the_fixture_puts_the_window_where_this_suite_says() -> None:
    """Every expectation below is read off these four numbers."""
    from lite_step.compiler import frames
    from lite_step.ifc.entity_cache import snap_to_precision

    proj = _project(_wall())
    from lite_step.ifc.generator import generate_ifc
    generate_ifc(proj)                       # runs the frame + anchor passes

    wall = proj.storeys[0].elements[0]
    frame = frames.opening_host_frame(frames._body_of(wall),
                                      snap=snap_to_precision, divisor=1000.0)
    prism = frames.opening_void_prism(_child(wall, "w0"), frame)
    (x0, _y0, z0), (x1, _y1, z1) = prism.aabb()
    assert (round(x0, 6), round(x1, 6)) == (_WIN_X0, _WIN_X1)
    assert (round(z0, 6), round(z1, 6)) == (_WIN_Z0, _WIN_Z1)
    assert prism.is_axis_aligned(1e-9)


# ---------------------------------------------------------------------------
# 1-2. The carve, and its exact width
# ---------------------------------------------------------------------------


def test_a_detail_standing_in_the_opening_loses_exactly_the_hole() -> None:
    proj = _project(_wall())
    ifc = _compiled(proj)

    assert _cuts(proj, "post_in") == 1
    full = _POST * _POST * _POST_H
    notch = _POST * _POST * (_WIN_Z1 - _WIN_Z0)
    assert _volume(ifc, "box:post_in") == pytest.approx(full - notch, rel=1e-6)


def test_a_detail_clear_of_the_opening_is_untouched() -> None:
    proj = _project(_wall())
    ifc = _compiled(proj)

    assert _cuts(proj, "post_out") == 0
    assert _volume(ifc, "box:post_out") == pytest.approx(
        _POST * _POST * _POST_H, rel=1e-6)


def test_the_wall_body_is_not_carved_twice() -> None:
    """The host product already carries the ``IfcRelVoidsElement``.

    A second, boolean carve of the same volume would be invisible in the
    render and visible only in the CSG depth budget — the resource this
    codebase rations.
    """
    proj = _project(_wall())
    ifc = _compiled(proj)

    assert _cuts(proj, "body") == 0
    assert ifc.count("IFCOPENINGELEMENT(") == 1
    # Exactly one boolean in the whole model: the notch in post_in.
    assert ifc.count("IFCBOOLEANRESULT(") == 1


def test_the_openings_own_joinery_is_never_carved() -> None:
    """A pane inside the window lives in the hole by definition."""
    wall = _wall()
    win = _child(wall, "w0")
    win.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=1200, y=50, z=1400),
                name="pane"))
    proj = _project(wall)
    _compiled(proj)

    pane = next(c for c in _child(proj, "w0")._elements
                if getattr(c, "name", None) == "pane")
    assert (getattr(pane, "_cuts", None) or []) == []


# ---------------------------------------------------------------------------
# 3-4. The width of the exception
# ---------------------------------------------------------------------------


def test_no_carve_exempts_a_detail_meant_to_stand_in_the_reveal() -> None:
    proj = _project(_wall(no_carve=True))
    ifc = _compiled(proj)

    assert _cuts(proj, "post_in") == 0
    assert _volume(ifc, "box:post_in") == pytest.approx(
        _POST * _POST * _POST_H, rel=1e-6)


def test_sibling_to_sibling_anchoring_is_still_exempt() -> None:
    """The invariant this change does NOT touch.

    Two anchored details overlapping each other carve nothing — that is
    ``_collect``'s exemption, and widening the opening exception into "anchored
    children carve each other" would silently delete a corbel's host.
    """
    wall = Wall(name="south")
    wall.add(Box(start=Point(x=0, y=-300, z=0),
                 end=Point(x=6000, y=0, z=2700), type="wall", name="body"))
    # Same anchor point, so they interpenetrate completely.
    wall.anchor(_post("first"), along=2100, up=0)
    wall.anchor(_post("second"), along=2100, up=0)
    proj = _project(wall)
    _compiled(proj)

    assert _cuts(proj, "first") == 0
    assert _cuts(proj, "second") == 0


# ---------------------------------------------------------------------------
# 5. ``.opening()`` on a container
# ---------------------------------------------------------------------------


def test_the_opening_verb_on_a_container_emits_the_window() -> None:
    """Emitting nothing at all — no void, no fill, no warning — is the failure.

    A container reads its openings from ``_elements``; ``.opening()`` appended
    to ``_openings``. The doorway that silently is not there.
    """
    ifc = _compiled(_project(_wall(verb="opening")))
    assert ifc.count("IFCWINDOW(") == 1
    assert ifc.count("IFCOPENINGELEMENT(") == 1


def test_the_two_verbs_compile_to_the_same_model() -> None:
    """Byte-identical once GlobalIds (which are random per run) are erased."""
    anchored = _without_guids(_compiled(_project(_wall(verb="anchor"))))
    opened = _without_guids(_compiled(_project(_wall(verb="opening"))))
    assert anchored == opened


def _without_guids(ifc_text: str) -> str:
    """Erase everything that varies between two runs of the SAME input.

    Three things do, and none of them is geometry:

    * **GlobalIds** — random per run, by design.
    * **``IFCUNITASSIGNMENT`` member order** — ifcopenshell emits the set in
      iteration order, which varies run to run.
    * **The ``FILE_NAME`` header timestamp** — whole seconds. The two models
      here are compiled one after the other, so whenever that pair straddles a
      second boundary the headers differ and the assertion fails on a clock
      tick. Measured at roughly one run in ten, and it failed CI on main that
      way (run 31081407880) with the two models byte-identical everywhere else.

    A renumber-invariant comparison confirmed the diagnosis: masking ``#N`` and
    diffing the multiset of entity bodies gives zero difference in both
    directions on a run where the raw text differs.
    """
    import re

    text = re.sub(r"'[0-9A-Za-z_$]{22}'", "'<guid>'", ifc_text)
    text = re.sub(r"(FILE_NAME\(''\,)'[^']*'", r"\1'<timestamp>'", text)
    return re.sub(
        r"IFCUNITASSIGNMENT\(\(([^)]*)\)\)",
        lambda m: "IFCUNITASSIGNMENT((%s))" % ",".join(sorted(m.group(1).split(","))),
        text)


# ---------------------------------------------------------------------------
# 6. Skew
# ---------------------------------------------------------------------------


def _skew_body(degrees: float):
    from lite_step.models import Extrude

    rad = math.radians(degrees)
    dx, dy = round(6000 * math.cos(rad)), round(6000 * math.sin(rad))
    return Extrude(
        contour=[Point(x=0, y=0, z=0), Point(x=dx, y=dy, z=0),
                 Point(x=dx, y=dy, z=2700), Point(x=0, y=0, z=2700)],
        thickness=300, type="wall", name="body")


def test_a_skew_void_is_not_its_own_bounding_box() -> None:
    """The decision, at the level it is made.

    A ``Box`` operand around this prism would carve its bounding box — 1.19 m
    wide against a 1.2 m window rotated 30 degrees, and 0.86 m through a
    0.3 m wall. That is a hole nobody authored, in the one direction nobody
    inspects, so there is no operand and the caller warns.
    """
    from lite_step.compiler import frames
    from lite_step.compiler.displacement import _void_operand
    from lite_step.ifc.entity_cache import snap_to_precision

    wall = Wall(name="skew")
    wall.add(_skew_body(30.0))
    wall.anchor(Window(width=1200, height=1400, name="w0"), along=1500, up=900)
    proj = _project(wall)
    _compiled(proj)

    frame = frames.opening_host_frame(frames._body_of(_child(proj, "body").
                                                      _parent or wall),
                                      snap=snap_to_precision, divisor=1000.0)
    prism = frames.opening_void_prism(_child(proj, "w0"), frame)
    assert prism is not None
    assert not prism.is_axis_aligned(1e-9)
    assert _void_operand(prism) is None


def test_a_skew_host_warns_instead_of_carving_a_bounding_box(caplog) -> None:
    """And says so, naming both the opening and the detail it left alone.

    The detail is ``.add()``ed at world coordinates known to sit inside the
    skew void — a guessed coordinate simply misses the void and proves
    nothing — and it arrives inside a nested container because a ``Wall``
    takes exactly one unanchored prism.

    **This fixture hosts on a ``Wall``, not ``Element(ifc_class="IfcWall")``**,
    whose
    docstring conceded "that host does not emit its Window at all — a
    separate, already-warned-about gap". The gap was closed by refusing
    that host, so the fixture moved to a ``Wall``, which emits. The skew void
    lands on the same world AABB either way — ~(2.86, 1.39) .. (4.05, 2.25),
    measured — so this still tests the displacement pass's refusal and nothing
    else. The ``IFCWINDOW`` assertion below is what keeps it from drifting
    back onto a host where the warning would be about a hole that is not in
    the file.
    """
    from lite_step.models import Element

    host = Wall(name="skew")
    host.add(_skew_body(30.0))
    cage = Element(ifc_class="IfcBuildingElementProxy", name="cage")
    cage.add(Box(start=Point(x=3400, y=1750, z=0),
                 end=Point(x=3500, y=1850, z=2700), name="post_in"))
    host.add(cage)
    host.anchor(Window(width=1200, height=1400, name="w0"), along=1500, up=900)
    proj = _project(host)

    with caplog.at_level(logging.WARNING,
                         logger="lite_step.compiler.displacement"):
        ifc = _compiled(proj)

    # The hole this warning is about is really in the file.
    assert ifc.count("IFCWINDOW(") == 1
    assert ifc.count("IFCOPENINGELEMENT(") == 1

    messages = [r.getMessage() for r in caplog.records]
    skew = [m for m in messages if "not axis-aligned" in m]
    assert skew, messages
    assert "w0" in skew[0] and "post_in" in skew[0]
    # It warned INSTEAD of carving — no bounding-box operand landed.
    assert not any(m for m in messages if "opening-void" in m)


# ---------------------------------------------------------------------------
# 7. The record
# ---------------------------------------------------------------------------


def test_the_carve_report_names_the_opening_that_did_it() -> None:
    proj = _project(_wall())
    _compiled(proj)
    assert ("box:post_in:wall:south", "window:w0:wall:south",
            "opening-void") in carve_pairs(proj)
