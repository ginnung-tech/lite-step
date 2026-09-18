"""An opening's size AND its origin are read off its joinery unless stated.

``width``/``height`` are the ROUGH OPENING — the hole in the wall — while an
opening's children are the joinery that sits in it, authored in opening-local
coordinates (X along the wall, Y through the depth, Z up). For the common case
where the joinery fills the hole, restating the size is duplication the author
can get wrong, so omitting it infers the hole from the children's IN-PLANE
extents.

The load-bearing invariant is that the hole IS the joinery's bounding box:
``infer_opening_size`` takes its SPAN and ``opening_local_origin`` takes the
corner that span starts from, both from one scan. Deriving them apart is what
let a Window whose children were authored at local x 10000..11000 carve an
ELEVEN METRE void starting back at the anchor — a hole through a 4 m wall the
window itself was nowhere near. Size alone does not fix that: a correctly
sized 1000 mm hole still left at the anchor is still a hole in the wall the
joinery never covers. Void and fill must come out of the same two numbers.

Placing joinery outside its host is NOT an error — the DSL does as the code
says, and an off-wall window is simply an off-wall window. What must not
happen is a hole where nothing was authored.

An explicit value always wins, and then the origin does NOT move: that is how
an author says "the hole is SMALLER than the joinery" — the projecting-sill
case, which is exactly where inference would be wrong.
"""
from __future__ import annotations

import os
import re
import tempfile

import pytest

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.compiler.frames import infer_opening_size, opening_local_origin
from lite_step.models import (
    Box,
    Extrude,
    Point,
    Project,
    Storey,
    Wall,
    Window,
)


def _pane(x0=0, x1=1200, z0=0, z1=1400, thickness=60):
    return Extrude(
        contour=[Point(x=x0, y=0, z=z0), Point(x=x1, y=0, z=z0),
                 Point(x=x1, y=0, z=z1), Point(x=x0, y=0, z=z1)],
        thickness=thickness,
    )


def _leaf(x0=0, x1=1000, z0=0, z1=1200, name="leaf"):
    """A box pane — its CENTRE is what the compiled child placement carries,
    which is how a test reads back where the joinery actually landed."""
    return Box(start=Point(x=x0, y=0, z=z0), end=Point(x=x1, y=90, z=z1),
               name=name)


def _hosted(opening):
    wall = Wall(name="s").add(
        Box(start=Point(x=0, y=0, z=0), end=Point(x=6000, y=300, z=2700)))
    wall.anchor(opening, along=1500, up=900)
    storey = Storey()
    storey.add(wall)
    proj = Project(name="t")
    proj.storeys = [storey]
    return proj


#: The scenario the fix exists for: a small wall, an ordinary anchor call, and
#: joinery authored 10 m from its own local origin.
SMALL_WALL_MM = 4000
ALONG_MM = 500


def _small_wall_hosting(opening):
    wall = Wall(name="small").add(
        Box(start=Point(x=0, y=0, z=0),
            end=Point(x=SMALL_WALL_MM, y=200, z=SMALL_WALL_MM), name="body"))
    wall.anchor(opening, along=ALONG_MM, up=500)
    storey = Storey()
    storey.add(wall)
    proj = Project(name="offset-joinery")
    proj.storeys = [storey]
    return proj


def _opening_of(project):
    for child in project.storeys[0].elements[0]._elements:
        if isinstance(child, Window):
            return child
    raise AssertionError("no opening found")


# ---------------------------------------------------------------------------
# Compiled-model helpers — both backends, read back through ifcopenshell so
# the cross-backend comparison is one parser reading two emitters.
# ---------------------------------------------------------------------------


def _compile(project, backend="ifcopenshell") -> str:
    pytest.importorskip("ifcopenshell")
    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(normalize_project_to_meters(project))
    assert result.success, result.error
    return result.ifc_content


def _opened(content: str):
    import ifcopenshell

    with tempfile.NamedTemporaryFile("w", suffix=".ifc", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(content)
        path = fh.name
    try:
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _world_origin(product) -> tuple:
    import ifcopenshell.util.placement as plc

    m = plc.get_local_placement(product.ObjectPlacement)
    return (float(m[0][3]), float(m[1][3]), float(m[2][3]))


def _void_span_x(content: str, width_m: float) -> tuple:
    """``(x_min, x_max)`` of the FIRST void, in world meters.

    The void's placement origin is its CENTRE along the run (and in the
    thickness), and these fixtures run their wall along world X, so the span
    is the centre plus/minus half the hole's width. ``width_m`` is handed in
    from the normalized model rather than read off ``IfcWindow.OverallWidth``
    — only the streaming backend fills that attribute in.
    """
    voids = _opened(content).by_type("IfcOpeningElement")
    assert voids, "no IfcOpeningElement in output"
    cx = _world_origin(voids[0])[0]
    return (cx - width_m / 2.0, cx + width_m / 2.0)


def _product_x(content: str, leaf: str) -> float:
    """World x of the product whose DSL v2.1 canonical name starts at ``leaf``
    (``box:leaf:window:w:wall:small``). A Box child's placement IS its centre."""
    for product in _opened(content).by_type("IfcProduct"):
        name = product.Name or ""
        if name.split(":")[1:2] == [leaf]:
            return _world_origin(product)[0]
    raise AssertionError(f"no product with leaf {leaf!r} in output")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_size_is_inferred_from_the_joinery():
    win = Window(name="w")
    win.add(_pane())
    assert infer_opening_size(win) == (1200, 1400)


def test_a_frame_authored_from_x50_gets_a_hole_from_x50():
    """THE x=50 decision, pinned: the hole is the joinery's bounding box, so
    a frame drawn from local x=50 gets a 1100 mm hole starting at 50 — not a
    1150 mm hole starting at 0.

    Measuring every span from local 0 reasons that "a frame authored from
    x=50 still needs the full hole beneath it". It does
    not: nothing is authored in that first 50 mm, so that opens a
    50 mm slot of daylight beside the frame — the same defect the 10 m case
    makes obvious, just too small to notice. The new rule leaves no hole any
    joinery does not fill, which is the property that matters at both scales.

    An author who really wants a hole wider than the joinery states it:
    ``Window(width=1150)``. That is what an explicit size is FOR, and it also
    keeps the origin at the anchor.
    """
    win = Window(name="w")
    win.add(_pane(x0=50, x1=1150, z0=100, z1=1300))
    assert infer_opening_size(win) == (1100, 1200), "hole = the joinery's span"
    assert opening_local_origin(win) == (50.0, 100.0), \
        "hole starts where the joinery starts"


def test_depth_is_ignored():
    """A frame runs through the wall and a sill projects past its face;
    neither says anything about how big the hole is."""
    win = Window(name="w")
    win.add(_pane(thickness=400))          # far deeper than the 300 mm wall
    assert infer_opening_size(win) == (1200, 1400)


def test_inferred_size_reaches_the_compiled_model():
    win = Window(name="w")
    win.add(_pane())
    proj = _hosted(win)
    assert validate_project_report(proj).errors == []
    opening = _opening_of(normalize_project_to_meters(proj))
    assert opening.width == pytest.approx(1.2)
    assert opening.height == pytest.approx(1.4)


def test_inference_does_not_mutate_the_authors_project():
    """The fill happens on the normalized COPY, so the author's source keeps
    its ``None`` and the inference stays re-derivable rather than baked in."""
    win = Window(name="w")
    win.add(_pane())
    proj = _hosted(win)
    normalize_project_to_meters(proj)
    assert win.width is None
    assert win.height is None


# ---------------------------------------------------------------------------
# Explicit wins
# ---------------------------------------------------------------------------


def test_explicit_size_beats_the_joinery():
    """THE reason inference cannot be mandatory: a projecting sill overhangs
    the hole, so the children are NOT the opening."""
    win = Window(name="w", width=1200, height=1400)
    win.add(_pane(x0=-80, x1=1280, z0=-40, z1=0, thickness=200))   # sill
    assert infer_opening_size(win) == (1200, 1400)
    opening = _opening_of(normalize_project_to_meters(_hosted(win)))
    assert opening.width == pytest.approx(1.2)


def test_an_explicit_size_pins_the_origin_to_the_anchor():
    """An explicit size states the hole outright and the anchor states where
    it goes, so nothing may slide it onto the joinery — otherwise the
    projecting-sill window above would drag its own hole 80 mm sideways and
    40 mm down, and ``width=``/``height=`` would stop meaning what they say."""
    win = Window(name="w", width=1200, height=1400)
    win.add(_pane(x0=-80, x1=1280, z0=-40, z1=0, thickness=200))   # sill
    assert opening_local_origin(win) == (0.0, 0.0)


def test_one_dimension_may_be_explicit_and_the_other_inferred():
    win = Window(name="w", width=900)
    win.add(_pane())
    assert infer_opening_size(win) == (900, 1400)


def test_a_bare_opening_still_takes_an_explicit_size():
    """No children at all — the LOD 200 / massing case — stays valid."""
    win = Window(name="w", width=1200, height=1400)
    assert infer_opening_size(win) == (1200, 1400)
    assert validate_project_report(_hosted(win)).errors == []


# ---------------------------------------------------------------------------
# Loud failures
# ---------------------------------------------------------------------------


def test_neither_size_nor_children_is_a_compile_error():
    errors = validate_project_report(_hosted(Window(name="bare"))).errors
    assert any("needs a size" in e for e in errors), errors


def test_a_degenerate_inferred_size_is_a_compile_error():
    """Children that collapse to a line infer a zero-width hole — refuse it
    rather than emit a void with no volume."""
    win = Window(name="w")
    win.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=0, y=100, z=1400)))
    errors = validate_project_report(_hosted(win)).errors
    assert any("must be positive" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Joinery authored away from the opening's own origin
#
# The wall is 4 m along world X and is the storey's only element, so the
# facing tie resolves to the MAX side: the exterior is +Y, local +x (right as
# seen from outside) runs -X, and the run starts at world x=4. An
# ``along=500`` opening 1000 wide therefore centres on 4 - (0.5 + 0.5) = 3.0.
# Measured, not assumed.
# ---------------------------------------------------------------------------


ON_WALL_VOID_X = (2.5, 3.5)
#: Joinery drawn 10 m along its own local x lands 10 m along the run: the
#: hole goes with it, entirely clear of the wall's 0..4 m span.
OFF_WALL_VOID_X = (-7.5, -6.5)


def _hole(project, backend="ifcopenshell"):
    """``(span, width_m, content)`` for a project's one opening."""
    width_m = _opening_of(normalize_project_to_meters(project)).width
    content = _compile(project, backend)
    return _void_span_x(content, width_m), width_m, content


def test_offset_joinery_infers_its_own_span_not_the_distance_to_the_anchor():
    """The bug, at the size half: joinery at local x 10000..11000 is 1000 mm
    of window, so the hole is 1000 mm. Measuring ``max - 0`` made it 11000 —
    an eleven metre void from an anchor call that read ``along=500``."""
    win = Window(name="w")
    win.add(_leaf(x0=10000, x1=11000))
    _span, width_m, _content = _hole(_small_wall_hosting(win), "ifcopenshell")
    assert width_m == pytest.approx(1.0), "the hole is the joinery's span"


def test_offset_joinery_leaves_no_hole_in_the_wall():
    """The bug, at the position half — and the half a size-only fix misses.

    A correctly sized 1000 mm hole still placed at the anchor is still a hole
    in a wall the window is 10 m away from. The origin follows the joinery,
    so the void clears the wall's 0..4 m span entirely: an off-wall window is
    allowed, a hole where nothing was authored is not.
    """
    win = Window(name="w")
    win.add(_leaf(x0=10000, x1=11000))
    span, _width_m, _content = _hole(_small_wall_hosting(win), "ifcopenshell")
    assert span == pytest.approx(OFF_WALL_VOID_X), "the hole followed the joinery"
    wall_min, wall_max = 0.0, SMALL_WALL_MM / 1000.0
    assert span[1] <= wall_min or span[0] >= wall_max, (
        f"void {span} intersects the wall [{wall_min}, {wall_max}] — the "
        f"window is 10 m away, so nothing may be carved out of the wall")


def test_the_hole_is_centred_on_the_joinery_it_was_inferred_from():
    """Void and fill come out of ONE bounding box, so they cannot disagree:
    the leaf's centre and the hole's centre are the same number."""
    win = Window(name="w")
    win.add(_leaf(x0=10000, x1=11000))
    span, _width_m, content = _hole(_small_wall_hosting(win), "ifcopenshell")
    hole_centre = (span[0] + span[1]) / 2.0
    assert _product_x(content, "leaf") == pytest.approx(hole_centre), \
        "the joinery sits in its own hole"


def test_joinery_filling_its_hole_from_local_zero_is_unchanged():
    """The ordinary case — the one the whole corpus is written in — must not
    move: origin offset zero, hole where the anchor put it."""
    win = Window(name="w")
    win.add(_leaf(x0=0, x1=1000))
    span, width_m, content = _hole(_small_wall_hosting(win), "ifcopenshell")
    assert width_m == pytest.approx(1.0)
    assert span == pytest.approx(ON_WALL_VOID_X), "the hole stayed at the anchor"
    assert _product_x(content, "leaf") == pytest.approx(3.0)


def test_offset_joinery_still_costs_zero_booleans():
    """A Window is a SEMANTIC opening wherever its joinery was drawn — the
    invariant ``test_anchor_openings`` exists for. An off-wall opening must
    not start routing through the boolean machinery to explain itself."""
    win = Window(name="w")
    win.add(_leaf(x0=10000, x1=11000))
    _span, _width_m, content = _hole(_small_wall_hosting(win), "ifcopenshell")
    assert not re.findall(r"=\s*IFCBOOLEANRESULT\(", content.upper()), \
        "an opening is never a carve"


def test_along_center_still_centres_the_hole_when_the_origin_moved():
    """``along_center`` centres the HOLE, and the hole is now the joinery's
    box — so the anchor steps back by the origin offset. Without that,
    centring would land the hole one whole offset off target."""
    win = Window(name="w")
    win.add(_leaf(x0=300, x1=1300))
    wall = Wall(name="small").add(
        Box(start=Point(x=0, y=0, z=0),
            end=Point(x=SMALL_WALL_MM, y=200, z=SMALL_WALL_MM), name="body"))
    wall.anchor(win, along_center=2000, up=500)
    storey = Storey()
    storey.add(wall)
    proj = Project(name="centred")
    proj.storeys = [storey]

    span, width_m, _content = _hole(proj, "ifcopenshell")
    assert width_m == pytest.approx(1.0)
    # Centre 2 m along a run that starts at world x=4 and points -X.
    assert (span[0] + span[1]) / 2.0 == pytest.approx(2.0)
