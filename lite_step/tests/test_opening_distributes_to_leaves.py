"""``.opening()`` on a representation-less container cuts its LEAVES.

``Column``, ``Beam`` and ``Element`` aggregate geometry and carry none of their
own. An ``IfcRelVoidsElement`` on such a product carves nothing in a renderer
that builds each element's mesh from its OWN representation — web-ifc /
ThatOpen, our production path — so without it the hole has nowhere to go:
**0 IfcOpeningElement, 0 IfcWindow** on all three, which turns that
silence into a compile refusal. This is the capability that refusal was holding
the place for.

The mechanism is not new. ``generator._process_container_voids`` has handed a
container's ``.void()`` down to the leaves it OVERLAPS since, through the
predicates in ``lite_step.ifc.voids``; ``_process_container_openings`` is the
same walk with the frame resolved first, because an ``.opening()`` carries
host-frame scalars where a ``.void()`` operand is already world geometry.

N openings, ONE fill — and the schema is why
--------------------------------------------

Measured against IFC4X3_ADD2 (pinned below in
``test_the_schema_bound_that_forces_one_fill_per_hole``)::

    IfcElement.FillsVoids          bound = 0 .. 1
    IfcElement.HasOpenings         bound = 0 .. *
    IfcOpeningElement.HasFillings  bound = 0 .. *

``FillsVoids`` is [0:1]: an element fills AT MOST ONE opening. So one authored
hole crossing N leaves is N ``IfcOpeningElement``s and exactly one
``IfcWindow``. The schema-clean alternative — one ``IfcWindow`` per leaf —
**double-counts every window in every schedule**, which is the objection that
blocked spelling a layered wall's core with ``.opening()`` in the first place
. The other N−1 leaves carry unfilled holes, which is also physically
honest: a window sits in ONE layer of a buildup and the rest simply have holes
through them.

Which leaf carries the fill is the OUTERMOST one, measured along the host
frame's outward normal — see ``generator._outermost_host_index`` for the three
reasons. Every fixture here adds its leaves INNER-FIRST, so "the outermost" and
"the first one" are different answers and the tests can tell them apart.

The trap this file is written around
------------------------------------

``along=`` is measured in the HOST's frame, whose ``+x`` routinely opposes world
``+x``. On a 6 m host, ``along=4200 width=1200`` lands at world ``x[0.6, 1.8]``,
NOT ``x[4.2, 5.4]``. Both were filed on that confound and both
closed invalid. Every ``.void()`` comparison here derives its box from the
opening's own prism rather than restating the scalars — the discipline
``test_opening_reach_matches_void.py`` established, and which that file's
container section now extends onto these hosts.

Out of scope, and still refused: ``Slab`` and ``Roof``. ``along``/``inset``/
``up`` assume an outer face and a run axis, which a HORIZONTAL host does not
have — a different question from where the hole lands, and stays open for
it. ``test_a_horizontal_container_still_refuses`` is the guard.

Coverage
--------

1.  The relation counts BY NAME — N ``IfcRelVoidsElement``, one
    ``IfcRelFillsElement``, one ``IfcWindow``. A product count passes on
    several wrong implementations.
2.  The carved SOLID on every leaf, and the full solid on the leaf the hole
    misses.
3.  The fill lands on the outermost leaf, not the first.
4.  ``.opening()`` and ``.void()`` agree on identical geometry.
5.  The phantom notch is gone — and now because the hole is REAL.
6.  ``Slab``/``Roof`` still refuse, and the refusal names why.
7.  Both backends.
"""

from __future__ import annotations

import logging
import os
import tempfile

import pytest

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.models import (
    Bar, Beam, Box, Column, Element, Point, Project, Roof, Slab, Window,
)

# ---------------------------------------------------------------------------
# The fixture. A 6 m three-layer buildup in a container, 400 mm deep overall,
# with one 1200 x 1400 hole at along=4200 / up=900 straight through it.
#
# Layers are stacked across the thickness, outer face at y = 0:
#
#     y[-100,    0]   "outer"   100 mm   <- the facade, and the fill's leaf
#     y[-300, -100]   "core"    200 mm
#     y[-400, -300]   "inner"   100 mm
#
# ...and one control leaf CLEAR of the hole: a 100 mm capping course sitting on
# top at z[2700, 2800]. It abuts the three layers rather than overlapping them
# (flush faces never trigger a displacement carve), so the fixture's only
# boolean would be one this feature emitted.
# ---------------------------------------------------------------------------

_RUN, _H = 6000, 2700

#: Where the hole actually lands in WORLD coordinates — host-frame ``along=``,
#: not world x. Derived rather than restated in
#: ``test_the_hole_lands_where_the_host_frame_says``.
_HOLE_X0, _HOLE_X1 = 0.6, 1.8
_HOLE_Z0, _HOLE_Z1 = 0.9, 2.3
_HOLE_W = _HOLE_X1 - _HOLE_X0
_HOLE_HT = _HOLE_Z1 - _HOLE_Z0

#: ``(name, y0, y1)`` in millimetres, OUTERMOST LAST — the add order is
#: inner-first on purpose, so "the outermost leaf" and "the first leaf" are
#: different products and ``test_the_one_fill_sits_on_the_outermost_leaf`` can
#: fail.
_LAYERS = (("inner", -400, -300), ("core", -300, -100), ("outer", -100, 0))

_CONTAINERS = {
    "Column": lambda: Column(name="host"),
    "Beam": lambda: Beam(name="host"),
    "Element": lambda: Element(ifc_class="IfcWall", name="host"),
}

_HORIZONTAL = {
    "Slab": lambda: Slab(name="host"),
    "Roof": lambda: Roof(name="host"),
}


def _box(name: str, y0: int, y1: int, z0: int = 0, z1: int = _H) -> Box:
    return Box(name=name, type="wall",
               start=Point(x=0, y=y0, z=z0), end=Point(x=_RUN, y=y1, z=z1))


def _window(name: str = "g0") -> Window:
    return Window(width=1200, height=1400, name=name)


def _bar(name: str = "det", x: int = 1200) -> Bar:
    """A 32 mm bar straight through the hole, full storey height. Its own AABB
    is DEGENERATE (path points only), so it is reachable at all through
    ``padded_aabb`` — the same shape measured losing 1.4 m of section to a
    window that was not in the file."""
    return Bar(path=[Point(x=x, y=-150, z=0), Point(x=x, y=-150, z=_H)],
               diameter=32, name=name)


def _host(kind: str = "Column", *, layers=_LAYERS, hole: str = "opening",
          cap: bool = True, bar: bool = False):
    """The fixture. ``hole`` picks the spelling; ``layers`` picks how many.

    ``hole="void"`` cuts the box the OPENING derives (``_HOLE_*``), never the
    box ``along=4200`` reads like — see the module docstring.
    """
    host = _CONTAINERS[kind]() if kind in _CONTAINERS else _HORIZONTAL[kind]()
    for name, y0, y1 in layers:
        host.add(_box(name, y0, y1))
    if cap:
        # Clear of the hole in Z, abutting the layers rather than crossing
        # them. The control that keeps "distributes to what it overlaps" from
        # being indistinguishable from "attaches to everything".
        host.add(_box("cap", -400, 0, z0=_H, z1=_H + 100))
    if bar:
        host.add(_bar())

    if hole == "opening":
        host.opening(_window(), along=4200, up=900)
    elif hole == "void":
        host.void(Box(start=Point(x=int(_HOLE_X0 * 1000), y=-500,
                                  z=int(_HOLE_Z0 * 1000)),
                      end=Point(x=int(_HOLE_X1 * 1000), y=100,
                                z=int(_HOLE_Z1 * 1000))), name="hole")
    elif hole == "none":
        pass
    else:                                     # pragma: no cover - typo guard
        raise AssertionError(hole)
    return host


def _project(element) -> Project:
    proj = Project(name="distributed-openings")
    proj.add(element)
    return proj


def _errors(element) -> list:
    return validate_project_report(_project(element)).errors


def _compiled(element, backend: str = "ifcopenshell") -> str:
    from lite_step.ifc.generator import generate_ifc

    proj = _project(element)
    assert validate_project_report(proj).errors == []
    result = generate_ifc(normalize_project_to_meters(proj) or proj)
    assert result.success, result.error
    return result.ifc_content


def _counts(ifc: str) -> dict:
    """RELATION counts by name, not product counts.

    A product count passes on several wrong implementations — one opening
    reused across leaves, or a second window emitted and orphaned — because it
    is the ``IfcRelVoidsElement`` / ``IfcRelFillsElement`` rows that say where
    a hole actually landed and what actually fills it.
    """
    return {k: ifc.count(f"{k}(") for k in
            ("IFCOPENINGELEMENT", "IFCWINDOW", "IFCDOOR", "IFCRELVOIDSELEMENT",
             "IFCRELFILLSELEMENT", "IFCBOOLEANRESULT")}


def _model(ifc_text: str):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(ifc_text)
    try:
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _volumes(ifc_text: str, *, carved: bool) -> dict:
    """Tessellated world volume of every named ``box:`` product.

    ``carved=True`` lets ``ifcopenshell.geom`` apply the product's
    ``IfcRelVoidsElement``s — which is the whole subject here, because a
    distributed opening carves through a RELATION and not through a boolean, so
    the ``disable-opening-subtractions`` setting that
    ``test_opening_reach_matches_void`` needs would measure exactly the thing
    this feature does not do. ``carved=False`` is the uncarved control, and the
    pair is what makes "the leaf really lost the hole" a measurement rather
    than a count.
    """
    geom = pytest.importorskip("ifcopenshell.geom")
    import ifcopenshell.util.shape

    model = _model(ifc_text)
    settings = geom.settings()
    settings.set("use-world-coords", True)
    settings.set("disable-opening-subtractions", not carved)
    out = {}
    for product in model.by_type("IfcProduct"):
        name = product.Name or ""
        if product.Representation is None or not name.startswith("box:"):
            continue
        shape = geom.create_shape(settings, product)
        out[name.split(":")[1]] = ifcopenshell.util.shape.get_volume(
            shape.geometry)
    return out


def _voided_hosts(ifc_text: str) -> list:
    """``[host name]`` for every ``IfcRelVoidsElement``, in file order."""
    model = _model(ifc_text)
    return [r.RelatingBuildingElement.Name
            for r in model.by_type("IfcRelVoidsElement")]


def _fill_host(ifc_text: str):
    """``(host name, fill class, fill name)`` for the ONE fill in the file."""
    model = _model(ifc_text)
    fills = model.by_type("IfcRelFillsElement")
    assert len(fills) == 1, [f.RelatedBuildingElement.Name for f in fills]
    rel = fills[0]
    host = next(v.RelatingBuildingElement
                for v in model.by_type("IfcRelVoidsElement")
                if v.RelatedOpeningElement == rel.RelatingOpeningElement)
    fill = rel.RelatedBuildingElement
    return host.Name, fill.is_a(), fill.Name


# ---------------------------------------------------------------------------
# 0. The premises this design was derived from
# ---------------------------------------------------------------------------


def test_the_schema_bound_that_forces_one_fill_per_hole() -> None:
    """N openings and ONE fill is not a preference — IFC4X3_ADD2 decides it.

    Read off the shipped schema rather than quoted from the spec, so a schema
    bump that relaxed ``FillsVoids`` would fail here and the design could be
    revisited instead of being carried forward as folklore.
    """
    pytest.importorskip("ifcopenshell")
    from ifcopenshell import ifcopenshell_wrapper

    schema = ifcopenshell_wrapper.schema_by_name("IFC4X3_ADD2")
    bounds = {}
    for cls, attrs in (("IfcElement", ("FillsVoids", "HasOpenings")),
                       ("IfcOpeningElement", ("HasFillings",))):
        entity = schema.declaration_by_name(cls).as_entity()
        for attr in entity.all_inverse_attributes():
            if attr.name() in attrs:
                bounds[f"{cls}.{attr.name()}"] = (attr.bound1(), attr.bound2())

    # An element fills AT MOST ONE opening — so one IfcWindow cannot fill the
    # N holes a distributed opening makes.
    assert bounds["IfcElement.FillsVoids"] == (0, 1)
    # ...while a host may carry many, which is what makes the distribution
    # legal at all. ``-1`` is ifcopenshell's spelling of an unbounded ``?``.
    assert bounds["IfcElement.HasOpenings"] == (0, -1)
    assert bounds["IfcOpeningElement.HasFillings"] == (0, -1)


def test_the_hole_lands_where_the_host_frame_says() -> None:
    """``along=`` is HOST-frame, and this host's run axis opposes world +X.

    both were both filed on this confound and both closed invalid.
    ``along=4200`` on a 6 m container puts the hole at world ``x[0.6, 1.8]``,
    so a ``.void(Box(x=4200..5400))`` meant as "the same hole" would cut 3.6 m
    away. Every ``hole="void"`` fixture in this file derives its box from the
    numbers asserted here.
    """
    from lite_step.compiler import frames
    from lite_step.ifc.entity_cache import snap_to_precision

    proj = _project(_host())
    assert validate_project_report(proj).errors == []
    normalized = normalize_project_to_meters(proj) or proj
    from lite_step.ifc.generator import generate_ifc
    assert generate_ifc(normalized).success        # runs the frame passes

    host = normalized.storeys[0].elements[0]
    frame = frames.opening_host_frame(host, snap=snap_to_precision,
                                      divisor=1000.0)
    assert frame is not None, "a container derives its frame from its leaves"
    window = next(c for c in host._elements if type(c).__name__ == "Window")
    (x0, _y0, z0), (x1, _y1, z1) = frames.opening_void_prism(
        window, frame).aabb()

    assert (round(x0, 6), round(x1, 6)) == (_HOLE_X0, _HOLE_X1)
    assert (round(z0, 6), round(z1, 6)) == (_HOLE_Z0, _HOLE_Z1)
    assert round(x0, 6) != 4.2                # ...the trap, in one line


# ---------------------------------------------------------------------------
# 1. The relations — counted by name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(_CONTAINERS))
def test_one_opening_lands_on_every_leaf_it_crosses(kind) -> None:
    """Three layers crossed, one capping course missed: 3 holes, 1 window.

    Counted as RELATIONS. A product count of 3 ``IfcOpeningElement`` passes on
    an implementation that mints three and links none of them, which is the
    silence refused, one indirection further along.
    """
    ifc = _compiled(_host(kind))
    counts = _counts(ifc)

    assert counts["IFCRELVOIDSELEMENT"] == 3, kind
    assert counts["IFCOPENINGELEMENT"] == 3, kind
    assert counts["IFCRELFILLSELEMENT"] == 1, kind
    assert counts["IFCWINDOW"] == 1, kind
    assert sorted(n.split(":")[1] for n in _voided_hosts(ifc)) == [
        "core", "inner", "outer"]


def test_the_leaf_the_hole_misses_gets_no_opening() -> None:
    """Distribution is by OVERLAP, not "attach to every child".

    That fallback existed once, and it put an ``IfcOpeningElement`` in the file
    for every leaf a hole missed — visible in the entity count, invisible in
    the render. The capping course sits 400 mm above the hole's head.
    """
    ifc = _compiled(_host())
    assert "cap" not in [n.split(":")[1] for n in _voided_hosts(ifc)]
    assert _volumes(ifc, carved=True)["cap"] == pytest.approx(
        _volumes(ifc, carved=False)["cap"], rel=1e-9)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_only_one_window_reaches_the_file_however_many_leaves_it_crosses(
        count) -> None:
    """The schedule objection, asserted as the count that would break it.

    One ``IfcWindow`` per leaf would be schema-clean and would double-count
    every window in every quantity take-off — the reason could not spell a
    layered wall's core with ``.opening()``. The number of holes scales with
    the buildup; the number of windows does not.
    """
    ifc = _compiled(_host(layers=_LAYERS[-count:]))
    counts = _counts(ifc)

    assert counts["IFCRELVOIDSELEMENT"] == count
    assert counts["IFCWINDOW"] == 1
    assert counts["IFCRELFILLSELEMENT"] == 1


# ---------------------------------------------------------------------------
# 2. The carved SOLID, per leaf
# ---------------------------------------------------------------------------


def test_every_crossed_leaf_really_loses_the_hole_from_its_solid() -> None:
    """The count says a relation exists; this says the material is gone.

    Each layer loses exactly ``width x height x ITS OWN thickness`` — the hole
    cuts the full depth of the container, so each leaf gives up its own share
    of it and no more.
    """
    ifc = _compiled(_host())
    full = _volumes(ifc, carved=False)
    carved = _volumes(ifc, carved=True)

    for name, y0, y1 in _LAYERS:
        thickness = (y1 - y0) / 1000.0
        lost = full[name] - carved[name]
        assert lost == pytest.approx(_HOLE_W * _HOLE_HT * thickness, rel=1e-6), (
            f"{name} lost {lost:.6f} m3")
        assert lost > 0                        # the fixture can lose volume


def test_the_hole_through_the_buildup_is_one_hole() -> None:
    """Every leaf's share sums to the whole prism — so the three openings are
    one hole through the assembly rather than three at three depths."""
    ifc = _compiled(_host())
    full = _volumes(ifc, carved=False)
    carved = _volumes(ifc, carved=True)

    total_lost = sum(full[n] - carved[n] for n, _y0, _y1 in _LAYERS)
    depth = (_LAYERS[-1][2] - _LAYERS[0][1]) / 1000.0        # 0.4 m
    assert total_lost == pytest.approx(_HOLE_W * _HOLE_HT * depth, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. Which leaf carries the one fill
# ---------------------------------------------------------------------------


def test_the_one_fill_sits_on_the_outermost_leaf_not_the_first_one() -> None:
    """The rule, and the fixture that can tell it from "the first one".

    ``_LAYERS`` is added INNER-FIRST, so emission order says ``inner`` and the
    outward normal says ``outer``. A reader will assume the first one; the
    source says otherwise (``generator._outermost_host_index``) and this is
    what holds it to that.

    Outermost is not arbitrary: ``inset=`` is measured from the OUTER FACE, so
    attaching the fill to the lining would put a flush window on a product two
    layers behind the facade it is flush with.
    """
    host_name, fill_class, fill_name = _fill_host(_compiled(_host()))

    assert host_name.split(":")[1] == "outer"
    assert host_name.split(":")[1] != _LAYERS[0][0]     # ...not "the first"
    assert fill_class == "IfcWindow"
    assert fill_name.startswith("window:g0")


def test_the_fill_follows_the_outer_face_when_the_layers_are_reordered() -> None:
    """Same three layers, reversed add order — the answer must not move.

    If the rule were emission order this flips and the test above cannot see
    it, because reversing ``_LAYERS`` makes "first" and "outermost" agree
    again. Two orders, one answer, is what pins the rule itself.
    """
    outer_first = _fill_host(_compiled(_host(layers=tuple(reversed(_LAYERS)))))
    inner_first = _fill_host(_compiled(_host()))

    assert outer_first[0].split(":")[1] == "outer"
    assert outer_first[0] == inner_first[0]


# ---------------------------------------------------------------------------
# 4. ``.opening()`` and ``.void()`` are two spellings of one hole
# ---------------------------------------------------------------------------


def test_the_two_spellings_of_one_hole_carve_a_container_identically() -> None:
    """The invariant pins for a Wall, on the hosts added.

    Same container, same leaves, one hole spelled twice. The ``.void()`` box is
    the OPENING's own prism (``_HOLE_*``), never the ``along=`` scalars — see
    the module docstring on why those are 3.6 m apart.

    ``.void()`` brings no fill, so the fill relations differ by exactly one and
    nothing else may.
    """
    by_opening = _compiled(_host(hole="opening"))
    by_void = _compiled(_host(hole="void"))

    assert _voided_hosts(by_opening) == _voided_hosts(by_void)
    assert _volumes(by_opening, carved=True) == pytest.approx(
        _volumes(by_void, carved=True), rel=1e-9)

    o_counts, v_counts = _counts(by_opening), _counts(by_void)
    assert o_counts["IFCRELVOIDSELEMENT"] == v_counts["IFCRELVOIDSELEMENT"] == 3
    assert o_counts["IFCBOOLEANRESULT"] == v_counts["IFCBOOLEANRESULT"] == 0
    assert (o_counts["IFCRELFILLSELEMENT"], v_counts["IFCRELFILLSELEMENT"]) == (1, 0)
    assert (o_counts["IFCWINDOW"], v_counts["IFCWINDOW"]) == (1, 0)


def test_a_bar_in_the_hole_is_notched_by_neither_spelling_and_holed_by_both() -> None:
    """The phantom notch, and its resolution. Measurement found a ``.add()``ed bar standing in a Column's hole losing 1.4 m
    of section to a window that was **not in the file**
    (``IFCOPENINGELEMENT: 0  IFCWINDOW: 0  IFCBOOLEANRESULT: 2``). The refusal
    stopped it by making the tree unbuildable. Now the tree builds, and the bar
    loses its material to a hole that IS there: an ``IfcRelVoidsElement`` on
    the bar's own product, not a boolean.

    Both spellings, because a boolean here would be the double-carve
    ``displacement`` skips these containers to avoid — one redundant operand
    per detail per hole, paid for in CSG depth.

    Measured against the SAME model with no hole at all, not against zero: the
    bar is buried in the core layer, so one ordinary displacement boolean is
    there before any window is authored and must still be there after. What
    must not appear is a SECOND one.
    """
    baseline = _counts(_compiled(
        _host(hole="none", cap=False, bar=True)))["IFCBOOLEANRESULT"]
    assert baseline == 1, "the fixture's own body-vs-bar carve"

    for hole in ("opening", "void"):
        ifc = _compiled(_host(hole=hole, cap=False, bar=True))
        counts = _counts(ifc)
        assert counts["IFCBOOLEANRESULT"] == baseline, hole
        assert "bar:det:column:host" in _voided_hosts(ifc), hole
        # ...and it is the SAME count of holes either way: 3 layers + the bar.
        assert counts["IFCRELVOIDSELEMENT"] == 4, hole


# ---------------------------------------------------------------------------
# 5. Still refused — the half that stays open
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(_HORIZONTAL))
def test_a_horizontal_container_still_refuses(kind) -> None:
    """``Slab`` and ``Roof`` are NOT in scope, and the refusal says why.

    Distribution answers *where the hole goes*. It does not answer *what a
    horizontal host's opening measures against*: a slab's outer face is the
    soffit or the topside depending which way you look, and its run axis is a
    plan direction with no canonical choice between the two in-plane axes. A
    floor hatch is ordinary BIM and needs a spelling of its own — that gap stays
    open for it, and this test is what stops it being closed by a widened
    predicate.
    """
    errors = _errors(_host(kind))

    assert len(errors) == 1, errors
    assert "no IFC emitter reads" in errors[0]
    assert "HORIZONTAL" in errors[0]
    assert "not implemented" in errors[0]


def test_a_container_with_no_leaves_refuses_rather_than_emitting_nothing() -> None:
    """The frame comes from the leaves, so no leaves is no frame AND no landing
    place. Loud at compile — the generator would otherwise raise mid-emission,
    which is a stack trace where an author needs a sentence."""
    host = Column(name="host")
    host.opening(_window(), along=4200, up=900)

    errors = _errors(host)
    assert any("no .add()ed children" in e for e in errors), errors


# ---------------------------------------------------------------------------
# 6. Loud failure
# ---------------------------------------------------------------------------


def test_an_opening_that_reaches_no_leaf_warns_and_emits_nothing(caplog) -> None:
    """A hole placed past the end of what the container aggregates.

    "No opening in the file" and "no leaf overlapped" must not look alike from
    the outside, so the miss is named on the log with the frame trap spelled
    out — it is by far the likeliest cause.
    """
    host = Column(name="host")
    host.add(_box("outer", -100, 0))
    host.add(_box("core", -300, -100))
    # up=9000 puts the sill 6.3 m above the top of the buildup.
    host.opening(_window(), along=4200, up=9000)

    with caplog.at_level(logging.WARNING):
        ifc = _compiled(host)

    counts = _counts(ifc)
    assert counts["IFCRELVOIDSELEMENT"] == 0
    assert counts["IFCWINDOW"] == 0
    logged = [r.getMessage() for r in caplog.records]
    assert any("overlaps no representation-bearing product" in m
               for m in logged), logged
    assert any("HOST's frame" in m for m in logged), logged


# ---------------------------------------------------------------------------
# 7. The compiler's container tree print
# ---------------------------------------------------------------------------


def test_the_container_tree_prints_the_openings_the_container_now_emits(
        capsys) -> None:
    """``openings=`` on the container row, and the layers beneath it.

    The tree print reads both lists already, so this is a check that the row it
    prints for a host that now EMITS is the row an author can act on — the
    count is the number of authored holes (one), not the number of
    ``IfcOpeningElement``s the distribution made (three).
    """
    from lite_step.compiler.tree_report import tree_report_lines

    proj = _project(_host())
    assert validate_project_report(proj).errors == []
    text = "\n".join(tree_report_lines(normalize_project_to_meters(proj)
                                       or proj))

    assert "column:host" in text
    assert "openings=1" in text
    for name, _y0, _y1 in _LAYERS:
        assert f"box:{name}" in text
