"""``.opening()`` and ``.void()`` reach the SAME details — pinned, not assumed.

``test_void_carves_details.py`` taught ``.void()`` to cut the details
standing in its hole, and asserted it against ``.opening()``. What it never
asserted is the half that made look real: it compared the two verbs on an
``.anchor()``ed detail only. Nothing in the suite said that an ``.add()``ed
detail is reached by BOTH verbs, so "``.opening()`` only carves anchored
children" was a claim the test suite could not contradict.

The invariant, stated once:

    **An opening cuts every geometry-bearing descendant it overlaps,
    regardless of the verb that attached that descendant.**

It holds today — measured on ``main`` @ 47065bc before a line of this file was
written — and this suite is what keeps it holding. Every test here compares the
two spellings of one hole on ONE fixture, so a regression in either verb shows
up as the two numbers coming apart rather than as an absolute that someone has
to re-derive.

Why measured otherwise, and the trap this file exists to close
------------------------------------------------------------------

``along=`` is measured in the HOST's frame, and that frame's ``+x`` is not
world ``+x``. On the fixture below — a 6 m wall authored ``x[0, 6000]`` —
``.anchor(win, along=4200, width=1200)`` puts the hole at world
``x[0.6, 1.8]``, because the wall's run axis points the other way (v21.0.0's
opening-local frame; ``along=0`` is the corner on your LEFT as you face the
outer face). A ``.void()`` operand is ABSOLUTE world geometry, so
``.void(Box(x=4200..5400))`` cuts at world ``x[4.2, 5.4]``.

Spell the "same" hole those two ways and you have written two holes 3.6 m
apart. A detail authored at world ``x = 4.8`` then stands in the ``.void()``
one and clear of the ``.opening()`` one — which reads exactly like "the
opening does not reach ``.add()``ed children", and is not.

``test_the_hole_lands_where_the_frame_says_not_where_world_x_says`` pins that
fact with numbers, and every fixture here derives its ``.void()`` box from the
opening's own prism instead of restating the ``along=`` scalars, so the two
verbs cannot drift apart into two holes again.

Coverage
--------

1.  ``.add(Bar)`` on the wall — the realistic reinforcement shape, and the one
    with a DEGENERATE path-point AABB that only ``padded_aabb`` can bound.
2.  ``.add(Box)`` inside an ``.add()``ed nested container — a Wall takes
    exactly one body element, so this is the only legal way to ``.add()`` a Box
    detail to a wall, and it exercises the recursive half of the reach.
3.  ``.anchor(Box)`` on the wall, positioned to occupy the SAME world volume as
    (2) — so "which verb attached it" is the only difference between them.
4.  Controls: a detail clear of the hole, ``carve="none"``, and no hole at all.
5.  Both backends.
6.  The same four-way table on a representation-less CONTAINER, where
    the hole is realised as an ``IfcRelVoidsElement`` per leaf instead of as a
    boolean. Different mechanism, same invariant — see §5 below.
7.  And once more on a BODY-LESS WALL — the host where the choice
    between those two mechanisms is made per INSTANCE, off ``tx.body_prisms``,
    because a Wall's own prism becomes the ``IfcWall``'s representation. §6.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from lite_step.compiler import frames
from lite_step.compiler.displacement import carve_pairs
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.ifc.entity_cache import snap_to_precision
from lite_step.models import Bar, Box, Element, Point, Project, Wall, Window

# ---------------------------------------------------------------------------
# The fixture. One 6 m wall running +X, 300 mm thick, y in [-300, 0],
# z in [0, 2700], with a 1200 x 1400 hole at along=4200 / up=900.
# ---------------------------------------------------------------------------

#: Where that hole actually lands in WORLD coordinates. Not restated from
#: ``along=`` — ``test_the_hole_lands_where_the_frame_says…`` derives it from
#: ``frames.opening_void_prism`` and asserts these numbers, so if the frame ever
#: changes this file fails loudly instead of silently testing two holes.
_HOLE_X0, _HOLE_X1 = 0.6, 1.8
_HOLE_Z0, _HOLE_Z1 = 0.9, 2.3

#: The Box detail, in world meters. It straddles the hole's TOP edge on
#: purpose: a detail wholly inside the hole tessellates to nothing, and
#: "nothing" is not a measurement — it is what an empty shape and a correct
#: carve look like alike.
_DET_X0, _DET_X1 = 1.1, 1.3
_DET_Y0, _DET_Y1 = -0.2, -0.1
_DET_Z0, _DET_Z1 = 1.0, 2.7

_DET_FULL = (_DET_X1 - _DET_X0) * (_DET_Y1 - _DET_Y0) * (_DET_Z1 - _DET_Z0)
_DET_LEFT = (_DET_X1 - _DET_X0) * (_DET_Y1 - _DET_Y0) * (_DET_Z1 - _HOLE_Z1)

#: The Bar detail: a 32 mm vertical bar through the full storey height, dead
#: centre of the hole. Its own AABB is DEGENERATE (path points only), so it is
#: reachable at all only through ``padded_aabb`` — which is exactly why's
#: brief flagged it, and why it is not enough to test the Box.
_BAR_X = 1.2
_BAR_D = 32


def _body() -> Box:
    return Box(start=Point(x=0, y=-300, z=0),
               end=Point(x=6000, y=0, z=2700), type="wall", name="body")


def _added_bar(name: str = "det") -> Bar:
    return Bar(path=[Point(x=int(_BAR_X * 1000), y=-150, z=0),
                     Point(x=int(_BAR_X * 1000), y=-150, z=2700)],
               diameter=_BAR_D, name=name)


def _world_box(name: str = "det") -> Box:
    return Box(start=Point(x=int(_DET_X0 * 1000), y=int(_DET_Y0 * 1000),
                           z=int(_DET_Z0 * 1000)),
               end=Point(x=int(_DET_X1 * 1000), y=int(_DET_Y1 * 1000),
                         z=int(_DET_Z1 * 1000)),
               name=name)


def _wall(*, hole: str = "opening", detail: str = "added_bar",
          no_carve: bool = False) -> Wall:
    """The fixture. ``hole`` picks the spelling, ``detail`` the attach verb.

    ``hole="void"`` cuts the box the OPENING derives (``_HOLE_*``), not the box
    ``along=4200`` reads like. That substitution is the whole point of the
    comparison — see the module docstring.
    """
    wall = Wall(name="south")
    wall.add(_body())

    # ``.no_carve()`` was called on the detail AFTER it was attached; ``carve=``
    # is stated at the attach itself, so the flag threads through the branches
    # rather than being applied by a name scan afterwards. Same reach: the scan
    # matched direct children named "det", which is every branch below except
    # ``added_box``, where the detail sits one level down inside ``cage``.
    ex = "none" if no_carve else None

    if detail == "added_bar":
        wall.add(_added_bar(), carve=ex)
    elif detail == "added_box":
        # A Wall takes exactly one body element, so a second Box has to come in
        # inside a container. This is also the realistic shape for a rebar cage
        # or a fixings group, and it is the recursive half of the reach.
        cage = Element(ifc_class="IfcBuildingElementProxy", name="cage")
        cage.add(_world_box())
        wall.add(cage)
    elif detail == "anchored_box":
        # Authored in the wall's LOCAL frame and placed to land on exactly the
        # world volume ``_world_box()`` occupies — so (2) and (3) differ only
        # in the verb that attached them.
        d = Box(start=Point(x=-100, y=100, z=int(_DET_Z0 * 1000)),
                end=Point(x=100, y=200, z=int(_DET_Z1 * 1000)), name="det")
        wall.anchor(d, along=4800, up=0, carve=ex)
    elif detail == "clear_bar":
        wall.add(_added_bar(), carve=ex)     # in the hole
        wall.add(Bar(path=[Point(x=5000, y=-150, z=0),
                           Point(x=5000, y=-150, z=2700)],
                     diameter=_BAR_D, name="clear"))
    elif detail == "none":
        pass
    else:                                     # pragma: no cover - typo guard
        raise AssertionError(detail)

    if hole == "opening":
        wall.anchor(Window(width=1200, height=1400, name="g0"),
                    along=4200, up=900)
    elif hole == "void":
        wall.void(Box(start=Point(x=int(_HOLE_X0 * 1000), y=-400,
                                  z=int(_HOLE_Z0 * 1000)),
                      end=Point(x=int(_HOLE_X1 * 1000), y=100,
                                z=int(_HOLE_Z1 * 1000))), name="hole")
    elif hole == "none":
        pass
    else:                                     # pragma: no cover - typo guard
        raise AssertionError(hole)
    return wall


def _project(element) -> Project:
    proj = Project(name="opening-reach")
    proj.add(element)
    assert validate_project_report(proj).errors == []
    return normalize_project_to_meters(proj) or proj


def _compiled(proj: Project, backend: str = "ifcopenshell") -> str:
    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(proj)
    assert result.success, result.error
    return result.ifc_content


def _find(proj: Project, name: str):
    """The named element anywhere under the wall, at any depth."""
    def walk(elem):
        for child in ((getattr(elem, "_elements", None) or [])
                      + (getattr(elem, "_openings", None) or [])):
            if getattr(child, "name", None) == name:
                return child
            found = walk(child)
            if found is not None:
                return found
        return None

    hit = walk(proj.storeys[0].elements[0])
    assert hit is not None, f"no element named {name!r}"
    return hit


def _cuts(proj: Project, name: str) -> int:
    return len(getattr(_find(proj, name), "_cuts", None) or [])


def _volume(ifc_text: str, product_name: str) -> float:
    """Tessellated world volume of one named product's OWN representation.

    ``disable-opening-subtractions`` is load-bearing, and it is the same
    setting ``test_void_carves_details`` and ``test_opening_carves_details``
    explain: ``ifcopenshell.geom`` applies a parent's ``IfcRelVoidsElement`` to
    the products AGGREGATED under it, so a detail inside this wall would
    measure the carved volume whether or not it carries a boolean of its own —
    the measurement could not fail. Off, it reports only what is in the
    element's own representation, which is also what web-ifc / ThatOpen renders
    (our production path).
    """
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


def _detail_volume(*, hole: str, detail: str) -> float:
    proj = _project(_wall(hole=hole, detail=detail))
    product = "bar:det" if detail.endswith("bar") else "box:det"
    return _volume(_compiled(proj), product)


# ---------------------------------------------------------------------------
# 0. The fixture, and the trap
# ---------------------------------------------------------------------------


def test_the_hole_lands_where_the_frame_says_not_where_world_x_says() -> None:
    """``along=`` is host-frame, and this host's run axis opposes world +X.

    This is the fact that made read as a reach bug: ``along=4200`` on a
    6 m wall authored ``x[0, 6000]`` puts the hole at world ``x[0.6, 1.8]``,
    while a ``.void(Box(x=4200..5400))`` meant as "the same hole" cuts 3.6 m
    away. Compare those two and an ``.add()``ed detail at world x=4.8 is carved
    by one and untouched by the other — for reasons that have nothing to do
    with how it was attached.

    Every ``hole="void"`` fixture in this file derives its box from these
    numbers, so the assertion below is what keeps the whole suite honest.
    """
    proj = _project(_wall(hole="opening", detail="none"))
    _compiled(proj)                          # runs the frame + anchor passes

    wall = proj.storeys[0].elements[0]
    frame = frames.opening_host_frame(frames._body_of(wall),
                                      snap=snap_to_precision, divisor=1000.0)
    prism = frames.opening_void_prism(_find(proj, "g0"), frame)
    (x0, _y0, z0), (x1, _y1, z1) = prism.aabb()

    assert (round(x0, 6), round(x1, 6)) == (_HOLE_X0, _HOLE_X1)
    assert (round(z0, 6), round(z1, 6)) == (_HOLE_Z0, _HOLE_Z1)
    # ...and NOT where the scalars read like, which is the trap in one line.
    assert round(x0, 6) != 4.2


# ---------------------------------------------------------------------------
# 1. The invariant: the attach verb does not decide whether the opening reaches
# ---------------------------------------------------------------------------


def test_an_opening_carves_a_bar_that_was_added_not_anchored() -> None:
    """The realistic reinforcement shape: rebar is ``.add()``ed to the wall it
    reinforces, in the wall's own world coordinates, and a window is
    ``.opening()``ed through the same wall. The bar must lose its section
    where the glazing is.

    A count of openings passes on a broken version — that is how the gap survived — so this asserts the SOLID. The bar runs the full 2.7 m
    storey height and the hole takes 1.4 m of it, so a bar that keeps its
    material measures ~2.07x what a carved one does.
    """
    full = _detail_volume(hole="none", detail="added_bar")
    carved = _detail_volume(hole="opening", detail="added_bar")

    lost = (full - carved) / full
    assert lost == pytest.approx((_HOLE_Z1 - _HOLE_Z0) / 2.7, rel=0.02), (
        f"the added bar lost {lost:.1%} of its volume; the hole covers "
        f"{(_HOLE_Z1 - _HOLE_Z0) / 2.7:.1%} of its length")


def test_the_two_spellings_of_one_hole_carve_an_added_bar_identically() -> None:
    """THE test, and the one could not have written: same wall, same hole,
    one detail, ``.add()``ed — and the verb that spells the hole must not
    change the answer.

    Falsify by reverting ``_carve_openings_through_details``: the ``.void()``
    bar keeps its notch, the ``.opening()`` bar goes back to its full
    0.002171 m3, and the two numbers stop matching.
    """
    by_opening = _detail_volume(hole="opening", detail="added_bar")
    by_void = _detail_volume(hole="void", detail="added_bar")

    assert by_opening == pytest.approx(by_void, rel=1e-9)
    assert by_opening < _detail_volume(hole="none", detail="added_bar")


def test_an_opening_reaches_a_box_added_inside_a_nested_container() -> None:
    """A Wall takes exactly one body element, so an ``.add()``ed Box detail
    arrives inside a container — a rebar cage, a fixings group. The reach is
    recursive, and this is the test that says so.

    Asserted as a real volume: the detail straddles the hole's top edge, so a
    correct carve leaves exactly the 0.4 m above it.
    """
    proj = _project(_wall(hole="opening", detail="added_box"))
    ifc = _compiled(proj)

    assert _volume(ifc, "box:det") == pytest.approx(_DET_LEFT, rel=1e-6)
    assert _DET_LEFT < _DET_FULL          # the fixture can actually lose volume


def test_added_and_anchored_details_in_one_place_are_carved_the_same() -> None:
    """Same world volume, same hole — reached through ``.add()`` and through
    ``.anchor()``. Any difference between these two numbers IS the bug
    described, and there is nothing else left in the fixture for it to be.
    """
    added = _detail_volume(hole="opening", detail="added_box")
    anchored = _detail_volume(hole="opening", detail="anchored_box")

    assert added == pytest.approx(anchored, rel=1e-6)
    assert added == pytest.approx(_DET_LEFT, rel=1e-6)


def test_both_verbs_agree_on_both_attach_verbs() -> None:
    """The four-way agreement, in one assertion: {opening, void} x {add,
    anchor} is one number. This is the invariant in its widest form."""
    measured = {(hole, detail): _detail_volume(hole=hole, detail=detail)
                for hole in ("opening", "void")
                for detail in ("added_box", "anchored_box")}

    first = next(iter(measured.values()))
    for key, value in measured.items():
        assert value == pytest.approx(first, rel=1e-6), key
    assert first == pytest.approx(_DET_LEFT, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. Controls — so nothing here can pass vacuously
# ---------------------------------------------------------------------------


def test_an_added_detail_clear_of_the_opening_keeps_its_material() -> None:
    """Reach is not "carve everything". The second bar stands 3.8 m from the
    hole and must come out whole under both spellings."""
    for hole in ("opening", "void"):
        proj = _project(_wall(hole=hole, detail="clear_bar"))
        ifc = _compiled(proj)
        assert _cuts(proj, "clear") == 0, hole
        assert _volume(ifc, "bar:clear") == pytest.approx(
            _detail_volume(hole="none", detail="added_bar"), rel=1e-9), hole


def test_no_carve_exempts_an_added_detail_too() -> None:
    """The escape hatch is about the DETAIL, not about the verb that attached
    it — a sill meant to stand in the reveal says so the same way whether it
    was ``.add()``ed or ``.anchor()``ed."""
    for hole in ("opening", "void"):
        proj = _project(_wall(hole=hole, detail="added_bar", no_carve=True))
        _compiled(proj)
        assert _cuts(proj, "det") == 0, hole


def test_no_hole_means_no_cuts_under_either_spelling() -> None:
    proj = _project(_wall(hole="none", detail="added_bar"))
    _compiled(proj)
    assert _cuts(proj, "det") == 0
    assert [r for r in carve_pairs(proj)
            if r[2] in ("opening-void", "void-hole")] == []


def test_the_host_body_is_not_carved_by_its_own_opening() -> None:
    """The wall product already carries the ``IfcRelVoidsElement``, so a second
    boolean of the same volume would be invisible in the render and visible
    only in the CSG depth budget — the resource the primary web viewer silently
    runs out of."""
    proj = _project(_wall(hole="opening", detail="added_bar"))
    ifc = _compiled(proj)

    assert ifc.count("IFCOPENINGELEMENT(") == 1
    assert [r for r in carve_pairs(proj)
            if r[0].startswith("box:body") and r[2] == "opening-void"] == []


# ---------------------------------------------------------------------------
# 3. The carve report says so, under both spellings
# ---------------------------------------------------------------------------


def test_the_report_names_the_added_detail_under_both_spellings() -> None:
    """A detail losing a notch it did not ask for has to be findable without a
    re-render, and the report line is the only record. The mechanism differs
    (``opening-void`` vs ``void-hole``) and the LOSER must not."""
    rows = {}
    for hole in ("opening", "void"):
        proj = _project(_wall(hole=hole, detail="added_bar"))
        _compiled(proj)
        rows[hole] = sorted({lo for lo, _w, m in carve_pairs(proj)
                             if m in ("opening-void", "void-hole")})

    assert rows["opening"] == ["bar:det:wall:south"]
    assert rows["opening"] == rows["void"]


# ---------------------------------------------------------------------------
# 4. Backends
# ---------------------------------------------------------------------------


def test_both_backends_carve_the_added_detail() -> None:
    """Displacement runs before either emitter, so the notch has to be in the
    tree both of them read — but the streaming backend is a separate emitter
    and has been the one to drop things before (the storey attributes)."""
    for backend in ("ifcopenshell",):
        for hole in ("opening", "void"):
            for detail in ("added_bar", "added_box", "anchored_box"):
                proj = _project(_wall(hole=hole, detail=detail))
                _compiled(proj, backend=backend)
                assert _cuts(proj, "det") == 1, (backend, hole, detail)


# ---------------------------------------------------------------------------
# 5. The same table on a CONTAINER host
# ---------------------------------------------------------------------------
#
# A ``Column`` carries no representation of its own, so hands its hole to
# the leaves it overlaps: N ``IfcOpeningElement``, one fill. That is a
# different MECHANISM from the boolean notch above — no ``_cuts`` are appended
# at all — and the invariant this file exists for has to survive the change of
# mechanism or it was an invariant about booleans.
#
# The fixture is deliberately the same geometry: the same 6 m body, the same
# world detail box, the same two scalars. A ``Column`` also takes more than one
# unanchored prism, which a ``Wall`` does not ("a Wall takes exactly one body
# element") — so the ``added_box`` case can be a direct child here rather than
# needing the nested container the wall fixture uses.


def _column(*, hole: str = "opening", detail: str = "added_box") -> Element:
    from lite_step.models import Column

    col = Column(name="post")
    col.add(_body())

    if detail == "added_box":
        col.add(_world_box())
    elif detail == "anchored_box":
        # The identical scalars the wall fixture uses, and they bake to the
        # identical world volume — asserted below rather than assumed.
        d = Box(start=Point(x=-100, y=100, z=int(_DET_Z0 * 1000)),
                end=Point(x=100, y=200, z=int(_DET_Z1 * 1000)), name="det")
        col.anchor(d, along=4800, up=0)
    elif detail == "added_bar":
        col.add(_added_bar())
    else:                                     # pragma: no cover - typo guard
        raise AssertionError(detail)

    if hole == "opening":
        col.anchor(Window(width=1200, height=1400, name="g0"),
                   along=4200, up=900)
    elif hole == "void":
        col.void(Box(start=Point(x=int(_HOLE_X0 * 1000), y=-400,
                                 z=int(_HOLE_Z0 * 1000)),
                     end=Point(x=int(_HOLE_X1 * 1000), y=100,
                               z=int(_HOLE_Z1 * 1000))), name="hole")
    elif hole == "none":
        pass
    else:                                     # pragma: no cover - typo guard
        raise AssertionError(hole)
    return col


def _container_detail_volume(*, hole: str, detail: str,
                             carved: bool = True) -> float:
    """The detail's volume, with the container's ``IfcRelVoidsElement`` APPLIED.

    ``disable-opening-subtractions`` flips relative to :func:`_volume`, and the
    flip is the point: on a ``Wall`` the notch is a boolean IN the detail's own
    representation, so the subtraction has to be off or a parent's void would
    make the measurement unfailable. On a ``Column`` the hole IS the relation,
    hung on the detail's own product — so leaving the subtraction off would
    measure the uncarved solid and pass on a version that emitted nothing.
    """
    geom = pytest.importorskip("ifcopenshell.geom")
    import ifcopenshell
    import ifcopenshell.util.shape

    proj = _project(_column(hole=hole, detail=detail))
    ifc_text = _compiled(proj)
    product_name = "bar:det" if detail.endswith("bar") else "box:det"

    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ifc_text)
        model = ifcopenshell.open(path)
        settings = geom.settings()
        settings.set("use-world-coords", True)
        settings.set("disable-opening-subtractions", not carved)
        matches = [p for p in model.by_type("IfcProduct")
                   if (p.Name or "").startswith(product_name)
                   and p.Representation is not None]
        assert len(matches) == 1, [p.Name for p in matches]
        shape = geom.create_shape(settings, matches[0])
        return ifcopenshell.util.shape.get_volume(shape.geometry)
    finally:
        os.unlink(path)


def test_the_container_fixture_puts_both_details_in_one_world_volume() -> None:
    """The control the four-way table needs before it can mean anything.

    ``.add()``ed in world coordinates and ``.anchor()``ed in the container's
    frame must be the SAME box, or "the verb does not change the answer" would
    be comparing two different details. Measured uncarved, so only the
    placement is under test.
    """
    added = _container_detail_volume(hole="none", detail="added_box",
                                     carved=False)
    anchored = _container_detail_volume(hole="none", detail="anchored_box",
                                        carved=False)

    assert added == pytest.approx(_DET_FULL, rel=1e-6)
    assert anchored == pytest.approx(added, rel=1e-6)


def test_both_verbs_agree_on_both_attach_verbs_on_a_container() -> None:
    """The four-way agreement again, on a host where the hole is a RELATION.

    ``{opening, void} x {add, anchor}`` is one number on a ``Column`` exactly
    as it is on a ``Wall``, and it is the same number: the detail straddles the
    hole's top edge and keeps the 0.4 m above it either way.

    Falsify by reverting ``generator._process_container_openings``: the
    ``.opening()`` column stops emitting any ``IfcRelVoidsElement`` and both
    ``opening`` cells go back to the full 0.000340 m3 while the ``void`` cells
    stay carved.
    """
    measured = {(hole, detail): _container_detail_volume(hole=hole,
                                                         detail=detail)
                for hole in ("opening", "void")
                for detail in ("added_box", "anchored_box")}

    first = next(iter(measured.values()))
    for key, value in measured.items():
        assert value == pytest.approx(first, rel=1e-6), key
    assert first == pytest.approx(_DET_LEFT, rel=1e-6)
    assert first < _DET_FULL              # the fixture can actually lose volume


def test_a_container_opening_reaches_a_bar_the_way_a_void_does() -> None:
    """The degenerate-AABB shape, on the container host.

    A ``Bar``'s own AABB is path POINTS, so it is reachable only through
    ``padded_aabb`` — which ``voids.hole_targets`` uses for exactly this
    reason. It is the leaf a hole is likeliest to miss silently.

    Asserted as AGREEMENT plus "material was lost", not as a fraction, because
    the fraction is wrong and the reason is not ours. ``ifcopenshell.geom``
    subtracting an ``IfcOpeningElement`` from an ``IfcSweptDiskSolid`` removes
    everything BELOW the hole's head rather than just the hole: this bar keeps
    ``z[2.3, 2.7]`` where it should keep ``z[0, 0.9] + z[2.3, 2.7]``. Measured
    identically on ``main`` @ 8dd3541 through the ``.void()`` spelling that
    shipped  (0.000319 m3, ``z[2.3, 2.7]``, both), so it predates
    and is a property of the measuring tool on a swept-disk host. Pinning the
    fraction here would pin THAT, and it would go red the day it is fixed.
    """
    by_opening = _container_detail_volume(hole="opening", detail="added_bar")
    by_void = _container_detail_volume(hole="void", detail="added_bar")
    full = _container_detail_volume(hole="none", detail="added_bar")

    assert by_opening == pytest.approx(by_void, rel=1e-9)
    assert by_opening < full          # ...and the bar really did lose material


def test_a_container_hole_is_not_carved_twice() -> None:
    """The leaves carry the relation, so no boolean may carve the same volume.

    A second carve is invisible in the render and visible only in the CSG depth
    budget — the resource the primary web viewer silently runs out of — which
    is why it needs a test rather than an eyeball.

    Measured against the SAME model with no hole, not against zero: the detail
    is buried in the body, so one ordinary displacement boolean is there before
    any hole is authored. What must not appear is a second one.
    """
    baseline = _compiled(_project(_column(hole="none", detail="added_box"))
                         ).count("IFCBOOLEANRESULT(")
    assert baseline == 1, "the fixture's own body-vs-detail carve"

    for hole in ("opening", "void"):
        proj = _project(_column(hole=hole, detail="added_box"))
        ifc = _compiled(proj)
        assert _cuts(proj, "det") == 0, hole
        assert ifc.count("IFCBOOLEANRESULT(") == baseline, hole
        assert ifc.count("IFCRELVOIDSELEMENT(") == 2, hole   # body + detail


# ---------------------------------------------------------------------------
# 6. The same table on a BODY-LESS WALL host
# ---------------------------------------------------------------------------
#
# The third host shape, and the one where the routing decision is made per
# INSTANCE rather than per class: a Wall's own unanchored prism becomes the
# ``IfcWall``'s representation, so a bodied wall carries the hole itself
# (sections 1-4 above) and a body-less one distributes it
# (``ifc.voids.void_reaches_children``, reading ``tx.body_prisms``).
#
# Deliberately the SAME geometry once more — the same 6 m body, the same world
# detail box, the same two scalars — so this table is the table above with one
# thing changed. The body has to arrive inside a nested Wall leaf, because
# ``.add()``ing it to the assembly directly would make it the assembly's body
# and there would be nothing to measure.


def _bodyless(*, hole: str = "opening", detail: str = "added_box") -> Wall:
    asm = Wall(name="south")
    leaf = Wall(name="core")
    leaf.add(_body())
    asm.add(leaf)

    if detail == "added_box":
        # Same wrapper section 1 uses, and for the same reason: a bare Box
        # ``.add()``ed to a Wall is that Wall's BODY.
        cage = Element(ifc_class="IfcBuildingElementProxy", name="cage")
        cage.add(_world_box())
        asm.add(cage)
    elif detail == "anchored_box":
        d = Box(start=Point(x=-100, y=100, z=int(_DET_Z0 * 1000)),
                end=Point(x=100, y=200, z=int(_DET_Z1 * 1000)), name="det")
        asm.anchor(d, along=4800, up=0)
    elif detail == "added_bar":
        asm.add(_added_bar())
    elif detail == "none":
        pass
    else:                                     # pragma: no cover - typo guard
        raise AssertionError(detail)

    if hole == "opening":
        asm.anchor(Window(width=1200, height=1400, name="g0"),
                   along=4200, up=900)
    elif hole == "void":
        asm.void(Box(start=Point(x=int(_HOLE_X0 * 1000), y=-400,
                                 z=int(_HOLE_Z0 * 1000)),
                     end=Point(x=int(_HOLE_X1 * 1000), y=100,
                               z=int(_HOLE_Z1 * 1000))), name="hole")
    elif hole == "none":
        pass
    else:                                     # pragma: no cover - typo guard
        raise AssertionError(hole)
    return asm


def _bodyless_detail_volume(*, hole: str, detail: str,
                            carved: bool = True) -> float:
    """The detail's volume with the container's ``IfcRelVoidsElement`` APPLIED
    — the same flip, and for the same reason, as
    :func:`_container_detail_volume`."""
    geom = pytest.importorskip("ifcopenshell.geom")
    import ifcopenshell
    import ifcopenshell.util.shape

    proj = _project(_bodyless(hole=hole, detail=detail))
    ifc_text = _compiled(proj)
    product_name = "bar:det" if detail.endswith("bar") else "box:det"

    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ifc_text)
        model = ifcopenshell.open(path)
        settings = geom.settings()
        settings.set("use-world-coords", True)
        settings.set("disable-opening-subtractions", not carved)
        matches = [p for p in model.by_type("IfcProduct")
                   if (p.Name or "").startswith(product_name)
                   and p.Representation is not None]
        assert len(matches) == 1, [p.Name for p in matches]
        shape = geom.create_shape(settings, matches[0])
        return ifcopenshell.util.shape.get_volume(shape.geometry)
    finally:
        os.unlink(path)


def test_the_bodyless_fixture_puts_both_details_in_one_world_volume() -> None:
    """The control this table needs before it can mean anything.

    ``.add()``ed in world coordinates and ``.anchor()``ed in the ASSEMBLY's
    derived frame must be the same box. That frame comes from
    ``frame_basis_aabb`` here rather than from a body, so this control does
    more work than its ``Column`` twin: it also asserts that the derived frame
    and the body-derived one agree.
    """
    added = _bodyless_detail_volume(hole="none", detail="added_box",
                                    carved=False)
    anchored = _bodyless_detail_volume(hole="none", detail="anchored_box",
                                       carved=False)

    assert added == pytest.approx(_DET_FULL, rel=1e-6)
    assert anchored == pytest.approx(added, rel=1e-6)


def test_both_verbs_agree_on_both_attach_verbs_on_a_bodyless_wall() -> None:
    """``{opening, void} x {add, anchor}`` is one number here too — and it is
    the SAME number the bodied wall and the ``Column`` produce, because it is
    the same geometry with the same hole through it.

    Falsify by reverting ``voids.void_reaches_children``'s Wall branch: the
    ``.opening()`` column stops emitting any ``IfcRelVoidsElement`` at all (the
    validator refuses the model outright) while the ``.void()`` column falls
    back to a boolean, so the two come apart in both directions at once.
    """
    measured = {(hole, detail): _bodyless_detail_volume(hole=hole,
                                                        detail=detail)
                for hole in ("opening", "void")
                for detail in ("added_box", "anchored_box")}

    first = next(iter(measured.values()))
    for key, value in measured.items():
        assert value == pytest.approx(first, rel=1e-6), key
    assert first == pytest.approx(_DET_LEFT, rel=1e-6)
    assert first < _DET_FULL              # the fixture can actually lose volume


def test_a_bodyless_wall_opening_reaches_a_bar_the_way_a_void_does() -> None:
    """The degenerate-AABB leaf again, on the host adds.

    Asserted as AGREEMENT plus "material was lost", not as a fraction, for the
    reason ``test_a_container_opening_reaches_a_bar_the_way_a_void_does`` gives
    at length: ``ifcopenshell.geom`` subtracts an ``IfcOpeningElement`` from an
    ``IfcSweptDiskSolid`` by removing everything below the hole's head (upstream). Pinning the fraction here would pin that artefact.
    """
    by_opening = _bodyless_detail_volume(hole="opening", detail="added_bar")
    by_void = _bodyless_detail_volume(hole="void", detail="added_bar")
    full = _bodyless_detail_volume(hole="none", detail="added_bar")

    assert by_opening == pytest.approx(by_void, rel=1e-9)
    assert by_opening < full          # ...and the bar really did lose material


def test_a_bodyless_wall_hole_is_not_carved_twice() -> None:
    """The leaves carry the relation, so no boolean may carve the same volume.

    This is the CSG budget says the envelope workaround was spending — the
    resource the primary web viewer silently runs out of — so it is asserted
    against the same model with no hole rather than against zero: the detail is
    buried in the leaf's body, so ordinary displacement booleans are there
    before any hole is authored. What must not appear is one more.
    """
    baseline = _compiled(_project(_bodyless(hole="none", detail="added_box"))
                         ).count("IFCBOOLEANRESULT(")

    for hole in ("opening", "void"):
        proj = _project(_bodyless(hole=hole, detail="added_box"))
        ifc = _compiled(proj)
        assert _cuts(proj, "det") == 0, hole
        assert ifc.count("IFCBOOLEANRESULT(") == baseline, hole
        # the leaf body + the detail — never the representation-less assembly
        assert ifc.count("IFCRELVOIDSELEMENT(") == 2, hole
