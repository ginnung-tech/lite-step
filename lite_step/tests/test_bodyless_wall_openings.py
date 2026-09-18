"""A body-less ``Wall`` cuts an opening without an envelope body. A body-less aggregator is legal; that taught ``Column``/``Beam``/
``Element`` to cut an opening by handing it to the geometry-bearing leaves it
overlaps. ``Wall`` was left out because it has its own emitter route, and the
consequence was that the moment a layered assembly needed a window the author
had to add an **envelope body** spanning the whole buildup purely so the hole
had somewhere to land.

That envelope is not inert, and the measurement is why this file exists. The
same 6 m wood-framed wall — a layered cladding leaf, a stud bay, a drywall
lining, one ``.opening(Window, along=1500, up=900)`` — is in one of two states,
and **which one depends on how well it is modelled**:

* **under-tiled** — the envelope silently becomes MATERIAL. The part of the stud
  bay the leaves do not occupy is reported as ``IfcWall`` volume, so a quantity
  takeoff counts insulation-grade air as wall. Nothing warns.
* **correctly tiled** — the envelope becomes a ZERO-VERTEX GHOST: an ``IfcWall``
  carrying a representation that ``ifcopenshell.geom`` evaluates to nothing,
  backed by a stack of booleans computing an empty solid, and holding the
  ``IfcRelVoidsElement`` for a window that is therefore hung off a product with
  no geometry.

The better the model, the more degenerate the artefact. That is the wrong
gradient, and both numbers are asserted below on the SAME fixture the fix is
measured against (section 1), so this file cannot pass by measuring a hazard
that was never there.

What replaced it
----------------

Nothing new. ``ifc.voids.void_reaches_children`` — the one landing rule both
verbs and the displacement pass read — now answers per INSTANCE for a ``Wall``,
off ``tx.body_prisms``: a bodied wall keeps its hole on the ``IfcWall`` whose
representation IS its body, and a body-less one distributes to its leaves
through the very functions a ``Column``'s hole goes through
(``generator._process_container_voids`` / ``_process_container_openings``).
N ``IfcOpeningElement``, exactly one ``IfcWindow`` — ``IfcElement.FillsVoids``
is [0:1], pinned in ``test_opening_distributes_to_leaves.py``.

The trap this file is written around
------------------------------------

``along=`` is measured in the HOST's frame, whose ``+x`` routinely opposes world
``+x``. On this 6 m assembly ``along=1500 width=1200`` lands at world
``x[3.3, 4.5]``, NOT ``x[1.5, 2.7]``. Both died on that
confound. Every world number here is DERIVED from ``frames.opening_void_prism``
in ``test_the_hole_lands_where_the_host_frame_says`` and read back from the
emitted ``IfcOpeningElement.ObjectPlacement``, never restated from the scalars.

The fixture stacks its buildup **inner-first**, so "the outermost leaf" and
"the first leaf" are different products and the fill rule can actually fail —
the convention ``test_opening_distributes_to_leaves.py`` established. It also
puts the cladding on the ``+y`` side, which is where this assembly's DERIVED
outward normal points (asserted in
``test_the_assemblys_outward_normal_is_derived_not_assumed``): a fixture whose
leaf named "cladding" sat on the inward face would be testing the fill rule
against a reader's intuition instead of against the frame.

Coverage
--------

1.  The two failure modes, on the envelope-bodied workaround — the phantom
    volume and the zero-vertex ghost.
2.  The acceptance case: no envelope, one window, one fill, one
    ``IfcRelVoidsElement`` per overlapping leaf, and no ``IfcWall`` in the file
    with an empty representation.
3.  The N holes are ONE hole — same world placement, read off the file.
4.  Every overlapping leaf really lost the material; every missed one kept it.
5.  The LAYERED leaf keeps its slices and its ``IfcMaterialLayerSetUsage``.
6.  The fill lands on the outermost leaf, not the first-added one.
7.  Both backends, byte-identical counts, and the CSG budget the envelope was
    spending.
8.  The refusal NARROWED: an empty basis still refuses, a bodied Wall is
    untouched.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Box, Column, Element, LayerSet, Material, Point, Project, Wall, Window,
)

# ---------------------------------------------------------------------------
# The fixture: a 6 m wood-framed wall, 306 mm overall, built as three leaves
# and NO envelope. Stacked INNER-FIRST so "outermost" != "first added":
#
#     y[  0,  13]  lining     13 mm  Wall + Box            (inner face)
#     y[ 13, 158]  stud bay  145 mm  Column of 45 mm studs
#     y[158, 306]  cladding  148 mm  Wall + Box + LayerSet (outer face)
#
# The cladding's LayerSet is plank / cavity / rigid / membrane = 22+25+80+21,
# which is 148 — so the leaf's own body and its buildup agree, and the cavity
# layer means its rendered volume is NOT its bounding box.
# ---------------------------------------------------------------------------

_RUN, _H = 6000, 3000
_LINING = (0, 13)
_BAY = (13, 158)
_CLAD = (158, 306)

#: 10 studs at 600 c/c, 45 mm wide, from x = 0. Without batts the bay is
#: deliberately UNDER-TILED — the air between the studs is what the envelope
#: workaround reports as wall. With them it tiles the full 6 m exactly, which
#: is what a real wall is and what turns the envelope into a ghost.
_STUD_N, _STUD_PITCH, _STUD_W, _STUD_X0 = 10, 600, 45, 0

#: Where the hole actually lands in WORLD coordinates — host-frame ``along=``,
#: not world x. Derived in ``test_the_hole_lands_where_the_host_frame_says``.
_HOLE_X0, _HOLE_X1 = 3.3, 4.5
_HOLE_Z0, _HOLE_Z1 = 0.9, 2.3
_HOLE_AREA = (_HOLE_X1 - _HOLE_X0) * (_HOLE_Z1 - _HOLE_Z0)

#: The two studs the hole crosses, out of ten. The other eight are the control
#: that keeps "distributes to what it overlaps" distinguishable from "attaches
#: to everything".
_HIT_STUDS = ("stud6", "stud7")


def _box(name: str, y0: int, y1: int, x0: int = 0, x1: int = _RUN) -> Box:
    return Box(name=name, type="wall",
               start=Point(x=x0, y=y0, z=0), end=Point(x=x1, y=y1, z=_H))


def _cladding() -> Wall:
    leaf = Wall(name="cladding")
    leaf.add(_box("clad_b", *_CLAD))
    leaf.layers = LayerSet(outward=(0, 1, 0), layers=[
        Material(key="Chipboard_P6", thickness_mm=22),
        Material(key="Cavity_Ventilated", thickness_mm=25),
        Material(key="MineralWool_Roof38", thickness_mm=80),
        Material(key="Underlay_Membrane", thickness_mm=21),
    ])
    return leaf


def _stud_bay(*, batts: bool) -> Column:
    """The stud bay. ``batts=True`` fills the gaps, which is what a real wall
    is — and what turns the envelope from phantom material into a ghost."""
    bay = Column(name="studbay")
    for i in range(_STUD_N):
        x0 = _STUD_X0 + i * _STUD_PITCH
        bay.add(_box(f"stud{i}", *_BAY, x0=x0, x1=x0 + _STUD_W))
    if batts:
        for i in range(_STUD_N):
            x0 = _STUD_X0 + i * _STUD_PITCH + _STUD_W
            batt = Element(ifc_class="IfcCovering", name=f"batt{i}")
            batt.add(_box("b", *_BAY, x0=x0,
                          x1=_STUD_X0 + (i + 1) * _STUD_PITCH))
            bay.add(batt)
    return bay


def _lining() -> Wall:
    leaf = Wall(name="lining")
    leaf.add(_box("lin_b", *_LINING))
    return leaf


def _assembly(*, envelope: bool = False, batts: bool = False,
              hole: str = "opening") -> Wall:
    """the fixture. ``envelope=True`` is the workaround it measures.

    ``hole="void"`` cuts the box the OPENING derives (``_HOLE_*``), never the
    box ``along=1500`` reads like — see the module docstring.
    """
    asm = Wall(name="south")
    if envelope:
        # The solid the author did not want: one prism spanning the whole
        # buildup, which then overlaps every leaf it contains.
        asm.add(_box("env", _LINING[0], _CLAD[1]))
    asm.add(_lining())                       # inner-first, on purpose
    asm.add(_stud_bay(batts=batts))
    asm.add(_cladding())                     # ...so this one is the outermost

    if hole == "opening":
        asm.opening(Window(name="g0", width=1200, height=1400),
                    along=1500, up=900)
    elif hole == "void":
        # The box the OPENING derives, in absolute world coordinates, cut
        # right through the buildup — never the box along=1500 reads like.
        asm.void(Box(start=Point(x=int(_HOLE_X0 * 1000), y=_LINING[0] - 100,
                                 z=int(_HOLE_Z0 * 1000)),
                     end=Point(x=int(_HOLE_X1 * 1000), y=_CLAD[1] + 100,
                               z=int(_HOLE_Z1 * 1000))), name="hole")
    elif hole == "none":
        pass
    else:                                     # pragma: no cover - typo guard
        raise AssertionError(hole)
    return asm


def _project(element) -> Project:
    proj = Project(name="bodyless-wall-openings")
    proj.add(element)
    return proj


def _errors(element) -> list:
    return validate_project_report(_project(element)).errors


def _compiled(element, backend: str = "ifcopenshell",
              expect_errors: bool = False) -> str:
    """``expect_errors`` is for the ENVELOPE pathology tests below.

    Authoring an envelope over a buildup is now a compile error
    (``executor._envelope_over_buildup_errors``), so those tests cannot
    reach the geometry through the public path — which is the entire point of
    the refusal. Their value is the EVIDENCE for it (phantom volume, the
    zero-vertex ghost), so they keep measuring by going around validation
    rather than being deleted. Every other caller still asserts a clean
    report.
    """
    proj = _project(element)
    errors = validate_project_report(proj).errors
    if expect_errors:
        assert any("envelope" in e for e in errors), (
            f"the envelope must be REFUSED; the measurement below is why: {errors}")
    else:
        assert errors == []
    result = generate_ifc(normalize_project_to_meters(proj) or proj)
    assert result.success, result.error
    return result.ifc_content


def _counts(ifc: str) -> dict:
    """RELATION counts by name. A product count passes on several wrong
    implementations — a second window emitted and orphaned, one opening reused
    across leaves — because it is the ``IfcRelVoidsElement`` /
    ``IfcRelFillsElement`` rows that say where a hole actually landed."""
    return {k: ifc.count(f"{k}(") for k in
            ("IFCOPENINGELEMENT", "IFCWINDOW", "IFCRELVOIDSELEMENT",
             "IFCRELFILLSELEMENT", "IFCBOOLEANRESULT", "IFCWALL")}


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


def _shapes(ifc_text: str, *, carved: bool) -> dict:
    """``{leaf name: (vertex count, volume)}`` for every product with a body.

    ``carved=True`` lets ``ifcopenshell.geom`` apply each product's
    ``IfcRelVoidsElement``, which is the whole subject here: a distributed
    opening carves through a RELATION and not through a boolean, so measuring
    with subtractions disabled would measure exactly the thing this feature
    does not do. ``carved=False`` is the uncarved control, and the PAIR is what
    makes "the leaf really lost the hole" a measurement rather than a count.

    A product whose representation tessellates to nothing is reported as
    ``(0, 0.0)`` rather than skipped — that is the ghost, and it has to be
    visible to a test.
    """
    geom = pytest.importorskip("ifcopenshell.geom")
    import ifcopenshell.util.shape

    settings = geom.settings()
    settings.set("use-world-coords", True)
    settings.set("disable-opening-subtractions", not carved)
    out = {}
    for product in _model(ifc_text).by_type("IfcProduct"):
        if product.is_a("IfcAnnotation") or product.is_a("IfcOpeningElement"):
            continue
        if product.Representation is None:
            continue
        shape = geom.create_shape(settings, product)
        verts = len(shape.geometry.verts) // 3
        out[_leaf(product.Name)] = (
            verts,
            ifcopenshell.util.shape.get_volume(shape.geometry) if verts
            else 0.0)
    return out


def _leaf(canonical) -> str:
    """``box:stud6:column:studbay:wall:south`` -> ``stud6``; the canonical's
    own leaf, which is what the fixture names things. A name with no type
    prefix (the ``IfcBuilding``) answers itself."""
    if not canonical:
        return "<anonymous>"
    parts = canonical.split(":")
    return parts[1] if len(parts) > 1 else parts[0]


def _representationless(ifc_text: str) -> set:
    return {_leaf(p.Name) for p in _model(ifc_text).by_type("IfcProduct")
            if p.Representation is None and not p.is_a("IfcSpatialElement")}


def _voided_leaves(ifc_text: str) -> list:
    return sorted(_leaf(r.RelatingBuildingElement.Name)
                  for r in _model(ifc_text).by_type("IfcRelVoidsElement"))


def _fill_host(ifc_text: str):
    """``(host leaf, fill class, fill name)`` for the ONE fill in the file."""
    model = _model(ifc_text)
    fills = model.by_type("IfcRelFillsElement")
    assert len(fills) == 1, [f.RelatedBuildingElement.Name for f in fills]
    host = next(v.RelatingBuildingElement
                for v in model.by_type("IfcRelVoidsElement")
                if v.RelatedOpeningElement == fills[0].RelatingOpeningElement)
    fill = fills[0].RelatedBuildingElement
    return _leaf(host.Name), fill.is_a(), fill.Name


def _opening_placements(ifc_text: str) -> list:
    import ifcopenshell.util.placement as placement

    out = []
    for opening in _model(ifc_text).by_type("IfcOpeningElement"):
        matrix = placement.get_local_placement(opening.ObjectPlacement)
        out.append(tuple(round(float(v), 6) for v in matrix[:3, 3]))
    return out


# ---------------------------------------------------------------------------
# 1. What the envelope workaround costs — the two measurements
# ---------------------------------------------------------------------------


def test_the_envelope_workaround_reports_phantom_wall_volume() -> None:
    """Under-tiled, the solid the author added for the hole becomes MATERIAL.

    The envelope spans the whole 6.0 x 0.306 x 3.0 buildup and the generator
    subtracts the leaves it contains out of it. What survives is the air
    between the studs — reported as ``IfcWall`` volume, so a takeoff counts it
    as wall, and nothing warns.

    This is the number the fix has to remove, so it is asserted rather than
    quoted. If it ever comes back as zero the workaround stopped being
    hazardous and this file's premise needs re-reading.
    """
    # The envelope prism IS the ``IfcWall``'s own representation
    # (``_create_wall`` builds the wall out of it), so it is measured as
    # ``wall:south`` — which is exactly why its residue is reported as WALL.
    shapes = _shapes(_compiled(_assembly(envelope=True), expect_errors=True), carved=False)
    verts, volume = shapes["south"]

    assert verts > 0, "under-tiled, the envelope is not yet a ghost"
    # Nominal 6.000 x 0.306 x 3.000 = 5.508 m3; what is left is the un-tiled
    # part of the stud bay, i.e. material the model does not contain.
    assert 1.5 < volume < 3.0, volume
    assert volume < 5.508


def test_the_envelope_workaround_becomes_a_zero_vertex_ghost_when_tiled() -> None:
    """Tiled the way a real wall is, the same solid evaluates to NOTHING.

    Studs at 600 c/c with batts between them leave the envelope no residual
    volume at all — so it is an ``IfcWall`` carrying a representation that
    tessellates to zero vertices, backed by a stack of booleans computing an
    empty solid, and it is the product holding the window's
    ``IfcRelVoidsElement``.

    The two tests together are the gradient objects to: the better the
    model, the more degenerate the artefact.
    """
    ifc = _compiled(_assembly(envelope=True, batts=True), expect_errors=True)
    verts, volume = _shapes(ifc, carved=False)["south"]

    assert verts == 0, "a correctly tiled envelope should evaluate to nothing"
    assert volume == 0.0
    # ...and it is not merely empty, it is expensive: the booleans are real.
    assert ifc.count("IFCBOOLEANRESULT(") > 20

    # The window hangs off that ghost — one IfcRelVoidsElement, on the product
    # with no geometry.
    assert _voided_leaves(ifc) == ["south"]


# ---------------------------------------------------------------------------
# 2. The acceptance case: the same wall, with no envelope at all
# ---------------------------------------------------------------------------


def test_a_bodyless_wall_with_an_opening_compiles() -> None:
    """The refusal this issue is about is gone, and gone for both spellings.

    ``.opening()`` on the assembly and the ``.anchor()`` twin it routes to are
    one code path (``BimElement.opening``), but the validator walks
    ``_elements`` and ``_openings`` separately, so both are asserted.
    """
    assert _errors(_assembly()) == []

    twin = _assembly(hole="none")
    twin.anchor(Window(name="g0", width=1200, height=1400),
                along=1500, up=900)
    assert _errors(twin) == []


def test_one_window_one_fill_and_one_hole_per_overlapping_leaf() -> None:
    """The acceptance assertion, by RELATION name.

    Four leaves stand in the hole — the cladding, the lining and the two studs
    it crosses — so four ``IfcOpeningElement``/``IfcRelVoidsElement`` pairs and
    exactly ONE ``IfcWindow`` filling one of them (``IfcElement.FillsVoids`` is
    [0:1]; pinned in ``test_opening_distributes_to_leaves.py``). The eight
    studs the hole misses are the control: a version that attached the opening
    to every child product would read fourteen here.

    And no booleans. The distribution adds RELATIONS — the 20-plus
    ``IfcBooleanResult`` the envelope spent are simply not emitted.
    """
    ifc = _compiled(_assembly())

    assert _counts(ifc) == {
        "IFCOPENINGELEMENT": 4, "IFCWINDOW": 1, "IFCRELVOIDSELEMENT": 4,
        "IFCRELFILLSELEMENT": 1, "IFCBOOLEANRESULT": 0,
        # the assembly + the two Wall leaves; there is no fourth
        "IFCWALL": 3,
    }
    assert _voided_leaves(ifc) == sorted(
        ["cladding", "lining", *_HIT_STUDS])


def test_no_wall_in_the_file_is_an_empty_representation() -> None:
    """the headline, stated as the property it wants.

    The assembly emits an ``IfcWall`` with ``Representation=None`` — an
    identity aggregating real parts, which is a coherent thing for a viewer and
    a takeoff to meet. What must NOT be in the file is the other shape: a wall
    carrying a representation that evaluates to nothing, or one carrying
    phantom residual volume.

    Both tiling states are checked, because the two failure modes are reached
    by different models and a fix for one is not a fix for the other.
    """
    for batts in (False, True):
        ifc = _compiled(_assembly(batts=batts))
        assert "south" in _representationless(ifc), batts

        for name, (verts, volume) in _shapes(ifc, carved=False).items():
            assert verts > 0, (name, batts)      # no ghost
            assert volume > 0.0, (name, batts)   # no empty solid
        # ...and the assembly itself has no body to be either of them.
        assert "south" not in _shapes(ifc, carved=False)


def test_the_hole_lands_where_the_host_frame_says() -> None:
    """``along=`` is HOST-frame, and this host's run axis opposes world +X.

    both were both filed on this confound and both closed invalid.
    ``along=1500`` on a 6 m assembly puts the hole at world ``x[3.3, 4.5]``, so
    a ``.void(Box(x=1500..2700))`` meant as "the same hole" would cut 1.8 m
    away. The ``hole="void"`` fixture derives its box from the numbers asserted
    here, and so does ``_HIT_STUDS``.
    """
    from lite_step.compiler import frames
    from lite_step.ifc.entity_cache import snap_to_precision

    proj = _project(_assembly())
    assert validate_project_report(proj).errors == []
    normalized = normalize_project_to_meters(proj) or proj
    assert generate_ifc(normalized).success       # runs the frame passes

    asm = normalized.storeys[0].elements[0]
    frame = frames.opening_host_frame(asm, snap=snap_to_precision,
                                      divisor=1000.0)
    assert frame is not None, "the container frame derives"
    window = next(c for c in asm._elements if isinstance(c, Window))
    (x0, _y0, z0), (x1, _y1, z1) = frames.opening_void_prism(
        window, frame).aabb()

    assert (x0, x1) == pytest.approx((_HOLE_X0, _HOLE_X1))
    assert (z0, z1) == pytest.approx((_HOLE_Z0, _HOLE_Z1))


def test_the_four_holes_are_one_hole() -> None:
    """N ``IfcOpeningElement`` at ONE world placement, read off the file.

    The distribution's correctness claim is "one authored hole, N relations",
    and the failure it has to exclude is N holes in N places — which every
    count-based assertion above passes happily. Placements are resolved through
    the full ``IfcLocalPlacement`` chain, because each opening is emitted
    relative to a different host product.
    """
    placements = _opening_placements(_compiled(_assembly()))

    assert len(placements) == 4
    assert len(set(placements)) == 1, placements
    # ...and it is the hole the frame derived: centred in plan, at the sill.
    (x, _y, z), = set(placements)
    assert x == pytest.approx((_HOLE_X0 + _HOLE_X1) / 2)
    assert z == pytest.approx(_HOLE_Z0)


# ---------------------------------------------------------------------------
# 3. The leaves really lost the material — and only the ones they should
# ---------------------------------------------------------------------------


def test_every_overlapping_leaf_loses_exactly_the_hole() -> None:
    """The carve, per leaf, as a volume — not as a relation count.

    Each leaf loses the hole's face area times ITS OWN depth, which is a
    different number per leaf and therefore a real measurement. The cladding's
    depth is its buildup MINUS the cavity layer, which is why its uncarved
    volume is not its bounding box either.
    """
    ifc = _compiled(_assembly())
    plain = _shapes(ifc, carved=False)
    carved = _shapes(ifc, carved=True)

    depths = {"lining": 0.013,
              "cladding": (148 - 25) / 1000.0,      # the cavity is an air gap
              **{s: 0.145 for s in _HIT_STUDS}}
    for name, depth in depths.items():
        lost = plain[name][1] - carved[name][1]
        width = _STUD_W / 1000.0 if name.startswith("stud") else None
        expected = _HOLE_AREA * depth if width is None else (
            (_HOLE_Z1 - _HOLE_Z0) * width * depth)
        assert lost == pytest.approx(expected, rel=1e-3), name


def test_the_studs_the_hole_misses_keep_every_bit_of_theirs() -> None:
    """The control. Eight of the ten studs are clear of the hole, and a
    distribution that attached to everything would be invisible without them —
    a non-overlapping ``IfcRelVoidsElement`` carves nothing, so the volumes
    would still be right and only the entity count would be wrong."""
    ifc = _compiled(_assembly())
    plain = _shapes(ifc, carved=False)
    carved = _shapes(ifc, carved=True)

    missed = [f"stud{i}" for i in range(_STUD_N)
              if f"stud{i}" not in _HIT_STUDS]
    assert len(missed) == 8
    for name in missed:
        assert carved[name][1] == pytest.approx(plain[name][1], rel=1e-9), name


def test_the_layered_leaf_keeps_its_slices_and_its_layer_set() -> None:
    """The case most likely to break: "the body" is not one object there.

    A ``LayerSet`` emits per-layer SLICES in place of the body child, so the
    cladding leaf's ``IfcWall`` carries four-minus-one items (the cavity is an
    air gap) on ONE representation. That is one shape slot, so it is one
    distribution target — and its single ``IfcRelVoidsElement`` has to carve
    every slice, not the first.

    Measured as: the leaf lost the hole through its FULL solid depth, and the
    ``IfcMaterialLayerSetUsage`` chain the buildup rides is still attached.
    """
    ifc = _compiled(_assembly())
    model = _model(ifc)

    clad = next(p for p in model.by_type("IfcWall")
                if _leaf(p.Name) == "cladding")
    items = clad.Representation.Representations[0].Items
    assert len(items) == 3, "plank + rigid + membrane; the cavity is air"

    assert model.by_type("IfcMaterialLayerSetUsage"), "the buildup chain"

    plain = _shapes(ifc, carved=False)["cladding"][1]
    carved = _shapes(ifc, carved=True)["cladding"][1]
    assert plain - carved == pytest.approx(
        _HOLE_AREA * (148 - 25) / 1000.0, rel=1e-3)


# ---------------------------------------------------------------------------
# 4. Which leaf carries the ONE fill
# ---------------------------------------------------------------------------


def test_the_assemblys_outward_normal_is_derived_not_assumed() -> None:
    """The premise the fill rule is measured against, stated as a number.

    ``_outermost_host_index`` measures along the host frame's OUTWARD normal.
    This fixture puts the cladding on ``+y`` because that is where this
    assembly's derived normal points — so if the facing rule ever changes, this
    fails HERE with one clear message instead of moving the fill and failing
    the next test with a confusing one.
    """
    from lite_step.compiler import frames
    from lite_step.ifc.entity_cache import snap_to_precision

    proj = normalize_project_to_meters(_project(_assembly())) or None
    assert proj is not None
    assert generate_ifc(proj).success
    frame = frames.opening_host_frame(proj.storeys[0].elements[0],
                                      snap=snap_to_precision, divisor=1000.0)

    assert frame.inward == pytest.approx((0.0, -1.0)), (
        "outward is +y, so the cladding at y[158, 306] is the outer leaf")


def test_the_one_fill_sits_on_the_outermost_leaf_not_the_first() -> None:
    """The window is in the CLADDING, and the lining was added first.

    Three reasons, stated in ``generator._outermost_host_index``: it is where
    the window physically is; ``inset=`` is measured from the OUTER FACE, so
    attaching the fill to the lining would put a flush window on a product two
    layers behind the facade it is flush with; and it is stable under editing.

    Falsified by the reordering below: moving the cladding to the FRONT of the
    add order must not move the fill, because the rule is geometric and not
    positional.
    """
    host, cls, name = _fill_host(_compiled(_assembly()))
    assert (host, cls) == ("cladding", "IfcWindow")
    assert name == "window:g0:wall:south"

    outer_first = Wall(name="south")
    outer_first.add(_cladding())
    outer_first.add(_stud_bay(batts=False))
    outer_first.add(_lining())
    outer_first.opening(Window(name="g0", width=1200, height=1400),
                        along=1500, up=900)
    assert _fill_host(_compiled(outer_first))[0] == "cladding"


def test_the_fill_is_aggregated_to_the_assembly_an_author_reads() -> None:
    """The hole hangs off one leaf; the WINDOW belongs to the wall.

    Same line the bodied Wall route ends on. Without it the window is reachable
    in a viewer's tree only under whichever leaf happened to be outermost,
    which is an implementation detail leaking into the model tree.
    """
    model = _model(_compiled(_assembly()))
    under_asm = {rel.RelatedObjects for rel in model.by_type("IfcRelAggregates")
                 if _leaf(rel.RelatingObject.Name) == "south"}
    names = {o.Name for group in under_asm for o in group}

    assert "window:g0:wall:south" in names
    assert {"wall:cladding:wall:south", "wall:lining:wall:south",
            "column:studbay:wall:south"} <= names


# ---------------------------------------------------------------------------
# 5. The budget the envelope was spending
# ---------------------------------------------------------------------------


def test_the_envelope_was_the_thing_spending_the_csg_budget() -> None:
    """The cost, as the number ``compiler.csg_depth`` reports. Measured max=5 of a budget of 10 on a 14-child assembly before the
    studs were even tiled, and every boolean of it was the envelope being
    hollowed out by the leaves it contains. With no envelope there is nothing
    to hollow: no displacement carve pairs, and no ``IfcBooleanResult``.
    """
    from lite_step.compiler.csg_depth import predict_csg_depth

    with_envelope = normalize_project_to_meters(
        _project(_assembly(envelope=True))) or None
    assert generate_ifc(with_envelope).success
    envelope_depth = predict_csg_depth(with_envelope).max_depth

    without = normalize_project_to_meters(_project(_assembly())) or None
    assert generate_ifc(without).success

    assert envelope_depth >= 5, envelope_depth
    assert predict_csg_depth(without).max_depth == 0
    assert _compiled(_assembly()).count("IFCBOOLEANRESULT(") == 0


def test_the_tree_print_says_the_assembly_has_no_body() -> None:
    """the print is what an author reads when a hole lands somewhere
    surprising, so it has to name the shape this feature introduces: a
    container with an authored opening and no body of its own."""
    from lite_step.compiler.tree_report import tree_report_lines

    proj = normalize_project_to_meters(_project(_assembly())) or None
    text = "\n".join(tree_report_lines(proj))

    line = next(ln for ln in text.splitlines() if "wall:south" in ln)
    assert "openings=1" in line
    assert "(no body)" in line


# ---------------------------------------------------------------------------
# 6. The refusal NARROWED — it did not disappear
# ---------------------------------------------------------------------------


def test_a_bodyless_wall_with_nothing_to_derive_a_frame_from_still_refuses() -> None:
    """No body AND no frame basis: there is no run axis for ``along=`` to walk
    and no leaf for the hole to land on, so it is still a compile error and the
    message still names both remedies.

    This is the half of the refusal that has to survive. It is reached from
    ``_emits_openings``, which returns ``has_frame_basis`` for a distributing
    host — the same call ``_bodyless_container_errors`` makes, so the two
    cannot disagree about what "has geometry" means.
    """
    empty = Wall(name="south")
    empty.anchor(Box(name="ledge", start=Point(x=0, y=0, z=0),
                     end=Point(x=200, y=200, z=200)), along=0, up=0)
    empty.opening(Window(name="g0", width=1200, height=1400),
                  along=1500, up=900)

    errors = _errors(empty)
    assert errors, "an empty basis must still refuse"
    assert any("no .add()ed children to derive a frame from" in e
               for e in errors), errors


def test_a_bodied_wall_is_untouched_and_keeps_its_hole_on_the_ifcwall() -> None:
    """The other half. A ``Wall`` WITH a body carries the hole on its own
    product, because that product's representation IS the body — one
    ``IfcOpeningElement``, on the ``IfcWall``, exactly as before.

    ``tx.body_prisms`` is the one call that tells the two walls apart, so this
    is the test that fails if the new rule ever widens to every Wall.
    """
    from lite_step.ifc.voids import void_reaches_children

    bodied = Wall(name="south")
    bodied.add(_box("body", _LINING[0], _CLAD[1]))
    bodied.opening(Window(name="g0", width=1200, height=1400),
                   along=1500, up=900)

    assert void_reaches_children(bodied) is False
    assert void_reaches_children(_assembly(hole="none")) is True

    ifc = _compiled(bodied)
    assert _counts(ifc)["IFCOPENINGELEMENT"] == 1
    assert _voided_leaves(ifc) == ["south"]


def test_a_bodyless_wall_void_and_opening_land_on_the_same_leaves() -> None:
    """the invariant, extended onto this host rather than restarted.

    ``.void()`` and ``.opening()`` are two rows of one table, and neither
    difference between them is about which product carries the hole. On a
    body-less Wall they now go through ONE landing rule
    (``voids.hole_targets``), so the set of voided leaves is the same set — and
    the ``.void()`` spelling cuts the box the OPENING derives, never the box
    ``along=1500`` reads like.
    """
    by_opening = _compiled(_assembly(hole="opening"))
    by_void = _compiled(_assembly(hole="void"))

    assert _voided_leaves(by_void) == _voided_leaves(by_opening)
    # ...and only the opening brings a fill. That IS the difference.
    assert _counts(by_void)["IFCRELFILLSELEMENT"] == 0
    assert _counts(by_opening)["IFCRELFILLSELEMENT"] == 1

    opened = _shapes(by_opening, carved=True)
    voided = _shapes(by_void, carved=True)
    for name in ("cladding", "lining", *_HIT_STUDS):
        assert voided[name][1] == pytest.approx(opened[name][1], rel=1e-6), name
