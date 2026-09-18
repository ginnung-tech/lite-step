"""A physical container needs a derivable FRAME, not a body of its own.

The aggregate-parent shape: a container that POSITIONS its children and
emits no solid. Until now every `Wall` had to carry a body, so the "advanced
wall"  had to author an envelope solid it did not want — one that the
generator then subtracts its own leaves out of, leaving an `IfcBooleanResult`
that tessellates to nothing (measured below, test 2).

Nothing about the frame had to be invented for this. Since
``frames.frame_basis_points`` is *own points plus every descendant not anchored
into me*, so ``.add()``ed leaves already supply thickness, run axis, extents and
facing — they are in world coordinates by definition, which is exactly
what a basis needs. One validator rule was the whole blocker, and it asked the
wrong question.

Pinned here:

1.  The frame of a body-less aggregator is DERIVED, and its thickness is the
    sum of the leaves it aggregates.
2.  It compiles, and its leaves land at the same world coordinates and volumes
    as the envelope-bodied workaround — which the body-less shape strictly
    improves on: no empty boolean husk, one product fewer.
3.  The issue's headline: ``.anchor()`` into a body-less container places the
    child in the derived frame AND leaves it its own product identity.
4.  An EMPTY basis is still refused, and the two ways to reach one — nothing at
    all, and only ``.anchor()``ed children — are reported differently.
5.  ``frames._bake_child``'s refusal is the backstop for the containers that
    have no compile-time body check at all, and it still fires.
6.  An anchored prism is a detail, never the body — the validator and BOTH IFC
    backends read one definition of that now (``tx.body_prisms``).
7.  A body-less wall does not stream: the streaming handler builds the
    ``IfcWall`` out of a body prism and drops the whole wall without one.
8.  An ``.opening()`` on a body-less container is DISTRIBUTED to the leaves it
    overlaps — refusing it is what forces the envelope workaround test 2
        measures. An empty basis still refuses.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from lite_step.compiler import frames
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Box, Mesh, Point, Project, Wall, Window,
)
from lite_step.models import taxonomy as tx

# ---------------------------------------------------------------------------
# The fixture: one 6 m wall running +X, built as TWO leaves and no envelope.
# Core y[0, 180], facade y[180, 508] — a 508 mm buildup, which is the number
# the aggregator has to derive without a body of its own.
# ---------------------------------------------------------------------------

_CORE_MM = (0, 180)
_FACADE_MM = (180, 508)
_RUN_MM = 6000
_HEIGHT_MM = 2700

_THICKNESS_M = 0.508
_RUN_M = 6.0
_HEIGHT_M = 2.7
#: 6.0 x 0.180 x 2.7 and 6.0 x 0.328 x 2.7 — the leaves, in cubic meters.
_CORE_VOLUME = 2.916
_FACADE_VOLUME = 5.3136


def _leaf(name: str, y0: int, y1: int) -> Wall:
    leaf = Wall(name=name)
    leaf.add(Box(name=f"{name}_body", type="wall",
                 start=Point(x=0, y=y0, z=0),
                 end=Point(x=_RUN_MM, y=y1, z=_HEIGHT_MM)))
    return leaf


def _aggregator(*, envelope: bool = False) -> Wall:
    """The fixture wall. ``envelope=True`` is the workaround, for test 2."""
    asm = Wall(name="asm")
    if envelope:
        asm.add(Box(name="env", type="wall",
                    start=Point(x=0, y=_CORE_MM[0], z=0),
                    end=Point(x=_RUN_MM, y=_FACADE_MM[1], z=_HEIGHT_MM)))
    asm.add(_leaf("core", *_CORE_MM))
    asm.add(_leaf("facade", *_FACADE_MM))
    return asm


def _project(element, expect_clean: bool = True) -> Project:
    proj = Project(name="bodyless-container")
    proj.add(element)
    if expect_clean:
        assert validate_project_report(proj).errors == []
    return normalize_project_to_meters(proj) or proj


def _errors(element) -> list:
    proj = Project(name="bodyless-container")
    proj.add(element)
    return validate_project_report(proj).errors


def _compiled(proj: Project, backend: str = "ifcopenshell") -> str:
    result = generate_ifc(proj)
    assert result.success, result.error
    return result.ifc_content


def _stamped(proj: Project) -> Project:
    """Run the passes ``generate_ifc`` runs, without emitting."""
    frames.stamp_frames(proj)
    frames.resolve_child_anchors(proj)
    frames.stamp_frames(proj)
    return proj


def _measure(ifc_text: str):
    """``{name: (aabb, volume, has_representation)}`` in WORLD coordinates.

    ``disable-opening-subtractions`` for the reason
    ``test_void_carves_details._volume`` gives at length: ``ifcopenshell.geom``
    applies a parent's ``IfcRelVoidsElement`` to the products aggregated under
    it, so a measurement taken without it can answer about a hole nobody cut
    here.
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
        out = {}
        for product in model.by_type("IfcProduct"):
            if product.is_a("IfcAnnotation"):
                continue
            name = product.Name or ""
            if product.Representation is None:
                out[name] = (None, 0.0, False)
                continue
            shape = geom.create_shape(settings, product)
            verts = shape.geometry.verts
            if not verts:
                out[name] = (None, 0.0, True)
                continue
            aabb = ((min(verts[0::3]), min(verts[1::3]), min(verts[2::3])),
                    (max(verts[0::3]), max(verts[1::3]), max(verts[2::3])))
            out[name] = (aabb,
                         ifcopenshell.util.shape.get_volume(shape.geometry),
                         True)
        return out
    finally:
        os.unlink(path)


def _named(measured: dict, prefix: str):
    matches = [k for k in measured if k.startswith(prefix)]
    assert len(matches) == 1, (prefix, sorted(measured))
    return measured[matches[0]]


def _aggregated_under(ifc_text: str, parent_prefix: str) -> list:
    import ifcopenshell

    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ifc_text)
        model = ifcopenshell.open(path)
        for rel in model.by_type("IfcRelAggregates"):
            if (rel.RelatingObject.Name or "").startswith(parent_prefix):
                return sorted(o.Name for o in rel.RelatedObjects)
        return []
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 1. The frame is already derived — the premise the whole change rests on
# ---------------------------------------------------------------------------


def test_a_bodyless_aggregators_thickness_is_the_sum_of_its_leaves() -> None:
    """No body, and every number a frame carries is nonetheless right.

    Thickness is 180 + 328, not 180 (the first leaf) and not 0 (no body). If
    this ever reads 0.18 the basis has stopped reaching ``.add()``ed children
    and the widened validator would be handing out frames derived from one
    leaf.
    """
    proj = _stamped(_project(_aggregator()))
    frame = proj.storeys[0].elements[0]._frame

    assert frame is not None
    assert frame.thickness == pytest.approx(_THICKNESS_M)
    assert frame.extent_along == pytest.approx(_RUN_M)
    assert frame.height == pytest.approx(_HEIGHT_M)
    assert frame.along == (1.0, 0.0, 0.0)
    assert frame.across == (0.0, 1.0, 0.0)
    assert frame.origin == pytest.approx((0.0, 0.0, 0.0))
    assert frame.facing == 1


def test_the_basis_is_what_answers_and_an_anchored_child_is_not_in_it() -> None:
    """``has_frame_basis`` is the widened rule, and it is the rule.

    An aggregator with two ``.add()``ed leaves has a basis; the same container
    with the same two children ``.anchor()``ed has none, because an anchored
    child is placed BY the frame and so cannot define it.
    """
    assert frames.has_frame_basis(_aggregator()) is True

    anchored = Wall(name="asm")
    anchored.anchor(_leaf("core", *_CORE_MM), along=0, up=0)
    anchored.anchor(_leaf("facade", *_FACADE_MM), along=0, up=0)
    assert frames.has_frame_basis(anchored) is False
    assert frames.has_frame_basis(Wall(name="empty")) is False


# ---------------------------------------------------------------------------
# 2. It compiles, and it beats the workaround it replaces
# ---------------------------------------------------------------------------


def test_the_leaves_land_where_the_envelope_workaround_puts_them() -> None:
    """The claim that makes this safe: dropping the envelope moves nothing.

    Same two leaves, same two world AABBs, same two volumes — with and without
    the envelope solid had to author. Compared as numbers rather than as
    a hash, because the two files legitimately differ by the envelope product.
    """
    bodyless = _measure(_compiled(_project(_aggregator())))
    workaround = _measure(_compiled(_project(_aggregator(envelope=True), expect_clean=False)))

    for prefix, volume, y_span in (
        ("wall:core", _CORE_VOLUME, (0.0, 0.180)),
        ("wall:facade", _FACADE_VOLUME, (0.180, 0.508)),
    ):
        aabb_a, vol_a, _ = _named(bodyless, prefix)
        aabb_b, vol_b, _ = _named(workaround, prefix)
        assert aabb_a[0] == pytest.approx(aabb_b[0])
        assert aabb_a[1] == pytest.approx(aabb_b[1])
        assert vol_a == pytest.approx(vol_b) == pytest.approx(volume)
        assert aabb_a[0] == pytest.approx((0.0, y_span[0], 0.0))
        assert aabb_a[1] == pytest.approx((_RUN_M, y_span[1], _HEIGHT_M))


def test_the_envelope_the_workaround_needs_renders_nothing_at_all() -> None:
    """Why the envelope is not a harmless extra line.

    The generator subtracts a wall's own child solids out of its body, and the
    leaves fill the envelope exactly — so the workaround's parent carries a
    ``Representation`` that tessellates to ZERO vertices: an
    ``IfcBooleanResult`` bought and paid for in CSG depth that draws nothing.
    The body-less parent carries no representation at all, which is what an
    aggregate parent is supposed to be.
    """
    bodyless = _named(_measure(_compiled(_project(_aggregator()))), "wall:asm")
    workaround = _named(
        _measure(_compiled(_project(_aggregator(envelope=True), expect_clean=False))), "wall:asm")

    assert bodyless == (None, 0.0, False)     # no Representation
    assert workaround == (None, 0.0, True)    # a Representation, drawing nothing


def test_the_body_less_parent_aggregates_the_leaves_it_positions() -> None:
    proj = _project(_aggregator())
    members = _aggregated_under(_compiled(proj), "wall:asm")
    assert members == ["wall:core:wall:asm", "wall:facade:wall:asm"]


# ---------------------------------------------------------------------------
# 3. The issue headline: .anchor() into a body-less container
# ---------------------------------------------------------------------------

#: 300 x 60 x 100 mm, anchored at along=1000, up=2000. ``along=0`` is the
#: corner on your LEFT as you face the outer face, and this host faces +y, so
#: along runs -X from x=6.0: [6.0 - 1.0 - 0.3, 6.0 - 1.0]. ``inset=0`` is flush
#: with the outer face at y=0.508, and the ledge is 60 mm deep INTO the wall.
_LEDGE_AABB = ((4.7, 0.448, 2.0), (5.0, 0.508, 2.1))


def _with_ledge() -> Wall:
    asm = _aggregator()
    asm.anchor(Box(name="ledge", start=Point(x=0, y=0, z=0),
                   end=Point(x=300, y=60, z=100)),
               along=1000, up=2000)
    return asm


def test_an_anchored_child_is_placed_by_the_frame_the_leaves_derived() -> None:
    """the title case. The frame came from the ``.add()``ed leaves, and the
    ledge is placed against it — on the OUTER face of the 508 mm buildup, not
    the 180 mm core's."""
    measured = _measure(_compiled(_project(_with_ledge())))
    aabb, volume, _ = _named(measured, "box:ledge")

    assert aabb[0] == pytest.approx(_LEDGE_AABB[0])
    assert aabb[1] == pytest.approx(_LEDGE_AABB[1])
    assert volume == pytest.approx(0.3 * 0.06 * 0.1)


def test_the_anchored_child_keeps_its_own_identity_in_the_file() -> None:
    """The gate on the generator half.

    Taking the FIRST prism child as the wall's body whatever it is means that
    on a body-less container it is the anchored detail:
    the ledge was emitted AS ``wall:asm``'s own geometry, so ``box:ledge:…``
    appeared nowhere in the file — no manifest key, nothing for a patch or a
    schedule to point at — and the parent claimed a body it never had.
    """
    ifc = _compiled(_project(_with_ledge()))
    measured = _measure(ifc)

    assert _named(measured, "wall:asm") == (None, 0.0, False)
    assert _aggregated_under(ifc, "wall:asm") == [
        "box:ledge:wall:asm", "wall:core:wall:asm", "wall:facade:wall:asm",
    ]


# ---------------------------------------------------------------------------
# 4. An empty basis is still refused, and says which kind of empty
# ---------------------------------------------------------------------------


def test_a_container_with_nothing_in_it_is_refused_by_what_it_lacks() -> None:
    errors = _errors(Wall(name="hollow"))
    assert len(errors) == 1
    assert "no children to derive a frame from" in errors[0]
    assert "wall:hollow" in errors[0]
    # This wording is not the rule and must not be what an author
    # reads: a body is one way to have a frame, not the requirement.
    assert "must contain a Box/Extrude body element" not in errors[0]


def test_a_container_holding_only_anchored_children_is_refused_by_count() -> None:
    """The case part 2  exists for — a declared extent. Until then it is
    a refusal, and it names how many children are waiting on the frame they
    cannot themselves define."""
    asm = Wall(name="asm")
    asm.anchor(_leaf("core", *_CORE_MM), along=0, up=0)
    asm.anchor(_leaf("facade", *_FACADE_MM), along=0, up=0)

    errors = _errors(asm)
    assert len(errors) == 1
    assert "no frame to place its 2 .anchor()ed child(ren) against" in errors[0]


def test_the_bake_still_refuses_a_frameless_container_one_pass_later() -> None:
    """``_bake_child`` is the backstop, and it is not redundant: the
    compile-time body check exists for ``Wall`` only, so a ``Column`` with a
    single anchored child reaches the bake with no frame and nothing else
    would have said so."""
    from lite_step.models import Column

    col = Column(name="col")
    col.anchor(Box(name="det", start=Point(x=0, y=0, z=0),
                   end=Point(x=100, y=100, z=100)), along=0, up=0)
    proj = _project(col)

    with pytest.raises(frames.ChildAnchorError) as excinfo:
        frames.resolve_child_anchors(proj)
    assert ".anchor() needs a frame" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 5. One definition of "the body", read by the validator and both backends
# ---------------------------------------------------------------------------


def test_an_anchored_prism_is_a_detail_and_never_the_body() -> None:
    """The disagreement that made the ledge disappear, as a number.

    The container holds THREE prism-bearing children by ``is_prism`` — two
    leaves are containers, so the only prism at this level is the anchored
    ledge — and zero of them are the body.
    """
    asm = _with_ledge()
    prisms = [c for c in asm._elements if tx.is_prism(c)]

    assert len(prisms) == 1                  # the ledge
    assert tx.body_prisms(asm) == []         # …and it is not the body

    bodied = _aggregator(envelope=True)
    assert len(tx.body_prisms(bodied)) == 1
    assert tx.body_prisms(bodied)[0].name == "env"


def test_a_second_unanchored_prism_is_still_the_error_it_was() -> None:
    """Widening "no body" must not have widened "two bodies": a Wall still
    takes exactly one, because the generator emits the first and would drop
    the rest."""
    wall = Wall(name="two")
    wall.add(Box(name="a", start=Point(x=0, y=0, z=0),
                 end=Point(x=1000, y=200, z=2000), type="wall"))
    wall.add(Box(name="b", start=Point(x=2000, y=0, z=0),
                 end=Point(x=3000, y=200, z=2000), type="wall"))

    errors = _errors(wall)
    assert any("contains 2 Box/Extrude children" in e for e in errors), errors


# ---------------------------------------------------------------------------
# 6. The streaming backend has no body-less wall, so it does not get one
# ---------------------------------------------------------------------------


def _mesh_wall() -> Wall:
    wall = Wall(name="mw")
    wall.add(Mesh(name="skin",
                  vertices=[Point(x=0, y=0, z=0), Point(x=_RUN_MM, y=0, z=0),
                            Point(x=_RUN_MM, y=0, z=_HEIGHT_MM),
                            Point(x=0, y=0, z=_HEIGHT_MM)],
                  faces=[[0, 1, 2], [0, 2, 3]]))
    return wall


def test_a_mesh_bodied_wall_emits_its_wall_and_its_tessellation() -> None:
    """A body-less ``Wall`` — one whose body is a ``Mesh`` rather than a prism
    — must still reach the file as an ``IfcWall`` carrying its tessellation,
    with its identity and its children intact.

    The failure this pins is silent: an emitter that builds the ``IfcWall`` out
    of a body PRISM and returns ``None`` without one drops the whole wall — no
    identity, no children, no warning.
    """
    streamed = _compiled(_project(_mesh_wall()))
    assert streamed.count("IFCWALL(") == 1
    assert streamed.count("IFCTRIANGULATEDFACESET(") == 1


def test_an_opening_on_a_bodyless_container_reaches_the_leaves() -> None:
    """What the refusal here was holding the place for.

    Refusing an opening here is right only for as long as an opening has
        nowhere to go: measured with the body check relaxed and nothing else, this model
    compiled ``success=True`` and emitted **0 IfcWindow, 0 IfcOpeningElement,
    0 IfcRelVoidsElement**. What it forced instead was the ENVELOPE workaround
    — a solid spanning the whole buildup, authored purely so the hole had a
    product to hang off — and measurement found what that costs: phantom
    ``IfcWall`` volume when the assembly is under-tiled, and a
    zero-vertex representation behind a stack of booleans when it is tiled
    correctly (test 2 above already pins the empty husk).

    The hole is now handed to the leaves it OVERLAPS, through the identical
    ``generator._process_container_openings`` a ``Column``'s goes through
, with ``ifc.voids.void_reaches_children`` answering per instance off
    ``tx.body_prisms``. Two leaves, two ``IfcOpeningElement``, one
    ``IfcWindow`` — ``IfcElement.FillsVoids`` is [0:1].

    The full acceptance case, with a layered leaf and a stud bay, is
    ``test_bodyless_wall_openings.py``. This one stays here because it is the
    minimal shape.
    """
    asm = _aggregator()
    asm.opening(Window(name="w0", width=1200, height=1400), along=1500, up=900)

    assert _errors(asm) == []
    ifc = _compiled(_project(asm))
    assert ifc.count("IFCWINDOW(") == 1
    assert ifc.count("IFCRELFILLSELEMENT(") == 1
    # both leaves stand in the hole: it cuts the full 508 mm buildup
    assert ifc.count("IFCOPENINGELEMENT(") == 2
    assert ifc.count("IFCRELVOIDSELEMENT(") == 2

    import ifcopenshell

    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ifc)
        model = ifcopenshell.open(path)
        hosts = sorted(r.RelatingBuildingElement.Name
                       for r in model.by_type("IfcRelVoidsElement"))
    finally:
        os.unlink(path)

    # The LEAVES, never the representation-less aggregate parent.
    assert hosts == ["wall:core:wall:asm", "wall:facade:wall:asm"]


def test_a_container_with_an_empty_basis_still_refuses_its_opening() -> None:
    """The refusal NARROWED — it did not disappear.

    With no ``.add()``ed children there is no run axis for ``along=`` to walk
    and no leaf for the hole to land on, so the opening is still a compile
    error. It is reported by ``_emits_openings``, which returns
    ``frames.has_frame_basis`` for a distributing host — the same call
    ``_bodyless_container_errors`` makes for the frame itself, so the two
    cannot disagree about what "has geometry" means.
    """
    empty = Wall(name="asm")
    empty.anchor(Box(name="ledge", type="wall", start=Point(x=0, y=0, z=0),
                     end=Point(x=200, y=200, z=200)), along=0, up=0)
    empty.opening(Window(name="w0", width=1200, height=1400),
                  along=1500, up=900)

    errors = _errors(empty)
    assert errors, "an empty basis must still refuse the opening"
    assert any("no .add()ed children to derive a frame from" in e
               for e in errors), errors


def test_the_same_opening_on_the_leaf_that_owns_it_is_accepted() -> None:
    """The remedy the refusal names has to work, or it is not a remedy.

    The window goes on the leaf it actually cuts, and the hole comes out where
    the leaf is: the void spans the core leaf's own 180 mm depth.
    """
    asm = Wall(name="asm")
    core = _leaf("core", *_CORE_MM)
    core.anchor(Window(name="w0", width=1200, height=1400), along=1500, up=900)
    asm.add(core)
    asm.add(_leaf("facade", *_FACADE_MM))

    proj = _project(asm)
    ifc = _compiled(proj)
    assert ifc.count("IFCWINDOW(") == 1
    assert ifc.count("IFCOPENINGELEMENT(") == 1
    assert ifc.count("IFCRELVOIDSELEMENT(") == 1

    measured = _measure(ifc)
    aabb, _volume_v, _ = _named(measured, "wall:core")
    assert aabb[0][1] == pytest.approx(0.0)
    assert aabb[1][1] == pytest.approx(0.180)
