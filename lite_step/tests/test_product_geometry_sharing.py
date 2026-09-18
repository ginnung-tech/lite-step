"""``Product`` geometry sharing — ``IfcRepresentationMap`` / ``IfcMappedItem``
(WS-B step 2).

Roadmap §V.4. Step 1 gave every Product one ``Ifc<Class>Type`` and left every
occurrence emitting its own body. Step 2 makes the type carry the body and each
occurrence reference it.

**The load-bearing test is world-coordinate equality** — everything else here
is a supporting instrument. Shared geometry that renders anywhere other than
where the unshared version did is the failure this file exists to catch, and it
is otherwise completely silent: the file opens, the schema validates, the
entity count goes down, and the building is wrong.

What is pinned, and why each one is here rather than assumed:

1.  **World geometry, before vs after**, through ``ifcopenshell.geom``, on a
    fixture whose occurrences sit at seven distinct positions across two
    storeys under four distinct rotations (identity, 90°, 180°, 45°). A fixture
    at the origin, at identity, or with every occurrence in one place cannot
    tell a correct transform from a dropped one.
2.  **Every compared product actually tessellates.** Two empty vertex arrays
    compare equal, so a comparison over geometry that renders to nothing passes
    for the wrong reason — and this compiler HAS such geometry (see
    ``catalog_unit``'s note). The count is asserted.
3.  **The baseline really is unshared.** Test 1 compares against the same
    compiler with the sharing pass switched off at its single call site; if
    that switch stopped working, test 1 would compare shared against shared and
    pass forever.
4.  **The 1000× trap.** ``normalize_project_to_meters`` walks the containment
    tree and a ``Product``'s snapshot is deliberately NOT in it, so the
    snapshot stays in millimetres while every occurrence is in metres. This
    implementation never reads the snapshot — it maps a body the OCCURRENCE
    already emitted, which is normalized — so the trap is absent by
    construction rather than avoided by care. That is worth exactly nothing
    unless something measures it, so the shared body's dimensions are asserted
    to be the metre values.
5.  **Nothing is orphaned.** ``IfcShapeModel`` WR11 says a shape representation
    is used by a product shape, a representation map or a shape aspect. The
    moment the first occurrence's body becomes the map's, it must leave that
    occurrence's ``IfcProductDefinitionShape`` — and the other occurrences'
    bodies must be DELETED, not merely unreferenced.
6.  **Both backends, and they agree.** Per.
7.  **The variant split**, which is not a defensive nicety: this compiler bakes
    an anchored child's host rotation into its PROFILE (a window on a +Y wall
    is ``IfcRectangleProfileDef(0.06, 1.4)`` where the same window on a +X wall
    is ``(1.4, 0.06)``, identity rotation in the placement on both), so
    occurrences on differently-aligned hosts are genuinely different bodies and
    one map for all of them would put a whole facade on its side.
"""

from __future__ import annotations

import contextlib
import io
import json
import re

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc import generator as G
from lite_step.ifc.product_types import (
    MAPPED_REPRESENTATION_TYPE,
    ShapeAdapter,
    geometry_fingerprint,
    purgeable_after_sharing,
)
from lite_step.models import (
    Bar, Box, Element, Point, Product, Project, Storey, Transform, Wall, Window,
)

pytest.importorskip("ifcopenshell")

import ifcopenshell                     # noqa: E402
import ifcopenshell.geom                # noqa: E402
import ifcopenshell.util.placement      # noqa: E402
import ifcopenshell.validate            # noqa: E402
import numpy as np                      # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — local geometry only (never exec a skill; see tests/_fixtures.py)
# ---------------------------------------------------------------------------

#: Deliberately NOT a unit cube at the origin. A 1×1×1 body cannot tell a
#: 1000× scale error from a correct one as clearly as a 1400×1600 unit can, and
#: a fixture at identity hides every transform bug there is.
WIDTH, HEIGHT, THICK = 1400, 1600, 60


def catalog_unit(name: str = None) -> Element:
    """A five-part joinery unit — the roadmap's worked case, in miniature.

    An ``Element(ifc_class="IfcPlate")`` rather than a ``Window``, and that is
    a deliberate choice with a reason worth writing down. A ``Window``
    occurrence anchored into a ``Wall`` is a semantic OPENING, and
    ``ifcopenshell.geom`` resolves its joinery parts as carrying the WALL's
    opening voids (the shape id comes back as ``72-openings-43-115-177``) — so
    four of these five parts tessellate to ZERO vertices, in the shared file
    and the unshared one alike. That is a pre-existing rendering defect, not
    one this change introduces, but a world-coordinate comparison run over it
    would be comparing empty against empty and would pass whatever the sharing
    pass did. ``anchored_facade`` below still covers the anchored/opening path
    for everything that does not need tessellation.
    """
    unit = Element(ifc_class="IfcPlate", name=name)
    unit.add(Box(name="jamb_l", start=Point(x=0, y=0, z=0),
                 end=Point(x=THICK, y=THICK, z=HEIGHT)))
    unit.add(Box(name="jamb_r", start=Point(x=WIDTH - THICK, y=0, z=0),
                 end=Point(x=WIDTH, y=THICK, z=HEIGHT)))
    unit.add(Box(name="head", start=Point(x=0, y=0, z=HEIGHT - THICK),
                 end=Point(x=WIDTH, y=THICK, z=HEIGHT)))
    unit.add(Box(name="sill", start=Point(x=0, y=0, z=0),
                 end=Point(x=WIDTH, y=THICK, z=THICK)))
    unit.add(Box(name="glass", color="glass",
                 start=Point(x=THICK, y=20, z=THICK),
                 end=Point(x=WIDTH - THICK, y=40, z=HEIGHT - THICK)))
    return unit


#: Seven placements: three plain translations, then 90°, 90°, 180° and 45°
#: about Z. Every one distinct, and the last four exercise the rotation the
#: mapped body has to survive — a shared block whose transform is dropped or
#: doubled shows up HERE and essentially nowhere else.
PLACEMENTS = (
    ((0, 0, 0), ()),
    ((3000, 0, 0), ()),
    ((6000, 0, 0), ()),
    ((12000, 0, 0), (("z", 9000),)),
    ((12000, 4000, 0), (("z", 9000),)),
    ((0, 9000, 0), (("z", 18000),)),
    ((4000, 9000, 0), (("z", 4500),)),
)

#: How many of the seven live on the upper storey. Non-zero on purpose:
#: ``_set_relative_placements`` subtracts the storey elevation from the
#: placement, so a single-storey fixture would be running that subtraction at
#: zero and every guard here would stay green with it deleted.
_ON_UPPER_STOREY = 2


def spread_units(count: int = len(PLACEMENTS)) -> Project:
    """THE acceptance fixture: one catalog item, seven occurrences, no two
    alike in where or how they stand."""
    product = Product(catalog_unit(), name="nordic")
    ground = Storey(name="ground", elevation=0)
    first = Storey(name="first", elevation=3000)
    for index, ((x, y, z), rotations) in enumerate(PLACEMENTS[:count]):
        occurrence = product.occurrence(name=f"u{index}")
        occurrence.placement = Transform(origin=Point(x=x, y=y, z=z),
                                         rotations=list(rotations))
        target = first if index >= len(PLACEMENTS) - _ON_UPPER_STOREY else ground
        target.add(occurrence)
    return Project(name="catalog", storeys=[ground, first])


#: 13 points around a circle — a BENT bar, which is what a multi-point path
#: exists for and what curved reinforcement always is.
_RING_POINTS = 13


def catalog_unit_with_bent_bar(name: str = None) -> Element:
    """The shape: a catalog item carrying a MULTI-POINT Bar.

    The path length is the whole variable. A 2-point ``Bar`` emits an
    ``IfcSweptDiskSolid`` over an ``IfcPolyline`` and always promoted fine;
    3+ points emit an ``IfcIndexedPolyCurve`` whose ``Segments`` are
    ``IfcLineIndex`` — a defined TYPE, which ``ifcopenshell`` hands back as an
    ``entity_instance`` with ``id() == 0``. The purge walk keyed those at 0
    (colliding every segment into one candidate) and called ``get_inverse`` on
    them, which raises.
    """
    import math
    unit = Element(ifc_class="IfcPlate", name=name)
    unit.add(Box(name="plate", start=Point(x=0, y=0, z=0),
                 end=Point(x=WIDTH, y=WIDTH, z=THICK)))
    unit.add(Bar(name="ring", diameter=12, grade="B500B",
                 path=[Point(x=int(WIDTH / 2 + 500 * math.cos(2 * math.pi * k / (_RING_POINTS - 1))),
                             y=int(WIDTH / 2 + 500 * math.sin(2 * math.pi * k / (_RING_POINTS - 1))),
                             z=THICK // 2)
                       for k in range(_RING_POINTS)]))
    return unit


def spread_bent_bar_units(count: int = len(PLACEMENTS)) -> Project:
    """``spread_units`` with the bent-bar catalog item — same placements, so
    the purge genuinely runs rather than the fixture quietly sharing nothing."""
    product = Product(catalog_unit_with_bent_bar(), name="ward_plate")
    ground = Storey(name="ground", elevation=0)
    first = Storey(name="first", elevation=3000)
    for index, ((x, y, z), rotations) in enumerate(PLACEMENTS[:count]):
        occurrence = product.occurrence(name=f"p{index}")
        occurrence.placement = Transform(origin=Point(x=x, y=y, z=z),
                                         rotations=list(rotations))
        target = first if index >= len(PLACEMENTS) - _ON_UPPER_STOREY else ground
        target.add(occurrence)
    return Project(name="ward", storeys=[ground, first])


def build_window() -> Window:
    win = Window(width=WIDTH, height=HEIGHT)
    win.add(Box(name="head", start=Point(x=0, y=0, z=HEIGHT - THICK),
                end=Point(x=WIDTH, y=THICK, z=HEIGHT)))
    win.add(Box(name="glass", color="glass",
                start=Point(x=THICK, y=20, z=THICK),
                end=Point(x=WIDTH - THICK, y=40, z=HEIGHT - THICK)))
    return win


def anchored_facade() -> Project:
    """Windows anchored into three walls, one of which runs along +Y.

    The rotated host is the point: it is what makes the occurrences two
    geometry VARIANTS rather than one.
    """
    nordic = Product(build_window(), name="nordic")

    ground = Storey(name="ground", elevation=0)
    south = Wall(name="south")
    south.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=12000, y=300, z=2700)))
    for i, along in enumerate((1200, 5200, 9200)):
        south.anchor(nordic.occurrence(name=f"s{i}"), along=along, up=900)
    east = Wall(name="east")
    east.add(Box(start=Point(x=12000, y=0, z=0), end=Point(x=12300, y=9000, z=2700)))
    for i, along in enumerate((1500, 6000)):
        east.anchor(nordic.occurrence(name=f"e{i}"), along=along, up=1100)
    ground.add(south, east)

    first = Storey(name="first", elevation=3000)
    north = Wall(name="north")
    north.add(Box(start=Point(x=0, y=8700, z=0), end=Point(x=12000, y=9000, z=2700)))
    for i, along in enumerate((2000, 8000)):
        north.anchor(nordic.occurrence(name=f"n{i}"), along=along, up=700)
    first.add(north)

    return Project(name="facade", storeys=[ground, first])


def build_panel(width: int = 2000, name: str = None) -> Wall:
    """A one-body wall panel — the shape BOTH backends emit identically.

    Deliberately one ``Box``, not three. A three-box wall is one body plus two
    aggregated proxies on the ifcopenshell backend and a single merged body on
    the streaming one — a real, pre-existing divergence that has nothing to do
    with sharing, and comparing the backends through it would only measure that.
    """
    wall = Wall(name=name)
    wall.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=width, y=300, z=2700)))
    return wall


def panel_run(count: int = 4, name: str = "panel_2000") -> Project:
    """``count`` occurrences of one wall panel, SPACED along +X.

    The spacing is load-bearing, not cosmetic. Occurrences dropped at the same
    coordinates are wholly inside one another, so the displacement pass carves
    every one but the last to nothing — four occurrences, four different
    bodies, and the variant report says so out loud. (That is the compiler
    behaving as designed on a nonsense model; it is written down because the
    obvious fixture IS that nonsense model.)
    """
    product = Product(build_panel(), name=name)
    storey = Storey(name="ground", elevation=0)
    for i in range(count):
        occurrence = product.occurrence(name=f"unit{i}")
        occurrence.placement = Transform(origin=Point(x=i * 2500, y=0, z=0))
        storey.add(occurrence)
    return Project(name="catalog", storeys=[storey])


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

#: The single call site of the sharing pass. Named here rather than patched
#: inline so ``test_switching_sharing_off_really_switches_it_off`` can prove the
#: switch still works: a baseline produced by a patch that no longer patches
#: anything is a comparison of a thing with itself.
_SHARING_HOOK = (G, "_share_product_geometry")


def compile_ifc(proj: Project, backend: str = "ifcopenshell", *,
                shared: bool = True, source_code: str = "src") -> str:
    """Compile, optionally with the geometry-sharing pass switched off."""
    normalized = normalize_project_to_meters(proj)
    module, attribute = _SHARING_HOOK
    original = getattr(module, attribute)
    if not shared:
        setattr(module, attribute, lambda *a, **k: None)
    try:
        result = G.generate_ifc(normalized, source_code=source_code)
    finally:
        setattr(module, attribute, original)
    assert result.success, result.error
    return result.ifc_content


def model_of(ifc: str):
    return ifcopenshell.file.from_string(ifc)


def schema_errors(model) -> list:
    """Every ``level == "error"`` ``ifcopenshell.validate`` reports.

    Same shape as ``tests/test_ifc43_conformance.py`` — including the stderr
    redirect, without which the one line that matters is buried in the
    validator's running commentary.
    """
    logger = ifcopenshell.validate.json_logger()
    with contextlib.redirect_stderr(io.StringIO()):
        ifcopenshell.validate.validate(model, logger)
    return [s for s in logger.statements if s.get("level") == "error"]


def world_vertices(model) -> dict:
    """``{product name: sorted world-space vertices}`` for every shaped product.

    ``use-world-coords`` is the whole point: it makes ``ifcopenshell.geom``
    resolve the placement chain and the mapped item TOGETHER, so the answer is
    where the thing actually IS — the only question worth asking of a change
    that moves geometry into a shared block.
    """
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    out = {}
    for product in model.by_type("IfcProduct"):
        if not product.Representation:
            continue
        name = product.Name or f"<#{product.id()}>"
        shape = ifcopenshell.geom.create_shape(settings, product)
        verts = np.array(shape.geometry.verts, dtype=float).reshape(-1, 3)
        out[name] = np.round(np.sort(verts, axis=0), 6)
    return out


def solid_count(model) -> int:
    return sum(1 for e in model
               if e.is_a("IfcSolidModel") or e.is_a("IfcBooleanResult"))


def manifest_of(ifc: str) -> dict:
    model = model_of(ifc)
    for pset in model.by_type("IfcPropertySet"):
        if pset.Name == "LITESTEP_META":
            for prop in pset.HasProperties:
                if prop.Name == "MANIFEST":
                    return json.loads(prop.NominalValue.wrappedValue)
    raise AssertionError("no LITESTEP_META manifest in the emitted file")


def _base_solid(item):
    """The swept solid under a mapped item, through any boolean wrapping it.

    A mapped body is not always a bare ``IfcExtrudedAreaSolid``. Since the parts of a catalog unit carve each other exactly as
    they always did when the unit was UNPLACED — ``head`` and ``sill`` overlap
    both jambs at the corners — so four of ``catalog_unit``'s five slots map an
    ``IfcBooleanResult``. The base is the FIRST operand: the carve subtracts
    from it, so the profile being measured is the one the base carries.
    """
    while item.is_a("IfcBooleanResult"):
        item = item.FirstOperand
    return item


def profile_dims(model) -> set:
    return {(round(solid.SweptArea.XDim, 6), round(solid.SweptArea.YDim, 6))
            for rep_map in model.by_type("IfcRepresentationMap")
            for item in rep_map.MappedRepresentation.Items
            for solid in [_base_solid(item)]
            if solid.is_a("IfcSweptAreaSolid")}


# ===========================================================================
# 1. The load-bearing test
# ===========================================================================

class TestWorldGeometryIsUnchanged:

    def test_every_occurrence_renders_exactly_where_it_did_unshared(self):
        """THE acceptance test.

        Seven occurrences of one catalog item, at seven positions, across two
        storeys, under four different rotations — tessellated in world
        coordinates and compared vertex for vertex against the same compiler
        with sharing switched off.
        """
        shared = world_vertices(model_of(compile_ifc(spread_units(), "ifcopenshell")))
        unshared = world_vertices(model_of(
            compile_ifc(spread_units(), "ifcopenshell", shared=False)))

        assert set(shared) == set(unshared), (
            "sharing changed WHICH products have geometry: "
            f"{sorted(set(shared) ^ set(unshared))}")

        moved = []
        for name in sorted(shared):
            a, b = shared[name], unshared[name]
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-6):
                moved.append(name)
        assert not moved, (
            f"{len(moved)} of {len(shared)} products render somewhere else "
            f"once their geometry is shared: {moved[:8]}")

    def test_the_comparison_is_not_empty_against_empty(self):
        """Guards the test above against the failure that already exists in
        this compiler: a product whose representation tessellates to NOTHING
        (see ``catalog_unit``). Two empty vertex arrays are equal, so a
        comparison run over them proves nothing at all."""
        verts = world_vertices(model_of(compile_ifc(spread_units(), "ifcopenshell")))
        assert len(verts) >= 35, f"only {len(verts)} shaped products in the fixture"
        empty = [name for name, v in verts.items() if len(v) == 0]
        assert not empty, f"{len(empty)} products tessellate to nothing: {empty[:8]}"

    def test_the_occurrences_really_are_at_seven_different_places(self):
        """Guards the fixture, not the feature.

        If every occurrence sat in the same spot — the natural mistake, since a
        Product's occurrences are copies of ONE snapshot — the acceptance test
        would compare one position against itself seven times.
        """
        verts = world_vertices(model_of(compile_ifc(spread_units(), "ifcopenshell")))
        corners = {name: tuple(np.round(np.min(v, axis=0), 4))
                   for name, v in verts.items()
                   if name.startswith("box:glass:plate:u")}
        assert len(corners) == len(PLACEMENTS), sorted(corners)
        assert len(set(corners.values())) == len(PLACEMENTS), corners

    def test_the_fixture_really_rotates(self):
        """The other half of the same guard. A translation-only fixture cannot
        distinguish a mapped body that survived its rotation from one whose
        rotation was silently dropped — every vertex would still land right."""
        model = model_of(compile_ifc(spread_units(), "ifcopenshell"))
        rotations = set()
        for product in model.by_type("IfcPlate"):
            matrix = ifcopenshell.util.placement.get_local_placement(product.ObjectPlacement)
            rotations.add(tuple(np.round(matrix[:3, :3], 6).ravel()))
        assert len(rotations) >= 4, rotations
        identity = tuple(np.eye(3).ravel())
        assert identity in rotations and len(rotations - {identity}) >= 3

    def test_switching_sharing_off_really_switches_it_off(self):
        """Guards the baseline.

        The unshared file is produced by patching one attribute. If that
        attribute is ever renamed, ``compile_ifc(shared=False)`` silently
        returns the SHARED file and the acceptance test compares a thing with
        itself — green, forever, measuring nothing.
        """
        shared = compile_ifc(spread_units(), "ifcopenshell")
        unshared = compile_ifc(spread_units(), "ifcopenshell", shared=False)
        assert "IFCMAPPEDITEM" in shared and "IFCREPRESENTATIONMAP" in shared
        assert "IFCMAPPEDITEM" not in unshared
        assert "IFCREPRESENTATIONMAP" not in unshared


# ===========================================================================
# 2. What the file says
# ===========================================================================

class TestEmittedShape:

    def test_every_shared_body_is_a_mapped_representation(self):
        model = model_of(compile_ifc(panel_run(4), "ifcopenshell"))
        occurrences = [p for p in model.by_type("IfcProduct")
                       if p.Name and p.Name.startswith("wall:unit")]
        assert len(occurrences) == 4, [p.Name for p in occurrences]
        for product in occurrences:
            reps = product.Representation.Representations
            assert reps, product.Name
            for rep in reps:
                assert rep.RepresentationType == MAPPED_REPRESENTATION_TYPE
                assert [i.is_a() for i in rep.Items] == ["IfcMappedItem"]
                # The identifier is carried over from the body being shared —
                # the Mapped Geometry concept keeps 'Body' / 'Axis' /
                # 'FootPrint' and changes only the TYPE.
                assert rep.RepresentationIdentifier == "Body"

    def test_the_type_carries_the_maps(self):
        model = model_of(compile_ifc(panel_run(4), "ifcopenshell"))
        maps = model.by_type("IfcWallType")[0].RepresentationMaps
        assert maps, "RepresentationMaps is unset — the type carries no shape"
        for rep_map in maps:
            assert rep_map.MappedRepresentation.RepresentationType != \
                MAPPED_REPRESENTATION_TYPE, "a map pointing at another mapping"

    def test_the_mapping_is_the_identity_on_both_ends(self):
        """The placement decision, asserted rather than described.

        ``MappingOrigin`` at the origin with no axes; ``MappingTarget`` at the
        origin with no axes and no scale. Anything else and the occurrence's
        world position becomes a function of how the READING toolkit composes
        origin with target — see the "Geometry sharing" header in
        ``product_types.py``.
        """
        model = model_of(compile_ifc(panel_run(4), "ifcopenshell"))
        for rep_map in model.by_type("IfcRepresentationMap"):
            origin = rep_map.MappingOrigin
            assert origin.is_a("IfcAxis2Placement3D")
            assert tuple(origin.Location.Coordinates) == (0.0, 0.0, 0.0)
            assert origin.Axis is None and origin.RefDirection is None
        for item in model.by_type("IfcMappedItem"):
            target = item.MappingTarget
            assert target.is_a("IfcCartesianTransformationOperator3D")
            assert tuple(target.LocalOrigin.Coordinates) == (0.0, 0.0, 0.0)
            assert target.Axis1 is None and target.Axis2 is None
            assert target.Axis3 is None and target.Scale is None

    def test_the_occurrence_keeps_its_own_object_placement(self):
        """The other half of the placement decision.

        The transform lives in ``ObjectPlacement`` and nowhere else, so it must
        still be there — and must still be the SAME matrix step 1 emitted. An
        implementation that moved the transform into ``MappingTarget`` and
        forgot to clear the placement would double it; one that cleared the
        placement would detach the occurrence from its storey chain.
        """
        shared = model_of(compile_ifc(spread_units(), "ifcopenshell"))
        unshared = model_of(compile_ifc(spread_units(), "ifcopenshell", shared=False))

        def placements(model):
            return {p.Name: np.round(
                ifcopenshell.util.placement.get_local_placement(p.ObjectPlacement), 9
            ).tolist()
                for p in model.by_type("IfcProduct")
                if p.Name and p.ObjectPlacement}

        before, after = placements(unshared), placements(shared)
        assert set(before) == set(after)
        assert before == after, "sharing changed an ObjectPlacement"
        assert any(m != np.eye(4).tolist() for m in after.values()), (
            "every placement is the identity — this fixture cannot detect a "
            "transform moving to the wrong attribute")


# ===========================================================================
# 3. The 1000× trap, measured
# ===========================================================================

class TestUnitsTrap:

    def test_the_shared_body_is_in_metres(self):
        """``Product.snapshot`` is NOT in the containment tree, so
        ``normalize_project_to_meters`` never reaches it and it stays in
        millimetres. This implementation maps a body the OCCURRENCE emitted,
        which IS normalized — so the trap is structurally absent rather than
        carefully avoided. That distinction is worth nothing unmeasured.

        A 1400 mm × 60 mm rail must appear as 1.4 × 0.06. At 1400 × 60 the unit
        is a kilometre wide and every viewer still opens it.
        """
        dims = profile_dims(model_of(compile_ifc(spread_units(), "ifcopenshell")))
        assert dims, "no mapped bodies to measure"
        assert (1.4, 0.06) in dims, sorted(dims)
        biggest = max(max(x, y) for x, y in dims)
        assert biggest < 10.0, (
            f"a shared body is {biggest} m across — a 1000× error would look "
            f"exactly like this")

    def test_the_fingerprint_separates_a_millimetre_body_from_a_metre_one(self):
        """The instrument that would catch it if the trap ever came back.

        If a future change DID map the un-normalized snapshot, the mm body and
        the m body would be different geometry, so the fingerprint has to tell
        them apart — otherwise they would land in one variant and the wrong one
        would win.
        """
        adapter = _FakeAdapter()
        metres = _fake("IFCEXTRUDEDAREASOLID",
                       [_fake("IFCRECTANGLEPROFILEDEF", [1.4, 0.06])])
        millis = _fake("IFCEXTRUDEDAREASOLID",
                       [_fake("IFCRECTANGLEPROFILEDEF", [1400.0, 60.0])])
        assert (geometry_fingerprint(metres, adapter)
                != geometry_fingerprint(millis, adapter))
        twin = _fake("IFCEXTRUDEDAREASOLID",
                     [_fake("IFCRECTANGLEPROFILEDEF", [1.4, 0.06])])
        assert (geometry_fingerprint(metres, adapter)
                == geometry_fingerprint(twin, adapter))


# ===========================================================================
# 4. Nothing is left behind — IfcShapeModel WR11
# ===========================================================================

class TestNoOrphans:

    def test_every_shape_representation_is_used_exactly_once(self):
        """``IfcShapeModel`` WR11: used by a product shape, a representation
        map or a shape aspect. ZERO users is the husk a naive replace leaves
        behind; TWO is the mistake of leaving the first occurrence's body in
        its own product shape AND making it the map's."""
        for proj in (panel_run(4), spread_units() if "ifcopenshell" == "ifcopenshell" else panel_run(2)):
            model = model_of(compile_ifc(proj))
            for rep in model.by_type("IfcShapeRepresentation"):
                users = [i for i in model.get_inverse(rep)
                         if i.is_a("IfcProductDefinitionShape")
                         or i.is_a("IfcRepresentationMap")
                         or i.is_a("IfcShapeAspect")]
                assert len(users) == 1, (rep, users)

    def test_nothing_references_a_line_that_is_gone(self):
        """A style is a referrer, not a referee — nothing points AT it — so a
        purge that only follows forward references leaves it dangling, and one
        that collects everything unreferenced deletes every style in the model.
        Both were live possibilities."""
        ifc = compile_ifc(panel_run(4), "ifcopenshell")
        defined = {int(n) for n in re.findall(r"^#(\d+)=", ifc, re.MULTILINE)}
        used = {int(n) for n in re.findall(r"#(\d+)", ifc)}
        assert not (used - defined), sorted(used - defined)[:10]
        for styled in model_of(ifc).by_type("IfcStyledItem"):
            assert styled.Item is not None

    def test_the_glass_style_survives_its_solid_being_shared(self):
        """The purge's other direction: exactly one styled item should remain
        for the glass pane (there were five, one per occurrence, and four of
        their solids are gone) — not zero."""
        model = model_of(compile_ifc(spread_units(), "ifcopenshell"))
        styled = model.by_type("IfcStyledItem")
        assert len(styled) == 1, styled
        assert styled[0].Item.is_a("IfcExtrudedAreaSolid")
        assert model.get_inverse(styled[0].Item), "the styled solid is orphaned"

    def test_a_product_bearing_model_is_ifc43_conformant(self):
        """The corpus gate cannot cover this: no committed skill uses
        ``Product``, so ``tests/test_ifc43_conformance.py`` validates twenty
        models containing not one ``IfcRepresentationMap``."""
        model = model_of(compile_ifc(panel_run(4), "ifcopenshell"))
        errors = schema_errors(model)
        assert not errors, [
            " ".join(str(e.get("message", "")).split())[:200] for e in errors[:6]]

    @pytest.mark.parametrize("project", ["spread", "anchored"])
    def test_the_richer_fixtures_are_conformant_too(self, project):
        proj = spread_units() if project == "spread" else anchored_facade()
        errors = schema_errors(model_of(compile_ifc(proj)))
        assert not errors, [
            " ".join(str(e.get("message", "")).split())[:200] for e in errors[:6]]


# ===========================================================================
# 5. The measured win
# ===========================================================================

class TestTheWin:

    def test_sharing_collapses_the_bodies(self):
        """Roadmap §V.4's number, on this fixture: 7 units' worth of solids
        where ONE unit's worth will do.

        The unit measured is SOLIDS, not total entities. Each occurrence still
        needs its own product shape, mapped representation and mapped item, so
        for geometry as cheap as an extruded rectangle the entity TOTAL goes
        up; what collapses is the geometry itself, which is the part that
        scales with real joinery (breps, tessellations, boolean trees).

        ``PER_UNIT`` is 13 rather than the 5 parts, and that is
        rather than a regression: ``head`` and ``sill`` each overlap both jambs
        at the corners, so four parts carve and each carve costs its operand
        entities. The unit emitted exactly this way when it was UNPLACED both
        before and after that — measured, and pinned one test down as
        ``test_a_placed_unit_emits_what_an_unplaced_one_does``. What changed is
        only that a PLACED unit now agrees with it.

        The ratio is what the roadmap claimed and is unchanged: 7x.
        """
        PER_UNIT = 13
        shared = model_of(compile_ifc(spread_units(), "ifcopenshell"))
        unshared = model_of(compile_ifc(spread_units(), "ifcopenshell", shared=False))
        before, after = solid_count(unshared), solid_count(shared)
        assert before == len(PLACEMENTS) * PER_UNIT, before
        assert after == PER_UNIT, f"{before} -> {after}"
        assert before // after == len(PLACEMENTS)

    def test_a_placed_unit_emits_what_an_unplaced_one_does(self):
        """The invariant, at the Product boundary.

        ``placement=`` positions a subtree; it must not change what that
        subtree IS. Before that an occurrence's parts were exempt from the
        inferred carve purely because their container was placed, so the same
        five boxes emitted 5 solids placed and 13 unplaced — the geometry
        depended on where you put it.

        This is the assertion that would have caught that, and it is written
        against a count derived from the unplaced case rather than a literal,
        so it keeps meaning the same thing if the fixture gains a part.
        """
        from lite_step.models import Project, Storey, Point, Transform

        def solids_for(placed: bool) -> int:
            proj = Project(name="frame_equivalence")
            storey = Storey(name="ground", elevation=0)
            proj.add_storey(storey)
            unit = catalog_unit("u")
            if placed:
                unit.placement = Transform(origin=Point(x=50000, y=0, z=0))
            storey.add(unit)
            return solid_count(model_of(compile_ifc(proj, "ifcopenshell",
                                                    shared=False)))

        unplaced = solids_for(False)
        assert unplaced > 5, (
            f"the unplaced control emitted {unplaced} solids — if the parts "
            f"stopped overlapping, this fixture proves nothing")
        assert solids_for(True) == unplaced

    def test_a_lone_occurrence_still_gives_its_type_a_shape(self):
        """A catalog entry whose ``RepresentationMaps`` is empty tells a
        downstream library nothing about the item it names, so the map is
        emitted even when there is no second occurrence to share it with."""
        model = model_of(compile_ifc(panel_run(1), "ifcopenshell"))
        assert len(model.by_type("IfcWallType")[0].RepresentationMaps) == 1
        assert len(model.by_type("IfcMappedItem")) == 1


# ===========================================================================
# 6. Variants — occurrences that are genuinely not the same body
# ===========================================================================

class TestVariants:

    def test_hosts_of_different_orientation_get_their_own_map(self):
        """Measured, not assumed: this compiler bakes an anchored child's host
        rotation into the PROFILE — ``IfcRectangleProfileDef(0.06, 1.4)`` on a
        wall running along +Y where a +X wall gives ``(1.4, 0.06)`` — with an
        identity rotation in the placement on both. One map for all seven
        windows would put the east facade's windows on their side."""
        dims = profile_dims(model_of(compile_ifc(anchored_facade(), "ifcopenshell")))
        assert (1.4, 0.06) in dims and (0.06, 1.4) in dims, sorted(dims)

    def test_the_east_windows_are_not_wearing_the_south_windows_body(self):
        """The consequence of the above, at the only place it can be seen: in
        world coordinates. A south glass pane runs 1280 mm along +X and 20 mm
        along +Y; the east one is the other way round."""
        verts = world_vertices(model_of(compile_ifc(anchored_facade(), "ifcopenshell")))
        south = verts["box:glass:window:s0:wall:south:storey:ground"]
        east = verts["box:glass:window:e0:wall:east:storey:ground"]
        assert len(south) and len(east)
        south_span = np.round(np.max(south, axis=0) - np.min(south, axis=0), 3)
        east_span = np.round(np.max(east, axis=0) - np.min(east, axis=0), 3)
        assert (south_span[0], south_span[1]) == (1.28, 0.02), south_span
        assert (east_span[0], east_span[1]) == (0.02, 1.28), east_span

    def test_one_orientation_means_one_map_per_part(self):
        """The control for the test above: with every occurrence standing the
        same way up, seven occurrences collapse to ONE variant and five maps —
        one per part, not one per unit."""
        model = model_of(compile_ifc(spread_units(), "ifcopenshell"))
        assert len(model.by_type("IfcWallType") or model.by_type("IfcPlateType")) == 1
        maps = model.by_type("IfcPlateType")[0].RepresentationMaps
        assert len(maps) == 5, [m.MappedRepresentation for m in maps]
        assert len(model.by_type("IfcMappedItem")) == 5 * len(PLACEMENTS)


# ===========================================================================
# 7. Identity — the manifest, and both backends
# ===========================================================================

class TestIdentityIsUntouched:

    @pytest.mark.parametrize("project", ["spread", "anchored"])
    def test_the_manifest_is_the_same_set_of_names(self, project):
        """``LITESTEP_META`` is patch identity. Step 2 changes how a shape is
        EXPRESSED, not who anything is — but "obvious" reasoning about this
        manifest has been wrong twice in this repo (a named ``Storey`` IS a key; so is ``"Site"``), so it is measured rather than
        argued.

        Note what a map is NOT: ``IfcRepresentationMap`` and ``IfcMappedItem``
        are representation resources, not ``IfcProduct``s, and the manifest
        walks named products — so neither can reach it. Asserted anyway,
        because that is exactly the reasoning ``IfcAnnotation`` defeated.
        """
        build = spread_units if project == "spread" else anchored_facade
        shared = manifest_of(compile_ifc(build(), "ifcopenshell"))
        unshared = manifest_of(compile_ifc(build(), "ifcopenshell", shared=False))
        assert set(shared) == set(unshared)
        assert shared, "the manifest is empty — this test would pass on nothing"
        assert "nordic" not in shared, "the catalog key reached patch identity"
        assert not any(k.startswith(("map", "mapped")) for k in shared), sorted(shared)

    def test_an_aggregate_of_occurrences_still_lands_where_it_did(self):
        """The one cross-feature ordering that could go wrong quietly.

        Sharing runs BEFORE ``_emit_groupings``, and grouping REBASES each
        member's ``ObjectPlacement`` into the aggregate parent's frame
. Two passes rewriting different halves of the same product
        in sequence is exactly the shape that moves a building by a storey
        height and reports nothing — so the storey here is deliberately NOT at
        elevation 0, where the rebase is a subtraction of zero and this test
        would stay green with the whole interaction broken.
        """
        def build():
            product = Product(build_panel(), name="panel")
            storey = Storey(name="ground", elevation=3000)
            members = []
            for i in range(4):
                occurrence = product.occurrence(name=f"unit{i}")
                occurrence.placement = Transform(origin=Point(x=i * 2500, y=0, z=0))
                storey.add(occurrence)
                members.append(f"wall:unit{i}:storey:ground")
            proj = Project(name="c", storeys=[storey])
            proj.aggregate("facade", members=members)
            return proj

        shared = model_of(compile_ifc(build(), "ifcopenshell"))
        unshared = model_of(compile_ifc(build(), "ifcopenshell", shared=False))
        a, b = world_vertices(shared), world_vertices(unshared)
        assert set(a) == set(b) and a
        moved = [k for k in sorted(a)
                 if a[k].shape != b[k].shape or not np.allclose(a[k], b[k], atol=1e-6)]
        assert not moved, moved
        assert solid_count(unshared) == 4 and solid_count(shared) == 1
        assert not schema_errors(shared)

# ===========================================================================
# 8. The policy functions, directly
# ===========================================================================

class _FakeEntity:
    __slots__ = ("entity_type", "attributes")

    def __init__(self, entity_type, attributes):
        self.entity_type = entity_type
        self.attributes = attributes


def _fake(entity_type, attributes):
    return _FakeEntity(entity_type, attributes)


class _FakeAdapter(ShapeAdapter):
    """A minimal adapter so the policy functions can be tested without a model.

    The referrer table is explicit, which is the point: the purge's behaviour
    is entirely a function of who refers to what, and building that by hand is
    the only way the "shared leaf survives" case can be stated at all.
    """

    def __init__(self, referrers=None):
        self._referrers = referrers or {}

    def is_entity(self, value):
        return isinstance(value, _FakeEntity)

    def key(self, handle):
        return id(handle)

    def entity_type(self, handle):
        return handle.entity_type

    def entity_attrs(self, handle):
        return handle.attributes

    def referrers(self, handle):
        return self._referrers.get(id(handle), [])


class TestPurgePolicy:

    def test_a_leaf_shared_with_a_survivor_is_kept(self):
        """The cached ``IfcRectangleProfileDef`` case. Deleting it because the
        body being purged pointed at it would leave the SURVIVING body — the
        one the map now owns — referring to a line that no longer exists."""
        profile = _fake("IFCRECTANGLEPROFILEDEF", [1.4, 0.06])
        doomed_solid = _fake("IFCEXTRUDEDAREASOLID", [profile])
        doomed_rep = _fake("IFCSHAPEREPRESENTATION", [[doomed_solid]])
        survivor = _fake("IFCEXTRUDEDAREASOLID", [profile])
        adapter = _FakeAdapter({
            id(profile): [doomed_solid, survivor],
            id(doomed_solid): [doomed_rep],
            id(doomed_rep): [],
            id(survivor): [_fake("IFCSHAPEREPRESENTATION", [[survivor]])],
        })
        dead = purgeable_after_sharing([doomed_rep], adapter)
        assert doomed_rep in dead and doomed_solid in dead
        assert profile not in dead

    def test_a_leaf_reached_only_by_the_doomed_body_goes(self):
        profile = _fake("IFCARBITRARYCLOSEDPROFILEDEF", [1])
        solid = _fake("IFCEXTRUDEDAREASOLID", [profile])
        rep = _fake("IFCSHAPEREPRESENTATION", [[solid]])
        adapter = _FakeAdapter({id(profile): [solid], id(solid): [rep], id(rep): []})
        dead = purgeable_after_sharing([rep], adapter)
        assert set(map(id, dead)) == {id(rep), id(solid), id(profile)}

    def test_a_style_on_a_doomed_solid_is_taken_with_it(self):
        solid = _fake("IFCEXTRUDEDAREASOLID", [])
        rep = _fake("IFCSHAPEREPRESENTATION", [[solid]])
        styled = _fake("IFCSTYLEDITEM", [solid])
        adapter = _FakeAdapter({id(solid): [rep, styled], id(rep): [], id(styled): []})
        dead = purgeable_after_sharing([rep], adapter)
        assert styled in dead and solid in dead

    def test_a_style_on_a_surviving_solid_is_left_alone(self):
        """The mirror of the case above, and the one that fails loudest if the
        purge collects everything nothing refers to: NOTHING refers to a styled
        item, ever."""
        survivor = _fake("IFCEXTRUDEDAREASOLID", [])
        survivor_rep = _fake("IFCSHAPEREPRESENTATION", [[survivor]])
        keep_style = _fake("IFCSTYLEDITEM", [survivor])
        doomed = _fake("IFCSHAPEREPRESENTATION", [[]])
        adapter = _FakeAdapter({
            id(survivor): [survivor_rep, keep_style],
            id(survivor_rep): [_fake("IFCPRODUCTDEFINITIONSHAPE", [[survivor_rep]])],
            id(keep_style): [],
            id(doomed): [],
        })
        dead = purgeable_after_sharing([doomed], adapter)
        assert keep_style not in dead and survivor not in dead


class TestMultiPointBarPromotes:
    """A catalog item carrying a BENT bar.

    Curved reinforcement is polyline reinforcement, so the element family that
    most wants promoting (identical reinforced plates) was exactly the family
    that could not be promoted.
    """

    def test_a_defined_type_is_not_an_addressable_entity(self):
        """The unit of the bug: ``IfcLineIndex`` looks like an entity.

        ``isinstance(value, entity_instance)`` — the whole of the old
        predicate — answers True for it, so the walk treated a list of
        integers as a node in the model graph.
        """
        model = ifcopenshell.file(schema="IFC4X3_ADD2")
        points = model.create_entity(
            "IfcCartesianPointList3D",
            CoordList=[(0., 0., 0.), (1., 0., 0.), (2., 1., 0.)])
        segment = model.create_entity("IfcLineIndex", (1, 2, 3))
        curve = model.create_entity("IfcIndexedPolyCurve", Points=points,
                                    Segments=[segment])
        held = curve.Segments[0]

        # The trap, pinned so it stays visible.
        assert isinstance(held, ifcopenshell.entity_instance)
        assert held.is_a() == "IfcLineIndex"
        assert held.id() == 0
        with pytest.raises(RuntimeError, match="entities with ids"):
            model.get_inverse(held)

        adapter = G._ModelShapeAdapter(model)
        assert adapter.is_entity(curve) is True
        assert adapter.is_entity(held) is False, (
            "a defined TYPE is a STEP scalar in a wrapper — treating it as an "
            "entity keys it at 0 and calls get_inverse on it")

    def test_promoting_an_element_with_a_bent_bar_compiles(self):
        project = normalize_project_to_meters(spread_bent_bar_units())
        result = G.generate_ifc(project)
        assert result.success, result.error

    def test_the_fixture_really_carries_the_offending_entity(self):
        """Guards the test above from passing vacuously.

        If the bar ever stopped emitting an ``IfcIndexedPolyCurve`` — a
        different directrix, a rounded path, a changed threshold — the compile
        would still succeed and would be proving nothing.
        """
        project = normalize_project_to_meters(spread_bent_bar_units())
        result = G.generate_ifc(project)
        assert result.success, result.error
        assert result.ifc_content.count("IFCINDEXEDPOLYCURVE") >= 1
        assert result.ifc_content.count("IFCLINEINDEX") >= _RING_POINTS - 1

    def test_the_fixture_really_shares_geometry(self):
        """…and from passing because nothing was purged.

        The raise happens INSIDE the purge, so a fixture whose occurrences
        each keep their own body never reaches it. Fewer maps than mapped
        items is what "sharing happened" looks like.
        """
        project = normalize_project_to_meters(spread_bent_bar_units())
        result = G.generate_ifc(project)
        model = ifcopenshell.file.from_string(result.ifc_content)
        maps = model.by_type("IfcRepresentationMap")
        items = model.by_type("IfcMappedItem")
        assert maps, "no RepresentationMap — the type carries no shape"
        assert len(items) > len(maps), (
            f"{len(items)} mapped items over {len(maps)} maps — nothing was "
            f"shared, so the purge never ran and this suite proves nothing")
