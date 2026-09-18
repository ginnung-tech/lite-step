"""``.anchor()`` accepts every type into every container, and every pairing
RENDERS.

This file is the gate that makes the removal of the anchor type-table honest.
Widening (or here, deleting) a table without the emitters behind it trades a
clear compile-time refusal for a silent warn-and-drop at emit time, which this
codebase ranks as worse than a crash. So the assertion is deliberately NOT
"the model compiles" — a dropped child compiles perfectly. It is:

    the anchored child's geometry is in the file, AT THE PLACE THE ANCHOR PUT IT.

Every child is authored at its container's local origin and anchored with
``up=_LIFT``, far above anything the container itself occupies. The compiled
STEP is then scanned for a Cartesian point at that elevation. A silent drop
scores zero; so does a bake that never ran. Falsification for the whole file
is in ``test_matrix_catches_a_silent_drop``, which stubs one emitter path to
return nothing and asserts the matrix goes red.

``Window``/``Door`` are not in the matrix on purpose. An opening is not a
placed child: it is a size-only VOID cut through the host at full thickness
whatever ``inset=`` says, with a fill riding alongside, pinned at exactly zero
boolean delta by ``test_anchor_openings.py``. It has its own path and keeps it.
"""
import pytest

from lite_step.models import (
    Point, Point2D, Project, Material, LayerSet,
    Box, Extrude, Sweep, Revolve, Pipe, Bar, Mesh,
    Wall, Column, Beam, Slab, Roof, Space, Element,
)
from lite_step.compiler.executor import (
    normalize_project_to_meters, validate_project_report,
)
from lite_step.ifc.generator import generate_ifc


#: How far above the container's own body every anchored child is placed. Well
#: clear of every container body below (all of which top out at 3000 mm), so a
#: hit at this elevation can only be the anchored child.
_LIFT = 9000

#: Half-height of the tallest child, plus slack for a swept profile's own
#: extent. The scan asserts a point ABOVE this floor, in metres.
_FLOOR_M = (_LIFT - 400) / 1000.0


# ── the children ─────────────────────────────────────────────────────────────
#
# Each is authored around its container's LOCAL origin, so the only thing that
# can put geometry near _LIFT is the anchor bake.

def _box():
    return Box(name="det", start=Point(x=0, y=0, z=0), end=Point(x=200, y=100, z=60))


def _extrude():
    return Extrude(name="det", thickness=100, contour=[
        Point(x=0, y=0, z=0), Point(x=200, y=0, z=0),
        Point(x=200, y=0, z=60), Point(x=0, y=0, z=60),
    ])


def _profile():
    return [Point2D(x=-20, y=-20), Point2D(x=20, y=-20),
            Point2D(x=20, y=20), Point2D(x=-20, y=20)]


def _radial_profile():
    """A Revolve's profile X is a radial distance and may not cross the axis."""
    return [Point2D(x=10, y=-20), Point2D(x=40, y=-20),
            Point2D(x=40, y=20), Point2D(x=10, y=20)]


def _sweep():
    return Sweep(name="det", profile=_profile(),
                 path=[Point(x=0, y=0, z=0), Point(x=400, y=0, z=0)])


def _revolve():
    return Revolve(name="det", profile=_radial_profile(), angle=18000,
                   path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=200)])


def _pipe():
    return Pipe(name="det", radius=25,
                path=[Point(x=0, y=0, z=0), Point(x=300, y=0, z=0)])


def _bar():
    return Bar(name="det", diameter=12,
               path=[Point(x=0, y=0, z=0), Point(x=300, y=0, z=0)])


def _mesh():
    return Mesh(
        name="det",
        vertices=[Point(x=0, y=0, z=0), Point(x=200, y=0, z=0),
                  Point(x=200, y=200, z=0), Point(x=0, y=0, z=100)],
        faces=[(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)],
    )


def _bodied(container):
    """A container child needs geometry of its own to be worth anchoring."""
    container.add(Box(name="body", start=Point(x=0, y=0, z=0),
                      end=Point(x=300, y=120, z=200)))
    return container


def _element():
    return _bodied(Element(ifc_class="IfcBuildingElementProxy", name="det"))


CHILDREN = {
    "Box": _box,
    "Extrude": _extrude,
    "Sweep": _sweep,
    "Revolve": _revolve,
    "Pipe": _pipe,
    "Bar": _bar,
    "Mesh": _mesh,
    "Element": _element,
    "Wall": lambda: _bodied(Wall(name="det")),
    "Column": lambda: _bodied(Column(name="det")),
    "Beam": lambda: _bodied(Beam(name="det")),
    "Slab": lambda: _bodied(Slab(name="det")),
    "Roof": lambda: _bodied(Roof(name="det")),
    "Space": lambda: _bodied(Space(name="det")),
}


# ── the containers ───────────────────────────────────────────────────────────
#
# Each carries its own body, because ``.anchor()`` derives the child's frame
# from the host's authored geometry and refuses a host that has none.

def _host_body(leaf):
    return Box(name=leaf, start=Point(x=0, y=0, z=0),
               end=Point(x=4000, y=300, z=3000))


def _layered_wall():
    """A LAYERED host — the shape a real facade leaf has.

    Its own class of bug, and not a hypothetical one: the layer slicer picks
    "the single Box/Extrude body child", and an anchored detail is a prism
    child. Counting it made a layered wall refuse every anchor with a message
    about the buildup envelope — found by anchoring an eaves cornice onto
    ``gable-detail``'s facade, which is precisely what this capability is for.
    """
    leaf = Wall(name="host").add(_host_body("body"))
    leaf.layers = LayerSet(
        name="host_type", outward=(0, -1, 0),
        layers=[Material(key="Brick_Red_DK", thickness_mm=108),
                Material(key="MineralWool_Facade34", thickness_mm=190)])
    return leaf


CONTAINERS = {
    "Wall": lambda: Wall(name="host").add(_host_body("body")),
    "LayeredWall": _layered_wall,
    "Column": lambda: Column(name="host").add(_host_body("body")),
    "Beam": lambda: Beam(name="host").add(_host_body("body")),
    "Slab": lambda: Slab(name="host").add(_host_body("body")),
    "Roof": lambda: Roof(name="host").add(_host_body("body")),
    "Space": lambda: Space(name="host").add(_host_body("body")),
    "Element": lambda: Element(ifc_class="IfcBuildingElementProxy",
                               name="host").add(_host_body("body")),
}


#: The cells that left the matrix, and why. A ``Space`` is AIR, so
#: a PHYSICAL child does not go inside one by ANY verb — ``.add()`` always said
#: so and ``.anchor()`` did not, which meant ``space.anchor(wall)`` compiled
#: clean and wrote ``IfcRelAggregates(IfcSpace -> IfcWall)``: the wall as a
#: PART of the room's air. The rows are listed rather than deleted, and
#: ``test_a_physical_child_is_refused_by_a_space_not_dropped`` walks exactly
#: this set asserting each RAISES — so a cell that quietly became legal again
#: fails there instead of vanishing. Geometry primitives and ``Space`` stay in
#: the matrix under ``Space``: a shape is not a building element, and a space
#: decomposing into spaces is what ``IfcRelAggregates`` is FOR.
REFUSED = frozenset(("Space", child) for child in
                    ("Wall", "Column", "Beam", "Slab", "Roof", "Element"))

#: Every pairing the matrix walks — the full product minus :data:`REFUSED`.
CELLS = [(container, child)
         for container in sorted(CONTAINERS) for child in sorted(CHILDREN)
         if (container, child) not in REFUSED]


def _project(container_name, child_name):
    proj = Project(name="matrix")
    host = CONTAINERS[container_name]()
    host.anchor(CHILDREN[child_name](), along=1000, up=_LIFT)
    proj.add(host)
    return proj


#: Every way a vertex coordinate is spelled in the emitted STEP. A Mesh does
#: NOT emit ``IFCCARTESIANPOINT`` — its vertices ride a tessellated point LIST
#: — so a scan that knew only the first spelling would have read "dropped" for
#: every Mesh cell and been wrong about which half of the matrix was broken.
_POINT_ENTITIES = ("IFCCARTESIANPOINT(", "IFCCARTESIANPOINTLIST3D(")


def _cartesian_z(step: str):
    """Every Z coordinate of every vertex in the STEP text.

    Reading the serialized file rather than the model tree on purpose: it is
    the artefact the viewer loads, so a product that exists but reaches no
    representation cannot fool it.
    """
    import re

    zs = []
    for line in step.splitlines():
        head = line.partition("=")[2].lstrip()
        if not head.startswith(_POINT_ENTITIES):
            continue
        for triple in re.findall(r"\(([-0-9.E+,]+)\)", head):
            coords = triple.split(",")
            if len(coords) == 3:
                try:
                    zs.append(float(coords[2]))
                except ValueError:
                    pass
    return zs


def compile_pair(container_name, child_name):
    proj = _project(container_name, child_name)
    report = validate_project_report(proj)
    assert not report.errors, (
        f"{container_name}.anchor({child_name}) failed validation: "
        f"{report.errors}")
    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, (
        f"{container_name}.anchor({child_name}) failed to compile: "
        f"{result.error}")
    return result.ifc_content


@pytest.mark.parametrize("container_name,child_name", CELLS)
def test_every_type_anchors_into_every_container_and_renders(
        container_name, child_name):
    """The matrix. Not "it compiled" — "the geometry is at the anchored height"."""
    step = compile_pair(container_name, child_name)
    top = max(_cartesian_z(step), default=float("-inf"))
    assert top > _FLOOR_M, (
        f"{container_name}.anchor({child_name}): nothing in the compiled IFC "
        f"reaches z={_FLOOR_M} m — the anchored child was DROPPED (highest "
        f"point found: {top}). A container that refuses the pairing raises; a "
        f"container that drops it compiles clean, which is what this asserts "
        f"against."
    )


def test_the_lift_is_what_makes_the_scan_discriminating():
    """Guards the fixture: with no anchored child, nothing reaches the floor.

    Without this, a host body that happened to be tall enough would make every
    cell above pass no matter what the emitters did.
    """
    proj = Project(name="matrix")
    proj.add(CONTAINERS["Wall"]())
    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success
    assert max(_cartesian_z(result.ifc_content), default=0.0) <= _FLOOR_M


@pytest.mark.parametrize("cell", [("Wall", "Sweep"), ("Slab", "Revolve"),
                                  ("Column", "Bar"), ("Space", "Pipe"),
                                  ("Roof", "Wall"), ("Beam", "Mesh")])
def test_matrix_catches_a_silent_drop(monkeypatch, cell):
    """Falsification 1 — the EMISSION half. Stub the generic child path to
    return nothing; the cells that ride it must go red.

    ``_create_child_product`` returning ``None`` is exactly the silent-drop
    shape: the caller logs a warning and carries on, and the compile still
    SUCCEEDS. If the matrix asserted "no exception" this test would fail to
    fail, which is the whole reason it exists.
    """
    from lite_step.ifc import generator

    container_name, child_name = cell
    # Assert the injection lands on the symbol the emitters actually call.
    assert generator._create_child_product.__module__ == generator.__name__
    monkeypatch.setattr(generator, "_create_child_product",
                        lambda *a, **k: None)

    step = compile_pair(container_name, child_name)
    top = max(_cartesian_z(step), default=float("-inf"))
    assert top <= _FLOOR_M, (
        f"{container_name}.anchor({child_name}): the stub did not drop the "
        f"child, so this cell's assertion is not exercising the emitter path "
        f"it claims to")


@pytest.mark.parametrize("container_name,child_name", CELLS)
def test_matrix_catches_an_unbaked_anchor(monkeypatch, container_name,
                                          child_name):
    """Falsification 2 — the PLACEMENT half, over EVERY cell.

    Neuter the bake and nothing moves: each child stays at its container's
    local origin, metres below ``_LIFT``. Every cell of the matrix above must
    therefore fail here, which is what makes each cell's own assertion
    discriminating rather than incidentally true. This is the falsification
    that covers the pairings the emission stub cannot reach — a ``Mesh``
    inside an ``Element`` merges into the wrapper's own representation and
    never touches ``_create_child_product``.
    """
    from lite_step.compiler import frames

    monkeypatch.setattr(frames, "_bake_subtree", lambda *a, **k: None)

    step = compile_pair(container_name, child_name)
    top = max(_cartesian_z(step), default=float("-inf"))
    assert top <= _FLOOR_M, (
        f"{container_name}.anchor({child_name}): geometry reached "
        f"z={top} m with the bake stubbed out — this cell would pass without "
        f"the anchor doing anything, so it proves nothing")


def test_add_and_anchor_take_the_same_children_and_still_differ():
    """``.anchor()`` has no table; ``.add()``'s is not a TYPE table.

    Every physical container takes every geometry primitive AND every product
    through both verbs. What is left in ``.add()``'s table is semantics — the
    ``IfcSpace`` row below — and the difference between the verbs moved
    entirely onto what happens to the child's COORDINATES
    (``test_add_takes_products.py`` measures that; a type check never could).

    (The predecessor of this test used an INVALID Revolve — a profile
    crossing the axis — so every `pytest.raises(ValueError)` in it was
    catching pydantic's construction error rather than the child table's.
    It would have passed with the table deleted entirely. Fixed by building
    each probe with the module's valid factories.)
    """
    for name, make in CONTAINERS.items():
        host = make()
        if name in ("Site", "Space"):
            with pytest.raises(ValueError):
                host.add(_revolve())
            continue
        host.add(_revolve())
        make().add(_sweep())
        # A CONTAINER child, through BOTH verbs now.
        make().add(Element(ifc_class="IfcBuildingElementProxy", name="p1"))
        assert make().anchor(
            Element(ifc_class="IfcBuildingElementProxy", name="p2")) is not None
    # The one product refusal that survives, and it is about the RELATION:
    # a Space is air, and IfcRelSpaceBoundary is what an author means. Since
    # BOTH verbs give it — the reason is about what a Space IS, not
    # about how the child's coordinates were expressed, so the verb cannot
    # change it.
    with pytest.raises(ValueError, match="IfcRelSpaceBoundary"):
        Space(name="sp").add(Element(ifc_class="IfcBuildingElementProxy"))
    with pytest.raises(ValueError, match="IfcRelSpaceBoundary"):
        CONTAINERS["Space"]().anchor(
            Element(ifc_class="IfcBuildingElementProxy", name="p3"))


@pytest.mark.parametrize("container_name,child_name", sorted(REFUSED))
def test_a_physical_child_is_refused_by_a_space_not_dropped(
        container_name, child_name):
    """The cells :data:`REFUSED` took off the matrix, pinned as REFUSALS.

    Deleting the rows would have left "a Space takes a Wall" merely untested,
    which is how the pairing survived in the first place. Each one raises at
    the authoring line, and the message carries the relation an author
    actually means.
    """
    host = CONTAINERS[container_name]()
    with pytest.raises(ValueError, match="IfcRelSpaceBoundary"):
        host.anchor(CHILDREN[child_name](), along=1000, up=_LIFT)


def test_anchor_still_refuses_a_second_placement():
    """The two refusals that are about SEMANTICS, not types, are untouched."""
    host = CONTAINERS["Wall"]()
    from lite_step.models import Transform

    placed = Box(start=Point(x=0, y=0, z=0), end=Point(x=10, y=10, z=10),
                 placement=Transform(origin=Point(x=0, y=0, z=0)))
    with pytest.raises(ValueError, match="already carries placement="):
        host.anchor(placed)

    twice = _box()
    host.anchor(twice)
    with pytest.raises(ValueError, match="already anchored"):
        CONTAINERS["Roof"]().anchor(twice)


# ---------------------------------------------------------------------------
# The invariant: .anchor() never mirrors, it rotates
# ---------------------------------------------------------------------------

def test_anchored_extrude_mirrors_like_an_anchored_box():
    """Identical local authoring must land on the same SIDE of both facades.

    A ledge whose contour sits on the outer face and extrudes 60 mm along its
    own normal has to be proud of the wall on EVERY facade. It was not, and
    the numbers below were measured before that was fixed:

        south (facing=-1, det=+1): world y = [-3.060, -3.000]   proud   OK
        north (facing=+1, det=-1): world y = [+2.940, +3.000]   sunk    BUG

    ``ElementFrame.local_to_world`` composed ``[along, -facing*across, up]``,
    whose determinant flips with ``facing`` — a reflection. A Box was immune
    (it re-normalises to min/max corners, which absorbs the mirror); an
    Extrude was not, because its winding survives instead of mirroring, so
    the extrusion ran the wrong way THROUGH the host. It never raised:
    ``IfcAxis2Placement3D`` cannot encode a mirror, so there was nothing for
    the emitter to refuse.

    This test carried ``xfail(strict=True)`` from the day the bug was
    measured until the day it was fixed, which is why the fix could not land
    silently.
    """
    import numpy as np
    from lite_step.compiler.frames import stamp_frames, resolve_child_anchors

    def ledge_span(which):
        proj = Project(name="p")
        south = Wall(name="south")
        south.add(Box(name="body", start=Point(x=-4000, y=-3000, z=0),
                      end=Point(x=4000, y=-2700, z=3000)))
        north = Wall(name="north")
        north.add(Box(name="body", start=Point(x=-4000, y=2700, z=0),
                      end=Point(x=4000, y=3000, z=3000)))
        proj.add(south, north)
        host = south if which == "south" else north
        ex = Extrude(name="ledge", thickness=60, contour=[
            Point(x=0, y=0, z=0), Point(x=400, y=0, z=0),
            Point(x=400, y=0, z=100), Point(x=0, y=0, z=100)])
        host.anchor(ex, along=2000, up=1000)
        stamp_frames(proj)
        resolve_child_anchors(proj)
        stamp_frames(proj)
        # The contour plane, plus the 60 mm the solid runs along its normal.
        ys = [p.y for p in ex.contour]
        normal_y = np.sign(_contour_normal_y(ex.contour))
        return sorted([ys[0], ys[0] + 60 * normal_y])

    def _contour_normal_y(contour):
        from lite_step.compiler.frames import contour_normal
        return contour_normal(contour)[1]

    south_lo, south_hi = ledge_span("south")
    north_lo, north_hi = ledge_span("north")
    # Proud of its own facade on BOTH: below the south outer face, above the north.
    assert south_lo < -2700, (south_lo, south_hi)
    assert north_hi > 3000, (north_lo, north_hi)


def _four_facades():
    """A conventional ring: south/north (running X) + west/east (running Y).

    Deliberately NOT centred on the origin — the building sits at
    ``(+1000, +500)`` — because a symmetric fixture cannot tell a rotation
    from a reflection about the same axis. Two facades come out ``facing=-1``
    and two ``facing=+1``, which is the pair the invariant is about.
    """
    proj = Project(name="ring")
    walls = {}
    for name, (x0, y0, x1, y1) in {
        "south": (-3000, -2500, 5000, -2200),
        "north": (-3000, 3200, 5000, 3500),
        "west": (-3000, -2200, -2700, 3200),
        "east": (4700, -2200, 5000, 3200),
    }.items():
        w = Wall(name=name)
        w.add(Box(name="body", start=Point(x=x0, y=y0, z=0),
                  end=Point(x=x1, y=y1, z=3000)))
        walls[name] = w
        proj.add(w)
    return proj, walls


def test_the_bake_is_a_rotation_on_every_host():
    """``.anchor()`` never mirrors, it rotates — ``det > 0``, every host.

    The invariant ``test_opening_local_frame.py`` asserts on every opening
    void/fill/child placement, extended to the GENERIC anchor path that
    ``local_to_world`` drives. Measured from the bake's own output rather
    than from the frame's fields: three unit steps in local
    ``along``/``inset``/``up`` are the columns of its Jacobian, and their
    determinant is what ``IfcAxis2Placement3D`` would have to encode.

    A left-handed frame is not merely inconsistent — it is not representable,
    so nothing raises and the solid silently runs backwards through its host.
    """
    import numpy as np
    from lite_step.compiler.frames import stamp_frames

    proj, walls = _four_facades()
    stamp_frames(proj)
    seen = set()
    for name, wall in walls.items():
        f = wall._frame
        seen.add(f.facing)
        o = np.array(f.local_to_world(0.0, 0.0, 0.0))
        jac = np.array([np.array(f.local_to_world(1.0, 0.0, 0.0)) - o,
                        np.array(f.local_to_world(0.0, 1.0, 0.0)) - o,
                        np.array(f.local_to_world(0.0, 0.0, 1.0)) - o]).T
        det = float(np.linalg.det(jac))
        assert det == pytest.approx(1.0), (
            f"{name} (facing={f.facing}): the anchor bake has det={det:+.3f} — "
            f"det<0 is a MIRROR, which IfcAxis2Placement3D cannot express, so "
            f"it fails silently rather than raising")
    assert seen == {1, -1}, (
        f"fixture is degenerate — every wall came out facing={seen}, so it "
        f"could not have caught a determinant that flips with facing")


def test_the_bake_is_a_rotation_on_every_container_type():
    """The same determinant, across the whole container matrix.

    ``CONTAINERS`` spans the frame RULES (wall / beam / slab / fallback), and
    each rule picks its own ``along``/``across``/``up``. The invariant is a
    property of the composition, not of any one rule, so it is measured on
    all of them.
    """
    import numpy as np
    from lite_step.compiler.frames import stamp_frames

    for cname, make in CONTAINERS.items():
        proj = Project(name="det")
        host = make()
        proj.add(host)
        stamp_frames(proj)
        f = host._frame
        o = np.array(f.local_to_world(0.0, 0.0, 0.0))
        jac = np.array([np.array(f.local_to_world(1.0, 0.0, 0.0)) - o,
                        np.array(f.local_to_world(0.0, 1.0, 0.0)) - o,
                        np.array(f.local_to_world(0.0, 0.0, 1.0)) - o]).T
        assert float(np.linalg.det(jac)) == pytest.approx(1.0), \
            f"{cname} (facing={f.facing}, rule={f.rule}): anchor bake mirrors"


def test_an_anchored_detail_lands_proud_on_all_four_facades():
    """End-to-end twin of the determinant: one authored detail, four hosts.

    The expected world spans are written out per facade rather than derived,
    because a derivation would be the code under test spelled twice. Read
    them off the mental model: stand outside, ``along`` runs to your RIGHT
    from the corner on your left, ``inset=-40`` puts the band 40 mm proud.

        south — outward -y, right +x, so 500 in from x=-3000  -> x [-2500, -2200]
        north — outward +y, right -x, so 500 in from x=+5000  -> x [+4200, +4500]
        west  — outward -x, right -y, so 500 in from y=+3200  -> y [+2400, +2700]
        east  — outward +x, right +y, so 500 in from y=-2200  -> y [-1700, -1400]

    North and west are the two that MOVED : before it, both
    ran the other way and read [-2500, -2200] / [-1700, -1400], i.e. they
    started from the corner on the viewer's right. ``inset`` was already
    correct on all four, which is exactly why the bug was a reflection rather
    than a translation error.
    """
    from lite_step.compiler.frames import stamp_frames, resolve_child_anchors

    proj, walls = _four_facades()
    details = {}
    for name, wall in walls.items():
        # Asymmetric in ``along``: 0..300 of a 500 mm allowance, so the near
        # end is identifiable and a reversed run is not hidden by symmetry.
        d = Extrude(name=f"band_{name}", thickness=300, contour=[
            Point(x=0, y=0, z=0), Point(x=300, y=0, z=0),
            Point(x=300, y=0, z=120), Point(x=0, y=0, z=120)])
        wall.anchor(d, along=500, inset=-40, up=2000)
        details[name] = d
    stamp_frames(proj)
    resolve_child_anchors(proj)

    # name -> (run axis index, expected run span, outward axis index,
    #          expected proud coordinate)
    expected = {
        "south": (0, (-2500.0, -2200.0), 1, -2540.0),
        "north": (0, (4200.0, 4500.0), 1, 3540.0),
        "west": (1, (2400.0, 2700.0), 0, -3040.0),
        "east": (1, (-1700.0, -1400.0), 0, 5040.0),
    }
    for name, d in details.items():
        run_i, run_span, out_i, proud = expected[name]
        pts = [(p.x, p.y, p.z) for p in d.contour]
        run = [p[run_i] for p in pts]
        assert (min(run), max(run)) == pytest.approx(run_span), (
            f"{name}: along=500 put the band's run at "
            f"{(min(run), max(run))}, expected {run_span} — the frame ran "
            f"the wrong way, i.e. the bake mirrored instead of rotating")
        assert {round(p[out_i], 6) for p in pts} == {proud}, (
            f"{name}: inset=-40 should sit the contour plane 40 mm proud at "
            f"{proud}; got {sorted({p[out_i] for p in pts})}")
        zs = [p[2] for p in pts]
        assert min(zs) == pytest.approx(2000.0) and max(zs) == pytest.approx(2120.0), \
            f"{name}: up= drifted, so the failures above are not about facing"
