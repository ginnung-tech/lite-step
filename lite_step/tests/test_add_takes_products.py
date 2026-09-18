"""``.add()`` takes products, and KEEPS their world coordinates.

The companion to ``test_anchor_matrix.py``, and deliberately built on its
shape rather than a re-derivation: that file already walks "every type into
every container, and every pairing renders" for ``.anchor()``, so the matrix
here imports its containers, its children and its STEP scanner. A second copy
of that walk would be the shape — two implementations of one idea
agreeing until they do not.

What is new is the second assertion, and it is the whole PR.
``.add()`` and ``.anchor()`` now accept the same children; the only thing left
that distinguishes them is what happens to the child's COORDINATES. So the
test that matters is not "a product came out" — a product count passes
whatever the transform did to it, which is exactly how a mirrored bake shipped
undetected. It is: **the emitted solid stands at the
authored coordinates, un-transformed.**

Refusals that survive are pinned here too, each by the fix its message names.
"""
from __future__ import annotations

import pytest

from lite_step.models import (
    Point, Project, Box, Wall, Column, Beam, Slab, Roof, Space, Element, Site,
    Window, Door, SpatialElement, Mesh,
)
from lite_step.compiler.executor import (
    normalize_project_to_meters, validate_project_report,
)
from lite_step.ifc.generator import generate_ifc

# The matrix fixtures, borrowed rather than restated. ``_cartesian_z`` reads
# the SERIALIZED STEP (both point spellings, including the tessellated list a
# Mesh rides), so a product that exists but reaches no representation cannot
# fool it.
from lite_step.tests.test_anchor_matrix import (
    CONTAINERS, _LIFT, _FLOOR_M, _cartesian_z, _bodied,
)


#: Every product a physical container's ``.add()`` takes. Geometry is covered
#: by ``test_container_children.py``; this file is about the row that is new.
def _product(kind):
    """A bodied product authored at ``_LIFT`` — i.e. in WORLD coordinates,
    far above every container body in ``CONTAINERS`` (all of which top out at
    3000 mm). ``.add()`` claims those coordinates are already final, so a hit
    at this elevation is the claim being honoured; a hit anywhere else is the
    claim being broken, and a hit nowhere is a silent drop.
    """
    def body(c):
        c.add(Box(name="body", start=Point(x=0, y=0, z=_LIFT),
                  end=Point(x=300, y=120, z=_LIFT + 200)))
        return c
    return body({
        "Element": lambda: Element(ifc_class="IfcBuildingElementProxy", name="det"),
        "Wall": lambda: Wall(name="det"),
        "Column": lambda: Column(name="det"),
        "Beam": lambda: Beam(name="det"),
        "Slab": lambda: Slab(name="det"),
        "Roof": lambda: Roof(name="det"),
        "Space": lambda: Space(name="det"),
    }[kind]())


PRODUCTS = ("Element", "Wall", "Column", "Beam", "Slab", "Roof", "Space")

#: The matrix's containers. ``Space`` is excluded and gets its own section
#: below — it refuses every product on purpose, and that refusal is a claim
#: about IFC's relations rather than a hole in the matrix.
PHYSICAL_CONTAINERS = tuple(n for n in sorted(CONTAINERS) if n != "Space")


def _compile_added(container_name, product_name):
    proj = Project(name="add-matrix")
    host = CONTAINERS[container_name]()
    host.add(_product(product_name))
    proj.add(host)
    report = validate_project_report(proj)
    assert not report.errors, (
        f"{container_name}.add({product_name}) failed validation: "
        f"{report.errors}")
    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, (
        f"{container_name}.add({product_name}) failed to compile: "
        f"{result.error}")
    return result.ifc_content


# ---------------------------------------------------------------------------
# 1. Every (container, product) pairing puts geometry in the compiled file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("container_name", PHYSICAL_CONTAINERS)
@pytest.mark.parametrize("product_name", PRODUCTS)
def test_every_product_adds_into_every_container_and_renders(
        container_name, product_name):
    """Not "it compiled" — a warn-and-drop compiles perfectly.

    Falsification for the whole row: revert ``_PRODUCT_CHILDREN`` out of
    ``allowed_children_for`` and every cell fails at CONSTRUCTION, inside
    ``_compile_added``'s ``.add()`` call, before any assertion runs.
    """
    step = _compile_added(container_name, product_name)
    top = max(_cartesian_z(step), default=float("-inf"))
    assert top > _FLOOR_M, (
        f"{container_name}.add({product_name}): nothing in the compiled IFC "
        f"reaches z={_FLOOR_M} m — the added product was DROPPED (highest "
        f"point found: {top}). The aggregation that renders it is the same "
        f"one .anchor() has used since v22.3.0; a backend without it must "
        f"fall back, never drop.")


def test_the_lift_is_what_makes_the_scan_discriminating():
    """Guards the fixture, exactly as the anchor matrix guards its own: with
    no added product, nothing reaches the floor. Without this, a container
    body that happened to be tall enough would make every cell above pass no
    matter what the emitters did."""
    proj = Project(name="add-matrix")
    proj.add(CONTAINERS["Wall"]())
    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success
    assert max(_cartesian_z(result.ifc_content), default=0.0) <= _FLOOR_M


_ADDED_SPAN_M = (4.0, 4.3, 1.2, 1.32, 9.0, 9.2)


def _added_wall_project():
    """One Wall added into another, at coordinates that are NOWHERE NEAR the
    host's frame origin — so any bake, any rebase and any mirror moves it.

    The host runs along +X from x=0; the child stands at x=[4000, 4300],
    y=[1200, 1320], z=[9000, 9200]. Its y and z are outside the host's body
    entirely, which is legal precisely because ``.add()`` makes no claim about
    the host's frame.
    """
    proj = Project(name="world-coords")
    host = CONTAINERS["Wall"]()
    child = Wall(name="det")
    child.add(Box(name="body", start=Point(x=4000, y=1200, z=9000),
                  end=Point(x=4300, y=1320, z=9200)))
    host.add(child)
    proj.add(host)
    return proj, child


def test_an_added_product_keeps_its_world_coordinates():
    """The whole claim of the PR, and the reason this test is not a count.

    ``.add()`` says "this child's coordinates are already world coordinates".
    A product COUNT — or a "geometry reached the file" scan — is satisfied by
    a child that was rotated, mirrored, rebased onto the host's frame, or
    translated by the host's origin. Each of those is a real failure mode this
    compiler has shipped (``test_anchor_matrix``'s own docstring records the
    mirror that ``IfcAxis2Placement3D`` could not express and therefore could
    not raise on), and each leaves the count at exactly one.

    So the assertion is on the NUMBERS: after the full normalize + stamp +
    bake + emit path, the child's authored extent is byte-for-byte the extent
    it was written with. ``test_the_coordinate_assertion_is_what_catches_it``
    below demonstrates the count passing where this fails.
    """
    proj, child = _added_wall_project()
    normalized = normalize_project_to_meters(proj)
    result = generate_ifc(normalized)
    assert result.success, result.error

    # The model tree, after every pass that could have moved it.
    body = next(c for c in _find(normalized, "det")._elements
                if type(c).__name__ == "Box")
    got = (body.start.x, body.end.x, body.start.y,
           body.end.y, body.start.z, body.end.z)
    assert got == pytest.approx(_ADDED_SPAN_M), (
        f"an .add()ed product moved: authored {_ADDED_SPAN_M}, emitted {got}. "
        f".add() claims the child's coordinates are already world "
        f"coordinates — a pass that transforms them has broken the verb, and "
        f"a product count would not have noticed")

    # …and the same numbers survive into the SERIALIZED file, which is the
    # artefact a viewer loads. A product whose tree is right and whose
    # representation is placed elsewhere renders in the wrong place.
    # A Box body emits as an extruded area solid whose placement sits at the
    # solid's CENTRE, so the highest Cartesian point in the file is the
    # authored span's midpoint (9.0..9.2 -> 9.1), not its top. Asserted on
    # that exact number rather than on "> some floor": a floor test passes for
    # a child dropped anywhere above it.
    zs = _cartesian_z(result.ifc_content)
    assert max(zs) == pytest.approx(9.1, abs=1e-6), sorted(set(zs))[-4:]


def _find(project, leaf):
    """The element whose ``name`` leaf is ``leaf``, anywhere in the tree."""
    from lite_step.compiler.executor import _iter_storey_identity_elements
    for storey in project.storeys:
        for elem in _iter_storey_identity_elements(storey):
            if getattr(elem, "name", None) == leaf:
                return elem
    raise AssertionError(f"no element named {leaf!r}")


def test_the_coordinate_assertion_is_what_catches_it():
    """Falsification of test 2, in the file rather than in a shell transcript.

    ``.anchor()`` is the transform ``.add()`` must not apply. Anchor the SAME
    child with the SAME authored coordinates and the product count is
    identical, the "geometry reached the file" scan still passes — and the
    coordinates are somewhere else entirely. A test written as a count cannot
    tell these two projects apart; this one can, and that is the only reason
    it is written the way it is.
    """
    proj = Project(name="anchored-twin")
    host = CONTAINERS["Wall"]()
    child = Wall(name="det")
    child.add(Box(name="body", start=Point(x=4000, y=1200, z=9000),
                  end=Point(x=4300, y=1320, z=9200)))
    host.anchor(child, along=0, up=0)
    proj.add(host)
    normalized = normalize_project_to_meters(proj)
    result = generate_ifc(normalized)
    assert result.success, result.error

    # The COUNT is the same as the added twin's — this is the assertion that
    # would have passed trivially.
    added_proj, _ = _added_wall_project()
    added = generate_ifc(normalize_project_to_meters(added_proj))
    def walls(step):
        return sum(1 for line in step.splitlines()
                   if line.partition("=")[2].lstrip().startswith("IFCWALL("))
    assert walls(result.ifc_content) == walls(added.ifc_content) > 1

    # The COORDINATES are not.
    body = next(c for c in _find(normalized, "det")._elements
                if type(c).__name__ == "Box")
    got = (body.start.x, body.end.x, body.start.y,
           body.end.y, body.start.z, body.end.z)
    assert got != pytest.approx(_ADDED_SPAN_M), (
        "the anchored twin landed on the authored coordinates, so the "
        "coordinate assertion in test 2 is not discriminating — it would "
        "pass whether the bake ran or not")


# ---------------------------------------------------------------------------
# 3. Space.add(<product>) still refuses, and names the relation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("product_name", PRODUCTS)
def test_space_add_product_refuses_and_names_the_boundary_relation(product_name):
    """A Space is AIR. "This wall bounds this room" is ``IfcRelSpaceBoundary``,
    which this compiler already emits from the geometry — so widening the row
    would ship a second, wrong spelling of a relation that exists.

    The message must name it: an author told only "invalid child" reaches for
    ``Element(ifc_class="IfcSpace")`` and gets the wrong model quietly.
    """
    space = Space(name="kitchen").add(
        Box(name="air", start=Point(x=0, y=0, z=0),
            end=Point(x=4000, y=3000, z=2700)))
    with pytest.raises(ValueError) as exc:
        space.add(_product(product_name))
    msg = str(exc.value)
    assert "IfcRelSpaceBoundary" in msg, msg
    assert "space_boundaries.py" in msg, msg


def test_the_relation_the_refusal_names_is_really_emitted():
    """The advice has to WORK — the probe finding, where the old
    refusal pointed at a spelling that hit a second refusal. A wall standing
    as a PEER of the space (what the message tells the author to write) really
    does produce the boundary relation."""
    pytest.importorskip("ifcopenshell")
    proj = Project(name="boundary")
    space = Space(name="kitchen").add(
        Box(name="air", start=Point(x=0, y=0, z=0),
            end=Point(x=4000, y=3000, z=2700)))
    wall = Wall(name="north").add(
        Box(name="body", start=Point(x=0, y=3000, z=0),
            end=Point(x=4000, y=3200, z=2700)))
    proj.add(space, wall)
    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    assert "IFCRELSPACEBOUNDARY" in result.ifc_content.upper()


# ---------------------------------------------------------------------------
# 4. Wall.add(Window) refuses at the .add() LINE, not at validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make,verb", [
    (lambda: Window(width=1200, height=1400, name="w0"), "window"),
    (lambda: Door(width=900, height=2100, name="d0"), "door"),
])
def test_adding_an_opening_raises_at_the_authoring_line(make, verb):
    """An opening carries a SIZE and no position since v20.0.0, so ``.add()``'s
    claim — "these coordinates are already world coordinates" — is one it
    cannot satisfy: there are no coordinates.

    This was already a compile error (``_opening_placement_errors``); moving
    it to the authoring line is what makes the traceback name the line that
    wrote it rather than a validation pass three calls deep. Same pairing as
    ``frames.check_anchor_rotation`` and the containment-cycle refusal: early
    check for the ordinary case, validator backstop for a hand-built tree.
    """
    wall = CONTAINERS["Wall"]()
    with pytest.raises(ValueError) as exc:
        wall.add(make())
    msg = str(exc.value)
    assert "cannot be ADDED" in msg, msg
    assert f"wall.anchor({verb}" in msg, msg
    assert f"body.opening({verb}" in msg, msg
    # …and NOTHING was appended: a refused .add() never partially mutates.
    assert all(type(c).__name__ == "Box" for c in wall.elements)


def test_the_opening_refusal_is_at_add_not_at_validate():
    """The distinction this test exists for. Before the early check, the same
    model VALIDATED red; now it never gets built. Proven by the fact that a
    tree assembled around ``.add()`` still validates red — so the validator
    check is intact, and the new refusal is genuinely earlier rather than a
    relocation of the same one."""
    wall = CONTAINERS["Wall"]()
    wall._elements.append(Window(width=1200, height=1400, name="w0"))
    proj = Project(name="hand-built")
    proj.add(wall)
    errors = validate_project_report(proj).errors
    assert any("needs a position" in e for e in errors), errors


@pytest.mark.parametrize("container_name", sorted(CONTAINERS))
def test_no_container_takes_an_opening_through_add(container_name):
    """Every container, not just the two that host opening fills. A row of
    ``IfcWall``/``IfcCurtainWall`` only decides WHICH error
    an author got rather than whether they got one."""
    with pytest.raises(ValueError, match="cannot be ADDED"):
        CONTAINERS[container_name]().add(
            Window(width=1200, height=1400, name="w0"))


# ---------------------------------------------------------------------------
# 5. The SpatialElement facility gate is untouched
# ---------------------------------------------------------------------------

def test_a_facility_part_outside_a_facility_is_still_refused():
    """``facilities.validate_nesting`` runs FIRST and unchanged. Nothing in
    this PR widens it: WR31 is the schema's call, and its docstring's claim to
    be "the only defence" is literal — ``ifcopenshell.validate`` does not
    report WR31, so a facility in the wrong relation passes the conformance
    gate and is simply invisible in the tree."""
    part = SpatialElement(ifc_class="IfcBridgePart", name="deck",
                          usage="VERTICAL")
    with pytest.raises(ValueError, match="must sit inside a facility"):
        Site(name="s").add(part)
    # …and the same part is refused by a PHYSICAL container too, with the
    # same message, because the gate reads the container's class always.
    with pytest.raises(ValueError, match="must sit inside a facility"):
        CONTAINERS["Wall"]().add(
            SpatialElement(ifc_class="IfcBridgePart", name="deck2",
                           usage="VERTICAL"))


def test_a_facility_inside_its_facility_still_works():
    """The gate refuses a wrong nesting, not nesting — the corpus pattern
    (``reference_bridge_facility_pattern``) is untouched."""
    bridge = SpatialElement(ifc_class="IfcBridge", name="storstroem")
    deck = SpatialElement(ifc_class="IfcBridgePart", name="deck",
                          usage="VERTICAL")
    deck.add(Box(name="slab", start=Point(x=0, y=0, z=0),
                 end=Point(x=20000, y=8000, z=400)))
    bridge.add(deck)
    site = Site(name="s")
    site.add(Mesh(name="terrain",
                  vertices=[Point(x=0, y=0, z=0), Point(x=30000, y=0, z=0),
                            Point(x=30000, y=20000, z=0), Point(x=0, y=0, z=100)],
                  faces=[(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]))
    site.add(bridge)
    proj = Project(name="bridge")
    proj.add(site)
    assert validate_project_report(proj).errors == []


def test_a_facility_into_a_wall_is_refused_without_offering_anchor():
    """The one pairing WR31 has nothing to say about, and the reason the
    refusal picks its own remedy.

    Neither clause of ``validate_nesting`` fires for a FACILITY inside an
    ``IfcWall``, so it falls to the table — and ``.anchor()`` WOULD take it,
    because that verb has no table. Offering it would be the probe
    finding one message later: advice that fails on the next line, or worse,
    advice that succeeds and builds nonsense. A facility is a PLACE."""
    with pytest.raises(ValueError) as exc:
        CONTAINERS["Wall"]().add(
            SpatialElement(ifc_class="IfcBridge", name="storstroem"))
    msg = str(exc.value)
    assert "Site -> facility -> part -> products" in msg, msg
    assert ".anchor()" not in msg, msg


def test_spatial_element_is_a_container_in_the_taxonomy():
    """It is ``_elements``-bearing and holds no points of its own — the tuple's
    own rule. It was missing, so every pass asking "is this a container?" got
    False for the one node type whose whole job is holding other things."""
    from lite_step.models import taxonomy as tx
    assert tx.is_container(SpatialElement) is True
    assert tx.is_container("SpatialElement") is True
    # …and openings stay OUT, for the reason the tuple's docstring gives.
    assert tx.is_container(Window) is False
    assert tx.is_container(Door) is False


# ---------------------------------------------------------------------------
# 6. A nested body-less Wall is now a validation error (the executor hoist)
# ---------------------------------------------------------------------------

def test_a_nested_wall_without_a_body_is_a_validation_error():
    """The pre-existing gap this PR's widening made reachable.

    The Wall body check was written per ELEMENT; only the WALK was per storey
    (``for elem in storey.elements``), so a Wall nested inside anything got NO
    body check at all. It then reached the generator, which picks "the first
    Box/Extrude child", found none, and emitted a wall-shaped hole with no
    error anywhere. Hoisting the walk onto ``_iter_identity_elements`` fixes
    it; reverting the hoist makes this test fail.

    The wall is EMPTY, not merely body-less: made a body-less container
    legal when it has ``.add()``ed children to derive a frame from, so the
    refusal this hoist has to reach is now the empty-basis one. See
    ``test_bodyless_container.py`` for the rule itself.
    """
    proj = Project(name="nested-bodyless")
    host = CONTAINERS["Wall"]()
    host.add(Wall(name="hollow"))          # no body AND no children
    proj.add(host)
    errors = validate_project_report(proj).errors
    assert any("no children to derive a frame from" in e for e in errors), errors
    assert any("hollow" in e or "Wall" in e for e in errors), errors


def test_the_top_level_wall_check_is_unchanged():
    """The hoist must not have moved the check, only widened its reach: the
    element is visited, and still first."""
    proj = Project(name="top-bodyless")
    proj.add(Wall(name="hollow"))
    errors = validate_project_report(proj).errors
    assert any("no children to derive a frame from" in e for e in errors), errors


def test_a_nested_wall_with_a_body_validates_clean():
    """The other side of the hoist — it must not manufacture errors for the
    nesting this PR exists to allow."""
    proj = Project(name="nested-bodied")
    host = CONTAINERS["Wall"]()
    host.add(_product("Wall"))
    proj.add(host)
    assert validate_project_report(proj).errors == []


def test_a_nested_wall_with_two_bodies_is_also_caught():
    """The second per-element Wall rule rides the same hoist: the generator
    uses the FIRST prism and would silently drop the rest."""
    proj = Project(name="nested-two-bodies")
    host = CONTAINERS["Wall"]()
    nested = Wall(name="det")
    nested.add(Box(name="a", start=Point(x=0, y=0, z=_LIFT),
                   end=Point(x=300, y=120, z=_LIFT + 200)))
    nested.add(Box(name="b", start=Point(x=400, y=0, z=_LIFT),
                   end=Point(x=700, y=120, z=_LIFT + 200)))
    host.add(nested)
    proj.add(host)
    errors = validate_project_report(proj).errors
    assert any("contains 2 Box/Extrude children" in e for e in errors), errors


def test_the_hoist_still_excludes_anchored_children_from_the_body_count():
    """The rule the hoist had to preserve. An anchored Box is a DETAIL placed
    in the wall's frame, not a candidate for its body — counting one would
    make ``.anchor()`` unusable on any wall, nested or not."""
    proj = Project(name="nested-anchored-detail")
    host = CONTAINERS["Wall"]()
    nested = Wall(name="det")
    nested.add(Box(name="body", start=Point(x=0, y=0, z=_LIFT),
                   end=Point(x=3000, y=300, z=_LIFT + 2000)))
    nested.anchor(Box(name="corbel", start=Point(x=0, y=0, z=0),
                      end=Point(x=200, y=400, z=200)), along=1000, up=500)
    host.add(nested)
    proj.add(host)
    errors = validate_project_report(proj).errors
    assert not any("Box/Extrude children" in e for e in errors), errors
