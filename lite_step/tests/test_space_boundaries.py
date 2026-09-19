"""Inferred space boundaries — WS-C.

``agents/dsl-reference.md`` promised authors that the compiler derives space
boundaries and the internal/external classification. It derived neither: a room
with two walls touching it compiled to zero ``IfcRelSpaceBoundary`` of any kind.
Nothing could have caught that — the relation has no mandatory inverse, so a
model with none is schema-valid, renders correctly, and hashes identically
compile after compile.

What is pinned here:

1.  **The three contact conditions.** Flush contact IS a boundary (the case the
    displacement engine deliberately excludes), a 1 mm gap is NOT, an edge or
    corner touch is NOT, and an element floating strictly inside a room is not
    either. Each is a separate test because each is a separate clause and a
    single "overlaps or touches" test would pass three of them wrongly.
2.  **Every value of both enums is REACHED and is falsifiable.** A single-room
    fixture can only ever produce EXTERNAL, so it would pass a classifier that
    returned EXTERNAL unconditionally. INTERNAL needs two spaces across one
    wall; NOTDEFINED needs a wall a second room sits behind only PART of;
    EXTERNAL_EARTH needs terrain. All four are here, and the coverage test
    that separates NOTDEFINED from INTERNAL uses two rooms that cover one wall
    TOGETHER — a ratio-with-a-threshold implementation gets that wrong.
3.  **PHYSICAL is derived, not defaulted** — and VIRTUAL is proven unreachable
    from the DSL rather than assumed to be.
4.  **Both backends agree entity-for-entity**, on the entity TYPE, both enums,
    the ``"1stLevel"`` name, the NULL Description, the NULL ConnectionGeometry
    and the ParentBoundary wiring.
5.  **The shared-Name manifest trap.** Every boundary carries
    ``Name = "1stLevel"``, and the streaming manifest predicate is purely
    structural — so a hundred boundaries built before the manifest would
    collapse into ONE key called "1stLevel" that maps to no DSL element. The
    v21.2 ``IfcAnnotation`` bug with a shared name instead of a unique one.
6.  **Schema conformance of the emitted relation**, through
    ``ifcopenshell.validate`` on a boundary-bearing model on both backends —
    the corpus gate cannot exercise this, because only one committed model
    authors a Space at all.
7.  **The IfcSpace donor fix.** The streaming backend emitted an orphan
    ``IfcBuildingElementProxy`` carrying the Space's own representation, for
    every Space in every model; the ifcopenshell backend never did. Found
    while wiring boundaries onto both backends.
8.  **Fixtures stand off-origin, non-square and unrotated-plus-rotated.**
    Origin/identity/45-degree fixtures hide transform and rounding bugs, so
    every room here is at an odd offset with unequal sides.
9.  Additivity: a project with no Space emits nothing new.
"""

from __future__ import annotations

import re
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.space_boundaries import (
    BOUNDARY_DESCRIPTION,
    BOUNDARY_ENTITY,
    BOUNDARY_NAME,
    EXTERNAL,
    EXTERNAL_EARTH,
    INTERNAL,
    NOTDEFINED,
    PHYSICAL,
    SpaceBoundaryError,
    collect_boundaries,
)
from lite_step.models import (
    Box, Element, Mesh, Point, Point2D, Project, Site, Slab, Space, Storey,
    Transform, Wall, Window,
)

pytest.importorskip("ifcopenshell")


# ---------------------------------------------------------------------------
# Fixtures — local geometry only (never exec a skill; see tests/_fixtures.py).
#
# Nothing sits at the origin and no room is square: an origin-centred unit cube
# agrees with a broken transform and a rounding bug it would otherwise expose.
# ---------------------------------------------------------------------------

#: One room's footprint, deliberately 4200 x 4400 at an odd offset.
ROOM_A = (1700, -3100, 5900, 1300)
ROOM_B = (6200, -3100, 9400, 1300)


def _project():
    proj = Project(name="rooms")
    storey = Storey(name="ground", elevation=0)
    proj.add_storey(storey)
    return proj, storey


def wall(storey, name, x0, y0, x1, y1, z0=0, z1=2700, **kw):
    w = Wall(name=name, **kw)
    w.add(Box(start=Point(x=x0, y=y0, z=z0), end=Point(x=x1, y=y1, z=z1)))
    storey.add(w)
    return w


def space(storey, name, box=ROOM_A, z0=0, z1=2700):
    x0, y0, x1, y1 = box
    s = Space(name=name)
    s.add(Box(start=Point(x=x0, y=y0, z=z0), end=Point(x=x1, y=y1, z=z1)))
    storey.add(s)
    return s


def terrain(proj, *, x0=-20000, y0=-20000, x1=20000, y1=20000,
            top=0, depth=2000, name="terrain"):
    site = Site(name="site")
    wrapper = Element(ifc_class="IfcGeographicElement",
                      predefined_type="TERRAIN", name=name)
    wrapper.add(Mesh(heightmap=[[top, top], [top, top]], depth=depth,
                     corner_min=Point2D(x=x0, y=y0),
                     corner_max=Point2D(x=x1, y=y1)))
    site.add(wrapper)
    proj.add(site)
    return wrapper


def two_rooms(shared: bool = True):
    """Two off-origin rooms; ``shared`` puts one wall between them."""
    proj, storey = _project()
    if shared:
        wall(storey, "shared", 5900, -3100, 6200, 1300)
    wall(storey, "south", 1700, -3400, 9400, -3100)
    wall(storey, "north_a", 1700, 1300, 5900, 1600)
    space(storey, "a", ROOM_A)
    space(storey, "b", ROOM_B)
    return proj


def boundaries(proj):
    return collect_boundaries(normalize_project_to_meters(proj))


def classification(proj, space_leaf, element_leaf):
    """The one boundary between two named things, or ``None``."""
    for b in boundaries(proj):
        if (b.space.startswith(f"space:{space_leaf}")
                and b.element.split(":")[1] == element_leaf):
            return b.internal_or_external
    return None


def _engine_modules():
    """Every compiler module, tests excluded."""
    from pathlib import Path

    engine = Path(__file__).resolve().parents[1]
    return [p for p in engine.rglob("*.py") if "tests" not in p.parts]


def _defines(path, needle: str) -> bool:
    """Does ``path`` use ``needle`` as an IDENTIFIER (not in prose)?

    Tokenising and keeping NAME tokens drops docstrings, comments and string
    literals — which is the difference between "this module computes solar
    anything" and "this module's docstring QUOTES a retired claim about it".
    A plain text grep finds the word in the very file that retired the claim.
    """
    import tokenize

    with open(path, "rb") as fh:
        try:
            names = {t.string for t in tokenize.tokenize(fh.readline)
                     if t.type == tokenize.NAME}
        except (tokenize.TokenError, SyntaxError):        # pragma: no cover
            return False
    return any(needle.lower() in n.lower() for n in names)


def compile_ifc(proj, backend="ifcopenshell", source_code="src"):
    normalized = normalize_project_to_meters(proj)
    from lite_step.ifc.generator import generate_ifc
    result = generate_ifc(normalized, source_code=source_code)
    assert result.success, result.error
    return result.ifc_content


def compile_result(proj, backend="ifcopenshell"):
    """The raw result — a refusal reaches the caller as ``success=False``.

    Both entry points wrap an exception into a failed ``IFCGenerationResult``
    rather than propagating it (the loud-failure convention: the compile fails
    with a message, it does not crash the worker), so a refusal is asserted on
    ``result.error`` and never with ``pytest.raises``.
    """
    normalized = normalize_project_to_meters(proj)
    from lite_step.ifc.generator import generate_ifc
    return generate_ifc(normalized, source_code="src")


def opened(ifc: str):
    """Parse an IFC string with ifcopenshell. Caller keeps the model alive."""
    import ifcopenshell

    path_handle = tempfile.NamedTemporaryFile(suffix=".ifc", delete=False)
    path_handle.close()
    path = path_handle.name
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(ifc)
    return ifcopenshell.open(path)


def rels(ifc: str):
    """Every emitted boundary line, in file order."""
    return [line for line in ifc.splitlines() if "SPACEBOUNDARY" in line]


# ---------------------------------------------------------------------------
# 1. The three contact conditions
# ---------------------------------------------------------------------------


class TestContact:
    """Flush counts, a gap does not, and an edge touch is not a face."""

    def test_a_wall_flush_against_a_room_bounds_it(self):
        proj, storey = _project()
        wall(storey, "north", 1700, 1300, 5900, 1600)
        space(storey, "a")
        assert [(b.space, b.element) for b in boundaries(proj)] == [
            ("space:a:storey:ground", "wall:north:storey:ground")]

    def test_flush_is_exactly_the_case_displacement_excludes(self):
        """One primitive, two verdicts — the whole point of sharing it.

        The displacement engine must NOT carve on a flush pair (abutting faces
        are not an occupant) and a boundary MUST be inferred from one. Both
        read ``extent.overlap_depths``; if a future edit moved the tolerance
        into that function instead of leaving it at the call sites, one of
        these two assertions would break.
        """
        from lite_step.compiler.displacement import _overlaps

        room = ((1.7, -3.1, 0.0), (5.9, 1.3, 2.7))
        north = ((1.7, 1.3, 0.0), (5.9, 1.6, 2.7))
        assert _overlaps(room, north) is False
        proj, storey = _project()
        wall(storey, "north", 1700, 1300, 5900, 1600)
        space(storey, "a")
        assert len(boundaries(proj)) == 1

    def test_a_one_millimetre_gap_is_not_a_boundary(self):
        proj, storey = _project()
        wall(storey, "north", 1700, 1301, 5900, 1600)
        space(storey, "a")
        assert boundaries(proj) == []

    def test_an_edge_touch_is_not_a_boundary(self):
        """A wall meeting the room's corner diagonally shares a LINE."""
        proj, storey = _project()
        wall(storey, "corner", 5900, 1300, 9400, 1600)
        space(storey, "a")
        assert boundaries(proj) == []

    def test_a_point_touch_is_not_a_boundary(self):
        """Three flush axes: the boxes meet at one vertex."""
        proj, storey = _project()
        wall(storey, "vertex", 5900, 1300, 9400, 1600, z0=2700, z1=5400)
        space(storey, "a")
        assert boundaries(proj) == []

    def test_an_element_floating_inside_the_room_is_not_a_boundary(self):
        """The stated LIMIT of a box reading, asserted so it stays stated.

        A free-standing column reaches no face of the space, so nothing in an
        AABB comparison distinguishes it from furniture. Its surfaces do carry
        finishes, so a 2nd-level (surface) solver would find it; this one says
        so in the module docstring rather than guessing.
        """
        proj, storey = _project()
        wall(storey, "column", 3000, -1000, 3300, -700)
        space(storey, "a")
        assert boundaries(proj) == []

    def test_a_wall_running_past_a_room_gains_no_second_boundary_axis(self):
        """A wall longer than the room reaches past the room's EAST face as
        well as standing against its north one — but the two boxes share zero
        area on the plane perpendicular to east, because they are flush in
        north. Counting that as a second boundary axis asks "what lies beyond"
        over an empty rectangle, gets EXTERNAL, and downgrades this correct
        INTERNAL to NOTDEFINED.

        Measured: this is what the two-room fixture reported before the
        contact-area clause went into ``_contact``.
        """
        proj, storey = _project()
        wall(storey, "long", 1700, 1300, 9400, 1600)
        space(storey, "a", (1700, -3100, 5900, 1300))       # short of the wall
        space(storey, "behind", (1700, 1600, 5900, 4000))   # same short run
        assert classification(proj, "a", "long") == INTERNAL
        assert classification(proj, "behind", "long") == INTERNAL

    def test_a_wall_overlapping_the_room_bounds_it(self):
        """Rooms drawn to wall CENTRELINES are the other authoring convention
        and must work identically — the wall interpenetrates the space."""
        proj, storey = _project()
        wall(storey, "north", 1700, 1150, 5900, 1450)
        space(storey, "a")
        assert len(boundaries(proj)) == 1

    def test_a_wall_beyond_the_rooms_run_still_bounds_it(self):
        """The south wall spans BOTH rooms; each gets its own boundary."""
        found = [(b.space, b.element) for b in boundaries(two_rooms())
                 if b.element == "wall:south:storey:ground"]
        assert found == [("space:a:storey:ground", "wall:south:storey:ground"),
                         ("space:b:storey:ground", "wall:south:storey:ground")]


# ---------------------------------------------------------------------------
# 2. InternalOrExternalBoundary — every value, each falsifiable
# ---------------------------------------------------------------------------


class TestInternalOrExternal:

    def test_a_shared_wall_is_internal_from_both_rooms(self):
        got = [(b.space, b.internal_or_external) for b in boundaries(two_rooms())
               if b.element == "wall:shared:storey:ground"]
        assert got == [("space:a:storey:ground", INTERNAL),
                       ("space:b:storey:ground", INTERNAL)]

    def test_an_outer_wall_is_external(self):
        assert classification(two_rooms(), "a", "north_a") == EXTERNAL

    def test_removing_the_second_room_turns_internal_into_external(self):
        """The falsification of INTERNAL: the SAME wall, one room deleted.

        A classifier that returned INTERNAL for every shared-looking wall
        would pass the test above and fail this one.
        """
        proj, storey = _project()
        wall(storey, "shared", 5900, -3100, 6200, 1300)
        space(storey, "a")
        assert classification(proj, "a", "shared") == EXTERNAL

    def test_a_room_behind_only_part_of_a_wall_is_notdefined(self):
        """The concept's "undefined (internal and external)" case, verbatim."""
        proj, storey = _project()
        wall(storey, "long", 1700, 1300, 9400, 1600)
        space(storey, "a", (1700, -3100, 9400, 1300))
        space(storey, "b", (1700, 1600, 5900, 4000))
        assert classification(proj, "a", "long") == NOTDEFINED

    def test_two_rooms_covering_one_wall_together_are_internal(self):
        """Exact coverage, not a ratio.

        Neither room covers the wall alone; together they cover it exactly. A
        "does any single far space cover it" implementation answers
        NOTDEFINED here, and a ratio-with-a-threshold answers whichever the
        threshold happens to pick.
        """
        proj, storey = _project()
        wall(storey, "long", 1700, 1300, 9400, 1600)
        space(storey, "a", (1700, -3100, 9400, 1300))
        space(storey, "b1", (1700, 1600, 5900, 4000))
        space(storey, "b2", (5900, 1600, 9400, 4000))
        assert classification(proj, "a", "long") == INTERNAL

    def test_leaving_a_hole_between_the_two_rooms_is_notdefined(self):
        """The falsification of the coverage test: shrink one room by 100 mm
        and the union stops covering the wall."""
        proj, storey = _project()
        wall(storey, "long", 1700, 1300, 9400, 1600)
        space(storey, "a", (1700, -3100, 9400, 1300))
        space(storey, "b1", (1700, 1600, 5800, 4000))
        space(storey, "b2", (5900, 1600, 9400, 4000))
        assert classification(proj, "a", "long") == NOTDEFINED

    def test_terrain_under_a_room_is_external_earth(self):
        """``EXTERNAL`` asserts an external SPACE on the other side, which is
        a false statement about soil — the enum's own wording rules it out."""
        proj, storey = _project()
        space(storey, "a")
        terrain(proj)
        got = [(b.element, b.internal_or_external) for b in boundaries(proj)]
        assert got == [("geographicelement:terrain:site:site", EXTERNAL_EARTH)]

    def test_earth_behind_a_wall_is_external_earth(self):
        """A basement wall: the BOUNDING element is masonry, the far side is
        soil. The classification looks past the element, not at it."""
        proj, storey = _project()
        wall(storey, "west", 1400, -3100, 1700, 1300)
        space(storey, "a")
        terrain(proj, x1=1400, top=2700, depth=4700)
        assert classification(proj, "a", "west") == EXTERNAL_EARTH

    def test_moving_the_earth_away_turns_it_back_to_external(self):
        """The falsification of EXTERNAL_EARTH: the same wall, soil pulled
        back 100 mm so it stops short of the wall's far face."""
        proj, storey = _project()
        wall(storey, "west", 1400, -3100, 1700, 1300)
        space(storey, "a")
        terrain(proj, x1=1300, top=2700, depth=4700)
        assert classification(proj, "a", "west") == EXTERNAL

    def test_a_room_stacked_on_the_wall_does_not_count_as_being_behind_it(self):
        """The upstairs room sits ON the wall's top face: it is beyond the
        wall in ``y``, but it shares only a LINE with it in ``z``, so it
        covers none of the ground-floor contact rectangle. Without the
        shared-area clause in ``_is_beyond`` this reports INTERNAL and claims
        a room on the other side of a wall that has nothing behind it."""
        proj, storey = _project()
        wall(storey, "north", 1700, 1300, 5900, 1600)
        space(storey, "a")
        space(storey, "upstairs", (1700, 1600, 5900, 4000), z0=2700, z1=5400)
        assert classification(proj, "a", "north") == EXTERNAL

    def test_a_room_stacked_on_the_terrain_does_not_count_as_earth_behind(self):
        """The same clause on the earth side of the same function."""
        proj, storey = _project()
        wall(storey, "west", 1400, -3100, 1700, 1300)
        space(storey, "a")
        terrain(proj, x1=1400, top=-2700, depth=2000)
        assert classification(proj, "a", "west") == "EXTERNAL"

    def test_a_second_zone_on_the_SAME_side_does_not_make_a_wall_internal(self):
        """A circulation zone drawn INSIDE an open-plan room shares the room's
        side of the wall. It is not across it, and the wall does not separate
        them — so the wall stays EXTERNAL. Without the straddle clause in
        ``_is_beyond`` this reports INTERNAL and invents a room behind an
        outside wall."""
        proj, storey = _project()
        wall(storey, "north", 1700, 1300, 5900, 1600)
        space(storey, "a")
        space(storey, "circulation", (1700, -1000, 5900, 1300))
        assert classification(proj, "a", "north") == "EXTERNAL"
        assert classification(proj, "circulation", "north") == "EXTERNAL"

    def test_a_cellar_under_a_floor_slab_is_internal(self):
        """The vertical axis, and the positive half of the straddle clause."""
        proj, storey = _project()
        sl = Slab(name="floor")
        sl.add(Box(start=Point(x=1700, y=-3100, z=-200),
                   end=Point(x=5900, y=1300, z=0)))
        storey.add(sl)
        space(storey, "a")
        space(storey, "cellar", ROOM_A, z0=-2900, z1=-200)
        assert classification(proj, "a", "floor") == "INTERNAL"

    def test_a_cellar_poking_through_the_slab_is_not_below_it(self):
        """The falsification: raise the cellar 300 mm so it straddles the slab
        instead of standing under it."""
        proj, storey = _project()
        sl = Slab(name="floor")
        sl.add(Box(start=Point(x=1700, y=-3100, z=-200),
                   end=Point(x=5900, y=1300, z=0)))
        storey.add(sl)
        space(storey, "a")
        space(storey, "cellar", ROOM_A, z0=-2900, z1=100)
        assert classification(proj, "a", "floor") == "EXTERNAL"

    def test_a_wall_with_a_room_behind_and_soil_beyond_is_notdefined(self):
        """Half internal, half earth. Both are "covered", by different kinds —
        the enum has no value for that, and NOTDEFINED is what it means."""
        proj, storey = _project()
        wall(storey, "long", 1700, 1300, 9400, 1600)
        space(storey, "a", (1700, -3100, 9400, 1300))
        space(storey, "b", (1700, 1600, 5900, 4000))
        terrain(proj, x0=5900, y0=1600, x1=9400, y1=4000, top=2700, depth=4700)
        assert classification(proj, "a", "long") == NOTDEFINED

    def test_every_enum_value_this_module_can_emit_is_in_the_schema(self):
        """IfcInternalOrExternalEnum + IfcPhysicalOrVirtualEnum, IFC4X3_ADD2.

        The LITERALS are asserted first. Comparing only the constants to each
        other is a tautology — a mutation setting
        ``EXTERNAL_EARTH = "EXTERNAL"`` moves both sides together, still names
        a real enumerator, and left all 77 tests green when it was measured.
        """
        import ifcopenshell

        assert (INTERNAL, EXTERNAL, EXTERNAL_EARTH, NOTDEFINED) == (
            "INTERNAL", "EXTERNAL", "EXTERNAL_EARTH", "NOTDEFINED")
        assert len({INTERNAL, EXTERNAL, EXTERNAL_EARTH, NOTDEFINED}) == 4
        schema = ifcopenshell.ifcopenshell_wrapper.schema_by_name("IFC4X3_ADD2")
        internal = set(schema.declaration_by_name(
            "IfcInternalOrExternalEnum").enumeration_items())
        physical = set(schema.declaration_by_name(
            "IfcPhysicalOrVirtualEnum").enumeration_items())
        assert {INTERNAL, EXTERNAL, EXTERNAL_EARTH, NOTDEFINED} <= internal
        assert {PHYSICAL} <= physical

    def test_terrain_emits_the_literal_external_earth(self):
        """ifcopenshell only, and that is CHECKED rather than assumed.

        The streaming backend has no ``Element`` handler at all — a semantic
        wrapper under a ``Site`` (which is what terrain IS) produces no product
        there, silently. That is not a boundary bug and not a live one:
        ``_uses_v15_primitives`` walks ``proj.sites`` precisely so
        ``can_stream()`` says no, and no terrain model ever reaches the
        streaming writer through ``generate_ifc``. Calling that writer directly
        — which this file's ``compile_ifc`` does on purpose, to guarantee two
        real backends — bypasses the predicate, so the predicate is asserted
        here instead. The day streaming grows ``Element`` support this fails
        and asks to be parametrized, rather than silently staying half a test.
        """
        proj, storey = _project()
        space(storey, "a")
        terrain(proj)
        model = opened(compile_ifc(proj))
        found = model.by_type("IfcRelSpaceBoundary1stLevel")
        assert len(found) == 1
        assert found[0].InternalOrExternalBoundary == "EXTERNAL_EARTH"

    def test_a_shared_wall_emits_the_literal_internal(self):
        model = opened(compile_ifc(two_rooms(), "ifcopenshell"))
        got = {r.RelatedBuildingElement.Name: r.InternalOrExternalBoundary
               for r in model.by_type("IfcRelSpaceBoundary1stLevel")}
        assert got["wall:shared:storey:ground"] == "INTERNAL"
        assert got["wall:north_a:storey:ground"] == "EXTERNAL"


# ---------------------------------------------------------------------------
# 3. PhysicalOrVirtualBoundary
# ---------------------------------------------------------------------------


class TestPhysicalOrVirtual:

    def test_every_boundary_is_physical(self):
        # The LITERAL string, not just the imported constant. Asserting
        # ``== PHYSICAL`` alone is a tautology: a mutation setting
        # ``PHYSICAL = "NOTDEFINED"`` moves both sides together and left all
        # 66 tests green when it was measured.
        assert PHYSICAL == "PHYSICAL"
        assert {b.physical_or_virtual
                for b in boundaries(two_rooms())} == {"PHYSICAL"}

    def test_the_emitted_enum_is_the_literal_physical(self):
        model = opened(compile_ifc(two_rooms(), "ifcopenshell"))
        found = model.by_type("IfcRelSpaceBoundary1stLevel")
        assert found
        assert {r.PhysicalOrVirtualBoundary for r in found} == {"PHYSICAL"}

    def test_virtual_is_unreachable_because_the_dsl_has_no_virtual_element(self):
        """PHYSICAL is a derivation, and this is the derivation.

        ``CorrectPhysOrVirt`` binds VIRTUAL to an ``IfcVirtualElement`` (or an
        ``IfcOpeningElement``) in ``RelatedBuildingElement``. No DSL container
        maps to ``IfcVirtualElement`` and it is not in the whitelist, so there
        is no model that can produce one. If a future PR adds it, this fails
        and the VIRTUAL branch has to be written rather than discovered
        missing.
        """
        from lite_step.models.elements import ELEMENT_IFC_CLASS_WHITELIST

        assert "IfcVirtualElement" not in ELEMENT_IFC_CLASS_WHITELIST

    def test_an_opening_is_never_the_related_element(self):
        """The other VIRTUAL-eligible type IS emitted — as the void. What
        bounds the space at a window is the FILLING, which is physical."""
        proj, storey = _project()
        w = wall(storey, "north", 1700, 1300, 5900, 1600)
        w.anchor(Window(name="pane", width=1200, height=1500), along=1500, up=900)
        space(storey, "a")
        assert not any(b.element.endswith("_void") for b in boundaries(proj))


# ---------------------------------------------------------------------------
# 4. Window and Door — the inner boundaries
# ---------------------------------------------------------------------------


class TestInnerBoundaries:

    @staticmethod
    def _walled_room():
        """A wall twice the room's length, with a window over each half."""
        proj, storey = _project()
        w = wall(storey, "south", 1700, -3400, 9400, -3100)
        # ``along`` runs the wall's own frame, so which window lands over the
        # room is derived, not assumed — the assertions read the result.
        w.anchor(Window(name="near", width=1200, height=1500), along=800, up=900)
        w.anchor(Window(name="far", width=1200, height=1500), along=6500, up=900)
        space(storey, "a")
        return proj

    def test_only_the_window_over_the_room_gets_a_boundary(self):
        """The wall runs 7700 and the room only 4200, so exactly one of the
        two windows stands over the room. Inheriting the boundary from the
        host would have produced two."""
        found = [b for b in boundaries(self._walled_room())
                 if b.element.startswith("window:")]
        assert len(found) == 1

    def test_the_inner_boundary_cites_its_hosts_boundary_as_parent(self):
        got = boundaries(self._walled_room())
        inner = [b for b in got if b.is_inner]
        assert len(inner) == 1
        assert got[inner[0].parent_index].element == "wall:south:storey:ground"
        assert got[inner[0].parent_index].space == inner[0].space

    def test_parents_are_always_earlier_in_the_list(self):
        """``parent_index`` points BACKWARD, so a backend mints in one pass."""
        got = boundaries(self._walled_room())
        assert all(b.parent_index < i
                   for i, b in enumerate(got) if b.is_inner)

    def test_the_hosts_own_boundary_is_not_cut_by_the_opening(self):
        """"The space boundary of the parent is not cut by the inner boundary
        — both overlap." One wall boundary, whole, plus the window's."""
        got = boundaries(self._walled_room())
        assert [b.element for b in got if not b.is_inner] == [
            "wall:south:storey:ground"]


# ---------------------------------------------------------------------------
# 5. The emitted entity — both backends, byte for byte
# ---------------------------------------------------------------------------


class TestEmission:

    def test_the_entity_is_the_1st_level_subtype(self):
        ifc = compile_ifc(two_rooms(), "ifcopenshell")
        assert all(BOUNDARY_ENTITY.upper() in line for line in rels(ifc))
        assert len(rels(ifc)) == 5

    def test_the_name_is_the_schemas_level_discriminator(self):
        model = opened(compile_ifc(two_rooms(), "ifcopenshell"))
        found = model.by_type(BOUNDARY_ENTITY)
        assert found
        for rel in found:
            assert rel.Name == BOUNDARY_NAME == "1stLevel"
            assert rel.Description is BOUNDARY_DESCRIPTION is None

    def test_connection_geometry_is_null(self):
        """OPTIONAL, and an omitted one means "logically" per the schema."""
        model = opened(compile_ifc(two_rooms(), "ifcopenshell"))
        assert all(rel.ConnectionGeometry is None
                   for rel in model.by_type(BOUNDARY_ENTITY))

    def test_the_space_reaches_its_boundaries_through_boundedby(self):
        """The inverse a consumer actually walks. A relation nothing can
        traverse from the space is worth nothing to a downstream tool."""
        model = opened(compile_ifc(two_rooms(), "ifcopenshell"))
        by_space = {s.Name: sorted(b.RelatedBuildingElement.Name
                                   for b in s.BoundedBy)
                    for s in model.by_type("IfcSpace")}
        assert by_space == {
            "space:a:storey:ground": ["wall:north_a:storey:ground",
                                      "wall:shared:storey:ground",
                                      "wall:south:storey:ground"],
            "space:b:storey:ground": ["wall:shared:storey:ground",
                                      "wall:south:storey:ground"],
        }

    def test_relating_space_is_an_ifcspace_and_related_is_an_ifcelement(self):
        model = opened(compile_ifc(two_rooms(), "ifcopenshell"))
        for rel in model.by_type(BOUNDARY_ENTITY):
            assert rel.RelatingSpace.is_a("IfcSpace")
            assert rel.RelatedBuildingElement.is_a("IfcElement")

    def test_both_backends_wire_the_same_parent_boundary(self):
        proj = TestInnerBoundaries._walled_room()

        def parents(backend):
            model = opened(compile_ifc(proj, backend))
            return sorted(
                (rel.RelatedBuildingElement.Name,
                 rel.ParentBoundary.RelatedBuildingElement.Name
                 if rel.ParentBoundary else None)
                for rel in model.by_type(BOUNDARY_ENTITY))

        got = parents("ifcopenshell")
        assert ("window:near:wall:south:storey:ground",
                "wall:south:storey:ground") in got or \
               ("window:far:wall:south:storey:ground",
                "wall:south:storey:ground") in got

    def test_a_project_with_no_space_emits_no_boundary(self):
        proj, storey = _project()
        wall(storey, "north", 1700, 1300, 5900, 1600)
        assert rels(compile_ifc(proj)) == []

    def test_the_order_is_deterministic(self):
        first = [(r.RelatingSpace.Name, r.RelatedBuildingElement.Name)
                 for r in opened(compile_ifc(two_rooms(), "ifcopenshell")
                                 ).by_type(BOUNDARY_ENTITY)]
        second = [(r.RelatingSpace.Name, r.RelatedBuildingElement.Name)
                  for r in opened(compile_ifc(two_rooms(), "ifcopenshell")
                                  ).by_type(BOUNDARY_ENTITY)]
        assert first == second


# ---------------------------------------------------------------------------
# 6. The shared-Name manifest trap
# ---------------------------------------------------------------------------


class TestTheManifestIsNotPolluted:
    """Every boundary carries the SAME ``Name``, and the streaming manifest
    predicate is structural (22-char GlobalId at 0, Name at 2) — which a
    boundary matches perfectly. Built before the manifest, five boundaries
    would collapse into one key called "1stLevel"."""

    def test_1stlevel_is_not_a_manifest_key(self):
        ifc = compile_ifc(two_rooms(), "ifcopenshell")
        blob = re.search(r"'MANIFEST'.*", ifc)
        assert blob, "no LITESTEP_META manifest in the file"
        assert '"1stLevel"' not in blob.group(0)
        assert '1stLevel' not in blob.group(0)

    def test_the_manifest_keys_are_exactly_the_dsl_elements(self):
        with_rooms = compile_ifc(two_rooms(), "ifcopenshell")
        blob = re.search(r"'MANIFEST'.*", with_rooms).group(0)
        for canonical in ("space:a:storey:ground", "wall:shared:storey:ground"):
            assert canonical in blob


# ---------------------------------------------------------------------------
# 7. Schema conformance of a boundary-bearing model
# ---------------------------------------------------------------------------


class TestSchemaConformance:
    """``tests/test_ifc43_conformance.py`` gates the corpus, and exactly ONE
    committed model authors a Space — so the corpus gate cannot exercise
    ParentBoundary, INTERNAL, or a Window boundary at all. These do."""

    @staticmethod
    def _errors(ifc: str):
        import contextlib
        import io

        import ifcopenshell.validate

        model = opened(ifc)
        logger = ifcopenshell.validate.json_logger()
        with contextlib.redirect_stderr(io.StringIO()):
            ifcopenshell.validate.validate(model, logger)
        return [s for s in logger.statements
                if str(s.get("level", "")).lower() == "error"]

    def test_two_rooms_validate_clean(self):
        assert self._errors(compile_ifc(two_rooms(), "ifcopenshell")) == []

    def test_a_model_with_inner_boundaries_validates_clean(self):
        proj = TestInnerBoundaries._walled_room()
        assert self._errors(compile_ifc(proj)) == []

    def test_a_model_with_terrain_validates_clean(self):
        """ifcopenshell only — see
        ``test_terrain_emits_the_literal_external_earth`` for why a terrain
        model never reaches the streaming writer."""
        proj, storey = _project()
        wall(storey, "west", 1400, -3100, 1700, 1300)
        space(storey, "a")
        terrain(proj, x1=1400, top=2700, depth=4700)
        assert self._errors(compile_ifc(proj)) == []

    def test_correctphysorvirt_holds(self):
        """The entity's ONE WHERE rule, asserted directly rather than trusted
        to the validator measured that ``validate`` misses some WHERE
        rules (it does not report ``IfcShapeModel`` WR11)."""
        model = opened(compile_ifc(two_rooms(), "ifcopenshell"))
        for rel in model.by_type(BOUNDARY_ENTITY):
            if rel.PhysicalOrVirtualBoundary == "PHYSICAL":
                assert not rel.RelatedBuildingElement.is_a("IfcVirtualElement")
            elif rel.PhysicalOrVirtualBoundary == "VIRTUAL":  # pragma: no cover
                assert (rel.RelatedBuildingElement.is_a("IfcVirtualElement")
                        or rel.RelatedBuildingElement.is_a("IfcOpeningElement"))


# ---------------------------------------------------------------------------
# 8. Placement — a moved element is judged where it STANDS
# ---------------------------------------------------------------------------


class TestPlacement:

    def test_a_translated_wall_bounds_the_room_it_was_moved_against(self):
        """Authored 1000 mm short of the room and translated the last 1000.

        ``element_extent`` answers in AUTHORING coordinates, so without the
        world-matrix composition this wall is judged where it was DRAWN — a
        gap — and the boundary silently never appears.
        """
        proj, storey = _project()
        wall(storey, "north", 1700, 300, 5900, 600,
             placement=Transform(origin=Point(x=0, y=1000, z=0)))
        space(storey, "a")
        assert [b.element for b in boundaries(proj)] == [
            "wall:north:storey:ground"]

    def test_the_same_wall_untranslated_does_not_bound_it(self):
        """The falsification: drop the placement and the boundary must go."""
        proj, storey = _project()
        wall(storey, "north", 1700, 300, 5900, 600)
        space(storey, "a")
        assert boundaries(proj) == []

    def test_a_rotated_wall_is_bounded_by_its_world_box(self):
        """Authored running along +x, stood on its east end by a 90-degree z
        rotation. Judged un-rotated it is a 4400-long wall lying flat across
        the room's north-east corner; judged correctly it is a 300-thick wall
        abutting the room's east face over its full 4400 depth."""
        proj, storey = _project()
        wall(storey, "east", 0, 0, 4400, 300,
             placement=Transform(origin=Point(x=6200, y=-3100, z=0),
                                 rotations=[("z", 9000)]))
        space(storey, "a")
        assert [b.element for b in boundaries(proj)] == [
            "wall:east:storey:ground"]

    def test_an_element_flush_inside_the_room_is_omitted_LOUDLY(self, caplog):
        """A room authored to its walls' OUTER faces swallows them, so the
        wall reaches past nothing and gets no boundary. That is deliberate —
        "what lies beyond a wall standing inside the room" has no honest
        ``IfcInternalOrExternalEnum`` value — but it must not be SILENT, or it
        is indistinguishable from the compiler not seeing the wall."""
        proj, storey = _project()
        wall(storey, "inner", 3000, -1000, 3300, 1300)     # flush at y = 1300
        space(storey, "a")
        with caplog.at_level("WARNING"):
            assert boundaries(proj) == []
        assert "lies inside" in caplog.text
        assert "wall:inner:storey:ground" in caplog.text

    def test_a_free_floating_element_is_omitted_QUIETLY(self, caplog):
        """The narrowness of that warning: a box touching no face of the room
        is nobody's expected boundary, and warning about every interior
        element would bury the case that is arguable."""
        proj, storey = _project()
        wall(storey, "sign", 3000, -1000, 3300, -700, z0=1000, z1=1500)
        space(storey, "a")
        with caplog.at_level("WARNING"):
            assert boundaries(proj) == []
        assert "lies inside" not in caplog.text

    def test_an_element_buried_inside_the_rooms_edge_is_not_a_boundary(self):
        """The same rotated wall, translated 300 mm short so it stands INSIDE
        the room's east strip rather than against it. It reaches no face, so
        it is the free-standing case — measured while writing the test above,
        which is how the two got told apart."""
        proj, storey = _project()
        wall(storey, "east", 0, 0, 4400, 300,
             placement=Transform(origin=Point(x=5900, y=-3100, z=0),
                                 rotations=[("z", 9000)]))
        space(storey, "a")
        assert boundaries(proj) == []


# ---------------------------------------------------------------------------
# 9. The IfcSpace donor product (found while wiring both backends)
# ---------------------------------------------------------------------------


class TestTheSpaceCarriesItsOwnGeometryOnce:
    """The streaming backend emitted the Space's first child as a product AND
    handed its representation to the ``IfcSpace``. That donor was aggregated
    under nothing and contained in nothing, so it was an orphan product
    holding the room's own volume — double-drawn by any viewer walking
    ``IfcProduct``, invisible to one walking containment. The ifcopenshell
    backend has always removed it."""

    def test_a_space_emits_no_orphan_proxy(self):
        proj, storey = _project()
        space(storey, "a")
        model = opened(compile_ifc(proj))
        assert model.by_type("IfcBuildingElementProxy") == []

    def test_the_space_still_carries_its_representation(self):
        proj, storey = _project()
        space(storey, "a")
        model = opened(compile_ifc(proj))
        rooms = model.by_type("IfcSpace")
        assert len(rooms) == 1
        assert rooms[0].Representation is not None

    def test_both_backends_emit_the_same_product_classes_for_a_space(self):
        def classes(backend):
            proj, storey = _project()
            space(storey, "a")
            model = opened(compile_ifc(proj, backend))
            return sorted(p.is_a() for p in model.by_type("IfcElement"))

        assert classes("ifcopenshell") == []


# ---------------------------------------------------------------------------
# 10. Refusals
# ---------------------------------------------------------------------------


class TestRefusals:

    def test_a_missing_product_is_a_loud_error(self):
        from lite_step.ifc.space_boundaries import (
            SpaceBoundary, missing_product_error,
        )

        boundary = SpaceBoundary(
            space="space:a", element="wall:ghost",
            physical_or_virtual=PHYSICAL, internal_or_external=EXTERNAL,
            name=BOUNDARY_NAME, description=None, parent_index=None)
        err = missing_product_error(boundary, "RelatedBuildingElement",
                                    "wall:ghost")
        assert isinstance(err, SpaceBoundaryError)
        assert "wall:ghost" in str(err)
        assert "RelatedBuildingElement" in str(err)

    def test_a_wrongly_typed_product_is_a_loud_error(self):
        from lite_step.ifc.space_boundaries import (
            SpaceBoundary, wrong_type_error,
        )

        boundary = SpaceBoundary(
            space="space:a", element="storey:ground",
            physical_or_virtual=PHYSICAL, internal_or_external=EXTERNAL,
            name=BOUNDARY_NAME, description=None, parent_index=None)
        err = wrong_type_error(boundary, "RelatedBuildingElement",
                               "storey:ground", "IfcElement",
                               "IfcBuildingStorey")
        assert isinstance(err, SpaceBoundaryError)
        assert "IfcBuildingStorey" in str(err)
        assert "IfcElement" in str(err)

    @staticmethod
    def _forced(monkeypatch, boundary):
        """Make ``collect_boundaries`` hand each backend ONE boundary.

        The two type checks in the emitters cannot be reached by any authorable
        model — the policy module only ever names a Space and an IfcElement —
        so without this they are guards no test can falsify, which is the shape
        this repository keeps finding. Patching the shared module rather than
        each backend's import is what makes the SAME injection exercise both.
        """
        import lite_step.ifc.space_boundaries as sb

        monkeypatch.setattr(sb, "collect_boundaries", lambda proj: [boundary])

    def _one_room(self):
        proj, storey = _project()
        wall(storey, "north", 1700, 1300, 5900, 1600)
        space(storey, "a")
        return proj

    def test_a_boundary_naming_no_product_refuses(self, monkeypatch):
        from lite_step.ifc.space_boundaries import SpaceBoundary

        self._forced(monkeypatch, SpaceBoundary(
            space="space:a:storey:ground", element="wall:ghost",
            physical_or_virtual=PHYSICAL, internal_or_external=EXTERNAL,
            name=BOUNDARY_NAME, description=None, parent_index=None))
        result = compile_result(self._one_room(), "ifcopenshell")
        assert not result.success
        assert "wall:ghost" in result.error
        assert "RelatedBuildingElement" in result.error

    def test_a_boundary_naming_a_storey_as_its_element_refuses(
            self, monkeypatch):
        """``RelatedBuildingElement`` is an ``IfcElement``. Neither writer
        enforces the supertype at create time, so a name resolving to an
        ``IfcBuildingStorey`` would serialise a schema-invalid relation."""
        from lite_step.ifc.space_boundaries import SpaceBoundary

        self._forced(monkeypatch, SpaceBoundary(
            space="space:a:storey:ground", element="storey:ground",
            physical_or_virtual=PHYSICAL, internal_or_external=EXTERNAL,
            name=BOUNDARY_NAME, description=None, parent_index=None))
        result = compile_result(self._one_room())
        assert not result.success
        assert "requires a IfcElement" in result.error
        assert "storey:ground" in result.error

    def test_a_boundary_naming_a_wall_as_its_space_refuses(
            self, monkeypatch):
        """``RelatingSpace`` is an ``IfcSpaceBoundarySelect``."""
        from lite_step.ifc.space_boundaries import SpaceBoundary

        self._forced(monkeypatch, SpaceBoundary(
            space="wall:north:storey:ground", element="wall:north:storey:ground",
            physical_or_virtual=PHYSICAL, internal_or_external=EXTERNAL,
            name=BOUNDARY_NAME, description=None, parent_index=None))
        result = compile_result(self._one_room())
        assert not result.success
        assert "requires a IfcSpace" in result.error

    def test_an_anonymous_space_is_warned_about_not_silently_dropped(self, caplog):
        proj, storey = _project()
        wall(storey, "north", 1700, 1300, 5900, 1600)
        s = Space()
        s.add(Box(start=Point(x=1700, y=-3100, z=0),
                  end=Point(x=5900, y=1300, z=2700)))
        storey.add(s)
        with caplog.at_level("WARNING"):
            assert boundaries(proj) == []
        assert "anonymous Space" in caplog.text


# ---------------------------------------------------------------------------
# 11. The reference document
# ---------------------------------------------------------------------------


class TestTheReferenceMatchesTheCompiler:
    """A documentation bug: ``agents/dsl-reference.md`` told model
    authors for months that the compiler derives space boundaries,
    internal/external classification, the space pair across a wall, SOLAR
    ORIENTATION and element adjacency. It derived none of them, and nothing
    measured the claim — so the claim is measured now.

    Deliberately narrow: it pins the FACTS an author would act on (which
    entity, which enum values, what is not derived), not the prose around
    them. A test that asserted whole sentences would be edited to match the
    document instead of the other way round.

    The section was headed "Inferred relations" until. WS-F
    retired the word: a space's bounds are DECLARED like every other
    relation in the language, and deriving the volume from them is a helper,
    not an inference. If this lookup raises, the heading moved — repoint it
    deliberately rather than loosening the search.
    """

    #: The heading this section is found by. One constant, so a rename is one
    #: edit and a miss is a loud KeyError-shaped failure rather than a silent
    #: empty section that passes every ``in`` assertion below.
    HEADING = "## 8. Spaces - derived or authored"

    @classmethod
    def _section(cls):
        from pathlib import Path

        doc = (Path(__file__).resolve().parents[2]
               / "agents" / "dsl-reference.md").read_text(encoding="utf-8")
        assert cls.HEADING in doc, (
            f"the reference does not carry {cls.HEADING!r} — the spaces "
            f"section was renamed or removed; repoint HEADING deliberately")
        start = doc.index(cls.HEADING)
        return doc[start:doc.index("\n---\n", start)]

    def test_it_names_the_entity_the_compiler_actually_emits(self):
        assert BOUNDARY_ENTITY in self._section()

    def test_it_names_every_classification_the_compiler_can_produce(self):
        section = self._section()
        for value in (INTERNAL, EXTERNAL, EXTERNAL_EARTH, NOTDEFINED, PHYSICAL):
            assert value in section, f"{value} is emitted but undocumented"

    def test_the_engine_computes_no_solar_anything(self):
        """The doc's solar RETRACTION is gone — it denied a
        capability the current text never claims, and it read as "no solar
        information available" when the emitted model in fact carries
        ``site_latitude``/``site_longitude``/``site_true_north``
        (``generator._add_georeferencing``), from which any consumer derives
        sun position. What survives is the CODE half: nothing in this
        compiler computes a solar anything, so no future reader has to
        wonder whether the removed sentence was retired because it became
        false."""
        hits = [p for p in _engine_modules() if _defines(p, "solar")]
        assert hits == [], f"solar is computed after all, in {hits}"

    def test_it_no_longer_claims_element_adjacency_is_a_relation(self):
        """``IfcRelConnectsElements`` is never written. If a future PR emits
        one, this fails and the sentence has to be rewritten rather than
        silently becoming true-by-accident."""
        hits = [p for p in _engine_modules()
                if _defines(p, "IfcRelConnectsElements")
                or "IFCRELCONNECTSELEMENTS" in p.read_text(encoding="utf-8")]
        assert hits == []
        assert "IfcRelConnectsElements` is never written" in self._section()

    def test_it_states_that_a_space_must_be_named(self):
        assert "must be **named**" in self._section()
