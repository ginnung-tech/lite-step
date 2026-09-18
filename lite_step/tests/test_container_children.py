"""Container child validation at ``.add()`` (DSL-layer, construction time).

The table is not a TYPE table. Every physical container takes every
geometry primitive AND every product — a wall holds a column, a roof holds a
beam — exactly as ``.anchor()`` does. What survives is three SEMANTIC
refusals, and this file is where each is pinned by the fix it names:

    IfcSpace + a product   -> IfcRelSpaceBoundary (the relation already exists)
    any container + an opening -> .anchor() / .opening() (no coordinates)
    a spatial node in the wrong place -> ifc/facilities.py::validate_nesting

``IfcSite`` is not a fourth entry. The generator's project-level site pass
takes the generic fallback, so the row is the ordinary one plus
``SpatialElement`` and there is no refusal. What a ``Site`` emits is measured
in ``test_site_children.py``, which is the half that makes the widening honest.

Refused children raise ``ValueError`` at the ``.add()`` call site — validity
is defined at the DSL layer; the generator translates, it does not validate.
Also covered here:

* the Wall >1-Solid compile ERROR in ``validate_project_report`` (the
  generator uses only the FIRST Solid as the wall body);
* the Space child appearance-precedence chain (spec §6): explicit
  ``color=`` > registry material render > transparent-blue auto-color.

And, at the bottom, the OTHER child verbs on the same containers:

* ``.opening()`` either routes to ``.anchor()`` or REFUSES — it never parks a
  Window in ``_openings``, the leaf list no container pass reads. That section
  also owns the taxonomy table behind both verbs: "is a container" and "has a
  local frame" are two different questions, and ``.opening()`` must not
  conflate them.
* ``.anchor()`` refuses a ``SpatialElement`` and NOTHING else — the third
  refusal above, applied at the third verb. ``.anchor()`` has no type table
  and must not grow one; what it has is the spatial/physical BOUNDARY, one
  question with two answers. The regression guard that says so is the anchor
  matrix, re-walked here against the refusal.
"""

from __future__ import annotations

import pytest

from lite_step.models import (
    Bar,
    Beam,
    Project,
    Column,
    Door,
    Element,
    Material,
    Mesh,
    Point,
    ReferencePoint,
    Point2D,
    Sweep,
    Pipe,
    Roof,
    Site,
    Slab,
    Box,
    Extrude,
    Space,
    SpatialElement,
    Revolve,
    Wall,
    Window,
)
from lite_step.compiler.executor import validate_project_report
#: The anchor matrix's own table, imported rather than re-derived: the
#: bottom section asserts ``.anchor()`` still has NO type table, and a second
#: hand-written list of pairs could quietly be narrower than the one the
#: emitters are actually held to.
from lite_step.tests.test_anchor_matrix import (
    CELLS as _MATRIX_CELLS,
    CHILDREN as _MATRIX_CHILDREN,
    CONTAINERS as _MATRIX_CONTAINERS,
    REFUSED as _MATRIX_REFUSED,
)


# ---------------------------------------------------------------------------
# Child factories (valid, minimal, int-mm)
# ---------------------------------------------------------------------------


def _solid(**kw):
    return Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=200, z=2000), **kw)


def _sheet(**kw):
    return Extrude(contour=[Point(x=0, y=0, z=0), Point(x=1200, y=0, z=0),
                          Point(x=1200, y=0, z=1400), Point(x=0, y=0, z=1400)],
                 thickness=6, **kw)


def _profile(**kw):
    return Sweep(profile=[Point2D(x=-50, y=-50), Point2D(x=50, y=-50),
                            Point2D(x=50, y=50), Point2D(x=-50, y=50)],
                   path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=3000)], **kw)


def _turn(**kw):
    return Revolve(profile=[Point2D(x=0, y=0), Point2D(x=500, y=0),
                         Point2D(x=500, y=360), Point2D(x=0, y=360)],
                path=[Point(x=0, y=0, z=0), Point(x=0, y=360, z=0)],
                angle=18000, **kw)


def _rod(**kw):
    return Pipe(path=[Point(x=0, y=0, z=0), Point(x=2000, y=0, z=0)], radius=25, **kw)


def _bar(**kw):
    return Bar(path=[Point(x=0, y=0, z=0), Point(x=2000, y=0, z=0)],
               diameter=8, **kw)


def _mesh(**kw):
    return Mesh(vertices=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0),
                          Point(x=0, y=1000, z=0)],
                faces=[(0, 1, 2)], **kw)


def _window(**kw):
    return Window(width=1200, height=1400, **kw)


def _door(**kw):
    return Door(width=900, height=2100, **kw)


# ---------------------------------------------------------------------------
# Off-table children raise at .add()
# ---------------------------------------------------------------------------


class TestOffTableChildrenRaise:
    """What ``.add()`` still refuses, now that "a container inside a container"
    is not on the list.

    Neither geometry nor products are refused by a physical container any
    more — a product's shape is not a function of its class, and elements
    contain elements. Each surviving refusal is a SEMANTIC statement, and each
    is asserted here by the fix it names.
    """

    def test_wall_into_site_is_no_longer_refused(self):
        """The one refusal here that was NOT semantic, and is now gone.

        It existed only because the generator's site pass had no generic
        fallback. Asserting the acceptance here — not just deleting
        a refusal — is what keeps the two halves paired: if the site pass
        ever loses its fallback again, ``test_site_children.py`` goes red while
        this line stays green, and the pair says which half broke.
        """
        assert Site(name="s1").add(Wall(name="w1")) is not None

    def test_wall_into_space(self):
        """A Space is AIR. The relation an author means here already exists."""
        with pytest.raises(ValueError, match="IfcRelSpaceBoundary"):
            Space(name="kitchen").add(Wall(name="w1"))

    def test_profile_into_space(self):
        """``IfcSpace``'s row is the volume it occupies — a Box, nothing else."""
        with pytest.raises(ValueError, match="IfcRelSpaceBoundary"):
            Space(name="kitchen").add(_profile())

    def test_window_into_anything(self):
        """An opening carries no coordinates, so world-coordinate containment
        has nothing to place — refused at the authoring line, for every
        container, naming both host verbs."""
        for host in (Wall(name="w1"), Roof(name="r1"), Slab(name="s1"),
                     Element(ifc_class="IfcStair", name="stair")):
            with pytest.raises(ValueError, match="cannot be ADDED") as exc:
                host.add(_window(name="w0"))
            assert ".anchor(window" in str(exc.value)
            assert ".opening(window" in str(exc.value)

    def test_door_into_a_wall(self):
        with pytest.raises(ValueError, match="cannot be ADDED"):
            Wall(name="w1").add(_door(name="d0"))

    def test_message_names_the_ifc_class_and_the_allowed_set(self):
        """A ``ReferencePoint`` is the shape that still reaches this message on
        a ``Site``: not a product at all, so it is on no container's row."""
        with pytest.raises(ValueError) as exc:
            Site(name="s1").add(ReferencePoint(location=Point(x=0, y=0, z=0)))
        msg = str(exc.value)
        assert "IfcSite" in msg
        assert "Box, Extrude, Sweep, Revolve, Pipe, Bar, Mesh" in msg
        assert "Wall, Column, Beam, Slab, Roof, Space" in msg
        assert "SpatialElement" in msg

    def test_message_advice_actually_works(self):
        """Advice naming ``Element(ifc_class=...)`` would be wrong:
        ``Element``'s own table refuses ``Element`` children too, so an author
        who followed it would hit a second error. Every refusal below names a
        fix, and every named fix is exercised right here rather than merely
        quoted.
        """
        # Space -> the boundary relation, which is derived from the geometry
        # of a wall standing as a PEER of the space.
        with pytest.raises(ValueError) as exc:
            Space(name="kitchen").add(Wall(name="w1"))
        assert "peer" in str(exc.value)
        proj = Project(name="p")
        space = Space(name="kitchen").add(_solid())
        wall = Wall(name="w1").add(_solid())
        proj.add(space, wall)
        assert validate_project_report(proj).errors == []

        # Site -> the Storey, which really does take a building's contents.
        # NOT .anchor(): a Site has no such method, and the message must not
        # offer it. A Site takes the product itself, so no
        # Element(ifc_class=...) wrapper is needed either.
        with pytest.raises(ValueError) as exc:
            Site(name="s1").add(ReferencePoint(location=Point(x=0, y=0, z=0)))
        msg = str(exc.value)
        assert "Storey" in msg, msg
        assert ".anchor()" not in msg.replace("has no .anchor()", ""), msg
        assert Site(name="s1").add(Wall(name="w1").add(_solid())) is not None

    def test_message_names_offending_child(self):
        with pytest.raises(ValueError,
                           match=r"invalid child ReferencePoint 'datum'"):
            Site(name="s1").add(
                ReferencePoint(name="datum", location=Point(x=0, y=0, z=0)))

    def test_failed_add_is_atomic(self):
        """A rejected varargs .add() appends NOTHING — not even the valid
        children ahead of the invalid one."""
        roof = Roof(name="r1")
        with pytest.raises(ValueError):
            roof.add(_solid(), _window(name="w0"))
        assert roof.elements == []


class TestTableIsKeyedOnTheIfcClass:
    """§1.2 option (c): the sugar and the escape hatch behave IDENTICALLY,
    because they resolve to the same key.

    An escape hatch MORE permissive than the sugar for the same IFC class
    would reward abandoning the semantic container.
    """

    @pytest.mark.parametrize("sugar,ifc_class", [
        (lambda: Wall(name="x"), "IfcWall"),
        (lambda: Roof(name="x"), "IfcRoof"),
        (lambda: Slab(name="x"), "IfcSlab"),
        (lambda: Column(name="x"), "IfcColumn"),
        (lambda: Beam(name="x"), "IfcBeam"),
    ])
    def test_sugar_and_escape_hatch_accept_the_same_children(
            self, sugar, ifc_class):
        from lite_step.models.elements import allowed_children_for
        assert (allowed_children_for(sugar())
                == allowed_children_for(Element(ifc_class=ifc_class, name="x")))

    def test_a_wall_may_now_be_curved(self):
        """The §1.2 damage, undone. ``Wall().add(sweep)`` was a compile error
        while ``Element(ifc_class="IfcWall").add(sweep)`` was legal, so there
        was no curved wall except by abandoning the semantic container.

        Falsification: put the old ``("Box", "Extrude", "Window", "Door")``
        table back on Wall and this raises.
        """
        wall = Wall(name="curved").add(_profile())
        assert len(wall._elements) == 1

    def test_the_escape_hatch_is_no_longer_more_permissive(self):
        """The direction that mattered: for every product class, anything the
        escape hatch takes, the sugar takes too."""
        from lite_step.models.elements import allowed_children_for
        for sugar, ifc_class in ((Wall(name="x"), "IfcWall"),
                                 (Roof(name="x"), "IfcRoof"),
                                 (Slab(name="x"), "IfcSlab"),
                                 (Column(name="x"), "IfcColumn"),
                                 (Beam(name="x"), "IfcBeam")):
            hatch = set(allowed_children_for(
                Element(ifc_class=ifc_class, name="x")))
            assert hatch <= set(allowed_children_for(sugar)), ifc_class


# ---------------------------------------------------------------------------
# The full allowed set still adds fine, per container
# ---------------------------------------------------------------------------


class TestAllowedChildrenAdd:
    def test_wall_full_set(self):
        """Openings are NOT in it — they are anchored, never added."""
        wall = Wall(name="w1")
        wall.add(_solid())
        wall.anchor(_window(name="w0"), along=500, up=900)
        wall.anchor(_door(name="d0"), along=2000, up=0)
        assert len(wall._elements) == 3

    def test_roof_full_set(self):
        roof = Roof(name="r1").add(_solid(), _profile())
        assert len(roof.elements) == 2

    def test_column_full_set(self):
        col = Column(name="c1").add(_solid(), _profile(), _turn())
        assert len(col.elements) == 3

    def test_beam_full_set(self):
        beam = Beam(name="b1").add(_solid(), _profile(), _bar())
        assert len(beam.elements) == 3

    def test_space_full_set(self):
        space = Space(name="k1").add(_solid())
        assert len(space.elements) == 1

    def test_site_full_set(self):
        site = Site(name="s1").add(_solid(), _mesh())
        assert len(site._elements) == 2

    def test_window_full_set(self):
        win = _window(name="win1").add(_sheet(color="glass"), _profile())
        assert len(win._elements) == 2

    def test_door_full_set(self):
        door = _door(name="d1").add(_sheet(), _profile())
        assert len(door._elements) == 2

    def test_element_full_set(self):
        el = Element(ifc_class="IfcStair", name="stair").add(
            _solid(), _profile(), _turn(), _rod(), _bar(), _mesh())
        assert len(el.elements) == 6

# ---------------------------------------------------------------------------
# Wall >1 Solid: compile-time ERROR (generator uses only the first Solid)
# ---------------------------------------------------------------------------


class TestWallSingleBodySolid:
    def test_two_solids_in_wall_is_error(self):
        wall = Wall(name="w1").add(_solid(name="body_a"),
                                   _solid(name="body_b"))
        b = Project(name="two-bodies")
        b.add(wall)
        report = validate_project_report(b)
        assert any("contains 2 Box/Extrude children" in e for e in report.errors)

    def test_one_solid_plus_openings_is_clean(self):
        wall = Wall(name="w1").add(_solid())
        wall.anchor(_window(name="w0"), along=500, up=900)
        wall.anchor(_door(name="d0"), along=2000, up=0)
        b = Project(name="one-body")
        b.add(wall)
        report = validate_project_report(b)
        assert not any("Box/Extrude children" in e for e in report.errors)

    def test_zero_solids_still_reports_missing_body(self):
        """Every child is ``.anchor()``ed, so the frame BASIS is empty. A body-less Wall is legal when it has ``.add()``ed children to
        derive a frame from; an anchored child is placed BY that frame and can
        never be what derives it, so this wall still has nothing — and the
        refusal now says which of the two it is.
        """
        wall = Wall(name="w1")
        wall.anchor(_window(name="w0"), along=500, up=900)
        b = Project(name="no-body")
        b.add(wall)
        report = validate_project_report(b)
        assert any("no frame to place its 1 .anchor()ed child(ren) against" in e
                   for e in report.errors), report.errors


# ---------------------------------------------------------------------------
# Space child appearance precedence (spec §6) — ifcopenshell backend
# ---------------------------------------------------------------------------

SPACE_BLUE = "0.678"  # 173/255 — first component of the auto-color RGB


def _generate(project: Project) -> str:
    # backend="ifcopenshell" pins these assertions to the backend that owns
    # the transparent-blue Space auto-color. NOTE (pre-existing divergence,
    # not introduced here): the default streaming backend renders Space
    # children with its plain default style and applies no transparent blue.
    pytest.importorskip("ifcopenshell")
    from lite_step.compiler.executor import normalize_project_to_meters
    from lite_step.ifc.generator import generate_ifc

    normalized = normalize_project_to_meters(project)
    result = generate_ifc(normalized)
    assert result.success, result.error
    return result.ifc_content


class TestSpacePrecedence:
    def test_default_space_stays_transparent_blue(self):
        """No color, no material -> the historical 70%-transparent light
        blue auto-color, unchanged."""
        b = Project(name="space-default")
        b.add(Space(name="kitchen", space_type="residential").add(_solid()))
        content = _generate(b)
        assert "IFCSPACE" in content
        assert SPACE_BLUE in content
        assert ",0.7," in content

    def test_explicit_color_wins_over_auto_blue(self):
        b = Project(name="space-color")
        b.add(Space(name="kitchen").add(_solid(color="glass")))
        content = _generate(b)
        # glass palette: 60% transparency, and NO auto-blue style
        assert ",0.6," in content
        assert SPACE_BLUE not in content

    def test_registry_material_render_wins_over_auto_blue(self):
        b = Project(name="space-material")
        b.add(Space(name="kitchen").add(_solid(material="Brick_Red_DK")))
        content = _generate(b)
        assert "IFCMATERIAL('Brick_Red_DK'" in content
        assert SPACE_BLUE not in content

    def test_second_child_follows_the_same_chain(self):
        """The aggregated (non-first) children gate the auto-color the
        same way as the representation-bearing first child."""
        b = Project(name="space-two")
        first = _solid(name="vol_a")
        second = Box(start=Point(x=2000, y=0, z=0),
                       end=Point(x=3000, y=200, z=2000),
                       name="vol_b", color="glass")
        b.add(Space(name="kitchen").add(first, second))
        content = _generate(b)
        # first child: auto blue present; second child: glass 0.6 present
        assert SPACE_BLUE in content
        assert ",0.6," in content


class TestTheSymmetryReachesTheEmitter:
    """§1.2 is only real if the EMITTER agrees with the table.

    Found by manual QA on the child-table change: `Wall` accepted a Sweep body
    at construction, then `_create_wall` found no prism, logged a warning and
    `return None`d — the whole wall (identity, Psets, geometry) vanished from
    the file, while `Element(ifc_class="IfcWall")` with the identical child
    emitted an IfcWall. That would have moved the asymmetry one layer down
    instead of removing it.
    """

    _CURVED = """
from lite_step.models import Project, Wall, Element, Sweep, Point, Material
def generate_project():
    proj = Project(name="t")
    host = %s
    host.add(Sweep(name="body",
                   material=Material(key="Timber_C24", profile_mm=(45, 195)),
                   path=[Point(x=0, y=0, z=0), Point(x=4000, y=0, z=0)]))
    proj.add(host)
    return proj
result = generate_project()
"""

    def _project(self, spelling):
        from lite_step.compiler.executor import execute_lite_step_script

        r = execute_lite_step_script(self._CURVED % spelling)
        assert r.success, r.error
        return r.project

    def _emit(self, spelling):
        import ifcopenshell
        from lite_step.ifc.generator import generate_ifc

        ifc = generate_ifc(self._project(spelling))
        assert ifc.success, ifc.error
        return ifcopenshell.file.from_string(ifc.ifc_content)

    def test_a_curved_wall_emits_an_ifcwall(self):
        pytest.importorskip("ifcopenshell")
        m = self._emit('Wall(name="curved")')
        walls = m.by_type("IfcWall")
        assert len(walls) == 1, "the wall must not vanish"
        assert walls[0].Name == "wall:curved"
        # …and its geometry rides an aggregated member, not nothing.
        geometry = [p for p in m.by_type("IfcProduct")
                    if p.Representation and p.is_a() != "IfcAnnotation"]
        assert geometry, "the swept body must emit somewhere"

    def test_sugar_and_escape_hatch_emit_the_same_shape(self):
        pytest.importorskip("ifcopenshell")
        sugar = self._emit('Wall(name="curved")')
        hatch = self._emit('Element(ifc_class="IfcWall", name="curved")')

        def shape(m):
            return sorted((p.is_a(), bool(p.Representation))
                          for p in m.by_type("IfcProduct")
                          if p.is_a() not in ("IfcAnnotation", "IfcSite",
                                              "IfcBuilding", "IfcBuildingStorey"))
        assert shape(sugar) == shape(hatch)


# ---------------------------------------------------------------------------
# .opening() on a container: ROUTED, or REFUSED — never parked
#
# ``BimElement.opening`` routed on ``tx.is_container(self) and hasattr(self,
# "anchor")``. ``Site`` and ``SpatialElement`` satisfy the first and not the
# second, so they fell through to the LEAF path and appended the Window to
# ``self._openings`` — a list nothing reads for a spatial node. Without this
# section: no error, ``_openings`` length 1, ``_elements``
# untouched, and a compiled file with zero ``IfcOpeningElement``, zero
# ``IfcRelVoidsElement`` and zero ``IfcWindow``.
#
# That is the failure ``.opening()``'s own routing comment describes — "the
# doorway that silently is not there" — reintroduced through the gap in the
# condition meant to prevent it. The gate is now a positive assertion, and a
# container with no local frame RAISES.
# ---------------------------------------------------------------------------


def _bim_element_subclasses():
    """Every ``BimElement`` subclass the models module defines, BY REFLECTION.

    Deliberately not ``taxonomy._ALL_ELEMENT_TYPES``: the table below exists
    to check the taxonomy, so it must not inherit the taxonomy's own idea of
    what the classes are. A class the taxonomy forgot is exactly the bug.
    """
    from lite_step.models import elements as _mod
    from lite_step.models.elements import BimElement

    return sorted(
        (obj for obj in vars(_mod).values()
         if isinstance(obj, type)
         and issubclass(obj, BimElement)
         and obj is not BimElement),
        key=lambda t: t.__name__,
    )


#: One instance per CONTAINER class, for the behavioural half of the table.
#: ``test_every_container_is_covered_by_the_factories`` asserts these keys ARE
#: ``tx.CONTAINERS``, so a new container class cannot join the taxonomy and
#: quietly skip every behavioural check in this section.
_CONTAINER_FACTORIES = {
    "Wall": lambda: Wall(name="host"),
    "Column": lambda: Column(name="host"),
    "Beam": lambda: Beam(name="host"),
    "Slab": lambda: Slab(name="host"),
    "Roof": lambda: Roof(name="host"),
    "Element": lambda: Element(ifc_class="IfcBuildingElementProxy", name="host"),
    "Space": lambda: Space(name="host"),
    "Site": lambda: Site(name="plot"),
    "SpatialElement": lambda: SpatialElement(ifc_class="IfcBridge",
                                             name="storstroem"),
}


def _park_in_openings(host, child, *, along=0, up=0):
    """Do by hand what the buggy routing did: stamp the anchor and append to
    ``host._openings``. This is the state the fix now refuses to construct,
    and the evidence test below compiles it to show what it was worth."""
    from lite_step.models.elements import ChildAnchor

    child._anchor_spec = ChildAnchor(along=along, along_center=None,
                                     inset=0, up=up, rotations=[])
    child._parent = host
    host._openings.append(child)
    return host


def _terrain(**kw):
    return Mesh(vertices=[Point(x=-5000, y=-5000, z=0),
                          Point(x=5000, y=-5000, z=0),
                          Point(x=5000, y=5000, z=0),
                          Point(x=-5000, y=5000, z=0)],
                faces=[(0, 1, 2), (0, 2, 3)],
                is_watertight=True, source_type="synthetic", **kw)


def _compile_ifc(proj):
    """Compile and hand back the STEP text, asserting a clean pass on the way.

    The assertions are the point: the silent-park failure produced a file that
    validated and generated without a single complaint."""
    from lite_step.compiler.executor import normalize_project_to_meters
    from lite_step.ifc.generator import generate_ifc

    assert validate_project_report(proj).errors == []
    result = generate_ifc(normalize_project_to_meters(proj) or proj)
    assert result.success, result.error
    return result.ifc_content


class TestOpeningRefusesWhereItWouldPark:

    def test_a_site_refuses_an_opening_and_names_add(self):
        """``Site.opening(Window(...))`` accepted the call and did nothing.

        A ``Site`` is a PLACE. ``along=`` / ``inset=`` / ``up=`` measure from
        an outer face along a run axis and terrain has neither, so the fix is
        NOT to give ``Site`` an ``.anchor()`` — that would invent a frame for
        something that has none. It is to refuse, and to name the verb the
        author actually wants: ``.add()``, world-coordinate containment.
        """
        site = Site(name="plot").add(_terrain(name="ground"))
        with pytest.raises(TypeError) as exc:
            site.opening(_window(name="g0"), along=1000, up=900)
        msg = str(exc.value)
        assert ".add()" in msg, msg
        assert "Site" in msg, msg
        assert "local frame" in msg, msg
        # …and nothing was parked on the way out.
        assert site._openings == []

    def test_a_spatial_element_refuses_an_opening_and_names_add(self):
        """Same shape, same silence: a facility node is ``_elements``-bearing
        and has no body, so it has no face to measure ``along=`` from
        either."""
        part = SpatialElement(ifc_class="IfcBridgePart", name="deck",
                              usage="LONGITUDINAL")
        with pytest.raises(TypeError) as exc:
            part.opening(_door(name="d0"), along=500)
        msg = str(exc.value)
        assert ".add()" in msg, msg
        assert "SpatialElement" in msg, msg
        assert part._openings == []

    def test_a_window_parked_on_a_site_is_now_refused_by_the_validator(self):
        """WHY a refusal rather than a warning — and the SECOND net that
        catches the same park.

        A park reconstructed by hand would compile clean with zero ``IfcWindow``,
        zero ``IfcOpeningElement`` and zero ``IfcRelVoidsElement``: no error
        anywhere, and no window — indistinguishable downstream from an author
        who wrote no window at all. That is the whole argument for the
        ``.opening()`` raise above, and it makes the hand-built state
        unreachable too. ``.opening()``'s
        raise guards the AUTHORING line; ``executor._opening_host_errors``
        guards the TREE, so a Window that arrives on a host no emitter reads
        — by ``.anchor()``, by a hand-stamped list, by any route — is a
        compile error rather than a file with a window missing from it. The
        old expectation is kept below as the measurement it was: those three
        counts are what compiling this project produced on ``main`` @ 24304c8.
        """
        pytest.importorskip("ifcopenshell")
        site = Site(name="plot").add(_terrain(name="ground"))
        _park_in_openings(site, _window(name="g0"), along=1000, up=900)
        assert len(site._openings) == 1

        proj = Project(name="parked-opening")
        proj.add(site)

        errors = validate_project_report(proj).errors
        assert len(errors) == 1, errors
        # It names the host, the child, and what happens instead.
        assert "no IFC emitter reads" in errors[0]
        assert "Window 'g0'" in errors[0]
        assert errors[0].startswith("Site '")

        # And the site itself is still a legal model without the park — so the
        # error above is about the Window and not about the fixture.
        clean = Project(name="parked-opening")
        clean.add(Site(name="plot").add(_terrain(name="ground")))
        ifc = _compile_ifc(clean).upper()
        assert "IFCSITE(" in ifc
        assert "IFCWINDOW(" not in ifc

    def test_every_container_is_covered_by_the_factories(self):
        """The table below is only exhaustive if this passes. A new container
        class joins ``tx.CONTAINERS`` by declaring ``_elements``; without this
        it would join and quietly skip every behavioural check here."""
        from lite_step.models import taxonomy as tx

        assert set(_CONTAINER_FACTORIES) == {t.__name__ for t in tx.CONTAINERS}

    @pytest.mark.parametrize("class_name", sorted(_CONTAINER_FACTORIES))
    def test_a_container_opening_routes_or_raises_but_never_parks(
            self, class_name):
        """The behavioural clincher. For EVERY container, ``.opening()`` either
        reaches ``.anchor()`` (the child lands in ``_elements``) or refuses.
        ``_openings`` — the leaf list, which no container pass reads — is empty
        afterwards either way.

        This is the assertion that would have caught the original bug without
        knowing in advance which classes were affected.

        TWO refusals, not one, since, and they are different
        questions: ``TypeError`` for a container with no local frame at all
        (nothing for ``along=`` to measure), ``ValueError`` for a ``Space``,
        which HAS a frame but is AIR — an opening cuts a body and there is no
        body here. Neither parks.
        """
        from lite_step.models import taxonomy as tx

        host = _CONTAINER_FACTORIES[class_name]()
        try:
            host.opening(_window(name="w0"), along=1000, up=900)
        except TypeError:
            assert not tx.has_local_frame(host), (
                f"{class_name} has a local frame and must have routed")
            assert host._openings == []
            return
        except ValueError:
            assert tx.is_spatial(host), (
                f"{class_name} is not AIR — only a spatial container refuses "
                f"an opening with a ValueError")
            assert host._openings == []
            return
        assert tx.has_local_frame(host), (
            f"{class_name} has no local frame and must have refused")
        assert not tx.is_spatial(host), (
            f"{class_name} is AIR and must have refused")
        assert host._openings == [], (
            f"{class_name}.opening() parked the child in _openings — the list "
            f"no container pass reads")
        assert len(host._elements) == 1

    def test_a_leaf_solid_still_parks_in_openings(self):
        """The control. ``_openings`` is not dead — it is the LEAF path, and
        both IFC backends read it for a standalone solid. A fix that emptied
        it everywhere would break the spelling ``.opening()``'s own docstring
        advertises."""
        from lite_step.models import taxonomy as tx

        body = _solid(name="south")
        assert not tx.is_container(body)
        body.opening(_window(name="w0"), along=1000, up=900)
        assert len(body._openings) == 1


class TestTheThreeDefinitionsOfContainerAgree:
    """Three definitions of "container" were live at once and disagreed
    pairwise — that mismatch IS the bug above.

    ``tx.is_container`` (holds no points of its own), "declares ``_elements``"
    (``Window``/``Door`` do, and are not containers) and "has ``.anchor()``"
    (``Site``/``SpatialElement`` do not). ``.opening()`` routed on the first
    AND-ed with an inline ``hasattr(self, "anchor")`` and fell through wherever
    the two disagreed.

    They are TWO now, each named: ``is_container`` and ``has_local_frame``.
    This table pins the relationship between them across every ``BimElement``
    subclass, by reflection, so a new class joins it automatically.
    """

    @pytest.mark.parametrize(
        "cls", _bim_element_subclasses(), ids=lambda c: c.__name__)
    def test_the_named_predicates_match_the_structure_they_describe(self, cls):
        from lite_step.models import taxonomy as tx

        declares_elements = "_elements" in cls.__private_attributes__
        has_anchor = hasattr(cls, "anchor")

        # 1. is_container == "declares _elements", minus the documented
        #    opening carve-out (a Window/Door holds opening-local children and
        #    is a special case in every pass, never a container).
        assert tx.is_container(cls) == (declares_elements
                                        and not cls.brings_void), cls.__name__

        # 2. has_local_frame == "has .anchor()". One fact, one spelling — the
        #    inline hasattr in .opening() is gone.
        assert tx.has_local_frame(cls) == has_anchor, cls.__name__

        # 3. …and it is strictly NARROWER. Nothing may carry a local frame
        #    without being a container; that direction is what makes the
        #    .opening() gate safe to write as a positive assertion.
        if has_anchor:
            assert tx.is_container(cls), cls.__name__

        # 4. Every predicate answers identically for a class and its NAME —
        #    the routing-table convention, and the string form is what the
        #    compiler passes reach for.
        assert tx.is_container(cls.__name__) == tx.is_container(cls)
        assert tx.has_local_frame(cls.__name__) == tx.has_local_frame(cls)

    def test_the_disagreement_set_is_exactly_the_refusing_containers(self):
        """Named, not implied: ``Site`` and ``SpatialElement`` are the two
        classes where "is a container" and "has a local frame" part company,
        and therefore the complete list of containers ``.opening()`` refuses.
        If this set ever changes, the refusal message — which talks about
        places and facility nodes — needs rewriting with it."""
        from lite_step.models import taxonomy as tx

        refusing = ({t.__name__ for t in tx.CONTAINERS}
                    - {t.__name__ for t in tx.LOCAL_FRAME})
        assert refusing == {"Site", "SpatialElement"}

    def test_containers_is_derived_from_the_rule_it_states(self):
        """``CONTAINERS`` was a hand-written tuple beside a docstring stating
        the rule — the shape, and ``SpatialElement`` is what it cost
        (absent for the one node type whose entire job is holding things).

        It is derived now. This test does the derivation independently, over a
        reflected class list, so the two cannot drift together."""
        from lite_step.models import taxonomy as tx

        expected = {c.__name__ for c in _bim_element_subclasses()
                    if "_elements" in c.__private_attributes__
                    and not c.brings_void}
        assert {t.__name__ for t in tx.CONTAINERS} == expected
        # The carve-out the docstring insists on, asserted rather than assumed.
        assert "Window" not in expected and "Door" not in expected


class TestAContainerWithAFrameStillEmitsItsWindow:
    """The regression guard on the routing being tightened. Everything above is
    about refusing; this is the half that must keep WORKING."""

    def test_wall_opening_routes_to_anchor_and_emits_an_ifcwindow(self):
        pytest.importorskip("ifcopenshell")
        wall = Wall(name="south").add(
            Box(start=Point(x=0, y=-300, z=0),
                end=Point(x=6000, y=0, z=2700), type="wall", name="body"))
        wall.opening(_window(name="w0"), along=1500, up=900)

        # Routed, not parked: the child is in _elements with an anchor spec.
        assert wall._openings == []
        win = [c for c in wall._elements if isinstance(c, Window)]
        assert len(win) == 1
        assert win[0]._anchor_spec is not None
        assert win[0]._anchor_spec.along == 1500

        proj = Project(name="routed-opening")
        proj.add(wall)
        ifc = _compile_ifc(proj).upper()
        assert ifc.count("IFCWINDOW(") == 1, "the window must emit"
        assert "IFCOPENINGELEMENT(" in ifc
        assert "IFCRELVOIDSELEMENT(" in ifc

    @pytest.mark.parametrize(
        "class_name", ["Column", "Beam", "Slab", "Roof", "Element"])
    def test_every_framed_container_routes_to_anchor(self, class_name):
        """Not just Wall. Each of these has an ``.anchor()``, so each must take
        the routed path — the gate is a positive assertion now, and a class
        missing from ``LOCAL_FRAME`` would refuse instead of route."""
        host = _CONTAINER_FACTORIES[class_name]()
        host.add(Box(start=Point(x=0, y=-300, z=0),
                     end=Point(x=6000, y=0, z=2700), name="body"))
        host.opening(_window(name="w0"), along=1500, up=900)
        assert host._openings == []
        assert any(isinstance(c, Window) for c in host._elements)


# ---------------------------------------------------------------------------
# .anchor() and the spatial/physical boundary
#
# ``.anchor()`` has no type table, deliberately, and must not grow one: every
# PHYSICAL child goes into every container and the emitters render the whole
# product (``test_anchor_matrix.py``). What it lacked was the OTHER question —
# the one ``.add()`` asks through ``facilities.validate_nesting`` — so
# ``host.anchor(SpatialElement(ifc_class="IfcBridge"))`` was accepted, and then
# both IFC backends dropped the node.
#
# Without this section:
#
#     anchor(): ACCEPTED — no exception
#     compile success: True
#     IFCBRIDGE in file: False
#     IFCBRIDGEPART in file: False
#     ifcopenshell.validate errors: 0
#
# One warning line in the log, nothing else. That combination — no exception,
# nothing in the file, and a validator reporting zero because it does not check
# WHERE rules — is the shape, and it is what
# ``test_a_facility_anchored_by_hand_compiles_to_nothing`` below keeps on the
# record what the authoring line cannot produce.
#
# The refusal is a BOUNDARY, not a table: a spatial node is a PLACE (no body
# for along=/inset=/up= to measure against, and not a product, so not a child
# of one at any coordinates). ``allowed_children_for`` already argues that
# shape — "a SpatialElement answers by which side of the spatial/physical
# boundary it is on rather than by its class, because there are ten spatial
# classes and exactly two answers".
# ---------------------------------------------------------------------------


def _facility(name="storstroem", ifc_class="IfcBridge"):
    return SpatialElement(ifc_class=ifc_class, name=name)


def _facility_part(name="deck", usage="LONGITUDINAL", ifc_class="IfcBridgePart"):
    """A part with a body, so a drop is visible as missing GEOMETRY too."""
    part = SpatialElement(ifc_class=ifc_class, name=name, usage=usage)
    part.add(Box(name="girder", start=Point(x=0, y=0, z=0),
                 end=Point(x=20000, y=8000, z=800)))
    return part


def _terrain_host(name="terrain"):
    """The container from the issue's repro — a product, with a body of its
    own, standing on a Site."""
    host = Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN",
                   name=name)
    host.add(Box(name="pad", start=Point(x=0, y=0, z=0),
                 end=Point(x=30000, y=20000, z=100)))
    return host


class TestAnchorRefusesASpatialNode:

    def test_the_measured_case_raises_and_names_the_way_out(self):
        """The reported call. It must raise rather than return the container.

        The message has to carry three things, because the author can see none
        of them from a traceback alone: WHICH container refused, WHICH node it
        refused, and the spelling that works. "Invalid child" would leave an
        author re-reading the DSL reference for the verb they already thought
        they were using.
        """
        host = _terrain_host()
        with pytest.raises(ValueError) as exc:
            host.anchor(_facility(name="b"))
        msg = str(exc.value)
        assert "Element.anchor()" in msg, msg             # the container
        assert "IfcBridge" in msg and "'b'" in msg, msg   # the child
        assert ".add()" in msg, msg                       # what to do instead
        assert "Site -> facility -> part -> products" in msg, msg
        # …and the reason, which is the part that generalizes: this verb places
        # a child in a FRAME, and a place has none.
        assert "LOCAL FRAME" in msg, msg

    @pytest.mark.parametrize("class_name", ["Wall", "Column", "Beam", "Slab",
                                            "Roof", "Space", "Element"])
    def test_every_local_frame_container_refuses_one(self, class_name):
        """Not just the ``Element`` from the repro. Anchoring is one shared
        function, so a refusal written into it covers every container that has
        the verb — and a container that grew its own copy would show up here.
        """
        host = _CONTAINER_FACTORIES[class_name]()
        host.add(Box(name="body", start=Point(x=0, y=0, z=0),
                     end=Point(x=4000, y=300, z=3000)))
        with pytest.raises(ValueError, match="cannot be anchored"):
            host.anchor(_facility(), along=1000, up=500)

    def test_the_refusal_covers_every_local_frame_container_there_is(self):
        """The list above is hand-written; this is what makes it exhaustive.

        ``tx.LOCAL_FRAME`` is derived ("is a container, has ``.anchor()``"), so a new anchorable container joins it by existing. Without
        this line it would join and skip the refusal check entirely, which is
        exactly how ``SpatialElement`` went missing from ``CONTAINERS``.
        """
        from lite_step.models import taxonomy as tx

        assert {t.__name__ for t in tx.LOCAL_FRAME} == {
            "Wall", "Column", "Beam", "Slab", "Roof", "Space", "Element"}

    def test_a_facility_part_gets_the_NESTING_rules_own_message(self):
        """One rule, two verbs, one implementation.

        ``.anchor()`` calls ``facilities.validate_nesting`` — the same function
        ``_validate_container_children`` calls — before its own refusal, so
        where the author's mistake is the spatial TREE (a part with no facility
        over it) the two verbs say the same sentence and name the same fix. A
        second, differently-worded copy of the nesting rule living inside
        ``.anchor()`` is the thing this must not become.
        """
        host = Wall(name="w1").add(Box(name="body", start=Point(x=0, y=0, z=0),
                                       end=Point(x=4000, y=300, z=3000)))
        with pytest.raises(ValueError) as exc:
            host.anchor(_facility_part(), along=0)
        anchored = str(exc.value)
        with pytest.raises(ValueError) as exc:
            Wall(name="w1").add(_facility_part())
        added = str(exc.value)

        assert anchored == added, (
            "the two verbs give different accounts of the same nesting rule:\n"
            f"  .anchor(): {anchored}\n  .add():    {added}")
        assert "must sit inside a facility" in anchored, anchored

    def test_a_refused_anchor_appends_nothing(self):
        """Atomic, like a refused ``.add()``. A container left holding a child
        it raised about is a model that compiles differently from the one the
        author was told they had."""
        host = _terrain_host()
        before = list(host._elements)
        node = _facility()
        with pytest.raises(ValueError):
            host.anchor(node)
        assert host._elements == before
        assert node._anchor_spec is None
        assert node._parent is None

    def test_a_facility_anchored_by_hand_compiles_to_nothing(self):
        """WHY a refusal and not a warning: this is what the accepted call was
        worth, kept on the record now that the authoring line cannot produce
        it.

        The tree is built by hand — the same technique as ``_park_in_openings``
        above — and the result is a project that validates clean, compiles
        clean, contains no ``IFCBRIDGE`` at all, and that
        ``ifcopenshell.validate`` reports ZERO errors on. All three at once is
        the whole argument: no exception, nothing in the file, and no
        downstream check that can tell the difference. WR31 is a WHERE rule and
        the validator does not read those (measured —
        ``tests/test_ifc43_conformance.py``).

        This assertion holds with or without the fix; it is the evidence. The
        refusal itself is pinned by the tests above, which fail with ``DID NOT
        RAISE`` the moment the raise is reverted.
        """
        pytest.importorskip("ifcopenshell")
        import contextlib
        import io

        import ifcopenshell
        import ifcopenshell.validate
        from lite_step.models.elements import ChildAnchor

        host = _terrain_host()
        bridge = _facility(name="b")
        bridge.add(_facility_part())
        # The state ``.anchor()`` would construct, reconstructed.
        bridge._anchor_spec = ChildAnchor(along=0, along_center=None, inset=0,
                                          up=0, rotations=[])
        bridge._parent = host
        host._elements.append(bridge)

        site = Site(name="site").add(host)
        proj = Project(name="anchored-facility")
        proj.add(site)
        ifc = _compile_ifc(proj)        # asserts validate + generate are clean
        upper = ifc.upper()

        # The file is REAL — the site and the terrain host are in it. Without
        # this the absences below would also hold for an empty string.
        assert "IFCSITE(" in upper
        assert "IFCGEOGRAPHICELEMENT(" in upper
        # …and the bridge is simply gone. Both backends drop it.
        assert "IFCBRIDGE(" not in upper, "the facility must be absent (it was)"
        assert "IFCBRIDGEPART(" not in upper

        # The third leg: the schema validator has nothing to say about it.
        model = ifcopenshell.file.from_string(ifc)
        logger = ifcopenshell.validate.json_logger()
        with contextlib.redirect_stderr(io.StringIO()):
            ifcopenshell.validate.validate(model, logger)
        errors = [s for s in logger.statements if s.get("level") == "error"]
        assert errors == [], (
            "the validator now reports something on a dropped facility — if "
            "that is WR31, the refusal above has a second net under it and "
            "this test's premise needs rewriting")


class TestTheLegalSpellingStillEmits:
    """The half that must keep WORKING. A refusal is only correct if the verb
    it points at does the job."""

    def test_site_add_facility_emits_the_whole_chain(self):
        """``Site.add(SpatialElement(...))`` — the spelling the refusal names.

        Asserted on the COMPILED FILE, not on the model tree: the failure this
        section is about was a model that looked right and a file that was
        missing a node, so a tree-level assertion would have passed straight
        through the entire bug.
        """
        pytest.importorskip("ifcopenshell")
        bridge = _facility(name="storstroem")
        bridge.add(_facility_part(name="deck"))
        site = Site(name="site").add(_terrain(name="ground"))
        site.add(bridge)
        proj = Project(name="legal-facility")
        proj.add(site)

        ifc = _compile_ifc(proj).upper()
        assert "IFCBRIDGE(" in ifc, "the facility the refusal points at must emit"
        assert "IFCBRIDGEPART(" in ifc
        # …carrying its products, not just the spatial husk.
        assert "IFCRELAGGREGATES(" in ifc
        assert "GIRDER" in ifc

    def test_the_host_can_still_anchor_a_physical_child(self):
        """The same container, the same verb, one line apart. The refusal is
        keyed on the CHILD, so the terrain host that cannot take a bridge takes
        a bollard exactly as before."""
        host = _terrain_host()
        host.anchor(Element(ifc_class="IfcBuildingElementProxy", name="bollard"
                            ).add(_solid()), along=1000, up=0)
        assert len(host._elements) == 2
        assert host._elements[-1]._anchor_spec is not None


class TestAnchorStillHasNoTypeTable:
    """The regression guard on the thing this must not break.

    The anchor type-table was removed on purpose (v22.3.0): a gate authors
    route around by wrapping the child in ``Element(ifc_class=…)`` is ceremony,
    and the corpus shipped that exact workaround. A spatial-boundary refusal is
    one question about one side of the model, not the table coming back — and
    the way to say so is to walk the matrix's own table rather than a fresh
    list that could quietly be narrower.

    The pairs are imported from ``test_anchor_matrix``, which owns the product
    and asserts the far stronger property (the geometry RENDERS, at the
    anchored height). What is added here is the GATE half at construction time,
    cheap enough to run over every cell in this file too.
    """

    def test_the_imported_table_is_the_matrix_table(self):
        """Imported, not re-derived — but assert it arrived non-empty and with
        the physical products in it. An import that silently became an empty
        dict would make every parametrized case below vacuous."""
        from lite_step.tests.test_anchor_matrix import CHILDREN, CONTAINERS

        assert len(CONTAINERS) >= 7 and len(CHILDREN) >= 13
        assert {"Wall", "Column", "Beam", "Slab", "Roof", "Space",
                "Element"} <= set(CHILDREN)
        assert {"Box", "Extrude", "Sweep", "Revolve", "Pipe", "Bar",
                "Mesh"} <= set(CHILDREN)
        assert "SpatialElement" not in CHILDREN, (
            "the matrix asserts every child RENDERS; a spatial node is refused "
            "now and belongs in this file, not in that one")

    def test_the_boundary_refusal_costs_six_cells_and_no_more(self):
        """The size of the hole the refusal punches in the matrix.

        A boundary question about one side of the model costs the ``Space``
        column its six PHYSICAL children and nothing else. Stated as a NUMBER
        so a refusal that grew into a type table — say by reading
        ``is_physical`` off the container too, or by refusing geometry — fails
        here with the count rather than passing quietly on a shorter matrix.
        """
        assert len(_MATRIX_REFUSED) == 6, sorted(_MATRIX_REFUSED)
        assert {c for c, _ in _MATRIX_REFUSED} == {"Space"}
        assert {ch for _, ch in _MATRIX_REFUSED} == {
            "Wall", "Column", "Beam", "Slab", "Roof", "Element"}
        assert len(_MATRIX_CELLS) == (
            len(_MATRIX_CONTAINERS) * len(_MATRIX_CHILDREN) - 6)

    @pytest.mark.parametrize("container_name,child_name", _MATRIX_CELLS)
    def test_every_physical_pairing_is_still_accepted(self, container_name,
                                                      child_name):
        """Every physical product into every PHYSICAL local-frame container,
        still taken without a word. If the refusal above were written as a type
        table — an allow-list of children — this is the sweep that would go
        red, cell by cell, naming which pairings it cost."""
        host = _MATRIX_CONTAINERS[container_name]()
        host.anchor(_MATRIX_CHILDREN[child_name](), along=1000, up=9000)
        assert host._elements[-1]._anchor_spec is not None, (
            f"{container_name}.anchor({child_name}) did not stamp an anchor")


class TestFacilityNestingRulesAreUnchanged:
    """The rule ``.anchor()`` now shares. Nothing about it moved, and these are
    here to say so rather than leave it implied — a refusal added at a second
    call site is the classic way a shared rule acquires a second, slightly
    different meaning."""

    def test_a_part_inside_a_part_is_still_legal(self):
        """IFC 4.3 composes an ``IfcFacilityPart`` out of further
        parts through the same ``IfcRelAggregates`` — a bridge's SUBSTRUCTURE
        holding its piers — so the chain has no fixed depth."""
        sub = SpatialElement(ifc_class="IfcBridgePart", name="substructure",
                             usage="LONGITUDINAL")
        sub.add(_facility_part(name="pier", usage="VERTICAL"))
        bridge = _facility().add(sub)
        # facility -> part -> part -> the product the deepest part holds.
        assert bridge._elements == [sub]
        pier = sub._elements[0]
        assert pier.ifc_class == "IfcBridgePart" and pier.name == "pier"
        assert [type(c).__name__ for c in pier._elements] == ["Box"]

    def test_a_part_outside_a_facility_is_still_refused(self):
        """The first WR31 clause, asserted at the verb that can also express
        the legal spelling. A Site is the likeliest place to put a part by
        mistake."""
        with pytest.raises(ValueError, match="must sit inside a facility"):
            Site(name="plot").add(_facility_part())

    def test_a_facility_inside_a_part_is_still_refused(self):
        """The second clause. Unreachable from ``.anchor()`` (no container with
        a local frame is a facility part), so it is asserted through ``.add()``,
        which is where it lives."""
        with pytest.raises(ValueError, match="is a facility and cannot sit"):
            _facility_part().add(_facility(name="inner"))

    def test_a_facility_on_a_site_is_still_legal(self):
        assert Site(name="plot").add(_facility()) is not None
