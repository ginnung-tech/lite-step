"""One ``IfcGrid`` per storey, not one per axis.

The corpus baseline for ``ifcopenshell.validate`` was 19/20 clean and 4 errors,
every one of them the same defect: each ``GuideLine(line_type="grid")`` emitted
its OWN ``IfcGrid`` holding its ONE axis, and one axis cannot fill both
mandatory ``LIST [1:?]`` lists. The letter axes duplicated themselves into
``UAxes`` and ``VAxes`` (valid, and a lie — it claims axis A runs in both
directions); the number axes left ``VAxes`` empty (invalid). What is pinned
here:

1.  **One grid, all the axes**, on BOTH backends, with the U/V split taken
    from the axis DIRECTION.
2.  **The labels are not consulted.** The fixture that proves it inverts the
    drafting convention — letters running in X, numbers running in Y — so a
    label-driven implementation partitions it backwards and this suite goes
    red.
3.  **Every refusal**, each asserted on its message rather than just its type,
    because a refusal whose message does not name the fix costs the author the
    same time as no refusal.
4.  **The falsification partner for each refusal**: the neighbouring case that
    must still compile. A guard that refuses everything is as broken as one
    that refuses nothing, and reads identically in a green run.
5.  **Schema conformance of the emitted file**, from both backends, through
    the same ``ifcopenshell.validate`` the corpus gate runs
    (``tests/test_ifc43_conformance.py``).
6.  **The grid stays out of the LITESTEP_META manifest** — it corresponds to
    no DSL element, and a named entity that matches none is the v21.2
    ``IfcAnnotation`` trap.
7.  **Statistics count AXES**, not grids: ``total_elements`` is documented as
    every AUTHORED element counted once, and merging entities must not make
    authored elements disappear from it.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc import grids
from lite_step.ifc.grids import GridError
from lite_step.models import GuideLine, Point, Project, Transform

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.validate  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — local geometry only
# ---------------------------------------------------------------------------


def axis(label, x0, y0, x1, y1, **kwargs) -> GuideLine:
    return GuideLine(
        points=[Point(x=x0, y=y0, z=0), Point(x=x1, y=y1, z=0)],
        line_type="grid", label=label, **kwargs,
    )


def rectangular_project() -> Project:
    """A 2x2 grid whose LABELS are scrambled across the two families.

    "A" and "2" run in X; "B" and "1" run in Y. That is deliberate and it is
    the whole point of the fixture: ANY label-driven partition — letters into
    U, numbers into U, either way round — splits ``{A, B}`` from ``{1, 2}``
    and lands two axes in the wrong list. Only reading the direction gets
    ``{A, 2}`` and ``{B, 1}``.

    They are also authored INTERLEAVED (A, B, 2, 1) so that "U axes before V
    axes" is a distinguishable claim rather than a restatement of authoring
    order.
    """
    proj = Project(name="grid")
    proj.add(axis("A", 0, 0, 20000, 0))          # runs in X
    proj.add(axis("B", 0, 0, 0, 6000))           # runs in Y
    proj.add(axis("2", 0, 6000, 20000, 6000))    # runs in X
    proj.add(axis("1", 5000, 0, 5000, 6000))     # runs in Y
    return proj


def compile_ifc(proj: Project, backend: str = "ifcopenshell",
                source_code: str = "src"):
    normalized = normalize_project_to_meters(proj)
    from lite_step.ifc.generator import generate_ifc
    return generate_ifc(normalized, source_code=source_code)


def compile_ok(proj: Project, backend: str = "ifcopenshell",
               source_code: str = "src") -> str:
    result = compile_ifc(proj, backend, source_code=source_code)
    assert result.success, result.error
    return result.ifc_content


def compile_error(proj: Project, backend: str = "ifcopenshell") -> str:
    """The error text of a compile that must FAIL.

    Both backends wrap the raise into ``IFCGenerationResult(success=False)``,
    so the assertion is on the reported error rather than on ``pytest.raises``.
    """
    result = compile_ifc(proj, backend)
    assert not result.success, "expected the compile to be refused, it succeeded"
    return result.error or ""


def open_ifc(content: str):
    path_handle = tempfile.NamedTemporaryFile(suffix=".ifc", delete=False)
    path_handle.close()
    path = path_handle.name
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def manifest_of(content: str) -> dict:
    model = open_ifc(content)
    for pset in model.by_type("IfcPropertySet"):
        if pset.Name == "LITESTEP_META":
            for prop in pset.HasProperties:
                if prop.Name == "MANIFEST":
                    return json.loads(prop.NominalValue.wrappedValue)
    raise AssertionError("no LITESTEP_META manifest in the file")


def grid_lines(content: str) -> list:
    return [line for line in content.splitlines()
            if re.match(r"^#\d+=IFCGRID\(", line)]


def schema_errors(content: str) -> list:
    logger = ifcopenshell.validate.json_logger()
    with contextlib.redirect_stderr(io.StringIO()):
        ifcopenshell.validate.validate(open_ifc(content), logger)
    return [s for s in logger.statements if s.get("level") == "error"]


def tags(axes) -> list:
    return [a.AxisTag for a in (axes or [])]


# ---------------------------------------------------------------------------
# 1. One grid, all the axes — both backends
# ---------------------------------------------------------------------------


class TestOneGridPerStorey:

    def test_four_axes_emit_exactly_one_grid(self):
        content = compile_ok(rectangular_project(), "ifcopenshell")
        assert len(grid_lines(content)) == 1, grid_lines(content)
        model = open_ifc(content)
        grid = model.by_type("IfcGrid")[0]
        assert len(grid.UAxes) + len(grid.VAxes) == 4

    def test_axes_partition_by_direction_not_by_label(self):
        """``A`` and ``2`` run in X, so they are the U axes — together.

        Any rule that read the labels would answer ``{A, B}`` / ``{1, 2}``.
        """
        model = open_ifc(compile_ok(rectangular_project(), "ifcopenshell"))
        grid = model.by_type("IfcGrid")[0]
        assert tags(grid.UAxes) == ["A", "2"]
        assert tags(grid.VAxes) == ["B", "1"]

    def test_no_axis_appears_in_two_lists(self):
        """A letter-grid spelling that puts ONE axis in both lists validates and
            claims the axis runs in both directions."""
        model = open_ifc(compile_ok(rectangular_project(), "ifcopenshell"))
        grid = model.by_type("IfcGrid")[0]
        ids = [a.id() for a in list(grid.UAxes) + list(grid.VAxes)]
        assert len(ids) == len(set(ids)), ids

    def test_waxes_is_unset_not_empty(self):
        """``WAxes`` is ``LIST [1:?] OPTIONAL`` — ``$`` is legal, ``()`` is
        not."""
        model = open_ifc(compile_ok(rectangular_project(), "ifcopenshell"))
        assert model.by_type("IfcGrid")[0].WAxes is None

    def test_the_orthogonal_grid_says_it_is_rectangular(self):
        model = open_ifc(compile_ok(rectangular_project(), "ifcopenshell"))
        assert model.by_type("IfcGrid")[0].PredefinedType == "RECTANGULAR"

    def test_two_storeys_get_two_grids(self):
        """A product lives in exactly ONE spatial structure element, so a
        shared grid would have to pick a storey and be wrong about the other."""
        from lite_step.models import Storey

        ground = Storey(name="ground", elevation=0)
        ground.add(axis("A", 0, 0, 20000, 0), axis("1", 0, 0, 0, 6000))
        first = Storey(name="first", elevation=3000)
        first.add(axis("A", 0, 0, 20000, 0), axis("1", 0, 0, 0, 6000))
        proj = Project(name="grid", storeys=[ground, first])
        content = compile_ok(proj)
        assert len(grid_lines(content)) == 2, grid_lines(content)

    def test_a_project_with_no_grid_lines_emits_no_grid(self):
        proj = Project(name="no grid")
        proj.add(GuideLine(points=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0)],
                           line_type="boundary"))
        assert grid_lines(compile_ok(proj)) == []

class TestTheEmittedGridIsConformant:

    def test_no_schema_errors(self):
        errors = schema_errors(compile_ok(rectangular_project(), "ifcopenshell"))
        assert not errors, [str(s.get("message"))[:120] for s in errors]

    def test_both_axis_lists_are_non_empty(self):
        """The literal cardinality the corpus violated four times."""
        grid = open_ifc(compile_ok(rectangular_project(), "ifcopenshell")).by_type("IfcGrid")[0]
        assert len(grid.UAxes) >= 1 and len(grid.VAxes) >= 1


# ---------------------------------------------------------------------------
# 3. The manifest trap, and the statistics
# ---------------------------------------------------------------------------


class TestManifestAndStats:

    def test_the_grid_is_unnamed(self):
        grid = open_ifc(compile_ok(rectangular_project(), "ifcopenshell")).by_type("IfcGrid")[0]
        assert grid.Name is None

    def test_the_grid_is_not_in_the_manifest(self):
        """It corresponds to no DSL element — it is derived from four — so a
        manifest entry would claim a patch identity nothing owns."""
        manifest = manifest_of(compile_ok(rectangular_project(), "ifcopenshell"))
        assert not any(key in manifest for key in ("A", "B", "1", "2")), manifest

    def test_every_authored_axis_counts_once(self):
        """Four authored elements stay four, however many entities they merge
        into — ``total_elements`` is a count of what the author wrote."""
        result = compile_ifc(rectangular_project(), "ifcopenshell")
        assert result.success, result.error
        assert result.stats["guide_lines"] == 4
        assert result.stats["total_elements"] == 4

    def test_the_grid_is_contained_in_its_storey(self):
        model = open_ifc(compile_ok(rectangular_project(), "ifcopenshell"))
        grid = model.by_type("IfcGrid")[0]
        contained = [rel for rel in model.by_type("IfcRelContainedInSpatialStructure")
                     if grid in rel.RelatedElements]
        assert len(contained) == 1, "the grid is not in the spatial hierarchy"
        assert contained[0].RelatingStructure.is_a("IfcBuildingStorey")


# ---------------------------------------------------------------------------
# 4. Refusals — each with its falsification partner
# ---------------------------------------------------------------------------


class TestSingleDirectionIsRefused:
    """``VAxes`` is ``LIST [1:?]``, so a one-direction grid is not expressible.
    Both spellings are covered."""

    def test_one_axis_refuses(self):
        proj = Project(name="grid")
        proj.add(axis("A", 0, 0, 0, 6000))
        message = compile_error(proj)
        assert "one direction" in message
        assert "LIST [1:?]" in message

    def test_four_parallel_axes_refuse_too(self):
        proj = Project(name="grid")
        for i in range(4):
            proj.add(axis(str(i), i * 5000, 0, i * 5000, 6000))
        assert "one direction" in compile_error(proj)

    def test_the_message_names_both_fixes(self):
        proj = Project(name="grid")
        proj.add(axis("A", 0, 0, 0, 6000))
        message = compile_error(proj)
        assert "crossing axis" in message
        assert 'line_type="boundary"' in message

    def test_a_lone_boundary_line_still_compiles(self):
        """The falsification partner: the refusal is about GRID lines only."""
        proj = Project(name="not a grid")
        proj.add(GuideLine(points=[Point(x=0, y=0, z=0), Point(x=0, y=6000, z=0)],
                           line_type="boundary"))
        assert grid_lines(compile_ok(proj)) == []


class TestThreeDirectionsAreRefused:

    def test_three_families_refuse(self):
        proj = Project(name="grid")
        proj.add(axis("A", 0, 0, 10000, 0))          # 0°
        proj.add(axis("1", 0, 0, 0, 10000))          # 90°
        proj.add(axis("D", 0, 0, 10000, 10000))      # 45°
        message = compile_error(proj)
        assert "3 different directions" in message
        assert "WAxes" in message

    def test_the_message_lists_every_bearing(self):
        proj = Project(name="grid")
        proj.add(axis("A", 0, 0, 10000, 0))
        proj.add(axis("1", 0, 0, 0, 10000))
        proj.add(axis("D", 0, 0, 10000, 10000))
        message = compile_error(proj)
        for bearing in ("0.0", "90.0", "45.0"):
            assert bearing in message, message

    def test_a_skewed_two_family_grid_is_accepted(self):
        """The falsification partner. A non-orthogonal grid is a real IFC grid
        — the refusal is about the number of DIRECTIONS, not about the angle
        between two of them — and the only consequence is that we decline to
        claim ``RECTANGULAR``."""
        proj = Project(name="skewed")
        proj.add(axis("A", 0, 0, 10000, 0))
        proj.add(axis("B", 0, 4000, 10000, 4000))
        proj.add(axis("1", 0, 0, 6000, 10000))       # ~59°
        proj.add(axis("2", 4000, 0, 10000, 10000))
        model = open_ifc(compile_ok(proj))
        grid = model.by_type("IfcGrid")[0]
        assert tags(grid.UAxes) == ["A", "B"]
        assert tags(grid.VAxes) == ["1", "2"]
        assert grid.PredefinedType is None
        assert not schema_errors(compile_ok(proj))


class TestBentAndDegenerateAxesAreRefused:

    def test_a_bent_axis_refuses(self):
        proj = Project(name="grid")
        proj.add(GuideLine(
            points=[Point(x=0, y=0, z=0), Point(x=5000, y=3000, z=0),
                    Point(x=10000, y=0, z=0)],
            line_type="grid", label="A"))
        proj.add(axis("1", 0, 0, 0, 10000))
        message = compile_error(proj)
        assert "is bent" in message
        assert "'A'" in message

    def test_a_straight_three_point_axis_is_accepted(self):
        """The falsification partner: a polyline whose points are collinear
        has one direction and is a perfectly good axis."""
        proj = Project(name="grid")
        proj.add(GuideLine(
            points=[Point(x=0, y=0, z=0), Point(x=5000, y=0, z=0),
                    Point(x=10000, y=0, z=0)],
            line_type="grid", label="A"))
        proj.add(axis("1", 0, 0, 0, 10000))
        grid = open_ifc(compile_ok(proj)).by_type("IfcGrid")[0]
        assert tags(grid.UAxes) == ["A"]

    def test_a_vertical_axis_refuses(self):
        proj = Project(name="grid")
        proj.add(GuideLine(points=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=3000)],
                           line_type="grid", label="Z"))
        proj.add(axis("1", 0, 0, 10000, 0))
        message = compile_error(proj)
        assert "no extent in plan" in message

    def test_a_sloping_axis_is_accepted(self):
        """The falsification partner: an axis with a Z rise still has a plan
        direction, and the refusal is only about having NONE."""
        proj = Project(name="grid")
        proj.add(GuideLine(points=[Point(x=0, y=0, z=0), Point(x=10000, y=0, z=3000)],
                           line_type="grid", label="A"))
        proj.add(axis("1", 0, 0, 0, 10000))
        grid = open_ifc(compile_ok(proj)).by_type("IfcGrid")[0]
        assert tags(grid.UAxes) == ["A"]


class TestDuplicateTagsAreRefused:
    """Only visible once the axes share an entity — eight separate grids each
    had a unique tag trivially."""

    def test_two_axes_with_one_tag_refuse(self):
        proj = Project(name="grid")
        proj.add(axis("A", 0, 0, 10000, 0))
        proj.add(axis("A", 0, 4000, 10000, 4000))
        proj.add(axis("1", 0, 0, 0, 10000))
        message = compile_error(proj)
        assert "tagged 'A'" in message

    def test_untagged_axes_do_not_collide(self):
        """The falsification partner: ``AxisTag`` is optional, so two
        anonymous axes are not a duplicate of each other."""
        proj = Project(name="grid")
        proj.add(GuideLine(points=[Point(x=0, y=0, z=0), Point(x=10000, y=0, z=0)],
                           line_type="grid"))
        proj.add(GuideLine(points=[Point(x=0, y=4000, z=0), Point(x=10000, y=4000, z=0)],
                           line_type="grid"))
        proj.add(GuideLine(points=[Point(x=0, y=0, z=0), Point(x=0, y=10000, z=0)],
                           line_type="grid"))
        grid = open_ifc(compile_ok(proj)).by_type("IfcGrid")[0]
        assert tags(grid.UAxes) == [None, None]
        assert tags(grid.VAxes) == [None]


class TestPlacementOnAGridLineIsRefused:

    def test_placement_refuses(self):
        """Only the ifcopenshell backend: ``can_stream`` routes any project
        carrying ``placement=`` away from the streaming path, so this is the
        only backend that can reach the condition."""
        proj = Project(name="grid")
        moved = axis("A", 0, 0, 10000, 0)
        moved.placement = Transform(origin=Point(x=0, y=0, z=0))
        proj.add(moved)
        proj.add(axis("1", 0, 0, 0, 10000))
        message = compile_error(proj)
        assert "carries placement=" in message
        assert "points=" in message

    def test_placement_on_a_boundary_line_still_compiles(self):
        """The falsification partner: the refusal is about grid lines."""
        proj = Project(name="lines")
        line = GuideLine(points=[Point(x=0, y=0, z=0), Point(x=10000, y=0, z=0)],
                         line_type="boundary")
        line.placement = Transform(origin=Point(x=1000, y=0, z=0))
        proj.add(line)
        compile_ok(proj)


# ---------------------------------------------------------------------------
# 5. The policy module in isolation
# ---------------------------------------------------------------------------


class TestPolicyUnits:

    def test_no_grid_lines_is_none_not_an_error(self):
        assert grids.collect_grid([], "storey 'ground'") is None

    def test_the_container_label_is_in_every_message(self):
        with pytest.raises(GridError, match="storey 'ground'"):
            grids.collect_grid([axis("A", 0, 0, 0, 6000)], "storey 'ground'")

    def test_u_is_the_family_closer_to_x(self):
        resolved = grids.collect_grid(
            [axis("steep", 0, 0, 1000, 9000), axis("shallow", 0, 0, 9000, 1000)],
            "storey 'ground'")
        assert [a.tag for a in resolved.u_axes] == ["shallow"]
        assert [a.tag for a in resolved.v_axes] == ["steep"]

    def test_a_45_degree_tie_breaks_on_authoring_order(self):
        """Both families sit 45° from +X, so the angle cannot decide. The
        answer is the first-authored family, deterministically — the
        alternative is whichever way a dict happened to iterate."""
        first = grids.collect_grid(
            [axis("up", 0, 0, 1000, 1000), axis("down", 0, 0, 1000, -1000)],
            "storey 'ground'")
        assert [a.tag for a in first.u_axes] == ["up"]
        second = grids.collect_grid(
            [axis("down", 0, 0, 1000, -1000), axis("up", 0, 0, 1000, 1000)],
            "storey 'ground'")
        assert [a.tag for a in second.u_axes] == ["down"]

    def test_direction_is_orientation_free(self):
        """An axis authored north-to-south is in the same family as one
        authored south-to-north — otherwise a reversed axis silently becomes
        its own direction and turns a 2-family grid into a refusal."""
        resolved = grids.collect_grid(
            [axis("A", 0, 0, 10000, 0), axis("B", 10000, 4000, 0, 4000),
             axis("1", 0, 0, 0, 10000)],
            "storey 'ground'")
        assert [a.tag for a in resolved.u_axes] == ["A", "B"]

    def test_axes_are_u_before_v_in_authoring_order(self):
        """The fixture is authored A, B, 2, 1 and the U family is {A, 2}, so
        this is a claim about emission order, not a restatement of the input."""
        resolved = grids.collect_grid(
            [a for a in rectangular_project().storeys[0].elements],
            "storey 'ground'")
        assert [a.tag for a in resolved.axes] == ["A", "2", "B", "1"]

    def test_is_grid_line_reads_line_type_only(self):
        assert grids.is_grid_line(axis("A", 0, 0, 1000, 0))
        assert not grids.is_grid_line(
            GuideLine(points=[Point(x=0, y=0, z=0), Point(x=1, y=0, z=0)],
                      line_type="setback"))
