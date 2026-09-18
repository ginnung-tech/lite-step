"""Tests for CST patcher (LibCST transformer)."""

import pytest

libcst = pytest.importorskip("libcst")

from lite_step.ifc.normalizer import ExtractedElement, PlacementInfo, GeometryInfo
from lite_step.transpiler.ast_walker import walk_source
from lite_step.transpiler.differ import DeltaCategory, ElementDelta
from lite_step.transpiler.patcher import apply_patches


SIMPLE_SOURCE = """\
from lite_step.models.project import Project, Storey
from lite_step.models.primitives import Point
from lite_step.models.elements import Wall, Box, Extrude

# This is a project
proj = Project(name="Test")
storey = Storey(elevation=0)
proj.add_storey(storey)

wall = Wall(id="wall_north")
wall.add(Box(
    start=Point(x=0, y=0, z=0),
    end=Point(x=6000, y=3000, z=200),
    type="sketch",
    id="body_north",
))
storey.add(wall)

# Another element
col = Box(
    start=Point(x=1000, y=0, z=1000),
    end=Point(x=1300, y=3000, z=1300),
    type="sketch",
    id="col_standalone",
)
storey.add(col)

result = proj
"""


def _make_element(name, start, end):
    return ExtractedElement(
        ifc_type="IfcWall",
        name=name,
        guid="test",
        storey_idx=0,
        storey_elevation=0,
        placement=PlacementInfo(translation=(0, 0, 0), rotation=(0, 0, 0)),
        geometry=GeometryInfo(mode="box", start=start, end=end),
    )


class TestPatchPlacement:
    def test_patch_point_literal(self):
        """Replace Point(0,0,0) -> Point(1500,0,0) with comments preserved."""
        cst_nodes = walk_source(SIMPLE_SOURCE)
        deltas = [
            ElementDelta(
                element_id="body_north",
                category=DeltaCategory.PLACEMENT_MOVED,
                original=_make_element("body_north", (0, 0, 0), (6000, 3000, 200)),
                modified=_make_element("body_north", (1500, 0, 0), (7500, 3000, 200)),
                details={"translation_delta": (1500, 0, 0)},
            ),
        ]

        patched, unhandled = apply_patches(SIMPLE_SOURCE, deltas, [], cst_nodes)

        # Comments should be preserved
        assert "# This is a project" in patched
        assert "# Another element" in patched

        # The body_north start should be patched
        assert "Point(x=1500, y=0, z=0)" in patched
        assert "Point(x=7500, y=3000, z=200)" in patched

    def test_no_unhandled_for_simple_patch(self):
        cst_nodes = walk_source(SIMPLE_SOURCE)
        deltas = [
            ElementDelta(
                element_id="col_standalone",
                category=DeltaCategory.PLACEMENT_MOVED,
                original=_make_element("col_standalone", (1000, 0, 1000), (1300, 3000, 1300)),
                modified=_make_element("col_standalone", (2000, 0, 1000), (2300, 3000, 1300)),
                details={"translation_delta": (1000, 0, 0)},
            ),
        ]

        patched, unhandled = apply_patches(SIMPLE_SOURCE, deltas, [], cst_nodes)
        assert len(unhandled) == 0
        assert "Point(x=2000, y=0, z=1000)" in patched


class TestPatchDeletion:
    def test_delete_element(self):
        cst_nodes = walk_source(SIMPLE_SOURCE)
        deltas = [
            ElementDelta(
                element_id="col_standalone",
                category=DeltaCategory.DELETED,
                original=_make_element("col_standalone", (1000, 0, 1000), (1300, 3000, 1300)),
            ),
        ]

        patched, unhandled = apply_patches(SIMPLE_SOURCE, deltas, [], cst_nodes)
        assert 'id="col_standalone"' not in patched
        # Wall should still be there
        assert 'id="wall_north"' in patched


class TestPatchAdded:
    def test_insert_element(self):
        cst_nodes = walk_source(SIMPLE_SOURCE)
        new_elem = _make_element("new_wall", (0, 0, 5000), (6000, 3000, 5200))
        new_elem.ifc_type = "IfcWall"

        deltas = [
            ElementDelta(
                element_id="new_wall",
                category=DeltaCategory.ADDED,
                modified=new_elem,
            ),
        ]

        patched, unhandled = apply_patches(SIMPLE_SOURCE, deltas, [], cst_nodes)
        assert "added by external editor" in patched


class TestParametricGuard:
    def test_parametric_expression_not_patched(self):
        """Variables in arguments should NOT be patched (FR-6.6)."""
        parametric_source = """\
base_x = 1000
col = Box(
    start=Point(x=base_x, y=0, z=0),
    end=Point(x=base_x + 300, y=3000, z=300),
    type="sketch",
    id="col_param",
)
"""
        cst_nodes = walk_source(parametric_source)
        deltas = [
            ElementDelta(
                element_id="col_param",
                category=DeltaCategory.PLACEMENT_MOVED,
                original=_make_element("col_param", (1000, 0, 0), (1300, 3000, 300)),
                modified=_make_element("col_param", (2000, 0, 0), (2300, 3000, 300)),
                details={"translation_delta": (1000, 0, 0)},
            ),
        ]

        patched, unhandled = apply_patches(parametric_source, deltas, [], cst_nodes)
        assert len(unhandled) == 1
        assert unhandled[0].element_id == "col_param"
        # Original code should be preserved
        assert "base_x" in patched
