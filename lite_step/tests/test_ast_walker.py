"""Tests for LibCST AST walker (source -> element map)."""

import pytest

libcst = pytest.importorskip("libcst")

from lite_step.transpiler.ast_walker import (
    walk_source,
    is_parametric,
    get_element_ids_for_pattern,
)


SIMPLE_SOURCE = """\
from lite_step.models.project import Project, Storey
from lite_step.models.primitives import Point
from lite_step.models.elements import Wall, Box, Extrude, Column

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

col = Column(id="col_1")
storey.add(col)

result = proj
"""

LOOP_SOURCE = """\
from lite_step.models.project import Project, Storey
from lite_step.models.primitives import Point
from lite_step.models.elements import Column, Box, Extrude

proj = Project(name="Test")
storey = Storey(elevation=0)
proj.add_storey(storey)

for i in range(8):
    col = Column(id=f"col_{i}")
    col.add(Box(
        start=Point(x=i * 3000, y=0, z=0),
        end=Point(x=i * 3000 + 300, y=3000, z=300),
        type="sketch",
    ))
    storey.add(col)

result = proj
"""

PARAMETRIC_SOURCE = """\
base_x = 1000
bonus = 500
wall = Wall(id="wall_param", start=Point(x=base_x + bonus, y=0, z=0))
"""


class TestWalkSource:
    def test_simple_element_extraction(self):
        elements = walk_source(SIMPLE_SOURCE)
        assert "wall_north" in elements
        assert "body_north" in elements
        assert "col_1" in elements

    def test_element_types(self):
        elements = walk_source(SIMPLE_SOURCE)
        assert elements["wall_north"].element_type == "Wall"
        assert elements["body_north"].element_type == "Box"
        assert elements["col_1"].element_type == "Column"

    def test_keyword_args_captured(self):
        elements = walk_source(SIMPLE_SOURCE)
        wall = elements["wall_north"]
        assert "id" in wall.keyword_args

    def test_no_loop_context_for_simple(self):
        elements = walk_source(SIMPLE_SOURCE)
        assert elements["wall_north"].loop_context is None
        assert elements["col_1"].loop_context is None


class TestLoopDetection:
    def test_loop_context_detected(self):
        elements = walk_source(LOOP_SOURCE)
        # Should find the pattern-based element
        pattern_keys = [k for k in elements if k.startswith("__pattern__")]
        assert len(pattern_keys) == 1

    def test_fstring_pattern(self):
        elements = walk_source(LOOP_SOURCE)
        pattern_keys = [k for k in elements if k.startswith("__pattern__")]
        assert len(pattern_keys) == 1
        node = elements[pattern_keys[0]]
        assert node.id_pattern == "col_"

    def test_loop_context_has_iterator(self):
        elements = walk_source(LOOP_SOURCE)
        pattern_keys = [k for k in elements if k.startswith("__pattern__")]
        node = elements[pattern_keys[0]]
        assert node.loop_context is not None
        assert node.loop_context.iterator_var == "i"

    def test_loop_context_range_args(self):
        elements = walk_source(LOOP_SOURCE)
        pattern_keys = [k for k in elements if k.startswith("__pattern__")]
        node = elements[pattern_keys[0]]
        assert node.loop_context.range_args == (0, 8, 1)


class TestParametricDetection:
    def test_literal_not_parametric(self):
        node = libcst.parse_expression("1000")
        assert is_parametric(node) is False

    def test_name_is_parametric(self):
        node = libcst.parse_expression("base_x")
        assert is_parametric(node) is True

    def test_binary_op_is_parametric(self):
        node = libcst.parse_expression("base_x + 500")
        assert is_parametric(node) is True

    def test_point_with_literals_not_parametric(self):
        node = libcst.parse_expression("Point(x=100, y=200, z=300)")
        assert is_parametric(node) is False

    def test_point_with_variable_is_parametric(self):
        node = libcst.parse_expression("Point(x=base_x, y=200, z=300)")
        assert is_parametric(node) is True

    def test_function_call_is_parametric(self):
        node = libcst.parse_expression("compute_offset()")
        assert is_parametric(node) is True


class TestPatternMatching:
    def test_match_loop_pattern(self):
        names = ["col_0", "col_1", "col_2", "col_3", "wall_north", "beam_1"]
        matches = get_element_ids_for_pattern("col_", names)
        assert matches == ["col_0", "col_1", "col_2", "col_3"]

    def test_no_match(self):
        names = ["wall_north", "beam_1"]
        matches = get_element_ids_for_pattern("col_", names)
        assert matches == []

    def test_partial_prefix_no_false_match(self):
        names = ["column_1", "col_0"]
        matches = get_element_ids_for_pattern("col_", names)
        assert matches == ["col_0"]  # "column_1" suffix "umn_1" is not a digit
