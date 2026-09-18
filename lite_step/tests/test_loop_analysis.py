"""Tests for loop analysis (uniform delta detection)."""

import pytest

from lite_step.transpiler.ast_walker import CSTElementNode, LoopContext
from lite_step.transpiler.differ import DeltaCategory, ElementDelta
from lite_step.transpiler.loop_analysis import analyze_loops

libcst = pytest.importorskip("libcst")


def _make_delta(eid: str, category: DeltaCategory, details=None) -> ElementDelta:
    return ElementDelta(
        element_id=eid,
        category=category,
        details=details or {},
    )


def _make_cst_node(eid: str, loop_id: str = None) -> CSTElementNode:
    """Create a mock CSTElementNode."""
    loop_ctx = None
    if loop_id:
        # Create a minimal For node for the loop context
        for_node = libcst.parse_statement("for i in range(8):\n    pass")
        loop_ctx = LoopContext(
            loop_node=for_node,
            loop_id=loop_id,
            iterator_var="i",
            range_args=(0, 8, 1),
        )

    call_node = libcst.parse_expression(f'Column(id="{eid}")')
    return CSTElementNode(
        element_id=eid,
        element_type="Column",
        cst_node=call_node,
        loop_context=loop_ctx,
    )


class TestAnalyzeLoops:
    def test_all_unchanged_keep(self):
        cst_nodes = {
            f"col_{i}": _make_cst_node(f"col_{i}", loop_id="loop_1")
            for i in range(4)
        }
        deltas = [
            _make_delta(f"col_{i}", DeltaCategory.UNCHANGED)
            for i in range(4)
        ]
        results = analyze_loops(deltas, cst_nodes)
        assert len(results) == 1
        assert results[0].action == "keep"

    def test_all_deleted_delete(self):
        cst_nodes = {
            f"col_{i}": _make_cst_node(f"col_{i}", loop_id="loop_1")
            for i in range(4)
        }
        deltas = [
            _make_delta(f"col_{i}", DeltaCategory.DELETED)
            for i in range(4)
        ]
        results = analyze_loops(deltas, cst_nodes)
        assert len(results) == 1
        assert results[0].action == "delete"

    def test_uniform_delta_patch(self):
        cst_nodes = {
            f"col_{i}": _make_cst_node(f"col_{i}", loop_id="loop_1")
            for i in range(4)
        }
        deltas = [
            _make_delta(f"col_{i}", DeltaCategory.PLACEMENT_MOVED,
                       details={"translation_delta": (500, 0, 0)})
            for i in range(4)
        ]
        results = analyze_loops(deltas, cst_nodes)
        assert len(results) == 1
        assert results[0].action == "patch_uniform"
        assert results[0].uniform_delta is not None

    def test_nonuniform_delta_llm_fallback(self):
        cst_nodes = {
            f"col_{i}": _make_cst_node(f"col_{i}", loop_id="loop_1")
            for i in range(4)
        }
        deltas = [
            _make_delta("col_0", DeltaCategory.PLACEMENT_MOVED,
                       details={"translation_delta": (500, 0, 0)}),
            _make_delta("col_1", DeltaCategory.PLACEMENT_MOVED,
                       details={"translation_delta": (500, 0, 0)}),
            _make_delta("col_2", DeltaCategory.PLACEMENT_MOVED,
                       details={"translation_delta": (1000, 0, 0)}),  # Different!
            _make_delta("col_3", DeltaCategory.PLACEMENT_MOVED,
                       details={"translation_delta": (500, 0, 0)}),
        ]
        results = analyze_loops(deltas, cst_nodes)
        assert len(results) == 1
        assert results[0].action == "llm_fallback"

    def test_mixed_categories_llm_fallback(self):
        cst_nodes = {
            "col_0": _make_cst_node("col_0", loop_id="loop_1"),
            "col_1": _make_cst_node("col_1", loop_id="loop_1"),
        }
        deltas = [
            _make_delta("col_0", DeltaCategory.UNCHANGED),
            _make_delta("col_1", DeltaCategory.DELETED),
        ]
        results = analyze_loops(deltas, cst_nodes)
        assert len(results) == 1
        assert results[0].action == "llm_fallback"

    def test_no_loops_empty_result(self):
        cst_nodes = {
            "wall_1": _make_cst_node("wall_1"),  # No loop context
        }
        deltas = [_make_delta("wall_1", DeltaCategory.UNCHANGED)]
        results = analyze_loops(deltas, cst_nodes)
        assert len(results) == 0
