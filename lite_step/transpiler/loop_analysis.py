"""
Loop Analysis: Detect loop-generated elements and test for uniform deltas.

Determines whether loop groups can be patched uniformly (all elements
changed by the same amount) or need LLM fallback.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from lite_step.transpiler.ast_walker import CSTElementNode, get_element_ids_for_pattern
from lite_step.transpiler.differ import DeltaCategory, ElementDelta

logger = logging.getLogger(__name__)

# Tolerance for uniform delta test (FR-5.3)
UNIFORM_TOLERANCE_MM = 0.1


@dataclass
class LoopAnalysis:
    """Analysis result for a group of loop-generated elements."""
    loop_id: str
    element_ids: List[str]
    uniform_delta: Optional[ElementDelta]  # Shared delta, or None if non-uniform
    action: str  # "keep", "delete", "patch_uniform", "llm_fallback"


def analyze_loops(
    deltas: List[ElementDelta],
    cst_nodes: Dict[str, CSTElementNode],
    all_element_names: Optional[List[str]] = None,
) -> List[LoopAnalysis]:
    """
    Analyze loop-generated elements and determine actions.

    Groups elements by their LoopContext, then for each group:
    - All UNCHANGED → keep
    - All DELETED → delete
    - All same delta within tolerance → patch_uniform
    - Otherwise → llm_fallback

    Args:
        deltas: Element deltas from differ
        cst_nodes: CST element map from ast_walker
        all_element_names: All element names from modified IFC (for pattern matching)

    Returns:
        List of LoopAnalysis, one per loop group
    """
    # Group elements by loop
    loop_groups: Dict[str, List[str]] = defaultdict(list)

    for node_id, node in cst_nodes.items():
        if node.loop_context is None:
            continue

        loop_id = node.loop_context.loop_id

        if node.id_pattern and all_element_names:
            # Pattern-based: match IFC elements by prefix
            matched = get_element_ids_for_pattern(node.id_pattern, all_element_names)
            loop_groups[loop_id].extend(matched)
        elif not node_id.startswith("__pattern__"):
            loop_groups[loop_id].append(node_id)

    if not loop_groups:
        return []

    # Build delta lookup
    delta_by_id = {d.element_id: d for d in deltas}

    results = []
    for loop_id, element_ids in loop_groups.items():
        if not element_ids:
            continue

        # Get deltas for all elements in this loop
        loop_deltas = [delta_by_id[eid] for eid in element_ids if eid in delta_by_id]

        if not loop_deltas:
            results.append(LoopAnalysis(
                loop_id=loop_id,
                element_ids=element_ids,
                uniform_delta=None,
                action="keep",
            ))
            continue

        analysis = _analyze_loop_group(loop_id, element_ids, loop_deltas)
        results.append(analysis)

    logger.info(
        "Loop analysis: %d groups — %s",
        len(results),
        {a: sum(1 for r in results if r.action == a)
         for a in set(r.action for r in results)},
    )
    return results


def _analyze_loop_group(
    loop_id: str,
    element_ids: List[str],
    loop_deltas: List[ElementDelta],
) -> LoopAnalysis:
    """Determine the action for a single loop group."""
    categories = {d.category for d in loop_deltas}

    # All unchanged → keep
    if categories == {DeltaCategory.UNCHANGED}:
        return LoopAnalysis(
            loop_id=loop_id,
            element_ids=element_ids,
            uniform_delta=None,
            action="keep",
        )

    # All deleted → delete
    if categories == {DeltaCategory.DELETED}:
        return LoopAnalysis(
            loop_id=loop_id,
            element_ids=element_ids,
            uniform_delta=None,
            action="delete",
        )

    # Check for uniform delta
    uniform = _check_uniform_delta(loop_deltas)
    if uniform is not None:
        return LoopAnalysis(
            loop_id=loop_id,
            element_ids=element_ids,
            uniform_delta=uniform,
            action="patch_uniform",
        )

    # Non-uniform → LLM fallback (FR-5.4)
    return LoopAnalysis(
        loop_id=loop_id,
        element_ids=element_ids,
        uniform_delta=None,
        action="llm_fallback",
    )


def _check_uniform_delta(deltas: List[ElementDelta]) -> Optional[ElementDelta]:
    """
    Check if all deltas in a loop group are uniform (same change within tolerance).

    Returns the representative delta if uniform, None otherwise.
    """
    if not deltas:
        return None

    # All must have the same category
    categories = {d.category for d in deltas}
    if len(categories) != 1:
        return None

    category = categories.pop()

    # UNCHANGED is trivially uniform
    if category == DeltaCategory.UNCHANGED:
        return deltas[0]

    # For placement changes, compare the deltas
    if category == DeltaCategory.PLACEMENT_MOVED:
        return _check_uniform_translation(deltas)

    if category == DeltaCategory.PLACEMENT_ROTATED:
        return _check_uniform_rotation(deltas)

    if category == DeltaCategory.GEOMETRY_RESIZED:
        # Geometry resizing is too complex for uniform detection
        return None

    # Other categories: can't determine uniformity easily
    return None


def _check_uniform_translation(deltas: List[ElementDelta]) -> Optional[ElementDelta]:
    """Check if all translation deltas are within tolerance of each other."""
    ref = deltas[0].details.get("translation_delta")
    if ref is None:
        return None

    for d in deltas[1:]:
        other = d.details.get("translation_delta")
        if other is None:
            return None
        for r, o in zip(ref, other):
            if abs(r - o) > UNIFORM_TOLERANCE_MM:
                return None

    return deltas[0]


def _check_uniform_rotation(deltas: List[ElementDelta]) -> Optional[ElementDelta]:
    """Check if all rotation deltas are within tolerance of each other."""
    ref = deltas[0].details.get("rotation_delta")
    if ref is None:
        return None

    # Tolerance: 0.1° * 1000 = 100 units
    tolerance = 100

    for d in deltas[1:]:
        other = d.details.get("rotation_delta")
        if other is None:
            return None
        for r, o in zip(ref, other):
            if abs(r - o) > tolerance:
                return None

    return deltas[0]
