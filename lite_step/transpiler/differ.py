"""
Element Diff Engine: Compare normalized IFC elements to detect changes.

Compares original (from embedded source) against modified (from imported IFC)
elements to categorize each as UNCHANGED, MOVED, RESIZED, DELETED, or ADDED.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from lite_step.ifc.normalizer import ExtractedElement

logger = logging.getLogger(__name__)

# Thresholds per FR-4.5, FR-4.6
PLACEMENT_THRESHOLD_MM = 1      # Ignore translation deltas < 1mm
ROTATION_THRESHOLD_DEG1000 = 100  # 0.1° * 1000
BBOX_OVERLAP_THRESHOLD = 0.90   # 90% overlap for spatial fallback (FR-4.7)


class DeltaCategory(Enum):
    UNCHANGED = "unchanged"
    PLACEMENT_MOVED = "placement_moved"
    PLACEMENT_ROTATED = "placement_rotated"
    GEOMETRY_RESIZED = "geometry_resized"
    GEOMETRY_CONTOUR = "geometry_contour"
    STOREY_MOVED = "storey_moved"
    PROPERTY_CHANGED = "property_changed"
    DELETED = "deleted"
    ADDED = "added"


@dataclass
class ElementDelta:
    """A detected change for a single element."""
    element_id: str
    category: DeltaCategory
    original: Optional[ExtractedElement] = None   # None for ADDED
    modified: Optional[ExtractedElement] = None    # None for DELETED
    details: Dict[str, Any] = field(default_factory=dict)


def diff_elements(
    original_elements: List[ExtractedElement],
    modified_elements: List[ExtractedElement],
) -> List[ElementDelta]:
    """
    Compare original and modified element lists to produce deltas.

    Elements are matched by name (= element.id set as IFC Name field).
    Applies thresholds to filter out floating-point noise.
    Uses spatial fallback matching for delete+recreate patterns.

    Args:
        original_elements: Elements from the original (embedded) IFC
        modified_elements: Elements from the modified IFC

    Returns:
        List of ElementDelta describing each change
    """
    original_by_name = {e.name: e for e in original_elements}
    modified_by_name = {e.name: e for e in modified_elements}

    deltas = []

    # Compare elements present in both
    for name, modified in modified_by_name.items():
        if name in original_by_name:
            original = original_by_name[name]
            delta = _compare_elements(name, original, modified)
            deltas.append(delta)
        else:
            # Potentially ADDED (will check spatial fallback below)
            deltas.append(ElementDelta(
                element_id=name,
                category=DeltaCategory.ADDED,
                modified=modified,
            ))

    # Find DELETED elements
    for name, original in original_by_name.items():
        if name not in modified_by_name:
            deltas.append(ElementDelta(
                element_id=name,
                category=DeltaCategory.DELETED,
                original=original,
            ))

    # Spatial fallback matching (FR-4.7)
    deltas = _apply_spatial_fallback(deltas)

    # Log summary
    categories = {}
    for d in deltas:
        categories[d.category.value] = categories.get(d.category.value, 0) + 1
    logger.info("Diff result: %s", categories)

    return deltas


def _compare_elements(
    name: str,
    original: ExtractedElement,
    modified: ExtractedElement,
) -> ElementDelta:
    """Compare two matched elements and determine the delta category."""

    # Check placement translation
    t_orig = original.placement.translation
    t_mod = modified.placement.translation
    t_delta = tuple(abs(m - o) for o, m in zip(t_orig, t_mod))

    if any(d > PLACEMENT_THRESHOLD_MM for d in t_delta):
        return ElementDelta(
            element_id=name,
            category=DeltaCategory.PLACEMENT_MOVED,
            original=original,
            modified=modified,
            details={
                "translation_delta": tuple(m - o for o, m in zip(t_orig, t_mod)),
                "old_translation": t_orig,
                "new_translation": t_mod,
            },
        )

    # Check rotation
    r_orig = original.placement.rotation
    r_mod = modified.placement.rotation
    r_delta = tuple(abs(m - o) for o, m in zip(r_orig, r_mod))

    if any(d > ROTATION_THRESHOLD_DEG1000 for d in r_delta):
        return ElementDelta(
            element_id=name,
            category=DeltaCategory.PLACEMENT_ROTATED,
            original=original,
            modified=modified,
            details={
                "rotation_delta": tuple(m - o for o, m in zip(r_orig, r_mod)),
                "old_rotation": r_orig,
                "new_rotation": r_mod,
            },
        )

    # Check storey
    if original.storey_idx != modified.storey_idx:
        return ElementDelta(
            element_id=name,
            category=DeltaCategory.STOREY_MOVED,
            original=original,
            modified=modified,
            details={
                "old_storey": original.storey_idx,
                "new_storey": modified.storey_idx,
            },
        )

    # Check geometry
    geom_delta = _compare_geometry(original, modified)
    if geom_delta is not None:
        return ElementDelta(
            element_id=name,
            category=geom_delta,
            original=original,
            modified=modified,
            details={"geometry_changed": True},
        )

    return ElementDelta(
        element_id=name,
        category=DeltaCategory.UNCHANGED,
        original=original,
        modified=modified,
    )


def _compare_geometry(
    original: ExtractedElement,
    modified: ExtractedElement,
) -> Optional[DeltaCategory]:
    """Compare geometry between two elements. Returns category or None if unchanged."""
    og = original.geometry
    mg = modified.geometry

    if og.mode != mg.mode:
        # Mode changed entirely
        if mg.mode == "contour":
            return DeltaCategory.GEOMETRY_CONTOUR
        return DeltaCategory.GEOMETRY_RESIZED

    if og.mode == "box" and og.start and og.end and mg.start and mg.end:
        # Compare box corners
        start_delta = sum(abs(m - o) for o, m in zip(og.start, mg.start))
        end_delta = sum(abs(m - o) for o, m in zip(og.end, mg.end))
        if start_delta > PLACEMENT_THRESHOLD_MM or end_delta > PLACEMENT_THRESHOLD_MM:
            return DeltaCategory.GEOMETRY_RESIZED

    if og.mode == "contour" and og.contour and mg.contour:
        if og.contour != mg.contour:
            return DeltaCategory.GEOMETRY_CONTOUR
        if og.thickness != mg.thickness:
            return DeltaCategory.GEOMETRY_RESIZED

    return None


# ---------------------------------------------------------------------------
# Spatial fallback matching (FR-4.7)
# ---------------------------------------------------------------------------

def _apply_spatial_fallback(deltas: List[ElementDelta]) -> List[ElementDelta]:
    """
    Match ADDED elements with DELETED elements by bounding box overlap.

    If overlap >= 90%, treat as MODIFIED (the editor deleted and recreated).
    """
    added = [d for d in deltas if d.category == DeltaCategory.ADDED]
    deleted = [d for d in deltas if d.category == DeltaCategory.DELETED]

    if not added or not deleted:
        return deltas

    matched_pairs = []

    for a in added:
        best_match = None
        best_overlap = 0.0

        a_bbox = _element_bbox(a.modified)
        if a_bbox is None:
            continue

        for d in deleted:
            if d.element_id in {p[1].element_id for p in matched_pairs}:
                continue  # Already matched

            d_bbox = _element_bbox(d.original)
            if d_bbox is None:
                continue

            overlap = _bbox_overlap_fraction(a_bbox, d_bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = d

        if best_match and best_overlap >= BBOX_OVERLAP_THRESHOLD:
            matched_pairs.append((a, best_match))

    if not matched_pairs:
        return deltas

    # Replace matched ADDED/DELETED with appropriate MODIFIED category
    added_ids = {a.element_id for a, _ in matched_pairs}
    deleted_ids = {d.element_id for _, d in matched_pairs}

    result = [d for d in deltas if d.element_id not in added_ids and d.element_id not in deleted_ids]

    for added_delta, deleted_delta in matched_pairs:
        # Re-compare using the matched pair
        delta = _compare_elements(
            deleted_delta.element_id,
            deleted_delta.original,
            added_delta.modified,
        )
        if delta.category == DeltaCategory.UNCHANGED:
            delta.category = DeltaCategory.PLACEMENT_MOVED
        delta.details["spatial_fallback"] = True
        delta.details["matched_from"] = added_delta.element_id
        result.append(delta)

    return result


def _element_bbox(
    elem: Optional[ExtractedElement],
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Get axis-aligned bounding box for an element."""
    if elem is None:
        return None

    g = elem.geometry
    if g.start and g.end:
        return (
            tuple(min(s, e) for s, e in zip(g.start, g.end)),
            tuple(max(s, e) for s, e in zip(g.start, g.end)),
        )
    return None


def _bbox_overlap_fraction(
    bb1: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
    bb2: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
) -> float:
    """Compute intersection volume / min(volume1, volume2)."""
    min1, max1 = bb1
    min2, max2 = bb2

    # Intersection
    inter_min = tuple(max(a, b) for a, b in zip(min1, min2))
    inter_max = tuple(min(a, b) for a, b in zip(max1, max2))

    inter_dims = tuple(max(0, mx - mn) for mn, mx in zip(inter_min, inter_max))
    inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]

    if inter_vol == 0:
        return 0.0

    dims1 = tuple(max(0, mx - mn) for mn, mx in zip(min1, max1))
    dims2 = tuple(max(0, mx - mn) for mn, mx in zip(min2, max2))
    vol1 = dims1[0] * dims1[1] * dims1[2]
    vol2 = dims2[0] * dims2[1] * dims2[2]

    min_vol = min(vol1, vol2)
    if min_vol == 0:
        return 0.0

    return inter_vol / min_vol
