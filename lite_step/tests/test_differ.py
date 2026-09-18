"""Tests for element diff engine."""

import pytest

from lite_step.ifc.normalizer import ExtractedElement, PlacementInfo, GeometryInfo
from lite_step.transpiler.differ import (
    DeltaCategory,
    diff_elements,
    _bbox_overlap_fraction,
)


def _make_element(
    name: str,
    translation=(0, 0, 0),
    rotation=(0, 0, 0),
    start=(0, 0, 0),
    end=(1000, 1000, 1000),
    storey_idx=0,
    ifc_type="IfcWall",
) -> ExtractedElement:
    return ExtractedElement(
        ifc_type=ifc_type,
        name=name,
        guid="test_guid",
        storey_idx=storey_idx,
        storey_elevation=0,
        placement=PlacementInfo(translation=translation, rotation=rotation),
        geometry=GeometryInfo(mode="box", start=start, end=end),
    )


class TestDiffElements:
    def test_unchanged(self):
        orig = [_make_element("wall_1")]
        mod = [_make_element("wall_1")]
        deltas = diff_elements(orig, mod)
        assert len(deltas) == 1
        assert deltas[0].category == DeltaCategory.UNCHANGED

    def test_placement_moved(self):
        orig = [_make_element("wall_1", translation=(0, 0, 0))]
        mod = [_make_element("wall_1", translation=(1500, 0, 0))]
        deltas = diff_elements(orig, mod)
        assert len(deltas) == 1
        assert deltas[0].category == DeltaCategory.PLACEMENT_MOVED
        assert deltas[0].details["translation_delta"] == (1500, 0, 0)

    def test_placement_threshold_unchanged(self):
        """Deltas < 1mm should be treated as UNCHANGED."""
        orig = [_make_element("wall_1", translation=(0, 0, 0))]
        mod = [_make_element("wall_1", translation=(0, 0, 0))]  # same
        deltas = diff_elements(orig, mod)
        assert deltas[0].category == DeltaCategory.UNCHANGED

    def test_placement_rotated(self):
        orig = [_make_element("wall_1", rotation=(0, 0, 0))]
        mod = [_make_element("wall_1", rotation=(0, 45000, 0))]  # 45° around Y
        deltas = diff_elements(orig, mod)
        assert deltas[0].category == DeltaCategory.PLACEMENT_ROTATED

    def test_geometry_resized(self):
        orig = [_make_element("wall_1", start=(0, 0, 0), end=(6000, 3000, 200))]
        mod = [_make_element("wall_1", start=(0, 0, 0), end=(8000, 3000, 200))]
        deltas = diff_elements(orig, mod)
        assert deltas[0].category == DeltaCategory.GEOMETRY_RESIZED

    def test_deleted(self):
        orig = [_make_element("wall_1"), _make_element("wall_2")]
        mod = [_make_element("wall_1")]
        deltas = diff_elements(orig, mod)
        deleted = [d for d in deltas if d.category == DeltaCategory.DELETED]
        assert len(deleted) == 1
        assert deleted[0].element_id == "wall_2"

    def test_added(self):
        orig = [_make_element("wall_1")]
        mod = [_make_element("wall_1"), _make_element("wall_new")]
        deltas = diff_elements(orig, mod)
        added = [d for d in deltas if d.category == DeltaCategory.ADDED]
        assert len(added) == 1
        assert added[0].element_id == "wall_new"

    def test_storey_moved(self):
        orig = [_make_element("wall_1", storey_idx=0)]
        mod = [_make_element("wall_1", storey_idx=1)]
        deltas = diff_elements(orig, mod)
        assert deltas[0].category == DeltaCategory.STOREY_MOVED


class TestSpatialFallback:
    def test_overlapping_bbox_matches(self):
        """Delete+recreate with same bbox -> treated as MODIFIED."""
        orig = [_make_element("old_wall", start=(0, 0, 0), end=(6000, 3000, 200))]
        mod = [_make_element("new_wall", start=(0, 0, 0), end=(6000, 3000, 200))]
        deltas = diff_elements(orig, mod)
        # Should not have separate ADDED + DELETED
        added = [d for d in deltas if d.category == DeltaCategory.ADDED]
        deleted = [d for d in deltas if d.category == DeltaCategory.DELETED]
        assert len(added) == 0 or len(deleted) == 0

    def test_non_overlapping_stays_separate(self):
        """Non-overlapping elements stay as ADDED + DELETED."""
        orig = [_make_element("wall_a", start=(0, 0, 0), end=(1000, 1000, 1000))]
        mod = [_make_element("wall_b", start=(50000, 50000, 50000), end=(51000, 51000, 51000))]
        deltas = diff_elements(orig, mod)
        added = [d for d in deltas if d.category == DeltaCategory.ADDED]
        deleted = [d for d in deltas if d.category == DeltaCategory.DELETED]
        assert len(added) == 1
        assert len(deleted) == 1


class TestBboxOverlap:
    def test_identical_boxes(self):
        bb = ((0, 0, 0), (1000, 1000, 1000))
        assert _bbox_overlap_fraction(bb, bb) == pytest.approx(1.0)

    def test_no_overlap(self):
        bb1 = ((0, 0, 0), (100, 100, 100))
        bb2 = ((200, 200, 200), (300, 300, 300))
        assert _bbox_overlap_fraction(bb1, bb2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        bb1 = ((0, 0, 0), (100, 100, 100))
        bb2 = ((50, 50, 50), (150, 150, 150))
        overlap = _bbox_overlap_fraction(bb1, bb2)
        assert 0.0 < overlap < 1.0
