"""Boolean COUNT per representation — depth's sibling metric.

Depth and width are independent hazards with opposite signatures, and each is
invisible to the other's instrument:

* **Depth** breaks HARD and silently. Past ~13 nested levels the web viewer
  stops evaluating and renders the UNCARVED base solid — fast, and wrong.
  A timing run never sees it.
* **Width** breaks SOFT. The geometry stays correct; evaluation cost grows
  superlinearly (~N^1.33 measured). A depth metric never sees it: a mitered
  sweep is wide and SHALLOW, because the segments are unioned as a balanced
  tree.

The model that forced this (a stellarator helical coil) sat at
depth 10 — inside budget, correctly rendered by our own viewer — while being
1,318 booleans wide, which made Blender Bonsai hang with no error anywhere.

Because width has no cliff there is no defensible pass/fail line, so the
budget is a REPORTING threshold: it decides when the number is said out loud.
That is why every assertion below is about the NUMBER being right, not about
a verdict.
"""
import math

import pytest

from lite_step.compiler.csg_depth import predict_csg_depth
from lite_step.ifc.boolean_tree import CSG_WIDTH_BUDGET
from lite_step.models import (
    Box, Element, Point, Point2D, Project, Site, Sweep,
)


def _loop(nseg, r=3000, amp=400):
    pts = []
    for i in range(nseg):
        t = 2 * math.pi * i / nseg
        rr = r + amp * math.cos(3 * t)
        pts.append(Point(x=int(round(rr * math.cos(t))),
                         y=int(round(rr * math.sin(t))),
                         z=int(round(amp * math.sin(3 * t)))))
    pts.append(pts[0])           # closed
    return pts


SQUARE = [Point2D(x=-80, y=-80), Point2D(x=80, y=-80),
          Point2D(x=80, y=80), Point2D(x=-80, y=80)]


def _swept(nseg):
    proj = Project(name=f"w{nseg}")
    site = Site(name="s")
    e = Element(name=f"e{nseg}", ifc_class="IfcBuildingElementProxy")
    e.add(Sweep(name=f"sw{nseg}", profile=SQUARE, path=_loop(nseg)))
    site.add(e, carve="none")
    proj.add(site)
    return proj


# ── the count is exact ──────────────────────────────────────────────────────
#
# Every number below was read off an EMITTED file (ifcopenshell, counting
# IfcBooleanResult), not derived from the same formula under test. A closed
# mitered sweep of n segments emits 2n differences + (n-1) unions = 3n-1.

@pytest.mark.parametrize("nseg,expected", [
    (4, 11), (8, 23), (16, 47), (32, 95), (96, 287), (220, 659),
])
def test_predicted_width_matches_emitted(nseg, expected):
    assert predict_csg_depth(_swept(nseg)).max_width == expected


def test_closed_path_does_not_count_the_duplicated_endpoint():
    """The +3 bug this metric shipped with on its first draft.

    `_segment_item_depths` sets n_seg = len(pts) for a closed path, counting
    the repeated first point as a segment. Depth cannot see it (balanced), a
    sum can. If someone 'simplifies' `_segment_width` back to
    `sum(seg) + len(seg) - 1`, this goes red.
    """
    # 4 real segments -> 3*4-1 = 11, NOT 3*5-1 = 14.
    assert predict_csg_depth(_swept(4)).max_width == 11


# ── the threshold reports, it does not gate ─────────────────────────────────

def test_under_budget_is_silent():
    rep = predict_csg_depth(_swept(8))          # 23 booleans
    assert rep.max_width < CSG_WIDTH_BUDGET
    assert not any("csg width" in w for w in rep.warnings())


def test_over_budget_warns_with_the_count_and_the_consequence():
    rep = predict_csg_depth(_swept(96))         # 287 booleans
    warns = [w for w in rep.warnings() if "csg width" in w]
    assert len(warns) == 1
    w = warns[0]
    assert "287" in w, "the author acts on the NUMBER, so it must be in the text"
    assert "Bonsai" in w and "ifcopenshell" in w, "name the consequence, not the metric"
    assert "Mesh" in w, "a warning without the alternative is just bad news"


def test_width_never_blocks_a_compile():
    """Explicit product decision, same as depth: the author needs to know, not
    to be stopped. A model that warns must still produce a report."""
    rep = predict_csg_depth(_swept(220))
    assert rep.max_width == 659
    assert rep.over_width_budget, "should warn"
    assert rep.stats_line(), "and still report normally"


def test_stats_line_carries_width_on_every_compile():
    line = predict_csg_depth(_swept(8)).stats_line()
    assert "csg depth:" in line and "width: max=23" in line


# ── the two metrics are independent ─────────────────────────────────────────

def test_wide_and_shallow():
    """The stellarator shape: many booleans, low nesting. If width were a
    function of depth this could not exist, and the depth metric alone would
    have been sufficient — it was not."""
    rep = predict_csg_depth(_swept(96))
    e = max(rep.elements, key=lambda x: x.width)
    assert e.width == 287
    assert e.depth <= CSG_WIDTH_BUDGET // 10, f"expected shallow, got {e.depth}"


def test_narrow_and_deep():
    """The opposite corner: a stack of carves is deep and narrow. Pins that
    width does not simply track depth in the other direction either."""
    proj = Project(name="deep")
    site = Site(name="s")
    host = Box(name="host", start=Point(x=0, y=0, z=0),
               end=Point(x=1000, y=1000, z=1000))
    inner = host
    for i in range(6):
        nxt = Box(name=f"c{i}", start=Point(x=i * 10, y=0, z=0),
                  end=Point(x=i * 10 + 50, y=1000, z=1000))
        inner.difference(nxt)
        inner = nxt
    site.add(host, carve="none")
    proj.add(site)
    rep = predict_csg_depth(proj)
    e = max(rep.elements, key=lambda x: x.depth)
    assert e.depth >= 5, f"expected deep, got {e.depth}"
    assert e.width < CSG_WIDTH_BUDGET, f"expected narrow, got {e.width}"


def test_a_plain_box_has_no_booleans_at_all():
    proj = Project(name="plain")
    site = Site(name="s")
    site.add(Box(name="b", start=Point(x=0, y=0, z=0),
                 end=Point(x=100, y=100, z=100)), carve="none")
    proj.add(site)
    assert predict_csg_depth(proj).max_width == 0
