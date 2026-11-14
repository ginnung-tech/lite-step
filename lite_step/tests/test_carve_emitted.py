"""The carve report and ``assert_carved`` describe the FILE (#20), and a contour
Wall body has the same frame as the contour Extrude it is (#21).

Displacement PLANS carves before any IFC exists; the generator records which
inferred carves it actually emitted. Both consumers that tell an author "this
carves that" — the ``carves:`` report and ``assert_carved`` — read the emitted
set, so a planned carve the file does not hold can satisfy neither.
"""
from __future__ import annotations

import pytest

pytest.importorskip("ifcopenshell")

from lite_step.compiler.displacement import apply_displacement, carve_pairs  # noqa: E402
from lite_step.compiler.executor import normalize_project_to_meters  # noqa: E402
from lite_step.ifc.generator import generate_ifc  # noqa: E402
from lite_step.models import Box, Extrude, Point, Project, Wall  # noqa: E402
from lite_step.models.project import Storey  # noqa: E402


def _overlapping(assert_pair=None):
    p = Project(name="carves")
    s = Storey(name="ground")
    s.add(Box(name="slab", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=1000, z=300)))
    s.add(Box(name="post", start=Point(x=900, y=400, z=-200), end=Point(x=1100, y=600, z=2000)))
    p.add_storey(s)
    if assert_pair:
        p.assert_carved(assert_pair[0], by=assert_pair[1])
    return normalize_project_to_meters(p)


def _plant_ghost(proj):
    """A planned inferred pair no operand carries: planned, never emitted."""
    apply_displacement(proj)
    proj._carve_pairs.append(("box:ghost:storey:ground", "box:post:storey:ground",
                              "inferred"))


def test_after_generation_the_pairs_are_the_emitted_ones(capsys):
    proj = _overlapping()
    _plant_ghost(proj)
    assert ("box:ghost:storey:ground", "box:post:storey:ground", "inferred") \
        in carve_pairs(proj), "before emission the planned set is the answer"

    res = generate_ifc(proj, source_code=None)
    assert res.success, res.error
    pairs = {(lo, w) for lo, w, _m in carve_pairs(proj)}
    assert ("box:slab:storey:ground", "box:post:storey:ground") in pairs
    assert ("box:ghost:storey:ground", "box:post:storey:ground") not in pairs

    err = capsys.readouterr().err
    assert "carves: 1 pairs" in err, err
    assert "carve emission: planned=2 emitted=1" in err
    assert "box:ghost:storey:ground <- box:post:storey:ground was planned" in err


def test_assert_carved_reads_the_emitted_set():
    proj = _overlapping(assert_pair=("box:ghost:storey:ground", "box:post:storey:ground"))
    _plant_ghost(proj)
    res = generate_ifc(proj, source_code=None)
    assert not res.success
    assert "assert_carved('box:ghost:storey:ground', by='box:post:storey:ground') is not satisfied" in res.error


def test_assert_carved_on_a_real_carve_still_passes():
    """The falsification partner: the switch must not refuse a true pair."""
    res = generate_ifc(_overlapping(assert_pair=("box:slab:storey:ground", "box:post:storey:ground")), source_code=None)
    assert res.success, res.error


def test_a_second_generation_does_not_reuse_the_first_runs_set():
    proj = _overlapping()
    assert generate_ifc(proj, source_code=None).success
    first = carve_pairs(proj)
    assert generate_ifc(proj, source_code=None).success
    assert carve_pairs(proj) == first


# ---------------------------------------------------------------------------
# #21 — one contour frame
# ---------------------------------------------------------------------------

_GABLE = [Point(x=0, y=0, z=0), Point(x=4000, y=0, z=0), Point(x=4000, y=0, z=2500),
          Point(x=2000, y=0, z=3800), Point(x=0, y=0, z=2500)]


def _ifc(*elements):
    import os
    import tempfile

    import ifcopenshell
    p = Project(name="frame")
    s = Storey(name="ground")
    for e in elements:
        s.add(e)
    p.add_storey(s)
    res = generate_ifc(normalize_project_to_meters(p), source_code=None)
    assert res.success, res.error
    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        f.write(res.ifc_content)
    try:
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _frame_of(product):
    rp = product.ObjectPlacement.RelativePlacement
    solid = product.Representation.Representations[0].Items[0]
    pts = [tuple(round(c, 9) for c in p.Coordinates)
           for p in solid.SweptArea.OuterCurve.Points]
    return (tuple(rp.Location.Coordinates), tuple(rp.Axis.DirectionRatios),
            tuple(rp.RefDirection.DirectionRatios), pts, solid.Depth)


def test_a_contour_wall_body_has_the_extrude_frame():
    wall = Wall(name="gable")
    wall.add(Extrude(contour=list(_GABLE), thickness=300))
    wall_model = _ifc(wall)
    extrude_model = _ifc(Extrude(name="gable", contour=list(_GABLE), thickness=300))
    assert _frame_of(wall_model.by_type("IfcWall")[0]) == \
        _frame_of(extrude_model.by_type("IfcBuildingElementProxy")[0])
