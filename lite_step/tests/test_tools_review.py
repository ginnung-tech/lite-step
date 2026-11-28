"""lite-step review: the triage finds defects by geometry and leaves intended contact alone.

Each defect has a model that shows it and, beside it, the same model without it: a rule that cannot stay quiet
on the clean twin is as wrong as one that misses the defect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("manifold3d")

from tools.review import cli, triage  # noqa: E402
from tools.survey.scene import Scene  # noqa: E402

_FACES = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                   [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=np.int32)


def box(name, cls, lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], dtype=np.float32)
    return (name, name, cls, v, _FACES)


def kinds(*meshes, parents=None):
    scene = Scene.from_meshes(meshes)
    if parents:
        idx = {e.guid: e.index for e in scene.elements}
        for child, parent in parents.items():
            scene.parent[idx[child]] = idx[parent]
    return [(f.kind, f.severity, f.group) for f in triage.run(scene).findings]


# a frame: two columns on a slab on the ground, a beam across their tops
SLAB = box("slab", "IfcSlab", (0, 0, 0), (6, 2, 0.3))
COL_A = box("col_a", "IfcColumn", (0.5, 0.8, 0.3), (0.9, 1.2, 3.3))
COL_B = box("col_b", "IfcColumn", (5.1, 0.8, 0.3), (5.5, 1.2, 3.3))
BEAM = box("beam", "IfcBeam", (0.5, 0.8, 3.3), (5.5, 1.2, 3.8))


def test_a_frame_that_stands_reports_nothing():
    assert kinds(SLAB, COL_A, COL_B, BEAM) == []


def test_a_lifted_column_and_beam_float():
    lifted = [box(n, c, (lo[0], lo[1], lo[2] + 0.5), (hi[0], hi[1], hi[2] + 0.5))
              for n, c, lo, hi in (("col_a", "IfcColumn", (0.5, 0.8, 0.3), (0.9, 1.2, 3.3)),
                                   ("col_b", "IfcColumn", (5.1, 0.8, 0.3), (5.5, 1.2, 3.3)),
                                   ("beam", "IfcBeam", (0.5, 0.8, 3.3), (5.5, 1.2, 3.8)))]
    assert kinds(SLAB, *lifted) == [("floating", "error", "floating")]


def test_a_beam_through_the_middle_of_a_wall_is_a_clash():
    wall = box("wall", "IfcWall", (2.8, 0, 0.3), (3.1, 2, 4.0))
    assert ("clash", "error", "clash:IfcBeam|IfcWall") in kinds(SLAB, COL_A, COL_B, BEAM, wall)


def test_a_beam_end_set_into_its_column_is_a_joint_not_a_clash():
    beam_in = box("beam", "IfcBeam", (0.6, 0.8, 3.0), (5.4, 1.2, 3.5))  # 300 mm into both column heads
    assert kinds(SLAB, COL_A, COL_B, beam_in) == []


def test_a_column_that_stops_short_of_its_beam_is_a_near_miss():
    short = box("col_b", "IfcColumn", (5.1, 0.8, 0.3), (5.5, 1.2, 3.26))  # 40 mm short
    found = kinds(SLAB, COL_A, short, BEAM)
    assert ("near_miss", "warning", "near_miss:IfcBeam|IfcColumn") in found


def test_parts_of_one_element_never_clash_with_each_other():
    part_a = box("part_a", "IfcBuildingElementProxy", (0.5, 0.8, 0.3), (0.9, 1.2, 2.0))
    part_b = box("part_b", "IfcBuildingElementProxy", (0.5, 0.8, 1.5), (0.9, 1.2, 3.3))  # overlaps part_a mid-way
    scene = Scene.from_meshes([SLAB, part_a, part_b])
    column = len(scene.elements)  # the IfcColumn both parts belong to: it has no body of its own
    scene.elements.append(type(scene.elements[0])(column, "col", "col", "IfcColumn"))
    scene.parent.append(-1)
    scene.parent[1] = scene.parent[2] = column
    assert triage.run(scene).findings == []
    scene.parent[1] = scene.parent[2] = -1  # the control: two separate elements overlapping mid-way do clash
    assert [f.kind for f in triage.run(scene).findings] == ["clash"]


def test_whatever_stands_in_the_ground_does_not_clash_with_it():
    ground = box("terrain", "IfcGeographicElement", (-5, -5, -2), (11, 7, 0.5))  # the slab is set into it
    assert kinds(ground, SLAB, COL_A, COL_B, BEAM) == []


def test_an_open_mesh_is_unchecked_never_floating():
    name, _, cls, v, f = box("col_b", "IfcColumn", (5.1, 0.8, 0.3), (5.5, 1.2, 3.3))
    open_col = (name, name, cls, v, f[:-2])  # one face missing: not a closed solid
    found = kinds(SLAB, COL_A, open_col, BEAM)
    assert [k for k, _, _ in found] == ["unchecked"]


def _bar(name, a, b, half=0.1):
    """A square bar from a to b (world metres), as a closed 8-corner solid."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) / np.linalg.norm(b - a)
    u = np.cross(d, [0, 0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(d, u)
    c = [p + s * half * u + t * half * w for p in (a, b) for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    v = np.array(c, dtype=np.float32)
    # winding so the normals point out: check one face and flip all when it points in
    faces = _FACES.copy()
    n = np.cross(v[faces[0, 1]] - v[faces[0, 0]], v[faces[0, 2]] - v[faces[0, 0]])
    if np.dot(n, v[faces[0]].mean(axis=0) - v.mean(axis=0)) < 0:
        faces = faces[:, ::-1].copy()
    return (name, name, "IfcMember", v, faces)


def test_an_x_of_bracing_is_a_crossing_warning_and_two_bars_along_each_other_are_a_clash():
    pier_a = box("pier_a", "IfcColumn", (0, 0, 0), (0.4, 0.4, 4))
    pier_b = box("pier_b", "IfcColumn", (5.6, 0, 0), (6, 0.4, 4))
    x1 = _bar("brace_a", (0.3, 0.2, 0.3), (5.7, 0.2, 3.7))
    x2 = _bar("brace_b", (0.3, 0.2, 3.7), (5.7, 0.2, 0.3))
    found = kinds(pier_a, pier_b, x1, x2)
    assert ("crossing", "warning", "crossing:IfcMember|IfcMember") in found
    assert not [f for f in found if f[0] == "clash"]
    doubled = _bar("brace_c", (0.3, 0.25, 0.3), (5.7, 0.25, 3.7))  # the control: a second bar drawn along the first
    assert ("clash", "error", "clash:IfcMember|IfcMember") in kinds(pier_a, pier_b, x1, doubled)


def test_walls_lapped_at_a_corner_are_a_joint_and_a_wall_through_the_middle_of_another_is_not():
    south = box("south", "IfcWall", (0, 0, 0), (6, 0.3, 2.4))
    east = box("east", "IfcWall", (5.7, -0.1, 0), (6.0, 4, 2.4))  # laps the south wall's end, as log walls notch
    assert kinds(south, east) == []
    tee = box("east", "IfcWall", (2.9, -0.1, 0), (3.2, 4, 2.4))  # ends against south's middle: a T, also a joint
    assert kinds(south, tee) == []
    through = box("east", "IfcWall", (2.9, -2, 0), (3.2, 2, 2.4))  # the control: through the middle of both
    assert ("clash", "error", "clash:IfcWall|IfcWall") in kinds(south, through)


# -- the command -----------------------------------------------------------------------------------------------


def _report(tmp_path, findings):
    out = tmp_path / "review"
    out.mkdir(exist_ok=True)
    ifc = tmp_path / "output.ifc"
    ifc.write_text("x", encoding="utf-8")
    code = cli._write(out, tmp_path / "output.py", ifc, findings, {})
    return code, json.loads((out / "findings.json").read_text(encoding="utf-8"))


def _clash(cls_a="IfcBeam", cls_b="IfcWall", sev="error"):
    return triage.Finding("", "clash", sev, "x", ["a", "b"], {"depth_mm": 300.0}, None, sorted([cls_a, cls_b]))


def test_an_open_error_group_fails_the_review_and_groups_by_kind_and_classes(tmp_path):
    code, r = _report(tmp_path, [_clash(), _clash(), _clash("IfcCourse", "IfcPavement", "warning")])
    assert code == 1 and r["openErrors"] == 1
    assert [(g["group"], g["count"]) for g in r["groups"]] == [("clash:IfcBeam|IfcWall", 2),
                                                               ("clash:IfcCourse|IfcPavement", 1)]
    assert (tmp_path / "review" / "findings.md").read_text(encoding="utf-8").startswith("# Review of output.py")


def test_an_accepted_group_needs_a_reason_and_then_passes(tmp_path):
    (tmp_path / "review").mkdir()
    acc = tmp_path / "review" / "accepted.json"
    acc.write_text(json.dumps([{"group": "clash:IfcBeam|IfcWall"}]), encoding="utf-8")
    assert _report(tmp_path, [_clash()])[0] == 1  # no reason: nothing accepted
    acc.write_text(json.dumps([{"group": "clash:IfcBeam|IfcWall", "reason": "the beam passes a slot"}]),
                   encoding="utf-8")
    code, r = _report(tmp_path, [_clash()])
    assert code == 0 and r["groups"][0]["accepted"] == "the beam passes a slot"
