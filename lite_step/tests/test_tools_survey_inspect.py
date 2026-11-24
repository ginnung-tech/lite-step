"""tools/survey inspect: a round of minions on a survey whose right answer is known.

A synthetic model (a column and a beam) seen through an orthographic camera; the "drawing" is ink made from
the model's own ID buffer, then moved, extended or emptied, so every measurement has a known value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
cv2 = pytest.importorskip("cv2")
try:  # the drone needs one ray caster: embreex (the default) or Open3D
    import embreex  # noqa: F401
except ImportError:
    pytest.importorskip("open3d")

from tools.survey import OrthoCamera  # noqa: E402
from tools.survey.drone import Drone  # noqa: E402
from tools.survey.inspect import InspectError, consensus, run_round  # noqa: E402
from tools.survey.scene import Scene  # noqa: E402

NORTH, Z = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
CAM = OrthoCamera(NORTH, Z, px_per_m=20.0, k=1.0, ax_px=200.0, c_px=350.0, image_w=400, image_h=400)


def box(x0, y0, z0, x1, y1, z1):
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float32)
    f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return v, np.array(f, dtype=np.int32)


def scene() -> Scene:
    return Scene.from_meshes([
        ("G-COL", "member:column", "IfcColumn", *box(-0.5, -0.5, 0.0, 0.5, 0.5, 10.0)),
        ("G-BEAM", "member:beam", "IfcBeam", *box(-6.0, -0.5, 10.0, 6.0, 0.5, 11.0)),
    ])


def make_survey(tmp: Path, ink: np.ndarray, sources=("pl-a",), gate="passed") -> Path:
    s = tmp / "survey"
    (s / "ink").mkdir(parents=True)
    src, cams, shots = [], [], []
    for sid in sources:
        src.append({"id": sid, "file": f"sources/{sid}.png", "sha256": "0" * 64, "kind": "elevation",
                    "convention": "front_slab", "title": sid, "addedBy": "person", "addedAt": "2026-11-24T00:00:00Z"})
        cams.append({"source": sid, "version": 1, "camera": CAM.to_contract(), "method": "labels", "residualMm": 10,
                     "gate": gate, "frozenAt": "2026-11-24T00:00:00Z", "reason": "test"})
        shots.append({"id": f"{sid}-all", "source": sid, "cameraVersion": 1, "cropPx": [0, 0, 400, 400], "purpose": "all"})
        cv2.imwrite(str(s / "ink" / f"{sid}.png"), ink.astype(np.uint8))
    (s / "sources.json").write_text(json.dumps({"version": "1.1", "sources": src}))
    (s / "cameras.json").write_text(json.dumps({"version": "1.1", "cameras": cams}))
    (s / "shots.json").write_text(json.dumps({"version": "1.1", "shots": shots}))
    (tmp / "output.ifc").write_text("ISO-10303-21;\n")  # present; the scene is passed in
    return tmp


def request(n=1, **kw):
    return {"version": "1.1", "n": n, "modelFile": "output.ifc", "modelVersion": 3, "requestedBy": "surveyor",
            "requestedAt": "2026-11-24T00:00:00Z", **kw}


def model_ink(dx_px=0) -> np.ndarray:
    """The model's own silhouette as structure ink (class 1), optionally moved sideways."""
    hit = Drone(scene()).render(CAM).hit
    return np.roll(hit, dx_px, axis=1).astype(np.uint8)


def test_ink_drawn_from_the_model_scores_one(tmp_path):
    wd = make_survey(tmp_path, model_ink())
    r = run_round(wd, request(), workers=1, scene=scene())
    assert r["status"] == "done", r.get("error")
    rep = json.loads((wd / "survey/rounds/1/shots/pl-a-all.json").read_text())
    assert rep["precision"] == 1.0 and rep["recall"] == 1.0
    assert all(g["support"] == 1.0 and "shiftMm" not in g for g in rep["groups"])
    assert r["scores"][0]["placement"] == 1.0 and r["scores"][0]["coverage"] == 1.0
    assert (wd / "survey/rounds/1/overlays/pl-a-all.webp").exists()
    assert rep["overlay"] == "rounds/1/overlays/pl-a-all.webp"


def test_a_drawing_half_a_metre_to_the_right_gives_the_shift_in_mm(tmp_path):
    wd = make_survey(tmp_path, model_ink(dx_px=10))  # 10 px at 20 px/m
    run_round(wd, request(), workers=1, scene=scene())
    rep = json.loads((wd / "survey/rounds/1/shots/pl-a-all.json").read_text())
    col = next(g for g in rep["groups"] if g["element"] == "member:column")
    assert col["shiftMm"] == [500, 0]
    assert col["supportShifted"] > col["support"]


def test_ink_the_model_lacks_is_boxed_in_world_metres(tmp_path):
    ink = model_ink()
    x0, y1 = (int(v) for v in CAM.project([[3.0, 0, 2.0]])[0])
    x1, y0 = (int(v) for v in CAM.project([[5.0, 0, 6.0]])[0])
    ink[y0:y1, x0:x1] = 1  # a 2 x 4 m block nobody modelled
    wd = make_survey(tmp_path, ink)
    run_round(wd, request(), workers=1, scene=scene())
    rep = json.loads((wd / "survey/rounds/1/shots/pl-a-all.json").read_text())
    m = rep["missing"][0]
    assert m["worldMin"][0] == pytest.approx(3.0, abs=0.1) and m["worldMax"][0] == pytest.approx(5.0, abs=0.1)
    assert m["worldMin"][2] == pytest.approx(2.0, abs=0.1) and m["worldMax"][2] == pytest.approx(6.0, abs=0.1)


def test_a_round_is_written_once_and_a_failure_is_said_not_hidden(tmp_path):
    wd = make_survey(tmp_path, model_ink())
    (wd / "survey/ink/pl-a.png").unlink()
    r = run_round(wd, request(), workers=1, scene=scene())
    assert r["status"] == "failed" and "ink" in r["error"] and "scores" in r and r["scores"] == []
    assert json.loads((wd / "survey/rounds/1/round.json").read_text())["status"] == "failed"
    with pytest.raises(InspectError, match="never rewritten"):
        run_round(wd, request(), workers=1, scene=scene())


@pytest.mark.parametrize("bad", ["../output.ifc", "/abs/output.ifc"])
def test_the_model_path_stays_inside_the_working_directory(tmp_path, bad):
    wd = make_survey(tmp_path, model_ink())
    r = run_round(wd, request(modelFile=bad), workers=1, scene=scene())
    assert r["status"] == "failed" and "relative path" in r["error"]


def test_a_camera_that_failed_its_gate_is_never_used(tmp_path):
    wd = make_survey(tmp_path, model_ink(), gate="failed")
    r = run_round(wd, request(), workers=1, scene=scene())
    assert r["status"] == "failed" and "gate" in r["error"]


def test_a_targeted_round_measures_only_the_named_shots(tmp_path):
    wd = make_survey(tmp_path, model_ink(), sources=("pl-a", "pl-b"))
    r = run_round(wd, request(shots=["pl-b-all"]), workers=1, scene=scene())
    assert r["shotCount"] == 1 and r["cameraVersions"] == {"pl-b": 1}
    assert not (wd / "survey/rounds/1/shots/pl-a-all.json").exists()


def test_the_process_pool_gives_the_same_reports_as_one_process(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    for wd in (a, b):
        make_survey(wd, model_ink(dx_px=10), sources=("pl-a", "pl-b"))
    one = run_round(a, request(), workers=1, scene=scene())
    two = run_round(b, request(), workers=2, scene=scene(), cache_dir=tmp_path / "cache")
    assert two["status"] == "done", two.get("error")
    for shot in ("pl-a-all", "pl-b-all"):
        ra = json.loads((a / f"survey/rounds/1/shots/{shot}.json").read_text())
        rb = json.loads((b / f"survey/rounds/1/shots/{shot}.json").read_text())
        assert ra == rb
    assert one["scores"] == two["scores"]


def test_consensus_needs_two_sources_not_two_shots():
    def rep(shot, source, shift):
        return {"shot": shot, "source": source, "groups": [{"element": "member:column", "pixels": 500, "support": 0.3,
                                                            "shiftMm": shift, "supportShifted": 0.9}], "missing": []}
    one_source = consensus([rep("a1", "pl-a", [500, 0]), rep("a2", "pl-a", [480, 0])])
    assert one_source["shifts"] == []
    two = consensus([rep("a1", "pl-a", [500, 0]), rep("b1", "pl-b", [520, 10])])
    assert two["shifts"][0]["sources"] == ["pl-a", "pl-b"]
    disagree = consensus([rep("a1", "pl-a", [500, 0]), rep("b1", "pl-b", [-500, 0])])
    assert disagree["shifts"] == []


def test_a_section_sees_only_what_lies_in_its_depth_range():
    d = Drone(Scene.from_meshes([
        ("FRONT", "front", "IfcWall", *box(-2, -3, 0, 2, -2, 4)),  # 2-3 m in front of the cut
        ("BACK", "back", "IfcWall", *box(-1, 1, 0, 1, 2, 4)),
    ]))
    full = d.render(CAM)
    names = {d.scene.elements[i].name for i in np.unique(full.element[full.hit])}
    assert names == {"front"}  # the wall in front hides the one behind it completely
    cut = OrthoCamera(NORTH, Z, 20.0, 1.0, 200.0, 350.0, 400, 400, depth_range_m=(0.0, 5.0))
    f = d.render(cut)
    names = {d.scene.elements[i].name for i in np.unique(f.element[f.hit])}
    assert names == {"back"}
    assert float(np.min(f.depth[f.hit])) == pytest.approx(1.0, abs=1e-3)


def test_a_compiled_lite_step_model_tessellates_into_named_elements(tmp_path):
    pytest.importorskip("ifcopenshell")
    from lite_step.compiler.executor import normalize_project_to_meters, validate_project_report
    from lite_step.ifc.generator import generate_ifc
    from lite_step.models import Box, Element, Point, Project
    from lite_step.models.project import Storey

    proj = Project(name="survey_test")
    st = Storey(elevation=0)
    st.add(Element(ifc_class="IfcColumn", name="column").add(Box(start=Point(x=-500, y=-500, z=0), end=Point(x=500, y=500, z=10000))))
    proj.add_storey(st)
    assert validate_project_report(proj).errors == []
    res = generate_ifc(normalize_project_to_meters(proj))
    assert res.success, res.error
    ifc = tmp_path / "m.ifc"
    ifc.write_text(res.ifc_content)
    sc = Scene.from_ifc(ifc, cache_dir=tmp_path / "cache")
    labels = [e.label for e in sc.elements if (sc.tri_element == e.index).any()]
    assert any("column" in lab for lab in labels)
    lo, hi = sc.vertices.min(axis=0), sc.vertices.max(axis=0)
    assert hi[2] - lo[2] == pytest.approx(10.0, abs=1e-3)
    again = Scene.from_ifc(ifc, cache_dir=tmp_path / "cache")  # from the cache: the same scene
    assert np.array_equal(again.vertices, sc.vertices)


def test_both_ray_casters_see_the_same_elements_at_the_same_depth():
    pytest.importorskip("embreex")
    pytest.importorskip("open3d")
    sc = scene()
    persp = __import__("tools.survey", fromlist=["PerspectiveCamera"]).PerspectiveCamera.look_at((15, -30, 12), (0, 0, 6), 40.0, 320, 240)
    for cam in (CAM, persp):
        a, b = Drone(sc, "embreex").render(cam), Drone(sc, "open3d").render(cam)
        same = (a.element == b.element).mean()
        assert same > 0.999  # edge pixels may differ by a hair
        both = a.hit & b.hit
        assert np.allclose(a.depth[both], b.depth[both], atol=1e-3)


def test_the_drone_says_which_ray_caster_it_uses():
    d = Drone(scene())
    assert d.backend in ("embreex", "open3d")
    with pytest.raises(ImportError, match="embreex or open3d"):
        Drone(scene(), backend="no-such-caster")
