"""tools/survey, what the m13 photo taught (2026-11-25): each test guards one way that run went wrong.

- a wrong pose of a frame-filling lattice scores near 1 on model pixels; its outline does not;
- the failures were labels, not pixels: one mislabelled control point is found and asked about;
- a perspective partial is a crop whose camera follows it (principal point), and lenses distort;
- behind a lattice face a photo is near-solid structure: only the openings tell a lattice from solid bars.
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

from tools.survey import OrthoCamera, PerspectiveCamera, agreement  # noqa: E402
from tools.survey.drone import Drone  # noqa: E402
from tools.survey.ink import INK_OPENING, INK_STRUCTURE, photo_ink  # noqa: E402
from tools.survey.inspect import load_shots, run_round  # noqa: E402
from tools.survey.intake import add_partials, add_source, check_points, fit_points  # noqa: E402
from tools.survey.overlay import outline_agreement  # noqa: E402
from tools.survey.scene import Scene  # noqa: E402

NOW = "2026-11-25T12:00:00Z"


def box(x0, y0, z0, x1, y1, z1):
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float32)
    f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return v, np.array(f, dtype=np.int32)


# -- the camera ------------------------------------------------------------------------------------------------

CAM = PerspectiveCamera.look_at((14.0, -22.0, 9.0), (0.0, 0.0, 6.0), 50.0, 400, 300)
LENS = PerspectiveCamera(CAM.position, CAM.direction, CAM.up, CAM.fov_v_deg, 400, 300, (190.0, 160.0),
                         (-0.12, 0.03, 0.001, -0.002))
WORLD = np.array([[0, 0, 0], [3, 1, 8], [-4, 2, 5], [1, -3, 11], [5, 4, 2], [-2, -2, 9]], dtype=float)


def test_a_lens_projects_and_its_rays_come_back_through_the_same_points():
    px = LENS.project(WORLD)
    o, d = LENS.ray(px)
    for X, u in zip(WORLD, d):  # the ray through the projected pixel passes through the world point
        v = (X - o) / np.linalg.norm(X - o)
        assert np.degrees(np.arccos(np.clip(u @ v, -1, 1))) < 1e-4
    plain = CAM.project(WORLD)
    assert np.abs(px - plain).max() > 5  # the lens and the principal point do move the picture


def test_the_lens_and_principal_point_survive_the_contract():
    d = LENS.to_contract()
    assert d["principalPointPx"] == [190.0, 160.0] and d["distortion"] == [-0.12, 0.03, 0.001, -0.002]
    back = PerspectiveCamera.from_contract(d)
    assert np.allclose(back.project(WORLD), LENS.project(WORLD), atol=1e-6)
    assert "principalPointPx" not in CAM.to_contract()  # a viewer camera stays as it was


def test_a_crop_sees_what_the_whole_picture_saw_there():
    crop = LENS.crop(100, 50, 350, 260)
    assert (crop.image_w, crop.image_h) == (250, 210)
    assert np.allclose(crop.project(WORLD), LENS.project(WORLD) - [100, 50], atol=1e-6)
    assert crop.focal_px == pytest.approx(LENS.focal_px)


# -- the photo gate --------------------------------------------------------------------------------------------


def lattice(h, w, sky_box, pitch=6):
    """A frame-filling lattice (lines every `pitch` px) with an open patch of sky."""
    m = np.zeros((h, w), bool)
    m[::pitch] = True
    m[:, ::pitch] = True
    x0, y0, x1, y1 = sky_box
    m[y0:y1, x0:x1] = False
    return m


def test_a_wrong_pose_of_a_frame_filling_lattice_passes_on_model_pixels_and_fails_on_its_outline():
    source = lattice(300, 400, (250, 0, 400, 120))   # sky top right
    wrong = lattice(300, 400, (0, 0, 150, 120))      # the model put the sky top left
    a = agreement(source, wrong)
    assert a.model_within_px > 0.8  # the old number cannot tell
    assert outline_agreement(source, wrong) < 0.6
    assert outline_agreement(source, source) == pytest.approx(1.0)


# -- labels, not pixels ----------------------------------------------------------------------------------------


def points(world, cam=CAM):
    return [{"px": q.tolist(), "world": X.tolist(), "by": "surveyor"} for X, q in zip(world, cam.project(world))]


def test_consistent_points_raise_no_doubt_and_one_mislabelled_point_is_named():
    good = points(WORLD)
    assert check_points(good, 400, 300) is None
    bad = [dict(p) for p in good]
    bad[3]["world"] = WORLD[4].tolist()  # the right pixel, named as another corner
    doubt = check_points(bad, 400, 300)
    assert doubt is not None and doubt["index"] == 3 and doubt["errPx"] > 20
    assert check_points(good[:4], 400, 300) is None  # four points cannot check one another
    two = [dict(p) for p in good]
    two[1]["world"], two[2]["world"] = WORLD[2].tolist(), WORLD[1].tolist()  # two labels swapped
    both = check_points(two, 400, 300)
    assert both is not None and (both["index"] is None or both["index"] in (1, 2))


def test_a_fit_with_a_mislabelled_point_fails_and_asks_about_that_point(tmp_path):
    pic = tmp_path / "photo.png"
    cv2.imwrite(str(pic), np.full((300, 400, 3), 200, np.uint8))
    src = add_source(tmp_path, pic, kind="photo", title="Photo", now=NOW)
    bad = points(WORLD)
    bad[3]["world"] = WORLD[4].tolist()
    r = fit_points(tmp_path, src["id"], bad, now=NOW)
    assert r["fit"] == "failed"
    cam = json.loads((tmp_path / "survey/cameras.json").read_text())["cameras"][0]
    assert "control point 4 disagrees" in cam["reason"]
    q = json.loads((tmp_path / "survey/questions/q-align-photo.json").read_text())
    x, y = (int(round(v)) for v in bad[3]["px"])
    assert f"({x}, {y})" in q["question"] and q["points"] == 2


# -- photo ink and partials ------------------------------------------------------------------------------------


def test_photo_ink_marks_the_sky_seen_through_the_structure():
    img = np.zeros((100, 100, 3), np.uint8)
    sky = np.ones((100, 100), bool)
    sky[20:80, 20:80] = False          # a structure ...
    sky[40:60, 40:60] = True           # ... with a hole in it
    ink = photo_ink(img, sky, close_px=31, exclude_px=[[0, 90, 100, 100]])
    assert ink[50, 50] == INK_OPENING and ink[25, 25] == INK_STRUCTURE and ink[5, 5] == 0
    assert (ink[90:] == 0).all()


def test_partials_are_crops_of_the_picture_and_a_perspective_camera_follows_them(tmp_path):
    pic = tmp_path / "photo.png"
    cv2.imwrite(str(pic), np.full((300, 400, 3), 200, np.uint8))
    src = add_source(tmp_path, pic, kind="photo", title="Photo", now=NOW)
    s = tmp_path / "survey"
    (s / "cameras.json").write_text(json.dumps({"version": "1.2", "cameras": [
        {"source": src["id"], "version": 1, "camera": LENS.to_contract(), "method": "labels", "residualMm": 0,
         "gate": "passed", "frozenAt": NOW, "reason": "test"}]}))
    (s / "ink").mkdir()
    cv2.imwrite(str(s / "ink" / f"{src['id']}.png"), np.ones((300, 400), np.uint8))
    add_partials(tmp_path, src["id"], [{"name": "lattice", "box": [100, 50, 350, 260], "purpose": "the face"}])
    add_partials(tmp_path, src["id"], [{"name": "lattice", "box": [100, 60, 350, 260], "purpose": "moved"}])
    shots = json.loads((s / "shots.json").read_text())["shots"]
    assert [x["id"] for x in shots] == ["photo-lattice"] and shots[0]["cropPx"] == [100, 60, 350, 260]
    ctx = load_shots(s)[0]
    assert ctx.camera["principalPointPx"] == [90.0, 100.0] and ctx.camera["imageW"] == 250
    with pytest.raises(ValueError, match="empty"):
        add_partials(tmp_path, src["id"], [{"name": "x", "box": [10, 10, 10, 20]}])


# -- the openings score ----------------------------------------------------------------------------------------

ORTHO = OrthoCamera((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), px_per_m=20.0, k=1.0, ax_px=200.0, c_px=350.0,
                    image_w=400, image_h=400)


def lattice_scene():
    """Five posts 1 m wide with 1 m gaps, under a beam: a lattice face seen square on."""
    parts = [(f"G-P{i}", f"member:post_{i}", "IfcMember", *box(-5 + 2 * i, -0.5, 0, -4 + 2 * i, 0.5, 10)) for i in range(5)]
    parts.append(("G-B", "member:beam", "IfcBeam", *box(-5, -0.5, 10, 5, 0.5, 11)))
    return Scene.from_meshes(parts)


def solid_scene():
    return Scene.from_meshes([("G-S", "member:solid", "IfcMember", *box(-5, -0.5, 0, 5, 0.5, 11))])


def openings_survey(tmp: Path) -> Path:
    s = tmp / "survey"
    (s / "ink").mkdir(parents=True)
    hit = Drone(lattice_scene()).render(ORTHO).hit
    ink = photo_ink(np.zeros((400, 400, 3), np.uint8), ~hit, close_px=41)  # the truth is the lattice
    cv2.imwrite(str(s / "ink" / "pl-a.png"), ink)
    (s / "sources.json").write_text(json.dumps({"version": "1.2", "sources": [
        {"id": "pl-a", "file": "sources/pl-a.png", "sha256": "0" * 64, "kind": "photo", "convention": "shaded",
         "title": "a", "addedBy": "person", "addedAt": NOW}]}))
    (s / "cameras.json").write_text(json.dumps({"version": "1.2", "cameras": [
        {"source": "pl-a", "version": 1, "camera": ORTHO.to_contract(), "method": "labels", "residualMm": 10,
         "gate": "passed", "frozenAt": NOW, "reason": "test"}]}))
    (s / "shots.json").write_text(json.dumps({"version": "1.2", "shots": [
        {"id": "a-all", "source": "pl-a", "cameraVersion": 1, "cropPx": [0, 0, 400, 400], "purpose": "all"}]}))
    (tmp / "output.ifc").write_text("ISO-10303-21;\n")
    return tmp


def request(n):
    return {"version": "1.2", "n": n, "modelFile": "output.ifc", "modelVersion": 3, "requestedBy": "surveyor",
            "requestedAt": NOW}


def test_only_the_openings_tell_a_lattice_from_solid_bars(tmp_path):
    wd = openings_survey(tmp_path)
    lat = run_round(wd, request(1), workers=1, scene=lattice_scene())
    sol = run_round(wd, request(2), workers=1, scene=solid_scene())
    assert lat["status"] == "done" and sol["status"] == "done", (lat.get("error"), sol.get("error"))
    rep_l = json.loads((wd / "survey/rounds/1/shots/a-all.json").read_text())
    rep_s = json.loads((wd / "survey/rounds/2/shots/a-all.json").read_text())
    # the structure score cannot tell them apart: every ink pixel lies on the solid block too
    assert rep_s["recall"] == pytest.approx(1.0) and rep_l["recall"] == pytest.approx(1.0)
    assert rep_l["openings"]["f1"] > 0.95 and rep_s["openings"]["recall"] == 0.0
    assert lat["openingsWeightedF1"] > 0.95 > sol["openingsWeightedF1"]


def test_a_source_without_openings_reports_none(tmp_path):
    wd = openings_survey(tmp_path)
    ink = cv2.imread(str(wd / "survey/ink/pl-a.png"), cv2.IMREAD_UNCHANGED)
    ink[ink == INK_OPENING] = 0
    cv2.imwrite(str(wd / "survey/ink/pl-a.png"), ink)
    r = run_round(wd, request(1), workers=1, scene=lattice_scene())
    rep = json.loads((wd / "survey/rounds/1/shots/a-all.json").read_text())
    assert "openings" not in rep and "openingsWeightedF1" not in r


def column_and_beam():
    return Scene.from_meshes([
        ("G-COL", "member:column", "IfcColumn", *box(-0.5, -0.5, 0.0, 0.5, 0.5, 10.0)),
        ("G-BEAM", "member:beam", "IfcBeam", *box(-6.0, -0.5, 10.0, 6.0, 0.5, 11.0)),
    ])


def test_points_that_agree_on_the_wrong_camera_pass_their_residual_and_fail_on_the_outline(tmp_path):
    # The picture was taken through CAM; the points were read as if through another camera: they agree with
    # each other perfectly (residual 0, no doubt), and only the outline shows the pose is wrong (m13).
    hit = Drone(column_and_beam()).render(CAM).hit
    img = np.zeros((300, 400, 3), np.uint8)
    img[...] = (230, 160, 90)  # a clear blue sky (BGR)
    img[hit] = (60, 75, 94)
    pic = tmp_path / "photo.png"
    cv2.imwrite(str(pic), img)
    src = add_source(tmp_path, pic, kind="photo", title="Photo", now=NOW)
    other = PerspectiveCamera.look_at((-16.0, -20.0, 4.0), (2.0, 0.0, 7.0), 50.0, 400, 300)
    pts = points(WORLD, other)
    assert check_points(pts, 400, 300) is None
    r = fit_points(tmp_path, src["id"], pts, now=NOW, scene=column_and_beam())
    cam = json.loads((tmp_path / "survey/cameras.json").read_text())["cameras"][0]
    assert r["fit"] == "failed" and cam["agreement"]["outlineWithinPx"] < 0.85, cam["agreement"]
    assert "outline" in cam["reason"]
    good = fit_points(tmp_path, src["id"], points(WORLD), now=NOW, scene=column_and_beam())
    assert good["fit"] == "passed"


# -- asking the person: candidate poses ----------------------------------------------------------------------

TOPIC = "3f2a9c1e-7b4d-4e8a-9c6f-1a2b3c4d5e6f"


def photo_of_the_model(tmp_path):
    hit = Drone(column_and_beam()).render(CAM).hit
    img = np.zeros((300, 400, 3), np.uint8)
    img[...] = (230, 160, 90)
    img[hit] = (60, 75, 94)
    pic = tmp_path / "photo.png"
    cv2.imwrite(str(pic), img)
    return add_source(tmp_path, pic, kind="photo", title="Photo", now=NOW)


def asked(tmp_path):
    src = photo_of_the_model(tmp_path)
    bad = points(WORLD)
    bad[3]["world"] = WORLD[4].tolist()  # one corner named as another
    r = fit_points(tmp_path, src["id"], bad, now=NOW, scene=column_and_beam(), topic=TOPIC)
    q = json.loads((tmp_path / "survey/questions/q-align-photo.json").read_text())
    return r, q


def test_a_failed_fit_offers_candidate_poses_ranked_by_outline_with_their_overlays(tmp_path):
    r, q = asked(tmp_path)
    assert r["fit"] == "failed"
    assert q["version"] == "1.2" and q["status"] == "open" and q["bcfTopic"] == TOPIC
    opts = q["options"]
    assert 2 <= len(opts) <= 3 and [o["id"] for o in opts] == [f"pose-{k}" for k in range(1, len(opts) + 1)]
    outlines = [o["pose"]["outlineWithinPx"] for o in opts]
    assert outlines == sorted(outlines, reverse=True) and outlines[0] > 0.9
    best = PerspectiveCamera.from_contract(opts[0]["pose"]["camera"])
    assert np.linalg.norm(np.subtract(best.position, CAM.position)) < 0.5  # the right pose comes first
    for o in opts:
        assert (tmp_path / "survey" / o["evidence"][0]).is_file() and o["label"].startswith("seen from the ")
    assert "(" in q["question"] and "Which of these" in q["question"]


def test_picking_a_candidate_freezes_it_and_closes_the_question(tmp_path):
    from tools.survey.intake import IntakeError, answer_align

    _, q = asked(tmp_path)
    r = answer_align(tmp_path, q["id"], "pose-1", now=NOW)
    assert r["fit"] == "passed" and r["cameraVersion"] == 2 and r["overlay"] == q["options"][0]["evidence"][0]
    assert r["outlineWithin2px"] == q["options"][0]["pose"]["outlineWithinPx"] and "modelPxWithin2px" not in r
    cam = json.loads((tmp_path / "survey/cameras.json").read_text())["cameras"][-1]
    assert cam["method"] == "control_points" and cam["gate"] == "passed" and "chosen by the person" in cam["reason"]
    after = json.loads((tmp_path / "survey/questions/q-align-photo.json").read_text())
    assert after["status"] == "answered" and after["answer"]["choice"] == "pose-1"
    with pytest.raises(IntakeError, match="not an open"):
        answer_align(tmp_path, q["id"], "pose-1", now=NOW)


def test_a_choice_that_is_not_a_candidate_is_refused(tmp_path):
    from tools.survey.intake import IntakeError, answer_align

    _, q = asked(tmp_path)
    with pytest.raises(IntakeError, match="not one of"):
        answer_align(tmp_path, q["id"], "pose-9", now=NOW)
    with pytest.raises(IntakeError, match="no question"):
        answer_align(tmp_path, "q-nothing", "pose-1", now=NOW)


def test_the_cli_answers(tmp_path, capsys):
    from tools.survey.__main__ import main

    _, q = asked(tmp_path)
    assert main(["answer", "--workdir", str(tmp_path), "--question", q["id"], "--choice", "pose-1"]) == 0
    assert json.loads(capsys.readouterr().out)["fit"] == "passed"
    assert main(["answer", "--workdir", str(tmp_path), "--question", q["id"], "--choice", "pose-1"]) == 2
