"""tools/survey intake: pictures taken into a survey, fitted, shown back, or asked about.

The model is a column and a beam; the "screenshots" are rendered from it, so the right camera is known.
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

from tools.survey import PerspectiveCamera  # noqa: E402
from tools.survey.__main__ import main  # noqa: E402
from tools.survey.drone import Drone  # noqa: E402
from tools.survey.intake import (  # noqa: E402
    IntakeError, add_source, camera_from_bcfv, camera_to_bcf, fit_points, fit_viewer, write_index,
)
from tools.survey.scene import Scene  # noqa: E402

NOW = "2026-11-24T21:00:00Z"
W, H = 320, 240


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


CAM = PerspectiveCamera.look_at((14.0, -22.0, 9.0), (0.0, 0.0, 6.0), 45.0, W, H)


def screenshot(tmp: Path, cam=CAM, name="grab.png") -> Path:
    """What our viewer would show: the model in grey on a flat background."""
    hit = Drone(scene()).render(cam).hit
    img = np.full((cam.image_h, cam.image_w, 3), 235, dtype=np.uint8)
    img[hit] = (120, 110, 100)
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(p), img)
    return p


def table(tmp: Path, name: str) -> dict:
    return json.loads((tmp / "survey" / name).read_text(encoding="utf-8"))


def test_a_picture_is_copied_hashed_and_listed_once(tmp_path):
    pic = screenshot(tmp_path)
    a = add_source(tmp_path, pic, kind="photo", title="Site photo", now=NOW, added_by="person")
    b = add_source(tmp_path, pic, kind="photo", title="Again", now=NOW)
    assert a == b and a["id"] == "grab" and a["file"] == "sources/grab.png"
    assert (tmp_path / "survey" / a["file"]).read_bytes() == pic.read_bytes()
    other = screenshot(tmp_path, PerspectiveCamera.look_at((20.0, 5.0, 4.0), (0, 0, 5), 40.0, W, H), "sub/grab.png")
    c = add_source(tmp_path, other, kind="photo", title="Another", now=NOW)
    assert c["id"] == "grab-2" and c["sha256"] != a["sha256"]
    assert [s["id"] for s in table(tmp_path, "sources.json")["sources"]] == ["grab", "grab-2"]


def test_a_viewer_screenshot_needs_its_camera_and_a_kind_must_be_known(tmp_path):
    pic = screenshot(tmp_path)
    with pytest.raises(IntakeError, match="viewer's camera"):
        add_source(tmp_path, pic, kind="viewer_screenshot", title="x", now=NOW)
    with pytest.raises(IntakeError, match="kind"):
        add_source(tmp_path, pic, kind="hologram", title="x", now=NOW)


def test_a_viewer_screenshot_fits_by_its_own_camera_and_is_shown_back(tmp_path):
    pic = screenshot(tmp_path)
    src = add_source(tmp_path, pic, kind="viewer_screenshot", title="My view", now=NOW, added_by="person",
                     viewer_camera=CAM.to_contract())
    r = fit_viewer(tmp_path, src["id"], scene(), now=NOW)
    assert r["fit"] == "passed" and r["cameraVersion"] == 1 and r["modelPxWithin2px"] > 0.99
    assert (tmp_path / "survey" / r["overlay"]).is_file()
    cam = table(tmp_path, "cameras.json")["cameras"][0]
    assert cam["method"] == "viewer" and cam["gate"] == "passed" and cam["overlay"] == r["overlay"]
    assert set(cam["agreement"]) == {"tolPx", "modelWithinPx", "sourceWithinPx"}
    assert not (tmp_path / "survey" / "questions").exists()
    assert r["bcfCamera"]["perspective_camera"]["field_of_view"] == pytest.approx(45.0)


def test_a_screenshot_whose_camera_is_wrong_fails_and_asks_never_guesses(tmp_path):
    turned = PerspectiveCamera.look_at((14.0, -22.0, 9.0), (4.0, 0.0, 6.0), 45.0, W, H)
    pic = screenshot(tmp_path, turned)  # the picture was taken through a different camera than it claims
    src = add_source(tmp_path, pic, kind="viewer_screenshot", title="x", now=NOW, viewer_camera=CAM.to_contract())
    r = fit_viewer(tmp_path, src["id"], scene(), now=NOW)
    assert r["fit"] == "failed" and r["modelPxWithin2px"] < 0.8
    cam = table(tmp_path, "cameras.json")["cameras"][0]
    assert cam["gate"] == "failed" and cam["camera"]["viewPoint"] == CAM.to_contract()["viewPoint"]
    q = json.loads((tmp_path / "survey/questions/q-align-grab.json").read_text(encoding="utf-8"))
    assert q["kind"] == "align" and q["source"] == "grab" and q["points"] == 4


def test_the_picture_size_wins_over_the_cameras(tmp_path):
    pic = screenshot(tmp_path)
    claimed = dict(CAM.to_contract(), imageW=999, imageH=999)
    src = add_source(tmp_path, pic, kind="viewer_screenshot", title="x", now=NOW, viewer_camera=claimed)
    r = fit_viewer(tmp_path, src["id"], scene(), now=NOW)
    cam = table(tmp_path, "cameras.json")["cameras"][0]["camera"]
    assert (cam["imageW"], cam["imageH"]) == (W, H) and r["fit"] == "passed"


def corners():
    return [[x, y, z] for x, y, z in [(-6, -0.5, 10), (6, -0.5, 10), (-6, -0.5, 11), (6, -0.5, 11),
                                      (-0.5, -0.5, 0), (0.5, -0.5, 0), (6, 0.5, 11), (-0.5, 0.5, 0)]]


def test_control_points_fit_a_photo_and_a_refit_is_a_new_version(tmp_path):
    pic = screenshot(tmp_path)
    src = add_source(tmp_path, pic, kind="photo", title="Photo", now=NOW)
    pts = [{"px": CAM.project([w])[0].tolist(), "world": w, "by": "surveyor"} for w in corners()]
    r = fit_points(tmp_path, src["id"], pts, now=NOW, scene=scene())
    assert r["fit"] == "passed" and r["residualMm"] < 5 and r["modelPxWithin2px"] > 0.95
    assert (tmp_path / "survey" / r["overlay"]).is_file()
    pts[0]["px"] = [pts[0]["px"][0] + 60, pts[0]["px"][1]]  # one point clicked badly
    r2 = fit_points(tmp_path, src["id"], pts, now=NOW)
    assert r2["cameraVersion"] == 2 and r2["fit"] == "failed"
    assert [c["version"] for c in table(tmp_path, "cameras.json")["cameras"]] == [1, 2]
    assert (tmp_path / "survey/questions/q-align-grab.json").is_file()


def test_bcf_viewpoint_cameras_read_and_write(tmp_path):
    bcfv = """<?xml version="1.0"?><VisualizationInfo Guid="a"><PerspectiveCamera>
      <CameraViewPoint><X>14</X><Y>-22</Y><Z>9</Z></CameraViewPoint>
      <CameraDirection><X>-0.5</X><Y>0.8</Y><Z>-0.1</Z></CameraDirection>
      <CameraUpVector><X>0</X><Y>0</Y><Z>1</Z></CameraUpVector>
      <FieldOfView>60</FieldOfView></PerspectiveCamera></VisualizationInfo>"""
    cam = camera_from_bcfv(bcfv, 1280, 720)
    assert cam["viewPoint"] == [14.0, -22.0, 9.0] and cam["fieldOfView"] == 60.0 and cam["imageW"] == 1280
    back = camera_to_bcf(cam)["perspective_camera"]
    assert back["camera_view_point"] == {"x": 14.0, "y": -22.0, "z": 9.0}
    assert back["aspect_ratio"] == pytest.approx(1280 / 720)
    with pytest.raises(IntakeError, match="PerspectiveCamera"):
        camera_from_bcfv('<VisualizationInfo><OrthogonalCamera/></VisualizationInfo>', 10, 10)


def test_the_index_lists_the_surveyors_files_not_the_rounds(tmp_path):
    pic = screenshot(tmp_path)
    add_source(tmp_path, pic, kind="photo", title="Photo", now=NOW)
    (tmp_path / "survey/rounds/1").mkdir(parents=True)
    (tmp_path / "survey/rounds/1/round.json").write_text("{}")
    idx = write_index(tmp_path, now=NOW)
    paths = [f["path"] for f in idx["files"]]
    assert paths == ["sources.json", "sources/grab.png"]
    import hashlib
    assert idx["files"][1]["sha256"] == hashlib.sha256(pic.read_bytes()).hexdigest()
    assert table(tmp_path, "index.json") == idx


def test_the_cli_takes_text_in_without_fitting_and_says_why_it_refuses(tmp_path, capsys):
    note = tmp_path / "brief.txt"
    note.write_text("The arches stand on pedestals.", encoding="utf-8")
    assert main(["add", "--workdir", str(tmp_path), "--picture", str(note), "--kind", "text", "--title", "Brief"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["fit"] == "not_a_picture" and out["id"] == "brief"
    assert table(tmp_path, "sources.json")["sources"][0]["convention"] == "none"
    pic = screenshot(tmp_path)
    assert main(["add", "--workdir", str(tmp_path), "--picture", str(pic), "--kind", "viewer_screenshot",
                 "--title", "x"]) == 2
    assert "viewer's camera" in json.loads(capsys.readouterr().out)["error"]
    assert main(["index", "--workdir", str(tmp_path)]) == 0
