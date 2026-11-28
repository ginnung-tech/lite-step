"""tools/survey for freeform shapes: a drawing fitted as an orthographic camera, and the outline measured.

The "drawings" are made from known shapes through a known camera, so every right answer is known: a box 2 m wide
and 10 m tall, seen from the south. Each defect has its twin without it, and the twin must stay quiet.
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
from tools.survey.intake import IntakeError, add_source, crop_picture, fit_points_ortho  # noqa: E402
from tools.survey.scene import Scene  # noqa: E402
from tools.survey.shape import Window, compare, drawing_silhouette  # noqa: E402

NOW = "2026-11-28T12:00:00Z"
W, H = 400, 600
PPM = 40.0  # px per metre
SOUTH = OrthoCamera((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), PPM, 1.0, 200.0, 550.0, W, H)  # x=0 at px 200, z=0 at row 550


def box(x0, y0, z0, x1, y1, z1):
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float32)
    f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return v, np.array(f, dtype=np.int32)


def scene_of(*parts):
    return Scene.from_meshes([(f"G-{n}", f"covering:{n}", "IfcCovering", *b) for n, b in parts])


BODY = ("body", box(-1, -1, 0, 1, 1, 10))


def drawing(*rects, gap_px=0, inner_lines=True):
    """A line drawing: the outline of each world rectangle (x0, z0, x1, z1) seen from the south, thin grey lines,
    with lines inside it (drapery) and, optionally, a gap broken into its left side."""
    img = np.full((H, W), 255, np.uint8)
    for x0, z0, x1, z1 in rects:
        (a, b), (c, d) = SOUTH.project([[x0, 0, z0], [x1, 0, z1]])
        p0, p1 = (int(round(min(a, c))), int(round(min(b, d)))), (int(round(max(a, c))), int(round(max(b, d))))
        cv2.rectangle(img, p0, p1, 150, 1)
        if inner_lines:
            for y in range(p0[1] + 7, p1[1], 9):
                cv2.line(img, (p0[0] + 3, y), (p1[0] - 3, y + 4), 170, 1)
        if gap_px:
            y = (p0[1] + p1[1]) // 2
            img[y:y + gap_px, p0[0]] = 255
    return img


def measure(scene, img, **kw):
    frame = Drone(scene).render(SOUTH)
    pic = drawing_silhouette(img, close_px=3)
    return compare(pic, frame.element, frame.depth, SOUTH, [e.label for e in scene.elements], **kw)


# -- the drawing's silhouette ----------------------------------------------------------------------------------


def test_a_line_drawing_fills_through_its_inner_lines_and_a_small_gap_in_its_outline():
    sil = drawing_silhouette(drawing((-1, 0, 1, 10), gap_px=3), close_px=3)
    inside = sil[int(550 - 5 * PPM), 200]  # the middle of the box
    assert inside and sil.sum() == pytest.approx(2 * 10 * PPM * PPM, rel=0.08)


def test_background_open_to_the_outside_stays_open():
    # a U: two arms 1 m wide on a base, 2 m of background between them open above (an arm raised beside the head)
    u = [(-2, 0), (2, 0), (2, 10), (1, 10), (1, 4), (-1, 4), (-1, 10), (-2, 10)]
    poly = SOUTH.project([[x, 0, z] for x, z in u]).round().astype(np.int32)
    img = np.full((H, W), 255, np.uint8)
    cv2.polylines(img, [poly], True, 150, 1)
    sil = drawing_silhouette(img, close_px=3)
    row = int(550 - 7 * PPM)
    assert not sil[row, 200] and sil[row, int(200 + 1.5 * PPM)] and sil[row, int(200 - 1.5 * PPM)]


# -- the outline compared --------------------------------------------------------------------------------------


def test_the_same_shape_has_no_findings():
    r = measure(scene_of(BODY), drawing((-1, 0, 1, 10)))
    assert r["findings"] == [] and r["iou"] > 0.95


def test_a_picture_wider_on_one_side_is_a_shortfall_on_that_side_at_that_height_by_the_element_there():
    head = ("head", box(-1, -1, 6, 1, 1, 10))
    robe = ("robe", box(-1, -1, 0, 1, 1, 6))
    img = drawing((-1, 0, 1, 6), (-1, 6, 1.5, 10))  # the picture's upper part is 500 mm wider on the east
    r = measure(scene_of(robe, head), img)
    f = r["findings"][0]
    assert f["kind"] == "falls_short" and f["side"] == "east (+X)"
    assert f["meanMm"] == pytest.approx(-500, abs=60) and f["elements"][0] == "covering:head"
    assert 5.6 <= f["vM"][0] <= 6.5 and 9.5 <= f["vM"][1] <= 10.1
    assert not [g for g in r["findings"] if g["side"] == "west (-X)" if "side" in g]


def test_what_the_model_lacks_entirely_is_missing_with_its_area():
    img = drawing((-1, 0, 1, 10), (-0.3, 10, 0.3, 13))  # a torch 3 m above the body the model does not have
    f = measure(scene_of(BODY), img)["findings"][0]
    assert f["kind"] == "missing" and f["areaM2"] == pytest.approx(0.6 * 3, rel=0.2)


def test_lettering_far_from_the_model_and_the_sheet_frame_are_left_out_and_reported():
    img = drawing((-1, 0, 1, 10))
    cv2.putText(img, "S", (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 3)  # lettering more than 3 m from the body
    cv2.line(img, (0, 5), (W - 1, 5), 0, 3)  # the sheet's frame
    r = measure(scene_of(BODY), img)
    assert r["findings"] == [] and r["ignoredPictureM2"] > 0.2


def test_a_window_limits_the_measurement_to_a_height_range():
    img = drawing((-1, 0, 1.6, 4), (-1, 4, 1, 10))  # wrong below 4 m only
    assert measure(scene_of(BODY), img)["findings"]
    assert measure(scene_of(BODY), img, window=Window(v0=4.5))["findings"] == []


# -- an orthographic fit at intake -----------------------------------------------------------------------------


def _survey_with_drawing(tmp_path, kind="elevation"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pic = tmp_path / "sheet.png"
    cv2.imwrite(str(pic), drawing((-1, 0, 1, 10)))
    return add_source(tmp_path, pic, kind=kind, title="south elevation", now=NOW)


def _points(world):
    return [{"px": SOUTH.project([w])[0].tolist(), "world": w} for w in world]


def test_a_drawing_is_fitted_orthographic_from_three_points_and_recovers_its_scale(tmp_path):
    src = _survey_with_drawing(tmp_path)
    r = fit_points_ortho(tmp_path, src["id"], _points([[-1, 0, 0], [1, 0, 10], [1, 0, 0]]), view="south", now=NOW)
    assert r["fit"] == "passed"
    cam = json.loads((tmp_path / "survey" / "cameras.json").read_text())["cameras"][-1]["camera"]
    assert cam["kind"] == "ortho" and cam["pxPerM"] == pytest.approx(PPM, rel=1e-6)


def test_a_mislabelled_point_fails_the_fit_and_is_named(tmp_path):
    src = _survey_with_drawing(tmp_path)
    pts = _points([[-1, 0, 0], [1, 0, 10], [1, 0, 0], [-1, 0, 10]])
    pts[3]["world"] = [1, 0, 10]  # the top-left corner labelled as the top-right one
    r = fit_points_ortho(tmp_path, src["id"], pts, view="south", now=NOW)
    assert r["fit"] == "failed"
    entry = json.loads((tmp_path / "survey" / "cameras.json").read_text())["cameras"][-1]
    assert "control point 4" in entry["reason"]


def test_two_points_need_the_drawings_scale_and_a_photo_is_never_fitted_orthographic(tmp_path):
    src = _survey_with_drawing(tmp_path)
    with pytest.raises(IntakeError, match="three|3"):
        fit_points_ortho(tmp_path, src["id"], _points([[-1, 0, 0], [1, 0, 10]]), view="south", now=NOW)
    r = fit_points_ortho(tmp_path, src["id"], _points([[-1, 0, 0], [1, 0, 10]]), view="south", now=NOW, px_per_m=PPM)
    assert r["fit"] == "passed"
    photo = _survey_with_drawing(tmp_path / "p", kind="photo")
    with pytest.raises(IntakeError, match="orthographic"):
        fit_points_ortho(tmp_path / "p", photo["id"], _points([[-1, 0, 0], [1, 0, 10], [1, 0, 0]]), view="south",
                         now=NOW)


def test_one_figure_of_a_sheet_is_cropped_into_a_picture_of_its_own(tmp_path):
    pic = tmp_path / "sheet.png"
    cv2.imwrite(str(pic), drawing((-1, 0, 1, 10)))
    out = crop_picture(tmp_path, pic, [100, 100, 300, 500])
    assert cv2.imread(str(out)).shape[:2] == (400, 200)
    with pytest.raises(IntakeError, match="not inside"):
        crop_picture(tmp_path, pic, [100, 100, 900, 500])
