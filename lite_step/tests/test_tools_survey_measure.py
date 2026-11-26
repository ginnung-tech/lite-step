"""tools/survey measure: the camera and the script's dimensions, fitted together from one picture.

Staging 2026-11-26: a carport built from one photo had the right parts and guessed sizes (150 mm posts on a
3 m grid); the photo showed ~300 mm posts on a ~4 m grid. Fitting a camera to the guess failed, so nothing was
measured. Here a picture rendered from the TRUE sizes must give them back from the guessed script.

Needs OpenCV and ifcopenshell (no open3d: nothing here ray casts), so it runs wherever Lite-STEP does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.survey import PerspectiveCamera  # noqa: E402
from tools.survey.measure import (  # noqa: E402
    MeasureError, corners, compile_script, feature_point, measure, measure_many, read_dimensions,
    with_dimensions,
)

SCRIPT = '''"""Two posts and a beam."""
from lite_step import compile_main, param
from lite_step.models import Project, Storey, Point, Column, Beam, Box, Material

SPAN = 3000
POST = param(150, "mm", "Post section")
HEIGHT = 2700
BEAM_H = 300
COLOUR_CODE = 7  # a number nothing geometric depends on


def generate_project():
    p = Project(name="frame")
    s = Storey(name="ground", elevation=0)
    p.add_storey(s)
    h = POST // 2
    for name, x in (("post_a", 0), ("post_b", SPAN)):
        c = Column(name=name, material=Material(key="Timber_C24"))
        c.add(Box(start=Point(x=x - h, y=-h, z=0), end=Point(x=x + h, y=h, z=HEIGHT)))
        s.add(c)
    b = Beam(name="beam", material=Material(key="Timber_C24"))
    b.add(Box(start=Point(x=-h, y=-h, z=HEIGHT), end=Point(x=SPAN + h, y=h, z=HEIGHT + BEAM_H)))
    s.add(b)
    return p


result = generate_project()

if __name__ == "__main__":
    compile_main()
'''

FEATURES = [
    "post_a:xmin,ymin,zmin", "post_a:xmax,ymin,zmin", "post_a:xmin,ymin,zmax", "post_a:xmax,ymax,zmin",
    "post_b:xmin,ymin,zmin", "post_b:xmax,ymin,zmin", "post_b:xmax,ymax,zmin", "post_b:xmax,ymin,zmax",
    "beam:xmin,ymin,zmax", "beam:xmax,ymin,zmax", "beam:xmax,ymax,zmax", "beam:xmin,ymax,zmin",
]


def test_the_scripts_numbers_are_its_dimensions_and_edits_touch_only_them():
    dims = {d.key: d for d in read_dimensions(SCRIPT + "POST_X = (-3000, 0, 3000)\n")}
    assert dims["SPAN"].value == 3000
    assert dims["POST"].value == 150  # param() counts by its value
    assert dims["POST_X[0]"].value == -3000
    assert dims["POST_X[2]"].value == 3000
    edited = with_dimensions(SCRIPT, read_dimensions(SCRIPT), {"SPAN": 4100.4, "POST": 290})
    assert "SPAN = 4100\n" in edited
    assert 'POST = param(290, "mm", "Post section")' in edited
    assert edited.replace("4100", "3000").replace("param(290", "param(150") == SCRIPT


def test_one_picture_of_the_true_frame_measures_the_guessed_script():
    # the truth: a 4.2 m span and 290 mm posts, seen by a camera standing off one corner
    truth = with_dimensions(SCRIPT, read_dimensions(SCRIPT), {"SPAN": 4200, "POST": 290})
    boxes = corners(compile_script(truth, ROOT), {"post_a", "post_b", "beam"})
    world = np.array([feature_point(boxes, f) for f in FEATURES])
    cam = PerspectiveCamera.look_at((-2.5, -6.0, 1.7), (2.0, 0.0, 1.6), 55.0, 1200, 900)
    px = cam.project(world)
    res = measure(SCRIPT, [{"px": p.tolist(), "feature": f} for p, f in zip(px, FEATURES)], ["HEIGHT"],
                  1200, 900, ROOT, keys=["SPAN", "POST", "BEAM_H", "COLOUR_CODE"])
    got = {r["key"]: r for r in res["dimensions"]}
    assert abs(got["SPAN"]["now"] - 4200) <= 30, got["SPAN"]
    assert abs(got["POST"]["now"] - 290) <= 10, got["POST"]
    assert abs(got["BEAM_H"]["now"] - 300) <= 10  # true value, seen through the beam's top corners
    assert got["COLOUR_CODE"] == {"key": "COLOUR_CODE", "was": 7.0, "now": 7.0, "sigma": 0.0, "seen": False,
                                  "fixed": False}  # nothing depends on it: kept, and said so
    assert got["HEIGHT"]["fixed"]
    assert got["HEIGHT"]["now"] == 2700
    assert res["rmsPx"] < 1.0
    edited = {d.key: d.value for d in read_dimensions(res["edited"])}
    assert edited["SPAN"] == got["SPAN"]["now"]
    assert edited["POST"] == got["POST"]["now"]
    assert edited["COLOUR_CODE"] == 7
    assert edited["HEIGHT"] == 2700


def test_it_refuses_what_cannot_give_a_measurement():
    script = SCRIPT
    pts = [{"px": [100.0 + i, 200.0 + 3 * i], "feature": f} for i, f in enumerate(FEATURES)]
    with pytest.raises(MeasureError, match="not a dimension"):
        measure(script, pts, ["NO_SUCH"], 1200, 900, ROOT)
    with pytest.raises(MeasureError, match="ELEMENT:xmin"):
        measure(script, [{"px": [1, 2], "feature": "post_a:left"}] * 6, ["HEIGHT"], 1200, 900, ROOT)
    with pytest.raises(MeasureError, match="no element named"):
        measure(script, [{"px": [1, 2], "feature": "roof:xmin,ymin,zmin"}] * 6, ["HEIGHT"], 1200, 900, ROOT)


# -- several pictures, dropped points, label checks (carport, 2026-11-26) -----------------------------------

THREE = SCRIPT.replace('(("post_a", 0), ("post_b", SPAN))', '(("post_a", 0), ("post_m", SPAN // 2), ("post_b", SPAN))')
FEET = [f"{p}:{x},{y},zmin" for p in ("post_a", "post_m", "post_b") for x, y in (("xmin", "ymin"), ("xmax", "ymin"))]
TOPS = [f"{p}:xmin,ymin,zmax" for p in ("post_a", "post_m", "post_b")]
THREE_FEATURES = FEET + TOPS + ["beam:xmin,ymin,zmax", "beam:xmax,ymin,zmax", "beam:xmax,ymax,zmax"]


def _picture(script: str, features: list[str], cam: PerspectiveCamera) -> list[dict]:
    els = {f.split(":", 1)[0] for f in features}
    boxes = corners(compile_script(script, ROOT), els)
    px = cam.project(np.array([feature_point(boxes, f) for f in features]))
    return [{"px": p.tolist(), "feature": f} for p, f in zip(px, features)]


def test_two_pictures_of_one_scene_give_one_set_of_sizes_and_a_camera_each():
    truth = with_dimensions(THREE, read_dimensions(THREE), {"SPAN": 4200, "POST": 290})
    left = PerspectiveCamera.look_at((-2.5, -6.0, 1.7), (2.0, 0.0, 1.6), 55.0, 1200, 900)
    right = PerspectiveCamera.look_at((6.5, -5.0, 1.5), (2.0, 0.0, 1.4), 60.0, 900, 1200)
    res = measure_many(THREE, [{"name": "left", "points": _picture(truth, THREE_FEATURES, left), "w": 1200, "h": 900},
                               {"name": "right", "points": _picture(truth, THREE_FEATURES, right), "w": 900,
                                "h": 1200}], ["HEIGHT"], ROOT, keys=["SPAN", "POST", "BEAM_H"])
    got = {r["key"]: r for r in res["dimensions"]}
    assert abs(got["SPAN"]["now"] - 4200) <= 30 and abs(got["POST"]["now"] - 290) <= 10
    assert [p["name"] for p in res["pictures"]] == ["left", "right"]
    assert all(p["rmsPx"] < 1.0 and p["verdict"] == "ok" and not p["dropped"] for p in res["pictures"])
    assert res["verdict"] == "ok"
    eyes = [p["camera"]["viewPoint"] for p in res["pictures"]]
    assert abs(eyes[0][0] - -2.5) < 0.2 and abs(eyes[1][0] - 6.5) < 0.2  # each picture keeps its own camera


def test_a_point_far_off_the_rest_is_dropped_and_reported_and_the_fit_holds():
    truth = with_dimensions(THREE, read_dimensions(THREE), {"SPAN": 4200, "POST": 290})
    cam = PerspectiveCamera.look_at((-2.5, -6.0, 1.7), (2.0, 0.0, 1.6), 55.0, 1200, 900)
    pts = _picture(truth, THREE_FEATURES, cam)
    pts[3]["px"] = [pts[3]["px"][0], pts[3]["px"][1] - 80.0]  # a foot read at the edge of what hides it
    res = measure(THREE, pts, ["HEIGHT"], 1200, 900, ROOT, keys=["SPAN", "POST", "BEAM_H"])
    assert [d["index"] for d in res["dropped"]] == [3]
    assert res["residualPx"][3] > 40  # still reported, under the final fit: the reason it went
    assert res["rmsPx"] < 1.0 and res["verdict"] == "ok"
    assert abs({r["key"]: r for r in res["dimensions"]}["SPAN"]["now"] - 4200) <= 30
    kept = measure(THREE, pts, ["HEIGHT"], 1200, 900, ROOT, keys=["SPAN", "POST", "BEAM_H"], drop=False)
    assert kept["dropped"] == [] and kept["rmsPx"] > 5  # and without dropping, the error shows


def test_labels_one_post_off_fail_the_line_check_and_the_picture_is_not_trusted():
    truth = with_dimensions(THREE, read_dimensions(THREE), {"SPAN": 4200, "POST": 290})
    cam = PerspectiveCamera.look_at((-2.5, -6.0, 1.7), (2.0, 0.0, 1.6), 55.0, 1200, 900)
    pts = _picture(truth, THREE_FEATURES, cam)
    # the middle post's feet named as the first post's and the reverse: every pixel right, two labels wrong
    for p in pts:
        el, rest = p["feature"].split(":", 1)
        if rest.endswith("zmin") and el in ("post_a", "post_m"):
            p["feature"] = {"post_a": "post_m", "post_m": "post_a"}[el] + ":" + rest
    res = measure(THREE, pts, ["HEIGHT"], 1200, 900, ROOT, keys=["SPAN", "POST", "BEAM_H"])
    assert res["labelChecks"], "the feet are no longer on one line in label order"
    assert res["verdict"] == "labels_suspect"
    named = {f.split(":", 1)[0] for c in res["labelChecks"] for f in c["features"]}
    assert {"post_a", "post_m"} <= named
    good = measure(THREE, _picture(truth, THREE_FEATURES, cam), ["HEIGHT"], 1200, 900, ROOT,
                   keys=["SPAN", "POST", "BEAM_H"])
    assert good["labelChecks"] == []  # a line the structure makes, not one the guessed sizes make by chance
