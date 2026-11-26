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
    MeasureError, corners, compile_script, feature_point, measure, read_dimensions, with_dimensions,
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
    assert dims["SPAN"].value == 3000 and dims["POST"].value == 150  # param() counts by its value
    assert dims["POST_X[0]"].value == -3000 and dims["POST_X[2]"].value == 3000
    edited = with_dimensions(SCRIPT, read_dimensions(SCRIPT), {"SPAN": 4100.4, "POST": 290})
    assert "SPAN = 4100\n" in edited and 'POST = param(290, "mm", "Post section")' in edited
    assert edited.replace("4100", "3000").replace("param(290", "param(150") == SCRIPT


def test_one_picture_of_the_true_frame_measures_the_guessed_script(tmp_path):
    # the truth: a 4.2 m span and 290 mm posts, seen by a camera standing off one corner
    truth = with_dimensions(SCRIPT, read_dimensions(SCRIPT), {"SPAN": 4200, "POST": 290})
    boxes = corners(compile_script(truth, ROOT), {"post_a", "post_b", "beam"})
    world = np.array([feature_point(boxes, f) for f in FEATURES])
    cam = PerspectiveCamera.look_at((-2.5, -6.0, 1.7), (2.0, 0.0, 1.6), 55.0, 1200, 900)
    px = cam.project(world)
    script = tmp_path / "model.py"
    script.write_text(SCRIPT, encoding="utf-8")

    res = measure(script, [{"px": p.tolist(), "feature": f} for p, f in zip(px, FEATURES)], ["HEIGHT"],
                  1200, 900, ROOT, keys=["SPAN", "POST", "BEAM_H", "COLOUR_CODE"])
    got = {r["key"]: r for r in res["dimensions"]}
    assert abs(got["SPAN"]["now"] - 4200) <= 30, got["SPAN"]
    assert abs(got["POST"]["now"] - 290) <= 10, got["POST"]
    assert abs(got["BEAM_H"]["now"] - 300) <= 10  # true value, seen through the beam's top corners
    assert got["COLOUR_CODE"] == {"key": "COLOUR_CODE", "was": 7.0, "now": 7.0, "sigma": 0.0, "seen": False,
                                  "fixed": False}  # nothing depends on it: kept, and said so
    assert got["HEIGHT"]["fixed"] and got["HEIGHT"]["now"] == 2700
    assert res["rmsPx"] < 1.0
    edited = {d.key: d.value for d in read_dimensions(res["edited"])}
    assert edited["SPAN"] == got["SPAN"]["now"] and edited["POST"] == got["POST"]["now"]
    assert edited["COLOUR_CODE"] == 7 and edited["HEIGHT"] == 2700


def test_it_refuses_what_cannot_give_a_measurement(tmp_path):
    script = tmp_path / "model.py"
    script.write_text(SCRIPT, encoding="utf-8")
    pts = [{"px": [100.0 + i, 200.0 + 3 * i], "feature": f} for i, f in enumerate(FEATURES)]
    with pytest.raises(MeasureError, match="not a dimension"):
        measure(script, pts, ["NO_SUCH"], 1200, 900, ROOT)
    with pytest.raises(MeasureError, match="ELEMENT:xmin"):
        measure(script, [{"px": [1, 2], "feature": "post_a:left"}] * 6, ["HEIGHT"], 1200, 900, ROOT)
    with pytest.raises(MeasureError, match="no element named"):
        measure(script, [{"px": [1, 2], "feature": "roof:xmin,ymin,zmin"}] * 6, ["HEIGHT"], 1200, 900, ROOT)
