"""tools/survey depth: what an outline cannot see, measured against a reconstruction's depth map.

The "reconstruction" is the model itself with one change (a wall set back, a balcony pulled out), rendered through
a known camera and stored in reconstruction units (1:37), the way COLMAP writes it. The changed element must be
found with its millimetres and its sign; the unchanged ones must stay quiet.
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
from tools.survey.camera import basis  # noqa: E402
from tools.survey.depth import read_depth_map, run_depth  # noqa: E402
from tools.survey.drone import Drone, rays  # noqa: E402
from tools.survey.intake import _record, add_source  # noqa: E402
from tools.survey.scene import Scene  # noqa: E402

NOW = "2026-11-28T20:00:00Z"
W, H = 320, 240
S = 37.0  # metres per reconstruction unit


def box(x0, y0, z0, x1, y1, z1):
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float32)
    f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return v, np.array(f, dtype=np.int32)


def facade(balcony_y=-1.5):
    return Scene.from_meshes([("G-F", "wall:facade", "IfcWall", *box(-6, 0, 0, 6, 0.4, 9)),
                              ("G-B", "slab:balcony", "IfcSlab", *box(-2, balcony_y, 3, 2, 0, 3.4)),
                              ("G-R", "wall:return", "IfcWall", *box(-2, balcony_y, 3.4, -1.8, 0, 4.4))])


CAM = PerspectiveCamera.look_at((0.0, -20.0, 6.0), (0.0, 0.0, 3.5), 50.0, W, H)


def measured_z(scene) -> np.ndarray:
    f = Drone(scene).render(CAM)
    _, _, fwd = basis(CAM.direction, CAM.up)
    z = f.depth * (rays(CAM)[..., 3:].astype(np.float64) @ fwd)
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32) / S  # reconstruction units; 0 = no measurement


def colmap_bin(path: Path, a: np.ndarray) -> None:
    h, w = a.shape
    path.write_bytes(f"{w}&{h}&1&".encode() + np.asarray(a, "<f4").reshape(h, w, 1).transpose(1, 0, 2).tobytes(order="F"))


def survey(tmp_path, built, truth, fmt="npy"):
    """The model `built` written as output.ifc stand-in (a cached scene), one frame with an aligned camera, and the
    depth map of `truth`."""
    img = np.full((H, W, 3), 200, np.uint8)
    pic = tmp_path / "frame.png"
    cv2.imwrite(str(pic), img)
    src = add_source(tmp_path, pic, kind="photo", title="frame", now=NOW)
    _record(tmp_path, {"source": src["id"], "version": 1, "camera": CAM.to_contract(), "method": "reconstruction",
                       "residualMm": 0.0, "gate": "passed", "frozenAt": NOW, "reason": "test",
                       "alignment": {"method": "points", "scale": S}})
    z = measured_z(truth)
    if fmt == "npy":
        np.save(tmp_path / "depth.npy", z)
        name = "depth.npy"
    else:
        colmap_bin(tmp_path / "depth.geometric.bin", z)
        name = "depth.geometric.bin"
    return src["id"], name


@pytest.fixture
def model_is(monkeypatch):
    """Point the scene loader at an in-memory scene instead of an IFC file."""
    def use(scene):
        monkeypatch.setattr(Scene, "from_ifc", classmethod(lambda cls, *a, **k: scene))
        return scene
    return use


def test_a_colmap_depth_map_reads_back_the_array_written(tmp_path):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    colmap_bin(tmp_path / "d.bin", a)
    assert np.array_equal(read_depth_map(tmp_path / "d.bin"), a)


def test_the_model_matching_the_measurement_has_no_findings(tmp_path, model_is):
    model_is(facade())
    sid, dm = survey(tmp_path, facade(), facade())
    r = run_depth(tmp_path, sid, dm)
    assert r["findings"] == [] and abs(r["medianMm"]) < 5 and r["scale"] == S


def test_a_balcony_built_too_deep_stands_in_front_and_is_named_with_its_millimetres(tmp_path, model_is):
    model_is(facade(balcony_y=-2.0))  # built 2.0 m deep
    sid, dm = survey(tmp_path, facade(balcony_y=-2.0), facade(balcony_y=-1.5), fmt="bin")  # measured 1.5 m
    r = run_depth(tmp_path, sid, dm)
    named = {f["element"]: f for f in r["findings"]}
    assert named["slab:balcony"]["medianMm"] == pytest.approx(-500, abs=30)  # stands 0.5 m in front
    assert "wall:return" in named and named["wall:return"]["medianMm"] < -100
    assert "wall:facade" not in named  # the wall behind it is where it was measured
    assert json.loads((tmp_path / "survey" / "depth" / f"{sid}-v1.json").read_text())["findings"]


def test_without_the_reconstructions_scale_it_refuses(tmp_path, model_is):
    model_is(facade())
    sid, dm = survey(tmp_path, facade(), facade())
    cams = json.loads((tmp_path / "survey" / "cameras.json").read_text())
    del cams["cameras"][0]["alignment"]
    (tmp_path / "survey" / "cameras.json").write_text(json.dumps(cams))
    with pytest.raises(ValueError, match="scale"):
        run_depth(tmp_path, sid, dm)
    assert run_depth(tmp_path, sid, dm, scale=S)["findings"] == []
