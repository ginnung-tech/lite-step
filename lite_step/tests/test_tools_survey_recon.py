"""tools/survey cameras from a reconstruction, and the view check on drawings.

A reconstruction is made here from known cameras, then moved into a frame of its own by a known similarity
(rotated, shifted, scaled 1:37 the way COLMAP's arbitrary units are): the import must bring every camera back to
where it was, and must refuse to guess when it is not told how.
"""

from __future__ import annotations

import json
import math
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

from tools.survey import OrthoCamera, PerspectiveCamera  # noqa: E402
from tools.survey.camera import basis  # noqa: E402
from tools.survey.drone import Drone  # noqa: E402
from tools.survey.recon import ReconError, import_frames, read_colmap, read_transforms, umeyama  # noqa: E402
from tools.survey.scene import Scene  # noqa: E402
from tools.survey.shape import view_check  # noqa: E402

NOW = "2026-11-28T18:00:00Z"
W, H = 320, 240


def box(x0, y0, z0, x1, y1, z1):
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float32)
    f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4],
         [1, 5, 7], [1, 7, 3]]
    return v, np.array(f, dtype=np.int32)


# an L: a tall block with a wing to the east, so every side looks different
SCENE = Scene.from_meshes([("G-T", "wall:tower", "IfcWall", *box(-2, -1, 0, 2, 1, 12)),
                           ("G-W", "wall:wing", "IfcWall", *box(2, -1, 0, 9, 1, 4))])

# the reconstruction's own frame: world = S * R @ recon + T
S = 37.0
ANG = math.radians(33.0)
R = np.array([[math.cos(ANG), -math.sin(ANG), 0.0], [math.sin(ANG), math.cos(ANG), 0.0], [0.0, 0.0, 1.0]]) @ \
    np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # also tipped over: recon "up" is not Z
T = np.array([120.0, -45.0, 7.0])


def to_recon(p):
    return (np.asarray(p, float) - T) @ R / S


def _rot_to_quat(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return tuple(q)


CAMS = [PerspectiveCamera.look_at(eye, (2.0, 0.0, 4.0), 50.0, W, H)
        for eye in ((2, -30, 8), (30, -18, 10), (-25, 15, 6), (5, 32, 14))]


def _reconstruction(tmp_path):
    """images/ rendered from CAMS, and a COLMAP text model of them in the reconstruction's frame."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    drone = Drone(SCENE)
    imgs, cams_txt = [], ["1 PINHOLE %d %d %.6f %.6f %.1f %.1f" % (W, H, CAMS[0].focal_px, CAMS[0].focal_px, W / 2, H / 2)]
    for i, cam in enumerate(CAMS):
        hit = drone.render(cam).hit
        img = np.full((H, W, 3), 235, np.uint8)
        img[hit] = (60, 60, 60)
        cv2.imwrite(str(img_dir / f"f{i:03d}.png"), img)
        right, up, fwd = basis(cam.direction, cam.up)
        c2w_world = np.stack([right, -up, fwd], axis=1)
        c2w = R.T @ c2w_world
        w2c = c2w.T
        centre = to_recon(np.asarray(cam.position))
        t = -w2c @ centre
        qw, qx, qy, qz = _rot_to_quat(w2c)
        imgs += [f"{i + 1} {qw:.12f} {qx:.12f} {qy:.12f} {qz:.12f} {t[0]:.12f} {t[1]:.12f} {t[2]:.12f} 1 f{i:03d}.png", ""]
    model = tmp_path / "sparse"
    model.mkdir()
    (model / "cameras.txt").write_text("\n".join(cams_txt) + "\n", encoding="utf-8")
    (model / "images.txt").write_text("# header\n" + "\n".join(imgs) + "\n", encoding="utf-8")
    return model


def _points(world):
    return {"points": [{"recon": to_recon(w).tolist(), "world": list(map(float, w))} for w in world]}


CORNERS = [(-2, -1, 0), (9, 1, 0), (2, -1, 12), (-2, 1, 12), (9, -1, 4)]


def test_umeyama_recovers_a_known_similarity():
    src = np.array([to_recon(c) for c in CORNERS])
    s, rot, t = umeyama(src, np.array(CORNERS, float))
    assert s == pytest.approx(S, rel=1e-9) and np.allclose(rot, R, atol=1e-9) and np.allclose(t, T, atol=1e-7)


def test_a_colmap_reconstruction_aligned_by_known_points_puts_every_camera_back(tmp_path):
    model = _reconstruction(tmp_path)
    r = import_frames(tmp_path, read_colmap(model), "images", _points(CORNERS), now=NOW, scene=SCENE)
    assert r["alignment"]["gate"] == "passed" and r["alignment"]["scale"] == pytest.approx(S, rel=1e-6)
    cams = json.loads((tmp_path / "survey" / "cameras.json").read_text())["cameras"]
    assert len(cams) == 4
    for entry, want in zip(sorted(cams, key=lambda c: c["source"]), CAMS):
        got = PerspectiveCamera.from_contract(entry["camera"])
        assert np.allclose(got.position, want.position, atol=1e-3)
        assert np.allclose(got.direction, want.direction, atol=1e-6) and got.fov_v_deg == pytest.approx(50.0, abs=1e-6)
        assert entry["agreement"]["outlineWithinPx"] > 0.95  # the model lies on its own picture


def test_without_an_alignment_the_import_refuses_and_says_why(tmp_path):
    model = _reconstruction(tmp_path)
    with pytest.raises(ReconError, match="similarity transform"):
        import_frames(tmp_path, read_colmap(model), "images", None, now=NOW)
    with pytest.raises(ReconError, match="reason"):
        import_frames(tmp_path, read_colmap(model), "images", {"identity": True}, now=NOW)
    with pytest.raises(ReconError, match="three or more"):
        import_frames(tmp_path, read_colmap(model), "images", _points(CORNERS[:2]), now=NOW)


def test_a_wrong_point_pair_fails_the_alignment_and_is_named(tmp_path):
    model = _reconstruction(tmp_path)
    al = _points(CORNERS)
    al["points"][3]["world"] = [9.0, 1.0, 4.0]  # the wing's far top corner given for the tower's top corner
    r = import_frames(tmp_path, read_colmap(model), "images", al, now=NOW)
    assert r["alignment"]["gate"] == "failed" and "point" in r["alignment"]["reason"]
    assert all(f["fit"] == "failed" for f in r["frames"])


def test_a_transforms_json_in_the_opengl_convention_reads_the_same_cameras(tmp_path):
    _reconstruction(tmp_path)
    frames = []
    for i, cam in enumerate(CAMS):
        right, up, fwd = basis(cam.direction, cam.up)
        c2w = np.eye(4)
        c2w[:3, :3] = R.T @ np.stack([right, up, -fwd], axis=1)  # OpenGL: y up, looking down -z
        c2w[:3, 3] = to_recon(np.asarray(cam.position))
        frames.append({"file_path": f"images/f{i:03d}.png", "transform_matrix": c2w.tolist()})
    tj = tmp_path / "transforms.json"
    tj.write_text(json.dumps({"fl_x": CAMS[0].focal_px, "fl_y": CAMS[0].focal_px, "cx": W / 2, "cy": H / 2, "w": W,
                              "h": H, "frames": frames}), encoding="utf-8")
    r = import_frames(tmp_path, read_transforms(tj), "images", _points(CORNERS), now=NOW, n=2)
    assert len(r["frames"]) == 2
    entry = json.loads((tmp_path / "survey" / "cameras.json").read_text())["cameras"][0]
    got = PerspectiveCamera.from_contract(entry["camera"])
    want = next(c for c in CAMS if np.allclose(c.position, got.position, atol=1e-3))
    assert np.allclose(got.direction, want.direction, atol=1e-6)


# -- the view check --------------------------------------------------------------------------------------------


def test_a_drawing_fitted_to_the_wrong_side_of_the_model_is_flagged_and_the_right_side_is_not():
    south = OrthoCamera((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 15.0, 1.0, 130.0, 220.0, W, H)
    drone = Drone(SCENE)
    picture = drone.render(south).hit  # the true south elevation, as a drawing would show it
    labels = [e.label for e in SCENE.elements]
    from tools.survey.shape import compare

    ok = view_check(picture, drone, south, labels, compare(picture, *_frame(drone, south), south, labels)["iou"])
    assert "warning" not in ok
    north = OrthoCamera((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), 15.0, 1.0, 130.0, 220.0, W, H)  # the title said north
    iou = compare(picture, *_frame(drone, north), north, labels)["iou"]
    bad = view_check(picture, drone, north, labels, iou)
    assert "warning" in bad and "--view north is likely the wrong view" in bad["warning"]
    assert max(bad["alternatives"], key=lambda a: a["iou"])["view"] == "south"


def _frame(drone, cam):
    f = drone.render(cam)
    return f.element, f.depth
