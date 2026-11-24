"""tools/survey: cameras, fitting, ink and overlays, each checked against a known answer.

The library lives in tools/ (not the package), so the repo root goes on the path; OpenCV is optional for
Lite-STEP itself, so these tests skip without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
cv2 = pytest.importorskip("cv2")

from tools.survey import (  # noqa: E402
    OrthoCamera, PerspectiveCamera, agreement, background_mask, camera_from_contract, chamfer, draw_polyline,
    fit_ortho, fit_perspective, ink_mask, refine, silhouette_profile,
)

NORTH, Z = (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


# -- cameras -------------------------------------------------------------------------------------------

def test_ortho_projects_with_separate_horizontal_stretch_and_inverts():
    cam = OrthoCamera(NORTH, Z, px_per_m=40.0, k=0.9, ax_px=500.0, c_px=900.0, image_w=1000, image_h=1000)
    px = cam.project([[10.0, 5.0, 2.0]])  # looking north with Z up: u = x, v = z, y is depth
    assert px[0] == pytest.approx([500 + 0.9 * 40 * 10, 900 - 40 * 2])
    back = cam.to_world(px, depth_m=5.0)
    assert back[0] == pytest.approx([10.0, 5.0, 2.0])


def test_cameras_round_trip_through_the_contract_shapes():
    o = OrthoCamera(NORTH, Z, 41.79, 0.905, 2353.2, 2799.8, 4000, 5000, depth_range_m=(-1.0, 3.0))
    assert camera_from_contract(o.to_contract()) == o
    p = PerspectiveCamera.look_at((80, 150, 42), (40, 45, 10), 30.0, 807, 1162, roll_deg=1.8)
    q = camera_from_contract(p.to_contract())
    pts = np.array([[50.0, 50.0, 0.0], [37.0, 62.0, -2.5], [60.0, 40.0, 30.0]])
    assert np.allclose(p.project(pts), q.project(pts), atol=1e-6)


def test_perspective_puts_points_behind_the_eye_out_of_the_picture():
    cam = PerspectiveCamera.look_at((0, -10, 0), (0, 0, 0), 60.0, 400, 300)
    px = cam.project([[0, 0, 0], [0, -20, 0]])
    assert px[0] == pytest.approx([200, 150])
    assert np.isnan(px[1]).all()


# -- fitting --------------------------------------------------------------------------------------------

def test_ortho_fit_recovers_scale_stretch_and_offsets_from_control_points():
    truth = OrthoCamera(NORTH, Z, 41.79, 0.905, 2353.2, 2799.8, 4000, 5000)
    world = np.array([[-37.3, 57.2, 2.0], [-20.0, 50.0, 30.0], [0.0, 40.0, 37.45]])
    fit = fit_ortho(world, truth.project(world), NORTH, Z, 4000, 5000)
    c = fit.camera
    assert (c.px_per_m, c.k, c.ax_px, c.c_px) == pytest.approx((41.79, 0.905, 2353.2, 2799.8), rel=1e-9)
    assert fit.residual_px < 1e-6


def test_ortho_fit_with_a_shared_scale_keeps_it_and_absorbs_the_error_in_the_offsets():
    world = np.array([[0.0, 0, 0.0], [10.0, 0, 10.0]])
    px = np.array([[100.0, 500.0], [520.0, 100.0]])  # s 40, k 1.05: a plate pasted wide
    fit = fit_ortho(world, px, NORTH, Z, 800, 600, px_per_m=40.0, k=1.0)
    assert (fit.camera.px_per_m, fit.camera.k) == (40.0, 1.0)
    assert fit.residual_px > 5  # the stretch the pinned k refuses shows up as residual, never hidden


def test_ortho_fit_refuses_one_point_and_a_mirrored_pairing():
    with pytest.raises(ValueError):
        fit_ortho([[0, 0, 0]], [[0, 0]], NORTH, Z, 10, 10)
    with pytest.raises(ValueError, match="mirrored"):
        fit_ortho([[0, 0, 0], [10, 0, 10]], [[500, 100], [100, 500]], NORTH, Z, 800, 600)


def test_perspective_fit_recovers_a_rolled_camera_from_corners():
    truth = PerspectiveCamera.look_at((79.7, 150.1, 42.1), (43.4, 45.6, 12.0), 30.0, 807, 1162, roll_deg=1.8)
    # the v12 corners: a plinth (top and bottom) and a pier's base frame
    world = np.array([[62.5, 37.17, -0.4], [62.5, 62.5, -0.4], [37.17, 62.5, -0.4], [62.5, 37.17, -2.5],
                      [62.5, 62.5, -2.5], [37.17, 62.5, -2.5], [58.18, 58.18, 0.0], [58.18, 43.18, 0.0],
                      [43.18, 58.18, 0.0], [50.0, 50.0, 30.0]])
    fit = fit_perspective(world, truth.project(world), 807, 1162, fov_range=(20.0, 45.0, 0.25))
    assert fit.residual_px < 0.5
    assert fit.camera.fov_v_deg == pytest.approx(30.0, abs=0.3)
    assert np.linalg.norm(np.subtract(fit.camera.position, truth.position)) < 1.5


def test_perspective_fit_needs_four_points():
    with pytest.raises(ValueError):
        fit_perspective([[0, 0, 0]] * 3, [[0, 0]] * 3, 100, 100)


def test_refine_finds_the_minimum_of_a_cost():
    p, c = refine([0.0, 0.0], lambda q: (q[0] - 1.3) ** 2 + (q[1] + 0.7) ** 2, steps=[1.0, 1.0], rounds=12)
    assert p == pytest.approx([1.3, -0.7], abs=0.02)
    assert c < 1e-3


def test_chamfer_is_zero_for_identical_masks_and_grows_with_a_shift():
    a = np.zeros((100, 100), bool)
    a[20:80, 48:52] = True
    b = np.roll(a, 6, axis=1)
    assert chamfer(a, a, 10) == 0.0
    assert 0.5 < chamfer(a, b, 10) < 2.0
    assert chamfer(a, np.zeros_like(a), 10) == 2.0


# -- ink -------------------------------------------------------------------------------------------------

def test_ink_mask_drops_short_strokes_and_closes_dotted_members():
    g = np.full((200, 200), 255, np.uint8)
    cv2.line(g, (10, 100), (190, 100), 0, 3)  # a member
    cv2.putText(g, "8", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)  # a label
    for x in range(20, 180, 6):  # a member drawn as rivet dots
        cv2.circle(g, (x, 160), 2, 0, -1)
    plain = ink_mask(g, min_stroke_px=40)
    assert plain[100, 100] and not plain[30:45, 18:35].any()
    assert not plain[160, 23]  # each dot is a short stroke on its own
    closed = ink_mask(g, min_stroke_px=40, close_px=3)
    assert closed[160, 20:178].mean() > 0.9


def test_silhouette_profile_reads_a_drawn_pedestal_in_metres():
    cam = OrthoCamera(NORTH, Z, px_per_m=40.0, k=0.9, ax_px=400.0, c_px=400.0, image_w=800, image_h=800)
    g = np.full((800, 800), 255, np.uint8)
    # a 1.5 m wide capital from z 1.0 to 1.75 centred on u = -3.0, outlined
    (x0, y1), (x1, y0) = cam.project([[-3.75, 0, 1.0], [-2.25, 0, 1.75]])
    cv2.rectangle(g, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 0, 2)
    rows = silhouette_profile(g < 128, cam, (-5.0, -1.0), (1.1, 1.65), step_m=0.1)
    widths = [r.width for r in rows if r.width is not None]
    assert len(widths) == len(rows)
    assert np.median(widths) == pytest.approx(1.5, abs=2 / (cam.k * cam.px_per_m))
    assert np.median([r.centre for r in rows]) == pytest.approx(-3.0, abs=0.05)


# -- overlays ---------------------------------------------------------------------------------------------

def test_agreement_counts_both_sides_and_colours_them():
    a = np.zeros((50, 50), bool)
    a[10:40, 10:20] = True
    same = agreement(a, a)
    assert (same.model_within_px, same.source_within_px) == (1.0, 1.0)
    far = np.roll(a, 25, axis=1)
    split = agreement(a, far)
    assert split.model_within_px == 0.0 and split.source_within_px == 0.0
    assert tuple(split.image[20, 15]) == (220, 120, 40) and tuple(split.image[20, 40]) == (40, 40, 220)


def test_background_mask_and_polyline_draw_where_the_camera_says():
    img = np.full((100, 100, 3), (38, 35, 34), np.uint8)
    img[40:60, 40:60] = (60, 75, 94)
    m = background_mask(img)
    assert m[50, 50] and not m[5, 5]
    cam = OrthoCamera(NORTH, Z, 10.0, 1.0, 50.0, 50.0, 100, 100)
    draw_polyline(img, cam, [[-2, 0, 0], [2, 0, 0]], colour=(0, 0, 255), thickness=1)
    b, g, r = (int(x) for x in img[50, 45])  # anti-aliased: red dominates on the line
    assert r > 200 and b < 40 and g < 40
    assert tuple(img[70, 45]) == (38, 35, 34)  # and nowhere else
