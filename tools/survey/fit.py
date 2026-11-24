"""Fitting a camera to a source: from control points first, then refined against the model.

Control points come first. On the v12 screenshot (Eiffel run, 2026-11-24) a silhouette-only search fell
into a wrong pose with the plinth turned; nine corners solved by PnP over the field of view (3 px), then
a silhouette refinement with roll free (94 % of model pixels within 2 px).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .camera import OrthoCamera, PerspectiveCamera, basis


@dataclass(frozen=True)
class Fit:
    camera: OrthoCamera | PerspectiveCamera
    residual_px: float  # mean distance between the control points and their projections
    per_point_px: tuple[float, ...]


def _residuals(cam, world, px) -> tuple[float, tuple[float, ...]]:
    d = np.linalg.norm(cam.project(world) - np.asarray(px, dtype=np.float64), axis=1)
    return float(np.mean(d)), tuple(float(x) for x in d)


def fit_ortho(world, px, direction, up, image_w: int, image_h: int,
              px_per_m: float | None = None, k: float | None = None) -> Fit:
    """Least squares for ``px = ax + k*s*u`` and ``py = c - s*v``: the two axes are independent.

    Two points are enough. Pass `px_per_m` and `k` to pin them, as for plates of one printed scale or
    sub-figures of one plate, which must share them: fitted alone, a model error leaks into the camera.
    """
    world = np.asarray(world, dtype=np.float64)
    px = np.asarray(px, dtype=np.float64)
    if len(world) < 2 or len(world) != len(px):
        raise ValueError("an ortho fit needs at least two control points, world and pixel paired")
    right, up_v, _ = basis(direction, up)
    u, v = world @ right, world @ up_v
    if px_per_m is None:
        a = np.stack([-v, np.ones_like(v)], axis=1)  # py = -s*v + c
        (s, c), *_ = np.linalg.lstsq(a, px[:, 1], rcond=None)
    else:
        s = px_per_m
        c = float(np.mean(px[:, 1] + s * v))
    if k is None:
        a = np.stack([u, np.ones_like(u)], axis=1)  # px = (k*s)*u + ax
        (ks, ax), *_ = np.linalg.lstsq(a, px[:, 0], rcond=None)
        k_fit = ks / s
    else:
        k_fit = k
        ax = float(np.mean(px[:, 0] - k * s * u))
    if s <= 0 or k_fit <= 0:
        raise ValueError(f"fit gave a mirrored or empty scale (s={s:.4g}, k={k_fit:.4g}): check the point pairs")
    cam = OrthoCamera(tuple(direction), tuple(up), float(s), float(k_fit), float(ax), float(c), image_w, image_h)
    return Fit(cam, *_residuals(cam, world, px))


def fit_perspective(world, px, image_w: int, image_h: int,
                    fov_range: tuple[float, float, float] = (10.0, 100.0, 0.5)) -> Fit:
    """PnP (OpenCV SQPnP) at each vertical field of view in `fov_range`; the best one wins.

    Four points at least; spread them in depth where you can (points on one plane pin the focal length
    poorly). Roll comes out of the solve; refine afterwards with `refine` if the image says it is level.
    """
    import cv2  # optional dependency: tools/survey/requirements.txt

    world = np.asarray(world, dtype=np.float64)
    px = np.asarray(px, dtype=np.float64)
    if len(world) < 4 or len(world) != len(px):
        raise ValueError("a perspective fit needs at least four control points, world and pixel paired")
    best: tuple[float, PerspectiveCamera] | None = None
    for fov in np.arange(*fov_range):
        f = (image_h / 2.0) / np.tan(np.radians(fov) / 2.0)
        kmat = np.array([[f, 0, image_w / 2.0], [0, f, image_h / 2.0], [0, 0, 1]])
        ok, r, t = cv2.solvePnP(world, px, kmat, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        rot, _ = cv2.Rodrigues(r)
        eye = (-rot.T @ t).ravel()
        fwd = rot.T @ np.array([0.0, 0.0, 1.0])
        up = -(rot.T @ np.array([0.0, 1.0, 0.0]))
        cam = PerspectiveCamera(tuple(eye.tolist()), tuple(fwd.tolist()), tuple(up.tolist()), float(fov), image_w, image_h)
        err, _ = _residuals(cam, world, px)
        if np.isfinite(err) and (best is None or err < best[0]):
            best = (err, cam)
    if best is None:
        raise ValueError("no field of view gave a pose: check the point pairs")
    return Fit(best[1], *_residuals(best[1], world, px))


def refine(params: Sequence[float], cost: Callable[[list[float]], float], steps: Sequence[float],
           rounds: int = 6, shrink: float = 0.6) -> tuple[list[float], float]:
    """Coordinate descent: nudge each parameter by +-step, keep any improvement, shrink the steps.

    Parametrise about the points' centre (not the image origin): a scale step about a far origin moves
    every point a long way, so the scale never improves (the Eiffel run's registration did exactly that).
    """
    p = list(map(float, params))
    best = cost(p)
    st = list(map(float, steps))
    for _ in range(rounds):
        for i, s in enumerate(st):
            for d in (s, -s):
                q = list(p)
                q[i] += d
                c = cost(q)
                if c < best:
                    p, best = q, c
        st = [x * shrink for x in st]
    return p, best


def chamfer(mask_ref: np.ndarray, mask_model: np.ndarray, tau_px: float) -> float:
    """Symmetric truncated chamfer between two boolean masks, divided by `tau_px` (0 = identical, 2 = apart).

    Model pixels far from the reference, plus reference pixels far from the model, each capped at `tau_px`
    so a missing part costs a fixed amount instead of dominating.
    """
    import cv2

    if not mask_model.any() or not mask_ref.any():
        return 2.0
    dt_ref = cv2.distanceTransform((~mask_ref).astype(np.uint8), cv2.DIST_L2, 3)
    dt_mod = cv2.distanceTransform((~mask_model).astype(np.uint8), cv2.DIST_L2, 3)
    return float(np.minimum(dt_ref[mask_model], tau_px).mean() + np.minimum(dt_mod[mask_ref], tau_px).mean()) / tau_px
