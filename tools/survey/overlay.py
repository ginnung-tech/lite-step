"""Pictures for people: the model drawn on their source, and where the two agree.

These are what a person is shown when they give a picture (the Surveyor's duty to fit every picture and
show it back): their image with the model's lines on it, and the agreement map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOTH = (80, 170, 60)  # BGR: green, model and source agree
MODEL_ONLY = (40, 40, 220)  # red
SOURCE_ONLY = (220, 120, 40)  # blue


def draw_polyline(img: np.ndarray, cam, world_points, colour=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    """Draw world points (metres) as one polyline through `cam` onto `img` (BGR), in place; returns img."""
    import cv2

    px = cam.project(world_points)
    ok = np.all(np.isfinite(px), axis=1)
    if ok.sum() >= 2:
        cv2.polylines(img, [np.round(px[ok]).astype(np.int32)], False, colour, thickness, cv2.LINE_AA)
    return img


@dataclass(frozen=True)
class Agreement:
    image: np.ndarray  # BGR, white background
    model_within_px: float  # share of model pixels within `tol_px` of the source's structure
    source_within_px: float  # share of source structure within `tol_px` of the model


def agreement(source_mask: np.ndarray, model_mask: np.ndarray, tol_px: float = 2.0) -> Agreement:
    """Green where both, red model only, blue source only; and the two shares within `tol_px`."""
    import cv2

    img = np.full((*source_mask.shape, 3), 255, dtype=np.uint8)
    img[source_mask & model_mask] = BOTH
    img[model_mask & ~source_mask] = MODEL_ONLY
    img[source_mask & ~model_mask] = SOURCE_ONLY
    dt_src = cv2.distanceTransform((~source_mask).astype(np.uint8), cv2.DIST_L2, 3)
    dt_mod = cv2.distanceTransform((~model_mask).astype(np.uint8), cv2.DIST_L2, 3)
    m = float((dt_src[model_mask] <= tol_px).mean()) if model_mask.any() else 0.0
    s = float((dt_mod[source_mask] <= tol_px).mean()) if source_mask.any() else 0.0
    return Agreement(img, m, s)


def background_mask(image_bgr: np.ndarray, background_bgr=None, tol: int = 24) -> np.ndarray:
    """Everything that is not the flat background of a viewer screenshot (taken from the corners if not given)."""
    if background_bgr is None:
        corners = np.array([image_bgr[0, 0], image_bgr[0, -1], image_bgr[-1, 0], image_bgr[-1, -1]], dtype=np.int32)
        background_bgr = np.median(corners, axis=0)
    diff = np.abs(image_bgr.astype(np.int32) - np.asarray(background_bgr, dtype=np.int32)).sum(axis=2)
    return diff > tol


def outline(mask: np.ndarray, close_px: int = 31) -> np.ndarray:
    """The boundary of a mask against the background once its holes are closed (a lattice becomes its
    silhouette)."""
    import cv2

    k = max(3, int(close_px) | 1)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return cv2.morphologyEx(closed, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)


def outline_agreement(source_mask: np.ndarray, model_mask: np.ndarray, tol_px: float = 2.0,
                      close_px: int = 31) -> float:
    """Share of the SOURCE's outline within `tol_px` of the model's outline (both closed first).

    The gate for photos of frame-filling structures: there `agreement().model_within_px` stays near 1 for a
    wrong pose, because almost any model pixel lands on some structure (m13 photo, 2026-11-25: 0.97-0.99 for
    wrong poses; this measure 0.53 and 0.61 wrong, 0.92 right)."""
    import cv2

    src, mod = outline(source_mask, close_px), outline(model_mask, close_px)
    if not src.any() or not mod.any():
        return 0.0
    dt = cv2.distanceTransform((~mod).astype(np.uint8), cv2.DIST_L2, 3)
    return float((dt[src] <= tol_px).mean())
