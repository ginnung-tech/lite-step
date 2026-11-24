"""Ink: what a drawing draws, as a mask, and measurements read from it through a camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import OrthoCamera


def ink_mask(gray: np.ndarray, threshold: int = 110, min_stroke_px: int = 0, close_px: int = 0,
             blur_px: int = 3) -> np.ndarray:
    """Dark pixels after a slight blur, as a boolean mask.

    - `min_stroke_px`: connected components whose longer side is shorter are dropped (lettering, dashes,
      dimension ticks). In the Eiffel run 3 m at the plate's scale.
    - `close_px`: a morphological closing first, for shaded plates where members are drawn as rows of
      rivet dots (0.25-0.6 m in the run).
    """
    import cv2

    g = cv2.GaussianBlur(gray, (blur_px, blur_px), 0) if blur_px > 1 else gray
    m = (g < threshold).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if min_stroke_px > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= min_stroke_px
        m = keep[labels].astype(np.uint8)
    return m.astype(bool)


@dataclass(frozen=True)
class ProfileRow:
    v: float  # height (or up coordinate) in metres
    left: float | None  # outermost ink at this height, u in metres
    right: float | None

    @property
    def width(self) -> float | None:
        return None if self.left is None or self.right is None else self.right - self.left

    @property
    def centre(self) -> float | None:
        return None if self.left is None or self.right is None else (self.left + self.right) / 2


def silhouette_profile(mask: np.ndarray, cam: OrthoCamera, u_range: tuple[float, float],
                       v_range: tuple[float, float], step_m: float = 0.05) -> list[ProfileRow]:
    """Per `step_m` of height, the outermost ink inside a world window: a profile in metres.

    How v12 measured the arch's pedestal on PL. III (collar, capital, socle, base). Keep the window clear
    of neighbouring ink; rows that cross hatching or lettering inside it read wide and must be judged.
    """
    h, w = mask.shape
    x0, x1 = sorted(cam.ax_px + cam.k * cam.px_per_m * np.asarray(u_range, dtype=np.float64))
    xl, xr = max(0, int(np.floor(x0))), min(w, int(np.ceil(x1)))
    rows = []
    for v in np.arange(v_range[0], v_range[1] + 1e-9, step_m):
        y = int(round(cam.c_px - cam.px_per_m * v))
        if not (0 <= y < h) or xr <= xl:
            rows.append(ProfileRow(float(v), None, None))
            continue
        xs = np.nonzero(mask[y, xl:xr])[0]
        if len(xs) < 2:
            rows.append(ProfileRow(float(v), None, None))
            continue
        u = cam.to_uv(np.array([[xl + xs[0], y], [xl + xs[-1], y]]))[:, 0]
        rows.append(ProfileRow(round(float(v), 6), float(u[0]), float(u[1])))
    return rows
