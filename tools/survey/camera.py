"""Cameras that map the model's world (metres, Z up) to a source's pixels (x right, y down).

Two kinds, both serialised in the shapes of the Ginnung `survey` contract (`orthoCamera`,
`perspectiveCamera`):

- `OrthoCamera` for drawings: ``px = ax + k*s*u``, ``py = c - s*v`` with ``(u, v)`` the world coordinates
  along the camera's right and up. ``k`` is the picture's horizontal stretch: pasted and scanned figures
  are rarely square, and fitting one scale for both axes pushes the error into the model.
- `PerspectiveCamera` for screenshots, renders and photos: a BCF perspective camera (eye, direction,
  up, vertical field of view) with the size of the image it belongs to.

Right is ``direction x up`` (right-handed): looking north (+Y) with Z up, right is east (+X).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _unit(v, what: str) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        raise ValueError(f"{what} has zero length")
    return a / n


def basis(direction, up) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, forward), orthonormal. `up` is re-derived to be square to `direction`."""
    fwd = _unit(direction, "direction")
    hint = _unit(up, "up")
    right = np.cross(fwd, hint)
    if float(np.linalg.norm(right)) < 1e-9:
        raise ValueError("direction is parallel to up: the camera's roll is undefined")
    right /= np.linalg.norm(right)
    return right, np.cross(right, fwd), fwd


@dataclass(frozen=True)
class OrthoCamera:
    direction: tuple[float, float, float]
    up: tuple[float, float, float]
    px_per_m: float
    k: float
    ax_px: float
    c_px: float
    image_w: int
    image_h: int
    depth_range_m: tuple[float, float] | None = None

    def uv(self, world) -> np.ndarray:
        """(N, 2) world coordinates along right and up."""
        right, up, _ = basis(self.direction, self.up)
        x = np.atleast_2d(np.asarray(world, dtype=np.float64))
        return np.stack([x @ right, x @ up], axis=1)

    def project(self, world) -> np.ndarray:
        """(N, 2) pixels for (N, 3) world points."""
        uv = self.uv(world)
        return np.stack([self.ax_px + self.k * self.px_per_m * uv[:, 0], self.c_px - self.px_per_m * uv[:, 1]], axis=1)

    def to_uv(self, px) -> np.ndarray:
        """(N, 2) (u, v) in metres for (N, 2) pixels: the inverse of `project` in the picture plane."""
        p = np.atleast_2d(np.asarray(px, dtype=np.float64))
        return np.stack([(p[:, 0] - self.ax_px) / (self.k * self.px_per_m), (self.c_px - p[:, 1]) / self.px_per_m], axis=1)

    def to_world(self, px, depth_m: float = 0.0) -> np.ndarray:
        """(N, 3) world points for pixels, on the plane `depth_m` along the view direction from the origin."""
        right, up, fwd = basis(self.direction, self.up)
        uv = self.to_uv(px)
        return uv[:, :1] * right + uv[:, 1:] * up + depth_m * fwd

    def to_contract(self) -> dict:
        out = {"kind": "ortho", "direction": list(map(float, self.direction)), "up": list(map(float, self.up)),
               "pxPerM": float(self.px_per_m), "k": float(self.k), "axPx": float(self.ax_px), "cPx": float(self.c_px),
               "imageW": int(self.image_w), "imageH": int(self.image_h)}
        if self.depth_range_m is not None:
            out["depthRangeM"] = [float(self.depth_range_m[0]), float(self.depth_range_m[1])]
        return out

    @classmethod
    def from_contract(cls, d: dict) -> OrthoCamera:
        if d.get("kind") != "ortho":
            raise ValueError(f"not an ortho camera: {d.get('kind')!r}")
        dr = d.get("depthRangeM")
        return cls(tuple(d["direction"]), tuple(d["up"]), d["pxPerM"], d["k"], d["axPx"], d["cPx"],
                   int(d["imageW"]), int(d["imageH"]), tuple(dr) if dr else None)


@dataclass(frozen=True)
class PerspectiveCamera:
    position: tuple[float, float, float]
    direction: tuple[float, float, float]
    up: tuple[float, float, float]
    fov_v_deg: float
    image_w: int
    image_h: int
    # 1.2: where the optical axis meets the image (a cropped photo moves it); None = the image centre
    principal_point: tuple[float, float] | None = None
    # 1.2: lens distortion, OpenCV order (k1, k2, p1, p2, k3), any prefix; None = none (a viewer screenshot)
    distortion: tuple[float, ...] | None = None

    @property
    def focal_px(self) -> float:
        return (self.image_h / 2.0) / math.tan(math.radians(self.fov_v_deg) / 2.0)

    @property
    def centre_px(self) -> tuple[float, float]:
        if self.principal_point is not None:
            return float(self.principal_point[0]), float(self.principal_point[1])
        return self.image_w / 2.0, self.image_h / 2.0

    def intrinsic(self) -> np.ndarray:
        f = self.focal_px
        cx, cy = self.centre_px
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    def _coeffs(self) -> tuple[float, float, float, float, float]:
        k = list(self.distortion or ())
        return tuple((k + [0.0] * 5)[:5])  # type: ignore[return-value]

    def _distort(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalised image coordinates (y down) through the lens."""
        k1, k2, p1, p2, k3 = self._coeffs()
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        return (x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)

    def _undistort(self, xd: np.ndarray, yd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The inverse of `_distort`, by fixed-point iteration (exact enough for a camera lens)."""
        if not self.distortion or not any(self.distortion):
            return xd, yd
        x, y = xd.copy(), yd.copy()
        for _ in range(20):
            dx, dy = self._distort(x, y)
            x, y = x - (dx - xd), y - (dy - yd)
        return x, y

    def project(self, world) -> np.ndarray:
        """(N, 2) pixels; points behind the camera come back as NaN."""
        right, up, fwd = basis(self.direction, self.up)
        d = np.atleast_2d(np.asarray(world, dtype=np.float64)) - np.asarray(self.position, dtype=np.float64)
        z = d @ fwd
        f = self.focal_px
        cx, cy = self.centre_px
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = (d @ right) / z, -(d @ up) / z
            x, y = self._distort(x, y)
            px = np.stack([cx + f * x, cy + f * y], axis=1)
        px[z <= 1e-9] = np.nan
        return px

    def ray(self, px) -> tuple[np.ndarray, np.ndarray]:
        """Origin and unit directions (N, 3) through pixels."""
        right, up, fwd = basis(self.direction, self.up)
        p = np.atleast_2d(np.asarray(px, dtype=np.float64))
        f = self.focal_px
        cx, cy = self.centre_px
        x, y = self._undistort((p[:, 0] - cx) / f, (p[:, 1] - cy) / f)
        d = fwd + np.outer(x, right) - np.outer(y, up)
        return np.asarray(self.position, dtype=np.float64), d / np.linalg.norm(d, axis=1, keepdims=True)

    def to_contract(self) -> dict:
        right, up, fwd = basis(self.direction, self.up)
        del right
        out = {"kind": "perspective", "viewPoint": list(map(float, self.position)), "direction": list(map(float, fwd)),
               "upVector": list(map(float, up)), "fieldOfView": float(self.fov_v_deg),
               "imageW": int(self.image_w), "imageH": int(self.image_h)}
        if self.principal_point is not None:
            out["principalPointPx"] = [float(v) for v in self.principal_point]
        if self.distortion:
            out["distortion"] = [float(v) for v in self.distortion]
        return out

    @classmethod
    def from_contract(cls, d: dict) -> PerspectiveCamera:
        if d.get("kind") != "perspective":
            raise ValueError(f"not a perspective camera: {d.get('kind')!r}")
        pp = d.get("principalPointPx")
        dist = d.get("distortion")
        return cls(tuple(d["viewPoint"]), tuple(d["direction"]), tuple(d["upVector"]), float(d["fieldOfView"]),
                   int(d["imageW"]), int(d["imageH"]), tuple(pp) if pp else None, tuple(dist) if dist else None)

    def crop(self, x0: int, y0: int, x1: int, y1: int) -> PerspectiveCamera:
        """The same camera for a crop of its picture: the focal length stays, the principal point moves, the
        vertical field of view follows the new height. A pixel of the crop sees what the same pixel of the
        whole picture saw."""
        f = self.focal_px
        cx, cy = self.centre_px
        h = y1 - y0
        fov = math.degrees(2 * math.atan((h / 2.0) / f))
        return PerspectiveCamera(self.position, self.direction, self.up, fov, x1 - x0, h,
                                 (cx - x0, cy - y0), self.distortion)

    @classmethod
    def look_at(cls, eye, target, fov_v_deg: float, image_w: int, image_h: int, roll_deg: float = 0.0) -> PerspectiveCamera:
        """Eye looking at target with Z as the up hint, turned by `roll_deg` about the view axis."""
        e = np.asarray(eye, dtype=np.float64)
        fwd = _unit(np.asarray(target, dtype=np.float64) - e, "target - eye")
        right, up, _ = basis(fwd, (0.0, 0.0, 1.0))
        a = math.radians(roll_deg)
        up_r = math.cos(a) * up + math.sin(a) * right
        return cls(tuple(e.tolist()), tuple(fwd.tolist()), tuple(up_r.tolist()), float(fov_v_deg), image_w, image_h)


def camera_from_contract(d: dict) -> OrthoCamera | PerspectiveCamera:
    return OrthoCamera.from_contract(d) if d.get("kind") == "ortho" else PerspectiveCamera.from_contract(d)
