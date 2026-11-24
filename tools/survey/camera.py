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

    @property
    def focal_px(self) -> float:
        return (self.image_h / 2.0) / math.tan(math.radians(self.fov_v_deg) / 2.0)

    def intrinsic(self) -> np.ndarray:
        f = self.focal_px
        return np.array([[f, 0.0, self.image_w / 2.0], [0.0, f, self.image_h / 2.0], [0.0, 0.0, 1.0]])

    def project(self, world) -> np.ndarray:
        """(N, 2) pixels; points behind the camera come back as NaN."""
        right, up, fwd = basis(self.direction, self.up)
        d = np.atleast_2d(np.asarray(world, dtype=np.float64)) - np.asarray(self.position, dtype=np.float64)
        z = d @ fwd
        f = self.focal_px
        with np.errstate(divide="ignore", invalid="ignore"):
            px = np.stack([self.image_w / 2.0 + f * (d @ right) / z, self.image_h / 2.0 - f * (d @ up) / z], axis=1)
        px[z <= 1e-9] = np.nan
        return px

    def ray(self, px) -> tuple[np.ndarray, np.ndarray]:
        """Origin and unit directions (N, 3) through pixels."""
        right, up, fwd = basis(self.direction, self.up)
        p = np.atleast_2d(np.asarray(px, dtype=np.float64))
        f = self.focal_px
        d = fwd * f + np.outer(p[:, 0] - self.image_w / 2.0, right) - np.outer(p[:, 1] - self.image_h / 2.0, up)
        return np.asarray(self.position, dtype=np.float64), d / np.linalg.norm(d, axis=1, keepdims=True)

    def to_contract(self) -> dict:
        right, up, fwd = basis(self.direction, self.up)
        del right
        return {"kind": "perspective", "viewPoint": list(map(float, self.position)), "direction": list(map(float, fwd)),
                "upVector": list(map(float, up)), "fieldOfView": float(self.fov_v_deg),
                "imageW": int(self.image_w), "imageH": int(self.image_h)}

    @classmethod
    def from_contract(cls, d: dict) -> PerspectiveCamera:
        if d.get("kind") != "perspective":
            raise ValueError(f"not a perspective camera: {d.get('kind')!r}")
        return cls(tuple(d["viewPoint"]), tuple(d["direction"]), tuple(d["upVector"]), float(d["fieldOfView"]),
                   int(d["imageW"]), int(d["imageH"]))

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
