"""The drone: look at the model through any survey camera and get, per pixel, the element it sees.

Ray casting with Open3D's Embree on the CPU: the same buffers in a container and on a desktop, no display.
The ID buffer is how anything a source shows becomes elements without a language model guessing ids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import OrthoCamera, PerspectiveCamera, basis
from .scene import Scene

_INVALID = np.uint32(0xFFFFFFFF)
STANDOFF_M = 5000.0  # ortho rays start this far behind the plane through the origin


@dataclass
class Frame:
    element: np.ndarray  # (H, W) int32, -1 = nothing
    depth: np.ndarray  # (H, W) float32, metres along the ray; inf = nothing

    @property
    def hit(self) -> np.ndarray:
        return self.element >= 0


def rays(cam: OrthoCamera | PerspectiveCamera) -> np.ndarray:
    """(H, W, 6) float32 rays [origin, unit direction] through every pixel centre."""
    w, h = cam.image_w, cam.image_h
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64) + 0.5, np.arange(h, dtype=np.float64) + 0.5)
    px = np.stack([xs.ravel(), ys.ravel()], axis=1)
    if isinstance(cam, OrthoCamera):
        _, _, fwd = basis(cam.direction, cam.up)
        # A section or front slab starts its rays ON the near plane, so what lies in front of the cut is
        # not seen at all (rather than hiding what is behind it); otherwise far behind everything.
        start = cam.depth_range_m[0] if cam.depth_range_m is not None else -STANDOFF_M
        o = cam.to_world(px, depth_m=start)
        d = np.broadcast_to(fwd, o.shape)
    else:
        origin, d = cam.ray(px)
        o = np.broadcast_to(origin, d.shape)
    return np.concatenate([o, d], axis=1).reshape(h, w, 6).astype(np.float32)


class Drone:
    def __init__(self, scene: Scene) -> None:
        import open3d as o3d  # optional dependency: tools/survey/requirements.txt

        self.scene = scene
        self._o3d = o3d
        self._rc = o3d.t.geometry.RaycastingScene()
        if len(scene.triangles):
            self._rc.add_triangles(o3d.core.Tensor(np.ascontiguousarray(scene.vertices, dtype=np.float32)),
                                   o3d.core.Tensor(np.ascontiguousarray(scene.triangles, dtype=np.uint32)))

    def render(self, cam: OrthoCamera | PerspectiveCamera) -> Frame:
        r = rays(cam)
        ans = self._rc.cast_rays(self._o3d.core.Tensor(r))
        t = ans["t_hit"].numpy()
        prim = ans["primitive_ids"].numpy()
        hit = prim != _INVALID
        element = np.full(prim.shape, -1, dtype=np.int32)
        if len(self.scene.tri_element):
            element[hit] = self.scene.tri_element[prim[hit].astype(np.int64)]
        depth = np.where(hit, t, np.inf).astype(np.float32)
        if isinstance(cam, OrthoCamera):  # depth from the plane through the world origin
            if cam.depth_range_m is None:
                depth = np.where(hit, depth - STANDOFF_M, np.inf).astype(np.float32)
            else:
                near, far = cam.depth_range_m
                depth = np.where(hit, depth + near, np.inf).astype(np.float32)
                out = hit & (depth > far)  # only what lies in the range
                element[out] = -1
                depth[out] = np.inf
        return Frame(element, depth)

    def pick(self, frame: Frame, x: float, y: float):
        xi, yi = int(x), int(y)
        if 0 <= yi < frame.element.shape[0] and 0 <= xi < frame.element.shape[1] and frame.element[yi, xi] >= 0:
            return self.scene.elements[int(frame.element[yi, xi])]
        return None
