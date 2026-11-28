"""Cameras from a reconstruction (COLMAP, or a Nerfstudio / instant-ngp `transforms.json`), aligned to the model.

Photogrammetry solves every frame's camera for free, but in a frame of its OWN: its origin, its axes and its
scale are arbitrary (COLMAP has no idea what a metre is, which way is up, or where the model's origin sits).
Before one of its cameras can look at the model, the reconstruction must be moved onto the model's frame by a
similarity transform: a rotation, a translation and ONE scale, ``world = s * R @ recon + t``. Nothing here guesses
it. The import needs one of:

- ``points``: three or more points known in both frames (a corner picked in the reconstruction's point cloud and
  the same corner in the model). Solved by least squares (Umeyama); each point's residual is reported and the
  alignment gates on them, so a mislabelled point shows.
- ``transform``: ``{"scale", "rotation" (3x3), "translation"}`` worked out elsewhere (from drone GPS and gravity,
  georeferenced into the model's frame), taken as given and said so.
- ``identity`` with a ``reason``: the reconstruction was already made in the model's frame and metres (a
  georeferenced pipeline that wrote model coordinates). Claiming it without a reason is refused.

Then each chosen frame is a source with a fitted perspective camera, shown back with the model drawn on it, like
any other picture; `shape` and `inspect` measure through it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .camera import PerspectiveCamera

ALIGN_GATE_M = 0.10  # an alignment's points must agree to this (or ALIGN_GATE_SHARE of their span), on average
ALIGN_GATE_SHARE = 0.005

ALIGNMENT_HELP = (
    "a reconstruction's cameras are in its own frame (arbitrary origin, axes and scale), so they cannot look at "
    "the model until that frame is moved onto the model's by a similarity transform (rotation, translation, one "
    "scale). Give --align FILE with one of: {\"points\": [{\"recon\": [x, y, z], \"world\": [x, y, z]}, ...]} "
    "(three or more points known in both: corners you can find in the point cloud and in the model), "
    "{\"transform\": {\"scale\": s, \"rotation\": [[...], [...], [...]], \"translation\": [x, y, z]}} (worked "
    "out elsewhere, e.g. from drone GPS and gravity), or {\"identity\": true, \"reason\": \"...\"} when the "
    "reconstruction was already made in the model's coordinates and metres.")


class ReconError(ValueError):
    """A reconstruction that cannot be read or aligned as asked."""


@dataclass(frozen=True)
class Frame:
    name: str  # the image file name as the reconstruction knows it
    centre: np.ndarray  # camera centre, reconstruction frame
    rot_c2w: np.ndarray  # columns: camera right, down, forward (OpenCV convention), reconstruction frame
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: tuple[float, ...] | None


# -- reading ----------------------------------------------------------------------------------------------------


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _colmap_intrinsics(model: str, p: list[float]) -> tuple[float, float, float, float, tuple[float, ...] | None]:
    """(fx, fy, cx, cy, distortion in OpenCV order) for the COLMAP camera models a survey can use."""
    if model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2], None
    if model == "PINHOLE":
        return p[0], p[1], p[2], p[3], None
    if model == "SIMPLE_RADIAL":
        return p[0], p[0], p[1], p[2], (p[3],)
    if model == "RADIAL":
        return p[0], p[0], p[1], p[2], (p[3], p[4])
    if model == "OPENCV":
        return p[0], p[1], p[2], p[3], (p[4], p[5], p[6], p[7])
    raise ReconError(f"COLMAP camera model {model} is not one a survey reads (SIMPLE_PINHOLE, PINHOLE, "
                     "SIMPLE_RADIAL, RADIAL, OPENCV): undistort the images in COLMAP first (image_undistorter)")


def _lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def read_colmap(model_dir: Path) -> list[Frame]:
    """A COLMAP text model (`cameras.txt`, `images.txt`; export a binary one with `colmap model_converter`)."""
    cams_txt, imgs_txt = model_dir / "cameras.txt", model_dir / "images.txt"
    if not cams_txt.exists() or not imgs_txt.exists():
        raise ReconError(f"{model_dir} has no cameras.txt and images.txt (a binary model? convert it with "
                         "`colmap model_converter --output_type TXT`)")
    cams: dict[int, tuple] = {}
    for ln in _lines(cams_txt):
        f = ln.split()
        cams[int(f[0])] = (f[1], int(f[2]), int(f[3]), [float(v) for v in f[4:]])
    frames = []
    rows = _lines(imgs_txt)
    # images.txt: one pose line, then one line of 2D points (which may be empty and so already skipped)
    for ln in rows:
        f = ln.split()
        if len(f) < 10 or not f[0].isdigit():
            continue
        try:
            qw, qx, qy, qz, tx, ty, tz = (float(v) for v in f[1:8])
            cam_id = int(f[8])
        except ValueError:
            continue
        if cam_id not in cams:
            continue
        model, w, h, params = cams[cam_id]
        fx, fy, cx, cy, dist = _colmap_intrinsics(model, params)
        r_w2c = _quat_to_rot(qw, qx, qy, qz)
        centre = -r_w2c.T @ np.array([tx, ty, tz])
        frames.append(Frame(" ".join(f[9:]), centre, r_w2c.T, fx, fy, cx, cy, w, h, dist))
    if not frames:
        raise ReconError(f"{imgs_txt} lists no registered images")
    return frames


def read_transforms(path: Path) -> list[Frame]:
    """A Nerfstudio / instant-ngp `transforms.json`: camera-to-world matrices in the OpenGL convention (x right,
    y up, looking down -z), intrinsics at the top level or per frame."""
    d = json.loads(path.read_text(encoding="utf-8"))
    frames = []
    for fr in d.get("frames", []):
        get = lambda k, default=None: fr.get(k, d.get(k, default))  # noqa: E731
        w, h = get("w"), get("h")
        fx = get("fl_x")
        fy = get("fl_y", fx)
        if fx is None and get("camera_angle_x") is not None and w:
            fx = fy = (w / 2.0) / math.tan(float(get("camera_angle_x")) / 2.0)
        if fx is None or not w or not h:
            raise ReconError(f"{path.name}: frame {fr.get('file_path')} has no focal length or image size")
        m = np.asarray(fr["transform_matrix"], dtype=np.float64)
        rot_gl = m[:3, :3]
        rot_cv = rot_gl @ np.diag([1.0, -1.0, -1.0])  # OpenGL camera axes -> OpenCV (y down, z forward)
        dist = tuple(float(get(k, 0.0)) for k in ("k1", "k2", "p1", "p2"))
        frames.append(Frame(str(fr.get("file_path", "")), m[:3, 3].copy(), rot_cv, float(fx), float(fy),
                            float(get("cx", w / 2.0)), float(get("cy", h / 2.0)), int(w), int(h),
                            dist if any(dist) else None))
    if not frames:
        raise ReconError(f"{path.name} lists no frames")
    return frames


# -- aligning ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Alignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    method: str  # points | transform | identity
    residual_m: float | None  # mean, for `points`
    per_point_m: tuple[float, ...] = ()
    gate: str = "passed"
    reason: str = ""

    def apply(self, p: np.ndarray) -> np.ndarray:
        return self.scale * (np.asarray(p, dtype=np.float64) @ self.rotation.T) + self.translation

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": self.method, "scale": self.scale, "rotation": self.rotation.tolist(),
                               "translation": self.translation.tolist(), "gate": self.gate, "reason": self.reason}
        if self.residual_m is not None:
            out["residualMm"] = round(self.residual_m * 1000.0, 1)
            out["perPointMm"] = [round(v * 1000.0, 1) for v in self.per_point_m]
        return out


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """The similarity (s, R, t) minimising |dst - (s R src + t)|^2 (Umeyama 1991); R is a proper rotation."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    u, d, vt = np.linalg.svd(cov)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rot = u @ sign @ vt
    var = (a ** 2).sum() / len(src)
    if var <= 0:
        raise ReconError("the alignment points coincide: spread them across the structure")
    s = float(np.trace(np.diag(d) @ sign) / var)
    return s, rot, mu_d - s * rot @ mu_s


def alignment_from(spec: dict | None) -> Alignment:
    if not spec:
        raise ReconError(ALIGNMENT_HELP)
    if spec.get("identity"):
        reason = str(spec.get("reason") or "").strip()
        if len(reason) < 10:
            raise ReconError("an identity alignment says the reconstruction is already in the model's coordinates "
                             "and metres: give the reason (which pipeline wrote model coordinates)")
        return Alignment(1.0, np.eye(3), np.zeros(3), "identity", None, reason=reason[:300])
    if "transform" in spec:
        t = spec["transform"]
        rot = np.asarray(t["rotation"], dtype=np.float64)
        if rot.shape != (3, 3) or abs(np.linalg.det(rot) - 1.0) > 1e-3 or not np.allclose(rot.T @ rot, np.eye(3), atol=1e-3):
            raise ReconError("transform.rotation must be a 3x3 rotation (orthonormal, determinant 1)")
        s = float(t["scale"])
        if s <= 0:
            raise ReconError("transform.scale must be positive")
        return Alignment(s, rot, np.asarray(t["translation"], dtype=np.float64), "transform", None,
                         reason=str(spec.get("reason") or "given")[:300])
    pts = spec.get("points") or []
    if len(pts) < 3:
        raise ReconError("an alignment from points needs three or more, each {\"recon\": [x, y, z], \"world\": "
                         "[x, y, z]}, spread across the structure (not on one line). " + ALIGNMENT_HELP)
    src = np.array([p["recon"] for p in pts], dtype=np.float64)
    dst = np.array([p["world"] for p in pts], dtype=np.float64)
    s, rot, t = umeyama(src, dst)
    per = np.linalg.norm(dst - (s * src @ rot.T + t), axis=1)
    span = float(np.linalg.norm(dst.max(axis=0) - dst.min(axis=0)))
    limit = max(ALIGN_GATE_M, ALIGN_GATE_SHARE * span)
    gate, reason = "passed", f"{len(pts)} points, Umeyama; mean {per.mean() * 1000:.0f} mm, scale {s:.5g}"
    if len(pts) == 3:
        reason += " (three points fit exactly: add a fourth so a wrong one can show)"
    if float(per.mean()) > limit:
        worst = int(np.argmax(per))
        gate = "failed"
        reason = (f"the points disagree: mean {per.mean() * 1000:.0f} mm against {limit * 1000:.0f} mm allowed; point "
                  f"{worst + 1} is off by {per[worst] * 1000:.0f} mm, likely a wrong pair")
    return Alignment(s, rot, t, "points", float(per.mean()), tuple(float(v) for v in per), gate, reason)


def aligned_camera(fr: Frame, al: Alignment) -> PerspectiveCamera:
    """One frame's camera in the model's frame. The image's own focal length stays (vertical field of view from
    fy); a non-square pixel is folded into the vertical one, with fx/fy kept within a percent by the caller."""
    centre = al.apply(fr.centre[None, :])[0]
    fwd = al.rotation @ fr.rot_c2w[:, 2]
    up = -(al.rotation @ fr.rot_c2w[:, 1])
    fov = math.degrees(2.0 * math.atan((fr.height / 2.0) / fr.fy))
    return PerspectiveCamera(tuple(centre.tolist()), tuple(fwd.tolist()), tuple(up.tolist()), fov, fr.width,
                             fr.height, (fr.cx, fr.cy), fr.distortion)


def import_frames(workdir: str | Path, frames: list[Frame], images: str | Path, align: dict | None, *, now: str,
                  n: int = 12, names: list[str] | None = None, scene=None, title: str = "reconstruction",
                  added_by: str = "heimdal") -> dict[str, Any]:
    """Take `n` well-spread frames (or the `names` given) in as photos, each with its aligned camera.

    The gate is the alignment's: a camera from a reconstruction is as good as the transform that brought it here.
    The model is drawn on each frame (overlay) and its outline agreement recorded, but a model that disagrees does
    not fail the camera: that disagreement is what `shape` and `inspect` are about to measure."""
    from .intake import (CAMERAS, OUTLINE_GATE, TOL_PX, _next_version, _read, _record, _save_overlay, add_source,
                         confined, structure_mask, survey_dir)

    al = alignment_from(align)
    wd = Path(workdir).resolve()
    img_dir = confined(wd, images)
    if names:
        by_name = {Path(f.name).name: i for i, f in enumerate(frames)}
        missing = [nm for nm in names if Path(nm).name not in by_name]
        if missing:
            raise ReconError(f"not in the reconstruction: {missing[:5]}")
        pick = [by_name[Path(nm).name] for nm in names]
    else:
        pick = spread(frames, n)
    out = []
    for i in pick:
        fr = frames[i]
        pic = img_dir / Path(fr.name).name
        if not pic.is_file():
            pic = img_dir / fr.name
        if not pic.is_file():
            raise ReconError(f"image {fr.name} is not in {images}")
        if abs(fr.fx / fr.fy - 1.0) > 0.01:
            raise ReconError(f"{fr.name}: fx/fy = {fr.fx / fr.fy:.3f}; a survey camera has square pixels: undistort "
                             "and resample the images first")
        src = add_source(wd, pic, kind="photo", title=f"{title}: {Path(fr.name).name}"[:300], now=now,
                         origin=f"reconstruction frame {fr.name}", added_by=added_by)
        cam = aligned_camera(fr, al)
        import cv2

        img = cv2.imread(str(confined(survey_dir(wd), src["file"])))
        if img is None:
            raise ReconError(f"{fr.name} is not a picture OpenCV can read")
        if img.shape[1] != fr.width or img.shape[0] != fr.height:
            raise ReconError(f"{fr.name} is {img.shape[1]}x{img.shape[0]} but the reconstruction solved it at "
                             f"{fr.width}x{fr.height}: take in the images the reconstruction used")
        hit, agree = None, None
        if scene is not None:
            from .drone import Drone
            from .overlay import agreement, outline_agreement

            hit = Drone(scene).render(cam).hit
            structure = structure_mask(img)
            a = agreement(structure, hit, tol_px=TOL_PX)
            agree = {"tolPx": TOL_PX, "modelWithinPx": round(a.model_within_px, 4),
                     "sourceWithinPx": round(a.source_within_px, 4),
                     "outlineWithinPx": round(outline_agreement(structure, hit, TOL_PX), 4)}
        cams = _read(survey_dir(wd) / CAMERAS, "cameras")
        version = _next_version(cams, src["id"])
        reason = f"from the reconstruction, aligned by {al.method}: {al.reason}"
        if agree is not None and agree["outlineWithinPx"] < OUTLINE_GATE:
            reason += (f"; only {agree['outlineWithinPx']:.0%} of the picture's outline lies on the model's: the model "
                       "differs, or the alignment is off (look at the overlay)")
        entry: dict[str, Any] = {"source": src["id"], "version": version, "camera": cam.to_contract(),
                                 "method": "reconstruction", "residualMm": round((al.residual_m or 0.0) * 1000.0, 1),
                                 "gate": al.gate, "frozenAt": now, "reason": reason[:500], "alignment": al.to_json(),
                                 "overlay": _save_overlay(wd, src["id"], version, img, hit)}
        if agree is not None:
            entry["agreement"] = agree
        _record(wd, entry)
        out.append({"id": src["id"], "frame": fr.name, "fit": "passed" if al.gate == "passed" else "failed",
                    "cameraVersion": version, "overlay": entry["overlay"],
                    **({"outlineWithinPx": agree["outlineWithinPx"]} if agree else {})})
    return {"alignment": al.to_json(), "frames": out, "of": len(frames)}


def spread(frames: list[Frame], n: int) -> list[int]:
    """`n` frames far apart (farthest-point order over the camera centres): a reconstruction is hundreds of near-
    duplicates, and a survey wants a few views from different sides."""
    if n >= len(frames):
        return list(range(len(frames)))
    c = np.array([f.centre for f in frames])
    chosen = [int(np.argmin(np.linalg.norm(c - c.mean(axis=0), axis=1)))]
    d = np.linalg.norm(c - c[chosen[0]], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(c - c[k], axis=1))
    return sorted(chosen)
