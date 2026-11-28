"""Depth: per element, how far the model's surface lies in front of or behind the surface a reconstruction measured.

An outline (`shape`) cannot see a recessed window, a set-back storey, a courtyard or a balcony that sticks out too
far: they all sit inside the silhouette. A reconstruction's depth map can. Through the same aligned camera, the
drone gives the model's depth at every pixel and which element it hits; the depth map gives the measured depth
there. Their difference, collected per element, says "this element is 140 mm behind where the footage puts its
surface".

The depth map comes from the reconstruction (COLMAP's dense `stereo/depth_maps/*.geometric.bin`, or any `.npy`),
in the reconstruction's own units: it is scaled by the camera's alignment (`cameras` records it) or by `--scale`.
COLMAP stores depth along the camera's axis (z); `--ray` reads a map of distances along each pixel's ray instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .camera import PerspectiveCamera, basis, camera_from_contract

TOL_MM = 100.0  # an element whose surface is off by less than this (median) agrees with the measurement
MIN_PX = 50  # an element seen on fewer pixels than this is not judged


def read_depth_map(path: Path) -> np.ndarray:
    """(H, W) float32. A COLMAP depth map (`W&H&C&` then float32, column-major) or a NumPy `.npy`."""
    if path.suffix.lower() == ".npy":
        a = np.load(path, allow_pickle=False)
        if a.ndim != 2:
            raise ValueError(f"{path.name}: a depth map is a 2D array, not {a.shape}")
        return a.astype(np.float32)
    raw = path.read_bytes()
    head = raw[:64].split(b"&")
    try:
        w, h, c = int(head[0]), int(head[1]), int(head[2])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{path.name}: not a COLMAP depth map (header W&H&C&) nor a .npy") from exc
    start = len(b"&".join(head[:3])) + 1
    a = np.frombuffer(raw, dtype="<f4", offset=start, count=w * h * c)
    return a.reshape((w, h, c), order="F").transpose(1, 0, 2)[..., 0].astype(np.float32)


def compare_depth(model_t: np.ndarray, ids: np.ndarray, measured: np.ndarray, cam: PerspectiveCamera,
                  labels: list[str], *, ray: bool = False, tol_mm: float = TOL_MM,
                  min_px: int = MIN_PX) -> dict[str, Any]:
    """`model_t` is the drone's distance along each ray, `measured` the reconstruction's depth in metres (z along
    the camera's axis, or along the ray with `ray`). Positive error: the model's surface is BEHIND the measured
    one (too far back, recessed too deep); negative: in FRONT of it (sticks out)."""
    from .drone import rays

    h, w = ids.shape
    if measured.shape != (h, w):
        import cv2

        measured = cv2.resize(measured, (w, h), interpolation=cv2.INTER_NEAREST)
    if ray:
        model_d = model_t
    else:
        _, _, fwd = basis(cam.direction, cam.up)
        dirs = rays(cam)[..., 3:].astype(np.float64)
        model_d = model_t * (dirs @ fwd)  # along the ray -> along the camera's axis
    hit = (ids >= 0) & np.isfinite(model_t)
    valid = np.isfinite(measured) & (measured > 0)
    both = hit & valid
    err = np.where(both, model_d - measured, np.nan)
    elements: list[dict[str, Any]] = []
    for e in np.unique(ids[both]).tolist():
        m = both & (ids == e)
        n = int(m.sum())
        if n < min_px:
            continue
        v = err[m] * 1000.0
        q1, med, q3 = (float(x) for x in np.percentile(v, [25, 50, 75]))
        elements.append({"element": labels[e], "pixels": n, "medianMm": round(med, 0), "iqrMm": [round(q1, 0),
                                                                                                round(q3, 0)]})
    elements.sort(key=lambda r: -abs(r["medianMm"]) * r["pixels"])
    findings = []
    for r in elements:
        med, (q1, q3) = r["medianMm"], r["iqrMm"]
        # a median beyond the tolerance: at least half the element's pixels are off the same way, so a part that is
        # right (the back of a balcony) or a few noisy pixels of the reconstruction cannot make or hide a finding
        if abs(med) > tol_mm:
            where = "behind" if med > 0 else "in front of"
            findings.append({**r, "text": f"{r['element']} lies {abs(med):.0f} mm {where} the surface the reconstruction "
                                          f"measured (median of {r['pixels']} px; middle half {q1:.0f} to {q3:.0f} mm)"})
    seen = err[both] * 1000.0
    return {
        "pixelsCompared": int(both.sum()), "modelPixels": int(hit.sum()), "measuredPixels": int(valid.sum()),
        "medianMm": round(float(np.median(seen)), 0) if seen.size else None,
        "absP90Mm": round(float(np.percentile(np.abs(seen), 90)), 0) if seen.size else None,
        "toleranceMm": tol_mm, "findings": findings, "elements": elements[:300], "_err": err,
    }


def run_depth(workdir: str | Path, source_id: str, depth_map: str, *, model_file: str = "output.ifc",
              camera_version: int | None = None, scale: float | None = None, ray: bool = False,
              tol_mm: float = TOL_MM) -> dict[str, Any]:
    import cv2

    from .drone import Drone
    from .intake import CAMERAS, IntakeError, _read, _source, confined, survey_dir
    from .scene import Scene

    wd = Path(workdir).resolve()
    _source(wd, source_id)
    cams = [c for c in _read(survey_dir(wd) / CAMERAS, "cameras")["cameras"]
            if c["source"] == source_id and c.get("gate") == "passed"
            and (camera_version is None or c["version"] == camera_version)]
    if not cams:
        raise IntakeError(f"{source_id} has no camera that passed its gate")
    entry = max(cams, key=lambda c: c["version"])
    cam = camera_from_contract(entry["camera"])
    if not isinstance(cam, PerspectiveCamera):
        raise IntakeError("a depth map belongs to a perspective camera (a reconstruction frame)")
    if scale is None:
        al = entry.get("alignment")
        if not al:
            raise IntakeError("the depth map is in the reconstruction's units: take the frame in with `cameras` (its "
                              "alignment carries the scale) or pass --scale (metres per reconstruction unit)")
        scale = float(al["scale"])
    measured = read_depth_map(confined(wd, depth_map)) * float(scale)
    scene = Scene.from_ifc(confined(wd, model_file), cache_dir=survey_dir(wd) / ".cache")
    frame = Drone(scene).render(cam)
    r = compare_depth(frame.depth, frame.element, measured, cam, [e.label for e in scene.elements], ray=ray,
                      tol_mm=tol_mm)
    err = r.pop("_err")
    # overlay: red where the model stands in front of the measured surface, blue where it is behind, grey within
    # the tolerance; black where nothing was compared
    vis = np.zeros((*err.shape, 3), np.uint8)
    ok = np.isfinite(err)
    mm = np.where(ok, err * 1000.0, 0.0)
    k = np.clip(np.abs(mm) / (5 * tol_mm), 0, 1)
    vis[ok] = (150, 150, 150)
    front, back = ok & (mm < -tol_mm), ok & (mm > tol_mm)
    vis[front] = np.stack([60 * (1 - k[front]), 60 * (1 - k[front]), 150 + 105 * k[front]], axis=1).astype(np.uint8)
    vis[back] = np.stack([150 + 105 * k[back], 90 * (1 - k[back]), 40 * (1 - k[back])], axis=1).astype(np.uint8)
    rel = f"depth/{source_id}-v{entry['version']}"
    png = confined(survey_dir(wd), rel + ".png")
    png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png), vis)
    report = {"version": "1.0", "source": source_id, "cameraVersion": entry["version"], "model": model_file,
              "depthMap": depth_map, "scale": scale, "along": "ray" if ray else "z", "overlay": rel + ".png", **r}
    (survey_dir(wd) / (rel + ".json")).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
