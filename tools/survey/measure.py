"""Measure a model's dimensions from a picture: the camera and the script's dimensions, fitted together.

A first build from a reference picture has the right parts and guessed sizes. Fitting a camera to that model
cannot succeed: the control points sit where the GUESS put them, so even correctly labelled points disagree,
and the fit is refused (staging 2026-11-26, a carport built from one photo: 150 mm posts on a 3 m grid against
300 mm posts on a 4 m grid in the picture). The loop Surveyor -> Gróa -> minions never starts.

Here the picture measures the guess instead. The person's (or the Surveyor's) control points name an
element's box corner, e.g. ``post_0_0:xmin,ymax,zmin``. Each dimension of the script (an UPPER_CASE
module-level number or tuple of numbers, ``param(...)`` included) is perturbed once and the script compiled,
which gives how every named corner moves with it; a model made of boxes moves linearly, so this is exact for
them and a first-order estimate otherwise. Then the camera (pose and focal length) and the free dimensions are
solved together by least squares on the pixel error.

One picture cannot tell a small carport near the lens from a big one far away: at least one dimension must be
held (``fixed``), the one the person stated or the one with the best outside evidence (a stepladder, a door).
Dimensions no point depends on keep their value (a weak pull to the old value, which the result reports as
``seen: false``), so the answer never invents a size the picture does not show.

Several pictures of one scene are fitted together (``measure_many``): one set of dimensions, a camera each.
Before the fit, points that lie on one straight line in the model (a row of post feet) must lie on one line in
the picture: a point that does not is named (``labelChecks``), because what goes wrong is the label, not the
pixel. After it, a point far worse than the rest (an occluded foot read at the trailer's edge) is dropped and
the fit repeated, at most a fifth of a picture's points, each reported with its error (``dropped``). Carport,
2026-11-26: two photos, labelled by rows, with the house and the shed as reference points, fitted to 6 px each
and agreed on every size within their spread.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .camera import PerspectiveCamera
from .fit import fit_perspective

_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_FEATURE = re.compile(r"^(?P<el>[^:]+):(?P<x>xmin|xmax),(?P<y>ymin|ymax),(?P<z>zmin|zmax)$")


#: a picture whose kept points miss the model by more than this many pixels on average (or this share of its
#: diagonal, whichever is larger) is a poor fit, not a measurement
POOR_FIT_PX = 15.0
POOR_FIT_FRACTION = 0.01
#: a measured size whose one-sigma spread is larger than this share of its value is flagged uncertain
UNCERTAIN_FRACTION = 0.5


class MeasureError(ValueError):
    """The inputs cannot give a measurement; the message says which input."""


@dataclass(frozen=True)
class Dimension:
    """One number of the script: ``POST_SIZE`` or ``POST_X[1]``, with where its literal is written."""

    key: str
    value: float
    is_int: bool
    line: int  # 1-based
    col: int  # 0-based, start of the literal (a leading minus included)
    end_col: int


def _number(node: ast.AST) -> tuple[float, bool, int] | None:
    """A numeric literal (a unary minus allowed): value, is it an int, start column."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value), isinstance(node.value, int), node.col_offset
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float)) and not isinstance(node.operand.value, bool)):
        return -float(node.operand.value), isinstance(node.operand.value, int), node.col_offset
    return None


def read_dimensions(source: str) -> list[Dimension]:
    """Every UPPER_CASE module-level number of a script, one entry per number (tuples give one per element).

    ``NAME = param(value, ...)`` counts by its value. Expressions (``A / 2``) are not dimensions: they follow
    the numbers they are made of."""
    out: list[Dimension] = []
    for stmt in ast.parse(source).body:
        for key, node in _candidates(stmt):
            num = _number(node)
            if num is not None and node.end_lineno == node.lineno:
                out.append(Dimension(key, num[0], num[1], node.lineno, num[2], node.end_col_offset))
    return out


def _candidates(stmt: ast.stmt) -> list[tuple[str, ast.AST]]:
    """The (key, value node) pairs one module-level statement could contribute: none unless it is
    ``NAME = ...`` with an UPPER_CASE name; ``param(v, ...)`` unwrapped to ``v``; a tuple split per element."""
    if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
        return []
    name = stmt.targets[0].id
    if not _NAME.match(name):
        return []
    value = stmt.value
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "param" and value.args:
        value = value.args[0]
    if isinstance(value, (ast.Tuple, ast.List)):
        return [(f"{name}[{i}]", v) for i, v in enumerate(value.elts)]
    return [(name, value)]


def with_dimensions(source: str, dims: list[Dimension], values: dict[str, float]) -> str:
    """The script with the given dimensions' literals replaced; every other byte unchanged."""
    lines = source.splitlines(keepends=True)
    edits = sorted((d for d in dims if d.key in values), key=lambda d: (d.line, -d.col))
    for d in edits:
        v = values[d.key]
        text = str(int(round(v))) if d.is_int else repr(float(v))
        row = lines[d.line - 1]
        lines[d.line - 1] = row[:d.col] + text + row[d.end_col:]
    return "".join(lines)


def compile_script(source: str, lite_step_root: str | os.PathLike, timeout_s: float = 600.0) -> Path:
    """Run a model script in a fresh directory; the path of the IFC it wrote. Raises with the tail of its
    output when it fails, because a dimension that breaks the script is an answer too."""
    work = Path(tempfile.mkdtemp(prefix="measure-"))
    script = work / "model.py"  # a fixed name in a directory made here: nothing in the path comes from input
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(source)
    env = dict(os.environ, PYTHONPATH=str(lite_step_root))
    p = subprocess.run([sys.executable, str(script)], cwd=work, env=env, capture_output=True, text=True,
                       timeout=timeout_s)
    out = work / "output.ifc"
    if p.returncode != 0 or not out.exists():
        tail = (p.stdout + p.stderr)[-800:]
        raise MeasureError(f"the script did not compile (exit {p.returncode}): {tail}")
    return out


def corners(ifc_path: str | os.PathLike, elements: set[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The world box (min, max) in metres of each named element of a compiled model."""
    from .scene import Scene

    scene = Scene.from_ifc(ifc_path)
    # a feature names an element as the script does (`post_0_0`); the IFC carries the canonical name
    # (`proxy#1:column:post_0_0:storey:ground`), so a name matches the whole name or one of its segments, and
    # every element it matches (a product and its body proxy) counts toward one box
    members: dict[str, list[int]] = {n: [] for n in elements}
    for el in scene.elements:
        if not el.name:
            continue
        parts = set(el.name.split(":"))
        for n in elements:
            if el.name == n or n in parts:
                members[n].append(el.index)
    out = {n: scene.bounds_of(ix) for n, ix in members.items() if ix}
    missing = elements - set(out)
    if missing:
        raise MeasureError(f"no element named {', '.join(sorted(missing))} in the model")
    return out


def feature_point(boxes: dict[str, tuple[np.ndarray, np.ndarray]], feature: str) -> np.ndarray:
    m = _FEATURE.match(feature)
    if not m:
        raise MeasureError(f"{feature!r} is not ELEMENT:xmin|xmax,ymin|ymax,zmin|zmax")
    lo, hi = boxes[m["el"]]
    return np.array([(lo if m[a] == f"{a}min" else hi)[i] for i, a in enumerate("xyz")], dtype=np.float64)


@dataclass
class Sensitivity:
    """Where each named corner is (metres) and how it moves per unit of each dimension."""

    keys: list[str]
    base: np.ndarray  # (P, 3)
    jac: np.ndarray  # (P, 3, D): metres per unit of the dimension


def sensitivity(source: str, dims: list[Dimension], keys: list[str], features: list[str],
                lite_step_root: str | os.PathLike, step_frac: float = 0.05) -> Sensitivity:
    """Compile the script once as it is and once per dimension in `keys`, nudged by `step_frac` of its value
    (at least one unit), and read the named corners from each."""
    elements = {f.split(":", 1)[0] for f in features}
    by_key = {d.key: d for d in dims}
    base_boxes = corners(compile_script(source, lite_step_root), elements)
    base = np.array([feature_point(base_boxes, f) for f in features])
    jac = np.zeros((len(features), 3, len(keys)))
    for j, k in enumerate(keys):
        d = by_key[k]
        h = max(abs(d.value) * step_frac, 1.0)
        boxes = corners(compile_script(with_dimensions(source, dims, {k: d.value + h}), lite_step_root), elements)
        moved = np.array([feature_point(boxes, f) for f in features])
        jac[:, :, j] = (moved - base) / h
    return Sensitivity(list(keys), base, jac)


def _project(rvec: np.ndarray, tvec: np.ndarray, focal: float, w: int, h: int, pts: np.ndarray) -> np.ndarray:
    import cv2

    k = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1.0]])
    uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, tvec, k, None)
    return uv.reshape(-1, 2)


@dataclass
class Measurement:
    camera: PerspectiveCamera
    values: dict[str, float]  # every dimension in `keys`, measured or kept
    sigma: dict[str, float]  # one standard deviation, in the dimension's unit
    seen: dict[str, bool]  # False: no control point depends on it, so the value was kept
    residual_px: list[float]
    rms_px: float


@dataclass
class Picture:
    """One picture's control points: pixels, the features they name (rows of the sensitivity), its size."""

    px: np.ndarray  # (P, 2)
    rows: np.ndarray  # (P,) indices into Sensitivity.base
    w: int
    h: int


@dataclass
class Joint:
    cameras: list[PerspectiveCamera]
    values: dict[str, float]
    sigma: dict[str, float]
    seen: dict[str, bool]
    residual_px: list[list[float]]  # per picture, per point
    rms_px: list[float]


def solve(px: np.ndarray, sens: Sensitivity, current: dict[str, float], fixed: set[str], w: int, h: int,
          prior_frac: float = 0.5, iterations: int = 60) -> Measurement:
    """One picture: least squares on its camera (rotation, position, focal length) and the free dimensions."""
    px = np.asarray(px, dtype=np.float64)
    if len(px) != len(sens.base):
        raise MeasureError("one pixel per feature")
    j = solve_many([Picture(px, np.arange(len(px)), w, h)], sens, current, fixed, prior_frac, iterations)
    return Measurement(j.cameras[0], j.values, j.sigma, j.seen, j.residual_px[0], j.rms_px[0])


def solve_many(pics: list[Picture], sens: Sensitivity, current: dict[str, float], fixed: set[str],
               prior_frac: float = 0.5, iterations: int = 80) -> Joint:
    """Least squares on a camera per picture and ONE set of free dimensions.

    A dimension no point of any picture depends on is not solved for. The free ones carry a weak prior to their
    current value (`prior_frac` of the larger of the value and the model's typical size, as one standard
    deviation)."""
    if not fixed:
        raise MeasureError("hold at least one dimension: pictures cannot give scale")
    for k, pic in enumerate(pics):
        if len(pic.px) < 6:
            raise MeasureError(f"picture {k + 1}: measuring needs at least six control points")
    col = {k: i for i, k in enumerate(sens.keys)}
    used = np.unique(np.concatenate([pic.rows for pic in pics]))
    seen = {k: bool(np.abs(sens.jac[used][:, :, col[k]]).max() > 1e-9) for k in sens.keys}
    free = [k for k in sens.keys if k not in fixed and seen[k]]
    d0 = np.array([current[k] for k in free])
    jf = sens.jac[:, :, [col[k] for k in free]]
    # a share of the dimension, never less than that share of the model's typical size, so a coordinate that
    # happens to be 0 (a grid line on the axis) is not frozen; the solver works in these units (u = (d-d0)/sig)
    # because millimetres next to radians make the normal equations useless (the carport stalled at 55 px)
    typical = float(np.median([abs(v) for v in current.values() if abs(v) > 1.0] or [1000.0]))
    sig = np.maximum(np.abs(d0), typical) * prior_frac
    nc = 7 * len(pics)

    def project_all(v: np.ndarray) -> list[np.ndarray]:
        world = sens.base + jf @ (v[nc:] * sig)
        return [_project(v[7 * k:7 * k + 3], v[7 * k + 3:7 * k + 6], math.exp(v[7 * k + 6]), pic.w, pic.h,
                         world[pic.rows]) for k, pic in enumerate(pics)]

    def resid(v: np.ndarray) -> np.ndarray:
        return np.concatenate([(uv - pic.px).ravel() for uv, pic in zip(project_all(v), pics)] + [v[nc:]])

    x0 = np.concatenate([_camera_vector(fit_perspective(sens.base[pic.rows], pic.px, pic.w, pic.h).camera)
                         for pic in pics] + [np.zeros(len(free))])
    x, r, jac = _levenberg_marquardt(resid, x0, iterations)
    n_px = sum(len(pic.px) for pic in pics)
    pix = r[:2 * n_px].reshape(-1, 2)
    sd = _spread(jac, pix, nc + len(free))[nc:] * sig
    values = {k: current[k] for k in sens.keys}
    sigma = dict.fromkeys(sens.keys, 0.0)
    for i, k in enumerate(free):
        values[k] = float(d0[i] + x[nc + i] * sig[i])
        sigma[k] = float(sd[i])
    errs = [np.linalg.norm(uv - pic.px, axis=1) for uv, pic in zip(project_all(x), pics)]
    return Joint([_camera_from_vector(x[7 * k:7 * k + 7], pic.w, pic.h) for k, pic in enumerate(pics)], values,
                 sigma, seen, [e.round(2).tolist() for e in errs], [float(np.sqrt((e ** 2).mean())) for e in errs])


def structure_variants(sens: Sensitivity, current: dict[str, float], share: float = 0.2) -> list[np.ndarray]:
    """The named corners as the model has them, and once more per dimension changed by `share` of its value (at
    least 100 units): what holds in all of them holds for any sizes, i.e. it is structure, not a guess."""
    out = [sens.base]
    for j, k in enumerate(sens.keys):
        out.append(sens.base + sens.jac[:, :, j] * max(abs(current[k]) * share, 100.0))
    return out


def _collinear(world: np.ndarray, a: int, b: int, c: int, tol: float) -> bool:
    d = world[c] - world[a]
    span = float(np.linalg.norm(d))
    if span < 1e-6:
        return False
    t = float((world[b] - world[a]) @ d) / span ** 2
    return 0.05 < t < 0.95 and float(np.linalg.norm(world[a] + t * d - world[b])) <= tol * span


def label_checks(worlds: list[np.ndarray], px: np.ndarray, features: list[str], tol_world: float = 0.01,
                 tol_px_frac: float = 0.03, min_px: float = 4.0, limit: int = 10) -> list[dict]:
    """Points on one straight line in the model must be on one line in the picture (a projection keeps lines).

    For every three points of three different elements that are collinear in every one of `worlds` (the middle
    one within `tol_world` of the span from the line through the other two, and between them; see
    `structure_variants`, so a line the guessed sizes make by chance does not count): the middle one's distance
    in the picture from the line through the other two. More than `tol_px_frac` of their span and `min_px`, and
    one of the three is mislabelled; so it is when the middle one is not between the other two in the picture
    (``outOfOrder``: a projection keeps the order along a line). Out of order first, then the farthest off the
    line, at most `limit`."""
    els = [f.split(":", 1)[0] for f in features]
    out = []
    n = len(features)
    for a in range(n):
        for c in range(a + 1, n):
            e = px[c] - px[a]
            span_px = float(np.linalg.norm(e))
            if span_px < 1e-6 or els[a] == els[c]:
                continue
            for b in range(n):
                if b in (a, c) or els[b] in (els[a], els[c]):
                    continue
                if not all(_collinear(w, a, b, c, tol_world) for w in worlds):
                    continue
                # a projection keeps the order along a line too: labels swapped within a row stay on the line
                s = float((px[b] - px[a]) @ e) / span_px ** 2
                if not -0.02 < s < 1.02:
                    out.append({"features": [features[a], features[b], features[c]], "offLinePx": 0.0,
                                "outOfOrder": True})
                    continue
                off = abs(float(e[0] * (px[b][1] - px[a][1]) - e[1] * (px[b][0] - px[a][0]))) / span_px
                if off > max(min_px, tol_px_frac * span_px):
                    out.append({"features": [features[a], features[b], features[c]], "offLinePx": round(off, 1),
                                "outOfOrder": False})
    out.sort(key=lambda r: (not r["outOfOrder"], -r["offLinePx"]))
    return out[:limit]


def _camera_vector(cam: PerspectiveCamera) -> np.ndarray:
    """(rvec, tvec, log focal) of a camera, OpenCV convention: rows of R are camera x (right), y (down), z."""
    import cv2

    right = np.cross(np.asarray(cam.direction), np.asarray(cam.up))
    rot = np.array([right, -np.asarray(cam.up), np.asarray(cam.direction)])
    rvec, _ = cv2.Rodrigues(rot)
    return np.concatenate([rvec.ravel(), (-rot @ np.asarray(cam.position)).ravel(), [math.log(cam.focal_px)]])


def _camera_from_vector(x: np.ndarray, w: int, h: int) -> PerspectiveCamera:
    import cv2

    rot, _ = cv2.Rodrigues(x[0:3])
    eye = -rot.T @ x[3:6]
    fov = math.degrees(2 * math.atan((h / 2.0) / math.exp(x[6])))
    return PerspectiveCamera(tuple(eye.tolist()), tuple(rot[2].tolist()), tuple((-rot[1]).tolist()), fov, w, h)


def _jacobian(resid, x: np.ndarray, r: np.ndarray) -> np.ndarray:
    jac = np.empty((len(r), len(x)))
    for i in range(len(x)):
        step = 1e-6 * max(1.0, abs(x[i]))
        xp = x.copy()
        xp[i] += step
        jac[:, i] = (resid(xp) - r) / step
    return jac


def _levenberg_marquardt(resid, x: np.ndarray, iterations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimise |resid(x)|^2; returns x, the residual there and the jacobian last used."""
    lam = 1e-3
    r = resid(x)
    jac = _jacobian(resid, x, r)
    for _ in range(iterations):
        jac = _jacobian(resid, x, r)
        a, g = jac.T @ jac, jac.T @ r
        improved = False
        dx = np.zeros_like(x)
        while lam <= 1e9 and not improved:
            dx = np.linalg.lstsq(a + lam * np.diag(np.diag(a) + 1e-12), -g, rcond=None)[0]
            rn = resid(x + dx)
            if rn @ rn < r @ r:
                x, r, lam, improved = x + dx, rn, max(lam / 3, 1e-9), True
            else:
                lam *= 4
        if not improved or np.linalg.norm(dx) < 1e-10 * (1 + np.linalg.norm(x)):
            break
    return x, r, jac


def _spread(jac: np.ndarray, pix: np.ndarray, n_params: int) -> np.ndarray:
    """One standard deviation per parameter from the jacobian and the pixel residuals (inf if singular)."""
    dof = max(1, pix.size - n_params)
    s2 = float(pix.ravel() @ pix.ravel()) / dof
    try:
        return np.sqrt(np.clip(np.diag(np.linalg.inv(jac.T @ jac)) * s2, 0, None))
    except np.linalg.LinAlgError:
        return np.full(jac.shape[1], float("inf"))


def measure(source: str, points: list[dict], fixed: list[str], image_w: int, image_h: int,
            lite_step_root: str | os.PathLike, keys: list[str] | None = None, drop: bool = True) -> dict:
    """The whole step on one picture: `points` are ``[{"px": [x, y], "feature": "ELEMENT:xmin,ymax,zmin"}]``.

    Returns the measured dimensions, ready to become an edit for Gróa: each with its old value, the measured
    one, one standard deviation, and whether the picture saw it at all."""
    r = measure_many(source, [{"name": "", "points": points, "w": image_w, "h": image_h}], fixed, lite_step_root,
                     keys, drop)
    p = r["pictures"][0]
    return {"dimensions": r["dimensions"], "verdict": p["verdict"], "camera": p["camera"], "rmsPx": p["rmsPx"],
            "poorFitPx": p["poorFitPx"],
            "residualPx": p["residualPx"], "dropped": p["dropped"], "labelChecks": p["labelChecks"],
            "edited": r["edited"]}


def _drop_next(errs: list[list[float]], keep: list[np.ndarray], budget: list[int],
               floor_px: float) -> tuple[int, int] | None:
    """The point to drop next, (picture, position among its kept points), or None: the worst error of all
    pictures, when it is above `floor_px` and three times its picture's median, and that picture may still drop
    one (a budget left, and more than six points)."""
    best = None
    for k, e in enumerate(errs):
        if budget[k] <= 0 or len(keep[k]) <= 6:
            continue
        e = np.asarray(e)
        i = int(np.argmax(e))
        if e[i] > max(floor_px, 3.0 * float(np.median(e))) and (best is None or e[i] > best[2]):
            best = (k, i, float(e[i]))
    return None if best is None else (best[0], best[1])


def measure_many(source: str, pictures: list[dict], fixed: list[str], lite_step_root: str | os.PathLike,
                 keys: list[str] | None = None, drop: bool = True, floor_px: float = 10.0) -> dict:
    """Several pictures of one scene, ``[{"name", "points": [{"px", "feature"}], "w", "h"}]``: one set of
    dimensions, a camera per picture.

    Returns the measured dimensions (old value, measured, one standard deviation, seen by any picture) and per
    picture: its camera, every point's error under the final fit (dropped points too, so the reason shows), the
    points dropped, and the label checks; with the edited script."""
    if not pictures:
        raise MeasureError("no pictures")
    dims = read_dimensions(source)
    known = {d.key for d in dims}
    if not dims:
        raise MeasureError("the script names no dimensions (UPPER_CASE module-level numbers)")
    for k in fixed:
        if k not in known:
            raise MeasureError(f"{k} is not a dimension of the script ({', '.join(sorted(known)[:12])}...)")
    keys = keys or [d.key for d in dims]
    unknown = [k for k in keys if k not in known]
    if unknown:
        raise MeasureError(f"not dimensions of the script: {', '.join(unknown)}")
    current = {d.key: d.value for d in dims}
    feats: list[str] = []
    for pic in pictures:
        for pt in pic["points"]:
            if pt["feature"] not in feats:
                feats.append(pt["feature"])
    sens = sensitivity(source, dims, [k for k in keys if k not in set(fixed)], feats, lite_step_root)
    # fixed dimensions sit outside the jacobian; the solver only needs their names to know scale is pinned
    sens_all = Sensitivity(sens.keys + list(fixed), sens.base,
                           np.concatenate([sens.jac, np.zeros((len(feats), 3, len(fixed)))], axis=2))
    full = [Picture(np.array([p["px"] for p in pic["points"]], dtype=np.float64),
                    np.array([feats.index(p["feature"]) for p in pic["points"]]), int(pic["w"]), int(pic["h"]))
            for pic in pictures]
    variants = structure_variants(sens, current)
    checks = [label_checks([v[f.rows] for v in variants], f.px, [p["feature"] for p in pic["points"]])
              for f, pic in zip(full, pictures)]
    keep = [np.arange(len(f.px)) for f in full]
    budget = [len(f.px) // 5 if drop else 0 for f in full]
    dropped: list[list[dict]] = [[] for _ in full]
    while True:
        m = solve_many([Picture(f.px[kp], f.rows[kp], f.w, f.h) for f, kp in zip(full, keep)], sens_all, current,
                       set(fixed))
        nxt = _drop_next(m.residual_px, keep, budget, floor_px)
        if nxt is None:
            break
        k, i = nxt
        idx = int(keep[k][i])
        pt = pictures[k]["points"][idx]
        dropped[k].append({"index": idx, "feature": pt["feature"], "px": pt["px"], "residualPx": m.residual_px[k][i]})
        keep[k] = np.delete(keep[k], i)
        budget[k] -= 1
    by_key = {d.key: d for d in dims}
    rows = []
    for k in sens_all.keys:
        was, now = current[k], m.values[k]
        if by_key[k].is_int:
            now = float(int(round(now)))
        row = {"key": k, "was": was, "now": now, "sigma": round(m.sigma[k], 1),
               "seen": m.seen[k] if k not in fixed else True, "fixed": k in fixed}
        # a measured size whose spread is as large as half of it is a guess, not a measurement: it is flagged
        # and not written into the edited script (staging 2026-11-26: BAY_X 691 +- 525 mm, "verdict ok")
        row["uncertain"] = bool(row["seen"] and not row["fixed"] and m.sigma[k] > UNCERTAIN_FRACTION * max(abs(now), 1.0))
        rows.append(row)
    delta = np.array([m.values[k] - current[k] for k in sens_all.keys])
    world = sens_all.base + sens_all.jac @ delta
    out = []
    for f, pic, cam, kp, dr, ck, left in zip(full, pictures, m.cameras, keep, dropped, checks, budget):
        err = np.linalg.norm(cam.project(world[f.rows]) - f.px, axis=1)
        # dropping hides a wrong label as well as an occluded point (carport, labels one column off: six points
        # dropped and 5.4 px, better than the true labels' 6.7), so a failed line check or a used-up budget
        # makes the whole picture untrusted, however small its error. A check is explained by a point dropped
        # alone from its element (one pixel misread, an occluded foot); a wrong label moves a whole element.
        per_el: dict[str, int] = {}
        for d in dr:
            el = d["feature"].split(":", 1)[0]
            per_el[el] = per_el.get(el, 0) + 1
        lone = {d["feature"] for d in dr if per_el[d["feature"].split(":", 1)[0]] == 1}
        for c in ck:
            c["explained"] = any(f in lone for f in c["features"])
        rms = float(np.sqrt((err[kp] ** 2).mean()))
        # a fit whose points sit far from the model is not a measurement, whatever the labels: the model is wrong
        # where it is held (an element placed at fixed coordinates the photo disagrees with, a missing member),
        # or the points are. Staging 2026-11-26: 42 and 99 px came back as "ok".
        limit = max(POOR_FIT_PX, POOR_FIT_FRACTION * float(np.hypot(f.w, f.h)))
        verdict = ("labels_suspect" if any(not c["explained"] for c in ck)
                   else "too_many_dropped" if drop and dr and left == 0
                   else "poor_fit" if rms > limit else "ok")
        out.append({"name": pic.get("name", ""), "verdict": verdict, "camera": cam.to_contract(),
                    "rmsPx": round(rms, 2), "poorFitPx": round(limit, 1), "residualPx": err.round(2).tolist(),
                    "dropped": dr, "labelChecks": ck})
    return {"dimensions": rows, "pictures": out,
            "verdict": "ok" if all(p["verdict"] == "ok" for p in out) else "untrusted",
            "edited": with_dimensions(source, dims, {r["key"]: r["now"] for r in rows
                                                     if not r["fixed"] and r["seen"] and not r["uncertain"]})}


def main_measure(args) -> int:
    """CLI: ``python -m tools.survey measure --workdir DIR --script output.py (--picture PTS.json:W:H [--picture
    ...] | --points PTS.json --image-w W --image-h H) --fix KEY [--fix KEY] [--keys A,B] [--keep-all]
    [--write edited.py]``. Every path is inside the working directory. Prints the result without the script."""
    from .intake import confined

    with open(confined(args.workdir, args.script), encoding="utf-8") as fh:
        source = fh.read()
    specs = []
    for spec in args.picture or []:
        rest, _, h = spec.rpartition(":")
        path, _, w = rest.rpartition(":")
        if not path or not w.isdigit() or not h.isdigit():
            raise MeasureError(f"--picture {spec!r} is not POINTS.json:W:H")
        specs.append((path, int(w), int(h)))
    if args.points:
        if not (args.image_w and args.image_h):
            raise MeasureError("--points needs --image-w and --image-h")
        specs.append((args.points, args.image_w, args.image_h))
    if not specs:
        raise MeasureError("give --picture POINTS.json:W:H (repeatable), or --points with --image-w and --image-h")
    pictures = []
    for path, w, h in specs:
        with open(confined(args.workdir, path), encoding="utf-8") as fh:
            pictures.append({"name": path, "points": json.load(fh), "w": w, "h": h})
    root = Path(__file__).resolve().parents[2]
    res = measure_many(source, pictures, args.fix, root, args.keys.split(",") if args.keys else None,
                       drop=not args.keep_all)
    if args.write:
        with open(confined(args.workdir, args.write), "w", encoding="utf-8") as fh:
            fh.write(res["edited"])
    res.pop("edited")
    print(json.dumps(res))
    return 0
