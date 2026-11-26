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
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
            continue
        name = stmt.targets[0].id
        if not _NAME.match(name):
            continue
        value = stmt.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "param"
                and value.args):
            value = value.args[0]
        items = [(name, value)]
        if isinstance(value, (ast.Tuple, ast.List)):
            items = [(f"{name}[{i}]", v) for i, v in enumerate(value.elts)]
        for key, node in items:
            num = _number(node)
            if num is None or node.end_lineno != node.lineno:
                continue
            out.append(Dimension(key, num[0], num[1], node.lineno, num[2], node.end_col_offset))
    return out


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
    script = work / "model.py"
    script.write_text(source, encoding="utf-8")
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


def solve(px: np.ndarray, sens: Sensitivity, current: dict[str, float], fixed: set[str], w: int, h: int,
          prior_frac: float = 0.5, iterations: int = 60) -> Measurement:
    """Least squares on camera (rotation, position, focal length) and the free dimensions.

    The free dimensions also carry a weak prior to their current value (`prior_frac` of it as one standard
    deviation), which only matters for a dimension no point depends on."""
    import cv2

    px = np.asarray(px, dtype=np.float64)
    if len(px) != len(sens.base):
        raise MeasureError("one pixel per feature")
    if not fixed:
        raise MeasureError("hold at least one dimension: one picture cannot give scale")
    if len(px) < 6:
        raise MeasureError("measuring needs at least six control points")
    col = {k: i for i, k in enumerate(sens.keys)}
    seen = {k: bool(np.abs(sens.jac[:, :, col[k]]).max() > 1e-9) for k in sens.keys}
    # a dimension no point depends on is not solved for at all: it keeps its value, and says so
    free = [k for k in sens.keys if k not in fixed and seen[k]]
    d0 = np.array([current[k] for k in free])
    jf = sens.jac[:, :, [col[k] for k in free]] if free else np.zeros((len(px), 3, 0))

    def world(dv: np.ndarray) -> np.ndarray:
        return sens.base + (jf @ (dv - d0)) if free else sens.base

    start = fit_perspective(sens.base, px, w, h)
    cam = start.camera
    f0 = cam.focal_px
    # rows: camera x (right), y (down), z (forward) in world terms (OpenCV)
    right = np.cross(np.asarray(cam.direction), np.asarray(cam.up))
    rot = np.array([right, -np.asarray(cam.up), np.asarray(cam.direction)])
    rvec, _ = cv2.Rodrigues(rot)
    tvec = -rot @ np.asarray(cam.position)
    # one standard deviation of the prior: a share of the dimension, but never less than that share of the
    # model's typical size, so a coordinate that happens to be 0 (a grid line on the axis) is not frozen
    typical = float(np.median([abs(v) for v in current.values() if abs(v) > 1.0] or [1000.0]))
    sig = np.maximum(np.abs(d0), typical) * prior_frac
    # the dimensions are solved in units of their own prior width (u = (d - d0) / sig): millimetres next to
    # radians make the normal equations useless (the 2026-11-26 carport stalled at 55 px before this)
    x = np.concatenate([rvec.ravel(), tvec.ravel(), [math.log(f0)], np.zeros(len(free))])

    def resid(v: np.ndarray) -> np.ndarray:
        uv = _project(v[0:3], v[3:6], math.exp(v[6]), w, h, world(d0 + v[7:] * sig))
        return np.concatenate([(uv - px).ravel(), v[7:]])

    lam = 1e-3
    r = resid(x)
    for _ in range(iterations):
        jac = np.empty((len(r), len(x)))
        for i in range(len(x)):
            hstep = 1e-6 * max(1.0, abs(x[i]))
            xp = x.copy()
            xp[i] += hstep
            jac[:, i] = (resid(xp) - r) / hstep
        a = jac.T @ jac
        g = jac.T @ r
        while True:
            try:
                dx = np.linalg.solve(a + lam * np.diag(np.diag(a) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = x + dx
            rn = resid(xn)
            if rn @ rn < r @ r:
                x, r, lam = xn, rn, max(lam / 3, 1e-9)
                break
            lam *= 4
            if lam > 1e9:
                break
        if lam > 1e9 or np.linalg.norm(dx) < 1e-10 * (1 + np.linalg.norm(x)):
            break
    rv, tv, focal = x[0:3], x[3:6], math.exp(x[6])
    rot, _ = cv2.Rodrigues(rv)
    eye = -rot.T @ tv
    fov = math.degrees(2 * math.atan((h / 2.0) / focal))
    camera = PerspectiveCamera(tuple(eye.tolist()), tuple(rot[2].tolist()), tuple((-rot[1]).tolist()), fov, w, h)
    n_px = 2 * len(px)
    pix = r[:n_px].reshape(-1, 2)
    err = np.linalg.norm(pix, axis=1)
    dof = max(1, n_px - (7 + len(free)))
    s2 = float(pix.ravel() @ pix.ravel()) / dof
    try:
        cov = np.linalg.inv(jac.T @ jac) * s2
        sd = np.sqrt(np.clip(np.diag(cov)[7:], 0, None)) * sig
    except np.linalg.LinAlgError:
        sd = np.full(len(free), float("inf"))
    values = {k: current[k] for k in sens.keys}
    sigma = {k: 0.0 for k in sens.keys}
    for i, k in enumerate(free):
        values[k] = float(d0[i] + x[7 + i] * sig[i])
        sigma[k] = float(sd[i])
    return Measurement(camera, values, sigma, seen, err.round(2).tolist(), float(np.sqrt((err ** 2).mean())))


def measure(script: str | os.PathLike, points: list[dict], fixed: list[str], image_w: int, image_h: int,
            lite_step_root: str | os.PathLike, keys: list[str] | None = None) -> dict:
    """The whole step: `points` are ``[{"px": [x, y], "feature": "ELEMENT:xmin,ymax,zmin"}, ...]``.

    Returns the measured dimensions, ready to become an edit for Gróa: each with its old value, the measured
    one, one standard deviation, and whether the picture saw it at all."""
    source = Path(script).read_text(encoding="utf-8")
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
    feats = [p["feature"] for p in points]
    px = np.array([p["px"] for p in points], dtype=np.float64)
    sens = sensitivity(source, dims, [k for k in keys if k not in set(fixed)], feats, lite_step_root)
    # fixed dimensions sit outside the jacobian; the solver only needs their names to know scale is pinned
    sens_all = Sensitivity(sens.keys + list(fixed), sens.base,
                           np.concatenate([sens.jac, np.zeros((len(feats), 3, len(fixed)))], axis=2))
    m = solve(px, sens_all, current, set(fixed), image_w, image_h)
    by_key = {d.key: d for d in dims}
    rows = []
    for k in sens_all.keys:
        was, now = current[k], m.values[k]
        if by_key[k].is_int:
            now = float(int(round(now)))
        rows.append({"key": k, "was": was, "now": now, "sigma": round(m.sigma[k], 1),
                     "seen": m.seen[k] if k not in fixed else True, "fixed": k in fixed})
    return {"dimensions": rows, "camera": m.camera.to_contract(), "rmsPx": round(m.rms_px, 2),
            "residualPx": m.residual_px,
            "edited": with_dimensions(source, dims, {r["key"]: r["now"] for r in rows
                                                     if not r["fixed"] and r["seen"]})}


def main_measure(args) -> int:
    """CLI: ``python -m tools.survey measure --script model.py --points pts.json --image-w W --image-h H
    --fix KEY [--fix KEY] [--keys A,B] [--write edited.py]``. Prints the result without the edited script."""
    pts = json.loads(Path(args.points).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    res = measure(args.script, pts, args.fix, args.image_w, args.image_h, root,
                  args.keys.split(",") if args.keys else None)
    if args.write:
        Path(args.write).write_text(res["edited"], encoding="utf-8")
    res.pop("edited")
    print(json.dumps(res))
    return 0
