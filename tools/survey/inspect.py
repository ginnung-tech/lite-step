"""A round of minions: every shot of the survey measured against one model version.

Reads only the working directory (the model file the request names, and `survey/` with `sources.json`,
`cameras.json`, `shots.json` and the ink masks `ink/{source}.png`) and writes only
`survey/rounds/{n}/`: one report and one overlay per shot, `consensus.json`, and `round.json` LAST, whose
presence means the round ended. Shapes: Ginnung contract `survey` 1.1 (`roundRequest`, `shotReport`,
`round`).

Deterministic and CPU only: ray casting (Open3D), distance transforms (OpenCV), no model calls. A shot's
report is a measurement; a finding needs two sources that agree (`consensus`).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .camera import OrthoCamera, PerspectiveCamera, camera_from_contract
from .drone import Drone
from .scene import Scene

logger = logging.getLogger(__name__)

INK_STRUCTURE = 1  # class value in ink/{source}.png (contract survey 1.1)
OVERLAY_MAX_PX = 1600
_SAFE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")


class InspectError(Exception):
    """A round that cannot be run or completed; the message names what is missing."""


def _safe_rel(path: str, what: str) -> str:
    if not _SAFE.match(path) or ".." in path:
        raise InspectError(f"{what} must be a relative path inside the working directory without '..': {path!r}")
    return path


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


# -- the survey on disk -------------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotContext:
    """Everything a minion needs for one shot, picklable."""

    id: str
    source: str
    camera_version: int
    camera: dict  # contract camera, already cropped to the shot
    ink_file: str  # absolute path of ink/{source}.png
    crop: tuple[int, int, int, int]
    region: tuple[tuple[float, float], ...] | None  # polygon in crop pixels
    ignore: tuple[tuple[int, int, int, int], ...]


def load_shots(survey: Path, only: list[str] | None = None) -> list[ShotContext]:
    def read(name: str) -> dict:
        p = survey / name
        if not p.exists():
            raise InspectError(f"survey/{name} is missing")
        return json.loads(p.read_text(encoding="utf-8"))

    sources = {s["id"]: s for s in read("sources.json")["sources"]}
    cams = {(c["source"], c["version"]): c for c in read("cameras.json")["cameras"]}
    shots = read("shots.json")["shots"]
    if only is not None:
        want = set(only)
        unknown = want - {s["id"] for s in shots}
        if unknown:
            raise InspectError(f"shots not in shots.json: {sorted(unknown)}")
        shots = [s for s in shots if s["id"] in want]
    out = []
    for s in shots:
        src = sources.get(s["source"])
        if src is None:
            raise InspectError(f"shot {s['id']}: source {s['source']} is not in sources.json")
        if src.get("illustrative"):
            raise InspectError(f"shot {s['id']}: source {s['source']} is an illustration and is never measured")
        cam = cams.get((s["source"], s["cameraVersion"]))
        if cam is None:
            raise InspectError(f"shot {s['id']}: camera {s['source']} v{s['cameraVersion']} is not in cameras.json")
        if cam.get("gate") != "passed":
            raise InspectError(f"shot {s['id']}: camera {s['source']} v{s['cameraVersion']} has not passed its gate")
        ink = survey / "ink" / f"{s['source']}.png"
        if not ink.exists():
            raise InspectError(f"shot {s['id']}: ink mask survey/ink/{s['source']}.png is missing (made at intake)")
        x0, y0, x1, y1 = s["cropPx"]
        c = dict(cam["camera"])
        if c["kind"] == "ortho":
            c.update(axPx=c["axPx"] - x0, cPx=c["cPx"] - y0, imageW=x1 - x0, imageH=y1 - y0)
        elif (x0, y0, x1, y1) != (0, 0, c["imageW"], c["imageH"]):
            raise InspectError(f"shot {s['id']}: a perspective shot covers its whole picture (crop it at intake)")
        region = tuple(tuple(p) for p in s["regionPx"]) if s.get("regionPx") else None
        out.append(ShotContext(s["id"], s["source"], int(s["cameraVersion"]), c, str(ink), (x0, y0, x1, y1), region,
                               tuple(tuple(b) for b in s.get("ignorePx", []))))
    return out


# -- one shot ---------------------------------------------------------------------------------------------


def _best_shift(near_ink: np.ndarray, ys: np.ndarray, xs: np.ndarray, reach: int) -> tuple[int, int, float]:
    """The pixel shift that puts most of a group's pixels on ink, and the share it reaches.

    A long thin group (a member) is searched ACROSS its axis only: sliding a line along itself says
    nothing. Drawings repeat lines (the next crossbar is a panel away), so the NEAREST offset reaching 90 %
    of the best wins, not the far one that scores a little higher."""
    h, w = near_ink.shape
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(pts) > 2:
        evals, evecs = np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))
    else:
        evals, evecs = np.array([1.0, 1.0]), np.eye(2)
    elongated = evals[1] > 9.0 * max(evals[0], 1e-6)

    def score(dx: int, dy: int) -> float:
        x, y = xs + dx, ys + dy
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        return float(near_ink[y[ok], x[ok]].sum()) / len(xs) if ok.any() else 0.0

    best = (0, 0, score(0, 0))
    if elongated:
        n = evecs[:, 0]
        line: list[tuple[int, tuple[int, int], float]] = []  # (signed offset, pixel shift, score) across the member
        for d in range(-reach, reach + 1):
            key = (int(round(d * n[0])), int(round(d * n[1])))
            if not line or line[-1][1] != key:
                line.append((d, key, score(*key)))
        top = max(s for _, _, s in line)
        if top > best[2]:
            # The stretch of offsets scoring >= 90 % of the best that lies nearest to no shift; inside it, the
            # centre of its highest plateau. Every offset within the tolerance band scores alike, so taking the
            # stretch's nearest edge would under-report the shift by up to the tolerance.
            good = [i for i, (_, _, s) in enumerate(line) if s >= 0.9 * top]
            runs, run = [], [good[0]]
            for i in good[1:]:
                if i == run[-1] + 1:
                    run.append(i)
                else:
                    runs.append(run)
                    run = [i]
            runs.append(run)
            near = min(runs, key=lambda r: min(abs(line[i][0]) for i in r))
            peak = max(line[i][2] for i in near)
            plateau = [i for i in near if line[i][2] >= peak - 1e-9]
            i = plateau[len(plateau) // 2]
            best = (line[i][1][0], line[i][1][1], line[i][2])
    else:
        step = max(1, reach // 15)
        for dy in range(-reach, reach + 1, step):
            for dx in range(-reach, reach + 1, step):
                s = score(dx, dy)
                if s > best[2]:
                    best = (dx, dy, s)
    return best


def _scale(cam: OrthoCamera | PerspectiveCamera, depth: np.ndarray, hit: np.ndarray) -> tuple[float, float]:
    """Pixels per metre (horizontal, vertical) where the model is: exact for ortho, at the median depth for
    perspective."""
    if isinstance(cam, OrthoCamera):
        return cam.k * cam.px_per_m, cam.px_per_m
    d = float(np.median(depth[hit])) if hit.any() else 10.0
    f = cam.focal_px / max(d, 1e-3)
    return f, f


def diff_shot(drone: Drone, shot: ShotContext, out_dir: Path, *, tol_m: float = 0.25, max_shift_m: float = 6.0,
              group_depth: int = -2, min_missing_m2: float = 1.0) -> dict:
    import cv2

    cam = camera_from_contract(shot.camera)
    full = cv2.imread(shot.ink_file, cv2.IMREAD_UNCHANGED)
    if full is None:
        raise InspectError(f"shot {shot.id}: cannot read {shot.ink_file}")
    x0, y0, x1, y1 = shot.crop
    ink = full[y0:y1, x0:x1] == INK_STRUCTURE
    if shot.region:
        region = np.zeros(ink.shape, np.uint8)
        cv2.fillPoly(region, [np.asarray(shot.region, dtype=np.int32)], 1)
        region = region.astype(bool)
    else:
        region = np.ones(ink.shape, bool)
    ink &= region
    for a, b, c, d in shot.ignore:
        ink[b:d, a:c] = False

    frame = drone.render(cam)
    ids = frame.element
    model = ids >= 0
    sx, sy = _scale(cam, frame.depth, model)
    tol = max(1.0, tol_m * min(sx, sy))
    d_ink = cv2.distanceTransform((~ink).astype(np.uint8), cv2.DIST_L2, 3)
    d_model = cv2.distanceTransform((~model).astype(np.uint8), cv2.DIST_L2, 3)
    m_in = model & region
    precision = float((d_ink[m_in] <= tol).mean()) if m_in.any() else 0.0
    recall = float((d_model[ink] <= tol).mean()) if ink.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    scene = drone.scene
    near_ink = d_ink <= tol
    groups: dict[int, list[int]] = defaultdict(list)
    for e in np.unique(ids[m_in]).tolist():
        groups[scene.group_of(e, group_depth)].append(e)
    reach = int(max_shift_m * min(sx, sy))
    measured = []
    for g, members in groups.items():
        mask = np.isin(ids, members) & region
        n = int(mask.sum())
        if n < 30:
            continue
        support = float(near_ink[mask].mean())
        row: dict[str, Any] = {"element": scene.elements[g].label, "pixels": n, "support": round(support, 4)}
        if support < 0.85:
            ys, xs = np.nonzero(mask)
            dx, dy, shifted = _best_shift(near_ink.astype(np.float32), ys, xs, reach)
            if shifted > support + 0.15 and (dx or dy):
                row["shiftMm"] = [int(round(dx / sx * 1000)), int(round(-dy / sy * 1000))]
                row["supportShifted"] = round(shifted, 4)
        tri = np.isin(scene.tri_element, np.asarray(members, dtype=np.int32))
        pts = scene.vertices[np.unique(scene.triangles[tri])]
        row["worldMin"] = [round(float(v), 3) for v in pts.min(axis=0)]
        row["worldMax"] = [round(float(v), 3) for v in pts.max(axis=0)]
        measured.append(row)
    measured.sort(key=lambda r: -r["pixels"] * (1 - r["support"]))

    far = (ink & (d_model > 3 * tol)).astype(np.uint8)
    far = cv2.morphologyEx(far, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(far, 8)
    min_px = min_missing_m2 * sx * sy
    median_depth = float(np.median(frame.depth[model])) if model.any() else 0.0
    missing = []
    for k in range(1, n_lab):
        x, y, w, h, area = (int(v) for v in stats[k])
        if w * h < min_px or area < 40:
            continue
        corners = np.array([[x, y + h], [x + w, y]], dtype=np.float64)
        if isinstance(cam, OrthoCamera):
            p = cam.to_world(corners)
        else:
            origin, dirs = cam.ray(corners)
            p = origin + dirs * median_depth
        missing.append({"boxPx": [x + x0, y + y0, x + w + x0, y + h + y0], "areaPx": area,
                        "worldMin": [round(float(v), 3) for v in p.min(axis=0)],
                        "worldMax": [round(float(v), 3) for v in p.max(axis=0)]})
    missing.sort(key=lambda m: -m["areaPx"])

    # overlay: ink black; model green where ink supports it, red where it does not; missing boxed blue
    img = np.full((*ink.shape, 3), 255, np.uint8)
    img[ink] = (0, 0, 0)
    sup = model & near_ink
    img[model & ~sup] = (40, 40, 230)
    img[sup] = (60, 170, 60)
    for m in missing[:40]:
        a, b, c, d = m["boxPx"]
        cv2.rectangle(img, (a - x0, b - y0), (c - x0, d - y0), (230, 120, 0), 2)
    scale = min(1.0, OVERLAY_MAX_PX / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    overlay = out_dir / "overlays" / f"{shot.id}.webp"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(overlay), img, [cv2.IMWRITE_WEBP_QUALITY, 80])

    return {"shot": shot.id, "source": shot.source, "cameraVersion": shot.camera_version,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "modelPx": int(model.sum()), "inkPx": int(ink.sum()), "groups": measured[:300], "missing": missing[:40],
            "overlay": overlay.relative_to(out_dir.parent.parent).as_posix()}


# -- consensus and scores ------------------------------------------------------------------------------------


def consensus(reports: list[dict]) -> dict:
    """Findings need two sources. Shifts of one group agreed by shots of at least two sources (direction within
    30 degrees, length within 40 % of the median); groups at least two sources do not show; missing ink in the
    same 2 m cell in at least two sources."""
    obs: dict[str, list[tuple[str, str, tuple[int, int]]]] = defaultdict(list)
    low: dict[str, set[str]] = defaultdict(set)
    cells: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    for r in reports:
        for g in r["groups"]:
            if "shiftMm" in g and g["supportShifted"] >= 0.6:
                obs[g["element"]].append((r["shot"], r["source"], tuple(g["shiftMm"])))
            if g["support"] < 0.4 and g["pixels"] >= 200:
                low[g["element"]].add(r["source"])
        for m in r["missing"]:
            lo, hi = np.floor(np.asarray(m["worldMin"]) / 2).astype(int), np.floor(np.asarray(m["worldMax"]) / 2).astype(int)
            for cx in range(lo[0], hi[0] + 1):
                for cy in range(lo[1], hi[1] + 1):
                    for cz in range(lo[2], hi[2] + 1):
                        cells[(cx, cy, cz)].add(r["source"])
    shifts = []
    for name, o in obs.items():
        mr = sorted(v[2][0] for v in o)[len(o) // 2]
        mu = sorted(v[2][1] for v in o)[len(o) // 2]
        ml = math.hypot(mr, mu)
        if ml < 150:
            continue
        agree = [x for x in o if math.hypot(*x[2]) > 0
                 and abs((math.degrees(math.atan2(x[2][1], x[2][0]) - math.atan2(mu, mr)) + 180) % 360 - 180) <= 30
                 and abs(math.hypot(*x[2]) - ml) <= 0.4 * ml]
        srcs = sorted({x[1] for x in agree})
        if len(srcs) >= 2:
            shifts.append({"element": name, "shiftMm": [mr, mu], "sources": srcs, "shots": sorted({x[0] for x in agree}),
                           "agreeing": len(agree), "measured": len(o)})
    shifts.sort(key=lambda s: -s["agreeing"])
    extra = [{"element": n, "sources": sorted(s)} for n, s in sorted(low.items()) if len(s) >= 2]
    missing = [{"cellMin": [2 * c[0], 2 * c[1], 2 * c[2]], "sources": sorted(s)} for c, s in sorted(cells.items()) if len(s) >= 2]
    return {"shifts": shifts, "extra": extra, "missingCells": missing}


def scores(reports: list[dict], found: dict) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in reports:
        by_source[r["source"]].append(r)
    seen_in: dict[str, set[str]] = defaultdict(set)
    for r in reports:
        for g in r["groups"]:
            seen_in[g["element"]].add(r["source"])
    out = []
    for src, rs in sorted(by_source.items()):
        gpx = sum(g["pixels"] for r in rs for g in r["groups"])
        placed = sum(g["pixels"] for r in rs for g in r["groups"] if g["support"] >= 0.8)
        ink = sum(r["inkPx"] for r in rs)
        covered = sum(r["recall"] * r["inkPx"] for r in rs)
        alone = sum(g["pixels"] for r in rs for g in r["groups"] if seen_in[g["element"]] == {src})
        open_n = sum(1 for k in ("shifts", "extra", "missingCells") for f in found[k] if src in f["sources"])
        out.append({"source": src, "placement": round(placed / gpx, 4) if gpx else 0.0,
                    "coverage": round(covered / ink, 4) if ink else 0.0, "openFindings": open_n,
                    "unverified": round(alone / gpx, 4) if gpx else 1.0})
    return out


# -- the round ----------------------------------------------------------------------------------------------

_DRONE: Drone | None = None


def _init_worker(scene_file: str) -> None:
    global _DRONE
    _DRONE = Drone(Scene.load(scene_file))


def _run_shot(args: tuple[ShotContext, str, float]) -> dict:
    shot, out_dir, tol_m = args
    assert _DRONE is not None
    return diff_shot(_DRONE, shot, Path(out_dir), tol_m=tol_m)


def run_round(workdir: str | Path, request: dict, workers: int | None = None,
              cache_dir: str | Path | None = None, scene: Scene | None = None) -> dict:
    """Run the round a `roundRequest` asks for and return its `round.json`. Raises nothing for a bad round:
    a failure is written as `round.json` with `status: failed` and its error, and returned."""
    workdir = Path(workdir)
    survey = workdir / "survey"
    n = int(request["n"])
    out_dir = survey / "rounds" / str(n)
    if (out_dir / "round.json").exists():
        raise InspectError(f"round {n} already ran: rounds are never rewritten, ask for round {n + 1}")
    t0 = time.monotonic()
    base: dict[str, Any] = {"n": n, "modelVersion": int(request["modelVersion"]), "cameraVersions": {}, "at": _now(),
                            "scores": []}
    try:
        model = workdir / _safe_rel(request["modelFile"], "modelFile")
        if not model.exists():
            raise InspectError(f"model file {request['modelFile']} is not in the working directory")
        if not (out_dir / "request.json").exists():
            _write_json(out_dir / "request.json", request)
        shots = load_shots(survey, request.get("shots"))
        if not shots:
            raise InspectError("no shots to inspect")
        base["cameraVersions"] = {s.source: s.camera_version for s in shots}
        tol = float(request.get("tolM", 0.25))
        cache = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "survey-scene-cache"
        if scene is None:
            scene = Scene.from_ifc(model, cache_dir=cache)
        scene_file = cache / f"round-{os.getpid()}-{n}.npz"
        workers = max(1, min(workers or (os.cpu_count() or 1), len(shots)))
        if workers == 1:
            global _DRONE
            _DRONE = Drone(scene)
            reports = [_run_shot((s, str(out_dir), tol)) for s in shots]
        else:
            scene.save(scene_file)
            try:
                with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(scene_file),)) as pool:
                    reports = list(pool.map(_run_shot, [(s, str(out_dir), tol) for s in shots], chunksize=1))
            finally:
                scene_file.unlink(missing_ok=True)
        for r in reports:
            _write_json(out_dir / "shots" / f"{r['shot']}.json", r)
        found = consensus(reports)
        _write_json(out_dir / "consensus.json", found)
        ink = sum(r["inkPx"] for r in reports)
        result = {**base, "scores": scores(reports, found), "status": "done", "shotCount": len(reports),
                  "durationS": round(time.monotonic() - t0, 2),
                  "inkWeightedF1": round(sum(r["f1"] * r["inkPx"] for r in reports) / ink, 4) if ink else 0.0}
    except Exception as exc:  # noqa: BLE001 - every failure becomes a failed round, said out loud
        logger.exception("[survey.inspect] round %d failed", n)
        result = {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:2000],
                  "durationS": round(time.monotonic() - t0, 2)}
    _write_json(out_dir / "round.json", result)
    return result
