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

import hashlib
import json
import logging
import multiprocessing
import shutil
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

from .camera import OrthoCamera, PerspectiveCamera, basis, camera_from_contract
from .drone import Drone
from .scene import Scene

logger = logging.getLogger(__name__)

INK_STRUCTURE = 1  # class value in ink/{source}.png (contract survey 1.1)
OVERLAY_MAX_PX = 1600
_SAFE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./-]*$")


class InspectError(Exception):
    """A round that cannot be run or completed; the message names what is missing."""


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


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


def _picture_axes(xs: np.ndarray, ys: np.ndarray) -> list[tuple[float, float]]:
    """The picture directions a group's position can be measured along: across a member only (sliding a line
    along itself says nothing), both for a compact group."""
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(pts) <= 2:
        return [(1.0, 0.0), (0.0, 1.0)]
    evals, evecs = np.linalg.eigh(np.cov((pts - pts.mean(axis=0)).T))
    if evals[1] > 9.0 * max(evals[0], 1e-6):
        return [(float(evecs[0, 0]), float(evecs[1, 0]))]
    return [(1.0, 0.0), (0.0, 1.0)]


def _best_shift(near_ink: np.ndarray, ys: np.ndarray, xs: np.ndarray, reach: int) -> tuple[int, int, float, list]:
    """The pixel shift that puts most of a group's pixels on ink, the share it reaches, and the picture
    directions (unit, x right / y down) it was measured along: one for a member, two for a compact group.

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
    axes = [(1.0, 0.0), (0.0, 1.0)]
    if elongated:
        n = evecs[:, 0]
        axes = [(float(n[0]), float(n[1]))]  # along the member nothing was measured
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
        # then pixel by pixel around the coarse best, and the CENTRE of the highest plateau there: every
        # offset inside the tolerance band scores alike, so the first one found is off by up to the band
        if best[0] or best[1]:
            cx, cy = best[0], best[1]
            win = max(step, 2)
            fine = [(dx, dy, score(dx, dy)) for dy in range(cy - 2 * win, cy + 2 * win + 1)
                    for dx in range(cx - 2 * win, cx + 2 * win + 1)]
            top = max(f[2] for f in fine)
            plateau = [(dx, dy) for dx, dy, sc in fine if sc >= top - 1e-9]
            mx = int(round(sum(p[0] for p in plateau) / len(plateau)))
            my = int(round(sum(p[1] for p in plateau) / len(plateau)))
            best = (mx, my, score(mx, my))
    return (*best, axes)


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
    right, up_v, _ = basis(cam.direction, cam.up)
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
        # A drawing draws lines. An element thicker than the tolerance band is compared by its OUTLINE:
        # its filled interior is never near ink, however it is moved. Members (about as thin as the drawn
        # line) keep their fill, so the line-drawing results do not change.
        if cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(tol) + 1).any():
            outline = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
            if int(outline.sum()) >= 30:
                mask = outline
        support = float(near_ink[mask].mean())
        row: dict[str, Any] = {"element": scene.elements[g].label, "pixels": n, "support": round(support, 4)}
        gy, gx = np.nonzero(mask)
        row["measuredAxes"] = [[round(float(v), 6) for v in _unit((ax / sx) * right - (ay / sy) * up_v)]
                               for ax, ay in _picture_axes(gx, gy)]
        if support < 0.85:
            ys, xs = np.nonzero(mask)
            dx, dy, shifted, _ = _best_shift(near_ink.astype(np.float32), ys, xs, reach)
            if shifted > support + 0.15 and (dx or dy):
                row["shiftMm"] = [int(round(dx / sx * 1000)), int(round(-dy / sy * 1000))]
                row["supportShifted"] = round(shifted, 4)
                world = (dx / sx) * right - (dy / sy) * up_v  # metres, in the world
                row["shiftWorldMm"] = [int(round(v * 1000)) for v in world]
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
        # Placed at the depth of the model nearest along its rays (a window around the box twice its
        # size), never on a plane through the world origin: a building is rarely centred on it.
        pad_x, pad_y = max(w, 10), max(h, 10)
        near = frame.depth[max(0, y - pad_y):y + h + pad_y, max(0, x - pad_x):x + w + pad_x]
        near = near[np.isfinite(near)]
        depth = float(np.median(near)) if near.size else median_depth
        corners = np.array([[x, y + h], [x + w, y]], dtype=np.float64)
        if isinstance(cam, OrthoCamera):
            p = cam.to_world(corners, depth_m=depth)
        else:
            origin, dirs = cam.ray(corners)
            p = origin + dirs * depth
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


MIN_SHIFT_MM = 150.0  # below this a displacement is drawing noise, not a finding
AXIS_SEEN = 0.3  # a direction counts as seen when it carries this share of the strongest constraint


def solve_displacement(obs: list[tuple[str, str, list, list]], min_sources: int = 2) -> dict | None:
    """One element's 3D displacement from what several views measured.

    Each observation is (shot, source, axes, shiftWorldMm): a view measured the element's position only along
    `axes` (world unit vectors: two for a compact group, one across a member); a view that found no shift
    (the element sits on its ink) says the displacement is ZERO along its axes, which is evidence too. Each
    axis gives one linear constraint `axis . d = axis . shift`; all are solved together by least squares. A
    view the solution misses by more than its tolerance (150 mm, or 30 % of what it measured) is dropped,
    worst first, and the rest solved again. Directions no view constrains are UNKNOWN, never zero.

    `consensus` is `two_sources` when at least `min_sources` sources see the displacement's direction, else
    `single_source`. Views that contradict each other (one had to be dropped, and fewer than `min_sources`
    sources remain) give no finding at all.
    """
    keep, dropped = list(obs), 0
    A = b = resid = d = None
    while keep:
        rows, rhs, owner = [], [], []
        for k, (_, _, axes, world) in enumerate(keep):
            w = np.asarray(world, dtype=np.float64)
            for a in axes:
                a = np.asarray(a, dtype=np.float64)
                rows.append(a)
                rhs.append(float(a @ w))
                owner.append(k)
        A, b = np.asarray(rows), np.asarray(rhs)
        d, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = np.abs(A @ d - b)
        owner_a = np.asarray(owner)
        worst_k, worst_excess = None, 0.0
        for k in range(len(keep)):
            sel = owner_a == k
            excess = float(np.max(resid[sel] - np.maximum(MIN_SHIFT_MM, 0.3 * np.abs(b[sel]))))
            if excess > worst_excess:
                worst_k, worst_excess = k, excess
        if worst_k is None:
            break
        keep.pop(worst_k)
        dropped += 1
    sources = sorted({o[1] for o in keep})
    if not keep or (dropped and len(sources) < min_sources):
        return None
    norm = float(np.linalg.norm(d))
    if norm < MIN_SHIFT_MM:
        return None
    u = d / norm
    seeing = sorted({o[1] for o in keep if any(abs(float(np.dot(a, u))) >= 0.5 for a in o[2])})
    _, sv, vt = np.linalg.svd(A)
    seen = [i for i in range(len(sv)) if sv[i] >= AXIS_SEEN * sv[0]]
    unknown = [vt[i] for i in range(3) if i not in seen]
    return {"displacementMm": [int(round(v)) for v in d],
            "consensus": "two_sources" if len(seeing) >= min_sources else "single_source",
            "seenBy": seeing, "seenAxes": len(seen),
            "unknownAxes": [[round(float(v), 4) for v in vec] for vec in unknown],
            "sources": sources, "shots": sorted({o[0] for o in keep}),
            "agreeing": len(keep), "measured": len(obs), "residualMm": int(round(float(resid.max())))}


def source_roots(survey: Path) -> dict[str, str]:
    """source id -> the source it is independent as: its `parent` plate's root, else itself. Sub-figures of
    one plate (and tiles cut from it) are one source: they share a camera fit and a draughtsman."""
    p = survey / "sources.json"
    if not p.exists():
        return {}
    parent = {s["id"]: s.get("parent") for s in json.loads(p.read_text(encoding="utf-8"))["sources"]}

    def root(i: str) -> str:
        seen = set()
        while parent.get(i) and i not in seen:
            seen.add(i)
            i = parent[i]
        return i

    return {i: root(i) for i in parent}


def consensus(reports: list[dict], roots: dict[str, str] | None = None) -> dict:
    """Findings need two sources that see them. Shifts: one element's measurements from every shot (shifted or
    sitting on its ink), solved together in 3D (`solve_displacement`): views at right angles share only one
    axis, so their picture-plane shifts cannot be compared directly. Extra: groups at least two sources do not
    show. Missing: ink in the same 2 m world cell in at least two sources. Sources are counted by their
    root (`source_roots`): two tiles of one plate are not two witnesses."""
    roots = roots or {}
    src = lambda r: roots.get(r["source"], r["source"])  # noqa: E731
    obs: dict[str, list[tuple[str, str, list, list]]] = defaultdict(list)
    moved: set[str] = set()
    low: dict[str, set[str]] = defaultdict(set)
    cells: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    for r in reports:
        for g in r["groups"]:
            axes = g.get("measuredAxes")
            if axes and "shiftWorldMm" in g and g["supportShifted"] >= 0.6:
                obs[g["element"]].append((r["shot"], src(r), axes, g["shiftWorldMm"]))
                moved.add(g["element"])
            elif axes and g["support"] >= 0.85:  # sits on its ink: zero shift along what this view sees
                obs[g["element"]].append((r["shot"], src(r), axes, [0, 0, 0]))
            if g["support"] < 0.4 and g["pixels"] >= 200:
                low[g["element"]].add(src(r))
        for m in r["missing"]:
            lo = np.floor(np.asarray(m["worldMin"]) / 2).astype(int)
            hi = np.floor(np.asarray(m["worldMax"]) / 2).astype(int)
            for cx in range(lo[0], hi[0] + 1):
                for cy in range(lo[1], hi[1] + 1):
                    for cz in range(lo[2], hi[2] + 1):
                        cells[(cx, cy, cz)].add(src(r))
    shifts = []
    for name in sorted(moved):
        found = solve_displacement(obs[name])
        if found is not None:
            shifts.append({"element": name, **found})
    shifts.sort(key=lambda f: (f["consensus"] != "two_sources", -f["agreeing"]))
    extra = [{"element": n, "sources": sorted(v)} for n, v in sorted(low.items()) if len(v) >= 2]
    missing = [{"cellMin": [2 * c[0], 2 * c[1], 2 * c[2]], "sources": sorted(v)} for c, v in sorted(cells.items()) if len(v) >= 2]
    return {"shifts": shifts, "extra": extra, "missingCells": missing}


def scores(reports: list[dict], found: dict, roots: dict[str, str] | None = None) -> list[dict]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in reports:
        by_source[r["source"]].append(r)
    seen_in: dict[str, set[str]] = defaultdict(set)
    for r in reports:
        for g in r["groups"]:
            seen_in[g["element"]].add((roots or {}).get(r["source"], r["source"]))
    out = []
    for src, rs in sorted(by_source.items()):
        gpx = sum(g["pixels"] for r in rs for g in r["groups"])
        placed = sum(g["pixels"] for r in rs for g in r["groups"] if g["support"] >= 0.8)
        ink = sum(r["inkPx"] for r in rs)
        covered = sum(r["recall"] * r["inkPx"] for r in rs)
        own = (roots or {}).get(src, src)
        alone = sum(g["pixels"] for r in rs for g in r["groups"] if seen_in[g["element"]] == {own})
        open_n = sum(1 for k in ("shifts", "extra", "missingCells") for f in found[k]
                     if own in f["sources"] and f.get("consensus", "two_sources") == "two_sources")
        out.append({"source": src, "placement": round(placed / gpx, 4) if gpx else 0.0,
                    "coverage": round(covered / ink, 4) if ink else 0.0, "openFindings": open_n,
                    "unverified": round(alone / gpx, 4) if gpx else 1.0})
    return out


# -- the round ----------------------------------------------------------------------------------------------

_DRONE: Drone | None = None


def _init_worker(scene_file: str, nice: int) -> None:
    global _DRONE
    if nice and hasattr(os, "nice"):  # behind the build: the person's compile and agent stay responsive
        try:
            os.nice(nice)
        except OSError:
            pass
    _DRONE = Drone(Scene.load(scene_file))


def _pool_context():
    """Processes are started fresh, never forked: forking a process that holds C++ thread pools (Embree,
    OpenCV, OpenCascade) can deadlock. forkserver where the platform has it, else spawn."""
    methods = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")


def _run_shot(args: tuple[ShotContext, str, float]) -> dict:
    shot, out_dir, tol_m = args
    assert _DRONE is not None
    return diff_shot(_DRONE, shot, Path(out_dir), tol_m=tol_m)


def run_round(workdir: str | Path, request: dict, workers: int | None = None,
              cache_dir: str | Path | None = None, scene: Scene | None = None, nice: int = 10) -> dict:
    """Run the round a `roundRequest` asks for and return its `round.json`. Raises nothing for a bad round:
    a failure is written as `round.json` with `status: failed` and its error, and returned.

    The model is measured from a SNAPSHOT taken first: a build may write a newer version meanwhile, and a
    round never mixes two. `workers` defaults to one per core but one; `nice` lowers the pool's priority."""
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
        cache.mkdir(parents=True, exist_ok=True)
        snapshot = cache / f"model-{os.getpid()}-{n}.ifc"
        shutil.copyfile(model, snapshot)
        try:
            base["modelSha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            if scene is None:
                scene = Scene.from_ifc(snapshot, cache_dir=cache)
        finally:
            snapshot.unlink(missing_ok=True)
        scene_file = cache / f"round-{os.getpid()}-{n}.npz"
        workers = max(1, min(workers or max(1, (os.cpu_count() or 2) - 1), len(shots)))
        if workers == 1:
            global _DRONE
            _DRONE = Drone(scene)
            reports = [_run_shot((s, str(out_dir), tol)) for s in shots]
        else:
            scene.save(scene_file)
            try:
                with ProcessPoolExecutor(workers, mp_context=_pool_context(), initializer=_init_worker,
                                         initargs=(str(scene_file), nice)) as pool:
                    reports = list(pool.map(_run_shot, [(s, str(out_dir), tol) for s in shots], chunksize=1))
            finally:
                scene_file.unlink(missing_ok=True)
        for r in reports:
            _write_json(out_dir / "shots" / f"{r['shot']}.json", r)
        roots = source_roots(survey)
        found = consensus(reports, roots)
        _write_json(out_dir / "consensus.json", found)
        ink = sum(r["inkPx"] for r in reports)
        result = {**base, "scores": scores(reports, found, roots), "status": "done", "shotCount": len(reports),
                  "durationS": round(time.monotonic() - t0, 2),
                  "inkWeightedF1": round(sum(r["f1"] * r["inkPx"] for r in reports) / ink, 4) if ink else 0.0}
    except Exception as exc:  # noqa: BLE001 - every failure becomes a failed round, said out loud
        logger.exception("[survey.inspect] round %d failed", n)
        result = {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:2000],
                  "durationS": round(time.monotonic() - t0, 2)}
    _write_json(out_dir / "round.json", result)
    return result
