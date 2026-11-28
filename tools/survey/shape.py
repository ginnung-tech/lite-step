"""Shape: where the model's outline differs from a picture's, in millimetres, and which elements make that edge.

The minions measure where an element IS (one rigid shift per element). A sculpture, a vault or any freeform mesh
is one element whose SHAPE is wrong, and a shift says nothing about it. This measures the outline instead,
through a fitted camera:

- the picture's silhouette: a drawing's ink with its outline closed and filled; a photo's not-sky;
- the model's silhouette: the drone's ID buffer through the same camera (optionally only some elements);
- per band of height (the camera's up), the outermost edge on each side of both, and the area each has where the
  other has none; runs of bands that are off the same way become findings, named by side (in world terms, east,
  west, ...) and by the elements that make the model's edge there.

Every number is measured, never seen: what an agent does with a finding ("the robe is 400 mm too wide on the east
side from 52 to 58 m") is its own decision. Orthographic cameras give exact millimetres; a perspective camera's
are at the model's median depth (`scaleNote` says so).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .camera import OrthoCamera, PerspectiveCamera, basis, camera_from_contract

TOL_MM = 100.0  # an edge off by less than this (or 1.5 px, whichever is more) is on the picture
BAND_M = 0.25
MIN_RUN_M = 0.5  # a finding spans at least this much height
NEAR_M = 3.0  # a part of the picture farther than this from the model is not measured (and is reported)
FRAME_SHARE = 0.9  # a part of the picture at least this share of its width is a frame or ground line


def leaf(label: str) -> str:
    """The readable head of a canonical name: `covering:left_arm:spatialelement:...` -> `covering:left_arm`."""
    return ":".join(label.split(":")[:2])


def _compass(vec: np.ndarray) -> str:
    """A world direction in words: the nearest of east/west/north/south/up/down."""
    names = [("east (+X)", (1, 0, 0)), ("west (-X)", (-1, 0, 0)), ("north (+Y)", (0, 1, 0)),
             ("south (-Y)", (0, -1, 0)), ("up (+Z)", (0, 0, 1)), ("down (-Z)", (0, 0, -1))]
    return max(names, key=lambda n: float(np.dot(vec, n[1])))[0]


def drawing_silhouette(gray: np.ndarray, *, threshold: int = 200, close_px: int = 3) -> np.ndarray:
    """A line drawing's filled outline: everything the picture's border cannot reach without crossing ink.

    Ink is anything darker than near-white (`threshold`): a survey sheet's outline is a thin anti-aliased line
    that a dark threshold loses. Gaps in it up to `close_px` are sealed by growing the ink by that much before the
    flood, and the filled shape is shrunk back by the same amount after, so its edge stays where it was drawn.
    Lines inside (drapery, joints) fill in. Background open to the outside (between a raised arm and the head,
    open above) stays open when the opening is wider than `close_px`; a hole the outline encloses all round fills,
    because a flood cannot tell it from drapery: mark such a hole with `excludePx` or measure it in a section."""
    import cv2

    r = max(0, int(close_px))
    ink = (gray < threshold).astype(np.uint8)
    if r:
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        ink = cv2.dilate(ink, disk)
    h, w = ink.shape
    reach = np.zeros((h + 2, w + 2), np.uint8)
    free = (1 - ink).copy()
    # flood the background from every border pixel that is not ink
    seeds = [(x, 0) for x in range(w)] + [(x, h - 1) for x in range(w)] + \
            [(0, y) for y in range(h)] + [(w - 1, y) for y in range(h)]
    for x, y in seeds:
        if free[y, x] == 1 and reach[y + 1, x + 1] == 0:
            cv2.floodFill(free, reach, (x, y), 2, flags=4 | (1 << 8))
    filled = (~reach[1:-1, 1:-1].astype(bool)).astype(np.uint8)
    if r:
        filled = cv2.erode(filled, disk)
    return filled.astype(bool)


def photo_silhouette(img_bgr: np.ndarray) -> np.ndarray:
    from .intake import structure_mask

    return structure_mask(img_bgr).astype(bool)


@dataclass
class Window:
    """The part of the picture measured, in the camera's (u, v) metres for ortho or pixels for perspective."""
    u0: float | None = None
    u1: float | None = None
    v0: float | None = None
    v1: float | None = None


def _scale(cam, depth: np.ndarray, hit: np.ndarray) -> tuple[float, float]:
    if isinstance(cam, OrthoCamera):
        return cam.k * cam.px_per_m, cam.px_per_m
    d = float(np.median(depth[hit])) if hit.any() else 10.0
    f = cam.focal_px / max(d, 1e-3)
    return f, f


def _window_mask(cam, shape: tuple[int, int], win: Window) -> np.ndarray:
    h, w = shape
    m = np.ones(shape, bool)
    if isinstance(cam, OrthoCamera):
        xs = cam.ax_px + cam.k * cam.px_per_m * np.array([win.u0 if win.u0 is not None else -1e9,
                                                          win.u1 if win.u1 is not None else 1e9])
        ys = cam.c_px - cam.px_per_m * np.array([win.v1 if win.v1 is not None else 1e9,
                                                 win.v0 if win.v0 is not None else -1e9])
    else:
        xs = np.array([win.u0 if win.u0 is not None else -1e9, win.u1 if win.u1 is not None else 1e9])
        ys = np.array([win.v1 if win.v1 is not None else -1e9, win.v0 if win.v0 is not None else 1e9])
    x0, x1 = int(max(0, np.floor(min(xs)))), int(min(w, np.ceil(max(xs))))
    y0, y1 = int(max(0, np.floor(min(ys)))), int(min(h, np.ceil(max(ys))))
    m[:] = False
    m[y0:y1, x0:x1] = True
    return m


def compare(picture: np.ndarray, ids: np.ndarray, depth: np.ndarray, cam: OrthoCamera | PerspectiveCamera,
            labels: list[str], *, selected: np.ndarray | None = None, window: Window | None = None,
            band_m: float = BAND_M, tol_mm: float = TOL_MM, min_run_m: float = MIN_RUN_M) -> dict[str, Any]:
    """The two silhouettes compared. `ids` is the drone's element buffer, `selected` a boolean per element (None:
    all). Returns the report's measurements (no file is written here)."""
    import cv2

    hit = ids >= 0
    model = hit if selected is None else hit & selected[np.clip(ids, 0, None)]
    win = _window_mask(cam, picture.shape, window or Window())
    pic = picture & win
    mod = model & win
    sx, sy = _scale(cam, depth, model)
    ortho = isinstance(cam, OrthoCamera)
    right, up, _ = basis(cam.direction, cam.up)
    tol_px = max(1.5, tol_mm / 1000.0 * sx)
    # Only what the model could be: picture parts that come within NEAR_M of it (lettering, a neighbouring figure
    # of the sheet, a tree in a photo are not the model's business). What is dropped is reported, never hidden.
    ignored_m2 = 0.0
    if mod.any() and pic.any():
        reach = int(np.ceil(NEAR_M * sx))
        near = cv2.dilate(mod.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(pic.astype(np.uint8), connectivity=8)
        keep = np.zeros(n, bool)
        keep[np.unique(lab[(near > 0) & pic])] = True
        keep[0] = False
        # a part as wide as the picture is its frame or a ground line, not the subject
        keep &= stats[:, cv2.CC_STAT_WIDTH] < FRAME_SHARE * picture.shape[1]
        kept = keep[lab]
        ignored_m2 = float((pic & ~kept).sum()) / (sx * sy)
        pic = kept

    inter, union = int((pic & mod).sum()), int((pic | mod).sum())
    iou = inter / union if union else 0.0
    edge_p = pic & ~cv2.erode(pic.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    edge_m = mod & ~cv2.erode(mod.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    contour: dict[str, float] = {}
    if edge_p.any() and edge_m.any():
        d_p = cv2.distanceTransform((~edge_p).astype(np.uint8), cv2.DIST_L2, 3)
        d_m = cv2.distanceTransform((~edge_m).astype(np.uint8), cv2.DIST_L2, 3)
        d = np.concatenate([d_p[edge_m], d_m[edge_p]]) / sx * 1000.0
        contour = {"meanMm": round(float(d.mean()), 1), "p90Mm": round(float(np.percentile(d, 90)), 1),
                   "maxMm": round(float(d.max()), 1)}

    # bands of rows, top to bottom of whatever either silhouette covers
    rows = np.nonzero((pic | mod).any(axis=1))[0]
    bands: list[dict[str, Any]] = []
    if len(rows):
        step = max(1, int(round(band_m * sy)))
        for y0 in range(int(rows.min()), int(rows.max()) + 1, step):
            y1 = y0 + step
            p, m = pic[y0:y1], mod[y0:y1]
            ycen = (y0 + y1) / 2.0
            v = float(cam.to_uv([[0.0, ycen]])[0, 1]) if ortho else None
            band: dict[str, Any] = {"rowsPx": [y0, y1]}
            if v is not None:
                band["vM"] = round(v, 3)
            pc, mc = np.nonzero(p.any(axis=0))[0], np.nonzero(m.any(axis=0))[0]
            band["picture"] = [int(pc.min()), int(pc.max())] if len(pc) else None
            band["model"] = [int(mc.min()), int(mc.max())] if len(mc) else None
            if len(pc) and len(mc):
                # positive: the model sticks out past the picture on that side
                band["outMm"] = [round((pc.min() - mc.min()) / sx * 1000.0, 1), round((mc.max() - pc.max()) / sx * 1000.0, 1)]
                ys, xs = np.nonzero(m)
                left_ids = [int(ids[y0 + y, x]) for y, x in zip(ys, xs) if x == mc.min()]
                right_ids = [int(ids[y0 + y, x]) for y, x in zip(ys, xs) if x == mc.max()]
                band["edgeElements"] = [left_ids[0] if left_ids else -1, right_ids[0] if right_ids else -1]
            band["extraM2"] = round(float((m & ~p).sum()) / (sx * sy), 4)
            band["missingM2"] = round(float((p & ~m).sum()) / (sx * sy), 4)
            bands.append(band)

    findings = _findings(bands, sx, sy, tol_px, min_run_m, right, labels)
    for b in bands:  # element indices become names in the file
        if "edgeElements" in b:
            b["edgeElements"] = [labels[i] if i >= 0 else None for i in b["edgeElements"]]
    out: dict[str, Any] = {
        "iou": round(iou, 4), "contour": contour, "pictureM2": round(float(pic.sum()) / (sx * sy), 2),
        "modelM2": round(float(mod.sum()) / (sx * sy), 2), "ignoredPictureM2": round(ignored_m2, 2), "mmPerPx": [round(1000.0 / sx, 2), round(1000.0 / sy, 2)],
        "toleranceMm": round(tol_px / sx * 1000.0, 1), "bandM": band_m, "findings": findings, "bands": bands,
        "sides": {"left": _compass(-right), "right": _compass(right)},
    }
    if not ortho:
        out["scaleNote"] = "perspective: millimetres at the model's median depth, so approximate away from it"
    return out


def _findings(bands, sx, sy, tol_px, min_run_m, right, labels) -> list[dict[str, Any]]:
    """Runs of bands off the same way on the same side; then whole runs the model lacks or adds."""
    out: list[dict[str, Any]] = []
    min_bands = 1
    if bands:
        height = (bands[0]["rowsPx"][1] - bands[0]["rowsPx"][0]) / sy
        min_bands = max(1, int(np.ceil(min_run_m / max(height, 1e-9))))
    tol_mm = tol_px / sx * 1000.0

    def where(run):
        a, b = run[0], run[-1]
        r: dict[str, Any] = {"rowsPx": [a["rowsPx"][0], b["rowsPx"][1]],
                             "heightM": round((b["rowsPx"][1] - a["rowsPx"][0]) / sy, 2)}
        if "vM" in a:
            r["vM"] = [round(min(a["vM"], b["vM"]), 2), round(max(a["vM"], b["vM"]), 2)]
        return r

    for side, idx, word in (("left", 0, _compass(-right)), ("right", 1, _compass(right))):
        run: list[dict] = []
        sign = 0

        def flush():
            if len(run) >= min_bands:
                vals = np.array([b["outMm"][idx] for b in run])
                elems: dict[int, int] = {}
                for b in run:
                    e = b["edgeElements"][idx]
                    if e >= 0:
                        elems[e] = elems.get(e, 0) + 1
                top = sorted(elems, key=lambda e: -elems[e])[:3]
                kind = "sticks_out" if sign > 0 else "falls_short"
                f = {"kind": kind, "side": word, **where(run), "meanMm": round(float(vals.mean()), 0),
                     "maxMm": round(float(vals[np.argmax(np.abs(vals))]), 0), "elements": [labels[e] for e in top]}
                f["text"] = _sentence(f, tol_mm)
                out.append(f)

        for b in bands:
            v = b.get("outMm", [0.0, 0.0])[idx] if b.get("outMm") else None
            s = 0 if v is None or abs(v) <= tol_mm else (1 if v > 0 else -1)
            if s != 0 and s == sign:
                run.append(b)
                continue
            flush()
            run, sign = ([b], s) if s != 0 else ([], 0)
        flush()

    for kind, test in (("missing", lambda b: b["picture"] is not None and b["model"] is None),
                       ("extra", lambda b: b["model"] is not None and b["picture"] is None)):
        run = []
        for b in bands + [None]:
            if b is not None and test(b):
                run.append(b)
                continue
            if len(run) >= min_bands:
                area = sum(x["missingM2" if kind == "missing" else "extraM2"] for x in run)
                f = {"kind": kind, **where(run), "areaM2": round(area, 2), "elements": []}
                f["text"] = _sentence(f, tol_mm)
                out.append(f)
            run = []
    # the outline area each finding gets wrong, in m2: a wide error over a tall run first
    for f in out:
        f["errorM2"] = round(f["areaM2"] if "areaM2" in f else abs(f["meanMm"]) / 1000.0 * f["heightM"], 2)
    out.sort(key=lambda f: -f["errorM2"])
    return out


def _sentence(f: dict, tol_mm: float) -> str:
    at = f"from {f['vM'][0]:.2f} to {f['vM'][1]:.2f} m" if "vM" in f else f"rows {f['rowsPx'][0]}-{f['rowsPx'][1]}"
    if f["kind"] == "missing":
        return f"{at}: the picture shows {f['areaM2']:.2f} m2 where the model has nothing"
    if f["kind"] == "extra":
        return f"{at}: the model has {f['areaM2']:.2f} m2 where the picture shows nothing"
    who = ", ".join(leaf(e) for e in f["elements"]) or "the model"
    how = "sticks out past" if f["kind"] == "sticks_out" else "falls short of"
    return (f"{at}, {f['side']} side: {who} {how} the picture's outline by {abs(f['meanMm']):.0f} mm on average "
            f"(at most {abs(f['maxMm']):.0f} mm; tolerance {tol_mm:.0f} mm)")


def overlay(img_bgr: np.ndarray, picture: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Both outlines over the picture: grey where they agree, red where only the model is, blue where only the
    picture is."""
    out = (img_bgr.astype(np.float32) * 0.45 + 140).clip(0, 255).astype(np.uint8)
    out[picture & model] = (170, 170, 170)
    out[model & ~picture] = (40, 40, 220)
    out[picture & ~model] = (220, 120, 30)
    return out


VIEW_MARGIN = 0.08  # another view whose IoU beats the chosen one by this much is very likely the right one


def _view_name(direction: np.ndarray) -> str:
    from .intake import VIEWS

    for name, (d, _) in VIEWS.items():
        if np.allclose(direction, d, atol=1e-6):
            return name
    return ",".join(f"{v:g}" for v in np.round(direction, 4))


def view_check(picture: np.ndarray, drone, cam: OrthoCamera, labels: list[str], chosen_iou: float, *,
               selected: np.ndarray | None = None, window: Window | None = None) -> dict[str, Any]:
    """The same picture against the model seen from the opposite side and from the two sides at right angles, at
    the same scale and height, centred where the chosen view centres the model. A drawing's title names a face of
    the SITE; `--view` names a face of the MODEL; when the model's axes are not the site's, the right view is not
    the titled one, and only the outlines can tell (a statue trial's EAST panel was the model's west view)."""
    import cv2

    _, up, fwd = basis(cam.direction, cam.up)
    win = _window_mask(cam, picture.shape, window or Window())
    n, lab, stats, _ = cv2.connectedComponentsWithStats((picture & win).astype(np.uint8), connectivity=8)
    if n <= 1:
        return {"chosen": {"view": _view_name(fwd), "iou": chosen_iou}, "alternatives": []}
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))  # the picture's subject: its largest part
    pic_cx = stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH] / 2.0

    def iou_centred(d: np.ndarray) -> float:
        """IoU of the model seen along `d`, slid sideways so its outline is centred on the picture's subject: each
        candidate gets its best placement, so the comparison is about shape, not about where a fit put it."""
        alt = OrthoCamera(tuple(np.asarray(d).tolist()), cam.up, cam.px_per_m, cam.k, cam.ax_px, cam.c_px,
                          cam.image_w, cam.image_h, cam.depth_range_m)
        f = drone.render(alt)
        m = f.element >= 0 if selected is None else (f.element >= 0) & selected[np.clip(f.element, 0, None)]
        cols = np.nonzero((m & win).any(axis=0))[0]
        if not len(cols):
            return 0.0
        shift = pic_cx - (cols.min() + cols.max() + 1) / 2.0
        alt = OrthoCamera(alt.direction, alt.up, alt.px_per_m, alt.k, alt.ax_px + shift, alt.c_px, alt.image_w,
                          alt.image_h, alt.depth_range_m)
        f = drone.render(alt)
        return compare(picture, f.element, f.depth, alt, labels, selected=selected, window=window)["iou"]

    chosen_best = max(chosen_iou, iou_centred(fwd))
    alts = {"opposite": -fwd, "turned left": np.cross(up, fwd), "turned right": np.cross(fwd, up)}
    out: dict[str, Any] = {"chosen": {"view": _view_name(fwd), "iou": chosen_iou, "iouCentred": round(chosen_best, 4)},
                           "alternatives": []}
    best = None
    for what, d in alts.items():
        iou = round(iou_centred(d), 4)
        row = {"view": _view_name(np.asarray(d)), "as": what, "iou": iou}
        out["alternatives"].append(row)
        if best is None or iou > best["iou"]:
            best = row
    if best is not None and best["iou"] > chosen_best + VIEW_MARGIN:
        ranked = ", ".join(f"{a['view']} {a['iou']:.2f}" for a in sorted(out["alternatives"], key=lambda a: -a["iou"]))
        out["warning"] = (f"--view {out['chosen']['view']} is likely the wrong view for this picture: IoU "
                          f"{chosen_best:.2f}, against {ranked}. A drawing's title names the site's face, not the "
                          "model's. Decide from what the figure shows (which side the arm, the door, the stair is on) "
                          "and refit; while the model is far from the picture the ranking says which view is wrong "
                          "better than which is right.")
    return out


def run_shape(workdir: str | Path, source_id: str, *, model_file: str = "output.ifc", camera_version: int | None = None,
              elements: list[str] | None = None, window: Window | None = None, band_m: float = BAND_M,
              tol_mm: float = TOL_MM, threshold: int = 200, close_m: float = 0.3) -> dict[str, Any]:
    """Measure one source's outline against the model; writes `survey/shape/{source}-v{camera}.json` and its
    overlay `.png`, and returns the report."""
    import cv2

    from .drone import Drone
    from .intake import CAMERAS, DRAWING_KINDS, IntakeError, _read, _read_image, _source, confined, survey_dir
    from .scene import Scene

    wd = Path(workdir).resolve()
    src = _source(wd, source_id)
    cams = [c for c in _read(survey_dir(wd) / CAMERAS, "cameras")["cameras"] if c["source"] == source_id]
    if camera_version is not None:
        cams = [c for c in cams if c["version"] == camera_version]
    cams = [c for c in cams if c.get("gate") == "passed"]
    if not cams:
        raise IntakeError(f"{source_id} has no camera that passed its gate: fit it first (add --points)")
    entry = max(cams, key=lambda c: c["version"])
    cam = camera_from_contract(entry["camera"])
    img = _read_image(confined(survey_dir(wd), src["file"]))
    if src["kind"] in DRAWING_KINDS:
        ppm = cam.px_per_m if isinstance(cam, OrthoCamera) else 50.0
        picture = drawing_silhouette(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), threshold=threshold,
                                     close_px=max(1, int(round(close_m * ppm))))
    else:
        picture = photo_silhouette(img)
    scene = Scene.from_ifc(confined(wd, model_file), cache_dir=survey_dir(wd) / ".cache")
    frame = Drone(scene).render(cam)
    labels = [e.label for e in scene.elements]
    selected = None
    if elements:
        selected = np.zeros(len(scene.elements), bool)
        for e in scene.elements:
            names = [e.label] + [scene.elements[a].label for a in scene.ancestors(e.index)]
            if any(n.endswith(s) or n == s for n in names for s in elements):
                selected[e.index] = True
        if not selected.any():
            raise IntakeError(f"no element's canonical name ends with any of {elements}")
    report = compare(picture, frame.element, frame.depth, cam, labels, selected=selected, window=window,
                     band_m=band_m, tol_mm=tol_mm)
    if isinstance(cam, OrthoCamera):
        drone = Drone(scene)
        report["viewCheck"] = view_check(picture, drone, cam, labels, report["iou"], selected=selected, window=window)
    model = frame.element >= 0 if selected is None else (frame.element >= 0) & selected[np.clip(frame.element, 0, None)]
    win = _window_mask(cam, picture.shape, window or Window())
    rel = f"shape/{source_id}-v{entry['version']}"
    png = confined(survey_dir(wd), rel + ".png")
    png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png), overlay(img, picture & win, model & win))
    report = {"version": "1.0", "source": source_id, "cameraVersion": entry["version"], "model": model_file,
              "elements": elements or [], "overlay": rel + ".png", **report}
    (survey_dir(wd) / (rel + ".json")).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report
