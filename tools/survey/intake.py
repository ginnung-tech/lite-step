"""Taking a picture into a survey: store it, fit a camera to it, show the fit back, or ask.

The Surveyor's duty is to fit every picture a person gives and show it back (spec: minion visual inspection,
3.0). These are the deterministic steps it calls; deciding what a picture is, and finding control points in a
photo, stay the Surveyor's own work (it writes scripts on this library for that).

- `add_source`: copy the picture into `survey/sources/`, hash it, list it in `sources.json`. The same bytes
  twice are one source.
- `fit_viewer`: a screenshot from our viewer comes with its camera: render the model through it and measure
  how well it lies on the picture. Exact when the canvas and the camera agree; when they do not, the gate
  fails and a question is asked instead of a guess.
- `fit_points`: a photo or another tool's screenshot, from control points (pixel and world, at least four).
- Every fit is a new camera version in `cameras.json` with its evidence (`overlay`, `agreement`), passed or
  failed. A failed fit leaves an `align` question in `questions/`.
- `write_index`: `index.json`, the Surveyor's files and their hashes, written last.

Shapes: the `survey` and `survey-question` contracts (version 1.2).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .camera import OrthoCamera, PerspectiveCamera
from .fit import fit_perspective

VERSION = "1.2"
TOL_PX = 2.0
#: a viewer screenshot passes when this share of the model's pixels lies within TOL_PX of the picture's
VIEWER_GATE = 0.80
#: a control-point fit passes when the mean residual is at most this many pixels
POINTS_GATE_PX = 4.0
#: a photo's fit also needs this share of its outline (holes closed) on the model's: the m13 lesson
OUTLINE_GATE = 0.85
#: a control point disagrees with the others when, fitted without it, the camera puts it this far away
DISAGREE_PX = 20.0
#: a photo whose background is mostly clear sky is measured against the sky, not its corners
SKY_SHARE = 0.05

PICTURE_KINDS = ("elevation", "plan", "section", "detail_plane", "perspective", "photo", "viewer_screenshot",
                 "screenshot", "sketch")
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SOURCES = "sources.json"
CAMERAS = "cameras.json"


class IntakeError(ValueError):
    """A picture that cannot be taken in as asked (unknown source, wrong kind, unreadable file)."""


def survey_dir(workdir: str | Path) -> Path:
    return Path(workdir).resolve() / "survey"


def confined(base: str | Path, rel: str | Path) -> Path:
    """`rel` under `base`, resolved; refused if it leaves `base` (a path read from a file or given on the
    command line never reaches outside the working directory)."""
    root = Path(base).resolve()
    p = (root / rel).resolve()
    if not p.is_relative_to(root):
        raise IntakeError(f"{rel} is outside {root.name}/")
    return p


def _check_id(source_id: str) -> str:
    if not _ID.match(source_id):
        raise IntakeError(f"source id {source_id!r} must match {_ID.pattern}")
    return source_id


def _read(path: Path, key: str) -> dict:
    if not path.exists():
        return {"version": VERSION, key: []}
    d = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d.get(key), list):
        raise IntakeError(f"{path.name} has no '{key}' list")
    return d


def _write(path: Path, obj: Any) -> None:
    """Whole file or nothing: a reader never sees half a table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9_-]+", "-", Path(name).stem.lower()).strip("-_")[:56]
    return s if s and _ID.match(s) else "source"


def _read_image(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IntakeError(f"{path.name} is not a picture OpenCV can read")
    return img


# --- cameras from and to BCF ---------------------------------------------------------------------------


def camera_from_bcfv(xml_text: str, image_w: int, image_h: int) -> dict:
    """The perspective camera of a BCF 2.1/3.0 viewpoint (.bcfv), as the survey contract's shape.

    BCF's FieldOfView is vertical, in degrees (our viewer writes it so; the bcf-issue contract). An
    orthogonal viewpoint is refused: our viewer's screenshots are perspective, and a guess would be wrong.
    """
    root = ET.fromstring(xml_text)
    pc = root.find("PerspectiveCamera")
    if pc is None:
        raise IntakeError("the viewpoint has no PerspectiveCamera")

    def vec(tag: str) -> list[float]:
        el = pc.find(tag)
        if el is None:
            raise IntakeError(f"the viewpoint's camera has no {tag}")
        return [float(el.findtext(a)) for a in ("X", "Y", "Z")]

    fov = float(pc.findtext("FieldOfView") or "nan")
    if not 0 < fov < 180:
        raise IntakeError(f"the viewpoint's FieldOfView {fov} is not a vertical angle in degrees")
    return {"kind": "perspective", "viewPoint": vec("CameraViewPoint"), "direction": vec("CameraDirection"),
            "upVector": vec("CameraUpVector"), "fieldOfView": fov, "imageW": int(image_w), "imageH": int(image_h)}


def camera_to_bcf(cam: dict) -> dict:
    """A survey camera as a BCF-API perspective_camera, so an issue can look through it.

    An orthographic survey camera becomes BCF's orthogonal_camera with the view-to-world scale (the height of
    the picture in metres).
    """
    def xyz(v: Iterable[float]) -> dict:
        x, y, z = (float(a) for a in v)
        return {"x": x, "y": y, "z": z}

    if cam["kind"] == "perspective":
        return {"perspective_camera": {"camera_view_point": xyz(cam["viewPoint"]),
                                       "camera_direction": xyz(cam["direction"]),
                                       "camera_up_vector": xyz(cam["upVector"]),
                                       "field_of_view": float(cam["fieldOfView"]),
                                       "aspect_ratio": cam["imageW"] / cam["imageH"]}}
    oc = OrthoCamera.from_contract(cam)
    centre = oc.to_world([[cam["imageW"] / 2.0, cam["imageH"] / 2.0]], depth_m=0.0)[0]
    eye = np.asarray(centre) - 1000.0 * np.asarray(cam["direction"], dtype=np.float64)
    return {"orthogonal_camera": {"camera_view_point": xyz(eye), "camera_direction": xyz(cam["direction"]),
                                  "camera_up_vector": xyz(cam["up"]),
                                  "view_to_world_scale": cam["imageH"] / cam["pxPerM"],
                                  "aspect_ratio": cam["imageW"] / cam["imageH"]}}


# --- sources -------------------------------------------------------------------------------------------


def add_source(workdir: str | Path, picture: str | Path, *, kind: str, title: str, now: str,
               convention: str | None = None, origin: str | None = None, added_by: str = "surveyor",
               source_id: str | None = None, viewer_camera: dict | None = None,
               illustrative: bool = False) -> dict:
    """Copy `picture` into survey/sources/ and list it. Returns the source entry (the existing one if these
    bytes are already a source)."""
    pic = Path(picture)
    if not pic.is_file():
        raise IntakeError(f"{pic} is not a file")
    if kind not in PICTURE_KINDS and kind != "text":
        raise IntakeError(f"kind {kind!r} is not a survey source kind")
    if kind == "viewer_screenshot" and viewer_camera is None:
        raise IntakeError("a viewer screenshot needs the viewer's camera")
    s = survey_dir(workdir)
    table = _read(s / SOURCES, "sources")
    digest = sha256_of(pic)
    for e in table["sources"]:
        if e.get("sha256") == digest:
            return e
    sid = _unique_id({e["id"] for e in table["sources"]}, _check_id(source_id or _slug(pic.name)))
    dest = confined(s, f"sources/{sid}{pic.suffix.lower() or '.bin'}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pic, dest)
    entry: dict[str, Any] = {
        "id": sid, "file": dest.relative_to(s).as_posix(), "sha256": digest, "kind": kind,
        "convention": "none" if kind == "text" else (convention or _default_convention(kind)),
        "title": (title or pic.name)[:300], "addedBy": added_by, "addedAt": now,
    }
    entry.update(_optional(origin, viewer_camera, illustrative))
    table["version"] = VERSION
    table["sources"].append(entry)
    _write(s / SOURCES, table)
    return entry


def _unique_id(taken: set[str], base: str) -> str:
    sid, n = base, 2
    while sid in taken:
        sid, n = f"{base[:60]}-{n}", n + 1
    return sid


def _optional(origin: str | None, viewer_camera: dict | None, illustrative: bool) -> dict:
    out: dict[str, Any] = {}
    if origin:
        out["origin"] = origin[:500]
    if viewer_camera is not None:
        out["viewerCamera"] = viewer_camera
    if illustrative:
        out["illustrative"] = True
    return out


def _default_convention(kind: str) -> str:
    return {"section": "section", "plan": "section", "detail_plane": "section", "photo": "shaded",
            "viewer_screenshot": "shaded", "screenshot": "shaded", "perspective": "shaded"}.get(kind, "front_slab")


def _source(workdir: Path, source_id: str) -> dict:
    for e in _read(survey_dir(workdir) / SOURCES, "sources")["sources"]:
        if e["id"] == source_id:
            return e
    raise IntakeError(f"no source {source_id!r} in sources.json")


# --- fits ----------------------------------------------------------------------------------------------


def _model_outline(hit: np.ndarray) -> np.ndarray:
    import cv2

    m = hit.astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)


def _save_overlay(workdir: Path, source_id: str, version: int, img: np.ndarray, hit: np.ndarray | None,
                  points: list[tuple[np.ndarray, np.ndarray]] | None = None, name: str | None = None) -> str:
    """The person's picture with the model's outline drawn on it (red), and control points: green where the
    picture says, a red ring where the camera puts them."""
    import cv2

    out = img.copy()
    if hit is not None:
        edge = _model_outline(hit)
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
        out[edge] = (40, 40, 220)
    for measured, projected in points or []:
        cv2.circle(out, tuple(int(round(v)) for v in measured), 5, (80, 170, 60), -1, cv2.LINE_AA)
        if np.all(np.isfinite(projected)):
            cv2.circle(out, tuple(int(round(v)) for v in projected), 9, (40, 40, 220), 2, cv2.LINE_AA)
    rel = f"sources/{_check_id(source_id)}.{name or f'fit-v{int(version)}'}.png"
    path = confined(survey_dir(workdir), rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), out):
        raise IntakeError(f"could not write {rel}")
    return rel


def _next_version(cams: dict, source_id: str) -> int:
    return 1 + max((c["version"] for c in cams["cameras"] if c["source"] == source_id), default=0)


def _record(workdir: Path, cam_entry: dict) -> None:
    s = survey_dir(workdir)
    cams = _read(s / CAMERAS, "cameras")
    cams["version"] = VERSION
    cams["cameras"].append(cam_entry)
    _write(s / CAMERAS, cams)


def _result(source: dict, cam_entry: dict) -> dict:
    """One source of the intake's answer (groa-cli SurveyIntakeResult)."""
    r: dict[str, Any] = {"id": source["id"], "sha256": source["sha256"],
                         "fit": "passed" if cam_entry["gate"] == "passed" else "failed",
                         "cameraVersion": cam_entry["version"], "residualMm": cam_entry["residualMm"],
                         "bcfCamera": camera_to_bcf(cam_entry["camera"])}
    if "overlay" in cam_entry:
        r["overlay"] = cam_entry["overlay"]
    if "agreement" in cam_entry:
        r["modelPxWithin2px"] = cam_entry["agreement"]["modelWithinPx"]
    return r


def structure_mask(img: np.ndarray) -> np.ndarray:
    """What is structure in a picture: everything that is not clear sky when the picture has sky (a photo),
    else everything that is not the flat background of its corners (a viewer screenshot)."""
    from .ink import sky_mask
    from .overlay import background_mask

    sky = sky_mask(img)
    return ~sky if sky.mean() >= SKY_SHARE else background_mask(img)


_LAST_TRIALS: list[tuple[float, float, int]] = []  # (rest residual, left-out error, index) of the last check


def check_points(points: list[dict], image_w: int, image_h: int, gate_px: float = POINTS_GATE_PX) -> dict | None:
    """Do the control points agree with each other? Leave each out in turn and fit the rest: the point whose
    removal leaves the others consistent is the doubt, when the camera they agree on puts it far from where it
    was clicked. Returns {"index", "errPx", "px", "world"}; {"index": None, ...} when no single removal makes
    the rest consistent (more than one doubt); None when they agree. Five points at least, six check better
    (four points fit almost anything).

    The m13 photo's failures were labels, not pixels: a corner of a symmetric structure named as another."""
    if len(points) < 5:
        return None
    world = np.array([p["world"] for p in points], dtype=np.float64)
    px = np.array([p["px"] for p in points], dtype=np.float64)
    limit = max(DISAGREE_PX, 5 * gate_px)
    trials = []
    for i in range(len(points)):
        keep = [j for j in range(len(points)) if j != i]
        try:
            fit = fit_perspective(world[keep], px[keep], image_w, image_h)
        except ValueError:
            continue
        q = fit.camera.project(world[i:i + 1])[0]
        err = float(np.hypot(*(q - px[i]))) if np.all(np.isfinite(q)) else float("inf")
        trials.append((fit.residual_px, err, i))
    if not trials:
        return None
    _LAST_TRIALS[:] = sorted(trials)
    rest, err, i = min(trials)  # the removal that leaves the others most consistent
    if rest > limit:
        return {"index": None, "errPx": float(rest), "px": None, "world": None}
    if err > limit and err > 3 * max(rest, 1.0):
        return {"index": i, "errPx": err, "px": px[i].tolist(), "world": world[i].tolist()}
    return None


def fit_viewer(workdir: str | Path, source_id: str, scene, *, now: str, gate: float = VIEWER_GATE,
               topic: str | None = None) -> dict:
    """Fit a viewer screenshot by its own camera. The camera is never changed here: if the model does not lie
    on the picture, the gate fails and an `align` question is asked."""
    from .drone import Drone
    from .overlay import agreement, background_mask, outline_agreement

    wd = Path(workdir)
    src = _source(wd, source_id)
    if src["kind"] != "viewer_screenshot":
        raise IntakeError(f"{source_id} is a {src['kind']}, not a viewer screenshot")
    img = _read_image(confined(survey_dir(wd), src["file"]))
    h, w = img.shape[:2]
    camd = dict(src["viewerCamera"], imageW=w, imageH=h)  # the picture's pixels are the truth about its size
    cam = PerspectiveCamera.from_contract(camd)
    hit = Drone(scene).render(cam).hit
    a = agreement(background_mask(img), hit, tol_px=TOL_PX)
    cams = _read(survey_dir(wd) / CAMERAS, "cameras")
    version = _next_version(cams, source_id)
    passed = bool(hit.any()) and a.model_within_px >= gate
    entry = {"source": source_id, "version": version, "camera": camd, "method": "viewer", "residualMm": 0.0,
             "gate": "passed" if passed else "failed", "frozenAt": now,
             "reason": (f"the viewer's own camera; {a.model_within_px:.0%} of model pixels within {TOL_PX:g} px"
                        if hit.any() else "the viewer's own camera sees none of the model"),
             "overlay": _save_overlay(wd, source_id, version, img, hit),
             "agreement": {"tolPx": TOL_PX, "modelWithinPx": round(a.model_within_px, 4),
                           "sourceWithinPx": round(a.source_within_px, 4),
                           "outlineWithinPx": round(outline_agreement(background_mask(img), hit, TOL_PX), 4)}}
    _record(wd, entry)
    if not passed:
        ask_align(wd, source_id, now=now, topic=topic)
    return _result(src, entry)


def fit_points(workdir: str | Path, source_id: str, points: list[dict], *, now: str, scene=None,
               gate_px: float = POINTS_GATE_PX, reason: str | None = None, topic: str | None = None) -> dict:
    """Fit a perspective camera to control points: `[{"px": [x, y], "world": [x, y, z], "by": ...}, ...]`,
    four at least, spread in depth where possible. With a `scene`, the overlay also draws the model and the fit
    must also put the picture's outline on the model's (`OUTLINE_GATE`).

    Five points or more are checked against each other first (`check_points`): when one disagrees, the fit
    is recorded as failed and the person is asked about THAT point, rather than searching on."""
    wd = Path(workdir)
    src = _source(wd, source_id)
    if src["kind"] == "text" or src.get("illustrative"):
        raise IntakeError(f"{source_id} is not a picture to fit")
    img = _read_image(confined(survey_dir(wd), src["file"]))
    h, w = img.shape[:2]
    world = np.array([p["world"] for p in points], dtype=np.float64)
    px = np.array([p["px"] for p in points], dtype=np.float64)
    _LAST_TRIALS.clear()
    doubt = check_points(points, w, h, gate_px)
    trials = list(_LAST_TRIALS)
    f = fit_perspective(world, px, w, h)
    cam = f.camera
    # a pixel's error in millimetres at the point's own distance
    dist = np.linalg.norm(world - np.asarray(cam.position), axis=1)
    residual_mm = float(np.mean(np.asarray(f.per_point_px) * dist / cam.focal_px) * 1000.0)
    projected = cam.project(world)
    hit = None
    agree = None
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
    version = _next_version(cams, source_id)
    passed = (f.residual_px <= gate_px and doubt is None
              and (agree is None or agree["outlineWithinPx"] >= OUTLINE_GATE))
    if doubt is not None and doubt["index"] is None:
        reason = ("the control points disagree with each other, and not one of them alone: more than one label "
                  "(which point on the model) is likely wrong")
    elif doubt is not None:
        reason = (f"control point {doubt['index'] + 1} disagrees with the others: fitted without it, the camera "
                  f"puts it {doubt['errPx']:.0f} px away. Its label (which point on the model) is the likely error.")
    elif agree is not None and agree["outlineWithinPx"] < OUTLINE_GATE:
        reason = (f"{len(points)} control points, PnP {f.residual_px:.1f} px, but only "
                  f"{agree['outlineWithinPx']:.0%} of the picture's outline lies on the model's")
    entry: dict[str, Any] = {
        "source": source_id, "version": version, "camera": cam.to_contract(), "method": "control_points",
        "controlPoints": [{"px": [float(v) for v in p["px"]], "world": [float(v) for v in p["world"]],
                           "by": p.get("by", "surveyor"), **({"note": p["note"][:300]} if p.get("note") else {})}
                          for p in points],
        "residualMm": round(residual_mm, 1), "gate": "passed" if passed else "failed", "frozenAt": now,
        "reason": (reason or f"{len(points)} control points, PnP; mean {f.residual_px:.1f} px")[:500],
        "overlay": _save_overlay(wd, source_id, version, img, hit, list(zip(px, projected))),
    }
    if agree is not None:
        entry["agreement"] = agree
    _record(wd, entry)
    if not passed:
        options = candidate_poses(wd, source_id, img, points, trials, scene, camera_version=version + 1)
        if len(options) >= 2:
            doubt_words = ""
            if doubt is not None and doubt["index"] is not None:
                x, y = (int(round(v)) for v in doubt["px"])
                doubt_words = f" I am least sure of the point I marked at ({x}, {y})."
            ask_align(wd, source_id, now=now, points=2, topic=topic, options=options, question=(
                "I could not be sure how your picture lines up with the model. Which of these looks right?"
                + doubt_words + " If none does, click a point on the picture and the same point on the model."))
        elif doubt is not None and doubt["index"] is not None:
            x, y = (int(round(v)) for v in doubt["px"])
            ask_align(wd, source_id, now=now, points=2, topic=topic, question=(
                f"I am not sure which point on the model I marked at ({x}, {y}) in your picture. Please click that "
                "point on the picture, then the same point on the model."))
        else:
            ask_align(wd, source_id, now=now, topic=topic)
    return _result(src, entry)


MAX_CANDIDATES = 3
DISTINCT_EYE_M = 2.0  # two candidate poses are the same pose when their eyes are closer than this ...
DISTINCT_TURN_DEG = 3.0  # ... and their views turn less than this
_COMPASS = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]


def pose_label(camera: dict) -> str:
    """A candidate in words: where the camera stands, seen from the model's origin (x east, y north)."""
    x, y, _ = camera.get("viewPoint", [0.0, 0.0, 0.0])
    ang = (np.degrees(np.arctan2(y, x)) + 360.0 + 22.5) % 360.0
    return f"seen from the {_COMPASS[int(ang // 45)]}, {np.hypot(x, y):.0f} m from the centre"


def candidate_poses(workdir: Path, source_id: str, img: np.ndarray, points: list[dict],
                    trials: list[tuple[float, float, int]], scene, *, camera_version: int) -> list[dict]:
    """The distinct poses the points allow: all of them, and each leave-one-out fit, ranked by how well the
    picture's outline lies on the model (by residual without a scene); at most three, each with its overlay
    (`sources/{id}.cand-{k}.png`). These are what the person is shown to pick from."""
    h, w = img.shape[:2]
    world = np.array([p["world"] for p in points], dtype=np.float64)
    px = np.array([p["px"] for p in points], dtype=np.float64)
    subsets = [list(range(len(points)))] + [[j for j in range(len(points)) if j != i] for _, _, i in trials]
    found: list[dict] = []
    for keep in subsets:
        if len(keep) < 4:
            continue
        try:
            fit = fit_perspective(world[keep], px[keep], w, h)
        except ValueError:
            continue
        cam = fit.camera
        fwd = np.asarray(cam.direction) / np.linalg.norm(cam.direction)
        if any(np.linalg.norm(np.subtract(cam.position, c["cam"].position)) < DISTINCT_EYE_M
               and np.degrees(np.arccos(np.clip(fwd @ c["fwd"], -1, 1))) < DISTINCT_TURN_DEG for c in found):
            continue
        outline = None
        hit = None
        if scene is not None:
            from .drone import Drone
            from .overlay import outline_agreement

            hit = Drone(scene).render(cam).hit
            outline = outline_agreement(structure_mask(img), hit, TOL_PX)
        found.append({"cam": cam, "fwd": fwd, "keep": keep, "residual": fit.residual_px, "outline": outline, "hit": hit})
    found.sort(key=lambda c: (-(c["outline"] or 0.0), c["residual"]))
    options = []
    for k, c in enumerate(found[:MAX_CANDIDATES], start=1):
        overlay = _save_overlay(workdir, source_id, camera_version, img, c["hit"],
                                list(zip(px[c["keep"]], c["cam"].project(world[c["keep"]]))), name=f"cand-{k}")
        camd = c["cam"].to_contract()
        pose: dict[str, Any] = {"camera": camd, "cameraVersion": camera_version,
                                "controlPoints": [{"px": [float(v) for v in px[j]], "world": [float(v) for v in world[j]]}
                                                  for j in c["keep"]]}
        if c["outline"] is not None:
            pose["outlineWithinPx"] = round(float(c["outline"]), 4)
        options.append({"id": f"pose-v{camera_version}-{k}", "label": pose_label(camd), "evidence": [overlay],
                        "pose": pose})
    return options


def _residual_mm(pose: dict) -> float:
    """The chosen camera's control-point error in millimetres at each point's distance (as fit_points)."""
    from .camera import camera_from_contract

    cam = camera_from_contract(pose["camera"])
    world = np.array([cp["world"] for cp in pose["controlPoints"]], dtype=np.float64)
    px = np.array([cp["px"] for cp in pose["controlPoints"]], dtype=np.float64)
    err = np.hypot(*(cam.project(world) - px).T)
    if not isinstance(cam, PerspectiveCamera):
        return round(float(np.mean(err)) / cam.px_per_m * 1000.0, 1)
    dist = np.linalg.norm(world - np.asarray(cam.position), axis=1)
    return round(float(np.mean(err * dist / cam.focal_px)) * 1000.0, 1)


def answer_align(workdir: str | Path, question_id: str, choice: str, *, now: str, words: str | None = None) -> dict:
    """The person picked a candidate pose: freeze it as the source's next camera (method control_points, the
    points the pose was fitted on), mark the question answered, and return the answer for that picture."""
    wd = Path(workdir)
    qpath = confined(survey_dir(wd), f"questions/{_check_id(question_id)}.json")
    if not qpath.exists():
        raise IntakeError(f"no question {question_id}")
    q = json.loads(qpath.read_text(encoding="utf-8"))
    if q.get("kind") != "align" or q.get("status", "open") != "open":
        raise IntakeError(f"question {question_id} is not an open align question")
    option = next((o for o in q.get("options", []) if o.get("id") == choice), None)
    if option is None or "pose" not in option:
        raise IntakeError(f"{choice!r} is not one of question {question_id}'s candidate poses")
    src = _source(wd, q["source"])
    pose = option["pose"]
    cams = _read(survey_dir(wd) / CAMERAS, "cameras")
    marker = f"({question_id}, {choice})"
    done = [c for c in cams["cameras"] if c["source"] == src["id"] and c.get("reason", "").endswith(marker)]
    if done:  # a retry after the question could not be closed: the camera is already frozen
        entry = done[-1]
    else:
        version = _next_version(cams, src["id"])
        # the evidence is frozen with the camera: candidate overlays are rewritten by the next failed fit
        frozen = f"sources/{_check_id(src['id'])}.fit-v{int(version)}.png"
        confined(survey_dir(wd), frozen).write_bytes(confined(survey_dir(wd), option["evidence"][0]).read_bytes())
        entry = {
            "source": src["id"], "version": version, "camera": pose["camera"], "method": "control_points",
            "controlPoints": [{"px": cp["px"], "world": cp["world"], "by": "surveyor"} for cp in pose["controlPoints"]],
            "residualMm": _residual_mm(pose), "gate": "passed", "frozenAt": now,
            "reason": f"chosen by the person from {len(q.get('options', []))} candidate poses {marker}",
            "overlay": frozen,
        }
        # No agreement block: only the outline share was measured for the candidate, and the camera's agreement
        # needs all its shares. The answer reports the outline share the person saw.
        _record(wd, entry)
    q["version"] = "1.2"
    q["status"] = "answered"
    q["answer"] = {"choice": choice, "words": (words or f"picked: {option['label']}")[:2000], "at": now}
    _write(qpath, q)
    out = _result(src, entry)
    if "outlineWithinPx" in pose:
        out["outlineWithin2px"] = pose["outlineWithinPx"]
    return out


def add_partials(workdir: str | Path, source_id: str, parts: list[dict], *, camera_version: int = 1) -> list[dict]:
    """One shot per detail of a picture: `[{"name", "box": [x0, y0, x1, y1], "purpose"}]`. The shot is the crop
    (a perspective camera follows it: `PerspectiveCamera.crop`); ids are `{source}-{name}`; a partial with the
    same id is replaced."""
    s = survey_dir(workdir)
    src = _source(Path(workdir), source_id)
    table = _read(s / "shots.json", "shots")
    new = []
    for p in parts:
        sid = _check_id(f"{src['id']}-{p['name']}"[:64])
        x0, y0, x1, y1 = (int(v) for v in p["box"])
        if not (0 <= x0 < x1 and 0 <= y0 < y1):
            raise IntakeError(f"partial {sid}: box {p['box']} is empty or negative")
        new.append({"id": sid, "source": src["id"], "cameraVersion": int(camera_version), "cropPx": [x0, y0, x1, y1],
                    "purpose": str(p.get("purpose") or p["name"])[:300]})
    ids = {n["id"] for n in new}
    table["version"] = VERSION
    table["shots"] = [t for t in table["shots"] if t["id"] not in ids] + new
    _write(s / "shots.json", table)
    return new


def write_photo_ink(workdir: str | Path, source_id: str) -> str:
    """ink/{source}.png for a photo: structure (1), openings (5: sky seen through the structure), nothing
    else; the source's `excludePx` boxes are left out. Returns the path inside survey/."""
    import cv2

    from .ink import photo_ink, sky_mask

    wd = Path(workdir)
    src = _source(wd, source_id)
    img = _read_image(confined(survey_dir(wd), src["file"]))
    ink = photo_ink(img, sky_mask(img), exclude_px=src.get("excludePx"))
    rel = f"ink/{_check_id(source_id)}.png"
    out = confined(survey_dir(wd), rel)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), ink):
        raise IntakeError(f"could not write {rel}")
    return rel


def ask_align(workdir: str | Path, source_id: str, *, now: str, question: str | None = None,
              points: int = 4, topic: str | None = None, options: list[dict] | None = None) -> dict:
    """An `align` question for a picture that could not be fitted: the person clicks `points` points on the
    picture and the same points on the model (contract survey-question)."""
    q = {"version": "1.0", "id": f"q-align-{_check_id(source_id)}"[:64], "kind": "align", "blocking": False, "askedAt": now,
         "question": question or (f"I could not line the model up with this picture. Please click {points} points "
                                  "on it, then the same points on the model."),
         "source": source_id, "points": points}
    if options:  # contract survey-question 1.2: candidate poses to pick from
        q.update(version="1.2", status="open", options=options)
    if topic:
        q["bcfTopic"] = topic
    _write(confined(survey_dir(workdir), f"questions/{q['id']}.json"), q)
    return q


def no_fit(workdir: str | Path, source_id: str) -> dict:
    """The answer for a source that is not fitted: text, or an illustrative picture."""
    src = _source(Path(workdir), source_id)
    fit = "illustrative" if src.get("illustrative") else "not_a_picture"
    return {"id": src["id"], "sha256": src["sha256"], "fit": fit}


# --- the index -----------------------------------------------------------------------------------------


def write_index(workdir: str | Path, *, now: str) -> dict:
    """survey/index.json: every file of the Surveyor's (not rounds/, which the minions write), with its size and
    hash. Write it last, after the files are kept."""
    s = survey_dir(workdir)
    files = []
    for rel, p in sorted((q.relative_to(s).as_posix(), q) for q in s.rglob("*")):
        if not p.is_file() or rel == "index.json" or rel.startswith("rounds/") or rel.endswith(".tmp"):
            continue
        files.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256_of(p)})
    idx = {"version": VERSION, "at": now, "files": files}
    _write(s / "index.json", idx)
    return idx


__all__ = ["IntakeError", "add_partials", "answer_align", "candidate_poses", "pose_label", "add_source", "ask_align", "camera_from_bcfv", "camera_to_bcf",
           "check_points", "fit_points", "fit_viewer", "no_fit", "structure_mask", "survey_dir", "write_index",
           "write_photo_ink"]
