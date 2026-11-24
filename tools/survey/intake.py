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

PICTURE_KINDS = ("elevation", "plan", "section", "detail_plane", "perspective", "photo", "viewer_screenshot",
                 "screenshot", "sketch")
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class IntakeError(ValueError):
    """A picture that cannot be taken in as asked (unknown source, wrong kind, unreadable file)."""


def survey_dir(workdir: str | Path) -> Path:
    return Path(workdir) / "survey"


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
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
    table = _read(s / "sources.json", "sources")
    digest = sha256_of(pic)
    for e in table["sources"]:
        if e.get("sha256") == digest:
            return e
    taken = {e["id"] for e in table["sources"]}
    base = source_id or _slug(pic.name)
    if not _ID.match(base):
        raise IntakeError(f"source id {base!r} must match {_ID.pattern}")
    sid, n = base, 2
    while sid in taken:
        sid, n = f"{base[:60]}-{n}", n + 1
    dest = s / "sources" / f"{sid}{pic.suffix.lower() or '.bin'}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pic, dest)
    entry: dict[str, Any] = {
        "id": sid, "file": dest.relative_to(s).as_posix(), "sha256": digest, "kind": kind,
        "convention": "none" if kind == "text" else (convention or _default_convention(kind)),
        "title": (title or pic.name)[:300], "addedBy": added_by, "addedAt": now,
    }
    if origin:
        entry["origin"] = origin[:500]
    if viewer_camera is not None:
        entry["viewerCamera"] = viewer_camera
    if illustrative:
        entry["illustrative"] = True
    table["version"] = VERSION
    table["sources"].append(entry)
    _write(s / "sources.json", table)
    return entry


def _default_convention(kind: str) -> str:
    return {"section": "section", "plan": "section", "detail_plane": "section", "photo": "shaded",
            "viewer_screenshot": "shaded", "screenshot": "shaded", "perspective": "shaded"}.get(kind, "front_slab")


def _source(workdir: Path, source_id: str) -> dict:
    for e in _read(survey_dir(workdir) / "sources.json", "sources")["sources"]:
        if e["id"] == source_id:
            return e
    raise IntakeError(f"no source {source_id!r} in sources.json")


# --- fits ----------------------------------------------------------------------------------------------


def _model_outline(hit: np.ndarray) -> np.ndarray:
    import cv2

    m = hit.astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)


def _save_overlay(workdir: Path, source_id: str, version: int, img: np.ndarray, hit: np.ndarray | None,
                  points: list[tuple[np.ndarray, np.ndarray]] | None = None) -> str:
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
    rel = f"sources/{source_id}.fit-v{version}.png"
    path = survey_dir(workdir) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), out):
        raise IntakeError(f"could not write {rel}")
    return rel


def _next_version(cams: dict, source_id: str) -> int:
    return 1 + max((c["version"] for c in cams["cameras"] if c["source"] == source_id), default=0)


def _record(workdir: Path, cam_entry: dict) -> None:
    s = survey_dir(workdir)
    cams = _read(s / "cameras.json", "cameras")
    cams["version"] = VERSION
    cams["cameras"].append(cam_entry)
    _write(s / "cameras.json", cams)


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


def fit_viewer(workdir: str | Path, source_id: str, scene, *, now: str, gate: float = VIEWER_GATE) -> dict:
    """Fit a viewer screenshot by its own camera. The camera is never changed here: if the model does not lie
    on the picture, the gate fails and an `align` question is asked."""
    from .drone import Drone
    from .overlay import agreement, background_mask

    wd = Path(workdir)
    src = _source(wd, source_id)
    if src["kind"] != "viewer_screenshot":
        raise IntakeError(f"{source_id} is a {src['kind']}, not a viewer screenshot")
    img = _read_image(survey_dir(wd) / src["file"])
    h, w = img.shape[:2]
    camd = dict(src["viewerCamera"], imageW=w, imageH=h)  # the picture's pixels are the truth about its size
    cam = PerspectiveCamera.from_contract(camd)
    hit = Drone(scene).render(cam).hit
    a = agreement(background_mask(img), hit, tol_px=TOL_PX)
    cams = _read(survey_dir(wd) / "cameras.json", "cameras")
    version = _next_version(cams, source_id)
    passed = bool(hit.any()) and a.model_within_px >= gate
    entry = {"source": source_id, "version": version, "camera": camd, "method": "viewer", "residualMm": 0.0,
             "gate": "passed" if passed else "failed", "frozenAt": now,
             "reason": (f"the viewer's own camera; {a.model_within_px:.0%} of model pixels within {TOL_PX:g} px"
                        if hit.any() else "the viewer's own camera sees none of the model"),
             "overlay": _save_overlay(wd, source_id, version, img, hit),
             "agreement": {"tolPx": TOL_PX, "modelWithinPx": round(a.model_within_px, 4),
                           "sourceWithinPx": round(a.source_within_px, 4)}}
    _record(wd, entry)
    if not passed:
        ask_align(wd, source_id, now=now)
    return _result(src, entry)


def fit_points(workdir: str | Path, source_id: str, points: list[dict], *, now: str, scene=None,
               gate_px: float = POINTS_GATE_PX, reason: str | None = None) -> dict:
    """Fit a perspective camera to control points: `[{"px": [x, y], "world": [x, y, z], "by": ...}, ...]`,
    four at least, spread in depth where possible. With a `scene`, the overlay also draws the model."""
    wd = Path(workdir)
    src = _source(wd, source_id)
    if src["kind"] == "text" or src.get("illustrative"):
        raise IntakeError(f"{source_id} is not a picture to fit")
    img = _read_image(survey_dir(wd) / src["file"])
    h, w = img.shape[:2]
    world = np.array([p["world"] for p in points], dtype=np.float64)
    px = np.array([p["px"] for p in points], dtype=np.float64)
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
        from .overlay import agreement, background_mask

        hit = Drone(scene).render(cam).hit
        a = agreement(background_mask(img), hit, tol_px=TOL_PX)
        agree = {"tolPx": TOL_PX, "modelWithinPx": round(a.model_within_px, 4),
                 "sourceWithinPx": round(a.source_within_px, 4)}
    cams = _read(survey_dir(wd) / "cameras.json", "cameras")
    version = _next_version(cams, source_id)
    passed = f.residual_px <= gate_px
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
        ask_align(wd, source_id, now=now)
    return _result(src, entry)


def ask_align(workdir: str | Path, source_id: str, *, now: str, question: str | None = None,
              points: int = 4) -> dict:
    """An `align` question for a picture that could not be fitted: the person clicks `points` points on the
    picture and the same points on the model (contract survey-question)."""
    q = {"version": "1.0", "id": f"q-align-{source_id}"[:64], "kind": "align", "blocking": False, "askedAt": now,
         "question": question or (f"I could not line the model up with this picture. Please click {points} points "
                                  "on it, then the same points on the model."),
         "source": source_id, "points": points}
    _write(survey_dir(workdir) / "questions" / f"{q['id']}.json", q)
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


__all__ = ["IntakeError", "add_source", "ask_align", "camera_from_bcfv", "camera_to_bcf", "fit_points", "fit_viewer",
           "no_fit", "survey_dir", "write_index"]
