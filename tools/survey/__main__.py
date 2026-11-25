"""python -m tools.survey inspect --workdir DIR (--request survey/rounds/N/request.json | --n N --model output.ifc --model-version V)
python -m tools.survey add --workdir DIR --picture FILE --kind KIND --title T [--bcfv VIEWPOINT | --points PTS.json] [--model output.ifc]
python -m tools.survey index --workdir DIR
python -m tools.survey partials --workdir DIR --source ID --parts PARTS.json
python -m tools.survey ink --workdir DIR --source ID"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.survey")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ins = sub.add_parser("inspect", help="run a round of minions on the working directory")
    ins.add_argument("--workdir", default=".")
    ins.add_argument("--request", help="a roundRequest file (relative to the workdir)")
    ins.add_argument("--n", type=int)
    ins.add_argument("--model", default="output.ifc")
    ins.add_argument("--model-version", type=int)
    ins.add_argument("--shots", nargs="*")
    ins.add_argument("--workers", type=int)
    add = sub.add_parser("add", help="take a picture into survey/: store it, fit a camera, show it back or ask")
    add.add_argument("--workdir", default=".")
    add.add_argument("--picture", required=True)
    add.add_argument("--kind", required=True)
    add.add_argument("--title", required=True)
    add.add_argument("--convention")
    add.add_argument("--origin")
    add.add_argument("--id", dest="source_id")
    add.add_argument("--by", default="surveyor", choices=["person", "surveyor", "groa", "minion"])
    add.add_argument("--bcfv", help="the viewer's viewpoint (.bcfv) a viewer_screenshot was taken with")
    add.add_argument("--points", help="a JSON list of control points [{px, world, by}] (four or more)")
    add.add_argument("--model", default="output.ifc", help="the model to fit against (relative to the workdir)")
    add.add_argument("--illustrative", action="store_true")
    idx = sub.add_parser("index", help="write survey/index.json (last, after the files are kept)")
    idx.add_argument("--workdir", default=".")
    par = sub.add_parser("partials", help="one shot per detail of a picture: crops, a perspective camera follows them")
    par.add_argument("--workdir", default=".")
    par.add_argument("--source", required=True)
    par.add_argument("--parts", required=True, help='a JSON list [{"name", "box": [x0, y0, x1, y1], "purpose"}]')
    par.add_argument("--camera-version", type=int, default=1)
    ink = sub.add_parser("ink", help="ink for a photo: structure (1) and openings (5, sky seen through it)")
    ink.add_argument("--workdir", default=".")
    ink.add_argument("--source", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.cmd == "add":
        return _add(args)
    if args.cmd in ("partials", "ink"):
        from .intake import add_partials, confined, write_photo_ink

        try:
            if args.cmd == "partials":
                parts = json.loads(confined(args.workdir, args.parts).read_text(encoding="utf-8"))
                out = add_partials(args.workdir, args.source, parts, camera_version=args.camera_version)
                print(json.dumps({"shots": [s["id"] for s in out]}))
            else:
                print(json.dumps({"ink": write_photo_ink(args.workdir, args.source)}))
        except (ValueError, OSError) as e:  # IntakeError is a ValueError
            print(json.dumps({"error": str(e)}))
            return 2
        return 0
    if args.cmd == "index":
        from .intake import write_index

        print(json.dumps({"files": len(write_index(args.workdir, now=_now())["files"])}))
        return 0

    from .inspect import run_round

    wd = Path(args.workdir)
    if args.request:
        req = json.loads((wd / args.request).read_text(encoding="utf-8"))
    else:
        if args.n is None or args.model_version is None:
            ap.error("give --request, or --n and --model-version")
        req = {"version": "1.1", "n": args.n, "modelFile": args.model, "modelVersion": args.model_version,
               "requestedBy": "person", "requestedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if args.shots:
            req["shots"] = args.shots
    result = run_round(wd, req, workers=args.workers)
    print(json.dumps({k: result.get(k) for k in ("n", "status", "error", "shotCount", "durationS", "inkWeightedF1")}))
    return 0 if result["status"] == "done" else 1


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _add(args) -> int:
    """One picture in; prints the intake's answer for it (groa-cli SurveyIntakeResult, one source).

    Every path is inside the working directory: the picture, the viewpoint, the points and the model."""
    from .intake import add_source, confined

    wd = Path(args.workdir).resolve()
    now = _now()
    try:
        picture = confined(wd, args.picture)
        cam = _viewer_camera(confined(wd, args.bcfv), picture) if args.bcfv else None
        src = add_source(wd, picture, kind=args.kind, title=args.title, now=now, convention=args.convention,
                         origin=args.origin, added_by=args.by, source_id=args.source_id, viewer_camera=cam,
                         illustrative=args.illustrative)
        result = _fit(wd, src, args, now)
    except (ValueError, OSError) as e:  # IntakeError is a ValueError
        print(json.dumps({"error": str(e)}))
        return 2
    print(json.dumps(result))
    return 0


def _viewer_camera(bcfv: Path, picture: Path) -> dict:
    import cv2

    from .intake import IntakeError, camera_from_bcfv

    img = cv2.imread(str(picture))
    if img is None:
        raise IntakeError(f"{picture.name} is not a picture OpenCV can read")
    return camera_from_bcfv(bcfv.read_text(encoding="utf-8"), img.shape[1], img.shape[0])


def _fit(wd: Path, src: dict, args, now: str) -> dict:
    from .intake import IntakeError, ask_align, confined, fit_points, fit_viewer, no_fit

    if src["kind"] == "text" or src.get("illustrative"):
        return no_fit(wd, src["id"])
    if src["kind"] != "viewer_screenshot" and not args.points:
        ask_align(wd, src["id"], now=now)
        return {"id": src["id"], "sha256": src["sha256"], "fit": "failed"}
    from .scene import Scene

    model = confined(wd, args.model)
    scene = Scene.from_ifc(model) if model.exists() else None
    if src["kind"] == "viewer_screenshot":
        if scene is None:
            raise IntakeError(f"no model at {args.model} to fit the screenshot against")
        return fit_viewer(wd, src["id"], scene, now=now)
    pts = json.loads(confined(wd, args.points).read_text(encoding="utf-8"))
    return fit_points(wd, src["id"], pts, now=now, scene=scene)


if __name__ == "__main__":
    sys.exit(main())
