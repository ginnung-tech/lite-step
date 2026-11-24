"""python -m tools.survey inspect --workdir DIR (--request survey/rounds/N/request.json | --n N --model output.ifc --model-version V)
python -m tools.survey add --workdir DIR --picture FILE --kind KIND --title T [--bcfv VIEWPOINT | --points PTS.json] [--model output.ifc]
python -m tools.survey index --workdir DIR"""

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
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.cmd == "add":
        return _add(args)
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
    """One picture in; prints the intake's answer for it (groa-cli SurveyIntakeResult, one source)."""
    from .intake import IntakeError, add_source, ask_align, camera_from_bcfv, fit_points, fit_viewer, no_fit

    wd = Path(args.workdir)
    now = _now()
    try:
        cam = None
        if args.bcfv:
            import cv2

            img = cv2.imread(args.picture)
            if img is None:
                raise IntakeError(f"{args.picture} is not a picture OpenCV can read")
            cam = camera_from_bcfv(Path(args.bcfv).read_text(encoding="utf-8"), img.shape[1], img.shape[0])
        src = add_source(wd, args.picture, kind=args.kind, title=args.title, now=now, convention=args.convention,
                         origin=args.origin, added_by=args.by, source_id=args.source_id, viewer_camera=cam,
                         illustrative=args.illustrative)
        if src["kind"] == "text" or src.get("illustrative"):
            result = no_fit(wd, src["id"])
        elif src["kind"] == "viewer_screenshot" or args.points:
            from .scene import Scene

            model = wd / args.model
            scene = Scene.from_ifc(model) if model.exists() else None
            if src["kind"] == "viewer_screenshot":
                if scene is None:
                    raise IntakeError(f"no model at {args.model} to fit the screenshot against")
                result = fit_viewer(wd, src["id"], scene, now=now)
            else:
                pts = json.loads(Path(args.points).read_text(encoding="utf-8"))
                result = fit_points(wd, src["id"], pts, now=now, scene=scene)
        else:
            ask_align(wd, src["id"], now=now)
            result = {"id": src["id"], "sha256": src["sha256"], "fit": "failed"}
    except (IntakeError, ValueError, OSError) as e:
        print(json.dumps({"error": str(e)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
