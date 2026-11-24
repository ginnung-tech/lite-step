"""python -m tools.survey inspect --workdir DIR (--request survey/rounds/N/request.json | --n N --model output.ifc --model-version V)"""

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
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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


if __name__ == "__main__":
    sys.exit(main())
