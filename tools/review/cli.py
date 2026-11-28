"""lite-step review [output.py]: compile, validate, and look for defects by geometry; write review/.

Every finding names elements by their canonical name (find them in the source with `python output.py --q NAME`)
and carries its numbers. Findings of one kind between the same IFC classes are one GROUP: fix the cause, or
accept the group with a reason in review/accepted.json. Exit 0 when no open group is an error, 1 otherwise,
2 when the review itself could not run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.review.triage import Finding  # noqa: E402

MAX_EXAMPLES = 5
MAX_VALIDATION = 50


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lite-step review", description=__doc__.split("\n\n")[0])
    ap.add_argument("model", nargs="?", default="output.py", help="the model script (default output.py)")
    ap.add_argument("--ifc", help="the compiled IFC (default: output.ifc beside the script)")
    ap.add_argument("--out", help="where to write the review (default: review/ beside the script)")
    ap.add_argument("--no-compile", action="store_true", help="review the IFC as it is, without compiling first")
    args = ap.parse_args(argv)

    model = Path(args.model).resolve()
    ifc = Path(args.ifc).resolve() if args.ifc else model.with_name("output.ifc")
    out = Path(args.out).resolve() if args.out else model.parent / "review"
    out.mkdir(parents=True, exist_ok=True)
    findings: list[Finding] = []
    steps: dict[str, Any] = {}

    if not args.no_compile:
        ok, compile_findings, steps["compile"] = _compile(model)
        findings += compile_findings
        if not ok:
            return _write(out, model, ifc, findings, steps)
    if not ifc.is_file():
        print(f"error: no IFC to review at {ifc}", file=sys.stderr)
        return 2

    validation, steps["validate"] = _validate(ifc)
    findings += validation

    try:
        from tools.review import triage
        from tools.survey.scene import Scene
    except ImportError as exc:  # manifold3d and numpy come with Lite-STEP; ifcopenshell too
        steps["triage"] = {"skipped": f"cannot import the triage: {exc}"}
    else:
        scene = Scene.from_ifc(ifc, cache_dir=out / ".cache")
        result = triage.run(scene)
        findings += result.findings
        steps["triage"] = result.stats
    return _write(out, model, ifc, findings, steps)


# -- steps ---------------------------------------------------------------------------------------------------


def _compile(model: Path) -> tuple[bool, list[Finding], dict[str, Any]]:
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(model)], cwd=model.parent, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    found: list[Finding] = []
    if proc.returncode != 0:
        tail = "\n".join(text.strip().splitlines()[-25:])
        found.append(Finding("", "compile", "error", f"the model does not compile (exit {proc.returncode}):\n{tail}",
                             [], classes=[]))
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("warning:"):
            found.append(Finding("", "compile_warning", "warning", s[len("warning:"):].strip(), []))
    return proc.returncode == 0, found, {"exit": proc.returncode, "seconds": round(time.time() - t0, 1)}


def _validate(ifc: Path) -> tuple[list[Finding], dict[str, Any]]:
    """ifcopenshell.validate, errors only (the schema's own rules: a file that breaks one may not open elsewhere)."""
    t0 = time.time()
    try:
        import ifcopenshell
        import ifcopenshell.validate
    except ImportError as exc:
        return [], {"skipped": str(exc)}
    log = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(ifcopenshell.open(str(ifc)), log)
    errors = [s for s in log.statements if str(s.get("level", "")).lower() == "error"]
    found = []
    for s in errors[:MAX_VALIDATION]:
        inst = str(s.get("instance") or "")
        found.append(Finding("", "schema", "error", f"{s.get('message', '').strip()} {inst[:160]}".strip(), []))
    return found, {"errors": len(errors), "seconds": round(time.time() - t0, 1)}


# -- the report ----------------------------------------------------------------------------------------------


def _accepted(out: Path) -> dict[str, str]:
    """review/accepted.json: [{"group" or "id": ..., "reason": ...}]. A row without a reason accepts nothing."""
    p = out / "accepted.json"
    if not p.is_file():
        return {}
    rows = json.loads(p.read_text(encoding="utf-8"))
    return {str(r.get("group") or r.get("id")): str(r["reason"]) for r in rows
            if isinstance(r, dict) and str(r.get("reason") or "").strip() and (r.get("group") or r.get("id"))}


def _groups(findings: list[Finding], accepted: dict[str, str]) -> list[dict[str, Any]]:
    by: dict[str, list[Finding]] = {}
    for f in findings:
        by.setdefault(f.group, []).append(f)
    out = []
    for key, fs in by.items():
        sev = "error" if any(f.severity == "error" for f in fs) else "warning"
        nums: dict[str, dict[str, float]] = {}
        for name in sorted({k for f in fs for k in f.numbers}):
            vals = [f.numbers[name] for f in fs if name in f.numbers]
            nums[name] = {"min": min(vals), "median": round(statistics.median(vals), 1), "max": max(vals)}
        examples = sorted(fs, key=lambda f: f.severity != "error")[:MAX_EXAMPLES]
        reason = accepted.get(key) or (accepted.get(fs[0].id) if len(fs) == 1 else None)
        out.append({"group": key, "kind": fs[0].kind, "severity": sev, "count": len(fs), "numbers": nums,
                    "accepted": reason, "examples": [f.to_json() for f in examples]})
    order = {"compile": 0, "schema": 1, "floating": 2, "clash": 3, "near_miss": 4, "unchecked": 5}
    out.sort(key=lambda g: (g["accepted"] is not None, g["severity"] != "error", order.get(g["kind"], 9),
                            -g["count"]))
    return out


def _write(out: Path, model: Path, ifc: Path, findings: list[Finding], steps: dict[str, Any]) -> int:
    for n, f in enumerate(findings, 1):
        f.id = f"{f.kind}-{n:03d}"
    groups = _groups(findings, _accepted(out))
    open_errors = [g for g in groups if g["severity"] == "error" and not g["accepted"]]
    sha = hashlib.sha256(ifc.read_bytes()).hexdigest() if ifc.is_file() else None
    report = {"version": "1.0", "model": model.name, "ifcSha256": sha, "steps": steps,
              "openErrors": len(open_errors), "groups": groups, "findings": [f.to_json() for f in findings]}
    (out / "findings.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "findings.md").write_text(_markdown(report), encoding="utf-8")
    print(_summary(report))
    return 1 if open_errors else 0


def _short(name: str) -> str:
    """A canonical name without its containment tail (`...:storey:ground:site:x`), for reading; the JSON has all."""
    parts = name.split(":")
    keep = []
    for i in range(0, len(parts) - 1, 2):
        if parts[i] in ("spatialelement", "storey", "site", "building", "project", "facility"):
            break
        keep += parts[i:i + 2]
    return ":".join(keep) if keep else name


def _markdown(r: dict[str, Any]) -> str:
    lines = [f"# Review of {r['model']}", "",
             f"{r['openErrors']} open error group(s). Steps: "
             + "; ".join(f"{k} {json.dumps(v)}" for k, v in r["steps"].items()), ""]
    if not r["groups"]:
        lines.append("No findings.")
    for g in r["groups"]:
        state = f"accepted: {g['accepted']}" if g["accepted"] else g["severity"]
        nums = ", ".join(f"{k} {v['min']}-{v['max']} (median {v['median']})" for k, v in g["numbers"].items())
        lines.append(f"## {g['group']} ({state}, {g['count']})")
        if nums:
            lines.append(nums)
        for ex in g["examples"]:
            text = ex["summary"].splitlines()[0] if ex["kind"] != "compile" else ex["summary"]
            for n in sorted(ex["elements"], key=len, reverse=True):  # short names to read; findings.json has all
                text = text.replace(n, f"`{_short(n)}`")
            lines.append(f"- {text}")
        if g["count"] > len(g["examples"]):
            lines.append(f"- … {g['count'] - len(g['examples'])} more in findings.json")
        lines.append("")
    lines += ["To accept a group, add it to review/accepted.json with the reason:",
              '`[{"group": "clash:IfcCourse|IfcPavement", "reason": "the course is a layer of the pavement"}]`', ""]
    return "\n".join(lines)


def _summary(r: dict[str, Any]) -> str:
    head = f"review: {r['openErrors']} open error group(s), {len(r['groups'])} group(s), {len(r['findings'])} finding(s)"
    rows = [f"  {'ACCEPTED' if g['accepted'] else g['severity'].upper():8} {g['count']:5}  {g['group']}"
            for g in r["groups"]]
    return "\n".join([head, *rows, "  details: review/findings.md"])


if __name__ == "__main__":
    raise SystemExit(main())
