"""Heimdal's plan across rounds (contract survey 1.5: ``survey/plan.json``, ``survey/iterations.jsonl``).

A round is one run of Heimdal on a request. Heimdal writes ``plan.json``; this module checks it against the plan
it started from, counts the rounds and appends the round to ``iterations.jsonl``. The same checks run when a host
runs Heimdal for you, and by hand through ``python -m tools.survey plan begin|check|record`` (the bottom of this file).

The rules that span two versions of the plan are checked here, in code, not asked of the model:

- ids are stable: every id of the previous ``items`` and ``open`` is still there or has a ``closed`` entry with
  its reason;
- a value that moves by more than ``sqrt(a^2 + b^2)`` of the two readings' uncertainties, or text that changes,
  is a ``changes`` entry for this round with ``from``, ``to`` and a reason;
- the round is counted per request; at ``MAX_ROUNDS`` the decision is ``converged`` or ``ask``;
- a check is a number measured on the person's own pictures, with its file written this round (survey 1.5).
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_ROUNDS = 3
PLAN_VERSION = "1.5"
MAX_PLAN_BYTES = 512 * 1024

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_DECISIONS = ("converged", "instruct", "ask")
_PREVIOUS = ("passed", "failed", "not_checked")
_PURPOSES = ("capture", "design", "mix")
_SURE = ("sure", "measured", "guess", "unreliable")
_MODEL = ("met", "not_met", "unknown")
_UNITS = ("mm", "deg", "count", "ratio")
_OPEN_KINDS = ("contradiction", "question", "missing")

Plan = dict[str, Any]


class PlanError(Exception):
    """plan.json breaks a rule. ``problems`` are sentences Heimdal can act on."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


def plan_path(workdir: Path) -> Path:
    return workdir / "survey" / "plan.json"


def read_plan(workdir: Path) -> Plan | None:
    """The plan on disk; None when there is none. A plan that is not JSON is an error, never "no plan"."""
    p = plan_path(workdir)
    if not p.is_file():
        return None
    data = p.read_bytes()
    if len(data) > MAX_PLAN_BYTES:
        raise PlanError([f"survey/plan.json is {len(data)} bytes, over {MAX_PLAN_BYTES}"])
    try:
        plan = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise PlanError([f"survey/plan.json is not valid JSON: {exc}"]) from exc
    if not isinstance(plan, dict):
        raise PlanError(["survey/plan.json is not a JSON object"])
    return plan


# -- rounds ------------------------------------------------------------------------------------------------


def iterations(workdir: Path) -> list[dict[str, Any]]:
    """``iterations.jsonl`` as rows. A line that is not an object is skipped, logged (the file is append-only and
    written only through append_iteration, so a bad line is a bug to see, not a reason to stop the round)."""
    p = workdir / "survey" / "iterations.jsonl"
    if not p.is_file():
        return []
    rows = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            logger.warning("survey plan: iterations.jsonl line %d is not JSON; skipped", n)
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def last_model_sha(workdir: Path, request: str) -> str | None:
    """The model the request's last round measured (its ``modelSha256``); None when no round recorded one."""
    rows = [r for r in iterations(workdir) if r.get("request") == request]
    return rows[-1].get("modelSha256") if rows else None


def next_round(workdir: Path, request: str) -> int:
    """This request's next round: one more than the rounds already recorded for it."""
    return 1 + sum(1 for r in iterations(workdir) if r.get("request") == request)


def append_iteration(workdir: Path, plan: Plan, at: str, model_sha256: str | None = None) -> dict[str, Any]:
    d = plan["decision"]
    row: dict[str, Any] = {"round": d["round"], "request": d["request"], "at": at}
    if model_sha256:
        row["modelSha256"] = model_sha256
    row["decision"] = d["kind"]
    for k in ("instruction", "check"):
        if d.get(k):
            row[k] = d[k]
    row["previousCheck"] = d["previousCheck"]
    row["met"] = sum(1 for i in plan["items"] if i.get("model") == "met")
    row["total"] = len(plan["items"])
    row["open"] = len(plan["open"])
    row["contradictions"] = sum(1 for o in plan["open"] if o.get("kind") == "contradiction")
    p = workdir / "survey" / "iterations.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


# -- one plan on its own (the schema's rules) ---------------------------------------------------------------


def _text(v: Any, lo: int, hi: int) -> bool:
    return isinstance(v, str) and lo <= len(v.strip()) and len(v) <= hi


def _value_ok(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)) or _text(v, 1, 200)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_shape(plan: Plan) -> list[str]:
    """The rules one plan.json must keep on its own (contract survey 1.4, definition ``plan``)."""
    bad: list[str] = []
    if plan.get("version") != PLAN_VERSION:
        bad.append(f'version must be "{PLAN_VERSION}"')
    if plan.get("purpose") not in _PURPOSES:
        bad.append(f"purpose must be one of {', '.join(_PURPOSES)}")
    for key in ("items", "open", "closed", "changes"):
        if not isinstance(plan.get(key), list):
            bad.append(f"{key} must be a list")
    if bad:
        return bad
    seen: set[str] = set()
    for i in plan["items"]:
        iid = i.get("id") if isinstance(i, dict) else None
        if not isinstance(iid, str) or not _ID.match(iid):
            bad.append(f"item {iid!r}: id must be lowercase letters, digits, - or _")
            continue
        if iid in seen:
            bad.append(f"item {iid}: the id is used twice")
        seen.add(iid)
        if not _text(i.get("name"), 3, 120) or not _value_ok(i.get("value")) or not _text(i.get("source"), 3, 300):
            bad.append(f"item {iid}: needs name, value (a number or short text) and source")
        if i.get("sure") not in _SURE or i.get("model") not in _MODEL:
            bad.append(f"item {iid}: sure is one of {', '.join(_SURE)}; model is one of {', '.join(_MODEL)}")
        if "unit" in i and i["unit"] not in _UNITS:
            # 2026-11-27: Heimdal wrote "unit": "layout" on text items; the contract refuses it
            bad.append(f"item {iid}: unit is one of {', '.join(_UNITS)}, or no unit for text")
        elif _is_number(i.get("value")):
            if i.get("unit") not in _UNITS:
                bad.append(f"item {iid}: a number needs a unit ({', '.join(_UNITS)})")
            elif i["unit"] == "mm" and not float(i["value"]).is_integer():
                bad.append(f"item {iid}: millimetres are whole numbers")
        u = i.get("uncertainty")
        if u is not None and (not _is_number(u) or u < 0):
            bad.append(f"item {iid}: uncertainty is a number of at least 0")
    for o in plan["open"]:
        oid = o.get("id") if isinstance(o, dict) else None
        if not isinstance(oid, str) or not _ID.match(oid):
            bad.append(f"open entry {oid!r}: id must be lowercase letters, digits, - or _")
            continue
        if oid in seen:
            bad.append(f"open entry {oid}: the id is used twice")
        seen.add(oid)
        if o.get("kind") not in _OPEN_KINDS or not _text(o.get("text"), 3, 600):
            bad.append(f"open entry {oid}: needs kind ({', '.join(_OPEN_KINDS)}) and text")
        if not isinstance(o.get("since"), int) or o["since"] < 1:
            bad.append(f"open entry {oid}: since is the round it was opened")
        if o.get("kind") == "contradiction":
            r = o.get("readings")
            if not isinstance(r, list) or len(r) < 2 or not all(
                    isinstance(x, dict) and _value_ok(x.get("value")) and _text(x.get("source"), 3, 300) for x in r):
                bad.append(f"open entry {oid}: a contradiction names at least two readings, each with value and source")
    for c in plan["closed"]:
        if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not _text(c.get("reason"), 3, 600) \
                or not isinstance(c.get("round"), int):
            bad.append(f"closed entry {c.get('id') if isinstance(c, dict) else c!r}: needs id, reason and round")
    for c in plan["changes"]:
        if not isinstance(c, dict) or not isinstance(c.get("item"), str) or not _value_ok(c.get("from")) \
                or not _value_ok(c.get("to")) or not _text(c.get("reason"), 3, 600) or not isinstance(c.get("round"), int):
            bad.append(f"change {c.get('item') if isinstance(c, dict) else c!r}: needs item, from, to, reason and round")
    d = plan.get("decision")
    if not isinstance(d, dict):
        return bad + ["decision is missing"]
    if d.get("kind") not in _DECISIONS:
        bad.append(f"decision.kind must be one of {', '.join(_DECISIONS)}")
    if d.get("previousCheck") not in _PREVIOUS:
        bad.append(f"decision.previousCheck must be one of {', '.join(_PREVIOUS)}")
    if d.get("kind") == "instruct" and not (_text(d.get("instruction"), 3, 2000) and _text(d.get("check"), 3, 600)):
        bad.append("decision instruct needs instruction and check")
    if d.get("kind") == "ask" and not _text(d.get("question"), 3, 600):
        bad.append("decision ask needs question")
    bad += _check_shapes(d)
    return bad


_CHECK_UNITS = ("mm", "deg", "count", "ratio", "px")


def _num(v: Any) -> bool:
    return _is_number(v) and math.isfinite(float(v))


def _check_shapes(d: dict[str, Any]) -> list[str]:
    """Survey 1.5: a check is a MEASUREMENT on the person's pictures, with its evidence (the shapes; the files are
    checked by ``check_evidence``). 2026-11-27: a round converged on "the beams are present in the model"."""
    bad: list[str] = []
    spec, res = d.get("checkSpec"), d.get("checkResult")
    kind, prev = d.get("kind"), d.get("previousCheck")
    for name, c in (("checkSpec", spec), ("checkResult", res)):
        if c is None:
            continue
        if not isinstance(c, dict) or not isinstance(c.get("sources"), list) or not c["sources"] \
                or not all(isinstance(s, str) and _ID.match(s) for s in c["sources"]):
            bad.append(f"decision.{name}.sources names the person's pictures it is measured on (at least one source id)")
            continue
        if not _num(c.get("expected")) or not _num(c.get("tolerance")) or float(c["tolerance"]) <= 0 \
                or c.get("unit") not in _CHECK_UNITS:
            bad.append(f"decision.{name} needs a numeric expected, a tolerance above 0 and a unit ({', '.join(_CHECK_UNITS)})")
        if name == "checkSpec" and not _text(c.get("quantity"), 3, 300):
            bad.append("decision.checkSpec.quantity says what is measured on the pictures")
        if name == "checkResult" and (not _num(c.get("measured")) or not isinstance(c.get("output"), str) or not c["output"]):
            bad.append("decision.checkResult needs the measured number and the measurement's output file")
    if kind == "instruct" and spec is None:
        bad.append("decision instruct needs checkSpec: what you will MEASURE on the person's pictures, with its tolerance "
                   "(that a part is present in the model is not a check)")
    if prev in ("passed", "failed") and res is None:
        bad.append(f"previousCheck {prev} needs checkResult: the number measured on the person's pictures and its file")
    if kind == "converged" and (res is None or prev != "passed"):
        bad.append("converged needs a passed check measured this round (checkResult): the model meets the plan as the "
                   "person's pictures show it, or it has not converged")
    if isinstance(res, dict) and _num(res.get("measured")) and _num(res.get("expected")) and _num(res.get("tolerance")) \
            and prev in ("passed", "failed"):
        within = abs(float(res["measured"]) - float(res["expected"])) <= float(res["tolerance"])
        if within != (prev == "passed"):
            bad.append(f"previousCheck {prev} disagrees with checkResult: measured {res['measured']} against "
                       f"{res['expected']} +/- {res['tolerance']}")
    return bad


def _rows(workdir: Path, name: str, key: str) -> list[dict[str, Any]]:
    p = workdir / "survey" / name
    if not p.is_file():
        return []
    try:
        rows = json.loads(p.read_text(encoding="utf-8")).get(key) or []
    except (ValueError, AttributeError):
        return []
    return [r for r in rows if isinstance(r, dict)]


def check_evidence(workdir: Path, plan: Plan, written_before: dict[str, str]) -> list[str]:
    """The files behind a check (survey 1.5): the sources are the person's pictures, a reported result's pictures
    have a passed camera, and its output file was written this round. ``written_before`` is every survey file's
    sha256 just before Heimdal ran."""
    import hashlib

    d = plan.get("decision") if isinstance(plan.get("decision"), dict) else {}
    srcs = {s["id"]: s for s in _rows(workdir, "sources.json", "sources") if isinstance(s.get("id"), str)}
    passed = {c.get("source") for c in _rows(workdir, "cameras.json", "cameras") if c.get("gate") == "passed"}
    bad: list[str] = []
    for name in ("checkSpec", "checkResult"):
        c = d.get(name)
        if not isinstance(c, dict):
            continue
        for sid in c.get("sources") or []:
            s = srcs.get(sid)
            if s is None:
                bad.append(f"decision.{name}: source {sid} is not in survey/sources.json")
            elif s.get("illustrative") or s.get("kind") in ("viewer_screenshot", "text"):
                bad.append(f"decision.{name}: source {sid} is not the person's picture; a check is measured on what "
                           "the person gave")
            elif name == "checkResult" and sid not in passed:
                bad.append(f"decision.checkResult: {sid} has no lined-up camera, so nothing was measured on it; line it "
                           "up or ask the person to, and do not report or converge on it")
    res = d.get("checkResult")
    if isinstance(res, dict) and isinstance(res.get("output"), str) and res["output"]:
        rel = res["output"].lstrip("/")
        base = (workdir / "survey").resolve()
        f = (base / rel).resolve()
        if ".." in rel.split("/") or base not in f.parents or not f.is_file():
            bad.append(f"decision.checkResult.output {res['output']} is not a file in survey/")
        elif written_before.get(rel) == hashlib.sha256(f.read_bytes()).hexdigest():
            bad.append(f"decision.checkResult.output {res['output']} was not written this round: measure this model")
    return bad


# -- two plans (the rules across rounds) --------------------------------------------------------------------


def _moved(old: dict[str, Any], new: dict[str, Any]) -> bool:
    a, b = old.get("value"), new.get("value")
    if _is_number(a) and _is_number(b) and old.get("unit") == new.get("unit"):
        tol = math.hypot(float(old.get("uncertainty") or 0), float(new.get("uncertainty") or 0))
        return abs(float(b) - float(a)) > tol
    return a != b


def check_rounds(prev: Plan | None, plan: Plan, *, round_n: int, request: str, max_rounds: int = MAX_ROUNDS,
                 model_unchanged: bool = False) -> list[str]:
    """``check_shape`` and the rules across two plans. ``prev`` is the plan this round started from."""
    bad = check_shape(plan)
    if bad:
        return bad
    d = plan["decision"]
    if d.get("round") != round_n:
        bad.append(f"decision.round must be {round_n} (this request's round)")
    if d.get("maxRounds") != max_rounds:
        bad.append(f"decision.maxRounds must be {max_rounds}")
    if d.get("request") != request:
        bad.append(f"decision.request must be {request}")
    if round_n == 1 and d.get("previousCheck") != "not_checked" and d.get("kind") != "converged":
        # 2026-11-27: a new request's round 1 said "the last step's check failed"; the step it meant was
        # another request's, and the person on this one never asked for it
        bad.append("this is the request's first round: no step of it has been built, so decision.previousCheck is "
                   "not_checked")
    if model_unchanged and d.get("previousCheck") != "not_checked":
        # 2026-11-27: round 2 said the last check "failed" when Gróa had built nothing since round 1
        bad.append("the model is the one the last round measured (no new version since), so decision.previousCheck "
                   "is not_checked")
    if round_n >= max_rounds and d.get("kind") == "instruct":
        bad.append(f"round {round_n} is the last of {max_rounds}: decide converged or ask (you may still write the "
                   "instruction you would give, for the person to decide)")
    if prev is None:
        return bad
    closed = {c["id"]: c for c in plan["closed"]}
    items = {i["id"]: i for i in plan["items"]}
    open_now = {o["id"] for o in plan["open"]}
    for o in prev.get("open") or []:
        oid = o.get("id")
        if isinstance(oid, str) and oid not in open_now and oid not in closed:
            bad.append(f"open entry {oid} ({str(o.get('text', ''))[:80]}) is gone: keep it open or close it "
                       "with a reason")
    changed_now = {c["item"] for c in plan["changes"] if c.get("round") == round_n}
    for old in prev.get("items") or []:
        iid = old.get("id")
        if not isinstance(iid, str):
            continue
        new = items.get(iid)
        if new is None:
            if iid not in closed:
                bad.append(f"item {iid} ({old.get('name')}) is gone: keep it or close it with a reason")
            continue
        if _moved(old, new) and iid not in changed_now:
            bad.append(f"item {iid} ({old.get('name')}) moved from {old.get('value')} to {new.get('value')}"
                       f"{' ' + str(new.get('unit')) if new.get('unit') else ''}, beyond the two readings' "
                       f"uncertainty: add a change for round {round_n} with its reason, or open a contradiction")
    return bad


# -- what the request shows ---------------------------------------------------------------------------------


def _shown(v: Any, unit: Any) -> str:
    if _is_number(v):
        n = int(v) if float(v).is_integer() else v
        return f"{n} {unit}" if unit and unit != "count" else str(n)
    return str(v)


def round_summary(prev: Plan | None, plan: Plan) -> dict[str, Any]:
    """This round as the request shows it: the decision, what changed, what is settled, what is open."""
    d = plan["decision"]
    r = d["round"]
    items = {i["id"]: i for i in plan["items"]}
    old_items = {i.get("id"): i for i in (prev or {}).get("items") or [] if isinstance(i, dict)}
    old_open = {o.get("id"): o for o in (prev or {}).get("open") or [] if isinstance(o, dict)}
    out: dict[str, Any] = {"round": r, "maxRounds": d["maxRounds"], "decision": d["kind"],
                           "previousCheck": d["previousCheck"],
                           "met": sum(1 for i in plan["items"] if i.get("model") == "met"), "total": len(plan["items"])}
    for k in ("instruction", "check", "question"):
        if d.get(k):
            out[k] = d[k]
    res = d.get("checkResult")
    if isinstance(res, dict) and _num(res.get("measured")):
        # survey 1.5: what the check measured, so the request shows the number and the pictures, not a verdict
        out["checked"] = {k: res[k] for k in ("sources", "measured", "expected", "tolerance", "unit") if k in res}
    changed = []
    for c in plan["changes"]:
        if c.get("round") != r:
            continue
        unit = (items.get(c["item"]) or old_items.get(c["item"]) or {}).get("unit")
        name = (items.get(c["item"]) or old_items.get(c["item"]) or {}).get("name") or c["item"]
        changed.append({"item": name, "from": _shown(c["from"], unit), "to": _shown(c["to"], unit), "reason": c["reason"]})
    out["changed"] = changed
    out["open"] = [{"kind": o["kind"], "text": o["text"]} for o in plan["open"]]
    closed = []
    for c in plan["closed"]:
        if c.get("round") != r:
            continue
        was = old_open.get(c["id"]) or old_items.get(c["id"]) or {}
        closed.append({"text": str(was.get("text") or was.get("name") or c["id"]), "reason": c["reason"]})
    out["closed"] = closed
    return out


# -- by hand: begin, check, record --------------------------------------------------------------------------
#
# A host that runs Heimdal snapshots the survey before the run and checks the plan after it. Run by hand (any
# coding agent with agents/heimdal/SKILL.md), the same three steps are commands:
#
#   python -m tools.survey plan begin  --workdir . --request R [--model output.ifc]   # before you read anything
#   python -m tools.survey plan check  --workdir .                                     # after you wrote plan.json
#   python -m tools.survey plan record --workdir .                                     # when check is clean
#
# ``begin`` keeps what the round started from in ``survey/.plan-round.json`` (a dot-file: not part of the survey).

ROUND_FILE = ".plan-round.json"


def _sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def survey_shas(workdir: Path) -> dict[str, str]:
    """Every survey/ file's sha256 now, by path inside survey/ (what "written this round" is measured against)."""
    base = workdir / "survey"
    if not base.is_dir():
        return {}
    return {p.relative_to(base).as_posix(): _sha256(p.read_bytes()) for p in base.rglob("*")
            if p.is_file() and p.name != ROUND_FILE}


def round_rules(round_n: int, *, request: str, has_plan: bool, model_unchanged: bool,
                max_rounds: int = MAX_ROUNDS) -> list[str]:
    """The round's facts, in the words a host gives Heimdal in its prompt."""
    last = round_n >= max_rounds
    return [
        f"This is round {round_n} of {max_rounds} on request {request}. In plan.json set decision.round {round_n}, "
        f"decision.maxRounds {max_rounds} and decision.request {request}.",
        ("survey/plan.json holds the plan as the last round left it. Keep every item's and open entry's id; nothing "
         "leaves without a closed entry and its reason; a value that moves beyond the two readings' uncertainty is a "
         "change for this round, or an open contradiction. Report the last instruction's check as "
         "decision.previousCheck." if has_plan else
         "There is no plan yet: write the first one. decision.previousCheck is not_checked."),
        ("This is the last round: decide converged or ask." if last else
         "Decide converged, instruct (one instruction and its check) or ask (one question)."),
        *(["The model is the one the last round measured: nothing was built since, so decision.previousCheck is "
           "not_checked."] if model_unchanged and round_n > 1 else []),
        *(["This is the request's first round: decision.previousCheck is not_checked."] if round_n == 1 else []),
        "A check is a number measured on the person's own pictures, with checkSpec on an instruction and checkResult "
        "(the number, its lined-up sources, its file under survey/measure/ written this round) on a reported check "
        "and on converged.",
    ]


def begin_round(workdir: Path, request: str, model: str = "output.ifc") -> dict[str, Any]:
    """Snapshot what this round starts from. Returns the round's facts and rules."""
    if not _text(request, 1, 200):
        raise PlanError(["--request names the request this round is on (any stable id)"])
    workdir = Path(workdir)
    prev = read_plan(workdir)
    m = workdir / model
    sha = _sha256(m.read_bytes()) if m.is_file() else None
    round_n = next_round(workdir, request)
    unchanged = sha is not None and sha == last_model_sha(workdir, request)
    state = {"request": request, "round": round_n, "maxRounds": MAX_ROUNDS, "model": model, "modelSha256": sha,
             "modelUnchanged": unchanged, "plan": prev, "shas": survey_shas(workdir)}
    p = workdir / "survey" / ROUND_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return {"request": request, "round": round_n, "maxRounds": MAX_ROUNDS, "modelUnchanged": unchanged,
            "rules": round_rules(round_n, request=request, has_plan=prev is not None, model_unchanged=unchanged)}


def _round_state(workdir: Path) -> dict[str, Any]:
    p = Path(workdir) / "survey" / ROUND_FILE
    if not p.is_file():
        raise PlanError([f"no round is open: run `python -m tools.survey plan begin --request R` first ({p.name})"])
    return json.loads(p.read_text(encoding="utf-8"))


def check_round(workdir: Path) -> list[str]:
    """Every rule a host checks after Heimdal's run, against what ``begin_round`` kept. Empty is clean."""
    workdir = Path(workdir)
    st = _round_state(workdir)
    plan = read_plan(workdir)
    if plan is None:
        return ["survey/plan.json is missing: write the plan"]
    return (check_rounds(st["plan"], plan, round_n=st["round"], request=st["request"],
                         max_rounds=st.get("maxRounds", MAX_ROUNDS), model_unchanged=bool(st.get("modelUnchanged")))
            + check_evidence(workdir, plan, st.get("shas") or {}))


def record_round(workdir: Path, at: str) -> dict[str, Any]:
    """Append the round to iterations.jsonl once it checks clean; returns the round as the request shows it."""
    workdir = Path(workdir)
    problems = check_round(workdir)
    if problems:
        raise PlanError(problems)
    st = _round_state(workdir)
    plan = read_plan(workdir)
    assert plan is not None
    row = append_iteration(workdir, plan, at, st.get("modelSha256"))
    (workdir / "survey" / ROUND_FILE).unlink()
    return {"iteration": row, "round": round_summary(st["plan"], plan)}
