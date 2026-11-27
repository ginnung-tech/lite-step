"""Heimdal's plan across rounds (tools/survey/plan.py, contract survey 1.5).

The fixtures are a real carport plan (2026-11-26/27): round 3 there moved the post height from 2465 (+/- 85) to
2060 (+/- 198) mm without saying so and dropped round 1's open items. Each rule below is what would have caught it.
The same checker runs when a host runs Heimdal; these tests hold the copy here to the same fixtures.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.survey import plan as sp  # noqa: E402
from tools.survey.__main__ import main  # noqa: E402

FIX = Path(__file__).resolve().parent / "fixtures" / "survey_plan"
REQ = "9f1c2e4a-0b7d-4c55-9f0e-2a1d6c3b8e71"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))




# -- the checker agrees with the contract ------------------------------------------------------------------


@pytest.mark.parametrize("name", ["plan_1_5.json", "plan_round2_1_5.json", "plan_round3_ask_1_5.json"])
def test_valid_plans_pass_the_checker(name):
    plan = _load(name)
    assert sp.check_shape(plan) == []


@pytest.mark.parametrize("path", sorted((FIX / "invalid").glob("plan-*.json")), ids=lambda p: p.name)
def test_invalid_plans_fail_the_checker(path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    assert sp.check_shape(plan), f"{path.name} passed the checker but breaks the contract"


# -- the rules across rounds -------------------------------------------------------------------------------


def test_round_two_as_written_keeps_every_rule():
    assert sp.check_rounds(_load("plan_1_5.json"), _load("plan_round2_1_5.json"), round_n=2, request=REQ) == []


def test_a_value_moved_beyond_its_uncertainty_without_a_change_is_refused():
    new = _load("plan_round2_1_5.json")
    new["changes"] = []
    bad = sp.check_rounds(_load("plan_1_5.json"), new, round_n=2, request=REQ)
    assert any("post-height" in b and "2465" in b and "2060" in b for b in bad), bad


def test_a_change_recorded_for_an_earlier_round_does_not_cover_this_one():
    new = _load("plan_round2_1_5.json")
    new["changes"][0]["round"] = 1
    assert any("post-height" in b for b in sp.check_rounds(_load("plan_1_5.json"), new, round_n=2, request=REQ))


def test_a_move_inside_the_two_readings_uncertainty_is_not_a_change():
    prev = _load("plan_1_5.json")
    new = _load("plan_round2_1_5.json")
    new["changes"] = []
    for i in new["items"]:
        if i["id"] == "post-height":
            i.update(value=2465 + 200, uncertainty=198)  # hypot(85, 198) = 215 > 200
    assert sp.check_rounds(prev, new, round_n=2, request=REQ) == []
    for i in new["items"]:
        if i["id"] == "post-height":
            i["value"] = 2465 + 220
    assert sp.check_rounds(prev, new, round_n=2, request=REQ)


def test_a_layout_whose_text_changes_is_a_change():
    new = _load("plan_round2_1_5.json")
    new["items"][0]["value"] = "4 rows of 3 posts"
    assert any("layout" in b for b in sp.check_rounds(_load("plan_1_5.json"), new, round_n=2, request=REQ))


def test_an_open_entry_that_vanishes_without_a_reason_is_refused():
    new = _load("plan_round2_1_5.json")
    new["closed"] = []  # photo-b dropped silently, as round 3 of the carport did
    bad = sp.check_rounds(_load("plan_1_5.json"), new, round_n=2, request=REQ)
    assert any("photo-b" in b for b in bad), bad


def test_an_item_that_vanishes_without_a_reason_is_refused_and_closing_it_is_enough():
    prev = _load("plan_1_5.json")
    new = _load("plan_round2_1_5.json")
    new["items"] = [i for i in new["items"] if i["id"] != "post"]
    assert any("item post " in b for b in sp.check_rounds(prev, new, round_n=2, request=REQ))
    new["closed"].append({"id": "post", "reason": "Merged into the post height item.", "round": 2})
    assert sp.check_rounds(prev, new, round_n=2, request=REQ) == []


def test_the_last_round_cannot_instruct_but_may_ask_with_the_instruction():
    prev = _load("plan_round2_1_5.json")
    ask = _load("plan_round3_ask_1_5.json")
    assert sp.check_rounds(prev, ask, round_n=3, request=REQ) == []
    instruct = copy.deepcopy(ask)
    instruct["decision"].update(kind="instruct", checkSpec=_load("plan_1_5.json")["decision"]["checkSpec"])
    instruct["decision"].pop("question")
    assert any("last of 3" in b for b in sp.check_rounds(prev, instruct, round_n=3, request=REQ))


def test_round_count_and_request_are_checked():
    plan = _load("plan_round2_1_5.json")
    assert any("decision.round must be 3" in b for b in sp.check_rounds(None, plan, round_n=3, request=REQ))
    assert any("decision.request" in b for b in sp.check_rounds(None, plan, round_n=2, request="another"))
    plan["decision"]["maxRounds"] = 9
    assert any("maxRounds" in b for b in sp.check_rounds(None, plan, round_n=2, request=REQ))


def test_a_plan_that_is_not_json_is_an_error_never_no_plan(tmp_path):
    (tmp_path / "survey").mkdir()
    assert sp.read_plan(tmp_path) is None
    (tmp_path / "survey" / "plan.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(sp.PlanError):
        sp.read_plan(tmp_path)


# -- rounds and what the request shows ---------------------------------------------------------------------


def test_rounds_are_counted_per_request(tmp_path):
    assert sp.next_round(tmp_path, REQ) == 1
    row = sp.append_iteration(tmp_path, _load("plan_1_5.json"), "2026-11-27T10:00:00Z", "a" * 64)
    assert (row["met"], row["total"], row["open"], row["contradictions"]) == (3, 4, 2, 0)
    other = _load("plan_1_5.json")
    other["decision"]["request"] = "another-request"
    sp.append_iteration(tmp_path, other, "2026-11-27T10:05:00Z")
    assert sp.next_round(tmp_path, REQ) == 2
    assert sp.next_round(tmp_path, "another-request") == 2
    assert sp.next_round(tmp_path, "a-third") == 1


def test_the_round_summary_is_what_the_request_shows():
    out = sp.round_summary(_load("plan_1_5.json"), _load("plan_round2_1_5.json"))
    assert out["changed"] == [{"item": "Post height, slab to beam", "from": "2465 mm", "to": "2060 mm",
                               "reason": _load("plan_round2_1_5.json")["changes"][0]["reason"]}]
    assert out["closed"] == [{"text": "Photo b (inside) is not labelled yet.",
                              "reason": "Labelled in round 2; its points agree with photos a and c."}]
    assert (out["round"], out["maxRounds"], out["decision"], out["met"], out["total"]) == (2, 3, "instruct", 4, 5)
    ask = sp.round_summary(_load("plan_round2_1_5.json"), _load("plan_round3_ask_1_5.json"))
    assert ask["question"] and ask["instruction"] and ask["changed"] == []


def test_a_unit_outside_the_contract_is_refused_on_text_items_too():
    # 2026-11-27: Heimdal wrote "unit": "layout" on text items; the schema refuses it, so must the checker
    plan = _load("plan_1_5.json")
    plan["items"][0]["unit"] = "layout"
    assert any("unit is one of" in b for b in sp.check_shape(plan))


def test_a_requests_first_round_reports_no_check():
    # 2026-11-27: round 1 of a new request said the last step's check failed (another request's step)
    plan = _load("plan_1_5.json")
    assert sp.check_rounds(None, plan, round_n=1, request=REQ) == []
    plan["decision"]["previousCheck"] = "failed"
    plan["decision"]["checkResult"] = {"sources": ["photo-a-outside-wide"], "measured": 40, "expected": 0,
                                       "tolerance": 15, "unit": "px", "output": "measure/round-1.json"}
    assert any("first round" in b for b in sp.check_rounds(None, plan, round_n=1, request=REQ))


# -- survey 1.5: a check is a measurement on the person's pictures, with its evidence -------------------------
#
# 2026-11-27: a round converged on "the top beams are present in the model" while the person's photo had
# never been lined up, and the beams it built ran the wrong way. The model is what is checked; it cannot be the
# evidence.

def _evidence_dir(tmp_path, *, fit="passed", kind="photo", output=b"measured"):
    s = tmp_path / "survey"
    (s / "measure").mkdir(parents=True)
    (s / "sources.json").write_text(json.dumps({"sources": [{"id": "photo-a-outside-wide", "kind": kind}]}))
    (s / "cameras.json").write_text(json.dumps({"cameras": [{"source": "photo-a-outside-wide", "gate": fit}]}))
    (s / "measure" / "round-2.json").write_bytes(output)
    return tmp_path


def test_a_converged_round_needs_a_passed_measurement():
    plan = _load("plan_converged_1_5.json")
    assert sp.check_shape(plan) == []
    plan["decision"].pop("checkResult")
    assert any("converged needs a passed check" in b for b in sp.check_shape(plan))


def test_a_check_measured_on_the_persons_aligned_picture_and_written_this_round_is_evidence(tmp_path):
    wd = _evidence_dir(tmp_path)
    assert sp.check_evidence(wd, _load("plan_converged_1_5.json"), written_before={}) == []


def test_an_unaligned_picture_measures_nothing(tmp_path):
    wd = _evidence_dir(tmp_path, fit="failed")
    bad = sp.check_evidence(wd, _load("plan_converged_1_5.json"), written_before={})
    assert any("no lined-up camera" in b for b in bad), bad


def test_the_apps_own_screenshot_is_not_the_persons_picture(tmp_path):
    wd = _evidence_dir(tmp_path, kind="viewer_screenshot")
    assert any("not the person's picture" in b for b in sp.check_evidence(wd, _load("plan_converged_1_5.json"), {}))


def test_a_measurement_not_written_this_round_is_refused(tmp_path):
    import hashlib

    wd = _evidence_dir(tmp_path, output=b"from an earlier round")
    before = {"measure/round-2.json": hashlib.sha256(b"from an earlier round").hexdigest()}
    assert any("not written this round" in b for b in sp.check_evidence(wd, _load("plan_converged_1_5.json"), before))


def test_a_measurement_outside_the_survey_is_refused(tmp_path):
    wd = _evidence_dir(tmp_path)
    plan = _load("plan_converged_1_5.json")
    plan["decision"]["checkResult"]["output"] = "../output.ifc"
    assert any("not a file in survey/" in b for b in sp.check_evidence(wd, plan, {}))


def test_a_verdict_must_agree_with_its_number():
    plan = _load("plan_round2_1_5.json")
    plan["decision"]["checkResult"]["measured"] = 40  # outside 0 +/- 15, yet "passed"
    assert any("disagrees with checkResult" in b for b in sp.check_shape(plan))


# -- by hand: plan begin / check / record ------------------------------------------------------------------


def _survey(tmp_path: Path, plan: dict | None = None) -> Path:
    (tmp_path / "survey").mkdir()
    (tmp_path / "output.ifc").write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
    photos = [{"id": s, "kind": "photo"} for s in _load("plan_1_5.json")["decision"]["checkSpec"]["sources"]]
    (tmp_path / "survey" / "sources.json").write_text(json.dumps({"sources": photos}), encoding="utf-8")
    if plan is not None:
        (tmp_path / "survey" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return tmp_path


def _first_round(request: str) -> dict:
    p = _load("plan_1_5.json")
    p["decision"].update(round=1, request=request, previousCheck="not_checked")
    return p


def test_begin_check_record_runs_a_round_by_hand(tmp_path, capsys):
    wd = _survey(tmp_path)
    assert main(["plan", "begin", "--workdir", str(wd), "--request", REQ]) == 0
    began = json.loads(capsys.readouterr().out)
    assert (began["round"], began["maxRounds"]) == (1, sp.MAX_ROUNDS)
    assert any("first round" in r for r in began["rules"])
    (wd / "survey" / "plan.json").write_text(json.dumps(_first_round(REQ)), encoding="utf-8")
    assert main(["plan", "check", "--workdir", str(wd)]) == 0, capsys.readouterr().out
    capsys.readouterr()
    assert main(["plan", "record", "--workdir", str(wd)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["iteration"]["round"] == 1 and out["round"]["decision"] == "instruct"
    assert not (wd / "survey" / sp.ROUND_FILE).exists()
    assert sp.next_round(wd, REQ) == 2


def test_record_refuses_a_plan_that_breaks_a_rule_and_keeps_the_round_open(tmp_path, capsys):
    wd = _survey(tmp_path)
    sp.begin_round(wd, REQ)
    bad = _first_round(REQ)
    bad["decision"]["round"] = 2
    (wd / "survey" / "plan.json").write_text(json.dumps(bad), encoding="utf-8")
    assert main(["plan", "record", "--workdir", str(wd)]) == 1
    assert any("decision.round must be 1" in p for p in json.loads(capsys.readouterr().out)["problems"])
    assert (wd / "survey" / sp.ROUND_FILE).exists() and sp.iterations(wd) == []


def test_check_compares_with_the_plan_the_round_began_from(tmp_path):
    """A vanished open entry is caught against what begin kept, not against whatever is on disk later."""
    wd = _survey(tmp_path, _first_round(REQ))
    sp.append_iteration(wd, _first_round(REQ), "2026-11-27T10:00:00Z", "0" * 64)  # round 1 measured another model
    sp.begin_round(wd, REQ)
    nxt = _load("plan_round2_1_5.json")
    nxt["decision"]["request"] = REQ
    nxt["open"] = []
    (wd / "survey" / "plan.json").write_text(json.dumps(nxt), encoding="utf-8")
    assert any("is gone" in p for p in sp.check_round(wd))


def test_an_unchanged_model_is_noticed(tmp_path):
    wd = _survey(tmp_path, _first_round(REQ))
    sha = sp._sha256((wd / "output.ifc").read_bytes())
    sp.append_iteration(wd, _first_round(REQ), "2026-11-27T10:00:00Z", sha)
    assert sp.begin_round(wd, REQ)["modelUnchanged"] is True


def test_check_without_begin_says_how_to_start(tmp_path, capsys):
    wd = _survey(tmp_path, _first_round(REQ))
    assert main(["plan", "check", "--workdir", str(wd)]) == 1
    assert "plan begin" in json.loads(capsys.readouterr().out)["problems"][0]
