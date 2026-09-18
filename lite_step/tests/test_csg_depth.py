"""Balanced CSG trees + the boolean-depth budget.

The failure these pin is INVISIBLE in the IFC: the primary web viewer stops
evaluating a deeply nested ``IfcBooleanResult`` chain and silently renders the
un-subtracted base solid. A foundation wall with 13 sequential DIFFERENCE
nestings came out as a solid block that swallowed the floor beams; a roof
covering at 30+ booleans was dropped entirely. Nothing logged an error anywhere.

Two properties are asserted here and must never both drift:

* the emitted tree is BALANCED (depth ``ceil(log2 V)`` instead of ``V``), and
  is byte-for-byte the left-deep output when there is only one operand;
* the predicted depth the compiler PRINTS equals the depth actually emitted.
  A metric that can lie about the one number it exists to report is worse than
  no metric, so the prediction is measured against the real ifcopenshell model.
"""

from __future__ import annotations

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.compiler.csg_depth import predict_csg_depth
from lite_step.ifc.boolean_tree import (
    CSG_DEPTH_BUDGET, apply_boolean_chain, balanced_depth, balanced_tree,
    balanced_union,
)
from lite_step.ifc.generator import generate_ifc
from lite_step.models import Box, Point, Project
from lite_step.models.project import Storey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bldg(*elements) -> Project:
    p = Project(name="t")
    s = Storey(elevation=0)
    for e in elements:
        s.add(e)
    p.add_storey(s)
    return p


def _actual_max_depth(ifc_content: str) -> int:
    """Deepest ``IfcBooleanResult`` nesting in the emitted model, measured by
    walking it with ifcopenshell (NOT by re-deriving it from the DSL — that
    would just re-assert the prediction against itself)."""
    model = ifcopenshell.file.from_string(ifc_content)
    memo: dict = {}

    def depth(entity) -> int:
        if entity is None or not entity.is_a("IfcBooleanResult"):
            return 0
        key = entity.id()
        if key in memo:
            return memo[key]
        memo[key] = 1  # cycle guard; IFC boolean trees are acyclic
        memo[key] = max(depth(entity.FirstOperand),
                        depth(entity.SecondOperand)) + 1
        return memo[key]

    return max((depth(e) for e in model.by_type("IfcBooleanResult")), default=0)


class _Recorder:
    """A ``create_entity`` stand-in that records the order operations happen in."""

    def __init__(self):
        self.calls: list = []

    def __call__(self, operator, first, second):
        node = (operator, first, second)
        self.calls.append(("bool", operator, first, second))
        return node

    def build(self, operand):
        self.calls.append(("build", operand))
        return operand


def _left_deep_reference(create, base, cuts=(), adds=(), intersects=(), build=None):
    """The PRE-balancing emitter, verbatim. Kept in the test as the reference
    the single-operand case must still match exactly."""
    result = base
    for c in cuts:
        s = build(c)
        if s is not None:
            result = create("DIFFERENCE", result, s)
    for a in adds:
        s = build(a)
        if s is not None:
            result = create("UNION", result, s)
    for i in intersects:
        s = build(i)
        if s is not None:
            result = create("INTERSECTION", result, s)
    return result


# ---------------------------------------------------------------------------
# WS1 — the tree shape
# ---------------------------------------------------------------------------


def test_balanced_union_of_one_operand_creates_nothing():
    rec = _Recorder()
    assert balanced_union(rec, ["solo"]) == "solo"
    assert rec.calls == []            # not a single entity emitted


def test_balanced_union_is_logarithmic_not_linear():
    rec = _Recorder()
    balanced_tree(rec, "UNION", list(range(13)))
    # 13 operands still need 12 binary UNIONs — balancing changes the SHAPE,
    # not the entity count (that matters: file size must not move).
    assert len(rec.calls) == 12
    assert balanced_depth([0] * 13) == 4       # vs 12 left-deep


def test_empty_operand_group_is_a_programming_error():
    with pytest.raises(ValueError):
        balanced_union(_Recorder(), [])


@pytest.mark.parametrize("cuts,adds,ints", [
    ([1], [], []),
    ([], [1], []),
    ([], [], [1]),
    ([1], [2], []),
    ([1], [2], [3]),
])
def test_single_operand_per_class_is_byte_identical_to_the_old_emitter(cuts, adds, ints):
    """THE no-op invariant. With one operand in a class, balancing must produce
    the same entities, with the same operands, created in the same ORDER — same
    STEP ids, same bytes. A model that gains no depth must not churn."""
    new_rec, old_rec = _Recorder(), _Recorder()
    new = apply_boolean_chain(new_rec, "BASE", cuts=cuts, adds=adds,
                              intersects=ints, build=new_rec.build)
    old = _left_deep_reference(old_rec, "BASE", cuts=cuts, adds=adds,
                               intersects=ints, build=old_rec.build)
    assert new == old
    assert new_rec.calls == old_rec.calls


def test_unbuildable_operands_emit_no_boolean():
    """A ``build`` that returns None (unsupported operand shape) must drop out
    of the tree entirely rather than nesting a null operand."""
    rec = _Recorder()
    result = apply_boolean_chain(rec, "BASE", cuts=["a", "b"],
                                 build=lambda o: None)
    assert result == "BASE"
    assert not [c for c in rec.calls if c[0] == "bool"]


def test_thirteen_cuts_collapse_from_thirteen_deep_to_five():
    """The measured real-world failure: a foundation wall with 13 pockets.
    Left-deep that is 13 nestings — past what the viewer evaluates, so it
    rendered as an uncarved block. Balanced it is 5."""
    host = Box(name="host", start=Point(x=0, y=0, z=0),
               end=Point(x=13000, y=1000, z=1000))
    for i in range(13):
        host.difference(Box(start=Point(x=i * 1000 + 100, y=-100, z=200),
                            end=Point(x=i * 1000 + 900, y=1100, z=800)))
    res = generate_ifc(_bldg(host), source_code=None)
    assert res.success, res.error
    assert _actual_max_depth(res.ifc_content) == 5
    # …and all 13 subtractions are still there (13 UNIONs of cuts + 1 DIFFERENCE)
    assert res.ifc_content.count("IFCBOOLEANRESULT(") == 13


# ---------------------------------------------------------------------------
# WS2 — the metric and the budget
# ---------------------------------------------------------------------------


def test_predicted_depth_equals_emitted_depth_with_cuts_adds_and_clips():
    """The honesty pin. Cuts + adds + clips on one element, all three classes
    plus a nested operand — the prediction must equal what ifcopenshell sees."""
    host = Box(name="host", start=Point(x=0, y=0, z=0),
               end=Point(x=6000, y=2000, z=3000))
    for i in range(5):
        host.difference(Box(start=Point(x=i * 1000 + 100, y=-100, z=100),
                            end=Point(x=i * 1000 + 900, y=2100, z=900)))
    merged = Box(start=Point(x=1000, y=500, z=1500),
                 end=Point(x=2000, y=1500, z=2500))
    merged.union(Box(start=Point(x=1500, y=600, z=1600),
                     end=Point(x=2500, y=1400, z=2400)))
    host.union(merged)
    host.intersection(Box(start=Point(x=-1000, y=-1000, z=-1000),
                          end=Point(x=5500, y=3000, z=4000)))
    host.clip(origin=Point(x=5000, y=0, z=0), normal=(1.0, 0.0, 0.0))
    host.clip(origin=Point(x=0, y=0, z=2800), normal=(0.0, 0.0, 1.0))

    proj = _bldg(host)
    res = generate_ifc(proj, source_code=None)
    assert res.success, res.error
    report = predict_csg_depth(proj)
    assert report.max_depth == _actual_max_depth(res.ifc_content)
    assert report.max_depth > 0        # the fixture must actually exercise CSG


def test_stats_line_is_printed_on_every_compile(capsys):
    box = Box(name="plain", start=Point(x=0, y=0, z=0),
              end=Point(x=1000, y=1000, z=1000))
    res = generate_ifc(_bldg(box), source_code=None)
    assert res.success, res.error
    err = capsys.readouterr().err
    assert "csg depth: max=0" in err
    assert f"budget {CSG_DEPTH_BUDGET}" in err


def test_over_budget_warns_loudly_but_does_not_fail_the_compile(capsys):
    """Explicit product decision: over budget is a WARNING. A deep model must
    still compile and still download — the author needs to know, not be blocked."""
    # Nested merged voids: each level adds a nesting the balanced tree can't
    # flatten, so this clears the budget without needing hundreds of operands.
    def _stack(depth: int, x: int):
        v = Box(start=Point(x=x, y=-100, z=100), end=Point(x=x + 400, y=1100, z=900))
        node = v
        for k in range(depth):
            child = Box(start=Point(x=x + 10 * k, y=-90, z=110),
                        end=Point(x=x + 400 + 10 * k, y=1090, z=890))
            node.union(child)
            node = child
        return v

    host = Box(name="deep", start=Point(x=0, y=0, z=0),
               end=Point(x=8000, y=1000, z=1000))
    host.difference(_stack(CSG_DEPTH_BUDGET + 2, 500))

    proj = _bldg(host)
    res = generate_ifc(proj, source_code=None)
    assert res.success, res.error                    # NOT an error
    assert predict_csg_depth(proj).max_depth > CSG_DEPTH_BUDGET
    err = capsys.readouterr().err
    assert "warning: csg depth" in err
    assert "exceeds the budget" in err
    assert "deep" in err                              # names the element


def test_report_warnings_are_empty_inside_budget():
    box = Box(name="ok", start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000))
    box.difference(Box(start=Point(x=100, y=100, z=100),
                       end=Point(x=200, y=200, z=200)))
    report = predict_csg_depth(_bldg(box))
    assert report.warnings() == []
    assert report.max_depth == 1
