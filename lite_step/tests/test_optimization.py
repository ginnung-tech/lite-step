"""
Unit tests for lite_step.optimization package.

Verifies:
- @jit_compile decorator execution and fallback behavior
- Google OR-Tools CP-SAT solve_sequence on known permutations
- SciPy HiGHS solve_milp and solve_path solvers
"""

import logging
import math

import numpy as np
import pytest
from scipy.optimize import LinearConstraint

from lite_step.optimization import (
    HAS_ORTOOLS,
    jit_compile,
    solve_milp,
    solve_path,
    solve_sequence,
    solver_backends,
    solver_report,
)
from lite_step.optimization import combinatorial


def test_jit_compile_decorator():
    @jit_compile(fastmath=True, nogil=True)
    def add(a: float, b: float) -> float:
        return a + b

    assert math.isclose(add(3.5, 4.5), 8.0)


def test_solve_sequence_ortools():
    # Cost matrix where optimal path is clearly 0 -> 2 -> 1 -> 3
    # 0->2: 1, 2->1: 2, 1->3: 1 (total = 4)
    # other transitions are expensive (100)
    n = 4
    costs = np.full((n, n), 100.0)
    costs[0, 2] = 1.0
    costs[2, 1] = 2.0
    costs[1, 3] = 1.0
    costs[0, 1] = 50.0
    costs[1, 2] = 50.0

    seq = solve_sequence(n, costs, base_costs=np.array([0.0, 50.0, 50.0, 50.0]))
    assert seq == [0, 2, 1, 3]


def test_solve_path_scipy():
    n = 4
    costs = np.full((n, n), 100.0)
    costs[0, 1] = 2.0
    costs[1, 2] = 3.0
    costs[2, 3] = 1.0

    best_seq, total_c = solve_path(costs, start_costs=np.array([0.0, 50.0, 50.0, 50.0]))
    assert best_seq == [0, 1, 2, 3]
    assert math.isclose(total_c, 6.0)


def test_solve_milp():
    # Minimize 2*x0 + 3*x1 s.t. x0 + x1 >= 5, x0, x1 >= 0 integers
    c = np.array([2.0, 3.0])
    A = np.array([[1.0, 1.0]])
    constraints = [LinearConstraint(A, [5.0], [np.inf])]
    integrality = np.array([1, 1])
    bounds = (0.0, np.inf)

    x_opt, fun, success = solve_milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    assert success
    assert x_opt is not None
    assert math.isclose(x_opt[0], 5.0)
    assert math.isclose(x_opt[1], 0.0)
    assert math.isclose(fun, 10.0)


# ── Solver visibility + determinism ───────────────────────────────────────
#
# Two failures that shared a root: nothing said which path ran. CP-SAT and the
# brute-force fallback return the same shape and logged nothing, so "are the
# solvers live in production?" could only be answered by timing a build — and
# a silent degrade looked exactly like a working compile.

def _costs(n: int) -> np.ndarray:
    """A matrix whose cheapest order is the reverse of the input order, so an
    UNOPTIMISED answer is distinguishable from a solved one."""
    m = np.full((n, n), 9.0)
    for i in range(n - 1):
        m[n - 1 - i, n - 2 - i] = 0.0
    return m


def test_has_ortools_is_public_like_its_siblings():
    """The one flag that mattered was the one not exported. HAS_NUMBA and
    HAS_SCIPY were already public; HAS_ORTOOLS was not, so nothing could ask
    whether CP-SAT was live without reading source."""
    import lite_step.optimization as opt
    assert "HAS_ORTOOLS" in opt.__all__
    assert isinstance(HAS_ORTOOLS, bool)


def test_solver_report_names_every_backend_and_its_state():
    report = solver_report()
    assert report.startswith("solvers: ")
    for name in ("numba", "ortools", "scipy"):
        assert f"{name}=" in report
    # Live/MISSING, never a bare truthy value a reader has to interpret.
    for token in report.removeprefix("solvers: ").split():
        assert token.split("=")[1] in ("live", "MISSING")


def test_solver_backends_agrees_with_the_module_flags():
    b = solver_backends()
    assert b["ortools"] is HAS_ORTOOLS
    assert set(b) == {"ortools", "numba", "scipy"}


@pytest.mark.skipif(not HAS_ORTOOLS, reason="ortools not installed")
def test_the_cp_sat_path_announces_itself(caplog):
    """A build must be able to prove which solver it used from its own log."""
    combinatorial._reported_paths.clear()
    with caplog.at_level(logging.INFO, logger="lite_step.optimization.combinatorial"):
        solve_sequence(4, _costs(4))
    assert "CP-SAT" in caplog.text
    assert "solvers:" in caplog.text


@pytest.mark.skipif(not HAS_ORTOOLS, reason="ortools not installed")
def test_the_announcement_is_once_per_process_not_once_per_call(caplog):
    """A wall with 40 logs calls this 40 times; 40 identical lines would push
    the signal out of a bounded log."""
    combinatorial._reported_paths.clear()
    with caplog.at_level(logging.INFO, logger="lite_step.optimization.combinatorial"):
        for _ in range(5):
            solve_sequence(4, _costs(4))
    assert caplog.text.count("using ortools CP-SAT") == 1


def test_the_brute_force_fallback_announces_itself(monkeypatch, caplog):
    combinatorial._reported_paths.clear()
    monkeypatch.setattr(combinatorial, "HAS_ORTOOLS", False)
    with caplog.at_level(logging.WARNING, logger="lite_step.optimization.combinatorial"):
        order = solve_sequence(4, _costs(4))
    assert "ortools is MISSING" in caplog.text
    # Still exact at this size — the fallback is correct, just unscalable.
    assert order == [3, 2, 1, 0]


def test_the_fallback_refuses_a_size_it_cannot_finish(monkeypatch, caplog):
    """n! at 12 items is 479 million permutations: the compile stops producing
    output with no error, which reads as a crash. The DSL's callers are
    unbounded (`num_logs = len(logs)`), so the bound has to live in the solver.

    Delete the guard and this test hangs rather than fails — which is the point.
    """
    combinatorial._reported_paths.clear()
    monkeypatch.setattr(combinatorial, "HAS_ORTOOLS", False)
    n = combinatorial.BRUTE_FORCE_MAX_ITEMS + 3
    with caplog.at_level(logging.WARNING, logger="lite_step.optimization.combinatorial"):
        order = solve_sequence(n, np.zeros((n, n)))
    assert order == list(range(n))          # input order, UNOPTIMISED
    assert "UNOPTIMISED" in caplog.text
    assert "brute-force bound" in caplog.text


def test_the_fallback_still_solves_at_the_bound(monkeypatch):
    """The counter-test: the guard must not have swallowed the working range."""
    monkeypatch.setattr(combinatorial, "HAS_ORTOOLS", False)
    n = combinatorial.BRUTE_FORCE_MAX_ITEMS
    order = solve_sequence(n, _costs(n))
    assert order == list(range(n - 1, -1, -1))


@pytest.mark.skipif(not HAS_ORTOOLS, reason="ortools not installed")
def test_the_deciding_limit_is_deterministic_not_wall_clock():
    """`num_search_workers=1` + `random_seed=42` remove thread nondeterminism,
    but a wall-clock cutoff reintroduces it: the same model on a loaded machine
    stops earlier than on an idle one and can return a different permutation.

    Asserted on the parameters the solver is actually given, because the
    failure it guards against — two machines disagreeing — cannot be
    reproduced inside one test process.
    """
    seen = {}
    real_solver_cls = combinatorial.cp_model.CpSolver

    class RecordingSolver(real_solver_cls):  # type: ignore[misc,valid-type]
        def Solve(self, model, *a, **kw):  # noqa: N802
            seen["deterministic"] = self.parameters.max_deterministic_time
            seen["wall"] = self.parameters.max_time_in_seconds
            return super().Solve(model, *a, **kw)

    combinatorial.cp_model.CpSolver = RecordingSolver
    try:
        solve_sequence(4, _costs(4))
    finally:
        combinatorial.cp_model.CpSolver = real_solver_cls

    # Asserted as an EQUALITY, not "> 0": OR-Tools defaults this parameter to a
    # very large value, so a >0 check passes whether or not we set anything and
    # would have reported success on a solver still deciding by wall clock.
    # (Caught by removing the assignment and watching this test still pass.)
    assert seen["deterministic"] == pytest.approx(
        combinatorial.DEFAULT_MAX_DETERMINISTIC_TIME)
    # The wall clock is a backstop only: it must never be the tighter of the two.
    assert seen["wall"] >= seen["deterministic"]


@pytest.mark.skipif(not HAS_ORTOOLS, reason="ortools not installed")
def test_a_solver_failure_is_not_silent(monkeypatch, caplog):
    """It returned the identity permutation on failure — a valid sequence, an
    unoptimised one, and indistinguishable from a solved answer."""
    monkeypatch.setattr(
        combinatorial.cp_model.CpSolver, "Solve",
        lambda self, model, *a, **kw: combinatorial.cp_model.UNKNOWN,
    )
    monkeypatch.setattr(
        combinatorial.cp_model.CpSolver, "status_name",
        lambda self, s: "UNKNOWN",
    )
    with caplog.at_level(logging.WARNING, logger="lite_step.optimization.combinatorial"):
        order = solve_sequence(4, _costs(4))
    assert order == [0, 1, 2, 3]
    assert "UNOPTIMISED" in caplog.text
    assert "no solution" in caplog.text


def test_a_trivial_sequence_needs_no_solver_and_says_nothing(caplog):
    combinatorial._reported_paths.clear()
    with caplog.at_level(logging.INFO, logger="lite_step.optimization.combinatorial"):
        assert solve_sequence(1, np.zeros((1, 1))) == [0]
        assert solve_sequence(0, np.zeros((0, 0))) == []
    assert caplog.text == ""


@pytest.mark.skipif(not HAS_ORTOOLS, reason="ortools not installed")
def test_the_announcement_reaches_stderr_not_just_the_logger(capsys):
    """`lite_step` configures no logging handlers, so Python's last-resort
    handler carries WARNING+ and silently drops INFO. A healthy "CP-SAT is
    live" line is not a warning and must not be logged as one to buy
    visibility — so it also prints to stderr beside `carves:` and `csg depth:`,
    which use that channel for the same reason.

    Without this the line exists, the test suite sees it via caplog, and the
    compile output shows nothing — which is the exact failure being fixed.
    """
    combinatorial._reported_paths.clear()
    solve_sequence(4, _costs(4))
    err = capsys.readouterr().err
    assert "using ortools CP-SAT" in err
    assert "solvers:" in err
