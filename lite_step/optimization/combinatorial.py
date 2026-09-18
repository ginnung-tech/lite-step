"""
Combinatorial Optimization Utilities for Lite-STEP.

Provides Google OR-Tools CP-SAT solvers for architectural element sequencing,
tier assignment, orientation selection, and nesting problems.

**Every path this module can take says so out loud.** Which solver ran was
invisible — CP-SAT and the brute-force fallback return the same shape and
logged nothing — and that is most of why "are the solvers actually live in
production?" took a day to answer. A compile that silently degrades to an
exponential fallback, or silently returns an unoptimised answer, must not look
identical to one that worked.
"""

import logging
import math
import sys
from typing import Any

import numpy as np

try:
    from ortools.sat.python import cp_model

    HAS_ORTOOLS = True
except ImportError:
    HAS_ORTOOLS = False
    cp_model = None

logger = logging.getLogger(__name__)

#: Above this item count the exact fallback is refused rather than attempted.
#: The fallback enumerates every permutation, so cost is n! — 9 items is 363k
#: and survivable, 12 is 479 million and hangs the compile with no output and
#: no error. The DSL's own callers are unbounded (``num_logs = len(logs)``), so
#: the bound has to live here.
BRUTE_FORCE_MAX_ITEMS = 9

#: Deterministic work units allowed before CP-SAT gives up. THIS is the limit
#: that decides the answer — see :func:`solve_sequence`.
DEFAULT_MAX_DETERMINISTIC_TIME = 30.0

#: Wall-clock backstop, so a pathological model cannot wedge a compile even if
#: the deterministic budget has not been spent. Never the deciding limit on any
#: machine that is not already in trouble.
DEFAULT_MAX_WALL_SECONDS = 60.0

#: One report per process, not per call — a wall with 40 logs would otherwise
#: emit 40 identical lines.
_reported_paths: set[str] = set()


def solver_backends() -> dict[str, bool]:
    """Which optional solver backends imported, as a plain dict.

    Exists because "are the solvers live in this image?" had no answer that did
    not involve reading source or timing a build. ``HAS_NUMBA`` and
    ``HAS_SCIPY`` were already public; ``HAS_ORTOOLS`` was not — which is
    exactly the one that mattered.
    """
    from lite_step.optimization.jit import HAS_NUMBA
    from lite_step.optimization.linear import HAS_SCIPY

    return {"ortools": HAS_ORTOOLS, "numba": HAS_NUMBA, "scipy": HAS_SCIPY}


def solver_report() -> str:
    """One line naming every backend and whether it is live.

    ``solvers: numba=live ortools=live scipy=MISSING``
    """
    return "solvers: " + " ".join(
        f"{name}={'live' if ok else 'MISSING'}"
        for name, ok in sorted(solver_backends().items())
    )


def _report_once(key: str, level: int, msg: str, *args: Any) -> None:
    """Announce ``msg`` the first time ``key`` is seen in this process.

    Goes to BOTH the logger and stderr, on purpose.

    ``lite_step`` configures no logging handlers, so Python's last-resort
    handler carries WARNING and above and drops INFO on the floor. A healthy
    "CP-SAT is live" line is not a warning and must not be logged as one to buy
    visibility — but it is exactly the line whose absence made "are the solvers
    live in production?" unanswerable. So it prints to stderr alongside the
    compile's other report lines (``carves:``, ``csg depth:``), which use the
    same channel for the same reason, and it stays a real log record for
    downstream breadcrumbs and for tests.
    """
    if key in _reported_paths:
        return
    _reported_paths.add(key)
    logger.log(level, msg, *args)
    try:
        print(msg % args if args else msg, file=sys.stderr)
    except Exception:  # pragma: no cover - a report must never break a compile
        pass


def solve_sequence(
    n_items: int,
    transition_costs: dict[tuple[int, int], float] | np.ndarray,
    base_costs: dict[int, float] | np.ndarray | None = None,
    *,
    max_time_seconds: float = DEFAULT_MAX_WALL_SECONDS,
    max_deterministic_time: float = DEFAULT_MAX_DETERMINISTIC_TIME,
) -> list[int]:
    """Solves for the optimal sequence (permutation) of n items to minimize total cost.

    Guarantees deterministic, reproducible results across platforms.

    Args:
        n_items: Number of elements/layers to sequence.
        transition_costs: Mapping or 2D array of (from_item, to_item) -> cost.
        base_costs: Optional cost for choosing the initial base element at layer 0.
        max_time_seconds: Wall-clock backstop. NOT the limit that decides the
            answer — see ``max_deterministic_time``.
        max_deterministic_time: CP-SAT deterministic work budget. This is the
            deciding limit, and it is what makes the result reproducible.

    Returns:
        List of item indices representing the optimal sequence [item_0, item_1, ..., item_{N-1}].

    **On determinism.** ``num_search_workers=1`` + ``random_seed=42`` remove
    thread nondeterminism, but a WALL-CLOCK limit reintroduces it by the back
    door: the same model on a loaded machine hits the limit and returns a
    merely-FEASIBLE permutation where an idle machine returns the OPTIMAL one.
    Same input, different geometry, depending on what else the box was doing.
    ``max_deterministic_time`` counts work units instead, so the cutoff falls in
    the same place everywhere; the wall clock stays only as a backstop against a
    wedged compile.

    A result that is FEASIBLE rather than OPTIMAL, and a solver that fails
    outright, are both reported. Returning the identity permutation in silence
    would give a valid sequence and an unoptimised one, which the
    caller had no way to tell apart from a solved answer.
    """
    if n_items <= 1:
        return list(range(n_items))

    def get_base_cost_int(idx: int) -> int:
        if base_costs is None:
            return 0
        raw = base_costs[idx] if isinstance(base_costs, np.ndarray) else base_costs.get(idx, 0.0)
        return int(round(raw * 100))

    def get_trans_cost_int(i: int, j: int) -> int:
        raw = transition_costs[i, j] if isinstance(transition_costs, np.ndarray) else transition_costs.get((i, j), 0.0)
        return int(round(raw * 100))

    # Deterministic exact brute force fallback when OR-Tools is absent
    if not HAS_ORTOOLS:
        import itertools

        if n_items > BRUTE_FORCE_MAX_ITEMS:
            # Refuse rather than hang. n! at 12 items is 479 million
            # permutations: the compile stops producing output and there is no
            # error to read, which is indistinguishable from a crash. The input
            # order is a correct sequence, just not a good one, so this
            # degrades loudly instead of failing.
            logger.warning(
                "solve_sequence: ortools is MISSING and %d items is past the "
                "brute-force bound of %d (%.3g permutations) - returning the "
                "input order UNOPTIMISED. Install the ortools extra to get a "
                "real answer at this size.",
                n_items, BRUTE_FORCE_MAX_ITEMS, math.factorial(n_items),
            )
            return list(range(n_items))
        _report_once(
            "brute-force", logging.WARNING,
            "solve_sequence: ortools is MISSING - using the exact brute-force "
            "fallback (n! and correct, but it does not scale). %s",
            solver_report(),
        )

        best_cost = float("inf")
        best_perm = list(range(n_items))
        for perm in itertools.permutations(range(n_items)):
            cost = get_base_cost_int(perm[0])
            for k in range(n_items - 1):
                cost += get_trans_cost_int(perm[k], perm[k + 1])
            if cost < best_cost:
                best_cost = cost
                best_perm = list(perm)
        return best_perm

    model = cp_model.CpModel()

    # Decision variables: x[i, k] == 1 iff item i is placed at layer k
    x: dict[tuple[int, int], Any] = {}
    for i in range(n_items):
        for k in range(n_items):
            x[i, k] = model.NewBoolVar(f"x_{i}_{k}")

    # Exactly one item per layer, exactly one layer per item
    for k in range(n_items):
        model.AddExactlyOne(x[i, k] for i in range(n_items))
    for i in range(n_items):
        model.AddExactlyOne(x[i, k] for k in range(n_items))

    obj_terms = []

    # Optional base placement cost
    if base_costs is not None:
        for i in range(n_items):
            c_val = get_base_cost_int(i)
            if c_val != 0:
                obj_terms.append(c_val * x[i, 0])

    # Inter-layer transition costs
    for k in range(n_items - 1):
        for i in range(n_items):
            for j in range(n_items):
                if i == j:
                    continue
                cost_int = get_trans_cost_int(i, j)
                t_var = model.NewBoolVar(f"trans_{k}_{i}_{j}")
                model.AddBoolAnd([x[i, k], x[j, k + 1]]).OnlyEnforceIf(t_var)
                model.AddBoolOr([x[i, k].Not(), x[j, k + 1].Not()]).OnlyEnforceIf(t_var.Not())
                obj_terms.append(cost_int * t_var)

    model.Minimize(sum(obj_terms))

    _report_once("cp-sat", logging.INFO,
                 "solve_sequence: using ortools CP-SAT. %s", solver_report())

    solver = cp_model.CpSolver()
    # The deciding limit is the DETERMINISTIC one; the wall clock is only a
    # backstop against a wedged compile. Setting the wall clock alone made the
    # answer depend on how busy the machine was.
    solver.parameters.max_deterministic_time = max_deterministic_time
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 1  # Deterministic single-threaded search
    solver.parameters.random_seed = 42
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        if status != cp_model.OPTIMAL:
            # Hit a limit before proving optimality. The answer is valid but
            # unproven, and - the part that matters - it is the answer THIS run
            # happened to reach, so two runs can differ. Loud, because geometry
            # that moves between compiles is the hardest kind of bug to chase.
            logger.warning(
                "solve_sequence: CP-SAT returned %s, not OPTIMAL, for %d items "
                "(deterministic budget %.1f, wall %.1fs). The sequence is valid "
                "but unproven, and another run may return a different one. "
                "Raise max_deterministic_time if this model needs a stable answer.",
                solver.status_name(status), n_items,
                max_deterministic_time, max_time_seconds,
            )
        result = []
        for k in range(n_items):
            for i in range(n_items):
                if solver.Value(x[i, k]):
                    result.append(i)
                    break
        return result

    # No solution at all. The input order is still a valid sequence, so this
    # degrades rather than raising - but doing so in silence is the hazard:
    # a caller cannot tell an unoptimised answer from a solved one by
    # looking at it.
    logger.warning(
        "solve_sequence: CP-SAT found no solution for %d items (status %s) - "
        "returning the input order UNOPTIMISED. Nothing was sequenced.",
        n_items, solver.status_name(status),
    )
    return list(range(n_items))
