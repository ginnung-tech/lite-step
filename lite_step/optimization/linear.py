"""
Linear and Mixed-Integer Optimization Utilities for Lite-STEP.

Provides:
- High-level SciPy MILP (HiGHS) wrapper for discrete and continuous optimization
- Vectorized dynamic programming path and transition matching
"""

from typing import Any
import numpy as np

try:
    from scipy.optimize import Bounds, LinearConstraint, milp

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    milp = None
    LinearConstraint = None
    Bounds = None


def solve_milp(
    c: np.ndarray,
    *,
    integrality: np.ndarray | list[int] | None = None,
    bounds: tuple[np.ndarray | float, np.ndarray | float] | None = None,
    constraints: list[Any] | None = None,
) -> tuple[np.ndarray | None, float, bool]:
    """Solves a Mixed-Integer Linear Program (MILP) using SciPy's C++ HiGHS solver.

    Minimizes:
        c @ x
    Subject to:
        constraints and bounds.

    Returns:
        (optimal_x, optimal_objective, success)
    """
    if not HAS_SCIPY:
        raise RuntimeError("scipy is required to use solve_milp.")

    c_arr = np.asarray(c, dtype=np.float64)
    res = milp(
        c=c_arr,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )
    if res.success:
        return res.x, float(res.fun), True
    return None, float("inf"), False


def solve_path(
    cost_matrix: np.ndarray,
    start_costs: np.ndarray | None = None,
) -> tuple[list[int], float]:
    """Finds the minimum-cost sequence (shortest Hamiltonian path) over N elements.

    Uses memoized dynamic programming (Held-Karp algorithm) for exact, instant
    resolution on N <= 16 elements.

    Args:
        cost_matrix: NxN matrix where cost_matrix[i, j] is the transition cost from item i to j.
        start_costs: Optional 1D array of initial costs for starting at item i.

    Returns:
        (best_sequence, total_cost)
    """
    n = len(cost_matrix)
    if n <= 1:
        return list(range(n)), 0.0

    memo: dict[tuple[int, int], tuple[float, list[int]]] = {}

    def tsp(mask: int, curr: int) -> tuple[float, list[int]]:
        if mask == (1 << n) - 1:
            return 0.0, []
        state = (mask, curr)
        if state in memo:
            return memo[state]

        best_c = 1e18
        best_p: list[int] = []
        for nxt in range(n):
            if not (mask & (1 << nxt)):
                if curr < 0:
                    c = start_costs[nxt] if start_costs is not None else 0.0
                else:
                    c = float(cost_matrix[curr, nxt])
                rem_c, rem_p = tsp(mask | (1 << nxt), nxt)
                tot = c + rem_c
                if tot < best_c:
                    best_c = tot
                    best_p = [nxt] + rem_p

        memo[state] = (best_c, best_p)
        return memo[state]

    best_cost, best_perm = tsp(0, -1)
    return best_perm, best_cost
