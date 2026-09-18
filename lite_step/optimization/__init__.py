"""
Lite-STEP Optimization Package.

Provides generic, unopinionated optimization methods for the Lite-STEP DSL:
- @jit_compile: Numba JIT compiler decorator for fast numerical loops
- solve_sequence: Google OR-Tools CP-SAT permutation & sequence solver
- solve_milp: SciPy HiGHS Mixed-Integer Linear Programming solver
- solve_path: Vectorized shortest Hamiltonian path / dynamic programming solver
"""

from .combinatorial import (
    HAS_ORTOOLS,
    solve_sequence,
    solver_backends,
    solver_report,
)
from .jit import (
    HAS_NUMBA,
    jit_compile,
)
from .linear import (
    HAS_SCIPY,
    solve_milp,
    solve_path,
)

__all__ = [
    "HAS_NUMBA",
    "HAS_ORTOOLS",
    "HAS_SCIPY",
    "jit_compile",
    "solve_milp",
    "solve_path",
    "solve_sequence",
    "solver_backends",
    "solver_report",
]
