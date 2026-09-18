"""
Numba JIT Acceleration Utilities for Lite-STEP.

Provides machine-code compilation for intensive geometric calculations,
parametric surface evaluations, raycasting, and numerical routines.
"""

from collections.abc import Callable
import functools
from typing import Any

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    njit = None


def jit_compile(
    func: Callable | None = None,
    *,
    fastmath: bool = True,
    parallel: bool = False,
    nogil: bool = True,
) -> Callable:
    """Decorator to JIT-compile a numerical function with Numba if available.

    Falls back cleanly to the pure Python function if Numba is not installed.
    """

    def decorator(fn: Callable) -> Callable:
        if HAS_NUMBA and njit is not None:
            compiled = njit(fastmath=fastmath, parallel=parallel, nogil=nogil)(fn)
            return compiled

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
