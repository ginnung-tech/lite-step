"""Strict int-millimeter enforcement switch (DSL v1.5, WS1 PR-C).

The v1.5 Lite-STEP contract (infrastructure/specs/lite-step-agent-instructions.md
RULE 2) says all geometry values are **int millimeters** — floats are rejected
at construction and every division is wrapped in ``I()``.

Mechanism: a module-level flag read by ``mode="before"`` validators on
:class:`~lite_step.models.Point` / ``Point2D``. Since the v10 hard cutover it
is unconditionally ON — the ``LITESTEP_STRICT_INT_MM`` env switch is GONE — and
there are two ways to move it, both internal:

* :func:`enable_strict_int_mm` / :func:`disable_strict_int_mm` (test fixtures),
* :func:`suspend_strict` — an internal context manager the meters-normalizer
  wraps itself in, because ``normalize_project_to_meters`` legitimately
  rebuilds Points as float **meters** post-validation. Strictness is a gate on
  USER input at construction time, not an internal pipeline invariant.

A plain module global (no threading.local): the worker executes one task per
process and the test-suite toggles are fixture-scoped. Documented trade-off —
if the compiler ever goes multi-threaded, revisit.
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Iterator, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

# ALWAYS ON. The ``LITESTEP_STRICT_INT_MM=0`` escape hatch is removed in the
# v10 hard cutover: an env var that silently relaxes an authoring contract is
# a way for a deploy to disagree with the spec, and the failure it produces —
# float millimetres accepted here, rounded somewhere downstream — surfaces as
# geometry that is subtly off rather than as a refusal. The programmatic
# toggles below stay: they are test fixtures, not a deploy switch, and
# ``suspend_strict`` is the sanctioned internal path.
_strict: bool = True


def strict_int_mm_enabled() -> bool:
    """True when float geometry values are rejected at construction."""
    return _strict


def enable_strict_int_mm() -> None:
    """Turn strict int-mm validation ON (v1.5 contract behavior)."""
    global _strict
    _strict = True


def disable_strict_int_mm() -> None:
    """Turn strict int-mm validation OFF (pre-cutover default)."""
    global _strict
    _strict = False


@contextmanager
def suspend_strict() -> Iterator[None]:
    """Temporarily disable strictness for an internal construction path.

    Used by ``normalize_project_to_meters`` (float-meter Point rebuilds).
    Restores the prior state even on exceptions; nesting is safe.
    """
    global _strict
    prior = _strict
    _strict = False
    try:
        yield
    finally:
        _strict = prior


def suspends_strict(fn: _F) -> _F:
    """Decorator form of :func:`suspend_strict` — the wrapped function's whole
    dynamic extent (including helpers it calls) runs with strictness off."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with suspend_strict():
            return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
