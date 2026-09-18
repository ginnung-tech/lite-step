"""ADVISORY facing audit — not a CI gate.

Run it deliberately::

    pytest -m facing_diagnostic
    LITESTEP_FACING_MODEL=path/to/output.py pytest -m facing_diagnostic -s

Every test here is marked ``facing_diagnostic`` and is EXCLUDED from the
default suite selection, because it is a cross-check on a heuristic rather
than a contract: it independently recomputes "which way is out" from the
nearest containment scope (Space -> Storey -> Building -> Site) and compares
it against the frame the compiler stamped.

A coding agent authoring a model can point ``LITESTEP_FACING_MODEL`` at it
and read the per-element table to sanity-check that exterior faces, layer
buildups and ``out=`` recesses will land on the side a human would call
"outside". Disagreements are reported with enough context to judge them —
walls between two rooms and free-standing elements are legitimately
ambiguous, which is exactly why this advises instead of failing the build.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest

from lite_step.compiler.frames import (
    _aabb_center,
    _enclosing_space_center,
    _spaces_in,
    element_aabb,
    stamp_frames,
)
from lite_step.models import Box, Point, Project, Space, Storey, Wall

pytestmark = pytest.mark.facing_diagnostic


def _independent_facing(elem, storey, building_center):
    """Recompute facing from scratch — deliberately NOT sharing the compiler's
    code path beyond the scope lookup, so a regression in ``_facing_for`` is
    visible rather than mirrored."""
    frame = elem._frame
    box = element_aabb(elem)
    if frame is None or box is None:
        return None
    spaces = _spaces_in(storey)
    center = _enclosing_space_center(box, spaces) or building_center
    if center is None:
        return None
    across = frame.across
    lo, hi = box
    c_lo = sum(lo[i] * across[i] for i in range(3))
    c_hi = sum(hi[i] * across[i] for i in range(3))
    lo_c, hi_c = min(c_lo, c_hi), max(c_lo, c_hi)
    ref = sum(center[i] * across[i] for i in range(3))
    return 1 if abs(hi_c - ref) >= abs(lo_c - ref) else -1


def _audit(project) -> list:
    rows = []
    boxes = [
        element_aabb(e)
        for st in project.storeys
        for e in st.elements
        if getattr(e, "placement", None) is None
    ]
    boxes = [b for b in boxes if b]
    building = None
    if boxes:
        building = _aabb_center((
            tuple(min(b[0][i] for b in boxes) for i in range(3)),
            tuple(max(b[1][i] for b in boxes) for i in range(3)),
        ))
    for storey in project.storeys:
        for elem in storey.elements:
            if getattr(elem, "_frame", None) is None:
                continue
            expected = _independent_facing(elem, storey, building)
            rows.append((
                getattr(elem, "ifc_name", None) or type(elem).__name__,
                elem._frame.rule,
                elem._frame.facing,
                expected,
            ))
    return rows


def _report(rows) -> str:
    lines = [f"{'element':<44} {'rule':<14} {'stamped':>7} {'expected':>8}"]
    for name, rule, got, exp in rows:
        flag = "" if exp is None or got == exp else "   <-- DISAGREES"
        lines.append(f"{str(name):<44} {rule:<14} {got:>+7d} {str(exp):>8}{flag}")
    return "\n".join(lines)


def test_facing_matches_an_independent_recomputation():
    """The stamped facing agrees with a from-scratch recomputation on a
    fixture with a room, an exterior ring and a free-standing element."""
    south = Wall(name="south").add(
        Box(start=Point(x=-6000, y=-4500, z=0), end=Point(x=6000, y=-4200, z=2700)))
    north = Wall(name="north").add(
        Box(start=Point(x=-6000, y=4200, z=0), end=Point(x=6000, y=4500, z=2700)))
    partition = Wall(name="partition").add(
        Box(start=Point(x=-2000, y=1000, z=0), end=Point(x=2000, y=1100, z=2700)))
    room = Space(name="room")
    room.add(Box(start=Point(x=-2500, y=1000, z=0), end=Point(x=2500, y=4200, z=2700)))
    storey = Storey()
    storey.add(south, north, partition, room)
    proj = Project(name="audit")
    proj.storeys = [storey]
    stamp_frames(proj)

    rows = _audit(proj)
    print("\n" + _report(rows))
    bad = [r for r in rows if r[3] is not None and r[2] != r[3]]
    assert not bad, f"facing disagreements:\n{_report(bad)}"


def test_audit_a_real_model_from_the_environment():
    """Point ``LITESTEP_FACING_MODEL`` at any model file to audit it.

    Skips when unset. Uses ``runpy`` with a non-``__main__`` run name so the
    model's ``compile_main()`` tail does not fire — we want the Project, not
    an IFC.
    """
    target = os.environ.get("LITESTEP_FACING_MODEL")
    if not target:
        pytest.skip("set LITESTEP_FACING_MODEL=<model.py> to audit a real model")
    path = Path(target)
    assert path.is_file(), f"no such model: {path}"

    ns = runpy.run_path(str(path), run_name="lite_step_facing_audit")
    project = ns.get("result")
    assert project is not None, f"{path} defines no module-level `result`"

    stamp_frames(project)
    rows = _audit(project)
    print(f"\n{path}\n" + _report(rows))
    bad = [r for r in rows if r[3] is not None and r[2] != r[3]]
    assert not bad, f"facing disagreements:\n{_report(bad)}"
