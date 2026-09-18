"""One model compiled N times must emit ONE file.

This suite exists because nothing asserted it, and the gap cost real time.

`ifcopenshell.api.unit.assign_unit` finishes with
``unit_assignment.Units = list(units)`` over a Python **set** of
``entity_instance`` (assign_unit.py:113-122), and ``entity_instance.__hash__``
is ``hash((step_id, wrapped_data.file_pointer()))`` — the C++ ``IfcFile``'s heap
ADDRESS. The step ids are fixed (``#2``/``#3``/``#4`` = LENGTH/AREA/VOLUME), so
the address is the only varying term, and every ``ifcopenshell.file()`` lands
somewhere new. Measured before the fix: **12 compiles in one process produced 3
distinct orderings**, unchanged at ``PYTHONHASHSEED=0`` and unchanged across
subprocesses — address-derived, so no seed pins it.

``Units`` is an IFC SET: every ordering is equally VALID. What it is not is
REPRODUCIBLE, and the corpus reference IFCs are committed build
products that people diff.

**Why nothing caught it for so long.** Two independent maskers:

1.  The corpus hash normalises by sorting set members before hashing,
    so the corpus gate reported 20/20 byte-identical throughout. That is correct
    behaviour for a gate that must read pre-fix files and third-party IFC — but
    it means the gate cannot be the thing that notices.
2.  Everything routes through ifcopenshell, so the nondeterminism is
    reachable from a plain ``Wall`` + ``Box``.

So the surviving symptom was one intermittently-red test
(``test_dsl_v21_naming::test_default_name_is_none_identical_ifc``) that compares
two compiles — which reads as flakiness rather than as a defect in emission.
These tests name it directly instead.
"""

from __future__ import annotations

import re

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc
from lite_step.models import Bar, Box, Point, Point2D, Project, Revolve, Wall

pytest.importorskip("ifcopenshell")

#: Enough repeats that a surviving shuffle is overwhelmingly likely to show.
#: The pre-fix baseline produced 2-3 distinct orderings in every batch of >=12,
#: so 30 is not a marginal call.
_N = 30

#: The two things that vary by DESIGN on every compile. Everything else must be
#: byte-stable, and that is the whole claim of this file.
_GUID = re.compile(r"'[0-9A-Za-z_$]{22}'")
_STAMP = re.compile(r"(FILE_NAME\('',)'[^']*'")


def _normalised(text: str) -> str:
    return _STAMP.sub(r"\1'<ts>'", _GUID.sub("'GUID'", text))


def _wall_project() -> Project:
    proj = Project(name="determinism")
    wall = Wall(name="north")
    wall.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=5000, y=3000, z=200),
                 type="wall"))
    proj.add(wall)
    return normalize_project_to_meters(proj)


def _curved_project() -> Project:
    """A model whose units go BEYOND the three SI defaults.

    ``_ensure_si_unit`` appends RADIAN and MASSUNIT lazily when a Revolve/Bar
    needs them, at ids ABOVE the first three. The canonical order must keep
    those at the tail rather than interleaving them, so this fixture is what
    stops the fix from being right only for the trivial case.
    """
    proj = Project(name="determinism-curved")
    wall = Wall(name="north")
    wall.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=4000, y=2000, z=200),
                 type="wall"))
    wall.add(Revolve(profile=[Point2D(x=0, y=0), Point2D(x=100, y=0),
                              Point2D(x=100, y=100), Point2D(x=0, y=100)],
                     path=[Point(x=1000, y=1000, z=0),
                           Point(x=1000, y=1000, z=1000)],
                     angle=90, name="turn"))
    proj.add(wall)
    return normalize_project_to_meters(proj)


def _units_line(text: str) -> str:
    return next(l for l in text.splitlines() if "IFCUNITASSIGNMENT" in l)


def test_one_model_compiled_n_times_emits_one_file():
    """THE assertion. GUIDs and the header timestamp normalised, nothing else.

    Reverting ``_canonical_unit_order`` makes this fail probabilistically —
    2-3 distinct texts per batch on the measured baseline — which is exactly
    why the symptom presented as a flaky unrelated test rather than as a bug.
    """
    texts = {_normalised(generate_ifc(_wall_project(), source_code=None).ifc_content)
             for _ in range(_N)}
    assert len(texts) == 1, (
        f"{len(texts)} distinct files from {_N} compiles of ONE model — "
        "emission is not reproducible")


def test_the_unit_assignment_is_in_creation_order():
    """Not just STABLE — stable at the RIGHT value.

    A Python set that happened to land ascending on every run of one batch
    would satisfy the test above and still be a set. Pinning the literal order
    means a re-introduced shuffle cannot hide behind a lucky hash layout.
    """
    for _ in range(_N):
        line = _units_line(generate_ifc(_wall_project(), source_code=None).ifc_content)
        assert line.endswith("=IFCUNITASSIGNMENT((#2,#3,#4));"), line


def test_lazily_added_units_stay_at_the_tail():
    """RADIAN/MASSUNIT are appended later, so they carry HIGHER ids.

    Sorting on ``id()`` therefore keeps them after the three SI defaults with no
    second rule — asserted rather than assumed, because a fix that sorted by
    anything else (name, type) would reorder these and churn every curved model.
    """
    seen = set()
    for _ in range(_N // 3):
        line = _units_line(generate_ifc(_curved_project(), source_code=None).ifc_content)
        ids = [int(x) for x in re.findall(r"#(\d+)", line.split("((", 1)[1])]
        assert ids == sorted(ids), f"unit ids not ascending: {ids}"
        assert ids[:3] == [2, 3, 4], f"the SI defaults moved: {ids}"
        seen.add(tuple(ids))
    assert len(seen) == 1, f"curved-model unit order is not stable: {seen}"
