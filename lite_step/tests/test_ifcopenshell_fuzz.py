"""Tests running IfcOpenShell hold-water fuzz validation via check_mesh_signed_volume.

Tests combinations of geometry (Box, Extrude, Bar, Pipe, Mesh) and operations
(CSG Difference, CSG Union, CSG Intersection, IfcRelVoidsElement) to verify:
1. Topological watertightness (0 boundary open edges).
2. Positive signed volume via Gauss Divergence Theorem.
3. Accurate volume conservation bounds.
4. Correctly isolates and documents known upstream IfcOpenShell issues (e.g.).
"""

import pytest

from lite_step.tests.tools.fuzz_ifcopenshell import (
    build_suite_coplanar,
    build_suite_csg,
    build_suite_mesh,
    build_suite_multi_cut,
    build_suite_swept_disk,
    run_fuzz_case,
)


@pytest.mark.parametrize("case", build_suite_swept_disk(), ids=lambda c: c.name)
def test_fuzz_swept_disk(case):
    res = run_fuzz_case(case)
    if case.is_known_bug:
        # Known upstream issue: void on swept disk solid trims directrix
        assert res.status in ("DISCARDED_MATERIAL", "EMPTY_OR_CRASH")
    else:
        assert res.status == "PASS", f"Case {case.name} failed: {res.error or [p.error_message for p in res.products]}"


@pytest.mark.parametrize("case", build_suite_coplanar(), ids=lambda c: c.name)
def test_fuzz_coplanar(case):
    res = run_fuzz_case(case)
    assert res.status == "PASS", f"Case {case.name} failed: {res.error or [p.error_message for p in res.products]}"


@pytest.mark.parametrize("case", build_suite_csg(), ids=lambda c: c.name)
def test_fuzz_csg(case):
    res = run_fuzz_case(case)
    assert res.status == "PASS", f"Case {case.name} failed: {res.error or [p.error_message for p in res.products]}"


@pytest.mark.parametrize("case", build_suite_mesh(), ids=lambda c: c.name)
def test_fuzz_mesh(case):
    res = run_fuzz_case(case)
    if case.name == "mesh_inverted_cube_detection":
        assert res.status == "INVERTED", f"Expected INVERTED detection, got {res.status}"
    else:
        assert res.status == "PASS", f"Case {case.name} failed: {res.error or [p.error_message for p in res.products]}"


@pytest.mark.parametrize("case", build_suite_multi_cut(), ids=lambda c: c.name)
def test_fuzz_multi_cut(case):
    res = run_fuzz_case(case)
    assert res.status == "PASS", f"Case {case.name} failed: {res.error or [p.error_message for p in res.products]}"
