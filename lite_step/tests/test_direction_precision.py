"""`IfcDirection` ratios must reach the file at full precision.

The deduplicator rounds floats to 4 decimals for `DEDUP_SAFE_TYPES`, a
tolerance chosen as "0.1 mm". `IfcDirection` was swept along with it — but a
direction ratio is **dimensionless**, so 4 decimals there is a bare 1e-4, and
105 of 178 emitted directions across the corpus came out non-unit.

The trap these tests exist to avoid: the value the issue reported,
`(0.7071,-0.7071,0.)`, is the ONE case where truncation is harmless. Equal
components cancel under normalisation, so it recovers true 45° exactly. Any
test written around a 45° miter passes against the unfixed code. Every
end-to-end case below is therefore deliberately **asymmetric**.
"""

from __future__ import annotations

import math
import re

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.deduplicator import deduplicate_ifc_step
from lite_step.models import Box, Point, Project
from lite_step.models.project import Storey

ifcopenshell = pytest.importorskip("ifcopenshell")

_DIRECTION_RE = re.compile(r"IFCDIRECTION\(\(([^)]*)\)\)")

_HEADER = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('','',(''),(''),'','','');
FILE_SCHEMA(('IFC4X3_ADD2'));
ENDSEC;
DATA;
"""
_FOOTER = "ENDSEC;\nEND-ISO-10303-21;\n"


def _step(*lines: str) -> str:
    return _HEADER + "\n".join(lines) + "\n" + _FOOTER


def _non_unit(content: str, tol: float = 1e-9):
    """Every emitted direction whose magnitude is not 1, with its error."""
    bad = []
    for m in _DIRECTION_RE.finditer(content):
        v = [float(x) for x in m.group(1).split(",")]
        err = abs(math.sqrt(sum(c * c for c in v)) - 1.0)
        if err > tol:
            bad.append((m.group(0), err))
    return bad


def _compile(proj: Project) -> str:
    from lite_step.ifc.generator import generate_ifc

    res = generate_ifc(normalize_project_to_meters(proj))
    assert res.success, res.error
    return res.ifc_content


def _clipped_box(normal) -> Project:
    proj = Project(name="dir-precision")
    storey = Storey(elevation=0)
    b = Box(name="b", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=2000, z=2000))
    b.clip(origin=Point(x=1000, y=1000, z=0), normal=normal)
    storey.add(b)
    proj.add_storey(storey)
    return proj


# ---------------------------------------------------------------------------
# mechanism — the deduplicator itself
# ---------------------------------------------------------------------------


def test_full_precision_direction_survives_dedup_verbatim():
    full = "0.7071067811865476"
    out, _ = deduplicate_ifc_step(_step(f"#1=IFCDIRECTION(({full},-{full},0.));"))
    assert full in out, out


def test_identical_directions_still_merge():
    """The exemption removes ROUNDING, not deduplication."""
    d = "#%d=IFCDIRECTION((0.5773502691896258,0.5773502691896258,0.5773502691896258));"
    out, stats = deduplicate_ifc_step(_step(d % 1, d % 2))
    assert stats["entities_removed"] == 1, stats
    assert out.count("IFCDIRECTION") == 1


def test_distinct_directions_are_not_merged_by_rounding():
    """Two directions differing by ~1e-4 are genuinely different directions —
    0.006 deg over a 10 m plane is 1 mm of rise. Rounding fuses them."""
    a = "#1=IFCDIRECTION((0.70710678,-0.70710678,0.));"
    b = "#2=IFCDIRECTION((0.70715678,-0.70705678,0.));"
    out, stats = deduplicate_ifc_step(_step(a, b))
    assert stats["entities_removed"] == 0, stats


# ---------------------------------------------------------------------------
# end to end — the artefact the bug actually lives in
# ---------------------------------------------------------------------------


def test_asymmetric_clip_emits_unit_directions_end_to_end():
    # 60 deg, NOT 45: at 45 the truncation self-cancels and this test would
    # pass against the unfixed code. That is the whole point of the angle.
    content = _compile(_clipped_box((0.5, -0.8660254037844386, 0.0)))
    assert _non_unit(content) == []


def test_miter_directions_are_unit():
    from lite_step.models.elements import miter

    proj = Project(name="miter-precision")
    storey = Storey(elevation=0)
    a = Box(name="a", start=Point(x=-3000, y=-300, z=0), end=Point(x=0, y=0, z=2400))
    b = Box(name="b", start=Point(x=-3000, y=-300, z=0), end=Point(x=-2700, y=2400, z=2400))
    miter(a, b, at=Point(x=-3000, y=-300, z=0), edge=(0, 0, 1))
    storey.add(a)
    storey.add(b)
    proj.add_storey(storey)
    assert _non_unit(_compile(proj)) == []


def test_the_45_degree_case_is_the_one_that_hides_the_bug():
    """Documents WHY the tests above are asymmetric.

    (a,-a,0) normalises to true 45 deg for any a, so the value the issue
    reported has exactly zero angular error. A regression suite built only on
    45 deg miters would be green against the unfixed compiler.
    """
    v = (0.7071, -0.7071, 0.0)
    n = math.sqrt(sum(c * c for c in v))
    assert abs(n - 1.0) == pytest.approx(9.59e-6, rel=1e-2)   # magnitude IS wrong
    # ...and the angle is EXACTLY right — bit-zero error. (a,-a,0) normalises
    # to (1/sqrt2, -1/sqrt2) for any a, so the truncation cancels completely.
    assert abs(math.atan2(-v[1] / n, v[0] / n) - math.pi / 4) == 0.0
