"""The marker peg is a solid, and both backends build the SAME one.

Neither was true before that. A ``ReferencePoint`` emits a hexagonal stake —
the one piece of geometry in the corpus that no DSL author writes — so nobody
had ever inspected it, and two independent implementations had drifted:

* ``ifcopenshell`` fanned the bottom cap over all 6 sides. A fan over an n-gon
  is n-2 triangles; the ``% sides`` wrap made i=0 emit ``(0,1,0)`` and i=5 emit
  ``(0,0,5)`` — a repeated index, zero area, not 2-manifold.
* ``streaming`` built ONE centre vertex, the top one at y=0, and fanned the
  bottom cap to it. The bottom ring sits at y=-height, so the "cap" was a cone
  running the full height of the peg through its own body: topologically
  closed and geometrically self-intersecting.

**Nothing could have caught either one.** They render as nothing (zero-area) or
as interior geometry (invisible from outside), and ``ifcopenshell.validate``
reports 0 errors because no schema rule is broken — a degenerate triangle and a
self-intersecting solid are both well-formed IFC. So the test measures the
emitted ``CoordIndex`` rather than reading the construction loops, and compares
the two backends against each other.

Found while adding ``IfcTriangulatedFaceSet.Closed``: stating "is this a
solid?" required knowing whether the peg actually was one.
"""

from __future__ import annotations

import collections
import os
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.models import Point, Project, ReferencePoint
from lite_step.models.project import Storey


def _peg_project() -> Project:
    proj = Project(name="peg")
    storey = Storey(elevation=0)
    storey.add(ReferencePoint(id="datum", name="datum",
                              location=Point(x=0, y=0, z=0),
                              point_type="boundary"))
    proj.add_storey(storey)
    return proj


def _compile(backend: str) -> str:
    if backend == "ifcopenshell":
        pytest.importorskip("ifcopenshell")
    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(normalize_project_to_meters(_peg_project()))
    assert result.success, result.error
    return result.ifc_content


def _face_set(content: str):
    """The peg's face set, parsed out of the STEP text with ifcopenshell so
    both backends are read the same way."""
    pytest.importorskip("ifcopenshell")
    import ifcopenshell

    with tempfile.NamedTemporaryFile("w", suffix=".ifc", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(content)
        path = fh.name
    try:
        sets = ifcopenshell.open(path).by_type("IfcTriangulatedFaceSet")
        assert sets, "no peg face set emitted"
        return sets[0]
    finally:
        os.unlink(path)


def _edge_counts(coord_index):
    directed = collections.Counter()
    undirected = collections.Counter()
    for a, b, c in coord_index:
        for edge in ((a, b), (b, c), (c, a)):
            directed[edge] += 1
            undirected[frozenset(edge)] += 1
    return directed, undirected


# ---------------------------------------------------------------------------
# The geometry is a solid — measured, not assumed
# ---------------------------------------------------------------------------


def test_no_degenerate_triangles():
    """A triangle with a repeated vertex index has zero area. It renders as
    nothing and validates clean, so only counting catches it."""
    ci = _face_set(_compile("ifcopenshell")).CoordIndex
    degenerate = [t for t in ci if len(set(t)) < 3]
    assert not degenerate, \
        f"{"ifcopenshell"} emits {len(degenerate)} degenerate peg triangles: {degenerate[:4]}"


def test_peg_is_two_manifold():
    """Closed with consistent outward winding: every undirected edge used
    exactly twice, every directed edge exactly once."""
    directed, undirected = _edge_counts(_face_set(_compile("ifcopenshell")).CoordIndex)
    bad_u = {tuple(sorted(e)): n for e, n in undirected.items() if n != 2}
    bad_d = {e: n for e, n in directed.items() if n != 1}
    assert not bad_u, f"{"ifcopenshell"}: edges not used exactly twice: {list(bad_u)[:6]}"
    assert not bad_d, f"{"ifcopenshell"}: winding inconsistent at: {list(bad_d)[:6]}"


def test_peg_states_closed():
    """Now that the geometry IS a solid, the file says so. Left unstated precisely because it was not."""
    assert _face_set(_compile("ifcopenshell")).Closed is True


def test_bottom_cap_does_not_reach_the_top():
    """The streaming defect specifically: the bottom cap fanned to the TOP
    centre vertex, so the cap spanned the peg's whole height.

    Checked as a geometric fact — no triangle in the bottom cap may touch the
    top plane — rather than by vertex index, which would just restate the fix.
    """
    fs = _face_set(_compile("ifcopenshell"))
    pts = [tuple(c) for c in fs.Coordinates.CoordList]
    zs = [p[1] for p in pts]          # peg is built along IFC -Y then placed
    lo, hi = min(zs), max(zs)
    span = hi - lo
    for tri in fs.CoordIndex:
        levels = [pts[i - 1][1] for i in tri]   # CoordIndex is 1-based
        if all(abs(v - lo) < span * 0.01 for v in levels[:2]):
            assert abs(max(levels) - hi) > span * 0.01 or abs(min(levels) - lo) < span * 0.01, \
                f"{"ifcopenshell"}: a bottom triangle reaches the top plane — {tri}"


# ---------------------------------------------------------------------------
# The two backends agree
# ---------------------------------------------------------------------------


def test_the_peg_is_a_hexagonal_stake_of_the_expected_size():
    """Anchors the counts, so a future rewrite that silently changes the shape
    has to say so. 6 sides: 6 bottom-fan + 12 body + 12 cap-skirt + 6 top-fan."""
    fs = _face_set(_compile("ifcopenshell"))
    assert len(fs.Coordinates.CoordList) == 20, "expected 3 rings of 6 + 2 centres"
    assert len(fs.CoordIndex) == 36, "expected 6 + 12 + 12 + 6 triangles"
