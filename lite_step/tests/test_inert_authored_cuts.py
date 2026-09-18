"""An AUTHORED subtraction that removes nothing must say so.

`.difference()` and `.void()` are instructions the author WROTE. Until this
existed, one that missed its target was indistinguishable from one that landed:

  * the compile exited 0,
  * the ``csg depth:`` line was identical (the boolean is counted whether or
    not it removes material),
  * an ANONYMOUS operand produced no report line at all, and
  * a NAMED operand produced a ``carve:`` line claiming the subtraction
    happened — which also satisfied ``assert_carved``, the DSL's own strongest
    guarantee about carves.

Measured before the fix, on a collar with one misplaced ``.difference()``: the
full compile output was byte-identical to the same model with the operand moved
onto the collar, and the host measured 0.026100 m³ either way — 3.3 litres of
difference between the two intents, zero difference in everything the author
can see.

That equality is what makes a correction loop unbounded. An author whose cut
did nothing has nothing to correct FROM, so the next attempt is a guess. The
observed cost was a production build that spent six compile-and-look passes on
one rafter void cut — 532.6 s and thirty agent turns to place one box.

The invariant here is the mirror of the narrow phase's
(``test_displacement_narrow_phase.py``): **an authored subtraction is called
inert only when the two solids are PROVABLY disjoint.** Anything unknown — a
curved operand, a mesh, a bent member — is left exactly as it was.
"""

from __future__ import annotations

import numpy as np
import pytest

from lite_step.compiler.displacement import (
    apply_displacement,
    carve_pairs,
    inert_authored_report_lines,
)
from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.models import Box, Material, Point, Project, Pipe, Sweep
from lite_step.models.project import Storey

TIMBER = "Timber_C24"
COLLAR = (45, 145)

#: The collar every test cuts at, in mm. Long enough that an operand can miss
#: it by a wide margin without leaving the model's own coordinate range.
COLLAR_A = Point(x=0, y=-2000, z=2400)
COLLAR_B = Point(x=0, y=2000, z=2400)


def _collar(name="collar"):
    return Sweep(name=name, path=[COLLAR_A, COLLAR_B],
                 material=Material(key=TIMBER, profile_mm=COLLAR))


def _compile(*elements):
    """Run the pass the way a real compile does — normalize to metres FIRST.

    `normalize_project_to_meters` RETURNS a deep copy and does not mutate.
    Dropping the return value leaves the project in millimetres while every
    extent is computed in metres, so nothing overlaps and every test in this
    file would report "inert" for the right answer as well as the wrong one.
    """
    proj = Project(name="inert authored cuts")
    storey = Storey(elevation=0)
    proj.add_storey(storey)
    storey.add(*elements)
    proj = normalize_project_to_meters(proj)
    apply_displacement(proj)
    return proj


def _inert(proj):
    return [(host.split(":")[1], operand.split(":")[0] if operand != "<anonymous>"
             else operand, mech)
            for host, operand, mech in (proj._carve_inert_authored or [])]


def _winners(proj):
    return {winner.split(":")[0] for _loser, winner, _mech in carve_pairs(proj)}


# ---------------------------------------------------------------------------
# The pair the whole file turns on: same operand, moved.
# ---------------------------------------------------------------------------


def test_a_cut_that_misses_is_reported_as_inert():
    collar = _collar()
    collar.difference(Box(name="trim",
                          start=Point(x=9000, y=9000, z=9000),
                          end=Point(x=9500, y=9500, z=9500)))
    proj = _compile(collar)
    assert _inert(proj) == [("collar", "box", "difference")]
    # And it is NOT recorded as a carve — the report must not claim it was.
    assert "box" not in _winners(proj)


def test_the_same_cut_moved_onto_the_host_is_a_real_carve():
    """The counter-test, and the one that must fail if the check over-fires.

    Identical operand, identical shape, identical mechanism — moved so that it
    intersects. Without this, a check that called EVERY authored cut inert
    would pass the test above and look correct.
    """
    collar = _collar()
    collar.difference(Box(name="trim",
                          start=Point(x=-500, y=1000, z=2300),
                          end=Point(x=500, y=1500, z=2500)))
    proj = _compile(collar)
    assert _inert(proj) == []
    assert "box" in _winners(proj)


# ---------------------------------------------------------------------------
# The case that had NO trace at all
# ---------------------------------------------------------------------------


def test_an_anonymous_operand_is_reported_too():
    """An unnamed operand never reached the carve report, so a missed cut left
    nothing behind — not even a wrong line. This is the silent case, and it is
    the shape the roof skill's own collar trim uses (`Sweep(profile=…, path=…)`
    with no name)."""
    collar = _collar()
    collar.difference(Box(start=Point(x=9000, y=9000, z=9000),
                          end=Point(x=9500, y=9500, z=9500)))
    proj = _compile(collar)
    assert _inert(proj) == [("collar", "<anonymous>", "difference")]


def test_the_warning_names_the_host_the_mechanism_and_the_operand(caplog):
    collar = _collar()
    collar.difference(Box(name="trim",
                          start=Point(x=9000, y=9000, z=9000),
                          end=Point(x=9500, y=9500, z=9500)))
    with caplog.at_level("WARNING", logger="lite_step.compiler.displacement"):
        _compile(collar)
    assert "collar" in caplog.text
    assert ".difference()" in caplog.text
    assert "trim" in caplog.text
    # It must not read as a tolerance question — the solids are PROVABLY apart,
    # and an author told "close but not touching" would nudge instead of fix.
    assert "provably disjoint" in caplog.text


# ---------------------------------------------------------------------------
# `assert_carved` must stop being satisfiable by a cut that does nothing
# ---------------------------------------------------------------------------


def test_assert_carved_is_no_longer_satisfied_by_an_inert_cut():
    """`assert_carved` is the strongest statement the DSL offers about a carve.
    It was satisfied by a subtraction that removed zero material, which made it
    a check on AUTHORING rather than on outcome — the opposite of its contract.
    """
    from lite_step.compiler.displacement import check_carve_assertions
    from lite_step.compiler.displacement import CarveAssertionError

    collar = _collar()
    collar.difference(Box(name="trim",
                          start=Point(x=9000, y=9000, z=9000),
                          end=Point(x=9500, y=9500, z=9500)))
    proj = Project(name="assert")
    storey = Storey(elevation=0)
    proj.add_storey(storey)
    storey.add(collar)
    # The canonical labels this model actually produces. Written out rather
    # than guessed: a first draft used `sweep:collar:storey:0`, which raised —
    # for a NAME mismatch, not for the inert cut — and so passed with the whole
    # check disabled. A test that cannot tell those two apart is not testing
    # the fix (cf. `feedback_changed_value_stale_test_must_fail`).
    proj.assert_carved("sweep:collar", by="box:trim:sweep:collar")
    proj = normalize_project_to_meters(proj)
    apply_displacement(proj)
    with pytest.raises(CarveAssertionError, match="removed nothing"):
        check_carve_assertions(proj)


# ---------------------------------------------------------------------------
# Conservatism: unknown shapes keep their carve
# ---------------------------------------------------------------------------


def test_an_operand_with_no_oriented_box_is_never_called_inert():
    """A `Pipe` is not a right-angled volume, so no exact box exists for it.
    The check has no answer, and no answer means "leave it alone" — the same
    rule the narrow phase follows. A check that guessed here would delete real
    joinery from every curved model in the corpus.
    """
    collar = _collar()
    collar.difference(Pipe(name="bore", radius=30,
                           path=[Point(x=9000, y=9000, z=9000),
                                 Point(x=9500, y=9500, z=9500)]))
    proj = _compile(collar)
    assert _inert(proj) == []
    assert "pipe" in _winners(proj)


# ---------------------------------------------------------------------------
# Frames: openings are host-frame and are deliberately NOT tested
# ---------------------------------------------------------------------------


def test_openings_are_left_out_of_the_check_on_purpose():
    """`.opening(along=…)` is expressed in the HOST's frame; the boxes this
    check compares are world-space. Folding openings in would compare two
    coordinate systems and report a miss for every correctly-placed opening.

    Pinned as a test rather than a comment so a later refactor that adds
    `_openings` to `_WORLD_FRAME_AUTHORED` has to answer the frame question
    instead of discovering it in a corpus sweep.
    """
    from lite_step.compiler.displacement import _WORLD_FRAME_AUTHORED
    assert [attr for attr, _ in _WORLD_FRAME_AUTHORED] == ["_cuts", "_voids"]


# ---------------------------------------------------------------------------
# The report line
# ---------------------------------------------------------------------------


def test_the_report_line_says_what_did_not_happen():
    collar = _collar()
    collar.difference(Box(name="trim",
                          start=Point(x=9000, y=9000, z=9000),
                          end=Point(x=9500, y=9500, z=9500)))
    proj = _compile(collar)
    lines = inert_authored_report_lines(proj)
    assert len(lines) == 1
    assert lines[0].startswith("inert-cut: ")
    assert "removed nothing" in lines[0]
    assert ".difference()" in lines[0]


def test_a_clean_model_prints_no_inert_line():
    """Silence on a correct model is the property that makes the line worth
    reading. Measured across all 27 reference examples: zero inert cuts."""
    collar = _collar()
    collar.difference(Box(start=Point(x=-500, y=1000, z=2300),
                          end=Point(x=500, y=1500, z=2500)))
    proj = _compile(collar)
    assert inert_authored_report_lines(proj) == []


# ---------------------------------------------------------------------------
# The geometry does not move
# ---------------------------------------------------------------------------


def test_reporting_an_inert_cut_changes_no_geometry(tmp_path):
    """The safety argument for landing this: it is a REPORT. The boolean is
    still emitted, so the file is what it was — the compiler simply stops
    describing a subtraction that did not occur as one that did.
    """
    import ifcopenshell

    def emit(name):
        collar = _collar()
        collar.difference(Box(name="trim",
                              start=Point(x=9000, y=9000, z=9000),
                              end=Point(x=9500, y=9500, z=9500)))
        proj = Project(name="geom")
        storey = Storey(elevation=0)
        proj.add_storey(storey)
        storey.add(collar)
        proj = normalize_project_to_meters(proj)
        apply_displacement(proj)
        from lite_step.ifc.generator import generate_ifc
        result = generate_ifc(proj)
        assert result.success, result.error
        path = tmp_path / name
        path.write_text(result.ifc_content, encoding="utf-8")
        return ifcopenshell.open(str(path))

    model = emit("a.ifc")
    # The dead boolean is still in the file — dropping it is a separate,
    # measurable change, and keeping it here is what makes this PR's geometry
    # byte-for-byte what it was.
    assert model.by_type("IfcBooleanResult"), (
        "the inert operand's boolean was dropped — that is a geometry change, "
        "and this PR is not one")


def test_the_host_volume_is_identical_with_and_without_the_inert_cut(tmp_path):
    """The claim the whole file rests on, measured rather than asserted."""
    import ifcopenshell
    import ifcopenshell.geom
    from lite_step.ifc.generator import generate_ifc

    def volume(with_cut: bool, name: str) -> float:
        collar = _collar()
        if with_cut:
            collar.difference(Box(start=Point(x=9000, y=9000, z=9000),
                                  end=Point(x=9500, y=9500, z=9500)))
        proj = Project(name="vol")
        storey = Storey(elevation=0)
        proj.add_storey(storey)
        storey.add(collar)
        proj = normalize_project_to_meters(proj)
        apply_displacement(proj)
        result = generate_ifc(proj)
        assert result.success, result.error
        path = tmp_path / name
        path.write_text(result.ifc_content, encoding="utf-8")
        model = ifcopenshell.open(str(path))
        settings = ifcopenshell.geom.settings()
        for product in model.by_type("IfcProduct"):
            if not (product.Name or "").startswith("sweep:collar"):
                continue
            shape = ifcopenshell.geom.create_shape(settings, product)
            v = np.array(shape.geometry.verts).reshape(-1, 3)
            f = np.array(shape.geometry.faces).reshape(-1, 3)
            a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
        raise AssertionError("collar not emitted")

    assert volume(True, "with.ifc") == pytest.approx(volume(False, "without.ifc"))
