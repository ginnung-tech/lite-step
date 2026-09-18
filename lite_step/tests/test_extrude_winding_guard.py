"""A vertical Extrude swept the wrong way by its winding order.

The defect is silent by construction. `apply_orientation_rules` canonicalises
a sloped or horizontal contour's normal but returns a VERTICAL one untouched —
correctly, because a vertical face has no "up" to canonicalise towards. So for
that one case the author's point order alone decides which side the solid
grows on, and getting it backwards sweeps the solid a full `thickness` away
from where it was meant to go. It compiles, validates and renders. It is
simply somewhere else.

Measured on a real failing build: two roof faces
authored at y=-2800 with thickness=7600 landed at y -10400..-2800, clear of
the house, where they wanted y -2800..4800.

The numbers below are that build, reduced to the smallest shape that
reproduces it.
"""
from __future__ import annotations

from lite_step.compiler.executor import validate_project_report
from lite_step.compiler.winding import check_extrude_winding
from lite_step.models import Box, Extrude, Point, Project

#: The walls the roof lands on: 8 m x 7 m, 3 m tall.
#:
#: They stop 800 mm SHORT of the roof contour at y=-2800, because a roof
#: overhangs its walls. That gap is not decoration — it is what makes the two
#: sweep directions distinguishable. A face swept away from a model it is
#: exactly flush with still abuts it, and this check cannot tell that from
#: resting on it; see the module docstring's note on what it does not catch.
WALLS = (-4000, -2000, 0, 4000, 4800, 3000)


def _box(name, b):
    return Box(name=name, start=Point(x=b[0], y=b[1], z=b[2]),
               end=Point(x=b[3], y=b[4], z=b[5]))


def _roof_contour(reversed_winding: bool):
    """A vertical roof face at y=-2800, spanning the house and rising to a ridge.

    Vertical means the Newell normal has no Z component, so the contour lies
    in a plane containing the Z axis — here the XZ plane at y=-2800. The
    thickness then sweeps it along Y, across the house or away from it.
    """
    pts = [
        Point(x=-4000, y=-2800, z=3000),
        Point(x=4000, y=-2800, z=3000),
        Point(x=0, y=-2800, z=5000),
    ]
    return list(reversed(pts)) if reversed_winding else pts


def _house(reversed_winding: bool, thickness: int = 7600) -> Project:
    proj = Project(name="cottage")
    proj.add(_box("walls", WALLS))
    proj.add(Extrude(name="roof", contour=_roof_contour(reversed_winding),
                     thickness=thickness, type="roof"))
    return proj


def _swept_y(proj):
    """The Y span the roof actually occupies — the whole point of the check."""
    from lite_step.compiler import winding as w
    for elem in w._iter_geometry(proj):
        ve = w._vertical_extrude(elem)
        if ve:
            box = w.swept_aabb(*ve)
            return (box[1], box[4])
    raise AssertionError("no vertical extrude in the fixture")


# ── The fixture is only worth anything if it reproduces the bug ────────────

def test_the_two_windings_sweep_the_roof_to_opposite_sides():
    """Sanity: reversing the contour is what MOVES the solid.

    Without this, every test below could pass on a fixture where the winding
    changes nothing, and the check would be verified against a defect it
    never saw.
    """
    away = _swept_y(_house(reversed_winding=False))
    onto = _swept_y(_house(reversed_winding=True))
    assert away != onto, "the winding does not move this solid; fixture is inert"

    def spans_the_walls(y):
        return y[0] < WALLS[4] and WALLS[1] < y[1]

    assert not spans_the_walls(away), (
        f"the bad winding was expected to sweep clear of the walls "
        f"(y {WALLS[1]}..{WALLS[4]}), but it lands at {away}")
    assert spans_the_walls(onto), (
        f"the good winding was expected to sweep across the walls "
        f"(y {WALLS[1]}..{WALLS[4]}), but it lands at {onto}")


# ── The check ──────────────────────────────────────────────────────────────

def test_a_roof_swept_clear_of_the_house_is_reported():
    warnings = check_extrude_winding(_house(reversed_winding=False))
    assert len(warnings) == 1, warnings
    text = warnings[0]
    assert "roof" in text
    assert "VERTICAL contour" in text
    # The numbers, so an author can check the claim rather than trust it.
    assert "-10,400" in text and "-2,800" in text, text


def test_the_correct_winding_says_nothing():
    assert check_extrude_winding(_house(reversed_winding=True)) == []


# ── THE REGRESSION THAT NEARLY SHIPPED ────────────────────────────────────
#
# The first version of this check asked "does the solid touch anything at
# all". On the real failing build BOTH roof faces were swept the wrong way,
# so each landed on the other, each "touched something", and the check
# reported the model clean — silent on the exact defect it was written for.
#
# A whole group misplaced together is the common case, not the corner case:
# the two faces of one roof are authored by the same loop, from the same
# mistake, in the same direction.

def test_a_pair_of_roofs_both_swept_wrong_cannot_vouch_for_each_other():
    proj = Project(name="cottage")
    proj.add(_box("walls", WALLS))
    for name, x in (("roof_west", -4000), ("roof_east", 0)):
        proj.add(Extrude(name=name, thickness=7600, type="roof", contour=[
            Point(x=x, y=-2800, z=3000),
            Point(x=x + 4000, y=-2800, z=3000),
            Point(x=x + 2000, y=-2800, z=5000),
        ]))

    warnings = check_extrude_winding(proj)
    assert len(warnings) == 2, (
        "both roofs were swept away from the house and landed on each other; "
        f"a check that counts a suspect as evidence sees this as fine: {warnings}")


# ── What must stay quiet ──────────────────────────────────────────────────

def test_a_genuinely_detached_element_is_not_reported():
    """A shed across the yard touches nothing EITHER way, so nothing is said.

    This is the whole reason the check is a comparison rather than a
    detached-solid detector: a warning that fires on every separate element
    trains people to ignore warnings.
    """
    proj = Project(name="cottage")
    proj.add(_box("walls", WALLS))
    proj.add(Extrude(name="shed", thickness=2000, type="wall", contour=[
        Point(x=40000, y=40000, z=0),
        Point(x=43000, y=40000, z=0),
        Point(x=43000, y=40000, z=2500),
    ]))
    assert check_extrude_winding(proj) == []


def test_a_sloped_contour_is_never_reported():
    """The compiler canonicalises a sloped normal, so its winding cannot be
    wrong in this way. Reporting one would be a warning nothing can act on."""
    proj = Project(name="cottage")
    proj.add(_box("walls", WALLS))
    # Rises in Z as it runs in Y -> the normal has a Z component.
    proj.add(Extrude(name="pitch", thickness=200, type="roof", contour=[
        Point(x=-4000, y=-20000, z=9000),
        Point(x=4000, y=-20000, z=9000),
        Point(x=0, y=-24000, z=12000),
    ]))
    assert check_extrude_winding(proj) == []


def test_a_project_with_nothing_trustworthy_to_measure_against_is_quiet():
    """Only vertical extrudes, so nothing in the project has a position we
    trust. Guessing here would be worse than staying silent."""
    proj = Project(name="sketch")
    proj.add(Extrude(name="a", contour=_roof_contour(False), thickness=7600))
    assert check_extrude_winding(proj) == []


# ── It is a WARNING ───────────────────────────────────────────────────────

def test_the_finding_is_a_warning_and_never_an_error():
    """House rule: degenerate-but-well-formed warns, and only a question the
    compiler cannot answer correctly raises. A detached solid is legal — it
    is just almost never intended — so the build must still complete."""
    report = validate_project_report(_house(reversed_winding=False))
    assert any("VERTICAL contour" in w for w in report.warnings), report.warnings
    assert not any("VERTICAL contour" in e for e in report.errors), report.errors


def test_the_check_reaches_the_report_at_all():
    """The wiring, separately from the check. A perfect detector nobody calls
    is the failure mode this test exists for."""
    clean = validate_project_report(_house(reversed_winding=True))
    dirty = validate_project_report(_house(reversed_winding=False))
    assert len(dirty.warnings) == len(clean.warnings) + 1


# ── A sweep of nothing ────────────────────────────────────────────────────
#
# The sibling defect, found while measuring the one above and fixed with it
# because they produce the SAME symptom — an element that is simply not there,
# with nothing anywhere saying why.

CONTOUR = [Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0), Point(x=1000, y=0, z=1000)]


def test_a_negative_thickness_builds_and_warns_rather_than_refusing():
    """It does NOT raise, and that is deliberate.

    An earlier version of this refused a negative thickness at construction,
    on the reasoning that the compiler "cannot answer correctly". It can: a
    thickness is a magnitude, the sign is discarded, and the result is a
    solid of zero depth — deterministic and well-formed. Degenerate-but-
    well-formed WARNS; only an unanswerable question raises.

    Refusing was also actively harmful once shipped: it turned a harmless
    useless solid into a lost build for any author already emitting one.
    """
    proj = Project(name="x")
    proj.add(Extrude(name="flat", thickness=-200, contour=CONTOUR))

    report = validate_project_report(proj)
    assert not report.errors, (
        f"a negative thickness raised: {report.errors} — it is well-formed "
        f"and must warn, not refuse")
    assert any("thickness=-200" in w for w in report.warnings), report.warnings


def test_the_warning_says_how_to_actually_sweep_the_other_way():
    """The likely intent is 'grow the other way', and the remedy is not the
    sign — it is the contour's winding order. Saying so is the whole value of
    the warning; without it the author retries with -400."""
    proj = Project(name="x")
    proj.add(Extrude(name="flat", thickness=-200, contour=CONTOUR))

    text = " ".join(validate_project_report(proj).warnings).lower()
    assert "reverse the contour" in text, text


def test_a_negative_thickness_really_does_produce_nothing():
    """The premise, MEASURED against the real pipeline rather than asserted.

    `authored_aabb` is the compiler; this module's own `swept_aabb` idealises
    the sweep and would say it grows the other way. The disagreement between
    the two is exactly why the warning is worth emitting.
    """
    e = Extrude(name="f", thickness=-200, contour=CONTOUR)
    box = e.authored_aabb()
    assert box.min.y == box.max.y == 0.0, (
        f"a negative thickness now sweeps somewhere (y {box.min.y}..{box.max.y}); "
        f"the warning claims zero depth and must be revisited")


def test_a_zero_thickness_warns_but_still_builds():
    """Well-formed and answerable — the compiler makes a zero-depth solid, so
    per the house rule this warns rather than raises. It still has to be
    SAID: nothing renders, and the model reads as missing a face."""
    proj = Project(name="x")
    proj.add(Extrude(name="flat", thickness=0, contour=CONTOUR))
    report = validate_project_report(proj)
    assert not report.errors, report.errors
    assert any("thickness=0" in w for w in report.warnings), report.warnings


def test_an_ordinary_thickness_says_nothing():
    proj = Project(name="x")
    proj.add(Extrude(name="ok", thickness=200, contour=CONTOUR))
    report = validate_project_report(proj)
    assert not any("thickness=0" in w for w in report.warnings), report.warnings
