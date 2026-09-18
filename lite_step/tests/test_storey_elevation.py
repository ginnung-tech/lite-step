"""``IfcBuildingStorey.Elevation`` is WRITTEN — and says the authored number.. ``Storey(elevation=3000)`` placed geometry correctly and then emitted::

    #25=IFCBUILDINGSTOREY(…,'storey:first',$,$,#29,$,$,$,$);

on BOTH backends. The trailing ``LongName, CompositionType, Elevation`` were all
``$`` on a storey standing three metres up.

Why nothing caught it, and why this file has the shape it has
-------------------------------------------------------------

``Elevation`` is **OPTIONAL** in IFC4X3_ADD2. A null is schema-valid, so
``tests/test_ifc43_conformance.py`` passes at 0 errors with the data missing —
the standing note that "a passing validator is a floor, not a ceiling" in its
concrete form. The corpus semantic hash could not see it either: the value had
never been emitted once, so there was no drift to notice. And the model still
RENDERS, because geometry is authored in world coordinates. Three independent
gates, none of which can observe a dropped OPTIONAL. The only thing that can is
a test asserting we said something.

**Every fixture here stands at a NON-ZERO elevation.** shipped a
falsification break that caught nothing for exactly this reason: its fixtures
all sat at elevation 0, where the placement rebase subtracts zero and "written"
is indistinguishable from "dropped". ``STOREY_ELEVATIONS_MM`` is deliberately
``(0, 3000)`` — the ground storey is there so the zero case is pinned too, but
every assertion that could pass on a dropped value is made against the 3000.

The three claims, in order of what they would have caught:

1.  The value is THERE and it is the authored one, converted to the project
    length unit (metres) exactly once.
2.  It equals the storey placement's Z. Two numbers that must agree, derived
    once in ``lite_step.ifc.storeys`` — the rule, because "two
    implementations of one idea agreeing *usually*" is what produced this.
3.  ``LongName`` and ``CompositionType`` are ``$`` **on purpose** (see
    ``lite_step.ifc.storeys``), so that stays a decision rather than
    tomorrow's oversight.

"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.models import Box, Point, Project, Storey, Wall

pytest.importorskip("ifcopenshell")

import ifcopenshell            # noqa: E402
import ifcopenshell.validate   # noqa: E402


#: Ground at the datum, first storey three metres up. The 3000 is what makes
#: this suite able to fail — see the module docstring.
STOREY_ELEVATIONS_MM = (0, 3000)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def two_storey_project() -> Project:
    """Two named storeys, one wall each, the upper one at 3000 mm.

    The walls are offset in Y from each other so neither encloses the other —
    an overlapping pair would fire the displacement carve and change what is
    emitted for reasons that have nothing to do with elevation.
    """
    proj = Project(name="elev")
    for index, (leaf, elevation) in enumerate(zip(("ground", "first"),
                                                  STOREY_ELEVATIONS_MM)):
        storey = Storey(name=leaf, elevation=elevation)
        wall = Wall(name="north")
        wall.add(Box(start=Point(x=0, y=index * 1000, z=0),
                     end=Point(x=4000, y=index * 1000 + 300, z=2700)))
        storey.add(wall)
        proj.add_storey(storey)
    return proj


def single_storey_at(elevation_mm: int) -> Project:
    proj = Project(name="elev")
    storey = Storey(name="ground", elevation=elevation_mm)
    wall = Wall(name="north")
    wall.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=4000, y=300, z=2700)))
    storey.add(wall)
    proj.add_storey(storey)
    return proj


# ---------------------------------------------------------------------------
# Compile helpers — the backend is NAMED, never defaulted
# ---------------------------------------------------------------------------

def compile_ok(proj: Project, backend: str = "ifcopenshell",
               source_code: str = "src") -> str:
    normalized = normalize_project_to_meters(proj)
    from lite_step.ifc.generator import generate_ifc
    result = generate_ifc(normalized, source_code=source_code)
    assert result.success, result.error
    return result.ifc_content


def open_ifc(content: str):
    path = tempfile.mktemp(suffix=".ifc")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def storeys_by_name(content: str) -> dict:
    return {s.Name: s for s in open_ifc(content).by_type("IfcBuildingStorey")}


def storey_lines(content: str) -> list:
    return [line for line in content.splitlines()
            if re.match(r"^#\d+=IFCBUILDINGSTOREY\(", line)]


def placement_z(storey) -> float:
    """The Z of the storey's own ``RelativePlacement`` — what MOVES geometry."""
    placement = storey.ObjectPlacement
    assert placement is not None, "storey has no ObjectPlacement"
    return float(placement.RelativePlacement.Location.Coordinates[2])


def manifest_of(content: str) -> dict:
    for pset in open_ifc(content).by_type("IfcPropertySet"):
        if pset.Name == "LITESTEP_META":
            for prop in pset.HasProperties:
                if prop.Name == "MANIFEST":
                    return json.loads(prop.NominalValue.wrappedValue)
    raise AssertionError("no LITESTEP_META manifest in the file")


def schema_errors(content: str) -> list:
    logger = ifcopenshell.validate.json_logger()
    with contextlib.redirect_stderr(io.StringIO()):
        ifcopenshell.validate.validate(open_ifc(content), logger)
    return [s for s in logger.statements if s.get("level") == "error"]


# ---------------------------------------------------------------------------
# 0. The guard on every guard below
# ---------------------------------------------------------------------------


def test_the_fixture_is_not_at_elevation_zero():
    """The falsification break shipped, made impossible here.

    A suite whose every storey sits at 0 cannot tell "Elevation written" from
    "Elevation dropped" — 0.0 is what a missing value reads as. If someone
    flattens the fixture, this fails before the assertions it would silently
    defang.
    """
    assert any(e != 0 for e in STOREY_ELEVATIONS_MM), STOREY_ELEVATIONS_MM


# ---------------------------------------------------------------------------
# 1. The value is there, and it is the authored one
# ---------------------------------------------------------------------------


class TestElevationIsWritten:

    def test_elevation_is_not_null(self):
        """The literal bug: every storey said ``$``."""
        found = storeys_by_name(compile_ok(two_storey_project(), "ifcopenshell"))
        assert set(found) == {"storey:ground", "storey:first"}, sorted(found)
        for name, storey in found.items():
            assert storey.Elevation is not None, (
                f"{name} emitted Elevation=$ — the authored elevation was "
                f"dropped")

    def test_elevation_equals_the_authored_value_in_metres(self):
        """3000 mm in, 3.0 out — the project length unit is METRE.

        Converted exactly ONCE, by ``normalize_project_to_meters``. A second
        division would land 0.003 here and still look like a written value.
        """
        found = storeys_by_name(compile_ok(two_storey_project(), "ifcopenshell"))
        assert found["storey:ground"].Elevation == pytest.approx(0.0)
        assert found["storey:first"].Elevation == pytest.approx(3.0)

    @pytest.mark.parametrize("elevation_mm", [0, 250, 3000, -1500])
    def test_any_authored_elevation_survives(self, elevation_mm):
        """Including a BASEMENT. A sign flip is invisible at 0 and at the
        defaults, and a storey below datum is the ordinary case that finds it.
        """
        content = compile_ok(single_storey_at(elevation_mm), "ifcopenshell")
        storey = storeys_by_name(content)["storey:ground"]
        assert storey.Elevation == pytest.approx(elevation_mm / 1000.0)

    def test_the_datum_storey_states_zero_rather_than_null(self):
        """``$`` means "not stated"; ``0.`` means "at the datum".

        The DSL always carries a concrete number, so there is no storey whose
        elevation we do not know — emitting ``$`` for the datum would make "at
        zero" indistinguishable from "unknown", which is the class of failure
        this whole change exists to remove. This is also why the entire corpus
        drifts on a change no committed model has a non-zero storey for.
        """
        content = compile_ok(single_storey_at(0), "ifcopenshell")
        lines = storey_lines(content)
        assert len(lines) == 1, lines
        assert lines[0].rstrip(";").endswith(",0.)"), lines[0]

class TestElevationMatchesThePlacement:

    def test_stated_elevation_equals_placement_z(self):
        """The attribute and the transform must not be two derivations.

        A file whose ``Elevation`` says 3.0 while its storey stands at 3.5 is
        schema-valid and internally contradictory — a consumer that reads the
        attribute and one that resolves the placement chain would disagree
        about the same building.
        """
        found = storeys_by_name(compile_ok(two_storey_project(), "ifcopenshell"))
        for name, storey in found.items():
            assert storey.Elevation == pytest.approx(placement_z(storey)), (
                f"{name}: Elevation={storey.Elevation} but the placement "
                f"stands at z={placement_z(storey)}")

    def test_the_upper_storey_actually_stands_at_three_metres(self):
        """Pins the pairing above to a NUMBER, not just to itself.

        ``Elevation == placement_z`` also holds when both are 0 because the
        elevation was dropped and the storey never moved.
        """
        found = storeys_by_name(compile_ok(two_storey_project(), "ifcopenshell"))
        assert placement_z(found["storey:first"]) == pytest.approx(3.0)
        assert found["storey:first"].Elevation == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 3. LongName and CompositionType are $ ON PURPOSE
# ---------------------------------------------------------------------------


class TestTheAttributesWeDeliberatelyLeaveNull:

    def test_long_name_is_null(self):
        """The DSL has ONE authored string per storey and ``Name`` already
        carries it as the stamped canonical. A ``LongName`` echoing it would
        be a second spelling of the same fact, free to drift; inventing
        "Level 1" from the index would be data no author wrote. See
        ``lite_step.ifc.storeys``. A decision, not an oversight — which is
        what this test makes it.
        """
        found = storeys_by_name(compile_ok(two_storey_project(), "ifcopenshell"))
        assert [s.LongName for s in found.values()] == [None, None], found

    def test_composition_type_is_null(self):
        """DEPRECATED in IFC4 — emitting it would re-introduce a retired
        attribute."""
        found = storeys_by_name(compile_ok(two_storey_project(), "ifcopenshell"))
        assert [s.CompositionType for s in found.values()] == [None, None], found


# ---------------------------------------------------------------------------
# 4. What the added attribute must NOT touch
# ---------------------------------------------------------------------------


class TestTheAttributeChangesNothingElse:

    def test_no_schema_errors(self):
        """The corpus gate's budget is 0 errors; a multi-storey model with a
        stated elevation must hold it too."""
        errors = schema_errors(compile_ok(two_storey_project(), "ifcopenshell"))
        assert not errors, [str(s.get("message"))[:120] for s in errors]

    def test_the_manifest_keys_are_exactly_the_authored_names(self):
        """A named storey IS a manifest key, and adding ``Elevation`` adds none.

        Measured, not reasoned, because the reasoning went the other way
        first. ``IfcBuildingStorey`` is an ``IfcProduct``, and the v21.2
        ``IfcAnnotation`` trap says an ``IfcProduct`` that matches no DSL
        element pollutes patch identity — so the obvious assertion was "no
        storey key". It is wrong: a ``Storey(name="ground")`` **is** a DSL
        object with a stamped canonical (``storey:ground``), so it is keyed
        deliberately. What matters is that the key SET is exactly the
        authored objects and nothing new appeared — an attribute on an
        existing entity mints no patch atom.

        Pinning the whole set also records ``"Site"``, which this project does
        NOT author: it is the literal default ``IfcSite`` Name
        (``proj.sites[0]._canonical_name if proj.sites else "Site"``, both
        backends), so a model with no ``Site`` container still contributes a
        manifest key matching no DSL element. Also pre-existing, also
        unchanged, and harmless today only because DSL v2.1 names
        are lowercase leaves and cannot collide with it.
        """
        manifest = manifest_of(compile_ok(two_storey_project(), "ifcopenshell"))
        assert sorted(manifest) == [
            "Site",                        # the default IfcSite Name — see below
            "elev",                        # the Project name
            "storey:first", "storey:ground",
            "wall:north:storey:first", "wall:north:storey:ground",
        ], sorted(manifest)

    def test_patch_identity_is_untouched_by_the_elevation(self):
        """Moving a storey changes no canonical name and no manifest key.

        ``Elevation`` is an attribute on an EXISTING entity, not a new named
        product, so it cannot mint a patch atom. The two manifests are
        compared key-for-key at two different elevations; only the GlobalIds
        (fresh per compile) may differ.
        """
        flat = manifest_of(compile_ok(single_storey_at(0), "ifcopenshell"))
        raised = manifest_of(compile_ok(single_storey_at(3000), "ifcopenshell"))
        assert sorted(flat) == sorted(raised), (sorted(flat), sorted(raised))

    def test_the_storey_entity_still_has_ten_attributes(self):
        """``IfcBuildingStorey`` takes exactly 10. The streaming writer passed
        NINE until then and the code comment said 10 the whole time — an
        arity drift is how ``Elevation`` went missing in the first place."""
        for line in storey_lines(compile_ok(two_storey_project(), "ifcopenshell")):
            body = line.split("(", 1)[1].rstrip(";").rstrip(")")
            assert body.count(",") == 9, line
