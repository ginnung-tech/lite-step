"""``color=`` means the same thing everywhere, on both backends.

Without it, the field has two disjoint halves and neither miss says anything:

* a palette name styled a SOLID and was ignored on a ``Mesh``;
* a hex string styled a ``Mesh`` and was ignored on a SOLID.

``Box(color="#FF8800")`` emitted ``IfcColourRgb=[]`` and rendered
viewer-default, so the element looked deliberately grey. That was a
silent failure documented in the DSL reference, and it was two bugs sharing a field
name. This suite pins the union, and pins that an unreadable value is now a
compile ERROR rather than a silent default.
"""
from __future__ import annotations

import re

import pytest

from lite_step.ifc.colors import (
    ELEMENT_TYPE_COLORS,
    resolve_color,
)
from lite_step.models import (
    Box, Element, Mesh, Point, Point2D, Project, Site, Storey, Wall,
)
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.ifc.generator import generate_ifc


# --------------------------------------------------------------------------
# The vocabulary itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value,rgb,transparency", [
    ("#FF8800", (255, 136, 0), 0.0),
    ("#ff8800", (255, 136, 0), 0.0),        # case-insensitive
    ("#f80", (255, 136, 0), 0.0),           # 3-digit shorthand expands
    ("#FF8800FF", (255, 136, 0), 0.0),      # explicit full opacity
    ("#FF880000", (255, 136, 0), 1.0),      # fully transparent
    ("  #FF8800  ", (255, 136, 0), 0.0),    # surrounding space tolerated
])
def test_hex_forms(value, rgb, transparency):
    got_rgb, got_t = resolve_color(value)
    assert got_rgb == rgb
    assert got_t == pytest.approx(transparency, abs=1e-6)


def test_alpha_is_opacity_not_transparency():
    """The byte is OPACITY; IFC stores the complement. Getting this backwards
    makes a half-transparent pane and a 60%-opaque one mean the same thing."""
    _rgb, half = resolve_color("#33669980")
    assert half == pytest.approx(1.0 - 0x80 / 255.0, abs=1e-6)
    assert 0.4 < half < 0.6


def test_palette_names_resolve_and_glass_is_transparent():
    assert resolve_color("wall")[0] == ELEMENT_TYPE_COLORS["wall"]
    assert resolve_color("glass")[1] > 0.0, "glass carries its own transparency"


@pytest.mark.parametrize("value", [
    None, "", "   ", "orangeish", "#GG0000", "#FF88", "#FF88000", "FF8800", 42,
])
def test_unreadable_values_resolve_to_none(value):
    assert resolve_color(value) is None


# --------------------------------------------------------------------------
# End to end — the halves that would otherwise be silent
# --------------------------------------------------------------------------

def _model(solid_color=None, mesh_color=None):
    proj = Project(name="t")
    storey = Storey(name="ground", elevation=0)
    proj.add_storey(storey)
    wall = Wall(name="south")
    wall.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=3000, y=200, z=2400), color=solid_color))
    storey.add(wall)
    site = Site(name="site")
    terrain = Element(ifc_class="IfcGeographicElement",
                      predefined_type="TERRAIN", name="terrain")
    terrain.add(Mesh(heightmap=[[0, 0], [0, 0]], depth=500,
                     corner_min=Point2D(x=-5000, y=-5000),
                     corner_max=Point2D(x=5000, y=5000), color=mesh_color))
    site.add(terrain)
    proj.add(site)
    return proj


def _colours(ifc_text):
    """Every emitted IfcColourRgb as an (r, g, b) triple in 0-1."""
    out = []
    for m in re.finditer(r"IFCCOLOURRGB\(\$,([0-9.eE+-]+),([0-9.eE+-]+),([0-9.eE+-]+)\)",
                         ifc_text):
        out.append(tuple(round(float(g), 4) for g in m.groups()))
    return out


def _compile(proj):
    report = validate_project_report(proj)
    assert not report.errors, report.errors
    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    return result.ifc_content


def test_hex_on_a_solid_reaches_the_file():
    """The headline: this emitted NO style at all before."""
    text = _compile(_model(solid_color="#FF8800"))
    assert (1.0, 0.5333, 0.0) in _colours(text), (
        f"the solid's hex colour is missing: {_colours(text)}")


def test_a_palette_name_on_a_mesh_reaches_the_file():
    """The other half — a named colour on a Mesh was ignored."""
    text = _compile(_model(mesh_color="glass"))
    r, g, b = ELEMENT_TYPE_COLORS["glass"]
    expected = (round(r / 255, 4), round(g / 255, 4), round(b / 255, 4))
    assert expected in _colours(text), (
        f"the mesh's palette colour is missing: {_colours(text)}")


def test_hex_on_a_solid_and_a_mesh_are_the_same_colour():
    """One field, one meaning, whatever it is attached to.

    The colour entity appears ONCE — ``EntityCache`` shares an
    ``IfcColourRgb``/``IfcSurfaceStyle`` between everything using it, which is
    what it is for — so the evidence that both carriers were styled is TWO
    ``IfcStyledItem``s pointing at that one colour. Asserting two colour
    entities instead would be asserting the cache is broken.
    """
    text = _compile(_model(solid_color="#12AB34", mesh_color="#12AB34"))
    wanted = (round(0x12 / 255, 4), round(0xAB / 255, 4), round(0x34 / 255, 4))
    assert wanted in _colours(text), (
        f"the shared colour is missing: {_colours(text)}")
    assert _colours(text).count(wanted) == 1, (
        "the colour entity should be SHARED, not duplicated per carrier")
    styled = re.findall(r"IFCSTYLEDITEM\(", text)
    assert len(styled) >= 2, (
        f"both the solid and the mesh must carry a styled item: {len(styled)}")


def test_hex_alpha_on_a_solid_emits_transparency():
    text = _compile(_model(solid_color="#3366CC80"))
    renderings = re.findall(r"IFCSURFACESTYLERENDERING\([^;]*;", text)
    assert renderings, "no surface-style rendering emitted"
    assert any("0.49" in r or "0.5" in r for r in renderings), (
        f"the alpha byte did not become transparency: {renderings[:2]}")


# --------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad,hint", [
    ("#GG0000", "hex colour is #RGB"),
    ("orangeish", "named colours"),
])
def test_an_unreadable_colour_is_a_compile_error(bad, hint):
    """Loud, because the behaviour it replaces was invisible: the element
    simply rendered viewer-default and nothing said why."""
    report = validate_project_report(_model(solid_color=bad))
    assert report.errors, f"{bad!r} must not pass validation"
    msg = " ".join(report.errors)
    assert bad in msg, f"the error must quote the value: {msg}"
    assert hint in msg, f"the error must name the remedy: {msg}"


def test_no_colour_is_still_no_error():
    """The overwhelmingly common case must stay silent."""
    assert not validate_project_report(_model()).errors


def test_a_hex_reaches_the_file_as_that_colour():
    """``#FF8800`` is 255/136/0, which is (1.0, 0.5333, 0.0) normalised. The
    emitted ``IfcColourRgb`` must carry exactly that — a palette lookup that
    silently fell through to the default would still produce a valid file."""
    text = _compile(_model(solid_color="#FF8800"))
    assert (1.0, 0.5333, 0.0) in _colours(text)
