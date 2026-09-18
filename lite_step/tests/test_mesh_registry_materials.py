"""Registry materials on meshes and semantic wrappers.

Solids have long resolved ``material="Concrete_C30-37"`` to a real
IfcMaterial + Psets + registry render colour; ``Mesh`` was excluded and the
mesh render paths only consumed the legacy hex channel. Now the
discriminator is hex-vs-grade, not element type:

* ``Mesh(material="Gravel_16-32")`` → IfcMaterial "Gravel_16-32" associated,
  registry Psets (MassDensity 1700, grain size, EPD), rendered ``#9B9284``.
* ``Mesh(material="#a7a29a")`` and ``mesh_type``/terrain defaults —
  byte-for-byte unchanged (hex never touches the registry).
* The stenkant form — ``Element(IfcEarthworksFill, material="Gravel_16-32")``
  holding un-materialed Mesh patches — associates the IfcMaterial to the
  WRAPPER product and passes the grade's render colour down to the patches.
* ``iter_material_elements`` walks ``project.sites`` (site groundwork was
  invisible to material emission AND the streaming ``can_stream`` gate — the/walk-coverage class), so a grade anywhere routes the build
  to the ifcopenshell backend, the only one that emits IfcMaterial.

Appearance precedence on a mesh: own hex > own grade > inherited wrapper
render > terrain default.
"""

from __future__ import annotations

import os
import tempfile

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc
from lite_step.materials import building_uses_registry_materials, element_registry_material
from lite_step.models import Project, Element, Mesh, Point2D, Site

GRAVEL_RGB = (155, 146, 132)  # #9B9284 — Gravel_16-32 registry render
TERRAIN_RGB = (100, 120, 90)  # the opaque Terrain default


def _model_of(result):
    fd, path = tempfile.mkstemp(suffix=".ifc")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(result.ifc_content)
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _style_rgbs(model):
    return {
        (round(c.SurfaceColour.Red * 255),
         round(c.SurfaceColour.Green * 255),
         round(c.SurfaceColour.Blue * 255))
        for style in model.by_type("IfcSurfaceStyle")
        for c in style.Styles
        if c.is_a("IfcSurfaceStyleShading") or c.is_a("IfcSurfaceStyleRendering")
    }


def _hm(x0, y0, x1, y1, *, h=50, depth=500, material=None):
    m = Mesh(heightmap=[[h, h], [h, h]], depth=depth,
             corner_min=Point2D(x=x0, y=y0), corner_max=Point2D(x=x1, y=y1))
    if material is not None:
        m.material = material
    return m


# ── resolution layer ──────────────────────────────────────────────────────────

def test_mesh_grade_resolves_hex_does_not():
    grade = _hm(0, 0, 1000, 1000, material="Gravel_16-32")
    hexed = _hm(0, 0, 1000, 1000, material="#a7a29a")
    legacy = _hm(0, 0, 1000, 1000, material="Concrete")
    bare = _hm(0, 0, 1000, 1000)
    assert element_registry_material(grade) is not None
    assert element_registry_material(grade).key == "Gravel_16-32"
    assert element_registry_material(hexed) is None
    assert element_registry_material(legacy) is None
    assert element_registry_material(bare) is None


def test_sites_walk_feeds_gate_and_emission():
    """A grade under project.sites must be seen (was invisible: storeys-only
    walk) — it is what flips ``building_uses_registry_materials``, and that is
    what reaches the IfcMaterial emitter."""
    b = Project(name="t")
    site = Site(name="site")
    g = Element(ifc_class="IfcEarthworksFill", name="stenkant")
    g.material = "Gravel_16-32"
    g.add(_hm(0, 0, 1000, 1000))
    site.add(g)
    b.add(site)
    assert building_uses_registry_materials(b) is True


# ── standalone grade mesh ─────────────────────────────────────────────────────

def test_grade_mesh_emits_ifcmaterial_psets_and_render():
    b = Project(name="t")
    b.add(_hm(0, 0, 2000, 2000, material="Gravel_16-32"))
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    m = _model_of(r)

    mats = [x for x in m.by_type("IfcMaterial") if x.Name == "Gravel_16-32"]
    assert mats, "IfcMaterial Gravel_16-32 must be emitted"
    density = [
        p for props in m.by_type("IfcMaterialProperties")
        for p in props.Properties if p.Name == "MassDensity"
    ]
    assert density and density[0].NominalValue.wrappedValue == 1700
    assert GRAVEL_RGB in _style_rgbs(m), _style_rgbs(m)


def test_hex_mesh_unchanged_no_ifcmaterial():
    b = Project(name="t")
    b.add(_hm(0, 0, 2000, 2000, material="#a7a29a"))
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    m = _model_of(r)
    assert not m.by_type("IfcMaterial"), "hex channel must not emit IfcMaterial"
    assert (167, 162, 154) in _style_rgbs(m)  # #a7a29a applied as before


def test_terrain_default_colour_unchanged():
    b = Project(name="t")
    site = Site(name="site")
    t = Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN",
                name="terrain")
    t.add(_hm(-16000, -14500, 16000, 14500, h=0, depth=15000))
    site.add(t)
    b.add(site)
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    m = _model_of(r)
    assert TERRAIN_RGB in _style_rgbs(m)
    assert not m.by_type("IfcMaterial"), "no material= given → no IfcMaterial"


# ── the stenkant form: wrapper grade + un-materialed patches ──────────────────

def _stenkant_building():
    b = Project(name="t")
    site = Site(name="site")
    tw = Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN",
                 name="terrain")
    tw.add(Mesh(heightmap=[[0, 0], [0, 0]], depth=15000,
                corner_min=Point2D(x=-16000, y=-14500),
                corner_max=Point2D(x=16000, y=14500)))
    site.add(tw)
    g = Element(ifc_class="IfcEarthworksFill", predefined_type="BACKFILL",
                name="stenkant")
    g.material = "Gravel_16-32"
    g.add(_hm(6000, -4500, 6500, 4500))
    g.add(_hm(-6500, -4500, -6000, 4500))
    site.add(g)
    b.add(site)
    return b


def test_wrapper_grade_associates_material_to_wrapper_product():
    r = generate_ifc(normalize_project_to_meters(_stenkant_building()))
    assert r.success, r.error
    m = _model_of(r)
    rels = [a for a in m.by_type("IfcRelAssociatesMaterial")
            if a.RelatingMaterial.Name == "Gravel_16-32"]
    assert rels, "wrapper's grade must associate an IfcMaterial"
    assert any(o.is_a("IfcEarthworksFill")
               for rel in rels for o in rel.RelatedObjects), (
        "the association must target the IfcEarthworksFill wrapper product")


def test_wrapper_render_inherited_by_unmaterialed_patches():
    r = generate_ifc(normalize_project_to_meters(_stenkant_building()))
    assert r.success, r.error
    m = _model_of(r)
    rgbs = _style_rgbs(m)
    assert GRAVEL_RGB in rgbs, f"patches must colour from the wrapper grade: {rgbs}"
    assert TERRAIN_RGB in rgbs, "the terrain sibling keeps its default style"


def test_patch_own_hex_beats_inherited_wrapper_render():
    """Precedence: a patch's own hex wins over the wrapper's grade render."""
    b = Project(name="t")
    site = Site(name="site")
    g = Element(ifc_class="IfcEarthworksFill", name="stenkant")
    g.material = "Gravel_16-32"
    g.add(_hm(0, 0, 1000, 1000, material="#112233"))
    site.add(g)
    b.add(site)
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    rgbs = _style_rgbs(_model_of(r))
    assert (17, 34, 51) in rgbs, rgbs          # own hex applied
    assert GRAVEL_RGB not in rgbs, rgbs        # wrapper render NOT painted over it
