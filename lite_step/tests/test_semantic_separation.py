"""Geometry ⊥ semantics ⊥ relations — the clean separation.

A ``Mesh`` is pure GEOMETRY (it carries no semantics); what something IS is
said by a semantic ``Element`` wrapper (``ifc_class`` + ``predefined_type``);
where it belongs is said by the relational container (``Site`` vs storey).

Canonical forms pinned here:

    terrain = Element(ifc_class="IfcGeographicElement",
                      predefined_type="TERRAIN").add(Mesh(...))   # ground
    gravel  = Element(ifc_class="IfcEarthworksFill",
                      predefined_type="BACKFILL").add(Mesh(...))  # groundwork
    site.add(terrain, gravel)                                     # under IfcSite

Also pinned:
* the IFC4.3 vocabulary (groundworks/landscape/geotechnical/marine — roads
  and rail EXCLUDED) on the Element whitelist;
* native IFC4X3_ADD2 emission — IfcEarthworksFill is an IfcEarthworksFill in
  the file (USERDEFINED+ObjectType only for enum values a class lacks);
* carve-host candidacy is GEOMETRIC: EVERY mesh hosts the displacement
  auto-carve, because a mesh is matter and a solid sunk into matter displaces
  it — a footing driven through a gravel bed into the terrain pits both;
* ``site.void()`` still folds into TERRAIN meshes only — that verb is an
  authored statement about the GROUND, not physical displacement, so a
  clearing void must not eat sibling groundwork;
* the legacy ``mesh_type="terrain"`` shim is REMOVED — the value is a loud
  construction error naming the wrapper form;
* ``Site(material=)`` is a loud construction error (a place, not a product);
* mesh children of a semantic Element merge into the wrapper's OWN
  representation (no semantically blank proxy);
* ``color="#RRGGBBAA"`` is the ghost-alpha replacement;
* normalizer mm→m coverage THROUGH the wrapper (the rule:
  every route a mm value can travel gets scaled + tested).
"""

from __future__ import annotations

import os
import tempfile

import pytest

from lite_step.compiler.displacement import _collect, apply_displacement
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    ELEMENT_IFC4_CLASSES,
    ELEMENT_IFC43_CLASSES,
    ELEMENT_IFC_CLASS_WHITELIST,
    Box,
    Project,
    Element,
    Mesh,
    Point,
    Point2D,
    Site,
    taxonomy as tx,
)


def _terrain_wrapper():
    t = Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN",
                name="terrain")
    t.add(Mesh(heightmap=[[0, 0], [0, 0]], depth=15000,
               corner_min=Point2D(x=-16000, y=-14500),
               corner_max=Point2D(x=16000, y=14500)))
    return t


def _gravel_wrapper():
    """Groundwork fill whose AABB deliberately ENCLOSES the test footing —
    the ring-steal shape."""
    g = Element(ifc_class="IfcEarthworksFill", predefined_type="BACKFILL",
                name="stenkant")
    g.add(Mesh(heightmap=[[50, 50], [50, 50]], depth=600,
               corner_min=Point2D(x=-8000, y=-8000),
               corner_max=Point2D(x=8000, y=8000)))
    return g


def _ifc_model(result):
    import ifcopenshell
    fd, path = tempfile.mkstemp(suffix=".ifc")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(result.ifc_content)
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


# ── whitelist: IFC4.3 vocabulary, roads/rail excluded ─────────────────────────

def test_ifc43_groundworks_classes_accepted():
    for cls in ("IfcEarthworksFill", "IfcCourse", "IfcPavement", "IfcKerb",
                "IfcGeotechnicalStratum", "IfcBorehole", "IfcReinforcedSoil"):
        assert cls in ELEMENT_IFC43_CLASSES
        Element(ifc_class=cls)  # constructs


def test_geographic_element_is_ifc4_native():
    assert "IfcGeographicElement" in ELEMENT_IFC4_CLASSES
    Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN")


@pytest.mark.parametrize("cls", ["IfcRail", "IfcTrackElement", "IfcSignal"])
def test_rail_products_are_accepted(cls):
    """Step 2 of infrastructure/specs/ifc43-facilities.md.

    These were refused wholesale as "roads and rail" until the exclusion was
    split by REASON — see ELEMENT_IFC43_CLASSES. They are ordinary products
    (a rail, a sleeper, a signal), they emit natively on IFC4X3_ADD2, and
    each was verified at 0 conformance errors before being added.
    """
    assert cls in ELEMENT_IFC_CLASS_WHITELIST
    Element(ifc_class=cls)  # constructs


@pytest.mark.parametrize("cls", ["IfcRoad", "IfcRailway", "IfcRoadPart",
                                 "IfcRailwayPart", "IfcBridge", "IfcBridgePart"])
def test_spatial_classes_are_refused_by_Element(cls):
    """A spatial class here would be joined to the tree by the WRONG relation.

    ``Element`` emits IfcRelContainedInSpatialStructure; a facility or part
    needs IfcRelAggregates, and WR31 forbids the former for spatial elements.
    ``ifcopenshell.validate`` does not report WHERE rules (measured), so the
    file would pass the conformance gate with the node missing from
    ``getSpatialStructure()``. Use ``SpatialElement``.
    """
    assert cls not in ELEMENT_IFC_CLASS_WHITELIST
    with pytest.raises(Exception, match="unknown ifc_class"):
        Element(ifc_class=cls)


@pytest.mark.parametrize("cls", ["IfcAlignment", "IfcVehicle", "IfcPavementPart"])
def test_still_out_of_scope(cls):
    """Three DIFFERENT reasons, kept apart because they resolve differently.

    ``IfcAlignment`` is deferred work (a whole geometry + placement domain —
    spec §5); ``IfcVehicle`` is rolling stock and not built infrastructure;
    ``IfcPavementPart`` does not exist in IFC4X3_ADD2 at all — it was named
    from memory during review and is simply wrong. Lumping them under one
    "roads and rail excluded" label is what made this test look like a policy
    rather than three separate facts.
    """
    assert cls not in ELEMENT_IFC_CLASS_WHITELIST
    with pytest.raises(Exception, match="unknown ifc_class"):
        Element(ifc_class=cls)


def test_site_accepts_element_children():
    site = Site(name="s")
    site.add(_terrain_wrapper())          # no raise
    with pytest.raises(Exception):
        site.add(Point(x=0, y=0, z=0))    # non-child types still rejected


# ── taxonomy predicates (single source of terrain semantics) ─────────────────

def test_terrain_predicates():
    """The wrapper is the ONLY terrain marker — the legacy shim is gone."""
    tw = _terrain_wrapper()
    gw = _gravel_wrapper()
    plain = Mesh(vertices=[Point(x=0, y=0, z=0)] * 3, faces=[(0, 1, 2)])
    assert tx.is_terrain_wrapper(tw) is True
    assert tx.is_terrain_wrapper(gw) is False
    assert not hasattr(tx, "is_legacy_terrain_mesh"), "shim must be gone"

    site = Site(name="s")
    site.add(tw); site.add(gw); site.add(plain)
    got = list(tx.iter_terrain_meshes(site))
    assert got == [tw._elements[0]], "wrapper meshes only — no bare, no gravel"


def test_mesh_type_is_gone_entirely():
    """``mesh_type`` was removed in the v10 hard cutover — the FIELD, not just
    its ``"terrain"`` value, so ``extra="forbid"`` rejects it by name.

    What protects a stored model is not a per-value migration message: ``LITESTEP_META`` version 10 refuses
    any older base before its source is executed at all, so a pre-cutover
    model never reaches this constructor. Two mechanisms saying the same thing
    is what the version ladder replaced (#663-era denylist lesson).
    """
    for value in ("terrain", "building_ghost", "simple"):
        with pytest.raises(Exception, match="mesh_type"):
            Mesh(mesh_type=value,
                 vertices=[Point(x=0, y=0, z=0)] * 3, faces=[(0, 1, 2)])


def test_site_material_rejected_loudly():
    """IfcSite is a place, not a product — Site(material=) is a construction
        error naming the wrapper form, not a silent drop."""
    with pytest.raises(Exception, match="spatial element"):
        Site(name="s", material="Gravel_16-32")


# ── emission: classes, predefined types, IfcSite placement ────────────────────

def _site_model(*, gravel=True):
    b = Project(name="t")
    site = Site(name="site")
    site.add(_terrain_wrapper())
    if gravel:
        site.add(_gravel_wrapper())
    b.add(site)
    return b, site


def test_terrain_wrapper_emits_native_geographic_terrain_under_site():
    b, _ = _site_model(gravel=False)
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    m = _ifc_model(r)
    geos = m.by_type("IfcGeographicElement")
    assert any(str(g.PredefinedType) == "TERRAIN" for g in geos)
    contained = {
        e.Name
        for rel in m.by_type("IfcRelContainedInSpatialStructure")
        if rel.RelatingStructure.is_a("IfcSite")
        for e in rel.RelatedElements
    }
    assert any("terrain" in (n or "") for n in contained), contained


def test_earthworksfill_emits_natively_under_site():
    """The output schema is IFC4X3_ADD2, so IfcEarthworksFill is an
    IfcEarthworksFill in the file — native class, native BACKFILL predefined
    type, placed under IfcSite. (Pre-flip it rode an IfcGeographicElement
    carrier with the truth in ObjectType.)"""
    b, _ = _site_model()
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    m = _ifc_model(r)
    fills = m.by_type("IfcEarthworksFill")
    assert fills, "IfcEarthworksFill must exist natively in the file"
    assert any(str(f.PredefinedType) == "BACKFILL" for f in fills), (
        [(f.Name, str(f.PredefinedType)) for f in fills])
    contained = {
        e.Name
        for rel in m.by_type("IfcRelContainedInSpatialStructure")
        if rel.RelatingStructure.is_a("IfcSite")
        for e in rel.RelatedElements
    }
    assert any("stenkant" in (n or "") for n in contained), contained


def test_earthworkscut_emits_natively_under_site():
    """IfcEarthworksCut — IfcEarthworksFill's sibling — is whitelisted and
    emits natively, so a schedulable excavation volume CAN be modelled
    explicitly alongside the anonymous baked-in mesh carve (the dsl-reference
    difference() bullet points here)."""
    b = Project(name="t")
    site = Site(name="site")
    cut = Element(ifc_class="IfcEarthworksCut", predefined_type="EXCAVATION",
                  name="excavation")
    cut.add(Box(start=Point(x=0, y=0, z=-900), end=Point(x=2000, y=500, z=0)))
    site.add(cut)
    b.add(site)
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    m = _ifc_model(r)
    cuts = m.by_type("IfcEarthworksCut")
    assert cuts, "IfcEarthworksCut must exist natively in the file"
    assert any(str(c.PredefinedType) == "EXCAVATION" for c in cuts), (
        [(c.Name, str(c.PredefinedType)) for c in cuts])


def test_storey_placed_ifc43_class_emits_natively():
    b = Project(name="t")
    kerb = Element(ifc_class="IfcKerb", name="kerb")
    kerb.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=200, z=150)))
    b.add(kerb)
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    m = _ifc_model(r)
    assert m.by_type("IfcKerb"), "IfcKerb must exist natively in the file"


def test_output_schema_is_ifc4x3_add2_both_backends():
    """The FILE_SCHEMA header is the single pinned output schema — both the
    ifcopenshell backend (via ifcopenshell.file(schema=...)) and the
    streaming backend (via step_header) stamp IFC4X3_ADD2. web-ifc (the
    SPA's fragment converter) parses this header natively."""
    from lite_step.ifc.schema_version import IFC_OUTPUT_SCHEMA
    from lite_step.ifc.step_writer import step_header
    assert IFC_OUTPUT_SCHEMA == "IFC4X3_ADD2"

    # streaming backend header
    assert f"FILE_SCHEMA(('{IFC_OUTPUT_SCHEMA}'));" in step_header()

    # ifcopenshell backend header
    b, _ = _site_model(gravel=False)
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    assert f"FILE_SCHEMA(('{IFC_OUTPUT_SCHEMA}'))" in r.ifc_content[:600]


def test_native_predefined_type_set_on_ifc4_class():
    b = Project(name="t")
    cov = Element(ifc_class="IfcCovering", predefined_type="FLOORING", name="cov")
    cov.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=20)))
    b.add(cov)
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    m = _ifc_model(r)
    covs = m.by_type("IfcCovering")
    assert covs and str(covs[0].PredefinedType) == "FLOORING"


def test_unknown_enum_value_falls_back_to_userdefined():
    b = Project(name="t")
    cov = Element(ifc_class="IfcCovering", predefined_type="GRAVELTOP", name="cov")
    cov.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=20)))
    b.add(cov)
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    m = _ifc_model(r)
    covs = m.by_type("IfcCovering")
    assert covs and str(covs[0].PredefinedType) == "USERDEFINED"
    assert covs[0].ObjectType == "GRAVELTOP"


# ── carve-host semantics (every mesh is matter) ───────────────────────────────

def _with_footing(b):
    b.add(Box(name="footing", start=Point(x=-500, y=-500, z=-900),
              end=Point(x=500, y=500, z=100)))
    return b


def test_collect_returns_every_mesh_as_a_host():
    """``_collect`` does not report host candidacy at all — there is nothing
    left to report, so the mesh tuple is ``(elem, aabb)``. A third field that
    is always True would be a constant dressed as a decision."""
    b, _ = _site_model()
    out = normalize_project_to_meters(b)
    _solids, meshes = _collect(out)
    assert all(len(entry) == 2 for entry in meshes), meshes
    assert len(meshes) == 2, "the terrain mesh and the gravel mesh"


def test_footing_pits_both_the_terrain_and_the_gravel_it_passes_through():
    """The footing is sunk through the gravel bed into the terrain, so it
    displaces both. Were host candidacy SEMANTIC, only the terrain would
        carve and the gravel would close over a footing driven through it."""
    pytest.importorskip("manifold3d")
    b, _ = _site_model()
    _with_footing(b)
    out = normalize_project_to_meters(b)
    terr = [s for s in out.sites[0]._elements
            if tx.is_terrain_wrapper(s)][0]._elements[0]
    grav = [s for s in out.sites[0]._elements
            if type(s).__name__ == "Element"
            and s.ifc_class == "IfcEarthworksFill"][0]._elements[0]
    tv, gv = len(terr.vertices), len(grav.vertices)
    apply_displacement(out)
    assert len(terr.vertices) > tv, "terrain must gain the footing pit"
    assert len(grav.vertices) > gv, "gravel must gain it too — it is matter"


def test_bare_mesh_under_site_hosts_too():
    """A bare Site-child mesh carries no semantics, and needs none: what a
    solid displaces does not depend on what the matter is called."""
    pytest.importorskip("manifold3d")
    b = Project(name="t")
    site = Site(name="site")
    bare = Mesh(heightmap=[[0, 0], [0, 0]], depth=15000,
                corner_min=Point2D(x=-16000, y=-14500),
                corner_max=Point2D(x=16000, y=14500))
    site.add(bare)
    b.add(site)
    _with_footing(b)
    out = normalize_project_to_meters(b)
    terr = [s for s in out.sites[0]._elements if tx.is_mesh(s)][0]
    tv = len(terr.vertices)
    apply_displacement(out)
    assert len(terr.vertices) > tv, "bare mesh must gain the footing pit"


def test_a_solid_clear_of_a_mesh_leaves_it_untouched():
    """The trigger is an AABB but the carve is an exact boolean, so clearance
    IS enough — the falsifier for ``test_bare_mesh_under_site_hosts_too``.
    A bridge deck spanning a valley overlaps the terrain's bounding box and
    must still take nothing out of it; the reference told authors otherwise
    and cost a model its joinery to a blanket ``.no_carve()``."""
    pytest.importorskip("manifold3d")
    b = Project(name="t")
    site = Site(name="site")
    # A trench with a FLAT floor at -3000 for |x| <= 3000, banks at 0. With
    # depth=1000 the mesh AABB is z=[-4000, 0], so anything spanning the
    # trench is inside it.
    xs = [-6000, -3000, 0, 3000, 6000]
    heightmap = [[(-3000 if abs(x) <= 3000 else 0) for x in xs] for _y in xs]
    bare = Mesh(heightmap=heightmap, depth=1000,
                corner_min=Point2D(x=-6000, y=-6000),
                corner_max=Point2D(x=6000, y=6000))
    site.add(bare)
    b.add(site)
    # Spans the trench 2000 mm clear of its floor, and 500 mm below the banks
    # — inside the mesh AABB, touching no surface.
    b.add(Box(name="deck", start=Point(x=-2500, y=-800, z=-1000),
              end=Point(x=2500, y=800, z=-500)))
    out = normalize_project_to_meters(b)
    terr = [s for s in out.sites[0]._elements if tx.is_mesh(s)][0]
    tv = len(terr.vertices)
    apply_displacement(out)
    assert len(terr.vertices) == tv, "clearance is enough — no carve"


def test_wrapper_mesh_geometry_merges_onto_the_semantic_product():
    """The blank-proxy regression: mesh children of a semantic Element merge
    INTO the wrapper's own representation — the IfcGeographicElement carries
    the faceset directly, and no IfcBuildingElementProxy rides along."""
    b, _ = _site_model()
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    m = _ifc_model(r)
    geo = [g for g in m.by_type("IfcGeographicElement")
           if str(g.PredefinedType) == "TERRAIN"][0]
    items = geo.Representation.Representations[0].Items
    assert items and all(i.is_a("IfcTriangulatedFaceSet") for i in items)
    fills = m.by_type("IfcEarthworksFill")
    assert fills and fills[0].Representation is not None
    assert len(fills[0].Representation.Representations[0].Items) == 1
    assert not m.by_type("IfcBuildingElementProxy"), "no blank proxies"


def test_mesh_color_rrggbbaa_is_the_ghost_replacement():
    """``color="#RRGGBBAA"`` (alpha byte) replaces mesh_type="building_ghost"
    — a 0x80 alpha renders ~50% transparent."""
    b = Project(name="t")
    ghost = Mesh(heightmap=[[3000, 3000], [3000, 3000]], depth=3000,
                 corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=8000, y=8000),
                 color="#9E9E9E80")
    b.add(ghost)
    r = generate_ifc(normalize_project_to_meters(b))
    assert r.success, r.error
    m = _ifc_model(r)
    trs = [getattr(c, "Transparency", None)
           for st in m.by_type("IfcSurfaceStyle") for c in st.Styles
           if c.is_a("IfcSurfaceStyleRendering") or c.is_a("IfcSurfaceStyleShading")]
    assert any(t is not None and abs(t - (1 - 128 / 255)) < 0.01 for t in trs), trs


# ── site.void folds into terrain only ─────────────────────────────────────────

def test_site_void_clears_terrain_never_the_gravel():
    pytest.importorskip("manifold3d")
    b, site = _site_model()
    site.void(Box(start=Point(x=2000, y=2000, z=-800),
                  end=Point(x=4000, y=4000, z=20000)))
    assert not validate_project_report(b).errors
    out = normalize_project_to_meters(b)
    terr = [s for s in out.sites[0]._elements
            if tx.is_terrain_wrapper(s)][0]._elements[0]
    grav = [s for s in out.sites[0]._elements
            if type(s).__name__ == "Element"
            and s.ifc_class == "IfcEarthworksFill"][0]._elements[0]
    tv, gv = len(terr.vertices), len(grav.vertices)
    apply_displacement(out)
    assert len(terr.vertices) > tv, "void must clear the terrain"
    assert len(grav.vertices) == gv, "void must NOT eat the gravel bed"


def test_site_void_without_terrain_is_rejected():
    """A site holding only groundwork (no terrain) has nothing for a
    clearing void to clear → validator error, with the wrapper form named."""
    b = Project(name="t")
    site = Site(name="site")
    site.add(_gravel_wrapper())
    site.void(Box(start=Point(x=0, y=0, z=-100), end=Point(x=100, y=100, z=100)))
    b.add(site)
    errs = [e for e in validate_project_report(b).errors if ".void()" in e]
    assert errs and "TERRAIN" in errs[0]


# ── routing + normalization coverage ──────────────────────────────────────────

def test_normalizer_scales_wrapper_children_mm_to_m():
    """The/rule applied to the new route: geometry inside an
    Element wrapper under a Site — vertices AND retained heightmap fields."""
    b, _ = _site_model()
    out = normalize_project_to_meters(b)
    terr = [s for s in out.sites[0]._elements
            if tx.is_terrain_wrapper(s)][0]._elements[0]
    grav = [s for s in out.sites[0]._elements
            if type(s).__name__ == "Element"
            and s.ifc_class == "IfcEarthworksFill"][0]._elements[0]
    assert abs(min(p.z for p in terr.vertices) - (-15.0)) < 1e-9
    assert abs(grav.depth - 0.6) < 1e-9
    assert abs(grav.corner_min.x - (-8.0)) < 1e-9


def test_the_schema_claims_in_the_whitelist_comment_are_true():
    """Pins two facts the ELEMENT_IFC43_CLASSES comment asserts.

    Stated from memory, one of the two was mistaken, so both are asserted
    rather than trusted: ``IfcPavementPart`` does not exist
    in IFC4X3_ADD2, and every rail product we just admitted does.
    """
    pytest.importorskip("ifcopenshell")
    from ifcopenshell import ifcopenshell_wrapper as w
    schema = w.schema_by_name("IFC4X3_ADD2")

    with pytest.raises(Exception):
        schema.declaration_by_name("IfcPavementPart")

    for cls in ("IfcRail", "IfcTrackElement", "IfcSignal"):
        d = schema.declaration_by_name(cls)
        assert not d.is_abstract(), f"{cls} is abstract and cannot be instantiated"
        chain, x = [], d
        while x:
            chain.append(x.name())
            x = x.supertype()
        assert "IfcProduct" in chain, f"{cls} is not an IfcProduct"
        assert "IfcSpatialElement" not in chain, (
            f"{cls} is SPATIAL — Element would emit it into a containment "
            f"relation (WR31), and no validator would say so")


@pytest.mark.parametrize("cls,predefined", [
    ("IfcRail", "RAIL"),
    ("IfcTrackElement", "SLEEPER"),
    ("IfcSignal", "VISUAL"),
])
def test_rail_products_emit_conformant_ifc(cls, predefined):
    """Measured before the whitelist was widened, pinned after.

    ``IfcSignal`` is the one worth having a test for: its supertype is
    ``IfcDistributionElement``, not ``IfcBuiltElement`` like the other two, so
    "the generic creator handles it" was an assumption until it was run.
    """
    import contextlib
    import io as _io
    import os
    import tempfile

    ios = pytest.importorskip("ifcopenshell")
    import ifcopenshell.validate

    from lite_step.compiler.executor import normalize_project_to_meters
    from lite_step.ifc.generator import generate_ifc
    from lite_step.models import Box, Material, Point, Project, Site

    proj = Project(name="railtest")
    site = Site(name="site")
    elem = Element(ifc_class=cls, predefined_type=predefined, name="unit")
    elem.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=1000, y=200, z=200),
                 material=Material(key="Steel_S250GD_Z275")))
    site.add(elem)
    proj.add(site)

    with contextlib.redirect_stdout(_io.StringIO()):
        res = generate_ifc(normalize_project_to_meters(proj))
    path = os.path.join(tempfile.mkdtemp(), "m.ifc")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(res.ifc_content)
    f = ios.open(path)

    emitted = f.by_type(cls)
    assert len(emitted) == 1, f"{cls} was not emitted natively"
    assert emitted[0].PredefinedType == predefined

    logger = ifcopenshell.validate.json_logger()
    with contextlib.redirect_stderr(_io.StringIO()):
        ifcopenshell.validate.validate(f, logger)
    errors = [s for s in logger.statements if s.get("level") == "error"]
    assert not errors, f"{cls}: {len(errors)} schema error(s): {errors[:2]}"
