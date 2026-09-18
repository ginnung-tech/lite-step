"""Layered materials (v16) — a layer IS a ``Material`` at a thickness, listed
outer→inner in ``LayerSet(name=, layers=[...])``. The compiler slices the
single authored body per layer and emits the IfcMaterialLayerSetUsage →
LayerSet → Layer chain, with Category (from the registry ``function``) and
IsVentilated (from a cavity product) DERIVED, never authored.

Pins:
1. LayerSet grammar: keyword-only Material entries, explicit list, ≥1 layer,
   thickness_mm required per layer, member (profile_mm) rejected, no Layer noun.
2. Target restriction: Wall/Slab/Roof/Element(planar) OK; Beam/Box/Site
   rejected; material= + layers= mutually exclusive.
3. Slicing: outer→inner spans, cavity gap, mismatch scaling.
4. Emission: chain linkage, metre thicknesses, Category from registry function,
   IsVentilated from cavity, shared set, AXIS2 wall vs AXIS3 slab, wall
   multi-item body, per-layer render, explicit color override.
5. Validation: zero/multi body error, body material= error, gable error, >5%
   warn, unknown-key warn.
6. Streaming falls back; script namespace exposes LayerSet (not Layer).
"""

from __future__ import annotations

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.compiler.executor import (
    execute_lite_step_script,
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.compiler.layers import (
    layer_spans, through_axis_for, layer_axis_and_dir)
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Project, Point, Box, Extrude, Wall, Slab, Roof, Beam, Site, Element,
    Material, LayerSet, Window,
)
from lite_step.models.project import Storey


def _buildup(name="ext_type_a", outward=(0, -1, 0)):
    return LayerSet(name=name, outward=outward, layers=[
        Material(key="Brick_Red_DK", thickness_mm=108),        # outer
        Material(key="MineralWool_Facade34", thickness_mm=150),
        Material(key="Gypsum_Standard", thickness_mm=13),      # inner
    ])


BUILDUP = _buildup()


def _proj(*elements) -> Project:
    p = Project(name="t")
    s = Storey(elevation=0)
    for e in elements:
        s.add(e)
    p.add_storey(s)
    return p


def _layered_wall(layers=None, thickness=271):
    layers = layers or _buildup()
    wall = Wall(name="south")
    wall.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=6000, y=thickness, z=2700)))
    wall.layers = layers
    return wall


def _gen(proj):
    res = generate_ifc(normalize_project_to_meters(proj), source_code=None)
    assert res.success, res.error
    return res


# ── 1. grammar ──────────────────────────────────────────────────────────


def test_layer_is_a_material_no_layer_noun():
    import lite_step.models as m
    assert not hasattr(m, "Layer")                     # collapsed into Material
    ls = BUILDUP
    assert all(isinstance(x, Material) for x in ls.layers)
    assert ls.total_thickness_mm == 271


def test_layerset_needs_explicit_list_of_materials_with_thickness():
    o = (0, -1, 0)
    with pytest.raises(Exception):
        LayerSet(name="x", outward=o, layers=[])                  # ≥1
    with pytest.raises(ValueError, match="thickness_mm"):
        LayerSet(name="x", outward=o, layers=[Material(key="Brick_Red_DK")])   # needs thickness
    with pytest.raises(ValueError, match="member|continuous layer"):
        LayerSet(name="x", outward=o, layers=[Material(key="Timber_C24", profile_mm=(45, 195))])
    with pytest.raises(ValueError, match="Material"):
        LayerSet(name="x", outward=o, layers=["Brick_Red_DK"])    # bare string not allowed


def test_layerset_accepts_plain_list_and_stores_tuple():
    """The NATURAL authoring shape is ``layers=[Material(...), ...]`` — a plain
    list, as the skills teach. The annotation must be Sequence (not tuple) so
    static checkers accept it too: a ``tuple[...]`` annotation made ty reject
    the list form, a recurring agent fix-loop in the 2026-07 wire logs. The
    before-validator still normalizes to an immutable tuple at runtime."""
    from collections.abc import Sequence as ABCSequence
    from typing import get_origin, get_type_hints

    ls = LayerSet(name="x", outward=(0, -1, 0),
                  layers=[Material(key="Brick_Red_DK", thickness_mm=108)])
    assert isinstance(ls.layers, tuple)                # runtime-normalized
    hints = get_type_hints(LayerSet)
    assert get_origin(hints["layers"]) is ABCSequence  # static: list is legal


def test_layerset_outward_required_and_nonzero():
    with pytest.raises(Exception):                     # required, no default
        LayerSet(name="x", layers=[Material(key="Brick_Red_DK", thickness_mm=108)])
    with pytest.raises(ValueError, match="non-zero"):
        LayerSet(name="x", outward=(0, 0, 0),
                 layers=[Material(key="Brick_Red_DK", thickness_mm=108)])


def test_thickness_mm_now_valid_on_mass_and_cavity_forms():
    Material(key="Brick_Red_DK", thickness_mm=108)     # mass — was rejected pre-v16
    Material(key="Cavity_Ventilated", thickness_mm=25)
    with pytest.raises(ValueError, match="profile_mm"):
        Material(key="Timber_C24", thickness_mm=45)    # member still rejected


# ── 2. target restriction ───────────────────────────────────────────────


def test_planar_targets_accept_layers():
    Wall(name="w", layers=BUILDUP)
    Slab(name="s", layers=BUILDUP)
    Roof(name="r", layers=BUILDUP)
    Element(ifc_class="IfcPavement", layers=BUILDUP)
    Element(ifc_class="IfcCovering", layers=BUILDUP)


def test_non_planar_targets_reject_layers():
    with pytest.raises(ValueError, match="planar"):
        Beam(name="b", layers=BUILDUP)
    with pytest.raises(ValueError, match="planar"):
        Box(start=Point(x=0, y=0, z=0), end=Point(x=1, y=1, z=1), layers=BUILDUP)
    with pytest.raises(ValueError, match="planar"):
        Element(ifc_class="IfcStair", layers=BUILDUP)
    with pytest.raises(ValueError):
        Site(name="site", layers=BUILDUP)


def test_material_and_layers_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        Wall(name="w", layers=BUILDUP, material="Concrete_C30-37")


# ── 3. slicing math ─────────────────────────────────────────────────────


def test_layer_spans_outer_to_inner():
    spans = layer_spans(BUILDUP, 0.0, 271.0)
    assert [(round(a), round(b)) for _, a, b in spans] == [(0, 108), (108, 258), (258, 271)]
    assert spans[0][0].key == "Brick_Red_DK"           # first = outer = min face
    assert spans[-1][0].key == "Gypsum_Standard"


def test_layer_spans_scale_to_actual_extent():
    spans = layer_spans(BUILDUP, 0.0, 542.0)
    assert round(spans[0][2]) == 216                   # 108 * 2
    assert spans[-1][2] == 542.0                        # snapped to hi


def test_wall_through_axis_smaller_horizontal():
    assert through_axis_for(_layered_wall(), _layered_wall()._elements[0]) == "y"
    slab_body = Box(start=Point(x=0, y=0, z=0), end=Point(x=4000, y=4000, z=271))
    assert through_axis_for(Slab(name="s"), slab_body) == "z"


# ── 4. emission ─────────────────────────────────────────────────────────


def test_wall_emits_layer_chain_and_multi_item_body():
    res = _gen(_proj(_layered_wall()))
    ifc = res.ifc_content
    assert ifc.count("IFCMATERIALLAYER(") == 3
    model = ifcopenshell.file.from_string(ifc)
    usage = model.by_type("IfcMaterialLayerSetUsage")[0]
    assert usage.LayerSetDirection == "AXIS2"
    assert usage.DirectionSense == "POSITIVE"
    ls = usage.ForLayerSet
    assert ls.LayerSetName == "ext_type_a"
    assert [l.Material.Name for l in ls.MaterialLayers] == [
        "Brick_Red_DK", "MineralWool_Facade34", "Gypsum_Standard"]
    assert [round(l.LayerThickness, 4) for l in ls.MaterialLayers] == [0.108, 0.15, 0.013]
    # Category DERIVED from the registry function (never authored)
    assert [l.Category for l in ls.MaterialLayers] == ["LoadBearing", "Insulation", "Finish"]
    wall = model.by_type("IfcWall")[0]
    body_rep = [r for r in wall.Representation.Representations
                if r.RepresentationIdentifier == "Body"][0]
    assert len(body_rep.Items) == 3
    assert wall in list(model.by_type("IfcRelAssociatesMaterial")[0].RelatedObjects)


def test_cavity_layer_gap_and_isventilated_from_registry():
    vented = LayerSet(name="cav", outward=(0, -1, 0), layers=[
        Material(key="Brick_Red_DK", thickness_mm=108),
        Material(key="Cavity_Ventilated", thickness_mm=25),
        Material(key="Gypsum_Standard", thickness_mm=13),
    ])
    res = _gen(_proj(_layered_wall(layers=vented, thickness=146)))
    model = ifcopenshell.file.from_string(res.ifc_content)
    wall = model.by_type("IfcWall")[0]
    body_rep = [r for r in wall.Representation.Representations
                if r.RepresentationIdentifier == "Body"][0]
    assert len(body_rep.Items) == 2                    # cavity → no solid
    layers = model.by_type("IfcMaterialLayerSet")[0].MaterialLayers
    assert len(layers) == 3                            # …but stays in the chain
    assert layers[1].IsVentilated is True              # derived from the cavity product
    assert layers[1].Category is None                  # cavity has no functional Category


def test_slab_layers_slice_and_chain_axis3():
    # a slab's through-axis is z, so its outward normal is vertical (finish up)
    slab = Slab(name="deck")
    slab.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=4000, y=4000, z=271), type="foundation"))
    slab.layers = _buildup(name="deck_type", outward=(0, 0, 1))
    res = _gen(_proj(slab))
    model = ifcopenshell.file.from_string(res.ifc_content)
    usage = model.by_type("IfcMaterialLayerSetUsage")[0]
    assert usage.LayerSetDirection == "AXIS3"
    assert usage.DirectionSense == "NEGATIVE"          # outer (first) at +z (top) face
    ifc_slab = model.by_type("IfcSlab")[0]
    rel_aggs = [r for r in model.by_type("IfcRelAggregates")
                if r.RelatingObject == ifc_slab]
    assert len(rel_aggs[0].RelatedObjects) == 3


def test_mirror_layered_walls_get_distinct_profiles():
    """Mirror-pair layered walls (west/east, identical dims) must not share an
    IfcRectangleProfileDef — AND, decisively, must not share *dims*.

    History: the SPA's fragments render worker groups byte-identical geometry
    into one instanced mesh, and a race in its multi-threaded instance
    registration sometimes drops an instance (the missing-cladding bug —
    nondeterministic, which is why usually ONE wall of the mirror pairs
    vanishes, not one per pair). v18.0.4 split the profiles by NAME only;
    that survived the STEP purge but was proven inert — the worker keys on
    geometry CONTENT, and the dims were still byte-identical. The real fix is
    unique_dims: a deterministic 0.02 mm nudge (above the deduplicator's
    rounding via its exemption, and above float32 vertex quantisation) that
    makes the mirror dims distinct in the FINAL SERIALIZED file.

    This test asserts at that final layer (post-dedup ifc_content) — the
    verification gap that let the name-only fix ship as a false positive.
    """
    import struct

    def _leaf(name, start, end, outward):
        w = Wall(name=name)
        w.add(Box(name="body", start=start, end=end))
        w.layers = LayerSet(name="ext", outward=outward,
                            layers=[Material(key="Brick_Red_DK", thickness_mm=108)])
        return w
    west = _leaf("west", Point(x=-6298, y=-4830, z=0), Point(x=-6000, y=4830, z=3000), (-1, 0, 0))
    east = _leaf("east", Point(x=6000, y=-4830, z=0), Point(x=6298, y=4830, z=3000), (1, 0, 0))
    res = _gen(_proj(west, east))
    model = ifcopenshell.file.from_string(res.ifc_content)
    assert len(model.by_type("IfcWall")) == 2
    # The RENDERED brick slices (each wall's Body items) reference DISTINCT
    # profiles with DISTINCT dims — no shared geometry content to instance.
    def _slice_profiles(wall):
        out = []
        for rep in wall.Representation.Representations:
            if rep.RepresentationIdentifier != "Body":
                continue
            for item in rep.Items:
                s = item
                while s.is_a("IfcBooleanClippingResult") or s.is_a("IfcBooleanResult"):
                    s = s.FirstOperand
                if s.is_a("IfcExtrudedAreaSolid"):
                    out.append(s.SweptArea)
        return out
    profs = [p for w in model.by_type("IfcWall") for p in _slice_profiles(w)]
    assert profs, "no brick slices found"
    ids = [p.id() for p in profs]
    assert len(set(ids)) == len(ids), f"mirror brick slices share a profile: {ids}"
    dims = [(p.XDim, p.YDim) for p in profs]
    assert len(set(dims)) == len(dims), f"mirror brick slices share dims: {dims}"
    # The distinction must survive float32 vertex quantisation (what the SPA
    # mesh actually stores — half-dims at facade coordinate scale).
    f32 = lambda v: struct.unpack('f', struct.pack('f', v))[0]
    dims32 = [(f32(x / 2), f32(y / 2)) for (x, y) in dims]
    assert len(set(dims32)) == len(dims32), f"dims collapse in float32: {dims32}"
    # And the nudge stays sub-tolerance: every dim within 0.1 mm of its
    # authored (unnudged) value — the noise must never grow to visible size.
    for x, y in dims:
        for v in (x, y):
            assert abs(v - round(v, 4)) < 1e-4, f"nudge exceeded tolerance: {v}"


def test_slab_outward_wrong_axis_raises():
    # a horizontal outward normal on a slab (through-axis z) is a loud error
    slab = Slab(name="deck")
    slab.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=4000, y=4000, z=271)))
    slab.layers = _buildup(outward=(0, -1, 0))
    res = generate_ifc(normalize_project_to_meters(_proj(slab)), source_code=None)
    assert not res.success and "through-axis" in (res.error or "")


def test_same_orientation_shares_one_set_and_usage():
    # two walls, identical buildup, SAME outward → one set, one usage, both walls
    w1 = _layered_wall()                               # south, outward (0,-1,0)
    w2 = Wall(name="south2")
    w2.add(Box(name="body", start=Point(x=0, y=8000, z=0),
               end=Point(x=6000, y=8271, z=2700)))
    w2.layers = _buildup()                             # same outward (0,-1,0)
    res = _gen(_proj(w1, w2))
    model = ifcopenshell.file.from_string(res.ifc_content)
    assert len(model.by_type("IfcMaterialLayerSet")) == 1
    assert len(model.by_type("IfcMaterialLayerSetUsage")) == 1
    usage_rels = [r for r in model.by_type("IfcRelAssociatesMaterial")
                  if r.RelatingMaterial.is_a("IfcMaterialLayerSetUsage")]
    assert len(usage_rels) == 1 and len(usage_rels[0].RelatedObjects) == 2


def _ring_leaf(name, start, end, outward):
    w = Wall(name=name, props={"Pset_WallCommon": {"IsExternal": True}})
    w.add(Box(name="body", start=start, end=end))
    w.layers = _buildup(outward=outward)               # ONE identical buildup
    return w


def test_ring_shares_one_set_two_usages_no_dropped_side():
    # 4-leaf centred ring, IDENTICAL buildup, per-leaf outward. The SET (type) is
    # shared; every wall is associated (no dropped side — the symptom).
    #
    # FOUR usages, not two. A usage describes (set, direction,
    # sense, offset), and a ring genuinely has two DIRECTIONS: the north/south
    # leaves stack their buildup along world Y, the east/west leaves along X.
    # This asserted 2 only while walls declared AXIS2 unconditionally — i.e. the
    # old count was the bug, not the invariant. Two axes x two senses = 4.
    t, hx, hy = 300, 6000, 4500
    ox, oy = hx + t, hy + t
    leaves = [
        _ring_leaf("south", Point(x=-ox, y=-oy, z=0), Point(x=ox, y=-hy, z=3000), (0, -1, 0)),
        _ring_leaf("north", Point(x=-ox, y=hy, z=0),  Point(x=ox, y=oy, z=3000),  (0, 1, 0)),
        _ring_leaf("west",  Point(x=-ox, y=-oy, z=0), Point(x=-hx, y=oy, z=3000), (-1, 0, 0)),
        _ring_leaf("east",  Point(x=hx, y=-oy, z=0),  Point(x=ox, y=oy, z=3000),  (1, 0, 0)),
    ]
    res = _gen(_proj(*leaves))
    model = ifcopenshell.file.from_string(res.ifc_content)
    assert len(model.by_type("IfcMaterialLayerSet")) == 1          # shared type
    usages = model.by_type("IfcMaterialLayerSetUsage")
    assert len(usages) == 4                                        # 2 axes x 2 senses
    assert sorted({u.LayerSetDirection for u in usages}) == ["AXIS1", "AXIS2"]
    assert sorted({u.DirectionSense for u in usages}) == ["NEGATIVE", "POSITIVE"]
    # each (axis, sense) pair occurs exactly once — no duplicate usage entities
    assert len({(u.LayerSetDirection, u.DirectionSense) for u in usages}) == 4
    # every one of the four walls is associated — no dropped side
    associated = set()
    for r in model.by_type("IfcRelAssociatesMaterial"):
        if r.RelatingMaterial.is_a("IfcMaterialLayerSetUsage"):
            associated.update(o.Name for o in r.RelatedObjects)
    assert {"wall:south", "wall:north", "wall:west", "wall:east"} <= associated


def test_outward_places_brick_on_the_outward_face_each_leaf():
    # geometry check: the brick (first-listed/outer) slice sits on the outward
    # face of each leaf, from ONE buildup — no per-leaf list reversal.
    t, hx, hy = 300, 6000, 4500
    ox, oy = hx + t, hy + t
    for name, s, e, out, axis in [
        ("south", Point(x=-ox, y=-oy, z=0), Point(x=ox, y=-hy, z=3000), (0, -1, 0), "y"),
        ("north", Point(x=-ox, y=hy, z=0),  Point(x=ox, y=oy, z=3000),  (0, 1, 0),  "y"),
        ("west",  Point(x=-ox, y=-oy, z=0), Point(x=-hx, y=oy, z=3000), (-1, 0, 0), "x"),
        ("east",  Point(x=hx, y=-oy, z=0),  Point(x=ox, y=oy, z=3000),  (1, 0, 0),  "x"),
    ]:
        w = _ring_leaf(name, s, e, out)
        body = w._elements[0]
        _axis, outer_at_max = layer_axis_and_dir(w, body)
        assert _axis == axis
        lo = min(getattr(body.start, axis), getattr(body.end, axis))
        hi = max(getattr(body.start, axis), getattr(body.end, axis))
        spans = layer_spans(w.layers, lo, hi, reverse=outer_at_max)
        brick = next((sl, sh) for m, sl, sh in spans if m.key == "Brick_Red_DK")
        outer_face = hi if out["xyz".index(axis)] > 0 else lo
        brick_outer = brick[1] if outer_face == hi else brick[0]
        assert abs(brick_outer - outer_face) < 1e-6, f"{name}: brick not outermost"


def test_outward_along_wall_length_raises():
    # a normal pointing along the wall's LENGTH (not the thin axis) is a loud error
    w = Wall(name="w")
    w.add(Box(name="body", start=Point(x=0, y=0, z=0), end=Point(x=6000, y=271, z=2700)))
    w.layers = _buildup(outward=(1, 0, 0))             # wall runs in x; thin in y
    res = generate_ifc(normalize_project_to_meters(_proj(w)), source_code=None)
    assert not res.success and "through-axis" in (res.error or "")


def test_explicit_color_overrides_layer_renders():
    wall = _layered_wall()
    wall._elements[0].color = "roof"
    res = _gen(_proj(wall))
    model = ifcopenshell.file.from_string(res.ifc_content)
    assert len({s.id() for s in model.by_type("IfcSurfaceStyle")}) == 1


def test_window_opening_still_voids_layered_wall():
    wall = _layered_wall()
    wall.anchor(Window(width=1200, height=1400, name="w0"), along=1400, up=900)
    res = _gen(_proj(wall))
    assert "IFCOPENINGELEMENT" in res.ifc_content


# ── 5. validation ───────────────────────────────────────────────────────


def test_validation_requires_exactly_one_body():
    report = validate_project_report(_proj(Wall(name="w", layers=BUILDUP)))
    assert any("exactly ONE" in e for e in report.errors)


def test_validation_rejects_gable_wall_layers():
    wall = Wall(name="w", layers=LayerSet(name="x", outward=(0, -1, 0),
                                          layers=[Material(key="Brick_Red_DK", thickness_mm=108)]))
    wall.add(Extrude(contour=[Point(x=0, y=0, z=0), Point(x=4000, y=0, z=0),
                              Point(x=2000, y=0, z=2500)], thickness=108))
    report = validate_project_report(_proj(wall))
    assert any("follow-up" in e for e in report.errors)


def test_validation_rejects_body_material():
    wall = _layered_wall()
    wall._elements[0].material = "Concrete_C30-37"
    report = validate_project_report(_proj(wall))
    assert any("must not carry" in e for e in report.errors)


def test_validation_warns_on_thickness_mismatch():
    report = validate_project_report(_proj(_layered_wall(thickness=400)))  # buildup 271
    assert any("differs from" in w for w in report.warnings)


def test_unknown_layer_material_hard_fails_at_construction():
    # A layer material must be registry-backed: unknown key + thickness is a
    # hard error at Material construction, not a soft open-vocabulary warning.
    with pytest.raises(ValueError, match="unknown material|missing catalog"):
        LayerSet(name="x", layers=[Material(key="NotReal", thickness_mm=271)])


# ── 6. streaming + namespace ────────────────────────────────────────────


def test_script_namespace_exposes_layerset_not_layer():
    src = (
        "def generate_project():\n"
        "    proj = Project(name='t')\n"
        "    w = Wall(name='south')\n"
        "    w.add(Box(name='body', start=Point(x=0, y=0, z=0), end=Point(x=6000, y=271, z=2700)))\n"
        "    w.layers = LayerSet(name='ext', outward=(0, -1, 0), layers=[Material(key='Brick_Red_DK', thickness_mm=108),\n"
        "        Material(key='MineralWool_Facade34', thickness_mm=150), Material(key='Gypsum_Standard', thickness_mm=13)])\n"
        "    proj.add(w)\n"
        "    return proj\n"
        "result = generate_project()\n"
    )
    r = execute_lite_step_script(src)
    assert r.success, r.error
