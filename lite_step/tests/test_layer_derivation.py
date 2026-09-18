"""The chain must derive the geometry we drew.

A consumer does not read our slices. Bonsai, Solibri and Revit all re-derive
the layer boundaries from the ``IfcMaterialLayerSetUsage`` chain — the
product's placement, ``OffsetFromReferenceLine``, ``DirectionSense`` and each
``LayerThickness`` — and bisect the body with them. The chain is therefore a
CLAIM about the geometry, and this file checks the claim against what we
actually emitted.

Two defects hid behind the absence of that check, because each side was only
ever tested against itself:

* ``OffsetFromReferenceLine`` was hardcoded ``0.0`` while our products are
  placed at their body's AABB **centre**. On a 108/43/190 wall a consumer then
  cut at +108 mm and +151 mm from the centre — both inside the insulation,
  43 mm apart. That pair of planes is the phantom band reported against the
  facade, and it tracked the middle layer's thickness because it *is* that
  thickness. Changing the cavity 30 mm → 43 mm moved the band to 43 mm, which
  is what convicted the datum rather than the cavity.
* ``layers.layer_spans`` fits the buildup proportionally onto the body's real
  extent, but the emitted ``LayerThickness`` stayed **nominal**. Wherever the
  two disagreed the chain described a buildup we did not draw, and bands
  drifted progressively.

One invariant catches both, and survives a refactor of either side.
"""

from __future__ import annotations

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Box, LayerSet, Material, Point, Project, Wall,
)
from lite_step.models.elements import miter
from lite_step.models.project import Storey

#: 0.1 mm — the display/QTO precision the repo already snaps linear measures
#: to. Deliberately looser than exact so the 0.02 mm ``unique_dims``
#: anti-instancing salt does not read as a datum error, and far tighter than
#: the 26 mm the bug produced.
TOL = 1e-4

_AXIS_I = {"AXIS1": 0, "AXIS2": 1, "AXIS3": 2}


def _buildup(name="ext_type_a", outward=(0, -1, 0)):
    return LayerSet(name=name, outward=outward, layers=[
        Material(key="Brick_Red_DK", thickness_mm=108),        # outer
        Material(key="MineralWool_Facade34", thickness_mm=150),
        Material(key="Gypsum_Standard", thickness_mm=13),      # inner
    ])


def _proj(*elements) -> Project:
    p = Project(name="t")
    s = Storey(elevation=0)
    for e in elements:
        s.add(e)
    p.add_storey(s)
    return p


def _layered_wall(layers=None, thickness=271, name="south"):
    wall = Wall(name=name)
    wall.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=6000, y=thickness, z=2700)))
    wall.layers = layers or _buildup()
    return wall


def _gen(proj):
    res = generate_ifc(normalize_project_to_meters(proj), source_code=None)
    assert res.success, res.error
    return ifcopenshell.file.from_string(res.ifc_content)


# ── the invariant ───────────────────────────────────────────────────────


def _chain_planes(model, product):
    """Boundary triples ``(lo, hi, is_cavity)`` a consumer derives from
    ``product``'s usage chain, in the product's local frame, in chain order."""
    rels = [r for r in model.by_type("IfcRelAssociatesMaterial")
            if product in r.RelatedObjects
            and r.RelatingMaterial.is_a("IfcMaterialLayerSetUsage")]
    assert len(rels) == 1, f"expected one layer usage, got {len(rels)}"
    usage = rels[0].RelatingMaterial
    step = 1.0 if usage.DirectionSense == "POSITIVE" else -1.0
    cur = usage.OffsetFromReferenceLine
    out = []
    for layer in usage.ForLayerSet.MaterialLayers:
        nxt = cur + step * layer.LayerThickness
        out.append((min(cur, nxt), max(cur, nxt), bool(layer.IsVentilated)))
        cur = nxt
    return usage, out


def _drawn_spans(model, product, axis_i):
    """``(lo, hi)`` of every emitted slice along ``axis_i``, ascending."""
    spans = []
    for rep in product.Representation.Representations:
        for item in rep.Items:
            # Descend a clip/boolean tree to its base solid; the SecondOperand
            # is the tool being applied, not the material.
            while item.is_a("IfcBooleanResult"):
                item = item.FirstOperand
            if not item.is_a("IfcExtrudedAreaSolid"):
                continue
            prof = item.SweptArea
            assert prof.is_a("IfcRectangleProfileDef"), prof
            t = (prof.XDim, prof.YDim)[axis_i]
            c = item.Position.Location.Coordinates[axis_i]
            spans.append((c - t / 2.0, c + t / 2.0))
    return sorted(spans)


def assert_chain_matches_geometry(model, product):
    usage, planes = _chain_planes(model, product)
    axis_i = _AXIS_I[usage.LayerSetDirection]
    drawn = _drawn_spans(model, product, axis_i)
    if usage.DirectionSense == "NEGATIVE":
        drawn = drawn[::-1]              # the chain walks the other way
    solid = [p for p in planes if not p[2]]
    assert len(solid) == len(drawn), (
        f"{product.Name}: chain describes {len(solid)} solid layers, "
        f"geometry has {len(drawn)}")
    for (lo, hi, _), (dlo, dhi) in zip(solid, drawn):
        assert abs(lo - dlo) < TOL and abs(hi - dhi) < TOL, (
            f"{product.Name}: chain says {lo:+.5f}..{hi:+.5f} but we drew "
            f"{dlo:+.5f}..{dhi:+.5f} — a consumer bisects where there is no "
            f"material boundary")
    return usage


# ── the cases ───────────────────────────────────────────────────────────


def test_chain_derives_the_geometry_we_drew_single_wall():
    model = _gen(_proj(_layered_wall()))
    usage = assert_chain_matches_geometry(model, model.by_type("IfcWall")[0])
    # 271 mm buildup, outer layer at the MIN face, placement at the centre →
    # the datum is half a thickness out, not zero.
    assert usage.OffsetFromReferenceLine == pytest.approx(-0.1355)
    assert usage.DirectionSense == "POSITIVE"


def _ring_leaf(name, start, end, outward):
    w = Wall(name=name)
    b = Box(name="body", start=start, end=end)
    w.add(b)
    w.layers = LayerSet(name="ext_wall_type_a", outward=outward, layers=[
        Material(key="Brick_Red_DK", thickness_mm=108),
        Material(key="Cavity_Ventilated", thickness_mm=43),
        Material(key="MineralWool_Facade34", thickness_mm=190),
    ])
    return w, b


def _ring():
    """Four mitered facade leaves with a cavity — the shape the bug was
    reported on. Two slicing axes and both senses in one file."""
    t, hx, hy, top = 341, 6000, 4500, 3000
    ox, oy = hx + t, hy + t
    s, s_b = _ring_leaf("facade_south", Point(x=-ox, y=-oy, z=0), Point(x=ox, y=-hy, z=top), (0, -1, 0))
    n, n_b = _ring_leaf("facade_north", Point(x=-ox, y=hy, z=0), Point(x=ox, y=oy, z=top), (0, 1, 0))
    w, w_b = _ring_leaf("facade_west", Point(x=-ox, y=-oy, z=0), Point(x=-hx, y=oy, z=top), (-1, 0, 0))
    e, e_b = _ring_leaf("facade_east", Point(x=hx, y=-oy, z=0), Point(x=ox, y=oy, z=top), (1, 0, 0))
    miter(s_b, w_b, at=Point(x=-ox, y=-oy, z=0), edge=(0, 0, 1))
    miter(s_b, e_b, at=Point(x=ox, y=-oy, z=0), edge=(0, 0, 1))
    miter(n_b, w_b, at=Point(x=-ox, y=oy, z=0), edge=(0, 0, 1))
    miter(n_b, e_b, at=Point(x=ox, y=oy, z=0), edge=(0, 0, 1))
    return s, n, w, e


def test_chain_derives_the_geometry_on_every_leaf_of_a_mitered_ring():
    model = _gen(_proj(*_ring()))
    walls = model.by_type("IfcWall")
    assert len(walls) == 4
    seen = {}
    for wall in walls:
        u = assert_chain_matches_geometry(model, wall)
        seen[wall.Name] = (u.LayerSetDirection, u.DirectionSense,
                           round(u.OffsetFromReferenceLine, 6))
    # Both axes and both senses are exercised, and NOT ONE datum is zero —
    # the hardcoded value that produced the phantom band.
    assert {v[0] for v in seen.values()} == {"AXIS1", "AXIS2"}, seen
    assert {v[1] for v in seen.values()} == {"POSITIVE", "NEGATIVE"}, seen
    assert all(v[2] != 0.0 for v in seen.values()), seen


def test_the_cavity_span_is_a_real_void_not_a_drawn_slice():
    """A cavity holds its span in the chain and emits NO solid — by design.
    That makes the chain the ONLY record of where the gap is, which is why a
    wrong datum surfaced as a band inside the neighbouring layer rather than
    as missing geometry. Pin both halves."""
    layers = LayerSet(name="cav", outward=(0, -1, 0), layers=[
        Material(key="Brick_Red_DK", thickness_mm=108),
        Material(key="Cavity_Ventilated", thickness_mm=43),
        Material(key="MineralWool_Facade34", thickness_mm=190),
    ])
    model = _gen(_proj(_layered_wall(layers=layers, thickness=341)))
    wall = model.by_type("IfcWall")[0]
    assert_chain_matches_geometry(model, wall)
    _, planes = _chain_planes(model, wall)
    cavities = [p for p in planes if p[2]]
    assert len(cavities) == 1
    assert cavities[0][1] - cavities[0][0] == pytest.approx(0.043)
    # three layers, two solids — the cavity really is a void
    assert len(_drawn_spans(model, wall, 1)) == 2


def test_a_stretched_buildup_reports_the_thickness_it_was_stretched_to():
    """The second, independent defect: 271 mm of nominal buildup drawn into a
    280 mm body (3.3% — under the 5% validation warn, so nothing else
    complains)."""
    model = _gen(_proj(_layered_wall(thickness=280)))
    assert_chain_matches_geometry(model, model.by_type("IfcWall")[0])
    layers = model.by_type("IfcMaterialLayerSet")[0].MaterialLayers
    assert sum(x.LayerThickness for x in layers) == pytest.approx(0.280)


def test_an_honest_buildup_still_reports_nominal_thicknesses():
    """The scale must be exactly 1.0 where the body matches its buildup — the
    normal case and every corpus model — so those files stay byte-identical."""
    model = _gen(_proj(_layered_wall(thickness=271)))
    layers = model.by_type("IfcMaterialLayerSet")[0].MaterialLayers
    assert [x.LayerThickness for x in layers] == [0.108, 0.15, 0.013]


def test_walls_of_different_thickness_do_not_share_one_usage():
    """The usage carries the datum, so two walls with the same buildup but
    different extents need DIFFERENT usages. Before the offset joined the cache
    key they collided and the second wall inherited the first one's datum."""
    thick = Wall(name="south_thick")
    thick.add(Box(name="body", start=Point(x=0, y=8000, z=0),
                  end=Point(x=6000, y=8000 + 331, z=2700)))
    thick.layers = _buildup()
    model = _gen(_proj(_layered_wall(), thick))
    for wall in model.by_type("IfcWall"):
        assert_chain_matches_geometry(model, wall)
    usages = model.by_type("IfcMaterialLayerSetUsage")
    assert len({u.OffsetFromReferenceLine for u in usages}) == 2, [
        u.OffsetFromReferenceLine for u in usages]


def test_the_derivation_check_actually_detects_the_bug_it_was_written_for():
    """Guard the guard. Re-run the assertion against the pre-fix datum and
    confirm it fails — a derivation test that cannot fail is decoration."""
    model = _gen(_proj(_layered_wall()))
    wall = model.by_type("IfcWall")[0]
    assert_chain_matches_geometry(model, wall)          # green as shipped
    model.by_type("IfcMaterialLayerSetUsage")[0].OffsetFromReferenceLine = 0.0
    with pytest.raises(AssertionError, match="material boundary"):
        assert_chain_matches_geometry(model, wall)


# -- the salt must never land in a layer thickness -----------------------


def test_the_anti_instancing_salt_never_lands_in_a_layer_thickness():
    """`unique_dims` nudges a profile dimension so two mirrored walls do not
    tessellate identically. That nudge is only harmless in a dimension nothing
    derives from — and on a wall sliced along local X, XDim is the layer
    THICKNESS. Salting it perturbed the buildup: measured in Blender, a 43 mm
    cavity read 42.98 mm on the short walls of a mitered ring."""
    model = _gen(_proj(*_ring()))
    thicknesses = set()
    for wall in model.by_type("IfcWall"):
        usage, _ = _chain_planes(model, wall)
        axis_i = _AXIS_I[usage.LayerSetDirection]
        for lo, hi in _drawn_spans(model, wall, axis_i):
            thicknesses.add(round(hi - lo, 9))
    # every leaf draws exactly the authored 108 / 190 — no leaf a hair thicker
    assert thicknesses == {0.108, 0.19}, sorted(thicknesses)


def test_mirrored_leaves_still_get_byte_distinct_profiles():
    """The salt still has to do its job: the west/east pair must not share
    geometry, or the SPA fragments worker instances them and drops one."""
    model = _gen(_proj(*_ring()))
    dims = {}
    for prof in model.by_type("IfcRectangleProfileDef"):
        dims.setdefault((prof.XDim, prof.YDim), set()).add(prof.ProfileName)
    shared = {k: v for k, v in dims.items() if len(v) > 1}
    assert not shared, f"walls sharing exact profile dims: {shared}"


def test_a_zero_salt_step_returns_instead_of_spinning():
    """Setting the step to 0 is the obvious way to test whether the salt is
    causing something. Without a guard the claim loop cannot advance and hangs
    forever — it read as a compiler hang for hours."""
    import ifcopenshell as ios

    from lite_step.ifc.entity_cache import EntityCache
    cache = EntityCache(ios.file(schema="IFC4"))
    cache._UNIQUE_DIMS_STEP = 0.0
    a = cache.get_or_create_rect_profile(1.0, 2.0, name="wall_a", unique_dims=True)
    b = cache.get_or_create_rect_profile(1.0, 2.0, name="wall_b", unique_dims=True)
    # identical dims (the salt is off) but distinct entities via ProfileName
    assert (a.XDim, a.YDim) == (b.XDim, b.YDim) == (1.0, 2.0)
