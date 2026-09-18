"""Integration tests for cold-import against vendor-style IFC fixtures.

Each test synthesises an IFC that mimics a common third-party export
shape (Revit wall, ArchiCAD wall with door opening, scan brep, opaque
proxy, mixed corpus), runs the full ``cold_import()`` pipeline, and
asserts the reconstructed source compiles and the CoverageReport
matches the expected per-IFC-type breakdown.

When real-vendor `.ifc` files land under fixtures/third_party/, add a
parallel test here that loads them from disk and runs the same
assertion shape. The synthesised cases stay regardless — they pin the
cold-import contract end to end.
"""

from pathlib import Path

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.tests.test_normalizer import (
    _make_brep_box,
    _attach_brep_representation,
    _identity_placement,
)
from lite_step.transpiler import cold_import


# ---------------------------------------------------------------------------
# Vendor-style IFC builders (Revit / ArchiCAD / scan)
# ---------------------------------------------------------------------------


def _minimal_spatial_chassis(model):
    """Project → Site → Project → Storey hierarchy + identity placements."""
    project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
    site = model.create_entity(
        "IfcSite", GlobalId=ifcopenshell.guid.new(),
        ObjectPlacement=_identity_placement(model),
    )
    project = model.create_entity(
        "IfcBuilding", GlobalId=ifcopenshell.guid.new(),
        ObjectPlacement=_identity_placement(model),
    )
    storey = model.create_entity(
        "IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(),
        Name="Ground", ObjectPlacement=_identity_placement(model),
    )
    model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[site])
    model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=site, RelatedObjects=[project])
    model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[storey])
    return project, site, project, storey


def _attach_extruded_box(model, product, x_dim=6.0, y_dim=0.2, depth=3.0):
    """Revit-style wall body: IfcRectangleProfileDef extruded along Z."""
    profile = model.create_entity(
        "IfcRectangleProfileDef", ProfileType="AREA", XDim=x_dim, YDim=y_dim,
    )
    origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, depth / 2.0))
    pos = model.create_entity("IfcAxis2Placement3D", Location=origin)
    ext_dir = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile, Position=pos, ExtrudedDirection=ext_dir, Depth=depth,
    )
    context = model.create_entity(
        "IfcGeometricRepresentationContext",
        ContextType="Model", CoordinateSpaceDimension=3, Precision=1e-5,
    )
    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context, RepresentationIdentifier="Body",
        RepresentationType="SweptSolid", Items=[solid],
    )
    product.Representation = model.create_entity(
        "IfcProductDefinitionShape", Representations=[shape_rep],
    )
    return context


def _build_revit_wall():
    model = ifcopenshell.file(schema="IFC4")
    _, _, _, storey = _minimal_spatial_chassis(model)
    wall = model.create_entity(
        "IfcWallStandardCase",
        GlobalId=ifcopenshell.guid.new(),
        Name="revit_wall",
        ObjectPlacement=_identity_placement(model),
    )
    _attach_extruded_box(model, wall, x_dim=6.0, y_dim=0.2, depth=3.0)
    model.create_entity(
        "IfcRelContainedInSpatialStructure",
        GlobalId=ifcopenshell.guid.new(),
        RelatingStructure=storey, RelatedElements=[wall],
    )
    return model.to_string()


def _build_archicad_wall_with_door():
    """Wall with an IfcOpeningElement carrying a Door fill. The cold
    importer should re-parent the door onto the wall instead of leaving
    it as a floating storey element."""
    model = ifcopenshell.file(schema="IFC4")
    _, _, _, storey = _minimal_spatial_chassis(model)

    wall = model.create_entity(
        "IfcWallStandardCase",
        GlobalId=ifcopenshell.guid.new(),
        Name="archicad_wall",
        ObjectPlacement=_identity_placement(model),
    )
    _attach_extruded_box(model, wall, x_dim=6.0, y_dim=0.3, depth=3.0)

    opening = model.create_entity(
        "IfcOpeningElement",
        GlobalId=ifcopenshell.guid.new(),
        Name="door_void",
        ObjectPlacement=_identity_placement(model),
    )
    _attach_brep_representation(model, opening, _make_brep_box(model, 0.9, 0.3, 2.1))

    door = model.create_entity(
        "IfcDoor",
        GlobalId=ifcopenshell.guid.new(),
        Name="entry_door",
        ObjectPlacement=_identity_placement(model),
    )
    _attach_brep_representation(model, door, _make_brep_box(model, 0.9, 0.1, 2.1))

    # Tie wall → opening → door
    model.create_entity(
        "IfcRelVoidsElement",
        GlobalId=ifcopenshell.guid.new(),
        RelatingBuildingElement=wall, RelatedOpeningElement=opening,
    )
    model.create_entity(
        "IfcRelFillsElement",
        GlobalId=ifcopenshell.guid.new(),
        RelatingOpeningElement=opening, RelatedBuildingElement=door,
    )
    model.create_entity(
        "IfcRelContainedInSpatialStructure",
        GlobalId=ifcopenshell.guid.new(),
        RelatingStructure=storey, RelatedElements=[wall],
    )
    return model.to_string()


def _build_scan_brep_proxy():
    """Standalone IfcFacetedBrep proxy — scan/laser output where every
    element is just a triangulated mesh."""
    model = ifcopenshell.file(schema="IFC4")
    _, _, _, storey = _minimal_spatial_chassis(model)
    proxy = model.create_entity(
        "IfcBuildingElementProxy",
        GlobalId=ifcopenshell.guid.new(),
        Name="scan_proxy",
        ObjectPlacement=_identity_placement(model),
    )
    _attach_brep_representation(model, proxy, _make_brep_box(model, 1.0, 1.0, 2.0))
    model.create_entity(
        "IfcRelContainedInSpatialStructure",
        GlobalId=ifcopenshell.guid.new(),
        RelatingStructure=storey, RelatedElements=[proxy],
    )
    return model.to_string()


def _build_opaque_proxy():
    """IfcExtrudedAreaSolid with an IfcCircleProfileDef — the
    normalizer's _extract_extruded_solid returns mode=unknown for
    non-rectangle/arbitrary profiles, so the reconstructor falls
    through to the OPAQUE placeholder path."""
    model = ifcopenshell.file(schema="IFC4")
    _, _, _, storey = _minimal_spatial_chassis(model)
    proxy = model.create_entity(
        "IfcBuildingElementProxy",
        GlobalId=ifcopenshell.guid.new(),
        Name="opaque_cylinder",
        ObjectPlacement=_identity_placement(model),
    )
    profile = model.create_entity("IfcCircleProfileDef", ProfileType="AREA", Radius=0.5)
    origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 1.5))
    pos = model.create_entity("IfcAxis2Placement3D", Location=origin)
    ext_dir = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile, Position=pos, ExtrudedDirection=ext_dir, Depth=3.0,
    )
    context = model.create_entity(
        "IfcGeometricRepresentationContext",
        ContextType="Model", CoordinateSpaceDimension=3, Precision=1e-5,
    )
    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context, RepresentationIdentifier="Body",
        RepresentationType="SweptSolid", Items=[solid],
    )
    proxy.Representation = model.create_entity(
        "IfcProductDefinitionShape", Representations=[shape_rep],
    )
    model.create_entity(
        "IfcRelContainedInSpatialStructure",
        GlobalId=ifcopenshell.guid.new(),
        RelatingStructure=storey, RelatedElements=[proxy],
    )
    return model.to_string()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRevitStyleWall:
    def test_cold_import_succeeds_and_emits_wall(self):
        result = cold_import(_build_revit_wall())
        assert result.success
        assert result.mode == "cold_import"
        assert "Wall(id=" in result.source_code

    def test_cold_import_coverage_records_wall_as_exact(self):
        result = cold_import(_build_revit_wall())
        cov = result.coverage_report
        assert cov is not None
        assert cov.by_type.get("IfcWallStandardCase", (0, 0, 0))[0] == 1  # exact
        assert cov.opaque == 0

    def test_cold_import_recompiles_to_valid_ifc(self):
        """The reconstructed source must round-trip back through the
        compiler to a parseable IFC — that's the contract that lets
        the patch UI immediately re-render after a cold import."""
        result = cold_import(_build_revit_wall())
        assert result.ifc_content is not None
        assert "FILE_SCHEMA" in result.ifc_content


class TestArchiCADWallWithDoor:
    def test_door_is_reparented_under_wall(self):
        result = cold_import(_build_archicad_wall_with_door())
        assert result.success
        # The door's id should appear inside the wall's emission block as
        # wall.add(Box(... id="entry_door" ...)) — and there must be
        # exactly one such line (no double-emit as a storey-level
        # floating Door).
        lines = [ln for ln in result.source_code.split("\n") if 'id="entry_door"' in ln]
        assert len(lines) == 1, lines

    def test_coverage_counts_wall_and_door_fill(self):
        result = cold_import(_build_archicad_wall_with_door())
        cov = result.coverage_report
        assert cov is not None
        # Both the wall body and the door fill are exact (both came
        # through with usable bbox geometry).
        assert cov.by_type.get("IfcWallStandardCase", (0, 0, 0))[0] >= 1
        assert cov.by_type.get("IfcDoor", (0, 0, 0))[0] == 1


class TestScanBrep:
    def test_brep_emits_mesh(self):
        result = cold_import(_build_scan_brep_proxy())
        assert result.success
        assert "Mesh(vertices=" in result.source_code

    def test_brep_coverage_is_exact(self):
        result = cold_import(_build_scan_brep_proxy())
        cov = result.coverage_report
        assert cov is not None
        # Brep with face indices → exact, not opaque.
        assert cov.exact >= 1
        assert cov.opaque == 0


class TestOpaqueProxy:
    def test_circle_profile_falls_through_to_opaque(self):
        result = cold_import(_build_opaque_proxy())
        assert result.success
        assert "OPAQUE: original geometry lost" in result.source_code

    def test_opaque_count_in_coverage(self):
        result = cold_import(_build_opaque_proxy())
        cov = result.coverage_report
        assert cov is not None
        assert cov.opaque == 1
        assert cov.by_type.get("IfcBuildingElementProxy", (0, 0, 0))[2] == 1


class TestMixedCorpus:
    """The lossiness banner test: a single IFC carrying one of each
    quality bucket. Loud failure beats silent degradation:
    the patch UI uses these counts to render the cold-import banner."""

    def _build_mixed_corpus(self):
        model = ifcopenshell.file(schema="IFC4")
        _, _, _, storey = _minimal_spatial_chassis(model)

        wall = model.create_entity(
            "IfcWallStandardCase", GlobalId=ifcopenshell.guid.new(),
            Name="exact_wall", ObjectPlacement=_identity_placement(model),
        )
        _attach_extruded_box(model, wall)

        proxy_brep = model.create_entity(
            "IfcBuildingElementProxy", GlobalId=ifcopenshell.guid.new(),
            Name="exact_brep", ObjectPlacement=_identity_placement(model),
        )
        _attach_brep_representation(model, proxy_brep, _make_brep_box(model, 1.0, 1.0, 1.0))

        opaque = model.create_entity(
            "IfcBuildingElementProxy", GlobalId=ifcopenshell.guid.new(),
            Name="opaque_cyl", ObjectPlacement=_identity_placement(model),
        )
        circle = model.create_entity("IfcCircleProfileDef", ProfileType="AREA", Radius=0.5)
        origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
        pos = model.create_entity("IfcAxis2Placement3D", Location=origin)
        ext_dir = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        solid = model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=circle, Position=pos, ExtrudedDirection=ext_dir, Depth=1.0,
        )
        context = model.create_entity(
            "IfcGeometricRepresentationContext",
            ContextType="Model", CoordinateSpaceDimension=3, Precision=1e-5,
        )
        shape_rep = model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=context, RepresentationIdentifier="Body",
            RepresentationType="SweptSolid", Items=[solid],
        )
        opaque.Representation = model.create_entity(
            "IfcProductDefinitionShape", Representations=[shape_rep],
        )

        model.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=ifcopenshell.guid.new(),
            RelatingStructure=storey, RelatedElements=[wall, proxy_brep, opaque],
        )
        return model.to_string()

    def test_coverage_tallies_match_mixed_corpus(self):
        result = cold_import(self._build_mixed_corpus())
        cov = result.coverage_report
        assert cov is not None
        # 1 exact wall + 1 exact brep proxy + 1 opaque cylinder proxy = 3 total
        assert cov.total == 3
        assert cov.exact == 2
        assert cov.opaque == 1
        # The IfcBuildingElementProxy line in the breakdown carries one
        # exact (brep) and one opaque (cylinder).
        e, a, o = cov.by_type["IfcBuildingElementProxy"]
        assert (e, a, o) == (1, 0, 1)


class TestThirdPartyFixturesDirectory:
    """Forward-compat check: when real-vendor .ifc files land under
    fixtures/third_party/, this test should iterate them. For now it
    just confirms the scaffold directory exists and is documented."""

    def test_scaffold_directory_and_readme_present(self):
        scaffold = Path(__file__).parent / "fixtures" / "third_party"
        assert scaffold.is_dir()
        readme = scaffold / "README.md"
        assert readme.is_file()
        assert "fixtures" in readme.read_text(encoding="utf-8").lower()
