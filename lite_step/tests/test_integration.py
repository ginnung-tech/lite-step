"""Integration tests for full round-trip and cold import flows.

These tests require ifcopenshell and exercise the full pipeline:
generate IFC -> modify -> round-trip -> verify.
"""

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
np = pytest.importorskip("numpy")

from lite_step.compiler.executor import execute_lite_step_script
from lite_step.ifc.generator import generate_ifc
from lite_step.ifc.embedder import extract_litestep_meta
from lite_step.ifc.normalizer import normalize_ifc


CARPORT_SOURCE = """\
from lite_step.models.project import Project, Storey
from lite_step.models.primitives import Point
from lite_step.models.elements import Wall, Box, Extrude, Column, Roof

def generate_project():
    proj = Project(name="Carport")
    storey = Storey(elevation=0)
    proj.add_storey(storey)

    # Four columns
    for i in range(4):
        x = (i % 2) * 6000
        z = (i // 2) * 5000
        col = Column(name=f"col_{i}")
        col.add(Box(
            start=Point(x=x, y=0, z=z),
            end=Point(x=x + 300, y=3000, z=z + 300),
            type="sketch",
        ))
        storey.add(col)

    # Roof
    roof = Roof(name="roof_main")
    roof.add(Box(
        start=Point(x=-200, y=2800, z=-200),
        end=Point(x=6200, y=3100, z=5200),
        type="sketch",
    ))
    storey.add(roof)

    return proj

result = generate_project()
"""


class TestEmbeddingRoundTrip:
    def test_generate_with_embedding(self):
        """Generate IFC with source embedding and verify metadata exists."""
        exec_result = execute_lite_step_script(CARPORT_SOURCE)
        assert exec_result.success
        assert exec_result.project is not None

        ifc_result = generate_ifc(exec_result.project, source_code=CARPORT_SOURCE)
        assert ifc_result.success
        assert ifc_result.ifc_content is not None

        # Extract metadata
        meta = extract_litestep_meta(ifc_result.ifc_content)
        assert meta is not None
        assert meta.source == CARPORT_SOURCE
        assert meta.hash_valid is True
        assert meta.version == "18"  # .no_carve() deleted

    def test_manifest_has_elements(self):
        """Manifest should contain all named elements."""
        exec_result = execute_lite_step_script(CARPORT_SOURCE)
        ifc_result = generate_ifc(exec_result.project, source_code=CARPORT_SOURCE)

        meta = extract_litestep_meta(ifc_result.ifc_content)
        assert meta is not None
        assert len(meta.manifest) > 0
        # Should contain at least the columns and roof
        named_keys = set(meta.manifest.keys())
        assert "roof:roof_main" in named_keys

    def test_generate_without_embedding(self):
        """Generate without source_code should produce no LITESTEP_META."""
        exec_result = execute_lite_step_script(CARPORT_SOURCE)
        ifc_result = generate_ifc(exec_result.project)  # No source_code
        assert ifc_result.success

        meta = extract_litestep_meta(ifc_result.ifc_content)
        assert meta is None


class TestNormalizationRoundTrip:
    def test_normalize_carport(self):
        """Normalize a generated IFC and verify elements are extracted."""
        exec_result = execute_lite_step_script(CARPORT_SOURCE)
        # Normalizer relies on ifcopenshell placement conventions — use legacy backend
        ifc_result = generate_ifc(exec_result.project, source_code=CARPORT_SOURCE)

        elements = normalize_ifc(ifc_result.ifc_content)
        assert len(elements) > 0

    def test_element_names_preserved(self):
        """IFC Name fields (= element IDs) should survive normalization."""
        exec_result = execute_lite_step_script(CARPORT_SOURCE)
        ifc_result = generate_ifc(exec_result.project, source_code=CARPORT_SOURCE)

        elements = normalize_ifc(ifc_result.ifc_content)
        names = {e.name for e in elements}
        # At least some column and roof elements should be present
        assert any("col_" in n for n in names) or any("roof" in n for n in names)


class TestColdImportFlow:
    def test_cold_import_from_generated(self):
        """Generate IFC, strip metadata, cold import -> compiles."""
        exec_result = execute_lite_step_script(CARPORT_SOURCE)
        ifc_result = generate_ifc(exec_result.project)  # No embedding

        from lite_step.transpiler import cold_import
        result = cold_import(ifc_result.ifc_content)

        assert result.success
        assert result.source_code is not None
        assert result.mode == "cold_import"

        # Source should be valid Python
        compile(result.source_code, "<cold_import>", "exec")

    def test_cold_import_has_header(self):
        """Cold import output should have the reconstruction header."""
        exec_result = execute_lite_step_script(CARPORT_SOURCE)
        ifc_result = generate_ifc(exec_result.project)

        from lite_step.transpiler import cold_import
        result = cold_import(ifc_result.ifc_content)

        assert result.source_code is not None
        assert "Reconstructed from IFC" in result.source_code


class TestGeoreferencing:
    """IFC georeferencing — IfcMapConversion + IfcProjectedCRS."""

    def test_no_georef_without_site_coords(self):
        """Project without site coords should produce no georef entities."""
        from lite_step.models.project import Project, Storey
        from lite_step.models.primitives import Point
        from lite_step.models.elements import Mesh

        b = Project(name="Plain")
        s = Storey(elevation=0)
        s.add(Mesh(id="m", name="m", material="#fff",
                   is_watertight=True,
                   vertices=[Point(x=0, y=0, z=0), Point(x=1, y=0, z=0),
                             Point(x=0, y=1, z=0), Point(x=0, y=0, z=1)],
                   faces=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]))
        b.add_storey(s)

        result = generate_ifc(b)
        assert result.success
        model = ifcopenshell.file.from_string(result.ifc_content)
        assert len(model.by_type("IfcMapConversion")) == 0
        assert len(model.by_type("IfcProjectedCRS")) == 0

    def test_georef_with_site_coords(self):
        """Project with site coords should produce IfcMapConversion + IfcProjectedCRS."""
        from lite_step.models.project import Project, Storey
        from lite_step.models.primitives import Point
        from lite_step.models.elements import Mesh

        # Copenhagen area: lat 55.67, lng 12.57 → UTM zone 33N (EPSG:32633)
        b = Project(
            name="Georef Test",
            site_latitude=55.67,
            site_longitude=12.57,
            site_elevation=5.0,
            site_true_north=0.0,
        )
        s = Storey(elevation=0)
        s.add(Mesh(id="m", name="m", material="#fff",
                   is_watertight=True,
                   vertices=[Point(x=0, y=0, z=0), Point(x=1, y=0, z=0),
                             Point(x=0, y=1, z=0), Point(x=0, y=0, z=1)],
                   faces=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]))
        b.add_storey(s)

        result = generate_ifc(b)
        assert result.success
        model = ifcopenshell.file.from_string(result.ifc_content)

        # IfcProjectedCRS should exist with correct EPSG
        crs_list = model.by_type("IfcProjectedCRS")
        assert len(crs_list) == 1
        crs = crs_list[0]
        assert "EPSG:32633" in crs.Name

        # IfcMapConversion should exist with plausible UTM coordinates
        mc_list = model.by_type("IfcMapConversion")
        assert len(mc_list) == 1
        mc = mc_list[0]
        assert mc.Eastings is not None
        assert mc.Northings is not None
        assert mc.OrthogonalHeight == 5.0
        assert mc.Scale == 1.0
        # UTM easting should be near 500,000 (central meridian offset)
        assert 300_000 < mc.Eastings < 700_000
        # UTM northing for lat ~55.67 should be in the millions
        assert mc.Northings > 1_000_000

    def test_georef_true_north_rotation(self):
        """Non-zero true_north should produce rotated XAxisAbscissa/XAxisOrdinate."""
        from lite_step.models.project import Project, Storey
        from lite_step.models.primitives import Point
        from lite_step.models.elements import Mesh
        import math

        b = Project(
            name="Rotated",
            site_latitude=55.67,
            site_longitude=12.57,
            site_elevation=0.0,
            site_true_north=30.0,  # 30° rotation
        )
        s = Storey(elevation=0)
        s.add(Mesh(id="m", name="m", material="#fff",
                   is_watertight=True,
                   vertices=[Point(x=0, y=0, z=0), Point(x=1, y=0, z=0),
                             Point(x=0, y=1, z=0), Point(x=0, y=0, z=1)],
                   faces=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]))
        b.add_storey(s)

        result = generate_ifc(b)
        assert result.success
        model = ifcopenshell.file.from_string(result.ifc_content)

        mc = model.by_type("IfcMapConversion")[0]
        assert abs(mc.XAxisAbscissa - math.cos(math.radians(30))) < 0.001
        assert abs(mc.XAxisOrdinate - math.sin(math.radians(30))) < 0.001


class TestMarkerPegs:
    """Visible 3D marker pegs for boundary/foundation points."""

    def test_boundary_marker_creates_proxy(self):
        """Boundary marker should produce IfcBuildingElementProxy, not IfcAnnotation."""
        from lite_step.models.project import Project, Storey
        from lite_step.models.primitives import Point
        from lite_step.models.elements import ReferencePoint

        b = Project(name="Markers")
        s = Storey(elevation=0)
        s.add(ReferencePoint(
            id="bnd_sw",
            name="bnd_sw",
            location=Point(x=0, y=0, z=0),
            point_type="boundary",
            marker_color="#f32ca5",
        ))
        s.add(ReferencePoint(
            id="fnd_0",
            name="fnd_0",
            location=Point(x=5000, y=0, z=3000),
            point_type="foundation",
        ))
        b.add_storey(s)

        result = generate_ifc(b)
        assert result.success
        model = ifcopenshell.file.from_string(result.ifc_content)

        # Should be IfcBuildingElementProxy (visible), not IfcAnnotation
        proxies = model.by_type("IfcBuildingElementProxy")
        assert len(proxies) >= 2
        names = {p.Name for p in proxies}
        assert "referencepoint:bnd_sw" in names
        assert "referencepoint:fnd_0" in names

        # No IfcAnnotation for these (they're visible pegs)
        annotations = model.by_type("IfcAnnotation")
        ann_names = {a.Name for a in annotations}
        assert "referencepoint:bnd_sw" not in ann_names
        assert "referencepoint:fnd_0" not in ann_names

    def test_survey_marker_stays_annotation(self):
        """Survey/datum markers should still be invisible IfcAnnotation."""
        from lite_step.models.project import Project, Storey
        from lite_step.models.primitives import Point
        from lite_step.models.elements import ReferencePoint

        b = Project(name="Survey")
        s = Storey(elevation=0)
        s.add(ReferencePoint(
            id="sp_datum",
            name="sp_datum",
            location=Point(x=0, y=0, z=0),
            point_type="datum",
        ))
        b.add_storey(s)

        result = generate_ifc(b)
        assert result.success
        model = ifcopenshell.file.from_string(result.ifc_content)

        annotations = model.by_type("IfcAnnotation")
        assert any(a.Name == "referencepoint:sp_datum" for a in annotations)
