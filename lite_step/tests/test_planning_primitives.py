"""Tests for planning primitives: ReferencePoint, GuideLine, Space, Mesh.

Test-first: these tests define the expected IFC output for each primitive.
They will fail until the model classes and IFC generators are implemented.
"""

import pytest
import tempfile
import os

ifcopenshell = pytest.importorskip("ifcopenshell")
np = pytest.importorskip("numpy")

from lite_step.compiler.executor import execute_lite_step_script, normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc


PLANNING_SCRIPT = """\
from lite_step.models.project import Project, Storey
from lite_step.models.primitives import Point
from lite_step.models.elements import Box, Extrude, Mesh, ReferencePoint, GuideLine, Space

proj = Project(name="Planning Test")

# 1. ReferencePoint — survey marker
proj.add(ReferencePoint(
    id="survey_pt_1",
    name="survey_pt_1",
    location=Point(x=0, y=0, z=0),
    point_type="survey",
    accuracy_mm=15,
    survey_method="GNSS_RTK",
    crs="ETRS89/UTM32N",
))

# 2. GuideLine — property boundary
proj.add(GuideLine(
    id="boundary_north",
    name="boundary_north",
    points=[Point(x=0, y=20000, z=0), Point(x=30000, y=20000, z=0)],
    line_type="boundary",
    source="Lokalplan_2025",
    constraint="NoConstruction",
))

# 3. GuideLine — grid axes. TWO of them, crossing: an IfcGrid carries all the
#    axes of one grid and both UAxes and VAxes are LIST [1:?], so a grid whose
#    axes run in a single direction is not expressible and
#    lite_step.ifc.grids refuses it by name.
proj.add(GuideLine(
    id="grid_A",
    name="grid_a",
    points=[Point(x=0, y=0, z=0), Point(x=0, y=20000, z=0)],
    line_type="grid",
    label="A",
))
proj.add(GuideLine(
    id="grid_1",
    name="grid_1",
    points=[Point(x=0, y=0, z=0), Point(x=30000, y=0, z=0)],
    line_type="grid",
    label="1",
))

# 4. Space — kitchen zone
kitchen = Space(id="kitchen", name="kitchen", space_type="residential")
kitchen.add(Box(
    id="kitchen_vol",
    start=Point(x=0, y=0, z=0),
    end=Point(x=5000, y=4000, z=3000),
))
proj.add(kitchen)

# 5. Mesh — simple terrain
proj.add(Mesh(
    id="terrain_01",
    name="terrain_01",
    vertices=[
        Point(x=0, y=0, z=0),
        Point(x=10000, y=0, z=0),
        Point(x=10000, y=10000, z=500),
        Point(x=0, y=10000, z=200),
    ],
    faces=[(0, 1, 2), (0, 2, 3)],
    is_watertight=False,
    source_type="LiDAR",
))

result = proj
"""


def _get_psets(model, element):
    """Extract property sets from an IFC element as {pset_name: {prop: value}}."""
    psets = {}
    for rel in model.by_type("IfcRelDefinesByProperties"):
        if element in rel.RelatedObjects:
            pset = rel.RelatingPropertyDefinition
            if pset.is_a("IfcPropertySet"):
                props = {}
                for prop in pset.HasProperties:
                    if prop.is_a("IfcPropertySingleValue") and prop.NominalValue:
                        props[prop.Name] = prop.NominalValue.wrappedValue
                psets[pset.Name] = props
    return psets


def _ifc_model_from_result(ifc_result):
    """Write IFC content to temp file and open with ifcopenshell."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ifc", delete=False) as f:
        f.write(ifc_result.ifc_content)
        f.flush()
        model = ifcopenshell.open(f.name)
    os.unlink(f.name)
    return model


class TestPlanningPrimitivesCompile:
    """Step 0: verify the script compiles and normalizes."""

    def test_script_executes(self):
        result = execute_lite_step_script(PLANNING_SCRIPT)
        assert result.success, f"Script failed: {result.error}"
        assert result.project is not None
        assert result.project.name == "Planning Test"

    def test_normalization(self):
        result = execute_lite_step_script(PLANNING_SCRIPT)
        assert result.success
        normalized = normalize_project_to_meters(result.project)
        # All elements should still be present after normalization
        all_elems = normalized.get_all_elements()
        assert len(all_elems) == 6  # refpt, boundary, 2 grid axes, space, mesh


class TestPlanningPrimitivesIFC:
    """Step 1: verify IFC output matches expected entities."""

    @pytest.fixture(autouse=True)
    def setup(self):
        result = execute_lite_step_script(PLANNING_SCRIPT)
        assert result.success
        normalized = normalize_project_to_meters(result.project)
        ifc_result = generate_ifc(normalized)
        assert ifc_result.success, f"IFC generation failed: {ifc_result.error}"
        self.model = _ifc_model_from_result(ifc_result)

    # --- ReferencePoint ---

    def test_reference_point_entity(self):
        """ReferencePoint creates IfcAnnotation with ObjectType='ReferencePoint'."""
        annotations = self.model.by_type("IfcAnnotation")
        ref_pts = [a for a in annotations if a.ObjectType == "ReferencePoint"]
        assert len(ref_pts) == 1
        assert ref_pts[0].Name == "referencepoint:survey_pt_1"

    def test_reference_point_representation(self):
        """ReferencePoint has GeometricSet representation with a CartesianPoint."""
        annotations = self.model.by_type("IfcAnnotation")
        ref_pts = [a for a in annotations if a.ObjectType == "ReferencePoint"]
        assert len(ref_pts) == 1
        rep = ref_pts[0].Representation
        assert rep is not None

    def test_reference_point_properties(self):
        """ReferencePoint has Pset_SurveyPoint with survey metadata."""
        annotations = self.model.by_type("IfcAnnotation")
        ref_pts = [a for a in annotations if a.ObjectType == "ReferencePoint"]
        psets = _get_psets(self.model, ref_pts[0])
        assert "Pset_SurveyPoint" in psets
        survey = psets["Pset_SurveyPoint"]
        assert survey["SurveyMethod"] == "GNSS_RTK"
        assert survey["CRS"] == "ETRS89/UTM32N"
        assert survey["PointType"] == "survey"

    # --- GuideLine (boundary) ---

    def test_guideline_boundary_entity(self):
        """GuideLine(boundary) creates IfcAnnotation with ObjectType='PropertyBoundary'."""
        annotations = self.model.by_type("IfcAnnotation")
        boundaries = [a for a in annotations if a.ObjectType == "PropertyBoundary"]
        assert len(boundaries) == 1
        assert boundaries[0].Name == "guideline:boundary_north"

    def test_guideline_boundary_representation(self):
        """GuideLine(boundary) has Curve representation."""
        annotations = self.model.by_type("IfcAnnotation")
        boundaries = [a for a in annotations if a.ObjectType == "PropertyBoundary"]
        rep = boundaries[0].Representation.Representations[0]
        assert rep.RepresentationType in ("Curve", "GeometricCurveSet")

    def test_guideline_boundary_properties(self):
        """GuideLine(boundary) has Pset_GuideLine with Source and Constraint."""
        annotations = self.model.by_type("IfcAnnotation")
        boundaries = [a for a in annotations if a.ObjectType == "PropertyBoundary"]
        psets = _get_psets(self.model, boundaries[0])
        assert "Pset_GuideLine" in psets
        assert psets["Pset_GuideLine"]["Source"] == "Lokalplan_2025"
        assert psets["Pset_GuideLine"]["Constraint"] == "NoConstruction"

    # --- GuideLine (grid) ---

    def test_guideline_grid_entity(self):
        """Both grid lines land in ONE IfcGrid, partitioned by direction.

        "1" runs in X and "A" in Y, so "1" is the U axis whatever the labels
        suggest — the letter/number convention is drafting practice, not a
        fact about the model.
        """
        grids = self.model.by_type("IfcGrid")
        assert len(grids) == 1
        assert [a.AxisTag for a in grids[0].UAxes] == ["1"]
        assert [a.AxisTag for a in grids[0].VAxes] == ["A"]
        assert grids[0].WAxes is None
        assert grids[0].Name is None

    # --- Space ---

    def test_space_entity(self):
        """Space creates IfcSpace."""
        spaces = self.model.by_type("IfcSpace")
        assert len(spaces) == 1
        assert spaces[0].Name == "space:kitchen"

    def test_space_has_geometry(self):
        """Space has geometric representation from child Solid."""
        spaces = self.model.by_type("IfcSpace")
        assert spaces[0].Representation is not None
        reps = spaces[0].Representation.Representations
        assert len(reps) >= 1
        assert any(r.RepresentationType == "SweptSolid" for r in reps)

    def test_space_properties(self):
        """Space has Pset_SpacePlanning."""
        spaces = self.model.by_type("IfcSpace")
        psets = _get_psets(self.model, spaces[0])
        assert "Pset_SpacePlanning" in psets
        assert psets["Pset_SpacePlanning"]["SpaceType"] == "residential"

    # --- Mesh ---

    def test_mesh_entity(self):
        """Mesh creates entity with IfcTriangulatedFaceSet representation."""
        # Could be IfcBuildingElementProxy or IfcGeographicElement for terrain
        proxies = self.model.by_type("IfcBuildingElementProxy")
        mesh_elems = [p for p in proxies if p.Name == "mesh:terrain_01"]
        geo_elems = self.model.by_type("IfcGeographicElement")
        geo_mesh = [g for g in geo_elems if g.Name == "mesh:terrain_01"]
        terrain = mesh_elems + geo_mesh
        assert len(terrain) >= 1

    def test_mesh_tessellation(self):
        """Mesh has Tessellation representation with IfcTriangulatedFaceSet."""
        proxies = self.model.by_type("IfcBuildingElementProxy")
        mesh_elems = [p for p in proxies if p.Name == "mesh:terrain_01"]
        geo_elems = self.model.by_type("IfcGeographicElement")
        geo_mesh = [g for g in geo_elems if g.Name == "mesh:terrain_01"]
        terrain = mesh_elems + geo_mesh
        rep = terrain[0].Representation.Representations[0]
        assert rep.RepresentationType == "Tessellation"
        tri_set = rep.Items[0]
        assert tri_set.is_a("IfcTriangulatedFaceSet")
        assert len(tri_set.CoordIndex) == 2  # 2 triangles

    def test_mesh_indices_one_based(self):
        """IFC face indices are 1-based (converted from 0-based DSL)."""
        proxies = self.model.by_type("IfcBuildingElementProxy")
        mesh_elems = [p for p in proxies if p.Name == "mesh:terrain_01"]
        geo_elems = self.model.by_type("IfcGeographicElement")
        geo_mesh = [g for g in geo_elems if g.Name == "mesh:terrain_01"]
        terrain = mesh_elems + geo_mesh
        rep = terrain[0].Representation.Representations[0]
        tri_set = rep.Items[0]
        # (0,1,2) → (1,2,3) and (0,2,3) → (1,3,4) in 1-based
        assert tri_set.CoordIndex[0] == (1, 2, 3)
        assert tri_set.CoordIndex[1] == (1, 3, 4)

    def test_mesh_properties(self):
        """Mesh has LiteStep_MeshVolume with source and counts."""
        proxies = self.model.by_type("IfcBuildingElementProxy")
        mesh_elems = [p for p in proxies if p.Name == "mesh:terrain_01"]
        geo_elems = self.model.by_type("IfcGeographicElement")
        geo_mesh = [g for g in geo_elems if g.Name == "mesh:terrain_01"]
        terrain = mesh_elems + geo_mesh
        psets = _get_psets(self.model, terrain[0])
        assert "LiteStep_MeshVolume" in psets
        mv = psets["LiteStep_MeshVolume"]
        assert mv["SourceType"] == "LiDAR"
        assert mv["VertexCount"] == 4
        assert mv["FaceCount"] == 2
