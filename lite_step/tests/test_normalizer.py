"""Tests for IFC normalization (IFC -> ExtractedElement list)."""

import math
import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
np = pytest.importorskip("numpy")

from lite_step.ifc.normalizer import (
    _ifc_to_dsl_point,
    _decompose_rotation,
    _resolve_placement,
    normalize_ifc,
)


class TestCoordinateTransform:
    def test_ifc_to_dsl_point_basic(self):
        """IFC Z-up meters -> DSL Z-up mm: multiply by 1000, no axis swap."""
        result = _ifc_to_dsl_point(1.0, 2.0, 3.0)
        assert result == (1000, 2000, 3000)

    def test_ifc_to_dsl_point_origin(self):
        result = _ifc_to_dsl_point(0.0, 0.0, 0.0)
        assert result == (0, 0, 0)

    def test_ifc_to_dsl_point_snapping(self):
        """Float noise should snap to nearest integer mm."""
        result = _ifc_to_dsl_point(1.0005, 2.0003, 3.0007)
        assert result == (1000, 2000, 3001)  # Python banker's rounding: 1000.5 → 1000

    def test_dsl_to_ifc_inverse(self):
        """Verify our transform is the inverse of generator's dsl_to_ifc_point."""
        from lite_step.ifc.geometry import dsl_to_ifc_point

        # DSL point: (5000, 3000, 2000) mm
        dsl_x, dsl_y, dsl_z = 5000, 3000, 2000

        # Forward: DSL -> IFC (in meters, after executor normalization)
        ifc_coords = dsl_to_ifc_point(dsl_x / 1000, dsl_y / 1000, dsl_z / 1000)

        # Inverse: IFC -> DSL
        back = _ifc_to_dsl_point(ifc_coords[0], ifc_coords[1], ifc_coords[2])
        assert back == (dsl_x, dsl_y, dsl_z)


class TestRotationDecomposition:
    def test_identity_rotation(self):
        R = np.eye(3)
        rx, ry, rz = _decompose_rotation(R)
        assert abs(rx) < 100  # < 0.1 degree * 1000
        assert abs(ry) < 100
        assert abs(rz) < 100

    def test_90_degree_z_rotation(self):
        """90° around IFC Z axis -> 90° around DSL Z axis (identity mapping)."""
        angle = math.pi / 2
        R = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ])
        rx, ry, rz = _decompose_rotation(R)
        # IFC rz=90° should map directly to DSL rz=90° (90000 in *1000 convention)
        assert abs(rz - 90000) < 200  # within 0.2 degree


class TestPlacementResolution:
    def test_identity_placement(self):
        """Product with no placement -> identity matrix."""
        model = ifcopenshell.file(schema="IFC4")
        product = model.create_entity(
            "IfcBuildingElementProxy",
            GlobalId=ifcopenshell.guid.new(),
        )
        mat = _resolve_placement(product)
        assert np.allclose(mat, np.eye(4))

    def test_simple_translation(self):
        """Product placed at (1, 2, 3) meters in IFC."""
        model = ifcopenshell.file(schema="IFC4")
        point = model.create_entity("IfcCartesianPoint", Coordinates=(1.0, 2.0, 3.0))
        ax = model.create_entity("IfcAxis2Placement3D", Location=point)
        lp = model.create_entity("IfcLocalPlacement", RelativePlacement=ax)

        product = model.create_entity(
            "IfcBuildingElementProxy",
            GlobalId=ifcopenshell.guid.new(),
            ObjectPlacement=lp,
        )
        mat = _resolve_placement(product)
        assert abs(mat[0, 3] - 1.0) < 1e-6
        assert abs(mat[1, 3] - 2.0) < 1e-6
        assert abs(mat[2, 3] - 3.0) < 1e-6


class TestNormalizeIfc:
    def _create_ifc_with_wall(self):
        """Create a minimal IFC with one wall product for testing."""
        model = ifcopenshell.file(schema="IFC4")

        project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
        site = model.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new())
        building = model.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new())

        # Set placements
        for entity in (site, building):
            origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
            ax = model.create_entity("IfcAxis2Placement3D", Location=origin)
            lp = model.create_entity("IfcLocalPlacement", RelativePlacement=ax)
            entity.ObjectPlacement = lp

        # Create storey at elevation 0
        storey = model.create_entity(
            "IfcBuildingStorey",
            GlobalId=ifcopenshell.guid.new(),
            Name="Ground Floor",
        )
        origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
        ax = model.create_entity("IfcAxis2Placement3D", Location=origin)
        lp = model.create_entity("IfcLocalPlacement", RelativePlacement=ax)
        storey.ObjectPlacement = lp

        # Aggregate hierarchy
        model.create_entity(
            "IfcRelAggregates",
            GlobalId=ifcopenshell.guid.new(),
            RelatingObject=project,
            RelatedObjects=[site],
        )
        model.create_entity(
            "IfcRelAggregates",
            GlobalId=ifcopenshell.guid.new(),
            RelatingObject=site,
            RelatedObjects=[building],
        )
        model.create_entity(
            "IfcRelAggregates",
            GlobalId=ifcopenshell.guid.new(),
            RelatingObject=building,
            RelatedObjects=[storey],
        )

        # Create a wall placed at (3, 1, 1.5) meters in IFC
        wall_point = model.create_entity("IfcCartesianPoint", Coordinates=(3.0, 1.0, 1.5))
        wall_ax = model.create_entity("IfcAxis2Placement3D", Location=wall_point)
        wall_lp = model.create_entity("IfcLocalPlacement", RelativePlacement=wall_ax)

        wall = model.create_entity(
            "IfcWall",
            GlobalId=ifcopenshell.guid.new(),
            Name="wall_north",
            ObjectPlacement=wall_lp,
        )

        # Create box geometry: 6m x 0.2m x 3m (width x depth x height)
        profile = model.create_entity(
            "IfcRectangleProfileDef",
            ProfileType="AREA",
            XDim=6.0,
            YDim=0.2,
        )
        profile_origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, -1.5))
        profile_ax = model.create_entity("IfcAxis2Placement3D", Location=profile_origin)
        ext_dir = model.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))

        solid = model.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile,
            Position=profile_ax,
            ExtrudedDirection=ext_dir,
            Depth=3.0,
        )

        context = model.create_entity(
            "IfcGeometricRepresentationContext",
            ContextType="Model",
            CoordinateSpaceDimension=3,
            Precision=1e-5,
        )
        shape_rep = model.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=context,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=[solid],
        )
        product_shape = model.create_entity(
            "IfcProductDefinitionShape",
            Representations=[shape_rep],
        )
        wall.Representation = product_shape

        # Contain wall in storey
        model.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=ifcopenshell.guid.new(),
            RelatingStructure=storey,
            RelatedElements=[wall],
        )

        return model.to_string()

    def test_normalize_extracts_elements(self):
        ifc_content = self._create_ifc_with_wall()
        elements = normalize_ifc(ifc_content)
        assert len(elements) >= 1

    def test_normalize_wall_name(self):
        ifc_content = self._create_ifc_with_wall()
        elements = normalize_ifc(ifc_content)
        wall = next((e for e in elements if e.name == "wall_north"), None)
        assert wall is not None
        assert wall.ifc_type == "IfcWall"

    def test_normalize_wall_geometry_is_box(self):
        """The mode AND the numbers.

        This asserted ``mode == "box"`` and that ``start``/``end`` were not
        ``None``, which is why the extrusion-axis offset shipped
        invisibly: the fixture's 6.0 x 0.2 x 3.0 m wall stands at world
        ``z 0..3000`` and came back at ``-1500..1500``, mode intact, both
        endpoints non-``None``.
        """
        ifc_content = self._create_ifc_with_wall()
        elements = normalize_ifc(ifc_content)
        wall = next((e for e in elements if e.name == "wall_north"), None)
        assert wall is not None
        assert wall.geometry.mode == "box"

        # Placed at (3, 1, 1.5) m, profile 6.0 x 0.2 offset z -1.5, extruded 3.0.
        assert wall.geometry.start == (0, 900, 0)
        assert wall.geometry.end == (6000, 1100, 3000)

    def test_storey_ordering(self):
        """Multiple storeys should be indexed by elevation order."""
        model = ifcopenshell.file(schema="IFC4")
        project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
        site = model.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new())
        building = model.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new())

        for entity in (site, building):
            origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
            ax = model.create_entity("IfcAxis2Placement3D", Location=origin)
            lp = model.create_entity("IfcLocalPlacement", RelativePlacement=ax)
            entity.ObjectPlacement = lp

        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[site])
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=site, RelatedObjects=[building])

        # Create storeys at different elevations (out of order)
        storeys = []
        for elev in [3.0, 0.0, 6.0]:
            s = model.create_entity("IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(), Name=f"Level_{elev}")
            pt = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, elev))
            ax = model.create_entity("IfcAxis2Placement3D", Location=pt)
            lp = model.create_entity("IfcLocalPlacement", RelativePlacement=ax)
            s.ObjectPlacement = lp
            storeys.append(s)

        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=building, RelatedObjects=storeys)

        ifc_content = model.to_string()
        elements = normalize_ifc(ifc_content)
        # No products, so no elements — but the storey ordering is tested internally
        assert isinstance(elements, list)


def _make_brep_box(model, dx: float, dy: float, dz: float):
    """Build a closed axis-aligned IfcFacetedBrep at the origin spanning
    (0,0,0) to (dx,dy,dz). Returns the IfcFacetedBrep entity."""
    # 8 corners
    cs = [
        (0.0, 0.0, 0.0), (dx, 0.0, 0.0), (dx, dy, 0.0), (0.0, dy, 0.0),
        (0.0, 0.0, dz), (dx, 0.0, dz), (dx, dy, dz), (0.0, dy, dz),
    ]
    pts = [model.create_entity("IfcCartesianPoint", Coordinates=c) for c in cs]
    # 6 quad faces (bottom, top, four sides) — winding chosen for outward normals
    quads = [
        (0, 3, 2, 1),  # bottom
        (4, 5, 6, 7),  # top
        (0, 1, 5, 4),  # +X side
        (1, 2, 6, 5),  # +Y side
        (2, 3, 7, 6),  # -X side
        (3, 0, 4, 7),  # -Y side
    ]
    faces = []
    for quad in quads:
        loop = model.create_entity("IfcPolyLoop", Polygon=[pts[i] for i in quad])
        bound = model.create_entity("IfcFaceOuterBound", Bound=loop, Orientation=True)
        faces.append(model.create_entity("IfcFace", Bounds=[bound]))
    shell = model.create_entity("IfcClosedShell", CfsFaces=faces)
    return model.create_entity("IfcFacetedBrep", Outer=shell)


def _attach_brep_representation(model, product, brep, context=None):
    """Attach a Body/Brep IfcShapeRepresentation built from `brep` to
    `product`. Reuses `context` when given so multiple products share
    the same IfcGeometricRepresentationContext."""
    if context is None:
        context = model.create_entity(
            "IfcGeometricRepresentationContext",
            ContextType="Model",
            CoordinateSpaceDimension=3,
            Precision=1e-5,
        )
    shape_rep = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=context,
        RepresentationIdentifier="Body",
        RepresentationType="Brep",
        Items=[brep],
    )
    product.Representation = model.create_entity(
        "IfcProductDefinitionShape",
        Representations=[shape_rep],
    )
    return context


def _identity_placement(model):
    """Return a fresh IfcLocalPlacement at the world origin."""
    origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    ax = model.create_entity("IfcAxis2Placement3D", Location=origin)
    return model.create_entity("IfcLocalPlacement", RelativePlacement=ax)


class TestBrepMeshExtraction:
    """IfcFacetedBrep extraction returns full mesh_vertices + mesh_faces,
    not just the bbox fallback. The reconstructor relies on these to emit
    DSL Mesh elements for cold-imported third-party IFCs."""

    def _build_ifc_with_brep_wall(self):
        model = ifcopenshell.file(schema="IFC4")
        project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
        site = model.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new())
        building = model.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new())
        storey = model.create_entity(
            "IfcBuildingStorey",
            GlobalId=ifcopenshell.guid.new(),
            Name="Ground",
        )
        for entity in (site, building, storey):
            entity.ObjectPlacement = _identity_placement(model)
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[site])
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=site, RelatedObjects=[building])
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=building, RelatedObjects=[storey])

        wall = model.create_entity(
            "IfcWall",
            GlobalId=ifcopenshell.guid.new(),
            Name="brep_wall",
            ObjectPlacement=_identity_placement(model),
        )
        brep = _make_brep_box(model, dx=2.0, dy=0.3, dz=3.0)
        _attach_brep_representation(model, wall, brep)

        model.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=ifcopenshell.guid.new(),
            RelatingStructure=storey,
            RelatedElements=[wall],
        )
        return model.to_string()

    def test_brep_wall_returns_mesh_geometry(self):
        elements = normalize_ifc(self._build_ifc_with_brep_wall())
        wall = next((e for e in elements if e.name == "brep_wall"), None)
        assert wall is not None
        assert wall.geometry.mode == "mesh"

    def test_brep_wall_populates_mesh_vertices_and_faces(self):
        elements = normalize_ifc(self._build_ifc_with_brep_wall())
        wall = next((e for e in elements if e.name == "brep_wall"), None)
        assert wall is not None
        # 8 unique vertices for a closed box
        assert wall.geometry.mesh_vertices is not None
        assert len(wall.geometry.mesh_vertices) == 8
        # 6 quad faces fan-triangulated = 12 triangles
        assert wall.geometry.mesh_faces is not None
        assert len(wall.geometry.mesh_faces) == 12
        # Every triangle index is in range
        for tri in wall.geometry.mesh_faces:
            assert all(0 <= i < 8 for i in tri)

    def test_brep_wall_bbox_still_populated(self):
        elements = normalize_ifc(self._build_ifc_with_brep_wall())
        wall = next((e for e in elements if e.name == "brep_wall"), None)
        assert wall is not None
        # Bbox is the cold-import fallback for renderers that don't
        # understand mesh — must stay populated alongside the mesh.
        assert wall.geometry.start == (0, 0, 0)
        assert wall.geometry.end == (2000, 300, 3000)


class TestSiteTopography:
    """IfcSite with a Body Representation (terrain mesh) is lifted into
    the elements list with storey_idx=-1 as the site-level sentinel."""

    def _build_ifc_with_site_terrain(self):
        model = ifcopenshell.file(schema="IFC4")
        project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
        site = model.create_entity(
            "IfcSite",
            GlobalId=ifcopenshell.guid.new(),
            Name="terrain_site",
            ObjectPlacement=_identity_placement(model),
        )
        building = model.create_entity(
            "IfcBuilding",
            GlobalId=ifcopenshell.guid.new(),
            ObjectPlacement=_identity_placement(model),
        )
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[site])
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=site, RelatedObjects=[building])
        brep = _make_brep_box(model, dx=10.0, dy=10.0, dz=0.5)  # 10m x 10m terrain pad
        _attach_brep_representation(model, site, brep)
        return model.to_string()

    def test_site_with_geometry_is_extracted(self):
        elements = normalize_ifc(self._build_ifc_with_site_terrain())
        site = next((e for e in elements if e.ifc_type == "IfcSite"), None)
        assert site is not None
        assert site.name == "terrain_site"

    def test_site_storey_idx_is_minus_one(self):
        elements = normalize_ifc(self._build_ifc_with_site_terrain())
        site = next((e for e in elements if e.ifc_type == "IfcSite"), None)
        assert site is not None
        assert site.storey_idx == -1, "Site uses -1 as the site-level sentinel"

    def test_bare_site_without_representation_is_not_extracted(self):
        """The default test_normalize_extracts_elements site has no
        Representation; it must not pollute the elements list."""
        model = ifcopenshell.file(schema="IFC4")
        project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
        site = model.create_entity(
            "IfcSite",
            GlobalId=ifcopenshell.guid.new(),
            Name="bare_site",
            ObjectPlacement=_identity_placement(model),
        )
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[site])
        elements = normalize_ifc(model.to_string())
        assert not any(e.ifc_type == "IfcSite" for e in elements)


class TestSpaceExtraction:
    """IfcSpace with a Body Representation (room volume) is lifted into
    the elements list and inherits its parent storey via Decomposes."""

    def _build_ifc_with_space(self):
        model = ifcopenshell.file(schema="IFC4")
        project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
        site = model.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new(), ObjectPlacement=_identity_placement(model))
        building = model.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new(), ObjectPlacement=_identity_placement(model))
        # Two storeys at elevations 0 and 3m so storey_idx mapping is meaningful
        s0 = model.create_entity(
            "IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(), Name="Ground",
            ObjectPlacement=_identity_placement(model),
        )
        s1_pt = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 3.0))
        s1_ax = model.create_entity("IfcAxis2Placement3D", Location=s1_pt)
        s1_lp = model.create_entity("IfcLocalPlacement", RelativePlacement=s1_ax)
        s1 = model.create_entity(
            "IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(), Name="Upper",
            ObjectPlacement=s1_lp,
        )
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[site])
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=site, RelatedObjects=[building])
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=building, RelatedObjects=[s0, s1])

        # Space on the upper storey (storey_idx should be 1)
        space = model.create_entity(
            "IfcSpace",
            GlobalId=ifcopenshell.guid.new(),
            Name="upper_room",
            ObjectPlacement=_identity_placement(model),
        )
        brep = _make_brep_box(model, dx=4.0, dy=3.0, dz=2.7)
        _attach_brep_representation(model, space, brep)
        model.create_entity(
            "IfcRelAggregates",
            GlobalId=ifcopenshell.guid.new(),
            RelatingObject=s1,
            RelatedObjects=[space],
        )
        return model.to_string()

    def test_space_with_geometry_is_extracted(self):
        elements = normalize_ifc(self._build_ifc_with_space())
        space = next((e for e in elements if e.ifc_type == "IfcSpace"), None)
        assert space is not None
        assert space.name == "upper_room"

    def test_space_inherits_parent_storey_idx(self):
        elements = normalize_ifc(self._build_ifc_with_space())
        space = next((e for e in elements if e.ifc_type == "IfcSpace"), None)
        assert space is not None
        # Upper storey is sorted second by elevation → idx 1, elev 3000mm
        assert space.storey_idx == 1
        assert space.storey_elevation == 3000

    def test_bare_space_without_representation_is_not_extracted(self):
        model = ifcopenshell.file(schema="IFC4")
        project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
        site = model.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new(), ObjectPlacement=_identity_placement(model))
        building = model.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new(), ObjectPlacement=_identity_placement(model))
        storey = model.create_entity("IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(), Name="Ground", ObjectPlacement=_identity_placement(model))
        space = model.create_entity(
            "IfcSpace",
            GlobalId=ifcopenshell.guid.new(),
            Name="bare_space",
            ObjectPlacement=_identity_placement(model),
        )
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=project, RelatedObjects=[site])
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=site, RelatedObjects=[building])
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=building, RelatedObjects=[storey])
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), RelatingObject=storey, RelatedObjects=[space])
        elements = normalize_ifc(model.to_string())
        assert not any(e.ifc_type == "IfcSpace" for e in elements)


# ---------------------------------------------------------------------------
# Nested aggregation — the importer walks IfcRelAggregates to any depth
# ---------------------------------------------------------------------------

def _walk(elements, depth=0):
    """``(depth, element)`` for the whole extracted tree, roots at depth 0."""
    for element in elements:
        yield depth, element
        yield from _walk(element.children, depth + 1)


def _world_centre_mm(model, name):
    """Tessellated world-space centre of the product called ``name``, in mm.

    Measured through ``ifcopenshell.geom`` with ``use-world-coords``, i.e. the
    same absolute placement chain ``_resolve_placement`` reconstructs, but
    computed by an INDEPENDENT implementation. Comparing the two is what makes
    the round-trip assertion about geometry rather than about names.
    """
    import ifcopenshell.geom as geom

    settings = geom.settings()
    settings.set("use-world-coords", True)
    for product in model.by_type("IfcProduct"):
        if product.Name != name or getattr(product, "Representation", None) is None:
            continue
        shape = geom.create_shape(settings, product)
        verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        centre = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
        return tuple(int(round(c * 1000)) for c in centre)
    raise AssertionError("no product named %r carries geometry" % (name,))


def _world_bbox_mm(model, name):
    """``(min, max)`` of the same tessellation ``_world_centre_mm`` averages.

    ``GeometryInfo.start``/``.end`` ARE a world-space bounding box, so this is
    the measurement that can falsify them — a centre alone is blind to a box
    that came back the right size in the wrong place *symmetrically*, and to
    one that came back the wrong size around the right centre.
    """
    import ifcopenshell.geom as geom

    settings = geom.settings()
    settings.set("use-world-coords", True)
    for product in model.by_type("IfcProduct"):
        if product.Name != name or getattr(product, "Representation", None) is None:
            continue
        shape = geom.create_shape(settings, product)
        verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        return (
            tuple(int(round(c * 1000)) for c in verts.min(axis=0)),
            tuple(int(round(c * 1000)) for c in verts.max(axis=0)),
        )
    raise AssertionError("no product named %r carries geometry" % (name,))


def _opened(ifc_content):
    """Parse an IFC string. The caller keeps the returned model alive."""
    import os
    import tempfile

    path = tempfile.mktemp(suffix=".ifc")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(ifc_content)
    try:
        return ifcopenshell.open(path)
    finally:
        try:
            os.unlink(path)
        except OSError:                                   # pragma: no cover
            pass


class TestNestedAggregation:
    """``IfcRelAggregates`` nests, and so must the walk that reads it back.

    The export side has nested since v1.5: an ``Element`` anchored into a
    container becomes its own IFC product that aggregates its own solids
    (``generator._create_element_generic``), so an authored depth-2 tree is a
    depth-2 IFC tree. ``_extract_aggregated_children`` iterated ONE level, and
    ``ExtractedElement.children`` already nested — so the shape was there and
    only the walk was missing. A depth-2 model exported correctly and
    re-imported with its grandchildren gone, no warning and no error.

    Three claims, and the last two are not about our own files:

    1.  **The grandchild survives, at its resolved world placement.** Asserted
        against an independent tessellation, because a name-only assertion
        passes just as well when the geometry comes back at the origin.
    2.  **The grouping/parts discriminator re-runs at every level.**
        ``_is_grouping_parent`` separates a grouping of peers from one
        element's parts on facts the schema declines to encode, and a nested
        grouping parent is a shape our emitter cannot write but a third-party
        assembly can. Consulted only at the top, the recursion would fold such
        a parent into its grandparent's geometry while ``_extract_groupings``
        still reported it and its members were still lifted as peers — the
        model would come back holding its members twice.
    3.  **A cyclic aggregation terminates.** Also third-party input. Our
        emitter cannot write one, which is not a reason the reader may hang on
        one; and because ``_try_extract_element`` catches ``Exception``, the
        stack blowing out does not surface as a crash — it deletes the whole
        root element and counts it as "skipped".
    """

    # -- fixtures ---------------------------------------------------------

    def _depth2_project(self):
        """A wall with an anchored sub-assembly that owns the only geometry.

        ``Wall`` -> ``Element(IfcMember)`` -> ``Box`` is the shape
        ``_create_element_generic`` documents: the assembly stays one
        addressable product instead of dissolving into loose solids, so the
        box is a GRANDchild of the wall and unreachable at depth 1.
        """
        from lite_step.models import Box, Element, Point, Project, Storey, Wall

        proj = Project(name="depth2")
        storey = Storey(name="ground", elevation=0)
        proj.add_storey(storey)
        wall = Wall(name="south")
        wall.add(Box(start=Point(x=0, y=0, z=0),
                     end=Point(x=6000, y=300, z=2700),
                     type="wall", name="body"))
        bracket = Element(name="bracket", ifc_class="IfcMember")
        bracket.add(Box(start=Point(x=0, y=0, z=0),
                        end=Point(x=200, y=200, z=400), name="cleat"))
        wall.anchor(bracket, along=2000, up=1000)
        storey.add(wall)
        return proj

    def _depth2_ifc(self):
        """Compile the fixture on the ifcopenshell backend.

        The backend is stated, not defaulted — but here it is also the only
        one production can take: ``can_stream()`` refuses a project carrying
        ``placement=Anchor``, so ``generate_ifc`` falls back for this model
        whatever it is asked for.
        """
        from lite_step.compiler.executor import normalize_project_to_meters
        from lite_step.ifc.generator import generate_ifc

        result = generate_ifc(normalize_project_to_meters(self._depth2_project()),
                              source_code="src")
        assert result.success, result.error
        return result.ifc_content

    def _nested_grouping_ifc(self):
        """A third-party shape: a roof body aggregating a truss ASSEMBLY.

        ``IfcRoof`` carries a ``Representation``, so its aggregation is one
        element's parts and the walk descends into it. The
        ``IfcElementAssembly`` it holds carries none and is the schema's own
        name for "a whole made of peer parts", so ``_is_grouping_parent``
        calls it a grouping — at level 2, which is the only level this can be
        decided at.
        """
        model = ifcopenshell.file(schema="IFC4")
        project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
        site = model.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new(),
                                   ObjectPlacement=_identity_placement(model))
        building = model.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new(),
                                       ObjectPlacement=_identity_placement(model))
        storey = model.create_entity("IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(),
                                     Name="Ground",
                                     ObjectPlacement=_identity_placement(model))
        for parent, kids in ((project, [site]), (site, [building]), (building, [storey])):
            model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                                RelatingObject=parent, RelatedObjects=kids)

        roof = model.create_entity("IfcRoof", GlobalId=ifcopenshell.guid.new(),
                                   Name="roof",
                                   ObjectPlacement=_identity_placement(model))
        context = _attach_brep_representation(
            model, roof, _make_brep_box(model, 8.0, 6.0, 0.3))

        # The nested grouping parent: no Representation, IfcElementAssembly.
        truss = model.create_entity("IfcElementAssembly", GlobalId=ifcopenshell.guid.new(),
                                    Name="truss:T1",
                                    ObjectPlacement=_identity_placement(model))
        chord = model.create_entity("IfcBeam", GlobalId=ifcopenshell.guid.new(),
                                    Name="chord",
                                    ObjectPlacement=_identity_placement(model))
        _attach_brep_representation(
            model, chord, _make_brep_box(model, 6.0, 0.1, 0.2), context)
        post = model.create_entity("IfcMember", GlobalId=ifcopenshell.guid.new(),
                                   Name="post",
                                   ObjectPlacement=_identity_placement(model))
        _attach_brep_representation(
            model, post, _make_brep_box(model, 0.1, 0.1, 1.4), context)

        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                            RelatingObject=roof, RelatedObjects=[truss])
        model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                            RelatingObject=truss, RelatedObjects=[chord, post])
        model.create_entity("IfcRelContainedInSpatialStructure",
                            GlobalId=ifcopenshell.guid.new(),
                            RelatingStructure=storey, RelatedElements=[roof])
        return model.to_string(), truss.GlobalId

    def _cyclic_ifc(self):
        """``host`` -> ``a`` -> ``b`` -> ``a``: a decomposition that loops.

        The cycle is deliberately BELOW the contained product rather than back
        onto it. A child that is spatially contained is dropped by the legacy
        peer skip before anything else looks at it, so a two-node loop through
        the storey-contained host would never reach the guard under test.
        """
        model = ifcopenshell.file(schema="IFC4")
        project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
        site = model.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new(),
                                   ObjectPlacement=_identity_placement(model))
        building = model.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new(),
                                       ObjectPlacement=_identity_placement(model))
        storey = model.create_entity("IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(),
                                     Name="Ground",
                                     ObjectPlacement=_identity_placement(model))
        for parent, kids in ((project, [site]), (site, [building]), (building, [storey])):
            model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                                RelatingObject=parent, RelatedObjects=kids)

        context = None
        products = {}
        for name, dims in (("host", (4.0, 0.3, 2.7)), ("a", (1.0, 0.2, 0.5)),
                           ("b", (0.5, 0.2, 0.5))):
            product = model.create_entity(
                "IfcBuildingElementProxy", GlobalId=ifcopenshell.guid.new(),
                Name=name, ObjectPlacement=_identity_placement(model))
            context = _attach_brep_representation(
                model, product, _make_brep_box(model, *dims), context)
            products[name] = product

        model.create_entity("IfcRelContainedInSpatialStructure",
                            GlobalId=ifcopenshell.guid.new(),
                            RelatingStructure=storey,
                            RelatedElements=[products["host"]])
        for parent, child in (("host", "a"), ("a", "b"), ("b", "a")):
            model.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                                RelatingObject=products[parent],
                                RelatedObjects=[products[child]])
        return model.to_string()

    # -- 1. the grandchild survives, where it actually is ------------------

    def test_depth2_grandchild_survives_with_its_world_placement(self):
        """Without the recursion the box is simply absent from the import.

        The placement is asserted against an independent tessellation of the
        same product rather than against a literal, and it is asserted at all
        because "a child with the right name exists" is satisfied by a
        grandchild that came back at the origin — which is the failure a
        reconstructor would render as a part detached from its host.
        """
        from lite_step.ifc.normalizer import normalize_ifc_full

        ifc_content = self._depth2_ifc()
        model = _opened(ifc_content)
        result = normalize_ifc_full(ifc_content)

        found = [(d, e) for d, e in _walk(result.elements)
                 if e.name.startswith("box:cleat")]
        assert len(found) == 1, [e.name for _, e in _walk(result.elements)]
        depth, cleat = found[0]

        # Depth 2: wall -> IfcMember sub-assembly -> the box.
        assert depth == 2
        assert cleat.ifc_type == "IfcBuildingElementProxy"

        # The resolved world matrix, not the origin, and not the host's.
        assert cleat.placement.translation == _world_centre_mm(model, cleat.name)
        assert cleat.placement.translation != (0, 0, 0)

    def test_depth2_intermediate_assembly_keeps_its_identity(self):
        """The body-less sub-assembly must stay in the tree, not be elided.

        It carries the assembly's name and class, and it is what an author
        wrote; hoisting the box straight onto the wall would round-trip the
        geometry while losing the ``Element`` that owns it.
        """
        from lite_step.ifc.normalizer import normalize_ifc_full

        result = normalize_ifc_full(self._depth2_ifc())
        wall = next(e for e in result.elements if e.ifc_type == "IfcWall")
        assembly = next(c for c in wall.children if c.ifc_type == "IfcMember")
        assert assembly.name.startswith("member:bracket")
        assert len(assembly.children) == 1
        assert assembly.children[0].name.startswith("box:cleat")

    # -- 2. the discriminator, re-run per level ----------------------------

    def test_nested_grouping_parent_is_reported_as_a_grouping(self):
        """Consulted only at the top, this assembly folds into the roof.

        It would then appear BOTH as a child element of the roof and, through
        ``_extract_groupings``, as a grouping whose members were separately
        lifted to top level — the same products in the model twice. The parent
        must therefore appear nowhere in ``elements``, at any depth.
        """
        from lite_step.ifc.normalizer import normalize_ifc_full

        ifc_content, truss_guid = self._nested_grouping_ifc()
        result = normalize_ifc_full(ifc_content)

        grouping = next((g for g in result.groupings if g.guid == truss_guid), None)
        assert grouping is not None, [g.name for g in result.groupings]
        assert grouping.kind == "aggregate"
        assert grouping.name == "truss:T1"
        assert sorted(grouping.members) == ["chord", "post"]

        assert truss_guid not in {e.guid for _, e in _walk(result.elements)}
        assert "truss:T1" not in {e.name for _, e in _walk(result.elements)}

    def test_nested_grouping_members_are_peer_elements_exactly_once(self):
        """The members are peers of the roof, and there is only one of each.

        Folding the parent in would nest them under the roof as well as
        lifting them, which is how a facade comes back with its walls twice.
        """
        from lite_step.ifc.normalizer import normalize_ifc_full

        ifc_content, _ = self._nested_grouping_ifc()
        result = normalize_ifc_full(ifc_content)

        names = [e.name for _, e in _walk(result.elements)]
        assert names.count("chord") == 1
        assert names.count("post") == 1
        assert {e.name for d, e in _walk(result.elements) if d == 0} == {
            "roof", "chord", "post"}

    # -- 3. a cyclic aggregation terminates --------------------------------

    def test_cyclic_aggregation_terminates(self):
        """Without the ``seen`` guard this recurses until the stack gives out.

        And it does not surface as a crash: ``_try_extract_element`` catches
        ``Exception``, so the ``RecursionError`` is swallowed and ``host``
        disappears from the import entirely. Asserting that ``host`` is still
        there — with ``a`` under it and the loop back to ``a`` refused rather
        than walked — is what distinguishes "guarded" from "blew up quietly".
        """
        from lite_step.ifc.normalizer import normalize_ifc_full

        result = normalize_ifc_full(self._cyclic_ifc())

        host = next((e for e in result.elements if e.name == "host"), None)
        assert host is not None, "the cyclic branch swallowed its root element"

        names = [e.name for _, e in _walk(result.elements)]
        assert names.count("host") == 1
        assert names.count("a") == 1
        assert names.count("b") == 1

        a = next(c for c in host.children if c.name == "a")
        b = next(c for c in a.children if c.name == "b")
        assert b.children == [], "the cycle back onto 'a' must be refused"


# ---------------------------------------------------------------------------
# Box reconstruction — an extrusion runs FORWARD from its profile plane
# ---------------------------------------------------------------------------

def _one_box_ifc(
    *,
    placement_m=(0.0, 0.0, 0.0),
    x_dim=6.0,
    y_dim=0.2,
    depth=3.0,
    profile_position_z=0.0,
    profile_2d_location=None,
    profile_2d_ref_direction=None,
    extruded_direction=(0.0, 0.0, 1.0),
    name="the_box",
):
    """One ``IfcWall`` whose Body is one ``IfcExtrudedAreaSolid``, in meters.

    Every knob a writer has over where the resulting box lands is a parameter
    here, because the whole point of the reconstruction is that it reads them
    off the file instead of assuming the convention this compiler happens to
    emit. Returns the STEP string.
    """
    model = ifcopenshell.file(schema="IFC4")
    project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
    site = model.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new())
    building = model.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new())
    storey = model.create_entity(
        "IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(), Name="Ground Floor",
    )
    for entity in (site, building, storey):
        origin = model.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
        ax = model.create_entity("IfcAxis2Placement3D", Location=origin)
        entity.ObjectPlacement = model.create_entity("IfcLocalPlacement", RelativePlacement=ax)

    for parent, child in ((project, site), (site, building), (building, storey)):
        model.create_entity(
            "IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
            RelatingObject=parent, RelatedObjects=[child],
        )

    point = model.create_entity(
        "IfcCartesianPoint", Coordinates=tuple(float(c) for c in placement_m),
    )
    ax = model.create_entity("IfcAxis2Placement3D", Location=point)
    wall = model.create_entity(
        "IfcWall", GlobalId=ifcopenshell.guid.new(), Name=name,
        ObjectPlacement=model.create_entity("IfcLocalPlacement", RelativePlacement=ax),
    )

    profile_kwargs = {"ProfileType": "AREA", "XDim": float(x_dim), "YDim": float(y_dim)}
    if profile_2d_location is not None or profile_2d_ref_direction is not None:
        loc = model.create_entity(
            "IfcCartesianPoint",
            Coordinates=tuple(float(c) for c in (profile_2d_location or (0.0, 0.0))),
        )
        ref = None
        if profile_2d_ref_direction is not None:
            ref = model.create_entity(
                "IfcDirection",
                DirectionRatios=tuple(float(c) for c in profile_2d_ref_direction),
            )
        profile_kwargs["Position"] = model.create_entity(
            "IfcAxis2Placement2D", Location=loc, RefDirection=ref,
        )
    profile = model.create_entity("IfcRectangleProfileDef", **profile_kwargs)

    prof_origin = model.create_entity(
        "IfcCartesianPoint", Coordinates=(0.0, 0.0, float(profile_position_z)),
    )
    solid = model.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=model.create_entity("IfcAxis2Placement3D", Location=prof_origin),
        ExtrudedDirection=model.create_entity(
            "IfcDirection", DirectionRatios=tuple(float(c) for c in extruded_direction),
        ),
        Depth=float(depth),
    )

    context = model.create_entity(
        "IfcGeometricRepresentationContext",
        ContextType="Model", CoordinateSpaceDimension=3, Precision=1e-5,
    )
    wall.Representation = model.create_entity(
        "IfcProductDefinitionShape",
        Representations=[model.create_entity(
            "IfcShapeRepresentation", ContextOfItems=context,
            RepresentationIdentifier="Body", RepresentationType="SweptSolid",
            Items=[solid],
        )],
    )
    model.create_entity(
        "IfcRelContainedInSpatialStructure", GlobalId=ifcopenshell.guid.new(),
        RelatingStructure=storey, RelatedElements=[wall],
    )
    return model.to_string()


def _imported_box(ifc_content, name):
    """``(start, end)`` of the imported element called ``name``."""
    from lite_step.ifc.normalizer import normalize_ifc_full

    result = normalize_ifc_full(ifc_content)
    match = [e for _, e in _walk(result.elements) if e.name == name]
    assert len(match) == 1, [e.name for _, e in _walk(result.elements)]
    assert match[0].geometry.mode == "box"
    return match[0].geometry.start, match[0].geometry.end


class TestBoxSpansTheExtrusionForward:
    """``GeometryInfo.start``/``.end`` is where the box actually is.

    An ``IfcExtrudedAreaSolid`` sweeps its profile from the plane its
    ``Position`` puts it on, forward by ``Depth`` along ``ExtrudedDirection``:
    the profile plane is the START face. The reconstruction spanned
    ``+/-Depth/2`` about the combined origin instead, so on every file whose
    ``Position`` already carries the offset — which is most of ours, since the
    wall/solid paths write ``z = -h/2`` — the offset was applied twice and the
    box came back ``Depth/2`` low along the extrusion axis. X, Y and
    ``placement.translation`` were all correct, and a reconstructor emits
    ``start``/``end`` as the DSL ``Box``, so a cold-imported model came back
    with every swept box sunk by half its own height.

    Every case is asserted against ``_world_bbox_mm`` — the same
    ``ifcopenshell.geom`` world tessellation ``TestNestedAggregation`` measures
    placement with, i.e. an INDEPENDENT implementation of the placement chain —
    as well as against the authored number, so neither a shared bug in our
    matrix maths nor a hand-copied expectation can make it pass.

    The last four cases are third-party input. Our own emitters write the
    profile ``Position`` three different ways already (``z = -h/2`` on the wall
    and solid paths, ``z = 0`` on both backends' opening paths, ``z = min`` on
    a contour cut operand) and never write an ``IfcAxis2Placement2D`` on the
    profile or a non-``+Z`` ``ExtrudedDirection`` at all — all of which are
    ordinary in a file another tool wrote. Spanning ``[0, Depth]`` from
    whatever the file says is the one rule that is right for all of them,
    which is why the fix is not "subtract the offset our generator adds".
    """

    # -- 1. our own files, compiled ---------------------------------------

    def test_compiled_wall_box_is_where_it_was_authored(self):
        """``Wall`` + ``Box(z=0..2700)`` on the ifcopenshell backend.

        Measured before the fix: ``-1350..1350`` — exactly ``-depth/2``.
        """
        ifc_content = TestNestedAggregation()._depth2_ifc()
        model = _opened(ifc_content)

        name = "wall:south:storey:ground"
        start, end = _imported_box(ifc_content, name)

        assert (start, end) == _world_bbox_mm(model, name)
        assert start == (0, 0, 0)
        assert end == (6000, 300, 2700)

    def test_anchored_box_is_where_it_was_authored(self):
        """The anchored 200x200x400 cleat at ``up=1000`` — true z ``1000..1400``.

        Measured before the fix: ``800..1200``. Kept separate from the wall
        case because the anchor bake composes another matrix into
        ``abs_matrix``, and a fix that merely cancelled the generator's own
        ``-h/2`` term would look right on the plain wall and stay wrong here.
        """
        ifc_content = TestNestedAggregation()._depth2_ifc()
        model = _opened(ifc_content)

        name = "box:cleat:member:bracket:wall:south:storey:ground"
        start, end = _imported_box(ifc_content, name)

        assert (start, end) == _world_bbox_mm(model, name)
        assert start == (3800, 100, 1000)
        assert end == (4000, 300, 1400)

    def test_hand_built_fixture_wall_spans_zero_to_three_metres(self):
        """The fixture was reported on: true ``0..3000``, was ``-1500..1500``."""
        ifc_content = TestNormalizeIfc()._create_ifc_with_wall()
        model = _opened(ifc_content)

        start, end = _imported_box(ifc_content, "wall_north")

        assert (start, end) == _world_bbox_mm(model, "wall_north")
        assert start == (0, 900, 0)
        assert end == (6000, 1100, 3000)

    # -- 2. third-party conventions ---------------------------------------

    def test_profile_plane_at_the_product_origin_third_party(self):
        """``Position`` z = 0 — the profile plane IS the bottom face.

        This is what a writer that does not pre-centre its solids emits (and
        what both of our own opening paths emit), so the box runs ``0..3000``
        UP from the placement. Reporting ``-1500..1500`` here too is not reading a
        convention, it is ignoring the file.
        """
        ifc_content = _one_box_ifc(placement_m=(0.0, 0.0, 0.0), profile_position_z=0.0)
        model = _opened(ifc_content)

        start, end = _imported_box(ifc_content, "the_box")

        assert (start, end) == _world_bbox_mm(model, "the_box")
        assert start == (-3000, -100, 0)
        assert end == (3000, 100, 3000)

    def test_centred_profile_position_third_party(self):
        """``Position`` z = ``-Depth/2`` with the product at the box CENTRE.

        The convention our wall and solid paths use, written by hand with
        different numbers so it cannot pass by matching a compiled fixture.
        The 2.0 m box is centred at z = 5.0, so it spans ``4000..6000`` — and
        the same ``[0, Depth]`` rule that gave ``0..3000`` above gives it,
        with no branch on who wrote the file.
        """
        ifc_content = _one_box_ifc(
            placement_m=(1.0, 2.0, 5.0), x_dim=0.4, y_dim=0.6, depth=2.0,
            profile_position_z=-1.0,
        )
        model = _opened(ifc_content)

        start, end = _imported_box(ifc_content, "the_box")

        assert (start, end) == _world_bbox_mm(model, "the_box")
        assert start == (800, 1700, 4000)
        assert end == (1200, 2300, 6000)

    def test_rect_profile_position_offsets_the_box(self):
        """``IfcRectangleProfileDef.Position`` places the rectangle in-plane.

        Optional, identity on every file we write, and honoured by every
        consumer — a rectangle centred 1.5 m along its sweep plane's X is
        1.5 m along, not at the origin. Ignoring it put the box in the wrong
        place in X and Y with the extrusion axis perfectly correct.
        """
        ifc_content = _one_box_ifc(
            x_dim=1.0, y_dim=1.0, depth=2.0,
            profile_2d_location=(1.5, -0.5),
        )
        model = _opened(ifc_content)

        start, end = _imported_box(ifc_content, "the_box")

        assert (start, end) == _world_bbox_mm(model, "the_box")
        assert start == (1000, -1000, 0)
        assert end == (2000, 0, 2000)

    def test_extruded_direction_is_read_not_assumed(self):
        """``ExtrudedDirection`` need not be local +Z.

        Both our backends always write ``(0,0,1)``; the attribute exists
        because a writer may sweep obliquely, and the swept span is
        ``[0, Depth]`` along THAT direction — ``Depth`` is measured along it,
        so the ratios are normalized before scaling. Sweeping 3 m along
        ``(0,1,1)`` puts ``3/sqrt(2)`` m into Y and the same into Z; assuming
        local +Z gives 3 m of Z and no Y at all.
        """
        ifc_content = _one_box_ifc(extruded_direction=(0.0, 1.0, 1.0))
        model = _opened(ifc_content)

        start, end = _imported_box(ifc_content, "the_box")

        assert (start, end) == _world_bbox_mm(model, "the_box")
        reach = int(round(3000.0 / math.sqrt(2.0)))          # 2121 mm
        assert start == (-3000, -100, 0)
        assert end == (3000, 100 + reach, reach)
