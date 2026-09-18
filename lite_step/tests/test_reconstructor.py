"""Tests for cold import reconstructor."""


from lite_step.ifc.normalizer import ExtractedElement, PlacementInfo, GeometryInfo
from lite_step.transpiler.reconstructor import reconstruct_litestep, _sanitize_id


def _make_element(
    name: str,
    ifc_type: str = "IfcBuildingElementProxy",
    start=(0, 0, 0),
    end=(1000, 1000, 1000),
    storey_idx=0,
    storey_elevation=0,
) -> ExtractedElement:
    return ExtractedElement(
        ifc_type=ifc_type,
        name=name,
        guid="abcd1234",
        storey_idx=storey_idx,
        storey_elevation=storey_elevation,
        placement=PlacementInfo(translation=(0, 0, 0), rotation=(0, 0, 0)),
        geometry=GeometryInfo(mode="box", start=start, end=end),
    )


class TestReconstructLitestep:
    def test_generates_valid_python(self):
        """Generated code should be valid Python syntax."""
        elements = [
            _make_element("wall_1", "IfcWall", (0, 0, 0), (6000, 3000, 200)),
            _make_element("col_1", "IfcColumn", (0, 0, 0), (300, 3000, 300)),
        ]
        source, _ = reconstruct_litestep(elements)
        # Should parse without syntax errors
        compile(source, "<test>", "exec")

    def test_header_comment(self):
        elements = [_make_element("elem_1")]
        source, _ = reconstruct_litestep(elements)
        assert "Reconstructed from IFC" in source

    def test_type_mapping_wall(self):
        elements = [_make_element("wall_1", "IfcWall")]
        source, _ = reconstruct_litestep(elements)
        assert "Wall(id=" in source

    def test_type_mapping_column(self):
        elements = [_make_element("col_1", "IfcColumn")]
        source, _ = reconstruct_litestep(elements)
        assert "Column(id=" in source

    def test_type_mapping_beam(self):
        elements = [_make_element("beam_1", "IfcBeam")]
        source, _ = reconstruct_litestep(elements)
        assert "Beam(id=" in source

    def test_type_mapping_slab(self):
        elements = [_make_element("slab_1", "IfcSlab")]
        source, _ = reconstruct_litestep(elements)
        assert "Slab(" in source

    def test_type_mapping_fallback_solid(self):
        elements = [_make_element("proxy_1", "IfcBuildingElementProxy")]
        source, _ = reconstruct_litestep(elements)
        assert "Box(" in source

    def test_multiple_storeys(self):
        elements = [
            _make_element("wall_g", "IfcWall", storey_idx=0, storey_elevation=0),
            _make_element("wall_1", "IfcWall", storey_idx=1, storey_elevation=3000),
        ]
        source, _ = reconstruct_litestep(elements)
        assert "storey_0" in source
        assert "storey_1" in source

    def test_empty_elements(self):
        source, _ = reconstruct_litestep([])
        assert "Reconstructed from IFC" in source
        assert "generate_project" in source
        compile(source, "<test>", "exec")

    def test_generate_project_wrapper(self):
        elements = [_make_element("elem_1")]
        source, _ = reconstruct_litestep(elements)
        assert "def generate_project():" in source
        assert "result = generate_project()" in source

    def test_box_geometry_coordinates(self):
        elements = [_make_element("wall_1", "IfcWall", (0, 0, 0), (6000, 3000, 200))]
        source, _ = reconstruct_litestep(elements)
        assert "Point(x=0, y=0, z=0)" in source
        assert "Point(x=6000, y=3000, z=200)" in source


class TestSanitizeId:
    def test_valid_identifier(self):
        assert _sanitize_id("wall_north", "Solid", "abc", set()) == "wall_north"

    def test_dunder_prefix_rejected(self):
        result = _sanitize_id("__private", "Solid", "abc", set())
        assert not result.startswith("__")

    def test_numeric_prefix_fixed(self):
        result = _sanitize_id("123abc", "Solid", "abc", set())
        assert result[0].isalpha() or result[0] == "_"

    def test_empty_name_uses_guid(self):
        result = _sanitize_id("", "Solid", "abcd1234", set())
        assert "abcd1234" in result

    def test_uniqueness(self):
        used = {"wall_1"}
        result = _sanitize_id("wall_1", "Wall", "abc", used)
        assert result != "wall_1"
        assert result.startswith("wall_1")


def _make_brep_element(name: str, ifc_type: str = "IfcBuildingElementProxy"):
    """Element whose geometry carries full mesh vertices + faces (the
    new Phase A.1 normalizer output for IfcFacetedBrep)."""
    verts = [
        (0, 0, 0), (1000, 0, 0), (1000, 500, 0), (0, 500, 0),
        (0, 0, 200), (1000, 0, 200), (1000, 500, 200), (0, 500, 200),
    ]
    faces = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)]  # bottom + top
    return ExtractedElement(
        ifc_type=ifc_type,
        name=name,
        guid="meshguid01",
        storey_idx=0,
        storey_elevation=0,
        placement=PlacementInfo(translation=(0, 0, 0), rotation=(0, 0, 0)),
        geometry=GeometryInfo(
            mode="mesh",
            start=(0, 0, 0),
            end=(1000, 500, 200),
            mesh_vertices=verts,
            mesh_faces=faces,
        ),
    )


def _make_boolean_element(name: str, ifc_type: str = "IfcWall"):
    """Container element whose geometry is a boolean difference of two
    box operands — exercises Solid.difference() emission."""
    primary = GeometryInfo(mode="box", start=(0, 0, 0), end=(6000, 3000, 200))
    void = GeometryInfo(mode="box", start=(2000, 0, 800), end=(3000, 2100, 200))
    return ExtractedElement(
        ifc_type=ifc_type,
        name=name,
        guid="boolguid01",
        storey_idx=0,
        storey_elevation=0,
        placement=PlacementInfo(translation=(0, 0, 0), rotation=(0, 0, 0)),
        geometry=GeometryInfo(
            mode="boolean",
            start=(0, 0, 0),
            end=(6000, 3000, 200),
            boolean_operands=[primary, void],
            boolean_operation="difference",
        ),
    )


def _make_opaque_element(name: str, ifc_type: str = "IfcBuildingElementProxy"):
    """Element with no usable geometry (unknown mode, no bbox) — must
    fall through to the OPAQUE placeholder path."""
    return ExtractedElement(
        ifc_type=ifc_type,
        name=name,
        guid="opaqueguid",
        storey_idx=0,
        storey_elevation=0,
        placement=PlacementInfo(translation=(500, 500, 500), rotation=(0, 0, 0)),
        geometry=GeometryInfo(mode="unknown"),
    )


def _make_wall_with_opening_fills():
    """Wall with one IfcOpeningElement child that contains a Door fill.
    Exercises the opening re-parenting path: the door should end up as
    a wall.add(Box(...)) child rather than a floating storey element."""
    door = ExtractedElement(
        ifc_type="IfcDoor",
        name="entry_door",
        guid="doorguid01",
        storey_idx=0,
        storey_elevation=0,
        placement=PlacementInfo(translation=(0, 0, 0), rotation=(0, 0, 0)),
        geometry=GeometryInfo(mode="box", start=(2000, 0, 0), end=(2900, 100, 2100)),
    )
    opening = ExtractedElement(
        ifc_type="IfcOpeningElement",
        name="opening_door",
        guid="openguid01",
        storey_idx=0,
        storey_elevation=0,
        placement=PlacementInfo(translation=(0, 0, 0), rotation=(0, 0, 0)),
        geometry=GeometryInfo(mode="box", start=(2000, 0, 0), end=(2900, 100, 2100)),
        children=[door],
    )
    wall = ExtractedElement(
        ifc_type="IfcWall",
        name="north_wall",
        guid="wallguid01",
        storey_idx=0,
        storey_elevation=0,
        placement=PlacementInfo(translation=(0, 0, 0), rotation=(0, 0, 0)),
        geometry=GeometryInfo(mode="box", start=(0, 0, 0), end=(6000, 100, 3000)),
        children=[opening],
    )
    return wall, door


class TestCoverageReport:
    def test_box_only_is_exact(self):
        elements = [_make_element("wall_1", "IfcWall", (0, 0, 0), (6000, 3000, 200))]
        _, coverage = reconstruct_litestep(elements)
        assert coverage.exact == 1
        assert coverage.approximated == 0
        assert coverage.opaque == 0
        assert coverage.by_type["IfcWall"] == (1, 0, 0)

    def test_opaque_element_increments_opaque(self):
        _, coverage = reconstruct_litestep([_make_opaque_element("mystery")])
        assert coverage.opaque == 1
        assert coverage.exact == 0
        # And the OPAQUE comment is in the source so the UI banner can
        # cite a count and a list (per loud-failure-budget).
        source, _ = reconstruct_litestep([_make_opaque_element("mystery")])
        assert "OPAQUE: original geometry lost" in source

    def test_mixed_quality_tallies_by_type(self):
        elements = [
            _make_element("wall_1", "IfcWall", (0, 0, 0), (6000, 3000, 200)),  # exact
            _make_opaque_element("proxy_1", "IfcBuildingElementProxy"),         # opaque
            _make_element("wall_2", "IfcWall", (0, 0, 0), (6000, 3000, 200)),  # exact
        ]
        _, coverage = reconstruct_litestep(elements)
        assert coverage.exact == 2
        assert coverage.opaque == 1
        assert coverage.by_type["IfcWall"] == (2, 0, 0)
        assert coverage.by_type["IfcBuildingElementProxy"] == (0, 0, 1)
        assert coverage.total == 3


class TestMeshEmission:
    def test_brep_emits_mesh_with_vertices_and_faces(self):
        source, coverage = reconstruct_litestep([_make_brep_element("brep_proxy")])
        assert "Mesh(vertices=" in source
        assert "faces=[" in source
        # 8 vertices + 4 triangles serialised
        assert source.count("Point(x=") == 8
        assert coverage.exact == 1

    def test_brep_without_faces_degrades_to_approximated_solid(self):
        verts = [(0, 0, 0), (1000, 0, 0), (1000, 500, 0), (0, 500, 0)]
        elem = ExtractedElement(
            ifc_type="IfcBuildingElementProxy",
            name="bbox_brep",
            guid="bb01",
            storey_idx=0,
            storey_elevation=0,
            placement=PlacementInfo(translation=(0, 0, 0), rotation=(0, 0, 0)),
            geometry=GeometryInfo(
                mode="mesh",
                start=(0, 0, 0),
                end=(1000, 500, 200),
                mesh_vertices=verts,
                mesh_faces=None,
            ),
        )
        source, coverage = reconstruct_litestep([elem])
        assert "approximated: brep without face indices" in source
        assert coverage.approximated == 1


class TestBooleanEmission:
    def test_wall_with_boolean_difference_emits_cuts(self):
        wall = _make_boolean_element("door_wall", "IfcWall")
        source, coverage = reconstruct_litestep([wall])
        # The wall.add() body should be a Box(...).difference(Box(material="Void", ...))
        assert ".difference(" in source
        assert 'material="Void"' in source
        assert coverage.exact == 1

    def test_wall_with_boolean_union_emits_adds(self):
        wall = _make_boolean_element("merged_wall", "IfcWall")
        wall.geometry.boolean_operation = "union"
        source, _ = reconstruct_litestep([wall])
        assert ".union(" in source
        assert ".difference(" not in source


class TestOpeningReparenting:
    def test_door_fill_attached_to_wall_not_storey(self):
        wall, door = _make_wall_with_opening_fills()
        source, coverage = reconstruct_litestep([wall, door])
        # Door fill is emitted as wall.add(Box(... id="entry_door" ...))
        assert 'id="entry_door"' in source
        assert "IfcDoor fill" in source
        # And the storey-level Door emission should NOT happen — there
        # should be exactly one Solid line carrying the door id, inside
        # the wall's emission block, not a separate Door at storey level.
        door_emit_lines = [ln for ln in source.split("\n") if 'id="entry_door"' in ln]
        assert len(door_emit_lines) == 1, door_emit_lines
        # Coverage records the wall as exact and the door fill as exact
        assert coverage.by_type["IfcWall"][0] == 1  # exact count
        assert coverage.by_type["IfcDoor"][0] == 1
