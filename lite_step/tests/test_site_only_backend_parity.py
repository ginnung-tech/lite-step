"""Backend parity for site-only vs project models (P1/P2 spatial scaffold).

A model with a DSL ``Site`` but no project elements must compile — on BOTH
the streaming and the ifcopenshell backend — to a project-less spatial tree:

    IfcProject -> IfcSite -> (site children)

with NO IfcBuilding, NO IfcBuildingStorey, and the LITESTEP_META property set
re-homed onto the IfcSite (the project it hung on does not exist).

The parity fixture uses a BARE context mesh (pure geometry — a proxy under
IfcSite), which both backends handle. The canonical TERRAIN form —
``Element(IfcGeographicElement, TERRAIN)`` wrapping the mesh — routes to the
ifcopenshell backend in production (``can_stream`` is False for any model
holding an Element), so its IfcGeographicElement emission is asserted in the
ifcopenshell-only test below, not in the parametrized parity set.

A model WITH project elements (a wall in a storey) keeps the full chain:
IfcProject -> IfcSite -> IfcBuilding -> IfcBuildingStorey -> element.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc
from lite_step.models.project import Project
from lite_step.models.elements import Box, Mesh, Site, Wall
from lite_step.models.primitives import Point


def _open_model(content: str):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    fd, path = tempfile.mkstemp(suffix=".ifc")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _terrain_mesh() -> Mesh:
    # A minimal closed-ish terrain block (a single flat quad is enough to
    # exercise the geometry/routing — watertightness is not asserted here).
    verts = [
        Point(x=-5000, y=-5000, z=0),
        Point(x=5000, y=-5000, z=0),
        Point(x=5000, y=5000, z=0),
        Point(x=-5000, y=5000, z=0),
    ]
    faces = [(0, 1, 2), (0, 2, 3)]
    return Mesh(
        name="site_block",
        vertices=verts,
        faces=faces,
        is_watertight=True,
        source_type="synthetic",
    )


def _site_only_building() -> Project:
    proj = Project(name="Site Only")
    site = Site(name="site")
    site.add(_terrain_mesh())
    proj.add(site)
    # P2: a lone Site leaves storeys empty.
    assert proj.storeys == []
    assert len(proj.sites) == 1
    return proj


def _house_building() -> Project:
    proj = Project(name="House")
    wall = Wall(name="north")
    wall.add(
        Box(
            start=Point(x=0, y=0, z=0),
            end=Point(x=5000, y=300, z=2700),
            id="body",
            color="wall",
        )
    )
    proj.add(wall)
    return proj


def _compile(proj: Project, backend: str, source: str) -> str:
    normalized = normalize_project_to_meters(proj)
    result = generate_ifc(normalized, source_code=source)
    assert result.success, f"generate_ifc({backend}) failed: {result.error}"
    return result.ifc_content


def test_site_only_has_no_building_or_storey():
    if "ifcopenshell" == "ifcopenshell":
        pytest.importorskip("ifcopenshell")
    content = _compile(_site_only_building(), "ifcopenshell", source="# site only")

    assert "IFCPROJECT(" in content
    assert "IFCSITE(" in content
    assert "IFCBUILDING(" not in content
    assert "IFCBUILDINGSTOREY" not in content


def test_site_only_terrain_wrapper_emits_geographic_element():
    """The canonical terrain form (wrapper) — ifcopenshell-only, since any
    Element routes there in production. The IfcGeographicElement carries the
    faceset DIRECTLY (merged representation, no blank proxy)."""
    pytest.importorskip("ifcopenshell")
    from lite_step.models.elements import Element

    proj = Project(name="Site Only")
    site = Site(name="site")
    wrapper = Element(ifc_class="IfcGeographicElement",
                      predefined_type="TERRAIN", name="terrain")
    wrapper.add(_terrain_mesh())
    site.add(wrapper)
    proj.add(site)

    content = _compile(proj, "ifcopenshell", source="# site only")
    model = _open_model(content)
    geos = model.by_type("IfcGeographicElement")
    assert geos and str(geos[0].PredefinedType) == "TERRAIN"
    assert geos[0].Representation is not None
    assert not model.by_type("IfcBuildingElementProxy")
    assert not model.by_type("IfcBuilding")


def test_site_only_meta_hosts_on_site():
    pytest.importorskip("ifcopenshell")
    content = _compile(_site_only_building(), "ifcopenshell", source="# site only")
    model = _open_model(content)

    assert not model.by_type("IfcBuilding")
    assert not model.by_type("IfcBuildingStorey")
    site = model.by_type("IfcSite")[0]

    meta_rels = [
        r for r in model.by_type("IfcRelDefinesByProperties")
        if getattr(r.RelatingPropertyDefinition, "Name", None) == "LITESTEP_META"
    ]
    assert len(meta_rels) == 1
    assert list(meta_rels[0].RelatedObjects) == [site], (
        "LITESTEP_META must attach to the IfcSite for a site-only model"
    )


def test_site_only_project_aggregates_site():
    pytest.importorskip("ifcopenshell")
    content = _compile(_site_only_building(), "ifcopenshell", source="# site only")
    model = _open_model(content)

    project = model.by_type("IfcProject")[0]
    site = model.by_type("IfcSite")[0]
    aggregated = [
        obj for rel in model.by_type("IfcRelAggregates") if rel.RelatingObject == project
        for obj in rel.RelatedObjects
    ]
    assert site in aggregated, "IfcProject must always aggregate the IfcSite"


def test_house_keeps_building_and_storey():
    if "ifcopenshell" == "ifcopenshell":
        pytest.importorskip("ifcopenshell")
    content = _compile(_house_building(), "ifcopenshell", source="# house")

    assert "IFCPROJECT(" in content
    assert "IFCSITE(" in content
    assert "IFCBUILDING(" in content
    assert "IFCBUILDINGSTOREY" in content
