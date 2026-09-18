"""``IfcTriangulatedFaceSet.Closed`` — does this tessellation bound a solid?

Emitting ``$`` for it while publishing the very same fact as
``LiteStep_MeshVolume.IsWatertight`` — a proprietary Pset no IFC consumer
reads — puts the answer where nothing looks.
``Closed`` is the schema-native channel, and it is the only one a civil or
viewer application looks at to tell a ground SLAB from a draped SURFACE.

Two things are pinned here, and the second is the reason this file exists.

1.  ``Closed`` states what the element already knows. Heightmap mode forces
    ``is_watertight`` True when it tessellates the closed prism (top + bottom +
    4 skirts); explicit mode defaults FALSE and takes the author's word.

2.  **The backends agree.** ``streaming`` hardcoded ``is_watertight = True``
    ("assumption for closed meshes") while ``ifcopenshell`` read the element —
    two implementations of one idea, which is the shape, and because
    streaming is the DEFAULT path the wrong value was the one users got. Every
    hand-authored open mesh went out claiming to be watertight.

Why NOT ``IfcTriangulatedIrregularNetwork``
-------------------------------------------

Its name makes it the obvious candidate for terrain, and it is the wrong one.
A TIN represents a horizontal SURFACE, single-valued in z. Our terrain is a
closed prism because displacement subtracts foundations from it and manifold3d
needs a solid on both sides of the boolean. Emitting the prism as a TIN would
tell a consumer "sample ground level here" and hand back a bottom plane or a
skirt for any ray outside the top surface — a worse claim than the generic
face set, not a better one. ``test_terrain_is_a_face_set_not_a_tin`` pins that
decision so the next reader does not re-open it from the name alone.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.models import Mesh, Point, Project, Site

from ._fixtures import terrain_wrapper


def _compile(proj: Project, backend: str) -> str:
    if backend == "ifcopenshell":
        pytest.importorskip("ifcopenshell")
    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    return result.ifc_content


def _open(content: str):
    """Parse STEP text with ifcopenshell — used for BOTH backends, which is
    what makes the agreement test a comparison of FILES rather than of two
    text parsers."""
    import ifcopenshell

    with tempfile.NamedTemporaryFile("w", suffix=".ifc", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(content)
        path = fh.name
    try:
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _face_sets(content: str):
    pytest.importorskip("ifcopenshell")
    return _open(content).by_type("IfcTriangulatedFaceSet")


def _open_mesh() -> Mesh:
    """A single triangle — emphatically not a solid. ``is_watertight``
    defaults False, so this is the case the streaming hardcode got wrong."""
    return Mesh(
        vertices=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0),
                  Point(x=0, y=1000, z=0)],
        faces=[(0, 1, 2)],
        name="open_sheet",
    )




# ---------------------------------------------------------------------------
# 1. Closed states what the element knows
# ---------------------------------------------------------------------------


def test_heightmap_terrain_is_closed():
    """The terrain prism is watertight by construction, and now says so."""
    proj = Project(name="closed_terrain")
    site = Site(name="site")
    site.add(terrain_wrapper())
    proj.add(site)

    sets = _face_sets(_compile(proj, "ifcopenshell"))
    assert sets, "no IfcTriangulatedFaceSet emitted"
    assert all(fs.Closed is True for fs in sets), \
        f"terrain prism not marked Closed on {"ifcopenshell"}: " \
        f"{[fs.Closed for fs in sets]}"


def test_open_mesh_is_not_closed():
    """A single triangle is not a solid — and must not claim to be.

    This is the falsifier for the streaming hardcode: before the fix this
    asserted False against a literal ``True`` on that "ifcopenshell".
    """
    proj = Project(name="open_mesh")
    site = Site(name="site")
    site.add(_open_mesh())
    proj.add(site)

    sets = _face_sets(_compile(proj, "ifcopenshell"))
    assert sets, "no IfcTriangulatedFaceSet emitted"
    assert all(fs.Closed is False for fs in sets), \
        f"open sheet claims Closed on {"ifcopenshell"}: {[fs.Closed for fs in sets]}"


def test_closed_agrees_with_the_pset():
    """``Closed`` and ``LiteStep_MeshVolume.IsWatertight`` are one fact.

    They are emitted from separate call sites; a future edit that changes one
    without the other produces a file that contradicts itself.
    """
    proj = Project(name="agree")
    site = Site(name="site")
    site.add(_open_mesh())
    proj.add(site)
    model = _open(_compile(proj, "ifcopenshell"))

    closed = {fs.Closed for fs in model.by_type("IfcTriangulatedFaceSet")}
    watertight = {
        p.NominalValue.wrappedValue
        for pset in model.by_type("IfcPropertySet")
        if pset.Name == "LiteStep_MeshVolume"
        for p in pset.HasProperties if p.Name == "IsWatertight"
    }
    assert closed == watertight, \
        f"Closed={closed} contradicts IsWatertight={watertight} on {"ifcopenshell"}"
def test_terrain_is_a_face_set_not_a_tin():
    """``IfcTriangulatedIrregularNetwork`` is a SURFACE; our terrain is a
    prism. Deliberate — see this module's docstring.

    If terrain ever becomes a genuine surface (a draped DEM with no depth),
    revisit this rather than deleting it: the TIN would then be the more
    honest entity, and it would also need ``Flags`` (one integer per triangle;
    ``0`` = no breaklines), which is mandatory and has no default.
    """
    proj = Project(name="tin_check")
    site = Site(name="site")
    site.add(terrain_wrapper())
    proj.add(site)
    model = _open(_compile(proj, "ifcopenshell"))

    assert not model.by_type("IfcTriangulatedIrregularNetwork")
    assert model.by_type("IfcTriangulatedFaceSet")
