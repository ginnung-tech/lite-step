"""``Mesh(asset=..., sha256=...)``: a cleaned scan read from the project instead of written into the script.

The script names the file by path and by the hash of its bytes, so it still says exactly what it builds; the
rules that keep a script honest (integer millimetres, inside the project, nothing read but what is named) hold.
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile

import numpy as np
import pytest

from lite_step.compiler.executor import execute_lite_step_script_inproc, normalize_project_to_meters
from lite_step.models import Element, Mesh, Project, Site
from lite_step.models.mesh_asset import MeshAssetError, load

CUBE_V = [(0, 0, 0), (1000, 0, 0), (1000, 1000, 0), (0, 1000, 0),
          (0, 0, 1000), (1000, 0, 1000), (1000, 1000, 1000), (0, 1000, 1000)]
CUBE_F = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
          (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]


def ply_ascii(verts=CUBE_V, faces=CUBE_F) -> bytes:
    head = (f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\n"
            f"property float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    return (head + "".join(f"{x} {y} {z}\n" for x, y, z in verts)
            + "".join(f"3 {a} {b} {c}\n" for a, b, c in faces)).encode()


def ply_binary(verts=CUBE_V, faces=CUBE_F) -> bytes:
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(verts)}\nproperty double x\n"
            f"property double y\nproperty double z\nelement face {len(faces)}\n"
            "property list uchar uint vertex_indices\nend_header\n").encode()
    body = b"".join(struct.pack("<3d", *v) for v in verts) + b"".join(struct.pack("<B3I", 3, *f) for f in faces)
    return head + body


def stl_binary(verts=CUBE_V, faces=CUBE_F) -> bytes:
    out = b"\0" * 80 + struct.pack("<I", len(faces))
    for f in faces:
        out += struct.pack("<3f", 0, 0, 0) + b"".join(struct.pack("<3f", *verts[i]) for i in f) + b"\0\0"
    return out


@pytest.fixture
def project_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("LITE_STEP_ASSET_ROOT", str(tmp_path))
    return tmp_path


def put(root, name, data):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("name,maker", [("cube.ply", ply_ascii), ("bin/cube.ply", ply_binary), ("cube.stl", stl_binary)])
def test_each_format_reads_the_same_closed_cube(project_dir, name, maker):
    h = put(project_dir, name, maker())
    m = Mesh(asset=name, sha256=h)
    assert len(m.vertices) == 8 and len(m.faces) == 12 and m.is_watertight
    assert {(p.x, p.y, p.z) for p in m.vertices} == set(CUBE_V)


def test_an_asset_mesh_compiles_to_a_closed_face_set(project_dir):
    import ifcopenshell
    from lite_step.ifc.generator import generate_ifc

    h = put(project_dir, "meshes/cube.ply", ply_binary())
    site = Site(name="site")
    sculpture = Element(name="statue", ifc_class="IfcBuildingElementProxy")
    sculpture.add(Mesh(name="skin", asset="meshes/cube.ply", sha256=h))
    site.add(sculpture)
    proj = Project(name="p")
    proj.add(site)
    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    with tempfile.NamedTemporaryFile("w", suffix=".ifc", delete=False, encoding="utf-8") as fh:
        fh.write(result.ifc_content)
    try:
        sets = ifcopenshell.open(fh.name).by_type("IfcTriangulatedFaceSet")
    finally:
        os.unlink(fh.name)
    assert sets and len(sets[0].CoordIndex) == 12 and sets[0].Closed


def test_a_changed_file_is_refused_never_built(project_dir):
    h = put(project_dir, "cube.ply", ply_ascii())
    put(project_dir, "cube.ply", ply_ascii(verts=[(x, y, z + 1) for x, y, z in CUBE_V]))
    with pytest.raises(MeshAssetError, match="has changed"):
        load("cube.ply", h)
    with pytest.raises(MeshAssetError, match="sha256"):
        load("cube.ply", None)


def test_paths_outside_the_project_are_refused(project_dir):
    outside = project_dir.parent / "secret.ply"
    outside.write_bytes(ply_ascii())
    h = hashlib.sha256(ply_ascii()).hexdigest()
    for bad in ("../secret.ply", str(outside), "C:/secret.ply"):
        with pytest.raises(MeshAssetError):
            load(bad, h)
    put(project_dir, "notes.txt", b"x")
    with pytest.raises(MeshAssetError, match=".ply or .stl"):
        load("notes.txt", hashlib.sha256(b"x").hexdigest())


def test_coordinates_that_are_not_whole_millimetres_are_refused(project_dir):
    h = put(project_dir, "metres.ply", ply_ascii(verts=[(x / 1000.0 + 0.0004, y / 1000.0, z / 1000.0)
                                                       for x, y, z in CUBE_V]))
    with pytest.raises(MeshAssetError, match="integer millimetres"):
        load("metres.ply", h)


def test_an_open_mesh_reads_but_is_not_called_watertight(project_dir):
    h = put(project_dir, "open.ply", ply_ascii(faces=CUBE_F[:-2]))
    m = Mesh(asset="open.ply", sha256=h)
    assert len(m.faces) == 10 and not m.is_watertight


def test_one_spelling_at_a_time(project_dir):
    h = put(project_dir, "cube.ply", ply_ascii())
    with pytest.raises(ValueError, match="ONE of"):
        Mesh(asset="cube.ply", sha256=h, vertices=[], faces=[])


def test_a_script_run_in_the_sandbox_reads_the_asset_from_its_project_and_nothing_else(tmp_path, monkeypatch):
    monkeypatch.delenv("LITE_STEP_ASSET_ROOT", raising=False)
    h = put(tmp_path, "cube.ply", ply_ascii())
    monkeypatch.chdir(tmp_path)
    script = f'''
proj = Project(name="p")
site = Site(name="site")
e = Element(name="statue", ifc_class="IfcBuildingElementProxy")
e.add(Mesh(name="skin", asset="cube.ply", sha256="{h}"))
site.add(e)
proj.add(site)
result = proj
'''
    r = execute_lite_step_script_inproc(script)
    assert r.success, r.error
    blocked = execute_lite_step_script_inproc("result = open('cube.ply').read()")
    assert not blocked.success  # the script itself still reads nothing
    assert np.isfinite(len(r.project.sites))
