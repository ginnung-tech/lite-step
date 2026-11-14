"""Any solid can cut any solid: contour-frame boolean operands.

Two operand builders used to exist,
each with its own subset: the box/world-frame hosts could not be cut by a
contour ``Extrude``, and a contour host (``Extrude``, contour ``Wall`` body)
could be cut by box Solids only — read as the local AABB of two mapped
corners, which is a different box as soon as the host is tilted. The park
bench lost 147 of its 149 inferred carves that way, with a warning and a
carve report that listed all 149.

Now one builder (``_create_hole_solid_local``) takes the host frame as a
``HostFrame``. These tests measure the emitted geometry through the kernel
against a manifold3d reference made from the UNCUT solids, each compiled on
its own:

    expected = vol(host) - vol(host ∩ cutter)
"""
from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.geom  # noqa: E402,F811
manifold3d = pytest.importorskip("manifold3d")

from lite_step.compiler.executor import normalize_project_to_meters  # noqa: E402
from lite_step.ifc.generator import generate_ifc  # noqa: E402
from lite_step.models import (  # noqa: E402
    Bar, Box, Extrude, Material, Pipe, Point, Point2D, Project, Revolve,
    Sweep, Wall,
)
from lite_step.models.project import Storey  # noqa: E402

MM3 = 1e-9


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def _compile(*elements):
    p = Project(name="carve")
    s = Storey(name="ground")
    for e in elements:
        s.add(e)
    p.add_storey(s)
    res = generate_ifc(normalize_project_to_meters(p), source_code=None)
    assert res.success, res.error
    return res


def _product_mesh(ifc_text: str, name_fragment: str):
    """World-coordinate (verts, faces) of the ONE product whose Name contains
    ``name_fragment``."""
    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(ifc_text)
        model = ifcopenshell.open(path)
        hits = [p for p in model.by_type("IfcProduct")
                if p.Representation is not None
                and name_fragment in (p.Name or "")]
        assert len(hits) == 1, [p.Name for p in hits]
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        sh = ifcopenshell.geom.create_shape(settings, hits[0])
        v = np.array(sh.geometry.verts, dtype=np.float64).reshape(-1, 3)
        f = np.array(sh.geometry.faces, dtype=np.int64).reshape(-1, 3)
        return v, f
    finally:
        os.unlink(path)


def _volume(mesh) -> float:
    v, f = mesh
    t = v[f]
    return abs(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)


def _manifold(mesh):
    v, f = mesh
    # The kernel emits one vertex per face corner; weld them (to 1 µm) so the
    # mesh is a closed 2-manifold manifold3d will accept.
    v, inverse = np.unique(np.round(v, 6), axis=0, return_inverse=True)
    f = inverse.reshape(-1)[f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    m = manifold3d.Manifold(manifold3d.Mesh(
        vert_properties=v.astype(np.float32), tri_verts=f.astype(np.uint32)))
    assert m.status() == manifold3d.Error.NoError, m.status()
    return m


def _disk_reference_mesh(elem):
    """World mesh of a Pipe/Bar as the extruded-cylinder solid it becomes in a
    boolean (``_create_disk_solid_boolean_safe``), built in the WORLD frame.

    The standalone render is an IfcSweptDiskSolid, and the kernel tessellates
    that into a mesh that is neither closed nor full-volume (13 % short on a
    straight pipe), so it cannot be a reference. The frame mapping under test
    is not involved: this is ``frame=None``.
    """
    from lite_step.ifc import generator as gen
    from lite_step.ifc.entity_cache import EntityCache

    p = Project(name="ref")
    s = Storey(name="ground")
    s.add(elem)
    p.add_storey(s)
    e = normalize_project_to_meters(p).storeys[0].elements[0]
    model = ifcopenshell.file(schema="IFC4X3_ADD2")
    radius = e.radius if isinstance(e, Pipe) else e.diameter / 2.0
    fillet = e.fillet_radius if isinstance(e, Pipe) else e.bend_radius
    solid = gen._create_disk_solid_boolean_safe(model, e.path, radius, fillet,
                                                EntityCache(model))
    sh = ifcopenshell.geom.create_shape(ifcopenshell.geom.settings(), solid)
    return (np.array(sh.verts, dtype=np.float64).reshape(-1, 3),
            np.array(sh.faces, dtype=np.int64).reshape(-1, 3))


def _reference_mesh(elem, tag):
    if isinstance(elem, (Pipe, Bar)):
        return _disk_reference_mesh(elem)
    return _product_mesh(_compile(elem).ifc_content, tag)


def _cut_and_measure(make_host, make_cutter, host_tag):
    """(actual, expected, host_volume, removed) for host − cutter."""
    host_alone = _reference_mesh(make_host()[0], host_tag)
    cutter_alone = _reference_mesh(make_cutter(), "cutter")
    removed = (_manifold(host_alone) ^ _manifold(cutter_alone)).volume()
    host_vol = _volume(host_alone)
    expected = host_vol - removed

    root, body = make_host()
    body.difference(make_cutter())
    actual = _volume(_product_mesh(_compile(root).ifc_content, host_tag))
    return actual, expected, host_vol, removed


# ---------------------------------------------------------------------------
# Hosts and cutters (mm). Every pair overlaps partially.
# ---------------------------------------------------------------------------

P = Point


def _tilted_contour(c, u, v, a, b):
    """Rectangle c + s*u + t*v, s ∈ [0, a], t ∈ [0, b]; normal u × v."""
    pts = [c, c + a * u, c + a * u + b * v, c + b * v]
    return [P(x=int(round(q[0])), y=int(round(q[1])), z=int(round(q[2]))) for q in pts]


def _host_box():
    b = Box(name="host", start=P(x=0, y=0, z=0), end=P(x=1000, y=600, z=400))
    return b, b


def _host_extrude_flat():
    e = Extrude(name="host", thickness=400, contour=[
        P(x=0, y=0, z=0), P(x=1000, y=0, z=0), P(x=1000, y=600, z=0), P(x=0, y=600, z=0)])
    return e, e


def _host_extrude_tilted():
    # Normal (0, 0.6, 0.8): a roof plane.
    e = Extrude(name="host", thickness=300, contour=_tilted_contour(
        np.array([0.0, 0.0, 200.0]), np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.8, -0.6]), 1000, 600))
    return e, e


def _host_wall():
    w = Wall(name="host")
    body = Extrude(thickness=300, contour=[
        P(x=0, y=600, z=0), P(x=1000, y=600, z=0), P(x=1000, y=600, z=500),
        P(x=0, y=600, z=500)])
    w.add(body)
    return w, body


def _host_sweep():
    s = Sweep(name="host", path=[P(x=-100, y=300, z=200), P(x=1100, y=300, z=200)],
              material=Material(key="Timber_C24", profile_mm=(45, 220)))
    return s, s


def _host_pipe():
    p = Pipe(name="host", radius=180,
             path=[P(x=-100, y=300, z=200), P(x=1100, y=300, z=200)])
    return p, p


def _host_revolve():
    r = Revolve(name="host", angle=36000,
                profile=[Point2D(x=0, y=0), Point2D(x=400, y=0),
                         Point2D(x=400, y=400), Point2D(x=0, y=400)],
                path=[P(x=500, y=300, z=0), P(x=500, y=300, z=400)])
    return r, r


HOSTS = {
    "box": (_host_box, "host"),
    "extrude-flat": (_host_extrude_flat, "host"),
    "extrude-tilted": (_host_extrude_tilted, "host"),
    "wall-contour": (_host_wall, "host"),
    "sweep": (_host_sweep, "host"),
    "pipe": (_host_pipe, "host"),
    "revolve": (_host_revolve, "host"),
}


CUTTERS = {
    "box": lambda: Box(name="cutter", start=P(x=400, y=200, z=100),
                       end=P(x=700, y=900, z=700)),
    # Normal (0.6, 0, 0.8).
    "extrude-tilted": lambda: Extrude(name="cutter", thickness=150, contour=_tilted_contour(
        np.array([600.0, 100.0, 150.0]), np.array([0.0, 1.0, 0.0]),
        np.array([-0.8, 0.0, 0.6]), 400, 500)),
    "pipe-straight": lambda: Pipe(name="cutter", radius=80, path=[
        P(x=500, y=-200, z=200), P(x=500, y=800, z=260)]),
    "pipe-bent-fillet": lambda: Pipe(name="cutter", radius=60, fillet_radius=150, path=[
        P(x=300, y=-200, z=200), P(x=300, y=300, z=200), P(x=800, y=300, z=350)]),
    "bar": lambda: Bar(name="cutter", diameter=32, path=[
        P(x=700, y=-200, z=100), P(x=700, y=800, z=300)]),
    # One vertical segment, a non-square section: a wrong section roll or a
    # path mapped before segment_axes shows up as a different volume.
    "sweep-3seg": lambda: Sweep(name="cutter", path=[
        P(x=450, y=300, z=-300), P(x=450, y=300, z=250), P(x=650, y=300, z=250),
        P(x=650, y=800, z=250)],
        material=Material(key="Timber_C24", profile_mm=(45, 120))),
    # Partial turn about a horizontal axis: a wrong start plane moves the gap.
    "revolve-270": lambda: Revolve(name="cutter", angle=27000,
                                   profile=[Point2D(x=60, y=0), Point2D(x=350, y=0),
                                            Point2D(x=350, y=300), Point2D(x=60, y=300)],
                                   path=[P(x=350, y=300, z=200), P(x=650, y=300, z=200)]),
}


# ---------------------------------------------------------------------------
# T2 — the matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cutter", sorted(CUTTERS))
@pytest.mark.parametrize("host", sorted(HOSTS))
def test_every_cutter_cuts_every_host_exactly(host, cutter, caplog):
    make_host, tag = HOSTS[host]
    with caplog.at_level(logging.WARNING):
        actual, expected, host_vol, removed = _cut_and_measure(
            make_host, CUTTERS[cutter], tag)
    assert not [r for r in caplog.records
                if "unsupported boolean operand" in r.getMessage()]
    assert removed > 0.001 * host_vol, "fixture: the pair must overlap"
    assert removed < 0.99 * host_vol, "fixture: the overlap must be partial"
    # The cut happened, and it removed the cutter's shape — not its box, not
    # a rotated copy. The bound is relative to the REMOVED volume so a thin
    # bar that is silently skipped still fails. Measured: ≤ 0.6 % everywhere
    # except a Pipe/Bar side, where the kernel's evaluation of the mitered
    # cylinder union is off by up to 6.3 % — on the untouched box host as much
    # as on a rotated one, so it is the kernel, not the frame.
    disk = host == "pipe" or cutter in ("bar", "pipe-straight", "pipe-bent-fillet")
    tol = 0.07 if disk else 0.01
    assert abs(actual - expected) <= tol * removed, (
        f"{host} − {cutter}: kernel {actual / MM3:.0f} mm³, reference "
        f"{expected / MM3:.0f} mm³ (uncut {host_vol / MM3:.0f}, "
        f"removed {removed / MM3:.0f})")


# ---------------------------------------------------------------------------
# T3 — a tilted contour host and an axis-aligned box
# ---------------------------------------------------------------------------


def test_a_box_cuts_a_tilted_roof_as_a_box_not_as_its_local_aabb():
    """The old contour path mapped two corners and took their AABB in the
    roof frame: a tilted, bigger box. Removed volume must match the true
    intersection, which that box cannot."""
    actual, expected, host_vol, removed = _cut_and_measure(
        _host_extrude_tilted,
        lambda: Box(name="cutter", start=P(x=300, y=100, z=0),
                    end=P(x=600, y=300, z=600)),
        "host")
    assert removed > 0.02 * host_vol
    assert abs(actual - expected) <= 0.002 * host_vol


# ---------------------------------------------------------------------------
# T1 — the bench repro: a contour rib carves a contour slat (inferred)
# ---------------------------------------------------------------------------


def _slat_and_rib():
    # Slat: profile 90 × 30 in the YZ plane, extruded 1800 along +X.
    slat = Extrude(name="seat_slat", thickness=1800, contour=[
        P(x=0, y=0, z=400), P(x=0, y=90, z=400), P(x=0, y=90, z=430),
        P(x=0, y=0, z=430)])
    # Rib: a 40-thick plate in the YZ plane at x = 600, crossing the slat.
    rib = Extrude(name="seat_rib", thickness=40, contour=[
        P(x=600, y=-100, z=300), P(x=600, y=300, z=300), P(x=600, y=300, z=415),
        P(x=600, y=-100, z=415)])
    return slat, rib


def test_bench_rib_carves_the_slat(caplog, capsys):
    slat, rib = _slat_and_rib()
    with caplog.at_level(logging.WARNING):
        res = _compile(slat, rib)                 # rib added after: rib carves slat
    assert not [r for r in caplog.records
                if "unsupported boolean operand" in r.getMessage()]
    err = capsys.readouterr().err
    assert "carve emission: planned=1 emitted=1" in err, err

    v = _volume(_product_mesh(res.ifc_content, "seat_slat"))
    slat_vol = 1800 * 90 * 30 * MM3
    notch = 40 * 90 * 15 * MM3                    # rib reaches 15 mm into the slat
    assert abs(v - (slat_vol - notch)) / slat_vol < 0.005, v / MM3
    assert "IFCBOOLEANRESULT(.DIFFERENCE." in res.ifc_content

    slat_alone, rib_alone = _slat_and_rib()
    host = _product_mesh(_compile(slat_alone).ifc_content, "seat_slat")
    cut = _product_mesh(_compile(rib_alone).ifc_content, "seat_rib")
    ref = _volume(host) - (_manifold(host) ^ _manifold(cut)).volume()
    assert abs(v - ref) / slat_vol < 0.005


# ---------------------------------------------------------------------------
# T6 — an inferred carve that cannot be built fails the compile, by name
# ---------------------------------------------------------------------------


def test_an_inferred_carve_that_cannot_be_emitted_fails_the_compile():
    from lite_step.compiler import displacement as disp
    from lite_step.tests._fixtures import terrain_mesh

    host = Box(name="host", start=P(x=0, y=0, z=0), end=P(x=1000, y=600, z=400))
    p = Project(name="carve")
    s = Storey(name="ground")
    s.add(host)
    p.add_storey(s)
    proj = normalize_project_to_meters(p)
    disp.apply_displacement(proj)
    body = proj.storeys[0].elements[0]
    # A Mesh is never a boolean operand; tag it the way displacement tags an
    # inferred carve and the builder must refuse out loud, not skip.
    disp._carve_solid_host(body, terrain_mesh())

    res = generate_ifc(proj, source_code=None)
    assert not res.success
    assert "inferred carve" in res.error and "could not be emitted" in res.error
    assert disp._carve_label(body) in res.error


def test_an_authored_unsupported_operand_fails_the_compile_too():
    """#20: an authored ``.difference()`` with an operand no builder can
    realise is a compile error as well, naming the operand — the cut the
    author wrote is not in the file otherwise. Measured before making it
    fatal: 0 of 36 corpus models and 0 of 14 examples carry one."""
    from lite_step.tests._fixtures import terrain_mesh

    host = Box(name="host", start=P(x=0, y=0, z=0), end=P(x=1000, y=600, z=400))
    host.difference(terrain_mesh())
    p = Project(name="carve")
    s = Storey(name="ground")
    s.add(host)
    p.add_storey(s)
    res = generate_ifc(normalize_project_to_meters(p), source_code=None)
    assert not res.success
    assert "unsupported boolean operand Mesh" in res.error
    assert "A Mesh is carved, never carves" in res.error


def test_a_degenerate_box_operand_still_only_warns():
    """The falsification partner: a well-formed operand that removes nothing
    (under 1 mm thick) is the author's mistake to see, not a compile error."""
    host = Box(name="host", start=P(x=0, y=0, z=0), end=P(x=1000, y=600, z=400))
    host.difference(Box(start=P(x=100, y=100, z=100), end=P(x=100, y=300, z=300)))
    _compile(host)
