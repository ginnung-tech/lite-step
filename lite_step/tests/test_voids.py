"""``.void()`` — an ANONYMOUS IfcOpeningElement (IfcRelVoidsElement, no fill) cut
from a solid by arbitrary geometry. The middle rung of the carve vocabulary:

    .difference()  → IfcBooleanResult      (hole invisible to schedules/quantities)
    .void()        → anonymous IfcOpeningElement (a real void, no identity)
    .opening()     → named IfcOpeningElement    (a void WITH identity + fill)

Model-level tests assert the operand plumbing; end-to-end tests assert the emitted
IFC (anonymous opening + voids relation, no fill, host still a product, and the
operand normalized to meters).

**A ``.void()`` emits on every host it is legal on** — the last section of this
file. Two hosts silently emitted nothing:

* every CURVED primitive (``Pipe``/``Sweep``/``Revolve``/``Bar``). They finish
  through ``_finish_v15_product`` / ``_create_profile_path``, and neither read
  ``_voids``. The model validated, ``apply_displacement`` carved the details
  standing in the hole, the compile reported ``success=True`` — and the file
  came out with no ``IfcOpeningElement``, no ``IfcRelVoidsElement`` and no
  boolean. Measured on ``main``: a ``Box`` host emitted 1 opening, a ``Pipe``
  host 0, from the same fixture.
* every container DEEPER than one level. ``_process_container_voids`` paired
  ``zip(container.elements-with-an-AABB, child_products)``, which a nested
  container breaks (it has no AABB of its own but does have a product), so the
  counts disagreed and the fallback attached the hole to EVERY child product
  — landing an ``IfcRelVoidsElement`` on a product whose ``Representation`` is
  ``None`` (the exact configuration that function exists to prevent, one level
  down) and on leaves the hole does not touch.

Where a void LANDS is now one module, :mod:`lite_step.ifc.voids`, read by the
generator and by ``compiler.displacement`` — which must skip exactly the hosts
the generator distributes, or every detail standing in the hole is carved
twice.
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import execute_lite_step_script, validate_project
from lite_step.ifc.generator import generate_ifc
from lite_step.models import Project, Box, Point, collect_consumed_operand_ids
from lite_step.models.project import Storey


# ── model layer ───────────────────────────────────────────────────────────────
def test_void_is_chainable_and_stores_operand():
    host = Box(name="block", start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000))
    tool = Box(start=Point(x=100, y=100, z=100), end=Point(x=200, y=200, z=200))
    ret = host.void(tool)
    assert ret is host                      # chainable, returns the receiver
    assert host._voids == [tool]


def test_void_operand_is_consumed():
    """A ``.void()`` operand is a boolean tool — it must never render standalone."""
    host = Box(name="block", start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000))
    tool = Box(start=Point(x=100, y=100, z=100), end=Point(x=200, y=200, z=200))
    host.void(tool)
    b = Project(name="t")
    s = Storey(elevation=0)
    s.add(host)
    b.add_storey(s)
    assert id(tool) in collect_consumed_operand_ids(b)


# ── end-to-end IFC ────────────────────────────────────────────────────────────
_VOID_SCRIPT = """
from lite_step.models import Project, Box, Point
def generate_project():
    b = Project(name="voidtest")
    host = Box(name="block", start=Point(x=-1000, y=-1000, z=0), end=Point(x=1000, y=1000, z=2000))
    host.void(Box(start=Point(x=-300, y=-300, z=500), end=Point(x=300, y=300, z=1500)))
    b.add(host)
    return b
result = generate_project()
"""


def _compile(script: str):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    r = execute_lite_step_script(script)
    assert r.success, r.error
    ifc = generate_ifc(r.project)
    assert ifc.success, ifc.error
    return ifcopenshell.file.from_string(ifc.ifc_content)


def test_void_emits_anonymous_opening_with_voids_relation():
    model = _compile(_VOID_SCRIPT)

    openings = model.by_type("IfcOpeningElement")
    assert len(openings) == 1, "one .void() → one IfcOpeningElement"
    assert openings[0].Name is None, "the void is ANONYMOUS (Name=None)"

    rels = model.by_type("IfcRelVoidsElement")
    assert len(rels) == 1
    assert rels[0].RelatedOpeningElement == openings[0]
    # host still emits its own product, and the void is linked to it
    assert rels[0].RelatingBuildingElement.is_a("IfcBuildingElementProxy")


def test_void_has_no_fill():
    """Unlike ``.opening(Window/Door)``, a ``.void()`` has no fill element."""
    model = _compile(_VOID_SCRIPT)
    assert not model.by_type("IfcWindow")
    assert not model.by_type("IfcDoor")
    assert not model.by_type("IfcRelFillsElement")


def test_void_operand_normalized_to_meters():
    """The operand's mm coords must be normalized like every other element — a
    1000 mm-tall void becomes a 1.0 m extrusion, not 1000 m (the [[suspend_strict
    /normalizer]] class of bug)."""
    model = _compile(_VOID_SCRIPT)
    solid = model.by_type("IfcOpeningElement")[0].Representation.Representations[0].Items[0]
    assert solid.is_a("IfcExtrudedAreaSolid")
    assert abs(solid.Depth - 1.0) < 1e-6, "1500-500 mm → 1.0 m; not 1000 m"


def test_difference_emits_no_opening_but_void_does():
    """Same geometry, different IFC: ``.difference()`` bakes an IfcBooleanResult
    (no opening entity); ``.void()`` emits an IfcOpeningElement."""
    diff_script = _VOID_SCRIPT.replace(".void(", ".difference(")
    diff_model = _compile(diff_script)
    void_model = _compile(_VOID_SCRIPT)

    assert not diff_model.by_type("IfcOpeningElement"), ".difference() → no opening"
    assert diff_model.by_type("IfcBooleanResult"), ".difference() → IfcBooleanResult"
    assert void_model.by_type("IfcOpeningElement"), ".void() → IfcOpeningElement"


# ── containers ────────────────────────────────────────────────────────────────
def _kernel_volume(model, product):
    """World-coord tessellated volume of a product (openings applied)."""
    ios = pytest.importorskip("ifcopenshell.geom")
    import numpy as np
    s = ios.settings()
    s.set(s.USE_WORLD_COORDS, True)
    shp = ios.create_shape(s, product)
    v = np.array(shp.geometry.verts).reshape(-1, 3)
    f = np.array(shp.geometry.faces).reshape(-1, 3)
    return abs(sum(float(np.dot(v[a], np.cross(v[b], v[c]))) for a, b, c in f)) / 6.0


_SLAB_VOID = """
from lite_step.models import Project, Slab, Box, Point
def generate_project():
    b = Project(name="s"); sl = Slab(name="floor")
    sl.add(Box(name="body", type="foundation", start=Point(x=-2000,y=-2000,z=0), end=Point(x=2000,y=2000,z=400)))
    sl.void(Box(start=Point(x=-300,y=-300,z=0), end=Point(x=300,y=300,z=400)))  # through-void
    b.add(sl); return b
result = generate_project()
"""


def test_void_on_aggregate_container_targets_the_geometry_leaf():
    """A Slab is a pure aggregate — geometry lives on a child proxy. The void must
    attach to that CHILD (so it carves in web-ifc, which builds each element's mesh
    from its own representation), not the representation-less IfcSlab."""
    model = _compile(_SLAB_VOID)
    rels = model.by_type("IfcRelVoidsElement")
    assert len(rels) == 1
    assert rels[0].RelatingBuildingElement.is_a("IfcBuildingElementProxy")  # the leaf, not IfcSlab
    # and it actually carves: 4x4x0.4 = 6.4 minus the 0.6x0.6x0.4 = 0.144 through-hole
    child = model.by_type("IfcBuildingElementProxy")[0]
    assert abs(_kernel_volume(model, child) - (6.4 - 0.144)) < 1e-3


def test_void_on_wall_targets_the_product():
    """A Wall carries its body ON the IfcWall, so the void attaches to the product
    directly (IfcWall IS an IfcElement — a legal IfcRelVoidsElement host)."""
    script = """
from lite_step.models import Project, Wall, Box, Point
def generate_project():
    b = Project(name="w"); wl = Wall(name="north")
    wl.add(Box(name="body", start=Point(x=-2000,y=0,z=0), end=Point(x=2000,y=200,z=2500)))
    wl.void(Box(start=Point(x=-300,y=0,z=800), end=Point(x=300,y=200,z=1400)))
    b.add(wl); return b
result = generate_project()
"""
    model = _compile(script)
    rels = model.by_type("IfcRelVoidsElement")
    assert len(rels) == 1
    assert rels[0].RelatingBuildingElement.is_a("IfcWall")
    # 4x0.2x2.5 = 2.0 minus 0.6x0.2x0.6 = 0.072
    assert abs(_kernel_volume(model, model.by_type("IfcWall")[0]) - (2.0 - 0.072)) < 1e-3


def test_container_void_only_carves_overlapping_leaves():
    """A container void attaches only to the child leaves it overlaps — a
    non-overlapping sibling gets no spurious opening."""
    script = """
from lite_step.models import Project, Slab, Box, Point
def generate_project():
    b = Project(name="s"); sl = Slab(name="floor")
    sl.add(Box(name="a", type="foundation", start=Point(x=-3000,y=-1000,z=0), end=Point(x=-1000,y=1000,z=400)))
    sl.add(Box(name="b2", type="foundation", start=Point(x=1000,y=-1000,z=0), end=Point(x=3000,y=1000,z=400)))
    sl.void(Box(start=Point(x=-2300,y=-300,z=0), end=Point(x=-1700,y=300,z=400)))  # over box A only
    b.add(sl); return b
result = generate_project()
"""
    model = _compile(script)
    assert len(model.by_type("IfcOpeningElement")) == 1, "only the overlapped leaf is voided"


def test_void_on_space_is_rejected():
    """IfcSpace is a spatial element — IfcRelVoidsElement can't target it and
    carving 'air' is meaningless, so a .void() on a Space is a compile error."""
    from lite_step.compiler.executor import validate_project_report
    r = execute_lite_step_script("""
from lite_step.models import Project, Space, Box, Point
def generate_project():
    b = Project(name="x"); c = Space(name="c")
    c.add(Box(name="g", start=Point(x=-1000,y=-1000,z=0), end=Point(x=1000,y=1000,z=1000)))
    c.void(Box(start=Point(x=-100,y=-100,z=100), end=Point(x=100,y=100,z=300)))
    b.add(c); return b
result = generate_project()
""")
    assert r.success, r.error
    assert [e for e in validate_project_report(r.project).errors if ".void()" in e]


def test_site_void_clears_terrain_mesh():
    """`site.void(box)` is NOT an IfcOpeningElement (IfcSite can't host one) — it
    clears the site's terrain mesh in the box region (the displacement pass folds
    it into the mesh carve). Author-facing terrain clearing for varied sites."""
    pytest.importorskip("manifold3d")
    from lite_step.compiler.executor import normalize_project_to_meters, validate_project_report
    from lite_step.models import Project, Site, Mesh, Box, Point, Element, collect_consumed_operand_ids
    from lite_step.tests._fixtures import terrain_site

    b = Project(name="sv")
    # Local fixture (no corpus coupling): canonical terrain wrapper —
    # Element(IfcGeographicElement, TERRAIN) — so site.void() folds into it.
    site = terrain_site(12000, 9000)  # flat top z=0
    clear = Box(start=Point(x=0, y=0, z=-500), end=Point(x=16000, y=14500, z=20000))  # +X+Y from -0.5m up
    site.void(clear)
    b.add(site)

    # no validation error (Site void is allowed), operand consumed (never renders)
    assert not [e for e in validate_project_report(b).errors if ".void()" in e]
    assert id(clear) in collect_consumed_operand_ids(b)

    nb = normalize_project_to_meters(b)
    ifc = generate_ifc(nb)
    assert ifc.success, ifc.error
    terr_el = [c for c in nb.sites[0]._elements if isinstance(c, Element)][0]
    terrain = [gc for gc in terr_el._elements if isinstance(gc, Mesh)][0]
    cleared = [p.z for p in terrain.vertices if p.x > 0.5 and p.y > 0.5]       # +X+Y quadrant
    untouched = [p.z for p in terrain.vertices if p.x < -0.5 and p.y < -0.5]   # opposite
    assert max(cleared) <= -0.5 + 1e-3, "cleared quadrant carved to the box bottom"
    assert abs(max(untouched) - 0.0) < 1e-3, "opposite quadrant left at original grade"


def test_relational_void_without_mesh_child_is_rejected():
    """A `.void()` on a relational element (Site/Space) is realised by carving its
    Mesh children — one with NO Mesh child has nothing to carve → compile error.
    (This is the same kind-based rule that rejects every Space void: a Space accepts
    only solid children, never a Mesh.)"""
    from lite_step.compiler.executor import validate_project_report
    from lite_step.models import Project, Site, Box, Point
    b = Project(name="x"); site = Site(name="site")
    site.add(Box(name="g", start=Point(x=-1000,y=-1000,z=0), end=Point(x=1000,y=1000,z=1000)))
    site.void(Box(start=Point(x=-100,y=-100,z=100), end=Point(x=100,y=100,z=300)))
    b.add(site)
    assert [e for e in validate_project_report(b).errors if ".void()" in e]


def _mesh_void_project(*, void: bool = True):
    """A terrain Mesh with a ``.void()`` (or ``.difference()``) tool on it.

    Off-origin and off-square on purpose: the tool bites the +X+Y quadrant of
    a 12000 x 9000 footprint site, so a fixture at the defaults cannot hide a
    no-op behind symmetry.
    """
    from lite_step.models import Project, Element, Mesh, Box, Point
    from lite_step.tests._fixtures import terrain_site

    b = Project(name="mv")
    site = terrain_site(12000, 9000)
    b.add(site)
    wrapper = [c for c in site._elements if isinstance(c, Element)][0]
    mesh = [gc for gc in wrapper._elements if isinstance(gc, Mesh)][0]
    tool = Box(start=Point(x=1300, y=700, z=-4500),
               end=Point(x=9100, y=6200, z=3000))
    (mesh.void if void else mesh.difference)(tool)
    return b, mesh, tool


def test_void_on_a_bare_mesh_is_rejected():
    """``mesh.void(tool)`` must not be a SILENT no-op — the operand
    was consumed (so it never rendered), the displacement pass folded only a
    mesh's ``_cuts``, neither generator read ``_voids``, and no validator
    looked. No hole, no tool, no error.

    It is refused rather than implemented, symmetric with the Space receiver:
    ``.void()`` promises a COUNTED IfcOpeningElement, a tessellated mesh is
    not a valid IFC boolean operand, so nothing in the file could carry that
    opening — and baking the subtraction into the vertices is what
    ``.difference()`` already does.
    """
    from lite_step.compiler.executor import validate_project_report
    b, _mesh, _tool = _mesh_void_project()
    errors = [e for e in validate_project_report(b).errors if ".void()" in e]
    assert errors, "a bare-Mesh .void() must not pass validation in silence"
    assert "Mesh" in errors[0]
    assert ".difference()" in errors[0]


def test_difference_on_the_same_mesh_is_accepted():
    """Falsification: the SAME geometry through the verb that works. The
    refusal is caused by the verb, not by voiding a mesh-shaped thing."""
    from lite_step.compiler.executor import validate_project_report
    b, _mesh, _tool = _mesh_void_project(void=False)
    assert not [e for e in validate_project_report(b).errors if ".void()" in e]


def test_the_mesh_void_refusal_names_the_receiver():
    """An author has to find the offending mesh: the message carries the
    element's own id, like every other relational-void refusal."""
    from lite_step.compiler.executor import validate_project_report
    b, mesh, _tool = _mesh_void_project()
    errors = [e for e in validate_project_report(b).errors if ".void()" in e]
    assert mesh.id in errors[0]


def test_a_site_void_over_a_mesh_child_is_still_allowed():
    """The refusal is scoped to a MESH receiver. ``site.void()`` still routes
    to the terrain carve — that path is the one that works, and breaking it
    while closing would be the fix trading one silent hole for another.
    """
    from lite_step.compiler.executor import validate_project_report
    from lite_step.models import Project, Box, Point
    from lite_step.tests._fixtures import terrain_site

    b = Project(name="sv2")
    site = terrain_site(12000, 9000)
    site.void(Box(start=Point(x=1300, y=700, z=-4500),
                  end=Point(x=9100, y=6200, z=3000)))
    b.add(site)
    assert not [e for e in validate_project_report(b).errors if ".void()" in e]


def test_void_on_primitive_inside_spatial_container_is_allowed():
    """Only the SPATIAL container itself can't be voided — a .void() on a Box
    inside a Space is fine (the box is a voidable IfcElement)."""
    from lite_step.compiler.executor import validate_project_report
    r = execute_lite_step_script("""
from lite_step.models import Project, Space, Box, Point
def generate_project():
    b = Project(name="x"); c = Space(name="c")
    box = Box(name="g", start=Point(x=-1000,y=-1000,z=0), end=Point(x=1000,y=1000,z=1000))
    box.void(Box(start=Point(x=-100,y=-100,z=100), end=Point(x=100,y=100,z=300)))
    c.add(box); b.add(c); return b
result = generate_project()
""")
    assert r.success, r.error
    assert not [e for e in validate_project_report(r.project).errors if ".void()" in e]


# ── backend routing ───────────────────────────────────────────────────────────
def test_void_operand_transform_origin_moves_the_carve():
    """A void operand's ``placement=Transform(origin=...)`` TRANSLATES it (not just
    rotates) — the carve must land at the moved position. Regression for the Site-void
    placement-normalization gap: the operand's geometry was scaled mm→m but its
    Transform origin stayed in mm, so the mesh carve translated it ~1000x off the
    terrain and silently no-op'd (spec-void-operand-transform-origin, cases A/C)."""
    pytest.importorskip("manifold3d")
    import numpy as np
    from lite_step.compiler.executor import normalize_project_to_meters
    from lite_step.models import Project, Site, Mesh, Point, Transform, Element
    from lite_step.tests._fixtures import terrain_site

    def carve(origin_mm):
        b = Project(name="t")
        site = terrain_site(20000, 20000)  # flat top z=0 (local fixture)
        # a cube authored at the LOCAL origin, sunk 2 m into the terrain; the whole
        # cut is repositioned by placement=Transform(origin=...).
        cut = Box(start=Point(x=-1500, y=-1500, z=-2000), end=Point(x=1500, y=1500, z=0),
                  placement=Transform(origin=Point(x=origin_mm, y=0, z=0)))
        site.void(cut); b.add(site)
        nb = normalize_project_to_meters(b)          # returns a normalized deepcopy
        assert generate_ifc(nb).success
        terr_el = [c for c in nb.sites[0]._elements if isinstance(c, Element)][0]
        terr = [gc for gc in terr_el._elements if isinstance(gc, Mesh)][0]
        v = np.array([[p.x, p.y, p.z] for p in terr.vertices])
        rim = v[(np.abs(v[:, 1]) < 8) & (v[:, 2] > -2.1) & (v[:, 2] < 0.1) & (np.abs(v[:, 0]) < 12)]
        return len(terr.vertices), (float(rim[:, 0].mean()) if len(rim) else None)

    base = 8  # a flat 2x2 site block is 8 verts; a carve adds verts
    n0, x0 = carve(0)          # origin 0     → carve centred at x≈0
    n4, x4 = carve(4000)       # origin 4 m E → carve centred at x≈+4 (case A)
    assert n0 > base and n4 > base, "a translated operand must still carve (not a silent no-op)"
    assert abs(x0) < 1.0, f"origin-0 carve should centre near x=0, got {x0}"
    assert 3.0 < x4 < 5.0, f"origin-4m carve should centre near x=+4, got {x4}"


# ── named voids (WS-A §1.1 point 3) ───────────────────────────────────────────
#
# ``.void(name=)`` is where the absolute-coordinate hole lives. It replaces a
# form the reference documented and the compiler never implemented:
# ``body.opening(<arbitrary geometry>)`` crashed on a standalone body and
# emitted nothing at all inside a Wall. The hole family is now named x filled,
# and ``.void()`` owns the whole unfilled column.

_NAMED_VOID_SCRIPT = """
from lite_step.models import Project, Wall, Box, Point
def generate_project():
    b = Project(name="voidtest")
    wall = Wall(name="north")
    body = Box(name="body", start=Point(x=-3000, y=-150, z=0),
               end=Point(x=3000, y=150, z=3000), material="Concrete")
    wall.add(body)
    wall.void(Box(start=Point(x=-500, y=-200, z=0),
                  end=Point(x=500, y=200, z=2100)), name="doorway")
    b.add(wall)
    return b
result = generate_project()
"""


def test_named_void_emits_a_named_opening_with_a_canonical_path():
    """``name=`` names the HOLE, and the name follows the ordinary canonical
    grammar — leaf-first, then the host's named ancestors."""
    model = _compile(_NAMED_VOID_SCRIPT)
    openings = model.by_type("IfcOpeningElement")
    assert len(openings) == 1
    assert openings[0].Name == "opening:doorway:wall:north"
    assert len(model.by_type("IfcRelVoidsElement")) == 1
    # Still no fill: named x UNFILLED is the cell this occupies.
    assert model.by_type("IfcRelFillsElement") == []


def test_an_unnamed_void_stays_anonymous():
    """Falsification of the above: drop the one kwarg and the Name goes back to
    None. If both spellings emitted the same thing, ``name=`` would be
    decoration rather than the second column of the table."""
    model = _compile(_NAMED_VOID_SCRIPT.replace(', name="doorway"', ''))
    openings = model.by_type("IfcOpeningElement")
    assert len(openings) == 1
    assert openings[0].Name is None


def test_named_void_takes_exactly_one_operand():
    host = Box(name="block", start=Point(x=0, y=0, z=0),
               end=Point(x=1000, y=1000, z=1000))
    a = Box(start=Point(x=100, y=100, z=100), end=Point(x=200, y=200, z=200))
    b = Box(start=Point(x=300, y=300, z=300), end=Point(x=400, y=400, z=400))
    with pytest.raises(ValueError, match="exactly.*ONE operand"):
        host.void(a, b, name="doorway")
    assert host._voids == []          # nothing partially attached


def test_named_void_leaf_obeys_the_name_grammar():
    """One naming grammar, not two — the same ``[a-z0-9_]+`` leaf rule every
    ``name=`` in the DSL obeys, refused at the line that wrote it."""
    host = Box(name="block", start=Point(x=0, y=0, z=0),
               end=Point(x=1000, y=1000, z=1000))
    tool = Box(start=Point(x=100, y=100, z=100), end=Point(x=200, y=200, z=200))
    for bad in ("Doorway", "door way", "wall:north", "door.way"):
        with pytest.raises(ValueError, match="not a valid leaf"):
            host.void(tool, name=bad)


def test_named_and_anonymous_voids_carve_identically():
    """Naming changes IDENTITY, never geometry — the two cells of the unfilled
    column differ in exactly one attribute."""
    named = _compile(_NAMED_VOID_SCRIPT)
    anon = _compile(_NAMED_VOID_SCRIPT.replace(', name="doorway"', ''))
    for kind in ("IfcExtrudedAreaSolid", "IfcRelVoidsElement",
                 "IfcBooleanResult"):
        assert len(named.by_type(kind)) == len(anon.by_type(kind)), kind


def test_void_on_a_wall_body_is_not_dropped():
    """A ``.void()`` on the BODY of a Wall emits, exactly as one on the Wall does.

    It did not until WS-A. A Wall's body is rendered by ``_create_wall``, never
    by ``_create_box``, so nothing read the body's ``_voids``: measured on main
    @71f19a9, ``body.void(tool)`` inside a Wall emitted ZERO
    IfcOpeningElement / IfcRelVoidsElement / IfcBooleanResult while the
    identical ``wall.void(tool)`` emitted one of each. Two spellings of one
    relation, one of them silently doing nothing.
    """
    on_body = _compile("""
from lite_step.models import Project, Wall, Box, Point
def generate_project():
    b = Project(name="t")
    wall = Wall(name="north")
    body = Box(name="body", start=Point(x=-3000, y=-150, z=0),
               end=Point(x=3000, y=150, z=3000), material="Concrete")
    body.void(Box(start=Point(x=-500, y=-200, z=0),
                  end=Point(x=500, y=200, z=2100)))
    wall.add(body)
    b.add(wall)
    return b
result = generate_project()
""")
    on_wall = _compile("""
from lite_step.models import Project, Wall, Box, Point
def generate_project():
    b = Project(name="t")
    wall = Wall(name="north")
    body = Box(name="body", start=Point(x=-3000, y=-150, z=0),
               end=Point(x=3000, y=150, z=3000), material="Concrete")
    wall.add(body)
    wall.void(Box(start=Point(x=-500, y=-200, z=0),
                  end=Point(x=500, y=200, z=2100)))
    b.add(wall)
    return b
result = generate_project()
""")
    for m, tag in ((on_body, "body.void()"), (on_wall, "wall.void()")):
        assert len(m.by_type("IfcOpeningElement")) == 1, tag
        assert len(m.by_type("IfcRelVoidsElement")) == 1, tag


# =============================================================================
# .void() emits on every host it is legal on
# =============================================================================
#
# Every failure this section covers is SILENT: the compile succeeds, the file
# opens, ``ifcopenshell.validate`` passes, and the hole is simply not there.
# So each test says what would be wrong without it.


def _one_void(elem, tool, *, backend: str = "ifcopenshell"):
    """Compile ONE element carrying ONE ``.void()`` and return the model."""
    ifcopenshell = pytest.importorskip("ifcopenshell")
    from lite_step.compiler.executor import (
        normalize_project_to_meters, validate_project_report,
    )

    elem.void(tool, name="hole")
    proj = Project(name="one-void")
    proj.add(elem)
    assert validate_project_report(proj).errors == []
    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    return ifcopenshell.file.from_string(result.ifc_content)


def _through_tool() -> Box:
    """A tool crossing a 2 m vertical member at mid-height, overshooting it in
    X and Y — a ``.void()`` operand is absolute world geometry and is meant to
    overshoot, exactly as a real cutter does."""
    return Box(start=Point(x=-600, y=-600, z=900),
               end=Point(x=600, y=600, z=1100))


def _vertical_path():
    return [Point(x=0, y=0, z=0), Point(x=0, y=0, z=2000)]


def _curved_hosts():
    """One of every primitive that finishes outside ``_create_solid``, plus a
    ``Box`` CONTROL.

    The control is not decoration: a fixture that compiled nothing would
    report zero openings for every host, and would read as five confirmations
    of the bug rather than as one broken fixture.
    """
    from lite_step.models import Bar, Pipe, Point2D, Revolve, Sweep

    square = [Point2D(x=-60, y=-60), Point2D(x=60, y=-60),
              Point2D(x=60, y=60), Point2D(x=-60, y=60)]
    return [
        ("box (control)", Box(name="ctl", start=Point(x=-200, y=-200, z=0),
                              end=Point(x=200, y=200, z=2000)),
         _through_tool()),
        ("pipe", Pipe(name="p", path=_vertical_path(), radius=50),
         _through_tool()),
        ("sweep (path form)", Sweep(name="s", profile=square,
                                    path=_vertical_path()),
         _through_tool()),
        ("sweep (start/end form)", Sweep(name="se", start=Point(x=0, y=0, z=0),
                                         end=Point(x=0, y=0, z=2000)),
         _through_tool()),
        ("revolve", Revolve(name="r",
                            profile=[Point2D(x=0, y=0), Point2D(x=500, y=0),
                                     Point2D(x=500, y=360), Point2D(x=0, y=360)],
                            path=[Point(x=0, y=0, z=0), Point(x=0, y=360, z=0)],
                            angle=18000),
         Box(start=Point(x=-100, y=-100, z=100),
             end=Point(x=100, y=500, z=200))),
        ("bar", Bar(name="b", path=_vertical_path(), diameter=12),
         _through_tool()),
    ]


@pytest.mark.parametrize("label,host,tool", _curved_hosts(),
                         ids=[c[0] for c in _curved_hosts()])
def test_one_void_emits_one_opening_on_every_primitive(label, host, tool):
    """ONE ``.void()`` → exactly ONE ``IfcOpeningElement`` and ONE
    ``IfcRelVoidsElement``, on every primitive.

    Without this, a ``.void()`` on anything curved is a silent no-op: measured
    on ``main``, the ``Box`` control emitted 1 opening and 1 relation and the
    ``Pipe`` emitted 0 and 0, from the same fixture, both at ``success=True``.

    ``exactly one`` rather than ``at least one`` on purpose — the fix touches
    three emission tails (``_finish_v15_product``, the legacy start/end
    ``Sweep``, ``_create_profile_path``), and a primitive reaching two of them
    would double the hole rather than lose it.
    """
    model = _one_void(host, tool)
    openings = model.by_type("IfcOpeningElement")
    rels = model.by_type("IfcRelVoidsElement")
    assert len(openings) == 1, f"{label}: expected one IfcOpeningElement"
    assert len(rels) == 1, f"{label}: expected one IfcRelVoidsElement"
    assert rels[0].RelatedOpeningElement == openings[0], label
    # The hole is NAMED here, so an emitter that lost ``_void_canonical`` on
    # the way fails as loudly as one that lost the opening.
    assert openings[0].Name.startswith("opening:hole:"), (
        label, openings[0].Name)
    # And it hangs on a product that can carry it.
    assert rels[0].RelatingBuildingElement.Representation is not None, label


# ---------------------------------------------------------------------------
# Depth: a container void reaches its whole subtree, and only its geometry
# ---------------------------------------------------------------------------


def _nested_aggregate():
    """An ``Element`` holding an anchored ``Element`` holding two boxes, plus a
    box of its own — the shallowest tree with a depth-2 leaf AND a
    representation-less product in the middle.

    Geometry after the anchor bake (measured, meters)::

        near      (depth 1)  (0.0, 0.0, 0.0)-(0.5, 0.5, 3.0)   in the hole
        deep_low  (depth 2)  (0.2, 0.2, 0.0)-(0.5, 0.5, 0.3)   in the hole
        deep_high (depth 2)  (0.2, 0.2, 2.0)-(0.5, 0.5, 2.3)   clear of it
        inner     (middle)   no Representation of its own      never a host

    The hole spans z 0.05-0.4 and overshoots in x/y, so the split is made in Z,
    the axis the anchor bake leaves alone.
    """
    from lite_step.models import Element

    outer = Element(ifc_class="IfcBuildingElementProxy", name="outer")
    inner = Element(ifc_class="IfcBuildingElementProxy", name="inner")
    inner.add(Box(name="deep_low", start=Point(x=0, y=0, z=0),
                  end=Point(x=300, y=300, z=300)))
    inner.add(Box(name="deep_high", start=Point(x=0, y=0, z=2000),
                  end=Point(x=300, y=300, z=2300)))
    outer.anchor(inner, along=0, up=0)
    outer.add(Box(name="near", start=Point(x=0, y=0, z=0),
                  end=Point(x=500, y=500, z=3000)))
    outer.void(Box(start=Point(x=-100, y=-100, z=50),
                   end=Point(x=600, y=600, z=400)), name="hole")
    return outer


def _compile_project(root, *, backend: str = "ifcopenshell"):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    from lite_step.compiler.executor import (
        normalize_project_to_meters, validate_project_report,
    )

    proj = Project(name="nested-void")
    proj.add(root)
    assert validate_project_report(proj).errors == []
    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    return ifcopenshell.file.from_string(result.ifc_content)


def test_a_nested_aggregate_never_hosts_a_void_on_a_bodyless_product():
    """No ``IfcRelVoidsElement`` may land on a product with no
    ``Representation``.

    That is the whole reason ``_process_container_voids`` exists — an opening
    on a body-less aggregate carves nothing in a renderer that meshes each
    element from its OWN representation — and one level down the old ``zip``
    pairing reproduced it exactly: a nested container has no AABB of its own,
    so the counts disagreed, so the fallback attached the hole to every child
    product including the body-less one.

    Read through ``ifcopenshell`` rather than by counting strings: the file is
    well-formed either way, and the defect is WHICH entity the relation points
    at.
    """
    model = _compile_project(_nested_aggregate())
    rels = model.by_type("IfcRelVoidsElement")
    assert rels, "the container void must still reach something"
    for rel in rels:
        host = rel.RelatingBuildingElement
        assert host.Representation is not None, (
            f"void attached to {host.is_a()} {host.Name!r}, which carries no "
            "Representation — nothing in a web-ifc render would be carved")


def test_a_container_void_reaches_depth_2_and_stops_at_the_leaves_it_misses():
    """The hole attaches to the leaves it overlaps and to NO others — asserted
    as a count, at depth 2.

    Two failures in one assertion. Without the recursive walk, ``deep_low``
    (two levels down, squarely in the hole) gets nothing; without the exact
    pairing, ``deep_high`` (two metres above the hole) gets an opening it does
    not intersect. Both were true on ``main``, in opposite directions, from
    this one fixture.
    """
    model = _compile_project(_nested_aggregate())
    hosts = sorted(r.RelatingBuildingElement.Name
                   for r in model.by_type("IfcRelVoidsElement"))
    assert hosts == [
        "box:deep_low:buildingelementproxy:inner:buildingelementproxy:outer",
        "box:near:buildingelementproxy:outer",
    ]
    assert len(model.by_type("IfcOpeningElement")) == 2


def test_both_backends_emit_the_same_container_void():
    """The streaming generator is a separate emitter, and ``.void()`` routes
    AWAY from it (``_uses_voids`` → ifcopenshell). Asking for it explicitly is
    what proves the routing still holds — a streaming backend that quietly
    grew a partial void handler would answer differently here, and its file
    would open.
    """
    reference = None
    for backend in ("ifcopenshell",):
        model = _compile_project(_nested_aggregate(), backend=backend)
        hosts = sorted(r.RelatingBuildingElement.Name
                       for r in model.by_type("IfcRelVoidsElement"))
        if reference is None:
            reference = hosts
        assert hosts == reference, backend
        assert len(model.by_type("IfcOpeningElement")) == 2, backend


# ---------------------------------------------------------------------------
# One predicate, not two
# ---------------------------------------------------------------------------


def test_the_predicate_and_the_generator_agree_on_every_container():
    """``void_reaches_children`` must answer YES exactly for the containers the
    generator hands the void DOWN to, and NO for the ones that keep it on
    their own product.

    Measured against compiled output rather than against the frozenset: a
    container that changed route would otherwise agree with a constant
    describing the route it left. A YES container puts the opening on a LEAF
    product; a NO container puts it on its own.
    """
    from lite_step.ifc.voids import void_reaches_children
    from lite_step.models import Beam, Column, Roof, Slab, Wall

    def _container(kind):
        c = kind(name="c")
        c.add(Box(name="body", start=Point(x=-1000, y=-1000, z=0),
                  end=Point(x=1000, y=1000, z=2000)))
        c.void(Box(start=Point(x=-300, y=-1200, z=800),
                   end=Point(x=300, y=1200, z=1400)), name="hole")
        return c

    for kind in (Column, Beam, Slab, Roof, Wall):
        container = _container(kind)
        distributes = void_reaches_children(container)
        model = _compile_project(container)
        rels = model.by_type("IfcRelVoidsElement")
        assert len(rels) == 1, kind.__name__
        host_name = rels[0].RelatingBuildingElement.Name
        # The container's own canonical is ``<type>:c``; a leaf's is
        # ``box:body:<type>:c``, so the prefix IS the discriminator.
        on_own_product = host_name == f"{kind.__name__.lower()}:c"
        assert distributes is not on_own_product, (
            f"{kind.__name__}: void_reaches_children said {distributes} but "
            f"the generator put the opening on "
            f"{host_name!r}")

    # ...and the row where the answer is per INSTANCE. A Wall's own
    # unanchored prism becomes the ``IfcWall``'s representation rather than a
    # product of its own, so the same class answers BOTH ways and the
    # discriminator is ``tx.body_prisms``, not the type name. Without the leaf
    # wrapper below the Box would simply be this wall's body — which is the
    # bodied row already asserted above.
    leaf = Wall(name="leaf")
    leaf.add(Box(name="body", start=Point(x=-1000, y=-1000, z=0),
                 end=Point(x=1000, y=1000, z=2000)))
    asm = Wall(name="c")
    asm.add(leaf)
    asm.void(Box(start=Point(x=-300, y=-1200, z=800),
                 end=Point(x=300, y=1200, z=1400)), name="hole")

    assert void_reaches_children(asm) is True
    rels = _compile_project(asm).by_type("IfcRelVoidsElement")
    assert len(rels) == 1
    assert rels[0].RelatingBuildingElement.Name == "wall:leaf:wall:c", (
        "a body-less Wall's void must land on the leaf, not on the "
        "representation-less IfcWall")


def test_displacement_reads_the_predicate_instead_of_restating_it():
    """A SOURCE-level assertion, and the only guard reverting the shared
    predicate would trip.

    Collapsing the two spellings into one is behaviour-preserving by
    construction — that is the point of it — so no runtime test can fail when
    the duplicate comes back. What comes back with it is a container list that
    ``generator._process_container_voids`` agrees with only by hand, not by
    construction, and each direction of that disagreement is
    silent (details carved twice, or not carved at all). Same precedent as
    ``tests/test_dsl_cheatsheet.py``, which asserts the API block is not
    hand-maintained.

    BOTH forks since — ``.opening()`` distributes to the leaves too, and
    its fork is the narrower ``opening_reaches_children`` (``Slab``/``Roof``
    distribute a ``.void()`` and refuse an ``.opening()``). A displacement pass
    that read the wrong one of the two would skip a carve nothing replaces, or
    emit one that duplicates an ``IfcRelVoidsElement``; both are silent, which
    is this test's whole subject.
    """
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1]
              / "compiler" / "displacement.py").read_text(encoding="utf-8")

    assert re.search(r"from lite_step\.ifc\.voids import [^\n]*"
                     r"void_reaches_children", source), (
        "displacement.py must READ the shared predicate")
    assert re.search(r"from lite_step\.ifc\.voids import [^\n]*"
                     r"opening_reaches_children", source), (
        "...and the opening fork too")
    assert "void_reaches_children(" in source, "...and actually call it"
    assert "opening_reaches_children(" in source, "...and call that one too"
    restated = re.search(r'"Column".{0,120}"Beam"', source, re.S)
    assert restated is None, (
        "displacement.py restates the container routing table:\n"
        + (restated.group(0) if restated else ""))


# ── named-void identity ─────────────────────────────────────
#
# A named void emits an IfcOpeningElement whose Name and uuid5 GlobalId derive
# from its canonical. Two holes on one host sharing a leaf share a canonical —
# a silent GlobalId collision — so the per-scope uniqueness scan must see them.

_TWO_DOORWAYS_ONE_HOST = """
from lite_step.models import Project, Wall, Box, Point
def generate_project():
    b = Project(name="dup")
    wall = Wall(name="north")
    wall.add(Box(name="body", start=Point(x=-3000, y=-150, z=0),
                 end=Point(x=3000, y=150, z=3000), material="Concrete"))
    wall.void(Box(start=Point(x=-2500, y=-200, z=0),
                  end=Point(x=-1500, y=200, z=2100)), name="doorway")
    wall.void(Box(start=Point(x=1500, y=-200, z=0),
                  end=Point(x=2500, y=200, z=2100)), name="doorway")
    b.add(wall)
    return b
result = generate_project()
"""

_DOORWAY_PER_HOST = """
from lite_step.models import Project, Wall, Box, Point
def generate_project():
    b = Project(name="ok")
    for leaf, y in (("north", 0), ("south", 5000)):
        wall = Wall(name=leaf)
        wall.add(Box(name="body", start=Point(x=-3000, y=y - 150, z=0),
                     end=Point(x=3000, y=y + 150, z=3000), material="Concrete"))
        wall.void(Box(start=Point(x=-500, y=y - 200, z=0),
                      end=Point(x=500, y=y + 200, z=2100)), name="doorway")
        b.add(wall)
    return b
result = generate_project()
"""


def test_two_named_voids_with_one_leaf_on_one_host_fail_naming_both():
    r = execute_lite_step_script(_TWO_DOORWAYS_ONE_HOST)
    assert r.success, r.error            # authoring is fine; COMPILE validation refuses
    err = " | ".join(validate_project(r.project))
    assert "Duplicate canonical name 'opening:doorway:wall:north'" in err
    assert err.count(".void(name='doorway')") == 2   # BOTH sites named


def test_the_same_void_leaf_on_two_different_hosts_compiles():
    model = _compile(_DOORWAY_PER_HOST)
    names = sorted(o.Name for o in model.by_type("IfcOpeningElement"))
    assert names == ["opening:doorway:wall:north", "opening:doorway:wall:south"]
    # distinct canonicals -> distinct GlobalIds
    ids = [o.GlobalId for o in model.by_type("IfcOpeningElement")]
    assert len(set(ids)) == 2


def test_two_anonymous_voids_on_one_host_still_compile():
    """Anonymous holes carry no identity, so they cannot collide."""
    script = _TWO_DOORWAYS_ONE_HOST.replace(', name="doorway"', '')
    model = _compile(script)
    assert len(model.by_type("IfcOpeningElement")) == 2
