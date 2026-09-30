"""``authored_aabb()`` — the extent as authored, per element kind.

Two halves.

**Stated definitions.** One test per row of the table in
``compiler/extent.py``: a Box's corners ARE its extent, a Pipe reaches its
radius, an Extrude sweeps ONE way along a known normal, a container is the
union of its children. These pin what each accessor MEANS, which is the
deliverable — an author has to be able to rely on the answer without reading
the compiler.

**The differential oracle.** Analytic maths that agrees with itself proves
nothing. So every primitive is ALSO compiled end to end and the shipped IFC is
tessellated, and the analytic box is checked to contain every vertex — and,
where we claim ``exact``, to hug it. That is the only test here that can
falsify a pad, and it is why the numbers above can be trusted.

Fixtures sit away from the origin, off round numbers, and at asymmetric
angles, deliberately. Origin makes translation invisible, identity makes a
transform invisible, symmetry makes rounding invisible — a bounds test built
on convenient values passes against deliberately broken code.
"""
import math

import pytest

from lite_step.compiler.extent import (
    element_extent,
    extent_is_exact,
    extrude_offset,
    inexact_reason,
)
from lite_step.models import (
    Bar,
    Box,
    Element,
    Extrude,
    Material,
    Mesh,
    Pipe,
    Point,
    Point2D,
    Project,
    Revolve,
    Space,
    Storey,
    Sweep,
    Wall,
    Window,
)
from lite_step.models.bounds import Bounds, BoundsError, round_half_up

# Off-origin, non-round. See the module docstring.
_X, _Y, _Z = 3170, -2410, 1130


def _box(name="body", dx=1210, dy=347, dz=2703, x=_X, y=_Y, z=_Z):
    return Box(start=Point(x=x, y=y, z=z),
               end=Point(x=x + dx, y=y + dy, z=z + dz), name=name)


def _span(bounds, axis):
    return (bounds.size.x, bounds.size.y, bounds.size.z)[axis]


# ---------------------------------------------------------------------------
# The stated definition, per kind
# ---------------------------------------------------------------------------


def test_box_corners_are_the_extent():
    b = _box().authored_aabb()
    assert (b.min.x, b.min.y, b.min.z) == (_X, _Y, _Z)
    assert (b.max.x, b.max.y, b.max.z) == (_X + 1210, _Y + 347, _Z + 2703)
    assert (b.size.x, b.size.y, b.size.z) == (1210, 347, 2703)
    assert b.exact is True


def test_pipe_reaches_its_radius_across_the_run():
    """The failure the whole ``extent`` module exists for: the authored POINTS
    of a Pipe are its centreline, so a points-only walk reports zero width."""
    p = Pipe(path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X + 3000, y=_Y, z=_Z)],
             radius=120, name="riser")
    b = p.authored_aabb()
    assert _span(b, 1) == 240, "across the run: 2 x radius"
    assert _span(b, 2) == 240
    assert b.exact is False, "flat-capped ends overshoot by one radius"
    assert "radius" in inexact_reason(p)


def test_bar_reaches_half_its_diameter():
    bar = Bar(path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X, y=_Y, z=_Z + 2000)],
              diameter=16, name="rebar")
    b = bar.authored_aabb()
    assert _span(b, 0) == 16
    assert _span(b, 1) == 16


def test_extrude_sweeps_ONE_way_along_its_normal():
    """A 350 mm wall is 350 mm thick, not 700.

    The carve trigger pads both sides of the contour plane because it does not
    want to depend on winding. The authoring query does not have that excuse —
    ``apply_orientation_rules`` resolves the direction, so the extent is
    one-sided and exact. A symmetric answer here would be a safe BOUND and a
    useless answer to "where is the face of this wall".
    """
    e = Extrude(
        contour=[Point(x=_X, y=_Y, z=_Z), Point(x=_X + 6000, y=_Y, z=_Z),
                 Point(x=_X + 6000, y=_Y, z=_Z + 2700), Point(x=_X, y=_Y, z=_Z + 2700)],
        thickness=350, name="gable")
    b = e.authored_aabb()
    assert _span(b, 1) == 350, "one thickness, not two"
    assert b.exact is True
    # ...and it lands on the side the generator actually extrudes toward.
    span = extrude_offset(e)
    far = _Y + span[1]
    assert min(_Y, far) == b.min.y and max(_Y, far) == b.max.y


def test_extrude_offset_is_the_generators_own_direction():
    """Not a second copy of the winding rule — the SAME two calls the
    generator makes (``newell_normal`` then ``apply_orientation_rules``)."""
    from lite_step.ifc.geometry import apply_orientation_rules, newell_normal

    contour = [Point(x=_X, y=_Y, z=_Z), Point(x=_X + 1000, y=_Y, z=_Z),
               Point(x=_X + 1000, y=_Y + 800, z=_Z), Point(x=_X, y=_Y + 800, z=_Z)]
    e = Extrude(contour=contour, thickness=200, name="deck")
    n = apply_orientation_rules(
        newell_normal([(float(p.x), float(p.y), float(p.z)) for p in contour]))
    assert extrude_offset(e) == pytest.approx(tuple(200 * c for c in n))


def test_mesh_is_its_vertex_hull():
    m = Mesh(vertices=[Point(x=_X, y=_Y, z=_Z),
                       Point(x=_X + 900, y=_Y, z=_Z),
                       Point(x=_X, y=_Y + 700, z=_Z + 300)],
             faces=[[0, 1, 2]], name="tri")
    b = m.authored_aabb()
    assert (b.size.x, b.size.y, b.size.z) == (900, 700, 300)
    assert b.exact is True


def test_container_is_the_union_of_its_children():
    w = Wall(name="south")
    w.add(_box(name="body"))
    b = w.authored_aabb()
    assert (b.size.x, b.size.y, b.size.z) == (1210, 347, 2703)


def test_container_pads_PER_CHILD_not_once_at_the_top():
    """C2 — the bug a naive ``pad(element_aabb(container))`` ships.

    ``element_aabb`` already recurses into ``_elements``, so padding once at
    the top keys the pad on the CONTAINER's type name. ``Element`` is not
    ``Pipe``, so the pad is zero and the container reports the pipe's
    centreline as its extent.
    """
    from lite_step.compiler.extent import padded_aabb

    pipe = Pipe(path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X + 3000, y=_Y, z=_Z)],
                radius=120, name="riser")
    host = Element(ifc_class="IfcCovering", name="shaft").add(pipe)

    top_padded = padded_aabb(host)
    assert top_padded[1][1] - top_padded[0][1] == 0, "the bug, still reproducible"
    assert _span(host.authored_aabb(), 1) == 240, "the fix"


def test_container_union_spans_several_children():
    space = Space(name="room")
    space.add(_box(name="a", dx=1000, dy=1000, dz=1000))
    space.add(_box(name="b", dx=1000, dy=1000, dz=1000,
                   x=_X + 5000, y=_Y, z=_Z))
    b = space.authored_aabb()
    assert _span(b, 0) == 6000


def test_storey_and_project_report_the_whole_model():
    w = Wall(name="south").add(_box(name="body"))
    proj = Project(name="t")
    proj.add(w)
    assert proj.storeys[0].authored_aabb().size.x == 1210
    assert proj.authored_aabb().size.x == 1210


# ---------------------------------------------------------------------------
# What must RAISE rather than return a plausible number
# ---------------------------------------------------------------------------


def test_empty_container_raises_and_names_itself():
    with pytest.raises(BoundsError, match=r"Wall 'hollow'.*no extent"):
        Wall(name="hollow").authored_aabb()


def test_empty_project_raises():
    with pytest.raises(BoundsError, match="no extent"):
        Project(name="empty").authored_aabb()


def test_opening_raises_and_points_at_the_host():
    """A Window carries a SIZE and no position (v20.0.0). Returning its size
    as if it were an extent would place joinery at the origin."""
    with pytest.raises(BoundsError, match=r"HOST decides|wall.authored_aabb"):
        Window(width=1210, height=1403, name="w0").authored_aabb()


# ---------------------------------------------------------------------------
# point_at
# ---------------------------------------------------------------------------


def test_point_at_centre_faces_and_corner():
    b = _box(dx=1000, dy=400, dz=2000).authored_aabb()
    assert b.point_at(0, 0, 0) == Point(x=_X + 500, y=_Y + 200, z=_Z + 1000)
    assert b.point_at(0, -100, 0) == Point(x=_X + 500, y=_Y, z=_Z + 1000)
    assert b.point_at(100, 100, 100) == Point(x=_X + 1000, y=_Y + 400, z=_Z + 2000)
    assert b.point_at(0, 0, -50) == Point(x=_X + 500, y=_Y + 200, z=_Z + 500)


def test_point_at_rejects_out_of_range_and_says_what_to_do():
    b = _box().authored_aabb()
    with pytest.raises(BoundsError, match=r"outside \[-100, 100\]"):
        b.point_at(150, 0, 0)
    with pytest.raises(BoundsError, match="add an offset"):
        b.point_at(0, 0, 101)


def test_point_at_rejects_floats():
    """Integer percent keeps the no-floats rule at the API surface."""
    b = _box().authored_aabb()
    with pytest.raises(BoundsError, match="int percent"):
        b.point_at(0.5, 0, 0)


def test_round_half_up_is_not_bankers_rounding():
    """The spec asks for half-UP; Python's ``round`` is half-to-EVEN.

    They disagree on exactly the case that shows up in practice — an odd
    extent, e.g. a 2701 mm wall centring at 1350.5. ``round`` gives 1350 and
    ``round(1351.5)`` gives 1352, so the error is not even consistent in
    direction. That is a 1 mm asymmetry someone chases for an afternoon.
    """
    assert round_half_up(1350.5) == 1351
    assert round(1350.5) == 1350          # the trap, pinned so it stays visible
    assert round_half_up(1351.5) == 1352
    assert round_half_up(-2.5) == -2


def test_odd_extent_centres_half_up():
    b = _box(dx=2701, dy=1, dz=1).authored_aabb()
    assert b.point_at(0, 0, 0).x == _X + 1351


def test_min_and_max_are_the_extreme_corners():
    b = _box().authored_aabb()
    assert b.min == b.point_at(-100, -100, -100)
    assert b.max == b.point_at(100, 100, 100)
    assert b.is_axis_aligned is True


# ---------------------------------------------------------------------------
# The differential oracle — the only thing here that can falsify a pad
# ---------------------------------------------------------------------------


def _salt_tolerance_mm() -> float:
    """How far the SHIPPED solid may exceed the AUTHORED extent, in mm.

    Not a fudge factor — the sum of TWO named pipeline facts, each read from
    its own source of truth so that if either moves, this follows it.

    1. **The instancing salt.** ``EntityCache`` nudges emitted dimensions by
       ``_UNIQUE_DIMS_STEP`` (2e-5 m = 0.02 mm) to defeat content-based
       instancing in the SPA fragments render worker, which drops geometry
       when two elements hash identically. So a compiled solid is up to one
       step larger than the shape the author wrote, and no authoring-space
       derivation can or should predict which elements got salted. Measured
       on the two-bend Sweep fixture below: 0.019 mm of overshoot.

    2. **The dedup snap.** ``deduplicator._normalize_floats`` rounds every
       length in a DEDUP_SAFE_TYPE to ``DEDUP_LENGTH_DECIMALS`` (0.1 mm), so
       a shipped vertex sits up to HALF a step (0.05 mm) either side of the
       coordinate the compiler computed.

    Fact 2 was absent while every oracle fixture happened to be immune to it:
    their vertices came from analytic solids (IfcRevolvedAreaSolid,
    IfcSweptDiskSolid) that the kernel tessellates in full precision from
    inputs — 0.14, 3.17, -2.41 — which are all exactly representable at 4
    decimals, so nothing ever snapped. The revolve case now ships an
    IfcFacetedBrep whose vertices are computed world coordinates
    (3170 + 140*cos(3.6 deg) = 3309.7237 mm), and those DO snap. The
    tolerance was passing by luck of the fixtures, not by argument.
    """
    from lite_step.ifc.entity_cache import EntityCache
    from lite_step.ifc.deduplicator import DEDUP_LENGTH_DECIMALS

    salt = EntityCache._UNIQUE_DIMS_STEP * 1000.0
    dedup_half_step = 0.5 * 10.0 ** (3 - DEDUP_LENGTH_DECIMALS)   # mm
    return salt + dedup_half_step + 1e-6


_SALT_TOL_MM = _salt_tolerance_mm()


def _shipped_vertices(elem):
    """World vertices of ``elem`` as it is ACTUALLY EMITTED, in mm.

    Compiles a one-element project end to end and tessellates the resulting
    IFC, rather than calling ``generator.tessellate_solid_to_mesh`` directly.

    Two reasons, and the second is not optional. First, this validates the
    artefact we ship rather than an internal helper. Second, that helper is
    BROKEN for ``Pipe`` and ``Bar``: it builds a throwaway single-element
    model and its geometry comes back MIRRORED THROUGH THE ORIGIN — a pipe
    authored at x=1000..3000 tessellates to x=-3000..-1000. The shipped IFC is
    correct (verified: 1.0..3.0 m), so nothing users see is wrong, but the
    helper also feeds mesh carving, ``raycast``, ``height_at`` and ``miter()``
    — see. Using it as the oracle here would have compared our
    maths against a mirror and reported a false failure on every ``Pipe``.
    """
    import numpy as np
    import ifcopenshell
    import ifcopenshell.geom

    from lite_step.compiler.executor import normalize_project_to_meters
    from lite_step.ifc.generator import generate_ifc

    proj = Project(name="oracle")
    proj.add(elem)
    result = generate_ifc(normalize_project_to_meters(proj))
    model = ifcopenshell.file.from_string(result.ifc_content)

    # Only the element's OWN product. A compiled model also carries one
    # `IfcAnnotation(ObjectType="DRAWING")` per storey (v21.2.0), and an
    # IfcAnnotation is an IfcProduct with a representation — sweeping up
    # "every product with geometry" pulled the drawing frame into the vertex
    # set and put points thousands of mm from the element.
    products = [p for p in model.by_type("IfcProduct")
                if getattr(p, "Representation", None) is not None
                and not p.is_a("IfcAnnotation")
                and not p.is_a("IfcSpatialStructureElement")]
    if not products:
        return None
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    verts = []
    for product in products:
        shape = ifcopenshell.geom.create_shape(settings, product)
        v = np.asarray(shape.geometry.verts).reshape(-1, 3)
        verts.append(v * 1000.0)          # compile normalises to metres
    return np.vstack(verts) if verts else None


_ORACLE_CASES = [
    ("pipe-diagonal", Pipe(
        path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X + 2300, y=_Y + 1700, z=_Z + 400)],
        radius=95, name="p")),
    ("bar-vertical", Bar(
        path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X, y=_Y, z=_Z + 1900)],
        diameter=25, name="b")),
    ("revolve-asymmetric", Revolve(
        profile=[Point2D(x=140, y=0), Point2D(x=470, y=0),
                 Point2D(x=470, y=210), Point2D(x=140, y=330)],
        path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X, y=_Y, z=_Z + 900)],
        angle=360, name="r")),
    # --- claim exact: the tightness half of the oracle needs cases that do,
    #     or `test_exact_extents_are_actually_tight` skips every row and the
    #     assertion never executes.
    ("box-offset", Box(
        start=Point(x=_X, y=_Y, z=_Z),
        end=Point(x=_X + 1210, y=_Y + 347, z=_Z + 2703), name="bx")),
    ("extrude-slanted", Extrude(
        contour=[Point(x=_X, y=_Y, z=_Z),
                 Point(x=_X + 2400, y=_Y + 900, z=_Z),
                 Point(x=_X + 2400, y=_Y + 900, z=_Z + 1700),
                 Point(x=_X, y=_Y, z=_Z + 1700)],
        thickness=220, name="ex")),
    ("sweep-single-segment", Sweep(
        profile=[Point2D(x=-90, y=-45), Point2D(x=90, y=-45),
                 Point2D(x=90, y=45), Point2D(x=-90, y=45)],
        path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X + 2600, y=_Y + 1100, z=_Z)],
        name="s1")),
    ("sweep-profiled-bent", Sweep(
        profile=[Point2D(x=-90, y=-45), Point2D(x=90, y=-45),
                 Point2D(x=90, y=45), Point2D(x=-90, y=45)],
        path=[Point(x=_X, y=_Y, z=_Z),
              Point(x=_X + 1900, y=_Y + 700, z=_Z),
              Point(x=_X + 3100, y=_Y + 700, z=_Z + 800)],
        name="s")),
    # The house-style member spelling. Declared inexact (obb still pads
    # isotropically), so it only runs the CONTAINMENT half — which is the
    # half that matters here: it proves the resolved 45x195 rectangle really
    # does bound the solid the kernel builds, rather than merely being a
    # number that looks better than zero.
    ("sweep-member-profile-mm", Sweep(
        path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X + 2600, y=_Y + 1100, z=_Z)],
        material=Material(key="Timber_C24", profile_mm=(45, 195)),
        name="m")),
]


@pytest.mark.parametrize("label,elem", _ORACLE_CASES, ids=[c[0] for c in _ORACLE_CASES])
def test_analytic_extent_contains_the_tessellated_solid(label, elem):
    """The analytic box must CONTAIN every vertex the kernel produces.

    This is the assertion that makes the numbers above dependable rather than
    merely self-consistent: it compares our arithmetic against an independent
    evaluation of the same authored shape. Under-approximating is the
    dangerous direction — ``displacement`` infers carves from these boxes, so
    a bound that is too small silently loses a boolean.

    Deliberately asymmetric paths and profiles: an axis-aligned run or a
    symmetric section lets a wrong pad cancel out and pass.
    """
    pytest.importorskip("ifcopenshell.geom")
    verts = _shipped_vertices(elem)
    if verts is None:
        pytest.skip(f"{label}: no representation emitted on this platform")

    box = element_extent(elem)
    assert box is not None
    (x0, y0, z0), (x1, y1, z1) = box
    tol = _SALT_TOL_MM
    for vx, vy, vz in verts:
        assert x0 - tol <= vx <= x1 + tol, f"{label}: vertex x={vx} outside [{x0}, {x1}]"
        assert y0 - tol <= vy <= y1 + tol, f"{label}: vertex y={vy} outside [{y0}, {y1}]"
        assert z0 - tol <= vz <= z1 + tol, f"{label}: vertex z={vz} outside [{z0}, {z1}]"


@pytest.mark.parametrize("label,elem", _ORACLE_CASES, ids=[c[0] for c in _ORACLE_CASES])
def test_exact_extents_are_actually_tight(label, elem):
    """The other half: a box that merely CONTAINS the solid could be the whole
    building. Where we claim ``exact``, the tessellation must reach the faces.

    Only asserted for the shapes that claim it — the ones flagged
    ``exact=False`` are allowed to overshoot, which is what the flag is for.
    """
    pytest.importorskip("ifcopenshell.geom")
    if not extent_is_exact(elem):
        pytest.skip(f"{label}: declared inexact — {inexact_reason(elem)}")
    verts = _shipped_vertices(elem)
    if verts is None:
        pytest.skip(f"{label}: no representation emitted on this platform")

    (x0, y0, z0), (x1, y1, z1) = element_extent(elem)
    for axis, lo, hi in ((0, x0, x1), (1, y0, y1), (2, z0, z1)):
        vals = [v[axis] for v in verts]
        slack = max(min(vals) - lo, hi - max(vals))
        extent = hi - lo
        assert slack <= 0.02 * extent + _SALT_TOL_MM, (
            f"{label}: axis {axis} claims exact but the analytic box is "
            f"{slack:.3f} wider than the tessellation on a {extent:.3f} span"
        )


def test_the_oracle_would_catch_a_broken_pad():
    """Proof the oracle has teeth.

    A Pipe whose radius pad is removed reports its centreline, and the
    tessellated surface then lies OUTSIDE the analytic box. If this ever stops
    failing, the oracle above is decoration.
    """
    pytest.importorskip("ifcopenshell.geom")
    pipe = Pipe(path=[Point(x=_X, y=_Y, z=_Z), Point(x=_X + 2000, y=_Y, z=_Z)],
                radius=110, name="p")
    verts = _shipped_vertices(pipe)
    if verts is None:
        pytest.skip("no representation emitted on this platform")

    from lite_step.compiler.placement import _element_aabb
    unpadded = _element_aabb(pipe)          # the points-only walk: no radius
    (_x0, y0, _z0), (_x1, y1, _z1) = unpadded
    assert any(not (y0 <= v[1] <= y1) for v in verts), (
        "an unpadded Pipe bound must fail to contain its own surface — if it "
        "contains it, the oracle cannot detect a missing pad"
    )


# ---------------------------------------------------------------------------
# The unit invariant: a bound is in the units of the coordinates it was given
# ---------------------------------------------------------------------------
#. ``Material(profile_mm=)`` and ``Material(thickness_mm=)`` are
# frozen registry FACTS in millimetres — ``normalize_project_to_meters`` never
# touches them — so every read of one has to state which domain the caller is
# in. The member-section pad hardcoded ``/ 1000.0``: right for displacement
# (metres), wrong by 1000x for the authoring query, where it rounded away and
# reported a 45x195 member as infinitely thin.


def _member(length_mm=1000, divisor=1.0):
    """One 45x195 member ``length_mm`` long, expressed in a domain where one
    coordinate unit is ``divisor`` millimetres."""
    return Sweep(name="stud", material=Material(key="Timber_C24",
                                                profile_mm=(45, 195)),
                 path=[Point.model_construct(x=0.0, y=0.0, z=0.0),
                       Point.model_construct(x=length_mm / divisor, y=0.0, z=0.0)])


def test_the_member_pad_scales_with_the_coordinate_domain():
    """THE invariant. The same physical member, in both unit domains, must be
    bounded by the same physical box — so the pad has to scale with the
    coordinates rather than sit at a hardcoded 1/1000.

    Falsifiable by construction: reinstate the hardcoded divisor in
    ``_member_section_reach`` and the mm row collapses to ~0.1 mm while the
    metres row keeps its 0.1 m, and the ratio below stops being 1000.
    """
    from lite_step.compiler.extent import MM_PER_METER, padded_aabb

    mm = padded_aabb(_member(1000, divisor=1.0), 1.0)
    m = padded_aabb(_member(1000, divisor=MM_PER_METER), MM_PER_METER)

    def span(box, axis):
        return box[1][axis] - box[0][axis]

    for axis in (0, 1, 2):
        assert span(mm, axis) == pytest.approx(span(m, axis) * MM_PER_METER), (
            f"axis {axis}: the mm-domain bound is not the metres-domain bound "
            f"scaled — the pad is not in the units of its coordinates"
        )
    # …and the absolute number is the section's half-diagonal, not a thousandth
    # of it. 0.5 * hypot(45, 195) = 100.06 mm.
    assert span(mm, 1) == pytest.approx(2 * 100.0625, abs=0.01)


def test_member_form_sweep_is_not_reported_infinitely_thin():
    """In the shape an author meets it: measure a 45x195 member and a
    section-less answer would be ``(1000, 0, 0)``."""
    b = _member(1000).authored_aabb()
    assert (b.size.x, b.size.y, b.size.z) == (1000, 45, 195)


def test_the_three_spellings_of_one_member_agree():
    """§1.6's table. The house style (mandated by the one-source rule) was the
    spelling that lied, and it lied by UNDER-approximating — which the pad
    rules forbid anywhere else in this module."""
    house = _member(1000)
    explicit = Sweep(name="explicit",
                     profile=[Point2D.model_construct(x=-22.5, y=-97.5),
                              Point2D.model_construct(x=22.5, y=-97.5),
                              Point2D.model_construct(x=22.5, y=97.5),
                              Point2D.model_construct(x=-22.5, y=97.5)],
                     path=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0)])
    boxed = Box(name="boxed", start=Point.model_construct(x=0.0, y=-22.5, z=-97.5),
                end=Point.model_construct(x=1000.0, y=22.5, z=97.5))

    sizes = {label: element_extent(e)
             for label, e in (("profile_mm", house), ("profile=", explicit),
                              ("Box", boxed))}
    ref = sizes["Box"]
    for label, box in sizes.items():
        for axis in (0, 1, 2):
            assert box[1][axis] - box[0][axis] == pytest.approx(
                ref[1][axis] - ref[0][axis], abs=0.001), (
                f"{label} disagrees with Box on axis {axis}: "
                f"{box} vs {ref}")


def test_sheet_material_thickness_answers_in_the_authoring_domain():
    """The second mm-fact read, found while fixing the first: an ``Extrude``
    with no authored ``thickness=`` takes it from ``Material(thickness_mm=)``,
    and that conversion was hardcoded to metres too. A 22 mm sheet reported a
    0.022 mm extent to ``authored_aabb()``."""
    from lite_step.compiler.extent import MM_PER_METER, extrude_offset

    sheet = Extrude(name="deck", material=Material(key="Chipboard_P6",
                                                   thickness_mm=22),
                    contour=[Point(x=_X, y=_Y, z=_Z),
                             Point(x=_X + 1200, y=_Y, z=_Z),
                             Point(x=_X + 1200, y=_Y, z=_Z + 900),
                             Point(x=_X, y=_Y, z=_Z + 900)])
    assert _span(sheet.authored_aabb(), 1) == 22
    # …and the metres-domain caller still gets metres.
    span_m = extrude_offset(sheet, MM_PER_METER)
    assert max(abs(c) for c in span_m) == pytest.approx(0.022)


def test_the_displacement_trigger_keeps_its_isotropic_pad():
    """The carve trigger is NOT retightened by the query fix.

    ``element_extent`` now resolves the member section into the real
    rectangle; ``padded_aabb`` deliberately still pads by the section
    half-diagonal in every axis, because tightening the trigger changes which
    pairs carve. Same split the contour ``Extrude`` already has (symmetric pad
    for the trigger, one-sided ``extrude_offset`` for the query) — on purpose,
    not by neglect. If these two ever agree, the corpus byte-comparison that
    licensed this split proves nothing.
    """
    from lite_step.compiler.extent import padded_aabb

    stud = _member(1000)
    trigger = padded_aabb(stud, 1.0)
    query = element_extent(stud)
    assert trigger[1][0] - trigger[0][0] == pytest.approx(1200.125, abs=0.01)
    assert query[1][0] - query[0][0] == pytest.approx(1000.0, abs=0.001)


# ---------------------------------------------------------------------------
# An unbaked anchor target is not an empty container
# ---------------------------------------------------------------------------


def test_unbaked_anchor_target_says_so_instead_of_claiming_it_is_empty():
    """The two reasons `element_extent` returns None must not share a message.

    An anchored child filters ITSELF out of `iter_geometry_points` while its
    `_anchor_spec` is unresolved, so `element_extent` sees nothing. The old
    text — "holds no geometry" — is FALSE for a Box with start/end, and sent a
    reader hunting a bug in their own geometry.

    `authored_aabb` is where this message now lives, and it is the honest
    place for it: that accessor does NOT trigger the bake, so the extent
    really is unavailable. `world_aabb` resolves first and then reads (see
    below) — it does not raise on its way to the pass that answers it.
    """
    wall = Wall(name="host")
    wall.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=4000, y=200, z=3000)))
    shelf = Box(name="shelf", start=Point(x=0, y=0, z=0),
                end=Point(x=500, y=300, z=40))
    wall.anchor(shelf, along=1000, up=1200)
    Project(name="t").add(wall)

    with pytest.raises(BoundsError, match="anchored but not yet resolved"):
        shelf.authored_aabb()

    # …and it names the way out, which is not discoverable otherwise.
    with pytest.raises(BoundsError, match="already-placed element first"):
        shelf.authored_aabb()


def test_world_aabb_on_an_unbaked_anchor_target_resolves_and_answers():
    """`world_aabb` triggers the bake, so it must READ AFTER it, not before.

    Snapshotting the extent first raises on the way to the very pass that
    would supply it (`spec-anchor-generalisation.md` C.2: fixable
    only by querying a different element first, and undiscoverable from the
    message). Reading second makes the accessor self-sufficient — and closes
    a silent wrong answer that container anchoring would otherwise expose: an
    `.add()`ed child of an anchored CONTAINER is readable both before and
    after the bake and MOVES in between, so a pre-bake read returns
    plausible, wrong, local coordinates with no error at all.
    """
    wall = Wall(name="host")
    wall.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=4000, y=200, z=3000)))
    shelf = Box(name="shelf", start=Point(x=0, y=0, z=0),
                end=Point(x=500, y=300, z=40))
    wall.anchor(shelf, along=1000, up=1200)
    Project(name="t").add(wall)

    world = shelf.world_aabb()
    # This lone wall's outer face is its +y one (``facing=+1``), so the
    # viewer's right runs -x and ``along=1000`` measures 1000 mm in from the
    # x-MAX end: 4000 - 1000 - 500. Before that it read
    # [1000, 1500] — the bake reversed ``across`` without reversing ``along``,
    # which is a mirror rather than a rotation.
    assert world.min.x == 2500 and world.max.x == 3000
    assert world.min.z == 1200 and world.max.z == 1240


def test_the_empty_container_message_is_unchanged():
    """The other branch must keep saying what IS true of it."""
    hollow = Wall(name="hollow")
    Project(name="t").add(hollow)
    with pytest.raises(BoundsError, match="holds no geometry"):
        hollow.authored_aabb()


def test_querying_a_placed_element_first_unblocks_the_anchored_one():
    """Pins the documented escape hatch, so the message cannot become a lie."""
    wall = Wall(name="host")
    wall.add(Box(name="body", start=Point(x=0, y=0, z=0),
                 end=Point(x=4000, y=200, z=3000)))
    shelf = Box(name="shelf", start=Point(x=0, y=0, z=0),
                end=Point(x=500, y=300, z=40))
    wall.anchor(shelf, along=1000, up=1200)
    Project(name="t").add(wall)

    wall.world_aabb()                      # the host bakes the whole project
    b = shelf.world_aabb()                 # …and now the child answers
    # ``facing=+1`` host: right runs -x, so along=1000 lands at 4000-1000-500.
    assert b.min.x == 2500, "the anchor's along= should place it 1000 in from x-max"
    assert shelf._anchor_resolved is True


# ---------------------------------------------------------------------------
# The scoped extent memo
# ---------------------------------------------------------------------------


class TestExtentMemo:
    """``extent.memoized_extents`` — a cache that is only sound while nothing
    mutates, so its SCOPE is the whole of its safety argument."""

    @staticmethod
    def _wall():
        from lite_step.models import Box, Point, Wall
        wall = Wall(name="north")
        wall.add(Box(name="body", start=Point(x=0, y=0, z=0),
                     end=Point(x=6000, y=300, z=2700)))
        return wall

    def test_it_is_off_by_default(self):
        """Nothing gets a cache it did not ask for.

        The default has to be OFF because most compiler passes run either side
        of a MUTATION (normalize divides by 1000; the anchor bake moves
        children into world space), and a cache spanning one of those hands
        out a box for geometry that does not exist.
        """
        from lite_step.compiler.extent import _EXTENT_MEMO
        assert _EXTENT_MEMO.get() is None

    def test_inside_the_block_the_answer_is_cached(self):
        from lite_step.compiler import extent as E
        wall = self._wall()
        calls = []
        inner = E._element_extent_inner

        def counting(elem, divisor, on_path):
            calls.append(id(elem))
            return inner(elem, divisor, on_path)

        E._element_extent_inner = counting
        try:
            with E.memoized_extents():
                first = E.element_extent(wall)
                n_after_first = len(calls)
                second = E.element_extent(wall)
                assert len(calls) == n_after_first, "second query recomputed"
            assert first == second
        finally:
            E._element_extent_inner = inner

    def test_it_does_not_leak_past_its_block(self):
        from lite_step.compiler.extent import _EXTENT_MEMO, memoized_extents
        with memoized_extents():
            assert _EXTENT_MEMO.get() is not None
        assert _EXTENT_MEMO.get() is None, (
            "the memo outlived its block — the next pass would read boxes for "
            "geometry it has since moved")

    def test_the_key_includes_the_divisor(self):
        """Both unit domains are live in one compile.

        ``divisor`` is not decoration: a Material-derived pad is in mm in BOTH
        domains, so the same element genuinely has two different extents. A
        memo keyed on the element alone would hand the metres caller the
        millimetre answer.
        """
        from lite_step.compiler.extent import (
            _EXTENT_MEMO, MM_PER_METER, element_extent, memoized_extents)
        wall = self._wall()
        with memoized_extents():
            element_extent(wall, 1.0)
            element_extent(wall, MM_PER_METER)
            entries = _EXTENT_MEMO.get()
            keyed_on_this_wall = [k for k in entries if k[0] == id(wall)]
        assert len(keyed_on_this_wall) == 2, (
            f"one entry for two unit domains ({keyed_on_this_wall}) — the "
            f"metres caller would be served the millimetre answer")

    def test_a_mutation_inside_the_block_is_NOT_seen(self):
        """The hazard, pinned — so the scoping stays deliberate.

        This is the behaviour that makes a compile-long memo wrong, and it is
        asserted rather than described: inside one block the cache is trusted
        absolutely. The tree report is safe only because it runs after every
        mutating pass and does not write.
        """
        from lite_step.compiler.extent import element_extent, memoized_extents
        from lite_step.models import Box, Point
        wall = self._wall()
        with memoized_extents():
            before = element_extent(wall)
            # Grow the container by adding a child far outside its current
            # box. (Points are frozen, so geometry moves by re-parenting, not
            # by writing coordinates.)
            wall.add(Box(name="wing", start=Point(x=90000, y=0, z=0),
                         end=Point(x=99000, y=300, z=2700)))
            after = element_extent(wall)
            assert after == before, (
                "the memo answered from fresh geometry — if that is now true, "
                "this test is obsolete, but so is the reason the memo is "
                "scoped; re-read memoized_extents() before widening it")
        assert element_extent(wall) != before, "outside the block it re-reads"


# ── .union() operands add material, so they add extent ──────────────────────
#
# Every hospital floorplate is ``Box.union(Revolve)`` — a straight run fused to
# a rounded tip. ``authored_aabb()`` used to answer the LEFT operand only, and
# the reinforcement layout derived from it stopped short of the tip.

def _fingertip():
    box = Box(name="b", material="Concrete_C30-37",
              start=Point(x=0, y=-12000, z=0), end=Point(x=48000, y=12000, z=275))
    rev = Revolve(name="t", angle=36000,
                  profile=[Point2D(x=0, y=0), Point2D(x=12000, y=0),
                           Point2D(x=12000, y=275), Point2D(x=0, y=275)],
                  path=[Point(x=48000, y=0, z=0), Point(x=48000, y=0, z=275)],
                  material=Material(key="Concrete_C30-37"))
    return box, rev


def test_union_extent_includes_the_right_operand():
    box, rev = _fingertip()
    assert (box.authored_aabb().min.x, box.authored_aabb().max.x) == (0, 48000)
    fused = box.union(rev)
    b = fused.authored_aabb()
    assert (b.min.x, b.max.x) == (0, 60000)      # was 0..48000


def test_union_extent_is_recursive():
    box, rev = _fingertip()
    far = Box(name="far", start=Point(x=70000, y=0, z=0),
              end=Point(x=71000, y=100, z=100))
    rev.union(far)                               # union of a union operand
    box.union(rev)
    assert box.authored_aabb().max.x == 71000


def test_union_extent_inexactness_follows_the_operand():
    """A Revolve bound is loose; a Box union must not claim to be exact."""
    box, rev = _fingertip()
    assert extent_is_exact(box) is True
    box.union(rev)
    assert extent_is_exact(box) is False
    assert "Revolve .union() operand" in inexact_reason(box)


def test_difference_operand_still_never_enlarges_the_host():
    box, _ = _fingertip()
    cutter = Box(name="c", start=Point(x=90000, y=0, z=0),
                 end=Point(x=91000, y=100, z=100))
    box.difference(cutter)
    assert box.authored_aabb().max.x == 48000


def test_build_reinforcement_covers_the_unioned_tip():
    """The real consumer: the rebar layout spans the whole capsule, tip included."""
    # Only the reinforcement helpers are lifted from the fixture: importing the
    # whole module registers its custom materials globally, which would leak
    # into the material-registry tests that run after this file.
    import ast
    from pathlib import Path

    src = (Path(__file__).parent / "fixtures" / "inverted_roof_building.py").read_text(encoding="utf-8")
    wanted = {"REBAR_COVER", "REBAR_PITCH_MAX", "REBAR_DIAMETER", "_AXIS", "I",
              "_bar_positions", "build_reinforcement"}
    picked = []
    for node in ast.parse(src).body:
        names = ({node.name} if isinstance(node, ast.FunctionDef)
                 else {t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)})
        if names & wanted:
            picked.append(node)
    assert {n.name for n in picked if isinstance(n, ast.FunctionDef)} >= {"build_reinforcement"}
    ns = {"Bar": Bar, "Point": Point, "I": lambda x: int(round(x))}
    exec(compile(ast.Module(body=picked, type_ignores=[]), "reinforcement", "exec"), ns)

    class _M:  # the fixture's namespace, used like the module was
        pass
    module = _M()
    module.build_reinforcement = ns["build_reinforcement"]
    module.REBAR_COVER = ns["REBAR_COVER"]

    box, rev = _fingertip()
    slab = box.union(rev)
    bars = module.build_reinforcement(slab)
    far_end = max(p.x for bar in bars for p in bar.path)
    # x-running bars stop one cover short of the 60000 tip, not of the 48000 box
    assert far_end == 60000 - module.REBAR_COVER
