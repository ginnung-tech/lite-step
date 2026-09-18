"""Pins the entity-kind catalog. If the sweep (or a new element type) silently
changes a membership, these fail — the taxonomy is the single source of truth the
whole compiler branches on."""

from __future__ import annotations

from lite_step.models import taxonomy as tx
from lite_step.models.elements import (
    Box, Extrude, Revolve, Sweep, Pipe, Bar, Mesh,
    Wall, Column, Beam, Slab, Roof, Window, Door, Element,
    Site, Space, ReferencePoint, GuideLine,
)


def test_membership_is_exact():
    assert set(tx.TESSELLATION_PRISM) == {Box, Extrude}
    assert set(tx.TESSELLATION_CURVED) == {Revolve, Sweep, Pipe, Bar}
    assert set(tx.SOLIDS) == {Box, Extrude, Revolve, Sweep, Pipe, Bar}
    assert set(tx.MESH) == {Mesh}
    assert set(tx.PHYSICAL) == {Wall, Column, Beam, Slab, Roof, Element, Window, Door}
    assert set(tx.OPENINGS) == {Window, Door}
    assert set(tx.SPATIAL) == {Site, Space}
    assert set(tx.ANNOTATION) == {ReferencePoint, GuideLine}
    # opening-eligibility axis: solids + non-fill project elements (cross-cuts the
    # element-nature partition — spans geometric SOLIDS and PHYSICAL project elements)
    assert set(tx.VOIDABLE) == {Box, Extrude, Revolve, Sweep, Pipe, Bar,
                                Wall, Column, Beam, Slab, Roof, Element}


def test_element_nature_predicates_partition_the_types():
    # is_geometric/is_physical/is_spatial/is_annotation partition the element TYPES
    # (the ROUTING tuples like VOIDABLE/TESSELLATION_* cross-cut them — see above).
    all_types = [Box, Extrude, Revolve, Sweep, Pipe, Bar, Mesh, Wall, Column, Beam,
                 Slab, Roof, Window, Door, Element, Site, Space, ReferencePoint, GuideLine]
    for t in all_types:
        primaries = [tx.is_geometric(t), tx.is_physical(t),
                     tx.is_spatial(t), tx.is_annotation(t)]
        assert sum(primaries) == 1, f"{t.__name__} must be in exactly one kind, got {primaries}"


def test_the_old_axis_name_is_gone():
    """WS-V phase 1: the element-nature axis is PHYSICAL, and there is no second
    spelling. A left-behind alias would let a stale caller keep compiling while
    the vocabulary drifts — the failure this rename exists to prevent."""
    assert not hasattr(tx, "SEMANTIC")
    assert not hasattr(tx, "is_semantic")
    assert not hasattr(tx, "_SEMANTIC_N")


def test_tessellation_kind():
    assert tx.tessellation_kind(Box) == "prism"
    assert tx.tessellation_kind(Extrude) == "prism"
    assert tx.tessellation_kind(Revolve) == "curved"
    assert tx.tessellation_kind(Pipe) == "curved"
    assert tx.tessellation_kind(Mesh) == "mesh"
    assert tx.tessellation_kind(Wall) is None
    assert tx.tessellation_kind(Site) is None


def test_predicates_accept_instance_class_and_string():
    from lite_step.models import Point, Point2D
    from lite_step.strict import suspend_strict
    with suspend_strict():
        box = Box(start=Point(x=0, y=0, z=0), end=Point(x=1, y=1, z=1))
    assert tx.is_solid(box) and tx.is_solid(Box) and tx.is_solid("Box")
    assert tx.is_prism(box) and not tx.is_curved(box)
    rev = Revolve(profile=[Point2D(x=0, y=0), Point2D(x=1, y=0), Point2D(x=1, y=1)],
                  path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=1)], angle=36000)
    assert tx.is_curved(rev) and tx.is_solid(rev) and tx.tessellation_kind(rev) == "curved"


def test_voidable():
    # geometric solids and physical project elements host an IfcOpeningElement
    assert tx.is_voidable(Box) and tx.is_voidable(Revolve)
    assert tx.is_voidable(Wall) and tx.is_voidable(Slab) and tx.is_voidable(Element)
    # meshes, relational containers, and opening-fills do NOT
    assert not tx.is_voidable(Mesh)
    assert not tx.is_voidable(Site) and not tx.is_voidable(Space)
    assert not tx.is_voidable(Window) and not tx.is_voidable(Door)


def test_kinds_match_the_pre_refactor_groupings():
    # displacement._solid_types() was (Box, Extrude, Sweep, Pipe, Revolve, Bar)
    assert set(tx.SOLIDS) == {Box, Extrude, Sweep, Pipe, Revolve, Bar}
    # generator._curved_solid_builders keys were Revolve/Sweep/Pipe/Bar
    assert set(tx.TESSELLATION_CURVED) == {Revolve, Sweep, Pipe, Bar}


