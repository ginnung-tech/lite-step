"""DSL v8 vocabulary cutover — Box/Extrude split, Slab container.intersection().

These pin the three behaviours introduced at the v8.0.0 hard cutover:

  1. ``.intersection()`` emits ``IfcBooleanResult`` with ``INTERSECTION`` on
     BOTH backends and round-trips through the normalizer as "intersection".
  2. ``Solid`` is split into ``Box`` (start/end) and ``Extrude``
     (contour/thickness) — two distinct classes with distinct fields, counted
     separately in the generator stats.
  3. ``Slab`` is a CONTAINER that aggregates Box/Extrude children (reversing
     the pre-v8 geometry-direct form).
"""

import pytest

from lite_step.models import Project, Wall, Slab, Box, Extrude, Window
from lite_step.models.primitives import Point
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.ifc.generator import generate_ifc


def _clipped_box_building():
    """A standalone Box carved down to the overlap with a clip Box."""
    b = Project(name="intersect")
    body = Box(start=Point(x=0, y=0, z=0), end=Point(x=4000, y=4000, z=3000),
               name="body", type="wall")
    clip = Box(start=Point(x=1000, y=1000, z=0), end=Point(x=3000, y=3000, z=3000),
               name="clip")
    body.intersection(clip)
    b.add(body)
    return normalize_project_to_meters(b)


class TestIntersection:
    def test_intersection_emits_on_ifcopenshell(self):
        res = generate_ifc(_clipped_box_building())
        assert res.success, res.error
        assert 'INTERSECTION' in res.ifc_content

    def test_intersection_round_trips_through_normalizer(self):
        # The reader (normalizer._extract_boolean) must classify an
        # IfcBooleanResult INTERSECTION as "intersection", not collapse it to
        # "union" — otherwise patch-mode round-trips lose the operation.
        from lite_step.ifc.normalizer import _extract_boolean

        class _FakeItem:
            Operator = "INTERSECTION"
            FirstOperand = None
            SecondOperand = None

        info = _extract_boolean(_FakeItem(), None)
        assert info.boolean_operation == "intersection"


class TestBoxExtrudeSplit:
    def test_box_and_extrude_are_distinct_classes(self):
        assert Box is not Extrude
        # Box carries start/end; Extrude carries contour/thickness.
        assert 'start' in Box.model_fields and 'contour' not in Box.model_fields
        assert 'contour' in Extrude.model_fields and 'start' not in Extrude.model_fields

    def test_generator_counts_boxes_and_extrudes_separately(self):
        b = Project(name="split")
        b.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000),
                  type="sketch"))
        b.add(Extrude(contour=[Point(x=0, y=0, z=2000), Point(x=1000, y=0, z=2000),
                               Point(x=1000, y=0, z=3000), Point(x=0, y=0, z=3000)],
                      thickness=100, type="sketch"))
        res = generate_ifc(normalize_project_to_meters(b))
        assert res.success, res.error
        assert res.stats["boxes"] == 1
        assert res.stats["extrudes"] == 1


class TestSlabContainer:
    def test_slab_is_a_container_of_box_extrude(self):
        slab = Slab(name="foundation")
        slab.add(Box(start=Point(x=-500, y=-500, z=-300),
                     end=Point(x=500, y=500, z=0)))
        assert len(slab.elements) == 1
        # A Slab takes products too now ("elements can contain other
        # elements"); what it still refuses is an OPENING, which has no
        # coordinates for world-coordinate containment to keep.
        slab.add(Wall(name="w"))
        assert len(slab.elements) == 2
        with pytest.raises(ValueError, match="cannot be ADDED"):
            slab.add(Window(width=900, height=1200, name="w0"))

    def test_slab_container_compiles_to_ifcslab(self):
        b = Project(name="slab")
        slab = Slab(name="foundation")
        slab.add(Box(start=Point(x=-5000, y=-4000, z=-300),
                     end=Point(x=5000, y=4000, z=0), type="foundation"))
        b.add(slab)
        assert validate_project_report(b).errors == []
        res = generate_ifc(normalize_project_to_meters(b))
        assert res.success, res.error
        assert 'IFCSLAB' in res.ifc_content.upper()
        assert res.stats.get("slabs") == 1
