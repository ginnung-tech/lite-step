"""``IfcRelAssociatesMaterial`` membership is accumulated, then written once.

The defect this pins was a performance one with no visible symptom:
``associate_material`` did

    rel.RelatedObjects = list(rel.RelatedObjects) + [product]

which reads the whole aggregate out of the C++ layer, appends one element and
writes it all back — so associating N products with one material costs O(N^2).
Measured on ``unitized-curtain-wall``: 15,000 calls, 74.3M
``entity_instance.walk`` steps; deferring the write took the model from 530 s
to 442 s and dropped the walk to 3.4M.

Nothing about the OUTPUT was supposed to change, which is exactly why these
assertions are about membership and ORDER rather than about speed. A timing
test would be flaky and would not catch the failure that matters — a
rewrite that is fast and silently drops or reorders members.
"""
from __future__ import annotations

import re

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc
from lite_step.models import Box, Point, Project, Slab, Storey

pytest.importorskip("ifcopenshell")

import ifcopenshell  # noqa: E402

MATERIAL = "Concrete_C30-37"


def _model(n: int) -> Project:
    """``n`` slabs, all of one material — so all n share ONE relationship."""
    proj = Project(name="assoc")
    storey = Storey(name="ground", elevation=0)
    proj.add_storey(storey)
    for i in range(n):
        slab = Slab(name=f"s{i}")
        slab.add(Box(name="body", material=MATERIAL,
                     start=Point(x=i * 2000, y=0, z=0),
                     end=Point(x=i * 2000 + 1000, y=1000, z=200)))
        storey.add(slab)
    return proj


def _associations(project: Project):
    result = generate_ifc(normalize_project_to_meters(project))
    assert result.success, result.error
    model = ifcopenshell.file.from_string(result.ifc_content)
    return model, model.by_type("IfcRelAssociatesMaterial")


class TestMembershipSurvivesTheDeferredWrite:

    @pytest.mark.parametrize("n", [1, 2, 7])
    def test_every_product_reaches_the_relationship(self, n):
        """The headline. A member that never gets flushed is a product with no
        material in the file — and it still compiles, validates and renders."""
        _model_ifc, rels = _associations(_model(n))
        concrete = [r for r in rels
                    if r.RelatingMaterial.Name == MATERIAL]
        assert len(concrete) == 1, "one relationship per material key"
        assert len(concrete[0].RelatedObjects) == n

    def test_the_single_member_case_is_written_at_creation(self):
        """``flush_material_associations`` deliberately skips relationships
        that never grew, so this path must not depend on the flush at all."""
        _model_ifc, rels = _associations(_model(1))
        assert len(rels[0].RelatedObjects) == 1

    def test_order_is_the_association_order(self):
        """Order is what makes the deferred write byte-identical rather than
        merely equivalent — the emitted aggregate is a LIST, and reordering it
        moves bytes even though it means the same thing."""
        _model_ifc, rels = _associations(_model(5))
        concrete = [r for r in rels if r.RelatingMaterial.Name == MATERIAL][0]
        names = [o.Name for o in concrete.RelatedObjects]

        # Canonical names are leaf-first — `box:body:slab:s3:storey:ground` —
        # so the authoring index is the `s<N>` segment, not a fixed position.
        def authored_index(canonical: str) -> int:
            for segment in canonical.split(":"):
                if re.fullmatch(r"s\d+", segment):
                    return int(segment[1:])
            raise AssertionError(f"no slab segment in {canonical!r}")

        assert [authored_index(n) for n in names] == [0, 1, 2, 3, 4], (
            f"members came out reordered: {names}")


class TestFlushIsIdempotent:

    def test_flushing_twice_does_not_double_the_membership(self):
        """The flush ASSIGNS rather than appends. If it ever grows an append,
        a second call — or a caller that flushes defensively — silently emits
        every product twice."""
        from lite_step.ifc.entity_cache import EntityCache

        model = ifcopenshell.file(schema="IFC4X3_ADD2")
        cache = EntityCache(model)
        material = model.create_entity("IfcMaterial", Name=MATERIAL)
        products = [model.create_entity(
            "IfcBuildingElementProxy",
            GlobalId=ifcopenshell.guid.new(), Name=f"p{i}") for i in range(4)]
        for product in products:
            cache.associate_material(product, material)

        cache.flush_material_associations()
        once = list(model.by_type("IfcRelAssociatesMaterial")[0].RelatedObjects)
        cache.flush_material_associations()
        twice = list(model.by_type("IfcRelAssociatesMaterial")[0].RelatedObjects)

        assert len(once) == 4
        assert once == twice
