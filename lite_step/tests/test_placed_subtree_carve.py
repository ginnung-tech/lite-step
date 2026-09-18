"""``.add()`` carves inside a ``placement=`` subtree.

The bug it closes was silent by construction: a solid ``.add()``ed inside a
placed container carved nothing, emitted nothing, warned about nothing, and
left a solid block exactly where the hole was authored.

The fix is not an exemption removed but a FRAME introduced. A placed element's
AABB is in pre-placement authoring coordinates, so it genuinely cannot be
compared with anything outside. What a blanket skip gets wrong is that
everything INSIDE the placed subtree sits in those same
coordinates, so those comparisons need no matrix at all.
"""
from __future__ import annotations

import pytest

from lite_step.compiler.displacement import (
    _collect_frames, apply_displacement, carve_pairs)
from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.models import (Box, Element, Point, Project, Slab, Storey,
                              Transform)

pytest.importorskip("ifcopenshell")

import ifcopenshell            # noqa: E402
import ifcopenshell.geom       # noqa: E402
import numpy as np             # noqa: E402

OFFSET = 50000


def _wing(placed: bool, atrium: bool = True, no_carve: bool = False) -> Project:
    """A plate with an atrium block through it, optionally inside a placed wing.

    ``placed`` is the ONLY variable — everything else is identical, so any
    difference in the result is attributable to it and to nothing else.
    """
    proj = Project(name="wing")
    storey = Storey(name="ground", elevation=0)
    proj.add_storey(storey)
    wing = Element(name="wing", ifc_class="IfcBuildingElementProxy")
    if placed:
        wing.placement = Transform(origin=Point(x=OFFSET, y=0, z=0))
    plate = Slab(name="plate")
    plate.add(Box(name="body", material="Concrete_C30-37",
                  start=Point(x=0, y=0, z=0), end=Point(x=10000, y=10000, z=300)))
    if atrium:
        block = Box(name="atrium", material="Concrete_C30-37",
                    start=Point(x=3000, y=3000, z=-500),
                    end=Point(x=7000, y=7000, z=800))
        plate.add(block, carve="none" if no_carve else None)
    wing.add(plate)
    storey.add(wing)
    return proj


def _pairs(project) -> list:
    normalized = normalize_project_to_meters(project)
    apply_displacement(normalized)
    return carve_pairs(normalized)


class TestPlacedSubtreeCarves:

    def test_a_placed_subtree_carves_exactly_as_an_unplaced_one(self):
        """The headline, on ONE subtree. ``placement=`` is the only difference
        between the two models, and it must not be the difference between a
        hole and none.

        Scoped to one subtree deliberately: with two, the unplaced model is
        not the right answer to match — see
        ``TestFramesAreNotJustPlacedEqualsUnplaced``.
        """
        unplaced = _pairs(_wing(placed=False))
        placed = _pairs(_wing(placed=True))
        assert unplaced, "the control found no carve — the fixture is broken"
        assert len(placed) == len(unplaced)

    def test_the_carve_is_in_the_right_place_once_placed(self):
        """A carve that emits in the WRONG place is worse than none.

        Measured through ``ifcopenshell`` in WORLD coordinates: the same
        volume must be removed, and the whole solid must sit at the offset the
        placement asked for.
        """
        from lite_step.ifc.generator import generate_ifc

        def measure(placed, atrium):
            result = generate_ifc(normalize_project_to_meters(
                _wing(placed=placed, atrium=atrium)))
            assert result.success, result.error
            model = ifcopenshell.file.from_string(result.ifc_content)
            target = [e for e in model.by_type("IfcProduct")
                      if e.Name and e.Name.startswith("box:body:slab:plate")]
            settings = ifcopenshell.geom.settings()
            settings.set("use-world-coords", True)
            walk = ifcopenshell.geom.iterator(settings, model, include=target)
            assert walk.initialize()
            shape = walk.get()
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            faces = np.array(shape.geometry.faces).reshape(-1, 3)
            volume = abs(sum(float(np.dot(verts[a], np.cross(verts[b], verts[c])))
                             for a, b, c in faces) / 6.0)
            return volume, verts.min(0)

        solid_u, min_u = measure(False, False)
        carved_u, _ = measure(False, True)
        solid_p, _ = measure(True, False)
        carved_p, min_p = measure(True, True)

        hole = 4.0 * 4.0 * 0.3           # a 4x4 m block through a 300 mm plate
        assert solid_u - carved_u == pytest.approx(hole, abs=1e-6)
        assert solid_p - carved_p == pytest.approx(hole, abs=1e-6), (
            "the placed plate lost a different volume — the operand is being "
            "carved in the wrong frame")
        assert min_p[0] - min_u[0] == pytest.approx(OFFSET / 1000.0, abs=1e-9), (
            "the placement stopped being applied")

    def test_no_carve_still_exempts_inside_a_placed_subtree(self):
        """The escape hatch has to keep working in the new frame, or authors
        who relied on ``placement=`` to mean "leave this alone" are left with
        no replacement for it."""
        assert _pairs(_wing(placed=True, no_carve=True)) == []

    def test_a_model_with_no_placement_has_exactly_one_frame(self):
        """The byte-safety argument, asserted rather than assumed.

        One frame means one pairing pass over the same list in the same order,
        which is what keeps ``host._cuts`` append order — and therefore
        ``boolean_tree.balanced_union``'s POSITIONAL pairing, and therefore the
        emitted bytes — identical for every model that places nothing.
        """
        normalized = normalize_project_to_meters(_wing(placed=False))
        frames = _collect_frames(normalized)
        assert len(frames) == 1
        assert frames[0][0] is None, "the world frame must be rooted at None"

    def test_a_placed_model_has_a_second_frame_rooted_at_the_placed_element(self):
        normalized = normalize_project_to_meters(_wing(placed=True))
        frames = _collect_frames(normalized)
        assert len(frames) == 2
        assert frames[0][0] is None
        assert getattr(frames[1][0], "name", None) == "wing"
        assert not frames[0][1], "the placed solids leaked into the world frame"


class TestCrossFrameStaysExempt:
    """The limit, stated and pinned.

    Judging a placed solid against an UNPLACED one needs the resolved
    placement matrix, and ``resolve_placement_matrices`` runs later than this
    pass. So cross-frame carving is still out — and that has to be asserted,
    because it is the difference between a bounded fix and a claim that
    everything now carves everything.
    """

    def test_a_placed_solid_does_not_carve_an_unplaced_neighbour(self):
        proj = Project(name="cross")
        storey = Storey(name="ground", elevation=0)
        proj.add_storey(storey)

        # An unplaced slab sitting at the origin...
        outside = Slab(name="ground_slab")
        outside.add(Box(name="body", material="Concrete_C30-37",
                        start=Point(x=0, y=0, z=0),
                        end=Point(x=10000, y=10000, z=300)))
        storey.add(outside)

        # ...and a placed post whose PRE-placement coordinates overlap it.
        post = Element(name="post_holder", ifc_class="IfcBuildingElementProxy")
        post.placement = Transform(origin=Point(x=OFFSET, y=0, z=0))
        post.add(Box(name="post", material="Concrete_C30-37",
                     start=Point(x=1000, y=1000, z=-500),
                     end=Point(x=1300, y=1300, z=500)))
        storey.add(post)

        labels = [(host, occ) for host, occ, _kind in _pairs(proj)]
        assert not any("ground_slab" in host and "post" in occ
                       for host, occ in labels), (
            "a placed solid carved an unplaced one from its PRE-placement "
            "coordinates — that is the ambiguity the frame split exists to "
            "avoid, not to resolve")


class TestFramesAreNotJustPlacedEqualsUnplaced:
    """The design's real payoff, which "make placed behave like unplaced"
    would have missed.

    The unitized-curtain-wall pattern places the SAME Product occurrence once
    per storey and relies on ``placement=`` to separate the levels. Before
    placement resolves, every level is authored at identical coordinates — so
    an unplaced reading of that model has every level interpenetrating every
    other, and the inferred carve dutifully pairs them.

    Frames make the placed model MORE correct than the unplaced one, not
    merely equal to it: each holder is judged against its own contents and
    nothing else, so the cross-level pairs never arise.
    """

    @staticmethod
    def _levels(placed: bool, n: int = 2) -> Project:
        proj = Project(name="cw")
        ground = Storey(name="ground", elevation=0)
        proj.add_storey(ground)
        source = Slab(name="finger")
        source.add(Box(name="body", material="Concrete_C30-37",
                       start=Point(x=0, y=0, z=0),
                       end=Point(x=24000, y=60000, z=300)))
        from lite_step.models import Product
        catalog = Product(source, name="ward_slab_24x60")
        for level in range(n):
            holder = Element(ifc_class="IfcBuildingElementProxy",
                             name=f"slab_l{level}")
            holder.add(catalog.occurrence(name="unit"))
            holder.add(Box(name=f"riser_l{level}", material="Concrete_C30-37",
                           start=Point(x=8000, y=20000, z=-500),
                           end=Point(x=10000, y=22000, z=800)))
            if placed:
                holder.placement = Transform(
                    origin=Point(x=0, y=0, z=level * 3500))
            ground.add(holder)
        return proj

    def test_each_holder_is_carved_only_by_its_own_riser(self):
        inferred = [(h, o) for h, o, kind in _pairs(self._levels(placed=True))
                    if kind == "inferred"]
        assert len(inferred) == 2, inferred
        for host, occ in inferred:
            level_of_host = host.split("slab_l")[1][0]
            level_of_occ = occ.split("slab_l")[1][0]
            assert level_of_host == level_of_occ, (
                f"riser on level {level_of_occ} carved level {level_of_host} "
                f"— frames are leaking into each other")

    def test_the_unplaced_reading_of_the_same_model_is_the_wrong_one(self):
        """Stated as a number so the argument above is not just prose.

        If this ever drops to 2, the levels have stopped coinciding
        pre-placement and this fixture demonstrates nothing.
        """
        inferred = [p for p in _pairs(self._levels(placed=False))
                    if p[2] == "inferred"]
        assert len(inferred) > 2, (
            "the unplaced model was expected to over-carve — every level is "
            "authored at the same coordinates before placement separates them")
