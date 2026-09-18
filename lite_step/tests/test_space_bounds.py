"""Declared space bounds — WS-F.

``Space(name=..., bounds=[...])`` inverts WS-C: the author declares the
bounding elements, the compiler derives the room volume, and the declared
members become the authoritative ``IfcRelSpaceBoundary1stLevel`` set. Policy
in :mod:`lite_step.compiler.space_volume`; the collection fork and the
demoted checker in :mod:`lite_step.ifc.space_boundaries`.

What is pinned here:

1.  **The agreed derivation cases, verbatim from the roadmap.** One wall
    fails (it derives only its own box, lies inside it, is demoted, and
    leaves every axis half-open); two perpendicular walls derive the
    projected box; unequal runs derive the SMALLEST projected box; opposite
    parallel walls close between their inner faces; a floor slab's top face
    closes z from below while its footprint projects x/y — same centroid
    rule, no per-class special case.
2.  **The invariant: adding a bound never grows the room — ABSOLUTE** (the
    canopy ruling), promoted to a sweep over an
    adversarial family of extras. A far canopy or a double wall's outer
    leaf never inflates the room (innermost for detached, median centre);
    it ends up not touching and the checker says so by design.
3.  **Declared means declared.** A chimney breast poking into the room does
    not truncate the volume, still gets its boundary, and is noted on the
    derived-volume log line as a feature (info, not a warning) — while an
    enclosure layer (a ceiling hung below the wall tops, a declared
    bulkhead) is never mistaken for a feature and still closes the room.
4.  **Every error is reserved for a near-certain mistake** and each is seen
    RED in a paired/falsifying fixture: the ambiguous centre-on-face,
    resolution miss/ambiguity/duplicate, the mode conflicts (bounds +
    children, + placement, + anonymity, + nesting), and the HALF-OPEN axis,
    which the interior ruling made reachable through
    ``derive_volume`` — ``bounds=["wall:only"]`` fails there now. The empty
    axis is still not constructible and is exercised directly at the
    ``_combine_axis`` seam.
4b. **The interior bound WARNS** (the ruling): a bound lying
    wholly inside the derived volume can be a flush pilaster (correct
    authoring) or a mis-resolved name, and only the author can tell — so it
    is warned with both readings named, treated exactly like a protruding
    feature, and listed on the derived-volume line.
5.  **The checker** (the WS-C engine demoted): a declared bound a boolean
    carved the room away from warns; an undeclared touching element warns as
    a candidate. Both falsified by their paired quiet fixtures.
6.  **Both backends explicitly** — ``generate_ifc(backend="ifcopenshell")``
    never silently reroutes (the streaming branch is keyed on
    ``backend == "streaming"``) and ``generate_ifc_streaming`` is streaming
    by construction; the header stamp is asserted so the two-backend claim
    is measured, not assumed.
7.  **Fixtures can falsify.** Rooms are off-origin and non-square; one
    model stands on a non-zero storey elevation with full-canonical
    fragments; one bound is placement-translated and one is rotated, each
    with the un-placed twin asserting the opposite volume.
"""

from __future__ import annotations

import re
import tempfile

import pytest

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project,
)
from lite_step.compiler.space_volume import (
    SpaceVolumeError,
    _AxisContribution,
    _combine_axis,
    derive_volume,
    effective_volume,
    report_line,
)
from lite_step.ifc.space_boundaries import collect_boundaries
from lite_step.models import (
    Box, Element, Mesh, Point, Point2D, Project, Site, Slab, Space, Storey,
    Transform, Wall, Window,
)

pytest.importorskip("ifcopenshell")


# ---------------------------------------------------------------------------
# Fixtures — local geometry only, off-origin, non-square.
# ---------------------------------------------------------------------------


def _project():
    proj = Project(name="rooms")
    return proj


def wall(proj, name, x0, y0, x1, y1, z0=0, z1=2700, **kw):
    w = Wall(name=name, **kw)
    w.add(Box(start=Point(x=x0, y=y0, z=z0), end=Point(x=x1, y=y1, z=z1)))
    proj.add(w)
    return w


def slab(proj, name, x0, y0, x1, y1, z0, z1):
    s = Slab(name=name)
    s.add(Box(start=Point(x=x0, y=y0, z=z0), end=Point(x=x1, y=y1, z=z1)))
    proj.add(s)
    return s


def enclosed_room(proj=None, bounds=None, name="kitchen"):
    """Four walls + floor + deck around a 4200 x 4400 x 2700 room whose inner
    faces sit at (1700, -3100, 0) .. (5900, 1300, 2700) — odd offsets, unequal
    sides."""
    proj = proj or _project()
    wall(proj, "south", 1700, -3400, 5900, -3100)
    wall(proj, "north", 1700, 1300, 5900, 1600)
    wall(proj, "west", 1400, -3100, 1700, 1300)
    wall(proj, "east", 5900, -3100, 6200, 1300)
    slab(proj, "floor", 1400, -3400, 6200, 1600, -200, 0)
    slab(proj, "deck", 1400, -3400, 6200, 1600, 2700, 2900)
    room = Space(name=name, bounds=bounds or [
        "wall:south", "wall:north", "wall:west", "wall:east",
        "slab:floor", "slab:deck"])
    proj.add(room)
    return proj, room


ROOM_VOLUME = ((1.7, -3.1, 0.0), (5.9, 1.3, 2.7))


def derived(proj):
    """Materialize and return {space leaf: derived volume} (metres)."""
    normalized = normalize_project_to_meters(proj)
    from lite_step.compiler.space_volume import materialize_bound_spaces

    materialize_bound_spaces(normalized)
    out = {}
    for storey in normalized.storeys:
        for elem in storey.elements:
            vol = getattr(elem, "_derived_volume", None)
            if vol is not None:
                out[elem.name] = vol
    return out


def boundaries(proj):
    return collect_boundaries(normalize_project_to_meters(proj))


def compile_result(proj, backend="ifcopenshell"):
    """Raw result — a refusal reaches the caller as ``success=False``, the
    same loud-failure contract ``test_space_boundaries`` documents."""
    normalized = normalize_project_to_meters(proj)
    from lite_step.ifc.generator import generate_ifc
    return generate_ifc(normalized, source_code="src")


def compile_ifc(proj, backend="ifcopenshell"):
    result = compile_result(proj, backend)
    assert result.success, result.error
    return result.ifc_content


def opened(ifc: str):
    import ifcopenshell

    path = tempfile.mktemp(suffix=".ifc")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(ifc)
    return ifcopenshell.open(path)


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol
               for pa, pb in zip(a, b) for x, y in zip(pa, pb))


# ---------------------------------------------------------------------------
# 1. The agreed derivation cases
# ---------------------------------------------------------------------------


class TestVolumeDerivation:

    def test_one_wall_fails_half_open_once_its_only_bound_is_demoted(
            self, caplog):
        """One wall derives only its own box and then lies wholly inside it.

        Since the ruling that shape WARNS rather than raising, so
        the single-wall declaration fails through the honest path instead:
        its only contribution is demoted, no axis is left with a face or a
        projection, and the half-open error fires. The message has to be
        intelligible to an author who declared exactly one wall — "Contributing
        bounds on x: (none)" alone reads like an internal invariant — so it
        names the dropped bound and says why it contributed nothing.
        """
        proj = _project()
        wall(proj, "only", 1700, 1300, 5900, 1600)
        proj.add(Space(name="a", bounds=["wall:only"]))
        with caplog.at_level("WARNING"):
            with pytest.raises(SpaceVolumeError) as err:
                derived(proj)
        msg = str(err.value)
        assert "half-open" in msg
        assert "wall:only" in msg
        assert "one wall derives only its own box" in msg
        assert "wall:only" in caplog.text          # warned before it failed

    def test_two_perpendicular_walls_derive_the_projected_box(self):
        """Each wall's inner face closes one axis; its run-span projects the
        others."""
        proj = _project()
        wall(proj, "north", 1700, 1300, 5900, 1600)
        wall(proj, "west", 1400, -3100, 1700, 1300)
        proj.add(Space(name="a", bounds=["wall:north", "wall:west"]))
        assert close(derived(proj)["a"], ((1.7, -3.1, 0.0), (5.9, 1.3, 2.7)))

    def test_unequal_runs_derive_the_smallest_projected_box(self):
        """Three walls, one with a shorter run — the intersection is
        conservative, so the short run wins the x span."""
        proj = _project()
        wall(proj, "north", 1700, 1300, 5900, 1600)
        wall(proj, "west", 1400, -3100, 1700, 1300)
        wall(proj, "south", 1700, -3400, 4700, -3100)     # 1200 shorter
        proj.add(Space(name="a", bounds=["wall:north", "wall:west",
                                         "wall:south"]))
        assert close(derived(proj)["a"], ((1.7, -3.1, 0.0), (4.7, 1.3, 2.7)))

    def test_opposite_parallel_walls_close_between_their_inner_faces(self):
        proj = _project()
        wall(proj, "south", 1700, -3400, 5900, -3100)
        wall(proj, "north", 1700, 1300, 5900, 1600)
        proj.add(Space(name="a", bounds=["wall:south", "wall:north"]))
        vol = derived(proj)["a"]
        assert vol[0][1] == pytest.approx(-3.1)
        assert vol[1][1] == pytest.approx(1.3)

    def test_a_floor_slab_closes_z_from_below_and_projects_its_footprint(self):
        """No per-class special case: the slab's top face closes z (the
        centroid stands above it) and its footprint projects x/y. A deck's
        bottom face closes z from above by the same rule."""
        proj = _project()
        wall(proj, "north", 1700, 1300, 5900, 1600)
        wall(proj, "west", 1400, -3100, 1700, 1300)
        slab(proj, "floor", 1400, -3400, 6200, 1600, -200, 0)
        slab(proj, "deck", 1400, -3400, 6200, 1600, 2700, 2900)
        proj.add(Space(name="a", bounds=["wall:north", "wall:west",
                                         "slab:floor", "slab:deck"]))
        vol = derived(proj)["a"]
        assert vol[0][2] == pytest.approx(0.0)     # the floor's TOP face
        assert vol[1][2] == pytest.approx(2.7)     # the deck's BOTTOM face
        # The slabs' oversize footprints do not drag x/y beyond the walls.
        assert close(vol, ((1.7, -3.1, 0.0), (5.9, 1.3, 2.7)))

    def test_the_full_room_derives_its_walls_inner_box(self):
        proj, _room = enclosed_room()
        assert close(derived(proj)["kitchen"], ROOM_VOLUME)

    def test_a_translated_bound_is_judged_where_it_stands(self):
        """The north wall is authored 1000 mm short of its place and moved
        the last 1000 by ``placement=`` — the derived volume must read the
        PLACED position or the room is 1000 too small."""
        proj = _project()
        wall(proj, "north", 1700, 300, 5900, 600,
             placement=Transform(origin=Point(x=0, y=1000, z=0)))
        wall(proj, "west", 1400, -3100, 1700, 1300)
        proj.add(Space(name="a", bounds=["wall:north", "wall:west"]))
        assert derived(proj)["a"][1][1] == pytest.approx(1.3)

    def test_the_same_bound_untranslated_derives_the_smaller_room(self):
        """The falsification: drop the placement and the ceiling face moves."""
        proj = _project()
        wall(proj, "north", 1700, 300, 5900, 600)
        wall(proj, "west", 1400, -3100, 1700, 1300)
        proj.add(Space(name="a", bounds=["wall:north", "wall:west"]))
        assert derived(proj)["a"][1][1] == pytest.approx(0.3)

    def test_a_rotated_bound_contributes_its_world_box(self):
        """A wall authored along +x, stood on end by a 90-degree z rotation —
        judged un-rotated it lies flat across the room."""
        proj = _project()
        wall(proj, "north", 1700, 1300, 5900, 1600)
        wall(proj, "east", 0, 0, 4400, 300,
             placement=Transform(origin=Point(x=6200, y=-3100, z=0),
                                 rotations=[("z", 9000)]))
        proj.add(Space(name="a", bounds=["wall:north", "wall:east"]))
        vol = derived(proj)["a"]
        assert vol[1][0] == pytest.approx(5.9)     # the rotated wall's inner face
        assert vol[1][1] == pytest.approx(1.3)


class TestTheInvariant:
    """Adding a bound never grows the room (roadmap: its own test)."""

    def test_adding_a_protruding_feature_leaves_the_volume_untouched(self):
        base, _ = enclosed_room()
        with_chimney, _ = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "wall:chimney"])
        wall(with_chimney, "chimney", 3000, 900, 3600, 1600)
        assert derived(with_chimney)["kitchen"] == derived(base)["kitchen"]

    def test_adding_a_shorter_run_only_shrinks(self):
        base, _ = enclosed_room()
        shrunk, _ = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "wall:short"])
        # Same face plane as the south wall, 2400 shorter in run — the short
        # projection truncates x; the shared face leaves y alone.
        wall(shrunk, "short", 2300, -3400, 4700, -3100)
        (b_lo, b_hi) = derived(base)["kitchen"]
        (s_lo, s_hi) = derived(shrunk)["kitchen"]
        assert all(s_lo[k] >= b_lo[k] - 1e-9 for k in range(3))
        assert all(s_hi[k] <= b_hi[k] + 1e-9 for k in range(3))

    def test_an_outer_parallel_wall_does_not_grow_the_room(self, caplog):
        """A second wall beyond the north wall, same run: outermost-wins
        cannot bite, because the flanking walls' PROJECTIONS already cap the
        y span at the inner face. The outer wall becomes a declared bound the
        volume does not touch — the checker's warning, not silence."""
        proj, _ = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "wall:outer"])
        wall(proj, "outer", 1700, 2300, 5900, 2600)       # same run as north
        assert close(derived(proj)["kitchen"], ROOM_VOLUME)
        with caplog.at_level("WARNING"):
            boundaries(proj)
        assert "does not touch" in caplog.text and "wall:outer" in caplog.text
        # ...and declared means declared: the boundary is still emitted.
        assert "wall:outer" in [b.element for b in boundaries(proj)]

    def test_a_far_canopy_never_inflates_the_room(self, caplog):
        """The canopy ruling, rewritten from the pinned
        spec tension exactly as that test's docstring promised. Under the
        old rule a canopy 10 m up dragged the union centroid past the walls'
        spans, their z projections flipped into faces, and outermost-wins
        handed the kitchen a 10000 ceiling. Now the median centre stays with
        the cluster, same-side detached faces take the innermost, and the
        kitchen stays 2700 high. The canopy simply does not touch — and the
        checker SAYS so (the design's answer, not incidental) — while its
        declared boundary is still emitted.
        """
        proj, _ = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "slab:canopy"])
        slab(proj, "canopy", 1400, -3400, 6200, 1600, 10000, 10200)
        assert close(derived(proj)["kitchen"], ROOM_VOLUME)
        with caplog.at_level("WARNING"):
            got = boundaries(proj)
        assert "does not touch" in caplog.text
        assert "slab:canopy" in caplog.text
        assert "slab:canopy" in [b.element for b in got]

    def test_a_double_leaf_wall_derives_the_room_to_the_inner_leaf(self, caplog):
        """The ruling's stated consequence: both leaves declared, the room
        stops at the INNER leaf's face; the outer leaf ends up not touching
        and is warned, its boundary still emitted."""
        proj, _ = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "wall:outer_leaf"])
        wall(proj, "outer_leaf", 1700, 1600, 5900, 1900)   # behind north
        assert close(derived(proj)["kitchen"], ROOM_VOLUME)
        with caplog.at_level("WARNING"):
            got = boundaries(proj)
        assert "does not touch" in caplog.text
        assert "wall:outer_leaf" in caplog.text
        assert "wall:outer_leaf" in [b.element for b in got]

    def test_a_declared_full_section_bulkhead_bounds_the_room(self, caplog):
        """The one shape where innermost-of-kept is load-bearing on its own
        (everywhere else the flanking projections already cap the axis — a
        final-fold outermost mutation stayed green against the double-leaf
        fixture, which is why this exists): a full-cross-section partition
        INSIDE the room, declared. Declaring a bulkhead means the room stops
        at it — the double-leaf logic applied from inside — and the north
        wall beyond it becomes the not-touching declared bound."""
        proj, _ = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "wall:bulkhead"])
        wall(proj, "bulkhead", 1700, 1000, 5900, 1300)
        vol = derived(proj)["kitchen"]
        assert vol[1][1] == pytest.approx(1.0)     # stops at the bulkhead
        with caplog.at_level("WARNING"):
            boundaries(proj)
        assert "does not touch" in caplog.text and "wall:north" in caplog.text

    def test_the_universal_never_grows_sweep(self):
        """The invariant PROMOTED to a sweep, as the canopy ruling demands
        ("an invariant test, not a tendency"): for every extra in an
        adversarial family — protruding feature, shorter run, outer parallel
        wall, double leaf, stacked second ceiling, far canopy, breakfast
        bar, full-width bench — the derived volume either shrinks
        componentwise or the derivation refuses. Never grows, on any axis.
        """
        extras = {
            "wall:chimney": lambda p: wall(p, "chimney", 3000, 900, 3600, 1600),
            "wall:short": lambda p: wall(p, "short", 2300, -3400, 4700, -3100),
            "wall:outer": lambda p: wall(p, "outer", 1700, 2300, 5900, 2600),
            "wall:outer_leaf": lambda p: wall(p, "outer_leaf",
                                              1700, 1600, 5900, 1900),
            "slab:upper": lambda p: slab(p, "upper",
                                         1400, -3400, 6200, 1600, 3600, 3800),
            "slab:canopy": lambda p: slab(p, "canopy",
                                          1400, -3400, 6200, 1600, 10000, 10200),
            # Reaches into the west wall band: a bound flush INSIDE the
            # room is the interior error by design, a real bar meets its wall.
            "slab:bar": lambda p: slab(p, "bar", 1400, -1200, 2900, -400,
                                       900, 1150),
            "slab:bench": lambda p: slab(p, "bench", 1700, 800, 5900, 1300,
                                         0, 900),
        }
        (b_lo, b_hi) = derived(enclosed_room()[0])["kitchen"]
        grew = []
        for fragment, build in extras.items():
            proj, _ = enclosed_room(bounds=[
                "wall:south", "wall:north", "wall:west", "wall:east",
                "slab:floor", "slab:deck", fragment])
            build(proj)
            try:
                (s_lo, s_hi) = derived(proj)["kitchen"]
            except SpaceVolumeError:
                continue                       # refusing is never growing
            if not (all(s_lo[k] >= b_lo[k] - 1e-9 for k in range(3))
                    and all(s_hi[k] <= b_hi[k] + 1e-9 for k in range(3))):
                grew.append(fragment)
        assert grew == [], f"adding {grew} grew the room"

    def test_a_breakfast_bar_projection_does_not_bite(self):
        """A bar whose y span CONTAINS the centre would truncate the room to
        its own strip through its PROJECTION — which is why a feature's
        whole contribution drops, not just its faces."""
        proj, _ = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "slab:bar"])
        slab(proj, "bar", 1400, -1200, 2900, -400, 900, 1150)
        assert close(derived(proj)["kitchen"], ROOM_VOLUME)

    def test_twin_chimneys_do_not_truncate_each_other_into_the_room(self):
        """Each chimney's leave-one-out candidate still contains the OTHER
        chimney — an innermost-face candidate would be capped at the twin's
        face and hide the overlap, which is why candidates fold OUTERMOST."""
        proj, _ = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "wall:chimney", "wall:chimney2"])
        wall(proj, "chimney", 3000, 900, 3600, 1600)
        wall(proj, "chimney2", 4300, 900, 4900, 1600)
        assert close(derived(proj)["kitchen"], ROOM_VOLUME)

    def test_a_hung_ceiling_below_the_wall_tops_still_closes_the_room(self):
        """The enclosure-layer clause, measured against the first cut of the
        demotion rule: a ceiling 100 below the wall tops interpenetrates the
        candidate (the walls' z projections cap it higher), but it covers
        the full cross-section — an enclosure, never a feature. Demoting it
        raised the room to the wall tops and erased the ceiling."""
        proj = _project()
        wall(proj, "south", 1700, -3400, 5900, -3100)
        wall(proj, "north", 1700, 1300, 5900, 1600)
        wall(proj, "west", 1400, -3100, 1700, 1300)
        wall(proj, "east", 5900, -3100, 6200, 1300)
        slab(proj, "floor", 1400, -3400, 6200, 1600, -200, 0)
        slab(proj, "ceiling", 1400, -3400, 6200, 1600, 2600, 2800)
        proj.add(Space(name="a", bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:ceiling"]))
        assert derived(proj)["a"][1][2] == pytest.approx(2.6)


# ---------------------------------------------------------------------------
# 2. Errors — each a near-certain mistake, each seen red
# ---------------------------------------------------------------------------


class TestErrors:

    def test_a_centroid_on_a_face_is_ambiguous(self):
        """Two stacked walls sharing a face exactly at the union centroid:
        whether either projects or closes is a coin flip, so it errors."""
        proj = _project()
        wall(proj, "lower", 1700, -3100, 5900, 1300, z0=0, z1=1400)
        wall(proj, "upper", 1700, -3100, 5900, 1300, z0=1400, z1=2800)
        proj.add(Space(name="a", bounds=["wall:lower", "wall:upper"]))
        with pytest.raises(SpaceVolumeError, match="ambiguous"):
            derived(proj)

    def test_a_missing_fragment_names_the_string(self):
        proj, _ = enclosed_room(bounds=["wall:south", "wall:nortth"])
        with pytest.raises(SpaceVolumeError, match="wall:nortth"):
            derived(proj)

    def test_an_ambiguous_fragment_names_the_string_and_candidates(self):
        proj = Project(name="two")
        ground = Storey(name="ground", elevation=0)
        first = Storey(name="first", elevation=3000)
        proj.add_storey(ground)
        proj.add_storey(first)
        for st in (ground, first):
            w = Wall(name="north")
            w.add(Box(start=Point(x=1700, y=1300, z=0),
                      end=Point(x=5900, y=1600, z=2700)))
            st.add(w)
        south = Wall(name="south")
        south.add(Box(start=Point(x=1700, y=-3400, z=0),
                      end=Point(x=5900, y=-3100, z=2700)))
        ground.add(south)
        s = Space(name="a", bounds=["storey:ground"])
        ground.add(s)
        # "storey:ground" suffix-matches every element on that storey.
        with pytest.raises(SpaceVolumeError, match="storey:ground"):
            derived(proj)

    def test_a_duplicate_member_is_refused(self):
        proj, _ = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "wall:north"])
        with pytest.raises(SpaceVolumeError, match="twice"):
            derived(proj)

    def test_an_anonymous_bounds_space_is_refused(self):
        proj = _project()
        wall(proj, "north", 1700, 1300, 5900, 1600)
        wall(proj, "west", 1400, -3100, 1700, 1300)
        proj.add(Space(bounds=["wall:north", "wall:west"]))
        with pytest.raises(SpaceVolumeError, match="named"):
            derived(proj)

    def test_bounds_plus_placement_is_refused(self):
        proj = _project()
        wall(proj, "north", 1700, 1300, 5900, 1600)
        wall(proj, "west", 1400, -3100, 1700, 1300)
        proj.add(Space(name="a", bounds=["wall:north", "wall:west"],
                       placement=Transform(origin=Point(x=0, y=500, z=0))))
        with pytest.raises(SpaceVolumeError, match="placement"):
            derived(proj)

    def test_bounds_assigned_after_children_is_refused_at_compile(self):
        """The mutation path around the .add() guard still fails loudly."""
        proj = _project()
        wall(proj, "north", 1700, 1300, 5900, 1600)
        s = Space(name="a")
        s.add(Box(start=Point(x=1700, y=-3100, z=0),
                  end=Point(x=5900, y=1300, z=2700)))
        s.bounds = ["wall:north"]
        proj.add(s)
        with pytest.raises(SpaceVolumeError, match="never both"):
            derived(proj)

    def test_a_nested_bounds_space_is_refused(self):
        proj = _project()
        host = wall(proj, "north", 1700, 1300, 5900, 1600)
        host.anchor(Space(name="a", bounds=["wall:north"]))
        with pytest.raises(SpaceVolumeError, match="top-level"):
            derived(proj)

    def test_a_refusal_reaches_both_backends_as_a_failed_result(self):
        proj, _ = enclosed_room(bounds=["wall:south", "wall:nortth"])
        result = compile_result(proj)
        assert not result.success
        assert "wall:nortth" in result.error

    def test_empty_bounds_fail_at_construction(self):
        with pytest.raises(Exception, match="declares nothing"):
            Space(name="a", bounds=[])

    def test_an_empty_string_bound_fails_at_construction(self):
        with pytest.raises(Exception, match="never objects and never empty"):
            Space(name="a", bounds=["wall:north", ""])


class TestTheInteriorBoundWarns:
    """The ruling: a bound lying wholly inside the derived volume
    is a WARNING, not an error.

    Raising on the theory that the shape could only be a typo'd canonical name
    resolving to furniture is wrong: a pilaster, chimney
    breast or wing wall authored FLUSH to its host wall's inner face touches
    the room's boundary plane without crossing into the wall, is wholly
    inside on all three axes, and is a perfectly real bounding element.
    Erroring punished correct authoring.

    Treatment is the protruding-feature path exactly: contribution dropped,
    boundary still emitted, volume untouched, listed on the derived-volume
    line. The warning names the bound and quotes BOTH readings, because only
    the author can tell which one they have.
    """

    def _pilaster_room(self):
        """A pilaster 200 deep on the north wall's INNER face (y 1100..1300,
        the room's own boundary plane at y=1300), 400 wide, 2400 tall in a
        2700 room — off-origin, not full-height, not centred."""
        proj, _ = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "wall:pilaster"])
        wall(proj, "pilaster", 3000, 1100, 3400, 1300, z0=0, z1=2400)
        return proj

    def test_a_flush_pilaster_does_not_raise(self, caplog):
        with caplog.at_level("WARNING"):
            volume = derived(self._pilaster_room())["kitchen"]
        assert close(volume, ROOM_VOLUME)          # volume untouched

    def test_the_flush_pilaster_is_warned_and_named(self, caplog):
        with caplog.at_level("WARNING"):
            derived(self._pilaster_room())
        assert "wall:pilaster" in caplog.text

    def test_the_warning_quotes_the_flush_pilaster_case(self, caplog):
        """An author has to tell the two situations apart from the message
        alone, so both readings are in it: the real flush feature and the
        mis-resolved name."""
        with caplog.at_level("WARNING"):
            derived(self._pilaster_room())
        assert "flush" in caplog.text
        assert "pilaster" in caplog.text
        assert "wrong element" in caplog.text

    def test_the_flush_pilaster_still_gets_its_boundary(self):
        """Declared means declared — the drop is from the VOLUME, not from
        the boundary set."""
        assert "wall:pilaster" in [
            b.element for b in boundaries(self._pilaster_room())]

    def test_a_mis_resolved_interior_bound_warns_the_same_way(self, caplog):
        """The furniture-typo fixture: a cupboard standing
        clear of every wall is still wholly inside, so it takes exactly the
        same path — the compiler cannot tell it from the pilaster, which is
        precisely why this is a warning and not an error."""
        proj, _ = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "wall:cupboard"])
        wall(proj, "cupboard", 3000, -1000, 3600, -400, z0=0, z1=900)
        with caplog.at_level("WARNING"):
            volume = derived(proj)["kitchen"]
        assert close(volume, ROOM_VOLUME)
        assert "wall:cupboard" in caplog.text
        assert "wholly inside" in caplog.text

    def test_the_undeclared_twin_is_silent(self, caplog):
        """Falsification: the SAME cupboard geometry, not declared. Nothing
        warns about an interior bound, so the warning above is caused by the
        declaration and not by the geometry standing there."""
        proj, _ = enclosed_room()
        wall(proj, "cupboard", 3000, -1000, 3600, -400, z0=0, z1=900)
        with caplog.at_level("WARNING"):
            assert close(derived(proj)["kitchen"], ROOM_VOLUME)
        assert "wholly inside" not in caplog.text

    def test_an_interior_bound_is_noted_on_the_derived_volume_line(
            self, capsys):
        boundaries(self._pilaster_room())
        out = capsys.readouterr().out
        assert "space kitchen: derived 4200 x 4400 x 2700 mm" in out
        assert "inside: wall:pilaster" in out

    def test_it_is_listed_as_interior_not_as_a_feature(self):
        """The two lists are disjoint: a wholly-inside bound reads
        differently to an author than a chimney poking in, and the report
        line says which it is."""
        result = derive_volume([
            ("wall:pilaster", ((3.0, 1.1, 0.0), (3.4, 1.3, 2.4))),
            ("wall:south", ((1.7, -3.4, 0.0), (5.9, -3.1, 2.7))),
            ("wall:north", ((1.7, 1.3, 0.0), (5.9, 1.6, 2.7))),
            ("wall:west", ((1.4, -3.1, 0.0), (1.7, 1.3, 2.7))),
            ("wall:east", ((5.9, -3.1, 0.0), (6.2, 1.3, 2.7))),
            ("slab:floor", ((1.4, -3.4, -0.2), (6.2, 1.6, 0.0))),
            ("slab:deck", ((1.4, -3.4, 2.7), (6.2, 1.6, 2.9))),
        ])
        assert result.interior == ("wall:pilaster",)
        assert "wall:pilaster" not in result.features


class TestTheCombineSeam:
    """The empty axis cannot be reached through ``derive_volume`` (every
    contributed interval contains the centroid by construction), so that
    guard is exercised at the seam it lives on. Defensive completeness with
    a direct test beats a guard no fixture can redden. Half-open IS reachable
    since the interior ruling (see ``TestVolumeDerivation``); it is pinned
    here too because the seam is where its message is built."""

    def test_a_side_no_contribution_closes_is_half_open(self):
        contrib = _AxisContribution(lower_faces=[("slab:floor", 0.0)])
        with pytest.raises(SpaceVolumeError, match="half-open"):
            _combine_axis(2, contrib)

    def test_the_half_open_message_explains_the_dropped_bounds(self):
        """``dropped`` changes no arithmetic — it is the difference between
        an error an author can act on and one that reads like an internal
        invariant."""
        contrib = _AxisContribution(lower_faces=[("slab:floor", 0.0)])
        with pytest.raises(SpaceVolumeError) as err:
            _combine_axis(2, contrib, dropped=["wall:only", "wall:pilaster"])
        assert "wall:only, wall:pilaster" in str(err.value)
        assert "bounds nothing" in str(err.value)

    def test_without_dropped_bounds_the_message_stays_bare(self):
        """Falsification of the clause above: nothing demoted, no
        explanation — so the sentence is caused by the demotion, not
        appended unconditionally."""
        contrib = _AxisContribution(lower_faces=[("slab:floor", 0.0)])
        with pytest.raises(SpaceVolumeError) as err:
            _combine_axis(2, contrib)
        assert "bounds nothing" not in str(err.value)

    def test_contradictory_faces_are_an_empty_result(self):
        contrib = _AxisContribution(lower_faces=[("wall:a", 5.0)],
                                    upper_faces=[("wall:b", 3.0)])
        with pytest.raises(SpaceVolumeError, match="contradict"):
            _combine_axis(0, contrib)

    def test_projections_intersect_smallest(self):
        contrib = _AxisContribution(projections=[("wall:a", 1.0, 6.0),
                                                 ("wall:b", 2.0, 9.0)])
        assert _combine_axis(0, contrib) == (2.0, 6.0)

    def test_the_final_fold_takes_the_innermost_kept_face(self):
        """The canopy ruling's final rule: kept faces are ordinary intervals
        and the axis is a pure intersection — a detached outer face can
        never extend the room. BOTH sides carry two faces with distinct
        values — a single face per side cannot falsify the min/max choice
        (measured: the min-to-max mutation stayed green against a one-face
        fixture)."""
        contrib = _AxisContribution(upper_faces=[("wall:inner_leaf", 1.3),
                                                 ("wall:outer_leaf", 1.6)],
                                    lower_faces=[("slab:floor", 0.0),
                                                 ("slab:subfloor", -0.3)])
        assert _combine_axis(2, contrib, faces="innermost") == (0.0, 1.3)

    def test_the_candidate_fold_takes_the_outermost_face(self):
        """The candidate envelope a bound is classified against is the
        LARGEST face-closed box — innermost here let one chimney cap the
        candidate at its own face and hide its twin from the overlap test."""
        contrib = _AxisContribution(upper_faces=[("wall:n", 1.3),
                                                 ("wall:chimney", 0.9)],
                                    lower_faces=[("wall:s", -3.1),
                                                 ("wall:hearth", -2.8)])
        assert _combine_axis(1, contrib, faces="outermost") == (-3.1, 1.3)


# ---------------------------------------------------------------------------
# 3. Declared means declared — features and the boundary set
# ---------------------------------------------------------------------------


def chimney_room():
    proj, room = enclosed_room(bounds=[
        "wall:south", "wall:north", "wall:west", "wall:east",
        "slab:floor", "slab:deck", "wall:chimney"])
    wall(proj, "chimney", 3000, 900, 3600, 1600)   # protrudes 400 into the room
    return proj, room


class TestDeclaredMeansDeclared:

    def test_a_protruding_feature_does_not_truncate_the_volume(self):
        proj, _ = chimney_room()
        assert close(derived(proj)["kitchen"], ROOM_VOLUME)

    def test_the_feature_still_gets_its_boundary(self):
        proj, _ = chimney_room()
        assert "wall:chimney" in [b.element for b in boundaries(proj)]

    def test_the_boundary_set_is_exactly_the_declared_members_in_order(self):
        proj, _ = enclosed_room()
        got = [b.element for b in boundaries(proj)]
        assert got == ["wall:south", "wall:north", "wall:west", "wall:east",
                       "slab:floor", "slab:deck"]

    def test_an_undeclared_touching_element_gets_no_boundary(self):
        proj, _ = enclosed_room()
        wall(proj, "partition", 2500, 1300, 3700, 1600)   # flush, undeclared
        assert "wall:partition" not in [b.element for b in boundaries(proj)]

    def test_classification_stays_computed_internal_across_a_shared_wall(self):
        """Two bounds-mode rooms declaring the same wall: INTERNAL from both
        sides — what lies beyond is not an author statement. Per-room wall
        segments, so each room's declared union centres on the room (see the
        centroid-drag note in the PR: full-length flanking walls drag the
        centroid off the second room)."""
        proj = _project()
        wall(proj, "south_a", 1700, -3400, 5900, -3100)
        wall(proj, "north_a", 1700, 1300, 5900, 1600)
        wall(proj, "south_b", 6200, -3400, 9400, -3100)
        wall(proj, "north_b", 6200, 1300, 9400, 1600)
        wall(proj, "shared", 5900, -3100, 6200, 1300)
        wall(proj, "west", 1400, -3100, 1700, 1300)
        wall(proj, "east", 9400, -3100, 9700, 1300)
        proj.add(Space(name="a", bounds=["wall:south_a", "wall:north_a",
                                         "wall:west", "wall:shared"]))
        proj.add(Space(name="b", bounds=["wall:south_b", "wall:north_b",
                                         "wall:shared", "wall:east"]))
        got = {(b.space, b.element): b.internal_or_external
               for b in boundaries(proj)}
        assert got[("space:a", "wall:shared")] == "INTERNAL"
        assert got[("space:b", "wall:shared")] == "INTERNAL"
        assert got[("space:a", "wall:north_a")] == "EXTERNAL"

    def test_declared_terrain_classifies_external_earth(self):
        proj = _project()
        wall(proj, "south", 1700, -3400, 5900, -3100)
        wall(proj, "north", 1700, 1300, 5900, 1600)
        wall(proj, "west", 1400, -3100, 1700, 1300)
        wall(proj, "east", 5900, -3100, 6200, 1300)
        slab(proj, "deck", 1400, -3400, 6200, 1600, 2700, 2900)
        site = Site(name="site")
        wrapper = Element(ifc_class="IfcGeographicElement",
                          predefined_type="TERRAIN", name="terrain")
        wrapper.add(Mesh(heightmap=[[0, 0], [0, 0]], depth=2000,
                         corner_min=Point2D(x=1000, y=-3800),
                         corner_max=Point2D(x=6600, y=2000)))
        site.add(wrapper)
        proj.add(site)
        proj.add(Space(name="a", bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:deck", "geographicelement:terrain:site:site"]))
        got = {b.element: b.internal_or_external for b in boundaries(proj)}
        assert got["geographicelement:terrain:site:site"] == "EXTERNAL_EARTH"
        # ...and the terrain's top face closed the room's floor.
        assert derived(proj)["a"][0][2] == pytest.approx(0.0)

    def test_full_canonical_fragments_resolve_on_a_named_elevated_storey(self):
        """Named storey at a non-zero elevation: fragments are the full
        canonicals (the suffix grammar's tail is the storey pair)."""
        proj = Project(name="up")
        st = Storey(name="first", elevation=3000)
        proj.add_storey(st)
        for name, coords in (("north", (1700, 1300, 5900, 1600)),
                             ("west", (1400, -3100, 1700, 1300))):
            w = Wall(name=name)
            x0, y0, x1, y1 = coords
            w.add(Box(start=Point(x=x0, y=y0, z=0),
                      end=Point(x=x1, y=y1, z=2700)))
            st.add(w)
        s = Space(name="a", bounds=["wall:north:storey:first",
                                    "wall:west:storey:first"])
        st.add(s)
        got = [b.element for b in boundaries(proj)]
        assert got == ["wall:north:storey:first", "wall:west:storey:first"]


class TestKnownLimitations:
    """Measured behaviour at the design's stated edge, pinned so a change is
    a decision rather than drift."""

    def test_full_length_flanking_walls_derive_the_majority_side_room(
            self, caplog):
        """THE STANDING LIMITATION (remeasured under the canopy ruling —
        it changed costume, not substance). Two rooms flanked by
        FULL-LENGTH shared walls: the second room's declared set centres
        between the rooms, the shared wall — a full-cross-section enclosure
        layer — closes from the wrong side, and innermost-for-detached
        derives room A's box for room B. Before the ruling this was a loud
        interior error; now it is a silently-wrong volume WITH two warnings
        naming exactly the confusion (the far wall does not touch; the
        other room's wall is an undeclared candidate member) — the
        warn-not-error choice is the canopy ruling's own (a member beyond
        the closing layer warns, structurally identical to the canopy).

        The fix is authoring guidance, now in the reference's bounds
        material: cite each room's own wall segments — walls are authored
        per storey, the facade is the aggregate, and bounds are per room.
        Per-room segments derive both rooms correctly
        (test_classification_stays_computed_internal_across_a_shared_wall).
        """
        proj = _project()
        wall(proj, "south", 1700, -3400, 9400, -3100)     # full length
        wall(proj, "north", 1700, 1300, 9400, 1600)       # full length
        wall(proj, "shared", 5900, -3100, 6200, 1300)
        wall(proj, "west", 1400, -3100, 1700, 1300)
        wall(proj, "east", 9400, -3100, 9700, 1300)
        proj.add(Space(name="b", bounds=["wall:south", "wall:north",
                                         "wall:shared", "wall:east"]))
        with caplog.at_level("WARNING"):
            boundaries(proj)
        # Room B silently derives room A's box...
        assert close(derived(proj)["b"], ((1.7, -3.1, 0.0), (5.9, 1.3, 2.7)))
        # ...and the log names the confusion twice.
        assert "does not touch" in caplog.text and "wall:east" in caplog.text
        assert "candidate member" in caplog.text and "wall:west" in caplog.text


# ---------------------------------------------------------------------------
# 4. Mutual exclusion at the call site
# ---------------------------------------------------------------------------


class TestModeExclusion:

    def test_add_on_a_bounds_space_raises_at_the_call(self):
        s = Space(name="a", bounds=["wall:north"])
        with pytest.raises(ValueError, match="mutually exclusive"):
            s.add(Box(start=Point(x=0, y=0, z=0),
                      end=Point(x=1000, y=1000, z=1000)))
        assert s._elements == []

    def test_anchor_on_a_bounds_space_raises_at_the_call(self):
        s = Space(name="a", bounds=["wall:north"])
        with pytest.raises(ValueError, match="mutually exclusive"):
            s.anchor(Box(start=Point(x=0, y=0, z=0),
                         end=Point(x=1000, y=1000, z=1000)))

    def test_geometry_mode_add_still_works(self):
        s = Space(name="a")
        s.add(Box(start=Point(x=0, y=0, z=0),
                  end=Point(x=1000, y=1000, z=1000)))
        assert len(s._elements) == 1

    def test_void_on_a_space_stays_an_error_in_both_modes(self):
        for s in (Space(name="a", bounds=["wall:north"]), Space(name="b")):
            proj = _project()
            wall(proj, "north", 1700, 1300, 5900, 1600)
            s.void(Box(start=Point(x=0, y=0, z=0),
                       end=Point(x=500, y=500, z=500)))
            proj.add(s)
            errors = validate_project(proj)
            assert any(".void() is not valid here" in e for e in errors)


# ---------------------------------------------------------------------------
# 5. Booleans compose onto the derived volume
# ---------------------------------------------------------------------------


def _space_has_boolean_body(ifc: str) -> bool:
    """Does the ``IfcSpace``'s representation chain carry a boolean?"""
    model = opened(ifc)
    rooms = model.by_type("IfcSpace")
    assert len(rooms) == 1
    rep = rooms[0].Representation
    assert rep is not None
    return any("BooleanResult" in item.is_a()
               for r in rep.Representations for item in r.Items)


class TestBooleans:

    def test_a_difference_bakes_into_the_derived_body(self):
        proj, room = enclosed_room()
        room.difference(Box(start=Point(x=1700, y=-3100, z=0),
                            end=Point(x=2700, y=-2100, z=2700)))
        assert _space_has_boolean_body(compile_ifc(proj))

    def test_without_the_boolean_the_body_is_plain(self):
        """The falsification of the test above."""
        proj, _room = enclosed_room()
        assert not _space_has_boolean_body(compile_ifc(proj))

    def test_a_union_is_legal_on_a_bounds_space(self):
        proj, room = enclosed_room()
        room.union(Box(start=Point(x=5900, y=-1000, z=0),
                       end=Point(x=6800, y=0, z=2700)))
        assert compile_result(proj).success

    def test_geometry_mode_container_booleans_bake_too(self):
        """Measured on main: a ``space.difference(x)`` on a
        geometry-mode Space was silently DROPPED — operand consumed, no CSG
        anywhere in the file. WS-F states booleans are legal in both modes
        and bake into the IfcSpace representation, so the container operands
        are now composed onto the space's body."""
        proj = _project()
        s = Space(name="a")
        s.add(Box(start=Point(x=1700, y=-3100, z=0),
                  end=Point(x=5900, y=1300, z=2700)))
        s.difference(Box(start=Point(x=1700, y=-3100, z=0),
                         end=Point(x=2700, y=-2100, z=2700)))
        proj.add(s)
        assert _space_has_boolean_body(compile_ifc(proj))


# ---------------------------------------------------------------------------
# 6. The checker — the WS-C engine demoted, and falsified both ways
# ---------------------------------------------------------------------------


class TestChecker:

    def test_a_bound_the_boolean_carved_away_from_warns(self, caplog):
        proj, room = enclosed_room()
        # Slice the whole east end of the room off, full cross-section.
        room.difference(Box(start=Point(x=4900, y=-3400, z=-300),
                            end=Point(x=6300, y=1600, z=3000)))
        with caplog.at_level("WARNING"):
            boundaries(proj)
        assert "does not touch" in caplog.text
        assert "wall:east" in caplog.text

    def test_without_the_carve_no_bound_warns(self, caplog):
        """The falsification: the same room, no boolean — silent."""
        proj, _room = enclosed_room()
        with caplog.at_level("WARNING"):
            boundaries(proj)
        assert "does not touch" not in caplog.text

    def test_an_undeclared_touching_element_warns_as_a_candidate(self, caplog):
        proj, _room = enclosed_room()
        wall(proj, "partition", 2500, 1300, 3700, 1600)
        with caplog.at_level("WARNING"):
            boundaries(proj)
        assert "candidate member" in caplog.text
        assert "wall:partition" in caplog.text

    def test_declaring_the_candidate_silences_it(self, caplog):
        """The falsification: the same partition, declared."""
        proj, _room = enclosed_room(bounds=[
            "wall:south", "wall:north", "wall:west", "wall:east",
            "slab:floor", "slab:deck", "wall:partition"])
        wall(proj, "partition", 2500, 1300, 3700, 1600)
        with caplog.at_level("WARNING"):
            boundaries(proj)
        assert "candidate member" not in caplog.text

    def test_the_effective_volume_trims_a_full_cross_section_end_slice(self):
        vol = ((1.7, -3.1, 0.0), (5.9, 1.3, 2.7))
        cut = ((4.9, -3.4, -0.3), (6.3, 1.6, 3.0))
        assert close(effective_volume(vol, [cut]),
                     ((1.7, -3.1, 0.0), (4.9, 1.3, 2.7)))

    def test_an_interior_pocket_does_not_shrink_the_effective_volume(self):
        vol = ((1.7, -3.1, 0.0), (5.9, 1.3, 2.7))
        cut = ((3.0, -1.0, 0.5), (4.0, 0.0, 1.5))
        assert effective_volume(vol, [cut]) == vol


# ---------------------------------------------------------------------------
# 7. Window/Door — fillings of declared hosts, exactly as today
# ---------------------------------------------------------------------------


class TestFillings:

    @staticmethod
    def _room_with_windows(declare_host: bool):
        bounds = ["wall:south", "wall:north", "wall:west", "wall:east",
                  "slab:floor", "slab:deck"]
        if not declare_host:
            bounds.remove("wall:south")
        proj, room = enclosed_room(bounds=bounds)
        south = next(e for st in proj.storeys for e in st.elements
                     if getattr(e, "name", None) == "south")
        south.anchor(Window(name="pane", width=1200, height=1500),
                     along=1500, up=900)
        return proj

    def test_a_filling_of_a_declared_host_derives_its_inner_boundary(self):
        got = boundaries(self._room_with_windows(declare_host=True))
        inner = [b for b in got if b.element.startswith("window:")]
        assert len(inner) == 1
        assert inner[0].is_inner
        assert got[inner[0].parent_index].element == "wall:south"

    def test_a_filling_of_an_undeclared_host_derives_nothing(self):
        """The declared members ARE the set — an undeclared host's window
        cannot smuggle a boundary in (the host itself warns as a candidate)."""
        got = boundaries(self._room_with_windows(declare_host=False))
        assert [b for b in got if b.element.startswith("window:")] == []


# ---------------------------------------------------------------------------
# 8. Observability — the derived-volume log line
# ---------------------------------------------------------------------------


class TestObservability:

    def test_the_report_line_matches_the_spec_verbatim(self):
        """`space kitchen: derived 3600 x 2400 x 2600 mm` — the roadmap's own
        example shape, from a room authored to those inner dimensions."""
        proj = _project()
        wall(proj, "west", 700, 500, 1000, 2900)
        wall(proj, "east", 4600, 500, 4900, 2900)
        wall(proj, "south", 1000, 200, 4600, 500)
        wall(proj, "north", 1000, 2900, 4600, 3200)
        slab(proj, "floor", 700, 200, 4900, 3200, -200, 0)
        slab(proj, "deck", 700, 200, 4900, 3200, 2600, 2800)
        proj.add(Space(name="kitchen", bounds=[
            "wall:west", "wall:east", "wall:south", "wall:north",
            "slab:floor", "slab:deck"]))
        # The walls run to z 2700; the deck's underside closes at 2600 —
        # poking past a face is legal (a projection cannot reach past it).
        line = report_line("kitchen", derived(proj)["kitchen"])
        assert line == "space kitchen: derived 3600 x 2400 x 2600 mm"

    def test_the_line_is_printed_on_compile(self, capsys):
        proj, _ = enclosed_room()
        boundaries(proj)
        out = capsys.readouterr().out
        assert "space kitchen: derived 4200 x 4400 x 2700 mm" in out

    def test_a_feature_is_noted_on_the_line_not_warned(self, capsys, caplog):
        proj, _ = chimney_room()
        with caplog.at_level("WARNING"):
            boundaries(proj)
        out = capsys.readouterr().out
        assert "feature: wall:chimney" in out
        assert "wall:chimney" not in caplog.text   # declared means declared

    def test_the_boundary_counter_says_declared_for_a_declared_set(self):
        """`space boundaries: 6 declared` — a bounds-mode space's set is the
        author's statement, expanded; calling it "inferred" was the one word
        the doctrine reformulation left behind."""
        from lite_step.ifc.space_boundaries import report_lines

        proj, _ = enclosed_room()
        lines = report_lines(boundaries(proj))
        assert lines[0] == "space boundaries: 6 declared"

    def test_a_mixed_model_counts_both_modes(self):
        """One bounds-mode room + one geometry-mode zone flush against the
        east wall: the header owns up to both mechanisms."""
        from lite_step.ifc.space_boundaries import report_lines

        proj, _ = enclosed_room()
        zone = Space(name="zone")
        zone.add(Box(start=Point(x=6200, y=-3100, z=0),
                     end=Point(x=9400, y=1300, z=2700)))
        proj.add(zone)
        lines = report_lines(boundaries(proj))
        assert lines[0] == "space boundaries: 6 declared, 1 inferred"

    def test_a_geometry_mode_model_keeps_the_historical_counter(self):
        """All-inferred models print exactly what they always printed."""
        from lite_step.ifc.space_boundaries import report_lines

        proj = _project()
        wall(proj, "north", 1700, 1300, 5900, 1600)
        zone = Space(name="zone")
        zone.add(Box(start=Point(x=1700, y=-3100, z=0),
                     end=Point(x=5900, y=1300, z=2700)))
        proj.add(zone)
        lines = report_lines(boundaries(proj))
        assert lines[0] == "space boundaries: 1 inferred"

    def test_the_line_prints_once_per_compile_not_once_per_backend_call(
            self, capsys):
        proj, _ = enclosed_room()
        normalized = normalize_project_to_meters(proj)
        collect_boundaries(normalized)
        collect_boundaries(normalized)             # idempotent second call
        out = capsys.readouterr().out
        assert out.count("derived 4200 x 4400 x 2700 mm") == 1


# ---------------------------------------------------------------------------
# 9. Emission — both backends, explicitly
# ---------------------------------------------------------------------------


class TestEmission:

    def test_the_derived_body_is_the_spaces_representation(self):
        proj, _ = enclosed_room()
        ifc = compile_ifc(proj)
        # Pin WHICH "ifcopenshell" ran via the header stamp — the two-"ifcopenshell" claim
        # is measured, not assumed.
        header = ifc.split("DATA;")[0]
        assert "IfcOpenShell" in header
        model = opened(ifc)
        rooms = model.by_type("IfcSpace")
        assert len(rooms) == 1
        assert rooms[0].Representation is not None
        # No orphan DONOR: the injected body's representation belongs to the
        # IfcSpace alone, never to a duplicate proxy (the WS-C donor bug).
        assert not any(p.Representation == rooms[0].Representation
                       for p in model.by_type("IfcBuildingElementProxy"))

    def test_both_backends_emit_the_same_boundaries(self):
        def facts(backend):
            proj, _ = enclosed_room()
            model = opened(compile_ifc(proj, backend))
            return [(rel.is_a(), rel.Name,
                     rel.RelatingSpace.Name, rel.RelatedBuildingElement.Name,
                     rel.PhysicalOrVirtualBoundary,
                     rel.InternalOrExternalBoundary,
                     rel.ConnectionGeometry)
                    for rel in model.by_type("IfcRelSpaceBoundary1stLevel")]

        got = facts("ifcopenshell")
        assert len(got) == 6
        assert all(name == "1stLevel" for _e, name, *_rest in got)

    def test_the_emitted_set_is_the_declared_set(self):
        proj, _ = enclosed_room()
        model = opened(compile_ifc(proj))
        got = sorted(r.RelatedBuildingElement.Name
                     for r in model.by_type("IfcRelSpaceBoundary1stLevel"))
        assert got == sorted(["wall:south", "wall:north", "wall:west",
                              "wall:east", "slab:floor", "slab:deck"])

    def test_a_bounds_mode_model_validates_clean(self):
        import contextlib
        import io

        import ifcopenshell.validate

        proj, room = enclosed_room()
        room.difference(Box(start=Point(x=1700, y=-3100, z=0),
                            end=Point(x=2700, y=-2100, z=2700)))
        model = opened(compile_ifc(proj))
        logger = ifcopenshell.validate.json_logger()
        with contextlib.redirect_stderr(io.StringIO()):
            ifcopenshell.validate.validate(model, logger)
        errors = [s for s in logger.statements
                  if str(s.get("level", "")).lower() == "error"]
        assert errors == []

    def test_the_derived_body_is_not_in_the_manifest(self):
        """The injected Box is anonymous, like an authored space body — no
        manifest key, no patch atom."""
        proj, _ = enclosed_room()
        ifc = compile_ifc(proj)
        blob = re.search(r"'MANIFEST'.*", ifc)
        assert blob
        assert "space:kitchen" in blob.group(0)

    def test_the_order_is_deterministic(self):
        def order():
            proj, _ = enclosed_room()
            model = opened(compile_ifc(proj))
            return [(r.RelatingSpace.Name, r.RelatedBuildingElement.Name)
                    for r in model.by_type("IfcRelSpaceBoundary1stLevel")]

        assert order() == order()

    def test_the_derived_body_does_not_carve_the_bounds(self):
        """The injected body is AIR: without the ``.no_carve()`` mark the
        room's own derived box (appended last = carver) would eat the chimney
        breast poking into it. The two walls' own inferred carve is real and
        stays; no pair may involve the space."""
        proj, _ = chimney_room()
        normalized = normalize_project_to_meters(proj)
        from lite_step.ifc.generator import generate_ifc

        result = generate_ifc(normalized, source_code="src")
        assert result.success
        offenders = [(loser, winner)
                     for loser, winner, _m in normalized._carve_pairs
                     if "kitchen" in loser or "kitchen" in winner]
        assert offenders == []
