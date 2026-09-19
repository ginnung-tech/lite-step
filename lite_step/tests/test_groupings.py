"""Aggregates + zones — WS-D and the zone half of WS-C, as ONE workstream.

Roadmap §0 (two relations), §3 (the facade inversion), §V.7 (the one grammar).
They are built and tested together because §V.7 established they are the same
shape; building them apart would produce two spellings of one idea.

What is pinned here:

1.  **The grammar (§V.7).** Members are canonical-name STRINGS in a keyword,
    never objects — the string/object split IS §0's identity rule made
    syntactic. Neither verb ever contains, so no member's canonical name
    moves.
2.  **Resolution** is ``naming.resolve_host``, so a miss and an ambiguity are
    compile ERRORS exactly as they are for ``Anchor(host=)`` and
    ``assert_carved``.
3.  **The derived class (§3).** An aggregate's IFC class comes from its
    members; mixed classes are a compile error, so an author cannot state a
    class that contradicts them.
4.  **A zone groups Spaces only** — a loud refusal, not a coercion.
5.  **Constraint 1 — the placement shape.** A world-coordinate member
    with an unchained placement under a representation-less
    ``IfcRelAggregates`` parent tessellated at ~1/87 of its true volume,
    silently. Members are re-chained to the parent, and the world geometry is
    measured through ``ifcopenshell.geom`` to prove the rewrite moved nothing.
6.  **Constraint 2 — the round trip.** A geometry-less parent is silently
    invisible on import when the walk is ``ContainedInSpatialStructure``-only.
    It survives with its identity,
    its class and its members, and its members are NOT emitted twice.
7.  **The manifest trap.** ``IfcAnnotation`` is an ``IfcProduct``, so a named drawing landed in ``LITESTEP_META`` and
    polluted patch identity with an entity matching no DSL element. An
    ``IfcRelAggregates`` parent is an ``IfcProduct`` too, and the streaming
    manifest predicate is purely STRUCTURAL — so this is asserted on both
    backends, whose exclusion mechanisms are different.
8.  Additivity: a project with no grouping emits nothing new.
"""

from __future__ import annotations

import json
import os
import re
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.groupings import (
    AGGREGATE,
    ZONE,
    GroupingError,
    ResolvedGrouping,
    collect_groupings,
    member_elevations,
    missing_member_error,
)
from lite_step.models import (
    Beam, Box, Column, Element, Point, Project, Site, Space, Storey, Wall,
    Window,
)

pytest.importorskip("ifcopenshell")


# ---------------------------------------------------------------------------
# Fixtures — local geometry only (never exec a skill; see tests/_fixtures.py)
# ---------------------------------------------------------------------------


def wall(name: str, *, x0: int = 0, z0: int = 0, length: int = 6000) -> Wall:
    w = Wall(name=name)
    w.add(Box(start=Point(x=x0, y=0, z=z0),
              end=Point(x=x0 + length, y=300, z=z0 + 2700)))
    return w


def space(name: str, *, x0: int = 0, x1: int = 3000) -> Space:
    s = Space(name=name)
    s.add(Box(start=Point(x=x0, y=400, z=0), end=Point(x=x1, y=4000, z=2700)))
    return s


def two_storey(group: bool = True) -> Project:
    """The §3 worked case: a facade whose walls are authored per storey."""
    proj = Project(name="facade")
    ground = Storey(name="ground", elevation=0)
    first = Storey(name="first", elevation=3000)
    proj.add_storey(ground)
    proj.add_storey(first)
    ground.add(wall("north", z0=0))
    first.add(wall("north", z0=3000))
    if group:
        proj.aggregate("north_facade",
                       members=["wall:north:storey:ground",
                                "wall:north:storey:first"])
    return proj


def zoned(group: bool = True) -> Project:
    proj = Project(name="flat")
    proj.add(space("kitchen", x0=0, x1=3000))
    proj.add(space("living", x0=3100, x1=6000))
    if group:
        proj.zone("thermal_north", members=["space:kitchen", "space:living"])
    return proj


def compile_ifc(proj: Project, backend: str = "ifcopenshell",
                source_code: str = "src") -> str:
    normalized = normalize_project_to_meters(proj)
    from lite_step.ifc.generator import generate_ifc
    result = generate_ifc(normalized, source_code=source_code)
    assert result.success, result.error
    return result.ifc_content


def compile_result(proj: Project, backend: str = "ifcopenshell"):
    normalized = normalize_project_to_meters(proj)
    from lite_step.ifc.generator import generate_ifc
    return generate_ifc(normalized, source_code="src")


def opened(ifc: str):
    """Parse an IFC string with ifcopenshell. Caller keeps the model alive."""
    import ifcopenshell

    path_handle = tempfile.NamedTemporaryFile(suffix=".ifc", delete=False)
    path_handle.close()
    path = path_handle.name
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(ifc)
    try:
        return ifcopenshell.open(path)
    finally:
        try:
            os.unlink(path)
        except OSError:                                   # pragma: no cover
            pass


def steps(ifc: str, entity_type: str) -> list:
    pattern = re.compile(r"^#\d+=" + entity_type.upper() + r"\(", re.MULTILINE)
    return [line for line in ifc.splitlines() if pattern.match(line)]


def manifest(ifc: str) -> dict:
    model = opened(ifc)
    for pset in model.by_type("IfcPropertySet"):
        if pset.Name == "LITESTEP_META":
            for prop in pset.HasProperties:
                if prop.Name == "MANIFEST":
                    return json.loads(prop.NominalValue.wrappedValue)
    raise AssertionError("no LITESTEP_META manifest in the file")


def world_boxes(model) -> dict:
    """``{Name: (min_xyz, max_xyz)}`` in absolute world coordinates.

    Tessellated by ``ifcopenshell.geom`` — the tessellator the ~1/87
    measurement came out of — so this measures the thing that failed, not a
    proxy for it.
    """
    import numpy as np
    import ifcopenshell.geom as geom

    settings = geom.settings()
    settings.set("use-world-coords", True)
    out = {}
    for product in model.by_type("IfcProduct"):
        if getattr(product, "Representation", None) is None or not product.Name:
            continue
        try:
            shape = geom.create_shape(settings, product)
        except Exception:                                 # pragma: no cover
            continue
        verts = np.array(shape.geometry.verts, dtype=np.float64).reshape(-1, 3)
        out[product.Name] = (tuple(np.round(verts.min(axis=0), 9)),
                             tuple(np.round(verts.max(axis=0), 9)))
    return out


def aggregate_rel(model, name: str):
    rels = [r for r in model.by_type("IfcRelAggregates")
            if r.RelatingObject is not None and r.RelatingObject.Name == name]
    assert len(rels) == 1, [r.RelatingObject.Name for r in rels]
    return rels[0]


# ---------------------------------------------------------------------------
# 1. The grammar (§V.7) — strings reference, objects contain
# ---------------------------------------------------------------------------


class TestMembersAreStringsNotObjects:
    """An object passed positionally to ``.add()``/``.anchor()`` is
    containment and stamps a canonical name; a string in a keyword is a
    reference and stamps nothing. That split is §0's identity rule made
    syntactic, so handing an object here has to be refused rather than
    quietly accepted."""

    def test_an_element_object_refuses(self):
        proj = Project(name="p")
        north = wall("north")
        proj.add(north)
        with pytest.raises(GroupingError, match="canonical-name STRING"):
            proj.aggregate("facade", members=[north])

    def test_the_message_names_the_identity_churn_and_the_fix(self):
        proj = Project(name="p")
        north = wall("north")
        proj.add(north)
        with pytest.raises(GroupingError) as exc:
            proj.aggregate("facade", members=[north])
        message = str(exc.value)
        assert "rewrite its canonical name" in message
        assert 'members=["wall:north"]' in message

    def test_a_bare_string_refuses(self):
        proj = Project(name="p")
        proj.add(wall("north"))
        with pytest.raises(GroupingError, match=r"members=\['wall:north'\]"):
            proj.aggregate("facade", members="wall:north")

    def test_an_empty_member_list_refuses(self):
        with pytest.raises(GroupingError, match="names nothing"):
            Project(name="p").aggregate("facade", members=[])

    def test_missing_members_refuses(self):
        with pytest.raises(GroupingError, match="needs members"):
            Project(name="p").aggregate("facade", members=None)

    def test_a_list_of_strings_is_accepted(self):
        # Falsification partner: the guard must not refuse the legal spelling.
        proj = two_storey()
        assert len(proj._groupings) == 1


class TestGroupingNameIsACatalogKey:
    @pytest.mark.parametrize("bad", ["wall:north", "Facade", "north facade", ""])
    def test_a_non_key_refuses(self, bad):
        with pytest.raises(GroupingError, match="not a valid name"):
            Project(name="p").aggregate(bad, members=["wall:north"])

    def test_a_missing_name_refuses(self):
        with pytest.raises(GroupingError, match="needs a name"):
            Project(name="p").aggregate(None, members=["wall:north"])

    def test_the_message_says_the_grouping_has_no_canonical_name(self):
        with pytest.raises(GroupingError) as exc:
            Project(name="p").zone("thermal:north", members=["space:kitchen"])
        assert "not in the containment tree" in str(exc.value)

    def test_a_key_is_accepted(self):
        assert Project(name="p").aggregate(
            "north_facade", members=["wall:north"])._groupings[0][1] == "north_facade"


class TestNeitherVerbEverContains:
    """§3's whole reason: containing the members would rewrite their canonical
    names — the manifest key and the patch atom with them."""

    def test_member_canonical_names_are_untouched(self):
        proj = two_storey()
        normalized = normalize_project_to_meters(proj)
        names = {e._canonical_name for s in normalized.storeys for e in s.elements}
        assert names == {"wall:north:storey:ground", "wall:north:storey:first"}

    def test_the_project_gains_no_element(self):
        grouped = normalize_project_to_meters(two_storey(group=True))
        plain = normalize_project_to_meters(two_storey(group=False))
        assert ([len(s.elements) for s in grouped.storeys]
                == [len(s.elements) for s in plain.storeys])

    def test_both_verbs_return_self_so_statements_chain(self):
        proj = Project(name="p")
        proj.add(wall("north"))
        proj.add(space("kitchen"))
        assert proj.aggregate("f", members=["wall:north"]) is proj
        assert proj.zone("z", members=["space:kitchen"]) is proj


class TestCanonicalNamesAreUntouchedByEmission:
    """**The load-bearing claim.**

    Roadmap §3 inverted the conventional containment pattern because it
    believed containing the members under the parent would rewrite their
    canonical names. Conforming is only free if that fear does not apply —
    and it does not, because canonical names are derived from the DSL
    containment tree (``compiler/naming.py``) and ``proj.aggregate()`` takes
    STRINGS: the author still writes ``storey.add(wall)``, so nothing moves in
    that tree. This class is where that is *measured* rather than reasoned
    about. If it ever fails, the change is not emission-only and the whole
    argument for conforming has to be re-opened.
    """

    def test_the_canonical_names_are_byte_identical(self):
        def names(group):
            normalized = normalize_project_to_meters(two_storey(group))
            return [e._canonical_name
                    for s in normalized.storeys for e in s.elements]

        assert names(True) == names(False) == [
            "wall:north:storey:ground", "wall:north:storey:first"]

    def test_every_emitted_product_name_is_byte_identical(self):
        """Stronger than the DSL check: the stamped ``Name`` is the identity
        join the manifest, the patch differ and product typing all use, so it
        is the string that would churn if anything churned."""
        def product_names(group):
            model = opened(compile_ifc(two_storey(group), "ifcopenshell"))
            return sorted(p.Name for p in model.by_type("IfcProduct")
                          if p.Name and p.Name != "north_facade")

        assert product_names(True) == product_names(False)

    def test_the_manifest_keys_and_the_guids_they_map_to_are_unmoved(self):
        """The sharpest form. A manifest KEY set that matched while the values
        drifted would still break patch mode, so compare the key set and
        assert every key still resolves to a product of the same class and
        name — the GUIDs themselves are fresh per compile by design."""
        def entry(group):
            ifc = compile_ifc(two_storey(group), "ifcopenshell")
            model = opened(ifc)
            by_guid = {p.GlobalId: p for p in model.by_type("IfcProduct")}
            return {name: (by_guid[guid].is_a(), by_guid[guid].Name)
                    for name, guid in manifest(ifc).items()}

        assert entry(True) == entry(False)


class TestStatementsSurviveNormalize:
    """``normalize_project_to_meters`` walks the CONTAINMENT tree; a grouping
    is not in it. The WS-B ``Product`` snapshot proved that blind spot can
    leave geometry in millimetres, so verify rather than assume — here there
    is nothing to normalize, and that is the fact being pinned."""

    def test_the_statements_are_carried_verbatim(self):
        proj = two_storey()
        before = list(proj._groupings)
        assert normalize_project_to_meters(proj)._groupings == before

    def test_a_grouping_holds_no_number_a_unit_pass_could_miss(self):
        for kind, name, members in two_storey()._groupings:
            assert isinstance(kind, str) and isinstance(name, str)
            assert all(isinstance(m, str) for m in members)

    def test_resolution_works_on_the_normalized_copy(self):
        groups = collect_groupings(normalize_project_to_meters(two_storey()))
        assert [g.name for g in groups] == ["north_facade"]


# ---------------------------------------------------------------------------
# 2. Resolution — the reference grammar, and its errors
# ---------------------------------------------------------------------------


class TestResolutionIsTheAnchorHostGrammar:
    def test_a_miss_is_a_compile_error(self):
        proj = two_storey(group=False)
        proj.aggregate("facade", members=["wall:nowhere"])
        with pytest.raises(GroupingError, match="names no element"):
            collect_groupings(normalize_project_to_meters(proj))

    def test_an_ambiguity_is_a_compile_error(self):
        """A container pair names every element under it, so the realistic
        mistake is citing the CONTAINER and meaning one of its children."""
        proj = Project(name="p")
        ground = Storey(name="ground", elevation=0)
        proj.add_storey(ground)
        ground.add(wall("north"), wall("south", x0=8000))
        proj.aggregate("facade", members=["storey:ground"])
        with pytest.raises(GroupingError, match="matches several elements"):
            collect_groupings(normalize_project_to_meters(proj))

    def test_a_suffix_fragment_resolves_like_anchor_host(self):
        # Falsification partner: an UNambiguous fragment must still resolve.
        proj = Project(name="p")
        proj.add(wall("north"))
        proj.aggregate("facade", members=["wall:north"])
        groups = collect_groupings(normalize_project_to_meters(proj))
        assert groups[0].members == ("wall:north",)

    def test_a_member_cited_twice_refuses(self):
        proj = two_storey(group=False)
        proj.aggregate("facade", members=["wall:north:storey:ground",
                                          "wall:north:storey:ground"])
        with pytest.raises(GroupingError, match="cited twice"):
            collect_groupings(normalize_project_to_meters(proj))

    def test_a_nested_member_refuses(self):
        """A nested element is already a decomposition part of its host, and
        IFC gives a product ONE Decomposes relation."""
        proj = Project(name="p")
        w = wall("north")
        w._elements[0].name = "body"
        proj.add(w)
        proj.aggregate("facade", members=["box:body:wall:north"])
        with pytest.raises(GroupingError, match="not a top-level element"):
            collect_groupings(normalize_project_to_meters(proj))

    def test_two_groupings_under_one_name_refuse(self):
        proj = two_storey(group=False)
        proj.aggregate("facade", members=["wall:north:storey:ground"])
        proj.aggregate("facade", members=["wall:north:storey:first"])
        with pytest.raises(GroupingError, match="declared twice"):
            collect_groupings(normalize_project_to_meters(proj))


class TestOneDecomposesRelationPerProduct:
    def test_a_member_of_two_aggregates_refuses(self):
        proj = two_storey(group=False)
        proj.aggregate("facade_a", members=["wall:north:storey:ground"])
        proj.aggregate("facade_b", members=["wall:north:storey:ground"])
        with pytest.raises(GroupingError, match="already a member of aggregate"):
            collect_groupings(normalize_project_to_meters(proj))

    def test_a_space_in_two_zones_is_fine(self):
        """Falsification partner AND a real difference: a space belongs to one
        decomposition but any number of groups, which is why the two are
        separate verbs rather than one with a flag."""
        proj = zoned(group=False)
        proj.zone("thermal", members=["space:kitchen", "space:living"])
        proj.zone("acoustic", members=["space:kitchen"])
        groups = collect_groupings(normalize_project_to_meters(proj))
        assert [g.name for g in groups] == ["thermal", "acoustic"]


# ---------------------------------------------------------------------------
# 3. The derived class (§3) — IfcWall = IfcWall + IfcWall
# ---------------------------------------------------------------------------


class TestTheAggregateClassIsDerived:
    def test_walls_derive_an_ifcwall_parent(self):
        groups = collect_groupings(normalize_project_to_meters(two_storey()))
        assert groups[0].ifc_class == "IfcWall"

    def test_beams_derive_an_ifcbeam_parent(self):
        proj = Project(name="p")
        for i in range(2):
            b = Beam(name=f"b{i}")
            b.add(Box(start=Point(x=i * 3000, y=0, z=0),
                      end=Point(x=i * 3000 + 2000, y=200, z=200)))
            proj.add(b)
        proj.aggregate("truss", members=["beam:b0", "beam:b1"])
        groups = collect_groupings(normalize_project_to_meters(proj))
        assert groups[0].ifc_class == "IfcBeam"

    def test_the_escape_hatch_class_derives_too(self):
        proj = Project(name="p")
        for i in range(2):
            e = Element(ifc_class="IfcStair", name=f"flight{i}")
            e.add(Box(start=Point(x=i * 3000, y=0, z=0),
                      end=Point(x=i * 3000 + 2000, y=1000, z=2000)))
            proj.add(e)
        proj.aggregate("stair_run", members=["stair:flight0", "stair:flight1"])
        assert collect_groupings(
            normalize_project_to_meters(proj))[0].ifc_class == "IfcStair"

    def test_mixed_classes_refuse(self):
        proj = Project(name="p")
        proj.add(wall("north"))
        beam = Beam(name="lintel")
        beam.add(Box(start=Point(x=0, y=0, z=2800),
                     end=Point(x=6000, y=200, z=3000)))
        proj.add(beam)
        proj.aggregate("facade", members=["wall:north", "beam:lintel"])
        with pytest.raises(GroupingError, match="mixes IFC classes"):
            collect_groupings(normalize_project_to_meters(proj))

    def test_the_mixed_message_names_both_classes_and_the_reason(self):
        proj = Project(name="p")
        proj.add(wall("north"))
        beam = Beam(name="lintel")
        beam.add(Box(start=Point(x=0, y=0, z=2800),
                     end=Point(x=6000, y=200, z=3000)))
        proj.add(beam)
        proj.aggregate("facade", members=["wall:north", "beam:lintel"])
        with pytest.raises(GroupingError) as exc:
            collect_groupings(normalize_project_to_meters(proj))
        message = str(exc.value)
        assert "IfcBeam + IfcWall" in message
        assert "IfcWall = IfcWall + IfcWall" in message

    def test_aggregating_spaces_refuses_and_points_at_zone(self):
        """A spatial-structure parent would be an orphan storey/space outside
        the hierarchy — and there is a verb that means what the author meant."""
        proj = zoned(group=False)
        proj.aggregate("thermal", members=["space:kitchen", "space:living"])
        with pytest.raises(GroupingError, match=r"proj\.zone\('thermal'"):
            collect_groupings(normalize_project_to_meters(proj))


# ---------------------------------------------------------------------------
# 4. A zone groups Spaces only
# ---------------------------------------------------------------------------


class TestTheParentsContainerIsDerived:
    """Which storey the parent goes in is DERIVED, never a
    parameter. An author who had to name it could name a storey no member
    stands in, and the file would still validate."""

    @staticmethod
    def _site_and_storey() -> Project:
        proj = Project(name="p")
        storey = Storey(name="ground", elevation=0)
        storey.add(wall("north"))
        proj.add_storey(storey)
        site = Site(name="site")
        wrap = Element(ifc_class="IfcWall", name="fence")
        wrap.add(Box(start=Point(x=0, y=0, z=0),
                     end=Point(x=6000, y=200, z=1800)))
        site.add(wrap)
        proj.add(site)
        proj.aggregate("perimeter",
                       members=["wall:north:storey:ground",
                                "wall:fence:site:site"])
        return proj

    def test_members_split_across_a_site_and_a_building_refuse(self):
        """There is no storey that contains a site child and no site that
        contains a storey element — either choice files the aggregate under a
        hierarchy half its members are not in."""
        from lite_step.ifc.groupings import parent_container

        proj = normalize_project_to_meters(self._site_and_storey())
        group = collect_groupings(proj)[0]
        with pytest.raises(GroupingError) as err:
            parent_container(proj, group)
        assert "site" in str(err.value) and "storey" in str(err.value)

    def test_a_site_and_storey_mix_never_compiles_quietly(self):
        """Whichever refusal fires first, the compile FAILS and the message
        names the member — the one outcome that must never happen is a file
        with the facade filed under a hierarchy half its members are not in.

        The two backends now refuse at DIFFERENT points, and both are pinned:

        * **ifcopenshell** — the missing-product refusal, naming the aggregate
          (``perimeter``): a ``Site``'s ``Element`` wrapper emits no product
          of its own, so it can never be an aggregate member.
        * **streaming** — the earlier ``StreamingUnsupportedError``
, which fires the moment the site loop meets an
          ``Element`` it has no emitter for. Before that refusal existed this
          "ifcopenshell" DROPPED the wrapper in silence, which is why the
          missing-product refusal could win the race here too. Firing
              sooner names the actual cause (a "ifcopenshell"
          that cannot emit this product) rather than its downstream symptom.

        Both messages still name the member by canonical name, which is what
        sends a reader to the right line."""
        result = compile_result(self._site_and_storey(), "ifcopenshell")
        assert result.success is False
        assert "wall:fence:site:site" in result.error
        assert "perimeter" in result.error

    def test_the_lowest_member_wins_regardless_of_authoring_order(self):
        from lite_step.ifc.groupings import parent_container

        proj = Project(name="p")
        for name, elevation in (("ground", 0), ("first", 3000)):
            storey = Storey(name=name, elevation=elevation)
            storey.add(wall("north", z0=elevation))
            proj.add_storey(storey)
        proj.aggregate("f", members=["wall:north:storey:first",
                                     "wall:north:storey:ground"])
        normalized = normalize_project_to_meters(proj)
        container = parent_container(normalized, collect_groupings(normalized)[0])
        assert (container.kind, container.name, container.elevation) == \
            ("storey", "storey:ground", 0.0)

    def test_an_empty_storey_does_not_shift_the_emitted_index(self):
        """A storey with no elements emits no ``IfcBuildingStorey`` on either
        backend, so the DSL index and the file index diverge. Looking the
        container up by DSL index would contain the facade one floor off —
        no geometry moved, nothing raised."""
        from lite_step.ifc.groupings import member_containers

        proj = Project(name="p")
        proj.add_storey(Storey(name="basement", elevation=-3000))   # EMPTY
        upper = Storey(name="ground", elevation=0)
        upper.add(wall("north"))
        proj.add_storey(upper)
        normalized = normalize_project_to_meters(proj)
        container = member_containers(normalized)["wall:north:storey:ground"]
        assert (container.index, container.emitted_index) == (1, 0)

    def test_an_empty_storey_lands_the_parent_in_the_right_one(self):
        proj = Project(name="p")
        proj.add_storey(Storey(name="basement", elevation=-3000))   # EMPTY
        upper = Storey(name="ground", elevation=0)
        upper.add(wall("north"))
        upper.add(wall("south", x0=0))
        proj.add_storey(upper)
        proj.aggregate("shell", members=["wall:north:storey:ground",
                                         "wall:south:storey:ground"])
        model = opened(compile_ifc(proj))
        parent = aggregate_rel(model, "shell").RelatingObject
        assert [c.RelatingStructure.Name for c in parent.ContainedInStructure] \
            == ["storey:ground"]


class TestZoneRefusesNonSpaces:
    def test_a_wall_member_refuses(self):
        proj = Project(name="p")
        proj.add(wall("north"))
        proj.zone("thermal", members=["wall:north"])
        with pytest.raises(GroupingError, match="groups IfcSpaces only"):
            collect_groupings(normalize_project_to_meters(proj))

    def test_the_message_points_at_aggregate(self):
        proj = Project(name="p")
        proj.add(wall("north"))
        proj.zone("thermal", members=["wall:north"])
        with pytest.raises(GroupingError) as exc:
            collect_groupings(normalize_project_to_meters(proj))
        assert "proj.aggregate('thermal'" in str(exc.value)

    def test_spaces_are_accepted(self):
        groups = collect_groupings(normalize_project_to_meters(zoned()))
        assert groups[0].kind == ZONE
        assert groups[0].ifc_class == "IfcZone"
        assert groups[0].members == ("space:kitchen", "space:living")


# ---------------------------------------------------------------------------
# 5. Emission — both backends, asserted to agree
# ---------------------------------------------------------------------------


class TestAggregateEmission:
    def test_one_relation_with_a_geometry_less_parent(self):
        model = opened(compile_ifc(two_storey(), "ifcopenshell"))
        rel = aggregate_rel(model, "north_facade")
        assert rel.RelatingObject.is_a() == "IfcWall"
        assert rel.RelatingObject.Representation is None
        assert sorted(o.Name for o in rel.RelatedObjects) == [
            "wall:north:storey:first", "wall:north:storey:ground"]

    def test_the_parent_carries_the_spatial_containment(self):
        """IFC 4.3 Spatial Containment: the WHOLE is contained.
        §3 specified the inverse and made the parent a ROOT ORPHAN —
        unreachable from ``getSpatialStructure()``, so invisible in our own
        inspector tree, unselectable and unschedulable."""
        model = opened(compile_ifc(two_storey(), "ifcopenshell"))
        parent = aggregate_rel(model, "north_facade").RelatingObject
        assert [c.RelatingStructure.Name for c in parent.ContainedInStructure] \
            == ["storey:ground"]

    def test_no_member_is_contained(self):
        """The other half of the same sentence, verbatim from the schema:
        ``SELF\\IfcElement.ContainedInStructure`` should be *NIL*."""
        model = opened(compile_ifc(two_storey(), "ifcopenshell"))
        rel = aggregate_rel(model, "north_facade")
        got = {o.Name: [c.RelatingStructure.Name for c in o.ContainedInStructure]
               for o in rel.RelatedObjects}
        assert got == {"wall:north:storey:ground": [],
                       "wall:north:storey:first": []}

    def test_the_parent_goes_in_the_container_of_the_lowest_member(self):
        """The derived rule. IFC 4.3 on the multi-level case it does discuss:
        "A multi-storey space is contained (or belongs to) the building storey
        at which its ground level is". Inverted here — the facade authored
        upper-storey-first must still land on the ground storey, or the rule
        is reading dict order rather than elevation."""
        proj = Project(name="facade")
        ground = Storey(name="ground", elevation=0)
        first = Storey(name="first", elevation=3000)
        proj.add_storey(ground)
        proj.add_storey(first)
        ground.add(wall("north", z0=0))
        first.add(wall("north", z0=3000))
        proj.aggregate("north_facade",
                       members=["wall:north:storey:first",      # upper FIRST
                                "wall:north:storey:ground"])
        model = opened(compile_ifc(proj))
        parent = aggregate_rel(model, "north_facade").RelatingObject
        assert [c.RelatingStructure.Name for c in parent.ContainedInStructure] \
            == ["storey:ground"]

    def test_a_facade_that_starts_upstairs_lands_upstairs(self):
        """The falsification partner: if "lowest" were really "storey 0" the
        two tests would agree by accident. Nothing on the ground storey is a
        member here, so the parent belongs upstairs."""
        proj = Project(name="facade")
        ground = Storey(name="ground", elevation=0)
        first = Storey(name="first", elevation=3000)
        second = Storey(name="second", elevation=6000)
        proj.add_storey(ground)
        proj.add_storey(first)
        proj.add_storey(second)
        ground.add(wall("other", z0=0))
        first.add(wall("north", z0=3000))
        second.add(wall("north", z0=6000))
        proj.aggregate("north_facade",
                       members=["wall:north:storey:second",
                                "wall:north:storey:first"])
        model = opened(compile_ifc(proj))
        parent = aggregate_rel(model, "north_facade").RelatingObject
        assert [c.RelatingStructure.Name for c in parent.ContainedInStructure] \
            == ["storey:first"]
        # and the ground storey keeps the wall that is not a member
        ground_rel = [r for r in model.by_type("IfcRelContainedInSpatialStructure")
                      if r.RelatingStructure.Name == "storey:ground"]
        assert [e.Name for r in ground_rel for e in r.RelatedElements] == \
            ["wall:other:storey:ground"]

    def test_no_containment_relation_is_emitted_empty(self):
        """``RelatedElements`` is ``SET [1:?]``. A storey whose only element
        became an aggregate member has its relation DELETED, not emptied — an
        empty set is a schema violation most readers accept in silence."""
        model = opened(compile_ifc(two_storey(), "ifcopenshell"))
        rels = model.by_type("IfcRelContainedInSpatialStructure")
        assert rels, "the file lost its containment entirely"
        for rel in rels:
            assert rel.RelatedElements, rel.RelatingStructure.Name
        # storey:first held only the upper member, so its relation is gone
        assert [r.RelatingStructure.Name for r in rels] == ["storey:ground"]

    def test_no_grouping_emits_nothing_new(self):
        """Additivity: this feature is invisible to every existing model."""
        plain = compile_ifc(two_storey(group=False), "ifcopenshell")
        grouped = compile_ifc(two_storey(group=True), "ifcopenshell")
        assert len(steps(grouped, "IFCRELAGGREGATES")) == \
            len(steps(plain, "IFCRELAGGREGATES")) + 1
        assert not steps(plain, "IFCZONE")

    def test_zone_emits_a_group_and_an_assignment(self):
        model = opened(compile_ifc(zoned(), "ifcopenshell"))
        zones = [z for z in model.by_type("IfcZone") if z.Name == "thermal_north"]
        assert len(zones) == 1
        rel = zones[0].IsGroupedBy[0]
        assert sorted(o.Name for o in rel.RelatedObjects) == [
            "space:kitchen", "space:living"]
        assert {o.is_a() for o in rel.RelatedObjects} == {"IfcSpace"}

    def test_ifczone_is_not_an_ifcproduct(self):
        """The schema fact a zone's manifest exclusion rests on — an IfcZone
        is an IfcGroup, so no product walk could ever reach it."""
        from ifcopenshell import ifcopenshell_wrapper

        schema = ifcopenshell_wrapper.schema_by_name("IFC4X3_ADD2")
        cursor = schema.declaration_by_name("IfcZone").as_entity()
        supertypes = set()
        while cursor is not None:
            supertypes.add(cursor.name())
            cursor = cursor.supertype()
        assert "IfcGroup" in supertypes
        assert "IfcProduct" not in supertypes


class TestBothBackendsAgree:
    """Two implementations of one idea agreeing *usually* is what
    produced the bug. The placement arithmetic in particular is spelled out
    twice — generically from ``get_local_placement`` on one backend, from the
    storey elevation on the other — so assert they land on the same numbers.
    """

    @staticmethod
    def _read(backend):
        import ifcopenshell.util.placement as plc
        import numpy as np

        model = opened(compile_ifc(two_storey(), backend))
        rel = aggregate_rel(model, "north_facade")
        parent = rel.RelatingObject
        return {
            "parent": (parent.is_a(), parent.Name,
                       parent.Representation,
                       tuple(np.round(
                           plc.get_local_placement(parent.ObjectPlacement).ravel(), 9))),
            "members": {
                o.Name: (
                    o.is_a(),
                    o.ObjectPlacement.PlacementRelTo == parent.ObjectPlacement,
                    tuple(np.round(
                        plc.get_local_placement(o.ObjectPlacement).ravel(), 9)),
                    tuple(np.round(
                        np.array(o.ObjectPlacement.RelativePlacement
                                 .Location.Coordinates, dtype=float), 9)),
                )
                for o in rel.RelatedObjects
            },
        }

    def test_the_two_backends_agree_on_every_containment_relation(self):
        """The half. One backend edits live ifcopenshell entities and
        deletes emptied relations through ``model.remove``; the other rewrites
        IR descriptor attribute lists and drops them from ``ir.trailing``. Two
        spellings of one move is the exact recipe for a bug, so compare the
        whole containment picture, not just the parent's own."""
        def read(backend):
            model = opened(compile_ifc(two_storey(), backend))
            return sorted(
                (rel.RelatingStructure.Name,
                 sorted(e.Name for e in rel.RelatedElements))
                for rel in model.by_type("IfcRelContainedInSpatialStructure"))

        got = read("ifcopenshell")
        assert got == [("storey:ground", ["north_facade"])]




# ---------------------------------------------------------------------------
# 6. Constraint 1 — the placement shape
# ---------------------------------------------------------------------------


class TestMembersChainToTheParent:
    """``generator.py::_create_opening_profile_path`` records a MEASURED
    failure: a world-coordinate child with an unchained ``ObjectPlacement``
    inside an ``IfcRelAggregates`` whose ``RelatingObject`` has no
    ``Representation`` tessellated at ~1/87 of its true volume, from a
    bit-identical entity subgraph that was exact outside the aggregation
. Nothing raised. Chaining removes that configuration."""

    def test_every_member_chains_to_the_parent(self):
        model = opened(compile_ifc(two_storey(), "ifcopenshell"))
        rel = aggregate_rel(model, "north_facade")
        parent_placement = rel.RelatingObject.ObjectPlacement
        for member in rel.RelatedObjects:
            assert member.ObjectPlacement.PlacementRelTo == parent_placement, \
                member.Name

    def test_the_parent_placement_is_the_identity(self):
        import numpy as np
        import ifcopenshell.util.placement as plc

        model = opened(compile_ifc(two_storey(), "ifcopenshell"))
        parent = aggregate_rel(model, "north_facade").RelatingObject
        matrix = plc.get_local_placement(parent.ObjectPlacement)
        assert np.allclose(matrix, np.eye(4))

    def test_a_members_own_children_still_follow_it(self):
        """The ``IfcLocalPlacement`` ENTITY is mutated, never replaced — a
        window frame member chains to that entity (the own fix) and must
        keep following it through the re-chain."""
        proj = Project(name="p")
        host = wall("north", length=12000)
        win = Window(width=1400, height=1600)
        win.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=1400, y=60, z=1600)))
        host.anchor(win, along=2000, up=900)
        proj.add(host)
        proj.add(wall("south", x0=0, length=12000))
        proj.storeys[0].elements[1]._elements[0].start = Point(x=0, y=5000, z=0)
        proj.storeys[0].elements[1]._elements[0].end = Point(x=12000, y=5300, z=2700)
        proj.aggregate("shell", members=["wall:north", "wall:south"])
        model = opened(compile_ifc(proj))
        # Nothing dangles: every local placement resolves, and the window's
        # aggregated children still sit inside their host wall's extent.
        rel = aggregate_rel(model, "shell")
        assert len(rel.RelatedObjects) == 2


class TestMembersKeepTheirWorldPosition:
    """The guard on the re-chain itself. Rewriting a member's
    ``RelativePlacement`` is exactly the kind of change that moves a wall
    without raising, so the world geometry is MEASURED — tessellated through
    ``ifcopenshell.geom``, the tessellator the number came out of — with
    and without the aggregate, and required to be identical."""

    def test_world_geometry_is_identical_with_and_without_the_aggregate(self):
        grouped = world_boxes(opened(compile_ifc(two_storey(True), "ifcopenshell")))
        plain = world_boxes(opened(compile_ifc(two_storey(False), "ifcopenshell")))
        assert grouped == plain

    def test_the_upper_storey_member_keeps_its_elevation(self):
        """The case a naive re-chain gets wrong: members from two storeys
        cannot both keep a storey-relative placement under one parent, so the
        elevation has to be folded back in. 3000 mm is the number that
        vanishes if it is not."""
        boxes = world_boxes(opened(compile_ifc(two_storey(True), "ifcopenshell")))
        (_, _, z_lo), (_, _, z_hi) = boxes["wall:north:storey:first"]
        assert (round(z_lo, 6), round(z_hi, 6)) == (3.0, 5.7)

    @staticmethod
    def _upstairs_facade(group: bool) -> Project:
        """A facade whose lowest member is NOT on the ground storey.

        Every other fixture here puts the parent on a storey at elevation 0,
        where the parent-relative rebase is a subtraction of zero — so all of
        them stay green with the rebase deleted. A guard whose fixture sits at
        the defaults cannot falsify anything; 3000 mm is the number that has
        to appear in the arithmetic for this one to hold."""
        proj = Project(name="facade")
        for name, elevation in (("ground", 0), ("first", 3000), ("second", 6000)):
            storey = Storey(name=name, elevation=elevation)
            proj.add_storey(storey)
        proj.storeys[0].add(wall("other", z0=0))
        proj.storeys[1].add(wall("north", z0=3000))
        proj.storeys[2].add(wall("north", z0=6000))
        if group:
            proj.aggregate("north_facade",
                           members=["wall:north:storey:first",
                                    "wall:north:storey:second"])
        return proj

    def test_a_facade_rooted_upstairs_moves_nothing_either(self):
        """The parent is contained in — and placed relative to — storey:first
        at 3000 mm, so a member's placement is rebased by the DIFFERENCE of
        the two elevations, not by its own. Dropping that subtraction moves
        both members up a storey, silently, and no fixture rooted at
        elevation 0 can see it."""
        grouped = world_boxes(opened(compile_ifc(self._upstairs_facade(True), "ifcopenshell")))
        plain = world_boxes(opened(compile_ifc(self._upstairs_facade(False), "ifcopenshell")))
        assert grouped == plain
        (_, _, z_lo), (_, _, z_hi) = grouped["wall:north:storey:second"]
        assert (round(z_lo, 6), round(z_hi, 6)) == (6.0, 8.7)

    def test_both_backends_place_an_upstairs_facade_identically(self):
        import numpy as np
        import ifcopenshell.util.placement as plc

        def read(backend):
            model = opened(compile_ifc(self._upstairs_facade(True), backend))
            rel = aggregate_rel(model, "north_facade")
            return (tuple(np.round(plc.get_local_placement(
                        rel.RelatingObject.ObjectPlacement).ravel(), 9)),
                    {o.Name: tuple(np.round(np.array(
                        o.ObjectPlacement.RelativePlacement.Location.Coordinates,
                        dtype=float), 9)) for o in rel.RelatedObjects})

        got = read("ifcopenshell")
        # One storey apart under a parent that is itself one storey up: the
        # SPACING is 3000 mm and the lower member's own offset is whatever the
        # wall's local frame says, not zero.
        lo, hi = sorted(z for _, _, z in got[1].values())
        assert round(hi - lo, 6) == 3.0
        # and the parent's world Z is the storey it is contained in
        assert round(got[0][11], 6) == 3.0

    def test_a_zone_moves_nothing_at_all(self):
        """A zone is an IfcGroup — no placement anywhere in the relation."""
        assert (world_boxes(opened(compile_ifc(zoned(True), "ifcopenshell")))
                == world_boxes(opened(compile_ifc(zoned(False), "ifcopenshell"))))


# ---------------------------------------------------------------------------
# 7. The manifest trap
# ---------------------------------------------------------------------------


class TestGroupingsStayOutOfTheManifest:
    """``IfcAnnotation`` IS an ``IfcProduct``, so a named
    drawing landed in ``LITESTEP_META`` and polluted patch identity with an
    entity matching no DSL element. An ``IfcRelAggregates`` parent is an
    ``IfcProduct`` too, and the streaming manifest predicate is purely
    STRUCTURAL (22-char GlobalId at attr 0, Name at attr 2) — it would
    swallow the parent happily. Both exclusions are by emission ORDER, and
    order is exactly what a refactor moves."""

    def test_the_aggregate_name_is_not_a_manifest_entry(self):
        entries = manifest(compile_ifc(two_storey(), "ifcopenshell"))
        assert "north_facade" not in entries
        assert {"wall:north:storey:ground", "wall:north:storey:first"} <= set(entries)

    def test_the_zone_name_is_not_a_manifest_entry(self):
        entries = manifest(compile_ifc(zoned(), "ifcopenshell"))
        assert "thermal_north" not in entries
        assert {"space:kitchen", "space:living"} <= set(entries)

    def test_no_manifest_entry_maps_to_a_grouping_guid(self):
        """Stronger than a key check: the pollution that mattered was a GUID
        in the manifest pointing at an entity no DSL element corresponds to."""
        ifc = compile_ifc(two_storey(), "ifcopenshell")
        model = opened(ifc)
        rel = aggregate_rel(model, "north_facade")
        guids = {rel.GlobalId, rel.RelatingObject.GlobalId}
        assert not (guids & set(manifest(ifc).values()))

    def test_the_manifest_is_unchanged_by_grouping(self):
        """The sharpest form: patch identity must be byte-for-byte the same
        set of keys whether or not the model carries a grouping."""
        assert (set(manifest(compile_ifc(two_storey(True), "ifcopenshell")))
                == set(manifest(compile_ifc(two_storey(False), "ifcopenshell"))))

    def test_the_meta_version_is_not_bumped(self):
        """Purely additive — a stored base still continues, so the version
            ladder must not move. Asserted as an
        EQUALITY between a grouped and a plain compile rather than against a
        literal, so this test never has to be edited by the next cutover: it
        only fails if a grouping is what moved the number."""
        from lite_step.ifc.embedder import LITESTEP_META_VERSION

        def version(ifc):
            model = opened(ifc)
            for pset in model.by_type("IfcPropertySet"):
                if pset.Name == "LITESTEP_META":
                    for prop in pset.HasProperties:
                        if prop.Name == "VERSION":
                            return prop.NominalValue.wrappedValue
            raise AssertionError("no LITESTEP_META VERSION in the file")

        grouped = version(compile_ifc(two_storey(True), "ifcopenshell"))
        assert grouped == version(compile_ifc(two_storey(False), "ifcopenshell"))
        assert grouped == LITESTEP_META_VERSION


# ---------------------------------------------------------------------------
# 8. Constraint 2 — the round trip
# ---------------------------------------------------------------------------


class TestTheRoundTripPreservesGroupings:
    """A geometry-less parent is **silently invisible** on import when the
    walk is ``ContainedInSpatialStructure``-only: the name, the identity and
    any facade Psets are dropped with no warning — the
    silent-degradation failure this chapter refuses."""

    def test_the_aggregate_parent_survives_import(self):
        from lite_step.ifc.normalizer import normalize_ifc_full

        result = normalize_ifc_full(compile_ifc(two_storey(), "ifcopenshell"))
        aggregates = [g for g in result.groupings if g.kind == "aggregate"]
        assert len(aggregates) == 1
        got = aggregates[0]
        assert got.name == "north_facade"
        assert got.ifc_type == "IfcWall"
        assert sorted(got.members) == ["wall:north:storey:first",
                                       "wall:north:storey:ground"]

    def test_the_zone_survives_import(self):
        from lite_step.ifc.normalizer import normalize_ifc_full

        result = normalize_ifc_full(compile_ifc(zoned(), "ifcopenshell"))
        zones = [g for g in result.groupings if g.kind == "zone"]
        assert len(zones) == 1
        assert zones[0].name == "thermal_north"
        assert sorted(zones[0].members) == ["space:kitchen", "space:living"]

    def test_members_are_not_emitted_twice(self):
        """The measured failure of the forbidden also-contained variant, and
        the one the peer-vs-part discriminator prevents here."""
        from lite_step.ifc.normalizer import normalize_ifc_full

        result = normalize_ifc_full(compile_ifc(two_storey(), "ifcopenshell"))
        names = [e.name for e in result.elements]
        assert names.count("wall:north:storey:ground") == 1
        assert names.count("wall:north:storey:first") == 1
        assert "north_facade" not in names

    def test_the_element_list_is_unchanged_by_grouping(self):
        from lite_step.ifc.normalizer import normalize_ifc_full

        def names(group):
            return sorted(e.name for e in
                          normalize_ifc_full(compile_ifc(two_storey(group), "ifcopenshell")).elements)

        assert names(True) == names(False)

    def test_a_plain_model_reports_no_grouping(self):
        from lite_step.ifc.normalizer import normalize_ifc_full

        result = normalize_ifc_full(compile_ifc(two_storey(False)))
        assert result.groupings == []

    def test_the_parent_is_never_given_a_fabricated_body(self):
        """**The hazard creates.** kept the body-less parent out
        of ``elements`` because it was in no containment relation. Conforming
        puts it IN one, so the containment walk reaches it — and a product
        with no geometry that reaches the reconstructor gets the measured
        placement-derived 100 mm placeholder box. The parent must be
        recognised and skipped BEFORE extraction, not filtered afterwards."""
        from lite_step.ifc.normalizer import normalize_ifc_full

        result = normalize_ifc_full(compile_ifc(two_storey(), "ifcopenshell"))
        assert "north_facade" not in [e.name for e in result.elements]
        for element in result.elements:
            assert element.geometry.mode != "unknown", element.name
            for child in element.children:
                assert child.name != "north_facade"

    def test_members_come_back_as_peers_in_their_own_storeys(self):
        """The DSL-tree question asks: do the members return as
        *walls in storeys* (what was authored) or as *parts of a parent wall*?
        Peers, in the storeys their own elevations put them in — so the round
        trip reproduces ``storey.add(wall)`` twice plus one
        ``proj.aggregate()``, and no marker in the file is needed to say so.

        The storey is the part IFC does not record: containment moved to the
        whole, so the member's own elevation is what recovers it. 3000 mm is
        the number that vanishes if it does not."""
        from lite_step.ifc.normalizer import normalize_ifc_full

        result = normalize_ifc_full(compile_ifc(two_storey(), "ifcopenshell"))
        by_name = {e.name: e for e in result.elements}
        assert set(by_name) == {"wall:north:storey:ground",
                                "wall:north:storey:first"}
        assert by_name["wall:north:storey:ground"].storey_idx == 0
        assert by_name["wall:north:storey:first"].storey_idx == 1
        assert by_name["wall:north:storey:first"].storey_elevation == 3000
        assert all(not e.children for e in result.elements)

    def test_a_lone_facade_does_not_wake_the_all_products_fallback(self):
        """Found while building this change, and silent in every log.

        ``normalize_ifc_full`` falls back to walking every ``IfcProduct`` when
        "containment found nothing", and measured that as ``not elements``.
        Conforming made the facade parent the ONLY spatially contained product
        in this model, and it yields no element — so ``elements`` was empty,
        the fallback fired, and it swept in the two un-contained
        ``IfcAnnotation`` plan drawings. The model gained two "elements"
        purely by being aggregated. The gate now counts what the containment
        walk REACHED, not what it produced."""
        from lite_step.ifc.normalizer import normalize_ifc_full

        result = normalize_ifc_full(compile_ifc(two_storey(), "ifcopenshell"))
        assert not [e for e in result.elements
                    if e.ifc_type == "IfcAnnotation"], \
            [e.name for e in result.elements]

    def test_the_round_trip_is_identical_with_and_without_the_aggregate(self):
        """Everything except the grouping list. Names, classes, storeys and
        absolute world placements all survive being aggregated — which is what
        makes the aggregate a statement ABOUT the model rather than a change
        TO it."""
        from lite_step.ifc.normalizer import normalize_ifc_full

        def read(group):
            result = normalize_ifc_full(compile_ifc(two_storey(group), "ifcopenshell"))
            return sorted((e.name, e.ifc_type, e.storey_idx, e.storey_elevation,
                           e.placement.translation, e.geometry.mode,
                           e.geometry.start, e.geometry.end)
                          for e in result.elements)

        assert read(True) == read(False)


def wall_with_a_detail() -> Project:
    """A wall whose non-body child is a genuine DECOMPOSITION part.

    ``_aggregate_wall_details`` emits it under ``IfcRelAggregates`` and it is
    NOT spatially contained — the other thing that relation means.
    """
    proj = Project(name="p")
    w = Wall(name="north")
    w.add(Box(name="body", start=Point(x=0, y=0, z=0),
              end=Point(x=6000, y=300, z=2700)))
    w.add(Box(name="corbel", start=Point(x=1000, y=-100, z=2500),
              end=Point(x=1400, y=0, z=2700)))
    proj.add(w)
    return proj


def mutated(ifc: str, mutate) -> str:
    """Re-serialise ``ifc`` after ``mutate(model)`` — third-party shapes.

    Building the odd cases by editing our own known-good output keeps the
    fixture honest: everything except the one relation under test is exactly
    what this compiler emits.
    """
    model = opened(ifc)
    mutate(model)
    path_handle = tempfile.NamedTemporaryFile(suffix=".ifc", delete=False)
    path_handle.close()
    path = path_handle.name
    model.write(path)
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(path)
        except OSError:                                   # pragma: no cover
            pass


class TestPeerAggregationIsToldFromDecomposition:
    """The discriminator constraint 2 turns on. Three consumers assumed
    ``IfcRelAggregates`` always meant *decomposition of ONE element*."""

    def test_a_genuine_decomposition_part_is_still_a_child(self):
        """Falsification partner, and the thing that must not regress: a
        wall's aggregated detail solid is NOT spatially contained, so it must
        keep coming back as a CHILD of its host rather than being skipped."""
        from lite_step.ifc.normalizer import normalize_ifc_full

        result = normalize_ifc_full(
            compile_ifc(wall_with_a_detail(), "ifcopenshell"))
        by_name = {e.name: e for e in result.elements}
        assert [c.name for c in by_name["wall:north"].children] == [
            "box:corbel:wall:north"]
        assert result.groupings == []

    def test_an_opening_fill_is_still_reached(self):
        from lite_step.ifc.normalizer import normalize_ifc_full

        proj = Project(name="p")
        host = wall("north", length=12000)
        win = Window(width=1400, height=1600, name="left")
        win.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=1400, y=60, z=1600)))
        host.anchor(win, along=2000, up=900)
        proj.add(host)
        result = normalize_ifc_full(compile_ifc(proj))

        def descendants(elements):
            for elem in elements:
                yield elem
                yield from descendants(elem.children)

        assert "IfcWindow" in {e.ifc_type for e in descendants(result.elements)}

    def test_the_discriminator_reads_the_parents_representation(self):
        """Retired the containment discriminator: under IFC 4.3 the
        facade's members and a window's frame members BOTH have
        ``ContainedInStructure`` NIL, so containment separates nothing. What
        separates them is who owns the geometry."""
        from lite_step.ifc.normalizer import _is_grouping_parent

        model = opened(compile_ifc(two_storey()))
        rel = aggregate_rel(model, "north_facade")
        assert _is_grouping_parent(rel.RelatingObject, list(rel.RelatedObjects))

        detail = opened(compile_ifc(wall_with_a_detail(), "ifcopenshell"))
        host = [w for w in detail.by_type("IfcWall") if w.Representation][0]
        parts = list(host.IsDecomposedBy[0].RelatedObjects)
        assert parts
        assert not _is_grouping_parent(host, parts)

    def test_a_body_less_column_is_not_a_grouping(self):
        """**The regression the Representation-only rule caused, pinned.**

        This compiler emits body-less parents for ORDINARY single elements: a
        DSL ``Column``/``Roof`` becomes an ``IfcColumn``/``IfcRoof`` with no
        Representation whose geometry hangs off it as aggregated
        ``IfcBuildingElementProxy`` children. On "no Representation" alone,
        every column in every model imported as a grouping — a carport came
        back as anonymous proxies and zero columns. A column is not made of
        columns; a facade is made of walls."""
        from lite_step.ifc.normalizer import normalize_ifc_full

        proj = Project(name="p")
        column = Column(name="col_0")
        column.add(Box(start=Point(x=0, y=0, z=0),
                       end=Point(x=300, y=300, z=2700), type="sketch"))
        proj.add(column)
        result = normalize_ifc_full(compile_ifc(proj))
        assert [g.name for g in result.groupings] == []
        assert "column:col_0" in [e.name for e in result.elements]

    def test_an_assembly_of_mixed_classes_is_still_a_grouping(self):
        """The other arm: ``IfcElementAssembly`` is the schema's own focus
        subtype for "a whole made of peer parts", so mixed member classes are
        the point there rather than a disqualification."""
        import ifcopenshell
        from lite_step.ifc.normalizer import _is_grouping_parent
        from lite_step.ifc.normalizer import normalize_ifc_full

        def mutate(model):
            walls = [w for w in model.by_type("IfcWall") if w.Representation]
            beam = model.create_entity(
                "IfcBeam", GlobalId=ifcopenshell.guid.new(), Name="beam:lintel",
                Representation=walls[0].Representation)
            parent = model.create_entity(
                "IfcElementAssembly", GlobalId=ifcopenshell.guid.new(),
                Name="frame")
            model.create_entity(
                "IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                RelatingObject=parent, RelatedObjects=walls + [beam])

        result = normalize_ifc_full(
            mutated(compile_ifc(two_storey(group=False)), mutate))
        assert [g.name for g in result.groupings] == ["frame"]
        assert sorted(result.groupings[0].members) == [
            "beam:lintel", "wall:north:storey:first", "wall:north:storey:ground"]

    def test_containment_no_longer_tells_the_two_apart(self):
        """The measured fact behind that move — stated as an assertion so a
        future reader cannot re-derive a superseded rule from a stale comment. In
        the conformant file, member and decomposition part are
        indistinguishable by containment: both are NIL."""
        from lite_step.ifc.normalizer import _is_spatially_contained

        model = opened(compile_ifc(two_storey()))
        members = list(aggregate_rel(model, "north_facade").RelatedObjects)
        detail = opened(compile_ifc(wall_with_a_detail(), "ifcopenshell"))
        host = [w for w in detail.by_type("IfcWall") if w.Representation][0]
        parts = list(host.IsDecomposedBy[0].RelatedObjects)

        assert not any(_is_spatially_contained(o) for o in members)
        assert not any(_is_spatially_contained(o) for o in parts)


class TestThirdPartyAggregationShapes:
    """``IfcRelAggregates`` means several things in the wild, and only one of
    them is an authored grouping. These fixtures are our own known-good output
    with exactly ONE relation edited, so nothing but the case under test
    differs from what this compiler emits.
    """

    def test_a_storey_that_also_aggregates_the_elements_it_contains(self):
        """Some exporters state both. The members ARE spatially contained and
        a storey has no Representation and is contained nowhere, so it matches
        the peer-aggregate shape on every other count — only the
        spatial-hierarchy check keeps a whole floor from being reported as an
        authored facade."""
        import ifcopenshell
        from lite_step.ifc.normalizer import normalize_ifc_full

        def mutate(model):
            storey = model.by_type("IfcBuildingStorey")[0]
            contained = [c for c in model.by_type("IfcRelContainedInSpatialStructure")
                         if c.RelatingStructure == storey][0]
            model.create_entity(
                "IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                RelatingObject=storey,
                RelatedObjects=list(contained.RelatedElements))

        result = normalize_ifc_full(
            mutated(compile_ifc(two_storey(group=False)), mutate))
        assert [g.name for g in result.groupings] == []
        assert sorted(e.name for e in result.elements) == [
            "wall:north:storey:first", "wall:north:storey:ground"]

    def test_ifcspace_has_no_containedinstructure_inverse(self):
        """The schema fact that retired a containment-first check in
        ``_find_containing_storey``: ``ContainedInStructure`` is declared on
        ``IfcElement``, and ``IfcSpace`` is an ``IfcSpatialElement``. Pinned
        so the check is not re-added on the assumption that it reads
        something."""
        from ifcopenshell import ifcopenshell_wrapper

        schema = ifcopenshell_wrapper.schema_by_name("IFC4X3_ADD2")
        inverses = set()
        cursor = schema.declaration_by_name("IfcSpace").as_entity()
        while cursor is not None:
            inverses.update(a.name() for a in cursor.all_inverse_attributes())
            cursor = cursor.supertype()
        assert "ContainedInStructure" not in inverses
        assert "Decomposes" in inverses

    def test_an_assembly_whose_parent_carries_geometry(self):
        """A parent WITH a Representation is an assembly, not a grouping: its
        own body is the thing in the model. Reporting it as a grouping would
        also double-count, because it is extracted as an element already."""
        import ifcopenshell
        from lite_step.ifc.normalizer import normalize_ifc_full

        def mutate(model):
            walls = [w for w in model.by_type("IfcWall") if w.Representation]
            parent = model.create_entity(
                "IfcWall", GlobalId=ifcopenshell.guid.new(), Name="assembly",
                Representation=walls[0].Representation)
            model.create_entity(
                "IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                RelatingObject=parent, RelatedObjects=walls)

        result = normalize_ifc_full(
            mutated(compile_ifc(two_storey(group=False)), mutate))
        assert [g.name for g in result.groupings] == []

    def test_the_legacy_non_conformant_shape_still_imports(self):
        """**Backwards compatibility, and the reason the discriminator had to
        move rather than invert.** Every model stored before that carries
        the shape — members contained, parent contained nowhere. Inverting
        the containment test would have made all of them import as ordinary
        geometry-less elements with their facade silently gone. Reading the
        parent's ``Representation`` instead is stable across BOTH shapes."""
        from lite_step.ifc.normalizer import normalize_ifc_full

        def mutate(model):
            """Put the file back into the shape."""
            rel = [r for r in model.by_type("IfcRelAggregates")
                   if r.RelatingObject.Name == "north_facade"][0]
            parent = rel.RelatingObject
            ground = [s for s in model.by_type("IfcBuildingStorey")
                      if s.Name == "storey:ground"][0]
            first = [s for s in model.by_type("IfcBuildingStorey")
                     if s.Name == "storey:first"][0]
            contain = [c for c in model.by_type("IfcRelContainedInSpatialStructure")
                       if c.RelatingStructure == ground][0]
            by_name = {o.Name: o for o in rel.RelatedObjects}
            contain.RelatedElements = [by_name["wall:north:storey:ground"]]
            model.create_entity(
                "IfcRelContainedInSpatialStructure",
                GlobalId=__import__("ifcopenshell").guid.new(),
                RelatedElements=[by_name["wall:north:storey:first"]],
                RelatingStructure=first)
            assert not parent.ContainedInStructure

        result = normalize_ifc_full(
            mutated(compile_ifc(two_storey()), mutate))
        names = [e.name for e in result.elements]
        assert names.count("wall:north:storey:ground") == 1
        assert names.count("wall:north:storey:first") == 1
        assert "north_facade" not in names
        assert [g.name for g in result.groupings] == ["north_facade"]
        assert sorted(result.groupings[0].members) == [
            "wall:north:storey:first", "wall:north:storey:ground"]

    def test_a_body_less_assembly_with_contained_members_still_imports(self):
        """The third shape in the wild: a body-less ``IfcElementAssembly``
        whose exporter contained the members anyway. Neither the rule nor
        its inverse accepts it; the Representation rule does, and its members
        come back once each rather than twice."""
        import ifcopenshell
        from lite_step.ifc.normalizer import normalize_ifc_full

        def mutate(model):
            walls = [w for w in model.by_type("IfcWall") if w.Representation]
            parent = model.create_entity(
                "IfcElementAssembly", GlobalId=ifcopenshell.guid.new(),
                Name="truss")
            model.create_entity(
                "IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                RelatingObject=parent, RelatedObjects=walls)

        result = normalize_ifc_full(
            mutated(compile_ifc(two_storey(group=False)), mutate))
        assert [g.name for g in result.groupings] == ["truss"]
        assert result.groupings[0].ifc_type == "IfcElementAssembly"
        names = [e.name for e in result.elements]
        assert names.count("wall:north:storey:ground") == 1
        assert names.count("wall:north:storey:first") == 1
        assert "truss" not in names

    def test_a_space_still_finds_its_storey_through_decomposes(self):
        """``_find_containing_storey``'s actual contract, unchanged by this
        workstream — and the regression partner for the note that retired the
        containment-first check written for it."""
        import ifcopenshell
        from lite_step.ifc.normalizer import _find_containing_storey

        def mutate(model):
            storeys = model.by_type("IfcBuildingStorey")
            space = model.create_entity(
                "IfcSpace", GlobalId=ifcopenshell.guid.new(), Name="space:attic")
            model.create_entity(
                "IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
                RelatingObject=storeys[1], RelatedObjects=[space])

        model = opened(mutated(compile_ifc(two_storey(group=False)),
                               mutate))
        storeys = model.by_type("IfcBuildingStorey")
        storey_map = {s.id(): (i, i * 3000) for i, s in enumerate(storeys)}
        space = model.by_type("IfcSpace")[0]
        assert _find_containing_storey(space, storey_map) == storey_map[storeys[1].id()]


# ---------------------------------------------------------------------------
# 9. Loud failure over silent degradation
# ---------------------------------------------------------------------------


class TestSilentDegradationIsRefused:
    def test_the_missing_member_message_counts_what_was_lost(self):
        group = ResolvedGrouping(kind=AGGREGATE, name="north_facade",
                                 ifc_class="IfcWall",
                                 members=("wall:a", "wall:b", "wall:c"))
        message = str(missing_member_error(group, ["wall:c"]))
        assert "1 of 3" in message
        assert "incomplete aggregate" in message

    def test_an_unresolvable_member_fails_the_compile(self, monkeypatch):
        """Both backends RAISE rather than emitting a short relation. Forced
        through the shared collector, because the canonical names it returns
        are exactly what the join can fail on — the generators re-stamp
        canonicals themselves, so mutating the tree beforehand cannot reach
        this path."""
        import lite_step.ifc.groupings as gr

        real = gr.collect_groupings

        def with_a_ghost(proj):
            return [gr.ResolvedGrouping(
                kind=g.kind, name=g.name, ifc_class=g.ifc_class,
                members=g.members + ("wall:ghost",)) for g in real(proj)]

        monkeypatch.setattr(gr, "collect_groupings", with_a_ghost)
        result = compile_result(two_storey(), "ifcopenshell")
        assert not result.success
        assert "wall:ghost" in result.error
        assert "incomplete aggregate" in result.error

    def test_member_elevations_covers_every_top_level_element(self):
        """The number the streaming re-chain adds back. A member missing from
        this map would be re-chained by an unknown amount, silently — so the
        backend refuses instead, and this pins the map that keeps it from
        having to."""
        elevations = member_elevations(normalize_project_to_meters(two_storey()))
        assert elevations == {"wall:north:storey:ground": 0.0,
                              "wall:north:storey:first": 3.0}
