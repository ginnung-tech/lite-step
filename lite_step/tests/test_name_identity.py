"""Name-keyed identity (DSL v1.5, WS1 PR-B) — additive tests.

``name=`` is the preferred element identity:

1.  Uniqueness across the Project is compile-enforced (containers and
    opening children included; anonymous elements never collide).
2.  Both IFC backends emit ``Name = elem.name or elem.id``
    (``BimElement.ifc_name``); the LITESTEP_META manifest keys on the
    emitted Name.
3.  The patch-mode differ keys on the emitted Name, so renaming via
    ``name=`` is delete + create (spatial fallback may re-match a rename
    with *identical* geometry — FR-4.7's external-editor recovery, pinned
    here as documented behavior).
4.  ast_walker / patcher resolve source identity ``name=`` first,
    ``id=`` fallback.

``id=`` stays fully functional — the hard cutover is WS2.
"""

import pytest

from lite_step.compiler.executor import (
    execute_lite_step_script,
    normalize_project_to_meters,
    validate_project,
)
from lite_step.ifc.generator import generate_ifc
from lite_step.models.project import Project, Storey
from lite_step.models.elements import Box, Extrude, Wall, Window
from lite_step.models.primitives import Point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid(offset_z: int = 0, **kwargs) -> "BimElement":
    """Minimal valid box Solid; stack via offset_z to avoid overlaps."""
    return Box(
        start=Point(x=0, y=0, z=offset_z),
        end=Point(x=5000, y=3000, z=offset_z + 200),
        type="sketch",
        **kwargs,
    )


def _building(*elements) -> Project:
    # Single storey is anonymous (DSL v2.1) -> no ``storey:`` path segment.
    proj = Project(name="Test")
    storey = Storey(elevation=0)
    for elem in elements:
        storey.add(elem)
    proj.add_storey(storey)
    return proj


def _name_errors(errors):
    return [e for e in errors if "Duplicate canonical name" in e]


# ---------------------------------------------------------------------------
# 1. Name uniqueness (compile-enforced)
# ---------------------------------------------------------------------------

class TestNameUniqueness:
    def test_unique_names_pass(self):
        proj = _building(_solid(0, name="wall_a"), _solid(1000, name="wall_b"))
        assert _name_errors(validate_project(proj)) == []

    def test_duplicate_names_same_storey_error(self):
        proj = _building(_solid(0, name="wall_a"), _solid(1000, name="wall_a"))
        errs = [e for e in validate_project(proj) if "Duplicate canonical name" in e]
        assert len(errs) == 1
        assert "box:wall_a" in errs[0]

    def test_same_leaf_across_named_storeys_coexists(self):
        """DSL v2.1: a NAMED storey contributes a ``storey:<leaf>`` segment, so
        the same element leaf on two floors derives DISTINCT canonicals
        (``box:shared:storey:ground`` vs ``box:shared:storey:first``) — no
        collision. (Inverts the pre-v2.1 flat-name behavior.)"""
        proj = Project(name="Test")
        s0 = Storey(name="ground", elevation=0)
        s0.add(_solid(0, name="shared"))
        s1 = Storey(name="first", elevation=3000)
        s1.add(_solid(0, name="shared"))
        proj.add_storey(s0)
        proj.add_storey(s1)
        assert _name_errors(validate_project(proj)) == []

    def test_same_leaf_different_scope_is_fine(self):
        """DSL v2.1: a container child and a top-level element sharing a leaf
        derive DIFFERENT canonical paths (box:dup_me:wall:wall_s vs box:dup_me)
        — no collision. The walk still descends into container children."""
        wall = Wall(name="wall_s")
        wall.add(_solid(0, name="dup_me"))
        proj = _building(wall, _solid(1000, name="dup_me"))
        assert _name_errors(validate_project(proj)) == []

    def test_duplicate_opening_children_same_wall_error(self):
        """Two windows in the SAME wall may not share a leaf — same scope, same
        type, same leaf -> identical canonical. (Different walls are fine —
        per-scope.)"""
        wall_a = Wall(name="wall_a")
        wall_a.add(_solid(0))
        wall_a.anchor(Window(width=1200, height=1400, name="win_1"),
                      along=500, up=900)
        wall_a.anchor(Window(width=1200, height=1400, name="win_1"),
                      along=3000, up=900)
        proj = _building(wall_a)
        errs = [e for e in validate_project(proj) if "Duplicate canonical name" in e]
        assert len(errs) == 1
        assert "window:win_1:wall:wall_a" in errs[0]

    def test_same_leaf_windows_different_walls_fine(self):
        """Per-scope: same window leaf in DIFFERENT walls -> distinct canonical."""
        wall_a = Wall(name="wall_a")
        wall_a.add(_solid(0))
        wall_a.anchor(Window(width=1200, height=1400, name="win_1"),
                      along=500, up=900)
        wall_b = Wall(name="wall_b")
        wall_b.add(_solid(1000))
        wall_b.anchor(Window(width=1200, height=1400, name="win_1"),
                      along=500, up=900)
        assert _name_errors(validate_project(_building(wall_a, wall_b))) == []

    def test_anonymous_elements_never_collide(self):
        proj = _building(_solid(0), _solid(1000), _solid(2000))
        assert _name_errors(validate_project(proj)) == []

    def test_same_name_across_different_buildings_fine(self):
        b1 = _building(_solid(0, name="wall_main"))
        b2 = _building(_solid(0, name="wall_main"))
        assert _name_errors(validate_project(b1)) == []
        assert _name_errors(validate_project(b2)) == []

    def test_name_matching_anonymous_id_is_fine(self):
        """DSL v2.1: id= is internal-only (anonymous emits Name=None), so a
        name= leaf equal to another element's id can never collide — the
        pre-v2.1 name-vs-anonymous-id check was deleted."""
        proj = _building(_solid(0, id="foo"), _solid(1000, name="foo"))
        assert _name_errors(validate_project(proj)) == []

    def test_name_equal_to_own_id_fine(self):
        proj = _building(_solid(0, id="foo", name="foo"))
        assert _name_errors(validate_project(proj)) == []

    def test_name_matching_named_elements_unused_id_fine(self):
        """An id shadowed by that element's own name is never emitted, so
        another element may carry it as a name."""
        proj = _building(
            _solid(0, id="shadowed", name="wall_a"),
            _solid(1000, name="shadowed"),
        )
        assert _name_errors(validate_project(proj)) == []

    def test_duplicate_ids_no_longer_error(self):
        """DSL v2.1: id= is internal-only and never emitted, so the
        there is no id-uniqueness check — a duplicate id= does not error."""
        proj = _building(_solid(0, id="same_id"), _solid(1000, id="same_id"))
        errs = [e for e in validate_project(proj) if "Duplicate element ID" in e]
        assert errs == []


# ---------------------------------------------------------------------------
# 2. Generators: IFC Name = name-when-set, id-fallback
# ---------------------------------------------------------------------------

def _generate(proj: Project, backend: str) -> str:
    normalized = normalize_project_to_meters(proj)
    result = generate_ifc(normalized)
    assert result.success, f"generate_ifc({backend}) failed: {result.error}"
    return result.ifc_content


class TestGeneratorNameEmission:
    def test_name_wins_when_set(self):
        if "ifcopenshell" == "ifcopenshell":
            pytest.importorskip("ifcopenshell")
        proj = _building(_solid(0, id="legacy_id", name="named_box"))
        content = _generate(proj, "ifcopenshell")
        assert "'box:named_box'" in content
        assert "'legacy_id'" not in content

    def test_anonymous_emits_no_name(self):
        """DSL v2.1: an anonymous element emits Name=None ($), never its id."""
        if "ifcopenshell" == "ifcopenshell":
            pytest.importorskip("ifcopenshell")
        proj = _building(_solid(0, id="only_id"))
        content = _generate(proj, "ifcopenshell")
        assert "'only_id'" not in content
        # The proxy product carries Name=$ (None), not a fabricated string.
        proxy = [l for l in content.splitlines() if "BUILDINGELEMENTPROXY" in l]
        assert proxy and ",$,$," in proxy[0].replace("'", "")

    def test_named_window_child_emits_name(self):
        if "ifcopenshell" == "ifcopenshell":
            pytest.importorskip("ifcopenshell")
        wall = Wall(name="wall_s")
        wall.add(
            Box(start=Point(x=0, y=0, z=0), end=Point(x=5000, y=300, z=2700),
                  type="wall"),
        )
        wall.anchor(Window(width=1200, height=1400,
                           id="win_id", name="win_named"), along=1500, up=900)
        content = _generate(_building(wall), "ifcopenshell")
        assert "'window:win_named:wall:wall_s'" in content
        # The opening void keys off the same identity
        assert "'window:win_named:wall:wall_s_void'" in content
        assert "'win_id'" not in content

    def test_manifest_keys_on_emitted_name(self):
        """LITESTEP_META manifest maps emitted Name → GlobalId, so a named
        element appears under its name, an anonymous one under its id."""
        from lite_step.ifc.embedder import extract_litestep_meta

        source = (
            'result = None  # placeholder — manifest test compiles directly\n'
        )
        proj = _building(_solid(0, name="named_box"), _solid(1000, id="anon_box"))
        normalized = normalize_project_to_meters(proj)
        result = generate_ifc(normalized, source_code=source)
        assert result.success
        meta = extract_litestep_meta(result.ifc_content)
        assert meta is not None
        assert "box:named_box" in meta.manifest
        # DSL v2.1: anonymous (id-only) elements emit Name=None -> absent
        assert "anon_box" not in meta.manifest


# ---------------------------------------------------------------------------
# 3. Differ: rename via name= is delete + create
# ---------------------------------------------------------------------------

from lite_step.ifc.normalizer import ExtractedElement, GeometryInfo, PlacementInfo  # noqa: E402
from lite_step.transpiler.differ import DeltaCategory, diff_elements  # noqa: E402


def _extracted(name, start=(0, 0, 0), end=(1000, 1000, 1000)) -> ExtractedElement:
    return ExtractedElement(
        ifc_type="IfcWall",
        name=name,
        guid="test_guid",
        storey_idx=0,
        storey_elevation=0,
        placement=PlacementInfo(translation=(0, 0, 0), rotation=(0, 0, 0)),
        geometry=GeometryInfo(mode="box", start=start, end=end),
    )


class TestDifferRenameSemantics:
    def test_rename_is_delete_plus_create(self):
        """Renamed element (geometry also moved out of overlap) diffs as
        DELETED old-name + ADDED new-name — name is the identity key."""
        orig = [_extracted("wall_old")]
        mod = [_extracted("wall_new", start=(50000, 0, 0), end=(51000, 1000, 1000))]
        deltas = diff_elements(orig, mod)
        cats = {d.element_id: d.category for d in deltas}
        assert cats == {
            "wall_old": DeltaCategory.DELETED,
            "wall_new": DeltaCategory.ADDED,
        }

    def test_rename_identical_geometry_hits_spatial_fallback(self):
        """FR-4.7 pin: a rename with byte-identical geometry is re-matched
        by the >=90%-bbox-overlap fallback (external editors delete+recreate
        elements under new names). The pair is reported under the OLD name
        with spatial_fallback details, not as delete+create."""
        orig = [_extracted("wall_old")]
        mod = [_extracted("wall_new")]
        deltas = diff_elements(orig, mod)
        assert len(deltas) == 1
        d = deltas[0]
        assert d.element_id == "wall_old"
        assert d.details.get("spatial_fallback") is True
        assert d.details.get("matched_from") == "wall_new"

    def test_stable_name_edit_in_place(self):
        """Same name, changed geometry → in-place delta, never delete+create."""
        orig = [_extracted("wall_a")]
        mod = [_extracted("wall_a", end=(2000, 1000, 1000))]
        deltas = diff_elements(orig, mod)
        assert len(deltas) == 1
        assert deltas[0].category == DeltaCategory.GEOMETRY_RESIZED


class TestPatchPipelineNameIdentity:
    """End-to-end: script → IFC → normalize; name is the identity that
    round-trips through the emitted Name field."""

    SOURCE_A = """
def generate_project():
    proj = Project(name="Patch Test")
    proj.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=5000, y=3000, z=200),
                   type="sketch", name="box_a"))
    return proj

result = generate_project()
"""

    def test_named_element_round_trips_through_ifc_name(self):
        from lite_step.ifc.normalizer import normalize_ifc

        exec_result = execute_lite_step_script(self.SOURCE_A)
        assert exec_result.success, exec_result.error
        ifc = generate_ifc(exec_result.project, source_code=self.SOURCE_A)
        assert ifc.success
        elements = normalize_ifc(ifc.ifc_content)
        names = {e.name for e in elements}
        assert "box:box_a" in names

    def test_rename_moves_manifest_key(self):
        """Recompiling with a changed name= drops the old manifest key and
        introduces the new one — delete + create at the identity level."""
        from lite_step.ifc.embedder import extract_litestep_meta

        source_b = self.SOURCE_A.replace('name="box_a"', 'name="box_b"')
        for source, expect_present, expect_absent in (
            (self.SOURCE_A, "box:box_a", "box:box_b"),
            (source_b, "box:box_b", "box:box_a"),
        ):
            exec_result = execute_lite_step_script(source)
            assert exec_result.success, exec_result.error
            ifc = generate_ifc(exec_result.project, source_code=source)
            assert ifc.success
            meta = extract_litestep_meta(ifc.ifc_content)
            assert meta is not None
            assert expect_present in meta.manifest
            assert expect_absent not in meta.manifest


# ---------------------------------------------------------------------------
# 4. ast_walker / patcher: name= first, id= fallback
# ---------------------------------------------------------------------------

from lite_step.transpiler.ast_walker import walk_source  # noqa: E402


class TestAstWalkerNameFirst:
    def test_name_only(self):
        nodes = walk_source('w = Wall(name="wall_n")\n')
        assert "wall_n" in nodes
        assert nodes["wall_n"].element_type == "Wall"

    def test_id_only_fallback(self):
        nodes = walk_source('w = Wall(id="wall_legacy")\n')
        assert "wall_legacy" in nodes

    def test_name_wins_over_id(self):
        nodes = walk_source('w = Wall(name="the_name", id="the_id")\n')
        assert "the_name" in nodes
        assert "the_id" not in nodes

    def test_explicit_none_name_falls_back_to_id(self):
        nodes = walk_source('w = Wall(name=None, id="the_id")\n')
        assert "the_id" in nodes

    def test_unresolvable_name_does_not_misbind_to_id(self):
        """name=<variable> emits a runtime Name the walker can't know;
        binding to id= would patch the wrong identity — skip instead."""
        nodes = walk_source('n = "x"\nw = Wall(name=n, id="the_id")\n')
        assert "the_id" not in nodes
        assert not any(k for k in nodes if not k.startswith("__pattern__"))

    def test_fstring_name_pattern(self):
        source = (
            "for i in range(3):\n"
            '    c = Column(name=f"col_{i}")\n'
        )
        nodes = walk_source(source)
        assert "__pattern__col_" in nodes
        assert nodes["__pattern__col_"].id_pattern == "col_"


from lite_step.transpiler.patcher import _extract_id_from_call, apply_patches  # noqa: E402
import libcst as cst  # noqa: E402


def _first_call(source: str) -> cst.Call:
    module = cst.parse_module(source)
    calls = []

    class _F(cst.CSTVisitor):
        def visit_Call(self, node: cst.Call) -> bool:
            calls.append(node)
            return True

    module.visit(_F())
    return calls[0]


class TestPatcherNameResolution:
    def test_extract_name_first(self):
        call = _first_call('Wall(name="the_name", id="the_id")')
        assert _extract_id_from_call(call) == "the_name"

    def test_extract_id_fallback(self):
        call = _first_call('Wall(id="the_id")')
        assert _extract_id_from_call(call) == "the_id"

    def test_extract_explicit_none_name_uses_id(self):
        call = _first_call('Wall(name=None, id="the_id")')
        assert _extract_id_from_call(call) == "the_id"

    def test_extract_unresolvable_name_returns_none(self):
        call = _first_call('Wall(name=some_var, id="the_id")')
        assert _extract_id_from_call(call) is None

    def test_apply_patches_moves_named_element(self):
        """A delta keyed by name= patches the matching constructor."""
        source = (
            'w = Box(name="box_a", start=Point(x=0, y=0, z=0), '
            "end=Point(x=5000, y=3000, z=200))\n"
        )
        nodes = walk_source(source)
        assert "box_a" in nodes
        from lite_step.transpiler.differ import ElementDelta

        delta = ElementDelta(
            element_id="box_a",
            category=DeltaCategory.PLACEMENT_MOVED,
            original=_extracted("box_a", start=(0, 0, 0), end=(5000, 3000, 200)),
            modified=_extracted("box_a", start=(1000, 0, 0), end=(6000, 3000, 200)),
        )
        patched, unhandled = apply_patches(source, [delta], [], nodes)
        assert unhandled == []
        assert "x=1000" in patched
        assert "x=6000" in patched

    def test_apply_patches_deletes_named_element(self):
        source = (
            'keep = Box(name="box_keep", start=Point(x=0, y=0, z=0), '
            "end=Point(x=5000, y=3000, z=200))\n"
            'gone = Box(name="box_gone", start=Point(x=0, y=0, z=1000), '
            "end=Point(x=5000, y=3000, z=1200))\n"
        )
        nodes = walk_source(source)
        from lite_step.transpiler.differ import ElementDelta

        delta = ElementDelta(
            element_id="box_gone",
            category=DeltaCategory.DELETED,
            original=_extracted("box_gone"),
        )
        patched, _ = apply_patches(source, [delta], [], nodes)
        assert "box_keep" in patched
        assert "box_gone" not in patched
