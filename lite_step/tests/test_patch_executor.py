"""Tests for the patch-mode entry point (lite_step.compiler.patch_executor)."""

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.compiler.patch_executor import patch_main
from lite_step.ifc.embedder import (
    extract_litestep_meta,
)

# A compilable model that produces a single-storey single-wall Project.
DESIRED_SOURCE = """\
from lite_step.models.project import Project, Storey
from lite_step.models.primitives import Point
from lite_step.models.elements import Wall, Box, Extrude

def generate_project():
    proj = Project(name="Patched")
    storey = Storey(elevation=0)
    proj.add_storey(storey)
    wall = Wall(id="wall_north")
    wall.add(Box(
        start=Point(x=0, y=0, z=0),
        end=Point(x=6000, y=200, z=3000),
        type="sketch",
    ))
    storey.add(wall)
    return proj

result = generate_project()
"""


def _build_litestep_base_ifc(source: str = DESIRED_SOURCE) -> str:
    """Run the lite_step compile pipeline once to get an IFC that
    carries LITESTEP_META — the canonical hot-path base."""
    from lite_step.compiler.executor import execute_lite_step_script
    from lite_step.ifc.generator import generate_ifc

    exec_result = execute_lite_step_script(source)
    assert exec_result.success, exec_result.error
    gen = generate_ifc(exec_result.project, source_code=source)
    assert gen.success, gen.error
    return gen.ifc_content


def _build_bare_base_ifc() -> str:
    """A minimal IFC with NO LITESTEP_META — the cold-path base."""
    model = ifcopenshell.file(schema="IFC4")
    project = model.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new())
    site = model.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new())
    building = model.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new())
    model.create_entity(
        "IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
        RelatingObject=project, RelatedObjects=[site],
    )
    model.create_entity(
        "IfcRelAggregates", GlobalId=ifcopenshell.guid.new(),
        RelatingObject=site, RelatedObjects=[project],
    )
    return model.to_string()


class TestHotPath:
    def test_base_with_meta_routes_to_round_trip_mode(self):
        base = _build_litestep_base_ifc()
        result = patch_main(base, DESIRED_SOURCE)
        assert result.success
        assert result.mode == "round_trip"

    def test_hot_path_output_has_fresh_meta(self):
        base = _build_litestep_base_ifc()
        result = patch_main(base, DESIRED_SOURCE)
        out_meta = extract_litestep_meta(result.ifc_content)
        assert out_meta is not None
        assert out_meta.hash_valid is True
        assert out_meta.cold_imported is False


class TestColdPath:
    def test_bare_base_routes_to_cold_import_mode(self):
        base = _build_bare_base_ifc()
        result = patch_main(base, DESIRED_SOURCE)
        assert result.success
        assert result.mode == "cold_import"

    def test_cold_path_output_marked_cold_imported(self):
        base = _build_bare_base_ifc()
        result = patch_main(base, DESIRED_SOURCE)
        out_meta = extract_litestep_meta(result.ifc_content)
        assert out_meta is not None
        assert out_meta.cold_imported is True

    def test_cold_imported_is_sticky_across_repeat_edits(self):
        """First edit: bare base → cold_imported=true. Second edit:
        use the FIRST edit's output as the new base. cold_imported
        must stay True across the subsequent edit."""
        bare = _build_bare_base_ifc()
        first = patch_main(bare, DESIRED_SOURCE)
        assert first.mode == "cold_import"
        # Second edit reuses first output as base
        second = patch_main(first.ifc_content, DESIRED_SOURCE)
        assert second.mode == "cold_import", "cold_imported flag must persist"
        out_meta = extract_litestep_meta(second.ifc_content)
        assert out_meta is not None
        assert out_meta.cold_imported is True


class TestFailureSurface:
    def test_unparseable_source_returns_error(self):
        base = _build_litestep_base_ifc()
        result = patch_main(base, "this isn't python at all #$")
        assert result.success is False
        assert result.error is not None or result.warnings

    def test_compilation_failure_surfaces_needs_llm_with_prompt(self):
        """Worker dispatcher relies on needs_llm + llm_prompt to route
        to an LLM-fix turn instead of returning the error to the user."""
        base = _build_litestep_base_ifc()
        result = patch_main(base, "this isn't python at all #$")
        assert result.success is False
        assert result.needs_llm is True
        assert result.llm_prompt is not None
        assert "Compilation Fix" in result.llm_prompt
        assert result.mode == "llm_assisted"


def _base_stamped(version: str, monkeypatch) -> str:
    """A LITESTEP base IFC carrying an arbitrary META ``version``.

    The guard reads the version embedded in the BASE, not the module
    constant, so the constant is restored before the base is used.
    """
    import lite_step.ifc.embedder as emb
    monkeypatch.setattr(emb, "LITESTEP_META_VERSION", version)
    base = _build_litestep_base_ifc()
    assert extract_litestep_meta(base).version == version
    monkeypatch.undo()
    return base


class TestForwardOnlyVersionGate:
    """A stored base may be continued only by the compiler that wrote it.

    This is an ALLOWLIST — anything that is not the current version is
    refused — and that is the whole point. The predecessor enumerated the
    known-bad versions ("1".."8"), which meant a base stamped by a NEWER
    compiler passed straight through unguarded. ``test_future_version_*``
    is the case a ladder structurally cannot catch.
    """

    # Every version the retired ladder listed, plus the two it could not:
    # a version far past the ladder's end, and a plausible NEXT one.
    @pytest.mark.parametrize(
        "version", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
                    "12", "13", "14", "15", "16", "17", "19", "99"],
    )
    def test_foreign_version_base_is_refused(self, version, monkeypatch):
        base = _base_stamped(version, monkeypatch)
        result = patch_main(base, DESIRED_SOURCE)
        assert result.success is False
        assert "different version of Lite-STEP" in result.error
        # Actionable: names what it found, what it wanted, and the way out.
        assert f"version {version}" in result.error
        assert "start a fresh build" in result.error
        assert result.needs_llm is False  # not an LLM-fixable compile error

    def test_future_version_is_refused_by_the_allowlist(self, monkeypatch):
        """The forward case a denylist cannot express.

        A base written by a compiler NEWER than this one carries a version
        no list of known-bad values will ever contain, and its stored
        source is precisely the source most likely to author DSL this
        compiler does not implement.
        """
        import lite_step.ifc.embedder as emb
        future = str(int(emb.LITESTEP_META_VERSION) + 1)
        base = _base_stamped(future, monkeypatch)
        result = patch_main(base, DESIRED_SOURCE)
        assert result.success is False
        assert f"version {future}" in result.error

    def test_current_base_continues_fine(self):
        import lite_step.ifc.embedder as emb
        base = _build_litestep_base_ifc()  # embeds the current version
        assert extract_litestep_meta(base).version == emb.LITESTEP_META_VERSION
        result = patch_main(base, DESIRED_SOURCE)
        assert result.success

    def test_bare_base_still_cold_imports(self):
        # No META = third-party export, NOT a stale model — must be exempt
        # from the gate and take the normal cold-import path.
        base = _build_bare_base_ifc()
        result = patch_main(base, DESIRED_SOURCE)
        assert result.success
        assert result.mode == "cold_import"
