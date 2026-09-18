"""Material treatments, `IfcMaterial.Category`, and the 2D drawing wiring.

The invariants worth protecting here are mostly about what must NOT change:

1.  Treatments are additive. A registry entry with no ``treatments`` compiles
    to the same entities it always did.
2.  ``IfcMaterial.Category`` is the material FAMILY and
    ``IfcMaterialLayer.Category`` is the layer ROLE. Merging them would look
    harmless and silently break both consumers.
3.  Drawings never enter the LITESTEP_META manifest. ``IfcAnnotation`` is an
    ``IfcProduct``, so an ordering slip in the generator would put a
    ``PLAN_*`` entity into patch identity — a class of bug that surfaces
    much later, as a mis-applied edit.
4.  The generated stylesheet references pattern ids and copies no artwork.
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import compile_main, normalize_project_to_meters
from lite_step.draft import (
    CATEGORY_HATCH,
    canonicalise_class_name,
    render_stylesheet,
)
from lite_step.materials import REGISTRIES, registry_definition
from lite_step.materials.dk import Draft, MaterialDef, MassGeometry, Render
from lite_step.materials.treatments import (
    MassTreatment,
    SectionTreatment,
    UnitTreatment,
    resolve_module_mm,
)
from lite_step.models import Box, Material, Point, Project, Wall

pytest.importorskip("ifcopenshell")


def _all_materials() -> dict:
    merged: dict = {}
    for registry in REGISTRIES.values():
        merged.update(registry)
    return merged


def _wall_project() -> Project:
    proj = Project(name="draft-test")
    wall = Wall(name="south")
    wall.add(Box(start=Point(x=-3000, y=-2000, z=0),
                 end=Point(x=3000, y=-1700, z=2500),
                 material=Material(key="Concrete_C25-30")))
    proj.add(wall)
    return proj


def _generate(proj: Project) -> str:
    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    return result.ifc_content


# ---------------------------------------------------------------------------
# 1. Treatments — the union, and its additivity
# ---------------------------------------------------------------------------


class TestTreatmentModel:
    def test_default_is_empty_so_the_registry_is_unchanged(self):
        assert all(m.treatments == {} for m in _all_materials().values())

    def test_section_treatment_holds_a_profile(self):
        t = SectionTreatment(kind="section", profile=[(0, 0), (300, 0), (150, 40)])
        assert len(t.profile) == 3

    def test_profile_must_not_repeat_the_closing_point(self):
        # An implicitly-closed outline that also repeats its first point emits
        # a degenerate zero-length edge — rejected at construction rather than
        # surviving into a sweep.
        with pytest.raises(ValueError, match="must not repeat"):
            SectionTreatment(kind="section",
                             profile=[(0, 0), (300, 0), (150, 40), (0, 0)])

    def test_profile_needs_three_points(self):
        with pytest.raises(ValueError, match="at least 3 points"):
            SectionTreatment(kind="section", profile=[(0, 0), (300, 0)])

    def test_duplicate_points_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            SectionTreatment(kind="section",
                             profile=[(0, 0), (300, 0), (300, 0), (150, 40)])

    def test_unit_treatment_requires_positive_extrusion(self):
        with pytest.raises(ValueError):
            UnitTreatment(kind="unit", profile=[(0, 0), (10, 0), (10, 10)],
                          extrude_mm=0)

    def test_discriminator_selects_the_right_member(self):
        mdef = MaterialDef(
            psets={}, render=Render(rgb="#AABBCC", alpha=1.0, finish="matt"),
            geometry=MassGeometry(form="mass"),
            treatments={"plain": {"kind": "mass"}},
        )
        assert isinstance(mdef.treatments["plain"], MassTreatment)

    def test_extra_fields_still_forbidden(self):
        with pytest.raises(ValueError):
            MassTreatment(kind="mass", nonsense=1)


class TestModulePitchIsDerivedNotRestated:
    def test_explicit_module_wins(self):
        t = UnitTreatment(kind="unit", profile=[(0, 0), (10, 0), (10, 10)],
                          extrude_mm=5, module_mm=(111, 222))
        assert resolve_module_mm(t, registry_definition("Brick_Red_DK").geometry) \
            == (111, 222)

    def test_sheet_pitch_comes_from_the_tile_product_facts(self):
        tile = registry_definition("ClayTile_Black_Engobed")
        t = UnitTreatment(kind="unit", profile=[(0, 0), (10, 0), (10, 10)],
                          extrude_mm=5)
        along, across = resolve_module_mm(t, tile.geometry)
        # The cover width and batten spacing are already product facts; the
        # treatment must not restate them.
        assert along == tile.geometry.cover_width_mm
        assert across == tile.geometry.batten_spacing_mm[1]

    def test_underivable_pitch_raises_rather_than_guessing(self):
        # A silently-guessed pitch tiles a roof wrong and looks plausible.
        plain = MassGeometry(form="mass")
        t = UnitTreatment(kind="unit", profile=[(0, 0), (10, 0), (10, 10)],
                          extrude_mm=5)
        with pytest.raises(ValueError, match="module_mm"):
            resolve_module_mm(t, plain)


# ---------------------------------------------------------------------------
# 2. IfcMaterial.Category — the family, NOT the layer role
# ---------------------------------------------------------------------------


class TestMaterialCategory:
    def test_every_registry_entry_declares_a_family(self):
        missing = sorted(k for k, m in _all_materials().items() if m.category is None)
        assert missing == [], f"registry entries with no category: {missing}"

    def test_every_family_has_a_hatch_mapping(self):
        used = {m.category for m in _all_materials().values() if m.category}
        assert used <= set(CATEGORY_HATCH), used - set(CATEGORY_HATCH)

    def test_category_reaches_the_ifc_material(self):
        content = _generate(_wall_project())
        assert "IFCMATERIAL('Concrete_C25-30',$,'concrete')" in content

    def test_family_and_layer_role_stay_different_attributes(self):
        # IfcMaterial.Category is the family (concrete); IfcMaterialLayer
        # .Category is the role (LoadBearing). Collapsing them would break
        # 2D hatching and buildup semantics at once.
        mdef = registry_definition("Concrete_C25-30")
        assert mdef.category == "concrete"
        assert mdef.function == "LoadBearing"

    def test_families_survive_class_canonicalisation_unchanged(self):
        # Drawing tools strip non-alphanumerics WITHOUT lowercasing, so a
        # family containing "_" or an uppercase letter would silently produce
        # a CSS token nobody's stylesheet matches.
        for family in CATEGORY_HATCH:
            assert canonicalise_class_name(family) == family


# ---------------------------------------------------------------------------
# 3. The generated stylesheet
# ---------------------------------------------------------------------------


class TestStylesheet:
    def test_emits_a_rule_per_family_present_in_the_registry(self):
        css = render_stylesheet(_all_materials())
        for family, hatch in CATEGORY_HATCH.items():
            if hatch is None:
                continue
            assert f".cut.layer-material-category-{family}" in css

    def test_no_rules_for_families_the_registry_lacks(self):
        one = {"X": MaterialDef(psets={},
                                render=Render(rgb="#AABBCC", alpha=1.0, finish="matt"),
                                geometry=MassGeometry(form="mass"),
                                category="timber")}
        css = render_stylesheet(one)
        assert "layer-material-category-timber" in css
        assert "layer-material-category-concrete" not in css

    def test_per_material_override_beats_the_family_rule(self):
        mats = {
            "Special_Board": MaterialDef(
                psets={}, render=Render(rgb="#AABBCC", alpha=1.0, finish="matt"),
                geometry=MassGeometry(form="mass"), category="board",
                draft=Draft(hatch="honeycomb", line_weight_mm=0.7)),
        }
        css = render_stylesheet(mats)
        family_at = css.index(".cut.layer-material-category-board")
        override_at = css.index(".cut.material-SpecialBoard")
        # Equal specificity (two classes each), so source order decides.
        assert override_at > family_at
        assert "url(#honeycomb)" in css
        assert "stroke-width: 0.7" in css

    def test_override_targets_both_the_element_and_the_layer_selector(self):
        mats = {"K_1": MaterialDef(
            psets={}, render=Render(rgb="#AABBCC", alpha=1.0, finish="matt"),
            geometry=MassGeometry(form="mass"), category="timber",
            draft=Draft(cut_rgb="#112233"))}
        css = render_stylesheet(mats)
        assert ".cut.material-K1" in css and ".cut.layer-material-K1" in css

    def test_ships_no_pattern_artwork(self):
        # We reference pattern ids; copying the artwork would drag a GPL
        # obligation into an Apache-2.0 repo.
        css = render_stylesheet(_all_materials())
        assert "<pattern" not in css and "<svg" not in css
        assert "url(#" in css

    def test_unstyled_material_contributes_nothing(self):
        mats = {"Nameless": MaterialDef(
            psets={}, render=Render(rgb="#AABBCC", alpha=1.0, finish="matt"),
            geometry=MassGeometry(form="mass"))}
        css = render_stylesheet(mats)
        assert "layer-material-category-" not in css


# ---------------------------------------------------------------------------
# 4. Drawing wiring inside the IFC
# ---------------------------------------------------------------------------


class TestDrawingWiring:
    def test_documentation_pset_points_at_the_generated_stylesheet(self):
        content = _generate(_wall_project())
        assert "'BBIM_Documentation'" in content
        assert "'StylesheetPath'" in content
        assert "styles/lite-step.css" in content

    def test_patterns_path_is_left_unset(self):
        # Unset means the drawing tool keeps its own pattern library, which
        # is what lets our CSS reference url(#concrete) without us shipping
        # anyone's artwork.
        content = _generate(_wall_project())
        assert "'PatternsPath'" not in content

    def test_one_plan_drawing_per_storey(self):
        content = _generate(_wall_project())
        assert content.count("IFCANNOTATION(") == 1
        assert "'DRAWING'" in content
        assert "IFCLABEL('PLAN_VIEW')" in content

    def test_pset_carries_every_property_the_tool_activates_on(self):
        # Bonsai errors rather than defaulting when a drawing property it
        # expects is absent — ShadingStyles missing is a red dialog on
        # activate, AFTER the view has already opened correctly, which is the
        # worst possible shape for a non-professional user. Measured against
        # Bonsai 0.8.5.
        content = _generate(_wall_project())
        for prop in ("TargetView", "Scale", "HumanScale", "Stylesheet",
                     "ShadingStyles", "CurrentShadingStyle", "Metadata"):
            assert f"'{prop}'" in content, prop

    def test_shading_style_name_is_one_the_shipped_file_defines(self):
        # A name absent from shading_styles.json degrades to "no active style"
        # and then fails activate_drawing_style. The shipped keys are
        # Technical / Shaded / Blender Default.
        from lite_step.ifc.drawings import DEFAULT_SHADING_STYLE
        assert DEFAULT_SHADING_STYLE in ("Technical", "Shaded", "Blender Default")

    def test_metadata_projects_material_identity_onto_classes(self):
        # Without Metadata the tool emits no material classes at all and the
        # whole stylesheet binds to nothing.
        content = _generate(_wall_project())
        assert "mats.Category" in content
        assert "mats.Name" in content

    def test_drawings_never_enter_the_manifest(self, tmp_path):
        from lite_step.ifc.embedder import extract_litestep_meta

        src = tmp_path / "model.py"
        src.write_text("# manifest purity fixture\n", encoding="utf-8")
        out = compile_main(result=_wall_project(), source_path=str(src))
        meta = extract_litestep_meta(out.read_text(encoding="utf-8", errors="replace"))
        assert meta is not None
        polluted = [k for k in meta.manifest if "PLAN" in k or "DRAWING" in k]
        assert polluted == [], polluted

    def test_kill_switch_removes_the_wiring_entirely(self, monkeypatch):
        monkeypatch.setenv("LITESTEP_EMIT_DRAWINGS", "0")
        content = _generate(_wall_project())
        assert "IFCANNOTATION(" not in content
        assert "BBIM_Documentation" not in content

    def test_compile_writes_the_stylesheet_beside_the_ifc(self, tmp_path):
        src = tmp_path / "model.py"
        src.write_text("# stylesheet fixture\n", encoding="utf-8")
        out = compile_main(result=_wall_project(), source_path=str(src))
        css = out.parent / "styles" / "lite-step.css"
        assert css.is_file()
        assert "layer-material-category-concrete" in css.read_text(encoding="utf-8")
