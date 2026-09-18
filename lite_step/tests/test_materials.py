"""DK material registry + canonical Material model (WS1 PR-A, DSL v1.5).

Covers:
- registry load + the dk module's own __main__ self-test
- Material construction validation matrix (off-catalog fail, off-stock warn,
  wrong-form fail, unknown-key-with-dimension fail)
- bare-string -> Material resolution and the legacy-vocabulary carve-out
- validate_project errors-vs-warnings split (warning does NOT raise,
  error does)
- one-source rule (Material dimension vs the element's own source)
- generator emission: IfcMaterial / IfcRelAssociatesMaterial /
  IfcSurfaceStyle-with-alpha for a Glass_VIG sheet
- legacy vocabulary backward compat (no warning, no IfcMaterial, streaming
  path retained)
- color="glass" sketch palette entry
"""

import runpy
import warnings as warnings_module

import pytest

from lite_step.models import Project, Material, Point, Sweep, Slab, Box, Extrude, Wall
from lite_step.compiler.executor import (
    ValidationReport,
    validate_project,
    validate_project_report,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestDkRegistry:
    def test_registry_loads_with_28_materials(self):
        from lite_step.materials.dk import MATERIALS

        assert len(MATERIALS) == 28

    def test_registry_covers_all_seven_forms(self):
        from lite_step.materials.dk import MATERIALS

        forms = {m.geometry.form for m in MATERIALS.values()}
        assert forms == {"mass", "sheet", "member", "bar", "membrane", "fill", "cavity"}

    def test_spec_key_list_matches(self):
        """The exact key list is part of the v1.5 agent-facing contract."""
        from lite_step.materials.dk import MATERIALS

        spec_keys = {
            "Concrete_C30-37", "Concrete_C25-30", "Steel_B500B",
            "Glass_Monolithic", "Glass_VIG", "Brick_Red_DK",
            "Mortar_KC50-50-700", "Timber_C24", "Glulam_GL24h",
            "Glulam_GL28h",
            "Chipboard_P6", "MineralWool_Acoustic37", "Steel_S250GD_Z275",
            "Gypsum_Standard", "Gypsum_Fiber", "WoodWool_Acoustic",
            "MineralWool_Roof38", "MineralWool_Facade34", "PE_VapourBarrier",
            "Gravel_16-32", "ClayTile_Black_Engobed", "RadonBarrier",
            "EPS_S250", "Underlay_Membrane",
            "Cavity_Ventilated", "Cavity_Unventilated",
            "Zinc_Titanium_EN988", "Steel_Stainless_304",
        }
        assert set(MATERIALS) == spec_keys

    def test_dk_module_self_test_passes(self, capsys):
        """The dk module's __main__ block is a runnable self-test."""
        runpy.run_module("lite_step.materials.dk", run_name="__main__")
        out = capsys.readouterr().out
        assert "OK — 28 materials validated at import" in out

    def test_glass_keys_carry_alpha(self):
        from lite_step.materials.dk import MATERIALS

        assert MATERIALS["Glass_VIG"].render.alpha == pytest.approx(0.30)
        assert MATERIALS["Glass_Monolithic"].render.alpha == pytest.approx(0.30)

    def test_gl28h_carries_an_8m_simply_supported_roof_beam(self):
        """Glulam_GL28h exists to span a roof with NO intermediate support.

        Pins the sizing envelope the material was added for: an 8 m clear
        span wants d ~ span/17..span/20 (450-495 mm) at a width that keeps
        it laterally stable, and the stock has to be long enough to reach
        both bearings in ONE piece (no splice mid-span on a simply
        supported beam).
        """
        from lite_step.materials.dk import MATERIALS

        gl = MATERIALS["Glulam_GL28h"]
        assert gl.function == "LoadBearing"
        assert gl.psets["Pset_MaterialWood"]["StrengthGrade"] == "GL28h"
        # One piece across an 8 m span, plus bearing.
        assert gl.geometry.max_length_mm >= 8000
        # The span/17..span/20 sections for 8 m are real stock…
        assert gl.geometry.has_profile(165, 450)
        assert gl.geometry.has_profile(140, 495)
        # …orientation-agnostic, like every member catalog.
        assert gl.geometry.has_profile(450, 165)
        # …and the deeper long-span sizes are available too.
        assert gl.geometry.has_profile(190, 585)
        # Shallow dimensional-lumber depths are NOT this material's job
        # (that's Timber_C24 / GL24h) — guards against the stock range
        # silently drifting down into rafter territory.
        assert not gl.geometry.has_profile(45, 195)


# ---------------------------------------------------------------------------
# Material construction validation matrix
# ---------------------------------------------------------------------------

class TestMaterialConstruction:
    def test_catalog_section_ok(self):
        Material(key="Timber_C24", profile_mm=(45, 195))

    def test_catalog_section_orientation_agnostic(self):
        Material(key="Timber_C24", profile_mm=(195, 45))

    def test_glulam_matrix_section_ok(self):
        Material(key="Glulam_GL24h", profile_mm=(115, 405))

    def test_stock_thickness_ok(self):
        Material(key="Glass_VIG", thickness_mm=8)

    def test_off_catalog_section_fails(self):
        with pytest.raises(ValueError, match="not in 'Timber_C24' stock catalog"):
            Material(key="Timber_C24", profile_mm=(50, 200))

    def test_off_stock_thickness_warns_non_fatal(self):
        with warnings_module.catch_warnings(record=True) as w:
            warnings_module.simplefilter("always")
            mat = Material(key="Chipboard_P6", thickness_mm=18)
        assert mat.thickness_mm == 18
        assert len(w) == 1
        assert "off-stock" in str(w[0].message)

    def test_dimension_on_wrong_form_fails(self):
        # sheet material cannot carry a member section
        with pytest.raises(ValueError, match="profile_mm is for members"):
            Material(key="Chipboard_P6", profile_mm=(45, 195))
        # member material cannot carry a thickness (linear product → profile_mm)
        with pytest.raises(ValueError, match="use profile_mm"):
            Material(key="Timber_C24", thickness_mm=22)
        # mass material takes no profile (that's for members)
        with pytest.raises(ValueError, match="profile_mm is for members"):
            Material(key="Concrete_C30-37", profile_mm=(45, 195))
        # …but a mass material CAN now carry thickness_mm — it's a layer leaf
        Material(key="Concrete_C30-37", thickness_mm=200)

    def test_unknown_key_with_dimension_fails(self):
        with pytest.raises(ValueError, match="unknown material 'NoSuchMaterial'"):
            Material(key="NoSuchMaterial", thickness_mm=22)
        with pytest.raises(ValueError, match="unknown material"):
            Material(key="NoSuchMaterial", profile_mm=(45, 195))

    def test_unknown_key_without_dimension_is_open_vocabulary(self):
        mat = Material(key="SomeVendorProduct")
        assert mat.key == "SomeVendorProduct"

    def test_material_is_frozen_and_strict(self):
        mat = Material(key="Timber_C24")
        with pytest.raises(Exception):
            mat.key = "other"
        with pytest.raises(Exception):
            Material(key="Timber_C24", surprise_field=1)


# ---------------------------------------------------------------------------
# Bare-string resolution + legacy carve-out
# ---------------------------------------------------------------------------

class TestResolution:
    def test_bare_registry_string_resolves_to_material(self):
        from lite_step.materials import resolve_material

        mat = resolve_material("Brick_Red_DK")
        assert isinstance(mat, Material)
        assert mat.key == "Brick_Red_DK"

    def test_material_instance_passes_through(self):
        from lite_step.materials import resolve_material

        mat = Material(key="Glass_VIG", thickness_mm=8)
        assert resolve_material(mat) is mat

    @pytest.mark.parametrize("legacy", ["Concrete", "Timber", "Steel", "Masonry", "Void", "#310344", "#fff"])
    def test_legacy_vocabulary_never_resolves(self, legacy):
        from lite_step.materials import resolve_material

        assert resolve_material(legacy) is None

    def test_none_is_sketch_stage(self):
        from lite_step.materials import resolve_material

        assert resolve_material(None) is None

    def test_unknown_string_resolves_open_vocabulary(self):
        from lite_step.materials import resolve_material

        mat = resolve_material("FancyVendorBoard")
        assert isinstance(mat, Material)
        assert mat.key == "FancyVendorBoard"


# ---------------------------------------------------------------------------
# validate_project: errors vs warnings split
# ---------------------------------------------------------------------------

def _box(**kwargs) -> "BimElement":
    return Box(start=Point(x=0, y=0, z=0), end=Point(x=1000, y=1000, z=1000), **kwargs)


def _sheet(**kwargs) -> "BimElement":
    return Extrude(contour=[Point(x=0, y=0, z=0), Point(x=1200, y=0, z=0),
                          Point(x=1200, y=0, z=1400), Point(x=0, y=0, z=1400)], **kwargs)


class TestValidationSplit:
    def test_report_shape(self):
        b = Project(name="ok")
        b.add(_box(id="s1"))
        report = validate_project_report(b)
        assert isinstance(report, ValidationReport)
        assert report.errors == []
        assert report.warnings == []

    def test_validate_project_returns_errors_only(self):
        """Backward-compat: the historical list[str] contract = fatal errors."""
        b = Project(name="warn-only")
        b.add(_box(id="s1", material="FancyVendorBoard"))
        report = validate_project_report(b)
        assert report.warnings and not report.errors
        assert validate_project(b) == []

    def test_unknown_key_is_warning_not_error(self):
        b = Project(name="warn")
        b.add(_box(id="s1", material="FancyVendorBoard"))
        report = validate_project_report(b)
        assert len(report.warnings) == 1
        assert "unknown material 'FancyVendorBoard'" in report.warnings[0]
        assert report.errors == []

    def test_legacy_vocabulary_stays_silent(self):
        b = Project(name="legacy")
        b.add(_box(id="s1", material="Concrete"))
        b.add(_box(id="s2", material="Timber"))
        b.add(_box(id="s3", material="Steel"))
        b.add(_box(id="s4", material="Masonry"))
        report = validate_project_report(b)
        assert report.errors == []
        assert report.warnings == []

    def test_registry_key_no_dimensions_is_clean(self):
        b = Project(name="clean")
        b.add(_box(id="s1", material="Brick_Red_DK"))
        report = validate_project_report(b)
        assert report.errors == []
        assert report.warnings == []

    def test_off_stock_thickness_warns_at_construction_not_fatal(self):
        with warnings_module.catch_warnings():
            warnings_module.simplefilter("ignore")
            mat = Material(key="Chipboard_P6", thickness_mm=18)
        b = Project(name="off-stock")
        b.add(_sheet(id="floor", material=mat))
        report = validate_project_report(b)
        assert report.errors == []

    def test_wall_children_are_validated(self):
        wall = Wall(id="w")
        wall.add(_box(id="body", material="FancyVendorBoard"))
        b = Project(name="children")
        b.add(wall)
        report = validate_project_report(b)
        assert any("FancyVendorBoard" in w for w in report.warnings)


class TestOneSourceRule:
    def test_sheet_material_thickness_plus_own_thickness_is_error(self):
        b = Project(name="double")
        b.add(_sheet(id="pane", thickness=6,
                     material=Material(key="Glass_VIG", thickness_mm=8)))
        report = validate_project_report(b)
        assert any("one-source rule" in e for e in report.errors)

    def test_sheet_material_thickness_alone_is_ok(self):
        b = Project(name="single")
        b.add(_sheet(id="pane", material=Material(key="Glass_VIG", thickness_mm=8)))
        report = validate_project_report(b)
        assert report.errors == []

    def test_thickness_material_on_box_solid_is_error(self):
        b = Project(name="box")
        b.add(_box(id="s", material=Material(key="Glass_VIG", thickness_mm=8)))
        report = validate_project_report(b)
        assert any("carries its own" in e for e in report.errors)

    def test_shape_kwarg_is_forbidden(self):
        # DSL v17: the legacy catalog shape= key was removed; passing it is a
        # forbidden extra field (member sections come from Material(profile_mm=)).
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Sweep(id="p", shape="Stud_2x4",
                  start=Point(x=0, y=0, z=0), end=Point(x=0, y=4000, z=0))

    def test_section_material_alone_on_profile_is_ok(self):
        b = Project(name="single-section")
        b.add(Sweep(id="p", material=Material(key="Timber_C24", profile_mm=(45, 195)),
                      start=Point(x=0, y=0, z=0), end=Point(x=0, y=4000, z=0)))
        report = validate_project_report(b)
        assert report.errors == []

    def test_legacy_form_without_material_falls_back_to_default_section(self):
        # DSL v17: with shape= gone, a legacy start/end Sweep that carries no
        # member Material profile is NOT an error — the generator falls back to
        # a standard 89x38 stud section (bim_catalog "Stud_2x4" default).
        b = Project(name="default-section")
        b.add(Sweep(id="p", start=Point(x=0, y=0, z=0), end=Point(x=0, y=4000, z=0)))
        report = validate_project_report(b)
        assert report.errors == []

    def test_section_material_on_solid_is_error(self):
        b = Project(name="wrong-primitive")
        b.add(_box(id="s", material=Material(key="Timber_C24", profile_mm=(45, 195))))
        report = validate_project_report(b)
        assert any("pairs with Sweep" in e for e in report.errors)

    def test_thickness_material_on_slab_is_error(self):
        # DSL v8: Slab is a container; a thickness_mm member material belongs
        # on the Extrude sheet child, not on the Slab container itself.
        b = Project(name="slab")
        b.add(Slab(id="sl", material=Material(key="Chipboard_P6", thickness_mm=22)))
        report = validate_project_report(b)
        assert any("carries its own" in e for e in report.errors)


# ---------------------------------------------------------------------------
# compile_main: warning does NOT raise, error does
# ---------------------------------------------------------------------------

class TestCompileMainSplit:
    def test_warning_compiles_and_prints_warning_line(self, tmp_path, capsys):
        from lite_step import compile_main

        b = Project(name="warns")
        b.add(_box(id="s1", material="FancyVendorBoard"))
        src = tmp_path / "model.py"
        src.write_text("# synthetic test model\n")
        out = compile_main(result=b, source_path=str(src))
        assert out is not None and out.exists()
        err = capsys.readouterr().err
        assert "warning:" in err
        assert "FancyVendorBoard" in err

    def test_error_raises_loud(self, tmp_path, capsys):
        from lite_step import LiteStepCompileError, compile_main

        b = Project(name="errors")
        b.add(_sheet(id="pane", thickness=6,
                     material=Material(key="Glass_VIG", thickness_mm=8)))
        src = tmp_path / "model.py"
        src.write_text("# synthetic test model\n")
        with pytest.raises(LiteStepCompileError) as exc_info:
            compile_main(result=b, source_path=str(src))
        assert "one-source rule" in str(exc_info.value.reasons)
        err = capsys.readouterr().err
        assert "error:" in err


# ---------------------------------------------------------------------------
# Generator emission (ifcopenshell backend)
# ---------------------------------------------------------------------------

def _generate(project: Project) -> str:
    from lite_step.compiler.executor import normalize_project_to_meters
    from lite_step.ifc.generator import generate_ifc

    normalized = normalize_project_to_meters(project)
    result = generate_ifc(normalized)
    assert result.success, result.error
    return result.ifc_content


class TestGeneratorEmission:
    def test_glass_vig_sheet_emits_material_rel_and_alpha_style(self):
        b = Project(name="vig")
        b.add(_sheet(id="pane", material=Material(key="Glass_VIG", thickness_mm=8)))
        content = _generate(b)

        assert "IFCMATERIAL('Glass_VIG'" in content
        assert "IFCRELASSOCIATESMATERIAL" in content
        # registry render color with alpha: 1 - 0.30 = 0.7 transparency
        assert "IFCSURFACESTYLERENDERING" in content
        assert ",0.7," in content
        # material-level Psets (product/EPD data)
        assert "IFCMATERIALPROPERTIES" in content
        assert "'DK_Product'" in content
        assert "'Ug_W_m2K'" in content

    def test_registry_member_profile_emits_material(self):
        b = Project(name="rafters")
        b.add(Sweep(id="r0", material=Material(key="Timber_C24", profile_mm=(45, 195)),
                      start=Point(x=0, y=0, z=0), end=Point(x=0, y=4000, z=0)))
        content = _generate(b)
        assert "IFCMATERIAL('Timber_C24'" in content
        assert "IFCRELASSOCIATESMATERIAL" in content
        assert "'Pset_MaterialWood'" in content

    def test_material_thickness_drives_sheet_extrusion_depth(self):
        import re

        b = Project(name="depth")
        b.add(_sheet(id="pane", material=Material(key="Glass_VIG", thickness_mm=8)))
        content = _generate(b)
        # 8 mm -> 0.008 m extrusion depth. Parse numerically — float
        # serialization differs across ifcopenshell versions (0.008 vs
        # 8.0000000000000002E-03).
        depths = [
            float(m.group(1))
            for m in re.finditer(r"IFCEXTRUDEDAREASOLID\(#\d+,#\d+,#\d+,([0-9.Ee+-]+)\);", content)
        ]
        assert depths, "no IfcExtrudedAreaSolid found"
        assert any(d == pytest.approx(0.008) for d in depths)

    def test_unknown_key_emits_plain_named_material_only(self):
        b = Project(name="plain")
        b.add(_box(id="s1", material="FancyVendorBoard"))
        content = _generate(b)
        assert "IFCMATERIAL('FancyVendorBoard'" in content
        assert "IFCRELASSOCIATESMATERIAL" in content
        assert "IFCMATERIALPROPERTIES" not in content

    def test_legacy_vocabulary_emits_no_ifcmaterial(self):
        b = Project(name="legacy")
        b.add(_box(id="s1", material="Concrete", type="sketch", color="wall"))
        content = _generate(b)
        assert "IFCMATERIAL(" not in content
        assert "IFCRELASSOCIATESMATERIAL" not in content

    def test_explicit_color_overrides_material_render(self):
        from lite_step.ifc.generator import ELEMENT_TYPE_COLORS

        b = Project(name="override")
        b.add(_sheet(id="pane", color="glass",
                     material=Material(key="Glass_VIG", thickness_mm=8)))
        content = _generate(b)
        # material + rel still emitted
        assert "IFCMATERIAL('Glass_VIG'" in content
        # but the surface style is the sketch glass palette (0.6 transparency),
        # not the registry render (0.7 transparency)
        assert ",0.6," in content
        assert ",0.7," not in content

    def test_material_rel_is_shared_across_products(self):
        b = Project(name="shared")
        b.add(_sheet(id="p1", material=Material(key="Glass_VIG", thickness_mm=8)))
        b.add(Extrude(id="p2", contour=[Point(x=0, y=500, z=0), Point(x=1200, y=500, z=0),
                                      Point(x=1200, y=500, z=1400), Point(x=0, y=500, z=1400)],
                    material=Material(key="Glass_VIG", thickness_mm=8)))
        content = _generate(b)
        assert content.count("IFCMATERIAL('Glass_VIG'") == 1
        assert content.count("IFCRELASSOCIATESMATERIAL") == 1
class TestGlassPalette:
    def test_glass_is_a_transparent_palette_entry(self):
        from lite_step.ifc.generator import ELEMENT_TYPE_COLORS, TRANSPARENT_TYPES

        assert "glass" in ELEMENT_TYPE_COLORS
        assert TRANSPARENT_TYPES.get("glass", 0.0) > 0.0

    def test_color_glass_renders_transparent_in_output(self):
        b = Project(name="sketch-glass")
        b.add(_sheet(id="pane", thickness=6, color="glass"))
        content = _generate(b)
        assert "IFCSURFACESTYLERENDERING" in content
        assert ",0.6," in content  # TRANSPARENT_TYPES["glass"]
        assert "IFCMATERIAL(" not in content  # sketch stage: no material data
