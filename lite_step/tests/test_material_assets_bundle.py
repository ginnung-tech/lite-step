"""Tests for Material texture mappings, the assets bundle, and packaging scripts."""

import struct
import zipfile
from pathlib import Path

import pytest

from lite_step.materials.dk import MATERIALS, MaterialDef

ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = ROOT / "lite_step" / "assets"
TEXTURES_DIR = ASSETS_DIR / "textures"


def _read_jpeg_size(p: Path) -> tuple[int, int]:
    """Read JPEG width and height using only the Python standard library."""
    data = p.read_bytes()
    assert data[:3] == b"\xff\xd8\xff", f"File {p} does not begin with JPEG SOI marker"
    i = 2
    while i < len(data) - 9:
        marker, = struct.unpack(">H", data[i:i + 2])
        i += 2
        # SOF markers: 0xFFC0 (baseline), 0xFFC1 (extended sequential), 0xFFC2 (progressive)
        if marker in (0xFFC0, 0xFFC1, 0xFFC2):
            _length, _bits, h, w = struct.unpack(">H B H H", data[i:i + 7])
            return w, h
        length, = struct.unpack(">H", data[i:i + 2])
        i += length
    raise ValueError(f"Could not find SOF marker in JPEG {p}")


class TestMaterialTextureMapping:
    def test_materialdef_has_texture_field(self):
        """MaterialDef supports optional texture string field."""
        m = MaterialDef(
            category="masonry",
            psets={"Pset_MaterialCommon": {"MassDensity": 1800}},
            render={"rgb": "#9E3B2B", "alpha": 1.0, "finish": "matt"},
            geometry={"form": "mass"},
            texture="test_texture.jpg",
        )
        assert m.texture == "test_texture.jpg"

    def test_dk_materials_carry_expected_polyhaven_textures(self):
        """Key materials in dk registry carry Poly Haven texture filenames."""
        expected = {
            "Brick_Red_DK": "red_brick_diff_1k.jpg",
            "Timber_C24": "wooden_floor_02_diff_1k.jpg",
            "Glulam_GL24h": "coated_pine_02_diff_1k.jpg",
            "Glulam_GL28h": "coated_pine_02_diff_1k.jpg",
            "Concrete_C30-37": "gravel_concrete_diff_1k.jpg",
            "Concrete_C25-30": "gravel_concrete_diff_1k.jpg",
            "ClayTile_Black_Engobed": "worn_tile_floor_diff_1k.jpg",
            "Gravel_16-32": "clean_pebbles_diff_1k.jpg",
            "Steel_B500B": "rusty_metal_04_diff_1k.jpg",
            "Steel_S250GD_Z275": "blue_metal_plate_diff_1k.jpg",
            "Zinc_Titanium_EN988": "blue_metal_plate_diff_1k.jpg",
            "Steel_Stainless_304": "blue_metal_plate_diff_1k.jpg",
        }
        for key, expected_tex in expected.items():
            assert key in MATERIALS, f"{key} missing from MATERIALS"
            m = MATERIALS[key]
            assert m.texture == expected_tex, f"{key}.texture was {m.texture!r}, expected {expected_tex!r}"

    def test_referenced_textures_exist_and_are_valid_images_without_pil(self):
        """All textures referenced in MATERIALS exist, are lowercase, and have valid 1024x1024 resolution."""
        referenced = {m.texture for m in MATERIALS.values() if m.texture is not None}
        assert len(referenced) > 0

        for tex_name in referenced:
            tex_path = TEXTURES_DIR / tex_name
            assert tex_path.exists(), f"Texture file {tex_name} missing from {TEXTURES_DIR}"
            assert tex_name == tex_name.lower(), f"Texture filename {tex_name} is not lowercase"
            assert tex_path.stat().st_size > 10_000, f"Texture {tex_name} suspiciously small"
            w, h = _read_jpeg_size(tex_path)
            assert w >= 1024 and h >= 1024, f"Texture {tex_name} resolution {w}x{h} below 1k"

    def test_texture_provenance_notice_documents_all_textures(self):
        """Verify textures/NOTICE.md exists and documents all active textures."""
        notice_path = TEXTURES_DIR / "NOTICE.md"
        assert notice_path.exists(), "NOTICE.md missing from textures directory"
        content = notice_path.read_text(encoding="utf-8")
        assert "Poly Haven" in content
        assert "CC0" in content

        referenced = {m.texture for m in MATERIALS.values() if m.texture is not None}
        for tex_name in referenced:
            assert tex_name in content, f"Texture {tex_name} not documented in NOTICE.md"


class TestAssetsBundleTriad:
    def test_triad_files_and_stylesheet_wiring(self):
        """Verify model.ifc, model.blend, and styles/lite-step.css exist and match IFC wiring."""
        ifc_file = ASSETS_DIR / "model.ifc"
        blend_file = ASSETS_DIR / "model.blend"
        css_file = ASSETS_DIR / "styles" / "lite-step.css"

        assert ifc_file.exists(), "model.ifc missing from assets"
        assert blend_file.exists(), "model.blend missing from assets"
        assert css_file.exists(), "styles/lite-step.css missing from assets"
        assert not (ASSETS_DIR / "model.css").exists(), "model.css at root should be removed in favor of styles/"

        assert ifc_file.stat().st_size > 100_000
        assert blend_file.stat().st_size > 10_000
        assert css_file.stat().st_size > 100

        # Verify IFC declares StylesheetPath pointing to styles/lite-step.css
        ifc_text = ifc_file.read_text(encoding="utf-8", errors="ignore")
        assert "styles/lite-step.css" in ifc_text

    def test_build_blend_scene_consumes_materials_texture(self):
        """Verify build_blend_scene module imports MATERIALS and resolves paths relative to __file__."""
        from lite_step.tests.tools.build_blend_scene import get_material_pbr_config

        # Check Brick_Red_DK mapping
        brick_cfg = get_material_pbr_config("Brick_Red_DK", MATERIALS["Brick_Red_DK"])
        assert brick_cfg["texture"] == "red_brick_diff_1k.jpg"
        assert brick_cfg["bump"] > 0

        # Check Timber_C24 mapping
        timber_cfg = get_material_pbr_config("Timber_C24", MATERIALS["Timber_C24"])
        assert timber_cfg["texture"] == "wooden_floor_02_diff_1k.jpg"

    def test_package_assets_script_creates_valid_archives(self, tmp_path):
        """Verify package_assets creates non-empty zip archives containing expected entries."""
        from lite_step.tests.tools.package_assets import create_bundle_zip, create_ifczip

        test_ifczip = tmp_path / "test.ifczip"
        test_bundle = tmp_path / "test_bundle.zip"

        create_ifczip(ASSETS_DIR / "model.ifc", test_ifczip)
        assert test_ifczip.exists() and test_ifczip.stat().st_size > 10_000
        with zipfile.ZipFile(test_ifczip, "r") as zf:
            assert zf.namelist() == ["model.ifc"]
            assert zf.getinfo("model.ifc").file_size > 100_000

        create_bundle_zip(ASSETS_DIR, test_bundle)
        assert test_bundle.exists() and test_bundle.stat().st_size > 500_000
        with zipfile.ZipFile(test_bundle, "r") as zf:
            names = set(zf.namelist())
            assert "model.ifc" in names
            assert "model.blend" in names
            assert "styles/lite-step.css" in names
            assert "NOTICE.md" in names
            assert "textures/NOTICE.md" in names
            assert "textures/red_brick_diff_1k.jpg" in names
            for info in zf.infolist():
                assert info.file_size > 0, f"Entry {info.filename} has 0 bytes"

    def test_package_assets_fails_loudly_on_missing_files(self, tmp_path):
        """Verify packaging fails with FileNotFoundError if any required file is missing."""
        from lite_step.tests.tools.package_assets import create_bundle_zip, create_ifczip

        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()

        # Missing IFC in create_ifczip
        with pytest.raises(FileNotFoundError):
            create_ifczip(empty_dir / "missing.ifc", tmp_path / "out.ifczip")

        # Missing files in create_bundle_zip
        with pytest.raises(FileNotFoundError):
            create_bundle_zip(empty_dir, tmp_path / "out_bundle.zip")
