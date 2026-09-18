"""Pin: the pre-v8 legacy element strata stay removed.

Member/Block/Plate/ModellingVolume/ExtrudedModellingVolume/Opening/Asset/
ContourExtrudedBase were deleted in the v6.0.0 breaking release. The DSL v8
cutover (v8.0.0) then removed the v1.5 vocabulary too — Solid/Profile/Rod/Turn
and the Volumetric/Linear aliases — replaced by Box/Extrude/Sweep/Pipe/Revolve.
All must be importable from NOWHERE and invisible to the LLM sandbox namespace.
"""

import importlib

import pytest

LEGACY_CLASS_NAMES = [
    "Member",
    "Block",
    "Plate",
    "ModellingVolume",
    "ExtrudedModellingVolume",
    "Opening",
    "Asset",
    "ContourExtrudedBase",
    # v1.5 vocabulary + aliases, removed at the DSL v8 cutover
    "Solid",
    "Profile",
    "Rod",
    "Turn",
    "Volumetric",
    "Linear",
]

LEGACY_CATALOG_NAMES = [
    "PlateInfo",
    "PLATES",
    "get_plate",
    "list_plates",
    "AssetInfo",
    "ASSETS",
    "get_asset",
    # bim_catalog itself was retired in DSL v17 (shape= removed; member
    # sections come from Material(profile_mm=)). The whole module and its
    # profile/material catalog are gone.
    "CATALOG",
    "PROFILES",
    "MATERIALS",
    "ProfileInfo",
    "MaterialInfo",
    "get_profile",
    "get_material",
    "get_catalog_prompt",
    "list_profiles",
    "list_materials",
]


class TestLegacyClassesUnimportable:
    @pytest.mark.parametrize("name", LEGACY_CLASS_NAMES)
    def test_not_in_models_package(self, name):
        with pytest.raises(ImportError):
            exec(f"from lite_step.models import {name}")

    @pytest.mark.parametrize("name", LEGACY_CLASS_NAMES)
    def test_not_in_elements_module(self, name):
        with pytest.raises(ImportError):
            exec(f"from lite_step.models.elements import {name}")

    @pytest.mark.parametrize("name", LEGACY_CLASS_NAMES)
    def test_not_in_models_all(self, name):
        models = importlib.import_module("lite_step.models")
        assert name not in models.__all__

    @pytest.mark.parametrize("name", LEGACY_CATALOG_NAMES)
    def test_not_in_bim_catalog(self, name):
        # bim_catalog was deleted in DSL v17 — the import raises
        # ModuleNotFoundError (a subclass of ImportError).
        with pytest.raises(ImportError):
            exec(f"from lite_step.context.bim_catalog import {name}")

    def test_bim_catalog_module_is_gone(self):
        with pytest.raises(ImportError):
            importlib.import_module("lite_step.context.bim_catalog")


class TestSandboxNamespaceExposesNoLegacy:
    def test_no_legacy_names_in_namespace(self):
        from lite_step.compiler.namespace import create_namespace

        ns = create_namespace()
        exposed = [
            n for n in LEGACY_CLASS_NAMES + LEGACY_CATALOG_NAMES if n in ns
        ]
        assert exposed == [], f"legacy names leaked into the sandbox: {exposed}"

    def test_current_vocabulary_is_exposed(self):
        """DSL v8 vocabulary (Box/Extrude/Sweep/Pipe/Revolve) is exposed; the
        removed Solid/Profile/Volumetric/Linear names are NOT."""
        from lite_step.compiler.namespace import create_namespace

        ns = create_namespace()
        for name in ("Box", "Extrude", "Sweep", "Pipe", "Revolve"):
            assert name in ns, f"expected current primitive {name} in sandbox"
        for name in ("Solid", "Profile", "Rod", "Turn", "Volumetric", "Linear"):
            assert name not in ns, f"removed name {name} leaked into the sandbox"

    def test_catalog_names_no_longer_exposed(self):
        """The bim_catalog profile/material catalog was retired in DSL v17 —
        CATALOG/PROFILES/MATERIALS/get_profile are gone from the sandbox.
        Member sections come from Material(profile_mm=)."""
        from lite_step.compiler.namespace import create_namespace

        ns = create_namespace()
        for name in ("CATALOG", "PROFILES", "MATERIALS",
                     "get_profile", "get_material",
                     "list_profiles", "list_materials"):
            assert name not in ns, f"retired catalog name {name} leaked into the sandbox"
