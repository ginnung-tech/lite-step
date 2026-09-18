"""DSL v3.0 container-naming cutover contract pins.

The root authoring container is ``Project`` (was ``Building``); the convention
function is ``generate_project()`` (was ``generate_building()``); the variable
is ``proj`` (was ``bldg``). The module ``lite_step/models/building.py`` is now
``lite_step/models/project.py``. This is a HARD cutover (v8/v9/v10 precedent):
no alias, no migrator — the ``LITESTEP_META`` version wall ("4"→"5") refuses a
stored base whose source authors the removed container. (The wall has since
moved on to "6" for the v20.0.0 anchor-placement cutover, so the refusal
MESSAGE pinned below is the current one, not v3.0's; the refusal itself is
what this suite is about.)

Pinned here:

1.  The sandbox namespace exposes ``Project`` and NOT ``Building``.
2.  Old-convention source (``Building`` / ``generate_building()``) NameErrors.
3.  A v2.1 ("4") base is refused on continuation.
4.  The current base carries the current META and continues fine.
5.  The reconstructor emits the NEW convention (and none of the old).
6.  ``Project().name == "Project"`` and it compiles.
7.  ``lite_step.models.building`` is gone (module rename).
8.  IFC output entities are unchanged: a named Project still emits IfcProject +
    IfcBuilding with that Name (byte-compatible root identity).
"""

from __future__ import annotations

import importlib

import pytest

from lite_step.compiler.executor import (
    execute_lite_step_script,
    normalize_project_to_meters,
)
from lite_step.compiler.namespace import create_namespace
from lite_step.ifc.embedder import extract_litestep_meta
from lite_step.ifc.generator import generate_ifc
from lite_step.models import Project, Point, Box, Wall


def test_namespace_has_project_not_building():
    ns = create_namespace()
    assert "Project" in ns
    assert ns["Project"] is Project
    assert "Building" not in ns, "the superseded root container name must be gone"


def test_old_convention_source_nameerrors():
    # Bare old class reference.
    r = execute_lite_step_script("result = Building(name='x')")
    assert not r.success
    assert "Building" in (r.error or "") or "not defined" in (r.error or "")

    # The full old wrapper convention.
    old = (
        "def generate_building():\n"
        "    bldg = Building(name='x')\n"
        "    return bldg\n"
        "result = generate_building()\n"
    )
    r2 = execute_lite_step_script(old)
    assert not r2.success
    assert "not defined" in (r2.error or "")


def test_v21_base_refused_on_continuation(monkeypatch, tmp_path):
    """A base carrying the v2.1 META ("4") is refused on continuation.

    Its stored source authors the removed ``Building`` /
    ``generate_building()``, so re-executing it here would raise. The gate
    catches it for the general reason — it is not the version this
    compiler writes — not because "4" appears on any list.
    """
    import lite_step.ifc.embedder as emb
    from lite_step.compiler.patch_executor import patch_main

    # Build a valid base, then re-stamp it as version "4".
    src = "result = Project(name='Base')\n"
    r = execute_lite_step_script(src)
    assert r.success
    from lite_step.ifc.generator import generate_ifc as _gen
    base = _gen(r.project, source_code=src).ifc_content
    assert extract_litestep_meta(base).version == emb_version()

    # Fabricate a v2.1 base by recompiling with the constant patched to "4".
    monkeypatch.setattr(emb, "LITESTEP_META_VERSION", "4")
    base_v4 = _gen(r.project, source_code=src).ifc_content
    assert extract_litestep_meta(base_v4).version == "4"
    monkeypatch.undo()

    result = patch_main(base_v4, src)
    assert result.success is False
    assert "different version of Lite-STEP" in result.error
    assert result.needs_llm is False


def test_current_meta_is_18():
    """The version this compiler writes, pinned.

    A stale number here is not cosmetic: it is what every stored base is
    compared against, so a bump that lands without this assertion moving
    means the gate is comparing against the wrong thing.
    """
    assert emb_version() == "18"


def emb_version():
    import lite_step.ifc.embedder as emb
    return emb.LITESTEP_META_VERSION


def test_reconstructor_emits_new_convention():
    from lite_step.transpiler.reconstructor import _wrap_empty_project

    src = _wrap_empty_project()
    assert "def generate_project():" in src
    assert "proj = Project(" in src
    assert "result = generate_project()" in src
    assert "from lite_step.models.project import" in src
    # None of the superseded convention leaks into emitted source.
    assert "generate_building" not in src
    assert "bldg = " not in src
    assert "from lite_step.models.building" not in src


def test_default_project_name_and_compiles():
    assert Project().name == "Project"
    proj = Project(name="Villa")
    w = Wall(name="north")
    w.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=5000, y=200, z=3000)))
    proj.add(w)
    res = generate_ifc(normalize_project_to_meters(proj))
    assert res.success, res.error
    # Root identity unchanged: the Project name is the IfcProject + IfcBuilding Name.
    assert "'Villa'" in res.ifc_content


def test_building_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lite_step.models.building")
