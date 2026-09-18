"""Tests for ``lite_step.compiler.executor.compile_main``.

Phase 3 of the ``compile_main`` rollout: ``result`` and ``source_path``
became optional — the helper grabs them from the caller's stack frame.
The canonical model-file tail is now the 2-line zero-arg form::

    if __name__ == "__main__":
        compile_main()

The eleven tests below cover the contract:

1.  happy path writes ``output.ifc`` next to the source (explicit args)
2.  validation warning -> ``None`` + stderr
3.  IFC generation failure -> ``None`` + stderr
4.  relative ``source_path`` is resolved before deriving output
5.  non-Project input raises ``TypeError`` with a helpful message
6.  ``output_name="…"`` overrides the filename next to source
7.  ``verbose=True`` adds ``Project: …`` / ``Elements: …`` to stdout
8.  absolute ``output_name`` is honored as-is (not joined to source dir)
9.  zero-arg call grabs ``result`` and ``__file__`` from caller's globals
10. zero-arg call without ``result`` in globals raises ``TypeError``
11. explicit args override caller-globals defaults
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.compiler.executor import compile_main
from lite_step.ifc.generator import IFCGenerationResult
from lite_step.models import Project, Point, Box, Extrude, Storey


REPO_ROOT = Path(__file__).resolve().parents[2]


def _minimal_building() -> Project:
    """A Project with one storey and one Solid block.

    Smallest shape that passes ``validate_project`` and exercises
    ``generate_ifc`` end-to-end.
    """
    proj = Project(name="test")
    storey = Storey(elevation=0)
    proj.add_storey(storey)
    storey.add(
        Box(
            id="blk",
            start=Point(x=0, y=0, z=0),
            end=Point(x=1000, y=1000, z=1000),
            type="sketch",
        )
    )
    return proj


def test_happy_path_writes_ifc(tmp_path):
    proj = _minimal_building()
    source = tmp_path / "output.py"  # file does not need to exist

    out = compile_main(proj, str(source))

    assert out == tmp_path / "output.ifc"
    assert out.exists()
    assert out.stat().st_size > 0
    contents = out.read_text()
    assert contents.startswith("ISO-10303-21;")


def test_validation_error_raises_loud(tmp_path, capsys):
    # Wall without a body Solid -> validate_project emits an error message.
    # compile_main must FAIL LOUD (raise LiteStepCompileError), not silently
    # return None — the canonical `compile_main()` tail ignores the return, so
    # a None let `python output.py` exit 0 with no IFC (the LOD 350 phase
    # cube-build hang). No output.ifc is written; the reasons are on stderr.
    from lite_step.models import Wall
    from lite_step import LiteStepCompileError

    proj = Project(name="bad")
    storey = Storey(elevation=0)
    proj.add_storey(storey)
    storey.add(Wall(id="bad_wall"))

    source = tmp_path / "output.py"

    with pytest.raises(LiteStepCompileError) as exc_info:
        compile_main(proj, str(source))

    assert exc_info.value.reasons  # carries the structured validation messages
    assert not (tmp_path / "output.ifc").exists()
    captured = capsys.readouterr()
    assert "error:" in captured.err


def test_ifc_generation_failure_raises_loud(tmp_path, capsys, monkeypatch):
    from lite_step import LiteStepCompileError

    proj = _minimal_building()
    source = tmp_path / "output.py"

    def _fake_generate_ifc(*_args, **_kwargs):
        return IFCGenerationResult(
            success=False,
            ifc_content=None,
            stats={},
            error="forced failure",
        )

    # compile_main does ``from lite_step.ifc.generator import generate_ifc``
    # at call time, so patch the module attribute the import resolves to.
    monkeypatch.setattr(
        "lite_step.ifc.generator.generate_ifc",
        _fake_generate_ifc,
    )

    # LOUD, not a silent None (same rationale as the validation-error test).
    with pytest.raises(LiteStepCompileError):
        compile_main(proj, str(source))

    assert not (tmp_path / "output.ifc").exists()
    captured = capsys.readouterr()
    assert "IFC generation failed: forced failure" in captured.err


def test_resolves_relative_source_path(tmp_path, monkeypatch):
    sub = tmp_path / "subdir"
    sub.mkdir()
    monkeypatch.chdir(tmp_path)

    proj = _minimal_building()

    out = compile_main(proj, "./subdir/output.py")

    assert out is not None
    assert out.is_absolute()
    assert out == (sub / "output.ifc").resolve()
    assert out.exists()


def test_non_building_raises_type_error():
    with pytest.raises(TypeError) as exc_info:
        compile_main({"not": "a project"}, "/tmp/output.py")  # type: ignore[arg-type]

    msg = str(exc_info.value)
    assert "Project" in msg
    assert "dict" in msg


def test_custom_output_name(tmp_path):
    proj = _minimal_building()
    source = tmp_path / "output.py"

    out = compile_main(proj, str(source), output_name="bodiam.ifc")

    assert out == tmp_path / "bodiam.ifc"
    assert out.exists()
    assert out.stat().st_size > 0
    # The default name is NOT written when override is supplied.
    assert not (tmp_path / "output.ifc").exists()


def test_verbose_prints_extra_diagnostics(tmp_path, capsys):
    proj = _minimal_building()
    source = tmp_path / "output.py"

    out = compile_main(proj, str(source), verbose=True)

    assert out == tmp_path / "output.ifc"
    captured = capsys.readouterr()
    # Both extra debug lines appear on stdout, in this order, before the
    # canonical wrote-line.
    stdout = captured.out
    assert "Project: test" in stdout
    assert "Elements: " in stdout
    assert stdout.index("Project:") < stdout.index("Elements:")
    assert stdout.index("Elements:") < stdout.index("wrote ")


def test_absolute_output_name_respected(tmp_path):
    proj = _minimal_building()
    source = tmp_path / "output.py"
    target = tmp_path / "subdir" / "thing.ifc"

    out = compile_main(proj, str(source), output_name=str(target))

    assert out == target
    assert out.exists()
    # Source-dir default should NOT be written when an absolute path is
    # supplied — confirms we don't accidentally write next to source.
    assert not (tmp_path / "output.ifc").exists()
    assert not (tmp_path / "thing.ifc").exists()


# ----------------------------------------------------------------------
# Zero-arg / stack-frame default tests
# ----------------------------------------------------------------------
#
# These run ``python <script>`` in a subprocess so the
# ``inspect.stack()[1]`` lookup sees a real script frame (not the
# pytest harness's frame). REPO_ROOT is added to PYTHONPATH so the
# child can ``from lite_step import compile_main``.


def _run_subprocess_script(
    script_path: Path, extra_env: dict | None = None
) -> subprocess.CompletedProcess:
    """Exec ``python script_path`` with REPO_ROOT on PYTHONPATH.

    ``extra_env`` is merged on top of the inherited environment so a
    test can set things like ``LITESTEP_BASE_IFC`` for one invocation
    without leaking into the parent test process.
    """
    import os

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_zero_arg_grabs_result_and_file_from_caller(tmp_path):
    script = tmp_path / "output.py"
    script.write_text(
        textwrap.dedent(
            """
            from lite_step import compile_main
            from lite_step.models import Project, Point, Box, Extrude, Storey

            proj = Project(name="zero_arg")
            storey = Storey(elevation=0)
            proj.add_storey(storey)
            storey.add(
                Box(
                    id="blk",
                    start=Point(x=0, y=0, z=0),
                    end=Point(x=1000, y=1000, z=1000),
                    type="sketch",
                )
            )
            result = proj

            if __name__ == "__main__":
                compile_main()
            """
        ).lstrip()
    )

    proc = _run_subprocess_script(script)

    assert proc.returncode == 0, (
        f"script failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    out_ifc = tmp_path / "output.ifc"
    assert out_ifc.exists()
    assert out_ifc.stat().st_size > 0
    assert out_ifc.read_text().startswith("ISO-10303-21;")


def test_zero_arg_raises_when_no_result_in_globals(tmp_path):
    script = tmp_path / "output.py"
    script.write_text(
        textwrap.dedent(
            """
            from lite_step import compile_main

            if __name__ == "__main__":
                compile_main()
            """
        ).lstrip()
    )

    proc = _run_subprocess_script(script)

    assert proc.returncode != 0
    assert "expects 'result' at module scope" in proc.stderr


def test_explicit_args_override_caller_globals(tmp_path):
    script = tmp_path / "output.py"
    # `result` in module globals is "from_globals"; explicit kwarg
    # passes a different project named "from_arg". The output IFC
    # should reflect the explicit arg, not the global.
    script.write_text(
        textwrap.dedent(
            """
            from lite_step import compile_main
            from lite_step.models import Project, Point, Box, Extrude, Storey


            def _make(name):
                proj = Project(name=name)
                storey = Storey(elevation=0)
                proj.add_storey(storey)
                storey.add(
                    Box(
                        id="blk",
                        start=Point(x=0, y=0, z=0),
                        end=Point(x=1000, y=1000, z=1000),
                        type="sketch",
                    )
                )
                return proj


            result = _make("from_globals")

            if __name__ == "__main__":
                compile_main(result=_make("from_arg"), source_path=__file__)
            """
        ).lstrip()
    )

    proc = _run_subprocess_script(script)

    assert proc.returncode == 0, (
        f"script failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    out_ifc = tmp_path / "output.ifc"
    assert out_ifc.exists()
    contents = out_ifc.read_text()
    # IFC stores the project name; the explicit-arg project must win.
    assert "from_arg" in contents
    assert "from_globals" not in contents


def _make_minimal_model_script(
    tmp_path: Path,
    name: str = "patch_demo",
    base_ifc: Optional[str] = None,
) -> Path:
    """Write a minimal compilable output.py under tmp_path.

    When ``base_ifc`` is provided, it lands as a module-global at the
    top of the script — the canonical patch-mode declaration shape.
    """
    base_decl = f'base_ifc = "{base_ifc}"\n\n' if base_ifc else ""
    script = tmp_path / "output.py"
    script.write_text(
        textwrap.dedent(
            f"""
            from lite_step import compile_main
            from lite_step.models import Project, Point, Box, Extrude, Storey

            {base_decl.rstrip()}
            proj = Project(name="{name}")
            storey = Storey(elevation=0)
            proj.add_storey(storey)
            storey.add(
                Box(
                    start=Point(x=0, y=0, z=0),
                    end=Point(x=1000, y=200, z=2000),
                    type="sketch",
                )
            )
            result = proj

            if __name__ == "__main__":
                compile_main()
            """
        ).lstrip()
    )
    return script


def test_base_ifc_global_marks_output_cold_imported(tmp_path):
    """Declaring ``base_ifc = "./base.ifc"`` at module scope makes
    compile_main patch against the adjacent IFC. When the base lacks
    LITESTEP_META, the output is marked cold_imported=true so the
    patch UI keeps the lossiness banner visible."""
    pytest.importorskip("ifcopenshell")
    import ifcopenshell as _ifc

    # Bare base IFC with no LITESTEP_META, sibling of the script
    base_path = tmp_path / "base.ifc"
    model = _ifc.file(schema="IFC4")
    model.create_entity("IfcProject", GlobalId=_ifc.guid.new(), Name="Base")
    base_path.write_text(model.to_string())

    script = _make_minimal_model_script(tmp_path, base_ifc="./base.ifc")
    proc = _run_subprocess_script(script)
    assert proc.returncode == 0, (
        f"patch script failed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "[patch base=base.ifc]" in proc.stdout

    out_ifc = tmp_path / "output.ifc"
    assert out_ifc.exists()
    from lite_step.ifc.embedder import extract_litestep_meta
    meta = extract_litestep_meta(out_ifc.read_text())
    assert meta is not None, "compile_main must embed LITESTEP_META unconditionally"
    assert meta.cold_imported is True


def test_base_ifc_missing_file_warns_and_falls_back(tmp_path):
    """When base_ifc points at a non-existent path, compile_main
    warns to stderr and patches against an empty base — equivalent to
    plain generate mode but with META still embedded."""
    script = _make_minimal_model_script(tmp_path, base_ifc="./does-not-exist.ifc")
    proc = _run_subprocess_script(script)
    assert proc.returncode == 0
    assert "not found" in proc.stderr.lower()
    out_ifc = tmp_path / "output.ifc"
    assert out_ifc.exists()
    # No patch suffix in stdout since the base wasn't loaded
    assert "[patch base=" not in proc.stdout


def _stamped_base_ifc(version: str, monkeypatch) -> str:
    """A LITESTEP base IFC carrying an arbitrary META ``version``.

    ``compile_main``'s gate reads the version out of the BASE FILE, not
    the module constant, so the constant is restored before the returned
    bytes are used.
    """
    import lite_step.ifc.embedder as emb
    from lite_step.compiler.executor import execute_lite_step_script
    from lite_step.ifc.generator import generate_ifc

    src = "result = Project(name='Base')\n"
    executed = execute_lite_step_script(src)
    assert executed.success, executed.error

    monkeypatch.setattr(emb, "LITESTEP_META_VERSION", version)
    generated = generate_ifc(executed.project, source_code=src)
    assert generated.success, generated.error
    monkeypatch.undo()

    from lite_step.ifc.embedder import extract_litestep_meta
    assert extract_litestep_meta(generated.ifc_content).version == version
    return generated.ifc_content


@pytest.mark.parametrize("version", ["17", "19"])
def test_base_ifc_from_a_foreign_version_is_refused(tmp_path, monkeypatch, version):
    """``compile_main`` refuses to continue a base it did not write.

    Both directions matter: "17" is a version below the floor (v18), "19" is a
    base written by a compiler NEWER than this one — which no enumeration of
    known-bad versions can ever contain. Both move with every
    ``LITESTEP_META_VERSION`` bump; a pair left behind stops testing the
    boundary and starts testing two arbitrary old numbers.
    """
    (tmp_path / "base.ifc").write_text(_stamped_base_ifc(version, monkeypatch))
    script = _make_minimal_model_script(tmp_path, base_ifc="./base.ifc")

    proc = _run_subprocess_script(script)

    assert proc.returncode != 0, (
        f"a version-{version} base was accepted: stdout={proc.stdout!r}"
    )
    assert "different version of Lite-STEP" in proc.stderr
    assert f"version {version}" in proc.stderr
    assert "start a fresh build" in proc.stderr
    assert not (tmp_path / "output.ifc").exists(), (
        "a refused base must not leave a half-written output.ifc behind"
    )


def test_base_ifc_at_the_current_version_continues_fine(tmp_path, monkeypatch):
    """The allowlist's one accepted case — a base this compiler wrote."""
    import lite_step.ifc.embedder as emb

    (tmp_path / "base.ifc").write_text(
        _stamped_base_ifc(emb.LITESTEP_META_VERSION, monkeypatch)
    )
    script = _make_minimal_model_script(tmp_path, base_ifc="./base.ifc")

    proc = _run_subprocess_script(script)

    assert proc.returncode == 0, (
        f"current-version base was refused: stderr={proc.stderr!r}"
    )
    out_ifc = tmp_path / "output.ifc"
    assert out_ifc.exists()
    from lite_step.ifc.embedder import extract_litestep_meta
    meta = extract_litestep_meta(out_ifc.read_text())
    # Round-trip, not cold-import: the base carried our META.
    assert meta is not None and meta.cold_imported is False


def test_no_base_ifc_global_still_embeds_meta(tmp_path):
    """When base_ifc is absent (plain generate flow), the output IFC
    still carries embedded LITESTEP_META so a subsequent edit on the
    same file can take the fast hot path. Unified-mode invariant:
    every compile_main output is round-tripable."""
    pytest.importorskip("ifcopenshell")
    script = _make_minimal_model_script(tmp_path)
    proc = _run_subprocess_script(script)
    assert proc.returncode == 0, f"script failed: stderr={proc.stderr!r}"
    out_ifc = tmp_path / "output.ifc"
    assert out_ifc.exists()
    from lite_step.ifc.embedder import extract_litestep_meta
    meta = extract_litestep_meta(out_ifc.read_text())
    assert meta is not None, "unified mode embeds META even with no base_ifc"
    assert meta.cold_imported is False
