"""Tests for the ``lite-step`` / ``python -m lite_step`` CLI.

The CLI exists so a developer can compile any output file from any cwd,
rather than having to remember which directory ``python output.py`` needs
to be invoked from. These tests pin that contract.

ON ``PYTHONPATH``
=================

The repo ships no package: it is consumed by
``pip install -r requirements.txt`` and imported from the checkout, so
there is no installed ``lite_step`` to find, and the subprocess tests
below have to put the repo root on the path themselves.

That was not caught for a while because a developer machine that had ever
installed the package (or a sibling checkout of it) resolves ``-m
lite_step`` from *that*, so the test passed locally while asserting
nothing about this repo. On a clean runner it failed immediately with
``No module named lite_step``. Passing ``PYTHONPATH`` explicitly makes
the test measure this checkout on every machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

#: Repo root — ``lite_step/tests/test_cli.py`` -> up two.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_with_repo_on_path() -> dict[str, str]:
    """A child env that can import ``lite_step`` from THIS checkout.

    Prepended, not assigned, so an inherited ``PYTHONPATH`` still works —
    and put first so a stale installed copy elsewhere on the path cannot
    win and make the test measure the wrong tree.
    """
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}{os.pathsep}{existing}" if existing else str(_REPO_ROOT)
    )
    return env

ifcopenshell = pytest.importorskip("ifcopenshell")

from lite_step.__main__ import main


CANONICAL_MODEL = textwrap.dedent(
    """
    from lite_step import compile_main
    from lite_step.models import Project, Point, Box, Extrude

    I = lambda x: int(round(x))  # noqa: E731

    def generate_project() -> Project:
        proj = Project(name="cli_test")
        proj.add(Box(
            id="room",
            start=Point(x=0, y=0, z=0),
            end=Point(x=4000, y=3000, z=2400),
            type="sketch",
        ))
        return proj

    result = generate_project()

    if __name__ == "__main__":
        compile_main()
    """
).strip()


def _write_model(tmp_path: Path, body: str = CANONICAL_MODEL) -> Path:
    model = tmp_path / "output.py"
    model.write_text(body, encoding="utf-8")
    return model


def test_cli_writes_ifc_next_to_source(tmp_path: Path) -> None:
    model = _write_model(tmp_path)
    rc = main([str(model)])
    assert rc == 0
    assert (tmp_path / "output.ifc").is_file()
    assert (tmp_path / "output.ifc").stat().st_size > 0


def test_cli_output_name_override(tmp_path: Path) -> None:
    model = _write_model(tmp_path)
    rc = main([str(model), "--output-name", "named.ifc"])
    assert rc == 0
    assert (tmp_path / "named.ifc").is_file()
    assert not (tmp_path / "output.ifc").exists()


def test_cli_missing_result_returns_nonzero(tmp_path: Path) -> None:
    # Module-scope ``result`` is the contract; without it, the CLI must
    # exit non-zero with a clear stderr message rather than crashing
    # inside compile_main's stack-frame inspector.
    model = tmp_path / "no_result.py"
    model.write_text(
        "from lite_step.models import Project\n"
        "proj = Project(name='oops')\n",
        encoding="utf-8",
    )
    rc = main([str(model)])
    assert rc == 2


def test_cli_nonexistent_path_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([str(tmp_path / "does_not_exist.py")])


def test_cli_non_building_result_returns_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Classic typo: `result = generate_project` (function reference)
    # instead of `result = generate_project()`. compile_main raises
    # TypeError on that; the CLI must surface a one-line stderr message
    # and rc=2, not a traceback. Pins the Seer-flagged edge case.
    model = tmp_path / "bad_result.py"
    model.write_text(
        "from lite_step.models import Project\n"
        "def generate_project():\n"
        "    return Project(name='oops')\n"
        "result = generate_project  # missing the call parens\n",
        encoding="utf-8",
    )
    rc = main([str(model)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Project" in err
    assert "Traceback" not in err


def test_python_dash_m_form_works(tmp_path: Path) -> None:
    # ``python -m lite_step output.py`` is the install-agnostic
    # invocation; verify the entrypoint dispatches correctly under it.
    model = _write_model(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "lite_step", str(model)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=_env_with_repo_on_path(),
    )
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "output.ifc").is_file()


def test_cli_does_not_double_compile(tmp_path: Path) -> None:
    # The model's ``if __name__ == "__main__": compile_main()`` tail
    # would write output.ifc itself if we ran with __main__ as the
    # run_name. Verify we run with a different run_name by counting
    # ``wrote ...`` lines on stdout.
    model = _write_model(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "lite_step", str(model)],
        capture_output=True,
        text=True,
        env=_env_with_repo_on_path(),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.count("wrote ") == 1, proc.stdout


# --------------------------------------------------------------------------- lite-step lapin


def test_lapin_ops_says_what_each_element_becomes_and_writes_nothing_beside_the_source(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    model = _write_model(tmp_path)
    out = tmp_path / "ops.json"
    assert main(["lapin", "ops", str(model), "-o", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (data["format"], data["version"]) == ("lite-step.lapin-ops", 1)
    (op,) = data["ops"]
    assert op["kind"] == "primitive"
    assert op["args"] == {"shape": "box", "at": [2.0, 1.5, 0.0], "rotation": 0.0,
                          "width": 4.0, "length": 3.0, "height": 2.4}
    assert "1  editable boxes and cylinders" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ops.json", "output.py"], "no output.ifc left behind"


def test_lapin_without_the_tools_folder_says_where_to_get_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """An installed copy of the compiler has no tools/; that must be said, not a traceback."""
    import lite_step.__main__ as cli

    monkeypatch.setattr(cli, "__file__", str(tmp_path / "lite_step" / "__main__.py"))
    assert main(["lapin", "ops", "model.py"]) == 2
    assert "needs the tools folder of a Lite-STEP checkout" in capsys.readouterr().err


def test_lapin_push_without_lapinbim_says_how_to_get_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """`lapinbim` is Lapin's package, not a requirement of the compiler: its absence is one clear message."""
    monkeypatch.setitem(sys.modules, "lapinbim", None)  # an import of it now raises ImportError
    assert main(["lapin", "push", str(_write_model(tmp_path)), "--port", "1"]) == 2
    err = capsys.readouterr().err
    assert "needs its `lapinbim` package" in err and "pip install lapinbim" in err
    assert "Traceback" not in err


def test_lapin_ops_needs_no_lapinbim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lapinbim", None)
    assert main(["lapin", "ops", str(_write_model(tmp_path))]) == 0
