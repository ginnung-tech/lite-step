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


def test_cli_lapin_compile_target_optional(tmp_path: Path) -> None:
    model = _write_model(tmp_path)
    assert main([str(model), "--format", "lapin"]) == 0
    assert (tmp_path / "output.lapin").is_file()
    assert not (tmp_path / "output.ifc").exists()

    (tmp_path / "output.lapin").unlink()
    assert main([str(model), "format=lapin"]) == 0
    assert (tmp_path / "output.lapin").is_file()

    assert main([str(model), "-o", "custom.lapin"]) == 0
    assert (tmp_path / "custom.lapin").is_file()


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


class _FakeLapin:
    """Lapin's engine as the CLI sees it through `lapinbim`: links by path, and the calls it was sent."""

    links: list = []
    calls: list = []
    params: dict = {}

    def __init__(self, port=None, project=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def call(self, method, params=None, timeout=None):
        _FakeLapin.calls.append((method, params))
        if method == "project.linkLapin":
            assert Path(params["path"]).is_file()
            link = {"id": f"L{len(_FakeLapin.links) + 1}", "kind": "lapin", "path": params["path"],
                    "elements": 1, "storeys": 1, "name": Path(params["path"]).stem}
            _FakeLapin.links.append(link)
            return {"linkId": link["id"]}
        if method == "project.importIfc":
            assert params["mode"] == "link" and Path(params["path"]).is_file()
            link = {"id": f"L{len(_FakeLapin.links) + 1}", "kind": "ifc", "path": params["path"],
                    "elements": 1, "storeys": 1, "name": Path(params["path"]).stem}
            _FakeLapin.links.append(link)
            return {"linkId": link["id"]}
        if method == "engine.hello":
            return {"productName": "Lapin", "version": "0.1.0", "project": None}
        if method == "project.reloadLink":
            assert any(x["id"] == params["id"] for x in _FakeLapin.links)
            return {}
        raise AssertionError(method)

    def query(self, name, args=None):
        if name == "project.info":
            return {"custom": [dict(v) for v in _FakeLapin.params.values()]}
        assert name == "link.list"
        return [dict(x) for x in _FakeLapin.links]

    def execute(self, command, args):
        _FakeLapin.calls.append((command, args))
        if command == "project.setParameter":
            if args["key"].lower() == "name":
                raise RuntimeError(f"{args['key']} is a field of the project's information already.")
            _FakeLapin.params[args["key"]] = dict(args)
        elif command == "project.removeParameter":
            del _FakeLapin.params[args["key"]]
        else:
            assert command == "link.unlink"
            _FakeLapin.links = [x for x in _FakeLapin.links if x["id"] != args["id"]]


@pytest.fixture
def fake_lapin(monkeypatch: pytest.MonkeyPatch):
    import types

    _FakeLapin.links, _FakeLapin.calls, _FakeLapin.params = [], [], {}
    module = types.SimpleNamespace(Lapin=_FakeLapin, LapinError=RuntimeError, engines=lambda: [])
    monkeypatch.setitem(sys.modules, "lapinbim", module)
    return _FakeLapin


def test_lapin_push_links_the_compiled_model_then_reloads_it(tmp_path: Path, fake_lapin) -> None:
    model = _write_model(tmp_path)
    assert main(["lapin", "push", str(model), "--port", "1"]) == 0
    assert (tmp_path / "output.lapin").is_file(), "default push compiles a .lapin package beside the source"
    assert not (tmp_path / "output.ifc").exists(), "default .lapin push leaves no stray output.ifc"
    assert [m for m, _ in fake_lapin.calls] == ["project.linkLapin"]

    fake_lapin.calls.clear()
    assert main(["lapin", "push", str(model), "--port", "1"]) == 0
    assert fake_lapin.calls == [("project.reloadLink", {"id": "L1"})], "the second push reloads the same link"
    assert len(fake_lapin.links) == 1


def test_lapin_push_format_ifc_uses_legacy_ifc_link(tmp_path: Path, fake_lapin) -> None:
    model = _write_model(tmp_path)
    assert main(["lapin", "push", str(model), "format=ifc", "--port", "1"]) == 0
    assert (tmp_path / "output.ifc").is_file(), "format=ifc compiles output.ifc beside the source"
    assert [m for m, _ in fake_lapin.calls] == ["project.importIfc"]

    fake_lapin.calls.clear()
    assert main(["lapin", "push", str(model), "--format", "ifc", "--port", "1"]) == 0
    assert fake_lapin.calls == [("project.reloadLink", {"id": "L1"})]


def test_lapin_push_of_another_folder_makes_its_own_link(tmp_path: Path, fake_lapin) -> None:
    """Two models both called output.py are two links: a link is found by its file's path."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert main(["lapin", "push", str(_write_model(tmp_path / "a")), "--port", "1"]) == 0
    assert main(["lapin", "push", str(_write_model(tmp_path / "b")), "--port", "1"]) == 0
    assert [m for m, _ in fake_lapin.calls] == ["project.linkLapin", "project.linkLapin"]


def test_lapin_remove_unlinks_this_model_and_no_other(tmp_path: Path, fake_lapin) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, b = _write_model(tmp_path / "a"), _write_model(tmp_path / "b")
    main(["lapin", "push", str(a), "--port", "1"])
    main(["lapin", "push", str(b), "--port", "1"])
    assert main(["lapin", "remove", str(a), "--port", "1"]) == 0
    assert [Path(x["path"]).parent.name for x in fake_lapin.links] == ["b"]


def test_lapin_status_lists_lapin_and_ifc_links_and_remove_respects_format(
    tmp_path: Path, fake_lapin, capsys: pytest.CaptureFixture
) -> None:
    model = _write_model(tmp_path)
    assert main(["lapin", "push", str(model), "--port", "1"]) == 0
    assert main(["lapin", "push", str(model), "format=ifc", "--port", "1"]) == 0
    assert main(["lapin", "status", "--port", "1"]) == 0
    out = capsys.readouterr().out
    assert "(lapin):" in out and "(ifc):" in out

    assert main(["lapin", "remove", str(model), "format=ifc", "--port", "1"]) == 0
    assert [x["kind"] for x in fake_lapin.links] == ["lapin"]


def test_lapin_without_the_tools_folder_says_where_to_get_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """An installed copy of the compiler has no tools/; that must be said, not a traceback."""
    import lite_step.__main__ as cli

    monkeypatch.setattr(cli, "__file__", str(tmp_path / "lite_step" / "__main__.py"))
    assert main(["lapin", "push", "model.py"]) == 2
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


PARAM_MODEL = textwrap.dedent(
    """
    from lite_step import compile_main, param
    from lite_step.models import Project, Point, Box

    HEIGHT = param(2400, "mm", "Room height")  # the clear height
    DOUBLE = param(HEIGHT * 2, "mm", "Twice")

    def generate_project() -> Project:
        proj = Project(name="cli_test")
        proj.add(Box(id="room", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=3000, z=HEIGHT), type="sketch"))
        return proj

    result = generate_project()

    if __name__ == "__main__":
        compile_main()
    """
).strip()


def test_lapin_push_publishes_the_parameters_and_pull_writes_lapins_value_back(tmp_path: Path, fake_lapin) -> None:
    model = _write_model(tmp_path, PARAM_MODEL)
    assert main(["lapin", "push", str(model), "--port", "1"]) == 0
    assert fake_lapin.params == {"HEIGHT": {"key": "HEIGHT", "group": "Lite-STEP output", "label": "Room height (mm)",
                                            "type": "number", "value": "2400"}}, "a computed parameter is not published"

    fake_lapin.params["HEIGHT"]["value"] = "2700"  # the user edits it in Lapin
    assert main(["lapin", "pull", str(model), "--port", "1"]) == 0
    text = model.read_text(encoding="utf-8")
    assert 'HEIGHT = param(2700, "mm", "Room height")  # the clear height' in text
    assert text.replace("param(2700,", "param(2400,") == PARAM_MODEL, "only that value changed"

    fake_lapin.calls.clear()
    assert main(["lapin", "push", str(model), "--port", "1"]) == 0
    assert not [c for c, _ in fake_lapin.calls if c == "project.setParameter"], "an unchanged parameter is not sent"


def test_lapin_push_leaves_other_models_parameters_alone_and_removes_its_own_gone_ones(tmp_path: Path, fake_lapin) -> None:
    fake_lapin.params["HEIGHT"] = {"key": "HEIGHT", "group": "Lite-STEP carport", "value": "2100"}
    fake_lapin.params["OLD"] = {"key": "OLD", "group": "Lite-STEP output", "value": "1"}
    assert main(["lapin", "push", str(_write_model(tmp_path, PARAM_MODEL)), "--port", "1"]) == 0
    assert fake_lapin.params["HEIGHT"]["group"] == "Lite-STEP carport", "another model's parameter is not taken over"
    assert "OLD" not in fake_lapin.params


def test_lapin_push_reports_a_parameter_lapin_refuses_and_carries_on(
    tmp_path: Path, fake_lapin, capsys: pytest.CaptureFixture
) -> None:
    """Lapin keeps some keys for its own fields; the push must not die after the link is reloaded."""
    model = _write_model(tmp_path, PARAM_MODEL.replace(
        'DOUBLE = param(', 'NAME = param("Shed", label="Building name")\nDOUBLE = param('))
    assert main(["lapin", "push", str(model), "--port", "1"]) == 0
    assert "parameter NAME: Lapin refused it" in capsys.readouterr().err
    assert list(fake_lapin.params) == ["HEIGHT"]


def test_lapin_ctrl_c_stops_without_a_traceback(
    tmp_path: Path, fake_lapin, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(fake_lapin, "call", interrupted)
    assert main(["lapin", "push", str(_write_model(tmp_path)), "--port", "1"]) == 130
    err = capsys.readouterr().err
    assert "stopped." in err
    assert "Traceback" not in err


def test_lapin_package_contains_valid_format3_tables_and_params(tmp_path: Path, fake_lapin) -> None:
    import hashlib
    import json
    import zipfile

    model = _write_model(tmp_path, PARAM_MODEL)
    assert main(["lapin", "push", str(model), "--port", "1"]) == 0
    pkg = tmp_path / "output.lapin"
    assert pkg.is_file()
    with zipfile.ZipFile(pkg) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["format"] == "lapin-project"
        assert manifest["version"] == 3
        assert "LapinSharedBody" in manifest["classes"]
        assert manifest["classes"]["LapinScript"] == "scripting"
        for part in manifest["parts"]:
            raw = zf.read(part["path"])
            assert len(raw) == part["size"]
            assert hashlib.sha256(raw).hexdigest() == part["sha256"]
            if part["path"].endswith(".arrow"):
                assert raw[:6] == b"ARROW1" and raw[-6:] == b"ARROW1"
        foreign_paths = {f["path"] for f in manifest.get("foreign", [])}
        assert "source/parametric.json" in foreign_paths
        ir = json.loads(zf.read("source/parametric.json"))
        assert [p["name"] for p in ir["params"]] == ["HEIGHT"]
        assert "scripts/output_params.py" in zf.namelist()
