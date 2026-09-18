"""``python output.py --check`` and ``--q <selector>``.

Coordinates only exist while the script runs, so the script is the right place
to ask. These two flags delete the probe-script habit: no more writing a
throwaway file that imports the model to print one number.

The load-bearing test in here is ``test_flags_are_ignored_under_pytest``.
``compile_main()`` takes no arguments from the canonical script tail, so
reading ``sys.argv`` unconditionally would make every test that calls it parse
PYTEST's argv — and ``-q`` is both a pytest flag and a plausible one of ours.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

_MODEL = '''
from lite_step import compile_main
from lite_step.models import Project, Wall, Box, Point, Extrude

I = lambda x: int(round(x))  # noqa: E731

result = Project(name="demo")
result.add(Wall(name="north").add(
    Box(start=Point(x=0, y=0, z=0), end=Point(x=6000, y=350, z=2700),
        name="body")))
result.add(Extrude(
    contour=[Point(x=20000, y=0, z=0), Point(x=23000, y=3000, z=0),
             Point(x=23000, y=3000, z=2500), Point(x=20000, y=0, z=2500)],
    thickness=300, name="slanted"))

if __name__ == "__main__":
    compile_main()
'''


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model.py"
    path.write_text(_MODEL, encoding="utf-8")
    return path


def _run(model_path, *args):
    return subprocess.run(
        [sys.executable, "-W", "ignore", str(model_path), *args],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)},
        timeout=180,
    )


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------


def test_check_compiles_but_writes_nothing(model):
    out = _run(model, "--check")
    assert out.returncode == 0, out.stderr
    assert "not written" in out.stdout
    assert not (model.parent / "output.ifc").exists()


def test_check_still_runs_validation_and_reports(model):
    """--check is not a syntax check — it validates, normalises and generates,
    so everything a real compile would say is said. Only the write is skipped,
    which is what makes it safe to run against a model a viewer is holding."""
    out = _run(model, "--check")
    assert "csg depth:" in out.stderr, "the per-compile metric still prints"
    assert "bytes of IFC" in out.stdout, "generation really ran"


def test_without_flags_the_file_is_written(model):
    out = _run(model)
    assert out.returncode == 0, out.stderr
    assert (model.parent / "output.ifc").exists()


# ---------------------------------------------------------------------------
# --q
# ---------------------------------------------------------------------------


def test_q_dumps_every_accessor_for_the_match(model):
    out = _run(model, "--q", "wall:north")
    assert out.returncode == 0, out.stderr
    for accessor in ("authored_aabb", "world_aabb", "obb"):
        assert accessor in out.stdout
    assert "size=(6000.0, 350.0, 2700.0)" in out.stdout


def test_q_matches_the_same_way_anchor_host_does(model):
    """Segment-aligned suffix match, via ``naming.host_matches`` — so
    ``wall:north`` finds the wall AND its body child. One matching rule in the
    DSL rather than a second one that drifts from it."""
    out = _run(model, "--q", "wall:north")
    assert "wall:north  (Wall)" in out.stdout
    assert "box:body:wall:north  (Box)" in out.stdout


def test_q_writes_no_ifc(model):
    """A query is a read. It must not clobber an output.ifc."""
    _run(model, "--q", "wall:north")
    assert not (model.parent / "output.ifc").exists()


def test_q_on_an_oriented_box_explains_its_own_corners(model):
    """``min``/``max`` on an oriented box are DIAGONAL corners, so ``min.z``
    can exceed ``max.z``. Printed without a word that reads as a bug."""
    out = _run(model, "--q", "extrude:slanted")
    assert "rule=extrude-normal" in out.stdout
    assert "size=(4243.0, 300.0, 2500.0)" in out.stdout
    assert "diagonal corners" in out.stdout


def test_q_with_no_match_lists_the_names_that_exist(model):
    """A miss is a typo far more often than a wrong model, so answer the
    question the author is about to ask."""
    out = _run(model, "--q", "wall:nope")
    assert "no element matches" in out.stderr
    assert "wall:north" in out.stderr


def test_q_without_a_selector_is_a_clear_error(model):
    out = _run(model, "--q")
    assert out.returncode != 0
    assert "--q needs a selector" in out.stderr


# ---------------------------------------------------------------------------
# The argv gate
# ---------------------------------------------------------------------------


def test_flags_are_ignored_under_pytest():
    """THE test this module exists for.

    We are inside pytest right now, and pytest's own argv is full of flags —
    including ``-q``. ``compile_main()`` must not parse them: the gate is that
    ``sys.argv[0]`` has to resolve to the same file as the caller's
    ``__file__``, which under pytest is pytest's path.

    If this ever fails, every test in the repo that calls ``compile_main()``
    becomes hostage to how the suite was invoked.
    """
    from lite_step.compiler.executor import _script_flags

    assert _script_flags(__file__) == {}, (
        "flags must only be read when the model was run directly as "
        f"`python <model>.py`; sys.argv[0] is {sys.argv[0]!r}"
    )


def test_flags_parse_when_argv0_is_the_model_itself(tmp_path, monkeypatch):
    from lite_step.compiler.executor import _script_flags

    model_path = tmp_path / "m.py"
    model_path.write_text("# model", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(model_path), "--check"])
    assert _script_flags(str(model_path)) == {"query": None, "check": True}


def test_lite_step_cli_still_writes_normally(model):
    """``lite-step model.py`` loads the module under a non-__main__ run name,
    so the script tail is skipped and this path is untouched by the gate."""
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-m", "lite_step", str(model)],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO)}, timeout=180)
    assert out.returncode == 0, out.stderr
    assert (model.parent / "output.ifc").exists()
