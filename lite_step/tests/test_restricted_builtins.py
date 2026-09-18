"""The sandbox builtins whitelist — what a model author may actually call.

The host's AST guard is only HALF the block on a
builtin. The in-process ``exec()`` runs with ``__builtins__`` set to
``get_restricted_builtins()``, so a name the guard permits but the whitelist
omits fails later with ``NameError: name 'getattr' is not defined`` — a
different error, the same dead end for the author.

That is why the tests here execute real scripts through
``execute_lite_step_script`` rather than only inspecting the dict: a
whitelist entry that never resolves at runtime is not a capability.

Read-only introspection (``getattr`` / ``type`` / ``vars`` / ``dir``) is
ALLOWED — survey code walks the project tree with it. Mutation, code-exec and
frame-escape (``setattr`` / ``delattr`` / ``eval`` / ``exec`` / ``open`` /
``globals`` / ``locals``) stay out.
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import execute_lite_step_script
from lite_step.compiler.namespace import get_restricted_builtins


INTROSPECTION_BUILTINS = ("getattr", "type", "vars", "dir")

STILL_EXCLUDED_BUILTINS = (
    "setattr",
    "delattr",
    "globals",
    "locals",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "breakpoint",
    "exit",
    "quit",
)


# ---------------------------------------------------------------------------
# The whitelist itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", INTROSPECTION_BUILTINS)
def test_introspection_builtins_are_whitelisted(name: str) -> None:
    assert name in get_restricted_builtins(), (
        f"{name!r} must resolve inside the sandbox — the AST guard allows it, "
        "so omitting it here just swaps unsafe_code for NameError"
    )


@pytest.mark.parametrize("name", STILL_EXCLUDED_BUILTINS)
def test_mutating_and_exec_builtins_stay_out(name: str) -> None:
    assert name not in get_restricted_builtins()


def test_already_present_companions_unchanged() -> None:
    """``hasattr`` / ``isinstance`` / ``issubclass`` / ``repr`` / ``id`` /
    ``callable`` were always there and stay there — this relaxation adds, it
    does not trade."""
    builtins_map = get_restricted_builtins()
    for name in ("hasattr", "isinstance", "issubclass", "repr", "id", "callable"):
        assert name in builtins_map


def test_the_import_whitelist_is_exactly_the_documented_four() -> None:
    """The frozenset also carried ``model_sketching``, a package
    that exists nowhere in the repo — importing it would ``ModuleNotFoundError``
    at exec time, while the agent-facing reference documents the whitelist as
    ``lite_step, math, numpy, random``. A whitelist entry nobody can use is
    not harmless: it is a name the guard promises and the runtime refuses,
    and the documented set and the implemented set have to be one thing."""
    from lite_step.compiler.namespace import ALLOWED_IMPORT_PACKAGES

    assert ALLOWED_IMPORT_PACKAGES == frozenset(
        {"lite_step", "math", "numpy", "random"})


def test_the_restricted_import_refuses_a_package_off_the_whitelist() -> None:
    """The guard's behaviour either side of the deleted entry: a whitelisted
    package still resolves, and the retired name is now refused like any
    other stranger — with the real, usable set in the message."""
    from lite_step.compiler.namespace import _make_whitelisted_import

    restricted = _make_whitelisted_import()
    assert restricted("math") is not None
    with pytest.raises(ImportError) as exc_info:
        restricted("model_sketching")
    assert "model_sketching" in str(exc_info.value)
    assert "'lite_step', 'math', 'numpy', 'random'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# They actually RESOLVE at runtime (the test that proves the whitelist edit)
# ---------------------------------------------------------------------------


SURVEY_SCRIPT = """
def generate_project():
    proj = Project(name="Survey")
    storey = Storey(name="ground", elevation=0)
    wall = Wall(name="north")
    wall.add(Box(name="body",
                 start=Point(x=0, y=0, z=0),
                 end=Point(x=4000, y=200, z=2400)))
    storey.add(wall)
    proj.add_storey(storey)
    return proj


result = generate_project()

# --- survey the tree we just built, using every introspection builtin ------
_kinds = []
for _storey in result.storeys:
    for _elem in _storey.elements:
        _kinds.append(type(_elem).__name__)
        _mat = getattr(_elem, "material", None)
        _name = getattr(_elem, "name", None)
        _fields = vars(_elem)
        _attrs = [_n for _n in dir(_elem) if not _n.startswith("_")]
        assert isinstance(_attrs, list)
        assert isinstance(_fields, dict)

assert _kinds == ["Wall"], _kinds
"""


def test_survey_script_executes_and_builds_a_project() -> None:
    """The end-to-end proof: a script that calls ``getattr``/``type``/``vars``/
    ``dir`` passes the guard AND runs to a valid Project.

    Part 1 alone (removing the AST rule) leaves this failing with NameError.
    """
    result = execute_lite_step_script(SURVEY_SCRIPT)
    assert result.success, (
        f"{result.error_type}: {result.error}\n{result.traceback_str or ''}"
    )
    assert result.project is not None
    assert result.project.name == "Survey"


@pytest.mark.parametrize(
    "name,snippet",
    [
        ("getattr", "_v = getattr(result, 'name', None)\nassert _v == 'X'\n"),
        ("type", "_v = type(result).__name__\nassert _v == 'Project'\n"),
        ("vars", "_v = vars(result)\nassert isinstance(_v, dict)\n"),
        ("dir", "_v = dir(result)\nassert 'storeys' in _v\n"),
    ],
)
def test_each_introspection_builtin_resolves_in_the_sandbox(
    name: str, snippet: str
) -> None:
    script = "result = Project(name='X')\n" + snippet
    outcome = execute_lite_step_script(script)
    assert outcome.success, (
        f"{name} did not resolve in the sandbox — "
        f"{outcome.error_type}: {outcome.error}"
    )
