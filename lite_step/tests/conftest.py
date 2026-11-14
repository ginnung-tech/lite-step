"""Test-session plumbing shared by every test module.

**IFC file lifetime (ifcopenshell >= 0.9).** In 0.9 an entity instance no
longer keeps its file alive: once the last reference to the ``ifcopenshell.file``
is gone, reading or tessellating one of its entities is a use-after-free and
the interpreter dies with an access violation, taking the whole run with it.
0.8.x kept the file alive through its entities, and the tests grew the idiom
that relies on it — ``_open(text).by_type("IfcWall")``, or a helper that opens
a model and returns one of its entities.

The fixture below holds every file a TEST MODULE opens until that test ends.
Only test code is covered, on purpose: a file opened by ``lite_step`` itself
is not held, so the same bug in compiler code still crashes the suite instead
of hiding behind this shim.
"""
from __future__ import annotations

import sys

import pytest

try:
    import ifcopenshell
except ImportError:  # pragma: no cover - ifcopenshell is a hard dependency
    ifcopenshell = None

_HELD: list = []

_REQUIREMENTS = __import__("pathlib").Path(__file__).resolve().parents[2] / "requirements.txt"


def _below_floor() -> list:
    """``(package, installed, floor)`` for every requirement installed below
    the version ``requirements.txt`` asks for. Several tests pin behaviour of a
    newer dependency (an upstream fix, a changed API), so an old environment
    fails them in ways that look like compiler bugs; say so up front."""
    import importlib.metadata as md
    import re

    try:
        from packaging.version import Version
    except ImportError:  # pragma: no cover - packaging ships with pytest
        return []
    out = []
    for line in _REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)\s*(?:>=|==)\s*([0-9][^,\s#]*)", line)
        if not m:
            continue
        name, floor = m.groups()
        try:
            installed = md.version(name)
        except md.PackageNotFoundError:
            continue
        if Version(installed) < Version(floor):
            out.append((name, installed, floor))
    return out


def pytest_sessionstart(session):
    stale = _below_floor()
    if stale:
        raise pytest.UsageError(
            "dependencies below requirements.txt: "
            + ", ".join(f"{n} {have} < {want}" for n, have, want in stale)
            + " — run `pip install -r requirements.txt`")


def _from_test_code() -> bool:
    caller = sys._getframe(2).f_globals.get("__name__", "")
    return caller.startswith("lite_step.tests") or caller.startswith("test_")


def _holding(fn):
    def wrapper(*args, **kwargs):
        model = fn(*args, **kwargs)
        if _from_test_code():
            _HELD.append(model)
        return model
    wrapper.__wrapped__ = fn
    return wrapper


@pytest.fixture(autouse=True)
def _hold_ifc_files_opened_by_tests(monkeypatch):
    if ifcopenshell is not None:
        monkeypatch.setattr(ifcopenshell, "open", _holding(ifcopenshell.open))
        monkeypatch.setattr(ifcopenshell.file, "from_string",
                            staticmethod(_holding(ifcopenshell.file.from_string)))
    yield
    _HELD.clear()
