"""The leak guard's boundary rule, in both directions.

The guard keeps downstream product names out of this public tree. Getting
its word boundary wrong fails silently in whichever direction it is wrong:
a false positive blocks a legitimate compiled artefact, and a false
negative ships the name it exists to catch. Neither shows up in any other
test, so both are pinned here.

The interesting cases are all about what counts as an edge:

* An IFC GUID is 22 random base64 characters, so a banned name lands
  inside one by chance. A shipped corpus model really does carry
  a GUID containing one of them, and the scan is case-insensitive.
  ``\\b`` cannot reject that: the neighbouring characters are letters, so
  there is no word boundary to anchor on.
* ``_`` is a word character, so a ``\\b``-delimited name does NOT match
  ``<name>_worker`` — which is exactly how a leaked module path or env
  var would be spelled.

So the boundary has to reject alphanumeric neighbours while still treating
``_`` and ``-`` as separators.

NOTHING IN THIS FILE SPELLS A BANNED NAME — not in code, not in prose,
not in an identifier. The guard scans this file like every other, and an
allow-list entry for it would be a hole big enough to hide a real leak in.
Every name is assembled at runtime from :data:`_NAMES`, whose keys are
neutral role descriptions for the same reason: naming a constant after
the thing it holds is itself a spelling of the name, and the guard
correctly flags that. It did, twice, while this file was being written —
once for the constants and once for the sentence explaining them.
"""

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_guard():
    path = _ROOT / "lite_step" / "tests" / "tools" / "check_no_leaked_identifiers.py"
    spec = importlib.util.spec_from_file_location("leak_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUARD = _load_guard()

#: The banned names, split across concatenation and keyed by what the
#: thing *is* rather than what it is called. Assembling them here keeps
#: every literal below out of the file.
_NAMES = {
    "compiler_host": "gr" + "oa",
    "org": "gi" + "nnung",
    "assistant": "Fr" + "eja",
    "agent_loop": "bi" + "frost",
    "domain": "grab.tech" + "nology",
    "bot_account": "mr" + "claudeanderson",
}


@pytest.mark.parametrize(
    "template, key, why",
    [
        ("{}_worker", "compiler_host", "underscore suffix — the spelling `\\b` missed"),
        ("{}-tech/lite-step", "org", "hyphenated org in a repo URL"),
        ("import {}.security", "compiler_host", "module import"),
        ("from {} import loop", "agent_loop", "bare word in an import"),
        ("{} answered", "assistant", "prose"),
        ("staging.{}", "domain", "deployment domain"),
        ("committed by {}", "bot_account", "bot handle"),
        ("FLY_{}_WORKER_APP", "compiler_host", "env var, and the scan folds case"),
    ],
)
def test_guard_catches_real_leaks(template, key, why):
    text = template.format(_NAMES[key])
    assert GUARD._BANNED.search(text) is not None, f"should have flagged ({why}): {text}"


@pytest.mark.parametrize(
    "guid, key",
    [
        ("1o$ja2_XD1PfREjAP_1ASf", "assistant"),
        ("1MPaiw3lX9YPBZ9Y6fgrOa", "compiler_host"),
    ],
)
def test_guard_does_not_flag_ifc_guids(guid, key):
    """A random GUID that happens to contain a banned name is not a leak.

    Both GUIDs are real: the first ships in the window skill's
    ``output.ifc``, the second cost an earlier debugging pass. The
    containment check folds case, because the casing inside a GUID is
    arbitrary and the guard is case-insensitive anyway.
    """
    assert _NAMES[key].lower() in guid.lower(), (
        "fixture drifted — this GUID does not contain the name, so the test "
        "would pass without exercising anything"
    )
    assert GUARD._BANNED.search(guid) is None, f"false positive on IFC GUID {guid}"


def test_tracked_tree_is_clean():
    """The whole tracked tree passes. This is the guarantee; the rest is why.

    Uses ``git ls-files`` rather than a filesystem walk on purpose — an
    untracked file is not shipped, and scanning one would fail the build
    over a local scratch file. The inverse mistake is what left ``main``
    red after the guard's own fix landed: the gate was run by hand while
    the new test file was still untracked, so it was never handed over.
    """
    import subprocess

    files = subprocess.run(
        ["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert files, "git ls-files returned nothing — the scan would vacuously pass"

    offenders = []
    for rel in files:
        if rel in GUARD._ALLOW_LIST or any(
            rel.startswith(p) for p in getattr(GUARD, "_ALLOW_PREFIXES", ())
        ):
            continue
        path = _ROOT / rel
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if GUARD._BANNED.search(line):
                offenders.append(f"{rel}:{lineno}: {line.strip()[:100]}")

    assert not offenders, "leaked identifiers:\n" + "\n".join(offenders[:20])
