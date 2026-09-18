#!/usr/bin/env python3
"""Refuse downstream identifiers in this tree.

    python -m lite_step.tests.tools.check_no_leaked_identifiers $(git ls-files)

Exit 0 clean, 1 on a match, 2 if given nothing to scan. Files are passed in
rather than walked: an untracked scratch file is not shipped and must not
fail the build.

The banned names are assembled from fragments so this file passes its own
scan; the allow-list entry below is the actual guarantee.

BOUNDARY RULE

``\b`` is wrong in both directions here, so the edges are explicit:

* a 22-character base64 GUID can contain a banned name by chance, and its
  neighbours are letters, so there is no word boundary to anchor on;
* ``_`` is a word character, so ``\bname\b`` does not match
  ``name_worker`` -- the exact spelling a leaked module path would use.

Rejecting alphanumeric neighbours gets both: ``name_worker`` and
``name-org`` match, a name embedded in a GUID does not.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_EDGE = (r"(?<![A-Za-z0-9])", r"(?![A-Za-z0-9])")


def _bare(name: str) -> str:
    return _EDGE[0] + name + _EDGE[1]


_BANNED = re.compile(
    "|".join((
        _bare(r"gi" + r"nnung"),
        r"grab\.tech" + r"nology",
        _bare(r"Fr" + r"eja"),
        _bare(r"gr" + r"oa"),
        _bare(r"bi" + r"frost"),
        _bare(r"mr" + r"claudeanderson"),
    )),
    re.IGNORECASE,
)

#: This file names the patterns it scans for, so it cannot scan itself.
_ALLOW_LIST = frozenset({
    "CHANGELOG.md",
    "lite_step/tests/tools/check_no_leaked_identifiers.py",
})

#: Directories holding proprietary content composed into the image at build
#: time and never travelling here. Inert where absent.
_ALLOW_PREFIXES: tuple[str, ...] = (
    "overlay/",
    "build/",
)


def _is_text_file(p: Path) -> bool:
    try:
        with p.open("rb") as fh:
            chunk = fh.read(4096)
    except OSError:
        return False
    return b"\x00" not in chunk


def check(paths: list[str]) -> int:
    if not paths:
        # Not a pass. Scanning nothing must not be reportable as clean.
        sys.stderr.write(
            "Leak guard: no paths given, so NOTHING was scanned. This is "
            "not a pass.\n"
            "  usage: python -m lite_step.tests.tools."
            "check_no_leaked_identifiers "
            "$(git ls-files)\n",
        )
        return 2

    bad: list[tuple[str, int, str]] = []
    for raw in paths:
        # Normalise separators so the allow-list matches on Windows shells.
        rel = raw.replace("\\", "/")
        if rel in _ALLOW_LIST:
            continue
        if any(rel.startswith(prefix) for prefix in _ALLOW_PREFIXES):
            continue
        p = Path(raw)
        if not p.is_file() or not _is_text_file(p):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _BANNED.search(line):
                bad.append((rel, lineno, line.strip()))

    if bad:
        sys.stderr.write(
            "\nLeak guard: these files name downstream infrastructure, which "
            "this tree must not carry. Parameterise via an env var instead.\n\n"
        )
        for rel, lineno, line in bad[:50]:
            sys.stderr.write(f"  {rel}:{lineno}: {line[:200]}\n")
        if len(bad) > 50:
            sys.stderr.write(f"  ... and {len(bad) - 50} more matches\n")
        sys.stderr.write("\n")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return check(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    sys.exit(main())
