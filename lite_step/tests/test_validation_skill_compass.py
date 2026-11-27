"""Compass parity between the DSL reference and the validation skills.

The DSL's RULE 4 pins the project compass: North = Y+, South = Y−,
East = X+, West = X−. Both validator skills describe positions with
the same compass — a validator running an inverted mapping reports
"door on the north wall" for a door the builder correctly placed
south, i.e. false findings on correct buildings (a class of bug where the axis blocks in both SKILLs had drifted to the exact
inverse of RULE 4 while their own worked examples used the correct
mapping).

Text-level contract test: cheap, and it turns any future drift into a
red test instead of a silently lying validator.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

DSL_REFERENCE = REPO_ROOT / "agents" / "dsl-reference.md"
SKILLS = {
    "ifc": REPO_ROOT / "agents" / "validation" / "ifc" / "SKILL.md",
    "vision": REPO_ROOT / "agents" / "validation" / "vision" / "SKILL.md",
}

# The canonical compass claims every validator skill must carry, in the
# exact backticked form the skills use.
CANONICAL_LINES = (
    "`x+ east`, `x- west`",
    "`y+ north`, `y- south`",
)

# The inverted claims that must never reappear (the pre-fix drift).
INVERTED_FRAGMENTS = (
    "x+ west",
    "x- east",
    "y+ south",
    "y- north",
)


def test_dsl_reference_pins_the_canonical_compass() -> None:
    """RULE 4 is the source of truth — if this line changes, the skill
    assertions below must be revisited deliberately, not silently."""
    text = DSL_REFERENCE.read_text(encoding="utf-8")
    assert "North = Y+, South = Y−, East = X+, West = X−" in text


@pytest.mark.parametrize("kind", sorted(SKILLS))
def test_validation_skill_compass_matches_rule_4(kind: str) -> None:
    text = SKILLS[kind].read_text(encoding="utf-8")
    for line in CANONICAL_LINES:
        assert line in text, f"{kind} SKILL.md lost the canonical compass line: {line}"
    for fragment in INVERTED_FRAGMENTS:
        assert fragment not in text, (
            f"{kind} SKILL.md carries the INVERTED compass claim {fragment!r} — "
            "it must match dsl-reference RULE 4 (N=Y+, S=Y-, E=X+, W=X-)"
        )
