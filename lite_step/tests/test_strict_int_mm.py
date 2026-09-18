"""Strict int-millimeter enforcement (DSL v1.5 RULE 2, WS1 PR-C).

Flag-gated: OFF by default (today's float-tolerant behavior unchanged), the
WS2 cutover flips it on. When ON, ``Point`` / ``Point2D`` reject ``float``
coordinates at construction with an actionable ``I()`` hint; the
meters-normalizer is immune (it runs inside ``suspend_strict()`` because its
float-meter Point rebuilds are internal pipeline state, not user input).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from lite_step import (
    disable_strict_int_mm,
    enable_strict_int_mm,
    strict_int_mm_enabled,
)
from lite_step.compiler.executor import compile_main, normalize_project_to_meters
from lite_step.models import Project, Point, Point2D, Box, Extrude
from lite_step.strict import suspend_strict


@pytest.fixture(autouse=True)
def _reset_strict_flag():
    """No leaked flag state between tests, whatever a test does.

    Restores the post-cutover DEFAULT (strict ON)."""
    yield
    enable_strict_int_mm()


# ── default (flag OFF) ──────────────────────────────────────────────


def test_default_is_on_since_cutover():
    # DSL v1.5 is the shipped contract — strict, with no way to opt out.
    assert strict_int_mm_enabled() is True
    with pytest.raises(Exception):
        Point(x=1.5, y=2.7, z=3.9)


def test_no_env_var_can_relax_the_contract():
    """The v10 hard cutover removed ``LITESTEP_STRICT_INT_MM``.

    Run in a SUBPROCESS with the legacy opt-out set, because the flag is read at
    import time — asserting it in-process would only prove this process
    already imported strictly. A deploy still carrying the legacy env var
    must be strict anyway, which is the point: an env switch that relaxes
    an authoring contract lets a deploy disagree with the spec, and the
    failure it produces (float mm accepted here, rounded downstream) is
    geometry that is subtly wrong rather than a refusal.
    """
    import os, subprocess, sys
    from pathlib import Path as _P
    code = (
        "from lite_step import strict_int_mm_enabled;"
        "print(strict_int_mm_enabled())"
    )
    env = dict(os.environ)
    env["LITESTEP_STRICT_INT_MM"] = "0"          # the retired escape hatch
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env=env, cwd=str(_P(__file__).resolve().parents[2]),
    )
    assert out.stdout.strip() == "True", out.stderr


def test_floats_accepted_when_explicitly_disabled():
    disable_strict_int_mm()
    p = Point(x=1.5, y=2.7, z=3.9)  # legacy escape-hatch behavior
    assert p.x == 1.5
    assert Point2D(x=0.25, y=0.75).y == 0.75


# ── flag ON: rejection semantics ────────────────────────────────────


def test_strict_rejects_float_with_field_name_and_hint():
    enable_strict_int_mm()
    with pytest.raises(Exception) as exc_info:
        Point(x=1.5, y=0, z=0)
    msg = str(exc_info.value)
    assert "x=1.5" in msg
    assert "I(" in msg  # the actionable fix
    assert "millimeters" in msg


def test_strict_rejects_integral_floats_too():
    # The spec rejects the TYPE, not just non-integral values: 2000.0 fails.
    enable_strict_int_mm()
    with pytest.raises(Exception):
        Point(x=2000.0, y=0, z=0)


def test_strict_accepts_ints():
    enable_strict_int_mm()
    p = Point(x=-6000, y=4500, z=0)
    assert (p.x, p.y, p.z) == (-6000, 4500, 0)


def test_strict_applies_to_point2d():
    enable_strict_int_mm()
    with pytest.raises(Exception) as exc_info:
        Point2D(x=0, y=97.5)
    assert "y=97.5" in str(exc_info.value)
    assert Point2D(x=45, y=195).x == 45


def test_strict_applies_to_storey_elevation():
    #: ``Storey.elevation`` is typed ``float`` and was the one hole
    # in RULE 2's "every mm dimension" claim — with strict ON,
    # ``Storey(elevation=1402.5)`` was accepted silently and the executor
    # divided it straight to metres, i.e. sub-mm drift where a raise was
    # promised. It is on the authoring surface (the cheat sheet), so the
    # claim has to be true of it.
    from lite_step.models.project import Storey

    enable_strict_int_mm()
    with pytest.raises(Exception) as exc_info:
        Storey(name="first", elevation=1402.5)
    assert "elevation=1402.5" in str(exc_info.value)
    # The TYPE is rejected, not just the non-integral value.
    with pytest.raises(Exception):
        Storey(name="first", elevation=3000.0)
    assert Storey(name="first", elevation=3000).elevation == 3000


def test_the_meters_normalizer_still_divides_storey_elevation():
    # The falsifying half: the normalizer ASSIGNS ``storey.elevation =
    # elevation / 1000.0``, and a gate that fired on assignment would break
    # every multi-storey compile. Pydantic does not validate on assignment
    # here, so the float metres never meet the gate — measured, not assumed,
    # on a storey deliberately NOT at zero.
    from lite_step.models.project import Storey

    enable_strict_int_mm()
    b = Project(name="Elevated")
    ground = Storey(name="ground", elevation=0)
    first = Storey(name="first", elevation=3400)
    b.add_storey(ground)
    b.add_storey(first)
    first.add(Box(name="c", start=Point(x=1300, y=700, z=0),
                  end=Point(x=3100, y=2600, z=2400)))
    normalized = normalize_project_to_meters(b)
    assert normalized.storeys[1].elevation == pytest.approx(3.4)


def test_strict_off_accepts_a_float_storey_elevation():
    # Control: the legacy escape hatch still opts out of this gate like
    # every other one.
    from lite_step.models.project import Storey

    disable_strict_int_mm()
    assert Storey(name="first", elevation=1402.5).elevation == 1402.5


def test_strict_point_arithmetic_on_validated_points_stays_valid():
    # __add__/__sub__ derive new Points from already-validated operands via
    # model_construct — the float-typed field storage (1000 → 1000.0) must
    # not make arithmetic on spec-conforming points trip the gate.
    enable_strict_int_mm()
    p = Point(x=1000, y=2000, z=0) + Point(x=500, y=-500, z=300)
    assert (p.x, p.y, p.z) == (1500, 1500, 300)
    q = p - Point(x=500, y=0, z=0)
    assert (q.x, q.y, q.z) == (1000, 1500, 300)


# NOTE: RULE 2's "floats are legal only inside props={}" — the universal
# `props={}` field ships in WS1 PR-D (derivation/props/Anchor.host); its
# strict-mode float-tolerance test lands there.


# ── keyword-only pinning (RULE 1) ───────────────────────────────────


def test_constructors_reject_positional_args():
    # pydantic BaseModel is keyword-only by construction; pin it so a future
    # refactor (dataclass swap, custom __init__) can't silently regress RULE 1.
    with pytest.raises(TypeError):
        Point(1, 2, 3)  # type: ignore[misc]
    with pytest.raises(TypeError):
        Point2D(1, 2)  # type: ignore[misc]


# ── toggling / suspension mechanics ─────────────────────────────────


def test_enable_disable_roundtrip():
    assert strict_int_mm_enabled() is True  # post-cutover default
    disable_strict_int_mm()
    assert strict_int_mm_enabled() is False
    enable_strict_int_mm()
    assert strict_int_mm_enabled() is True


def test_suspend_strict_restores_even_on_exception():
    enable_strict_int_mm()
    with pytest.raises(RuntimeError):
        with suspend_strict():
            assert strict_int_mm_enabled() is False
            assert Point(x=1.5, y=0, z=0).x == 1.5  # allowed inside
            raise RuntimeError("boom")
    assert strict_int_mm_enabled() is True  # restored


def test_suspend_strict_nesting_is_safe():
    enable_strict_int_mm()
    with suspend_strict():
        with suspend_strict():
            assert strict_int_mm_enabled() is False
        assert strict_int_mm_enabled() is False
    assert strict_int_mm_enabled() is True


def test_env_var_enables_at_import():
    # A fresh interpreter with LITESTEP_STRICT_INT_MM=1 boots strict.
    import os

    code = (
        "from lite_step import strict_int_mm_enabled;"
        "print(strict_int_mm_enabled())"
    )
    env = dict(os.environ)
    env["LITESTEP_STRICT_INT_MM"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )
    assert out.stdout.strip() == "True", out.stderr


# ── normalizer immunity (the load-bearing design point) ─────────────


def test_normalizer_is_immune_under_strict():
    # normalize_project_to_meters rebuilds Points as float METERS — under
    # strict that construction must not trip the gate (@suspends_strict).
    enable_strict_int_mm()
    b = Project(name="Immune")
    b.add(Box(name="c", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=2000, z=2000)))
    normalized = normalize_project_to_meters(b)
    assert normalized is not None
    assert strict_int_mm_enabled() is True  # restored after the call


def test_full_compile_succeeds_with_strict_on(tmp_path):
    # End-to-end: int-clean model + strict ON → validate → normalize →
    # generate → output.ifc on disk. Proves the whole pipeline is
    # strict-compatible for spec-conforming input.
    enable_strict_int_mm()
    b = Project(name="StrictCube")
    b.add(
        Box(
            name="cube",
            start=Point(x=0, y=0, z=0),
            end=Point(x=2000, y=2000, z=2000),
            material="Concrete",
        )
    )
    source = tmp_path / "output.py"
    source.write_text("# strict e2e", encoding="utf-8")
    out = compile_main(result=b, source_path=str(source))
    assert out is not None and out.exists() and out.stat().st_size > 0
    assert strict_int_mm_enabled() is True


def test_strict_off_full_compile_unchanged(tmp_path):
    # Control: the same model with a float coordinate compiles when the flag
    # is explicitly opted out (legacy escape hatch).
    disable_strict_int_mm()
    b = Project(name="FloatTolerant")
    b.add(Box(start=Point(x=0.0, y=0, z=0), end=Point(x=2000, y=2000, z=2000)))
    source = tmp_path / "output.py"
    source.write_text("# off e2e", encoding="utf-8")
    out = compile_main(result=b, source_path=str(source))
    assert out is not None and out.exists()
