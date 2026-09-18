"""WS-A §1.1 — what carves is a property of the CHILD, not of the verb.

Three claims, each with its falsification:

1. ``brings_void`` is the ONE place the carve trigger is stated, and the
   routing tuple ``taxonomy.OPENINGS`` is DERIVED from it rather than restated
   beside it.
2. ``.opening()`` has ONE signature. It places a void-bringing child in the
   host's LOCAL frame, and refuses anything else — naming the verb that does
   take absolute world coordinates.
3. The displacement exemption has no "Window/Door subtrees" ROW any more —
   the two early returns became one condition. An opening is exempt for the
   reason every anchored child is exempt, and the compile error on
   ``wall.add(win)`` is what makes that safe rather than assumed.
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import execute_lite_step_script
from lite_step.models import (
    Box, Door, Element, Point, Project, Wall, Window,
)
from lite_step.models import taxonomy as tx
from lite_step.models.project import Storey


# ── 1. one source of truth for "does this bring a void" ──────────────────────

def test_openings_tuple_is_derived_from_brings_void():
    """``OPENINGS`` is computed from the class attribute, not written twice.

    Falsification: flip ``Window.brings_void`` to False and the tuple has to
    follow. A hand-written ``(Window, Door)`` beside the attribute would not —
    that is the shape (two implementations of one idea agreeing usually).
    """
    assert tx.OPENINGS == (Window, Door)
    assert all(t.brings_void for t in tx.OPENINGS)
    assert not any(
        t.brings_void for t in tx._PRODUCT_TYPES if t not in tx.OPENINGS
    )


def test_brings_void_reads_instance_class_and_name():
    win = Window(width=1000, height=1000)
    box = Box(start=Point(x=0, y=0, z=0), end=Point(x=100, y=100, z=100))
    for probe in (win, Window, "Window"):
        assert tx.brings_void(probe) is True
    for probe in (box, Box, "Box"):
        assert tx.brings_void(probe) is False
    # is_opening is the same question, one implementation.
    assert tx.is_opening(win) is tx.brings_void(win)
    assert tx.is_opening(box) is tx.brings_void(box)


def test_a_plain_solid_never_brings_a_void():
    """The default is False on the base class, so a new element type is inert
    until someone says otherwise — the safe direction for a silent trigger."""
    assert Element(ifc_class="IfcWall").brings_void is False
    assert Box(start=Point(x=0, y=0, z=0),
               end=Point(x=1, y=1, z=1)).brings_void is False


# ── 2. .opening() has one signature ──────────────────────────────────────────

def _body(**kw):
    return Box(start=Point(x=-5000, y=-150, z=0),
               end=Point(x=5000, y=150, z=3000),
               material="Concrete", **kw)


def test_opening_accepts_a_void_bringing_child():
    body = _body(name="south")
    assert body.opening(Window(width=1200, height=1400, name="w0"),
                        along=1500, up=900) is body
    assert len(body._openings) == 1


@pytest.mark.parametrize("bad", [
    Box(start=Point(x=0, y=0, z=0), end=Point(x=100, y=100, z=100)),
    Element(ifc_class="IfcWall", name="thing"),
])
def test_opening_refuses_a_child_that_brings_no_void(bad):
    """The absolute-coordinate "second signature" is refused, and the refusal
    names the verb that replaces it.

    This form was DOCUMENTED (``n_body.opening(door.union(arch))``) and never
    worked: measured on main @71f19a9 it raised ``TypeError: unsupported
    operand type(s) for *: 'float' and 'NoneType'`` inside
    ``frames.opening_placement`` on a standalone body, and emitted zero
    ``IfcOpeningElement``/``IfcRelVoidsElement``/``IfcBooleanResult`` when the
    body belonged to a Wall. A hole that silently is not there is what the
    loud-failure rule exists to stop.
    """
    body = _body(name="south")
    with pytest.raises(ValueError) as exc:
        body.opening(bad)
    msg = str(exc.value)
    assert "brings no void" in msg
    assert ".void(tool, name=" in msg          # the migration, spelled out
    assert body._openings == []                # nothing partially attached


def test_opening_still_refuses_a_list():
    body = _body(name="south")
    with pytest.raises(ValueError, match="takes ONE opening"):
        body.opening([Window(width=100, height=100)])


# ── 3. the exemption table lost its Window/Door row ──────────────────────────

_ANCHORED_WINDOW = """
from lite_step.models import Project, Wall, Box, Point, Window
def generate_project():
    proj = Project(name="t")
    wall = Wall(name="south")
    wall.add(Box(name="body", start=Point(x=-5000, y=-150, z=0),
                 end=Point(x=5000, y=150, z=3000), material="Concrete"))
    wall.anchor(Window(width=1200, height=1400, name="w0"), along=1500, up=900)
    proj.add(wall)
    return proj
result = generate_project()
"""


def test_an_anchored_opening_is_exempt_because_it_is_anchored():
    """The wall keeps its body — no inferred carve — and the reason is the
    anchor, not the type.

    Falsification of the "it needed its own row" claim: the standalone
    ``if tx.is_opening(elem): return`` early return was deleted from
    ``displacement._collect._walk`` and this still holds, because an opening
    always carries an ``_anchor_spec`` (the bake deliberately never consumes
    it) and the anchored-child branch already covered it.
    """
    from lite_step.compiler.displacement import carve_pairs

    r = execute_lite_step_script(_ANCHORED_WINDOW)
    assert r.success, r.error
    from lite_step.ifc.generator import generate_ifc
    ifc = generate_ifc(r.project)
    assert ifc.success, ifc.error
    pairs = carve_pairs(r.project)
    assert pairs == [], f"an anchored opening must not infer a carve: {pairs}"


def test_an_opening_cannot_reach_containment_unanchored(tmp_path):
    """What makes the merged row safe on every REAL path: no Window/Door
    reaches the displacement pass without an anchor spec.

    Two gates now, and the test exercises both because they cover different
    callers. ``wall.add(win)`` is refused at the AUTHORING line, so a model
    SOURCE carrying it never produces a ``Project`` at all — which matters
    because ``execute_lite_step_script`` is a raw exec that does NOT run
    ``validate_project_report``. A tree assembled around ``.add()`` still
    reaches ``compile_main``, and the validator refusal is still there for it.
    That second gate is why ``brings_void`` survives as a named backstop
    inside the merged condition rather than being deleted outright.
    """
    from lite_step.compiler.executor import LiteStepCompileError, compile_main

    head = """
from lite_step.models import Project, Wall, Box, Point, Window
def generate_project():
    proj = Project(name="t")
    wall = Wall(name="south")
    wall.add(Box(name="body", start=Point(x=-5000, y=-150, z=0),
                 end=Point(x=5000, y=150, z=3000), material="Concrete"))
    {attach}
    proj.add(wall)
    return proj
result = generate_project()
"""
    # 1. The authoring line. The source does not even finish executing.
    src = head.format(attach='wall.add(Window(width=1200, height=1400, name="w0"))')
    model = tmp_path / "model.py"
    model.write_text(src, encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be ADDED"):
        exec(compile(src, str(model), "exec"), {"__file__": str(model), "__name__": "m"})

    # 2. The backstop, for a tree built around ``.add()``.
    src = head.format(
        attach='wall._elements.append(Window(width=1200, height=1400, name="w0"))')
    model = tmp_path / "model2.py"
    model.write_text(src, encoding="utf-8")
    ns = {"__file__": str(model), "__name__": "m"}
    exec(compile(src, str(model), "exec"), ns)
    with pytest.raises(LiteStepCompileError) as exc:
        compile_main(result=ns["result"], source_path=str(model))
    assert "needs a position" in str(exc.value)


def test_an_anchored_solid_is_still_exempt():
    """The other half of the same row — unchanged, and re-pinned here because
    the two now share ONE branch."""
    from lite_step.compiler.displacement import carve_pairs

    proj = Project(name="t")
    storey = Storey(elevation=0)
    wall = Wall(name="south")
    wall.add(_body(name="body"))
    wall.anchor(Box(name="corbel", start=Point(x=0, y=0, z=0),
                    end=Point(x=200, y=400, z=200), material="Concrete"),
                along=1000, up=2000)
    storey.add(wall)
    proj.add_storey(storey)
    from lite_step.compiler.executor import normalize_project_to_meters
    from lite_step.compiler.displacement import apply_displacement
    norm = normalize_project_to_meters(proj)
    apply_displacement(norm)
    assert carve_pairs(norm) == []
