"""Every refusal the DSL reference LISTS is enforced by the compiler.

The reference carried a "Refusals worth knowing" block and a "do NOT author an
envelope solid over a buildup" line. Prose is a poor place for a rule: an
author who does not read it gets a model that compiles and is wrong. The
block is deleted, so this suite is what keeps the refusals real — a dropped
line is only safe while the compiler still raises, and nothing else pins that.

Each case asserts BOTH halves: that it raises, and that the message names a
remedy. A refusal without a remedy is a bare ``AttributeError`` —
technically loud, practically useless.
"""
from __future__ import annotations

import pytest

from lite_step.models import (
    Box, Element, Point, Project, Site, Space, SpatialElement, Storey,
    Wall, Window,
)
from lite_step.compiler.executor import validate_project_report


def _wall(name="south", thickness=300):
    w = Wall(name=name)
    w.add(Box(name="body", start=Point(x=0, y=0, z=0),
              end=Point(x=6000, y=thickness, z=2400)))
    return w


def _leaf(name, y0, y1):
    w = Wall(name=name)
    w.add(Box(name="body", start=Point(x=0, y=y0, z=0),
              end=Point(x=6000, y=y1, z=2400)))
    return w


def _project(*storey_elements):
    proj = Project(name="t")
    storey = Storey(name="ground", elevation=0)
    proj.add_storey(storey)
    for elem in storey_elements:
        storey.add(elem)
    return proj


# --------------------------------------------------------------------------
# The block that was deleted — each line, still enforced
# --------------------------------------------------------------------------

def test_add_of_an_opening_is_refused_with_the_anchor_remedy():
    with pytest.raises(ValueError) as exc:
        _wall().add(Window(width=1200, height=1400, name="w1"))
    msg = str(exc.value)
    assert ".anchor(" in msg, f"must name the verb that works: {msg}"


def test_a_product_into_a_space_is_refused():
    space = Space(name="room")
    with pytest.raises(ValueError) as exc:
        space.add(_wall("inner"))
    assert "AIR" in str(exc.value) or "boundary" in str(exc.value).lower()


def test_a_spatial_element_into_a_product_is_refused():
    with pytest.raises(ValueError) as exc:
        _wall().add(SpatialElement(ifc_class="IfcSpatialZone", name="z"))
    # The remedy here is the allowed set, which the message enumerates.
    assert "invalid child" in str(exc.value)


def test_a_second_placement_is_refused():
    win = Window(width=1200, height=1400, name="w1")
    a, b = _wall("south"), _wall("north")
    a.anchor(win, along=1000, up=900)
    with pytest.raises(ValueError) as exc:
        b.anchor(win, along=1000, up=900)
    assert "already anchored" in str(exc.value)


def test_a_containment_cycle_is_refused():
    a, b = _wall("a"), _wall("b")
    a.add(b)
    with pytest.raises(ValueError) as exc:
        b.add(a)
    assert "cycle" in str(exc.value).lower()


def test_a_storey_into_add_is_refused():
    with pytest.raises(ValueError) as exc:
        _wall().add(Storey(name="second", elevation=3000))
    assert "invalid child" in str(exc.value)


def test_anchor_on_a_spatial_container_still_raises():
    """It raises — as a bare AttributeError, which names NO remedy.

    Kept as a test rather than fixed, because the fix is not local:
    ``taxonomy.LOCAL_FRAME`` is derived as
    ``tuple(t for t in CONTAINERS if hasattr(t, "anchor"))``, so defining an
    ``anchor`` method that merely refuses would silently enrol ``Site`` and
    ``SpatialElement`` as local-frame containers. Measured: it breaks the
    oracle in ``test_container_children.py`` that pins
    ``has_local_frame(cls) == hasattr(cls, "anchor")``. That derivation is the
    real defect — a semantic fact inferred from method presence — and it is
    filed rather than half-done here. This is why the reference still carries
    that ONE line while the other six are gone.
    """
    with pytest.raises(AttributeError):
        Site(name="site").anchor(_wall(), along=0, up=0)


# --------------------------------------------------------------------------
# The envelope prohibition, now a compile error
# --------------------------------------------------------------------------

def test_an_envelope_over_a_buildup_is_refused():
    """Was prose only. Under-tiled it reports phantom volume; tiled correctly
    it renders as nothing behind a boolean stack. Both silent."""
    wall = Wall(name="south")
    wall.add(Box(name="envelope", start=Point(x=0, y=0, z=0),
                 end=Point(x=6000, y=300, z=2400)))
    wall.add(_leaf("core", 0, 150))
    wall.add(_leaf("skin", 150, 300))
    report = validate_project_report(_project(wall))
    assert report.errors, "the envelope must be refused"
    msg = " ".join(report.errors)
    assert "envelope" in msg
    assert "Drop the body" in msg, f"must name the remedy: {msg}"


def test_the_buildup_without_an_envelope_still_compiles():
    """The falsifier — the refusal must not catch the shape it recommends."""
    wall = Wall(name="south")
    wall.add(_leaf("core", 0, 150))
    wall.add(_leaf("skin", 150, 300))
    assert not validate_project_report(_project(wall)).errors


def test_a_bodied_wall_with_an_ANCHORED_detail_still_compiles():
    """`.add()` vs `.anchor()` again: a detail anchored ON a bodied wall is
    not a layer OF it, and must stay legal."""
    wall = _wall()
    detail = Element(ifc_class="IfcBuildingElementProxy", name="corbel")
    detail.add(Box(name="blk", start=Point(x=0, y=0, z=0),
                   end=Point(x=200, y=100, z=200)))
    wall.anchor(detail, along=1000, up=2000)
    assert not validate_project_report(_project(wall)).errors


def test_a_container_with_nothing_to_derive_from_is_still_refused():
    """The other half of the deleted line — already enforced, kept pinned."""
    report = validate_project_report(_project(Wall(name="south")))
    assert report.errors
    assert "derive" in " ".join(report.errors).lower()


# --------------------------------------------------------------------------
# A boolean operand's own booleans — was a documented silent failure
# --------------------------------------------------------------------------

def _boxes():
    body = Box(name="body", start=Point(x=0, y=0, z=0),
               end=Point(x=3000, y=500, z=2000))          # 3.0 m3
    tool = Box(name="tool", start=Point(x=1000, y=0, z=0),
               end=Point(x=2000, y=500, z=2000))          # 1.0 m3
    return body, tool


def test_a_plain_boolean_operand_still_compiles():
    """The falsifier — the refusal must not catch an ordinary difference."""
    body, tool = _boxes()
    body.difference(tool)
    wall = Wall(name="south")
    wall.add(body)
    assert not validate_project_report(_project(wall)).errors


def test_an_operand_carrying_its_own_cut_is_refused():
    """Measured before the refusal: a 3.0 m3 body cut by a 1.0 m3 tool that
    itself carried a 0.5 m3 cut emitted **2.000 m3 either way**. Honouring the
    operand's cut would give 2.5. The two models were indistinguishable
    without tessellating the file."""
    body, tool = _boxes()
    tool.difference(Box(name="inner", start=Point(x=1000, y=0, z=0),
                        end=Point(x=1500, y=500, z=2000)))
    body.difference(tool)
    wall = Wall(name="south")
    wall.add(body)
    report = validate_project_report(_project(wall))
    assert report.errors, "the discarded operand booleans must be refused"
    msg = " ".join(report.errors)
    assert "DISCARDED" in msg
    assert ".intersection()" in msg, f"must name the real verb: {msg}"


def test_the_fake_intersection_shape_is_refused():
    """`a.difference(b.difference(c))` reads as 'a minus (b minus c)' and
    compiles to plain `a - b`. That is the shape the deleted doc line warned
    about, and the reason `.intersection()` exists."""
    body, tool = _boxes()
    body.difference(tool.difference(Box(name="c", start=Point(x=1200, y=0, z=0),
                                        end=Point(x=1400, y=500, z=2000))))
    wall = Wall(name="south")
    wall.add(body)
    assert validate_project_report(_project(wall)).errors


def test_a_union_on_an_operand_is_HONOURED_and_stays_legal():
    """The counter-example that narrowed this rule.

    ``wall.difference(door.union(arch))`` is how an arched doorway is
    authored, and the generator emits a nested ``IfcBooleanResult(UNION)`` as
    the second operand — measured by
    ``test_primitives_ifc.py::test_arched_doorway_measured_volume``. So the
    discard is VERB-SPECIFIC: an operand's cuts vanish, an operand's unions do
    not. A symmetric-looking rule that refused both would forbid a working,
    tested composition, which is exactly what the first version of this check
    did.
    """
    body, tool = _boxes()
    body.difference(tool.union(Box(name="bump", start=Point(x=2000, y=0, z=0),
                                   end=Point(x=2200, y=500, z=2000))))
    wall = Wall(name="south")
    wall.add(body)
    assert not validate_project_report(_project(wall)).errors
