"""``.add(carve=)`` — who loses material, said at the call that creates the overlap.

When two solids overlap, the inferred pass carves one of them, and **tree order
decides which**: the last-added element cuts, the incumbent loses material. That
is invisible in the source, invisible in the output, and flips if two ``.add()``
calls are swapped.

``carve=`` names the direction for the overlaps THIS add creates:

===========  ==================================================
``"other"``  the element already there loses (the default)
``"self"``   the element being added yields instead
``"none"``   neither — overlap is preserved
===========  ==================================================

An ARGUMENT rather than a field or a verb, and the two things that buys are
tested here rather than argued: it cannot be too late (``Product`` snapshots its
source, so a verb called afterwards was a silent no-op) and it can differ per
placement (a flag riding on the element cannot).
"""
from __future__ import annotations

import logging

import pytest

from lite_step.compiler.displacement import apply_displacement, carve_pairs
from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.models import Box, Point, Project, Slab
from lite_step.models.project import Storey


def _overlapping(a_end=6000, b_start=3000):
    """Two plates overlapping between ``b_start`` and ``a_end``."""
    a = Box(name="plate_a", start=Point(x=0, y=0, z=0),
            end=Point(x=a_end, y=4000, z=300))
    b = Box(name="plate_b", start=Point(x=b_start, y=0, z=0),
            end=Point(x=9000, y=4000, z=300))
    return a, b


def _compile(*adds):
    """``adds`` is a list of ``(element, carve)`` — added in order."""
    proj = Project(name="carve")
    storey = Storey(name="ground", elevation=0)
    for elem, carve in adds:
        if carve is None:
            storey.add(elem)
        else:
            storey.add(elem, carve=carve)
    proj.add_storey(storey)
    apply_displacement(proj)
    return proj


def _pairs(proj):
    return [(loser, winner) for loser, winner, _m in carve_pairs(proj)]


# ── the default, stated and unstated ─────────────────────────────────────────

def test_the_default_is_the_incumbent_loses():
    a, b = _overlapping()
    proj = _compile((a, None), (b, None))
    assert _pairs(proj) == [("plate_a", "plate_b")]


def test_carve_other_is_the_default_stated():
    """Same geometry, same pair, same direction — stating it changes nothing.

    That equality is the whole claim of the migration: an author can annotate
    an existing model without moving a millimetre.
    """
    a, b = _overlapping()
    stated = _pairs(_compile((a, None), (b, "other")))
    a2, b2 = _overlapping()
    unstated = _pairs(_compile((a2, None), (b2, None)))
    assert stated == unstated == [("plate_a", "plate_b")]


# ── "self" inverts it ────────────────────────────────────────────────────────

def test_carve_self_makes_the_newcomer_yield():
    a, b = _overlapping()
    proj = _compile((a, None), (b, "self"))
    assert _pairs(proj) == [("plate_b", "plate_a")], (
        "carve=\"self\" must invert the pair — the element being added yields "
        "to what is already there"
    )


def test_carve_self_is_read_off_the_newcomer_not_the_incumbent():
    """The incumbent's own declaration governed the adds IT was new in.

    Declaring on `a` must not reach the pair where `b` is the newcomer — a
    child has no say over an add it is not part of, which is what makes the
    argument's scope readable at the call site.
    """
    a, b = _overlapping()
    proj = _compile((a, "self"), (b, None))
    assert _pairs(proj) == [("plate_a", "plate_b")]


def test_carve_self_on_an_enclosed_element_warns_and_empties_it(caplog):
    """Loud, not fatal — an impossible state is user error, not code error.

    The author asked an element to yield to something that encloses it, which
    is asking for nothing. The compiler answers "here is nothing", and SAYS so,
    because the failure mode is an element that simply is not there.
    """
    big = Box(name="big", start=Point(x=0, y=0, z=0),
              end=Point(x=9000, y=4000, z=300))
    small = Box(name="small", start=Point(x=1000, y=1000, z=50),
                end=Point(x=2000, y=2000, z=250))
    with caplog.at_level(logging.WARNING):
        proj = _compile((big, None), (small, "self"))
    text = caplog.text
    assert "small" in text and "big" in text, text
    assert "EMPTIED" in text, text
    # and it still compiled — the direction was honoured, not refused
    assert _pairs(proj) == [("small", "big")]


# ── "none" exempts the pair ──────────────────────────────────────────────────

def test_carve_none_exempts_the_pair():
    a, b = _overlapping()
    proj = _compile((a, None), (b, "none"))
    assert _pairs(proj) == []


def test_carve_none_sets_the_flag_the_pass_actually_reads():
    """Asserted on ``_no_carve`` rather than only on the outcome, because that
    flag is what ``displacement._walk`` PRUNES on — a ``"none"`` that produced
    no pairs for some other reason (nothing overlapped, the walk never got
    there) would read the same from the outside.

    There is one spelling: ``carve="none"``. The removal of the second is
    pinned in ``test_carve_exemption.py``.
    """
    a, b = _overlapping()
    _compile((a, None), (b, "none"))
    assert b._no_carve is True


def test_carve_covers_the_whole_subtree():
    """Same scope as ``placement=``: an exempt assembly
    whose own parts kept carving would let the bricks inside a bracket eat the
    wall the bracket hangs off."""
    plate = Box(name="plate", start=Point(x=0, y=0, z=0),
                end=Point(x=6000, y=4000, z=300))
    holder = Slab(name="holder")
    inner = Box(name="inner", start=Point(x=3000, y=0, z=0),
                end=Point(x=9000, y=4000, z=300))
    holder.add(inner)
    proj = _compile((plate, None), (holder, "none"))
    assert _pairs(proj) == []

    # Asserted on ``inner._no_carve`` until the direction stopped being COPIED
    # onto descendants — see the scope tests at the bottom of this file. The
    # flag now sits on the element the author named, and ``_walk`` prunes
    # there, so a descendant carrying it was never the guarantee: it was the
    # mechanism, and asserting the mechanism is what let a copy taken at the
    # wrong MOMENT pass as a scope that holds.
    assert holder._no_carve is True
    assert inner._no_carve is False


# ── the argument is per-ADD, which the flag cannot be ────────────────────────

def test_the_same_source_can_be_exempt_in_one_place_and_not_another():
    """Inexpressible while the flag rides on the element — it is snapshotted
    with it. This is one of the two things the argument buys over the verb."""
    plate = Box(name="plate", start=Point(x=0, y=0, z=0),
                end=Point(x=6000, y=4000, z=300))
    exempt = Box(name="exempt", start=Point(x=3000, y=0, z=0),
                 end=Point(x=9000, y=4000, z=300))
    carving = Box(name="carving", start=Point(x=3000, y=2000, z=0),
                  end=Point(x=9000, y=6000, z=300))
    proj = _compile((plate, None), (exempt, "none"), (carving, "other"))
    losers = {loser for loser, _w in _pairs(proj)}
    assert "exempt" not in losers        # exempt neither cuts nor is cut
    assert "plate" in losers             # the other add still carves


# ── refusals and reporting ───────────────────────────────────────────────────

def test_an_unknown_direction_is_refused_at_the_call():
    plate = Box(name="p", start=Point(x=0, y=0, z=0),
                end=Point(x=1000, y=1000, z=100))
    storey = Storey(name="ground", elevation=0)
    with pytest.raises(ValueError, match="is not a direction"):
        storey.add(plate, carve="host")      # the name an earlier draft used


def test_undeclared_pairs_are_counted_separately():
    """The residual an author can drive to zero.

    A count rather than a warning per pair: the hospital has ~4,000 inferred
    carves, and a line each is the 12 KB-of-stderr failure the carve report
    already had to be capped for.
    """
    a, b = _overlapping()
    undeclared = _compile((a, None), (b, None))
    assert undeclared._carve_undeclared == 1

    a2, b2 = _overlapping()
    declared = _compile((a2, None), (b2, "other"))
    assert declared._carve_undeclared == 0


# ── the scope is the SUBTREE, and it is READ, not copied ─────────────────────
#
# ``carve=`` covers the whole subtree, like ``placement=`` and for the same
# reason: an exempt assembly whose own parts kept carving would let the bricks
# inside a bracket eat the wall the bracket hangs off. The first implementation
# bought that by COPYING the direction onto every descendant at ``.add()``
# time, and a copy is only as good as the moment it was taken. Both of the
# following were measured against that implementation and both failed.

def _assembly_over_a_plate(direction, late: bool):
    """A plate, then an assembly added with ``direction`` whose overlapping part
    is added either BEFORE the assembly is added (``late=False``) or after."""
    plate = Box(name="plate_a", start=Point(x=0, y=0, z=0),
                end=Point(x=6000, y=4000, z=300))
    asm = Slab(name="asm")
    part = Box(name="plate_b", start=Point(x=3000, y=0, z=0),
               end=Point(x=9000, y=4000, z=300))
    if not late:
        asm.add(part)

    proj = Project(name="carve")
    storey = Storey(name="ground", elevation=0)
    storey.add(plate)
    storey.add(asm, carve=direction)
    if late:
        asm.add(part)                # AFTER the add that carried the argument
    proj.add_storey(storey)
    apply_displacement(proj)
    return _pairs(proj)


@pytest.mark.parametrize("direction", ["other", "self", "none"])
def test_a_part_added_after_the_add_is_still_inside_the_scope(direction):
    """``asm.add(part)`` written BELOW ``storey.add(asm, carve=…)`` must mean
    the same as written above it.

    The copy could not reach a part that did not exist yet, so the part
    silently reverted to the default direction. Measured on the failing
    implementation, only ``"self"`` actually broke — ``"none"`` was immune by
    accident (``_walk`` prunes that subtree at walk time and never reads the
    descendants) and ``"other"`` was masked because it names the default
    anyway. So the one direction that CHANGES anything was the one that lost
    its meaning, and the two that could have exposed it could not.

    Parametrized over all three so a future change cannot quietly re-acquire
    the same blind spot in a different direction.
    """
    assert _assembly_over_a_plate(direction, late=True) == \
           _assembly_over_a_plate(direction, late=False)


def test_the_scope_really_does_reach_a_part_that_is_never_named():
    """The control for the test above: without it, "the two agree" could mean
    the scope works OR that neither reached the part at all."""
    assert _assembly_over_a_plate("none", late=False) == []
    assert _assembly_over_a_plate("self", late=False) == \
           [("plate_b", "plate_a")]          # inverted: the newcomer yields
    assert _assembly_over_a_plate("other", late=False) == \
           [("plate_a", "plate_b")]


def test_a_nested_declaration_outranks_the_one_enclosing_it():
    """Nearest declaration wins.

    The copy walked the subtree unconditionally, so ``storey.add(asm,
    carve="self")`` rewrote a ``carve="other"`` the author had written one
    level deeper — an enclosing add overruling an explicit statement about a
    specific element. Every other scoped thing in this DSL reads inside-out;
    this now does too.
    """
    plate = Box(name="plate_a", start=Point(x=0, y=0, z=0),
                end=Point(x=6000, y=4000, z=300))
    asm = Slab(name="asm")
    part = Box(name="plate_b", start=Point(x=3000, y=0, z=0),
               end=Point(x=9000, y=4000, z=300))
    asm.add(part, carve="other")             # inner: this part carves the plate
    proj = Project(name="carve")
    storey = Storey(name="ground", elevation=0)
    storey.add(plate)
    storey.add(asm, carve="self")            # outer: the assembly yields
    proj.add_storey(storey)
    apply_displacement(proj)

    assert part._carve == "other", "the enclosing add overwrote the inner one"
    assert _pairs(proj) == [("plate_a", "plate_b")]   # the INNER direction


# ── the argument on .anchor(), where it reaches a DIFFERENT pass ─────────────
#
# Deleting ``.no_carve()`` would have REMOVED capability without this. An
# anchored child is already out of the inferred-carve pairing — its position is
# a relationship to its host, not an accidental overlap — so ``.add(carve=)``
# could not reach it, and what the verb was actually doing on four fixtures was
# exempting an anchored detail from the OPENING/VOID carve: a sill or lintel
# meant to stand in the reveal. That is the canonical case for the exemption
# and it is anchored, not added.

def _post_cuts_in_the_reveal(carve=None):
    """A wall with a window, a post standing in the opening, and the number of
    boolean operands the opening pushed into that post.

    Compiled rather than merely displaced: the opening carve reads a stamped,
    metre-space host FRAME, and ``generate_ifc`` is what runs the frame + anchor
    passes that stamp it. Without the compile ``opening_host_frame`` returns
    ``None`` and the pass returns silently — which the baseline test below
    caught, and which is exactly how an exemption test passes vacuously.
    """
    from lite_step.ifc.generator import generate_ifc
    from lite_step.models import Wall, Window

    wall = Wall(name="south")
    wall.add(Box(name="body", type="wall",
                 start=Point(x=0, y=-300, z=0), end=Point(x=6000, y=0, z=2700)))
    post = Box(name="post", start=Point(x=0, y=100, z=0),
               end=Point(x=50, y=150, z=2700))
    wall.anchor(post, along=2100, up=0, carve=carve)
    wall.opening(Window(width=1200, height=1400, name="w0"),
                 along=1500, up=900)

    proj = Project(name="reveal")
    proj.add(wall)
    norm = normalize_project_to_meters(proj) or proj
    assert generate_ifc(norm).success
    compiled_post = next(e for e in norm.storeys[0].elements[0]._elements
                         if getattr(e, "name", None) == "post")
    return len(compiled_post._cuts or [])


def test_an_anchored_detail_in_the_reveal_is_carved_by_default():
    """The baseline. Without it the exemption below would pass on a fixture
    where the hole reaches nothing — and it did, twice, while this fixture was
    being written."""
    assert _post_cuts_in_the_reveal() == 1


def test_anchor_carve_none_exempts_the_detail_from_the_OPENING_carve():
    assert _post_cuts_in_the_reveal(carve="none") == 0


def test_anchor_refuses_an_unknown_direction_like_add_does():
    """One argument, one vocabulary. A value accepted on one verb and refused
    on the other is two arguments wearing one name."""
    from lite_step.models import Wall

    wall = Wall(name="south")
    with pytest.raises(ValueError, match="is not a direction"):
        wall.anchor(Box(name="p", start=Point(x=0, y=0, z=0),
                        end=Point(x=100, y=100, z=100)),
                    along=0, up=0, carve="host")


@pytest.mark.parametrize("direction", ["other", "self"])
def test_anchor_accepts_the_other_two_directions_and_they_are_INERT(direction):
    """Accepted, not refused, and they change nothing here.

    The solid pairing they name does not happen for an anchored child at all,
    so there is no direction to state. Refusing them would make one argument
    mean different things on the two verbs — the author would have to remember
    which subset each accepts, which is the kind of asymmetry this codebase
    keeps finding as a bug. Pinned so "inert" stays a decision rather than an
    accident nobody measured.
    """
    assert _post_cuts_in_the_reveal(direction) == _post_cuts_in_the_reveal()
