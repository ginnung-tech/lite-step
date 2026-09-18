"""A layered container and the solid its buildup is cut from name ONE face.. ``LayerSet.outward`` is authority 1 in ``frames._facing_for``
— it beats every derivation — but reading ``layers`` off the element ALONE is
not enough: a layered element's geometry lives on a body CHILD that carries no
``layers`` of its own. So the container answered from the author's
statement and the body answered from the scope centre, and on a top-level
``.add()``ed wall they came apart::

    element                      before      after
    wall (layers=LayerSet(...))    -1          -1
    its Box body                   +1          -1

The anchored half of that split was closed : a layered leaf ANCHORED
into a host hands its own stamped frame down the subtree, so its body inherits
the container's authored answer. A TOP-LEVEL wall is placed by nothing — there
is no reference to inherit and the body falls through to the scope-centre rule
— so no carry can ever reach it and only the authored claim can.

**This is not a second inheritance channel.** Nothing about the container
travels down: ``_outward_of`` reads one authored statement, and it reads it for
the one child that statement is ABOUT. A ``LayerSet.outward`` says "this
buildup faces that way", and a layered element's body IS the buildup — the
solid ``layers.box_layer_slices`` cuts the stack out of. Which child that is
comes from ``layers.layered_body_child``, the slicer's own predicate, rather
than from a second reading of it (test 5): a body deriving the stack's outward
while the stack is cut from a DIFFERENT solid is exactly the drift
``frames.py``'s header describes for ``segment_axes``.

Why it is worth a fix, and why the fixture below is a ``Box`` and a ``Window``:
**the two consumers of ``facing`` read different objects.** ``.anchor()`` bakes
through the CONTAINER's frame (``_bake_child`` reads ``parent._frame``) while
an opening reads the BODY's (``opening_host_frame(_body_of(wall))``). One
``along=1000`` on one 6 m wall therefore resolved to opposite ends of it —
the own measurement, reached by the other route, and on a TOP-LEVEL wall,
which is where every committed model's geometry lives.

What each test is here to catch:

1.  ``test_a_layered_walls_body_names_the_face_its_buildup_states`` is the
    split itself, as the two numbers above.
2.  ``test_a_batten_and_a_window_at_the_same_along_land_together`` is the
    falsifiable one — both world spans asserted, because the bug is precisely
    that the two were derived from frames that disagreed and either span alone
    is satisfiable by a change that moves the other consumer instead.
3.  ``test_the_agreement_survives_the_compile`` re-runs the whole pass through
    ``generate_ifc``, which stamps its own copy — a fix holding only for a
    hand-driven ``stamp_frames`` sequence would still emit the wrong file.
4.  ``test_a_child_that_is_not_the_buildup_still_derives_its_own_facing``
    keeps the reading scoped. A nested wall ``.add()``ed into a layered
    container is not what the buildup statement is about, and it must go on
    answering from the scope centre — which here is the OPPOSITE of the
    container's answer, so the test can fail.
5.  ``test_the_buildup_is_read_through_the_slicers_own_predicate`` is a
    SOURCE-level assertion, and the only guard the sharing has: restating the
    predicate is behaviour-preserving on every fixture here, so no runtime test
    can fail on it. Same precedent as
    ``test_voids.py::test_displacement_reads_the_predicate_instead_of_restating_it``.
6.  ``test_an_anchored_layered_leaf_is_unchanged`` pins: the anchored half
    was already closed by the subtree carry, and the buildup reading must land
    on the same number rather than move it.

``test_facing_inheritance.py``'s section 8 is the matching boundary: a
top-level ``.add()`` still hands NO reference down. It records the agreement,
and its non-buildup child is what keeps it falsifiable.
"""

from __future__ import annotations

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.compiler.frames import (
    _body_of,
    opening_host_frame,
    opening_void_prism,
    resolve_child_anchors,
    stamp_frames,
)
from lite_step.models import Box, Point, Project, Wall, Window
from lite_step.models.material import LayerSet, Material

# ---------------------------------------------------------------------------
# Fixtures
#
# One layered wall, alone on its storey, 120 mm thick and standing at
# y[180, 300]. Alone is deliberate for the same reason the anchored fixtures in
# test_facing_inheritance.py are alone: the scope centre is then the wall's own
# mid-plane, and ``_facing_for``'s tie hands the body ``+1`` — the answer that
# DISAGREES with the authored outward. With the rest of a building on one side
# the two rules agree by accident, which is how this shipped unnoticed.
# ---------------------------------------------------------------------------

_T = 120             # the layered wall's thickness, mm
_Y0, _Y1 = 180, 300  # ... and where it stands, mm
_BATTEN_T = 40


def _layered_wall() -> Wall:
    """A 6 m x 120 mm wall whose buildup faces ``-Y`` by the author's word."""
    wall = Wall(name="solo")
    wall.add(Box(start=Point(x=0, y=_Y0, z=0), end=Point(x=6000, y=_Y1, z=2700),
                 type="wall", name="solo_body"))
    wall.layers = LayerSet(
        name="solo_type", outward=(0.0, -1.0, 0.0),
        layers=[Material(key="Concrete_C25-30", thickness_mm=_T)])
    return wall


def _resolved(wall: Wall) -> Project:
    """The production order: validate -> metres -> stamp -> bake -> stamp."""
    proj = Project(name="716-layered")
    proj.add(wall)
    assert validate_project_report(proj).errors == []
    proj = normalize_project_to_meters(proj) or proj
    stamp_frames(proj)
    resolve_child_anchors(proj)
    stamp_frames(proj)
    return proj


def _wall_of(proj: Project) -> Wall:
    return proj.storeys[0].elements[0]


def _named(elem, name: str):
    for child in (getattr(elem, "_elements", None) or []):
        if getattr(child, "name", None) == name:
            return child
    raise AssertionError(f"no child named {name!r} on {elem.name!r}")


def _span(box) -> dict[str, tuple[float, float]]:
    return {
        "x": (round(box.start.x, 6), round(box.end.x, 6)),
        "y": (round(box.start.y, 6), round(box.end.y, 6)),
        "z": (round(box.start.z, 6), round(box.end.z, 6)),
    }


def _void_x_span(opening, host_body) -> tuple:
    """The world x-span of ``opening``'s void, through the shared derivation.

    ``opening_host_frame(_body_of(wall))`` is the exact chain both IFC backends
    walk to place a hole. Reading the BODY there — rather than the container
    ``.anchor()`` bakes through — is what made the two answers separable.
    """
    frame = opening_host_frame(host_body)
    assert frame is not None, "the fixture's body must bear an opening frame"
    prism = opening_void_prism(opening, frame)
    assert prism is not None, "the fixture's opening must cut a prism"
    xs = [round(corner[0], 6) for corner in prism.corners]
    return (min(xs), max(xs))


# ---------------------------------------------------------------------------
# 1. The split itself
# ---------------------------------------------------------------------------


def test_a_layered_walls_body_names_the_face_its_buildup_states() -> None:
    """Container ``-1``, body ``-1``.

    The scope-centre answer is checked to DISAGREE with the authored one before
    the assertion that matters, because a fixture in which they agree would
    pass with the fix deleted.
    """
    proj = _resolved(_layered_wall())
    wall = _wall_of(proj)
    body = _named(wall, "solo_body")

    # -- the fixture disagrees, which is the precondition -------------------
    assert _span(body)["y"] == (0.18, 0.3), "the wall stands where it is placed"
    scope_center_y = 0.24                        # the lone wall's mid-plane
    self_derived = 1 if abs(0.3 - scope_center_y) >= abs(0.18 - scope_center_y) else -1
    assert self_derived == 1, (
        "the scope-centre rule must answer +1 on this body (the tie goes to "
        "the max side), or this test passes for the wrong reason")

    # -- and the author's statement is the one that stands ------------------
    assert wall._frame.facing == -1, (
        "the container states outward=(0,-1,0) and an authored outward is "
        "authority 1")
    assert body._frame.facing == -1, (
        f"the body reports facing={body._frame.facing:+d}. It IS the buildup "
        f"the LayerSet describes — the solid box_layer_slices cuts the stack "
        f"out of — so it cannot name the other face. Before that it derived "
        f"+1 from the scope centre and disagreed with the wall it is")
    assert body._frame.across == wall._frame.across, "same plane, same across"


# ---------------------------------------------------------------------------
# 2. What the split was worth, on both consumers of ``facing``
# ---------------------------------------------------------------------------


def test_a_batten_and_a_window_at_the_same_along_land_together() -> None:
    """One wall, one ``along=1000``, one answer.

    The wall carries BOTH consumers at the same anchor — a ``Box``, which the
    bake positions through the CONTAINER's frame, and a ``Window``, whose void
    is derived from the BODY's::

        element            before                after
        solo (container)   facing -1             facing -1
        solo_body (body)   facing +1             facing -1
        batten             x[1.000, 1.300]       x[1.000, 1.300]
        window void        x[4.400, 5.000]       x[1.000, 1.600]

    3.4 m apart on one 6 m wall, and nothing raised. BOTH spans are asserted,
    because either alone is satisfiable by a change that moves the other
    consumer instead: the requirement is that they agree.
    """
    wall = _layered_wall()
    wall.anchor(Box(start=Point(x=0, y=0, z=0),
                    end=Point(x=300, y=_BATTEN_T, z=300), name="batten"),
                along=1000, up=1000)
    wall.anchor(Window(width=600, height=600, name="w0"), along=1000, up=1000)
    proj = _resolved(wall)

    wall = _wall_of(proj)
    batten = _span(_named(wall, "batten"))
    void_x = _void_x_span(_named(wall, "w0"), _body_of(wall))

    assert _body_of(wall) is _named(wall, "solo_body"), (
        "the opening chain must reach the same solid the buildup statement "
        "governs, or the two spans below are about different objects")
    assert batten["x"] == (1.0, 1.3), (
        f"the batten is at x{batten['x']} — the container's facing=-1 leaves "
        f"the run unreversed, so along=1000 is 1000 mm in from x=0")
    assert batten["y"] == (0.18, 0.22), (
        f"the batten is at y{batten['y']} — anchored at inset=0 it belongs on "
        f"the face the author called outward, y=0.180")
    assert void_x == (1.0, 1.6), (
        f"the window's void is at x{void_x}. It must start from the SAME end "
        f"as the batten (x=1.000): both were authored at along=1000 on the "
        f"same wall. Revert the fix and the void reads the body's re-derived "
        f"+1 instead — x[4.400, 5.000], 3.4 m away")
    assert min(void_x) == min(batten["x"]), (
        "the two consumers must measure `along` from ONE end — that is the "
        "whole, and it is falsifiable without either literal above")


def test_the_agreement_survives_the_compile() -> None:
    """The stamp is not the artefact — the file is.

    ``generate_ifc`` re-runs the frame pass on its own copy, so a fix holding
    only for a hand-driven ``stamp_frames`` sequence would still emit the void
    at the wrong end.
    """
    from lite_step.ifc.generator import generate_ifc

    wall = _layered_wall()
    wall.anchor(Box(start=Point(x=0, y=0, z=0),
                    end=Point(x=300, y=_BATTEN_T, z=300), name="batten"),
                along=1000, up=1000)
    wall.anchor(Window(width=600, height=600, name="w0"), along=1000, up=1000)
    proj = Project(name="716-compile")
    proj.add(wall)
    normalized = normalize_project_to_meters(proj)

    result = generate_ifc(normalized)
    assert result.success, result.error

    wall = _wall_of(normalized)
    assert _named(wall, "solo_body")._frame.facing == -1
    assert _span(_named(wall, "batten"))["x"] == (1.0, 1.3)
    assert _void_x_span(_named(wall, "w0"), _body_of(wall)) == (1.0, 1.6)


# ---------------------------------------------------------------------------
# 3. The reading is scoped to the buildup, and to nothing else
# ---------------------------------------------------------------------------


def test_a_child_that_is_not_the_buildup_still_derives_its_own_facing() -> None:
    """A nested wall inside a layered container takes nothing from it.

    ``pier`` is ``.add()``ed into the layered wall and is NOT its buildup — the
    stack is cut from ``solo_body`` and from nothing else — so the author's
    statement is not about it and the scope-centre rule remains the only thing
    that speaks. Here that answers ``+1`` while the container answers ``-1``,
    so widening the reading into an inheritance reports ``-1`` and this fails.

    ``far`` is scenery: it drags the storey scope centre out to y=-1.79 so that
    the body and the pier BOTH self-derive ``+1``. Without it the two straddle
    their own union's centre and one of them reads ``-1`` for a reason that has
    nothing to do with this change.
    """
    wall = _layered_wall()
    pier = Wall(name="pier")
    pier.add(Box(start=Point(x=0, y=_Y1, z=0), end=Point(x=6000, y=420, z=2700),
                 type="wall", name="pier_body"))
    wall.add(pier)

    far = Wall(name="far")
    far.add(Box(start=Point(x=0, y=-4000, z=0),
                end=Point(x=6000, y=-3880, z=2700), type="wall",
                name="far_body"))

    proj = Project(name="716-scope")
    proj.add(wall)
    proj.add(far)
    assert validate_project_report(proj).errors == []
    proj = normalize_project_to_meters(proj) or proj
    stamp_frames(proj)
    resolve_child_anchors(proj)
    stamp_frames(proj)

    wall = _wall_of(proj)
    pier = _named(wall, "pier")

    assert wall._frame.facing == -1, "the container states its own outward"
    assert _named(wall, "solo_body")._frame.facing == -1, (
        "its BUILDUP does take the statement — otherwise this test would pass "
        "with the whole change deleted")
    assert pier._frame.across == wall._frame.across, (
        "the pier runs east-west like its container, so the two `across` axes "
        "are parallel and a leaked reading would be readable rather than "
        "falling through facing_from's perpendicular None")
    assert pier._frame.facing == 1, (
        f"the pier reports facing={pier._frame.facing:+d} — it is not the "
        f"buildup, so it must come from the scope centre. Read the outward off "
        f"any ancestor instead of off the buildup relation and this is -1")


def test_the_buildup_is_read_through_the_slicers_own_predicate() -> None:
    """A SOURCE-level assertion, and the only guard the sharing has.

    Which child is the buildup is a policy with three parts — exactly one
    non-anchored prism child, or none — and ``layers.layered_body_child`` is
    where the SLICER decides it. Restating it in ``frames._outward_of`` passes
    every fixture in this file, so no runtime test can fail on it; what comes
    back is a body free to read the stack's outward while the stack is cut from
    a different solid. ``frames._body_of``, the nearby lookalike, is NOT that
    predicate: it returns the first geometry-bearing child including an
    ANCHORED one, so a cornice hung on a layered wall would be read as the
    wall's buildup.
    """
    import inspect

    from lite_step.compiler import frames

    source = inspect.getsource(frames._outward_of)
    assert "layered_body_child" in source, (
        "_outward_of must READ the slicer's predicate, not restate which child "
        "is the buildup")
    assert "_anchor_resolved" not in source and "is_prism" not in source, (
        "restating the predicate's parts is the drift this assertion exists "
        "to prevent")


# ---------------------------------------------------------------------------
# 4. The anchored half is landed on, not moved
# ---------------------------------------------------------------------------


def test_an_anchored_layered_leaf_is_unchanged() -> None:
    """The anchored half was already closed by the subtree carry.

    A layered leaf anchored into a host reaches the same body facing by TWO
    routes now — the carry from the container's already-stamped frame, and
    the buildup statement itself. They must agree, or this change moved a case
    it was not about. The leaf states an outward OPPOSITE to what it would
    inherit, so the number is the authored one on both routes and the test
    would report ``+1`` if the buildup reading were lost.
    """
    host = Wall(name="south")
    host.add(Box(start=Point(x=0, y=-300, z=0), end=Point(x=6000, y=0, z=2700),
                 type="wall", name="body"))
    leaf = Wall(name="core")
    leaf.add(Box(start=Point(x=0, y=180, z=0), end=Point(x=6000, y=300, z=2700),
                 type="wall", name="core_body"))
    leaf.layers = LayerSet(
        name="core_type", outward=(0.0, -1.0, 0.0),
        layers=[Material(key="Concrete_C25-30", thickness_mm=_T)])
    host.anchor(leaf, along=0, up=0)

    proj = Project(name="716-anchored")
    proj.add(host)
    assert validate_project_report(proj).errors == []
    proj = normalize_project_to_meters(proj) or proj
    stamp_frames(proj)
    resolve_child_anchors(proj)
    stamp_frames(proj)

    host = _wall_of(proj)
    leaf = _named(host, "core")

    assert host._frame.facing == 1, "the lone host's exterior is +Y by the tie"
    assert leaf._frame.facing == -1, (
        "an authored outward beats the inherited facing — the authority "
        "order, untouched")
    assert _named(leaf, "core_body")._frame.facing == -1, (
        "the body agrees, by the carry and by the buildup statement "
        " alike — this change must land on that number, not move it")
