"""An anchored child re-derives its axes and INHERITS its facing.

``.anchor(child)`` says "you are placed by my frame". That stopped
the child from polluting the frame that placed it; this is the other half —
what the child derives for ITSELF once it stands in world coordinates.

Two of the three answers are genuinely its own. ``along``/``across`` are
geometry: a sub-wall running ``+Y`` runs ``+Y`` whatever its host does, and a
grandchild anchored into it must measure along ITS axis, not the host's. The
extents are its own body. ``facing`` is not, and that is the whole of this
change: **facing is a statement about what lies OUTSIDE the element**, and for
something placed inside a host, outside is the host's outside. Re-deriving it
from a placed AABB and the storey scope centre answers a different question —
"which of my two faces is farther from the middle of the building" — and the
two answers come apart exactly where composition happens:

*   the INNER leaf of a wall is, by construction, the half nearer the building
    centre, so it reads its own outer face as its inner one;
*   an assembly turned 90 degrees reaches out past its host, so it reads the
    overhang as the exterior (``test_anchor_containers``' rotated chain).

Either way nothing raises. The leaf compiles, validates and renders; only
everything anchored INTO it stands on the wrong face.

What each test is here to catch:

1.  ``test_an_anchored_leaf_keeps_its_hosts_facing`` is the falsifiable one.
    The fixture is built so the leaf's own placed AABB and the storey scope
    centre BOTH say ``+1`` while the host says ``-1``… strictly, so that the
    scope-centre rule and the host disagree — otherwise the test would pass on
    a fix that does nothing. Revert the inheritance in ``_facing_for`` and the
    batten anchored into that leaf moves 80 mm, from the leaf's outer face to
    its inner one — two batten thicknesses — and 3.7 m along the wall.
2.  ``test_the_697_nested_core_lands_where_the_host_frame_says`` is issue
    §1's headline measurement, as a number. It is a PIN rather than a
    falsifier: the span was never wrong (the issue read ``y[0, 0.180]`` as
    "mirrored"; it is the core standing proud of the outer face the host's
    ``facing`` names, which is what ``y ∈ [-180, 0]`` asks for in a frame whose
    ``+y`` runs inward). What test 1 adds is that the core's OWN frame now
    names the same face.
3.  ``test_a_perpendicular_sub_element_re_derives_from_the_scope_centre``
    exercises the ``None`` fall-through rather than merely asserting it
    exists. A return wall running north-south takes nothing from a host
    running east-west — the host's ``across`` is at right angles to its own
    and says nothing about which side is out — so it derives its own answer,
    and that answer is the OPPOSITE of the host's. Make ``facing_from``
    degrade to the reference's raw facing (which is what its other caller,
    ``opening_out_sign``, chooses to do) and this test reports ``+1``.
4.  ``test_opening_out_sign_is_unchanged_by_the_extraction`` pins the shared
    helper's other reader against the expression it was lifted out of, over
    both facings and both run axes.
5.  ``test_an_authored_outward_beats_an_inherited_facing`` keeps the authority
    order. ``LayerSet.outward`` is the author stating which way THIS buildup
    faces; it is not a repetition of the host's claim and must beat it.

The end-to-end behaviour of an opening's frame is already pinned by
``test_opening_local_frame.py`` (the determinant of every placement) and
``test_anchor_openings.py`` (the boolean delta of a void); test 4 stays a unit
table so it can cover host shapes those fixtures do not build.

Section 6 is — **how far down the reference goes**. The
inheritance above stopped AT the anchored node, so an anchored container and
its ``.add()``ed body could answer differently, and the two consumers read
different ones of them: ``.anchor()`` bakes through the CONTAINER's frame,
while an opening reads the BODY's
(``opening_host_frame(_body_of(wall))``). Same leaf, same ``along=1000``,
3.7 m apart. The reference now travels down the whole anchored SUBTREE, and
those tests fix where it stops:

6.  ``test_a_batten_and_a_window_at_the_same_along_land_together`` is the own fixture and the falsifiable one — both world spans asserted, because
    the bug is precisely that the two were derived from frames that disagreed.
7.  ``test_the_carry_is_transitive_not_one_extra_hop`` puts a third generation
    below the anchored node. A one-hop widening passes test 6 and fails this.
8.  ``test_a_top_level_added_element_still_derives_its_own_facing`` is THE
    BOUNDARY, asserted rather than assumed: nothing placed a top-level
    ``.add()``ed subtree, so it derives from the scope centre as it always
    has. The fixture is a LAYERED wall carrying a nested wall that is NOT its
    buildup, and the nested wall is what makes the boundary falsifiable: it
    self-derives the OPPOSITE of the container's authored answer, so a
    reference leaking past a top-level ``.add()`` moves it.

    **The boundary is closed through the authored statement, not the carry.**
    Reading the layered container's BODY as the thing the carry must not reach
    would leave the container-vs-body split open. It is closed instead by: an authored ``LayerSet.outward`` is a statement about
    the BUILDUP and the body IS the buildup, so the body now reads the same
    authored claim the container does (``frames._outward_of``). The body is
    therefore not a witness to the boundary — it agrees with its
    container for a reason that has nothing to do with inheritance — and the
    non-buildup nested wall took over that job. What the test still pins is
    unchanged: a top-level ``.add()`` hands NO reference down.
    ``test_layered_facing.py`` carries the own measurement.
9.  ``test_a_perpendicular_child_inside_an_anchored_subtree_still_falls_through``
    is test 3's case one generation on: the carry must not bypass
    ``facing_from``'s ``None``.
10. ``test_the_carry_stops_at_an_opening`` pins the other stop. A
    ``Window``/``Door`` is stamped ``None`` (``frame_basis_points`` yields
    nothing for one) and its children are authored in OPENING-LOCAL
    coordinates, so a world-space facing restated on one of their ``across``
    axes would be an answer about a different coordinate system. Passing the
    reference through a frame-less link instead was measured at 110 further
    corpus frames, every one inside a ``Window`` or ``Door``.
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.compiler.frames import (
    ElementFrame,
    _body_of,
    facing_from,
    opening_host_frame,
    opening_out_sign,
    opening_void_prism,
)
from lite_step.models import Box, Point, Project, Wall, Window
from lite_step.models.elements import Element
from lite_step.models.material import LayerSet, Material

# ---------------------------------------------------------------------------
# Fixtures
#
# One wall, alone on its storey. That is deliberate: the scope centre is then
# the wall's own mid-plane, which is the only configuration in which the
# scope-centre rule can disagree with the host about a leaf INSIDE the wall.
# With the rest of a building on one side every sub-element of the wall sits on
# the same side of a distant centroid and the two rules agree by accident —
# which is exactly how this shipped unnoticed.
#
# The lone wall's facing is +1 (the exterior is +Y) by ``_facing_for``'s tie,
# the same tie ``test_opening_carves_details`` and ``test_anchor_openings``
# quote.
# ---------------------------------------------------------------------------

_HOST_T = 300        # the host wall's thickness, mm
_LEAF_T = 120        # the inner leaf's
_BATTEN_T = 40       # the batten anchored onto the leaf


def _host() -> Wall:
    wall = Wall(name="south")
    wall.add(Box(start=Point(x=0, y=-_HOST_T, z=0),
                 end=Point(x=6000, y=0, z=2700), type="wall", name="body"))
    return wall


def _inner_leaf(layers: LayerSet | None = None) -> Wall:
    """The INNER 120 mm of a 300 mm wall, with a batten on its outer face.

    Authored in the host's frame: ``inset`` 180 → 300 is the far side of a
    300 mm wall, so this is the leaf whose own placed AABB is nearest the
    building centre — the half that reads its outer face as ``-1``.
    """
    leaf = Wall(name="core")
    leaf.add(Box(start=Point(x=0, y=_HOST_T - _LEAF_T, z=0),
                 end=Point(x=6000, y=_HOST_T, z=2700), type="wall",
                 name="core_body"))
    leaf.anchor(Box(start=Point(x=0, y=0, z=0),
                    end=Point(x=300, y=_BATTEN_T, z=300), name="batten"),
                along=1000, up=1000)
    if layers is not None:
        leaf.layers = layers
    return leaf


def _return_wall() -> Wall:
    """A buttress running NORTH-SOUTH off an east-west host.

    Its own AABB is longer across the host's thickness than along its run, so
    ``_axes_from_aabb`` gives it ``along=(0,1,0)`` and ``across=(-1,0,0)`` —
    perpendicular to the host's ``across``, which is the case the inheritance
    has no opinion about.
    """
    pier = Wall(name="pier")
    pier.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=200, y=900, z=2700),
                 type="wall", name="pier_body"))
    return pier


def _resolved(child: Wall, along: int = 0) -> Project:
    """The production order: validate -> metres -> stamp -> bake -> stamp."""
    from lite_step.compiler.frames import resolve_child_anchors, stamp_frames

    host = _host()
    host.anchor(child, along=along, up=0)
    proj = Project(name="facing-inheritance")
    proj.add(host)
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


# ---------------------------------------------------------------------------
# 1. The measurement: an anchored leaf keeps the face its host named
# ---------------------------------------------------------------------------


def test_an_anchored_leaf_keeps_its_hosts_facing() -> None:
    """The leaf's own AABB says ``-1``; its host says ``+1``; the host wins.

    Both self-derivations are checked to disagree with the host BEFORE the
    assertion that matters, because a fixture in which they agree would pass
    with the inheritance deleted.
    """
    proj = _resolved(_inner_leaf())
    host = _wall_of(proj)
    leaf = _named(host, "core")

    # -- the fixture disagrees, which is the precondition ------------------
    assert host._frame.facing == 1, (
        "the lone wall's exterior is +Y by the tie rule; the fixture is built "
        "on that")
    body = _named(leaf, "core_body")
    assert _span(body)["y"] == (-0.3, -0.18), (
        "the leaf must be the INNER 120 mm of the 300 mm wall — the half "
        "nearer the storey scope centre, or the two rules do not disagree")
    scope_center_y = -0.15                       # the lone wall's mid-plane
    self_derived = 1 if abs(-0.18 - scope_center_y) >= abs(-0.3 - scope_center_y) else -1
    assert self_derived == -1, (
        "the scope-centre rule must answer -1 on this leaf, or this test "
        "passes for the wrong reason")

    # -- and the host's answer is the one that stands ----------------------
    assert leaf._frame.facing == 1, (
        f"the leaf reports facing={leaf._frame.facing:+d} — it derived its "
        f"exterior from its own placed AABB, so the inner half of a wall "
        f"calls its own inner face the outside")
    assert leaf._frame.across == host._frame.across, "same plane, same across"
    assert leaf._frame.thickness == pytest.approx(_LEAF_T / 1000.0), (
        "the extents stay the leaf's OWN — only facing is inherited")

    # -- what that is worth, one generation down ---------------------------
    batten = _span(_named(leaf, "batten"))
    assert batten["y"] == (-0.22, -0.18), (
        f"the batten is at y{batten['y']} — anchored at inset=0 it belongs on "
        f"the leaf's OUTER face (y=-0.18). Without the inherited facing it "
        f"lands at y[-0.30, -0.26], on the opposite face of the leaf: 80 mm "
        f"away, two batten thicknesses, on the room side of a 120 mm leaf")
    assert batten["x"] == (4.7, 5.0), (
        f"the batten is at x{batten['x']} — facing also reverses the run "
        f"direction (``outer_face_axis``), so a flipped leaf puts along=1000 "
        f"at the other end: x[1.0, 1.3], a 3.7 m jump")
    assert batten["z"] == (1.0, 1.3), "up is unsigned and must not move"


def test_the_inherited_facing_survives_the_compile() -> None:
    """The stamp is not the artefact — the file is.

    ``generate_ifc`` re-runs the whole pass on its own copy, so a fix that
    only held for a hand-driven ``stamp_frames`` sequence would still emit the
    old geometry. Asserted on the tree the backend serialized.
    """
    from lite_step.ifc.generator import generate_ifc

    host = _host()
    host.anchor(_inner_leaf(), along=0, up=0)
    proj = Project(name="facing-compile")
    proj.add(host)
    normalized = normalize_project_to_meters(proj)

    result = generate_ifc(normalized)
    assert result.success, result.error

    leaf = _named(_wall_of(normalized), "core")
    assert _span(_named(leaf, "batten"))["y"] == (-0.22, -0.18)


# ---------------------------------------------------------------------------
# 2. Issue §1, as a number
# ---------------------------------------------------------------------------


def test_the_697_nested_core_lands_where_the_host_frame_says() -> None:
    """The reported fixture and its measurement, read correctly.

    The report quotes ``envelope y[-0.300, 0.000]`` / ``core_body y[0.000,
    0.180]`` and calls the second one MIRRORED. It is not: the envelope's
    ``facing=+1`` puts its outer face at ``y=0``, and in a frame whose ``+y``
    runs INTO the element from that face, an authored ``y ∈ [-180, 0]`` is a
    request to stand 180 mm PROUD of it. Which is where it stands.

    So this is a pin, not a falsifier — the span does not move with this
    change, and a PR that rebased an anchored container onto its own AABB min
    corner (considered, and refused) would break it. What the change adds is
    the line below it: the core's own frame now names the same face as the
    frame that placed it, so a second generation anchored INTO the core builds
    on the face the author sees. Test 1 is where that is falsifiable.
    """
    envelope = _host()
    core = Wall(name="core")
    core.add(Box(start=Point(x=0, y=-180, z=0), end=Point(x=6000, y=0, z=2700),
                 type="wall", name="core_body"))
    envelope.anchor(core, along=0, up=0)

    proj = Project(name="697-1")
    proj.add(envelope)
    assert validate_project_report(proj).errors == []
    proj = normalize_project_to_meters(proj) or proj

    from lite_step.compiler.frames import resolve_child_anchors, stamp_frames

    stamp_frames(proj)
    resolve_child_anchors(proj)
    stamp_frames(proj)

    asm = _wall_of(proj)
    core = _named(asm, "core")
    assert _span(_named(asm, "body"))["y"] == (-0.3, 0.0), "the envelope"
    assert _span(_named(core, "core_body"))["y"] == (0.0, 0.18), (
        "the core body: authored y[-180, 0] against an outer face at y=0, "
        "so it stands proud from 0.000 to 0.180 — the issue's number, and the "
        "frame's own statement rather than a mirror")
    assert core._frame.facing == asm._frame.facing == 1, (
        "the core must call the same face 'outside' as the envelope that "
        "placed it, or anything anchored into the core goes the other way")


# ---------------------------------------------------------------------------
# 3. Perpendicular: the reference says nothing, so the scope centre answers
# ---------------------------------------------------------------------------


def test_a_perpendicular_sub_element_re_derives_from_the_scope_centre() -> None:
    """A north-south return wall on an east-west host inherits NOTHING.

    ``facing_from`` returns ``None`` here, and the fall-through is what
    produces the answer — measurably, because the answer is the opposite of
    the host's. Degrade the ``None`` to the reference's raw facing (the choice
    ``opening_out_sign`` makes for its own degenerate case) and the pier
    reports ``+1``.
    """
    proj = _resolved(_return_wall(), along=2000)
    host = _wall_of(proj)
    pier = _named(host, "pier")

    assert pier._frame.across == (-1.0, 0.0, 0.0), (
        "the pier's own run is north-south, so its across is the world -X — "
        "the axes are re-derived, never inherited")
    assert pier._frame.along == (0.0, 1.0, 0.0)
    assert facing_from(host._frame.across, host._frame.facing,
                       pier._frame.across) is None, (
        "the two across axes are perpendicular; the host has no opinion")

    # The pier spans x[3.8, 4.0]; the wall's mid-point is x=3.0, so the farther
    # face is x=4.0 — the -across side, hence -1.
    assert _span(_named(pier, "pier_body"))["x"] == (3.8, 4.0), "the fixture"
    assert pier._frame.facing == -1, (
        f"the pier reports facing={pier._frame.facing:+d} — it must come from "
        f"the scope centre, not from the host")
    assert pier._frame.facing != host._frame.facing, (
        "the fall-through and the inheritance must be distinguishable here, "
        "or this test cannot fail")


# ---------------------------------------------------------------------------
# 4. The extraction changed no behaviour in the OTHER reader
# ---------------------------------------------------------------------------


class _StampedHost:
    """The one thing ``opening_out_sign`` reads off a host."""

    def __init__(self, frame):
        self._frame = frame


def _frame(across, along, facing) -> ElementFrame:
    return ElementFrame(origin=(0.0, 0.0, 0.0), along=along, across=across,
                        up=(0.0, 0.0, 1.0), facing=facing, rule="wall-box",
                        extent_along=6.0, thickness=0.3, height=2.7)


def _out_sign_before_the_extraction(host, dir_x: float, dir_y: float) -> int:
    """``opening_out_sign``'s body as it stood before ``facing_from`` was
    lifted out of it — copied verbatim, so the table below is pinned against
    that implementation rather than against a re-reading of the current one."""
    frame = getattr(host, "_frame", None)
    if frame is None:
        return 1
    ax, ay = -dir_y, dir_x
    dot = frame.across[0] * ax + frame.across[1] * ay
    if abs(dot) < 1e-9:
        return frame.facing
    return frame.facing if dot > 0 else -frame.facing


#: Both run axes a wall body can carry, as ``(across, along)``. An east-west
#: wall's across is ``+Y``; a north-south wall's is ``-X`` (``up x along``).
_HOST_AXES = [((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
              ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0))]

#: Every direction an opening's own run axis can point, including the two that
#: are PERPENDICULAR to each host — the standalone ``.opening()`` case, where
#: the host took the world-axes fallback rule and its across need not agree
#: with the axis the opening actually opens along.
_RUNS = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]


@pytest.mark.parametrize("across,along", _HOST_AXES)
@pytest.mark.parametrize("facing", [1, -1])
@pytest.mark.parametrize("run", _RUNS)
def test_opening_out_sign_is_unchanged_by_the_extraction(across, along,
                                                         facing, run) -> None:
    """16 hosts: both facings, both wall run axes, four opening directions."""
    host = _StampedHost(_frame(across, along, facing))
    assert opening_out_sign(host, *run) == _out_sign_before_the_extraction(
        host, *run), f"across={across} facing={facing:+d} run={run}"


def test_both_readers_go_through_the_one_projection() -> None:
    """A SOURCE-level assertion, and the only guard the extraction has.

    Putting the projection back inline in both readers is behaviour-preserving
    by construction — that is what makes it an extraction — so the table above
    passes with the duplicate restored, and no runtime test can fail on it.
    What comes back with the duplicate is two copies of a signed projection
    free to drift, which is the shape that produced and
    and the reason ``outer_face_axis`` is one function. Same precedent as
    ``test_voids.py::test_displacement_reads_the_predicate_instead_of_restating_it``
    and ``tests/test_dsl_cheatsheet.py``.
    """
    import inspect
    from pathlib import Path

    from lite_step.compiler import frames

    for reader in (frames.opening_out_sign, frames._facing_for):
        assert "facing_from(" in inspect.getsource(reader), (
            f"{reader.__name__} must READ the shared projection, not restate "
            f"it")

    source = (Path(__file__).resolve().parents[1]
              / "compiler" / "frames.py").read_text(encoding="utf-8")
    assert source.count("abs(dot) < 1e-9") == 1, (
        "the perpendicular test belongs to facing_from alone — a second copy "
        "is a second implementation of the same idea")


def test_opening_out_sign_still_degrades_its_own_way() -> None:
    """The two readers make DIFFERENT choices about ``facing_from``'s ``None``.

    ``opening_out_sign`` degrades to the host's raw facing; ``_facing_for``
    falls through to the scope centre (test 3). Sharing the projection must not
    quietly unify the two, and an unstamped host must still read ``+1``.
    """
    perpendicular = _StampedHost(_frame((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), -1))
    assert facing_from(perpendicular._frame.across,
                       perpendicular._frame.facing, (0.0, 1.0, 0.0)) is None
    assert opening_out_sign(perpendicular, 1.0, 0.0) == -1, (
        "a perpendicular run keeps the host's stamped facing")
    assert opening_out_sign(_StampedHost(None), 1.0, 0.0) == 1, (
        "no stamp at all is the pre-frame reading, +1")


# ---------------------------------------------------------------------------
# 5. An authored outward outranks both
# ---------------------------------------------------------------------------


def test_an_authored_outward_beats_an_inherited_facing() -> None:
    """``LayerSet.outward`` is the author's own claim about THIS element.

    Same leaf, same host, same anchor — the only difference is that the leaf
    now states which way its buildup faces, and it states the opposite of what
    it would inherit. The batten follows the statement, back to the leaf's
    ``-Y`` face. The numbers are the un-inherited ones from test 1, and that
    is the point: the same position, reached by authority rather than by
    accident.
    """
    buildup = LayerSet(name="core_type", outward=(0.0, -1.0, 0.0), layers=[
        Material(key="Concrete_C25-30", thickness_mm=_LEAF_T)])
    proj = _resolved(_inner_leaf(layers=buildup))
    host = _wall_of(proj)
    leaf = _named(host, "core")

    assert host._frame.facing == 1, "the host still says +1"
    assert leaf._frame.facing == -1, (
        "an authored outward must win outright — it is not a repetition of "
        "the host's claim, it is a claim about this leaf")
    assert _span(_named(leaf, "batten"))["y"] == (-0.3, -0.26), (
        "the batten belongs on the face the AUTHOR named, not the inherited "
        "one")


# ---------------------------------------------------------------------------
# 6. — how far down the anchored subtree the reference goes
# ---------------------------------------------------------------------------


def _void_x_span(opening, host_body) -> tuple:
    """The world x-span of ``opening``'s void, through the shared derivation.

    ``opening_host_frame(_body_of(wall))`` is the exact chain both IFC
    backends walk to place a hole, and reading the BODY there — rather than
    the container ``.anchor()`` bakes through — is the half that made
    the two answers separable in the first place.
    """
    frame = opening_host_frame(host_body)
    assert frame is not None, "the fixture's body must bear an opening frame"
    prism = opening_void_prism(opening, frame)
    assert prism is not None, "the fixture's opening must cut a prism"
    xs = [round(corner[0], 6) for corner in prism.corners]
    return (min(xs), max(xs))


def test_a_batten_and_a_window_at_the_same_along_land_together() -> None:
    """the fixture: one leaf, one ``along=1000``, one answer.

    The leaf carries BOTH consumers at the same anchor — a ``Box``, which the
    bake positions through the CONTAINER's frame, and a ``Window``, whose void
    is derived from the BODY's. Inheriting the host's ``facing`` while the
    ``.add()``ed body re-derives its own from the storey scope centre makes the
    two disagree, so one ``along`` resolves to
    opposite ends of a 6 m leaf::

        element            before                after
        leaf (container)   facing +1             facing +1
        core_body (body)   facing -1             facing +1
        batten             x[4.700, 5.000]       x[4.700, 5.000]
        window void        x[1.000, 1.600]       x[4.400, 5.000]

    3.7 m apart, on one leaf, and nothing raised. BOTH spans are asserted,
    because either alone is satisfiable by a change that moves the other
    consumer instead: the requirement is that they agree.
    """
    leaf = _inner_leaf()
    window = Window(width=600, height=600, name="w0")
    leaf.anchor(window, along=1000, up=1000)
    proj = _resolved(leaf)

    host = _wall_of(proj)
    leaf = _named(host, "core")
    body = _named(leaf, "core_body")

    # -- the split itself, which is what reported ---------------------
    assert host._frame.facing == 1, "the lone wall's exterior is +Y"
    assert leaf._frame.facing == 1, "the container inherits"
    assert body._frame.facing == 1, (
        f"the BODY reports facing={body._frame.facing:+d}. It stands inside a "
        f"subtree its host anchored, so it must name the same face as the "
        f"container — openings read the body and .anchor() reads the "
        f"container, and there is only one leaf")

    # -- and what the split was worth, measured on both consumers ----------
    batten = _span(_named(leaf, "batten"))
    void_x = _void_x_span(_named(leaf, "w0"), _body_of(leaf))

    assert batten["x"] == (4.7, 5.0), (
        f"the batten is at x{batten['x']} — anchored at along=1000 on a leaf "
        f"whose facing reverses the run, it belongs 1000 mm in from x=6000")
    assert void_x == (4.4, 5.0), (
        f"the window's void is at x{void_x}. It must start from the SAME end "
        f"as the batten (x=5.000): both were authored at along=1000 on the "
        f"same leaf. Revert the carry and the void reads the body's "
        f"re-derived facing instead — x[1.000, 1.600], 3.7 m away")
    assert max(void_x) == max(batten["x"]), (
        "the two consumers must measure `along` from ONE end — that is the "
        "whole, and it is falsifiable without either literal above")


def test_the_inherited_facing_reaches_the_body_through_the_compile() -> None:
    """The stamp is not the artefact — the file is.

    ``generate_ifc`` re-runs the frame pass on its own copy, so a fix holding
    only for a hand-driven ``stamp_frames`` sequence would still emit the void
    at the wrong end. The body-half mirror of
    ``test_the_inherited_facing_survives_the_compile``.
    """
    from lite_step.ifc.generator import generate_ifc

    leaf = _inner_leaf()
    leaf.anchor(Window(width=600, height=600, name="w0"), along=1000, up=1000)
    host = _host()
    host.anchor(leaf, along=0, up=0)
    proj = Project(name="711-compile")
    proj.add(host)
    normalized = normalize_project_to_meters(proj)
    result = generate_ifc(normalized)
    assert result.success, result.error

    leaf = _named(_wall_of(normalized), "core")
    assert _named(leaf, "core_body")._frame.facing == 1
    assert _void_x_span(_named(leaf, "w0"), _body_of(leaf)) == (4.4, 5.0)


def _proxy(name: str) -> Element:
    return Element(name=name, ifc_class="IfcBuildingElementProxy")


def test_the_carry_is_transitive_not_one_extra_hop() -> None:
    """Three generations below the anchor, not one.

    ``assembly`` is anchored into the host, ``band`` is ``.add()``ed into it
    and ``course`` into ``band`` — so ``course`` is TWO ``.add()`` hops down
    the anchored subtree, and it is ``course`` that hosts the anchor whose
    position is measured. All three derive ``-1`` from the scope centre on
    their own (they share one basis AABB, in the inner half of the host), so a
    carry reaching only the anchored node's immediate children stamps
    ``+1, +1, -1`` and puts the batten at the far end. Such a widening passes
    ``test_a_batten_and_a_window_at_the_same_along_land_together`` and fails
    here, which is why both exist.
    """
    from lite_step.compiler.frames import resolve_child_anchors, stamp_frames

    host = _host()
    assembly, band, course = _proxy("assembly"), _proxy("band"), _proxy("course")
    course.add(Box(start=Point(x=0, y=_HOST_T - _LEAF_T, z=0),
                   end=Point(x=6000, y=_HOST_T, z=2700), name="course_body"))
    course.anchor(Box(start=Point(x=0, y=0, z=0),
                      end=Point(x=300, y=_BATTEN_T, z=300), name="batten"),
                  along=1000, up=1000)
    band.add(course)
    assembly.add(band)
    host.anchor(assembly, along=0, up=0)

    proj = Project(name="711-transitive")
    proj.add(host)
    assert validate_project_report(proj).errors == []
    proj = normalize_project_to_meters(proj) or proj
    stamp_frames(proj)
    resolve_child_anchors(proj)
    stamp_frames(proj)

    host = _wall_of(proj)
    assembly = _named(host, "assembly")
    band = _named(assembly, "band")
    course = _named(band, "course")

    assert _span(_named(course, "course_body"))["y"] == (-0.3, -0.18), (
        "every generation must sit in the INNER half of the host, or none of "
        "them self-derives -1 and nothing here can fail")
    chain = [f._frame.facing for f in (host, assembly, band, course)]
    assert chain == [1, 1, 1, 1], (
        f"the chain reports {chain} — a one-hop carry gives [1, 1, 1, -1], "
        f"and the generation it drops is the one hosting the anchor")

    batten = _span(_named(course, "batten"))
    assert batten["x"] == (4.7, 5.0), (
        f"the batten is at x{batten['x']} — anchored into the THIRD "
        f"generation, it measures from the face the host named only if the "
        f"reference reached all the way down. One hop short: x[1.0, 1.3]")
    assert batten["y"] == (-0.22, -0.18), "and on the outer face of that course"


def test_a_top_level_added_element_still_derives_its_own_facing() -> None:
    """THE BOUNDARY: nothing placed a top-level ``.add()``, so nothing crosses one.

    A subtree ``.add()``ed at the top level was authored in world coordinates
    and is not standing where some frame put it, so the scope-centre rule
    remains the only statement available about it.

    The witness is ``pier``: a nested wall inside a LAYERED container, running
    PARALLEL to it so that a leaked reference would project cleanly rather than
    fall through ``facing_from``'s perpendicular ``None``, and standing where
    it self-derives ``+1`` against the container's authored ``-1``. Leak the
    carry past the top-level ``.add()`` and it reports ``-1``.

    ``far`` is scenery, and load-bearing scenery: it drags the storey scope
    centre out to y=-1.79 so the pier and the body BOTH self-derive ``+1``.
    Without it they straddle their own union's centre, one of them reads ``-1``
    for a reason unrelated to any boundary, and the test passes hollow.

    See the module docstring's section 8. The layered split is closed through
    the authored ``LayerSet.outward`` rather than through the carry, so the
    body agrees with its container and cannot witness a boundary about
    inheritance.
    """
    from lite_step.compiler.frames import resolve_child_anchors, stamp_frames

    wall = Wall(name="solo")
    wall.add(Box(start=Point(x=0, y=_HOST_T - _LEAF_T, z=0),
                 end=Point(x=6000, y=_HOST_T, z=2700), type="wall",
                 name="solo_body"))
    wall.layers = LayerSet(name="solo_type", outward=(0.0, -1.0, 0.0), layers=[
        Material(key="Concrete_C25-30", thickness_mm=_LEAF_T)])
    pier = Wall(name="pier")
    pier.add(Box(start=Point(x=0, y=_HOST_T, z=0),
                 end=Point(x=6000, y=_HOST_T + _LEAF_T, z=2700), type="wall",
                 name="pier_body"))
    wall.add(pier)

    far = Wall(name="far")
    far.add(Box(start=Point(x=0, y=-4000, z=0),
                end=Point(x=6000, y=-3880, z=2700), type="wall",
                name="far_body"))

    proj = Project(name="711-boundary")
    proj.add(wall)
    proj.add(far)
    assert validate_project_report(proj).errors == []
    proj = normalize_project_to_meters(proj) or proj
    stamp_frames(proj)
    resolve_child_anchors(proj)
    stamp_frames(proj)

    wall = _wall_of(proj)
    pier = _named(wall, "pier")

    assert wall._frame.facing == -1, (
        "the container states its own outward, and an authored outward wins")
    assert _named(wall, "solo_body")._frame.facing == -1, (
        "the BUILDUP takes that same authored statement — stated here "
        "so the claim is stated, not merely dropped")
    assert pier._frame.across == wall._frame.across, (
        "the pier must run parallel to its container, or a leaked reference "
        "would fall through facing_from's None and this could not fail")
    assert pier._frame.facing == 1, (
        f"the pier reports facing={pier._frame.facing:+d}. Nothing anchored "
        f"this wall, so a child that is not its buildup derives from the scope "
        f"centre as it always has — the carry must not follow a top-level "
        f"`.add()`")


def test_a_perpendicular_child_inside_an_anchored_subtree_still_falls_through() -> None:
    """The carry must not bypass ``facing_from``'s ``None``.

    Test 3 pins the fall-through AT the anchored node; this is the same
    question one generation on, where the reference now arrives through the
    carry rather than through ``is_anchored_child``. A north-south buttress
    ``.add()``ed into an east-west leaf takes nothing from it and derives its
    own answer — which is the OPPOSITE of the leaf's, so the test can fail.
    """
    leaf = _inner_leaf()
    pier = Wall(name="pier")
    pier.add(Box(start=Point(x=2000, y=_HOST_T - _LEAF_T, z=0),
                 end=Point(x=2200, y=_HOST_T + 700, z=2700), type="wall",
                 name="pier_body"))
    leaf.add(pier)
    proj = _resolved(leaf)

    host = _wall_of(proj)
    leaf = _named(host, "core")
    pier = _named(leaf, "pier")

    assert leaf._frame.facing == 1, "the reference is present and says +1"
    assert _named(leaf, "core_body")._frame.facing == 1, (
        "its PARALLEL sibling does take it — otherwise this test would pass "
        "with the whole carry deleted")
    assert pier._frame.across == (-1.0, 0.0, 0.0), (
        "the pier runs north-south; axes are re-derived, never inherited")
    assert facing_from(leaf._frame.across, leaf._frame.facing,
                       pier._frame.across) is None, (
        "perpendicular — the leaf has no opinion about the pier's faces")
    assert pier._frame.facing == -1, (
        f"the pier reports facing={pier._frame.facing:+d} — it must come from "
        f"the scope centre. Degrade the None to the reference's raw facing "
        f"(what opening_out_sign chooses for its own case) and this is +1")


def test_the_carry_stops_at_an_opening() -> None:
    """A ``Window``/``Door`` subtree is a different coordinate system.

    An opening bears no frame — ``frame_basis_points`` yields nothing for one
    — and ``_bake_walk`` never drags its children into world space, so they
    stand in OPENING-LOCAL coordinates that ``opening_placement`` resolves.
    Restating a world-space facing on one of their ``across`` axes answers a
    question about the wrong frame. So a frame-less link ENDS the carry rather
    than passing the reference it was handed straight through; measured, the
    difference is 110 further corpus frames, every one inside a ``Window`` or
    ``Door``.

    The leaf states an outward of its own so that it reads ``-1`` while the
    karm inside its window still derives ``+1``. Pass the reference through
    the opening and the karm reports ``-1``.
    """
    leaf = _inner_leaf(layers=LayerSet(
        name="core_type", outward=(0.0, -1.0, 0.0),
        layers=[Material(key="Concrete_C25-30", thickness_mm=_LEAF_T)]))
    window = Window(width=600, height=600, name="w0")
    window.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=600, y=60, z=80),
                   name="karm"))
    leaf.anchor(window, along=1000, up=1000)
    proj = _resolved(leaf)

    leaf = _named(_wall_of(proj), "core")
    window = _named(leaf, "w0")
    karm = _named(window, "karm")

    assert leaf._frame.facing == -1, (
        "the leaf states its own outward, so it disagrees with everything "
        "around it — which is what makes the karm's answer readable")
    assert window._frame is None, (
        "an opening bears no frame of its own; that is the link the carry "
        "ends at")
    assert karm._frame.facing == 1, (
        f"the karm reports facing={karm._frame.facing:+d}. It is authored in "
        f"opening-local coordinates and must not restate the leaf's "
        f"world-space facing — passing the reference through the frame-less "
        f"opening gives -1")
