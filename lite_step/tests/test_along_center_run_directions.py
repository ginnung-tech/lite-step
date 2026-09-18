"""``along_center=`` centres the child on EVERY run direction, not just ``+x``.

It did not. ``resolve_along`` measured the child's span by projecting the
child's own parent-local points onto ``frame.along`` — the HOST's run axis, in
WORLD coordinates. Those are two different frames. On an x-run host the two
happen to agree, so the projection read the child's local x-extent and the
answer was right by coincidence; on a y-run host it read the child's local
*y* extent instead, the half-span correction collapsed toward zero, and the
child LEFT-ALIGNED on the host's midpoint instead of centring on it
. Silent — a sill half its own length off, no warning.

The corpus is x-run dominated, and the suite pinned two x-run rows, so nothing
could falsify it. Hence the table below is the whole point: four run
directions, and a child asymmetric in x and y so a wrong axis cannot land on
the right number by accident.

Pinned here:

1.  All four ring walls centre the same child on their own midpoint. Two of
    these four pass even with the wrong axis; a fixture that cannot tell them
    apart is what lets that ship.
2.  A child rotated 90 degrees about z centres too. This is the row that says
    WHICH axis is correct: not a constant local ``+x``, but local ``+x`` turned
    by the anchor's own ``rotations=``. A 90-degree turn presents the child's
    100 mm local *y* to the run, not its 300 mm local *x*, and the world
    measurement below agrees with that and not with the other candidate.
3.  A child authored ACROSS its own origin centres too — the correction is the
    MIDPOINT of the projected span, not half its length. Those two agree only
    while the near edge sits on the local origin, which is exactly the case
    a naive implementation silently assumes and a rotated child never satisfies.
4.  A ``Window`` centres on all four directions as well. It takes the OTHER
    branch of the same function (``infer_opening_size``, already in the child's
    own frame) and was correct throughout — pinned so the two conversions
    cannot drift apart, since a future edit to one is exactly how they would.
5.  The resolver's projection axis, asserted directly, so the claim in 2 is
    stated and not merely implied by a coordinate.

Every geometric assertion reads WORLD coordinates back out of the emitted
file through ``ifcopenshell.geom.iterator`` — not internal state, and not
``create_shape``, which has been observed returning zero verts
non-deterministically in this repo.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile

import pytest

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.compiler.frames import resolve_along
from lite_step.models import Box, Point, Project, Storey, Wall, Window

try:
    import ifcopenshell
    import ifcopenshell.geom
    HAVE_IFC = True
except ImportError:                                   # pragma: no cover
    HAVE_IFC = False

pytestmark = pytest.mark.skipif(not HAVE_IFC, reason="ifcopenshell not installed")

# ---------------------------------------------------------------------------
# A 10 m x 10 m ring of 300 mm walls, centred on the origin. The ring is what
# makes the four rows genuinely different: a wall's frame direction is
# containment-derived, so the north wall's ``along`` runs the opposite world
# way from the south one's, and likewise east against west. Four walls is the
# smallest fixture that reaches all four.
#
# Every wall is 10 m long, so ``along_center=5000`` is its midpoint on all
# four — and its midpoint is the world origin on the run axis. One expected
# number, 0.0, for every row.
# ---------------------------------------------------------------------------

_RING = {
    #        start                              end                            run
    "south": (Point(x=-5000, y=-5000, z=0), Point(x=5000, y=-4700, z=3000), "x"),
    "north": (Point(x=-5000, y=4700, z=0), Point(x=5000, y=5000, z=3000), "x"),
    "west":  (Point(x=-5000, y=-5000, z=0), Point(x=-4700, y=5000, z=3000), "y"),
    "east":  (Point(x=4700, y=-5000, z=0), Point(x=5000, y=5000, z=3000), "y"),
}

_MIDPOINT = 5000        # mm along the run — half of every wall's 10 m length

#: Asymmetric in x and y: 300 long, 100 deep. Projecting it onto the wrong
#: in-plane axis is off by 100 mm, which no rounding can hide.
_KID_END = Point(x=300, y=100, z=200)


def _ring_project(child_for, **anchor):
    """The ring, with ``child_for(tag)`` anchored on each wall by ``anchor``."""
    storey = Storey(name="g", elevation=0)
    project = Project(name="run_directions")
    project.add_storey(storey)
    for tag, (start, end, _run) in _RING.items():
        wall = Wall(name=tag)
        wall.add(Box(name="body", start=start, end=end, material="Brick_Red_DK"))
        wall.anchor(child_for(tag), **anchor)
        storey.add(wall)
    return project


def _world_run_spans(project, needle):
    """``{wall_tag: (lo_mm, hi_mm)}`` — the world span of every shape whose
    name contains ``needle``, measured on ITS OWN wall's run axis."""
    from lite_step.ifc.generator import generate_ifc

    report = validate_project_report(project)
    assert not report.errors, report.errors

    with contextlib.redirect_stdout(io.StringIO()):
        content = generate_ifc(normalize_project_to_meters(project)).ifc_content
    handle = tempfile.NamedTemporaryFile("w", suffix=".ifc", delete=False)
    handle.write(content)
    handle.close()
    try:
        model = ifcopenshell.open(handle.name)
        settings = ifcopenshell.geom.settings()
        settings.set("use-world-coords", True)
        walker = ifcopenshell.geom.iterator(settings, model, 1)
        spans: dict = {}
        assert walker.initialize(), "no geometry in the emitted file"
        while True:
            shape = walker.get()
            name = shape.name or ""
            tag = next((t for t in _RING if f":wall:{t}:" in name), None)
            if tag is not None and needle in name:
                verts = list(shape.geometry.verts)
                seq = verts[0::3] if _RING[tag][2] == "x" else verts[1::3]
                lo, hi = min(seq) * 1000.0, max(seq) * 1000.0
                prev = spans.get(tag)
                spans[tag] = (min(lo, prev[0]), max(hi, prev[1])) if prev else (lo, hi)
            if not walker.next():
                break
    finally:
        os.unlink(handle.name)
    assert set(spans) == set(_RING), f"missing walls for {needle!r}: {sorted(spans)}"
    return spans


def _centres(spans):
    return {tag: (lo + hi) / 2.0 for tag, (lo, hi) in spans.items()}


# ---------------------------------------------------------------------------
# 1-3. The generic-solid branch, across all four run directions
# ---------------------------------------------------------------------------

#: ``(id, child factory, anchor kwargs, expected run-axis extent in mm)``.
#: The extent is asserted alongside the centre because a centre alone cannot
#: tell "correctly centred" from "correctly centred on the wrong dimension" —
#: the rotated row is exactly that hazard, since its 100 mm span and the
#: unrotated 300 mm one share a midpoint.
_ROWS = [
    (
        "an asymmetric solid",
        lambda tag: Box(name=f"kid_{tag}", start=Point(x=0, y=0, z=0),
                        end=_KID_END, material="Zinc_Titanium_EN988"),
        dict(along_center=_MIDPOINT, up=500, inset=0),
        300.0,
    ),
    (
        "a solid rotated a quarter turn about z",
        lambda tag: Box(name=f"kid_{tag}", start=Point(x=0, y=0, z=0),
                        end=_KID_END, material="Zinc_Titanium_EN988"),
        dict(along_center=_MIDPOINT, up=500, inset=0, rotations=[("z", 9000)]),
        100.0,          # the local Y extent, because the turn presents it
    ),
    (
        "a solid authored across its own origin",
        lambda tag: Box(name=f"kid_{tag}", start=Point(x=-400, y=0, z=0),
                        end=Point(x=100, y=100, z=200),
                        material="Zinc_Titanium_EN988"),
        dict(along_center=_MIDPOINT, up=500, inset=0),
        500.0,
    ),
]


@pytest.mark.parametrize("label,make_child,anchor,extent",
                         _ROWS, ids=[r[0] for r in _ROWS])
def test_along_center_puts_the_child_on_the_host_midpoint_on_every_run_direction(
        label, make_child, anchor, extent):
    """The host midpoint is the world origin on all four walls, so every row's
    expected centre is the same 0.0 — north against south and east against
    west, which is where a frame mismatch shows."""
    spans = _world_run_spans(_ring_project(make_child, **anchor), "kid_")
    for tag, centre in sorted(_centres(spans).items()):
        assert centre == pytest.approx(0.0, abs=1e-9), (
            f"{label} on the {tag} wall: centred at {centre:+.3f} mm, "
            f"expected the host midpoint (0.0); span {spans[tag]}")
    for tag, (lo, hi) in sorted(spans.items()):
        assert hi - lo == pytest.approx(extent, abs=1e-9), (
            f"{label} on the {tag} wall: spans {hi - lo:.3f} mm along the run, "
            f"expected {extent} — the centring used the wrong dimension")


# ---------------------------------------------------------------------------
# 4. The Window/Door branch of the same function
# ---------------------------------------------------------------------------


def test_a_centred_window_lands_on_the_host_midpoint_on_every_run_direction():
    """The other branch of ``resolve_along``. It reads ``infer_opening_size``,
    which is already in the child's own frame, so it never had the bug — and
    an opening is not baked at all, which is why ``rotations=`` does not reach
    it and this branch takes no angle. Pinned so the two conversions cannot
    drift apart.

    Measured on the VOID, because ``along_center`` on an opening centres the
    HOLE — that is what the resolver's own docstring promises."""
    project = _ring_project(
        lambda tag: Window(name=f"win_{tag}", width=1200, height=1400),
        along_center=_MIDPOINT, up=900)
    spans = _world_run_spans(project, "_void")
    for tag, centre in sorted(_centres(spans).items()):
        assert centre == pytest.approx(0.0, abs=1e-9), (
            f"window void on the {tag} wall: centred at {centre:+.3f} mm")
    for tag, (lo, hi) in sorted(spans.items()):
        assert hi - lo == pytest.approx(1200.0, abs=1e-9), (
            f"window void on the {tag} wall: {hi - lo:.3f} mm wide")


# ---------------------------------------------------------------------------
# 5. The axis itself, stated
# ---------------------------------------------------------------------------


def test_the_resolver_projects_onto_the_rotated_local_axis_not_a_fixed_one():
    """Which axis, said out loud rather than inferred from a coordinate.

    The child spans 300 mm in local x and 100 mm in local y. Unrotated, the
    run sees 300 and the centring steps back 150. Turned a quarter turn, the
    run sees 100 — and the turned span is ``[-100, 0]``, so the correction is
    its midpoint ``-50`` and the offset moves the other way. A constant
    ``(1, 0, 0)`` would answer ``-150`` for both; the host's world ``along``
    would answer ``-50`` for both on a y-run host. Neither matches.
    """
    def anchored(**kw):
        wall = Wall(name="s")
        wall.add(Box(name="body", start=Point(x=-5000, y=-5000, z=0),
                     end=Point(x=5000, y=-4700, z=3000)))
        kid = Box(name="kid", start=Point(x=0, y=0, z=0), end=_KID_END)
        wall.anchor(kid, **kw)
        return kid._anchor_spec, kid

    spec, kid = anchored(along_center=_MIDPOINT)
    assert resolve_along(spec, kid) == pytest.approx(_MIDPOINT - 150.0)

    spec, kid = anchored(along_center=_MIDPOINT, rotations=[("z", 9000)])
    # cos 90 = 0, sin 90 = 1: projections are ``-y``, spanning [-100, 0].
    assert resolve_along(spec, kid, cos_a=0.0, sin_a=1.0) == pytest.approx(
        _MIDPOINT + 50.0)


def test_the_resolver_refuses_a_frame_where_an_angle_belongs():
    """``resolve_along(spec, child, host._frame)`` passes a world frame where
    the child's own angle belongs. The angle arguments are keyword-only, so
    that call is a ``TypeError`` rather than a frame silently read as a
    cosine."""
    wall = Wall(name="s")
    wall.add(Box(name="body", start=Point(x=-5000, y=-5000, z=0),
                 end=Point(x=5000, y=-4700, z=3000)))
    kid = Box(name="kid", start=Point(x=0, y=0, z=0), end=_KID_END)
    wall.anchor(kid, along_center=_MIDPOINT)
    with pytest.raises(TypeError):
        resolve_along(kid._anchor_spec, kid, object())      # type: ignore[misc]
