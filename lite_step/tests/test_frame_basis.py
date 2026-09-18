"""An anchored child cannot help define the frame that placed it.

``.add()`` and ``.anchor()`` make different claims about coordinates::

    .add(child)     -> "already world coordinates"  -> defines the container's frame
    .anchor(child)  -> "relative to YOUR frame"     -> placed BY that frame

Deriving every frame from ``element_aabb`` — *everything the element
occupies* — and carrying the anchored/not distinction in a single attribute,
``_anchor_spec``, that the bake CLEARS, does not hold. So the exclusion held for exactly as long as the anchor was pending:
``resolve_child_anchors`` set ``child._anchor_spec = None``, the caller
re-stamped, and every anchored child walked straight back into the AABB its own
placement had been derived from. Two measurements on ``main``:

1.  A 180 mm core leaf ``.anchor()``ed into a 180 mm wall took the HOST's
    ``_frame.thickness`` from 180 to **360** across the bake. The host got
    thicker because something was hung on it.
2.  Adding a nested anchored ``Wall`` to the ordinary window fixture flipped the
    host's own ``facing`` and moved that wall's EXISTING window from world
    ``x[3.3, 4.5]`` to ``x[1.5, 2.7]``. Adding a leaf moved a window by 1.8 m,
    with no error and no warning.

Both are the same loop, and the fix is to break it in one place: the frame is
now derived from :func:`~lite_step.compiler.frames.frame_basis_aabb` — the
element plus every descendant that is NOT anchored into it — and
"is this child anchored" is answered by
:func:`~lite_step.compiler.frames.is_anchored_child`, which reads
``_anchor_resolved`` as well as ``_anchor_spec`` and therefore does not flip at
the bake.

What each test is here to catch:

1.  ``test_the_host_keeps_its_own_thickness_across_the_bake`` and
    ``test_a_nested_anchored_wall_does_not_move_the_hosts_window`` are the two
    measurements above, as numbers. Revert ``frame_for`` to ``element_aabb``
    and they report 360 and ``x[1.5, 2.7]`` again.
2.  ``test_frames_are_invariant_across_the_bake`` is the general statement the
    two numbers are instances of: for every element that is not itself
    anchored, the stamped frame is the SAME object-for-object before and after
    ``resolve_child_anchors``. A fix that only handled ``thickness``, or only
    the direct-child case, passes both numeric tests and fails this one.
3.  ``test_an_anchored_container_still_gets_its_own_frame_in_generation_2``
    guards the asymmetry that makes the whole pass converge. The exclusion is
    on CHILDREN; the element being asked about is exempt. Apply
    ``is_anchored_child`` to the root as well and a resolved container — which
    holds true world coordinates and whose frame is what its own inner anchors
    resolve against — never gets a frame again, and
    ``resolve_child_anchors`` runs out of generations instead.
4.  ``test_the_full_extent_still_sees_the_anchored_child`` pins the other half
    of the split. ``element_aabb`` / ``world_aabb()`` / ``project_view_extents``
    still mean "everything I occupy". Collapsing the two meanings back into one
    function would clip an anchored cornice out of an orthographic view — the
    same class of silent wrongness, pointing the other way.

Units: tests 1a and 3 run on the RAW mm project, so the numbers read exactly as
issue quotes them (180 / 360); the passes are unit-agnostic. Tests 1b, 2
and 4 run the normalized metre project through ``generate_ifc``, which is the
production order (``stamp_frames`` -> ``resolve_child_anchors`` ->
``stamp_frames``).
"""

from __future__ import annotations

import pytest

from lite_step.compiler import frames
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.compiler.frames import (
    element_aabb,
    frame_basis_aabb,
    is_anchored_child,
    resolve_child_anchors,
    stamp_frames,
)
from lite_step.models import Box, Point, Project, Wall, Window

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Issue the assembly: a 180 mm wall with a 180 mm core leaf anchored flush
#: into it. Both leaves are the same thickness so the pollution is unambiguous
#: — a host that reports anything other than 180 is reporting its child.
_T = 180


def _leaf(name: str, thickness: int = _T) -> Box:
    return Box(start=Point(x=0, y=-thickness, z=0),
               end=Point(x=6000, y=0, z=2700), type="wall", name=name)


def _assembly() -> Wall:
    """A wall whose only ``.add()``ed geometry is its own 180 mm body."""
    asm = Wall(name="south")
    asm.add(_leaf("body"))
    core = Wall(name="core")
    core.add(_leaf("core_body"))
    asm.anchor(core, along=0, up=0)
    return asm


def _window_wall(*, nested: bool) -> Wall:
    """``test_void_carves_details``' wall, optionally with a nested leaf.

    The ONLY difference between the two spellings is the anchored ``Wall``.
    Everything the window's position is derived from is identical.
    """
    wall = Wall(name="south")
    wall.add(Box(start=Point(x=0, y=-300, z=0), end=Point(x=6000, y=0, z=2700),
                 type="wall", name="body"))
    if nested:
        core = Wall(name="core")
        core.add(_leaf("core_body"))
        wall.anchor(core, along=0, up=0)
    wall.anchor(Window(width=1200, height=1400, name="w0"), along=1500, up=900)
    return wall


def _three_details(*, cornice: bool = True) -> Wall:
    """One wall, three anchored details, each a different shape of nuisance.

    * ``post`` — an ordinary leaf, inside the host body.
    * ``cornice`` — 400 mm LONGER than the wall it hangs on, so it extends the
      full-extent AABB along the run axis. Overhang in ``along`` rather than
      ``across`` because the SPAN grows either way: which end it overhangs
      still depends on the host's ``facing``, and this suite must not depend
      on that (inheriting a host's facing is a separate change).
    * ``leaf`` — a nested CONTAINER, which is the case that also carries a
      second generation.
    """
    wall = Wall(name="south")
    wall.add(Box(start=Point(x=0, y=-300, z=0), end=Point(x=6000, y=0, z=2700),
                 type="wall", name="body"))
    wall.anchor(Box(start=Point(x=0, y=0, z=0), end=Point(x=50, y=50, z=2700),
                    name="post"), along=2100, up=0)
    if cornice:
        wall.anchor(Box(start=Point(x=0, y=0, z=0), end=Point(x=6400, y=50, z=120),
                        name="cornice"), along=0, up=2580)
    leaf = Wall(name="leaf")
    leaf.add(_leaf("leaf_body"))
    wall.anchor(leaf, along=0, up=0)
    return wall


def _raw(element) -> Project:
    """A validated project in AUTHORED millimetres — no unit normalization."""
    proj = Project(name="frame-basis")
    proj.add(element)
    assert validate_project_report(proj).errors == []
    return proj


def _metres(element) -> Project:
    proj = _raw(element)
    return normalize_project_to_meters(proj) or proj


def _wall_of(proj: Project) -> Wall:
    return proj.storeys[0].elements[0]


def _named(elem, name: str):
    for child in (getattr(elem, "_elements", None) or []):
        if getattr(child, "name", None) == name:
            return child
    raise AssertionError(f"no child named {name!r} on {elem.name!r}")


def _hole_x(proj: Project) -> tuple[float, float]:
    """World x-span of the window's void, read the way the emitter reads it."""
    from lite_step.ifc.entity_cache import snap_to_precision

    wall = _wall_of(proj)
    frame = frames.opening_host_frame(frames._body_of(wall),
                                      snap=snap_to_precision, divisor=1000.0)
    prism = frames.opening_void_prism(_named(wall, "w0"), frame)
    (x0, _y0, _z0), (x1, _y1, _z1) = prism.aabb()
    return round(x0, 6), round(x1, 6)


def _compile(proj: Project) -> None:
    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(proj)
    assert result.success, result.error


# ---------------------------------------------------------------------------
# 1. The two measurements from issue, as numbers
# ---------------------------------------------------------------------------


def test_the_host_keeps_its_own_thickness_across_the_bake() -> None:
    """Measurement 1: the host got THICKER because of what was hung on it.

    ``thickness`` is the frame's span along ``across``, and ``.anchor(inset=)``
    is measured in it — so a host that reports 360 places the next child at
    twice the depth the author sees in the model.
    """
    proj = _raw(_assembly())
    stamp_frames(proj)
    before = _wall_of(proj)._frame.thickness

    resolve_child_anchors(proj)
    stamp_frames(proj)
    after = _wall_of(proj)._frame.thickness

    assert before == pytest.approx(_T), (
        "the host's own 180 mm body is the whole basis before the bake")
    assert after == pytest.approx(_T), (
        f"the host reports {after} mm thick after baking a 180 mm core into "
        f"it — on main this was 360, the host's body plus the child the "
        f"host's own frame had just placed")


def test_a_nested_anchored_wall_does_not_move_the_hosts_window() -> None:
    """Measurement 2: adding a leaf moved an EXISTING window by 1.8 m.

    The hole is the same authored ``.opening(along=1500)`` in both models. If
    the nested wall reaches the host's frame at all it reaches ``facing``
    through it, and ``opening_out_sign`` reverses the run direction — which
    reads as a window that silently jumped to the other end of the wall.
    """
    plain = _metres(_window_wall(nested=False))
    _compile(plain)
    nested = _metres(_window_wall(nested=True))
    _compile(nested)

    assert _hole_x(plain) == (3.3, 4.5), "the fixture this suite quotes"
    assert _hole_x(nested) == _hole_x(plain), (
        f"the nested wall moved the HOST's window to {_hole_x(nested)} — on "
        f"main this was x[1.5, 2.7], an 1.8 m jump caused by adding a child "
        f"that says nothing about where the window goes")


# ---------------------------------------------------------------------------
# 2. The general statement the two numbers are instances of
# ---------------------------------------------------------------------------


def _frames_by_id(proj):
    """``{id(elem): (elem, frame)}`` for every element the bake does not MOVE.

    An anchored element's whole SUBTREE is excluded, not just the anchored
    element itself: ``_bake_subtree`` rewrites the descendants' coordinates
    too, so their frames are supposed to change. Excluding only the anchored
    node would make this test assert that the bake does nothing, which is the
    opposite of the claim.
    """
    out = {}

    def walk(elem):
        if is_anchored_child(elem):
            return
        out[id(elem)] = (elem, getattr(elem, "_frame", None))
        for child in (getattr(elem, "_elements", None) or []):
            walk(child)

    for storey in proj.storeys:
        for elem in storey.elements:
            walk(elem)
    return out


def test_frames_are_invariant_across_the_bake() -> None:
    """The bake moves anchored children and NOTHING a frame is derived from.

    Asserted element-by-element over a wall carrying three anchored details —
    a leaf inside the body, a cornice projecting past its outer face, and a
    nested container. A fix that only repaired the direct-child case, or only
    the ``across`` axis, passes the two numeric tests above and fails here.
    """
    proj = _metres(_three_details())
    stamp_frames(proj)
    before = {k: v[1] for k, v in _frames_by_id(proj).items()}
    assert before, "fixture built no non-anchored elements"

    resolve_child_anchors(proj)
    stamp_frames(proj)
    after = _frames_by_id(proj)

    assert set(after) == set(before), (
        "the bake changed WHICH elements count as anchored")
    for key, (elem, now) in after.items():
        was = before[key]
        assert (was is None) == (now is None), f"{elem.name}: frame appeared/vanished"
        if was is None:
            continue
        assert now.facing == was.facing, (
            f"{elem.name}: facing flipped {was.facing:+d} -> {now.facing:+d} "
            f"across the bake — its exterior side now depends on what was "
            f"anchored to it")
        assert now.origin == pytest.approx(was.origin, abs=1e-9), elem.name
        assert now.along == was.along and now.across == was.across, elem.name
        assert now.thickness == pytest.approx(was.thickness, abs=1e-9), elem.name
        assert now.extent_along == pytest.approx(was.extent_along, abs=1e-9), elem.name
        assert now.height == pytest.approx(was.height, abs=1e-9), elem.name


# ---------------------------------------------------------------------------
# 3. The asymmetry: children are excluded, the element asked about is not
# ---------------------------------------------------------------------------


def test_an_anchored_container_still_gets_its_own_frame_in_generation_2() -> None:
    """A resolved container holds WORLD coordinates, so it derives a frame.

    This is what breaks if the exclusion is applied to the element being asked
    about rather than to its children: the container would have no frame, the
    anchor authored inside it would have nothing to resolve against, and
    ``resolve_child_anchors`` would burn all 32 generations and raise. The
    frame it gets must also be its OWN — its body only, not its body plus the
    shelf hanging off it.
    """
    inner = Wall(name="leaf")
    inner.add(_leaf("leaf_body"))
    inner.anchor(Box(start=Point(x=0, y=0, z=0), end=Point(x=100, y=100, z=100),
                     name="shelf"), along=500, up=500)
    host = Wall(name="south")
    host.add(Box(start=Point(x=0, y=-300, z=0), end=Point(x=6000, y=0, z=2700),
                 type="wall", name="body"))
    host.anchor(inner, along=0, up=0)

    proj = _metres(host)
    stamp_frames(proj)
    leaf = _named(_wall_of(proj), "leaf")
    assert leaf._frame is None, (
        "before the bake the container's points are host-LOCAL; deriving a "
        "world frame from them is what iter_geometry_points already refuses")

    resolve_child_anchors(proj)
    stamp_frames(proj)

    assert leaf._frame is not None, (
        "generation 2 must give the placed container a frame, or the anchor "
        "authored inside it can never resolve")
    assert is_anchored_child(leaf), "the fixture stopped testing what it says"
    assert getattr(_named(leaf, "shelf"), "_anchor_spec", None) is None, (
        "the inner anchor never resolved")
    assert leaf._frame.thickness == pytest.approx(_T / 1000.0), (
        f"the container's frame is {leaf._frame.thickness} thick — its own "
        f"180 mm body plus the 100 mm shelf it placed, which is the same "
        f"loop one level down")


# ---------------------------------------------------------------------------
# 4. The other half of the split — nothing SHRANK
# ---------------------------------------------------------------------------


def test_the_full_extent_still_sees_the_anchored_child() -> None:
    """``element_aabb`` still means "everything I occupy", and must.

    The cornice is a 6.4 m board hung on a 6.0 m wall, so it overhangs by
    400 mm. The frame basis stops at 6.0 — that is this PR, and it is why the
    wall's ``extent_along`` does not grow when you hang trim on it. But an
    orthographic view that stopped at 6.0 would CLIP the overhang off the
    drawing, and a ``world_aabb()`` that stopped there would report a wall
    smaller than the solids it emits. Asserted, because the two halves of a
    deliberate split are exactly the pair that drifts.
    """
    from lite_step.ifc.drawings import project_view_extents

    proj = _metres(_three_details())
    _compile(proj)
    wall = _wall_of(proj)

    def _x_span(box):
        (lo, hi) = box
        return round(hi[0] - lo[0], 6)

    assert _x_span(element_aabb(wall)) == pytest.approx(6.4), (
        "the full extent must still reach the whole cornice")
    assert _x_span(frame_basis_aabb(wall)) == pytest.approx(6.0), (
        "the frame basis must be the wall's own 6 m body")
    assert wall._frame.extent_along == pytest.approx(6.0), (
        f"the wall's run is {wall._frame.extent_along} m — hanging a longer "
        f"board on a wall must not lengthen the wall")

    # ``world_aabb()`` is the AUTHORING query and reports int mm, so it is
    # asked on the un-normalized project — the same tree, unconverted. It
    # triggers the bake itself.
    mm = _raw(_three_details())
    bounds = _wall_of(mm).world_aabb()
    assert bounds.max.x - bounds.min.x == 6400, (
        f"world_aabb() spans {bounds.max.x - bounds.min.x} mm — it must "
        f"report the cornice, not the frame basis")

    bare = _metres(_three_details(cornice=False))
    _compile(bare)
    assert project_view_extents(proj)[0] > project_view_extents(bare)[0], (
        "the ortho view must grow to cover the anchored cornice; reading the "
        "frame basis here would clip it off the sheet")
