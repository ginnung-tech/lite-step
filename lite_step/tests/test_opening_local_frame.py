"""DSL v21 — the opening-local frame IS the wall's own frame, out-positive.

Every wall is authored as if it were the south wall. Stand where the camera
stands (``y-``, a little up), look at the south facade, and that view is the
local frame of every wall in the model:

===========  ==========================================================
origin       the opening's lower-left corner on the wall's **OUTER** face
**+x**       right, as seen from outside
**+y**       **into** the wall — depth from the outer face
**+z**       up
===========  ==========================================================

Measured against ``20.2.0``, the frame it replaces had local ``+y`` at the
wall MIDLINE pointing along a run axis canonicalized toward ``+X`` — so the
identical local coordinate ran inward on a south wall and OUTWARD on a north
one, and every window/door skill carried a per-facade ``local_y_sign`` to
undo it.

What this suite pins, and why each one is load-bearing:

1.  **Pure rotations only.** ``det(placement[:3, :3]) > 0`` on every opening,
    fill and child product. A frame with ``+y`` outward and ``+x`` still on
    the viewer's right is LEFT-handed on every wall, and
    ``IfcAxis2Placement3D`` derives ``y = z x x`` — it cannot express that,
    so it would silently emit MIRRORED geometry instead of a rotation. A
    negative determinant here means the frame is wrong, not the geometry.
2.  **One authoring, four walls.** The same window, authored once with no
    facing sign anywhere, must land at the same depth relative to ITS OWN
    outer face on all four walls of a ring — and the four must be exact
    quarter-turn images of each other.
3.  **``inset=n`` and a child at ``y=n`` are the same plane.** One axis, one
    direction, one zero. They were two different data before: ``inset``
    measured from the midline outward, a child's ``y`` from the midline along
    whichever way the canonicalized run happened to point.
"""

from __future__ import annotations

import math
import os
import tempfile

import pytest

from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.models import (
    Project, Point, Box, Extrude, Sweep, Wall, Window, Material,
)

# A 4-fold symmetric PINWHEEL ring: each leaf is the previous one turned 90°
# about Z, so "the four walls are quarter-turn images" is a statement about
# the FRAME and not about an accidentally-symmetric footprint. Outer faces
# sit at ±4.5 m, leaves are 300 mm thick, 2.7 m tall.
_H = 4500
_IN = _H - 300

_LEAVES = {
    #  name:  (start, end, outward normal, outer-face plane)
    "south": ((-_H, -_H), (_IN, -_IN), (0, -1), ("y", -4.5)),
    "east":  ((_IN, -_H), (_H, _IN), (1, 0), ("x", 4.5)),
    "north": ((-_IN, _IN), (_H, _H), (0, 1), ("y", 4.5)),
    "west":  ((-_H, -_IN), (-_IN, _H), (-1, 0), ("x", -4.5)),
}


def _rot_z(pt, quarter_turns: int):
    """``pt`` rotated ``quarter_turns`` x 90° about Z: ``(x, y) -> (-y, x)``."""
    x, y = pt[0], pt[1]
    for _ in range(quarter_turns % 4):
        x, y = -y, x
    return (x, y) + tuple(pt[2:])


def _window(name, children=(), width=1200, height=1400):
    """THE window — authored ONCE, no facing sign, no per-wall anything."""
    win = Window(width=width, height=height, name=name)
    for child in children:
        win.add(child)
    return win


def _ring(child_factory=None, inset=0, along=1000, up=900) -> Project:
    proj = Project(name="opening-frame-ring")
    for name, (start, end, _out, _face) in _LEAVES.items():
        wall = Wall(name=name)
        wall.add(Box(name="body", type="wall",
                     start=Point(x=start[0], y=start[1], z=0),
                     end=Point(x=end[0], y=end[1], z=2700)))
        kids = child_factory(name) if child_factory is not None else ()
        wall.anchor(_window(f"w_{name}", kids), along=along, up=up, inset=inset)
        proj.add(wall)
    return proj


def _compile(proj: Project, backend: str = "ifcopenshell") -> str:
    pytest.importorskip("ifcopenshell")
    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    return result.ifc_content


def _open(content: str):
    import ifcopenshell

    with tempfile.NamedTemporaryFile("w", suffix=".ifc", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(content)
        path = fh.name
    try:
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _matrix(product):
    import ifcopenshell.util.placement as plc

    return plc.get_local_placement(product.ObjectPlacement)


def _origin(product) -> tuple:
    m = _matrix(product)
    return (float(m[0][3]), float(m[1][3]), float(m[2][3]))


def _det(product) -> float:
    import numpy as np

    return float(np.linalg.det(_matrix(product)[:3, :3]))


def _by_wall(model, entity: str) -> dict:
    """``{wall leaf name: product}`` keyed off the canonical name tail."""
    out = {}
    for p in model.by_type(entity):
        if not p.Name:
            continue
        leaf = p.Name.rsplit(":", 1)[-1]
        out[leaf] = p
    return out


def _depth_from_outer_face(leaf: str, point) -> float:
    """Distance from the leaf's OUTER face to ``point``, positive INWARD."""
    _s, _e, outward, (axis, plane) = _LEAVES[leaf]
    coord = point[0] if axis == "x" else point[1]
    sign = outward[0] if axis == "x" else outward[1]
    return (coord - plane) * -sign


# ---------------------------------------------------------------------------
# 1. Pure rotations — no mirrored geometry, ever
# ---------------------------------------------------------------------------


def _joinery(_leaf):
    """Frame + pane + leaf-panel, authored once for every wall."""
    return (
        Sweep(name="karm",
              material=Material(key="Timber_C24", profile_mm=(120, 45)),
              path=[Point(x=60, y=100, z=60), Point(x=1140, y=100, z=60),
                    Point(x=1140, y=100, z=1340), Point(x=60, y=100, z=1340),
                    Point(x=60, y=100, z=60)]),
        Box(name="panel", start=Point(x=100, y=120, z=100),
            end=Point(x=1100, y=160, z=600)),
    )


def _joinery_with_glass(leaf):
    return _joinery(leaf) + (
        Extrude(name="glass", color="glass", thickness=6,
                contour=[Point(x=120, y=140, z=700),
                         Point(x=1080, y=140, z=700),
                         Point(x=1080, y=140, z=1300),
                         Point(x=120, y=140, z=1300)]),
    )


def test_every_opening_placement_is_a_pure_rotation():
    """THE guard against a mirror sneaking back in.

    Checked on the void, the fill and every aggregated child product, on all
    four walls. ifcopenshell backend only — see
    ``test_the_ring_never_reaches_the_streaming_writer`` for why there is no
    second backend to check here.
    """
    model = _open(_compile(_ring(child_factory=_joinery)))
    products = (model.by_type("IfcOpeningElement")
                + model.by_type("IfcWindow")
                + model.by_type("IfcMember")
                + model.by_type("IfcBuildingElementProxy"))
    assert len(products) >= 4 * 4, "expected a void + fill + 2 children per wall"
    for p in products:
        assert _det(p) > 0, (
            f"{p.is_a()} {p.Name!r} placement has det={_det(p):+.6f} — a "
            f"negative determinant is a MIRROR, which means the opening "
            f"frame is left-handed and IfcAxis2Placement3D cannot express it")


def test_contour_children_are_pure_rotations_too():
    """A glass pane's placement is built from a Newell normal + a contour
    edge, not from the frame matrix — a different code path, same rule.
    (ifcopenshell backend only: ``can_stream`` routes contour children here.)
    """
    model = _open(_compile(_ring(child_factory=_joinery_with_glass)))
    panes = [p for p in model.by_type("IfcBuildingElementProxy")
             if p.Name and p.Name.startswith("extrude:glass")]
    assert len(panes) == 4
    for p in panes:
        assert _det(p) > 0, f"glass pane {p.Name!r} det={_det(p):+.6f}"


# ---------------------------------------------------------------------------
# 2. One authoring, four walls
# ---------------------------------------------------------------------------


def _probe(_leaf):
    """One box, authored 50..80 mm in from the outer face on every wall."""
    return (Box(name="probe", start=Point(x=100, y=50, z=100),
                end=Point(x=200, y=80, z=200)),)


def test_one_authoring_lands_at_the_same_depth_on_all_four_walls():
    model = _open(_compile(_ring(child_factory=_probe)))
    probes = _by_wall(model, "IfcBuildingElementProxy")
    assert set(probes) == set(_LEAVES)

    for leaf, product in probes.items():
        depth = _depth_from_outer_face(leaf, _origin(product))
        # The box spans local y 50..80, so its CENTRE is 65 mm in from the
        # outer face — on every wall, from one authoring, with no sign.
        assert depth == pytest.approx(0.065, abs=1e-9), (
            f"{leaf}: probe centre sits {depth * 1000:.1f} mm from its own "
            f"outer face, not 65 mm")


def test_the_four_walls_are_exact_quarter_turn_images():
    """Nothing is mirrored and nothing is offset: the west/north/east probes
    are the south one turned 90/180/270 degrees about Z."""
    model = _open(_compile(_ring(child_factory=_probe)))
    probes = _by_wall(model, "IfcBuildingElementProxy")
    south = _origin(probes["south"])
    for turns, leaf in ((1, "east"), (2, "north"), (3, "west")):
        assert _origin(probes[leaf]) == pytest.approx(
            _rot_z(south, turns), abs=1e-9), (
            f"{leaf} is not the south probe turned {turns * 90} degrees")


def test_no_skill_reintroduces_a_facing_sign():
    """``local_y_sign`` is deleted from the skills and must not come back —
    it exists only to undo a frame that has nothing to undo."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    offenders = [
        p for p in (root / "skills").rglob("*.py")
        if "local_y_sign" in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert offenders == [], (
        f"per-facade sign helper reappeared in {[str(p) for p in offenders]}")


# ---------------------------------------------------------------------------
# 3. inset=n and a child at y=n are the same plane
# ---------------------------------------------------------------------------


def _probe_at_80(_leaf):
    """A box whose NEAR face is authored at local y = 80 — the same 80 the
    ``inset=`` test below hands to the anchor."""
    return (Box(name="probe80", start=Point(x=100, y=80, z=100),
                end=Point(x=200, y=110, z=200)),)


def test_inset_and_a_child_depth_measure_the_same_axis_from_the_same_zero():
    flush = _open(_compile(_ring(child_factory=_probe_at_80, inset=0)))
    recessed = _open(_compile(_ring(inset=80)))

    child_by_wall = _by_wall(flush, "IfcBuildingElementProxy")
    fill_by_wall = {w.Name.rsplit(":", 2)[-1]: w
                    for w in recessed.by_type("IfcWindow")}

    for leaf in _LEAVES:
        # inset=80 puts the FILL's origin 80 mm in from the outer face …
        fill_depth = _depth_from_outer_face(leaf, _origin(fill_by_wall[leaf]))
        assert fill_depth == pytest.approx(0.080, abs=1e-9), leaf
        # … and a child authored at local y=80 starts on exactly that plane
        # (its centre is half its 30 mm depth further in).
        child_depth = _depth_from_outer_face(leaf, _origin(child_by_wall[leaf]))
        assert child_depth - 0.015 == pytest.approx(fill_depth, abs=1e-9), leaf


def test_inset_zero_is_flush_with_the_facade():
    model = _open(_compile(_ring(inset=0)))
    for w in model.by_type("IfcWindow"):
        leaf = w.Name.rsplit(":", 2)[-1]
        assert _depth_from_outer_face(leaf, _origin(w)) == pytest.approx(
            0.0, abs=1e-9), f"{leaf}: inset=0 is not flush with the facade"


def test_the_void_still_cuts_the_full_thickness_centred():
    """The datum moved to the outer face; the HOLE did not move at all."""
    model = _open(_compile(_ring()))
    for void in model.by_type("IfcOpeningElement"):
        leaf = void.Name.split(":wall:")[-1].removesuffix("_void")
        depth = _depth_from_outer_face(leaf, _origin(void))
        assert depth == pytest.approx(0.150, abs=1e-9), leaf
        solid = void.Representation.Representations[0].Items[0]
        assert solid.SweptArea.YDim == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 4. The frame table itself — recomputed, not quoted
# ---------------------------------------------------------------------------


#: ``leaf -> (local +x in world, local +y in world)``, from the spec.
_EXPECTED_AXES = {
    "south": ((1, 0), (0, 1)),
    "north": ((-1, 0), (0, -1)),
    "east": ((0, 1), (-1, 0)),
    "west": ((0, -1), (1, 0)),
}


def test_the_frame_is_the_south_wall_on_every_facade():
    from lite_step.compiler import frames
    from lite_step.ifc.generator import _derive_opening_frame

    proj = normalize_project_to_meters(_ring())
    frames.stamp_frames(proj)
    for wall in proj.storeys[0].elements:
        frame = _derive_opening_frame(wall._elements[0])
        right = (frame.dir_x, frame.dir_y)
        inward = frame.inward
        assert right == pytest.approx(_EXPECTED_AXES[wall.name][0], abs=1e-9), wall.name
        assert inward == pytest.approx(_EXPECTED_AXES[wall.name][1], abs=1e-9), wall.name
        # right x inward = up: right-handed on every facade.
        cross_z = right[0] * inward[1] - right[1] * inward[0]
        assert cross_z == pytest.approx(1.0, abs=1e-9), wall.name
        # The origin is ON the outer face, not on the midline.
        assert _depth_from_outer_face(wall.name, frame.wall_start) == \
            pytest.approx(0.0, abs=1e-9), wall.name


def test_the_frame_rotation_angles_match_the_spec_table():
    """south +0°, north +180°, east +90°, west -90° (== 270°) — all pure
    rotations about Z, which is the whole reason nothing is mirrored."""
    from lite_step.compiler import frames
    from lite_step.ifc.generator import _derive_opening_frame

    expected = {"south": 0.0, "north": 180.0, "east": 90.0, "west": 270.0}
    proj = normalize_project_to_meters(_ring())
    frames.stamp_frames(proj)
    for wall in proj.storeys[0].elements:
        frame = _derive_opening_frame(wall._elements[0])
        deg = math.degrees(math.atan2(frame.dir_y, frame.dir_x)) % 360.0
        assert deg == pytest.approx(expected[wall.name], abs=1e-6), wall.name


# ---------------------------------------------------------------------------
# 5. the docstring an author actually reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls_name", ["Window", "Door"])
def test_the_add_docstring_does_not_promise_world_coordinates(cls_name):
    """``Window.add`` / ``Door.add`` said "in WORLD coordinates" for as long as
    the frame has existed, contradicting their own class docstrings and
    ``agents/dsl-reference.md`` (§5: "Unit children live in the opening's local
    frame, identical on every facade").

    The line is one shared sentence pasted onto seven ``.add()`` overrides.
    On ``Column``/``Beam``/``Slab``/``Roof``/``Element`` it is true — those
    children ARE world geometry. On the two opening units it is not, because
    an opening unit carries no coordinates of its own. So this is a guard
    against the paste coming back, not against a typo: the wrong half of a
    correct sentence.

    Wrong in the expensive direction, too. A child authored at the world
    coordinates the docstring invites is a legal call that compiles clean and
    puts the glass metres from its hole — see
    ``test_a_world_coordinate_child_lands_nowhere_near_the_hole`` below.
    """
    import lite_step.models as models

    doc = getattr(models, cls_name).add.__doc__ or ""
    assert "WORLD coordinates" not in doc, (
        f"{cls_name}.add() promises world coordinates again — its children are "
        f"opening-local, and the reference tells the author so"
    )
    assert "OPENING-LOCAL" in doc, (
        f"{cls_name}.add() does not name the frame its children are in"
    )


@pytest.mark.parametrize("cls_name", ["Column", "Beam", "Slab", "Roof", "Element"])
def test_the_other_containers_still_say_world(cls_name):
    """The other half of the same sentence, so a future sweep cannot 'fix'
    all seven copies in the wrong direction. These children really are world
    geometry — the fix above is narrow on purpose."""
    import lite_step.models as models

    doc = getattr(models, cls_name).add.__doc__ or ""
    assert "WORLD coordinates" in doc, (
        f"{cls_name}.add() stopped saying its children are world geometry"
    )


def test_a_world_coordinate_child_lands_nowhere_near_the_hole():
    """Why the docstring mattered: nothing raises, the number is just wrong.

    A pane authored at opening-local ``(0, 0, 0)`` resolves to the hole. The
    same pane authored at the world coordinates the wall body uses resolves a
    further wall-length away, because the host transform is applied on top of
    whatever was written.
    """
    from lite_step.models import Box, Point, Project, Storey, Wall, Window

    def _built(pane_start_x):
        proj = Project(name="doc_frame")
        storey = Storey(name="ground", elevation=0)
        wall = Wall(name="south")
        wall.add(Box(name="body", start=Point(x=-3000, y=-200, z=0),
                     end=Point(x=3000, y=0, z=2400)))
        win = Window(name="w1", width=1200, height=1400)
        win.add(Box(name="glass", color="glass",
                    start=Point(x=pane_start_x, y=0, z=0),
                    end=Point(x=pane_start_x + 1200, y=20, z=1400)))
        wall.opening(win, along=1000, up=900)
        storey.add(wall)
        proj.add_storey(storey)
        return win._elements[0].world_aabb()

    local = _built(0)
    # z is the cleanest axis: opening-local z=0 IS the sill, at up=900.
    assert local.min.z == pytest.approx(900), (
        "an opening-local child stopped landing at the sill — this test's "
        "premise is gone, not just its conclusion"
    )
    assert local.min.x != pytest.approx(0), (
        "the child was NOT transformed by its host — if authored coordinates "
        "survive to world, the docstring's 'WORLD' claim was true after all"
    )

    world_written = _built(-3000)          # the wall body's own left edge
    assert world_written.min.x != pytest.approx(local.min.x), (
        "authoring the pane in world coordinates put it in the same place as "
        "authoring it opening-local — the frames would be interchangeable"
    )
    assert abs(world_written.min.x - local.min.x) == pytest.approx(3000), (
        "the offset between the two authorings should be exactly what was "
        "written into the child, applied on top of the host transform"
    )
