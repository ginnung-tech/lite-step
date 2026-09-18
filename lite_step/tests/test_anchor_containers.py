"""``.anchor()`` accepts a CONTAINER — an assembly authored once in its own
local frame and mounted on a host, its children resolving in ITS frame, which
resolves in the host's, recursively.

The cross-check pattern is the one
``infrastructure/orchestration/authoring-queries-tests/scenario_b_anchor_chain.py``
already uses for the ``placement=Transform`` chain: build the same answer a
SECOND time with plain numpy, from the authored inputs plus the published
:class:`~lite_step.compiler.frames.ElementFrame` fields, and compare. The
composition under test never appears in the cross-check — ``local_to_world``,
``_bake_point`` and ``_bake_subtree`` are all absent from this file.
"""
import math

import numpy as np
import pytest

from lite_step.compiler.frames import ChildAnchorError
from lite_step.models import Box, Element, Extrude, Point, Project, Sweep, Wall

MAT = "Concrete_C30-37"


# ---------------------------------------------------------------------------
# Independent numpy composition (nothing here imports the code under test)
# ---------------------------------------------------------------------------


def _frame_matrix(frame):
    """The host frame as a 4x4, from the PUBLISHED ``ElementFrame`` fields.

    ``.anchor(along=, inset=, up=)`` is documented as: stand in FRONT of the
    element's outer face, start at the lower-left corner you see, run
    ``along`` to your right, ``inset`` INTO the element, ``up`` its height.
    On a ``facing=+1`` host the outer face is the ``+across`` one, and
    walking round to stand in front of it reverses BOTH in-plane axes —
    ``along`` as well as ``across`` — so the origin steps to the
    along-max/across-max corner and both columns flip. Reversing only
    ``across`` (what the bake did before that) would make this a
    reflection; ``test_the_bake_is_a_rotation_on_every_host`` measures the
    determinant to keep that honest.
    """
    a = np.array(frame.along, dtype=float)
    c = np.array(frame.across, dtype=float)
    u = np.array(frame.up, dtype=float)
    s = frame.facing
    m = np.eye(4)
    m[:3, 0] = -s * a
    m[:3, 1] = -s * c
    m[:3, 2] = u
    m[:3, 3] = (np.array(frame.origin, dtype=float)
                + a * (frame.extent_along if s > 0 else 0.0)
                + c * (frame.thickness if s > 0 else 0.0))
    return m


def _rz(centideg):
    t = math.radians(centideg / 100.0)
    m = np.eye(4)
    m[0, 0], m[0, 1] = math.cos(t), -math.sin(t)
    m[1, 0], m[1, 1] = math.sin(t), math.cos(t)
    return m


def _translate(along, inset, up):
    m = np.eye(4)
    m[:3, 3] = (float(along), float(inset), float(up))
    return m


def _step(frame, along, inset, up, rotations=()):
    """One generation: host frame ∘ anchor offset ∘ the child's own rotation.

    The rotation is RIGHTMOST — it acts on the child's own coordinates first,
    before the host frame moves the whole assembly. That order is the claim
    this file exists to falsify.
    """
    total = sum(cd for _axis, cd in rotations or ())
    return _frame_matrix(frame) @ _translate(along, inset, up) @ _rz(total)


class _DerivedFrame:
    """A fallback-rule ``ElementFrame``, re-derived from FIRST PRINCIPLES.

    The chain check must not read the library's frame for the intermediate
    generations: each of those frames is itself an output of the bake being
    tested, so composing through them would only prove the last hop. This
    rebuilds them from the module's own documented rules — world axes for the
    fallback rule, origin at the AABB min corner — applied to a world box this
    file computed, never one the compiler reported.

    ``facing`` follows the two-part rule: an element the host
    ``.anchor()``ed INHERITS the host's, restated on its own ``across``; every
    other element takes the across-face FARTHER from the containment scope's
    centre. Every generation of this chain carries the fallback rule's world
    axes, so ``across`` is ``(0, 1, 0)`` throughout and the restatement is the
    identity — pass the host's number straight down.

    The distinction is not decorative here. Turn a 2000 mm assembly 90 degrees
    and it reaches 1610 mm out the ``-y`` side of the 400 mm host it is mounted
    on; the scope-centre rule reads that overhang as "this side is farther from
    the middle of the building" and declares the assembly's exterior to be the
    face its host calls INTERIOR, which then places the next generation on the
    wrong face. The un-rotated chain is the control: everything stays inside
    the host body, both rules answer ``+1``, and the composition is unchanged.
    """

    along = (1.0, 0.0, 0.0)
    across = (0.0, 1.0, 0.0)
    up = (0.0, 0.0, 1.0)

    def __init__(self, world_box, scope_center, inherited_facing=None):
        lo, hi = np.asarray(world_box[0], float), np.asarray(world_box[1], float)
        ref = float(scope_center[1])
        self.origin = tuple(lo)
        self.thickness = float(hi[1] - lo[1])
        self.extent_along = float(hi[0] - lo[0])
        self.facing = (inherited_facing if inherited_facing is not None
                       else (1 if abs(hi[1] - ref) >= abs(lo[1] - ref) else -1))


def _same_frame(derived, stamped) -> bool:
    return (np.allclose(derived.origin, stamped.origin, atol=1e-6)
            and derived.along == stamped.along
            and derived.across == stamped.across
            and derived.up == stamped.up
            and derived.facing == stamped.facing
            and math.isclose(derived.thickness, stamped.thickness, abs_tol=1e-6)
            and math.isclose(derived.extent_along, stamped.extent_along,
                             abs_tol=1e-6))


def _apply(matrix, point):
    v = matrix @ np.array([float(point[0]), float(point[1]), float(point[2]), 1.0])
    return v[:3]


def _corners(box):
    return np.array([[box.start.x, box.start.y, box.start.z],
                     [box.end.x, box.end.y, box.end.z]], dtype=float)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _box(name, x0, y0, z0, x1, y1, z1):
    return Box(name=name, start=Point(x=x0, y=y0, z=z0),
               end=Point(x=x1, y=y1, z=z1), material=MAT)


def _assembly(name, box):
    el = Element(ifc_class="IfcBuildingElementProxy", name=name)
    el.add(box)
    return el


def _chain(rotations=(None, None, None)):
    """A 4-generation chain: g0 hosts g1 hosts g2 hosts g3.

    Every assembly's own body sits strictly INSIDE its host's body, so a
    host's frame is the same before and after its children are baked into it
    — asserted below, because the cross-check reads the frames post-resolve
    and would otherwise be comparing against a frame the bake never saw.
    """
    proj = Project(name="container anchor chain")
    specs = [
        # (along, inset, up), own local body
        ((1000, 10, 200), (0, 0, 0, 2000, 300, 500)),
        ((500, 20, 100), (0, 0, 0, 800, 200, 200)),
        ((200, 30, 50), (0, 0, 0, 300, 100, 100)),
    ]
    g0 = _assembly("g0", _box("b0", 0, 0, 0, 6000, 400, 3000))
    proj.add(g0)

    gens = [g0]
    host = g0
    for i, (offset, local) in enumerate(specs, start=1):
        child = _assembly(f"g{i}", _box(f"b{i}", *local))
        along, inset, up = offset
        host.anchor(child, along=along, inset=inset, up=up,
                    rotations=rotations[i - 1])
        gens.append(child)
        host = child
    return proj, gens, specs


def _locals(specs):
    return [np.array(local, dtype=float).reshape(2, 3) for _off, local in specs]


# ---------------------------------------------------------------------------


def _compose_chain(specs, rotations=(None, None, None)):
    """The whole chain, generation by generation, in numpy only.

    Returns ``(world boxes, frames)`` per generation. Nothing here reads the
    compiler: each generation's frame is re-derived from the world box the
    PREVIOUS generation produced — plus that generation's ``facing``, which is
    inherited rather than re-derived — which is what makes this a composition
    rather than a per-hop spot check.
    """
    body0 = np.array([[0.0, 0.0, 0.0], [6000.0, 400.0, 3000.0]])
    scope_center = (body0[0] + body0[1]) / 2.0
    boxes = [(body0[0], body0[1])]
    frames = [_DerivedFrame(boxes[0], scope_center)]     # g0 is not anchored
    for (offset, local), rot in zip(specs, rotations):
        matrix = _step(frames[-1], *offset, rotations=rot)
        corners = np.array(local, dtype=float).reshape(2, 3)
        placed = np.array([_apply(matrix, corners[0]), _apply(matrix, corners[1])])
        boxes.append((placed.min(axis=0), placed.max(axis=0)))
        frames.append(_DerivedFrame(boxes[-1], scope_center,
                                    inherited_facing=frames[-1].facing))
    return boxes, frames


def test_three_generation_chain_matches_numpy_composition():
    """g0 -> g1 -> g2 -> g3, cross-checked against an independent composition."""
    proj, gens, specs = _chain()
    g0, g1, g2, g3 = gens

    # Triggering the resolve through the public query API is half the point:
    # the acceptance is that world_aabb() ANSWERS, not that the tree bakes.
    deep = g3._elements[0]
    world = deep.world_aabb()

    expected, derived = _compose_chain(specs)
    lo, hi = expected[-1]
    assert np.allclose([world.min.x, world.min.y, world.min.z], lo, atol=1e-6)
    assert np.allclose([world.max.x, world.max.y, world.max.z], hi, atol=1e-6)

    # obb() on a plain Box degrades to the AABB by design; assert it agrees
    # rather than assuming, since the acceptance names both accessors.
    ob = deep.obb()
    assert np.allclose([ob.min.x, ob.min.y, ob.min.z], lo, atol=1e-6)
    assert np.allclose([ob.max.x, ob.max.y, ob.max.z], hi, atol=1e-6)

    # Every intermediate generation landed too — a chain that only got the last
    # hop right would sail through an end-to-end check.
    for gen, (elo, ehi), frame in zip(gens, expected, derived):
        body = gen._elements[0]
        assert np.allclose(_corners(body)[0], elo, atol=1e-6), gen.name
        assert np.allclose(_corners(body)[1], ehi, atol=1e-6), gen.name
        # ... and the frame this file derived is the frame the compiler
        # stamped, so the two derivations have not silently diverged.
        assert _same_frame(frame, gen._frame), \
            f"{gen.name}: derived frame disagrees with the stamp"
        assert _contains(gen, body), (
            f"{gen.name}: the assembly escaped its host body, so the "
            f"post-resolve frame is not the frame the bake saw")


def _contains(host, body):
    """True when every anchored descendant of ``host`` lies inside ``body``."""
    lo = np.array([body.start.x, body.start.y, body.start.z], dtype=float)
    hi = np.array([body.end.x, body.end.y, body.end.z], dtype=float)
    stack = [c for c in host._elements if c is not body]
    while stack:
        elem = stack.pop()
        box = getattr(elem, "start", None)
        if box is not None:
            pts = _corners(elem)
            if (pts.min(axis=0) < lo - 1e-9).any() or (pts.max(axis=0) > hi + 1e-9).any():
                return False
        stack.extend(getattr(elem, "_elements", None) or [])
    return True


def test_rotations_compose_through_two_generations():
    """A rotation on an anchored container turns its children BEFORE the host
    frame places them — reverse the order and the answer moves."""
    rotations = ([("z", 9000)], [("z", 9000)], None)
    proj, gens, specs = _chain(rotations=rotations)
    deep = gens[-1]._elements[0]
    world = deep.world_aabb()

    expected, _derived = _compose_chain(specs, rotations)
    lo, hi = expected[-1]
    assert np.allclose([world.min.x, world.min.y, world.min.z], lo, atol=1e-6)
    assert np.allclose([world.max.x, world.max.y, world.max.z], hi, atol=1e-6)

    # Falsifier: composing the rotation on the WRONG side (host frame first,
    # then the turn) gives a different answer, so the assertion above is
    # actually discriminating and not passing on a symmetry.
    frame0 = _DerivedFrame(((0, 0, 0), (6000, 400, 3000)), (3000, 200, 1500))
    right = _step(frame0, *specs[0][0], rotations=[("z", 9000)])
    wrong = _frame_matrix(frame0) @ _rz(9000) @ _translate(*specs[0][0])
    assert not np.allclose(right, wrong)


def test_add_widened_too_and_the_two_verbs_still_differ():
    """The inverse of the narrow claim, for the same reason ``.anchor()`` has
        no table.

    ``.add()`` refused a container child on the grounds that world-coordinate
    containment was "a different (weaker) claim". It IS a different claim —
    but a weaker claim about a child the compiler emits identically either
    way is not a reason to refuse it, and the author routed around it by
    wrapping the child in ``Element(ifc_class=…)``, which produced the same
    aggregate through one more object.

    What still separates the two verbs is what happens to the COORDINATES,
    which is asserted in ``test_add_takes_products.py``: ``.add()`` leaves
    them alone, ``.anchor()`` bakes them through the host's frame."""
    host = _assembly("host", _box("b", 0, 0, 0, 1000, 200, 1000))
    sub = _assembly("sub", _box("s", 0, 0, 0, 100, 100, 100))
    assert host.add(sub) is host
    assert sub in host.elements
    assert getattr(sub, "_anchor_spec", None) is None, (
        ".add() must not stamp an anchor — that is the other verb")


def test_wall_anchor_takes_a_container_and_emits_it():
    """The inverse of the narrow claim.

    ``Wall`` was left out of the widening on the grounds that its emitter
    had no nested-container branch, so legalising it would trade a clear
    refusal for a warn-and-drop. The right answer was to give the Wall the
    branch (``_aggregate_wall_details``), not to keep the refusal — the
    workaround the refusal pushed authors to (wrap it in an
    ``Element(ifc_class="IfcWall")``) produced the same geometry through one
    more object. The assertion is on the compiled file, not on the call
    returning, because a warn-and-drop also lets the call return."""
    from lite_step.compiler.executor import normalize_project_to_meters
    from lite_step.ifc.generator import generate_ifc

    wall = Wall(name="w")
    wall.add(_box("body", 0, 0, 0, 5000, 200, 3000))
    wall.anchor(_assembly("sub", _box("s", 0, 0, 0, 100, 100, 100)), along=100)
    proj = Project(name="p")
    proj.add(wall)

    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    assert "sub" in result.ifc_content


def test_primitive_anchor_takes_no_new_ceremony():
    """The pre-existing leaf case is untouched — same call, same numbers."""
    proj = Project(name="primitive")
    wall = Wall(name="south")
    wall.add(_box("body", -3000, 0, 0, 3000, 200, 2500))
    shelf = _box("shelf", 0, 0, 0, 400, 100, 40)
    wall.anchor(shelf, along=1000, inset=0, up=900)
    proj.add(wall)

    world = shelf.world_aabb()
    expected = np.array([_apply(_step(wall._frame, 1000, 0, 900), (0, 0, 0)),
                         _apply(_step(wall._frame, 1000, 0, 900), (400, 100, 40))])
    assert np.allclose([world.min.x, world.min.y, world.min.z],
                       expected.min(axis=0), atol=1e-6)


def test_anchored_container_and_its_descendants_skip_the_carve_pass():
    """Exempted an anchored child from the inferred carve. An anchored
    ASSEMBLY's descendants are exempt on the same grounds — abutting the host
    is what anchoring MEANS, and a bracket whose bricks carved the wall it
    hangs off would defeat the whole point."""
    from lite_step.compiler.displacement import _collect
    from lite_step.compiler.frames import resolve_child_anchors, stamp_frames
    from lite_step.compiler.naming import stamp_canonical_names

    proj = Project(name="carve exemption")
    host = _assembly("host", _box("body", 0, 0, 0, 4000, 300, 3000))
    bracket = _assembly("bracket", _box("brick", 0, 0, 0, 200, 200, 100))
    # DEEP inside the host, so an unexempted descendant would certainly carve.
    host.anchor(bracket, along=1000, inset=50, up=1000)
    proj.add(host)

    stamp_canonical_names(proj)
    stamp_frames(proj)
    resolve_child_anchors(proj)
    stamp_frames(proj)

    solids, _meshes = _collect(proj)
    names = {getattr(s, "name", None) for s, _aabb in solids}
    assert "body" in names, "the host body must still be a carve candidate"
    assert "brick" not in names, (
        "a brick inside an anchored bracket entered the inferred-carve pass")


def test_container_rotation_still_meets_each_child_s_own_rule():
    """A rotation carried down does not become a loophole: the Box
    quarter-turn rule fires on the DESCENDANT, naming it.

    It fires at the ``.anchor()`` LINE — the author is told
    which leaf refused while they are still looking at the call that asked
    for the rotation, rather than at compile depth inside the bake.
    """
    host = _assembly("host", _box("body", 0, 0, 0, 4000, 300, 3000))
    bracket = _assembly("bracket", _box("brick", 0, 0, 0, 200, 200, 100))

    with pytest.raises(ChildAnchorError, match="quarter turns") as exc:
        host.anchor(bracket, along=1000, up=100, rotations=[("z", 3000)])
    assert "'brick'" in str(exc.value), "must name the offending descendant"
    assert "'bracket'" in str(exc.value), "…and the child being anchored"
    # …and the refusal left nothing half-attached.
    assert bracket._anchor_spec is None
    assert bracket not in host._elements


def test_container_rotation_reaches_an_extrude_descendant():
    """The same 30-degree turn is fine on an Extrude — the contour rotates and
    its winding (hence the extrusion normal) rotates with it."""
    proj = Project(name="rotated extrude")
    host = _assembly("host", _box("body", 0, 0, 0, 4000, 300, 3000))
    panel = Element(ifc_class="IfcBuildingElementProxy", name="panel")
    panel.add(Extrude(name="leaf", thickness=50, material=MAT, contour=[
        Point(x=0, y=0, z=0), Point(x=600, y=0, z=0),
        Point(x=600, y=0, z=400), Point(x=0, y=0, z=400)]))
    host.anchor(panel, along=1000, up=100, rotations=[("z", 3000)])
    proj.add(host)

    host._elements[0].world_aabb()          # resolve
    leaf = panel._elements[0]
    pts = np.array([[p.x, p.y, p.z] for p in leaf.contour])
    edge = pts[1] - pts[0]
    # 600 mm run, turned 30 degrees in plan, then placed by the host frame.
    assert math.isclose(float(np.linalg.norm(edge)), 600.0, abs_tol=1e-6)
    assert not math.isclose(abs(float(edge[1])), 0.0, abs_tol=1e-6), \
        "a 30-degree turn must leave the run off the host's along axis"


def test_a_sweep_descendant_still_refuses_a_rotation():
    """The section-orientation refusal survives the container path — it is a
    property of the SWEEP, not of how the rotation arrived. Also at the
    ``.anchor()`` line, naming the Sweep."""
    host = _assembly("host", _box("body", 0, 0, 0, 4000, 300, 3000))
    frame = Element(ifc_class="IfcBuildingElementProxy", name="frame")
    frame.add(Sweep(name="post", material=MAT,
                    path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=500)],
                    profile=[Point(x=0, y=0, z=0)][:0] or None))

    with pytest.raises(ChildAnchorError, match="not bakeable") as exc:
        host.anchor(frame, along=1000, up=100, rotations=[("z", 3000)])
    assert "'post'" in str(exc.value)


def test_the_bake_still_refuses_a_subtree_built_after_the_anchor():
    """The call-site check is early, not complete — and the backstop stays.

    An author may anchor an empty container and fill it afterwards, in which
    case there was nothing to walk at the ``.anchor()`` line. The bake-time
    refusal must still fire, or moving the error earlier would have DELETED
    a rule rather than relocating it.
    """
    proj = Project(name="late fill")
    host = _assembly("host", _box("body", 0, 0, 0, 4000, 300, 3000))
    bracket = Element(ifc_class="IfcBuildingElementProxy", name="bracket")
    host.anchor(bracket, along=1000, up=100, rotations=[("z", 3000)])  # empty: ok
    bracket.add(_box("brick", 0, 0, 0, 200, 200, 100))                 # now illegal
    proj.add(host)

    with pytest.raises(ChildAnchorError, match="quarter turns"):
        host._elements[0].world_aabb()


def test_an_anchored_assembly_reaches_the_compiled_ifc():
    """The capability is only real if the assembly SURVIVES to the file.

    ``_create_element_generic`` warned-and-skipped an off-table child, so
    without its nested-container branch a mounted assembly would author fine,
    query fine, and then quietly not exist in the IFC. Asserts the products
    are there, that the geometry is at the ANCHORED coordinates, and that the
    assembly cost zero booleans (the carve exemption, end to end).
    """
    from lite_step.compiler.executor import normalize_project_to_meters
    from lite_step.ifc.generator import generate_ifc

    proj = Project(name="nested assembly")
    gable = _assembly("gable", _box("body", 0, 0, 0, 6000, 300, 3000))
    bracket = _assembly("bracket", _box("brick", 0, 0, 0, 200, 150, 100))
    bracket.add(_box("brick2", 0, 0, 100, 300, 150, 200))
    gable.anchor(bracket, along=1000, inset=0, up=2000)
    proj.add(gable)

    normalized = normalize_project_to_meters(proj)
    result = generate_ifc(normalized)
    assert result.success, result.error
    ifc = result.ifc_content

    # wrapper + body, wrapper + two bricks
    assert ifc.count("IFCBUILDINGELEMENTPROXY") == 5
    assert "IFCBOOLEANRESULT" not in ifc, (
        "an anchored assembly carved its host — the exemption did not reach "
        "the compiled output")
    # The tree the backend actually serialized (normalization COPIES) holds
    # METRES at the anchored position — 1.0 along, 2.0 up — not the assembly's
    # own local origin.
    mounted = None
    for storey in normalized.storeys:
        for elem in storey.elements:
            for child in elem._elements:
                if child.name == "bracket":
                    mounted = child._elements
    assert mounted is not None, "the anchored assembly vanished in normalization"
    # The gable stands alone, so its facing tie-breaks to +1 (exterior on
    # +across): the viewer stands at +y and their right runs -x. ``along=1000``
    # therefore starts 1.0 m in from the x-MAX end — 6.0 - 1.0 - width — and
    # the two bricks (0.2 m and 0.3 m wide) land at 4.8 and 4.7. Before
    # Both read 1.0, because the bake reversed ``across``
    # without reversing ``along`` and so mirrored instead of rotating.
    assert [round(b.start.x, 3) for b in mounted] == [4.8, 4.7]
    assert [round(b.start.z, 3) for b in mounted] == [2.0, 2.1]


def test_anchor_nesting_guard_is_finite():
    """A cycle in the tree must raise, not spin."""
    from lite_step.compiler.frames import _MAX_ANCHOR_GENERATIONS

    assert _MAX_ANCHOR_GENERATIONS > 3
