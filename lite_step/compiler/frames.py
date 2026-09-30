"""Authoritative element frames (WS0 of the frames + ``.anchor()`` cutover).

Without one place for it, "which way does this element run, and which side
is out" is re-derived from an AABB in six independent places (the Anchor host
table, the box-mode and contour-mode wall opening frames, the layer
through-axis, the sweep section reference) with three different verticality
thresholds and two spellings of the same
tie-break. That is the exact hazard ``lite_step/ifc/geometry.py`` documents
for ``segment_axes``: *a second, subtly different copy is how an AABB
silently stops matching the solid it describes*.

This module computes each element's frame ONCE, top-down, and stamps it on
the element as ``_frame``. Every consumer reads the stamp instead of
re-deriving. The pass mirrors :mod:`lite_step.compiler.naming` exactly — a
frame, like a canonical name, depends on the whole containment path and so
cannot be a bottom-up property. (``BimElement.parent`` now exists, but it is
stamped BY the naming walk; deriving frames from it would make this pass
depend on the output of another rather than on the tree.)

**The local origin is authoritative and uniform.** ``ElementFrame.origin``
is the element's AABB **min corner expressed in frame axes** — "lower left
back", i.e. ``(along-min, across-min, up-min)``. That point is local
``(0, 0, 0)`` for every rule; ``.anchor(child, along=, inset=, up=)`` measures
from it. The per-rule reference points that ``placement=Anchor``'s
``attach_to`` uses (wall baseline at base-z centered in thickness, beam
centerline, slab top surface) are DERIVED from the origin plus the extents
— they are attach conveniences, not the origin.

**The across axis comes from the authored extrusion direction when there is
one.** An ``Extrude`` body carries real authoring intent: the Newell normal
of its contour (winding order) is the direction the author extruded along.
For a wall-rule element that normal IS ``across``; for a slab-rule element
it is ``up``. Only box-mode bodies — which carry no authored direction —
fall back to the AABB longer-horizontal-axis rule (ties -> +X), preserving
today's behaviour bit-for-bit on the existing corpus.

**"What defines my frame" and "how much space do I take up" are two
questions, and this module answers them with two functions.**
:func:`frame_basis_aabb` is the element plus every descendant that is NOT
anchored into it; :func:`element_aabb` is everything the element occupies,
anchored children included. Only :func:`frame_for` and :func:`_scope_aabb`
read the basis. Everything else — orthographic view extents, ``world_aabb()``
/ ``obb()``, the layer datum, the displacement trigger — reads the full
extent, and would silently CLIP an anchored cornice if it read the basis
instead.

The split exists because ``.add()`` and ``.anchor()`` make different claims
about coordinates. ``.add(child)`` says "already world", so the child helps
define its container's frame. ``.anchor(child)`` says "relative to your
frame", so the child was positioned BY that frame and cannot be part of what
derives it. Overloading one function with both meanings is how the two
start to disagree — and did: see :func:`frame_basis_points` for the measured
symptoms.

**Facing** (``+1`` = exterior on the ``+across`` side) points AWAY from the
center of the element's nearest containment scope, walking outward:
``Space -> Storey -> Building -> Site``. This generalizes the previous rule,
which was literally "the face farther from the world **origin**" and was
only correct while RULE 4 ("building centered at origin") held. On a
RULE-4-compliant model both rules agree; an off-origin model stops getting
inverted exteriors. An authored ``LayerSet.outward`` still wins outright.

Units follow the project instance the pass runs on (mm before
``normalize_project_to_meters``, meters after). The pass is a full
recompute, never incremental, so it is stamped at every entry point and
staleness cannot survive one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lite_step.compiler.walkguard import WalkGuard

Vec3 = Tuple[float, float, float]
AABB = Tuple[Vec3, Vec3]

#: Container attrs the frame walk descends into. Only real containment —
#: boolean operands (``_cuts``/``_adds``/...) carry no frame of their own
#: (a void often extends far past its host on purpose), and ``_openings``
#: attachments are placed by the host's frame, not their own.
_WALK_ATTRS = ("_elements",)

#: generic ``Element`` ifc_class -> frame rule (sugar containers map by type).
#: THE rule — ownership moved here, and the ``placement`` alias that
#: mirrored it is gone (nothing imported it).
ELEMENT_CLASS_RULES = {"IfcWall": "wall", "IfcBeam": "beam", "IfcSlab": "slab"}

#: A contour whose normal is this close to horizontal counts as a vertical
#: plane (a wall face). Matches ``_derive_contour_wall_frame``'s threshold
#: so the migrated contour path reproduces today's accept/reject decisions.
VERTICAL_PLANE_NZ_MAX = 0.01


def outer_face_axis(out_sign: int, axis: Vec3) -> Vec3:
    """Reverse ``axis`` for a viewer standing in FRONT of the outer face.

    THE single derivation of that reversal. Two callers read it:
    :func:`opening_right_axis` (the opening frame's ``+x``) and
    :meth:`ElementFrame.local_to_world` (the generic ``.anchor()`` bake).
    They are one idea, and two implementations of one idea agreeing
    *usually* is what produced and — so there is one.

    An element's raw ``along``/``across`` are geometry: they point wherever
    the AABB or the authored contour put them, with no relation to which
    side is out. The authoring frame is the element's own **seen from
    outside** — stand where the camera stands, look at the facade, ``+x``
    is your right and ``+y`` runs INTO the element from its outer face.
    Walking around to the front of the other face reverses BOTH of those:
    with ``out_sign = +1`` (exterior on the ``+across`` side) right becomes
    ``-along`` and inward becomes ``-across``.

    **Reversing both is what makes it a rotation.** ``det`` of
    ``[-s*along, -s*across, up]`` is ``s**2 = +1`` for either sign, so
    every host is the south wall's frame turned about Z. Reversing one
    alone — which the anchor bake did until then — has ``det = -s`` and is
    a reflection, and ``IfcAxis2Placement3D`` cannot encode a mirror: it
    does not raise, the contour winding survives unmirrored, and the solid
    runs backwards through its host.
    """
    return (-out_sign * axis[0], -out_sign * axis[1], -out_sign * axis[2])


def facing_from(ref_across: Vec3, ref_facing: int, across: Vec3) -> Optional[int]:
    """Restate a reference element's ``facing`` on an ``across`` axis of one's own.

    THE single projection of one element's "which side is out" onto another
    element's raw ``across``, and there is one for the reason
    :func:`outer_face_axis` is one: two implementations of a signed projection
    agreeing *usually* is what produced and. Two readers,
    and they ask the same question about different references —
    :func:`opening_out_sign` projects the host's own stamp onto the run axis
    the opening actually opens along, and :func:`_facing_for` projects an
    anchoring host's stamp onto the axes an anchored child derived for itself.

    ``None`` means the two axes are PERPENDICULAR, and it is a real answer
    rather than a failure: a reference whose ``across`` is at right angles to
    yours says nothing at all about which side of *your* ``across`` the
    exterior lies on. Each caller decides what to do with that — degrade to
    the reference's raw facing, or fall through to the rule it would have used
    without a reference.
    """
    dot = (ref_across[0] * across[0]
           + ref_across[1] * across[1]
           + ref_across[2] * across[2])
    if abs(dot) < 1e-9:
        return None
    return ref_facing if dot > 0 else -ref_facing


@dataclass(frozen=True)
class ElementFrame:
    """One element's authoritative local frame.

    ``origin`` is local ``(0, 0, 0)``: the AABB min corner in frame axes
    (along-min, across-min, up-min) — "lower left back". ``along``/
    ``across``/``up`` are right-handed unit vectors (``along x across =
    up``); ``across`` is the RAW geometric axis and is NOT flipped by
    ``facing``. Extents are measured along the frame axes.
    """

    origin: Vec3
    along: Vec3
    across: Vec3
    up: Vec3
    facing: int          # +1: exterior on the +across side; -1: on -across
    rule: str            # "wall-box" | "wall-contour" | "beam" | "slab" | "fallback"
    extent_along: float
    thickness: float     # extent along ``across``
    height: float        # extent along ``up``

    # -- derived attach points (conveniences; the origin stays the corner) --

    def point(self, name: str) -> Vec3:
        """Attach point in authoring space for ``attach_to`` semantics.

        ``start``/``end`` are the ends of the rule's anchor LINE, ``center``
        the 3D AABB center, ``face`` the exterior across-face (wall) or top
        face (everything else) — the pinned semantics of the previous
        ``_host_frame.points`` table, now derived from the corner origin.
        """
        if name == "center":
            return self._local_to_world(
                self.extent_along / 2.0, self.thickness / 2.0, self.height / 2.0
            )
        if name == "face":
            if self.rule.startswith("wall"):
                across_c = self.thickness if self.facing > 0 else 0.0
                return self._local_to_world(
                    self.extent_along / 2.0, across_c, self.height / 2.0
                )
            return self._local_to_world(
                self.extent_along / 2.0, self.thickness / 2.0, self.height
            )
        if name in ("start", "end"):
            along_c = 0.0 if name == "start" else self.extent_along
            return self._local_to_world(along_c, self.thickness / 2.0, self._line_up())
        raise ValueError(f"unknown attach point {name!r}")

    def _line_up(self) -> float:
        """Height of the rule's anchor line above the frame origin."""
        if self.rule.startswith("wall"):
            return 0.0                    # baseline at the wall base
        if self.rule == "beam":
            return self.height / 2.0      # centerline
        return self.height                # slab / fallback: top surface

    def _local_to_world(self, along_c: float, across_c: float, up_c: float) -> Vec3:
        return (
            self.origin[0] + self.along[0] * along_c + self.across[0] * across_c + self.up[0] * up_c,
            self.origin[1] + self.along[1] * along_c + self.across[1] * across_c + self.up[1] * up_c,
            self.origin[2] + self.along[2] * along_c + self.across[2] * across_c + self.up[2] * up_c,
        )

    def local_to_world(self, along_c: float, inset_c: float, up_c: float) -> Vec3:
        """Map ``.anchor(child, along=, inset=, up=)`` to authoring space.

        THE mental model, and it is the same one on every wall of the
        building: stand where the camera stands (``y-``, a little up) and
        look at the south facade. That view IS this frame.

        * origin — the element's lower-left corner on its **OUTER** face
        * ``along`` — right, as seen from outside
        * ``inset`` — **into** the element, depth from the outer face
        * ``up`` — up

        So the outward normal is local ``(0, -1, 0)`` and everything an
        author writes sits in the positive octant. ``inset=0`` is flush with
        the facade, ``inset=50`` sits 50 mm behind it, ``inset=-20`` is proud
        of it.

        The south wall is the identity and every other host is that frame
        ROTATED about Z — nothing is ever mirrored. The code EARNS that:
        both in-plane axes come from the one :func:`outer_face_axis`
        reversal, so the basis determinant is ``+1`` on either ``facing``.
        (Reversing ``across`` alone — which this did until then — is a reflection, and it fails silently rather
        than loudly: ``IfcAxis2Placement3D`` cannot express a mirror, so an
        anchored ``Extrude`` simply ran backwards through its host.)
        """
        right = outer_face_axis(self.facing, self.along)
        inward = outer_face_axis(self.facing, self.across)
        # Local (0, 0, 0) is the corner ``right`` and ``inward`` both point
        # AWAY from: the raw origin on a ``facing=-1`` host, the
        # along-max/across-max corner on a ``facing=+1`` one.
        ox, oy, oz = self._local_to_world(
            self.extent_along if self.facing > 0 else 0.0,
            self.thickness if self.facing > 0 else 0.0,
            0.0,
        )
        return (
            ox + right[0] * along_c + inward[0] * inset_c + self.up[0] * up_c,
            oy + right[1] * along_c + inward[1] * inset_c + self.up[1] * up_c,
            oz + right[2] * along_c + inward[2] * inset_c + self.up[2] * up_c,
        )


# ---------------------------------------------------------------------------
# Authoring-space geometry probes (moved here from placement.py — frames is
# the lower layer; placement imports these back).
# ---------------------------------------------------------------------------


def iter_geometry_points(elem, skip_anchored: bool = True, _on_path=None):
    """World/authoring-space Points that define ``elem``'s geometry.

    Walks the element's own point-bearing fields plus container children.
    Window/Door subtrees are SKIPPED — their children live in opening-LOCAL
    coordinates and would corrupt the host AABB. Boolean operands are
    skipped too (voids often extend past the host on purpose).

    An UNRESOLVED anchored child is skipped for the same reason: until
    ``resolve_child_anchors`` bakes it, its coordinates are relative to the
    very frame being derived here, so counting them would drag the parent's
    origin toward the local origin — and then place the child against a
    frame its own presence corrupted. (Once baked it holds world
    coordinates and contributes normally.)

    A child already on the current path — a containment cycle — is skipped
    too, and this one is a TERMINATION guard rather than a correctness rule:
    the model is refused by ``validate_project_report`` before it reaches
    generation, and this walk only has to survive long enough to be asked.
    ``_on_path`` is threaded by the recursion and is not part of the API.
    """
    from lite_step.models.elements import Door, Window

    if isinstance(elem, (Window, Door)):
        return
    if skip_anchored and getattr(elem, "_anchor_spec", None) is not None:
        return
    guard = _on_path if _on_path is not None else WalkGuard()
    if not guard.enter(elem):
        return
    try:
        yield from _own_points(elem)
        for child in (getattr(elem, "_elements", None) or []):
            yield from iter_geometry_points(child, _on_path=guard)
    finally:
        guard.leave(elem)


#: The two field groups ``_own_points`` reads, in yield order. Scalars first —
#: some consumers read the sequence positionally, so the order is part of the
#: contract rather than an accident of how the loops were written.
_POINT_SCALARS = ("start", "end", "location")
_POINT_SEQUENCES = ("contour", "path", "vertices", "points")

#: ``type(elem)`` -> the subset of the above that class actually declares.
#: Keyed on the CLASS, so it is bounded by the number of DSL primitives (~20)
#: rather than by model size, and never needs invalidating: a pydantic model's
#: field set is fixed at class creation.
_POINT_FIELDS: dict = {}


def _point_fields(cls):
    """Which of the seven names ``cls`` declares.

    A blind ``getattr`` probe for all seven is what this replaces, and the
    misses were not free: measured on ``roof-covering``, 15,159 of 19,194
    probes (79%) missed, and a miss on a pydantic model does not stop at the
    instance ``__dict__`` — it falls through ``BaseModel.__getattr__`` into
    private-attr lookup, ``model_extra``, and the metaclass, before the
    ``AttributeError`` is swallowed by ``getattr(..., None)``. That frame was
    the #1 ``tottime`` entry in a cProfile of three stamp passes, at 46%.

    **Non-pydantic objects probe all seven.** Test stubs and hand-rolled
    stand-ins reach this walk and have no ``model_fields``; narrowing them to
    an empty tuple would make them silently point-less rather than raise.
    """
    got = _POINT_FIELDS.get(cls)
    if got is None:
        declared = getattr(cls, "model_fields", None)
        if declared is None:
            got = (_POINT_SCALARS, _POINT_SEQUENCES)
        else:
            got = (tuple(a for a in _POINT_SCALARS if a in declared),
                   tuple(a for a in _POINT_SEQUENCES if a in declared))
        _POINT_FIELDS[cls] = got
    return got


def _own_points(elem):
    """The Points ``elem`` itself carries — no recursion, no filtering.

    Shared verbatim by :func:`iter_geometry_points` and
    :func:`frame_basis_points`, because the two walks differ ONLY in which
    children they descend into. A second copy of this field list is how one
    of them would quietly stop seeing a primitive the other reads — the same
    hazard this module's header describes for ``segment_axes``.
    """
    from lite_step.models import Point

    scalars, sequences = _point_fields(type(elem))
    for attr in scalars:
        p = getattr(elem, attr, None)
        if isinstance(p, Point):
            yield p
    for attr in sequences:
        seq = getattr(elem, attr, None)
        if isinstance(seq, list):
            for p in seq:
                if isinstance(p, Point):
                    yield p


def _aabb_of(points) -> Optional[AABB]:
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for p in points:
        xs.append(float(p.x))
        ys.append(float(p.y))
        zs.append(float(p.z))
    if not xs:
        return None
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def element_aabb(elem) -> Optional[AABB]:
    """Axis-aligned bounds of everything ``elem`` OCCUPIES.

    Anchored children included, once baked — see
    :func:`frame_basis_aabb` for the other half of the split and why the two
    must not be merged back together.

    Revolve/Pipe contribute their axis/centerline path points only (radial
    extents not padded) — the documented limitation carried over from the
    placement engine.
    """
    return _aabb_of(iter_geometry_points(elem))


# ---------------------------------------------------------------------------
# The frame BASIS — a strict subset of "everything I occupy"
# ---------------------------------------------------------------------------


def is_anchored_child(elem) -> bool:
    """True when ``elem`` stands where its container's ``.anchor()`` put it.

    Two attributes spell one fact, because the bake consumes the first into
    the second: ``_anchor_spec`` while the anchor is pending,
    ``_anchor_resolved`` once :func:`resolve_child_anchors` has applied it.
    Reading only ``_anchor_spec`` — which is what ``iter_geometry_points``
    does — makes the answer FLIP at the bake, and that flip is: the child re-entered the very AABB its own placement was
    derived from, one generation late.
    """
    return (getattr(elem, "_anchor_spec", None) is not None
            or bool(getattr(elem, "_anchor_resolved", False)))


def frame_basis_points(elem, _on_path=None):
    """The points that DEFINE ``elem``'s frame — never the ones it merely holds.

    ``.add()`` and ``.anchor()`` mean different things about coordinates, and
    the frame is where that difference has to be honoured::

        .add(child)     -> "already world coordinates"   -> defines the frame
        .anchor(child)  -> "relative to YOUR frame"      -> cannot define it

    An anchored child is positioned BY the container's frame. Letting it back
    into the AABB that frame is derived from closes a loop: the container's
    thickness, run axis and facing all start depending on what was hung on it.
    Measured on ``main``: anchoring a 180 mm core leaf into a 180 mm wall took
    the host's ``thickness`` to 360, and anchoring any nested ``Wall`` flipped
    the host's own ``facing``, moving that wall's existing windows by 1.8 m.

    **The exclusion is on CHILDREN, and the element asked about is exempt.**
    A resolved anchored container holds true world coordinates and must
    derive its own frame from them — that frame is what the anchors authored
    INSIDE it resolve against in the next generation, and suppressing it
    would leave ``resolve_child_anchors`` unable to converge. An
    *unresolved* anchored root is still skipped, exactly as
    :func:`iter_geometry_points` skips it and for its reason: its points are
    not world coordinates yet, so there is nothing to derive a world frame
    from.

    Window/Door subtrees, boolean operands and on-path re-entry are handled
    exactly as in :func:`iter_geometry_points`. ``_on_path`` is threaded by
    the recursion and is not part of the API.
    """
    from lite_step.models.elements import Door, Window

    if isinstance(elem, (Window, Door)):
        return
    if getattr(elem, "_anchor_spec", None) is not None:
        return
    yield from _frame_basis_walk(elem, _on_path=_on_path)


def _frame_basis_walk(elem, _on_path=None):
    """The recursion of :func:`frame_basis_points`, minus the ROOT gates.

    Split out so :func:`has_frame_basis` can ask *do these points exist* on an
    element :func:`frame_basis_points` declines to derive a frame FROM. The
    children still go back through :func:`frame_basis_points`, so every gate
    applies below the root exactly as before.
    """
    guard = _on_path if _on_path is not None else WalkGuard()
    if not guard.enter(elem):
        return
    try:
        yield from _own_points(elem)
        for child in (getattr(elem, "_elements", None) or []):
            if is_anchored_child(child):
                continue
            yield from frame_basis_points(child, _on_path=guard)
    finally:
        guard.leave(elem)


def has_frame_basis(elem) -> bool:
    """True when ``elem`` carries the points a frame can be DERIVED from.

    The question ``executor.validate_project_report`` asks of a container that
    has no body of its own: a physical container is legal without a solid of
    its own as long as it has ``.add()``ed children to derive its frame from —
    they are in world coordinates by definition, which is exactly what
    a basis needs. With an empty basis there is nothing to derive, and
    ``.anchor()`` into it would have no frame to place against.

    Two deliberate differences from ``frame_basis_aabb(elem) is not None``:

    * an UNRESOLVED anchored ``elem`` is not skipped. Whether its own points
      are world coordinates yet is what makes them unusable as a *derivation*,
      and it is why :func:`frame_basis_points` declines; it says nothing about
      whether they EXIST. This question is asked at validation, before the
      bake, and the container's frame is re-derived after it.
    * a single point is enough. A degenerate basis is a frame with zero
      thickness, not a missing one, and the checks that care about extents own
      that separately.
    """
    from lite_step.models.elements import Door, Window

    if isinstance(elem, (Window, Door)):
        return False
    for _ in _frame_basis_walk(elem):
        return True
    return False


def frame_basis_aabb(elem) -> Optional[AABB]:
    """Axis-aligned bounds of :func:`frame_basis_points`.

    What :func:`frame_for` and :func:`_scope_aabb` read. Everything else that
    asks "how much space does this take up" — view extents, ``world_aabb()``,
    the layer datum, the displacement trigger — keeps reading
    :func:`element_aabb` and keeps seeing anchored children, because clipping
    an anchored cornice out of an orthographic view is just as wrong as
    letting it steer the wall's thickness.
    """
    return _aabb_of(frame_basis_points(elem))


def _union_aabb(boxes) -> Optional[AABB]:
    los: List[Vec3] = []
    his: List[Vec3] = []
    for box in boxes:
        if box is None:
            continue
        los.append(box[0])
        his.append(box[1])
    if not los:
        return None
    return (
        (min(p[0] for p in los), min(p[1] for p in los), min(p[2] for p in los)),
        (max(p[0] for p in his), max(p[1] for p in his), max(p[2] for p in his)),
    )


def _aabb_center(box: AABB) -> Vec3:
    (x0, y0, z0), (x1, y1, z1) = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)


def _scope_aabb(elements) -> Optional[AABB]:
    """Union AABB over top-level ``elements``, skipping placed ones.

    An element carrying ``placement=`` authors its geometry local to that
    placement, so its raw coordinates are not where it stands; including
    them would drag the scope center to a phantom location. Composing the
    resolved matrices here is not an option — anchor resolution needs the
    frames this pass produces, so that would be circular.

    The union is over :func:`frame_basis_aabb`, not :func:`element_aabb`: a
    scope center exists to answer "which way is OUT", and a detail hung on a
    wall by ``.anchor()`` was positioned by that answer. Feeding it back in
    is the same loop :func:`frame_basis_points` breaks one level down.
    """
    return _union_aabb(
        frame_basis_aabb(e) for e in elements if getattr(e, "placement", None) is None
    )


# ---------------------------------------------------------------------------
# Newell normal — the authored extrusion direction of a contour
# ---------------------------------------------------------------------------


def contour_normal(points) -> Optional[Vec3]:
    """Unit Newell normal of a 3D contour, or ``None`` when degenerate.

    The winding order IS the author's extrusion direction, which is why
    this is preferred over an AABB whenever it exists.
    """
    pts = [(float(p.x), float(p.y), float(p.z)) for p in points]
    if len(pts) < 3:
        return None
    nx = ny = nz = 0.0
    for i, (x0, y0, z0) in enumerate(pts):
        x1, y1, z1 = pts[(i + 1) % len(pts)]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-12:
        return None
    return (nx / length, ny / length, nz / length)


def opening_axes(elem, snap=None) -> "Optional[Tuple[Vec3, Vec3]]":
    """``(along, thickness_dir)`` for a voidable body — THE run-axis derivation.

    ``thickness_dir`` is the direction to walk HALF the thickness to reach
    the body's midline, which is where an opening is centered. It is NOT the
    right-handed ``up x along``: for a box the generator walks the POSITIVE
    perpendicular axis (``+Y`` for an X-running wall, ``+X`` for a Y-running
    one), and for a contour it walks the authored plane NORMAL (the contour
    is one wall FACE, so the midline is half a thickness along its normal).
    Returning the axis each consumer actually uses is what keeps the sign
    conventions from silently inverting.

    An opening (``Window``/``Door`` on a wall, or a standalone ``.opening()``)
    is placed by walking ``offset`` along the body's run axis and ``sill`` up,
    voiding through its thickness. That run axis was derived in three places
    with subtly different code — the box-mode AABB rule in the generator and the
    contour-mode Newell rule beside it. This is the one implementation.

    Wall-LIKE semantics apply regardless of the element's own frame rule: a
    standalone Box being voided still runs along its longer horizontal span.
    That is why this does not simply read ``_frame`` (whose rule follows
    containment) — though it agrees with the stamp for every wall body.

    * **Box** — along = the longer horizontal span, ``+X`` on ties. ``snap``
      (the caller's precision snapper, or ``None``) is applied to the spans
      BEFORE the comparison, so a hair of float noise cannot flip the axis.
    * **Extrude** — the contour must be a near-VERTICAL plane (a wall face);
      along is the in-plane horizontal ``up x normal``, canonicalized toward
      ``+X`` (ties ``+Y``) so offsets are winding-independent. A non-vertical
      or degenerate contour yields ``None`` — there is no horizontal run axis
      to walk, which the validator reports as a compile error.
    * **Container** (``Column``/``Beam``/``Element``) — the same
      longer-horizontal-span rule, read off :func:`frame_basis_aabb` instead of
      off a body the container does not have. ``frame_basis_aabb`` and not
      ``element_aabb`` for the reason it exists: an ``.anchor()``ed child is
      positioned BY this frame, so letting it back in closes a loop (anchoring
      a 180 mm leaf into a 180 mm host took the host's thickness to 360).

    Returns ``None`` when the element bears no usable run axis.
    """
    start = getattr(elem, "start", None)
    end = getattr(elem, "end", None)
    if start is not None and end is not None:
        x_span = abs(float(end.x) - float(start.x))
        y_span = abs(float(end.y) - float(start.y))
        return _axes_from_spans(x_span, y_span, snap)

    contour = getattr(elem, "contour", None)
    if contour:
        normal = contour_normal(contour)
        if normal is None:
            return None
        axes = _axes_from_extrude("wall", normal)
        if axes is None:
            return None            # not a vertical plane: no horizontal run
        along, _across, _up = axes
        return along, normal        # thickness runs along the plane normal

    box = opening_container_aabb(elem)
    if box is not None:
        (x0, y0, _z0), (x1, y1, _z1) = box
        return _axes_from_spans(abs(x1 - x0), abs(y1 - y0), snap)
    return None


def _axes_from_spans(x_span: float, y_span: float, snap
                     ) -> "Tuple[Vec3, Vec3]":
    """The box-mode run-axis rule: along = the longer horizontal span, ``+X``
    on ties. ``snap`` is applied BEFORE the comparison so a hair of float noise
    cannot flip the axis on a square-ish wall."""
    if snap is not None:
        x_span, y_span = snap(x_span), snap(y_span)
    if x_span >= y_span:
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    return (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)


def opening_container_aabb(elem) -> Optional[AABB]:
    """The bounds a representation-less CONTAINER's opening frame is built on,
    or ``None`` when ``elem`` is not one of those containers.

    Gated on :func:`~lite_step.models.taxonomy.hosts_openings` AND
    :func:`~lite_step.models.taxonomy.is_container`, not on "has no
    ``start``/``contour``", so widening this to a new host type is one edit to
    ``OPENING_HOSTS`` and the fallback cannot be reached by accident from a
    leaf whose own geometry failed to parse — that case must keep answering
    ``None`` and reaching the validator's refusal.
    """
    from lite_step.models import taxonomy as tx

    if not (tx.is_container(elem) and tx.hosts_openings(elem)):
        return None
    return frame_basis_aabb(elem)


def opening_spec(elem):
    """The ``ChildAnchor`` that places a ``Window``/``Door``, as SCALARS.

    An opening is the one anchored child that is NOT baked into geometry:
    ``along`` (the pre-v20 ``offset=``) and ``up`` (the pre-v20 ``sill=``)
    are consumed as numbers by the opening machinery, which walks the HOST's
    run axis and cuts an axis-aligned void. Units follow the project — meters
    by the time either backend runs, mm during validation.

    Loud on a missing spec: ``validate_project_report`` rejects an unplaced
    opening, so reaching a backend without one means validation was skipped,
    and the alternative to raising is a window silently compiled into the
    host's start corner at floor level.
    """
    spec = getattr(elem, "_anchor_spec", None)
    if spec is None:
        raise ChildAnchorError(
            f"{type(elem).__name__} "
            f"{getattr(elem, 'ifc_name', None) or getattr(elem, 'name', None)!r}"
            f": no position — an opening is placed by its host "
            f"(wall.anchor(x, along=…, up=…), or body.opening(x, along=…, "
            f"up=…) on a standalone solid). validate_project_report should "
            f"have rejected this."
        )
    return spec


def resolve_along(spec, child, *, cos_a: float = 1.0, sin_a: float = 0.0) -> float:
    """The effective ``along`` for an anchor spec — THE one place
    ``along_center`` is turned into a near-edge offset.

    ``along`` positions the child's near edge; ``along_center`` positions its
    CENTRE. The conversion needs where the child's span sits along the run
    axis, and that comes from two different places depending on what the
    child is:

    * a ``Window``/``Door`` — its ``width``, explicit or inferred from its
      own children (:func:`infer_opening_size`). The opening's width is the
      hole, which is what a centred window means. An opening is never baked,
      and ``opening_placement`` lays its width along the host's run from the
      host's OWN direction, so ``rotations=`` does not reach it — this branch
      takes no angle;
    * anything else — its own points, in the child's authoring (parent-local)
      coordinates, projected onto the local axis that BECOMES the run axis.

    That second axis is the ONLY thing this function may project onto, and it
    is local ``+x`` turned by this anchor's own ``rotations=`` — never the
    host's ``frame.along``. ``frame.along`` is a WORLD direction; the points
    are parent-local, where the run is ``+x`` however the host is oriented.
    Projecting one onto the other agreed with the world on an x-run host by
    coincidence and left-aligned every child on a y-run one.

    The correction is the MIDPOINT of that span, not half its length. The two
    agree only while the child's near edge sits on its own local origin —
    which a rotated child, or one authored across its origin, never does.

    Units follow the spec: mm before normalization, meters after. Returns
    ``spec.along`` untouched when ``along_center`` is not used, so the common
    path costs nothing.
    """
    center = getattr(spec, "along_center", None)
    if center is None:
        return spec.along

    from lite_step.models.elements import Door, Window

    if isinstance(child, (Window, Door)):
        width, _height = infer_opening_size(child)
        if width is None:
            # No size to centre against; the validator reports this
            # separately, so degrade to the near-edge reading rather than
            # crash mid-emission.
            return center
        # ``along_center`` centres the HOLE. When the hole was inferred, its
        # local origin has slid onto the joinery (:func:`opening_local_origin`),
        # so the anchor has to step back by that same offset or the centring
        # would be off by it.
        d_along, _d_up = opening_origin_offset(child)
        return center - width / 2.0 - d_along

    # ``_bake_point`` spells this same rotation: a baked point's along
    # coordinate is ``along_eff + (x*cos_a - y*sin_a)``. This projection IS
    # that expression with ``along_eff`` dropped, so the two cannot drift.
    projections = [
        float(p.x) * cos_a - float(p.y) * sin_a
        for p in iter_geometry_points(child, skip_anchored=False)
    ]
    if not projections:
        return center
    return center - (max(projections) + min(projections)) / 2.0


def opening_out_sign(host, dir_x: float, dir_y: float) -> int:
    """+1 when ``up x (dir_x, dir_y)`` points OUT of the building.

    The opening frame is facing-aware, so it needs the host's stamped
    ``facing``. The stamp's ``across`` follows the host's frame RULE
    (containment-derived); the run axis from :func:`opening_axes` has its own
    perpendicular ``up x along``. For every wall body the two agree — but a
    standalone ``.opening()`` on a Box with no Wall ancestor gets the
    world-axes fallback rule, whose ``across`` can be perpendicular to the run
    axis it actually opens along. So compare the two and carry the sign across
    rather than assuming they match.

    No stamp at all (a hand-built project that skipped the pass) degrades to
    ``+1`` — the raw geometric ``+across`` side, which is what every
    pre-frame model already meant.

    Perpendicular — :func:`facing_from`'s ``None`` — degrades to the host's
    raw ``facing``. That is this caller's own choice and not a shared default:
    a standalone ``.opening()`` whose host got the world-axes fallback rule is
    the only way to reach it, and the stamp is still the best statement about
    the building available.
    """
    frame = getattr(host, "_frame", None)
    if frame is None:
        return 1
    run_across = (-dir_y, dir_x, 0.0)           # up x along, in the XY plane
    sign = facing_from(frame.across, frame.facing, run_across)
    return frame.facing if sign is None else sign


def opening_right_axis(host, along: Vec3) -> Vec3:
    """The opening frame's **+x**: right, as seen from OUTSIDE the host.

    :func:`opening_axes` returns a run axis canonicalized toward ``+X``, which
    has no stable relationship to the building — local ``+x`` pointed the same
    world way on a north wall as on a south one, and the joinery inside every
    opening had to carry a per-facade sign to compensate.

    The frame this feeds is the wall's own: stand outside, look at the facade,
    and ``+x`` is your right, ``+y`` runs INTO the wall, ``+z`` is up. Right is
    ``inward x up``, and with ``inward = -out_sign * (up x along)`` that
    reduces to ``-out_sign * along`` — the canonical run, reversed on exactly
    the walls whose outward normal sits on the ``+(up x along)`` side (north
    and west of a conventional ring; south and east keep their direction).

    That reversal is :func:`outer_face_axis`, which this delegates to and
    which :meth:`ElementFrame.local_to_world` reads for the generic
    ``.anchor()`` bake — ONE derivation, so the two frames cannot drift.
    The basis is right-handed by construction (``right x inward = up``), so
    every placement built from it is a pure rotation about Z — never a
    mirror, which ``IfcAxis2Placement3D`` could not express anyway.

    This host's run axis is horizontal, so ``+z`` is pinned to a literal
    ``0.0`` rather than carried through the reversal (which would spell an
    unsigned zero ``-0.0``).
    """
    sign = opening_out_sign(host, along[0], along[1])
    right = outer_face_axis(sign, (along[0], along[1], 0.0))
    return (right[0], right[1], 0.0)


def opening_inward_axis(right: Vec3) -> Vec3:
    """``+y`` of the opening frame — INTO the element, from the outer face."""
    return (-right[1], right[0], 0.0)


def opening_box_frame(elem, min_pt, max_pt, snap=None):
    """``(origin, thickness, right)`` for a BOX body — THE single derivation.

    ``min_pt``/``max_pt`` are the body's axis-aligned bounds already converted
    to IFC space (meters); the emitter hands in the same pair, which is what
    keeps them from drifting apart the way the two inline copies of this rule
    did before.

    ``origin`` is opening-local ``(0, 0, 0)``: the body's lower corner in the
    ``right`` direction, **on the OUTER face**, at the body's minimum z. It was
    the thickness MIDLINE until v21 — a datum that is not a surface anyone can
    see or measure, and that made ``inset=`` and an authored child depth mean
    two different things.

    ``thickness`` is the snapped span across the run axis: the full depth the
    void still cuts, centred by stepping ``thickness / 2`` along ``inward``.

    Returns ``None`` when the element bears no usable run axis.
    """
    axes = opening_axes(elem, snap=snap)
    if axes is None:
        return None
    (ax, ay, _), _thickness_dir = axes
    right = opening_right_axis(elem, (ax, ay, 0.0))
    inward = opening_inward_axis(right)

    run_i = 0 if ax else 1                       # world index of the run axis
    thk_i = 1 - run_i                            # ... and of the thickness
    span = (max_pt[0] - min_pt[0], max_pt[1] - min_pt[1])
    if snap is not None:
        span = (snap(span[0]), snap(span[1]))
    thickness = span[thk_i]

    origin = [0.0, 0.0, min_pt[2]]
    # Run: start at whichever end ``right`` points AWAY from.
    origin[run_i] = min_pt[run_i] if right[run_i] > 0 else max_pt[run_i]
    # Thickness: the outer face is the one ``inward`` points away from.
    origin[thk_i] = min_pt[thk_i] if inward[thk_i] > 0 else max_pt[thk_i]
    return (origin[0], origin[1], origin[2]), thickness, right


def opening_contour_frame(contour_3d, along: Vec3, normal: Vec3,
                          thickness: float, right: Vec3):
    """``origin`` for a CONTOUR body — the contour twin of
    :func:`opening_box_frame`.

    The contour is a planar VERTICAL polygon extruded ``thickness`` along its
    Newell ``normal``, so the contour plane is ONE face of the wall and the
    body spans ``[plane_d, plane_d + thickness]`` along the normal. Which of
    those two planes is the OUTER face is decided by ``right`` (whose
    ``inward`` is ``up x right``), not by the winding — an author who reverses
    a contour must not thereby move the frame.
    """
    inward = opening_inward_axis(right)
    ax, ay, _ = along
    nx, ny, _nz = normal

    run_lo = min(p[0] * ax + p[1] * ay for p in contour_3d)
    run_hi = max(p[0] * ax + p[1] * ay for p in contour_3d)
    run_c = run_lo if (right[0] * ax + right[1] * ay) > 0 else run_hi

    plane_d = contour_3d[0][0] * nx + contour_3d[0][1] * ny
    # ``inward . normal > 0`` means the contour plane IS the outer face.
    thk_c = plane_d if (inward[0] * nx + inward[1] * ny) > 0 else plane_d + thickness

    z_min = min(p[2] for p in contour_3d)
    return (run_c * ax + thk_c * nx, run_c * ay + thk_c * ny, z_min)


@dataclass(frozen=True)
class OpeningFrame:
    """A voidable BODY's opening frame — where an opening's numbers start.

    ``wall_start`` is the frame origin: the body's lower corner in the
    ``right`` direction, **on the OUTER face**, at the body's minimum z.
    ``dir_x``/``dir_y`` are ``right``'s XY components (right, as seen from
    outside); ``cos_a``/``sin_a`` are the SAME direction re-derived through
    ``atan2`` — kept as separate fields rather than aliased because they are
    not bit-identical (``sin(atan2(0, -1))`` is 1.2e-16, not 0) and each
    consumer must keep reading the one it always read.

    Units are the caller's: meters in either IFC backend (which runs on the
    normalized project), authoring millimetres in the query layer. Nothing
    here divides or scales, so the frame is unit-agnostic by construction —
    the only unit-aware input is ``snap``, which each caller supplies.
    """

    wall_start: Vec3
    thickness: float
    dir_x: float
    dir_y: float
    cos_a: float
    sin_a: float

    @property
    def inward(self) -> tuple:
        """Local ``+y`` in world XY — INTO the body from the outer face."""
        return (-self.dir_y, self.dir_x)


@dataclass(frozen=True)
class OpeningPlacement:
    """Where ONE opening's three coordinate origins land, in world space.

    * ``void`` — the ``IfcOpeningElement``'s placement origin. Centred on the
      hole in plan, so it steps half a width along the run and half a
      thickness inward.
    * ``fill`` — the ``IfcWindow``/``IfcDoor``'s placement origin. The same
      point, moved back by ``inset=`` only.
    * ``corner`` — the origin the opening's CHILDREN are authored from: the
      opening's lower-left corner on the outer face, ``+x`` right as seen
      from outside, ``+y`` into the body, ``+z`` up.

    ``cos_a``/``sin_a`` rotate an opening-local ``(x, y)`` into world XY, so
    ``corner`` + that rotation is the full rigid transform from opening-local
    to world — which is what makes a query on a window's own frame box
    answerable with the SAME numbers the generator emits.
    """

    void: Vec3
    fill: Vec3
    corner: Vec3
    width: float
    height: float
    thickness: float
    cos_a: float
    sin_a: float


def opening_body_thickness(body, divisor: float = 1.0):
    """A CONTOUR body's through-thickness, or ``None``.

    Authored ``thickness=`` wins; otherwise a registry ``Material``'s
    ``thickness_mm``. Material dimensions are always mm (a frozen Material is
    never normalized), so ``divisor`` converts them into the CALLER's units:
    ``1000.0`` for a backend running on the normalized (meters) project,
    ``1.0`` for the authoring-millimetre query layer. Division by ``1.0`` is
    exact, so the query layer pays no float cost for the shared path.
    """
    thickness = getattr(body, "thickness", None)
    if thickness is not None:
        return thickness
    from lite_step.materials import element_registry_material

    mat = element_registry_material(body)
    if mat is not None and mat.thickness_mm is not None:
        return mat.thickness_mm / divisor
    return None


def host_opening_frame(host, snap=None, divisor: float = 1.0
                       ) -> Optional[OpeningFrame]:
    """The opening frame of a HOST — the solid, its container, or a body-less container.

    The ONE answer to "where do this host's openings sit", for every consumer
    that starts from whatever an opening's ``.parent`` points at (the query
    layer's ``--q``/``--check``, the space-boundary pass):

    * the host IS a ``Box``/``Extrude`` (``body.opening(...)``) -> its own frame;
    * the host is a container with a ``Box``/``Extrude`` child (a bodied Wall)
      -> that body's frame;
    * the host is a container with NO such child — a body-less Wall that only
      aggregates leaf walls (#731/#742) — -> the frame of its aggregated
      bounds, :func:`opening_container_aabb`, exactly what the emitter reads.

    Reading ``_body_of`` alone answered ``None`` for the third case, so the
    query surface claimed "compiling would reject the same model" and the
    space-boundary pass dropped the window's boundary, while the compiler
    emitted the window correctly. A container that HAS a body whose own frame
    fails does not fall back to its bounds: that failure is the validator's
    refusal, and falling through would hide it.
    """
    from lite_step.models import taxonomy as tx

    if tx.is_prism(host):
        return opening_host_frame(host, snap=snap, divisor=divisor)
    for child in (getattr(host, "_elements", None) or []):
        if tx.is_prism(child):
            return opening_host_frame(child, snap=snap, divisor=divisor)
    return opening_host_frame(host, snap=snap, divisor=divisor)


def opening_host_frame(body, snap=None, divisor: float = 1.0
                       ) -> Optional[OpeningFrame]:
    """THE opening frame of a voidable body — box, contour or container.

    Box mode (``Box`` with ``start``/``end``) reads the axis-aligned bounds;
    contour mode (``Extrude`` with a vertical planar ``contour``) reads the
    plane; CONTAINER mode (``Column``/``Beam``/``Element``) reads
    :func:`frame_basis_aabb` and is otherwise the box rule verbatim — the
    container has no body, so its ``.add()``ed leaves ARE the bounds, and they
    are world coordinates by definition.

    ``None`` means the body bears no usable frame — a degenerate or
    non-vertical contour, a missing thickness source, an empty container, or
    none of the three modes — all of which ``validate_project_report`` rejects
    at compile, so a ``None`` here is the defensive path for a hand-built
    project.

    The emitter AND the authoring-query composer read this one
    function. Deriving the box bounds inline in each caller puts the contour
    mode out of the query layer's reach, which is how a query inside a Window
    answers in opening-local coordinates while the compiled IFC is metres away.
    """
    from lite_step.models.elements import Box, Extrude

    container_box = opening_container_aabb(body)
    if isinstance(body, Box) and body.start is not None and body.end is not None:
        s, e = body.start, body.end
        min_pt = (min(s.x, e.x), min(s.y, e.y), min(s.z, e.z))
        max_pt = (max(s.x, e.x), max(s.y, e.y), max(s.z, e.z))
        derived = opening_box_frame(body, min_pt, max_pt, snap=snap)
    elif container_box is not None:
        # A representation-less container: the bounds of everything it
        # aggregates, run axis and outer face derived from them by the SAME
        # ``opening_box_frame`` a Box body uses. One rule, two sources of
        # bounds — not a second frame derivation to keep in step.
        min_pt, max_pt = container_box
        derived = opening_box_frame(body, min_pt, max_pt, snap=snap)
    elif isinstance(body, Extrude) and body.contour is not None:
        contour_3d = [(pt.x, pt.y, pt.z) for pt in body.contour]
        if len(contour_3d) < 3:
            return None
        thickness = opening_body_thickness(body, divisor)
        if thickness is None:
            return None
        axes = opening_axes(body)
        if axes is None:
            return None
        along, normal = axes
        right = opening_right_axis(body, along)
        derived = (opening_contour_frame(contour_3d, along, normal,
                                         thickness, right),
                   thickness, right)
    else:
        return None

    if derived is None:
        return None
    wall_start, thickness, right = derived
    angle = math.atan2(right[1], right[0])
    return OpeningFrame(
        wall_start=wall_start, thickness=thickness,
        dir_x=right[0], dir_y=right[1],
        cos_a=math.cos(angle), sin_a=math.sin(angle),
    )


def opening_placement(opening, frame: OpeningFrame) -> OpeningPlacement:
    """Where ONE opening's void, fill and child frame land — THE derivation.

    Three callers read this: the emitter (which emits the void and fill
    placements from it) and ``BimElement._world_matrix_for`` (which composes
    ``corner`` + the rotation so a query inside an opening resolves through
    the same arithmetic the generator emits). They were three separate
    copies; the query layer never had one at all, so it answered from
    opening-LOCAL coordinates with no error — a silently wrong number for
    anything nested inside an anchored ``Window``/``Door``.

    The inferred-origin offset cancels out of ``corner`` exactly: the hole
    slides onto the joinery while the joinery stays where it was drawn. It is
    still threaded through rather than simplified away, because the void and
    the fill DO move by it and the three answers must come out of one
    expression chain.
    """
    spec = opening_spec(opening)
    # along_center= centres the opening on its own width instead of placing
    # its near edge; resolved in one shared helper.
    offset = resolve_along(spec, opening)
    sill = spec.up
    origin_along, origin_up = opening_origin_offset(opening)
    # An explicit size returns straight through; an inferred one is read off
    # the joinery. The backends run post-normalization, where the size is
    # always written, so this is the identity for them and the live
    # inference for the query layer.
    width, height = infer_opening_size(opening)

    dir_x, dir_y = frame.dir_x, frame.dir_y
    cos_a, sin_a = frame.cos_a, frame.sin_a
    thickness = frame.thickness
    in_x, in_y = -dir_y, dir_x                  # local +y: into the body

    # Opening position: wall_start + offset along the frame's +x, then the
    # inferred-origin step onto the joinery.
    opening_x = frame.wall_start[0] + dir_x * (offset + origin_along)
    opening_y = frame.wall_start[1] + dir_y * (offset + origin_along)
    opening_z = frame.wall_start[2] + sill + origin_up

    # The void's rect profile is centred on its own placement, so it steps
    # half a width along the run and half a thickness inward to still span
    # the FULL body depth.
    center_offset_x = dir_x * width / 2.0
    center_offset_y = dir_y * width / 2.0
    half_t = thickness / 2.0
    void = (opening_x + center_offset_x + in_x * half_t,
            opening_y + center_offset_y + in_y * half_t,
            opening_z)

    # ``inset=`` moves the FILL only — the void always cuts full depth.
    fill_x = opening_x + center_offset_x
    fill_y = opening_y + center_offset_y
    if spec.inset:
        fill_x += -sin_a * spec.inset
        fill_y += cos_a * spec.inset

    # The children are authored relative to the ANCHOR, not to the hole, so
    # their corner steps the inferred-origin offset back out.
    corner = (fill_x - cos_a * (width / 2.0 + origin_along),
              fill_y - sin_a * (width / 2.0 + origin_along),
              opening_z - origin_up)

    return OpeningPlacement(
        void=void, fill=(fill_x, fill_y, opening_z), corner=corner,
        width=width, height=height, thickness=thickness,
        cos_a=cos_a, sin_a=sin_a,
    )


@dataclass(frozen=True)
class VoidPrism:
    """The world-space volume ONE opening's void removes.

    Four world corners of the void's BOTTOM face, counter-clockwise seen from
    ``+Z``, plus the height it rises. That is the shape a boolean operand
    needs, and it is the SAME box the generator emits: a
    ``width x thickness`` rectangle centred on :attr:`OpeningPlacement.void`,
    rotated into the host frame and extruded ``height`` upward (see
    ``generator._create_opening_from_assembly``). Read off
    :func:`opening_placement` rather than re-derived, so the hole in the wall
    and the hole in the wall's reinforcement cannot be two different holes.

    Corner order is load-bearing. A horizontal contour's extrusion direction is
    resolved by ``ifc.geometry.apply_orientation_rules``, which forces any
    non-vertical normal to point UP — so a bottom-face contour extrudes upward
    whatever the winding, and the sign trap that makes vertical contours
    unpredictable (see ``extent.extrude_offset``) cannot bite here. CCW is
    stated anyway so the Newell normal agrees with the rule instead of relying
    on it.
    """

    corners: Tuple[Vec3, Vec3, Vec3, Vec3]
    height: float

    def aabb(self) -> Tuple[Vec3, Vec3]:
        """World ``(min, max)`` — the overlap trigger's currency."""
        xs = [c[0] for c in self.corners]
        ys = [c[1] for c in self.corners]
        z0 = self.corners[0][2]
        return ((min(xs), min(ys), z0),
                (max(xs), max(ys), z0 + self.height))

    def is_axis_aligned(self, eps: float) -> bool:
        """True when the prism IS its own AABB — an orthogonal host.

        Measured off the geometry rather than off ``cos_a``/``sin_a``, so it
        answers the question a boolean operand actually asks: can this volume
        be written as ``Box(start, end)`` without growing?
        """
        (x0, y0, _z0), (x1, y1, _z1) = self.aabb()
        box_area = abs(x1 - x0) * abs(y1 - y0)
        # The rectangle's OWN area, as the cross product of two adjacent
        # edges. It equals the bounding area only when the two coincide.
        e1 = (self.corners[1][0] - self.corners[0][0],
              self.corners[1][1] - self.corners[0][1])
        e2 = (self.corners[2][0] - self.corners[1][0],
              self.corners[2][1] - self.corners[1][1])
        own_area = abs(e1[0] * e2[1] - e1[1] * e2[0])
        return abs(box_area - own_area) <= eps * max(1.0, box_area)


def opening_void_prism(opening, frame: OpeningFrame) -> Optional[VoidPrism]:
    """The world box :func:`opening_placement` puts this opening's void in.

    ``None`` when the opening has no size to cut with — an inferred size with
    no joinery to infer from, which ``validate_project_report`` reports
    separately. Returning ``None`` rather than a zero-volume prism keeps a
    sizeless opening from carving a degenerate operand into a detail.
    """
    place = opening_placement(opening, frame)
    width, height, thickness = place.width, place.height, place.thickness
    if not width or not height or not thickness:
        return None

    cx, cy, z0 = place.void
    cos_a, sin_a = place.cos_a, place.sin_a
    # local +x (along the run) and local +y (into the body) at half extent.
    ux, uy = cos_a * width / 2.0, sin_a * width / 2.0
    vx, vy = -sin_a * thickness / 2.0, cos_a * thickness / 2.0
    # ``u x v`` is +z for any angle (cos^2 + sin^2 = 1), so this order is CCW
    # seen from above on every host, including a skew wall.
    corners = (
        (cx - ux - vx, cy - uy - vy, z0),
        (cx + ux - vx, cy + uy - vy, z0),
        (cx + ux + vx, cy + uy + vy, z0),
        (cx - ux + vx, cy - uy + vy, z0),
    )
    return VoidPrism(corners=corners, height=height)


def _joinery_extents(opening):
    """``(x0, x1, z0, z1)`` — the IN-PLANE bounds of an opening's joinery, in
    opening-local coordinates, or ``None`` when there is nothing to measure.

    THE single scan behind both :func:`infer_opening_size` and
    :func:`opening_local_origin`: the hole's size and the hole's position are
    two readings of one bounding box, and deriving them apart is how a void
    and its fill end up disagreeing.

    Depth (local Y) is deliberately absent: a frame runs through the wall and
    a sill projects past its face, neither of which says anything about how
    big the hole is.
    """
    xs, zs = [], []
    for child in (getattr(opening, "_elements", None) or []):
        for p in iter_geometry_points(child):
            xs.append(float(p.x))
            zs.append(float(p.z))
    if not xs:
        return None
    return min(xs), max(xs), min(zs), max(zs)


def infer_opening_size(opening):
    """``(width, height)`` for a Window/Door — explicit if stated, else inferred.

    An opening's ``width``/``height`` are the ROUGH OPENING: the hole in the
    wall. Its children are the joinery that sits in that hole, authored in
    OPENING-LOCAL coordinates (X along the wall, Y through the depth, Z up —
    see ``test_opening_local_zup_contract``). For the common case where the
    joinery fills the hole, restating the size is duplication the author can
    get wrong, so omitting it infers the hole from the children's IN-PLANE
    extents.

    The inferred size is the joinery's OWN SPAN — ``max - min``, not
    ``max - 0``. Measuring from local 0 made the hole grow with the distance
    between the opening's origin and the joinery an author actually drew, so
    joinery authored at local x 10000..11000 carved an ELEVEN METRE hole that
    started back at the anchor. The span is paired with
    :func:`opening_local_origin`, which slides the hole onto the joinery: the
    void and the fill are then two readings of one bounding box and cannot
    disagree. A frame authored from x=50 therefore gets a hole from 50, not a
    50 mm sliver of daylight beside it.

    An explicit value always wins — that is how an author says "the hole is
    SMALLER than the joinery", which is exactly the projecting-sill case that
    makes inference wrong. Returns ``(None, None)`` when there is nothing to
    infer from; the validator turns that into a compile error.
    """
    width = getattr(opening, "width", None)
    height = getattr(opening, "height", None)
    if width is not None and height is not None:
        return width, height

    extents = _joinery_extents(opening)
    if extents is None:
        return width, height
    x0, x1, z0, z1 = extents

    if width is None:
        width = int(round(x1 - x0))
    if height is None:
        height = int(round(z1 - z0))
    return width, height


def opening_local_origin(opening):
    """``(along, up)`` — where an INFERRED opening's local origin sits,
    relative to the anchor point its host placed it at.

    The twin of :func:`infer_opening_size`: the size is the joinery's span,
    and this is the corner that span starts from. Together they say "the hole
    IS the joinery's bounding box", so a Window whose children are drawn far
    from its own local origin carves a hole around THEM rather than a hole
    stretching back to the anchor — off-wall joinery carves off-wall, and
    there is no hole in the wall it never covered.

    Zero on any axis whose size is EXPLICIT: an explicit size is the author
    stating the hole outright, and the anchor states where it goes. Only
    inference — which reads the joinery — may move the origin onto it.

    Units are the children's, so this is mm before normalization and meters
    after. The value is stamped onto the normalized copy
    (:func:`stamp_opening_origin`), because normalization overwrites
    ``width``/``height`` with the resolved numbers and the generators can no
    longer tell an inferred size from an authored one.
    """
    extents = _joinery_extents(opening)
    if extents is None:
        return 0.0, 0.0
    x0, _x1, z0, _z1 = extents
    along = 0.0 if getattr(opening, "width", None) is not None else float(round(x0))
    up = 0.0 if getattr(opening, "height", None) is not None else float(round(z0))
    return along, up


def stamp_opening_origin(opening, along: float, up: float) -> None:
    """Record :func:`opening_local_origin`'s answer on the normalized copy."""
    opening._opening_origin = (float(along), float(up))


def opening_origin_offset(opening) -> Tuple[float, float]:
    """The ``(along, up)`` origin offset: the stamp, else derived live.

    Normalization stamps it (:func:`stamp_opening_origin`) because that pass
    overwrites ``width``/``height`` with the resolved numbers and the
    generators cannot tell an inferred size from an authored one — so
    on either backend the stamp is always there and always wins.

    The AUTHORING project is never stamped, and that is the query layer's
    view of the world. Returning ``(0, 0)`` there is wrong rather than
    conservative: an inferred opening's hole HAS slid onto its joinery, so a
    query composing the opening's frame would answer a whole offset away from
    the placement the compiler emits. Deriving it live from the children is
    the same reading the stamp records, just taken at a stage where the size
    is still ``None`` and therefore still legible.
    """
    stamped = getattr(opening, "_opening_origin", None)
    if not stamped:
        return opening_local_origin(opening)
    return float(stamped[0]), float(stamped[1])


def _body_of(elem):
    """The element's geometry-bearing body child, if it has exactly one kind.

    Semantic containers (Wall/Slab/Roof/...) carry no geometry themselves —
    the body is a Box or Extrude child. Returns the FIRST such child; a
    multi-body element (layered wall leaves) has a consistent orientation
    across bodies, so the first is representative.
    """
    for child in (getattr(elem, "_elements", None) or []):
        if getattr(child, "contour", None) or getattr(child, "start", None) is not None:
            return child
    return None


# ---------------------------------------------------------------------------
# Frame derivation
# ---------------------------------------------------------------------------


def rule_for(elem, ancestors: Tuple = ()) -> str:
    """Frame rule for ``elem``: own type, else nearest Wall/Beam/Slab
    ancestor container (a named wall-body Box gets the wall rule), else the
    world-axes fallback. ``ancestors`` is ordered outermost-first."""
    from lite_step.models import Beam, Slab, Wall
    from lite_step.models.elements import Element

    for e in (elem,) + tuple(reversed(ancestors)):
        if isinstance(e, Wall):
            return "wall"
        if isinstance(e, Beam):
            return "beam"
        if isinstance(e, Slab):
            return "slab"
        if isinstance(e, Element):
            mapped = ELEMENT_CLASS_RULES.get(e.ifc_class)
            if mapped:
                return mapped
    return "fallback"


def _axes_from_extrude(rule: str, normal: Vec3) -> Optional[Tuple[Vec3, Vec3, Vec3]]:
    """(along, across, up) from an authored extrusion normal, or None.

    Wall rule: the normal is the through-thickness direction (``across``);
    the run axis is the in-plane horizontal ``up x normal``, canonicalized
    toward +X (ties toward +Y) so offsets are winding-independent — the
    convention ``_derive_contour_wall_frame`` already pinned. Slab rule: a
    near-vertical normal is ``up`` and the world axes stand; anything else
    is not expressible in v1 and falls back to the AABB rule.
    """
    nx, ny, nz = normal
    if rule == "wall":
        if abs(nz) > VERTICAL_PLANE_NZ_MAX:
            return None                      # not a vertical plane: no run axis
        dx, dy = -ny, nx                     # up x normal
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return None
        dx /= length
        dy /= length
        if dx < -1e-9 or (abs(dx) <= 1e-9 and dy < 0):
            dx, dy = -dx, -dy                # canonicalize toward +X, ties +Y
        along = (dx, dy, 0.0)
        up = (0.0, 0.0, 1.0)
        across = (up[1] * along[2] - up[2] * along[1],
                  up[2] * along[0] - up[0] * along[2],
                  up[0] * along[1] - up[1] * along[0])
        return along, across, up
    if abs(nz) > 0.9:                        # slab-ish: extruded vertically
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    return None


def _axes_from_aabb(rule: str, box: AABB) -> Tuple[Vec3, Vec3, Vec3]:
    """(along, across, up) by the legacy AABB rule — the box-mode path.

    Wall/Beam: along = the longer horizontal span, +X on ties. Slab and the
    fallback: literal world axes. Reproduces ``_host_frame`` exactly.
    """
    (x0, y0, _z0), (x1, y1, _z1) = box
    if rule in ("wall", "beam"):
        ai = 0 if (x1 - x0) >= (y1 - y0) else 1
    else:
        ai = 0
    along = (1.0, 0.0, 0.0) if ai == 0 else (0.0, 1.0, 0.0)
    across = (0.0, 1.0, 0.0) if ai == 0 else (-1.0, 0.0, 0.0)  # up x along
    return along, across, (0.0, 0.0, 1.0)


def _extents_and_origin(box: AABB, along: Vec3, across: Vec3, up: Vec3):
    """Project the AABB corners onto the frame axes -> (origin, extents).

    ``origin`` is the min corner in frame coordinates ("lower left back");
    extents are the spans along each axis. For axis-aligned frames this is
    exactly the AABB min corner and its spans.
    """
    (x0, y0, z0), (x1, y1, z1) = box
    corners = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]

    def _proj(axis: Vec3):
        vals = [c[0] * axis[0] + c[1] * axis[1] + c[2] * axis[2] for c in corners]
        return min(vals), max(vals)

    a_min, a_max = _proj(along)
    c_min, c_max = _proj(across)
    u_min, u_max = _proj(up)
    origin = (
        along[0] * a_min + across[0] * c_min + up[0] * u_min,
        along[1] * a_min + across[1] * c_min + up[1] * u_min,
        along[2] * a_min + across[2] * c_min + up[2] * u_min,
    )
    return origin, (a_max - a_min), (c_max - c_min), (u_max - u_min)


def _facing_for(across: Vec3, box: AABB, scope_center: Optional[Vec3],
                outward: Optional[Vec3],
                inherited: "Optional[ElementFrame]" = None) -> int:
    """+1 when the exterior lies on the +across side, else -1.

    Three statements, in order of authority.

    1.  An authored ``LayerSet.outward`` wins outright. It is the author
        saying which way the buildup faces, and it beats every derivation
        including an inherited one — a leaf that states its own outward is
        making a claim about ITSELF, not repeating its host's. The BODY of a
        layered container is governed by the same statement (:func:`_outward_of`
        — the body IS the buildup), so a layered wall and the solid its stack is
        cut from cannot name different faces.
    2.  An ``inherited`` reference frame — the frame that PLACED this element,
        which is its anchoring host or, one generation on, its parent inside
        that anchored subtree — restated on this element's own ``across``
        (:func:`facing_from`). ``facing`` is the ONE frame property
        an anchored child cannot derive for itself: it is a statement about
        what lies OUTSIDE the element, and for a child placed inside a host
        "outside" is the host's outside. The axes are deliberately NOT
        inherited — a leaf running ``+Y`` genuinely runs ``+Y`` — but
        re-deriving the exterior side from a placed AABB and a scope center is
        what makes the composition lossy: it answers "which side of me is
        farther from the middle of the building", and a leaf buried in the
        inner half of its host answers ``-1`` while the wall it is part of
        says ``+1``.
    3.  Otherwise the exterior is the across-face FARTHER from the nearest
        containment scope's center (ties -> the max side, matching the legacy
        origin rule's tie outcome). With no scope center at all (site-only /
        everything placed) this falls back verbatim to the legacy "farther
        from the world origin" rule.

    A PERPENDICULAR reference (:func:`facing_from` returning ``None``) falls
    through to 3 unchanged: a host running east-west says nothing about which
    side of a north-south return wall is out, and inheriting its number
    anyway would be worse than deriving one.
    """
    if outward is not None:
        dot = outward[0] * across[0] + outward[1] * across[1] + outward[2] * across[2]
        if abs(dot) > 1e-9:
            return 1 if dot > 0 else -1

    if inherited is not None:
        sign = facing_from(inherited.across, inherited.facing, across)
        if sign is not None:
            return sign

    lo, hi = box
    c_lo = lo[0] * across[0] + lo[1] * across[1] + lo[2] * across[2]
    c_hi = hi[0] * across[0] + hi[1] * across[1] + hi[2] * across[2]
    lo_c, hi_c = min(c_lo, c_hi), max(c_lo, c_hi)

    if scope_center is None:
        ref = 0.0                                  # legacy: the world origin
    else:
        ref = (scope_center[0] * across[0]
               + scope_center[1] * across[1]
               + scope_center[2] * across[2])
    return 1 if abs(hi_c - ref) >= abs(lo_c - ref) else -1


def _authored_outward(elem) -> Optional[Vec3]:
    """The ``LayerSet.outward`` normal authored ON this element, if any."""
    layers = getattr(elem, "layers", None)
    outward = getattr(layers, "outward", None) if layers is not None else None
    if outward is None:
        return None
    try:
        return (float(outward[0]), float(outward[1]), float(outward[2]))
    except (TypeError, IndexError, ValueError):
        return None


def _outward_of(elem, ancestors: Tuple = ()) -> Optional[Vec3]:
    """The authored ``LayerSet.outward`` that governs ``elem``, if any.

    Its own, first. Failing that, **the one authored on the semantic container
    whose BUILDUP it is** — read the way :func:`rule_for` reads its ancestors,
    and for the same reason: some frame facts are properties of the composition
    rather than of the object holding the geometry.

    A ``LayerSet.outward`` is a statement about the BUILDUP ("this stack faces
    that way"), and a layered element's body child IS that buildup — the very
    solid ``layers.box_layer_slices`` cuts the stack out of. So this is not an
    inheritance channel beside ``inherit_from``: nothing about the CONTAINER
    travels down, and the reading stops at the buildup relation. The container's
    own facing already derives from the same statement, projected on its own
    ``across``; the body simply stops ignoring a claim that was always about it.

    Without this the two consumers of ``facing`` read different answers on one
    layered wall — ``.anchor()`` bakes through the CONTAINER's frame while an
    opening reads ``opening_host_frame(_body_of(wall))`` — so a ``Box`` and a
    ``Window`` at the same ``along`` land at opposite ends of it
. The anchored half of that split was closed by the subtree carry; a TOP-LEVEL ``.add()``ed layered wall is placed by nothing,
    so no reference exists to carry and only the authored claim can reach it.

    Which child is the buildup is ``layers.layered_body_child``'s answer, not a
    second reading of it: that predicate is what the slicer itself uses to pick
    the solid it stacks into, and it excludes an ANCHORED child (a cornice hung
    on the wall is not the wall's buildup) and refuses a multi-solid element
    outright. A body deriving the stack's outward while the stack is cut from a
    different solid is precisely the drift this module's header describes.
    """
    own = _authored_outward(elem)
    if own is not None:
        return own
    if not ancestors:
        return None
    container = ancestors[-1]
    container_outward = _authored_outward(container)
    if container_outward is None:
        return None

    from lite_step.compiler.layers import layered_body_child

    if layered_body_child(container) is not elem:
        return None
    return container_outward


def frame_for(elem, ancestors: Tuple = (), scope_center: Optional[Vec3] = None,
              inherit_from: "Optional[ElementFrame]" = None
              ) -> Optional[ElementFrame]:
    """Derive ``elem``'s frame, or ``None`` when it bears no geometry.

    The box is the frame BASIS — ``elem`` plus every descendant that is not
    anchored INTO it — never the full extent. See
    :func:`frame_basis_points`.

    ``inherit_from`` is the frame that PLACED ``elem`` — its anchoring host if
    ``elem`` is an anchored child, else its parent when that parent was itself
    placed (see :func:`_walk`) — and it contributes to exactly one field:
    ``facing``. Everything else is re-derived from ``elem``'s own placed
    geometry, because everything else IS a fact about that geometry. See
    :func:`_facing_for`.
    """
    box = frame_basis_aabb(elem)
    if box is None:
        return None

    rule = rule_for(elem, ancestors)

    axes = None
    kind = rule
    body = _body_of(elem) if getattr(elem, "_elements", None) else None
    source = body if body is not None else elem
    contour = getattr(source, "contour", None)
    if contour:
        normal = contour_normal(contour)
        if normal is not None:
            axes = _axes_from_extrude(rule, normal)
            if axes is not None and rule == "wall":
                kind = "wall-contour"
    if axes is None:
        axes = _axes_from_aabb(rule, box)
        if rule == "wall":
            kind = "wall-box"

    along, across, up = axes
    origin, extent_along, thickness, height = _extents_and_origin(box, along, across, up)
    facing = _facing_for(across, box, scope_center,
                         _outward_of(elem, ancestors), inherit_from)
    return ElementFrame(
        origin=origin,
        along=along,
        across=across,
        up=up,
        facing=facing,
        rule=kind,
        extent_along=extent_along,
        thickness=thickness,
        height=height,
    )


# ---------------------------------------------------------------------------
# The stamping pass
# ---------------------------------------------------------------------------


def scope_centers(project) -> Dict[str, Optional[Vec3]]:
    """Containment-scope centers: building + per-storey (+ site).

    Space centers are computed during the walk (a Space is an element, so
    its own AABB is its center). Keys: ``"building"``, ``"site"``, and
    ``id(storey)`` for each storey.
    """
    centers: Dict[str, Optional[Vec3]] = {}
    per_storey = []
    for storey in project.storeys:
        box = _scope_aabb(storey.elements)
        centers[f"storey:{id(storey)}"] = _aabb_center(box) if box else None
        per_storey.append(box)
    building = _union_aabb(per_storey)
    centers["building"] = _aabb_center(building) if building else None
    site_box = _union_aabb(
        element_aabb(s) for s in (getattr(project, "sites", None) or [])
    )
    centers["site"] = _aabb_center(site_box) if site_box else None
    return centers


def _spaces_in(storey) -> List[Tuple[AABB, Vec3]]:
    """(aabb, center) for every ``Space`` on a storey, smallest volume first.

    A ``Space`` accepts only ``Box`` children — a Wall can never be a TREE
    child of one — so "the room this wall belongs to" is necessarily a
    SPATIAL question, not a containment-walk question. Smallest-first so a
    nested room wins over the open-plan volume enclosing it.
    """
    from lite_step.models.elements import Space

    out: List[Tuple[AABB, Vec3]] = []
    for elem in storey.elements:
        if isinstance(elem, Space) and getattr(elem, "placement", None) is None:
            box = element_aabb(elem)
            if box is not None:
                out.append((box, _aabb_center(box)))

    def _volume(item) -> float:
        (x0, y0, z0), (x1, y1, z1) = item[0]
        return (x1 - x0) * (y1 - y0) * (z1 - z0)

    out.sort(key=_volume)
    return out


def _enclosing_space_center(box: AABB, spaces) -> Optional[Vec3]:
    """Center of the smallest Space whose volume contains ``box``'s center.

    A wall BETWEEN two rooms is genuinely ambiguous — whichever room's
    volume its centerline falls in wins, and an authored
    ``LayerSet.outward`` overrides the whole question anyway.
    """
    if not spaces:
        return None
    c = _aabb_center(box)
    for (lo, hi), center in spaces:          # smallest-volume first
        if (lo[0] <= c[0] <= hi[0]
                and lo[1] <= c[1] <= hi[1]
                and lo[2] <= c[2] <= hi[2]):
            return center
    return None


def _walk(elem, ancestors: Tuple, scope_center: Optional[Vec3], spaces=(),
          _on_path=None, inherit_from: "Optional[ElementFrame]" = None) -> None:
    """Stamp ``elem`` and recurse.

    Facing reference = the nearest containment scope: the enclosing Space
    (spatially, see :func:`_enclosing_space_center`) if any, else the storey
    center passed in, else the building/site center. Plain geometry children
    inherit the scope their semantic container resolved, so a wall-body Box
    reads the same reference the Wall did.

    A child gets its OWN frame (bounded by its own geometry) but takes the
    RULE from its nearest semantic ancestor via ``ancestors`` — that is what
    makes a named wall-body Box usable as an Anchor host with wall
    semantics, instead of reconstructing it with a separate ancestor walk.

    An ANCHORED child takes one more thing from that ancestor: ``facing``,
    read off the host's already-stamped ``_frame`` (the walk is top-down, so
    the host was stamped on the way in). Its axes and extents stay its own.
    ``facing`` is a statement about what lies outside the element, and for
    something placed by a host's frame that is the host's outside; re-deriving
    it from the placed AABB and the scope center is what let a leaf buried in
    the inner half of a wall call its own inner face the exterior — and then
    build everything anchored INTO it on the wrong face.

    **The reference travels down the whole anchored SUBTREE, and stops dead at
    anything ``.add()``ed outside one** — each generation inheriting from its
    own parent's already-stamped frame, so ``facing_from`` re-projects at every
    link and a rotated container composes correctly. That boundary is the
    ``.add()``/``.anchor()`` distinction one more time: a subtree ``.add()``ed
    at the top level was authored in world coordinates and nothing placed it,
    so it derives its own facing from the scope centre, exactly as it always
    has. Stopping AT the anchored node instead was — the
    container inherited and its ``.add()``ed body did not, and the two readers
    read different ones of them (``.anchor()`` bakes through the CONTAINER's
    frame, an opening reads ``opening_host_frame(_body_of(wall))``), so a
    batten and a window at the same ``along=1000`` on the same leaf landed
    3.7 m apart at ``x[4.700, 5.000]`` and ``x[1.000, 1.600]``.

    A link with no frame of its own ends the carry rather than passing the
    reference it was handed straight through, and the case that matters is a
    ``Window``/``Door``: :func:`frame_basis_points` yields nothing for one, so
    it is stamped ``None``, and its children are authored in OPENING-LOCAL
    coordinates that ``_bake_walk`` deliberately never drags into world space
    (:func:`opening_placement` is what resolves them). A world-space facing
    restated on an opening-local ``across`` would be an answer about a
    different coordinate system. Measured: passing the reference through
    instead reaches 110 more corpus frames, every one of them inside a
    ``Window`` or ``Door`` subtree.

    On-path re-entry (a containment cycle) is PRUNED — a termination guard,
    not a correctness rule. ``validate_project_report`` refuses a cyclic
    model before generation; this walk only has to survive being asked.
    """
    guard = _on_path if _on_path is not None else WalkGuard()
    if not guard.enter(elem):
        return
    try:
        own_center = scope_center
        # The AABB here is computed for ONE question — which Space encloses this
        # element — and ``_enclosing_space_center`` answers ``None`` on its first
        # line when ``spaces`` is empty. So without this gate every element in a
        # Space-less scope pays for a full subtree walk to learn nothing.
        #
        # Measured over the corpus: 25,777 of 40,858 calls (63.1%) arrived with
        # an empty list. Provably equivalent — the guarded block's only effect is
        # through ``room``, which is ``None`` whenever ``spaces`` is falsy.
        if spaces:
            box = element_aabb(elem)
            if box is not None:
                room = _enclosing_space_center(box, spaces)
                if room is not None:
                    own_center = room

        host_frame = inherit_from
        if ancestors and is_anchored_child(elem):
            host_frame = getattr(ancestors[-1], "_frame", None)
        elem._frame = frame_for(elem, ancestors, own_center, host_frame)

        # THE BOUNDARY: an element hands its OWN stamped frame down as the next
        # generation's reference exactly when it was itself placed by one — so
        # the reference reaches every descendant of an anchored node, and
        # nothing that was `.add()`ed outside one.
        child_reference = elem._frame if host_frame is not None else None

        child_ancestors = ancestors + (elem,)
        for attr in _WALK_ATTRS:
            for child in (getattr(elem, attr, None) or []):
                _walk(child, child_ancestors, own_center, spaces, guard,
                      child_reference)
    finally:
        guard.leave(elem)


class ChildAnchorError(ValueError):
    """An anchored child could not be resolved into its parent's frame."""


def _rotation_z_only(rotations) -> "Optional[float]":
    """Total z-rotation in centidegrees, or ``None`` if any axis is x/y.

    Only rotations about the frame's ``up`` axis are bakeable: an x/y
    rotation tilts the solid out of its authoring plane, which a Box's
    ``start``/``end`` pair cannot represent at all and which would change a
    Sweep's section roll rather than move it rigidly.
    """
    total = 0.0
    for axis, angle in rotations or []:
        if axis != "z":
            return None
        total += float(angle)
    return total


def rotation_refusal(elem, total_cd: float) -> "Optional[str]":
    """Why ``elem`` cannot receive ``total_cd`` of anchor z-rotation, or
    ``None`` when it can.

    THE one statement of the per-leaf rotation rules. Two callers read it and
    they must never diverge: ``.anchor()`` checks it against the whole subtree
    at the AUTHORING line, and :func:`_bake_geometry` checks
    the leaf it is actually about to bake. Two copies of one rule agreeing
    *usually* is what produced — and here the divergence would
    be worse than wrong, it would be an early error that fires on cases the
    bake accepts, or stays silent on cases it refuses.

    Rules, unchanged in substance:

    * ``Box`` — quarter turns only. Its start/end pair IS an axis-aligned box;
      any other angle has no representation in those two corners.
    * ``Extrude`` — any angle. The contour is a point list and rotates.
    * a container (bears no points of its own) — any angle. It has nothing to
      move; each descendant meets the same rotation under its own rule, which
      is exactly why the call-site check walks the subtree rather than
      stopping at the child it was handed.
    * everything else that bears points (``Sweep``/``Revolve``/``Pipe``/
      ``Bar``/``Mesh``) — translation only. Their section orientation derives
      from world-frame references, so rotating here changes the SHAPE.
    """
    from lite_step.models.elements import Box, Extrude

    if isinstance(elem, Box):
        if total_cd % 9000 != 0:
            return (f"a Box is defined by opposite corners, so .anchor() can "
                    f"only rotate it in quarter turns (multiples of 9000 "
                    f"centidegrees); got {total_cd:g}. Use an Extrude for an "
                    f"arbitrary plan angle.")
        return None
    if isinstance(elem, Extrude) and elem.contour:
        return None
    if not _bears_points(elem):
        return None
    if total_cd:
        return (".anchor(rotations=) on this type is not bakeable — its "
                "section orientation derives from world-frame references, so "
                "rotating it here would change its shape rather than move it. "
                "Anchor it without rotations, or use placement=Transform via "
                ".add().")
    return None


def iter_rotated_descendants(child):
    """``child`` and every descendant an ``.anchor()`` rotation will reach.

    The SAME walk :func:`_bake_subtree` performs, and for the same two
    stopping rules — a ``Window``/``Door`` (its spec is scalars the opening
    machinery resolves separately, and its children are opening-local) and a
    child that is ITSELF anchored (it is baked in the next generation, inside
    its container's frame). Anything this yields is something the rotation
    will be applied to, so it is exactly the set the call-site check must
    validate.
    """
    from lite_step.models.elements import Door, Window

    yield child
    for sub in (getattr(child, "_elements", None) or []):
        if isinstance(sub, (Window, Door)):
            continue
        if getattr(sub, "_anchor_spec", None) is not None:
            continue
        yield from iter_rotated_descendants(sub)


def check_anchor_rotation(container, child, rotations) -> None:
    """Refuse an illegal ``.anchor(rotations=)`` AT THE AUTHORING LINE.

    Discovering rotation legality at compile depth means a `Box` refusing
    17 degrees three levels inside an anchored assembly, with a traceback
    pointing at the bake rather than at the line that asked for it. That is the
    worst possible discovery point — the author has to reconstruct which of a
    dozen `.anchor()` calls owns the leaf, and the fix (switch the leaf to an
    `Extrude`) is nowhere near the error.

    So the assembly operation validates what its rotation will reach, and
    names BOTH the offending descendant and the rule it broke.

    This does NOT generalise leaf rotation support — the rules are unchanged
    and :func:`_bake_geometry` still enforces them. It also cannot be complete:
    an author who anchors an empty container and fills it afterwards presents
    nothing to check here, which is why the bake-time refusal stays as the
    backstop rather than being replaced.
    """
    total_cd = _rotation_z_only(rotations)
    child_label = (f"{type(child).__name__} "
                   f"{getattr(child, 'name', None) or getattr(child, 'id', None)!r}")
    if total_cd is None:
        raise ChildAnchorError(
            f"{type(container).__name__}.anchor(): {child_label}: "
            f".anchor(rotations=) supports rotation about the parent frame's "
            f"up axis ('z') only — an x/y rotation tilts the solid out of its "
            f"authoring plane. Author the tilt in the child's own geometry, or "
            f"place it with placement=Transform via .add()."
        )
    if not total_cd:
        return
    for elem in iter_rotated_descendants(child):
        reason = rotation_refusal(elem, total_cd)
        if reason is None:
            continue
        where = ("" if elem is child else
                 f" (reached through {child_label}, which you are anchoring)")
        label = (f"{type(elem).__name__} "
                 f"{getattr(elem, 'name', None) or getattr(elem, 'id', None)!r}")
        raise ChildAnchorError(
            f"{type(container).__name__}.anchor(rotations={list(rotations)!r}): "
            f"{label}{where} cannot take this rotation — {reason}"
        )


def _rotate_xy(x: float, y: float, cos_a: float, sin_a: float):
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def _bake_point(p, frame: ElementFrame, spec, cos_a: float, sin_a: float,
                along_eff: float = 0.0):
    """Map one authoring-space point from parent-local into world."""
    lx, ly, lz = float(p.x), float(p.y), float(p.z)
    if cos_a != 1.0 or sin_a != 0.0:
        lx, ly = _rotate_xy(lx, ly, cos_a, sin_a)
    base = frame.local_to_world(
        float(along_eff) + lx, float(spec.inset) + ly, float(spec.up) + lz
    )
    return base


def resolve_child_anchors(project) -> "object":
    """Bake every ``parent.anchor(child)`` into the child's coordinates.

    An anchored SOLID child is a representation item, not an IFC product —
    there is no ``IfcLocalPlacement`` of its own to compose a matrix onto
    (that is what ``_apply_dsl_placement`` does for top-level products), so
    the parent's frame is applied to the authored points directly.

    Runs on the normalized (meters) project, BEFORE ``apply_displacement``,
    so anchored children carry true world coordinates by the time carves are
    inferred — an anchored shelf that penetrates its wall carves it, exactly
    as a world-authored one does. Idempotent: the spec is consumed into
    ``_anchor_resolved``.

    The bake is exact or it is a loud error. Bakeable:

    * any child, translation only;
    * ``Box`` with z-rotations in multiples of 90 degrees (an axis-aligned
      box maps to an axis-aligned box; the corners just swap);
    * ``Extrude`` with any z-rotation (contour points rotate; the winding,
      and therefore the extrusion normal, rotates consistently).

    Anything else — an x/y rotation, a Box at 30 degrees, a rotated Sweep
    whose section roll derives from world-frame references — raises
    :class:`ChildAnchorError` naming the representable alternatives, rather
    than silently emitting a differently-shaped solid.

    **An anchored CONTAINER takes one generation per nesting level.** A
    container holds no points of its own, so the bake reaches through to its
    descendants (:func:`_bake_subtree`); but an anchor authored INSIDE that
    container measures from the container's OWN frame, which does not exist
    until its geometry stands in world coordinates. So the pass baked a
    generation, re-stamps (:func:`stamp_frames` — a full recompute, which is
    what makes the composed frame the ORDINARY frame of the now-placed
    container rather than a second derivation), and goes round again. A model
    with no container anchoring defers nothing, never re-stamps, and takes
    exactly the single pass it always did.
    """
    for _ in range(_MAX_ANCHOR_GENERATIONS):
        deferred = False
        for storey in project.storeys:
            for elem in storey.elements:
                deferred |= _bake_walk(elem)
        for site in (getattr(project, "sites", None) or []):
            deferred |= _bake_walk(site)
        if not deferred:
            return project
        stamp_frames(project)
    # Do NOT name a cycle here. This guard cannot see one: a containment
    # cycle is refused at the .anchor()/.add() line and again by
    # ``naming.find_containment_cycles`` as a compile error, and every walk
    # this loop drives carries an on-path guard — so a cyclic tree is
    # rejected long before it could spend a generation here. The only thing
    # left that reaches this line is genuine anchor nesting depth.
    raise ChildAnchorError(
        f".anchor() nesting exceeded {_MAX_ANCHOR_GENERATIONS} generations of "
        f"anchored containers. This is a runaway guard on anchor nesting "
        f"DEPTH, not a design limit — each generation is one anchored "
        f"container whose own subtree still holds anchors, and real "
        f"assemblies nest a handful deep. Flatten the assembly, or place the "
        f"inner children with .add() in world coordinates."
    )


#: Anchored-container nesting levels one :func:`resolve_child_anchors` call
#: will resolve. A runaway guard, not a design limit — real assemblies nest a
#: handful deep, and the alternative to a cap is an unbounded loop if a tree
#: ever grows a cycle.
_MAX_ANCHOR_GENERATIONS = 32


def _bake_walk(elem, _on_path=None) -> bool:
    """Recurse ``_elements``, baking every unresolved anchor spec.

    Returns True when a bake was DEFERRED — i.e. an anchored container was
    placed whose own subtree still holds anchors. Those resolve against the
    container's frame, and that frame is stale (the container had no
    world geometry to derive one from) until the caller re-stamps.

    ``Window``/``Door`` are skipped ENTIRELY — spec and subtree both. An
    opening is not a solid the parent frame translates: its ``_anchor_spec``
    is consumed as SCALARS by the opening machinery (``along`` is the old
    ``offset=``, ``up`` the old ``sill=``), which walks the host's own run
    axis and cuts a void. Baking it would move nothing that renders and would
    destroy the numbers the void is built from. Its children are
    opening-LOCAL (origin at the opening corner, Z up — see
    ``test_opening_local_zup_contract``), so they must not be dragged into
    world coordinates by the WALL's frame either.

    On-path re-entry is PRUNED, and here that is more than termination: a
    cyclic subtree would otherwise bake the same child into its own frame
    over and over, each pass rewriting coordinates the previous pass wrote.
    A cyclic model is refused before generation; this only has to survive.
    """
    from lite_step.models.elements import Door, Window

    guard = _on_path if _on_path is not None else WalkGuard()
    if not guard.enter(elem):
        return False
    try:
        deferred = False
        for child in (getattr(elem, "_elements", None) or []):
            if isinstance(child, (Window, Door)):
                continue
            spec = getattr(child, "_anchor_spec", None)
            if spec is not None and not getattr(child, "_anchor_resolved", False):
                _bake_child(elem, child, spec)
                if _has_unresolved_anchor(child):
                    deferred = True
                    continue      # its frame is stale until the re-stamp
            deferred |= _bake_walk(child, guard)
        return deferred
    finally:
        guard.leave(elem)


def _has_unresolved_anchor(elem, _on_path=None) -> bool:
    """True when ``elem``'s subtree still holds an anchor the bake must reach.

    ``Window``/``Door`` specs do not count: they are never baked (they are
    consumed as scalars), so counting them would defer forever. Neither does
    a child already on the current path — a cycle carries no anchor the bake
    has not already seen this descent, and answering ``True`` for it would
    defer forever too, which is exactly what ``_MAX_ANCHOR_GENERATIONS``
    catches, a whole 32 generations later.
    """
    from lite_step.models.elements import Door, Window

    guard = _on_path if _on_path is not None else WalkGuard()
    if not guard.enter(elem):
        return False
    try:
        for child in (getattr(elem, "_elements", None) or []):
            if isinstance(child, (Window, Door)):
                continue
            if getattr(child, "_anchor_spec", None) is not None:
                return True
            if _has_unresolved_anchor(child, guard):
                return True
        return False
    finally:
        guard.leave(elem)


def _bake_child(parent, child, spec) -> None:
    frame = getattr(parent, "_frame", None)
    if frame is None:
        raise ChildAnchorError(
            f"{type(parent).__name__} "
            f"{getattr(parent, 'ifc_name', None) or getattr(parent, 'name', None)!r}"
            f": .anchor() needs a frame, but this container has no "
            f"point-bearing geometry to derive one from — give it a body "
            f"child, or author the child in world coordinates with .add()."
        )

    total_cd = _rotation_z_only(spec.rotations)
    label = (f"{type(child).__name__} "
             f"{getattr(child, 'ifc_name', None) or getattr(child, 'name', None)!r}")
    if total_cd is None:
        raise ChildAnchorError(
            f"{label}: .anchor(rotations=) supports rotation about the "
            f"parent frame's up axis ('z') only — an x/y rotation tilts the "
            f"solid out of its authoring plane. Author the tilt in the "
            f"child's own geometry, or place it with placement=Transform "
            f"via .add()."
        )
    angle = math.radians(total_cd / 100.0)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # along_center -> near-edge offset, resolved ONCE for this child, and
    # therefore ONCE for its whole subtree — the offset is part of the single
    # rigid transform, not a per-element correction.
    along_eff = resolve_along(spec, child, cos_a=cos_a, sin_a=sin_a)

    _bake_subtree(child, frame, spec, cos_a, sin_a, along_eff, total_cd)

    child._anchor_resolved = True
    child._anchor_spec = None


def _bake_subtree(elem, frame: ElementFrame, spec, cos_a: float, sin_a: float,
                  along_eff: float, total_cd: float, _on_path=None) -> None:
    """Map ``elem``'s own points into ``frame``, then its descendants'.

    A geometry primitive IS its points, so for the ordinary
    ``wall.anchor(box, along=…)`` case this bottoms out immediately in
    :func:`_bake_geometry` and the walk below never runs. A CONTAINER holds no
    points at all — its children do — so without this reach an anchored
    assembly would bake to nothing and keep its local coordinates.

    The walk stops at two kinds of child, for the same reason in both cases:
    their coordinates are not in this container's frame.

    * a ``Window``/``Door`` — its spec is scalars the opening machinery
      resolves against the container's own frame once that frame is world
      (``_bake_walk`` skips openings on identical grounds), and its children
      are opening-local;
    * a child that is ITSELF anchored — its points are relative to the
      container's frame, which does not exist until this bake has run and
      ``stamp_frames`` has re-derived it from the placed geometry. It is baked
      in the NEXT generation. That deferral IS the composition: the inner
      child's own rotation applies inside the container's frame, the
      container's rotation applies inside the host's, and so on outward —
      never the other way round.

    It stops at a third: a child already on the current path. That is a
    containment cycle, and baking the same points twice in one descent would
    apply the transform twice — silently, since the second application still
    produces coordinates. A cyclic model is refused before generation.
    """
    from lite_step.models.elements import Door, Window

    guard = _on_path if _on_path is not None else WalkGuard()
    if not guard.enter(elem):
        return
    try:
        _bake_geometry(elem, frame, spec, cos_a, sin_a, along_eff, total_cd)
        for child in (getattr(elem, "_elements", None) or []):
            if isinstance(child, (Window, Door)):
                continue
            if getattr(child, "_anchor_spec", None) is not None:
                continue
            _bake_subtree(child, frame, spec, cos_a, sin_a, along_eff,
                          total_cd, guard)
    finally:
        guard.leave(elem)


def _bears_points(elem) -> bool:
    """True when ``elem`` carries geometry of its OWN (rather than in children).

    Distinguishes a container — for which a z-rotation is carried down to the
    descendants that do bear points — from a ``Sweep``/``Revolve``/``Pipe``/
    ``Bar``/``Mesh``, for which it is refused.
    """
    from lite_step.models import Point

    for attr in ("start", "end", "location"):
        if isinstance(getattr(elem, attr, None), Point):
            return True
    for attr in ("contour", "path", "vertices", "points"):
        seq = getattr(elem, attr, None)
        if isinstance(seq, list) and seq and isinstance(seq[0], Point):
            return True
    return False


def _bake_geometry(child, frame: ElementFrame, spec, cos_a: float,
                   sin_a: float, along_eff: float, total_cd: float) -> None:
    """Rewrite ONE element's own point-bearing fields into ``frame``."""
    from lite_step.models.elements import Box, Extrude

    label = (f"{type(child).__name__} "
             f"{getattr(child, 'ifc_name', None) or getattr(child, 'name', None)!r}")

    # The rules themselves live in ``rotation_refusal`` — ONE statement, read
    # both here and by the call-site check. This is the backstop: it catches
    # the subtree an author populated AFTER anchoring, which the call site
    # could not have seen.
    refusal = rotation_refusal(child, total_cd)
    if refusal is not None:
        raise ChildAnchorError(f"{label}: {refusal}")

    if isinstance(child, Box):
        s = _bake_point(child.start, frame, spec, cos_a, sin_a, along_eff)
        e = _bake_point(child.end, frame, spec, cos_a, sin_a, along_eff)
        # Re-normalize to min/max so start stays the low corner.
        child.start = child.start.model_copy(update={
            "x": min(s[0], e[0]), "y": min(s[1], e[1]), "z": min(s[2], e[2])})
        child.end = child.end.model_copy(update={
            "x": max(s[0], e[0]), "y": max(s[1], e[1]), "z": max(s[2], e[2])})
    elif isinstance(child, Extrude) and child.contour:
        child.contour = [
            pt.model_copy(update=dict(zip(
                ("x", "y", "z"), _bake_point(pt, frame, spec, cos_a, sin_a, along_eff))))
            for pt in child.contour
        ]
    elif not _bears_points(child):
        # A container. It has nothing of its own to move — its geometry lives
        # in the children ``_bake_subtree`` is walking — and a z-rotation on it
        # is legal precisely BECAUSE it is not applied here: each descendant
        # meets the same rotation under its own type's rule.
        return
    else:
        # ``rotation_refusal`` has already established total_cd == 0 here.
        _bake_translation_only(child, frame, spec, along_eff)


def _bake_translation_only(child, frame: ElementFrame, spec,
                           along_eff: float = 0.0) -> None:
    """Translate every point-bearing field of an arbitrary solid."""
    for attr in ("start", "end", "location"):
        p = getattr(child, attr, None)
        if p is not None and hasattr(p, "x"):
            v = _bake_point(p, frame, spec, 1.0, 0.0, along_eff)
            setattr(child, attr, p.model_copy(update=dict(zip(("x", "y", "z"), v))))
    for attr in ("contour", "path", "vertices", "points"):
        seq = getattr(child, attr, None)
        if isinstance(seq, list) and seq and hasattr(seq[0], "x"):
            setattr(child, attr, [
                pt.model_copy(update=dict(zip(
                    ("x", "y", "z"), _bake_point(pt, frame, spec, 1.0, 0.0, along_eff))))
                for pt in seq
            ])


def stamp_frames(project) -> "object":
    """Stamp ``_frame`` on every element in ``project`` (top-down).

    Mirrors :func:`lite_step.compiler.naming.stamp_canonical_names`: a full
    recompute from the tree, idempotent, safe to call at every pipeline
    entry point. Units follow the project instance (mm pre-normalize,
    meters post-normalize), which is why the pass is re-run after
    normalization rather than trying to be unit-agnostic.

    Returns the same project, stamped in place.
    """
    centers = scope_centers(project)
    building = centers.get("building")
    for storey in project.storeys:
        storey_center = centers.get(f"storey:{id(storey)}") or building
        spaces = _spaces_in(storey)
        for elem in storey.elements:
            _walk(elem, (), storey_center, spaces)
    site_center = centers.get("site")
    for site in (getattr(project, "sites", None) or []):
        _walk(site, (), site_center)
    return project
