"""The ONE padded-AABB derivation — the extent of a solid, in authoring space.

Two consumers read this module and they must never diverge:

* :mod:`lite_step.compiler.displacement` infers carves from AABB overlap, so
  an under-approximation silently loses a boolean and an over-approximation
  manufactures a phantom one (each of which costs CSG depth).
* ``BimElement.authored_aabb()`` hands the same box to a model author who is
  about to place joinery against it.

A second, subtly different copy is how a bound silently stops bounding the
solid it describes — the same reasoning that keeps ``segment_axes`` in one
place (``lite_step/ifc/geometry.py``).

**Why this is NOT ``frames.element_aabb``.** That function walks authored
POINTS: ``start``/``end``/``location``/``contour``/``path``/``vertices``. For a
``Box`` the points *are* the extent, but for five of the seven primitives they
are not —

===============  ==================================================
``Pipe``         the points are the centreline; the solid reaches
                 ``radius`` in every perpendicular direction
``Bar``          same, at ``diameter / 2``
``Extrude``      the contour is a zero-width slab; the solid is
                 that polygon swept ``thickness`` along its normal
``Sweep``        the path is the axis; the profile is what has width
``Revolve``      the path is the AXIS OF REVOLUTION, not the body
===============  ==================================================

so ``element_aabb`` alone under-reports, which is the dangerous direction.
This module is that walk plus the per-type pad that turns it back into a
bound.

**Containers pad per CHILD, never once at the top.** ``element_aabb`` already
recurses into ``_elements``, so a naive ``pad(element_aabb(wall))`` keys the
pad on the *wall's* type name — and ``Wall`` is not ``Pipe``, so the pad is
zero and a wall containing a pipe reports the pipe's centreline. See
:func:`element_extent`.

**The unit invariant — every function here answers in the units of the
coordinates it was given.** The two consumers run in DIFFERENT unit domains:
displacement runs POST-``normalize_project_to_meters``, in metres; the
bounding queries run at authoring time, in millimetres. Most of the pads are
derived from the element's own coordinates (a ``Pipe``'s ``radius``, a
``Sweep``'s explicit ``profile`` points, an ``Extrude``'s authored
``thickness``) and are therefore unit-agnostic for free — the normalizer
scaled them along with everything else.

Two are NOT, because their source is a ``Material`` FACT and a Material is
frozen: ``Material(profile_mm=)`` and ``Material(thickness_mm=)`` are in
millimetres in BOTH domains, forever. Nothing about an element's own
coordinates can disclose which domain it is in — a path 1000 long is 1000 mm
or 1000 m with equal plausibility — so the conversion cannot be derived; it
has to be *stated*. Every function that reads one of those two facts takes
``divisor``, exactly as ``frames.opening_body_thickness`` already does:
:data:`MM_PER_METER` for a caller in metres, the default ``1.0`` for the
authoring-millimetre query layer (division by ``1.0`` is exact, so the query
layer pays no float cost for the shared path).

The default is the mm domain deliberately. A caller that forgets it in the
metres domain over-approximates by 1000x — phantom carves, which cost CSG
depth and show up immediately in a corpus byte-comparison. The other way
round the pad rounds to nothing and the bound silently stops bounding, which
is what the queries did until then: a 45x195 member reported
``size == (1000, 0, 0)``, so the author who went and MEASURED was told their
member was infinitely thin.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import List, Optional, Tuple

import numpy as np

Aabb = Tuple[Tuple[float, float, float], Tuple[float, float, float]]

#: ``divisor`` for a caller whose coordinates are in metres — i.e. anything
#: running after ``normalize_project_to_meters``. See the unit invariant in
#: the module docstring.
MM_PER_METER = 1000.0

#: Scoped memo for :func:`element_extent`, or ``None`` when off. OFF is the
#: default and must stay that way — see :func:`memoized_extents` for the one
#: condition under which turning it on is sound. A ContextVar rather than a
#: module global so a nested or concurrent caller cannot inherit someone
#: else's cache by accident.
_EXTENT_MEMO: contextvars.ContextVar = contextvars.ContextVar(
    "lite_step_extent_memo", default=None)


@contextlib.contextmanager
def memoized_extents():
    """Cache :func:`element_extent` for the duration of a READ-ONLY walk.

    Why it is worth having: a container's extent is the union of each child's
    OWN extent, computed by recursing back into ``element_extent``. So asking
    for every node in a tree — which the compile-time tree report does, one
    row per node — recomputes each subtree once per ancestor above it. The
    work is O(nodes x depth) where O(nodes) would do.

    **Why it is OPT-IN, and must stay so.** Geometry MUTATES between compiler
    passes: ``normalize_project_to_meters`` divides by 1000, and
    ``resolve_child_anchors`` bakes anchored children into world coordinates.
    A memo that outlived a pass would hand out a pre-normalize box a thousand
    times too large, or a pre-bake box at the authoring origin — the exact
    stale-frame class, and the
    reason the equivalent memo for ``frames`` was measured and then DROPPED
    rather than shipped.

    The tree report is the case that is safe, and it is safe for a specific
    reason rather than by inspection: it runs AFTER every mutating pass, it
    only reads, and the memo dies with the ``with`` block. Do not widen this
    to a compile-long cache without re-arguing that.

    The key includes ``divisor``: the same element has different numbers in
    the mm and metre domains, and both are live in one compile.
    """
    token = _EXTENT_MEMO.set({})
    try:
        yield
    finally:
        _EXTENT_MEMO.reset(token)


def overlap_depths(a: Aabb, b: Aabb) -> Tuple[float, float, float]:
    """Per-axis overlap depth of two boxes: ``> 0`` interpenetrating, ``0``
    flush, ``< 0`` a gap of that size.

    The ONE axis-wise comparison behind every "do these two solids meet"
    question in the compiler, and it lives beside the extent derivation
    because the boxes it compares come from here.

    Two consumers, and they need OPPOSITE sides of the same number.
    ``displacement._overlaps`` demands POSITIVE overlap, because abutting
    faces must never trigger a carve. ``ifc.space_boundaries`` accepts a FLUSH
    face, because a wall standing against a room's face is precisely what
    bounds it. Written twice, those two would differ by one comparison
    operator and agree on every model where the difference did not matter —
    the shape ("two implementations of one idea agreeing *usually* is
    what produced the bug"). Written once, the difference is the ``eps`` each
    caller passes, and it is visible at both call sites.

    Unit-agnostic: it compares coordinates and returns their difference, so it
    answers in whatever units the boxes are in.
    """
    (a0, a1), (b0, b1) = a, b
    return (min(a1[0], b1[0]) - max(a0[0], b0[0]),
            min(a1[1], b1[1]) - max(a0[1], b0[1]),
            min(a1[2], b1[2]) - max(a0[2], b0[2]))


def _member_section_mm(elem) -> Optional[Tuple[float, float]]:
    """The (width_mm, depth_mm) section of a profile member whose ``profile``
    contour is empty — from ``Material(profile_mm=(w,h))`` (the one-source
    member idiom). Mirrors the generator's read. Returns None when no member
    material profile applies."""
    try:
        from lite_step.materials import element_registry_material
        sel = element_registry_material(elem)
        if sel is not None and getattr(sel, "profile_mm", None):
            w, h = sel.profile_mm
            return float(w), float(h)
    except Exception:
        pass
    return None


def _member_section_reach(elem, divisor: float = 1.0) -> float:
    """Half-diagonal reach of a member's ``Material(profile_mm=)`` section, in
    the CALLER's units — ``0.0`` when the element carries no member section.

    THE one place that expression lives. It was written out twice (the
    displacement pad and the OBB radial reach) with two different unit
    treatments, which is precisely how one of them came to be right for one
    caller and wrong by 1000x for the other. See the module docstring.
    """
    wh = _member_section_mm(elem)
    if wh is None:
        return 0.0
    w_mm, h_mm = wh
    return 0.5 * (float(w_mm) ** 2 + float(h_mm) ** 2) ** 0.5 / divisor


def _member_section_profile(elem, divisor: float = 1.0):
    """The member's ``Material(profile_mm=)`` section as an explicit centred
    rectangle in the CALLER's units — or ``None`` when it has no such section.

    The generator emits a member as an ``IfcRectangleProfileDef(x_dim=w,
    y_dim=h)`` on an ``IfcAxis2Placement3D`` built from ``segment_axes``, and
    an ``IfcRectangleProfileDef`` is CENTRED on its position. So these four
    corners are the same section, in the same frame, that the shipped solid
    carries — which is what lets :func:`_sweep_path_aabb` bound a member the
    same way it already bounds a ``Sweep`` with an authored ``profile=``.

    ``model_construct``, not the validating constructor: a 45 mm section has
    half-width 22.5 and the strict int-mm rule (DSL v1.5 RULE 2) refuses
    floats. That rule polices what an AUTHOR writes; these four points are
    derived internally, never emitted, and read only for their ``x``/``y`` —
    rounding them to keep a validator happy would make the bound wrong by up
    to half a millimetre in the under-approximating direction.
    """
    wh = _member_section_mm(elem)
    if wh is None:
        return None
    from lite_step.models import Point2D

    hw = 0.5 * float(wh[0]) / divisor
    hh = 0.5 * float(wh[1]) / divisor
    return [Point2D.model_construct(x=-hw, y=-hh),
            Point2D.model_construct(x=hw, y=-hh),
            Point2D.model_construct(x=hw, y=hh),
            Point2D.model_construct(x=-hw, y=hh)]


def _angular_component_range(r_min: float, r_max: float,
                            a0: float, a1: float, phase: float):
    """``(lo, hi)`` of ``R * cos(phi + phase)`` for R in [r_min, r_max],
    phi in [a0, a1] — the exact extent of an annulus SECTOR on one axis.

    A full revolution reaches +/-1 on both axes and the sector reduces to the
    endpoints plus whichever quadrant boundaries it crosses. Doing this
    exactly is what turns a 60-degree sector's bound from the full circle
    into the wedge it actually occupies.
    """
    import math

    lo_c, hi_c = math.cos(a0 + phase), math.cos(a1 + phase)
    lo, hi = min(lo_c, hi_c), max(lo_c, hi_c)
    # A quadrant boundary inside the swept range pins the extreme at +/-1.
    k = math.floor((a0 + phase) / (math.pi / 2.0))
    while k * (math.pi / 2.0) <= a1 + phase + 1e-12:
        angle = k * (math.pi / 2.0)
        if a0 + phase - 1e-12 <= angle <= a1 + phase + 1e-12:
            c = math.cos(angle)
            lo, hi = min(lo, c), max(hi, c)
        k += 1
    # R >= 0, so a positive cosine is largest at r_max and a negative one is
    # most negative there; the near radius bounds the other side.
    hi_v = r_max * hi if hi > 0 else r_min * hi
    lo_v = r_max * lo if lo < 0 else r_min * lo
    return lo_v, hi_v


def _revolve_aabb(elem):
    """The TIGHT world box of a ``Revolve``, or ``None`` to fall back.

    A bound that pads the axis endpoints by the profile's radial maximum
    **isotropically** — in all three axes — and said so in its own comment:
    "the extent overshoots along the revolve AXIS". Conservative against a
    missed carve, and it manufactures phantom ones instead. Measured on a
    5-wing 6-storey plan of annular floorplates: **1335 carve pairs of which
    ~30 are genuine**, CSG depth 7 of 10 before a single wall exists, because
    a 275 mm plate claimed a 60 m cube and every storey "overlapped" every
    other.

    The geometry is knowable exactly, and ``generator._create_revolve`` fixes
    the frame: profile local **Y is the axis** (``path[0] -> path[1]``), local
    **X is radial** from ``r = unit(world_Z x a)``. So the solid spans the
    profile's Y range along the axis and its X range radially — and the sweep
    ANGLE clips the radial disc to a sector.
    """
    import math

    path = getattr(elem, "path", None) or []
    prof = getattr(elem, "profile", None) or []
    if len(path) < 2 or not prof:
        return None

    p0, p1 = path[0], path[1]
    ax = (float(p1.x) - float(p0.x), float(p1.y) - float(p0.y),
          float(p1.z) - float(p0.z))
    norm = math.sqrt(sum(c * c for c in ax))
    if norm <= 1e-12:
        return None
    a = tuple(c / norm for c in ax)

    # Same radial start direction the generator uses; degenerate for a
    # vertical axis, where it falls back to world +X exactly as _create_revolve
    # does.
    rx = (-a[1], a[0], 0.0)                       # world_Z x a
    rn = math.sqrt(sum(c * c for c in rx))
    r_hat = (1.0, 0.0, 0.0) if rn <= 1e-9 else tuple(c / rn for c in rx)
    s_hat = (a[1] * r_hat[2] - a[2] * r_hat[1],
             a[2] * r_hat[0] - a[0] * r_hat[2],
             a[0] * r_hat[1] - a[1] * r_hat[0])   # a x r

    r_min = min(abs(float(q.x)) for q in prof)
    r_max = max(abs(float(q.x)) for q in prof)
    y_min = min(float(q.y) for q in prof)
    y_max = max(float(q.y) for q in prof)

    angle_deg = float(getattr(elem, "angle", 36000) or 36000) / 100.0
    theta = math.radians(min(abs(angle_deg), 360.0))
    if theta >= 2 * math.pi - 1e-9:
        r_lo, r_hi = -r_max, r_max
        s_lo, s_hi = -r_max, r_max
    else:
        r_lo, r_hi = _angular_component_range(r_min, r_max, 0.0, theta, 0.0)
        # sin(phi) = cos(phi - pi/2)
        s_lo, s_hi = _angular_component_range(r_min, r_max, 0.0, theta,
                                              -math.pi / 2.0)

    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    base = (float(p0.x), float(p0.y), float(p0.z))
    for ay in (y_min, y_max):
        for cr in (r_lo, r_hi):
            for cs in (s_lo, s_hi):
                for i in range(3):
                    v = base[i] + a[i] * ay + r_hat[i] * cr + s_hat[i] * cs
                    lo[i] = min(lo[i], v)
                    hi[i] = max(hi[i], v)
    return (tuple(lo), tuple(hi))


def padded_aabb(elem, divisor: float = 1.0) -> Optional[Aabb]:
    """AABB of a solid for the displacement trigger. ``_element_aabb`` tracks
    path/contour POINTS only, so a curved solid with a straight axis-aligned
    path (a rebar ``Bar``, a ``Pipe`` run) collapses to a degenerate box the
    positive-overlap test can never pass. Pad by the solid's reach.

    The pad is a BOUND, not exact. Over-approximation is tolerated (a boolean
    with a non-intersecting operand is a geometric no-op) but is NOT free: an
    inflated AABB manufactures PHANTOM carves — booleans against elements the
    solid never touches — and every phantom carve deepens the host's CSG chain
    toward the depth at which the primary web viewer silently drops the
    subtractions and renders the base solid. So the pad is as tight as it can
    be while staying a bound:

    * ``Bar`` / ``Pipe`` — the exact radius, isotropic (a swept disk really does
      reach that far in every direction perpendicular to the path).
    * path-form ``Sweep`` with an explicit ``profile`` — a DIRECTIONAL pad from
      the real section frame (see ``_sweep_path_aabb``). An eave-to-ridge roof
      profile reaches ~6.4 m ACROSS the sweep and a few hundred mm through it;
      padding 6.4 m in every axis (a radial-max rule) swallows the whole building
    and makes every enclosed element a spurious carve.
    * member-form ``Sweep`` (section from ``Material(profile_mm=)``) — the
      section half-diagonal, isotropic. Small by construction, and it is what
      makes a rafter participate in displacement at all.
    * ``Revolve`` — the profile's radial max, isotropic. A revolve genuinely
      sweeps its profile around the axis, so the radial bound is the tight one.
    * contour-mode ``Extrude`` — the contour points describe a zero-width slab;
      pad by the extrusion ``thickness`` projected onto each axis (see
      ``_extrude_thickness_pad``). Without it the AABB can never register
      overlap along the extrusion axis and the element silently never carves.

    ``divisor`` states the caller's unit domain — see the module docstring.
    The returned box is always in the units of ``elem``'s own coordinates.
    """
    from lite_step.compiler.placement import _element_aabb

    aabb = _element_aabb(elem)
    if aabb is None:
        return None
    tname = type(elem).__name__
    pad = 0.0
    if tname == "Bar":
        pad = float(getattr(elem, "diameter", 0) or 0) / 2.0
    elif tname == "Pipe":
        pad = float(getattr(elem, "radius", 0) or 0)
    elif tname == "Extrude":
        return _pad_axes(aabb, _extrude_thickness_pad(elem, divisor))
    elif tname in ("Sweep", "Revolve"):
        prof = getattr(elem, "profile", None) or []
        if prof:
            if tname == "Sweep" and len(getattr(elem, "path", None) or []) >= 2:
                swept = _sweep_path_aabb(elem, prof)
                if swept is not None:
                    return swept
            if tname == "Revolve":
                exact = _revolve_aabb(elem)
                if exact is not None:
                    return exact
            pad = max((float(p.x) ** 2 + float(p.y) ** 2) ** 0.5 for p in prof)
        else:
            # No explicit contour: the section comes from Material(profile_mm=
            # (w,h)) (the canonical member idiom). elem.profile
            # is None, so without this the pad is 0 and the AABB collapses to a
            # zero-width line — the overlap trigger can never fire and the member
            # never participates in displacement. Pad by the section's half-
            # diagonal reach, converted into the CALLER's units: profile_mm is a
            # frozen Material fact and stays mm in both domains, so ``divisor``
            # is the only thing that knows which domain this is.
            pad = _member_section_reach(elem, divisor)
    if pad <= 0.0:
        return aabb
    return _pad_axes(aabb, (pad, pad, pad))


def _pad_axes(aabb: Aabb, pad: Tuple[float, float, float]) -> Aabb:
    (x0, y0, z0), (x1, y1, z1) = aabb
    px, py, pz = pad
    return (x0 - px, y0 - py, z0 - pz), (x1 + px, y1 + py, z1 + pz)


def _extrude_thickness_pad(elem, divisor: float = 1.0) -> Tuple[float, float, float]:
    """Per-axis pad for a contour ``Extrude``: the extrusion depth projected
    onto each world axis.

    A contour ``Extrude`` is a planar polygon swept by ``thickness`` along its
    plane normal, but ``_element_aabb`` only sees the contour POINTS — so the
    AABB is a zero-width slab in the extrusion direction and
    :func:`_overlaps` (which demands POSITIVE overlap on every axis) can never
    fire. That is a silent non-carve: an insulation slab authored as a contour
    Extrude sat inside the rafters and was never subtracted from them.

    Padded on BOTH sides of the plane, deliberately: an over-approximation of
    one slab thickness is the cheap direction to be wrong in for a carve
    TRIGGER. :func:`extrude_offset` knows which side is real and is what the
    authoring query uses — see the note there on why the two differ on
    purpose rather than by neglect.
    """
    span = extrude_offset(elem, divisor)
    if span is None:
        return (0.0, 0.0, 0.0)
    return tuple(abs(c) for c in span)


def _extrude_normal(elem):
    """Unit normal the generator extrudes this contour along, or ``None``.

    THE shared derivation — ``generator.py`` builds its extrusion axis from
    exactly this pair of calls (``_create_extruded_body`` sweeps local +Z by
    ``Depth``, and the local frame's Z is this normal), so a second copy here
    would be the "AABB silently stops bounding its solid" failure in its
    purest form.
    """
    contour = getattr(elem, "contour", None) or []
    if len(contour) < 3:
        return None
    from lite_step.ifc.geometry import apply_orientation_rules, newell_normal
    n = newell_normal([(float(p.x), float(p.y), float(p.z)) for p in contour])
    return apply_orientation_rules(n) if n is not None else None


def extrude_offset(elem, divisor: float = 1.0
                   ) -> Optional[Tuple[float, float, float]]:
    """Signed per-axis offset from the contour plane to the extruded far face.

    A contour ``Extrude`` is a planar polygon swept ``thickness`` along
    :func:`_extrude_normal`, and the sweep goes ONE way — the ``+normal`` way,
    because ``apply_orientation_rules`` has already resolved the winding
    ambiguity. So the solid occupies the contour hull plus this offset, and
    nothing on the other side.

    That is what makes ``authored_aabb()`` on an ``Extrude`` exact. Padding
    symmetrically (what the carve trigger does) reports a 350 mm gable wall as
    700 mm across, which is a safe BOUND and a useless answer to "where is the
    face of this wall" — the question the authoring query exists for.

    Returns ``None`` when there is no contour or no thickness source.

    ``divisor`` applies to the ``Material(thickness_mm=)`` fallback ONLY — an
    authored ``thickness=`` is an element coordinate and the normalizer has
    already scaled it. This is the second of the two mm-fact reads described
    in the module docstring: a 22 mm sheet answered ``0.022`` to the
    authoring-mm query layer until then.
    """
    n = _extrude_normal(elem)
    if n is None:
        return None
    thickness = getattr(elem, "thickness", None)
    t = float(thickness) if thickness else _material_thickness(elem, divisor)
    if t <= 0:
        return None
    return (t * n[0], t * n[1], t * n[2])


def _sweep_path_aabb(elem, prof) -> Optional[Aabb]:
    """Tight world AABB of a path-form ``Sweep`` with an explicit profile.

    Per path segment the profile sits in a KNOWN frame — ``segment_axes`` (the
    single shared definition the generator emits with) maps the section's local
    (x, y) onto world axes — so the profile's local min/max in x and y map to
    four corner offsets, applied at both segment ends. The hull of those corners
    over all segments is the swept body's bound, and it is directional: a roof
    profile that reaches 6.3 m ACROSS the sweep contributes 6.3 m across and
    nothing along or (much) above.

    Two places the bound is deliberately loosened rather than risk
    UNDER-approximating:

    * a near-vertical segment's frame depends on the caller's ``vertical_ref``
      (window frames pass the wall direction), so its rotation about the path
      axis is not knowable here — use the rotation-invariant bound instead
      (radial reach ``r`` perpendicular to the path: ``r * sqrt(1 - d_k^2)``).
    * an interior joint is MITERED — the outer corner of the joint reaches past
      both segments' end boxes, and the segments are built with an overshoot
      before being trimmed. Pad joint vertices isotropically by
      ``r + overshoot`` with the generator's own overshoot formula.

    Returns ``None`` when the path is degenerate (caller falls back to the
    isotropic radial pad).
    """
    path = list(getattr(elem, "path", None) or [])
    if len(path) < 2:
        return None
    pts = [(float(p.x), float(p.y), float(p.z)) for p in path]
    px = [float(p.x) for p in prof]
    py = [float(p.y) for p in prof]
    lo_x, hi_x, lo_y, hi_y = min(px), max(px), min(py), max(py)
    corners_2d = [(lo_x, lo_y), (hi_x, lo_y), (hi_x, hi_y), (lo_x, hi_y)]
    r = max((cx ** 2 + cy ** 2) ** 0.5 for cx, cy in corners_2d)
    rot_cd = float(getattr(elem, "profile_rotation", 0) or 0)

    from lite_step.ifc.geometry import segment_axes, VERTICAL_SEGMENT_DZ

    world: List[np.ndarray] = []
    dirs: List[Optional[np.ndarray]] = []
    for i in range(len(pts) - 1):
        a = np.array(pts[i], dtype=np.float64)
        b = np.array(pts[i + 1], dtype=np.float64)
        seg = b - a
        ln = float(np.linalg.norm(seg))
        if ln < 1e-12:
            dirs.append(None)
            continue
        d = seg / ln
        dirs.append(d)
        if abs(d[2]) > VERTICAL_SEGMENT_DZ:
            # Frame ambiguous around the axis → rotation-invariant bound.
            pad = np.array([r * (max(0.0, 1.0 - d[k] ** 2)) ** 0.5
                            for k in range(3)])
            world += [a - pad, a + pad, b - pad, b + pad]
            continue
        x_axis, y_axis = segment_axes(d, profile_rotation_cd=rot_cd)
        for cx, cy in corners_2d:
            off = cx * np.asarray(x_axis) + cy * np.asarray(y_axis)
            world += [a + off, b + off]

    # Mitered interior joints: the outer corner reaches past both end boxes.
    real = [d for d in dirs if d is not None]
    if len(real) >= 2:
        closed = len(pts) >= 4 and pts[0] == pts[-1]
        joints = [(i, np.array(pts[i + 1], dtype=np.float64))
                  for i in range(len(dirs) - 1)
                  if dirs[i] is not None and dirs[i + 1] is not None]
        if closed and dirs[0] is not None and dirs[-1] is not None:
            joints.append((len(dirs) - 1, np.array(pts[0], dtype=np.float64)))
        for i, vertex in joints:
            d_in = dirs[i]
            d_out = dirs[(i + 1) % len(dirs)]
            if d_in is None or d_out is None:
                continue
            cos_t = max(-1.0, min(1.0, float(np.dot(d_in, d_out))))
            theta = float(np.arccos(cos_t))
            # Same overshoot the generator extends each segment by before the
            # bisector half-space trims it back.
            over = r * np.tan(theta / 2.0) * 1.05 + 0.001 if theta > 1e-9 else 0.0
            j = r + float(over)
            world += [vertex - j, vertex + j]

    if not world:
        return None
    arr = np.asarray(world, dtype=np.float64)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    return (float(lo[0]), float(lo[1]), float(lo[2])), \
           (float(hi[0]), float(hi[1]), float(hi[2]))


#: An oriented box: ``(centre, axes, half_extents)``. ``axes`` is three unit
#: row vectors; ``half_extents`` are measured along them.
Obb = Tuple[np.ndarray, np.ndarray, np.ndarray]


def convex_solid_obb(elem, divisor: float = 1.0) -> Optional[Obb]:
    """``elem`` as an ORIENTED BOX that contains its solid — or ``None``.

    For a right-angled volume the OBB is not an approximation of the solid,
    it IS the solid: a ``Box`` and a straight member with a rectangular
    section are each exactly six planes. So a separating-axis test between two
    of them is an exact intersection answer, for fifteen dot products and no
    tessellation, no mesh boolean and no geometry dependency.

    That is what the displacement narrow phase runs on. The trigger in front of
    it is a padded AABB — a deliberate over-approximation (see
    :func:`padded_aabb`) that pads a member by its section reach so it takes
    part in displacement at all. The price of that pad is a boolean handed to
    every pair that merely comes CLOSE, and a boolean that removes nothing is
    not free: it adds vertices, it adds a level to the host's CSG chain, and
    web-ifc drops the deepest chains outright. The pad is the right broad
    phase and the wrong final answer.

    ``None`` means "not a right-angled volume" — a curved solid, a mesh, a bent
    member, a non-rectangular profile — and the caller must then KEEP the
    carve. Nothing here ever proves an intersection; it only ever proves the
    absence of one.

    Two deliberate over-approximations, both in the safe direction (a box
    larger than the solid can only fail to drop a carve, never drop a real
    one):

    * **Near-vertical members.** ``segment_axes``' rotation about the path
      axis depends on a ``vertical_ref`` the generator supplies and this
      module cannot see, so the section rectangle's orientation is unknowable
      here. Substitute the axis-aligned box of the rotation-invariant radial
      reach — still right-angled, just not tight. Same substitution, and the
      same reason, as :func:`_sweep_path_aabb`.
    * **The AUTHORED solid**, before ``.difference()`` operands and miter
      clips. Those only ever remove material, so ignoring them over-states.
    """
    tname = type(elem).__name__
    if tname == "Box":
        s, e = getattr(elem, "start", None), getattr(elem, "end", None)
        if s is None or e is None:
            return None
        # No ``divisor`` on an element's OWN coordinates — they are already in
        # the caller's domain. ``divisor`` converts the frozen millimetre facts
        # (``Material(profile_mm=)``) and nothing else; see the module
        # docstring's unit invariant.
        lo = np.array([min(float(s.x), float(e.x)), min(float(s.y), float(e.y)),
                       min(float(s.z), float(e.z))])
        hi = np.array([max(float(s.x), float(e.x)), max(float(s.y), float(e.y)),
                       max(float(s.z), float(e.z))])
        if np.any(hi - lo <= 0.0):
            # A flat box is a plane, and a zero-extent box separates from
            # EVERYTHING — it would silently delete every carve this element
            # takes part in. Refuse to answer rather than answer wrongly.
            return None
        return (lo + hi) / 2.0, np.eye(3), (hi - lo) / 2.0
    if tname != "Sweep":
        return None
    path = list(getattr(elem, "path", None) or [])
    if len(path) != 2:
        # A bent member is excluded on purpose. Its OBB fills the inside of the
        # bend, which is sound but so loose it would decide nothing.
        return None
    prof = getattr(elem, "profile", None) or _member_section_profile(elem, divisor)
    if not prof:
        return None
    px = [float(p.x) for p in prof]
    py = [float(p.y) for p in prof]
    lo_x, hi_x, lo_y, hi_y = min(px), max(px), min(py), max(py)
    # The section must be a rectangle SQUARE to the section frame, because that
    # is the only shape whose sweep is an oriented box. Compare the point set
    # against the four corners of its own bounding rectangle: an L, a triangle,
    # or a rectangle rotated inside the frame all fail, and correctly so — for
    # them the box would be a bound, and a bound cannot drop a carve.
    corners = {(round(x, 9), round(y, 9)) for x, y in zip(px, py)}
    if corners != {(round(u, 9), round(v, 9))
                   for u in (lo_x, hi_x) for v in (lo_y, hi_y)}:
        return None
    if hi_x - lo_x <= 0.0 or hi_y - lo_y <= 0.0:
        # A section with no width or no height collapses the box to a plane,
        # and a zero-extent box separates from EVERYTHING — it would silently
        # delete every carve the element takes part in. Refuse to answer.
        return None

    a = np.array([float(path[0].x), float(path[0].y), float(path[0].z)])
    b = np.array([float(path[1].x), float(path[1].y), float(path[1].z)])
    seg = b - a
    ln = float(np.linalg.norm(seg))
    if ln < 1e-12:
        return None
    d = seg / ln
    centre_2d = np.array([(lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0])
    half = np.array([(hi_x - lo_x) / 2.0, (hi_y - lo_y) / 2.0])

    from lite_step.ifc.geometry import segment_axes, VERTICAL_SEGMENT_DZ

    if abs(d[2]) > VERTICAL_SEGMENT_DZ:
        r = max((cx ** 2 + cy ** 2) ** 0.5
                for cx in (lo_x, hi_x) for cy in (lo_y, hi_y))
        lo = np.minimum(a, b) - r
        hi = np.maximum(a, b) + r
        return (lo + hi) / 2.0, np.eye(3), (hi - lo) / 2.0

    x_axis, y_axis = segment_axes(
        d, profile_rotation_cd=float(getattr(elem, "profile_rotation", 0) or 0))
    x_axis = np.asarray(x_axis, dtype=np.float64)
    y_axis = np.asarray(y_axis, dtype=np.float64)
    centre = (a + b) / 2.0 + centre_2d[0] * x_axis + centre_2d[1] * y_axis
    axes = np.array([d, x_axis, y_axis])
    return centre, axes, np.array([ln / 2.0, half[0], half[1]])


#: Contact tolerance, RELATIVE to the size of the two boxes. Absolute would
#: have to know the unit domain — this module is called in metres by
#: displacement and in millimetres by the query layer — and would be either
#: meaningless or dangerous in the other one.
#:
#: Contact must land on the DISJOINT side: two members butted face to face
#: share a plane and nothing else, so a boolean between them removes exactly
#: zero. At 1e-12 of the summed extents that is ~1 nm on a 1 m member: far
#: below anything authorable (the DSL is integer millimetres, and the smallest
#: real joint in the corpus is a 1 mm birdsmouth, a million times larger), and
#: far above double-precision noise, so an exact abutment cannot come back as a
#: sliver overlap.
_CONTACT_REL_EPS = 1e-12

#: Below this, an edge cross-product is numerically meaningless — the two boxes
#: are near-parallel about that pair and the axis carries no information. Such
#: an axis is SKIPPED, never allowed to report a separation, because the noise
#: in it is larger than the separation it would claim.
_DEGENERATE_AXIS = 1e-12


def obbs_are_disjoint(one: Obb, two: Obb) -> bool:
    """Exact separating-axis test between two oriented boxes.

    ``True`` iff some axis separates them, which for convex bodies means they
    provably do not intersect. The fifteen candidate axes — three faces of each
    box plus the nine edge cross-products — are a COMPLETE set for boxes, so
    ``False`` really does mean they overlap, not "no separating axis found".

    Contact counts as separated; see :data:`_CONTACT_REL_EPS`.
    """
    ca, aa, ea = one
    cb, ab, eb = two
    t = cb - ca
    r = aa @ ab.T                                   # b's axes in a's frame
    abs_r = np.abs(r)
    ta = aa @ t                                     # translation in a's frame
    eps = _CONTACT_REL_EPS * (float(ea.sum()) + float(eb.sum()))

    for i in range(3):                              # a's three face normals
        if abs(ta[i]) > ea[i] + float(eb @ abs_r[i]) - eps:
            return True
    tb = ab @ t
    for j in range(3):                              # b's three face normals
        if abs(tb[j]) > eb[j] + float(ea @ abs_r[:, j]) - eps:
            return True
    # The nine edge pairs. Note the epsilon flips sign here: a face axis is
    # UNIT length, so `sum - eps` is a real distance and contact-as-separated
    # is meaningful on it. An edge axis is the raw cross product, whose length
    # varies with the angle between the boxes, so the same slack would mean a
    # different distance on every pair. These axes are therefore held to the
    # strict test — separate only when clearly separated — which can only ever
    # KEEP a carve. Nothing is lost: two boxes in face contact are already
    # separated by a face axis above, and that is the case this exists for.
    for i in range(3):
        for j in range(3):
            i1, i2 = (i + 1) % 3, (i + 2) % 3
            j1, j2 = (j + 1) % 3, (j + 2) % 3
            axis = np.cross(aa[i], ab[j])
            if float(axis @ axis) < _DEGENERATE_AXIS:
                continue
            ra = ea[i1] * abs_r[i2, j] + ea[i2] * abs_r[i1, j]
            rb = eb[j1] * abs_r[i, j2] + eb[j2] * abs_r[i, j1]
            sep = abs(ta[i2] * r[i1, j] - ta[i1] * r[i2, j])
            if sep > ra + rb + eps:
                return True
    return False


def _mesh_aabb(mesh) -> Optional[Aabb]:
    if not mesh.vertices:
        return None
    xs = [float(p.x) for p in mesh.vertices]
    ys = [float(p.y) for p in mesh.vertices]
    zs = [float(p.z) for p in mesh.vertices]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _material_thickness(ext, divisor: float = 1.0) -> float:
    """Extrusion depth from a sheet ``Material(thickness_mm=…)``, in the
    CALLER's units.

    ``Material.thickness_mm`` is NOT touched by ``normalize_project_to_meters``
    (only ``elem.thickness`` is), so the conversion belongs here — same as the
    generator's own ``_material_thickness_m``, which is fixed at
    :data:`MM_PER_METER` because the generator only ever runs in metres.
    Returning raw mm to the metres domain made a material-thickness Extrude
    tessellate 1000x too deep on the mesh-carve path; returning metres to the
    authoring domain made the same sheet report a 0.022 mm extent.
    """
    mat = getattr(ext, "material", None)
    t = getattr(mat, "thickness_mm", None) if mat is not None else None
    return float(t) / divisor if t else 0.0


# ---------------------------------------------------------------------------
# The public entry — one rule per element KIND
# ---------------------------------------------------------------------------


def _union(boxes) -> Optional[Aabb]:
    lo = hi = None
    for box in boxes:
        if box is None:
            continue
        (x0, y0, z0), (x1, y1, z1) = box
        if lo is None:
            lo, hi = [x0, y0, z0], [x1, y1, z1]
        else:
            lo = [min(lo[0], x0), min(lo[1], y0), min(lo[2], z0)]
            hi = [max(hi[0], x1), max(hi[1], y1), max(hi[2], z1)]
    return (tuple(lo), tuple(hi)) if lo is not None else None


def element_extent(elem, divisor: float = 1.0, _on_path=None) -> Optional[Aabb]:
    """Authoring-space extent of ANY element, or ``None`` when it has none.

    The one derivation behind ``BimElement.authored_aabb()``. Four kinds, and
    they need four different answers:

    * **geometry** (Box/Extrude/Sweep/Pipe/Revolve/Bar) -> :func:`padded_aabb`,
      except the two shapes whose real extent is TIGHTER than the carve
      trigger's bound: a contour ``Extrude`` (one-sided along its known
      normal) and a member-form ``Sweep`` (the ``Material(profile_mm=)``
      section resolved into the real rectangle, bounded directionally).
    * **mesh** -> its vertex hull, which is already exact
    * **container** (Wall/Slab/Roof/Column/Beam/Element/Space/Site) -> the
      union of each child's OWN extent, recursively
    * **opening** (Window/Door) -> ``None`` here. An opening carries a size
      and no position at all (v20.0.0); where it lands is derived from its
      host, which this module cannot see. ``authored_aabb()`` handles it.

    **Why the container case cannot just call** :func:`padded_aabb`:
    ``frames.element_aabb`` already recurses into ``_elements``, so
    ``padded_aabb(wall)`` returns the recursive POINT hull and then pads it by
    a per-type amount keyed on ``type(wall).__name__``. ``Wall`` is not
    ``Pipe``, so that pad is zero — and a wall containing a pipe reports the
    pipe's CENTRELINE as its extent. The pad has to be applied per child, at
    the leaf that knows its own reach, and unioned upward.

    Boolean operands are not included. ``.difference()`` REMOVES material, so
    counting a cutter would enlarge the host by the very volume being taken
    out of it; ``.union()`` operands genuinely add, and are a documented gap
    rather than an oversight (they are also unreachable from ``.parent``, so
    the two limits agree).

    ``divisor`` states the caller's unit domain and is threaded down every
    branch — see the module docstring. ``authored_aabb``/``world_aabb`` leave
    it at the authoring-mm default; the generator's tessellation position
    check runs in metres and passes :data:`MM_PER_METER`.

    **A containment cycle RAISES here** — :class:`ContainmentCycleError` —
    where the analysis walks merely prune. This is the module whose whole
    contract is that a loose or truncated number is worse than no number:
    ``Bounds`` refuses an empty container rather than report a degenerate box
    at the origin, and ``Bounds.exact`` exists so a bound is never presented
    as an extent. An element that contains itself has no finite extent, and
    the union of the finite prefix the walk happened to reach before turning
    back is exactly the plausible-looking wrong answer this module refuses to
    hand out. ``_on_path`` is threaded by the recursion and is not part of the
    API.
    """
    from lite_step.compiler.walkguard import ContainmentCycleError, WalkGuard

    memo = _EXTENT_MEMO.get()
    memo_key = None
    if memo is not None:
        memo_key = (id(elem), divisor)
        if memo_key in memo:
            return memo[memo_key]

    guard = _on_path if _on_path is not None else WalkGuard()
    if not guard.enter(elem):
        label = (getattr(elem, "_canonical_name", None)
                 or getattr(elem, "name", None) or getattr(elem, "id", None))
        raise ContainmentCycleError(
            f"{type(elem).__name__} '{label}' contains itself — a containment "
            f"cycle has no finite extent, so there is no box to report. The "
            f"compile error naming the whole cycle comes from "
            f"validate_project_report(); fix the cycle and query again."
        )
    try:
        box = _element_extent_inner(elem, divisor, guard)
    finally:
        guard.leave(elem)
    # Stored AFTER the guard releases, so a cycle raises on every query rather
    # than being answered from a half-built cache the first raise poisoned.
    if memo is not None:
        memo[memo_key] = box
    return box


def _element_extent_inner(elem, divisor: float, _on_path) -> Optional[Aabb]:
    """The body of :func:`element_extent`, past its cycle guard."""
    from lite_step.models import taxonomy as tx

    if tx.is_opening(elem):
        return None
    if tx.is_mesh(elem):
        return _mesh_aabb(elem)
    if type(elem).__name__ == "Extrude":
        # One-sided, because the extrusion direction IS known — see
        # ``extrude_offset``. The carve trigger keeps its symmetric pad.
        from lite_step.compiler.placement import _element_aabb
        base = _element_aabb(elem)
        if base is None:
            return None
        span = extrude_offset(elem, divisor)
        if span is None:
            return base
        (x0, y0, z0), (x1, y1, z1) = base
        lo = [x0 + min(span[0], 0.0), y0 + min(span[1], 0.0), z0 + min(span[2], 0.0)]
        hi = [x1 + max(span[0], 0.0), y1 + max(span[1], 0.0), z1 + max(span[2], 0.0)]
        return (tuple(lo), tuple(hi))
    if type(elem).__name__ == "Sweep" and not (getattr(elem, "profile", None) or []):
        # Member form: the section lives on ``Material(profile_mm=)``. Resolve
        # it into the real rectangle and take the DIRECTIONAL bound — the same
        # ``_sweep_path_aabb`` derivation, in the same ``segment_axes`` frame,
        # that a Sweep with an authored ``profile=`` already gets. Without it
        # the three spellings of one 45x195 member disagreed: the house-style
        # one answered a 200x200 isotropic envelope 200 mm LONGER than the
        # member, and an author butting joinery against that measurement is
        # 100 mm out — the same class of error §1.6 exists to stop.
        #
        # The carve TRIGGER deliberately keeps its isotropic pad: tightening
        # it changes which pairs carve, which is a separate and measurable
        # change. Same shape as the ``Extrude`` split two branches up — the
        # query is allowed to be tighter than the trigger, on purpose rather
        # than by neglect.
        prof = _member_section_profile(elem, divisor)
        if prof is not None and len(getattr(elem, "path", None) or []) >= 2:
            swept = _sweep_path_aabb(elem, prof)
            if swept is not None:
                return swept
    if tx.is_solid(elem):
        return padded_aabb(elem, divisor)

    children = getattr(elem, "_elements", None) or []
    if children:
        return _union(element_extent(c, divisor, _on_path) for c in children)
    # Annotations (ReferencePoint/GuideLine) bear points but no volume; the
    # bare point walk is the honest answer for them.
    from lite_step.compiler.frames import element_aabb
    return element_aabb(elem)


#: Which pads are TIGHT (the extent is the solid's real reach) and which are
#: merely a BOUND. Keyed by type name, deliberately beside the pad table in
#: :func:`padded_aabb` so the two are read together — a pad that changes
#: without its verdict changing is how a loose number starts being presented
#: as an exact one.
_INEXACT_REASON = {
    "Pipe":  "the radius pad is isotropic, so the two path ENDS overshoot by "
             "one radius (a pipe is flat-capped, not hemispherical)",
    "Bar":   "the diameter/2 pad is isotropic, so the two path ENDS overshoot "
             "by one radius",
    "Revolve": "the profile's radial maximum is applied isotropically, so the "
               "extent overshoots along the revolve AXIS",
}


def extent_is_exact(elem) -> bool:
    """True when :func:`element_extent` is the solid's real extent.

    ``False`` means the answer is a BOUND — never smaller than the solid,
    possibly larger. Surfaced as ``Bounds.exact`` rather than hidden, because
    a loose bound presented as an extent is a plausible wrong number, and
    those are the ones that cost an afternoon.

    Exact today: ``Box`` (the corners ARE the extent), ``Mesh`` (vertex hull),
    ``Extrude`` (one-sided along a known normal — see :func:`extrude_offset`),
    and a path-form ``Sweep`` with an explicit profile (``_sweep_path_aabb``
    builds a directional bound from the real per-segment section frame).

    Not exact: the three in :data:`_INEXACT_REASON`, plus EVERY member-form
    ``Sweep`` whose section comes from ``Material(profile_mm=)``.
    :func:`element_extent` now bounds those directionally, exactly like the
    explicit-``profile=`` form — but ``exact`` is ONE flag serving
    ``authored_aabb``, ``world_aabb`` AND ``obb``, and ``obb`` still reads the
    isotropic :func:`_radial_reach` (a 45x195 section reports a 200x200
    envelope; recorded follow-up). Flipping the flag on the strength of the
    AABB alone would print ``exact=True`` beside an OBB that is not, which is
    a worse lie than the conservative ``False``.
    """
    from lite_step.models import taxonomy as tx

    tname = type(elem).__name__
    if tname in _INEXACT_REASON:
        return False
    if tname == "Sweep":
        # Member form (no explicit profile): see the docstring — the AABB is
        # directional now, the OBB is not, and one flag covers both. Path form
        # with a profile gets the directional bound — tight on a SINGLE
        # segment, but a bend adds the generator's mitered-joint overshoot pad,
        # measured at ~4.7% of the span on a two-bend fixture. Bound, not
        # extent.
        if not getattr(elem, "profile", None):
            return False
        return len(getattr(elem, "path", None) or []) <= 2
    if tx.is_opening(elem):
        return False
    children = getattr(elem, "_elements", None) or []
    if children and not tx.is_solid(elem) and not tx.is_mesh(elem):
        return all(extent_is_exact(c) for c in children)
    return True


def inexact_reason(elem) -> Optional[str]:
    """Why :func:`extent_is_exact` said no, for an error or a ``--q`` dump."""
    tname = type(elem).__name__
    if tname in _INEXACT_REASON:
        return _INEXACT_REASON[tname]
    if tname == "Sweep":
        if not getattr(elem, "profile", None):
            return ("the section comes from Material(profile_mm=): the AABB "
                    "resolves it into the real rectangle, but obb() still pads "
                    "by the section half-diagonal isotropically, so an oriented "
                    "box overshoots ACROSS the member")
        if len(getattr(elem, "path", None) or []) > 2:
            return ("a multi-segment sweep carries the mitered-joint overshoot "
                    "pad at each bend, so the extent is a bound rather than "
                    "the swept surface")
    children = getattr(elem, "_elements", None) or []
    for child in children:
        reason = inexact_reason(child)
        if reason:
            return f"{type(child).__name__} child: {reason}"
    return None


# ---------------------------------------------------------------------------
# Orientation — the axes an OBB is expressed in
# ---------------------------------------------------------------------------


def _canonicalize(axis: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Flip ``axis`` to a canonical sign: toward +X, ties toward +Y, then +Z.

    Without this the SAME shape returns SWAPPED corners depending on how its
    contour was wound, because a Newell normal flips with the winding. The
    wall rule already solves it exactly this way (``frames._axes_from_extrude``
    canonicalizes toward +X "so offsets are winding-independent"); this is that
    convention applied to raw primitives rather than a second one.
    """
    x, y, z = axis
    eps = 1e-9
    if x < -eps or (abs(x) <= eps and (y < -eps or (abs(y) <= eps and z < -eps))):
        return (-x, -y, -z)
    return (x, y, z)


def _orthonormal_from(primary: Tuple[float, float, float]):
    """A right-handed (along, across, up) triple containing ``primary``.

    ``primary`` becomes ``along``; ``up`` is world +Z unless that is parallel
    to it, in which case +X stands in. Deterministic, so two runs of the same
    model produce the same frame.
    """
    ax = np.array(primary, dtype=float)
    n = np.linalg.norm(ax)
    if n < 1e-12:
        return None
    ax = ax / n
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ax, ref))) > 0.999:
        ref = np.array([1.0, 0.0, 0.0])
    across = np.cross(ref, ax)
    across /= np.linalg.norm(across)
    up = np.cross(ax, across)
    return (tuple(ax), tuple(across), tuple(up))


def authored_axes(elem):
    """``(axes, rule)`` — the frame an element's OBB is expressed in.

    Oriented to the element's OWN AUTHORED CONSTRUCTION DIRECTION, never a
    freshly fitted minimal box. That distinction is the whole design:

    * A minimal fit swaps ``along`` and ``up`` on a near-cube, and a 1 mm edit
      flips them back — so a helper reading ``point_at(100, 0, 0)`` silently
      changes which face it means.
    * An authored direction is stable under any edit that does not change how
      the shape was written.

    Returns world axes with ``rule="aabb"`` when there is genuinely no single
    direction — a multi-segment sweep, a mesh, a plain Box. That is the honest
    answer to "which way does this point", and ``Bounds.rule`` reports it so
    the author can tell an oriented box from an axis-aligned one.
    """
    tname = type(elem).__name__

    if tname == "Extrude":
        n = _extrude_normal(elem)
        if n is not None:
            across = _canonicalize(n)
            # Built the way ``frames._axes_from_extrude`` builds a wall frame,
            # and in the same ORDER: the normal is the through-thickness
            # direction (`across`), the run axis is the in-plane `up x normal`,
            # and `up` is world Z. Index order is (along, across, up)
            # everywhere, so `size.y` is always the thickness.
            up = (0.0, 0.0, 1.0)
            a = np.cross(np.array(up), np.array(across))
            if float(np.linalg.norm(a)) > 1e-9:
                along = tuple(a / np.linalg.norm(a))
                real_up = tuple(np.cross(np.array(along), np.array(across)))
                return (along, across, real_up), "extrude-normal"
            frame = _orthonormal_from(across)
            if frame is not None:
                primary, perp, third = frame
                return (perp, primary, third), "extrude-normal"

    if tname in ("Sweep", "Pipe", "Bar"):
        path = getattr(elem, "path", None) or []
        if len(path) == 2:
            d = (float(path[1].x) - float(path[0].x),
                 float(path[1].y) - float(path[0].y),
                 float(path[1].z) - float(path[0].z))
            axes = _orthonormal_from(_canonicalize(d))
            if axes is not None:
                return axes, "path-direction"
        # More than two points: several directions, no single frame. Say so
        # rather than picking the first segment and hoping.

    if tname == "Revolve":
        path = getattr(elem, "path", None) or []
        if len(path) == 2:
            d = (float(path[1].x) - float(path[0].x),
                 float(path[1].y) - float(path[0].y),
                 float(path[1].z) - float(path[0].z))
            axes = _orthonormal_from(_canonicalize(d))
            if axes is not None:
                # The revolve axis is `up` — the body is radially symmetric
                # about it, so calling it `along` would name the one direction
                # that carries no information.
                along, across, up = axes
                return (across, up, along), "revolve-axis"

    stamped = getattr(elem, "_frame", None)
    if stamped is not None and getattr(stamped, "rule", None) not in (None, "fallback"):
        return (stamped.along, stamped.across, stamped.up), stamped.rule

    from lite_step.models.bounds import WORLD_AXES
    return WORLD_AXES, "aabb"


def oriented_extent(elem, axes, rule: str, divisor: float = 1.0):
    """``(centre, size)`` of ``elem`` measured along ``axes``, or ``None``.

    Projects the AUTHORED GEOMETRY POINTS onto the frame and adds the solid's
    reach beyond them — the same two-part shape as the axis-aligned path
    (points, then the per-type pad), expressed in the frame's directions
    instead of the world's.

    It has to be the points, not the axis-aligned box: the AABB of a slanted
    Extrude already contains the slant, so projecting ITS corners onto the
    slanted axes yields a box LARGER than the AABB rather than the tight one
    the whole exercise is for.
    """
    from lite_step.compiler.frames import iter_geometry_points

    pts = [(float(p.x), float(p.y), float(p.z))
           for p in iter_geometry_points(elem)]
    if not pts:
        return None
    P = np.array(pts)
    spans = []
    for a in axes:
        proj = P @ np.array(a, dtype=float)
        spans.append([float(proj.min()), float(proj.max())])

    tname = type(elem).__name__
    if rule == "extrude-normal":
        # axes are (along, across=normal, up). The contour is planar, so its
        # projection onto `across` is a single value and the whole thickness
        # comes from the extrusion — one-sided, along +normal.
        span = extrude_offset(elem, divisor)
        if span is not None:
            depth = float(np.dot(np.array(span), np.array(axes[1], dtype=float)))
            lo, hi = spans[1]
            spans[1] = [min(lo, lo + depth), max(hi, hi + depth)]
    elif rule == "path-direction":
        # axes are (along=path, across, up); the section fills the other two.
        r = _radial_reach(elem, divisor)
        for i in (1, 2):
            spans[i][0] -= r
            spans[i][1] += r
    elif rule == "revolve-axis":
        # axes are (across, along, up=axis) — index 2 IS the axis and must not
        # be padded; the profile sweeps a disc in the other two.
        r = _radial_reach(elem, divisor)
        for i in (0, 1):
            spans[i][0] -= r
            spans[i][1] += r

    centre = np.zeros(3)
    size = []
    for a, (lo, hi) in zip(axes, spans):
        centre = centre + np.array(a, dtype=float) * ((lo + hi) / 2.0)
        size.append(hi - lo)
    return tuple(float(c) for c in centre), tuple(size)


def _radial_reach(elem, divisor: float = 1.0) -> float:
    """How far the solid reaches perpendicular to its path/axis, in the
    CALLER's units (``divisor`` — see the module docstring)."""
    tname = type(elem).__name__
    if tname == "Pipe":
        return float(getattr(elem, "radius", 0) or 0)
    if tname == "Bar":
        return float(getattr(elem, "diameter", 0) or 0) / 2.0
    prof = getattr(elem, "profile", None) or []
    if prof:
        if tname == "Revolve":
            # A Revolve profile is (radius, height-along-axis), so the radial
            # reach is max(x) — exactly what the generator uses to bound it
            # (`max_r = max(p.x for p in elem.profile)`). The displacement pad
            # uses the full 2D norm instead, which over-approximates: safe for
            # a carve trigger, wrong as an extent.
            return max(abs(float(p.x)) for p in prof)
        return max((float(p.x) ** 2 + float(p.y) ** 2) ** 0.5 for p in prof)
    return _member_section_reach(elem, divisor)
