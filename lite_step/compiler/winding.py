"""Catch a vertical Extrude whose winding order swept it somewhere else.

# THE ONE BRANCH WITH NO GUARD

``apply_orientation_rules`` canonicalises the normal Newell's method returns::

    if abs(nz) > 0.001:      # sloped or horizontal
        if nz < 0: flip      # -> always points up
    return normal            # vertical: trust the author's winding

That last line is deliberate and correct — for a vertical face there is no
"up" to canonicalise towards, so which side the solid grows on is genuinely
the author's choice. It is also the only branch where getting the winding
backwards has a geometric consequence, and the consequence is invisible:
``Extrude`` sweeps ``thickness`` along the normal, so a reversed contour
sweeps the solid a full thickness the OTHER WAY. The model compiles,
validates and renders. It is simply somewhere else.

Measured on a real failing build: two roof contours
authored at ``y=-2800`` with ``thickness=7600`` landed at ``y -10400..-2800``,
clear of the house. Reversed they occupy ``y -2800..4800``, sitting on the
walls. Nothing in the pipeline said a word.

# WHY THIS IS A WARNING AND NOT AN ERROR

House rule: degenerate-but-well-formed warns; only a question the compiler
cannot answer correctly raises. A detached solid is legal — a shed across the
yard is a detached solid — so the compiler must build it. It is just almost
never what was meant.

# THE SIGNATURE, AND WHY IT DOES NOT CRY WOLF

Not "is this solid detached", which would fire on every legitimately separate
element. The test is a COMPARISON:

    as authored, it touches nothing  AND  reversed, it touches something

A genuinely detached element touches nothing either way, so it stays quiet.
Only a solid that would land on the building if its winding were flipped is
reported — which is exactly the mistake, and nothing else looks like it.

Validated against the corpus before landing: flags all three suspect solids
in the failing build, and stays silent on all 34 reference examples.

# WHAT IT DOES NOT CATCH, DELIBERATELY

A solid swept the wrong way out of a face it is EXACTLY FLUSH with still
abuts the model, and this check cannot tell that from resting on it. That is
not an oversight to fix by shrinking the boxes: the same shape, correctly
authored, is a cladding panel swept outward off a wall face — flush behind,
clear in front — and a tolerance tight enough to catch the first flags every
one of the second. The real defect has a gap (a roof overhangs its walls),
which is why the measurable case is the one worth reporting.
"""

from __future__ import annotations

from lite_step.ifc.geometry import apply_orientation_rules

#: Matches ``apply_orientation_rules``' own epsilon. A contour is "vertical"
#: exactly when that function declines to canonicalise it — the two must agree
#: or this check reasons about a different set of solids than the compiler.
VERTICAL_EPSILON = 0.001


def newell(pts):
    """The unit surface normal of a polygon, by Newell's method.

    Local rather than imported so this check reads the winding the same way
    the generator does even if the generator's helper moves.
    """
    nx = ny = nz = 0.0
    n = len(pts)
    for i in range(n):
        x0, y0, z0 = pts[i]
        x1, y1, z1 = pts[(i + 1) % n]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    m = (nx * nx + ny * ny + nz * nz) ** 0.5
    return (nx / m, ny / m, nz / m) if m else (0.0, 0.0, 0.0)


def swept_aabb(pts, thickness):
    """Where the solid actually lands: the contour hull plus thickness*normal.

    Goes through ``apply_orientation_rules`` rather than the raw Newell normal
    on purpose — for a sloped contour the compiler flips and this must model
    the flip, or the check would report a face the compiler has already fixed.
    """
    n = apply_orientation_rules(newell(pts))
    off = [c * thickness for c in n]
    xs = [p[0] for p in pts] + [p[0] + off[0] for p in pts]
    ys = [p[1] for p in pts] + [p[1] + off[1] for p in pts]
    zs = [p[2] for p in pts] + [p[2] + off[2] for p in pts]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def overlaps(a, b):
    """Do two axis-aligned boxes share any volume, touching included?"""
    return all(a[i] <= b[i + 3] and b[i] <= a[i + 3] for i in range(3))


def _is_vertical(pts):
    return abs(newell(pts)[2]) <= VERTICAL_EPSILON


def _points(contour):
    return [(float(p.x), float(p.y), float(p.z)) for p in contour]


def _iter_geometry(project):
    """Yield every LEAF geometry element in the project, once.

    Leaves only. A container's own AABB encloses its children, so including
    containers would make every extrude "touch" its own parent and the check
    would never fire — the bug this is looking for would be reported as fine.
    """
    seen = set()

    def walk(elem):
        if elem is None or id(elem) in seen:
            return
        seen.add(id(elem))
        children = list(getattr(elem, "_elements", None) or [])
        for slot in ("_openings", "_cuts", "_adds", "_intersects", "_fills", "_voids"):
            children.extend(getattr(elem, slot, None) or [])
        if not children:
            yield elem
        for child in children:
            yield from walk(child)

    for storey in getattr(project, "storeys", []) or []:
        for elem in getattr(storey, "elements", []) or []:
            yield from walk(elem)
    for site in getattr(project, "sites", []) or []:
        yield from walk(site)


def _aabb(elem):
    try:
        b = elem.authored_aabb()
    except Exception:
        # An element whose extent cannot be computed contributes no evidence.
        # Silently skipped: it is not this check's job to report it, and
        # something that genuinely cannot be placed fails elsewhere, loudly.
        return None
    if b is None:
        return None
    try:
        return (b.min.x, b.min.y, b.min.z, b.max.x, b.max.y, b.max.z)
    except AttributeError:
        return None


def _vertical_extrude(elem):
    """``(points, thickness)`` if this is a vertical Extrude, else ``None``."""
    if type(elem).__name__ != "Extrude":
        return None
    contour = getattr(elem, "contour", None)
    thickness = getattr(elem, "thickness", None)
    if not contour or not thickness:
        return None
    pts = _points(contour)
    return (pts, thickness) if _is_vertical(pts) else None


def _label(elem):
    for attr in ("ifc_name", "name", "id"):
        v = getattr(elem, attr, None)
        if v:
            return str(v)
    return type(elem).__name__


def check_extrude_zero_thickness(project):
    """Report Extrudes swept to no depth at all.

    Two inputs land here and both are well-formed, so both WARN rather than
    raise — the compiler answers each of them deterministically, and a
    degenerate answer is the author's to correct, not the compiler's to
    refuse:

    * ``thickness=0`` — a solid of zero depth, exactly as asked.
    * ``thickness<0`` — the same. A thickness is a magnitude; the sign is
      discarded rather than sweeping the solid the other way. That is
      CORRECT: the direction of an extrusion is decided by the contour's
      winding order, and letting the sign decide it too would be a second,
      redundant control over the same thing — the kind of corrective helper
      this compiler deliberately does not grow.

    What is worth saying either way is that nothing was built. A zero-depth
    solid has no volume, cannot take a boolean cut and does not render, so
    the model comes back missing a face; an author who wrote ``-200``
    believed they were making something 200 thick, and silence here reads as
    the element having been dropped.

    An earlier version of this refused a negative thickness at construction.
    That is wrong twice over: it contradicts the rule that
degenerate-but-well-formed warns, and it turns a harmless useless solid
    into a lost build for anyone already emitting one.
    """
    out = []
    for elem in _iter_geometry(project):
        if type(elem).__name__ != "Extrude":
            continue
        if getattr(elem, "contour", None) is None:
            continue
        thickness = getattr(elem, "thickness", None)
        if thickness is None or thickness > 0:
            continue

        if thickness < 0:
            detail = (
                f"has thickness={thickness}. A thickness is a magnitude, so the "
                f"sign is discarded and this sweeps to a solid of ZERO depth "
                f"rather than growing the other way — to grow the other way, "
                f"reverse the contour point order, which is what decides an "
                f"extrusion's direction (on a VERTICAL contour; a sloped or "
                f"horizontal one is canonicalised to point up)"
            )
        else:
            detail = "has thickness=0, so it sweeps to a solid of zero depth"

        out.append(
            f"Extrude '{_label(elem)}' {detail}: it has no volume, cannot "
            f"receive a boolean cut, and does not render."
        )
    return out


def check_extrude_winding(project):
    """Report vertical Extrudes whose winding swept them clear of the model.

    Returns warning strings for :class:`ValidationReport`, one per suspect
    solid, each naming where it landed and where it would land reversed.
    Numbers, not adjectives: the author has to be able to check the claim.
    """
    leaves = list(_iter_geometry(project))

    # THE EVIDENCE SET, AND WHY IT EXCLUDES THE THING BEING TESTED.
    #
    # "Touches nothing" has to mean "touches nothing we trust the position
    # of", and a vertical Extrude's position is precisely what is in
    # question — so one cannot vouch for another. Measured on the failing
    # build: BOTH roof faces were swept the wrong way, so each landed on the
    # other, each "touched something", and a check that counted them as
    # evidence reported the model clean. A whole group misplaced together is
    # the common case, not the corner case.
    #
    # Everything else IS evidence. A sloped or horizontal contour is
    # canonicalised by apply_orientation_rules, and a Box, Wall or Slab has
    # no winding to get wrong, so their positions are not in doubt.
    suspects = {id(e) for e in leaves if _vertical_extrude(e)}
    evidence = []
    for elem in leaves:
        if id(elem) in suspects:
            continue
        box = _aabb(elem)
        if box is not None:
            evidence.append(box)
    if not evidence:
        # Nothing trustworthy to measure against. A project made only of
        # vertical extrudes cannot be judged this way, and guessing would be
        # worse than staying quiet.
        return []

    out = []
    for elem in leaves:
        ve = _vertical_extrude(elem)
        if ve is None:
            continue
        pts, thickness = ve

        here = swept_aabb(pts, thickness)
        there = swept_aabb(list(reversed(pts)), thickness)

        if any(overlaps(here, o) for o in evidence):
            continue  # it lands on the model as authored — nothing to say
        if not any(overlaps(there, o) for o in evidence):
            continue  # detached either way; a genuinely separate element

        out.append(
            f"Extrude '{_label(elem)}' has a VERTICAL contour and its winding "
            f"order sweeps it clear of the model: as authored it occupies "
            f"x {here[0]:,.0f}..{here[3]:,.0f}, y {here[1]:,.0f}..{here[4]:,.0f}, "
            f"z {here[2]:,.0f}..{here[5]:,.0f} mm and touches nothing. Reversing the "
            f"contour puts it at x {there[0]:,.0f}..{there[3]:,.0f}, "
            f"y {there[1]:,.0f}..{there[4]:,.0f}, z {there[2]:,.0f}..{there[5]:,.0f} mm, "
            f"where it meets the model. A vertical contour is the one case the "
            f"compiler does NOT canonicalise (apply_orientation_rules trusts your "
            f"winding), so the solid grows along the normal your point order "
            f"implies. If this element is meant to stand apart, ignore this."
        )
    return out
