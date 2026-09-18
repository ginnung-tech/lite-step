"""Distribute a container's ``.clip()`` half-spaces to the parts under it.

``.clip()`` was written for a single solid. A ``Slab`` holding
``Box.union(Revolve)`` plus 212 ``Bar`` s is one ELEMENT but many SOLIDS, and
both spellings of the cut got it wrong without saying so:

* ``slab.clip(...)`` appended to a ``_clips`` list nothing downstream reads —
  byte-identical IFC, no warning, no half-space in the file (#807,
  reproduced: container 0 ``IfcHalfSpaceSolid``, primitive 1);
* ``body.clip(...)`` cut the concrete and left every bar where it was —
  measured at **1007 m of 6941 m (14.5%)** of a hospital arm's reinforcement
  standing outside its own mitered sector, one bar entirely outside.

This pass makes the container spelling the working one. Two properties are
load-bearing and neither is free.

**The distribution is DEFERRED to the end of authoring.** The cut reaches the
children that exist when this pass runs, so parts added after the ``.clip()``
line are still cut. Resolving eagerly would reintroduce the trap this project
has already paid for twice: ``Product(source)`` snapshots at construction, so a
a verb called afterwards silently did nothing — five ground plates kept
carving because of it. A cut that depends on where in the file it was written
is a cut nobody can read off the source.

**A path is TRIMMED, not subtracted**, and that asymmetry is the whole reason
this is worth doing rather than just looping the boolean over the children.
See ``trim_path``.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Child lists a container can hold geometry under. ``_cuts``/``_adds``/
# ``_intersects``/``_fills``/``_voids`` are boolean OPERANDS — they are merged
# into their consumer's shape, so they are reached through the consumer and
# must never be trimmed as parts in their own right. Trimming a cutter
# separately would shrink the HOLE, not the solid.
_CHILD_ATTRS = ("_elements",)


class CompositeClipError(ValueError):
    """A clip could not be applied to anything.

    Its own error type because the failure is AUTHORED, not internal: the
    receiver had no geometry under it at the end of compile, so the call could
    not act. Without this pass it returns ``self`` in silence.
    """


# ---------------------------------------------------------------------------
# Subtree walk
# ---------------------------------------------------------------------------


def _geometry_bearing(elem) -> bool:
    """Does this element carry tessellable geometry of its own?

    ``tessellation_kind`` is the same predicate the geometry queries and the
    displacement engine dispatch on, so "has a body" means ONE thing across the
    compiler rather than three nearly-identical lists free to drift (the
    ``grids.py`` / ``voids.py`` precedent).
    """
    from lite_step.models import taxonomy as tx

    return tx.tessellation_kind(elem) is not None


def iter_parts(elem) -> Iterator:
    """Yield every geometry-bearing descendant of ``elem``, depth-first.

    ``elem`` itself is included when it carries geometry — a ``Wall`` with a
    body AND children is both, and the body must be cut like any other part.
    """
    if _geometry_bearing(elem):
        yield elem
    for attr in _CHILD_ATTRS:
        for child in (getattr(elem, attr, None) or []):
            yield from iter_parts(child)


# ---------------------------------------------------------------------------
# Path trim — the reason bars stay bars
# ---------------------------------------------------------------------------


def _unit(normal) -> Tuple[float, float, float]:
    """``normal`` as a unit vector.

    ``.clip()`` accepts any non-zero direction and the boolean does not care
    about its length — but ``overrun`` is a LENGTH in the same domain as the
    model, so the signed distance it is compared against has to be a real
    distance. With an unnormalized ``(2,0,0)`` an overrun of 300 mm would trim
    at 150.
    """
    m = (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2) ** 0.5
    return (normal[0] / m, normal[1] / m, normal[2] / m)


def _signed(pt, origin, normal) -> float:
    """Distance from the plane, positive on the REMOVED side.

    ``HalfSpace.normal`` points at the side that goes, matching ``.clip()``'s
    documented convention, so "keep" is ``<= 0`` (``<= overrun`` once a lap is
    asked for). ``normal`` must already be a unit vector — see ``_unit``.
    """
    return ((pt.x - origin.x) * normal[0]
            + (pt.y - origin.y) * normal[1]
            + (pt.z - origin.z) * normal[2])


def _lerp(p, q, t):
    from lite_step.models import Point

    return Point.model_construct(
        x=p.x + (q.x - p.x) * t,
        y=p.y + (q.y - p.y) * t,
        z=p.z + (q.z - p.z) * t,
    )


def trim_path(path: List, origin, normal, *, overrun: float = 0.0,
              eps: float = 1e-9):
    """Shorten a polyline to the kept side of the half-space.

    Returns the trimmed path, or ``None`` when nothing survives.

    A **trim, not a boolean**, for three reasons that all showed up in
    measurement: it costs 0 CSG depth instead of one boolean per bar (the
    hospital slab went to CSG 12 — over budget — with the bars subtracted, and
    the viewer silently renders the UNCARVED solid past ~13); it keeps a ``Bar``
    a ``Bar`` for schedules and quantities; and it sidesteps,
    where subtracting from an ``IfcSweptDiskSolid`` makes ifcopenshell trim the
    directrix and report the bar truncated at an unchanged vertex count.

    ``overrun`` offsets the keep test along the normal, so a path stops at
    ``plane + overrun``. **The trim can only ever SHORTEN**: every emitted
    point is either one of the path's own or interpolated BETWEEN two of them,
    so a bar that already ends short of the lap line is returned untouched
    rather than stretched to reach it. That is why the acceptance criterion
    reads *at most* ``overrun`` past the plane, and *exactly* ``overrun`` only
    for a path with the length to get there — extending a short bar would be
    inventing steel that is not in the schedule.

    Only the segment that CROSSES the offset plane gains a point; a path lying
    wholly on the kept side is returned unchanged (the same list object), so an
    untouched bar stays byte-identical rather than being rebuilt from
    arithmetic.
    """
    if not path or len(path) < 2:
        return path
    normal = _unit(normal)
    keep = overrun
    dists = [_signed(p, origin, normal) for p in path]
    if all(d <= keep + eps for d in dists):
        return path                       # wholly kept — do not rebuild it
    if all(d >= keep - eps for d in dists):
        return None                       # wholly removed

    out: List = []
    for i, (p, d) in enumerate(zip(path, dists)):
        if d <= keep + eps:
            out.append(p)
        if i + 1 < len(path):
            d_next = dists[i + 1]
            # Strict side change only: a vertex sitting ON the plane is already
            # emitted by the branch above, and interpolating it again would
            # duplicate a point and give a zero-length segment.
            if (d > keep + eps) != (d_next > keep + eps):
                span = d - d_next
                if abs(span) > eps:
                    out.append(_lerp(p, path[i + 1], (d - keep) / span))
    if len(out) < 2:
        return None
    return out


# ---------------------------------------------------------------------------
# Applying one half-space to one part
# ---------------------------------------------------------------------------


def _path_attr(part) -> Optional[str]:
    """``"path"`` for the linear primitives, else ``None``.

    Duck-typed on the field rather than on a class list: ``Bar``, ``Pipe`` and
    a path-form ``Sweep`` are exactly the elements that carry a directrix, and a
    class list here would be a second copy of that fact, free to drift from the
    models the day a fourth one is added.
    """
    path = getattr(part, "path", None)
    if isinstance(path, list) and len(path) >= 2:
        return "path"
    return None


def _apply_to_part(part, half_space) -> str:
    """Cut one part. Returns ``"kept"``, ``"trimmed"`` or ``"dropped"``."""
    origin, normal = half_space.origin, half_space.normal
    attr = _path_attr(part)
    if attr is not None:
        before = getattr(part, attr)
        after = trim_path(before, origin, normal,
                          overrun=getattr(half_space, "overrun", 0.0) or 0.0)
        if after is None:
            return "dropped"
        if after is before:
            return "kept"
        object.__setattr__(part, attr, after)
        return "trimmed"
    # Everything else takes the half-space as a boolean, which is what
    # ``.clip()`` has always meant for a single solid — FLUSH at the plane,
    # ``overrun`` deliberately ignored. See HalfSpace's docstring.
    part._clips.append(half_space)
    return "trimmed"


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def _distribute(elem, half_spaces, *, label: str) -> Tuple[int, int, int]:
    """Apply every half-space to every part under ``elem``.

    Returns ``(parts, dropped, trimmed)``. Dropped parts are removed from their
    parent's child list — a bar trimmed away to nothing must LEAVE, because
    emitting a zero-length swept disk is the silent failure this corpus keeps
    paying for.
    """
    parts = list(iter_parts(elem))
    if not parts:
        raise CompositeClipError(
            f"{label}: nothing to cut — {type(elem).__name__} has no "
            f"geometry-bearing part under it at the end of compile, so the "
            f".clip() call cannot act. Clip the element that carries the "
            f"body, or add the parts before compiling."
        )

    dropped_ids = set()
    trimmed = 0
    for part in parts:
        outcome = "kept"
        for hs in half_spaces:
            r = _apply_to_part(part, hs)
            if r == "dropped":
                outcome = "dropped"
                break
            if r == "trimmed":
                outcome = "trimmed"
        if outcome == "dropped":
            dropped_ids.add(id(part))
        elif outcome == "trimmed":
            trimmed += 1

    if dropped_ids:
        _prune(elem, dropped_ids)
    return len(parts), len(dropped_ids), trimmed


def _prune(elem, dropped_ids) -> None:
    """Remove dropped parts from the tree."""
    for attr in _CHILD_ATTRS:
        children = getattr(elem, attr, None)
        if not children:
            continue
        children[:] = [c for c in children if id(c) not in dropped_ids]
        for child in children:
            _prune(child, dropped_ids)


def resolve(project) -> None:
    """Derive every recorded miter, then distribute every container clip.
    Idempotent.

    Runs after the mm -> m normalisation (so plane origins, overruns and paths
    are all in the same domain — a plane left in mm misses its own element by
    1000x and removes nothing, which reads as "the clip silently did not
    apply") and before frames are stamped (frames are derived from geometry,
    and this changes geometry).

    **Two passes, and the split is load-bearing.** A miter's angle is derived
    from BOTH sides' footprints, and distributing a clip TRIMS geometry — so
    resolving one element's joint after its partner had already been trimmed
    would read a footprint the author never authored, and which of the two got
    the pristine reading would come down to tree order. Deriving every joint
    first, off untouched geometry, is what makes the answer the same for both
    sides of every joint.
    """
    for elem in _iter_roots(project):
        derive_pending_miters(elem)
    for elem in _iter_roots(project):
        _resolve_subtree(elem)


def derive_pending_miters(elem) -> None:
    """Pass 1: turn every recorded joint under ``elem`` into a half-space.

    Public because it is the whole of what ``miter()`` defers, and a test
    that wants to read a derived plane should not have to run a compile to
    get one. Scale-free: ``at`` and the geometry only have to agree with each
    other, so this answers the same normal in the authored mm domain as in
    metres.
    """
    pending = getattr(elem, "_pending_miters", None)
    if pending:
        from lite_step.models.elements import derive_miter_pair

        for req in list(pending):
            partner = req.partner
            own, theirs = derive_miter_pair(elem, req)
            elem._clips.append(own)
            partner._clips.append(theirs)
            # Retire the partner's half of this joint NOW. Leaving it would let
            # the walk derive the same plane a second time, and by then this
            # side is clipped — which is exactly the contamination the single
            # derivation exists to remove. Matched on the shared joint id, so
            # two joints between the same pair stay distinct.
            _retire(partner, req.jid)
        pending.clear()          # consumed — idempotence lives here
    for attr in _CHILD_ATTRS:
        for child in list(getattr(elem, attr, None) or []):
            derive_pending_miters(child)


def _retire(elem, jid) -> None:
    pending = getattr(elem, "_pending_miters", None)
    if not pending:
        return
    pending[:] = [r for r in pending if getattr(r, "jid", None) != jid]


def _iter_roots(project) -> Iterator:
    for storey in getattr(project, "storeys", []) or []:
        for elem in getattr(storey, "elements", []) or []:
            yield elem
    for site in getattr(project, "sites", []) or []:
        yield site


def _resolve_subtree(elem) -> None:
    # Depth-first: an inner container resolves its own clips before an outer
    # one distributes over it, so a part is never cut twice by the same plane.
    for attr in _CHILD_ATTRS:
        for child in list(getattr(elem, attr, None) or []):
            _resolve_subtree(child)

    # A geometry-bearing element consumes its own clips through the generator,
    # exactly as before — this pass must not touch that path. Only a CONTAINER
    # needs distributing.
    if _geometry_bearing(elem):
        return
    clips = getattr(elem, "_clips", None) or []
    if not clips:
        return
    label = f"clip[{type(elem).__name__}:{getattr(elem, 'name', None) or '?'}]"
    # Read the overruns BEFORE distributing — the list is consumed below, and
    # a report derived from state the pass has already cleared reads zero.
    overruns = sorted({int(round(getattr(hs, "overrun", 0.0) * 1000.0))
                       for hs in clips})
    n, dropped, trimmed = _distribute(elem, list(clips), label=label)
    clips.clear()          # consumed — the parts carry it now
    _report(label, n, dropped, trimmed, overruns)


def _report(label, parts, dropped, trimmed, overruns) -> None:
    """One line on stderr per container clip, beside ``carves:`` and
    ``csg depth:`` — same channel, same compile, for the same reason.

    Unconditional, with no ``LITESTEP_*`` kill-switch of its own: this report's
    volume is bounded by the number of container clips in the model (zero
    across all 20 corpus models today), so it cannot become the 12 KB of
    stderr that made the carve and tree reports need one.

    **``overrun`` is stated even at 0.** Flush IS a construction joint — a real
    choice with real consequences for the reinforcement crossing it — so it
    goes on the record as a decision rather than staying the value nobody
    picked. That is also the only way a reader can tell a lap that was asked
    for and applied from one that was asked for and silently dropped.
    """
    import sys

    lap = "flush" if overruns == [0] else \
        "overrun=" + "/".join(f"{o}mm" for o in overruns)
    print(f"{label}: {parts} parts, {dropped} dropped entirely, "
          f"{trimmed} trimmed, {lap}", file=sys.stderr)
    logger.info("%s: %d parts, %d dropped, %d trimmed, %s",
                label, parts, dropped, trimmed, lap)
