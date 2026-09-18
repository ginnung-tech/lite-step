"""DECLARED SPACE BOUNDS (WS-F) — the derived room volume, as POLICY.

``Space(name="kitchen", bounds=["wall:north", "slab:floor", ...])`` names the
room's bounding elements; this module DERIVES the ``IfcSpace`` volume from
them. It is the inversion of WS-C's shape, where the author hand-computed the
room box and the compiler inferred the bounds back out of it — with WS-F every
relation in the language is *declared once and expanded by the compiler*, and
geometric inference survives only as displacement and as CHECKS
(:mod:`lite_step.ifc.space_boundaries` keeps inference for geometry-mode
spaces and runs the demoted contact engine as a checker on bounds-mode ones).

The ONE place the emitter reads, mirroring
:mod:`lite_step.ifc.space_boundaries`: every policy decision — how a bound
string resolves, what volume the bounds derive, which shapes are errors, what
the compile log says — is decided HERE. The backends only ever see a ``Space``
that already carries its derived body, through the same emission path an
authored body takes.

The volume derivation — one rule, no wall/slab special cases
------------------------------------------------------------

Let C = the per-axis MEDIAN of the resolved bounds' interval midpoints (see
:func:`_centre` for why not the union-AABB centroid). Every bound
contributes one interval per world axis:

* C **inside** the bound's span on that axis → the bound contributes its own
  span (a *projection*).
* C **outside** the span → the bound contributes the half-space from its
  nearest face toward C (an *inner face*).

Combining is **overlap-aware** (the canopy ruling — the
resolution of the spec tension PR pinned):

* A bound that positively interpenetrates the candidate envelope the OTHER
  bounds derive on their own — WITHOUT covering the candidate's full
  cross-section on any axis (that would make it an enclosure layer: a hung
  ceiling, a double wall's leaf) — is a protruding FEATURE: a chimney
  breast, wing wall, breakfast bar. Its whole contribution is dropped from
  the volume (faces AND projections — either interval is the same bite), so
  it "loses to the outermost": the room takes the faces behind it. Boundary
  still emitted, volume untouched, noted on the derived-volume log line —
  declared means declared; ``.difference()`` is the author's tool when the
  air should exclude it. Classification is LEAVE-ONE-OUT
  (:func:`_demoted_features`), which is what makes it order-independent,
  single-pass and trivially terminating — the iterative
  demote-and-re-derive fixed point this replaces depends on demotion order.
* Everything kept folds as a PURE INTERSECTION: projections intersect
  (smallest span wins — three walls of unequal length derive the smallest
  projected box) and same-side kept faces take the **innermost**. A
  detached bound — a canopy 10 m up, the outer leaf of a double wall — can
  therefore never extend the room past a nearer contributor; it simply ends
  up not touching the derived volume, and the checker in
  ``space_boundaries`` warns so by design.

A floor slab needs no special case: its top face closes z from below and its
footprint projects x/y, from the same centre rule. Invariant (absolute, with
its own sweep test): **adding a bound never grows the room** — the final
fold is an intersection of the kept intervals, a feature contributes
nothing, and the median centre cannot be dragged outside the cluster by a
minority of far bounds (which is what flips projections into
wrong-way faces and grow the canopy kitchen to its canopy).

The INTERIOR bound is a warning, not an error (the ruling)
---------------------------------------------------------------------

A bound lying wholly — or near-wholly, within tolerance — inside the derived
volume on all three axes does NOT raise. The shape is not necessarily a
typo'd canonical name resolving to furniture: a
**pilaster, chimney breast or wing wall authored FLUSH to its host wall's
inner face** — touching the room's boundary plane without crossing into the
wall — is wholly inside on all three axes and is a perfectly real bounding
element. Erroring punished correct authoring, and a rule an author must be
surprised by once to learn is a defect in the rule.

So it WARNS, names the bound, and quotes the flush-pilaster case in the
message so the author can tell instantly which of the two situations they
have. Treatment is exactly the protruding-feature path: contribution
dropped from the volume, boundary still emitted, volume untouched, listed
on the derived-volume log line.

The demotion happens BEFORE the final fold (that is what "contribution
dropped" means), so ``bounds=["wall:only"]`` still fails — but through the
HONEST path rather than a special case: its only contribution is demoted,
no axis is left with a face or a projection, and the half-open error fires.
:func:`derive_volume` therefore re-folds once when the interior pass demotes
a bound the leave-one-out classifier had kept. That re-fold terminates by
construction and cannot loop: demotion only ever SHRINKS the volume, and a
smaller volume can only make interior-ness harder, never newly true.

Errors — reserved for shapes that are almost certainly mistakes
---------------------------------------------------------------

Per the principle of least surprise (roadmap §5): declared intent is honored
at face value; derivations are simple, deterministic and printed; errors are
reserved for near-certain mistakes, never for unusual-but-intentional
declarations. Three, each naming the axis and the bound(s):

* **C within ``CONTACT_TOL``-scale tolerance of a contributing face** — the
  centre sits ON a bound's face, so whether that bound projects or closes
  is a coin flip. Reachable when the declared set is symmetric about a face.
* **An axis left half-open** (no face and no projection closes a side).
  **REACHABLE through** :func:`derive_volume` **since the interior ruling**,
  and it is the error a one-bound declaration now fails with: demoting every
  bound that bounds nothing can strip a side of its only closure. It was
  described as "largely DEFENSIVE" while the interior check raised first and
  got there before it could; that is not so, and the message names the
  dropped bounds and says why they contributed nothing.
* **An empty result** (contradictory bounds). Still DEFENSIVE completeness:
  every contributed interval contains C by construction (a bound below C
  closes the bottom, a bound above C closes the top, a bound containing C
  closes both), so an EMPTY kept intersection stays unreachable through
  :func:`derive_volume` — demotion only REMOVES constraints and cannot
  create emptiness. It is a real raise with a direct test on
  :func:`_combine_axis` rather than a silent impossibility, because the
  interval combiner is a policy seam and a caller feeding it hand-made
  contributions gets the loud answer.

Units
-----

:func:`derive_volume` and :func:`_combine_axis` are unit-agnostic like
:mod:`lite_step.compiler.extent` — they answer in the units of the boxes they
were given, against a ``tol`` the caller states. :func:`materialize_bound_spaces`
runs in the MODEL's units: the emitter calls it after
``normalize_project_to_meters`` (the same contract as
``space_boundaries.collect_boundaries``), so that is metres, the extent
derivation is told so via ``MM_PER_METER``, and the report line converts to
millimetres for printing.

:data:`CONTACT_TOL` is the same magnitude as ``space_boundaries.CONTACT_TOL``
and ``displacement._EPS``, for the same reason: authoring is int mm, so two
faces an author placed at the same coordinate divide to the same double and
any difference is float noise. 1e-6 m is a micron; the smallest authorable
distance is a thousand times larger.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from lite_step.compiler.extent import MM_PER_METER, Aabb, overlap_depths

logger = logging.getLogger(__name__)

#: See the module docstring. Own constant rather than an import from
#: ``lite_step.ifc.space_boundaries`` because THAT module imports this one
#: (the collection fork), and a module-level cycle is a worse spelling than a
#: duplicated one-liner whose value is pinned by test to agree.
CONTACT_TOL = 1e-6

_AXIS_NAMES = ("x", "y", "z")


class SpaceVolumeError(ValueError):
    """A ``Space(bounds=[...])`` statement cannot be realised.

    Always raised (never warned): each of these shapes produces a plausible
    room silently otherwise — the exact class of failure declared bounds
    exist to remove. The INTERIOR bound left this set (see the
    module docstring): it is a warning, because a flush pilaster is a real
    bounding element and erroring punished correct authoring.
    """


@dataclass(frozen=True)
class DerivedVolume:
    """One bounds-mode space's derived volume, as POLICY rather than geometry.

    ``face_axes`` records, per bound, the axes on which it contributed an
    INNER FACE with the boundary sign ``space_boundaries.Contact`` uses
    (``+1`` = the bound sits past the volume's maximum on that axis). It is
    the classification fallback for a declared bound the derived volume does
    not geometrically touch — declared means declared, so its boundary is
    emitted regardless, and the sign says which way "beyond" points.

    ``features`` lists the bounds that poke INTO the derived volume (partial
    overlap): boundary emitted, volume untouched, noted on the report line.

    ``interior`` lists the bounds lying WHOLLY inside it — a flush pilaster
    or a mis-resolved name (see the module docstring). Same treatment as a
    feature, plus a warning: this is the one shape that can be either correct
    authoring or a typo, and only the author can tell which.
    """

    volume: Aabb
    face_axes: Dict[str, Tuple[Tuple[int, int], ...]]
    features: Tuple[str, ...]
    interior: Tuple[str, ...] = ()


@dataclass
class _AxisContribution:
    """Everything the bounds said about ONE world axis."""

    lower_faces: List[Tuple[str, float]] = field(default_factory=list)
    upper_faces: List[Tuple[str, float]] = field(default_factory=list)
    projections: List[Tuple[str, float, float]] = field(default_factory=list)

    def names(self) -> List[str]:
        return ([n for n, _ in self.lower_faces]
                + [n for n, _ in self.upper_faces]
                + [n for n, _lo, _hi in self.projections])


def _combine_axis(axis: int, contrib: _AxisContribution,
                  tol: float = CONTACT_TOL,
                  faces: str = "innermost",
                  dropped: Sequence[str] = ()) -> Tuple[float, float]:
    """One axis's derived interval from its contributions, or raise.

    * projections INTERSECT — the smallest span wins;
    * same-side faces combine per ``faces``:

      - ``"innermost"`` — the FINAL fold's rule (the canopy ruling,
): every kept face is an ordinary interval and the whole
        axis is a pure intersection, so a face can never extend the room
        past a nearer contributor — adding a bound never grows the room.
        Feature demotion (:func:`_demoted_features`) has already removed the
        faces this must not apply to (a chimney's inner face).
      - ``"outermost"`` — the CANDIDATE fold's rule: the largest face-closed
        box, the envelope a bound is classified against. Innermost here
        would let one protruding feature cap the candidate at its own face
        and hide a second, identical feature from the overlap test
        (measured: twin chimney breasts each classified KEPT against the
        candidate the other had truncated, and the pair truncated the room).

    ``dropped`` names the bounds whose contribution was demoted before this
    fold (features and interior bounds). It changes no arithmetic — it is
    carried only so the half-open message can say WHY an axis the author
    thought they closed has nothing on it. Without it the one-bound case
    reads "Contributing bounds on x: (none)" to an author who declared one,
    which is an internal invariant talking to itself.

    Raises the half-open and empty errors — see the module docstring for
    their reachability status; both are directly testable here.
    """
    axis_name = _AXIS_NAMES[axis]
    lo_candidates: List[float] = []
    hi_candidates: List[float] = []
    if contrib.lower_faces:
        values = [v for _n, v in contrib.lower_faces]
        # Lower faces close z >= f. Outermost = the SMALLEST closing value
        # (largest room); innermost = the largest (pure intersection).
        lo_candidates.append(min(values) if faces == "outermost"
                             else max(values))
    if contrib.upper_faces:
        values = [v for _n, v in contrib.upper_faces]
        hi_candidates.append(max(values) if faces == "outermost"
                             else min(values))
    for _name, p_lo, p_hi in contrib.projections:
        lo_candidates.append(p_lo)
        hi_candidates.append(p_hi)

    if not lo_candidates or not hi_candidates:
        side = "minimum" if not lo_candidates else "maximum"
        because = ""
        if dropped:
            because = (
                f" {', '.join(sorted(dropped))} contributed nothing: each one "
                f"lies inside the volume the OTHER declared bounds derive, so "
                f"it bounds nothing and its contribution was dropped. A SINGLE "
                f"declared bound is always in that position — one wall derives "
                f"only its own box, so it cannot enclose a room."
            )
        raise SpaceVolumeError(
            f"space bounds: the {axis_name} axis is left half-open — no inner "
            f"face and no projection closes its {side} side. Contributing "
            f"bounds on {axis_name}: {', '.join(contrib.names()) or '(none)'}."
            f"{because} Declare the elements that actually enclose the room: "
            f"at least two perpendicular walls close x and y, and a floor plus "
            f"a ceiling or deck close z.")

    lo = max(lo_candidates)
    hi = min(hi_candidates)
    if hi <= lo + tol:
        raise SpaceVolumeError(
            f"space bounds: the declared bounds contradict each other on the "
            f"{axis_name} axis — the derived interval is empty "
            f"({lo:.6g} .. {hi:.6g}). Contributing bounds: "
            f"{', '.join(contrib.names())}.")
    return lo, hi


def _median(values: Sequence[float]) -> float:
    """Deterministic median: middle value, or the mean of the two middles."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _centre(bounds: Sequence[Tuple[str, Aabb]]) -> List[float]:
    """C — the per-axis MEDIAN of the bounds' interval midpoints.

    Not the centroid of the union AABB, deliberately (the one deviation from
    the roadmap's mechanism sentence, forced by its own ruling — see the
    module docstring): a single far bound drags a union centroid past the
    other bounds' spans, their projections flip into faces pointing the
    wrong way, and no face-choice rule can recover the room (the canopy
    kitchen's 2700 ceiling is algebraically unreachable once the walls
    contribute ``z >= 2700``). A median cannot be moved outside the cluster
    by a minority of far bounds, so the sides the MAJORITY of the declared
    set describes stay assigned as the author reads them. The honest limit:
    a majority of far bounds redefines where the room is — at that point
    the declaration itself says the room is elsewhere, and the interior /
    does-not-touch machinery answers loudly.
    """
    return [_median([(b[0][k] + b[1][k]) / 2.0 for _n, b in bounds])
            for k in range(3)]


def _contributions(
    bounds: Sequence[Tuple[str, Aabb]], centre: Sequence[float], tol: float,
) -> Tuple[List[_AxisContribution], Dict[str, List[Tuple[int, int]]]]:
    """Per-axis contributions + per-bound inner-face axes, at ``centre``.

    Raises the ambiguous-side error when C sits on any bound's face — a
    degenerate declaration regardless of whether that face would win.
    """
    contribs = [_AxisContribution() for _ in range(3)]
    face_axes: Dict[str, List[Tuple[int, int]]] = {n: [] for n, _b in bounds}
    for name, (b_lo, b_hi) in bounds:
        for k in range(3):
            c = centre[k]
            if abs(c - b_lo[k]) <= tol or abs(c - b_hi[k]) <= tol:
                raise SpaceVolumeError(
                    f"space bounds: the centre of the declared bounds lies "
                    f"on a face of {name} on the {_AXIS_NAMES[k]} axis "
                    f"(within tolerance), so whether that bound projects its "
                    f"span or closes a side is ambiguous. Move or split the "
                    f"declared set so the centre falls clear of every "
                    f"bound's faces.")
            if b_lo[k] < c < b_hi[k]:
                contribs[k].projections.append((name, b_lo[k], b_hi[k]))
            elif c > b_hi[k]:
                # Bound below C: its top face closes the volume from below.
                contribs[k].lower_faces.append((name, b_hi[k]))
                face_axes[name].append((k, -1))
            else:
                contribs[k].upper_faces.append((name, b_lo[k]))
                face_axes[name].append((k, +1))
    return contribs, face_axes


def _fold(bounds: Sequence[Tuple[str, Aabb]], tol: float,
          faces: str) -> Aabb:
    """One straight derivation pass over ``bounds`` — no feature logic, no
    interior check. The final fold runs it over the KEPT set with
    ``faces="innermost"``; the candidate folds over leave-one-out sets with
    ``faces="outermost"``."""
    contribs, _ = _contributions(bounds, _centre(bounds), tol)
    spans = [_combine_axis(k, contribs[k], tol, faces=faces)
             for k in range(3)]
    return (tuple(s[0] for s in spans), tuple(s[1] for s in spans))


def _demoted_features(bounds: Sequence[Tuple[str, Aabb]],
                      tol: float) -> frozenset:
    """The bounds whose contribution is DROPPED from the final fold.

    The overlap-aware half of the canopy ruling, as LEAVE-ONE-OUT
    classification rather than the sketched iterative fixed point: bound
    ``i`` is a protruding FEATURE iff its body positively interpenetrates
    the candidate envelope the OTHER bounds derive on their own
    (outermost-face fold — see :func:`_combine_axis` for why). Leave-one-out
    is what makes the answer order-independent and single-pass: an iterative
    demote-and-re-derive loop's fixed point depends on the order bounds are
    demoted in, and a bound can never vote on its own classification here
    because its own faces are absent from the envelope it is judged against.
    Terminates by construction — N straight folds, no iteration.

    The WHOLE contribution drops, faces and projections both ("the volume
    derivation must not let such features bite"): a breakfast bar whose span
    contains the centre would otherwise truncate the room to its own strip
    through its projection, which is the same bite by another interval.

    Poking in is necessary but NOT sufficient — an ENCLOSURE LAYER also
    pokes. A ceiling hung below the wall tops interpenetrates the candidate
    (the walls' z projections cap it above the ceiling's underside), and so
    does the inner leaf of a double wall or a declared full-section
    bulkhead; demoting those would raise the room to the wall tops and erase
    the leaf (measured — the report-line fixture derived 2700 instead of its
    2600 ceiling on the first cut of this rule). What separates a chimney
    from a ceiling is CROSS-SECTION COVERAGE: an enclosure layer spans the
    candidate's full cross-section perpendicular to some axis (it closes the
    room there); a protruding feature covers no full cross-section on any
    axis. So: FEATURE iff positive interpenetration AND no axis on which the
    bound covers both perpendicular candidate spans. Consequence, deliberate
    (least surprise): declaring a full-cross-section partition inside the
    room means the room stops at it — a declared bulkhead IS a wall of the
    room, exactly the double-leaf logic applied from inside.

    A candidate that cannot derive (a leave-one-out subset raising
    half-open/ambiguous/empty) classifies the bound as KEPT — conservative,
    because kept faces combine innermost and can only shrink.
    """
    if len(bounds) < 2:
        return frozenset()
    out = set()
    for i, (name, box) in enumerate(bounds):
        rest = list(bounds[:i]) + list(bounds[i + 1:])
        try:
            candidate = _fold(rest, tol, faces="outermost")
        except SpaceVolumeError:
            continue
        if not all(d > tol for d in overlap_depths(candidate, box)):
            continue                     # flush or clear of the envelope
        (b_lo, b_hi) = box
        (c_lo, c_hi) = candidate
        encloses = any(
            all(b_lo[j] <= c_lo[j] + tol and b_hi[j] >= c_hi[j] - tol
                for j in range(3) if j != k)
            for k in range(3))
        if not encloses:
            out.add(name)
    return frozenset(out)


def derive_volume(bounds: Sequence[Tuple[str, Aabb]],
                  tol: float = CONTACT_TOL) -> DerivedVolume:
    """The derived volume of one space from its resolved bounds' world boxes.

    ``bounds`` is ``[(canonical_name, world Aabb), ...]`` in DECLARED order.
    Unit-agnostic — answers in the units of the boxes. Raises
    :class:`SpaceVolumeError` on every mistake shape the module docstring
    lists.

    Two passes, both deterministic (the canopy ruling —
    "innermost for detached", the never-grows invariant absolute):

    1. classify every bound against the leave-one-out candidate envelope
       (:func:`_demoted_features`) — a bound poking INTO what the rest
       declare is a feature and contributes nothing to the volume;
    2. fold the kept bounds' contributions at the median centre with
       innermost faces — a pure intersection, so a detached same-side face
       (a canopy, the outer leaf of a double wall) can never extend the
       room past a nearer contributor; it simply ends up not touching, and
       the checker in ``space_boundaries`` says so.

    Then the INTERIOR pass (the ruling — see the module
    docstring): a bound lying wholly inside the folded volume warns and is
    demoted too. When that demotes something pass 1 had kept, the fold runs
    once more over the smaller kept set — which is how ``bounds=["wall:only"]``
    reaches the half-open error instead of a special-cased interior raise.
    """
    if not bounds:                                        # pragma: no cover
        raise SpaceVolumeError("space bounds: no bounds to derive from")

    centre = _centre(bounds)
    # Ambiguity is checked over ALL bounds (features included) against the
    # final centre: C on any declared face is degenerate authoring, and the
    # classification below needs every bound's side assignment anyway.
    _, face_axes = _contributions(bounds, centre, tol)

    demoted = set(_demoted_features(bounds, tol))
    volume = _fold_kept(bounds, demoted, centre, tol)

    # The interior pass. Demotion only ever SHRINKS the volume, and a smaller
    # volume can only make "wholly inside" HARDER to satisfy, so one re-fold
    # is a fixed point — no iteration, no ordering question.
    interior = _interior_bounds(bounds, volume, tol)
    for name in interior:
        logger.warning(
            "space bounds: %s lies wholly inside the derived volume on all "
            "three axes, so it closes nothing and its contribution is dropped "
            "(its boundary is still emitted and the volume is untouched). "
            "That is CORRECT for a flush feature — a pilaster, chimney breast "
            "or wing wall authored flush to its host wall's inner face touches "
            "the room's boundary plane without crossing into the wall, and is "
            "wholly inside by construction. If you did not author such a "
            "feature, the fragment resolved to the wrong element (a typo'd "
            "canonical name landing on furniture) — check the name.", name)
    newly = [n for n in interior if n not in demoted]
    if newly:
        demoted.update(newly)
        volume = _fold_kept(bounds, demoted, centre, tol)

    features: List[str] = []
    for name, box in bounds:
        if name in interior:
            continue
        if all(d > tol for d in overlap_depths(volume, box)):
            # Positive interpenetration on every axis, and NOT wholly inside:
            # a chimney breast / wing wall / breakfast bar. A feature, never
            # a warning — declared means declared.
            features.append(name)

    return DerivedVolume(
        volume=volume,
        face_axes={n: tuple(v) for n, v in face_axes.items()},
        features=tuple(features),
        interior=tuple(interior),
    )


def _fold_kept(bounds: Sequence[Tuple[str, Aabb]], demoted, centre,
               tol: float) -> Aabb:
    """The final innermost-faces fold over ``bounds`` minus ``demoted``.

    Folded at the centre derived from ALL bounds (demotion does not move C —
    C is where the declaration says the room is), and told which names were
    dropped so a half-open axis can explain itself.
    """
    kept = [(n, b) for n, b in bounds if n not in demoted]
    contribs, _ = _contributions(kept, centre, tol)
    spans = [_combine_axis(k, contribs[k], tol, faces="innermost",
                           dropped=sorted(demoted))
             for k in range(3)]
    return (tuple(s[0] for s in spans), tuple(s[1] for s in spans))


def _interior_bounds(bounds: Sequence[Tuple[str, Aabb]], volume: Aabb,
                     tol: float) -> List[str]:
    """The bounds lying wholly (or near-wholly) inside ``volume``, in order."""
    (v_lo, v_hi) = volume
    return [name for name, (b_lo, b_hi) in bounds
            if all(b_lo[k] >= v_lo[k] - tol and b_hi[k] <= v_hi[k] + tol
                   for k in range(3))]


# ---------------------------------------------------------------------------
# Materialization — the compile pass the emitter shares
# ---------------------------------------------------------------------------


def report_line(space_leaf: str, volume: Aabb,
                features: Sequence[str] = (),
                interior: Sequence[str] = ()) -> str:
    """The compile log's derived-volume line, sizes in millimetres.

    Front-loaded observability, same spirit as ``carves: N pairs``: a derived
    volume nobody authored is invisible in the source, so the compile output
    is the only place a truncated or inflated room is visible. Assumes the
    metres domain (post-normalize — the only place materialization runs).

    ``features`` (partial overlap) and ``interior`` (wholly inside) are noted
    separately because they read differently to an author: one is a chimney
    poking in, the other is a flush pilaster — or a name that resolved to the
    wrong element. The interior list is ALSO warned (see
    :func:`derive_volume`); it appears here so the derived-volume line stays
    the one place a room's whole story is told.
    """
    (lo, hi) = volume
    dims = " x ".join(str(round((hi[k] - lo[k]) * MM_PER_METER))
                      for k in range(3))
    note = ""
    if features:
        note += (" (feature: " + ", ".join(features)
                 + " extends into the volume — boundary emitted, volume "
                   "untouched)")
    if interior:
        note += (" (inside: " + ", ".join(interior)
                 + " lies wholly within the volume and bounds nothing — "
                   "boundary emitted, volume untouched; see the warning)")
    return f"space {space_leaf}: derived {dims} mm{note}"


def _candidate_index(project) -> Dict[str, Tuple[Any, Aabb]]:
    """``{canonical_name: (element, world box)}`` for every boundable element.

    Exactly the candidate set ``space_boundaries._collect_elements`` walks —
    the top-level elements the backends turn into products — read through the
    same ``_world_box`` derivation, so a name that resolves here is a name
    the boundary emission can join. A second, subtly different walk is the shape.
    """
    from lite_step.ifc.space_boundaries import _matrices, _top_level, _world_box
    from lite_step.models import taxonomy as tx

    matrices = _matrices(project)
    index: Dict[str, Tuple[Any, Aabb]] = {}
    for elem in _top_level(project):
        if tx.is_spatial(elem) or tx.is_annotation(elem):
            continue
        name = getattr(elem, "_canonical_name", None)
        if not name:
            continue
        aabb = _world_box(elem, matrices)
        if aabb is None:
            continue
        index[name] = (elem, aabb)
    return index


def _resolve_bound(fragment: str, space_name: str,
                   index: Dict[str, Tuple[Any, Aabb]]) -> str:
    """One ``bounds=`` fragment -> its full canonical name, or raise.

    Resolution is ``naming.resolve_host`` — the reference grammar
    ``Anchor(host=)``, ``assert_carved`` and ``proj.aggregate()`` already use
    — so an author writes the same fragment everywhere.
    """
    from lite_step.compiler.naming import resolve_host

    status, matched, candidates = resolve_host(fragment, index.keys())
    if status == "ambiguous":
        raise SpaceVolumeError(
            f"space bounds: {space_name} declares {fragment!r}, which matches "
            f"several elements ({', '.join(candidates)}) — write more "
            f"leaf-first pairs to say which one. A room derived from the "
            f"wrong wall looks exactly like one derived from the right one.")
    if status == "miss":
        raise SpaceVolumeError(
            f"space bounds: {space_name} declares {fragment!r}, which names "
            f"no boundable element in this model. Bounds are canonical-name "
            f"fragments matched on whole ``type:leaf`` pairs from the tail, "
            f"exactly as Anchor(host=) matches, and must name a TOP-LEVEL "
            f"element with geometry (a wall standing in a storey, not a box "
            f"inside one, and never a Space).")
    return matched


def _cut_boxes(space) -> List[Aabb]:
    """Authoring-extent boxes of a space's ``.difference()`` operands.

    Read BEFORE the operands are moved onto the derived body, for the
    checker's effective volume. Placed (``placement=``) operands contribute
    their un-placed box — the same stated limit the displacement pass has —
    which only ever costs a checker WARNING, never geometry.
    """
    from lite_step.compiler.extent import element_extent

    out = []
    for cut in getattr(space, "_cuts", None) or []:
        box = element_extent(cut, MM_PER_METER)
        if box is not None:
            out.append(box)
    return out


def effective_volume(volume: Aabb, cuts: Sequence[Aabb],
                     tol: float = CONTACT_TOL) -> Aabb:
    """``volume`` with end-slices removed by ``cuts`` — the checker's box.

    An interval approximation of the boolean, not the boolean: a cut that
    covers the volume's full cross-section on two axes and includes one END
    of the third trims that end. That is the authored case the checker
    exists for ("a boolean carved the room away from its wall"); a cut that
    only nibbles an interior pocket leaves the AABB honest as it stands.
    Only ever feeds WARNINGS, never geometry — the emitted body carries the
    real CSG.
    """
    lo = list(volume[0])
    hi = list(volume[1])
    for (c_lo, c_hi) in cuts:
        for k in range(3):
            others = [j for j in range(3) if j != k]
            covers = all(c_lo[j] <= lo[j] + tol and c_hi[j] >= hi[j] - tol
                         for j in others)
            if not covers:
                continue
            if c_lo[k] <= lo[k] + tol and c_hi[k] > lo[k] + tol:
                lo[k] = min(c_hi[k], hi[k])
            if c_hi[k] >= hi[k] - tol and c_lo[k] < hi[k] - tol:
                hi[k] = max(c_lo[k], lo[k])
    return (tuple(lo), tuple(hi))


def _push_booleans_onto(body, container) -> None:
    """Move a Space's own boolean operands onto its representation body.

    Booleans on a Space are authored on the CONTAINER (`kitchen.difference(
    ...)`) — in bounds-mode there is no child to call them on — but the emitter realise booleans on the SOLID whose representation the IfcSpace
    receives. Moving the operands is what makes the composition identical to
    an authored body's. Applies to BOTH modes: before WS-F a geometry-mode
    ``space.difference(x)`` was silently dropped (operand consumed, no CSG
    emitted — measured), which is the silent-degradation shape
    this repo refuses to ship.
    """
    for attr in ("_cuts", "_adds", "_intersects"):
        ops = getattr(container, attr, None)
        if ops:
            getattr(body, attr).extend(ops)
            ops.clear()


def materialize_bound_spaces(project) -> None:
    """Resolve, derive and MATERIALIZE every bounds-mode Space in ``project``.

    The one shared compile pass (idempotent, like ``apply_displacement``):

    * resolves every ``bounds=`` fragment against the boundable-element index;
    * derives the volume (:func:`derive_volume`) and prints the report line;
    * injects the derived box as the space's body — an anonymous ``Box``
      child, exactly the shape an authored space carries, so the emitter
      emits the IfcSpace representation through the path they already have.
      The box is ``carve="none"``-marked: the derived volume is AIR, and
      letting it displace the very bounds it was derived from (a chimney
      breast poking into the room) would eat declared geometry;
    * moves the space's own boolean operands onto that body (both modes —
      see :func:`_push_booleans_onto`) so ``kitchen.difference(...)``
      composes onto the derived volume exactly as on an authored body;
    * stamps ``_bounds_resolved`` / ``_bounds_face_axes`` /
      ``_derived_volume`` / ``_effective_volume`` for the boundary-collection
      fork and the demoted WS-C checker in
      ``space_boundaries.collect_boundaries``.

    Runs in the MODEL's units (metres — the emitter calls it after
    ``normalize_project_to_meters``, the ``collect_boundaries`` contract).
    Raises :class:`SpaceVolumeError`; the emitter surfaces it as a failed
    compile result.
    """
    if getattr(project, "_space_bounds_materialized", False):
        return

    from lite_step.ifc.space_boundaries import _top_level
    from lite_step.models import taxonomy as tx

    spaces = [e for e in _top_level(project) if type(e).__name__ == "Space"]
    _refuse_nested_bounds_spaces(project, spaces)

    bound_spaces = [s for s in spaces if getattr(s, "bounds", None)]
    index: Optional[Dict[str, Tuple[Any, Aabb]]] = None

    for space in bound_spaces:
        name = getattr(space, "_canonical_name", None)
        if not name:
            raise SpaceVolumeError(
                "space bounds: an anonymous Space declares bounds= — the "
                "declared members join by the space's canonical name, so a "
                "bounds-mode Space must be named (Space(name=..., "
                "bounds=[...])).")
        if getattr(space, "placement", None) is not None:
            raise SpaceVolumeError(
                f"space bounds: {name} carries both bounds= and placement= — "
                f"bounds are world-space references, so the derived volume "
                f"already stands where the bounds stand and a placement "
                f"would move the room off them. Drop placement=.")
        if getattr(space, "_elements", None):
            # The .add()/.anchor() guards catch this at the call site; this
            # is the mutation path (bounds assigned after children).
            raise SpaceVolumeError(
                f"space bounds: {name} declares bounds= and also carries "
                f"authored child geometry — either the compiler derives the "
                f"volume from the bounds, or you author it; never both.")

        if index is None:
            index = _candidate_index(project)

        resolved: List[str] = []
        for fragment in space.bounds:
            canonical = _resolve_bound(fragment, name, index)
            if canonical in resolved:
                raise SpaceVolumeError(
                    f"space bounds: {name} declares {canonical} twice (as "
                    f"{fragment!r}) — a member listed twice would emit two "
                    f"identical boundaries.")
            resolved.append(canonical)

        derived = derive_volume([(c, index[c][1]) for c in resolved])
        cuts = _cut_boxes(space)

        _inject_body(space, derived.volume)
        try:
            space._bounds_resolved = tuple(resolved)
            space._bounds_face_axes = dict(derived.face_axes)
            space._derived_volume = derived.volume
            space._effective_volume = effective_volume(derived.volume, cuts)
        except Exception:  # pragma: no cover - non-pydantic stand-ins
            pass
        print(report_line(getattr(space, "name", None) or name,
                          derived.volume, derived.features,
                          derived.interior))

    # Geometry-mode spaces: compose container-level booleans onto the body
    # they already have (see _push_booleans_onto).
    for space in spaces:
        if getattr(space, "_bounds_resolved", None) is not None:
            continue
        if not (getattr(space, "_cuts", None) or getattr(space, "_adds", None)
                or getattr(space, "_intersects", None)):
            continue
        body = next((c for c in space._elements if tx.is_prism(c)), None)
        if body is None:
            logger.warning(
                "space bounds: %s carries boolean operands but no prism body "
                "to compose them onto — the booleans are dropped",
                getattr(space, "_canonical_name", None) or "a Space")
            continue
        _push_booleans_onto(body, space)

    try:
        project._space_bounds_materialized = True
    except Exception:  # pragma: no cover - non-pydantic stand-ins in tests
        pass


def _refuse_nested_bounds_spaces(project, top_spaces) -> None:
    """A bounds-mode Space anywhere but a storey/site is refused, loudly.

    The candidate walk, the boundary walk and the emission walk all read
    TOP-LEVEL spaces; a bounds-carrying Space nested inside a container would
    silently skip materialization and emit an empty room.
    """
    top_ids = {id(s) for s in top_spaces}

    def _walk(elem):
        if (type(elem).__name__ == "Space" and getattr(elem, "bounds", None)
                and id(elem) not in top_ids):
            raise SpaceVolumeError(
                f"space bounds: Space "
                f"'{getattr(elem, 'name', None) or '<anonymous>'}' declares "
                f"bounds= but does not stand in a storey or site — a "
                f"bounds-mode Space must be a top-level element, or its "
                f"derived volume and boundaries would silently never be "
                f"emitted.")
        for child in getattr(elem, "_elements", None) or []:
            _walk(child)

    for storey in getattr(project, "storeys", None) or []:
        for elem in storey.elements:
            _walk(elem)
    for site in getattr(project, "sites", None) or []:
        _walk(site)


def _inject_body(space, volume: Aabb) -> None:
    """The derived volume as the space's anonymous Box body.

    Appended to ``_elements`` directly — ``Space.add()`` refuses children on
    a bounds-mode space by design, and this body is the compiler's, not the
    author's. Anonymous like an authored space body (no canonical name, no
    manifest key, no patch atom). ``suspend_strict`` because the strict
    int-mm gate polices what an AUTHOR writes and these floats are derived
    metres ([[feedback_suspend_strict_for_internal_geometry]]).
    """
    from lite_step.models import Box, Point
    from lite_step.models.elements import _bump_generation
    from lite_step.strict import suspend_strict

    (lo, hi) = volume
    with suspend_strict():
        body = Box(start=Point(x=lo[0], y=lo[1], z=lo[2]),
                   end=Point(x=hi[0], y=hi[1], z=hi[2]))
    body._no_carve = True
    body._parent = space
    _push_booleans_onto(body, space)
    space._elements.append(body)
    _bump_generation()
