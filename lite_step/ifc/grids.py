"""The structural GRID — one ``IfcGrid`` per container, not one per axis.

Every policy decision — what counts as a grid line, how an axis direction is
derived, which family becomes U and which V, every refusal, and what the
emitted ``IfcGrid`` carries — is decided HERE; ``generator.py`` contributes
only the lines that mint an entity. Two implementations of one idea agreeing
*usually* is what produces the bug.

``IfcGrid`` carries ALL the axes of ONE grid — ``UAxes=(A,B,C,D)``,
``VAxes=(1,2,3,4)``. One grid per axis cannot fill both mandatory lists: the
axis has to go into ``UAxes`` *and* ``VAxes``, claiming it runs in both grid
directions at once, or ``VAxes`` is left empty, which ``LIST [1:?]`` forbids.

The rule
--------

**One ``IfcGrid`` per spatial container.** Every grid line standing in one
storey is one grid. Grid lines are top-level storey elements by construction
(``allowed_children_for`` gives no container ``GuideLine`` as a child, and
``IfcSite``'s row is ``Box/Extrude/Mesh/Element``), so "the container" is
always a storey and two storeys authoring grid lines get two grids — a product
lives in exactly one spatial structure element, so a shared grid would have to
pick one storey and be wrong about the other.

**U and V are derived from the axis DIRECTION, never from the label.**
"A, B, C, D" and "1, 2, 3, 4" is a drafting convention, not a fact about the
model; a model labelling its axes "N1, N2 / E1, E2" is the same grid and must
partition the same way. Axes are grouped into families of PARALLEL directions
(within :data:`PARALLEL_TOL_DEG`), and the family lying closer to global +X
becomes ``UAxes``.

That reading of the schema is the literal one: *"UAxes: List of grid axes
defining the first grid direction"* — the axes that lie ALONG u. It also
preserves what this compiler already did per-axis (``dx >= dy`` -> U), so the
change is a merge, not a re-partition. Note that drafting practice often
labels the *other* family with letters; the letters are not consulted either
way. Ties (two families equidistant from +X, e.g. ±45°) break on authoring
order, so the answer is deterministic rather than dict-ordered.

The refusals, and why each is loud
----------------------------------

Every one of these produces a file that PARSES, which is exactly the class of
failure this repo refuses to ship. In each case the message names the axes and
the fix.

* **One direction family** (the single-axis case). ``VAxes`` is
  ``LIST [1:?]`` — mandatory, non-empty — so a grid with axes in only one
  direction is *not expressible* as an ``IfcGrid``. Refused: add a crossing
  axis, or use
  ``line_type="boundary"``/``"setback"`` for a lone reference line (which
  emits an ``IfcAnnotation`` and is what a single line actually is).
* **Three or more families.** ``WAxes`` is the *third* direction of a grid,
  and nothing in a set of coplanar lines says which of three families is
  "the third" — any assignment is a guess that parses. A triangular grid is a
  real IFC concept (``PredefinedType=TRIANGULAR``); building it on a guess is
  not. Refused, with every family's bearing in the message, so a model that
  legitimately needs it arrives as a bug report with its own repro.
* **A bent axis.** ``AxisCurve`` may be any ``IfcCurve``, but the U/V split is
  a claim about DIRECTION and a polyline that turns has no single one.
  Refused rather than classified off its endpoints.
* **A vertical or zero-length axis.** A grid axis is a PLAN axis; one with no
  extent in XY has no bearing to classify.
* **Two axes with the same tag.** ``AxisTag`` is how a drawing cites an axis
  ("gridline B"); two axes answering to "B" in one grid make every dimension
  read off it ambiguous. This one only became visible when the axes were
  merged into a single grid — eight separate grids each had a unique tag
  trivially.
* **``placement=`` on a grid line.** The merged grid is ONE product with ONE
  ``ObjectPlacement``; a per-axis transform cannot be composed onto it
  and silently dropping it would move the axis. Move the axis by its
  ``points=``.

What the emitted entity carries
-------------------------------

* ``Name`` is **None**, always (:data:`GRID_NAME`). The grid corresponds to no
  DSL element — it is derived from N of them — so any synthetic name would
  enter the ``LITESTEP_META`` manifest and claim a patch identity for an
  entity no DSL element owns. That is the v21.2 ``IfcAnnotation`` trap
  (``generator.py`` emits the drawings *after* the manifest build for exactly
  this reason), and the cheapest way not to fall into it is not to name the
  thing. The axis labels are not lost: they are ``IfcGridAxis.AxisTag``, which
  is where a tag belongs and where a drawing reads it from.
* ``PredefinedType`` is ``RECTANGULAR`` when the two families are
  perpendicular within :data:`PARALLEL_TOL_DEG`, and ``$`` otherwise. It is
  optional in the schema, so omitting it is conformant; asserting
  ``IRREGULAR`` over a skewed-but-regular grid would not be, in the sense
  that matters — it would be a statement we cannot back.
* ``SameSense`` is ``True`` on every axis, unchanged: the axis curve is
  emitted in its authored direction, so its sense agrees with itself.

Statistics count AXES, not grids: ``stats["guide_lines"]`` (and therefore
``total_elements``) is documented as "every AUTHORED element counted exactly
once", and eight authored grid lines are eight authored elements however many
entities they merge into.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

#: ``GuideLine.line_type`` that routes to ``IfcGrid``. Every other value is an
#: ``IfcAnnotation`` and never reaches this module.
GRID_LINE_TYPE = "grid"

#: Two axes are the same FAMILY when their directions differ by no more than
#: this, and the grid is ``RECTANGULAR`` when the two families are
#: perpendicular to within it. One degree is ~26 mm of drift over a 1.5 m
#: axis — far looser than any authored grid and far tighter than any pair of
#: directions an author means as different.
PARALLEL_TOL_DEG = 1.0

#: Max perpendicular deviation of an intermediate point from the chord, as a
#: fraction of the axis length, for the axis to count as straight. Authored
#: axes are integer-millimetre endpoints and deviate by exactly zero; a
#: deliberate polyline deviates by metres. The threshold is nowhere near
#: either, so it is not a tuning knob.
STRAIGHTNESS_TOL_RATIO = 1e-6

#: Below this the XY projection of an axis has no direction (unit-agnostic:
#: the model is metres by the time a backend calls in, and no real axis is a
#: picometre long).
DEGENERATE_LENGTH = 1e-12

#: The merged grid's ``IfcGrid.Name``. ``None`` on purpose — see the module
#: docstring: a named grid would enter the LITESTEP_META manifest as an entity
#: no DSL element corresponds to.
GRID_NAME: Optional[str] = None

_PARALLEL_SIN = math.sin(math.radians(PARALLEL_TOL_DEG))


class GridError(ValueError):
    """A set of ``GuideLine(line_type="grid")`` cannot become one ``IfcGrid``."""


@dataclass(frozen=True)
class GridAxis:
    """One authored grid line, with the direction its family was decided on."""

    element: Any                      #: the ``GuideLine`` — the backend reads ``points`` off it
    tag: Optional[str]                #: ``IfcGridAxis.AxisTag``
    direction: Tuple[float, float]    #: unit XY direction, canonicalised to the upper half-plane
    bearing_deg: float                #: ``direction`` as degrees from +X, in ``[0, 180)``


@dataclass(frozen=True)
class ResolvedGrid:
    """The single ``IfcGrid`` a container's grid lines become."""

    u_axes: Tuple[GridAxis, ...]      #: non-empty by construction
    v_axes: Tuple[GridAxis, ...]      #: non-empty by construction
    predefined_type: Optional[str]    #: ``"RECTANGULAR"`` or ``None``

    @property
    def axes(self) -> Tuple[GridAxis, ...]:
        """Every axis, U before V, each in authoring order.

        The emission order the emitter uses, so the entity sequence in the
        STEP file is a property of this module rather than of two loops.
        """
        return self.u_axes + self.v_axes


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def is_grid_line(elem: Any) -> bool:
    """True for a ``GuideLine`` routed to ``IfcGrid``.

    Duck-typed on ``line_type`` rather than ``isinstance(elem, GuideLine)`` to
    keep this module import-free of the model layer (which imports the IFC
    layer lazily in several places); nothing else in the DSL carries that
    attribute.
    """
    return getattr(elem, "line_type", None) == GRID_LINE_TYPE


def grid_lines(elements: Sequence[Any]) -> List[Any]:
    """The grid lines among ``elements``, in authoring order."""
    return [elem for elem in elements if is_grid_line(elem)]


# ---------------------------------------------------------------------------
# Per-axis derivation
# ---------------------------------------------------------------------------


def axis_label(elem: Any) -> str:
    """How an axis is named in a refusal — its tag, else its canonical, else its id."""
    for attr in ("label", "ifc_name", "id"):
        value = getattr(elem, attr, None)
        if value:
            return str(value)
    return repr(elem)


def _axis_direction(elem: Any, where: str) -> Tuple[float, float]:
    """The unit XY direction of one grid line, or raise.

    Canonicalised into the upper half-plane (``dy > 0``, or ``dy == 0`` and
    ``dx > 0``) so that an axis authored south-to-north and one authored
    north-to-south land in the same family. That canonicalisation is for
    CLASSIFICATION only — the emitted ``AxisCurve`` keeps the authored point
    order and ``SameSense`` stays ``True``.
    """
    points = list(getattr(elem, "points", None) or ())
    if len(points) < 2:                                   # pragma: no cover
        raise GridError(
            f"{where}: grid axis {axis_label(elem)!r} has {len(points)} "
            f"point(s). A grid axis is a line; GuideLine requires at least 2.")

    x0, y0 = float(points[0].x), float(points[0].y)
    x1, y1 = float(points[-1].x), float(points[-1].y)
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length <= DEGENERATE_LENGTH:
        raise GridError(
            f"{where}: grid axis {axis_label(elem)!r} has no extent in plan "
            f"(its endpoints share x and y). A grid axis is a PLAN axis — U "
            f"and V are directions in the XY plane — so a vertical or "
            f"zero-length line has no bearing to classify and cannot be one. "
            f"Author it as line_type=\"boundary\" if it is a reference line, "
            f"or give it a plan direction.")

    # Straightness: the U/V split is a claim about direction, and a polyline
    # that turns has no single one. Measured against the chord, so a straight
    # axis with collinear intermediate points passes exactly.
    for point in points[1:-1]:
        px, py = float(point.x) - x0, float(point.y) - y0
        deviation = abs(dx * py - dy * px) / length
        if deviation > STRAIGHTNESS_TOL_RATIO * length:
            raise GridError(
                f"{where}: grid axis {axis_label(elem)!r} is bent — its "
                f"intermediate point deviates {deviation:.4g} from the line "
                f"through its endpoints. U and V are DIRECTIONS, so an axis "
                f"has to have one; classifying a bent polyline off its "
                f"endpoints would file it under a direction it does not run "
                f"in. Split it into one grid axis per straight run, or author "
                f"it as line_type=\"boundary\".")

    ux, uy = dx / length, dy / length
    if uy < 0 or (uy == 0.0 and ux < 0):
        ux, uy = -ux, -uy
    return ux, uy


def _bearing_deg(direction: Tuple[float, float]) -> float:
    """``direction`` as degrees from +X, in ``[0, 180)``."""
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    if angle < 0:
        angle += 180.0
    if angle >= 180.0:                                    # atan2 == pi exactly
        angle -= 180.0
    return angle


def _are_parallel(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Within :data:`PARALLEL_TOL_DEG`. ``|cross|`` of two unit vectors is
    ``sin`` of the angle between them, and both are canonicalised into one
    half-plane, so this needs no wrap handling."""
    return abs(a[0] * b[1] - a[1] * b[0]) <= _PARALLEL_SIN


def _are_perpendicular(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Within :data:`PARALLEL_TOL_DEG` of 90°."""
    return abs(a[0] * b[0] + a[1] * b[1]) <= _PARALLEL_SIN


def _distance_to_x_deg(bearing: float) -> float:
    """Angular distance from the +X axis, in ``[0, 90]``."""
    return min(bearing, 180.0 - bearing)


# ---------------------------------------------------------------------------
# The container-level rule
# ---------------------------------------------------------------------------


def collect_grid(elements: Sequence[Any], where: str) -> Optional[ResolvedGrid]:
    """The one :class:`ResolvedGrid` a container's grid lines form, or ``None``.

    ``elements`` is a container's element list (a storey's, in every model this
    DSL can express); ``where`` is how the container is named in a refusal,
    e.g. ``"storey 'ground'"``. Returns ``None`` when there are no grid lines
    at all — the container simply emits no ``IfcGrid``.

    Raises :class:`GridError` — never warns — on each of the conditions in the
    module docstring.
    """
    lines = grid_lines(elements)
    if not lines:
        return None

    placed = [elem for elem in lines if getattr(elem, "placement", None) is not None]
    if placed:
        raise GridError(
            f"{where}: grid axis "
            + ", ".join(repr(axis_label(e)) for e in placed)
            + f" carries placement=. Every axis of one grid is now ONE IfcGrid "
            f"with one ObjectPlacement (#625 — a grid per axis is what "
            f"produced the schema violation), so a per-axis transform has "
            f"nowhere to be composed and dropping it would move the axis "
            f"without saying so. Author the axis at its final position via "
            f"points=.")

    axes: List[GridAxis] = []
    tags: dict = {}
    for elem in lines:
        direction = _axis_direction(elem, where)
        tag = getattr(elem, "label", None) or getattr(elem, "ifc_name", None)
        if tag is not None:
            first = tags.get(tag)
            if first is not None:
                raise GridError(
                    f"{where}: two grid axes are tagged {tag!r}. AxisTag is "
                    f"how a drawing cites an axis, and one IfcGrid now holds "
                    f"them all, so two axes answering to {tag!r} "
                    f"make every dimension read off this grid ambiguous — "
                    f"and neither the file nor any viewer would say so. Give "
                    f"each axis its own label=.")
            tags[tag] = elem
        axes.append(GridAxis(element=elem, tag=tag, direction=direction,
                             bearing_deg=_bearing_deg(direction)))

    # Cluster into families of parallel axes, first-match-wins in authoring
    # order so the grouping is deterministic.
    families: List[List[GridAxis]] = []
    for axis in axes:
        for family in families:
            if _are_parallel(family[0].direction, axis.direction):
                family.append(axis)
                break
        else:
            families.append([axis])

    if len(families) == 1:
        only = families[0]
        raise GridError(
            f"{where}: all {len(only)} grid axis/axes run in one direction "
            f"({only[0].bearing_deg:.1f}° from +X): "
            + ", ".join(repr(a.tag or axis_label(a.element)) for a in only)
            + ". IfcGrid.VAxes is LIST [1:?] — mandatory and non-empty — so a "
            "grid with axes in a single direction is not expressible: it is "
            "either an empty VAxes (a schema violation) or the same axis "
            "listed in both directions (valid, and a lie). Add at least one "
            "crossing axis, or author these as line_type=\"boundary\" / "
            "\"setback\", which emits an IfcAnnotation and is what a lone "
            "reference line is.")

    if len(families) > 2:
        detail = "; ".join(
            f"{family[0].bearing_deg:.1f}° from +X: "
            + ", ".join(repr(a.tag or axis_label(a.element)) for a in family)
            for family in families)
        raise GridError(
            f"{where}: the grid axes run in {len(families)} different "
            f"directions ({detail}). An IfcGrid has UAxes, VAxes and WAxes, "
            f"and WAxes is the THIRD direction of a grid — nothing in a set "
            f"of coplanar lines says which of {len(families)} families is "
            f"that third one, so any assignment here would be a guess that "
            f"parses. Split them into separate models, or open an issue with "
            f"this model: a genuinely triangular grid (IfcGridTypeEnum "
            f"TRIANGULAR) is a real thing and deserves a real rule rather "
            f"than a fallback.")

    # Two families. U is the one lying closer to +X — the schema's "axes
    # defining the first grid direction", read literally. Ties break on
    # authoring order via the index, so ±45° is deterministic.
    ordered = sorted(
        enumerate(families),
        key=lambda item: (_distance_to_x_deg(item[1][0].bearing_deg), item[0]),
    )
    u_family = ordered[0][1]
    v_family = ordered[1][1]

    rectangular = _are_perpendicular(u_family[0].direction, v_family[0].direction)
    return ResolvedGrid(
        u_axes=tuple(u_family),
        v_axes=tuple(v_family),
        predefined_type="RECTANGULAR" if rectangular else None,
    )


def container_label(kind: str, name: Optional[str], index: int) -> str:
    """The ``where`` string a backend passes to :func:`collect_grid`.

    Shared so refusals are byte-identical for the same
    model — the property ``test_grids.py`` asserts.
    """
    if name:
        return f"{kind} {name!r}"
    return f"{kind} #{index}"
