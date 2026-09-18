"""``Bounds`` — what every bounding query returns.

One type for all three accessors (``authored_aabb`` / ``world_aabb`` / ``obb``),
so a helper written against one works against the others. What differs between
them is not the shape of the answer but the *directions* the two corners are
separated along.

**Two corners describe an AABB completely, and an OBB only partially.**
Infinitely many oriented boxes share one diagonal. So:

* on an AABB, ``min`` and ``max`` are literally the componentwise minimum and
  maximum, and ``max - min`` is the size;
* on a rotated OBB they are two DIAGONAL corners, ``max.x`` can be less than
  ``min.x``, and ``max - min`` is **not** a width.

:attr:`Bounds.size` is therefore the documented extents accessor on all three
— a stored projection onto the box's own axes, never derived by subtracting
corners.

Units are **integer millimetres**, always — and that is a promise this type
keeps rather than a description of whatever it was handed. A caller whose
coordinates are in metres (anything past ``normalize_project_to_meters``)
states so with ``divisor``, exactly as :mod:`lite_step.compiler.extent`
already requires, and the constructor converts. Handed metres unstated, every
sub-metre extent rounded to nothing and the box came back a lie:
``size=(1.0, 0.0, 0.0)`` for a 1000x200x400 mm wall.
:meth:`Bounds.__post_init__` refuses the part of that it can SEE — an extent
that is real but too small to survive the rounding. It cannot see the rest:
nothing about a coordinate discloses its unit, which is the whole reason the
domain has to be stated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from lite_step.models.primitives import Point
from lite_step.strict import suspend_strict

Vec3 = Tuple[float, float, float]

#: World axes — what an AABB's ``axes`` always are.
WORLD_AXES: Tuple[Vec3, Vec3, Vec3] = (
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
)


class BoundsError(ValueError):
    """A bounding query cannot be answered — and would have to guess to try.

    Raised rather than returning a plausible number: an empty container, an
    element whose world position is not decided yet, a named face on a box
    whose extents are too close to tell apart. Every one of those has a
    defensible-looking wrong answer, which is exactly why it must not be
    returned.
    """


def round_half_up(value: float) -> int:
    """Round to the nearest integer, ``.5`` going UP (toward +infinity).

    NOT :func:`round`, and not the repo's ``I = lambda x: int(round(x))``:
    Python rounds half to EVEN, so ``round(1350.5)`` is ``1350`` while
    ``round(1351.5)`` is ``1352``. A 2701 mm wall centres at 1350.5, and the
    two conventions disagree there — which is the sort of 1 mm asymmetry
    someone spends an afternoon chasing.
    """
    return int(math.floor(value + 0.5))


#: Anything at or above this is REAL geometry; anything below is float noise.
#: World coordinates here run to ~1e5 mm, where a float64 ULP is ~1e-11 mm, so
#: a 1 micrometre floor sits eight orders of magnitude clear of the residue a
#: rotation leaves behind — and eight orders below the smallest extent DSL
#: v1.5's strict int-mm rule lets an author write.
_NOISE_FLOOR_MM = 1e-3

#: Below this an extent does not survive :func:`round_half_up`. ``0.5`` exactly
#: rounds UP to 1, so the destroyed band is open at the top.
_ROUNDS_TO_NOTHING_MM = 0.5


def _pt(x: float, y: float, z: float) -> Point:
    """A ``Point`` from computed floats, rounded to int mm.

    ``suspend_strict`` because the strict int-mm gate rejects float input by
    design — it is there to catch an AUTHOR writing ``Point(x=2000.0)``, not
    to stop the compiler handing back the result of its own arithmetic.
    Missing this once broke every site-context build for a day.
    """
    with suspend_strict():
        return Point(x=round_half_up(x), y=round_half_up(y), z=round_half_up(z))


@dataclass(frozen=True)
class Bounds:
    """A box in world millimetres, with the axes it is oriented to.

    Construct via :meth:`from_aabb` or :meth:`from_frame`; the accessors on
    ``BimElement`` are the only callers that should build one.
    """

    #: Centre of the box in world mm, kept as UNROUNDED floats. Every corner
    #: and face is reached from here, which is what makes ``point_at``'s
    #: percent-of-half-extent addressing straightforward.
    #:
    #: Unrounded on purpose. Rounding the centre and then deriving corners
    #: from it loses the corners: a 347 mm extent starting at y=-2410 has its
    #: centre at -2236.5, which rounds to -2236, and ``min`` then comes back
    #: -2409 — one millimetre off the number the author wrote. Rounding
    #: happens once, at the OUTPUT of ``point_at``, so an authored corner
    #: round-trips exactly.
    center_mm: Tuple[float, float, float]
    #: Extents along ``axes``, unrounded floats. See ``size``.
    size_mm: Tuple[float, float, float]
    #: Three world unit vectors: (along, across, up) for an OBB, the world
    #: axes for an AABB.
    axes: Tuple[Vec3, Vec3, Vec3] = WORLD_AXES
    #: ``False`` when this is a BOUND rather than the true surface — see
    #: ``compiler.extent``. A loose bound presented as an extent is a
    #: plausible wrong number, so it is labelled rather than hidden.
    exact: bool = True
    #: Which derivation produced the orientation. ``"aabb"`` means the axes
    #: are the world axes and there is nothing oriented about this box.
    rule: str = "aabb"

    def __post_init__(self) -> None:
        """Refuse a box whose extents cannot survive being reported.

        ``size`` / ``min`` / ``max`` / ``center`` round to int mm, so an
        extent under half a millimetre comes back ZERO — a number that is not
        the answer, handed out as if it were. Every alternative to raising is
        worse: reporting the zero is (a 200 mm wall leaf
        answering ``size.y == 0.0``), and reporting the unrounded float would
        break the no-floats rule at the ``Point`` surface that every author
        builds geometry back out of.

        Two causes, and the message names both because they are not
        distinguishable from here. Either the geometry really is sub-millimetre
        — which strict int-mm authoring cannot express, so it is derived — or,
        far likelier, the caller is in the METRE domain and did not say so
        (:mod:`lite_step.compiler.extent`'s unit invariant: nothing about a
        coordinate discloses its unit).

        **This is a partial guard and is not sold as more.** It catches the
        collapse to nothing; it cannot catch a metre-domain 0.508 arriving as
        "1 mm", because 0.508 is a perfectly ordinary millimetre extent. Only
        stating the domain fixes that, which is what ``divisor`` is for.
        """
        for axis, value in zip("xyz", self.size_mm):
            v = abs(float(value))
            if _NOISE_FLOOR_MM <= v < _ROUNDS_TO_NOTHING_MM:
                raise BoundsError(
                    f"Bounds(rule={self.rule!r}) has size_mm.{axis}={value!r}, "
                    f"which is real but rounds to 0 mm — reporting it would "
                    f"answer 'this has no extent along {axis}'. If these "
                    f"coordinates are in METRES (anything past "
                    f"normalize_project_to_meters), state it: pass "
                    f"divisor=1000.0 to the bounding query, the same divisor "
                    f"lite_step.compiler.extent takes. If the geometry really "
                    f"is sub-millimetre, it cannot be reported in the integer "
                    f"millimetres this type promises — read size_mm directly."
                )

    # -- corners ---------------------------------------------------------

    @property
    def center(self) -> Point:
        """The box centre, int mm."""
        return _pt(*self.center_mm)

    @property
    def size(self) -> Point:
        """Extents along :attr:`axes`, int mm. Always positive.

        THE accessor for extents. On an AABB ``max - min`` happens to agree;
        on a rotated OBB the two corners are a diagonal and subtracting them
        is meaningless, so this is what every caller should use.
        """
        return _pt(*self.size_mm)

    @property
    def min(self) -> Point:
        """The ``point_at(-100, -100, -100)`` corner.

        On an AABB this is the componentwise minimum. On a rotated OBB it is
        one end of a diagonal and nothing more — use :attr:`size` for extents.
        """
        return self.point_at(-100, -100, -100)

    @property
    def max(self) -> Point:
        """The ``point_at(+100, +100, +100)`` corner. See :attr:`min`."""
        return self.point_at(100, 100, 100)

    @property
    def is_axis_aligned(self) -> bool:
        """True when :attr:`min`/:attr:`max` really are componentwise
        extremes — i.e. when subtracting them is a legal way to get a size."""
        return self.axes == WORLD_AXES

    # -- addressing ------------------------------------------------------

    def point_at(self, i: int, j: int, k: int) -> Point:
        """A point on or in the box, addressed in percent of the half-extent.

        ``i``, ``j``, ``k`` run ``-100 .. +100`` along the box's own three
        axes. ``0`` is the centre; ``±100`` are the faces. Read it as "percent
        of the distance FROM the centre TO the face" — "percent of the
        distance to the centre" is the inverse and someone will implement it
        backwards.

            b.point_at(   0,    0,    0)    # centre
            b.point_at(   0, -100,    0)    # centre of one across-face
            b.point_at( 100,  100,  100)    # a corner
            b.point_at(   0,    0,  -50)    # quarter height, on the axis

        Outside ``[-100, 100]`` is an error, not an extrapolation: go past the
        box with ordinary arithmetic (``b.point_at(100, 0, 0) + Point(x=300)``)
        so that "on the box" and "off the box" stay visibly different at the
        call site.

        Returns a concrete ``Point`` in int mm, evaluated immediately — these
        are crutches for extrusion and placement maths, not lazy proxies.
        Halves round UP (see :func:`round_half_up`), so a 2701 mm extent
        centres at 1351 rather than 1350.
        """
        for name, v in (("i", i), ("j", j), ("k", k)):
            if not isinstance(v, int) or isinstance(v, bool):
                raise BoundsError(
                    f"point_at({name}=) takes an int percent in [-100, 100], "
                    f"got {v!r}. Integer percent keeps the no-floats rule at "
                    f"the API surface."
                )
            if not -100 <= v <= 100:
                raise BoundsError(
                    f"point_at({name}={v}) is outside [-100, 100]. ±100 IS "
                    f"the face; to reach past the box add an offset instead — "
                    f"point_at(100, 0, 0) + Point(x={v * 10}) — so that on-box "
                    f"and off-box read differently."
                )
        half = tuple(c / 2.0 for c in self.size_mm)
        frac = (i / 100.0, j / 100.0, k / 100.0)
        x, y, z = self.center_mm
        for axis, h, f in zip(self.axes, half, frac):
            x += axis[0] * h * f
            y += axis[1] * h * f
            z += axis[2] * h * f
        return _pt(x, y, z)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_aabb(cls, aabb, *, exact: bool = True, rule: str = "aabb",
                  divisor: float = 1.0) -> "Bounds":
        """From a ``((x0,y0,z0), (x1,y1,z1))`` box in the CALLER's units.

        ``divisor`` is the one from :mod:`lite_step.compiler.extent` — how many
        millimetres one caller unit is: ``1.0`` for the authoring domain,
        :data:`~lite_step.compiler.extent.MM_PER_METER` past
        ``normalize_project_to_meters``. That module DIVIDES frozen millimetre
        facts down into the caller's units; this one MULTIPLIES the caller's
        coordinates back up into the millimetres it promises. One number, two
        directions, both stated rather than guessed.
        """
        (x0, y0, z0), (x1, y1, z1) = aabb
        d = float(divisor)
        return cls(
            center_mm=((x0 + x1) / 2.0 * d, (y0 + y1) / 2.0 * d,
                       (z0 + z1) / 2.0 * d),
            size_mm=(abs(x1 - x0) * d, abs(y1 - y0) * d, abs(z1 - z0) * d),
            axes=WORLD_AXES,
            exact=exact,
            rule=rule,
        )

    @classmethod
    def from_frame(cls, center, size, axes, *, exact: bool = True,
                   rule: str = "obb", divisor: float = 1.0) -> "Bounds":
        """From a centre, an (along, across, up) extent triple and unit axes.

        ``center`` and ``size`` are in the caller's units — see
        :meth:`from_aabb` for ``divisor``. ``axes`` are UNIT vectors and carry
        no length, so they are never scaled.
        """
        d = float(divisor)
        return cls(
            center_mm=tuple(float(c) * d for c in center),
            size_mm=tuple(float(c) * d for c in size),
            axes=tuple(tuple(float(c) for c in a) for a in axes),
            exact=exact,
            rule=rule,
        )

    def __repr__(self) -> str:                          # pragma: no cover
        c, s = self.center, self.size
        head = (f"Bounds(center=({c.x}, {c.y}, {c.z}) "
                f"size=({s.x}, {s.y}, {s.z}) rule={self.rule!r}")
        if not self.exact:
            head += " exact=False"
        if not self.is_axis_aligned:
            head += f" axes={self.axes}"
        return head + ")"
