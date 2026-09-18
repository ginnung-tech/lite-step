"""Geometric treatments — how a material is *shaped*, as opposed to what it IS.

A ``MaterialDef`` answers "what is this stuff" (psets, render, product facts in
``geometry``). A ``Treatment`` answers "what does one of me look like" — the
pantile wave, the moulding profile, the single brick. Identity on top,
geometric treatments underneath.

Treatments are **additive and optional**. A registry entry with no
``treatments`` behaves exactly as it always has (implicitly ``mass``), and its
IFC output is byte-identical — pinned by ``test_treatments.py``.

Why this is a sibling of ``geometry`` and not a replacement for it: ``geometry``
records *product facts* (a clay tile IS a sheet product — it has a format, a
cover width, a batten spacing), and 7 non-test consumers read ``.geometry.form``.
A treatment is a *representation choice* on top of those facts. Conflating them
would mean a clay tile is simultaneously a sheet and a member.

Selection between treatments is the author's, never the compiler's. The
compiler emits what it was asked for; it never inspects LOD.

Coordinates are int mm, like the rest of the DSL. Profiles are plain
``(x, y)`` tuples rather than ``Point2D`` on purpose: the registry is product
data and must not depend on the geometry model layer (it would also drag the
strict int-mm gate into import-time registry construction).
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: A closed 2D outline in local mm. First point is NOT repeated at the end.
Profile2D = list[Tuple[int, int]]


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


def _validate_profile(profile: Profile2D, what: str) -> None:
    if len(profile) < 3:
        raise ValueError(f"{what} needs at least 3 points, got {len(profile)}")
    if profile[0] == profile[-1]:
        raise ValueError(
            f"{what} must not repeat the first point at the end — the outline "
            "is implicitly closed"
        )
    if len(set(profile)) != len(profile):
        raise ValueError(f"{what} has duplicate points")


class MassTreatment(_Strict):
    """Plain extrude/box — the implicit default.

    Naming it explicitly is still useful: it lets a registry entry SAY that
    mass is the intended representation rather than leaving it inferred, and
    it gives the skill variable a third value to select.
    """
    kind: Literal["mass"]


class SectionTreatment(_Strict):
    """A 2D section swept along a run — mouldings, cladding profiles, the
    pantile wave.

    ``profile`` is the section in the sweep's local plane. For a roof covering
    the wave runs across the slope and repeats every ``cover_width_mm``; the
    skill sweeps it along the course.
    """
    kind: Literal["section"]
    profile: Profile2D

    @model_validator(mode="after")
    def _check(self) -> "SectionTreatment":
        _validate_profile(self.profile, "SectionTreatment.profile")
        return self


class UnitTreatment(_Strict):
    """One discrete physical unit plus its repeat pitch — a single tile, brick
    or block, placed N times by transform.

    The unit is ``profile`` extruded ``extrude_mm``. That is deliberately
    narrow for v1: it covers tiles, bricks and blocks, and anything that needs
    a curved unit is a ``section`` sweep instead.

    ``module_mm`` is the (along, across) repeat pitch. Leave it ``None`` and it
    resolves from the entry's own ``geometry`` — ``cover_width_mm`` /
    ``format_mm`` on a sheet, ``format_mm`` / ``module_length_mm`` on a mass.
    One-source rule: a dimension already stated as a product fact is never
    restated here. See :func:`resolve_module_mm`.
    """
    kind: Literal["unit"]
    profile: Profile2D
    extrude_mm: int = Field(gt=0)
    module_mm: Optional[Tuple[int, int]] = None

    @model_validator(mode="after")
    def _check(self) -> "UnitTreatment":
        _validate_profile(self.profile, "UnitTreatment.profile")
        if self.module_mm is not None and not all(v > 0 for v in self.module_mm):
            raise ValueError(f"UnitTreatment.module_mm must be positive, got {self.module_mm}")
        return self


Treatment = Annotated[
    Union[MassTreatment, SectionTreatment, UnitTreatment],
    Field(discriminator="kind"),
]


def resolve_module_mm(treatment: "UnitTreatment", geometry: object) -> Tuple[int, int]:
    """The (along, across) repeat pitch for a unit treatment.

    Explicit ``module_mm`` wins. Otherwise it is derived from the entry's
    product facts, so the pitch is stated exactly once in the registry:

    * ``sheet``  — ``(cover_width_mm or format_mm[0], batten_spacing_mm[1] or
      format_mm[1])``. A roof tile states its cover width and batten spacing as
      product facts already; that IS the pitch.
    * ``mass``   — ``(format_mm[0] + joint, format_mm[2] + joint)`` for a brick
      (length x height in the face plane), ``module_length_mm`` where given.

    Raises loudly when neither an explicit pitch nor derivable facts exist —
    a silently-guessed pitch would tile the roof wrong and look plausible.
    """
    if treatment.module_mm is not None:
        return treatment.module_mm

    form = getattr(geometry, "form", None)

    if form == "sheet":
        fmt = getattr(geometry, "format_mm", None)
        along = getattr(geometry, "cover_width_mm", None) or (fmt[0] if fmt else None)
        battens = getattr(geometry, "batten_spacing_mm", None)
        across = (battens[1] if battens else None) or (fmt[1] if fmt else None)
        if along and across:
            return (int(along), int(across))

    if form == "mass":
        fmt = getattr(geometry, "format_mm", None)
        joint = getattr(geometry, "joint_thickness_mm", None) or 0
        if fmt:
            return (int(fmt[0]) + joint, int(fmt[2]) + joint)
        mod_len = getattr(geometry, "module_length_mm", None)
        masonry = getattr(geometry, "masonry_module", None)
        if mod_len and masonry:
            return (int(mod_len), int(masonry.module_height_mm))

    raise ValueError(
        f"unit treatment on a '{form}' material states no module_mm and its "
        "geometry carries no derivable pitch (cover_width_mm / format_mm / "
        "module_length_mm) — state module_mm explicitly"
    )
