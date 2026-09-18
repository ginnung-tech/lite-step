"""Canonical ``Material`` selection model (DSL v1.5).

``material=`` on any geometry primitive takes ``Optional[str | Material]``,
default ``None``. Resolution semantics (see ``lite_step.materials``):

* ``None`` (default)  -> sketch stage: palette render (``color=`` / container
  auto-color), no material data.
* bare string          -> resolves as ``Material(key=...)``. Legacy sketch
  vocabulary ("Concrete", "Timber", "Steel", "Masonry"), the ``"Void"``
  sentinel, and ``"#rrggbb"`` hex strings keep their pre-v1.5 behavior and
  never become a registry Material.
* ``Material(...)``    -> registry-backed material: Psets + render color
  attach automatically at IFC generation.

This module holds only the selection model so that ``lite_step.models`` can
export it without importing the registries at module scope (the registries
import this class back — the lookup happens lazily inside the validator).
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Material(BaseModel):
    """Material selection on an element: registry key + optional product dimensions.

    The material IS the product spec — construction-time validation against
    the registered registries (``lite_step.materials``):

    * off-catalog ``profile_mm`` FAILS (orientation-agnostic — ``(45, 195)``
      and ``(195, 45)`` hit the same stock entry);
    * off-stock ``thickness_mm`` WARNS (non-fatal — deliberate custom
      fabrication stays expressible);
    * a dimension on the wrong ``geometry.form`` FAILS;
    * an unknown key with any dimension field FAILS (nothing to validate
      against). An unknown key alone is allowed — open vocabulary; the
      compiler warns at validation time and emits a plain named material.

    The one-source rule (a Material dimension vs the element's own
    ``section=``/``thickness=`` source) is enforced where the element pairs
    the two — ``validate_project`` — not at Material construction.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    key: str
    profile_mm: Optional[tuple[int, int]] = None    # member form only
    thickness_mm: Optional[int] = None              # planar-form layer/sheet thickness

    @model_validator(mode="after")
    def _against_catalog(self) -> "Material":
        # Lazy import: lite_step.materials imports this class at module
        # scope; deferring the registry lookup to validation time breaks
        # the would-be import cycle.
        from lite_step.materials import registry_definition

        mdef = registry_definition(self.key)
        if mdef is None:
            if self.profile_mm or self.thickness_mm:
                raise ValueError(f"unknown material {self.key!r}: "
                                 f"cannot validate dimensions against a missing catalog")
            return self    # open vocabulary — compiler warns at emit
        geo = mdef.geometry
        if self.profile_mm is not None:
            if geo.form != "member":
                raise ValueError(f"{self.key!r} is form={geo.form!r}; profile_mm is for members")
            w, h = self.profile_mm
            if not geo.has_profile(w, h):
                raise ValueError(f"section {w}x{h} not in {self.key!r} stock catalog")
        if self.thickness_mm is not None:
            # thickness_mm is the OCCUPIED thickness of a planar product — as a
            # standalone sheet/membrane (replacing the primitive's thickness=),
            # OR as one leaf of a LayerSet buildup (mass/cavity/fill included).
            # Only the LINEAR forms (member/bar) reject it — those carry a
            # profile, not a thickness (a member in a stack is the repeated-
            # member follow-up, not a continuous layer).
            if geo.form in ("member", "bar"):
                raise ValueError(f"{self.key!r} is form={geo.form!r}; use profile_mm, "
                                 f"not thickness_mm (a linear/reinforcement product)")
            # Stock check applies to catalogued flat goods only.
            stock = getattr(geo, "thicknesses_mm", None) or []
            if stock and self.thickness_mm not in stock:
                warnings.warn(f"thickness {self.thickness_mm} off-stock for {self.key!r} "
                              f"(stock: {stock}) — deliberate custom fabrication?")
        return self


class LayerSet(BaseModel):
    """Ordered planar buildup (→ ``IfcMaterialLayerSetUsage`` → ``…LayerSet``
    → ``…Layer``). Attach via ``layers=`` on a planar element (Wall / Slab /
    Roof / ``Element`` with a planar ``ifc_class``).

    A layer IS a ``Material`` at a thickness — no separate ``Layer`` noun. Each
    entry is a ``Material(key=, thickness_mm=)``; its function
    (→ ``IfcMaterialLayer.Category``) and, for a cavity product, its ventilation
    (→ ``IsVentilated``) are REGISTRY facts, never re-authored. The ordering IS
    the buildup, so the layers go in an explicit ordered ``layers=[...]`` list
    (not varargs — that syntax means an order-independent set elsewhere).

    Authoring rules (spec §MATERIALS):

    * the list is the buildup **outer → inner**: the FIRST material is the
      OUTER layer, the last the innermost. This ordering is fixed — you never
      reverse the list.
    * **`outward` (required) places the buildup**: a `(dx, dy, dz)` normal
      pointing inner→outer. The compiler puts the first-listed (outer) layer on
      the face this normal points at and stacks the rest inward, so a centred
      ring of four leaves reuses ONE identical `layers=` list and only differs
      in `outward` (S `(0,-1,0)`, N `(0,1,0)`, W `(-1,0,0)`, E `(1,0,0)`; a slab
      `(0,0,1)`). No default — the direction is geometry the script knows when
      it places the layers (same rationale as `miter(edge=)`). v1 requires an
      axis-aligned normal along the body's thin axis; a full layer transform
      (oblique buildups on rotated walls) is a follow-up.
    * the element authors its body envelope ONCE (a single Box/Extrude); the
      compiler slices it into per-layer solids. A cavity material
      (``Cavity_Ventilated`` / ``Cavity_Unventilated``) holds its span but
      emits NO solid — a literal air gap.
    * the last non-cavity layer's registry render colors the element when no
      explicit ``color=`` is set.

    ``LayerSet(name="ext_type_a", outward=(0,-1,0),
               layers=[Material(key=, thickness_mm=), …])``.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    # Sequence, not tuple: static checkers must accept the natural
    # ``layers=[Material(...), ...]`` list authoring the skills teach (a
    # ``tuple[...]`` annotation made ty reject it — a recurring agent trip in
    # the 2026-07 wire logs). The before-validator normalizes to a tuple at
    # runtime, so the stored value is immutable either way.
    layers: Sequence[Material]
    outward: Tuple[float, float, float]                    # inner→outer normal (placement)
    name: Optional[str] = None                             # → LayerSetName

    @field_validator("outward")
    @classmethod
    def _non_zero_outward(cls, v):
        if (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5 < 1e-12:
            raise ValueError(
                "LayerSet.outward must be a non-zero inner→outer normal (got "
                f"{v}) — it names where the outer (first-listed) layer faces, "
                "e.g. (0,-1,0) for a south wall or (0,0,1) for a slab")
        return v

    @field_validator("layers", mode="before")
    @classmethod
    def _coerce_and_check(cls, v: object) -> object:
        if isinstance(v, (list, tuple)):
            for i, m in enumerate(v):
                if not isinstance(m, Material):
                    raise ValueError(
                        f"LayerSet.layers[{i}] must be a Material(key=, thickness_mm=) — "
                        f"a layer is a material at a thickness (got {type(m).__name__})")
                if m.profile_mm is not None:
                    raise ValueError(
                        f"LayerSet.layers[{i}] ({m.key!r}) carries profile_mm — a member "
                        f"is not a continuous layer (repeated-member layers are a follow-up)")
                if m.thickness_mm is None:
                    raise ValueError(
                        f"LayerSet.layers[{i}] ({m.key!r}) needs thickness_mm — every "
                        f"layer states its own thickness")
            return tuple(v)
        return v

    @model_validator(mode="after")
    def _at_least_one(self) -> "LayerSet":
        if not self.layers:
            raise ValueError("LayerSet requires at least one layer Material")
        return self

    @property
    def total_thickness_mm(self) -> int:
        return sum(m.thickness_mm for m in self.layers)
