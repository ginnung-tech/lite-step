"""
Lite-STEP Primitives - Basic types for BIM geometry.

All dimensions are integers in millimeters (mm).
Coordinate system: Z-up, right-handed (IFC/CAD convention).
DSL and IFC share the same axis convention — no transform needed.
"""

from pydantic import BaseModel, Field, ValidationInfo, field_validator
from typing import Annotated, Any, Tuple, TypeVar

from lite_step.strict import strict_int_mm_enabled

# Type aliases for clarity in LLM-generated code
BimInt = Annotated[int, Field(description="Integer value in millimeters")]
BimFloat = Annotated[float, Field(description="Float value in millimeters")]


_P = TypeVar("_P", bound=BaseModel)


def _derive_frozen(instance: _P, overrides: dict[str, Any]) -> _P:
    """Derivation-by-call core for frozen primitives (DSL v1.5 §DERIVATION).

    ``instance(field=value)`` returns a **new** instance (the originals are
    frozen/immutable) with those fields replaced. The override values are
    validated exactly like construction — a *probe* instance is built from
    just the overrides (every primitive field has a default, so this always
    constructs), which runs the full validator stack including the strict
    int-mm gate. The non-overridden values are carried over untouched via
    ``model_construct``: they were already validated at their own
    construction, and the ``float``-typed storage means re-validating them
    under strict int-mm would wrongly reject spec-conforming points (same
    rationale as ``Point.__add__``).
    """
    cls = type(instance)
    unknown = sorted(set(overrides) - set(cls.model_fields))
    if unknown:
        raise TypeError(
            f"{cls.__name__} derivation got unknown field(s) "
            f"{', '.join(repr(f) for f in unknown)} — valid fields: "
            f"{', '.join(sorted(cls.model_fields))}"
        )
    probe = cls(**overrides)  # construction-equivalent validation of NEW values
    merged = {name: getattr(instance, name) for name in cls.model_fields}
    for name in overrides:
        merged[name] = getattr(probe, name)
    return cls.model_construct(**merged)


def _reject_float_when_strict(v: object, info: ValidationInfo) -> object:
    """DSL v1.5 RULE 2 gate (flag-gated — see ``lite_step.strict``).

    When strict int-mm is enabled, a ``float`` coordinate is rejected at
    construction — including integral ones like ``2000.0`` (the spec rejects
    the type, not just the value). ``bool``/``int`` pass untouched. The error
    names the field and points at the ``I()`` helper so the fix is one edit
    away. The meters-normalizer runs inside ``suspend_strict()`` — its
    float-meter Point rebuilds never hit this gate.
    """
    if strict_int_mm_enabled() and isinstance(v, float):
        raise ValueError(
            f"{info.field_name}={v!r} is a float — geometry values are int "
            f"millimeters (DSL v1.5 RULE 2). Wrap divisions in I(), e.g. "
            f"{info.field_name}=I({v!r})."
        )
    return v


class Point(BaseModel):
    """
    3D point (Z-up, right-handed).

    At script execution time: integer values in millimeters
    After compilation: float values in meters (normalized by executor)

    Coordinate system (Z-up, IFC-aligned):
    - Origin (0,0,0) at building center, ground level
    - X-axis points East (right)
    - Y-axis points North (forward/depth)
    - Z-axis points Up (height)
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    _strict_int_mm = field_validator("x", "y", "z", mode="before")(
        _reject_float_when_strict
    )

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to float tuple for ifcopenshell API."""
        return (float(self.x), float(self.y), float(self.z))

    def __add__(self, other: "Point") -> "Point":
        """Add two points (vector addition).

        Uses ``model_construct`` (skips validators): the operands were already
        validated at construction, and the ``float``-typed fields store even
        int inputs as floats — re-validating the sum under strict int-mm would
        wrongly reject arithmetic on spec-conforming points.
        """
        return Point.model_construct(
            x=self.x + other.x, y=self.y + other.y, z=self.z + other.z
        )

    def __sub__(self, other: "Point") -> "Point":
        """Subtract two points (vector subtraction). See ``__add__`` re
        ``model_construct``."""
        return Point.model_construct(
            x=self.x - other.x, y=self.y - other.y, z=self.z - other.z
        )

    def __call__(self, **overrides: Any) -> "Point":
        """Derivation-by-call (DSL v1.5): ``p2 = p1(y=4200)`` returns a new
        Point with those fields replaced. Points are frozen — the original is
        never mutated. Override values validate exactly like construction
        (the strict int-mm gate applies to them); carried-over values are not
        re-validated (see :func:`_derive_frozen`)."""
        return _derive_frozen(self, overrides)

    class Config:
        frozen = True  # Points are immutable


class Point2D(BaseModel):
    """
    2D point for local contour definitions.

    Used in ``section=`` lists (Profile, Turn) where the cross-section is
    defined on a local 2D plane and then transformed to 3D.

    Coordinates are in millimeters.
    """
    x: float = 0.0
    y: float = 0.0

    _strict_int_mm = field_validator("x", "y", mode="before")(
        _reject_float_when_strict
    )

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to float tuple."""
        return (float(self.x), float(self.y))

    def __call__(self, **overrides: Any) -> "Point2D":
        """Derivation-by-call (DSL v1.5) — see :meth:`Point.__call__`."""
        return _derive_frozen(self, overrides)

    class Config:
        frozen = True
