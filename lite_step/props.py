"""Element-level property sets — ``props={}`` (DSL v1.5, WS1 PR-D).

Every element carries a universal ``props`` field (see
:class:`lite_step.models.elements.BimElement`)::

    props={"Pset_ConcreteElementGeneral": {"ExposureClass": "XC2"}}

Semantics (spec §SIGNATURES / RULE 2):

* one **IfcPropertySet** per top-level key, attached to the element's IFC
  product via IfcRelDefinesByProperties;
* property values are non-geometric scalars — ``bool | int | float | str``;
  **floats are legal** here (the strict int-mm gate covers geometry only);
* registry material Psets never go in ``props=`` — the material registry
  attaches those to the IfcMaterial; ``props=`` is for element-level Psets
  (ExposureClass, U-values, FireRating, ...).

Shape violations are compile ERRORS (``validate_project_report``), matching
the source posture for structural violations: fail loud at compile, not at
render. Well-formed props on a boolean operand (``.cuts()`` / ``.adds()`` /
``.fills()`` geometry, which never becomes an IFC product) are a compile
WARNING — the data would be silently dropped otherwise.

Props are emitted as ``IfcPropertySet``. :func:`project_uses_element_props`
is unused: it gated a routing decision that does not exist.
"""

from __future__ import annotations

from typing import Iterator, Tuple

#: Legal property value types inside a Pset dict. ``bool`` first —
#: it is an ``int`` subclass and maps to IfcBoolean, not IfcInteger.
PSET_SCALAR_TYPES: Tuple[type, ...] = (bool, int, float, str)


def iter_props_carriers(project: object) -> Iterator[Tuple[object, bool]]:
    """Yield ``(element, emits_product)`` for every element that can carry
    ``props=``.

    ``emits_product`` is True for elements reachable through the identity
    tree (storey elements, container ``_elements`` children, ``_openings``
    attachments) — their props land on their own IFC product or, for
    consumed geometry like a Wall body Solid, on the nearest ancestor
    product. It is False for boolean-operand geometry (``_cuts`` /
    ``_adds`` / ``_fills``), which is consumed into the parent's shape and
    never emits a product — props there can never be attached.
    """
    for storey in project.storeys:  # type: ignore[attr-defined]
        for elem in storey.elements:
            yield from _walk(elem, True)


def _walk(elem: object, emits_product: bool) -> Iterator[Tuple[object, bool]]:
    yield elem, emits_product
    for child in (getattr(elem, "_elements", None) or []):
        yield from _walk(child, emits_product)
    for child in (getattr(elem, "_openings", None) or []):
        yield from _walk(child, emits_product)
    for list_name in ("_cuts", "_adds", "_fills"):
        for child in (getattr(elem, list_name, None) or []):
            yield from _walk(child, False)


def project_uses_element_props(project: object) -> bool:
    """True when any element carries a non-empty ``props={}``.

    Nothing in the package calls this; it gated a routing decision that no
    longer exists.
    """
    for elem, _ in iter_props_carriers(project):
        if getattr(elem, "props", None):
            return True
    return False


def props_shape_errors(elem: object) -> list[str]:
    """Actionable messages for every shape violation in ``elem.props``.

    The contract is ``props={"Pset_X": {"Key": scalar}}`` — string keys at
    both levels, one dict per Pset, ``bool | int | float | str`` values.
    Empty ``props`` / empty Pset dicts are fine (nothing is emitted).
    """
    errors: list[str] = []
    props = getattr(elem, "props", None) or {}
    label = f"{type(elem).__name__} '{getattr(elem, 'ifc_name', None) or getattr(elem, 'id', '?')}'"
    for pset_name, pset_props in props.items():
        if not isinstance(pset_name, str):
            errors.append(
                f"{label}: props keys must be Pset name strings; got "
                f"{pset_name!r} ({type(pset_name).__name__})"
            )
            continue
        if not isinstance(pset_props, dict):
            errors.append(
                f"{label}: props[{pset_name!r}] must be a dict of property "
                f"values — the shape is props={{'Pset_X': {{'Key': value}}}} "
                f"(one IfcPropertySet per top-level key); got "
                f"{type(pset_props).__name__}"
            )
            continue
        for prop_name, value in pset_props.items():
            if not isinstance(prop_name, str):
                errors.append(
                    f"{label}: props[{pset_name!r}] property names must be "
                    f"strings; got {prop_name!r} ({type(prop_name).__name__})"
                )
            if not isinstance(value, PSET_SCALAR_TYPES):
                errors.append(
                    f"{label}: props[{pset_name!r}][{prop_name!r}] must be a "
                    f"scalar (bool, int, float, or str); got "
                    f"{type(value).__name__}"
                )
    return errors
