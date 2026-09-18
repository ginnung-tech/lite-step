"""Material registries (DSL v1.5) and ``material=`` resolution.

Each registry module is a per-market product catalog (``dk`` is the first).
A registry maps ``key -> MaterialDef`` (Psets + render + geometry/stock).
The canonical selection model ``Material`` lives in ``lite_step.models`` and
validates against *all* registries registered here.

Resolution contract for ``material=`` on a geometry primitive
(``Optional[str | Material]``, default ``None``):

* ``None``                        -> sketch stage (no material).
* legacy vocabulary string        -> pre-v1.5 behavior, untouched: the four
  legacy names ("Concrete", "Timber", "Steel", "Masonry"), the ``"Void"``
  boolean-cut sentinel, and ``"#rrggbb"`` hex color strings (Mesh channel).
  No warning, no IfcMaterial emission.
* any other bare string           -> ``Material(key=...)``; registry keys get
  the full registry treatment, unknown keys emit a plain named IfcMaterial
  plus a compile warning (open vocabulary).
* ``Material(...)``               -> used as-is (construction already
  validated dimensions against the registry).
"""
from __future__ import annotations

from typing import Iterator, Optional, Union

from lite_step.models.material import Material

from . import dk
from .dk import MaterialDef

__all__ = [
    "Material",
    "MaterialDef",
    "REGISTRIES",
    "LEGACY_MATERIAL_KEYS",
    "registry_definition",
    "resolve_material",
    "element_registry_material",
    "iter_material_elements",
    "building_uses_registry_materials",
    "project_uses_layers",
    "dk",
]

#: Registered registries by market code. Future registries (fi, uk, …) are
#: added here; ``registry_definition`` consults them in insertion order.
REGISTRIES: dict[str, dict[str, MaterialDef]] = {"dk": dk.MATERIALS}

#: Legacy sketch-stage vocabulary that keeps its exact pre-v1.5 behavior:
#: the four legacy bim_catalog material names (retired in DSL v17, inlined
#: here) plus the "Void" boolean-cut sentinel. These never resolve to a
#: registry Material — no warning, no IfcMaterial emission.
LEGACY_MATERIAL_KEYS: frozenset[str] = frozenset(
    {"Concrete", "Timber", "Steel", "Masonry"}
) | frozenset({"Void"})


def registry_definition(key: str) -> Optional[MaterialDef]:
    """The ``MaterialDef`` for ``key`` across all registries, or ``None``."""
    for registry in REGISTRIES.values():
        mdef = registry.get(key)
        if mdef is not None:
            return mdef
    return None


def resolve_material(value: Union[str, Material, None]) -> Optional[Material]:
    """Resolve a primitive's ``material=`` value to a ``Material`` selection.

    Returns ``None`` for the sketch stage (``value is None``) and for the
    legacy string channels (see module docstring) — those keep today's
    behavior byte-for-byte. Everything else resolves to a ``Material``.
    """
    if value is None:
        return None
    if isinstance(value, Material):
        return value
    if not isinstance(value, str):
        return None
    if value in LEGACY_MATERIAL_KEYS or value.startswith("#"):
        return None
    return Material(key=value)


def element_registry_material(elem: object) -> Optional[Material]:
    """The registry ``Material`` selection carried by a DSL element, or ``None``.

    ``Mesh`` participates like every other element (spec-mesh-registry-
    materials /): the discriminator is hex-vs-grade, not element
    type. ``resolve_material`` already returns ``None`` for the legacy hex
    channel (``material="#RRGGBB"``) and the legacy vocabulary, so those
    keep their pre-v1.5 behavior byte-for-byte; a registry grade
    (``material="Gravel_16-32"``) gets the full registry treatment —
    IfcMaterial + Psets + render colour.
    """
    return resolve_material(getattr(elem, "material", None))


def iter_material_elements(building: object) -> Iterator[object]:
    """Every element that can carry ``material=``: storey elements AND
    ``building.sites`` children (site groundwork — a gravel
    ``IfcEarthworksFill`` wrapper or a graded terrain mesh — carries
    materials too), plus all container/opening children (``_elements``),
    recursively. Missing the sites walk makes site materials invisible to
    IfcMaterial emission."""
    for storey in building.storeys:
        for elem in storey.elements:
            yield from _walk(elem)
    for site in (getattr(building, "sites", None) or []):
        yield from _walk(site)


def _walk(elem: object) -> Iterator[object]:
    yield elem
    for child in (getattr(elem, "_elements", None) or []):
        yield from _walk(child)


def project_uses_layers(building: object) -> bool:
    """True when any element carries a ``layers=`` buildup.

    Nothing in the package calls this; it gated a routing decision that no
    longer exists."""
    for elem in iter_material_elements(building):
        if getattr(elem, "layers", None) is not None:
            return True
    return False


def building_uses_registry_materials(building: object) -> bool:
    """True when any element carries a registry material (``Material``
    instance or a non-legacy string).

    Nothing in the package calls this; it gated a routing decision that no
    longer exists."""
    for elem in iter_material_elements(building):
        if element_registry_material(elem) is not None:
            return True
    return False
