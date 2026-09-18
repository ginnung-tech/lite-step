"""Generate a drawing stylesheet from the material registry.

## Why a stylesheet and not IFC entities

Every 2D pipeline we target is CSS-driven. IfcOpenShell marks
``IfcFillAreaStyleHatching`` and ``IfcFillAreaStyleTiles`` "no support
planned" upstream; its SVG serializer never reads ``IfcCurveStyle`` either
— line weight is CSS ``stroke-width``. What the pipeline DOES do is project
material identity onto SVG classes, and let a stylesheet decide what those
look like. So the registry's ``category`` / ``draft`` data compiles to CSS.

## The classes we target

Bonsai's drawing operator builds them like this (``get_svg_classes``)::

    IfcWall  material-<LayerSetName|Material.Name>  cut
    IfcWall  layer-material-<Layer.Material.Name>
             layer-material-category-<Layer.Material.Category>  cut

so a rule matches on our registry KEY (``IfcMaterial.Name``) or on our
material FAMILY (``IfcMaterial.Category``). Both come straight out of
``MaterialDef``. Names run through :func:`canonicalise_class_name` first.

## Patterns are referenced, never shipped

``fill: url(#concrete)`` resolves against the drawing tool's OWN pattern
library, which it installs itself. We reference ids; we copy no assets. That
keeps this file original work (MIT, like the rest of the repo) with no
licence entanglement, and it means an upstream pattern improvement lands in
our drawings for free instead of leaving us with a stale fork.

Point a drawing at this file by setting ``BBIM_Documentation.StylesheetPath``
on ``IfcProject`` — see ``lite_step/ifc/drawings.py``. That is what makes the
whole thing zero-setup for the person who opens our IFC.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Optional

#: Default SVG ``<pattern>`` id per material family. Ids are the ones shipped
#: by Bonsai's own ``patterns.svg``; a material may override via
#: ``Draft(hatch=...)``. ``None`` means "no fill rule" — the tool's default
#: (white) stands, which is correct for a cavity.
CATEGORY_HATCH: dict[str, Optional[str]] = {
    "concrete":   "concrete",
    "masonry":    "brick",
    "timber":     "wood",
    "metal":      "steel",
    "insulation": "crosshatch2",
    "glass":      "glass",
    "board":      "sand",
    "membrane":   "diagonal1",
    "aggregate":  "earth",
    "ceramic":    "diagonal2",
    "cavity":     None,
}

#: Default cut-line weight in mm per family, where it differs from the
#: stylesheet default. Structure reads heavier than finish — ordinary
#: architectural convention.
CATEGORY_LINE_WEIGHT_MM: dict[str, float] = {
    "concrete": 0.5,
    "masonry":  0.5,
    "metal":    0.5,
    "timber":   0.35,
}

_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")


def canonicalise_class_name(name: str) -> str:
    """Turn an IFC string into the CSS class token a drawing tool will emit.

    This mirrors Bonsai's ``canonicalise_class_name`` exactly:
    ``re.sub("[^0-9a-zA-Z]+", "", name)``. Note what it does NOT do — it does
    not lowercase. Matching is therefore case-sensitive, which is why the
    ``MaterialCategory`` vocabulary is lowercase and separator-free: it has to
    survive this transform to a predictable token.

        Concrete_C25-30 -> ConcreteC2530
        concrete        -> concrete
    """
    return _NON_ALNUM.sub("", name)


_HEADER = """\
/*
 * Drawing stylesheet — generated from the Lite-STEP material registry.
 * Do not edit by hand; regenerate with the model.
 *
 * Pattern ids (url(#concrete), url(#brick), …) resolve against the drawing
 * tool's own pattern library. No pattern artwork is shipped or copied here.
 *
 * Wire it up by setting BBIM_Documentation.StylesheetPath on IfcProject —
 * lite_step emits that automatically, so opening the IFC is enough.
 *
 * SPDX-License-Identifier: MIT
 */
"""


def _rule(selector: str, decls: list[str]) -> str:
    return f"{selector} {{ {' '.join(d + ';' for d in decls)} }}"


def _decls_for(hatch: Optional[str], line_weight_mm: Optional[float],
               cut_rgb: Optional[str]) -> list[str]:
    decls: list[str] = []
    if hatch:
        decls.append(f"fill: url(#{hatch})")
    if cut_rgb:
        decls.append(f"fill: {cut_rgb}")
    if line_weight_mm:
        decls.append(f"stroke-width: {line_weight_mm:g}")
    return decls


def render_stylesheet(materials: Mapping[str, object]) -> str:
    """Render the CSS for a registry.

    Two tiers, emitted in this order because they have EQUAL CSS specificity
    (two classes each) and the later rule therefore wins:

    1. **family** — ``.cut.layer-material-category-<category>``. Broad, and it
       is what makes a wall buildup legible without naming every product.
    2. **material** — ``.cut.material-<Key>`` and
       ``.cut.layer-material-<Key>``, emitted only for entries carrying an
       explicit ``draft``. This is the per-product override.

    Materials with neither a ``category`` nor a ``draft`` contribute nothing:
    an unstyled material must not silently acquire someone else's hatch.
    """
    lines: list[str] = [_HEADER]

    lines.append("/* --- material family (IfcMaterial.Category) --- */")
    for category in sorted(CATEGORY_HATCH):
        if not any(getattr(m, "category", None) == category for m in materials.values()):
            continue  # don't ship rules for families this model's registry lacks
        decls = _decls_for(
            CATEGORY_HATCH[category], CATEGORY_LINE_WEIGHT_MM.get(category), None
        )
        if not decls:
            continue
        token = canonicalise_class_name(category)
        lines.append(_rule(f".cut.layer-material-category-{token}", decls))

    overrides = [
        (key, mdef) for key, mdef in sorted(materials.items())
        if getattr(mdef, "draft", None) is not None
    ]
    if overrides:
        lines.append("")
        lines.append("/* --- per-material override (IfcMaterial.Name) --- */")
    for key, mdef in overrides:
        draft = mdef.draft
        hatch = draft.hatch or CATEGORY_HATCH.get(getattr(mdef, "category", "") or "")
        decls = _decls_for(hatch, draft.line_weight_mm, draft.cut_rgb)
        if not decls:
            continue
        token = canonicalise_class_name(key)
        # Both selectors: a single-material element carries `material-<Key>`,
        # a layer inside a buildup carries `layer-material-<Key>`.
        lines.append(_rule(f".cut.material-{token}.cut.layer-material-{token}", decls))

    lines.append("")
    return "\n".join(lines)


def write_stylesheet(path: Path, materials: Optional[Mapping[str, object]] = None) -> Path:
    """Write the generated stylesheet, creating parent dirs.

    ``materials`` defaults to every registered registry, merged. Later
    registries win on a key clash, matching ``registry_definition``'s
    insertion-order lookup.
    """
    if materials is None:
        from lite_step.materials import REGISTRIES
        merged: dict[str, object] = {}
        for registry in REGISTRIES.values():
            merged.update(registry)
        materials = merged
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_stylesheet(materials), encoding="utf-8")
    return path
