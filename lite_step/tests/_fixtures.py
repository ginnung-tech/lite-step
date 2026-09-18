"""Local geometry fixtures for the compiler test suite.

Compiler-feature tests must NOT exec corpus files as fixtures —
skills depend on ``lite_step``, never the reverse, and fixture coupling makes
every skill migration ripple into the compiler suite ( wrapper
migration). The ONE sanctioned skills touchpoint in this tree is
``test_skill_examples_compile.py`` (compile-only, structure-agnostic); the
skill-CONTRACT structure tests live in the top-level ``tests/`` tree.

``terrain_mesh`` reproduces the site-block geometry once exec'd from the
foundation skill (footprint + buffer extents, flat default top at
z=0, one flat bottom ``depth`` below) — now a one-call heightmap-mode Mesh,
since the engine owns the prism tessellation.
"""

from __future__ import annotations

from typing import List, Optional

from lite_step.models import Element, Mesh, Point2D, Site


def terrain_mesh(
    footprint_x: int = 12000,
    footprint_y: int = 9000,
    *,
    buffer: int = 10000,
    depth: int = 15000,
    heightmap: Optional[List[List[float]]] = None,
) -> Mesh:
    """A site-block-shaped terrain Mesh (mm): footprint + ``buffer`` on every
    side, flat top at z=0 by default, flat bottom ``depth`` below the lowest
    height. Pure geometry — wrap it to give it semantics."""
    half_x = footprint_x // 2 + buffer
    half_y = footprint_y // 2 + buffer
    return Mesh(
        heightmap=heightmap if heightmap is not None else [[0, 0], [0, 0]],
        depth=depth,
        corner_min=Point2D(x=-half_x, y=-half_y),
        corner_max=Point2D(x=half_x, y=half_y),
    )


def terrain_wrapper(
    footprint_x: int = 12000,
    footprint_y: int = 9000,
    *,
    buffer: int = 10000,
    depth: int = 15000,
    heightmap: Optional[List[List[float]]] = None,
    name: str = "terrain",
) -> Element:
    """The canonical terrain form: ``Element(IfcGeographicElement, TERRAIN)``
    wrapping one heightmap Mesh — THE carve host by semantic tag."""
    wrapper = Element(ifc_class="IfcGeographicElement",
                      predefined_type="TERRAIN", name=name)
    wrapper.add(terrain_mesh(footprint_x, footprint_y, buffer=buffer,
                             depth=depth, heightmap=heightmap))
    return wrapper


def terrain_site(
    footprint_x: int = 12000,
    footprint_y: int = 9000,
    *,
    buffer: int = 10000,
    depth: int = 15000,
    heightmap: Optional[List[List[float]]] = None,
    name: str = "site",
) -> Site:
    """A ``Site`` holding one canonical terrain wrapper."""
    site = Site(name=name)
    site.add(terrain_wrapper(footprint_x, footprint_y, buffer=buffer,
                             depth=depth, heightmap=heightmap))
    return site
