"""
Lite-STEP Models - Pydantic models for BIM elements.

These models are injected into the LLM script execution namespace
and validated during Project extraction.
"""

from .primitives import BimInt, BimFloat, Point, Point2D
from .material import Material, LayerSet
from .elements import (
    BimElement,
    Transform,
    Anchor,
    HalfSpace,
    miter,
    Door,
    Window,
    Wall,
    Site,
    # Semantic element classes for architecture sketching
    Column,
    Beam,
    Slab,
    Roof,
    # Geometry primitives
    Box,
    Extrude,
    Sweep,
    Sheet,
    Pipe,
    Revolve,
    Bar,
    Element,
    SpatialElement,
    ELEMENT_IFC_CLASS_WHITELIST,
    ELEMENT_IFC4_CLASSES,
    ELEMENT_IFC43_CLASSES,
    BAR_TYPE_TO_IFC,
    collect_consumed_operand_ids,
    Mesh,
    # Planning primitives
    ReferencePoint,
    GuideLine,
    Space,
)
from .product import Product
from .project import Storey, Project

__all__ = [
    # Primitives
    "BimInt",
    "BimFloat",
    "Point",
    "Point2D",
    # Material selection (v1.5 — registry-backed product spec)
    "Material", "LayerSet",
    # Placement types
    "Transform",
    "Anchor",
    "HalfSpace",
    "miter",
    # Assembly Elements (hierarchical containers)
    "Door",
    "Window",
    "Wall",
    "Site",
    "Roof",
    # Semantic element classes for architecture sketching
    "Column",
    "Beam",
    "Slab",
    # Geometry Primitives
    "BimElement",
    "Box",
    "Extrude",
    "Sweep",
    "Sheet",
    "Pipe",
    "Revolve",
    "Bar",
    "Element",
    "SpatialElement",
    "ELEMENT_IFC_CLASS_WHITELIST",
    "ELEMENT_IFC4_CLASSES",
    "ELEMENT_IFC43_CLASSES",
    "BAR_TYPE_TO_IFC",
    "collect_consumed_operand_ids",
    "Mesh",
    # Planning primitives
    "ReferencePoint",
    "GuideLine",
    "Space",
    # Catalog item (WS-B — promotion, not declaration)
    "Product",
    # Project (root container) + Storey
    "Storey",
    "Project",
]
