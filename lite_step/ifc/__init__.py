"""
Lite-STEP IFC Generator - Convert Project to IFC using ifcopenshell.

This module generates IFC4 files from validated Project objects.
Uses ifcopenshell.api.run() for high-level operations.
"""

from .generator import generate_ifc, IFCGenerationResult
from .entity_cache import EntityCache
from .deduplicator import deduplicate_ifc_step

__all__ = [
    "generate_ifc",
    "IFCGenerationResult",
    "EntityCache",
    "deduplicate_ifc_step",
]
