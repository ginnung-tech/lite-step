"""
Lite-STEP Definitions

A lightweight representation of IFC STEP data organized into:
- context: Available profiles, precision, and units (common for all users)
- content: Extrusions, products, and other building elements
- hierarchy: Spatial structure and relationships
- models: Pydantic models for BIM elements (Point, Wall, Solid, Project)
- compiler: Safe execution of LLM-generated scripts
- ifc: IFC file generation using ifcopenshell
"""

from . import context
from . import content
from . import hierarchy
from . import models
from . import compiler
from . import ifc
from . import optimization
from lite_step.compiler.executor import compile_main, LiteStepCompileError
from lite_step.strict import (
    enable_strict_int_mm,
    disable_strict_int_mm,
    strict_int_mm_enabled,
)

__all__ = [
    "context",
    "content",
    "hierarchy",
    "models",
    "compiler",
    "ifc",
    "optimization",
    "compile_main",
    "LiteStepCompileError",
    "enable_strict_int_mm",
    "disable_strict_int_mm",
    "strict_int_mm_enabled",
]
