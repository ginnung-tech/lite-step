"""
Restricted Namespace for LLM Script Execution.

This module creates a sandboxed execution environment with:
- Lite-STEP model classes (Point, Wall, Box, Project, etc.)
- BIM catalog access (profiles, dimensions)
- Safe Python builtins (math operations, loops)
- NO file I/O, NO imports, NO dangerous operations
"""

import math
import builtins as _builtins
import numpy as np
from typing import Dict, Any


# Packages allowed in sandbox import statements
ALLOWED_IMPORT_PACKAGES = frozenset({"lite_step", "math", "numpy", "random"})

_original_import = _builtins.__import__


def _make_whitelisted_import():
    """
    Create a restricted __import__ that only allows whitelisted packages.

    Generated scripts contain standard import statements so they can run
    standalone (``python current.py``).  Inside the sandbox we honour those
    imports for the packages the DSL actually uses, but block everything else.
    """
    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        top_level = name.split(".")[0]
        if top_level not in ALLOWED_IMPORT_PACKAGES:
            raise ImportError(
                f"Import not allowed in sandbox: '{name}'. "
                f"Only {sorted(ALLOWED_IMPORT_PACKAGES)} imports are permitted."
            )
        return _original_import(name, globals, locals, fromlist, level)

    return restricted_import


def create_namespace() -> Dict[str, Any]:
    """
    Create restricted namespace for LLM script execution.

    The namespace includes:
    1. Model classes - for creating building geometry
    2. Catalog access - for profile dimensions and IFC types
    3. Safe builtins - for calculations and loops
    4. Math module - for trigonometry and sqrt

    Returns:
        Dictionary to use as both globals and locals in exec()
    """
    # Import model classes
    from lite_step.models import (
        Point,
        Point2D,
        Material, LayerSet,
        BimElement,
        Transform,
        Anchor,
        HalfSpace,
        miter,
        Door,
        Window,
        Wall,
        Site,
        # Geometry primitives
        Box,
        Extrude,
        Sweep,
        Sheet,
        Pipe,
        Revolve,
        Bar,
        Element,
        Mesh,
        # Planning primitives
        ReferencePoint,
        GuideLine,
        Space,
        # Catalog item (WS-B) — promotion of a built element
        Product,
        # Project (root container) + Storey
        Storey,
        Project,
    )

    # Path helpers (DSL v1.5 — arc_path/stirrup_path for Pipe/Bar/Sweep)
    from lite_step.helpers import (
        arc_path,
        stirrup_path,
    )

    return {
        # __name__ set so `if __name__ == "__main__":` is skipped in sandbox
        "__name__": "__lite_step__",

        # === Model Classes (LLM uses these to create geometry) ===
        "Point": Point,
        "Point2D": Point2D,
        "Material": Material,  # v1.5 registry material selection
        "LayerSet": LayerSet,    # → IfcMaterialLayerSetUsage chain
        "BimElement": BimElement,
        # Opening & Fill Elements (for doors, windows, recesses)
        "Door": Door,
        "Window": Window,
        # Primary Building Elements
        "Wall": Wall,
        "Site": Site,
        # Geometry primitives
        "Box": Box,
        "Extrude": Extrude,
        "Sweep": Sweep,
        "Sheet": Sheet,
        "Pipe": Pipe,
        "Revolve": Revolve,
        "Bar": Bar,
        "Element": Element,
        "Mesh": Mesh,
        # Planning primitives
        "ReferencePoint": ReferencePoint,
        "GuideLine": GuideLine,
        "Space": Space,
        # Placement types
        "Transform": Transform,
        "Anchor": Anchor,
        "HalfSpace": HalfSpace,
        "miter": miter,
        # Catalog item (WS-B): Product(built_elem, name=) + .occurrence(name=)
        "Product": Product,
        "Storey": Storey,
        "Project": Project,

        # === Path Helpers (DSL v1.5) ===
        "arc_path": arc_path,
        "stirrup_path": stirrup_path,

        # === Integer Helper ===
        "I": lambda x: int(round(x)),

        # === Safe Built-in Functions ===
        # Iteration
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "map": map,
        "filter": filter,
        "reversed": reversed,
        "sorted": sorted,

        # Length and math
        "len": len,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "round": round,
        "pow": pow,
        "divmod": divmod,

        # Type conversions
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,

        # Data structures
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,

        # Boolean constants
        "True": True,
        "False": False,
        "None": None,

        # Math module for advanced calculations
        "math": math,

        # NumPy for advanced 3D math (row-major / C-style order by default)
        "np": np,
        "numpy": np,

        # Print for debugging (output captured but harmless)
        "print": print,
    }


def get_restricted_builtins() -> Dict[str, Any]:
    """
    Get safe __builtins__ for exec().

    Returns a dict with safe builtins that:
    - Allow Python internals to work correctly (exception handling, etc.)
    - Block dangerous operations (imports, file I/O, code execution)
    
    Returning an empty dict can cause strange errors like "__import__ not found"
    when Python's internal exception handling or traceback formatting tries
    to use builtins that were blocked.
    """
    import builtins
    
    # Safe builtins whitelist - operations that can't harm the system
    safe_builtins = {
        # Exception types (needed for proper error handling)
        "Exception": builtins.Exception,
        "BaseException": builtins.BaseException,
        "TypeError": builtins.TypeError,
        "ValueError": builtins.ValueError,
        "KeyError": builtins.KeyError,
        "IndexError": builtins.IndexError,
        "AttributeError": builtins.AttributeError,
        "NameError": builtins.NameError,
        "RuntimeError": builtins.RuntimeError,
        "StopIteration": builtins.StopIteration,
        "ZeroDivisionError": builtins.ZeroDivisionError,
        "ArithmeticError": builtins.ArithmeticError,
        "LookupError": builtins.LookupError,
        
        # Type checking. ``type`` is present: survey code classifies elements
        # by type, and denying it never contained anything — the escape it was
        # blamed for is ``type(x).__subclasses__()``, whose blocked half is the
        # ``.__subclasses__`` attribute access (still rejected by the AST
        # guard's dunder rule).
        "type": builtins.type,
        "isinstance": builtins.isinstance,
        "issubclass": builtins.issubclass,
        "object": builtins.object,
        
        # Iteration and sequences (safe, no side effects)
        "range": builtins.range,
        "enumerate": builtins.enumerate,
        "zip": builtins.zip,
        "map": builtins.map,
        "filter": builtins.filter,
        "reversed": builtins.reversed,
        "sorted": builtins.sorted,
        "iter": builtins.iter,
        "next": builtins.next,
        "slice": builtins.slice,
        
        # Math and comparison (safe, pure functions)
        "len": builtins.len,
        "abs": builtins.abs,
        "min": builtins.min,
        "max": builtins.max,
        "sum": builtins.sum,
        "round": builtins.round,
        "pow": builtins.pow,
        "divmod": builtins.divmod,
        "all": builtins.all,
        "any": builtins.any,
        
        # Type conversions (safe)
        "int": builtins.int,
        "float": builtins.float,
        "str": builtins.str,
        "bool": builtins.bool,
        "bytes": builtins.bytes,
        
        # Data structures (safe to create)
        "list": builtins.list,
        "dict": builtins.dict,
        "tuple": builtins.tuple,
        "set": builtins.set,
        "frozenset": builtins.frozenset,
        
        # String operations
        "chr": builtins.chr,
        "ord": builtins.ord,
        "repr": builtins.repr,
        "ascii": builtins.ascii,
        "format": builtins.format,
        
        # Boolean and None
        "True": True,
        "False": False,
        "None": None,
        
        # Object introspection — READ-only. Model authors survey the project
        # tree (and ifcopenshell's object tree) with exactly these four, so
        # they must resolve at runtime, not just pass the AST guard. The
        # mutating counterparts (setattr/delattr) stay out.
        "getattr": builtins.getattr,
        "dir": builtins.dir,
        "vars": builtins.vars,
        "hasattr": builtins.hasattr,
        "callable": builtins.callable,
        "id": builtins.id,
        "hash": builtins.hash,
        
        # Print for debugging (safe)
        "print": builtins.print,
        
        # Property decorators (needed for class definitions)
        "property": builtins.property,
        "staticmethod": builtins.staticmethod,
        "classmethod": builtins.classmethod,
        
        # Whitelisted import — allows lite_step / math / numpy / random
        # imports so generated scripts can be standalone-executable
        "__import__": _make_whitelisted_import(),

        # NOTE: Explicitly NOT included (dangerous):
        # - open, file, input
        # - eval, exec, compile
        # - globals, locals (could leak context)
        # - setattr, delattr (could modify objects)
        # - exit, quit, SystemExit
        #
        # Read-only introspection (getattr/dir/vars/type) IS included, above —
        # it is how a model author explores an object tree, and the AST guard
        # keeps the dunder names it could otherwise reach out of range.
    }
    
    return safe_builtins
