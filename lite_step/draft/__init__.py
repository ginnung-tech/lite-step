"""2D presentation — the drawing side of the material registry.

Separate from ``lite_step/ifc/`` on purpose: none of this becomes IFC
entities. ``IfcFillAreaStyleHatching`` / ``IfcFillAreaStyleTiles`` are marked
"no support planned" upstream in IfcOpenShell, and the drawing tools
we target style plans and sections with **CSS keyed on material identity**
instead. So the registry's 2D data compiles to a stylesheet, not to STEP.
"""
from lite_step.draft.stylesheet import (
    CATEGORY_HATCH,
    canonicalise_class_name,
    render_stylesheet,
    write_stylesheet,
)

__all__ = [
    "CATEGORY_HATCH",
    "canonicalise_class_name",
    "render_stylesheet",
    "write_stylesheet",
]
