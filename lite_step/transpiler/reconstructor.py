"""
Cold Import Reconstructor: Generate flat Lite-STEP from extracted IFC elements.

Produces compilable Lite-STEP code from third-party IFC files with no
embedded LITESTEP_META. Output is flat (no loops, no helpers) but valid.

Per-element coverage tracking (exact / approximated / opaque) flows back
through CoverageReport so the UI can render "12 of 47 entities became
opaque meshes" without parsing comments.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from lite_step.ifc.normalizer import ExtractedElement

if TYPE_CHECKING:
    from lite_step.transpiler import ImportReport

logger = logging.getLogger(__name__)

# IFC type → DSL type mapping (from spec §11)
IFC_TO_DSL_TYPE = {
    "IfcWall": "Wall",
    "IfcWallStandardCase": "Wall",
    "IfcColumn": "Column",
    "IfcBeam": "Beam",
    "IfcSlab": "Slab",
    "IfcRoof": "Roof",
    "IfcDoor": "Door",
    "IfcWindow": "Window",
    "IfcMember": "Sweep",
    "IfcPlate": "Box",  # DSL v8: box/contour successor is Box/Extrude
    "IfcBuildingElementProxy": "Box",
    "IfcStair": "Box",
    "IfcRailing": "Box",
}

# Container types that need .add(body Box/Extrude) pattern.
# Slab is also a container in DSL v8 but has its own specialised emit path
# (_generate_slab), so it is intentionally NOT listed here.
CONTAINER_TYPES = {"Wall", "Column", "Beam", "Roof"}

# Standard imports for generated code. Mesh covers IfcFacetedBrep cold
# import; Box.difference() inlined-Box void chain covers IfcBooleanResult.
STANDARD_IMPORTS = """\
from lite_step.models.project import Project, Storey
from lite_step.models.primitives import Point, Point2D
from lite_step.models.elements import (
    Wall, Column, Beam, Box, Extrude, Sweep,
    Slab, Roof, Door, Window, Site, Mesh,
)"""


# ---------------------------------------------------------------------------
# Coverage report — what we got back from cold import
# ---------------------------------------------------------------------------


@dataclass
class CoverageReport:
    """Per-element reconstruction quality, surfaced to the patch UI.

    - exact: the source-code shape captures the IFC geometry losslessly
      (box / contour / mesh-with-vertices+faces / boolean-with-operands).
    - approximated: lossy but useful (e.g. boolean with only the primary
      operand, brep with vertices but no faces).
    - opaque: geometry could not be reconstructed meaningfully; a
      placement-derived placeholder Solid was emitted with the
      ``# OPAQUE: original geometry lost`` comment so the UI can grey out
      structured-edit affordances on those elements.
    """
    exact: int = 0
    approximated: int = 0
    opaque: int = 0
    by_type: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)

    def record(self, ifc_type: str, quality: str) -> None:
        """quality: "exact" | "approximated" | "opaque"."""
        if quality == "exact":
            self.exact += 1
        elif quality == "approximated":
            self.approximated += 1
        elif quality == "opaque":
            self.opaque += 1
        else:
            raise ValueError(f"unknown coverage quality: {quality!r}")
        e, a, o = self.by_type.get(ifc_type, (0, 0, 0))
        if quality == "exact":
            e += 1
        elif quality == "approximated":
            a += 1
        else:
            o += 1
        self.by_type[ifc_type] = (e, a, o)

    @property
    def total(self) -> int:
        return self.exact + self.approximated + self.opaque


def reconstruct_litestep(
    elements: List[ExtractedElement],
    import_report: Optional[ImportReport] = None,
) -> Tuple[str, CoverageReport]:
    """
    Generate flat Lite-STEP code from extracted IFC elements.

    Groups elements by storey, maps IFC types to DSL types, and produces
    a compilable script with `generate_project()` wrapper. Returns the
    source plus a CoverageReport describing how lossy the reconstruction
    was per element type.

    Args:
        elements: Normalized elements from IFC normalization
        import_report: Optional report with notes about skipped elements

    Returns:
        (source_code, coverage_report) — source is a complete Lite-STEP
        Python module; coverage_report counts exact / approximated /
        opaque emissions per IFC type.
    """
    coverage = CoverageReport()

    if not elements:
        return _wrap_empty_project(import_report), coverage

    # Pre-pass: collect GUIDs of opening fills that will be re-parented
    # under their wall container. The main per-storey loop skips these
    # so they don't double-emit as floating standalone Door/Window
    # elements at the storey level.
    fill_guids: set[str] = set()
    for elem in elements:
        for opening in elem.children:
            if opening.ifc_type != "IfcOpeningElement":
                continue
            for fill in opening.children:
                if fill.guid:
                    fill_guids.add(fill.guid)

    # Group by storey
    by_storey: Dict[int, List[ExtractedElement]] = defaultdict(list)
    storey_elevations: Dict[int, int] = {}

    for elem in elements:
        if elem.guid in fill_guids:
            continue  # re-parented under a wall via opening walk
        by_storey[elem.storey_idx].append(elem)
        storey_elevations[elem.storey_idx] = elem.storey_elevation

    # Sort storeys by index
    sorted_storeys = sorted(by_storey.keys())

    # Generate code
    lines = [
        "# Reconstructed from IFC -- structure not preserved",
    ]

    # Embed import notes as comments
    if import_report:
        lines.extend(_format_report_comments(import_report))

    lines.extend([
        STANDARD_IMPORTS,
        "",
        "",
        "def generate_project():",
        '    proj = Project(name="Imported Project")',
        "",
    ])

    # Create storeys. DSL v2.1: a single storey is emitted ANONYMOUS (no
    # name= -> no ``storey:`` path segment); a multi-storey project requires
    # every storey to be named, so each gets a leaf name.
    single_storey = len(sorted_storeys) == 1
    for idx in sorted_storeys:
        elev = storey_elevations.get(idx, 0)
        if single_storey:
            lines.append(f'    storey_{idx} = Storey(elevation={elev})')
        else:
            name = _storey_name(idx, elev)
            lines.append(
                f'    storey_{idx} = Storey(name="{name}", elevation={elev})')
        lines.append(f"    proj.add_storey(storey_{idx})")
        lines.append("")

    # Generate elements per storey
    used_ids: set[str] = set()
    for idx in sorted_storeys:
        lines.append(f"    # --- Storey {idx} ---")
        for elem in by_storey[idx]:
            elem_lines = _generate_element(elem, idx, used_ids, coverage)
            for line in elem_lines:
                lines.append(f"    {line}")
            lines.append("")

    lines.append("    return proj")
    lines.append("")
    lines.append("")
    lines.append("result = generate_project()")
    lines.append("")

    return "\n".join(lines), coverage


def _generate_element(
    elem: ExtractedElement,
    storey_idx: int,
    used_ids: set,
    coverage: CoverageReport,
) -> List[str]:
    """Generate code lines for a single element."""
    dsl_type = IFC_TO_DSL_TYPE.get(elem.ifc_type, "Box")
    safe_id = _sanitize_id(elem.name, dsl_type, elem.guid, used_ids)
    used_ids.add(safe_id)

    if dsl_type in CONTAINER_TYPES:
        return _generate_container(dsl_type, safe_id, elem, storey_idx, used_ids, coverage)
    if dsl_type == "Slab":
        return _generate_slab(safe_id, elem, storey_idx, coverage)
    if dsl_type in ("Door", "Window"):
        return _generate_opening_standalone(dsl_type, safe_id, elem, storey_idx, coverage)
    return _generate_solid(safe_id, elem, storey_idx, coverage)


def _emit_body_geometry(
    elem: ExtractedElement,
    body_id: Optional[str] = None,
) -> Tuple[str, str]:
    """Render a Solid (or Solid.cuts(Solid…) / Mesh) expression for an
    element's body. Returns (expression, quality) where quality is
    "exact" | "approximated" | "opaque" so the caller can update the
    CoverageReport. The expression intentionally has no trailing
    semicolon or newline so it can be wrapped by the caller (e.g.
    ``wall.add(<expr>)`` or assigned to a name).
    """
    g = elem.geometry
    id_kw = f', id="{body_id}"' if body_id else ""

    if g.mode == "box" and g.start and g.end:
        return (
            f"Box(start=Point(x={g.start[0]}, y={g.start[1]}, z={g.start[2]}), "
            f"end=Point(x={g.end[0]}, y={g.end[1]}, z={g.end[2]}), "
            f'type="sketch"{id_kw})',
            "exact",
        )

    if g.mode == "contour" and g.contour and g.thickness:
        contour_str = ", ".join(
            f"Point(x={x}, y=0, z={y})" for x, y in g.contour
        )
        return (
            f"Extrude(contour=[{contour_str}], thickness={g.thickness}, "
            f'type="sketch"{id_kw})',
            "exact",
        )

    if g.mode == "boolean" and g.boolean_operands:
        # Use the primary operand as the body, emit the rest as cuts/adds.
        # Operand[0] is the primary; subsequent are the secondary (void)
        # operands. The IFC operator on the parent is captured in
        # ``g.boolean_operation`` ("difference" → cuts, "union" → adds).
        primary = g.boolean_operands[0]
        body_expr, body_quality = _emit_geometry_for_operand(primary, id_kw)
        if body_expr is None:
            return _fallback_body(elem, id_kw)
        ops = g.boolean_operands[1:]
        if not ops:
            # Boolean with only one operand — reconstruct as just the body
            return (body_expr, "approximated" if body_quality == "approximated" else "exact")
        op_strs: List[str] = []
        any_opaque = False
        for op in ops:
            op_expr, op_quality = _emit_geometry_for_operand(op, "", void=True)
            if op_expr is None:
                any_opaque = True
                continue
            op_strs.append(op_expr)
        if not op_strs:
            return (body_expr, "approximated")
        method = {
            "difference": "difference",
            "union": "union",
            "intersection": "intersection",
        }.get(g.boolean_operation, "union")
        joined = ", ".join(op_strs)
        quality = "approximated" if (body_quality == "approximated" or any_opaque) else "exact"
        return (f"{body_expr}.{method}({joined})", quality)

    if g.mode == "mesh" and g.mesh_vertices and g.mesh_faces:
        verts = ", ".join(
            f"Point(x={x}, y={y}, z={z})" for x, y, z in g.mesh_vertices
        )
        faces = ", ".join(f"({a}, {b}, {c})" for a, b, c in g.mesh_faces)
        return (
            f'Mesh(vertices=[{verts}], faces=[{faces}]{id_kw})',
            "exact",
        )

    if g.mode == "mesh" and (g.start and g.end):
        # Mesh with bbox but no triangles — degrade to bbox Solid
        return (
            f"Box(start=Point(x={g.start[0]}, y={g.start[1]}, z={g.start[2]}), "
            f"end=Point(x={g.end[0]}, y={g.end[1]}, z={g.end[2]}), "
            f'type="sketch"{id_kw})  # approximated: brep without face indices',
            "approximated",
        )

    return _fallback_body(elem, id_kw)


def _emit_geometry_for_operand(
    operand, id_kw: str, void: bool = False
) -> Tuple[Optional[str], str]:
    """Render a GeometryInfo as a Solid expression for use inside a
    boolean expression. ``void=True`` marks the operand with
    ``material="Void"`` so the IFC generator emits a true subtraction.
    Returns (expression or None on failure, quality).
    """
    material_kw = ', material="Void"' if void else ""
    if operand.mode == "box" and operand.start and operand.end:
        return (
            f"Box(start=Point(x={operand.start[0]}, y={operand.start[1]}, z={operand.start[2]}), "
            f"end=Point(x={operand.end[0]}, y={operand.end[1]}, z={operand.end[2]}), "
            f'type="sketch"{material_kw}{id_kw})',
            "exact",
        )
    if operand.mode == "contour" and operand.contour and operand.thickness:
        contour_str = ", ".join(
            f"Point(x={x}, y=0, z={y})" for x, y in operand.contour
        )
        return (
            f"Extrude(contour=[{contour_str}], thickness={operand.thickness}, "
            f'type="sketch"{material_kw}{id_kw})',
            "exact",
        )
    if (operand.mode == "mesh" or operand.mode == "boolean") and operand.start and operand.end:
        return (
            f"Box(start=Point(x={operand.start[0]}, y={operand.start[1]}, z={operand.start[2]}), "
            f"end=Point(x={operand.end[0]}, y={operand.end[1]}, z={operand.end[2]}), "
            f'type="sketch"{material_kw}{id_kw})  # approximated: complex operand → bbox',
            "approximated",
        )
    return (None, "opaque")


def _fallback_body(elem: ExtractedElement, id_kw: str) -> Tuple[str, str]:
    """Placement-derived placeholder; marks as opaque so the UI can
    grey out structured-edit affordances on this element."""
    t = elem.placement.translation
    return (
        f"Box(start=Point(x={t[0]}, y={t[1]}, z={t[2]}), "
        f"end=Point(x={t[0]+100}, y={t[1]+100}, z={t[2]+100}), "
        f'type="sketch"{id_kw})  # OPAQUE: original geometry lost',
        "opaque",
    )


def _split_inline_comment(expr: str) -> Tuple[str, Optional[str]]:
    """Split a body expression from a trailing ``  # note`` comment.

    ``_emit_body_geometry`` appends an inline ``# OPAQUE`` / ``# approximated``
    comment to fallback bodies. That is valid on a standalone assignment line
    (``x = Box(...)  # note``) but NOT when the body is inlined inside a call
    (``container.add(Box(...)  # note)``) — the comment swallows the closing
    paren and the whole reconstructed module fails to compile. Callers that
    inline the body must move the note outside the parens.
    """
    if '  # ' in expr:
        code, note = expr.rsplit('  # ', 1)
        return code, note
    return expr, None


def _generate_container(
    dsl_type: str,
    safe_id: str,
    elem: ExtractedElement,
    storey_idx: int,
    used_ids: set,
    coverage: CoverageReport,
) -> List[str]:
    """Generate container element (Wall, Column, Beam, Roof) with body Solid."""
    lines = [f'{safe_id} = {dsl_type}(id="{safe_id}")']

    body_expr, body_quality = _emit_body_geometry(elem)
    coverage.record(elem.ifc_type, body_quality)
    body_code, body_note = _split_inline_comment(body_expr)
    lines.append(f"{safe_id}.add({body_code})" + (f"  # {body_note}" if body_note else ""))

    # Aggregated children (non-opening solids attached via IfcRelAggregates)
    for child in elem.children:
        if child.ifc_type == "IfcOpeningElement":
            continue  # handled separately as wall fills below
        cg = child.geometry
        if cg.mode == "box" and cg.start and cg.end:
            child_id = _sanitize_id(child.name, "Box", child.guid, used_ids)
            used_ids.add(child_id)
            lines.append(
                f'{safe_id}.add(Box('
                f'start=Point(x={cg.start[0]}, y={cg.start[1]}, z={cg.start[2]}), '
                f'end=Point(x={cg.end[0]}, y={cg.end[1]}, z={cg.end[2]}), '
                f'type="sketch", id="{child_id}"))'
            )

    # Opening re-parenting: emit each IfcOpeningElement's fill (Door,
    # Window) as a child of the container, instead of letting them float
    # to the storey level. The pre-pass in reconstruct_litestep already
    # removed these GUIDs from the storey-level list, so this is the
    # canonical emission site for them. Uses the same _emit_body_geometry
    # path as the main body so mesh / boolean / contour fills emit
    # losslessly instead of degrading to the OPAQUE placeholder.
    for opening in elem.children:
        if opening.ifc_type != "IfcOpeningElement":
            continue
        for fill in opening.children:
            if not fill.guid:
                continue
            fill_id = _sanitize_id(fill.name, fill.ifc_type.replace("Ifc", ""), fill.guid, used_ids)
            used_ids.add(fill_id)
            body_expr, quality = _emit_body_geometry(fill, body_id=fill_id)
            coverage.record(fill.ifc_type, quality)
            fill_code, fill_note = _split_inline_comment(body_expr)
            note = f"{fill.ifc_type} fill" + (f" [{fill_note}]" if fill_note else "")
            lines.append(f"{safe_id}.add({fill_code})  # {note}")

    lines.append(f"storey_{storey_idx}.add({safe_id})")
    return lines


def _generate_slab(
    safe_id: str,
    elem: ExtractedElement,
    storey_idx: int,
    coverage: CoverageReport,
) -> List[str]:
    """Generate Slab element as a container aggregating a Box/Extrude body (DSL v8)."""
    g = elem.geometry

    if g.mode == "box" and g.start and g.end:
        coverage.record(elem.ifc_type, "exact")
        return [
            f'{safe_id} = Slab(id="{safe_id}")',
            f'{safe_id}.add(Box('
            f'start=Point(x={g.start[0]}, y={g.start[1]}, z={g.start[2]}), '
            f'end=Point(x={g.end[0]}, y={g.end[1]}, z={g.end[2]}), '
            f'type="sketch"))',
            f"storey_{storey_idx}.add({safe_id})",
        ]

    if g.mode == "contour" and g.contour and g.thickness:
        coverage.record(elem.ifc_type, "exact")
        contour_str = ", ".join(
            f"Point(x={x}, y=0, z={y})" for x, y in g.contour
        )
        return [
            f'{safe_id} = Slab(id="{safe_id}")',
            f'{safe_id}.add(Extrude('
            f'contour=[{contour_str}], '
            f'thickness={g.thickness}, '
            f'type="sketch"))',
            f"storey_{storey_idx}.add({safe_id})",
        ]

    # Opaque fallback
    t = elem.placement.translation
    coverage.record(elem.ifc_type, "opaque")
    return [
        f'{safe_id} = Slab(id="{safe_id}")  # OPAQUE: original geometry lost',
        f'{safe_id}.add(Box('
        f'start=Point(x={t[0]}, y={t[1]}, z={t[2]}), '
        f'end=Point(x={t[0]+100}, y={t[1]+10}, z={t[2]+100}), '
        f'type="sketch"))',
        f"storey_{storey_idx}.add({safe_id})",
    ]


def _generate_opening_standalone(
    dsl_type: str,
    safe_id: str,
    elem: ExtractedElement,
    storey_idx: int,
    coverage: CoverageReport,
) -> List[str]:
    """Standalone Door/Window emit path — only hit when the element is
    NOT a fill of an IfcOpeningElement (those are re-parented onto their
    wall in _generate_container). Emits as a Solid because there's no
    parent wall to attach a real Door/Window to."""
    g = elem.geometry
    if g.mode == "box" and g.start and g.end:
        coverage.record(elem.ifc_type, "approximated")
        return [
            f'# {dsl_type} reconstructed as Box (no parent wall context)',
            f'{safe_id} = Box('
            f'start=Point(x={g.start[0]}, y={g.start[1]}, z={g.start[2]}), '
            f'end=Point(x={g.end[0]}, y={g.end[1]}, z={g.end[2]}), '
            f'type="sketch", id="{safe_id}")',
            f"storey_{storey_idx}.add({safe_id})",
        ]

    t = elem.placement.translation
    coverage.record(elem.ifc_type, "opaque")
    return [
        f'# {dsl_type} reconstructed as Box (OPAQUE: original geometry lost)',
        f'{safe_id} = Box('
        f'start=Point(x={t[0]}, y={t[1]}, z={t[2]}), '
        f'end=Point(x={t[0]+900}, y={t[1]+2100}, z={t[2]+200}), '
        f'type="sketch", id="{safe_id}")',
        f"storey_{storey_idx}.add({safe_id})",
    ]


def _generate_solid(
    safe_id: str,
    elem: ExtractedElement,
    storey_idx: int,
    coverage: CoverageReport,
) -> List[str]:
    """Generate a standalone Solid / Mesh / Boolean element at storey level."""
    body_expr, quality = _emit_body_geometry(elem, body_id=safe_id)
    coverage.record(elem.ifc_type, quality)
    return [
        f"{safe_id} = {body_expr}",
        f"storey_{storey_idx}.add({safe_id})",
    ]


def _wrap_empty_project(import_report: Optional[ImportReport] = None) -> str:
    """Generate minimal empty project code with import notes."""
    lines = ["# Reconstructed from IFC -- no mappable elements found"]
    if import_report:
        lines.extend(_format_report_comments(import_report))
    lines.extend([
        STANDARD_IMPORTS,
        "",
        "",
        "def generate_project():",
        '    proj = Project(name="Imported Project")',
        '    proj.add_storey(Storey(elevation=0))  # anonymous single storey',
        "    return proj",
        "",
        "",
        "result = generate_project()",
        "",
    ])
    return "\n".join(lines)


def _format_report_comments(report: ImportReport) -> List[str]:
    """Format ImportReport as Python comment lines."""
    lines = ["#"]
    if report.mapped_elements or report.total_ifc_products:
        lines.append(
            f"# Mapped {report.mapped_elements} of "
            f"{report.total_ifc_products} IFC products"
        )
    if report.schema != "UNKNOWN":
        schema_note = f"# Schema: {report.schema}"
        if report.schema_rewritten:
            schema_note += " (rewritten for parsing)"
        lines.append(schema_note)
    if report.skipped_by_type:
        lines.append("#")
        lines.append("# Skipped IFC types (not supported in Lite-STEP):")
        for ifc_type, count in sorted(report.skipped_by_type.items()):
            lines.append(f"#   {ifc_type}: {count}")
    if report.notes:
        lines.append("#")
        lines.append("# Notes:")
        for note in report.notes:
            lines.append(f"#   - {note}")
    lines.append("#")
    return lines


def _storey_name(idx: int, elevation_mm: int) -> str:
    """Generate a leaf storey name (DSL v2.1: lowercase ``[a-z0-9_]+``).

    Only used when a reconstructed project has MORE THAN ONE storey (a
    single storey is emitted anonymously — see the emission loop). Multi-storey
    buildings require every storey to be named, so these are leaf-safe.
    """
    if elevation_mm < 0:
        return f"basement_{abs(idx)}"
    if idx == 0:
        return "ground"
    return f"level_{idx}"


def _sanitize_id(
    name: str, dsl_type: str, guid: str, used_ids: set
) -> str:
    """
    Sanitize element name for use as a Python identifier.

    Uses IFC Name if valid, otherwise generates {type}_{guid[:8]}.
    Ensures uniqueness within used_ids.
    """
    if name and name.isidentifier() and not name.startswith("__"):
        candidate = name
    else:
        clean = "".join(c if c.isalnum() or c == "_" else "_" for c in (name or ""))
        if clean and clean[0].isdigit():
            clean = f"_{clean}"
        # Strip leading underscores to avoid dunder names
        clean = clean.lstrip("_") if clean.startswith("__") else clean
        candidate = clean or f"{dsl_type.lower()}_{guid[:8]}"

    # Ensure uniqueness
    if candidate not in used_ids:
        return candidate

    for i in range(1, 1000):
        unique = f"{candidate}_{i}"
        if unique not in used_ids:
            return unique

    return f"{dsl_type.lower()}_{guid[:8]}"
