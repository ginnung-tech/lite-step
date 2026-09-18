"""
CST Patcher: Apply diffs to Lite-STEP source via LibCST transformers.

Handles PLACEMENT_MOVED, GEOMETRY_RESIZED, DELETED, ADDED elements.
Guards against patching parametric expressions (FR-6.6).
Preserves all comments, formatting, and helper functions.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import libcst as cst

from lite_step.ifc.normalizer import ExtractedElement
from lite_step.transpiler.ast_walker import (
    CSTElementNode, ELEMENT_TYPES, is_parametric, _get_call_name,
)
from lite_step.transpiler.differ import DeltaCategory, ElementDelta
from lite_step.transpiler.loop_analysis import LoopAnalysis

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


def apply_patches(
    source: str,
    deltas: List[ElementDelta],
    loop_analyses: List[LoopAnalysis],
    cst_nodes: Dict[str, CSTElementNode],
) -> Tuple[str, List[ElementDelta]]:
    """
    Apply diffs to source via LibCST transformers.

    Returns:
        (patched_source, unhandled_deltas) — unhandled deltas need LLM fallback
    """
    module = cst.parse_module(source)

    # Build lookup maps
    delta_by_id = {d.element_id: d for d in deltas}
    loop_by_id = {la.loop_id: la for la in loop_analyses}

    # Determine which loops to delete
    delete_loop_ids = {
        la.loop_id for la in loop_analyses if la.action == "delete"
    }

    # Determine which elements to delete (non-loop)
    delete_element_ids = {
        d.element_id for d in deltas
        if d.category == DeltaCategory.DELETED
    }
    # Remove elements handled by loop deletion
    for la in loop_analyses:
        if la.action == "delete":
            delete_element_ids -= set(la.element_ids)

    # Elements needing LLM fallback
    llm_fallback = []
    for la in loop_analyses:
        if la.action == "llm_fallback":
            for eid in la.element_ids:
                if eid in delta_by_id:
                    llm_fallback.append(delta_by_id[eid])

    # Build replacement map: element_id -> new Point values for patching
    patch_map: Dict[str, Dict[str, Tuple[int, int, int]]] = {}
    too_complex: List[ElementDelta] = []

    for delta in deltas:
        if delta.category in (DeltaCategory.UNCHANGED, DeltaCategory.DELETED, DeltaCategory.ADDED):
            continue

        eid = delta.element_id
        node = cst_nodes.get(eid)
        if node is None:
            continue

        # Skip elements in LLM-fallback loops
        if node.loop_context:
            loop_id = node.loop_context.loop_id
            if loop_id in loop_by_id and loop_by_id[loop_id].action == "llm_fallback":
                continue

        if delta.category in (DeltaCategory.PLACEMENT_MOVED, DeltaCategory.GEOMETRY_RESIZED):
            if delta.modified and delta.modified.geometry.start and delta.modified.geometry.end:
                # Check for parametric expressions
                has_parametric = False
                for kwarg in ("start", "end"):
                    if kwarg in node.keyword_args:
                        if is_parametric(node.keyword_args[kwarg]):
                            has_parametric = True
                            break

                if has_parametric:
                    too_complex.append(delta)
                    continue

                patch_map[eid] = {
                    "start": delta.modified.geometry.start,
                    "end": delta.modified.geometry.end,
                }

    llm_fallback.extend(too_complex)

    # Apply transformations
    transformer = _DeltaPatcher(
        patch_map=patch_map,
        delete_element_ids=delete_element_ids,
        delete_loop_ids=delete_loop_ids,
        cst_nodes=cst_nodes,
    )
    modified_tree = module.visit(transformer)
    patched_source = modified_tree.code

    # Append new elements for ADDED deltas
    added = [d for d in deltas if d.category == DeltaCategory.ADDED]
    if added:
        new_code = _generate_added_elements(added)
        if new_code:
            patched_source = patched_source.rstrip() + "\n\n" + new_code + "\n"

    return patched_source, llm_fallback


class _DeltaPatcher(cst.CSTTransformer):
    """LibCST transformer that patches element constructors in place."""

    def __init__(
        self,
        patch_map: Dict[str, Dict[str, Tuple[int, int, int]]],
        delete_element_ids: Set[str],
        delete_loop_ids: Set[str],
        cst_nodes: Dict[str, CSTElementNode],
    ):
        self.patch_map = patch_map
        self.delete_element_ids = delete_element_ids
        self.delete_loop_ids = delete_loop_ids
        self.cst_nodes = cst_nodes
        # Track which loop nodes to delete
        self._loop_nodes_to_delete: Set[int] = set()
        for node in cst_nodes.values():
            if node.loop_context and node.loop_context.loop_id in delete_loop_ids:
                self._loop_nodes_to_delete.add(id(node.loop_context.loop_node))

    def leave_Call(
        self, original_node: cst.Call, updated_node: cst.Call
    ) -> cst.BaseExpression:
        """Patch element constructor Point arguments."""
        func_name = _get_call_name(updated_node)
        if func_name not in ELEMENT_TYPES:
            return updated_node

        # Find element ID from this call
        eid = _extract_id_from_call(updated_node)
        if eid is None:
            return updated_node

        # Check if this element needs patching
        if eid not in self.patch_map:
            return updated_node

        patches = self.patch_map[eid]
        result = updated_node

        for kwarg_name, new_values in patches.items():
            result = _replace_point_kwarg(result, kwarg_name, new_values)

        return result

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> Union[cst.SimpleStatementLine, cst.RemovalSentinel]:
        """Remove statements that create deleted elements."""
        if self._statement_creates_deleted_element(updated_node):
            return cst.RemovalSentinel.REMOVE
        return updated_node

    def leave_For(
        self, original_node: cst.For, updated_node: cst.For
    ) -> Union[cst.For, cst.RemovalSentinel]:
        """Remove entire for-loops whose elements are all deleted."""
        if id(original_node) in self._loop_nodes_to_delete:
            return cst.RemovalSentinel.REMOVE
        return updated_node

    def _statement_creates_deleted_element(
        self, stmt: cst.SimpleStatementLine
    ) -> bool:
        """Check if a statement contains a call creating a deleted element."""
        for body_item in stmt.body:
            calls = _find_calls_in_statement(body_item)
            for call in calls:
                eid = _extract_id_from_call(call)
                if eid and eid in self.delete_element_ids:
                    return True
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_id_from_call(node: cst.Call) -> Optional[str]:
    """Extract element identity from a Call node: name= first, id= fallback.

    Mirrors the walker (v1.5): ``name=`` is the emitted IFC Name whenever
    set, so it is the identity the differ produced deltas for. A ``name=``
    that exists but is not a static string literal (variable, f-string,
    explicit None) falls through to ``id=`` only when it is a literal
    ``None`` — otherwise identity is unresolvable here and we return None.
    """
    name_arg = None
    id_arg = None
    for arg in node.args:
        if arg.keyword and arg.keyword.value == "name":
            name_arg = arg.value
        elif arg.keyword and arg.keyword.value == "id":
            id_arg = arg.value

    def _literal(value) -> Optional[str]:
        if isinstance(value, cst.SimpleString):
            raw = value.value
            if raw.startswith(('"""', "'''")):
                return raw[3:-3]
            return raw[1:-1]
        return None

    if name_arg is not None:
        if isinstance(name_arg, cst.Name) and name_arg.value == "None":
            pass  # explicit name=None → anonymous, id is the identity
        else:
            return _literal(name_arg)
    if id_arg is not None:
        return _literal(id_arg)
    return None


def _find_calls_in_statement(node: cst.BaseSmallStatement) -> List[cst.Call]:
    """Find all Call nodes in a statement (handles assignment, expression, etc.)."""
    calls = []

    class _CallFinder(cst.CSTVisitor):
        def visit_Call(self, n: cst.Call) -> bool:
            calls.append(n)
            return True

    try:
        cst.metadata.MetadataWrapper(cst.parse_module(""))
    except Exception:
        pass

    # Simple approach: walk the statement
    if isinstance(node, cst.Expr):
        _walk_for_calls(node.value, calls)
    elif isinstance(node, (cst.Assign, cst.AnnAssign)):
        if hasattr(node, "value") and node.value:
            _walk_for_calls(node.value, calls)

    return calls


def _walk_for_calls(node: cst.BaseExpression, calls: List[cst.Call]) -> None:
    """Recursively find Call nodes in an expression."""
    if isinstance(node, cst.Call):
        calls.append(node)
        for arg in node.args:
            _walk_for_calls(arg.value, calls)
    elif isinstance(node, cst.Attribute):
        _walk_for_calls(node.value, calls)
    elif isinstance(node, cst.BinaryOperation):
        _walk_for_calls(node.left, calls)
        _walk_for_calls(node.right, calls)


def _replace_point_kwarg(
    call: cst.Call, kwarg_name: str, values: Tuple[int, int, int]
) -> cst.Call:
    """Replace a Point(...) keyword argument with new coordinates."""
    new_args = []
    replaced = False

    for arg in call.args:
        if arg.keyword and arg.keyword.value == kwarg_name:
            # Build new Point(x=..., y=..., z=...) node
            new_point = _make_point_call(values)
            new_args.append(arg.with_changes(value=new_point))
            replaced = True
        else:
            new_args.append(arg)

    if replaced:
        return call.with_changes(args=new_args)
    return call


def _make_point_call(values: Tuple[int, int, int]) -> cst.Call:
    """Create a Point(x=..., y=..., z=...) CST node."""
    return cst.Call(
        func=cst.Name("Point"),
        args=[
            cst.Arg(
                keyword=cst.Name("x"),
                value=cst.Integer(str(values[0])),
                equal=cst.AssignEqual(
                    whitespace_before=cst.SimpleWhitespace(""),
                    whitespace_after=cst.SimpleWhitespace(""),
                ),
            ),
            cst.Arg(
                keyword=cst.Name("y"),
                value=cst.Integer(str(values[1])),
                equal=cst.AssignEqual(
                    whitespace_before=cst.SimpleWhitespace(""),
                    whitespace_after=cst.SimpleWhitespace(""),
                ),
            ),
            cst.Arg(
                keyword=cst.Name("z"),
                value=cst.Integer(str(values[2])),
                equal=cst.AssignEqual(
                    whitespace_before=cst.SimpleWhitespace(""),
                    whitespace_after=cst.SimpleWhitespace(""),
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# New element generation (ADDED)
# ---------------------------------------------------------------------------

def _generate_added_elements(deltas: List[ElementDelta]) -> str:
    """Generate Lite-STEP code for ADDED elements."""
    lines = ["# --- Elements added by external editor ---"]

    for delta in deltas:
        elem = delta.modified
        if elem is None:
            continue

        code = _element_to_code(elem)
        if code:
            lines.append(code)

    return "\n".join(lines) if len(lines) > 1 else ""


def _element_to_code(elem: ExtractedElement) -> str:
    """Generate Lite-STEP code for a single extracted element."""
    dsl_type = IFC_TO_DSL_TYPE.get(elem.ifc_type, "Box")
    safe_id = _sanitize_id(elem.name, dsl_type)

    g = elem.geometry

    if dsl_type in ("Wall", "Column", "Beam", "Roof"):
        # Container types: create container + body Box/Extrude
        return _generate_container_code(dsl_type, safe_id, g, elem.storey_idx)

    if g.mode == "box" and g.start and g.end:
        return (
            f'{safe_id} = Box('
            f'start=Point(x={g.start[0]}, y={g.start[1]}, z={g.start[2]}), '
            f'end=Point(x={g.end[0]}, y={g.end[1]}, z={g.end[2]}), '
            f'type="sketch", id="{safe_id}"'
            f')  # added by external editor'
        )

    if g.mode == "contour" and g.contour and g.thickness:
        contour_str = ", ".join(
            f"Point(x={x}, y=0, z={y})" for x, y in g.contour
        )
        return (
            f'{safe_id} = Extrude('
            f'contour=[{contour_str}], '
            f'thickness={g.thickness}, '
            f'type="sketch", id="{safe_id}"'
            f')  # added by external editor'
        )

    # Fallback: use placement as position
    t = elem.placement.translation
    return (
        f'{safe_id} = Box('
        f'start=Point(x={t[0]}, y={t[1]}, z={t[2]}), '
        f'end=Point(x={t[0]+100}, y={t[1]+100}, z={t[2]+100}), '
        f'type="sketch", id="{safe_id}"'
        f')  # added by external editor (geometry approximated)'
    )


def _generate_container_code(
    dsl_type: str, safe_id: str, g, storey_idx: int
) -> str:
    """Generate container element code (Wall, Column, Beam, Roof)."""
    lines = [f'{safe_id} = {dsl_type}(id="{safe_id}")  # added by external editor']

    if g.mode == "box" and g.start and g.end:
        lines.append(
            f'{safe_id}.add(Box('
            f'start=Point(x={g.start[0]}, y={g.start[1]}, z={g.start[2]}), '
            f'end=Point(x={g.end[0]}, y={g.end[1]}, z={g.end[2]}), '
            f'type="sketch"))'
        )

    return "\n".join(lines)


def _sanitize_id(name: str, dsl_type: str) -> str:
    """Sanitize an element name for use as a Python identifier."""
    if name and name.isidentifier() and not name.startswith("__"):
        return name
    # Generate from type + truncated name
    clean = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if clean and clean[0].isdigit():
        clean = f"{dsl_type.lower()}_{clean}"
    return clean or f"{dsl_type.lower()}_imported"
