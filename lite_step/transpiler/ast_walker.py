"""
LibCST AST Walker: Parse Lite-STEP source and build element ID to CST node map.

Uses LibCST (not the standard ast module) to preserve comments, formatting,
and whitespace for round-trip fidelity.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import libcst as cst

logger = logging.getLogger(__name__)

# Known DSL element types that produce IFC products
ELEMENT_TYPES = frozenset({
    "Wall", "Column", "Beam", "Box", "Extrude", "Sweep",
    "Slab", "Roof", "Door", "Window", "Site",
    "Mesh",
    # DSL v8 path primitives
    "Pipe", "Revolve", "Bar", "Element",
})


@dataclass
class LoopContext:
    """Information about an enclosing for-loop."""
    loop_node: cst.For
    loop_id: str                # Hash identifier for this loop
    iterator_var: str           # e.g., "i", "storey"
    range_args: Optional[Tuple[int, ...]] = None  # (start, stop, step) if detectable


@dataclass
class CSTElementNode:
    """A DSL element found in the source code."""
    element_id: str
    element_type: str               # "Wall", "Column", "Solid", etc.
    cst_node: cst.Call              # The Call node creating this element
    loop_context: Optional[LoopContext] = None
    keyword_args: Dict[str, cst.BaseExpression] = field(default_factory=dict)
    # Pattern for loop-generated IDs: "col_" means matches "col_0", "col_1", etc.
    id_pattern: Optional[str] = None


def walk_source(source: str) -> Dict[str, CSTElementNode]:
    """
    Parse Lite-STEP source with LibCST and extract element declarations.

    Returns a dict mapping element identity -> CSTElementNode for every
    element constructor found in the source. Identity is ``name=`` when
    set (v1.5 — it is the emitted IFC Name), else ``id=``
    (Wall(..., name="wall_north") or Wall(..., id="wall_north")).

    For loop-generated identities with f-strings like f"col_{i}",
    the id_pattern field is set to the base prefix (e.g., "col_").
    """
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as e:
        logger.error("Failed to parse source with LibCST: %s", e)
        return {}

    visitor = _ElementVisitor()
    cst.metadata.MetadataWrapper(module).visit(visitor)
    return visitor.elements


class _ElementVisitor(cst.CSTVisitor):
    """Walks the CST to find element constructor calls."""

    def __init__(self):
        self.elements: Dict[str, CSTElementNode] = {}
        self._loop_stack: List[LoopContext] = []

    def visit_For(self, node: cst.For) -> bool:
        """Track entering a for-loop."""
        loop_ctx = _make_loop_context(node)
        if loop_ctx:
            self._loop_stack.append(loop_ctx)
        return True  # Continue visiting children

    def leave_For(self, node: cst.For) -> None:
        """Track leaving a for-loop."""
        if self._loop_stack and self._loop_stack[-1].loop_node is node:
            self._loop_stack.pop()

    def visit_Call(self, node: cst.Call) -> bool:
        """Check if this is an element constructor call."""
        func_name = _get_call_name(node)
        if func_name not in ELEMENT_TYPES:
            return True

        # Extract keyword arguments
        kwargs = {}
        for arg in node.args:
            if arg.keyword is not None:
                kwargs[arg.keyword.value] = arg.value

        # Extract element identity: name= first (v1.5 — it wins as the
        # emitted IFC Name whenever set), id= fallback. When name= is
        # present but not statically extractable (a variable, a call),
        # do NOT fall back to id — the IFC Name will be the runtime name
        # value, so keying on id would bind the patcher to the wrong
        # identity. Skip instead (LLM fallback handles it).
        id_value = None
        id_pattern = None

        name_kwarg = kwargs.get("name")
        if name_kwarg is not None and not _is_none_literal(name_kwarg):
            id_value, id_pattern = _extract_id(name_kwarg)
            if id_value is None and id_pattern is None:
                return True
        elif "id" in kwargs:
            id_value, id_pattern = _extract_id(kwargs["id"])

        if id_value is None and id_pattern is None:
            # No name=/id= keyword or couldn't extract — skip
            return True

        loop_ctx = self._loop_stack[-1] if self._loop_stack else None

        if id_value:
            self.elements[id_value] = CSTElementNode(
                element_id=id_value,
                element_type=func_name,
                cst_node=node,
                loop_context=loop_ctx,
                keyword_args=kwargs,
                id_pattern=id_pattern,
            )
        elif id_pattern:
            # Loop-generated element — register with pattern
            # The actual IDs (col_0, col_1, ...) will be matched during diff
            self.elements[f"__pattern__{id_pattern}"] = CSTElementNode(
                element_id=f"__pattern__{id_pattern}",
                element_type=func_name,
                cst_node=node,
                loop_context=loop_ctx,
                keyword_args=kwargs,
                id_pattern=id_pattern,
            )

        return True


def _is_none_literal(node: cst.BaseExpression) -> bool:
    """True for a literal ``None`` (an explicit ``name=None`` means anonymous —
    the emitted IFC Name falls back to id, so identity extraction must too)."""
    return isinstance(node, cst.Name) and node.value == "None"


def _get_call_name(node: cst.Call) -> str:
    """Extract the function name from a Call node."""
    if isinstance(node.func, cst.Name):
        return node.func.value
    elif isinstance(node.func, cst.Attribute):
        return node.func.attr.value
    return ""


def _extract_id(node: cst.BaseExpression) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract element identity from a name= or id= keyword value.

    Returns (literal_id, pattern) where:
    - literal_id: the exact string value if it's a literal
    - pattern: the base prefix for f-string patterns like f"col_{i}"
    """
    # Simple string literal: id="wall_north"
    if isinstance(node, (cst.SimpleString, cst.ConcatenatedString)):
        value = _string_literal_value(node)
        if value:
            return (value, None)

    # F-string: id=f"col_{i}"
    if isinstance(node, cst.FormattedString):
        return _extract_fstring_pattern(node)

    return (None, None)


def _string_literal_value(node: cst.BaseExpression) -> Optional[str]:
    """Extract the string value from a SimpleString or ConcatenatedString node."""
    if isinstance(node, cst.SimpleString):
        # Remove quotes
        raw = node.value
        if raw.startswith(('"""', "'''")):
            return raw[3:-3]
        elif raw.startswith(('"', "'")):
            return raw[1:-1]
    elif isinstance(node, cst.ConcatenatedString):
        # Try to evaluate concatenated string parts
        parts = []
        for part in (node.left, node.right):
            val = _string_literal_value(part)
            if val is not None:
                parts.append(val)
            else:
                return None  # Can't evaluate
        return "".join(parts)
    return None


def _extract_fstring_pattern(node: cst.FormattedString) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract pattern from f-string like f"col_{i}".

    Returns (None, "col_") — the prefix before the variable part.
    """
    prefix_parts = []

    for part in node.parts:
        if isinstance(part, cst.FormattedStringText):
            prefix_parts.append(part.value)
        elif isinstance(part, cst.FormattedStringExpression):
            # Found the variable part — everything before is the pattern
            pattern = "".join(prefix_parts)
            if pattern:
                return (None, pattern)
            return (None, None)

    # No expression parts — it's just a string
    value = "".join(prefix_parts)
    return (value, None) if value else (None, None)


def _make_loop_context(node: cst.For) -> Optional[LoopContext]:
    """Create a LoopContext from a for-loop CST node."""
    # Extract iterator variable name
    target = node.target
    if isinstance(target, cst.Name):
        var_name = target.value
    elif isinstance(target, cst.Tuple):
        # Tuple unpacking: for i, x in ...
        parts = []
        for el in target.elements:
            if isinstance(el.value, cst.Name):
                parts.append(el.value.value)
        var_name = ",".join(parts)
    else:
        var_name = "__unknown__"

    # Try to detect range() arguments
    range_args = _detect_range_args(node.iter)

    # Generate a stable loop ID from the code position
    loop_id = hashlib.md5(
        f"{var_name}_{_node_code(node.iter)}".encode()
    ).hexdigest()[:12]

    return LoopContext(
        loop_node=node,
        loop_id=loop_id,
        iterator_var=var_name,
        range_args=range_args,
    )


def _detect_range_args(iter_node: cst.BaseExpression) -> Optional[Tuple[int, ...]]:
    """Try to extract (start, stop, step) from a range() call."""
    if not isinstance(iter_node, cst.Call):
        return None
    if not (isinstance(iter_node.func, cst.Name) and iter_node.func.value == "range"):
        return None

    int_args = []
    for arg in iter_node.args:
        if arg.keyword is not None:
            return None  # Named args in range — unusual
        val = _try_int(arg.value)
        if val is None:
            return None  # Non-literal argument
        int_args.append(val)

    if len(int_args) == 1:
        return (0, int_args[0], 1)
    elif len(int_args) == 2:
        return (int_args[0], int_args[1], 1)
    elif len(int_args) == 3:
        return tuple(int_args)
    return None


def _try_int(node: cst.BaseExpression) -> Optional[int]:
    """Try to extract an integer value from an expression."""
    if isinstance(node, cst.Integer):
        return int(node.value)
    if isinstance(node, cst.UnaryOperation) and isinstance(node.operator, cst.Minus):
        val = _try_int(node.expression)
        return -val if val is not None else None
    return None


def _node_code(node: cst.BaseExpression) -> str:
    """Get approximate code string for a CST node (for hashing)."""
    try:
        wrapper = cst.parse_module("")
        return wrapper.code_for_node(node)
    except Exception:
        return str(type(node).__name__)


def is_parametric(node: cst.BaseExpression) -> bool:
    """
    Check if an expression is parametric (contains variables or operations).

    Returns True if the expression cannot be safely replaced with a literal.
    Used as a guard to prevent patching complex expressions (FR-6.6).
    """
    if isinstance(node, (cst.Name,)):
        return True
    if isinstance(node, cst.BinaryOperation):
        return True
    if isinstance(node, cst.UnaryOperation):
        return is_parametric(node.expression)
    if isinstance(node, cst.Call):
        # A Call that's a known constructor (Point, etc.) is not parametric
        name = _get_call_name(node)
        if name in ("Point", "Point2D"):
            # Check if any argument is parametric
            return any(
                is_parametric(arg.value) for arg in node.args
            )
        return True  # Unknown function call = parametric
    if isinstance(node, (cst.Integer, cst.Float, cst.SimpleString)):
        return False
    if isinstance(node, cst.Attribute):
        return True
    if isinstance(node, cst.IfExp):
        return True
    if isinstance(node, cst.Subscript):
        return True
    # For compound expressions (tuples, lists)
    if isinstance(node, (cst.Tuple, cst.List)):
        return any(is_parametric(el.value) for el in node.elements)
    return False


def get_element_ids_for_pattern(
    pattern: str, element_names: List[str]
) -> List[str]:
    """
    Match IFC element names against a loop-generated ID pattern.

    Args:
        pattern: Base prefix from f-string (e.g., "col_")
        element_names: List of IFC element Name values

    Returns:
        Sorted list of names matching the pattern
    """
    matches = []
    for name in element_names:
        if name.startswith(pattern):
            suffix = name[len(pattern):]
            # Suffix should be a valid integer (loop index)
            if suffix.isdigit() or (suffix.startswith("-") and suffix[1:].isdigit()):
                matches.append(name)
    return sorted(matches)
