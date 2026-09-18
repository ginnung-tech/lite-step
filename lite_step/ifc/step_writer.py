"""
Pure Python STEP P21 writer for IFC4 files.

Encodes IFC entities as STEP text lines without requiring ifcopenshell.
Designed for compatibility with the existing deduplicator (_ENTITY_RE, _FLOAT_RE)
and parallel merger (_ENTITY_RE, _REF_RE) regex patterns.

STEP P21 encoding rules (ISO 10303-21):
    Entity line:    #ID=TYPE(attr1,attr2,...);
    Null:           $
    Derived:        *
    Boolean:        .T. / .F.
    Integer:        literal (no decimal point)
    Real:           always has decimal point (0., 3.5, 1.E-05)
    String:         'single-quoted' with apostrophe doubling
    Enum:           .ENUMVALUE.
    Entity ref:     #N
    List:           (item,item,...)
    Empty list:     ()
"""

import re
import uuid
from dataclasses import dataclass
from lite_step.ifc.schema_version import IFC_OUTPUT_SCHEMA
from datetime import datetime, timezone
from typing import List, Sequence, Tuple, Union


# ============================================================================
# STEP value type markers
# ============================================================================

class _StepSentinel:
    """Singleton marker for STEP special values."""
    __slots__ = ('_symbol',)

    def __init__(self, symbol: str):
        self._symbol = symbol

    def __repr__(self) -> str:
        return self._symbol


STEP_NULL = _StepSentinel('$')
STEP_DERIVED = _StepSentinel('*')


@dataclass(frozen=True, slots=True)
class StepEnum:
    """STEP enumeration value, encoded as .VALUE. in output."""
    value: str


@dataclass(frozen=True, slots=True)
class StepRef:
    """Reference to another entity by ID, encoded as #N."""
    entity_id: int


@dataclass(frozen=True, slots=True)
class StepTyped:
    """STEP typed SELECT value, encoded as TYPENAME(inner_value).

    Used for IFC property values: IFCTEXT('...'), IFCLABEL('...'),
    IFCINTEGER(42), IFCBOOLEAN(.T.), IFCLENGTHMEASURE(1.5), etc.
    """
    type_name: str
    inner: object  # any StepValue — encoded via encode_value()


# Public type alias
StepValue = Union[
    None, bool, int, float, str,
    StepEnum, StepRef,
    list, tuple,
    _StepSentinel,
]


# ============================================================================
# IFC GUID generation (pure Python)
# ============================================================================

_GUID_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$'


def new_guid() -> str:
    """Generate a 22-character IFC GloballyUniqueId from a random UUID4.

    Encoding: 128-bit UUID -> 22 base-64 chars using the IFC alphabet.
    First character is always 0-3 (only 2 bits of the top 4 are used).
    """
    n = uuid.uuid4().int
    chars = []
    for _ in range(22):
        chars.append(_GUID_CHARS[n & 63])
        n >>= 6
    return ''.join(reversed(chars))


# ============================================================================
# ID Allocator
# ============================================================================

class IdAllocator:
    """Sequential entity ID counter with range reservation for parallel work."""

    __slots__ = ('_next',)

    def __init__(self, start: int = 1):
        self._next = start

    def next_id(self) -> int:
        eid = self._next
        self._next += 1
        return eid

    def reserve_range(self, count: int) -> Tuple[int, int]:
        """Reserve a contiguous ID block. Returns (start_inclusive, end_exclusive)."""
        start = self._next
        self._next += count
        return start, self._next

    @property
    def current(self) -> int:
        """Next ID that would be allocated."""
        return self._next

    def skip_to(self, target: int) -> None:
        """Advance counter to at least *target* (for range alignment)."""
        if target > self._next:
            self._next = target


# ============================================================================
# Value encoding
# ============================================================================

def encode_value(value: StepValue) -> str:
    """Encode a Python value to STEP P21 text."""
    if value is None or value is STEP_NULL:
        return '$'
    if value is STEP_DERIVED:
        return '*'
    # NumPy scalar coercion — load-bearing. np.float64 IS a subclass of
    # float, so it would slip into the float branch below; but on numpy>=2.0
    # repr(np.float64(-0.7)) == 'np.float64(-0.7)' (not '-0.7'), which writes
    # invalid STEP P21 (e.g. IFCCARTESIANPOINT((np.float64(-0.7),...))).
    # web-ifc then mis-parses the coordinate, the geometry transform blows up,
    # and ThatOpen Fragments silently skips the element as ">100000 m from
    # origin" — the whole building renders blank. Coercing every numpy scalar
    # to its native Python type here means no upstream call site (contour math,
    # np.dot projections, mm→m division) can leak it into the wire. Duck-typed
    # so step_writer keeps no hard numpy dependency.
    if type(value).__module__ == 'numpy' and hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, bool):          # before int — bool is subclass
        return '.T.' if value else '.F.'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return encode_real(value)
    if isinstance(value, str):
        return encode_string(value)
    if isinstance(value, StepEnum):
        return f'.{value.value}.'
    if isinstance(value, StepRef):
        return f'#{value.entity_id}'
    if isinstance(value, StepTyped):
        return f'{value.type_name}({encode_value(value.inner)})'
    if isinstance(value, (list, tuple)):
        return '(' + ','.join(encode_value(v) for v in value) + ')'
    if isinstance(value, _StepSentinel):
        return value._symbol
    raise TypeError(f"Cannot encode {type(value).__name__}: {value!r}")


def encode_real(value: float) -> str:
    """Encode float to STEP P21 real number.

    Always contains a decimal point.  Matches IfcOpenShell output closely
    enough for deduplicator float-normalization compatibility.

    Examples::

        0.0   -> '0.'
        3.0   -> '3.'
        3.5   -> '3.5'
       -1.2   -> '-1.2'
        1e-5  -> '1.E-05'
        0.15  -> '0.15'
    """
    if value == 0.0:
        return '0.'

    # float() coercion guards direct callers passing a numpy scalar:
    # repr(np.float64(x)) == 'np.float64(x)' on numpy>=2.0, which is invalid
    # STEP. encode_value coerces upstream too; this is belt-and-braces.
    s = repr(float(value))

    # Scientific notation: Python '1e-05' -> STEP '1.E-05'
    if 'e' in s or 'E' in s:
        parts = s.lower().split('e')
        mantissa = parts[0]
        exp = int(parts[1])

        if '.' in mantissa:
            mantissa = mantissa.rstrip('0')
            # keep trailing dot: '1.' not '1'
        else:
            mantissa += '.'

        sign = '-' if exp < 0 else '+'
        return f'{mantissa}E{sign}{abs(exp):02d}'

    # Regular decimal: ensure dot, strip trailing zeros
    if '.' not in s:
        return s + '.'

    s = s.rstrip('0')
    # s ends with '.' or has digits after dot — both valid
    return s


#: One STEP token: either a single-quoted string (apostrophes doubled to
#: escape) OR a REAL literal. Alternating so the string branch consumes
#: quoted content first — a real-looking substring inside a name/GUID
#: (``'v2.4'``) is never rewritten. A STEP REAL always carries a decimal
#: point (integers/refs/enums don't), so the number branch matches only
#: ``digits '.' digits* [exponent]`` and leaves counts (``3``) and refs
#: (``#33``) untouched.
_STEP_TOKEN_RE = re.compile(
    r"'(?:[^']|'')*'"                       # quoted string (skip)
    r"|(?<![\w#])(-?\d+\.\d*(?:[eE][+-]?\d+)?)"  # REAL literal (rewrite)
)


def normalize_step_floats(step: str) -> str:
    """Re-encode every STEP REAL to its shortest round-tripping form.

    IfcOpenShell's C++ serializer emits raw full-precision doubles
    (``2.3999999999999999``, ``1.0000000000000001E-05``); this rewrites each
    through Python's shortest-``repr`` STEP encoder (:func:`encode_real`), so
    ``2.3999999999999999 -> 2.4`` and ``1.0000000000000001E-05 -> 1.E-05``.

    Bit-exact: each token is parsed to the identical ``float`` and re-emitted,
    so geometry is unchanged — purely cosmetic + a file-size win. Quoted strings and GUIDs are never touched (the string branch of
    the token regex consumes them first). Idempotent: already-short reals
    round-trip to themselves, so it is safe to run over any emitted output.
    """
    def _repl(m: "re.Match[str]") -> str:
        num = m.group(1)
        if num is None:            # matched a quoted string — leave verbatim
            return m.group(0)
        return encode_real(float(num))

    return _STEP_TOKEN_RE.sub(_repl, step)


def encode_string(value: str) -> str:
    """Encode to STEP P21 string: single-quoted with doubled apostrophes."""
    return "'" + value.replace("'", "''") + "'"


# ============================================================================
# Entity line formatting
# ============================================================================

def format_entity(entity_id: int, entity_type: str, attributes: Sequence) -> str:
    """Format a STEP entity line: ``#ID=TYPE(attr1,attr2,...);``"""
    encoded = ','.join(encode_value(a) for a in attributes)
    return f'#{entity_id}={entity_type}({encoded});'


# ============================================================================
# File structure
# ============================================================================

def step_header(
    description: str = 'ViewDefinition [CoordinationView]',
    schema: str = IFC_OUTPUT_SCHEMA,
    originating_system: str = 'Lite-STEP Compiler',
) -> List[str]:
    """Generate STEP P21 HEADER section lines (including ISO preamble and DATA start)."""
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    return [
        'ISO-10303-21;',
        'HEADER;',
        f"FILE_DESCRIPTION(('{description}'),'2;1');",
        f"FILE_NAME('','{ts}',(''),(''),'{originating_system}','{originating_system}','');",
        f"FILE_SCHEMA(('{schema}'));",
        'ENDSEC;',
        'DATA;',
    ]


def step_footer() -> List[str]:
    """Generate STEP P21 footer lines."""
    return [
        'ENDSEC;',
        'END-ISO-10303-21;',
    ]


def assemble_step_file(
    header_lines: List[str],
    data_lines: List[str],
    footer_lines: List[str],
) -> str:
    """Join header, data entities, and footer into a complete STEP P21 file string."""
    return '\n'.join(header_lines + data_lines + footer_lines) + '\n'
