"""What ``color=`` means — one decision, in one place.

A palette name (``"glass"``) and a hex string (``"#FF8800"``) both work on
everything, solids and meshes alike. ``resolve_color`` is the whole
vocabulary; a value it cannot read is a compile ERROR
(``executor._validate_color``) rather than a silent default, because the
failure it replaces was invisible by construction.

Alpha is the last byte: ``"#RRGGBBAA"``, where ``AA`` is OPACITY (``FF``
opaque). IFC stores the complement, so the conversion happens here once —
``transparency = 1 - alpha`` — rather than in each caller, which is how a
half-transparent pane and a 60%-opaque one end up meaning the same thing.
"""
from __future__ import annotations

from typing import Optional, Tuple

#: Named colours. Keys are the closed vocabulary the DSL reference publishes;
#: values are RGB 0-255 (the form ``IfcColourRgb`` emission divides by 255).
ELEMENT_TYPE_COLORS = {
    "site": (132, 148, 168),          # Muted blue-grey
    "foundation": (120, 120, 120),    # Light grey
    "ceiling": (255, 255, 255),       # Warm white
    "floor": (225, 205, 185),         # Natural light beech
    "wall": (245, 242, 225),          # Powder Sand
    "wall-opening": (255, 255, 255),  # White
    "roof": (90, 80, 75),             # Dark brown grey
    "beam": (70, 130, 180),           # Steel blue
    "column": (139, 90, 43),          # Brown (saddle brown)
    "wood_panels": (210, 180, 140),   # Tan/wood colour for floor panels
    "floor_panels": (210, 180, 140),  # Alias of wood_panels
    "slab": (200, 200, 200),          # Default slab grey
    "glass": (176, 196, 222),         # Slight blue tint glass
}

#: Palette keys that carry a transparency of their own. A hex colour states
#: its own alpha instead, so it never consults this.
TRANSPARENT_TYPES = {
    "glass": 0.6,
}

#: Rendered when a mesh names no colour at all — kept here so "no colour" is
#: one value rather than a literal repeated in each backend.
DEFAULT_RGB: Tuple[int, int, int] = (204, 204, 204)

_HEX_DIGITS = set("0123456789abcdefABCDEF")


def _parse_hex(value: str) -> Optional[Tuple[Tuple[int, int, int], float]]:
    """``#RGB`` / ``#RRGGBB`` / ``#RRGGBBAA`` → ``(rgb, transparency)``."""
    body = value[1:]
    if len(body) not in (3, 6, 8) or any(c not in _HEX_DIGITS for c in body):
        return None
    if len(body) == 3:                       # #f80 -> #ff8800
        body = "".join(c * 2 for c in body)
    rgb = (int(body[0:2], 16), int(body[2:4], 16), int(body[4:6], 16))
    # The last byte is OPACITY; IFC stores its complement.
    transparency = 1.0 - int(body[6:8], 16) / 255.0 if len(body) == 8 else 0.0
    return rgb, transparency


def resolve_color(value) -> Optional[Tuple[Tuple[int, int, int], float]]:
    """``(rgb 0-255, transparency 0..1)`` for any accepted ``color=``.

    ``None`` means "this names nothing" — for a value that is itself ``None``
    that is simply "no colour stated", and for a non-empty string it is an
    authoring error the validator turns into a compile failure. Returning the
    same ``None`` for both is deliberate: the CALLER knows which case it is
    (it has the original value), and giving the parser two failure channels
    would put the "is this a mistake?" decision in two places again.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("#"):
        return _parse_hex(text)
    if text in ELEMENT_TYPE_COLORS:
        return ELEMENT_TYPE_COLORS[text], TRANSPARENT_TYPES.get(text, 0.0)
    return None


def color_vocabulary_hint(value) -> str:
    """The remedy half of a refusal — what the author probably meant.

    Split from the message so the emitter and the validator say the same
    thing, and so a new palette key never has to be added to a docstring.
    """
    text = (value or "").strip()
    if text.startswith("#"):
        return ("a hex colour is #RGB, #RRGGBB or #RRGGBBAA (the last byte is "
                "OPACITY — FF opaque, 80 half). Only 0-9a-f are allowed")
    names = ", ".join(sorted(ELEMENT_TYPE_COLORS))
    return (f"use a hex colour (#RRGGBB / #RRGGBBAA) or one of the named "
            f"colours: {names}")
