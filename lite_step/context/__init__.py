"""
Context Module - Common definitions for all users

Contains:
- Precision settings
- Unit definitions

(The legacy bim_catalog profile/material catalog was retired in DSL v17 —
member sections come from ``Material(profile_mm=)``; see
infrastructure/specs/archive/bim-catalog.md for the archived data.)
"""

from .precision import PRECISION
from .units import UNITS

__all__ = [
    "PRECISION",
    "UNITS",
]
