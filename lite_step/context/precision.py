"""
Precision Settings for Lite-STEP

Defines precision and tolerance settings for geometric operations.
Common for all users.
"""

from dataclasses import dataclass


@dataclass
class PrecisionSettings:
    """Precision settings for geometric calculations."""
    # Number of decimal places for coordinates
    coordinate_precision: int = 6

    # Tolerance for comparing floating point values (in model units)
    comparison_tolerance: float = 1e-6

    # Tolerance for geometric operations (in model units)
    geometric_tolerance: float = 1e-5

    # Angular precision in radians
    angular_precision: float = 1e-8


# Default precision settings
PRECISION = PrecisionSettings()
