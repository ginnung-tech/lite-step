"""
Unit Definitions for Lite-STEP

Defines the unit system used in Lite-STEP models.
Common for all users.
"""

from dataclasses import dataclass
from enum import Enum


class LengthUnit(str, Enum):
    """Supported length units."""
    METRE = "METRE"
    MILLIMETRE = "MILLIMETRE"
    CENTIMETRE = "CENTIMETRE"
    FOOT = "FOOT"
    INCH = "INCH"


class AreaUnit(str, Enum):
    """Supported area units."""
    SQUARE_METRE = "SQUARE_METRE"
    SQUARE_MILLIMETRE = "SQUARE_MILLIMETRE"
    SQUARE_FOOT = "SQUARE_FOOT"


class VolumeUnit(str, Enum):
    """Supported volume units."""
    CUBIC_METRE = "CUBIC_METRE"
    CUBIC_MILLIMETRE = "CUBIC_MILLIMETRE"
    CUBIC_FOOT = "CUBIC_FOOT"


class AngleUnit(str, Enum):
    """Supported angle units."""
    RADIAN = "RADIAN"
    DEGREE = "DEGREE"


@dataclass
class UnitSettings:
    """Unit settings for a Lite-STEP model."""
    length: LengthUnit = LengthUnit.MILLIMETRE
    area: AreaUnit = AreaUnit.SQUARE_METRE
    volume: VolumeUnit = VolumeUnit.CUBIC_METRE
    angle: AngleUnit = AngleUnit.RADIAN

    # Conversion factors to SI base units (metres, radians)
    @property
    def length_to_metres(self) -> float:
        """Get conversion factor from current length unit to metres."""
        factors = {
            LengthUnit.METRE: 1.0,
            LengthUnit.MILLIMETRE: 0.001,
            LengthUnit.CENTIMETRE: 0.01,
            LengthUnit.FOOT: 0.3048,
            LengthUnit.INCH: 0.0254,
        }
        return factors[self.length]

    @property
    def angle_to_radians(self) -> float:
        """Get conversion factor from current angle unit to radians."""
        import math
        factors = {
            AngleUnit.RADIAN: 1.0,
            AngleUnit.DEGREE: math.pi / 180.0,
        }
        return factors[self.angle]


# Default unit settings (millimetres, radians - typical for IFC)
UNITS = UnitSettings()
