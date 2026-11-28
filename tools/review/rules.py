"""Which classes matter, and which contact is by design. Every row says why; a row is added only when a clean
model (the skills corpus, the worked examples) shows the contact is meant, never to quiet a real defect."""

from __future__ import annotations

# Elements that carry load: a clash or a missing support between two of them is an error, not a warning.
STRUCTURAL_CLASSES = frozenset({
    "IfcBeam", "IfcColumn", "IfcMember", "IfcSlab", "IfcWall", "IfcFooting", "IfcPile", "IfcCaissonFoundation",
    "IfcPlate", "IfcStair", "IfcRamp", "IfcBearing", "IfcTendon", "IfcCableCarrierSegment",
})
# IfcRoof is not in it: an IfcRoof with a body of its own is the covering (tiles on battens), and what carries it
# is modelled as members and slabs, which are.

# Bars that may cross mid-length as an X of bracing: reported as a crossing (warning), never as an error.
CROSSING_CLASSES = frozenset({"IfcMember"})

# Walls meet at their ends (a corner, a T): a lap inside the outer fifth of a wall's length is a joint.
WALL_CLASSES = frozenset({"IfcWall", "IfcWallStandardCase"})

# A gap at one of these is a joint that should close: a column that stops short of its beam.
NEAR_MISS_CLASSES = frozenset({"IfcBeam", "IfcColumn", "IfcMember"})

# Whatever touches one of these stands on the ground.
GROUND_CLASSES = frozenset({
    "IfcGeographicElement",  # terrain
    "IfcFooting", "IfcPile", "IfcCaissonFoundation",  # foundations sit in the ground they are drawn above
    "IfcEarthworksFill", "IfcEarthworksCut",
})

# Class pairs whose shared volume or gap is by design, in either order.
_BY_DESIGN: dict[frozenset[str], str] = {
    # a bar or tendon runs inside the concrete it reinforces
    frozenset({"IfcReinforcingBar", "IfcColumn"}): "reinforcement inside its host",
    frozenset({"IfcReinforcingBar", "IfcBeam"}): "reinforcement inside its host",
    frozenset({"IfcReinforcingBar", "IfcSlab"}): "reinforcement inside its host",
    frozenset({"IfcReinforcingBar", "IfcWall"}): "reinforcement inside its host",
    frozenset({"IfcReinforcingBar", "IfcFooting"}): "reinforcement inside its host",
    frozenset({"IfcTendon", "IfcColumn"}): "a tendon runs inside its host",
    frozenset({"IfcTendon", "IfcBeam"}): "a tendon runs inside its host",
    # a pile or caisson head is cast into the cap or footing it carries
    frozenset({"IfcPile", "IfcFooting"}): "a pile head is embedded in its cap",
    frozenset({"IfcCaissonFoundation", "IfcFooting"}): "a caisson head is embedded in its footing",
    # a skewback or deck seat is cast around what it receives: the arch foot, the end floor beam
    frozenset({"IfcFooting", "IfcMember"}): "a member seated in its footing or seat",
    frozenset({"IfcFooting", "IfcColumn"}): "a column foot set into its footing",
}


# Ground and water: whatever stands in them is set into them (a pier into the valley side, a bank into the
# river), so contact or shared volume with one is never a clash. Floating is still found through them.
GROUND_CONTACT_CLASSES = frozenset({"IfcGeographicElement", "IfcEarthworksFill", "IfcEarthworksCut"})


def contact_by_design(a: str, b: str) -> bool:
    if a in GROUND_CONTACT_CLASSES or b in GROUND_CONTACT_CLASSES:
        return True
    return frozenset({a, b}) in _BY_DESIGN
