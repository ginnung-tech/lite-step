"""The ``IfcBuildingStorey`` ATTRIBUTES — Name, LongName, CompositionType, Elevation.

Every policy decision about what a storey entity SAYS is decided HERE;
``generator.py`` contributes only the lines that mint an entity. Two
implementations of one idea agreeing *usually* is what produces the bug.

``Elevation`` is OPTIONAL in IFC4X3_ADD2, so emitting ``$`` for it is
schema-valid and no validator or byte-hash can notice the omission. The file
still renders, because geometry is authored in world coordinates; what breaks
is every consumer that READS storey elevation instead of re-deriving it from a
placement chain — storey schedules, section and plan generation, storey
filters, floor-to-floor heights. A dropped OPTIONAL has no gate except a test
that asserts we said something.

The rules
---------

**Elevation is ALWAYS stated, including when it is zero.** ``$`` means "not
stated"; ``0.`` means "at the project datum". The DSL always carries a
concrete number (``Storey.elevation`` defaults to ``0.0``), so there is never
a storey whose elevation we do not know, and emitting ``$`` for the datum
storey would make "at zero" indistinguishable from "unknown" — the same
silent-degradation class the bug came from. This is why the whole corpus
drifts on this change even though every committed model is single-storey at
elevation 0.

**Elevation is the storey placement's Z, not a second derivation of it.**
``IfcBuildingStorey.Elevation`` is an ``IfcLengthMeasure`` in the project
length unit (METRE — ``run("unit.assign_unit", …, raw="METRE")`` on one
backend, the hand-written ``IFCSIUNIT(…,.METRE.)`` on the other), and the emitter already place the storey at ``(0, 0, storey.elevation)`` relative to
the building. :func:`elevation_of` returns that exact same float, so a model
whose ``Elevation`` and whose placement disagree is impossible by
construction rather than by agreement.

**The unit conversion happens ONCE, upstream.**
``lite_step.compiler.executor.normalize_project_to_meters`` divides
``Storey.elevation`` by 1000 (mm at authoring time → m after compilation; see
``Storey``'s docstring). The emitter runs on the normalized project, so this
module does NO conversion — it must not, or the value would be divided twice
on the compile path and once on the direct-``generate_ifc`` test path.

**LongName stays ``$``, deliberately.** IFC's convention is ``Name`` = the
short code, ``LongName`` = the full human name ("L1" / "Level 1 — Ground
Floor"). The DSL has exactly ONE authored string for a storey
(``Storey(name="ground")``), and ``Name`` already carries it as the stamped
canonical (``storey:ground``). Putting ``"ground"`` in ``LongName`` too would
be a second spelling of the same fact, free to drift from the first, and
``LongName`` would still not be the thing IFC means by it. Inventing
``"Level 1"`` from the storey index would be worse: data no author wrote.
Until the DSL grows a distinct human title, ``$`` is the honest answer. Pinned
by a test so it stays a decision rather than an oversight.

**CompositionType stays ``$``.** It is DEPRECATED in IFC4 (``IfcElementCompositionEnum``
on ``IfcSpatialStructureElement``); emitting it would be re-introducing a
retired attribute. Also pinned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class StoreyAttributes:
    """What one ``IfcBuildingStorey`` says about itself.

    Positional-tail order matches the entity: ``…, LongName,
    CompositionType, Elevation`` are attributes 8, 9, 10 (1-based), and
    ``name`` is attribute 3.
    """

    name: Optional[str]              #: IfcRoot.Name — the stamped canonical
    long_name: Optional[str]         #: IfcSpatialStructureElement.LongName
    composition_type: Optional[str]  #: DEPRECATED in IFC4 — always None
    elevation: float                 #: IfcBuildingStorey.Elevation, in METRES


def elevation_of(storey: Any) -> float:
    """The storey's elevation in the project length unit (metres).

    The SAME float the emitter puts in the storey's ``ObjectPlacement``
    ``RelativePlacement.Location`` Z, so ``Elevation`` and the placement can
    never disagree.

    No unit conversion: ``normalize_project_to_meters`` already did it
    (mm → m). ``float()`` is a coercion, not a conversion — a numpy scalar
    reaching the STEP writer serialises as ``np.float64(3.0)`` on numpy>=2,
    which is invalid P21 (see ``step_writer.encode_real``).
    """
    return float(getattr(storey, "elevation", 0.0) or 0.0)


def attributes_of(storey: Any) -> StoreyAttributes:
    """Every attribute of the ``IfcBuildingStorey`` for ``storey``.

    Read by the emitter. See the module docstring for why ``long_name`` and
    ``composition_type`` are ``None`` on purpose.
    """
    return StoreyAttributes(
        name=getattr(storey, "_canonical_name", None),
        long_name=None,
        composition_type=None,
        elevation=elevation_of(storey),
    )
