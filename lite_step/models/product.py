"""``Product`` — promotion of a built element into a catalog item (roadmap §V.5).

A Product is a **finished, reusable catalog item**: a window unit, a rafter cut
to spec. It is not declared abstractly — a concrete element is authored
bottom-up with the full DSL and then PROMOTED::

    win    = build_window(width=1400, height=1600)     # authored, UNPLACED
    NORDIC = Product(win, name="nordic_1400x1600")     # promote — snapshots
    occ    = NORDIC.occurrence(name="g0")              # identity only
    leaf.anchor(occ, along_center=..., up=...)         # placement unchanged

Why promotion beats declaration: authoring stays bottom-up (how the corpus
already works), it needs ONE construct instead of a parallel
``WindowType``/``MemberType``/``DoorType`` hierarchy, and the catalog geometry
is authored with the whole language rather than through a restricted second
surface.

**The hard rule — a Product takes no variables (§V.6).** Every variable is
consumed by the function that builds the source element, *before* promotion.
``.occurrence()`` accepts IDENTITY only. Two windows differing by 200 mm ARE
two Products: two cut-list lines, two order codes, two prices. You do not
order "a window, parameterised". The rule is load-bearing rather than tidy —
a silently ignored override makes the geometry wrong, and a silently
un-shared occurrence makes the type relation a lie, and both are silent.

The same answer falls out of geometry: ``.anchor()`` never mirrors, it rotates
(``det = +1`` on every host), so a left-hand and a right-hand door
cannot be one Product placed two ways either.
"""

from __future__ import annotations

import copy
import itertools
import json
from typing import Any, List, Optional, Tuple

from lite_step.models.elements import BimElement, _LEAF_NAME_RE
from lite_step.models import taxonomy as tx

#: Fields a Product occurrence may legitimately differ from its snapshot in,
#: and the reason for each. Everything NOT listed here is geometry or spec,
#: and a difference in it makes ``IfcRelDefinesByType`` a lie.
#:
#: * ``name``/``id`` — identity. Every occurrence has its own by construction.
#: * ``placement`` — WHERE the unit stands. That is the occurrence's whole job.
#: * ``props`` — occurrence-level Psets. IFC separates type properties from
#:   occurrence properties deliberately, so differing here is legal.
_OCCURRENCE_MAY_DIFFER = frozenset({"name", "id", "placement", "props"})

#: Private lists that carry GEOMETRY and therefore belong in the signature: a
#: ``.difference()`` applied to one occurrence changes its shape and nothing
#: else records that. ``_openings`` is deliberately absent — an opening is
#: anchored BY the host, so it is placement, not catalog shape.
_OCCURRENCE_GEOMETRY_OPERANDS = ("_cuts", "_adds", "_intersects", "_voids")


def catalog_signature(elem) -> Tuple:
    """A structural fingerprint of an element's SHAPE, ignoring identity.

    Recursive over ``_elements`` and over the boolean operands, so a change
    anywhere in the subtree registers. Compared, never displayed — the
    divergence message names the element, not this.

    Deliberately taken at the DSL level rather than on emitted geometry.
    Emitted bodies legitimately differ between occurrences of one Product: an
    anchored child is baked into world-axis-aligned coordinates, so the same
    window on a +Y wall and a +X wall are genuinely two bodies (which is why
    ``product_types`` groups occurrences into VARIANTS). Fingerprinting the
    emitted shape would therefore refuse the rotated facade, which is legal;
    fingerprinting the authored DSL separates "placed differently" from
    "authored differently", and only the second is the lie.
    """
    declared = getattr(elem, "model_dump", None)
    if declared is None:                       # not a pydantic model
        return (type(elem).__name__, repr(elem))
    fields = elem.model_dump(exclude=_OCCURRENCE_MAY_DIFFER, mode="json")
    return (
        type(elem).__name__,
        json.dumps(fields, sort_keys=True, default=str),
        tuple(catalog_signature(c)
              for c in (getattr(elem, "_elements", None) or [])),
        tuple((slot, tuple(catalog_signature(o)
                           for o in (getattr(elem, slot, None) or [])))
              for slot in _OCCURRENCE_GEOMETRY_OPERANDS),
    )


def find_diverged_occurrences(project) -> List[str]:
    """Occurrences whose shape no longer matches the Product they claim.

    **Why this exists rather than a fourth guard.** ``Product.occurrence()``
    refuses variables and ``BimElement.__call__`` refuses derivation, so the
    two spellings that ANNOUNCE an override are already closed. Measured,
    three that do not announce it were still open — ``occ.model_copy(update=
    {"width": 900})``, plain attribute assignment ``occ.width = 900``, and
    mutating one of the occurrence's own children. Each produced an element
    still carrying ``_product_key`` while carrying different geometry, i.e.
    an ``IfcRelDefinesByType`` pointing at a type whose shape it does not
    have.

    Guarding spellings is a losing game — the set of ways to mutate a mutable
    object is open. Checking the INVARIANT is not: whatever route was taken,
    the occurrence either still matches its snapshot or it does not. Same
    reasoning as the renderability gate, which stopped asking which
    IFC entity might not draw and started asking whether anything failed to.

    An ERROR, like ``naming.find_shared_elements``, and returned as strings
    for the same reason: the caller composes them into the one report.

    Runs at VALIDATE time, which is before ``normalize_project_to_meters`` —
    so the occurrence and the snapshot are both in authoring millimetres and
    directly comparable. Moving this after normalize would compare metres
    against millimetres and fire on everything.
    """
    out: List[str] = []
    seen: set = set()

    def visit(elem):
        if id(elem) in seen:
            return
        seen.add(id(elem))
        product = getattr(elem, "_product", None)
        if product is not None and not product.matches(elem):
            label = (getattr(elem, "_canonical_name", None)
                     or getattr(elem, "name", None) or type(elem).__name__)
            out.append(
                f"{type(elem).__name__} '{label}' is an occurrence of Product "
                f"{product.name!r} but its geometry no longer matches the "
                f"promoted item. An occurrence carries IDENTITY and PLACEMENT "
                f"only — changing its shape (by assignment, model_copy(update=), "
                f"or editing one of its children) leaves IfcRelDefinesByType "
                f"claiming a catalog item this unit is not. Two units of "
                f"different shape are TWO Products: build the second with the "
                f"function that built the first and promote it under its own "
                f"name=."
            )
        for child in (getattr(elem, "_elements", None) or []):
            visit(child)

    for site in getattr(project, "sites", None) or []:
        visit(site)
    for storey in getattr(project, "storeys", None) or []:
        for elem in storey.elements or []:
            visit(elem)
    return out


#: Promotion serial. A plain monotonic int, so it survives ``deepcopy``
#: verbatim (the ``_miter_joints`` precedent) — which is what makes "are these
#: two occurrences of the SAME promotion?" answerable after ``normalize``,
#: where object identity no longer is. Value equality on the snapshot cannot
#: answer it either: two elements built by the same function differ by their
#: auto-generated ``id``.
_PROMOTION_SERIAL = itertools.count(1)


class Product:
    """A promoted catalog item. See the module docstring for the grammar.

    ``Product(source, name=)`` snapshots ``source``; the snapshot is what every
    occurrence copies, so mutating ``source`` afterwards never retroactively
    changes placed occurrences (§V.5 rule 2).
    """

    __slots__ = ("_name", "_snapshot", "_ifc_class", "_type_entity", "_serial",
                 "_snapshot_signature")

    def __init__(self, source: Any = None, *, name: str = None, **forbidden: Any):
        if forbidden:
            raise ValueError(
                f"Product() got {', '.join(repr(k) for k in sorted(forbidden))} "
                f"— a Product takes NO variables. It is finished physical "
                f"dimensions: every parameter is consumed by the function that "
                f"builds the source element, before promotion. Pass "
                f"{', '.join(repr(k) for k in sorted(forbidden))} to that "
                f"function and promote the result."
            )
        if source is None:
            raise ValueError(
                "Product(source, name=...) needs the built element to promote "
                "— a Product is not declared abstractly, it is a snapshot of a "
                "concrete element authored bottom-up.")
        if not isinstance(source, BimElement):
            raise TypeError(
                f"Product(source=) takes a built element, got "
                f"{type(source).__name__}. Author the catalog item with the "
                f"normal DSL, then promote it.")

        self._name = _validate_catalog_key(name)
        self._serial = next(_PROMOTION_SERIAL)

        # Rule 2 (§V.6 / V.2.3) — the source must be a PHYSICAL element. A bare
        # Box is GEOMETRIC: it has no Ifc*Type counterpart, so there is nothing
        # for IfcRelDefinesByType to point at. Wrap it in the element it IS
        # (Wall/Beam/Element(ifc_class=...)) and promote that.
        if not tx.is_physical(source):
            raise ValueError(
                f"Product({type(source).__name__}) — only a PHYSICAL element "
                f"can be promoted. {type(source).__name__} is geometry; the "
                f"IFC schema types PRODUCTS (IfcWallType, IfcWindowType, ...) "
                f"and has no type counterpart for a raw solid. Wrap the "
                f"geometry in the element it is — Wall/Column/Beam/Slab/Roof/"
                f"Window/Door, or Element(ifc_class=...) — and promote that."
            )

        # Rule 1 (§V.5) — the source must be UNPLACED. A catalog object has no
        # canonical path; that is exactly what distinguishes it from a tree
        # member. Promoting a placed element would either consume a live
        # element (it vanishes from the model) or strand a duplicate where the
        # source sat.
        _refuse_if_placed(source, self._name)

        # Rule 3 (§V.5) — SNAPSHOT. Holding a live reference is the obvious
        # implementation and the wrong one: a later `win.material = ...` would
        # retroactively rewrite every occurrence already placed.
        # ``BimElement.__deepcopy__`` drops ``_parent`` for us.
        snapshot = copy.deepcopy(source)
        # A snapshot is a catalog item in its own right, never an occurrence of
        # whatever the source may have been an occurrence of.
        snapshot._product_key = None
        snapshot._product = None
        self._snapshot = snapshot
        # Taken ONCE, at promotion, from the snapshot rather than the source:
        # the source stays live and an author may keep editing it, which
        # rule 3 already decided must not reach placed occurrences.
        self._snapshot_signature = catalog_signature(snapshot)

        from lite_step.ifc.product_types import element_ifc_class, type_entity_for

        self._ifc_class = element_ifc_class(snapshot)
        # Refuse HERE, at the line that wrote Product(...), rather than
        # emitting an entity name IFC4X3_ADD2 does not define.
        self._type_entity = type_entity_for(self._ifc_class)

    # -- identity -----------------------------------------------------------

    @property
    def name(self) -> str:
        """The catalog KEY — not a canonical path (§V.5 rule 3).

        It names an item you could order, so it is stable across models and
        owes nothing to where any occurrence sits in the tree.
        """
        return self._name

    @property
    def snapshot(self) -> BimElement:
        """The promoted element as it was at promotion time.

        **Units trap.** ``normalize_project_to_meters`` walks the containment
        tree; a Product is not in it, so this snapshot stays in the units it
        was authored in (mm) while its occurrences are normalized to metres.

        Neither WS-B step reads this geometry. Step 1 reads the catalog key and
        the serial; step 2 builds its ``IfcRepresentationMap`` from a body an
        OCCURRENCE emitted — already normalized, already carved, already what
        the file would have contained anyway — so the trap is structurally
        absent rather than avoided by care, and the shared body is byte-for-byte
        a body the unshared file had. Anything that starts reading the snapshot
        for geometry must normalize it first, or every shared body comes out
        1000× too big; ``test_product_geometry_sharing.py::TestUnitsTrap``
        measures the emitted dimensions so that stays a statement about the
        file rather than about the code.
        """
        return self._snapshot

    @property
    def serial(self) -> int:
        """Which ``Product(...)`` call this is — see :data:`_PROMOTION_SERIAL`.

        Two promotions are two catalog items even when they were built from
        the same function, so two of them under one key is an authoring error
        the compiler can name.
        """
        return self._serial

    @property
    def ifc_class(self) -> str:
        return self._ifc_class

    @property
    def type_entity(self) -> str:
        """The ``Ifc<Class>Type`` entity this Product emits as."""
        return self._type_entity

    def __repr__(self) -> str:
        return (f"Product(name={self._name!r}, {self._ifc_class} -> "
                f"{self._type_entity})")

    # -- the occurrence verb ------------------------------------------------

    def matches(self, occurrence) -> bool:
        """True when ``occurrence`` still has the shape this Product promised.

        Compares :func:`catalog_signature`, so identity, placement and
        occurrence-level props are excluded and everything else — every field,
        every child, every boolean operand — must agree.
        """
        return catalog_signature(occurrence) == self._snapshot_signature

    def occurrence(self, *, name: str = None, **forbidden: Any) -> BimElement:
        """A placeable occurrence of this Product. IDENTITY ONLY.

        ``name`` is the DSL leaf the occurrence carries in the tree, and it is
        REQUIRED: the type relation is built by joining occurrence canonical
        names against the emitted products, so an anonymous occurrence would
        drop out of ``IfcRelDefinesByType`` and undercount the schedule.

        Placement is unchanged and belongs to the host —
        ``leaf.anchor(occ, along_center=..., up=...)``.
        """
        if forbidden:
            raise ValueError(
                f"{self._name}.occurrence() got "
                f"{', '.join(repr(k) for k in sorted(forbidden))} — a Product "
                f"takes NO variables (roadmap §V.6). An occurrence carries "
                f"IDENTITY only; it cannot restate geometry, because a Product "
                f"IS finished physical dimensions. Two units differing by any "
                f"of {', '.join(repr(k) for k in sorted(forbidden))} are TWO "
                f"Products — two cut-list lines, two order codes. Build the "
                f"second with the function that built this one and promote it "
                f"under its own name=."
            )
        if name is None:
            raise ValueError(
                f"{self._name}.occurrence(name=...) needs a name — the type "
                f"relation joins occurrences by their canonical name, so an "
                f"anonymous occurrence would be missing from "
                f"{self._type_entity}'s IfcRelDefinesByType and the count "
                f"would be wrong."
            )

        # Derivation-by-call gives a validated copy with a fresh id, deep-copied
        # private state and ``_parent=None`` — precisely an unattached new
        # element. Doing it by hand would be a second implementation of it.
        occ = self._snapshot(name=name)
        occ._product_key = self._name
        occ._product = self
        return occ

    def __call__(self, *args: Any, **kwargs: Any):
        """Refuse derivation-by-call on a Product (§V.5, "the trap").

        Part IV of the reference makes every element callable:
        ``instance(field=value)`` returns an independent COPY. If
        ``NORDIC(name="g0")`` also worked and meant *shared occurrence*, one
        spelling would mean two things dispatched on the non-local fact of
        whether the receiver happens to be promoted. Give the occurrence its
        own verb, and make the other spelling say so.
        """
        raise TypeError(
            f"Product {self._name!r} is not callable. "
            f"{self._name}(...) would read as derivation-by-call, which means "
            f"an independent COPY everywhere else in the DSL. An occurrence is "
            f"a different relation and has its own verb: "
            f"{self._name}.occurrence(name=...)."
        )


def _validate_catalog_key(name) -> str:
    if name is None:
        raise ValueError(
            "Product(source, name=...) needs a catalog key — the name is what "
            "a schedule counts by (\"9 x nordic_1400x1600\") and what the "
            "Ifc*Type entity carries.")
    if not isinstance(name, str) or not _LEAF_NAME_RE.match(name):
        raise ValueError(
            f"Product name={name!r} is not a valid catalog key — use a single "
            f"lowercase word matching [a-z0-9_]+ (no uppercase, ':', '.', or "
            f"whitespace). ':' in particular is the canonical-path separator, "
            f"and a catalog key is NOT a path: it names an item you can order, "
            f"independently of where any occurrence sits in the tree.")
    return name


def _refuse_if_placed(source: BimElement, key: str) -> None:
    """Rule 1 — refuse a source that is already in the containment tree."""
    parent = source.parent
    anchored = getattr(source, "_anchor_spec", None) is not None
    if parent is None and not anchored:
        return
    where = (f"contained in a {type(parent).__name__}" if parent is not None
             else "anchored into a host")
    raise ValueError(
        f"Product(..., name={key!r}): the source element is already placed "
        f"({where}) — only an UNPLACED element can be promoted. A catalog "
        f"object has no canonical path; that is what distinguishes it from a "
        f"tree member. Promoting a placed element would either consume the "
        f"element that is standing there or strand a duplicate beside it. "
        f"Build the catalog item on its own line, promote it, and place "
        f"{key}.occurrence(name=...) instead."
    )
