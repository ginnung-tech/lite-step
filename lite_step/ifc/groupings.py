"""Project-level GROUPINGS — aggregates and zones (WS-D + the zone half of WS-C).

The ONE place the emitter reads, mirroring
:mod:`lite_step.ifc.product_types`. Two implementations of one idea agreeing *usually* is what produces the bug, so every policy decision —
resolution, the class derivation, every refusal, the placement rule and the
missing-member message — is decided HERE; ``generator.py`` contributes only
the lines that mint an entity.

**One grammar, two kinds** (roadmap §V.7). An aggregate and a zone are the
same shape — a named, geometry-less grouping of tree members cited by
**canonical-name STRING**, never by object, and never ``.add()``ed::

    proj.aggregate("north_facade", members=["wall:north:storey:ground",
                                            "wall:north:storey:first",
                                            "wall:gable_end"])   # IfcRelAggregates
    proj.zone("thermal_north", members=["space:kitchen", "space:living"])  # IfcZone

They are built together on purpose: §V.7 established they are one idea, and
building them apart would produce two spellings of it.

Why strings, and why the DSL tree never moves (roadmap §0 + §3):

* **Identity flows only down the DSL containment tree.** ``.add()``-ing the
  members under the parent would rewrite their canonical names
  (``wall:north:storey:ground`` becomes
  ``wall:north:storey:ground:wall:north_facade``) — the identity churn the
  facade inversion exists to avoid, and it would churn the manifest key and
  the patch atom with it. A grouping takes STRINGS, so the author still
  writes ``storey.add(wall)`` and the tree — and therefore every canonical
  name — is untouched. That is what makes the IFC emission below a pure
  emission concern with **zero identity churn**; ``test_groupings.py::
  TestCanonicalNamesAreUntouchedByEmission`` measures it byte-for-byte.
* A string in a keyword is therefore a **reference into the tree**, resolved
  late by the same segment-aligned suffix match ``Anchor(host=)`` and
  ``assert_carved`` use. Miss and ambiguity are compile ERRORS there, and
  they are compile errors here, for the same reason: a grouping that
  silently matched nothing is worse than no grouping.

Two rules that are structural rather than stylistic:

* **An aggregate's IFC class is DERIVED from its members; mixed classes are a
  compile error.** §3 requires same-class (``IfcWall = IfcWall + IfcWall``).
  Deriving it means an author cannot state a class that contradicts the
  members, and the failure is loud instead of a silently mis-typed parent.
* **A zone groups Spaces only.** ``IfcZone`` is an ``IfcGroup``; the schema's
  own ``IfcRelAssignsToGroup`` for a zone relates spatial elements. A
  non-Space member is a loud refusal, not a coerced one.

**The containment rule (IFC 4.3 *Spatial Containment*).** The
schema states it verbatim, and its scope is *"Any subtype of ``IfcElement``
can be an element assembly"* — so it governs our ``IfcWall`` facade:

    "In this case it should not be additionally contained in the project
    spatial hierarchy, i.e. ``SELF\\IfcElement.ContainedInStructure`` should
    be *NIL*."

**The WHOLE carries spatial containment; the PARTS have
``ContainedInStructure`` NIL.** Roadmap §3 specified the inverse — members
contained, parent in no relation at all — which made the parent a ROOT
ORPHAN, unreachable from any spatial walk including our own SPA inspector
tree. The call was made: conform. §3 is overruled on this point, and only on this
point: it inverted the pattern because it believed containing the members
under the parent would rewrite their canonical names, and that fear simply
does not apply, because a grouping cites STRINGS and moves nothing in the
DSL tree.

**Which container the parent goes in** is DERIVED, never a parameter — see
:func:`parent_container`.

**The placement rule (constraint 1, roadmap §3).**
``generator.py::_create_opening_profile_path`` records a measured failure for
a world-coordinate child with an UNCHAINED ``ObjectPlacement`` inside an
``IfcRelAggregates`` whose ``RelatingObject`` has no ``Representation``: the
mitered frame tessellated at ~1/87 of its true volume from a bit-identical
entity subgraph that was exact outside the aggregation. Nothing
raised. A geometry-less aggregate parent is exactly that configuration, so
members are re-chained: the parent carries an IDENTITY placement rooted at
**its own container** (the storey it is now contained in — or the site, in a
building-less model), and each member's ``ObjectPlacement`` chains to it,
its ``RelativePlacement`` rewritten from its own storey's frame into the
parent's so the resolved world position is unchanged to the bit. That
equality is the guard —
``test_groupings.py::TestMembersKeepTheirWorldPosition`` measures it through
``ifcopenshell.geom``, because a placement rewrite that moved a wall would
otherwise be as silent as the bug it prevents.

Re-chaining is also what IFC asks for: a decomposed element's placement is
relative to the whole, not to the container — and rooting the parent at the
structure that contains it is the ``IfcLocalPlacement`` convention
("PlacementRelTo should point to the object placement of the spatial
structure element that contains this element"), so the whole chain
storey → parent → member is conventional at every link.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: The two kinds. One grammar, two emissions.
AGGREGATE = "aggregate"
ZONE = "zone"

#: The only IFC class a zone may group. ``IfcZone`` is defined over spaces;
#: grouping a wall in one is not a lenient reading of the schema, it is a
#: different relation.
ZONE_MEMBER_IFC_CLASS = "IfcSpace"

#: Classes that ARE the spatial structure. The aggregate parent is now
#: CONTAINED in a storey, and ``IfcRelContainedInSpatialStructure``
#: "shall not be used to include other spatial structure elements into a
#: spatial structure element" — so a spatial-structure parent would be an
#: illegal relation rather than merely an odd one. Refused with a pointer at
#: the verb that does mean this.
SPATIAL_STRUCTURE_IFC_CLASSES = frozenset({
    "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace",
    "IfcSpatialZone", "IfcExternalSpatialElement",
})

#: Containment + reference attributes the member index descends. Mirrors
#: ``compiler.naming._WALK_ATTRS`` — a member is cited by its canonical name,
#: so it is reached exactly the way that canonical name is derived. A second,
#: subtly different walk is how two spellings of one index start to disagree.
_WALK_ATTRS = ("_elements", "_openings", "_cuts", "_adds", "_intersects", "_fills")


class GroupingError(ValueError):
    """A ``proj.aggregate()`` / ``proj.zone()`` statement cannot be realised."""


@dataclass(frozen=True)
class ResolvedGrouping:
    """One grouping, with every member resolved to a full canonical name."""

    kind: str                      #: :data:`AGGREGATE` or :data:`ZONE`
    name: str                      #: the catalog-style key the author wrote
    ifc_class: str                 #: aggregate: the derived member class. zone: ``IfcZone``
    members: Tuple[str, ...]       #: resolved canonical names, authoring order


# ---------------------------------------------------------------------------
# Author-time validation — loud at the line that wrote the statement
# ---------------------------------------------------------------------------


def validate_grouping_name(name: Any, kind: str) -> str:
    """A grouping name is a catalog-style KEY, not a canonical path.

    Same grammar as ``Product``'s ``name=`` and for the same reason: it names
    a thing you could put on a schedule, independently of where any member
    sits in the tree. ``':'`` in particular is the canonical-path separator.
    """
    from lite_step.models.elements import _LEAF_NAME_RE

    if name is None:
        raise GroupingError(
            f"proj.{kind}(name, members=[...]) needs a name — it is what the "
            f"emitted {'IfcRelAggregates parent' if kind == AGGREGATE else 'IfcZone'} "
            f"carries as its Name, and what a schedule groups by.")
    if not isinstance(name, str) or not _LEAF_NAME_RE.match(name):
        raise GroupingError(
            f"proj.{kind}({name!r}, ...) is not a valid name — use a single "
            f"lowercase word matching [a-z0-9_]+ (no uppercase, ':', '.', or "
            f"whitespace). ':' is the canonical-path separator, and a "
            f"{kind} name is NOT a path: the {kind} is not in the containment "
            f"tree and has no canonical name of its own.")
    return name


def validate_members(members: Any, kind: str, name: str) -> Tuple[str, ...]:
    """Members are canonical-name STRINGS, keyword-bound, never objects (§V.7).

    The string/object split is not stylistic — it IS the presence or absence
    of a canonical path, which is §0's identity rule made syntactic. An object
    passed here would read as containment, and containment would rewrite the
    member's canonical name.
    """
    if members is None:
        raise GroupingError(
            f"proj.{kind}({name!r}, members=[...]) needs members — an empty "
            f"{kind} names nothing and would emit a parent no one can reach.")
    if isinstance(members, str):
        raise GroupingError(
            f"proj.{kind}({name!r}, members={members!r}) got a single string. "
            f"members= is a LIST of canonical-name fragments, even for one "
            f"member: members=[{members!r}].")
    if not isinstance(members, (list, tuple)):
        raise GroupingError(
            f"proj.{kind}({name!r}, members=...) takes a list of "
            f"canonical-name strings, got {type(members).__name__}.")
    if not members:
        raise GroupingError(
            f"proj.{kind}({name!r}, members=[]) is empty — an empty {kind} "
            f"names nothing and would emit a parent no one can reach.")
    out: List[str] = []
    for member in members:
        if not isinstance(member, str):
            raise GroupingError(
                f"proj.{kind}({name!r}, members=...) got a "
                f"{type(member).__name__} object. Members are cited by "
                f"canonical-name STRING, never by object (roadmap §V.7): an "
                f"object passed positionally is CONTAINMENT, and containing a "
                f"member would rewrite its canonical name — the identity churn "
                f"the {kind} exists to avoid. Write "
                f"members=[\"{_suggest_fragment(member)}\"].")
        if not member:
            raise GroupingError(
                f"proj.{kind}({name!r}, members=...) got an empty string.")
        out.append(member)
    return tuple(out)


def _suggest_fragment(obj: Any) -> str:
    """A plausible canonical fragment for an object handed in by mistake."""
    from lite_step.compiler.naming import type_segment

    leaf = getattr(obj, "name", None)
    try:
        segment = type_segment(obj)
    except Exception:                                   # pragma: no cover
        segment = type(obj).__name__.lower()
    return f"{segment}:{leaf}" if leaf else f"{segment}:<name it first>"


# ---------------------------------------------------------------------------
# Compile-time resolution
# ---------------------------------------------------------------------------


def _index_members(proj) -> Dict[str, Tuple[Any, bool]]:
    """``{canonical_name: (element, is_spatially_contained)}``.

    ``is_spatially_contained`` marks a TOP-LEVEL element — a direct child of a
    storey or of a ``Site`` container, i.e. one the generator puts in an
    ``IfcRelContainedInSpatialStructure``. Nested elements are excluded from
    grouping because they are already a decomposition part of their host, and
    IFC gives a product at most one ``Decomposes`` relation.

    On-path re-entry (a containment cycle) is PRUNED — a termination guard.
    ``validate_project_report`` refuses a cyclic model before emission, so
    this index is never asked to describe one; it only has to survive being
    asked.
    """
    from lite_step.compiler.walkguard import WalkGuard

    index: Dict[str, Tuple[Any, bool]] = {}
    guard = WalkGuard()

    def walk(elem, top: bool) -> None:
        if not guard.enter(elem):
            return
        try:
            canonical = getattr(elem, "_canonical_name", None)
            if canonical and canonical not in index:
                index[canonical] = (elem, top)
            for attr in _WALK_ATTRS:
                for child in getattr(elem, attr, None) or []:
                    walk(child, False)
        finally:
            guard.leave(elem)

    for storey in getattr(proj, "storeys", None) or []:
        for elem in storey.elements:
            walk(elem, True)
    for site in getattr(proj, "sites", None) or []:
        canonical = getattr(site, "_canonical_name", None)
        if canonical and canonical not in index:
            index[canonical] = (site, False)      # IfcSite IS the structure
        for child in getattr(site, "_elements", None) or []:
            walk(child, True)
    return index


def _resolve_member(fragment: str, kind: str, name: str,
                    index: Dict[str, Tuple[Any, bool]]) -> str:
    """One member fragment -> its full canonical name, or raise.

    Resolution is ``naming.resolve_host`` — the reference grammar
    ``Anchor(host=)`` and ``assert_carved`` already use — so an author writes
    the same fragment everywhere.
    """
    from lite_step.compiler.naming import resolve_host

    status, matched, candidates = resolve_host(fragment, index.keys())
    if status == "ambiguous":
        raise GroupingError(
            f"proj.{kind}({name!r}, ...): member {fragment!r} matches several "
            f"elements ({', '.join(candidates)}) — write more leaf-first pairs "
            f"to say which one. A {kind} that grouped the wrong wall would "
            f"look exactly like one that grouped the right one.")
    if status == "miss":
        raise GroupingError(
            f"proj.{kind}({name!r}, ...): member {fragment!r} names no element "
            f"in this model. Members are canonical-name fragments matched on "
            f"whole ``type:leaf`` pairs from the tail, exactly as "
            f"Anchor(host=) matches — so \"wall:north\" matches both "
            f"wall:north and box:body:wall:north. A silently empty {kind} is "
            f"worse than no {kind}.")
    return matched


def _member_ifc_class(elem) -> str:
    from lite_step.ifc.product_types import element_ifc_class

    return element_ifc_class(elem)


def collect_groupings(proj) -> List[ResolvedGrouping]:
    """Every ``proj.aggregate()`` / ``proj.zone()`` statement, resolved.

    Raises :class:`GroupingError` — never warns — on every way a grouping can
    be wrong, because each of them is silent otherwise:

    * two groupings under one name (one entity would stand for two);
    * a member fragment that misses or is ambiguous;
    * a member cited twice in one grouping;
    * a member that is not a top-level element (it is already a decomposition
      part of its host, and IFC gives a product one ``Decomposes`` at most);
    * an aggregate whose members are not all one IFC class (§3);
    * an aggregate over spatial-structure classes (an orphan storey/space);
    * a member in two aggregates (again, one ``Decomposes``);
    * a zone member that is not an ``IfcSpace``.
    """
    statements = list(getattr(proj, "_groupings", None) or [])
    if not statements:
        return []

    index = _index_members(proj)
    seen_names: Dict[str, str] = {}
    aggregate_owner: Dict[str, str] = {}
    out: List[ResolvedGrouping] = []

    for kind, name, fragments in statements:
        if name in seen_names:
            raise GroupingError(
                f"{name!r} is declared twice — once as a {seen_names[name]}, "
                f"once as a {kind}. The name is the emitted entity's Name and "
                f"the key anything downstream groups by, so two declarations "
                f"under it would be read as one thing.")
        seen_names[name] = kind

        resolved: List[str] = []
        for fragment in fragments:
            canonical = _resolve_member(fragment, kind, name, index)
            if canonical in resolved:
                raise GroupingError(
                    f"proj.{kind}({name!r}, ...): {canonical} is cited twice "
                    f"(as {fragment!r}). A member listed twice would appear "
                    f"twice in the emitted relation and double-count in any "
                    f"schedule read off it.")
            resolved.append(canonical)

        elements = [index[c] for c in resolved]
        for canonical, (elem, top) in zip(resolved, elements):
            if not top:
                raise GroupingError(
                    f"proj.{kind}({name!r}, ...): {canonical} is not a "
                    f"top-level element — it is contained in "
                    f"{type(getattr(elem, 'parent', None)).__name__}, which "
                    f"already decomposes it. IFC gives a product ONE "
                    f"Decomposes relation, so grouping it here would give it "
                    f"two parents and whichever the viewer honours is "
                    f"undefined. Group the element that stands in the storey.")

        classes = [_member_ifc_class(elem) for elem, _ in elements]

        if kind == ZONE:
            wrong = [(c, k) for c, k in zip(resolved, classes)
                     if k != ZONE_MEMBER_IFC_CLASS]
            if wrong:
                raise GroupingError(
                    f"proj.zone({name!r}, ...) groups "
                    + ", ".join(f"{c} ({k})" for c, k in wrong)
                    + f" — a zone groups {ZONE_MEMBER_IFC_CLASS}s only. "
                    f"IfcZone is defined over spaces; a zone of walls is a "
                    f"different relation, not a lenient reading of this one. "
                    f"Use proj.aggregate({name!r}, ...) for same-class "
                    f"element grouping.")
            out.append(ResolvedGrouping(kind=ZONE, name=name,
                                        ifc_class="IfcZone",
                                        members=tuple(resolved)))
            continue

        distinct = sorted(set(classes))
        if len(distinct) > 1:
            detail = ", ".join(f"{c} ({k})" for c, k in zip(resolved, classes))
            raise GroupingError(
                f"proj.aggregate({name!r}, ...) mixes IFC classes "
                f"({' + '.join(distinct)}): {detail}. An aggregate's class is "
                f"DERIVED from its members and §3 requires them to agree — "
                f"IfcWall = IfcWall + IfcWall. There is no class the parent "
                f"could carry that does not contradict half its members.")
        ifc_class = distinct[0]
        if ifc_class in SPATIAL_STRUCTURE_IFC_CLASSES:
            raise GroupingError(
                f"proj.aggregate({name!r}, ...) aggregates {ifc_class}, which "
                f"IS the spatial structure. An aggregate parent is never "
                f"contained (§3), so this would emit an orphan {ifc_class} "
                f"outside the storey/site hierarchy. "
                + (f"For spaces, proj.zone({name!r}, ...) is the verb — it "
                   f"emits IfcZone + IfcRelAssignsToGroup and leaves the "
                   f"spatial tree alone."
                   if ifc_class == ZONE_MEMBER_IFC_CLASS else
                   "Spatial containers are not grouped this way."))

        for canonical in resolved:
            owner = aggregate_owner.get(canonical)
            if owner is not None:
                raise GroupingError(
                    f"proj.aggregate({name!r}, ...): {canonical} is already a "
                    f"member of aggregate {owner!r}. IFC gives a product ONE "
                    f"Decomposes relation, so the second aggregation is "
                    f"either dropped or overwrites the first depending on the "
                    f"reader. Zones may overlap; aggregates may not.")
            aggregate_owner[canonical] = name

        out.append(ResolvedGrouping(kind=AGGREGATE, name=name,
                                    ifc_class=ifc_class,
                                    members=tuple(resolved)))
    return out


# ---------------------------------------------------------------------------
# Shared failure — the emitter raises the identical message
# ---------------------------------------------------------------------------


def missing_member_error(group: ResolvedGrouping,
                         missing: Sequence[str]) -> GroupingError:
    """Raised when a resolved member emitted no IFC product.

    Loud rather than a short relation: an ``IfcRelAggregates`` listing 2 of 3
    walls is a facade with a wall missing from every schedule and every
    selection read off the file, and it is exactly the kind of wrong that
    reads as success in every log.
    """
    return GroupingError(
        f"{group.kind} {group.name!r}: no IFC product was emitted for "
        f"{len(missing)} of {len(group.members)} member(s) "
        f"({', '.join(missing)}). The emitted relation would list fewer "
        f"members than were authored, so anything reading this file would see "
        f"an incomplete {group.kind}. A member must be an element that emits "
        f"its own product.")


@dataclass(frozen=True)
class Container:
    """The spatial structure element a top-level DSL element stands in."""

    kind: str            #: ``"storey"`` or ``"site"``
    index: int           #: index into ``proj.storeys`` / ``proj.sites``
    emitted_index: int   #: index among the containers that actually EMIT
    name: Optional[str]  #: the container's canonical name — its emitted Name
    elevation: float     #: storey elevation; 0.0 for a site


def member_containers(proj) -> Dict[str, Container]:
    """``{canonical_name: Container}`` for every top-level element.

    One walk, so the elevation a backend folds into a re-chained placement and
    the container a grouping parent is put in can never come from two
    subtly different traversals.

    ``emitted_index`` exists because a storey with no elements emits no
    ``IfcBuildingStorey`` on either backend, so the DSL index and the index of
    the entity in the file diverge the moment a model carries an empty storey.
    A backend that looked up the wrong storey would contain the facade one
    floor off — geometry untouched, so nothing would raise.
    """
    out: Dict[str, Container] = {}
    emitted = 0
    for index, storey in enumerate(getattr(proj, "storeys", None) or []):
        if not storey.elements:
            continue                     # emits no IfcBuildingStorey
        container = Container(
            kind="storey", index=index, emitted_index=emitted,
            name=getattr(storey, "_canonical_name", None),
            elevation=float(getattr(storey, "elevation", 0.0) or 0.0),
        )
        emitted += 1
        for elem in storey.elements:
            canonical = getattr(elem, "_canonical_name", None)
            if canonical:
                out[canonical] = container
    for index, site in enumerate(getattr(proj, "sites", None) or []):
        container = Container(
            kind="site", index=index, emitted_index=index,
            name=getattr(site, "_canonical_name", None),
            elevation=0.0,               # site children are site-relative
        )
        for child in getattr(site, "_elements", None) or []:
            canonical = getattr(child, "_canonical_name", None)
            if canonical:
                out[canonical] = container
    return out


def member_elevations(proj) -> Dict[str, float]:
    """``{canonical_name: storey elevation}`` for every top-level element.

    The number a backend folds into a member's ``RelativePlacement`` when it
    re-chains the member to a grouping parent. Decided here so no caller
    re-derives it: the elevation is read from the model, and the emitter
    asserts against the absolute placement it computes from the file.
    """
    return {canonical: container.elevation
            for canonical, container in member_containers(proj).items()}


def parent_container(proj, group: ResolvedGrouping) -> Container:
    """The spatial structure element the aggregate PARENT is contained in.

    DERIVED, never a parameter. An author who had to name the
    storey could name one no member stands in, and the file would still
    validate — the wrong answer, silently.

    **The rule: the container of the LOWEST member.** IFC 4.3 states it for
    the multi-level case it does discuss — *"A multi-storey space is contained
    (or belongs to) the building storey at which its ground level is, but it
    is referenced by all the other building storeys, in which it spans"*, and
    *"A lift shaft might be contained by the basement, but referenced by all
    storeys, through which it spans."* An element spanning storeys belongs to
    the one its base is in. A facade of stacked walls is that case exactly, so
    the parent goes in the container of its lowest member. Ties (two storeys
    at one elevation) break on DSL order, so the answer is deterministic
    rather than dict-ordered.

    ``IfcRelReferencedInSpatialStructure`` — the "referenced by all the other
    storeys" half of that quote — is deliberately NOT emitted: it is a second
    relation carrying no information this file does not already hold in
    ``IfcRelAggregates``, and an extra relation is an extra thing to keep in
    step. Add it when a consumer is measured to need it.

    Refusals are loud, because each of these produces a plausible file:

    * a member with no container at all (not a top-level element);
    * members split across a site and a building — there is no single
      structure that contains both, and picking either buries the facade
      under a hierarchy half its members are not in;
    * members in two different sites, for the same reason.

    Multiple ``IfcBuilding``s are not expressible in this DSL (``proj.storeys``
    is one flat list under one building), so the cross-building case cannot be
    constructed here; the site/building split is its representative, and it is
    refused.
    """
    containers = member_containers(proj)
    missing = [m for m in group.members if m not in containers]
    if missing:
        raise GroupingError(
            f"aggregate {group.name!r}: no spatial container is known for "
            f"{', '.join(missing)}. The aggregate parent is contained in the "
            f"container of its lowest member (IFC 4.3 Spatial Containment), "
            f"so a member standing in no storey and no site leaves the parent "
            f"with nowhere to go — and a parent in no containment is a root "
            f"orphan, invisible to every spatial walk.")

    chosen = [containers[m] for m in group.members]
    kinds = sorted({c.kind for c in chosen})
    if len(kinds) > 1:
        detail = ", ".join(f"{m} ({containers[m].kind})" for m in group.members)
        raise GroupingError(
            f"proj.aggregate({group.name!r}, ...) spans a {' and a '.join(kinds)}: "
            f"{detail}. The parent is contained in ONE spatial structure "
            f"element, and there is no storey that contains a site child nor "
            f"a site that contains a storey element — either choice would file "
            f"the aggregate under a hierarchy half its members are not in. "
            f"Group site elements and building elements separately.")
    if kinds == ["site"]:
        sites = sorted({c.index for c in chosen})
        if len(sites) > 1:
            raise GroupingError(
                f"proj.aggregate({group.name!r}, ...) spans {len(sites)} "
                f"different sites. The parent is contained in ONE spatial "
                f"structure element; there is no container above two sites "
                f"other than the project, and IfcProject is not a spatial "
                f"structure element. Group per site.")
        return chosen[0]
    return min(chosen, key=lambda c: (c.elevation, c.index))
