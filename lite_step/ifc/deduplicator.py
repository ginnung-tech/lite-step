"""
IFC STEP Post-Processing Deduplicator.

Eliminates duplicate entities in IFC STEP files by finding entities with
identical content and merging references to the canonical (first) occurrence.

Handles duplicates created by ifcopenshell's internal API calls that
bypass the EntityCache (e.g., directions, points, placements created by
run("geometry.add_profile_representation", ...)).

Floating-point values are rounded to 0.1 mm precision (4 decimal places in
meters) before comparison — for LENGTH-valued entity types only. Dimensionless
ones are exempt; see ``DEDUP_NO_FLOAT_ROUND_TYPES``.
"""

import re
from typing import Dict, List, Optional, Tuple, Set


# Entity types safe to deduplicate (geometric/style primitives only).
# Entities with GUIDs or semantic identity are never touched.
DEDUP_SAFE_TYPES = frozenset({
    'IFCDIRECTION',
    'IFCCARTESIANPOINT',
    'IFCAXIS2PLACEMENT3D',
    'IFCAXIS2PLACEMENT2D',
    'IFCRECTANGLEPROFILEDEF',
    'IFCARBITRARYCLOSEDPROFILEDEF',
    'IFCCOLOURRGB',
    'IFCSURFACESTYLESHADING',
    'IFCSURFACESTYLERENDERING',
    'IFCSURFACESTYLE',
    'IFCPOLYLINE',
    # NOT IfcExtrudedAreaSolid — each solid is styled individually via IfcStyledItem
    # and StyledByItem has SET [0:1] constraint. Sharing solids breaks this.
    'IFCLOCALPLACEMENT',
})

# Types that stay dedup-eligible but are EXEMPT from the 0.1 mm float
# rounding below. Rect profiles carry the unique_dims geometry-signature
# noise (EntityCache._UNIQUE_DIMS_STEP, 0.02 mm): mirror facade leaves get
# deliberately sub-tolerance-distinct dims so the SPA's fragments render
# worker never groups them into one instanced mesh (whose multi-threaded
# instance registration sometimes drops one — the missing-cladding bug).
# Rounding to 0.1 mm here would erase that noise and re-merge the dims.
# Genuinely identical profiles still dedup: their full-precision floats
# serialize byte-identically, so the merge step still sees them as equal.
#
# IfcDirection carries DIMENSIONLESS ratios, and that is the whole reason it
# belongs here. The rounding below was tuned as a "0.1 mm"
# length tolerance — a unit that a direction ratio does not have. Applied to a
# unit vector, 4 decimals is an absolute 1e-4 error per component: measured on
# this corpus, 105 of 178 emitted directions were non-unit, worst |v|-1 =
# 6.05e-5, i.e. 600x outside the 1e-7 normalisation tolerance geometry kernels
# check against. Rounding at 6 dp (matching DirectionKey's cache precision) is
# NOT enough either — it still leaves 3.4e-7 — and any higher figure is a magic
# number, so the exemption is wholesale.
#
# Nothing is lost: directions are built by explicit normalisation, so genuinely
# identical ones still serialise byte-identically and still merge (the same
# property the rect-profile exemption relies on), and EntityCache
# .get_or_create_direction already merges within 1e-6 upstream. Two directions
# differing by more than 1e-4 are genuinely different directions — 0.006 deg
# over a 10 m roof plane is 1 mm of rise, not noise.
DEDUP_NO_FLOAT_ROUND_TYPES = frozenset({
    'IFCRECTANGLEPROFILEDEF',
    'IFCDIRECTION',
})

#: Decimals a LENGTH is snapped to before dedup comparison — 4 in metres,
#: i.e. 0.1 mm. Named rather than inline because it is a PIPELINE FACT that
#: consumers have to reason about: a shipped vertex can sit up to half a step
#: (0.05 mm) from the coordinate the compiler computed, so any bound asserted
#: against emitted geometry must allow for it. See
#: ``test_authored_aabb._salt_tolerance_mm``, which derives from this.
DEDUP_LENGTH_DECIMALS = 4

# Regex to parse STEP entity lines: #ID=TYPE(args);
_ENTITY_RE = re.compile(r'^#(\d+)\s*=\s*(\w+)\s*\((.*)\)\s*;\s*$')
# Regex to find entity references (#N) in args
_REF_RE = re.compile(r'#(\d+)')
# Regex to find STEP real numbers (must have a dot, e.g. 3.5, 0., -1.2).
# Plain integers are NOT matched — they may be entity reference IDs inside #N.
_FLOAT_RE = re.compile(r'(-?\d+\.\d*)')


def deduplicate_ifc_step(
    ifc_content: str,
    renumber: bool = False,
    drop_ids: Optional[Set[int]] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Deduplicate entities in an IFC STEP string.

    Args:
        ifc_content: The complete IFC STEP file content
        renumber: If True, renumber entities sequentially (slower but tidier).
                  Default False — STEP P21 allows non-sequential IDs.
        drop_ids: Entity ids to delete outright, on top of the deduplication.
                  This is the Product geometry purge — see
                  :func:`_drop_purged` for why it happens HERE rather than
                  through ``ifcopenshell.file.remove``.

    Returns:
        Tuple of (optimized_content, stats_dict)
    """
    # Split into sections
    header, data_lines, footer = _split_step_sections(ifc_content)

    # Parse entities
    entities = _parse_entities(data_lines)
    if not entities:
        return ifc_content, {"entities_before": 0, "entities_after": 0,
                             "entities_removed": 0, "reduction_percent": 0.0}

    entities_before = len(entities)
    purged = _drop_purged(entities, drop_ids)

    # Normalize floats in safe entity types (mm precision rounding)
    _normalize_floats(entities)

    # Build dependency graph
    deps = _build_dependencies(entities)

    # Topological sort (leaves first)
    sorted_ids = _topological_sort(entities, deps)

    # Find duplicates
    canonical_map = _find_duplicates(entities, sorted_ids)

    if not canonical_map and not purged:
        return ifc_content, {"entities_before": entities_before,
                             "entities_after": entities_before,
                             "entities_removed": 0, "reduction_percent": 0.0}

    # Rewrite references and remove duplicates. NOTE the early return above is
    # guarded on `purged` too: returning the ORIGINAL text when nothing was
    # deduplicated would resurrect every entity the purge just dropped.
    surviving = _rewrite_and_remove(entities, canonical_map)

    if renumber:
        renumbered = _renumber(surviving)
        result = _reconstruct(header, renumbered, footer)
        entities_after = len(renumbered)
    else:
        # Fast path: emit surviving entities without renumbering
        result_lines = header[:]
        for eid in sorted(surviving.keys()):
            etype, args = surviving[eid]
            result_lines.append(f'#{eid}={etype}({args});')
        result_lines.extend(footer)
        result = '\n'.join(result_lines) + '\n'
        entities_after = len(surviving)

    stats = {
        "entities_before": entities_before,
        "entities_after": entities_after,
        "entities_removed": entities_before - entities_after,
        "reduction_percent": round((1 - entities_after / max(entities_before, 1)) * 100, 1)
    }

    return result, stats


def normalize_via_ifcopenshell(step_content: str) -> str:
    """Round-trip through IfcOpenShell for canonical entity ordering and validation.

    Used for 'download for Revit' quality output. Produces canonical STEP with
    proper inverse attributes and entity ordering. Spikes memory briefly but
    is bounded to a single post-processing pass.
    """
    try:
        import ifcopenshell
        from lite_step.ifc.step_writer import normalize_step_floats
        model = ifcopenshell.file.from_string(step_content)
        # IfcOpenShell's serializer emits raw full-precision doubles — re-encode
        # to shortest STEP reals (bit-exact;).
        return normalize_step_floats(model.to_string())
    except ImportError:
        return step_content  # ifcopenshell not available, return as-is


def _split_step_sections(content: str) -> Tuple[List[str], List[str], List[str]]:
    """Split IFC STEP into header, data, and footer sections."""
    lines = content.splitlines()
    header = []
    data = []
    footer = []

    section = 'header'
    for line in lines:
        stripped = line.strip()
        if stripped == 'DATA;':
            header.append(line)
            section = 'data'
            continue
        elif stripped == 'ENDSEC;' and section == 'data':
            footer.append(line)
            section = 'footer'
            continue

        if section == 'header':
            header.append(line)
        elif section == 'data':
            data.append(line)
        else:
            footer.append(line)

    return header, data, footer


def _parse_entities(data_lines: List[str]) -> Dict[int, Tuple[str, str]]:
    """Parse STEP entity lines into {id: (type, args_string)}."""
    entities = {}
    for line in data_lines:
        m = _ENTITY_RE.match(line.strip())
        if m:
            eid = int(m.group(1))
            etype = m.group(2)
            args = m.group(3)
            entities[eid] = (etype, args)
    return entities


def _normalize_floats(entities: Dict[int, Tuple[str, str]]):
    r"""Round floating-point values to 0.1 mm precision (4 decimals in meters)
    for safe entity types.

    Was mm (3 decimals) — too coarse for GEOMETRY-DEFINING floats: 2D profile
    coordinates (a 22.5 mm section half-width mangled to 22/23 mm = ~2% member
    volume error) and CSG half-space plane locations both live in
    DEDUP_SAFE_TYPES. 0.1 mm keeps the dedup benefit (merging genuinely
    identical primitives) without deforming geometry beyond any consumer's
    tolerance.

    **Precision is a property of the QUANTITY'S UNITS, not of the entity
    table.** The 3->4 decimal fix reasoned in "0.1 mm" throughout and swept
    IfcDirection along with it — but a direction ratio is dimensionless, so
    there 4 decimals is a bare 1e-4, not 0.1 mm of anything. That mismatch is; IfcDirection is now exempt. Weigh new entries the same
    way: ask what unit the numbers carry before assuming a length tolerance.

    TODO: ``_FLOAT_RE`` has no exponent branch, so it matches only the mantissa
    of a value like ``6.12E-17`` and leaves it unchanged (an earlier version of
    this docstring claimed such values collapse to 0 — they do not). Compare
    ``step_writer._STEP_TOKEN_RE``, which handles ``[eE][+-]?\d+`` and guards
    against matching inside identifiers. Widening this regex is NOT a drive-by:
    rounding the mantissa of a large-exponent real would be a far worse bug
    than the one it fixes.
    """
    for eid in list(entities.keys()):
        etype, args = entities[eid]
        if etype not in DEDUP_SAFE_TYPES:
            continue
        if etype in DEDUP_NO_FLOAT_ROUND_TYPES:
            # Carries deliberate sub-tolerance geometry-signature noise
            # (unique_dims) — rounding would erase it. See the frozenset doc.
            continue

        def _round_float(match):
            val = float(match.group(0))
            rounded = round(val, DEDUP_LENGTH_DECIMALS)
            if rounded == int(rounded):
                return f'{int(rounded)}.'
            return f'{rounded:.{DEDUP_LENGTH_DECIMALS}f}'.rstrip('0')

        new_args = _FLOAT_RE.sub(_round_float, args)
        entities[eid] = (etype, new_args)


def _drop_purged(entities: Dict[int, Tuple[str, str]],
                 drop_ids: Optional[Set[int]]) -> int:
    """Delete ``drop_ids`` from ``entities`` in ONE pass. Returns the count.

    **Why the purge lands here instead of on the model.** The Product geometry
    purge deletes the follower occurrences' now-orphaned bodies, which WR11
    requires (an ``IfcShapeRepresentation`` used by no product shape, no
    representation map and no shape aspect is a schema error). Doing it with
    ``ifcopenshell.file.remove`` costs a scan of the whole file PER entity, to
    null out inverse references — measured on ``unitized-curtain-wall``,
    60,189 removals at 6.3 ms each, 381.8 s, about 72% of that model's compile
. ``file.batch()``/``unbatch()`` does not help: it defers
    that bookkeeping rather than avoiding it, moving the same 382 s from
    ``file_remove`` to ``file_unbatch``. Here the file is already parsed into a
    dict, so the whole purge is a dict-delete.

    **The safety property, asserted rather than assumed.**
    ``product_types.purgeable_after_sharing`` runs a fixpoint that drops any
    candidate having a referrer outside the candidate set, so the set it
    returns is REFERENCE-CLOSED — nothing surviving points into it. That is
    what makes deleting outright equivalent to ``file.remove``'s
    null-out-the-references behaviour: there are no references to null. If that
    ever stops being true the file would carry a dangling ``#id`` and still
    parse, so this checks instead of trusting.
    """
    if not drop_ids:
        return 0
    present = drop_ids & set(entities)
    for eid in present:
        del entities[eid]
    if present:
        dangling = {
            eid: ref
            for eid, (_etype, args) in entities.items()
            for ref in (int(m.group(1)) for m in _REF_RE.finditer(args))
            if ref in present
        }
        if dangling:
            sample = list(dangling.items())[:5]
            raise ValueError(
                f"purge set is not reference-closed: {len(dangling)} surviving "
                f"entities still reference purged ones (e.g. {sample}). "
                "purgeable_after_sharing's fixpoint should make this "
                "impossible — do not 'fix' this by widening the purge.")
    return len(present)


def _build_dependencies(entities: Dict[int, Tuple[str, str]]) -> Dict[int, Set[int]]:
    """Build dependency graph: entity -> set of entity IDs it references."""
    deps = {}
    entity_ids = set(entities.keys())
    for eid, (etype, args) in entities.items():
        refs = set()
        for m in _REF_RE.finditer(args):
            ref_id = int(m.group(1))
            if ref_id in entity_ids:
                refs.add(ref_id)
        deps[eid] = refs
    return deps


def _topological_sort(entities: Dict[int, Tuple[str, str]],
                      deps: Dict[int, Set[int]]) -> List[int]:
    """Topological sort: leaves first, then composites."""
    visited = set()
    result = []

    def visit(eid):
        if eid in visited:
            return
        visited.add(eid)
        for dep in deps.get(eid, set()):
            if dep in entities:
                visit(dep)
        result.append(eid)

    for eid in sorted(entities.keys()):
        visit(eid)

    return result


def _find_duplicates(entities: Dict[int, Tuple[str, str]],
                     sorted_ids: List[int]) -> Dict[int, int]:
    """
    Find duplicate entities. Returns mapping {duplicate_id: canonical_id}.

    Processes in topological order so child references are already canonicalized.
    """
    canonical_map = {}  # duplicate -> canonical
    seen = {}  # canonical_form -> first_id

    for eid in sorted_ids:
        etype, args = entities[eid]

        # Only deduplicate safe types
        if etype not in DEDUP_SAFE_TYPES:
            continue

        # Canonicalize references in args (replace duplicates with their canonical)
        canonical_args = _replace_refs(args, canonical_map)
        canonical_form = (etype, canonical_args)

        if canonical_form in seen:
            canonical_map[eid] = seen[canonical_form]
        else:
            seen[canonical_form] = eid

    return canonical_map


def _replace_refs(args: str, canonical_map: Dict[int, int]) -> str:
    """Replace entity references with their canonical equivalents."""
    def _sub(m):
        ref_id = int(m.group(1))
        canonical = canonical_map.get(ref_id, ref_id)
        return f'#{canonical}'
    return _REF_RE.sub(_sub, args)


def _rewrite_and_remove(entities: Dict[int, Tuple[str, str]],
                        canonical_map: Dict[int, int]) -> Dict[int, Tuple[str, str]]:
    """Rewrite references in surviving entities and remove duplicates."""
    surviving = {}
    for eid, (etype, args) in entities.items():
        if eid in canonical_map:
            continue  # Skip duplicate
        new_args = _replace_refs(args, canonical_map)
        surviving[eid] = (etype, new_args)
    return surviving


def _renumber(entities: Dict[int, Tuple[str, str]]) -> List[Tuple[int, str, str]]:
    """Renumber entities sequentially starting from 1."""
    # Sort by original ID to preserve order
    sorted_items = sorted(entities.items())

    # Build old->new ID mapping
    id_map = {}
    for new_id, (old_id, _) in enumerate(sorted_items, start=1):
        id_map[old_id] = new_id

    # Rewrite references with new IDs
    result = []
    for old_id, (etype, args) in sorted_items:
        new_id = id_map[old_id]

        def _sub(m):
            ref = int(m.group(1))
            return f'#{id_map.get(ref, ref)}'

        new_args = _REF_RE.sub(_sub, args)
        result.append((new_id, etype, new_args))

    return result


def _reconstruct(header: List[str], renumbered: List[Tuple[int, str, str]],
                 footer: List[str]) -> str:
    """Reconstruct the STEP file from sections."""
    lines = header[:]
    for eid, etype, args in renumbered:
        lines.append(f'#{eid}={etype}({args});')
    lines.extend(footer)
    return '\n'.join(lines) + '\n'
