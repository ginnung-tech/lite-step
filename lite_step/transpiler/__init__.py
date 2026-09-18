"""
IFC Import Transpiler: Round-trip and cold import entry points.

round_trip(ifc_content) — extract embedded source, diff, patch, recompile
cold_import(ifc_content) — heuristic reconstruction for third-party IFC
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# LLM fallback threshold: >30% named elements changed
LLM_THRESHOLD_FRACTION = 0.30


@dataclass
class ImportReport:
    """Report of what was and wasn't imported from an IFC file."""
    total_ifc_products: int = 0
    mapped_elements: int = 0
    skipped_by_type: Dict[str, int] = field(default_factory=dict)
    schema: str = "UNKNOWN"
    schema_rewritten: bool = False
    parse_error: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class RoundTripResult:
    """Result of a round-trip or cold import operation."""
    success: bool
    source_code: Optional[str] = None
    ifc_content: Optional[str] = None
    mode: str = "round_trip"  # "round_trip", "cold_import", "llm_assisted"
    changes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    needs_llm: bool = False
    llm_prompt: Optional[str] = None
    import_report: Optional[ImportReport] = None
    # Cold-import path only. Counts of exact / approximated / opaque
    # per-element emissions plus by_type breakdown. The patch UI uses
    # this to render the lossiness banner ("12 of 47 entities became
    # opaque meshes; edits to those will not round-trip cleanly").
    coverage_report: Optional["CoverageReport"] = None


def round_trip(ifc_content: str) -> RoundTripResult:
    """
    Import a (possibly modified) IFC file and produce updated Lite-STEP code.

    If the IFC contains embedded LITESTEP_META, extracts the original source,
    diffs against the modified IFC, and patches the source code.
    Falls back to cold_import() if no metadata found.

    Args:
        ifc_content: IFC STEP file content as string

    Returns:
        RoundTripResult with patched source and recompiled IFC
    """
    from lite_step.ifc.embedder import extract_litestep_meta
    from lite_step.ifc.normalizer import normalize_ifc

    # Step 1: Extract embedded metadata
    meta = extract_litestep_meta(ifc_content)
    if meta is None:
        logger.info("No LITESTEP_META found — delegating to cold_import")
        return cold_import(ifc_content)

    if not meta.hash_valid:
        logger.warning("Source hash mismatch — proceeding with embedded source as canonical")

    try:
        # Step 2: Re-execute original source to get original elements
        original_elements = _compile_and_normalize(meta.source)

        # Step 3: Normalize modified IFC
        modified_elements = normalize_ifc(ifc_content)

        if not modified_elements:
            return RoundTripResult(
                success=False,
                error="Failed to normalize modified IFC — no elements extracted",
            )

        # Step 4: Walk source with LibCST
        from lite_step.transpiler.ast_walker import walk_source
        cst_nodes = walk_source(meta.source)

        # Step 5: Diff elements
        from lite_step.transpiler.differ import diff_elements, DeltaCategory
        deltas = diff_elements(original_elements, modified_elements)

        # Step 6: Check LLM threshold
        named_count = sum(
            1 for d in deltas if d.category != DeltaCategory.UNCHANGED
        )
        total_count = len(deltas) or 1
        change_fraction = named_count / total_count

        needs_llm = change_fraction > LLM_THRESHOLD_FRACTION

        # Step 7: Analyze loops
        from lite_step.transpiler.loop_analysis import analyze_loops
        all_names = [e.name for e in modified_elements]
        loop_analyses = analyze_loops(deltas, cst_nodes, all_names)

        # Check for loop fallbacks
        for la in loop_analyses:
            if la.action == "llm_fallback":
                needs_llm = True
                break

        # Step 8: Apply patches
        from lite_step.transpiler.patcher import apply_patches
        patched_source, unhandled = apply_patches(
            meta.source, deltas, loop_analyses, cst_nodes
        )

        if unhandled:
            needs_llm = True

        # Build change descriptions
        changes = _describe_changes(deltas)
        warnings = []

        if needs_llm:
            llm_prompt = _build_llm_prompt(deltas, unhandled)
            return RoundTripResult(
                success=True,
                source_code=patched_source,
                mode="llm_assisted",
                changes=changes,
                warnings=warnings,
                needs_llm=True,
                llm_prompt=llm_prompt,
            )

        # Step 9: Validate and recompile
        result = _validate_and_recompile(patched_source)
        if result.error:
            # Try LLM fallback on compilation failure
            return RoundTripResult(
                success=True,
                source_code=patched_source,
                mode="llm_assisted",
                changes=changes,
                warnings=[f"Compilation failed: {result.error}"],
                needs_llm=True,
                llm_prompt=_build_compilation_fix_prompt(patched_source, result.error),
            )

        return RoundTripResult(
            success=True,
            source_code=patched_source,
            ifc_content=result.ifc_content,
            mode="round_trip",
            changes=changes,
            warnings=warnings,
        )

    except Exception as e:
        # Plan §3.D E.4: compiler-development trace, not an infra bug.
        logger.warning("Round-trip failed: %s", e, exc_info=True)
        return RoundTripResult(
            success=False,
            error=f"Round-trip failed: {e}",
        )


def cold_import(ifc_content: str) -> RoundTripResult:
    """
    Generate Lite-STEP from a third-party IFC with no embedded source.

    Always succeeds — produces an empty building with import notes when
    no elements can be mapped. No element-count cap; deterministic
    reconstruction handles any number of elements.

    Args:
        ifc_content: IFC STEP file content as string

    Returns:
        RoundTripResult with generated source and import report
    """
    from lite_step.ifc.normalizer import normalize_ifc_full
    from lite_step.transpiler.reconstructor import reconstruct_litestep

    try:
        # Step 1: Normalize (full result with metadata)
        norm = normalize_ifc_full(ifc_content)
        elements = norm.elements

        report = ImportReport(
            total_ifc_products=norm.total_products_found,
            mapped_elements=len(elements),
            skipped_by_type=norm.skipped_types,
            schema=norm.schema,
            schema_rewritten=norm.schema_rewritten,
            parse_error=norm.parse_error,
        )

        if norm.parse_error:
            report.notes.append(f"Parse error: {norm.parse_error}")

        if norm.schema_rewritten:
            report.notes.append(f"Schema {norm.schema} rewritten for parsing")

        if norm.skipped_types:
            for ifc_type, count in sorted(norm.skipped_types.items()):
                report.notes.append(f"Skipped {count}x {ifc_type}")

        warnings = ["Round-trip source not found -- structure not preserved"]

        if not elements:
            # Empty building — always succeeds
            report.notes.insert(0, "0 mappable elements found")
            source, coverage = reconstruct_litestep([], import_report=report)
            return RoundTripResult(
                success=True,
                source_code=source,
                mode="cold_import",
                changes=["Cold import: 0 elements (empty building)"],
                warnings=warnings + report.notes,
                import_report=report,
                coverage_report=coverage,
            )

        # Step 2: Reconstruct (no element-count cap)
        source, coverage = reconstruct_litestep(elements, import_report=report)

        # Step 3: Optional LLM enhancement hint (not a gate)
        llm_enhancement = False
        llm_prompt = None
        if len(elements) > 50:
            type_counts = Counter(e.ifc_type for e in elements)
            if any(c > 5 for c in type_counts.values()):
                llm_enhancement = True
                llm_prompt = _build_enhancement_prompt(elements)

        # Step 4: Validate and recompile
        result = _validate_and_recompile(source)
        if result.error:
            warnings.append(f"Compilation failed: {result.error}")
            return RoundTripResult(
                success=True,
                source_code=source,
                mode="cold_import",
                changes=[f"Cold import: {len(elements)} elements"],
                warnings=warnings,
                needs_llm=llm_enhancement,
                llm_prompt=llm_prompt or _build_compilation_fix_prompt(source, result.error),
                import_report=report,
                coverage_report=coverage,
            )

        return RoundTripResult(
            success=True,
            source_code=source,
            ifc_content=result.ifc_content,
            mode="cold_import",
            changes=[f"Cold import: {len(elements)} elements reconstructed"],
            warnings=warnings,
            needs_llm=llm_enhancement,
            llm_prompt=llm_prompt,
            import_report=report,
            coverage_report=coverage,
        )

    except Exception as e:
        # Plan §3.D E.4: compiler-development trace, not an infra bug.
        logger.warning("Cold import failed: %s", e, exc_info=True)
        # Even exceptions produce an empty building, never fail
        from lite_step.transpiler.reconstructor import reconstruct_litestep as _rls
        report = ImportReport(parse_error=str(e), notes=[f"Exception: {e}"])
        source, coverage = _rls([], import_report=report)
        return RoundTripResult(
            success=True,
            source_code=source,
            mode="cold_import",
            changes=["Cold import: exception recovery (empty building)"],
            warnings=[str(e)],
            import_report=report,
            coverage_report=coverage,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@dataclass
class _CompileResult:
    ifc_content: Optional[str] = None
    error: Optional[str] = None


def _compile_and_normalize(source: str):
    """Execute source → Project → generate_ifc → normalize → elements."""
    from lite_step.compiler.executor import execute_lite_step_script
    from lite_step.ifc.generator import generate_ifc
    from lite_step.ifc.normalizer import normalize_ifc

    exec_result = execute_lite_step_script(source)
    if not exec_result.success or exec_result.project is None:
        raise RuntimeError(f"Failed to compile original source: {exec_result.error}")

    ifc_result = generate_ifc(exec_result.project)
    if not ifc_result.success or ifc_result.ifc_content is None:
        raise RuntimeError(f"Failed to generate IFC from original source: {ifc_result.error}")

    return normalize_ifc(ifc_result.ifc_content)


def _validate_and_recompile(source: str) -> _CompileResult:
    """Execute source and recompile to IFC with embedding."""
    from lite_step.compiler.executor import execute_lite_step_script
    from lite_step.ifc.generator import generate_ifc

    exec_result = execute_lite_step_script(source)
    if not exec_result.success or exec_result.project is None:
        return _CompileResult(error=exec_result.error or "Compilation failed")

    ifc_result = generate_ifc(exec_result.project, source_code=source)
    if not ifc_result.success or ifc_result.ifc_content is None:
        return _CompileResult(error=ifc_result.error or "IFC generation failed")

    return _CompileResult(ifc_content=ifc_result.ifc_content)


def _describe_changes(deltas) -> List[str]:
    """Generate human-readable change descriptions."""
    from lite_step.transpiler.differ import DeltaCategory

    descriptions = []
    for d in deltas:
        if d.category == DeltaCategory.UNCHANGED:
            continue
        elif d.category == DeltaCategory.PLACEMENT_MOVED:
            delta = d.details.get("translation_delta", (0, 0, 0))
            descriptions.append(
                f"{d.element_id}: moved ({delta[0]:+d}, {delta[1]:+d}, {delta[2]:+d})mm"
            )
        elif d.category == DeltaCategory.PLACEMENT_ROTATED:
            descriptions.append(f"{d.element_id}: rotated")
        elif d.category == DeltaCategory.GEOMETRY_RESIZED:
            descriptions.append(f"{d.element_id}: resized")
        elif d.category == DeltaCategory.GEOMETRY_CONTOUR:
            descriptions.append(f"{d.element_id}: contour changed")
        elif d.category == DeltaCategory.STOREY_MOVED:
            descriptions.append(
                f"{d.element_id}: moved to storey {d.details.get('new_storey', '?')}"
            )
        elif d.category == DeltaCategory.DELETED:
            descriptions.append(f"{d.element_id}: deleted")
        elif d.category == DeltaCategory.ADDED:
            descriptions.append(f"{d.element_id}: added")

    return descriptions


def _build_llm_prompt(deltas, unhandled) -> str:
    """Build the LLM delta prompt per spec §4.1."""
    from lite_step.transpiler.differ import DeltaCategory

    lines = [
        "# Delta Modification",
        "",
        "Read the current code at output.py, then apply the changes below.",
        "After editing, run `python output.py` to verify compilation.",
        "",
        "## Changes Required",
        "",
        "The user modified the IFC file in an external editor. The following changes",
        "were detected but are too complex for automatic patching. Update the",
        "Lite-STEP code to match these new conditions:",
        "",
    ]

    for d in deltas:
        if d.category == DeltaCategory.UNCHANGED:
            continue
        if d.category == DeltaCategory.PLACEMENT_MOVED:
            delta = d.details.get("translation_delta", (0, 0, 0))
            old = d.details.get("old_translation", (0, 0, 0))
            new = d.details.get("new_translation", (0, 0, 0))
            lines.append(
                f"- {d.element_id}: PLACEMENT_MOVED -- "
                f"translated ({delta[0]:+d}, {delta[1]:+d}, {delta[2]:+d})mm "
                f"(old: {old[0]},{old[1]},{old[2]} -> new: {new[0]},{new[1]},{new[2]})"
            )
        elif d.category == DeltaCategory.DELETED:
            lines.append(f"- Deleted: '{d.element_id}' -- no longer present in IFC")
        elif d.category == DeltaCategory.ADDED:
            if d.modified:
                lines.append(
                    f"- New element: {d.modified.ifc_type} '{d.element_id}' detected"
                )
        else:
            lines.append(f"- {d.element_id}: {d.category.value}")

    lines.extend([
        "",
        "IMPORTANT:",
        "- Only change what's needed for this modification.",
        "- Use Edit tool for targeted changes.",
        "- Preserve loops and helper functions where possible.",
        "- Fix any compilation errors before finishing.",
    ])

    return "\n".join(lines)


def _build_compilation_fix_prompt(source: str, error: str) -> str:
    """Build a prompt for the LLM to fix compilation errors."""
    return (
        "# Compilation Fix\n\n"
        "The imported code at output.py has a compilation error.\n"
        f"Error: {error}\n\n"
        "Read the code, fix the error, and run `python output.py` to verify.\n"
        "Make minimal changes to fix the compilation issue.\n"
    )


def _build_cold_import_llm_prompt(elements) -> str:
    """Build a prompt for LLM-assisted cold import of complex IFC files."""
    lines = [
        "# IFC Cold Import (Complex)",
        "",
        f"A third-party IFC file with {len(elements)} elements needs reconstruction.",
        "The partial code at output.py contains the first 100 elements.",
        "Complete the reconstruction to include all elements.",
        "",
        "## Element Summary",
        "",
    ]

    type_counts = Counter(e.ifc_type for e in elements)
    for ifc_type, count in type_counts.most_common():
        lines.append(f"- {ifc_type}: {count}")

    lines.extend([
        "",
        "IMPORTANT:",
        "- Generate flat Lite-STEP (no loops, no helpers)",
        "- Use standard element types: Wall, Column, Beam, Solid, etc.",
        "- Wrap in generate_project() with proper storey assignment",
        "- Run `python output.py` to verify compilation",
    ])

    return "\n".join(lines)


def _build_enhancement_prompt(elements) -> str:
    """Build prompt for optional LLM enhancement of flat reconstruction."""
    type_counts = Counter(e.ifc_type for e in elements)

    lines = [
        "# Lite-STEP Enhancement",
        "",
        f"The flat code at output.py has {len(elements)} elements.",
        "Improve it by:",
        "- Detecting repetitive patterns and converting to `for` loops",
        "- Improving element names (from IFC names or sequential patterns)",
        "- Adding helper functions for repeated geometry patterns",
        "",
        "Element type distribution:",
    ]
    for ifc_type, count in type_counts.most_common():
        lines.append(f"  {ifc_type}: {count}")

    lines.extend([
        "",
        "IMPORTANT:",
        "- The flat code already compiles correctly. Only improve structure.",
        "- Run `python output.py` to verify compilation after changes.",
        "- Do NOT change geometry values or coordinates.",
    ])
    return "\n".join(lines)
