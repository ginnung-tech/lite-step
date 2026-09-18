"""Patch-mode entry point for Lite-STEP.

Every ``compile_main`` call in Lite-STEP is conceptually a patch: the
script's ``Project`` is the desired state, applied on top of an
optional adjacent base IFC. ``patch_main`` is the programmatic flavour
of the same operation — useful when the caller has the base IFC bytes
and the desired-state source in memory (typical in worker dispatch
handlers) and doesn't want to round-trip through the file system.

``patch_main(base_ifc, source)``:

- Detects ``LITESTEP_META`` in the base IFC.
- Runs ``source`` through the existing compile pipeline (which embeds
  fresh ``LITESTEP_META`` in the output).
- When the base had no ``LITESTEP_META`` (cold path) — or had one
  already flagged ``cold_imported=true`` — the output IFC is post-
  marked with ``cold_imported=true`` so the patch UI keeps showing
  the lossiness banner across subsequent edits of the same file.

The CLI / model-file surface uses the module-global declaration
``base_ifc = "./input.ifc"`` next to ``result = Project(...)``::

    base_ifc = "./input.ifc"          # resolved relative to the model file
    result = Project(name="patch", ...)

When ``base_ifc`` is absent, the patch lands on an empty base —
equivalent to a plain generate. When it names a file that does not
exist, ``compile_main`` warns and falls back to that empty base rather
than failing, so a missing base is visible but never fatal.

(Declared inline here rather than by reference: the worked example this
it pointed at, ``examples/03_patch_existing_ifc.py``, does not exist, and a
docstring that names a missing file is worse than one that shows the
two lines outright.)

``editIntent`` dispatch (structured / NL / reflow) lands in Phase D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PatchResult:
    """Outcome of a single ``patch_main`` invocation.

    Fields mirror what the patch UI needs to render: the new IFC bytes,
    the source that produced them (for next-edit hot-path resolution),
    the mode label, any warnings worth surfacing in the SPA banner.

    ``needs_llm`` + ``llm_prompt`` mirror the same shape on
    :class:`lite_step.transpiler.RoundTripResult` — the worker
    dispatcher checks these flags and routes to an LLM call when the
    deterministic recompile path can't produce a valid IFC.
    ``coverage_report`` is reserved for the cold path (populated once
    cold_import is invoked inline; Phase B.1 leaves it None and relies
    on the script supplying a desired-state ``source`` already).
    """
    success: bool
    source_code: Optional[str] = None
    ifc_content: Optional[str] = None
    mode: str = "round_trip"  # "round_trip" | "cold_import" | "llm_assisted"
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    needs_llm: bool = False
    llm_prompt: Optional[str] = None


def patch_main(base_ifc_content: str, source_code: str) -> PatchResult:
    """Recompile ``source_code`` and emit a patched IFC for the given base.

    Hot path (LITESTEP_META present in ``base_ifc_content``): the new
    IFC carries fresh embedded source + manifest + hash. Mode is
    reported as ``round_trip``.

    Cold path (no META in the base, or META with ``cold_imported=true``):
    the new IFC is post-marked ``cold_imported=true`` so the UI keeps
    its lossiness banner. Mode is reported as ``cold_import``. Note:
    Phase B.1 assumes ``source_code`` is already the desired-state
    script the caller produced from cold-importing the base; this
    function does NOT re-run the cold-import reconstructor. Callers
    that need full cold-then-patch in one shot use
    ``transpiler.cold_import`` to obtain the desired-state source
    first, edit it, then pass it here.
    """
    from lite_step.ifc.embedder import (
        extract_litestep_meta,
        set_cold_imported_flag,
        base_version_refusal,
    )
    from lite_step.transpiler import (
        _build_compilation_fix_prompt,
        _validate_and_recompile,
    )

    meta = extract_litestep_meta(base_ifc_content)
    # Forward-only version gate — the same one ``compile_main`` applies to an
    # adjacent ``base_ifc``. A base may be patched only by the compiler that
    # wrote it; anything else is refused. A base with NO META is a legitimate
    # third-party cold-import, exempt. See ``embedder.base_version_refusal``.
    err_msg = base_version_refusal(meta, base_name="The base IFC")
    if err_msg is not None:
        return PatchResult(
            success=False,
            source_code=source_code,
            mode="round_trip",
            warnings=[err_msg],
            error=err_msg,
            needs_llm=False,
        )
    if meta is None:
        cold_imported = True
        mode = "cold_import"
    else:
        cold_imported = meta.cold_imported
        mode = "cold_import" if cold_imported else "round_trip"

    result = _validate_and_recompile(source_code)
    if result.error or not result.ifc_content:
        # Compilation failed: surface needs_llm so the worker dispatcher
        # can route to an LLM-fix turn. The patch UI keeps the editor
        # open with the error banner; the LLM call lands the fix in a
        # follow-up patch_main invocation.
        err_msg = result.error or "No IFC content produced"
        return PatchResult(
            success=False,
            source_code=source_code,
            mode="llm_assisted",
            warnings=[err_msg],
            error=err_msg,
            needs_llm=True,
            llm_prompt=_build_compilation_fix_prompt(source_code, err_msg),
        )

    ifc_out = result.ifc_content
    if cold_imported:
        ifc_out = set_cold_imported_flag(ifc_out)

    return PatchResult(
        success=True,
        source_code=source_code,
        ifc_content=ifc_out,
        mode=mode,
        warnings=[],
    )
