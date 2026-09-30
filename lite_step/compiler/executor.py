"""
Safe Execution of LLM-Generated Lite-STEP Scripts.

This module executes Python scripts in a restricted namespace
and extracts the Project object for IFC compilation.
"""

import inspect
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional
import traceback

from .namespace import create_namespace, get_restricted_builtins
from lite_step.models import Mesh, Point, Project
from lite_step.models import taxonomy as tx
from lite_step.strict import suspends_strict

# Configure module logger
logger = logging.getLogger(__name__)

# Enable verbose logging for sandbox debugging
# Set to True in production to see detailed execution traces
VERBOSE_SANDBOX_LOGGING = True


@dataclass
class ExecutionResult:
    """
    Result of LLM script execution.

    Attributes:
        success: True if script executed and produced a valid Project
        project: The extracted Project object (if successful)
        error: Error message (if failed)
        error_type: Type of error (SyntaxError, NameError, etc.)
        line_number: Line number where error occurred (if applicable)
        traceback_str: Full traceback for debugging
    """
    success: bool
    project: Optional[Project] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    line_number: Optional[int] = None
    traceback_str: Optional[str] = None


def execute_lite_step_script(script: str) -> ExecutionResult:
    """
    Execute LLM-generated Lite-STEP script and extract Project object.

    The script MUST define a variable named 'result' containing a Project instance.

    Example valid script:
        def generate_project():
            proj = Project(name="My House")
            proj.add(Wall(start=Point(x=0, y=0, z=0), end=Point(x=5000, y=0, z=0)))
            return proj

        result = generate_project()

    Security defense layers, each supplied by the host:

    1. Layer 1 — AST pre-exec guard (``security.ast_guard``): rejects
       obvious network/shell/code-exec imports before ``compile()``.
    2. Layer 2 — runtime egress allowlist (``security.egress_guard``):
       blocks Python-level outbound connections in the trusted parent
       worker to destinations outside the allowlist. Installed by the
       worker entrypoint.
    3. Layer 3 — subprocess isolation (``security.sandboxed_exec``):
       when enabled, ``exec()`` of the compiled script runs in a child
       process that has unshared its network namespace
       (``CLONE_NEWUSER | CLONE_NEWNET``). The kernel — not the Python
       shim — enforces "zero sockets": ``getaddrinfo`` / ``connect``
       fail with ``ENETUNREACH`` / ``EADDRNOTAVAIL`` for any real
       destination, and a ``ctypes.CDLL('libc.so.6').socket(...)``
       bypass of Layers 1+2 still cannot reach the network.

    The first two layers stay installed regardless. Layer 3 can be
    disabled for local dev/tests by the host that installs it.

    Args:
        script: Python script content as string

    Returns:
        ExecutionResult with Project on success, or error details on failure
    """
    if not script or not script.strip():
        logger.warning("[Sandbox] Empty script provided")
        return ExecutionResult(
            success=False,
            error="Empty script provided",
            error_type="ValueError"
        )

    if VERBOSE_SANDBOX_LOGGING:
        logger.info("[Sandbox] Received script (%d chars)", len(script))
        # Log first 500 chars for debugging (avoid huge logs)
        script_preview = script[:500] + "..." if len(script) > 500 else script
        logger.debug("[Sandbox] Script preview:\n%s", script_preview)

    # Security Layer 1: static AST check before we ever touch compile()/exec().
    # Rejects socket/subprocess/eval/miner-keyword scripts.
    #
    # Supplied by the HOST through ``compiler.sandbox``, not imported from
    # the host by name — see that module for why, and for what a host-less
    # (public) Lite-STEP does instead. A host installs its hooks at import
    # installs it.
    from lite_step.compiler.sandbox import assert_installed, get_sandbox

    # Loud refusal where the deployment says containment is mandatory. The
    # seat can only be filled by a host that was imported; this is what stops
    # "nobody installed a sandbox" from degrading to unguarded exec in silence.
    assert_installed()

    _hooks = get_sandbox()
    _ast_check = _hooks.check_code
    _UnsafeCodeError = _hooks.unsafe_code_error

    if _ast_check is not None:
        try:
            _ast_check(script)
        except _UnsafeCodeError as exc:  # type: ignore[misc]
            reason = exc.args[0] if exc.args else "unsafe code"
            logger.error("[Sandbox] AST guard rejected script: %s", reason)
            return ExecutionResult(
                success=False,
                error=f"unsafe_code: {reason}",
                error_type="UnsafeCodeError",
            )

    # Security Layer 3: subprocess isolation. When enabled, exec() runs
    # in a child with CLONE_NEWNET so the kernel — not the Python
    # socket shim — enforces "no egress". The child runs the *same*
    # in-process executor (``execute_lite_step_script_inproc``), so the
    # output contract is identical.
    run_sandboxed = _hooks.run_sandboxed
    subprocess_sandbox_enabled = _hooks.subprocess_enabled or (lambda: False)

    if run_sandboxed is not None and subprocess_sandbox_enabled():
        return _execute_via_subprocess(script, run_sandboxed)

    return execute_lite_step_script_inproc(script)


def _execute_via_subprocess(script: str, run_sandboxed) -> ExecutionResult:
    """Dispatch to Layer 3 subprocess and map SandboxResult → ExecutionResult.

    Kept separate so unit tests can target it directly with a mocked
    ``run_sandboxed``.
    """
    if VERBOSE_SANDBOX_LOGGING:
        logger.info("[Sandbox] Layer 3 enabled — dispatching to subprocess runner")
    sb = run_sandboxed(script)
    if sb.success:
        if VERBOSE_SANDBOX_LOGGING:
            logger.info(
                "[Sandbox] Layer 3 OK (child pid=%s, exit=%s)",
                sb.child_pid,
                sb.child_exitcode,
            )
        return ExecutionResult(success=True, project=sb.project)
    logger.warning(
        "[Sandbox] Layer 3 failed: type=%s message=%s child_pid=%s exit=%s",
        sb.error_type,
        sb.error,
        sb.child_pid,
        sb.child_exitcode,
    )
    return ExecutionResult(
        success=False,
        error=sb.error,
        error_type=sb.error_type,
        line_number=sb.line_number,
        traceback_str=sb.traceback_str,
    )


def execute_lite_step_script_inproc(script: str) -> ExecutionResult:
    """
    In-process core of ``execute_lite_step_script`` — actually runs exec().

    This is the function the Layer 3 child process calls after it has
    unshared its network namespace. Callers that don't need Layer 3
    (local tests, the child itself) can invoke this directly; the
    parent worker should always go through ``execute_lite_step_script``.

    The AST guard (Layer 1) is **not** re-run here — ``execute_lite_step_script``
    and the child-process parent have already applied it. Re-running
    would be redundant and slow.

    Args:
        script: Python script content as string

    Returns:
        ExecutionResult with Project on success, or error details on failure
    """
    if not script or not script.strip():
        return ExecutionResult(
            success=False,
            error="Empty script provided",
            error_type="ValueError",
        )

    # Pre-compile to check syntax before execution
    # This gives clearer error messages and catches issues early
    try:
        if VERBOSE_SANDBOX_LOGGING:
            logger.info("[Sandbox] Pre-compiling script...")
        compiled_code = compile(script, "<lite-step-script>", "exec")
        if VERBOSE_SANDBOX_LOGGING:
            logger.info("[Sandbox] Pre-compilation successful")
    except SyntaxError as e:
        # Plan §3.D E.3: user-generated sandbox Python, not an infra bug.
        logger.warning("[Sandbox] Syntax error at line %s: %s", e.lineno, e.msg)
        return ExecutionResult(
            success=False,
            error=f"Syntax error: {e.msg}",
            error_type="SyntaxError",
            line_number=e.lineno,
            traceback_str=traceback.format_exc()
        )

    # Create restricted namespace with model classes and builtins
    namespace = create_namespace()
    # Merge builtins into namespace - functions need model classes in globals
    restricted_builtins = get_restricted_builtins()
    namespace["__builtins__"] = restricted_builtins

    if VERBOSE_SANDBOX_LOGGING:
        logger.info("[Sandbox] Namespace initialized with %d builtins: %s",
                    len(restricted_builtins), list(restricted_builtins.keys()))

    try:
        # Execute pre-compiled code - namespace is both globals and locals
        # This ensures functions defined in the script can access Project, Point, etc.
        if VERBOSE_SANDBOX_LOGGING:
            logger.info("[Sandbox] Executing script...")
        exec(compiled_code, namespace, namespace)
        if VERBOSE_SANDBOX_LOGGING:
            logger.info("[Sandbox] Script executed successfully")

        # Extract 'result' variable (primary) or call main() (fallback)
        if "result" not in namespace:
            # Fallback: try calling main() if defined
            main_fn = namespace.get("main")
            if callable(main_fn):
                try:
                    result = main_fn()
                except Exception as e:
                    return ExecutionResult(
                        success=False,
                        error=f"main() raised {type(e).__name__}: {e}",
                        error_type=type(e).__name__,
                        traceback_str=traceback.format_exc()
                    )
            else:
                return ExecutionResult(
                    success=False,
                    error="Script must define a 'result' variable containing a Project object, "
                          "or a main() function that returns one. "
                          "Example: result = generate_project()",
                    error_type="MissingResult"
                )
        else:
            result = namespace["result"]

        # Validate result is a Project
        if not isinstance(result, Project):
            return ExecutionResult(
                success=False,
                error=f"'result' must be a Project instance, got {type(result).__name__}. "
                      f"Make sure you return Project(...) not a function or other type.",
                error_type="TypeError"
            )

        # Normalize from millimeters to meters for IFC generation
        # (IFC file uses meters as base unit)
        normalized_project = normalize_project_to_meters(result)

        return ExecutionResult(success=True, project=normalized_project)

    except SyntaxError as e:
        return ExecutionResult(
            success=False,
            error=f"Syntax error: {e.msg}",
            error_type="SyntaxError",
            line_number=e.lineno,
            traceback_str=traceback.format_exc()
        )

    except NameError as e:
        # Extract the undefined name from the error message
        # Plan §3.D E.3: user-generated sandbox Python, not an infra bug.
        logger.warning("[Sandbox] NameError: %s", e)
        if VERBOSE_SANDBOX_LOGGING:
            logger.debug("[Sandbox] Full traceback:\n%s", traceback.format_exc())
        return ExecutionResult(
            success=False,
            error=f"Undefined name: {e}. Check spelling and ensure all variables are defined.",
            error_type="NameError",
            traceback_str=traceback.format_exc()
        )

    except TypeError as e:
        # Plan §3.D E.3: user-generated sandbox Python, not an infra bug.
        logger.warning("[Sandbox] TypeError: %s", e)
        if VERBOSE_SANDBOX_LOGGING:
            logger.debug("[Sandbox] Full traceback:\n%s", traceback.format_exc())
        return ExecutionResult(
            success=False,
            error=f"Type error: {e}. Check argument types match expected parameters.",
            error_type="TypeError",
            traceback_str=traceback.format_exc()
        )

    except KeyError as e:
        # Plan §3.D E.3: user-generated sandbox Python, not an infra bug.
        logger.warning("[Sandbox] KeyError: %s", e)
        if VERBOSE_SANDBOX_LOGGING:
            logger.debug("[Sandbox] Full traceback:\n%s", traceback.format_exc())
        return ExecutionResult(
            success=False,
            error=f"Unknown key: {e}. If this is a material key, check the registry (lite_step.materials) for valid keys.",
            error_type="KeyError",
            traceback_str=traceback.format_exc()
        )

    except ImportError as e:
        # Special handling for blocked imports
        # Plan §3.D E.3: user-generated sandbox Python, not an infra bug.
        logger.warning("[Sandbox] ImportError (blocked): %s", e)
        return ExecutionResult(
            success=False,
            error=str(e),
            error_type="ImportError",
            traceback_str=traceback.format_exc()
        )

    except Exception as e:
        # Generic error handler
        # Plan §3.D E.3: user-generated sandbox Python, not an infra bug.
        logger.warning("[Sandbox] %s: %s", type(e).__name__, e)
        if VERBOSE_SANDBOX_LOGGING:
            logger.debug("[Sandbox] Full traceback:\n%s", traceback.format_exc())
        return ExecutionResult(
            success=False,
            error=f"{type(e).__name__}: {e}",
            error_type=type(e).__name__,
            traceback_str=traceback.format_exc()
        )


def _normalize_private_modifiers(elem) -> None:
    """
    Normalize _cuts, _adds, _intersects, _fills, _openings PrivateAttr lists.

    Converts all nested element coordinates from mm to meters.
    Handles both box mode (start/end) and contour mode (contour/thickness).
    Also normalizes Window/Door opening children and their sub-elements.

    ``_intersects`` (``.intersection()``) is scaled here too — it is a boolean
    operand exactly like ``_cuts``/``_adds``. ``_voids`` is DELIBERATELY excluded — it is scaled
    by ``_normalize_voids_recursive``; including it here would double-scale.

    Args:
        elem: Any BimElement with _cuts, _adds, _intersects, _fills, _openings
    """
    from lite_step.models import Window, Door

    for modifier_list in [elem._cuts, elem._adds, elem._intersects, elem._fills]:
        for sub_elem in modifier_list:
            _normalize_sub_element(sub_elem)
            # Recursively normalize nested modifiers
            if hasattr(sub_elem, '_cuts'):
                _normalize_private_modifiers(sub_elem)

    # Normalize _openings (Window/Door children)
    if hasattr(elem, '_openings') and elem._openings:
        for opening in elem._openings:
            if isinstance(opening, (Window, Door)):
                # Opening POSITION (along/up/out) rides ``_anchor_spec`` and is
                # normalized by ``_normalize_placements_recursive`` with every
                # other placement — only the SIZE is per-type here.
                # Resolve an omitted size from the children's in-plane
                # extents BEFORE scaling. This runs on the normalized COPY,
                # so the author's project keeps its ``None`` and the
                # inference stays re-derivable rather than baked into their
                # source. A size that cannot be resolved is already a
                # compile error in validate_project_report.
                from lite_step.compiler import frames
                w, h = frames.infer_opening_size(opening)
                # Read the origin BEFORE the size is written: it is gated on
                # the size still being ``None`` (inferred), and the write
                # below erases that distinction for good.
                d_along, d_up = frames.opening_local_origin(opening)
                opening.width = (w or 0) / 1000.0
                opening.height = (h or 0) / 1000.0
                frames.stamp_opening_origin(
                    opening, d_along / 1000.0, d_up / 1000.0)
                # Normalize child elements (Profile frames, glass, Solid inserts)
                if hasattr(opening, '_elements') and opening._elements:
                    for child in opening._elements:
                        _normalize_sub_element(child)


def _normalize_void_operand(operand) -> None:
    """Normalize a ``.void()`` operand (mm -> m): its own coordinates, its
    boolean-operand modifiers, and any nested ``.void()`` operands."""
    _normalize_sub_element(operand)
    _normalize_private_modifiers(operand)          # operand's _cuts/_adds/_fills/_openings
    for nested in (getattr(operand, "_voids", None) or []):
        _normalize_void_operand(nested)


def _normalize_voids_recursive(elem) -> None:
    """Normalize ``.void()`` operands on ``elem`` and every descendant.

    Voids live on CONTAINERS (Slab/Column/Beam/Roof/Element) as well as on
    geometry primitives; the per-type normalization only recurses ``_elements``,
    so a container's own ``_voids`` are missed. This ONE generic walk is the
    single place ``.void()`` operands are scaled — which is why ``_voids`` is
    deliberately NOT in the ``_normalize_private_modifiers`` list (double-scaling
    would shrink an operand 1000x).
    """
    for operand in (getattr(elem, "_voids", None) or []):
        _normalize_void_operand(operand)
    for child in (getattr(elem, "_elements", None) or []):
        _normalize_voids_recursive(child)
    for child in (getattr(elem, "_openings", None) or []):
        _normalize_voids_recursive(child)


def _normalize_clips_recursive(elem) -> None:
    """Normalize ``.clip()`` half-space planes (mm -> m) on ``elem`` and every
    descendant.

    The plane ``origin`` and the ``overrun`` distance scale; ``normal`` is a
    unit-free direction and stays. ``HalfSpace`` is frozen, so each is
    REPLACED via ``model_construct``
    (validators skipped — same rationale as ``_normalize_placement_field``).
    This ONE generic walk is the single place ``_clips`` are scaled (kept out
    of ``_normalize_private_modifiers`` — its self-recursion would double-scale)
    — the same pattern as ``_normalize_voids_recursive``. Clips carry no
    placement, so the placement walk has nothing to do for them.
    """
    from lite_step.models import HalfSpace, Point

    clips = getattr(elem, "_clips", None) or []
    for i, hs in enumerate(clips):
        clips[i] = HalfSpace.model_construct(
            origin=Point.model_construct(
                x=hs.origin.x / 1000.0,
                y=hs.origin.y / 1000.0,
                z=hs.origin.z / 1000.0,
            ),
            normal=hs.normal,
            overrun=hs.overrun / 1000.0,
        )
    # A joint recorded by ``miter()`` carries an authored mm point and an mm
    # overrun, and is derived into a HalfSpace LATER (composite_clip). It is
    # scaled here, with the clips, so mm -> m stays one walk rather than two
    # that can disagree: a plane left in mm misses its own element by 1000x and
    # removes nothing, which reads as "the miter silently did not apply".
    # ``edge`` is a direction and does not scale.
    for req in (getattr(elem, "_pending_miters", None) or []):
        req.at = Point.model_construct(
            x=req.at.x / 1000.0,
            y=req.at.y / 1000.0,
            z=req.at.z / 1000.0,
        )
        req.overrun = req.overrun / 1000.0
    for attr in ("_elements", "_openings", "_cuts", "_adds",
                 "_intersects", "_fills", "_voids"):
        for child in (getattr(elem, attr, None) or []):
            _normalize_clips_recursive(child)


def _normalize_sub_element(sub_elem) -> None:
    """
    Normalize a single sub-element's coordinates from mm to meters.
    Handles box mode (start/end), contour mode (contour/thickness), and the
    v1.5 path primitives (path/section + their scalar mm fields).
    """
    from lite_step.models import Point

    # Box mode: normalize start/end
    if hasattr(sub_elem, 'start') and sub_elem.start is not None:
        sub_elem.start = Point(
            x=sub_elem.start.x / 1000.0,
            y=sub_elem.start.y / 1000.0,
            z=sub_elem.start.z / 1000.0
        )
    if hasattr(sub_elem, 'end') and sub_elem.end is not None:
        sub_elem.end = Point(
            x=sub_elem.end.x / 1000.0,
            y=sub_elem.end.y / 1000.0,
            z=sub_elem.end.z / 1000.0
        )
    # Contour mode: normalize contour points and thickness
    if hasattr(sub_elem, 'contour') and sub_elem.contour is not None:
        sub_elem.contour = [
            Point(x=pt.x / 1000.0, y=pt.y / 1000.0, z=pt.z / 1000.0)
            for pt in sub_elem.contour
        ]
    if hasattr(sub_elem, 'thickness') and sub_elem.thickness is not None:
        sub_elem.thickness = sub_elem.thickness / 1000.0
    _normalize_v15_path_fields(sub_elem)
    # Mesh operand (a Mesh cut/void/add tool): scale its vertices. Gated on the
    # Mesh type so nothing with an incidental ``vertices`` attr is touched. A
    # host Mesh has its vertices scaled by the per-type branch instead; this
    # covers a Mesh used as a boolean OPERAND — ``terrain.difference(Mesh)`` or
    # ``site.void(Mesh)`` — which otherwise stays in mm while the host is in
    # meters, so the manifold3d carve silently removes nothing.
    if type(sub_elem).__name__ == "Mesh" and getattr(sub_elem, "vertices", None):
        sub_elem.vertices = [
            Point(x=pt.x / 1000.0, y=pt.y / 1000.0, z=pt.z / 1000.0)
            for pt in sub_elem.vertices
        ]
        # Heightmap-mode operand: scale the retained authoring fields too
        # (heightmap/depth/corners) so nothing reads mm from a meters-domain
        # object later. No-op for explicit-mode meshes.
        _normalize_heightfield_fields(sub_elem)


def _normalize_v15_path_fields(elem) -> None:
    """Normalize the v1.5 path-primitive fields from mm to meters.

    Covers Pipe (path/radius/fillet_radius), Revolve (path/profile — angle is
    centidegrees and stays), Bar (path/diameter/bend_radius) and the
    path-form Sweep (path/profile/fillet_radius — profile_rotation is
    centidegrees and stays). Guarded by hasattr/type so other element
    types pass through untouched.
    """
    from lite_step.models import Point, Point2D, Pipe, Revolve, Bar, Sweep

    if not isinstance(elem, (Pipe, Revolve, Bar, Sweep)):
        return
    if getattr(elem, 'path', None) is not None:
        elem.path = [
            Point(x=pt.x / 1000.0, y=pt.y / 1000.0, z=pt.z / 1000.0)
            for pt in elem.path
        ]
    profile = getattr(elem, 'profile', None)
    if isinstance(profile, list) and profile and isinstance(profile[0], Point2D):
        elem.profile = [
            Point2D(x=pt.x / 1000.0, y=pt.y / 1000.0) for pt in profile
        ]
    for fname in ('radius', 'fillet_radius', 'diameter', 'bend_radius'):
        value = getattr(elem, fname, None)
        if value is not None:
            setattr(elem, fname, value / 1000.0)


def _normalize_box_or_extrude(elem) -> None:
    """Normalize a Box (start/end) or Extrude (contour/thickness) mm -> m.

    DSL v8: the former Solid box/contour split into two classes. Box carries
    start/end (never contour); Extrude carries contour/thickness (never
    start/end). Dispatch by class so field access never touches the missing
    attribute. Also normalizes the chainable boolean modifiers.
    """
    etype = type(elem).__name__
    if etype == "Box":
        elem.start = Point(
            x=elem.start.x / 1000.0, y=elem.start.y / 1000.0, z=elem.start.z / 1000.0
        )
        elem.end = Point(
            x=elem.end.x / 1000.0, y=elem.end.y / 1000.0, z=elem.end.z / 1000.0
        )
    elif etype == "Extrude":
        if elem.contour is not None:
            elem.contour = [
                Point(x=pt.x / 1000.0, y=pt.y / 1000.0, z=pt.z / 1000.0)
                for pt in elem.contour
            ]
        if elem.thickness is not None:
            elem.thickness = elem.thickness / 1000.0
    _normalize_private_modifiers(elem)


def _normalize_heightfield_fields(mesh) -> None:
    """Scale a heightmap-mode Mesh's RETAINED authoring fields mm -> m.

    Heightmap mode tessellates to ``vertices``/``faces`` at construction (so
    the vertex walks scale the actual geometry as usual), but the compact
    authoring fields — ``heightmap`` heights, ``depth``, ``corner_min``/
    ``corner_max`` — are retained on the model. Scale them too, so any
    consumer reading them after normalization (future surface queries) sees
    meters, never a silent mm/m mix (the bug class).
    Explicit-mode meshes have all four as None — no-op.
    """
    from lite_step.models import Point2D

    if getattr(mesh, "heightmap", None) is not None:
        mesh.heightmap = [[h / 1000.0 for h in row] for row in mesh.heightmap]
    if getattr(mesh, "depth", None) is not None:
        mesh.depth = mesh.depth / 1000.0
    if getattr(mesh, "bottom_z", None) is not None:
        mesh.bottom_z = mesh.bottom_z / 1000.0
    for corner_attr in ("corner_min", "corner_max"):
        corner = getattr(mesh, corner_attr, None)
        if corner is not None:
            # Point2D is frozen — rebuild via model_construct (validators
            # skipped; float meters are legitimate here, same rationale as
            # the Transform/Point rebuilds above).
            setattr(mesh, corner_attr, Point2D.model_construct(
                x=corner.x / 1000.0, y=corner.y / 1000.0,
            ))


def _normalize_mesh(mesh) -> None:
    """Normalize a Mesh HOST mm -> m: its own ``vertices`` AND its boolean-operand
    modifiers.

    A terrain Mesh carved by ``terrain.difference(tool)`` / ``.union()`` /
    ``.intersection()`` keeps the tool in ``_cuts`` / ``_adds`` / ``_intersects``.
    Scaling only the mesh's own vertices without calling
    ``_normalize_private_modifiers`` leaves those operands in mm while
    the host went to meters — the manifold3d boolean then silently removed
    nothing. ``_voids`` operands are scaled by the separate
    ``_normalize_voids_recursive`` walk (kept out of ``_normalize_private_modifiers``
    to avoid double-scaling), so a mesh host is covered by both walks together.
    """
    mesh.vertices = [
        Point(x=pt.x / 1000.0, y=pt.y / 1000.0, z=pt.z / 1000.0)
        for pt in mesh.vertices
    ]
    # Heightmap-mode host: scale the retained authoring fields too
    # (heightmap/depth/corners). No-op for explicit-mode meshes.
    _normalize_heightfield_fields(mesh)
    _normalize_private_modifiers(mesh)


def _normalize_placement_field(elem) -> None:
    """Normalize a ``placement=`` from mm to meters (v1.5 WS1 PR-F).

    Placement origins/offsets are geometry values and divide by 1000 with
    the rest; rotations are centidegrees and stay. Transform/Anchor are
    frozen, so the placement is REPLACED with a ``model_construct`` copy —
    validators skipped, because the normalized values are legitimately
    float meters (same rationale as the Point rebuilds; this whole pass
    runs inside the ``@suspends_strict`` extent anyway).
    """
    from lite_step.models import Anchor, Point, Transform

    placement = getattr(elem, "placement", None)
    if isinstance(placement, Transform):
        elem.placement = Transform.model_construct(
            origin=Point.model_construct(
                x=placement.origin.x / 1000.0,
                y=placement.origin.y / 1000.0,
                z=placement.origin.z / 1000.0,
            ),
            rotations=list(placement.rotations),
        )
    elif isinstance(placement, Anchor):
        elem.placement = Anchor.model_construct(
            host=placement.host,
            attach_to=placement.attach_to,
            offset_along=placement.offset_along / 1000.0,
            offset_inset=placement.offset_inset / 1000.0,
            offset_up=placement.offset_up / 1000.0,
            rotations=list(placement.rotations),
        )


def _normalize_child_anchor(elem) -> None:
    """mm -> m for a ``.anchor()`` spec, mirroring the placement rebuild.

    ``model_construct`` skips validation deliberately — this runs inside the
    ``@suspends_strict`` extent, where float offsets are the point.
    """
    from lite_step.models.elements import ChildAnchor

    spec = getattr(elem, "_anchor_spec", None)
    if spec is None:
        return
    elem._anchor_spec = ChildAnchor.model_construct(
        along=spec.along / 1000.0,
        along_center=(None if spec.along_center is None
                      else spec.along_center / 1000.0),
        inset=spec.inset / 1000.0,
        up=spec.up / 1000.0,
        rotations=list(spec.rotations),
    )


def _normalize_opening(opening) -> None:
    """mm -> m for a ``Window``/``Door``: its SIZE and its local origin.

    Its position is not here — that rides ``_anchor_spec`` and is scaled by
    :func:`_normalize_placements_recursive`. Order matters: the size may be
    INFERRED from the children's extents, and the origin from the same, so
    both are read while the children are still in millimetres. The caller
    recurses into the children afterwards.
    """
    from lite_step.compiler import frames

    cw, ch = frames.infer_opening_size(opening)
    # Origin BEFORE size — see the twin in ``_normalize_private_modifiers``.
    cd_along, cd_up = frames.opening_local_origin(opening)
    opening.width = (cw or 0) / 1000.0
    opening.height = (ch or 0) / 1000.0
    frames.stamp_opening_origin(opening, cd_along / 1000.0, cd_up / 1000.0)


def _normalize_element_tree(elem) -> None:
    """mm -> m for ONE element and every descendant, whatever their types.

    **Dispatch is on the element itself, never on "what may a Wall contain".**
    That distinction is the whole point: since ``.anchor()`` stopped gating
    child types (v22.3.0) any container can hold any type, so the old
    per-container child tables here would have silently skipped every newly
    legal pairing — a ``Revolve`` anchored into a ``Slab`` would have stayed in
    millimetres while the model went to metres. That is a 1000x error which
    RENDERS (a corbel the size of a city block) rather than raising, which is
    why this pass is generic rather than enumerated.

    ``placement=``/``.anchor()`` specs, ``.void()`` operands and ``.clip()``
    half-spaces are scaled by their own generic walks, which were already
    type-blind.
    """
    from lite_step.models import Point

    if tx.is_opening(elem):
        _normalize_opening(elem)
    elif tx.is_prism(elem):
        _normalize_box_or_extrude(elem)
    elif tx.is_mesh(elem):
        _normalize_mesh(elem)
    elif tx.is_container(elem):
        # No points of its own — the children carry them and are recursed
        # below — but CONTAINER-LEVEL boolean operands are the container's
        # own: ``kitchen.difference(tool)`` on a Space (WS-F) composes onto
        # the derived body, and without this line those operands are never
            # scaled (dead code — no emitter read container operands until the
        # Space push-down in ``space_volume``). An unscaled operand carves
        # 1000x off-position, which RENDERS rather than raising.
        _normalize_private_modifiers(elem)
    elif type(elem).__name__ == "ReferencePoint":
        elem.location = Point(
            x=elem.location.x / 1000.0,
            y=elem.location.y / 1000.0,
            z=elem.location.z / 1000.0,
        )
    elif type(elem).__name__ == "GuideLine":
        elem.points = [
            Point(x=pt.x / 1000.0, y=pt.y / 1000.0, z=pt.z / 1000.0)
            for pt in elem.points
        ]
        if elem.buffer_width is not None:
            elem.buffer_width = elem.buffer_width / 1000.0
    else:
        # Curved primitives (Sweep/Revolve/Pipe/Bar), both the legacy
        # start/end form and the v1.5 path form.
        _normalize_sub_element(elem)
        _normalize_private_modifiers(elem)

    for child in (getattr(elem, "_elements", None) or []):
        _normalize_element_tree(child)


def _normalize_placements_recursive(elem) -> None:
    """Normalize ``placement=`` on ``elem`` and every descendant.

    One generic walk (children, openings, boolean operands) — placement is
    a universal field, not per-type, so it does not belong in the per-type
    branches of ``normalize_project_to_meters``. Nested placements are
    compile ERRORS (the engine consumes top-level placements only), but
    normalizing them is harmless and keeps units consistent for the
    validation messages.
    """
    _normalize_placement_field(elem)
    _normalize_child_anchor(elem)
    for attr in ("_elements", "_openings", "_cuts", "_adds", "_fills", "_voids"):
        for child in (getattr(elem, attr, None) or []):
            _normalize_placements_recursive(child)


@suspends_strict
def normalize_project_to_meters(project: Project) -> Project:
    """
    Normalize all Project dimensions from millimeters to meters.
    
    Converts all coordinates and dimensions by dividing by 1000.
    This ensures IFC output uses SI units (meters).
    
    Note: Coordinate system transformation (DSL and IFC both use Z-up, so no axis swap is needed) happens
    in the generator, not here. This function only handles unit conversion.
    
    Args:
        project: Project object with dimensions in millimeters
        
    Returns:
        New Project object with dimensions in meters
    """
    from copy import deepcopy

    # Deep copy to avoid mutating original
    normalized = deepcopy(project)

    # DSL v2.1: stamp canonical names top-down so the emitter reads
    # ``ifc_name`` = the stamped canonical. Idempotent; deepcopy preserves
    # any prior stamp, but re-stamping keeps the normalized copy correct even
    # when this is the only entry point (the sandbox executor path).
    from lite_step.compiler.naming import stamp_canonical_names
    stamp_canonical_names(normalized)

    # Convert all storeys
    for storey in normalized.storeys:
        # Convert storey elevation (mm -> m)
        storey.elevation = storey.elevation / 1000.0
        
        # Convert all elements in storey. ONE generic recursion over the
        # whole tree (``_normalize_element_tree``) rather than a per-type
        # chain with a per-container child table inside each branch: since
        # ``.anchor()`` accepts any type into any container, an enumerated
        # table here silently leaves a newly legal pairing in millimetres.
        for elem in storey.elements:
            _normalize_element_tree(elem)

    # DSL Site containers live in ``project.sites`` (routed out of storeys),
    # so their children must be normalized mm -> m here too. Without this a
    # site-only model's terrain vertices stay in millimetres and render 1000x
    # too large. Same generic recursion as the storey walk.
    for site in getattr(normalized, "sites", []):
        _normalize_element_tree(site)

    # Placement geometry (v1.5 WS1 PR-F): Transform origins / Anchor offsets
    # normalize with everything else, via one generic walk — placement is a
    # universal field on every element. Cover storeys AND project.sites (mirror
    # the voids walk below): a placed element under a Site — notably a
    # ``site.void(op)`` / ``terrain.difference(op)`` operand with
    # ``placement=Transform(origin=…)`` — otherwise keeps its origin in mm, so the
    # mesh carve translates it ~1000x off the terrain and silently no-ops.
    for storey in normalized.storeys:
        for elem in storey.elements:
            _normalize_placements_recursive(elem)
    for site in getattr(normalized, "sites", []):
        _normalize_placements_recursive(site)

    # .void() operands (mm -> m) — one generic walk over storeys AND sites, so
    # container-level voids (missed by the per-type child normalization) are
    # covered uniformly with primitive-level voids.
    for storey in normalized.storeys:
        for elem in storey.elements:
            _normalize_voids_recursive(elem)
    for site in getattr(normalized, "sites", []):
        _normalize_voids_recursive(site)

    # .clip() half-space origins (mm -> m) — same two-route generic walk.
    for storey in normalized.storeys:
        for elem in storey.elements:
            _normalize_clips_recursive(elem)
    for site in getattr(normalized, "sites", []):
        _normalize_clips_recursive(site)

    # A CONTAINER's clips reach the parts under it. Here and
    # not at call time, so a part added after the .clip() line is still cut;
    # here and not later, because trimming a Bar's path changes geometry and
    # frames are derived from geometry.
    from lite_step.compiler.composite_clip import resolve as _resolve_container_clips
    _resolve_container_clips(normalized)

    # Frames LAST: they are derived from geometry, so they must be stamped
    # AFTER every mm -> m conversion above. A frame stamped pre-normalize
    # would deepcopy through verbatim and then be wrong by 1000x — the pass
    # is a full recompute, so re-running here is the whole fix.
    from lite_step.compiler.frames import stamp_frames
    stamp_frames(normalized)

    return normalized


class LiteStepCompileError(RuntimeError):
    """A model failed to compile to IFC — validation errors or IFC
    generation failure.

    Raised by :func:`compile_main` so a failed build is LOUD, never silent.
    The canonical model tail is ``if __name__ == "__main__": compile_main()``,
    which ignores the return value — so a bare ``return None`` on failure let
    ``python output.py`` exit 0 with no ``output.ifc`` written. The worker's
    the calling path then saw a clean exit with no artefact and hung with no
    terminal status (LOD 350 phase cube-build investigation).
    Raising instead makes the process exit non-zero with the reasons on
    stderr, so the agent (and the worker) sees a real failure it can act on.
    ``reasons`` carries the individual validation messages for callers that
    want them structured (e.g. the CLI / MCP)."""

    def __init__(self, message: str, reasons: Optional[list[str]] = None) -> None:
        super().__init__(message)
        self.reasons: list[str] = reasons or []


# ---------------------------------------------------------------------------
# `python output.py` flags
# ---------------------------------------------------------------------------


def _script_flags(source_path) -> dict:
    """Flags from ``sys.argv``, but ONLY when we really were run as
    ``python <this model>.py``.

    The gate is not paranoia. ``compile_main()`` takes no arguments from the
    canonical script tail, so reading ``sys.argv`` unconditionally would make
    every test that calls it parse PYTEST's argv — and ``-q`` is both a pytest
    flag and a plausible one of ours. Comparing ``sys.argv[0]`` with the
    caller's ``__file__`` is exact: under pytest it is pytest's path, and
    ``lite-step model.py`` skips this tail by design (it loads the module
    under a non-``__main__`` run name).
    """

    try:
        argv0 = Path(sys.argv[0]).resolve()
        if argv0 != Path(source_path).resolve():
            return {}
    except (OSError, ValueError, TypeError):
        return {}

    flags: dict = {"query": None, "check": False}
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--check":
            flags["check"] = True
        elif arg == "--q":
            if i + 1 >= len(args):
                raise LiteStepCompileError(
                    "--q needs a selector, e.g. --q wall:north. It matches "
                    "canonical names the same way Anchor(host=) does — whole "
                    "type:leaf pairs from the tail."
                )
            flags["query"] = args[i + 1]
        elif arg.startswith("--q="):
            flags["query"] = arg[4:]
    return flags


def _iter_named(project):
    """Every element in the project with its canonical name."""
    from lite_step.compiler.naming import _WALK_ATTRS

    def walk(elem):
        yield elem
        for attr in _WALK_ATTRS:
            for child in getattr(elem, attr, None) or []:
                yield from walk(child)

    for storey in project.storeys:
        for elem in storey.elements:
            yield from walk(elem)
    for site in getattr(project, "sites", []):
        yield from walk(site)


def _run_query(project, selector: str):
    """``python output.py --q <selector>`` — dump every bounding accessor.

    The selector matches with ``naming.host_matches``, the segment-aligned
    SUFFIX match ``Anchor(host=)`` already uses, so ``wall:north`` finds both
    ``wall:north`` and ``box:body:wall:north``. One matching rule in the DSL
    rather than a second one that drifts from it.

    A suffix match means the selector has to end where the canonical name
    ends. Naming the storey appends to that name, so once the wall above is
    ``wall:north:storey:ground`` the bare ``wall:north`` is a PREFIX and
    matches nothing -- pass ``wall:north:storey:ground`` instead.

    Deliberately NOT an expression evaluator. The obvious shape --
    ``--q name ".authored_aabb()"`` -- means ``eval``, which is on the AST
    guard's deny list and against the curated-read-only-surface rule this
    whole accessor family follows.
    """
    from lite_step.compiler.naming import host_matches, stamp_canonical_names
    from lite_step.models.bounds import BoundsError

    stamp_canonical_names(project)
    matches = [e for e in _iter_named(project)
               if e._canonical_name and host_matches(selector, e._canonical_name)]
    if not matches:
        names = sorted({e._canonical_name for e in _iter_named(project)
                        if e._canonical_name})
        print(f"no element matches {selector!r}.", file=sys.stderr)
        if names:
            print("known names:", file=sys.stderr)
            for n in names[:40]:
                print(f"  {n}", file=sys.stderr)
            if len(names) > 40:
                print(f"  ... and {len(names) - 40} more", file=sys.stderr)
        return None

    for elem in matches:
        print(f"{elem._canonical_name}  ({type(elem).__name__})")
        for label, call in (("authored_aabb", elem.authored_aabb),
                            ("world_aabb", getattr(elem, "world_aabb", None)),
                            ("obb", getattr(elem, "obb", None))):
            if call is None:
                continue
            try:
                b = call()
            except BoundsError as exc:
                print(f"  {label:14s} -- {exc}")
                continue
            print(f"  {label:14s} min=({b.min.x}, {b.min.y}, {b.min.z})  "
                  f"max=({b.max.x}, {b.max.y}, {b.max.z})")
            print(f"  {'':14s} size=({b.size.x}, {b.size.y}, {b.size.z})  "
                  f"rule={b.rule}"
                  + ("" if b.exact else "  exact=False"))
            if not b.is_axis_aligned:
                # Otherwise min/max above read as a mistake: on an oriented box
                # they are two DIAGONAL corners, so min.z can exceed max.z.
                print(f"  {'':14s} oriented — min/max are diagonal corners, "
                      f"not componentwise extremes; size is along/across/up")
            if not b.exact:
                from lite_step.compiler.extent import inexact_reason
                reason = inexact_reason(elem)
                if reason:
                    print(f"  {'':14s} bound, not extent: {reason}")
        print()
    return None


def compile_main(
    result: Optional[Project] = None,
    source_path: Optional[str] = None,
    output_name: str = "output.ifc",
    verbose: bool = False,
    base_ifc: Optional[str] = None,
) -> Optional[Path]:
    """Validate -> normalize -> generate IFC -> write next to source.

    Canonical use: zero-arg call from a model script.

        if __name__ == "__main__":
            compile_main()

    The helper reads ``result`` and ``__file__`` from the calling
    module's globals via stack-frame inspection. Pass them explicitly
    to override (useful in test scaffolding):

        compile_main(result=my_building, source_path="/tmp/test.py")

    The stack-frame default looks one frame up (``inspect.stack()[1]``).
    If ``compile_main`` is invoked from inside another helper, that
    helper's globals are inspected — not the script's. The contract is
    "called directly from the model script's ``__main__`` block".

    **Patch mode.** A model file MAY declare an adjacent IFC at module
    scope::

        base_ifc = "./input.ifc"   # relative to this script

        result = Project(name="patched", ...)

        if __name__ == "__main__":
            compile_main()

    When ``base_ifc`` is set (either as a module global or via the
    explicit kwarg), ``compile_main`` switches to patch mode: the
    script's Project is treated as the *desired state*; the output
    IFC carries embedded ``LITESTEP_META`` (so subsequent edits can
    take the fast hot path) and is post-marked ``cold_imported=true``
    when the base lacked META or was itself cold-imported. Generate
    mode (no ``base_ifc``) is byte-stable with the previous behaviour.

    Args:
        result: The Project to compile. Defaults to caller's ``result``
            module-global. Raises TypeError if neither is provided.
        source_path: Path to derive output dir from. Defaults to
            caller's ``__file__`` module-global.
        output_name: Filename written next to source. ``"output.ifc"``
            default matches the canonical working-dir shape
            (``input.ifc`` + ``output.py`` + ``output.ifc``).
            Relative names land next to ``source_path``; absolute paths
            are honored as-is.
        verbose: When True, prints ``Project: <name>`` and
            ``Elements: <N>`` to stdout before the standard
            ``wrote <path> (<bytes>)`` line.
        base_ifc: Optional path to an adjacent IFC to patch. Defaults
            to caller's ``base_ifc`` module-global. Relative paths
            resolve against ``source_path``'s parent. Absent / None
            means generate mode (the existing behaviour).

    Returns:
        Written .ifc Path on success.

    Raises:
        TypeError: if ``result`` cannot be resolved to a ``Project``,
            or if ``source_path`` cannot be resolved.
        LiteStepCompileError: on validation errors or IFC-generation
            failure (with stderr diagnostics). Raised — not returned as
            None — so the canonical ``compile_main()`` script tail exits
            non-zero instead of silently succeeding with no output.ifc.
    """
    if result is None or source_path is None or base_ifc is None:
        caller_globals = inspect.stack()[1].frame.f_globals
        if result is None:
            result = caller_globals.get("result")
            if result is None:
                raise TypeError(
                    "compile_main() expects 'result' at module scope. "
                    "Define 'result = generate_project()' before calling, "
                    "or pass result= explicitly."
                )
        if source_path is None:
            source_path = caller_globals.get("__file__")
            if source_path is None:
                raise TypeError(
                    "compile_main() couldn't find __file__ in caller's globals. "
                    "Pass source_path= explicitly."
                )
        if base_ifc is None:
            # Optional — absence means generate mode (no global declared).
            base_ifc = caller_globals.get("base_ifc")

    if not isinstance(result, Project):
        raise TypeError(
            f"compile_main expects a Project, got {type(result).__name__}. "
            "Did you forget `result = generate_project()`?"
        )

    flags = _script_flags(source_path)
    if flags.get("query") is not None:
        return _run_query(result, flags["query"])
    check_only = flags.get("check", False)

    if verbose:
        # Match Shape-B's debug shape exactly so future migrate-on-demand
        # of custom-telemetry files (e.g. lod100/planning) is mechanical.
        print(f"Project: {result.name}")
        print(f"Elements: {len(result.get_all_elements())}")

    report = validate_project_report(result)
    # Warnings are LOUD but non-fatal: `warning:` lines on stderr, compile
    # continues. Errors keep raising as before.
    for w in report.warnings:
        print(f"warning: {w}", file=sys.stderr)
    errors = report.errors
    if errors:
        for e in errors:
            print(f"error: {e}", file=sys.stderr)
        # LOUD failure, not a silent `return None`: the canonical
        # `compile_main()` tail ignores the return, so returning None here
        # let `python output.py` exit 0 with no IFC. Raise so the process
        # exits non-zero and the agent/worker sees a real, actionable failure.
        raise LiteStepCompileError(
            f"validation failed ({len(errors)} error"
            f"{'s' if len(errors) != 1 else ''}): {errors[0]}"
            + (f" (+{len(errors) - 1} more)" if len(errors) > 1 else ""),
            reasons=errors,
        )

    normalized = normalize_project_to_meters(result)

    from lite_step.ifc.generator import generate_ifc

    # Every compile_main call is conceptually a patch: the script's
    # Project is the desired state, applied on top of an optional
    # adjacent base IFC. When ``base_ifc`` is unset, the patch lands
    # on an empty base — equivalent to the historical "generate"
    # behaviour. The output ALWAYS carries embedded LITESTEP_META so
    # the next edit on the same file can take the fast hot path.
    try:
        source_code_for_embedding: Optional[str] = Path(source_path).read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        # Source unreadable is rare (in-memory test cases pass an
        # ephemeral source_path). Skip META embedding rather than
        # failing the compile — future edits just lose hot path.
        print(
            f"warning: couldn't read source for LITESTEP_META embedding ({exc})",
            file=sys.stderr,
        )
        source_code_for_embedding = None

    # Resolve and validate base_ifc when declared. Relative paths
    # resolve against the script's parent dir, matching the example
    # convention ``base_ifc = "./input.ifc"``.
    base_path: Optional[Path] = None
    if base_ifc is not None:
        candidate = Path(base_ifc)
        if not candidate.is_absolute():
            candidate = Path(source_path).resolve().parent / candidate
        if not candidate.is_file():
            print(
                f"warning: base_ifc={base_ifc} (resolved {candidate}) not found; "
                "patching against empty base instead",
                file=sys.stderr,
            )
        else:
            base_path = candidate

    # Forward-only version gate. A stored base may be continued/edited only
    # by the compiler that wrote it; every other LITESTEP_META version is
    # refused. The rule (and why it is an allowlist rather than a list of
    # known-bad versions) lives in ``embedder.base_version_refusal`` — one
    # decision, read here and by ``patch_executor.patch_main``.
    if base_path is not None:
        from lite_step.ifc.embedder import (
            extract_litestep_meta as _extract_meta,
            base_version_refusal,
            LITESTEP_META_VERSION,
        )
        _guard_meta = _extract_meta(base_path.read_text(encoding="utf-8"))
        _refusal = base_version_refusal(
            _guard_meta, base_name=f"The base IFC ({base_path.name})",
        )
        if _refusal is not None:
            raise LiteStepCompileError(
                _refusal,
                reasons=["base IFC carries a foreign LITESTEP_META version "
                         f"({_guard_meta.version}, expected "
                         f"{LITESTEP_META_VERSION})"],
            )

    ifc_result = generate_ifc(normalized, source_code=source_code_for_embedding)
    if not ifc_result.success:
        print(f"IFC generation failed: {ifc_result.error}", file=sys.stderr)
        # LOUD failure — see the validation branch above.
        raise LiteStepCompileError(
            f"IFC generation failed: {ifc_result.error}",
            reasons=[str(ifc_result.error)],
        )

    cold_imported = False
    if base_path is not None:
        from lite_step.ifc.embedder import (
            extract_litestep_meta,
            set_cold_imported_flag,
        )
        base_meta = extract_litestep_meta(base_path.read_text(encoding="utf-8"))
        # Sticky: base had no META (third-party export) OR base was
        # already cold-imported → output stays cold_imported.
        cold_imported = (base_meta is None) or base_meta.cold_imported
        if cold_imported and source_code_for_embedding is not None:
            ifc_result.ifc_content = set_cold_imported_flag(ifc_result.ifc_content)
        if verbose:
            mode_label = "cold_import" if cold_imported else "round_trip"
            print(f"Patch: base={base_path} mode={mode_label}")

    if check_only:
        # --check: everything above ran (validation, normalize, generate, the
        # csg-depth line, every warning), and nothing is written. The point is
        # a fast "would this compile, and what would it complain about" that
        # cannot clobber an output.ifc a viewer is holding.
        print(f"checked {Path(source_path).name} — "
              f"{len(ifc_result.ifc_content)} bytes of IFC, not written")
        return None

    out_arg = Path(output_name)
    if out_arg.is_absolute():
        out = out_arg
    else:
        out = Path(source_path).resolve().parent / out_arg
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ifc_result.ifc_content)

    # 2D drawing stylesheet, beside the IFC at the path the file's
    # BBIM_Documentation pset points at. Best-effort and LOUD: a missing
    # stylesheet costs hatching in a downstream drawing, never the model.
    try:
        from lite_step.draft import write_stylesheet
        from lite_step.ifc.drawings import DEFAULT_STYLESHEET_PATH
        write_stylesheet(out.parent / DEFAULT_STYLESHEET_PATH)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"warning: drawing stylesheet not written ({exc}) — 2D drawings "
              "fall back to the viewer's default styles", file=sys.stderr)

    suffix = f" [patch base={base_path.name}]" if base_path else ""
    print(f"wrote {out} ({len(ifc_result.ifc_content)} bytes){suffix}")
    return out


@dataclass
class ValidationReport:
    """Split validation outcome (DSL v1.5).

    ``errors`` are fatal — ``compile_main`` raises :class:`LiteStepCompileError`
    exactly as before. ``warnings`` are loud but non-fatal — printed as
    ``warning:`` lines on stderr, compile continues. The split exists so that
    open-vocabulary conditions (unknown material key, off-stock thickness)
    don't kill a build while structural violations still do.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_project(project: Project) -> list[str]:
    """Backward-compatible view of :func:`validate_project_report`.

    Returns only the FATAL errors — the historical contract every existing
    caller (agent tools, MCP validate, tests) relies on. Non-fatal warnings
    live on :class:`ValidationReport`; use ``validate_project_report`` when
    you want both channels.
    """
    return validate_project_report(project).errors


def _iter_identity_elements(elem):
    """Yield ``elem`` and every product-capable descendant.

    Walks container/opening children (``_elements``) and ``.opening()``
    attachments (``_openings``) — everything that can emit its own IFC
    product with its own Name. Boolean modifier operands (``_cuts`` /
    ``_adds`` / ``_fills``) are consumed into the parent's geometry and
    never become products, so they are not identity-bearing (the spec's
    "leave internal geometry anonymous").
    """
    yield elem
    for child in (getattr(elem, "_elements", None) or []):
        yield from _iter_identity_elements(child)
    for child in (getattr(elem, "_openings", None) or []):
        yield from _iter_identity_elements(child)


def _iter_identity_sites(top, scope: str):
    """Yield ``(canonical, site_label)`` for every identity-bearing thing under ``top``.

    Everything :func:`_iter_identity_elements` yields, PLUS each named
    ``.void()`` hole. A named void is not an element — its tool operand is
    consumed — but it emits an ``IfcOpeningElement`` whose ``Name`` and
    uuid5 GlobalId both derive from ``_void_canonical``, so two holes sharing
    one are a silent GlobalId collision and an ambiguous patch-mode target.
    ``_iter_identity_elements`` deliberately stays ``_voids``-free (other
    callers must not visit operands); the void scan lives here instead.
    Anonymous elements and voids (canonical ``None``) are skipped.
    """
    for elem in _iter_identity_elements(top):
        if elem.ifc_name:
            yield elem.ifc_name, f"{scope}/{type(elem).__name__}(id={elem.id})"
        for operand in (getattr(elem, "_voids", None) or []):
            canonical = getattr(operand, "_void_canonical", None)
            if canonical:
                yield canonical, (
                    f"{scope}/{type(elem).__name__}(id={elem.id})"
                    f".void(name={getattr(operand, '_void_leaf', None)!r})"
                )


def _iter_storey_identity_elements(storey):
    """Every product-capable element in ``storey`` — tops AND their descendants.

    The per-element checks below (Wall body, Box volume, Extrude/Sweep
    geometry sources) were always written per ELEMENT; only the WALK was per
    storey, so anything nested inside a container was never checked. That was
    invisible while ``.add()`` refused product children and ``.anchor()``'s
    only nested products were details — a nested ``Wall`` with no body reached
    the generator, which picks "the first Box/Extrude child", found none, and
    emitted a wall-shaped hole with no error anywhere.

    Yields the top element FIRST (``_iter_identity_elements`` does), so every
    element is visited, in that order.
    """
    for top in storey.elements:
        yield from _iter_identity_elements(top)


def _opening_placement_errors(project) -> "list[str]":
    """Every ``Window``/``Door`` in ``project`` must be placed by its host.

    DSL v20.0.0 removed ``Window(offset=, sill=)`` / ``Door(offset=, sill=)``:
    an opening has a SIZE of its own and nothing else, and the host states
    where it goes — ``wall.anchor(win, along=, up=)`` for a Wall child,
    ``body.opening(win, along=, up=)`` for a standalone solid. Both stamp the
    same ``ChildAnchor``.

    Two failures are reported:

    * **no spec** — the child arrived through ``.add()``, which is pure
      world-coordinate containment and a Window/Door has no world coordinates
      to keep. Without this check it would compile at the host's start corner
      at floor level, which is a plausible-looking wrong answer.
    * **rotations** — ``.anchor(rotations=)`` is not representable on an
      opening in v1: the void is an axis-aligned box built from the host's
      run axis, and turning the fill without turning the void would leave the
      leaf sticking through the reveal.
    """
    from lite_step.models import taxonomy as tx

    out: list[str] = []
    tops = [e for s in project.storeys for e in s.elements]
    tops += list(getattr(project, "sites", None) or [])
    for top in tops:
        for elem in _iter_identity_elements(top):
            # Read off the child (``brings_void``), not an isinstance list —
            # the WS-A §1.1 rule: what carves is a property of the CHILD.
            if not tx.brings_void(elem):
                continue
            label = f"{type(elem).__name__} '{elem.ifc_name or elem.id}'"

            # Size: explicit wins, else inferred from the children's in-plane
            # extents. Neither available means there is nothing to derive the
            # hole from — say so here rather than emit a zero-sized void.
            from lite_step.compiler.frames import infer_opening_size
            w, h = infer_opening_size(elem)
            if w is None or h is None:
                out.append(
                    f"{label}: needs a size — give width=/height=, or add "
                    f"the joinery as children and the compiler infers the "
                    f"opening from their in-plane extents. An opening with "
                    f"neither has no hole to cut."
                )
            elif w <= 0 or h <= 0:
                out.append(
                    f"{label}: opening size must be positive (got "
                    f"{w}x{h} mm)."
                )

            spec = getattr(elem, "_anchor_spec", None)
            if spec is None:
                out.append(
                    f"{label}: needs a position — use "
                    f"wall.anchor({type(elem).__name__.lower()}, along=…, "
                    f"up=…) on a Wall, or body.opening("
                    f"{type(elem).__name__.lower()}, along=…, up=…) on a "
                    f"standalone solid. An opening carries no coordinates of "
                    f"its own (DSL v20.0.0 removed offset=/sill=), so .add() "
                    f"— which keeps the child's world coordinates — cannot "
                    f"place one."
                )
            elif spec.rotations:
                out.append(
                    f"{label}: .anchor(rotations=) is not supported on an "
                    f"opening — the void is built axis-aligned from the "
                    f"host's run axis, so rotating the fill alone would push "
                    f"the leaf through the reveal. Rotate the HOST, or author "
                    f"the angle in the opening's own child geometry."
                )
    return out


def _emits_openings(elem, parent) -> bool:
    """Will an emitter read THIS element's ``Window``/``Door`` children?

    ``tx.hosts_openings`` answers for the TYPE; this adds the one positional
    exception, which is a ``Box``/``Extrude`` that is a ``Wall``'s body. A
    Wall's body is not a product of its own — ``_create_wall`` turns it into
    the ``IfcWall``'s representation and ``_create_box`` never runs on it — so
    the ``_openings`` list that ``_process_openings_on_element`` would have
    read is never reached. An ANCHORED prism inside a Wall is a detail with a
    product of its own and does emit; ``tx.body_prisms`` is the shared
    definition that tells the two apart, and it is the same call the Wall body
    check and the emitter makes.

    The second positional exception is a distributing CONTAINER with an
    EMPTY frame basis. ``Column``/``Beam``/``Element``, and since a
    BODY-LESS ``Wall``, emit through
    ``generator._process_container_openings``, which derives the opening frame
    from ``frames.frame_basis_aabb`` — the container's ``.add()``ed children,
    which are world coordinates by definition. With nothing to derive
    from there is no run axis for ``along=`` to walk and no leaf for the hole
    to land on, so the container emits nothing; ``has_frame_basis`` is the same
    call ``_bodyless_container_errors`` asks, so the two refusals cannot
    disagree about what "has geometry" means.

    Note the two Wall exceptions do not overlap and could not: the first is a
    prism child of a wall THAT HAS a body, the second is the wall itself when
    it has NONE. ``tx.body_prisms`` is the one call both read.
    """
    if not tx.hosts_openings(elem):
        return False
    if tx.is_prism(elem) and type(parent).__name__ == "Wall":
        return not any(b is elem for b in tx.body_prisms(parent))
    from lite_step.ifc.voids import opening_reaches_children

    if opening_reaches_children(elem):
        from lite_step.compiler import frames
        return frames.has_frame_basis(elem)
    return True


def _opening_host_errors(project) -> "list[str]":
    """An opening only compiles on a host that will actually EMIT it.

    ``agents/dsl-reference.md`` named seven opening hosts and the emitter
    implements three. Measured with one
    6 m body ``Box(x 0..6000, y -300..0, z 0..2700)`` and one
    ``host.opening(Window(width=1200, height=1400), along=4200, up=900)``:

    ======================================  ===================  ===========
    host                                    IfcOpeningElement    IfcWindow
    ======================================  ===================  ===========
    ``Wall``                                1                    1
    standalone ``Box`` / ``Extrude``        1                    1
    ``Column`` ``Beam`` ``Slab`` ``Roof``   0                    0
    ``Element(ifc_class="IfcWall")``        0                    0
    ``Sweep`` ``Pipe`` ``Bar`` ``Revolve``  0                    0
    ``Mesh``                                0                    0
    a ``Wall``'s BODY child                 0                    0
    ======================================  ===================  ===========

    **Three rows of that table now emit.** ``Column``, ``Beam`` and
    ``Element`` distribute the hole to the geometry-bearing leaves it overlaps
    — N ``IfcOpeningElement``, one ``IfcWindow`` — through
    ``generator._process_container_openings``. This function was NARROWED, not
    deleted: ``Slab`` and ``Roof`` still refuse (``along``/``inset``/``up``
    assume an outer face and a run axis a HORIZONTAL host does not have, which
    is a frame question leaves open), so do ``Sweep``/``Pipe``/``Bar``/
    ``Revolve``/``Mesh``, and so does the Wall-BODY position. The rule is
    stated once in ``taxonomy.OPENING_HOSTS`` / ``tx.hosts_openings``; this
    reads it.

    The five containers at least logged ``"child Window … has no IFC emitter
    and was skipped"``; the leaf solids and the Wall body emitted **no warning
    on any stream** — the Wall-body model was byte-identical to the same model
    with the ``.opening()`` line deleted (6049 bytes either way).

    **And the details were notched anyway.** ``displacement.
    _carve_openings_through_details`` runs off the DSL tree, not off what the
    emitter did, so on a ``Column`` a ``.add()``ed bar standing in the phantom
    hole lost 1.4 m of its section to a window that is not in the file
    (``IFCOPENINGELEMENT: 0  IFCWINDOW: 0  IFCBOOLEANRESULT: 2``). Material
    missing with nothing in the model to explain it is worse than the silent
    no-op, and a compile refusal stops it by construction — the tree the carve
    pass would have walked never gets built.

    **Why the two HORIZONTAL containers still refuse.** ``along=``/``inset=``/
    ``up=`` measure from an outer FACE along a run axis. A ``Column``, a
    ``Beam`` and a generic ``Element`` have both — their frame basis is
    wall-shaped — which is why could give them the vertical hosts' scalars
    unchanged. A ``Slab`` or ``Roof`` has neither in that sense: its "outer
    face" is the soffit or the topside depending on which way you are looking,
    and its "run axis" is a plan direction with no canonical choice between the
    two in-plane axes. Emitting one anyway would put the hole somewhere an
    author cannot predict, which is worse than the refusal. A floor hatch is
    ordinary BIM and it is a real capability gap; stays open for it, and
    the answer is a SPELLING (a plan-XY frame, or a separate verb), not a
    widened predicate. Both refusals are corpus-neutral: **zero** of the 20
    committed models put an opening on anything but a ``Wall``, so nothing had
    to migrate either way.

    **Why the Wall BODY is refused rather than hoisted onto the Wall.**
    ``displacement._voids_of`` DOES hoist a body's ``_voids`` onto its Wall, so
    ``body.void(tool)`` works where ``body.opening(win, …)`` does not, and the
    asymmetry is real. Hoisting the openings too was measured and rejected: a
    ``.void()`` operand is absolute world geometry, so hoisting it moves
    nothing and emits nothing new, whereas an opening carries HOST-FRAME
    scalars and a FILL product — the hoist would have to reparent an
    ``IfcWindow`` onto a host the author did not name, and change the fill's
    canonical name depending on which of two spellings was used for one hole.
    That is the ``wall.add(win)`` case again, which v20.0.0 made a compile
    error for exactly this reason: it "buys one redundant way to say what
    ``.anchor()`` says exactly". ``wall.anchor(win, along=…, up=…)`` takes the
    identical scalars — the Wall's opening frame IS derived from that same
    body — so the refusal's remedy is a mechanical, lossless rewrite.
    """
    out: list[str] = []
    tops = [e for s in project.storeys for e in s.elements]
    tops += list(getattr(project, "sites", None) or [])

    def walk(elem, parent):
        openings = [c for c in (getattr(elem, "_elements", None) or [])
                    if tx.brings_void(c)]
        openings += [c for c in (getattr(elem, "_openings", None) or [])
                     if tx.brings_void(c)]
        if openings and not _emits_openings(elem, parent):
            out.append(_opening_host_message(elem, parent, openings))
        for attr in ("_elements", "_openings"):
            for child in (getattr(elem, attr, None) or []):
                walk(child, elem)

    for top in tops:
        walk(top, None)
    return out


def _opening_host_message(elem, parent, openings) -> str:
    """The refusal, with the remedy that WORKS for this particular host.

    Four different remedies, because the failures are four different shapes: a
    HORIZONTAL container has no run axis for the scalars to mean anything
    against, an EMPTY distributing container has no leaves for the hole to land
    on, a Wall's body has a Wall above it that already takes the same scalars,
    and a leaf solid has ``.void()``.
    """
    from lite_step.ifc.voids import opening_reaches_children
    kind = type(elem).__name__
    article = "an" if kind[0] in "AEIOU" else "a"
    label = f"{kind} '{elem.ifc_name or elem.id}'"
    names = ", ".join(
        f"{type(o).__name__} {getattr(o, 'name', None)!r}" for o in openings
    )
    head = (f"{label}: hosts {len(openings)} opening(s) ({names}) that no IFC "
            f"emitter reads. ")

    if tx.is_prism(elem) and type(parent).__name__ == "Wall":
        return head + (
            f"A Wall's BODY is not a product of its own — _create_wall turns "
            f"it into the IfcWall's representation, so its .opening() list is "
            f"never read. Put the opening on the WALL: "
            f"wall.anchor(window, along=…, up=…) takes the identical scalars, "
            f"because the wall's opening frame is derived from this very "
            f"body. (.void() on a body DOES reach the wall, because a void "
            f"operand is absolute world geometry and brings no fill product; "
            f"an opening is neither.) Otherwise the window compiles "
                f"to no IfcOpeningElement, no IfcWindow and no warning on any "
            f"stream — byte-identical to omitting the line."
        )

    if opening_reaches_children(elem):
        # A Column/Beam/Element that WOULD distribute and has nothing to
        # distribute to. The frame comes from the .add()ed children, so an
        # empty basis is a missing run axis AND a missing landing place.
        return head + (
            f"{article.capitalize()} {kind} cuts an opening by handing it to "
            f"the geometry-bearing children it overlaps, and this one has no "
            f".add()ed children to derive a frame from or to put the hole in "
            f"— along=/up= are measured against the bounds of what the "
            f"container aggregates. .add() the Box/Extrude leaves this "
            f"assembly is made of (world coordinates), then keep the opening "
            f"where it is; or put it on the leaf directly with "
            f"leaf.opening(window, along=…, up=…). An .anchor()ed child cannot "
            f"supply the frame — it is positioned BY it."
        )

    if tx.is_container(elem):
        # Slab / Roof — HORIZONTAL hosts, and the one half left open.
        return head + (
            f"A Wall, a Box/Extrude body, and a Column/Beam/Element assembly "
            f"emit their opening children; {article} {kind} is HORIZONTAL and "
            f"does not, because along=/inset=/up= measure from an outer FACE "
            f"along a run axis and a {kind} has neither — its outer face is "
            f"the soffit or the topside depending which way you look, and its "
            f"run axis is a plan direction with no canonical choice between "
            f"the two in-plane axes. Emitting one anyway would put the hole "
            f"somewhere you cannot predict. Put the opening on the SOLID it "
            f"cuts — body.opening(window, along=…, up=…) on the Box/Extrude "
            f"you .add()ed here, which emits the void and the fill and "
            f"measures along= in that body's frame — or make the host a Wall, "
            f"or cut it with .void(tool, name='…') in absolute world "
            f"coordinates (geometry but no fill, and no schedule row). A "
            f"full-fidelity {kind.lower()} penetration is not implemented."
        )

    return head + (
        f"An opening is emitted off a Wall or off a Box/Extrude body "
        f"(generator._process_openings_on_element derives the void frame from "
        f"a box AABB or a vertical planar contour, and {article} "
        f"{kind} is neither). Cut the hole with "
        f".void(tool, name='…') — absolute world coordinates, any solid as the "
        f"tool — or move the opening to a Box/Extrude body. Otherwise the "
            f"window compiles to no IfcOpeningElement, no IfcWindow "
        f"and no warning."
    )


#: |sin(tilt)| of the contour-plane normal against horizontal beyond which a
#: wall contour is rejected as tilted (~0.006° — int-mm authored vertical
#: polygons compute an exactly-horizontal normal up to float noise).
_WALL_CONTOUR_VERTICAL_EPS = 1e-4


def _has_thickness_source(body) -> bool:
    """True when a contour body resolves an extrusion depth — ``thickness=``
    or a sheet ``Material(key=..., thickness_mm=...)`` (v1.5 one-source rule).
    The extrusion depth IS the body's wall thickness; without one the
    generator has no body to build.
    """
    if body.thickness is not None:
        return True
    from lite_step.materials import element_registry_material
    mat = element_registry_material(body)
    return mat is not None and getattr(mat, "thickness_mm", None) is not None


def _contour_frame_errors(contour_pts, has_thickness_source, id_label,
                          escape_hint) -> list[str]:
    """Geometric-plane checks shared by contour-mode Wall bodies (gable
    walls) and contour-mode Solid bodies that carry openings.

    The generator's opening frame (run axis, thickness, wall start) only
    holds for a planar VERTICAL polygon extruded horizontally by its
    thickness. Anything else must fail HERE, loudly, instead of the
    generator dropping or mangling the voids (#391 made the wall-body drop
    loud; extends the same discipline to standalone Solid openings).

    Args:
        contour_pts: the body's ``.contour`` Points (raw, pre-normalization).
        has_thickness_source: pass ``_has_thickness_source(body)`` — a
            missing thickness source appends the thickness error.
        id_label: element identity for the message (e.g. ``"Wall 'x'"``).
        escape_hint: the constraint's trailing clause (mode-specific
            actionable guidance).

    Pure Python on the raw contour points (validation runs pre-
    normalization, int mm — tolerances are scale-relative so a meters
    Project validates identically). No Point construction, so strict
    int-mm is not disturbed.
    """
    import math as _math

    constraint = (
        f"{id_label}: contour body must be a planar vertical polygon "
        f"{escape_hint}"
    )

    errors: list[str] = []
    pts = [(float(p.x), float(p.y), float(p.z)) for p in contour_pts]
    n = len(pts)

    # Contour extent, for scale-relative tolerances.
    diag = _math.sqrt(
        (max(p[0] for p in pts) - min(p[0] for p in pts)) ** 2
        + (max(p[1] for p in pts) - min(p[1] for p in pts)) ** 2
        + (max(p[2] for p in pts) - min(p[2] for p in pts)) ** 2
    )

    # Newell normal (matches lite_step.ifc.geometry.newell_normal, inlined
    # so validation stays numpy-free).
    nx = ny = nz = 0.0
    for i in range(n):
        cx, cy, cz = pts[i]
        qx, qy, qz = pts[(i + 1) % n]
        nx += (cy - qy) * (cz + qz)
        ny += (cz - qz) * (cx + qx)
        nz += (cx - qx) * (cy + qy)
    length = _math.sqrt(nx * nx + ny * ny + nz * nz)

    # Newell length is twice the enclosed area — near-zero relative to the
    # extent means collinear/degenerate points with no usable plane.
    if diag <= 0.0 or length < 1e-9 * diag * diag:
        errors.append(
            constraint + " (degenerate contour: points are collinear — "
            "no plane normal)"
        )
        return errors
    ux, uy, uz = nx / length, ny / length, nz / length

    # Vertical polygon ⇔ horizontal plane normal.
    if abs(uz) > _WALL_CONTOUR_VERTICAL_EPS:
        tilt_deg = _math.degrees(_math.asin(min(1.0, abs(uz))))
        errors.append(
            constraint
            + f" (contour plane tilts {tilt_deg:.1f}° out of vertical)"
        )

    # Planarity: max point deviation from the Newell plane.
    p0 = pts[0]
    deviation = max(
        abs((p[0] - p0[0]) * ux + (p[1] - p0[1]) * uy + (p[2] - p0[2]) * uz)
        for p in pts
    )
    if deviation > max(1e-6, 1e-6 * diag):
        errors.append(
            constraint
            + f" (points deviate up to {deviation:.3g} mm from the "
            f"contour plane)"
        )

    # Thickness source — the extrusion depth IS the wall thickness; without
    # one the generator has no body to build (loud here, never a drop).
    if not has_thickness_source:
        errors.append(
            f"{id_label}: contour body needs a thickness source — "
            f"thickness= or material=Material(key=..., thickness_mm=...)"
        )

    return errors


def _wall_contour_body_errors(wall, body) -> list[str]:
    """Compile-time checks for a contour-mode Wall body Solid (gable walls).

    Thin wrapper over the shared ``_contour_frame_errors`` with the Wall
    identity label and the free-form-wall escape hatch.
    """
    return _contour_frame_errors(
        body.contour,
        _has_thickness_source(body),
        f"Wall '{wall.id}'",
        "(gable walls); tilted/non-planar contours are not supported — use "
        "Element(ifc_class='IfcWall') with a Solid child for free-form walls",
    )


def _bodyless_container_errors(elem) -> list[str]:
    """What a container with NO body of its own still has to satisfy.

    The rule is a derivable FRAME, not a solid — ``frames.has_frame_basis`` is
    the one place that is decided, so this does not restate what "has
    geometry" means. Two refusals remain, and each replaces a silent failure
    rather than a working model.

    **An empty basis.** The refusal names what is MISSING rather than
    repeating "must contain a Box/Extrude body", which is not the rule:
    an aggregator whose leaves are ``.add()``ed derives its thickness, run
    axis and extents from them and is legal without a body. The two ways to
    reach an empty basis read very differently to an author, so they are
    reported differently — an empty container is a modelling slip, whereas a
    container whose children are all ``.anchor()``ed looks full and is not,
    and the reason is the one rule that makes ``.anchor()`` mean anything (a
    child placed BY the frame cannot define it).

    **An OPENING on a body-less container is not refused HERE.** It needs two
    pieces: an opening frame that a container can derive, and a product with a
    ``Representation`` to carry the ``IfcRelVoidsElement``. Both are supplied for
    ``Column``/``Beam``/``Element`` — the frame from ``frame_basis_aabb``
    (``frames.opening_container_aabb``) and the void from
    ``generator._process_container_openings``, which hands the hole to the
    geometry-bearing LEAVES it overlaps instead of to the aggregate parent —
    and routes a body-less ``Wall`` into the same two functions.
    ``ifc.voids.void_reaches_children`` is what decides that, per instance,
    off ``tx.body_prisms``.

    So the refusal MOVED rather than vanished, and moving it is the point: an
    opening now compiles exactly when an emitter reads it, which is
    ``_opening_host_errors`` / ``_emits_openings``' question and not this
    function's. A body-less container with an EMPTY frame basis still refuses,
    from the branch above (no frame to derive) AND from
    ``_emits_openings``, which returns ``has_frame_basis`` for a distributing
    host — the same call, so the two cannot disagree about what "has geometry"
    means.

    **What the envelope workaround cost, which is why this had to change.** measured the solid an author had to add purely so the hole had
    somewhere to land, on a 6 m wood-framed wall: **0.920 m3** of phantom
    ``IfcWall`` volume when the stud bay is partly tiled (2.128 m3 with the
    batts omitted), and, once the bay is tiled the way a real wall is,
    ``ifcopenshell.geom create_shape verts = 0`` behind 29 ``IfcBooleanResult``
    — an ``IfcWall`` whose representation evaluates to nothing while holding
    the ``IfcRelVoidsElement`` for the window. The better the model, the more
    degenerate the artefact.

    **Do not "fix" the leaves by widening ``displacement.
    _iter_opening_carvees``'s body exemption.** Two issues proposed it () and the measurement refuted both: the limiter was the frame and not
    the reach, and dropping the exemption emitted one ``IfcBooleanResult``
    notching the first leaf for a hole that was still not in the file
    (``IFCOPENINGELEMENT=0``) — worse than the silent no-op. The
    distribution is the fix, and ``displacement`` now SKIPS such a host
    entirely (``opening_reaches_children``) because its leaves each receive a
    real ``IfcRelVoidsElement``.
    """
    from lite_step.compiler import frames

    label = f"{type(elem).__name__} '{elem.ifc_name or elem.id}'"
    if not frames.has_frame_basis(elem):
        anchored = [c for c in (getattr(elem, "_elements", None) or [])
                    if frames.is_anchored_child(c)]
        if anchored:
            return [
                f"{label}: has no body and no frame to place its "
                f"{len(anchored)} .anchor()ed child(ren) against — an "
                f"anchored child is positioned BY the frame, so it cannot be "
                f"what derives it. Give this container a Box/Extrude body, "
                f"or .add() at least one child in world coordinates for the "
                f"frame to be derived from."
            ]
        return [
            f"{label}: has no body and no children to derive a frame from — "
            f"a container may omit its own Box/Extrude body when it "
            f"aggregates .add()ed children in world coordinates, but this "
            f"one is empty. Add a Box/Extrude body, or .add() the children "
            f"it aggregates."
        ]

    # An opening is NOT checked here. With a frame basis this container
    # distributes its holes to the leaves they overlap, exactly as a Column
    # does; without one, the branch above has already refused and
    # ``_emits_openings`` refuses again by name. Restating the rule here would
    # be a second opinion about which hosts emit, which is the thing
    # ``taxonomy.OPENING_HOSTS`` exists to prevent.
    return []


def validate_project_report(project: Project) -> ValidationReport:
    """
    Validate a Project for IFC compatibility.

    Error checks (fatal — compile raises):
    - Canonical-name uniqueness across the Project, children included (DSL
      v2.1 identity — per-scope: no two same-type siblings sharing a leaf in
      one container). Replaces the pre-v2.1 id-uniqueness and
      name-vs-anonymous-id checks (id is internal-only; anonymous emits
      Name=None).
    - Wall body: exactly one Solid child (zero = no body; more than one =
      the generator would silently drop all but the first)
    - Wall contour body (gable walls): planar VERTICAL polygon with a
      thickness source — tilted/non-planar contours error here instead of
      the generator dropping the wall
    - All elements have valid dimensions
    - Profile shapes exist in catalog
    - Registry-material one-source rule (Material dimension vs the element's
      own section/thickness source) and wrong-primitive dimension pairings
    - props= shape (props={"Pset_X": {"Key": scalar}} — one dict per Pset,
      bool/int/float/str values)
    - Anchor host resolution: host= is a segment-aligned suffix of a canonical
      name (miss and ambiguity are both errors)
    - Placement engine (v1.5 WS1 PR-F): placement= on nested elements /
      Site containers / boolean operands, anchor cycles, hosts without
      point-bearing geometry, malformed rotation tuples / unknown rotation
      axes, off-enum attach_to

    Warning checks (loud, non-fatal):
    - Unknown (non-registry, non-legacy) material keys — emitted as plain
      named materials, open vocabulary
    - props= on boolean-operand geometry (never emitted — no IFC product)

    Args:
        project: The Project to validate

    Returns:
        ValidationReport with separate errors and warnings lists
    """
    # DSL v2.1: canonical names drive per-scope uniqueness + Anchor host
    # resolution below. Stamp before any name-based check reads ``ifc_name``.
    from lite_step.compiler.naming import stamp_canonical_names
    # count_visits: this is the ONE pipeline entry point that reads the visit
    # counts (the shared-element check at the end of this function), so it is
    # the only one that pays for them.
    stamp_canonical_names(project, count_visits=True)

    errors: list[str] = []
    warnings_out: list[str] = []

    # Containment cycles FIRST, ahead of every other check including the
    # frame stamp. Not a preference — a cycle is the one defect that
    # invalidates the inputs of everything below it: the canonical name of an
    # element inside a cycle is whatever truncated string the walk got to
    # before it turned back, so the uniqueness scan, Anchor host resolution
    # and grouping member resolution would all be reading a name no author
    # wrote and reporting misses and collisions that describe nothing.
    #
    # Reported ALONE for the same reason. Every follow-on error a cyclic tree
    # produces is an artefact of the cycle, and burying the one actionable
    # message under twenty of them is how an author ends up fixing the
    # symptom. Fix the cycle, re-run, see the real errors.
    from lite_step.compiler.naming import find_containment_cycles
    errors.extend(find_containment_cycles(project))
    if errors:
        return ValidationReport(errors=errors, warnings=warnings_out)

    # Authoritative frames (mm at this point — validation runs pre-normalize).
    from lite_step.compiler.frames import stamp_frames
    stamp_frames(project)

    # .void() maps to IfcRelVoidsElement, which targets an IfcElement. A SPATIAL
    # element (IfcSpace/IfcSite) can't host one; a .void() on one is realised by
    # carving its TERRAIN meshes (the displacement pass folds it into the mesh
    # carve — semantic targeting via taxonomy.iter_terrain_meshes: meshes
    # inside the terrain wrapper
    # ``Element(IfcGeographicElement, TERRAIN)``). A site void clears the GROUND;
    # sibling groundwork (a gravel IfcEarthworksFill bed) is never a target. So a
    # relational void is valid IFF the container has a terrain mesh; otherwise
    # there is nothing to carve → reject. (A Space can hold no terrain, so its
    # void is always rejected — the "no not space" rule falls out of the
    # kind-based check, not a Space-specific branch.)
    def _check_relational_voids(elem):
        if getattr(elem, "_voids", None) and tx.is_spatial(elem):
            if not any(True for _ in tx.iter_terrain_meshes(elem)):
                errors.append(
                    f"{type(elem).__name__} '{elem.id}': .void() is not valid here — a "
                    f"spatial element (IfcSpace/IfcSite) can't host an IfcOpeningElement, "
                    f"and a .void() on one is realised by carving its TERRAIN meshes, but "
                    f"this {type(elem).__name__} has none. Void a project element or a "
                    f"geometry primitive instead (or, for terrain, add a Mesh inside an "
                    f'Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN")).'
                )
        # A ``.void()`` on a bare MESH receiver would be a silent no-op
        #: the operand was consumed by ``collect_consumed_operand_ids``
        # so it never rendered, the displacement pass folded only a mesh's
        # ``_cuts`` (never its ``_voids``), and neither generator read them —
        # the author asked for a counted hole and got nothing, no error, no
        # warning. It is REFUSED rather than implemented, symmetric with the
        # Space receiver above: ``.void()`` promises an IfcOpeningElement
        # linked by IfcRelVoidsElement, and a tessellated mesh
        # (IfcTriangulatedFaceSet) is not a valid IFC boolean operand, so
        # nothing in the file could carry that opening. Baking the subtraction
        # into the vertices instead is exactly what ``.difference()`` already
        # does for a Mesh — implementing ``.void()`` that way would ship a
        # second spelling of one verb whose only distinguishing promise (that
        # the hole is COUNTED) the file could not keep.
        elif getattr(elem, "_voids", None) and tx.is_mesh(elem):
            errors.append(
                f"{type(elem).__name__} '{elem.id}': .void() is not valid on a Mesh — a "
                f"tessellated mesh can't be an IFC boolean operand, so there is nothing "
                f"for the IfcOpeningElement to cut and the hole would never be counted. "
                f"Use .difference() to sculpt a mesh (it is realised as mesh CSG), or "
                f"void the solid element the hole belongs to."
            )
        for _child in (getattr(elem, "_elements", None) or []):
            _check_relational_voids(_child)
    for _storey in project.storeys:
        for _elem in _storey.elements:
            _check_relational_voids(_elem)
    for _site in (getattr(project, "sites", None) or []):
        _check_relational_voids(_site)

    # DSL v2.1: a Storey obeys the universal segment rule — a NAMED storey
    # contributes ``storey:<leaf>`` to every canonical path under it, an
    # ANONYMOUS storey contributes nothing. That is only sound while there is
    # at most one anonymous storey: two anonymous storeys would be
    # indistinguishable scopes (``wall:north`` on both floors would collide),
    # and a named-plus-anonymous mix hides which floor the anonymous one is.
    # So once a project has more than one storey, EVERY storey must be named.
    # (The common single-storey case stays anonymous → segment-free paths.)
    if len(project.storeys) > 1:
        anon = [i for i, s in enumerate(project.storeys) if not s.name]
        if anon:
            errors.append(
                f"Multi-storey project has {len(anon)} anonymous storey(s) at "
                f"index {anon}: when a project has more than one storey, every "
                f"storey must be named (a lowercase leaf, e.g. "
                f"Storey(name=\"ground\")). Anonymous storeys are only allowed "
                f"for a single-storey project, where the storey contributes no "
                f"canonical-path segment."
            )

    # DSL v2.1: identity is the CANONICAL name (``type:leaf`` pairs, stamped
    # by the naming pass; ``ifc_name`` reads it). Uniqueness is PER-SCOPE — no
    # two same-type siblings sharing a leaf in one container — which is exactly
    # canonical-name uniqueness across the Project: the canonical encodes the
    # full containment path, so a collision can only be (a) two same-type
    # same-leaf siblings in one container, or (b) two top-level elements that
    # would emit the same IFC ``Name`` (identical canonical, e.g. across
    # storeys). Different types with the same leaf coexist (their canonicals
    # differ by the type segment). Anonymous elements (canonical None) never
    # collide. BOTH construction sites are reported (the internal ``id`` in the
    # site label disambiguates same-type siblings).
    #
    # The pre-v2.1 ``id``-uniqueness-across-storeys and name-vs-anonymous-id
    # collision checks are DELETED: ``id`` is internal-only (never an emitted
    # Name) and anonymous elements emit ``Name=None``, so neither can collide.
    seen_canonical: dict[str, str] = {}
    for storey in project.storeys:
        storey_name = storey.name or storey.elevation
        for top in storey.elements:
            for canonical, site in _iter_identity_sites(top, str(storey_name)):
                if canonical in seen_canonical:
                    errors.append(
                        f"Duplicate canonical name '{canonical}': two elements "
                        f"resolve to the same identity — found in "
                        f"{seen_canonical[canonical]} and {site}. Names are "
                        f"per-scope (DSL v2.1): two same-type siblings in one "
                        f"container may not share a leaf. Override name= on one, "
                        f"or move it to a different container so its canonical "
                        f"path differs."
                    )
                else:
                    seen_canonical[canonical] = site

    # DSL Site containers live in ``project.sites`` (P2), not storeys — feed
    # their canonical names into the same per-scope uniqueness check.
    for site_container in getattr(project, "sites", []):
        for canonical, site in _iter_identity_sites(site_container, "site"):
            if canonical in seen_canonical:
                errors.append(
                    f"Duplicate canonical name '{canonical}': two elements "
                    f"resolve to the same identity — found in "
                    f"{seen_canonical[canonical]} and {site}. Names are "
                    f"per-scope (DSL v2.1): two same-type siblings in one "
                    f"container may not share a leaf. Override name= on one, "
                    f"or move it to a different container so its canonical "
                    f"path differs."
                )
            else:
                seen_canonical[canonical] = site

    # DSL v20.0.0: an opening is placed by its HOST, never by itself. The
    # ``offset=``/``sill=`` fields are gone, so a Window/Door reaching the
    # generator without an anchor spec has no position at all — and the
    # generator would silently drop it at the host's start corner. Refuse
    # here, at compile, naming the call that fixes it.
    errors.extend(_opening_placement_errors(project))
    errors.extend(_opening_host_errors(project))

    # DSL Site containers (P2): validate each has at least one renderable
    # child (mirrors the in-storey Site branch below) — Box/Extrude/Mesh, or
    # a semantic Element wrapper (terrain/groundwork) carrying geometry.
    for site_container in getattr(project, "sites", []):
        has_body = any(
            tx.is_prism(c) or tx.is_mesh(c)
            or (type(c).__name__ == "Element"
                and any(tx.is_geometric(s)
                        for s in (getattr(c, "_elements", None) or [])))
            for c in site_container._elements
        )
        if not has_body:
            errors.append(
                f"Site '{site_container.id}': must contain at least one "
                f"renderable child — Box/Extrude/Mesh, or an Element wrapper "
                f"with geometry children"
            )

    # Over EVERY product-capable element, not just the storey's top-level
    # ones. ``.add()`` now takes products, so a Wall can be nested — and a
    # nested Wall gets NO body check otherwise: it reaches the generator,
    # which picks "the first Box/Extrude child", finds none, and emits a
    # wall-shaped hole. Same for a nested Box with zero volume or a nested
    # path-Sweep with no profile source. The checks below were always written
    # per ELEMENT; only the walk was per storey, which is a pre-existing gap
    # this PR's widening makes reachable, so it is fixed here.
    #
    # ``_iter_identity_elements`` yields the top element first, so every
    # top-level element is checked exactly as before.
    for storey in project.storeys:
        for elem in _iter_storey_identity_elements(storey):
            elem_type = type(elem).__name__

            if elem_type == "Wall":
                # Wall is a pure container — validate it has a body Box/Extrude
                from lite_step.models import Extrude as Ex
                # ``tx.body_prisms`` is THE definition, shared with both IFC
                # backends — it excludes an ANCHORED child, which is a detail
                # placed in the wall's frame (a sign, a shelf, a cladding
                # piece) and never the body that DEFINES that frame.
                body_solids = tx.body_prisms(elem)
                # An anchored Box/Extrude is NOT refused here: ``_create_wall``
                # aggregates every non-body child, so the geometry renders. ``test_anchor_matrix`` is the gate that
                # keeps it rendering.
                if not body_solids:
                    # The rule is a derivable FRAME, not a body. A Wall
                    # that aggregates ``.add()``ed leaves needs no solid of its
                    # own: since the frame basis is the element's own
                    # points plus every descendant not anchored INTO it, so
                    # those leaves already supply thickness, run axis and
                    # extents — and ``_create_wall`` emits the identity with
                    # them aggregated under it. Requiring a body on top forced
                    # the author to draw an envelope solid overlapping the very
                    # leaves it contains (the "advanced wall").
                    #
                    # An EMPTY basis is still refused, and loudly: a Wall whose
                    # children are all ``.anchor()``ed has nothing to derive
                    # from — an anchored child is placed BY the frame, so it
                    # cannot define it — and ``frames._bake_child`` would raise
                    # on the missing frame one pass later.
                    errors.extend(_bodyless_container_errors(elem))
                elif len(body_solids) > 1:
                    # The wall body is ONE Box/Extrude: the generator uses the
                    # first and would silently ignore the rest — reject at
                    # compile instead of dropping geometry.
                    errors.append(
                        f"Wall '{elem.id}': contains {len(body_solids)} Box/Extrude "
                        f"children — a Wall takes exactly one body element (the "
                        f"generator would silently drop the others); merge the "
                        f"geometry into one Box/Extrude or split into separate Walls"
                    )
                else:
                    # ONE body — but a body PLUS a buildup of container
                    # children is the envelope the reference describes in
                    # prose (the "advanced wall"). Enforced here so the
                    # prose can go.
                    errors.extend(_envelope_over_buildup_errors(elem))
                if len(body_solids) == 1 and isinstance(body_solids[0], Ex):
                    # Contour-mode wall body (gable walls): the generator's
                    # wall/opening math only holds for a planar VERTICAL
                    # polygon extruded by thickness — loud compile error,
                    # never a generator drop.
                    errors.extend(
                        _wall_contour_body_errors(elem, body_solids[0])
                    )
                # Opening size (explicit or inferred, and positive) is
                # validated for every Window/Door by
                # ``opening_placement_errors`` — checking raw fields here
                # would crash on a legitimately-omitted size.

            elif elem_type == "Site":
                # Site is a container — validate it has at least one renderable
                # child: a Box/Extrude terrain volume, a Mesh (terrain block or
                # site context), or a semantic Element wrapper carrying geometry
                # (terrain/groundwork feature).
                has_body = any(
                    tx.is_prism(c) or tx.is_mesh(c)
                    or (type(c).__name__ == "Element"
                        and any(tx.is_geometric(s)
                                for s in (getattr(c, "_elements", None) or [])))
                    for c in elem._elements
                )
                if not has_body:
                    errors.append(
                        f"Site '{elem.id}': must contain at least one renderable "
                        f"child — Box/Extrude/Mesh, or an Element wrapper with "
                        f"geometry children"
                    )

            elif elem_type == "Box":
                if elem.start is None or elem.end is None:
                    # Defensive: a Box always carries start/end at construction;
                    # this only fires for hand-mutated elements. If it also
                    # carries openings, fail loudly rather than let the generator
                    # crash on the missing frame.
                    if getattr(elem, "_openings", None):
                        errors.append(
                            f"Box '{elem.id}': openings require a box body "
                            f"(start/end) or a vertical planar contour body"
                        )
                else:
                    # Box mode: validate positive volume
                    dx = abs(elem.end.x - elem.start.x)
                    dy = abs(elem.end.y - elem.start.y)
                    dz = abs(elem.end.z - elem.start.z)
                    if dx <= 0 or dy <= 0 or dz <= 0:
                        errors.append(
                            f"Box '{elem.id}': box must have positive dimensions "
                            f"(got {dx}x{dy}x{dz})"
                        )
                    # Openings on a standalone Box: the generator can only
                    # void a box body (volume-checked above) — allowed here.

            elif elem_type == "Extrude":
                # Openings on a standalone Extrude: the generator can
                # only void a vertical planar contour body (same frame math as
                # a gable wall). Anything else must fail HERE, loudly — never
                # let the generator drop the voids silently.
                if getattr(elem, "_openings", None):
                    if elem.contour is not None:
                        errors.extend(_contour_frame_errors(
                            elem.contour,
                            _has_thickness_source(elem),
                            f"Extrude '{elem.id}'",
                            "(a vertical planar contour, like a gable wall); "
                            "for openings use a Box (start=/end=) or a "
                            "vertical planar contour body",
                        ))
                    else:
                        errors.append(
                            f"Extrude '{elem.id}': openings require a Box body "
                            f"(start/end) or a vertical planar contour body"
                        )

            elif elem_type == "Sweep":
                if elem.path is not None:
                    # v1.5 path form: the profile source is profile= or a
                    # member Material carrying profile_mm (one-source rule —
                    # both at once is rejected by the material walk below).
                    if elem.profile is None:
                        from lite_step.materials import element_registry_material
                        mat = element_registry_material(elem)
                        if mat is None or mat.profile_mm is None:
                            errors.append(
                                f"Sweep '{elem.id}': path form needs a profile "
                                f"source — profile=[Point2D x3+] or "
                                f"material=Material(key=..., profile_mm=(w, h))"
                            )
                # Legacy start/end form (Window/Door frames): the section comes
                # from a member Material(profile_mm=); when absent the generator
                # falls back to a standard 89x38 stud section (bim_catalog
                # "Stud_2x4" default; catalog retired). No profile-source error.

    # Registry-material checks (v1.5) — walk every element that can carry
    # material=, including container and opening children.
    from lite_step.materials import iter_material_elements
    for elem in iter_material_elements(project):
        _validate_material_usage(elem, errors, warnings_out)
        _validate_nested_booleans(elem, errors)
        _validate_color(elem, errors)

    # Element-level props checks (v1.5 WS1 PR-D). Shape violations are
    # errors anywhere; well-formed props on boolean-operand geometry
    # (consumed into the parent's shape, never an IFC product) warn — the
    # data would otherwise be dropped silently.
    from lite_step.props import iter_props_carriers, props_shape_errors
    for elem, emits_product in iter_props_carriers(project):
        elem_props = getattr(elem, "props", None) or {}
        if not elem_props:
            continue
        errors.extend(props_shape_errors(elem))
        if not emits_product:
            warnings_out.append(
                f"{type(elem).__name__} '{elem.ifc_name}': props= on boolean "
                f"operand geometry (.cuts()/.adds()/.fills()) is never "
                f"emitted — the operand is consumed into the parent's shape "
                f"and gets no IFC product; move the Pset to the host element"
            )

    # Anchor host resolution (DSL v2.1): host= is a segment-aligned SUFFIX of a
    # canonical name — whole ``type:leaf`` pairs from the tail, leaf-first (e.g.
    # ``host="wall:north"`` matches both ``wall:north`` and
    # ``box:body:wall:north``). Zero matches → miss ERROR; two or more →
    # ambiguity ERROR listing the candidates so the author writes the shortest
    # unambiguous fragment. Caught at compile, not at render.
    from lite_step.models import Anchor as _Anchor
    from lite_step.compiler.naming import resolve_host
    canonical_names = list(seen_canonical.keys())
    for storey in project.storeys:
        storey_name = storey.name or storey.elevation
        for top in storey.elements:
            for elem in _iter_identity_elements(top):
                placement = getattr(elem, "placement", None)
                if not isinstance(placement, _Anchor):
                    continue
                label = (
                    f"{type(elem).__name__} '{elem.ifc_name or elem.id}' "
                    f"({storey_name})"
                )
                status, _matched, candidates = resolve_host(
                    placement.host, canonical_names
                )
                if status == "miss":
                    errors.append(
                        f"{label}: Anchor host {placement.host!r} matches no "
                        f"element — host= is a segment-aligned suffix of a "
                        f"canonical name (whole 'type:leaf' pairs, leaf-first, "
                        f"e.g. 'wall:north'). Known canonical names: "
                        f"{sorted(canonical_names)}"
                    )
                elif status == "ambiguous":
                    errors.append(
                        f"{label}: Anchor host {placement.host!r} is ambiguous "
                        f"— it suffix-matches {len(candidates)} canonical "
                        f"names: {candidates}. Write a longer leaf-first "
                        f"fragment (prepend more 'type:leaf' pairs) to name "
                        f"exactly one."
                    )

    # Placement engine checks (v1.5 WS1 PR-F): a resolver dry-run in
    # collector mode, so everything generation would refuse fails at
    # compile instead — nested/Site/operand placements, anchor cycles,
    # frame-less hosts, malformed rotations, off-enum attach_to. Host
    # existence itself is already covered by the PR-D block above (the
    # resolver's kind="host" failures are filtered as duplicates).
    from lite_step.compiler.placement import placement_validation_issues
    placement_errors, placement_warnings = placement_validation_issues(project)
    errors.extend(placement_errors)
    warnings_out.extend(placement_warnings)

    # Layered elements (v15.2 — layers=LayerSet on planar archetypes): the
    # compiler slices the SINGLE authored body into per-layer solids, so the
    # authoring shape must be checkable here — loud at compile, never a
    # generator drop.
    from lite_step.materials import iter_material_elements
    from lite_step.compiler.layers import through_axis_for
    for elem in iter_material_elements(project):
        layer_set = getattr(elem, "layers", None)
        if layer_set is None:
            continue
        label = f"{type(elem).__name__} '{elem.ifc_name or elem.id}'"
        # An anchored solid is a DETAIL in this element's frame, not a second
        # body — same rule (and same reason) as the plain Wall body check
        # above. Without this, anchoring a cornice onto a layered facade leaf
        # fails validation with a message about the buildup envelope.
        from lite_step.compiler.layers import is_anchored as _is_anchored
        body_solids = [c for c in (getattr(elem, "_elements", None) or [])
                       if tx.is_prism(c) and not _is_anchored(c)]
        if len(body_solids) != 1:
            errors.append(
                f"{label}: layers= requires exactly ONE Box/Extrude body child "
                f"to slice (found {len(body_solids)}) — author the buildup "
                f"envelope once; the compiler slices it per layer")
            continue
        body = body_solids[0]
        if type(elem).__name__ == "Wall" and type(body).__name__ == "Extrude":
            errors.append(
                f"{label}: layers= on a contour-mode (gable) Wall body is a "
                f"follow-up — use a Box body, or split the gable into an "
                f"unlayered Wall")
            continue
        from lite_step.materials import element_registry_material
        if element_registry_material(body) is not None:
            errors.append(
                f"{label}: the body of a layered element must not carry "
                f"material= — per-layer materials come from the layers")
        # Envelope-vs-buildup thickness (WARN): the through extent of the
        # authored body should match the layer sum; slices scale to fill the
        # actual extent, so a mismatch renders without gaps but the buildup
        # data would misstate reality.
        if type(body).__name__ == "Box" and body.start is not None and body.end is not None:
            axis = through_axis_for(elem, body)
            extent = abs(getattr(body.end, axis) - getattr(body.start, axis))
        elif type(body).__name__ == "Extrude" and body.thickness:
            extent = body.thickness
        else:
            extent = None
        total = layer_set.total_thickness_mm
        if extent and total and abs(extent - total) / total > 0.05:
            warnings_out.append(
                f"{label}: body through-extent {extent:.0f} mm differs from "
                f"the layer sum {total} mm by more than 5% — slices scale to "
                f"fill the body; align the envelope or the layer thicknesses")
        # (Unknown layer keys can't reach here: a layer Material carries a
        # thickness, and unknown-key + dimension is a hard error at Material
        # construction — layer materials are registry-backed by contract.)

    # Boolean-operand consumption (v1.5 WS1 PR-E): an element consumed by
    # .cuts()/.adds()/.fills() merges into the consumer's shape and never
    # renders standalone. If the SAME object is also added to the project,
    # the generator will skip it — warn at compile so the double-use is
    # visible before render time.
    from lite_step.models import collect_consumed_operand_ids
    consumed_objs = collect_consumed_operand_ids(project)
    if consumed_objs:
        for storey in project.storeys:
            storey_name = storey.name or storey.elevation
            for elem in storey.elements:
                if id(elem) in consumed_objs:
                    warnings_out.append(
                        f"{type(elem).__name__} '{elem.ifc_name}' "
                        f"({storey_name}): consumed as a boolean operand "
                        f"(.cuts()/.adds()/.fills()) AND added to the "
                        f"project — it will not render standalone; drop "
                        f"the proj.add() if the operand-only behavior is "
                        f"intended"
                    )

    # Shared elements: ONE object reached through more than one path. The
    # canonical name, frame and anchor are stamped per OBJECT, so every
    # placement but the last is silently lost — and per-scope uniqueness means
    # nothing collides and nothing raises.
    #
    # An ERROR since the v10 hard cutover. It was a warning only because
    # refusing it would have stopped the skills corpus compiling, so it waited
    # on the shim-first sequence; the corpus is now clean (measured: zero
    # models warn), which is the condition that release was waiting for.
    # Promoting it matters more than the usual tidy-up because the failure is
    # UNDEFINED rather than wrong: eight placements, one slot to record them
    # in, and a bounding query on such an element answers from whichever write
    # landed last.
    from lite_step.compiler.naming import find_shared_elements
    errors.extend(find_shared_elements(project))

    # A Product occurrence whose shape drifted from the item it names. Beside
    # the shared-element check because it is the same kind of claim — an
    # element asserting a relation the model does not actually support — and
    # because both are cheap tree walks the report already pays for.
    from lite_step.models.product import find_diverged_occurrences
    errors.extend(find_diverged_occurrences(project))

    # Stale world reads: a coordinate answered before its element moved. The
    # value cannot be corrected retroactively, so name the reads to move.
    from lite_step.models.elements import _MUTATION_GENERATION
    stale = sorted({label for label, gen in getattr(project, "_world_queries", [])
                    if gen < _MUTATION_GENERATION})
    if stale:
        warnings_out.append(
            f"world coordinates were read before the model finished changing: "
            f"{', '.join(stale)}. Those reads answered against an earlier "
            f"state, so any value kept from them may be stale. Place parents "
            f"before you read them — root the element, then query it, then "
            f"author against what you got."
        )

    # Explicit Mesh validation: watertightness check and "hold water" (signed volume) check.
    # Only tests geometry emitted as a Mesh (other primitives like Wall/Slab are compiler-managed solids).
    open_mesh_names = []
    inverted_mesh_names = []
    for mesh in _iter_all_project_meshes(project):
        faces = getattr(mesh, "faces", None)
        vertices = getattr(mesh, "vertices", None)
        mesh_name = getattr(mesh, "name", None) or getattr(mesh, "id", "mesh")

        # C3: Validate face index bounds upfront. If invalid, record a compile error.
        if faces is not None and vertices is not None and len(faces) > 0 and len(vertices) > 0:
            import numpy as np
            f_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
            f_min, f_max = int(f_arr.min()), int(f_arr.max())
            if f_min < 0 or f_max >= len(vertices):
                errors.append(
                    f"Mesh '{mesh_name}': face index out of range [0, {len(vertices) - 1}] "
                    f"(min={f_min}, max={f_max}) — invalid face indices cannot be compiled."
                )
                continue

        # N7 & C1: Skip guard and opt-outs.
        # Heightmaps set is_watertight=True (closed by construction).
        # Authors can explicitly opt out of open-surface warnings for scans/ribbons/canopies via
        # is_watertight=False or recognized scan/context source_type.
        is_wt = getattr(mesh, "is_watertight", False)
        source_type = getattr(mesh, "source_type", None)
        fields_set = getattr(mesh, "model_fields_set", set())

        if source_type in ("LiDAR", "photogrammetry", "scan", "survey", "context", "Canopy detection") or (
            "is_watertight" in fields_set and not is_wt
        ):
            continue

        if is_wt:
            is_closed, _ = check_mesh_watertightness(mesh)
            if is_closed:
                vol = check_mesh_signed_volume(mesh)
                if vol < -1e-4:
                    inverted_mesh_names.append((mesh_name, vol))
            continue

        is_closed, _ = check_mesh_watertightness(mesh)
        if not is_closed:
            open_mesh_names.append(mesh_name)
        else:
            vol = check_mesh_signed_volume(mesh)
            if vol < -1e-4:
                inverted_mesh_names.append((mesh_name, vol))

    # N8: Aggregate warnings per project rather than repeating verbatim per mesh.
    if open_mesh_names:
        names_str = ", ".join(f"'{n}'" for n in open_mesh_names)
        warnings_out.append(
            f"{len(open_mesh_names)} mesh(es) are open surfaces (not watertight solid volumes): {names_str}. "
            f"Building elements (such as roofs, slabs, and walls) should be authored as closed watertight solids with thickness. "
            f"Downsides: open meshes cannot receive boolean cuts/openings (.difference()), disappear from underneath in 3D viewers due to backface culling, and have zero physical BIM volume."
        )

    if inverted_mesh_names:
        details = ", ".join(f"'{n}' ({v:,.0f} mm³)" for n, v in inverted_mesh_names)
        warnings_out.append(
            f"{len(inverted_mesh_names)} mesh(es) have negative signed volume (inverted inside-out winding): {details}. "
            f"Surface cannot hold water. Reverse triangle vertex winding order (i, j, k) -> (i, k, j) so face normals point outward."
        )

    # A vertical Extrude swept the wrong way by its winding order. Last,
    # because it reads every element's AABB and there is no reason to pay for
    # that on a project that already failed something structural.
    #
    # The neighbouring inverted-mesh check above is a different defect with a
    # similar name: that one is a mesh whose FACES point inward, this one is a
    # solid that is the right shape in the wrong PLACE. Both are winding, and
    # confusing them costs an author an afternoon.
    from lite_step.compiler.winding import (check_extrude_winding,
                                             check_extrude_zero_thickness)
    warnings_out.extend(check_extrude_zero_thickness(project))
    warnings_out.extend(check_extrude_winding(project))

    return ValidationReport(errors=errors, warnings=warnings_out)


def _iter_all_project_meshes(project: Project) -> Iterator[Mesh]:
    """Yield every Mesh in project (storeys, sites, and boolean operand trees)."""
    seen = set()

    def _walk(elem) -> Iterator[Mesh]:
        if elem is None or id(elem) in seen:
            return
        seen.add(id(elem))
        if tx.is_mesh(elem):
            yield elem
        # N1: descend child elements, openings, and boolean operands
        for child in (getattr(elem, "_elements", None) or []):
            yield from _walk(child)
        for child in (getattr(elem, "_openings", None) or []):
            yield from _walk(child)
        for slot in ("_cuts", "_adds", "_intersects", "_fills", "_voids"):
            for op in (getattr(elem, slot, None) or []):
                yield from _walk(op)

    for storey in getattr(project, "storeys", []) or []:
        for elem in getattr(storey, "elements", []) or []:
            yield from _walk(elem)
    for site in getattr(project, "sites", []) or []:
        yield from _walk(site)


def check_mesh_watertightness(mesh: Mesh) -> tuple[bool, int]:
    """Check if a mesh is a topologically closed 2-manifold without boundary edges.

    Welds coincident vertices geometrically before edge counting (C2) to avoid
    false positives on closed solids with unshared vertex indices, and packs
    undirected edges into a 1-D 64-bit integer key for high performance (N2).

    Returns (is_watertight, num_boundary_edges).
    """
    faces = getattr(mesh, "faces", None)
    vertices = getattr(mesh, "vertices", None)
    if faces is None or vertices is None or len(faces) == 0 or len(vertices) == 0:
        return False, 0
    if len(faces) < 4 or len(vertices) < 4:
        return False, len(faces) * 3

    import numpy as np
    from lite_step.ifc.geometry import weld_coincident_vertices

    if isinstance(vertices, np.ndarray):
        v = vertices.astype(np.float64).reshape(-1, 3)
    else:
        v = np.array([(p.x, p.y, p.z) for p in vertices], dtype=np.float64).reshape(-1, 3)

    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    # C3: Check index bounds upfront
    f_min = int(f.min())
    f_max = int(f.max())
    if f_min < 0 or f_max >= len(v):
        raise ValueError(f"Mesh face index out of range [0, {len(v) - 1}]: min={f_min}, max={f_max}")

    # C2: Weld coincident vertices geometrically
    welded_v, welded_f = weld_coincident_vertices(v, f)
    if len(welded_f) == 0:
        return False, 0

    # N2: 1-D packed 64-bit integer key (30x faster than np.unique(axis=0))
    n_verts = len(welded_v)
    e0 = np.minimum(welded_f[:, 0], welded_f[:, 1])
    e1 = np.maximum(welded_f[:, 0], welded_f[:, 1])
    e2 = np.minimum(welded_f[:, 1], welded_f[:, 2])
    e3 = np.maximum(welded_f[:, 1], welded_f[:, 2])
    e4 = np.minimum(welded_f[:, 2], welded_f[:, 0])
    e5 = np.maximum(welded_f[:, 2], welded_f[:, 0])

    u = np.concatenate([e0, e2, e4]).astype(np.int64)
    w = np.concatenate([e1, e3, e5]).astype(np.int64)
    packed = u * n_verts + w
    _, counts = np.unique(packed, return_counts=True)
    open_edges = int(np.sum(counts != 2))
    return open_edges == 0, open_edges


def check_mesh_signed_volume(mesh: Mesh) -> float:
    """Compute exact signed volume of a mesh via Gauss's Divergence Theorem.

    For any closed solid, positive volume means outward-facing normals;
    negative volume means inward-facing (inverted) normals.
    """
    faces = getattr(mesh, "faces", None)
    vertices = getattr(mesh, "vertices", None)
    if faces is None or vertices is None or len(faces) == 0 or len(vertices) == 0:
        return 0.0
    if len(faces) < 4 or len(vertices) < 4:
        return 0.0

    import numpy as np

    if isinstance(vertices, np.ndarray):
        v = vertices.astype(np.float64).reshape(-1, 3)
    else:
        v = np.array([(p.x, p.y, p.z) for p in vertices], dtype=np.float64).reshape(-1, 3)

    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    # C3: Check index bounds upfront
    f_min = int(f.min())
    f_max = int(f.max())
    if f_min < 0 or f_max >= len(v):
        raise ValueError(f"Mesh face index out of range [0, {len(v) - 1}]: min={f_min}, max={f_max}")

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    c0 = v0[:, 1] * v1[:, 2] - v0[:, 2] * v1[:, 1]
    c1 = v0[:, 2] * v1[:, 0] - v0[:, 0] * v1[:, 2]
    c2 = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    return float(np.sum(c0 * v2[:, 0] + c1 * v2[:, 1] + c2 * v2[:, 2]) / 6.0)


def _envelope_over_buildup_errors(elem) -> list[str]:
    """A body AND a buildup of container children is an envelope over parts.

    The reference told authors "do NOT author an envelope solid over a
    buildup", and told them why: it becomes phantom volume where the parts
    stop, or a zero-vertex boolean stack where they do not. Measured on the
    shape that motivated it — under-tiled, the ``IfcWall`` reports 0.92-2.13
    m3 of volume that is not there; tiled correctly, ``create_shape`` returns
    ZERO vertices behind 29 ``IfcBooleanResult``s, so the wall renders as
    nothing at all. Both are silent.

    A prohibition the compiler does not enforce is a rule that exists only in
    prose, and this one is cheap to enforce: the two shapes are distinguished
    by whether the container has a body of its own. A Wall aggregating leaves
    derives thickness, run axis and extents FROM them and needs no body; a
    Wall with a body is one element. Wanting both is wanting the leaves to be
    parts and the envelope to be the whole, which is what an aggregate is —
    so the remedy is to drop the envelope.

    Corpus reach: ZERO of 20 models.
    Anchored children are exempt — a corbel or a sill anchored to a bodied
    wall is a detail ON it, not a layer OF it, which is the ``.add()`` /
    ``.anchor()`` distinction doing its usual work.
    """
    from lite_step.compiler import frames as _frames
    from lite_step.models import taxonomy as _tx

    body = _tx.body_prisms(elem)[0]
    body_box = _frames.element_aabb(body)
    if body_box is None:
        return []

    def _inside(box, outer, eps=1e-6):
        (a0, b0, c0), (a1, b1, c1) = box
        (x0, y0, z0), (x1, y1, z1) = outer
        return (a0 >= x0 - eps and b0 >= y0 - eps and c0 >= z0 - eps
                and a1 <= x1 + eps and b1 <= y1 + eps and c1 <= z1 + eps)

    def _vol(box):
        (x0, y0, z0), (x1, y1, z1) = box
        return max(0.0, x1 - x0) * max(0.0, y1 - y0) * max(0.0, z1 - z0)

    parts = []
    for child in (getattr(elem, "_elements", None) or []):
        if _frames.is_anchored_child(child):
            continue
        if _tx.is_solid(child) or _tx.is_mesh(child) or _tx.brings_void(child):
            continue
        kids = getattr(child, "_elements", None) or []
        if not kids:
            continue
        if not any(_tx.is_solid(k) or _tx.is_mesh(k) or (getattr(k, "_elements", None))
                   for k in kids):
            continue
        # GEOMETRY decides, not structure. A Column or an Element assembly
        # ADDED to a bodied wall is ordinary composition ("elements can
        # contain other elements") and must stay legal; what makes an
        # ENVELOPE is that the body wraps the parts and adds nothing. So the
        # part must lie INSIDE the body. A purely structural test — "a body
        # plus container children" — refuses 50 legitimate fixtures, measured.
        child_box = _frames.element_aabb(child)
        if child_box is None or not _inside(child_box, body_box):
            continue
        parts.append((child, child_box))

    if not parts:
        return []
    # ...and together they must FILL it. A small detail sitting inside the
    # wall's bounding box (a socket box, a niche lining) is not a buildup;
    # a buildup reconstructs the body. Half the body's volume is the line:
    # two leaves of a cavity wall clear it comfortably, a detail does not.
    filled = sum(_vol(box) for _child, box in parts)
    if _vol(body_box) <= 0 or filled < 0.5 * _vol(body_box):
        return []
    parts = [child for child, _box in parts]

    named = ", ".join(
        f"{type(pt).__name__} '{getattr(pt, 'name', None) or pt.id}'"
        for pt in parts[:4])
    return [
        f"Wall '{getattr(elem, 'name', None) or elem.id}': has its own body AND "
        f"{len(parts)} container child(ren) that carry geometry ({named}) — an "
        f"envelope solid over a buildup. The envelope becomes phantom volume "
        f"where the parts stop, or a zero-vertex boolean stack where they do "
        f"not, and both are silent. Drop the body: a Wall aggregating "
        f".add()ed leaves derives its thickness, run axis and extents from "
        f"them and still hosts .anchor()/.opening(). Keep the body instead if "
        f"the parts were meant as details, and .anchor() them."
    ]


def _validate_nested_booleans(elem, errors: list[str]) -> None:
    """A boolean operand carrying booleans OF ITS OWN is refused.

    ``a.difference(b)`` takes b's SHAPE, not b's boolean tree — every cut,
    union, intersection and clip already on b is dropped on the floor.
    Measured on a 3.0 m3 body cut by a 1.0 m3 tool that itself had a 0.5 m3
    cut: the result is **2.000 m3 either way**. Honouring b would give 2.5.
    Nothing warned, and the two models are indistinguishable without
    tessellating the file.

    The shape it is usually reached for is faking an intersection —
    ``a.difference(b.difference(c))`` reads like "a minus (b minus c)" and
    compiles to plain ``a - b``. ``.intersection()`` exists and emits
    ``IfcBooleanResult INTERSECTION``, so the remedy is a real verb rather
    than a nesting trick.

    An ERROR rather than a warning: the operand's own geometry is *authored
    intent* that the file will not contain, and unlike a colour it changes
    volumes and schedules. There is no reading under which the discard is
    what the author wanted.
    """
    label = f"{type(elem).__name__} '{getattr(elem, 'ifc_name', None) or elem.id}'"
    for verb, attr in (("difference", "_cuts"),
                       ("union", "_adds"),
                       ("intersection", "_intersects")):
        for operand in (getattr(elem, attr, None) or []):
            # ONLY the operand's own CUTS. A ``.union()`` on an operand is
            # HONOURED — the generator emits a nested
            # ``IfcBooleanResult(UNION)`` as the second operand, which is how
            # an arched doorway is authored (``wall.difference(door.union(
            # arch))``) and what ``test_arched_doorway_measured_volume``
            # measures. Refusing that too would forbid a working, tested
            # composition. The discard is verb-specific, and the DSL
            # reference said so; this check is narrowed to the half that was
            # measured rather than the half that reads symmetrically.
            carried = [".difference()"] if getattr(operand, "_cuts", None) else []
            if not carried:
                continue
            oname = (getattr(operand, "ifc_name", None)
                     or getattr(operand, "name", None) or operand.id)
            errors.append(
                f"{label}: the .{verb}() operand {type(operand).__name__} "
                f"{oname!r} carries its own {', '.join(sorted(set(carried)))} — "
                f"a boolean operand contributes its SHAPE only, so those are "
                f"DISCARDED and the emitted volume is as if they were never "
                f"written. Flatten it: pass the operands to {label.split()[0]} "
                f"directly. If you meant 'keep only the common volume', that "
                f"is .intersection(), which nesting differences does not "
                f"reproduce."
            )
def _validate_color(elem, errors: list[str]) -> None:
    """A ``color=`` the renderer cannot read is a compile ERROR.

    Without it, ``Box(color="#FF8800")`` is accepted at construction, emits
    ``IfcColourRgb=[]`` and renders viewer-default — the element looks
    *deliberately* grey, with no log line and no way to notice short of opening
    the file. That is the failure mode this codebase refuses to ship.

    Loud rather than a warning because a colour is never load-bearing for
    geometry: fixing the string costs nothing, and the alternative is a model
    that is quietly the wrong colour for as long as nobody looks.
    """
    from lite_step.ifc.colors import color_vocabulary_hint, resolve_color

    value = getattr(elem, "color", None)
    if value is None or not str(value).strip():
        return
    if resolve_color(value) is not None:
        return
    label = f"{type(elem).__name__} '{getattr(elem, 'ifc_name', None) or elem.id}'"
    errors.append(
        f"{label}: color={value!r} names no colour — "
        f"{color_vocabulary_hint(value)}"
    )


def _validate_material_usage(elem, errors: list[str], warnings_out: list[str]) -> None:
    """Registry-material checks for a single element (DSL v1.5).

    Channels (see ``lite_step.materials``):
    - ``None`` / legacy vocabulary / ``"Void"`` / hex strings → silent,
      exact pre-v1.5 behavior;
    - registry key → OK; dimension pairing enforced below;
    - unknown key → WARNING (open vocabulary — emitted as a plain named
      IfcMaterial by the generator);
    - one-source violations and dimensions on the wrong primitive → ERROR.
    """
    from lite_step.materials import element_registry_material, registry_definition

    mat = element_registry_material(elem)
    if mat is None:
        return

    etype = type(elem).__name__
    label = f"{etype} '{elem.id}'"

    mdef = registry_definition(mat.key)
    if mdef is None:
        # Unknown key without dimensions (with dimensions, Material
        # construction already raised). Open vocabulary: loud, non-fatal.
        warnings_out.append(
            f"{label}: unknown material {mat.key!r} — emitted as a plain named "
            f"material (open vocabulary); check the registry keys if a catalog "
            f"material was intended"
        )

    if mat.profile_mm is not None:
        if etype != "Sweep":
            errors.append(
                f"{label}: Material(profile_mm=...) is member stock and pairs "
                f"with Sweep; it cannot size a {etype}"
            )
        elif getattr(elem, "profile", None) is not None:
            errors.append(
                f"{label}: one-source rule — Material(profile_mm=...) replaces "
                f"profile=; provide exactly one profile source"
            )

    if mat.thickness_mm is not None:
        if etype == "Extrude":
            if getattr(elem, "contour", None) is None:
                errors.append(
                    f"{label}: Material(thickness_mm=...) pairs with contour "
                    f"mode; an Extrude already defines its own dimensions"
                )
            elif getattr(elem, "thickness", None) is not None:
                errors.append(
                    f"{label}: one-source rule — Material(thickness_mm=...) "
                    f"replaces thickness=; provide exactly one thickness source"
                )
        elif etype == "Box":
            errors.append(
                f"{label}: Material(thickness_mm=...) pairs with sheet "
                f"primitives (Extrude); a Box carries its own dimensions"
            )
        else:
            errors.append(
                f"{label}: Material(thickness_mm=...) pairs with sheet "
                f"primitives (Extrude); {etype} carries its own "
                f"thickness"
            )
