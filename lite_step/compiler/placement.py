"""DSL v1.5 placement engine (WS1 PR-F): Transform + Anchor consumption.

Computes the 4x4 world matrix of every placed top-level element. The
generator composes each matrix ON TOP of the product's own IFC object
placement (see ``lite_step.ifc.generator._apply_dsl_placement`` for the
application-point decision), giving rigid-body semantics: an element with
both its own geometry coordinates and a ``placement=`` authors its geometry
LOCAL to the placement origin, and the placement moves/rotates the whole
element as one.

Pinned conventions (WS1 PR-F — proven by measured-geometry tests in
``lite_step/tests/test_placement_engine.py``):

**Transform** — ``world = T(origin) @ R(rot_1) @ R(rot_2) @ ... @ local``:

* Rotations are about the element's LOCAL ORIGIN (0, 0, 0) of its authoring
  frame — the point that lands at ``origin`` in world — not about any
  geometry center.
* The rotation list composes left-to-right in list order
  (``R_total = R_1 @ R_2 @ ... @ R_n``), i.e. an INTRINSIC sequence: each
  subsequent rotation is about the element's already-rotated axes. This is
  the composition order the removed ``Box(rotations=)`` field used
  (``rotation_matrix = rotation_matrix @ rot`` in list order), kept so stored
  sources that rotated through ``Transform`` do not change.
* Angles are centidegrees (RULE 3), right-hand rule about the named axis;
  axes are ``"x"``/``"y"``/``"z"`` (lowercase, construction-enforced).

**Anchor** — ``world = M_host @ T(attach + offsets) @ B_host @ R(rots)``:

* ``host=`` resolves against CANONICAL names by segment-aligned suffix match
  (DSL v2.1 — whole ``type:leaf`` pairs from the tail, e.g. ``host="wall:north"``
  matches ``box:body:wall:north``; ambiguity and miss are compile ERRORs). The
  host may be a container child (e.g. a named wall-body Box) — its world matrix
  is its top-level ancestor's resolved placement (children ride rigidly with
  their container).
* The host's local frame is the one STAMPED by
  :mod:`lite_step.compiler.frames` — this module does not derive it. The
  stamp carries along/across/up, the corner origin and ``facing``; the
  table below is the pinned per-rule semantics it implements (the "rule"
  is the host's own type or its nearest Wall/Beam/Slab ancestor container;
  a generic ``Element`` maps via its ``ifc_class``):

  ============  =========================================================
  rule          frame
  ============  =========================================================
  Wall          along = the LONGER horizontal AABB axis (+X on ties —
                same tie-break as the wall-opening math), up = +Z,
                across = up x along (right-handed).
                Anchor line = the wall BASELINE: along the wall at the
                base (z = AABB min), centered in the thickness.
  Beam          along = the longer horizontal AABB axis (+X on ties),
                up = +Z, across = up x along. Anchor line = the beam
                CENTERLINE (across mid, up mid).
  Slab          along = +X, across = +Y, up = +Z (literal, per spec).
                Anchor line = along X on the TOP surface (z = AABB max)
                at mid Y — "up = above/below surface".
  fallback      the Slab rule. Applies to every other host type
                (Column, Roof, bare Solid, ...) — documented deferral;
                the spec table only defines Wall/Beam/Slab frames.
  ============  =========================================================

* ``attach_to`` points (host authoring space): ``start``/``end`` = the
  anchor line's min-along / max-along ends; ``center`` = the host's 3D
  geometric (AABB) center; ``face`` = Wall: center of the EXTERIOR face —
  the across-side facing AWAY from the nearest containment scope (the
  enclosing Space, else the storey, else the building; ties pick the max
  side) — Beam/Slab/fallback: center of the TOP face.
* Offsets are applied along the UNROTATED host frame axes
  (``offset_along * along - offset_inset * facing * across + offset_up * up``)
  from the attach point. ``offset_inset`` is FACING-SIGNED: positive goes
  INTO the host, away from the weather, so one value behaves the same on a
  north and a south wall (the raw ``across`` axis sign is the
  cross-product's and flips between them). It is the same axis and the same
  direction as an opening child's local ``+y`` — see ``compiler.frames``. ``rotations=`` then rotate the
  anchored element about the offset attach point, with ``"x"``/``"y"``/
  ``"z"`` naming the anchored frame's along/across/up axes (same intrinsic
  composition as Transform).
* The anchored element's local axes align to the host frame: local X =
  along, local Y = across, local Z = up. Geometry is authored local to the
  attach point (same rule as Transform).
* Chained anchors (host itself placed, by Transform or Anchor) resolve
  recursively; a cycle is a compile ERROR (``validate_project_report``)
  and a loud :class:`PlacementResolutionError` here.

Scope (WS1 PR-F): placements are consumed on TOP-LEVEL storey elements
only. A placement on a nested element (container child, opening child,
boolean operand) or on a ``Site`` container (whose children emit as
individual site products — the container is never a product) is a compile
ERROR — loud, never silently ignored. Place the container instead and
author children in container-local coordinates.

Turn/Rod hosts: the AABB derives from the axis/centerline path points only
(radial extents are not padded) — anchor to boxy hosts when sub-radius
precision matters. Documented limitation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lite_step.compiler import frames
from lite_step.compiler.walkguard import WalkGuard

logger = logging.getLogger(__name__)

#: attach_to vocabulary (mirrors the Anchor model Literal — re-checked at
#: compile for placements built via ``model_construct`` / legacy paths).
ATTACH_TO_VALUES = ("start", "end", "center", "face")

#: rotation axis vocabulary (mirrors the construction validator).
ROTATION_AXES = ("x", "y", "z")


class PlacementResolutionError(ValueError):
    """A placement could not be resolved to a world matrix.

    ``kind`` classifies the failure so ``validate_project_report`` can
    fold resolver errors into its report without double-reporting the
    host-existence errors the PR-D block already emits:

    * ``"host"`` — the Anchor host reference does not resolve (already an
      ERROR in the PR-D validation block; filtered as a duplicate there).
    * ``"cycle"`` — anchor chain loops back on itself.
    * ``"frame"`` — the host has no geometry to derive a frame from.
    * ``"unsupported"`` — placement on a nested element or Site container.
    * ``"rotation"`` — malformed rotation list (defense for
      ``model_construct``-built placements; construction validates).
    """

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Rotation / Transform math
# ---------------------------------------------------------------------------


def compose_rotations(rotations: Optional[List[Tuple[str, int]]]) -> np.ndarray:
    """3x3 rotation from a centidegree ``rotations=`` list.

    PINNED (WS1 PR-F): composes in list order, ``R = R_1 @ R_2 @ ... @ R_n``
    — an intrinsic sequence (each rotation about the already-rotated element
    axes), matching the legacy box-rotation composition in
    ``_create_solid_box``. Applied to column vectors: ``world = R @ local``.
    """
    from lite_step.ifc.geometry import axis_rotation_matrix_3x3

    m = np.eye(3, dtype=np.float64)
    for axis, angle_centideg in rotations or []:
        m = m @ axis_rotation_matrix_3x3(
            axis, math.radians(float(angle_centideg) / 100.0)
        )
    return m


def transform_matrix(placement) -> np.ndarray:
    """4x4 world matrix for a ``Transform`` placement.

    ``world = T(origin) @ R_seq @ local`` — rotations about the element's
    local origin (the point that lands at ``origin``), see module docstring.
    """
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = compose_rotations(placement.rotations)
    m[0, 3] = float(placement.origin.x)
    m[1, 3] = float(placement.origin.y)
    m[2, 3] = float(placement.origin.z)
    return m


def rotation_issues(rotations: Any, label: str) -> List[str]:
    """Report-level defense for rotation lists (construction validates the
    same rules; this catches ``model_construct``-built placements)."""
    issues: List[str] = []
    if rotations is None:
        return issues
    if not isinstance(rotations, (list, tuple)):
        return [f"{label}: rotations must be a list of (axis, centidegrees) "
                f"tuples — got {type(rotations).__name__}"]
    for i, item in enumerate(rotations):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            issues.append(
                f"{label}: rotations[{i}] must be an (axis, centidegrees) "
                f"tuple, e.g. (\"z\", 4500) — got {item!r}"
            )
            continue
        axis, angle = item
        if not isinstance(axis, str) or axis not in ROTATION_AXES:
            issues.append(
                f"{label}: rotations[{i}] has unknown axis {axis!r} — axes "
                f"are \"x\", \"y\", \"z\" (lowercase)"
            )
        if isinstance(angle, bool) or not isinstance(angle, (int, float)):
            issues.append(
                f"{label}: rotations[{i}] angle must be int centidegrees — "
                f"got {angle!r}"
            )
    return issues


# ---------------------------------------------------------------------------
# Project walk (identity + host maps, authoring-space geometry)
# ---------------------------------------------------------------------------


@dataclass
class _HostNode:
    """A host-resolvable element with its ancestry."""

    elem: Any
    top: Any                 # top-level storey element it rides with
    chain: Tuple[Any, ...]   # ancestors, top-level element first


def _iter_identity_nodes(top) -> "list[_HostNode]":
    """``top`` and every product-capable descendant, with ancestry.

    Mirrors ``executor._iter_identity_elements`` (container ``_elements``
    and ``.opening()`` attachments) — the population Anchor hosts resolve
    against per PR-D.

    On-path re-entry (a containment cycle) is PRUNED — a termination guard.
    This walk is reachable from ``BimElement.world_aabb()``, which an author
    may call at any time, so it must survive a tree the validator has not
    seen yet; ``compiler.extent.element_extent`` is the one that then refuses
    to answer.
    """
    out: List[_HostNode] = []
    guard = WalkGuard()

    def walk(elem, chain: Tuple[Any, ...]) -> None:
        if not guard.enter(elem):
            return
        try:
            out.append(_HostNode(elem=elem, top=top, chain=chain))
            for child in (getattr(elem, "_elements", None) or []):
                walk(child, chain + (elem,))
            for child in (getattr(elem, "_openings", None) or []):
                walk(child, chain + (elem,))
        finally:
            guard.leave(elem)

    walk(top, ())
    return out


def _iter_operands(elem, _on_path=None):
    """Every element consumed as a boolean operand under ``elem``
    (recursively) — never a product, so a placement there is dead data.

    Same on-path guard, same reason: reachable from a world query.
    """
    guard = _on_path if _on_path is not None else WalkGuard()
    if not guard.enter(elem):
        return
    try:
        for list_name in ("_cuts", "_adds", "_fills"):
            for op in (getattr(elem, list_name, None) or []):
                yield op
                yield from _iter_operands(op, guard)
        for child in (getattr(elem, "_elements", None) or []):
            yield from _iter_operands(child, guard)
        for child in (getattr(elem, "_openings", None) or []):
            yield from _iter_operands(child, guard)
    finally:
        guard.leave(elem)


def _elem_label(elem) -> str:
    name = getattr(elem, "ifc_name", None) or getattr(elem, "id", None)
    return f"{type(elem).__name__} '{name}'"


# The authoring-space AABB probe lives in ``lite_step.compiler.frames`` — the
# single derivation this module reads. Re-exported under its historical private
# name because ``extent.py`` imports it from here (extent.py:309 and :643, the
# only non-test consumers); the alias keeps those call sites working without a
# second copy of the walk existing anywhere.
#
# Its two former siblings — ``_iter_geometry_points`` and
# ``_ELEMENT_CLASS_RULES`` — are gone: nothing imported either, and the comment
# they shared claimed ``displacement.py`` and ``generator.py`` read them, which
# neither did.
_element_aabb = frames.element_aabb


@dataclass
class _HostFrame:
    along: np.ndarray
    across: np.ndarray
    up: np.ndarray
    facing: int                    # +1: outward is +across; -1: outward is -across
    points: Dict[str, np.ndarray]  # attach_to -> host authoring-space point


def _host_frame_from_stamp(stamped) -> _HostFrame:
    """Adapt an :class:`~lite_step.compiler.frames.ElementFrame` stamp.

    The stamp is the SINGLE derivation of along/across/up + facing (see
    ``lite_step.compiler.frames``); this only reshapes it into the numpy
    ``_HostFrame`` the anchor math consumes, deriving the four attach
    points from the stamped corner origin + extents. ``facing`` is carried
    through so ``offset_inset`` can be resolved inward-positive.
    """
    return _HostFrame(
        along=np.array(stamped.along, dtype=np.float64),
        across=np.array(stamped.across, dtype=np.float64),
        up=np.array(stamped.up, dtype=np.float64),
        facing=int(stamped.facing),
        points={
            name: np.array(stamped.point(name), dtype=np.float64)
            for name in ATTACH_TO_VALUES
        },
    )


def _anchor_local_matrix(anchor, stamped) -> np.ndarray:
    """Host-authoring-space matrix for an Anchor (before the host's own
    placement composes on the left): ``T(attach + offsets) @ B @ R_seq``.

    ``offset_inset`` is FACING-signed: positive always moves INWARD, so the
    same value behaves identically on a north and a south wall.
    """
    frame = _host_frame_from_stamp(stamped)
    attach_to = getattr(anchor, "attach_to", "center")
    attach = frame.points[attach_to]
    origin = (
        attach
        + frame.along * float(anchor.offset_along)
        - frame.across * (float(anchor.offset_inset) * frame.facing)
        + frame.up * float(anchor.offset_up)
    )
    m = np.eye(4, dtype=np.float64)
    basis = np.column_stack([frame.along, frame.across, frame.up])
    # rotations about the anchored frame ("x"=along, "y"=across, "z"=up),
    # intrinsic list order — composing R_seq on the RIGHT of the basis is
    # exactly "rotate about the anchored frame's axes".
    m[:3, :3] = basis @ compose_rotations(anchor.rotations)
    m[:3, 3] = origin
    return m


# ---------------------------------------------------------------------------
# Resolution driver
# ---------------------------------------------------------------------------


def resolve_placement_matrices(
    project,
    collect_errors: Optional[List[PlacementResolutionError]] = None,
) -> Dict[int, np.ndarray]:
    """Resolve every placed TOP-LEVEL storey element to a 4x4 world matrix.

    Returns ``{id(element): matrix}`` keyed by Python object identity —
    the generator looks products up by the exact element instance it is
    iterating (the normalized project), so identity is the right key.

    Unit-agnostic: matrices come out in whatever units the project's
    coordinates are in (the generator calls this AFTER
    ``normalize_project_to_meters``; validation calls it in mm — the
    checks are unit-independent).

    Failure behavior: with ``collect_errors=None`` (the generation path)
    the first :class:`PlacementResolutionError` raises — LOUD, wrapped by
    ``generate_ifc`` into a failed result. With a list (the validation
    path) every error is collected per element and resolution continues.
    """
    from lite_step.models import Anchor, Site, Transform
    from lite_step.compiler.naming import resolve_host, stamp_canonical_names

    # DSL v2.1: hosts resolve against CANONICAL names (segment-aligned suffix
    # match). Ensure stamps exist — idempotent; the compile/validate callers
    # already stamp, this covers direct resolver calls (tests).
    stamp_canonical_names(project)
    # Same reasoning for frames — idempotent, covers direct resolver calls.
    from lite_step.compiler.frames import stamp_frames
    stamp_frames(project)

    by_canonical: Dict[str, _HostNode] = {}
    tops: List[Any] = []
    nested_placed: List[_HostNode] = []

    for storey in project.storeys:
        for top in storey.elements:
            tops.append(top)
            for node in _iter_identity_nodes(top):
                canonical = getattr(node.elem, "_canonical_name", None)
                if canonical and canonical not in by_canonical:
                    by_canonical[canonical] = node
                if node.chain and getattr(node.elem, "placement", None) is not None:
                    nested_placed.append(node)

    # DSL Site containers live in ``project.sites`` (P2), not storeys — a Site
    # (or a site child) carrying placement= must still fail the same checks
    # below (Site placement is unsupported; nested placement on a site child is
    # unconsumed). Walk them the same way as storey tops.
    for site in getattr(project, "sites", []):
        tops.append(site)
        for node in _iter_identity_nodes(site):
            canonical = getattr(node.elem, "_canonical_name", None)
            if canonical and canonical not in by_canonical:
                by_canonical[canonical] = node
            if node.chain and getattr(node.elem, "placement", None) is not None:
                nested_placed.append(node)

    errors_out = collect_errors

    def _fail(message: str, kind: str) -> None:
        exc = PlacementResolutionError(message, kind)
        if errors_out is not None:
            errors_out.append(exc)
        else:
            raise exc

    # Placements this pass does not consume are ERRORS, never silent skips.
    for node in nested_placed:
        _fail(
            f"{_elem_label(node.elem)}: placement= on a nested element "
            f"(child of {_elem_label(node.chain[-1])}) is not consumed by "
            f"the placement engine (WS1 PR-F scope) — place the top-level "
            f"element and author children in container-local coordinates",
            kind="unsupported",
        )
    for top in tops:
        for op in _iter_operands(top):
            if getattr(op, "placement", None) is not None:
                _fail(
                    f"{_elem_label(op)}: placement= on a boolean operand "
                    f"(.cuts()/.adds()/.fills()) is never consumed — the "
                    f"operand merges into its host's shape; author the "
                    f"operand in the host's coordinates",
                    kind="unsupported",
                )

    def _resolve_host_node(anchor) -> Optional[_HostNode]:
        # host= resolves against CANONICAL names by segment-aligned suffix
        # match (DSL v2.1). A unique match resolves; miss / ambiguity return
        # None here (the executor's host-resolution block reports the precise
        # miss-vs-ambiguity error, and kind="host" resolver errors are filtered
        # as duplicates in placement_validation_issues).
        status, matched, _candidates = resolve_host(anchor.host, by_canonical.keys())
        if status == "ok":
            return by_canonical[matched]
        return None

    resolved: Dict[int, np.ndarray] = {}
    in_progress: List[Any] = []  # top-level elements on the resolution stack

    def _top_matrix(top) -> np.ndarray:
        key = id(top)
        if key in resolved:
            return resolved[key]
        if any(id(t) == key for t in in_progress):
            chain = " -> ".join(
                _elem_label(t) for t in in_progress + [top]
            )
            raise PlacementResolutionError(
                f"Anchor cycle detected: {chain} — an anchor chain must "
                f"terminate at an element that is unplaced or placed by "
                f"Transform",
                kind="cycle",
            )
        placement = getattr(top, "placement", None)
        if placement is None:
            m = np.eye(4, dtype=np.float64)
        elif isinstance(placement, Transform):
            m = transform_matrix(placement)
        elif isinstance(placement, Anchor):
            in_progress.append(top)
            try:
                m = _anchor_world_matrix(top, placement)
            finally:
                in_progress.pop()
        else:  # pragma: no cover — PlacementType is the closed union
            raise PlacementResolutionError(
                f"{_elem_label(top)}: unknown placement type "
                f"{type(placement).__name__}",
                kind="unsupported",
            )
        resolved[key] = m
        return m

    def _anchor_world_matrix(top, anchor) -> np.ndarray:
        node = _resolve_host_node(anchor)
        if node is None:
            raise PlacementResolutionError(
                f"{_elem_label(top)}: Anchor host {anchor.host_ref!r} does "
                f"not resolve to any element",
                kind="host",
            )
        attach_to = getattr(anchor, "attach_to", "center")
        if attach_to not in ATTACH_TO_VALUES:
            raise PlacementResolutionError(
                f"{_elem_label(top)}: Anchor attach_to={attach_to!r} is "
                f"off-enum — one of {', '.join(ATTACH_TO_VALUES)}",
                kind="unsupported",
            )
        m_host = _top_matrix(node.top)
        # The host frame is the STAMP (compiler.frames), not a local AABB
        # re-derivation. ``stamp_frames`` ran at the top of this function,
        # so every geometry-bearing element carries one.
        stamped = getattr(node.elem, "_frame", None)
        if stamped is None:
            raise PlacementResolutionError(
                f"{_elem_label(top)}: Anchor host {anchor.host_ref!r} "
                f"({_elem_label(node.elem)}) has no point-bearing geometry "
                f"to derive a host frame from",
                kind="frame",
            )
        return m_host @ _anchor_local_matrix(anchor, stamped)

    out: Dict[int, np.ndarray] = {}
    for top in tops:
        placement = getattr(top, "placement", None)
        if placement is None:
            continue
        if isinstance(top, Site):
            _fail(
                f"{_elem_label(top)}: placement= on a Site container is "
                f"not supported — Site children emit as individual site "
                f"products (the container is never an IFC product); author "
                f"site geometry in world coordinates",
                kind="unsupported",
            )
            continue
        rot_issues = rotation_issues(
            getattr(placement, "rotations", None), _elem_label(top)
        )
        if rot_issues:
            for issue in rot_issues:
                _fail(issue, kind="rotation")
            continue
        try:
            out[id(top)] = _top_matrix(top)
        except PlacementResolutionError as exc:
            if errors_out is not None:
                errors_out.append(exc)
                continue
            raise
    return out


def placement_validation_issues(project) -> Tuple[List[str], List[str]]:
    """Placement checks for ``validate_project_report`` (WS1 PR-F).

    Runs the resolver in collector mode and folds its errors into the
    report — EXCEPT ``kind="host"`` failures, which the PR-D host-existence
    block already reports (keeping one error per problem). Everything the
    resolver would refuse at generation time therefore fails at compile:
    nested/Site/operand placements, anchor cycles, frame-less hosts,
    malformed rotations, off-enum attach_to.
    """
    collected: List[PlacementResolutionError] = []
    resolve_placement_matrices(project, collect_errors=collected)
    errors = [str(exc) for exc in collected if exc.kind != "host"]
    return errors, []
