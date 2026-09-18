"""
Lite-STEP Elements - Building elements (Wall, Solid, Window, etc.).

These are the primary geometry objects that the LLM generates.
Each element has an id for tracking and IFC GlobalId generation.
"""

import copy
import itertools
import logging
import re

from pydantic import BaseModel, Field, PrivateAttr, ValidationInfo, field_validator, model_validator
from typing import (
    Any, ClassVar, Dict, Optional, List, Literal, Union, Tuple, TYPE_CHECKING,
)
from uuid import uuid4

from .material import Material, LayerSet

#: The planar ifc_class values a generic ``Element(ifc_class=)`` may combine
#: with ``layers=`` (IfcMaterialLayerSet targets — planar/shell archetypes).
PLANAR_LAYER_IFC_CLASSES = frozenset({
    "IfcWall", "IfcCurtainWall", "IfcSlab", "IfcRoof",
    "IfcPlate", "IfcCovering", "IfcPavement",
})
from .primitives import Point, Point2D, _reject_float_when_strict

# Named for its one consumer rather than the module: the only thing in
# elements.py that logs is the miter derivation, and a module-wide
# `logger` here would invite unrelated chatter into a hot construction path.
_miter_logger = logging.getLogger(__name__)
from lite_step.strict import strict_int_mm_enabled

# Forward reference for type hints
if TYPE_CHECKING:
    pass  # All types defined in this module


# =============================================================================
# PLACEMENT TYPES
# =============================================================================

#: rotation axis vocabulary for placement rotations= lists.
_PLACEMENT_ROTATION_AXES = ("x", "y", "z")


def _validate_placement_rotations(v: object, info: ValidationInfo) -> object:
    """Construction gate for Transform/Anchor ``rotations=`` (WS1 PR-F).

    Checks tuple shape and axis vocabulary at construction so a bad
    rotation fails at the line that wrote it, not at compile. Angles are
    int centidegrees: under strict int-mm (RULE 2/3 posture, same as
    ``Revolve.angle``) a float angle is rejected outright; otherwise pydantic's
    ``Tuple[str, int]`` coercion still rejects non-integral floats.
    """
    if v is None:
        return v
    if not isinstance(v, (list, tuple)):
        raise ValueError(
            f"{info.field_name} must be a list of (axis, centidegrees) "
            f'tuples, e.g. rotations=[("z", 4500)] — got '
            f"{type(v).__name__}"
        )
    for i, item in enumerate(v):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"{info.field_name}[{i}] must be an (axis, centidegrees) "
                f'tuple, e.g. ("z", 4500) — got {item!r}'
            )
        axis, angle = item
        if not isinstance(axis, str) or axis not in _PLACEMENT_ROTATION_AXES:
            raise ValueError(
                f"{info.field_name}[{i}]: unknown rotation axis {axis!r} — "
                f'axes are "x", "y", "z" (lowercase)'
            )
        if strict_int_mm_enabled() and isinstance(angle, float):
            raise ValueError(
                f"{info.field_name}[{i}]: angle {angle!r} is a float — "
                f"angles are int centidegrees (DSL v1.5 RULE 3); wrap "
                f"divisions in I()"
            )
    return v


class Transform(BaseModel):
    """
    Local Placement: Position + Orientation.

    Places an element in the world: the element's geometry is authored
    LOCAL to the placement origin — local (0, 0, 0) lands at ``origin`` —
    and the placement applies on top (rigid-body semantics).

    Pinned conventions (WS1 PR-F — consumed by the placement engine,
    ``lite_step.compiler.placement``):

    - ``world = T(origin) @ R(rot_1) @ R(rot_2) @ ... @ local``
    - Rotations are about the element's LOCAL ORIGIN (0, 0, 0), not about
      any geometry center.
    - The list composes left-to-right in list order (an INTRINSIC
      sequence: each subsequent rotation is about the already-rotated
      element axes) — the same composition order as the legacy box
      ``rotations=`` field.
    - Angles are centidegrees (4500 = 45°, 9000 = 90°), right-hand rule;
      axes are "x"/"y"/"z" (lowercase, construction-enforced).

    Example:
        Extrude(
            contour=[...], thickness=200,
            placement=Transform(
                origin=Point(x=3000, y=2000, z=0),
                rotations=[("z", 4500)]
            )
        )
    """
    origin: Point                                                    # World position (mm)
    rotations: List[Tuple[str, int]] = Field(default_factory=list)   # Sequential rotations (centidegrees)

    _rotations_gate = field_validator("rotations", mode="before")(
        _validate_placement_rotations
    )

    class Config:
        extra = "forbid"
        frozen = True


class Anchor(BaseModel):
    """
    Relative Placement: Attach to a host element.

    Positions an element relative to a host element referenced by ``host=``
    — the ``name=`` of a NAMED element (DSL v1.5). Offsets use the host
    element's local coordinate frame.

    Consumed by the placement engine (WS1 PR-F,
    ``lite_step.compiler.placement`` — see its module docstring for the
    full pinned host-frame table). The anchored element's geometry is
    authored LOCAL to the attach point, with local X = along, local Y =
    across, local Z = up of the host frame. Chained anchors (host itself
    placed) resolve recursively; cycles are a compile ERROR.

    Offset semantics (relative to host, applied along the UNROTATED host
    frame axes before ``rotations=``):
    - offset_along:  Distance along the host's primary axis (mm).
                     Wall: along baseline. Beam: along axis. Slab: along X extent.
    - offset_inset:  Perpendicular to the host's primary axis, POSITIVE =
                     INTO the host, away from the weather (mm). Wall: depth
                     from the outer face. Beam: perpendicular in the
                     horizontal plane. The sign is resolved against the
                     host's stamped ``facing`` (``frames.ElementFrame``), so
                     ``inset=100`` sits a detail 100 mm behind the facade on
                     EVERY wall — unlike the raw geometric across axis, whose
                     sign is the cross-product's and therefore flips between
                     a north and a south wall. Negative is proud of the face.
    - offset_up:     Vertical offset from the attachment point (mm).
                     Wall: height from base. Slab: above/below surface.

    attach_to semantics (pinned WS1 PR-F, derived from the host's
    authoring-space AABB):
    - "start":  Anchor-line start (wall baseline / beam centerline /
                slab top-surface X-line — min along-axis end).
    - "end":    Anchor-line end (max along-axis end).
    - "center": Host's geometric (AABB) center (default).
    - "face":   Wall: center of the EXTERIOR face — the across side facing
                AWAY from the nearest containment scope (enclosing Space,
                else storey, else building; see ``compiler.frames``).
                Beam/Slab/other: top face center.

    ``rotations=`` rotate the anchored element about the (offset) attach
    point; axes "x"/"y"/"z" name the anchored frame's along/across/up.

    Example:
        Extrude(
            contour=[...], thickness=50,
            placement=Anchor(
                host="wall_south",
                attach_to="face",
                offset_along=2000,
                offset_up=1500
            )
        )
    """
    host: Optional[str] = None                                       # name= of the NAMED host element (v1.5)
    attach_to: Literal["start", "end", "center", "face"] = "center"
    offset_along: int = 0                                            # Along host's primary axis (mm)
    offset_inset: int = 0                                            # Perpendicular, POSITIVE = INTO the host (mm)
    offset_up: int = 0                                               # Vertical offset (mm)
    rotations: List[Tuple[str, int]] = Field(default_factory=list)   # Optional rotation after anchoring (centidegrees)

    _rotations_gate = field_validator("rotations", mode="before")(
        _validate_placement_rotations
    )
    # Placement offsets are geometry values — the strict int-mm gate
    # applies to user input (never to the meters-normalizer, which rebuilds
    # via model_construct inside the @suspends_strict extent).
    _strict_int_mm = field_validator(
        "offset_along", "offset_inset", "offset_up", mode="before"
    )(_reject_float_when_strict)

    @model_validator(mode="after")
    def _validate_host_reference(self):
        if self.host is None:
            raise ValueError(
                "Anchor requires host= — the name= of a NAMED element"
            )
        return self

    @property
    def host_ref(self) -> str:
        """The host reference to resolve: the ``name=`` of the host element."""
        return self.host  # type: ignore[return-value]

    class Config:
        extra = "forbid"
        frozen = True


class ChildAnchor(BaseModel):
    """Where a child sits in its PARENT's local frame — ``parent.anchor(child)``.

    ``.add(child)`` is pure containment: the child keeps the world
    coordinates it was authored in, untouched. ``.anchor(child, ...)`` says
    the opposite — author the child around a local origin and let the
    parent place it. The parent's frame
    (:class:`lite_step.compiler.frames.ElementFrame`) supplies that origin:
    local ``(0, 0, 0)`` is the parent's "lower left back" corner (along-min,
    across-min, up-min), so a wall's ``along`` runs its length, ``up`` its
    height, and ``out`` through its thickness.

    ``out`` is FACING-signed — positive is toward the OUTSIDE of the
    building, negative recesses into it — so one value behaves the same on
    a north and a south wall. Same convention as ``Anchor.offset_inset``.

    This is a private record stamped by ``.anchor()``, not a ``placement=``
    value: the two are mutually exclusive and the DSL rejects carrying both.
    Frozen — where a child sits is a fact about the call that placed it.
    """
    along: int = 0
    #: Alternative to ``along``: put the child's CENTRE here instead of its
    #: near edge. Mutually exclusive with ``along`` — a distinct name rather
    #: than a mode flag, so ``along`` never means two things depending on a
    #: second parameter. Centring an opening in a bay is the common case and
    #: was costing every author a ``- width // 2`` they could get wrong.
    along_center: Optional[int] = None
    inset: int = 0
    up: int = 0
    rotations: List[Tuple[str, int]] = Field(default_factory=list)

    _rotations_gate = field_validator("rotations", mode="before")(
        _validate_placement_rotations
    )
    # Offsets are geometry values — same strict int-mm gate as Anchor's
    # (the meters-normalizer rebuilds via model_construct, bypassing it).
    _strict_int_mm = field_validator(
        "along", "along_center", "inset", "up", mode="before"
    )(_reject_float_when_strict)

    @model_validator(mode="after")
    def _one_along(self):
        if self.along_center is not None and self.along:
            raise ValueError(
                "give along= OR along_center=, not both — they are two ways "
                "to say the same thing (near edge vs centre), and honouring "
                "both would need a rule about which wins"
            )
        return self

    class Config:
        extra = "forbid"
        frozen = True


class HalfSpace(BaseModel):
    """A clipping plane: everything on the side the ``normal`` points at is
    removed (``.clip()``).

    ``origin`` is any point on the plane (world mm — strict int-mm rides
    ``Point``'s own validators); ``normal`` is the plane normal as a
    unit-free ``(dx, dy, dz)`` tuple pointing at the REMOVED side. Maps 1:1
    to IFC: the compiler emits ``IfcBooleanClippingResult`` with an
    ``IfcHalfSpaceSolid`` second operand. Frozen value-object — a clip is a
    fact about the cut, not a mutable element.

    ``overrun`` (mm, >= 0) applies to **path trims only** — a ``Bar``/``Pipe``/
    path-``Sweep`` directrix stops at ``plane + overrun`` instead of dead on
    the plane. The boolean half-space is ALWAYS flush at the plane, whatever
    ``overrun`` says, and that asymmetry is the point: at a mitered joint the
    concrete is cut on the bisector while the reinforcement laps past it into
    the neighbour. Do not "fix" the boolean emitters to honour this field.
    """
    origin: Point
    normal: Tuple[float, float, float]
    overrun: float = 0.0

    # Geometry value — same strict int-mm gate as every other mm scalar (the
    # meters-normalizer rebuilds via model_construct and bypasses it).
    _strict_int_mm = field_validator("overrun", mode="before")(
        _reject_float_when_strict
    )

    @field_validator("overrun")
    @classmethod
    def _non_negative_overrun(cls, v):
        if v < 0:
            raise ValueError(
                f"HalfSpace.overrun must be >= 0 (got {v}) — it lets a "
                f"trimmed path REACH past the plane; pulling one short of "
                f"the plane is a shorter element, not a clip"
            )
        return v

    @field_validator("normal")
    @classmethod
    def _non_zero_normal(cls, v):
        if (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5 < 1e-12:
            raise ValueError(
                "HalfSpace.normal must be a non-zero direction (got "
                f"{v}) — it names the plane's facing, not a magnitude"
            )
        return v

    class Config:
        extra = "forbid"
        frozen = True


PlacementType = Union[None, Transform, Anchor]

#: DSL v2.1 leaf-name grammar: ``name=`` is an optional lowercase LEAF, not the
#: full identity. The canonical name (``type:leaf`` pairs) is derived by the
#: compiler; the author writes only the leaf word.
_LEAF_NAME_RE = re.compile(r"^[a-z0-9_]+$")


# =============================================================================
# BASE ELEMENT
# =============================================================================

class BimElement(BaseModel):
    """
    Base class for all BIM elements.

    Provides:
    - id: Internal bookkeeping identifier (auto ``elem_<hash>`` when
      omitted) — used only for boolean-operand tracking and ast/patcher
      fallback. NEVER emitted as an IFC Name (DSL v2.1).
    - name: Optional lowercase LEAF (``[a-z0-9_]+``, DSL v2.1). The compiler
      derives the **canonical name** from containment as ``type:leaf`` pairs
      (leaf-first) and stamps it on ``_canonical_name``; that canonical is
      THE identity — the IFC ``Name`` attribute, the LITESTEP_META manifest
      key, and the patch-mode differ key. Anonymous elements (``name=None``)
      emit ``Name=None`` (no ``id`` fallback) and belong to their nearest
      named ancestor.
    - material: Optional material (v1.5): None = sketch stage; bare string
      resolves as Material(key=...) unless it is legacy vocabulary
      ("Concrete"/"Timber"/"Steel"/"Masonry", "Void", "#rrggbb" hex) which
      keeps its pre-v1.5 behavior; Material(...) = registry product spec
      (Psets + render color + stock-validated dimensions)
    - placement: Optional placement (Transform, Anchor, or Point)
    - props: Element-level property sets (v1.5): ``props={"Pset_X":
      {"Key": value}}`` becomes one IfcPropertySet per top-level key,
      attached to the element's IFC product. Values are non-geometric —
      floats are legal here (the strict int-mm gate covers geometry only).
      Registry material Psets do NOT go here (the registry attaches them);
      props= is for element-level Psets (ExposureClass, U-values,
      FireRating). Shape is enforced at compile by
      ``validate_project_report``.
    - Derivation-by-call (v1.5): ``instance(field=value, ...)`` returns a
      copy with those fields replaced — see :meth:`__call__`.
    - Chainable boolean modifiers: .difference().union().intersection().fills()
    - extra="forbid": Catches typos in LLM scripts
    """
    #: Does placing this element in a host cut a VOID in that host?
    #:
    #: **The carve trigger is a property of the CHILD, never of the verb that
    #: placed it.** Otherwise ``.anchor()`` would mean two
    #: different things depending on the argument's type — "place this child"
    #: for a solid, "place this child AND cut a hole for it" for a Window or a
    #: Door — so the one verb an author reaches for fifty times could not be
    #: read without knowing the class of its argument. Now ``.anchor()``
    #: always places, and a Window/Door brings its void because of what IT is.
    #:
    #: Consequences worth stating, because they are what the property buys:
    #:
    #: * ``.opening(child)`` is the same relation spelled for a standalone
    #:   solid host, and it REFUSES a child that brings no void — the hole
    #:   would silently not exist (it did not, before: see ``.opening()``).
    #: * The displacement exemption table lost its "Window/Door subtrees" row.
    #:   A void-bringing child is exempt from the inferred-carve pass for the
    #:   ordinary reason every anchored child is: its position is a
    #:   RELATIONSHIP to its host. It declares its void; it does not infer one.
    #:
    #: A ``ClassVar``, not a field: it is a fact about the TYPE, so it is not
    #: authorable, not patchable, and not something a derived copy can flip.
    brings_void: ClassVar[bool] = False

    id: str = Field(default_factory=lambda: f"elem_{uuid4().hex[:8]}")
    name: Optional[str] = None
    material: Optional[Union[str, Material]] = None
    layers: Optional[LayerSet] = None
    placement: PlacementType = None
    props: Dict[str, Any] = Field(default_factory=dict)

    def __setattr__(self, name, value):
        """Assigning ``placement`` closes measurement-dependency edges.

        This hook is why the defect is observable at all. ``placement`` is a
        plain field on a model with no ``validate_assignment``, so a
        ``field_validator`` never fires on assignment and nothing else in the
        compiler sees the write — which is also why the existing stale-read
        warning (``executor._validation_report``) stays silent on a
        measurement cycle: it watches tree mutations and this is not one.

        Every other attribute takes the fast path untouched; ``__setattr__``
        runs on every field of every element and must not become a tax.
        """
        if name != "placement":
            return super().__setattr__(name, value)

        root = None
        try:
            root = self._root_project()
        except Exception:  # noqa: BLE001 - an unrooted element has no graph
            root = None

        pending = getattr(root, "_pending_reads", None) if root is not None else None
        if pending:
            drained, pending[:] = list(pending), []
            for target in drained:
                # Every drained read becomes an edge. NOT conditioned on
                # whether the value survived the author's arithmetic: reaching
                # for b.world_aabb() while positioning a IS the statement that
                # the two are related, and `* 0` does not retract it. Deciding
                # otherwise would mean evaluating author expressions to choose
                # whether to refuse — the compiler guessing at intent.
                #
                # Raises MeasurementCycleError on the closing edge, BEFORE the
                # assignment lands, so a refused model does not keep the write
                # that made it invalid.
                _add_measure_edge(root, self, target)

        return super().__setattr__(name, value)

    @model_validator(mode="after")
    def _layers_on_planar_elements_only(self) -> "BimElement":
        """``layers=`` (a planar buildup → IfcMaterialLayerSet chain) is legal
        only on the planar archetypes: Wall, Slab, Roof, and Element with a
        planar ifc_class (checked in Element's own validator). Members take
        profile-based materials (``Material(profile_mm=)``); geometry
        primitives never carry a buildup — the layers describe the ELEMENT.

        ``material=`` and ``layers=`` are mutually exclusive: per-layer
        materials come from the layers.
        """
        if self.layers is None:
            return self
        if self.material is not None:
            raise ValueError(
                "material= and layers= are mutually exclusive — a layered "
                "element's materials come from its layers")
        cls_name = type(self).__name__
        if cls_name in ("Wall", "Slab", "Roof"):
            return self
        if cls_name == "Element":
            return self  # planar ifc_class enforced in Element's validator
        raise ValueError(
            f"layers= is not valid on {cls_name} — layer buildups are for "
            f"planar elements (Wall/Slab/Roof, or Element with ifc_class in "
            f"{sorted(PLANAR_LAYER_IFC_CLASSES)}). Members (Beam/Column) take "
            f"profile-based materials: Material(key=, profile_mm=).")

    @field_validator("name", mode="after")
    @classmethod
    def _validate_leaf_name(cls, v: Optional[str]) -> Optional[str]:
        """DSL v2.1: ``name=`` is an optional lowercase LEAF (``[a-z0-9_]+``).

        The author writes only the leaf word; the compiler derives the
        canonical ``type:leaf`` path from containment. Reject uppercase, ``:``
        (a path separator, never authored), ``.``, and whitespace at
        construction — loud, echoing the value — so a bad name fails at the
        line that wrote it rather than corrupting a canonical path downstream.
        """
        if v is None:
            return None
        if not _LEAF_NAME_RE.match(v):
            raise ValueError(
                f"name={v!r} is not a valid leaf — DSL v2.1 names are a single "
                f"lowercase word matching [a-z0-9_]+ (no uppercase, ':', '.', "
                f"or whitespace). The compiler derives the canonical "
                f"'type:leaf' path from containment; write only the leaf."
            )
        return v

    @property
    def ifc_name(self) -> Optional[str]:
        """The canonical name this element emits as its IFC ``Name`` (v2.1).

        Returns the ``_canonical_name`` stamped by the top-down naming pass
        (``lite_step.compiler.naming``): ``type:leaf`` pairs leaf-first for a
        named element, ``None`` for an anonymous one. NEVER falls back to
        ``id`` — anonymous elements emit ``Name=None`` and belong to their
        nearest named ancestor. The emitter writes this into ``Name``,
        the LITESTEP_META manifest keys on it, and the patch-mode differ
        matches on it, so moving an element to another container (which
        changes its canonical) is a delete + create.
        """
        return self._canonical_name

    def __deepcopy__(self, memo=None):
        """Deep copy WITHOUT following the parent pointer.

        Mirrors ``BaseModel.__deepcopy__`` exactly but drops ``_parent``, and
        that omission is the whole point. The pointer runs upward, so a naive
        deep copy of any subtree walks child -> parent -> the parent's OTHER
        children -> the storey -> the project — i.e. copying one element
        copies the entire model. A beam with a few dozen stirrups was enough
        to wedge the compiler.

        Dropping it is safe because ``_parent`` is DERIVED: the naming walk
        re-stamps it from the tree at every pipeline entry point. A copy that
        kept it would in fact be worse — it would point into the ORIGINAL
        tree, which is the escaped-reference bug this design already fixes
        once (see ``compiler.naming.stamp_canonical_names``).

        ``deepcopy(project)`` is unaffected either way: the project enters the
        memo first, so parent pointers resolve to the copy rather than
        expanding. Only subtree copies — derivation-by-call, and every
        ``copy.deepcopy`` in the boolean-operand paths — hit the blow-up.
        """
        cls = type(self)
        new = cls.__new__(cls)
        if memo is not None:
            memo[id(self)] = new
        object.__setattr__(new, "__dict__", copy.deepcopy(self.__dict__, memo))
        object.__setattr__(new, "__pydantic_extra__",
                           copy.deepcopy(self.__pydantic_extra__, memo))
        object.__setattr__(new, "__pydantic_fields_set__",
                           copy.copy(self.__pydantic_fields_set__))
        private = getattr(self, "__pydantic_private__", None)
        if private is None:
            object.__setattr__(new, "__pydantic_private__", None)
        else:
            object.__setattr__(new, "__pydantic_private__", {
                k: (None if k == "_parent" else copy.deepcopy(v, memo))
                for k, v in private.items()
            })
        return new

    def __eq__(self, other: object) -> bool:
        """Value equality, ignoring ``_parent``.

        MUST override pydantic's. ``BaseModel.__eq__`` compares
        ``__pydantic_private__`` wholesale, so the moment ``_parent`` exists,
        equality walks UPWARD: comparing two walls compares their storey,
        which compares its element list, which compares those same two walls,
        forever. Without this override: ``RecursionError`` after 663
        frames, and a test suite that ran 1208 tests in 18 s took over nine
        minutes and 13 GB.

        Excluding the one upward pointer restores a DAG — equality still
        descends into ``_elements`` and the boolean operands and terminates at
        the leaves, exactly as it did before the back-reference existed.
        """
        if other.__class__ is not self.__class__:
            return NotImplemented
        if self.__dict__ != other.__dict__:
            return False
        if getattr(self, "__pydantic_extra__", None) != getattr(other, "__pydantic_extra__", None):
            return False
        mine = self.__pydantic_private__ or {}
        theirs = other.__pydantic_private__ or {}
        if mine.keys() != theirs.keys():
            return False
        return all(v == theirs[k] for k, v in mine.items() if k != "_parent")

    @property
    def parent(self) -> Optional["BimElement"]:
        """The container this element was attached to, or ``None``.

        Set by ``.add()``, ``.anchor()`` and ``.opening()`` at the moment of
        attachment. ``Storey.add()`` and ``Project.add()`` set it too, so a
        top-level element's parent is its :class:`~lite_step.models.project.Storey`
        (or the ``Site``'s parent is the :class:`~lite_step.models.project.Project`)
        — walking ``.parent`` from any attached element therefore reaches the
        root, which is what a world-coordinate query needs.

        NOT set for boolean operands (``.difference()`` / ``.union()`` /
        ``.intersection()`` / ``.void()`` / ``.fills()``). An operand is
        consumed into its host's shape and never renders standalone, so it is
        attachment of a different kind; conflating the two would let a world
        query silently apply a host's placement to coordinates that already
        carry it. The shared-element check in
        ``lite_step.compiler.naming.find_shared_elements`` DOES cover operands,
        because the double-stamp hazard it reports is about the naming walk,
        not about containment.
        """
        return self._parent

    #: DSL v2.1 canonical name, stamped top-down by
    #: ``lite_step.compiler.naming.stamp_canonical_names`` before generation.
    #: ``None`` until stamped and for anonymous elements. ``ifc_name`` reads
    #: it; NEVER falls back to ``id``.
    _canonical_name: Optional[str] = PrivateAttr(default=None)

    #: Authoritative local frame, stamped top-down by
    #: ``lite_step.compiler.frames.stamp_frames`` at every pipeline entry
    #: point. Carries the corner origin (local 0,0,0 = along/across/up min),
    #: the along/across/up axes and ``facing`` (+1 = exterior on +across).
    #: ``None`` until stamped and for elements bearing no geometry. Consumers
    #: READ this instead of re-deriving a run axis from an AABB. Units follow
    #: the project instance (mm pre-normalize, meters after), which is why
    #: the pass is a full recompute at each entry point.
    _frame: Optional[Any] = PrivateAttr(default=None)

    #: The container this element was ``.add()`` / ``.anchor()`` / ``.opening()``
    #: -ed into, or ``None`` while it is unattached. Set at AUTHORING time by
    #: the attach helpers below — the one piece of tree state that is known
    #: the moment it is written rather than derived by a compiler pass.
    #:
    #: Every other tree property here (``_canonical_name``, ``_frame``) is
    #: computed by a TOP-DOWN walk precisely because this back-reference did
    #: not exist; they stay top-down. This is not a second source of truth for
    #: containment — it is the handle a bounding query uses to reach the root,
    #: and the two must agree (``tests/test_parent_backref.py`` pins that).
    #:
    #: Deliberately EXCLUDED from derivation-by-call's private-state copy (see
    #: ``__call__``): a derived copy is a new, unattached element, and
    #: deep-copying a parent pointer would both lie about attachment and drag
    #: the entire ancestor tree through the copy.
    #:
    #: This is a STRONG reference, so an element tree is now a cycle
    #: (child -> parent -> child) rather than a DAG, and refcounting alone can
    #: cannot free a dropped subtree — the cyclic collector does. Measured:
    #: ~6 MB per 2 000-element project lingers until the next GC pass, and
    #: ``gc.collect()`` reclaims all of it. Lingering garbage, not a leak, and
    #: peak memory during a build is unchanged. ``weakref`` would avoid it but
    #: costs more than it buys here: it deep-copies as a reference to the
    #: ORIGINAL (so the normalize case would still need the re-stamp), and a
    #: dropped parent would turn ``.parent`` into ``None`` silently — a wrong
    #: answer where an error is wanted.
    _parent: Optional[Any] = PrivateAttr(default=None)

    #: Set by ``parent.anchor(child, ...)`` — a :class:`ChildAnchor` saying
    #: the child's geometry is authored in the PARENT's local frame. Resolved
    #: (and baked into the child's coordinates) by
    #: ``compiler.frames.resolve_child_anchors`` at generation. Mutually
    #: exclusive with ``placement=``; ``.add()`` never sets it.
    _anchor_spec: Optional[Any] = PrivateAttr(default=None)
    #: Flipped once the bake has consumed ``_anchor_spec`` — keeps the pass
    #: idempotent (a second run must not translate the child twice).
    _anchor_resolved: bool = PrivateAttr(default=False)

    #: Window/Door only: ``(along, up)`` — where an opening whose size was
    #: INFERRED has its local origin, relative to the anchor point its host
    #: placed it at. Stamped on the NORMALIZED copy by
    #: ``compiler.frames.stamp_opening_origin``, because normalization writes
    #: the resolved size into ``width``/``height`` and after that the
    #: generators cannot tell an inferred size from an authored one.
    #: ``None`` = no offset: an explicit size, or a project the pass never
    #: touched.
    _opening_origin: Optional[Any] = PrivateAttr(default=None)

    # Chainable modifier storage (not exposed as Pydantic fields).
    # NOTE: PrivateAttr names stay `_cuts`/`_adds` (internal); the DSL
    # contract is the METHOD names (.difference()/.union()). `_intersects`
    # is the new INTERSECTION operand store (DSL v8).
    _cuts: List = PrivateAttr(default_factory=list)
    _adds: List = PrivateAttr(default_factory=list)
    _intersects: List = PrivateAttr(default_factory=list)
    _fills: List = PrivateAttr(default_factory=list)
    _openings: List = PrivateAttr(default_factory=list)
    #: `.void()` operands — arbitrary solids that become IfcOpeningElements
    #: (IfcRelVoidsElement, no fill). See `.void()`.
    _voids: List = PrivateAttr(default_factory=list)
    #: Set on a `.void()` OPERAND by `.void(tool, name="leaf")`. It names the
    #: HOLE, not the tool: the tool is consumed and never emitted, so its own
    #: `name=` (which names the tool as an element) cannot be reused without
    #: making one word mean two things. `compiler.naming` turns this leaf into
    #: `_void_canonical`; `None` = the anonymous void.
    _void_leaf: Optional[str] = PrivateAttr(default=None)
    #: The canonical name the emitted IfcOpeningElement carries, stamped by
    #: `compiler.naming` from `_void_leaf` + the host's named-ancestor stack.
    _void_canonical: Optional[str] = PrivateAttr(default=None)
    #: `.clip()` half-space planes — each baked into the body CSG as an
    #: IfcBooleanClippingResult. See `.clip()`.
    _clips: List = PrivateAttr(default_factory=list)
    #: Set by `.add(carve="none")` — this subtree opts OUT of the
    #: inferred-carve pass. Kept as its own flag rather than derived from
    #: `_carve` because it is what `displacement._walk` PRUNES on, one
    #: `getattr` per element on the hot pairing loop.
    _no_carve: bool = PrivateAttr(default=False)
    #: The ``carve=`` direction the ``.add()`` that placed this element
    #: declared, or ``None`` when it was omitted. ``None`` behaves as
    #: ``"other"`` — the distinction is kept because an UNDECLARED pair is
    #: decided by tree order and flips when two ``.add()`` calls are swapped,
    #: and the carve report counts those separately so the number can trend
    #: to zero as a corpus migrates.
    _carve: Optional[str] = PrivateAttr(default=None)
    # Shared joint ids from ``miter()`` — two elements are an authored joint (and
    # must NOT auto-carve each other under displacement) iff their id sets
    # intersect. A plain int survives normalize's deepcopy verbatim (unlike
    # ``id()``), so the relation holds through compile. See ``miter()`` and
    # ``displacement._are_mitered``.
    _miter_joints: List[int] = PrivateAttr(default_factory=list)
    # Joints recorded by ``miter()`` and DERIVED later, by
    # ``compiler.composite_clip``, from the assembly as it finally stands. See
    # ``miter()`` for why the derivation is deferred and what it costs to do
    # it eagerly. Each entry holds a reference to the PARTNER element; the
    # normalizer deep-copies the whole Project in ONE call, so its memo remaps
    # that reference to the copied partner rather than leaving it pointing at
    # the pre-normalize tree.
    _pending_miters: List = PrivateAttr(default_factory=list)

    #: Catalog key of the :class:`~lite_step.models.product.Product` this
    #: element is an OCCURRENCE of, set by ``Product.occurrence(name=)``.
    #: ``None`` on every ordinary element. A plain string, so it survives
    #: normalize's deepcopy verbatim — the same reason ``_miter_joints`` is
    #: ints — and the emitter groups by it to emit one ``Ifc<Class>Type``
    #: + one ``IfcRelDefinesByType``. It is a REFERENCE (§0): it points at a
    #: catalog object that has no canonical path, and it carries no identity of
    #: its own — the occurrence's identity is still its containment path.
    _product_key: Optional[str] = PrivateAttr(default=None)
    #: The Product object itself. Carried alongside the key so two promotions
    #: that collide on one catalog key can be caught by comparing SNAPSHOTS
    #: rather than trusting the name; see
    #: ``lite_step.ifc.product_types.collect_product_groups``.
    _product: Optional[Any] = PrivateAttr(default=None)

    def __call__(self, **overrides: Any) -> "BimElement":
        """Derivation-by-call (DSL v1.5 §DERIVATION): copy with overrides.

        ``instance(field=value, ...)`` returns a copy of this element with
        those fields replaced. The copy is a **validated** construction —
        the full validator stack runs (field validators, mode validators,
        the strict int-mm gate on any Point built for the call), exactly as
        if the merged field set had been passed to the constructor.

        Semantics:

        - Unknown field → ``TypeError`` naming the field and the valid ones
          (``extra="forbid"`` posture).
        - ``id`` regenerates unless explicitly overridden: a derived copy is
          a NEW element and must not silently share the template's legacy
          identity.
        - ``name`` is carried over unless overridden — deriving from a NAMED
          template is allowed *at copy time*. Whether a rename is required is
          decided by CONTAINMENT (DSL v2.1): deriving into the SAME container
          without overriding ``name=`` yields a duplicate canonical name and
          fails at compile; deriving into a DIFFERENT container needs no
          rename (the canonical differs by the ancestor path).
        - All carried-over field values are deep-copied, and so is all
          private mutable state (``_elements`` children, ``_cuts`` /
          ``_adds`` / ``_intersects`` / ``_fills`` boolean operands,
          ``_openings``): mutating the template after deriving never leaks
          into the copy, and vice versa.
        - ``_parent`` is the ONE exception, and is reset to ``None``. A
          derived copy is a new element that has not been attached anywhere
          — and the pointer runs UPWARD, so deep-copying it would drag the
          template's whole ancestor tree (and, through it, most of the model)
          into every ``instance(...)`` call. Derivation-by-call is the idiom
          the shared-element check recommends, so it is on the hot path.
        """
        cls = type(self)
        if self._product_key is not None:
            # A Product takes no variables (roadmap §V.6), and derivation-by-call
            # is the one spelling that could smuggle them in AFTER the
            # ``.occurrence()`` gate: ``occ(width=1200)`` would keep the
            # ``_product_key`` (private state is carried over) while changing the
            # geometry, so the type relation would claim a shape the occurrence
            # does not have. Dropping the key instead would be the other silent
            # failure — a copy that quietly stops being a catalog occurrence.
            # Refuse, and name the verb that is correct.
            raise TypeError(
                f"{cls.__name__} is an occurrence of Product "
                f"{self._product_key!r} and cannot be derived from. "
                f"Derivation-by-call means an independent COPY with overrides, "
                f"but a Product takes NO variables — a copy carrying different "
                f"geometry under the same catalog key would make "
                f"IfcRelDefinesByType a lie. For another unit of the same "
                f"product use {self._product_key}.occurrence(name=...); for a "
                f"DIFFERENT unit, build and promote a second Product."
            )
        unknown = sorted(set(overrides) - set(cls.model_fields))
        if unknown:
            raise TypeError(
                f"{cls.__name__} derivation got unknown field(s) "
                f"{', '.join(repr(f) for f in unknown)} — valid fields: "
                f"{', '.join(sorted(cls.model_fields))}"
            )
        merged: Dict[str, Any] = {}
        for fname in cls.model_fields:
            if fname in overrides:
                merged[fname] = overrides[fname]
            else:
                merged[fname] = copy.deepcopy(getattr(self, fname))
        if "id" not in overrides:
            # Fresh auto id via the field's default_factory — a copy is a
            # new element; silently duplicating legacy identity would
            # collide in the ID-uniqueness compile check.
            merged.pop("id", None)
        derived = cls(**merged)
        # Deep-copy private mutable state (child lists, boolean operands) so
        # a derived copy never shares containers with its template.
        private_state = self.__pydantic_private__ or {}
        for pname, pvalue in private_state.items():
            if pname == "_parent":
                continue          # upward pointer — see the docstring
            setattr(derived, pname, copy.deepcopy(pvalue))
        derived._parent = None
        return derived

    def difference(self, *elements: "BimElement") -> "BimElement":
        """Subtract geometry from this element (boolean DIFFERENCE)."""
        self._cuts.extend(elements)
        return self

    def union(self, *elements: "BimElement") -> "BimElement":
        """Merge geometry into this element (boolean UNION)."""
        self._adds.extend(elements)
        return self

    def intersection(self, *elements: "BimElement") -> "BimElement":
        """Keep only the common volume (boolean INTERSECTION, DSL v8)."""
        self._intersects.extend(elements)
        return self

    def fills(self, *elements: "BimElement") -> "BimElement":
        """Add geometry as a fill (creates its own void)."""
        self._fills.extend(elements)
        return self

    def opening(self, element: "BimElement", *, along: int = 0,
                along_center: int = None, inset: int = 0,
                up: int = 0, rotations=None) -> "BimElement":
        """
        Open a Window/Door in this element, placed in ITS local frame.

        **ONE signature, host-frame coordinates.** This is ``.anchor()``
        spelled for a standalone solid host: the child must be one that
        BRINGS A VOID (``brings_void`` — Window/Door), and it is placed in
        this body's local frame exactly as ``wall.anchor(win, ...)`` places
        it in the wall's. A child that brings no void is refused here, and
        the refusal names ``.void(tool, name=…)`` — the verb that takes
        absolute world coordinates.

        That refusal replaces a documented form that never worked. The
        reference advertised ``n_body.opening(door.union(arch))`` as an
        absolute-coordinate spelling; measured on main, a
        non-void-bringing child either **crashed** (``TypeError`` inside
        ``frames.opening_placement``, on a standalone body) or emitted
        **nothing at all** — zero ``IfcOpeningElement``, zero
        ``IfcRelVoidsElement``, zero booleans — when the body was a Wall's,
        because a Wall's body is rendered by ``_create_wall`` and its
        ``_openings`` list is never read. A doorway that silently is not
        there is the exact failure mode the loud-failure rule exists for, so
        it is refused rather than shimmed.

        Creates an IfcOpeningElement void in this geometry, filled by the
        Window/Door. **Six hosts**: a ``Wall``, a
        ``Box`` with start/end, an ``Extrude`` with a vertical planar contour —
        each of which carries the hole on its own representation — and a
        ``Column`` / ``Beam`` / ``Element`` assembly, which carries no
        representation of its own and therefore DISTRIBUTES the hole to the
        geometry-bearing leaves it overlaps, exactly as a container's
        ``.void()`` has since. A distributed hole is N
        ``IfcOpeningElement``s and ONE ``IfcWindow``/``IfcDoor``:
        ``IfcElement.FillsVoids`` is [0:1], and a fill per leaf would
        double-count every window in every schedule. The fill sits on the
        OUTERMOST leaf — the face ``inset=`` measures from — and the others
        carry plain holes, which is what a window in a buildup physically is.

        Every other host is a compile refusal naming the remedy, because it
        emitted no ``IfcOpeningElement`` and no ``IfcWindow`` — a ``Slab`` or
        ``Roof`` (still refused: ``along=``/``inset=``/``up=`` measure from an
        outer FACE along a run axis, and a HORIZONTAL host has neither, which
        is the open half), a ``Sweep`` / ``Pipe`` / ``Bar`` / ``Revolve``
        / ``Mesh`` with no warning at all. Put the opening on the SOLID such a
        container aggregates — ``body.opening(win, along=…, up=…)`` — and it
        emits.

        **A Wall's BODY is NOT one of the three.** It emits nothing and warns
        nowhere, because
        ``_create_wall`` turns the body into the ``IfcWall``'s representation
        and ``_create_box`` — the only caller of the ``_openings`` emitter —
        never runs on it. It is refused, and the refusal names
        ``wall.anchor(win, along=…, up=…)``, which takes the identical
        scalars: the wall's opening frame is derived from that very body.
        (``body.void(tool)`` DOES reach the wall — ``displacement._voids_of``
        hoists it — because a void operand is absolute world geometry and
        brings no fill product. An opening is neither, so it is not hoisted.)

        This is the standalone-solid twin of ``wall.anchor(win, ...)`` and
        takes the SAME coordinates: ``along`` walks this body's run axis from
        its start, ``up`` is the sill height above its base, ``out`` shifts
        the fill through its depth (POSITIVE = outward). The void cuts the
        full thickness regardless of ``out``. Since v20.0.0 a Window/Door
        carries no position of its own, so the position is stated here.

        ONE opening per call — each opening has its own position, and sharing
        one ``along=`` between two would stack them. Chain for more.

        On a CONTAINER this routes to ``.anchor()``. A container with no local
        frame of its own — ``Site``, ``SpatialElement`` — REFUSES: ``along=``
        needs a face to measure from, and a place does not have one. The
        refusal names ``.add()`` (world-coordinate containment) and the host
        body's own ``.opening()``.

        Example:
            body = Box(
                start=Point(x=-5000, y=300, z=-4000),
                end=Point(x=5000, y=3000, z=-3700),
                color="wall", name="south"
            ).opening(
                Window(width=1200, height=1400, name="w0"),
                along=2000, up=900,
            )
        """
        if isinstance(element, (list, tuple)):
            raise ValueError(
                f"{type(self).__name__}.opening() takes ONE opening — each "
                f"opening has its own along=/up=, so a shared position would "
                f"stack them. Chain the calls: "
                f".opening(a, along=…).opening(b, along=…)."
            )
        if not getattr(element, "brings_void", False):
            raise ValueError(
                f"{type(self).__name__}.opening(): "
                f"{type(element).__name__} brings no void, so there is "
                f"nothing for .opening() to place. .opening() takes a child "
                f"that IS an opening (Window/Door) and puts it in this "
                f"body's LOCAL frame (along=/up=/inset=). For a hole cut by "
                f"arbitrary geometry in ABSOLUTE world coordinates, use "
                f".void(tool, name='…') — a named IfcOpeningElement that "
                f"quantities and opening schedules count — or .void(tool) "
                f"for an anonymous one, or .difference(tool) to sculpt the "
                f"shape without declaring a void."
            )
        # On a CONTAINER, ``.opening()`` is ``.anchor()`` — routed, not
        # duplicated. A container's openings are read from ``_elements``
        # (``generator._create_wall_openings_from_assembly``), so a Window
        # parked in ``_openings`` emitted no
        # IfcOpeningElement, no IfcRelVoidsElement and no IfcWindow: the
        # doorway that silently is not there, on the spelling this method's
        # own docstring calls the twin of ``wall.anchor(win, …)``. The two
        # verbs already stamp the same ``ChildAnchor`` (v20.0.0), so there was
        # never a second behaviour here to preserve — only a second list.
        #
        # The gate is a POSITIVE assertion, not a ``hasattr`` that falls
        # through. Reading ``is_container(self) and hasattr(self,
        # "anchor")`` instead lets the classes where those two disagree — ``Site``,
        # ``SpatialElement`` — took the leaf path and parked the Window in
        # ``_openings``, reintroducing through the routing condition the exact
        # failure the comment above describes. A container that cannot host a
        # local-frame child raises here; it never falls through to a list
        # nothing reads.
        from lite_step.models import taxonomy as tx

        if tx.is_container(self):
            if not tx.has_local_frame(self):
                raise TypeError(
                    f"{type(self).__name__}.opening(): a "
                    f"{type(self).__name__} has no local frame to place "
                    f"{type(element).__name__} "
                    f"{getattr(element, 'name', None)!r} in. "
                    f"along=/inset=/up= measure from an outer FACE along a "
                    f"run axis, and a {type(self).__name__} has neither — it "
                    f"says where things BELONG, not what shape they are. Its "
                    f"containment verb is .add(), which takes WORLD "
                    f"coordinates and never moves the child. An opening "
                    f"carries no position of its own, so it goes on the HOST "
                    f"that does have a face: wall.anchor(window, along=…, "
                    f"up=…) on a Wall, or body.opening(window, along=…, "
                    f"up=…) on the standalone solid you .add()ed here. "
                    f"Otherwise the child reaches a list "
                    f"nothing reads for a {type(self).__name__} — no "
                    f"IfcOpeningElement, no Ifc{type(element).__name__}, no "
                    f"warning."
                )
            return self.anchor(element, along=along,
                               along_center=along_center, inset=inset,
                               up=up, rotations=rotations)

        element._anchor_spec = ChildAnchor(
            along=along, along_center=along_center, inset=inset, up=up,
            rotations=list(rotations or [])
        )
        element._parent = self
        self._openings.append(element)
        _bump_generation()
        return self

    def void(self, *elements: "BimElement",
             name: Optional[str] = None) -> "BimElement":
        """Cut a void into this solid, in ABSOLUTE world coordinates.

        Each operand (any solid — Box/Extrude/Sweep/Revolve/Pipe/Bar, authored in
        absolute coords) becomes an ``IfcOpeningElement`` linked by
        ``IfcRelVoidsElement``: a hole that is a first-class IFC void, so
        net/gross quantities and opening schedules see it.

        ``name=`` decides whether that void carries IDENTITY. Omit it and the
        opening emits ``Name=None`` — counted, anonymous. Give it and the void
        gets a canonical name derived exactly like every other element's
        (``opening:<leaf>`` leaf-first, then this host's named ancestors), so
        a schedule row, a patch and a reviewer all have something to point at.
        A named void takes exactly ONE operand — one name cannot identify two
        holes — and the leaf obeys the ordinary ``[a-z0-9_]+`` grammar.

        The hole family is **named × filled**, and this verb owns the whole
        unfilled column:

        ======================  ==========  ========  ====================
        spelling                counted?    named?    fill?
        ======================  ==========  ========  ====================
        ``.difference(tool)``   no          no        no
        ``.void(tool)``         yes         no        no
        ``.void(tool, name=)``  yes         yes       no
        ``.opening(child)``     yes         yes       yes (Window/Door)
        ======================  ==========  ========  ====================

        ``.void(x)`` and ``.difference(x)`` carve geometrically identical
        holes in the HOST; they differ in whether the hole is a first-class IFC
        void, and — following from that — in what happens to the details
        standing IN it. A ``.void()`` hole cuts through this element's own
        anchored details (rebar, ties, framing) exactly as an ``.opening()``
        does, because matter inside a hole is not matter; a ``.difference()``
        declares no void and leaves them standing. ``carve="none"`` on a detail
        meant to stay in the reveal opts it back out. See
        ``compiler.displacement._carve_voids_through_details``.

        The operand is consumed (never renders standalone), like a boolean tool.
        """
        if name is not None:
            if len(elements) != 1:
                raise ValueError(
                    f"{type(self).__name__}.void(name={name!r}) takes exactly "
                    f"ONE operand (got {len(elements)}) — a name identifies "
                    f"one hole. Chain the calls: "
                    f".void(a, name='…').void(b, name='…'), or drop name= for "
                    f"anonymous voids."
                )
            if not _LEAF_NAME_RE.match(name):
                raise ValueError(
                    f".void(name={name!r}) is not a valid leaf — DSL v2.1 "
                    f"names are a single lowercase word matching [a-z0-9_]+ "
                    f"(no uppercase, ':', '.', or whitespace). The compiler "
                    f"derives the canonical 'type:leaf' path from the host."
                )
            elements[0]._void_leaf = name
        self._voids.extend(elements)
        return self

    def clip(self, *, origin: Point,
             normal: Tuple[float, float, float],
             overrun: float = 0) -> "BimElement":
        """Cut this element by a half-space plane — remove everything on the
        side the ``normal`` points at.

        ``origin`` is any point on the plane (absolute world coords, same
        convention as ``.void()`` operands); ``normal`` is the plane's
        facing ``(dx, dy, dz)`` pointing at the REMOVED side. The cut is
        baked into the body CSG like ``.difference()`` — an
        ``IfcBooleanClippingResult`` over an ``IfcHalfSpaceSolid``, invisible
        to schedules/quantities. Geometry queries read the AUTHORED shape, so
        a clip is not reflected in ``height_at``/``raycast`` results.

        The half-space is infinite: use it for angled trims (a mitered joint
        — see ``miter()``), a level cutoff, or lopping a corner — anywhere a
        finite cutter box is just noise. Chainable; multiple clips compose.

        **On a CONTAINER the cut reaches every geometry-bearing part under
        it**, at the end of authoring — so parts added after this call are
        cut too. A linear primitive (``Bar``/``Pipe``/path-``Sweep``) is
        TRIMMED rather than subtracted: it stays a bar for schedules, costs
        no CSG depth, and sidesteps. Everything else takes
        the half-space as the boolean it has always been.

        ``overrun`` (mm) lets a trimmed path REACH past the plane, which is
        what reinforcement does at a joint — the concrete is mitered on the
        bisector, the steel laps into the neighbour. It applies to path trims
        ONLY; the boolean stays flush. ``overrun=0`` is a real construction
        joint, so it is stated on the compile's ``clip[...]`` line rather
        than left as the thing nobody chose. Lap LENGTH is not the compiler's
        to know — it depends on bar diameter, steel grade and bond
        conditions.
        """
        self._clips.append(
            HalfSpace(origin=origin, normal=normal, overrun=overrun)
        )
        return self

    # ── geometry queries — every GEOMETRIC primitive, world space ────────
    #
    # Available on Box/Extrude/Sweep/Revolve/Pipe/Bar/Mesh — the queries
    # tessellate the element's EXACT shape (prisms via numpy, curved via the
    # ifcopenshell kernel, meshes as-is; the displacement engine's own
    # tessellator) and answer in WORLD space: placement=Transform is applied;
    # placement=Anchor is a loud error (unresolvable standalone). Containers
    # and annotations reject loudly. Queries read the AUTHORED shape — later
    # carves are not reflected.
    #
    # "World space" here means the element's OWN placement=, not its
    # ancestors'. _apply_operand_placement (compiler/displacement.py) reads
    # getattr(elem, "placement") and never walks _parent, so a primitive
    # rotated by a wrapping Element's placement= is queried un-rotated. The
    # BOUNDING family (world_aabb/obb) does walk the chain via
    # _world_matrix_for, so the two families disagree on exactly that case.
    # Deliberate for now, not an oversight — composing here means changing
    # what the boolean carve subtracts (_occupant_mesh is shared with it),
    # tripping the _RESOLVING_WORLD re-entrancy guard when compiler passes
    # re-enter, and requiring a rooted Project where these queries currently
    # work standalone. Reported as Finding 3 of the
    # authoring-queries stress test; use world_aabb()/obb() meanwhile.

    def _query_triangles(self):
        from lite_step.models import taxonomy as tx
        if tx.tessellation_kind(self) is None:
            raise ValueError(
                f"{type(self).__name__} is not a geometric primitive — "
                f"surface/ray queries need Box, Extrude, Sweep, Revolve, "
                f"Pipe, Bar or Mesh."
            )
        from lite_step.compiler.displacement import (
            DisplacementError, _occupant_mesh)
        try:
            return _occupant_mesh(self)
        except DisplacementError as e:
            raise ValueError(str(e)) from e

    def raycast(self, *, origin, direction, round: bool = True):
        """The NEAREST intersection of the ray ``origin + t*direction``
        (t ≥ 0) with this element's surface, or ``None`` on a miss.

        ``origin`` is a ``Point`` (or any (x, y, z) triple); ``direction``
        is the full 3D measurement vector, e.g. ``(0, 0, -1)`` straight
        down, ``(1, 0, 0)`` horizontally against a face, a slope normal —
        the general form of ``height_at``. An origin INSIDE the geometry
        reports the EXIT point — the first surface crossed along the
        direction (faces count regardless of orientation) — which is how
        interior clearance is measured; an origin exactly on the surface
        casting outward returns itself. Returns the intersection ``Point``
        Returns int mm — the language's unit, so the result drops straight
        into geometry. ``round=False`` returns raw float coords for
        measurement math ONLY: a float-bearing ``Point`` is built with
        ``model_construct`` and so BYPASSES the int-mm validator, which means
        it flows into ``Box(start=...)`` unchallenged even though Rule 2
        forbids it. Never feed an unrounded result back into geometry.

        Applies THIS element's own ``placement=`` and stops there — it does
        not walk ancestors. So a box rotated by a wrapping ``Element``'s
        ``placement=`` is raycast in its UNROTATED authoring position, and a
        ray aimed at where it actually stands misses. The bounding family
        (``world_aabb``/``obb``) does resolve the full chain, so the two
        disagree on exactly this case — use them to find a surface point on
        anything placed by an ancestor. Reported in the
        authoring-queries stress test as Finding 3."""
        import builtins
        from .heightfield import mesh_raycast
        from .primitives import Point as _P
        o = ((origin.x, origin.y, origin.z)
             if hasattr(origin, "x") else tuple(origin))
        verts, faces = self._query_triangles()
        hit = mesh_raycast(verts, faces, o, direction)
        if hit is None:
            return None
        if round:
            return _P.model_construct(x=int(builtins.round(hit[0])),
                                      y=int(builtins.round(hit[1])),
                                      z=int(builtins.round(hit[2])))
        return _P.model_construct(x=hit[0], y=hit[1], z=hit[2])

    def height_at(self, *, x: float, y: float, round: bool = False):
        """The TOP surface height at ``(x, y)`` — the highest intersection
        of the vertical line through the point with this element's surface
        (``raycast`` straight down, as a height). A point outside the
        element's footprint is a LOUD ``ValueError``. ``Mesh`` overrides
        this with an exact bilinear fast path on its retained heightmap
        (which also clamps at the rectangle edges). Returns a float;
        ``round=True`` → int mm.

        Applies THIS element's own ``placement=`` only, not an ancestor's —
        see ``raycast`` for why that matters and what to use instead."""
        import builtins
        from .heightfield import mesh_surface_z_at
        verts, faces = self._query_triangles()
        z = mesh_surface_z_at(verts, faces, x, y)
        if z is None:
            raise ValueError(
                f"height_at: the vertical line through ({x}, {y}) misses "
                f"this element's footprint — no surface to sample there."
            )
        return int(builtins.round(z)) if round else z

    def heightmap_at(self, *, corner_min, corner_max,
                     rows: int, cols: int,
                     round: bool = False) -> List[List[float]]:
        """Sample a ``rows × cols`` heightmap over ``corner_min``→
        ``corner_max`` from this element's top surface — the CONFORMING top
        in one call (e.g. a gravel patch following grade).

        ``rows`` = sample count along Y (south→north), ``cols`` = sample
        count along X (west→east), both ≥ 2 — the RESULT grid's resolution,
        with the standard heightmap convention ``result[r][c]`` (row 0 at
        ``corner_min.y``, col 0 at ``corner_min.x``). Heights come back as
        floats; pass ``round=True`` for int mm — required when the grid
        feeds ``Mesh(heightmap=...)`` under strict int-mm. A sample point
        off the element's footprint raises loud.

        Applies THIS element's own ``placement=`` only, not an ancestor's —
        see ``raycast`` for why that matters and what to use instead."""
        import builtins
        from .heightfield import mesh_surface_z_at
        if rows < 2 or cols < 2:
            raise ValueError(
                f"heightmap_at needs rows/cols >= 2 (got {rows}×{cols})")
        verts, faces = self._query_triangles()   # tessellate ONCE
        out: List[List[float]] = []
        for r in range(rows):
            y = corner_min.y + (corner_max.y - corner_min.y) * r / (rows - 1)
            row: List[float] = []
            for c in range(cols):
                x = corner_min.x + (corner_max.x - corner_min.x) * c / (cols - 1)
                z = mesh_surface_z_at(verts, faces, x, y)
                if z is None:
                    raise ValueError(
                        f"heightmap_at: sample point ({x}, {y}) misses this "
                        f"element's footprint."
                    )
                row.append(int(builtins.round(z)) if round else z)
            out.append(row)
        return out

    # -- bounding queries ------------------------------------------------

    def authored_aabb(self, divisor: float = 1.0) -> "Bounds":
        """This element's extent AS AUTHORED, in world-axis-aligned int mm.

        The stage is in the NAME, and it never changes meaning: these are the
        coordinates as written, before any container places the element. A
        single ``.aabb`` that meant "wherever it is now" would be convenient
        for four lines and unusable at scale — a helper 400 lines away would
        return a different stage depending on whether its caller had anchored
        yet, with nothing at either site saying which.

            b = wall.authored_aabb()
            b.min, b.max              # the two diagonal corners
            b.size                    # extents — the accessor to subtract with
            b.point_at(0, -100, 0)    # centre of one face

        Never raises for placement reasons — an unattached element has an
        authored extent, that is the whole point of the stage. It DOES raise
        for an element that has no extent to report:

        * an empty container, rather than a degenerate box at the origin;
        * a ``Window``/``Door``, which since v20.0.0 carries a SIZE and no
          position at all — where it lands is a property of its host, so ask
          the host (``wall.authored_aabb()``), which is the common case
          anyway and the one that is unambiguous.

        Reads the AUTHORED shape, pre-carve: a later ``.difference()`` or
        ``.clip()`` is not reflected, the same contract ``raycast`` and
        ``height_at`` already have.

        ``divisor`` states the unit domain THIS ELEMENT's coordinates are in —
        see :meth:`world_aabb`. Authors leave it alone.
        """
        from lite_step.models.bounds import Bounds
        from lite_step.compiler.extent import extent_is_exact

        return Bounds.from_aabb(_authored_box(self, divisor),
                                exact=extent_is_exact(self), divisor=divisor)

    def obb(self, divisor: float = 1.0) -> "Bounds":
        """Placed extent, oriented to the element's OWN authored direction.

        Same two corners as the other accessors, in world mm — what differs is
        the DIRECTIONS they are separated along, which :attr:`Bounds.axes`
        carries. Two corners describe an AABB completely and an OBB only
        partially (infinitely many oriented boxes share one diagonal), so on
        an oriented box ``max - min`` is not a width and ``.size`` is the
        extents accessor.

            b = wall.obb()
            b.point_at(0, -100, 0)     # centre of one across-face
            b.size                     # (along, across, up)
            b.rule                     # which derivation oriented it

        **Oriented to how the shape was WRITTEN, never a fitted minimal box.**
        A minimal fit swaps ``along`` and ``up`` on a near-cube and a 1 mm
        edit flips them back, so a helper reading ``point_at(100, 0, 0)``
        silently changes which face it means. The authored direction — a
        contour normal, a path, a revolve axis, a stamped wall frame — is
        stable under any edit that does not change how the shape was written.

        Where there is genuinely no single direction (a multi-segment sweep, a
        mesh, a plain Box) ``rule`` is ``"aabb"`` and this returns exactly what
        :meth:`world_aabb` returns. That is the honest answer to "which way
        does this point", and it is reported rather than guessed.

        RAISES until the element is reachable from the Project, same as
        :meth:`world_aabb`.

        ``divisor`` states the unit domain THIS ELEMENT's coordinates are in —
        see :meth:`world_aabb`. Authors leave it alone.
        """
        from lite_step.models.bounds import Bounds
        from lite_step.compiler.extent import (
            authored_axes,
            extent_is_exact,
            oriented_extent,
        )

        root = self._root_project()
        _authored_box(self, divisor)        # raise early with the same message
        _record_world_query(root, self)
        matrix = _world_matrix_for(self, root)

        axes, rule = authored_axes(self)
        if rule == "aabb":
            placed = self.world_aabb(divisor)
            return Bounds(center_mm=placed.center_mm, size_mm=placed.size_mm,
                          axes=placed.axes, exact=placed.exact, rule="aabb")

        oriented = oriented_extent(self, axes, rule, divisor)
        if oriented is None:
            return self.world_aabb(divisor)
        centre, size = oriented
        exact = extent_is_exact(self)
        if matrix is not None:
            import numpy as _np
            m = _np.asarray(matrix)
            centre = tuple(float(c) for c in (m[:3, :3] @ _np.array(centre) + m[:3, 3]))
            axes = tuple(tuple(float(c) for c in (m[:3, :3] @ _np.array(a)))
                         for a in axes)
        return Bounds.from_frame(centre, size, axes, exact=exact, rule=rule,
                                 divisor=divisor)

    def _root_project(self):
        """Walk ``.parent`` to the ``Project``, or raise saying what is missing.

        World coordinates only exist once the chain reaches the root. An
        earlier draft of the spec allowed a rootless element to answer in its
        own space — "correct under the stage rule, and surprising". It is
        worse than surprising: it hands out a world coordinate that is still
        going to change, which is the same defect as a single ``.aabb`` one
        level up.
        """
        from lite_step.models.bounds import BoundsError
        from lite_step.models.project import Project

        label = self._canonical_name or self.name or self.id
        node, seen = self, set()
        while node is not None:
            if isinstance(node, Project):
                return node
            if id(node) in seen:                    # defensive: a broken tree
                break
            seen.add(id(node))
            node = getattr(node, "_parent", None)
        raise BoundsError(
            f"{type(self).__name__} '{label}' is not reachable from a Project "
            f"— its world position is not decided yet, so there is no world "
            f"box to report. Add it (or the container holding it) to the "
            f"project first, then query: proj.add(wall) before "
            f"wall.world_aabb(). Its extent AS AUTHORED is available now via "
            f"authored_aabb()."
        )

    def world_aabb(self, divisor: float = 1.0) -> "Bounds":
        """This element's extent AS PLACED, world-axis-aligned, in int mm.

        RAISES until the element is reachable from the ``Project``. That is
        the point of naming the stage: a helper called too early fails at dev
        time instead of returning numbers that are about to change.

            wall.world_aabb()          # BoundsError — not in a storey yet
            proj.add(wall)
            wall.world_aabb()          # now defined

        For most elements this equals :meth:`authored_aabb` exactly, because
        this DSL has no local coordinates — a Box authored at x=3170 IS at
        x=3170, and storey elevation does not shift it (the generator
        subtracts the elevation from the placement and re-adds it on the
        storey). The two differ only where something MOVES the element: a
        ``placement=Transform`` on it or an ancestor, a ``placement=Anchor``,
        or a ``.anchor()`` into a host frame.

        **Side effect, and it is deliberate.** This triggers the anchor bake
        (``frames.resolve_child_anchors``), which rewrites an anchored child's
        coordinates into its parent's space. The pass is idempotent and the
        compile runs it anyway — but it means ``authored_aabb()`` on an
        ANCHORED child reports different numbers before and after the first
        world query. Pinned by a test rather than left to be discovered.

        When the placement carries a rotation the result is a BOUND: the
        axis-aligned box around a rotated box is larger than the rotated box.
        ``Bounds.exact`` says so, and :meth:`obb` is the tight answer.

        **``divisor`` states what unit THIS ELEMENT's coordinates are in**, and
        an author never touches it: a model authors in millimetres, so the
        ``1.0`` default is the whole authoring surface. It exists because a
        compiler pass may query the tree ``normalize_project_to_meters``
        returned, whose coordinates are METRES — and reading that tree at the
        default rounded every sub-metre extent to nothing while raising nothing
        (#735: a 1000x200x400 mm wall answering
        ``size=(1.0, 0.0, 0.0)``). Such a caller passes
        :data:`~lite_step.compiler.extent.MM_PER_METER` and gets the same
        millimetres back that the authoring domain gives. It is the SAME
        ``divisor`` :mod:`lite_step.compiler.extent` takes, for the same
        reason: nothing about a coordinate discloses its unit, so the domain
        cannot be derived — only stated.
        """
        from lite_step.models.bounds import Bounds
        from lite_step.compiler.extent import extent_is_exact

        root = self._root_project()
        _record_world_query(root, self)
        # Resolve FIRST, read the extent second. The bake REWRITES coordinates,
        # so reading the box before triggering it answered from pre-bake
        # numbers — invisible while only leaf primitives could be anchored
        # (the leaf is skipped pre-bake, so the read raised), but a silent
        # wrong answer for an ``.add()``ed child of an anchored CONTAINER,
        # which is readable both before and after and moves in between.
        matrix = _world_matrix_for(self, root)
        box = _authored_box(self, divisor)
        if matrix is None:
            return Bounds.from_aabb(box, exact=extent_is_exact(self),
                                    rule="world", divisor=divisor)
        placed, rotated = _transform_aabb(box, matrix)
        return Bounds.from_aabb(
            placed,
            exact=extent_is_exact(self) and not rotated,
            rule="world",
            divisor=divisor,
        )

    # -- reading a container's children ------------------------------------
    #
    # ONE accessor, inherited by every container, because the alternative is
    # what this codebase kept shipping: ``.add()`` on thirteen containers and
    # four different ways to read them back — ``get_elements()`` on seven,
    # nothing at all on Door/Site/Wall/Window, a public ``elements`` field on
    # Storey. ``_add_children`` already names the pattern ("the silent-
    # asymmetry failure this codebase keeps finding").
    #
    # The failure mode is not a crash, it is SILENCE: a caller guards with
    # ``hasattr(container, "elements")``, gets False on a container that
    # obviously has children, and skips it without a word. That cost a whole
    # subsystem in one assembled model — every ``Site``'s contents dropped,
    # including a terrain whose absence then read as a renderer bug.

    @property
    def elements(self) -> List:
        """This container's children, as a copy — safe to iterate and mutate.

        Available on every container, so ``hasattr(x, "elements")`` is a
        reliable question. A LEAF (``Box``, ``Extrude``, … — anything with no
        ``.add()``) raises :class:`AttributeError` naming itself, which also
        makes ``hasattr`` correctly False rather than a misleading ``[]``.
        """
        kids = getattr(self, "_elements", None)
        if kids is None:
            raise AttributeError(
                f"{type(self).__name__} holds no child elements — it is a "
                f"geometry leaf, not a container. Containers are the classes "
                f"with .add(): Wall, Site, Slab, Column, Beam, Roof, Space, "
                f"Door, Window, Element, SpatialElement (plus Storey)."
            )
        return list(kids)

    class Config:
        extra = "forbid"  # Catch undefined fields in LLM scripts


# =============================================================================
# CONTAINER CHILD VALIDATION (DSL v1.5)
# =============================================================================

# =============================================================================
# THE CHILD TABLE — a SEMANTICS table, keyed on the IFC CLASS
# =============================================================================
#
# WS-A §1.2. ``Wall().add(sweep)`` was a compile error while
# ``Element(ifc_class="IfcWall").add(sweep)`` was legal — **the escape hatch
# was more permissive than the sugar for the SAME IFC class.** That does not
# merely look inconsistent; it teaches authors to reach for
# ``Element(ifc_class=…)`` by default and lose the semantic container, and the
# production bridge is the receipt (everything in it is an
# ``IfcBuildingElementProxy``). The table below is keyed on the IFC class both
# spellings resolve to, so the two behave identically by construction rather
# than by two lists agreeing.
#
# ``.anchor()`` reached the same conclusion first, for the same reason, and
# threw its table away entirely (see ``_anchor_child``): a gate authors route
# around by impersonating the gated type is ceremony. ``.add()`` keeps a table
# because the two verbs make different CLAIMS — ``.add()`` says "this child's
# coordinates are already world coordinates", ``.anchor()`` says "place this
# child in my frame".
#
# **What remains here is therefore NOT a type table.** "Elements can contain
# other elements" is the rule, and every physical container now takes every
# product (``_PRODUCT_CHILDREN``) as well as every geometry primitive, exactly
# as ``.anchor()`` does. What the rows still encode is SEMANTICS — three
# statements this compiler can make about a child that ``.add()``'s claim
# contradicts:
#
# * ``IfcSpace`` takes no products. A space is AIR. "This wall bounds this
#   room" already has a relation in IFC — ``IfcRelSpaceBoundary``, which this
#   repo emits (``lite_step/ifc/space_boundaries.py``) — and a second spelling
#   of an existing relation is worse than none.
# * ``Window``/``Door`` are refused by EVERY container's ``.add()`` (see
#   :func:`_validate_container_children`). An opening carries no coordinates
#   of its own since v20.0.0, so world-coordinate containment has nothing to
#   place.
# * A ``SpatialElement`` is a PLACE, not a product, and IFC's WR31 decides
#   where one may sit — delegated to ``ifc/facilities.py::validate_nesting``,
#   which runs first and is the only defence there is.
#
# None of those three is "this type is not allowed in that type". Each is a
# claim about what the containment MEANS.


#: Every geometry primitive. A product's SHAPE is not a function of its class:
#: a wall may be swept (a curved wall), a slab revolved, a column extruded.
#: The per-class rows below therefore vary only in their NON-geometry entries.
_GEOMETRY_CHILDREN = ("Box", "Extrude", "Sweep", "Revolve", "Pipe", "Bar", "Mesh")

#: Every PRODUCT a physical container takes. "Elements can contain other
#: elements": a wall holds a column, a roof holds a beam, and the aggregation
#: that emits them (``IfcRelAggregates``, multi-level) is the same one
#: ``.anchor()`` has used since v22.3.0 — this row is the ``.add()`` half of a
#: capability the generator already had. ``SpatialElement`` is NOT here: it is
#: a spatial node, and where one may sit is WR31's call, not a row's (see
#: ``ifc/facilities.py``). ``Window``/``Door`` are not here either — refused
#: for every container in :func:`_validate_container_children`, because an
#: opening has no coordinates for ``.add()`` to keep.
_PRODUCT_CHILDREN = ("Element", "Wall", "Column", "Beam", "Slab", "Roof", "Space")

#: Whole-row overrides for the SPATIAL containers.
#:
#: ``IfcSpace`` keeps ``("Box",)``, and that is a SEMANTIC refusal rather than
#: a leftover. A space is AIR — its Box is the volume it occupies. The
#: relation IFC has for "this wall bounds this room" is ``IfcRelSpaceBoundary``
#: (emitted by ``lite_step/ifc/space_boundaries.py``), so putting the wall
#: INSIDE the space would ship a second, wrong spelling of a relation the
#: compiler already writes.
#:
#: ``IfcSite`` adds ``SpatialElement`` to the ordinary row and is otherwise
#: not special: a facility (``IfcBridge``/``IfcRoad``/…) sits under the site
#: and nowhere else, which is WR31's call rather than this row's (see
#: ``ifc/facilities.py::validate_nesting``). The row was narrow until
#: for a reason that was NOT semantic — the project-level site pass
#: in ``generate_ifc`` carried its own branch list and warned-and-dropped
#: everything else, so widening the row first would have traded a clear
#: refusal for a silent hole. That pass now ends in the same
#: ``_emit_offtable_children`` fallback every other container has, and every
#: name in this row is measured to put geometry in the file
#: (``lite_step/tests/test_site_children.py``).
_SPATIAL_CHILDREN_BY_IFC_CLASS = {
    "IfcSite": _GEOMETRY_CHILDREN + _PRODUCT_CHILDREN + ("SpatialElement",),
    "IfcSpace": ("Box",),
}

#: Children a ``SpatialElement`` accepts, by whether it is a facility or a
#: part. A facility holds PARTS (spatial decomposition); a part holds the
#: PRODUCTS that stand in it — the two halves of the tree in
#: ``infrastructure/specs/ifc43-facilities.md``. Kept beside the table above
#: rather than folded into it because the key is not the IFC class but which
#: SIDE of the spatial/physical boundary the children fall on.
_FACILITY_CHILDREN = ("SpatialElement",)
#: Spelled through ``_PRODUCT_CHILDREN`` rather than restated: a part contains
#: the products, and "the products" is now one list. Written out by hand it was
#: the same eight names in a different order — the second implementation that
#: agrees *usually*.
_FACILITY_PART_CHILDREN = (
    _GEOMETRY_CHILDREN + _PRODUCT_CHILDREN + ("SpatialElement",)
)


def container_ifc_class(container: "BimElement") -> str:
    """The IFC class a container resolves to — the child table's key.

    ``Element`` states it outright; every sugar container IS its class with
    the ``Ifc`` prefix (``Wall`` -> ``IfcWall``), which is the exact inverse of
    ``compiler.naming.type_segment``. Deriving it beats a second dict: a new
    sugar container gets its row for free, and cannot get a DIFFERENT one from
    its own escape-hatch spelling.

    Takes an element INSTANCE. Handed a CLASS, a naive fallback answers
    ``"Ifc" + type(Wall).__name__`` == ``"IfcModelMetaclass"`` — pydantic's
    metaclass, a class no schema has ever defined — with no error, because the
    fallback reads ``type(x)`` and a class's type is its metaclass. Every
    caller passes an instance, so this is one convenience call away from
    being wrong silently. Classes are answered properly, and the one case a class genuinely cannot answer (``Element``
    states its ``ifc_class`` per INSTANCE) raises.
    """
    if isinstance(container, type):
        if "ifc_class" in getattr(container, "model_fields", {}):
            raise TypeError(
                f"container_ifc_class({container.__name__}) is undefined for "
                f"the CLASS — {container.__name__} states its ifc_class per "
                f"instance. Pass an instance.")
        return "Ifc" + container.__name__
    explicit = getattr(container, "ifc_class", None)
    if explicit:
        return explicit
    return "Ifc" + type(container).__name__


def allowed_children_for(container: "BimElement") -> tuple:
    """The child types ``container.add()`` accepts, by IFC class.

    Every physical container takes every geometry primitive AND every product
    — "elements can contain other elements". The two rows that differ say
    something (see the table's header comment): ``IfcSpace`` refuses products
    because ``IfcRelSpaceBoundary`` is the relation for what an author would
    mean, and a ``SpatialElement`` answers by which side of the
    spatial/physical boundary it is on rather than by its class, because there
    are ten spatial classes and exactly two answers.
    """
    ifc_class = container_ifc_class(container)
    # Checked before the IFC-class table — see the docstring.
    if type(container).__name__ == "SpatialElement":
        from lite_step.ifc import facilities as _fac
        if _fac.is_facility(ifc_class):
            return _FACILITY_CHILDREN
        return _FACILITY_PART_CHILDREN
    override = _SPATIAL_CHILDREN_BY_IFC_CLASS.get(ifc_class)
    if override is not None:
        return override
    return _GEOMETRY_CHILDREN + _PRODUCT_CHILDREN


def reject_opening_children(container, elements: tuple) -> None:
    """Refuse a ``Window``/``Door`` handed to ANY ``.add()``, at the authoring
    line.

    Since v20.0.0 an opening carries a size and nothing else — the HOST states
    where it goes — so ``.add()``'s whole claim ("this child's coordinates are
    already world coordinates") is one an opening cannot satisfy: there are no
    coordinates to keep. ``executor._opening_placement_errors`` already reports
    this at compile and STAYS as the backstop for a hand-built tree; this is
    the early check that names the author's line, the same pairing
    ``frames.check_anchor_rotation`` and :func:`_refuse_containment_cycle` use.

    **Public, and a separate function, because "every container" has to mean
    every container**. The refusal lived inline in
    :func:`_validate_container_children`, and ``Storey.add`` / ``Project.add``
    are the two containers that do not route through it — so
    ``storey.add(Window(...))`` was accepted at the line while the message
    said an opening "cannot be ADDED", full stop. Restating the message at the
    second call site is the shape; one function called from both is not.

    ``container`` is anything with a class name — a ``BimElement``, a
    ``Storey`` or a ``Project`` — because all this reads off it is that name.
    Raises on the FIRST opening, before anything is appended.
    """
    from lite_step.models import taxonomy as _tx    # runtime: taxonomy imports us
    for child in elements:
        # Read off the CHILD (``brings_void``), not an isinstance list — the
        # WS-A §1.1 rule the opening machinery already follows.
        if _tx.brings_void(child):
            verb = type(child).__name__.lower()
            child_label = (getattr(child, "name", None)
                           or getattr(child, "id", None) or repr(child))
            raise ValueError(
                f"{type(container).__name__}.add(): "
                f"{type(child).__name__} '{child_label}' cannot be ADDED — an "
                f"opening carries a size and no position of its own (DSL "
                f"v20.0.0 removed offset=/sill=), and .add() is WORLD-"
                f"coordinate containment, so there is nothing for it to keep. "
                f"The HOST places it: wall.anchor({verb}, along=…, up=…) on a "
                f"Wall, or body.opening({verb}, along=…, up=…) on a standalone "
                f"solid. Both stamp the same anchor and cut the void."
            )


def _validate_container_children(container: "BimElement", elements: tuple) -> None:
    """Construction gate for container ``.add()`` children.

    The table is :func:`allowed_children_for` — keyed on the container's IFC
    CLASS, so ``Wall`` and ``Element(ifc_class="IfcWall")`` accept exactly the
    same children (§1.2). Rejected HERE, at the ``.add()`` call site, so the
    traceback points at the line that wrote it. The generator translates; it
    does not validate (its per-container skip warnings are defensive dead code
    behind this gate). Validating at the model layer covers emission in one
    place.

    Matching is by exact type name (``type(child).__name__``) — string names,
    not class references, because the containers (Wall, Site, ...) are defined
    before the primitives they accept (Box, Extrude, ...), so class-body
    references would NameError.

    **The refusal gives advice that works.** "A container inside a
    container" is not refused at all: every physical
    container takes every product, and the aggregation that emits them shipped
    with ``.anchor()``. What is left is the three SEMANTIC refusals in the
    table's header comment, and each names its own fix rather than the generic
    "use another verb":

    * a product into an ``IfcSpace`` -> ``IfcRelSpaceBoundary``;
    * a ``Window``/``Door`` into anything -> ``.anchor()`` / ``.opening()``;
    * a spatial node in the wrong place -> ``facilities.validate_nesting``.

    Raises on the FIRST invalid child, before anything is appended — a
    failed ``.add()`` never partially mutates the container.
    """
    # Spatial nesting FIRST, and unchanged. A part added to a Site is refused
    # by the table below too, but with "Site does not accept SpatialElement",
    # which does not tell the author the part needs a FACILITY. This gate names
    # the fix. It is also the ONLY defence: ifcopenshell.validate does not
    # report WR31 (measured — tests/test_ifc43_conformance.py), so a facility
    # emitted into a containment relation produces a file that passes the
    # conformance gate and is invisible in the inspector. Nothing in this PR
    # widens it: WR31 is the schema's call, not a row's.
    for child in elements:
        if type(child).__name__ == "SpatialElement":
            from lite_step.ifc import facilities as _fac
            # The container's class ALWAYS, not only when it is spatial: a
            # part added to a Site must be refused too, and Site is the most
            # likely place to put one by mistake.
            _fac.validate_nesting(container_ifc_class(container), child.ifc_class)

    # An OPENING, refused for every container — including the two that do not
    # reach this function, which is why it is its own callable.
    reject_opening_children(container, elements)

    allowed = allowed_children_for(container)
    for child in elements:
        child_type = type(child).__name__
        if child_type not in allowed:
            # Construction-time label (runs at .add(), before the canonical
            # naming pass), so use the local leaf/id — ifc_name is not
            # stamped yet.
            child_label = getattr(child, "name", None) or getattr(child, "id", None) or repr(child)
            container_label = getattr(container, "name", None) or getattr(container, "id", None)
            ifc_class = container_ifc_class(container)
            if ifc_class == "IfcSpace":
                # The one refusal a wider table would have SHIPPED rather than
                # fixed. A Space is AIR; "this wall bounds this room" is
                # IfcRelSpaceBoundary, which this compiler already emits from
                # the geometry (lite_step/ifc/space_boundaries.py). Putting the
                # wall inside the space would be a second, wrong spelling of a
                # relation that already exists.
                raise ValueError(
                    f"Space '{container_label}': invalid child {child_type} "
                    f"'{child_label}' — a Space is AIR, and .add() into one "
                    f"takes only the Box that is the volume it occupies. A "
                    f"physical element does not go INSIDE a space; it BOUNDS "
                    f"one, and IFC's relation for that is IfcRelSpaceBoundary "
                    f"— which this compiler already derives from the geometry "
                    f"(lite_step/ifc/space_boundaries.py). Add the "
                    f"{child_type} to the storey as a peer of the Space."
                )
            # The remedy is chosen, not asserted. A blanket "or .anchor() it" is
                # wrong here: BOTH containers that can reach this
            # line — Site and SpatialElement — have no ``.anchor()`` method at
            # all, so it was advice that fails on the next line. That is the
            # probe finding repeating itself one message later.
            if child_type == "SpatialElement":
                # Reachable when ``validate_nesting`` had nothing to say —
                # e.g. a FACILITY into a Wall, where neither WR31 clause
                # fires. ``.anchor()`` would take it (that verb has no table),
                # which is precisely why it must not be offered: a facility is
                # a PLACE, and the tree it belongs in is Site -> facility ->
                # part -> products.
                remedy = (
                    f"A spatial node is not a product and does not go inside "
                    f"one. The chain is Site -> facility -> part -> products "
                    f"(infrastructure/specs/ifc43-facilities.md); add the "
                    f"{child.ifc_class} to the Site.")
            elif hasattr(container, "anchor"):
                remedy = (".add() is WORLD-coordinate containment; to place "
                          "this child in the container's own frame instead, "
                          ".anchor() it (any element type into any container).")
            elif ifc_class == "IfcSite":
                # Every geometry primitive and every product is ON this row
                # since, so what reaches here is a type that is not a
                # product at all (a Storey, a Site, a GuideLine). The remedy is
                # therefore about the TREE, not about a gap in the emitter —
                # the site pass does not have one.
                remedy = (
                    f"A {type(container).__name__} has no .anchor() — its "
                    f"children are placed in world coordinates. A Site holds "
                    f"the geometry and products that stand ON it plus the "
                    f"facilities (SpatialElement) that subdivide it; a "
                    f"{child_type} is neither. A building's contents go in a "
                    f"Storey, which the Project holds beside the Site.")
            else:
                remedy = (
                    f"A {type(container).__name__} has no .anchor() — it is a "
                    f"spatial node, and its children are decided by the "
                    f"spatial/physical boundary rather than by a frame.")
            raise ValueError(
                f"{type(container).__name__} '{container_label}' "
                f"({ifc_class}): invalid child "
                f"{child_type} '{child_label}' — .add() into a "
                f"{ifc_class} takes: {', '.join(allowed)}. {remedy}"
            )


def _authored_box(elem, divisor: float = 1.0):
    """The authored extent tuple, or a ``BoundsError`` explaining the absence.

    Shared by ``authored_aabb`` and ``world_aabb`` so the two cannot drift on
    which elements have an extent or on what they say when one does not.

    Answers in the CALLER's units — ``divisor`` is threaded into
    ``element_extent`` so the two frozen-millimetre Material facts
    (``profile_mm`` / ``thickness_mm``) convert with everything else.
    """
    from lite_step.models.bounds import BoundsError
    from lite_step.models import taxonomy as tx
    from lite_step.compiler.extent import element_extent

    label = elem._canonical_name or elem.name or elem.id
    if tx.is_opening(elem):
        raise BoundsError(
            f"{type(elem).__name__} '{label}' has no extent of its own — an "
            f"opening carries a size (width={getattr(elem, 'width', None)}, "
            f"height={getattr(elem, 'height', None)}) and no position; the "
            f"HOST decides where it goes. Ask the host: "
            f"wall.authored_aabb().point_at(...) — which is the common case "
            f"and the unambiguous one."
        )
    box = element_extent(elem, divisor)
    if box is None:
        # An UNBAKED ANCHOR TARGET reaches here looking exactly like an empty
        # container, and it is not: it has geometry, it just cannot be seen
        # yet. `element_extent` reads `iter_geometry_points`, which skips any
        # element still carrying an unresolved `_anchor_spec` (frames.py) —
        # so the element filters itself out, `element_extent` returns None,
        # and this raise fires BEFORE `world_aabb` reaches the bake its own
        # docstring promises as a side effect. Chicken-and-egg: the query that
        # would resolve the anchor is blocked by the anchor being unresolved.
        #
        # Saying "holds no geometry" about a Box with start/end is simply
        # false, and it sent a reader looking for a bug in their geometry.
        # The way out is real and non-obvious: query any already-resolvable
        # element in the tree — the bake walks the WHOLE project and is
        # idempotent — and this element answers on the next call.
        if getattr(elem, "_anchor_spec", None) is not None and not getattr(
                elem, "_anchor_resolved", False):
            raise BoundsError(
                f"{type(elem).__name__} '{label}' is anchored but not yet "
                f"resolved, so its position is not known — it does have "
                f"geometry. Nothing has triggered the anchor bake yet, and "
                f"this query cannot trigger its own. Query an already-placed "
                f"element first (its host, or anything with non-anchored "
                f"geometry): that bakes every pending anchor in the project, "
                f"and this element answers straight after. Compiling does the "
                f"same."
            )
        raise BoundsError(
            f"{type(elem).__name__} '{label}' has no extent to report — it "
            f"holds no geometry. An empty container has no bounding box, and "
            f"returning a degenerate one at the origin would place whatever "
            f"you author against it at the origin too."
        )
    return box


#: Bumped by every containment mutation. A world query records the value it
#: resolved at; anything still holding a value from an older generation read a
#: coordinate that has since moved. There is no declarative dependency graph
#: here — `.anchor(along=...)` takes an int, evaluated once — so a true cycle
#: cannot form and this is the only staleness that can bite:
#:
#:     p = wall.obb().point_at(0, -100, 0)   # read now
#:     storey.add(wall)                      # ...wall moves
#:     shelf_x = p.x                         # baked from a stale read
#:
#: Root-reachability kills the common case (that first read now raises). What
#: remains is an ancestor moving AFTER a legitimate read, which cannot be
#: fixed retroactively — so it is made loud at compile instead. Two ints.
_MUTATION_GENERATION = 0


def _bump_generation() -> int:
    global _MUTATION_GENERATION
    _MUTATION_GENERATION += 1
    return _MUTATION_GENERATION


def _record_world_query(root, elem) -> None:
    """Note that ``elem`` was resolved at the current generation.

    Also queues ``elem`` as a PENDING READ. The next placement assignment
    drains the queue into measurement-dependency edges — see
    ``BimElement.__setattr__``. Both records are written here because this is
    the single point every world-resolving query passes through.
    """
    label = elem._canonical_name or elem.name or elem.id
    queries = getattr(root, "_world_queries", None)
    if queries is None:
        return
    queries.append((f"{type(elem).__name__} '{label}'", _MUTATION_GENERATION))

    pending = getattr(root, "_pending_reads", None)
    if pending is not None:
        pending.append(elem)


def _measure_label(elem) -> str:
    """``mesh:mesh3:storey:ground`` — the tree-report spelling.

    ``_canonical_name`` already carries the type and the container path, so
    it is used as-is; prefixing the class again yields ``mesh:mesh:mesh3``.
    Only the bare ``name``/``id`` fallback needs the type prepended.
    """
    canon = elem._canonical_name
    if canon:
        return canon
    return f"{type(elem).__name__.lower()}:{elem.name or elem.id}"


def _measure_path(edges, nodes, start, target):
    """The path ``start → … → target`` through ``edges``, or ``None``.

    Iterative, not recursive: these graphs follow the authored tree and a
    ``RecursionError`` raised from inside the compiler would report the wrong
    problem entirely.
    """
    stack = [(start, [start])]
    seen = set()
    while stack:
        node, path = stack.pop()
        if node == target:
            return path
        if node in seen:
            continue
        seen.add(node)
        for nxt in edges.get(node, ()):  # noqa: SIM118 - dict of sets
            stack.append((nxt, path + [nxt]))
    return None


def _add_measure_edge(root, source, target) -> None:
    """Record ``source`` was placed from a measurement of ``target``.

    Refuses on the edge that CLOSES a loop, so the traceback points at the
    statement that completed it. The check runs before the edge is stored:
    a refused model must not be left carrying the edge that killed it.
    """
    if source is target:
        pass  # self-measurement is a cycle of length 1; fall through to check
    edges = getattr(root, "_measure_edges", None)
    nodes = getattr(root, "_measure_nodes", None)
    if edges is None or nodes is None:
        return
    s_id, t_id = source.id, target.id

    # Does target already reach source? Then source→target closes the loop.
    back = _measure_path(edges, nodes, t_id, s_id)
    if back is not None or s_id == t_id:
        from lite_step.compiler.walkguard import MeasurementCycleError
        ring = [s_id] + (back if back is not None else [s_id])
        nodes.setdefault(s_id, source)
        nodes.setdefault(t_id, target)
        pretty = " -> ".join(_measure_label(nodes[n]) for n in ring if n in nodes)
        raise MeasurementCycleError(
            f"placement depends on a measurement of itself:\n"
            f"  {pretty}\n"
            f"Each of these is positioned from a measurement of the next, so "
            f"the model describes more than one design and statement order "
            f"decides which you get. That is a choice the compiler will not "
            f"make for you.\n"
            f"Anchor ONE of them to a fixed coordinate and measure outward "
            f"from it."
        )

    nodes.setdefault(s_id, source)
    nodes.setdefault(t_id, target)
    edges.setdefault(s_id, set()).add(t_id)


#: Re-entrancy guard for world resolution. A world query bakes anchors and
#: resolves placements; if anything INSIDE that bake reached back through the
#: public accessor the bake would call itself and the author would see a
#: ``RecursionError`` from deep inside the compiler. Internal passes must use
#: ``frames.*`` / ``placement.*`` directly — this turns the mistake into a
#: sentence naming the rule.
_RESOLVING_WORLD = False


def _opening_ancestor(elem):
    """The nearest ``Window``/``Door`` ANCESTOR of ``elem``, or ``None``.

    Shaped after :meth:`BimElement._root_project` — the same ``_parent`` walk
    with the same defensive cycle guard, because a broken tree must produce a
    plain answer rather than a hang.

    Starts at ``elem``'s parent, never at ``elem``: an opening queried
    directly is a documented raise (:func:`_authored_box`), and this walk
    exists to place its CHILDREN, which is a different question.
    """
    from lite_step.models import taxonomy as tx

    node, seen = getattr(elem, "_parent", None), set()
    while node is not None:
        if id(node) in seen:                    # defensive: a broken tree
            return None
        seen.add(id(node))
        if tx.is_opening(node):
            return node
        node = getattr(node, "_parent", None)
    return None


def _opening_host_body(opening):
    """The solid whose frame an opening is placed in, or ``None``.

    ``body.opening(win, …)`` stamps the BODY as the parent, so the host is
    already the solid. ``wall.anchor(win, …)`` stamps the container, whose
    body is its first ``Box``/``Extrude`` child — the same "first prism
    child" rule ``generator._create_wall`` uses to pick the body it derives
    the opening frame from, and the reason a container cannot be passed to
    ``frames.opening_host_frame`` directly (that reads ``start``/``end`` and
    would answer ``None`` for a Wall).
    """
    from lite_step.models import taxonomy as tx

    host = getattr(opening, "_parent", None)
    if host is None:
        return None
    if tx.is_prism(host):
        return host
    for child in (getattr(host, "_elements", None) or []):
        if tx.is_prism(child):
            return child
    return None


def _opening_child_matrix(opening):
    """4x4 opening-local -> authoring-space matrix for ``opening``'s children.

    An opening is the one anchored child ``frames._bake_walk`` skips
    ENTIRELY, spec and subtree both — deliberately, because its
    ``_anchor_spec`` is consumed as SCALARS the void is built from and its
    children are opening-LOCAL Z-up. That skip is correct and stays. What was
    missing is this: nothing ever composed the opening's own frame back on
    for a QUERY, so ``world_aabb()`` on a window's frame box answered in
    opening-local coordinates — silently, and identically for two windows
    anchored metres apart.

    The transform is rigid: a translation to the opening's lower-left corner
    on the host's outer face, plus the host's run angle about Z. Both come
    out of :func:`frames.opening_placement`, which is the SAME call the emitter makes — so the number this returns and the number the compiler
    emits cannot drift.

    Raises ``BoundsError`` when the host bears no usable opening frame. That
    is the honest answer: the compiler would reject the same model, and
    returning ``None`` here would put the caller straight back into reading
    opening-local coordinates as if they were world ones.
    """
    import numpy as _np

    from lite_step.compiler import frames
    from lite_step.models.bounds import BoundsError

    label = opening._canonical_name or opening.name or opening.id
    body = _opening_host_body(opening)
    frame = None
    if body is not None:
        # Authoring millimetres here, so no snapper and no mm->m divisor —
        # the shared derivation is unit-agnostic and each caller states its
        # own units.
        frame = frames.opening_host_frame(body, snap=None, divisor=1.0)
    if frame is None:
        raise BoundsError(
            f"{type(opening).__name__} '{label}' has no usable host frame, so "
            f"nothing inside it has a world position yet — an opening is "
            f"placed by its host, and that host needs a box body (start=/"
            f"end=) or a vertical planar contour body with a thickness. "
            f"Fix the host body and the query answers; compiling would "
            f"reject the same model."
        )

    place = frames.opening_placement(opening, frame)
    cos_a, sin_a = place.cos_a, place.sin_a
    return _np.array([
        [cos_a, -sin_a, 0.0, place.corner[0]],
        [sin_a, cos_a, 0.0, place.corner[1]],
        [0.0, 0.0, 1.0, place.corner[2]],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def _world_matrix_for(elem, root):
    """4x4 world matrix for ``elem``, or ``None`` when nothing moves it.

    ``placement.resolve_placement_matrices`` keys by ``id()`` and covers
    TOP-LEVEL storey elements and site tops. A nested child is already in its
    parent's authoring coordinates once the anchor bake has run, so it
    inherits the nearest ancestor that has a matrix — which is reachable now
    that ``.parent`` exists, and was the reason §A and §B could not be
    separated.

    ONE subtree is not in its parent's coordinates: everything under a
    ``Window``/``Door``. The bake skips openings on purpose, so their
    children still hold opening-LOCAL coordinates and need the opening's own
    frame composed on FIRST, then any ancestor placement on top of that.
    """
    global _RESOLVING_WORLD
    from lite_step.models.bounds import BoundsError

    if _RESOLVING_WORLD:
        raise BoundsError(
            "world query re-entered during resolution — internal compiler "
            "passes must use frames.* / placement.* directly rather than the "
            "public world_aabb()/obb() accessors."
        )
    _RESOLVING_WORLD = True
    try:
        from lite_step.compiler.frames import resolve_child_anchors, stamp_frames
        from lite_step.compiler.naming import stamp_canonical_names
        from lite_step.compiler.placement import resolve_placement_matrices

        # The SAME sequence generate_ifc runs, in the same order, because the
        # order is load-bearing: the anchor bake resolves a child against its
        # parent's FRAME, so frames must be stamped first or the bake raises
        # "no point-bearing geometry"; and the bake MOVES geometry, so frames
        # are recomputed after it. Every pass here is idempotent.
        stamp_canonical_names(root)
        stamp_frames(root)
        resolve_child_anchors(root)
        stamp_frames(root)
        matrices = resolve_placement_matrices(root)
    finally:
        _RESOLVING_WORLD = False

    # Composed BEFORE the ancestor walk, and the walk then starts at the
    # opening: an ancestor placement moves the whole host, so it multiplies
    # ONTO the opening's local frame, never the other way round.
    opening = _opening_ancestor(elem)
    local = _opening_child_matrix(opening) if opening is not None else None

    node = opening if opening is not None else elem
    seen: set = set()                       # defensive: a broken tree
    while node is not None:
        m = matrices.get(id(node))
        if m is not None:
            if local is None:
                return m
            import numpy as _np

            return _np.asarray(m) @ local
        if id(node) in seen:
            break
        seen.add(id(node))
        node = getattr(node, "_parent", None)
    return local


def _transform_aabb(box, matrix):
    """``(world_box, rotated)`` — the AABB of ``box``'s eight corners under
    ``matrix``, and whether the matrix rotates.

    All eight corners, not just min/max: under a rotation the transformed
    min corner is not the min of the transformed box, and using two corners
    would silently return a box that does not contain the solid.

    ``rotated`` is returned rather than inferred by the caller because the
    axis-aligned box around a rotated box is strictly larger — the answer is
    a bound, and ``Bounds.exact`` has to say so.
    """
    import numpy as _np

    (x0, y0, z0), (x1, y1, z1) = box
    corners = _np.array([[x, y, z, 1.0]
                         for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    placed = (matrix @ corners.T).T[:, :3]
    lo = placed.min(axis=0)
    hi = placed.max(axis=0)
    rot = _np.asarray(matrix)[:3, :3]
    rotated = not _np.allclose(rot, _np.eye(3), atol=1e-9)
    return ((float(lo[0]), float(lo[1]), float(lo[2])),
            (float(hi[0]), float(hi[1]), float(hi[2]))), rotated


def _refuse_containment_cycle(container: "BimElement", child, verb: str) -> None:
    """Refuse ``container.<verb>(child)`` when ``container`` is already inside
    ``child`` — AT THE AUTHORING LINE.

    ``w1.anchor(w2); w2.anchor(w1)`` and ``a.add(b); b.add(a)`` are two
    ordinary authoring lines each, and both were accepted. What they build is
    an element that contains itself, which has no canonical name, no frame and
    no finite extent, and which took the compile down with a bare
    ``RecursionError`` raised inside ``compiler.naming._walk`` — naming no
    element, at a stack depth thousands of frames from the line that caused it.

    **This is IN ADDITION to the validator, not instead of it.** Same shape as
    :func:`frames.check_anchor_rotation`, which moved a refusal to the
    authoring line and kept the bake-time one "as the backstop rather than
    being replaced". Here the reason it cannot be complete is ``_parent``:
    the stamp is LAST-WRITE-WINS (see :func:`_add_children`), so an object
    attached in two places carries a pointer to only one of them, and this
    walk therefore climbs one of the several chains that reach ``container``.
    A cycle closed through any of the others is invisible here.
    ``compiler.naming.find_containment_cycles`` descends the real tree and is
    authoritative; this exists so the ordinary two-line case fails on the
    second line, with both elements named.
    """
    node, seen = container, set()
    while node is not None:
        if node is child:
            child_label = getattr(child, "name", None) or getattr(child, "id", None)
            container_label = (getattr(container, "name", None)
                               or getattr(container, "id", None))
            raise ValueError(
                f"{type(container).__name__}.{verb}(): "
                f"{type(child).__name__} '{child_label}' already contains "
                f"{type(container).__name__} '{container_label}', so this "
                f"would make each of them contain the other. A containment "
                f"cycle has no canonical name, no frame and no extent — every "
                f"one of those is derived by descending containment. Detach "
                f"one of the two attachments: a container may hold a child, "
                f"or be held by it, never both."
            )
        if id(node) in seen:                    # defensive: an existing cycle
            return
        seen.add(id(node))
        node = getattr(node, "_parent", None)


#: The three directions ``.add(carve=)`` accepts, and what each one means for
#: the overlaps this add creates. Role names rather than positions
#: (``left``/``right``): at the call site only one operand is visible, so
#: "left of what?" needs a convention lookup while "self" and "other" do not.
CARVE_DIRECTIONS = ("other", "self", "none")


def _stamp_carve(elem, direction: str) -> None:
    """Record ``direction`` on ``elem`` — the element the author NAMED, and
    only that one.

    The scope is still the whole SUBTREE (an exempt assembly whose own parts
    kept carving would let the bricks inside a bracket eat the wall the bracket
    hangs off) — but the subtree is reached by READING through ancestors at
    displacement time, not by copying the direction down at ``.add()`` time.
    Copying broke twice, both measured:

    * a part added to the assembly AFTER this ``.add()`` line missed the copy
      and silently reverted to the default direction;
    * an enclosing ``.add(carve=)`` overwrote a nested declaration the author
      had written explicitly one level deeper.

    Both are the same shape as the container-``miter()`` bug: a subtree-scoped
    statement recorded at one node and read at another. See
    ``compiler.displacement._carve_direction``, which is the reader, and
    ``_joint_ids`` beside it.
    """
    elem._carve = direction
    if direction == "none":
        # ``_no_carve`` is a separate flag from ``_carve`` because it is what
        # ``displacement._walk`` PRUNES on — one ``getattr`` per element on the
        # hot pairing loop, rather than an ancestor walk that would answer the
        # same question more slowly at every node.
        #
        # Set on this element ALONE for the same reason the direction is:
        # ``displacement._walk`` prunes at the first ``_no_carve`` it meets and
        # never descends, so the flag on the named element already governs
        # everything under it — including parts added later, which a copy could
        # not have reached.
        elem._no_carve = True


def _add_children(container: "BimElement", elements: tuple,
                  carve: str = None) -> None:
    """Shared body of every container's ``.add()``: validate, stamp the
    parent back-reference, append.

    ``carve`` names who loses material in the overlaps THIS add creates:
    ``"other"`` (the default — the element already there), ``"self"`` (the
    element being added yields instead), ``"none"`` (neither). Omitted leaves
    it undeclared, which behaves as ``"other"`` and is counted separately in
    the carve report, because an undeclared pair is decided by tree order and
    flips if two ``.add()`` calls are swapped.

    **An argument, not a field or a verb.** A field on ``BimElement`` would be
    a PRODUCT fact — authored, validated, normalised, patched, round-tripped —
    and this is a statement about one compiler PASS. The argument also fixes
    two things a VERB on the element cannot: ``Product(source)`` snapshots at
    construction, so exempting the source afterwards was a silent no-op (it
    cost a wrong measurement — five ground plates kept carving), and the same
    source added in two places can be exempt in one and not the other, which
    is inexpressible while the flag rides on the element.

    One helper rather than three lines repeated per container, so that
    ``_parent`` cannot be set on eight of the nine ``.add()`` methods — the
    silent-asymmetry failure this codebase keeps finding (``Wall`` and
    ``Site`` still have no ``get_elements()`` for exactly that reason).

    The stamp is deliberately LAST-WRITE-WINS rather than a hard refusal of a
    second parent. Making it raise is a breaking change that stops the skills
    corpus compiling, so it follows the shim-first sequence:
    ``compiler.naming.find_shared_elements`` reports the condition as a
    compile ``warning:`` now, and the refusal lands once the corpus is clean.
    """
    if carve is not None and carve not in CARVE_DIRECTIONS:
        raise ValueError(
            f"add(carve={carve!r}) is not a direction — use "
            f"{', '.join(repr(d) for d in CARVE_DIRECTIONS)}. "
            f"'other' is the default (the element already there loses "
            f"material), 'self' inverts it, 'none' exempts both."
        )
    _validate_container_children(container, elements)
    for child in elements:
        _refuse_containment_cycle(container, child, "add")
    for child in elements:
        child._parent = container
    if carve is not None:
        for child in elements:
            _stamp_carve(child, carve)
    container._elements.extend(elements)
    _bump_generation()


def _anchor_child(
    container: "BimElement", child,
    along: int, inset: int, up: int, rotations,
    along_center=None, carve: str = None,
) -> "BimElement":
    """Shared body of every container's ``.anchor()``.

    **There is no type table here, deliberately.** ``.anchor()`` accepts ANY
    element type into ANY container. A gate here would prevent
    nothing: ``Element(ifc_class=…)`` is anchorable and holds every geometry type, so an author refused by a narrow
    table simply wrapped the child in an ``Element`` impersonating the type
    they wanted — paying a wrapper and losing the semantic class to satisfy
    a check that let the same geometry through one line later. The workaround
    shipped: gable ends authored as ``Element(ifc_class="IfcWall")`` rather
    than ``Wall``, because ``Wall``'s table was too narrow for what it hosts.
    A gate authors route around by impersonating the gated type is ceremony,
    so it is gone — and the emitters were taught to RENDER every pairing
    instead, which is the guarantee that actually matters
    (``lite_step/tests/test_anchor_matrix.py`` walks the whole product).

    ``.add()`` keeps its table. The two verbs make different claims: ``.add()``
    says "this child's coordinates are already world coordinates",
    ``.anchor()`` says "place this child in my frame". The narrower ``.add()``
    table is what catches an author who reached for the wrong verb.

    The ONE thing refused here is not a type at all: it is the spatial/physical
    BOUNDARY, asked in BOTH directions. Spatial nodes are PLACES and physical
    elements are PRODUCTS; ``.anchor()`` refuses a crossing either way, and
    that is one question with two answers (the reasoning
    :func:`allowed_children_for` already gives for why it does not enumerate
    the ten spatial classes), not a table of element types.

    * **A place does not go inside a product**. A
      ``SpatialElement`` has no body, so there is nothing for ``along=`` /
      ``inset=`` / ``up=`` to measure against, and it is not a product, so it
      does not belong inside one at any coordinates. Before that refusal the
      call was accepted and the emitter dropped the node: ``success=True``,
      no ``IFCBRIDGE`` in the file, ``ifcopenshell.validate`` at 0 errors
      (the shape).
    * **A product does not go inside AIR**. ``.add()`` has always
      refused a product into a ``Space``, and its message gives a reason about
      what a Space IS rather than about coordinates — so it applies to
      ``.anchor()`` word for word, and only ``.add()`` enforced it. Measured on
      ``main`` @ ``a7f41e9``: ``space.anchor(wall)`` was accepted at the line,
      compiled clean, and wrote ``IfcRelAggregates(IfcSpace -> IfcWall)`` —
      a decomposition claiming the wall is a PART of the room's air. IFC's
      relation for "this wall bounds this room" is ``IfcRelSpaceBoundary``,
      which this repo already derives from the geometry
      (``lite_step/ifc/space_boundaries.py``), so the emitted relation was not
      merely unhelpful, it was false.

    Both read a predicate, not a class list: :func:`taxonomy.is_spatial` for
    the container and :func:`taxonomy.is_physical` for the child. Two pairings
    stay legal on purpose, and neither crosses the boundary:

    * ``space.anchor(<geometry primitive>)`` — a ``Box``/``Sweep``/``Mesh`` is
      a SHAPE, not a building element, and shapes are how every container in
      this DSL carries geometry. (NOTE, because the obvious reason is the
      wrong one: a Space's own volume comes from ``.add(Box)``, which becomes
      the ``IfcSpace``'s representation. An ``.anchor()``-ed Box is a separate
      ``IfcBuildingElementProxy`` child, not part of the room's volume.)
    * ``space.anchor(space)`` — place inside place. IFC decomposes an
      ``IfcSpace`` into ``IfcSpace`` through ``IfcRelAggregates``, which is
      exactly what this emits, so it is the one aggregation from a Space that
      says something true.

    Every PHYSICAL child still goes into every PHYSICAL container, which is
    what ``test_anchor_matrix.py`` walks.

    Refuses the two ways a child could end up with two placements, checks that
    every descendant the rotation will reach can actually take it, stamps the
    :class:`ChildAnchor`, and appends to the ordinary ``_elements`` list — so
    naming, normalization, displacement and the emitter sees the child
    exactly as they see an ``.add()``-ed one, with no walk changes.

    Raises at the ``.anchor()`` call site (like ``.add()``), so the
    traceback points at the authoring line.
    """
    child_label = getattr(child, "name", None) or getattr(child, "id", None)
    # A SPATIAL NODE, refused FIRST — before the placement questions, because
    # "this is not a placeable child at all" precedes "which placement does it
    # carry". Read off the class name, the same one-line spelling
    # ``_validate_container_children`` uses, and NOT a list of accepted types.
    if type(child).__name__ == "SpatialElement":
        from lite_step.ifc import facilities as _fac
        # The nesting rule speaks first, through the SAME function ``.add()``
        # calls — one rule, two verbs, one implementation. Where the author's
        # mistake is the spatial tree (a facility PART with no facility over
        # it), its message names the chain precisely, and the two verbs must
        # not disagree about a tree. It is also the only defence there is:
        # ifcopenshell.validate does not report WR31.
        _fac.validate_nesting(container_ifc_class(container), child.ifc_class)
        # …and the local-frame refusal covers what the nesting rule has nothing
        # to say about — a FACILITY into a product fires neither WR31 clause,
        # which is exactly the pairing measured being dropped.
        raise ValueError(
            f"{type(container).__name__}.anchor(): child SpatialElement("
            f"{child.ifc_class}) '{child_label}' cannot be anchored — a "
            f"spatial node is a PLACE, not a product. .anchor() puts a child "
            f"in this {type(container).__name__}'s LOCAL FRAME: along= walks "
            f"its run axis, inset= its depth, up= its base — and a "
            f"spatial node has no body for any of those to measure against, "
            f"so the call would place nothing. The chain is Site -> facility "
            f"-> part -> products (infrastructure/specs/ifc43-facilities.md): "
            f".add() the {child.ifc_class} to the Site, and .add() the "
            f"products it holds into its parts. Otherwise the node is "
                f"dropped — success=True, no "
            f"{child.ifc_class.upper()} in the file, and the IFC validator "
            f"silent at 0 errors, because it does not check WHERE rules."
        )
    # The SAME boundary, the OTHER way round — a PRODUCT into a
    # SPATIAL container. Also before the placement questions, and for the same
    # reason: whether this child belongs here at all precedes which placement
    # it carries. Two predicates, no type table.
    from lite_step.models import taxonomy as _tx    # runtime: taxonomy imports us
    if _tx.is_spatial(container) and _tx.is_physical(child):
        container_label = (getattr(container, "name", None)
                           or getattr(container, "id", None))
        if _tx.brings_void(child):
            # An opening is physical too, but "add it to the storey" is the
            # wrong advice for one: it carries no coordinates, and a Space is
            # not voidable, so nothing here could host it. Measured pre-fix:
            # ``space.anchor(window)`` compiled clean and emitted NO IfcWindow
            # at all — a silent drop, not a wrong relation.
            remedy = (
                f"An opening is placed by the body it cuts, and a Space is "
                f"AIR — there is nothing here to cut. Anchor it to the WALL "
                f"that bounds this room: "
                f"wall.anchor({type(child).__name__.lower()}, along=…, up=…). "
                f"The IfcRelSpaceBoundary for the filling is derived from "
                f"that, not authored here. Accepting it here would emit no "
                f"{container_ifc_class(child).upper()} at all — a silent drop.")
        else:
            remedy = (
                f"A physical element does not go INSIDE a space; it BOUNDS "
                f"one, and IFC's relation for that is IfcRelSpaceBoundary — "
                f"which this compiler already derives from the geometry "
                f"(lite_step/ifc/space_boundaries.py). Add the "
                f"{type(child).__name__} to the storey as a PEER of the "
                f"{type(container).__name__}, in world coordinates. "
                f"Aggregating it instead would write IfcRelAggregates("
                f"{container_ifc_class(container)} -> "
                f"{container_ifc_class(child)}) — a decomposition saying the "
                f"child is a PART of the room's air.")
        raise ValueError(
            f"{type(container).__name__}.anchor(): child "
            f"{type(child).__name__} '{child_label}' cannot be anchored into "
            f"{type(container).__name__} '{container_label}' — a "
            f"{type(container).__name__} is AIR, and .anchor() aggregates the "
            f"child INTO it. {remedy} This is the same refusal .add() gives, "
            f"for the same reason: it is a statement about what a "
            f"{type(container).__name__} IS, not about how the child's "
            f"coordinates were expressed, so the verb cannot change it. "
            f"Geometry primitives and sub-Spaces still "
            f"anchor here."
        )
    if getattr(child, "placement", None) is not None:
        raise ValueError(
            f"{type(container).__name__}.anchor(): child "
            f"{type(child).__name__} '{child_label}' already carries "
            f"placement= — an anchored child is placed by its parent's "
            f"frame, so the two are mutually exclusive. Use .add() to keep "
            f"the placement, or drop placement= to anchor it."
        )
    if getattr(child, "_anchor_spec", None) is not None:
        # Deliberately does NOT say "another container": anchoring the same
        # object twice into the SAME container hits this too, and naming the
        # wrong cause sends an author looking for a second call site that
        # does not exist.
        #
        # The remedy depends on what the child IS, so state the one that
        # applies rather than a menu. Placing "the same thing" twice is
        # always a second OBJECT — `_canonical_name`, `_frame` and
        # `_anchor_spec` are single-valued per object, so one object in two
        # places is undefined, not merely ambiguous (see
        # ``naming.find_shared_elements``).
        product_key = getattr(child, "_product_key", None)
        remedy = (
            f"For another unit of the same product call "
            f"{product_key}.occurrence(name=...) again — one occurrence object "
            f"per placement."
            if product_key else
            f"For a second placement derive a copy: "
            f"{child_label or 'child'}(name='...') returns a NEW element with "
            f"the same geometry."
        )
        raise ValueError(
            f"{type(container).__name__}.anchor(): child "
            f"{type(child).__name__} '{child_label}' is already anchored — a "
            f"child has exactly one parent frame, so one object cannot stand "
            f"in two places. {remedy}"
        )
    _refuse_containment_cycle(container, child, "anchor")
    # Rotation legality, HERE rather than at compile depth. A leaf's type must
    # not be what decides whether the assembly you just wrote is legal — and
    # if it does, the error belongs on the line that asked for the rotation,
    # not three levels down inside the bake. Local import: ``frames`` reaches
    # back into this module for its own type checks.
    from lite_step.compiler.frames import check_anchor_rotation

    check_anchor_rotation(container, child, rotations)

    child._anchor_spec = ChildAnchor(
        along=along, along_center=along_center, inset=inset, up=up,
        rotations=list(rotations or [])
    )
    child._parent = container
    if carve is not None:
        if carve not in CARVE_DIRECTIONS:
            raise ValueError(
                f"anchor(carve={carve!r}) is not a direction — use "
                f"{', '.join(repr(d) for d in CARVE_DIRECTIONS)}. On an "
                f"anchored child the one that does work is 'none', which "
                f"keeps an opening or void from notching a detail meant to "
                f"stand in the reveal."
            )
        _stamp_carve(child, carve)
    container._elements.append(child)
    _bump_generation()
    return container


# =============================================================================
# OPENING & FILL ELEMENTS
# =============================================================================

class Door(BimElement):
    """
    Door assembly. Positioned on its host by ``.anchor()``.

    Creates IfcOpeningElement (void) + IfcDoor (fill) in IFC. A Door is a
    SEMANTIC opening — never a boolean carve — whichever way it is placed.

    Placement is the DSL's one placement idiom (v20.0.0): the host puts it
    where it goes. There is no world-coordinate form — a Door carries no
    coordinates of its own, only its opening size.

        wall.anchor(door, along=1000, up=0)      # along the baseline, above the base
        body.opening(door, along=1000, up=0)     # same, on a standalone solid

    ``along`` walks the host's run axis from its start, ``up`` is the sill
    height above the host's base, and ``out`` shifts the leaf through the
    host's depth (POSITIVE = outward; negative recesses it). The VOID always
    cuts the full thickness regardless of ``out``.

    Can contain child primitives (leaf, frame) via add(); their coordinates
    are opening-LOCAL (origin at the opening's lower-left corner, Z up):
        door = Door(width=900, height=2100, name="front")
        door.add(Extrude(contour=[...], thickness=45, type="sketch", name="leaf"))
        wall.anchor(door, along=1000)
    """
    #: A Door IS an opening — placing it cuts the void. See
    #: ``BimElement.brings_void``: the trigger is the child, not the verb.
    brings_void: ClassVar[bool] = True
    id: str = Field(default_factory=lambda: f"door_{uuid4().hex[:8]}")
    #: Rough-opening size (mm). OPTIONAL — omit and the compiler infers it
    #: from the children's in-plane extents (opening-local X = along the wall,
    #: Z = up; depth is ignored, since projecting through the wall is normal).
    #: State it explicitly when the joinery deliberately overflows the hole —
    #: a projecting sill, a proud handle — because then the children are NOT
    #: the opening. With no children and no explicit size there is nothing to
    #: infer from, and that is a compile error.
    width: Optional[int] = None
    height: Optional[int] = None
    swing: str = "left"   # "left", "right", "double"
    style: str = "hinged"  # "hinged", "sliding", "folding"
    _elements: List = PrivateAttr(default_factory=list)

    def add(self, *elements, carve: str = None) -> "Door":
        """Add joinery in OPENING-LOCAL coordinates — origin at the opening's
        lower-left corner, Z up, identical on every facade. A Door carries no
        world coordinates of its own; the host places the whole unit, so a
        world-coordinate child lands that far from the hole. The allowed-children
        table is keyed on this container's IFC class (see
        :func:`allowed_children_for`), so the `Element(ifc_class=…)` spelling
        accepts exactly the same children."""
        _add_children(self, elements, carve=carve)
        return self


class Window(BimElement):
    """
    Window assembly. Positioned on its host by ``.anchor()``.

    Creates IfcOpeningElement (void) + IfcWindow (fill) in IFC. A Window is a
    SEMANTIC opening — never a boolean carve — whichever way it is placed.

    Placement is the DSL's one placement idiom (v20.0.0): the host puts it
    where it goes. There is no world-coordinate form — a Window carries no
    coordinates of its own, only its opening size.

        wall.anchor(win, along=2000, up=900)     # along the baseline, sill height
        body.opening(win, along=2000, up=900)    # same, on a standalone solid

    ``along`` walks the host's run axis from its start, ``up`` is the sill
    height above the host's base, and ``out`` shifts the sash through the
    host's depth (POSITIVE = outward; negative recesses it). The VOID always
    cuts the full thickness regardless of ``out``.

    Can contain child primitives (glass + frame) via add(); their coordinates
    are opening-LOCAL (origin at the opening's lower-left corner, Z up):
        win = Window(width=1200, height=1400, name="w0")
        win.add(Extrude(contour=[...], thickness=6, color="glass", name="glass"))
        wall.anchor(win, along=2000, up=900)
    """
    #: A Window IS an opening — placing it cuts the void. See
    #: ``BimElement.brings_void``: the trigger is the child, not the verb.
    brings_void: ClassVar[bool] = True
    id: str = Field(default_factory=lambda: f"window_{uuid4().hex[:8]}")
    #: Rough-opening size (mm). OPTIONAL — omit and the compiler infers it
    #: from the children's in-plane extents (opening-local X = along the wall,
    #: Z = up; depth is ignored, since projecting through the wall is normal).
    #: State it explicitly when the joinery deliberately overflows the hole —
    #: a projecting sill, a proud handle — because then the children are NOT
    #: the opening. With no children and no explicit size there is nothing to
    #: infer from, and that is a compile error.
    width: Optional[int] = None
    height: Optional[int] = None
    style: str = "fixed"  # "fixed", "casement", "sliding", "awning"
    panes: int = 1
    _elements: List = PrivateAttr(default_factory=list)

    def add(self, *elements, carve: str = None) -> "Window":
        """Add joinery in OPENING-LOCAL coordinates — origin at the opening's
        lower-left corner, Z up, identical on every facade. A Window carries no
        world coordinates of its own; the host places the whole unit, so a
        world-coordinate child lands that far from the hole. The allowed-children
        table is keyed on this container's IFC class (see
        :func:`allowed_children_for`), so the `Element(ifc_class=…)` spelling
        accepts exactly the same children."""
        _add_children(self, elements, carve=carve)
        return self


# =============================================================================
# PRIMARY BUILDING ELEMENTS
# =============================================================================

class Wall(BimElement):
    """
    Wall assembly container.

    Contains an explicit Solid body + Window/Door children.
    The body Solid defines the wall geometry.
    Window/Door children define openings with positioning metadata.

    IFC mapping:
    - Body Solid → IfcWall representation
    - Window/Door children → IfcOpeningElement + IfcRelVoidsElement + fill

    Example:
        wall = Wall(name="south")
        wall.add(
            Box(
                start=Point(x=-6000, y=300, z=-4000),
                end=Point(x=6000, y=3000, z=-3700),
                type="wall", name="body"
            ),
        )
        # Openings are ANCHORED — .add() is world coordinates, and a
        # Window/Door has none of its own (v20.0.0).
        wall.anchor(Window(width=1200, height=1400, name="w0"),
                    along=2000, up=900)
        wall.anchor(Door(width=900, height=2100, name="d0"), along=5000)
        proj.add(wall)
    """
    _elements: List = PrivateAttr(default_factory=list)

    def add(self, *elements, carve: str = None) -> "Wall":
        """Add the body geometry, or a Window/Door fill. in WORLD coordinates — the table is keyed on this
        container's IFC class (see :func:`allowed_children_for`), so the
        `Element(ifc_class=…)` spelling accepts exactly the same children."""
        _add_children(self, elements, carve=carve)
        return self

    def anchor(self, child, *, along: int = 0, along_center: int = None,
               inset: int = 0, up: int = 0, rotations=None,
               carve: str = None) -> "Wall":
        """Place ``child`` in THIS element's local frame.

        ``along`` runs the length, ``up`` the height, and ``out`` through the
        thickness with POSITIVE = outward from the building. Local (0, 0, 0)
        is this element's lower-left-back corner. Contrast ``.add()``, which
        is pure containment and never moves the child.

        ``child`` may be ANY element type — geometry primitive or container.
        See :func:`_anchor_child` for why there is no table.

        ``carve`` is the same argument ``.add()`` takes and means the same
        thing. **Anchoring already grants half of it**: an anchored child is
        out of the inferred-carve pairing by construction, because its position
        is a relationship to its host rather than an accidental overlap — which
        is the whole of what ``carve="none"`` says on an ``.add()``. Stating it
        here extends that to the other carve an anchored child can still take,
        the OPENING/VOID one: a hole removes matter regardless of how the
        matter got there, and ``carve="none"`` is the author saying THIS matter
        is meant to be in it — a sill or a lintel standing in the reveal.

        So ``"other"`` and ``"self"`` are accepted and INERT: the pairing they
        name does not happen for an anchored child at all, so there is no
        direction to state. Refused instead, one argument would mean different
        things depending on which verb you wrote it on, and the author would
        have to remember which subset each accepts.
        """
        return _anchor_child(self, child,
                             along, inset, up, rotations, along_center,
                             carve=carve)


class Site(BimElement):
    """
    SPATIAL container — everything inside is assigned to IfcSite (never a
    storey). The three domains stay separate: geometry children (Box/Extrude/
    Mesh) say what SHAPE something is (GEOMETRIC); an ``Element`` wrapper says
    what it IS (PHYSICAL — IfcGeographicElement TERRAIN, IfcEarthworksFill, …);
    the Site says where it BELONGS (SPATIAL).

    Canonical terrain (the carve host):

        site = Site(name="site")
        terrain = Element(ifc_class="IfcGeographicElement",
                          predefined_type="TERRAIN", name="terrain")
        terrain.add(Mesh(heightmap=..., depth=..., corner_min=..., corner_max=...))
        site.add(terrain)
        proj.add(site)

    Site groundwork (gravel beds, pavements, kerbs, …) is another semantic
    ``Element`` sibling — e.g. ``Element(ifc_class="IfcEarthworksFill")``
    with a Mesh child. It is NOT a carve host.

    ``material=`` is REJECTED at construction: IfcSite is an
    IfcSpatialStructureElement — a place, not a product — and materials
    belong on products. Silently accepting-and-dropping it was the
    silent-degradation class this codebase hunts; put the material on the
    semantic ``Element`` wrapper instead.

    (The former legacy routing — ``mesh_type="terrain"`` children becoming
    IfcGeographicElement(TERRAIN) — is removed; ``mesh_type="terrain"`` is a
    construction error naming the wrapper form. Bare Box/Extrude children
    are still terrain-reassigned for now.)
    """
    _elements: List = PrivateAttr(default_factory=list)

    @field_validator("material")
    @classmethod
    def _no_material_on_site(cls, v: object) -> object:
        if v is not None:
            raise ValueError(
                "Site does not take material= — IfcSite is a spatial element "
                "(a place, not a product), and the generator would silently "
                "drop it. Put material= on the semantic Element wrapper "
                'instead, e.g. Element(ifc_class="IfcEarthworksFill", '
                'material="Gravel_16-32").'
            )
        return v

    def add(self, *elements, carve: str = None) -> "Site":
        """Add geometry (Box/Extrude/Mesh) or semantic ``Element`` wrappers.

        IfcSite keeps a table of its own: a spatial structure element is a
        PLACE, not a product, and is not reachable through
        ``Element(ifc_class=…)`` at all — so §1.2's escape-hatch asymmetry
        cannot arise here and there is nothing to reconcile."""
        _add_children(self, elements, carve=carve)
        return self


class Column(BimElement):
    """
    Column assembly container for vertical structural elements.

    Maps to IfcColumn. Use for posts, pillars, vertical supports that are
    visible/exposed. Contains Solid/Sweep/Revolve children for geometry
    definition.

    The Column container:
    - Creates IfcColumn as the parent element
    - Child Solid/Sweep/Revolve elements define the geometry
    - Child appearance follows the spec §6 precedence chain (see the
      generator's ``_apply_container_color``); material=None children
      render sketch-stage

    Example - Carport post:
        col = Column(id="post_sw")
        col.add(Box(
            start=Point(x=-2875, y=0, z=-1375),
            end=Point(x=-2725, y=2400, z=-1225),
            type="sketch", color="column", id="body"
        ))
        proj.add(col)

    The compiler transforms to IFC convention (Z-up) during generation.
    """
    _elements: List = PrivateAttr(default_factory=list)

    def add(self, *elements, carve: str = None) -> "Column":
        """Add child geometry in WORLD coordinates — the table is keyed on this
        container's IFC class (see :func:`allowed_children_for`), so the
        `Element(ifc_class=…)` spelling accepts exactly the same children."""
        _add_children(self, elements, carve=carve)
        return self

    def anchor(self, child, *, along: int = 0, along_center: int = None,
               inset: int = 0, up: int = 0, rotations=None,
               carve: str = None) -> "Column":
        """Place ``child`` in THIS element's local frame.

        ``along`` runs the length, ``up`` the height, and ``out`` through the
        thickness with POSITIVE = outward from the building. Local (0, 0, 0)
        is this element's lower-left-back corner. Contrast ``.add()``, which
        is pure containment and never moves the child.

        ``child`` may be ANY element type — geometry primitive or container.
        See :func:`_anchor_child` for why there is no table.

        ``carve`` is the same argument ``.add()`` takes and means the same
        thing. **Anchoring already grants half of it**: an anchored child is
        out of the inferred-carve pairing by construction, because its position
        is a relationship to its host rather than an accidental overlap — which
        is the whole of what ``carve="none"`` says on an ``.add()``. Stating it
        here extends that to the other carve an anchored child can still take,
        the OPENING/VOID one: a hole removes matter regardless of how the
        matter got there, and ``carve="none"`` is the author saying THIS matter
        is meant to be in it — a sill or a lintel standing in the reveal.

        So ``"other"`` and ``"self"`` are accepted and INERT: the pairing they
        name does not happen for an anchored child at all, so there is no
        direction to state. Refused instead, one argument would mean different
        things depending on which verb you wrote it on, and the author would
        have to remember which subset each accepts.
        """
        return _anchor_child(self, child,
                             along, inset, up, rotations, along_center,
                             carve=carve)



class Beam(BimElement):
    """
    Beam assembly container for horizontal structural elements.

    Maps to IfcBeam. Use for exposed beams, lintels, headers, purlins.
    Contains Solid/Sweep children for geometry definition plus Bar
    children for reinforcement.

    The Beam container:
    - Creates IfcBeam as the parent element
    - Child Solid/Sweep elements define the geometry; Bar children
      aggregate as reinforcement (they keep their own appearance)
    - Child appearance follows the spec §6 precedence chain (see the
      generator's ``_apply_container_color``); material=None children
      render sketch-stage

    Example - Carport beam:
        beam = Beam(id="beam_south")
        beam.add(Box(
            start=Point(x=-2800, y=2400, z=-1350),
            end=Point(x=2800, y=2600, z=-1250),
            type="sketch", color="beam", id="body"
        ))
        proj.add(beam)

    The compiler transforms to IFC convention (Z-up) during generation.
    """
    _elements: List = PrivateAttr(default_factory=list)

    def add(self, *elements, carve: str = None) -> "Beam":
        """Add child geometry in WORLD coordinates — the table is keyed on this
        container's IFC class (see :func:`allowed_children_for`), so the
        `Element(ifc_class=…)` spelling accepts exactly the same children."""
        _add_children(self, elements, carve=carve)
        return self

    def anchor(self, child, *, along: int = 0, along_center: int = None,
               inset: int = 0, up: int = 0, rotations=None,
               carve: str = None) -> "Beam":
        """Place ``child`` in THIS element's local frame.

        ``along`` runs the length, ``up`` the height, and ``out`` through the
        thickness with POSITIVE = outward from the building. Local (0, 0, 0)
        is this element's lower-left-back corner. Contrast ``.add()``, which
        is pure containment and never moves the child.

        ``child`` may be ANY element type — geometry primitive or container.
        See :func:`_anchor_child` for why there is no table.

        ``carve`` is the same argument ``.add()`` takes and means the same
        thing. **Anchoring already grants half of it**: an anchored child is
        out of the inferred-carve pairing by construction, because its position
        is a relationship to its host rather than an accidental overlap — which
        is the whole of what ``carve="none"`` says on an ``.add()``. Stating it
        here extends that to the other carve an anchored child can still take,
        the OPENING/VOID one: a hole removes matter regardless of how the
        matter got there, and ``carve="none"`` is the author saying THIS matter
        is meant to be in it — a sill or a lintel standing in the reveal.

        So ``"other"`` and ``"self"`` are accepted and INERT: the pairing they
        name does not happen for an anchored child at all, so there is no
        direction to state. Refused instead, one argument would mean different
        things depending on which verb you wrote it on, and the author would
        have to remember which subset each accepts.
        """
        return _anchor_child(self, child,
                             along, inset, up, rotations, along_center,
                             carve=carve)



class Slab(BimElement):
    """
    Structural horizontal slab container - foundation, inter-storey slabs,
    ceilings, and thin sheet floors (DSL v8).

    Maps to IfcSlab. Like Roof/Column/Beam, Slab is a CONTAINER: it holds
    Box/Extrude geometry children (reversing the pre-v8 geometry-direct
    form where start/end/contour lived on the Slab itself). Children
    auto-color "slab".

    - Foundation / structural slab: box or contour child, material="Concrete_..."
    - Thin sheet floor: contour Extrude child with Material(key=, thickness_mm=)

    Example - Foundation:
        foundation = Slab(name="foundation")
        foundation.add(Box(
            start=Point(x=-6500, y=-5000, z=-300),
            end=Point(x=6500, y=5000, z=0),
            material="Concrete_C30-37",
        ))
        proj.add(foundation)

    The compiler transforms to IFC convention (Z-up) during generation.
    """
    _elements: List = PrivateAttr(default_factory=list)

    def add(self, *elements, carve: str = None) -> "Slab":
        """Add child geometry in WORLD coordinates — the table is keyed on this
        container's IFC class (see :func:`allowed_children_for`), so the
        `Element(ifc_class=…)` spelling accepts exactly the same children."""
        _add_children(self, elements, carve=carve)
        return self

    def anchor(self, child, *, along: int = 0, along_center: int = None,
               inset: int = 0, up: int = 0, rotations=None,
               carve: str = None) -> "Slab":
        """Place ``child`` in THIS element's local frame.

        ``along`` runs the length, ``up`` the height, and ``out`` through the
        thickness with POSITIVE = outward from the building. Local (0, 0, 0)
        is this element's lower-left-back corner. Contrast ``.add()``, which
        is pure containment and never moves the child.

        ``child`` may be ANY element type — geometry primitive or container.
        See :func:`_anchor_child` for why there is no table.

        ``carve`` is the same argument ``.add()`` takes and means the same
        thing. **Anchoring already grants half of it**: an anchored child is
        out of the inferred-carve pairing by construction, because its position
        is a relationship to its host rather than an accidental overlap — which
        is the whole of what ``carve="none"`` says on an ``.add()``. Stating it
        here extends that to the other carve an anchored child can still take,
        the OPENING/VOID one: a hole removes matter regardless of how the
        matter got there, and ``carve="none"`` is the author saying THIS matter
        is meant to be in it — a sill or a lintel standing in the reveal.

        So ``"other"`` and ``"self"`` are accepted and INERT: the pairing they
        name does not happen for an anchored child at all, so there is no
        direction to state. Refused instead, one argument would mean different
        things depending on which verb you wrote it on, and the author would
        have to remember which subset each accepts.
        """
        return _anchor_child(self, child,
                             along, inset, up, rotations, along_center,
                             carve=carve)



class Roof(BimElement):
    """
    Roof assembly container for grouping roof face elements.

    Maps to IfcRoof. Contains Solid children for each roof face/slope and
    Sweep children for rafters/purlins.

    The Roof container:
    - Creates IfcRoof as the parent element
    - Child Solid/Sweep elements become aggregated roof faces/members
    - Child appearance follows the spec §6 precedence chain (see the
      generator's ``_apply_container_color``); material=None children
      render sketch-stage

    Example - Gable roof:
        roof = Roof(id="main_roof")
        roof.add(
            Extrude(
                contour=[
                    Point(x=-4600, y=0, z=-3600),
                    Point(x=4600, y=0, z=-3600),
                    Point(x=4600, y=1500, z=0),
                    Point(x=-4600, y=1500, z=0),
                ],
                thickness=200, id="south_slope"
            ),
            Extrude(
                contour=[
                    Point(x=4600, y=0, z=3600),
                    Point(x=-4600, y=0, z=3600),
                    Point(x=-4600, y=1500, z=0),
                    Point(x=4600, y=1500, z=0),
                ],
                thickness=200, id="north_slope"
            ),
        )
        proj.add(roof)

    For simple flat roofs, a single Solid child is sufficient.
    """
    _elements: List = PrivateAttr(default_factory=list)

    def add(self, *elements, carve: str = None) -> "Roof":
        """Add child geometry in WORLD coordinates — the table is keyed on this
        container's IFC class (see :func:`allowed_children_for`), so the
        `Element(ifc_class=…)` spelling accepts exactly the same children."""
        _add_children(self, elements, carve=carve)
        return self

    def anchor(self, child, *, along: int = 0, along_center: int = None,
               inset: int = 0, up: int = 0, rotations=None,
               carve: str = None) -> "Roof":
        """Place ``child`` in THIS element's local frame.

        ``along`` runs the length, ``up`` the height, and ``out`` through the
        thickness with POSITIVE = outward from the building. Local (0, 0, 0)
        is this element's lower-left-back corner. Contrast ``.add()``, which
        is pure containment and never moves the child.

        ``child`` may be ANY element type — geometry primitive or container.
        See :func:`_anchor_child` for why there is no table.

        ``carve`` is the same argument ``.add()`` takes and means the same
        thing. **Anchoring already grants half of it**: an anchored child is
        out of the inferred-carve pairing by construction, because its position
        is a relationship to its host rather than an accidental overlap — which
        is the whole of what ``carve="none"`` says on an ``.add()``. Stating it
        here extends that to the other carve an anchored child can still take,
        the OPENING/VOID one: a hole removes matter regardless of how the
        matter got there, and ``carve="none"`` is the author saying THIS matter
        is meant to be in it — a sill or a lintel standing in the reveal.

        So ``"other"`` and ``"self"`` are accepted and INERT: the pairing they
        name does not happen for an anchored child at all, so there is no
        direction to state. Refused instead, one argument would mean different
        things depending on which verb you wrote it on, and the author would
        have to remember which subset each accepts.
        """
        return _anchor_child(self, child,
                             along, inset, up, rotations, along_center,
                             carve=carve)



# =============================================================================
# GEOMETRY PRIMITIVES
# =============================================================================

class Box(BimElement):
    """
    Axis-aligned box primitive (DSL v8).

    Two opposite corner points define the box; corners auto-normalize
    (any opposite pair is accepted). Supports sequential rotations about
    the box center for authoring convenience.

    IFC mapping: IfcExtrudedAreaSolid.

    Parameters:
        start: First corner point (mm, integers)
        end: Opposite corner point (mm, integers)
        rotations: Sequential (axis, angle_centidegrees) tuples
        type: Semantic type for coloring/IFC routing
        color: Optional color override ("foundation", "wall", "slab", ...)

    Example - Foundation:
        Box(
            start=Point(x=-5500, y=0, z=-4500),
            end=Point(x=5500, y=300, z=4500),
            color="foundation", id="foundation"
        )

    Example - Wall with a door cut (chainable difference):
        Box(
            start=Point(x=-6000, y=300, z=-4000),
            end=Point(x=6000, y=3000, z=-3700),
            id="south_wall"
        ).difference(
            Box(start=Point(x=-500, y=300, z=-4100),
                end=Point(x=400, y=2400, z=-3600),
                material="Void", id="door_void")
        )
    """
    start: Optional[Point] = None
    end: Optional[Point] = None
    type: str = ""  # "site", "foundation", "ceiling", "wall", "wall-opening", "roof", "sketch"
    rotations: List[Tuple[str, int]] = Field(default_factory=list)
    color: Optional[str] = None

    @model_validator(mode='after')
    def validate_box(self):
        if self.start is None or self.end is None:
            raise ValueError(
                "Box requires both start= and end= corner points"
            )
        return self


class Extrude(BimElement):
    """
    World-space polygon extruded normal to its plane (DSL v8).

    The contour is a 3D polygon in world coordinates; the system computes
    the best-fit plane using Newell's Method, determines the surface normal
    from winding order, and extrudes by ``thickness`` along the normal. A
    thin Extrude is a sheet/pane.

    IFC mapping: IfcExtrudedAreaSolid, IfcFacetedBrep.

    Parameters:
        contour: List of 3D Points (min 3, world coords)
        thickness: Extrusion thickness (mm)
        type: Semantic type for coloring/IFC routing
        color: Optional color override

    Example - Pitched roof face:
        Extrude(
            contour=[
                Point(x=-6300, y=3400, z=-4300),
                Point(x=6300, y=3400, z=-4300),
                Point(x=0, y=6200, z=0),
            ],
            thickness=200, id="roof_south"
        )
    """
    contour: Optional[List[Point]] = None
    thickness: Optional[int] = None
    type: str = ""  # "site", "foundation", "ceiling", "wall", "roof", "sketch"
    color: Optional[str] = None

    @model_validator(mode='after')
    def validate_extrude(self):
        if self.contour is None:
            raise ValueError("Extrude requires contour= (min 3 points)")
        if len(self.contour) < 3:
            raise ValueError("Contour must have at least 3 points")
        first = self.contour[0]
        all_same = all(
            p.x == first.x and p.y == first.y and p.z == first.z
            for p in self.contour
        )
        if all_same:
            raise ValueError("Contour points must differ (cannot all be the same)")
        return self


class Sweep(BimElement):
    """
    2D profile swept along a path (DSL v8), or a linear extrusion between
    two points (legacy/transitional form). The two forms are discriminated
    on which kwargs are present — mixing them is a construction error.

    **Path form** — ``Sweep(profile=[Point2D x3+], path=[Point x2+],
    fillet_radius=0, profile_rotation=0, name=)``:

    - ``profile``: 2D cross-section polygon; (0, 0) is the path axis.
      Per-segment orientation (pinned decision, WS1 PR-E): local X =
      ``unit(up_ref x d)`` and local Y = ``d x X`` for segment direction
      ``d``, with ``up_ref = Z-up`` for non-vertical segments (profile
      X = horizontal "across", Y = "up", per spec) and ``up_ref = world X``
      for near-vertical segments (``|d.z| > 0.99``) — for a window frame in
      a wall this puts profile X through the wall on every member, so
      ``Material(profile_mm=(95, 45))`` reads "95 = through wall" on
      verticals and horizontals alike. Rectangular material profiles are
      symmetric about both axes, so axis signs never change the geometry.
    - ``profile`` may be omitted when a member Material carries the profile:
      ``Sweep(material=Material(key="Timber_C24", profile_mm=(45, 195)),
      path=[...])`` (one-source rule — providing both is a compile error).
    - Straight 2-point path -> plain extrusion (IfcExtrudedAreaSolid).
    - Multi-point path -> one extrusion per segment, MITERED at each joint:
      every segment is trimmed by the half-space at the joint's angle
      bisector plane, so adjacent segments meet exactly (no double-solid
      overlap, no gap).
    - Closed-loop path (first point == last point) -> a frame: the wrap
      joint at the loop start is mitered too (the spec's window-frame
      example).
    - **Joint count is a cost.** Each mitered joint is a half-space
      DIFFERENCE, so a path of ``n`` segments emits ``3n-1`` booleans
      (``2n`` differences + ``n-1`` unions). That is cheap for a building
      member — a window frame is 11, the steel-bridge example's worst is 45
      — and superlinear past that: 287 booleans takes ifcopenshell ~17s and
      659 produced no shape at all in 420s, which is an IFC that validates,
      hashes and renders in our viewer but cannot be opened in Blender
      Bonsai. The compiler prints ``csg width`` on every compile and warns
      past 99. **For a tube along a dense polyline, author a Mesh instead** —
      no booleans, and you control the frame.
    - ``fillet_radius > 0``: OPEN paths only — corners become faceted arcs
      (chords <= 15 deg) with mitered sub-joints; a closed loop with a
      fillet is a construction error (frame corners are mitered).
    - ``profile_rotation``: centidegrees, rotates the profile about the
      path axis.

    **Legacy form** — ``Sweep(start=, end=, wide_axis=)``: kept working during
    the transition (full-form removal is WS2). The section comes from a member
    ``Material(profile_mm=(w, h))`` — the ``shape=`` catalog key was REMOVED
    (the bim_catalog profile catalog is retired; see
    infrastructure/specs/archive/bim-catalog.md).
    Sweep in local XY plane, extrusion along local Z, placed at midpoint with
    lookAt orientation. IFC mapping: IfcExtrudedAreaSolid, IfcBeam/IfcColumn.

    Example - Rafter (material carries the profile):
        Sweep(
            material=Material(key="Timber_C24", profile_mm=(45, 195)),
            path=[Point(x=-6000, y=-4800, z=2700), Point(x=-6000, y=0, z=4500)],
        )

    Example - Mitered window frame (closed loop):
        frame = [Point(x=22, y=100, z=22), Point(x=1178, y=100, z=22),
                 Point(x=1178, y=100, z=1378), Point(x=22, y=100, z=1378),
                 Point(x=22, y=100, z=22)]      # closed -> mitered corners
        Sweep(material=Material(key="Timber_C24", profile_mm=(95, 45)),
              path=frame)

    Example - Column (legacy form, section from the member material):
        Sweep(
            material=Material(key="Concrete_C30-37", profile_mm=(100, 100)),
            start=Point(x=0, y=0, z=0),
            end=Point(x=0, y=3000, z=0),
            id="corner_column"
        )
    """
    # Legacy form fields (transitional — WS2 removes the form, not this PR)
    start: Optional[Point] = None
    end: Optional[Point] = None
    wide_axis: str = "x"  # "x" = standing, "y" = flat (legacy form only)
    # Path form fields
    profile: Optional[List[Point2D]] = None  # 2D cross-section; (0,0) = path axis
    path: Optional[List[Point]] = None         # sweep path (2+ points; closed loop = frame)
    fillet_radius: int = 0                     # mm; open paths only (faceted arcs)
    profile_rotation: int = 0                  # centidegrees about the path axis
    color: Optional[str] = None

    _strict_int_mm = field_validator(
        "fillet_radius", "profile_rotation", mode="before"
    )(_reject_float_when_strict)

    @model_validator(mode='after')
    def validate_form(self):
        has_legacy = self.start is not None or self.end is not None
        has_path = self.path is not None
        if has_legacy and (has_path or self.profile is not None):
            raise ValueError(
                "Sweep: provide either the legacy form (start=, end=) or "
                "the path form (path=, profile=) — not both in one call"
            )
        if not has_legacy and not has_path:
            raise ValueError(
                "Sweep: needs a geometry source — path= (path form) or "
                "start= + end= (legacy form)"
            )
        if has_path:
            if self.wide_axis != "x":
                raise ValueError(
                    "Sweep: wide_axis= is legacy-form only; the path form "
                    "orients the profile via profile=/profile_rotation="
                )
            if self.profile is not None:
                if len(self.profile) < 3:
                    raise ValueError(
                        f"Sweep: profile needs at least 3 points "
                        f"(got {len(self.profile)})"
                    )
                first = self.profile[0]
                if all(p.x == first.x and p.y == first.y for p in self.profile):
                    raise ValueError("Sweep: profile points must differ")
            if self.fillet_radius < 0:
                raise ValueError("Sweep: fillet_radius must be >= 0")
            pts = [(p.x, p.y, p.z) for p in self.path]
            closed = len(pts) >= 2 and pts[0] == pts[-1]
            if closed and len(pts) < 4:
                raise ValueError(
                    "Sweep: a closed-loop path needs at least 4 points "
                    "(3 distinct vertices)"
                )
            if closed and self.fillet_radius > 0:
                raise ValueError(
                    "Sweep: fillet_radius on a closed-loop path is not "
                    "supported — closed frames get mitered corners "
                    "(fillet_radius=0)"
                )
            from lite_step.ifc.geometry import validate_fillet_path
            # For a closed loop, extend the walk by one point so the wrap
            # joint at pts[0] (last segment -> first segment) is checked too.
            check_pts = (pts + [pts[1]]) if closed else pts
            errors = validate_fillet_path(
                check_pts, self.fillet_radius,
                f"Sweep '{self.name or self.id}'",
            )
            if errors:
                raise ValueError("; ".join(errors))
        else:
            if self.start is None or self.end is None:
                raise ValueError(
                    "Sweep: legacy form requires both start= and end= "
                    "(or use the path form)"
                )
            if self.fillet_radius != 0 or self.profile_rotation != 0:
                raise ValueError(
                    "Sweep: fillet_radius=/profile_rotation= are path-form "
                    "fields; the legacy start=/end= form does not take them"
                )
        return self


class _SheetSpec(BaseModel):
    """The construction-time validation surface of :class:`Sheet`.

    ``Sheet`` itself lowers to a ``Sweep`` and therefore cannot BE a pydantic
    model; this private model is what gives its two own fields the same
    validator stack every sibling primitive gets — field types, ``extra
    ="forbid"``, and the strict int-mm gate on ``thickness``.
    """
    profile: List[Point2D]
    thickness: float

    _strict_int_mm = field_validator("thickness", mode="before")(
        _reject_float_when_strict
    )

    class Config:
        extra = "forbid"


class _SheetMeta(type):
    """Makes ``isinstance(x, Sheet)`` fail LOUDLY instead of answering wrongly.

    ``Sheet`` is a constructor that returns a ``Sweep``, so a plain
    ``isinstance(elem, Sheet)`` is ``False`` for every object ``Sheet()`` ever
    produced. That is a silently wrong answer to a reasonable question, and
    ``isinstance`` is the idiom the DSL reference points survey authors at for
    classifying elements. A survey looking for sheet parts would quietly find
    none and report a confident zero.

    Raising converts that into a message that names the fix.
    """

    def __instancecheck__(cls, obj):
        raise TypeError(
            "isinstance(x, Sheet) is always False and never what you want — "
            "Sheet is a constructor, not an element type: it closes the face "
            "line and returns a Sweep. To find sheet parts in a survey, test "
            "isinstance(x, Sweep) and narrow on the material "
            "(x.material and x.material.key == 'Zinc_Titanium_EN988'), or tag "
            "them with name= when you build them."
        )

    def __subclasscheck__(cls, sub):
        raise TypeError(
            "issubclass(..., Sheet) is meaningless — Sheet is a constructor "
            "that returns a Sweep, not a base class."
        )


class Sheet(metaclass=_SheetMeta):
    """
    Folded SHEET METAL from its open face line plus a thickness (DSL v19.20).

    ``Sheet(profile=[Point2D x2+], thickness=<int mm>, path=[Point x2+],
    material=, name=, color=)``

    A sheet part — sålbænk (window sill), inddækning (flashing), gutter,
    capping, cold-formed section — is specified the way it is fabricated: the
    VISIBLE OUTER FACE as an **open** polyline, plus a stock gauge. A line has
    no area, so it cannot be extruded; ``thickness`` closes it into a loop by
    offsetting the line onto its material side and joining the two runs
    (``outer + reversed(inner)``).

    **WHICH SIDE GETS THE MATERIAL IS DECIDED BY THE DIRECTION OF TRAVEL OF
    THE POLYLINE.** For a segment direction ``d = (dx, dy)`` the material lies
    along ``n = (-dy, dx)`` — 90 degrees to the LEFT of travel. Reversing the
    point order puts the same face line's material on the OTHER side. This is
    the single most surprising thing about the primitive: if a sill comes out
    thickened upward into the window instead of downward into the reveal, the
    fix is to reverse ``profile``, not to change any coordinate.

    ``Sheet`` is a CONSTRUCTOR, not an element type: it returns a path-form
    :class:`Sweep` carrying the closed profile. Everything the path form
    already does then applies unchanged — per-segment orientation, mitered
    joints, ``fillet_radius``, displacement bounds — and the emitted geometry
    is one ``IfcExtrudedAreaSolid`` per path segment over an
    ``IfcArbitraryClosedProfileDef``. (``IfcCenterLineProfileDef`` looks like
    the right entity for exactly this job and is a trap: against ifcopenshell
    0.8.5 a ``Thickness`` of 2 and of 20 produce byte-identical geometry — the
    kernel accepts the entity, raises nothing, and builds a ZERO-THICKNESS
    ribbon. An explicit closed profile is what both kernels provably render.)

    Profile rules:

    * ``profile`` is OPEN and needs at least 2 points, in the same profile
      plane as ``Sweep``'s path form ((0, 0) is the path axis).
    * first point == last point is a construction error — a closed sheet is
      meaningless; supply the open face line and a thickness.
    * coordinates are int millimeters (RULE 2). The offset run is inherently
      fractional and is rounded to int mm, so a finely tessellated bend
      normally collapses several arc points into duplicates — those are
      DEDUPED, not rejected. Fewer than 3 distinct points surviving IS an
      error.
    * a thickness that exceeds the profile's inner turn radius makes the
      offset face cross itself. That is a compile error naming the thickness
      and the offending segment pair — silently emitting the knotted loop is
      how a part renders as garbage with nothing in any log.

    When ``material=`` resolves to a registry entry whose geometry is
    ``form="sheet"`` with a stock list, ``thickness`` must be one of its
    ``thicknesses_mm``.

    Example — sålbænk under a window opening, 1 mm titanium zinc::

        Sheet(name="sill",
              profile=[Point2D(x=122, y=49), Point2D(x=72, y=49),
                       Point2D(x=0, y=30), Point2D(x=0, y=3)],
              thickness=1,
              path=[Point(x=-600, y=100, z=900),
                    Point(x=600, y=100, z=900)],
              material=Material(key="Zinc_Titanium_EN988"))
    """

    def __new__(cls, *, profile, thickness, path, **kwargs) -> "Sweep":
        from lite_step.ifc.geometry import (
            dedupe_consecutive_2d,
            find_loop_self_intersection,
            offset_open_polyline,
        )

        name = kwargs.get("name")
        label = f"Sheet '{name}'" if name else "Sheet"

        spec = _SheetSpec(profile=profile, thickness=thickness)
        if len(spec.profile) < 2:
            raise ValueError(
                f"{label}: profile needs at least 2 points — it is the OPEN "
                f"outer face line of the part (got {len(spec.profile)})"
            )
        if spec.thickness <= 0:
            raise ValueError(
                f"{label}: thickness must be > 0 (got {spec.thickness}) — it "
                f"is the sheet gauge that closes the face line into a solid"
            )

        pts = [(p.x, p.y) for p in spec.profile]
        if (pts[0][0] == pts[-1][0]) and (pts[0][1] == pts[-1][1]):
            raise ValueError(
                f"{label}: profile first point == last point — a closed sheet "
                f"is meaningless; supply the open face line and a thickness "
                f"(use Sweep(profile=...) for an already-closed section)"
            )

        Sheet._check_registry_thickness(spec.thickness, kwargs.get("material"),
                                        label)

        # Exact consecutive duplicates in the AUTHORED line would give the
        # offset a zero-length segment (an undefined normal), so they go first.
        outer = dedupe_consecutive_2d(pts, tol=0.0)
        if len(outer) < 2:
            raise ValueError(
                f"{label}: profile collapses to a single point — the face "
                f"line has no length"
            )
        inner = offset_open_polyline(outer, spec.thickness)
        loop = outer + list(reversed(inner))

        hit = find_loop_self_intersection(loop)
        if hit is not None:
            i, j = hit
            raise ValueError(
                f"{label}: thickness {_mm(spec.thickness)} mm exceeds the "
                f"profile's inner turn radius; the offset face "
                f"self-intersects between segments {i} and {j} (indices into "
                f"the closed {len(loop)}-point loop). Reduce thickness or "
                f"open the bend radius."
            )

        rounded = [(int(round(x)), int(round(y))) for x, y in loop]
        closed = dedupe_consecutive_2d(rounded, tol=0.0, wrap=True)
        if len(closed) < 3:
            raise ValueError(
                f"{label}: the closed profile has only {len(closed)} distinct "
                f"point(s) after rounding to int millimeters — the part is "
                f"smaller than the millimeter grid. Scale the profile up or "
                f"model it as an Extrude."
            )

        return Sweep(profile=[Point2D(x=x, y=y) for x, y in closed],
                     path=path, **kwargs)

    @staticmethod
    def _check_registry_thickness(thickness, material, label: str) -> None:
        """Stock-gauge check against a ``form="sheet"`` registry entry.

        Unlike ``Material(thickness_mm=)`` — which WARNS off-stock, because a
        planar buildup layer is routinely custom-cut — a folded part's gauge is
        the product: zinc is sold in the gauges it is sold in, and a part
        drawn at a gauge nobody rolls is a fabrication error, not a note.
        """
        from lite_step.materials import registry_definition, resolve_material

        mat = resolve_material(material)
        if mat is None:
            return
        mdef = registry_definition(mat.key)
        if mdef is None:
            return
        geo = mdef.geometry
        if geo.form != "sheet":
            return
        stock = geo.thicknesses_mm or []
        if stock and thickness not in stock:
            raise ValueError(
                f"{label}: thickness {_mm(thickness)} mm is not a stock gauge "
                f"for {mat.key!r} — available: "
                f"{', '.join(str(s) for s in stock)} mm"
            )


def _mm(value) -> str:
    """Render an int-mm value without a trailing ``.0`` (fields are float-typed)."""
    return str(int(value)) if float(value).is_integer() else str(value)


class Pipe(BimElement):
    """
    Circular profile swept along a centerline path (DSL v8).

    ``Pipe(path=[Point x2+], radius=, fillet_radius=0, name=)``

    Bends between consecutive segments become TRUE ARCS of radius
    ``fillet_radius`` (0 = sharp corners). Use for handrails, pipes/plumbing,
    dowels, cables — anything with a round section following a path.

    IFC mapping: IfcSweptDiskSolid.

    Parameters:
        path: Centerline points (mm, world coords), minimum 2.
        radius: Section radius (mm), > 0. NOTE: radius, not diameter.
        fillet_radius: Centerline bend radius at interior path joints (mm).
            0 = sharp. Must fit within the adjacent segments
            (r*tan(theta/2) tangent trim per side).
    """
    path: List[Point]
    radius: int
    fillet_radius: int = 0
    color: Optional[str] = None

    _strict_int_mm = field_validator("radius", "fillet_radius", mode="before")(
        _reject_float_when_strict
    )

    @model_validator(mode='after')
    def validate_pipe(self):
        if self.radius <= 0:
            raise ValueError(f"Pipe: radius must be > 0 (got {self.radius})")
        if self.fillet_radius < 0:
            raise ValueError("Pipe: fillet_radius must be >= 0")
        from lite_step.ifc.geometry import validate_fillet_path
        errors = validate_fillet_path(
            [(p.x, p.y, p.z) for p in self.path],
            self.fillet_radius, f"Pipe '{self.name or self.id}'",
        )
        if errors:
            raise ValueError("; ".join(errors))
        return self


class Revolve(BimElement):
    """
    2D profile revolved around an axis (DSL v8).

    ``Revolve(profile=[Point2D x3+], path=[Point x2 AXIS], angle=36000, name=)``

    The 2-point ``path`` IS the axis of revolution — NOT a travel path.
    Sweep coordinates: X = radial distance from the axis (all X >= 0,
    construction-enforced), Y = distance along the axis from path[0]
    toward path[1]. ``angle`` is in centidegrees (36000 = full turn).

    Pinned conventions (WS1 PR-E — proven by the measured arched-doorway
    test in lite_step/tests/test_primitives_ifc.py):

    * **Winding**: the revolution follows the RIGHT-HAND RULE about the
      axis direction path[0] -> path[1].
    * **Start plane**: the profile's radial X-axis starts at
      ``r = unit(world_Z x axis)``. For any horizontal axis the sweep
      therefore starts in a horizontal direction and rotates UPWARD first
      — so a semicircular ``angle=18000`` Revolve about a horizontal axis is
      the UPPER half (the spec's arched doorway), regardless of which way
      the axis points. For a (near-)vertical axis (|axis.z| ~ 1, where
      world_Z x axis degenerates) the start plane falls back to world +X
      and the sweep is counterclockwise seen from above for an axis
      pointing up.
    * **Angle sign**: angle must be in (0, 36000] — a negative sweep is a
      construction error (flip the axis direction instead).

    IFC mapping: IfcRevolvedAreaSolid (ifcopenshell backend).
    """
    profile: List[Point2D]
    path: List[Point]
    angle: int = 36000
    color: Optional[str] = None

    _strict_int_mm = field_validator("angle", mode="before")(
        _reject_float_when_strict
    )

    @model_validator(mode='after')
    def validate_revolve(self):
        if len(self.profile) < 3:
            raise ValueError(
                f"Revolve: profile needs at least 3 points (got {len(self.profile)})"
            )
        bad_x = [p for p in self.profile if p.x < 0]
        if bad_x:
            raise ValueError(
                f"Revolve: profile X is the radial distance from the axis and "
                f"must be >= 0 (got x={bad_x[0].x}); mirror the profile "
                f"instead of crossing the axis"
            )
        first = self.profile[0]
        if all(p.x == first.x and p.y == first.y for p in self.profile):
            raise ValueError("Revolve: profile points must differ")
        if len(self.path) != 2:
            raise ValueError(
                f"Revolve: path is the 2-point AXIS of revolution "
                f"(got {len(self.path)} points) — it is not a travel path"
            )
        p0, p1 = self.path
        if (p0.x, p0.y, p0.z) == (p1.x, p1.y, p1.z):
            raise ValueError("Revolve: the two axis points must differ")
        if not (0 < self.angle <= 36000):
            raise ValueError(
                f"Revolve: angle must be in (0, 36000] centidegrees "
                f"(got {self.angle}); flip the axis direction instead of "
                f"using a negative sweep"
            )
        return self


#: EC2 (DS/EN 1992-1-1) Table 8.1N minimum mandrel diameters, keyed by the
#: registry BarGeometry ``bend_rule``. phi <= 16 mm -> 4*phi; phi > 16 mm ->
#: 7*phi. The auto ``bend_radius`` is the CENTERLINE radius:
#: (mandrel_diameter + bar_diameter) / 2.
_BEND_RULES = {
    "EC2_8.1N": lambda d: ((4 * d if d <= 16 else 7 * d) + d) // 2,
}

#: Bar bar_type -> IFC4 IfcReinforcingBarTypeEnum. "stirrup" maps to
#: LIGATURE — IFC4 has no STIRRUP literal; LIGATURE is the enclosing-link
#: (stirrup) member of the enum.
BAR_TYPE_TO_IFC = {
    "main": "MAIN",
    "shear": "SHEAR",
    "stirrup": "LIGATURE",
    "edge": "EDGE",
}


class Bar(BimElement):
    """
    Reinforcement bar (DSL v1.5, WS1 PR-E).

    ``Bar(path=, diameter=, bend_radius=None, grade="B500B", bar_type="main",
    mark=None, name=)``

    * ``diameter`` is the NOMINAL diameter (mm, not radius) and must be in
      the grade's registry ``stock_diameters_mm`` — off-stock diameters are
      a construction error.
    * ``grade`` resolves against the material registry: a bare grade like
      ``"B500B"`` resolves to ``"Steel_B500B"`` (an exact registry key is
      also accepted). The resolved key is auto-assigned as the Bar's
      ``material=`` so registry Psets + render attach as usual.
    * ``bend_radius=None`` -> auto EC2 mandrel from the registry's
      ``bend_rule`` (EC2_8.1N: mandrel 4*phi for phi <= 16, 7*phi above;
      centerline radius = (mandrel + phi) / 2 — e.g. phi 8 -> 20 mm).
    * ``bar_type``: main | shear | stirrup | edge (IFC PredefinedType:
      MAIN | SHEAR | LIGATURE | EDGE — IFC4 has no STIRRUP literal).
    * ``mark`` is the schedule mark — SHARED across bars of one position,
      not identity. It is emitted as the IFC ``Tag`` attribute.

    Derived quantities (emitted as IfcElementQuantity
    "Qto_ReinforcingElementBaseQuantities"): Length = true centerline
    length (arc-corrected at bends); Weight = Length * area * density,
    density pinned to the registry's
    ``psets["Pset_MaterialCommon"]["MassDensity"]`` (7850 kg/m3 for
    Steel_B500B).

    IFC mapping: IfcReinforcingBar with NominalDiameter; geometry is an
    IfcSweptDiskSolid along the (possibly bent) path.

    Add to a host container: ``beam.add(*bars)``.
    """
    path: List[Point]
    diameter: int
    bend_radius: Optional[int] = None
    grade: str = "B500B"
    bar_type: Literal["main", "shear", "stirrup", "edge"] = "main"
    mark: Optional[str] = None
    color: Optional[str] = None

    _strict_int_mm = field_validator("diameter", "bend_radius", mode="before")(
        _reject_float_when_strict
    )

    @model_validator(mode='after')
    def validate_bar(self):
        # Lazy registry import — same cycle-avoidance as models.material.
        from lite_step.materials import registry_definition

        resolved_key = self.grade
        mdef = registry_definition(resolved_key)
        if mdef is None:
            resolved_key = f"Steel_{self.grade}"
            mdef = registry_definition(resolved_key)
        if mdef is None:
            raise ValueError(
                f"Bar: unknown grade {self.grade!r} — no registry material "
                f"{self.grade!r} or {resolved_key!r}"
            )
        geo = mdef.geometry
        if geo.form != "bar":
            raise ValueError(
                f"Bar: material {resolved_key!r} is form={geo.form!r}, "
                f"not reinforcement bar stock"
            )
        if self.diameter not in geo.stock_diameters_mm:
            raise ValueError(
                f"Bar: diameter {self.diameter} is not in {resolved_key!r} "
                f"stock_diameters_mm {geo.stock_diameters_mm}"
            )
        if self.bend_radius is None:
            rule = _BEND_RULES.get(geo.bend_rule)
            if rule is None:  # registry literal guarantees EC2_8.1N today
                raise ValueError(
                    f"Bar: no bend rule implementation for {geo.bend_rule!r}"
                )
            self.bend_radius = rule(self.diameter)
        elif self.bend_radius <= 0:
            raise ValueError("Bar: bend_radius must be > 0 (None = EC2 auto)")
        # The grade IS the material: auto-assign the resolved registry key
        # so Psets/render attach through the normal material machinery.
        if self.material is None:
            self.material = resolved_key
        from lite_step.ifc.geometry import validate_fillet_path
        errors = validate_fillet_path(
            [(p.x, p.y, p.z) for p in self.path],
            self.bend_radius, f"Bar '{self.name or self.id}'",
        )
        if errors:
            raise ValueError("; ".join(errors))
        return self


#: The IFC4-era element classes (all still valid in IFC4X3). Built from
#: the IfcBuildingElement subtype tree (minus
#: IfcWallStandardCase — requires standard material-layer setup) plus
#: ``IfcGeographicElement`` — the landscape/ground-feature class that lives
#: under ``IfcSite`` (terrain, planting, site features).
ELEMENT_IFC4_CLASSES = frozenset({
    "IfcBeam", "IfcBuildingElementProxy", "IfcChimney", "IfcColumn",
    "IfcCovering", "IfcCurtainWall", "IfcDoor", "IfcFooting",
    "IfcGeographicElement", "IfcMember",
    "IfcPile", "IfcPlate", "IfcRailing", "IfcRamp", "IfcRampFlight",
    "IfcRoof", "IfcShadingDevice", "IfcSlab", "IfcStair", "IfcStairFlight",
    "IfcWall", "IfcWindow",
})

#: IFC4.3 semantic classes — the groundworks / geotechnical / landscape /
#: marine vocabulary, plus the rail and road-furniture PRODUCTS. The output
#: schema is IFC4X3_ADD2 (``lite_step/ifc/schema_version.py``), so these emit
#: NATIVELY — IfcEarthworksFill is an IfcEarthworksFill in the file. Kept as
#: a distinct set for documentation and tests; emission does not branch
#: on it.
#:
#: What is absent, and why:
#:
#: * ``IfcRoad`` / ``IfcRailway`` / ``IfcRoadPart`` / ``IfcRailwayPart`` are
#:   SPATIAL, not products. They go through ``SpatialElement``; putting one
#:   here would emit it into a containment relation and violate WR31 with no
#:   validator to say so (see ``lite_step/ifc/facilities.py``).
#: * ``IfcAlignment`` and its segment/placement family are a whole geometry
#:   and placement domain — deferred, see
#:   ``infrastructure/specs/ifc43-facilities.md`` §5. Without it a road model
#:   is not referenceable by station, which is how that industry works.
#: * ``IfcVehicle`` is rolling stock, not built infrastructure. Nothing here
#:   authors a train.
#: * ``IfcPavementPart`` does NOT exist in IFC4X3_ADD2 (it was named from
#:   memory during review and is wrong); ``IfcPavement`` does and is below.
ELEMENT_IFC43_CLASSES = frozenset({
    "IfcBearing", "IfcBorehole", "IfcCaissonFoundation", "IfcCourse",
    "IfcEarthworksCut", "IfcEarthworksFill", "IfcGeomodel", "IfcGeoslice",
    "IfcGeotechnicalStratum", "IfcKerb", "IfcMooringDevice",
    "IfcNavigationElement", "IfcPavement", "IfcReinforcedSoil", "IfcSign",
    # Rail products (step 2 of the facilities spec). IfcSignal's supertype is
    # IfcDistributionElement rather than IfcBuiltElement like the other two —
    # verified to emit natively at 0 conformance errors anyway, because the
    # generic creator keys on the class name, not on the supertype branch.
    "IfcRail", "IfcTrackElement", "IfcSignal",
})

#: Every class a generic ``Element`` accepts. An unknown ifc_class is a
#: construction error.
ELEMENT_IFC_CLASS_WHITELIST = ELEMENT_IFC4_CLASSES | ELEMENT_IFC43_CLASSES


class Element(BimElement):
    """
    Generic SEMANTIC wrapper for any whitelisted IFC class (DSL v1.5, WS1 PR-E).

    ``Element(ifc_class="IfcStair", predefined_type=None, name=, props=)``

    The escape hatch for IFC classes without a semantic sugar container
    (Wall/Roof/Slab/... stay as-is — they are sugar for Element). Children
    define the geometry per the container pattern: Box, Extrude, Sweep,
    Revolve, Pipe, Bar, Mesh — geometry and semantics are SEPARATE layers:
    the child says what shape it is, the Element says what it IS, and the
    parent container (Building storey vs ``Site``) says where it belongs.
    An Element with no children is legal — it emits a product with identity
    + props but no geometry (e.g. a stair placeholder carrying
    Pset_StairCommon).

    ``ifc_class`` must be in ELEMENT_IFC_CLASS_WHITELIST — IFC4-native
    classes plus the IFC4.3 groundworks/landscape vocabulary (roads and rail
    excluded). The output schema is IFC4X3_ADD2, so every class emits
    NATIVELY. ``predefined_type`` refines the class (e.g. ``"TERRAIN"`` on
    IfcGeographicElement, ``"BACKFILL"`` on IfcEarthworksFill); a value the
    class's enum doesn't know is emitted as ``USERDEFINED`` with the truth
    in ``ObjectType``.

    Canonical site-work forms (Element under Site → placed under IfcSite):

        terrain = Element(ifc_class="IfcGeographicElement",
                          predefined_type="TERRAIN", name="terrain")
        terrain.add(Mesh(heightmap=..., depth=..., corner_min=..., corner_max=...))
        site.add(terrain)                       # THE carve-host marker

        gravel = Element(ifc_class="IfcEarthworksFill",
                         predefined_type="BACKFILL", name="stenkant")
        gravel.add(Mesh(heightmap=..., ...))    # NOT a carve host — never
        site.add(gravel)                        # steals a foundation carve

    Example:
        stair = Element(ifc_class="IfcStair", name="stair_main",
                        props={"Pset_StairCommon": {"NumberOfRiser": 16}})
        stair.add(Box(start=Point(...), end=Point(...)))
        proj.add(stair)
    """
    ifc_class: str
    #: Refines the class (IFC PredefinedType). Free string by design: a value
    #: in the class's IFC4 enum is emitted natively; anything else (and every
    #: 4.3-only class) becomes USERDEFINED + ObjectType — the sanctioned
    #: mechanism, never a silent drop.
    predefined_type: Optional[str] = None
    _elements: List = PrivateAttr(default_factory=list)

    @field_validator("ifc_class")
    @classmethod
    def validate_ifc_class(cls, v: str) -> str:
        if v not in ELEMENT_IFC_CLASS_WHITELIST:
            raise ValueError(
                f"Element: unknown ifc_class {v!r} — whitelisted classes "
                f"(IFC4 + IFC4.3 groundworks/landscape, no roads/rail): "
                f"{', '.join(sorted(ELEMENT_IFC_CLASS_WHITELIST))}"
            )
        return v

    @model_validator(mode="after")
    def _layers_need_planar_class(self) -> "Element":
        if self.layers is not None and self.ifc_class not in PLANAR_LAYER_IFC_CLASSES:
            raise ValueError(
                f"layers= on Element requires a planar ifc_class "
                f"({', '.join(sorted(PLANAR_LAYER_IFC_CLASSES))}); got "
                f"{self.ifc_class!r}. Members take Material(profile_mm=).")
        return self

    def add(self, *elements, carve: str = None) -> "Element":
        """Add child geometry in WORLD coordinates — the table is keyed on this
        container's IFC class (see :func:`allowed_children_for`), so the
        `Element(ifc_class=…)` spelling accepts exactly the same children."""
        _add_children(self, elements, carve=carve)
        return self

    def anchor(self, child, *, along: int = 0, along_center: int = None,
               inset: int = 0, up: int = 0, rotations=None,
               carve: str = None) -> "Element":
        """Place ``child`` in THIS element's local frame.

        ``along`` runs the length, ``up`` the height, and ``out`` through the
        thickness with POSITIVE = outward from the building. Local (0, 0, 0)
        is this element's lower-left-back corner. Contrast ``.add()``, which
        is pure containment and never moves the child.

        ``child`` may be ANY element type — geometry primitive or container.
        See :func:`_anchor_child` for why there is no table.

        ``carve`` is the same argument ``.add()`` takes and means the same
        thing. **Anchoring already grants half of it**: an anchored child is
        out of the inferred-carve pairing by construction, because its position
        is a relationship to its host rather than an accidental overlap — which
        is the whole of what ``carve="none"`` says on an ``.add()``. Stating it
        here extends that to the other carve an anchored child can still take,
        the OPENING/VOID one: a hole removes matter regardless of how the
        matter got there, and ``carve="none"`` is the author saying THIS matter
        is meant to be in it — a sill or a lintel standing in the reveal.

        So ``"other"`` and ``"self"`` are accepted and INERT: the pairing they
        name does not happen for an anchored child at all, so there is no
        direction to state. Refused instead, one argument would mean different
        things depending on which verb you wrote it on, and the author would
        have to remember which subset each accepts.
        """
        return _anchor_child(self, child,
                             along, inset, up, rotations, along_center,
                             carve=carve)



class SpatialElement(BimElement):
    """A node in the IFC 4.3 SPATIAL structure — a facility or a part of one.

    ``SpatialElement(ifc_class="IfcBridge", name="storstroem")``

    The spatial mirror of :class:`Element`, and the distinction is not
    ergonomic. ``Element`` is the PHYSICAL path — what it emits joins the tree
    with ``IfcRelContainedInSpatialStructure``. A spatial node joins with
    ``IfcRelAggregates``, and that relation's **WR31** forbids the mix
    outright: *"The relationship object shall not be used to include other
    spatial structure elements into a spatial structure element."*

    The whole chain, from ``infrastructure/specs/ifc43-facilities.md``::

        Project -> Site -> FACILITY -> PART -> elements
                            ^          ^       ^
                            aggregate  aggr.   CONTAINMENT

    ``IfcBuilding`` is itself an ``IfcFacility``, so a building takes the same
    shape — it is simply derived from having storeys rather than authored, and
    ``IfcBuildingStorey`` fills the PART row without being an
    ``IfcFacilityPart``. Classes with a dedicated container (``Site``,
    ``Storey``, ``Space``) are refused here, naming the one to use.

    Example — a bridge with two parts::

        bridge = SpatialElement(ifc_class="IfcBridge", name="storstroem",
                                predefined_type="ARCHED")
        deck = SpatialElement(ifc_class="IfcBridgePart", name="deck",
                              usage="LONGITUDINAL", predefined_type="DECK")
        deck.add(Box(...), Box(...))
        bridge.add(deck)
        site.add(bridge)

    ``usage`` is REQUIRED on every facility part (IFC marks
    ``IfcFacilityPart.UsageType`` mandatory) and refused on anything else. It
    is deliberately not defaulted: ``LONGITUDINAL`` vs ``LATERAL`` is a real
    claim about how the asset subdivides, and a plausible default is worse
    than a refusal because nothing downstream can tell it from a decision.

    ``predefined_type`` is OPTIONAL everywhere — the inverse of ``usage``,
    which is easy to get backwards. It refines the class
    (``"ARCHED"`` on ``IfcBridge``, ``"DECK"`` on ``IfcBridgePart``); a value
    the class's enum does not know emits as ``USERDEFINED`` with the truth in
    ``ObjectType``, exactly as ``Element(predefined_type=)`` does.

    Every policy decision here lives in :mod:`lite_step.ifc.facilities`, read
    by the emitter.
    """
    ifc_class: str
    #: IfcFacilityUsageEnum. Required on facility PARTS, refused elsewhere.
    usage: Optional[str] = None
    #: Refines the class. Optional on every spatial class — see the docstring.
    predefined_type: Optional[str] = None
    #: IfcSpatialElement.LongName. Optional; distinct from the canonical Name.
    long_name: Optional[str] = None
    _elements: List = PrivateAttr(default_factory=list)

    @field_validator("ifc_class")
    @classmethod
    def _validate_ifc_class(cls, v: str) -> str:
        from lite_step.ifc import facilities as _fac
        return _fac.validate_ifc_class(v)

    @model_validator(mode="after")
    def _validate_usage(self) -> "SpatialElement":
        from lite_step.ifc import facilities as _fac
        _fac.validate_usage(self.ifc_class, self.usage)
        return self

    def add(self, *elements, carve: str = None) -> "SpatialElement":
        """Add children — PARTS to a facility, PRODUCTS to a part.

        The split is what the spatial/physical boundary means: a facility
        decomposes, a part contains. ``allowed_children_for`` keys on that
        rather than on the IFC class, because there are ten spatial classes
        and exactly two answers.
        """
        _add_children(self, elements, carve=carve)
        return self



class Mesh(BimElement):
    """
    A triangle mesh, specified one of two ways:

    * **Explicit mode** — ``vertices`` + ``faces`` as before.
    * **Heightmap mode** — ``heightmap`` + ``depth`` + ``corner_min`` +
      ``corner_max``: a ``rows × cols`` grid of literal top-surface heights
      (mm) over the XY rectangle ``corner_min``→``corner_max``, extruded down
      to one flat bottom plane ``depth`` mm below the LOWEST height. The
      compiler tessellates it into a closed prism (top + bottom + 4 skirts,
      outward 0-based windings) at construction — ``vertices``/``faces`` are
      populated and ``is_watertight`` is forced True, so everything downstream
      (the emitter, ``.difference()``/``.void()`` carving, displacement)
      treats it as an ordinary watertight mesh. The heightmap fields are
      RETAINED (not consumed): the script stays compact and future
      surface-query features can read the grid.

      Grid orientation: ``heightmap[0][0]`` at ``corner_min`` (SW), rows
      advance +y, columns advance +x. Heights are taken literally — no
      auto-shift; z-datum conventions belong to the authoring skill.

    IFC mapping: IfcTriangulatedFaceSet, IfcPolygonalFaceSet

    Mesh is pure GEOMETRY — it carries no semantics. What a mesh IS (terrain,
    gravel fill, scanned context) is said by wrapping it in a semantic
    ``Element`` (e.g. ``Element(ifc_class="IfcGeographicElement",
    predefined_type="TERRAIN")``); where it belongs is said by the parent
    container (``Site`` vs a storey).

    Used for: Terrain blocks, gravel beds/aprons, scanned objects, complex
    imported geometry — anything prism-like with a varying top surface fits
    heightmap mode; everything else uses explicit mode.

    Watertight Solid Rule:
        Building elements (such as roof plates, custom slabs, and walls) authored
        as Mesh MUST be closed, watertight 3D volumes with thickness (top, bottom,
        and edge/eave faces). Open, zero-thickness sheets are strongly discouraged:
        - Boolean cuts (.difference().void(), openings) fail because manifold3d
          strictly requires closed 2-manifolds.
        - Backface culling in WebGL/BIM viewers makes open sheets invisible from
          below/behind.
        - Physical BIM quantities (Qto_RoofBaseQuantities, volume) are zero.
        - Triangle vertices must be wound counter-clockwise when viewed from the
          outside so face normals point outward and the volume "holds water"
          (positive signed volume).

    Parameters:
        vertices: List of 3D Points defining mesh vertices (explicit mode)
        faces: List of (i, j, k) tuples indexing into vertices (0-based)
        heightmap: rows × cols grid of literal top heights, mm (heightmap mode)
        depth: positive mm below the lowest height to the flat bottom
        corner_min: SW corner (min x, min y) of the grid's XY rectangle
        corner_max: NE corner (max x, max y) of the grid's XY rectangle
        color: Appearance override — ``"#RRGGBB"`` or ``"#RRGGBBAA"`` (alpha
            in the last byte, e.g. ``"#9E9E9E80"`` for 50%-transparent ghost
            massing). Highest appearance precedence (spec §6: explicit
            ``color=`` beats material render).
        source_type: Origin of mesh data (e.g., "LiDAR", "photogrammetry")
        is_watertight: Whether the mesh forms a closed volume
    """
    vertices: Optional[List[Point]] = None
    faces: Optional[List[Tuple[int, int, int]]] = None  # Triangle indices into vertices
    heightmap: Optional[List[List[float]]] = None
    depth: Optional[float] = None
    #: Absolute level-bottom alternative to ``depth=`` (exactly one of the
    #: two): the flat bottom plane at this z, strictly below the lowest
    #: height. Sibling prisms share one level bottom (the stenkant's four
    #: patches) without per-patch depth arithmetic.
    bottom_z: Optional[float] = None
    corner_min: Optional[Point2D] = None
    corner_max: Optional[Point2D] = None
    source_type: Optional[str] = None
    is_watertight: bool = False
    # Appearance override, consumed by the emitter: "#RRGGBB" or
    # "#RRGGBBAA" (alpha byte — e.g. "#9E9E9E80" = ghost massing at 50%).
    # Highest appearance precedence (spec §6): color= > material hex >
    # registry grade render > inherited wrapper render > terrain default.
    color: Optional[str] = None

    # ``mesh_type`` is GONE (v10 hard cutover). ``extra="forbid"`` now rejects
    # it by name, which is the loud failure this needs — every value it could
    # carry has a replacement: "terrain" became the semantic wrapper
    # (Element(IfcGeographicElement, TERRAIN)), "building_ghost" became
    # ``color="#RRGGBBAA"`` (alpha in the last byte), and "simple"/"scan" were
    # inert annotations that said nothing the file recorded.

    # Strict int-mm gates run BEFORE pydantic's float coercion (the field
    # types store floats, so an after-validator would see 0 already coerced
    # to 0.0 and wrongly reject spec-conforming ints — same rationale as the
    # Point gate). depth/bottom_z reuse the shared Point-field gate verbatim.
    _strict_depth = field_validator("depth", "bottom_z", mode="before")(
        _reject_float_when_strict
    )

    @field_validator("heightmap", mode="before")
    @classmethod
    def _reject_float_heights_when_strict(cls, v: object) -> object:
        """Strict int-mm for heightmap heights, with a grid-specific message
        (clearer than the Point gate firing on an anonymous ``z=``)."""
        if strict_int_mm_enabled() and isinstance(v, list):
            for r, row in enumerate(v):
                if not isinstance(row, list):
                    continue  # let pydantic's own type error report it
                for c, h in enumerate(row):
                    if isinstance(h, float):
                        raise ValueError(
                            f"heightmap[{r}][{c}]={h!r} is a float — heights "
                            f"are int millimeters (DSL v1.5 RULE 2). Wrap "
                            f"divisions in I()."
                        )
        return v

    @model_validator(mode='after')
    def _resolve_specification_mode(self) -> "Mesh":
        """Exactly one specification mode, expanded at construction.

        Explicit mode: ``vertices`` + ``faces``, no heightmap fields.
        Heightmap mode: all four heightmap fields, no vertices/faces —
        tessellated HERE so every downstream consumer (normalizer, backends,
        displacement, validators) sees an ordinary watertight mesh. The
        invariant after construction: ``vertices`` and ``faces`` are always
        populated lists, regardless of mode.
        """
        from .heightfield import tessellate_heightmap

        hm_fields = (self.heightmap, self.depth, self.bottom_z,
                     self.corner_min, self.corner_max)
        wants_heightmap = any(f is not None for f in hm_fields)
        wants_explicit = self.vertices is not None or self.faces is not None

        if wants_heightmap and wants_explicit:
            raise ValueError(
                "Mesh takes EITHER vertices+faces OR "
                "heightmap+depth|bottom_z+corner_min+corner_max — not both."
            )
        if wants_heightmap:
            missing = [
                name for name, val in (
                    ("heightmap", self.heightmap),
                    ("corner_min", self.corner_min),
                    ("corner_max", self.corner_max),
                )
                if val is None
            ]
            if self.depth is None and self.bottom_z is None:
                missing.append("depth (or bottom_z)")
            if missing:
                raise ValueError(
                    f"Mesh heightmap mode needs heightmap, corner_min, "
                    f"corner_max AND exactly one of depth/bottom_z — "
                    f"missing: {', '.join(missing)}."
                )
            self.vertices, self.faces = tessellate_heightmap(
                self.heightmap, self.depth, self.corner_min, self.corner_max,
                bottom_z=self.bottom_z,
            )
            # Closed prism by construction.
            self.is_watertight = True
            return self
        if self.vertices is None or self.faces is None:
            raise ValueError(
                "Mesh needs vertices+faces (explicit mode) or "
                "heightmap+depth|bottom_z+corner_min+corner_max (heightmap mode)."
            )
        return self

    # ── surface-query overrides: exact heightmap fast path ──────────────
    #
    # The general queries live on BimElement (any geometric primitive,
    # tessellate + ray-cast). A heightmap-mode Mesh WITHOUT a placement=
    # answers from its retained grid instead: exact bilinear, and clamped at
    # the rectangle edges (the documented heightfield behavior). With a
    # placement= the grid is not world-space — fall through to the general
    # path, which applies the Transform.

    def height_at(self, *, x: float, y: float, round: bool = False):
        import builtins
        from .heightfield import surface_z_at as _sz
        if self.heightmap is not None and self.placement is None:
            z = _sz(self.heightmap, self.corner_min, self.corner_max, x, y)
            return int(builtins.round(z)) if round else z
        return super().height_at(x=x, y=y, round=round)

    def heightmap_at(self, *, corner_min: "Point2D", corner_max: "Point2D",
                     rows: int, cols: int,
                     round: bool = False) -> List[List[float]]:
        from .heightfield import sample_grid as _sg
        if self.heightmap is not None and self.placement is None:
            return _sg(self.heightmap, self.corner_min, self.corner_max,
                       corner_min, corner_max, rows, cols, round=round)
        return super().heightmap_at(corner_min=corner_min,
                                    corner_max=corner_max,
                                    rows=rows, cols=cols, round=round)

    # ── mesh operations: subdivide, smooth, add_random, simplify ─────────

    def subdivide(self, *, height_map_only: bool = False) -> "Mesh":
        """Subdivides mesh triangles (1 -> 4 midpoint split).

        Chainable: ``mesh.subdivide().subdivide()``.
        """
        from .mesh_ops import mesh_subdivide
        return mesh_subdivide(self, height_map_only=height_map_only)

    def smooth(self, strength: int = 50, *, height_map_only: bool = False) -> "Mesh":
        """Applies Laplacian smoothing to the mesh.

        Parameters:
            strength: Smoothing intensity, integer 1..100 (default 50).
            height_map_only: If True, applies only to top face of a heightmap mesh
                (raises ValueError if not generated by a heightmap).
        """
        from .mesh_ops import mesh_smooth
        return mesh_smooth(self, strength=strength, height_map_only=height_map_only)

    def add_random(
        self,
        seed: int = 42,
        strength: int = 500,
        walk: int = 0,
        *,
        height_map_only: bool = False,
    ) -> "Mesh":
        """Adds random perturbation with spatial neighbor correlation (walk).

        Parameters:
            seed: Random seed for deterministic reproducibility.
            strength: Peak perturbation amplitude in millimeters, integer 1..10,000,000 mm (default 500).
            walk: Neighbor correlation, integer 0..9 (default 0: uncorrelated white noise;
                9: 90% neighbor-diffused spatial correlation).
            height_map_only: If True, applies only to top face of a heightmap mesh
                (raises ValueError if not generated by a heightmap).
        """
        from .mesh_ops import mesh_add_random
        return mesh_add_random(
            self,
            seed=seed,
            strength=strength,
            walk=walk,
            height_map_only=height_map_only,
        )

    def simplify(self, strength: int = 50, *, height_map_only: bool = False) -> "Mesh":
        """Simplifies the mesh by reducing triangle and vertex count.

        Parameters:
            strength: Decimation intensity, integer 1..100 (default 50).
            height_map_only: If True, applies only to top face of a heightmap mesh
                (raises ValueError if not generated by a heightmap).
        """
        from .mesh_ops import mesh_simplify
        return mesh_simplify(self, strength=strength, height_map_only=height_map_only)


# =============================================================================
# PLANNING PRIMITIVES
# =============================================================================


class ReferencePoint(BimElement):
    """
    Survey marker, datum point, or guide point.

    IFC mapping:
      - survey/datum/setback/guide: IfcAnnotation (invisible metadata)
      - boundary/foundation: IfcBuildingElementProxy with visible 3D peg mesh

    Used for: GNSS survey markers, datum points, setback reference points,
    site boundary corners, foundation footprint corners.

    Parameters:
        location: 3D position (mm at script time)
        point_type: Classification of the point
        marker_color: Hex color for visible peg types (boundary/foundation)
        accuracy_mm: Measurement accuracy (stays in mm — metadata, not geometry)
        survey_method: How the point was captured
        crs: Coordinate reference system identifier
    """
    id: str = Field(default_factory=lambda: f"refpt_{uuid4().hex[:8]}")
    location: Point
    point_type: Literal["survey", "datum", "setback", "guide", "boundary", "foundation"] = "survey"
    marker_color: Optional[str] = None
    accuracy_mm: Optional[int] = None
    survey_method: Optional[str] = None
    crs: Optional[str] = None


class GuideLine(BimElement):
    """
    Grid axis, setback line, boundary, or utility corridor.

    IFC mapping:
    - line_type="grid" → IfcGrid with IfcGridAxis
    - Other types → IfcAnnotation with IfcPolyline

    Parameters:
        points: List of 3D points defining the line (min 2)
        line_type: Classification determining IFC routing
        label: Axis tag (e.g., "A", "1") for grid lines
        buffer_width: Width of corridor/easement (mm)
        source: Regulatory source (e.g., "Lokalplan_2025")
        constraint: Associated constraint (e.g., "NoConstruction")
    """
    id: str = Field(default_factory=lambda: f"guide_{uuid4().hex[:8]}")
    points: List[Point]
    line_type: Literal["grid", "boundary", "setback", "easement", "corridor"] = "boundary"
    label: Optional[str] = None
    buffer_width: Optional[int] = None
    source: Optional[str] = None
    constraint: Optional[str] = None

    @field_validator("points")
    @classmethod
    def validate_min_points(cls, v: List[Point]) -> List[Point]:
        if len(v) < 2:
            raise ValueError("GuideLine requires at least 2 points")
        return v


class Space(BimElement):
    """
    Programmatic zone / spatial reservation container.

    IFC mapping: IfcSpace

    TWO authoring modes, mutually exclusive (the layers doctrine — shortcut
    OR structure, never both):

    * **Declared bounds (WS-F, the primary form).** Name the bounding
      elements and the compiler DERIVES the volume; the declared members
      become the authoritative ``IfcRelSpaceBoundary1stLevel`` set::

          kitchen = Space(name="kitchen",
                          bounds=["wall:north", "wall:k1", "wall:k2",
                                  "slab:floor", "slab:deck"])
          storey.add(kitchen)
          kitchen.difference(Sweep(...))   # booleans compose onto the
                                           # derived volume

      Fragments resolve like ``Anchor(host=)`` — whole ``type:leaf`` pairs
      from the tail; a miss or an ambiguity is a compile error. Policy in
      :mod:`lite_step.compiler.space_volume`.

    * **Geometry mode.** ``.add(Box(...))`` authors the volume directly —
      LOD 100 planning zones have no walls to cite. Boundaries are inferred
      (WS-C).

    Booleans (``.union``/``.difference``) are legal in BOTH modes and bake
    into the IfcSpace representation; ``.void()`` on a Space is an error in
    both.

    Parameters:
        space_type: Programmatic classification
        bounds: Canonical-name fragments of the bounding elements
    """
    id: str = Field(default_factory=lambda: f"space_{uuid4().hex[:8]}")
    space_type: Optional[str] = None
    #: Declared bounding elements (WS-F), by canonical-name fragment. Set it
    #: and the compiler derives the volume; leave it ``None`` and the volume
    #: is the authored children. Never both — ``.add()``/``.anchor()`` refuse.
    bounds: Optional[List[str]] = None
    _elements: List = PrivateAttr(default_factory=list)

    @field_validator("bounds")
    @classmethod
    def validate_bounds(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        if not v:
            raise ValueError(
                "Space(bounds=[]) declares nothing — list at least one "
                "canonical-name fragment (e.g. bounds=[\"wall:north\"]), or "
                "drop bounds= and author the volume with .add(Box(...))."
            )
        for entry in v:
            if not isinstance(entry, str) or not entry:
                raise ValueError(
                    f"Space(bounds=...) got {entry!r} — bounds are "
                    f"canonical-name STRINGS (the reference relation, like "
                    f"members=), never objects and never empty."
                )
        return v

    def _refuse_bounds_conflict(self, verb: str) -> None:
        """The layers doctrine at the call site: declared bounds OR authored
        geometry, never both. Raised HERE (like ``.add()``'s child table) so
        the traceback points at the authoring line."""
        if self.bounds is not None:
            raise ValueError(
                f"Space '{self.name}' declares bounds= — its volume is "
                f"DERIVED from the declared bounding elements, so authored "
                f"child geometry via .{verb}() is mutually exclusive with "
                f"it (either the compiler derives the volume or you author "
                f"it, never both). Drop bounds= to author the volume, or "
                f"drop the .{verb}() call and declare the room's bounds. "
                f"Booleans (.union/.difference) stay legal on a bounds-mode "
                f"Space and compose onto the derived volume."
            )

    def add(self, *elements, carve: str = None) -> "Space":
        """Add child geometry to this space.

        IfcSpace keeps a table of its own (Box), for the reason IfcSite does:
        it is a spatial structure element with no ``Element(ifc_class=…)``
        twin, so there is no asymmetry to fix."""
        self._refuse_bounds_conflict("add")
        _add_children(self, elements, carve=carve)
        return self

    def anchor(self, child, *, along: int = 0, along_center: int = None,
               inset: int = 0, up: int = 0, rotations=None,
               carve: str = None) -> "Space":
        """Place ``child`` in THIS element's local frame.

        ``along`` runs the length, ``up`` the height, and ``out`` through the
        thickness with POSITIVE = outward from the building. Local (0, 0, 0)
        is this element's lower-left-back corner. Contrast ``.add()``, which
        is pure containment and never moves the child.

        ``child`` may be ANY element type — geometry primitive or container.
        See :func:`_anchor_child` for why there is no table.

        ``carve`` is the same argument ``.add()`` takes and means the same
        thing. **Anchoring already grants half of it**: an anchored child is
        out of the inferred-carve pairing by construction, because its position
        is a relationship to its host rather than an accidental overlap — which
        is the whole of what ``carve="none"`` says on an ``.add()``. Stating it
        here extends that to the other carve an anchored child can still take,
        the OPENING/VOID one: a hole removes matter regardless of how the
        matter got there, and ``carve="none"`` is the author saying THIS matter
        is meant to be in it — a sill or a lintel standing in the reveal.

        So ``"other"`` and ``"self"`` are accepted and INERT: the pairing they
        name does not happen for an anchored child at all, so there is no
        direction to state. Refused instead, one argument would mean different
        things depending on which verb you wrote it on, and the author would
        have to remember which subset each accepts.
        """
        self._refuse_bounds_conflict("anchor")
        return _anchor_child(self, child,
                             along, inset, up, rotations, along_center,
                             carve=carve)



# =============================================================================
# BOOLEAN-OPERAND CONSUMPTION (DSL v1.5, WS1 PR-E)
# =============================================================================


def _miter_world_verts(elem):
    """World-space vertices of ``elem`` (placement applied) for miter geometry.

    ``_query_triangles`` is the exact, world-space tessellation the geometry
    queries use — it respects ``placement=Transform`` (rotation included), so
    an element authored at any orientation contributes its true oriented
    footprint, not an axis-aligned box. Reads the AUTHORED shape (clips from an
    earlier miter on the same element are not reflected), so repeated miters on
    one element derive from a consistent footprint.

    **A CONTAINER contributes its parts, stacked.** ``_query_triangles`` refuses
    one ("Wall is not a geometric primitive"). Mitering the BODIES instead
    cuts the concrete and leaves every bar where it was:
    1007 m of 6941 m (14.5%) of a hospital arm's reinforcement outside its own
    sector. The joint belongs to the ASSEMBLY, so the footprint it is derived
    from has to be the assembly's; deriving per-part would give a bar near the
    flank a different bisector from the bar beside it.

    A part that cannot tessellate standalone (an unresolved anchor, or a
    ``Bar``/``Pipe`` whose mesh the guard refuses) contributes nothing
    rather than killing the derivation — the assembly's other parts still
    describe the run. **It says so, naming the part**, because the angle it
    was excluded from is not recoverable from the output: a joint derived from
    the concrete alone and one derived from concrete-plus-steel are both
    plausible bisectors, and nothing downstream records which you got. This is
    the compiler's own rule (loud failure over silent degradation) applied to
    the one place that swallows an exception on purpose.

    A container with NO tessellable part yields an empty array, and the caller
    turns that into the same loud refusal a collapsed footprint gets."""
    import numpy as np

    from lite_step.compiler.composite_clip import iter_parts

    chunks = []
    for part in iter_parts(elem):
        try:
            verts, _ = part._query_triangles()
        except Exception as exc:  # noqa: BLE001
            _miter_logger.warning(
                "miter: %s '%s' contributed nothing to the joint angle — it "
                "could not be tessellated standalone (%s: %s). The bisector "
                "comes from the assembly's other parts.",
                type(part).__name__,
                getattr(part, "name", None) or getattr(part, "id", "?"),
                type(exc).__name__, exc,
            )
            continue
        if len(verts):
            chunks.append(np.asarray(verts, float))
    if not chunks:
        return np.empty((0, 3), float)
    return np.vstack(chunks)


def _miter_run_dir(verts, at_p, edge_u):
    """The element's RUN direction from the joint, in the plane ⟂ ``edge_u``.

    Projects the element's world vertices onto the plane perpendicular to the
    joint edge, then takes the **oriented long axis** of that footprint (the
    principal component — an OBB axis, not the AABB centre). This is the wall's
    true run direction regardless of where ``at`` sits along it or how asymmetric
    the element is about the corner — the corner→centroid heuristic skewed the
    angle for a corner at a wall's END. Sign is set by the centroid so the axis
    points INTO the element from ``at``. Falls back to the centroid direction for
    a non-elongated footprint (near-square: no distinct long axis).

    Returns a unit 3-vector, or ``None`` if the projected footprint collapses to
    a point at ``at`` (no in-plane direction — a degenerate miter)."""
    import numpy as np
    Vp = verts - np.outer(verts @ edge_u, edge_u)      # project ⟂ edge
    ap = at_p - (at_p @ edge_u) * edge_u
    centroid_dir = Vp.mean(axis=0) - ap
    cd_norm = float(np.linalg.norm(centroid_dir))
    C = Vp - Vp.mean(axis=0)
    w, vecs = np.linalg.eigh(C.T @ C)                  # ascending eigenvalues
    elongated = w[-1] > 1e-9 and (w[-1] - w[-2]) > 1e-6 * w[-1]
    if elongated:
        axis = vecs[:, -1]
        axis = axis - (axis @ edge_u) * edge_u
        n = float(np.linalg.norm(axis))
        if n > 1e-9:
            axis = axis / n
            if cd_norm > 1e-9 and float(np.dot(axis, centroid_dir)) < 0:
                axis = -axis
            return axis
    # near-square footprint (or degenerate axis): fall back to centroid dir
    if cd_norm < 1e-6:
        return None
    return centroid_dir / cd_norm


# Monotonic source of authored-joint ids (see miter() + BimElement._miter_joints).
# Module-level: ids only need to be unique within one authoring pass, and a plain
# int survives normalize's deepcopy verbatim.
_miter_joint_counter = itertools.count()


def miter(a: "BimElement", b: "BimElement", *, at: Point,
          edge: Tuple[float, float, float],
          overrun: float = 0) -> "Tuple":
    """Miter two elements meeting at a joint — the derived angle cut.

    Both ``a`` and ``b`` are clipped by the SHARED bisector plane through the
    joint, so each keeps exactly its half and the two faces meet flush. A
    miter is fixed by a POINT and a VECTOR, both required: ``at`` is the joint
    corner, and ``edge`` is the direction of the shared joint edge (the line
    the two elements meet along) — for two upright wall/cladding leaves that
    is vertical, ``edge=(0, 0, 1)``. The cut *angle* is still DERIVED — the
    directions from ``at`` toward each element's centroid, **projected onto
    the plane perpendicular to** ``edge``, define the inner angle; the
    bisector splits it in half. A 90° corner yields two 45° faces; a 120°
    corner two 60°.

    Projecting onto the edge-perpendicular plane is what keeps the plane in
    the joint (vertical for a vertical ``edge``) and the angle clean:
    deriving the edge from the raw 3D centroid directions instead (as v1 did)
    let the ``at``-at-base vs centroid-at-mid-height offset tilt the plane out
    of vertical and skew the angle. ``edge`` has no default on purpose — the
    joint line is a property of the geometry the author knows, not something
    to guess (pass ``(0,0,1)`` for an upright corner, a horizontal vector for
    a raked corner or two roof planes meeting at a hip).

    **The angle is derived at COMPILE, not here.** This call records the joint;
    ``compiler.composite_clip`` derives the bisector from both assemblies as
    they finally stand. That is what makes the result independent of where the
    line sits in the file — measured, because the alternative is not
    hypothetical: deriving eagerly, the same joint came out at 45° with a
    return wing added before this line and 33.8° with it added after, from
    identical source, with nothing recording which you got. A part that stays
    INSIDE the existing footprint (reinforcement, typically) does not move the
    angle either way.

    Deriving late costs one thing, and it is guarded rather than accepted: a
    recorded joint is only resolved by ``normalize_project_to_meters``, so a
    path reaching the generator without it would lose the cut in silence.
    ``generate_ifc`` refuses a model carrying an unresolved joint.

    ``overrun`` (mm) is passed through to both clips: the concrete is cut on
    the bisector while reinforcement laps ``overrun`` past it into the
    neighbour. See ``.clip()``; the lap LENGTH is not the compiler's to know.

    ``at`` and ``edge`` are both required. A zero ``edge`` raises HERE (nothing
    later can make it valid). An element with no geometry, one whose footprint
    lies ON the joint edge, and two elements running parallel in the
    edge-perpendicular plane all raise at COMPILE — each depends on geometry
    that is still being authored when this returns. Returns ``(a, b)``, so
    ``a, b = miter(a, b, at=c, edge=(0,0,1))`` reads naturally.
    """
    import numpy as np

    if np.linalg.norm(np.array(edge, float)) < 1e-9:
        raise ValueError(
            "miter: edge= is a zero vector — pass the joint edge direction "
            "((0,0,1) for an upright wall corner, a horizontal vector for a "
            "raked/roof joint).")
    if overrun < 0:
        raise ValueError(
            f"miter: overrun={overrun} must be >= 0 — it lets reinforcement "
            f"LAP past the joint into the neighbour; pulling it short of the "
            f"joint is a shorter bar, not a miter."
        )
    # Checked HERE, not on the HalfSpace: the derivation builds that object via
    # model_construct (its inputs are derived, and the strict int-mm gate
    # refuses the floats a derivation produces), so its own validators never
    # run on this path. A refusal the author can act on belongs at the line
    # they wrote anyway.

    # Recorded, not derived. The remaining refusals (a footprint with no
    # geometry, one collapsed onto the joint edge, two elements running
    # parallel) all depend on geometry that is still being authored, so they
    # fire from derive_miter_clips at compile. A zero edge= is a typo in this
    # line and nothing later can make it valid, so it raises HERE.
    # Record the authored joint: the miter clips ARE the corner resolution, so
    # the two must NOT also auto-carve each other under displacement (a redundant
    # near-no-op carve that only deepens the CSG — web-ifc drops the deepest
    # chain). A shared int id on both marks the pair; displacement skips any pair
    # whose id sets intersect. See displacement._are_mitered.
    jid = next(_miter_joint_counter)
    a._miter_joints.append(jid)
    b._miter_joints.append(jid)
    # The same id pairs the two half-requests, so the resolver can derive the
    # joint's ONE plane once and retire both sides together.
    edge_t = tuple(float(v) for v in edge)
    a._pending_miters.append(_MiterRequest(
        at=at, edge=edge_t, partner=b, overrun=float(overrun), jid=jid))
    b._pending_miters.append(_MiterRequest(
        at=at, edge=edge_t, partner=a, overrun=float(overrun), jid=jid))
    return a, b


class _MiterRequest:
    """One side of a recorded joint, waiting for its angle.

    ``__slots__`` rather than a pydantic model on purpose: this never crosses
    a boundary, never validates, and is created twice per ``miter()`` call in
    a hot authoring loop.
    """
    __slots__ = ("at", "edge", "partner", "overrun", "jid")

    def __init__(self, *, at, edge, partner, overrun=0.0, jid=None):
        self.at = at
        self.edge = edge
        self.partner = partner
        self.overrun = overrun
        self.jid = jid


def derive_miter_pair(elem, req):
    """The joint's ONE plane, as ``(half-space for elem, for the partner)``.

    Split out of ``miter()`` so the DERIVATION can run at compile while the
    CALL stays where the author wrote it. Reads both sides' footprints as they
    finally stand, so the answer does not depend on line order — see
    ``miter()``.

    **Both sides come out of a single derivation, and that is not tidiness.**
    Deriving each side separately meant tessellating the partner a second time
    — by which point the first side had already taken its clip, so the second
    reading saw a CLIPPED footprint. Measured on the corpus roof: 13 of 26
    elements came back with a different vertex count on the second call (8 -> 10
    — a clipped box has more corners), and the two sides of one joint stopped
    being exact opposites (``[0,-1,0]`` against ``[0,-1,-9.9e-05]``). A joint
    has one plane; deriving it twice is what let the second reading see the
    first's cut.
    """
    import numpy as np

    at = req.at
    p = np.array([at.x, at.y, at.z], float)
    e = np.array(req.edge, float)
    e = e / np.linalg.norm(e)

    def _dir(elem, label):
        verts = _miter_world_verts(elem)
        if not len(verts):
            raise ValueError(
                f"miter: {label} ({type(elem).__name__}) has no geometry to "
                f"derive a joint angle from — a container must hold its parts "
                f"before it can be mitered, because the bisector comes from "
                f"the assembly's footprint.")
        d = _miter_run_dir(verts, p, e)
        if d is None:
            raise ValueError(
                f"miter: {label}'s footprint collapses onto the joint edge "
                f"through at={at} — no in-plane run direction to derive the "
                f"joint angle from. Check at= is the corner and edge= the "
                f"joint line.")
        return d

    a1 = _dir(elem, "this element")
    a2 = _dir(req.partner, "the partner")
    if np.linalg.norm(a1 + a2) < 1e-9 or np.linalg.norm(np.cross(a1, a2)) < 1e-9:
        raise ValueError(
            "miter: the two elements run parallel (or anti-parallel) in the "
            "edge-perpendicular plane — they share no joint angle, so there "
            "is no bisector plane to cut. Miter is for elements meeting at an "
            "ANGLE.")
    m = a1 + a2
    m /= np.linalg.norm(m)
    n = np.cross(m, e)                       # plane contains the edge + bisector
    n /= np.linalg.norm(n)
    # The clip normal points AWAY from this element's own centroid, i.e. at the
    # overshoot past the bisector that must go. dot(n, a1) picks the side.
    #
    # Derived per SIDE from the SAME pair of directions, so the two sides get
    # opposite normals on one plane — a single derivation handing out both
    # would have to be told which element it is answering for anyway, and this
    # way each side's answer is recomputed from the geometry rather than
    # remembered from the other's call.
    # Each side's clip normal points AWAY from its own run direction, i.e. at
    # the overshoot past the bisector that must go. One plane, two signs — so
    # the pair is antipodal BY CONSTRUCTION rather than by two derivations
    # happening to agree.
    s = float(np.dot(n, a1))
    own = -np.sign(s) * n
    overrun = getattr(req, "overrun", 0.0) or 0.0

    # ``model_construct``: every input here is DERIVED, not authored — the
    # normal is a unit cross product (non-zero by construction, and the
    # parallel check above is what guarantees it), the origin was validated
    # when ``miter()`` took it, and the overrun when it was passed. Running the
    # authoring validators over compiler output is the trap this repo has
    # already named: the strict int-mm gate refuses a float, and float is what
    # a derivation produces. The same reason ``_normalize_clips_recursive``
    # rebuilds this exact class that way.
    #
    # NOT rounded. Snapping the direction to 4 places was tried: it is inert,
    # because ``_apply_clips`` re-normalizes to unit length before emitting, so
    # the snapped value comes back out at full width. It cost precision in the
    # DSL value and changed nothing downstream.
    def _hs(vec):
        return HalfSpace.model_construct(
            origin=at,
            normal=tuple(float(c) for c in vec),
            overrun=overrun,
        )

    return _hs(own), _hs(-own)


def collect_consumed_operand_ids(project) -> set:
    """Python object ids of every element consumed as a boolean operand.

    An element passed to ``.difference()`` / ``.union()`` / ``.fills()`` is merged
    into the consumer's shape — it must never ALSO render standalone. The
    generator skips (and ``validate_project_report`` warns about) any
    storey element whose ``id()`` is in this set. Object identity (not the
    ``id``/``name`` fields) is the right key: a derived copy is a different
    object and legitimately renders on its own. ``deepcopy`` of the whole
    Project (the meters-normalizer) preserves shared references via its
    memo, so the identity relation survives normalization.
    """
    consumed: set = set()

    def _walk(elem) -> None:
        for list_name in ("_cuts", "_adds", "_intersects", "_fills", "_voids"):
            for op in (getattr(elem, list_name, None) or []):
                consumed.add(id(op))
                _walk(op)
        for child in (getattr(elem, "_elements", None) or []):
            _walk(child)
        for child in (getattr(elem, "_openings", None) or []):
            _walk(child)

    for storey in project.storeys:
        for elem in storey.elements:
            _walk(elem)
    for site in (getattr(project, "sites", None) or []):
        _walk(site)                                   # site.void()/.difference() operands
    return consumed


# =============================================================================
# FORWARD REFERENCE RESOLUTION
# =============================================================================
# Resolve forward references for geometry primitives
Box.model_rebuild()
Extrude.model_rebuild()
Sweep.model_rebuild()
Pipe.model_rebuild()
Revolve.model_rebuild()
Bar.model_rebuild()
Element.model_rebuild()
Mesh.model_rebuild()
# Resolve forward references for assembly classes
Door.model_rebuild()
Window.model_rebuild()
Wall.model_rebuild()
Site.model_rebuild()
# Resolve forward references for semantic element classes
Column.model_rebuild()
Beam.model_rebuild()
Slab.model_rebuild()
Roof.model_rebuild()
# Resolve forward references for planning primitives
ReferencePoint.model_rebuild()
GuideLine.model_rebuild()
Space.model_rebuild()
