"""Predicted CSG (boolean) nesting depth, per element, from the MODEL tree.

**Why a metric at all.** Boolean depth is the one geometry property that fails
INVISIBLY. Past ~13 nested ``IfcBooleanResult`` levels the primary web viewer
stops evaluating the chain and renders the un-subtracted base solid — every
pocket, ledge and notch gone, no error in any log. A model can be perfectly
valid IFC, open fine in a desktop viewer, and be wrong in the app. So the
compiler measures the depth it is about to emit and says so out loud, every
compile, whether or not anything is wrong.

**Budget.** ``boolean_tree.CSG_DEPTH_BUDGET`` (10), set below the measured ~13
tolerance so a model has headroom. Exceeding it is a WARNING, never a compile
error — a deep model must still compile and still download; the author needs to
know, not to be blocked. (Explicit product decision.)

The prediction is computed from the DSL tree, after the displacement pass has
added its inferred carves, so it holds for whatever emits. The depth ARITHMETIC is
imported from :mod:`lite_step.ifc.boolean_tree` — the same functions the
emitters are built on — so the prediction cannot silently drift from the
emitted tree. ``lite_step/tests/test_csg_depth.py`` pins the two against each
other by walking the real emitted model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from lite_step.compiler.walkguard import WalkGuard
from lite_step.ifc.boolean_tree import (
    CSG_DEPTH_BUDGET, CSG_WIDTH_BUDGET, balanced_depth, chain_depth,
)

logger = logging.getLogger(__name__)


@dataclass
class ElementDepth:
    """One element's predicted boolean nesting depth + why it is that deep."""

    name: str
    depth: int
    #: Total IfcBooleanResult nodes this element's Body representation emits.
    #: Depth's sibling metric, NOT a function of it — a mitered sweep is wide
    #: and shallow (balanced), a stack of carves is narrow and deep.
    width: int = 0
    cuts: int = 0
    adds: int = 0
    intersects: int = 0
    clips: int = 0
    segments: int = 0

    def breakdown(self) -> str:
        parts = []
        for label, n in (("cuts", self.cuts), ("adds", self.adds),
                         ("intersects", self.intersects), ("clips", self.clips),
                         ("segments", self.segments)):
            if n:
                parts.append(f"{n} {label}")
        return ", ".join(parts) or "no operands"


@dataclass
class CsgDepthReport:
    """Model-wide CSG depth, plus the elements that carry it."""

    elements: List[ElementDepth] = field(default_factory=list)
    budget: int = CSG_DEPTH_BUDGET
    width_budget: int = CSG_WIDTH_BUDGET

    @property
    def max_depth(self) -> int:
        return max((e.depth for e in self.elements), default=0)

    @property
    def max_width(self) -> int:
        return max((e.width for e in self.elements), default=0)

    @property
    def over_budget(self) -> List[ElementDepth]:
        return [e for e in self.elements if e.depth > self.budget]

    @property
    def over_width_budget(self) -> List[ElementDepth]:
        return [e for e in self.elements if e.width > self.width_budget]

    def stats_line(self, top: int = 3) -> str:
        """The one line printed on EVERY compile — silence here would mean the
        only signal is the failure case, and depth creeps up long before it
        breaks."""
        ranked = sorted(self.elements, key=lambda e: (-e.depth, e.name))
        head = (f"csg depth: max={self.max_depth} (budget {self.budget})"
                f"  width: max={self.max_width} (budget {self.width_budget})")
        listed = [e for e in ranked if e.depth > 0][:top]
        if not listed:
            return head
        return head + " — top: " + ", ".join(f"{e.name}={e.depth}" for e in listed)

    def warnings(self) -> List[str]:
        """One ``warning:``-worthy string per over-budget element."""
        out = []
        for e in sorted(self.over_budget, key=lambda e: (-e.depth, e.name)):
            out.append(
                f"csg depth {e.depth} exceeds the budget of {self.budget} on "
                f"{e.name} ({e.breakdown()}) — web viewers stop evaluating deep "
                f"boolean chains and silently render the UNCARVED base solid; "
                f"merge or drop operands, or split the element"
            )
        for e in sorted(self.over_width_budget, key=lambda e: (-e.width, e.name)):
            out.append(
                f"csg width {e.width} boolean operations on {e.name} "
                f"({e.breakdown()}) — the geometry is CORRECT but evaluation "
                f"cost grows superlinearly: expect minutes in ifcopenshell and "
                f"an apparent hang in Blender Bonsai, with no error anywhere. "
                f"For a tube along a dense polyline use a Mesh; for a building "
                f"member reduce the joint count"
            )
        return out


# ---------------------------------------------------------------------------
# Per-element prediction — mirrors the emitters in lite_step/ifc/generator.py
# ---------------------------------------------------------------------------


def _n(seq) -> int:
    return len(seq or [])


def _label(elem) -> str:
    name = getattr(elem, "ifc_name", None) or getattr(elem, "_canonical_name", None)
    return name or f"<anonymous {type(elem).__name__}>"


def _segment_item_depths(path, fillet_radius) -> List[int]:
    """Per-segment depth of a mitered swept path: one half-space DIFFERENCE per
    joint the segment touches (0 at a free end, 1 or 2 otherwise).

    Mirrors ``_profile_path_items`` / ``_create_disk_solid_boolean_safe``.
    """
    pts = [(float(p.x), float(p.y), float(p.z)) if hasattr(p, "x") else tuple(p)
           for p in (path or [])]
    if len(pts) < 2:
        return [0]
    closed = len(pts) >= 4 and pts[0] == pts[-1]
    if fillet_radius and not closed:
        try:
            from lite_step.ifc.geometry import facet_fillet_path
            pts = [tuple(p) for p in facet_fillet_path(pts, float(fillet_radius))]
        except Exception:  # pragma: no cover - construction already validates
            pass
    n_seg = len(pts) if closed else len(pts) - 1
    if closed:
        return [2] * n_seg
    if n_seg == 1:
        return [0]
    return [1] + [2] * (n_seg - 2) + [1]


def _base_depth(elem, *, as_operand: bool) -> tuple:
    """``(depth, n_segments)`` of the element's base solid, before its own
    boolean classes are applied."""
    tname = type(elem).__name__
    path = getattr(elem, "path", None)
    if tname == "Sweep" and _n(path) >= 2:
        # Planar closed rect-section frames compile to boolean-free prisms and
        # (as an ELEMENT) are left as multiple items — no union at all.
        from lite_step.ifc.generator import _resolve_profile_section
        section = None
        try:
            section = _resolve_profile_section(elem)
        except Exception:  # pragma: no cover - validation reports this
            pass
        pts = [(float(p.x), float(p.y), float(p.z)) for p in path]
        closed = len(pts) >= 4 and pts[0] == pts[-1]
        fillet = getattr(elem, "fillet_radius", 0) or 0
        rot = getattr(elem, "profile_rotation", 0) or 0
        if (closed and not fillet and section is not None
                and section[0] == "rect" and not rot):
            return (0, len(pts) - 1) if as_operand else (0, 0)
        seg = _segment_item_depths(path, fillet)
        return balanced_depth(seg), len(seg)
    if tname in ("Pipe", "Bar"):
        # The boolean-safe cylinder form only kicks in when the element itself
        # carries booleans; otherwise it stays one IfcSweptDiskSolid.
        has_own = bool(getattr(elem, "_cuts", None) or getattr(elem, "_adds", None)
                       or getattr(elem, "_intersects", None))
        if as_operand or has_own:
            fillet = (getattr(elem, "fillet_radius", None)
                      if tname == "Pipe" else getattr(elem, "bend_radius", None))
            seg = _segment_item_depths(path, fillet or 0)
            return balanced_depth(seg), len(seg)
    return 0, 0


def _segment_width(path, fillet) -> int:
    """Booleans a mitered swept path emits: one half-space DIFFERENCE per
    joint each segment touches, plus the unions joining the segments.

    Does NOT use ``len(_segment_item_depths(...))`` as the segment count.
    That helper sets ``n_seg = len(pts)`` for a closed path, counting the
    duplicated endpoint as a segment — 5 entries for a 4-segment loop. Depth
    never notices, because ``balanced_depth`` is logarithmic and one extra
    equal operand does not move it. A SUM notices immediately: it was exactly
    the constant +3 error this function was written to fix.
    """
    seg = _segment_item_depths(path, fillet)
    pts = [(float(p.x), float(p.y), float(p.z)) if hasattr(p, "x") else tuple(p)
           for p in (path or [])]
    closed = len(pts) >= 4 and pts[0] == pts[-1]
    n_items = len(seg) - 1 if (closed and len(seg) > 1) else len(seg)
    return sum(seg[:n_items]) + max(0, n_items - 1)


def _base_width(elem, *, as_operand: bool) -> int:
    """Boolean COUNT of the element's base solid.

    Mirrors ``_base_depth`` exactly, but sums the per-segment half-spaces
    instead of balancing them, and adds the unions that join the segments
    into one solid. For a closed mitered sweep of ``n`` segments that is
    ``2n`` differences + ``n-1`` unions = ``3n-1`` — verified against emitted
    files at n = 4, 16, 96 and 220 (11, 47, 287, 659).
    """
    tname = type(elem).__name__
    path = getattr(elem, "path", None)
    if tname == "Sweep" and _n(path) >= 2:
        from lite_step.ifc.generator import _resolve_profile_section
        section = None
        try:
            section = _resolve_profile_section(elem)
        except Exception:  # pragma: no cover - validation reports this
            pass
        pts = [(float(p.x), float(p.y), float(p.z)) for p in path]
        closed = len(pts) >= 4 and pts[0] == pts[-1]
        fillet = getattr(elem, "fillet_radius", 0) or 0
        rot = getattr(elem, "profile_rotation", 0) or 0
        # The boolean-free prism form: multiple items, no union, no booleans.
        if (closed and not fillet and section is not None
                and section[0] == "rect" and not rot):
            return 0
        return _segment_width(path, fillet)
    if tname in ("Pipe", "Bar"):
        has_own = bool(getattr(elem, "_cuts", None) or getattr(elem, "_adds", None)
                       or getattr(elem, "_intersects", None))
        if as_operand or has_own:
            fillet = (getattr(elem, "fillet_radius", None)
                      if tname == "Pipe" else getattr(elem, "bend_radius", None))
            return _segment_width(path, fillet or 0)
    return 0


def _operand_width(operand) -> int:
    """Booleans an OPERAND contributes: its own base + clips + sub-operands,
    plus the ONE node joining it to the host."""
    w = _base_width(operand, as_operand=True)
    w += _n(getattr(operand, "_clips", None))
    for sub in ("_cuts", "_adds"):
        w += sum(_operand_width(o) for o in (getattr(operand, sub, None) or [])
                 if _operand_emits(o))
    return w + 1


def _operand_emits(operand) -> bool:
    """Whether ``_create_operand_base_solid`` can realise this operand at all.

    An operand it cannot build (a contour ``Extrude``, a ``Mesh``, a sectionless
    ``Sweep``) emits NO boolean — it is warned about and skipped at build time.
    Counting it here would over-predict depth, and an over-predicting metric
    gets ignored just as fast as an under-predicting one.
    """
    tname = type(operand).__name__
    if tname in ("Revolve", "Pipe", "Bar"):
        return True
    if tname == "Sweep" and _n(getattr(operand, "path", None)) >= 2:
        try:
            from lite_step.ifc.generator import _resolve_profile_section
            return _resolve_profile_section(operand) is not None
        except Exception:  # pragma: no cover - validation reports this
            return False
    return (getattr(operand, "start", None) is not None
            and getattr(operand, "end", None) is not None)


def _operand_depths(operands) -> List[int]:
    return [_operand_depth(o) for o in (operands or []) if _operand_emits(o)]


def _operand_depth(operand) -> int:
    """Depth of a boolean OPERAND as ``_create_hole_solid_local`` builds it:
    base → its own ``.clip()`` half-spaces → sub-cuts → sub-adds."""
    base, _segs = _base_depth(operand, as_operand=True)
    base += _n(getattr(operand, "_clips", None))
    return chain_depth(
        base,
        _operand_depths(getattr(operand, "_cuts", None)),
        _operand_depths(getattr(operand, "_adds", None)),
    )


def element_depth(elem) -> ElementDepth:
    """Predicted boolean nesting depth of one element's Body representation.

    Emission order (pinned, semantically load-bearing): base → cuts → adds →
    intersects → clips.
    """
    cuts = getattr(elem, "_cuts", None) or []
    adds = getattr(elem, "_adds", None) or []
    intersects = getattr(elem, "_intersects", None) or []
    clips = getattr(elem, "_clips", None) or []
    base, segments = _base_depth(elem, as_operand=False)
    cut_d = _operand_depths(cuts)
    add_d = _operand_depths(adds)
    int_d = _operand_depths(intersects)
    depth = chain_depth(base, cut_d, add_d, int_d) + len(clips)
    width = _base_width(elem, as_operand=False) + len(clips)
    for group in (cuts, adds, intersects):
        width += sum(_operand_width(o) for o in group if _operand_emits(o))
    return ElementDepth(name=_label(elem), depth=depth, width=width,
                        cuts=len(cut_d), adds=len(add_d), intersects=len(int_d),
                        clips=len(clips), segments=segments)


# ---------------------------------------------------------------------------
# Model walk
# ---------------------------------------------------------------------------


def _walk(elem, out: List[ElementDepth], _on_path=None) -> None:
    """Collect every solid's predicted depth, depth-first.

    On-path re-entry (a containment cycle) is PRUNED: this walk reports a
    METRIC, and a metric that blows the stack reports nothing. A cyclic model
    is refused by ``validate_project_report`` ahead of this.
    """
    from lite_step.models import taxonomy as tx

    guard = _on_path if _on_path is not None else WalkGuard()
    if not guard.enter(elem):
        return
    try:
        if tx.is_solid(elem):
            d = element_depth(elem)
            if (d.depth or d.width or d.cuts or d.adds
                    or d.intersects or d.clips):
                out.append(d)
        for child in (getattr(elem, "_elements", None) or []):
            _walk(child, out, guard)
        for opening in (getattr(elem, "_openings", None) or []):
            _walk(opening, out, guard)
    finally:
        guard.leave(elem)


def predict_csg_depth(project) -> CsgDepthReport:
    """Walk the model and predict every element's boolean nesting depth.

    Run AFTER ``apply_displacement`` — most real depth comes from inferred
    carves, not from authored ``.difference()`` calls, so predicting before the
    displacement pass measures the wrong tree.
    """
    report = CsgDepthReport()
    for storey in getattr(project, "storeys", None) or []:
        for elem in storey.elements:
            _walk(elem, report.elements)
    for site in getattr(project, "sites", None) or []:
        _walk(site, report.elements)
    return report
