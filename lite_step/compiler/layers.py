"""Layer slicing — geometry-bearing ``LayerSet`` on planar elements.

A layered element authors its body envelope ONCE (a single Box/Extrude child);
the compiler slices that body along the through-axis into stacked per-layer
solids. Each layer IS a ``Material`` at a thickness; the list is fixed
outer→inner and ``LayerSet.outward`` (a required inner→outer normal) places it —
``layer_axis_and_dir`` reads the normal to put the first-listed (outer) layer on
the face it points at. A cavity material (registry ``form="cavity"``, e.g.
``Cavity_Ventilated``) occupies its span but yields NO solid — a literal gap
between neighbours (``IsVentilated`` derives from the registry product).

Slice thicknesses scale proportionally to fill the body's actual through
extent (``validate_project`` warns when the mismatch exceeds 5% — geometry
stays authoritative, the buildup never leaves gaps or overhangs).

Consumers:

* ``_create_wall_box`` builds one multi-item Body from ``box_layer_spans``.
* ``_create_slab`` / ``_create_roof`` / ``_create_element_generic`` replace
  the single body child with ``box_layer_slices`` / ``extrude_layer_slices``
  (each slice an anonymous derived child carrying ``material=layer.key``) and
  ride the existing aggregated-children emission unchanged.

All derivations use ``model_copy`` (non-validating) — safe on normalized
float geometry, same pattern as the displacement engine's cut copies.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def layered_body_child(elem) -> Optional[object]:
    """The single Box/Extrude body child of a layered element, or ``None``.

    ``validate_project_report`` ERRORs when a layered element has zero or
    multiple solid children; this helper is the tolerant runtime counterpart
    (returns None → caller emits unsliced with a warning).

    An ANCHORED solid is not a candidate body — it is a detail placed in this
    element's frame (a cornice, a sign, a cladding piece), and the body is what
    defines the frame the detail is placed against. Counting it would make
    ``.anchor()`` unusable on any layered wall, which is exactly what the plain
    ``Wall`` body check already recognises."""
    from lite_step.models import taxonomy as tx

    solids = [c for c in (getattr(elem, "_elements", None) or [])
              if tx.is_prism(c) and not is_anchored(c)]
    return solids[0] if len(solids) == 1 else None


def is_anchored(elem) -> bool:
    """True for a child placed by ``.anchor()``, before OR after the bake.

    Both markers are read because the bake CONSUMES ``_anchor_spec``:
    validation runs pre-bake and sees the spec, the generator runs post-bake
    and sees only ``_anchor_resolved`` (the same asymmetry the displacement
    exemption keys on)."""
    return (getattr(elem, "_anchor_spec", None) is not None
            or bool(getattr(elem, "_anchor_resolved", False)))


def through_axis_for(elem, body) -> str:
    """The through-thickness axis ('x'|'y'|'z') for a layered element.

    Walls (incl. ``Element(ifc_class="IfcWall"/"IfcCurtainWall")``) → the
    horizontal axis with the SMALLER extent (the wall's thickness). Every
    other planar archetype (slab/roof/plate/covering/pavement) → z.
    """
    cls = type(elem).__name__
    ifc_class = getattr(elem, "ifc_class", None)
    is_wall = cls == "Wall" or ifc_class in ("IfcWall", "IfcCurtainWall")
    if not is_wall:
        return "z"
    # The through-axis is the one ACROSS the run axis. This was an inline
    # ``sx < sy`` comparison — equivalent to, but spelled as the complement
    # of, the generator's ``x_span >= y_span`` rule. Reading the single
    # derivation removes the chance of the two drifting apart.
    from lite_step.compiler.frames import opening_axes

    axes = opening_axes(body)
    if axes is None:
        return "y"
    along, _thickness_dir = axes
    return "y" if along[0] else "x"


def layer_axis_and_dir(elem, body) -> Tuple[str, bool]:
    """``(axis, outer_at_max)`` for a layered element from ``layers.outward``.

    ``axis`` is the body's through/thin axis (``through_axis_for``); the placement
    normal ``layers.outward`` must point ALONG it (across the layers). ``outer_at_max``
    is True when the normal points toward +axis — the first-listed (outer) layer
    then sits on the axis MAX face and the stack runs inward toward MIN.

    v1 requires an axis-aligned ``outward`` whose dominant axis IS the through-axis
    (a normal along the wall's length can't stack layers within the thin body; an
    oblique normal is a full-transform follow-up) — both raise loud.
    """
    axis = through_axis_for(elem, body)
    n = elem.layers.outward
    ai = "xyz".index(axis)
    dom = max(range(3), key=lambda i: abs(n[i]))
    others = [abs(n[i]) for i in range(3) if i != dom]
    if max(others, default=0.0) > 1e-6 * abs(n[dom]):
        raise ValueError(
            f"LayerSet.outward {n} must be axis-aligned in v1 — an oblique "
            f"buildup on a rotated element is a follow-up (full layer transform)")
    if dom != ai:
        raise ValueError(
            f"LayerSet.outward {n} points along {'xyz'[dom]}, but the layered "
            f"body's through-axis is {axis} — the normal must point ACROSS the "
            f"layers (the thin dimension), e.g. (0,-1,0) for a wall thin in y")
    return axis, n[ai] > 0


def layer_outer_at_max(elem) -> bool:
    """Whether the first-listed (outer) layer sits at the through-axis MAX face
    — the same orientation the slicer uses, for the usage chain's DirectionSense.
    Box → ``layer_axis_and_dir``; Extrude → ``outward``·extrusion-normal > 0.
    Returns False when the element has no single sliceable body (the caller emits
    a POSITIVE default usage; ``validate_project`` already flags such shapes)."""
    from lite_step.models import Extrude
    body = layered_body_child(elem)
    if body is None:
        return False
    if isinstance(body, Extrude):
        normal = contour_normal(body.contour)
        if normal is None:
            return False
        out = elem.layers.outward
        return (out[0] * normal[0] + out[1] * normal[1] + out[2] * normal[2]) > 0
    return layer_axis_and_dir(elem, body)[1]


def layer_spans(layers, lo: float, hi: float, *,
                reverse: bool = False) -> List[Tuple[object, float, float]]:
    """Per-layer spans covering ``[lo, hi]`` along the through-axis.

    Returns ``[(material, span_lo, span_hi), ...]``. The buildup list is
    outer→inner; ``reverse=False`` lands the first-listed (outer) material at
    ``lo`` (the MIN face), ``reverse=True`` lands it at ``hi`` (the MAX face) —
    driven by ``layer_axis_and_dir``'s ``outer_at_max``. Thicknesses scale
    proportionally onto the actual extent; the last span is snapped to ``hi`` so
    float drift never leaves a sliver. Units follow ``lo``/``hi`` (mm pre-normalize,
    meters at generation) — only thickness ratios are used.
    """
    seq = list(layers.layers)
    if reverse:
        seq = seq[::-1]
    total = sum(m.thickness_mm for m in seq)
    extent = hi - lo
    spans: List[Tuple[object, float, float]] = []
    cum = lo
    for i, mat in enumerate(seq):
        t = extent * (mat.thickness_mm / total)
        span_hi = hi if i == len(seq) - 1 else cum + t
        spans.append((mat, cum, span_hi))
        cum = span_hi
    return spans


def is_cavity(material) -> bool:
    """True when a layer material is a cavity (air gap) — its registry
    ``geometry.form`` is ``"cavity"``. Cavity layers hold their span in the
    stack but emit NO solid slice; ``IsVentilated`` rides the registry def."""
    from lite_step.materials import registry_definition
    mdef = registry_definition(material.key)
    return mdef is not None and mdef.geometry.form == "cavity"


def _derive_box_slice(body, axis: str, lo: float, hi: float, material) -> object:
    """An anonymous derived Box occupying ``[lo, hi]`` on ``axis``, carrying
    the layer material's key as its ``material=`` (so the existing registry
    path gives it Psets + render for free; an explicit body ``color=`` still
    wins the render for every slice, the material association riding along).
    Non-validating copies throughout."""
    s, e = body.start, body.end
    s_lo, e_hi = (lo, hi) if getattr(s, axis) <= getattr(e, axis) else (hi, lo)
    new_start = s.model_copy(update={axis: s_lo})
    new_end = e.model_copy(update={axis: e_hi})
    sl = body.model_copy(deep=True, update={
        "start": new_start, "end": new_end,
        "name": None, "material": material.key,
    })
    # model_copy(deep=True) carries private attrs — clear the copied
    # canonical (slices are anonymous; a copied Name would collide N-fold).
    sl._canonical_name = None
    # Same hygiene for the stamped frame: a slice is a generation-time
    # derivative with different bounds than the body it came from, so a
    # carried-over frame would describe the wrong geometry.
    sl._frame = None
    return sl


def box_layer_slices(elem, body) -> List[object]:
    """Per-layer Box slices for a layered container (slab/roof/Element path).

    Cavity materials yield no slice (air gap). Each slice inherits the body's
    type/color/props/booleans via the deep copy."""
    axis, outer_at_max = layer_axis_and_dir(elem, body)
    lo = min(getattr(body.start, axis), getattr(body.end, axis))
    hi = max(getattr(body.start, axis), getattr(body.end, axis))
    slices = []
    for mat, s_lo, s_hi in layer_spans(elem.layers, lo, hi, reverse=outer_at_max):
        if is_cavity(mat):
            continue
        slices.append(_derive_box_slice(body, axis, s_lo, s_hi, mat))
    logger.info("layers: %s sliced into %d solids (+%d air gaps) along %s",
                getattr(elem, "ifc_name", None) or type(elem).__name__,
                len(slices), len(elem.layers.layers) - len(slices), axis)
    return slices


def contour_normal(contour) -> Optional[Tuple[float, float, float]]:
    """Unit Newell normal of a 3D contour (None when degenerate)."""
    n = [0.0, 0.0, 0.0]
    pts = [(p.x, p.y, p.z) for p in contour]
    cnt = len(pts)
    for i in range(cnt):
        cx, cy, cz = pts[i]
        nx, ny, nz = pts[(i + 1) % cnt]
        n[0] += (cy - ny) * (cz + nz)
        n[1] += (cz - nz) * (cx + nx)
        n[2] += (cx - nx) * (cy + ny)
    length = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
    if length < 1e-9:
        return None
    return (n[0] / length, n[1] / length, n[2] / length)


def extrude_layer_slices(elem, body) -> Optional[List[object]]:
    """Per-layer Extrude slices along the body's extrusion normal.

    The body extrudes its contour by ``thickness`` along the Newell normal;
    slice i is the SAME contour shifted by the cumulative offset with
    thickness_i. Returns None (caller warns + emits unsliced) when the
    contour is degenerate."""
    normal = contour_normal(body.contour)
    if normal is None:
        return None
    # Slices run 0→thickness along +normal. outward·normal > 0 ⇒ the outer
    # (first-listed) layer belongs at the +normal (thickness) end → reverse.
    out = elem.layers.outward
    d = out[0] * normal[0] + out[1] * normal[1] + out[2] * normal[2]
    om = (out[0] ** 2 + out[1] ** 2 + out[2] ** 2) ** 0.5
    if om < 1e-12 or abs(d) < 0.999 * om:
        raise ValueError(
            f"LayerSet.outward {out} must be parallel to the extrusion normal "
            f"{tuple(round(c, 3) for c in normal)} (point it across the layers, "
            f"along the extrusion) — an oblique buildup is a follow-up")
    spans = layer_spans(elem.layers, 0.0, float(body.thickness), reverse=d > 0)
    slices = []
    for mat, s_lo, s_hi in spans:
        if is_cavity(mat):
            continue
        shifted = [
            p.model_copy(update={
                "x": p.x + normal[0] * s_lo,
                "y": p.y + normal[1] * s_lo,
                "z": p.z + normal[2] * s_lo,
            })
            for p in body.contour
        ]
        thickness = s_hi - s_lo
        sl = body.model_copy(deep=True, update={
            "contour": shifted, "thickness": thickness,
            "name": None, "material": mat.key,
        })
        sl._canonical_name = None  # anonymous — see _derive_box_slice
        slices.append(sl)
    return slices
