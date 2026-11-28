"""Triage: defects found by geometry, never by looking (no model calls).

On a cached ``Scene`` (``tools/survey/scene.py``): every pair of elements whose boxes come within the near-miss
reach is measured exactly with manifold3d (intersection volume, ``min_gap``). From those numbers:

- ``clash``: two elements that are not parts of one thing share volume;
- ``near_miss``: a column, beam or member and another element are 5-150 mm apart without meeting;
- ``floating``: a group of elements that touch each other but nothing that reaches the ground;
- ``unchecked``: an element whose mesh is not a closed solid, so none of the above could be measured.

Intended contact is the norm in a Lite-STEP model (parts of one element, a bearing on a pier, a slab on a wall),
so what is NOT reported matters as much: parts of one physical element never clash with each other, and the
class pairs in ``rules.py`` are contact by design. Numbers are millimetres and cubic metres, measured.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .rules import (CROSSING_CLASSES, GROUND_CLASSES, NEAR_MISS_CLASSES, STRUCTURAL_CLASSES, WALL_CLASSES,
                    contact_by_design)

logger = logging.getLogger(__name__)

TOUCH_M = 0.005  # closer than this is touching
NEAR_MISS_M = 0.150  # a gap up to this, at a column/beam/member, is a near miss
CLASH_MIN_M3 = 1e-7  # shared volume below this is numerical noise (0.1 cm3)
CLASH_ERROR_MM = 20.0  # a structural clash deeper than this is an error (less is bearing tolerance: a slab seated in its wall)


@dataclass
class Finding:
    id: str
    kind: str
    severity: str  # error | warning
    summary: str
    elements: list[str]
    numbers: dict[str, float] = field(default_factory=dict)
    where: list[float] | None = None  # a point in world metres to look at
    classes: list[str] = field(default_factory=list)  # the IFC classes involved, sorted: the finding's group

    @property
    def group(self) -> str:
        return f"{self.kind}:{'|'.join(self.classes)}" if self.classes else self.kind

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind, "severity": self.severity, "group": self.group,
                               "summary": self.summary, "elements": self.elements, "numbers": self.numbers}
        if self.where is not None:
            out["where"] = [round(float(v), 3) for v in self.where]
        return out


@dataclass
class Triage:
    findings: list[Finding]
    stats: dict[str, Any]


# -- the scene as owners and solids -------------------------------------------------------------------------


def owner_of(scene, index: int) -> int:
    """The physical element a shaped part belongs to: its top aggregation ancestor, or itself."""
    chain = [index, *scene.ancestors(index)]
    return chain[-1]


def _boxes(scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per shaped element: (indices, lo, hi)."""
    order = np.argsort(scene.tri_element, kind="stable")
    tri_el = scene.tri_element[order]
    pts = scene.vertices[scene.triangles[order]]  # (T, 3, 3)
    tlo, thi = pts.min(axis=1), pts.max(axis=1)
    idx, start = np.unique(tri_el, return_index=True)
    lo = np.minimum.reduceat(tlo, start)
    hi = np.maximum.reduceat(thi, start)
    return idx.astype(int), lo.astype(np.float64), hi.astype(np.float64)


def _candidate_pairs(lo: np.ndarray, hi: np.ndarray, pad: float) -> list[tuple[int, int]]:
    """Index pairs (into lo/hi) whose boxes, grown by ``pad``, overlap: sort and sweep along x."""
    lo_p, hi_p = lo - pad, hi + pad
    order = np.argsort(lo_p[:, 0])
    pairs: list[tuple[int, int]] = []
    active: list[int] = []
    for i in order:
        x0 = lo_p[i, 0]
        active = [j for j in active if hi_p[j, 0] >= x0]
        if active:
            aj = np.asarray(active)
            ok = np.all((lo_p[aj, 1:] <= hi_p[i, 1:]) & (hi_p[aj, 1:] >= lo_p[i, 1:]), axis=1)
            pairs.extend((int(min(i, j)), int(max(i, j))) for j in aj[ok])
        active.append(int(i))
    return pairs


def _triangles_by_element(scene) -> dict[int, np.ndarray]:
    """Each element's triangles, grouped once: a mask per element is quadratic on a large model."""
    order = np.argsort(scene.tri_element, kind="stable")
    owners = scene.tri_element[order]
    idx, start = np.unique(owners, return_index=True)
    ends = list(start[1:]) + [len(order)]
    return {int(i): scene.triangles[order[s:e]] for i, s, e in zip(idx, start, ends)}


def _solid(scene, tris: np.ndarray):
    """An element's triangles as a manifold3d solid; None when they do not close a volume."""
    import manifold3d as m3d

    used, inverse = np.unique(tris.ravel(), return_inverse=True)
    mesh = m3d.Mesh(vert_properties=scene.vertices[used].astype(np.float32),
                    tri_verts=inverse.reshape(-1, 3).astype(np.uint32))
    mesh.merge()
    solid = m3d.Manifold(mesh)
    if solid.status() != m3d.Error.NoError or solid.is_empty():
        return None
    return solid


PER_OWNER_PAIR = 3  # exact checks per pair of elements: one found clash says what the pair does
BUDGET_S = 120.0  # past this, suspects are counted as not measured, loudly
END_ZONE = 0.20  # an overlap inside the outer fifth of an element's length, at one end, is a joint


LINEAR_RATIO = 3.0  # an element is linear (a bar) when it is at least this many times longer than it is wide


def _axis(scene, tris: np.ndarray, wall: bool = False) -> tuple[np.ndarray, float, float] | None:
    """A linear element's main axis and its extent on it; None for a slab or a block, whose "end" means nothing
    (a beam through the top of a wall is a clash, not a joint). A wall's axis is its horizontal length, so two
    walls lapped at a corner (log walls notched into each other) meet at their ends."""
    pts = scene.vertices[np.unique(tris)].astype(np.float64)
    if wall:
        xy = pts[:, :2] - pts[:, :2].mean(axis=0)
        _, _, vt2 = np.linalg.svd(xy, full_matrices=False)
        ext = [float(np.ptp(xy @ v)) for v in vt2]
        if ext[0] < LINEAR_RATIO * max(ext[1], 1e-9):
            return None
        axis = np.array([vt2[0][0], vt2[0][1], 0.0])
        t = pts @ axis
        return axis, float(t.min()), float(t.max())
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    ext = [float(np.ptp(pts @ v)) for v in vt[:2]]
    if ext[0] < LINEAR_RATIO * max(ext[1], 1e-9):
        return None
    t = pts @ vt[0]
    return vt[0], float(t.min()), float(t.max())


CROSSING_DEG = 20.0  # two bars whose axes differ by more than this cross; less, and they lie along each other


def _crossing_deg(axes, a: int, b: int) -> float | None:
    """The angle between two linear elements' axes, when both are linear."""
    if axes[a] is None or axes[b] is None:
        return None
    c = abs(float(np.dot(axes[a][0], axes[b][0])))
    return float(np.degrees(np.arccos(min(1.0, c))))


def _end_joint(shared, axes, a: int, b: int) -> bool:
    """The overlap sits at one END of a linear element: an embedded cable anchor, a beam set into its column,
    two spans meeting over a bearing. A beam through a window crosses the middle and is not a joint."""
    pts = np.asarray(shared.to_mesh().vert_properties, dtype=np.float64)[:, :3]
    if len(pts) == 0:
        return False
    for e in (a, b):
        if axes[e] is None:
            continue
        axis, t0, t1 = axes[e]
        length = t1 - t0
        if length <= 0:
            continue
        t = pts @ axis
        zone = END_ZONE * length
        if t.max() <= t0 + zone or t.min() >= t1 - zone:
            return True
    return False


# -- the triage ----------------------------------------------------------------------------------------------


def run(scene, budget_s: float = BUDGET_S) -> Triage:
    t0 = time.time()
    shaped, lo, hi = _boxes(scene)
    names = [e.label for e in scene.elements]
    classes = [e.ifc_class for e in scene.elements]
    own = np.array([owner_of(scene, int(i)) for i in shaped], dtype=np.int64)
    owner = {int(i): int(o) for i, o in zip(shaped, own)}
    owner_class = {i: classes[o] for i, o in owner.items()}
    by_element = _triangles_by_element(scene)

    solids: dict[int, Any] = {}
    unchecked: set[int] = set()

    def solid(e: int):
        if e not in solids and e not in unchecked:
            s = _solid(scene, by_element[e])
            if s is None:
                unchecked.add(e)
            else:
                solids[e] = s
        return solids.get(e)

    axes: dict[int, tuple[np.ndarray, float, float] | None] = {}

    def axis(e: int):
        if e not in axes:
            axes[e] = _axis(scene, by_element[e], wall=owner_class[e] in WALL_CLASSES)
        return axes[e]

    pairs = np.asarray(_candidate_pairs(lo, hi, NEAR_MISS_M), dtype=np.int64).reshape(-1, 2)
    pa, pb = pairs[:, 0], pairs[:, 1]
    # box arithmetic for every pair at once: the separation per axis (0 when the boxes overlap on it)
    sep = np.maximum(0.0, np.maximum(lo[pb] - hi[pa], lo[pa] - hi[pb]))
    box_gap = np.linalg.norm(sep, axis=1)
    overlap = np.clip(np.minimum(hi[pa], hi[pb]) - np.maximum(lo[pa], lo[pb]), 0.0, None).prod(axis=1)
    same_owner = own[pa] == own[pb]

    # Touching, for the floating test: parts of one element hold together, and boxes within reach touch. Boxes
    # can only make two elements look closer than they are, so this may miss a floating part, never invent one.
    touch: dict[int, set[int]] = {int(i): set() for i in shaped}
    for x, y in pairs[same_owner | (box_gap <= TOUCH_M)]:
        a, b = int(shaped[x]), int(shaped[y])
        touch[a].add(b)
        touch[b].add(a)

    by_design = np.array([contact_by_design(owner_class[int(shaped[x])], owner_class[int(shaped[y])])
                          for x, y in pairs], dtype=bool) if len(pairs) else np.zeros(0, dtype=bool)
    near_class = np.array([classes[own[x]] in NEAR_MISS_CLASSES or classes[own[y]] in NEAR_MISS_CLASSES
                           for x, y in pairs], dtype=bool) if len(pairs) else np.zeros(0, dtype=bool)
    clash_cand = ~same_owner & ~by_design & (overlap > CLASH_MIN_M3)
    near_cand = ~same_owner & ~by_design & near_class & ~clash_cand & (box_gap < NEAR_MISS_M)

    clashes: dict[tuple[int, int], tuple[float, float, list[float]]] = {}
    near: dict[tuple[int, int], tuple[float, list[float]]] = {}
    tried: dict[tuple[int, int], int] = {}
    exact = skipped = 0
    # biggest box overlaps first, so a budget that runs out leaves the smallest suspects unchecked
    order = np.concatenate([np.flatnonzero(clash_cand)[np.argsort(-overlap[clash_cand])],
                            np.flatnonzero(near_cand)[np.argsort(box_gap[near_cand])]])
    for k in order:
        x, y = int(pairs[k, 0]), int(pairs[k, 1])
        a, b = int(shaped[x]), int(shaped[y])
        key = (owner[a], owner[b]) if owner[a] <= owner[b] else (owner[b], owner[a])
        if tried.get(key, 0) >= PER_OWNER_PAIR or (key in clashes and not clash_cand[k]):
            continue
        if time.time() - t0 > budget_s:
            skipped += 1
            continue
        sa, sb = solid(a), solid(b)
        if sa is None or sb is None:
            continue
        tried[key] = tried.get(key, 0) + 1
        exact += 1
        if clash_cand[k]:
            shared = sa ^ sb
            vol = float(shared.volume()) if not shared.is_empty() else 0.0
            ax = {a: axis(a), b: axis(b)}
            if vol > CLASH_MIN_M3 and not _end_joint(shared, ax, a, b):
                bb = shared.bounding_box()
                ext = np.asarray(bb[3:]) - np.asarray(bb[:3])
                centre = [float(v) for v in (np.asarray(bb[:3]) + np.asarray(bb[3:])) / 2]
                both_members = owner_class[a] in CROSSING_CLASSES and owner_class[b] in CROSSING_CLASSES
                deg = _crossing_deg(ax, a, b) if both_members else None
                crossing = deg if deg is not None and deg > CROSSING_DEG else None
                prev = clashes.get(key)
                if prev is None or vol > prev[0]:
                    clashes[key] = (vol, float(np.min(ext)) * 1000.0, centre, crossing)
            continue
        # min_gap answers the search length itself when nothing is that close: strictly less is a measured gap
        gap = float(sa.min_gap(sb, NEAR_MISS_M))
        if TOUCH_M < gap < NEAR_MISS_M and (key not in near or gap < near[key][0]):
            near[key] = (gap, [float(v) for v in (lo[x] + hi[x] + lo[y] + hi[y]) / 4])

    findings: list[Finding] = []
    for (oa, ob), (vol, depth_mm, centre, crossing) in sorted(clashes.items(), key=lambda kv: -kv[1][0]):
        if crossing is not None:
            # an X of bracing drawn as two whole bars: on site one runs through and the other is split, or both
            # meet a gusset. Worth knowing, not a defect of the model's intent.
            findings.append(Finding("", "crossing", "warning",
                                    f"{names[oa]} ({classes[oa]}) and {names[ob]} ({classes[ob]}) cross through each "
                                    f"other at {crossing:.0f} deg mid-length, {vol * 1e6:.0f} cm3 shared",
                                    [names[oa], names[ob]], {"volume_cm3": round(vol * 1e6, 1),
                                                             "angle_deg": round(crossing, 1)}, centre,
                                    sorted([classes[oa], classes[ob]])))
            continue
        structural = classes[oa] in STRUCTURAL_CLASSES and classes[ob] in STRUCTURAL_CLASSES
        sev = "error" if structural and depth_mm > CLASH_ERROR_MM else "warning"
        findings.append(Finding("", "clash", sev,
                                f"{names[oa]} ({classes[oa]}) and {names[ob]} ({classes[ob]}) share "
                                f"{vol * 1e6:.0f} cm3, {depth_mm:.0f} mm deep",
                                [names[oa], names[ob]], {"volume_cm3": round(vol * 1e6, 1),
                                                         "depth_mm": round(depth_mm, 1)}, centre,
                                sorted([classes[oa], classes[ob]])))
    for (oa, ob), (gap, centre) in sorted(near.items(), key=lambda kv: kv[1][0]):
        findings.append(Finding("", "near_miss", "warning",
                                f"{names[oa]} ({classes[oa]}) and {names[ob]} ({classes[ob]}) are "
                                f"{gap * 1000:.0f} mm apart without meeting",
                                [names[oa], names[ob]], {"gap_mm": round(gap * 1000, 1)}, centre,
                                sorted([classes[oa], classes[ob]])))
    findings += _floating(scene, shaped, lo, hi, touch, owner, names, classes)
    for i in sorted(unchecked):
        findings.append(Finding("", "unchecked", "warning",
                                f"{names[owner[i]]} ({classes[owner[i]]}): its mesh is not a closed solid, so it was "
                                "not checked for clashes, gaps or support", [names[owner[i]]], {}, None,
                                [classes[owner[i]]]))
    for n, f in enumerate(findings, 1):
        f.id = f"{f.kind}-{n:03d}"
    if skipped:
        findings.append(Finding("", "budget", "warning",
                                f"the triage ran out of its {budget_s:.0f} s budget: {skipped} suspect pair(s) were "
                                "not measured; run again with --budget to check them all", [], {"pairs": skipped}))
    stats = {"elements": len(shaped), "pairs": len(pairs), "exact": exact, "notMeasured": skipped,
             "unchecked": len(unchecked), "seconds": round(time.time() - t0, 2)}
    logger.info("[review.triage] %s", stats)
    return Triage(findings, stats)


def _floating(scene, shaped, lo, hi, touch, owner, names, classes) -> list[Finding]:
    """Groups of touching elements that reach neither the ground nor the model's lowest level."""
    z_min = float(lo[:, 2].min()) if len(lo) else 0.0
    low = {int(i) for i, z in zip(shaped, lo[:, 2]) if z <= z_min + TOUCH_M}
    grounded = {int(i) for i in shaped if classes[owner[int(i)]] in GROUND_CLASSES} | low
    seen: set[int] = set()
    out: list[Finding] = []
    pos = {int(i): k for k, i in enumerate(shaped)}
    for start in (int(i) for i in shaped):
        if start in seen:
            continue
        group, stack = set(), [start]
        while stack:
            x = stack.pop()
            if x in group:
                continue
            group.add(x)
            stack.extend(touch.get(x, ()) - group)
        seen |= group
        if group & grounded:
            continue
        owners = sorted({owner[x] for x in group}, key=lambda o: names[o])
        bottom = min(float(lo[pos[x], 2]) for x in group)
        centre = [float(v) for v in np.mean([(lo[pos[x]] + hi[pos[x]]) / 2 for x in group], axis=0)]
        structural = any(classes[o] in STRUCTURAL_CLASSES for o in owners)
        shown = ", ".join(f"{names[o]} ({classes[o]})" for o in owners[:6]) + (" …" if len(owners) > 6 else "")
        out.append(Finding("", "floating", "error" if structural else "warning",
                           f"{len(owners)} element(s) touch nothing that reaches the ground: {shown}; "
                           f"lowest point {bottom - z_min:.3f} m above the model's lowest",
                           [names[o] for o in owners], {"height_above_lowest_m": round(bottom - z_min, 3)}, centre))
    return out
