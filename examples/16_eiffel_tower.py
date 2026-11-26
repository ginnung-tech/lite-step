"""The Eiffel Tower, from Eiffel's own dimensioned plates (as reproduced in the
thesis "3D Modelling of Eiffel Tower", Politecnico di Torino).

Frame: x east, y north, z up, mm. z = 0 is the level of the support centres
("centres des sommiers", absolute +36.00); the ground at the tower's centre is
z = -2500 (+33.50). Every level below is "metres above the support centres".

Structure (plates PL. II-VII, thesis Fig. 1-19):
- 30 panel lines L0..L30. Below the 2nd floor each of the four piers is four
  crossbows ("arbalétriers") on a square: 16 crossbows.
- At L12 (2nd floor) the four inner crossbows end; the rest become one shaft
  of 12 crossbows; the two mid-side crossbows of each face converge and merge
  at L18, leaving 8 (four corners, four mid-sides) up to the 3rd floor (L30).
- Each panel face is braced by a St Andrew cross and a crossbar on every line.
- 1st floor girder (L5..L6), the arc beam and the decorative arch under it;
  2nd floor bands (L10, L11..L12); 3rd floor platform; campanile.

The tables are the drawings' dimensions; the minion diffs the model against
the camera-aligned drawings and the tables are corrected from its findings.

Compile with::

    python -m lite_step examples/16_eiffel_tower.py
"""
# ruff: noqa: F401 — the import block below is the full DSL menu, kept intact
# by design: unused names are a feature (later tasks use them), never lint
# errors, and never to be pruned.

import math
import numpy as np
from lite_step import compile_main
from lite_step.models import (Project, Storey, Point, Point2D,
    Wall, Window, Door, Slab, Roof, Column, Beam, Space, Site, Element,
    Box, Extrude, Sweep, Sheet, Pipe, Revolve, Mesh, Bar, Material, LayerSet,
    ReferencePoint, GuideLine, Transform, Anchor)
from lite_step.helpers import arc_path, stirrup_path
from lite_step.materials.dk import MATERIALS   # stock introspection

I = lambda x: int(round(x))  # noqa: E731

IRON = "#5E4B3C"  # Eiffel Tower brown

# Panel heights (m), bottom up, from the support centres (PL. II, right column).
PANEL_HEIGHTS = [4.13, 11.0, 11.0, 11.0, 11.0, 7.0, 11.0, 11.0, 11.0, 10.20, 10.00, 4.90,
                 11.300, 10.700, 10.600, 10.500, 10.400, 10.000, 10.000, 9.50359, 9.03615,
                 8.59654, 8.13334, 7.63810, 7.20027, 6.88318, 6.51702, 6.16588, 5.83364]
THIRD_FLOOR_Z = 273.88   # measured on PL. VII (the table's 273.63 is the level of the panel lines)
THIRD_SIDE = 18.65
THIRD_GIRDER_BOTTOM = 271.37
CAMPANILE_BOX_HALF, CAMPANILE_BOX_TOP = 5.85, 280.5
# (z, radius) of the dome, drum and cupola, measured on PL. VII
CAMPANILE_PROFILE = [(280.5, 5.85), (281.0, 5.66), (281.5, 5.53), (282.0, 5.35), (282.5, 5.19), (283.0, 4.90),
                     (283.5, 4.58), (284.0, 4.16), (284.5, 3.60), (285.0, 2.89), (285.5, 2.75), (287.0, 2.79),
                     (289.0, 2.85), (289.5, 2.36), (290.0, 1.85), (291.0, 1.81), (292.0, 1.80), (293.0, 1.49),
                     (294.0, 1.36), (295.5, 1.33), (296.0, 0.99), (296.5, 0.65)]
CAMPANILE_POLE_TOP = 304.0

# Lower piers, L0..L12: (inner, outer) crossbow axis distance from the tower axis (m),
# measured in the face elevation: inner = G/2, outer = G/2 + W (pier width W).
PIER = {
    0: (43.180, 58.180),   # extrapolated from L1-L2
    1: (41.1435, 56.1435),  # G 82.287, W 15.00
    2: (35.7135, 50.7135),  # G 71.427
    3: (30.434, 45.434),    # G 60.868
    4: (24.4045, 39.4045),  # G 48.809
    5: (19.125, 34.125),    # G 38.25, W 15.00 (read 16.00; the minion measured 0.98 m in, v2)
    6: (16.3476, 32.0476),  # G 32.6952, W 15.70 (1st floor)
    7: (13.55, 27.65),      # G 27.10, W 14.10
    8: (11.325, 24.475),    # G 22.65, W 13.15
    9: (9.40, 21.60),       # G 18.80, W 12.20
    10: (7.84, 19.16),      # G 15.68, W 11.32
    11: (6.3558, 17.2134),  # G 12.7116, W 10.8576
    12: (5.34, 15.752),     # G 10.680, W 10.412 (2nd floor)
}
# Shaft, L12..L30: corner crossbow half-width T/2 (m) and mid-side pair half-gap (m).
SHAFT_HALF = {12: 15.752, 13: 14.346, 14: 12.90, 15: 11.80, 16: 10.80, 17: 10.0, 18: 9.0, 19: 8.70,
              20: 8.23122, 21: 7.8377, 22: 7.36889, 23: 6.98108, 24: 6.58526, 25: 6.20508,
              26: 5.90365, 27: 5.56772, 28: 5.25579, 29: 5.0, 30: 5.0}
MID_HALF_GAP = {12: 5.34, 13: 4.45, 14: 3.50, 15: 2.20, 16: 1.20, 17: 0.50, 18: 0.0}

# The decorative arch (PL. III, fitted on the 1:200 plate): intrados R 40.38373 (the plate's label),
# crown 37.45 m; band 3.27 m deep at the crown; the arc beam's bottom chord at 41.06 m.
ARCH_CROWN_R, ARCH_CROWN_Z = 40.384, -2.934          # crown arc: centre (0, -2.934)
ARCH_SIDE_R, ARCH_SIDE_U, ARCH_SIDE_Z = 34.125, 2.859, 2.898  # side arcs: centres (+-2.859, 2.898)
ARCH_FOOT = 36.86
ARCH_BAND = 3.27
# The arch's foot (PL. III, ink silhouette through the frozen fig7 camera, v12): below the side arc's
# vertical tangent (u = +-36.984, z = 2.898) the intrados drops straight to a collar at z 2.0; the shaft
# there is 0.58 m wide, axis at u = +-37.30. It stands on a pedestal, not on the pier: collar 0.79 m
# (1.75..2.0), moulded capital 1.55 m (1.0..1.75), stone socle 1.38 m (-1.2..1.0), base 1.8 m (-1.9..-1.2).
# Square in plan: the elevation gives widths only (assumed, single source).
ARCH_TANGENT_U = ARCH_SIDE_U + ARCH_SIDE_R    # 36.984
ARCH_SHAFT_U, ARCH_SHAFT_W, ARCH_COLLAR_Z = 37.30, 580, 2.0
PEDESTAL = [("collar", 1.75, 2.0, 0.79), ("capital", 1.0, 1.75, 1.55),
            ("socle", -1.2, 1.0, 1.38), ("base", -1.9, -1.2, 1.80)]  # (part, z0, z1, width m)


def arch_intrados(u):
    """Height of the arch's intrados at u (m), and the centre of the arc that holds it."""
    if abs(u) <= 25.0:
        return ARCH_CROWN_Z + math.sqrt(max(0.0, ARCH_CROWN_R ** 2 - u * u)), (0.0, ARCH_CROWN_Z)
    cu = -ARCH_SIDE_U if u < 0 else ARCH_SIDE_U
    return ARCH_SIDE_Z + math.sqrt(max(0.0, ARCH_SIDE_R ** 2 - (u - cu) ** 2)), (cu, ARCH_SIDE_Z)
ARC_BEAM_BOTTOM = 41.06
FIRST_FLOOR_HALF = 70.695 / 2
# Masonry under each pier (PL. I): 25.334 m across at the ground, top at the support centres.
PLINTH = 25.334

BOW = 800       # crossbow (arbalétrier) section, lattice box as its envelope (PL. III: ~0.8 m)
BRACE = 600     # lattice diagonal ("Treillis"), PL. III ~0.6-0.65 m
BAR = 550       # crossbar ("Corn. 80x80x10" lattice), PL. III ~0.55 m


def levels():
    z = [0.0]
    for h in PANEL_HEIGHTS:
        z.append(z[-1] + h)
    z.append(THIRD_FLOOR_Z)
    return z


Z = levels()  # Z[i] = height of line Li (m)


def outer_at(z):
    """Outer crossbow distance from the axis at height z (m), piecewise linear over L0..L12."""
    for i in range(12):
        if Z[i] <= z <= Z[i + 1]:
            t = (z - Z[i]) / (Z[i + 1] - Z[i])
            return PIER[i][1] + t * (PIER[i + 1][1] - PIER[i][1])
    return PIER[12][1]


FACES = ("s", "n", "w", "e")


def inner_at(z):
    """Inner crossbow distance from the axis at height z (m), piecewise linear over L0..L12."""
    for i in range(12):
        if Z[i] <= z <= Z[i + 1]:
            t = (z - Z[i]) / (Z[i + 1] - Z[i])
            return PIER[i][0] + t * (PIER[i + 1][0] - PIER[i][0])
    return PIER[12][0]


def flat_point(face, u, z, d):
    """A point on a vertical face plane at distance d from the axis."""
    return {"s": P(u, -d, z), "n": P(u, d, z), "w": P(-d, u, z), "e": P(d, u, z)}[face]


def face_point(face, u, z):
    """A point on a face plane of the tower: u along the face (left to right seen from outside is
    not needed here: u is world x on s/n, world y on w/e), at height z, in the outer face plane."""
    d = outer_at(z)
    return {"s": P(u, -d, z), "n": P(u, d, z), "w": P(-d, u, z), "e": P(d, u, z)}[face]


def lattice_girder(name, face, z0, z1, half_span, _plane, bay=3.8825, chord=500, web=250):
    """A lattice girder in a face plane between heights z0 and z1: top and bottom chords spanning
    -half_span(z)..+half_span(z), verticals every `bay` from the centre out, a St Andrew cross in each bay."""
    g = Element(ifc_class="IfcBeam", name=name)
    zb, zt = z0 + chord / 2000, z1 - chord / 2000
    hb, ht = half_span(zb), half_span(zt)
    g.add(member("chord_bottom", [face_point(face, -hb, zb), face_point(face, hb, zb)], chord))
    g.add(member("chord_top", [face_point(face, -ht, zt), face_point(face, ht, zt)], chord))
    n = max(1, int(round(2 * min(hb, ht) / bay)))
    us_b = [-hb + 2 * hb * k / n for k in range(n + 1)]
    us_t = [-ht + 2 * ht * k / n for k in range(n + 1)]
    for k in range(n + 1):
        g.add(member(f"post_{k:02d}", [face_point(face, us_b[k], zb), face_point(face, us_t[k], zt)], web))
        if k < n:
            g.add(member(f"x1_{k:02d}", [face_point(face, us_b[k], zb), face_point(face, us_t[k + 1], zt)], web))
            g.add(member(f"x2_{k:02d}", [face_point(face, us_t[k], zt), face_point(face, us_b[k + 1], zb)], web))
    return g


def P(x, y, z):
    return Point(x=I(x * 1000), y=I(y * 1000), z=I(z * 1000))


def square(size):
    h = size // 2
    return [Point2D(x=-h, y=-h), Point2D(x=h, y=-h), Point2D(x=h, y=h), Point2D(x=-h, y=h)]


def segmented(name, pts, size, first_line):
    """A crossbow as one member per panel (segment k spans lines first_line+k .. +k+1)."""
    bow = Element(ifc_class="IfcBuildingElementProxy", name=name)
    for k in range(len(pts) - 1):
        bow.add(member(f"seg_{first_line + k:02d}", [P(*pts[k]), P(*pts[k + 1])], size))
    return bow


def member(name, pts, size, ifc="IfcMember"):
    e = Element(ifc_class=ifc, name=name)
    e.add(Sweep(name="body", profile=square(size), path=pts, color=IRON))
    return e


def bow_points(sx, sy, which, top):
    """Crossbow axis points from L0 up to `top` for the pier in quadrant (sx, sy).
    which: 'a' inner-inner, 'b' inner x / outer y, 'c' outer-outer, 'd' outer x / inner y."""
    pts = []
    for i in range(0, top + 1):
        inner, outer = PIER[i]
        ux = inner if which in ("a", "b") else outer
        uy = inner if which in ("a", "d") else outer
        pts.append((sx * ux, sy * uy, Z[i]))
    return pts


def shaft_points(kind, sx, sy, start, end):
    """Shaft crossbows above L12: 'corner' at (sx*T/2, sy*T/2); 'midx' at (sx*m, sy*T/2) (on the
    face y = sy*T/2); 'midy' at (sx*T/2, sy*m)."""
    pts = []
    for i in range(start, end + 1):
        t = SHAFT_HALF[i]
        m = MID_HALF_GAP.get(i, 0.0)
        if kind == "corner":
            pts.append((sx * t, sy * t, Z[i]))
        elif kind == "midx":
            pts.append((sx * m, sy * t, Z[i]))
        else:
            pts.append((sx * t, sy * m, Z[i]))
    return pts


def mid(a, b, t=0.5):
    return tuple(a[i] + t * (b[i] - a[i]) for i in range(3))


# Pier panel bracing by panel (PL. II/III): 0 base strut zone; 1-4 "union jack" (X + mid
# crossbar + mid-strut); 5 the girder zone, a grid of small crosses; 6-11 X + mid-strut.
PIER_PANEL_STYLE = {0: "x", 1: "jack", 2: "jack", 3: "jack", 4: "jack", 5: "grid",
                    6: "strut", 7: "strut", 8: "strut", 9: "strut", 10: "strut", 11: "strut"}


def brace_face(container, tag, left, right, lines, union_jack=False, styles=None):
    """Crossbars on every line and a St Andrew cross in every panel between two crossbow
    polylines (lists of (x,y,z) indexed by the same panel lines). `union_jack`: the pier
    panels of PL. III also carry a mid-height crossbar and a mid-strut joining the middles
    of the two line crossbars, all four crossing at the panel's centre."""
    for k in range(len(lines)):
        a, b = left[k], right[k]
        container.add(member(f"{tag}_bar_{lines[k]:02d}", [P(*a), P(*b)], BAR))
        if k + 1 < len(lines):
            a2, b2 = left[k + 1], right[k + 1]
            container.add(member(f"{tag}_x1_{lines[k]:02d}", [P(*a), P(*b2)], BRACE))
            container.add(member(f"{tag}_x2_{lines[k]:02d}", [P(*b), P(*a2)], BRACE))
            style = (styles or {}).get(lines[k], "jack" if union_jack else "x")
            if style in ("jack",):
                container.add(member(f"{tag}_midbar_{lines[k]:02d}", [P(*mid(a, a2)), P(*mid(b, b2))], BAR))
            if style in ("jack", "strut"):
                container.add(member(f"{tag}_midstrut_{lines[k]:02d}", [P(*mid(a, b)), P(*mid(a2, b2))], BRACE))
            if style == "grid":  # two-by-two small crosses
                for i in range(2):
                    for j in range(2):
                        q00 = mid(mid(a, b, i / 2), mid(a2, b2, i / 2), j / 2)
                        q11 = mid(mid(a, b, (i + 1) / 2), mid(a2, b2, (i + 1) / 2), (j + 1) / 2)
                        q01 = mid(mid(a, b, i / 2), mid(a2, b2, i / 2), (j + 1) / 2)
                        q10 = mid(mid(a, b, (i + 1) / 2), mid(a2, b2, (i + 1) / 2), j / 2)
                        container.add(member(f"{tag}_g{i}{j}a_{lines[k]:02d}", [P(*q00), P(*q11)], BAR))
                        container.add(member(f"{tag}_g{i}{j}b_{lines[k]:02d}", [P(*q01), P(*q10)], BAR))


def build_pier(name, sx, sy):
    pier = Element(ifc_class="IfcBuildingElementProxy", name=name)
    bows = {}
    for w in "abcd":
        top = 12
        bows[w] = bow_points(sx, sy, w, top)
        bow = Element(ifc_class="IfcBuildingElementProxy", name=f"bow_{w}")
        for k in range(top):  # one segment per panel, as it was built
            bow.add(member(f"seg_{k:02d}", [P(*bows[w][k]), P(*bows[w][k + 1])], BOW))
        pier.add(bow)
    lines = list(range(0, 13))
    # the pier's four faces: a-b (inner x), c-d (outer x), a-d (inner y), b-c (outer y)
    for tag, l, r in (("face_ix", "a", "b"), ("face_ox", "d", "c"), ("face_iy", "a", "d"), ("face_oy", "b", "c")):
        f = Element(ifc_class="IfcBuildingElementProxy", name=tag)
        brace_face(f, tag, bows[l], bows[r], lines, styles=PIER_PANEL_STYLE)
        pier.add(f)
    return pier


def build_shaft():
    shaft = Element(ifc_class="IfcBuildingElementProxy", name="shaft")
    corners = {}
    for sx, sy, q in ((1, 1, "ne"), (-1, 1, "nw"), (-1, -1, "sw"), (1, -1, "se")):
        corners[q] = shaft_points("corner", sx, sy, 12, 30)
        shaft.add(segmented(f"bow_corner_{q}", corners[q], BOW, 12))
    # mid-side crossbows: a converging pair L12..L18, one merged crossbow L18..L30
    mids = {}
    for face, kind, sx, sy in (("n", "midx", None, 1), ("s", "midx", None, -1), ("e", "midy", 1, None), ("w", "midy", -1, None)):
        pair = []
        for side in (-1, 1):
            if kind == "midx":
                pts = shaft_points("midx", side, sy, 12, 18)
            else:
                pts = shaft_points("midy", sx, side, 12, 18)
            pair.append(pts)
            shaft.add(segmented(f"bow_mid_{face}_{'l' if side < 0 else 'r'}", pts, BOW, 12))
        merged = shaft_points(kind, 0 if kind == "midx" else sx, 0 if kind == "midy" else sy, 18, 30) if False else None
        if kind == "midx":
            merged = [(0.0, sy * SHAFT_HALF[i], Z[i]) for i in range(18, 31)]
        else:
            merged = [(sx * SHAFT_HALF[i], 0.0, Z[i]) for i in range(18, 31)]
        shaft.add(segmented(f"bow_mid_{face}", merged, BOW, 18))
        mids[face] = (pair, merged)
    # bracing on each face, L18..L30: corner-mid-corner, two bays
    faces = {"n": ("nw", "ne"), "s": ("sw", "se"), "e": ("se", "ne"), "w": ("sw", "nw")}
    for face, (lq, rq) in faces.items():
        f = Element(ifc_class="IfcBuildingElementProxy", name=f"face_{face}")
        pair, merged = mids[face]
        lines = list(range(18, 31))
        left = corners[lq][18 - 12:]
        right = corners[rq][18 - 12:]
        brace_face(f, f"{face}_l", left, merged, lines)
        brace_face(f, f"{face}_r", merged, right, lines)
        # L12..L18: corner to its near mid crossbow
        lines2 = list(range(12, 19))
        brace_face(f, f"{face}_lo_l", corners[lq][:7], pair[0], lines2)
        brace_face(f, f"{face}_lo_r", pair[1], corners[rq][:7], lines2)
        brace_face(f, f"{face}_lo_c", pair[0], pair[1], lines2[:-1])  # the centre bay, until they merge
        shaft.add(f)
    return shaft


def build_floors():
    floors = Element(ifc_class="IfcBuildingElementProxy", name="floors")
    # 1st floor girder (Poutre du 1er Etage), L5..L6, outer crossbow to outer crossbow, in the outer
    # face planes; lattice: chords, verticals at the plate's bay 3.8825 m, a St Andrew cross per bay.
    for face in FACES:  # the floor's exterior side is 70.695 m (PL. I): the girder runs past the crossbows
        floors.add(lattice_girder(f"first_girder_{face}", face, Z[5], Z[6],
                                  lambda z: FIRST_FLOOR_HALF, lambda z: outer_at(z)))
    # arc beam (Poutre de l'Arc), L4..L5, between the inner crossbows
    for face in FACES:  # bottom chord at 41.06 m (PL. III, measured at the crown), top at L5
        floors.add(lattice_girder(f"arc_beam_{face}", face, ARC_BEAM_BOTTOM, Z[5],
                                  lambda z: inner_at(z), lambda z: outer_at(z)))
    # decorative arch (PL. III, fitted on the 1:200 plate): a compound intrados - the crown arc
    # (centre on the axis, R 40.384) between u = +-25 m, the side arcs (centres +-2.859 m, z 2.898,
    # R 34.125) down to the feet at u = +-36.86 on the support level. The band above it is 3.27 m
    # deep, radially divided; the arcade's posts stand on it up to the arc beam.
    for face in FACES:
        arch = Element(ifc_class="IfcMember", name=f"arch_{face}")
        intr, extr = [], []
        u = -ARCH_FOOT
        while u <= ARCH_FOOT + 1e-9:
            z, (cu, cz_) = arch_intrados(u)
            if z >= 0:
                intr.append(face_point(face, u, z))
                d = math.hypot(u - cu, z - cz_)
                ue, ze = u + (u - cu) / d * ARCH_BAND, z + (z - cz_) / d * ARCH_BAND
                if abs(ue) <= inner_at(ze) and ze < ARC_BEAM_BOTTOM:
                    extr.append(face_point(face, ue, ze))
            u += 0.5
        # the feet: from the arc's end to its vertical tangent, then straight down to the collar
        foot = lambda sgn: [face_point(face, sgn * ARCH_TANGENT_U, ARCH_SIDE_Z),  # noqa: E731
                            face_point(face, sgn * ARCH_TANGENT_U, ARCH_COLLAR_Z)]
        intr = foot(-1)[::-1] + intr + foot(1)
        arch.add(member("intrados", intr, 300))
        for sgn, side in ((-1, "l"), (1, "r")):
            arch.add(member(f"foot_{side}", [face_point(face, sgn * ARCH_SHAFT_U, ARCH_COLLAR_Z),
                                             face_point(face, sgn * ARCH_SHAFT_U, 5.8)], ARCH_SHAFT_W))
        arch.add(member("extrados", extr, 300))
        for k in range(1, 60):
            u = -ARCH_FOOT + 2 * ARCH_FOOT * k / 60
            z, (cu, cz_) = arch_intrados(u)
            d = math.hypot(u - cu, z - cz_)
            ue, ze = u + (u - cu) / d * ARCH_BAND, z + (z - cz_) / d * ARCH_BAND
            if z < 0 or abs(ue) > inner_at(ze) or ze >= ARC_BEAM_BOTTOM:
                continue
            arch.add(member(f"rib_{k:02d}", [face_point(face, u, z), face_point(face, ue, ze)], 200))
        n = int(round(2 * inner_at(ARC_BEAM_BOTTOM) / 3.8825))
        for j in range(1, n):
            u = -inner_at(ARC_BEAM_BOTTOM) + 2 * inner_at(ARC_BEAM_BOTTOM) * j / n
            z, (cu, cz_) = arch_intrados(u)
            d = math.hypot(u - cu, z - cz_)
            zt = z + (z - cz_) / d * ARCH_BAND
            if zt < ARC_BEAM_BOTTOM - 0.3:
                arch.add(member(f"post_{j:02d}", [face_point(face, u, zt), face_point(face, u, ARC_BEAM_BOTTOM)], 250))
        # a pedestal under each foot, in the face plane at the collar's height
        d = outer_at(ARCH_COLLAR_Z)
        for sgn, side in ((-1, "l"), (1, "r")):
            ped = Element(ifc_class="IfcColumn", name=f"pedestal_{side}")
            for part, z0, z1, w in PEDESTAL:
                h = w / 2
                a, b = flat_point(face, sgn * ARCH_SHAFT_U - h, z0, d - h), flat_point(face, sgn * ARCH_SHAFT_U + h, z1, d + h)
                ped.add(Element(ifc_class="IfcColumn", name=part).add(Box(  # corners in mm (flat_point returns mm)
                    start=Point(x=I(min(a.x, b.x)), y=I(min(a.y, b.y)), z=I(z0 * 1000)),
                    end=Point(x=I(max(a.x, b.x)), y=I(max(a.y, b.y)), z=I(z1 * 1000)), color="#A39E93")))
            arch.add(ped)
        floors.add(arch)
    # 2nd floor girders: L10 band (3.60 m) and L11..L12, outer to outer
    for face in FACES:
        floors.add(lattice_girder(f"band_10_{face}", face, Z[10], Z[10] + 3.6,
                                  lambda z: outer_at(z), lambda z: outer_at(z), bay=1.8))
        floors.add(lattice_girder(f"band_11_{face}", face, Z[11], Z[12],
                                  lambda z: outer_at(z), lambda z: outer_at(z), bay=1.8))
    # the X between facing piers in the middle bay, L10..L11 (PL. II)
    for face in FACES:
        xb = Element(ifc_class="IfcMember", name=f"mid_cross_{face}")
        a0, a1 = face_point(face, -inner_at(Z[10]), Z[10]), face_point(face, inner_at(Z[10]), Z[10])
        b0, b1 = face_point(face, -inner_at(Z[11]), Z[11]), face_point(face, inner_at(Z[11]), Z[11])
        xb.add(member("d1", [a0, b1], BRACE))
        xb.add(member("d2", [a1, b0], BRACE))
        floors.add(xb)
    # 1st floor deck (ring: outer 70.695 m, open centre)
    o, i = 70.695 / 2, PIER[6][0]
    for name, (x0, y0, x1, y1) in {"deck1_s": (-o, -o, o, -i), "deck1_n": (-o, i, o, o),
                                   "deck1_w": (-o, -i, -i, i), "deck1_e": (i, -i, o, i)}.items():
        floors.add(Element(ifc_class="IfcSlab", name=name).add(Box(start=P(x0, y0, Z[6] - 0.3), end=P(x1, y1, Z[6]), color=IRON)))
    s2 = 37.68 / 2
    floors.add(Element(ifc_class="IfcSlab", name="deck2").add(Box(start=P(-s2, -s2, Z[12] - 0.3), end=P(s2, s2, Z[12]), color=IRON)))
    # 3rd floor (PL. I/VII): exterior side 18.65 m, floor at 273.88 m measured on PL. VII (table 273.63);
    # its girder (271.37..273.88) all round, carried by curved corbels from panel 29 out to the edge.
    s3 = THIRD_SIDE / 2
    top3 = Element(ifc_class="IfcBuildingElementProxy", name="third_floor")
    top3.add(Element(ifc_class="IfcSlab", name="deck3").add(
        Box(start=P(-s3, -s3, THIRD_FLOOR_Z - 0.3), end=P(s3, s3, THIRD_FLOOR_Z), color=IRON)))
    for face in FACES:
        g = Element(ifc_class="IfcBeam", name=f"girder3_{face}")
        for tag, z in (("bottom", THIRD_GIRDER_BOTTOM), ("top", THIRD_FLOOR_Z - 0.3)):
            g.add(member(tag, [flat_point(face, -s3, z, s3), flat_point(face, s3, z, s3)], 400))
        top3.add(g)
        for side in (-1, 1):  # a corbel near each corner of the face: quarter-ellipse out to the edge
            pts = []
            for k in range(13):
                t = math.pi / 2 * k / 12
                u = side * (SHAFT_HALF[29] + (s3 - SHAFT_HALF[29]) * (1 - math.cos(t)))
                z = Z[29] + (THIRD_GIRDER_BOTTOM - Z[29]) * math.sin(t)
                pts.append(flat_point(face, u, z, SHAFT_HALF[29] + (s3 - SHAFT_HALF[29]) * (1 - math.cos(t))))
            top3.add(member(f"corbel_{face}_{'l' if side < 0 else 'r'}", pts, 400))
    floors.add(top3)
    # campanile, measured on PL. VII: lattice box (half 5.85) up to 280.5, dome to 285.5, drum
    # (half 2.8) to 289.5, cupola to 296.5, pole to 304.
    top = Element(ifc_class="IfcBuildingElementProxy", name="campanile")
    b = CAMPANILE_BOX_HALF
    for face in FACES:
        f = Element(ifc_class="IfcMember", name=f"box_{face}")
        for tag, z in (("sill", THIRD_FLOOR_Z), ("mid", (THIRD_FLOOR_Z + CAMPANILE_BOX_TOP) / 2), ("head", CAMPANILE_BOX_TOP)):
            f.add(member(tag, [flat_point(face, -b, z, b), flat_point(face, b, z, b)], 300))
        for j in range(5):  # lattice: four bays of crossing diagonals
            u0, u1 = -b + 2 * b * j / 4, -b + 2 * b * (j + 1) / 4
            if j < 4:
                f.add(member(f"d1_{j}", [flat_point(face, u0, THIRD_FLOOR_Z, b), flat_point(face, u1, CAMPANILE_BOX_TOP, b)], 200))
                f.add(member(f"d2_{j}", [flat_point(face, u1, THIRD_FLOOR_Z, b), flat_point(face, u0, CAMPANILE_BOX_TOP, b)], 200))
        top.add(f)
    for q, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
        top.add(member(f"post_{q}", [P(sx * b, sy * b, THIRD_FLOOR_Z), P(sx * b, sy * b, CAMPANILE_BOX_TOP)], 450))
    prof = CAMPANILE_PROFILE
    for m in range(8):  # meridian ribs of dome, drum and cupola
        ang = math.pi * 2 * m / 8
        pts = [P(r * math.cos(ang), r * math.sin(ang), z) for z, r in prof]
        top.add(member(f"rib_{m}", pts, 250))
    for z, r in prof[::3]:
        ring = [P(r * math.cos(2 * math.pi * j / 24), r * math.sin(2 * math.pi * j / 24), z) for j in range(25)]
        top.add(member(f"ring_{int(round(z * 10))}", ring, 200))
    top.add(member("mast", [P(0, 0, prof[-1][0]), P(0, 0, CAMPANILE_POLE_TOP)], 150))
    floors.add(top)
    return floors


def build_plinths():
    """The stone bed under each pier: from the ground at the centre (-2.50) to the support centres
    (0), centred on the pier's crossbow square at L0, 25.334 m across (PL. I)."""
    g = Element(ifc_class="IfcBuildingElementProxy", name="masonry")
    h = PLINTH / 2
    inner = 74.338 / 2  # PL. I: 74.338 between the masonry of facing piers, 25.334 each
    for name, sx, sy in (("ne", 1, 1), ("nw", -1, 1), ("sw", -1, -1), ("se", 1, -1)):
        cx = sx * (inner + h)
        cy = sy * (inner + h)
        g.add(Element(ifc_class="IfcFooting", name=f"plinth_{name}").add(
            Box(start=P(cx - h, cy - h, -2.5), end=P(cx + h, cy + h, -0.4), color="#A39E93")))
    return g


def generate_project():
    proj = Project(name="Eiffel Tower")
    storey = Storey(name="ground", elevation=0)
    proj.add_storey(storey)
    tower = Element(ifc_class="IfcBuildingElementProxy", name="tower")
    for name, sx, sy in (("pier_ne", 1, 1), ("pier_nw", -1, 1), ("pier_sw", -1, -1), ("pier_se", 1, -1)):
        tower.add(build_pier(name, sx, sy), carve="none")
    tower.add(build_shaft(), carve="none")
    tower.add(build_floors(), carve="none")
    tower.add(build_plinths(), carve="none")
    storey.add(tower, carve="none")
    return proj


result = generate_project()

if __name__ == "__main__":
    compile_main()
