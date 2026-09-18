"""Lite-STEP project script."""
# ruff: noqa: F401
import math
import numpy as np
from lite_step import compile_main
from lite_step.models import (Project, Storey, Point, Point2D,
    Wall, Window, Door, Slab, Roof, Column, Beam, Space, Site, Element,
    Box, Extrude, Sweep, Sheet, Pipe, Revolve, Mesh, Bar, Material, LayerSet,
    ReferencePoint, GuideLine, Transform, Anchor)
from lite_step.materials import dk, MaterialDef
from lite_step.materials.dk import Render, SheetGeometry

if "ClayTile_Mossy_YellowBrown" not in dk.MATERIALS:
    dk.MATERIALS["ClayTile_Mossy_YellowBrown"] = MaterialDef(
        category="ceramic",
        function="Finish",
        psets={
            "Pset_MaterialCommon": {"MassDensity": 1900},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_m3": 450, "DataSource": "generic"},
        },
        render=Render(rgb="#827B48", alpha=1.0, finish="matt"),
        geometry=SheetGeometry(form="sheet", thicknesses_mm=[15]),
    )

I = lambda x: int(round(x))  # noqa: E731, E741

FOOTPRINT_X = 14000
FOOTPRINT_Y = 12000
SITE_BLOCK_BUFFER = 5000
SITE_BLOCK_DEPTH = 15000

FOUNDATION_SLAB_THICKNESS = 200
FOUNDATION_TERRAIN_CLEARANCE = 50
FOUNDATION_STRIP_WIDTH = 400
FOUNDATION_STRIP_DEPTH = 900
FOUNDATION_STRIP_GAP_MIN = 800
FOUNDATION_STRIP_GAP_MAX = 1400
FOUNDATION_STRIP_EDGE_INSET = 100
FOUNDATION_CLEARING_HEIGHT = 20000
FOUNDATION_CONCRETE = "Concrete_C25-30"

REBAR_COVER = 50
REBAR_PITCH_MAX = 400
REBAR_DIAMETER = 12
_AXIS = ("x", "y", "z")

def _bar_positions(span, cover=REBAR_COVER, pitch_max=REBAR_PITCH_MAX):
    usable = span - 2 * cover
    if usable <= 0:
        return [I(span / 2)]
    n = max(1, -(-usable // pitch_max))
    return [I(cover + usable * k / n) for k in range(int(n) + 1)]

def build_reinforcement(element, cover=REBAR_COVER, pitch_max=REBAR_PITCH_MAX,
                        diameter=REBAR_DIAMETER, grade="B500B"):
    b = element.authored_aabb()
    lo = (b.min.x, b.min.y, b.min.z)
    hi = (b.max.x, b.max.y, b.max.z)
    size = [hi[i] - lo[i] for i in range(3)]
    thin = size.index(min(size))
    mid = (lo[thin] + hi[thin]) / 2
    u, v = [i for i in range(3) if i != thin]

    bars = []
    for k, (run, dist) in enumerate(((u, v), (v, u))):
        ends = _bar_positions(size[run], cover, pitch_max)
        a, z = I(lo[run] + ends[0]), I(lo[run] + ends[-1])
        off = I(mid + (k - 0.5) * diameter)
        for i, p in enumerate(_bar_positions(size[dist], cover, pitch_max)):
            fixed = {_AXIS[thin]: off, _AXIS[dist]: I(lo[dist] + p)}
            bars.append(Bar(name=f"{_AXIS[run]}_{i:02d}",
                            path=[Point(**fixed, **{_AXIS[run]: a}),
                                  Point(**fixed, **{_AXIS[run]: z})],
                            diameter=diameter, grade=grade,
                            mark=f"R{_AXIS[run].upper()}"))
    return bars

def build_site_block(footprint_x, footprint_y,
                     buffer=SITE_BLOCK_BUFFER, depth=SITE_BLOCK_DEPTH,
                     heightmap=None):
    half_x = I(footprint_x / 2) + buffer
    half_y = I(footprint_y / 2) + buffer
    if heightmap is None:
        heightmap = [[0, 0], [0, 0]]
    floor = min(h for row in heightmap for h in row)
    grid = [[I(h - floor) for h in row] for row in heightmap]
    return Mesh(heightmap=grid, depth=depth,
                corner_min=Point2D(x=-half_x, y=-half_y),
                corner_max=Point2D(x=half_x, y=half_y),
                source_type="synthetic")

def _terrain_mesh(site):
    for child in site.elements:
        if (isinstance(child, Element)
                and child.ifc_class == "IfcGeographicElement"
                and child.predefined_type == "TERRAIN"):
            for gc in child.elements:
                if isinstance(gc, Mesh):
                    return gc
    return None

def build_strip_foundation(footprint_x, footprint_y, site,
                           slab_thickness=FOUNDATION_SLAB_THICKNESS,
                           terrain_clearance=FOUNDATION_TERRAIN_CLEARANCE,
                           strip_width=FOUNDATION_STRIP_WIDTH,
                           strip_depth=FOUNDATION_STRIP_DEPTH,
                           strip_gap_min=FOUNDATION_STRIP_GAP_MIN,
                           strip_gap_max=FOUNDATION_STRIP_GAP_MAX,
                           strip_edge_inset=FOUNDATION_STRIP_EDGE_INSET,
                           clearing_height=FOUNDATION_CLEARING_HEIGHT):
    mesh = _terrain_mesh(site)
    z0 = mesh.height_at(x=0, y=0, round=True) if mesh is not None else 0
    slab_top = z0 + terrain_clearance
    slab_bottom = slab_top - slab_thickness
    strip_top = slab_bottom
    strip_bottom = slab_bottom - strip_depth

    hx, hy = I(footprint_x / 2), I(footprint_y / 2)

    slab = Slab(name="foundation")
    slab.add(Box(start=Point(x=-hx, y=-hy, z=slab_bottom),
                 end=Point(x=hx, y=hy, z=slab_top),
                 type="foundation", material=FOUNDATION_CONCRETE))
    slab.add(*build_reinforcement(slab))

    if footprint_x >= footprint_y:
        long_dim, short_dim, long_is_x = footprint_x, footprint_y, True
    else:
        long_dim, short_dim, long_is_x = footprint_y, footprint_x, False

    centre_span = long_dim - strip_width - 2 * strip_edge_inset
    if centre_span <= 0:
        centres = [0]
    else:
        pitch_max = strip_gap_max + strip_width
        n = max(1, -(-centre_span // pitch_max))
        centres = [I(-centre_span / 2 + centre_span * k / n) for k in range(n + 1)]

    half_len = I(short_dim / 2) - strip_edge_inset
    half_w = I(strip_width / 2)

    strips = []
    for i, c in enumerate(centres):
        if long_is_x:
            start = Point(x=c - half_w, y=-half_len, z=strip_bottom)
            end = Point(x=c + half_w, y=half_len, z=strip_top)
        else:
            start = Point(x=-half_len, y=c - half_w, z=strip_bottom)
            end = Point(x=half_len, y=c + half_w, z=strip_top)
        footing = Element(ifc_class="IfcFooting", name=f"strip_{i}")
        footing.add(Box(start=start, end=end, material=FOUNDATION_CONCRETE))
        footing.add(*build_reinforcement(footing))
        strips.append(footing)

    clearing_void = Box(start=Point(x=-hx, y=-hy, z=slab_top),
                        end=Point(x=hx, y=hy, z=slab_top + clearing_height))

    return slab, strips, clearing_void

def build_ground_floor(footprint_x, footprint_y, corner_r, base_z, top_z):
    hx, hy = I(footprint_x / 2), I(footprint_y / 2)
    cx, cy = hx - corner_r, -hy + corner_r
    wall_thick = 328
    elements = []

    s_wall = Wall(name="south_ground")
    s_wall.add(Box(start=Point(x=-hx, y=-hy, z=base_z),
                   end=Point(x=cx, y=-hy + wall_thick, z=top_z),
                   material="Brick_Red_DK", color="#A34129"))
    elements.append(s_wall)

    e_wall = Wall(name="east_ground")
    e_wall.add(Box(start=Point(x=hx - wall_thick, y=cy, z=base_z),
                   end=Point(x=hx, y=hy, z=top_z),
                   material="Brick_Red_DK", color="#A34129"))
    elements.append(e_wall)

    n_wall = Wall(name="north_ground")
    n_wall.add(Box(start=Point(x=-hx, y=hy - wall_thick, z=base_z),
                   end=Point(x=hx, y=hy, z=top_z),
                   material="Brick_Red_DK", color="#A34129"))
    elements.append(n_wall)

    w_wall = Wall(name="west_ground")
    w_wall.add(Box(start=Point(x=-hx, y=-hy, z=base_z),
                   end=Point(x=-hx + wall_thick, y=hy, z=top_z),
                   material="Brick_Red_DK", color="#A34129"))
    elements.append(w_wall)

    n_segments = 24
    outer_pts = []
    inner_pts = []
    for i in range(n_segments + 1):
        angle = -math.pi / 2 + (math.pi / 2) * (i / n_segments)
        ox = cx + corner_r * math.cos(angle)
        oy = cy + corner_r * math.sin(angle)
        outer_pts.append(Point(x=I(ox), y=I(oy), z=base_z))
        ix = cx + (corner_r - wall_thick) * math.cos(angle)
        iy = cy + (corner_r - wall_thick) * math.sin(angle)
        inner_pts.append(Point(x=I(ix), y=I(iy), z=base_z))

    corner_contour = outer_pts + inner_pts[::-1]
    c_wall = Element(ifc_class="IfcWall", name="corner_ground")
    c_wall.add(Extrude(contour=corner_contour, thickness=top_z - base_z,
                       material="Brick_Red_DK", color="#A34129"), carve="none")
    elements.append(c_wall)

    door_cx = -2000
    steps_group = Element(ifc_class="IfcStair", name="entrance_steps")
    steps_group.add(Box(start=Point(x=door_cx - 850, y=-hy - 350, z=base_z),
                        end=Point(x=door_cx + 850, y=-hy, z=base_z + 120),
                        material="Brick_Red_DK", color="#8B3622"))
    steps_group.add(Box(start=Point(x=door_cx - 950, y=-hy - 700, z=base_z - 80),
                        end=Point(x=door_cx + 950, y=-hy - 350, z=base_z + 40),
                        material="Brick_Red_DK", color="#8B3622"))
    steps_group.add(Box(start=Point(x=door_cx - 1050, y=-hy - 1050, z=base_z - 150),
                        end=Point(x=door_cx + 1050, y=-hy - 700, z=base_z - 40),
                        material="Brick_Red_DK", color="#8B3622"))
    elements.append(steps_group)

    east_vent_ys = [-2000, 0, 2000, 4000]
    for i, vy in enumerate(east_vent_ys):
        vent_box = Element(ifc_class="IfcBuildingElementProxy", name=f"basement_vent_{i}")
        vent_box.add(Box(start=Point(x=hx - wall_thick - 20, y=vy - 200, z=base_z + 50),
                         end=Point(x=hx + 10, y=vy + 200, z=base_z + 250),
                         material="Steel_Stainless_304", color="#2A2A2A"))
        for s in range(4):
            sz = base_z + 70 + s * 45
            vent_box.add(Box(start=Point(x=hx - 30, y=vy - 180, z=sz),
                             end=Point(x=hx + 5, y=vy + 180, z=sz + 15),
                             material="Steel_Stainless_304", color="#4A4A4A"))
        elements.append(vent_box)

    return elements, s_wall, e_wall, c_wall

def build_second_floor(footprint_x, footprint_y, corner_r, base_z, top_z):
    hx, hy = I(footprint_x / 2), I(footprint_y / 2)
    cx, cy = hx - corner_r, -hy + corner_r
    wall_thick = 328
    elements = []

    deck = Slab(name="deck_second")
    deck.add(Box(start=Point(x=-hx + wall_thick, y=-hy + wall_thick, z=base_z - 220),
                 end=Point(x=hx - wall_thick, y=hy - wall_thick, z=base_z),
                 material="Concrete_C30-37", color="#B0B0B0"))
    elements.append(deck)

    seam_h = 100
    seam_proud = 25
    seam_group = Element(ifc_class="IfcBuildingElementProxy", name="brick_seam_band")
    seam_group.add(Box(start=Point(x=-hx, y=-hy - seam_proud, z=base_z),
                       end=Point(x=cx, y=-hy, z=base_z + seam_h),
                       material="Brick_Red_DK", color="#8B3622"))
    seam_group.add(Box(start=Point(x=hx, y=cy, z=base_z),
                       end=Point(x=hx + seam_proud, y=hy, z=base_z + seam_h),
                       material="Brick_Red_DK", color="#8B3622"))
    seam_group.add(Box(start=Point(x=-hx, y=hy, z=base_z),
                       end=Point(x=hx, y=hy + seam_proud, z=base_z + seam_h),
                       material="Brick_Red_DK", color="#8B3622"))
    seam_group.add(Box(start=Point(x=-hx - seam_proud, y=-hy, z=base_z),
                       end=Point(x=-hx, y=hy, z=base_z + seam_h),
                       material="Brick_Red_DK", color="#8B3622"))

    n_segments = 24
    s_outer_pts = []
    s_inner_pts = []
    for i in range(n_segments + 1):
        angle = -math.pi / 2 + (math.pi / 2) * (i / n_segments)
        ox = cx + (corner_r + seam_proud) * math.cos(angle)
        oy = cy + (corner_r + seam_proud) * math.sin(angle)
        s_outer_pts.append(Point(x=I(ox), y=I(oy), z=base_z))
        ix = cx + corner_r * math.cos(angle)
        iy = cy + corner_r * math.sin(angle)
        s_inner_pts.append(Point(x=I(ix), y=I(iy), z=base_z))
    seam_contour = s_outer_pts + s_inner_pts[::-1]
    seam_group.add(Extrude(contour=seam_contour, thickness=seam_h,
                           material="Brick_Red_DK", color="#8B3622"), carve="none")
    elements.append(seam_group)

    s_wall2 = Wall(name="south_second")
    s_wall2.add(Box(start=Point(x=-hx, y=-hy, z=base_z),
                    end=Point(x=cx, y=-hy + wall_thick, z=top_z),
                    material="Brick_Red_DK", color="#A34129"))
    elements.append(s_wall2)

    e_wall2 = Wall(name="east_second")
    e_wall2.add(Box(start=Point(x=hx - wall_thick, y=cy, z=base_z),
                    end=Point(x=hx, y=hy, z=top_z),
                    material="Brick_Red_DK", color="#A34129"))
    elements.append(e_wall2)

    n_wall2 = Wall(name="north_second")
    n_wall2.add(Box(start=Point(x=-hx, y=hy - wall_thick, z=base_z),
                    end=Point(x=hx, y=hy, z=top_z),
                    material="Brick_Red_DK", color="#A34129"))
    elements.append(n_wall2)

    w_wall2 = Wall(name="west_second")
    w_wall2.add(Box(start=Point(x=-hx, y=-hy, z=base_z),
                    end=Point(x=-hx + wall_thick, y=hy, z=top_z),
                    material="Brick_Red_DK", color="#A34129"))
    elements.append(w_wall2)

    outer_pts2 = []
    inner_pts2 = []
    for i in range(n_segments + 1):
        angle = -math.pi / 2 + (math.pi / 2) * (i / n_segments)
        ox = cx + corner_r * math.cos(angle)
        oy = cy + corner_r * math.sin(angle)
        outer_pts2.append(Point(x=I(ox), y=I(oy), z=base_z))
        ix = cx + (corner_r - wall_thick) * math.cos(angle)
        iy = cy + (corner_r - wall_thick) * math.sin(angle)
        inner_pts2.append(Point(x=I(ix), y=I(iy), z=base_z))

    corner_contour2 = outer_pts2 + inner_pts2[::-1]
    c_wall2 = Element(ifc_class="IfcWall", name="corner_second")
    c_wall2.add(Extrude(contour=corner_contour2, thickness=top_z - base_z,
                        material="Brick_Red_DK", color="#A34129"), carve="none")
    elements.append(c_wall2)

    return elements, s_wall2, e_wall2, c_wall2

def build_openings_and_joinery(footprint_x, footprint_y, corner_r):
    hx, hy = I(footprint_x / 2), I(footprint_y / 2)
    cx, cy = hx - corner_r, -hy + corner_r
    wall_thick = 328
    gf_elements = []
    sf_elements = []

    w1_cx = -5200
    w1_w, w1_h = 2200, 2400
    w1_sill = 450
    w1_left_w = 650

    win_large = Element(ifc_class="IfcWindow", name="win_far_left_large")
    win_large.add(Box(start=Point(x=w1_cx - I(w1_w / 2), y=-hy - 10, z=w1_sill),
                      end=Point(x=w1_cx + I(w1_w / 2), y=-hy + 100, z=w1_sill + w1_h),
                      material="Timber_C24", color="#FFFFFF"))
    mullion_x = w1_cx - I(w1_w / 2) + w1_left_w
    win_large.add(Box(start=Point(x=mullion_x - 30, y=-hy - 15, z=w1_sill),
                      end=Point(x=mullion_x + 30, y=-hy + 90, z=w1_sill + w1_h),
                      material="Timber_C24", color="#FFFFFF"))
    grille_h = 600
    win_large.add(Box(start=Point(x=w1_cx - I(w1_w / 2), y=-hy - 15, z=w1_sill + w1_h - grille_h),
                      end=Point(x=mullion_x, y=-hy + 90, z=w1_sill + w1_h - grille_h + 50),
                      material="Timber_C24", color="#FFFFFF"))
    for g in range(8):
        gz = w1_sill + w1_h - grille_h + 60 + g * 60
        win_large.add(Box(start=Point(x=w1_cx - I(w1_w / 2) + 40, y=-hy - 5, z=gz),
                          end=Point(x=mullion_x - 40, y=-hy + 20, z=gz + 20),
                          material="Steel_Stainless_304", color="#404040"))
    win_large.add(Box(start=Point(x=w1_cx - I(w1_w / 2) + 40, y=-hy + 40, z=w1_sill + 40),
                      end=Point(x=mullion_x - 40, y=-hy + 50, z=w1_sill + w1_h - grille_h - 10),
                      material="Glass_Monolithic", color="glass"))
    win_large.add(Box(start=Point(x=mullion_x + 40, y=-hy + 40, z=w1_sill + 40),
                      end=Point(x=w1_cx + I(w1_w / 2) - 40, y=-hy + 50, z=w1_sill + w1_h - 40),
                      material="Glass_Monolithic", color="glass"))
    n_blinds = 32
    blind_spacing = I((w1_h - 80) / n_blinds)
    for b in range(n_blinds):
        bz = w1_sill + 50 + b * blind_spacing
        win_large.add(Box(start=Point(x=mullion_x + 35, y=-hy + 20, z=bz),
                          end=Point(x=w1_cx + I(w1_w / 2) - 35, y=-hy + 35, z=bz + 12),
                          material="Steel_Stainless_304", color="#E0E0E0"))
    win_large.add(Box(start=Point(x=w1_cx - I(w1_w / 2) - 50, y=-hy - 60, z=w1_sill - 40),
                      end=Point(x=w1_cx + I(w1_w / 2) + 50, y=-hy + 120, z=w1_sill),
                      material="Zinc_Titanium_EN988", color="#7A8B99"))
    gf_elements.append(win_large)

    door_cx = -2000
    door_w = 1200
    door_h = 2100
    transom_h = 500
    door_sill = 50

    door_unit = Element(ifc_class="IfcDoor", name="front_entrance_door")
    door_unit.add(Box(start=Point(x=door_cx - I(door_w / 2), y=-hy + 80, z=door_sill),
                      end=Point(x=door_cx + I(door_w / 2), y=-hy + 220, z=door_sill + door_h + transom_h),
                      material="Timber_C24", color="#FFFFFF"))
    door_unit.add(Box(start=Point(x=door_cx - I(door_w / 2) + 60, y=-hy + 130, z=door_sill + 10),
                      end=Point(x=door_cx + I(door_w / 2) - 60, y=-hy + 180, z=door_sill + door_h),
                      material="Timber_C24", color="#F8F8F8"))
    for row in range(3):
        pz_low = door_sill + 150 + row * 600
        pz_high = pz_low + 480
        for col in range(2):
            px_low = door_cx - I(door_w / 2) + 120 + col * 480
            px_high = px_low + 400
            door_unit.add(Box(start=Point(x=px_low, y=-hy + 115, z=pz_low),
                              end=Point(x=px_high, y=-hy + 135, z=pz_high),
                              material="Timber_C24", color="#EBEBEB"))
    door_unit.add(Box(start=Point(x=door_cx + I(door_w / 2) - 180, y=-hy + 90, z=door_sill + 1000),
                      end=Point(x=door_cx + I(door_w / 2) - 80, y=-hy + 140, z=door_sill + 1040),
                      material="Steel_Stainless_304", color="#C0C0C0"))
    door_unit.add(Box(start=Point(x=door_cx - I(door_w / 2), y=-hy + 70, z=door_sill + door_h),
                      end=Point(x=door_cx + I(door_w / 2), y=-hy + 230, z=door_sill + door_h + 60),
                      material="Timber_C24", color="#FFFFFF"))
    door_unit.add(Box(start=Point(x=door_cx - I(door_w / 2) + 60, y=-hy + 140, z=door_sill + door_h + 70),
                      end=Point(x=door_cx + I(door_w / 2) - 60, y=-hy + 150, z=door_sill + door_h + transom_h - 20),
                      material="Glass_Monolithic", color="glass"))
    for sp in [-180, 180]:
        door_unit.add(Box(start=Point(x=door_cx + sp - 15, y=-hy + 130, z=door_sill + door_h + 70),
                          end=Point(x=door_cx + sp + 15, y=-hy + 160, z=door_sill + door_h + transom_h - 20),
                          material="Timber_C24", color="#FFFFFF"))
    gf_elements.append(door_unit)

    for i, wx in enumerate([1200, 3200]):
        win_r = Element(ifc_class="IfcWindow", name=f"win_front_ground_right_{i}")
        win_r.add(Box(start=Point(x=wx - 700, y=-hy - 10, z=900),
                      end=Point(x=wx + 700, y=-hy + 100, z=2700),
                      material="Timber_C24", color="#FFFFFF"))
        win_r.add(Box(start=Point(x=wx - 30, y=-hy - 15, z=900),
                      end=Point(x=wx + 30, y=-hy + 90, z=2700),
                      material="Timber_C24", color="#FFFFFF"))
        win_r.add(Box(start=Point(x=wx - 700, y=-hy - 15, z=1950),
                      end=Point(x=wx + 700, y=-hy + 90, z=2010),
                      material="Timber_C24", color="#FFFFFF"))
        for px_lo, px_hi in [(wx - 650, wx - 40), (wx + 40, wx + 650)]:
            win_r.add(Box(start=Point(x=px_lo, y=-hy + 40, z=950),
                          end=Point(x=px_hi, y=-hy + 50, z=1940),
                          material="Glass_Monolithic", color="glass"))
            win_r.add(Box(start=Point(x=px_lo, y=-hy + 40, z=2020),
                          end=Point(x=px_hi, y=-hy + 50, z=2650),
                          material="Glass_Monolithic", color="glass"))
        win_r.add(Box(start=Point(x=wx - 750, y=-hy - 60, z=860),
                      end=Point(x=wx + 750, y=-hy + 120, z=900),
                      material="Zinc_Titanium_EN988", color="#7A8B99"))
        gf_elements.append(win_r)

    for i, wy in enumerate([-2000, 0, 2000, 4000]):
        win_side = Element(ifc_class="IfcWindow", name=f"win_side_ground_{i}")
        win_side.add(Box(start=Point(x=hx - 100, y=wy - 400, z=900),
                         end=Point(x=hx + 10, y=wy + 400, z=2700),
                         material="Timber_C24", color="#FFFFFF"))
        win_side.add(Box(start=Point(x=hx - 90, y=wy - 400, z=1950),
                         end=Point(x=hx + 15, y=wy + 400, z=2010),
                         material="Timber_C24", color="#FFFFFF"))
        win_side.add(Box(start=Point(x=hx - 50, y=wy - 350, z=950),
                         end=Point(x=hx - 40, y=wy + 350, z=1940),
                         material="Glass_Monolithic", color="glass"))
        win_side.add(Box(start=Point(x=hx - 50, y=wy - 350, z=2020),
                         end=Point(x=hx - 40, y=wy + 350, z=2650),
                         material="Glass_Monolithic", color="glass"))
        win_side.add(Box(start=Point(x=hx - 120, y=wy - 450, z=860),
                         end=Point(x=hx + 60, y=wy + 450, z=900),
                         material="Zinc_Titanium_EN988", color="#7A8B99"))
        gf_elements.append(win_side)

    for i, wx in enumerate([-5200, -2000, 1200, 3200]):
        win_sf = Element(ifc_class="IfcWindow", name=f"win_front_second_{i}")
        win_sf.add(Box(start=Point(x=wx - 700, y=-hy - 10, z=4100),
                       end=Point(x=wx + 700, y=-hy + 100, z=5700),
                       material="Timber_C24", color="#FFFFFF"))
        win_sf.add(Box(start=Point(x=wx - 30, y=-hy - 15, z=4100),
                       end=Point(x=wx + 30, y=-hy + 90, z=5700),
                       material="Timber_C24", color="#FFFFFF"))
        win_sf.add(Box(start=Point(x=wx - 700, y=-hy - 15, z=5050),
                       end=Point(x=wx + 700, y=-hy + 90, z=5110),
                       material="Timber_C24", color="#FFFFFF"))
        for px_lo, px_hi in [(wx - 650, wx - 40), (wx + 40, wx + 650)]:
            win_sf.add(Box(start=Point(x=px_lo, y=-hy + 40, z=4150),
                           end=Point(x=px_hi, y=-hy + 50, z=5040),
                           material="Glass_Monolithic", color="glass"))
            win_sf.add(Box(start=Point(x=px_lo, y=-hy + 40, z=5120),
                           end=Point(x=px_hi, y=-hy + 50, z=5650),
                           material="Glass_Monolithic", color="glass"))
        win_sf.add(Box(start=Point(x=wx - 750, y=-hy - 60, z=4060),
                       end=Point(x=wx + 750, y=-hy + 120, z=4100),
                       material="Zinc_Titanium_EN988", color="#7A8B99"))
        sf_elements.append(win_sf)

    for i, wy in enumerate([-2000, 0, 2000]):
        win_side2 = Element(ifc_class="IfcWindow", name=f"win_side_second_{i}")
        win_side2.add(Box(start=Point(x=hx - 100, y=wy - 400, z=4100),
                          end=Point(x=hx + 10, y=wy + 400, z=5700),
                          material="Timber_C24", color="#FFFFFF"))
        win_side2.add(Box(start=Point(x=hx - 90, y=wy - 400, z=5050),
                          end=Point(x=hx + 15, y=wy + 400, z=5110),
                          material="Timber_C24", color="#FFFFFF"))
        win_side2.add(Box(start=Point(x=hx - 50, y=wy - 350, z=4150),
                          end=Point(x=hx - 40, y=wy + 350, z=5040),
                          material="Glass_Monolithic", color="glass"))
        win_side2.add(Box(start=Point(x=hx - 50, y=wy - 350, z=5120),
                          end=Point(x=hx - 40, y=wy + 350, z=5650),
                          material="Glass_Monolithic", color="glass"))
        win_side2.add(Box(start=Point(x=hx - 120, y=wy - 450, z=4060),
                          end=Point(x=hx + 60, y=wy + 450, z=4100),
                          material="Zinc_Titanium_EN988", color="#7A8B99"))
        sf_elements.append(win_side2)

    porthole_r = 500
    mid_angle = -math.pi / 4
    pc_x = cx + (corner_r - I(wall_thick / 2)) * math.cos(mid_angle)
    pc_y = cy + (corner_r - I(wall_thick / 2)) * math.sin(mid_angle)
    pc_z = 4800

    porthole = Element(ifc_class="IfcWindow", name="porthole_corner_second")
    n_ring = 32
    u_x, u_y = -math.sin(mid_angle), math.cos(mid_angle)
    n_x, n_y = math.cos(mid_angle), math.sin(mid_angle)

    for i in range(n_ring):
        a1 = 2 * math.pi * i / n_ring
        a2 = 2 * math.pi * (i + 1) / n_ring
        p1 = Point(x=I(pc_x + porthole_r * math.cos(a1) * u_x + 60 * n_x),
                   y=I(pc_y + porthole_r * math.cos(a1) * u_y + 60 * n_y),
                   z=I(pc_z + porthole_r * math.sin(a1)))
        p2 = Point(x=I(pc_x + porthole_r * math.cos(a2) * u_x + 60 * n_x),
                   y=I(pc_y + porthole_r * math.cos(a2) * u_y + 60 * n_y),
                   z=I(pc_z + porthole_r * math.sin(a2)))
        porthole.add(Sweep(path=[p1, p2],
                           material=Material(key="Timber_C24", profile_mm=(45, 95)),
                           color="#FFFFFF"))

    p_horiz_1 = Point(x=I(pc_x - (porthole_r - 35) * u_x + 20 * n_x),
                      y=I(pc_y - (porthole_r - 35) * u_y + 20 * n_y), z=pc_z)
    p_horiz_2 = Point(x=I(pc_x + (porthole_r - 35) * u_x + 20 * n_x),
                      y=I(pc_y + (porthole_r - 35) * u_y + 20 * n_y), z=pc_z)
    porthole.add(Sweep(path=[p_horiz_1, p_horiz_2],
                       material=Material(key="Timber_C24", profile_mm=(45, 45)),
                       color="#FFFFFF"))

    p_vert_1 = Point(x=I(pc_x + 20 * n_x), y=I(pc_y + 20 * n_y), z=pc_z - porthole_r + 35)
    p_vert_2 = Point(x=I(pc_x + 20 * n_x), y=I(pc_y + 20 * n_y), z=pc_z + porthole_r - 35)
    porthole.add(Sweep(path=[p_vert_1, p_vert_2],
                       material=Material(key="Timber_C24", profile_mm=(45, 45)),
                       color="#FFFFFF"))

    g_pts = []
    for i in range(n_ring):
        a = 2 * math.pi * i / n_ring
        gx = pc_x + (porthole_r - 30) * math.cos(a) * u_x
        gy = pc_y + (porthole_r - 30) * math.cos(a) * u_y
        gz = pc_z + (porthole_r - 30) * math.sin(a)
        g_pts.append(Point(x=I(gx), y=I(gy), z=I(gz)))
    porthole.add(Extrude(contour=g_pts, thickness=12,
                         material="Glass_Monolithic", color="glass"), carve="none")
    sf_elements.append(porthole)

    return gf_elements, sf_elements

def build_curved_hipped_roof(footprint_x, footprint_y, corner_r, wall_top_z):
    hx, hy = I(footprint_x / 2), I(footprint_y / 2)
    cx, cy = hx - corner_r, -hy + corner_r
    overhang = 300
    eave_z = wall_top_z + 50
    ridge_z = wall_top_z + 2400

    ox_w = -hx - overhang
    ox_e = hx + overhang
    oy_s = -hy - overhang
    oy_n = hy + overhang
    r_eave = corner_r + overhang

    rx_w = -hx + 3500
    rx_e = cx - 1500
    ry = 0

    roof_elem = Roof(name="curved_hipped_roof")

    fascia = Element(ifc_class="IfcCovering", name="eave_fascia")
    fascia.add(Box(start=Point(x=-hx - overhang, y=oy_s, z=wall_top_z - 60),
                   end=Point(x=cx, y=oy_s + 60, z=eave_z),
                   material="Timber_C24", color="#5C4033"))
    fascia.add(Box(start=Point(x=ox_e - 60, y=cy, z=wall_top_z - 60),
                   end=Point(x=ox_e, y=oy_n, z=eave_z),
                   material="Timber_C24", color="#5C4033"))
    fascia.add(Box(start=Point(x=-hx - overhang, y=oy_n - 60, z=wall_top_z - 60),
                   end=Point(x=ox_e, y=oy_n, z=eave_z),
                   material="Timber_C24", color="#5C4033"))
    fascia.add(Box(start=Point(x=-hx - overhang, y=oy_s, z=wall_top_z - 60),
                   end=Point(x=-hx - overhang + 60, y=oy_n, z=eave_z),
                   material="Timber_C24", color="#5C4033"))
    roof_elem.add(fascia)

    verts = []
    verts.append(Point(x=rx_w, y=ry, z=ridge_z))
    verts.append(Point(x=rx_e, y=ry, z=ridge_z))
    verts.append(Point(x=ox_w, y=oy_n, z=eave_z))
    verts.append(Point(x=ox_e, y=oy_n, z=eave_z))
    verts.append(Point(x=ox_e, y=cy, z=eave_z))

    n_arc = 12
    arc_start_idx = len(verts)
    for i in range(n_arc + 1):
        angle = -math.pi / 2 + (math.pi / 2) * (i / n_arc)
        px = cx + r_eave * math.cos(angle)
        py = cy + r_eave * math.sin(angle)
        verts.append(Point(x=I(px), y=I(py), z=eave_z))
    arc_end_idx = len(verts) - 1

    sw_idx = len(verts)
    verts.append(Point(x=ox_w, y=oy_s, z=eave_z))

    faces = []
    # Inverted triangles (clockwise winding when viewed from exterior)
    faces.append((2, 3, 1))
    faces.append((2, 1, 0))
    faces.append((2, 0, sw_idx))
    faces.append((sw_idx, 0, 1))
    faces.append((sw_idx, 1, arc_start_idx))

    for i in range(n_arc):
        a1 = arc_start_idx + i
        a2 = arc_start_idx + i + 1
        faces.append((a1, 1, a2))

    faces.append((arc_end_idx, 1, 3))

    roof_mesh = Mesh(name="roof_tiles", vertices=verts, faces=faces,
                     material="ClayTile_Mossy_YellowBrown", color="#827B48")
    roof_elem.add(roof_mesh, carve="none")

    ridge_cap = Element(ifc_class="IfcMember", name="roof_ridge_cap")
    ridge_cap.add(Pipe(path=[Point(x=rx_w - 50, y=ry, z=ridge_z + 40),
                             Point(x=rx_e + 50, y=ry, z=ridge_z + 40)],
                       radius=110, color="#6E6738"))
    ridge_cap.add(Pipe(path=[Point(x=ox_w, y=oy_n, z=eave_z + 30),
                             Point(x=rx_w, y=ry, z=ridge_z + 30)],
                       radius=90, color="#6E6738"))
    ridge_cap.add(Pipe(path=[Point(x=ox_w, y=oy_s, z=eave_z + 30),
                             Point(x=rx_w, y=ry, z=ridge_z + 30)],
                       radius=90, color="#6E6738"))
    ridge_cap.add(Pipe(path=[Point(x=ox_e, y=oy_n, z=eave_z + 30),
                             Point(x=rx_e, y=ry, z=ridge_z + 30)],
                       radius=90, color="#6E6738"))
    roof_elem.add(ridge_cap)

    return roof_elem

def generate_project():
    proj = Project(name="Red Brick Corner Building")
    s_ground = Storey(elevation=0, name="ground")
    s_second = Storey(elevation=3200, name="second")
    proj.add_storey(s_ground)
    proj.add_storey(s_second)

    heightmap = [
        [-100, -100, -100, -100, -100],
        [-50,  -50,  -50,  -50,  -50],
        [0,    0,    0,    0,    0],
        [50,   50,   50,   50,   50],
        [100,  100,  100,  100,  100],
    ]
    site = Site(name="site")
    terrain = Element(ifc_class="IfcGeographicElement",
                      predefined_type="TERRAIN", name="terrain")
    terrain.add(build_site_block(FOOTPRINT_X, FOOTPRINT_Y, heightmap=heightmap))
    site.add(terrain)
    proj.add(site)

    slab, strips, clearing_void = build_strip_foundation(
        FOOTPRINT_X, FOOTPRINT_Y, site)
    s_ground.add(slab, *strips)
    site.void(clearing_void)

    CORNER_R = 2500
    WALL_BASE_Z = 50
    GROUND_TOP_Z = 3200
    gf_elements, s_wall, e_wall, c_wall = build_ground_floor(
        FOOTPRINT_X, FOOTPRINT_Y, CORNER_R, WALL_BASE_Z, GROUND_TOP_Z)
    s_ground.add(*gf_elements)

    SECOND_TOP_Z = 6400
    sf_elements, s_wall2, e_wall2, c_wall2 = build_second_floor(
        FOOTPRINT_X, FOOTPRINT_Y, CORNER_R, GROUND_TOP_Z, SECOND_TOP_Z)
    s_second.add(*sf_elements)

    gf_openings, sf_openings = build_openings_and_joinery(
        FOOTPRINT_X, FOOTPRINT_Y, CORNER_R)
    s_ground.add(*gf_openings)
    s_second.add(*sf_openings)

    roof = build_curved_hipped_roof(FOOTPRINT_X, FOOTPRINT_Y, CORNER_R, SECOND_TOP_Z)
    s_second.add(roof)

    return proj

result = generate_project()

if __name__ == "__main__":
    compile_main()
