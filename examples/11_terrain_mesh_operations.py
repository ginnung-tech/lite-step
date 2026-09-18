"""Procedural Mountain Terrain & Observation Tower Example.

Demonstrates:
- 100m x 100m alpine terrain heightmap with sharp mountain summits and natural ridgelines.
- The 4 chainable Mesh geometric processing methods:
  * .subdivide(height_map_only=True) — upsampling grid resolution 1 -> 4
  * .add_random(seed=42, strength=25, walk=6, height_map_only=True) — coherent natural terrain perturbation
  * .smooth(strength=30, height_map_only=True) — Laplacian contour relaxation
  * .simplify(strength=20, height_map_only=True) — triangle decimation
- Structural catenary suspension wire between the two highest summits.
- Cylindrical brick observation tower on Peak 2 with an internal spiral staircase,
  entrance portal, viewing slit windows, and cantilevered viewing platform.

Compile with::

    python examples/11_terrain_mesh_operations.py
"""

import math
from pathlib import Path
import sys
import numpy as np

# Ensure repo root is on import path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from lite_step import compile_main  # noqa: E402
from lite_step.models import (  # noqa: E402
    Beam,
    Box,
    Column,
    Element,
    Extrude,
    Material,
    Mesh,
    Pipe,
    Point,
    Point2D,
    Project,
    Revolve,
    Site,
    Slab,
    Storey,
    Wall,
)

I = lambda x: int(round(x))  # noqa: E731,E741


def generate_project() -> Project:
    proj = Project(name="Terrain and Mountain Tower Project")
    site = Site(name="site")

    # Generate 100m x 100m jagged terrain with max elevation 80m (80,000 mm)
    np.random.seed(42)
    grid_size = 31
    x_min, x_max = -50000, 50000
    y_min, y_max = -50000, 50000

    # Base sharp mountain peaks
    peaks = [
        (10, 10, 80000, 4.5),
        (22, 22, 76000, 5.0),
        (8, 24, 62000, 4.0),
        (24, 8, 65000, 4.2),
        (16, 16, 52000, 6.0),
        (4, 14, 58000, 3.8),
        (28, 18, 55000, 4.0),
    ]

    heights = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            h_val = 8000.0
            for pr, pc, p_height, p_width in peaks:
                dist = np.hypot(r - pr, c - pc)
                h_val += p_height * np.exp(-dist / p_width) * (0.85 + 0.15 * np.cos(dist * 3.5))
            noise = np.random.uniform(-3000, 3000) + 1500 * np.sin(r * 1.5) * np.cos(c * 1.5)
            heights[r, c] = max(2000.0, h_val + noise)

    # Normalize peak to exactly 80,000 mm
    heights = heights * (80000.0 / heights.max())
    heightmap = [[I(heights[r, c]) for c in range(grid_size)] for r in range(grid_size)]

    terrain_mesh = Mesh(
        name="terrain_mesh",
        heightmap=heightmap,
        corner_min=Point2D(x=x_min, y=y_min),
        corner_max=Point2D(x=x_max, y=y_max),
        depth=10000,
        color="site",
    )

    # ── DEMONSTRATE THE 4 CHAINABLE MESH METHODS ─────────────────────────
    # 1. Subdivide: upsample grid resolution
    # 2. Add random: add coherent ridgeline perturbation (walk=6, strength=10000 mm = 10m)
    # 3. Smooth: Laplacian relaxation to blend ridgelines into alpine contours
    # 4. Simplify: decimate grid for optimized polygon count
    terrain_mesh.subdivide(height_map_only=True).add_random(
        seed=42, strength=10000, walk=6, height_map_only=True
    ).smooth(
        strength=25, height_map_only=True
    ).simplify(
        strength=15, height_map_only=True
    )

    terrain_element = Element(
        ifc_class="IfcGeographicElement",
        predefined_type="TERRAIN",
        name="terrain",
    ).add(terrain_mesh)

    # Identify the two highest peaks from updated mesh
    updated_hm = np.array(terrain_mesh.heightmap)
    r1, c1 = np.unravel_index(np.argmax(updated_hm), updated_hm.shape)
    z1 = float(updated_hm[r1, c1])
    h_rows, h_cols = updated_hm.shape
    x1 = x_min + c1 * (x_max - x_min) / (h_cols - 1)
    y1 = y_min + r1 * (y_max - y_min) / (h_rows - 1)

    min_dist_cells = h_rows // 3
    mask = np.ones_like(updated_hm, dtype=bool)
    for r in range(h_rows):
        for c in range(h_cols):
            if np.hypot(r - r1, c - c1) < min_dist_cells:
                mask[r, c] = False

    masked_heights = np.where(mask, updated_hm, 0)
    r2, c2 = np.unravel_index(np.argmax(masked_heights), updated_hm.shape)
    z2 = float(updated_hm[r2, c2])
    x2 = x_min + c2 * (x_max - x_min) / (h_cols - 1)
    y2 = y_min + r2 * (y_max - y_min) / (h_rows - 1)

    # Model structural catenary wire between Peak 1 and Peak 2
    span_horiz = math.hypot(x2 - x1, y2 - y1)
    sag = 0.12 * span_horiz

    num_samples = 60
    wire_points = []
    for i in range(num_samples + 1):
        t = i / float(num_samples)
        wx = (1.0 - t) * x1 + t * x2
        wy = (1.0 - t) * y1 + t * y2
        wz = (1.0 - t) * z1 + t * z2 - 4.0 * sag * t * (1.0 - t)
        wire_points.append(Point(x=I(wx), y=I(wy), z=I(wz)))

    wire_geom = Pipe(
        name="catenary_cable_geom",
        path=wire_points,
        radius=75,
        color="#222222",
    )

    anchor1 = Column(name="summit_anchor_1").add(
        Box(
            name="pylon_1",
            start=Point(x=I(x1 - 300), y=I(y1 - 300), z=I(z1 - 1000)),
            end=Point(x=I(x1 + 300), y=I(y1 + 300), z=I(z1 + 500)),
            color="#444444",
        )
    )
    anchor2 = Column(name="summit_anchor_2").add(
        Box(
            name="pylon_2",
            start=Point(x=I(x2 - 300), y=I(y2 - 300), z=I(z2 - 1000)),
            end=Point(x=I(x2 + 300), y=I(y2 + 300), z=I(z2 + 500)),
            color="#444444",
        )
    )

    wire_element = Element(
        ifc_class="IfcMember",
        predefined_type="TENSION_MEMBER",
        name="catenary_wire",
    ).add(wire_geom)

    site.add(terrain_element)
    site.add(anchor1, anchor2, wire_element, carve="none")
    proj.add(site)

    # ── TOWER ON PEAK 2 ──────────────────────────────────────────────────
    z_base = I(z2)
    tower_storey = Storey(name="ground", elevation=z_base)

    # 1. Stepped Foundation Plinth
    plinth_profile = [
        Point2D(x=0, y=-1500),
        Point2D(x=3300, y=-1500),
        Point2D(x=3300, y=0),
        Point2D(x=2900, y=500),
        Point2D(x=0, y=500),
    ]
    plinth_geom = Revolve(
        name="plinth_geom",
        profile=plinth_profile,
        path=[Point(x=I(x2), y=I(y2), z=z_base), Point(x=I(x2), y=I(y2), z=z_base + 500)],
        material=Material(key="Concrete_C30-37"),
        color="#686868",
    )
    foundation = Element(
        ifc_class="IfcFooting",
        predefined_type="PAD_FOOTING",
        name="tower_foundation",
    ).add(plinth_geom)

    # 2. Cylindrical Masonry Tower Shaft
    r_out = 2700
    r_in = 2200
    tower_height = 13000
    shaft_profile = [
        Point2D(x=r_in, y=0),
        Point2D(x=r_out, y=0),
        Point2D(x=r_out, y=tower_height),
        Point2D(x=r_in, y=tower_height),
    ]
    shaft_geom = Revolve(
        name="shaft_geom",
        profile=shaft_profile,
        path=[Point(x=I(x2), y=I(y2), z=z_base + 500), Point(x=I(x2), y=I(y2), z=z_base + 500 + tower_height)],
        material=Material(key="Brick_Red_DK"),
        color="wall",
    )

    tower_wall = Wall(name="tower_masonry_shaft").add(shaft_geom)

    # Arched entrance doorway at base
    door_cut = Box(
        name="doorway_void",
        start=Point(x=I(x2 - 600), y=I(y2 - 3200), z=z_base + 500),
        end=Point(x=I(x2 + 600), y=I(y2 - 1800), z=z_base + 500 + 2400),
    )
    # Viewing slit windows spiraling up the masonry shaft
    slit_1 = Box(
        name="slit_cut_1",
        start=Point(x=I(x2 + 1800), y=I(y2 - 250), z=z_base + 3500),
        end=Point(x=I(x2 + 3200), y=I(y2 + 250), z=z_base + 4700),
    )
    slit_2 = Box(
        name="slit_cut_2",
        start=Point(x=I(x2 - 250), y=I(y2 + 1800), z=z_base + 6500),
        end=Point(x=I(x2 + 250), y=I(y2 + 3200), z=z_base + 7700),
    )
    slit_3 = Box(
        name="slit_cut_3",
        start=Point(x=I(x2 - 3200), y=I(y2 - 250), z=z_base + 9500),
        end=Point(x=I(x2 - 1800), y=I(y2 + 250), z=z_base + 10700),
    )
    tower_wall.void(door_cut).void(slit_1).void(slit_2).void(slit_3)

    # Entrance door frame & timber door leaf
    door_frame_l = Column(name="portal_jamb_l").add(
        Box(
            name="jamb_l",
            start=Point(x=I(x2 - 650), y=I(y2 - 2800), z=z_base + 500),
            end=Point(x=I(x2 - 550), y=I(y2 - 2650), z=z_base + 500 + 2450),
            color="#3a2518",
        )
    )
    door_frame_r = Column(name="portal_jamb_r").add(
        Box(
            name="jamb_r",
            start=Point(x=I(x2 + 550), y=I(y2 - 2800), z=z_base + 500),
            end=Point(x=I(x2 + 650), y=I(y2 - 2650), z=z_base + 500 + 2450),
            color="#3a2518",
        )
    )
    door_lintel = Beam(name="portal_lintel").add(
        Box(
            name="lintel",
            start=Point(x=I(x2 - 700), y=I(y2 - 2850), z=z_base + 500 + 2400),
            end=Point(x=I(x2 + 700), y=I(y2 - 2600), z=z_base + 500 + 2600),
            color="#4a3b32",
        )
    )

    # 3. Internal Spiral Staircase & Central Core
    newel_column = Column(name="spiral_newel_core").add(
        Revolve(
            name="newel_core_geom",
            profile=[Point2D(x=0, y=0), Point2D(x=250, y=0), Point2D(x=250, y=tower_height), Point2D(x=0, y=tower_height)],
            path=[Point(x=I(x2), y=I(y2), z=z_base + 500), Point(x=I(x2), y=I(y2), z=z_base + 500 + tower_height)],
            material=Material(key="Timber_C24"),
            color="#5c3a21",
        )
    )

    spiral_stair = Element(ifc_class="IfcStair", name="internal_spiral_staircase")
    num_steps = 72
    total_rise = tower_height - 500
    step_rise = total_rise // num_steps
    total_turns = 3.5
    d_theta = (total_turns * 2.0 * math.pi) / num_steps
    step_r_in = 240
    step_r_out = 2150

    rail_path_points = []
    for k in range(num_steps):
        t0 = -0.5 * math.pi + k * d_theta
        t1 = t0 + (1.15 * d_theta)
        z_step = z_base + 500 + k * step_rise

        p0 = Point(x=I(x2 + step_r_in * math.cos(t0)), y=I(y2 + step_r_in * math.sin(t0)), z=z_step)
        p1 = Point(x=I(x2 + step_r_out * math.cos(t0)), y=I(y2 + step_r_out * math.sin(t0)), z=z_step)
        p2 = Point(x=I(x2 + step_r_out * math.cos(t1)), y=I(y2 + step_r_out * math.sin(t1)), z=z_step)
        p3 = Point(x=I(x2 + step_r_in * math.cos(t1)), y=I(y2 + step_r_in * math.sin(t1)), z=z_step)

        step_solid = Extrude(
            name=f"step_tread_{k:02d}",
            contour=[p0, p1, p2, p3],
            thickness=step_rise,
            material=Material(key="Timber_C24"),
            color="#8b7355",
        )
        spiral_stair.add(step_solid)

        rail_x = I(x2 + 2050 * math.cos(t0))
        rail_y = I(y2 + 2050 * math.sin(t0))
        rail_z = z_step + 900
        rail_path_points.append(Point(x=rail_x, y=rail_y, z=rail_z))

    spiral_handrail = Element(ifc_class="IfcRailing", name="spiral_stair_handrail").add(
        Pipe(
            name="handrail_pipe",
            path=rail_path_points,
            radius=25,
            color="#222222",
        )
    )

    # 4. Observation Viewing Platform
    z_top = z_base + 500 + tower_height

    corbel_ring = Element(ifc_class="IfcBuildingElementProxy", name="platform_corbel_cornice").add(
        Revolve(
            name="corbel_geom",
            profile=[
                Point2D(x=2700, y=-600),
                Point2D(x=3300, y=0),
                Point2D(x=2700, y=0),
            ],
            path=[Point(x=I(x2), y=I(y2), z=z_top), Point(x=I(x2), y=I(y2), z=z_top + 10)],
            material=Material(key="Concrete_C30-37"),
            color="#5a524a",
        )
    )

    deck_slab = Slab(name="observation_deck_slab")
    deck_geom = Revolve(
        name="deck_slab_geom",
        profile=[
            Point2D(x=0, y=0),
            Point2D(x=3300, y=0),
            Point2D(x=3300, y=300),
            Point2D(x=0, y=300),
        ],
        path=[Point(x=I(x2), y=I(y2), z=z_top), Point(x=I(x2), y=I(y2), z=z_top + 300)],
        material=Material(key="Concrete_C30-37"),
        color="#7a7065",
    )
    stair_hatch_void = Box(
        name="stair_hatch_void",
        start=Point(x=I(x2 - 1200), y=I(y2 - 2200), z=z_top - 50),
        end=Point(x=I(x2 + 1200), y=I(y2 - 400), z=z_top + 350),
    )
    deck_slab.add(deck_geom).void(stair_hatch_void)

    platform_parapet = Wall(name="platform_parapet")
    parapet_geom = Revolve(
        name="parapet_ring_geom",
        profile=[
            Point2D(x=3000, y=0),
            Point2D(x=3300, y=0),
            Point2D(x=3300, y=1100),
            Point2D(x=3000, y=1100),
        ],
        path=[Point(x=I(x2), y=I(y2), z=z_top + 300), Point(x=I(x2), y=I(y2), z=z_top + 1400)],
        material=Material(key="Brick_Red_DK"),
        color="wall",
    )
    platform_parapet.add(parapet_geom)

    for i in range(8):
        ang = i * (2.0 * math.pi / 8.0)
        cx = I(x2 + 3150 * math.cos(ang))
        cy = I(y2 + 3150 * math.sin(ang))
        crenel_void = Box(
            name=f"crenel_cut_{i}",
            start=Point(x=cx - 300, y=cy - 300, z=z_top + 300 + 600),
            end=Point(x=cx + 300, y=cy + 300, z=z_top + 300 + 1150),
        )
        platform_parapet.void(crenel_void)

    parapet_rail_points = []
    num_rail_segments = 32
    for j in range(num_rail_segments + 1):
        ang = j * (2.0 * math.pi / num_rail_segments)
        rx = I(x2 + 3150 * math.cos(ang))
        ry = I(y2 + 3150 * math.sin(ang))
        parapet_rail_points.append(Point(x=rx, y=ry, z=z_top + 1450))

    perimeter_railing = Element(ifc_class="IfcRailing", name="platform_perimeter_railing").add(
        Pipe(
            name="top_guardrail_pipe",
            path=parapet_rail_points,
            radius=20,
            color="#1a1a1a",
        )
    )

    hatch_rail = Element(ifc_class="IfcRailing", name="stair_hatch_safety_railing").add(
        Pipe(
            name="hatch_guard_n",
            path=[
                Point(x=I(x2 - 1200), y=I(y2 - 400), z=z_top + 300 + 1000),
                Point(x=I(x2 + 1200), y=I(y2 - 400), z=z_top + 300 + 1000),
            ],
            radius=20,
            color="#1a1a1a",
        ),
        Pipe(
            name="hatch_guard_e",
            path=[
                Point(x=I(x2 + 1200), y=I(y2 - 400), z=z_top + 300 + 1000),
                Point(x=I(x2 + 1200), y=I(y2 - 2200), z=z_top + 300 + 1000),
            ],
            radius=20,
            color="#1a1a1a",
        ),
    )

    viewing_pedestal = Column(name="viewing_pedestal").add(
        Revolve(
            name="pedestal_geom",
            profile=[
                Point2D(x=0, y=0),
                Point2D(x=350, y=0),
                Point2D(x=350, y=900),
                Point2D(x=450, y=1000),
                Point2D(x=0, y=1000),
            ],
            path=[Point(x=I(x2), y=I(y2), z=z_top + 300), Point(x=I(x2), y=I(y2), z=z_top + 1300)],
            material=Material(key="Concrete_C30-37"),
            color="#3d3731",
        )
    )

    tower_storey.add(
        foundation,
        tower_wall,
        door_frame_l,
        door_frame_r,
        door_lintel,
        newel_column,
        spiral_stair,
        spiral_handrail,
        corbel_ring,
        deck_slab,
        platform_parapet,
        perimeter_railing,
        hatch_rail,
        viewing_pedestal,
        carve="none",
    )
    proj.add_storey(tower_storey)
    return proj


result = generate_project()

if __name__ == "__main__":
    compile_main()
