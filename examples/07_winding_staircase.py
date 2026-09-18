"""Winding Metal Staircase & Observation Deck Example.

Demonstrates:
- 3-storey helical stainless steel spiral staircase with 60 risers across 3 levels.
- Central structural mast column with base plate, gusset stiffeners, and pad footing.
- Continuous helical outer stringer, spiral handrails, midrails, and vertical balusters.
- Top observation deck with circular opening void and transitional stepping ramp.
- Structural radial cantilever beams and 90-degree orthogonal cradle frames
  (columns and beams) supporting both the deck above and spiral stair treads below.

Compile with::

    lite-step examples/07_winding_staircase.py
    # or:
    python examples/07_winding_staircase.py
"""
import math
from pathlib import Path
import sys

# Ensure repo root is in sys.path when run directly
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
    Pipe,
    Point,
    Project,
    Slab,
    Storey,
)

I = lambda x: int(round(x))  # noqa: E731,E741


def generate_project():
    proj = Project(name="Winding Metal Staircase")

    # Storeys
    ground = Storey(name="ground", elevation=0)
    level1 = Storey(name="level_1", elevation=3200)
    level2 = Storey(name="level_2", elevation=6400)
    level3 = Storey(name="level_3", elevation=9600)

    # 1. Foundation & Base
    footing = Element(ifc_class="IfcFooting", predefined_type="PAD_FOOTING", name="foundation_pad")
    footing.add(
        Box(
            name="pad_body",
            start=Point(x=-1600, y=-1600, z=-500),
            end=Point(x=1600, y=1600, z=0),
            material="Concrete_C30-37",
            color="#A0AEC0",
        )
    )
    ground.add(footing)

    # Base plate & stiffeners
    base_plate = Element(ifc_class="IfcPlate", predefined_type="BASE_PLATE", name="mast_base_plate")
    base_plate.add(
        Box(
            name="flange_plate",
            start=Point(x=-350, y=-350, z=0),
            end=Point(x=350, y=350, z=30),
            material="Steel_Stainless_304",
            color="#4A5568",
        )
    )
    # Stiffener gussets
    gussets = [
        Box(name="gusset_n", start=Point(x=-15, y=150, z=30), end=Point(x=15, y=320, z=250), material="Steel_Stainless_304", color="#4A5568"),
        Box(name="gusset_s", start=Point(x=-15, y=-320, z=30), end=Point(x=15, y=-150, z=250), material="Steel_Stainless_304", color="#4A5568"),
        Box(name="gusset_e", start=Point(x=150, y=-15, z=30), end=Point(x=320, y=15, z=250), material="Steel_Stainless_304", color="#4A5568"),
        Box(name="gusset_w", start=Point(x=-320, y=-15, z=30), end=Point(x=-150, y=15, z=250), material="Steel_Stainless_304", color="#4A5568"),
    ]
    base_plate.add(*gussets)
    ground.add(base_plate)

    # Central structural column mast (Ø300 mm, height 10,700 mm)
    mast = Column(name="central_mast")
    mast.add(
        Pipe(
            name="mast_tube",
            path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=10700)],
            radius=150,
            material="Steel_Stainless_304",
            color="#718096",
        )
    )
    ground.add(mast)

    # Helper for step polygon
    def make_tread_contour(step_idx: int, z_base: int):
        theta_start = step_idx * (2 * math.pi / 16)  # 22.5 deg per step
        theta_end = theta_start + math.radians(26.0)  # with nosing overlap
        r_in = 140
        r_out = 1200
        pts = []
        # Inner arc
        n_in = 3
        for i in range(n_in + 1):
            t = theta_start + (theta_end - theta_start) * (i / n_in)
            pts.append(Point(x=I(r_in * math.cos(t)), y=I(r_in * math.sin(t)), z=z_base))
        # Outer arc
        n_out = 6
        for i in range(n_out + 1):
            t = theta_end - (theta_end - theta_start) * (i / n_out)
            pts.append(Point(x=I(r_out * math.cos(t)), y=I(r_out * math.sin(t)), z=z_base))
        return pts

    # 2. Helical Treads and Cantilever Brackets (60 risers across 3 storeys)
    flight_1 = Element(ifc_class="IfcStairFlight", name="spiral_flight_ground_to_l1")
    flight_2 = Element(ifc_class="IfcStairFlight", name="spiral_flight_l1_to_l2")
    flight_3 = Element(ifc_class="IfcStairFlight", name="spiral_flight_l2_to_l3")

    flights = [flight_1, flight_2, flight_3]

    for k in range(60):
        flight_idx = k // 20
        z_top = (k + 1) * 160
        z_tread = z_top - 40
        contour = make_tread_contour(k, z_tread)
        tread = Extrude(
            name=f"tread_{k:02d}",
            contour=contour,
            thickness=40,
            material="Steel_Stainless_304",
            color="#CBD5E0",
        )
        flights[flight_idx].add(tread)

        # Cantilever support bracket under the tread
        mid_theta = k * (2 * math.pi / 16) + math.radians(13.0)
        p_in = Point(x=I(150 * math.cos(mid_theta)), y=I(150 * math.sin(mid_theta)), z=z_tread - 70)
        p_out = Point(x=I(1150 * math.cos(mid_theta)), y=I(1150 * math.sin(mid_theta)), z=z_tread - 10)
        bracket = Pipe(
            name=f"bracket_{k:02d}",
            path=[p_in, p_out],
            radius=20,
            material="Steel_Stainless_304",
            color="#4A5568",
        )
        flights[flight_idx].add(bracket)

    ground.add(flight_1, carve="none")
    level1.add(flight_2, carve="none")
    level2.add(flight_3, carve="none")

    # 3. Outer Helical Stringer, Spiral Handrails, and Balusters
    def generate_helix_points(start_k: float, end_k: float, radius: int, z_offset: int, num_samples: int):
        pts = []
        for i in range(num_samples + 1):
            t = start_k + (end_k - start_k) * (i / num_samples)
            theta = t * (2 * math.pi / 16)
            z = I(t * 160 + z_offset)
            x = I(radius * math.cos(theta))
            y = I(radius * math.sin(theta))
            pts.append(Point(x=x, y=y, z=z))
        return pts

    railing_1 = Element(ifc_class="IfcRailing", predefined_type="HANDRAIL", name="spiral_railing_ground_to_l1")
    railing_2 = Element(ifc_class="IfcRailing", predefined_type="HANDRAIL", name="spiral_railing_l1_to_l2")
    railing_3 = Element(ifc_class="IfcRailing", predefined_type="HANDRAIL", name="spiral_railing_l2_to_l3")
    railings = [railing_1, railing_2, railing_3]

    for flight_idx in range(3):
        k_start = flight_idx * 20
        k_end = (flight_idx + 1) * 20

        # Outer helical stringer pipe (along outer edge of treads)
        stringer_pts = generate_helix_points(k_start, k_end, 1205, -20, 40)
        stringer = Pipe(
            name=f"outer_stringer_f{flight_idx+1}",
            path=stringer_pts,
            radius=25,
            material="Steel_Stainless_304",
            color="#4A5568",
        )
        railings[flight_idx].add(stringer)

        # Outer spiral handrail (+900 mm)
        handrail_pts = generate_helix_points(k_start, k_end, 1200, 900, 40)
        handrail = Pipe(
            name=f"outer_handrail_f{flight_idx+1}",
            path=handrail_pts,
            radius=22,
            material="Steel_Stainless_304",
            color="#E2E8F0",
        )
        railings[flight_idx].add(handrail)

        # Mid-rail safety guard (+450 mm)
        midrail_pts = generate_helix_points(k_start, k_end, 1200, 450, 40)
        midrail = Pipe(
            name=f"outer_midrail_f{flight_idx+1}",
            path=midrail_pts,
            radius=12,
            material="Steel_Stainless_304",
            color="#A0AEC0",
        )
        railings[flight_idx].add(midrail)

        # Inner spiral handrail (+900 mm at r=240 mm around column)
        inner_handrail_pts = generate_helix_points(k_start, k_end, 240, 900, 30)
        inner_handrail = Pipe(
            name=f"inner_handrail_f{flight_idx+1}",
            path=inner_handrail_pts,
            radius=16,
            material="Steel_Stainless_304",
            color="#E2E8F0",
        )
        railings[flight_idx].add(inner_handrail)

        # Vertical balusters for each step in this flight
        for k in range(k_start, k_end):
            theta_mid = k * (2 * math.pi / 16) + math.radians(13.0)
            z_tread = (k + 1) * 160
            bx = I(1185 * math.cos(theta_mid))
            by = I(1185 * math.sin(theta_mid))
            baluster = Pipe(
                name=f"baluster_{k:02d}",
                path=[Point(x=bx, y=by, z=z_tread - 40), Point(x=bx, y=by, z=z_tread + 900)],
                radius=10,
                material="Steel_Stainless_304",
                color="#718096",
            )
            railings[flight_idx].add(baluster)

    ground.add(railing_1, carve="none")
    level1.add(railing_2, carve="none")
    level2.add(railing_3, carve="none")

    # 4. Top Observation Deck

    # Level 3 Top Observation Deck (Z = 9600)
    deck = Slab(name="observation_deck")
    # Observation deck platform plate
    deck.add(
        Box(
            name="deck_plate",
            start=Point(x=-2000, y=-2200, z=9520),
            end=Point(x=2000, y=2000, z=9600),
            material="Steel_Stainless_304",
            color="#CBD5E0",
        )
    )

    # Stair opening void tool through the top deck
    stair_void = Pipe(
        name="stair_opening_void",
        path=[Point(x=0, y=0, z=9450), Point(x=0, y=0, z=9650)],
        radius=1300,
    )
    deck.void(stair_void)

    # Stepping ramp in front of top stair (tread_59) joined to observation deck
    theta_ramp_start = 59 * (2 * math.pi / 16) + math.radians(22.5)  # 270 deg (aligned with tread_59)
    theta_ramp_end = theta_ramp_start + math.radians(45.0)  # 315 deg (transitions into deck)
    r_in = 150
    r_out = 1400  # overlaps cleanly into deck plate at r=1300 void boundary
    ramp_pts = []
    # Inner arc
    for i in range(4):
        t = theta_ramp_start + (theta_ramp_end - theta_ramp_start) * (i / 3)
        ramp_pts.append(Point(x=I(r_in * math.cos(t)), y=I(r_in * math.sin(t)), z=9520))
    # Outer arc
    for i in range(6):
        t = theta_ramp_end - (theta_ramp_end - theta_ramp_start) * (i / 5)
        ramp_pts.append(Point(x=I(r_out * math.cos(t)), y=I(r_out * math.sin(t)), z=9520))

    stepping_ramp = Element(ifc_class="IfcRampFlight", name="stepping_ramp")
    stepping_ramp.add(
        Extrude(
            name="ramp_body",
            contour=ramp_pts,
            thickness=80,
            material="Steel_Stainless_304",
            color="#CBD5E0",
        )
    )

    # Structural support radial cantilever beams underneath observation deck
    deck_beams = [
        # South radial deck beam reaching directly under stepping ramp to central mast (r=150)
        Beam(name="deck_beam_s").add(
            Box(
                name="body",
                start=Point(x=-75, y=-2150, z=9370),
                end=Point(x=75, y=-150, z=9520),
                material="Steel_Stainless_304",
                color="#4A5568",
            )
        ),
        # North radial deck beam with 90-deg column drop under innermost part of deck support (y=1300..1450) reaching under tread_51
        Beam(name="deck_beam_n").add(
            Box(
                name="body",
                start=Point(x=-75, y=1300, z=9370),
                end=Point(x=75, y=1950, z=9520),
                material="Steel_Stainless_304",
                color="#4A5568",
            )
        ),
        Column(name="deck_column_n").add(
            Box(
                name="body",
                start=Point(x=-75, y=1300, z=8130),
                end=Point(x=75, y=1450, z=9370),
                material="Steel_Stainless_304",
                color="#4A5568",
            )
        ),
        Beam(name="stair_support_beam_n").add(
            Box(
                name="body",
                start=Point(x=-75, y=150, z=8130),
                end=Point(x=75, y=1450, z=8280),
                material="Steel_Stainless_304",
                color="#4A5568",
            )
        ),
        # East radial deck beam with 90-deg column drop under innermost part of deck support (x=1300..1450) reaching under tread_47/48
        Beam(name="deck_beam_e").add(
            Box(
                name="body",
                start=Point(x=1300, y=-75, z=9370),
                end=Point(x=1950, y=75, z=9520),
                material="Steel_Stainless_304",
                color="#4A5568",
            )
        ),
        Column(name="deck_column_e").add(
            Box(
                name="body",
                start=Point(x=1300, y=-75, z=7490),
                end=Point(x=1450, y=75, z=9370),
                material="Steel_Stainless_304",
                color="#4A5568",
            )
        ),
        Beam(name="stair_support_beam_e").add(
            Box(
                name="body",
                start=Point(x=150, y=-75, z=7490),
                end=Point(x=1450, y=75, z=7640),
                material="Steel_Stainless_304",
                color="#4A5568",
            )
        ),
        # West radial deck beam with 90-deg column drop under innermost part of deck support (x=-1450..-1300) reaching under tread_55/56
        Beam(name="deck_beam_w").add(
            Box(
                name="body",
                start=Point(x=-1950, y=-75, z=9370),
                end=Point(x=-1300, y=75, z=9520),
                material="Steel_Stainless_304",
                color="#4A5568",
            )
        ),
        Column(name="deck_column_w").add(
            Box(
                name="body",
                start=Point(x=-1450, y=-75, z=8770),
                end=Point(x=-1300, y=75, z=9370),
                material="Steel_Stainless_304",
                color="#4A5568",
            )
        ),
        Beam(name="stair_support_beam_w").add(
            Box(
                name="body",
                start=Point(x=-1450, y=-75, z=8770),
                end=Point(x=-150, y=75, z=8920),
                material="Steel_Stainless_304",
                color="#4A5568",
            )
        ),
        Beam(name="deck_beam_se").add(Pipe(name="body", path=[Point(x=I(150*math.cos(math.radians(292.5))), y=I(150*math.sin(math.radians(292.5))), z=9475), Point(x=I(1350*math.cos(math.radians(292.5))), y=I(1350*math.sin(math.radians(292.5))), z=9475)], radius=40, material="Steel_Stainless_304", color="#4A5568")),
        # Perimeter ring beams
        Beam(name="deck_rim_n").add(Box(name="body", start=Point(x=-1950, y=1900, z=9370), end=Point(x=1950, y=2000, z=9520), material="Steel_Stainless_304", color="#4A5568")),
        Beam(name="deck_rim_s").add(Box(name="body", start=Point(x=-1950, y=-2200, z=9370), end=Point(x=1950, y=-2100, z=9520), material="Steel_Stainless_304", color="#4A5568")),
        Beam(name="deck_rim_e").add(Box(name="body", start=Point(x=1900, y=-2200, z=9370), end=Point(x=2000, y=2000, z=9520), material="Steel_Stainless_304", color="#4A5568")),
        Beam(name="deck_rim_w").add(Box(name="body", start=Point(x=-2000, y=-2200, z=9370), end=Point(x=-1900, y=2000, z=9520), material="Steel_Stainless_304", color="#4A5568")),
    ]

    # Observation Deck Guardrails (+1000 mm above deck)
    deck_guard = Element(ifc_class="IfcRailing", predefined_type="GUARDRAIL", name="deck_perimeter_guardrail")
    deck_guard.add(
        # Top perimeter handrail
        Pipe(name="deck_top_rail_n", path=[Point(x=-1950, y=1950, z=10600), Point(x=1950, y=1950, z=10600)], radius=22, material="Steel_Stainless_304", color="#E2E8F0"),
        Pipe(name="deck_top_rail_e", path=[Point(x=1950, y=1950, z=10600), Point(x=1950, y=-2150, z=10600)], radius=22, material="Steel_Stainless_304", color="#E2E8F0"),
        Pipe(name="deck_top_rail_s", path=[Point(x=1950, y=-2150, z=10600), Point(x=-1950, y=-2150, z=10600)], radius=22, material="Steel_Stainless_304", color="#E2E8F0"),
        Pipe(name="deck_top_rail_w", path=[Point(x=-1950, y=-2150, z=10600), Point(x=-1950, y=1950, z=10600)], radius=22, material="Steel_Stainless_304", color="#E2E8F0"),
        # Mid rails
        Pipe(name="deck_mid_rail_n", path=[Point(x=-1950, y=1950, z=10100), Point(x=1950, y=1950, z=10100)], radius=14, material="Steel_Stainless_304", color="#A0AEC0"),
        Pipe(name="deck_mid_rail_e", path=[Point(x=1950, y=1950, z=10100), Point(x=1950, y=-2150, z=10100)], radius=14, material="Steel_Stainless_304", color="#A0AEC0"),
        Pipe(name="deck_mid_rail_s", path=[Point(x=1950, y=-2150, z=10100), Point(x=-1950, y=-2150, z=10100)], radius=14, material="Steel_Stainless_304", color="#A0AEC0"),
        Pipe(name="deck_mid_rail_w", path=[Point(x=-1950, y=-2150, z=10100), Point(x=-1950, y=1950, z=10100)], radius=14, material="Steel_Stainless_304", color="#A0AEC0"),
        # Corner & perimeter posts
        Pipe(name="deck_post_ne", path=[Point(x=1950, y=1950, z=9600), Point(x=1950, y=1950, z=10600)], radius=20, material="Steel_Stainless_304", color="#718096"),
        Pipe(name="deck_post_nw", path=[Point(x=-1950, y=1950, z=9600), Point(x=-1950, y=1950, z=10600)], radius=20, material="Steel_Stainless_304", color="#718096"),
        Pipe(name="deck_post_se", path=[Point(x=1950, y=-2150, z=9600), Point(x=1950, y=-2150, z=10600)], radius=20, material="Steel_Stainless_304", color="#718096"),
        Pipe(name="deck_post_sw", path=[Point(x=-1950, y=-2150, z=9600), Point(x=-1950, y=-2150, z=10600)], radius=20, material="Steel_Stainless_304", color="#718096"),
        Pipe(name="deck_post_n_mid", path=[Point(x=0, y=1950, z=9600), Point(x=0, y=1950, z=10600)], radius=16, material="Steel_Stainless_304", color="#718096"),
        Pipe(name="deck_post_s_mid", path=[Point(x=0, y=-2150, z=9600), Point(x=0, y=-2150, z=10600)], radius=16, material="Steel_Stainless_304", color="#718096"),
        Pipe(name="deck_post_e_mid", path=[Point(x=1950, y=-100, z=9600), Point(x=1950, y=-100, z=10600)], radius=16, material="Steel_Stainless_304", color="#718096"),
        Pipe(name="deck_post_w_mid", path=[Point(x=-1950, y=-100, z=9600), Point(x=-1950, y=-100, z=10600)], radius=16, material="Steel_Stainless_304", color="#718096"),
    )
    level3.add(deck, *deck_beams, deck_guard, stepping_ramp, carve="none")

    proj.add_storey(ground)
    proj.add_storey(level1)
    proj.add_storey(level2)
    proj.add_storey(level3)

    return proj


result = generate_project()


if __name__ == "__main__":
    compile_main()
