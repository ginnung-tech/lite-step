from pathlib import Path
import sys
import time
import math
import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from lite_step import compile_main
from lite_step.models import Door, Element, Material, Mesh, Point, Point2D, Product, Project, Site, Sweep, Wall, Window, miter
from lite_step.optimization import jit_compile, solve_sequence

I = lambda x: int(round(x))  # noqa: E731

RANDOM_SEED = int(time.time())

CABIN_LENGTH = 6000.0
CABIN_WIDTH = 4000.0
CORNER_OVERHANG = 250.0

LOG_LEN_LONG = CABIN_LENGTH + 2.0 * CORNER_OVERHANG
LOG_LEN_SHORT = CABIN_WIDTH + 2.0 * CORNER_OVERHANG

NUM_LOGS_PER_WALL = 7
NUM_X_STATIONS = 61
NUM_THETA_SEGS = 36

LOG_R_BUTT_MIN, LOG_R_BUTT_MAX = 170.0, 195.0
LOG_R_TIP_MIN, LOG_R_TIP_MAX = 155.0, 175.0
AVG_LOG_DIAMETER = 350.0
WALL_Z_RAISE = AVG_LOG_DIAMETER / 2.0

SLAB_EXTEND = 150.0
SLAB_HEIGHT = 300.0
SLAB_X_MIN = -(CABIN_LENGTH / 2.0 + SLAB_EXTEND)
SLAB_X_MAX = (CABIN_LENGTH / 2.0 + SLAB_EXTEND)
SLAB_Y_MIN = -(CABIN_WIDTH / 2.0 + SLAB_EXTEND)
SLAB_Y_MAX = (CABIN_WIDTH / 2.0 + SLAB_EXTEND)

MOSS_WIDTH = 50.0
MOSS_COLOR = "#4A7C3B"
SLAB_COLOR = "#8D99AE"

LOG_PALETTES = [
    ["#825833", "#94673D", "#6E4725", "#9C7146", "#784E2A", "#8C6038", "#704828"],
    ["#7D532F", "#8E6238", "#694321", "#966C41", "#734A26", "#875C34", "#6B4424"],
    ["#865C37", "#986B41", "#724B29", "#9F7449", "#7C522E", "#90643C", "#744C2C"],
    ["#805632", "#91653B", "#6C4523", "#996F44", "#764D28", "#8A5E36", "#6E4626"],
]

SITE_HEIGHTMAP = [
    [-3000, -2600, -2300, -2150, -2100, -2150, -2300, -2600, -3000],
    [-2400, -1900, -1200, -1050, -1000, -1050, -1200, -1900, -2400],
    [-1800, -1100, 0, 0, 0, 0, 0, -1100, -1800],
    [-1600, -900, 0, 0, 0, 0, 0, -900, -1600],
    [-1500, -850, 0, 0, 0, 0, 0, -850, -1500],
    [-1600, -900, 0, 0, 0, 0, 0, -900, -1600],
    [-1800, -1100, 0, 0, 0, 0, 0, -1100, -1800],
    [-2400, -1900, -1200, -1050, -1000, -1050, -1200, -1900, -2400],
    [-3000, -2600, -2300, -2150, -2100, -2150, -2300, -2600, -3000],
]

CLEARING = Mesh(
    name="clearing",
    heightmap=SITE_HEIGHTMAP,
    depth=4000,
    corner_min=Point2D(x=-12000, y=-10000),
    corner_max=Point2D(x=12000, y=10000),
    source_type="synthetic",
    color="#5b6b45",
)


def make_log(log_id: int, length: float, rng: np.random.RandomState) -> dict:
    u = rng.uniform
    pi2 = 2 * math.pi
    log = {
        "log_id": log_id, "length": length,
        "r_butt": u(LOG_R_BUTT_MIN, LOG_R_BUTT_MAX), "r_tip": u(LOG_R_TIP_MIN, LOG_R_TIP_MAX),
        "twist_rate": u(1.2, 2.5) * rng.choice([-1.0, 1.0]),
        "camber_amp_y1": u(22, 42), "camber_amp_y2": u(8, 18), "camber_amp_y3": u(3, 8),
        "camber_phase_y1": u(0, pi2), "camber_phase_y2": u(0, pi2), "camber_phase_y3": u(0, pi2),
        "camber_amp_z1": u(20, 40), "camber_amp_z2": u(8, 16), "camber_amp_z3": u(3, 8),
        "camber_phase_z1": u(0, pi2), "camber_phase_z2": u(0, pi2), "camber_phase_z3": u(0, pi2),
        "ovality": u(0.06, 0.11), "oval_angle": u(0, math.pi),
        "bark_amp3": u(0.018, 0.040), "bark_phase3": u(0, pi2),
        "bark_amp5": u(0.008, 0.020), "bark_phase5": u(0, pi2),
    }
    log["knobs"] = [
        {"u": u(0.18, 0.82), "angle": u(0, pi2), "height": u(45, 75), "sigma_u": u(0.04, 0.09), "sigma_th": u(0.35, 0.65)}
        for _ in range(rng.randint(1, 4))
    ]
    return log


def log_camber(log: dict, u: float) -> tuple[float, float]:
    dy = sum(log[f"camber_amp_y{i}"] * math.sin(i * math.pi * u + log[f"camber_phase_y{i}"]) for i in (1, 2, 3))
    dz = sum(log[f"camber_amp_z{i}"] * math.sin(i * math.pi * u + log[f"camber_phase_z{i}"]) for i in (1, 2, 3))
    return dy, dz


def log_raw_radius(log: dict, u: float, theta: float) -> float:
    r_nominal = log["r_butt"] * (1.0 - u) + log["r_tip"] * u
    theta_grain = theta + log["twist_rate"] * u
    ovality_mod = 1.0 + log["ovality"] * math.cos(2 * (theta_grain - log["oval_angle"]))
    bark_mod = (
        1.0
        + log["bark_amp3"] * math.cos(3 * theta_grain + log["bark_phase3"])
        + log["bark_amp5"] * math.cos(5 * theta_grain + log["bark_phase5"])
    )
    r = r_nominal * ovality_mod * bark_mod
    for knob in log["knobs"]:
        du = (u - knob["u"]) / knob["sigma_u"]
        dth = (theta - knob["angle"] + math.pi) % (2 * math.pi) - math.pi
        dth_norm = dth / knob["sigma_th"]
        r += knob["height"] * math.exp(-0.5 * (du * du + dth_norm * dth_norm))
    return max(r, 80.0)


def log_profiles(log: dict, flip: bool, rot_angle: float):
    n = int(NUM_X_STATIONS)
    x_vals = np.linspace(-log["length"] / 2.0, log["length"] / 2.0, n)
    z_top = np.zeros(n, dtype=np.float64)
    z_bot = np.zeros(n, dtype=np.float64)

    knobs = log["knobs"]
    _jit_eval_profile(
        float(log["r_butt"]), float(log["r_tip"]), float(log["twist_rate"]),
        float(log["camber_amp_y1"]), float(log["camber_amp_y2"]), float(log["camber_amp_y3"]),
        float(log["camber_phase_y1"]), float(log["camber_phase_y2"]), float(log["camber_phase_y3"]),
        float(log["camber_amp_z1"]), float(log["camber_amp_z2"]), float(log["camber_amp_z3"]),
        float(log["camber_phase_z1"]), float(log["camber_phase_z2"]), float(log["camber_phase_z3"]),
        float(log["ovality"]), float(log["oval_angle"]),
        float(log["bark_amp3"]), float(log["bark_phase3"]),
        float(log["bark_amp5"]), float(log["bark_phase5"]),
        np.array([k["u"] for k in knobs], dtype=np.float64),
        np.array([k["angle"] for k in knobs], dtype=np.float64),
        np.array([k["height"] for k in knobs], dtype=np.float64),
        np.array([k["sigma_u"] for k in knobs], dtype=np.float64),
        np.array([k["sigma_th"] for k in knobs], dtype=np.float64),
        x_vals, z_top, z_bot, flip, rot_angle,
    )
    return x_vals, z_top, z_bot


@jit_compile(fastmath=True, nogil=True)
def _jit_eval_profile(
    r_butt, r_tip, twist_rate,
    c_y1, c_y2, c_y3, p_y1, p_y2, p_y3,
    c_z1, c_z2, c_z3, p_z1, p_z2, p_z3,
    ovality, oval_angle,
    b_a3, b_p3, b_a5, b_p5,
    knobs_u, knobs_ang, knobs_h, knobs_su, knobs_sth,
    x_vals, z_top, z_bot,
    flip, rot_angle,
):
    num_stations = len(x_vals)
    length = x_vals[num_stations - 1] - x_vals[0]
    cos_rot, sin_rot = np.cos(rot_angle), np.sin(rot_angle)
    d_th = 2.0 * np.pi / 120.0

    for i in range(num_stations):
        x = x_vals[i]
        u = (x + length / 2.0) / length
        u_log = (1.0 - u) if flip else u

        dy = (c_y1 * np.sin(np.pi * u_log + p_y1)
            + c_y2 * np.sin(2.0 * np.pi * u_log + p_y2)
            + c_y3 * np.sin(3.0 * np.pi * u_log + p_y3))
        dz = (c_z1 * np.sin(np.pi * u_log + p_z1)
            + c_z2 * np.sin(2.0 * np.pi * u_log + p_z2)
            + c_z3 * np.sin(3.0 * np.pi * u_log + p_z3))
        dz_c = dy * sin_rot + dz * cos_rot

        r_nom = r_butt * (1.0 - u_log) + r_tip * u_log

        max_z, min_z = -1e9, 1e9
        for j in range(120):
            th = j * d_th
            theta_grain = (th - rot_angle) + twist_rate * u_log
            ovality_mod = 1.0 + ovality * np.cos(2.0 * (theta_grain - oval_angle))
            bark_mod = 1.0 + b_a3 * np.cos(3.0 * theta_grain + b_p3) + b_a5 * np.cos(5.0 * theta_grain + b_p5)
            r = r_nom * ovality_mod * bark_mod

            for k_idx in range(len(knobs_u)):
                du = (u_log - knobs_u[k_idx]) / knobs_su[k_idx]
                dth = (th - rot_angle - knobs_ang[k_idx] + np.pi) % (2.0 * np.pi) - np.pi
                r += knobs_h[k_idx] * np.exp(-0.5 * (du * du + (dth / knobs_sth[k_idx]) ** 2))

            if r < 80.0:
                r = 80.0
            z_sample = dz_c + r * np.sin(th)
            if z_sample > max_z:
                max_z = z_sample
            if z_sample < min_z:
                min_z = z_sample

        z_top[i] = max_z
        z_bot[i] = min_z


@jit_compile(fastmath=True, nogil=True)
def upper_convex_hull(x_vals: np.ndarray, h_vals: np.ndarray) -> np.ndarray:
    n = len(x_vals)
    hull = np.empty(n, dtype=np.int64)
    hull_len = 0
    for i in range(n):
        x, y = x_vals[i], h_vals[i]
        while hull_len >= 2:
            idx1, idx2 = hull[hull_len - 2], hull[hull_len - 1]
            x1, y1 = x_vals[idx1], h_vals[idx1]
            x2, y2 = x_vals[idx2], h_vals[idx2]
            if (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0:
                hull_len -= 1
            else:
                break
        hull[hull_len] = i
        hull_len += 1
    return hull[:hull_len]


@jit_compile(fastmath=True, nogil=True)
def compute_resting_offset_and_gap(
    lower_top: np.ndarray,
    upper_bot: np.ndarray,
    dx: float,
    x_vals: np.ndarray,
    min_span: float,
) -> tuple[float, float, float, float]:
    n = len(lower_top)
    h = np.empty(n, dtype=np.float64)
    for k in range(n):
        h[k] = lower_top[k] - upper_bot[k]

    hull = upper_convex_hull(x_vals, h)
    m = len(hull)
    best_gap, best_z0, best_s, best_dist = 1e18, 0.0, 0.0, 0.0

    for u1 in range(m):
        i = hull[u1]
        for u2 in range(u1 + 1, m):
            j = hull[u2]
            dist = x_vals[j] - x_vals[i]
            if dist < min_span:
                continue

            s_cand = (h[j] - h[i]) / dist
            z0_cand = h[i] - s_cand * x_vals[i]

            valid = True
            for uk in range(u1 + 1, u2):
                k = hull[uk]
                if z0_cand + s_cand * x_vals[k] < h[k] - 1e-4:
                    valid = False
                    break

            if valid:
                gap_sum = 0.0
                for k in range(n):
                    gap_sum += (z0_cand + s_cand * x_vals[k]) - h[k]
                gap_area = gap_sum * dx
                if gap_area < best_gap:
                    best_gap = gap_area
                    best_z0 = z0_cand
                    best_s = s_cand
                    best_dist = dist

    if best_gap >= 1e17:
        max_idx = 0
        max_h = h[0]
        for k in range(1, n):
            if h[k] > max_h:
                max_h, max_idx = h[k], k
        xi, hi = x_vals[max_idx], max_h
        for uk in range(m):
            k = hull[uk]
            dist = np.abs(x_vals[k] - xi)
            if dist < min_span:
                continue
            s_cand = (h[k] - hi) / (x_vals[k] - xi)
            z0_cand = -1e18
            for p in hull:
                val = h[p] - s_cand * x_vals[p]
                if val > z0_cand:
                    z0_cand = val

            gap_sum = 0.0
            for p in range(n):
                gap_sum += (z0_cand + s_cand * x_vals[p]) - h[p]
            gap_area = gap_sum * dx + 5000.0
            if gap_area < best_gap:
                best_gap, best_z0, best_s, best_dist = gap_area, z0_cand, s_cand, dist

    return best_z0, best_s, best_gap, best_dist


def optimize_log_wall(
    logs: list,
    base_height: float,
    num_rotations: int = 24,
) -> tuple[list[int], list[bool], list[float], list[tuple[float, float]], list[float], float, float]:
    num_logs = len(logs)
    length = logs[0]["length"]
    min_span = 0.30 * length
    x_vals = np.linspace(-length / 2.0, length / 2.0, NUM_X_STATIONS)
    dx = length / (NUM_X_STATIONS - 1)
    rotation_candidates = list(np.linspace(0, 2 * math.pi, num_rotations, endpoint=False))

    orientations = {}
    num_oris = 2 * num_rotations
    for idx, log in enumerate(logs):
        orientations[idx] = []
        for flip in [False, True]:
            for rot in rotation_candidates:
                _, z_top, z_bot = log_profiles(log, flip, rot)
                orientations[idx].append({"z_top": z_top, "z_bot": z_bot, "flip": flip, "rot": rot})

    foundation_top = np.full(NUM_X_STATIONS, float(base_height), dtype=np.float64)
    base_seating = {
        idx: [compute_resting_offset_and_gap(foundation_top, orientations[idx][o]["z_bot"], dx, x_vals, min_span)
              for o in range(num_oris)]
        for idx in range(num_logs)
    }

    pairwise_best = {i: {o: {} for o in range(num_oris)} for i in range(num_logs)}
    for la in range(num_logs):
        for oa in range(num_oris):
            zta = orientations[la][oa]["z_top"]
            for lb in range(num_logs):
                if lb == la:
                    continue
                best_g, best_e = 1e18, None
                for ob in range(num_oris):
                    z0, s, gap, dist = compute_resting_offset_and_gap(zta, orientations[lb][ob]["z_bot"], dx, x_vals, min_span)
                    if gap < best_g:
                        best_g, best_e = gap, (ob, z0, s, gap, dist)
                pairwise_best[la][oa][lb] = best_e

    cost_matrix = np.full((num_logs, num_logs), 1e8)
    for i in range(num_logs):
        for j in range(num_logs):
            if i != j:
                cost_matrix[i, j] = min(pairwise_best[i][o][j][3] for o in range(num_oris))

    base_costs = np.array([min(base_seating[i][o][2] for o in range(num_oris)) for i in range(num_logs)])
    best_perm = solve_sequence(num_logs, cost_matrix, base_costs=base_costs)

    curr_o = min(range(num_oris), key=lambda o: base_seating[best_perm[0]][o][2])
    best_ori_indices = [curr_o]
    best_total_gap = 0.0
    for layer in range(num_logs - 1):
        la, lb = best_perm[layer], best_perm[layer + 1]
        next_o, _, _, gap, _ = pairwise_best[la][curr_o][lb]
        best_total_gap += gap
        curr_o = next_o
        best_ori_indices.append(curr_o)

    best_flips, best_rots, best_placements, best_dists = [], [], [], []
    base_log = best_perm[0]
    z0_world, s_world, _, dist = base_seating[base_log][best_ori_indices[0]]
    best_flips.append(orientations[base_log][best_ori_indices[0]]["flip"])
    best_rots.append(orientations[base_log][best_ori_indices[0]]["rot"])
    best_placements.append((z0_world, s_world))
    best_dists.append(dist)

    for layer in range(num_logs - 1):
        la, lb = best_perm[layer], best_perm[layer + 1]
        oa, ob = best_ori_indices[layer], best_ori_indices[layer + 1]
        _, z0_rel, s_rel, _, dist = pairwise_best[la][oa][lb]
        z0_world += z0_rel
        s_world += s_rel
        best_flips.append(orientations[lb][ob]["flip"])
        best_rots.append(orientations[lb][ob]["rot"])
        best_placements.append((z0_world, s_world))
        best_dists.append(dist)

    return best_perm, best_flips, best_rots, best_placements, best_dists, 0.0, best_total_gap


def build_log_mesh(
    log: dict, flip_x: bool, rotation_rad: float, z0: float, slope_s: float,
    color: str, name: str, axis: str = "X", origin_x: float = 0.0, origin_y: float = 0.0, origin_z: float = 0.0,
) -> Mesh:
    verts: list[Point] = []
    faces: list[tuple[int, int, int]] = []

    x_vals = np.linspace(-log["length"] / 2.0, log["length"] / 2.0, NUM_X_STATIONS)
    theta_vals = np.linspace(0, 2 * math.pi, NUM_THETA_SEGS, endpoint=False)
    cos_rot, sin_rot = math.cos(rotation_rad), math.sin(rotation_rad)

    for i, x in enumerate(x_vals):
        u = (x + log["length"] / 2.0) / log["length"]
        u_log = (1.0 - u) if flip_x else u

        raw_dy, raw_dz = log_camber(log, u_log)
        dy_center = raw_dy * cos_rot - raw_dz * sin_rot
        dz_center = raw_dy * sin_rot + raw_dz * cos_rot
        z_offset_x = z0 + slope_s * x

        for th in theta_vals:
            r = log_raw_radius(log, u_log, th - rotation_rad)
            py = dy_center + r * math.cos(th)
            pz = dz_center + r * math.sin(th) + z_offset_x
            wx = origin_x + (x if axis == "X" else py)
            wy = origin_y + (py if axis == "X" else x)
            verts.append(Point(x=I(wx), y=I(wy), z=I(origin_z + pz)))

    for i in range(NUM_X_STATIONS - 1):
        ring0, ring1 = i * NUM_THETA_SEGS, (i + 1) * NUM_THETA_SEGS
        for t in range(NUM_THETA_SEGS):
            tn = (t + 1) % NUM_THETA_SEGS
            v00, v01, v10, v11 = ring0 + t, ring0 + tn, ring1 + t, ring1 + tn
            if axis == "X":
                faces.extend([(v00, v01, v11), (v00, v11, v10)])
            else:
                faces.extend([(v00, v11, v01), (v00, v10, v11)])

    for sign, u_cap in ((-1.0, 1.0 if flip_x else 0.0), (1.0, 0.0 if flip_x else 1.0)):
        dy_c, dz_c = log_camber(log, u_cap)
        dy_rot = dy_c * cos_rot - dz_c * sin_rot
        dz_rot = dy_c * sin_rot + dz_c * cos_rot
        zc = z0 + slope_s * (sign * log["length"] / 2.0)
        c_idx = len(verts)

        wx = origin_x + (sign * log["length"] / 2.0 if axis == "X" else dy_rot)
        wy = origin_y + (dy_rot if axis == "X" else sign * log["length"] / 2.0)
        verts.append(Point(x=I(wx), y=I(wy), z=I(origin_z + dz_rot + zc)))

        ring_base = 0 if sign < 0 else (NUM_X_STATIONS - 1) * NUM_THETA_SEGS
        for t in range(NUM_THETA_SEGS):
            tn = (t + 1) % NUM_THETA_SEGS
            if (axis == "X" and sign < 0) or (axis != "X" and sign > 0):
                faces.append((c_idx, ring_base + tn, ring_base + t))
            else:
                faces.append((c_idx, ring_base + t, ring_base + tn))

    return Mesh(name=name, vertices=verts, faces=faces, is_watertight=True, color=color)


def raytrace_log_z_surfaces(
    log: dict, flip: bool, rot: float, z0: float, s: float, x: float, y: float
) -> tuple[float, float]:
    u = (x + log["length"] / 2.0) / log["length"]
    u_log = (1.0 - u) if flip else u

    raw_dy, raw_dz = log_camber(log, u_log)
    cos_rot, sin_rot = math.cos(rot), math.sin(rot)
    yc = raw_dy * cos_rot - raw_dz * sin_rot
    zc = raw_dy * sin_rot + raw_dz * cos_rot + z0 + s * x

    theta_vals = np.linspace(0, 2 * math.pi, 180, endpoint=False)
    pts = [
        (yc + log_raw_radius(log, u_log, th - rot) * math.cos(th),
         zc + log_raw_radius(log, u_log, th - rot) * math.sin(th))
        for th in theta_vals
    ]

    z_hits = []
    num_pts = len(pts)
    for i in range(num_pts):
        y1, z1 = pts[i]
        y2, z2 = pts[(i + 1) % num_pts]
        if (y1 <= y <= y2) or (y2 <= y <= y1):
            if abs(y2 - y1) > 1e-6:
                z_hits.append(z1 + (y - y1) / (y2 - y1) * (z2 - z1))
            else:
                z_hits.append(z1)

    if len(z_hits) >= 2:
        return max(z_hits), min(z_hits)
    if len(z_hits) == 1:
        return z_hits[0], z_hits[0]
    cpz = min(pts, key=lambda p: abs(p[0] - y))[1]
    return cpz, cpz


def build_moss_meshes(
    foundation_top: float, logs: list, best_perm: list[int], best_flips: list[bool],
    best_rots: list[float], best_placements: list[tuple[float, float]], prefix: str, axis: str,
    origin_x: float, origin_y: float, origin_z: float, x_min: float, x_max: float,
) -> list[Mesh]:
    x_vals = np.linspace(x_min, x_max, NUM_X_STATIONS)
    y_front, y_back = -MOSS_WIDTH / 2.0, MOSS_WIDTH / 2.0
    EMBED_MM = 3.0

    moss_meshes = []
    for cav_idx in range(len(logs)):
        name = f"moss_{prefix}_foundation_joint" if cav_idx == 0 else f"moss_{prefix}_cavity_tier_{cav_idx}_to_{cav_idx + 1}"

        if cav_idx > 0:
            low_log = logs[best_perm[cav_idx - 1]]
            low_flip = best_flips[cav_idx - 1]
            low_rot = best_rots[cav_idx - 1]
            low_z0, low_s = best_placements[cav_idx - 1]

        high_log = logs[best_perm[cav_idx]]
        high_flip = best_flips[cav_idx]
        high_rot = best_rots[cav_idx]
        high_z0, high_s = best_placements[cav_idx]

        verts: list[Point] = []
        faces: list[tuple[int, int, int]] = []

        for x in x_vals:
            if cav_idx == 0:
                z_low_f = z_low_b = foundation_top
            else:
                z_low_f, _ = raytrace_log_z_surfaces(low_log, low_flip, low_rot, low_z0, low_s, x, y_front)
                z_low_b, _ = raytrace_log_z_surfaces(low_log, low_flip, low_rot, low_z0, low_s, x, y_back)

            _, z_high_f = raytrace_log_z_surfaces(high_log, high_flip, high_rot, high_z0, high_s, x, y_front)
            _, z_high_b = raytrace_log_z_surfaces(high_log, high_flip, high_rot, high_z0, high_s, x, y_back)

            z_lf = z_low_f - (0.0 if cav_idx == 0 else EMBED_MM)
            z_lb = z_low_b - (0.0 if cav_idx == 0 else EMBED_MM)
            z_hf = max(z_lf + 1.0, z_high_f + EMBED_MM)
            z_hb = max(z_lb + 1.0, z_high_b + EMBED_MM)

            for py, pz in ((y_front, z_lf), (y_front, z_hf), (y_back, z_hb), (y_back, z_lb)):
                wx = origin_x + (x if axis == "X" else py)
                wy = origin_y + (py if axis == "X" else x)
                verts.append(Point(x=I(wx), y=I(wy), z=I(origin_z + pz)))

        for i in range(NUM_X_STATIONS - 1):
            s0, s1 = i * 4, (i + 1) * 4
            for j in range(4):
                jn = (j + 1) % 4
                if axis == "X":
                    faces.extend([(s0 + j, s1 + j, s1 + jn), (s0 + j, s1 + jn, s0 + jn)])
                else:
                    faces.extend([(s0 + j, s1 + jn, s1 + j), (s0 + j, s0 + jn, s1 + jn)])

        last = (NUM_X_STATIONS - 1) * 4
        if axis == "X":
            faces.extend([(0, 1, 2), (0, 2, 3), (last, last + 3, last + 2), (last, last + 2, last + 1)])
        else:
            faces.extend([(0, 2, 1), (0, 3, 2), (last, last + 2, last + 3), (last, last + 1, last + 2)])

        moss_meshes.append(Mesh(name=name, vertices=verts, faces=faces, is_watertight=True, color=MOSS_COLOR))

    return moss_meshes


def build_foundation_slab_mesh() -> Mesh:
    x0, x1 = SLAB_X_MIN, SLAB_X_MAX
    y0, y1 = SLAB_Y_MIN, SLAB_Y_MAX
    z0, z1 = 0.0, SLAB_HEIGHT

    verts = [
        Point(x=I(x0), y=I(y0), z=I(z0)), Point(x=I(x1), y=I(y0), z=I(z0)),
        Point(x=I(x1), y=I(y1), z=I(z0)), Point(x=I(x0), y=I(y1), z=I(z0)),
        Point(x=I(x0), y=I(y0), z=I(z1)), Point(x=I(x1), y=I(y0), z=I(z1)),
        Point(x=I(x1), y=I(y1), z=I(z1)), Point(x=I(x0), y=I(y1), z=I(z1)),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return Mesh(name="foundation_slab_body", vertices=verts, faces=faces, is_watertight=True, color=SLAB_COLOR)


def build_site_block() -> Site:
    site = Site(name="clearing")
    terrain = Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN", name="terrain")
    terrain.add(CLEARING)
    site.add(terrain)
    return site


def assert_the_slab_sits_on_the_ground(tol=25.0):
    for tag, x, y in [
        ("SW", SLAB_X_MIN, SLAB_Y_MIN), ("SE", SLAB_X_MAX, SLAB_Y_MIN),
        ("NE", SLAB_X_MAX, SLAB_Y_MAX), ("NW", SLAB_X_MIN, SLAB_Y_MAX), ("centre", 0.0, 0.0),
    ]:
        ground = CLEARING.height_at(x=x, y=y)
        assert abs(0.0 - ground) <= tol, f"Slab {tag} corner hangs {0.0 - ground:.0f}mm over clearing"


def make_frame_sweeps(prefix: str, width: int, height: int, y: int, profile: tuple[int, int]) -> list[Sweep]:
    k0 = profile[1] // 2
    mat = Material(key="Timber_C24", profile_mm=profile)
    sill = Sweep(name=f"{prefix}_sill", path=[Point(x=0, y=y, z=k0), Point(x=width, y=y, z=k0)], material=mat)
    head = Sweep(name=f"{prefix}_head", path=[Point(x=0, y=y, z=height - k0), Point(x=width, y=y, z=height - k0)], material=mat)
    left = Sweep(name=f"{prefix}_left", path=[Point(x=k0, y=y, z=0), Point(x=k0, y=y, z=height)], material=mat)
    right = Sweep(name=f"{prefix}_right", path=[Point(x=width - k0, y=y, z=0), Point(x=width - k0, y=y, z=height)], material=mat)
    miter(sill, left, at=Point(x=0, y=y, z=0), edge=(0, 1, 0))
    miter(sill, right, at=Point(x=width, y=y, z=0), edge=(0, 1, 0))
    miter(head, left, at=Point(x=0, y=y, z=height), edge=(0, 1, 0))
    miter(head, right, at=Point(x=width, y=y, z=height), edge=(0, 1, 0))
    return [sill, head, left, right]


def make_opening_sweeps(width: int, height: int) -> list[Sweep]:
    return (
        make_frame_sweeps("frame", width, height, 100, (45, 120))
        + make_frame_sweeps("cladding", width, height, 183, (120, 45))
    )


def make_window(name: str = "window", width: int = 1000, height: int = 1200) -> Window:
    win = Window(name=name, width=width, height=height)
    win.add(*make_opening_sweeps(width, height))
    return win


CABIN_WINDOW = Product(make_window("source", 1000, 1200), name="cabin_window_1000x1200")


def make_door(name: str, width: int = 1000, height: int = 2100) -> Door:
    door = Door(name=name, width=width, height=height)
    door.add(*make_opening_sweeps(width, height))
    return door


def generate_project() -> Project:
    proj = Project(name="Laftehus — Procedural Log Cabin (4x6 m Single Room)")

    site = build_site_block()
    assert_the_slab_sits_on_the_ground()
    proj.add(site)

    rng = np.random.RandomState(RANDOM_SEED)

    logs_s = [make_log(i, LOG_LEN_LONG, rng) for i in range(NUM_LOGS_PER_WALL)]
    logs_n = [make_log(i + NUM_LOGS_PER_WALL, LOG_LEN_LONG, rng) for i in range(NUM_LOGS_PER_WALL)]
    logs_w = [make_log(i + 2 * NUM_LOGS_PER_WALL, LOG_LEN_SHORT, rng) for i in range(NUM_LOGS_PER_WALL)]
    logs_e = [make_log(i + 3 * NUM_LOGS_PER_WALL, LOG_LEN_SHORT, rng) for i in range(NUM_LOGS_PER_WALL)]

    opt_s = optimize_log_wall(logs_s, base_height=SLAB_HEIGHT)
    opt_n = optimize_log_wall(logs_n, base_height=SLAB_HEIGHT)
    opt_w = optimize_log_wall(logs_w, base_height=SLAB_HEIGHT + WALL_Z_RAISE)
    opt_e = optimize_log_wall(logs_e, base_height=SLAB_HEIGHT + WALL_Z_RAISE)

    slab_elem = Element(ifc_class="IfcSlab", name="foundation_slab")
    slab_elem.add(build_foundation_slab_mesh())
    proj.add(slab_elem)

    half_lx = CABIN_LENGTH / 2.0
    half_ly = CABIN_WIDTH / 2.0

    wall_configs = [
        ("south", logs_s, opt_s, "X", 0.0, -half_ly, -half_lx, half_lx, LOG_PALETTES[0]),
        ("north", logs_n, opt_n, "X", 0.0, half_ly, -half_lx, half_lx, LOG_PALETTES[1]),
        ("west", logs_w, opt_w, "Y", -half_lx, 0.0, -half_ly, half_ly, LOG_PALETTES[2]),
        ("east", logs_e, opt_e, "Y", half_lx, 0.0, -half_ly, half_ly, LOG_PALETTES[3]),
    ]

    for prefix, w_logs, opt, axis, ox, oy, xmin, xmax, pal in wall_configs:
        wall = Wall(name=f"log_wall_{prefix}")
        for rank in range(NUM_LOGS_PER_WALL):
            lid = opt[0][rank]
            wall.add(
                build_log_mesh(
                    w_logs[lid], opt[1][rank], opt[2][rank], opt[3][rank][0], opt[3][rank][1],
                    pal[rank % len(pal)], f"{prefix}_tier_{rank + 1}_id_{lid}", axis, ox, oy, 0.0,
                )
            )

        for m in build_moss_meshes(
            SLAB_HEIGHT, w_logs, opt[0], opt[1], opt[2], opt[3], prefix, axis, ox, oy, 0.0, xmin, xmax
        ):
            wall.add(m)

        if prefix == "south":
            wall.opening(make_door("door_south", 1000, 2100), along=1200, up=0)
            wall.opening(CABIN_WINDOW.occurrence(name="win_south"), along=4200, up=800)
        elif prefix == "north":
            wall.opening(CABIN_WINDOW.occurrence(name="win_north"), along_center=3250, up=800)
        elif prefix == "west":
            wall.opening(CABIN_WINDOW.occurrence(name="win_west"), along_center=2250, up=800)
        elif prefix == "east":
            wall.opening(CABIN_WINDOW.occurrence(name="win_east"), along_center=2250, up=800)

        proj.add(wall)

    return proj


result = generate_project()

if __name__ == "__main__":
    compile_main()
