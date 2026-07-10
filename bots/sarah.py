from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

import numpy as np


# =========================
# Constants
# =========================

NUM_DIRECTIONS = 16
DIRECTIONS = np.array([
    [np.cos(2 * np.pi * i / NUM_DIRECTIONS),
     np.sin(2 * np.pi * i / NUM_DIRECTIONS)]
    for i in range(NUM_DIRECTIONS)
], dtype=float)

# Used only when already split. Chase is disabled then, so 8-way movement is
# enough and cuts repeated safety scoring roughly in half.
DIRECTIONS_FAST = DIRECTIONS[::2]

ARENA_SIZE = 60.0
OFF_MAP_PENALTY = 1_000_000_000.0

# Performance caps
MAX_FOOD_CONSIDERED = 28
MAX_BLOBS_CONSIDERED = 22
MAX_VIRUSES_CONSIDERED = 12
MAX_OWN_BLOBS_CONSIDERED = 3


# Movement
STEP_DISTANCE_MULT = 1.5
TURN_WEIGHT = 80.0
LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)

# Food target locking / clustering
FOOD_REACHED_DIST_MULT = 0.40
FOOD_REACHED_DIST_MIN = 0.35
FOOD_TARGET_MAX_TICKS = 10
FOOD_NO_PROGRESS_LIMIT = 4
FOOD_BLACKLIST_TICKS = 25
FOOD_CORNER_MARGIN = 0.45
FOOD_CORNER_RADIUS_MULT = 1.6  # corner exclusion also scales with own size

# Corner/dead-end awareness. Danger/virus scoring alone doesn't discourage
# heading into a pocket where two walls meet, so most overrides that read
# report["score"] never learn to avoid it until they're already stuck there.
CORNER_AWARE_RADIUS_MULT = 6.0
CORNER_PRESSURE_WEIGHT = 2200.0
CORNER_ESCAPE_WEIGHT = 0.6
FOOD_CLUSTER_RADIUS = 4.0
FOOD_CLUSTER_WEIGHT = 0.55
FOOD_DISTANCE_POWER = 1.15
FOOD_MIN_TARGET_DIST = 0.15
FOOD_SCORE_WEIGHT = 900.0

# Roam / unstuck
ROAM_LOCK_TICKS = 6
ROAM_CENTER_WEIGHT = 350.0
ROAM_MOVE_WEIGHT = 1000.0
ROAM_STICKINESS_WEIGHT = 250.0
WALL_BUFFER = 0.25
STUCK_TICK_LIMIT = 4
STUCK_MOVE_EPS_MULT = 0.04
STUCK_CENTER_WEIGHT = 600.0
STUCK_MOVE_WEIGHT = 2000.0

# Enemy danger
DANGER_OVERRIDE_RANGE_MULT = 9.0
DANGER_DIRECT_RATIO = 1.06
DANGER_SPLIT_RATIO = 1.08
DANGER_SPLIT_RANGE_MULT = 5.0
DANGER_WEIGHT = 120000.0
DANGER_DISTANCE_POWER = 1.2
DANGER_HARD_CLOSE_MULT = 2.5
DANGER_WALL_PUSH_PENALTY = 50000.0
DANGER_MOVE_REWARD = 5000.0

# Enemy scoring / chasing
EAT_RATIO = 1.12
ENEMY_DANGER_WEIGHT = 14000.0
ENEMY_HUNT_WEIGHT = 2500.0
CHASE_RANGE_MULT = 7.0
CHASE_LOCK_TICKS = 14
CHASE_LOST_LIMIT = 8
CHASE_LEAD_TICKS = 2.6
CHASE_MIN_CLOSING_RATE = -1.25
CHASE_WALL_DIST = 8.0
CHASE_CLOSE_WEIGHT = 24.0
CHASE_SIZE_WEIGHT = 5.0
CHASE_STICKINESS_WEIGHT = 8.0
CHASE_BLOCK_DIST_WEIGHT = 8.0
CHASE_CENTER_SIDE_WEIGHT = 42.0
CHASE_DIRECT_EDGE_MULT = 2.8

# Virus safety / farming
VIRUS_DANGER_WEIGHT = 12000.0
VIRUS_PATH_BUFFER_MULT = 0.65
VIRUS_FARM_EAT_RATIO = 1.10
VIRUS_FARM_ENEMY_RANGE_MULT = 8.0
VIRUS_FARM_PIECE_RADIUS_MULT = 0.35
VIRUS_MEMORY_REACHED_DIST_MULT = 1.4
VIRUS_MEMORY_KEY_DECIMALS = 1
VIRUS_LOCK_TICKS = 18
VIRUS_SWITCH_DIST_MULT = 0.65
VIRUS_MEMORY_MAX = 30
VIRUS_MAX_CANDIDATES = 8

# Split kill
SPLIT_MIN_RADIUS = 1.35
SPLIT_EAT_RATIO = 1.10
SPLIT_RANGE_MULT = 2.5
SPLIT_RANGE_SAFETY_MULT = 0.75
SPLIT_TARGET_MIN_RADIUS_MULT = 0.16
SPLIT_VIRUS_BUFFER_MULT = 1.3
SPLIT_POST_DANGER_RANGE_MULT = 5.5
SPLIT_HIT_BUFFER = 0.75
SPLIT_MAX_OWN_BLOBS = 4
SPLIT_HUGE_RADIUS = 40.0
SPLIT_HUGE_MAX_OWN_BLOBS = 2
SPLIT_REACH_MARGIN_MULT = 0.12
SPLIT_LANDING_DANGER_RATIO = 1.08
SPLIT_MAX_TARGETS = 7
SPLIT_MAX_LAUNCHERS = 3
SPLIT_TOP_EXACT_CHECKS = 2
SPLIT_PRECHECK_LIMIT = 10



# =========================
# Global memory
# =========================

FOOD_TARGET_KEY = None
FOOD_TARGET_TICKS = 0
FOOD_TARGET_NO_PROGRESS = 0
FOOD_TARGET_LAST_DIST = None
FOOD_BLACKLIST = {}

ROAM_DIRECTION = None
ROAM_TICKS = 0

LAST_POSITION = None
STUCK_TICKS = 0

ENEMY_LAST_POS = {}
CHASE_TARGET_KEY = None
CHASE_TARGET_TICKS = 0
CHASE_LOST_TICKS = 0

VIRUS_MEMORY = {}
VIRUS_TARGET_KEY = None
VIRUS_TARGET_TICKS = 0


# =========================
# Basic helpers
# =========================

def vec_norm(vec):
    norm = np.linalg.norm(vec)
    if norm <= 1e-9:
        return None
    return vec / norm


def corner_pressure_batch(x_arr, y_arr, player_radius):
    """How boxed-in a set of positions is, in [0, 1].

    Only high when BOTH axes are close to a wall (an actual corner pocket);
    being near a single wall alone stays low, since one escape axis remains
    free. The ramp-in distance scales with player_radius so a bigger blob
    starts feeling squeezed further out from the literal corner point, which
    is what actually determines maneuverability.
    """
    scale = max(player_radius * CORNER_AWARE_RADIUS_MULT, 3.0)
    dx_wall = np.minimum(x_arr, ARENA_SIZE - x_arr)
    dy_wall = np.minimum(y_arr, ARENA_SIZE - y_arr)
    close_x = np.clip(1.0 - dx_wall / scale, 0.0, 1.0)
    close_y = np.clip(1.0 - dy_wall / scale, 0.0, 1.0)
    return close_x * close_y


def clamp_for_radius(x, y, radius):
    min_pos = radius + WALL_BUFFER
    max_pos = ARENA_SIZE - radius - WALL_BUFFER

    if min_pos > max_pos:
        min_pos = radius
        max_pos = ARENA_SIZE - radius

    return (
        min(max(float(x), min_pos), max_pos),
        min(max(float(y), min_pos), max_pos),
    )


def clamp_many_for_radius(points, radii):
    """Vectorized clamp for many own blobs."""
    min_pos = radii + WALL_BUFFER
    max_pos = ARENA_SIZE - radii - WALL_BUFFER

    too_big = min_pos > max_pos
    if np.any(too_big):
        min_pos = np.where(too_big, radii, min_pos)
        max_pos = np.where(too_big, ARENA_SIZE - radii, max_pos)

    out = points.copy()
    out[:, 0] = np.minimum(np.maximum(out[:, 0], min_pos), max_pos)
    out[:, 1] = np.minimum(np.maximum(out[:, 1], min_pos), max_pos)
    return out


def clamp_player(cache, x, y):
    return clamp_for_radius(x, y, cache["player_radius"])


def cap_nearest(locs, max_items, player_pos, *arrays):
    if len(locs) <= max_items:
        return (locs, *arrays)

    dists = np.sum((locs - player_pos) ** 2, axis=1)
    keep = np.argpartition(dists, max_items - 1)[:max_items]

    result = [locs[keep]]
    for arr in arrays:
        result.append(arr[keep])

    return tuple(result)


def move_future(cache, dx, dy, step_distance):
    future_x = cache["player_x"] + dx * step_distance
    future_y = cache["player_y"] + dy * step_distance
    return clamp_player(cache, future_x, future_y)


def move_distance(cache, future_x, future_y):
    return float(np.linalg.norm(
        np.array([
            future_x - cache["player_x"],
            future_y - cache["player_y"],
        ], dtype=float)
    ))


def report_to_result(report, split=False):
    return float(report["dx"]), float(report["dy"]), bool(split)


def report_towards(cache, target_vec, reports=None, stickiness=True, danger_scale=0.001):
    target_dir = vec_norm(target_vec)
    if target_dir is None:
        return None

    if reports is None:
        reports = cache["safe_reports"]

    best_score = -float("inf")
    best_report = None

    for report in reports:
        direction = report["dir"]
        score = 1000.0 * np.dot(direction, target_dir)
        score += danger_scale * report["score"]
        if stickiness:
            score += TURN_WEIGHT * np.dot(direction, LAST_DIRECTION)

        if score > best_score:
            best_score = score
            best_report = report

    return best_report


def report_for_direction(cache, direction):
    if direction is None:
        return None

    best_dot = -float("inf")
    best_report = None

    for report in cache["move_reports"]:
        dot = float(np.dot(report["dir"], direction))
        if dot > best_dot:
            best_dot = dot
            best_report = report

    return best_report


def exact_move_safe(cache, direction, step_distance, check_virus=True, enemy_only=False):
    """Score one exact direction when 16-way snapping is too coarse."""
    if direction is None:
        return False

    dx, dy = float(direction[0]), float(direction[1])

    if enemy_only:
        enemy = score_enemy_threat(cache, dx, dy, step_distance)
        return enemy["safe"]

    report = score_move(cache, dx, dy, step_distance, check_virus=check_virus)
    return report["safe"] and report["actual_move"] >= step_distance * 0.18


# =========================
# Raw cache
# =========================

def cache_raw(game):
    me = game.state.me

    player_pos = np.array([me.x, me.y], dtype=float)
    player_radius = float(me.radius)
    player_id = me.player_id

    own_blobs = list(me.blobs.values())
    if own_blobs:
        own_locs_all = np.array([blob.pos for blob in own_blobs], dtype=float)
        own_rads_all = np.array([blob.radius for blob in own_blobs], dtype=float)
    else:
        own_locs_all = np.array([[me.x, me.y]], dtype=float)
        own_rads_all = np.array([me.radius], dtype=float)

    own_blob_count = len(own_rads_all)

    # Safety calculations are the main timeout risk when we are heavily split.
    # Keep the true blob count for decision logic, but only score the largest
    # pieces for repeated movement-safety reports.
    if own_blob_count > MAX_OWN_BLOBS_CONSIDERED:
        keep = np.argsort(own_rads_all)[-MAX_OWN_BLOBS_CONSIDERED:]
        own_locs = own_locs_all[keep]
        own_rads = own_rads_all[keep]
    else:
        own_locs = own_locs_all
        own_rads = own_rads_all

    foods = game.state.visible_food
    if foods:
        food_locs = np.array([food.pos for food in foods], dtype=float)
        if len(food_locs) > MAX_FOOD_CONSIDERED:
            food_locs, = cap_nearest(food_locs, MAX_FOOD_CONSIDERED, player_pos)
    else:
        food_locs = np.empty((0, 2), dtype=float)

    visible_blobs = game.state.visible_blobs
    enemy_blobs = [
        blob for blob in visible_blobs
        if getattr(blob, "player_id", None) != player_id
    ]

    if enemy_blobs:
        blob_locs = np.array([blob.pos for blob in enemy_blobs], dtype=float)
        blob_rads = np.array([blob.radius for blob in enemy_blobs], dtype=float)
        blob_player_ids = np.array([blob.player_id for blob in enemy_blobs], dtype=int)

        if len(blob_locs) > MAX_BLOBS_CONSIDERED:
            blob_locs, blob_rads, blob_player_ids = cap_nearest(
                blob_locs,
                MAX_BLOBS_CONSIDERED,
                player_pos,
                blob_rads,
                blob_player_ids,
            )

        merged_rads = np.zeros(len(blob_rads), dtype=float)
        for pid in np.unique(blob_player_ids):
            same_player = blob_player_ids == pid
            total_mass = np.sum(blob_rads[same_player] ** 2)
            merged_rads[same_player] = np.sqrt(total_mass)
    else:
        blob_locs = np.empty((0, 2), dtype=float)
        blob_rads = np.empty(0, dtype=float)
        blob_player_ids = np.empty(0, dtype=int)
        merged_rads = np.empty(0, dtype=float)

    viruses = game.state.visible_viruses
    if viruses:
        virus_locs = np.array([virus.pos for virus in viruses], dtype=float)
        virus_rads = np.array([virus.radius for virus in viruses], dtype=float)
        if len(virus_locs) > MAX_VIRUSES_CONSIDERED:
            virus_locs, virus_rads = cap_nearest(
                virus_locs,
                MAX_VIRUSES_CONSIDERED,
                player_pos,
                virus_rads,
            )
    else:
        virus_locs = np.empty((0, 2), dtype=float)
        virus_rads = np.empty(0, dtype=float)

    return {
        "player_pos": player_pos,
        "player_x": float(player_pos[0]),
        "player_y": float(player_pos[1]),
        "player_radius": player_radius,
        "player_id": player_id,

        "own_locs": own_locs,
        "own_rads": own_rads,
        "own_blob_count": own_blob_count,

        "food_locs": food_locs,

        "blob_locs": blob_locs,
        "blob_rads": blob_rads,
        "blob_player_ids": blob_player_ids,
        "merged_rads": merged_rads,

        "virus_locs": virus_locs,
        "virus_rads": virus_rads,
    }


# =========================
# Score functions
# =========================

def score_virus_segment(cache, start_pos, end_pos, radius, buffer_mult):
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if len(virus_locs) == 0:
        return 0.0

    dangerous = radius > virus_rads * 1.1
    if not np.any(dangerous):
        return 0.0

    segment = end_pos - start_pos
    segment_len_sq = np.dot(segment, segment)

    if segment_len_sq <= 1e-9:
        dists = np.linalg.norm(virus_locs - start_pos, axis=1)
    else:
        to_viruses = virus_locs - start_pos
        t = np.sum(to_viruses * segment, axis=1) / segment_len_sq
        t = np.clip(t, 0.0, 1.0)
        closest_points = start_pos + t.reshape(-1, 1) * segment
        dists = np.linalg.norm(virus_locs - closest_points, axis=1)

    clearances = dists - radius - virus_rads

    if np.any(dangerous & (clearances < radius * buffer_mult)):
        return -OFF_MAP_PENALTY

    clearances[clearances < 1.0] = 1.0
    return -float(np.sum(VIRUS_DANGER_WEIGHT / (clearances[dangerous] ** 3)))


def score_own_blob_virus(cache, dx, dy, step_distance):
    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if len(own_locs) == 0 or len(virus_locs) == 0:
        return 0.0

    dangerous = own_rads[:, None] > virus_rads[None, :] * 1.1
    if not np.any(dangerous):
        return 0.0

    starts = own_locs
    ends = own_locs + np.array([dx * step_distance, dy * step_distance], dtype=float)
    ends = clamp_many_for_radius(ends, own_rads)

    segments = ends - starts
    seg_len_sq = np.sum(segments * segments, axis=1)

    to_viruses = virus_locs[None, :, :] - starts[:, None, :]

    # Projection of each virus onto each own-blob movement segment.
    denom = np.where(seg_len_sq <= 1e-9, 1.0, seg_len_sq)
    t = np.sum(to_viruses * segments[:, None, :], axis=2) / denom[:, None]
    t = np.clip(t, 0.0, 1.0)
    t[seg_len_sq <= 1e-9, :] = 0.0

    closest = starts[:, None, :] + t[:, :, None] * segments[:, None, :]
    dists = np.linalg.norm(virus_locs[None, :, :] - closest, axis=2)
    clearances = dists - own_rads[:, None] - virus_rads[None, :]

    if np.any(dangerous & (clearances < own_rads[:, None] * VIRUS_PATH_BUFFER_MULT)):
        return -OFF_MAP_PENALTY

    safe_clearances = np.maximum(clearances, 1.0)
    scores = VIRUS_DANGER_WEIGHT / (safe_clearances ** 3)
    return -float(np.sum(scores[dangerous]))

def score_enemy_threat(cache, dx=0.0, dy=0.0, step_distance=0.0):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) == 0:
        return {"active": False, "safe": True, "score": 0.0}

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]

    if len(own_locs) == 0:
        return {"active": False, "safe": True, "score": 0.0}

    player_radius = cache["player_radius"]
    danger_size = cache.get("enemy_danger_size")
    if danger_size is None or len(danger_size) != len(blob_rads):
        danger_size = np.maximum(blob_rads, cache["merged_rads"])

    enemy_split_rads = cache.get("enemy_split_rads")
    if enemy_split_rads is None or len(enemy_split_rads) != len(blob_rads):
        enemy_split_rads = blob_rads / np.sqrt(2.0)

    futures = own_locs + np.array([dx * step_distance, dy * step_distance], dtype=float)
    futures = clamp_many_for_radius(futures, own_rads)

    dists = np.linalg.norm(blob_locs[None, :, :] - futures[:, None, :], axis=2)
    edge_dists = dists - own_rads[:, None] - blob_rads[None, :]

    direct_danger = danger_size[None, :] > own_rads[:, None] * DANGER_DIRECT_RATIO
    direct_active = direct_danger & (edge_dists < player_radius * DANGER_OVERRIDE_RANGE_MULT)
    direct_unsafe = direct_danger & (edge_dists < own_rads[:, None] * DANGER_HARD_CLOSE_MULT)

    split_danger = enemy_split_rads[None, :] > own_rads[:, None] * DANGER_SPLIT_RATIO
    split_near = edge_dists < blob_rads[None, :] * DANGER_SPLIT_RANGE_MULT
    split_active = split_danger & split_near

    active = bool(np.any(direct_active | split_active))
    unsafe = bool(np.any(direct_unsafe | split_active))

    score_dists = np.maximum(edge_dists, 1.0)
    scored = direct_danger | split_active
    danger_scores = (
        DANGER_WEIGHT
        * (danger_size[None, :] / own_rads[:, None])
        / (score_dists ** DANGER_DISTANCE_POWER)
    )

    return {
        "active": active,
        "safe": not unsafe,
        "score": -float(np.sum(danger_scores[scored])),
    }

def score_move(cache, dx, dy, step_distance, check_virus=True):
    enemy = score_enemy_threat(cache, dx, dy, step_distance)
    virus_score = score_own_blob_virus(cache, dx, dy, step_distance) if check_virus else 0.0
    virus_safe = virus_score > -OFF_MAP_PENALTY / 2

    future_x, future_y = move_future(cache, dx, dy, step_distance)
    actual_move = move_distance(cache, future_x, future_y)

    return {
        "enemy_active": enemy["active"],
        "enemy_safe": enemy["safe"],
        "virus_safe": bool(virus_safe),
        "safe": bool(enemy["safe"] and virus_safe),
        "score": float(enemy["score"] + virus_score),
        "enemy_score": float(enemy["score"]),
        "virus_score": float(virus_score),
        "actual_move": float(actual_move),
        "future_x": float(future_x),
        "future_y": float(future_y),
    }


def score_moves_batch(cache, directions, step_distance, check_virus=True):
    """Vectorized equivalent of calling score_move() once per direction.

    This computes the exact same per-direction quantities as score_move ->
    score_enemy_threat / score_own_blob_virus, but does it for every direction
    at once via broadcasting instead of looping in Python and re-entering NumPy
    16 (or 8) separate times per tick. The formulas are untouched; only the
    evaluation strategy changed. This is the main late-game CPU saver, since
    cache_move_reports runs unconditionally every tick regardless of which
    override ends up firing.
    """
    directions = np.asarray(directions, dtype=float)
    num_dirs = len(directions)
    deltas = directions * step_distance  # (D, 2)

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]
    num_own = len(own_locs)

    # --- future positions of every own blob, for every direction: (D, O, 2) ---
    futures_own = own_locs[None, :, :] + deltas[:, None, :]

    if num_own > 0:
        min_pos = own_rads + WALL_BUFFER
        max_pos = ARENA_SIZE - own_rads - WALL_BUFFER
        too_big = min_pos > max_pos
        if np.any(too_big):
            min_pos = np.where(too_big, own_rads, min_pos)
            max_pos = np.where(too_big, ARENA_SIZE - own_rads, max_pos)
        futures_own[:, :, 0] = np.clip(futures_own[:, :, 0], min_pos[None, :], max_pos[None, :])
        futures_own[:, :, 1] = np.clip(futures_own[:, :, 1], min_pos[None, :], max_pos[None, :])

    # --- enemy threat, batched over directions ---
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    player_radius = cache["player_radius"]

    if len(blob_locs) == 0 or num_own == 0:
        enemy_active = np.zeros(num_dirs, dtype=bool)
        enemy_safe = np.ones(num_dirs, dtype=bool)
        enemy_score = np.zeros(num_dirs, dtype=float)
    else:
        danger_size = cache.get("enemy_danger_size")
        if danger_size is None or len(danger_size) != len(blob_rads):
            danger_size = np.maximum(blob_rads, cache["merged_rads"])

        enemy_split_rads = cache.get("enemy_split_rads")
        if enemy_split_rads is None or len(enemy_split_rads) != len(blob_rads):
            enemy_split_rads = blob_rads / np.sqrt(2.0)

        # (D, O, E)
        dists = np.linalg.norm(
            blob_locs[None, None, :, :] - futures_own[:, :, None, :], axis=3
        )
        edge_dists = dists - own_rads[None, :, None] - blob_rads[None, None, :]

        direct_danger = danger_size[None, None, :] > own_rads[None, :, None] * DANGER_DIRECT_RATIO
        direct_active = direct_danger & (edge_dists < player_radius * DANGER_OVERRIDE_RANGE_MULT)
        direct_unsafe = direct_danger & (edge_dists < own_rads[None, :, None] * DANGER_HARD_CLOSE_MULT)

        split_danger = enemy_split_rads[None, None, :] > own_rads[None, :, None] * DANGER_SPLIT_RATIO
        split_near = edge_dists < blob_rads[None, None, :] * DANGER_SPLIT_RANGE_MULT
        split_active = split_danger & split_near

        active_mask = direct_active | split_active
        unsafe_mask = direct_unsafe | split_active

        enemy_active = np.any(active_mask, axis=(1, 2))
        enemy_safe = ~np.any(unsafe_mask, axis=(1, 2))

        score_dists = np.maximum(edge_dists, 1.0)
        scored = direct_danger | split_active
        danger_scores = (
            DANGER_WEIGHT
            * (danger_size[None, None, :] / own_rads[None, :, None])
            / (score_dists ** DANGER_DISTANCE_POWER)
        )
        enemy_score = -np.sum(np.where(scored, danger_scores, 0.0), axis=(1, 2))

    # --- own-blob virus safety, batched over directions ---
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if not check_virus or len(virus_locs) == 0 or num_own == 0:
        virus_score = np.zeros(num_dirs, dtype=float)
        virus_safe = np.ones(num_dirs, dtype=bool)
    else:
        dangerous = own_rads[:, None] > virus_rads[None, :] * 1.1  # (O, V)
        if not np.any(dangerous):
            virus_score = np.zeros(num_dirs, dtype=float)
            virus_safe = np.ones(num_dirs, dtype=bool)
        else:
            starts = own_locs  # (O, 2)
            segments = futures_own - starts[None, :, :]  # (D, O, 2)
            seg_len_sq = np.sum(segments * segments, axis=2)  # (D, O)

            to_viruses = virus_locs[None, None, :, :] - starts[None, :, None, :]  # (1, O, V, 2)
            denom = np.where(seg_len_sq <= 1e-9, 1.0, seg_len_sq)[:, :, None]  # (D, O, 1)
            t = np.sum(to_viruses * segments[:, :, None, :], axis=3) / denom  # (D, O, V)
            t = np.clip(t, 0.0, 1.0)
            zero_seg = seg_len_sq <= 1e-9  # (D, O)
            t = np.where(zero_seg[:, :, None], 0.0, t)

            closest = starts[None, :, None, :] + t[:, :, :, None] * segments[:, :, None, :]  # (D, O, V, 2)
            dists_v = np.linalg.norm(virus_locs[None, None, :, :] - closest, axis=3)  # (D, O, V)
            clearances = dists_v - own_rads[None, :, None] - virus_rads[None, None, :]

            danger_broadcast = dangerous[None, :, :]  # (1, O, V)
            unsafe_mask_v = danger_broadcast & (clearances < own_rads[None, :, None] * VIRUS_PATH_BUFFER_MULT)
            unsafe_any = np.any(unsafe_mask_v, axis=(1, 2))  # (D,)

            safe_clearances = np.maximum(clearances, 1.0)
            v_scores = VIRUS_DANGER_WEIGHT / (safe_clearances ** 3)
            virus_score = -np.sum(np.where(danger_broadcast, v_scores, 0.0), axis=(1, 2))
            virus_score = np.where(unsafe_any, -OFF_MAP_PENALTY, virus_score)
            virus_safe = ~unsafe_any

    # --- reported (single-point) future position / actual move distance ---
    min_p = player_radius + WALL_BUFFER
    max_p = ARENA_SIZE - player_radius - WALL_BUFFER
    if min_p > max_p:
        min_p = player_radius
        max_p = ARENA_SIZE - player_radius

    future_player = cache["player_pos"][None, :] + deltas  # (D, 2)
    future_player[:, 0] = np.clip(future_player[:, 0], min_p, max_p)
    future_player[:, 1] = np.clip(future_player[:, 1], min_p, max_p)
    actual_move = np.linalg.norm(future_player - cache["player_pos"][None, :], axis=1)

    # Dead-end awareness: penalize directions that land somewhere boxed in by
    # two walls at once, so roam/food/chase/virus scoring (all of which read
    # report["score"]) naturally steer around corner pockets instead of only
    # noticing a problem once already stuck in one.
    corner_pressure = corner_pressure_batch(future_player[:, 0], future_player[:, 1], player_radius)
    corner_score = -CORNER_PRESSURE_WEIGHT * corner_pressure

    safe = enemy_safe & (virus_score > -OFF_MAP_PENALTY / 2)
    total_score = enemy_score + virus_score + corner_score

    reports = []
    for i in range(num_dirs):
        reports.append({
            "idx": i,
            "dir": directions[i],
            "dx": float(directions[i, 0]),
            "dy": float(directions[i, 1]),
            "enemy_active": bool(enemy_active[i]),
            "enemy_safe": bool(enemy_safe[i]),
            "virus_safe": bool(virus_safe[i]),
            "safe": bool(safe[i]),
            "score": float(total_score[i]),
            "enemy_score": float(enemy_score[i]),
            "virus_score": float(virus_score[i]),
            "corner_score": float(corner_score[i]),
            "actual_move": float(actual_move[i]),
            "future_x": float(future_player[i, 0]),
            "future_y": float(future_player[i, 1]),
        })

    return reports


def score_food_position(cache, x, y):
    food_locs = cache["food_locs"]
    if len(food_locs) == 0:
        return 0.0

    pos = np.array([x, y], dtype=float)
    dists = np.linalg.norm(food_locs - pos, axis=1)
    dists[dists < 1.0] = 1.0
    return float(np.sum(FOOD_SCORE_WEIGHT / (dists ** 2)))


def score_enemy_position(cache, x, y):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return 0.0

    pos = np.array([x, y], dtype=float)
    player_radius = cache["player_radius"]

    dists = np.linalg.norm(blob_locs - pos, axis=1)
    edge_dists = dists - player_radius - blob_rads
    edge_dists[edge_dists < 1.0] = 1.0

    danger_size = np.maximum(blob_rads, merged_rads)
    dangerous = danger_size > player_radius * 1.08
    edible = player_radius > blob_rads * EAT_RATIO

    danger_scores = (
        ENEMY_DANGER_WEIGHT
        * (danger_size / player_radius)
        / (edge_dists ** 1.3)
    )
    hunt_scores = (
        ENEMY_HUNT_WEIGHT
        * (player_radius / blob_rads)
        / (edge_dists ** 2)
    )

    score = -np.sum(danger_scores[dangerous])
    if cache["own_blob_count"] == 1:
        score += np.sum(hunt_scores[edible])

    return float(score)


# =========================
# Derived cache functions
# =========================

def cache_enemy_velocity(cache):
    """Estimate per-enemy velocity by matching blobs from the same player."""
    global ENEMY_LAST_POS

    blob_locs = cache["blob_locs"]
    blob_player_ids = cache["blob_player_ids"]
    blob_vels = np.zeros_like(blob_locs, dtype=float)

    if len(blob_locs) == 0:
        ENEMY_LAST_POS = {}
        cache["blob_vels"] = blob_vels
        return

    previous_by_pid = {}
    for key, pos in ENEMY_LAST_POS.items():
        previous_by_pid.setdefault(key[0], []).append((key, pos))

    new_last = {}

    for pid in np.unique(blob_player_ids):
        indices = np.where(blob_player_ids == pid)[0]
        previous = previous_by_pid.get(int(pid), [])
        used_previous = set()

        for local_num, idx in enumerate(indices):
            pos = blob_locs[idx].copy()
            best_key = None
            best_pos = None
            best_dist = float("inf")

            for prev_key, prev_pos in previous:
                if prev_key in used_previous:
                    continue

                dist = float(np.sum((pos - prev_pos) ** 2))
                if dist < best_dist:
                    best_dist = dist
                    best_key = prev_key
                    best_pos = prev_pos

            if best_pos is None:
                vel = np.array([0.0, 0.0], dtype=float)
            else:
                vel = pos - best_pos
                used_previous.add(best_key)

            speed = np.linalg.norm(vel)
            max_speed = cache["player_radius"] * 2.5
            if speed > max_speed and speed > 1e-9:
                vel = vel / speed * max_speed

            blob_vels[idx] = vel
            new_last[(int(pid), int(local_num))] = pos

    ENEMY_LAST_POS = new_last
    cache["blob_vels"] = blob_vels


def cache_enemy(cache):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    if len(blob_locs) == 0:
        cache["enemy_vectors"] = np.empty((0, 2), dtype=float)
        cache["enemy_dists"] = np.empty(0, dtype=float)
        cache["enemy_edge_dists"] = np.empty(0, dtype=float)
        cache["enemy_dirs"] = np.empty((0, 2), dtype=float)
        cache["enemy_edible"] = np.empty(0, dtype=bool)
        cache["enemy_pred_locs"] = np.empty((0, 2), dtype=float)
        cache["enemy_danger_size"] = np.empty(0, dtype=float)
        cache["enemy_split_rads"] = np.empty(0, dtype=float)
        return

    vectors = blob_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)
    safe_dists = dists.copy()
    safe_dists[safe_dists < 1.0] = 1.0

    blob_vels = cache.get("blob_vels")
    if blob_vels is None or len(blob_vels) != len(blob_locs):
        blob_vels = np.zeros_like(blob_locs, dtype=float)

    pred_locs = blob_locs + blob_vels * CHASE_LEAD_TICKS
    pred_locs[:, 0] = np.clip(pred_locs[:, 0], 0.0, ARENA_SIZE)
    pred_locs[:, 1] = np.clip(pred_locs[:, 1], 0.0, ARENA_SIZE)

    danger_size = np.maximum(blob_rads, cache["merged_rads"])

    cache["enemy_vectors"] = vectors
    cache["enemy_dists"] = dists
    cache["enemy_edge_dists"] = dists - player_radius - blob_rads
    cache["enemy_dirs"] = vectors / safe_dists.reshape(-1, 1)
    cache["enemy_edible"] = player_radius > blob_rads * EAT_RATIO
    cache["enemy_pred_locs"] = pred_locs
    cache["enemy_danger_size"] = danger_size
    cache["enemy_split_rads"] = blob_rads / np.sqrt(2.0)


def cache_food(cache):
    food_locs = cache["food_locs"]
    player_pos = cache["player_pos"]

    if len(food_locs) == 0:
        cache["food_dists"] = np.empty(0, dtype=float)
        cache["food_dirs"] = np.empty((0, 2), dtype=float)
        cache["food_scores"] = np.empty(0, dtype=float)
        return

    vectors = food_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)
    safe_dists = dists.copy()
    safe_dists[safe_dists < 1.0] = 1.0

    pairwise = np.linalg.norm(
        food_locs[:, None, :] - food_locs[None, :, :],
        axis=2,
    )
    cluster_density = np.exp(-((pairwise / FOOD_CLUSTER_RADIUS) ** 2)).sum(axis=1) - 1.0

    scores = (
        (1.0 + FOOD_CLUSTER_WEIGHT * cluster_density)
        / (safe_dists ** FOOD_DISTANCE_POWER)
    )

    cache["food_dists"] = dists
    cache["food_dirs"] = vectors / safe_dists.reshape(-1, 1)
    cache["food_scores"] = scores


def virus_key(pos):
    return (
        round(float(pos[0]), VIRUS_MEMORY_KEY_DECIMALS),
        round(float(pos[1]), VIRUS_MEMORY_KEY_DECIMALS),
    )


def cache_virus_memory(cache):
    global VIRUS_MEMORY

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]
    visible_keys = set()

    for pos, rad in zip(cache["virus_locs"], cache["virus_rads"]):
        key = virus_key(pos)
        visible_keys.add(key)
        VIRUS_MEMORY[key] = {
            "pos": np.array(pos, dtype=float),
            "radius": float(rad),
        }

    reached_dist = max(1.0, player_radius * VIRUS_MEMORY_REACHED_DIST_MULT)
    expired = []

    for key, info in VIRUS_MEMORY.items():
        if key in visible_keys:
            continue
        if np.linalg.norm(info["pos"] - player_pos) < reached_dist:
            expired.append(key)

    for key in expired:
        del VIRUS_MEMORY[key]

    # Long games can remember too many old viruses. Keep closest remembered
    # viruses only; far-away memory should not cost time every tick.
    if len(VIRUS_MEMORY) > VIRUS_MEMORY_MAX:
        ordered = sorted(
            VIRUS_MEMORY.items(),
            key=lambda item: float(np.sum((item[1]["pos"] - player_pos) ** 2)),
        )
        VIRUS_MEMORY = dict(ordered[:VIRUS_MEMORY_MAX])


def cache_move_reports(cache, step_distance):
    # When we are already split, chase/virus farming are disabled and the
    # expensive part is repeated safety checks over own pieces. 8 directions is
    # enough for split-state navigation and saves a lot of late-game CPU.
    directions = DIRECTIONS_FAST if cache["own_blob_count"] > 1 else DIRECTIONS

    # v9 optimisation: score all directions in one batched/vectorized pass
    # instead of looping and calling score_move() (and therefore NumPy) once
    # per direction. Same formulas, same results, far fewer Python<->NumPy
    # round trips per tick. This is the function that runs unconditionally
    # every single tick, so its cost is what accumulates into the late-game
    # cumulative timeout.
    reports = score_moves_batch(cache, directions, step_distance, check_virus=True)

    cache["enemy_active"] = any(report["enemy_active"] for report in reports)
    cache["move_reports"] = reports
    cache["safe_reports"] = [
        report for report in reports
        if report["safe"] and report["actual_move"] >= step_distance * 0.20
    ]
    cache["enemy_safe_reports"] = [
        report for report in reports
        if report["enemy_safe"] and report["actual_move"] >= step_distance * 0.20
    ]


def cache_frame(game):
    cache = cache_raw(game)
    # Enemy velocity is only useful for chase, and chase is disabled while split.
    # Skipping matching in split states saves late-game CPU.
    if cache["own_blob_count"] == 1:
        cache_enemy_velocity(cache)
    else:
        cache["blob_vels"] = np.zeros_like(cache["blob_locs"], dtype=float)

    cache_enemy(cache)
    cache_virus_memory(cache)

    step_distance = STEP_DISTANCE_MULT * cache["player_radius"]
    cache["step_distance"] = step_distance
    cache_move_reports(cache, step_distance)

    return cache


# =========================
# Split helpers
# =========================

def split_path_safe(cache, start_pos, split_radius, split_landing):
    score = score_virus_segment(
        cache,
        start_pos,
        split_landing,
        split_radius,
        SPLIT_VIRUS_BUFFER_MULT,
    )
    return score > -OFF_MAP_PENALTY / 2


def split_candidate(launcher_pos, launcher_radius, target_pos, target_radius):
    """Return exact split geometry if this launcher can reach target."""
    split_radius = launcher_radius / np.sqrt(2.0)
    split_range = launcher_radius * SPLIT_RANGE_MULT

    vector = target_pos - launcher_pos
    dist = np.linalg.norm(vector)
    if dist <= 1e-9:
        return None

    direction = vector / dist
    safe_range = split_range * SPLIT_RANGE_SAFETY_MULT
    max_hit_dist = safe_range + split_radius + target_radius
    reach_margin = max_hit_dist - dist

    if reach_margin <= launcher_radius * SPLIT_REACH_MARGIN_MULT:
        return None

    split_landing = launcher_pos + direction * split_range

    return {
        "dir": direction,
        "split_radius": split_radius,
        "split_landing": split_landing,
        "dist": float(dist),
        "reach_margin": float(reach_margin),
    }


def split_landing_enemy_safe(cache, landing_pos, split_radius, launcher_radius, target_idx):
    """Check that the new split piece is not landing next to something that eats it."""
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return True

    danger_size = np.maximum(blob_rads, merged_rads)
    dists = np.linalg.norm(blob_locs - landing_pos, axis=1)
    edge_dists = dists - split_radius - blob_rads

    dangerous = danger_size > split_radius * SPLIT_LANDING_DANGER_RATIO
    nearby = edge_dists < launcher_radius * SPLIT_POST_DANGER_RANGE_MULT

    # Ignore the intended victim, because the whole point is to land on it.
    if 0 <= target_idx < len(dangerous):
        dangerous[target_idx] = False

    return not bool(np.any(dangerous & nearby))

def virus_enemy_safe(cache, virus_pos, virus_rad):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return True

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]
    estimated_piece_radius = player_radius * VIRUS_FARM_PIECE_RADIUS_MULT
    danger_size = np.maximum(blob_rads, merged_rads)

    can_eat_pieces = danger_size > estimated_piece_radius * 1.08

    dists_to_player = np.linalg.norm(blob_locs - player_pos, axis=1)
    edge_to_player = dists_to_player - player_radius - blob_rads

    dists_to_virus = np.linalg.norm(blob_locs - virus_pos, axis=1)
    edge_to_virus = dists_to_virus - virus_rad - blob_rads

    close_to_player = edge_to_player < player_radius * VIRUS_FARM_ENEMY_RANGE_MULT
    close_to_virus = edge_to_virus < player_radius * VIRUS_FARM_ENEMY_RANGE_MULT

    return not bool(np.any(can_eat_pieces & (close_to_player | close_to_virus)))


def enemy_near_wall(pos):
    x, y = pos
    dist_left = x
    dist_right = ARENA_SIZE - x
    dist_bottom = y
    dist_top = ARENA_SIZE - y

    near_wall = min(dist_left, dist_right, dist_bottom, dist_top) < CHASE_WALL_DIST
    near_corner = (
        (dist_left < CHASE_WALL_DIST or dist_right < CHASE_WALL_DIST)
        and (dist_bottom < CHASE_WALL_DIST or dist_top < CHASE_WALL_DIST)
    )
    return near_wall or near_corner


def chase_block_point(enemy_pos, enemy_radius, player_radius, enemy_vel=None):
    """Point that cuts off the enemy's escape route.

    In open space, being directly behind the enemy just follows their trail.
    This point is biased to the centre-side of the enemy, with a smaller bias
    toward their current velocity, so we intercept/cut off instead of shadowing.
    """
    center = np.array([ARENA_SIZE / 2.0, ARENA_SIZE / 2.0], dtype=float)
    centre_dir = vec_norm(center - enemy_pos)

    if centre_dir is None:
        return enemy_pos.copy(), None

    block_dir = centre_dir.copy()

    if enemy_vel is not None:
        vel_dir = vec_norm(enemy_vel)
        if vel_dir is not None:
            mixed = 0.75 * centre_dir + 0.25 * vel_dir
            mixed_dir = vec_norm(mixed)
            if mixed_dir is not None:
                block_dir = mixed_dir

    offset = min(player_radius * 1.15, enemy_radius * 2.5 + player_radius * 0.45)
    block_point = enemy_pos + block_dir * offset

    block_point[0] = np.clip(block_point[0], 0.0, ARENA_SIZE)
    block_point[1] = np.clip(block_point[1], 0.0, ARENA_SIZE)

    return block_point, centre_dir


# =========================
# Food memory helpers
# =========================

def food_key(food_pos):
    return (round(float(food_pos[0]), 1), round(float(food_pos[1]), 1))


def food_update_blacklist():
    global FOOD_BLACKLIST

    expired = []
    for key in FOOD_BLACKLIST:
        FOOD_BLACKLIST[key] -= 1
        if FOOD_BLACKLIST[key] <= 0:
            expired.append(key)

    for key in expired:
        del FOOD_BLACKLIST[key]


def food_blacklist(key):
    global FOOD_TARGET_KEY, FOOD_TARGET_TICKS
    global FOOD_TARGET_NO_PROGRESS, FOOD_TARGET_LAST_DIST

    if key is None:
        return

    FOOD_BLACKLIST[key] = FOOD_BLACKLIST_TICKS

    if FOOD_TARGET_KEY == key:
        FOOD_TARGET_KEY = None
        FOOD_TARGET_TICKS = 0
        FOOD_TARGET_NO_PROGRESS = 0
        FOOD_TARGET_LAST_DIST = None


def food_unreachable_corner(food_pos, player_radius=0.0):
    # The fixed FOOD_CORNER_MARGIN only rules out food sitting right on the
    # geometric corner point. For any real-sized blob that's not the region
    # that actually restricts movement, so also scale the exclusion zone with
    # how big we are: a bigger blob needs more clearance from both walls at
    # once before a corner food item is worth detouring into a pocket for.
    margin = max(FOOD_CORNER_MARGIN, player_radius * FOOD_CORNER_RADIUS_MULT)
    x, y = food_pos
    near_left = x < margin
    near_right = ARENA_SIZE - x < margin
    near_bottom = y < margin
    near_top = ARENA_SIZE - y < margin
    return (near_left or near_right) and (near_bottom or near_top)


def food_ranked_targets(cache):
    global FOOD_TARGET_KEY, FOOD_TARGET_TICKS
    global FOOD_TARGET_NO_PROGRESS, FOOD_TARGET_LAST_DIST

    # Food clustering is only needed if we actually reach food/roam mode.
    # Higher-priority split/chase/virus ticks should not pay the pairwise cost.
    if "food_scores" not in cache:
        cache_food(cache)

    food_update_blacklist()

    food_locs = cache["food_locs"]
    if len(food_locs) == 0:
        FOOD_TARGET_KEY = None
        FOOD_TARGET_TICKS = 0
        FOOD_TARGET_NO_PROGRESS = 0
        FOOD_TARGET_LAST_DIST = None
        return []

    food_keys = [food_key(food_pos) for food_pos in food_locs]
    dists = cache["food_dists"]
    dirs = cache["food_dirs"]
    scores = cache["food_scores"].copy()

    current_idx = None
    if FOOD_TARGET_KEY is not None:
        for i, key in enumerate(food_keys):
            if key == FOOD_TARGET_KEY:
                current_idx = i
                break
        if current_idx is None:
            FOOD_TARGET_KEY = None
            FOOD_TARGET_TICKS = 0
            FOOD_TARGET_NO_PROGRESS = 0
            FOOD_TARGET_LAST_DIST = None

    valid = dists > FOOD_MIN_TARGET_DIST
    for i, key in enumerate(food_keys):
        if key in FOOD_BLACKLIST or food_unreachable_corner(food_locs[i], cache["player_radius"]):
            valid[i] = False

    if not np.any(valid):
        return []

    reached_dist = max(
        FOOD_REACHED_DIST_MIN,
        cache["player_radius"] * FOOD_REACHED_DIST_MULT,
    )

    if current_idx is not None and valid[current_idx]:
        current_dist = dists[current_idx]
        FOOD_TARGET_TICKS += 1

        if FOOD_TARGET_LAST_DIST is not None:
            if current_dist >= FOOD_TARGET_LAST_DIST - 0.03:
                FOOD_TARGET_NO_PROGRESS += 1
            else:
                FOOD_TARGET_NO_PROGRESS = 0

        FOOD_TARGET_LAST_DIST = current_dist

        if (
            current_dist < reached_dist
            or FOOD_TARGET_TICKS >= FOOD_TARGET_MAX_TICKS
            or FOOD_TARGET_NO_PROGRESS >= FOOD_NO_PROGRESS_LIMIT
        ):
            food_blacklist(FOOD_TARGET_KEY)
            current_idx = None

    scores[~valid] = -1.0

    if current_idx is not None and scores[current_idx] > 0:
        scores[current_idx] *= 1.25

    candidates = []
    for idx in np.argsort(scores)[::-1]:
        if scores[idx] <= 0:
            break

        direction = dirs[idx]
        candidates.append((
            float(scores[idx]),
            float(direction[0]),
            float(direction[1]),
            food_keys[idx],
        ))

    if candidates and FOOD_TARGET_KEY is None:
        FOOD_TARGET_KEY = candidates[0][3]
        FOOD_TARGET_TICKS = 0
        FOOD_TARGET_NO_PROGRESS = 0
        FOOD_TARGET_LAST_DIST = None

    return candidates


# =========================
# State helpers
# =========================

def stuck_now(cache):
    global LAST_POSITION, STUCK_TICKS

    pos = cache["player_pos"]
    r = cache["player_radius"]

    if LAST_POSITION is None:
        LAST_POSITION = pos.copy()
        return False

    moved = np.linalg.norm(pos - LAST_POSITION)
    if moved < r * STUCK_MOVE_EPS_MULT:
        STUCK_TICKS += 1
    else:
        STUCK_TICKS = 0

    LAST_POSITION = pos.copy()
    return STUCK_TICKS >= STUCK_TICK_LIMIT


# =========================
# Override functions
# =========================

def override_escape(cache, step_distance):
    if not cache["enemy_active"]:
        return None

    best_score = -float("inf")
    best_report = None

    for report in cache["move_reports"]:
        if not report["virus_safe"]:
            continue

        score = report["enemy_score"]
        # Tiebreaker, not an override of safety: if two directions are
        # similarly safe right now, prefer the one that doesn't run into a
        # dead end. Weighted low enough that genuine danger avoidance always
        # wins first.
        score += CORNER_ESCAPE_WEIGHT * report["corner_score"]
        if report["actual_move"] < step_distance * 0.20:
            score -= DANGER_WALL_PUSH_PENALTY
        else:
            score += DANGER_MOVE_REWARD * report["actual_move"]

        if score > best_score:
            best_score = score
            best_report = report

    if best_report is None:
        return None

    return report_to_result(best_report, split=False)


def override_unstuck(cache, step_distance):
    if not stuck_now(cache):
        return None

    center = np.array([ARENA_SIZE / 2.0, ARENA_SIZE / 2.0], dtype=float)
    center_dir = vec_norm(center - cache["player_pos"])

    best_score = -float("inf")
    best_report = None

    for report in cache["safe_reports"]:
        if report["actual_move"] < step_distance * 0.25:
            continue

        score = 0.0
        score += STUCK_MOVE_WEIGHT * report["actual_move"]
        score += report["score"]
        if center_dir is not None:
            score += STUCK_CENTER_WEIGHT * np.dot(report["dir"], center_dir)

        if score > best_score:
            best_score = score
            best_report = report

    if best_report is None:
        return None

    return report_to_result(best_report, split=False)


def override_split(cache, step_distance):
    """Exact-aim split kill with bounded late-game work.

    v8 optimisation: first collect cheap geometry candidates, then run the
    expensive virus-path / landing-danger / exact-enemy checks only on the best
    few. This keeps the strong multi-split behaviour but avoids cumulative
    timeout when many enemies and own blobs are visible.
    """
    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]
    own_blob_count = cache["own_blob_count"]
    player_radius = cache["player_radius"]

    max_blobs = SPLIT_HUGE_MAX_OWN_BLOBS if player_radius >= SPLIT_HUGE_RADIUS else SPLIT_MAX_OWN_BLOBS
    if own_blob_count > max_blobs:
        return None

    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    if len(blob_locs) == 0 or len(own_locs) == 0:
        return None

    late_fast = own_blob_count > 1 or player_radius >= SPLIT_HUGE_RADIUS
    max_targets = 5 if late_fast else SPLIT_MAX_TARGETS
    max_launchers = 2 if late_fast else SPLIT_MAX_LAUNCHERS
    top_exact = 1 if late_fast else SPLIT_TOP_EXACT_CHECKS
    precheck_limit = 6 if late_fast else SPLIT_PRECHECK_LIMIT

    max_split_radius = float(np.max(own_rads)) / np.sqrt(2.0)
    target_ok = blob_rads < max_split_radius / SPLIT_EAT_RATIO
    if not np.any(target_ok):
        return None

    target_indices = np.where(target_ok)[0]
    target_order = target_indices[np.argsort(blob_rads[target_indices])[::-1]][:max_targets]

    cheap_candidates = []

    for target_idx in target_order:
        target_pos = blob_locs[target_idx]
        target_rad = blob_rads[target_idx]

        launcher_dists = np.sum((own_locs - target_pos) ** 2, axis=1)
        launcher_order = np.argsort(launcher_dists)[:max_launchers]

        for launcher_i in launcher_order:
            launcher_pos = own_locs[launcher_i]
            launcher_radius = own_rads[launcher_i]

            if launcher_radius < SPLIT_MIN_RADIUS:
                continue

            if target_rad <= launcher_radius * SPLIT_TARGET_MIN_RADIUS_MULT:
                continue

            candidate = split_candidate(
                launcher_pos,
                launcher_radius,
                target_pos,
                target_rad,
            )
            if candidate is None:
                continue

            split_radius = candidate["split_radius"]
            if split_radius <= target_rad * SPLIT_EAT_RATIO:
                continue

            split_dir = candidate["dir"]
            score = 0.0
            score += target_rad * 4.0
            score -= candidate["dist"] * 0.25
            score += candidate["reach_margin"] * 0.35
            score -= own_blob_count * 0.75

            cheap_candidates.append((
                score,
                float(split_dir[0]),
                float(split_dir[1]),
                launcher_pos,
                float(launcher_radius),
                float(split_radius),
                candidate["split_landing"],
                int(target_idx),
            ))

            # Keep cheap list bounded throughout the loop.
            if len(cheap_candidates) > precheck_limit * 2:
                cheap_candidates.sort(key=lambda item: item[0], reverse=True)
                cheap_candidates = cheap_candidates[:precheck_limit]

    if not cheap_candidates:
        return None

    cheap_candidates.sort(key=lambda item: item[0], reverse=True)

    checked_exact = 0
    for _, dx, dy, launcher_pos, launcher_radius, split_radius, split_landing, target_idx in cheap_candidates[:precheck_limit]:
        if not split_path_safe(cache, launcher_pos, split_radius, split_landing):
            continue

        if not split_landing_enemy_safe(
            cache,
            split_landing,
            split_radius,
            launcher_radius,
            target_idx,
        ):
            continue

        # Exact enemy safety is the most expensive finalist check; bound it.
        enemy = score_enemy_threat(cache, dx, dy, step_distance)
        checked_exact += 1
        if enemy["safe"]:
            return dx, dy, True

        if checked_exact >= top_exact:
            break

    return None

def override_virus(cache, step_distance):
    global VIRUS_TARGET_KEY, VIRUS_TARGET_TICKS

    if cache["own_blob_count"] != 1:
        VIRUS_TARGET_KEY = None
        VIRUS_TARGET_TICKS = 0
        return None

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    candidates = []
    for key, info in VIRUS_MEMORY.items():
        virus_pos = info["pos"]
        virus_rad = info["radius"]

        if player_radius <= virus_rad * VIRUS_FARM_EAT_RATIO:
            continue

        dist = np.linalg.norm(virus_pos - player_pos)
        candidates.append((dist, key, virus_pos, virus_rad))

    if not candidates:
        VIRUS_TARGET_KEY = None
        VIRUS_TARGET_TICKS = 0
        return None

    candidates.sort(key=lambda item: item[0])
    if len(candidates) > VIRUS_MAX_CANDIDATES:
        candidates = candidates[:VIRUS_MAX_CANDIDATES]

    visible_keys = {virus_key(pos) for pos in cache["virus_locs"]}
    reached_dist = max(1.0, player_radius * VIRUS_MEMORY_REACHED_DIST_MULT)

    # Lock the same virus briefly. This prevents left/right target swapping and
    # the visible-virus wobble that was happening near virus clusters.
    locked = None
    if VIRUS_TARGET_KEY is not None and VIRUS_TARGET_TICKS < VIRUS_LOCK_TICKS:
        for item in candidates:
            if item[1] == VIRUS_TARGET_KEY:
                locked = item
                break

    if locked is not None:
        best_dist = candidates[0][0]
        # Only switch away if another virus is dramatically closer.
        if best_dist < locked[0] * VIRUS_SWITCH_DIST_MULT:
            ordered = candidates
        else:
            ordered = [locked] + [item for item in candidates if item[1] != VIRUS_TARGET_KEY]
    else:
        ordered = candidates

    for dist, key, virus_pos, virus_rad in ordered:
        is_visible = key in visible_keys

        if dist < reached_dist and not is_visible:
            if key in VIRUS_MEMORY:
                del VIRUS_MEMORY[key]
            if VIRUS_TARGET_KEY == key:
                VIRUS_TARGET_KEY = None
                VIRUS_TARGET_TICKS = 0
            continue

        if is_visible and not virus_enemy_safe(cache, virus_pos, virus_rad):
            continue

        direction = vec_norm(virus_pos - player_pos)
        if direction is None:
            continue

        if is_visible:
            # Important: visible virus farming needs exact aiming. Snapping to
            # the nearest cached 16-way report caused oscillation around viruses.
            if not exact_move_safe(
                cache,
                direction,
                step_distance,
                check_virus=False,
                enemy_only=True,
            ):
                continue

            if VIRUS_TARGET_KEY == key:
                VIRUS_TARGET_TICKS += 1
            else:
                VIRUS_TARGET_KEY = key
                VIRUS_TARGET_TICKS = 0

            return float(direction[0]), float(direction[1]), False

        # Travelling toward remembered unseen virus: stay conservative and keep
        # using cached safe movement, because we are not intentionally popping yet.
        report = report_towards(
            cache,
            virus_pos - player_pos,
            reports=cache["safe_reports"],
            danger_scale=0.001,
        )

        if report is None:
            continue

        if VIRUS_TARGET_KEY == key:
            VIRUS_TARGET_TICKS += 1
        else:
            VIRUS_TARGET_KEY = key
            VIRUS_TARGET_TICKS = 0

        return report_to_result(report, split=False)

    return None


def override_chase(cache, step_distance):
    global CHASE_TARGET_KEY, CHASE_TARGET_TICKS, CHASE_LOST_TICKS

    if cache["own_blob_count"] != 1:
        CHASE_TARGET_KEY = None
        CHASE_TARGET_TICKS = 0
        CHASE_LOST_TICKS = 0
        return None

    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    blob_player_ids = cache["blob_player_ids"]

    if len(blob_locs) == 0:
        CHASE_TARGET_KEY = None
        CHASE_TARGET_TICKS = 0
        CHASE_LOST_TICKS = 0
        return None

    edge_dists = cache["enemy_edge_dists"]
    edible = cache["enemy_edible"]
    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]
    blob_vels = cache.get("blob_vels")

    in_range = edge_dists < player_radius * CHASE_RANGE_MULT
    candidates = edible & in_range

    if not np.any(candidates):
        CHASE_TARGET_KEY = None
        CHASE_TARGET_TICKS = 0
        CHASE_LOST_TICKS = 0
        return None

    candidate_indices = np.where(candidates)[0]

    if CHASE_TARGET_KEY is not None and CHASE_TARGET_TICKS < CHASE_LOCK_TICKS:
        locked = [
            idx for idx in candidate_indices
            if int(blob_player_ids[idx]) == CHASE_TARGET_KEY
        ]
        if locked:
            candidate_indices = np.array(locked, dtype=int)

    best_score = -float("inf")
    best_report = None
    best_target_idx = None
    best_closing = 0.0
    best_wall = False
    best_aim_point = None
    best_exact_direct = False

    for idx in candidate_indices:
        enemy_vel = blob_vels[idx] if blob_vels is not None and len(blob_vels) > idx else np.array([0.0, 0.0], dtype=float)
        enemy_speed = np.linalg.norm(enemy_vel)

        # Dynamic prediction: lead more when far away, less when already close.
        lead_ticks = np.clip(
            edge_dists[idx] / max(step_distance + enemy_speed, 1.0),
            1.0,
            CHASE_LEAD_TICKS,
        )
        predicted_pos = blob_locs[idx] + enemy_vel * lead_ticks
        predicted_pos[0] = np.clip(predicted_pos[0], 0.0, ARENA_SIZE)
        predicted_pos[1] = np.clip(predicted_pos[1], 0.0, ARENA_SIZE)

        current_edge = edge_dists[idx]
        near_wall = enemy_near_wall(predicted_pos)
        catchable_now = current_edge < max(player_radius * CHASE_DIRECT_EDGE_MULT, step_distance * 1.25)

        block_point, centre_dir = chase_block_point(
            predicted_pos,
            blob_rads[idx],
            player_radius,
            enemy_vel,
        )

        direct_finish = near_wall or catchable_now
        aim_point = predicted_pos if direct_finish else block_point
        direct_dir = vec_norm(predicted_pos - player_pos)

        for report in cache["safe_reports"]:
            future_pos = np.array([report["future_x"], report["future_y"]], dtype=float)
            future_edge = np.linalg.norm(predicted_pos - future_pos) - player_radius - blob_rads[idx]
            closing = current_edge - future_edge

            score = 0.0
            score += CHASE_SIZE_WEIGHT * blob_rads[idx]
            score -= max(current_edge, 0.0) * 0.20
            score += 0.0002 * report["score"]
            score += CHASE_STICKINESS_WEIGHT * np.dot(report["dir"], LAST_DIRECTION)

            if direct_finish:
                # If we probably can catch them, be much more direct/aggressive.
                score += CHASE_CLOSE_WEIGHT * 1.35 * closing
                if direct_dir is not None:
                    score += 120.0 * np.dot(report["dir"], direct_dir)
                if near_wall:
                    score += 18.0
            else:
                # Open space: get to the centre-side blocking point so they are
                # forced outward instead of circling around us forever.
                block_dist = np.linalg.norm(block_point - future_pos)
                score += 0.75 * CHASE_CLOSE_WEIGHT * closing
                score -= CHASE_BLOCK_DIST_WEIGHT * block_dist

                if centre_dir is not None:
                    future_from_enemy = vec_norm(future_pos - predicted_pos)
                    if future_from_enemy is not None:
                        centre_side = np.dot(future_from_enemy, centre_dir)
                        score += CHASE_CENTER_SIDE_WEIGHT * centre_side

            if CHASE_TARGET_KEY == int(blob_player_ids[idx]):
                score += 12.0

            if score > best_score:
                best_score = score
                best_report = report
                best_target_idx = idx
                best_closing = closing
                best_wall = near_wall
                best_aim_point = aim_point
                best_exact_direct = direct_finish

    if best_report is None or best_target_idx is None:
        return None

    # Prefer exact aiming toward the chosen intercept/block point, but only if
    # the exact vector is safe. The cached report remains a safe fallback.
    exact_dir = vec_norm(best_aim_point - player_pos) if best_aim_point is not None else None
    if exact_dir is not None:
        if exact_move_safe(cache, exact_dir, step_distance, check_virus=True, enemy_only=False):
            chosen_result = (float(exact_dir[0]), float(exact_dir[1]), False)
        else:
            chosen_result = report_to_result(best_report, split=False)
    else:
        chosen_result = report_to_result(best_report, split=False)

    # Do not give up too quickly while attempting a trap. Trapping may briefly
    # reduce direct closing distance but still improves position.
    if best_closing < CHASE_MIN_CLOSING_RATE and not best_wall and best_exact_direct:
        CHASE_LOST_TICKS += 1
    else:
        CHASE_LOST_TICKS = 0

    if CHASE_LOST_TICKS >= CHASE_LOST_LIMIT:
        CHASE_TARGET_KEY = None
        CHASE_TARGET_TICKS = 0
        CHASE_LOST_TICKS = 0
        return None

    chosen_key = int(blob_player_ids[best_target_idx])
    if CHASE_TARGET_KEY == chosen_key:
        CHASE_TARGET_TICKS += 1
    else:
        CHASE_TARGET_KEY = chosen_key
        CHASE_TARGET_TICKS = 0

    return chosen_result

def override_food_roam(cache, step_distance):
    global ROAM_DIRECTION, ROAM_TICKS

    food_candidates = food_ranked_targets(cache)

    for _, dx, dy, key in food_candidates:
        target_vec = np.array([dx, dy], dtype=float)
        report = report_towards(cache, target_vec, reports=cache["safe_reports"])

        if report is None:
            food_blacklist(key)
            continue

        return report_to_result(report, split=False)

    center = np.array([ARENA_SIZE / 2.0, ARENA_SIZE / 2.0], dtype=float)
    center_dir = vec_norm(center - cache["player_pos"])

    if ROAM_DIRECTION is not None and ROAM_TICKS < ROAM_LOCK_TICKS:
        report = report_for_direction(cache, ROAM_DIRECTION)
        if report is not None and report["safe"] and report["actual_move"] >= step_distance * 0.20:
            ROAM_TICKS += 1
            return report_to_result(report, split=False)

    best_score = -float("inf")
    best_report = None

    for report in cache["safe_reports"]:
        score = 0.0
        score += report["score"]
        score += ROAM_MOVE_WEIGHT * report["actual_move"]
        score += ROAM_STICKINESS_WEIGHT * np.dot(report["dir"], LAST_DIRECTION)

        if center_dir is not None:
            score += ROAM_CENTER_WEIGHT * np.dot(report["dir"], center_dir)

        if score > best_score:
            best_score = score
            best_report = report

    if best_report is None:
        ROAM_DIRECTION = None
        ROAM_TICKS = 0
        return None

    ROAM_DIRECTION = np.array(best_report["dir"], dtype=float)
    ROAM_TICKS = 0

    return report_to_result(best_report, split=False)


def override_fallback(cache, step_distance):
    reports = cache["safe_reports"]

    if not reports:
        fallback = vec_norm(LAST_DIRECTION)
        if fallback is None:
            return 1.0, 0.0, False
        return float(fallback[0]), float(fallback[1]), False

    best_score = -float("inf")
    best_report = None

    for report in reports:
        score = 0.0
        score += report["score"]
        score += 0.30 * score_food_position(cache, report["future_x"], report["future_y"])
        # Enemy danger is already represented inside report["score"]. Avoid
        # another per-direction enemy scan in the final fallback.
        score += TURN_WEIGHT * np.dot(report["dir"], LAST_DIRECTION)

        if score > best_score:
            best_score = score
            best_report = report

    return report_to_result(best_report, split=False)


# =========================
# Direction choice
# =========================

def choose_direction(game: Game):
    global LAST_DIRECTION

    cache = cache_frame(game)
    step_distance = cache["step_distance"]

    overrides = [
        override_escape,
        override_unstuck,
        override_split,
        override_chase,
        override_virus,
        override_food_roam,
        override_fallback,
    ]

    for override in overrides:
        result = override(cache, step_distance)
        if result is None:
            continue

        dx, dy, should_split = result
        final_direction = vec_norm(np.array([dx, dy], dtype=float))

        if final_direction is None:
            final_direction = LAST_DIRECTION

        LAST_DIRECTION = final_direction

        return (
            float(final_direction[0]),
            float(final_direction[1]),
            bool(should_split),
            cache,
        )

    return 1.0, 0.0, False, cache


# =========================
# Main loop
# =========================

def main() -> None:
    game = Game()

    while True:
        query = game.get_next_query()

        match query:
            case QueryMovePlayer():
                try:
                    dx, dy, should_split, cache = choose_direction(game)

                    game.send_move(
                        MovePlayer(
                            player_id=cache["player_id"],
                            direction=DirectionModel(
                                x=float(dx),
                                y=float(dy),
                            ),
                            split=bool(should_split),
                        )
                    )

                except Exception:
                    game.send_move(
                        MovePlayer(
                            player_id=game.state.me.player_id,
                            direction=DirectionModel(
                                x=1.0,
                                y=0.0,
                            ),
                            split=False,
                        )
                    )

            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()
