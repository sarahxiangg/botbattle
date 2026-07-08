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

ARENA_SIZE = 60.0
OFF_MAP_PENALTY = 1_000_000_000.0

# Performance
MAX_FOOD_CONSIDERED = 60
MAX_BLOBS_CONSIDERED = 40
MAX_VIRUSES_CONSIDERED = 999

# Movement / fallback beam
LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)
SMOOTHING = 0.0
STEP_DISTANCE_MULT = 1.5
ROLLOUT_STEPS = 1
ROLLOUT_DISCOUNT = 0.7
BEAM_WIDTH = 1
TURN_PENALTY_WEIGHT = 80.0

# Food farming
FOOD_RUSH_MAX_RADIUS = 2.2
FOOD_TARGET_REACHED_DIST_MULT = 0.40
FOOD_TARGET_REACHED_MIN = 0.35

FOOD_TARGET_KEY = None
FOOD_TARGET_TICKS = 0
FOOD_TARGET_NO_PROGRESS = 0
FOOD_TARGET_LAST_DIST = None
FOOD_BLACKLIST = {}

FOOD_TARGET_MAX_TICKS = 10
FOOD_NO_PROGRESS_LIMIT = 4
FOOD_BLACKLIST_TICKS = 25
FOOD_CORNER_MARGIN = 0.45

FOOD_CLUSTER_RADIUS = 4.0
FOOD_CLUSTER_WEIGHT = 0.55
FOOD_DISTANCE_POWER = 1.15
FOOD_MIN_TARGET_DIST = 0.15

# Danger override
DANGER_OVERRIDE_RANGE_MULT = 9.0
DANGER_DIRECT_RATIO = 1.06
DANGER_SPLIT_RATIO = 1.08
DANGER_SPLIT_RANGE_MULT = 5.0
DANGER_OVERRIDE_WEIGHT = 120000.0
DANGER_DISTANCE_POWER = 1.2
DANGER_WALL_PUSH_PENALTY = 50000.0
DANGER_MOVE_REWARD = 5000.0

# Normal scoring
FOOD_SCORE_WEIGHT = 900.0
ENEMY_DANGER_WEIGHT = 14000.0
ENEMY_HUNT_WEIGHT = 2500.0
EAT_RATIO = 1.12

# Virus safety / farming
VIRUS_DANGER_WEIGHT = 12000.0
VIRUS_HARD_BUFFER_MULT = 0.65
VIRUS_PATH_BUFFER_MULT = 0.65

VIRUS_FARM_MIN_RADIUS = 0.0
VIRUS_FARM_EAT_RATIO = 1.10
VIRUS_FARM_ENEMY_RANGE_MULT = 8.0
VIRUS_FARM_PIECE_RADIUS_MULT = 0.45

# Close kill / split
CHASE_RANGE_MULT = 5.0
CLOSE_KILL_RANGE_MULT = 1.5

SPLIT_MIN_RADIUS = 1.45
SPLIT_EAT_RATIO = 1.15
SPLIT_RANGE_MULT = 2.6
SPLIT_RANGE_SAFETY_MULT = 0.75
SPLIT_ALIGNMENT_MIN = 0.92
SPLIT_TARGET_MIN_RADIUS_MULT = 0.28
SPLIT_VIRUS_BUFFER_MULT = 1.4
POST_SPLIT_DANGER_RANGE_MULT = 6.0

# Stuck fallback
WALL_BUFFER = 0.25
WALL_PENALTY_WEIGHT = 300.0

STUCK_TICK_LIMIT = 4
STUCK_MOVE_EPS_MULT = 0.04
STUCK_CENTER_WEIGHT = 600.0
STUCK_MOVE_WEIGHT = 2000.0

LAST_POSITION = None
STUCK_TICKS = 0


# =========================
# Basic utilities
# =========================

def normalize(vec):
    norm = np.linalg.norm(vec)

    if norm <= 1e-9:
        return None

    return vec / norm


def clamp_point_for_radius(x, y, radius):
    min_pos = radius + WALL_BUFFER
    max_pos = ARENA_SIZE - radius - WALL_BUFFER

    if min_pos > max_pos:
        min_pos = radius
        max_pos = ARENA_SIZE - radius

    x = min(max(x, min_pos), max_pos)
    y = min(max(y, min_pos), max_pos)

    return x, y


def clamp_position(cache, x, y):
    return clamp_point_for_radius(x, y, cache["player_radius"])


def cap_nearest(locs, max_items, player_pos, *arrays):
    if len(locs) <= max_items:
        return (locs, *arrays)

    dists = np.sum((locs - player_pos) ** 2, axis=1)
    keep = np.argpartition(dists, max_items - 1)[:max_items]

    result = [locs[keep]]

    for arr in arrays:
        result.append(arr[keep])

    return tuple(result)


# =========================
# Cache
# =========================

def build_cache(game):
    me = game.state.me

    player_pos = np.array([me.x, me.y], dtype=float)
    player_radius = float(me.radius)
    player_id = me.player_id

    own_blobs = list(me.blobs.values())

    if own_blobs:
        own_locs = np.array([blob.pos for blob in own_blobs], dtype=float)
        own_rads = np.array([blob.radius for blob in own_blobs], dtype=float)
        own_blob_count = len(own_blobs)
    else:
        own_locs = np.array([[me.x, me.y]], dtype=float)
        own_rads = np.array([me.radius], dtype=float)
        own_blob_count = 1

    # ---------- Food ----------
    foods = game.state.visible_food

    if foods:
        food_locs = np.array([food.pos for food in foods], dtype=float)

        if len(food_locs) > MAX_FOOD_CONSIDERED:
            food_locs, = cap_nearest(
                food_locs,
                MAX_FOOD_CONSIDERED,
                player_pos,
            )
    else:
        food_locs = np.empty((0, 2), dtype=float)

    # ---------- Enemies ----------
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

    # ---------- Viruses ----------
    viruses = game.state.visible_viruses

    if viruses:
        virus_locs = np.array([virus.pos for virus in viruses], dtype=float)
        virus_rads = np.array([virus.radius for virus in viruses], dtype=float)
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

        "virus_farm_mode": False,
    }


# =========================
# Virus safety
# =========================

def segment_virus_score(cache, start_pos, end_pos, radius, buffer_mult):
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


def point_virus_score(cache, x, y, radius, buffer_mult):
    pos = np.array([x, y], dtype=float)

    return segment_virus_score(
        cache,
        pos,
        pos,
        radius,
        buffer_mult,
    )


def movement_virus_score(cache, start_x, start_y, end_x, end_y):
    start_pos = np.array([start_x, start_y], dtype=float)
    end_pos = np.array([end_x, end_y], dtype=float)

    return segment_virus_score(
        cache,
        start_pos,
        end_pos,
        cache["player_radius"],
        VIRUS_PATH_BUFFER_MULT,
    )


def own_blob_virus_score(cache, dx, dy, step_distance):
    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]

    delta = np.array([dx * step_distance, dy * step_distance], dtype=float)
    total_score = 0.0

    for own_pos, own_rad in zip(own_locs, own_rads):
        end_pos = own_pos + delta
        end_x, end_y = clamp_point_for_radius(end_pos[0], end_pos[1], own_rad)
        end_pos = np.array([end_x, end_y], dtype=float)

        score = segment_virus_score(
            cache,
            own_pos,
            end_pos,
            own_rad,
            VIRUS_PATH_BUFFER_MULT,
        )

        if score <= -OFF_MAP_PENALTY / 2:
            return -OFF_MAP_PENALTY

        total_score += score

    return total_score


# =========================
# Enemy danger
# =========================

def enemy_danger_exists(cache):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return False

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]
    player_radius = cache["player_radius"]

    danger_size = np.maximum(blob_rads, merged_rads)
    enemy_split_rads = blob_rads / np.sqrt(2.0)

    for own_pos, own_rad in zip(own_locs, own_rads):
        dists = np.linalg.norm(blob_locs - own_pos, axis=1)
        edge_dists = dists - own_rad - blob_rads

        direct_danger = danger_size > own_rad * DANGER_DIRECT_RATIO
        direct_near = edge_dists < player_radius * DANGER_OVERRIDE_RANGE_MULT

        split_danger = enemy_split_rads > own_rad * DANGER_SPLIT_RATIO
        split_near = edge_dists < blob_rads * DANGER_SPLIT_RANGE_MULT

        if np.any((direct_danger & direct_near) | (split_danger & split_near)):
            return True

    return False


def direction_enemy_danger_score(cache, dx, dy, step_distance):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return 0.0

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]

    danger_size = np.maximum(blob_rads, merged_rads)
    enemy_split_rads = blob_rads / np.sqrt(2.0)

    delta = np.array([dx * step_distance, dy * step_distance], dtype=float)

    score = 0.0

    for own_pos, own_rad in zip(own_locs, own_rads):
        future_pos = own_pos + delta
        future_x, future_y = clamp_point_for_radius(
            future_pos[0],
            future_pos[1],
            own_rad,
        )
        future_pos = np.array([future_x, future_y], dtype=float)

        dists = np.linalg.norm(blob_locs - future_pos, axis=1)
        edge_dists = dists - own_rad - blob_rads
        edge_dists[edge_dists < 1.0] = 1.0

        direct_danger = danger_size > own_rad * DANGER_DIRECT_RATIO

        split_danger = enemy_split_rads > own_rad * DANGER_SPLIT_RATIO
        split_near = edge_dists < blob_rads * DANGER_SPLIT_RANGE_MULT

        dangerous = direct_danger | (split_danger & split_near)

        danger_scores = (
            DANGER_OVERRIDE_WEIGHT *
            (danger_size / own_rad) /
            (edge_dists ** DANGER_DISTANCE_POWER)
        )

        score -= np.sum(danger_scores[dangerous])

    return float(score)


def move_safe_from_enemies(cache, dx, dy, step_distance):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return True

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]

    danger_size = np.maximum(blob_rads, merged_rads)
    enemy_split_rads = blob_rads / np.sqrt(2.0)

    delta = np.array([dx * step_distance, dy * step_distance], dtype=float)

    for own_pos, own_rad in zip(own_locs, own_rads):
        future_pos = own_pos + delta
        future_x, future_y = clamp_point_for_radius(
            future_pos[0],
            future_pos[1],
            own_rad,
        )
        future_pos = np.array([future_x, future_y], dtype=float)

        dists = np.linalg.norm(blob_locs - future_pos, axis=1)
        edge_dists = dists - own_rad - blob_rads

        direct_danger = danger_size > own_rad * DANGER_DIRECT_RATIO
        direct_too_close = edge_dists < own_rad * 2.5

        split_danger = enemy_split_rads > own_rad * DANGER_SPLIT_RATIO
        split_near = edge_dists < blob_rads * DANGER_SPLIT_RANGE_MULT

        if np.any((direct_danger & direct_too_close) | (split_danger & split_near)):
            return False

    return True


# =========================
# Override 1: danger avoidance
# =========================

def danger_avoidance_override(cache, step_distance):
    if not enemy_danger_exists(cache):
        return None

    best_score = -float("inf")
    best_direction = None

    for direction in DIRECTIONS:
        dx, dy = direction

        virus_score = own_blob_virus_score(cache, dx, dy, step_distance)

        if virus_score <= -OFF_MAP_PENALTY / 2:
            continue

        future_x = cache["player_x"] + dx * step_distance
        future_y = cache["player_y"] + dy * step_distance
        future_x, future_y = clamp_position(cache, future_x, future_y)

        actual_move = np.linalg.norm(
            np.array([
                future_x - cache["player_x"],
                future_y - cache["player_y"],
            ], dtype=float)
        )

        score = virus_score
        score += direction_enemy_danger_score(cache, dx, dy, step_distance)

        if actual_move < step_distance * 0.20:
            score -= DANGER_WALL_PUSH_PENALTY
        else:
            score += DANGER_MOVE_REWARD * actual_move

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        for direction in DIRECTIONS:
            dx, dy = direction
            score = direction_enemy_danger_score(cache, dx, dy, step_distance)

            if score > best_score:
                best_score = score
                best_direction = direction

    if best_direction is None:
        return None

    return float(best_direction[0]), float(best_direction[1]), False


# =========================
# Override 2: food farming
# =========================

def food_key(food_pos):
    return (round(float(food_pos[0]), 1), round(float(food_pos[1]), 1))


def update_food_blacklist():
    global FOOD_BLACKLIST

    expired = []

    for key in FOOD_BLACKLIST:
        FOOD_BLACKLIST[key] -= 1

        if FOOD_BLACKLIST[key] <= 0:
            expired.append(key)

    for key in expired:
        del FOOD_BLACKLIST[key]


def blacklist_food_key(key):
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


def food_in_unreachable_corner(food_pos):
    x, y = food_pos

    near_left = x < FOOD_CORNER_MARGIN
    near_right = ARENA_SIZE - x < FOOD_CORNER_MARGIN
    near_bottom = y < FOOD_CORNER_MARGIN
    near_top = ARENA_SIZE - y < FOOD_CORNER_MARGIN

    return (near_left or near_right) and (near_bottom or near_top)


def ranked_food_directions(cache):
    global FOOD_TARGET_KEY, FOOD_TARGET_TICKS
    global FOOD_TARGET_NO_PROGRESS, FOOD_TARGET_LAST_DIST

    update_food_blacklist()

    food_locs = cache["food_locs"]

    if len(food_locs) == 0:
        FOOD_TARGET_KEY = None
        FOOD_TARGET_TICKS = 0
        FOOD_TARGET_NO_PROGRESS = 0
        FOOD_TARGET_LAST_DIST = None
        return []

    player_pos = cache["player_pos"]

    food_keys = [food_key(food_pos) for food_pos in food_locs]

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

    vectors = food_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)

    reached_dist = max(
        FOOD_TARGET_REACHED_MIN,
        cache["player_radius"] * FOOD_TARGET_REACHED_DIST_MULT,
    )

    valid = dists > FOOD_MIN_TARGET_DIST

    for i, key in enumerate(food_keys):
        if key in FOOD_BLACKLIST:
            valid[i] = False

        if food_in_unreachable_corner(food_locs[i]):
            valid[i] = False

    if not np.any(valid):
        return []

    # Progress tracking for current target.
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
            current_dist < reached_dist or
            FOOD_TARGET_TICKS >= FOOD_TARGET_MAX_TICKS or
            FOOD_TARGET_NO_PROGRESS >= FOOD_NO_PROGRESS_LIMIT
        ):
            blacklist_food_key(FOOD_TARGET_KEY)
            current_idx = None

    safe_dists = dists.copy()
    safe_dists[safe_dists < 1.0] = 1.0

    pairwise = np.linalg.norm(
        food_locs[:, None, :] - food_locs[None, :, :],
        axis=2,
    )

    cluster_density = np.exp(
        -((pairwise / FOOD_CLUSTER_RADIUS) ** 2)
    ).sum(axis=1) - 1.0

    scores = (
        (1.0 + FOOD_CLUSTER_WEIGHT * cluster_density) /
        (safe_dists ** FOOD_DISTANCE_POWER)
    )

    scores[~valid] = -1.0

    if current_idx is not None and scores[current_idx] > 0:
        scores[current_idx] *= 1.25

    candidate_indices = np.argsort(scores)[::-1]

    candidates = []

    for idx in candidate_indices:
        if scores[idx] <= 0:
            break

        direction = normalize(vectors[idx])

        if direction is None:
            continue

        candidates.append(
            (
                float(scores[idx]),
                float(direction[0]),
                float(direction[1]),
                food_keys[idx],
            )
        )

    if candidates and FOOD_TARGET_KEY is None:
        FOOD_TARGET_KEY = candidates[0][3]
        FOOD_TARGET_TICKS = 0
        FOOD_TARGET_NO_PROGRESS = 0
        FOOD_TARGET_LAST_DIST = None

    return candidates


def food_farming_override(cache, step_distance):
    if cache["own_blob_count"] == 1 and cache["player_radius"] >= FOOD_RUSH_MAX_RADIUS:
        return None

    food_candidates = ranked_food_directions(cache)

    if not food_candidates:
        return None

    for _, dx, dy, key in food_candidates:
        future_x = cache["player_x"] + dx * step_distance
        future_y = cache["player_y"] + dy * step_distance
        future_x, future_y = clamp_position(cache, future_x, future_y)

        actual_move = np.linalg.norm(
            np.array([
                future_x - cache["player_x"],
                future_y - cache["player_y"],
            ], dtype=float)
        )

        if actual_move < step_distance * 0.20:
            blacklist_food_key(key)
            continue

        if movement_virus_score(
            cache,
            cache["player_x"],
            cache["player_y"],
            future_x,
            future_y,
        ) <= -OFF_MAP_PENALTY / 2:
            blacklist_food_key(key)
            continue

        if own_blob_virus_score(cache, dx, dy, step_distance) <= -OFF_MAP_PENALTY / 2:
            blacklist_food_key(key)
            continue

        if not move_safe_from_enemies(cache, dx, dy, step_distance):
            blacklist_food_key(key)
            continue

        return dx, dy, False

    return None


# =========================
# Override 3: close kill / split
# =========================

def post_split_enemy_safe(cache, split_radius):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return True

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    danger_size = np.maximum(blob_rads, merged_rads)

    dists = np.linalg.norm(blob_locs - player_pos, axis=1)
    edge_dists = dists - split_radius - blob_rads

    dangerous = danger_size > split_radius * 1.08
    nearby = edge_dists < player_radius * POST_SPLIT_DANGER_RANGE_MULT

    return not bool(np.any(dangerous & nearby))


def split_path_safe(cache, split_radius, split_landing):
    score = segment_virus_score(
        cache,
        cache["player_pos"],
        split_landing,
        split_radius,
        SPLIT_VIRUS_BUFFER_MULT,
    )

    return score > -OFF_MAP_PENALTY / 2


def close_kill_override(cache, step_distance):
    if cache["own_blob_count"] != 1:
        return None

    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) == 0:
        return None

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    vectors = blob_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)

    safe_dists = dists.copy()
    safe_dists[safe_dists < 1.0] = 1.0

    dirs = vectors / safe_dists.reshape(-1, 1)

    edge_dists = dists - player_radius - blob_rads

    can_eat = player_radius > blob_rads * EAT_RATIO

    if not np.any(can_eat):
        return None

    # =========================
    # Conservative split finisher
    # =========================

    split_radius = player_radius / np.sqrt(2.0)
    split_range = player_radius * SPLIT_RANGE_MULT

    useful_target = blob_rads > player_radius * SPLIT_TARGET_MIN_RADIUS_MULT

    hit_reach = split_range * SPLIT_RANGE_SAFETY_MULT + split_radius + blob_rads

    split_candidates = (
        (player_radius >= SPLIT_MIN_RADIUS) &
        (split_radius > blob_rads * SPLIT_EAT_RATIO) &
        (dists <= hit_reach) &
        useful_target &
        can_eat
    )

    if np.any(split_candidates) and post_split_enemy_safe(cache, split_radius):
        scores = (
            blob_rads * 3.0
            - dists * 0.25
        )

        scores[~split_candidates] = -1.0

        candidate_indices = np.argsort(scores)[::-1]

        for idx in candidate_indices:
            if scores[idx] <= 0:
                break

            split_dir = dirs[idx]
            split_landing = player_pos + split_dir * split_range

            if not split_path_safe(cache, split_radius, split_landing):
                continue

            dx, dy = float(split_dir[0]), float(split_dir[1])

            if not move_safe_from_enemies(cache, dx, dy, step_distance):
                continue

            return dx, dy, True

    # =========================
    # Chase instead of risky split
    # =========================

    chase_candidates = can_eat & (edge_dists < player_radius * CHASE_RANGE_MULT)

    if not np.any(chase_candidates):
        return None

    scores = blob_rads / safe_dists
    scores[~chase_candidates] = -1.0

    candidate_indices = np.argsort(scores)[::-1]

    for idx in candidate_indices:
        if scores[idx] <= 0:
            break

        chase_dir = dirs[idx]
        dx, dy = float(chase_dir[0]), float(chase_dir[1])

        future_x = cache["player_x"] + dx * step_distance
        future_y = cache["player_y"] + dy * step_distance
        future_x, future_y = clamp_position(cache, future_x, future_y)

        if movement_virus_score(
            cache,
            cache["player_x"],
            cache["player_y"],
            future_x,
            future_y,
        ) <= -OFF_MAP_PENALTY / 2:
            continue

        if not move_safe_from_enemies(cache, dx, dy, step_distance):
            continue

        return dx, dy, False

    return None


# =========================
# Override 4: virus farming
# =========================

def virus_farm_enemy_safe(cache, virus_pos, virus_rad):
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

    punish_threat = can_eat_pieces & (close_to_player | close_to_virus)

    return not bool(np.any(punish_threat))


def virus_farming_override(cache, step_distance):
    if cache["own_blob_count"] != 1:
        return None

    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if len(virus_locs) == 0:
        return None

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    vectors = virus_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)

    safe_dists = dists.copy()
    safe_dists[safe_dists < 1.0] = 1.0

    # This is the only real "is virus farming possible?" check.
    can_farm = player_radius > virus_rads * VIRUS_FARM_EAT_RATIO

    if not np.any(can_farm):
        return None

    scores = 1.0 / safe_dists
    scores[~can_farm] = -1.0

    candidate_indices = np.argsort(scores)[::-1]

    for idx in candidate_indices:
        if scores[idx] <= 0:
            break

        virus_pos = virus_locs[idx]
        virus_rad = virus_rads[idx]

        if not virus_farm_enemy_safe(cache, virus_pos, virus_rad):
            continue

        direction = normalize(vectors[idx])

        if direction is None:
            continue

        dx, dy = float(direction[0]), float(direction[1])

        if not move_safe_from_enemies(cache, dx, dy, step_distance):
            continue

        cache["virus_farm_mode"] = True
        return dx, dy, False

    return None


# =========================
# Override wrapper
# =========================

def is_stuck(cache):
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


def unstuck_override(cache, step_distance):
    if not is_stuck(cache):
        return None

    best_score = -float("inf")
    best_direction = None

    player_pos = cache["player_pos"]
    center = np.array([ARENA_SIZE / 2.0, ARENA_SIZE / 2.0], dtype=float)

    center_vec = center - player_pos
    center_dir = normalize(center_vec)

    for direction in DIRECTIONS:
        dx, dy = direction

        if not move_safe_from_enemies(cache, dx, dy, step_distance):
            continue

        future_x = cache["player_x"] + dx * step_distance
        future_y = cache["player_y"] + dy * step_distance
        future_x, future_y = clamp_position(cache, future_x, future_y)

        actual_move = np.linalg.norm(
            np.array(
                [
                    future_x - cache["player_x"],
                    future_y - cache["player_y"],
                ],
                dtype=float,
            )
        )

        if actual_move < step_distance * 0.25:
            continue

        virus_score = movement_virus_score(
            cache,
            cache["player_x"],
            cache["player_y"],
            future_x,
            future_y,
        )

        if virus_score <= -OFF_MAP_PENALTY / 2:
            continue

        own_virus_score = own_blob_virus_score(cache, dx, dy, step_distance)

        if own_virus_score <= -OFF_MAP_PENALTY / 2:
            continue

        score = 0.0
        score += STUCK_MOVE_WEIGHT * actual_move
        score += wall_score(cache, future_x, future_y)
        score += virus_score
        score += own_virus_score

        if center_dir is not None:
            score += STUCK_CENTER_WEIGHT * np.dot(direction, center_dir)

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        return None

    return float(best_direction[0]), float(best_direction[1]), False


def override_direction(cache, step_distance):
    # 1. Most important: avoid enemies at all costs.
    danger_dir = danger_avoidance_override(cache, step_distance)

    if danger_dir is not None:
        return danger_dir

    # 2. If stuck, force a safe move away from walls/corners.
    unstuck_dir = unstuck_override(cache, step_distance)

    if unstuck_dir is not None:
        return unstuck_dir

    # 3. Virus farming.
    virus_dir = virus_farming_override(cache, step_distance)

    if virus_dir is not None:
        return virus_dir


    # 4. Food farming.
    food_dir = food_farming_override(cache, step_distance)

    if food_dir is not None:
        return food_dir

    # 5. Close kill / chase / split.
    kill_dir = close_kill_override(cache, step_distance)

    if kill_dir is not None:
        return kill_dir

    return None


# =========================
# Fallback scoring / beam
# =========================

def wall_score(cache, x, y):
    return 0.0


def food_score(cache, x, y):
    food_locs = cache["food_locs"]

    if len(food_locs) == 0:
        return 0.0

    pos = np.array([x, y], dtype=float)

    dists = np.linalg.norm(food_locs - pos, axis=1)
    dists[dists < 1.0] = 1.0

    return float(np.sum(FOOD_SCORE_WEIGHT / (dists ** 2)))


def enemy_score(cache, x, y):
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
        ENEMY_DANGER_WEIGHT *
        (danger_size / player_radius) /
        (edge_dists ** 1.3)
    )

    hunt_scores = (
        ENEMY_HUNT_WEIGHT *
        (player_radius / blob_rads) /
        (edge_dists ** 2)
    )

    score = 0.0
    score -= np.sum(danger_scores[dangerous])

    if cache["own_blob_count"] == 1:
        score += np.sum(hunt_scores[edible])

    return float(score)


def score_position(cache, x, y):
    score = 0.0
    score += food_score(cache, x, y)
    score += enemy_score(cache, x, y)
    score += point_virus_score(
        cache,
        x,
        y,
        cache["player_radius"],
        VIRUS_HARD_BUFFER_MULT,
    )

    score += wall_score(cache, x, y)

    return score


def beam_search_direction(cache, step_distance):
    best_score = -float("inf")
    best_direction = None

    for direction in DIRECTIONS:
        dx, dy = direction

        if not move_safe_from_enemies(cache, dx, dy, step_distance):
            continue

        future_x = cache["player_x"] + dx * step_distance
        future_y = cache["player_y"] + dy * step_distance
        future_x, future_y = clamp_position(cache, future_x, future_y)

        virus_score = movement_virus_score(
            cache,
            cache["player_x"],
            cache["player_y"],
            future_x,
            future_y,
        )

        if virus_score <= -OFF_MAP_PENALTY / 2:
            continue

        own_virus_score = own_blob_virus_score(cache, dx, dy, step_distance)

        if own_virus_score <= -OFF_MAP_PENALTY / 2:
            continue

        score = score_position(cache, future_x, future_y)
        score += virus_score
        score += own_virus_score

        turn_alignment = np.dot(direction, LAST_DIRECTION)
        score -= TURN_PENALTY_WEIGHT * (1.0 - turn_alignment)

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is not None:
        return float(best_direction[0]), float(best_direction[1]), False

    danger_dir = danger_avoidance_override(cache, step_distance)

    if danger_dir is not None:
        return danger_dir

    return 1.0, 0.0, False


# =========================
# Direction choice
# =========================

def choose_direction(game: Game):
    global LAST_DIRECTION

    cache = build_cache(game)
    step_distance = STEP_DISTANCE_MULT * cache["player_radius"]

    override_dir = override_direction(cache, step_distance)

    if override_dir is not None:
        dx, dy, should_split = override_dir
        LAST_DIRECTION = np.array([dx, dy], dtype=float)
        return dx, dy, should_split, cache

    dx, dy, should_split = beam_search_direction(cache, step_distance)

    final_direction = np.array([dx, dy], dtype=float)
    final_direction = normalize(final_direction)

    if final_direction is None:
        final_direction = LAST_DIRECTION

    LAST_DIRECTION = final_direction

    return float(final_direction[0]), float(final_direction[1]), should_split, cache


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