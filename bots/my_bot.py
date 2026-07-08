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

# Performance caps
MAX_FOOD_CONSIDERED = 45
MAX_BLOBS_CONSIDERED = 35
MAX_VIRUSES_CONSIDERED = 999

# Movement / beam
LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)
SMOOTHING = 0.0
ROLLOUT_STEPS = 2
ROLLOUT_DISCOUNT = 0.7
STEP_DISTANCE_MULT = 1.5
BEAM_WIDTH = 1
TURN_PENALTY_WEIGHT = 120.0

# Food farming
FOOD_RUSH_MAX_RADIUS = 2.2
FOOD_RUSH_DANGER_RANGE_MULT = 3.0
FOOD_DISTANCE_WEIGHT = 8.0
FOOD_CLUSTER_WEIGHT = 0.35

# Enemy danger / hunting
EAT_RATIO = 1.12
ENEMY_DANGER_WEIGHT = 14000.0
ENEMY_HUNT_WEIGHT = 3500.0
DANGER_DISTANCE_POWER = 1.3

SINGLE_ESCAPE_RANGE_MULT = 3.8
SINGLE_ESCAPE_WEIGHT = 22000.0

# Fragmented / post-virus-crash danger
FRAGMENTED_ESCAPE_RANGE_MULT = 8.0
FRAGMENTED_DANGER_RATIO = 1.05
FRAGMENTED_ESCAPE_WEIGHT = 30000.0
FRAGMENTED_DISTANCE_POWER = 1.25

# Virus avoidance
VIRUS_DANGER_WEIGHT = 12000.0
VIRUS_HARD_BUFFER_MULT = 0.65
VIRUS_PATH_BUFFER_MULT = 0.65
SPLIT_VIRUS_BUFFER_MULT = 1.4

# Virus farming
VIRUS_FARM_MIN_RADIUS = 2.2
VIRUS_FARM_EAT_RATIO = 1.10
VIRUS_FARM_ENEMY_RANGE_MULT = 7.0
VIRUS_FARM_PIECE_RADIUS_MULT = 0.45

# Close kill / splitting
CLOSE_KILL_RANGE_MULT = 1.4
SPLIT_EAT_RATIO = 1.15
SPLIT_RANGE_MULT = 3.0
SPLIT_ALIGNMENT_MIN = 0.90
SPLIT_MIN_RADIUS = 1.45
POST_SPLIT_DANGER_RANGE_MULT = 5.0

# Stuck detector
LAST_POSITION = None
STUCK_TICKS = 0
STUCK_TICK_LIMIT = 4
STUCK_MOVE_EPS_MULT = 0.05


# =========================
# Basic utilities
# =========================

def normalize(vec):
    norm = np.linalg.norm(vec)

    if norm <= 1e-9:
        return None

    return vec / norm


def clamp_point_for_radius(x, y, radius):
    x = min(max(x, radius), ARENA_SIZE - radius)
    y = min(max(y, radius), ARENA_SIZE - radius)

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

    # ---------- Enemy blobs ----------
    all_blobs = game.state.visible_blobs
    enemy_blobs = [
        blob for blob in all_blobs
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

def point_virus_score(cache, pos, radius, buffer_mult):
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if len(virus_locs) == 0:
        return 0.0

    dangerous = radius > virus_rads * 1.1

    if not np.any(dangerous):
        return 0.0

    dists = np.linalg.norm(virus_locs - pos, axis=1)
    clearances = dists - radius - virus_rads

    if np.any(dangerous & (clearances < radius * buffer_mult)):
        return -OFF_MAP_PENALTY

    clearances[clearances < 1.0] = 1.0

    return -float(np.sum(VIRUS_DANGER_WEIGHT / (clearances[dangerous] ** 3)))


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

    if len(own_locs) <= 1:
        return 0.0

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

def dangerous_enemy_near(cache, range_mult, ratio=1.10):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return False

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    dists = np.linalg.norm(blob_locs - player_pos, axis=1)
    edge_dists = dists - player_radius - blob_rads

    danger_size = np.maximum(blob_rads, merged_rads)

    dangerous = danger_size > player_radius * ratio
    nearby = edge_dists < player_radius * range_mult

    return bool(np.any(dangerous & nearby))


def enemy_score_single(cache, future_x, future_y, danger_weight, hunt_weight):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return 0.0

    future_pos = np.array([future_x, future_y], dtype=float)
    player_radius = cache["player_radius"]

    dists = np.linalg.norm(blob_locs - future_pos, axis=1)
    edge_dists = dists - player_radius - blob_rads
    edge_dists[edge_dists < 1.0] = 1.0

    danger_size = np.maximum(blob_rads, merged_rads)

    dangerous = danger_size > player_radius * 1.10
    edible = player_radius > blob_rads * EAT_RATIO

    danger_scores = (
        danger_weight *
        (danger_size / player_radius) /
        (edge_dists ** DANGER_DISTANCE_POWER)
    )

    hunt_scores = (
        hunt_weight *
        (player_radius / blob_rads) /
        (edge_dists ** 2)
    )

    score = 0.0
    score -= np.sum(danger_scores[dangerous])
    score += np.sum(hunt_scores[edible])

    return float(score)


def fragmented_threat_exists(cache):
    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]

    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(own_locs) <= 1 or len(blob_locs) == 0:
        return False

    danger_size = np.maximum(blob_rads, merged_rads)
    player_radius = cache["player_radius"]

    for own_pos, own_rad in zip(own_locs, own_rads):
        dists = np.linalg.norm(blob_locs - own_pos, axis=1)
        edge_dists = dists - own_rad - blob_rads

        dangerous = danger_size > own_rad * FRAGMENTED_DANGER_RATIO
        nearby = edge_dists < player_radius * FRAGMENTED_ESCAPE_RANGE_MULT

        if np.any(dangerous & nearby):
            return True

    return False


def fragmented_escape_direction(cache, step_distance):
    if not fragmented_threat_exists(cache):
        return None

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]

    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    danger_size = np.maximum(blob_rads, merged_rads)
    player_radius = cache["player_radius"]

    best_score = -float("inf")
    best_direction = None

    for direction in DIRECTIONS:
        dx, dy = direction
        delta = np.array([dx * step_distance, dy * step_distance], dtype=float)

        virus_score = own_blob_virus_score(cache, dx, dy, step_distance)

        if virus_score <= -OFF_MAP_PENALTY / 2:
            continue

        score = virus_score

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

            dangerous = danger_size > own_rad * FRAGMENTED_DANGER_RATIO
            nearby = edge_dists < player_radius * FRAGMENTED_ESCAPE_RANGE_MULT

            danger_scores = (
                FRAGMENTED_ESCAPE_WEIGHT *
                (danger_size / own_rad) /
                (edge_dists ** FRAGMENTED_DISTANCE_POWER)
            )

            score -= np.sum(danger_scores[dangerous & nearby])

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        return None

    return float(best_direction[0]), float(best_direction[1])


def enemy_escape_direction(cache, step_distance):
    if cache["own_blob_count"] != 1:
        return None

    if not dangerous_enemy_near(cache, SINGLE_ESCAPE_RANGE_MULT, ratio=1.08):
        return None

    best_score = -float("inf")
    best_direction = None

    for direction in DIRECTIONS:
        dx, dy = direction

        future_x = cache["player_x"] + dx * step_distance
        future_y = cache["player_y"] + dy * step_distance
        future_x, future_y = clamp_position(cache, future_x, future_y)

        score = 0.0

        score += enemy_score_single(
            cache,
            future_x,
            future_y,
            danger_weight=SINGLE_ESCAPE_WEIGHT,
            hunt_weight=0.0,
        )

        score += movement_virus_score(
            cache,
            cache["player_x"],
            cache["player_y"],
            future_x,
            future_y,
        )

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        return None

    return float(best_direction[0]), float(best_direction[1])


# =========================
# Food farming
# =========================

def food_score(cache, future_x, future_y, weight):
    food_locs = cache["food_locs"]

    if len(food_locs) == 0:
        return 0.0

    future_pos = np.array([future_x, future_y], dtype=float)

    dists = np.linalg.norm(food_locs - future_pos, axis=1)
    dists[dists < 1.0] = 1.0

    return float(np.sum(weight / (dists ** 2)))


def food_direction(cache):
    food_locs = cache["food_locs"]

    if len(food_locs) == 0:
        return None

    player_pos = cache["player_pos"]

    vectors = food_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)

    if len(dists) == 0:
        return None

    safe_dists = dists.copy()
    safe_dists[safe_dists < 1.0] = 1.0

    if len(food_locs) == 1:
        idx = 0
    else:
        pairwise = np.linalg.norm(
            food_locs[:, None, :] - food_locs[None, :, :],
            axis=2,
        )
        pairwise[pairwise < 1.0] = 1.0

        cluster_density = np.sum(1.0 / (pairwise ** 2), axis=1)

        scores = (
            FOOD_DISTANCE_WEIGHT / safe_dists +
            FOOD_CLUSTER_WEIGHT * cluster_density
        )

        idx = int(np.argmax(scores))

    direction = normalize(vectors[idx])

    if direction is None:
        return None

    return float(direction[0]), float(direction[1])


def food_farm_direction(cache, step_distance):
    if len(cache["food_locs"]) == 0:
        return None

    # If we are one big blob, only hard-rush food in early game.
    if cache["own_blob_count"] == 1 and cache["player_radius"] > FOOD_RUSH_MAX_RADIUS:
        return None

    # If small and in real danger, escaping matters more than farming.
    if cache["own_blob_count"] == 1:
        if dangerous_enemy_near(cache, FOOD_RUSH_DANGER_RANGE_MULT, ratio=1.08):
            return None

    food_dir = food_direction(cache)

    if food_dir is None:
        return None

    dx, dy = food_dir

    future_x = cache["player_x"] + dx * step_distance
    future_y = cache["player_y"] + dy * step_distance
    future_x, future_y = clamp_position(cache, future_x, future_y)

    virus_safety = movement_virus_score(
        cache,
        cache["player_x"],
        cache["player_y"],
        future_x,
        future_y,
    )

    if virus_safety <= -OFF_MAP_PENALTY / 2:
        return None

    own_virus_safety = own_blob_virus_score(cache, dx, dy, step_distance)

    if own_virus_safety <= -OFF_MAP_PENALTY / 2:
        return None

    return dx, dy


# =========================
# Virus farming
# =========================

def virus_farm_target_safe(cache, virus_pos, virus_rad):
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


def virus_farm_direction(cache, step_distance):
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if len(virus_locs) == 0:
        return None

    if cache["own_blob_count"] != 1:
        return None

    player_radius = cache["player_radius"]

    if player_radius < VIRUS_FARM_MIN_RADIUS:
        return None

    player_pos = cache["player_pos"]

    vectors = virus_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)

    safe_dists = dists.copy()
    safe_dists[safe_dists < 1.0] = 1.0

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

        if not virus_farm_target_safe(cache, virus_pos, virus_rad):
            continue

        direction = normalize(vectors[idx])

        if direction is None:
            continue

        return float(direction[0]), float(direction[1])

    return None


# =========================
# Close kill / split
# =========================

def close_kill_direction(cache, step_distance):
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

    edge_dists = dists - player_radius - blob_rads

    can_eat = player_radius > blob_rads * EAT_RATIO
    close = edge_dists < player_radius * CLOSE_KILL_RANGE_MULT

    candidates = can_eat & close

    if not np.any(candidates):
        return None

    scores = blob_rads / safe_dists
    scores[~candidates] = -1.0

    idx = int(np.argmax(scores))

    direction = normalize(vectors[idx])

    if direction is None:
        return None

    dx, dy = float(direction[0]), float(direction[1])

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
        return None

    return dx, dy


def split_penalty(cache, split_radius):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return False

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    danger_size = np.maximum(blob_rads, merged_rads)

    dists = np.linalg.norm(blob_locs - player_pos, axis=1)
    edge_dists = dists - split_radius - blob_rads

    can_eat_split_piece = danger_size > split_radius * SPLIT_EAT_RATIO
    nearby = edge_dists < player_radius * POST_SPLIT_DANGER_RANGE_MULT

    return bool(np.any(can_eat_split_piece & nearby))


def split_path_safe(cache, split_radius, split_landing):
    score = segment_virus_score(
        cache,
        cache["player_pos"],
        split_landing,
        split_radius,
        SPLIT_VIRUS_BUFFER_MULT,
    )

    return score > -OFF_MAP_PENALTY / 2


def get_split_decision(move_direction, cache):
    if cache["virus_farm_mode"]:
        return False, None

    if cache["own_blob_count"] != 1:
        return False, None

    if dangerous_enemy_near(cache, 4.0, ratio=1.05):
        return False, None

    player_radius = cache["player_radius"]

    if player_radius < SPLIT_MIN_RADIUS:
        return False, None

    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) == 0:
        return False, None

    move_direction = np.array(move_direction, dtype=float)
    move_direction = normalize(move_direction)

    if move_direction is None:
        return False, None

    player_pos = cache["player_pos"]

    split_radius = player_radius / np.sqrt(2.0)
    split_range = player_radius * SPLIT_RANGE_MULT

    if split_penalty(cache, split_radius):
        return False, None

    vectors = blob_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)

    safe_dists = dists.copy()
    safe_dists[safe_dists < 1.0] = 1.0

    blob_dirs = vectors / safe_dists.reshape(-1, 1)

    edge_dists = dists - player_radius - blob_rads
    alignments = blob_dirs @ move_direction

    can_eat_after_split = split_radius > blob_rads * SPLIT_EAT_RATIO
    close_enough = edge_dists <= split_range
    aligned = alignments >= SPLIT_ALIGNMENT_MIN

    candidates = can_eat_after_split & close_enough & aligned

    if not np.any(candidates):
        return False, None

    scores = blob_rads.copy()
    scores[~candidates] = -1.0

    candidate_indices = np.argsort(scores)[::-1]

    for idx in candidate_indices:
        if scores[idx] <= 0:
            break

        split_dir = blob_dirs[idx]
        split_landing = player_pos + split_dir * split_range

        if not split_path_safe(cache, split_radius, split_landing):
            continue

        return True, split_dir

    return False, None


# =========================
# General scoring / beam
# =========================

def wall_score(cache, x, y):
    return 0.0


def score_position(cache, x, y):
    r = cache["player_radius"]

    if r < 1.2:
        food_weight = 1600.0
        hunt_weight = 500.0
        danger_weight = 8000.0
    elif r < 2.2:
        food_weight = 1200.0
        hunt_weight = 2500.0
        danger_weight = 11000.0
    else:
        food_weight = 600.0
        hunt_weight = ENEMY_HUNT_WEIGHT
        danger_weight = ENEMY_DANGER_WEIGHT

    score = 0.0

    score += food_score(cache, x, y, food_weight)

    if cache["own_blob_count"] == 1:
        score += enemy_score_single(
            cache,
            x,
            y,
            danger_weight=danger_weight,
            hunt_weight=hunt_weight,
        )
    else:
        score += enemy_score_single(
            cache,
            x,
            y,
            danger_weight=danger_weight,
            hunt_weight=0.0,
        )

    score += point_virus_score(
        cache,
        np.array([x, y], dtype=float),
        cache["player_radius"],
        VIRUS_HARD_BUFFER_MULT,
    )

    score += wall_score(cache, x, y)

    return score


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


def unstuck_direction(cache, step_distance):
    best_score = -float("inf")
    best_direction = None

    for direction in DIRECTIONS:
        dx, dy = direction

        raw_x = cache["player_x"] + dx * step_distance
        raw_y = cache["player_y"] + dy * step_distance

        future_x, future_y = clamp_position(cache, raw_x, raw_y)

        actual_move = np.linalg.norm(
            np.array([future_x - cache["player_x"], future_y - cache["player_y"]])
        )

        if actual_move < step_distance * 0.3:
            continue

        score = score_position(cache, future_x, future_y)

        score += movement_virus_score(
            cache,
            cache["player_x"],
            cache["player_y"],
            future_x,
            future_y,
        )

        score += own_blob_virus_score(cache, dx, dy, step_distance)

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        return None

    return float(best_direction[0]), float(best_direction[1])


def choose_direction(game: Game):
    global LAST_DIRECTION

    cache = build_cache(game)
    step_distance = STEP_DISTANCE_MULT * cache["player_radius"]

    # 1. After virus crash / split: survival comes first.
    fragmented_dir = fragmented_escape_direction(cache, step_distance)

    if fragmented_dir is not None:
        dx, dy = fragmented_dir
        LAST_DIRECTION = np.array([dx, dy], dtype=float)
        return dx, dy, cache

    # 2. Single blob emergency escape.
    escape_dir = enemy_escape_direction(cache, step_distance)

    if escape_dir is not None:
        dx, dy = escape_dir
        LAST_DIRECTION = np.array([dx, dy], dtype=float)
        return dx, dy, cache

    # 3. Big enough: aggressively farm virus, but only if not punishable.
    virus_dir = virus_farm_direction(cache, step_distance)

    if virus_dir is not None:
        dx, dy = virus_dir
        cache["virus_farm_mode"] = True
        LAST_DIRECTION = np.array([dx, dy], dtype=float)
        return dx, dy, cache

    # 4. Early game / fragmented safe state: farm food directly.
    food_dir = food_farm_direction(cache, step_distance)

    if food_dir is not None:
        dx, dy = food_dir
        LAST_DIRECTION = np.array([dx, dy], dtype=float)
        return dx, dy, cache

    # 5. Close kill only if obvious and safe.
    kill_dir = close_kill_direction(cache, step_distance)

    if kill_dir is not None:
        dx, dy = kill_dir
        LAST_DIRECTION = np.array([dx, dy], dtype=float)
        return dx, dy, cache

    # 6. Stuck recovery.
    if is_stuck(cache):
        unstuck_dir = unstuck_direction(cache, step_distance)

        if unstuck_dir is not None:
            dx, dy = unstuck_dir
            LAST_DIRECTION = np.array([dx, dy], dtype=float)
            return dx, dy, cache

    # 7. Beam search fallback.
    beam = [
        (
            0.0,
            cache["player_x"],
            cache["player_y"],
            None,
            LAST_DIRECTION,
        )
    ]

    for step in range(1, ROLLOUT_STEPS + 1):
        new_beam = []
        discount = ROLLOUT_DISCOUNT ** (step - 1)

        for current_score, x, y, first_direction, previous_direction in beam:
            for direction in DIRECTIONS:
                dx, dy = direction

                future_x = x + dx * step_distance
                future_y = y + dy * step_distance
                future_x, future_y = clamp_position(cache, future_x, future_y)

                position_score = score_position(cache, future_x, future_y)

                position_score += movement_virus_score(
                    cache,
                    x,
                    y,
                    future_x,
                    future_y,
                )

                if step == 1:
                    position_score += own_blob_virus_score(
                        cache,
                        dx,
                        dy,
                        step_distance,
                    )

                total_score = current_score + discount * position_score

                turn_alignment = np.dot(direction, previous_direction)
                turn_penalty = TURN_PENALTY_WEIGHT * (1.0 - turn_alignment)
                total_score -= turn_penalty

                if first_direction is None:
                    new_first_direction = direction
                else:
                    new_first_direction = first_direction

                new_beam.append(
                    (
                        total_score,
                        future_x,
                        future_y,
                        new_first_direction,
                        direction,
                    )
                )

        new_beam.sort(key=lambda item: item[0], reverse=True)
        beam = new_beam[:BEAM_WIDTH]

    _, _, _, best_direction, _ = beam[0]

    final_direction = SMOOTHING * LAST_DIRECTION + (1 - SMOOTHING) * best_direction
    final_direction = normalize(final_direction)

    if final_direction is None:
        final_direction = best_direction

    LAST_DIRECTION = final_direction

    return float(final_direction[0]), float(final_direction[1]), cache


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
                    dx, dy, cache = choose_direction(game)

                    should_do_split, split_direction = get_split_decision(
                        (dx, dy),
                        cache,
                    )

                    if should_do_split:
                        sx, sy = split_direction

                        game.send_move(
                            MovePlayer(
                                player_id=cache["player_id"],
                                direction=DirectionModel(
                                    x=float(sx),
                                    y=float(sy),
                                ),
                                split=True,
                            )
                        )
                    else:
                        game.send_move(
                            MovePlayer(
                                player_id=cache["player_id"],
                                direction=DirectionModel(
                                    x=float(dx),
                                    y=float(dy),
                                ),
                                split=False,
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