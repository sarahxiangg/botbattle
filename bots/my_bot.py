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

# Movement
LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)
STEP_DISTANCE_MULT = 1.5
TURN_PENALTY_WEIGHT = 80.0

# Food / roam target locking
FOOD_TARGET_REACHED_DIST_MULT = 0.40
FOOD_TARGET_REACHED_MIN = 0.35
FOOD_TARGET_MAX_TICKS = 10
FOOD_NO_PROGRESS_LIMIT = 4
FOOD_BLACKLIST_TICKS = 25
FOOD_CORNER_MARGIN = 0.45

FOOD_CLUSTER_RADIUS = 4.0
FOOD_CLUSTER_WEIGHT = 0.55
FOOD_DISTANCE_POWER = 1.15
FOOD_MIN_TARGET_DIST = 0.15

FOOD_TARGET_KEY = None
FOOD_TARGET_TICKS = 0
FOOD_TARGET_NO_PROGRESS = 0
FOOD_TARGET_LAST_DIST = None
FOOD_BLACKLIST = {}

ROAM_DIRECTION = None
ROAM_TICKS = 0
ROAM_LOCK_TICKS = 6
ROAM_CENTER_WEIGHT = 350.0
ROAM_MOVE_WEIGHT = 1000.0
ROAM_STICKINESS_WEIGHT = 250.0

# Unified danger settings
DANGER_OVERRIDE_RANGE_MULT = 9.0
DANGER_DIRECT_RATIO = 1.06
DANGER_SPLIT_RATIO = 1.08
DANGER_SPLIT_RANGE_MULT = 5.0
DANGER_OVERRIDE_WEIGHT = 120000.0
DANGER_DISTANCE_POWER = 1.2
DANGER_HARD_CLOSE_MULT = 2.5
DANGER_WALL_PUSH_PENALTY = 50000.0
DANGER_MOVE_REWARD = 5000.0

# Normal scoring / chase
FOOD_SCORE_WEIGHT = 900.0
ENEMY_DANGER_WEIGHT = 14000.0
ENEMY_HUNT_WEIGHT = 2500.0
EAT_RATIO = 1.12
CHASE_RANGE_MULT = 5.0

# Virus safety / farming
VIRUS_DANGER_WEIGHT = 12000.0
VIRUS_HARD_BUFFER_MULT = 0.65
VIRUS_PATH_BUFFER_MULT = 0.65
VIRUS_FARM_EAT_RATIO = 1.10
VIRUS_FARM_ENEMY_RANGE_MULT = 8.0
VIRUS_FARM_PIECE_RADIUS_MULT = 0.35

# Split kill
SPLIT_MIN_RADIUS = 1.35
SPLIT_EAT_RATIO = 1.10
SPLIT_RANGE_MULT = 2.5
SPLIT_RANGE_SAFETY_MULT = 0.75
SPLIT_TARGET_MIN_RADIUS_MULT = 0.16
SPLIT_VIRUS_BUFFER_MULT = 1.3
POST_SPLIT_DANGER_RANGE_MULT = 5.5
SPLIT_ALIGNMENT_MIN = 0.92

# Stuck safety net
WALL_BUFFER = 0.25
STUCK_TICK_LIMIT = 4
STUCK_MOVE_EPS_MULT = 0.04
STUCK_CENTER_WEIGHT = 600.0
STUCK_MOVE_WEIGHT = 2000.0

LAST_POSITION = None
STUCK_TICKS = 0

#chase prediction
ENEMY_LAST_POS = {}
ENEMY_VEL = {}

CHASE_TARGET_KEY = None
CHASE_TARGET_TICKS = 0
CHASE_LOST_TICKS = 0

CHASE_LOCK_TICKS = 14
CHASE_LOST_LIMIT = 4
CHASE_LEAD_TICKS = 2.2
CHASE_WALL_COMMIT_DIST = 8.0
CHASE_MIN_CLOSING_RATE = -0.15

# Virus memory
VIRUS_MEMORY = {}
VIRUS_MEMORY_REACHED_DIST_MULT = 1.4
VIRUS_MEMORY_KEY_DECIMALS = 1


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


def direction_future_pos(cache, dx, dy, step_distance):
    future_x = cache["player_x"] + dx * step_distance
    future_y = cache["player_y"] + dy * step_distance
    return clamp_position(cache, future_x, future_y)


def actual_move_distance(cache, future_x, future_y):
    return float(np.linalg.norm(
        np.array([
            future_x - cache["player_x"],
            future_y - cache["player_y"],
        ], dtype=float)
    ))


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
# Unified danger system
# =========================

def segment_virus_score(cache, start_pos, end_pos, radius, buffer_mult):
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if len(virus_locs) == 0:
        return 0.0

    # Viruses are only dangerous when we are large enough to pop on them.
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


def enemy_threat_report(cache, dx=0.0, dy=0.0, step_distance=0.0):
    """
    One source of truth for enemy danger.

    active: enemy is close enough that danger override should take control.
    safe: this direction is not immediately unsafe.
    score: continuous penalty used to rank escape directions.
    """
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return {
            "active": False,
            "safe": True,
            "score": 0.0,
        }

    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]
    player_radius = cache["player_radius"]

    danger_size = np.maximum(blob_rads, merged_rads)
    enemy_split_rads = blob_rads / np.sqrt(2.0)
    delta = np.array([dx * step_distance, dy * step_distance], dtype=float)

    active = False
    unsafe = False
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

        direct_danger = danger_size > own_rad * DANGER_DIRECT_RATIO
        direct_active = direct_danger & (
            edge_dists < player_radius * DANGER_OVERRIDE_RANGE_MULT
        )
        direct_unsafe = direct_danger & (
            edge_dists < own_rad * DANGER_HARD_CLOSE_MULT
        )

        split_danger = enemy_split_rads > own_rad * DANGER_SPLIT_RATIO
        split_near = edge_dists < blob_rads * DANGER_SPLIT_RANGE_MULT
        split_active = split_danger & split_near

        if np.any(direct_active | split_active):
            active = True

        if np.any(direct_unsafe | split_active):
            unsafe = True

        score_dists = edge_dists.copy()
        score_dists[score_dists < 1.0] = 1.0

        scored_danger = direct_danger | split_active
        danger_scores = (
            DANGER_OVERRIDE_WEIGHT
            * (danger_size / own_rad)
            / (score_dists ** DANGER_DISTANCE_POWER)
        )
        score -= np.sum(danger_scores[scored_danger])

    return {
        "active": bool(active),
        "safe": not bool(unsafe),
        "score": float(score),
    }


def danger_report(cache, dx=0.0, dy=0.0, step_distance=0.0, check_virus=True):
    """
    Unified danger interface for normal movement.

    Use check_virus=False only for intentional virus farming.
    """
    enemy = enemy_threat_report(cache, dx, dy, step_distance)

    if check_virus:
        virus_score = own_blob_virus_score(cache, dx, dy, step_distance)
    else:
        virus_score = 0.0

    virus_safe = virus_score > -OFF_MAP_PENALTY / 2

    future_x, future_y = direction_future_pos(cache, dx, dy, step_distance)
    actual_move = actual_move_distance(cache, future_x, future_y)

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


# =========================
# Mode 1: danger override
# =========================

def danger_override_mode(cache, step_distance):
    # Only enemy danger activates emergency mode. Virus danger is handled as a
    # hard safety filter inside normal movement.
    if not danger_report(cache)["enemy_active"]:
        return None

    best_score = -float("inf")
    best_direction = None

    for direction in DIRECTIONS:
        dx, dy = direction
        report = danger_report(cache, dx, dy, step_distance)

        # Never intentionally run through a virus unless virus farming mode has
        # explicitly chosen to do so.
        if not report["virus_safe"]:
            continue

        score = report["enemy_score"]

        if report["actual_move"] < step_distance * 0.20:
            score -= DANGER_WALL_PUSH_PENALTY
        else:
            score += DANGER_MOVE_REWARD * report["actual_move"]

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        return None

    return float(best_direction[0]), float(best_direction[1]), False


# =========================
# Mode 2: split kill
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

def split_can_reach_target(player_pos, player_radius, target_pos, target_radius):
    split_radius = player_radius / np.sqrt(2.0)
    split_range = player_radius * SPLIT_RANGE_MULT * SPLIT_RANGE_SAFETY_MULT

    vector = target_pos - player_pos
    dist = np.linalg.norm(vector)

    if dist <= 1e-9:
        return False

    direction = vector / dist

    landing = player_pos + direction * split_range

    # How close the landing blob centre gets to target centre.
    landing_dist = np.linalg.norm(target_pos - landing)

    # Require overlap with a strict buffer.
    hit_radius = split_radius + target_radius

    return landing_dist <= hit_radius * 0.75

def split_kill_mode(cache, step_distance):
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

    can_eat_now = player_radius > blob_rads * EAT_RATIO

    if not np.any(can_eat_now):
        return None

    split_radius = player_radius / np.sqrt(2.0)
    split_range = player_radius * SPLIT_RANGE_MULT

    useful_target = blob_rads > player_radius * SPLIT_TARGET_MIN_RADIUS_MULT
    reach_ok = np.array([
        split_can_reach_target(player_pos, player_radius, blob_locs[i], blob_rads[i])
        for i in range(len(blob_locs))
    ], dtype=bool)

    candidates = (
        (player_radius >= SPLIT_MIN_RADIUS)
        & (split_radius > blob_rads * SPLIT_EAT_RATIO)
        & useful_target
        & can_eat_now
        & reach_ok
    )

    if not np.any(candidates):
        return None

    if not post_split_enemy_safe(cache, split_radius):
        return None

    scores = blob_rads * 3.0 - dists * 0.25
    scores[~candidates] = -1.0

    for idx in np.argsort(scores)[::-1]:
        if scores[idx] <= 0:
            break

        split_dir = dirs[idx]
        split_landing = player_pos + split_dir * split_range

        if not split_path_safe(cache, split_radius, split_landing):
            continue

        dx, dy = float(split_dir[0]), float(split_dir[1])

        # Enemy safety only. The split path check already handles virus risk.
        if not danger_report(cache, dx, dy, step_distance, check_virus=False)["enemy_safe"]:
            continue

        return dx, dy, True

    return None


# =========================
# Mode 3: virus farm
# =========================

def virus_key(pos):
    return (
        round(float(pos[0]), VIRUS_MEMORY_KEY_DECIMALS),
        round(float(pos[1]), VIRUS_MEMORY_KEY_DECIMALS),
    )


def update_virus_memory(cache):
    """
    Remember every virus we see.
    If we visit a remembered virus location and it is no longer there,
    remove it from memory.
    """
    global VIRUS_MEMORY

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    visible_keys = set()

    # Add / refresh currently visible viruses.
    for pos, rad in zip(virus_locs, virus_rads):
        key = virus_key(pos)
        visible_keys.add(key)

        VIRUS_MEMORY[key] = {
            "pos": np.array(pos, dtype=float),
            "radius": float(rad),
        }

    # Remove remembered viruses if we reached their location and they are gone.
    reached_dist = max(
        1.0,
        player_radius * VIRUS_MEMORY_REACHED_DIST_MULT,
    )

    expired = []

    for key, info in VIRUS_MEMORY.items():
        if key in visible_keys:
            continue

        dist = np.linalg.norm(info["pos"] - player_pos)

        if dist < reached_dist:
            expired.append(key)

    for key in expired:
        del VIRUS_MEMORY[key]

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


def virus_farm_mode(cache, step_distance):
    if cache["own_blob_count"] != 1:
        return None

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    # Build candidates from virus memory, not just visible viruses.
    candidates = []

    for key, info in VIRUS_MEMORY.items():
        virus_pos = info["pos"]
        virus_rad = info["radius"]

        # Only go if physically big enough to farm it.
        if player_radius <= virus_rad * VIRUS_FARM_EAT_RATIO:
            continue

        dist = np.linalg.norm(virus_pos - player_pos)

        candidates.append((
            dist,
            key,
            virus_pos,
            virus_rad,
        ))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])

    visible_keys = {
        virus_key(pos)
        for pos in cache["virus_locs"]
    }

    for dist, key, virus_pos, virus_rad in candidates:
        # If we reached a remembered virus and it is not visible, delete it.
        reached_dist = max(
            1.0,
            player_radius * VIRUS_MEMORY_REACHED_DIST_MULT,
        )

        if dist < reached_dist and key not in visible_keys:
            if key in VIRUS_MEMORY:
                del VIRUS_MEMORY[key]
            continue

        # If the virus is currently visible, check post-pop enemy safety.
        # If it is not visible yet, we can still travel toward it, but safely.
        is_visible = key in visible_keys

        if is_visible:
            if not virus_farm_enemy_safe(cache, virus_pos, virus_rad):
                continue

        direction = normalize(virus_pos - player_pos)

        if direction is None:
            continue

        dx, dy = float(direction[0]), float(direction[1])

        if is_visible:
            # Intentional virus contact: enemy safety only.
            report = danger_report(
                cache,
                dx,
                dy,
                step_distance,
                check_virus=False,
            )

            if not report["enemy_safe"]:
                continue
        else:
            # Travelling toward remembered virus: still avoid currently visible viruses.
            report = danger_report(
                cache,
                dx,
                dy,
                step_distance,
                check_virus=True,
            )

            if not report["safe"]:
                continue

        if report["actual_move"] < step_distance * 0.20:
            continue

        return dx, dy, False

    return None


# =========================
# Mode 4: chase
# =========================

def update_enemy_velocity(cache):
    """
    Estimate enemy velocities from the previous tick.

    We do not assume blob IDs exist. Instead, for each player_id, we match each
    current blob to the closest previous blob from the same player. This is not
    perfect, but it is stable enough for lead-pursuit chasing.
    """
    global ENEMY_LAST_POS, ENEMY_VEL

    blob_locs = cache["blob_locs"]
    blob_player_ids = cache["blob_player_ids"]

    blob_vels = np.zeros_like(blob_locs, dtype=float)

    if len(blob_locs) == 0:
        ENEMY_LAST_POS = {}
        ENEMY_VEL = {}
        cache["blob_vels"] = blob_vels
        return

    previous_by_pid = {}
    for key, pos in ENEMY_LAST_POS.items():
        pid = key[0]
        previous_by_pid.setdefault(pid, []).append((key, pos))

    new_last = {}
    new_vel = {}

    for pid in np.unique(blob_player_ids):
        indices = np.where(blob_player_ids == pid)[0]
        previous = previous_by_pid.get(int(pid), [])
        used_previous = set()

        for local_num, idx in enumerate(indices):
            pos = blob_locs[idx].copy()
            best_prev_key = None
            best_prev_pos = None
            best_dist = float("inf")

            for prev_key, prev_pos in previous:
                if prev_key in used_previous:
                    continue

                dist = float(np.sum((pos - prev_pos) ** 2))

                if dist < best_dist:
                    best_dist = dist
                    best_prev_key = prev_key
                    best_prev_pos = prev_pos

            if best_prev_pos is None:
                vel = np.array([0.0, 0.0], dtype=float)
            else:
                vel = pos - best_prev_pos
                used_previous.add(best_prev_key)

            # Clamp extreme tracking jumps so bad matching does not cause insane prediction.
            speed = np.linalg.norm(vel)
            max_reasonable_speed = cache["player_radius"] * 2.5
            if speed > max_reasonable_speed and speed > 1e-9:
                vel = vel / speed * max_reasonable_speed

            # Store per current blob index.
            blob_vels[idx] = vel

            # Current key only needs to be unique within this tick/player.
            new_key = (int(pid), int(local_num))
            new_last[new_key] = pos
            new_vel[new_key] = vel

    ENEMY_LAST_POS = new_last
    ENEMY_VEL = new_vel
    cache["blob_vels"] = blob_vels


def enemy_near_wall_or_corner(pos):
    x, y = pos

    dist_left = x
    dist_right = ARENA_SIZE - x
    dist_bottom = y
    dist_top = ARENA_SIZE - y

    nearest_wall_dist = min(dist_left, dist_right, dist_bottom, dist_top)

    near_wall = nearest_wall_dist < CHASE_WALL_COMMIT_DIST

    near_corner = (
        (dist_left < CHASE_WALL_COMMIT_DIST or dist_right < CHASE_WALL_COMMIT_DIST)
        and
        (dist_bottom < CHASE_WALL_COMMIT_DIST or dist_top < CHASE_WALL_COMMIT_DIST)
    )

    return near_wall or near_corner


def predicted_enemy_position(cache, idx, lead_ticks):
    blob_locs = cache["blob_locs"]
    blob_vels = cache.get("blob_vels")

    pos = blob_locs[idx]

    if blob_vels is None or len(blob_vels) <= idx:
        vel = np.array([0.0, 0.0], dtype=float)
    else:
        vel = blob_vels[idx]

    predicted = pos + vel * lead_ticks

    # Keep prediction inside board.
    predicted_x = min(max(predicted[0], 0.0), ARENA_SIZE)
    predicted_y = min(max(predicted[1], 0.0), ARENA_SIZE)

    return np.array([predicted_x, predicted_y], dtype=float), vel


def enemy_running_toward_wall(pos, vel):
    """Return True if the enemy's velocity is carrying it toward the nearest wall."""
    speed = np.linalg.norm(vel)

    if speed <= 1e-9:
        return False

    x, y = pos
    wall_vectors = [
        np.array([-1.0, 0.0], dtype=float) if x < ARENA_SIZE / 2.0 else np.array([1.0, 0.0], dtype=float),
        np.array([0.0, -1.0], dtype=float) if y < ARENA_SIZE / 2.0 else np.array([0.0, 1.0], dtype=float),
    ]

    vel_dir = vel / speed
    return any(np.dot(vel_dir, wall_dir) > 0.55 for wall_dir in wall_vectors)


def chase_mode(cache, step_distance):
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

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    vectors = blob_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)
    edge_dists = dists - player_radius - blob_rads

    can_eat = player_radius > blob_rads * EAT_RATIO
    in_chase_range = edge_dists < player_radius * CHASE_RANGE_MULT
    candidates = can_eat & in_chase_range

    if not np.any(candidates):
        CHASE_TARGET_KEY = None
        CHASE_TARGET_TICKS = 0
        CHASE_LOST_TICKS = 0
        return None

    candidate_indices = np.where(candidates)[0]

    # True target locking: briefly prefer the same enemy player if still valid.
    if CHASE_TARGET_KEY is not None and CHASE_TARGET_TICKS < CHASE_LOCK_TICKS:
        locked_indices = [
            idx for idx in candidate_indices
            if int(blob_player_ids[idx]) == CHASE_TARGET_KEY
        ]

        if locked_indices:
            candidate_indices = np.array(locked_indices, dtype=int)

    best_score = -float("inf")
    best_idx = None
    best_direction = None
    best_closing = 0.0
    best_near_wall = False

    for idx in candidate_indices:
        target_key = int(blob_player_ids[idx])

        predicted_pos, enemy_vel = predicted_enemy_position(
            cache,
            idx,
            CHASE_LEAD_TICKS,
        )

        chase_vec = predicted_pos - player_pos
        chase_dir = normalize(chase_vec)

        if chase_dir is None:
            continue

        dx, dy = float(chase_dir[0]), float(chase_dir[1])

        report = danger_report(cache, dx, dy, step_distance)

        if not report["safe"]:
            continue

        future_pos = np.array([report["future_x"], report["future_y"]], dtype=float)

        # Are we gaining against where the enemy is likely to be?
        current_dist = np.linalg.norm(blob_locs[idx] - player_pos)
        future_dist = np.linalg.norm(predicted_pos - future_pos)
        closing = float(current_dist - future_dist)

        target_near_wall = enemy_near_wall_or_corner(predicted_pos)
        target_running_wall = enemy_running_toward_wall(blob_locs[idx], enemy_vel)

        # Do not tunnel in open space if we clearly are not gaining.
        if closing < CHASE_MIN_CLOSING_RATE and not target_near_wall and not target_running_wall:
            score_penalty = 12.0
        else:
            score_penalty = 0.0

        # Reward valuable targets, closer targets, good closing, and wall traps.
        score = 0.0
        score += blob_rads[idx] * 4.0
        score -= max(edge_dists[idx], 0.0) * 0.35
        score += closing * 8.0
        score += report["score"] * 0.0002
        score -= score_penalty

        if target_near_wall:
            score += 10.0

        if target_running_wall:
            score += 5.0

        if CHASE_TARGET_KEY == target_key:
            score += 6.0

        if score > best_score:
            best_score = score
            best_idx = idx
            best_direction = chase_dir
            best_closing = closing
            best_near_wall = target_near_wall or target_running_wall

    if best_direction is None:
        CHASE_LOST_TICKS += 1

        if CHASE_LOST_TICKS >= CHASE_LOST_LIMIT:
            CHASE_TARGET_KEY = None
            CHASE_TARGET_TICKS = 0
            CHASE_LOST_TICKS = 0

        return None

    # Give up only after repeated non-closing chase attempts in open space.
    if best_closing < CHASE_MIN_CLOSING_RATE and not best_near_wall:
        CHASE_LOST_TICKS += 1
    else:
        CHASE_LOST_TICKS = 0

    if CHASE_LOST_TICKS >= CHASE_LOST_LIMIT:
        CHASE_TARGET_KEY = None
        CHASE_TARGET_TICKS = 0
        CHASE_LOST_TICKS = 0
        return None

    chosen_key = int(blob_player_ids[best_idx])

    if CHASE_TARGET_KEY == chosen_key:
        CHASE_TARGET_TICKS += 1
    else:
        CHASE_TARGET_KEY = chosen_key
        CHASE_TARGET_TICKS = 0

    return float(best_direction[0]), float(best_direction[1]), False


# =========================
# Mode 5: food / roam target locking
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


def ranked_food_targets(cache):
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
            current_dist < reached_dist
            or FOOD_TARGET_TICKS >= FOOD_TARGET_MAX_TICKS
            or FOOD_TARGET_NO_PROGRESS >= FOOD_NO_PROGRESS_LIMIT
        ):
            blacklist_food_key(FOOD_TARGET_KEY)
            current_idx = None

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

    scores[~valid] = -1.0

    # Target locking: prefer the current target unless it is clearly bad.
    if current_idx is not None and scores[current_idx] > 0:
        scores[current_idx] *= 1.25

    candidates = []

    for idx in np.argsort(scores)[::-1]:
        if scores[idx] <= 0:
            break

        direction = normalize(vectors[idx])

        if direction is None:
            continue

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


def choose_roam_direction(cache, step_distance):
    global ROAM_DIRECTION, ROAM_TICKS

    center = np.array([ARENA_SIZE / 2.0, ARENA_SIZE / 2.0], dtype=float)
    center_dir = normalize(center - cache["player_pos"])

    # Keep roaming direction briefly to avoid big-blob jitter.
    if ROAM_DIRECTION is not None and ROAM_TICKS < ROAM_LOCK_TICKS:
        dx, dy = ROAM_DIRECTION
        report = danger_report(cache, dx, dy, step_distance)

        if report["safe"] and report["actual_move"] >= step_distance * 0.20:
            ROAM_TICKS += 1
            return float(dx), float(dy), False

    best_score = -float("inf")
    best_direction = None

    for direction in DIRECTIONS:
        dx, dy = direction
        report = danger_report(cache, dx, dy, step_distance)

        if not report["safe"]:
            continue

        if report["actual_move"] < step_distance * 0.20:
            continue

        score = 0.0
        score += report["score"]
        score += ROAM_MOVE_WEIGHT * report["actual_move"]
        score += ROAM_STICKINESS_WEIGHT * np.dot(direction, LAST_DIRECTION)

        if center_dir is not None:
            score += ROAM_CENTER_WEIGHT * np.dot(direction, center_dir)

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        ROAM_DIRECTION = None
        ROAM_TICKS = 0
        return None

    ROAM_DIRECTION = np.array(best_direction, dtype=float)
    ROAM_TICKS = 0

    return float(best_direction[0]), float(best_direction[1]), False


def food_roam_mode(cache, step_distance):
    food_candidates = ranked_food_targets(cache)

    for _, dx, dy, key in food_candidates:
        report = danger_report(cache, dx, dy, step_distance)

        if report["actual_move"] < step_distance * 0.20:
            blacklist_food_key(key)
            continue

        if not report["safe"]:
            blacklist_food_key(key)
            continue

        return dx, dy, False

    return choose_roam_direction(cache, step_distance)


# =========================
# Stuck safety net
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


def unstuck_mode(cache, step_distance):
    if not is_stuck(cache):
        return None

    center = np.array([ARENA_SIZE / 2.0, ARENA_SIZE / 2.0], dtype=float)
    center_dir = normalize(center - cache["player_pos"])

    best_score = -float("inf")
    best_direction = None

    for direction in DIRECTIONS:
        dx, dy = direction
        report = danger_report(cache, dx, dy, step_distance)

        if not report["safe"]:
            continue

        if report["actual_move"] < step_distance * 0.25:
            continue

        score = 0.0
        score += STUCK_MOVE_WEIGHT * report["actual_move"]
        score += report["score"]

        if center_dir is not None:
            score += STUCK_CENTER_WEIGHT * np.dot(direction, center_dir)

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        return None

    return float(best_direction[0]), float(best_direction[1]), False


# =========================
# Final fallback scorer
# =========================

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
        ENEMY_DANGER_WEIGHT
        * (danger_size / player_radius)
        / (edge_dists ** 1.3)
    )

    hunt_scores = (
        ENEMY_HUNT_WEIGHT
        * (player_radius / blob_rads)
        / (edge_dists ** 2)
    )

    score = 0.0
    score -= np.sum(danger_scores[dangerous])

    if cache["own_blob_count"] == 1:
        score += np.sum(hunt_scores[edible])

    return float(score)


def tiny_fallback_scorer(cache, step_distance):
    best_score = -float("inf")
    best_direction = None

    for direction in DIRECTIONS:
        dx, dy = direction
        report = danger_report(cache, dx, dy, step_distance)

        if not report["safe"]:
            continue

        if report["actual_move"] < step_distance * 0.15:
            continue

        score = 0.0
        score += report["score"]
        score += 0.35 * food_score(cache, report["future_x"], report["future_y"])
        score += 0.35 * enemy_score(cache, report["future_x"], report["future_y"])

        turn_alignment = np.dot(direction, LAST_DIRECTION)
        score -= TURN_PENALTY_WEIGHT * (1.0 - turn_alignment)

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is not None:
        return float(best_direction[0]), float(best_direction[1]), False

    # Absolute last resort: keep previous direction, otherwise east.
    fallback = normalize(LAST_DIRECTION)

    if fallback is None:
        return 1.0, 0.0, False

    return float(fallback[0]), float(fallback[1]), False


# =========================
# Direction choice
# =========================

def choose_direction(game: Game):
    global LAST_DIRECTION

    cache = build_cache(game)
    update_enemy_velocity(cache)
    update_virus_memory(cache)
    
    step_distance = STEP_DISTANCE_MULT * cache["player_radius"]

    # Strategic priority order:
    # 1. danger override
    # 2. split kill
    # 3. virus farm
    # 4. chase
    # 5. food / roam mode with target locking
    # 6. tiny fallback scorer
    modes = [
        danger_override_mode,
        unstuck_mode,
        split_kill_mode,
        virus_farm_mode,
        chase_mode,
        food_roam_mode,
        tiny_fallback_scorer,
    ]

    for mode in modes:
        result = mode(cache, step_distance)

        if result is None:
            continue

        dx, dy, should_split = result
        final_direction = normalize(np.array([dx, dy], dtype=float))

        if final_direction is None:
            final_direction = LAST_DIRECTION

        LAST_DIRECTION = final_direction

        return (
            float(final_direction[0]),
            float(final_direction[1]),
            bool(should_split),
            cache,
        )

    # Should almost never happen because tiny_fallback_scorer returns something.
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
