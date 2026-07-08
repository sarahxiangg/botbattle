from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

import numpy as np


# =========================
# Constants
# =========================

NUM_DIRECTIONS = 10

DIRECTIONS = np.array([
    [np.cos(2 * np.pi * i / NUM_DIRECTIONS),
     np.sin(2 * np.pi * i / NUM_DIRECTIONS)]
    for i in range(NUM_DIRECTIONS)
], dtype=float)

# Main weights
FOOD_WEIGHT_TINY = 1600.0
FOOD_WEIGHT_SMALL = 1200.0
FOOD_WEIGHT_MEDIUM = 900.0
FOOD_WEIGHT_BIG = 700.0

ENEMY_DANGER_WEIGHT = 13000.0
VIRUS_DANGER_WEIGHT = 10000.0
VIRUS_SAFE_WEIGHT = 0.0
DANGER_DISTANCE_POWER = 1.3

# Anti-jitter
LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)
SMOOTHING = 0.10
STICKINESS_WEIGHT = 300.0
TURN_PENALTY_WEIGHT = 150.0

# Lookahead / performance
ROLLOUT_STEPS = 2
ROLLOUT_DISCOUNT = 0.7
STEP_DISTANCE_MULT = 1.5
BEAM_WIDTH = 1

MAX_FOOD_CONSIDERED = 25
MAX_BLOBS_CONSIDERED = 30
MAX_VIRUSES_CONSIDERED = 10

# Eating / hunting
EAT_RATIO = 1.12
CLOSE_KILL_RANGE_MULT = 2.2
CLOSE_OVERRIDE_RANGE_MULT = 1.4

# Splitting
SPLIT_EAT_RATIO = 1.15
SPLIT_RANGE_MULT = 3.0
SPLIT_ALIGNMENT_MIN = 0.88
SPLIT_MIN_RADIUS = 1.45
POST_SPLIT_DANGER_RANGE_MULT = 5.0

# Enemy split prediction
ENEMY_SPLIT_THREAT_WEIGHT = 30000.0
ENEMY_SPLIT_RANGE_MULT = 4.0
ENEMY_SPLIT_EAT_RATIO = 1.08
ENEMY_SPLIT_CONE_ALIGNMENT = 0.65

# Walls
ARENA_SIZE = 60.0
WALL_DANGER_WEIGHT = 3000.0
WALL_MARGIN_MULT = 1.2
OFF_MAP_PENALTY = 1_000_000_000.0


# =========================
# Cache utilities
# =========================

def clamp_position(cache, x, y):
    r = cache["player_radius"]

    x = min(max(x, r), ARENA_SIZE - r)
    y = min(max(y, r), ARENA_SIZE - r)

    return x, y

def cap_nearest(locs, max_items, player_pos, *arrays):
    if len(locs) <= max_items:
        return (locs, *arrays)

    dists = np.sum((locs - player_pos) ** 2, axis=1)
    keep = np.argpartition(dists, max_items - 1)[:max_items]

    result = [locs[keep]]

    for arr in arrays:
        result.append(arr[keep])

    return tuple(result)


def build_cache(game):
    me = game.state.me

    player_pos = np.array([me.x, me.y], dtype=float)
    player_radius = float(me.radius)
    player_id = me.player_id
    own_blob_count = len(me.blobs)

    # ---------- FOOD ----------
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

    # ---------- ENEMY BLOBS ----------
    blobs = game.state.visible_blobs

    if blobs:
        blob_locs = np.array([blob.pos for blob in blobs], dtype=float)
        blob_rads = np.array([blob.radius for blob in blobs], dtype=float)
        blob_player_ids = np.array([blob.player_id for blob in blobs], dtype=int)

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

    # ---------- VIRUSES ----------
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
# Mode weights
# =========================

def get_mode_weights(cache):
    r = cache["player_radius"]

    if r < 1.2:
        return {
            "food": FOOD_WEIGHT_TINY,
            "enemy_danger": 7000.0,
            "enemy_hunt": 800.0,
            "hunt_ratio": 0.55,
            "merged_safe_ratio": 0.80,
        }

    if r < 1.7:
        return {
            "food": FOOD_WEIGHT_SMALL,
            "enemy_danger": 5500.0,
            "enemy_hunt": 4500.0,
            "hunt_ratio": 0.95,
            "merged_safe_ratio": 1.30,
        }

    if r < 2.5:
        return {
            "food": FOOD_WEIGHT_MEDIUM,
            "enemy_danger": 8500.0,
            "enemy_hunt": 3000.0,
            "hunt_ratio": 0.85,
            "merged_safe_ratio": 1.15,
        }

    return {
        "food": FOOD_WEIGHT_BIG,
        "enemy_danger": 11000.0,
        "enemy_hunt": 1500.0,
        "hunt_ratio": 0.65,
        "merged_safe_ratio": 0.90,
    }


# =========================
# Scoring functions
# =========================

def food_score(cache, future_x, future_y, weight):
    food_locs = cache["food_locs"]

    if len(food_locs) == 0:
        return 0.0

    future_pos = np.array([future_x, future_y], dtype=float)

    dists = np.linalg.norm(food_locs - future_pos, axis=1)
    dists[dists < 1.0] = 1.0

    return float(np.sum(weight / (dists ** 2)))


def enemy_score(cache, future_x, future_y, danger_weight, hunt_weight, hunt_ratio, merged_safe_ratio):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return 0.0

    future_pos = np.array([future_x, future_y], dtype=float)
    player_radius = cache["player_radius"]

    center_dists = np.linalg.norm(blob_locs - future_pos, axis=1)
    edge_dists = center_dists - player_radius - blob_rads
    edge_dists[edge_dists < 1.0] = 1.0

    individually_dangerous = blob_rads > player_radius * 1.10

    merged_dangerous = (
        (merged_rads > player_radius * 1.25) &
        (edge_dists < player_radius * 4.0)
    )

    bigger = individually_dangerous | merged_dangerous

    can_eat = player_radius > blob_rads * EAT_RATIO

    normal_hunt = (
        can_eat &
        (blob_rads < player_radius * hunt_ratio) &
        (merged_rads < player_radius * merged_safe_ratio)
    )

    close_kill = (
        can_eat &
        (edge_dists < player_radius * CLOSE_KILL_RANGE_MULT)
    )

    smaller = normal_hunt | close_kill

    danger_size = np.maximum(blob_rads, merged_rads)

    danger_scores = danger_weight * (danger_size / player_radius) / (edge_dists ** DANGER_DISTANCE_POWER)
    hunt_scores = hunt_weight * (player_radius / blob_rads) / (edge_dists ** 2)

    score = 0.0
    score -= np.sum(danger_scores[bigger])
    score += np.sum(hunt_scores[smaller])

    return float(score)


def virus_score(cache, future_x, future_y, danger_weight):
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if len(virus_locs) == 0:
        return 0.0

    future_pos = np.array([future_x, future_y], dtype=float)
    player_radius = cache["player_radius"]

    center_dists = np.linalg.norm(virus_locs - future_pos, axis=1)
    edge_dists = center_dists - player_radius - virus_rads

    dangerous = player_radius > virus_rads * 1.1

    if np.any(dangerous & (edge_dists < player_radius * 0.25)):
        return -OFF_MAP_PENALTY

    edge_dists[edge_dists < 1.0] = 1.0

    danger_scores = danger_weight / (edge_dists ** 3)

    return -float(np.sum(danger_scores[dangerous]))


def wall_score(cache, x, y):
    # player_radius = cache["player_radius"]

    # if (
    #     x - player_radius <= 0 or
    #     x + player_radius >= ARENA_SIZE or
    #     y - player_radius <= 0 or
    #     y + player_radius >= ARENA_SIZE
    # ):
    #     return -OFF_MAP_PENALTY

    return 0.0


def enemy_split_threat_score(cache, future_x, future_y):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) == 0:
        return 0.0

    player_radius = cache["player_radius"]
    current_pos = cache["player_pos"]
    future_pos = np.array([future_x, future_y], dtype=float)

    enemy_split_rads = blob_rads / np.sqrt(2.0)

    can_split_eat_us = enemy_split_rads > player_radius * ENEMY_SPLIT_EAT_RATIO

    if not np.any(can_split_eat_us):
        return 0.0

    to_current = current_pos - blob_locs
    to_future = future_pos - blob_locs

    dist_current = np.linalg.norm(to_current, axis=1)
    dist_future = np.linalg.norm(to_future, axis=1)

    dist_current[dist_current < 1.0] = 1.0
    dist_future[dist_future < 1.0] = 1.0

    unit_current = to_current / dist_current.reshape(-1, 1)
    unit_future = to_future / dist_future.reshape(-1, 1)

    alignment = np.sum(unit_current * unit_future, axis=1)

    split_reach = blob_rads * ENEMY_SPLIT_RANGE_MULT
    edge_dist = dist_future - enemy_split_rads - player_radius

    in_split_range = edge_dist < split_reach
    in_split_cone = alignment > ENEMY_SPLIT_CONE_ALIGNMENT

    threatened = can_split_eat_us & in_split_range & in_split_cone

    if not np.any(threatened):
        return 0.0

    closeness = (split_reach - edge_dist) / split_reach
    closeness = np.clip(closeness, 0.0, 1.0)

    size_ratio = enemy_split_rads / player_radius
    threat_scores = ENEMY_SPLIT_THREAT_WEIGHT * size_ratio * (closeness ** 2)

    return -float(np.sum(threat_scores[threatened]))


def score_position(cache, weights, x, y):
    score = 0.0

    score += food_score(
        cache,
        x,
        y,
        weight=weights["food"],
    )

    score += enemy_score(
        cache,
        x,
        y,
        danger_weight=weights["enemy_danger"],
        hunt_weight=weights["enemy_hunt"],
        hunt_ratio=weights["hunt_ratio"],
        merged_safe_ratio=weights["merged_safe_ratio"],
    )

    score += virus_score(
        cache,
        x,
        y,
        danger_weight=VIRUS_DANGER_WEIGHT,
    )

    score += wall_score(
        cache,
        x,
        y,
    )

    return score


# =========================
# Tactical direction helpers
# =========================

def food_direction(cache):
    food_locs = cache["food_locs"]

    if len(food_locs) == 0:
        return None

    player_pos = cache["player_pos"]

    vectors = food_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)
    dists[dists < 1.0] = 1.0

    weights = 1.0 / (dists ** 2)
    target = np.sum(food_locs * weights.reshape(-1, 1), axis=0) / np.sum(weights)

    direction = target - player_pos
    norm = np.linalg.norm(direction)

    if norm == 0:
        return None

    direction = direction / norm

    return float(direction[0]), float(direction[1])


def close_kill_direction(cache):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) == 0:
        return None

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    vectors = blob_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)
    dists[dists < 1.0] = 1.0

    edge_dists = dists - player_radius - blob_rads

    can_eat = player_radius > blob_rads * EAT_RATIO
    close = edge_dists < player_radius * CLOSE_OVERRIDE_RANGE_MULT

    targets = can_eat & close

    if not np.any(targets):
        return None

    scores = blob_rads / dists
    scores[~targets] = -1.0

    idx = int(np.argmax(scores))
    direction = vectors[idx] / dists[idx]

    return float(direction[0]), float(direction[1])


def enemy_escape_direction(cache, step_distance):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]

    if len(blob_locs) == 0:
        return None

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    dists = np.linalg.norm(blob_locs - player_pos, axis=1)
    edge_dists = dists - player_radius - blob_rads

    danger_size = np.maximum(blob_rads, merged_rads)

    dangerous = danger_size > player_radius * 1.10
    too_close = edge_dists < player_radius * 3.5

    if not np.any(dangerous & too_close):
        return None

    best_score = -float("inf")
    best_direction = None

    for direction in DIRECTIONS:
        dx, dy = direction

        future_x = cache["player_x"] + dx * step_distance
        future_y = cache["player_y"] + dy * step_distance

        future_x, future_y = clamp_position(cache, future_x, future_y)

        score = 0.0

        score += enemy_score(
            cache,
            future_x,
            future_y,
            danger_weight=ENEMY_DANGER_WEIGHT,
            hunt_weight=0.0,
            hunt_ratio=0.0,
            merged_safe_ratio=0.0,
        )

        score += enemy_split_threat_score(cache, future_x, future_y)
        score += virus_score(cache, future_x, future_y, VIRUS_DANGER_WEIGHT)
        score += wall_score(cache, future_x, future_y)

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        return None

    return float(best_direction[0]), float(best_direction[1])


def override_direction(cache, step_distance):
    # 1. Emergency escape from bigger enemies
    escape_dir = enemy_escape_direction(cache, step_distance)

    if escape_dir is not None:
        return escape_dir

    # 2. Close kill finisher
    kill_dir = close_kill_direction(cache)

    if kill_dir is not None:
        dx, dy = kill_dir

        future_x = cache["player_x"] + dx * step_distance
        future_y = cache["player_y"] + dy * step_distance
        
        future_x, future_y = clamp_position(cache, future_x, future_y)

        safe_kill = (
            wall_score(cache, future_x, future_y) != -OFF_MAP_PENALTY and
            enemy_split_threat_score(cache, future_x, future_y) > -20000 and
            virus_score(cache, future_x, future_y, VIRUS_DANGER_WEIGHT) > -OFF_MAP_PENALTY / 2
        )

        if safe_kill:
            return dx, dy

    # 3. Pure food mode
    if len(cache["blob_locs"]) == 0 and len(cache["virus_locs"]) == 0:
        food_dir = food_direction(cache)

        if food_dir is not None:
            dx, dy = food_dir

            future_x = cache["player_x"] + dx * step_distance
            future_y = cache["player_y"] + dy * step_distance

            future_x, future_y = clamp_position(cache, future_x, future_y)

            if wall_score(cache, future_x, future_y) > -OFF_MAP_PENALTY / 2:
                return dx, dy

    return None


# =========================
# Split functions
# =========================

def split_penalty(cache, split_radius):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) == 0:
        return False

    player_pos = cache["player_pos"]
    player_radius = cache["player_radius"]

    danger_range = player_radius * POST_SPLIT_DANGER_RANGE_MULT

    dists = np.linalg.norm(blob_locs - player_pos, axis=1)
    edge_dists = dists - split_radius - blob_rads

    can_eat_split_piece = blob_rads > split_radius * SPLIT_EAT_RATIO
    too_close = edge_dists < danger_range

    return bool(np.any(can_eat_split_piece & too_close))


def split_path_safe(cache, split_radius, split_landing):
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if len(virus_locs) == 0:
        return True

    player_pos = cache["player_pos"]

    segment = split_landing - player_pos
    segment_len_sq = np.dot(segment, segment)

    if segment_len_sq <= 0:
        return True

    to_viruses = virus_locs - player_pos

    t = np.sum(to_viruses * segment, axis=1) / segment_len_sq
    t = np.clip(t, 0.0, 1.0)

    closest_points = player_pos + t.reshape(-1, 1) * segment
    dists_to_path = np.linalg.norm(virus_locs - closest_points, axis=1)

    dangerous_viruses = split_radius > virus_rads * 1.1

    too_close_to_virus_path = (
        dists_to_path <
        split_radius + virus_rads + split_radius * 0.8
    )

    return not bool(np.any(dangerous_viruses & too_close_to_virus_path))


def get_split_decision(move_direction, cache):
    player_radius = cache["player_radius"]

    if player_radius < SPLIT_MIN_RADIUS:
        return False, None

    if cache["own_blob_count"] != 1:
        return False, None

    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) == 0:
        return False, None

    move_direction = np.array(move_direction, dtype=float)
    move_norm = np.linalg.norm(move_direction)

    if np.isclose(move_norm, 0.0):
        return False, None

    move_direction = move_direction / move_norm

    player_pos = cache["player_pos"]

    split_radius = player_radius / np.sqrt(2.0)
    split_range = player_radius * SPLIT_RANGE_MULT

    if split_penalty(cache, split_radius):
        return False, None

    vectors = blob_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)

    valid_dist = dists > 1e-9

    safe_dists = dists.copy()
    safe_dists[safe_dists < 1.0] = 1.0

    blob_uvs = vectors / safe_dists.reshape(-1, 1)

    edge_dists = dists - player_radius - blob_rads
    alignments = blob_uvs @ move_direction

    can_eat_after_split = split_radius > blob_rads * SPLIT_EAT_RATIO
    close_enough = edge_dists <= split_range
    aligned = alignments >= SPLIT_ALIGNMENT_MIN

    candidates = valid_dist & can_eat_after_split & close_enough & aligned

    if not np.any(candidates):
        return False, None

    scores = blob_rads.copy()
    scores[~candidates] = -1.0

    candidate_indices = np.argsort(scores)[::-1]

    for idx in candidate_indices:
        if scores[idx] <= 0:
            break

        blob_uv = blob_uvs[idx]
        split_landing = player_pos + blob_uv * split_range

        if not split_path_safe(cache, split_radius, split_landing):
            continue

        return True, blob_uv

    return False, None


# =========================
# Direction choice
# =========================

def choose_direction(game: Game):
    global LAST_DIRECTION

    cache = build_cache(game)
    weights = get_mode_weights(cache)

    step_distance = STEP_DISTANCE_MULT * cache["player_radius"]

    override_dir = override_direction(cache, step_distance)

    if override_dir is not None:
        dx, dy = override_dir
        LAST_DIRECTION = np.array([dx, dy], dtype=float)
        return dx, dy, cache

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

                position_score = score_position(
                    cache,
                    weights,
                    future_x,
                    future_y,
                )

                if step == 1:
                    position_score += enemy_split_threat_score(
                        cache,
                        future_x,
                        future_y,
                    )

                total_score = current_score + discount * position_score

                turn_alignment = np.dot(direction, previous_direction)
                turn_penalty = TURN_PENALTY_WEIGHT * (1.0 - turn_alignment)
                total_score -= turn_penalty

                if step == 1:
                    if wall_score(cache, future_x, future_y) > -OFF_MAP_PENALTY / 2:
                        total_score += STICKINESS_WEIGHT * np.dot(direction, LAST_DIRECTION)

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

    best_score, best_x, best_y, best_direction, _ = beam[0]

    final_direction = SMOOTHING * LAST_DIRECTION + (1 - SMOOTHING) * best_direction

    norm = np.linalg.norm(final_direction)

    if norm == 0:
        final_direction = best_direction
    else:
        final_direction = final_direction / norm

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