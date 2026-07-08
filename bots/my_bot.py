from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

import numpy as np

# =========================
# Direction search
# =========================

NUM_DIRECTIONS = 10              # directions tested in search

DIRECTIONS = np.array([
    [np.cos(2 * np.pi * i / NUM_DIRECTIONS),
     np.sin(2 * np.pi * i / NUM_DIRECTIONS)]
    for i in range(NUM_DIRECTIONS)
], dtype=float)

# =========================
# Scoring weights
# =========================

ENEMY_DANGER_WEIGHT = 11000.0     # avoid bigger enemies
VIRUS_DANGER_WEIGHT = 10000.0    # avoid dangerous viruses
VIRUS_SAFE_WEIGHT = 200.0        # slight reward for safe viruses


# =========================
# Anti-jitter
# =========================

LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)  # previous direction
SMOOTHING = 0.10                                    # old/new direction blend
STICKINESS_WEIGHT = 300.0                           # reward same direction


# =========================
# Lookahead
# =========================

ROLLOUT_STEPS = 2                 # future steps checked
ROLLOUT_DISCOUNT = 0.7            # future score discount
STEP_DISTANCE_MULT = 1.5          # step size multiplier
BEAM_WIDTH = 2                    # paths kept per step
TURN_PENALTY_WEIGHT = 150.0       # discourage sharp turns
MAX_FOOD_CONSIDERED = 25
MAX_BLOBS_FOR_PREDICTION = 20

# =========================
# Splitting
# =========================

SPLIT_EAT_RATIO = 1.15              # required size advantage
SPLIT_RANGE_MULT = 3.0              # split reach estimate
SPLIT_ALIGNMENT_MIN = 0.88          # must face target
POST_SPLIT_DANGER_RANGE_MULT = 5.0  # danger scan range

SPLIT_MIN_RADIUS = 1.45             # min distance to consider a split


ENEMY_SPLIT_THREAT_WEIGHT = 30000.0
ENEMY_SPLIT_RANGE_MULT = 4.0
ENEMY_SPLIT_EAT_RATIO = 1.08
ENEMY_SPLIT_CONE_ALIGNMENT = 0.65

# =========================
# Walls
# =========================

ARENA_SIZE = 60.0                 # map is 0..60 in both x/y
WALL_DANGER_WEIGHT = 3000.0       # avoid wall edges
WALL_MARGIN_MULT = 1.2            # avoid within 4 radii
OFF_MAP_PENALTY = 1_000_000_000.0 # huge penalty if touching wall

EAT_RATIO = 1.12
CLOSE_KILL_RANGE_MULT = 2.2
CLOSE_OVERRIDE_RANGE_MULT = 1.8


'''
    Score functions: food_score, enemy_score, virus_score, wall score, multi-blob-safety score
'''

def food_score(cache, future_x, future_y, weight):
    food_locs = cache["food_locs"]

    if len(food_locs) == 0:
        return 0.0

    future_pos = np.array([future_x, future_y], dtype=float)
    
    dists = np.linalg.norm(food_locs - future_pos, axis=1)
    dists[dists < 1.0] = 1.0
    scores = weight / (dists**2)

    return float(np.sum(scores))

def food_direction(cache, player):
    food_locs = cache["food_locs"]

    if len(food_locs) == 0:
        return None

    player_pos = np.array([player.x, player.y], dtype=float)

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


def enemy_score(cache, player, future_x, future_y, danger_weight, hunt_weight, hunt_ratio, merged_safe_ratio):
    future_pos = np.array([future_x, future_y], dtype=float)

    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]
    potential_rads = cache["potential_rads"]

    if len(blob_locs) == 0:
        return 0.0

    player_radius = player.radius

    center_dists = np.linalg.norm(blob_locs - future_pos, axis=1)

    edge_dists = center_dists - player_radius - blob_rads
    edge_dists[edge_dists < 1.0] = 1.0

    # Individual blob can directly eat us
    individually_dangerous = blob_rads > player_radius * 1.1

    # Enemy as a whole is dangerous, but only treat as danger if close
    future_dangerous = (potential_rads > player_radius * 1.15) & (edge_dists < player_radius * 5.0)
    merged_dangerous = (merged_rads > player_radius * 1.25) & (edge_dists < player_radius * 4.0)

    bigger = individually_dangerous | merged_dangerous | future_dangerous

    # Hunt small blobs unless the enemy's total visible mass is way too large
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

    danger_size = np.maximum.reduce([blob_rads, merged_rads, potential_rads])

    danger_scores = danger_weight * (danger_size / player_radius) / (edge_dists ** 2)
    hunt_scores = hunt_weight * (player_radius / blob_rads) / (edge_dists ** 2)

    score = 0.0
    score -= np.sum(danger_scores[bigger])
    score += np.sum(hunt_scores[smaller])

    return float(score)

def close_kill_direction(cache, player):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) == 0:
        return None

    player_pos = np.array([player.x, player.y], dtype=float)
    player_radius = player.radius

    vectors = blob_locs - player_pos
    dists = np.linalg.norm(vectors, axis=1)
    dists[dists < 1.0] = 1.0

    edge_dists = dists - player_radius - blob_rads

    can_eat = player_radius > blob_rads * 1.12
    close = edge_dists < player_radius * 2.6

    targets = can_eat & close

    if not np.any(targets):
        return None

    scores = blob_rads / dists
    scores[~targets] = -1.0

    idx = int(np.argmax(scores))

    direction = vectors[idx] / dists[idx]
    return float(direction[0]), float(direction[1])

def enemy_split_threat_score(cache, player, future_x, future_y):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) == 0:
        return 0.0

    player_radius = player.radius

    current_pos = np.array([player.x, player.y], dtype=float)
    future_pos = np.array([future_x, future_y], dtype=float)

    enemy_split_rads = blob_rads / np.sqrt(2.0)

    # Only care about enemies whose split piece can eat us
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

    # Are we staying in the direction they would naturally split toward?
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


def virus_score(cache, player, future_x, future_y, danger_weight, safety_weight):
    future_pos = np.array([future_x, future_y], dtype=float)

    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if len(virus_locs) == 0:
        return 0.0

    player_radius = player.radius

    center_dists = np.linalg.norm(virus_locs - future_pos, axis=1)
    edge_dists = center_dists - player_radius - virus_rads

    dangerous = player_radius > virus_rads * 1.1
    safe = ~dangerous

    # hard reject if we are basically colliding with a dangerous virus
    if np.any(dangerous & (edge_dists < player_radius * 0.25)):
        return -OFF_MAP_PENALTY

    # do not clamp before the hard collision check
    edge_dists[edge_dists < 1.0] = 1.0

    danger_scores = danger_weight / (edge_dists ** 3)
    safety_scores = safety_weight / (edge_dists ** 2)

    score = 0.0
    score -= np.sum(danger_scores[dangerous])
    score += np.sum(safety_scores[safe])

    return float(score)

def wall_score(player, x, y):
    left = x - player.radius
    right = ARENA_SIZE - x - player.radius
    bottom = y - player.radius
    top = ARENA_SIZE - y - player.radius

    nearest_wall = min(left, right, bottom, top)

    if nearest_wall <= 0:
        return -OFF_MAP_PENALTY

    safe_margin = WALL_MARGIN_MULT * player.radius

    if nearest_wall >= safe_margin:
        return 0.0

    closeness = (safe_margin - nearest_wall) / safe_margin

    return -WALL_DANGER_WEIGHT * (closeness ** 3)

def multi_blob_safety_score(cache, player, future_x, future_y):
    own_locs = cache["own_locs"]
    own_rads = cache["own_rads"]

    if len(own_locs) <= 1:
        return 0.0

    current_pos = np.array([player.x, player.y], dtype=float)
    future_pos = np.array([future_x, future_y], dtype=float)

    # approximate all own blobs moving by the same chosen direction
    delta = future_pos - current_pos
    future_own_locs = own_locs + delta

    score = 0.0

    # virus safety for each own blob
    virus_locs = cache["virus_locs"]
    virus_rads = cache["virus_rads"]

    if len(virus_locs) > 0:
        diffs = future_own_locs[:, None, :] - virus_locs[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)

        edge_dists = dists - own_rads[:, None] - virus_rads[None, :]
        dangerous = own_rads[:, None] > virus_rads[None, :] * 1.1

        if np.any(dangerous & (edge_dists < own_rads[:, None] * 0.6)):
            return -OFF_MAP_PENALTY

        edge_dists[edge_dists < 1.0] = 1.0
        score -= float(np.sum(VIRUS_DANGER_WEIGHT / (edge_dists[dangerous] ** 3)))

    # enemy safety for each own blob
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) > 0:
        diffs = future_own_locs[:, None, :] - blob_locs[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)

        edge_dists = dists - own_rads[:, None] - blob_rads[None, :]
        edge_dists[edge_dists < 1.0] = 1.0

        dangerous = blob_rads[None, :] > own_rads[:, None] * 1.1

        score -= float(np.sum(
            ENEMY_DANGER_WEIGHT *
            (blob_rads[None, :] / own_rads[:, None]) /
            (edge_dists ** 2) *
            dangerous
        ))

    return score

'''
Split functions: get_split_decision -> (yes/no, direction), split_penalty (if too dangerous)
'''

def split_penalty(cache, player, split_radius):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]

    if len(blob_locs) == 0:
        return False

    player_pos = np.array([player.x, player.y], dtype=float)

    danger_range = player.radius * POST_SPLIT_DANGER_RANGE_MULT

    dists = np.linalg.norm(blob_locs - player_pos, axis=1)
    edge_dists = dists - split_radius - blob_rads

    can_eat_split_piece = blob_rads > split_radius * SPLIT_EAT_RATIO
    too_close = edge_dists < danger_range

    return bool(np.any(can_eat_split_piece & too_close))

def get_split_decision(game, move_direction, cache):
    player = game.state.me

    if player.radius < SPLIT_MIN_RADIUS:
        return False, None

    #only split if we are a single blob
    if len(player.blobs) != 1: 
        return False, None

    blobs = game.state.visible_blobs

    #if there are no visible blobs
    if not blobs: 
        return False, None

    player_pos = np.array([player.x, player.y], dtype=float)

    move_direction = np.array(move_direction, dtype=float)
    move_norm = np.linalg.norm(move_direction)

    if np.isclose(move_norm, 0.0):
        return False, None

    move_direction = move_direction / move_norm

    split_radius = player.radius / np.sqrt(2)
    split_range = player.radius * SPLIT_RANGE_MULT

    #do not split if splitting makes us vulnerable
    if split_penalty(cache, player, split_radius):
        return False, None

    best_target = None
    best_target_size = -1

    #for each blob evaluate if its worth it to split
    for blob in blobs:
        blob_pos = np.array(blob.pos, dtype=float)
        blob_vector = blob_pos - player_pos
        dist = np.linalg.norm(blob_vector)

        if np.isclose(dist, 0.0):
            continue

        blob_uv = blob_vector / dist
        edge_dist = dist - player.radius - blob.radius

        alignment = np.dot(move_direction, blob_uv)

        #pass of any of these conditions
        if (
            alignment < SPLIT_ALIGNMENT_MIN or
            split_radius <= blob.radius * SPLIT_EAT_RATIO or
            edge_dist > split_range   
        ):
            continue

        virus_locs = cache["virus_locs"]
        virus_rads = cache["virus_rads"]

        if len(virus_locs) > 0:
            split_landing = player_pos + blob_uv * split_range

            dists_to_viruses = np.linalg.norm(virus_locs - split_landing, axis=1)
            edge_to_viruses = dists_to_viruses - split_radius - virus_rads

            dangerous_viruses = split_radius > virus_rads * 1.1

            if np.any(dangerous_viruses & (edge_to_viruses < split_radius * 0.4)):
                continue

        #pick best split
        if blob.radius > best_target_size:
            best_target_size = blob.radius
            best_target = blob_uv

    if best_target is None:
            return False, None

    return True, best_target


def score_position(player, cache, weights, x, y):
    score = 0.0

    score += food_score(
        cache,
        x,
        y,
        weight=weights["food"]
    )

    score += enemy_score(
        cache,
        player,
        x,
        y,
        danger_weight=weights["enemy_danger"],
        hunt_weight=weights["enemy_hunt"],
        hunt_ratio=weights["hunt_ratio"],
        merged_safe_ratio=weights["merged_safe_ratio"]
    )

    score += virus_score(
        cache,
        player,
        x,
        y,
        danger_weight=VIRUS_DANGER_WEIGHT,
        safety_weight=VIRUS_SAFE_WEIGHT
    )

    score += wall_score(
        player, 
        x, 
        y
    )

    score += multi_blob_safety_score(
        cache, 
        player, 
        x, 
        y
    )

    return score 

def build_cache(game):
    # --- OWN BLOBS ---
    own_blobs = list(game.state.me.blobs.values())

    if own_blobs:
        own_locs = np.array([blob.pos for blob in own_blobs], dtype=float)
        own_rads = np.array([blob.radius for blob in own_blobs], dtype=float)
    else:
        own_locs = np.array([[game.state.me.x, game.state.me.y]], dtype=float)
        own_rads = np.array([game.state.me.radius], dtype=float)

    # --- FOOD ---
    foods = game.state.visible_food

    if foods:
        food_locs = np.array([food.pos for food in foods], dtype=float)
    else:
        food_locs = np.empty((0, 2), dtype=float)
    
    if len(food_locs) > MAX_FOOD_CONSIDERED:
        me = game.state.me
        player_pos = np.array([me.x, me.y], dtype=float)

        dists = np.sum((food_locs - player_pos) ** 2, axis=1)
        keep = np.argpartition(dists, MAX_FOOD_CONSIDERED - 1)[:MAX_FOOD_CONSIDERED]
        food_locs = food_locs[keep]

    # --- ENEMY BLOBS ---
    blobs = game.state.visible_blobs

    if blobs:
        blob_locs = np.array([blob.pos for blob in blobs], dtype=float)
        blob_rads = np.array([blob.radius for blob in blobs], dtype=float)
        blob_player_ids = np.array([blob.player_id for blob in blobs], dtype=int)

        merged_rads = np.zeros(len(blob_rads), dtype=float)
        for pid in np.unique(blob_player_ids):
            same_player = blob_player_ids == pid
            total_mass = np.sum(blob_rads[same_player] ** 2)
            merged_radius = np.sqrt(total_mass)
            merged_rads[same_player] = merged_radius

        potential_rads = blob_rads.copy()

        if len(blob_rads) <= MAX_BLOBS_FOR_PREDICTION:
            for i in range(len(blob_rads)):
                predator_pos = blob_locs[i]
                predator_rad = blob_rads[i]
                predator_pid = blob_player_ids[i]

                vectors = blob_locs - predator_pos
                dists = np.linalg.norm(vectors, axis=1)
                edge_dists = dists - predator_rad - blob_rads

                different_player = blob_player_ids != predator_pid
                can_eat = predator_rad > blob_rads * 1.12
                very_close = edge_dists < predator_rad * 1.2

                possible_prey = different_player & can_eat & very_close

                if np.any(possible_prey):
                    prey_mass = np.max(blob_rads[possible_prey] ** 2)
                    potential_rads[i] = np.sqrt(predator_rad ** 2 + prey_mass)

    else:
        blob_locs = np.empty((0, 2), dtype=float)
        blob_rads = np.empty(0, dtype=float)
        blob_player_ids = np.empty(0, dtype=int)
        merged_rads = np.empty(0, dtype=float)
        potential_rads = np.empty(0, dtype=float)

    # --- VIRUSES ---
    viruses = game.state.visible_viruses

    if viruses:
        virus_locs = np.array([virus.pos for virus in viruses], dtype=float)
        virus_rads = np.array([virus.radius for virus in viruses], dtype=float)
    else:
        virus_locs = np.empty((0, 2), dtype=float)
        virus_rads = np.empty(0, dtype=float)

    return {
        "food_locs": food_locs,
        "blob_locs": blob_locs,
        "blob_rads": blob_rads,
        "blob_player_ids": blob_player_ids,
        "merged_rads": merged_rads,
        "virus_locs": virus_locs,
        "virus_rads": virus_rads,
        "potential_rads": potential_rads,
        "own_locs": own_locs,
        "own_rads": own_rads
    }

def get_mode_weights(player):
    r = player.radius

    # Tiny/start: farm food, do not chase unless it is a free close kill
    if r < 1.2:
        return {
            "food": 1600.0,
            "enemy_danger": 7000.0,
            "enemy_hunt": 800.0,
            "hunt_ratio": 0.55,
            "merged_safe_ratio": 0.80,
        }

    # Small but viable: now start hunting aggressively
    if r < 1.7:
        return {
            "food": 1200.0,
            "enemy_danger": 5500.0,
            "enemy_hunt": 4500.0,
            "hunt_ratio": 0.95,
            "merged_safe_ratio": 1.30,
        }

    # Medium
    if r < 2.5:
        return {
            "food": 900.0,
            "enemy_danger": 8500.0,
            "enemy_hunt": 3000.0,
            "hunt_ratio": 0.85,
            "merged_safe_ratio": 1.15,
        }

    # Big
    return {
        "food": 700.0,
        "enemy_danger": 11000.0,
        "enemy_hunt": 1500.0,
        "hunt_ratio": 0.65,
        "merged_safe_ratio": 0.90,
    }

def enemy_escape_direction(cache, player, step_distance):
    blob_locs = cache["blob_locs"]
    blob_rads = cache["blob_rads"]
    merged_rads = cache["merged_rads"]
    potential_rads = cache["potential_rads"]

    if len(blob_locs) == 0:
        return None

    player_pos = np.array([player.x, player.y], dtype=float)
    player_radius = player.radius

    dists = np.linalg.norm(blob_locs - player_pos, axis=1)
    edge_dists = dists - player_radius - blob_rads

    danger_size = np.maximum.reduce([blob_rads, merged_rads, potential_rads])

    dangerous = danger_size > player_radius * 1.10
    too_close = edge_dists < player_radius * 2.5

    if not np.any(dangerous & too_close):
        return None

    best_score = -float("inf")
    best_direction = None

    for direction in DIRECTIONS:
        dx, dy = direction
        future_x = player.x + dx * step_distance
        future_y = player.y + dy * step_distance

        score = 0.0

        score += enemy_score(
            cache,
            player,
            future_x,
            future_y,
            danger_weight=ENEMY_DANGER_WEIGHT,
            hunt_weight=0.0,
            hunt_ratio=0.0,
            merged_safe_ratio=0.0,
        )

        score += enemy_split_threat_score(cache, player, future_x, future_y)
        score += virus_score(cache, player, future_x, future_y, VIRUS_DANGER_WEIGHT, VIRUS_SAFE_WEIGHT)
        score += wall_score(player, future_x, future_y)

        if score > best_score:
            best_score = score
            best_direction = direction

    if best_direction is None:
        return None

    return float(best_direction[0]), float(best_direction[1])

def override_direction(cache, player, step_distance):
    # ---------- ENEMY ESCAPE OVERRIDE ----------
    escape_dir = enemy_escape_direction(cache, player, step_distance)

    if escape_dir is not None:
        return escape_dir
    
    # ---------- CLOSE KILL OVERRIDE ----------
    # If a target is very close and edible, chase it before thinking about food.
    kill_dir = close_kill_direction(cache, player)

    if kill_dir is not None:
        dx, dy = kill_dir
        future_x = player.x + dx * step_distance
        future_y = player.y + dy * step_distance

        if (
            enemy_split_threat_score(cache, player, future_x, future_y) > -20000 and
            virus_score(cache, player, future_x, future_y, VIRUS_DANGER_WEIGHT, VIRUS_SAFE_WEIGHT) > -OFF_MAP_PENALTY / 2
        ):
            return dx, dy

    # ---------- FOOD OVERRIDE ----------
    # Only use pure food direction when there are no threats/targets around.
    if len(cache["blob_locs"]) == 0 and len(cache["virus_locs"]) == 0:
        food_dir = food_direction(cache, player)

        if food_dir is not None:
            dx, dy = food_dir
            future_x = player.x + dx * step_distance
            future_y = player.y + dy * step_distance

            return dx, dy

    return None

def choose_direction(game: Game):
    global LAST_DIRECTION

    player = game.state.me

    step_distance = STEP_DISTANCE_MULT * player.radius
    cache = build_cache(game)
    weights = get_mode_weights(player)

    override_dir = override_direction(cache, player, step_distance)

    if override_dir is not None:
        dx, dy = override_dir
        LAST_DIRECTION = np.array([dx, dy], dtype=float)
        return dx, dy, cache

    #(total_score, x, y, first_direction, previous_direction)
    beam = [
        (
            0.0,
            player.x,
            player.y,
            None,
            LAST_DIRECTION,
        )
    ]

    #MAIN BEAM SEARCH
    for step in range(1, ROLLOUT_STEPS + 1):
        new_beam = []

        discount = ROLLOUT_DISCOUNT ** (step - 1)

        for current_score, x, y, first_direction, previous_direction in beam:
            for direction in DIRECTIONS:
                dx, dy = direction

                future_x = x + dx * step_distance
                future_y = y + dy * step_distance

                position_score = score_position(player, cache, weights, future_x, future_y)

                if step == 1:
                    position_score += enemy_split_threat_score(
                        cache,
                        player,
                        future_x,
                        future_y
                    )

                total_score = current_score + discount * position_score

                turn_alignment = np.dot(direction, previous_direction)
                turn_penalty = TURN_PENALTY_WEIGHT * (1.0 - turn_alignment)
                total_score -= turn_penalty

                if step == 1:
                    if wall_score(player, future_x, future_y) > -OFF_MAP_PENALTY / 2:
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

        #keep best candidates
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

    
def main() -> None:
    game = Game()

    while True:
        query = game.get_next_query()

        match query:
            case QueryMovePlayer():
                try:
                    dx, dy, cache = choose_direction(game)

                    should_do_split, split_direction = get_split_decision(game, (dx, dy), cache)

                    if should_do_split:
                        sx, sy = split_direction
                        game.send_move(
                            MovePlayer(
                                player_id=game.state.me.player_id,
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
                                player_id=game.state.me.player_id,
                                direction=DirectionModel(
                                    x=float(dx),
                                    y=float(dy),
                                ),
                                split=False,
                            )
                        )

                except Exception:
                    # emergency fallback so we still send a move
                    game.send_move(
                        MovePlayer(
                            player_id=game.state.me.player_id,
                            direction=DirectionModel(x=1.0, y=0.0),
                            split=False,
                        )
                    )

            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")

if __name__ == "__main__":
    main()
