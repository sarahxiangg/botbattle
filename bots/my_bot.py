from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

import numpy as np

# =========================
# Direction search
# =========================

NUM_DIRECTIONS = 16              # directions tested in search

# =========================
# Scoring weights
# =========================

FOOD_WEIGHT = 1000.0             # food attraction
ENEMY_DANGER_WEIGHT = 8000.0     # avoid bigger enemies
ENEMY_HUNT_WEIGHT = 2500.0       # chase smaller enemies
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

ROLLOUT_STEPS = 3                 # future steps checked
ROLLOUT_DISCOUNT = 0.7            # future score discount
STEP_DISTANCE_MULT = 1.5          # step size multiplier
BEAM_WIDTH = 6                    # paths kept per step
TURN_PENALTY_WEIGHT = 150.0       # discourage sharp turns


# =========================
# Splitting
# =========================

SPLIT_EAT_RATIO = 1.20              # required size advantage
SPLIT_RANGE_MULT = 4.0              # split reach estimate
SPLIT_ALIGNMENT_MIN = 0.85          # must face target
POST_SPLIT_DANGER_RANGE_MULT = 6.0  # danger scan range

'''
    Vector functions: food_score, enemy_score
'''

def food_score(game, player, future_x, future_y, weight):
    foods = game.state.visible_food

    if not foods:
        return 0.0

    future_pos = np.array([future_x, future_y], dtype=float)
    food_locs = np.array([food.pos for food in foods], dtype=float)
    
    dists = np.linalg.norm(food_locs - future_pos, axis=1)
    dists[dists == 0] = 1.0
    scores = weight / (dists**2)

    return float(np.sum(scores))


def enemy_score(game, player, future_x, future_y, danger_weight, hunt_weight):
    blobs = game.state.visible_blobs

    if not blobs:
        return 0.0

    future_pos = np.array([future_x, future_y], dtype=float)

    blob_locs = np.array([blob.pos for blob in blobs], dtype=float)
    blob_rads = np.array([blob.radius for blob in blobs], dtype=float)

    player_radius = player.radius

    center_dists = np.linalg.norm(blob_locs - future_pos, axis=1)

    edge_dists = center_dists - player_radius - blob_rads
    edge_dists[edge_dists < 1.0] = 1.0

    bigger = blob_rads > player_radius * 1.1
    smaller = blob_rads < player_radius * 0.8

    danger_scores = danger_weight * (blob_rads / player_radius) / (edge_dists ** 2)
    hunt_scores = hunt_weight * (player_radius / blob_rads) / (edge_dists ** 2)

    score = 0.0
    score -= np.sum(danger_scores[bigger])
    score += np.sum(hunt_scores[smaller])

    return float(score)

def split_would_be_dangerous(game, split_radius):
    player = game.state.me
    player_pos = np.array([player.x, player.y], dtype=float)

    danger_range = player.radius * POST_SPLIT_DANGER_RANGE_MULT

    for blob in game.state.visible_blobs:
        blob_pos = np.array(blob.pos, dtype=float)

        # Can this enemy eat our split piece?
        if blob.radius <= split_radius * SPLIT_EAT_RATIO:
            continue

        dist = np.linalg.norm(blob_pos - player_pos)
        edge_dist = dist - split_radius - blob.radius

        if edge_dist < danger_range:
            return True

    return False

def get_split_decision(game, move_direction):
    player = game.state.me

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
    if split_would_be_dangerous(game, split_radius):
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


        #pick best split
        if blob.radius > best_target_size:
            best_target_size = blob.radius
            best_target = blob_uv

    if best_target is None:
            return False, None

    return True, best_target


def virus_score(game, player, future_x, future_y, danger_weight, safety_weight):
    viruses = game.state.visible_viruses

    if not viruses:
        return 0.0

    future_pos = np.array([future_x, future_y], dtype=float)

    virus_locs = np.array([virus.pos for virus in viruses], dtype=float)
    virus_rads = np.array([virus.radius for virus in viruses], dtype=float)

    player_radius = player.radius

    center_dists = np.linalg.norm(virus_locs - future_pos, axis=1)

    edge_dists = center_dists - player_radius - virus_rads
    edge_dists[edge_dists < 1.0] = 1.0

    dangerous = player_radius > virus_rads * 1.1
    safe = ~dangerous

    danger_scores = danger_weight / (edge_dists ** 3)
    safety_scores = safety_weight / (edge_dists ** 2)

    score = 0.0
    score -= np.sum(danger_scores[dangerous])
    score += np.sum(safety_scores[safe])

    return float(score)

def score_position(game, player, x, y):
    score = 0.0

    score += food_score(
        game,
        player,
        x,
        y,
        weight=FOOD_WEIGHT
    )

    score += enemy_score(
        game,
        player,
        x,
        y,
        danger_weight=ENEMY_DANGER_WEIGHT,
        hunt_weight=ENEMY_HUNT_WEIGHT
    )

    score += virus_score(
        game,
        player,
        x,
        y,
        danger_weight=VIRUS_DANGER_WEIGHT,
        safety_weight=VIRUS_SAFE_WEIGHT
    )

    return score

def choose_direction(game: Game) -> tuple[float, float]:
    global LAST_DIRECTION

    player = game.state.me

    step_distance = STEP_DISTANCE_MULT * player.radius

    directions = []

    for i in range(NUM_DIRECTIONS):
        angle = 2 * np.pi * i / NUM_DIRECTIONS
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        directions.append(direction)

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

    for step in range(1, ROLLOUT_STEPS + 1):
        new_beam = []

        discount = ROLLOUT_DISCOUNT ** (step - 1)

        for current_score, x, y, first_direction, previous_direction in beam:
            for direction in directions:
                dx, dy = direction

                future_x = x + dx * step_distance
                future_y = y + dy * step_distance

                position_score = score_position(
                    game,
                    player,
                    future_x,
                    future_y
                )

                total_score = current_score + discount * position_score

                turn_alignment = np.dot(direction, previous_direction)
                turn_penalty = TURN_PENALTY_WEIGHT * (1.0 - turn_alignment)
                total_score -= turn_penalty

                if step == 1:
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

    return float(final_direction[0]), float(final_direction[1])


def main() -> None:
    game = Game()

    while True:
        query = game.get_next_query()
        match query:
            case QueryMovePlayer():
                dx, dy = choose_direction(game)

                should_do_split, split_direction = get_split_decision(game, (dx, dy))

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

            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")

if __name__ == "__main__":
    main()
