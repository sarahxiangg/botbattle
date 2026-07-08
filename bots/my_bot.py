from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

import numpy as np

NUM_DIRECTIONS = 16

FOOD_WEIGHT = 1000.0
ENEMY_DANGER_WEIGHT = 8000.0
ENEMY_HUNT_WEIGHT = 2500.0
VIRUS_DANGER_WEIGHT = 10000.0
VIRUS_SAFE_WEIGHT = 200.0

LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)
SWITCH_THRESHOLD = 75.0

SMOOTHING = 0.10
STICKINESS_WEIGHT = 300.0

ROLLOUT_STEPS = 3
ROLLOUT_DISCOUNT = 0.7
STEP_DISTANCE_MULT = 1.5
BEAM_WIDTH = 6
TURN_PENALTY_WEIGHT = 150.0

SPLIT_EAT_RATIO = 1.20
SPLIT_RANGE_MULT = 4.0
SPLIT_ALIGNMENT_MIN = 0.85

POST_SPLIT_DANGER_RANGE_MULT = 6.0
SPLIT_KILL_VALUE = 500.0
SPLIT_DANGER_WEIGHT = 5000.0
MIN_SPLIT_SCORE = 10000.04


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

def post_split_danger_penalty(game, player, split_radius, player_pos):
    blobs = game.state.visible_blobs

    if not blobs:
        return 0.0

    blob_locs = np.array([blob.pos for blob in blobs], dtype=float)
    blob_rads = np.array([blob.radius for blob in blobs], dtype=float)

    # enemies that can eat our split piece
    dangerous = blob_rads > split_radius * 1.15

    if not np.any(dangerous):
        return 0.0

    danger_locs = blob_locs[dangerous]
    danger_rads = blob_rads[dangerous]

    center_dists = np.linalg.norm(danger_locs - player_pos, axis=1)

    edge_dists = center_dists - split_radius - danger_rads
    edge_dists[edge_dists < 1.0] = 1.0

    danger_range = player.radius * POST_SPLIT_DANGER_RANGE_MULT
    close_danger = edge_dists < danger_range

    if not np.any(close_danger):
        return 0.0

    close_rads = danger_rads[close_danger]
    close_edge_dists = edge_dists[close_danger]

    danger_scores = (
        SPLIT_DANGER_WEIGHT
        * (close_rads / split_radius)
        / (close_edge_dists ** 2)
    )

    return float(np.sum(danger_scores))

def get_split_decision(game, move_direction):
    """
    Returns:
        (False, None) if we should not split.
        (True, split_direction) if we should split.

    split_direction is aimed directly at the best edible target.
    """
    player = game.state.me
    blobs = game.state.visible_blobs

    if not blobs:
        return False, None

    player_pos = np.array([player.x, player.y], dtype=float)

    move_direction = np.array(move_direction, dtype=float)
    move_norm = np.linalg.norm(move_direction)

    if move_norm == 0:
        return False, None

    move_direction = move_direction / move_norm

    blob_locs = np.array([blob.pos for blob in blobs], dtype=float)
    blob_rads = np.array([blob.radius for blob in blobs], dtype=float)

    split_radius = player.radius / np.sqrt(2)
    split_range = player.radius * SPLIT_RANGE_MULT

    # vectors from us to every visible blob
    to_blobs = blob_locs - player_pos
    center_dists = np.linalg.norm(to_blobs, axis=1)

    valid_dist = center_dists > 0

    safe_center_dists = center_dists.copy()
    safe_center_dists[safe_center_dists == 0] = 1.0

    to_blob_units = to_blobs / safe_center_dists.reshape(-1, 1)

    edge_dists = center_dists - player.radius - blob_rads

    # how much the target is in the same direction we are already moving
    alignments = to_blob_units @ move_direction

    in_front = alignments >= SPLIT_ALIGNMENT_MIN

    # split piece must still be clearly bigger than target
    can_eat = split_radius > blob_rads * SPLIT_EAT_RATIO

    # target must be within estimated split launch range
    in_range = edge_dists <= split_range

    eligible = valid_dist & in_front & can_eat & in_range

    if not np.any(eligible):
        return False, None

    danger_penalty = post_split_danger_penalty(
        game,
        player,
        split_radius,
        player_pos
    )

    target_values = (blob_rads ** 2) * SPLIT_KILL_VALUE
    split_scores = target_values - danger_penalty

    # ignore non-eligible targets
    split_scores[~eligible] = -float("inf")

    best_idx = int(np.argmax(split_scores))
    best_score = split_scores[best_idx]

    if best_score <= MIN_SPLIT_SCORE:
        return False, None

    best_split_direction = to_blob_units[best_idx]

    return True, best_split_direction


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

    danger_scores = danger_weight / (edge_dists ** 4)
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
