from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

import numpy as np

NUM_DIRECTIONS = 16

FOOD_WEIGHT = 1000.0
ENEMY_DANGER_WEIGHT = 8000.0
ENEMY_HUNT_WEIGHT = 2500.0
VIRUS_DANGER_WEIGHT = 20000.0
VIRUS_SAFE_WEIGHT = 200.0

SMOOTHING = 0.65
LAST_DIRECTION = np.array([1.0, 0.0], dtype=float)
STICKINESS_WEIGHT = 300.0



'''
    Vector functions: food_vector, enemy_vector
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

def virus_score(game, player, future_x, future_y, danger_weight, safety_weight):
    viruses = game.state.visible_viruses

    if not viruses:
        return 0.0

    future_pos = np.array([future_x, future_y], dtype=float)

    virus_locs = np.array([virus.pos for virus in viruses], dtype=float)
    virus_rads = np.array([virus.radius for virus in viruses], dtype=float)

    player_radius = player.radius
    player_mass = player_radius ** 2

    center_dists = np.linalg.norm(virus_locs - future_pos, axis=1)

    edge_dists = center_dists - player_radius - virus_rads
    edge_dists[edge_dists < 1.0] = 1.0

    dangerous = player_mass > virus_rads * 1.2
    safe = ~dangerous

    danger_scores = danger_weight / (edge_dists ** 4)
    safety_scores = safety_weight / (edge_dists ** 2)

    score = 0.0
    score -= np.sum(danger_scores[dangerous])
    score += np.sum(safety_scores[safe])

    return float(score)

def choose_direction(game: Game) -> tuple[float, float]:
    global LAST_DIRECTION

    player = game.state.me

    best_score = -float("inf")
    best_direction = np.array([1.0, 0.0], dtype=float)

    lookahead_distance = 2 * player.radius

    for i in range(NUM_DIRECTIONS):
        angle = 2 * np.pi * i / NUM_DIRECTIONS

        direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)
        dx, dy = direction

        future_x = player.x + dx * lookahead_distance
        future_y = player.y + dy * lookahead_distance

        score = 0.0

        score += food_score(
            game, 
            player, 
            future_x, 
            future_y, 
            weight=FOOD_WEIGHT
        )

        score += enemy_score(
            game, 
            player, 
            future_x, 
            future_y, 
            danger_weight=ENEMY_DANGER_WEIGHT, 
            hunt_weight=ENEMY_HUNT_WEIGHT
        )

        score += virus_score(
            game, 
            player, 
            future_x, 
            future_y, 
            danger_weight=VIRUS_DANGER_WEIGHT,
            safety_weight=VIRUS_SAFE_WEIGHT 
        )

        #smoothing
        score += STICKINESS_WEIGHT * np.dot(direction, LAST_DIRECTION)

        if score > best_score:
            best_score = score
            best_direction = direction
        
    final_direction = SMOOTHING * LAST_DIRECTION + (1 - SMOOTHING) * best_direction
    final_direction = final_direction / np.linalg.norm(final_direction)

    LAST_DIRECTION = final_direction

    return float(final_direction[0]), float(final_direction[1])


def main() -> None:
    game = Game()

    while True:
        query = game.get_next_query()
        match query:
            case QueryMovePlayer():
                dx, dy = choose_direction(game)
                game.send_move(
                    MovePlayer(
                        player_id=game.state.me.player_id,
                        direction=DirectionModel(x=dx, y=dy),
                    )
                )
            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()
