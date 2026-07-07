from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

import numpy as np


'''
    Vector functions: food_vector, enemy_vector
'''

def food_vector(game, weight, player):
    foods = game.state.visible_food

    if not foods:
        return np.array([0.0, 0.0])

    #obtain food/player arrays
    food_locs = np.array([food.pos for food in foods], dtype=float)
    player_loc = np.array([player.x, player.y], dtype=float)

    #obtain vectors and distance
    vectors = food_locs - player_loc
    dists = np.linalg.norm(vectors, axis=1)
    dists[dists == 0] = 1
    unit_vectors = vectors / dists.reshape(-1, 1)
    
    #calculate attraction score
    attractions = weight/(dists**2)
    
    #calculate scores and return vector sum
    food_forces = unit_vectors * attractions.reshape(-1, 1)

    return food_forces.sum(axis=0)


def enemy_vector(game, danger_weight, hunt_weight, player):
    blobs = game.state.visible_blobs

    if not blobs:
        return np.array([0.0, 0.0])

    blob_locs = np.array([blob.pos for blob in blobs], dtype=float)
    blob_radii = np.array([blob.radius for blob in blobs], dtype=float)
    player_loc = np.array([player.x, player.y], dtype=float)

    #vector from us to enemy
    vectors_to_enemy = blob_locs - player_loc
    dists = np.linalg.norm(vectors_to_enemy, axis=1)
    dists[dists == 0] = 1

    unit_to_enemy = vectors_to_enemy / dists.reshape(-1, 1)

    #filter conditions for bigger vs smaller enemies
    bigger = blob_radii > player.radius * 1.1
    smaller = blob_radii < player.radius * 0.8

    forces = np.zeros_like(unit_to_enemy)

    danger = danger_weight * (blob_radii / player.radius) / (dists ** 2)
    hunt = hunt_weight * (player.radius / blob_radii) / (dists ** 2)

    forces[bigger] = -unit_to_enemy[bigger] * danger[bigger].reshape(-1, 1)
    forces[smaller] = unit_to_enemy[smaller] * hunt[smaller].reshape(-1, 1)

    return forces.sum(axis=0)

def virus_vector(game, danger_weight, safety_weight, player):
    pass



def choose_direction(game: Game) -> tuple[float, float]:
    me = game.state.me

    food_force = food_vector(game, weight=1000, player=me)
    enemy_force = enemy_vector(game, danger_weight=3000, hunt_weight=1500, player=me)
    
    dx = food_force[0] + enemy_force[0] 
    dy = food_force[1] + enemy_force[1] 

    if dx == 0 and dy == 0:
        return (1.0, 0.0)
    
    return (dx, dy)


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
