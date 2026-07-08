from helper.game import Game
from lib.interface.events.moves.move_player import MovePlayer
from lib.interface.queries.query_move import QueryMovePlayer
from lib.models.penguin_model import DirectionModel

import math
import random


def choose_direction(game: Game) -> tuple[float, float]:
    me = game.state.me

    if game.state.visible_food:
        target = min(
            game.state.visible_food,
            key=lambda food: (food.pos[0] - me.x) ** 2 + (food.pos[1] - me.y) ** 2,
        )

        dx = target.pos[0] - me.x
        dy = target.pos[1] - me.y
    else:
        # fallback random direction if no food is visible
        angle = random.random() * 2 * math.pi
        dx = math.cos(angle)
        dy = math.sin(angle)

    norm = math.sqrt(dx * dx + dy * dy)

    if norm == 0:
        return 1.0, 0.0

    return dx / norm, dy / norm


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
                        split=False,
                    )
                )

            case _:
                raise RuntimeError(f"Unsupported query type: {type(query)}")


if __name__ == "__main__":
    main()