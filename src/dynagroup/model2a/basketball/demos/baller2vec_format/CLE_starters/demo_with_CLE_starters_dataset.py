from dynagroup.model2a.basketball.data.baller2vec.CLE_starters_dataset import (
    get_basketball_games_for_CLE_dataset,
)


games = get_basketball_games_for_CLE_dataset()
plays_per_game = [len(game.events) for game in games]
print(f"The plays per game are {plays_per_game}.")
