from RandomEdgeMarkov import GaussianEdgeMarkov as gev
import pandas as pd
import numpy as np
from sportsreference.ncaab.teams import Teams
from sportsreference.ncaab.schedule import Game
from sportsreference.ncaab.boxscore import Boxscores

teams = Teams()
team_names = [team.name for team in teams]
# could do ols to get a real estimate for this; just using estimate from other analysis rn
hca = 1.25

def margin_f(margin, k=.03):
    return 1 / (1 + np.exp(-k * margin))

def get_games_played(data):
    res = {name: 0 for name in team_names}
    for idx, game_data in data.iterrows():
        res[game_data['team_name']] += 1
        res[game_data['opponent_name']] += 1
    return res

def load_data(use_cache=False):
    if use_cache:
        try:
            return pd.read_csv('data/games.csv')
        except FileNotFoundError:
            print('File not found, loading from scratch')
            return pd.DataFrame()
    else:
        print('Not using cache, starting from scratch')
        return pd.DataFrame()

def update_data(data):
    print(data.head())
    # keep track of which boxscore indices have been counted
    print('Updating data')
    data = data.to_dict(orient='list')
    tracked_games = data.index.values.tolist() + [None]
    for team in teams:
        team_name = team.name
        for game in team.schedule:
            if game.boxscore_index not in tracked_games:
                if game.points_for is None: # game has not been played yet
                    continue
                if game.opponent_name not in team_names:
                    continue
                else:
                    data[game.boxscore_index] = [game.boxscore_index, team_name, game.opponent_name, game.points_for, game.points_against, game.location, game.datetime]
                print('New game: ', game.boxscore_index, data[game.boxscore_index])
                tracked_games.append(game.boxscore_index)
    return pd.DataFrame.from_dict(data, orient='index', columns=['index', 'team_name', 'opponent_name', 'points_for', 'points_against', 'location', 'datetime'])

def get_adjacency_matrix(data):
    mat = pd.DataFrame(np.zeros((len(team_names), len(team_names))), index=team_names, columns=team_names)
    mat = gev(mat)
    tracked_games = data.index.tolist()
    for g in tracked_games:
        if g is None:
            continue
        game_data = data.loc[g,]
        margin = game_data['points_for'] - game_data['points_against']
        if game_data['location'] == 'Home':
            margin -= hca
        elif game_data['location'] == 'Away':
            margin += hca
        # else margin is neutral and we do not adjust
        margin_val = margin_f(margin)
        # TODO: how to choose sigma
        sigma = .2
        team_name = game_data['team_name']
        opp_name = game_data['opponent_name']
        if mat.exists_edge(team_name, opp_name):
            # im gonna have to handle these differently im just not ready yet
            mat.add_to_edge(team_name, opp_name, margin_val, sigma)
            mat.add_to_edge(opp_name, team_name, 1 - margin_val, sigma)
        else: 
            mat.add_edge(team_name, opp_name, margin_val, sigma)
            mat.add_edge(opp_name, team_name, 1 - margin_val, sigma)
    return mat

def normalize_row_el(el, games_played):
    if el == 0:
        return 0
    else:
        return (el[0] / games_played, el[1]/ np.sqrt(games_played))

def normalize_mat(mat, games_played_counter):
    # TODO: change notation; mat should just be the matrix in the graph object
    # mat = np.transpose(mat.mat.to_numpy())
    mat = mat.mat.to_numpy()
    new_mat = np.array(mat.shape)
    for i in range(len(mat)):
        # when we scale down by a factor k, variance of Gaussian changes by factor of sqrt(k)
        team_name = team_names[i]
        # a little hacky but had to create list of same int for map() to work
        new_row = list(map(normalize_row_el, mat[i].tolist(), [games_played_counter[team_name] for _ in range(len(mat))]))
        new_mat = np.append(new_mat, new_row, axis=0)
    mat = np.transpose(mat)
    mat = gev(pd.DataFrame(mat, index=team_names, columns=team_names))
    return mat

def get_rankings(mat, games_played_counter):
    mat = normalize_mat(mat, games_played_counter)
    r = mat.mc_stationary_dist(n=100)
    return r

def write_games_to_csv(data):
    data.to_csv('data/games.csv')

def display_rankings(vec, teams_list):
    sorted_indices = vec.argsort()
    # This is arbitrary to turn the rankings nicer looking, trying to reproduce a normal qq distribution
    ranked = [(teams_list[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    ranked = {i + 1 : ranked[i] for i in range(len(ranked))}
    for key, value in ranked.items():
        print(key, value[0], round(value[1]*100, 2))

def main():
    data = load_data(use_cache=True)
    # data = update_data(data)
    write_games_to_csv(data)
    data.set_index('index', inplace=True)
    mat = get_adjacency_matrix(data)
    games_played_counter = get_games_played(data)
    r = get_rankings(mat, games_played_counter)
    display_rankings(r, team_names)

if __name__ == '__main__':
    main()

