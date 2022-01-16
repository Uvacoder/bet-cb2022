from re import L
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

# TODO: pull from other file
SIGMA = 15

def margin_f(margin, k=.05):
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
    # keep track of which boxscore indices have been counted
    print('Updating data')
    data = data.to_dict(orient='list')
    tracked_games = list(data.keys()) + [None]
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
    for key, value in data.items():
        print(key, value)
    return pd.DataFrame.from_dict(data, orient='index', columns=['index', 'team_name', 'opponent_name', 'points_for', 'points_against', 'location', 'datetime'])

def get_adjacency_matrix(data):
    mat = pd.DataFrame(np.zeros((len(team_names), len(team_names))), index=team_names, columns=team_names)
    graph = gev(mat)
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
        sigma = SIGMA
        team_name = game_data['team_name']
        opp_name = game_data['opponent_name']
        if graph.exists_edge(team_name, opp_name):
            graph.mat.loc[team_name, opp_name] += [margin]
            graph.mat.loc[opp_name, team_name] += [-margin]
        else: 
            graph.mat.loc[team_name, opp_name] = [margin]
            graph.mat.loc[opp_name, team_name] = [-margin]
    return graph

def normalize_row_el(el, games_played):
    return el / games_played

def adjust_margin(graph):
    dim = graph.mat.shape[0]
    mat = graph.mat
    for i in range(dim):
        for j in range(dim):
            if mat.iloc[i, j] == 0:
                continue
            mat.iloc[i, j] = margin_f(mat.iloc[i, j])
    return mat

def transform_mat(graph):
    # takes matrix of lists and turns it into matrix of tuples of form (mean, stdev)
    dim = graph.mat.shape[0]
    mat = graph.mat
    for i in range(dim):
        for j in range(dim):
            if mat.iloc[i, j] == 0:
                continue
            n = len(mat.iloc[i, j])
            stdev = SIGMA / np.sqrt(n)
            mean = np.mean(mat.iloc[i, j])
            mat.iloc[i, j] = (mean, stdev)
    graph = gev(mat)
    return graph

def get_sample(graph):
    sample = graph.monte_carlo_sample(n=100)
    res = []
    for s in sample:
        s = adjust_margin(s)
        res.append(s)
    return res
    
def normalize_mat(mat, games_played_counter):
    # TODO: change notation; mat should just be the matrix in the graph object
    mat = mat.to_numpy()
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
    graph = transform_mat(mat)
    sample = get_sample(graph)
    res = []
    counter = 1
    for s in sample:
        print(counter)
        counter += 1
        s = normalize_mat(s, games_played_counter)
        res.append(s)
    r = gev.mean_stationary_dist(res)
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
    # write_games_to_csv(data)
    data.set_index('index', inplace=True)
    mat = get_adjacency_matrix(data)
    games_played_counter = get_games_played(data)
    r = get_rankings(mat, games_played_counter)
    display_rankings(r, team_names)

if __name__ == '__main__':
    main()

