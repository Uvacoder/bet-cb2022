from re import L
from RandomEdgeMarkov import GaussianEdgeMarkov as gev
import pandas as pd
import numpy as np
from sportsreference.ncaab.teams import Teams
import csv

teams = Teams()
team_names = [team.name for team in teams]
# could do ols to get a real estimate for this; just using estimate from other analysis rn
hca = 1.25

# TODO: pull from other file
SIGMA = 15

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
            with open('data/games.csv', 'r') as f:
                reader = csv.reader(f)
                data = {rows[0]:rows[1:] for rows in reader}
                return data
        except FileNotFoundError:
            print('File not found, loading from scratch')
            return {'boxscore_index': ['team_name', 'opponent_name', 'points_for', 'points_against', 'location', 'datetime']}
    else:
        print('Not using cache, starting from scratch')
        return {'boxscore_index': ['team_name', 'opponent_name', 'points_for', 'points_against', 'location', 'datetime']}

def update_data(data, write=False):
    # keep track of which boxscore indices have been counted
    print('Updating data')
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
                    data[game.boxscore_index] = [team_name, game.opponent_name, game.points_for, game.points_against, game.location, game.datetime]
                print(data[game.boxscore_index])
                tracked_games.append(game.boxscore_index)
    cols = ['team_name', 'opponent_name', 'points_for', 'points_against', 'location', 'datetime']
    if write:
        with open('data/games.csv', 'w') as f:
            writer = csv.writer(f)
            # writer.writerow(cols)
            for key in data.keys():
                writer.writerow([key] + data[key])
    df = pd.DataFrame.from_dict(data, orient='index', columns=cols)
    return df

def get_adjacency_matrix(data):
    mat = pd.DataFrame(np.zeros((len(team_names), len(team_names))), index=team_names, columns=team_names)
    graph = gev(mat)
    tracked_games = data.index.tolist()
    for g in tracked_games:
        if g is None:
            continue
        if g == 'nan':
            continue
        if g not in data.index.values:
            print('Error :' + str(g))
            continue
        game_data = data.loc[g,]
        print(game_data)
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
    r_lst, r = gev.mean_stationary_dist(res)
    return r_lst, r

def display_rankings(vec, teams_list):
    sorted_indices = vec.argsort()
    # This is arbitrary to turn the rankings nicer looking, trying to reproduce a normal qq distribution
    ranked = [(teams_list[i], vec[i]) for i in sorted_indices]
    ranked.reverse()
    ranked = {i + 1 : ranked[i] for i in range(len(ranked))}
    for key, value in ranked.items():
        print(key, value[0], round(value[1]*100, 2))

def write_rankings(r):
    import csv
    dct = {}
    with open('data/rankings.csv', 'w') as f:
        writer = csv.writer(f)
        r = r.tolist()
        res = zip([i for i in range(1, len(team_names)+1)], team_names, r)
        for tup in res:
            dct[tup[1]] = [tup[0], tup[2]]
            writer.writerow(list(tup))
    return dct

def game_winner(game_data):
    pf = game_data.loc['points_for',]
    pa = game_data.loc['points_against',]
    if pf > pa:
        return game_data.loc['team_name',]
    elif pa > pf:
        return game_data.loc['opponent_name',]
    else:
        print(game_data)
        print('Game ended in a tie')
        return None

def test_retrodictive_simple(rankings_dict, data):
    print(data.head())
    # rankings_dict is in the form {team_name:}
    res = []
    for g in data.index.values:
        if not g.startswith('2'):
            continue
        if g is None:
            continue
        if g == 'nan':
            continue
        if g not in data.index.values:
            print('Error :' + str(g))
            continue
        game_data = data.loc[g,]
        team_rating = rankings_dict[game_data.loc['team_name',]][1]
        opp_rating = rankings_dict[game_data.loc['opponent_name',]][1]
        winner = game_winner(game_data)
        if winner is None:
            continue
        if winner == game_data['team_name']:
            prediction = team_rating > opp_rating
        elif winner == game_data['opponent_name']:
            prediction = team_rating < opp_rating
        print(prediction, game_data['team_name'], game_data['opponent_name'], winner)
        res.append(int(prediction))
    
    print('Simple Retrodictive Accuracy: '  + str(np.mean(res)))
        

def main():
    data = load_data(use_cache=True)
    data = update_data(data, write=True)

    mat = get_adjacency_matrix(data)
    games_played_counter = get_games_played(data)
    r_lst, r = get_rankings(mat, games_played_counter)

    import pickle
    # save r as pickle
    # with open('data/rankings.pkl', 'wb') as f:
    #     pickle.dump(r, f)

    # save r list as pickle
    # with open('data/r_lst.pkl', 'wb') as f:
    #     pickle.dump(r_lst, f)

    # open r_lst from pickle
    # with open('data/r_lst.pkl', 'rb') as f:
    #     r_lst = pickle.load(f)

    # load r as pickle
    # with open('data/rankings.pkl', 'rb') as f:
    #     r = pickle.load(f)

    rankings_dict = write_rankings(r)
    test_retrodictive_simple(rankings_dict, data)
    display_rankings(r, team_names)
    
    # write rankings to csv file
    with open ('data/sample_rankings.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(team_names)
        for ranking in r_lst:
            writer.writerow(ranking)


if __name__ == '__main__':
    main()

