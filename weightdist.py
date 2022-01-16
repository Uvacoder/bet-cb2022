import pandas as pd
import csv
import numpy as np

data = pd.read_csv('data/games.csv')

matchups = {}
repeated = {}
for idx, row in data.iterrows():
    if frozenset([row['team_name'], row['opponent_name']]) not in matchups:
        # TODO: adjust for hca
        # sort alphabetically
        team1 = sorted([row['team_name'], row['opponent_name']])[0]
        team2 = sorted([row['team_name'], row['opponent_name']])[1]
        if row['location'] == 'Home':
            row['points_for'] -= 1.25
        elif row['location'] == 'Away':
            row['points_for'] += 1.25
        if team1 == row['team_name']:
            matchups[frozenset([team1, team2])] = [row['points_for'] - row['points_against']]
        else:
            matchups[frozenset([team1, team2])] = [row['points_against'] - row['points_for']]
    else:
        # sort alphabetically
        team1 = sorted([row['team_name'], row['opponent_name']])[0]
        team2 = sorted([row['team_name'], row['opponent_name']])[1]
        if team1 == row['team_name']:
            matchups[frozenset([team1, team2])].append(row['points_for'] - row['points_against'])
        else:
            matchups[frozenset([team1, team2])].append(row['points_against'] - row['points_for'])


res = {key: value for key, value in matchups.items() if len(value) > 1}

with open('data/repeated_matchups.csv', 'w') as f:
    writer = csv.writer(f)
    for key, value in res.items():
        writer.writerow([key] + value)

for key, value in res.items():
    print(key, value)

diffs = list(map(lambda x: x[1] - x[0], res.values()))
print(diffs)
variance = np.var(list(map(lambda x: x[1] - x[0], res.values())))
print(variance)
print(np.sqrt(variance))

