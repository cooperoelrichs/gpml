import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random


matplotlib.style.use('ggplot')


def open_data_files(data_dir, file_names):
    frames = {}

    for name, file_name in file_names.items():
        print('loading: %s' % name)
        frames[name] = pd.read_csv(data_dir + file_name)

    return frames


def calculate_regular_season_win_ratio(regular_season):
    w_team_counts = regular_season['Wteam'].value_counts().sort_index()
    l_team_counts = regular_season['Lteam'].value_counts().sort_index()
    w_team_ratio = w_team_counts / (w_team_counts + l_team_counts)
    return w_team_ratio


def make_dict_for_game(series):
    if random.random() < 0.5:
        return make_results_dict(series, 'Wteam', 'Lteam', 1)
    else:
        return make_results_dict(series, 'Lteam', 'Wteam', 0)

def make_results_dict(series, team1, team2, result):
    return{
        'team1': series[team1],
        'team2': series[team2],
        'result': result
    }

def extract_games_from_regular_season(regular_season):
    random.seed(0)
    games = [
        make_dict_for_game(series)
        for _, series in regular_season[['Wteam', 'Lteam']].iterrows()
    ]
    return pd.DataFrame.from_dict(games)


def add_win_ratios_to_games(games, win_ratios):
    print(games[0:1])
    print(win_ratios[0:1])

    result = None
    for column in ['team1', 'team2']:
        win_ratios.name = column
        print(pd.concat([win_ratios], axis=1)[0:9])
        result = pd.merge(games, pd.concat([win_ratios], axis=1),
                          on=column, how='left')

    print(result[0:9])

    # team1_win_ratios = win_ratios[games['team1']]
    # team2_win_ratios = win_ratios[games['team2']]
    # differences = team1_win_ratios - team2_win_ratios
    #
    # print(games)
    # print(team1_win_ratios)
    #
    # games['team1_win_ratio'] = team1_win_ratios
    # games['team2_win_ratio'] = team2_win_ratios
    # games['win_ratio_difference'] = differences

    return games


def make_mmlm2016_data_set():
    project_dir = '~/Projects/Kaggle/'
    data_dir = project_dir + 'march-machine-learning-mania-2016-v1/'

    file_names = {
        'regular_season_detailed_results': 'RegularSeasonDetailedResults.csv',
        'seasons': 'Seasons.csv',
        'teams': 'Teams.csv',
        'tourney_detailed_results': 'TourneyDetailedResults.csv',
        'tourney_seeds': 'TourneySeeds.csv',
        'tourney_ slots': 'TourneySlots.csv',
    }

    data_frames = open_data_files(data_dir, file_names)
    regular_season = data_frames['regular_season_detailed_results']
    regular_season_win_ratios = calculate_regular_season_win_ratio(regular_season)
    games = extract_games_from_regular_season(regular_season)
    basic_data_set = add_win_ratios_to_games(games, regular_season_win_ratios)

    # plt.figure()
    # regular_season_win_ratios[0:40].plot(kind='bar')
    # plt.show()

    # [team_1, team_2, win_1_%, win_2_%, win_%_diff, result]
