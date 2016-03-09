import pandas as pd
import numpy as np
import random
from . import configer


def open_data_files(data_dir, file_names):
    frames = {}

    for name, file_name in file_names.items():
        print('loading: %s' % name)
        frames[name] = pd.read_csv(data_dir + file_name)

    return frames


def check_all_teams_are_accounted_for(regular_season, w_team_counts,
                                      l_team_counts, w_team_ratios):
    w_teams, l_teams = regular_season['Wteam'], regular_season['Lteam']
    sorted_teams = w_teams.append(l_teams).sort_values().unique()

    if (np.all([sorted_teams, w_team_counts.index.values]) and
            np.all([sorted_teams, l_team_counts.index.values]) and
            np.all([sorted_teams, w_team_ratios.index.values])):
        return True
    else:
        return False


def validate_w_team_ratios(regular_season, w_team_counts, l_team_counts,
                           w_team_ratios):
    if check_all_teams_are_accounted_for(
            regular_season, w_team_counts, l_team_counts, w_team_ratios):
        pass
    else:
        raise RuntimeError('Not all teams are accounted for!')


def calculate_regular_season_win_ratios(regular_season):
    w_team_counts = regular_season['Wteam'].value_counts().sort_index()
    l_team_counts = regular_season['Lteam'].value_counts().sort_index()
    w_team_ratios = w_team_counts / (w_team_counts + l_team_counts)
    w_team_ratios.name = 'win_ratio'
    w_team_ratios = pd.DataFrame(w_team_ratios)

    validate_w_team_ratios(
        regular_season, w_team_ratios, l_team_counts, w_team_ratios)

    w_team_ratios['team1'] = w_team_ratios.index
    w_team_ratios['team2'] = w_team_ratios.index
    return w_team_ratios


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


def merge_win_ratios_on_team(games, win_ratios, team):
    wr_string = 'win_ratio_%s' % team
    win_ratios[wr_string] = win_ratios['win_ratio']
    return games.merge(win_ratios[[team, wr_string]], on=team, how='left')


def join_games_and_win_ratios(games, win_ratios):
    games = merge_win_ratios_on_team(games, win_ratios, 'team1')
    games = merge_win_ratios_on_team(games, win_ratios, 'team2')

    games['win_ratio_difference'] = (games['win_ratio_team1'] -
                                     games['win_ratio_team2'])
    return games


def make_mmlm2016_data_set():
    config = configer.from_json('model/config_mmlm2016.json')

    data_frames = open_data_files(config.data_dir, config.file_names)
    regular_season = data_frames['regular_season_detailed_results']
    win_ratios = calculate_regular_season_win_ratios(regular_season)
    games = extract_games_from_regular_season(regular_season)
    basic_data_set = join_games_and_win_ratios(games, win_ratios)

    basic_data_set.to_hdf(config.basic_data_set_file_name,
                          key='table', append=False)

    print('Basic Data Set example:')
    print(type(basic_data_set))
    print(basic_data_set[0:2])
    print('Finished.')

    # Plot straight from Pandas
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.style.use('ggplot')
    # plt.figure()
    # regular_season_win_ratios[0:40].plot(kind='bar')
    # plt.show()

    # Write JSON
    # with open('model/config_mmlm2016.json', 'w') as f:
    #     json.dump(json_config, f, sort_keys=True, indent=4)
    #     f.write('\n')


def make_mmlm2016_submission_set():
    data_frames = get_data_frames()
    tourney_slots = data_frames['tourney_slots']

    # This is complex, I need a seeds to all possible matchups algo. Trees???

    submission_ids = [
        series['']
        for _, series in teams.iterrows()
    ]
