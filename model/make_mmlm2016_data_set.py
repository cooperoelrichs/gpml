import pandas as pd
import numpy as np
import random
import re
from . import configer


def check_all_teams_are_accounted_for(regular_season, w_team_counts,
                                      l_team_counts, w_team_ratios):
    w_teams, l_teams = regular_season['Wteam'], regular_season['Lteam']
    sorted_teams = w_teams.append(l_teams).sort_values().unique()

    if (
            sorted_teams.size == w_team_counts.index.values.size and
            sorted_teams.size == l_team_counts.index.values.size and
            sorted_teams.size == w_team_ratios.index.values.size
        ) and (
            np.all([sorted_teams, w_team_counts.index.values]) and
            np.all([sorted_teams, l_team_counts.index.values]) and
            np.all([sorted_teams, w_team_ratios.index.values])
        ):
        return True
    else:
        return False


def validate_w_team_ratios(regular_season, w_team_counts, l_team_counts,
                           w_team_ratios):
    if check_all_teams_are_accounted_for(
            regular_season, w_team_counts, l_team_counts, w_team_ratios):
        pass
    else:
        pass
        # raise RuntimeError('Not all teams are accounted for!')


def calculate_regular_season_win_ratios(regular_season, years):
    ratios = []

    for year in years:
        regular_season_for_year = regular_season[
            regular_season['Season'] == year]

        w_team_counts = regular_season_for_year['Wteam'].value_counts().sort_index()
        l_team_counts = regular_season_for_year['Lteam'].value_counts().sort_index()
        w_team_ratios = w_team_counts / (w_team_counts + l_team_counts)
        w_team_ratios.name = 'win_ratio_%i' % year

        validate_w_team_ratios(
            regular_season_for_year, w_team_ratios,
            l_team_counts, w_team_ratios
        )

        ratios.append(w_team_ratios)

    ratios_by_year = pd.concat(ratios, axis=1)
    return ratios_by_year


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

    regular_season = config.data_frames['regular_season_detailed_results']
    win_ratios = calculate_regular_season_win_ratios(
        regular_season, np.unique(regular_season['Season'].values))
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


def make_games_dict(id_str):
    split_on_underscore = re.compile('([^_]+)')
    year, team1, team2 = split_on_underscore.findall(id_str)
    return {
        'Id': id_str,
        'year': int(year),
        'team1': int(team1),
        'team2': int(team2)
    }


def make_mmlm2016_submission_set():
    config = configer.from_json('model/config_mmlm2016.json')
    regular_season = config.data_frames['regular_season_detailed_results']
    win_ratios = calculate_regular_season_win_ratios(regular_season)

    # TODO Generate this properly, Hax for now.
    ss = config.data_frames['sample_submission']

    submission_games = [
        make_games_dict(series['Id'])
        for _, series in ss.iterrows()
    ]

    submission_games = pd.DataFrame.from_dict(submission_games)
    submission_games = join_games_and_win_ratios(submission_games, win_ratios)
    submission_games.to_hdf(config.submission_data_set_file_name,
                            key='table', append=False)

    print('Submission Data Set example:')
    print(type(submission_games))
    print(submission_games[0:2])
    print('Finished.')
