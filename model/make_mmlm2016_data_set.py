import pandas as pd
import numpy as np
import random
import re
from . import configer


def get_all_teams_sorted(regular_season):
    w_teams, l_teams = regular_season['Wteam'], regular_season['Lteam']
    sorted_teams = w_teams.append(l_teams).sort_values().unique()
    return sorted_teams


def calculate_regular_season_win_ratios(regular_season, years):
    ratios_by_year = {}

    for year in years:
        regular_season_for_year = regular_season[
            regular_season['Season'] == year]

        w_team = regular_season_for_year['Wteam']
        l_team = regular_season_for_year['Lteam']
        w_team_counts = w_team.value_counts().sort_index()
        l_team_counts = l_team.value_counts().sort_index()

        w_team_ratios = w_team_counts / (w_team_counts + l_team_counts)
        w_team_ratios.name = 'win_ratio_%i' % year
        ratios_by_year[year] = w_team_ratios

        if year == 2005:
            print(w_team_ratios[1366])
            print(l_team_counts[1366])
            print(w_team_counts[1366])
            raise('!!!')

    all_teams = get_all_teams_sorted(regular_season)
    ratios_by_year = fill_in_missing_teams(ratios_by_year, all_teams)
    ratios = pd.concat(list(ratios_by_year.values()), axis=1)

    ratios['team1'] = ratios.index
    ratios['team2'] = ratios.index
    return ratios


def fill_in_missing_teams(ratios_by_year, all_teams):
    all_years = sorted(list(ratios_by_year.keys()))

    for year, ratios in ratios_by_year.items():
        missing_teams = np.setdiff1d(all_teams, ratios.index.values)

        if year == min(all_years):
            for team in missing_teams:
                for replacment_year in all_years:
                    if team in ratios_by_year[replacment_year].index:
                        ratios_by_year[year].set_value(
                            team, ratios_by_year[replacment_year][team])
                        break
                    elif replacment_year == max(all_years):
                        raise RuntimeError('No replacment year found.')
                    else:
                        pass
        elif year == max(all_years):
            for team in missing_teams:
                for replacment_year in reversed(all_years):
                    if team in ratios_by_year[replacment_year].index:
                        ratios_by_year[year].set_value(
                            team, ratios_by_year[replacment_year][team])
                        break
                    elif replacment_year == min(all_years):
                        raise RuntimeError('No replacment year found.')
                    else:
                        pass
        else:
            for team in missing_teams:
                for left_replacment_year in all_years:
                    if team in ratios_by_year[left_replacment_year].index:
                        for right_replacment_year in reversed(all_years[left_replacment_year:]):
                            if team in ratios_by_year[right_replacment_year].index:
                                if year == 2004 and team == 1289:
                                    print(right_replacment_year)
                                    print(left_replacment_year)

                                ratios_by_year[year].set_value(team, (
                                    (ratios_by_year[right_replacment_year][team] +
                                     ratios_by_year[left_replacment_year][team]) / 2
                                ))
                                break
                            elif right_replacment_year == min(all_years):
                                raise RuntimeError('No replacment year found - hit min.')
                            else:
                                pass
                        break
                    elif left_replacment_year == max(all_years):
                        raise RuntimeError('No replacment year found - hit max.')
                    else:
                        pass

    # Check
    for year, ratios in ratios_by_year.items():
        if np.isnan(ratios.values).any():
            print(ratios[np.isnan(ratios.values)])
            raise RuntimeError('Still contains NaNs - %i' % year)
        elif np.array_equal(ratios.sort_index().index.values, all_teams):
            pass
        else:
            print(np.setdiff1d(all_teams, ratios.index.values))
            raise RuntimeError('Still missing teams - %i' % year)

    # years = years.sort()
    # column_and_index_of_nans =
    # for column_name in games.columns:
    #     row_indicies = games[column_name].isnull()
    #     column_and_index_of_nans[column_name] = row_indicies
    #
    # df.T.fillna(df.mean(axis=1)).T
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


def merge_win_ratios_on_team(games, win_ratios, team, year):
    wr_string = 'win_ratio_%i_%s' % (year, team)
    win_ratios[wr_string] = win_ratios['win_ratio_%i' % year]
    return games.merge(win_ratios[[team, wr_string]], on=team, how='left')


def join_games_and_win_ratios(games, win_ratios, years):
    for year in years:
        games = merge_win_ratios_on_team(games, win_ratios, 'team1', year)
        games = merge_win_ratios_on_team(games, win_ratios, 'team2', year)

        games['win_ratio_%i_difference' % year] = (
            games['win_ratio_%i_team1' % year] -
            games['win_ratio_%i_team2' % year]
        )

    return games


def which_years(regular_season):
    return np.unique(regular_season['Season'].values)


def check_and_save_to_hdf(df, file_name):
    if np.isnan(df.values).any():
        raise RuntimeError('df contains NaNs - %s' % file_name)
    df.to_hdf(file_name, key='table', append=False)


def make_mmlm2016_data_set():
    config = configer.from_json('model/config_mmlm2016.json')

    regular_season = config.data_frames['regular_season_detailed_results']
    years = which_years(regular_season)
    win_ratios = calculate_regular_season_win_ratios(regular_season, years)
    games = extract_games_from_regular_season(regular_season)
    basic_data_set = join_games_and_win_ratios(games, win_ratios, years)

    check_and_save_to_hdf(basic_data_set, config.basic_data_set_file_name)

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
    years = which_years(regular_season)
    win_ratios = calculate_regular_season_win_ratios(regular_season, years)

    # TODO Generate this properly, Hax for now.
    ss = config.data_frames['sample_submission']

    submission_games = [
        make_games_dict(series['Id'])
        for _, series in ss.iterrows()
    ]

    submission_games = pd.DataFrame.from_dict(submission_games)
    submission_games = join_games_and_win_ratios(
        submission_games, win_ratios, years)
    check_and_save_to_hdf(submission_games,
                          config.submission_data_set_file_name)

    print('Submission Data Set example:')
    print(type(submission_games))
    print(submission_games[0:2])
    print('Finished.')
