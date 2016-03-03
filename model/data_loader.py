import pandas as pd

def open_data_files(data_dir, file_names):
    frames = {}

    for name, file_name in file_names.items():
        print('loading: %s' % name)
        frames[name] = pd.read_csv(data_dir + file_name)

    return frames

def open_mmlm2016_data_files():
    project_dir = '~/Projects/Kaggle/'
    data_dir = project_dir + 'march-machine-learning-mania-2016-v1/'

    file_names = {
        'regular_season_detailed_results' : 'RegularSeasonDetailedResults.csv',
        'seasons' : 'Seasons.csv',
        'teams' : 'Teams.csv',
        'tourney_detailed_results' : 'TourneyDetailedResults.csv',
        'tourney_seeds' : 'TourneySeeds.csv',
        'tourney_ slots' : 'TourneySlots.csv',
    }

    return open_data_files(data_dir, file_names)
