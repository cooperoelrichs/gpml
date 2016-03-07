import pandas as pd


def make_mmlm2016_model():
    project_dir = '~/Projects/Kaggle/'
    data_dir = project_dir + 'march-machine-learning-mania-2016-v1/'
    basic_data_set_file_name = 'basic_data_set.csv'

    basic_data_set = pd.read_csv(data_dir + basic_data_set_file_name,
                                 index_col='index')

    print(type(basic_data_set))
    print(basic_data_set[0:2])
