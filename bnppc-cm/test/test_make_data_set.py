import sys
sys.path.append('/Users/cooperoelrichs/Projects/gpml/')

import unittest
from model.make_data_set import (
    split_local_train_and_test_data, extract_transform_load
)


class TestMakeDataSet(unittest.TestCase):

    def test_make_mmlm2016_data_set(self):
        split_local_train_and_test_data()
        extract_transform_load()


if __name__ == '__main__':
    unittest.main()
