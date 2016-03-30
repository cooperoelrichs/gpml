import sys
sys.path.append('../')

import unittest
from model.make_data_set import (
    split_local_train_and_test_data, extract_transform_load
)


class TestMakeDataSet(unittest.TestCase):

    def test_make_data_set(self):
        extract_transform_load()
        split_local_train_and_test_data()


if __name__ == '__main__':
    unittest.main()
