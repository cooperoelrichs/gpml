import unittest
from model.make_mmlm2016_data_set import make_mmlm2016_data_set


class TestDataLoader(unittest.TestCase):

    def test_make_mmlm2016_data_set(self):
        make_mmlm2016_data_set()

if __name__ == '__main__':
    unittest.main()
