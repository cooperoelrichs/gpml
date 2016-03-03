import unittest
import pandas as pd
from model.data_loader import *

class TestDataLoader(unittest.TestCase):

    def test_open_mmlm2016_data_files(self):
        data_frames = open_mmlm2016_data_files()
        self.assertIsInstance(data_frames, dict)
        key = list(data_frames.keys())[0]
        self.assertIsInstance(key, str)
        print(type(data_frames[key]))
        self.assertIsInstance(data_frames[key], pd.core.frame.DataFrame)

if __name__ == '__main__':
    unittest.main()
