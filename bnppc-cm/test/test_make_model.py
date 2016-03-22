import sys
sys.path.append('/Users/cooperoelrichs/Projects/gpml/')

import unittest
from model.make_model import (
    train_and_validate_model
)


class TestMakeModel(unittest.TestCase):

    def test_make_model(self):
        train_and_validate_model()


if __name__ == '__main__':
    unittest.main()
