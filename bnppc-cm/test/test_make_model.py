import sys
sys.path.append('../')

import unittest
from model.make_model import (
    train_and_validate_model,
    make_a_submission
)


class TestMakeModel(unittest.TestCase):

    def test_make_model(self):
        train_and_validate_model()
        make_a_submission()


if __name__ == '__main__':
    unittest.main()
