import sys
sys.path.append('../')

import unittest
from model.make_model import (
    train_and_validate_lr,
    train_and_validate_svc,
    train_and_validate_sgdc,
    make_sgdc_submission,
    train_and_validate_xgb
)


class TestMakeModel(unittest.TestCase):

    def test_make_model(self):
        # train_and_validate_lr()
        # train_and_validate_svc()
        # train_and_validate_sgdc()
        # make_sgdc_submission()
        train_and_validate_xgb()

    # def test_make_lr_model():
    #     train_and_validate_lr_model()
    #
    # def test_make_svc_model():
    #     train_and_validate_svc_model()


if __name__ == '__main__':
    unittest.main()
