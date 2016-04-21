# import multiprocessing
# multiprocessing.set_start_method('forkserver')

import sys
sys.path.append('../')

import unittest
from model.make_model import (
    train_and_validate_lr,
    train_and_validate_svc,
    train_and_validate_sgdc,
    make_sgdc_submission,

    train_and_validate_xgb,
    make_xgb_submission,

    train_and_validate_etc,
    make_etc_submission,

    train_and_validate_rfc,
    make_rfc_submission,

    train_and_validate_model_average,
    make_model_average_submission,
    train_and_validate_lr_ensemble,
    make_lr_ensemble_submission,
    train_and_validate_etc_ensemble,
    make_etc_ensemble_submission,
    train_and_validate_xgbc_ensemble,
    make_xgbc_ensemble_submission
)


class TestMakeModel(unittest.TestCase):

    def test_make_model(self):
        # train_and_validate_lr()
        # train_and_validate_svc()
        # train_and_validate_sgdc()
        # make_sgdc_submission()
        # train_and_validate_xgb()
        # make_xgb_submission()
        # train_and_validate_etc()
        # make_etc_submission()
        # train_and_validate_rfc()
        # make_rfc_submission()

        # train_and_validate_model_average()
        # make_model_average_submission()
        train_and_validate_lr_ensemble()
        make_lr_ensemble_submission()
        train_and_validate_etc_ensemble()
        make_etc_ensemble_submission()
        train_and_validate_xgbc_ensemble()
        make_xgbc_ensemble_submission()

    # def test_make_lr_model():
    #     train_and_validate_lr_model()
    #
    # def test_make_svc_model():
    #     train_and_validate_svc_model()


if __name__ == '__main__':
    unittest.main()
