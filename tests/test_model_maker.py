import unittest
from model.make_mmlm2016_model import make_mmlm2016_model


class TestModelMaker(unittest.TestCase):

    def test_make_mmlm2016_model(self):
        make_mmlm2016_model()

if __name__ == '__main__':
    unittest.main()
