from libsa4py.utils import read_file, load_json
import unittest
import requests


class TestPredictEndpoint(unittest.TestCase):
    """
    It tests the predict endpoint using the deployed server.
    """

    TYPE4PY_PRED_EP = "https://type4py.com/api/predict?tc=0&fp=1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TYPE4PY_PRED_EP = "https://type4py.com/api/predict/"

    @classmethod
    def setUpClass(cls):
        # Input files to get predictions
        cls.test_file1 = read_file('./resources/test_file1.py')
        cls.test_file2 = read_file('./resources/test_file2.py')

        # Expected JSON response from the server
        cls.test_file1_exp = load_json('./resources/test_file1_exp.json')
        cls.test_file2_exp = load_json('./resources/test_file2_exp.json')

    def test_get_preds_file1(self):
        r = requests.post(self.TYPE4PY_PRED_EP, self.test_file1)
        print(r.status_code)