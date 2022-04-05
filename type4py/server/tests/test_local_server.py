"""
This module tests the Type4Py's local model using the Docker image
"""

import unittest
import requests
from utils import read_file

class TestLocalPredictEndpoint(unittest.TestCase):
    """
    An integration test for testing the Type4Py pipeline end-to-end locally
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TYPE4PY_PRED_EP = "http://localhost:5001/api/predict?tc=0"

    @classmethod
    def setUpClass(cls):
        # Input files to test the predict endpoint
        cls.test_file1 = read_file('./resources/test_file1.py')
        cls.test_file2 = read_file('./resources/test_file2.py')
        cls.test_file3 = read_file('./resources/test_file3.py')
    
    # TODO: Assert the model's predicted types against an expected JSON file, similar to test_server.py
    def test_preds_file1(self):
        r = requests.post(self.TYPE4PY_PRED_EP, self.test_file1)
        self.assertNotEqual(r.json()['response'], None)

    def test_preds_file2(self):
        r = requests.post(self.TYPE4PY_PRED_EP, self.test_file2)
        self.assertNotEqual(r.json()['response'], None)

    def test_preds_file3(self):
        r = requests.post(self.TYPE4PY_PRED_EP, self.test_file3)
        self.assertNotEqual(r.json()['response'], None)
