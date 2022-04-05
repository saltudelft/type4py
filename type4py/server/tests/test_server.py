from typing import Dict, List, Tuple
import unittest

from utils import read_file, load_json
import requests
import pytest

class TestPredictEndpoint(unittest.TestCase):
    """
    It tests the predict endpoint using the deployed server.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @pytest.fixture(autouse=True)
    def __set_env(self, pytestconfig):
        if pytestconfig.getoption("env") == 'dev':
            self.TYPE4PY_PRED_EP = "https://dev.type4py.com/api/predict?tc=0"
        elif pytestconfig.getoption("env") == 'local':
            self.TYPE4PY_PRED_EP = "http://localhost:5001/api/predict?tc=0"
        else:
            self.TYPE4PY_PRED_EP = "https://type4py.com/api/predict?tc=0"

    @classmethod
    def setUpClass(cls):
        # Input files to test the predict endpoint
        cls.test_file1 = read_file('./resources/test_file1.py')
        cls.test_file2 = read_file('./resources/test_file2.py')
        cls.test_file3 = read_file('./resources/test_file3.py')

        # Expected JSON response from the server
        cls.test_file1_exp = load_json('./resources/test_file1_exp.json')
        cls.test_file2_exp = load_json('./resources/test_file2_exp.json')
        cls.test_file3_exp = load_json('./resources/test_file3_exp.json')

    def __get_preds_from_JSON(self, file_json_repr: dict) -> List[Dict[str, List[Tuple[str, float]]]]:

        def round_preds_confidence(preds: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[Tuple[str, float]]]:
            return {n: [(n, round(s, 3)) for n,s in p] for n, p in preds.items()}
    
        all_preds: List[Dict[str, List[Tuple[str, float]]]] = []
        all_preds.append(round_preds_confidence(file_json_repr['variables_p'])) # Module-level vars

        for cls in file_json_repr['classes']:
            all_preds.append(round_preds_confidence(cls['variables_p'])) # Class variables
            for fn in cls['funcs']:
                all_preds.append(round_preds_confidence(fn['params_p'])) # Class methods parameters
                all_preds.append(round_preds_confidence(fn['variables_p'])) # Class methods local vars

                if 'ret_type_p' in fn:
                    all_preds.append(round_preds_confidence({fn['name']: fn['ret_type_p']})) # Class methods return type

        for fn in file_json_repr['funcs']:
            all_preds.append(round_preds_confidence(fn['params_p'])) # Function parameters
            all_preds.append(round_preds_confidence(fn['variables_p'])) # Function local vars

            if 'ret_type_p' in fn:
                all_preds.append(round_preds_confidence({fn['name']: fn['ret_type_p']})) # Function return type

        return all_preds

    def test_preds_file1(self):
        r = requests.post(self.TYPE4PY_PRED_EP, self.test_file1)
        self.assertEqual(self.__get_preds_from_JSON(r.json()['response']), self.__get_preds_from_JSON(self.test_file1_exp))

    def test_preds_file2(self):
        r = requests.post(self.TYPE4PY_PRED_EP, self.test_file2)
        self.assertEqual(self.__get_preds_from_JSON(r.json()['response']), self.__get_preds_from_JSON(self.test_file2_exp))

    def test_preds_file3(self):
        r = requests.post(self.TYPE4PY_PRED_EP, self.test_file3)
        self.assertEqual(self.__get_preds_from_JSON(r.json()['response']), self.__get_preds_from_JSON(self.test_file3_exp))
