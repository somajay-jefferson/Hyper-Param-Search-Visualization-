import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Grid_Search_Visual import gridsearch_visual

class TestGridSearchVisual(unittest.TestCase):

    def setUp(self):
        
        # Create a sample dataset for testing
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.param_grid = {'n_estimators': [10, 20, 30], 'max_depth': [None, 5, 10]}
        self.clf_base = RandomForestClassifier(random_state=9)
        self.test_obj = gridsearch_visual(self.clf_base, self.X, self.y, self.param_grid)

    def test_initialization(self):
        # Check if the object is initialized correctly
        self.assertEqual(self.test_obj.cv, 4)
        self.assertEqual(self.test_obj.grid_key, np.nan)
        self.assertEqual(self.test_obj.grid_value, np.nan)

    def test_train_and_evaluate(self):
        # Test the train_and_evaluate method
        num = 0
        search_param = {'n_estimators': [10, 20, 30]}
        best_param_metric = 'accuracy'
        self.test_obj.train_and_evaluate(num, search_param, best_param_metric)

        # Check if the results_dict is updated
        self.assertIn('clf1', self.test_obj.results_dict)

    def tearDown(self):
        # Clean up resources if needed
        pass

if __name__ == '__main__':
    unittest.main()