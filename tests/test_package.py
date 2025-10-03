import unittest
import numpy as np
from nnstacking import NNS, NNPredict, LinearStack
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from generate_data import generate_data

class TestNNStacking(unittest.TestCase):

    def setUp(self):
        """Set up a small dataset for testing."""
        self.n_train = 100
        self.n_test = 50
        self.x_train, self.y_train = generate_data(self.n_train)
        self.x_test, self.y_test = generate_data(self.n_test)
        self.estimators = [
            LinearRegression(),
            Lasso(),
            RandomForestRegressor(n_estimators=2),
        ]

    def test_nns(self):
        """Test the NNS class."""
        model = NNS(
            estimators=self.estimators,
            num_layers=1,
            hidden_size=10,
            es=False,
            nepoch=2,
        )
        model.fit(self.x_train, self.y_train)
        predictions = model.predict(self.x_test)
        self.assertEqual(predictions.shape, (self.n_test, 1))
        score = model.score(self.x_test, self.y_test)
        self.assertFalse(np.isnan(score))

    def test_nnpredict(self):
        """Test the NNPredict class."""
        model = NNPredict(
            num_layers=1,
            hidden_size=10,
            es=False,
            nepoch=2,
        )
        model.fit(self.x_train, self.y_train)
        predictions = model.predict(self.x_test)
        self.assertEqual(predictions.shape, (self.n_test, 1))
        score = model.score(self.x_test, self.y_test)
        self.assertFalse(np.isnan(score))

    def test_linear_stack(self):
        """Test the LinearStack class."""
        model = LinearStack(
            estimators=self.estimators,
        )
        model.fit(self.x_train, self.y_train.ravel())
        predictions = model.predict(self.x_test)
        self.assertEqual(predictions.shape, (self.n_test,))
        score = model.score(self.x_test, self.y_test.ravel())
        self.assertFalse(np.isnan(score))

if __name__ == '__main__':
    unittest.main()
