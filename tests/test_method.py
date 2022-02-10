import numpy as np
from varmethods import linear_var, explinear_var

from unittest import TestCase



class TestMethodRun(TestCase):

    def test_linear_var(self):
        # Simple example: Generate some random data
        data = np.random.randn(1000, 3)
        
        # Create a causal link 0 --> 1 at lag 2
        data[2:, 1] -= 0.5*data[:-2, 0]

        # Estimate VAR model
        vals, pvals, lags = linear_var(data, maxlags=3)

        # Score is just absolute coefficient value, significant p-value is at entry
        # (0, 1) and corresponding lag is 2

        self.assertIsInstance(vals, np.ndarray)
        self.assertIsInstance(pvals, np.ndarray)
        self.assertIsInstance(lags, np.ndarray)
        self.assertEqual(vals.shape[0], vals.shape[1])
        self.assertEqual(pvals.shape[0], pvals.shape[1])
        self.assertEqual(lags.shape[0], lags.shape[1])
        self.assertEqual(vals.shape[0], pvals.shape[0])
        self.assertEqual(vals.shape[0], lags.shape[0])


    def test_iexplinear_var(self):
        # Simple example: Generate some random data
        data = np.random.randn(1000, 3)
        
        # Create a causal link 0 --> 1 at lag 2
        data[2:, 1] -= 0.5*data[:-2, 0]

        # Estimate VAR model
        vals, pvals, lags = explinear_var(data, maxlags=3)

        # Score is just absolute coefficient value, significant p-value is at entry
        # (0, 1) and corresponding lag is 2

        self.assertIsInstance(vals, np.ndarray)
        self.assertIsInstance(pvals, np.ndarray)
        self.assertIsInstance(lags, np.ndarray)
        self.assertEqual(vals.shape[0], vals.shape[1])
        self.assertEqual(pvals.shape[0], pvals.shape[1])
        self.assertEqual(lags.shape[0], lags.shape[1])
        self.assertEqual(vals.shape[0], pvals.shape[0])
        self.assertEqual(vals.shape[0], lags.shape[0])
