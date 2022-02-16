"""Test that methods run properly."""
from unittest import TestCase

from linearmethods import explinear_var, linear_var

import numpy as np


class TestMethodsRun(TestCase):
    """Simple test of methods run.

    Run methods on small generated data and check
    shapes of output
    """

    def test_linear_var(self):
        """Test linear_var."""
        # Simple example: Generate some random data
        data = np.random.randn(1000, 3)

        # Create a causal link 0 --> 1 at lag 2
        data[2:, 1] -= 0.5*data[:-2, 0]

        # Estimate VAR model
        vals, pvals, lags = linear_var(data, maxlags=3)

        self.assertIsInstance(vals, np.ndarray)
        self.assertIsInstance(pvals, np.ndarray)
        self.assertIsInstance(lags, np.ndarray)
        self.assertEqual(vals.shape[0], vals.shape[1])
        self.assertEqual(pvals.shape[0], pvals.shape[1])
        self.assertEqual(lags.shape[0], lags.shape[1])
        self.assertEqual(vals.shape[0], pvals.shape[0])
        self.assertEqual(vals.shape[0], lags.shape[0])

    def test_explinear_var(self):
        """Test explinear_var."""
        # Simple example: Generate some random data
        data = np.random.randn(1000, 3)

        # Create a causal link 0 --> 1 at lag 2
        data[2:, 1] -= 0.5*data[:-2, 0]

        # Estimate VAR model
        vals, pvals, lags = explinear_var(data, maxlags=3)

        self.assertIsInstance(vals, np.ndarray)
        self.assertIsInstance(pvals, np.ndarray)
        self.assertIsInstance(lags, np.ndarray)
        self.assertEqual(vals.shape[0], vals.shape[1])
        self.assertEqual(pvals.shape[0], pvals.shape[1])
        self.assertEqual(lags.shape[0], lags.shape[1])
        self.assertEqual(vals.shape[0], pvals.shape[0])
        self.assertEqual(vals.shape[0], lags.shape[0])
