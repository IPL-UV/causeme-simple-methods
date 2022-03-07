"""Granger causality models.

Simple implementations of Granger causality models.
"""
from typing import Literal, Tuple

import numpy as np

from statsmodels.tsa.stattools import grangercausalitytests


def granger2d(data: np.ndarray,
              maxlags: int = 1,
              test: Literal['params_ftest',
                            'ssr_ftest',
                            'ssr_chi2test',
                            'lrtest'] = 'ssr_ftest') -> Tuple[np.ndarray,
                                                              np.ndarray,
                                                              np.ndarray]:
    """Perform pairwise Granger causality tests.

    It uses test statistics as scores, pvalues are also returned.
    Args:
        data: matrix of observations (t x n)
        maxlags: positive integer, the maximum lag considered in the VAR model.
        test: name of the test used

    code is adapted from https://github.com/cmu-phil/causal-learn
    """
    # check maxlags
    assert maxlags > 0

    # Input data is of shape (time, variables)
    T, N = data.shape

    # CauseMe requires to upload a score matrix and
    # optionally a matrix of p-values and time lags where
    # the links occur
    val_matrix = np.zeros((N, N), dtype='float32')

    # Matrix of p-values
    p_matrix = np.ones((N, N), dtype='float32')

    # Matrix of time lags
    lag_matrix = np.zeros((N, N), dtype='uint8')

    for j in range(N):
        for i in range(N):
            if i != j:
                res = grangercausalitytests(data[:, [j, i]],
                                            maxlag=maxlags,
                                            verbose=False)
                pvals = [res[lag + 1][0][test][1] for lag in range(maxlags)]
                # Store only values at lag with minimum p-value
                tau_min_pval = np.argmin(pvals) + 1
                p_matrix[i, j] = pvals[tau_min_pval - 1]

                # Store test statistic as score
                val_matrix[i, j] = res[tau_min_pval][0][test][0]

                # Store lag
                lag_matrix[i, j] = tau_min_pval

    return val_matrix, p_matrix, lag_matrix
