import numpy as np
import statsmodels.tsa.api as tsa
from typing import Tuple


def linear_var(data: np.ndarray,
               maxlags: int = 1,
               correct_pvalues: bool = True) -> Tuple[np.ndarray,
                                                      np.ndarray,
                                                      np.ndarray]:
    """Fit a linear VAR model and return coefficients as edge scores,
    additionally return (corrected) pvalues and lags.

    Args:
        data: matrix of observations (t x n)
        maxlag: positive integer, the maximum lag considered in the VAR model.
        correct_pvalues: if True

    """

    # Input data is of shape (time, variables)
    T, N = data.shape

    # Standardize data
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    # Fit VAR model and get coefficients and p-values
    tsamodel = tsa.var.var_model.VAR(data)
    results = tsamodel.fit(maxlags=maxlags,  trend='n')
    pvalues = results.pvalues
    values = results.coefs

    # CauseMe requires to upload a score matrix and
    # optionally a matrix of p-values and time lags where
    # the links occur

    # In val_matrix an entry [i, j] denotes the score for the link i --> j and
    # must be a non-negative real number with higher values denoting a higher
    # confidence for a link.
    # Fitting a VAR model results in several lagged coefficients for a
    # dependency of j on i.
    # Here we pick the absolute value of the coefficient corresponding to the
    # lag with the smallest p-value.
    val_matrix = np.zeros((N, N), dtype='float32')

    # Matrix of p-values
    p_matrix = np.ones((N, N), dtype='float32')

    # Matrix of time lags
    lag_matrix = np.zeros((N, N), dtype='uint8')

    for j in range(N):
        for i in range(N):

            # Store only values at lag with minimum p-value
            tau_min_pval = np.argmin(pvalues[
                                    (np.arange(1, maxlags+1)-1)*N + i, j]) + 1
            p_matrix[i, j] = pvalues[(tau_min_pval-1)*N + i, j]

            # Store absolute coefficient value as score
            val_matrix[i, j] = np.abs(values[tau_min_pval-1, j, i])

            # Store lag
            lag_matrix[i, j] = tau_min_pval

    # Optionally adjust p-values since we took the minimum over all lags
    # [1..maxlags] for each i-->j; should lead to an expected false positive
    # rate of 0.05 when thresholding the (N, N) p-value matrix at alpha=0.05
    # You can, of course, use different ways or none. This will only affect
    # evaluation metrics that are based on the p-values, see Details on CauseMe
    if correct_pvalues:
        p_matrix *= float(maxlags)
        p_matrix[p_matrix > 1.] = 1.

    return val_matrix, p_matrix, lag_matrix


def explinear_var(data: np.ndarray,
                  maxlags: int = 1,
                  correct_pvalues: bool = True) -> Tuple[np.ndarray,
                                                         np.ndarray,
                                                         np.ndarray]:
    """Fit a linear VAR model and return coefficients as edge scores,
    additionally return (corrected) pvalues and lags.

    Args:
        data: matrix of observations (t x n)
        maxlag: positive integer, the maximum lag considered in the VAR model.
        correct_pvalues: if True

    """

    # Input data is of shape (time, variables)
    T, N = data.shape

    # Standardize data
    data -= data.mean(axis=0)
    data /= data.std(axis=0)

    # Fit VAR model and get coefficients and p-values
    # fit over exponential of data
    tsamodel = tsa.var.var_model.VAR(np.exp(data))
    results = tsamodel.fit(maxlags=maxlags,  trend='n')
    pvalues = results.pvalues
    values = results.coefs

    # CauseMe requires to upload a score matrix and
    # optionally a matrix of p-values and time lags where
    # the links occur

    # In val_matrix an entry [i, j] denotes the score for the link i --> j and
    # must be a non-negative real number with higher values denoting a higher
    # confidence for a link.
    # Fitting a VAR model results in several lagged coefficients for a
    # dependency of j on i.
    # Here we pick the absolute value of the coefficient corresponding to the
    # lag with the smallest p-value.
    val_matrix = np.zeros((N, N), dtype='float32')

    # Matrix of p-values
    p_matrix = np.ones((N, N), dtype='float32')

    # Matrix of time lags
    lag_matrix = np.zeros((N, N), dtype='uint8')

    for j in range(N):
        for i in range(N):

            # Store only values at lag with minimum p-value
            tau_min_pval = np.argmin(pvalues[
                                    (np.arange(1, maxlags+1)-1)*N + i, j]) + 1
            p_matrix[i, j] = pvalues[(tau_min_pval-1)*N + i, j]

            # Store absolute coefficient value as score
            val_matrix[i, j] = np.abs(values[tau_min_pval-1, j, i])

            # Store lag
            lag_matrix[i, j] = tau_min_pval

    # Optionally adjust p-values since we took the minimum over all lags
    # [1..maxlags] for each i-->j; should lead to an expected false positive
    # rate of 0.05 when thresholding the (N, N) p-value matrix at alpha=0.05
    # You can, of course, use different ways or none. This will only affect
    # evaluation metrics that are based on the p-values, see Details on CauseMe
    if correct_pvalues:
        p_matrix *= float(maxlags)
        p_matrix[p_matrix > 1.] = 1.

    return val_matrix, p_matrix, lag_matrix
