# from imports import *
#from plots import *
# REMOVE: from imports import *
import numpy as np
from numpy.linalg import pinv


def polynomial_features(x, p, intercept=False):
    """
    Generate a polynomial feature matrix from input data.

    Parameters
    ----------
    x : array-like, shape (n_samples,)
        Input feature values.
    p : int
        Polynomial degree.
    intercept : bool, default=False
        If True, includes a column of ones for the intercept term.

    Returns
    -------
    ndarray, shape (n_samples, p) or (n_samples, p+1)
        Design matrix with polynomial features up to degree p.
    """
    n = len(x)

    if intercept:
        X = np.zeros((n, p + 1))

        for i in range(p + 1):
            X[:, i] = x**i

        return X
    
    X = np.zeros((n, p))

    for i in range(1, p + 1):
        X[:, i - 1] = x**i

    return X


def runge(x):
    """
    Evaluate the Runge function.

    Parameters
    ----------
    x : array-like
        Input values.

    Returns
    -------
    ndarray
        Values of the Runge function 1 / (1 + 25 * x^2) at the given inputs.
    """
    return 1 / (1 + 25 * (x**2))


def ols(X, y):
    """
    Ordinary Least Squares regression.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix of input features.
    y : ndarray, shape (n_samples,)
        Target values.

    Returns
    -------
    ndarray, shape (n_features,)
        Estimated regression coefficients (beta).
    """
    return (pinv(X.T @ X)) @ X.T @ y


def ridge(X, y, lam=0.1):
    """
    Ridge regression.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix of input features.
    y : ndarray, shape (n_samples,)
        Target values.
    lam : float, default=0.1
        Regularization strength (lambda).

    Returns
    -------
    ndarray, shape (n_features,)
        Estimated regression coefficients (beta).
    """
    n_features = X.shape[1]

    return np.linalg.pinv(X.T @ X + lam * np.eye(n_features)) @ X.T @ y

def soft_threshold(z, alpha):
    """
    Used element-wise in gradient descent for Lasso regression.
    Shrinks large values, and sets small values to zero.

    Parameters
    ----------
    z : ndarray
        Predicted betas.
    lam : float, default=0.1
        Regularization strength (lambda).

    Returns
    -------
    float 
        Estimated regression coefficient (beta).
        Returns 0 if the absoulte value of y is less than or equal to alpha
    """
        
    return np.sign(z) * np.maximum(np.abs(z) - alpha, 0.0)

    
def mse(y_true, y_pred):
    """
    Compute mean squared error (MSE) as a scalar.
    Works for both vectors (1D) and matrices (2D).
    For matrices, the MSE is computed column-wise and then averaged.

    Parameters
    ----------
    y_true : array-like
        True target values. Shape (n_samples,) or (n_samples, n_outputs).
    y_pred : array-like
        Predicted target values. Same shape as `y_true`.

    Returns
    -------
    float
        Mean squared error. Always a scalar.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. y: {y_true.shape}, p:{y_pred.shape}")

    if y_true.ndim == 1:
        n = len(y_true)
        error_sum = 0.0
        for i in range(n):
            error_sum += (y_true[i] - y_pred[i])**2

        return error_sum / n


    elif y_true.ndim == 2:
        n, m = y_true.shape 
        mse_total = 0.0

        for j in range(m): 
            error_sum = 0.0

            for i in range(n): 
                error_sum += (y_true[i, j] - y_pred[i, j])**2
            mse_total += error_sum / n 

        return mse_total / m  

    else:
        raise ValueError("Inputs must be 1D or 2D arrays")
    

def r_squared(t, p):
    """
    Compute coefficient of determination (R^2) as a scalar.
    Works for both vectors (1D) and matrices (2D).
    For matrices, R^2 is computed column-wise and then averaged.

    Parameters
    ----------
    t : array-like
        True target values. Shape (n_samples,) or (n_samples, n_outputs).
    p : array-like
        Predicted target values. Same shape as `t`.

    Returns
    -------
    float
        R^2 score. Always a scalar.
    """
    t = np.asarray(t)
    p = np.asarray(p)

    assert t.shape == p.shape, "Input arrays must have the same shape"

    if t.ndim == 1:
        ss_res = 0.0
        ss_tot = 0.0
        mean_t = np.mean(t)

        for i in range(len(t)):
            ss_res += (t[i] - p[i])**2
            ss_tot += (t[i] - mean_t)**2

        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    elif t.ndim == 2:
        n, m = t.shape
        r2_total = 0.0

        for j in range(m):
            ss_res = 0.0
            ss_tot = 0.0
            mean_t = np.mean(t[:, j])

            for i in range(n):
                ss_res += (t[i, j] - p[i, j])**2
                ss_tot += (t[i, j] - mean_t)**2
            r2_total += 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return r2_total / m


def gradient(X, y, beta, lam=0.0):
    """
    Compute the gradient of the cost function for linear regression.

    Supports both Ordinary Least Squares (OLS) and Ridge regression
    depending on the value of the regularization parameter `lam`.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Design matrix of input features.
    y : ndarray, shape (n_samples,)
        Target values.
    beta : ndarray, shape (n_features,)
        Current estimate of regression coefficients.
    lam : float, default=0.0
        Regularization parameter.
        - lam = 0.0: OLS gradient
        - lam > 0.0: Ridge gradient

    Returns
    -------
    ndarray, shape (n_features,)
        Gradient of the cost function with respect to `beta`.
    """
    n = X.shape[0]
    if lam != 0.0:
        return (2 / n) * X.T @ ((X @ beta) - y) + 2 * lam * beta

    return (2 / n) * X.T @ ((X @ beta) - y)

def bias(y_true, y_pred):
    """
    Compute squared bias as a scalar.
    Works for both vectors (1D) and matrices (2D).
    For matrices, bias is computed per sample across models,
    then averaged over all samples.

    Parameters
    ----------
    y_true : array-like
        True target values. Shape (n_samples,) or (n_samples,).
    y_pred : array-like
        Predicted target values. Shape (n_samples,) or (n_models, n_samples).

    Returns
    -------
    float
        Squared bias. Always a scalar.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_pred.ndim == 1:
        n = len(y_true)
        bias_sum = 0.0

        for i in range(n):
            bias_sum += (y_true[i] - y_pred[i])**2

        return bias_sum / n

    elif y_pred.ndim == 2:
        n_models, n_samples = y_pred.shape
        bias_sum = 0.0

        for j in range(n_samples):
            mean_pred = 0.0

            for i in range(n_models):
                mean_pred += y_pred[i, j]
            mean_pred /= n_models
            bias_sum += (y_true[j] - mean_pred)**2

        return bias_sum / n_samples

    else:
        raise ValueError("y_pred must be 1D or 2D array")


def var(P):
    """
    Compute variance of predictions as a scalar.
    Works for both vectors (1D) and matrices (2D).
    For vectors, computes the variance across all elements.
    For matrices, computes variance across models for each sample,
    then averages over all samples.

    Parameters
    ----------
    P : array-like
        Predicted target values. Shape (n_samples,) or (n_models, n_samples).

    Returns
    -------
    float
        Variance of predictions. Always a scalar.
    """
    P = np.asarray(P)

    if P.ndim == 1:
        mean_val = np.mean(P)
        var = 0.0

        for i in range(len(P)):
            var += (P[i] - mean_val) ** 2
        var /= len(P)

        return var

    elif P.ndim == 2:
        var = 0.0

        rows, cols = P.shape

        for n in range(cols): 
            avg_val = (1 / rows) * sum(P[:, n])

            for m in range(rows): 
                var += (P[m, n] - avg_val) ** 2
        var /= (rows * cols)

        return var

    else:
        raise ValueError("Input must be 1D or 2D array")

def scale(X, y):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std

    y_mean = y.mean()
    y_centered = y-y_mean

    return X_norm, y_centered