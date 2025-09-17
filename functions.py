from imports import *
# Define the polynomial_features method from week 36
n=1000
def polynomial_features(x, p, intercept=False):
    n = len(x)
    X = np.zeros((n, p + 1))
    if intercept==True:
        for i in(0,p):
            X[:,i] = x**i
        return X
    X = np.zeros((n, p))
    for i in range(1,p+1):
        X[:, i-1] = x**(i)
    return X

def runge(x):
    return 1/(1+25*(x**2)) + np.random.normal(0, 0.001, n)

def ols(X, y):
    """Define the ordinary least squares. Returns the optimal parameters, beta"""
    return (pinv(X.T@X))@X.T@y

def mse(true, pred):

    return None

def r_squared():
    return None


def gradient(X, y, theta, lam=0):
    """Generic gradient method"""
    return (2.0/n) * (X.T @ (X @ theta - y)) + 2.0 * lam * theta



    # Initialize lists for mse, r squared and parameters. Use 
def dummy():
    # OLS analysis with polynomial order from 1 to 15
    degrees = np.arange(1,21) # Define the degrees as going from 1 to 15
    datapoints = [100, 200, 500, 1000, 2000, 5000, 10000] # Define arbitrary datapoints
    mse_train = list()
    r2_train = list()
    mse_test = list()
    r2_test = list()
    params = list() # The parameters stays fixed, and we only need one list
    """For every combination of polynomial degree and size of data (n), we compute mse, r_squared and theta values."""
    for deg in degrees:
        for n in datapoints:
            """Since we have a varying n, we must always create a new function for every inner loop iteration"""
            # Define interval [-1,1]
            x = np.random.normal(-1,1, n)
            # Define the Runge's function with normalized noise values
            y = 1/(1+25*(x**2)) + np.random.normal(0, 1, n)
            X_train, X_test, y_train, y_test = train_test_split(x, y) # By default we split into 75-25 in favour of training data
            X = polynomial_features(X_train, deg)
            Y = polynomial_features(X_test, deg)
            # Standardize features (zero mean, unit variance for each feature)
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0)
            X_std[X_std == 0] = 1  # safeguard to avoid division by zero for constant features
            X_norm = (X - X_mean) / X_std
            # Standardize features (zero mean, unit variance for each feature) for X_test
            Y_mean = Y.mean(axis=0)
            Y_std = Y.std(axis=0)
            Y_std[Y_std == 0] = 1  # safeguard to avoid division by zero for constant features
            Y_norm = (Y - Y_mean) / Y_std
            y_offset = np.mean(y_train)
            y_centered = y_train - y_offset
            beta_ols = ols(X_norm, y_centered)
            pred_train = X_norm@beta_ols + y_offset
            pred_test = Y_norm@beta_ols + y_offset
            mse_train.append(mse(y_centered, pred_train))
            mse_test.append(mse(y_test, pred_test))