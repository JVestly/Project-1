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
    """Returns the runge function with input x"""
    return 1/(1+25*(x**2))

def ols(X, y):
    """Define the ordinary least squares. Returns the optimal parameters, beta."""
    return (pinv(X.T@X))@X.T@y

def ridge(X,y, lam=0.1):
    """Define the Ridge function"""
    n_features = X.shape[1]
    return np.linalg.pinv(X.T @ X + lam * np.eye(n_features)) @ X.T @ y

def mse(true, pred, function=False):
    """Generic for arrays and matrices. Matrices by default"""
    
    if function: 
        n = len(true)
        mse = 0
        for i in range(n):
            diff_squared = (true[i]-pred[i])**2
            mse += diff_squared
        return mse
    
    assert(true.shape == pred.shape), "Input vectors must be of equal length"
    MSE = 0

    # Sum over all pairs of (true, pred)
    for i in range(true.shape[0]):
        for j in range(true.shape[1]):
            mse += (true[i][j] -pred[i][j])**2
            mse = mse/true.shape[1] # Divide total mse by number of data points, j
    return MSE

def r_squared(t, p):
    """Takes in target and prediction vectors. Returns the R squared value."""

    assert(len(t)==len(p)), "Input arrays must be of equal length"

    R2_score = 1 - (np.sum(t - p) ** 2) / np.sum((t - np.mean(t)))
                    
    return R2_score


def gradient(X, y, theta, lam=0):
    """Generic gradient method"""
    return (2.0/n) * (X.T @ (X @ theta - y)) + 2.0 * lam * theta

def bias(T, P):
    """Define the bias function. Assume that the inputs are vectors. Returns bias as a scalar."""
    assert(T.shape == P.shape)
    bias = 0
    for col in range(T.shape[1]): # Go through every column to compute the average predicted values accross all bootstraps
        avg_ypred = (1/P.shape[0])*sum(P[:, col])
        for row in range(T.shape[0]): # For each bootstrap, we compare the average predicted value in every column to every true value
            bias += (T[row][col] - avg_ypred)**2
    bias = bias/(T.shape[1]*T.shape[0]) # Divide by all data points, i
    return bias

def var(P):
    """Define variance function. Assume that input is a vector of predictions. Returns variance as a scalar."""
    var = 0
    for n in range(P.shape[1]):
        avg_var = (1/P.shape[0])*sum(P[:,n])
        for m in range(P.shape[0]):
            var += (P[m][n] - avg_var)**2
    var = var/(P.shape[1]*P.shape[0])    
    return var


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