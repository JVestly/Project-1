"""Main file"""
from imports import *

if __name__ == "__main__":
    x = np.linspace(-1, 1, 1000)
    y = runge(x)
    X = polynomial_features(x,3)
    print("Main file")
    a = GradientDescent(X,y,100)
    ols_params = a.gradOrd()
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1  # safeguard to avoid division by zero for constant features
    X_norm = (X - X_mean) / X_std
    best_pred = X@ols_params
    mse_ = mse(y, best_pred, True)
    print(mse)