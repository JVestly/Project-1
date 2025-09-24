"""Main file"""
# from imports import *
# from functions import runge, polynomial_features, mse
# from classes import GradientDescent
# from plots import heatMap

import numpy as np
from functions import polynomial_features, mse, runge
from classes import GradientDescent
from plots import plotPD, heatMap  # plotting-only utilities


if __name__ == "__main__":
    n = 1000
    x = np.linspace(-1, 1, n)
    y = runge(x) + np.random.normal(0, 0.1, n)
    x1 = np.arange(1,6)
    x2 = np.arange(-1,-6, -1)
    #heatMap(x1, x2, True)
    #plotPD(15, x, y, "a", "mse")
    #plotPD(15, x, y, "b", "mse")
    X = polynomial_features(x,10)
    print("Main file")
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1 
    X_norm = (X - X_mean) / X_std
    y_mean = y.mean()
    y_centered = y - y_mean
    # a = GradientDescent(X_norm,y_centered,iters=100)
    # bestError = (a.gradOrd())[1]
    # print(bestError)
    # ols_params = (a.gradOrd())[0]
    # best_pred = X_norm@ols_params
    # mse_ = mse(y_centered, best_pred)
    # print(mse_)
    iterations = [10, 50, 100, 500, 1000, 5000]
    heatMap(x1, x2, x, y)
    # lasso = GradientDescent(X, y, iters = 500, l1=True)             
    # l = lasso.gradOrd(lam=0.3)
    # pred_lasso = X@l
    # print(l)
    # mseLasso = mse(y, pred_lasso)
    # print(f"MSE Lasso: {mseLasso}")

    # plt.scatter(x, y, s=10, label="Data")
    # plt.plot(x, best_pred, color="red", label="Fitted model")
    # plt.legend()
    # plt.show()