from plots import plotPD, heatMap, convergence, test_resampling, mse_complexity
import numpy as np
import matplotlib.pyplot as plt

"""Runs all tests"""

def ab ():
    """Plotting code for task (a) and "b"""
    plotPD(14, "a", [250, 500, 1000, 2000], type_="mse")
    plotPD(14, "b", [250, 500, 1000, 2000], type_="mse")
    plotPD(14, "a", [250, 500, 1000, 2000], type_="r2")
    plotPD(14, "b", [250, 500, 1000, 2000], type_="r2")
    plotPD(14, "lasso", [250, 500, 1000, 2000], type_="r2")
    plotPD(10, "a", type_="beta")
    plotPD(10, "b", type_="beta")
    plotPD(10, t=None, type_="beta")

def gradPlots():
    heatMap([i for i in range(2,16)], [0.000, 0.0005, 0.001, 0.01, 0.1, 0.2], None, "ridge", pd=True)
    heatMap([10, 50, 100, 200, 500, 1000], [0.0001, 0.0005, 0.001, 0.01, 0.05, 0.075, 0.1, 0.2], "c", "ols", pd=True)
    heatMap([0.1,0.05, 0.01, 0.001, 0.0001], [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], None, None, False)
    convergence(loop=[500,1000, 2000, 3000, 5000], l1=True)

def gh():
    """Method doing plots for problem g and h"""
    mse_complexity((np.arange(2,16)))