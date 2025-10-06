"""Main file"""


import numpy as np
from functions import polynomial_features, mse, runge
from classes import GradientDescent
from plots import plotPD, heatMap, poly_fit
import pandas as pd
from test import ab, gradPlots, gh
from plots import test_lasso, test_resampling, testK_fold


if __name__ == "__main__":
    print("Main file")
    print(test_lasso())
    poly_fit()
    ab()
    gradPlots()
    test_resampling()
    testK_fold()