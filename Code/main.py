"""Main file"""
# from imports import *
# from functions import runge, polynomial_features, mse
# from classes import GradientDescent
# from plots import heatMap

import numpy as np
from functions import polynomial_features, mse, runge
from classes import GradientDescent
from plots import plotPD, heatMap, poly_fit
import pandas as pd
from test import ab, gradPlots


if __name__ == "__main__":
    print("Main file")
    poly_fit()
    #ab()
    #gradPlots()