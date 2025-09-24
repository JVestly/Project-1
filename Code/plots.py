# from imports import *
# from functions import polynomial_features, scale, ols, ridge, mse, r_squared
# from classes import GradientDescent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functions import polynomial_features, scale, ols, ridge, mse, r_squared
from sklearn.model_selection import train_test_split
from classes import GradientDescent

def plotPD(deg, x, y, t, type_=" "):
    """
    Plots a variable as a function of another.
    Plots between one and three graphs in a single plot.

    Parameters
    ----------
    deg: int
        Sets the roof of polynomial degrees.

    x: array-like
        X-data. Specifies the x-axis' range

    y: array-like
        Y-data defined in main file

    t: string
        Specifies which task (a or b). (a) uses OLS, 
        and (b) uses Ridge. Assumes either "a" or "b"
        is given as argument.

    type_: string
        Specifies what is to be plotted.
        The code plots mse, r squared or beta values
        depending on the argument to type_.

    """
    
    plot_list = list()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    for i in range(1, deg+1):
        X = polynomial_features(x_train, i)
        Y = polynomial_features(x_test, i)
        scaled_train = (scale(X, y))[0]
        scaled_test = (scale(Y, y))[0]
        if t == "a":
            beta = ols(scaled_train, y_train)
        elif t == "b":
            beta = ridge(scaled_train, y_train, 0.01)
        pred = scaled_test@beta + np.mean(y_train)
        if type_ == "mse":
            plot_list.append(mse(y_test, pred))
        elif type_ == "r2":
            plot_list.append(r_squared(y_test, pred))
        else: plot_list.append(np.mean(beta))

    orders = np.arange(1, deg+1)

    plt.figure()
    plt.plot(orders, plot_list, marker='o')
    plt.plot(orders, plot_list, marker='o')
    plt.xlabel("Polynomial degree")
    if type_ == "mse":
        plt.ylabel("MSE")
        plt.title(f"MSE Test")
    elif type_ == "r2":
        plt.ylabel("R2")
        plt.title(f"R Squared Test")
    else:
        plt.ylabel("Beta")
        plt.title(f"Beta Test")
    plt.grid(True)
    plt.show()


def heatMap(outer, inner, x, y, pd=False):
    """
    Heat map as a function of two variables.
    
    Parameters
    ----------

    """
    plot_matrix = np.empty((len(x), len(y)))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    n = 3
    X = polynomial_features(x_train, n)
    Y = polynomial_features(x_test, n)
    scaled_train, y_scaled = scale(X, y)
    scaled_test = (scale(Y, y))[0]
    y_mean = np.mean(y)
    y_centered = y - y_mean
    for i, deg in enumerate(outer):
        if pd: 
            X = polynomial_features(x_train, i)
            Y = polynomial_features(x_test, i)
            scaled_train, y_scaled = scale(X, y)
            scaled_test = (scale(Y, y))[0]
            y_mean = np.mean(y)
            y_centered = y - y_mean
        for e, lam in enumerate(inner):
            if pd:
                beta = ols(scaled_train, y_train)
                pred = scaled_test@beta + np.mean(y_train)
            elif not pd:
                grad = GradientDescent(scaled_train, y_train, deg, lam, l1=True)
                betas = (grad.gradOrd(lam=lam))[0]
                pred = scaled_test@betas
            pointMse = mse(y_test, pred)
            plot_matrix[i,e] = pointMse


    f, ax = plt.subplots(figsize=(9, 6))
    ax = sns.heatmap(plot_matrix, annot = True,
                 cmap = 'coolwarm', linecolor = 'black',
                 linewidths = 2, robust = True)
    
    ax.set_title('Heatmap')
    ax.set_xlabel('My X label')
    ax.set_ylabel('My Y label')
    #f.savefig('My heatmap.png')
    plt.show()



## Auxiliary plot functions

def makeEmptyLists(train=None):
    x, y = list()
    if train:
        x_train = list()
        y_train = list()
        return x, y, x_train, y_train
    return x, y

# def makeLists(outer, inner, empty, train=False)
#     """Define a helper function which iterates and makes plot lists"""

#def scaling(pd):    
