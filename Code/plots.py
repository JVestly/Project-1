from imports import *

def plota(i, x, y, xname="pd", type_=" "):
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

    type_: string
        Specifies what is to be plotted.
        The code plots mse, r squared or beta values
        depending on the argument to type_.

    """
    
    plot_list = list()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    for i in range(1, i+1):
        X = polynomial_features(x_train, i)
        Y = polynomial_features(x_test, i)
        scaled_train = (scale(X, y))[0]
        scaled_test = (scale(Y, y))[0]
        beta = ols(scaled_train, y_train)
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


def heatMap(x, y, z):
    """
    Heat map as a function of two variables.
    
    Parameters
    ----------

    """
    return None


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

def polynomialDegree_plot(lists, outer, inner, train=False):
    scaling()
