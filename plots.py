from imports import *

def plot2d(outer, inner, train=False, pd=False):
    """This function takes: outer and inner, specifying the number of loop iterations (intrval ofplotting axes).
    Train: by default False. True if we want to add performance metric for training set in the plot.
    pd (polynomial degrees): if we're plotting polynomial degree. If True, we call a function for doing so"""
    
    empty_lists = makeEmptyLists(train)
    # plot_lists = makeLists(outer, inner, empty_lists, train)
    for i in outer:
        for j in inner:
            if pd:
                polynomialDegree_plot()


def heatMap(x, y, z):
    """Returns a heat map with axis x and y, and plotted z-values"""
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
