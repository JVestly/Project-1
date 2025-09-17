from imports import *

def plot2d(outer, inner, train=False):
    """This function takes in arrays x and y and z, being a single- or a list of arrays, 
    if there are more data to plot in a single plot"""
    
    lists = makeLists(train)
    for i in range(outer):
        for j in range(inner):


def heatMap(x, y, z):
    """Returns a heat map with axis x and y, and plotted z-values"""
    return None


## Auxiliary plot functions
def makeLists(train=None):
    x, y = list()
    if train:
        x_train = list()
        y_train = list()
        return x, y, x_train, y_train
    return x, y