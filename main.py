"""Main file"""
from imports import *

if __name__ == "__main__":
    x = np.linspace(-1, 1, n)
    y = runge(x)
    print("Main file")
    plt.figure()
    plt.plot(x,y)
    plt.show()