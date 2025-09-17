from imports import *

class GradientDescent():
    """TD: Define class"""
    def __init__(self, X_norm, iters, eta=0.1 eps=0):
        self._X = X_norm
        self._iters = iters
        self._eta = eta

    def gradOrd():
        """Define the ordinary gradient descent. Can be used for both OLS and Ridge, since lamda is 0 by default"""
        theta = np.zeros(m)
        for i in range(self._iters):
            gradient = gradient(self.X_norm, y_centered, theta)
            theta = theta - self._eta*gradient
        return theta
    
    def gradLasso():
        """Define the gradient LASSO method"""
        return None

    def gradMomentum():
        """Define the gradient momentum method"""
        return None

    def gradAda():
        """Define the AdaGrad method"""
        return None
    
    def gradADAM():
        """Define the ADAM method"""
        return None
    
    def gradStoc():
        """Define the stochastic gradient descent method"""
        return None
    
    def gradRMS():
        """Define the RMSprop method"""


class Resampling():
    """TD: Define class"""
    def __init__(self, x):
        self._x = x

    def bootstrap():
        """Define the bootstrap method"""
        return None
    
    def kCross():
        """Define the k-cross validation method"""
        return None