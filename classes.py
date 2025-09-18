from imports import *

class GradientDescent():
    """TD: Define class"""
    def __init__(self, X_norm, iters, eps=1e-8):
        self._X = X_norm
        self._iters = iters
        self._eps = eps

    def gradOrd(y_centered, eta=0.1):
        """Define the ordinary gradient descent. Can be used for both OLS and Ridge, since lamda is 0 by default"""
        m = self.norm.shape[1]
        theta = np.zeros(m)
        preds = np.zeros(m)
        for i in range(self._iters):
            if i!=0:
                if stopping(gradient): break
            gradient = gradient(self._X_norm, y_centered, theta)
            theta = theta - new_velocity
            preds = self._X_norm@theta

        return theta, preds
    
    def gradLasso():
        """Define the gradient LASSO method"""
        return None

    def gradMomentum(momentum, eta=0.1):
        """Define the gradient momentum method"""
        theta = np.zeros(m)
        velocity = 0
        for i in range(self._iters):
            gradient = gradient(self.X_norm, y_centered, theta)
            new_velocity = eta*gradient + momentum*velocity
            theta = theta - new_velocity
            velocity = new_velocity
        return theta

    def gradAda():
        """Define the AdaGrad method"""
        return None
    
    def gradADAM():
        """Define the ADAM method"""
        return None
    
    def gradStoc(batch_size=1):
        """Define the stochastic gradient descent method"""
        
    
    def gradRMS():
        """Define the RMSprop method"""
        return None
    
    def stopping(grad, e=10**(-8)):
        """Define a helper function for gradient descent"""
        if np.linalg.norm(grad) < e: return True


class Resampling():
    """TD: Define class"""
    def __init__(self, x):
        self._x = x

    def bootstrap(data, bootstraps):
        """Define the bootstrap method"""
        return None
    
    def kCross(data, k):
        """Define the k-cross validation method"""
        return None