# from imports import *
# from functions import soft_threshold, mse
# from plots import plotPD, heatMap

import numpy as np
from functions import soft_threshold, mse, gradient

class GradientDescent:
    """
    Gradient descent class
    
    """
    def __init__(self, X_norm, y_centered, iters, eps=1e-8, l1=False):
        self._X = X_norm
        self._iters = iters
        self._eps = eps
        self._y = y_centered
        self._m = X_norm.shape[1]
        self._l1 = l1

    
    def gradOrd(self, eta=0.001, l=0.1):
        """
        Ordinary gradient descent. Can be used for both 
        OLS and Ridge, since lam=0 by default.
        Also including an invariant for using LASSO. 

        Parameters
        ----------
        eta : float, default 0.1
            Learning rate.
        l : float, default 0.0
            Ridge and LASSO penalty strength.
            
        Returns
        -------
        ndarray of shape (m,)
            The optimized parameter vector.
        """
        theta = np.zeros(self._m)
        thetas = []

        msError = []

        for t in range(self._iters):
            grad = gradient(self._X, self._y, theta, lam=l)


            z = theta - eta * grad
            
            if self._l1:
                alpha = eta*l
                z = theta - eta*grad
                theta = soft_threshold(z, alpha)
                thetas.append(theta)
                pred = self._X@theta
                msError.append(mse(self._y, pred))
            else:
                theta = z
                thetas.append(theta)
                pred = self._X@theta
                msError.append(mse(self._y, pred))
            if self.stopping(grad):
                break

        best_pred = min(msError)
        idx = msError.index(best_pred)
        theta = thetas[idx]
        return theta, best_pred

    
    def pgdUpdate(self, beta, grad, eta, lam=0.1):
        """
        REDUNDANT FUNCTION?
        Proximal gradient descent used for LASSO regression.
        
        Parameters
        ----------
        lam: float, default 0.1.
            Lasso penalty.

        Returns
        -------
        ndarray of shape (m, )
            The optimized parameter vector
        """
        z = beta - eta*grad
        alpha = eta*lam

        return soft_threshold(z, alpha)
    
    def gradMomentum(self, momentum=0.9, eta=0.001, lam=0.0):
        """
        Gradient Descent with momentum for OLS/Ridge.
        Parameters
        ----------
        momentum : float, default 0.9
            Exponential weight on the previous update.
        eta : float, default 0.1
            Learning rate.
        lam : float, default 0.0
            Ridge penalty strength.
        Returns
        -------
        ndarray of shape (m,)
            The optimized parameter vector.
        """
        theta = np.zeros(self._m)
        thetas = []

        msError = []
        velocity = 0
        for _ in range(self._iters):
            grad = gradient(self._X, self._y, theta, lam)
            new_velocity = eta*grad + momentum*velocity
            z = theta - eta * velocity
            if self._l1:
                alpha = eta*lam
                theta = soft_threshold(z, alpha)
            else: 
                theta = z
            thetas.append(theta)
            pred = self._X@theta
            msError.append(mse(self._y, pred))
            if self.stopping(grad):
                break
  
        best_pred = min(msError)
        idx = msError.index(best_pred)
        theta = thetas[idx]
        return theta, best_pred

    
    def gradAda(self, eta=0.1, lam=0.0, eps=1e-8, theta0=None):
        """
        AdaGrad with r_t accumulation and H_t^{-1/2} scaling as in the slides.
        Parameters
        ----------
        eta : float, default 0.1
            Base learning rate.
        lam : float, default 0.0
            Ridge penalty strength.
        eps : float, default 1e-8
            Numerical stabilizer inside the square root.
        theta0 : array-like or None, default None
            Optional warm-start parameters.
        Returns
        -------
        ndarray of shape (m,)
            The optimized parameter vector.
        """
        if theta0 is None:
            theta = np.zeros(self._m)
        else:
            theta = np.asarray(theta0).reshape(-1).copy()
        r = np.zeros_like(theta)
        for _ in range(self._iters):
            g = gradient(self._X, self._y, theta, lam=lam)
            if self.stopping(g):
                break
            r = r + g * g
            e = np.sqrt(r + eps)
            z = theta - eta * g / e
            if self._l1:
                alpha = e*lam
                theta = soft_threshold(z, alpha)
            else:
                theta = z
        
        return theta


    
    def gradADAM(self, eta=0.001, beta1=0.9, beta2=0.999, lam=0.0, eps=1e-8, bias_correction=True):
        """
        Adam optimizer for OLS/Ridge.
        Parameters
        ----------
        eta : float, default 0.001
            Base learning rate.
        beta1 : float, default 0.9
            Exponential decay for the first moment.
        beta2 : float, default 0.999
            Exponential decay for the second moment.
        lam : float, default 0.0
            Ridge penalty strength.
        eps : float, default 1e-8
            Numerical stabilizer.
        bias_correction : bool, default True
            Whether to use bias-corrected moments.
        Returns
        -------
        ndarray of shape (m,)
            The optimized parameter vector.
        """
        theta = np.zeros(self._m)
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        t = 0
        for _ in range(self._iters):
            grad = gradient(self._X, self._y, theta, lam=lam)

            if self.stopping(grad):
                break
            t += 1
            m = beta1 * m + (1.0 - beta1) * grad
            v = beta2 * v + (1.0 - beta2) * (grad * grad)

            if bias_correction:
                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)
            else:
                m_hat = m
                v_hat = v
            z = theta - eta * m_hat / (np.sqrt(v_hat) + eps)

            if self._l1:
                alpha = (eta / (np.sqrt(v_hat) + eps)) * lam
                theta = soft_threshold(z, alpha)
            
            else:
                theta = z
    
        return theta

        
    def gradStoc(self, batch_size=1, eta=0.1, lam=0.0, shuffle=True):
        """
        Stochastic (mini-batch) Gradient Descent for OLS/Ridge.
        Parameters
        ----------
        batch_size : int, default 1
            Number of samples per update step.
        eta : float, default 0.1
            Learning rate.
        lam : float, default 0.0
            Ridge penalty strength.
        shuffle : bool, default True
            Whether to shuffle indices between passes.
        Returns
        -------
        ndarray of shape (n_features,)
            The optimized parameter vector.
        """
        n_samples = self._X.shape[0]
        n_features = self._X.shape[1]
        minibatch_size = max(1, int(batch_size))

        theta = np.zeros(n_features)      
        data_indices = np.arange(n_samples)    
        steps = 0                       

        while steps < self._iters:
            if shuffle:
                np.random.shuffle(data_indices)

            for start_idx in range(0, n_samples, minibatch_size):
                if steps >= self._iters:
                    break

                batch_idx = data_indices[start_idx:start_idx + minibatch_size]
                X_batch = self._X[batch_idx]
                y_batch = self._y[batch_idx]

                grad = gradient(X_batch, y_batch, theta, lam=lam)
                if self.stopping(grad):
                    break

                z = theta - eta * grad
                if self._l1:
                    alpha = eta*lam
                    z = theta - eta*grad
                    theta = soft_threshold(z, alpha)
                else:
                    theta = z

                steps += 1

        return theta

    
    def gradRMS(self, eta=0.01, rho=0.9, lam=0.0, eps=1e-8):
        """
        RMSprop for OLS/Ridge.
        Parameters
        ----------
        eta : float, default 0.01
            Base learning rate.
        rho : float, default 0.9
            Exponential decay for the squared-gradient accumulator.
        lam : float, default 0.0
            Ridge penalty strength.
        eps : float, default 1e-8
            Numerical stabilizer.
        Returns
        -------
        ndarray of shape (m,)
            The optimized parameter vector.
        """
        theta = np.zeros(self._m)
        s = np.zeros_like(theta)
        for _ in range(self._iters):
            grad = gradient(self._X, self._y, theta, lam=lam)
            if self.stopping(grad):
                break
            s = rho * s + (1.0 - rho) * (grad * grad)
            z = theta - (eta / (np.sqrt(s) + eps)) * grad
            
            if self._l1:
                alpha = (eta / (np.sqrt(s) + eps)) * lam
                theta = soft_threshold(z, alpha)

            else:
                theta = z

        return theta

    
    def stopping(self, grad, e=None):
        """
        Early-stopping criterion based on the Euclidean norm of the gradient.
        Parameters
        ----------
        grad : array-like
            Current gradient vector.
        e : float or None, default None
            Absolute tolerance. Uses the instance tolerance when None.
        Returns
        -------
        bool
            True if the gradient norm is below tolerance, else False.
        """
        tol = self._eps if e is None else float(e)
        return float(np.linalg.norm(grad)) < tol
    

    def scale_eta(self, x, x0, x1):
        """
        Logarithmic learning-rate: x0 / (x + x1).
        Parameters
        ----------
        x : int or float
            Iteration index.
        x0 : float
            Numerator scale parameter.
        x1 : float
            Shift parameter to avoid division by zero.
        Returns
        -------
        float
            Scaled learning rate.
        """
        return x0 / (x + x1)   


class Resampling:
    """
    Implements resampling methods such as bootstrap and k-fold cross-validation.
    """

    def __init__(self, X, y):
        """
        Initialize the resampling class with data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Design matrix of input features.
        y : ndarray, shape (n_samples,)
            Target values.
        """
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.n_samples = self.X.shape[0]


    def bootstrap(self, n_bootstraps=100):
        """
        Generate bootstrap resamples of the dataset.

        Parameters
        ----------
        n_bootstraps : int, default=100
            Number of bootstrap resamples to generate.

        Returns
        -------
        list of tuples
            Each element is a tuple (X_resampled, y_resampled, X_oob, y_oob),
            where:
            - X_resampled, y_resampled are the bootstrap samples
            - X_oob, y_oob are the corresponding out-of-bag samples
        """
        resamples = []

        for _ in range(n_bootstraps):
            indices = np.random.randint(0, self.n_samples, self.n_samples)

            oob_indices = []
            for i in range(self.n_samples):
                if i not in indices:
                    oob_indices.append(i)

            X_resampled = self.X[indices]
            y_resampled = self.y[indices]
            X_oob = self.X[oob_indices]
            y_oob = self.y[oob_indices]

            resamples.append((X_resampled, y_resampled, X_oob, y_oob))

        return resamples
    

    def kCross(self, k=5, shuffle=True):
        """
        Perform k-fold cross-validation splitting.

        Parameters
        ----------
        k : int, default=5
            Number of folds.
        shuffle : bool, default=True
            If True, shuffle the dataset before splitting.

        Returns
        -------
        list of tuples
            Each element is a tuple (X_train, y_train, X_val, y_val),
            where one fold is used for validation and the rest for training.
        """
        indices = list(range(self.n_samples))
        if shuffle:
            np.random.shuffle(indices)

        fold_size = self.n_samples // k
        folds = []

        for i in range(k):
            start = i * fold_size
            if i == k - 1:  # last fold takes the remainder
                end = self.n_samples
            else:
                end = (i + 1) * fold_size
            folds.append(indices[start:end])

        splits = []
        for i in range(k):
            val_idx = folds[i]

            train_idx = []
            for j in range(k):
                if j != i:
                    train_idx.extend(folds[j])

            X_train = self.X[train_idx]
            y_train = self.y[train_idx]
            X_val = self.X[val_idx]
            y_val = self.y[val_idx]

            splits.append((X_train, y_train, X_val, y_val))

        return splits