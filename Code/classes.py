import numpy as np
from functions import soft_threshold, mse, gradient, bias, var, polynomial_features, runge, scale, ols
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class GradientDescent:
    """
    Gradient descent class.
    Methods performind different types of Gradient descent. 
    """
    def __init__(self, X_norm, iters, y_centered, eps=1e-6, l1=False):
        self._X = X_norm
        self._iters = iters
        self._eps = eps
        self._y = y_centered
        self._m = X_norm.shape[1]
        self._l1 = l1


    
    def gradOrd(self, iters=None, eta=0.1, l=0):
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
        theta = np.zeros(self._X.shape[1])


        for t in range(self._iters):
            grad = gradient(self._X, self._y, theta, lam=l)

            z = theta - eta * grad
            
            if self._l1:
                alpha = eta*l
                z = theta - eta*grad
                theta = soft_threshold(z, alpha)
                
            theta = z
            
            if t!= 0  and  self.stopping(grad):
                break
        
        return theta

    
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
            velocity = eta*grad + momentum*velocity
            z = theta - velocity
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
  
        return theta

    
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


    
    def gradADAM(self, eta=0.1, beta1=0.9, beta2=0.999, lam=0.0, eps=1e-8):
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
            Numerical stabilizer
        Returns
        -------
        ndarray of shape (m,)
            The optimized parameter vector.
        """
        theta = np.zeros(self._m)
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        t = 0
        first_moment = 0.0
        second_moment = 0.0

        for iter in range(1, self._iters+1):
            grad = gradient(self._X, self._y, theta, lam=lam)

            if self.stopping(grad):
                break
            
            else:
                first_moment = beta1*first_moment + (1-beta1)*grad
                second_moment = beta2*second_moment+(1-beta2)*grad*grad
                first_term = first_moment/(1.0-beta1**iter)
                second_term = second_moment/(1.0-beta2**iter)
                update = eta*first_term/(np.sqrt(second_term)+eps)
                theta = update
    
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

        x0 = 5
        x1 = 10
        eta = x0/x1    

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
                #eta = self.scale_eta(steps, x0,x1)#. Using dynamic step size. 

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
            z =  (eta / (np.sqrt(s) + eps)) * grad
            theta -= z
            
            if self._l1:
                alpha = (eta / (np.sqrt(s) + eps)) * lam
                theta -= soft_threshold(z, alpha)


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
       
        return float(np.linalg.norm(grad)) < self._eps
    

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

    def __init__(self, X=None, y=None):
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
        #self.n_samples = self.X.shape[0]


    def bootstrap(self, bootstraps=1000, dp=100, hm=False):
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
        np.random.seed(16)
        x = np.linspace(-1, 1, dp)
        y = runge(x) + np.random.normal(0,0.1,dp)

        biases = []
        variances = []
        mses = []
        X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=16)
        predictions = np.zeros((bootstraps, len(y_test)))
        flatPreds = []
        targetsFlat = []
        targets = np.zeros((bootstraps, len(y_test)))
        datapoints = [20,50,100,200,500,1000]

        if hm:
            for d, dp in enumerate(datapoints):
                x = np.linspace(-1, 1, dp)
                y = runge(x) + np.random.normal(0,0.2,dp)
                predictions = np.zeros((bootstraps, len(y_test)))
                targets = np.zeros((bootstraps, len(y_test)))
                X = polynomial_features(X_train, 10)
                Y = polynomial_features(X_test, 10)
                scaled_train, y_scaled = scale(X, y)
                scaled_test = (scale(Y, y))[0]
                y_mean = np.mean(y)
                y_centered = y_train - y_mean
                #model = make_pipeline(PolynomialFeatures(degree=p), LinearRegression(fit_intercept=False))
                for b in range(bootstraps):
                    x_sample, y_sample = resample(scaled_train, y_centered)
                    #this is where you fit your model on the sampled data
                    betaOLS = ols(x_sample, y_sample)
                    bootstrap_pred = scaled_test@betaOLS + y_mean

                    predictions[b, :] = bootstrap_pred
                    targets[b, :] = y_test

                from sklearn.metrics import mean_squared_error
                biases.append(bias(targets, predictions))
                variances.append(var(predictions))
                mses.append(mse(targets, predictions))

            return biases, variances, mses
                

        else:
            for p in range(2, 16):
                predictions = np.zeros((bootstraps, len(y_test)))
                targets = np.zeros((bootstraps, len(y_test)))
                X = polynomial_features(X_train, p)
                Y = polynomial_features(X_test, p)
                scaled_train, y_scaled = scale(X, y)
                scaled_test = (scale(Y, y))[0]
                y_mean = np.mean(y)
                y_centered = y_train - y_mean
                for b in range(bootstraps):
                    x_sample, y_sample = resample(scaled_train, y_centered)
                    betaOLS = ols(x_sample, y_sample)
                    bootstrap_pred = scaled_test@betaOLS + y_mean

                    predictions[b, :] = bootstrap_pred
                    targets[b, :] = y_test

                biases.append(bias(targets, predictions))
                variances.append(var(predictions))
                mses.append(mse(targets, predictions))

            return biases, variances, mses


    

    def kCross(self, k=5, shuffle=True):
        """
        Performs k-fold cross-validation splitting. Redundant (?)
        Use for Benchmark sklearn.

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
            if i == k - 1: 
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