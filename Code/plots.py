import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pnd
from functions import polynomial_features, scale, ols, ridge, mse, r_squared, runge
from sklearn.model_selection import train_test_split
from classes import GradientDescent, Resampling
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


def plotPD(deg, t, n=None, type_=" "):
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

    t: str
        Specifies which task (a or b). (a) uses OLS, 
        and (b) uses Ridge. Assumes either "a" or "b"
        is given as argument.

    n: array-like
        Specifies a list of data points to plot. 

    type_: string
        Specifies what is to be plotted.
        The code plots mse, r squared or beta values
        depending on the argument to type_.

    """

    if type_=="beta":
        plot_betas(deg,t)

    else: 
        plot_matrix = np.zeros((len(n), deg))

        for i, datapoints in enumerate(n):
            np.random.seed(1)
            x = np.linspace(-1, 1, datapoints)
            y = runge(x) + np.random.normal(0, 0.1, datapoints)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        
            for j in range(2, deg+2):
                X = polynomial_features(x_train, j)
                Y = polynomial_features(x_test, j)
                scaled_train = (scale(X, y))[0]
                scaled_test = (scale(Y, y))[0]
                if t == "a":
                    beta = ols(scaled_train, y_train)
                elif t == "b":
                    beta = ridge(scaled_train, y_train, 0.1)
                elif t == "lasso":
                    g = GradientDescent(scaled_train, 5000, y_train, l1=True)
                    beta = g.gradOrd(l=0.1)
                pred = scaled_test@beta + np.mean(y_train)
                if type_ == "mse":
                    plot_matrix[i, j-2] = mse(y_test, pred)
                elif type_ == "r2":
                    plot_matrix[i, j-2] = r_squared(y_test, pred)


        orders = np.arange(2, deg+2)

        fig, ax = plt.subplots(figsize=(6,4))

        for dp, no in enumerate(n):
            ax.plot(orders, plot_matrix[dp, : ], label=f"Datapoints: {n[dp]}")
        ax.set_xlabel("Polynomial degree", fontsize=13)
        if type_ == "mse":
            ax.set_ylabel("MSE", fontsize=15)
            ax.legend()
            fig.tight_layout
            if t=='a':
                ax.set_title("MSE OLS")
                fig.savefig("ols_mse.pdf")
            else:
                ax.set_title("MSE Ridge (lambda 0.01)")
                fig.savefig("ridge_mse.pdf")
        elif type_ == "r2":
            ax.set_ylabel("R2", fontsize=15)
            ax.legend()
            fig.tight_layout
            if t == "a":
                ax.set_title("R squared OLS")
                fig.savefig("r2_ols.pdf")
            elif t == "lasso":
                ax.set_title("R squared OLS")
            else:
                ax.set_title("R squared Ridge (lambda 0.01)")
                fig.savefig("r2_ridge.pdf")
        else:
            ax.set_ylabel("Beta", fontsize=14)
            ax.legend()
            fig.tight_layout
            if t=='a':
                ax.set_title("Beta OLS")
                fig.savefig("beta_a.pdf")
            else:
                ax.set_title("Beta Ridge (lambda 0.1)")
                fig.savefig('beta_ridge.pdf')
        plt.show()    


def heatMap(outer, inner, task, model, pd=False):
    """
    Heat map as a function of two variables.
    
    Parameters
    ----------

    """
    x = np.linspace(-1, 1, 1000)
    y = runge(x) + np.random.normal(0, 0.1, 1000)
    if task == "c":
        plot_c(outer, inner, model)
    
    else:
        plot_matrix = np.zeros((len(outer), len(inner)))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        if not pd:
            n = 6
            X = polynomial_features(x_train, n)
            Y = polynomial_features(x_test, n)
            scaled_train, y_scaled = scale(X, y)
            scaled_test = (scale(Y, y))[0]
            y_mean = np.mean(y)
            y_centered = y_train - y_mean
        for i, deg in enumerate(outer):
            if pd: 
                X = polynomial_features(x_train, deg)
                Y = polynomial_features(x_test, deg)
                scaled_train, y_scaled = scale(X, y)
                scaled_test = (scale(Y, y))[0]
                y_mean = np.mean(y)
                y_centered = y_train- y_mean
            for e, lamd in enumerate(inner):
                if pd:
                    if model=="ols":
                        beta = ols(scaled_train, y_centered)
                        pred = scaled_test@beta + np.mean(y_train)
                    elif model=="ridge":
                        beta = ridge(scaled_train, y_centered, lam=lamd)
                        pred = scaled_test@beta + np.mean(y_train)
                elif not pd:
                    grad = GradientDescent(scaled_train,10000, y_centered, l1=True)
                    betas = grad.gradRMS(eta=deg, lam=lamd)
                    pred = scaled_test@betas + y_mean
                pointMse = mse(y_test, pred)
                plot_matrix[i,e] = pointMse


        f, ax = plt.subplots(figsize=(9, 6))
       
        df = pnd.DataFrame(plot_matrix, index=outer, columns=inner)
        ax = sns.heatmap(df, annot = True,
                    cmap = 'coolwarm', linecolor = 'black',
                    linewidths = 2, robust = True)
        
        ax.set_title('Heatmap Analytical Ridge ', fontsize=16)
        ax.set_ylabel('Polynomial degrees', fontsize=16)
        ax.set_xlabel(r'$\lambda$', fontsize=16)
        f.savefig('Ridge_analyticalHM.pdf')
        plt.show()


def plot_betas(deg, t):
    """Plots the evolution of betas across polynomial degrees."""

    np.random.seed(2)
    x = np.linspace(-1, 1, 1000)
    y = runge(x) + np.random.normal(0, 0.1, 1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    beta_matrix = np.zeros((deg, deg))
    for j in range(1, deg+1):
        X = polynomial_features(x_train, j)
        scaled_train = (scale(X, y))[0]
        if t == "a":
            beta = ols(scaled_train, y_train)
        elif t == "b":
            beta = ridge(scaled_train, y_train, 0.01)
        else:
            g = GradientDescent(scaled_train, 5000, y_train, l1=True)
            beta = g.gradMomentum(lam=0.01)
        sparse_beta = np.pad(beta, (0, deg - len(beta)), mode='constant')
        beta_matrix[:, j-1] = sparse_beta


    
    orders = np.arange(2, deg+2)
    fig, ax = plt.subplots(figsize=(6,4))
    for d in range(deg):
        ax.plot(orders, beta_matrix[d, :], label=f"Beta {d+1}")

    ax.set_ylabel("Beta", fontsize=14)
    ax.set_xlabel("Polynomial degrees", fontsize=13)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout
    if t == "a":
        ax.legend(fontsize=9, loc="lower left", ncol=2)
        ax.set_title("Beta OLS")
        fig.savefig("beta_a.pdf")
    elif t == "b":
        ax.set_title("Beta Ridge (lambda 0.01)")
        ax.legend(fontsize=9, loc="lower left", ncol=2)
        fig.savefig("beta_b.pdf")
    else:
        ax.set_title("Beta LASSO (lambda 0.01)")
        ax.legend(fontsize=9, loc="lower right", ncol=2)
        fig.savefig("beta_l1.pdf")
    plt.show()


def plot_c(outer, inner, model):
    """Helper plot function for heatmap.
        Plots MSE as function of eta and polynomial degree.
    """
    x = np.linspace(-1, 1, 1000)
    y = runge(x) + np.random.normal(0, 0.1, 1000)
    plot_matrix = np.zeros((len(outer), len(inner)))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    n = 10
    X = polynomial_features(x_train, n)
    Y = polynomial_features(x_test, n)
    scaled_train, y_scaled = scale(X, y)
    scaled_test = (scale(Y, y))[0]
    y_mean = np.mean(y)
    for i, iters in enumerate(outer):
            
        for e, eta in enumerate(inner):
            grad = GradientDescent(scaled_train, iters, y_train)
            if model == "ols":
                betas = grad.gradOrd(eta=eta)
            else:
                betas = grad.gradOrd(iters, eta, l=0.01)
            pred = scaled_test@betas + y_mean
            pointMse = mse(y_test, pred)
            plot_matrix[i,e] = pointMse


    f, ax = plt.subplots(figsize=(9, 6))
    df = pnd.DataFrame(plot_matrix, index=outer, columns=inner)
    ax = sns.heatmap(df, annot = True,
                 cmap = 'coolwarm', linecolor = 'black',
                 linewidths = 2, robust = True)
    
    ax.set_ylabel('Iterations')
    ax.set_xlabel(r'$\eta$')
    if model== "ols":
        ax.set_title('Heatmap OLS GD')
        #f.savefig('gradOLS_c.pdf')
    else:
        ax.set_title('Heatmap Ridge GD (lambda = 0.01)')
        f.savefig("gradRidge_c.pdf")
    plt.show()


def convergence(loop, stoc=False, l1=False):
    """
    Routine that plots convergence for different GD methods against each other and analytical OLS.

    Parameters
    ----------
    loop: array-like
        A list of iterations for total convergence analysis, or batch sizes for SGD
    stoc: bool
        Determines if the function performs total convergence with all GD methods or just for SGD. 
    """

    np.random.seed(60)
    x = np.linspace(-1, 1, 1000)
    y = runge(x) + np.random.normal(0, 0.1, 1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=60)
    n = 6
    X = polynomial_features(x_train, n)
    Y = polynomial_features(x_test, n)
    scaled_train, y_scaled = scale(X, y)
    scaled_test = (scale(Y, y))[0]
    y_mean = np.mean(y)
    y_centered = y_train - y_mean


    iterations = [500,1000,2000,3500,5000,10000]
    plot_matrix = np.zeros((len(loop), 7))

    if stoc:
        for i, dp in enumerate(loop):
            for j, iters in enumerate(iterations):

                grad = GradientDescent(scaled_train, iters, y_centered, l1=l1)

                betaStoc = grad.gradStoc(batch_size=dp, eta=0.2)
                predS = scaled_test@betaStoc + y_mean
                mseS = mse(y_test, predS)
                plot_matrix[i][j] = mseS       

    
            betaOLS = ols(scaled_train, y_centered)
            predOLS = scaled_test@betaOLS + y_mean
            mseOLS = mse(y_test, predOLS)
            mseArray = np.full(len(iterations), mseOLS)
    
        fig, ax = plt.subplots(figsize=(9,4))

        for model, dp in enumerate(iterations):
            ax.plot(iterations, plot_matrix[:,model], label=f"Batch size: {loop[model]}")
        ax.plot(iterations, mseArray, label="Analytical OLS")
        ax.set_xlabel("Iterations", fontsize=12)
        ax.set_ylabel("MSE", fontsize=12)
        ax.set_title("MSE SGD as function of batchsizes")
        ax.legend()
        fig.tight_layout
        fig.savefig("convergence_sgd.pdf")
        plt.show()

    else:
        for i, iters in enumerate(loop):
            
            grad = GradientDescent(scaled_train, iters, y_centered, l1)
            # betaRidge = grad.gradOrd(iters=iters, eta=0.15, l=0.0001)
            # predRidge = scaled_test@betaRidge + y_mean
            # mseRidge = mse(y_test, predRidge)
            # plot_matrix[i][0] = mseRidge
            
            betaMom = grad.gradMomentum(eta=0.2)
            predMom = scaled_test@betaMom + y_mean
            mseMom = mse(y_test, predMom)
            plot_matrix[i][0] = mseMom

            betaStoc = grad.gradStoc(eta=0.2, batch_size=100)
            predS = scaled_test@betaStoc + y_mean
            mseS = mse(y_test, predS)
            plot_matrix[i][1] = mseS
            
            betaAda = grad.gradAda(eta=0.05)
            predAda = scaled_test@betaAda + y_mean
            mseAda = mse(y_test, predAda)
            plot_matrix[i][2] = mseAda
            
            betaRMS = grad.gradRMS(eta=0.001)
            predRMS = scaled_test@betaRMS + y_mean
            mseRMS = mse(y_test, predRMS)
            plot_matrix[i][3] = mseRMS
            
            betaAdam = grad.gradADAM(eta=0.1)
            predAdam = scaled_test@betaAdam + y_mean
            mseAdam = mse(y_test, predAdam)
            plot_matrix[i][4] = mseAdam

            betaOLS = ols(scaled_train, y_centered)
            predOLS = scaled_test@betaOLS + y_mean
            mseOLS = mse(y_test, predOLS)
            plot_matrix[i][5] = mseOLS

            betaGradOLS = grad.gradOrd(iters, eta=0.15)
            predGradOLS = scaled_test@betaGradOLS + y_mean
            mseGradOLS = mse(y_test, predGradOLS)
            plot_matrix[i][6] = mseGradOLS

        fig, ax = plt.subplots(figsize=(9,4))
        models = ["Momentum", "Stochastic GD", "AdaGrad", "RMS", "ADAM", "Analytical OLS", "Ordinary LASSO"]
        for model in range(7):
            ax.plot(loop, plot_matrix[:, model], label=f"Model: {models[model]}")
        ax.set_xlabel("Iterations", fontsize=15)
        ax.set_ylabel("MSE", fontsize=15)
        ax.set_title("Convergence Gradient Descent LASSO", fontsize=16)
        ax.legend()
        fig.tight_layout
        fig.savefig("convergence_GD.pdf")
        plt.show()



def poly_fit():
    """Funtion for fitting models to the Runge's function"""

    np.random.seed(3)
    x = np.linspace(-1, 1, 1000)
    y = runge(x) + np.random.normal(0, 0.03, 1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

    fig, ax = plt.subplots(figsize=(6,4))

    n = 10
    X = polynomial_features(x_train, n)
    Y = polynomial_features(x_test, n)

    
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(X)
    Xte_s = scaler.transform(Y)   

    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    g = GradientDescent(Xtr_s, 10000, y_train, eps=1e-5, l1=True)
    ridgeBetas = ridge(Xtr_s, y_centered, lam=0.01)
    pred = Xte_s@ridgeBetas + y_mean
    ols_betas = ols(Xtr_s, y_train)
    pred_ = Xte_s@ols_betas + y_mean
    lassoBetas = g.gradOrd(0.01, 0.1)
    lassoPred = Xte_s@lassoBetas + y_mean

    sort_indices = np.argsort(x_test.flatten())

    ax.plot(x,y, label="Runge")
    ax.plot(x_test[sort_indices], pred[sort_indices], label="Ridge (lam=0.01)")
    ax.plot(x_test[sort_indices], pred_[sort_indices], label="Analytical OLS")
    ax.plot(x_test[sort_indices], lassoPred[sort_indices], label="Lasso (lam=0.1)")
    ax.set_title("Example fitting (deg 10)")
    ax.legend()
    fig.savefig("fit_example03Noise.pdf")
    plt.show()


def test_ridge():
    """Example of benchmark testing with sklearn"""

    x = np.linspace(-1, 1, 1000)
    y = runge(x) + np.random.normal(0, 0.1, 1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    from sklearn.linear_model import SGDClassifier
    n = 6
    X = polynomial_features(x_train, n)
    Y = polynomial_features(x_test, n)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(X)
    Xte_s = scaler.transform(Y) 
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean

    from sklearn.linear_model import Ridge

    ridge_ = Ridge(alpha=0.01, fit_intercept=False)  
    ridge_.fit(Xtr_s, y_centered)
    y_pred = ridge_.predict(Xte_s)

    testRidge = ridge(Xtr_s, y_centered, lam=0.01)
    predRidge = Xte_s@testRidge

    return np.allclose(y_pred, predRidge)

def test_GD():
    model = SGDRegressor(
    loss="squared_error",
    learning_rate="constant",
    eta0=0.01,
    max_iter=1000,
    tol=1e-6,
    penalty=None,
    shuffle=False,
    random_state=42
    )
    x = np.linspace(-1, 1, 1000)
    y = runge(x) + np.random.normal(0, 0.1, 1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    n = 6
    X = polynomial_features(x_train, n)
    Y = polynomial_features(x_test, n)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(X)
    Xte_s = scaler.transform(Y) 
    y_mean = np.mean(y_train)
    y_centered = y_train - y_mean
    n_epochs = 1000
    for epoch in range(n_epochs):
        model.partial_fit(Xtr_s, y_centered)

    print("Weights:", model.coef_)
    print("Intercept:", model.intercept_)

    gradStochastic = GradientDescent(Xtr_s, 10000, y_centered)
    betaStoc = gradStochastic.gradOrd(eta=0.01)
    pred = Xte_s@betaStoc

    print(betaStoc
          )
    print
    (pred)


def mse_complexity(degs):
    """Function for test and train."""

    x = np.linspace(-1, 1, 100)
    y = runge(x) + np.random.normal(0, 0.1, 100)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    mse_trainList = list()
    mse_testList = list()


    for i, deg in enumerate(degs):
        X = polynomial_features(x_train, deg)
        Y = polynomial_features(x_test, deg)
        scaled_train, y_scaled = scale(X, y)
        scaled_test = (scale(Y, y))[0]
        y_mean = np.mean(y_train)
        y_centered = y_train - y_mean
        params = ols(scaled_train, y_centered)
        pred_train = scaled_train@params + y_mean
        pred_test = scaled_test@params + y_mean
        mseTrain = mse(y_train, pred_train)
        mseTest = mse(y_test, pred_test)
        mse_trainList.append(mseTrain)
        mse_testList.append(mseTest)

    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(degs, mse_trainList, marker='o', label="Training loss")
    ax.plot(degs, mse_testList, marker='s', label="Test loss")

    ax.set_xlabel("Polynomial Order")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("MSE train vs test (noise [0, 0.1], datapoints=100)")
    ax.legend()
    fig.savefig("f_mse_test_train2.pdf")
    plt.show()


def test_resampling():
    """Function for testing bootstrap"""

    r = Resampling()
    biases, variances, mses = r.bootstrap(1000, 100)


    fig, ax = plt.subplots(figsize=(8, 5))
    degrees = np.arange(2,16)
    datapoints = [20,50,100,200,500,1000]

    ax.plot(degrees, biases, marker='o', label='Bias')
    ax.plot(degrees, variances, marker='o', label='Variance')
    ax.plot(degrees, mses, label='MSE')

    ax.set_xlabel('Polynomial degree', fontsize=15)
    ax.set_ylabel('Average over test points', fontsize=15)
    ax.set_title('Bias–Variance vs. polynomial degree (1000 boots, 100 dps)', fontsize=15)

    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12) 

    fig.tight_layout()
    fig.savefig("g_bvto.pdf")
    plt.show()


def testK_fold():
    """Function for testing and comparing k-fold."""
    np.random.seed(67)
    x = np.linspace(-1, 1, 100)
    y = runge(x) + np.random.normal(0, 0.1, 100)
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    k = Resampling(x, y)

    degrees = np.arange(2,16)
    k_splits = 10

 
    mse_ols   = np.zeros((len(degrees), k_splits))
    mse_ridge = np.zeros_like(mse_ols)
    mse_lasso = np.zeros_like(mse_ols)

    kf = KFold(n_splits=k_splits, shuffle=True)

    for d_idx, d in enumerate(degrees):
        for f_idx, (tr, te) in enumerate(kf.split(x, y)):
            Xtr, Xte = x[tr], x[te]
            ytr, yte = y[tr], y[te]

            X = polynomial_features(Xtr, d)   
            Y = polynomial_features(Xte, d)
            scaler = StandardScaler(with_mean=True, with_std=True)
            Xtr_s = scaler.fit_transform(X)
            Xte_s = scaler.transform(Y) 
            y_mean = np.mean(ytr)
            y_centered = ytr - y_mean

            beta_ols = ols(Xtr_s, y_centered)
            yhat     = Xte_s @ beta_ols + y_mean
            mse_ols[d_idx, f_idx] = mse(yte, yhat)

            beta_r  = ridge(Xtr_s, y_centered, lam=0.01)
            ridge_yhat    = Xte_s @ beta_r + y_mean
            mse_ridge[d_idx, f_idx] = mse(yte, ridge_yhat)

            Grad = GradientDescent(Xtr_s,5000, y_centered, l1=True)
            beta_l  = Grad.gradMomentum(eta=0.1,lam=0.0001)
            lassoPred = Xte_s@beta_l + y_mean
            mse_lasso[d_idx, f_idx] = mse(yte, lassoPred)

    fig, ax = plt.subplots(figsize=(8,4))

    mean_ols   = mse_ols.mean(axis=1)
    mean_ridge = mse_ridge.mean(axis=1)
    mean_lasso = mse_lasso.mean(axis=1)

    ax.plot(degrees, mean_ols,   marker='o', label="OLS")
    ax.plot(degrees, mean_ridge, marker='s', label=f"Ridge (lam={0.01})")
    ax.plot(degrees, mean_lasso, marker='^', label=f"Lasso (lam={0.0001})")

    ax.set_xlabel("Polynomial degree", fontsize=16)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=16)
    ax.set_title(f"{k_splits}-Fold CV: MSE vs Polynomial Degree", fontsize=17)
    ax.legend()

    fig.tight_layout()
    fig.savefig("kfold_lowerl1.pdf")
    plt.show()