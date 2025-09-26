# from imports import *
# from functions import polynomial_features, scale, ols, ridge, mse, r_squared
# from classes import GradientDescent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pnd
from functions_ import polynomial_features, scale, ols, ridge, mse, r_squared, runge
from sklearn.model_selection import train_test_split
from classes import GradientDescent
from scipy import sparse

def plotPD(deg, t, n=1000, type_=" "):
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
            np.random.seed(42)
            x = np.linspace(-1, 1, datapoints)
            y = runge(x) + np.random.normal(0, 0.1, datapoints)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        
            for j in range(1, deg+1):
                X = polynomial_features(x_train, j)
                Y = polynomial_features(x_test, j)
                scaled_train = (scale(X, y))[0]
                scaled_test = (scale(Y, y))[0]
                if t == "a":
                    beta = ols(scaled_train, y_train)
                elif t == "b":
                    beta = ridge(scaled_train, y_train, 0.1)
                pred = scaled_test@beta + np.mean(y_train)
                if type_ == "mse":
                    plot_matrix[i, j-1] = mse(y_test, pred)
                elif type_ == "r2":
                    plot_matrix[i, j-1] = r_squared(y_test, pred)


        orders = np.arange(1, deg+1)

        fig, ax = plt.subplots(figsize=(6,4))

        for dp, no in enumerate(n):
            ax.plot(orders, plot_matrix[dp, : ], label=f"Datapoints: {n[dp]}")
        ax.set_xlabel("Polynomial degree")
        if type_ == "mse":
            ax.set_ylabel("MSE")
            ax.set_title("MSE Test")
            ax.legend()
            fig.tight_layout
            fig.savefig("ols_mse.pdf")
        elif type_ == "r2":
            ax.set_ylabel("R2")
            ax.set_title("R Squared Test")
            ax.legend()
            fig.tight_layout
            if t == "a":
                fig.savefig("r2_ols.pdf")
            else:
                fig.savefig("r2_ridge.pdf")
        else:
            ax.set_ylabel("Beta")
            ax.title(f"Beta Test")
            ax.legend()
            fig.tight_layout
            fig.savefig("beta_a.pdf")
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
            for e, lamd in enumerate(inner):
                if not pd:
                    beta = ols(scaled_train, y_train)
                    pred = scaled_test@beta + np.mean(y_train)
                elif  pd:
                    grad = GradientDescent(scaled_train, y_train, 750)
                    betas = (grad.gradOrd())[0]
                    pred = scaled_test@betas + y_mean
                pointMse = mse(y_test, pred)
                plot_matrix[i,e] = pointMse


        f, ax = plt.subplots(figsize=(9, 6))
        # x_vals = your x grid (e.g., log10(λ): [-8,-7,...,0])
            # y_vals = your y grid (e.g., degrees: [0,1,...,7])
        # Z shape must be (len(y_vals), len(x_vals))
        df = pnd.DataFrame(plot_matrix, index=outer, columns=inner)
        ax = sns.heatmap(df, annot = True,
                    cmap = 'coolwarm', linecolor = 'black',
                    linewidths = 2, robust = True)
        
        ax.set_title('Heatmap')
        ax.set_ylabel('Polynomial degrees')
        ax.set_xlabel('lambda')
        f.savefig('ridge_heatmap.pdf')
        plt.show()


## Auxiliary plot functions

def plot_betas(deg, t):

    x = np.linspace(-1, 1, 1000)
    y = runge(x) + np.random.normal(0, 0.1, 1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    beta_matrix = np.zeros((deg, deg))
    for j in range(1, deg+1):
            X = polynomial_features(x_train, j)
            scaled_train = (scale(X, y))[0]
            if t == "a":
                beta = ols(scaled_train, y_train)
            elif t == "b":
                beta = ridge(scaled_train, y_train, 0.2)
            sparse_beta = np.pad(beta, (0, deg - len(beta)), mode='constant')
            beta_matrix[:, j-1] = sparse_beta
    
    orders = np.arange(1, deg+1)
    fig, ax = plt.subplots(figsize=(6,4))
    for d in range(deg):
        ax.plot(orders, beta_matrix[ : ,d], label=f"Beta {d+1}")

    ax.set_ylabel("Beta")
    ax.set_title("Beta Test")
    ax.legend()
    fig.tight_layout
    if t == "a":
        fig.savefig("beta_a.pdf")
    else:
        fig.savefig("beta_b.pdf")
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
            grad = GradientDescent(scaled_train, y_train, iters)
            if model == "ols":
                betas = (grad.gradOrd(eta))[0]
            else:
                betas = (grad.gradOrd(eta, l = 0.001))[0]
            pred = scaled_test@betas + y_mean
            pointMse = mse(y_test, pred)
            plot_matrix[i,e] = pointMse


    f, ax = plt.subplots(figsize=(9, 6))
    # x_vals = your x grid (e.g., log10(λ): [-8,-7,...,0])
        # y_vals = your y grid (e.g., degrees: [0,1,...,7])
    # Z shape must be (len(y_vals), len(x_vals))
    df = pnd.DataFrame(plot_matrix, index=outer, columns=inner)
    ax = sns.heatmap(df, annot = True,
                 cmap = 'coolwarm', linecolor = 'black',
                 linewidths = 2, robust = True)
    
    ax.set_title('Heatmap')
    ax.set_ylabel('Iterations')
    ax.set_xlabel(r'$\eta$')
    if model== "c":
        f.savefig('gradOLS_c.pdf')
    else:
        f.savefig("gradRidge_c.pdf")
    plt.show()

def convergence():

    x = np.linspace(-1, 1, 1000)
    y = runge(x) + np.random.normal(0, 0.1, 1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    fig, ax = plt.subplots(figsize=(6,4))

    n = 10
    X = polynomial_features(x_train, n)
    Y = polynomial_features(x_test, n)
    scaled_train, y_scaled = scale(X, y)
    scaled_test = (scale(Y, y))[0]
    y_mean = np.mean(y)

    its = [10,50,100,200,400,600,800,1000]

    plot_matrix = np.zeros((len(its), 6))

    for i, iters in enumerate(its):
        grad = GradientDescent(scaled_train, y_train, iters)

        betaOrd = (grad.gradOrd())[0]
        predOrd = scaled_test@betaOrd
        mseOrd = mse(y_test, predOrd)
        plot_matrix[i][0] = mseOrd
        
        betaMom = (grad.gradMomentum(0.9))[0]
        predMom = scaled_test@betaMom
        mseMom = mse(y_test, predMom)
        plot_matrix[i][1] = mseMom

        betaStoc = grad.gradStoc(batch_size=10)
        predS = scaled_test@betaStoc
        mseS = mse(y_test, predS)
        plot_matrix[i][2] = mseS
        
        betaAda = grad.gradAda()
        predAda = scaled_test@betaAda
        mseAda = mse(y_test, predAda)
        plot_matrix[i][3] = mseAda
        
        betaRMS = grad.gradRMS(rho=0.2)
        predRMS = scaled_test@betaRMS
        mseRMS = mse(y_test, predRMS)
        plot_matrix[i][4] = mseRMS
        
        betaAdam = grad.gradADAM(beta1=0.5, beta2=0.75)
        predAdam = scaled_test@betaAdam
        mseAdam = mse(y_test, predAdam)
        plot_matrix[i][5] = mseAdam

    for model in range(6):
        ax.plot(its, plot_matrix[:, model], label=f"Model: {model}")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("MSE")
    ax.set_title("MSE Gradient Descent")
    ax.legend()
    fig.tight_layout
    fig.savefig("convergence_d.pdf")
    plt.show()


def poly_fit():
    x = np.linspace(-1, 1, 1000)
    y = runge(x) + np.random.normal(0, 0.1, 1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    fig, ax = plt.subplots(figsize=(6,4))

    n = 10
    X = polynomial_features(x_train, n)
    Y = polynomial_features(x_test, n)
    scaled_train, y_scaled = scale(X, y)
    scaled_test = (scale(Y, y))[0]
    y_mean = np.mean(y)
    gradOLS = ols(X, y_train)
    pred = scaled_test@gradOLS

    plt.figure()
    plt.plot(x,y)
    plt.plot(x, pred)
    plt.show()