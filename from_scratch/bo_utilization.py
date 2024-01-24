"""
Author: https://github.com/pradeepsinngh/
"""
import numpy as np
import matplotlib.pyplot as plt

#CC to https://github.com/pradeepsinngh/Bayesian-Optimization/blob/master/bayesian_optimization_util.py
#which enables a plotting schematic of bayesian optimization very comfortable
#package is not avaible as pip


####################################################

def plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next=None, show_legend=False):
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(),
                     mu.ravel() + 1.96 * std,
                     mu.ravel() - 1.96 * std,
                     alpha=0.1)
    plt.plot(X, Y, color = "salmon", ls = '--', lw=1, label='Noise-free objective $Z(x)$')
    plt.plot(X, mu, color = 'cornflowerblue', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    plt.ylabel("$Z(x)$")
    plt.xlabel("$x$")
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()

def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, color = 'mediumseagreen', lw=1, label='Acquisition function EI')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    plt.ylabel("EI$(x)$")
    plt.xlabel("$x$")
    if show_legend:
        plt.legend()

def plot_convergence(X_sample, Y_sample, n_init=2):
    plt.figure(figsize=(12, 3))

    x = X_sample[n_init:].ravel()
    y = Y_sample[n_init:].ravel()
    r = range(1, len(x)+1)

    x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]
    y_max_watermark = np.maximum.accumulate(y)

    plt.subplot(1, 2, 1)
    plt.plot(r[1:], x_neighbor_dist, color = "salmon")
    plt.xlabel('Iteration i')
    plt.ylabel('$\Delta x$')
    plt.title('Distance $\Delta x$ between consecutive $x_i$ of $Z(x)$')

    plt.subplot(1, 2, 2)
    plt.plot(r, y_max_watermark, color = 'cornflowerblue')
    plt.xlabel('Iteration i')
    plt.ylabel('Best function value $\\tilde{Z}(x)$')
    plt.title('Values of best selected sample of $Z(x)$ over iterations')
