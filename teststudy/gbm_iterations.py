"""
Author: Maximilian Gschaider
MN: 12030366
"""
########################################
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
sys.path.append('../')
from from_scratch.gradient_boosting_from_scratch import GradientBoosting
from testfunction import testfunction, X
from repo_utils import repo_path

########################################
#testfunction
Y = testfunction(X, noise = 0.5)
Y_ = testfunction(X, noise = 0)

########################################
#hyperparameters
learning_rate = 0.1
n_trees = 1
max_depth = 1

X_0 = X.reshape(-1,1)

########################################

def linear_regression(save_plot) -> None:
    y = testfunction(X, noise = 0.5)

    p = np.polyfit(X, y, 1)
    n = y.size
    m = p.size
    dof = n - m
    # Significance level
    alpha = 0.05
    # We're using a two-sided test
    tails = 2
    t_critical = stats.t.ppf(1 - (alpha / tails), dof)
    # Model the data using the parameters of the fitted straight line
    y_model = np.polyval(p, X)
    # Create the linear (1 degree polynomial) model
    model = np.poly1d(p)
    y_model = model(X)
    y_bar = np.mean(y)

    # Coefficient of determination, R²
    R2 = np.sum((y_model - y_bar)**2) / np.sum((y - y_bar)**2)
    print(f'R² = {R2:.2f}')

    resid = y - y_model

    chi2 = sum((resid / y_model)**2)

    chi2_red = chi2 / dof

    std_err = np.sqrt(sum(resid**2) / dof)

    plt.scatter(X, y, c='gray', marker='o', edgecolors='k', s=18)

    xlim = plt.xlim() ; ylim = plt.ylim()

    plt.scatter(X, y, c='gray', marker='o', edgecolors='k', s=18, label = 'sample points')
    plt.plot(np.array(xlim), p[1] + p[0] * np.array(xlim), label=f'Line of Best Fit, R² = {R2:.2f}')

    x_fitted = np.linspace(xlim[0], xlim[1], 100)
    y_fitted = np.polyval(p, x_fitted)
    # Confidence interval
    ci = t_critical * std_err * np.sqrt(1 / n + (x_fitted - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
    plt.fill_between(
        x_fitted, y_fitted + ci, y_fitted - ci, facecolor='#b9cfe7', zorder=0,
        label= '95 % Confidence Interval')
    # Prediction Interval
    pi = t_critical * std_err * np.sqrt(1 + 1 / n + (x_fitted - np.mean(X))**2 / np.sum((X - np.mean(X))**2))
    plt.plot(x_fitted, y_fitted - pi, '--', color='0.5', label='95 % Prediction Limits')
    plt.plot(x_fitted, y_fitted + pi, '--', color='0.5')

    plt.xlabel('x') ; plt.ylabel('F(x)')
    plt.title('Linear Regression on test function F(x) = [sin(x) + ln(x)] + $\Theta$ $\cdot \mathcal{N}(\cdot)$ at domain x $\in (0, 10]$', fontsize = 12)

    plt.legend(fontsize=8)
    #plt.xlim(xlim)
    #plt.ylim(ylim)
    fig = plt.gcf()
    fig.set_size_inches(12,8)

    if save_plot == True:
        plt.savefig(repo_path + "/figures/" + 'linear_regression.png', dpi=300)
    plt.show()

linear_regression(save_plot = False)

########################################

def gbm_iterations(save_plot) -> None:
    k = 5
    fig, axs = plt.subplots(k,2, sharex = True)
    fig.set_size_inches(12,18)
    #fig.suptitle('Gradient Boosting with Decision Tree Regressor from Scratch with max. depth of trees $d_{max}$, nr. of trees $n_{trees}$ and learning rate $\\alpha$ = ' + str(learning_rate))

    colors = ['mediumseagreen','lightskyblue',"mediumpurple",'salmon',"palevioletred"]

    Y_previous = np.zeros(k, dtype = object)

    k_ = [0, 0, 10 , 20, 30]

    for i in range(k):
        if i == 0:
            n_trees = 1
            y_hat = np.ones(len(X))
            y_hat[:] = np.mean(Y)
        else:
            n_trees = i * 10
            gbm = GradientBoosting(learning_rate, max_depth, n_trees, X_0, Y)
            gbm.fit(X_0,Y)
            y_hat, h_m = gbm.predict(X_0)

        Y_previous[i] = y_hat

        if i != 0:
            residuals = Y - Y_previous[i-1]
            markerline, stemlines, baseline = axs[i][0].stem(X, residuals, linefmt='grey', markerfmt='D', bottom=1.1, label = "y - $F_{%s}(x)$" %(k_[i]))
            markerline.set_markerfacecolor('none')
            axs[i][0].plot(X, h_m[-1],label = "$h_{%s}$(x)" %(n_trees), color = 'darkcyan', linewidth = 3)
            axs[i][0].set_ylim(-5,5)
        
        axs[i][1].plot(X, y_hat, color = colors[i], label = "$\\tilde{F}_{%s}(x) = \hat{y}$" %(n_trees), linewidth = 5)
        axs[i][1].scatter(X, Y, color = 'gray', marker='o', edgecolors='k', s=18, label = "sample points")

        axs[i][0].set_title('$n_{trees}$ = ' + str(n_trees), loc = 'left', pad=10)
        axs[i][1].set_title('$n_{trees}$ = ' + str(n_trees), loc = 'left', pad=10)

        axs[i][0].set_ylabel("residuals")
        axs[i][1].set_ylabel("F(x)")
    
        axs[i][0].legend()
        axs[i][1].legend()
    
    axs[-1][0].set_xlabel("x")
    axs[-1][1].set_xlabel("x")

    fig.suptitle("Gradient Boosting with Decision Tree Regressor from Scratch with max. depth of trees $d_{max}$ = " + str(max_depth) + ", learning rate $\\alpha$ = " + str(learning_rate) + " and nr. of trees $n_{trees}$", fontsize = 12)
    fig.tight_layout()

    if save_plot == True:
        plt.savefig(repo_path + "/figures/" + "gbm_iterations.png", dpi=300)
    plt.show()

#gbm_iterations(save_plot = False)
