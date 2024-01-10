"""
Author: Maximilian Gschaider
MN: 12030366
"""
import sys
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('../')
from repo_utils import repo_path, fontsize, fontsize_title

########################################
#hyperparameters
bo_file = "params_bayes_ip=10_ni=100_2023-12-18_n=full.txt"

with open(repo_path + "/outputs/" + bo_file) as file:
    lines = file.readlines()

    count = 0

    targets = [] ; cbts = [] ; lrs = []; mds = [] ; nests = []

    for line in lines:
        count += 1
        Line = line.strip()
        if not count % 2:
            Dict = eval(Line)
            
            target = Dict["target"]
            cbt = Dict["params"]["colsample_bytree"]
            lr = Dict["params"]["learning_rate"]
            md = Dict["params"]["max_depth"]
            n_est = Dict["params"]["n_estimators"]

            targets.append(target)
            cbts.append(cbt)
            lrs.append(lr)
            mds.append(md)
            nests.append(n_est)

df = pd.DataFrame({
    "target" : targets,
    "colsample_bytree" : cbts,
    "learning_rate" : lrs,
    "max_depth" : mds,
    "n_estimators" : nests
    })
print(df)

max_row_index = df["target"].idxmax()
max_row = df.loc[max_row_index]
print(max_row)

df.at[max_row_index, "target"] += 0.1

def plot_hyperparameter_scatter_plot(save_plot) -> None:

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12,6)

    fig.suptitle("Scatter plot of objective function $Z_F(\\theta_i)$ BO iterations over the hyperparameter space $\\theta$", fontsize = fontsize_title)

    cm = plt.cm.get_cmap('Spectral')
    im_1 = axs[0].scatter(df["max_depth"], df["n_estimators"], c = df["target"], cmap = cm, s = 100)
    axs[0].set_xlabel("max. depth / $d_{max}$", fontsize = fontsize)
    axs[0].set_ylabel("nr. of trees / $n_{trees}$ = $m$", fontsize = fontsize)
    cbar = plt.colorbar(im_1)
    cbar.set_label('$Z_F(\\theta_i)$', rotation=90, fontsize = fontsize)

    im_2 = axs[1].scatter(df["learning_rate"], df["colsample_bytree"], c= df["target"], cmap = cm, s = 100)

    axs[1].set_xlabel("learning rate / $\\nu$", fontsize = fontsize)
    axs[1].set_ylabel("colsample by tree / $\\epsilon$", fontsize = fontsize)
    cbar = plt.colorbar(im_2)
    cbar.set_label('$Z_F(\\theta_i)$', rotation=90, fontsize = fontsize)

    fig.tight_layout()
    if save_plot == True:
        plt.savefig(repo_path + "/figures/" + "bo_numerai_scatter_plot.png", dpi=300)
    plt.show()


plot_hyperparameter_scatter_plot(save_plot = True)
