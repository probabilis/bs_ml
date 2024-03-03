"""
Author: Maximilian Gschaider
MN: 12030366
"""
import numpy as np
import pandas as pd
import tqdm
import sys
import gc
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
#############################################
sys.path.append("../")
from repo_utils import fontsize,fontsize_title, loading, gh_repos_path

#############################################
train, feature_cols, target_cols, targets_df, t20s, t60s = loading()
target = "target_cyrus_v4_20"

del targets_df
gc.collect()
#############################################
#NN initialzation

model = nn.Sequential(
    nn.Linear(len(feature_cols), 1000),
    nn.ReLU(),
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 250),
    nn.ReLU(),
    nn.Linear(250, 1)
)

loss_fn = nn.MSELoss()
#Using the adpative Adam optimizer
#https://www.lightly.ai/post/which-optimizer-should-i-use-for-my-machine-learning-project
optimizer = optim.Adam(model.parameters(), lr = 0.01)

print("created model sucessfully")

#############################################
#split up training data for back-propagation

X_train, X_test, y_train, y_test = train_test_split(train[feature_cols], train[target], train_size=0.7, shuffle=True)
X_train = torch.tensor(train[feature_cols].values, dtype=torch.float32)
y_train = torch.tensor(train[target].values, dtype=torch.float32).reshape(-1, 1)

last_train_era = int(train["era"].unique()[-1])

del train
gc.collect()

#can be used for validating with parquet file instead of back-optimizing the model with the training data
#side information: used this dataset but the overall convergence rate was lower because of the incosistent mapping between train & validation data
"""
validation = pd.read_parquet(gh_repos_path + "/validation.parquet", columns = ["era", "data_type"] + feature_cols + target_cols)

validation = validation[validation["data_type"] == "validation"]

del validation["data_type"]

validation = validation[validation["era"].isin(validation["era"].unique()[::4])]

eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]

X_test = torch.tensor(validation[feature_cols].values, dtype = torch.float32)
y_test = torch.tensor(validation[target].values, dtype = torch.float32).reshape(-1, 1)
"""

#needed if validation data should be extracted from training data
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

#del validation
gc.collect()

print("prepared data sucessfully")

#############################################
#number of total ebochs for training the MLP-NN regression model
n_epochs = 10
#batch size for splitting the training dataframe up into number of total -> (N rows of the training dataframe) / (batch_size) 
batch_size = 40
batch_start = torch.arange(0, len(X_train), batch_size)

best_mse = np.inf
best_weights = None
history = []

#Optimization Loop for training the MLP-NN structure
for epoch in range(n_epochs):
    #initilization of the training model
    model.train()
    with tqdm.tqdm(batch_start, unit = "batch", mininterval = 0, disable = False) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            #initilization of the X & y batch frame through batch_size parameter
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]

            #calculation of prediction values y_pred through current model parametrization 
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            #setting the gradients to zero (the gradients get accumulated on subsequent backward propagation)
            #but the old gradients should be set to zero because of different X & y sample room (each batch_size induces a new batch room for training the NN model)
            optimizer.zero_grad()
            #computing dloss/dx for every parameter x (get accumulated into x.grad for every parameter x) 
            # x.grad += dloss/dx
            loss.backward()
            #updating the value of x using the current gradient x
            optimizer.step()
            bar.set_postfix(mse = float(loss))
    #evaluating current model for measureing performance through loss function via L² metric
    model.eval()
    y_pred = model(X_test)
    #calcuating L²
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    #appending current MSE @history
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
        print("mse :", best_mse)

#############################################
torch.save(model.state_dict(), f"nn_model_n_epochs={n_epochs}_batch_size={batch_size}")
model.load_state_dict(best_weights)

if __name__ == "__main__":
    print(best_mse)
    fig, ax = plt.subplots(1)
    fig.set_size_inches(10,6)

    #plotting the MSE over epoch iterations
    ax.set_title('Deviance $\\Delta$ / MSE over epoch iterations $i$ from Neural Network (PyTorch)', fontsize = fontsize_title)
    ax.legend(loc='upper right', fontsize = fontsize)
    ax.set_xlabel('Epoch iterations $i$', fontsize = fontsize)
    ax.set_ylabel('Deviance $\Delta$ / MSE ', fontsize = fontsize)
    ax.plot(history)
    fig.tight_layout()
    plt.savefig(f"loss_deviance_mse_nn_n_epochs={n_epochs}_batch_size={batch_size}.png")
    plt.show()
