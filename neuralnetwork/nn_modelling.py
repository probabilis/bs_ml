import numpy as np
import pandas as pd
import torch
import tqdm
import sys
import gc
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split 
sys.path.append("../")
from repo_utils import loading

train, feature_cols, target_cols, targets_df, t20s, t60s = loading()

del targets_df

gc.collect()

input_features_amount = len(feature_cols)

model = nn.Sequential(
    nn.Linear(input_features_amount, 1000),
    nn.ReLU(),
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 250),
    nn.ReLU(),
    nn.Linear(250, 1)
)

print("created model sucessfully")

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.1)

target = "target_cyrus_v4_20"

X_train, X_test, y_train, y_test = train_test_split(train[feature_cols], train[target], train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

del train 
gc.collect()

print("prepared data sucessfully")

n_epochs = 10
batch_size = 5_000
batch_start = torch.arange(0, len(X_train), batch_size)

best_mse = np.inf
best_weights = None
history = []

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit = "batch", mininterval = 0, disable = False) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            X_batch = X_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            bar.set_postfix(mse = float(loss))

    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred, y_test)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
        print("mse :", best_mse)
torch.save(model.state_dict(), "nn_model_0")

model.load_state_dict(best_weights)
print(best_mse)
plt.plot(history)
plt.savefig("mse_nn_test.png")
plt.show()
