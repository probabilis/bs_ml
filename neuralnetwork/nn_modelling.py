import numpy as np
import pandas as pd
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from repo_utils import loading

train, feature_cols, target_cols, targets_df, t20s, t60s = loading()

del targets_df

input_features_amount = len(feature_cols)
ifa = input_features_amount


model = nn.Sequential(
    nn.Linear(ifa, ifa * 2),
    nn.ReLU(),
    nn.Linear(ifa * 2, ifa),
    nn.ReLU(),
    nn.Linear(ifa, ifa / 2),
    nn.ReLU(),
    nn.Linear(ifa / 2, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

target = "target_cyrus_v4_20"

X_train, X_test, y_train, y_test = train_test_split(train[feature_cols], train[target], train_size=0.7, shuffle=True)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

n_epochs = 100
batch_size = 10
batch_start = torch.arange(0, len(X_train, batch_size))

best_mse = np.inf
best_weights = None
history = []

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit = "batch", mininterval = 0, diasble = True) as bar:
        for start in bar:
            X_batch = X_train[start : start+batch_size]
            y_batch = y_batch[start : start + batch_size]

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
        best_weights = copy.deepcopy(model.state_dict))

model.load_state_dict(best_weights)
print(best_mse)
plt.plot(history)
plt.savefig("mse_nn_test.py")
plt.show()
