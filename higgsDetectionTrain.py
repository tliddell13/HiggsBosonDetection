# This file is for training the MLP's on a larger amount of the data using a GPU
import sys
import torch
import pickle
import torch.nn as nn
import torch.optim as optim 
import MLPfunctions as mlp
import pandas as pd
import numpy as np
import importlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# Load the data
dataset = pd.read_csv('HIGGS_train.csv')
# Take about a third of the data (a couple million rows)
dataset = dataset.sample(frac=0.33)
# This time I am going to use cyclic feature encoding on the angular features and scaling all the features
angular_feats = ['lepton phi', 'missing energy phi', 'jet 1 phi', 'jet 2 phi', 'jet 3 phi', 'jet 4 phi']
X = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

# Make it cyclical! (sin and cos)
for feat in angular_feats:
    sin = feat + '_sin'
    cos = feat + '_cos'
    X = X.assign(**{sin: np.sin(X[feat] * np.pi / 180)})
    X = X.assign(**{cos: np.cos(X[feat] * np.pi / 180)})

# Drop the original features
X = X.drop(angular_feats, axis=1)

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Set aside some validation data
X_val = X[:290000]
y_val = y[:290000].values

# Remove the validation data from the dataset
X = X[290000:]
y = y[290000:].values

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Print all the sizes
print("Sizes:")
print("Training")
print(X_train.shape)
print("Testing")
print(X_test.shape)
print("Validation")
print(X_val.shape)

# Convert the data to PyTorch tensors
X_train = torch.FloatTensor(X_train,)
X_test = torch.FloatTensor(X_test)
X_val = torch.FloatTensor(X_val)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

input_size = X_train.shape[1]

# Set the tensors to the device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)

model5 = mlp.MLP_mach5(input_size, 260, 200, 140, 100, 60, 20, dropout=.2)
model5.to(device)

n_epochs = 6000
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model5.parameters(), lr=0.002)
train_losses, test_losses = mlp.train_model_dist(model5, X_train, y_train, X_test, y_test, criterion, optimizer, n_epochs,patience=800)
# Evaluate the model using our function
f1, acc, cm = mlp.getResults(train_losses, test_losses, model5, X_val, y_val)
print(f"F1: {f1}")
print(f"Accuracy: {acc}")
print(f"Confusion Matrix: {cm}")
# Save the model
torch.save(model5.state_dict(), 'bn_model5.pth')
# Save the training and testing losses as well  
with open('bn_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)

with open('bn_losses.pkl', 'wb') as f:
    pickle.dump(test_losses, f)





