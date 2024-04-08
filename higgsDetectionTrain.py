# This file is formatted to train the MLP models using the GPU
import torch
import pickle
import torch.nn as nn
import torch.optim as optim 
import MLPfunctions as mlp
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load the data
dataset = pd.read_csv('HIGGS_train.csv')
dataset = dataset.sample(frac=0.33)
# This time I am going to use cyclic feature encoding on the angular features and scaling all the features
# I discuss this more in the data manip ipynb
angular_feats = ['lepton phi', 'missing energy phi', 'jet 1 phi', 'jet 2 phi', 'jet 3 phi', 'jet 4 phi']
X = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

# Make it cyclical! (sin and cos)
for feat in angular_feats:
    sin = feat + '_sin'
    cos = feat + '_cos'
    X = X.assign(**{sin: np.sin(X[feat] * np.pi / 180)})
    X = X.assign(**{cos: np.cos(X[feat] * np.pi / 180)})

# Drop the original angle features
X = X.drop(angular_feats, axis=1)

# Use the scaler to scale the data
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

input_size = X_train.shape[1] # Number of features

# Set the tensors to the device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)

# Below I test out 5 different models with different architectures. Refer to the MLPfunctions.py file for the definitions.
# Model 1 
model1 = mlp.MLP_mach1(input_size, 30)
model1.to(device)
n_epochs = 4000
lr = 0.0098
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=lr)
# Measure the time it takes to train the model
start = time.time()
train_losses, test_losses = mlp.train_model(model1, X_train, y_train, X_test, y_test, criterion, optimizer, n_epochs,patience=800)
end = time.time()
print(f"Training time: {end - start}")
# Evaluate the model using our function
f1, acc, cm = mlp.getResults(train_losses, test_losses, model1, X_val, y_val)
print(f"F1: {f1}")
print(f"Accuracy: {acc}")
print(f"Confusion Matrix: {cm}")
# Save the model
torch.save(model1.state_dict(), 'model1.pth')
# Save the training and testing losses as well
with open('model1_train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)

with open('model1_test_losses.pkl', 'wb') as f:
    pickle.dump(test_losses, f)


# Model 2
model2 = mlp.MLP_mach2(input_size, 200, 200, 200, 200, 0.2)
model2.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model2.parameters(), lr=lr)
# Measure the time it takes to train the model
start = time.time()
train_losses, test_losses = mlp.train_model(model2, X_train, y_train, X_test, y_test, criterion, optimizer, n_epochs,patience=800)
end = time.time()
print(f"Training time: {end - start}")
# Evaluate the model using our function
f1, acc, cm = mlp.getResults(train_losses, test_losses, model2, X_val, y_val)
print(f"F1: {f1}")
print(f"Accuracy: {acc}")
print(f"Confusion Matrix: {cm}")
# Save the model
torch.save(model2.state_dict(), 'model2.pth')
# Save the training and testing losses as well
with open('model2_train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)

with open('model2_test_losses.pkl', 'wb') as f:
    pickle.dump(test_losses, f)

# Model 3
model3 = mlp.MLP_mach3(input_size, 260, 200, 140, 100, 60, 20, dropout=.2)
model3.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model3.parameters(), lr=lr)
# Measure the time it takes to train the model
start = time.time()
train_losses, test_losses = mlp.train_model(model3, X_train, y_train, X_test, y_test, criterion, optimizer, n_epochs,patience=800)
end = time.time()
print(f"Training time: {end - start}")
# Evaluate the model using our function
f1, acc, cm = mlp.getResults(train_losses, test_losses, model3, X_val, y_val)
print(f"F1: {f1}")
print(f"Accuracy: {acc}")
print(f"Confusion Matrix: {cm}")
# Save the model
torch.save(model3.state_dict(), 'model3.pth')
# Save the training and testing losses as well
with open('model3_train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)

with open('model3_test_losses.pkl', 'wb') as f:
    pickle.dump(test_losses, f)


# Model 4
model4 = mlp.MLP_mach4(input_size, 300, 250, 200, 150, 100, 50, dropout=.2)
model4.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model4.parameters(), lr=lr)
# Measure the time it takes to train the model
start = time.time()
train_losses, test_losses = mlp.train_model(model4, X_train, y_train, X_test, y_test, criterion, optimizer, n_epochs,patience=800)
end = time.time()
print(f"Training time: {end - start}")
# Evaluate the model using our function
f1, acc, cm = mlp.getResults(train_losses, test_losses, model4, X_val, y_val)
print(f"F1: {f1}")
print(f"Accuracy: {acc}")
print(f"Confusion Matrix: {cm}")
# Save the model
torch.save(model4.state_dict(), 'model4.pth')
# Save the training and testing losses as well
with open('model4_train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)

with open('model4_test_losses.pkl', 'wb') as f:
    pickle.dump(test_losses, f)

# Model 5
model5 = mlp.MLP_mach5(input_size, 300, 260, 220, 180, 140, 100, 60, dropout=.2)
model5.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model5.parameters(), lr=lr)
# Measure the time it takes to train the model
start = time.time()
train_losses, test_losses = mlp.train_model(model5, X_train, y_train, X_test, y_test, criterion, optimizer, n_epochs,patience=800)
end = time.time()
print(f"Training time: {end - start}")
# Evaluate the model using our function
f1, acc, cm = mlp.getResults(train_losses, test_losses, model5, X_val, y_val)
print(f"F1: {f1}")
print(f"Accuracy: {acc}")
print(f"Confusion Matrix: {cm}")
# Save the model
torch.save(model5.state_dict(), 'model5.pth')
# Save the training and testing losses as well  
with open('model5_train_losses.pkl', 'wb') as f:
    pickle.dump(train_losses, f)

with open('model5_test_losses.pkl', 'wb') as f:
    pickle.dump(test_losses, f)





