# This file is for training the MLP's on a larger amount of the data using a GPU
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import MLPfunctions as mlp
import pandas as pd
from sklearn.model_selection import train_test_split

sys.stdout = open('bosonTrain.txt', 'w')

# Load the data
dataset = pd.loadcsv('HIGGS_train.csv')
# Take about a third of the data (a couple million rows)
dataset = dataset.sample(frac=0.33)
# Set aside some validation data
val_data = dataset.sample(frac=0.1)
# Remove the validation data from the dataset
dataset = dataset.drop(val_data.index)
# Split the data into features and labels
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
X_val = val_data.iloc[:, 1:].values
y_val = val_data.iloc[:, 0].values
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

model1 = mlp.MLP_mach1(input_size, 30)
model1.to(device)
# Set the tensors to the device
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)

n_epochs = 2000

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model1.parameters(), lr=0.001)
train_losses, test_losses = mlp.train_model(model1, X_train, y_train, X_test, y_test, criterion, optimizer, n_epochs)
# Evaluate the model using our function
f1, acc, cm = mlp.getResults(train_losses, test_losses, model1, X_val, y_val)
print(f"F1: {f1}")
print(f"Accuracy: {acc}")
print(f"Confusion Matrix: {cm}")
# Save the model
torch.save(model1.state_dict(), 'model1.pth')

model2 = mlp.MLP_mach2(input_size, 100, 70, 50, 25, 0.2)
model2.to(device)
n_epochs = 3000
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model2.parameters(), lr=0.001)
train_losses, test_losses = mlp.train_model(model2, X_train, y_train, X_test, y_test, criterion, optimizer, n_epochs)
# Evaluate the model using our function
f1, acc, cm = mlp.getResults(train_losses, test_losses, model2, X_val, y_val)
print(f"F1: {f1}")
print(f"Accuracy: {acc}")
print(f"Confusion Matrix: {cm}")
# Save the model
torch.save(model2.state_dict(), 'model2.pth')

model3 = mlp.MLP_mach3(28, 300, 250, 200, 150, 100, 50, 0.2)
model3.to(device)
n_epochs = 6000
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model3.parameters(), lr=0.001)
train_losses, test_losses = mlp.train_model(model3, X_train, y_train, X_test, y_test, criterion, optimizer, n_epochs)
# Evaluate the model using our function
f1, acc, cm = mlp.getResults(train_losses, test_losses, model3, X_val, y_val)
print(f"F1: {f1}")
print(f"Accuracy: {acc}")
print(f"Confusion Matrix: {cm}")
# Save the model
torch.save(model3.state_dict(), 'model3.pth')

sys.stdout.close()