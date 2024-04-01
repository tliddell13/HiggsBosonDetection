import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# First attempt at a MLP model
# Has 2 hidden layers and uses ReLU activation function and softmax for output
class MLP_mach1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP_mach1, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()  # Use nn.ReLU for activation
        self.softmax = nn.Softmax(dim=1)  # Use nn.Softmax for output

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = self.relu(hidden)  
        hidden = self.fc2(hidden)
        hidden = self.relu(hidden)
        hidden = self.fc3(hidden)
        output = self.softmax(hidden)  # Apply softmax to the final output
        return output

# Second attempt at a MLP model
# Has 4 hidden layers and uses dropout.
# The dropout rate can be set when creating the model as well as the number of neurons in each hidden layer
class MLP_mach2(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, hidden4, dropout):
        super(MLP_mach2, self).__init__()
        self.input_size = input_size 
        self.hidden_size1 = hidden1
        self.hidden_size2 = hidden2
        self.hidden_size3 = hidden3
        self.hidden_size4 = hidden4
        self.output_size = 2  

        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout) 

        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(self.hidden_size2, self.hidden_size3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)

        self.fc4 = nn.Linear(self.hidden_size3, self.hidden_size4)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=dropout)

        self.fc5 = nn.Linear(self.hidden_size4, self.output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)

        out = self.fc4(out)
        out = self.relu4(out)
        out = self.dropout4(out)

        out = self.fc5(out)
        return out
    
# Bigger model with 6 hidden layers. Going to start with the hidden neurons set to 300 and drop by 50 each layer
# ending with 50 neurons in the last hidden layer. Trying a softmax activation function for the output layer
# If it seems to work at all, will train on a gpu with a larger amount of the data
class MLP_mach3(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, dropout):
        super(MLP_mach3, self).__init__()
        self.input_size = input_size 
        self.hidden_size1 = hidden1
        self.hidden_size2 = hidden2
        self.hidden_size3 = hidden3
        self.hidden_size4 = hidden4
        self.hidden_size5 = hidden5
        self.hidden_size6 = hidden6
        self.output_size = 2  

        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout) 

        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(self.hidden_size2, self.hidden_size3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)

        self.fc4 = nn.Linear(self.hidden_size3, self.hidden_size4)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=dropout)

        self.fc5 = nn.Linear(self.hidden_size4, self.hidden_size5)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=dropout)

        self.fc6 = nn.Linear(self.hidden_size5, self.hidden_size6)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=dropout)

        self.fc7 = nn.Linear(self.hidden_size6, self.output_size)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)

        out = self.fc4(out)
        out = self.relu4(out)
        out = self.dropout4(out)

        out = self.fc5(out)
        out = self.relu5(out)
        out = self.dropout5(out)

        out = self.fc6(out)
        out = self.relu6(out)
        out = self.dropout6(out)

        out = self.fc7(out)
        out = self.softmax(out)
        return out
    
# Function to easily train the model. Can set the number of epochs, the criterion and the optimizer
def train_model(model, X_train, y_train, X_test, y_test, criterion, optimizer, n_epochs):
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        model.train()
        # Forward pass
        # Set the gradients to zero
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred.squeeze(), y_train)
        # Backpropagation
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        model.eval()
        y_pred = model(X_test)
        loss = criterion(y_pred.squeeze(), y_test)
        test_losses.append(loss.item())
        print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}')
    return train_losses, test_losses

# This function plots the training and test losses, and returns f1 score, accuracy and confusion matrix
def getResults(train_losses, test_losses, model, X_test, y_test):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.show()
    model.eval()
    # Make predictions
    y_pred = model(X_test)
    # Calculate the results
    y_pred = torch.argmax(y_pred, dim=1)
    y_test = y_test.squeeze()
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return f1, acc, cm
