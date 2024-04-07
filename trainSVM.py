# Train the SVM on a GPU, so we can get a fair comparison with the neural network.
import time
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load the data
dataset = pd.read_csv('HIGGS_train.csv')
# Take about a third of the data (a couple million rows)
dataset = dataset.sample(frac=0.09)
# This time I am going to use cyclic feature encoding on the angular features and scaling all the features
# I discuss this more in the data manip ipynb
angular_feats = ['lepton phi', 'missing energy phi', 'jet 1 phi', 'jet 2 phi', 'jet 3 phi', 'jet 4 phi']
X = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]

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

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train the SVM using the best parameters found in the SVM .ipynb
svm_model = svm.SVC(kernel='linear', C=26)
# Time the training
start = time.time()
svm_model.fit(X_train, y_train)
end = time.time()
print("Training time: ", end - start)
# Make predictions
y_pred = svm_model.predict(X_test)
# Calculate the accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
# Calculate the F1 score
f1 = f1_score(y, y_pred)
print("F1 Score: ", f1)
# Create a confusion matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix: ")
print(cm)

# Save the model
dump(svm_model, 'svm_model.joblib')




