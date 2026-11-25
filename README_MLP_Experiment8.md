# Experiment 5 : Multilayer Perceptron (MLP) Neural Network

## Aim

To implement a Multilayer Perceptron (MLP) neural network for
handwritten digit classification using Python and evaluate its
performance on the test dataset.

> **Note:** The source page contains an inconsistent aim statement
> mentioning K-Means clustering. The experiment title and algorithm
> steps describe an MLP neural network, so this README follows the MLP 
> experiment.  

## Overview

A **Multilayer Perceptron (MLP)** is a feed-forward artificial neural    
network consisting of an input layer, one or more hidden layers, and  an  
output layer. It learns patterns in data by adjusting weights and biases  
during training using backpropagation.  

In this experiment, the MLP model is trained on the **Digits dataset**
available in scikit-learn. Each sample is an 8 × 8 image of a
handwritten digit (0--9). The pixel values are used as input features,
and the model predicts the corresponding digit.

## Algorithm

1.  **Start.**
2.  Load the Digits dataset and separate the input features and target
    labels.
3.  Split the dataset into training and testing sets.
4.  Standardize the input features.
5.  Create an MLP classifier with the specified hidden-layer structure.
6.  Train the model using the training data.
7.  Predict the labels for the test data.
8.  Evaluate the model using test accuracy and a classification report.
9.  Plot the training loss curve.
10. **Stop.**
ry slightly with changes to the train-test split,
    random seed, or model parameters.


## Program
# Step 1: Import required libraries
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.datasets import load_digits    
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import accuracy_score, classification_report  

print("Libraries imported successfully.")  
# Step 2: Load the Digits dataset and separate features and target labels  
digits = load_digits()  
X = digits.data  
y = digits.target  

print("Dataset loaded successfully.")  
print("Feature matrix shape:", X.shape)  
print("Target vector shape:", y.shape)  
print("Digit classes:", np.unique(y))  

# Display five sample handwritten digit images  
fig, axes = plt.subplots(1, 5, figsize=(10, 3))  
for ax, image, label in zip(axes, digits.images[:5], y[:5]):  
    ax.imshow(image, cmap="gray_r")  
    ax.set_title(f"Label: {label}")  
    ax.axis("off")  
plt.suptitle("Sample Images from the Digits Dataset")  
plt.tight_layout()  
plt.show()  

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Step 4: Standardize the input features  
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)
print("Feature standardization completed.")  
# Step 5: Create an MLP classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True
)
print(mlp)  
# Step 6: Train the model   
mlp.fit(X_train_scaled, y_train)  
print("MLP model training completed.")  
print("Number of training iterations:", mlp.n_iter_)  

# Step 7: Predict labels for the test dataset
y_pred = mlp.predict(X_test_scaled)    
print("Predicted labels:", y_pred[:20])  
print("Actual labels:   ", y_test[:20])  

# Step 7: Predict labels for the test dataset
y_pred = mlp.predict(X_test_scaled)
print("Predicted labels:", y_pred[:20])
print("Actual labels:   ", y_test[:20])

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 9: Plot the training loss curve
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_, label="Training Loss")
plt.title("MLP Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss") 
plt.grid(True)  
plt.legend()
plt.tight_layout()  
plt.show ()  
  

## output



## Result

