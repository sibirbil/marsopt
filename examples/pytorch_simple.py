"""
PyTorch Hyperparameter Optimization using MarsOpt

This script demonstrates how to optimize PyTorch neural network hyperparameters using MarsOpt.

- Loads the Diabetes dataset.
- Defines a PyTorch neural network model for regression.
- Defines an objective function that trains the PyTorch model.
- Uses MarsOpt to optimize hyperparameters for better performance.
- Prints the best trial with optimal parameters and MSE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from marsopt import Study, Trial
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes


# Load dataset
data = load_diabetes()
X, y = data.data, data.target

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(NeuralNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
    
    predictions = np.vstack(predictions).flatten()
    actuals = np.vstack(actuals).flatten()
    
    return mean_squared_error(actuals, predictions)

# Objective function for hyperparameter tuning
def objective(trial: Trial):
    # Define hyperparameters to optimize
    params = {
        "batch_size": trial.suggest_int("batch_size", 8, 128),
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 128),
        "num_layers": trial.suggest_int("num_layers", 1, 5),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.0, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "epochs": trial.suggest_int("epochs", 10, 100)
    }
    
    # Create data loaders with the suggested batch size
    train_loader = DataLoader(
        train_dataset, 
        batch_size=params["batch_size"], 
        shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=params["batch_size"], 
        shuffle=False
    )
    
    # Initialize model with suggested hyperparameters
    input_dim = X_train.shape[1]

    torch.manual_seed(42)
    np.random.seed(42)

    model = NeuralNetwork(
        input_dim=input_dim,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        dropout_rate=params["dropout_rate"]
    )
    
    # Set up optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(), 
        lr=params["learning_rate"], 
        weight_decay=params["weight_decay"]
    )
    criterion = nn.MSELoss()
    
    # Train model
    train_model(model, train_loader, criterion, optimizer, params["epochs"])
    
    # Evaluate model
    mse = evaluate_model(model, test_loader)
    return mse

# Create MarsOpt study
study = Study(random_state=42)
study.optimize(objective, n_trials=100)

# Print best trial
best_trial = study.best_trial
print("Best trial:")
print(f"MSE: {best_trial['objective_value']}")
print("Best hyperparameters:")
for key, value in best_trial["variables"].items():
    print(f"  {key}: {value}")