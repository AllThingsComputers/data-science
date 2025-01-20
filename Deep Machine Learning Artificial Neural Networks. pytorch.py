import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load the dataset, split into input (X) and output (y) variables
dataset = np.loadtxt('SecurityDataANN.csv', delimiter=',')  # Ensure the correct delimiter is used
X = dataset[:, 0:8]
y = dataset[:, 8]

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# Define the model
model = nn.Sequential(
    nn.Linear(8, 12),  # Input layer to hidden layer 1
    nn.ReLU(),         # Activation function for hidden layer 1
    nn.Linear(12, 8),  # Hidden layer 1 to hidden layer 2
    nn.ReLU(),         # Activation function for hidden layer 2
    nn.Linear(8, 1),   # Hidden layer 2 to output layer
    nn.Sigmoid()       # Activation function for output layer
)
print(model)

# Define the loss function and optimizer
loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use "lr" instead of "Ir"

# Training settings
n_epochs = 100
batch_size = 10

# Train the model
for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        # Forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Log progress
    print(f"Finished epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

# Compute accuracy
with torch.no_grad():
    y_pred = model(X)
    accuracy = (y_pred.round() == y).float().mean().item()  # Calculate accuracy
    print(f"Accuracy: {accuracy:.4f}")
