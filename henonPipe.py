import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
input_size = 10  # Input features (sequence length)
output_size = 1  # Output features
num_heads = 1
hidden_size = 64
num_layers = 2
dropout = 0.1
lr = 0.001
batch_size = 8
epochs = 20

# Transformer model
class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout
        )
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, src):
        # Ensure that the dimensions of src have shape (sequence_length, batch_size, input_size)
        src = src.permute(1, 0, 3)

        # Forward pass through the transformer
        output = self.transformer(src, src)

        # Take the last layer of the output and apply the linear layer
        output = output[-1, :, :]
        output = self.fc(output)

        return output

def henonPipe(X,y,v):
    # Generate synthetic data

    # X = np.random.rand(100, input_size)
    # y = np.sum(X, axis=1, keepdims=True)  # Sum of input features as the target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoader for training set
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = TransformerModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            output = model(batch_X)

            # Compute loss
            loss = criterion(output, batch_y)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        # Print average training loss for the epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        test_loss = 0

        for batch_X, batch_y in test_loader:
            # Forward pass
            output = model(batch_X)

            # Compute loss
            test_loss += criterion(output, batch_y).item()

        # Print average test loss
        print(f"Average Test Loss: {test_loss / len(test_loader)}")
