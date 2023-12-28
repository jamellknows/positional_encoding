
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
from torchsummary import summary


# Set random seed for reproducibility

input_size = 4  # Input features
output_size = 4  # Output features
num_heads = 4
hidden_size = 64
num_layers = 2
dropout = 0.1
lr = 0.001
batch_size = 4
epochs = 50

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
        self.fc = nn.Linear(3*input_size, 4)

    def forward(self, src,tgt):
        # Ensure that the dimensions of src and tgt are the same
        # if src.size(-1) != input_size or tgt.size(-1) != input_size:
        #     raise RuntimeError("the feature number of src and tgt must be equal to d_model")
        # src = src.permute(1, 0, 2)
        fc_layer = nn.Linear(input_size, output_size)

        # Forward pass through the transformer
        output = self.transformer(src, src)

        # Take the last layer of the output and apply the linear layer
        # output = output[-1, :, :]
        output = fc_layer(src,tgt)
        # output = self.fc(output.t())

        return output

def output_mod(X):
    output = torch.tensor(2*X+1, requires_grad=True)

    return output

def henonTransformer(X,y,v, v2):
    
        # Hyperparameters
   
    torch.manual_seed(42)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    v = torch.tensor(v, dtype=torch.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # y_train = y_train.t()
    
# Create DataLoader for training and testing sets
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    X_pred_tensor = torch.tensor(v, dtype=torch.float32)
    pred_data = TensorDataset(X_pred_tensor)
    pred_loader = DataLoader(pred_data, batch_size=batch_size, shuffle=False)
    
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
            output = output_mod(batch_X)

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
            output = output_mod(batch_X)

            # Compute loss
            test_loss += criterion(output, batch_y).item()

        # Print average test loss
        print(f"Average Test Loss: {test_loss / len(test_loader)}")

    # Validation
    print(model.eval())
    
    predictions = []

    with torch.no_grad():
        for batch_X_pred in pred_loader:
            # Forward pass
            output = output_mod(batch_X_pred[0])  # Use the transformer model

            # Append predictions to the list
            predictions.append(output.numpy())

    # Convert predictions to a numpy array
    predictions = np.concatenate(predictions, axis=0)
    # print(f" The model summary is {summary(model, input_size)}")
    print(f"The predicitions are {predictions}")
    # print(model.predict(v))
    # with torch.no_grad():
    #     # v = v.unsqueeze(0)  # Add batch dimension
    #     validation_output = model(v, v2)

    # # Print the validation output
    # print("Validation Output:")
    # print(validation_output.squeeze().numpy())

