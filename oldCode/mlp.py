import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

def create_pytorch_model(input_dim, hidden_dims=[64, 32, 16, 8], output_dim=1):
    """
    Return an instance of the PyTorch MLP model.
    """
    return MLPModel(input_dim, hidden_dims, output_dim)

def train_pytorch_model(model, 
                        train_loader, 
                        val_loader, 
                        epochs: int = 50, 
                        lr: float = 1e-3, 
                        device: torch.device = torch.device("cpu")):
    """
    Trains the PyTorch model. Also records training and validation loss per epoch.
    Progress is displayed via tqdm progress bars.
    Returns training and validation loss history.
    """
    criterion = nn.L1Loss()  # Using L1 loss (MAE) similar to before
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Lists to store loss for each epoch
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
        # Training progress bar
        prog_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Training]", leave=False)
        for batch_X, batch_y in prog_bar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
            prog_bar.set_postfix(loss=loss.item())
            
        avg_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_prog = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Validation]", leave=False)
            for batch_X, batch_y in val_prog:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item() * batch_X.size(0)
                val_prog.set_postfix(loss=loss.item())
                
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch}/{epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
    return train_losses, val_losses
