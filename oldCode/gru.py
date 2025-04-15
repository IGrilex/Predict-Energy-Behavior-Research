import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

# GRU Model with increased capacity
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, fc_hidden_dim=64, output_dim=1):
        """
        input_dim   : number of features per timestep.
        hidden_dim  : number of hidden units in GRU.
        num_layers  : number of stacked GRU layers.
        fc_hidden_dim: number of neurons in FC layer after GRU.
        output_dim  : output dimension (typically 1 for regression).
        """
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        # GRU returns (output, h_n), where h_n is of shape (num_layers, batch, hidden_dim)
        gru_out, h_n = self.gru(x)
        # Use the hidden state from the last GRU layer
        last_hidden = h_n[-1]  # [batch, hidden_dim]
        out = self.relu(self.fc1(last_hidden))
        out = self.fc2(out)
        return out

def create_gru_model(input_dim, hidden_dim=128, num_layers=2, fc_hidden_dim=64, output_dim=1):
    """
    Create and return an instance of the GRU model.
    """
    return GRUModel(input_dim, hidden_dim, num_layers, fc_hidden_dim, output_dim)

def train_pytorch_model(model, 
                        train_loader, 
                        val_loader, 
                        epochs: int = 50, 
                        lr: float = 1e-3, 
                        device: torch.device = torch.device("cpu")):
    """
    Trains the PyTorch model using MAE (L1Loss) as the loss function.
    Uses tqdm progress bars to report progress.
    
    Returns:
        train_losses: List of average training loss (MAE) per epoch.
        val_losses: List of average validation loss (MAE) per epoch.
    """
    criterion = nn.L1Loss()  # Using MAE loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        
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
            prog_bar.set_postfix(MAE=loss.item())
            
        avg_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
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
                val_prog.set_postfix(MAE=loss.item())
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch}/{epochs} -> Train MAE: {avg_train_loss:.4f}, Val MAE: {avg_val_loss:.4f}")
        
    return train_losses, val_losses
