import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Simulated neural data and labels as per your description
neural_data = torch.randn(10178, 120)  # Simulated neural data (timesteps, features)
labels = torch.randn(10178, 3)         # Simulated continuous labels

# DataLoader setup
dataset = TensorDataset(neural_data, labels)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

class ConvNet(nn.Module):
    def __init__(self, num_features, num_hidden_units, output_dim):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=num_hidden_units, kernel_size=5, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_hidden_units, out_channels=num_hidden_units, kernel_size=5, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_hidden_units, out_channels=output_dim, kernel_size=5, stride=1, padding=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = x.unsqueeze(1).permute(0, 2, 1)  # Change shape to [batch_size, channels, length]
        x = self.conv_layers(x)
        x = x.squeeze(2)  # Remove the last dimension after pooling
        return x

    
def info_nce_loss(features, labels, temperature=1.0):
    # Normalize features
    features = F.normalize(features, dim=1)
    
    # Calculate cosine similarity matrix
    similarity_matrix = torch.matmul(features, features.T)
    
    # Adjust labels to be properly compared
    labels = labels.squeeze(1)  # Change shape from [batch_size, 1, num_features] to [batch_size, num_features]
    
    # Create mask for identifying positive and negative samples
    positive_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).all(2).float()
    
    # Calculate logits
    exp_sim = torch.exp(similarity_matrix / temperature)
    
    # Sum of exp similarities for positive and all pairs
    pos_sum = torch.sum(exp_sim * positive_mask, dim=1)
    all_sum = torch.sum(exp_sim, dim=1)  # The denominator includes positive pairs which is fine for stability
    
    # Calculate the actual loss
    loss = -torch.log(pos_sum / all_sum + 1e-6)
    
    return torch.mean(loss)


def info_nce_loss(features, labels, temperature=1.0, threshold=1):
    """
    Compute the InfoNCE loss using Euclidean distance threshold for positive pairs.
    Args:
        features (torch.Tensor): The embeddings or features output by the model.
        labels (torch.Tensor): Continuous labels associated with each feature.
        temperature (float): The temperature scaling factor for the softmax.
        threshold (float): The threshold for determining positive samples based on label distance.
    Returns:
        torch.Tensor: The mean InfoNCE loss.
    """
    # Normalize features to unit length
    features = F.normalize(features, dim=1)
    
    # Calculate the cosine similarity matrix
    similarity_matrix = torch.matmul(features, features.T)
    
    # Calculate Euclidean distances between labels
    labels_diff = torch.matmul(labels, labels.T)
    #print(labels_diff)
    
    # Create positive mask based on the Euclidean distance threshold
    positive_mask = (labels_diff <= threshold).float()
    
    # Calculate exponentiated similarities scaled by temperature
    exp_sim = torch.exp(similarity_matrix / temperature)
    
    # Compute sums of exponentiated similarities where masks apply
    pos_sum = torch.sum(exp_sim * positive_mask, dim=1)
    all_sum = torch.sum(exp_sim, dim=1)
    
    # Calculate the InfoNCE loss
    loss = -torch.log(pos_sum / all_sum + 1e-6)
    return torch.mean(loss)

# Example usage in a training loop
# Assume `features` and `labels` are outputs from your model and target labels respectively
# loss = info_nce_loss(features, labels, temperature=1.0, threshold=0.1)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, optimizer
model = ConvNet(num_features=120, num_hidden_units=16, output_dim=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

def train_epoch(loader, model, optimizer):
    model.train()
    total_loss = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = info_nce_loss(outputs, targets, temperature=1.0)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    loss = train_epoch(loader, model, optimizer)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Save the trained model if needed
torch.save(model.state_dict(), 'model.pth')
