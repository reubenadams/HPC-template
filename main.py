import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

class SimpleMLP(nn.Module):
    def __init__(self, hidden_width):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

def train():
    # Initialize wandb
    wandb.init()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False
    )
    
    # Initialize model
    model = SimpleMLP(wandb.config.hidden_width).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    
    # Training loop
    for epoch in range(5):  # Train for 5 epochs
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Log training metrics
            if batch_idx % 100 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "epoch": epoch
                })
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = correct / len(test_dataset)
        
        # Log test metrics
        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "epoch": epoch
        })
        
        print(f"Epoch {epoch}: Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

    model_path = f"trained_models/mlp_w{wandb.config.hidden_width}_lr{wandb.config.learning_rate}_b{wandb.config.batch_size}"
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    os.mkdir("trained_models")
    train()