import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Define the model architecture (must match training)
class MNISTTwoHidden(nn.Module):
    def __init__(self, input_dim: int, hidden_dim1: int, hidden_dim2: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def test_on_subset(model, device, num_samples=1000):
    # 1. Prepare data (must use same transform as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(root="../data/MNIST", train=False, download=True, transform=transform)
    
    # 2. Create a subset of specified number of images
    indices = torch.arange(num_samples)
    subset = Subset(test_dataset, indices)
    loader = DataLoader(subset, batch_size=100, shuffle=False)

    model.eval() # Set to evaluation mode
    correct = 0
    total = 0
    
    print(f"Testing on {num_samples} samples...")
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get prediction (the index of the maximum logit)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Accuracy on {num_samples} images: {accuracy:.2f}%")
    return accuracy

if __name__ == "__main__":
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters (must match training)
    input_dim = 28 * 28
    hidden_dim1 = 16
    hidden_dim2 = 16
    num_classes = 10
    
    # Create model
    model = MNISTTwoHidden(
        input_dim=input_dim, 
        hidden_dim1=hidden_dim1, 
        hidden_dim2=hidden_dim2, 
        num_classes=num_classes
    ).to(device)
    
    # Try to load trained model
    model_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "model.pth")
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"WARNING: No trained model found at {model_path}")
        print("Please run the training script first and make sure it saves the model.")
        print("You need to add model saving to ../training/siec_2_ukryte.py")
        exit(1)
    
    # Run test
    test_on_subset(model, device, num_samples=1000)