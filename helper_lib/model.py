import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

def get_model(model_name):
    """
    Define and return the appropriate model based on model_name.
    
    Args:
        model_name (str): Name of the model - one of: FCNN, CNN, EnhancedCNN, SimpleCNN, CIFAR_CNN
    
    Returns:
        nn.Module: PyTorch model
    """
    if model_name == "FCNN":
        # Fully Connected Neural Network
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    elif model_name == "CNN":
        # Convolutional Neural Network
        model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )
    elif model_name == "EnhancedCNN":
        # Enhanced CNN using ResNet18 architecture
        model = resnet18(weights=None)  # Updated parameter name
        # Modify first conv layer for single channel input (e.g., MNIST)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify final layer for 10 classes
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_name == "SimpleCNN":
        # SimpleCNN from Module 4 Practical 3 - for CIFAR-10
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.fc1 = nn.Linear(32 * 8 * 8, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))  # -> (?, 16, 16, 16)
                x = self.pool(F.relu(self.conv2(x)))  # -> (?, 32, 8, 8)
                x = x.view(-1, 32 * 8 * 8)  # -> (?, 2048)
                x = F.relu(self.fc1(x))  # -> (?, 128)
                x = self.fc2(x)  # -> (?, 10)
                return x
        
        model = SimpleCNN()
    elif model_name == "CIFAR_CNN":
        # Enhanced CNN from Module 4 Practical 3 - for CIFAR-10
        class EnhancedCIFARCNN(nn.Module):
            def __init__(self):
                super(EnhancedCIFARCNN, self).__init__()
                # Convolutional layers with BatchNorm
                self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(16)
                
                self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(32)
                
                self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm2d(64)
                
                self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.bn4 = nn.BatchNorm2d(128)
                
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                
                # Fully connected layers with Dropout
                self.fc1 = nn.Linear(128 * 2 * 2, 128)
                self.dropout = nn.Dropout(0.5)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                # Conv layers with BatchNorm and ReLU
                x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (?, 16, 16, 16)
                x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (?, 32, 8, 8)
                x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (?, 64, 4, 4)
                x = self.pool(F.relu(self.bn4(self.conv4(x))))  # -> (?, 128, 2, 2)
                
                # Flatten
                x = x.view(-1, 128 * 2 * 2)  # -> (?, 512)
                
                # Fully connected layers with dropout
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        model = EnhancedCIFARCNN()
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: FCNN, CNN, EnhancedCNN, SimpleCNN, CIFAR_CNN")
    
    return model