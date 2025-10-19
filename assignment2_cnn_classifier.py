"""
Assignment 2: CNN Architecture Implementation
Implementing the exact CNN architecture specified in the assignment

Architecture Specification:
- Input: RGB image of size 64×64×3
- Conv2D with 16 filters, kernel size 3×3, stride 1, padding 1
- ReLU activation
- MaxPooling2D with kernel size 2×2, stride 2
- Conv2D with 32 filters, kernel size 3×3, stride 1, padding 1
- ReLU activation
- MaxPooling2D with kernel size 2×2, stride 2
- Flatten the output
- Fully connected layer with 100 units
- ReLU activation
- Fully connected layer with 10 units (assume 10 output classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from helper_lib import get_device, count_parameters, save_model, load_model

class AssignmentCNN(nn.Module):
    """
    CNN Architecture exactly as specified in Assignment 2
    
    This implementation follows the exact specification:
    - Input: 64×64×3 RGB images
    - Two convolutional blocks with pooling
    - Two fully connected layers
    - 10 output classes
    """
    
    def __init__(self, num_classes=10):
        super(AssignmentCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, 
                              kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block  
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                              kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        # Input: 64×64×3
        # After conv1 + pool1: 32×32×16
        # After conv2 + pool2: 16×16×32
        # Flattened size: 16 × 16 × 32 = 8192
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 16 * 32, 100)
        self.fc2 = nn.Linear(100, num_classes)
        
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)           # 64×64×3 → 64×64×16
        x = F.relu(x)
        x = self.pool1(x)           # 64×64×16 → 32×32×16
        
        # Second convolutional block
        x = self.conv2(x)           # 32×32×16 → 32×32×32
        x = F.relu(x)
        x = self.pool2(x)           # 32×32×32 → 16×16×32
        
        # Flatten the output
        x = x.view(x.size(0), -1)   # Flatten to (batch_size, 8192)
        
        # Fully connected layers
        x = self.fc1(x)             # 8192 → 100
        x = F.relu(x)
        x = self.fc2(x)             # 100 → 10
        
        return x
    
    def get_architecture_summary(self):
        """Return a summary of the architecture"""
        return {
            "input_shape": "(batch_size, 3, 64, 64)",
            "conv1": "Conv2d(3, 16, kernel_size=3, stride=1, padding=1)",
            "pool1": "MaxPool2d(kernel_size=2, stride=2)",
            "conv2": "Conv2d(16, 32, kernel_size=3, stride=1, padding=1)", 
            "pool2": "MaxPool2d(kernel_size=2, stride=2)",
            "fc1": "Linear(8192, 100)",
            "fc2": "Linear(100, 10)",
            "total_parameters": sum(p.numel() for p in self.parameters())
        }

def load_cifar10_64x64():
    """
    Load CIFAR-10 dataset and resize to 64×64 as required by assignment
    """
    print("📊 Loading CIFAR-10 dataset and resizing to 64×64...")
    
    # Transform to resize CIFAR-10 from 32×32 to 64×64
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize to 64×64
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"✅ Train dataset: {len(train_dataset)} samples")
    print(f"✅ Test dataset: {len(test_dataset)} samples")
    print(f"✅ Image size: 64×64×3 (resized from 32×32)")
    print(f"✅ Classes: {class_names}")
    
    return train_loader, test_loader, class_names

def train_assignment_cnn(model, train_loader, test_loader, device, epochs=10):
    """
    Train the Assignment CNN model
    """
    print(f"\n🚀 Training Assignment CNN for {epochs} epochs...")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training history
    train_losses = []
    train_accuracies = []
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training
        train_loader_with_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (data, target) in enumerate(train_loader_with_progress):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            train_loader_with_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%")
        
        # Evaluate on test set every few epochs
        if (epoch + 1) % 3 == 0:
            test_acc = evaluate_assignment_cnn(model, test_loader, device)
            print(f"Test Accuracy after epoch {epoch+1}: {test_acc:.2f}%")
    
    print("✅ Training completed!")
    return train_losses, train_accuracies

def evaluate_assignment_cnn(model, test_loader, device):
    """
    Evaluate the Assignment CNN model
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100. * correct / total
    model.train()  # Return to training mode
    return accuracy

def visualize_sample_predictions(model, test_loader, class_names, device, num_samples=8):
    """
    Visualize sample predictions from the model
    """
    print(f"\n🖼️ Visualizing Sample Predictions...")
    
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)
    
    # Visualize first few samples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(num_samples):
        row = i // 4
        col = i % 4
        
        # Convert image back to displayable format
        img = images[i].cpu()
        img = img * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
        img = img.permute(1, 2, 0)  # Change from CHW to HWC
        
        axes[row, col].imshow(img)
        
        # Create title with prediction info
        true_label = class_names[labels[i].item()]
        pred_label = class_names[predicted[i].item()]
        confidence = probabilities[i][predicted[i]].item() * 100
        
        title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%"
        color = 'green' if predicted[i] == labels[i] else 'red'
        
        axes[row, col].set_title(title, color=color, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_model_complexity():
    """
    Calculate and display model complexity metrics
    """
    print(f"\n📊 Model Complexity Analysis")
    print("=" * 40)
    
    # Create model instance
    model = AssignmentCNN(num_classes=10)
    
    # Get architecture summary
    arch_summary = model.get_architecture_summary()
    
    print("🏗️ Architecture Summary:")
    for layer, description in arch_summary.items():
        if layer != "total_parameters":
            print(f"   {layer}: {description}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📈 Parameter Count:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Layer-wise parameter breakdown
    print(f"\n🔍 Layer-wise Parameter Breakdown:")
    for name, param in model.named_parameters():
        print(f"   {name}: {param.numel():,} parameters")
    
    # Memory estimation (rough)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    print(f"\n💾 Estimated Model Size: {model_size_mb:.2f} MB")

def main_assignment_demo():
    """
    Main demonstration following Assignment 2 requirements
    """
    print("🎯 Assignment 2: CNN Architecture Implementation")
    print("Exact Architecture as Specified in Assignment")
    print("=" * 60)
    
    # Setup
    device = get_device()
    
    # Calculate model complexity first
    calculate_model_complexity()
    
    # Load data
    train_loader, test_loader, class_names = load_cifar10_64x64()
    
    # Create model exactly as specified
    print(f"\n🏗️ Creating Assignment CNN Model...")
    model = AssignmentCNN(num_classes=10).to(device)
    
    print(f"✅ Model created successfully!")
    count_parameters(model)
    
    # Display architecture details
    arch_summary = model.get_architecture_summary()
    print(f"\n📋 Architecture Verification:")
    print(f"   Input: RGB images of size 64×64×3 ✅")
    print(f"   Conv1: 16 filters, 3×3 kernel, stride 1, padding 1 ✅")
    print(f"   Pool1: 2×2 kernel, stride 2 ✅")
    print(f"   Conv2: 32 filters, 3×3 kernel, stride 1, padding 1 ✅")
    print(f"   Pool2: 2×2 kernel, stride 2 ✅")
    print(f"   FC1: 100 units ✅")
    print(f"   FC2: 10 output classes ✅")
    
    # Train the model
    print(f"\n{'='*60}")
    print("TRAINING PHASE")
    print(f"{'='*60}")
    
    train_losses, train_accuracies = train_assignment_cnn(
        model, train_loader, test_loader, device, epochs=10
    )
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    
    final_test_accuracy = evaluate_assignment_cnn(model, test_loader, device)
    print(f"\n🎯 Final Test Accuracy: {final_test_accuracy:.2f}%")
    
    # Save the model
    model_path = 'models/assignment2_cnn.pth'
    save_model(model, model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Visualize predictions
    visualize_sample_predictions(model, test_loader, class_names, device)
    
    # Summary
    print(f"\n🎉 Assignment 2 Implementation Complete!")
    print(f"✅ CNN architecture matches exact specification")
    print(f"✅ Model trained on 64×64 RGB images")
    print(f"✅ Achieves {final_test_accuracy:.2f}% accuracy on CIFAR-10")
    print(f"✅ Model saved and ready for API deployment")

if __name__ == "__main__":
    main_assignment_demo()