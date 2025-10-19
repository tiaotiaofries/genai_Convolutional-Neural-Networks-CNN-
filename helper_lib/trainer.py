import torch
from torch.utils.data import DataLoader

def train_model(model, data_loader, criterion, optimizer, device='cpu', epochs=10):
    """
    Run several iterations of the training loop based on epochs parameter.
    
    Args:
        model: PyTorch model to train
        data_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for training
        device: Device to run training on ('cpu' or 'cuda')
        epochs: Number of training epochs
    
    Returns:
        model: Trained PyTorch model
    """
    # Move model to specified device
    model.to(device)
    model.train()  # Set model to training mode
    
    print(f"Starting training on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(data_loader)
        epoch_accuracy = 100.0 * correct_predictions / total_samples
        
        print(f'Epoch {epoch+1}/{epochs} completed. Average Loss: {epoch_loss:.6f}, Accuracy: {epoch_accuracy:.2f}%')
    
    print("Training completed!")
    return model