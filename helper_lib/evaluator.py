import torch

def evaluate_model(model, data_loader, criterion, device='cpu'):
    """
    Calculate average loss and accuracy on the test dataset.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run evaluation on ('cpu' or 'cuda')
    
    Returns:
        tuple: (avg_loss, accuracy)
    """
    # Move model to specified device
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    print(f"Starting evaluation on {device}...")
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            # Optional: Print progress for large datasets
            if batch_idx % 50 == 0 and batch_idx > 0:
                current_accuracy = 100.0 * correct_predictions / total_samples
                print(f'Evaluated {batch_idx} batches, Current Accuracy: {current_accuracy:.2f}%')
    
    # Calculate final metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct_predictions / total_samples
    
    print(f"Evaluation completed!")
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")
    
    return avg_loss, accuracy