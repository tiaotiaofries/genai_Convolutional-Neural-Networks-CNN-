import torch
import os

def save_model(model, path):
    """
    Save a PyTorch model to the specified path.
    
    Args:
        model: PyTorch model to save
        path (str): File path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """
    Load a PyTorch model from the specified path.
    
    Args:
        model: PyTorch model (architecture should match saved model)
        path (str): File path to load the model from
    
    Returns:
        model: Loaded PyTorch model
    """
    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Load model state dict
    model.load_state_dict(torch.load(path, map_location='cpu'))
    print(f"Model loaded from {path}")
    return model

def get_device():
    """
    Get the best available device (CUDA if available, otherwise CPU).
    
    Returns:
        torch.device: Device to use for training/inference
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} trainable parameters")
    return total_params