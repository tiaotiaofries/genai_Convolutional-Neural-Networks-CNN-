import torch
from torchvision import datasets, transforms

def get_data_loader(data_dir, batch_size=32, train=True):
    """
    Create and return a data loader for the specified dataset.
    
    Args:
        data_dir (str): Path to the data directory
        batch_size (int): Batch size for the data loader
        train (bool): Whether to load training or test data
    
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    # Define transforms
    if train:
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    # Load dataset - handles both ImageFolder and MNIST-style datasets
    try:
        # Try ImageFolder first (for custom datasets)
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    except:
        # Fallback to MNIST if ImageFolder fails
        dataset = datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
    
    # Create data loader
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=train,
        num_workers=2 if torch.cuda.is_available() else 0
    )
    
    return loader