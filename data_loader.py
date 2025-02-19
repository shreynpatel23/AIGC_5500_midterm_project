import torchvision.transforms as transforms
from torchvision.datasets import KMNIST
from torch.utils.data import DataLoader, SubsetRandomSampler

def get_data():
    """Loads the KMNIST dataset and applies normalization."""
    
    # Initial transformation: Convert images to tensors
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Load KMNIST dataset (train and test sets)
    trainset = KMNIST(root='./k_mnist_data', train=True, download=True, transform=transform)
    testset = KMNIST(root='./k_mnist_data', train=False, download=True, transform=transform)

    # Compute mean and standard deviation for normalization
    trainloader = DataLoader(trainset, batch_size=len(trainset), shuffle=True)
    data = next(iter(trainloader))  # Get all training data in one batch
    mean, stddev = data[0].mean(), data[0].std()

    # Update transform to include normalization
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, stddev)])
    
    # Apply updated transform to datasets
    trainset.transform = transform
    testset.transform = transform

    return trainset, testset

def get_data_loaders(trainset, train_idx, val_idx, batch_size=32):
    """Returns train and validation data loaders using indices from KFold."""
    
    # Samplers to select training and validation subsets
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    # Data loaders for training and validation
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(trainset, batch_size=batch_size, sampler=val_sampler)
    
    return train_loader, val_loader