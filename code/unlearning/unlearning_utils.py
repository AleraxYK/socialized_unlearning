import numpy as np
import torch
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# Load CIFAR-10 Dataset
def unlearning_get_cifar10_dataloaders(batch_size: int=64, target_classes: list=None) -> tuple[DataLoader, DataLoader]:
    """
    # CIFAR-10 DataLoader for Unlearning Tasks

    The **unlearning_get_cifar10_dataloaders** function prepares data loaders for the CIFAR-10 dataset, 
    with an option to filter the dataset based on target and non-target classes for unlearning tasks.

    Args:
        - batch_size (int): The number of samples per batch (default is 64).
        - target_classes (list, optional): A list of class indices to be considered as target classes 
          for unlearning. If None, no filtering is done, and the full CIFAR-10 dataset is used.

    Returns:
        - target_train_loader (DataLoader): A DataLoader for the target classes.
        - non_target_train_loader (DataLoader): A DataLoader for the non-target classes.
        - target_test_loader (DataLoader): A DataLoader for the target classes.
        - non_target_test_loader (DataLoader): A DataLoader for the non-target classes.
        - non_target_val_loader (DataLoader): A DataLoader for the non-target classes.

    """
    # Dataset CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor)
    images = np.stack([train_dataset[i][0] for i in range(len(train_dataset))])

    # Compute the mean and standard deviation for each channel
    mean = images.mean(axis=(0, 2, 3))
    std = images.std(axis=(0, 2, 3))

    # Define the augmentations for the training set
    cifar_transforms = transforms.Compose([
        transforms.ToTensor(),                    # Convert the image to a PyTorch tensor
        transforms.Normalize(mean, std),          # Normalize the image channel
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cifar_transforms)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cifar_transforms)

    # Filter target and non target classes
    # TRAIN
    target_train_data = filter_dataset(train_dataset, target_classes, keep=True)
    non_target_train_data = filter_dataset(train_dataset, target_classes, keep=False)
    
    # TEST
    non_target_test_data = filter_dataset(test_dataset, target_classes, keep=False)

    # VALIDATION
    non_target_val_data, non_target_test_data = torch.utils.data.random_split(non_target_test_data, [1500, 6500])

    target_train_loader = DataLoader(target_train_data, batch_size=batch_size, shuffle=True)
    non_target_train_loader = DataLoader(non_target_train_data, batch_size=batch_size, shuffle=True)

    non_target_test_loader = DataLoader(non_target_test_data, batch_size=batch_size, shuffle=True)

    non_target_val_loader = DataLoader(non_target_val_data, batch_size=batch_size, shuffle=True)

    return ((target_train_loader, non_target_train_loader), non_target_test_loader, non_target_val_loader)


# Filter dataset to exclude classes
def filter_dataset(dataset: torch.utils.data.Dataset, target_classes: list, keep: bool=True) -> torch.utils.data.Subset:
    """
    # Dataset Filtering for Target and Non-Target Classes

    The **filter_dataset** function filters a dataset based on the specified target classes and a flag 
    indicating whether to keep the target classes or exclude them.

    Args:
        - dataset (torch.utils.data.Dataset): The dataset to be filtered.
        - target_classes (list): A list of class labels to be considered as target or non-target classes.
        - keep (bool): If True, the dataset will keep only the target classes; if False, it will exclude them.

    Returns:
        - torch.utils.data.Subset: A subset of the original dataset containing only the target or non-target classes.
    """
    labels = np.array([label for _, label in dataset])

    if keep:
        indices = [i for i, label in enumerate(labels) if label in target_classes]
    else:
        indices = [i for i, label in enumerate(labels) if label not in target_classes]

    filtered_dataset = torch.utils.data.Subset(dataset, indices)
    return filtered_dataset

# Evaluate the model
def unlearning_evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
    """
    # Model Evaluation for Unlearning Tasks

    The **unlearning_evaluate_model** function evaluates a model on a given dataset and computes its 
    accuracy. It runs the model in evaluation mode, calculates predictions, and compares them with the true labels.

    Args:
        - model (torch.nn.Module): The model to be evaluated.
        - dataloader (torch.utils.data.DataLoader): The DataLoader containing the dataset on which the model is evaluated.

    Returns:
        - accuracy (float): The accuracy of the model on the dataset, as a percentage.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy
