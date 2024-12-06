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
        - target_loader (DataLoader): A DataLoader for the target classes if `target_classes` is provided.
        - non_target_loader (DataLoader): A DataLoader for the non-target classes if `target_classes` is provided.
        - train_loader (DataLoader): A DataLoader for the training set if no filtering is applied.
        - test_loader (DataLoader): A DataLoader for the test set if no filtering is applied.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Filtrare classi target e non target
    if target_classes is not None:
        target_data = filter_dataset(train_dataset, target_classes, keep=True)
        non_target_data = filter_dataset(train_dataset, target_classes, keep=False)
        target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=True)
        non_target_loader = DataLoader(non_target_data, batch_size=batch_size, shuffle=True)
        return target_loader, non_target_loader
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader


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
    indices = [
        i for i, (_, label) in enumerate(dataset)
        if (label in target_classes) == keep
    ]
    return Subset(dataset, indices)

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
