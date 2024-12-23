import numpy as np
import torch
import torch.nn as nn
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
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
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
    target_train_data = filter_dataset(train_dataset, target_classes, non_keep=True)
    non_target_train_data = filter_dataset(train_dataset, target_classes, non_keep=False)
    
    # TEST
    target_test_data = filter_dataset(test_dataset, target_classes, non_keep=True)
    non_target_test_data = filter_dataset(test_dataset, target_classes, non_keep=False)

    # VALIDATION
    non_target_val_data, non_target_test_data = torch.utils.data.random_split(non_target_test_data, [1500, 6500])

    target_train_loader = DataLoader(target_train_data, batch_size=batch_size, shuffle=True)
    non_target_train_loader = DataLoader(non_target_train_data, batch_size=batch_size, shuffle=True)

    target_test_loader = DataLoader(target_test_data, batch_size=batch_size, shuffle=True)
    non_target_test_loader = DataLoader(non_target_test_data, batch_size=batch_size, shuffle=True)

    non_target_val_loader = DataLoader(non_target_val_data, batch_size=batch_size, shuffle=True)

    return ((target_train_loader, non_target_train_loader), (target_test_loader, non_target_test_loader), non_target_val_loader)


# Filter dataset to exclude classes
def filter_dataset(dataset: torch.utils.data.Dataset, target_classes: list, non_keep: bool=True) -> torch.utils.data.Subset:
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

    if non_keep:
        indices = [i for i, label in enumerate(labels) if label in target_classes]
    else:
        indices = [i for i, label in enumerate(labels) if label not in target_classes]

    filtered_dataset = torch.utils.data.Subset(dataset, indices)
    return filtered_dataset

def feature_extractor(model, data):
    features = nn.Sequential(*list(model.children())[:-3])
    return features(data)

def classifier_extractor(model, data):
    classifier = nn.Sequential(*list(model.children())[-3:])
    return classifier(data)

def show_params(model):
    for idx, (name, param) in enumerate(model.named_parameters()):
        print(f"Param: {param}")


# accuracy 
def evaluate_model(model, loader, device):
    '''
    Function to calculate the accuracy of the model on the test set
    '''
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient calculation
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            
            # Get model predictions
            scores = model(data)
            # percentage = torch.softmax(scores, dim=1)
            predictions = torch.argmax(scores, dim=1)  # Get the index of the max logit

            # print(f"PREDICTIONS: {predictions[0]}")
            # print(f"TARGETS: {targets[0]}")
            
            correct += (predictions == targets).sum().item()  # Convert tensor to scalar
            total += targets.size(0)  # Total number of samples
    
    acc = correct / total  # Calculate accuracy
    print(f"CORRECT: {correct}, TOTAL: {total}")
    print(f"Accuracy: {100 * acc:.2f}%")  # Print with 2 decimal places
