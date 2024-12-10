import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size: int=64) -> tuple:
    """
    # CIFAR-10 DataLoader Function

    The **get_cifar10_dataloaders** function prepares DataLoaders for training, validation, 
    and testing using the CIFAR-10 dataset. The dataset is normalized and split into 
    training, validation, and testing subsets.

    Args:
        - batch_size (int): Batch size for the DataLoaders (default is 64).

    Returns:
        - tuple: A tuple containing three DataLoaders:
            - train_loader: DataLoader for the training dataset.
            - test_loader: DataLoader for the testing dataset.
            - val_loader: DataLoader for the validation dataset.
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

    

    #split test into test and validation
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [2000, 8000])

    # Dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

def feature_extractor(model, data):
    features = nn.Sequential(*list(model.children())[:-3])
    return features(data)

def classifier_extractor(model, data):
    classifier = nn.Sequential(*list(model.children())[-3:])
    return classifier(data)


def evaluate_model(model, dataloader, device = "mps") -> float:
    """
    # Model Evaluation Function

    The **evaluate_model** function calculates the accuracy of a given model on a specified dataset.

    Args:
        - model: The PyTorch model to be evaluated.
        - dataloader: A DataLoader containing the dataset for evaluation.

    Returns:
        - float: The accuracy of the model on the dataset, expressed as a percentage.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy
