from torch.utils.data import Dataset
from src.data.cifar10 import get_cifar10
from src.data.mnist import get_mnist
from src.data.tinyimagenet import get_tinyimagenet

def get_dataset(name: str, root: str, train: bool) -> Dataset:
    """
    Get a dataset by name.

    Args:
        name: Name of the dataset.
        root: Root directory to store the dataset.
        train: Whether to return the training set.

    Returns:
        Dataset: The dataset.
    """
    name = name.strip().lower()
    if name == "cifar10": return get_cifar10(root, train)
    elif name == "mnist": return get_mnist(root, train)
    elif name == "tinyimagenet": return get_tinyimagenet(root, train)
    else: raise ValueError(f"Unknown dataset: {name}")
    

def get_num_classes(name):
    """
    Get the number of classes for a dataset.

    Args:
        name: Name of the dataset.

    Returns:
        int: The number of classes.
    """
    name = name.strip().lower()
    if name == "cifar10": return 10
    elif name == "mnist": return 10
    elif name == "tinyimagenet": return 200
    else: raise ValueError(f"Unknown dataset: {name}")