import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_trasforms(train: bool = True) -> transforms.Compose:
    """
    Get the transforms for the dataset.

    Args:
        train (bool): Whether the dataset is for training.

    Returns:
        transforms.Compose: The transforms for the dataset.
    """

    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def get_cifar10(data_root: str = "./data", train: bool = True) -> datasets.CIFAR10:
    """
    Get the CIFAR-10 dataset.

    Args:
        data_root (str): Root directory where CIFAR-10 is stored/downloaded.
        train (bool): Whether to load the training split.

    Returns:
        datasets.CIFAR10: CIFAR-10 dataset with the appropriate transforms.
    """

    return datasets.CIFAR10(
        root = data_root,
        train = train,
        download = True,
        transform = get_trasforms(train) 
    )