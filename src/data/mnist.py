from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def _to_rgb(x):
    return x.repeat(3,1,1)

def get_mnist(data_root: str ="./data", train: bool = True) -> datasets.MNIST:
    """
    Get the MNIST dataset.

    Args:
        data_root (str): Root directory where MNIST is stored/downloaded.
        train (bool): Whether to load the training split.

    Returns:
        datasets.MNIST: MNIST dataset with the appropriate transforms.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(_to_rgb),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return datasets.MNIST(
        root = data_root,
        train = train,
        download = True,
        transform = transform
    )