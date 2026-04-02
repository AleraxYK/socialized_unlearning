import os
import shutil
import urllib.request
import zipfile

from torchvision import datasets, transforms


TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _download_and_prepare(root: str) -> str:
    """
    Downloads TinyImageNet and restructures the validation directory
    so that torchvision.datasets.ImageFolder can read it correctly.

    Args:
        root (str): Root directory where data will be stored.

    Returns:
        str: Path to the prepared tiny-imagenet-200 directory.
    """
    base = os.path.join(root, "tiny-imagenet-200")

    if os.path.exists(base):
        return base

    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")

    print("Downloading TinyImageNet (~235MB)...")
    urllib.request.urlretrieve(TINYIMAGENET_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as f:
        f.extractall(root)
    os.remove(zip_path)

    # Restructure val/ so ImageFolder can read it
    val_dir   = os.path.join(base, "val")
    img_dir   = os.path.join(val_dir, "images")
    annot     = os.path.join(val_dir, "val_annotations.txt")

    with open(annot, "r") as f:
        for line in f:
            fname, cls = line.strip().split("\t")[:2]
            cls_folder = os.path.join(val_dir, cls)
            os.makedirs(cls_folder, exist_ok=True)
            src = os.path.join(img_dir, fname)
            dst = os.path.join(cls_folder, fname)
            if os.path.exists(src):
                os.rename(src, dst)

    shutil.rmtree(img_dir)
    os.remove(annot)
    print("TinyImageNet ready!")

    return base


def get_tinyimagenet(root: str = "./data", train: bool = True):
    """
    Loads the TinyImageNet dataset (200 classes, 64x64 pixels).
    Downloads and restructures it automatically on first call.

    Args:
        root (str): Root directory where data will be stored.
        train (bool): If True, loads the training split; otherwise the validation split.

    Returns:
        torchvision.datasets.ImageFolder: The requested dataset split.
    """
    base = _download_and_prepare(root)

    mean = [0.4802, 0.4481, 0.3975]
    std  = [0.2302, 0.2265, 0.2262]

    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    split = "train" if train else "val"
    return datasets.ImageFolder(
        root=os.path.join(base, split),
        transform=transform,
    )
