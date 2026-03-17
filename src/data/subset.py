from typing import Iterable, List
from torch.utils.data import Subset

def filter_by_classes(dataset, classes: Iterable[int]) -> Subset:
    """
    Return a Subset containing only samples whose label is in `classes`.

    Args:
        dataset: A PyTorch Dataset object.
        classes: An iterable of class labels to include in the subset.
    Returns:
        A Subset of the original dataset containing only samples with labels in `classes`.
    """
    class_set = set(int(c) for c in classes)
    idx = [i for i, (_, y) in enumerate(dataset) if int(y) in class_set]
    return Subset(dataset, idx)

def filter_out_classes(dataset, excluded: Iterable[int]) -> Subset:
    """
    Return a Subset containing only samples whose label is NOT in `excluded`.

    Args:
        dataset: A PyTorch Dataset object.
        excluded: An iterable of class labels to exclude from the subset.
    Returns:
        A Subset of the original dataset containing only samples with labels NOT in `excluded`.
    """
    excl = set(int(c) for c in excluded)
    idx = [i for i, (_, y) in enumerate(dataset) if int(y) not in excl]
    return Subset(dataset, idx)