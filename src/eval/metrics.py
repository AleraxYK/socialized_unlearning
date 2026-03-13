import torch
import torch.nn.functional as F


@torch.no_grad
def accuracy(model, loader, device: torch.device) -> float:
    """
    Compute classification accuracy over a dataloader.

    Args:
        model: PyTorch model.
        loader: DataLoader providing (x, y).
        device (torch.device): Device to run on.

    Returns:
        float: Accuracy in [0, 1].
    """
    model.eval()
    correct = 0
    total   = 0
    
    for x, y in loader: 
        x = x.to(device)
        y = y.to(device)

        pred = model(x).argmax(dim=1)
        correct+= ( pred == y).sum().item()
        total += y.numel()

    return correct / total if total > 0 else float("nan")


@torch.no_grad
def mean_confidence(model, loader, device: torch.device) -> float:
    """
    Compute mean max-softmax confidence over a dataloader.

    Args:
        model: PyTorch model.
        loader: DataLoader providing (x, y) or (x, _).
        device (torch.device): Device to run on.

    Returns:
        float: Mean confidence in [0, 1].
    """
    s = 0.0
    n = 0

    for x, _ in loader:
        x = x.to(device)
        probs = F.softmax(model(x), dim=1)
        conf  = probs.max(dim=1).values
        s += conf.sum()
        n += conf.numel()

    return s / n if n > 0 else float("nan")


@torch.no_grad
def evaluate(model, loaders: dict, device: torch.device) -> dict:
    """
    loaders can contain keys like:
      - "test_all"
      - "retain"
      - "forget"
      - "forget_subset_train"
    Returns a dict with acc + confidence for each available split.
    """
    out = {}
    for name, loader in loaders.items():
        out[f"{name}_acc"] = accuracy(model, loader, device)
        out[f"{name}_conf"] = mean_confidence(model, loader, device)

    return out