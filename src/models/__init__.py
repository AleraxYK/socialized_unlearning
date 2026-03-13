import torch.nn as nn
from torchvision.models import resnet18, resnet34

def create_model(backbone: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """
    Build a ResNet model with a custom classification head.

    Args:
        backbone (str): ResNet variant to use (``"resnet18"`` or ``"resnet34"``).
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: ResNet with the ``fc`` layer replaced.

    Raises:
        ValueError: If ``backbone`` is not supported.
    """
    backbone = backbone.lower()

    if   backbone == "resnet18":    model = resnet18(weights = None)
    elif backbone == "resnet34":    model = resnet34(weights = None)
    else:                           raise ValueError(f"Unknown type of model: {backbone}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


