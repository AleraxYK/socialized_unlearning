import torch
import torch.nn as nn

def feature_extractor(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """
    Extracts features from the model up to the final fully connected layer.
    Assumes standard ResNet-like architecture where the final layer is named 'fc'.
    """
    features = inputs
    for name, module in model.named_children():
        if name == 'fc':
            features = torch.flatten(features, 1)
            break
        features = module(features)
    return features


def classifier_extractor(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """
    Passes extracted features through the final classifier layer of the model.
    Assumes the final layer is named 'fc'.
    """
    return model.fc(features)
