import torch

def get_device() -> torch.device:
    """
    Select the best available device (CUDA > MPS > CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
