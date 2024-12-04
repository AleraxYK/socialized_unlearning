import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Load CIFAR-10 Dataset
def unlearning_get_cifar10_dataloaders(batch_size=64, target_classes=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Dataset CIFAR-10
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Filtrare classi target e non target
    if target_classes is not None:
        target_data = filter_dataset(train_dataset, target_classes, keep=True)
        non_target_data = filter_dataset(train_dataset, target_classes, keep=False)
        target_loader = DataLoader(target_data, batch_size=batch_size, shuffle=True)
        non_target_loader = DataLoader(non_target_data, batch_size=batch_size, shuffle=True)
        return target_loader, non_target_loader
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

# Filter dataset to exclude classes
def filter_dataset(dataset, target_classes, keep=True):
    indices = [
        i for i, (_, label) in enumerate(dataset)
        if (label in target_classes) == keep
    ]
    return Subset(dataset, indices)

# Evaluate the model
def unlearning_evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy
