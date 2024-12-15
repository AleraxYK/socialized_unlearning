import torch
import platform as pl
from code.models import create_model
# Learning
from code.learning.utils import evaluate_model
# Unlearning
from code.unlearning.unlearning_utils import evaluate_model, unlearning_get_cifar10_dataloaders

# Configurazione
if pl.system() == "Darwin":
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

student_model = create_model()
student_model.load_state_dict(torch.load("code/checkpoint/UNLEARNED_student_trained_model.pth", weights_only=True))
student_model.to(device)

target_classes = [0, 4]
dataset_loaders = unlearning_get_cifar10_dataloaders(batch_size=512, target_classes=target_classes)
target_train_loader, non_target_train_loader, non_target_test_loader, non_target_val_loader = dataset_loaders[0][0], dataset_loaders[0][1], dataset_loaders[1], dataset_loaders[2]

print("Evaluating student model after unlearning...")
evaluate_model(student_model, non_target_test_loader, device)
print("Evaluating teachers models after unlearning...")

