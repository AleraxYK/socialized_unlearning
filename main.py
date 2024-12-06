import torch
import platform as pl
from torch import optim
from code.models import CNN
# Learning
from code.learning.train import collaborative_collaboration, reciprocal_altruism
from code.learning.utils import get_cifar10_dataloaders, evaluate_model
# Unlearning
from code.unlearning.unlearning_train import collaborative_unlearning, reciprocal_unlearning
from code.unlearning.unlearning_utils import unlearning_evaluate_model, unlearning_get_cifar10_dataloaders

# Configurazione
if pl.system() == "Darwin":
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelli
student_model = CNN(num_classes=10, batch_size=64).to(device)
teacher_models = [CNN(num_classes=10, batch_size=64).to(device) for _ in range(2)]

# Ottimizzatori e perdita
optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)
optimizer_teachers = [optim.Adam(model.parameters(), lr=0.001) for model in teacher_models]
criterion_ce = torch.nn.CrossEntropyLoss()

num_epochs = 10
best_student_loss = 0
best_teachers_losses = [0 for _ in range(len(teacher_models))]

def socialized_learning():
    # Dataset
    train_loader, test_loader, val_loader = get_cifar10_dataloaders(batch_size=64)
    # Training loop
    for epoch in range(num_epochs):
        
        # Collaborative Collaboration
        best_student_loss = collaborative_collaboration(epoch, num_epochs, best_student_loss, student_model, teacher_models, train_loader, val_loader, optimizer_student, criterion_ce, device=device)
        
        # Reciprocal Altruism
        for teacher_idx, (teacher_model, optimizer_teacher) in enumerate(zip(teacher_models, optimizer_teachers)):
            best_teachers_losses[teacher_idx] = reciprocal_altruism(epoch, num_epochs, best_teachers_losses[teacher_idx], teacher_idx, teacher_model, student_model, train_loader, val_loader, optimizer_teacher, criterion_ce, device=device)

        print("Training epoch completed.")

    # Evaluation
    print("Evaluating student model...")
    evaluate_model(student_model, test_loader)

def socialized_unlearning():
    target_classes = [0, 1]
    target_loader, non_target_loader = get_cifar10_dataloaders(batch_size=64, target_classes=target_classes)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}: Unlearning Step")
        collaborative_unlearning(epoch, num_epochs, student_model, teacher_models, target_loader, non_target_loader, optimizer_student, criterion_ce, device=device)
        for teacher_model, optimizer_teacher in zip(teacher_models, optimizer_teachers):
            reciprocal_unlearning(epoch, num_epochs, teacher_model, student_model, target_loader, non_target_loader, optimizer_teacher, criterion_ce, device=device)

        print("Evaluating student model after unlearning...")
        evaluate_model(student_model, non_target_loader)

if __name__=="__main__":
    choice = int(input("PRESS:\n0: Learning\n1: Unlearning"))
    match(choice):
        case 0:
            socialized_learning()
        case 1:
            socialized_unlearning()

