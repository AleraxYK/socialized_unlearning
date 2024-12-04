import torch
from torch import optim
from code.models import CNN
from code.losses import energy_alignment_loss, knowledge_distillation_loss
from code.train import collaborative_collaboration, reciprocal_altruism
from code.utils import get_cifar10_dataloaders, evaluate_model

# Configurazione
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
train_loader, test_loader, val_loader = get_cifar10_dataloaders(batch_size=64)

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

# Training loop
for epoch in range(num_epochs):  # Numero di epoche
    
    # Collaborative Collaboration
    best_student_loss = collaborative_collaboration(epoch, num_epochs, best_student_loss, student_model, teacher_models, train_loader, val_loader, optimizer_student, criterion_ce)
    
    # Reciprocal Altruism
    for teacher_idx, (teacher_model, optimizer_teacher) in enumerate(zip(teacher_models, optimizer_teachers)):
        best_teachers_losses[teacher_idx] = reciprocal_altruism(epoch, num_epochs, best_teachers_losses[teacher_idx], teacher_idx, teacher_model, student_model, train_loader, val_loader, optimizer_teacher, criterion_ce)

    print("Training epoch completed.")

# Valutazione sul test set
print("Evaluating student model...")
evaluate_model(student_model, test_loader)
