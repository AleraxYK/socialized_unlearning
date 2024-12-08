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

def socialized_learning(student_model, teacher_models, optimizer_student, optimizer_teachers, criterion_ce, num_epochs):
    best_student_loss = 0
    best_teachers_losses = [0 for _ in range(len(teacher_models))]
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

def socialized_unlearning(student_model, teacher_models, optimizer_student, optimizer_teachers, criterion_ce, num_epochs):
    best_student_loss = 0
    best_teachers_losses = [0 for _ in range(len(teacher_models))]

    target_classes = [0, 1]
    dataset_loaders = unlearning_get_cifar10_dataloaders(batch_size=64, target_classes=target_classes)
    target_train_loader, non_target_train_loader, non_target_test_loader, non_target_val_loader = dataset_loaders[0][0], dataset_loaders[0][1], dataset_loaders[1], dataset_loaders[2]

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}: Unlearning Step")
        collaborative_unlearning(epoch, num_epochs, best_student_loss, student_model, teacher_models, target_train_loader, non_target_train_loader, non_target_val_loader, optimizer_student, criterion_ce, device=device)
        for teacher_idx, (teacher_model, optimizer_teacher) in enumerate(zip(teacher_models, optimizer_teachers)):
            best_teachers_losses[teacher_idx] = reciprocal_unlearning(epoch, num_epochs, best_teachers_losses[teacher_idx], teacher_idx, teacher_model, student_model, target_train_loader, non_target_train_loader, non_target_val_loader, optimizer_teacher, criterion_ce, device=device)

    print("Evaluating student model after unlearning...")
    unlearning_evaluate_model(student_model, non_target_test_loader)
    print("Evaluating teachers models after unlearning...")
    

if __name__=="__main__": 
    # choice = int(input("PRESS:\n0: Learning\n1: Unlearning"))
    # match(choice):
    #     case 0: # LEARNING
    #         # Models
    #         student_learning_model = CNN(num_classes=10, batch_size=64).to(device)
    #         teacher_learning_models = [CNN(num_classes=10, batch_size=64).to(device) for _ in range(2)]

    #         # Optimizer and loss
    #         optimizer_student = optim.Adam(student_learning_model.parameters(), lr=0.001)
    #         optimizer_teachers = [optim.Adam(model.parameters(), lr=0.001) for model in teacher_learning_models]
    #         criterion_ce = torch.nn.CrossEntropyLoss()

    #         num_epochs = 10

    #         socialized_learning(student_learning_model, teacher_learning_models, optimizer_student, optimizer_teachers, criterion_ce, num_epochs)

    #     case 1: # UNLEARNING
    #         # Load student
    #         student_unlearning_model = CNN(num_classes=10, batch_size=64).to(device)
    #         student_unlearning_model.load_state_dict(torch.load("code/checkpoint/student_trained_model.pth", map_location="mps", weights_only=False))
    #         # Load teachers
    #         teacher_unlearning_models = [CNN(num_classes=10, batch_size=64).to(device) for _ in range(2)]
    #         for idx in range(len(teacher_unlearning_models)):
    #             teacher_unlearning_models[idx].load_state_dict(torch.load("code/checkpoint/teacher_"+str(idx)+"_trained_model.pth", map_location="mps", weights_only=False))
            
    #         # Optimizer and loss
    #         optimizer_student = optim.Adam(student_unlearning_model.parameters(), lr=0.001)
    #         optimizer_teachers = [optim.Adam(model.parameters(), lr=0.001) for model in teacher_unlearning_models]
    #         criterion_ce = torch.nn.CrossEntropyLoss()

    #         num_epochs = 10
            
    #         socialized_unlearning(student_unlearning_model, teacher_unlearning_models, optimizer_student, optimizer_teachers, criterion_ce, num_epochs)
    target_classes = [0, 1]
    dataset_loaders = unlearning_get_cifar10_dataloaders(batch_size=64, target_classes=target_classes)

