import torch
import platform as pl
from torch import optim
from code.models import create_model
# Learning
from code.learning.train import collaborative_collaboration, reciprocal_altruism
from code.learning.utils import get_cifar10_dataloaders, evaluate_model
# Unlearning
from code.unlearning.unlearning_train import collaborative_unlearning, unlearning_reciprocal_altruism
from code.unlearning.unlearning_utils import evaluate_model, unlearning_get_cifar10_dataloaders
from code.unlearning.evaluations import retention_score_computation

# Configurazione
if pl.system() == "Darwin":
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def socialized_learning():
    teacher_models = [create_model() for _ in range(5)]
    for idx in range(5):
        # agents[idx].load_state_dict(torch.load('code/preprocessing/checkpoint/agent_'+str(idx)+'_trained_model.pth', map_location="mps", weights_only=True))
        teacher_models[idx].load_state_dict(torch.load('code/preprocessing/checkpoint/agent_'+str(idx)+'_trained_model.pth',map_location='mps', weights_only=True))
        teacher_models[idx].to(device)

    student_model = create_model()
    student_model.load_state_dict(torch.load("code/preprocessing/checkpoint/model_weights.pth",map_location='mps', weights_only=True))
    student_model.to(device)

    # Optimizer and loss
    optimizer_student = optim.Adam(student_model.parameters(), lr=0.005)
    optimizer_teachers = [optim.Adam(teacher_models[idx].parameters(), lr=0.005) for idx in range(5)]
    criterion_ce = torch.nn.CrossEntropyLoss()

    student_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_student, mode='min', factor=0.1, patience=5, verbose=True)
    teachers_scheduler = [optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) for optimizer in optimizer_teachers]

    num_epochs = 50 # SET TO 50
    
    best_student_loss = 0
    best_teachers_losses = [0 for _ in range(len(teacher_models))]
    # Dataset
    train_loader, test_loader, val_loader = get_cifar10_dataloaders(batch_size=512) # ON THE PAPER batch_size = 128
    # Training loop
    for epoch in range(num_epochs):
        
        # Collaborative Collaboration
        student_model, optimizer_student, student_scheduler, best_student_loss = collaborative_collaboration(epoch, num_epochs, best_student_loss, student_model, teacher_models, train_loader, val_loader, optimizer_student, criterion_ce, scheduler=student_scheduler, device=device)
        
        # Reciprocal Altruism
        for teacher_idx, (teacher_model, optimizer_teacher) in enumerate(zip(teacher_models, optimizer_teachers)):
            teacher_models[teacher_idx], optimizer_teachers[teacher_idx], teachers_scheduler[teacher_idx], best_teachers_losses[teacher_idx] = reciprocal_altruism(epoch, num_epochs, best_teachers_losses[teacher_idx], teacher_idx, teacher_model, student_model, train_loader, val_loader, optimizer_teacher, criterion_ce, scheduler=teachers_scheduler[teacher_idx], device=device)

        print("Training epoch completed.")

    # Evaluation
    print("Evaluating student model...")
    evaluate_model(student_model, test_loader, device)

def socialized_unlearning():
    #### ENVIRONMENT CONFIGURATION ####
    # Models:
    agents_unlearning = [create_model() for _ in range(2)]  # create 2 agents
    for idx in range(2):
        # load pre-trained weights for the agents
        agents_unlearning[idx].load_state_dict(torch.load(
            'code/preprocessing/checkpoint/unlearning_agent_' + str(idx) + '_trained_model.pth',map_location='cpu', weights_only=True))
        agents_unlearning[idx].to(device)

    # Creation of the student model
    student_unlearning = create_model()
    # Load pretrained weights
    student_unlearning.load_state_dict(torch.load(
        "code/preprocessing/checkpoint/unlearning_student_trained_model.pth", map_location='cpu', weights_only=True))
    student_unlearning.to(device)

    student_model = student_unlearning

    
    criterion_ce = torch.nn.CrossEntropyLoss()  # Loss function

    target_classes = [0, 4]
    dataset_loaders = unlearning_get_cifar10_dataloaders(batch_size=512, target_classes=target_classes)
    target_train_loader, non_target_train_loader, _, _, non_target_val_loader = dataset_loaders[0][0], dataset_loaders[0][1], dataset_loaders[1][0], dataset_loaders[1][1], dataset_loaders[2]

    # Optimization
    optimizer_student = optim.Adam(filter(lambda p: p.requires_grad, student_unlearning.parameters()), lr=0.005, weight_decay=1e-4) 
    optimizer_teachers = [optim.Adam(filter(lambda p: p.requires_grad, agents_unlearning[idx].parameters()), lr=0.005, weight_decay=1e-4) for idx in range(2)]

    # Scheduler
    student_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_student, mode='min', factor=0.1, patience=5, verbose=True)
    teachers_scheduler = [optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True) for optimizer in optimizer_teachers]
    
    num_epochs = 50

    #### END ENVIRONMENT CONFIGURATION ####

    best_student_loss_unlearning = 0
    best_student_loss_learning = 0
    best_teachers_losses_unlearning = [0 for _ in range(len(agents_unlearning))]
    best_teachers_losses_learning = [0 for _ in range(len(agents_unlearning))]

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}: Unlearning Step")
        student_model, optimizer_student, student_scheduler, best_student_loss_unlearning = collaborative_unlearning(epoch, num_epochs, best_student_loss_unlearning, student_model, agents_unlearning, target_train_loader, non_target_val_loader, optimizer_student, criterion_ce, student_scheduler, device=device)
        student_model, optimizer_student, student_scheduler, best_student_loss_learning = collaborative_collaboration(epoch, num_epochs, best_student_loss_learning, student_model, agents_unlearning, non_target_train_loader, non_target_val_loader, optimizer_student, criterion_ce, student_scheduler, device=device)
        for teacher_idx, (teacher_model, optimizer_teacher) in enumerate(zip(agents_unlearning, optimizer_teachers)):
            agents_unlearning[teacher_idx], optimizer_teachers[teacher_idx], teachers_scheduler[teacher_idx], best_teachers_losses_unlearning[teacher_idx] = unlearning_reciprocal_altruism(epoch, num_epochs, best_teachers_losses_unlearning[teacher_idx], teacher_idx, teacher_model, student_model, target_train_loader, non_target_val_loader, optimizer_teacher, criterion_ce, teachers_scheduler[teacher_idx], device=device)
            agents_unlearning[teacher_idx], optimizer_teachers[teacher_idx], teachers_scheduler[teacher_idx], best_teachers_losses_learning[teacher_idx] = reciprocal_altruism(epoch, num_epochs, best_teachers_losses_learning[teacher_idx], teacher_idx, teacher_model, student_model, non_target_train_loader, non_target_val_loader, optimizer_teacher, criterion_ce, teachers_scheduler[teacher_idx], device=device)

if __name__ == "__main__":
    choice = int(input("PRESS:\n0: Learning\n1: Unlearning\n2: Compute retention Score"))
    match(choice):
        case 0:
            # LEARNING
            socialized_learning()
        case 1:
            # UNLEARNING
            socialized_unlearning()
        case 2:
            retention_score_computation()