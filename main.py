import torch
import platform as pl
from torch import optim
from code.models import create_model
# Learning
from code.learning.train import collaborative_collaboration, reciprocal_altruism
from code.learning.utils import get_cifar10_dataloaders, evaluate_model
# Unlearning
from code.unlearning.unlearning_train import collaborative_unlearning, unlearning_reciprocal_altruism, find_freezable_params
from code.unlearning.unlearning_utils import evaluate_model, unlearning_get_cifar10_dataloaders, show_params

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
            'code/preprocessing/checkpoint/unlearning_agent_' + str(idx) + '_trained_model.pth',map_location='mps', weights_only=True))
        agents_unlearning[idx].to(device)

    # Creation of the student model
    student_unlearning = create_model()
    # Load pretrained weights
    student_unlearning.load_state_dict(torch.load(
        "code/preprocessing/checkpoint/unlearning_student_trained_model.pth", map_location='mps', weights_only=True))
    student_unlearning.to(device)

    
    criterion_ce = torch.nn.CrossEntropyLoss()  # Loss function

    target_classes = [0, 4]
    dataset_loaders = unlearning_get_cifar10_dataloaders(batch_size=512, target_classes=target_classes)
    target_train_loader, non_target_train_loader, target_test_loader, non_target_test_loader, non_target_val_loader = dataset_loaders[0][0], dataset_loaders[0][1], dataset_loaders[1][0], dataset_loaders[1][1], dataset_loaders[2]
    # FREEZE params that are influenced more by the retain set
    student_model = find_freezable_params(student_unlearning, non_target_train_loader, criterion_ce, device)
    # _, params_target = find_freezable_params(student_unlearning, target_train_loader, criterion_ce)
    teacher_models = [find_freezable_params(agents_unlearning[i], non_target_train_loader, criterion_ce, device) for i in range(2)]
    # print(f"paramteri target len: {len(params_target)}")
    # print(f"paramteri non target len: {len(params_nontarget)}")
    # print(f"Parametri in comune: {[valore for valore in params_target if valore in params_nontarget]}")

    # Optimization
    optimizer_student = optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=0.005) 
    optimizer_teachers = [optim.Adam(filter(lambda p: p.requires_grad, teacher_models[idx].parameters()), lr=0.005) for idx in range(2)]

    # Scheduler
    student_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_student, mode='min', factor=0.1, patience=5, verbose=True)
    teachers_scheduler = [optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True) for optimizer in optimizer_teachers]
    
    num_epochs = 50

    #### END ENVIRONMENT CONFIGURATION ####

    best_student_loss = 0
    best_teachers_losses = [0 for _ in range(len(teacher_models))]

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}: Unlearning Step")
        student_model, optimizer_student, student_scheduler, best_student_loss = collaborative_unlearning(epoch, num_epochs, best_student_loss, student_model, teacher_models, target_train_loader, non_target_train_loader, non_target_val_loader, optimizer_student, criterion_ce, student_scheduler, device=device)
        for teacher_idx, (teacher_model, optimizer_teacher) in enumerate(zip(teacher_models, optimizer_teachers)):
            teacher_models[teacher_idx], optimizer_teachers[teacher_idx], teachers_scheduler[teacher_idx], best_teachers_losses[teacher_idx] = unlearning_reciprocal_altruism(epoch, num_epochs, best_teachers_losses[teacher_idx], teacher_idx, teacher_model, student_model, target_train_loader, non_target_train_loader, non_target_val_loader, optimizer_teacher, criterion_ce, teachers_scheduler[teacher_idx], device=device)

    
    # # TODO: fixare la evaluation con le metrics che avevamo deciso di usare
    # student_model = create_model()
    # student_model.load_state_dict(torch.load("code/checkpoint/UNLEARNED_student_trained_model.pth", weights_only=True))
    # student_model.to(device)
    # print("Evaluating student model after unlearning...")
    # evaluate_model(student_model, non_target_test_loader, device)
    # print("Evaluating teachers models after unlearning...")

def evaluation():
    # Models:
    agents_unlearning = [create_model() for _ in range(2)]  # create 2 agents
    for idx in range(2):
        # load pre-trained weights for the agents
        agents_unlearning[idx].load_state_dict(torch.load(
            'code/checkpoint/UNLEARNED_teacher_' + str(idx) + '_trained_model.pth',map_location='mps', weights_only=True))
        agents_unlearning[idx].to(device)

    # Creation of the student model
    student_unlearning = create_model()
    # Load pretrained weights
    # student_unlearning.load_state_dict(torch.load(
    #     "code/checkpoint/UNLEARNED_student_trained_model.pth", weights_only=True))
    student_unlearning.load_state_dict(torch.load(
        "code/checkpoint/UNLEARNED_student_trained_model.pth", map_location="mps", weights_only=True))
    student_unlearning.to(device)

    # show_params(student_unlearning)
    

def evaluation2():
    agents_unlearning = [create_model() for _ in range(2)]
    for idx in range(2):
        checkpoint_path = f'code/checkpoint/UNLEARNED_teacher_{idx}_trained_model.pth'
        print(f"Loading weights for agent {idx} from {checkpoint_path}...")

        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
            agents_unlearning[idx].load_state_dict(checkpoint)
            agents_unlearning[idx].to(device)
            print(f"Agent {idx} loaded successfully.")
        except Exception as e:
            print(f"Error loading agent {idx}: {e}")
            return

    student_checkpoint_path = "code/checkpoint/UNLEARNED_student_trained_model.pth"
    print(f"Loading student model from {student_checkpoint_path}...")
    student_unlearning = create_model()

    try:
        checkpoint = torch.load(student_checkpoint_path, weights_only=True, map_location=device)
        student_unlearning.load_state_dict(checkpoint)
        student_unlearning.to(device)
        print(f"Student loaded successfully.")
    except Exception as e:
        print(f"Error loading student model: {e}")
        return
    # Datasets:
    target_classes = [0, 4]
    dataset_loaders = unlearning_get_cifar10_dataloaders(batch_size=512, target_classes=target_classes)
    target_train_loader, non_target_train_loader, target_test_loader, non_target_test_loader, non_target_val_loader = dataset_loaders[0][0], dataset_loaders[0][1], dataset_loaders[1][0], dataset_loaders[1][1], dataset_loaders[2]

    # Evaluation:
    print(f"STUDENT Evaluate on forget set: ")
    evaluate_model(student_unlearning, target_test_loader, device=device)
    print(f"STUDENT Evaluate on retain set: ")
    evaluate_model(student_unlearning, target_test_loader, device=device)
    for idx, teacher in enumerate(agents_unlearning):
        print(f"TEACHER {idx} Evaluate on forget set: ")
        evaluate_model(teacher, target_test_loader, device=device)
        print(f"TEACHER {idx} Evaluate on retain set: ")
        evaluate_model(teacher, target_test_loader, device=device)
     



if __name__ == "__main__":
    #choice = int(input("PRESS:\n0: Learning\n1: Unlearning\n"))
    #match(choice):
        #case 0:
            # LEARNING
            #socialized_learning()
        #case 1:
            # UNLEARNING
    #socialized_unlearning()
    evaluation2()