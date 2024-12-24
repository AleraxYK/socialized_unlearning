from code.models import create_model
from code.unlearning.unlearning_utils import evaluate_model, unlearning_get_cifar10_dataloaders
import torch
import platform as pl

# Configurazione
if pl.system() == "Darwin":
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluation():
    #### MODELS LOADING ####
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
    
    already_forget_model = create_model()
    already_forget_model_path = f"code/preprocessing/checkpoint/forget_model.pth"
    try:
        checkpoint = torch.load(already_forget_model_path, weights_only=True, map_location=device)
        already_forget_model.load_state_dict(checkpoint)
        already_forget_model.to(device)
        print(f"Already forget model loaded successfully.")
    except Exception as e:
        print(f"Error loading already forget model: {e}")
        return
    ################
    #### DATASET LOADING ####
    target_classes = [0, 4]
    dataset_loaders = unlearning_get_cifar10_dataloaders(batch_size=512, target_classes=target_classes)
    _, _, target_test_loader, non_target_test_loader, _ = dataset_loaders[0][0], dataset_loaders[0][1], dataset_loaders[1][0], dataset_loaders[1][1], dataset_loaders[2]
    ################

    #### EVALUATION OF THE MODELS ####
    student_accuracy = 0
    teachers_accuracy = [0 for _ in range(2)]

    student_accuracy = evaluate_model(student_unlearning, non_target_test_loader, device=device)
    for idx, teacher in enumerate(agents_unlearning):
        teachers_accuracy[idx] = evaluate_model(teacher, non_target_test_loader, device=device)

def retention_score(actual_accuracy, pre_accuracy):
    return actual_accuracy / pre_accuracy