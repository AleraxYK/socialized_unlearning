from code.models import create_model
from code.unlearning.unlearning_utils import evaluate_model, unlearning_get_cifar10_dataloaders
import torch
import platform as pl

# Configurazione
if pl.system() == "Darwin":
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def retention_score_computation():
    #### MODELS LOADING ####
    agents_unlearning = [create_model() for _ in range(2)]
    agents_learning = [create_model() for _ in range(2)]
    for idx in range(2):
        checkpoint_path = f'code/checkpoint/UNLEARNED_teacher_{idx}_trained_model.pth'
        checkpoint_learning_path = f'code/preprocessing/checkpoint/unlearning_agent_{idx}_trained_model.pth'
        print(f"Loading weights for agent {idx} from {checkpoint_path}...")
        print(f"Loading weights for agent {idx} from {checkpoint_learning_path}...")

        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
            agents_unlearning[idx].load_state_dict(checkpoint)
            agents_unlearning[idx].to(device)
            print(f"Agent {idx} loaded successfully.")
            checkpoint = torch.load(checkpoint_learning_path, weights_only=True, map_location=device)
            agents_learning[idx].load_state_dict(checkpoint)
            agents_learning[idx].to(device)
            print(f"Agent learning {idx} loaded successfully.")
        except Exception as e:
            print(f"Error loading agent {idx}: {e}")
            return

    student_unlearning_checkpoint_path = "code/checkpoint/UNLEARNED_student_trained_model.pth"
    student_learning_checkpoint_path = "code/preprocessing/checkpoint/unlearning_student_trained_model.pth"
    print(f"Loading student model from {student_unlearning_checkpoint_path}...")
    print(f"Loading student model from {student_learning_checkpoint_path}...")
    student_unlearning = create_model()
    student_learning = create_model()

    try:
        checkpoint = torch.load(student_unlearning_checkpoint_path, weights_only=True, map_location=device)
        student_unlearning.load_state_dict(checkpoint)
        student_unlearning.to(device)
        print(f"Student loaded successfully.")
        checkpoint = torch.load(student_learning_checkpoint_path, weights_only=True, map_location=device)
        student_learning.load_state_dict(checkpoint)
        student_learning.to(device)
        print(f"Student loaded successfully.")
    except Exception as e:
        print(f"Error loading student model: {e}")
        return
    ################
    #### DATASET LOADING ####
    target_classes = [0, 4]
    dataset_loaders = unlearning_get_cifar10_dataloaders(batch_size=512, target_classes=target_classes)
    _, _, target_test_loader, non_target_test_loader, _ = dataset_loaders[0][0], dataset_loaders[0][1], dataset_loaders[1][0], dataset_loaders[1][1], dataset_loaders[2]
    ################

    #### EVALUATION OF THE MODELS ####
    student_pre_accuracy = 0
    student_post_accuracy = 0
    teachers_pre_accuracy = [0 for _ in range(2)]
    teachers_post_accuracy = [0 for _ in range(2)]

    student_pre_accuracy = evaluate_model(student_learning, non_target_test_loader, device=device)
    student_post_accuracy = evaluate_model(student_unlearning, non_target_test_loader, device=device)

    for idx, teacher in enumerate(agents_unlearning):
        teachers_pre_accuracy[idx] = evaluate_model(agents_learning[idx], non_target_test_loader, device=device)
        teachers_post_accuracy[idx] = evaluate_model(teacher, non_target_test_loader, device=device)

    # COMPUTE RETENTION SCORE
    student_retention_score = retention_score(student_post_accuracy, student_pre_accuracy)
    teachers_retention_score = [retention_score(teachers_post_accuracy[idx], teachers_pre_accuracy[idx]) for idx in range(2)]

    print(f"STUDENT RETENTION SCORE: {student_retention_score}")
    for i in range(2):
        print(f"TEACHER {i} RETENTION SCORE: {teachers_retention_score[i]}")

def retention_score(actual_accuracy, pre_accuracy):
    return actual_accuracy / pre_accuracy