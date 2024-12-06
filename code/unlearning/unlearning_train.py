import torch
from tqdm import tqdm
import os
from .unlearning_losses import unlearning_knowledge_distillation_loss, unlearning_energy_alignment_loss

def collaborative_unlearning(epoch: int, num_epochs: int, student_model, teacher_models, target_loader, non_target_loader, optimizer, criterion_ce, lambda_1: float=1.0, lambda_2: float=0.1, delta_target: float=-5, delta_non_target: float=-20, device: str="cpu") -> tuple:
    """
    # Collaborative Unlearning

    The **collaborative_unlearning** function enables a student model to forget information about 
    specific target classes while retaining information about non-target classes. This is achieved 
    by leveraging multiple teacher models for knowledge distillation and energy alignment.

    Args:
        - epoch (int): Current training epoch.
        - num_epochs (int): Total number of training epochs.
        - student_model: The student model being trained to forget target classes.
        - teacher_models (list): List of teacher models providing knowledge for unlearning.
        - target_loader: DataLoader containing samples from target classes.
        - non_target_loader: DataLoader containing samples from non-target classes.
        - optimizer: Optimizer used for training the student model.
        - criterion_ce: Cross-entropy loss criterion for classification tasks.
        - lambda_1 (float): Weight for the reverse knowledge distillation loss (default is 1.0).
        - lambda_2 (float): Weight for the energy alignment loss (default is 0.1).
        - delta_target (float): Desired energy level for target classes (default is -5).
        - delta_non_target (float): Desired energy level for non-target classes (default is -20).
        - device (str): Device to perform computations (default is "cpu").

    Returns:
        - tuple: Average losses for target and non-target samples over the epoch.
    """
    student_model.train()
    running_target_loss = 0
    running_non_target_loss = 0

    loop_target = tqdm(target_loader, total=len(target_loader), leave=True)

    # Forget target classes
    for data, labels in loop_target:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # Student output
        student_output = student_model(data)

        # Teacher outputs
        teacher_outputs = []
        for teacher_model in teacher_models:
            teacher_model.eval()
            with torch.no_grad():
                teacher_outputs.append(teacher_model(data))

        # Reverse knowledge distillation loss
        loss_forget = sum(-unlearning_knowledge_distillation_loss(student_output, teacher_output)
                          for teacher_output in teacher_outputs)

        # Energy alignment to reduce confidence
        loss_energy_target = unlearning_energy_alignment_loss(student_output, delta_target)

        # Total loss
        loss = lambda_1 * loss_forget + lambda_2 * loss_energy_target
        loss.backward()
        optimizer.step()

        running_target_loss += loss.item()

        # Update progress bar
        loop_target.set_description(f"TARGET Epoch [{epoch+1}]")
        loop_target.set_postfix(loss=loss.item())
    
    avg_target_loss = running_target_loss / len(target_loader)
    tqdm.write(f"\033[34mEpoch [{epoch+1}/{num_epochs}]\033[0m, Average TARGET Loss: {avg_target_loss:.4f}")
    
    loop_non_target = tqdm(non_target_loader, total=len(non_target_loader), leave=True)

    # Preserve non-target classes
    for data, labels in loop_non_target:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # Student output
        student_output = student_model(data)

        # Energy alignment to keep confidence
        loss_energy_non_target = unlearning_energy_alignment_loss(student_output, delta_non_target)

        # Total loss
        loss = criterion_ce(student_output, labels) + lambda_2 * loss_energy_non_target
        loss.backward()
        optimizer.step()

        running_non_target_loss += loss.item()

        # Update progress bar
        loop_non_target.set_description(f"NON TARGET Epoch [{epoch+1}]")
        loop_non_target.set_postfix(loss=loss.item())
    
    avg_non_target_loss = running_non_target_loss / len(non_target_loader)
    tqdm.write(f"\033[34mEpoch [{epoch+1}/{num_epochs}]\033[0m, Average NON TARGET Loss: {avg_non_target_loss:.4f}")

    #### VALIDATION ####
    student_model.eval()
    validation_loss = 0
    total_samples, correct_predictions = 0, 0

    with torch.no_grad():
        for data, labels in target_loader:
            data, labels = data.to(device), labels.to(device)

            # Student output
            student_output = student_model(data)

            # Compute validation loss for target classes
            loss_val_target = unlearning_energy_alignment_loss(student_output, delta_target)
            validation_loss += loss_val_target.item()

            # Accuracy calculation
            _, predicted = torch.max(student_output, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_validation_loss = validation_loss / len(target_loader)
    accuracy = 100 * correct_predictions / total_samples

    print(f"Validation Loss: {avg_validation_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_target_loss, avg_non_target_loss



def reciprocal_unlearning(epoch: int, num_epochs: int, teacher_model, student_model, target_loader, non_target_loader, optimizer, criterion_ce, lambda_1: float=1.0, lambda_2: float=0.1, delta_target: float=-5, delta_non_target: float=-20, device: str="cpu") -> tuple:
    """
    # Reciprocal Unlearning

    The **reciprocal_unlearning** function implements a training loop to achieve unlearning 
    of specific target classes while preserving non-target classes using a teacher-student 
    model setup. It minimizes the knowledge about target classes in the teacher model while 
    ensuring the retention of information about non-target classes.

    Args:
        - epoch (int): Current training epoch.
        - num_epochs (int): Total number of training epochs.
        - teacher_model: The teacher model being trained to forget target classes.
        - student_model: The student model providing prior knowledge for target unlearning.
        - target_loader: DataLoader containing samples from target classes.
        - non_target_loader: DataLoader containing samples from non-target classes.
        - optimizer: Optimizer used for training the teacher model.
        - criterion_ce: Cross-entropy loss criterion for classification tasks.
        - lambda_1 (float): Weight for the inverse knowledge distillation loss (default is 1.0).
        - lambda_2 (float): Weight for the energy alignment loss (default is 0.1).
        - delta_target (float): Desired energy level for target classes (default is -5).
        - delta_non_target (float): Desired energy level for non-target classes (default is -20).
        - device (str): Device to perform computations (default is "cpu").

    Returns:
        - tuple: Average losses for target and non-target samples over the epoch.
    """
    teacher_model.train()
    running_target_loss = 0
    running_non_target_loss = 0

    loop_target = tqdm(target_loader, total=len(target_loader), leave=True)

    # Forget target classes
    for data, labels in loop_target:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # Teacher output
        teacher_output = teacher_model(data)

        # Student output (eval mode to avoid gradients)
        with torch.no_grad():
            student_output = student_model(data)

        # Forgetting loss
        loss_forget_kd = -unlearning_energy_alignment_loss(teacher_output, student_output)
        loss_energy_target = unlearning_energy_alignment_loss(teacher_output, delta_target)
        loss = lambda_1 * loss_forget_kd + lambda_2 * loss_energy_target

        loss.backward()
        optimizer.step()

        running_target_loss += loss.item()

        # Update progress bar
        loop_target.set_description(f"TARGET Epoch [{epoch+1}]")
        loop_target.set_postfix(loss=loss.item())
    
    avg_target_loss = running_target_loss / len(target_loader)
    tqdm.write(f"\033[34mEpoch [{epoch+1}/{num_epochs}]\033[0m, Average TARGET Loss: {avg_target_loss:.4f}")
    
    loop_non_target = tqdm(non_target_loader, total=len(non_target_loader), leave=True)

    # Preserve non-target classes
    for data, labels in loop_non_target:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # Teacher output
        teacher_output = teacher_model(data)

        # Retaining loss
        loss_retain_ce = criterion_ce(teacher_output, labels)
        loss_energy_non_target = unlearning_energy_alignment_loss(teacher_output, delta_non_target)
        loss_retain = loss_retain_ce + lambda_2 * loss_energy_non_target

        loss_retain.backward()
        optimizer.step()

        running_non_target_loss += loss_retain.item()

        # Update progress bar
        loop_non_target.set_description(f"NON TARGET Epoch [{epoch+1}]")
        loop_non_target.set_postfix(loss=loss_retain.item())
    
    avg_non_target_loss = running_non_target_loss / len(non_target_loader)
    tqdm.write(f"\033[34mEpoch [{epoch+1}/{num_epochs}]\033[0m, Average NON TARGET Loss: {avg_non_target_loss:.4f}")

    #### VALIDATION ####
    teacher_model.eval()
    validation_loss = 0
    total_samples, correct_predictions = 0, 0

    with torch.no_grad():
        for data, labels in target_loader:
            data, labels = data.to(device), labels.to(device)
            
            # Teacher output
            teacher_output = teacher_model(data)

            # Compute validation loss (target classes)
            loss_val_target = unlearning_energy_alignment_loss(teacher_output, delta_target)
            validation_loss += loss_val_target.item()

            # Accuracy calculation
            _, predicted = torch.max(teacher_output, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    avg_validation_loss = validation_loss / len(target_loader)
    accuracy = 100 * correct_predictions / total_samples

    print(f"Validation Loss: {avg_validation_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_target_loss, avg_non_target_loss
