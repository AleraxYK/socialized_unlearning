import torch
from tqdm import tqdm
import os
from .unlearning_losses import unlearning_energy_alignment_loss, unlearning_knowledge_distillation_loss
from .unlearning_utils import feature_extractor, classifier_extractor

# Collaborative Unlearning
def collaborative_unlearning(epoch: int, num_epochs: int, best_loss: float, student_model, teacher_models, target_train_loader, non_target_train_loader, non_target_val_loader, optimizer, criterion_ce, scheduler, initial_lambda_1: float=1.0, lambda_2: float=0.1, delta_target: float=-5, delta_non_target: float=-20, device="cpu") -> float:
    """
    # Collaborative Unlearning Training

    The **collaborative_unlearning** function trains a student model collaboratively with multiple teacher models 
    and uses techniques to forget information about specific target classes while preserving information about non-target classes.

    Args:
        - epoch (int): Current epoch number.
        - num_epochs (int): Total number of epochs.
        - best_loss (float): Best validation loss recorded so far.
        - student_model: The student model being trained.
        - teacher_models: List of teacher models for knowledge distillation.
        - target_train_loader: DataLoader for training data containing target classes.
        - non_target_train_loader: DataLoader for training data containing non-target classes.
        - non_target_val_loader: DataLoader for validation data containing non-target classes.
        - optimizer: Optimizer used for updating the model parameters.
        - criterion_ce: Cross-entropy loss function.
        - scheduler: Learning rate scheduler
        - initial_lambda_1 (float): Weight for the reverse knowledge distillation loss term (default is 1.0).
        - lambda_2 (float): Weight for the energy alignment loss term (default is 0.1).
        - delta_target (float): Desired energy alignment value for target classes (default is -5).
        - delta_non_target (float): Desired energy alignment value for non-target classes (default is -20).
        - device (str): Device for training ('cpu' or 'cuda').

    Returns:
        - best_loss (float): Best validation loss recorded during training.
    """
    
    student_model.train()
    running_loss = 0

    loop_target = tqdm(target_train_loader, total=len(target_train_loader), leave=True)

    # Forget target classes by applying reverse distillation
    for data, labels in loop_target:
        data, labels = data.to(device), labels.to(device)

        # Student output
        student_output = student_model(data)

        # Reverse knowledge distillation loss (forgetting target classes)
        loss_kd = 0
        for teacher_model in teacher_models:
            teacher_model.eval()
            with torch.no_grad():
                modified_input = feature_extractor(student_model, data)
                teacher_output1 = classifier_extractor(teacher_model, modified_input)
                teacher_output2 = teacher_model(data)
                
                loss_kd += unlearning_knowledge_distillation_loss(teacher_output1, teacher_output2)

        loss_kd /= len(teacher_models)  # Normalize the KD loss

        lambda_1 = initial_lambda_1 * (1 - epoch / num_epochs)

        # Energy alignment to reduce confidence for target classes
        loss_energy_target = unlearning_energy_alignment_loss(student_output, delta_target)

        # Total loss for target classes (loss for unlearning)
        loss = lambda_1 * loss_kd + lambda_2 * loss_energy_target
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar
        loop_target.set_description(f"TARGET Epoch [{epoch+1}]")

    loop_non_target = tqdm(non_target_train_loader, total=len(non_target_train_loader), leave=True)

    # Preserve non-target classes by using energy alignment and cross-entropy
    for data, labels in loop_non_target:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # Student output
        student_output = student_model(data)

        # Energy alignment to keep confidence for non-target classes
        loss_energy_non_target = unlearning_energy_alignment_loss(student_output, delta_non_target)

        # Cross-entropy loss for non-target classes
        loss_ce = criterion_ce(student_output, labels)

        # Total loss for non-target classes (preservation)
        loss = loss_ce + lambda_2 * loss_energy_non_target
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Update progress bar
        loop_non_target.set_description(f"NON TARGET Epoch [{epoch+1}]")
        loop_non_target.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(non_target_train_loader)
    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Average NON TARGET Loss: {avg_loss:.4f}")

    #### VALIDATION ####
    student_model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, labels in non_target_val_loader:
            data, labels = data.to(device), labels.to(device)

            # Student output
            student_output = student_model(data)

            # Cross-entropy loss for non-target classes
            loss_ce = criterion_ce(student_output, labels)

            # Knowledge distillation loss for non-target classes
            loss_kd = 0
            for teacher_model in teacher_models:
                teacher_model.eval()
                modified_input = feature_extractor(student_model, data)
                teacher_output1 = classifier_extractor(teacher_model, modified_input)
                teacher_output2 = teacher_model(data)
                
                loss_kd += unlearning_knowledge_distillation_loss(teacher_output1, teacher_output2)
            
            loss_kd /= len(teacher_models)

            # Energy alignment loss for non-target classes
            loss_energy_non_target = unlearning_energy_alignment_loss(student_output, delta_non_target)

            # Total loss
            loss = loss_ce + lambda_1 * loss_kd + lambda_2 * loss_energy_non_target
            val_loss += loss.item()

        avg_val_loss = val_loss / len(non_target_val_loader)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        if epoch == 0 or avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(student_model.state_dict(), "./code/checkpoint/student_unlearning_trained_model.pth")

    return best_loss
        

# Reciprocal Altruism
def reciprocal_altruism(epoch: int, num_epochs: int, best_loss: float, teacher_idx: int, teacher_model, student_model, train_loader, val_loader, optimizer, criterion_ce, scheduler, initial_lambda_1: float=0.5, lambda_2: float=0.3, delta: float=-20, device: str="mps") -> float:
    """
    # Reciprocal Altruism Training Function

    The **reciprocal_altruism** function trains a teacher model in a reciprocal setup 
    where knowledge is transferred from a pre-trained student model to improve the teacher's performance. 
    This involves a combination of cross-entropy loss, knowledge distillation loss, 
    and energy alignment loss.

    Args:
        - epoch (int): Current epoch number.
        - num_epochs (int): Total number of epochs.
        - best_loss (float): Best validation loss recorded so far.
        - teacher_idx (int): Index of the current teacher model.
        - teacher_model: The teacher model being trained.
        - student_model: The pre-trained student model.
        - train_loader: DataLoader for training data.
        - val_loader: DataLoader for validation data.
        - optimizer: Optimizer for updating the teacher model parameters.
        - criterion_ce: Cross-entropy loss function.
        - lambda_1 (float): Weight for the knowledge distillation loss term (default is 1.0).
        - lambda_2 (float): Weight for the energy alignment loss term (default is 0.1).
        - delta (float): Desired energy alignment value (default is -20).
        - device (str): Device for training ('cpu' or 'cuda').

    Returns:
        - float: The updated best validation loss.
    """
    teacher_model.train()
    running_loss = 0

    loop = tqdm(train_loader, total= len(train_loader), leave=True)

    for data, labels in loop:
        data, labels = data.to(device), labels.to(device)

        teacher_output = teacher_model(data)  # Teacher's direct output

        # Cross-entropy loss using teacher's output
        loss_ce = criterion_ce(teacher_output, labels)

        # Distillation loss for the current teacher
        with torch.no_grad():
            student_features = feature_extractor(student_model, data)
            student_output = classifier_extractor(student_model, student_features)

        teacher_output_after_student = classifier_extractor(teacher_model, student_features)

        loss_kd = unlearning_knowledge_distillation_loss(teacher_output_after_student, teacher_output)  # Compare with features

        # Energy alignment loss
        loss_al = unlearning_energy_alignment_loss(teacher_output, delta)
        lambda_1 = initial_lambda_1 * (1 - epoch / num_epochs)

        # Total loss combining cross-entropy, distillation, and energy alignment
        loss = loss_ce + lambda_1 * loss_kd + lambda_2 * loss_al

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss
        # Accumulate loss
        running_loss += loss.item()

        # Update progress bar with loss and epoch information
        loop.set_description(f"\033[34mTeacher {teacher_idx} learning Epoch [{epoch+1}]\033[0m")
        loop.set_postfix(loss=loss.item())
    
    # Calculate average loss for the epoch
    avg_loss = running_loss / len(train_loader)

    scheduler.step(avg_loss)

    # Print loss for this epoch
    tqdm.write(f"\033[34mTeacher {teacher_idx} learning Epoch [{epoch+1}/{num_epochs}]\033[0m, Average Loss: {avg_loss:.4f}")

    #### VALIDATION ####
    teacher_model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device=device)
            labels = labels.to(device=device)

            teacher_output = teacher_model(data)  # Teacher's direct output

            # Cross-entropy loss using teacher's output
            loss_ce = criterion_ce(teacher_output, labels)

            # Distillation loss for the current teacher
            student_features = feature_extractor(student_model, data)

            teacher_output_after_student = classifier_extractor(teacher_model, student_features)

            loss_kd = unlearning_knowledge_distillation_loss(teacher_output_after_student, teacher_output)  # Compare with features

            # Energy alignment loss
            loss_al = unlearning_energy_alignment_loss(teacher_output, delta)

            # Total loss combining cross-entropy, distillation, and energy alignment
            loss = loss_ce + lambda_1 * loss_kd + lambda_2 * loss_al
            val_loss += loss.item()
        # Calculate average loss for the epoch
        avg_val_loss = val_loss / len(val_loader)

        print(f"Teacher {teacher_idx} learning Validation Loss: {avg_val_loss:.4f}")
        # if avg val_loss is better than the one before, save the model
        if epoch == 0:
            # create directory if not exist
            os.makedirs("checkpoint", exist_ok=True)
            best_loss = avg_val_loss
            torch.save(teacher_model.state_dict(), "./code/checkpoint/teacher_"+str(teacher_idx)+"_socialized_trained_model.pth")
        elif avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(teacher_model.state_dict(), "./code/checkpoint/teacher_"+str(teacher_idx)+"_socialized_trained_model.pth")
    
    return teacher_model, optimizer, scheduler, best_loss
