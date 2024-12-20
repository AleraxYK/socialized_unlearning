import torch
from tqdm import tqdm
import os
from .unlearning_losses import unlearning_energy_alignment_loss, unlearning_knowledge_distillation_loss, erasure_loss
from .unlearning_utils import feature_extractor, classifier_extractor

# Collaborative Unlearning
def collaborative_unlearning(epoch: int, num_epochs: int, best_loss: float, student_model, teacher_models, target_train_loader, non_target_train_loader, non_target_val_loader, optimizer, criterion_ce, scheduler, initial_lambda_1: float=1.0, lambda_2: float=0.1, lambda_3: float=0.5, delta_target: float=-5, delta_non_target: float=-20, device="cpu") -> float:
    """
    Collaborative Unlearning Training with additional Cross Entropy Loss based on disagreement.

    Args:
        - epoch (int): Current epoch number.
        - num_epochs (int): Total number of epochs.
        - best_loss (float): Best validation loss recorded so far.
        - student_model: The student model being trained.
        - teacher_models: List of teacher models for knowledge distillation.
        - target_train_loader: DataLoader for training data containing target classes that has to be forgotten.
        - non_target_train_loader: DataLoader for training data containing non-target classes.
        - non_target_val_loader: DataLoader for validation data containing non-target classes.
        - optimizer: Optimizer used for updating the model parameters.
        - criterion_ce: Cross-entropy loss function.
        - scheduler: Learning rate scheduler.
        - initial_lambda_1 (float): Weight for the reverse knowledge distillation loss term (default is 1.0).
        - lambda_2 (float): Weight for the energy alignment loss term (default is 0.1).
        - lambda_3 (float): Weight for the disagreement loss term (default is 0.5).
        - lambda_4 (float): Weight for the forgetting cross-entropy loss term (default is 0.5).
        - delta_target (float): Desired energy alignment value for target classes (default is -5).
        - delta_non_target (float): Desired energy alignment value for non-target classes (default is -20).
        - device (str): Device for training ('cpu' or 'cuda').

    Returns:
        - best_loss (float): Best validation loss recorded during training.
    """
    student_model.train()
    running_loss = 0
    target_running_loss = 0

    loop_target = tqdm(target_train_loader, total=len(target_train_loader), leave=True)

    # Forget target classes
    forget_classes = [0, 4]

    for data, labels in loop_target:
        data, labels = data.to(device), labels.to(device)

        # Student output
        student_output = student_model(data)

        # Learning from teacher
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

        lambda_loss_kd = lambda_1 * loss_kd

        # Erasure Loss (maximize entropy for forget classes)
        loss_erasure = erasure_loss(student_output, forget_classes)
        lambda_loss_erasure = lambda_3 * loss_erasure

        # Energy alignment loss
        loss_al = unlearning_energy_alignment_loss(student_output, delta_target)
        lambda_loss_al = lambda_2 * loss_al

        # Total loss
        loss = lambda_loss_erasure + lambda_loss_kd + lambda_loss_al
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        target_running_loss += loss.item()

        # Update progress bar
        loop_target.set_description(f"Student - Target Epoch [{epoch+1}]")
        loop_target.set_postfix(loss=loss.item())

    avg_target_loss = target_running_loss / len(target_train_loader)

    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], STUDENT Average TARGET (to forget) Loss: {avg_target_loss:.4f}")

    # loop_non_target = tqdm(non_target_train_loader, total=len(non_target_train_loader), leave=True)

    # Preserve non-target classes by using energy alignment and cross-entropy
    # TODO: capire se invece di allenare congelare i parametri influenzati dal retain set in modo da mantenere le prestazioni
    # for data, labels in loop_non_target:
    #     data, labels = data.to(device), labels.to(device)
    #     optimizer.zero_grad()

    #     # Student output
    #     student_output = student_model(data)

    #     # Cross-entropy loss for non-target classes
    #     loss_ce = criterion_ce(student_output, labels)

    #     # Knowledge distillation loss for non-target classes
    #     loss_kd = 0
    #     for teacher_model in teacher_models:
    #         teacher_model.eval()
    #         modified_input = feature_extractor(student_model, data)
    #         teacher_output1 = classifier_extractor(teacher_model, modified_input)
    #         teacher_output2 = teacher_model(data)

    #         loss_kd += unlearning_knowledge_distillation_loss(teacher_output1, teacher_output2)

    #     loss_kd /= len(teacher_models)

    #     # Energy alignment loss for non-target classes
    #     loss_energy_non_target = unlearning_energy_alignment_loss(student_output, delta_non_target)

    #     # Total loss for non-target classes (preservation)
    #     loss = loss_ce + lambda_1 * loss_kd + lambda_2 * loss_energy_non_target
    #     loss.backward()
    #     optimizer.step()

    #     running_loss += loss.item()

    #     # Update progress bar
    #     loop_non_target.set_description(f"Student - Non-target Epoch [{epoch+1}]")
    #     loop_non_target.set_postfix(loss=loss.item())

    # avg_loss = running_loss / len(non_target_train_loader)
    # scheduler.step(avg_loss)

    # tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], STUDENT Average NON TARGET (to preserve) Loss: {avg_loss:.4f}")

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

        print(f"STUDENT Validation Loss: {avg_val_loss:.4f}")
        if epoch == 0 or avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(student_model.state_dict(), "./code/checkpoint/UNLEARNED_student_trained_model.pth")

    return student_model, optimizer, scheduler, best_loss
        

# Reciprocal Altruism
def unlearning_reciprocal_altruism(epoch: int, num_epochs: int, best_loss: float, teacher_idx: int, teacher_model, student_model, 
                                 target_train_loader, non_target_train_loader, non_target_val_loader, optimizer, 
                                 criterion_ce, scheduler, initial_lambda_1: float=0.5, lambda_2: float=0.3, 
                                 delta_target: float=-5, delta_non_target: float=-20, device: str="cpu") -> float:
    """
    Reciprocal Altruism Training Function with separate handling for target and non-target classes.

    Args:
        - epoch (int): Current epoch number.
        - num_epochs (int): Total number of epochs.
        - best_loss (float): Best validation loss recorded so far.
        - teacher_idx (int): Index of the current teacher model.
        - teacher_model: The teacher model being trained.
        - student_model: The pre-trained student model.
        - target_train_loader: DataLoader for training data containing target classes to be forgotten.
        - non_target_train_loader: DataLoader for training data containing non-target classes.
        - non_target_val_loader: DataLoader for validation data containing non-target classes.
        - optimizer: Optimizer for updating the teacher model parameters.
        - criterion_ce: Cross-entropy loss function.
        - scheduler: Learning rate scheduler.
        - initial_lambda_1 (float): Initial weight for the knowledge distillation loss term (default is 0.5).
        - lambda_2 (float): Weight for the energy alignment loss term (default is 0.3).
        - delta_target (float): Desired energy alignment value for target classes (default is -5).
        - delta_non_target (float): Desired energy alignment value for non-target classes (default is -20).
        - device (str): Device for training ('cpu' or 'cuda').

    Returns:
        - float: The updated best validation loss.
    """
    teacher_model.train()
    running_loss = 0
    target_running_loss = 0

    forget_classes = [0, 4]  # Classi da dimenticare

    # Handle target classes (to forget)
    loop_target = tqdm(target_train_loader, total=len(target_train_loader), leave=True)
    for data, labels in loop_target:
        data, labels = data.to(device), labels.to(device)

        # Teacher's direct output
        teacher_output = teacher_model(data)

        # Erasure Loss for target classes
        loss_erasure = erasure_loss(teacher_output, forget_classes)

        # Distillation Loss (teacher follows student)
        with torch.no_grad():
            student_features = feature_extractor(student_model, data)
        teacher_output_after_student = classifier_extractor(teacher_model, student_features)
        loss_kd = unlearning_knowledge_distillation_loss(teacher_output_after_student, teacher_output)

        # Energy Alignment Loss
        loss_al = unlearning_energy_alignment_loss(teacher_output, delta_target)
        lambda_1 = initial_lambda_1 * (1 - epoch / num_epochs)

        # Total loss for target classes
        loss_target = loss_erasure + lambda_1 * loss_kd + lambda_2 * loss_al

        optimizer.zero_grad()
        loss_target.backward()
        optimizer.step()

        target_running_loss += loss_target.item()
        loop_target.set_description(f"Teacher {teacher_idx} - Target Epoch [{epoch+1}]")
        loop_target.set_postfix(loss=loss_target.item())

    avg_target_loss = target_running_loss / len(target_train_loader)

    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], TEACHER {teacher_idx} Average TARGET (to forget) Loss: {avg_target_loss:.4f}")

    # TODO: capire se invece di allenare congelare i parametri influenzati dal retain set in modo da mantenere le prestazioni
    # loop_non_target = tqdm(non_target_train_loader, total=len(non_target_train_loader), leave=True)
    # for data, labels in loop_non_target:
    #     data, labels = data.to(device), labels.to(device)

    #     # Teacher's direct output
    #     teacher_output = teacher_model(data)

    #     # Cross-Entropy Loss for non-target classes
    #     loss_ce = criterion_ce(teacher_output, labels)

    #     # Distillation Loss (teacher follows student)
    #     with torch.no_grad():
    #         student_features = feature_extractor(student_model, data)
    #     teacher_output_after_student = classifier_extractor(teacher_model, student_features)
    #     loss_kd = unlearning_knowledge_distillation_loss(teacher_output_after_student, teacher_output)

    #     # Energy Alignment Loss
    #     loss_al = unlearning_energy_alignment_loss(teacher_output, delta_non_target)

    #     # Total loss for non-target classes
    #     loss_non_target = loss_ce + initial_lambda_1 * loss_kd + lambda_2 * loss_al

    #     optimizer.zero_grad()
    #     loss_non_target.backward()
    #     optimizer.step()

    #     running_loss += loss_non_target.item()
    #     loop_non_target.set_description(f"Teacher {teacher_idx} - Non-Target Epoch [{epoch+1}]")
    #     loop_non_target.set_postfix(loss=loss_non_target.item())

    # avg_loss = running_loss / (len(target_train_loader) + len(non_target_train_loader))
    # scheduler.step(avg_loss)
    # tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], TEACHER {teacher_idx} Average NON TARGET (to preserve) Loss: {avg_target_loss:.4f}")

    # Validation
    teacher_model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, labels in non_target_val_loader:
            data, labels = data.to(device), labels.to(device)

            teacher_output = teacher_model(data)
            loss_ce = criterion_ce(teacher_output, labels)

            with torch.no_grad():
                student_features = feature_extractor(student_model, data)
            teacher_output_after_student = classifier_extractor(teacher_model, student_features)
            loss_kd = unlearning_knowledge_distillation_loss(teacher_output_after_student, teacher_output)

            loss_al = unlearning_energy_alignment_loss(teacher_output, delta_non_target)

            loss = loss_ce + initial_lambda_1 * loss_kd + lambda_2 * loss_al
            val_loss += loss.item()

        avg_val_loss = val_loss / len(non_target_val_loader)

        print(f"TEACHER {teacher_idx} Validation Loss: {avg_val_loss:.4f}")

        if epoch == 0 or avg_val_loss < best_loss:
            best_loss = avg_val_loss
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(teacher_model.state_dict(), f"./code/checkpoint/UNLEARNED_teacher_{teacher_idx}_trained_model.pth")

    return teacher_model, optimizer, scheduler, best_loss


def find_freezable_params(model, retain_set, ce_loss):
    """
    Function that freeze the parameters that are mostly involved by the retain set

    Args:
        - model: our trainable model
    
    Returns:
        - model: model with freezed params
    """
    loop_target = tqdm(retain_set, total=len(retain_set), leave=True)
    # Compute gradients to see which params are involved more during the computation
    for input, targets in loop_target:
        outputs = model(input)
        loss = ce_loss(outputs, targets)
        loss.backward()
    
    # Freezing part
    threshold = 0.01
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"Gradient norm for {name}: {grad_norm}")
            if grad_norm > threshold:
                print(f"Freezing parameter: {name}")
                param.require_grad = False

    return model
