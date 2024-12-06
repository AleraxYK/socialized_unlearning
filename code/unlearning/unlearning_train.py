import torch
from tqdm import tqdm
import os
from .unlearning_losses import unlearning_knowledge_distillation_loss, unlearning_energy_alignment_loss

def collaborative_unlearning(epoch, num_epochs, student_model, teacher_models, target_loader, non_target_loader, optimizer, criterion_ce, lambda_1=1.0, lambda_2=0.1, delta_target=-5, delta_non_target=-20, device="cpu"):
    student_model.train()
    running_target_loss = 0
    running_non_target_loss = 0

    loop_target = tqdm(target_loader, total=len(target_loader), leave=True)

    # Dimenticare le classi target
    for data, labels in loop_target:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # Output dello studente
        student_output = student_model(data)

        # Teacher output
        teacher_outputs = []
        for teacher_model in teacher_models:
            teacher_model.eval()
            with torch.no_grad():
                teacher_outputs.append(teacher_model(data))

        # reverse Knowledge distillation
        loss_forget = 0
        for teacher_output in teacher_outputs:
            loss_forget += -unlearning_knowledge_distillation_loss(student_output, teacher_output)

        # Energy alignment to reduce confidence
        loss_energy_target = unlearning_energy_alignment_loss(student_output, delta_target)

        # Total loss
        loss = lambda_1 * loss_forget + lambda_2 * loss_energy_target
        loss.backward()
        optimizer.step()

        running_target_loss += loss.item()

        # Update progress bar with loss and epoch information
        loop_target.set_description(f"TARGET Epoch [{epoch+1}]")
        loop_target.set_postfix(loss=loss.item())
    
    # Calculate average loss for the epoch
    avg_target_loss = running_target_loss / len(target_loader)
    # Print loss for this epoch
    tqdm.write(f"\033[34mEpoch [{epoch+1}/{num_epochs}]\033[0m, Average TARGET Loss: {avg_target_loss:.4f}")
    
    loop_non_target = tqdm(non_target_loader, total=len(non_target_loader), leave=True)

    # Preserve knowledge for non target classes
    for data, labels in loop_non_target:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # Student output
        student_output = student_model(data)

        # Energy alignment to keep confidence
        loss_energy_non_target = unlearning_energy_alignment_loss(student_output, delta_non_target)

        # Loss totale
        loss = criterion_ce(student_output, labels) + lambda_2 * loss_energy_non_target
        loss.backward()
        optimizer.step()

        running_non_target_loss += loss.item()

        # Update progress bar with loss and epoch information
        loop_non_target.set_description(f"NON TARGET Epoch [{epoch+1}]")
        loop_non_target.set_postfix(loss=loss.item())
    
    # Calculate average loss for the epoch
    avg_non_target_loss = running_non_target_loss / len(non_target_loader)

    # Print loss for this epoch
    tqdm.write(f"\033[34mEpoch [{epoch+1}/{num_epochs}]\033[0m, Average NON TARGET Loss: {avg_non_target_loss:.4f}")

    #### VALIDATION #### TODOOOOOOOOOO


def reciprocal_unlearning(epoch, num_epochs, teacher_model, student_model, target_loader, non_target_loader, optimizer, criterion_ce, lambda_1=1.0, lambda_2=0.1, delta_target=-5, delta_non_target=-20, device="cpu"):
    teacher_model.train()
    running_target_loss = 0
    running_non_target_loss = 0

    loop_target = tqdm(target_loader, total=len(target_loader), leave=True)

    # Forget target classes
    for data, labels in loop_target:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # Output del teacher
        teacher_output = teacher_model(data)

        # Output dello studente (in modalità eval per evitare gradienti)
        with torch.no_grad():
            student_output = student_model(data)

        # Perdita per dimenticare (Knowledge Distillation inversa + Energy Alignment)
        loss_forget_kd = -unlearning_energy_alignment_loss(teacher_output, student_output)
        loss_energy_target = unlearning_energy_alignment_loss(teacher_output, delta_target)
        loss = lambda_1 * loss_forget_kd + lambda_2 * loss_energy_target

        loss.backward()
        optimizer.step()

        running_target_loss += loss.item()

        # Update progress bar with loss and epoch information
        loop_target.set_description(f"TARGET Epoch [{epoch+1}]")
        loop_target.set_postfix(loss=loss.item())
    
    # Calculate average loss for the epoch
    avg_target_loss = running_target_loss / len(target_loader)
    # Print loss for this epoch
    tqdm.write(f"\033[34mEpoch [{epoch+1}/{num_epochs}]\033[0m, Average TARGET Loss: {avg_target_loss:.4f}")
    
    loop_non_target = tqdm(non_target_loader, total=len(non_target_loader), leave=True)

    # Preservare le classi non target
    for data, labels in loop_non_target:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # Output del teacher
        teacher_output = teacher_model(data)

        # Perdita per preservare (Cross-Entropy Loss + Energy Alignment)
        loss_retain_ce = criterion_ce(teacher_output, labels)
        loss_energy_non_target = unlearning_energy_alignment_loss(teacher_output, delta_non_target)
        loss_retain = loss_retain_ce + lambda_2 * loss_energy_non_target

        loss_retain.backward()
        optimizer.step()

        running_non_target_loss += loss.item()

        # Update progress bar with loss and epoch information
        loop_non_target.set_description(f"NON TARGET Epoch [{epoch+1}]")
        loop_non_target.set_postfix(loss=loss.item())
    
    # Calculate average loss for the epoch
    avg_non_target_loss = running_non_target_loss / len(non_target_loader)

    # Print loss for this epoch
    tqdm.write(f"\033[34mEpoch [{epoch+1}/{num_epochs}]\033[0m, Average NON TARGET Loss: {avg_non_target_loss:.4f}")

    #### VALIDATION #### TODO

