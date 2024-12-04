import torch
from tqdm import tqdm
import os
from .losses import energy_alignment_loss, knowledge_distillation_loss

# Collaborative Collaboration
def collaborative_collaboration(epoch, num_epochs, best_loss, student_model, teacher_models, train_loader, val_loader, optimizer, criterion_ce, lambda_1=1.0, lambda_2=0.1, delta=-20, device="cpu"):
    student_model.train()
    running_loss = 0

    loop = tqdm(train_loader, total= len(train_loader), leave=True)
                
    for data, labels in loop:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # Student output
        student_output = student_model(data)

        # Cross-entropy loss
        loss_ce = criterion_ce(student_output, labels)

        # Distillation loss
        loss_kd = 0
        for teacher_model in teacher_models:
            teacher_model.eval()
            with torch.no_grad():
                teacher_output = teacher_model(data)
            loss_kd += knowledge_distillation_loss(student_output, teacher_output)

        # Energy alignment loss
        loss_al = energy_alignment_loss(student_output, delta)

        # Total loss
        loss = loss_ce + lambda_1 * loss_kd + lambda_2 * loss_al

        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item()

        # Update progress bar with loss and epoch information
        loop.set_description(f"Epoch [{epoch+1}]")
        loop.set_postfix(loss=loss.item())
    
    # Calculate average loss for the epoch
    avg_loss = running_loss / len(train_loader)

    # Print loss for this epoch
    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    #### VALIDATION ####
    student_model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device=device)
            labels = labels.to(device=device)

            # Student output
            student_output = student_model(data)

            # Cross-entropy loss
            loss_ce = criterion_ce(student_output, labels)

            # Distillation loss
            loss_kd = 0
            for teacher_model in teacher_models:
                teacher_model.eval()
                teacher_output = teacher_model(data)
                loss_kd += knowledge_distillation_loss(student_output, teacher_output)

            # Energy alignment loss
            loss_al = energy_alignment_loss(student_output, delta)

            # Total loss
            loss = loss_ce + lambda_1 * loss_kd + lambda_2 * loss_al
            val_loss += loss.item()
        # Calculate average loss for the epoch
        avg_val_loss = val_loss / len(val_loader)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        # if avg val_loss is better than the one before, save the model
        if epoch == 0:
            # create directory if not exist
            os.makedirs("checkpoint", exist_ok=True)
            best_loss = avg_val_loss
            torch.save(student_model.state_dict(), "./code/checkpoint/student_trained_model.pth")
        elif avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(student_model.state_dict(), "./code/checkpoint/student_trained_model.pth")
    return best_loss
        

# Reciprocal Altruism
def reciprocal_altruism(epoch, num_epochs, best_loss, teacher_idx, teacher_model, student_model, train_loader, val_loader, optimizer, criterion_ce, lambda_1=1.0, lambda_2=0.1, delta=-20, device="cpu"):
    teacher_model.train()
    running_loss = 0

    loop = tqdm(enumerate(train_loader), total= len(train_loader), leave=True)

    for batch_idx, (data, labels) in loop:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        # Teacher output
        teacher_output = teacher_model(data)

        # Student output
        with torch.no_grad():
            student_output = student_model(data)

        # Cross-entropy loss
        loss_ce = criterion_ce(teacher_output, labels)

        # Distillation loss
        loss_kd = knowledge_distillation_loss(teacher_output, student_output)

        # Energy alignment loss
        loss_al = energy_alignment_loss(teacher_output, delta)

        # Total loss
        loss = loss_ce + lambda_1 * loss_kd + lambda_2 * loss_al
        loss.backward()
        optimizer.step()

        running_loss += loss
        # Accumulate loss
        running_loss += loss.item()

        # Update progress bar with loss and epoch information
        loop.set_description(f"Epoch [{epoch+1}]")
        loop.set_postfix(loss=loss.item())
    
    # Calculate average loss for the epoch
    avg_loss = running_loss / len(train_loader)

    # Print loss for this epoch
    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    #### VALIDATION ####
    teacher_model.eval()
    val_loss = 0

    with torch.no_grad():
        for data, labels in val_loader:
            data = data.to(device=device)
            labels = labels.to(device=device)

            # Teacher output
            teacher_output = teacher_model(data)

            # Student output
            student_output = student_model(data)

            # Cross-entropy loss
            loss_ce = criterion_ce(teacher_output, labels)

            # Distillation loss
            loss_kd = knowledge_distillation_loss(teacher_output, student_output)

            # Energy alignment loss
            loss_al = energy_alignment_loss(teacher_output, delta)

            # Total loss
            loss = loss_ce + lambda_1 * loss_kd + lambda_2 * loss_al
            val_loss += loss.item()
        # Calculate average loss for the epoch
        avg_val_loss = val_loss / len(val_loader)

        print(f"Validation Loss: {avg_val_loss:.4f}")
        # if avg val_loss is better than the one before, save the model
        if epoch == 0:
            # create directory if not exist
            os.makedirs("checkpoint", exist_ok=True)
            best_loss = avg_val_loss
            torch.save(student_model.state_dict(), "./code/checkpoint/teacher_"+str(teacher_idx)+"_trained_model.pth")
        elif avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(student_model.state_dict(), "./code/checkpoint/teacher_"+str(teacher_idx)+"_trained_model.pth")
    
    return best_loss
