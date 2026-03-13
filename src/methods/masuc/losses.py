import torch
import torch.nn as nn
import torch.nn.functional as F

def unlearning_energy_alignment_loss(predictions: torch.Tensor, training_energies_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes an energy alignment loss to enforce or reduce the model's confidence in specific classes.

    Args:
        - predictions (torch.Tensor): A tensor of raw model outputs (logits), shape (batch_size, num_classes).
        - training_energies_target: Energy of the already computed data of the target train set

    Returns:
        - torch.Tensor: A scalar tensor representing the calculated loss value.
        - training_energies_target: updated energies of the computed data
    """
    energy = -torch.logsumexp(predictions, dim=1)
    training_energies_target = torch.cat((training_energies_target, energy))
    delta = training_energies_target.mean().item()

    return ((energy - delta) ** 2).mean(), training_energies_target


def unlearning_knowledge_distillation_loss(student_output: torch.Tensor, teacher_output: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """
    Calculates the distillation loss between the student and teacher outputs.

    Args:
        - student_output (torch.Tensor): The raw output (logits) of the student model.
        - teacher_output (torch.Tensor): The raw output (logits) of the teacher model.
        - temperature (float): A scaling factor for the logits before calculating the loss.

    Returns:
        - torch.Tensor: A scalar tensor representing the distillation loss (KL divergence).
    """
    soft_student = torch.log_softmax(student_output / temperature, dim=1)
    soft_teacher = torch.softmax(teacher_output / temperature, dim=1)
    
    kd_loss = nn.KLDivLoss(reduction="batchmean")(soft_student, soft_teacher)
    return kd_loss


def erasure_loss(student_output: torch.Tensor, forget_classes: list[int]) -> torch.Tensor:
    """
    Compute the erasure loss to maximize entropy for the forget classes.

    Args:
        student_output (Tensor): Output logits from the student model (batch_size, num_classes).
        forget_classes (list): List of class indices to forget.

    Returns:
        Tensor: The erasure loss value.
    """
    probabilities = F.softmax(student_output, dim=1)
    
    forget_probs = probabilities[:, forget_classes]
    
    loss = -torch.mean(torch.sum(forget_probs * torch.log(forget_probs + 1e-8), dim=1))
    return loss
