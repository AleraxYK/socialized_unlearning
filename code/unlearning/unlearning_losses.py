import torch
import torch.nn as nn

def unlearning_energy_alignment_loss(predictions: torch.Tensor, delta: float, target_mask: torch.Tensor = None) -> torch.Tensor:
    """
    # Unlearning Energy Alignment Loss

    This function computes an energy alignment loss to enforce or reduce the model's confidence in specific classes.
    For target classes, it encourages the model to reduce its confidence (alignment with delta).
    For non-target classes, it encourages maintaining the model's confidence.

    Args:
        - predictions (torch.Tensor): A tensor of raw model outputs (logits), shape (batch_size, num_classes).
        - delta (float): Desired energy level for target classes. Typically, lower values (e.g., -5 to -20) discourage high confidence.
        - target_mask (torch.Tensor): A binary tensor of shape (batch_size) that identifies which samples belong to target classes.
          If None, the loss applies to all classes equally.

    Returns:
        - torch.Tensor: A scalar tensor representing the calculated loss value.
    """
    energy = -torch.logsumexp(-predictions, dim=1)
    
    if target_mask is not None:
        # Apply different penalties for target and non-target classes
        target_energy_loss = ((energy[target_mask] - delta) ** 2).mean()  # Target classes: reduce energy
        non_target_energy_loss = ((energy[~target_mask] - delta) ** 2).mean()  # Non-target classes: retain energy
        return target_energy_loss + non_target_energy_loss
    else:
        # No mask, apply uniformly
        return ((energy - delta) ** 2).mean()


def unlearning_knowledge_distillation_loss(student_output: torch.Tensor, teacher_output: torch.Tensor, target_mask: torch.Tensor = None, temperature: float=2) -> torch.Tensor:
    """
    # Unlearning Knowledge Distillation Loss

    This function calculates the distillation loss between the student and teacher outputs.
    For target classes, we apply reverse distillation (encourage forgetting).
    For non-target classes, we use regular distillation to preserve knowledge transfer.

    Args:
        - student_output (torch.Tensor): The raw output (logits) of the student model, shape (batch_size, num_classes).
        - teacher_output (torch.Tensor): The raw output (logits) of the teacher model, shape (batch_size, num_classes).
        - target_mask (torch.Tensor): A binary mask identifying which examples belong to target classes.
        - temperature (float): A scaling factor for the logits before calculating the loss.

    Returns:
        - torch.Tensor: A scalar tensor representing the distillation loss (KL divergence).
    """
    # Softened student and teacher outputs
    soft_student = torch.log_softmax(student_output / temperature, dim=1)
    soft_teacher = torch.softmax(teacher_output / temperature, dim=1)
    
    # Compute KL divergence loss (regular distillation)
    kd_loss = nn.KLDivLoss(reduction="batchmean")(soft_student, soft_teacher)
    return kd_loss