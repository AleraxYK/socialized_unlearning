import torch
import torch.nn as nn

# Energy Alignment Loss
def energy_alignment_loss(predictions, delta : float) -> float:
    """
    # Energy Aligment Loss
    The **energy_alignment_loss** function implements a loss measure called Energy Alignment Loss.

    Args:
        - predictions (tensor) : A tensor of dimensions (batch_size, num_classes). 
        - delta (float) : A constant or target value representing the desired energy level.

    Returns:
        - A scaled value representing the overall loss.
    """
    with torch.no_grad():
        energy_values = -torch.logsumexp(-predictions, dim=1)
        delta = energy_values.mean().item()
    
    return torch.abs(energy_values - delta).mean()


# Knowledge Distillation Loss
def knowledge_distillation_loss(student_output, teacher_output, temperature: float = 2) -> float:
    """
    # Knowledge Distillation Loss
    The **knowledge_distillation_loss** function implements a loss measure for Knowledge Distillation.
    This approach is used to transfer knowledge from a more complex model (teacher) to a simpler one (student), 
    aiming to align the student's predictions with the teacher's outputs.

    Args:
        - student_output (tensor): A tensor of dimensions (batch_size, num_classes) containing the logits from the student model.
        - teacher_output (tensor): A tensor of dimensions (batch_size, num_classes) containing the logits from the teacher model.
        - temperature (float): A parameter controlling the smoothing of probability distributions (default is 2).

    Returns:
        - A scalar value representing the KL divergence loss between the student and teacher predictions.
    """

    soft_student = torch.log_softmax(student_output / temperature, dim=1)
    soft_teacher = torch.softmax(teacher_output / temperature, dim=1)
    return nn.KLDivLoss(reduction="batchmean")(soft_student, soft_teacher)

