import torch
import torch.nn as nn

# Energy alignment loss
def unlearning_energy_alignment_loss(predictions: torch.Tensor, delta: float) -> torch.Tensor:
    """
    # Unlearning Energy Alignment Loss

    The **unlearning_energy_alignment_loss** function computes a loss measure based on energy alignment 
    to encourage or discourage specific patterns in the model's outputs. 

    Args:
        - predictions (torch.Tensor): A tensor of dimensions (batch_size, num_classes) containing 
          the model's raw predictions (logits).
        - delta (float): A target or reference value representing the desired energy level.

    Returns:
        - torch.Tensor: A scalar tensor representing the computed loss value.
    """
    energy = -torch.logsumexp(-predictions, dim=1)
    return ((energy - delta) ** 2).mean()


# Knowledge Distillation Loss
def unlearning_knowledge_distillation_loss(student_output: torch.Tensor, teacher_output: torch.Tensor, temperature: float=2) -> torch.Tensor:
    """
    # Unlearning Knowledge Distillation Loss

    The **unlearning_knowledge_distillation_loss** function calculates a knowledge distillation loss 
    to evaluate the divergence between the softened probability distributions of a student model 
    and a teacher model. This function supports the process of unlearning by aligning model predictions 
    to achieve specific goals.

    Args:
        - student_output (torch.Tensor): The raw output (logits) of the student model, of shape 
          (batch_size, num_classes).
        - teacher_output (torch.Tensor): The raw output (logits) of the teacher model, of shape 
          (batch_size, num_classes).
        - temperature (float): A scaling factor to soften the logits before calculating the loss 
          (default is 2).

    Returns:
        - torch.Tensor: A scalar tensor representing the KL divergence loss between the softened 
          student and teacher distributions.
    """
    soft_student = torch.log_softmax(student_output / temperature, dim=1)
    soft_teacher = torch.softmax(teacher_output / temperature, dim=1)
    return nn.KLDivLoss(reduction="batchmean")(soft_student, soft_teacher)

def forget_kd_loss(student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
    return -nn.KLDivLoss(reduction="batchmean")( torch.log_softmax(teacher_output, dim=1), torch.softmax(student_output, dim=1))