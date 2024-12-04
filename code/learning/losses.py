import torch
import torch.nn as nn

# Energy Alignment Loss
def energy_alignment_loss(predictions, delta):
    energy = -torch.logsumexp(-predictions, dim=1)
    return ((energy - delta) ** 2).mean()

# Knowledge Distillation Loss
def knowledge_distillation_loss(student_output, teacher_output, temperature=2):
    soft_student = torch.log_softmax(student_output / temperature, dim=1)
    soft_teacher = torch.softmax(teacher_output / temperature, dim=1)
    return nn.KLDivLoss(reduction="batchmean")(soft_student, soft_teacher)
