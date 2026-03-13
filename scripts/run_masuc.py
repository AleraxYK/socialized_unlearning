import os
import json as js
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.data.cifar10 import get_cifar10
from src.models import create_model
from src.utils.device import get_device
from src.eval.metrics import evaluate

from src.methods.masuc.train import collaborative_unlearning, reciprocal_altruism

def run_masuc():
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/reports", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    hyperparams = {
        "backbone":     "resnet18",
        "epochs":       5,              
        "batch_size":   128,            
        "lr_student":   0.001,          
        "lr_teacher":   0.0001,         
        "momentum":     0.9,
        "weight_decay": 5e-4,
        "num_workers":  4,
        "lambda_1":     1.0,            
        "lambda_2":     0.1,            
        "lambda_3":     0.5,            
        "forget_class": 3               
    }

    train_ds = get_cifar10("./data", train=True)
    test_ds  = get_cifar10("./data", train=False)

    with open(f"results/splits/split_forget_{hyperparams['forget_class']}.json", "r") as f:
        split = js.load(f)

    forget_train_ds = Subset(train_ds, split["indices"]["forget_train"])
    forget_test_ds  = Subset(test_ds,  split["indices"]["forget_test"])
    retain_test_ds  = Subset(test_ds,  split["indices"]["retain_test"])

    forget_train_loader = DataLoader(forget_train_ds, batch_size=hyperparams["batch_size"], shuffle=True,  num_workers=hyperparams["num_workers"])
    forget_test_loader  = DataLoader(forget_test_ds,  batch_size=256,                       shuffle=False, num_workers=hyperparams["num_workers"])
    retain_test_loader  = DataLoader(retain_test_ds,  batch_size=256,                       shuffle=False, num_workers=hyperparams["num_workers"])


    print("Loading models...")
    student = create_model(hyperparams["backbone"], num_classes=10)
    student.load_state_dict(torch.load("results/checkpoints/model_before_best.pth", weights_only=True, map_location=device))
    student = student.to(device)

    teachers = {}
    teacher_optimizers = {}
    for i in range(5):
        t = create_model(hyperparams["backbone"], num_classes=10)
        t.load_state_dict(torch.load(f"results/checkpoints/teacher_{i}.pth", weights_only=True, map_location=device))
        t = t.to(device)
        teachers[i] = t
        teacher_optimizers[i] = optim.SGD(t.parameters(), lr=hyperparams["lr_teacher"], momentum=hyperparams["momentum"], weight_decay=hyperparams["weight_decay"])

    student_optimizer = optim.SGD(
        student.parameters(),
        lr=hyperparams["lr_student"],
        momentum=hyperparams["momentum"],
        weight_decay=hyperparams["weight_decay"],
    )

    history = {"epoch": [], "train_loss": [], "retain_acc": [], "forget_acc": []}

    print(f"\n{'='*60}\n\t=== MASUC Unlearning ===\n{'='*60}\n")
    t0_total = time.time()

    for ep in range(1, hyperparams["epochs"] + 1):
        print(f"\n--- Epoch {ep}/{hyperparams['epochs']} ---")
        
        for tid, teacher in teachers.items():
            reciprocal_altruism(
                ep=ep,
                num_epochs=hyperparams["epochs"],
                teacher_id=tid,
                teacher_model=teacher,
                student_model=student,
                forget_train_loader=forget_train_loader,
                forget_class=hyperparams["forget_class"],
                optimizer=teacher_optimizers[tid],
                initial_lambda_1=hyperparams["lambda_1"],
                lambda_2=hyperparams["lambda_2"],
                device=device
            )

        metrics, _ = collaborative_unlearning(
            ep=ep,
            num_epochs=hyperparams["epochs"],
            student_model=student,
            teacher_models=teachers,
            forget_train_loader=forget_train_loader,
            retain_test_loader=retain_test_loader,
            forget_test_loader=forget_test_loader,
            optimizer=student_optimizer,
            forget_class=hyperparams["forget_class"],
            initial_lambda_1=hyperparams["lambda_1"],
            lambda_2=hyperparams["lambda_2"],
            lambda_3=hyperparams["lambda_3"],
            device=device
        )

        history["epoch"].append(ep)
        history["train_loss"].append(metrics["train_loss"])
        history["retain_acc"].append(metrics["retain_acc"])
        history["forget_acc"].append(metrics["forget_acc"])

    print(f"\nMASUC Finished! Total elapsed: {time.time()-t0_total:.1f}s")

    torch.save(student.state_dict(), "results/checkpoints/masuc_final.pth")
    print("\nSalvato: results/checkpoints/masuc_final.pth")

    with open("results/reports/masuc_curve.json", "w") as f:
        js.dump(history, f, indent=2)

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["retain_acc"], label="retain_acc")
    plt.plot(history["epoch"], history["forget_acc"], label="forget_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MASUC — Retain vs Forget Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/masuc_acc.png", dpi=400, bbox_inches="tight")
    plt.close()
    print("Salvato: results/plots/masuc_acc.png")

if __name__ == "__main__":
    run_masuc()
