import os
import time
import json as js
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.data.cifar10 import get_cifar10
from src.models import create_model
from src.eval.metrics import evaluate
from src.utils.device import get_device


def baseline_ft():
    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/reports", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    hyperparams = {
        "backbone":     "resnet18",
        "epochs":       5,
        "batch_size":   128,
        "lr":           0.01,
        "momentum":     0.9,
        "weight_decay": 5e-4,
        "num_workers":  4,
    }

    train_ds = get_cifar10("./data", train=True)
    test_ds  = get_cifar10("./data", train=False)

    with open("results/splits/split_forget_3.json", "r") as f:
        split = js.load(f)

    retain_train_ds = Subset(train_ds, split["indices"]["retain_train"])
    forget_test_ds  = Subset(test_ds,  split["indices"]["forget_test"])
    retain_test_ds  = Subset(test_ds,  split["indices"]["retain_test"])

    retain_train_loader = DataLoader(retain_train_ds, batch_size=hyperparams["batch_size"], shuffle=True,  num_workers=hyperparams["num_workers"])
    forget_test_loader  = DataLoader(forget_test_ds,  batch_size=256,                       shuffle=False, num_workers=hyperparams["num_workers"])
    retain_test_loader  = DataLoader(retain_test_ds,  batch_size=256,                       shuffle=False, num_workers=hyperparams["num_workers"])

    model = create_model(hyperparams["backbone"], num_classes=10)
    ckpt  = torch.load("results/checkpoints/model_before_best.pth", weights_only=True, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparams["lr"],
        momentum=hyperparams["momentum"],
        weight_decay=hyperparams["weight_decay"],
    )

    print(f"\n{'='*60}\n\t=== Baseline Fine-Tuning ===\n{'='*60}\n")
    t0 = time.time()
    history = {"epoch": [], "train_loss": [], "retain_acc": [], "forget_acc": []}

    for ep in range(1, hyperparams["epochs"] + 1):
        model.train()
        running_loss, n = 0.0, 0

        for x, y in retain_train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            n += y.size(0)

        metrics = evaluate(model, {"retain": retain_test_loader, "forget": forget_test_loader}, device)
        metrics = {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in metrics.items()}

        history["epoch"].append(ep)
        history["train_loss"].append(running_loss / n)
        history["retain_acc"].append(metrics["retain_acc"])
        history["forget_acc"].append(metrics["forget_acc"])

        print(
            f"[FT] epoch {ep:02d}/{hyperparams['epochs']} "
            f"| loss {running_loss/n:.4f} "
            f"| retain_acc {metrics['retain_acc']:.4f} "
            f"| forget_acc {metrics['forget_acc']:.4f} "
            f"| elapsed {time.time()-t0:.1f}s"
        )

    torch.save(model.state_dict(), "results/checkpoints/baseline_ft.pth")
    print("\nSalvato: results/checkpoints/baseline_ft.pth")

    with open("results/reports/baseline_ft_curve.json", "w") as f:
        js.dump(history, f, indent=2)

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["retain_acc"], label="retain_acc")
    plt.plot(history["epoch"], history["forget_acc"], label="forget_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Baseline FT — Retain vs Forget Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/baseline_ft_acc.png", dpi=400, bbox_inches="tight")
    plt.close()
    print("Salvato: results/plots/baseline_ft_acc.png")


if __name__ == "__main__":
    baseline_ft()