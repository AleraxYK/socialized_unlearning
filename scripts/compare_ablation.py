import os
import json
import argparse
import matplotlib.pyplot as plt


def compare_ablation():
    """
    Reads MASUC ablation results and generates a comparative bar plot.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    args = parser.parse_args()
    ds = args.dataset

    os.makedirs("results/plots", exist_ok=True)

    ablations = [
        {"suffix": "",            "label": "Full MASUC"},
        {"suffix": "_no_ea",      "label": "w/o Energy Alignment"},
        {"suffix": "_no_kd",      "label": "w/o Knowledge Distillation"},
        {"suffix": "_no_ra",      "label": "w/o Reciprocal Altruism"},
        {"suffix": "_no_erasure", "label": "w/o Erasure Loss"},
    ]

    results = []
    for ab in ablations:
        path = f"results/reports/{ds}_masuc{ab['suffix']}_curve.json"
        if not os.path.exists(path):
            print(f"[!] Missing: {path} — skipping.")
            continue
        with open(path) as f:
            data = json.load(f)
        results.append({
            "label":      ab["label"],
            "retain_acc": data["retain_acc"][-1],
            "forget_acc": data["forget_acc"][-1],
        })

    if not results:
        print("No ablation results found. Did you run scripts.run_ablation first?")
        return

    labels      = [r["label"]      for r in results]
    retain_accs = [r["retain_acc"] for r in results]
    forget_accs = [r["forget_acc"] for r in results]
    x     = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar([i - width/2 for i in x], retain_accs, width, label="Retain Acc", color="teal")
    bars2 = ax.bar([i + width/2 for i in x], forget_accs, width, label="Forget Acc", color="crimson")

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.1%}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Accuracy")
    ax.set_title(f"MASUC Ablation Study — {ds.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = f"results/plots/{ds}_ablation_bar.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    compare_ablation()
