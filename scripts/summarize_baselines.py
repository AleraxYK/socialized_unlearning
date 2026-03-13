import os
import json
import csv
import matplotlib.pyplot as plt


def summarize_baselines() -> None:
    """
    Reads the baseline JSON reports and creates summary Markdown, CSV, and PNG plots.
    Compares the 'before', 'baseline_ft', and 'baseline_retrain' strategies.
    """
    os.makedirs("results/summaries", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    results = []

    results.append({
        "method": "Before",
        "retain_acc": 0.8279,
        "forget_acc": 0.8200
    })

    ft_path = "results/reports/baseline_ft_curve.json"
    if os.path.exists(ft_path):
        with open(ft_path, "r") as f:
            data = json.load(f)
            results.append({
                "method": "Baseline FT",
                "retain_acc": data["retain_acc"][-1],
                "forget_acc": data["forget_acc"][-1]
            })

    retrain_path = "results/reports/baseline_retrain_curve.json"
    if os.path.exists(retrain_path):
        with open(retrain_path, "r") as f:
            data = json.load(f)
            results.append({
                "method": "Baseline Retrain",
                "retain_acc": data["retain_acc"][-1],
                "forget_acc": data["forget_acc"][-1]
            })

    with open("results/summaries/baselines_summary.json", "w") as f:
        json.dump(results, f, indent=4)


    with open("results/summaries/baselines_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "retain_acc", "forget_acc"])
        writer.writeheader()
        writer.writerows(results)


    with open("results/summaries/baselines_summary.md", "w") as f:
        f.write("# Baselines Comparison\n\n")
        f.write("| Method | Retain Accuracy | Forget Accuracy |\n")
        f.write("|---|---|---|\n")
        for res in results:
            f.write(f"| {res['method']} | {res['retain_acc']:.2%} | {res['forget_acc']:.2%} |\n")


    methods = [r["method"] for r in results]
    retain_accs = [r["retain_acc"] for r in results]
    forget_accs = [r["forget_acc"] for r in results]

    x = range(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar([i - width/2 for i in x], retain_accs, width, label='Retain Acc')
    rects2 = ax.bar([i + width/2 for i in x], forget_accs, width, label='Forget Acc')

    ax.set_ylabel('Accuracy')
    ax.set_title('Baseline Unlearning Methods Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/baselines_comparison.png", dpi=400, bbox_inches="tight")
    plt.close()

    print("Creato: results/summaries/baselines_summary.json")
    print("Creato: results/summaries/baselines_summary.csv")
    print("Creato: results/summaries/baselines_summary.md")
    print("Creato: results/plots/baselines_comparison.png")


if __name__ == "__main__":
    summarize_baselines()
