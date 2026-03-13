import os
import json
import csv
import matplotlib.pyplot as plt


def compare_methods() -> None:
    """
    Reads the baseline and MASUC JSON reports and creates final summary Markdown, CSV, and PNG plots.
    Compares the 'Before', 'Baseline FT', 'Baseline Retrain', and 'MASUC' strategies.
    This fulfills Microstep 10.
    """
    os.makedirs("results/summaries", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    results = []

    results.append({
        "method": "Before Unlearning",
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

    masuc_path = "results/reports/masuc_curve.json"
    if os.path.exists(masuc_path):
        with open(masuc_path, "r") as f:
            data = json.load(f)
            results.append({
                "method": "MASUC (Ours)",
                "retain_acc": data["retain_acc"][-1],
                "forget_acc": data["forget_acc"][-1]
            })

    with open("results/summaries/final_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    with open("results/summaries/final_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "retain_acc", "forget_acc"])
        writer.writeheader()
        writer.writerows(results)

    with open("results/summaries/final_comparison.md", "w") as f:
        f.write("# Final Unlearning Methods Comparison\n\n")
        f.write("This table compares the original model ('Before') against two standard baselines and our proposed MASUC method.\n\n")
        f.write("| Method | Retain Accuracy | Forget Accuracy |\n")
        f.write("|:---|:---:|:---:|\n")
        for res in results:
            f.write(f"| **{res['method']}** | {res['retain_acc']:.2%} | {res['forget_acc']:.2%} |\n")
            
        f.write("\n\n### Objective\n")
        f.write("- **Retain Accuracy** should be as high as possible (close to or higher than 'Before' or 'Retrain').\n")
        f.write("- **Forget Accuracy** should be as low as possible (ideally close to 0%, meaning the model forgot the target class).\n")

    methods = [r["method"] for r in results]
    retain_accs = [r["retain_acc"] for r in results]
    forget_accs = [r["forget_acc"] for r in results]

    x = range(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar([i - width/2 for i in x], retain_accs, width, label='Retain Acc', color='teal')
    rects2 = ax.bar([i + width/2 for i in x], forget_accs, width, label='Forget Acc', color='crimson')

    ax.set_ylabel('Accuracy')
    ax.set_title('Socialized Unlearning (MASUC) vs Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    ax.grid(True, axis='y', alpha=0.3)
    
    for i, rect in enumerate(rects1):
        height = rect.get_height()
        ax.annotate(f'{height:.1%}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    for i, rect in enumerate(rects2):
        height = rect.get_height()
        ax.annotate(f'{height:.1%}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("results/plots/final_comparison_bar.png", dpi=400, bbox_inches="tight")
    plt.close()

    print("Creato: results/summaries/final_comparison.json")
    print("Creato: results/summaries/final_comparison.csv")
    print("Creato: results/summaries/final_comparison.md")
    print("Creato: results/plots/final_comparison_bar.png")


if __name__ == "__main__":
    compare_methods()
