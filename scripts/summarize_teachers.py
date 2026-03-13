import csv
import json as js
import matplotlib.pyplot as plt


reports = []
for teacher_id in range(5):
    with open(f"results/reports/teacher_{teacher_id}.json", "r") as f:
        reports.append(js.load(f))

header = ["teacher_id", "blind_class", "seen_acc_final", "blind_acc_final", "elapsed_sec"]

with open("results/reports/teachers_summary.csv", "w", newline="") as f:
    wrt = csv.writer(f)
    wrt.writerow(header)
    for r in reports:
        wrt.writerow([
            r["teacher_id"],
            r["blind_class"],
            r["metrics_final"]["seen_acc"],
            r["metrics_final"]["blind_acc"],
            r["elapsed_sec"],
        ])


with open("results/reports/teachers_summary.json", "w") as f:
    js.dump(reports, f, indent=2)

with open("results/reports/teacher_summary.md", "w") as f:
    f.write("| " + " | ".join(header) + " |\n")
    f.write("| " + " | ".join(["---"] * len(header)) + " |\n")
    for r in reports:
        f.write(f"| {r['teacher_id']} | {r['blind_class']} | {r['metrics_final']['seen_acc']:.4f} | {r['metrics_final']['blind_acc']:.4f} | {r['elapsed_sec']:.1f} |\n")

tids = [r["teacher_id"] for r in reports]
seen = [r["metrics_final"]["seen_acc"] for r in reports]
blind = [r["metrics_final"]["blind_acc"] for r in reports]

x = range(len(tids))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar([i - width/2 for i in x], seen, width, label="seen_acc")
plt.bar([i + width/2 for i in x], blind, width, label="blind_acc")
plt.xlabel("Teacher ID")
plt.ylabel("Accuracy")
plt.title("Teacher Society — Seen vs Blind Accuracy")
plt.xticks(list(x), [f"T{t}\n(blind={r['blind_class']})" for t, r in zip(tids, reports)])
plt.legend()
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("results/plots/teachers_seen_vs_blind.png", dpi=400, bbox_inches="tight")
plt.close()
