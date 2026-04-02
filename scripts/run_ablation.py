import os
import argparse


def run_ablation():
    """
    Orchestrator that runs MASUC multiple times, disabling one component at a time,
    to produce a complete Ablation Study on the chosen dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--forget_class", type=int, default=3)
    args = parser.parse_args()

    ds = args.dataset
    fc = args.forget_class

    runs = [
        ("Full MASUC",                  ""),
        ("No Energy Alignment",         "--no_ea"),
        ("No Knowledge Distillation",   "--no_kd"),
        ("No Reciprocal Altruism",      "--no_ra"),
        ("No Erasure Loss",             "--no_erasure"),
    ]

    print(f"\n{'='*60}")
    print(f"  Ablation Study | Dataset: {ds.upper()} | Forget class: {fc}")
    print(f"{'='*60}\n")

    for name, flag in runs:
        cmd = f"python -m scripts.run_masuc --dataset {ds} --forget_class {fc} {flag}".strip()
        print(f"\n>>> [{name}]\n>>> {cmd}\n")
        code = os.system(cmd)
        if code != 0:
            print(f"\n[!] Run '{name}' failed (exit code {code}). Aborting.")
            break

    print(f"\n{'='*60}")
    print(f"  Done! Run: python -m scripts.compare_ablation --dataset {ds}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_ablation()
