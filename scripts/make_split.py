import os
import argparse
import json as js

from src.data import get_dataset


def make_split():
    """
    Partition a dataset into forget and retain splits and save the indices to a JSON file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "mnist", "tinyimagenet"])
    parser.add_argument("--forget_class", type=int, default=3)
    args = parser.parse_args()

    ds           = args.dataset
    forget_class = args.forget_class
    SEED         = 42

    os.makedirs("results/splits", exist_ok=True)

    train_ds = get_dataset(ds, "./data", train=True)
    test_ds  = get_dataset(ds, "./data", train=False)

    forget_train, retain_train = [], []
    forget_test,  retain_test  = [], []

    for idx, target in enumerate(train_ds.targets):
        if target == forget_class: forget_train.append(idx)
        else: retain_train.append(idx)

    for idx, target in enumerate(test_ds.targets):
        if target == forget_class: forget_test.append(idx)
        else: retain_test.append(idx)

    print(f"Dataset: {ds.upper()} | Forget class: {forget_class}")
    print(f"Forget Train: {len(forget_train)} | Retain Train: {len(retain_train)}")
    print(f"Forget Test:  {len(forget_test)}  | Retain Test:  {len(retain_test)}")

    split_data = {
        "dataset":      ds,
        "forget_class": forget_class,
        "seed":         SEED,
        "split_info": {
            "forget_train": len(forget_train),
            "retain_train": len(retain_train),
            "forget_test":  len(forget_test),
            "retain_test":  len(retain_test),
        },
        "indices": {
            "forget_train": forget_train,
            "retain_train": retain_train,
            "forget_test":  forget_test,
            "retain_test":  retain_test,
        },
    }

    split_path = f"results/splits/{ds}_split_forget_{forget_class}.json"
    with open(split_path, "w") as f:
        js.dump(split_data, f, indent=2)

    print(f"Split saved to: {split_path}")


if __name__ == "__main__":
    make_split()