import os
import json as js
from src.data.cifar10 import get_cifar10

os.makedirs('results/splits', exist_ok=True)

FORGET_CLASS = 3
SEED         = 42

forget_train = []
retain_train = []
forget_test  = []
retain_test  = []

train_ds = get_cifar10(train=True)
test_ds  = get_cifar10(train=False)

for idx, target in enumerate(train_ds.targets):
    if target == FORGET_CLASS: forget_train.append(idx)
    else: retain_train.append(idx)

for idx, target in enumerate(test_ds.targets):
    if target == FORGET_CLASS: forget_test.append(idx)
    else: retain_test.append(idx)

print(f"Forget Train size: {len(forget_train)}\nRetain Train size: {len(retain_train)}\nForget Test size: {len(forget_test)}\nRetain Test size: {len(retain_test)}")

split_data = {
    "forget_class": FORGET_CLASS,
    "seed": SEED,
    "split_info": {
        "forget_train": len(forget_train),
        "retain_train": len(retain_train),
        "forget_test": len(forget_test),
        "retain_test": len(retain_test)
    },
    "indices": {
        "forget_train": forget_train,
        "retain_train": retain_train,
        "forget_test": forget_test,
        "retain_test": retain_test
    }
}


split_path = f"results/splits/split_forget_{FORGET_CLASS}.json"

with open(split_path, "w") as f:
    js.dump(split_data, f, indent=2)

print(f"Split data saved to: {split_path}")