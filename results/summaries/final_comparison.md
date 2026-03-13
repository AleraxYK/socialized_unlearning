# Final Unlearning Methods Comparison

This table compares the original model ('Before') against two standard baselines and our proposed MASUC method.

| Method | Retain Accuracy | Forget Accuracy |
|:---|:---:|:---:|
| **Before Unlearning** | 82.79% | 82.00% |
| **Baseline FT** | 94.13% | 4.40% |
| **Baseline Retrain** | 86.20% | 0.00% |
| **MASUC (Ours)** | 83.31% | 9.50% |


### Objective
- **Retain Accuracy** should be as high as possible (close to or higher than 'Before' or 'Retrain').
- **Forget Accuracy** should be as low as possible (ideally close to 0%, meaning the model forgot the target class).
