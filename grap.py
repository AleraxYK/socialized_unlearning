import matplotlib.pyplot as plt

out = []
aloss = []
t0 = []
t1 = []
with open("output.txt", "r") as f:

    for line in f:
        if "STUDENT Validation Loss:" in line:
            out.append(float(line.split(" ")[-1]))
        elif "Epoch [" in line and "50], Average NON TARGET Loss:" in line:
            aloss.append(float(line.split(" ")[-1]))
        elif "TEACHER 0 Validation Loss:" in line:
            t0.append(float(line.split(" ")[-1]))
        elif "TEACHER 1 Validation Loss:" in line:
            t1.append(float(line.split(" ")[-1]))

print("out:", len(out))
print("aloss:", len(aloss))
print("t0:", len(t0))
print("t1:", len(t1))

plt.figure(figsize=(15, 5))
plt.plot(range(1,51), out, label="Validation")
plt.plot(range(1,51), aloss, label="Training")
plt.plot(range(1,51), t0, label="Teacher 0")
plt.plot(range(1,51), t1, label="Teacher 1")

plt.legend()
plt.title("Unlearning")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()