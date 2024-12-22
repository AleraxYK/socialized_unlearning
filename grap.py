import matplotlib.pyplot as plt

t0 = []
t1 = []
t2 = []
t3 = []
t4 = []

with open("outLearning.txt", "r") as f:

    for line in f:
        if "Teacher 0 learning Validation Loss:" in line: t0.append(float(line.split(" ")[-1]))
        elif "Teacher 1 learning Validation Loss:" in line: t1.append(float(line.split(" ")[-1]))
        elif "Teacher 2 learning Validation Loss:" in line: t2.append(float(line.split(" ")[-1]))
        elif "Teacher 3 learning Validation Loss:" in line: t3.append(float(line.split(" ")[-1]))
        elif "Teacher 4 learning Validation Loss:" in line: t4.append(float(line.split(" ")[-1]))

print(len(t0),len(t1),len(t2),len(t3),len(t4))
plt.figure(figsize=(15, 5))
plt.plot(range(1,51), t0, label="T0")
plt.plot(range(1,51), t1, label="T1")
plt.plot(range(1,51), t2, label="T2")
plt.plot(range(1,51), t3, label="T3")
plt.plot(range(1,51), t4, label="T4")

plt.legend()
plt.title("Learning")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()