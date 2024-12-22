import matplotlib.pyplot as plt

t0 = []
t1 = []
s  = []

with open("outUnlearning.txt", "r") as f:

    for line in f:
        if "TEACHER 0 Validation Loss:" in line: t0.append(float(line.split(" ")[-1]))
        elif "TEACHER 1 Validation Loss:" in line: t1.append(float(line.split(" ")[-1]))
        elif "STUDENT Validation Loss:" in line: s.append(float(line.split(" ")[-1]))

print(len(t0),len(t1), len(s))

plt.figure(figsize=(15, 5))
plt.plot(range(1,51), t0, label="T0")
plt.plot(range(1,51), t1, label="T1")
plt.plot(range(1,51), s, label="S")


plt.legend()
plt.title("Learning")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()