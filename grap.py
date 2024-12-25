import matplotlib.pyplot as plt

t0 = []
t1 = []
s  = []

t0MAX = 0
t1MAX = 0
sMAX  = 0

with open("outUnlearning.txt", "r") as f:

    for line in f:
        if "Teacher 0 unlearning Epoch [" in line: t0.append(float(line.split(" ")[-1]))
        elif "Teacher 1 unlearning Epoch [" in line: t1.append(float(line.split(" ")[-1]))
        elif "Student unlearning Epoch [" in line: s.append(float(line.split(" ")[-1]))


t0MAX = max(t0)
t1MAX = max(t1)
sMAX  = max(s)


for i in range(0,50,1):
    t0[i] = sMAX - t0[i]
    t1[i] = sMAX - t1[i]
    s[i]  = sMAX - s[i] 

plt.figure(figsize=(15, 5))
plt.plot(range(1,51), t0, label="T0")
plt.plot(range(1,51), t1, label="T1")
plt.plot(range(1,51), s, label="S")


plt.legend()
plt.title("Unlearning AL")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid()
plt.show()