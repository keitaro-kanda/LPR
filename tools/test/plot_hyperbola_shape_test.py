import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
t0 = np.arange(0, 101e-9, 50e-9)
er = np.arange(2, 4.5, 0.4)
c = 3e8

# calc hyperbola
t = np.zeros((len(t0), len(er), len(x)))
for i in range(len(t0)):
    for j in range(len(er)):
        t[i, j, :] = t0[i] * np.sqrt(1 + (x / (c * t0[i] / 2 / np.sqrt(er[j])))**2)

# plot
plt.figure(figsize=(8, 18))
colors = plt.cm.viridis(np.linspace(0, 1, len(t0) * len(er)))
color_idx = 0
for i in range(len(t0)):
    for j in range(len(er)):
        plt.plot(x, t[i, j] / 1e-9, label=f"t0={t0[i]} ns, er={er[j]}", color=colors[color_idx])
        color_idx += 1
plt.title("Hyperbola Shape")
plt.xlabel("x [m]")
plt.ylabel("t [ns]")
plt.ylim(0, 600)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.gca().invert_yaxis()
plt.tight_layout()  # レイアウトを自動調整
plt.show()
