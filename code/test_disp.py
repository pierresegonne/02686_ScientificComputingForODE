import matplotlib.pyplot as plt
import numpy as np


def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 3.0, 0.01)

fig, axs = plt.subplots(ncols=2, nrows=2)
gs = axs[0, 0].get_gridspec()
for ax in axs[0:, 0]:
    ax.remove()
axbig = fig.add_subplot(gs[0:, 0])
axbig.plot(t1, f(t1))
axs[0, 1].plot(t1, f(t1))
axs[1, 1].plot(t1, f(t1))

fig.tight_layout()
plt.show()
