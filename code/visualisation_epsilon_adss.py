import matplotlib.pyplot as plt
import numpy as np

eps = np.array([0.01, 0.05, 0.1, 0.5, 0.8])
e = np.array([0.00014935, 0.00033389, 0.00047215, 0.00105526, 0.00133444])
n_x = np.array([15002, 6711, 4746, 2124, 1680])
n_dt = np.array([15006, 6715, 4750, 2128, 1684])

plt.rcParams.update({
    'axes.labelsize': 'x-large',
    'font.size': 20,
})

plt.figure()
plt.plot(eps, e, '-o', color='black')
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$e_{N}$')

plt.figure()
plt.plot(eps, n_x, '-o', color='grey', label='Number of steps used')
plt.xlabel(r'$\epsilon$')
plt.ylabel('Number of steps')

plt.show()

