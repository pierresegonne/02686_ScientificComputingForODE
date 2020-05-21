import matplotlib.pyplot as plt
import numpy as np
import os

from sys import path

path.append(os.path.realpath('.'))

from sklearn.linear_model import LinearRegression

from problems.test_equation import f, J
from solvers.explicit_euler import ode_solver

# Parameters
t0 = 0
tf = 10
Ns = np.geomspace(1000000, tf*5, 300, dtype=int)
dts = np.array([(tf-t0)/N for N in Ns])

# Test eq
lbd = -1
x0 = np.array([1])

e = []

for i, N in enumerate(Ns):
    X, T, _ = ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, lbd=lbd)

    x_true = x0 * np.exp(lbd * tf)
    ek = np.abs(X[-1] - x_true)
    e.append(ek)

e = np.array(e)

regressor = LinearRegression().fit(np.log(dts[:, None]), np.log(e))
e_theoretical = regressor.predict(np.log(dts[:, None]))

plt.rcParams.update({
    'axes.labelsize': 'x-large',
    'font.size': 20,
})

plt.figure()
plt.plot(dts, e, label=r'True $e_{t_{f}}$', linewidth=2, color='olivedrab')
plt.plot(dts, np.exp(e_theoretical), label=r'Theoretical $e_{t_{f}}$, $\mathcal{O}(h)$', linewidth=2, color='grey')
plt.legend()
plt.xlabel('h')
plt.ylabel(r'$e_{t_{f}}$')
plt.yscale('log')
plt.xscale('log')

plt.show()
