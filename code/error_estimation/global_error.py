import matplotlib.pyplot as plt
import numpy as np
import os

from sys import path

path.append(os.path.realpath('..'))

from sklearn.linear_model import LinearRegression

from problems.test_equation import f, J
from solvers.dopri54 import ode_solver
solver_colour = 'darkorange'
'''------------- Expected order -------------'''
p = 5
'''------------------------------------------'''

# Parameters
t0 = 0
tf = 10
Ns = np.geomspace(10000, tf*2, 100, dtype=int)
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

regressor = LinearRegression().fit(p*np.log(dts[:, None]), np.log(e))
e_theoretical = regressor.predict(p*np.log(dts[:, None]))

plt.rcParams.update({
    'axes.labelsize': 'x-large',
    'font.size': 20,
})

plt.figure()
plt.plot(dts, e, label=r'True $e_{t_{f}}$', linewidth=3, color=solver_colour)
plt.plot(dts, (dts**p)/(dts[0]**p/e[0]), label=r'Theoretical $e_{t_{f}}$, $\mathcal{O}(h^{5})$', linewidth=2, color='grey')
plt.legend()
plt.xlabel('h')
plt.ylabel(r'$e_{t_{f}}$')
plt.yscale('log')
plt.xscale('log')

plt.show()
