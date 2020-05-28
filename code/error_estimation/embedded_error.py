import matplotlib.pyplot as plt
import numpy as np
import os

from sys import path

path.append(os.path.realpath('..'))

from sklearn.linear_model import LinearRegression

from problems.test_equation import f, J

# To modify when changing solvers
from solvers.dopri54 import ode_solver
solver_colour = 'darkorange'
'''------------- Expected order -------------'''
p = 5
'''------------------------------------------'''
# -------

# Parameters
t0 = 0
tf = 10
Ns = np.geomspace(9000, tf*2, 10, dtype=int)
dts = np.array([(tf-t0)/N for N in Ns])

# Test eq
lbd = -1
x0 = np.array([1])

l = []
e = []
k = 10

for i, N in enumerate(Ns):
    X, T, controllers = ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, lbd=lbd)

    x_local = X[k-1]*np.exp(lbd*dts[i])
    lk = np.abs(X[k] - x_local) + 1e-22
    l.append(lk)
    e.append(controllers['e'][k])

l = np.array(l)
e = np.abs(np.array(e))

plt.rcParams.update({
    'axes.labelsize': 'x-large',
    'font.size': 13,
})

plt.figure()
plt.plot(dts, (dts**(p+1))/(dts[0]**(p+1)/l[0]), label=r'Theoretical $l_{k}$, $\mathcal{O}(h^{6})$', linestyle='dashed', linewidth=1.5, color='grey')
plt.plot(dts, l, label=r'True $l_{k}$', linewidth=2, color='black')
plt.plot(dts, (dts**(p))/(dts[0]**(p)/e[0]), label=r'Asymptotic Order for Embedded Error $e_{k}$, $\mathcal{O}(h^{4})$', linestyle='dashed', linewidth=1.5, color='grey')
plt.plot(dts, e, label=r'Estimated Error $e_{k}$', linewidth=3, color=solver_colour)
plt.legend()
plt.xlabel('h')
plt.ylabel('error')
plt.yscale('log')
plt.xscale('log')

plt.show()