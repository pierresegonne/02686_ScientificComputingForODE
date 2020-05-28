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
Ns = np.geomspace(2000, tf*2, 100, dtype=int)
dts = np.array([(tf-t0)/N for N in Ns])

# Test eq
lbd = -1
x0 = np.array([1])

l = []
k = 10

for i, N in enumerate(Ns):
    X, T, _ = ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, lbd=lbd)

    x_local = X[k-1]*np.exp(lbd*dts[i])
    lk = np.abs(X[k] - x_local) + 1e-18
    l.append(lk)

l = np.array(l)


regressor = LinearRegression().fit(np.log(dts[:, None]**(p+1)), np.log(l+1e-12))
l_theoretical = regressor.predict(np.log(dts[:, None]**(p+1)))

plt.rcParams.update({
    'axes.labelsize': 'x-large',
    'font.size': 20,
})

plt.figure()
plt.plot(dts, l, label=r'True $l_{k}$', linewidth=2, color=solver_colour)
plt.plot(dts, (dts**(p+1))/(dts[0]**(p+1)/l[0]), label=r'Theoretical $l_{k}$, $\mathcal{O}(h^{6})$', linewidth=2, color='grey')
plt.legend()
plt.xlabel('h')
plt.ylabel(r'$l_{k}$')
plt.yscale('log')
plt.xscale('log')

plt.show()