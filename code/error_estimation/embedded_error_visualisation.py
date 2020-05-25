import matplotlib.pyplot as plt
import numpy as np
import os

from sys import path

path.append(os.path.realpath('..'))

from problems.test_equation import f, J

# To modify when changing solvers
from solvers.own_rk import ode_solver
solver_colour = 'pink'
# -------


# Parameters
t0 = 0
tf = 10
N = 40
dt = (tf - t0) / N

ticks_for_true = np.linspace(t0, tf, 1000)

# Test eq
lbd = -1
x0 = np.array([1])

l = [0]

X, T, controllers = ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, lbd=lbd)
for i in range(1, len(T)):
    x_local = X[i - 1] * np.exp(lbd * dt)
    l.append(X[i] - x_local)

plt.rcParams.update({
    'axes.labelsize': 'x-large',
    'font.size': 20,
})

plt.figure()
plt.plot(T, X, color=solver_colour, label='Approximation')
plt.plot(ticks_for_true, x0 * np.exp(lbd * ticks_for_true), color='black', label='True Solution')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()

plt.figure()
plt.plot(T, np.abs(controllers['e']), '-o', color=solver_colour, label='Estimated Error')
plt.plot(T, np.abs(l), '-D', color='black', label='True Local Error')
plt.xlabel('t')
plt.ylabel(r'|$l_{k}$|')
plt.legend()

plt.show()
