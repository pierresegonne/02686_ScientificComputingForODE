import matplotlib.pyplot as plt
import numpy as np

from problems.vanderpol import f, J
from solvers.implicit_euler import ode_solver

# Parameters
t0 = 0;
tf = 100;
N = 5000;

# Specific to problem
mu = 12
x0 = np.array([0.5, 0.5])

# Test eq
lbd = -1
#x0 = np.array([1])


X, T = ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, mu=mu)

plt.plot(X[:, 0], X[:, 1])
plt.show()