import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression

from problems.test_equation import f, J
from solvers.explicit_euler import ode_solver

# Parameters
t0 = 0
tf = 10
Ns = np.linspace(100, 1000, 50, dtype=int)
dts = np.array([(tf-t0)/N for N in Ns])

# Test eq
lbd = -1
x0 = np.array([1])

e = []

for i, N in enumerate(Ns):
    X, T = ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, lbd=lbd)

    x_true = x0*np.exp(lbd*tf)
    ek = np.abs(X[-1] - x_true)
    e.append(ek)

e = np.array(e)

regressor = LinearRegression().fit(dts[:, None], e)
e_theoretical = regressor.predict(dts[:, None])

plt.plot(dts, e)
plt.plot(dts, e_theoretical)
plt.legend(['e', 'O(dt)'])
plt.yscale('log')
plt.xscale('log')
plt.show()