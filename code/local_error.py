import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from problems.test_equation import f, J
from solvers.explicit_euler import ode_solver

# Parameters
t0 = 0
tf = 100
Ns = np.linspace(tf*10, 100000, 100, dtype=int)
dts = np.array([(tf-t0)/N for N in Ns])

# Test eq
lbd = -1
x0 = np.array([1])

l = []
k = 1

for i, N in enumerate(Ns):
    X, T = ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, lbd=lbd)

    x_local = X[k-1]*np.exp(lbd*dts[i])
    lk = np.abs(X[k] - x_local)
    l.append(lk)

l = np.array(l)

# dts_2 = PolynomialFeatures(2).fit_transform(dts[:, None])
regressor = LinearRegression().fit(dts[:, None]**2, l)
l_theoretical = regressor.predict(dts[:, None]**2)

plt.plot(dts, l)
plt.plot(dts, l_theoretical)
plt.legend(['l', r'$O(dt^{2})$'])
# plt.yscale('log')
# plt.xscale('log')
plt.show()