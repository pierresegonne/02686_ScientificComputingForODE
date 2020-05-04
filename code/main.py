import matplotlib.pyplot as plt
import numpy as np

from problems.vanderpol import f, J

from solvers.default import ode_solver as default_ode_solver
from solvers.dopri54 import ode_solver as dopri54_ode_solver
# from solvers.esdirk23 import ode_solver
from solvers.explicit_euler import ode_solver as explicit_euler_ode_solver
from solvers.implicit_euler import ode_solver as implicit_euler_ode_solver
# from solvers.own_rk import ode_solver
from solvers.rk4 import ode_solver as rk4_ode_solver


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

# List of solvers.
solvers = ['default', 'dopri54']

plt.figure()
for solver in solvers:
    if solver == 'default':
        ode_solver = default_ode_solver
    elif solver == 'dopri54':
        ode_solver = dopri54_ode_solver
    elif solver == 'esdirk23':
        raise NotImplementedError
    elif solver == 'explicit_euler':
        ode_solver = explicit_euler_ode_solver
    elif solver == 'implicit_euler':
        ode_solver = implicit_euler_ode_solver
    elif solver == 'own_rk':
        raise NotImplementedError
    elif solver == 'rk4':
        ode_solver = rk4_ode_solver
    else:
        raise NameError("Incorrect solver name")

    print(f"Running solver {solver}")

    X, T = ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=True, mu=mu)

    plt.plot(X[:, 0], X[:, 1], label=f"{solver}")

plt.legend()
plt.show()