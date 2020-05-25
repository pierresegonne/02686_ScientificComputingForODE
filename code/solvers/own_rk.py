import numpy as np

from solvers.utils import parse_adaptive_step_params, rk_step

# Butcher Tableau
C = np.array([0, 1 / 3, 2 / 3])
A = np.array([
    [0, 0, 0],
    [1 / 3, 0, 0],
    [0, 2 / 3, 0],
])
B = np.array([1 / 4, 0, 3 / 4])
B_hat = np.array([-1 / 2, 3 / 2, 0])
E = B - B_hat

# Order
P_OWNRK = 3

def own_rk_step(f, t, x, dt, **kwargs):
    butcher_tableau = {
        'A': A,
        'B': B,
        'C': C,
        'E': E,
    }

    return rk_step(f, t, x, dt, butcher_tableau, **kwargs)


def ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, **kwargs):
    dt = (tf - t0) / N

    T = [t0]
    X = [x0]
    controllers = {
        'e': [0],
        'dt': [dt]
    }

    if not adaptive_step_size:

        for k in range(N):
            x, e = own_rk_step(f, T[-1], X[-1], dt, **kwargs)
            X.append(x)
            T.append(T[-1] + dt)
            controllers['e'].append(e)

    T = np.array(T)
    X = np.array(X)
    controllers['dt'] = np.array(controllers['dt'])
    controllers['e'] = np.array(controllers['e'])

    return X, T, controllers
