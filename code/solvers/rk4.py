import numpy as np

from solvers.utils import parse_adaptive_step_params, rk_step

# Butcher Tableau
C = np.array([0, 1 / 2, 1 / 2, 1])
A = np.array([
    [0, 0, 0, 0],
    [1 / 2, 0, 0, 0],
    [0, 1 / 2, 0, 0],
    [0, 0, 1, 0],
])
B = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])


def rk4_step(f, t, x, dt, **kwargs):
    butcher_tableau = {
        'A': A,
        'B': B,
        'C': C,
    }

    return rk_step(f, t, x, dt, butcher_tableau, **kwargs)[0]  # no error estimation.


def ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, **kwargs):
    dt = (tf - t0) / N

    T = [t0]
    X = [x0]

    if not adaptive_step_size:

        for k in range(N):
            X.append(rk4_step(f, T[-1], X[-1], dt, **kwargs))
            T.append(T[-1] + dt)

    if adaptive_step_size:

        kwargs, abstol, reltol, epstol, facmax, facmin = parse_adaptive_step_params(kwargs)

        t = t0
        x = x0

        while t < tf:
            if (t + dt > tf):
                dt = tf - t

            accept_step = False
            while not accept_step:
                # Take step of size dt
                x_1 = rk4_step(f, t, x, dt, **kwargs)

                # Take two steps of size dt/2
                x_hat_12 = rk4_step(f, t, x, dt / 2, **kwargs)
                t_hat_12 = t + (dt / 2)
                x_hat = rk4_step(f, t_hat_12, x_hat_12, dt / 2, **kwargs)

                # Error estimation
                e = np.abs(x_1 - x_hat)
                r = np.max(np.abs(e) / np.maximum(abstol, np.abs(x_hat) * reltol))

                accept_step = (r <= 1)
                if accept_step:
                    t = t + dt
                    x = x_hat

                    T.append(t)
                    X.append(x)

                dt = np.maximum(facmin, np.minimum(np.sqrt(epstol / r), facmax)) * dt

    T = np.array(T)
    X = np.array(X)

    return X, T
