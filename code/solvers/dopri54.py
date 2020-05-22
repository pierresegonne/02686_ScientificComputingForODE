import numpy as np

from solvers.utils import parse_adaptive_step_params, rk_step

# Butcher Tableau
C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
A = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
])
B = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100,
              1/40])
B_hat = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
E = np.array([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
              1/40])

"""
Might be typo on the figures verify with wikipedia:
	0
1/5	1/5
3/10	3/40	9/40
4/5	44/45	−56/15	32/9
8/9	19372/6561	−25360/2187	64448/6561	−212/729
1	9017/3168	−355/33	46732/5247	49/176	−5103/18656
1	35/384	0	500/1113	125/192	−2187/6784	11/84	
35/384	0	500/1113	125/192	−2187/6784	11/84	0
5179/57600	0	7571/16695	393/640	−92097/339200	187/2100	1/40
"""

# Order
P_DOPRI54 = 5


def dopri54_step(f, t, x, dt, **kwargs):

    butcher_tableau = {
        'A': A,
        'B': B,
        'C': C,
        'E': E,
    }

    return rk_step(f, t, x, dt, butcher_tableau, **kwargs)


def ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, **kwargs):

    dt = (tf - t0)/N

    T = [t0]
    X = [x0]
    controllers = {
        'r': [0.01],
        'dt': [dt],
    }

    if not adaptive_step_size:

        for k in range(N):
            x, _ = dopri54_step(f, T[-1], X[-1], dt, **kwargs)
            X.append(x)
            T.append(T[-1] + dt)

    if adaptive_step_size:

        kwargs, abstol, reltol, epstol, facmax, facmin = parse_adaptive_step_params(kwargs)
        p = P_DOPRI54
        k_p = 0.4/(p+1)
        k_i =0.3/(p+1)

        t = t0
        x = x0

        while t < tf:
            if (t + dt > tf):
                dt = tf - t

            accept_step = False
            while not accept_step:
                x_hat, e = dopri54_step(f, T[-1], X[-1], dt, **kwargs)
                r = np.max(np.abs(e) / np.maximum(abstol, np.abs(x_hat)*reltol))

                accept_step = (r <= 1)
                if accept_step:
                    t = t + dt
                    x = x_hat
                    dt = dt * (epstol/r)**(k_i) * (controllers['r'][-1]/r)**(k_p)

                    T.append(t)
                    X.append(x)
                else:
                    dt = dt * (epstol / r)**(1/(p+1))
                controllers['dt'].append(dt)
                controllers['r'].append(r)


    T = np.array(T)
    X = np.array(X)
    controllers['dt'] = np.array(controllers['dt'])
    controllers['r'] = np.array(controllers['r'])

    return X, T, controllers
