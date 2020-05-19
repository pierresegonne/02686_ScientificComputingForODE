import numpy as np
from scipy.integrate import ode

def ode_solver(f, J, t0, tf, N, x0, **kwargs):

    if 'adaptive_step_size' in kwargs.keys():
        del kwargs['adaptive_step_size']

    r = ode(f, J).set_integrator('dopri5')
    r.set_initial_value(x0, t0).set_f_params(*kwargs.values()).set_jac_params(*kwargs.values())

    dt = (tf - t0)/N

    T = [t0]
    X = [x0]

    while r.successful() and r.t < tf:
        T.append(r.t + dt)
        X.append(r.integrate(r.t + dt))

    T = np.array(T)
    X = np.array(X)

    return X, T, {'dt': np.array([dt]), 'E': np.array([0])}
