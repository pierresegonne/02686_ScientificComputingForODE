import numpy as np
from scipy.integrate import ode
from solvers.utils import parse_one_param, parse_default_ode_params

def ode_solver(f, J, t0, tf, N, x0, **kwargs):

    parse_one_param(kwargs, 'adaptive_step_size', None)

    kwargs, atol, rtol, nsteps, dfactor = parse_default_ode_params(kwargs)

    r = ode(f, J).set_integrator('dopri5',
                                 atol=atol,
                                 rtol=rtol,
                                 nsteps=nsteps,
                                 dfactor=dfactor,
    )
    r.set_initial_value(x0, t0).set_f_params(*kwargs.values()).set_jac_params(*kwargs.values())

    if N < 5000:
        N = N * 100
    dt = (tf - t0)/N

    T = [t0]
    X = [x0]

    while r.successful() and r.t < tf:
        T.append(r.t + dt)
        X.append(r.integrate(r.t + dt))

    T = np.array(T)
    X = np.array(X)

    return X, T, {'dt': np.array([dt]), 'E': np.array([0])}
