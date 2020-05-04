import numpy as np

from scipy.linalg import norm
from solvers.utils import parse_adaptive_step_params, parse_newtons_params

def newtons_method(f, J, t, x, dt, x_init, tol, max_iters, **kwargs):
    k = 0
    x_iter = x_init
    t_iter = t + dt
    f_eval = f(t_iter, x_iter, **kwargs)
    J_eval = J(t_iter, x_iter, **kwargs)

    R = x_iter - dt*f_eval - x
    I = np.eye(x.shape[0])
    while ((k < max_iters) & (norm(R, np.inf) > tol)):
        k += 1
        M = I - dt*J_eval
        dx_iter = np.linalg.solve(M,R)
        x_iter -= dx_iter
        f_eval = f(t_iter, x_iter, **kwargs)
        J_eval = J(t_iter, x_iter, **kwargs)
        R = x_iter - dt*f_eval - x

    return x_iter

def ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, **kwargs):

    dt = (tf - t0)/N

    T = [t0]
    X = [x0]

    kwargs, newtons_tol, newtons_max_iters = parse_newtons_params(kwargs)

    if not adaptive_step_size:

        for k in range(N):

            # Use explicit form to start off newton
            f_eval = f(T[-1], X[-1], **kwargs)
            x_init = X[-1] + dt*f_eval
            X.append(newtons_method(f, J, T[-1], X[-1], dt, x_init, newtons_tol, newtons_max_iters, **kwargs))
            T.append(T[-1] + dt)

    if adaptive_step_size:

        kwargs, abstol, reltol, epstol, facmax, facmin = parse_adaptive_step_params(kwargs)

        t = t0
        x = x0

        while t < tf:
            if (t + dt > tf):
                dt = tf - t

            f_eval = f(t, x, **kwargs)
            accept_step = False
            while not accept_step:
                # Take initial guess step of size dt
                x_1_init = x + dt*f_eval
                x_1 = newtons_method(f, J, t+dt, x, dt, x_1_init, newtons_tol, newtons_max_iters, **kwargs)

                # Take two steps of size dt/2
                x_hat_12_init = x + (dt/2)*f_eval
                t_hat_12 = t + (dt/2)
                x_hat_12 = newtons_method(f, J, t_hat_12, x, dt/2, x_hat_12_init, newtons_tol, newtons_max_iters, **kwargs)

                f_eval_12 = f(t_hat_12, x_hat_12, **kwargs)
                x_hat_init = x_hat_12 + (dt/2)*f_eval_12
                x_hat = newtons_method(f, J, t+dt, x_hat_12, dt/2, x_hat_init, newtons_tol, newtons_max_iters, **kwargs)

                # Error estimation
                e = np.abs(x_1 - x_hat)
                r = np.max(np.abs(e) / np.maximum(abstol, np.abs(x_hat)*reltol))

                accept_step = (r <= 1)
                if accept_step:
                    t = t + dt
                    x = x_hat

                    T.append(t)
                    X.append(x)

                dt = np.maximum(facmin, np.minimum(np.sqrt(epstol/r), facmax)) * dt

    T = np.array(T)
    X = np.array(X)

    return X, T