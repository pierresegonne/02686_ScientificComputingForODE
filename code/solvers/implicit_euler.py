import numpy as np

from scipy.linalg import norm

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

    # TODO parse kwargs
    tol = 1e-8
    max_iters = 100

    if not adaptive_step_size:

        for k in range(N):

            # Use explicit form to start off newton
            f_eval = f(T[-1], X[-1], **kwargs)
            x_init = X[-1] + dt*f_eval
            X.append(newtons_method(f, J, T[-1], X[-1], dt, x_init, tol, max_iters, **kwargs))
            T.append(T[-1] + dt)

    if adaptive_step_size:
        pass

    T = np.array(T)
    X = np.array(X)

    return X, T