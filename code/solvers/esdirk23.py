import numpy as np
import scipy as sp

from solvers.utils import parse_adaptive_step_params, parse_adaptive_step_params_esdirk, parse_inexact_newtons_params, \
    parse_newtons_params

# ESDIRK23
# Butcher Tableau
gamma = (2 - np.sqrt(2)) / 2
C = np.array([0, 2 * gamma, 1])
A = np.array([
    [0, 0, 0],
    [gamma, gamma, 0],
    [(1 - gamma) / 2, (1 - gamma) / 2, gamma]
])
B = np.array([(1 - gamma) / 2, (1 - gamma) / 2, gamma])
B_hat = np.array(
    [(6 * gamma - 1) / (12 * gamma), 1 / (12 * gamma * (1 - 2 * gamma)), (1 - 3 * gamma) / (3 * (1 - 2 * gamma))])
E = B - B_hat

P_ESDIRK23 = 3


def ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=False, **kwargs):

    dt = (tf - t0) / N
    t = t0
    x = x0

    T = [t0]
    X = [x0]
    controllers = {
        'dt': [dt],
        'r': [0],
    }

    # Parameters needed
    kwargs, newtons_tol, newtons_max_iters = parse_newtons_params(kwargs)
    kwargs, abstol, reltol, epstol, facmax, facmin = parse_adaptive_step_params(kwargs)
    kwargs, max_diverged_steps, newtons_tau = parse_inexact_newtons_params(kwargs, epstol)
    kwargs, hmax, hmin = parse_adaptive_step_params_esdirk(kwargs)

    p = P_ESDIRK23
    D = x0.shape[0]
    I = np.eye(D)
    T_stages = np.zeros((p,))
    X_stages = np.zeros((p, D))
    F_stages = np.zeros((p, D))

    F_stages[-1, :] = f(t, x, **kwargs)

    n_steps = 1
    n_diverged_steps = 0

    while (t < tf) & (n_diverged_steps < max_diverged_steps):

        if (t + dt > tf):
            dt = tf - t

        J_eval = J(t, x, **kwargs)
        M = I - dt * gamma * J_eval
        L, U = sp.linalg.lu(M, permute_l=True)

        # Stage 0
        T_stages[0] = t
        X_stages[0, :] = x
        F_stages[0, :] = F_stages[-1, :]

        i = 1
        step_diverged = False
        while (i < p) & (not step_diverged):

            # Inexact Newton's method
            T_stages[i] = t + dt * C[i]
            X_stages[i, :] = x + dt * C[i] * F_stages[i, :]  # Initial EE guess
            sum_i = x + dt * A[i, :i] @ F_stages[:i, :]  # i not included in :i while it is in matlab
            F_stages[i, :] = f(T_stages[i], X_stages[i, :], **kwargs)

            R = X_stages[i, :] - dt * gamma * F_stages[i, :] - sum_i

            # Control convergence of inexact Newton's method
            r_newton = np.amax(np.abs(R) / np.maximum(abstol, np.abs(X_stages[i, :]) * reltol))

            n_iter_in_newton = 0
            r_newton_prev = r_newton
            while (r_newton > newtons_tau) & (not step_diverged):
                # Next iteration
                dX = np.linalg.solve(U, np.linalg.solve(L, -R))
                X_stages[i, :] = X_stages[i, :] + dX

                # Convergence
                F_stages[i, :] = f(T_stages[i], X_stages[i, :], **kwargs)
                R = X_stages[i, :] - (dt * gamma * F_stages[i, :]) - sum_i
                r_newton = np.amax(np.abs(R) / np.maximum(abstol, np.abs(X_stages[i, :]) * reltol))
                n_iter_in_newton += 1

                # Convergence rate
                alpha = r_newton / r_newton_prev
                newton_step_diverged = (alpha >= 1)

                # Nbr iterations
                reached_max_newton_iterations = (n_iter_in_newton >= newtons_max_iters)

                # Composed convergence
                step_diverged = newton_step_diverged | reached_max_newton_iterations

            # Next stage
            i += 1

        # Error estimation
        e = dt * E @ F_stages
        r_step_control = np.amax(
            np.abs(e) / np.maximum(abstol, np.abs(X_stages[-1, :]) * reltol))

        if adaptive_step_size:
            if step_diverged:
                accept_step = False
                dt = dt / 2
            else:
                accept_step = (r_step_control <= 1)

                if accept_step:
                    if n_steps == 1:
                        dt_modified = (epstol / r_step_control) ** (
                                    1 / (p + 1)) * dt  # First step -> asymptotic controller
                    else:
                        dt_modified = (dt / controllers['dt'][-1]) \
                                      * ((epstol / r_step_control) ** (1 / (p + 1))) \
                                      * ((controllers['r'][-1] / r_step_control) ** (1 / (p + 1))) \
                                      * dt  # PI controller

                    dt = np.minimum(np.maximum(dt_modified, dt * hmin), dt * hmax)
                else:
                    dt = (epstol / r_step_control) ** (1 / (p + 1)) * dt  # Rejected step -> asymptotic controller

        if not adaptive_step_size:
            # Fixed step size controller
            if step_diverged:
                accept_step = False
                dt = dt / 2
            else:
                accept_step = True
                dt = dt

        controllers['dt'].append(dt)
        controllers['r'].append(r_step_control)

        if accept_step:

            t = T_stages[-1]
            x = np.copy(X_stages[-1, :])

            T.append(t)
            X.append(x)

            n_steps += 1
            n_diverged_steps = 0

        else:
            n_diverged_steps += 1

    if n_diverged_steps == max_diverged_steps:
        print(f"/!\\ Max diverged steps {max_diverged_steps} reached, the solver stopped.")

    T = np.array(T)
    X = np.array(X).reshape((-1, x0.shape[0]))
    controllers['dt'] = np.array(controllers['dt'])
    controllers['r'] = np.array(controllers['r'])

    return X, T, controllers
