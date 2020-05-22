import numpy as np


def sde_solver(f, J, g, dW, t0, tf, N, x0, **kwargs):
    dt = (tf - t0) / N

    T = [t0]
    X = [x0]

    for k in range(N):
        f_eval = f(T[-1], X[-1], **kwargs)
        g_eval = g(T[-1], X[-1], **kwargs)
        X.append(X[-1] + dt * f_eval + dW[k] * g_eval)
        T.append(T[-1] + dt)

    T = np.array(T)
    X = np.array(X)

    return X, T
