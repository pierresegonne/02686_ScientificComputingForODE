import numpy as np

def f(t, X, mu=1):
    return np.array([
            X[1],
            mu*(1-(X[0]**2))*X[1] - X[0]
        ])

def J(t, X, mu=1):
    return np.array([
        [0, 1],
        [-2*mu*X[0]*X[1]-1, mu*(1-(X[0]**2))]
        ])