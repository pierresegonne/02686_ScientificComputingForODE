import numpy as np

def f(t, X, lbd=1):
    return lbd*X

def J(t, X, lbd=1):
    return np.array([[0, lbd]])