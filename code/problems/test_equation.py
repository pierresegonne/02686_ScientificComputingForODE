import numpy as np

x_dimension = 1


def f(t, X, lbd=1, **kwargs):
    return lbd * X


def J(t, X, lbd=1, **kwargs):
    return np.array([[lbd]])


'''--------------------- Plotting ---------------------'''

def plot_states(T, X, solvers, solver_options):
    pass
