import matplotlib.pyplot as plt
import numpy as np

x_dimension = 1


def f(t, X, lbd=1, **kwargs):
    return lbd * X


def J(t, X, lbd=1, **kwargs):
    return np.array([[lbd]])


'''--------------------- Plotting ---------------------'''


def plot_states(T, X, solvers, solver_options, with_true_solution=True, **kwargs):
    plt.figure()
    plt.rcParams.update({'axes.labelsize': 'x-large'})

    if with_true_solution:
        if ('lbd' not in kwargs.keys()) & ('x0' not in kwargs.keys()):
            raise AttributeError('To display the true solution, the parameters `lbd` and `x0` are required.')
        times = np.linspace(T[0][0], T[0][-1], 10000)
        plt.plot(times, kwargs['x0'] * np.exp(kwargs['lbd'] * times), label="True Solution", color='black')

    for i, solver in enumerate(solvers):
        plt.plot(T[i], X[i], label=f"{solver}", color=solver_options[solver]['color'])

    plt.legend()
