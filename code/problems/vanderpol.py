import matplotlib.pyplot as plt
import numpy as np

x_dimension = 2


def f(t, X, mu=1, **kwargs):
    return np.array([
        X[1],
        mu * (1 - (X[0] ** 2)) * X[1] - X[0]
    ])


def J(t, X, mu=1, **kwargs):
    return np.array([
        [0, 1],
        [-2 * mu * X[0] * X[1] - 1, mu * (1 - (X[0] ** 2))]
    ])


'''--------------------- Plotting ---------------------'''


def plot_states(T, X, solvers, solver_options):
    plt.rcParams.update({'axes.labelsize': 'x-large'})

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12,10))
    gs = axs[0, 0].get_gridspec()
    for ax in axs[0:, 0]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, 0])

    for i, solver in enumerate(solvers):
        # Phase plot
        axbig.plot(X[i][:, 0], X[i][:, 1], label=f"{solver}", color=solver_options[solver]['color'])
        axbig.set_ylabel(r'$x_{2}$')
        axbig.set_xlabel(r'$x_{1}$')
        # x1 vs time
        axs[0, 1].plot(T[i], X[i][:, 0], label=f"{solver}", color=solver_options[solver]['color'])
        axs[0, 1].set_ylabel(r'$x_{1}$')
        axs[1, 1].set_xlabel('t')
        # x2 vs time
        axs[1, 1].plot(T[i], X[i][:, 1], label=f"{solver}", color=solver_options[solver]['color'])
        axs[1, 1].set_ylabel(r'$x_{2}$')
        axs[1, 1].set_xlabel('t')


    plt.tight_layout()
    plt.legend()
