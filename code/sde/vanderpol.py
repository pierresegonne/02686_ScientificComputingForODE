import matplotlib.pyplot as plt
import numpy as np

from utils import get_rcolor

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


def diffusion_1(t, X, sigma=1, **kwargs):
    return np.array([0., sigma])


def diffusion_2(t, X, sigma=1, **kwargs):
    return np.array([0., sigma * (1 + X[0] * X[0])])


'''--------------------- Plotting ---------------------'''


def plot_states(T, X, ode_T, ode_X):
    plt.rcParams.update({'axes.labelsize': 'x-large'})

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 10))
    gs = axs[0, 0].get_gridspec()
    for ax in axs[0:, 0]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:, 0])

    n_realisations = len(T)
    for i in range(n_realisations):
        realisation_color = get_rcolor()
        # Phase plot
        axbig.plot(X[i][:, 0], X[i][:, 1], color=realisation_color)
        axbig.set_ylabel(r'$x_{2}$')
        axbig.set_xlabel(r'$x_{1}$')
        # x1 vs time
        axs[0, 1].plot(T[i], X[i][:, 0], color=realisation_color)
        axs[0, 1].set_ylabel(r'$x_{1}$')
        axs[1, 1].set_xlabel('t')
        # x2 vs time
        axs[1, 1].plot(T[i], X[i][:, 1], color=realisation_color)
        axs[1, 1].set_ylabel(r'$x_{2}$')
        axs[1, 1].set_xlabel('t')

    axbig.plot(ode_X[:, 0], ode_X[:, 1], color='black', label='ODE reference', linewidth=2.5)
    axs[0, 1].plot(ode_T, ode_X[:, 0], color='black', linewidth=2.5)
    axs[1, 1].plot(ode_T, ode_X[:, 1], color='black', linewidth=2.5)


    plt.tight_layout()
    plt.legend()
