import matplotlib.pyplot as plt
import numpy as np

from utils import get_rcolor
from cstr_shared import *

x_dimension = 3


def f(t, X, **kwargs):
    def r(Ca, Cb, T):
        return k0 * np.exp(-EaR / T) * Ca * Cb

    def Ra(Ca, Cb, T):
        return -r(Ca, Cb, T)

    def Rb(Ca, Cb, T):
        return -2 * r(Ca, Cb, T)

    def Rt(Ca, Cb, T):
        return Beta * r(Ca, Cb, T)

    # X[0] = Ca, X[1] = Cb, X[2] = T
    Ca, Cb, T = X[0], X[1], X[2]
    return np.array([
        (F(t) / V) * (Cain - Ca) + Ra(Ca, Cb, T),
        (F(t) / V) * (Cbin - Cb) + Rb(Ca, Cb, T),
        (F(t) / V) * (Tin - T) + Rt(Ca, Cb, T),
    ])


def J(t, X, **kwargs):
    def k(T):
        return k0 * np.exp(-EaR / T)

    def k_prime(T):
        return EaR / (T ** 2) * k(T)

    Ca, Cb, T = X[0], X[1], X[2]
    return np.array([
        [-(F(t) / V) - k(T) * Cb, -k(T) * Ca, -k_prime(T) * Ca * Cb],
        [-2 * k(T) * Cb, -(F(t) / V) - 2 * k(T) * Ca, -2 * k_prime(T) * Ca * Cb],
        [Beta * k(T) * Cb, Beta * k(T) * Ca, -(F(t) / V) + Beta * k_prime(T) * Ca * Cb],
    ])


def diffusion(t, X, sigma=1, **kwargs):
    return (F(t) / V) * np.array([0, 0, sigma])


'''--------------------- Plotting ---------------------'''


def plot_states(T, X, ode_T, ode_X):
    plt.rcParams.update({'axes.labelsize': 'x-large'})
    fig2, axs2 = plt.subplots(ncols=1, nrows=3, figsize=(13, 8))

    axs2[0].set_xlabel('t (min)')
    axs2[0].set_ylabel(r'$C_{a} (mol/L)$')
    axs2[1].set_xlabel('t (min)')
    axs2[1].set_ylabel(r'$C_{b}$ (mol/L)')
    axs2[2].set_xlabel('t (min)')
    axs2[2].set_ylabel('T (°C)')

    n_realisations = len(T)
    for i in range(n_realisations):
        realisation_color = get_rcolor()
        # Vs time
        axs2[0].plot(T[i], X[i][:, 0], color=realisation_color)
        axs2[1].plot(T[i], X[i][:, 1], color=realisation_color)
        axs2[2].plot(T[i], X[i][:, 2] - 273.15, color=realisation_color)

    axs2[0].plot(ode_T, ode_X[:, 0], color='black', linewidth=2.5, label='ODE reference')
    axs2[1].plot(ode_T, ode_X[:, 1], color='black', linewidth=2.5)
    axs2[2].plot(ode_T, ode_X[:, 2] - 273.15, color='black', linewidth=2.5)

    plt.tight_layout()
    plt.legend()
