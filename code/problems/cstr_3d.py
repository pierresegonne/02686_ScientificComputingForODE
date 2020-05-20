import matplotlib.pyplot as plt
import numpy as np

from problems.cstr_shared import *

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


def reaction_extent(Cb):
    return 1 - (Cb / Cbin)


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


'''--------------------- Plotting ---------------------'''

def plot_states(T, X, solvers, solver_options):
    plt.rcParams.update({'axes.labelsize': 'x-large'})
    fig1, axs1 = plt.subplots(ncols=1, nrows=3, figsize=(13,8))
    fig2, axs2 = plt.subplots(ncols=1, nrows=3, figsize=(13,8))

    for i, solver in enumerate(solvers):
        # Phase plots x1vx2, x1vx3, x2vx3
        # Reminder: x1 = Ca, x2 = Cb, x3 = T
        axs1[0].plot(X[i][:, 0], X[i][:, 1], label=f"{solver}", color=solver_options[solver]['color'])
        axs1[0].set_xlabel(r'$C_{a}$ (mol/L)')
        axs1[0].set_ylabel(r'$C_{b}$ (mol/L)')
        axs1[0].plot(X[i][:, 0], X[i][:, 2] - 273.15, label=f"{solver}", color=solver_options[solver]['color'])
        axs1[1].set_xlabel(r'$C_{a}$ (mol/L)')
        axs1[1].set_ylabel('T (°C)')
        axs1[2].plot(X[i][:, 1], X[i][:, 2] - 273.15, label=f"{solver}", color=solver_options[solver]['color'])
        axs1[2].set_xlabel(r'$C_{b}$ (mol/L)')
        axs1[2].set_ylabel('T (°C)')
        # Vs time
        axs2[0].plot(T[i], X[i][:, 0], label=f"{solver}", color=solver_options[solver]['color'])
        axs2[0].set_xlabel('t (min)')
        axs2[0].set_ylabel(r'$C_{a} (mol/L)$')
        axs2[1].plot(T[i], X[i][:, 1], label=f"{solver}", color=solver_options[solver]['color'])
        axs2[1].set_xlabel('t (min)')
        axs2[1].set_ylabel(r'$C_{b}$ (mol/L)')
        axs2[2].plot(T[i], X[i][:, 2] - 273.15, label=f"{solver}", color=solver_options[solver]['color'])
        axs2[2].set_xlabel('t (min)')
        axs2[2].set_ylabel('T (°C)')

    # Add phase portrait for Temperature and Extent
    # plt.figure()
    # for i, solver in enumerate(solvers):
    #     plt.plot(reaction_extent(X[i][:, 1]), X[i][:, 2], label=f"{solver}", color=solver_options[solver]['color'])

    plt.tight_layout()
    plt.legend()