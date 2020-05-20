import matplotlib.pyplot as plt
import numpy as np

from problems.cstr_shared import *

x_dimension = 1


def f(t, X, **kwargs):
    def Ca(T):
        return Cain + (1 / Beta) * (Tin - T)

    def Cb(T):
        return Cbin + (2 / Beta) * (Tin - T)

    def r(Ca, Cb, T):
        return k0 * np.exp(-EaR / T) * Ca * Cb

    def Rt(Ca, Cb, T):
        return Beta * r(Ca, Cb, T)

    T = X
    return (F(t) / V) * (Tin - T) + Rt(Ca(T), Cb(T), T)


def reaction_extent(T):
    return -2 / (Beta * Cbin) * (Tin - T)


def J(t, X, **kwargs):
    def Ca(T):
        return Cain + (1 / Beta) * (Tin - T)

    def Ca_prime(T):
        return -(1 / Beta)

    def Cb(T):
        return Cbin + (2 / Beta) * (Tin - T)

    def Cb_prime(T):
        return -(2 / Beta)

    def k(T):
        return k0 * np.exp(-EaR / T)

    def k_prime(T):
        return EaR / (T ** 2) * k(T)

    T = X
    return np.array([[
        -(F(t) / V) + Beta * (k_prime(T) * Ca(T) * Cb(T) + k(T) * Ca_prime(T) * Cb(T) + k(T) * Ca(T) * Cb_prime(T))
    ]])


'''--------------------- Plotting ---------------------'''


def plot_states(T, X, solvers, solver_options):
    plt.rcParams.update({'axes.labelsize': 'x-large'})
    plt.figure(figsize=(10, 3))

    for i, solver in enumerate(solvers):
        plt.plot(T[i], np.array(X[i]) - 273.15, label=f"{solver}", color=solver_options[solver]['color'])
        plt.xlabel('t (min)')
        plt.ylabel('T (°C)')
        xticks = [5 * i for i in range(8)]
        xlabels = [5 * i for i in range(8)]
        plt.xticks(ticks=xticks, labels=xlabels)


    plt.legend()
