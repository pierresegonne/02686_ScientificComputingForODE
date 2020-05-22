import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.stats import norm as normal_distribution

from utils import get_rcolor, wiener_process
from explicit_explicit import sde_solver as ee_solver
from implicit_explicit import sde_solver as ie_solver

'''-------------------------- Test equation function --------------------------'''


def f(t, X, lbd=1, **kwargs):
    return lbd * X


def J(t, X, lbd=1, **kwargs):
    return np.array([[lbd]])


def g(t, X, sigma=1, **kwargs):
    return X * sigma


'''--------------------------  Parameters --------------------------'''

lbd = 0.1
sigma = 0.15
x0 = np.array([1])

T = 10
N = 500

x = np.linspace(0, T, N)

x_dimension = 1
t0 = 0
tf = T
params = {'lbd': lbd, 'sigma': sigma}

'''--------------------------  Realisations --------------------------'''

if __name__ == '__main__':

    # First let's compute the mean and std over a great number of realizations
    N_REALISATIONS = 1000
    all_realisations_log = np.zeros((N_REALISATIONS, N))
    for n in range(N_REALISATIONS):
        dW, W = wiener_process(T, N, dims=1)
        W = W.flatten()
        all_realisations_log[n, :] = np.log(x0 * np.exp((lbd - (sigma ** 2) / 2) * x + sigma * W))
    all_realisations_log_mean = np.mean(all_realisations_log, axis=0)
    all_realisations_log_std = np.std(all_realisations_log, axis=0)

    # Then numerical simulations

    # Explicit-Explicit
    all_realisations_ee_log = np.zeros((N_REALISATIONS, N))
    start = time.time()
    for n in range(N_REALISATIONS):
        dW, _ = wiener_process(tf, N, dims=x_dimension)
        X, _ = ee_solver(f, J, g, dW, t0, tf, N, x0, **params)
        all_realisations_ee_log[n, :] = np.log(X[:-1].flatten())
    end = time.time()
    print(f'EE {end - start}')
    all_realisations_ee_log_mean = np.mean(all_realisations_ee_log, axis=0)
    all_realisations_ee_log_std = np.std(all_realisations_ee_log, axis=0)

    # Implicit-Explicit
    all_realisations_ie_log = np.zeros((N_REALISATIONS, N))
    start = time.time()
    for n in range(N_REALISATIONS):
        dW, _ = wiener_process(tf, N, dims=x_dimension)
        X, _ = ie_solver(f, J, g, dW, t0, tf, N, x0, **params)
        all_realisations_ie_log[n, :] = np.log(X[:-1].flatten())
    end = time.time()
    print(f'IE {end - start}')
    all_realisations_ie_log_mean = np.mean(all_realisations_ie_log, axis=0)
    all_realisations_ie_log_std = np.std(all_realisations_ie_log, axis=0)

    # Then plot it against a few realisations
    # N_REALISATIONS = 10
    # plot_realisations = np.zeros((N_REALISATIONS, N))
    # r_colors = [get_rcolor() for _ in range(N_REALISATIONS)]
    # for n in range(N_REALISATIONS):
    #     dW, W = wiener_process(T, N, dims=1)
    #     W = W.flatten()
    #     plot_realisations[n, :] = x0 * np.exp((lbd - (sigma ** 2) / 2) * x + sigma * W)

    plt.rcParams.update({
        'axes.labelsize': 'x-large',
        'font.size': 14,
    })
    plt.figure()
    # Analytical
    plt.plot(x, np.exp(all_realisations_log_mean), color='black', linewidth=2, label='Analytical')
    plt.plot(x, np.exp(all_realisations_log_mean + (1.96 * all_realisations_log_std)), color='black', linewidth=2)
    plt.plot(x, np.exp(all_realisations_log_mean - (1.96 * all_realisations_log_std)), color='black', linewidth=2)
    # Explicit
    plt.plot(x, np.exp(all_realisations_ee_log_mean), color='olivedrab', label='Explicit-Explicit')
    plt.plot(x, np.exp(all_realisations_ee_log_mean + (1.96 * all_realisations_ee_log_std)), color='olivedrab')
    plt.plot(x, np.exp(all_realisations_ee_log_mean - (1.96 * all_realisations_ee_log_std)), color='olivedrab')
    # Implicit
    plt.plot(x, np.exp(all_realisations_ie_log_mean), color='brown', label='Implicit-Explicit')
    plt.plot(x, np.exp(all_realisations_ie_log_mean + (1.96 * all_realisations_ie_log_std)), color='brown')
    plt.plot(x, np.exp(all_realisations_ie_log_mean - (1.96 * all_realisations_ie_log_std)), color='brown')
    # No deviation
    # plt.plot(x, x0 * np.exp((lbd - (sigma ** 2) / 2) * x), color='pink', linewidth=2)
    # for n in range(N_REALISATIONS):
    #     plt.plot(x, plot_realisations[n], color=r_colors[n])
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend()

    plt.figure()
    # Analytical
    plt.plot(x, all_realisations_log_mean, color='black', linewidth=2, label='Analytical')
    plt.plot(x, all_realisations_log_mean + (1.96 * all_realisations_log_std), color='black', linewidth=2)
    plt.plot(x, all_realisations_log_mean - (1.96 * all_realisations_log_std), color='black', linewidth=2)
    # Analytical
    plt.plot(x, all_realisations_ee_log_mean, color='olivedrab', label='Explicit-Explicit')
    plt.plot(x, all_realisations_ee_log_mean + (1.96 * all_realisations_ee_log_std), color='olivedrab')
    plt.plot(x, all_realisations_ee_log_mean - (1.96 * all_realisations_ee_log_std), color='olivedrab')
    # Analytical
    plt.plot(x, all_realisations_ie_log_mean, color='brown', label='Implicit-Explicit')
    plt.plot(x, all_realisations_ie_log_mean + (1.96 * all_realisations_ie_log_std), color='brown')
    plt.plot(x, all_realisations_ie_log_mean - (1.96 * all_realisations_ie_log_std), color='brown')
    # for n in range(N_REALISATIONS):
    #     plt.plot(x, np.log(plot_realisations[n] / x0), color=r_colors[n])
    plt.xlabel('t')
    plt.ylabel(r'log(x(t)/$x_{0}$)')
    plt.legend()

    # Compute Distributions
    n_bins = 50

    # Analytical
    xT_mean = np.exp(all_realisations_log_mean[-1])
    xT_std = np.exp(all_realisations_log_std[-1])
    print(f"For the Analytical Solution, mean={xT_mean} (log:{np.log(xT_mean)}), std={xT_std} (log:{np.log(xT_std)})")
    norm = normal_distribution(loc=np.log(xT_mean), scale=np.log(xT_std))
    plt.figure()
    count, bins, ignored = plt.hist(all_realisations_log[:, -1], n_bins, color='grey')
    plt.plot(bins, norm.pdf(bins), color='black')
    plt.axvline(x=all_realisations_log_mean[-1] + (1.96 * all_realisations_log_std[-1]), color='darkmagenta')
    plt.axvline(x=all_realisations_log_mean[-1] - (1.96 * all_realisations_log_std[-1]), color='darkmagenta')

    # Explicit-Explicit
    xT_mean_ee = np.exp(all_realisations_ee_log_mean[-1])
    xT_std_ee = np.exp(all_realisations_ee_log_std[-1])
    print(f"For the Explicit-Explicit Method, mean={xT_mean_ee} (log:{np.log(xT_mean_ee)}), std={xT_std_ee} (log:{np.log(xT_std_ee)})")
    norm = normal_distribution(loc=np.log(xT_mean_ee), scale=np.log(xT_std_ee))
    plt.figure()
    count, bins, ignored = plt.hist(all_realisations_ee_log[:, -1], n_bins, color='yellowgreen')
    plt.plot(bins, norm.pdf(bins), color='olivedrab')
    plt.axvline(x=all_realisations_ee_log_mean[-1] + (1.96 * all_realisations_ee_log_std[-1]), color='darkmagenta')
    plt.axvline(x=all_realisations_ee_log_mean[-1] - (1.96 * all_realisations_ee_log_std[-1]), color='darkmagenta')

    # Implicit-Explicit
    xT_mean_ie = np.exp(all_realisations_ie_log_mean[-1])
    xT_std_ie = np.exp(all_realisations_ie_log_std[-1])
    print(f"For the Analytical Solution, mean={xT_mean_ie} (log:{np.log(xT_mean_ie)}), std={xT_std_ie} (log:{np.log(xT_std_ie)})")
    norm = normal_distribution(loc=np.log(xT_mean_ie), scale=np.log(xT_std_ie))
    plt.figure()
    count, bins, ignored = plt.hist(all_realisations_ie_log[:, -1], n_bins, color='salmon')
    plt.plot(bins, norm.pdf(bins), color='brown')
    plt.axvline(x=all_realisations_ie_log_mean[-1] + (1.96 * all_realisations_ie_log_std[-1]), color='darkmagenta')
    plt.axvline(x=all_realisations_ie_log_mean[-1] - (1.96 * all_realisations_ie_log_std[-1]), color='darkmagenta')



    plt.show()
