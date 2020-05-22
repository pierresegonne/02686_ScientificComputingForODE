import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.linear_model import LinearRegression

from explicit_explicit import sde_solver as ee_solver
from implicit_explicit import sde_solver as ie_solver

from test_equation import lbd, sigma, T, x0, wiener_process
from test_equation import f, J, g, t0, tf, params, x_dimension

N_REALISATIONS = 1000

n_step_trials = 25
Ns = np.geomspace(10000, 10, n_step_trials, dtype=int)
dts = T / Ns

ee_E = np.zeros((n_step_trials,))
ie_E = np.zeros((n_step_trials,))

print(f"[N REALISATIONS {N_REALISATIONS}]")

for i, N in enumerate(Ns):
    print(f"Currently evaluating the error for step size {T/N}")

    # Need to compare on the same wiener process.
    all_W = np.zeros((N_REALISATIONS, N))
    all_dW = np.zeros((N_REALISATIONS, N))
    for n in range(N_REALISATIONS):
        dW, W = wiener_process(T, N, dims=1)
        all_W[n, :] = W.flatten()
        all_dW[n, :] = dW.flatten()
    print('  - Wiener processes generated')

    # Analytical solution
    # Note that we can directly access x(T) is we know the wiener process
    all_xT = np.zeros((N_REALISATIONS,))
    for n in range(N_REALISATIONS):
        all_xT[n] = x0 * np.exp((lbd - (sigma ** 2) / 2) * T + sigma * all_W[n, -1])
    print('  - Analytical Solution | OK')

    # Explicit Explicit
    all_xT_ee = np.zeros((N_REALISATIONS,))
    start = time.time()
    for n in range(N_REALISATIONS):
        X, _ = ee_solver(f, J, g, all_dW[n, :], t0, tf, N, x0, **params)
        all_xT_ee[n] = X[-1][0]
    end = time.time()
    print(f'  - Explicit Explicit | OK | {end - start}')


    # Explicit Explicit
    all_xT_ie = np.zeros((N_REALISATIONS,))
    start = time.time()
    for n in range(N_REALISATIONS):
        X, _ = ie_solver(f, J, g, all_dW[n, :], t0, tf, N, x0, **params)
        all_xT_ie[n] = X[-1][0]
    end = time.time()
    print(f'  - Implicit Explicit | OK | {end - start}')

    ee_E[i] = np.mean(np.abs(all_xT - all_xT_ee))
    ie_E[i] = np.mean(np.abs(all_xT - all_xT_ie))

regressor = LinearRegression().fit(np.log(dts[:, None]), np.log(ee_E))
e_theoretical_ee = regressor.predict(np.log(dts[:, None]))
regressor = LinearRegression().fit(np.log(dts[:, None]), np.log(ie_E))
e_theoretical_ie = regressor.predict(np.log(dts[:, None]))

plt.rcParams.update({
    'axes.labelsize': 'x-large',
    'font.size': 20,
})

plt.figure()
plt.plot(dts, ee_E, label=r'True $e_{k}$', linewidth=2, color='olivedrab')
plt.plot(dts, np.exp(e_theoretical_ee), label=r'Theoretical $e_{k}$, $\mathcal{O}(h)$', linewidth=2, color='grey')
plt.xlabel('h')
plt.ylabel(r'$l_{k}$')
plt.legend()

plt.figure()
plt.plot(dts, ie_E, label=r'True $e_{k}$', linewidth=2, color='brown')
plt.plot(dts, np.exp(e_theoretical_ie), label=r'Theoretical $e_{k}$, $\mathcal{O}(h)$', linewidth=2, color='grey')
plt.xlabel('h')
plt.ylabel(r'$l_{k}$')
plt.legend()

plt.show()
