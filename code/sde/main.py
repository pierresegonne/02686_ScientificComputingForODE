import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import ode

from explicit_explicit import sde_solver as ee_sde_solver
from implicit_explicit import sde_solver as ie_sde_solver

from vanderpol import x_dimension as vanderpol_x_dimension, f as vanderpol_f, J as vanderpol_J, \
    diffusion_1 as vanderpol_g_1, diffusion_2 as vanderpol_g_2, plot_states as vanderpol_plot_states

from utils import DEFAULT_ABS_TOL, DEFAULT_REL_TOL, DEFAULT_NEWTONS_TOL, DEFAULT_NEWTONS_MAX_ITERS
from utils import wiener_process


def get_problem(problem):
    if problem == 'vanderpol_1':
        x_dimension, f, J, g, plotter = vanderpol_x_dimension, vanderpol_f, vanderpol_J, vanderpol_g_1, vanderpol_plot_states
        x0 = np.array([0.5, 0.5])
        params = {'mu': 12, 'sigma': 0.5}
    elif problem == 'vanderpol_2':
        x_dimension, f, J, g, plotter = vanderpol_x_dimension, vanderpol_f, vanderpol_J, vanderpol_g_2, vanderpol_plot_states
        x0 = np.array([0.5, 0.5])
        params = {'mu': 2, 'sigma': 0.5}
    elif problem == 'cstr_1d':
        pass
    elif problem == 'ctsr_3d':
        pass
    return x_dimension, f, J, g, plotter, x0, params


def get_sde_solver(solver):
    if solver == 'explicit_explicit':
        return ee_sde_solver
    elif solver == 'implicit_explicit':
        return ie_sde_solver
    else:
        print(f'Wrong solver name {solver}')
        exit()


def ode_solver(f, J, t0, tf, N, x0, **kwargs):
    atol, rtol, nsteps, dfactor = DEFAULT_ABS_TOL, DEFAULT_REL_TOL, DEFAULT_NEWTONS_TOL, DEFAULT_NEWTONS_MAX_ITERS

    r = ode(f, J).set_integrator('dopri5',
                                 atol=atol,
                                 rtol=rtol,
                                 nsteps=nsteps,
                                 dfactor=dfactor,
                                 )
    # Fixup to not mess with function arguments. only function values are passed down, not arguments
    del kwargs['sigma']
    r.set_initial_value(x0, t0).set_f_params(*kwargs.values()).set_jac_params(*kwargs.values())

    if N < 5000:
        N = N * 100
    dt = (tf - t0) / N

    ode_T = [t0]
    ode_X = [x0]

    while r.successful() and r.t < tf:
        ode_T.append(r.t + dt)
        ode_X.append(r.integrate(r.t + dt))

    ode_T = np.array(ode_T)
    ode_X = np.array(ode_X)

    return ode_X, ode_T


'''------------------ RUN ------------------'''
solver = 'explicit_explicit'
problem = 'vanderpol_1'

t0 = 0
tf = 15
N = 1000

N_REALISATIONS = 5

# Simple check for time scale of CSTR
if (problem in ['cstr_1d', 'cstr_3d']) & (tf != 35):
    print('Wrong time scale for CSTR!')
    exit()
'''------------------     ------------------'''

x_dimension, f, J, g, plotter, x0, params = get_problem(problem)
sde_solver = get_sde_solver(solver)

all_X, all_T = [], []
for n in range(N_REALISATIONS):
    dW, _ = wiener_process(tf, N, x_dimension)
    X, T = sde_solver(f, J, g, dW, t0, tf, N, x0, **params)
    all_X.append(X)
    all_T.append(T)

# ODE reference
ode_X, ode_T = ode_solver(f, J, t0, tf, N, x0, **params)

plotter(all_T, all_X, ode_T, ode_X)
plt.show()
