import matplotlib.pyplot as plt
import numpy as np

from problems.cstr_1d import x_dimension as cstr_1d_x_dimension, f as cstr_1d_f, J as cstr_1d_J, \
    plot_states as cstr_1d_plot_states
from problems.cstr_3d import x_dimension as cstr_3d_x_dimension, f as cstr_3d_f, J as cstr_3d_J, \
    plot_states as cstr_3d_plot_states
from problems.test_equation import x_dimension as test_x_dimension, f as test_f, J as test_J, \
    plot_states as test_plot_states
from problems.vanderpol import x_dimension as vanderpol_x_dimension, f as vanderpol_f, J as vanderpol_J, \
    plot_states as vanderpol_plot_states

from solvers.default import ode_solver as default_ode_solver
from solvers.dopri54 import ode_solver as dopri54_ode_solver
from solvers.esdirk23 import ode_solver as esdirk23_ode_solver
from solvers.explicit_euler import ode_solver as explicit_euler_ode_solver
from solvers.implicit_euler import ode_solver as implicit_euler_ode_solver
from solvers.own_rk import ode_solver as own_rk_ode_solver
from solvers.rk4 import ode_solver as rk4_ode_solver

from utils import *


# Match problem
def get_problem(problem):
    if problem == 'test_equation':
        x_dimension, f, J, plotter_t = test_x_dimension, test_f, test_J, test_plot_states
        x0 = np.array([20])
        params = {'lbd': -1}

        def plotter(*args, **kwargs):
            return plotter_t(*args, **kwargs, with_true_solution=True, lbd=params['lbd'], x0=x0)
    elif problem == 'vanderpol':
        x_dimension, f, J, plotter = vanderpol_x_dimension, vanderpol_f, vanderpol_J, vanderpol_plot_states
        x0 = np.array([0.5, 0.5])
        params = {'mu': 12}
    elif problem == 'cstr_1d':
        x_dimension, f, J, plotter = cstr_1d_x_dimension, cstr_1d_f, cstr_1d_J, cstr_1d_plot_states
        x0 = np.array([273.65])
        params = {}
    elif problem == 'cstr_3d':
        x_dimension, f, J, plotter = cstr_3d_x_dimension, cstr_3d_f, cstr_3d_J, cstr_3d_plot_states
        x0 = np.array([0., 0., 273.65])
        params = {}
    else:
        raise NameError(f"Incorrect problem name {problem}")
    return x_dimension, f, J, plotter, x0, params


# Match solvers
def get_ode_solver(solver):
    if solver == 'default':
        ode_solver = default_ode_solver
    elif solver == 'dopri54':
        ode_solver = dopri54_ode_solver
    elif solver == 'esdirk23':
        ode_solver = esdirk23_ode_solver
    elif solver == 'explicit_euler':
        ode_solver = explicit_euler_ode_solver
    elif solver == 'implicit_euler':
        ode_solver = implicit_euler_ode_solver
    elif solver == 'own_rk':
        ode_solver = own_rk_ode_solver
    elif solver == 'rk4':
        ode_solver = rk4_ode_solver
    else:
        raise NameError(f"Incorrect solver name {solver}")
    return ode_solver


'''------------------ RUN ------------------'''
solvers = ['default', 'dopri54']
problem = 'cstr_3d'

adaptive_step_size = True

t0 = 0
tf = 35
# bh vdp 4000 cstr 200
# sh vdp cstr 100000
N = 200


# Simple check for time scale of CSTR
if (problem in ['cstr_1d', 'cstr_3d']) & (tf != 35):
    print('Wrong time scale for CSTR!')
    exit()

'''------------------     ------------------'''
all_X, all_T, all_controllers = [], [], []
x_dimension, f, J, plotter, x0, params = get_problem(problem)

for solver in solvers:
    ode_solver = get_ode_solver(solver)

    print(f"Running solver {solver}")

    X, T, controllers = ode_solver(f, J, t0, tf, N, x0, adaptive_step_size=adaptive_step_size, **params)
    all_X.append(X)
    all_T.append(T)
    all_controllers.append(controllers)

plotter(all_T, all_X, solvers, solver_options)
if adaptive_step_size:
    controller_plot(all_T, all_controllers, solvers)

plt.show()
