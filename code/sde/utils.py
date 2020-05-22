import numpy as np

# Adaptive step size parameters
DEFAULT_ABS_TOL = 1e-6
DEFAULT_REL_TOL = 1e-6

# Newton's method parameters
DEFAULT_NEWTONS_TOL = 1e-8
DEFAULT_NEWTONS_MAX_ITERS = 100

# Default solver
# Note originally in scipy the default for Atol is 1e-12
# https://github.com/scipy/scipy/blob/v1.4.1/scipy/integrate/_ode.py line 1198+
DEFAULT_NSTEPS = 500
DEFAULT_DFACTOR = 0.3


def parse_one_param(params, param_name, param_default_value):
    if param_name in params.keys():
        param = params[param_name]
        del params[param_name]
    else:
        param = param_default_value
    return params, param


def parse_newtons_params(params):
    params, newtons_tol = parse_one_param(params, 'newtons_tol', DEFAULT_NEWTONS_TOL)
    params, newtons_max_iters = parse_one_param(params, 'newtons_max_iters', DEFAULT_NEWTONS_MAX_ITERS)

    return params, newtons_tol, newtons_max_iters


colors = [
    # 'black',
    'grey',
    'brown',
    'peru',
    'darkorange',
    'olivedrab',
    'lightseagreen',
    'steelblue',
    'darkorchid',
    'pink'
]


def get_rcolor(seed=None):
    if seed:
        np.random.seed(seed)
    return colors[np.random.randint(0, len(colors))]


def wiener_process(T, N, dims=2):
    W = np.zeros((N, dims))
    dt = T / N
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size=(N - 1, dims))
    W[1:, :] = np.cumsum(dW, axis=0)

    # adapt size of dW array
    dW = np.vstack((np.zeros((1, dims)), dW))

    return dW, W
