
# Adaptive step size parameters
DEFAULT_ABS_TOL = 1e-6
DEFAULT_REL_TOL = 1e-6
DEFAULT_EPS_TOL = 0.8
DEFAULT_FACTOR_MAX = 5
DEFAULT_FACTOR_MIN = 0.1

# Newton's method parameters
DEFAULT_NEWTONS_TOL = 1e-8
DEFAULT_NEWTONS_MAX_ITERS = 100


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


def parse_adaptive_step_params(params):

    params, abstol = parse_one_param(params, 'abstol', DEFAULT_ABS_TOL)
    params, reltol = parse_one_param(params, 'reltol', DEFAULT_REL_TOL)
    params, epstol = parse_one_param(params, 'epstol', DEFAULT_EPS_TOL)
    params, facmax = parse_one_param(params, 'facmax', DEFAULT_FACTOR_MAX)
    params, facmin = parse_one_param(params, 'facmin', DEFAULT_FACTOR_MIN)

    return params, abstol, reltol, epstol, facmax, facmin