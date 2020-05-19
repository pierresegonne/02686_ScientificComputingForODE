import matplotlib.pyplot as plt
import numpy as np

colors = [
    'black',
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

solver_options = {
    'default': {
        'color': 'lightseagreen'
    },
    'dopri54': {
        'color': 'black'
    },
    'esdirk23': {
        'color': 'black'
    },
    'explicit_euler': {
        'color': 'olivedrab'
    },
    'implicit_euler': {
        'color': 'brown'
    },
    'own_rk': {
        'color': 'black'
    },
    'rk4': {
        'color': 'black'
    }
}

def get_rcolor(seed=None):
    if seed:
        np.random.seed(seed)
    return colors[np.random.randint(0, len(colors))]

def correct_controllers_shape(controllers, N):
    '''Can either be 1 or N for each key'''
    for key in controllers.keys():
        if controllers[key].shape[0] == 1:
            controllers[key] = np.repeat(controllers[key], N)
        elif controllers[key].shape[0] == N:
            pass
        else:
            print(f"Incorrect controller parameter dimension {controllers[key].shape}. Can only be 1 or {N}")
    return controllers

def controller_plot(T, all_controllers, solvers):
    for i_c, controllers in enumerate(all_controllers):
        n = len(controllers.keys())
        fig, axs = plt.subplots(n)
        fig.suptitle(f"{solvers[i_c]}")
        controllers = correct_controllers_shape(controllers, T[i_c].shape[0])
        for i, key in enumerate(controllers.keys()):
            try:
                axs[i].plot(T[i_c], controllers[key], color=get_rcolor())
                axs[i].set_xlabel('t')
            except ValueError:
                axs[i].plot([i for i in range(controllers[key].shape[0])], controllers[key], color=get_rcolor())
            axs[i].set_ylabel(f"{key}")


