import matplotlib.pyplot as plt
import numpy as np

colors = [
    'black', # true
    'grey',
    'brown', # ie
    'peru', # esdirk23
    'darkorange', # dopri54
    'olivedrab', # ee
    'lightseagreen', # default
    'steelblue',
    'darkorchid', # rk4
    'pink' # ownrk
]

solver_options = {
    'default': {
        'color': 'lightseagreen'
    },
    'dopri54': {
        'color': 'darkorange'
    },
    'esdirk23': {
        'color': 'peru'
    },
    'explicit_euler': {
        'color': 'olivedrab'
    },
    'implicit_euler': {
        'color': 'brown'
    },
    'own_rk': {
        'color': 'pink'
    },
    'rk4': {
        'color': 'darkorchid'
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
    plt.rcParams.update({'axes.labelsize': 'x-large'})
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


