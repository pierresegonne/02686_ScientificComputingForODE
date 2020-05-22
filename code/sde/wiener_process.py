import matplotlib.pyplot as plt
import numpy as np
import os

from sys import path

path.append(os.path.realpath('.'))

from mpl_toolkits.mplot3d import Axes3D

from utils import get_rcolor


def wiener_process(T, N, dims=2):
    W = np.zeros((N, dims))
    dt = T / N
    dW = np.random.normal(loc=0, scale=dt, size=(N - 1, dims))
    W[1:, :] = np.cumsum(dW, axis=0)

    # adapt size of dW array
    dW = np.vstack((np.zeros((1, dims)), dW))

    return dW, W


# Parameters
T = 10
N = 1000
d = 2

plt.rcParams.update({
    'axes.labelsize': 'x-large',
    'font.size': 14,
})

if d == 1:
    fig = plt.figure()
    for i in range(10):
        dW, W = wiener_process(T, N, dims=d)
        plt.plot(np.linspace(0, T, N), W[:, 0], color=get_rcolor(), label=f"Realisation #{i + 1}")
    plt.xlabel('t')
    plt.ylabel(r'$W_{1}$')
elif d == 2:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(10):
        dW, W = wiener_process(T, N, dims=d)
        ax.plot(W[:, 0], W[:, 1], np.linspace(0, T, N), color=get_rcolor(), label=f"Realisation #{i + 1}")
    ax.set_xlabel(r'$W_{1}$')
    ax.set_ylabel(r'$W_{2}$')
    ax.set_zlabel('t')
else:
    print('no visualisation for d > 2')

plt.show()
