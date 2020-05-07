import matplotlib.pyplot as plt
import numpy as np


def explicit_euler(n=1000):
    """
    |1+hlambda| < 1
    """
    theta = np.linspace(0, 2 * np.pi, n)
    x = np.cos(theta) - 1
    y = np.sin(theta)
    return x, y


def implicit_euler(n=1000):
    """
    |1+hlambda| < 1
    """
    theta = np.linspace(0, 2 * np.pi, n)
    x = 1 - np.cos(theta)
    y = np.sin(theta)
    return x, y


x_ee, y_ee = explicit_euler()
x_ie, y_ie = implicit_euler()

green_color = 'lightgreen'
red_color = 'firebrick'

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.axis('equal')
ax.fill(x_ee, y_ee, green_color)
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.set_facecolor(red_color)
ax.set_ylabel(r'Im(h$\lambda$)')
ax.set_xlabel(r'Re(h$\lambda$)')
ax.set_title('Explicit Euler Method')

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.axis('equal')
ax.fill(x_ie, y_ie, red_color)
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
ax.set_facecolor(green_color)
ax.set_ylabel(r'Im(h$\lambda$)')
ax.set_xlabel(r'Re(h$\lambda$)')
ax.set_title('Implicit Euler Method')

plt.show()
