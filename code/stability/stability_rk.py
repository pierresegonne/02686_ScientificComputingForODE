import matplotlib.pyplot as plt
import numpy as np

# RK4
# Butcher Tableau
C = np.array([0, 1 / 2, 1 / 2, 1])
A = np.array([
    [0, 0, 0, 0],
    [1 / 2, 0, 0, 0],
    [0, 1 / 2, 0, 0],
    [0, 0, 1, 0],
])
B = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])

# DOPRI54
# Butcher Tableau
C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
A = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [1 / 5, 0, 0, 0, 0, 0, 0],
    [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
    [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
    [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
])
B = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100,
              1 / 40])
B_hat = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
E = np.array([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525,
              1 / 40])

# ESDIRK23
# Butcher Tableau
gamma = (2 - np.sqrt(2)) / 2
C = np.array([0, 2 * gamma, 1])
A = np.array([
    [0, 0, 0],
    [gamma, gamma, 0],
    [(1 - gamma) / 2, (1 - gamma) / 2, gamma]
])
B = np.array([(1 - gamma) / 2, (1 - gamma) / 2, gamma])
B_hat = np.array(
    [(6 * gamma - 1) / (12 * gamma), 1 / (12 * gamma * (1 - 2 * gamma)), (1 - 3 * gamma) / (3 * (1 - 2 * gamma))])
E = B - B_hat

# Meshgrid
n = 500
plot_span = 20
real = np.linspace(-plot_span, plot_span, n)
imag = np.linspace(-plot_span, plot_span, n)

I = np.eye(A.shape[0])
e = np.ones((A.shape[0], 1))
absR = np.zeros((n, n))

for i_r, r in enumerate(real):
    for i_i, i in enumerate(imag):
        z = r + 1j * i
        R = 1 + z * B.T @ np.linalg.pinv(I - z * A) @ e
        absR[i_r, i_i] = np.abs(R)

absR = np.clip(absR, 0, 1).T

fig, ax = plt.subplots(nrows=1, ncols=1)
im = ax.imshow(absR, extent=[-plot_span, plot_span, -plot_span, plot_span], aspect='auto', cmap='RdYlGn_r')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
fig.colorbar(im, ax=ax)
ax.set_ylabel(r'Im(h$\lambda$)')
ax.set_xlabel(r'Re(h$\lambda$)')
ax.set_title('TODO')

plt.show()
