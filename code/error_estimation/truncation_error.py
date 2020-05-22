import matplotlib.pyplot as plt
import numpy as np

"""
This is a simple demonstrator for definitions of truncation errors.
"""

x = np.linspace(0,1,1000)
approx = np.array([20, 10, 5])

plt.rcParams.update({
    'axes.labelsize': 'x-large',
    'font.size': 20,
})


plt.figure()
plt.plot(x, 20*np.exp(-x), label=r'x(t)=$x_{0}$(t)', color='black')
plt.plot(x[500:], 10*np.exp(-x[500:]+ 0.5), label='$x_{1}$(t)', color='steelblue')
plt.plot([0, 0.5, 1], approx, label='explicit_euler', color='olivedrab')

plt.annotate(r'$x_{0}$', (0,20))
plt.annotate(r'$x_{1}$', (0.5,10))
plt.annotate(r'$x_{2}$', (1,5))
plt.annotate(r'$x(t_{2})$', (1,7.3))
plt.annotate(r'$x_{1}(t_{2})$', (1,6))
plt.xticks([0, 0.5, 1], [r'$t_{0}$=0', r'$t_{1}$=0.5', r'$t_{2}$=1'])

plt.legend()

print('local errors', np.abs(20*np.exp(-x[499]) - 10), np.abs(10*np.exp(-x[-1]+ 0.5)) - 5)
print('global error', np.abs(20*np.exp(-x[-1]) - 5))

plt.show()
