import numpy as np

x_dimension = 3

'''--------------------- Constants ---------------------'''
DeltaHr = -560  # Reaction Enthalpy
rho = 1  # Density
cp = 4.186  # Specific Heat Capacity
Beta = DeltaHr / (rho * cp)
Cain = 1.6 / 2  # Inlet Concentration A
Cbin = 2.4 / 2  # Inlet Concentration B
EaR = 8500  # Activation Energy
Tin = 273.65  # Inlet Temperature Or 0.5 degree Celsius
k0 = np.exp(24.6)  # Arrhenius Constant
V = 0.105  # Volume of Reactor


def f(t, X, F=0.1):
    def r(Ca, Cb, T):
        return k0 * np.exp(-EaR / T) * Ca * Cb

    def Ra(Ca, Cb, T):
        return -r(Ca, Cb, T)

    def Rb(Ca, Cb, T):
        return -2 * r(Ca, Cb, T)

    def Rt(Ca, Cb, T):
        return Beta * r(Ca, Cb, T)

    # X[0] = Ca, X[1] = Cb, X[2] = T
    Ca, Cb, T = X[0], X[1], X[2]
    return np.array([
        (F / V) * (Cain - Ca) + Ra(Ca, Cb, T),
        (F / V) * (Cbin - Cb) + Rb(Ca, Cb, T),
        (F / V) * (Tin - T) + Rt(Ca, Cb, T),
    ])


def reaction_extent(Cb):
    return 1 - (Cb / Cbin)


def J(t, X, F=0.1):
    def k(T):
        return k0 * np.exp(-EaR / T)

    def k_prime(T):
        return EaR / (T ** 2) * k(T)

    Ca, Cb, T = X[0], X[1], X[2]
    return np.array([
        [-(F / V) - k(T) * Cb, -k(T) * Ca, -k_prime(T) * Ca * Cb],
        [-2 * k(T) * Cb, -(F / V) - 2 * k(T) * Ca, -2 * k_prime(T) * Ca * Cb],
        [Beta * k(T) * Cb, Beta * k(T) * Ca, -(F / V) + Beta * k_prime(T) * Ca * Cb],
    ])


'''--------------------- Plotting ---------------------'''

def plot_states(T, X, solvers, solver_options):
    pass
