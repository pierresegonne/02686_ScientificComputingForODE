import numpy as np

x_dimension = 1

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


def f(t, X, F=0.1, **kwargs):
    def Ca(T):
        return Cain + (1 / Beta) * (Tin - T)

    def Cb(T):
        return Cbin + (2 / Beta) * (Tin - T)

    def r(Ca, Cb, T):
        return k0 * np.exp(-EaR / T) * Ca * Cb

    def Rt(Ca, Cb, T):
        return Beta * r(Ca, Cb, T)

    T = X
    return (F / V) * (Tin - T) + Rt(Ca(T), Cb(T), T)


def reaction_extent(T):
    return -2 / (Beta * Cbin) * (Tin - T)


def J(t, X, F=0.1, **kwargs):
    def Ca(T):
        return Cain + (1 / Beta) * (Tin - T)

    def Ca_prime(T):
        return -(1 / Beta)

    def Cb(T):
        return Cbin + (2 / Beta) * (Tin - T)

    def Cb_prime(T):
        return -(2 / Beta)

    def k(T):
        return k0 * np.exp(-EaR / T)

    def k_prime(T):
        return EaR / (T ** 2) * k(T)

    T = X
    return np.array([[
        -(F / V) + Beta*(k_prime(T)*Ca(T)*Cb(T) + k(T)*Ca_prime(T)*Cb(T) + k(T)*Ca(T)*Cb_prime(T))
    ]])


'''--------------------- Plotting ---------------------'''

def plot_states(T, X, solvers, solver_options):
    pass
