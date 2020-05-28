from numpy import exp

'''--------------------- Constants ---------------------'''
DeltaHr = -560  # Reaction Enthalpy kJ/mol
rho = 1  # Density kg/L
cp = 4.186  # Specific Heat Capacity kJ/(kg * K)
Beta = -DeltaHr / (rho * cp)  # K * L/mol
Cain = 1.6 / 2  # Inlet Concentration A mol/L
Cbin = 2.4 / 2  # Inlet Concentration B mol/L
EaR = 8500  # Activation Energy K
Tin = 273.65  # Inlet Temperature K
k0 = exp(24.6) * 60  # Arrhenius Constant L/(mol * min)
V = 0.105  # Volume of Reactor L


def F(t):
    flow = 1000
    if t <= 3:
        flow = 700
    elif t <= 5:
        flow = 600
    elif t <= 7:
        flow = 500
    elif t <= 9:
        flow = 400
    elif t <= 12:
        flow = 300
    elif t <= 16:
        flow = 200
    elif t <= 18:
        flow = 300
    elif t <= 20:
        flow = 400
    elif t <= 22:
        flow = 500
    elif t <= 24:
        flow = 600
    elif t <= 28:
        flow = 700
    elif t <= 32:
        flow = 200
    else:
        flow = 700
    return flow / 1000
