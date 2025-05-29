import numpy as np
from scipy.optimize import fsolve

# Definir la función diferencia
def f(x):
    return 0.011*x**4 - 1.9619*x**3 + 130.181*x**2 - 3836.59*x + 42373

# Hacer una estimación inicial del caudal, por ejemplo 45
x0 = 45

# Encontrar la raíz (el caudal de operación)
Q_operacion = fsolve(f, x0)[0]

# Calcular H en ese punto (opcional)
def H_bomba(x):
    return 0.0133*x**4 - 2.3488*x**3 + 155.11*x**2 - 4550.6*x + 50047

H_operacion = H_bomba(Q_operacion)

print(f"Caudal de operación: {Q_operacion:.2f} L/min")
print(f"H de operación: {H_operacion:.2f} m")

