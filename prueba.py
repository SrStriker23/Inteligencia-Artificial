import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 7))
y = np.linspace(0, 35, 400)

# Restricciones originales
plt.axvline(x=5, color='b', label='x ≥ 5')
plt.axvline(x=10, color='g', label='x ≤ 10')
plt.axhline(y=30, color='orange', linestyle='--', label='Nueva y ≤ 30')  # Nueva capacidad
plt.plot(28 - y, y, 'r', label='x + y ≤ 28')
plt.plot((2/3)*y + 1, y, 'purple', label='x ≤ (2/3)y + 1')

# Región factible
y_fill = np.linspace(6, 28, 400)
x_lower = np.maximum(5, (2/3)*y_fill + 1)
x_upper = np.minimum(10, 28 - y_fill)
plt.fill_betweenx(y_fill, x_lower, x_upper, color='gray', alpha=0.3)

# Solución óptima
plt.plot(10, 18, 'ro', markersize=8, label='Óptimo (10,18)')

plt.xlim(0, 15); plt.ylim(0, 35)
plt.xlabel('Carga Frágil (x)'); plt.ylabel('Carga Normal (y)')
plt.legend(bbox_to_anchor=(1.05, 1)); plt.grid(True)
plt.show()