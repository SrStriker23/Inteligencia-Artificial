import numpy as np
import matplotlib.pyplot as plt
# Datos

with open('data1.txt', 'r') as file:
    texto = file.readlines()
x = []
y = []
for coordenadas in texto:
    values = coordenadas.strip().split(',')
    x.append(float(values[0]))
    y.append(float(values[1]))

x = np.array(x)
y = np.array(y)

def regresion_lineal(x, y):
    n = np.size(x)
    media_x = np.mean(x)
    media_y = np.mean(y)
    Sxx = np.sum(x * y) - n * media_x * media_y
    Sxy = np.sum(x * x) - n * media_x * media_x
    theta1 = Sxy / Sxx
    theta0 = media_y - theta1 * media_x
    return (theta0, theta1)

def grafico(x, y, recta):
    plt.scatter(x, y, color="m", marker="o", s=30)
    y_prediccion = recta[0] + recta[1] * x
    plt.plot(x, y_prediccion, color="g")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

x_b = np.c_[np.ones((x.shape[0], 1)), x] 
theta = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y
print("Theta:", theta)
theta_lineal= regresion_lineal(x, y)
print("Theta lineal:", theta_lineal)
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Primer subplot: Datos originales con la recta de regresión
axs[0].scatter(x, y, color="m", marker="o", s=30)
y_prediccion = theta_lineal[0] + theta_lineal[1] * x
axs[0].plot(x, y_prediccion, color="g")
axs[0].set_title("Regresión Lineal")
plt.figure(1)
plt.scatter(x, y, color="m", marker="o", s=30)
y_prediccion = theta_lineal[0] + theta_lineal[1] * x
plt.plot(x, y_prediccion, color="g")
plt.title("Regresión Lineal")
plt.xlabel("x")
plt.ylabel("y")

plt.figure(2)
errores = y - (theta_lineal[0] + theta_lineal[1] * x)
plt.scatter(x, errores, color="b", marker="x", s=30)
plt.axhline(0, color="r", linestyle="--")
plt.title("Errores")
plt.xlabel("x")
plt.ylabel("Error")

plt.show()
axs[0].set_ylabel("y")

# Segundo subplot: Error cuadrático
errores = y - (theta_lineal[0] + theta_lineal[1] * x)
axs[1].scatter(x, errores, color="b", marker="x", s=30)
axs[1].axhline(0, color="r", linestyle="--")
axs[1].set_title("Errores")
axs[1].set_xlabel("x")
axs[1].set_ylabel("Error")

plt.tight_layout()
plt.show()