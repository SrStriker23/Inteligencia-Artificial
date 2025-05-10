import numpy as np
import matplotlib.pyplot as plt
with open('data1.txt', 'r') as file:
    texto = file.readlines()
x = []
y = []
for coordenadas in texto:
    values = coordenadas.strip().split(',')
    x.append(float(values[0]))
    y.append(float(values[1]))

x_array = np.array(x)
y_array = np.array(y)

tamaño= len(x_array)   

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
    plt.title("Regresión Lineal")
    plt.scatter(x, y, color="red", marker="x", s=30)
    y_prediccion = recta[0] + recta[1] * x
    plt.plot(x, y_prediccion, color="g")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["Datos", f"Regresión Lineal: {recta[0]:.2f} + {recta[1]:.2f} x"])
    plt.title("Regresión Lineal")
    plt.show()

recta = regresion_lineal(x_array, y_array)
grafico(x_array, y_array, recta)