import numpy as np
import matplotlib.pyplot as plt
with open('data1.txt', 'r') as file:
    texto = file.readlines()
print(texto)
x = []
y = []
for coordenadas in texto:
    values = coordenadas.strip().split(',')
    x.append(float(values[0]))
    y.append(float(values[1]))

x_array = np.array(x)
y_array = np.array(y)
  
def grafico(x, y):
    plt.title("Grafico de datos")
    plt.scatter(x, y, color="red", marker="x", s=20)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["Datos"])
    plt.show()

grafico(x_array, y_array)