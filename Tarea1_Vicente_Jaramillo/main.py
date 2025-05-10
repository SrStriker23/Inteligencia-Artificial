import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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

x_array = np.column_stack((np.ones(x_array.shape[0]), x_array))

#caracteristicas iniciales
alpha = 0.01
iteraciones = 1500
theta=np.array([0, 0])

def compute_cost(x, y, theta):
    m = len(y)
    J = (1/(2*m)*np.sum((x @ theta - y) ** 2))
    return J

def gradiente_descendente_grafico(x, y, theta, alpha, iteraciones):    
    m = len(y)
    J_historia = np.zeros(iteraciones)
    for i in range(iteraciones):
        recta = theta[0] + theta[1] * x[:, 1]
        plt.plot(x[:, 1], recta, color='green')
        plt.scatter(x[:, 1], y, color='red', marker='x', s=30)
        plt.xlabel('Poblacion de la ciudad en 10000s')
        plt.ylabel('Beneficio en 10000s')
        plt.legend(['Trainig data', 'linear regression'])
        plt.title(f'Recta generada {theta[0]:.3f} + {theta[1]:.3f} * x')
        plt.grid(True)
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()
        
        error = x.dot(theta) - y
        theta0 = theta[0] - alpha * (1/m) * np.sum(error * x[:, 0])
        theta1 = theta[1] - alpha * (1/m) * np.sum(error * x[:, 1])
        
        # Actualizar theta
        theta = np.array([theta0, theta1])
        
        # Guardar el costo en cada iteración
        J_historia[i] = compute_cost(x, y, theta)

def gradiente_descendente(x, y, theta, alpha, iteraciones):    
    m = len(y)
    J_historia = np.zeros(iteraciones)
    theta_historia = [np.array([10, 2])]
    for i in range(iteraciones):        
        error = x.dot(theta) - y
        theta0 = theta[0] - alpha * (1/m) * np.sum(error * x[:, 0])
        theta1 = theta[1] - alpha * (1/m) * np.sum(error * x[:, 1])
    
        # Actualizar theta
        theta = np.array([theta0, theta1])
        theta_historia.append(theta.copy())
        
        # Guardar el costo en cada iteración
        J_historia[i] = compute_cost(x, y, theta)
    
    return theta, J_historia,theta_historia

valores_algoritmo = gradiente_descendente(x_array, y_array, theta, alpha, iteraciones)

def graficar_costos(J_historia,alpha):
    plt.plot(J_historia, color='red')
    plt.title(f"Costo vs Iteraciones con alpha={alpha}")
    plt.xlabel("Iteraciones")
    plt.ylabel("Costo")
    plt.grid(True)
    plt.show()

def prediccion(theta, x_nuevo): 
    x_nuevo = np.array([1, x_nuevo])
    plt.show(block=False) 
    nuevo_x = float(input("Introduce un valor para x: "))
    prediccion= theta[0] + theta[1] * nuevo_x
    plt.scatter(nuevo_x, prediccion, color='blue', marker='o', s=20)
    plt.axhline(y=prediccion, color='blue', linestyle='--', xmin=0, xmax=nuevo_x/75)
    plt.axvline(x=nuevo_x, color='blue', linestyle='--', ymin=0, ymax=prediccion/75)
    plt.title(f"Recta generada {theta[0]} + {theta[1]} * x \nPunto de interseccion ({nuevo_x}, {prediccion})")

#graficar_costos(valores_algoritmo[1], alpha)
#gradiente_descendente_grafico(x_array, y_array,theta , alpha, iteraciones)

def graficar_funcion_costo_3d_con_camino(x, y, theta_historia):
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            theta_temp = np.array([theta0, theta1])
            J_vals[i, j] = compute_cost(x, y, theta_temp)

    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals.T, cmap='viridis', alpha=0.8)
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Costo')
    ax.set_title('Funcion de Costo con Camino del Gradiente Descendente')

    # Graficar el camino del gradiente descendente como puntos individuales
    theta0_historia = [theta[0] for theta in theta_historia]
    theta1_historia = [theta[1] for theta in theta_historia]
    J_historia = [compute_cost(x, y, theta) for theta in theta_historia]
    ax.scatter(theta0_historia, theta1_historia, J_historia, color='red', marker='x', s=30, label='Camino del Gradiente')
    ax.legend()
    plt.show()

def graficar_curvas_nivel_con_camino(x, y, theta_historia):
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            theta_temp = np.array([theta0, theta1])
            J_vals[i, j] = compute_cost(x, y, theta_temp)

    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

    plt.contour(theta0_vals, theta1_vals, J_vals.T, levels=np.logspace(-2, 3, 20), cmap='viridis')
    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.title('Curvas de Nivel de la Funcion de Costo')

    # Graficar solo las marcas de x del camino del gradiente descendente
    theta0_historia = [theta[0] for theta in theta_historia]
    theta1_historia = [theta[1] for theta in theta_historia]
    plt.scatter(theta0_historia, theta1_historia, color='red', marker='x', s=10, label='Camino del Gradiente')
    plt.legend()
    plt.grid(True)
    plt.show()

def ecuacion_normal(x, y):
    x_transpuesta = np.transpose(x)
    theta = np.linalg.inv(x_transpuesta @ x) @ x_transpuesta @ y
    return theta


valores_algoritmo = gradiente_descendente(x_array, y_array, theta, alpha, iteraciones)
#gradiente_descendente_grafico(x_array, y_array, theta, alpha, iteraciones)
#graficar_curvas_nivel_con_camino(x_array, y_array, valores_algoritmo[2])
#graficar_funcion_costo_3d_con_camino(x_array, y_array, valores_algoritmo[2])
graficar_costos(valores_algoritmo[1], alpha)
