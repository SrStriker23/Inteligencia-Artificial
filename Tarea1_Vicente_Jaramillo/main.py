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

# Datos normales
x_array = np.array(x)
y_array = np.array(y)

x_array = np.column_stack((np.ones(x_array.shape[0]), x_array))

# Datos normalizados
x_norm = x_array.copy()
x_norm[:, 1] = (x_array[:, 1] - np.mean(x_array[:, 1])) / np.std(x_array[:, 1])

#caracteristicas iniciales
alpha = 0.01
iteraciones = 1500
theta=np.array([0, 0])

def compute_cost(x, y, theta):
    m = len(y)
    J = (1/(2*m)*np.sum((x @ theta - y) ** 2))
    return J

def graficar_gradiente_y_costos(x, y, theta, alpha, iteraciones, J_historia):
    m = len(y)

    # Crear la primera ventana para el gradiente descendente
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    fig1.suptitle(f"Gradiente Descendente con alpha={alpha}")

    # Crear la segunda ventana para el costo vs iteraciones
    fig2, ax2 = plt.subplots(figsize=(6, 6))

    for i in range(iteraciones):
        # Gradiente descendente
        recta = theta[0] + theta[1] * x[:, 1]
        ax1.plot(x[:, 1], recta, color='green')
        ax1.scatter(x[:, 1], y, color='red', marker='x', s=30)
        ax1.set_xlabel('Poblacion de la ciudad en 10000s')
        ax1.set_ylabel('Beneficio en 10000s')
        ax1.legend(['Training data', 'Linear regression'])
        ax1.set_title(f'Recta generada {theta[0]:.3f} + {theta[1]:.3f} * x')
        ax1.grid(True)

        # Costo vs Iteraciones
        ax2.plot(J_historia[:i+1], color='red')
        ax2.set_title("Costo vs Iteraciones")
        ax2.set_xlabel("Iteraciones")
        ax2.set_ylabel("Costo")
        ax2.grid(True)

        plt.pause(0.001)  # Pausa para actualizar los gráficos

        # Limpiar el gráfico de gradiente descendente para la siguiente iteración
        ax1.cla()

        # Actualizar theta
        error = x.dot(theta) - y
        theta0 = theta[0] - alpha * (1/m) * np.sum(error * x[:, 0])
        theta1 = theta[1] - alpha * (1/m) * np.sum(error * x[:, 1])
        theta = np.array([theta0, theta1])

    plt.show()
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
valores_costo=gradiente_descendente(x_norm, y_array, theta, alpha, iteraciones)
graficar_gradiente_y_costos(x_norm, y_array, theta, alpha, iteraciones, valores_costo[1])
def graficar_funcion_costo_3d_con_camino(x, y, theta_historia):
    theta0_vals = np.linspace(-200, 200, 100)  # Expandir el rango para cubrir más área
    theta1_vals = np.linspace(-200, 200, 100)  # Expandir el rango para cubrir más área
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            theta_temp = np.array([theta0, theta1])
            J_vals[j, i] = compute_cost(x, y, theta_temp)  # Corregir el índice para que coincida con la malla

    theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis', alpha=0.8)
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
valores=gradiente_descendente(x_norm, y_array, theta, alpha, iteraciones)
graficar_funcion_costo_3d_con_camino(x_norm, y_array, valores[2])

def graficar_curvas_nivel_con_camino(x, y, theta_historia):
    theta0_vals = np.linspace(-50, 55, 100)
    theta1_vals = np.linspace(-50, 55, 100)
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

