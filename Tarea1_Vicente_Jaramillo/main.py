import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------------------------#

"""extraemos los datos de un archivo y los convertimos a listas para despues convertirlos a arrays de numpy"""

with open('data1.txt', 'r') as file:
    texto = file.readlines()
x = []
y = []
for coordenadas in texto:
    values = coordenadas.strip().split(',')
    x.append(float(values[0]))
    y.append(float(values[1]))

"""Datos normales"""

x_array = np.array(x)
y_array = np.array(y)

"""agragamos la fila de 1 a la matriz de x"""
x_array = np.column_stack((np.ones(x_array.shape[0]), x_array))


"""Normalizamos los datos para que los graficos sean mas claros"""
x_norm = x_array.copy()
x_norm[:, 1] = (x_array[:, 1] - np.mean(x_array[:, 1])) / np.std(x_array[:, 1])

"""Creamos variables para controlar los datos iniciales de las funciones"""
alpha = 0.1
iteraciones = 100
theta=np.array([200, 200])

#------------------------------------------------------------------------------------------------#

def compute_cost(x, y, theta):
    """creamos una funcion para guardar el valor de la funcion de costo en cada iteracion"""
    m = len(y)
    J = (1/(2*m)*np.sum((x @ theta - y) ** 2))
    return J

#------------------------------------------------------------------------------------------------#
def gradiente_descendente(x, y, theta, alpha, iteraciones):    
    """Funcion para obener datos del gradiente descendente sin graficar"""
    m = len(y)
    J_historia = np.zeros(iteraciones)
    theta_historia = [np.array([200, 200])]
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

#------------------------------------------------------------------------------------------------#
def graficar_gradiente_y_costos(x, y, theta, alpha, iteraciones, J_historia):
    """Funcion para graficar el gradiente descendente y la funcion de costo al mismo tiempo y ver como van cambiando a medida que se va iterando"""
    m = len(y)

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    fig1.suptitle(f"Gradiente Descendente con alpha={alpha}")

    fig2, ax2 = plt.subplots(figsize=(6, 6))

    fig1.canvas.manager.window.wm_geometry("+100+100")
    fig2.canvas.manager.window.wm_geometry("+800+100")

    for i in range(iteraciones):
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

        plt.pause(0.01)  

        if i < iteraciones - 1:
            ax1.cla()

        error = x.dot(theta) - y
        theta0 = theta[0] - alpha * (1/m) * np.sum(error * x[:, 0])
        theta1 = theta[1] - alpha * (1/m) * np.sum(error * x[:, 1])
        theta = np.array([theta0, theta1])

    fig2.subplots_adjust(bottom=0.22)
    ax_button = fig2.add_axes([0.3, 0.01, 0.4, 0.07])
    btn = Button(ax_button, 'Siguiente gráfico')

    def cerrar(_):
        plt.close(fig1)
        plt.close(fig2)
    btn.on_clicked(cerrar)
    plt.show()
#------------------------------------------------------------------------------------------------#

def graficar_funcion_costo_3d_con_camino(x, y, theta_historia):
    """Grafica la función de costo en 3D y el camino del gradiente descendente animado."""

    theta0_vals = np.linspace(-200, 200, 100)
    theta1_vals = np.linspace(-200, 200, 100)
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    
    for i, theta0 in enumerate(theta0_vals):
        for j, theta1 in enumerate(theta1_vals):
            theta_temp = np.array([theta0, theta1])
            J_vals[j, i] = compute_cost(x, y, theta_temp)

    theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    try:
        fig.canvas.manager.window.wm_geometry("+600+100")
    except Exception:
        pass
    surf = ax.plot_surface(theta0_mesh, theta1_mesh, J_vals, cmap='viridis', alpha=0.8, edgecolor='none')
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.set_zlabel('Costo')
    ax.set_title('Función de Costo y Camino del Gradiente Descendente')

    
    theta0_hist = [theta[0] for theta in theta_historia]
    theta1_hist = [theta[1] for theta in theta_historia]
    J_hist = [compute_cost(x, y, theta) for theta in theta_historia]

    camino, = ax.plot([], [], [], color='red', marker='x', label='Camino del Gradiente', linewidth=2, markersize=6)

    for i in range(1, len(theta0_hist) + 1):
        camino.set_data(theta0_hist[:i], theta1_hist[:i])
        camino.set_3d_properties(J_hist[:i])
        plt.pause(0.01)

    plt.subplots_adjust(bottom=0.18)
    ax_button = plt.axes([0.35, 0.05, 0.3, 0.07])
    btn = Button(ax_button, 'Siguiente gráfico')

    def cerrar(_):
        plt.close(fig)

    btn.on_clicked(cerrar)
    ax.legend()
    plt.show()

#------------------------------------------------------------------------------------------------#

def graficar_curvas_nivel_con_camino(x, y, theta_historia):
    """Funcion para graficar las curvas de nivel de la funcion de costo y el camino del gradiente descendente a medias que se va iterando"""

    theta0_vals2 = np.linspace(-200, 200, 100)
    theta1_vals2 = np.linspace(-200, 200, 100)
    J_vals = np.zeros((len(theta0_vals2), len(theta1_vals2)))

    for i, theta0 in enumerate(theta0_vals2):
        for j, theta1 in enumerate(theta1_vals2):
            theta_temp = np.array([theta0, theta1])
            J_vals[i, j] = compute_cost(x, y, theta_temp)

    theta0_vals2, theta1_vals2 = np.meshgrid(theta0_vals2, theta1_vals2)

    fig,ax= plt.subplots(figsize=(6, 6))
    try:
        fig.canvas.manager.window.wm_geometry("+600+100")
    except Exception:
        pass
    plt.contour(theta0_vals2, theta1_vals2, J_vals, levels=20, cmap='viridis')
    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.title('Curvas de Nivel de la Funcion de Costo')
    
    theta0_historia = [theta[0] for theta in theta_historia]
    theta1_historia = [theta[1] for theta in theta_historia]
    for i in range(len(theta0_historia)):
        plt.scatter(theta0_historia[i], theta1_historia[i], color='red', marker='x', s=10, label='Camino del Gradiente' if i == 0 else "")
        if i == 0:  
            plt.legend()
        plt.grid(True)
        plt.pause(0.01)

    plt.subplots_adjust(bottom=0.18)
    ax_button = plt.axes([0.3, 0.01, 0.4, 0.07])
    btn = Button(ax_button, 'Finalizar codigo')

    def cerrar(_):
        plt.close(plt.gcf())

    btn.on_clicked(cerrar)
    plt.show()

#------------------------------------------------------------------------------------------------#
def ecuacion_normal(x, y):
    """funcion para calcular los valores de theta usando la ecuacion normal"""
    x_transpuesta = np.transpose(x)
    theta = np.linalg.inv(x_transpuesta @ x) @ x_transpuesta @ y
    return theta
#------------------------------------------------------------------------------------------------#

def graficar_regresion_lineal(x, y, theta):
    """Funcion para graficar la regresion lineal utilizando los valores de theta obtenidos por la ecuacion normal y poder realizar predicciones"""

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.scatter(x[:, 1], y, color='red', marker='x', label='Datos de entrenamiento')
    ax.plot(x[:, 1], x @ theta, color='green', label='Regresión lineal')
    ax.set_xlabel('Población de la ciudad en 10,000s')
    ax.set_ylabel('Beneficio en 10,000s')
    ax.set_title(f'Regresión Lineal usando la Ecuación Normal \nrecta generada {theta[0]:.3f} + {theta[1]:.3f} * x')
    ax.legend()
    ax.grid(True)
    axbox = plt.axes([0.5, 0.02, 0.3, 0.05])   
    text_box = TextBox(axbox, 'Ingresar Población (en 10,000s):')  
    puntos_prediccion = []
    lineas_discontinuas = []

    xlim_original = ax.get_xlim()
    ylim_original = ax.get_ylim()

    def submit(text):
        nonlocal puntos_prediccion, lineas_discontinuas, xlim_original, ylim_original
        try:
            poblacion = int(text)
            beneficio = theta[0] + theta[1] * poblacion
            ax.set_title(f'Predicción: beneficio de {beneficio * 10000:.3f} para una población de {poblacion * 1000}')
            
            for punto in puntos_prediccion:
                punto.remove()
            for linea in lineas_discontinuas:
                linea.remove()
            puntos_prediccion.clear()
            lineas_discontinuas.clear()
           
            punto_prediccion, = ax.plot([poblacion], [beneficio], color='blue', marker='o', label='Predicción')
            puntos_prediccion.append(punto_prediccion)

            margen_x = (xlim_original[1] - xlim_original[0]) * 0.1
            margen_y = (ylim_original[1] - ylim_original[0]) * 0.1
            nuevo_xlim = list(xlim_original)
            nuevo_ylim = list(ylim_original)
            if poblacion < xlim_original[0]:
                nuevo_xlim[0] = poblacion - margen_x
            if poblacion > xlim_original[1]:
                nuevo_xlim[1] = poblacion + margen_x
            if beneficio < ylim_original[0]:
                nuevo_ylim[0] = beneficio - margen_y
            if beneficio > ylim_original[1]:
                nuevo_ylim[1] = beneficio + margen_y
            ax.set_xlim(nuevo_xlim)
            ax.set_ylim(nuevo_ylim)

            linea_x = ax.axhline(y=beneficio, color='blue', linestyle='--', xmin=0, xmax=(poblacion - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0]))
            linea_y = ax.axvline(x=poblacion, color='blue', linestyle='--', ymin=0, ymax=(beneficio - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]))
            lineas_discontinuas.extend([linea_x, linea_y])
            ax.legend()
            fig.canvas.draw_idle()  
        except ValueError:
            print("Por favor, ingrese un número válido.")
    text_box.on_submit(submit)
    plt.subplots_adjust(bottom=0.18)
    fig.set_size_inches(8, 6)
    try:
        fig.canvas.manager.window.wm_geometry("+400+100")
    except Exception:
        pass  
    ax_button = plt.axes([0.05, 0.05, 0.15, 0.05])
    btn = Button(ax_button, 'Siguiente gráfico')

    def cerrar(_):
        plt.close(fig)

    btn.on_clicked(cerrar)
    plt.show()
#------------------------------------------------------------------------------------------------#
"""DATOS PARA PODER EJECUTAR LAS FUNCIONES Y GRAFICAR LOS RESULTADOS"""
valores_graficos = gradiente_descendente(x_norm, y_array, theta, alpha, iteraciones)
valores_ecuacion_normal = ecuacion_normal(x_array, y_array)
graficar_gradiente_y_costos(x_norm, y_array, theta, alpha, iteraciones, valores_graficos[1])
graficar_regresion_lineal(x_array, y_array, valores_ecuacion_normal)
graficar_funcion_costo_3d_con_camino(x_norm, y_array, valores_graficos[2])
graficar_curvas_nivel_con_camino(x_norm, y_array, valores_graficos[2])
