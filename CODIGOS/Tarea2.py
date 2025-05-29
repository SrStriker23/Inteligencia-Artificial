import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

# Generar puntos aleatorios
np.random.seed(42)  # Para reproducibilidad
x = np.random.rand(50) * 10  # 50 puntos aleatorios entre 0 y 10
y = 2.5 * x + np.random.randn(50) * 5  # y = 2.5x + ruido

# Ajustar el modelo de regresión lineal
x_reshaped = x.reshape(-1, 1)  # Cambiar la forma para sklearn
model = LinearRegression()
model.fit(x_reshaped, y)

# Coeficientes de la recta
slope = model.coef_[0]
intercept = model.intercept_

# Mostrar la ecuación de la recta
print(f"Ecuación de la recta: y = {slope:.2f}x + {intercept:.2f}")

# Función para actualizar la predicción
def update_prediction(text):
    try:
        x_pred = float(text)
        y_pred = model.predict([[x_pred]])
        prediction_point.set_offsets([[x_pred, y_pred[0]]])
        prediction_text.set_text(f"Predicción: y = {y_pred[0]:.2f}")
        
        # Actualizar líneas hacia los ejes
        hline.set_ydata([y_pred[0], y_pred[0]])
        vline.set_xdata([x_pred, x_pred])
        
        plt.draw()
    except ValueError:
        prediction_text.set_text("Entrada inválida")

# Graficar los puntos y la recta de regresión
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
ax.scatter(x, y, color='blue', label='Puntos aleatorios')
ax.plot(x, model.predict(x_reshaped), color='red', label='Recta de regresión')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Regresión Lineal')
ax.legend()

# Punto de predicción
prediction_point = ax.scatter([], [], color='green', label='Predicción', zorder=5)
prediction_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')

# Líneas hacia los ejes
hline = ax.axhline(y=0, color='green', linestyle='--', linewidth=0.8 ,)
vline = ax.axvline(x=0, color='green', linestyle='--', linewidth=0.8)

# Cuadro de texto para ingresar valores
axbox = plt.axes([0.2, 0.05, 0.6, 0.075])  # [left, bottom, width, height]
text_box = TextBox(axbox, 'Ingresa x: ')
text_box.on_submit(update_prediction)

plt.show()