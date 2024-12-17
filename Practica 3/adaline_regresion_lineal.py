# Importaciones
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import ttk
import random
import threading

# --------------------------------------------------------
# Variables
epocas = 200 # Ciclos
eta = 0.1 # Tasa de aprendizaje
x = np.array([]) # Entradas
d = np.array([]) # Datos esperados
w = np.array([random.random(), random.random()]) # Pesos (bias y peso de la entrada)


# --------------------------------------------------------
# Funciones
def puntos(event):
    global x, d
    ix, iy = event.xdata, event.ydata
    print("{:.2f}".format(ix) + "/" + "{:.2f}".format(iy))
    
    # Agregar el nuevo punto a x
    if x.size == 0:
        x = np.array([[ix]])
    else:
        x = np.vstack((x, [[ix]])) # Apila el nuevo punto

    # Agregar la salida correspondiente a d
    if d.size == 0:
        d = np.array([iy])
    else:
        d = np.hstack((d, iy))

    # Graficar el punto
    ax.scatter(ix, iy, marker='D', color='b', edgecolor='black')
    canvas.draw()

def prediccion(p_punto, w):
    return np.dot(p_punto, w[1]) + w[0]

def grafica():
    ax.clear()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(0, color='k') # Eje horizontal
    ax.axvline(0, color='k') # Eje vertical
    
    # Dibujar puntos
    ax.scatter(x, d, color='b', edgecolor='black')
    
    # Dibujar línea de regresión si hay al menos dos puntos
    if len(x) >= 2:
        x_min, x_max = np.min(x), np.max(x) # Límites de los datos ingresados
        x_vals = np.linspace(x_min, x_max, 200) # Rango ajustado a los puntos ingresados
        y_vals = prediccion(x_vals, w)
        ax.plot(x_vals, y_vals, color='r')

    canvas.draw()


def on_closing():
    mainwindow.quit()


def clean():
    global x, d, w
    ax.cla()
    w = np.array([random.random(), random.random()]) # Reiniciar pesos

    # Limites de los ejes
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(0, color='k')
    ax.axvline(0, color='k')
    canvas.draw()

    x = np.array([])
    d = np.array([])
    W_label.config(text="Peso (W): {:.2f}".format(w[1]))
    Theta_label.config(text="Bias: {:.2f}".format(w[0]))
    train_button.config(state=NORMAL)


def AdalineRegresion():
    global x, d, w, ax, fig
    if x.size == 0 or d.size == 0:
        print("No hay datos para entrenar.")
        train_button.config(state=DISABLED)
        return

    train_button.config(state=DISABLED)
    x_bias = np.hstack((np.ones((x.shape[0], 1)), x))

    # Entrenamiento
    for epoca in range(epocas):
        y_pred = np.dot(x_bias, w)
        error = d - y_pred
        # Ajuste de pesos
        w += eta * np.dot(x_bias.T, error) / len(d)

        # Limpiar y actualizar la gráfica
        grafica()

        # Actualizar los valores de los pesos en la interfaz gráfica
        W_label.config(text="Peso (W): {:.2f}".format(w[1]))
        Theta_label.config(text="Bias: {:.2f}".format(w[0]))

        # Detener el entrenamiento si el error medio es suficientemente pequeño
        if np.mean(np.abs(error)) < 0.01:
            print("Entrenamiento completado con éxito.")
            break
    print(f"Entrenamiento terminado {w}")


# --------------------------------------------------------
# TKINTER
fig, ax = plt.subplots(facecolor='#FFFFFF')
fig.canvas.mpl_connect('button_press_event', puntos)

# Configurar límites y aspecto del gráfico
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axhline(0, color='k')
ax.axvline(0, color='k')

# Interfaz Gráfica
mainwindow = Tk()
mainwindow.geometry('1110x660')
mainwindow.wm_title('Práctica 3. Adaline con Regresión Lineal')
mainwindow.configure(bg='#4A235A')
style = ttk.Style()
style.configure("TLabel", background="#664474", foreground="#d5aee6", font=("Arial", 20))
style.configure("TEntry", font=("Arial", 12))

# Frame para los parámetros
param_frame2 = Frame(mainwindow, bg='#4A235A')
param_frame2.place(x=20, y=0, width=580, height=45)
param_frame = Frame(mainwindow, bg='#2c0348', bd=10, relief=GROOVE)
param_frame.place(x=620, y=45, width=480, height=580)

# Gráfica en la interfaz
canvas = FigureCanvasTkAgg(fig, master=mainwindow)
canvas.get_tk_widget().place(x=10, y=50, width=580, height=550)

# Título
Titulo1 = ttk.Label(param_frame2, background="#4A235A", foreground="#910bf3", text="Práctica 3. Adaline con Regresión", font=("Arial", 20, "bold"))
Titulo1.grid(row=0, column=0, padx=10, pady=10, sticky="w")

# Titulo2
Titulo2 = ttk.Label(param_frame, background="#2c0348", foreground="#c43cd6", text="Datos", font=("Arial", 20, "bold"))
Titulo2.grid(row=1, column=0, padx=10, pady=10, sticky="w")

# Etiquetas dentro del frame de parámetros
W_label = ttk.Label(param_frame, background="#2c0348", foreground="#6907ad", text="Peso (W): {:.2f}".format(w[1]), font=("Arial", 18, "bold"))
W_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
Theta_label = ttk.Label(param_frame, background="#2c0348", foreground="#6907ad", text="Bias: {:.2f}".format(w[0]), font=("Arial", 18, "bold"))
Theta_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
eta_label = ttk.Label(param_frame, background="#2c0348", foreground="#6907ad", text="Eta: {:.2f}".format(eta), font=("Arial", 18, "bold"))
eta_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")

# Botones
train_button = Button(param_frame, text="Graficar", command=lambda:
threading.Thread(target=AdalineRegresion).start(), bg="purple", fg="#FFFFFF", font=("Arial", 18, "bold"))
train_button.grid(row=9, column=0, padx=40, pady=40, sticky="w")
clean_button = Button(param_frame, text="Limpiar", command=clean, bg="purple", fg="#FFFFFF", font=("Arial", 18, "bold"))
clean_button.grid(row=9, column=1, padx=10, pady=10, columnspan=2)

# Configurar la ventana de cierre
mainwindow.protocol("WM_DELETE_WINDOW", on_closing)
mainwindow.mainloop()