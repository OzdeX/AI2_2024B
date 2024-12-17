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
epocas = 100 # Ciclos
eta = 0.1 # Tasa de aprendizaje
x = np.array([]) # Entradas
d = np.array([]) # Datos esperadas
ndatos = np.array([])
w = np.array([random.random(),random.random(),random.random()]) #Pesos
flag = False


# --------------------------------------------------------
# Funciones
def puntos(event):
    global x, d, ndatos
    ix, iy = event.xdata, event.ydata
    print("{:.2f}".format(ix) + "/" + "{:.2f}".format(iy))
    ndatos = np.array([[ix,iy]])
    if x.size == 0:
        x = np.array([[ix, iy]])
    else:
        x = np.vstack((x, ndatos)) # Apila el nuevo punto

    if event.button == 1:
        d = np.hstack((d, np.array([1])))
        ax.scatter(ix, iy, marker='D', color='b', edgecolor='black')
    elif event.button == 3:
        d = np.hstack((d, np.array([0])))
        ax.scatter(ix, iy, marker='D', color='r', edgecolor='black')
    canvas.draw()

def sigmoide(p_punto):
    return 1 / (1 + np.exp(-p_punto))

def derivada_sigmoide(y_sigmoide):
    return sigmoide(y_sigmoide) * (1 - sigmoide(y_sigmoide))

def prediccion(p_punto,w):
    p_punto = np.dot(w,p_punto.T)
    return 1/(1 + np.exp(- p_punto))

def grafica():
    x1_range = np.linspace(-5, 5, 200)
    x2_range = np.linspace(-5, 5, 200)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    # Aplanar la cuadrícula para que coincida con las entradas del modelo
    x_test = np.hstack((np.ones((X1.ravel().shape[0], 1)), X1.ravel().reshape(-1, 1), X2.ravel().reshape(-1, 1)))

    # Predecir los valores para toda la cuadrícula
    Z = prediccion(x_test, w).reshape(X1.shape)
    Z = Z.reshape(X1.shape)
    levels = np.linspace(Z.min(), Z.max(), 50)
    ax.contourf(X1, X2, Z, levels=levels, cmap='RdYlBu', alpha=0.7)

def on_closing():
    mainwindow.quit()

def clean():
    global x, d, w
    ax.cla()
    w = np.array([random.random(),random.random(),random.random()])
    # Limites de los ejes
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Dibujar líneas de los ejes
    ax.axhline(0, color='k') # línea en el eje x
    ax.axvline(0, color='k') # línea en el eje y
    canvas.draw()

    x = np.array([])
    d = np.array([])
    W1_label.config(text="Peso 1 (W1): {:.2f}".format(w[1]))
    W2_label.config(text="Peso 2 (W2): {:.2f}".format(w[2]))
    Theta_label.config(text="Bias: {:.2f}".format(w[0]))
    train_button.config(state=NORMAL)


# --------------------------------------------------------
# Adaline
def AdalineLogistica():
    global x, d, w, ax, fig
    train_button.config(state=DISABLED)
    x = np.hstack((np.ones((x.shape[0], 1)), x)) # Añadir columna de bias a las entradas
    
    # Entrenamiento
    for epoca in range(epocas):
        p_punto = np.dot(w, x.T)
        y_sigmoide = sigmoide(p_punto)
        derivada = derivada_sigmoide(y_sigmoide)
        error = d - y_sigmoide
        
        # Ajuste de pesos
        for i in range(w.shape[0]):
            w[i] = w[i] + eta * np.sum(error * derivada * x[:, i])

        # Limpiar el gráfico antes de redibujar
        ax.clear()

        # Reestablecer los ejes y las líneas
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.axhline(0, color='k') # Eje horizontal
        ax.axvline(0, color='k') # Eje vertical

        grafica()
        # Dibujar los puntos (con sus colores correspondientes según la predicción)
        for i in range(len(x)):
            p_punto = np.dot(x[i], w)
            y = sigmoide(p_punto)
            color = 'b' if y >= 0.5 else 'r'
            ax.scatter(x[i][1], x[i][2], marker='D', color=color, edgecolor='black')
        
        # Redibujar el gráfico en el canvas
        canvas.draw()

        # Actualizar los valores de los pesos en la interfaz gráfica
        W1_label.config(text="Peso 1 (W1): {:.2f}".format(w[1]))
        W2_label.config(text="Peso 2 (W2): {:.2f}".format(w[2]))
        Theta_label.config(text="Bias: {:.2f}".format(w[0]))
        # Detener el entrenamiento si el error es suficientemente pequeño
        if np.all(np.abs(error) < 0.01):
            print("Entrenamiento completado con éxito.")
            break

    print(f"Entrenamiento terminado {w}")


# --------------------------------------------------------
# TKINTER
# Inicializar la gráfica con Matplotlib
fig, ax = plt.subplots(facecolor='#FFFFFF')
fig.canvas.mpl_connect('button_press_event', puntos)

# Configurar límites y aspecto del gráfico
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('auto')

# Dibujar líneas de los ejes
ax.axhline(0, color='k') # línea en el eje x
ax.axvline(0, color='k') # línea en el eje y

# Interfaz Gráfica
mainwindow = Tk()
mainwindow.geometry('1110x660')
mainwindow.wm_title('Practica 3')
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

# Titulo1
Titulo1 = ttk.Label(param_frame2, background="#4A235A", foreground="#910bf3", text="Práctica 3. Adaline", font=("Arial", 20, "bold"))
Titulo1.grid(row=0, column=0, padx=10, pady=10, sticky="w")

# Titulo2
Titulo2 = ttk.Label(param_frame, background="#2c0348", foreground="#c43cd6", text="Datos", font=("Arial", 20, "bold"))
Titulo2.grid(row=1, column=0, padx=10, pady=10, sticky="w")

# Etiquetas dentro del frame de parámetros
W1_label = ttk.Label(param_frame, background="#2c0348", foreground="#6907ad", text="Peso 1 (W1): {:.2f}".format(w[1]), font=("Arial", 18, "bold"))
W1_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
W2_label = ttk.Label(param_frame, background="#2c0348", foreground="#6907ad", text="Peso 2 (W2): {:.2f}".format(w[2]), font=("Arial", 18, "bold"))
W2_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")
Theta_label = ttk.Label(param_frame, background="#2c0348", foreground="#6907ad", text="Bias: {:.2f}".format(w[0]), font=("Arial", 18, "bold"))
Theta_label.grid(row=4, column=0, padx=10, pady=5, sticky="w")

# Entradas para la tasa de aprendizaje y épocas
eta_label = ttk.Label(param_frame, background="#2c0348", foreground="#6907ad", text="Eta: {:.2f}".format(eta), font=("Arial", 18, "bold"))
eta_label.grid(row=5, column=0, padx=10, pady=5, sticky="w")

# Botón para iniciar el perceptrón
train_button = Button(param_frame, text="Graficar", command=lambda: threading.Thread(target=AdalineLogistica).start(), bg="purple", fg="#FFFFFF", font=("Arial", 18, "bold"))
train_button.grid(row=9, column=0, padx=40, pady=40, sticky="w")
clean_button = Button(param_frame, text="Limpiar", command=clean, bg="purple", fg="#FFFFFF", font=("Arial", 18, "bold"))
clean_button.grid(row=9, column=1, padx=10, pady=10, columnspan=2)

# Configurar la ventana de cierre
mainwindow.protocol("WM_DELETE_WINDOW", on_closing)
mainwindow.mainloop()