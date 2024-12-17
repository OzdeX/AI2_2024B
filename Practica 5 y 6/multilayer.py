# -------------- IMPORTACIONES --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import threading
from tkinter import ttk
from tkinter import *
from activations import *
from tkinter import filedialog, messagebox
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

# -------------- VARIABLES GLOBALES --------------
entradas_path = ""
salidas_path = ""


# -------------- RED NEURONAL MULTICAPA --------------
class MLP:
    def __init__(self, layers_dims, hidden_activation = tanh, output_activation=logistic):
        # Atributos
        self.L = len(layers_dims) - 1
        self.w = [None] * (self.L + 1)
        self.b = [None] * (self.L + 1)
        self.f = [None] * (self.L + 1)

        # Inicialización de los pesos
        for l in range(1, self.L + 1):
            self.w[l] = -1 + 2 * np.random.rand(layers_dims[l], layers_dims[l-1])
            self.b[l] = -1 + 2 * np.random.rand(layers_dims[l], 1)

            if l == self.L:
                self.f[l] = output_activation
            else:
                self.f[l] = hidden_activation


    def predict(self, X):
        a = np.asanyarray(X)
        for l in range(1, self.L + 1):
            z = np.dot(self.w[l], a) + self.b[l]
            a = self.f[l](z)
        return a


    def train(self, X, Y, net, epochs=800, lr=0.1):
        P = X.shape[1]
        tol = 0.001
        for _ in range(epochs):
            for p in range(P):
                # Inicializar activaciones
                a = [None] * (self.L + 1)
                da = [None] * (self.L + 1)
                lg = [None] * (self.L + 1)
                # Propagación
                a[0] = X[:, p]. reshape(-1, 1)
                for l in range(1, self.L + 1):
                    z = np.dot(self.w[l], a[l-1]) + self.b[l]
                    a[l], da[l] = self.f[l](z, derivative=True)

                # Retropropagación
                for l in range(self.L, 0, -1):
                    if l == self.L:
                        lg[l] = (Y[:, p].reshape(-1, 1) - a[l]) * da[l]
                    else:
                        lg[l] = np.dot(self.w[l+1].T, lg[l + 1]) * da[l]

                # Gradiente descendiente
                for l in range(1, self.L + 1):
                    self.w[l] += lr * np.dot(lg[l], a[l - 1].T)
                    self.b[l] += lr * lg[l]
            self.MLP_binary_classification_2d(X, Y, net)
            # mainwindow.after(0, self.MLP_binary_classification, X, Y, net)


            mse = np.mean((Y - self.predict(X))**2)
            if mse < tol:
                print("Fin - por Error")
                return
        print("Fin - por epocas")


    def MLP_binary_classification_2d(self, X, Y, net):
        ax.clear()
        for i in range(X.shape[1]):
            if Y[0, i] == 0:
                plt.plot(X[0, i], X[1, i], '.r', markersize=15)
            else:
                plt.plot(X[0, i], X[1, i], '.b', markersize=15)

        xmin, ymin=np.min(X[0,:])-0.5, np.min(X[1,:])-0.5
        xmax, ymax=np.max(X[0,:])+0.5, np.max(X[1,:])+0.5
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
        data = np.vstack([xx.ravel(), yy.ravel()])
        zz = net.predict(data)
        zz = zz.reshape(xx.shape)
        ax.contourf(xx, yy, zz, alpha = 0.90, cmap = plt.cm.RdBu)
        canvas.draw()


# -------------- PROCESO --------------
def Proceso():
    global net, entradas_path, salidas_path
    entradas = pd.read_csv(entradas_path)
    salidas = pd.read_csv(salidas_path)
    X = entradas.T.values
    Y = salidas.T.values
    
    print(X)
    print(Y)

    # Definir parámetros
    n_inputs = X.shape[0] #2
    n_outputs = Y.shape[0] #4

    # Solamente trabaja con 1 salida y 2 entradas
    net = MLP((n_inputs, 100, n_outputs))
    net.train(X, Y, net)


# -------------- FUNCIONES --------------
#Cerrar programa sin consumir recursos
def on_closing():
    plt.close('all')
    mainwindow.quit()
    mainwindow.destroy()
    sys.exit()


#Agregar archivos
def open_file_explorer():
    global entradas_path, salidas_path
    
    # Abrir el cuadro de diálogo para seleccionar un archivo
    entradas_path = filedialog.askopenfilename(title="Selecciona el archivo de entradas")
    if entradas_path:
        print(f"Archivo seleccionado: {entradas_path}")
    else:
        messagebox.showwarning("Advertencia", "No seleccionaste ningún archivo.")
    
    salidas_path = filedialog.askopenfilename(title="Selecciona el archivo de salidas")
    if salidas_path:
        print(f"Archivo seleccionado: {salidas_path}")
    else:
        messagebox.showwarning("Advertencia", "No seleccionaste ningún archivo.")



#-------------------- INTERFAZ --------------------
# Inicializar la gráfica con Matplotlib
fig, ax = plt.subplots(facecolor='#FFFFFF')

# Configurar límites y aspecto del gráfico
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('auto')

# Establecer los ticks en intervalos de 1 en los ejes
ax.set_xticks(range(-5, 5, 1))  # Eje X: desde -5 hasta 5, de 1 en 1
ax.set_yticks(range(-5, 5, 1))  # Eje Y: desde -5 hasta 5, de 1 en 1

# Interfaz Gráfica 
mainwindow = Tk()
mainwindow.geometry('1110x660')
mainwindow.wm_title('Practica 5 y 6')
mainwindow.configure(bg='#4A235A')

style = ttk.Style()
style.configure("TLabel", background="#664474", foreground="#d5aee6", font=("Arial", 20)) 
style.configure("TEntry", font=("Arial", 12))


# Frame para los parámetros
param_frame2 = Frame(mainwindow, bg='#4A235A')
param_frame2.place(x=20, y=0, width=580, height=45)
param_frame = Frame(mainwindow, bg='#2c0348', bd=10, relief=GROOVE)
param_frame.place(x=620, y=45, width=480, height=580)

#Titulo 1
Titulo1 = ttk.Label(param_frame2, background="#4A235A",foreground="#910bf3", text="Práctica 5 y 6. Red neuronal multicapa", font=("Arial", 20, "bold"))
Titulo1.grid(row=0, column=0, padx=10, pady=10, sticky="w")

#Titulo 2
Titulo2 = ttk.Label(param_frame,background="#2c0348", foreground="#c43cd6", text="Datos", font=("Arial", 20, "bold"))
Titulo2.grid(row=1, column=0, padx=10, pady=10, sticky="w")

# Gráfica en la interfaz 
canvas = FigureCanvasTkAgg(fig, master=mainwindow)
canvas.get_tk_widget().place(x=10, y=50, width=580, height=550)

# Botón para iniciar la red neuronal
train_button = Button(param_frame, text="Graficar", command=lambda: threading.Thread(target=Proceso).start(), bg="purple", fg="#FFFFFF", font=("Arial", 18, "bold"))
train_button.grid(row=9, column=0, padx=40, pady=20, sticky="w")

# Botón para abrir el administrador de archivos
button = Button(param_frame, text="Carga de archivos", command=open_file_explorer, bg="purple", fg="#FFFFFF", font=("Arial", 18, "bold"))
button.grid(row=10, column=0, padx=40, pady=20, sticky="w")

# Configurar la ventana de cierre
mainwindow.protocol("WM_DELETE_WINDOW", on_closing)
mainwindow.mainloop()