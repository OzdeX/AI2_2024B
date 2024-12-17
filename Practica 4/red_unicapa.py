#-------------------- Importaciones --------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.colors import hsv_to_rgb
import tkinter
from tkinter import *
from tkinter import ttk
import threading
from tkinter import filedialog, messagebox
from sklearn.metrics import confusion_matrix
import seaborn as sns

#-------------------- Variables globales -------------------- 
entradas_path = ""
salidas_path = ""

#-------------------- Funciones de activación --------------------
def linear(z, derivate=False):
    a = z
    if derivate:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivate=False):
    a = 1 / (1 + np.exp(-z))
    if derivate:
        da = a * (1 - a)  # Derivada de la función logística
        return a, da
    return a


def softmax(z, derivate=False):
    e_z = np.exp(z - np.max(z, axis=0))
    a = e_z / np.sum(e_z, axis=0)
    if derivate:
        da = np.ones(z.shape)
        return a, da
    return a


#-------------------- Red nueronal unicapa --------------------
class OLN:
    def __init__(self, n_inputs, n_outputs, activation_function=linear):
        self.w = -1 + 2 * np.random.rand(n_outputs, n_inputs)
        self.b = -1 + 2 * np.random.rand(n_outputs, 1)
        self.f = activation_function

    def getWeights(self):
        return self.w
    
    def getBias(self):
        return self.b
    
    #Predicción
    def predict(self, X):
        Z = self.w @ X + self.b
        return self.f(Z)
    
    #Entrenamiento
    def fit(self,X, Y, epochs = 200, lr=0.1,n_classes=0):
        #Ir learning rate
        p = X.shape[1] #Numero de ejemplos
        tol = 0.03
        for _ in range(epochs):
            Z = np.dot(self.w, X) + self.b
            #Yest: y estimado, dY: derivada de Y
            Yest, dY = self.f(Z, derivate = True)
            #Calcular gradiente local
            lg = (Y - Yest)* dY
            #Actualizar matriz de pesos
            self.w += (lr/p) * np.dot(lg,X.T)
            self.b += (lr/p) * np.sum(lg, axis = 1).reshape(-1, 1) #Hacerlo de forma no iterativa
            
            mainwindow.after(0, self.MP_b_draw, X, n_classes, Y)
            # self.MP_b_draw(X,n_classes,Y)
            mse = np.mean((Y - Yest)**2)

            if mse < tol:
                print("Fin - por Error")
                return
        print("Fin - por epocas")
    
    # Graficación
    def MP_b_draw(self, X, n_classes, Y):
        ax.clear()
        hue_values = np.linspace(0,1,n_classes, endpoint=False)
        colors = [hsv_to_rgb([hue, 1, 1]) for hue in hue_values]
        cmap_custom  = ListedColormap(colors)
        for i in range(n_classes):
            indices = np.where(Y[i]==1)[0]
            # ax.scatter(X[0, indices], X[1, indices], edgecolors='k', c=colors[i], marker='o', s=100, label=f'Clase {i+1}')
            ax.scatter(X[0, indices], X[1, indices], edgecolors='k', color=colors[i], marker='o', s=100)

        x_min, x_max = X[0, :].min() -1, X[0, :].max() + 1
        y_min, y_max = X[1, :].min() -1, X[1, :].max() + 1
        # x_min, x_max = 0, 5
        # y_min, y_max = 0, 5

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

        grid = np.c_[xx.ravel(), yy.ravel()].T
        Z = self.predict(grid)
        Z = np.argmax(Z, axis=0)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes+1)-0.5, cmap=cmap_custom)

        # ax.legend()
        canvas.draw()
    
    def get_predictions(self, X):
        return np.argmax(self.predict(X), axis=0)


def Proceso():
    global red, entradas_path, salidas_path
    entradas = pd.read_csv(entradas_path)
    salidas = pd.read_csv(salidas_path)
    X = entradas.T.values
    Y = salidas.T.values

    # Definir parámetros
    n_inputs = X.shape[0] #2
    n_outputs = Y.shape[0] #4
    # learning_rate = .2

    epochs = int(Epochs.get())
    eta = float(Eta.get())

    red = OLN(n_inputs, n_outputs, activation_function=logistic)
    red.fit(X, Y, epochs, lr=eta, n_classes=n_outputs)


#-------------------- Funciones --------------------
#Cerrar programa sin consumir recursos
def on_closing():
    mainwindow.quit()

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


def show_confusion_matrix():
    global red
    plt.close('all')

    entradas = pd.read_csv(entradas_path)
    salidas = pd.read_csv(salidas_path)
    X = entradas.T.values
    Y = salidas.T.values
    Y_pred = red.get_predictions(X)
    Y_true = np.argmax(Y, axis=0)

    # Crear y mostrar la matriz de confusión
    cm = confusion_matrix(Y_true, Y_pred)

    # Programar la visualización de la matriz de confusión en el hilo principal
    mainwindow.after(0, lambda: plot_confusion_matrix(cm))


def plot_confusion_matrix(cm):
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=np.arange(cm.shape[0]), yticklabels=np.arange(cm.shape[0]))
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()



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
mainwindow.wm_title('Practica 4')
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
Titulo1 = ttk.Label(param_frame2, background="#4A235A",foreground="#910bf3", text="Práctica 4. Red neuronal unicapa", font=("Arial", 20, "bold"))
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

# Botón para mostrar la matriz de confusión
confusion_button = Button(param_frame, text="Mostrar matriz de confusión", command=show_confusion_matrix, bg="purple", fg="#FFFFFF", font=("Arial", 18, "bold"))
confusion_button.grid(row=11, column=0, padx=40, pady=20, sticky="w")

# Etiqueta y Entry de epocas
Epochs_label = ttk.Label(param_frame, background="#2c0348", foreground="#7d09cd", text="Epocas:",font=("Arial", 18, "bold"))
Epochs_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
Epochs = StringVar(mainwindow)
Epochs_entry = ttk.Entry(param_frame, textvariable=Epochs, width=20, font=("Arial", 12,))
Epochs_entry.grid(row=2, column=0, padx=20, pady=10)

#Etiqueta y Entry de Eta
Eta_label = ttk.Label(param_frame, background="#2c0348", foreground="#7d09cd", text="Eta:", font=("Arial", 18, "bold"))
Eta_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
Eta = StringVar(mainwindow)
Eta_entry = ttk.Entry(param_frame, textvariable=Eta, width=20, font=("Arial", 12,))
Eta_entry.grid(row=3, column=0, padx=20, pady=10)

# Configurar la ventana de cierre
mainwindow.protocol("WM_DELETE_WINDOW", on_closing)
mainwindow.mainloop()