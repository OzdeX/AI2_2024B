import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import ttk
from sklearn.base import BaseEstimator
import random
import threading


X = []
d = [] #arreglo de deseados


def plot_point(event):
    ix, iy = event.xdata, event.ydata
    print("{:.2f}".format(ix) + "/" +
    "{:.2f}".format(iy))
    X.append((ix, iy))
    if event.button == 1:
        d.append(1)
        ax.plot(ix,iy, 'Dg')
    elif event.button == 3:
        d.append(0)
        ax.plot(ix,iy, 'Db')
    canvas.draw()


def clean():
    global X
    global d
    ax.cla()
    ejeX = [-10,10] #marcar las rectas del plano cartesiano

    ejeY = [-10,10]
    ceros = [0,0]
    plt.plot(ejeX, ceros, 'k') #se grafica una linea en el eje x
    plt.plot(ceros, ejeY, 'k') #se grafica una linea en el eje y
    canvas.draw()
    X = []
    d = []


def Perceptron():
    global W1, W2, Theta, Eta, Epocas_str, X, d
    Epocas = int(Epocas_str.get())
    error = True
    while Epocas and error:
        e = [] #arreglo de errores

        for i in range(len(X)):
            Y = np.dot(X[i],[W1,W2]) - Theta >=0
            e.append(d[i]-Y)
            W1 = W1 + (float(Eta.get())*e[-1]*X[i][0])
            W2 = W2 + (float(Eta.get())*e[-1]*X[i][1])
            Theta = Theta - (float(Eta.get())*e[-1]*1)
        W = [W1,W2]
        
        #Generamos los valores para la funcion de la recta
        m = -(W[0]/W[1]) #pendiente
        b = -(Theta/W[1]) #altura en la que se coloca el punto

        ax.cla()
        #Imprimir plano cartesiano

        ejeX = [-10,10] #marcar las rectas del plano cartesiano
        ejeY = [-10,10]
        ceros = [0,0]
        plt.plot(ejeX, ceros, 'k') #se grafica una linea en el eje x
        plt.plot(ceros, ejeY, 'k') #se grafica una linea en el eje y

        #Recorrer la matriz y graficar los puntos correspondientes
        for i in range(len(X)):
            if X[i][0]*W[0] + X[i][1]*W[1] + Theta >=0:
                #Si es mayor o igual a cero pasa
                plt.plot(X[i][0],X[i][1],'Dg')
            else:
                plt.plot(X[i][0],X[i][1],'Db')
        plt.axline((X[0][0], (X[0][0]*m)+b), slope=m, color='y') #Dibujar la recta

        canvas.draw()
        W1_label.config(text = "Peso 1 (W1): {:.2f}".format(W1))

        W2_label.config(text = "Peso 2 (W2): {:.2f}".format(W2))

        Theta_label.config(text = "Bias: {:.2f}".format(Theta))

        if not(1 in e) and not(-1 in e):
            error = False
        
        Epocas = Epocas - 1


def on_closing():
    mainwindow.quit()


#-------------------- INTERFAZ --------------------
#Inicializar la grafica con mathplotlib
fig, ax = plt.subplots(facecolor='#FFFFFF')
fig.canvas.mpl_connect('button_press_event', plot_point)
plt.xlim(-10,10)
plt.ylim(-10,10)
#Imprimir plano cartesiano
ejeX = [-10,10] #marcar las rectas del plano cartesiano
ejeY = [-10,10]
ceros = [0,0]
plt.plot(ejeX, ceros, 'k') #se grafica una linea en el eje x
plt.plot(ceros, ejeY, 'k') #se grafica una linea en el eje y

#Interfaz Grafica
mainwindow = Tk()
mainwindow.geometry('1110x660')
mainwindow.wm_title('Practica 2. Equipo G.')
mainwindow.configure(bg='#4A235A')
style = ttk.Style()
style.configure("TLabel", background="#664474", foreground="#d5aee6", font=("Arial", 20))
style.configure("TEntry", font=("Arial", 12))

# Frame para los parámetros
param_frame2 = Frame(mainwindow, bg='#4A235A')
param_frame2.place(x=20, y=0, width=580, height=45)
param_frame = Frame(mainwindow,bg='#2c0348',bd=10,
relief=GROOVE)

param_frame.place(x=620, y=45, width=480, height=580)

#Valores de los pesos
W1 = random.random()
W2 = random.random()
Theta = random.random()
Eta = StringVar(mainwindow)
Epocas_str = StringVar(mainwindow)

#Grafica en la interfaz
canvas = FigureCanvasTkAgg(fig, master = mainwindow)
canvas.get_tk_widget().place(x=10, y=50, width=580, height=550)

#Titulo1
Titulo1 = ttk.Label(param_frame2,
background="#4A235A",foreground="#910bf3", text="Práctica 2. Perceptron con entrenamiento", font=("Arial", 20, "bold"))
Titulo1.grid(row=0, column=0, padx=10, pady=10, sticky="w")
# Titulo2
Titulo2 = ttk.Label(param_frame,background="#2c0348", foreground="#c43cd6", text="Datos", font=("Arial", 20, "bold"))
Titulo2.grid(row=1, column=0, padx=10, pady=10, sticky="w")

#Etiquetas dentro del frame de parámetros
W1_label = ttk.Label(param_frame, background="#2c0348", foreground="#6907ad",text = "Peso 1 (W1): {:.2f}".format(W1),font=("Arial", 18, "bold"))
W1_label.grid(row=2, column=0, padx=10, pady=5,sticky="w")

W2_label = ttk.Label(param_frame, background="#2c0348", foreground="#6907ad",text = "Peso 2 (W2): {:.2f}".format(W2),font=("Arial", 18, "bold"))
W2_label.grid(row=3, column=0, padx=10, pady=5,sticky="w")
Theta_label = ttk.Label(param_frame,
background="#2c0348", foreground="#6907ad", text = "Bias: {:.2f}".format(Theta),font=("Arial", 18, "bold"))
Theta_label.grid(row=4, column=0, padx=10, pady=5,sticky="w")
Eta_label = ttk.Label(param_frame, background="#2c0348", foreground="#7d09cd", text = "Valor de Eta: ",font=("Arial", 18, "bold"))
Eta_label.grid(row=5, column=0, padx=10, pady=5,sticky="w")
Eta_entry = ttk.Entry(param_frame, textvariable=Eta, font=("Arial", 12,))
Eta_entry.grid(row=5, column=1, padx=5, pady=10)
Epocas_label = ttk.Label(param_frame, background="#2c0348", foreground="#7d09cd", text = "Epocas: ",font=("Arial", 18, "bold"))
Epocas_label.grid(row=6, column=0, padx=10, pady=5,sticky="w")
Epocas_entry = ttk.Entry(param_frame,
textvariable=Epocas_str,font=("Arial", 12,))
Epocas_entry.grid(row=6, column=1, padx=5, pady=10)

start_button = Button(param_frame, text="Graficar", command=lambda:threading.Thread(target=Perceptron).start(),bg="purple", fg="#FFFFFF", font=("Arial", 18, "bold"))

start_button.grid(row=8, column=0, padx=10, pady=40)
clean_button = Button(param_frame, text="Limpiar", command=clean, bg="purple", fg="#FFFFFF", font=("Arial", 18, "bold"))
clean_button.grid(row=8, column=1, padx=10, pady=40)
mainwindow.protocol("WM_DELETE_WINDOW", on_closing)
mainwindow.mainloop()