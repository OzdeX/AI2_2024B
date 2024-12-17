import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import messagebox
from tkinter import *
from tkinter import ttk


X = [] # Creación de lista para almacenar las coordenadas de los puntos
def plot_point(event): # Función para capturar los clics (coordenadas)
    ix, iy = event.xdata, event.ydata # Coordenadas del punto.
    X.append((ix, iy)) # Guardamos las coordenadas en la lista.
    ax.plot(ix,iy,'Pk', linewidth=2, markersize=5) # Dibujo del punto en el gráfico
    canvas.draw() # Actualizamos el gráfico.


def clean(): #Función para hacer limpieza en el gráfico.
    global X
    ax.cla()
    ejeX = [-5,5]
    ejeY = [-5,5]
    ceros = [0,0]
    plt.plot(ejeX, ceros, 'black')
    plt.plot(ceros, ejeY, 'black')
    plt.grid(axis='x', color='0.95')
    plt.grid(axis='y', color='0.95')
    canvas.draw() # Actualizamos el gráfico.
    X = [] # Reiniciamos la lista para admitir valores nuevos en espacios anteriores


def percep(): # Función para aplicar el algoritmo de Perceptron
    if len(X) == 0:
        messagebox.showerror(title="Advertencia", message="No existen coordenadas")
    else:
        if (W1.get() != '' and W2.get() != '' and Bias_str.get() != ''):
            w1 = float(W1.get())
            w2 = float(W2.get())
            bias = float(Bias_str.get())

            # Formulas para calcular la pendiente de la linea y la intersección en Y.
            m = -(w1/w2)
            b = -(bias/w2)
            
            # Limites para para el gráfico (aumentar si se necesita en la parte de la interfaz)
            ax.cla()
            ejeX = [-5,5]
            ejeY = [-5,5]
            ceros = [0,0]
            plt.plot(ejeX, ceros, 'black') # Dibujamos ambos ejes.
            plt.plot(ceros, ejeY, 'black')
            plt.grid(axis='x', color='0.95') # Agrega una cuadrícula en el eje X.
            plt.grid(axis='y', color='0.95') # Agrega una cuadrícula en el eje Y.
            
            for i in range(len(X)):
                if X[i][0]*w1 + X[i][1]*w2 + bias >=0:
                    plt.plot(X[i][0],X[i][1],'Pb', linewidth=2, markersize=5)
                else:
                    plt.plot(X[i][0],X[i][1],'Pr', linewidth=2, markersize=5)
            plt.axline((X[0][0], (X[0][0]*m)+b), slope=m, color='violet') # Dibuja la línea que divide el plano en violeta.
            canvas.draw()
        else:
            messagebox.showerror(title="Advertencia", message="Faltan valores")


def on_closing():
    if messagebox.askokcancel("Salir", "Esta seguro que desea salir?"):
        mainwindow.quit()


#-------------------- INTERFAZ --------------------
# Definir ventana
mainwindow = Tk()
mainwindow.geometry('1080x640')
mainwindow.wm_title('Equipo G - Practica 1')
mainwindow.configure(bg='#4A235A')
# Definir estilo
style = ttk.Style()
style.configure("TLabel", background="#664474", foreground="#d5aee6", font=("Arial", 20)) # Ajustes para los labels en general (Color de fondo, color exterior y tamaño de fuente)

style.configure("TButton", background="#8E44AD", foreground="#4A235A", font=("Arial", 18,
"bold"), padding=5) # Ajustes para los botones
style.configure("TEntry", font=("Arial", 12))
# Definir plano interactivo
fig, ax = plt.subplots(facecolor='#664474',)
fig.canvas.mpl_connect('button_press_event', plot_point)
plt.xlim(-5, 5)
plt.ylim(-5, 5)
ejeX = [-5, 5]
ejeY = [-5, 5]
ceros = [0, 0]
plt.plot(ejeX, ceros, 'black')
plt.plot(ceros, ejeY, 'black')
plt.grid(axis='x', color='0.95')
plt.grid(axis='y', color='0.95')
canvas = FigureCanvasTkAgg(fig, master=mainwindow)
canvas.get_tk_widget().place(x=20, y=45, width=580, height=580)
# Contenedor de controles
controls_frame = Frame(mainwindow, bg='#4A235A')
controls_frame.place(x=20, y=0, width=440, height=45)
control_frame = Frame(mainwindow, bg='#664474', bd=10, relief=GROOVE) #Creación del marco
control_frame.place(x=620, y=45, width=440, height=580)
#Titulo1
Titulo1 = ttk.Label(controls_frame, background="#4A235A",foreground="#910bf3",
text="Práctica 1. Perceptron", font=("Arial", 20, "bold"))
Titulo1.grid(row=0, column=0, padx=10, pady=10, sticky="w")
# Titulo2
Titulo2 = ttk.Label(control_frame, foreground="#c43cd6", text="Ingresar datos",
font=("Arial", 20, "bold"))
Titulo2.grid(row=0, column=0, padx=10, pady=10, sticky="w")
#
# Definir Peso 1
W1_label = ttk.Label(control_frame, foreground="#3c0e40", text="Valor para w1:",
font=("Arial", 18, "bold"))
W1_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")
W1 = StringVar(mainwindow)
W1_entry = ttk.Entry(control_frame, textvariable=W1, width=20, font=("Arial", 12,))
W1_entry.grid(row=1, column=1, padx=10, pady=10)

# Definir Peso 2
W2_label = ttk.Label(control_frame, foreground="#3c0e40", text="Valor para w2:",
font=("Arial", 18, "bold"))
W2_label.grid(row=2, column=0, padx=10, pady=10, sticky="w")
W2 = StringVar(mainwindow)
W2_entry = ttk.Entry(control_frame, textvariable=W2, width=20, font=("Arial", 12,))
W2_entry.grid(row=2, column=1, padx=10, pady=10)
# Definir Sesgo
Bias_label = ttk.Label(control_frame, foreground="#3c0e40", text="Bias:", font=("Arial", 18,
"bold"))
Bias_label.grid(row=3, column=0, padx=10, pady=10, sticky="w")
Bias_str = StringVar(mainwindow)
Bias_entry = ttk.Entry(control_frame, textvariable=Bias_str, width=20, font=("Arial", 12,))
Bias_entry.grid(row=3, column=1, padx=10, pady=10)
# Definir Boton de Calculo
start_button = ttk.Button(control_frame, text="Calcular", command=percep)
start_button.grid(row=4, column=0, padx=10, pady=200)
# Definir Boton de Limpieza
clean_button = ttk.Button(control_frame, text="Limpiar", command=clean)
clean_button.grid(row=4, column=1, padx=10, pady=200)
# Ciclo principal
mainwindow.protocol("WM_DELETE_WINDOW", on_closing)
# mainwindow.after_idle(check_backup)
mainwindow.mainloop()