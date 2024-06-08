import numpy as np
import mdptoolbox.mdp as mdp
from .util import politica_procesable



class Q_Learning:

    """
    Clase que implementa el algoritmo de Q-Learning para la resolución de problemas de aprendizaje por refuerzo.

    Parámetros:
    -----------
    transiciones: List
        Matriz de probabilidades de transición. Cada fila representa un estado y cada columna una acción.

    recompensas: List
        Matriz de recompensas. Cada fila representa un estado y cada columna una acción.
    
    factor_descuento: float
        Factor de descuento. Por defecto es 0.9.

    max_iteraciones: int
        Número máximo de iteraciones. Por defecto es 10000.
    
    Atributos:
    -----------

    modelo: Objeto
        Modelo de Q-Learning que implementa el algoritmo pasando los parámetros especificados.
    
    
    Métodos:
    -----------

    entrenar() -> None
        Entrena el modelo de Q-Learning mediante el metodo run() del objeto modelo.
    
    obtener_politica() -> List
        Devuelve la política óptima obtenida tras haber entrenado el modelo.

    """

    def __init__(self, transiciones, recompensas, factor_descuento=0.9 , max_iteraciones=10000):

        self.transiciones = transiciones

        self.recompensas = recompensas

        self.factor_descuento = factor_descuento

        self.max_iteraciones = max_iteraciones
    

        self.modelo = mdp.QLearning(self.transiciones, self.recompensas, self.factor_descuento, n_iter=self.max_iteraciones)

    def entrenar(self):
        self.modelo.run()
        
    def obtener_politica(self):
        acciones = ['esperar','N','NE','E','SE','S','SO','O','NO']
        return politica_procesable(self.modelo.policy,acciones)
    
    

