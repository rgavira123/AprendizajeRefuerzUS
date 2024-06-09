import numpy as np
import AprendizajeRefuerzUS.problem_utils as problem_utils
import matplotlib.pyplot as plt

class Problem:

    """
    Clase que instancia un problema de aprendizaje por refuerzo.

    Parámetros:
    -----------

    mapa: Array
        Matriz que representa el mapa.

    prob_error: float
        Probabilidad de error en el movimiento.

    Atributos:
    -----------

    acciones: List
        Lista de acciones.
    
    destino: Tuple
        Coordenadas del destino.
    
    estados: List
        Lista de estados.
 
    politica: Dict
        Política óptima.

    recompensas: Array
        Matriz de recompensas.
    
    transiciones: Array
        Matriz de probabilidades de transición.
    
    Métodos:
    -----------

    visualiza_mapa()
        Muestra el mapa del problema a partir de un mapa y un destino.
    
    crea_recompensas_sistema()
        Crea una matriz de recompensas para un sistema a partir de una lista de estados, un destino, un mapa y una lista de acciones.
    
    crea_transiciones_sistema(prob_error)
        Crea una matriz de transiciones para un sistema a partir de una probabilidad de error.
        Las distintas transiciones se generan a partir de:
            - Una acción
            - Una probabilidad de error
            - Una lista de estados
            - Un mapa.
    
    actualiza_politica(politica)
        Actualiza la política del problema.
    
    visualiza_politica()
        Muestra la política del problema.


    """
    def __init__(self, mapa, prob_error):
        self.mapa = mapa
        self.prob_error = prob_error

        self.mapa,self.destino = problem_utils.lee_mapa(mapa)
        self.estados = problem_utils.genera_estados(self.mapa)
        self.acciones = ['esperar','N','NE','E','SE','S','SO','O','NO']

        self.recompensas = self.crea_recompensas_sistema()
        self.transiciones = self.crea_transiciones_sistema(prob_error)

        self.politica = problem_utils.crea_politica_greedy(self.estados,self.acciones,self.mapa,self.destino)
    

    def visualiza_mapa(self):
        problem_utils.visualiza_mapa(self.mapa,self.destino)

    def crea_recompensas_sistema(self):
        return problem_utils.crea_recompensas_sistema(self.estados,self.destino,self.mapa,self.acciones)
    
    def crea_transiciones_sistema(self, prob_error):
        return np.array([problem_utils.crea_transiciones_movimiento('esperar',prob_error,self.estados,self.mapa), 
                     problem_utils.crea_transiciones_movimiento('N',prob_error,self.estados,self.mapa),
                     problem_utils.crea_transiciones_movimiento('NE',prob_error,self.estados,self.mapa),
                     problem_utils.crea_transiciones_movimiento('E',prob_error,self.estados,self.mapa),
                     problem_utils.crea_transiciones_movimiento('SE',prob_error,self.estados,self.mapa),
                     problem_utils.crea_transiciones_movimiento('S',prob_error,self.estados,self.mapa),
                     problem_utils.crea_transiciones_movimiento('SO',prob_error,self.estados,self.mapa),
                     problem_utils.crea_transiciones_movimiento('O',prob_error,self.estados,self.mapa),
                     problem_utils.crea_transiciones_movimiento('NO',prob_error,self.estados,self.mapa)])

    def actualiza_politica(self, politica):
        self.politica = politica
    
    def visualiza_politica(self):
        problem_utils.visualiza_politica(self.politica,self.mapa,self.destino,self.estados)
        