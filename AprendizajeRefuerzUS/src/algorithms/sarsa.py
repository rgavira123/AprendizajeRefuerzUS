import numpy as np
import pandas as pd
import math as math
from .util import calcula_dimensiones
from .util import politica_procesable

class SARSA(object):

    """
    Clase que implementa el algoritmo de SARSA para la resolución de problemas de aprendizaje por refuerzo.

    Parámetros:
    -----------
    transiciones: List
        Matriz de probabilidades de transición. Cada fila representa un estado y cada columna una acción.
    
    recompensas: List
        Matriz de recompensas. Cada fila representa un estado y cada columna una acción.
    
    factor_descuento: float
        Factor de descuento. Por defecto es 0.9.
    
    factor_aprendizaje: float
        Factor de aprendizaje. Por defecto es 0.5.

    max_iteraciones: int
        Número máximo de iteraciones. Por defecto es 1000.
    
    epsilon: float
        Probabilidad de exploración. Por defecto es 0.1.
    
    Atributos:
    -----------

    estados: int
        Número de estados del problema.
    
    acciones: int
        Número de acciones del problema.

    Q: Dict
        Diccionario que almacena los valores de la función Q para cada par (estado, acción).
    
    Métodos:
    -----------

    es_terminal(estado) -> bool
        Comprueba si un estado es terminal. Consideramos que un estado es terminal si la suma de las recompensas asociadas a cada acción es 0.

    seleccionar_accion(estado) -> int
        Selecciona una acción en un estado dado siguiendo una política epsilon-greedy, es decir, 
        con probabilidad 1-epsilon elige la acción con mayor valor Q y con probabilidad epsilon elige una acción aleatoria.
    
    siguiente_estado(estado, accion) -> int
        Devuelve el siguiente estado tras ejecutar una acción en un estado dado, en este caso la eleccion del siguiente estado se ve afectada por la probabilidad de transición.
        Cuanto mayor sea la probabilidad de transición, mayor será la probabilidad de que el siguiente estado sea el estado correspondiente a esa probabilidad.
    
    recompensa_accion(estado, accion) -> float
        Devuelve la recompensa asociada a una acción en un estado dado.
    
    entrenar() -> Dict
        Entrena el algoritmo de SARSA y devuelve la función Q aprendida. Con un criterio de parada basado en el número de iteraciones o 
        cuando el contador sea múltiplo de 200, es decir, cada 200 iteraciones se detiene el entrenamiento para evitar bucles infinitos.
    
    obtener_politica() -> List
        Devuelve la política óptima aprendida a partir de la función Q.
    """

    def __init__(self, transiciones, recompensas, factor_descuento=0.9, factor_aprendizaje=0.5, max_iteraciones=1000, epsilon=0.1):

        self.transiciones = transiciones
        self.recompensas = recompensas

    
        self.factor_descuento = factor_descuento
        assert 0.0 < self.factor_descuento <= 1.0, "El valor del descuento debe estar entre 0 y 1"

        self.factor_aprendizaje = factor_aprendizaje
        assert 0.0 < self.factor_aprendizaje <= 1.0, "El valor del factor de aprendizaje debe estar entre 0 y 1"

        self.max_iteraciones = int(max_iteraciones)
        assert self.max_iteraciones > 0, "El número máximo de iteraciones debe ser un entero positivo"

        self.epsilon = epsilon
        assert 0.0 < self.epsilon <= 1.0, "El valor de epsilon debe estar entre 0 y 1"

        self.estados,self.acciones = calcula_dimensiones(transiciones)

        self.Q = {s: {a: 0.0 for a in range(self.acciones)} for s in range(self.estados)}


    def es_terminal(self, estado):
        # el estado es un índice, de 0 a 756, entonces yo miro las recompensas en el indice de estado self.recompensas[estado]
        # que es una lista de recompensas por accion, si la suma de 0, es terminal:

        return sum(self.recompensas[estado]) == 0
    

    def seleccionar_accion(self,estado):
        # con probabilidad 1 - self.epsilon eligo la accion con mayor Q en el diccionario
        # es decir Q[estado][accion] para accion en acciones, y cojo el maximo
        # con probabilidad epsilon, una accion aleatoria de las accione posibles self.acciones

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.acciones)
        else:
            return max(self.Q[estado], key=self.Q[estado].get)
        
    def siguiente_estado(self,estado,accion):

        matriz_accion_elegida = self.transiciones[accion]

        # ahora me quiero quedar con la fila del estado elegido, y quiero muestrear de ella
        # para obtener el siguiente estado
        # matriz_accion_elegida[estado] es la fila de la matriz de transicion que me interesa
        # y quiero muestrear de ella, en ella tenemos una distribucion de probabilidades, con valoes de 0 a 1 en
        # cada columna, que representan la probabilidad de ir a cada estado dado que estoy en el estado actual
        '''
        un ejemplo sería el siguiente, yo he hecho la accion 1 y estoy en el estado 35
        entonces haria matriz_accion_elegida = self.transiciones[1]
        y en la fila 35 tengo algo así

        [0,0,0,0.2,0,0,0,0,0,0,0.5,0,0,0,0,0.3] pero con 765 elementos

        entonces lo que hay dentro es una posibilidad y quiero que me devuelva el indice del estado al que voy

        '''

        posibles_siguientes_estados = matriz_accion_elegida[estado]
        return np.random.choice(range(self.estados), p=posibles_siguientes_estados)
    
    def recompensa_accion(self,estado,accion):
        return self.recompensas[estado][accion]

    def entrenar(self):
        entero = 0
        for _ in range(self.max_iteraciones+1):
            estado = np.random.choice(range(self.estados))
            while(self.es_terminal(estado)):
                estado = np.random.choice(range(self.estados))
            while not self.es_terminal(estado):
                accion = self.seleccionar_accion(estado)
                siguiente_estado = self.siguiente_estado(estado,accion)  
                recompensa = self.recompensa_accion(estado,accion)
                accion_prima = self.seleccionar_accion(siguiente_estado)
                self.Q[estado][accion] = self.Q[estado][accion] + self.factor_aprendizaje*(recompensa + self.factor_descuento*self.Q[siguiente_estado][accion_prima] - self.Q[estado][accion])
                estado = siguiente_estado
                accion = accion_prima
                entero += 1
                if(entero % self.max_iteraciones*0.1 == 0):
                    break

            
    
    def obtener_politica(self):
        politica = {s: max(self.Q[s], key=self.Q[s].get) for s in range(self.estados)}
        acciones = ['esperar','N','NE','E','SE','S','SO','O','NO']
        return politica_procesable(politica,acciones)


                
            


