

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from .util import politica_procesable
from .util import calcula_dimensiones

class MonteCarlo:

    """
    Clase que implementa el algoritmo de Monte Carlo para la resolución de problemas de aprendizaje por refuerzo.

    Parámetros:
    -----------
    
    transiciones: array
        Matriz de probabilidades de transición. Cada fila representa un estado y cada columna una acción.
    
    recompensas: array
        Matriz de recompensas. Cada fila representa un estado y cada columna una acción.

    politica0: dict
        Política inicial. Si no se especifica, se inicializa aleatoriamente.
    
    factor_descuento: float
        Factor de descuento. Por defecto es 0.9.

    max_iteraciones: int 
        Número máximo de iteraciones. Por defecto es 1000.
    
    Atributos:
    -----------

    estados: int
        Número de estados del problema.

    acciones: int
        Número de acciones del problema.

    racum: Dict
        Diccionario que almacena las recompensas acumuladas por cada par (estado, acción).

    q: Dict
        Diccionario que almacena los valores de la función Q para cada par (estado, acción).

    Métodos:
    -----------

    es_terminal(estado) -> bool
        Comprueba si un estado es terminal. Consideramos que un estado es terminal si la suma de las recompensas asociadas a cada acción es 0.
    
    siguiente_estado(estado, accion) -> int
        Devuelve el siguiente estado tras ejecutar una acción en un estado dado, en este caso la eleccion del siguiente estado se ve afectada por la probabilidad de transición.
        Cuanto mayor sea la probabilidad de transición, mayor será la probabilidad de que el siguiente estado sea el estado correspondiente a esa probabilidad.
    
    recompensa_accion(estado, accion) -> float
        Devuelve la recompensa asociada a una acción en un estado dado.
    
    estado_aleatorio_no_terminal() -> int
        Devuelve un estado aleatorio no terminal y realiza una comprobación para asegurarse de que el estado no es terminal.
    
    generar_episodio() -> List
        Genera un episodio siguiendo la política actual. Un episodio es una lista de tuplas de la forma (estado, acción, recompensa).
        Comienza en un estado aleatorio no terminal y termina cuando se alcanza un estado terminal, cuando se supera el número máximo de iteraciones o 
        cuando el contador supera el 10% del número máximo de iteraciones para evitar bucles infinitos.
    
    entrenar_primera_visita() -> None
        Entrena el algoritmo de Monte Carlo con el método de primera visita. Se generan episodios y se actualizan los valores de la función Q y la política 
        solo si el par (estado, acción) no ha sido visitado previamente en el episodio.
    
    entrenar_cada_visita() -> None
        Entrena el algoritmo de Monte Carlo con el método de cada visita. Se generan episodios y se actualizan los valores de la función Q y la política 
        en cada visita.
    
    obtener_politica() -> List
        Devuelve la política óptima obtenida tras entrenar el algoritmo de Monte Carlo. La política se devuelve en un formato procesable por la función 
        politica_procesable().
    """


    
    def __init__(self, transiciones, recompensas, politica0=None, factor_descuento=0.9, max_iteraciones=1000):


        self.transiciones = transiciones 
        self.recompensas = recompensas 

        self.estados, self.acciones = calcula_dimensiones(transiciones) 
        
        if politica0 is None:
            self.politica = {s: np.random.choice(self.acciones) for s in range(self.estados)} 
        else :
            self.politica = politica0

        self.factor_descuento = factor_descuento 
        assert 0.0 < self.factor_descuento <= 1.0, "El valor del descuento debe estar entre 0 y 1"

        self.max_iteraciones = max_iteraciones 
        

        self.racum = {s: {a: [] for a in range(self.acciones)} for s in range(self.estados)}


        self.q = {s: {a: 0.0 for a in range(self.acciones)} for s in range(self.estados)} 


    def es_terminal(self, estado):
    
        return sum(self.recompensas[estado]) == 0
 

    def siguiente_estado(self,estado,accion):
        matriz_accion_elegida = self.transiciones[accion]
        posibles_siguientes_estados = matriz_accion_elegida[estado]
        return np.random.choice(range(self.estados), p=posibles_siguientes_estados)
    

    def recompensa_accion(self,estado,accion):
        return self.recompensas[estado][accion]
       

    def estado_aleatorio_no_terminal(self):
        estado = np.random.choice(range(self.estados))
        while self.es_terminal(estado):
            estado = np.random.choice(range(self.estados))
        return estado
    

    def generar_episodio(self):
        episodio = [] # Lista de tuplas (estado, acción, recompensa)
        estado = self.estado_aleatorio_no_terminal() # Estado inicial
        while(self.es_terminal(estado)):
            estado = np.random.choice(range(self.estados))
        accion= np.random.choice(range(self.acciones))
        recompensa = self.recompensas[estado][accion]
        contador = 0

        while not self.es_terminal(estado):
            episodio.append((estado, accion, recompensa))
            estado = self.siguiente_estado(estado, accion)
            accion = self.politica[estado]
            recompensa = self.recompensas[estado][accion]
            contador += 1

            if(contador > self.max_iteraciones * 0.1):
                break

        return episodio


    def entrenar_primera_visita(self):
        for _ in range(self.max_iteraciones + 1):
            episodio = self.generar_episodio()
            visitados = set()
            for t in range(len(episodio)):
                estado, accion, reward = episodio[t]
                if (estado, accion) not in visitados:
                    visitados.add((estado, accion))
                    U = sum([self.factor_descuento**(i-t)*reward for i in range(t, len(episodio))])
                    self.racum[estado][accion].append(U)
                    self.q[estado][accion] = np.mean(self.racum[estado][accion])
                    self.politica[estado] = max(self.q[estado], key=self.q[estado].get)


    def entrenar_cada_visita(self):
        for _ in range(self.max_iteraciones + 1):
            episodio = self.generar_episodio()
            for t in range(len(episodio)):
                estado, accion, reward = episodio[t]
                U = sum([self.factor_descuento**(i-t)*reward for i in range(t, len(episodio))])
                self.racum[estado][accion].append(U)
                self.q[estado][accion] = np.mean(self.racum[estado][accion])
                self.politica[estado] = max(self.q[estado], key=self.q[estado].get)
        

    def obtener_politica(self):
        acciones = ['esperar','N','NE','E','SE','S','SO','O','NO']
        return politica_procesable(self.politica, acciones)


        

        




