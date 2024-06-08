

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

from .util import politica_procesable
from .util import calcula_dimensiones

class MonteCarlo:

    def __init__(self, transiciones, recompensas, politica0=None, factor_descuento=0.9, max_iteraciones=1000):
        self.transiciones = transiciones # Matriz de transición
        self.recompensas = recompensas # Recompensas
        self.factor_descuento = factor_descuento # Factor de descuento
        assert 0.0 < self.factor_descuento <= 1.0, "El valor del descuento debe estar entre 0 y 1"
        self.max_iteraciones = max_iteraciones # Número máximo de iteraciones
        self.racum = defaultdict(lambda: defaultdict(list)) # Recompensas acumuladas
        self.estados, self.acciones = calcula_dimensiones(transiciones) # Calculamos las dimensiones del problema
        self.racum = {s: {a: [] for a in range(self.acciones)} for s in range(self.estados)}
        if politica0 is None:
            self.politica = {s: np.random.choice(self.acciones) for s in range(self.estados)} # Política inicial
        else :
            self.politica = politica0
        self.q = {s: {a: 0.0 for a in range(self.acciones)} for s in range(self.estados)} # Función de valor Qi(s, a)


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
        #visitados = set()  # Conjunto de tuplas (estado, acción)
        episodio = [] # Lista de tuplas (estado, acción, recompensa)
        estado = self.estado_aleatorio_no_terminal() # Estado inicial
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
        return politica_procesable(self.politica,acciones)


        

        




