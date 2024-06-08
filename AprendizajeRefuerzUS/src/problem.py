import numpy as np
import problem_utils as problem_utils
import matplotlib.pyplot as plt

class Problem:
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
        