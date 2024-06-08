import numpy as np
import mdptoolbox.mdp as mdp
from .util import politica_procesable



class Q_Learning:

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
    
    

