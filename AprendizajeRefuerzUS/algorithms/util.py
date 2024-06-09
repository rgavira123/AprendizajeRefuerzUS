import numpy as np
import math as math


def calcula_dimensiones(transiciones):
    acciones = len(transiciones)
    if transiciones.ndim==3:
        estados = transiciones.shape[1]
    else:
        estados = transiciones[0].shape[0]
    return estados, acciones

def politica_procesable(politica,acciones):
    if isinstance(politica, dict):
        return [acciones[i] for i in politica.values()]
    else:
        return [acciones[i] for i in politica]
    
def obtener_politica_final(politica):
    indices_nav_acciones = {'esperar': 0, 'N': 1, 'NE': 2, 'E': 3, 'SE': 4, 'S': 5, 'SO': 6, 'O': 7, 'NO': 8}
    politica_greedy_indices = [indices_nav_acciones[a] for a in politica]
    politica = enumerate(politica_greedy_indices)
    politica_final = [(i, a) for i,a in politica]
    politica_final = dict(politica_final)
    return politica_final
    

