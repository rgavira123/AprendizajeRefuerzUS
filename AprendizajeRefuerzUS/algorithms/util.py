import numpy as np
import math as math


def calcula_dimensiones(transiciones):
    """
    Calcula las dimensiones de la matriz de transiciones

    Parametros
    ----------

    transiciones : Array
        Matriz de transiciones
    
    Si la matriz de transiciones es de dimension 3, se asume que la primera dimension es el numero de acciones
    y la segunda dimension es el numero de estados.
    Si la matriz de transiciones es de dimension 2, se asume que la primera dimension es el numero de estados
    """
    acciones = len(transiciones)
    if transiciones.ndim==3:
        estados = transiciones.shape[1]
    else:
        estados = transiciones[0].shape[0]
    return estados, acciones

def politica_procesable(politica,acciones):
    """
    Comprueba si la politica es procesable

    Parametros
    ----------
    politica : Array
        Array con la politica

    acciones : Array
        Array con las acciones posibles
    """
    if isinstance(politica, dict):
        return [acciones[i] for i in politica.values()]
    else:
        return [acciones[i] for i in politica]
    
def obtener_politica_final(politica):
    """
    Obtiene la politica final

    Parametros
    ----------

    politica : Array
        Array con la politica
    """
    
    indices_nav_acciones = {'esperar': 0, 'N': 1, 'NE': 2, 'E': 3, 'SE': 4, 'S': 5, 'SO': 6, 'O': 7, 'NO': 8}
    politica_greedy_indices = [indices_nav_acciones[a] for a in politica]
    politica = enumerate(politica_greedy_indices)
    politica_final = [(i, a) for i,a in politica]
    politica_final = dict(politica_final)
    return politica_final
    

