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
