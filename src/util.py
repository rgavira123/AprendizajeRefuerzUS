import numpy as np
import math as math

def calcula_dimensiones(transiciones):
    acciones = len(transiciones)
    if transiciones.ndim==3:
        estados = transiciones.shape[1]
    else:
        estados = transiciones[0].shape[0]
    return estados, acciones
