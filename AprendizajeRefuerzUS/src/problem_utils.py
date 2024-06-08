import numpy as np
import matplotlib.pyplot as plt

def lee_mapa(fichero):
    with open(fichero,'r') as archivo:
        lineas = archivo.readlines()
    numeros = [float(numero) for numero in lineas[0].split()]
    lineas.pop(0)
    lineas.reverse()
    matriz = []
    for linea in lineas:
        fila = [int(caracter) for caracter in linea.strip()]
        matriz.append(fila)
    return np.array(matriz),(numeros[0],numeros[1])

def visualiza_mapa(mapa,destino):
    plt.figure(figsize=(len(mapa[0]), len(mapa)))
    plt.imshow(1-mapa, cmap='gray', interpolation='none')
    plt.xlim(-0.5, len(mapa[0]) - 0.5) # vemos que se puede omitir
    plt.ylim(-0.5, len(mapa) - 0.5) # vemos que se puede omitir
    plt.gca().add_patch(plt.Rectangle((destino[0] - 0.5, destino[1] - 0.5), 1, 1, edgecolor='black', facecolor='red', lw=5))

def genera_estados(mapa):
    estados = []
    for i in range(0,mapa.shape[1]):
        for j in range(0,mapa.shape[0]):
            estados.append(tuple([i,j]))
    return estados

def es_obstaculo(estado,mapa):
    return mapa[estado[1],estado[0]] == 1

def aplica_accion(estado,accion,mapa):
    if es_obstaculo(estado,mapa):
        return estado
    x = estado[0]
    y = estado[1]
    
    if accion == 'N':
        y += 1
    elif accion == 'S':
        y -= 1
    elif accion == 'E':
        x += 1
    elif accion == 'O':
        x -= 1
    elif accion == 'NE':
        y += 1
        x += 1
    elif accion == 'SE':
        y -= 1
        x += 1
    elif accion == 'SO':
        y -= 1
        x -= 1
    elif accion == 'NO':
        y += 1
        x -= 1
    return x,y

def obtiene_posibles_errores(accion):
    if accion=='N':
        errores = ['NE','NO']
    elif accion=='S':
        errores = ['SE','SO']
    elif accion=='E':
        errores = ['NE','SE']
    elif accion=='O':
        errores = ['NO', 'SO']
    elif accion=='NE':
        errores = ['N','E']
    elif accion=='NO':
        errores = ['N','O']
    elif accion=='SE':
        errores = ['S','E']
    elif accion == 'SO':
        errores = ['S','O']
    else:
        errores = []
    return errores


def obtiene_recompensa(estado,destino,mapa):
    K = 1000
    if es_obstaculo(estado,mapa):
        valor = -K
    else:
        valor = - np.sqrt((estado[0]-destino[0])**2 + (estado[1]-destino[1])**2)
    return valor

def crea_recompensas_sistema(estados,destino,mapa,acciones):
    matriz = []
    for e in estados:
        r = obtiene_recompensa(e,destino,mapa)
        fila = [r]*len(acciones)
        if e != destino:
            fila[0]=-100
        matriz.append(fila)
    return np.array(matriz)

def obtiene_indice_estado(estado,mapa):
    return int(estado[0]*mapa.shape[0]+estado[1])

def crea_transiciones_movimiento(accion, prob_error,estados,mapa):
    matriz = []
    for e0 in estados:
        fila = [0]*len(estados)
        if es_obstaculo(e0,mapa):
            fila[obtiene_indice_estado(e0,mapa)]=1
        else:
            goal = aplica_accion(e0,accion,mapa)
            errores = obtiene_posibles_errores(accion)
            if len(errores)==0:
                fila[obtiene_indice_estado(goal,mapa)] = 1
            else:
                fila[obtiene_indice_estado(goal,mapa)] = 1 - prob_error
                for error in errores:
                    goal_error = aplica_accion(e0,error,mapa)
                    fila[obtiene_indice_estado(goal_error,mapa)] = prob_error/len(errores)
        matriz.append(fila)
    return np.array(matriz)

def visualiza_politica(politica,mapa,destino,estados):
    visualiza_mapa(mapa,destino)
    for p in zip(estados,politica):
        accion = p[1]
        if accion=='esperar':
            continue
        estado = p[0]
        e1 = aplica_accion(estado,accion,mapa)
        x0 = estado[0]
        y0 = estado[1]
        x1 = e1[0]
        y1 = e1[1]
        
        plt.gca().arrow(x0, y0, (x1 - x0)*0.6, (y1 - y0)*0.6,
         head_width=0.3, head_length=0.3, fc='black', ec='black')
        
def crea_politica_greedy(estados,acciones,mapa,destino):
    p = []
    for e in estados:
        valores = []
        for a in acciones:
            e1 = aplica_accion(e,a,mapa)
            valores.append(obtiene_recompensa(e1,destino,mapa))
        accion = acciones[np.argmax(valores)]
        p.append(accion)
    return p
