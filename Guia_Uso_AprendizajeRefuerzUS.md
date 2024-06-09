# GUÍA DE USO DE LA BIBLIOTECA APRENDIZAJE REFUERZUS

El objetivo de este documento es describir los pasos necesarios para utilizar los algoritmos:

    - Q_Learning
    - Monte Carlo
    - SARSA

## Instalación

Todos los pasos que se describen a continuación y las pruebas de los algoritmos se aconsejan realizarlos en un modulo .ipynb.

Para realizar las pruebas de los algoritmos se deberan seguir los siguientes pasos:

1. Descargar el tar.gz que se encuentra en la carpeta `dist` del repositorio.

2. Descomprimir el archivo en el directorio de trabajo deseado.

3. En tu direcotrio de trabajo, create un archivo .ipynb y añadir el tar.gz

4. Instalaremos la biblioteca en nuestro entorno con el siguiente comando:

```python
pip install AprendizajeRefuerzUS-0.1.0.tar.gz (ruta al .tar.gz)
```

> **Nota:** Si quieres comprobar que la biblioteca se ha instalado correctamente, puedes ejecutar estos comandos:

```python
pip show AprendizajeRefuerzUS
```
```python
pip list 
```


5. Importar la biblioteca en tu archivo .ipynb con el siguiente comando:

```python
import AprendizajeRefuerzUS
```

6. Ahora ya puedes utilizar los algoritmos de la biblioteca.

## Problema

Nosotros partimos de un problema, definido por un mapa y una tasa de error. El mapa es una matriz de 0s y 1s, donde los 0s representan los caminos por los que se puede mover el agente y los 1s representan las paredes. La tasa de error es la probabilidad de que el agente se mueva en una dirección diferente a la que ha elegido.

Para definir un problema, se deberá seguir los siguientes pasos:

1. Importar el modulo `problem` de la biblioteca AprendizajeRefuerzUS.

```python
from AprendizajeRefuerzUS import problem as prob
```

A partir de este problema (que es siempre el mismo, solo cambia el mapa y la tasa de error) podremos utilizar los algoritmos de la biblioteca AprendizajeRefuerzUS.

### Instanciar un problema

Para instanciar un problema, se deberá seguir los siguientes pasos:

1. Crear una instancia de la clase `Problem` con el mapa y la tasa de error deseados. 
En este caso proporcionamos uan serie de mapas predefinidos que se pueden utilizar, en el caso de que quieras utilizar un mapa personalizado, deberás proporcionar una matriz de 0s y 1s.

```python

# Mapa predefinido
import pkg_resources
from AprendizajeRefuerzUS import problem as prob

# Ruta relativa dentro del paquete
map_path = pkg_resources.resource_filename('AprendizajeRefuerzUS', 'maps/map1.txt')

# Instanciamos el problema

problem = prob.Problem(map_path, 0.2)
```

> **Nota:** Si quieres visualizar el mapa, puedes ejecutar el siguiente comando:

```python
problem.visualiza_mapa()
```

2. El problema por defecto usa una politica greedy, si quieres visualizarla puedes ejecutar el siguiente comando:

```python
problem.visualiza_politica()
```

## Algoritmos

Una vez ya tenemos el problema instanciado, podemos utilizar los algoritmos de la biblioteca AprendizajeRefuerzUS.

Antes de utilizar los algoritmos, deberemos inicializar la matriz de transición y la matriz de recompensas. 

```python
transiciones = problem.transiciones
recompensas = problem.recompensas
```


### Monte Carlo

Para la implementación de Monte Carlo, se deberán seguir los siguientes pasos:

1. Importar el modulo `MonteCarlo` de la biblioteca AprendizajeRefuerzUS.

```python
from AprendizajeRefuerzUS.algorithms import MonteCarlo as mc # Lo llamamos mc para mayor comodidad en el futuro
```

2. Crear una instancia de la clase `MonteCarlo` con las matrices de transición y recompensas.

```python
modelo_mc = mc.MonteCarlo(transiciones, recompensas, max_iteraciones=10000)
```

>**Nota:** Si no se especifica la politica de exploración, se escogerá una politica aleatoria. En el caso de que queramos especificar una politica de exploración, deberemos hacerlo de la siguiente manera:

```python

## Normalmente la política nos entra de la forma [esperar,esperar,norte,...] pero esto no es procesable por el algoritmo, debe ser un diccionario, para ello podemos usar este método (por defecto hemos resuelto con greedy)

from AprendizajeRefuerzUS.algorithms import util

politica_greedy = problem.politica

politica_procesable = util.obtener_politica_final(politica_greedy)

## Ahora ya podemos instanciar el algoritmo con la politica de exploración

modelo_mc = mc.MonteCarlo(transiciones, recompensas,politica0=politica_procesable max_iteraciones=10000)
```

3. Entrenar el modelo, dependiendo de si queremos utilizar Montecarlo de primera visita o de cada visita, deberemos hacerlo de la siguiente manera:

```python
modelo_mc.entrenar_primera_visita()
modelo_mc.entrenar_cada_visita()
```

4. Una vez entrenado el modelo, podemos obtener la politica óptima con el siguiente comando:

```python
politica_mc = modelo_mc.obtener_politica()
problem.actualiza_politica(politica_mc)
problem.visualiza_politica()
```

Estos serian los pasos a seguir para utilizar el algoritmo de Monte Carlo.

### Q-Learning

Para la implementación de Q-Learning, se deberán seguir los siguientes pasos:

1. Importar el modulo `QLearning` de la biblioteca AprendizajeRefuerzUS.

```python
from AprendizajeRefuerzUS.algorithms import Q_Learning as ql # Lo llamamos ql para mayor comodidad en el futuro
```

2. Crear una instancia de la clase `Q_Learning` con las matrices de transición y recompensas.

```python
modelo_ql = ql.Q_Learning(transiciones, recompensas, max_iteraciones=10000)
```

3. Entrenar el modelo, deberemos hacerlo de la siguiente manera:

```python
modelo_ql.entrenar()
```

4. Una vez entrenado el modelo, podemos obtener la politica óptima con el siguiente comando:

```python
politica_qlearning = modelo_qlearning.obtener_politica()
problem.actualiza_politica(politica_qlearning)
problem.visualiza_politica()
```

Estos serian los pasos a seguir para utilizar el algoritmo de Q-Learning.

### SARSA

Para la implementación de SARSA, se deberán seguir los siguientes pasos:

1. Importar el modulo `SARSA` de la biblioteca AprendizajeRefuerzUS.

```python
from AprendizajeRefuerzUS.algorithms import sarsa as sarsa # Lo llamamos sarsa para mayor comodidad en el futuro
```

2. Crear una instancia de la clase `SARSA` con las matrices de transición y recompensas.

```python
modelo_sarsa = sarsa.SARSA(transiciones, recompensas, max_iteraciones=10000)
```

3. Entrenar el modelo, deberemos hacerlo de la siguiente manera:

```python
modelo_sarsa.entrenar()
```

4. Una vez entrenado el modelo, podemos obtener la politica óptima con el siguiente comando:

```python
politica_sarsa = modelo_sarsa.obtener_politica()
problem.actualiza_politica(politica_sarsa)
problem.visualiza_politica()
```

Estos serian los pasos a seguir para utilizar el algoritmo de SARSA.



