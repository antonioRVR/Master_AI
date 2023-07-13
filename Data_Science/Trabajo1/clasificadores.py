# Inteligencia Artificial para la Ciencia de los Datos
# Implementación de clasificadores 
# Dpto. de C. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================

# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Vázquez Ramírez
# NOMBRE: Antonio Ramón
#
# Segundo componente (si se trata de un grupo):
#
# APELLIDOS: Rodríguez García 
# NOMBRE: Rafael
# ----------------------------------------------------------------------------


import numpy as np
import pandas as pd
import random as rd


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo
# que debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite, pero NO AL
# NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED, DE HERRAMIENTAS DE GENERACIÓN DE CÓDIGO o cualquier otro medio, 
# se considerará plagio. Si tienen dificultades para realizar el ejercicio, 
# consulten con el profesor. En caso de detectarse plagio, supondrá 
# una calificación de cero en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************

# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.
# NOTAS: 
# * En este trabajo NO se permite usar Scikit Learn, EXCEPTO algunos métodos 
#   que se indican exprésamente en el enunciado. En particular no se permite
#   usar ningún clasificador de Scikit Learn.  
# * Supondremos que los conjuntos de datos se presentan en forma de arrays numpy, 
#   Se valorará el uso eficinte de numpy. 

print(" CLASIFICADOR NAIVE BAYES \n")
# ====================================================
# PARTE I: IMPLEMENTACIÓN DEL CLASIFICADOR NAIVE BAYES
# ====================================================

# Se pide implementar el clasificador Naive Bayes, en su versión categórica
# con suavizado y log probabilidades (descrito en el tema 2, diapositivas 22 a
# 34). En concreto:

# ----------------------------------
# I.1) Implementación de Naive Bayes
# ----------------------------------

# Definir una clase NaiveBayes con la siguiente estructura:
class NaiveBayes():
    
    ##Inicializar el constructor
    def __init__(self, k = 1):
        self.k = k 
        self.entrenamiento = False # Variable lógica (flag). Indica si se ha entrenado el modelo
    def entrena(self, X,y):
        ## Para extraer los atributos y las clases podemos usar unique de numpy 
        ## Asi obtenemos un array sin elementos repetidos
        atributos = np.unique(X)
        clases = np.unique(y)
        self.atributos = atributos
        self.clases = clases
        d_ej = np.shape(X) [1] # Dimensión de un vector ejemplo de atributos del conjunto de entrenamiento
        ## Entrenar el NaiveBayes significa estimar las probabilidades de cada clase (frecuencia relativa)
        ## y las probabilidades de que se de un atributo dado que tenemos una clase c P(A=v|c)

        ## Se crea para cada clase un vector lógico que indica cúantos elementos hay de dicha clase. 
        # Contando los elementos no nulos obtenemos la cantidad de elementos de la clase ya que el elemento 
        # nulo es el que no pertenece a la clase y por tanto no se cuenta.
        
        frec_clases = [np.count_nonzero(y == v) for v in clases]
        frec_clases = np.array(list(frec_clases)) # Convertir objeto map a numpy array 
        #Por lo tanto esto nos dará los elementos de cada clase, es decir su frecuencia

        P_clases = frec_clases/(len(y)) ## Para obtener prob hay que normalizar. Dividimos el número de elementos de cada clase entre el total de elementos
                                        # y así obtenemos la probabilidad de cada clase 

        ## Ahora hay que estimar la probabilidad de cada atributo dado una clase
        ## Creamos un array tridimensional de numpy para almacenar los valores de las probabilidades.
        ## Para cada clase tenemos un array bidimensional con las probabilidades.
        P_atributos = np.zeros([len(clases), len(atributos), d_ej]) # Por orden : n clases , n atributos, y dimensión de un ejemplo del conjunto de entrenamiento
        ## Para cada clase cogemos un atributo y se cuentan las veces que aparece en cada columna del conjunto de entrenamiento
        ## esto lo hacemos porque por ejemplo en el ejemplo del tenis hay atributos de distintas columnas con el mismo nombre ( Humedad -> alta , Temp -> alta )

        i = 0 ## Para indicar los indices
        for t in clases:
            ## Con un bucle recorremos los distintos valores de las clases
            ## Miramos sólo los ejemplos que correspondan a la clase en cada iter:  X[y == t,]
            ## Para contar los valores de atributos es igual que antes con las clases , ponemos axis =0 para que cuente 
            ## la frecuencia de cada valor en cada uno de los atributos ( P ej n de apariciones de soleado en el primera columna, segunda...)
            frec_atributos = map(lambda v: np.count_nonzero( (X[y == t, ] == v) , axis = 0), atributos )
            frec_atributos=np.array(list(frec_atributos)) # lo pasamos a array de numpy
            P_atributos[i, :, :] = frec_atributos
            i += 1 # Actualiza indice
        
       
        # Tenemos que calcular probabilidades utilizando suavizado de Laplace.
        # vamos a crear un vector auxiliar
        P_aux=P_atributos
        P_aux=np.sum(P_aux, axis = 0) ## Sumamos los arrays bidimensionales para quedarnos con una sola matriz. En las posiciones no nulas hay que aplicar suavizado
        P_aux= P_aux > 0  # creamos una máscara que nos dice en que posiciones hay que aplicar Suavizado 
        
        #Calculamos el número de posibles valores para cada atributo. Contamos los valores no nulos para cada columna de la máscara 
        A= np.count_nonzero(P_aux, axis = 0)
      
        P_atributos = P_atributos + self.k * P_aux
        for t in range(len(clases)):
            P_atributos[t, :, :] = P_atributos[t, :, :] / [frec_clases[t] + self.k*A] 
        
        self.P_atributos = P_atributos
        self.P_clases = P_clases
        P_atributos_df = pd.DataFrame(P_atributos[0,:,:], index= atributos)
        self.entrenamiento = True # Entrenamiento realizado
        return P_atributos, P_clases
        
        ## Metodo clasifica
    def clasifica_prob(self, ejemplo):

        if not self.entrenamiento:
            raise ClasificadorNoEntrenado('No se ha entrenado el modelo')
        # Hay que ver los atributos del ejemplo en qué posicion estan en nuestro 
        # array de probabilidades
        d_ej = np.shape(ejemplo)[0]
        d_clases = len(self.clases)

        pos = [np.where(self.atributos == v)[0][0] for v in ejemplo]
        # Con pos podemos acceder a las posiciones de la matriz de prob.


        # Ahora Para cada clase calculamos las logprob
        suma_logprob=np.zeros(d_clases)
        dic_prob= dict.fromkeys(self.clases) # Creamos diccionario

        for t in range(len(self.clases)):
            suma_logprob[t]= np.log(self.P_clases[t]) + np.sum(np.log(self.P_atributos[t, pos, range(d_ej)]))
            dic_prob[self.clases[t]]=np.exp(suma_logprob[t])
        
        # Normalizar los valores del diccionario
        cte_norm = 1/np.sum(np.exp(suma_logprob))
        dic_prob.update((key,v*cte_norm) for key,v in dic_prob.items())
        self.dic_prob=dic_prob

        #resultado
        return dic_prob
    
    ## Metodo Clasifica. Se le da un ejemplo y devuelve su clase.
    def clasifica(self,ejemplo):
        ## Llamamos a la clase anterior para obtener el diccionario
        diccionario = self.clasifica_prob(ejemplo)
        return max(diccionario , key=self.dic_prob.get)
# * El constructor recibe como argumento la constante k de suavizado (por
#   defecto 1) 
# * Método entrena, recibe como argumentos dos arrays de numpy, X e y, con los
#   datos y los valores de clasificación respectivamente. Tiene como efecto el
#   entrenamiento del modelo sobre los datos que se proporcionan.  
# * Método clasifica_prob: recibe un ejemplo (en forma de array de numpy) y
#   devuelve una distribución de probabilidades (en forma de diccionario) que
#   a cada clase le asigna la probabilidad que el modelo predice de que el
#   ejemplo pertenezca a esa clase. 
# * Método clasifica: recibe un ejemplo (en forma de array de numpy) y
#   devuelve la clase que el modelo predice para ese ejemplo.   

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

  
# Ejemplo "jugar al tenis":


# >>> nb_tenis=NaiveBayes(k=0.5)
# >>> nb_tenis.entrena(X_tenis,y_tenis)
# >>> ej_tenis=np.array(['Soleado','Baja','Alta','Fuerte'])
# >>> nb_tenis.clasifica_prob(ej_tenis)
# {'no': 0.7564841498559081, 'si': 0.24351585014409202}
# >>> nb_tenis.clasifica(ej_tenis)
# 'no'

# ----------------------------------------------
# I.2) Implementación del cálculo de rendimiento
# ----------------------------------------------

def rendimiento(clasificador, X,y):
    # Si el clasificador esta entrenado , habrá que ver para todos los valores si hay una clasificacion correcta
    d_y= len(y) # Numero de ejemplos

    y_p = map(clasificador.clasifica, X) # A todos los ejemplos se le aplica el metodo clasifica
    y_p = np.array( list(y_p) ) # Convertimos a array de numpy 

    # Comparar y_p co(n y
    aciertos = np.sum(y_p == y)
    
    ## Accuracy = aciertos / n_ejemplos 
    accuracy= aciertos / d_y
    return accuracy 

# Definir una función "rendimiento(clasificador,X,y)" que devuelve la
# proporción de ejemplos bien clasificados (accuracy) que obtiene el
# clasificador sobre un conjunto de ejemplos X con clasificación esperada y. 

# Ejemplo:

# >>> rendimiento(nb_tenis,X_tenis,y_tenis)
# 0.9285714285714286

# --------------------------
# I.3) Aplicando Naive Bayes
# --------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Concesión de prestamos
# - Críticas de películas en IMDB (ver NOTA con instrucciones para obtenerlo)

# En todos los casos, será necesario separar un conjunto de test para dar la
# valoración final de los clasificadores obtenidos. Si fuera necesario, se permite usar
# train_test_split de Scikit Learn, para separar el conjunto de test y/o
# validación. Ajustar también el valor del parámetro de suavizado k. 

# Mostrar el proceso realizado en cada caso, y los rendimientos obtenidos. 


def divide_entrenamiento_prueba(X,y,porcentaje_entrenamiento):
    #unimos los atributos y las clases para poder hacer la permutación de forma conjunta y que no se desordenen los datos
    # esto es con el fin de que no se desordenen los datos y se mantenga la correspondencia entre los datos y sus clases
    conjunto=np.column_stack((X,y))
    rd.shuffle(conjunto) #permutamos los datos aleatoriamente
    #volvemos a separar los atributos y las clases
    X=conjunto[:,:-1]
    y=conjunto[:,-1] 
    #dividimos los datos en entrenamiento y prueba
    Xe=X[:int(len(X)*porcentaje_entrenamiento)]
    Xt=X[int(len(X)*(porcentaje_entrenamiento)):]
    ye=y[:int(len(y)*porcentaje_entrenamiento)]
    yt=y[int(len(y)*(porcentaje_entrenamiento)):]
    return Xe,Xt,ye,yt


#-----------------------------------------------------------------
# Votos de congresistas
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Votos de congresistas
#-----------------------------------------------------------------

print("Votos de congresistas US:")

# usamos la libreria importlib.util para poder importar el módulo votos.py
# que se encuentra en la carpeta data
import importlib.util
archivo_votos = 'data.votos'

# Importa el módulo "votos.py"
votos = importlib.import_module(archivo_votos)

# guardamos los arrays de votos.py en variables para poder usarlos en el archivo actual
votos_datos = votos.datos
votos_clasif=votos.clasif

#dividimos los datos en entrenamiento y prueba
Xe_votos,Xt_votos,ye_votos,yt_votos=divide_entrenamiento_prueba(votos_datos,votos_clasif,0.7)

# Entrenamos un modelo NB
NB_votos = NaiveBayes()
NB_votos.entrena(Xe_votos, ye_votos)
print('El rendimiento en el conjunto de entrenamiento')
print(rendimiento(NB_votos, Xe_votos, ye_votos))
print('El rendimiento en el conjunto de test')
print(rendimiento(NB_votos, Xt_votos, yt_votos))


# --------------------------------------------------------
# Conjunto de datos de credito
# --------------------------------------------------------

print('Conjunto de datos de crédito:')

archivo_credito = 'data.credito'
credito = importlib.import_module(archivo_credito)

X_credito = credito.X_credito
y_credito = credito.y_credito 

#dividimos los datos en entrenamiento y prueba
Xe_credito,Xt_credito,ye_credito,yt_credito=divide_entrenamiento_prueba(X_credito, y_credito ,0.7)

# Entrenamos un modelo NB
NB_credito = NaiveBayes()
NB_credito.entrena(Xe_credito, ye_credito)


print('El rendimiento en el conjunto de entrenamiento')
print(rendimiento(NB_credito, Xe_credito, ye_credito))
print('El rendimiento en el conjunto de test')
print(rendimiento(NB_credito, Xt_credito, yt_credito))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NOTA:
# INSTRUCCIONES PARA OBTENER EL CONJUNTO DE DATOS IMDB A USAR EN EL TRABAJO

# Este conjunto de datos ya se ha usado en un ejercicio del tema de modelos 
# probabilísticos. Los textos en bruto y comprimidos están en aclImdb.tar.gz, 
# que se ha de descomprimir previamente (NOTA: debido a la gran cantidad de archivos
# que aparecen al descomprimir, se aconseja pausar la sincronización si se está conectado
# a algún servicio en la nube).

# NO USAR TODO EL CONJUNTO: extraer, usando random.sample, 
# 2000 críticas en el conjunto de entrenamiento y 400 del conjunto de test. 
# Usar por ejemplo la siguiente secuencia de instrucciones, para extraer los textos:


# >>> import random as rd
# >>> from sklearn.datasets import load_files
# >>> reviews_train = load_files("data/aclImdb/train/")
# >>> muestra_entr=random.sample(list(zip(reviews_train.data,
#                                     reviews_train.target)),k=2000)
# >>> text_train=[d[0] for d in muestra_entr]
# >>> text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
# >>> yimdb_train=np.array([d[1] for d in muestra_entr])
# >>> reviews_test = load_files("data/aclImdb/test/")
# >>> muestra_test=random.sample(list(zip(reviews_test.data,
#                                         reviews_test.target)),k=400)
# >>> text_test=[d[0] for d in muestra_test]
# >>> text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
# >>> yimdb_test=np.array([d[1] for d in muestra_test])

# Ahora restaría vectorizar los textos. Puesto que la versión NaiveBayes que
# se ha pedido implementar es la categórica (es decir, no es la multinomial),
# a la hora de vectorizar los textos lo haremos simplemente indicando en cada
# componente del vector si el correspondiente término del vocabulario ocurre
# (1) o no ocurre (0). Para ello, usar CountVectorizer de Scikit Learn, con la
# opción binary=True. Para reducir el número de características (es decir,
# reducir el vocabulario), usar "stop words" y min_df=50. Se puede ver cómo
# hacer esto en el ejercicio del tema de modelos probabilísticos.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

print("\nCríticas de películas en IMDB:")

from sklearn.datasets import load_files

# Seleccionamos una muestra aleatoria de 2000 criticas para entrenar
reviews_train = load_files("data/aclImdb/train/")
muestra_entr=rd.sample(list(zip(reviews_train.data,
                                reviews_train.target)),k=2000)
# Extraemos el texto y las etiquetas del entrenamiento
text_train=[d[0] for d in muestra_entr]
text_train = [doc.replace(b"<br />", b" ") for doc in text_train] #quitar saltos linea
yimdb_train=np.array([d[1] for d in muestra_entr])



# Para test
reviews_test = load_files("data/aclImdb/test/")
muestra_test=rd.sample(list(zip(reviews_test.data,
                                    reviews_test.target)),k=400)
text_test=[d[0] for d in muestra_test]
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
yimdb_test=np.array([d[1] for d in muestra_test])


# Importamos vectorizador 
from sklearn.feature_extraction.text import CountVectorizer
# Creamos una instancia de la clase vectorizador y lo entrenamos
vectorizador = CountVectorizer( binary = True, stop_words= 'english', min_df = 50  )

#Hay que entrenarlo con los datos de entrenamiento

vec_text = vectorizador.fit_transform(text_train) # Se entrena con el vocabulario del conjunto de entrenamiento y se transforma
vec_text_test = vectorizador.transform(text_test) # Transforma el vocabulario del conjunto de test a un array que indica la presencia de cada término

# El vocabulario
vocabulario = vectorizador.get_feature_names_out()
#vocabulario
#El texto vectorizado, lo pasamos a un array de numpy con toarray
# Nos dice si un elemento del vocabulario aparece o no en un documento
vec_text= vec_text.toarray()
vec_text_test = vec_text_test.toarray()
#print(vec_text_test)


NB_text = NaiveBayes()
NB_text.entrena(vec_text , yimdb_train )

print('La precisión en el conjunto de test es:')
print(rendimiento(NB_text, vec_text_test, yimdb_test ))
print('La precisión en el conjunto de entrenamiento es:')
print(rendimiento(NB_text, vec_text, yimdb_train))

# Con scikit learn sale el mismo rendimiento
from sklearn.naive_bayes import CategoricalNB
nb_cat = CategoricalNB()
print('La precisión en el conjunto de entrenamiento de NB_categorico sklearn:')
nb_cat.fit(vec_text, yimdb_train)
print(nb_cat.score(vec_text_test, yimdb_test))
# 0.795
print('La precisión en el conjunto de test de NB_categorico sklearn:')
print(nb_cat.score(vec_text, yimdb_train))
#0.8345










print("\n REGRESIÓN LOGÍSTICA MINI BATCH \n")
# =====================================================
# PARTE II: MODELOS LINEALES PARA CLASIFICACIÓN BINARIA
# =====================================================

# En esta SEGUNDA parte se pide implementar en Python un clasificador binario
# lineal, basado en regresión logística. 


# ---------------------------------------------
# II.1) Implementación de un clasificador lineal
# ---------------------------------------------

# En esta sección se pide implementar un clasificador BINARIO basado en
# regresión logística, con algoritmo de entrenamiento de descenso por el
# gradiente mini-batch (para minimizar la entropía cruzada).



class RegresionLogisticaMiniBatch():

    #asiganamos los valores iniciales y por defecto a las variables dentro de la clase
    def __init__(self,clases=[0,1],normalizacion=False, rate=0.1, rate_decay=False, batch_tam=64):
        #asignamos los valores de los argumentos a los atributos correspondientes para poder reutilizarlos en los metodos de la clase
        self.clases=clases
        self.normalizacion=normalizacion
        self.rate=rate
        self.rate_decay=rate_decay
        self.batch_tam=batch_tam
        self.rate_decay=rate_decay
        self.entrenamiento = False
    
    def sigmoide(self, x):
        from scipy.special import expit
        return expit(x)
        
    def entrena(self,X,y,n_epochs,reiniciar_pesos=False,pesos_iniciales=None):
        self.X=np.array(X)
        self.y=np.array(y)
        self.n_epochs=n_epochs
        self.reiniciar_pesos=reiniciar_pesos
        indices=list(range(len(X))) #creamos una lista con los indices de los datos para permutarlos aleatoriamente al hacer el entrenamiento

        # Estructura if para elegir pesos 
        if self.reiniciar_pesos:
            # Si es True hay que reiniciar los pesos cada vez que se entrena
            self.pesos_iniciales=np.random.uniform(-1,1,len(X[0]))
        elif pesos_iniciales is not None: 
            # Si se ha proporcionado un vector de pesos se utiliza ese.
            # Hacemos una copia para no alterar el valor de pesos_inciales ( si no estaríamos haciendo una referencia)
            self.pesos_iniciales = pesos_iniciales.copy()
        elif pesos_iniciales is None and not self.entrenamiento:
            # Si no hay un vector de entrada de pesos iniciales y no se ha entrenado antes el modelo hay que generar pesos
            self.pesos_iniciales=np.random.uniform(-1,1,len(X[0]))
            # Si el modelo ya se ha entrenado anteriormente se cogerán los pesos del último entrenamiento 
            
        self.pesos=self.pesos_iniciales
        
        #hacemos la normalización si es necesaria para el entrenamiento
        if self.normalizacion:
            #especificamos que la normalización se hace por columnas (axis=0) para que se normalicen los atributos y no las instancias
            #para que los atributos se hallen en el rango [0,1]
            self.media=np.mean(self.X,axis=0)
            self.std=np.std(self.X,axis=0)
            self.X=(self.X-self.media)/self.std


        #Actualizamos los pesos en cada epoch definido
        for n in range(self.n_epochs):
            #actualizamos el rate en cada epoch si es necesario, usamos el estandar
            if self.rate_decay:
                self.rate=self.rate/(1+n*self.rate_decay)

            rd.shuffle(indices) #permutamos los indices aleatoriamente
            for i in range(0,len(self.X),self.batch_tam):

                X_batch = self.X[indices[i:i+self.batch_tam]]
                y_minibatch = self.y[indices[i:i+self.batch_tam]]

                predicciones=self.sigmoide(np.dot(X_batch,self.pesos))
                self.pesos+=self.rate*np.dot(X_batch.T,(y_minibatch-predicciones))


        #igualamos la variable de pesos iniciales a los pesos finales obtenidos tras el entrenamiento ( para poder utilizarlo la siguiente vez)
        self.pesos_iniciales=self.pesos
        self.entrenamiento = True # Entrenamiento realizado

    def clasifica_prob(self,ejemplo):
        #escribimos clasificador no entrenado
        
        if not self.entrenamiento:
            raise ClasificadorNoEntrenado('No se ha entrenado el modelo')
        ejemplo=np.array(ejemplo)
        self.ejemplo=ejemplo
        pos= np.where(self.X==self.ejemplo)
        self.pos=pos

        # calculamos la probabilidad de pertenencia a una clase del ejemplo con la función sigmoide
        self.probabilidad=self.sigmoide(np.dot(self.ejemplo,self.pesos))
        #Creamos un diccionario con la probabilidad, la clave asignada es irrelevante
        probabilidad_dict = {1: float(self.probabilidad)} 
        self.probabilidad_dict=probabilidad_dict

        return self.probabilidad_dict

    def clasifica(self,ejemplo):
        self.ejemplo=ejemplo 
        #Si es necesario normalizamos el ejemplo de entrada que queremos clasificar
        if self.normalizacion:
           self.ejemplo=(self.ejemplo-self.media)/self.std

        #Obtenemos la probabilidad de pertenencia a la clase 1 del ejemplo mediante un diccionario
        self.probabilidad_dict=self.clasifica_prob(self.ejemplo)

        #Comparamos la probabilidad de pertenencia a la clase 1 con 0.5 para clasificar el ejemplo
        if self.probabilidad_dict[1] >= 0.5:
            clasificacion=1
        else:
            clasificacion=0
            
        self.clasificacion=clasificacion

        return clasificacion

# -----------------------------------
# II.2) Aplicando Regresión Logística 
# -----------------------------------

# Usando el clasificador implementado, obtener clasificadores con el mejor
# rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Como antes, será necesario separar conjuntos de validación y test para dar
# la valoración final de los clasificadores obtenidos Se permite usar
# train_test_split de Scikit Learn para esto. Ajustar los parámetros de tamaño
# de batch, tasa de aprendizaje y rate_decay. En alguno de los conjuntos de
# datos puede ser necesaria normalización.

# Mostrar el proceso realizado en cada caso, y los rendimientos obtenidos. 


#Cramos una funcion que se encargue de dividir los datos en entrenamiento y prueba dados los datos iniciales y el porcentaje de entrenamiento





# -----------------------------------------------------------------
# CANCER DE MAMA
# -----------------------------------------------------------------


print("Cáncer de mama:")
from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_cancer,y_cancer=cancer.data,cancer.target

#dividimos los datos en entrenamiento y prueba
Xe_cancer,Xt_cancer,ye_cancer,yt_cancer=divide_entrenamiento_prueba(X_cancer,y_cancer,0.7)

#entrenamos el clasificador
lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
lr_cancer.entrena(Xe_cancer,ye_cancer,10000)


#calculamos la precisión en el entrenamiento
precision_entrenamiento=rendimiento(lr_cancer,Xe_cancer,ye_cancer)
print("la precisión en el entrenamiento es: ", precision_entrenamiento)
# 0.9906103286384976
precision_test=rendimiento(lr_cancer,Xt_cancer,yt_cancer)
print("la precisión en el test es: ", precision_test)
# 0.9720279720279724


print("")


# -----------------------------------------------------------------
# VOTOS DE CONGRESISTAS US
# -----------------------------------------------------------------

print("Votos de congresistas US:")

# usamos la libreria importlib.util para poder importar el módulo votos.py
# que se encuentra en la carpeta data
import importlib.util
archivo_votos = 'data.votos'

# Importa el módulo "votos.py"
votos = importlib.import_module(archivo_votos)

# guardamos los arrays de votos.py en variables para poder usarlos en el archivo actual
votos_datos = votos.datos
votos_clasif=votos.clasif

#dividimos los datos en entrenamiento y prueba
Xe_votos,Xt_votos,ye_votos,yt_votos=divide_entrenamiento_prueba(votos_datos,votos_clasif,0.7)

#entrenamos el clasificador
lr_votos=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
lr_votos.entrena(Xe_votos,ye_votos,500)

#calculamos la precisión en el entrenamiento
precision_entrenamiento_votos=rendimiento(lr_votos,Xe_votos,ye_votos)
print("la precisión en el entrenamiento es: ", precision_entrenamiento_votos)

#calculamos la precisión en el test
precision_test_votos=rendimiento(lr_votos,Xt_votos,yt_votos)
print("la precisión en el test es: ", precision_test_votos)

# -----------------------------------------------------------------
# CRÍTICAS DE PELÍCULAS EN IMDB
# -----------------------------------------------------------------

print("\nCríticas de películas en IMDB:")

NB_text=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
NB_text.entrena(vec_text,yimdb_train,1000)

precision_entrenamiento_text=rendimiento(NB_text,vec_text,yimdb_train)
print("la precisión en el entrenamiento es: ", precision_entrenamiento_text)
precision_test_text=rendimiento(NB_text,vec_text_test,yimdb_test)
print("la precisión en el test es: ", precision_test_text)


print("\n CLASIFICACIÓN MULTICLASE \n")

# ===================================
# PARTE III: CLASIFICACIÓN MULTICLASE
# ===================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica
# de "One vs Rest" (OvR)

# ------------------------------------
# III.1) Implementación de One vs Rest
# ------------------------------------


#  En concreto, se pide implementar una clase python RL_OvR con la siguiente
#  estructura, y que implemente un clasificador OvR usando como base el
#  clasificador binario del apartado anterior.

# class RL_OvR():

#     def __init__(self,clases,rate=0.1,rate_decay=False,batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs):

#        .......

#     def clasifica(self,ejemplo):

#        ......


class RL_OvR():

    def __init__(self,clases,rate=0.1,rate_decay=False,batch_tam=64):
      self.clases=clases
      self.rate=rate
      self.rate_decay=rate_decay
      self.batch_tam=batch_tam
      self.clasificadores = {}  # Diccionario para almacenar los clasificadores binarios


    def sigmoide(self, x):
      from scipy.special import expit
      return expit(x)

    def entrena(self,X,y,n_epochs):
        self.X=X
        self.y=y
        self.n_epochs=n_epochs
        indices=list(range(len(X)))
        self.indices=indices
        rango_pesos=np.random.uniform(-0.1,0.1,len(X[0]))
        self.rango_pesos=rango_pesos

        # usamos el clasificador binario del apartado anterior para cada clase posible

        for clase in self.clases:
            # Creamos un objeto de la clase regresion logistica. Uno para cada clase 
            self.clf_log = RegresionLogisticaMiniBatch(clases=[clase], rate=self.rate, rate_decay=self.rate_decay,
                                              batch_tam=self.batch_tam)
           
            # Formamos un vector con 1 en las posiciones correspondientes a la clase que estudia el clasificador y 0 para las demas
            yc= np.where(y == clase , 1, 0)
            self.clf_log.entrena(X, yc , n_epochs) # Entrenamos cada clasificador

            # Guardamos cada modelo (ya entrenado) en un diccionario. La clave para acceder a cada uno es la clase que entrena
            self.clasificadores[clase] = self.clf_log




    def clasifica(self,ejemplo):
        self.ejemplo=ejemplo
        probabilidades = {}
        # Clasificamos el ejemplo con cada clasificador binario
        for clase, self.clf_log in self.clasificadores.items():
            probabilidad = self.clf_log.clasifica_prob(ejemplo)[1]  # Probabilidad de pertenencia a la clase
            probabilidades[clase] = probabilidad
        #print("las probabilidades son:",probabilidades)
        # Seleccionamos la clase con la mayor probabilidad
        clasificacion = max(probabilidades, key=probabilidades.get)
        return clasificacion






#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, excepto que ahora "clases" puede ser una lista con más de dos
#  elementos. 

#  Un ejemplo de sesión, con el problema del iris:




# --------------------------------------------------------------------
# >>> from sklearn.datasets import load_iris
# >>> iris=load_iris()
# >>> X_iris=iris.data
# >>> y_iris=iris.target
# >>> Xe_iris,Xt_iris,ye_iris,yt_iris=train_test_split(X_iris,y_iris)

# >>> rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20)

# >>> rl_iris.entrena(Xe_iris,ye_iris,n_epochs=1000)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris,Xt_iris,yt_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------
print("Iris:")
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris=load_iris()
X_iris=iris.data
y_iris=iris.target

Xe_iris,Xt_iris,ye_iris,yt_iris=train_test_split(X_iris,y_iris)


rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20)

rl_iris.entrena(Xe_iris,ye_iris,n_epochs=1000)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
rl_iris_rendimiento=rendimiento(rl_iris,Xe_iris,ye_iris)
print("la precisión en el entrenamiento es", rl_iris_rendimiento )
rl_iris_rendimiento_test=rendimiento(rl_iris,Xt_iris,yt_iris)
print("la precisión en el test es", rl_iris_rendimiento_test )


# ------------------------------------------------------------
# III.2) Clasificación de imágenes de dígitos escritos a mano
# ------------------------------------------------------------

print("\nDigitos escritos a mano")

#  Aplicar la implementación del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. Si el
#  tiempo de cómputo en el entrenamiento no permite terminar en un tiempo
#  razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 



#Definimos una función que procese nuestras imagenes y las cargue
def cargaImágenes(fichero,ancho,alto):
    #Definimos una función para convertir los caracteres a 0 y 1 que se interpretarán
    #como trazo o no trazo de imagen
    def convierte_0_1(c):
        if c==" ": 
            return 0 #no trazo
        else:
            return 1 # todo lo que no sea vacio es trazo, independientemente de 
            #si es un + o un #, ya que no nos interesa diferenciar entre borde e interior
       
    # cargamos las imagenes, las imagenes estan en un fichero donde se separan con espacios
    with open(fichero) as f:
        #creamos una lista para almacenar las imagenes
        lista_imagenes=[] #lista de imagenes
        #creamos una lista ejemplo que será donde construyamos cada imagen
        ejemplo=[]
        cont_lin=0 #esta variable cuenta las lineas que se han procesado 
        for lin in f: #iteramos sobre cada linea
            #para cada linea se utiliza la función convierte hasta que se llega al
            #final (ancho de la imagen). Esto se realiza mediante map para aplicarlo a toda la linea
            #con .extend se agregan los valores convertidos de la linea a la lista ejemplo
            ejemplo.extend(list(map(convierte_0_1,lin[:ancho])))
            cont_lin+=1
            #una vez se ha procesado la linea se suma 1 al contador de lineas
            if cont_lin == alto: 
                #Cuando se ha detectado que las lineas procesadas son igual al alto de la imagen,
                # quiere decir que hemos procesado la imagen y la añadimos el ejemplo a la lista de imagenes
                lista_imagenes.append(ejemplo)  
                ejemplo=[] #reiniciamos ejemplo
                cont_lin=0 #reiniciamos el contador de lineas
    return np.array(lista_imagenes) #devolvemos un array de numpy con las imagenes

# cargamos las clases 
def cargaClases(fichero):
    with open(fichero) as f:
        # parar cada caracter c en el archivo f de clases lo convierte a entero
        #para tener una lista con las clases formateadas como enteros
        return np.array([int(c) for c in f])
    
   
trainingdigits="data/digitdata/trainingimages"
validationdigits="data/digitdata/validationimages"
testdigits="data/digitdata/testimages"
trainingdigitslabels="data/digitdata/traininglabels"
validationdigitslabels="data/digitdata/validationlabels"
testdigitslabels="data/digitdata/testlabels"
X_train_dg=cargaImágenes(trainingdigits,28,28)
y_train_dg=cargaClases(trainingdigitslabels)
#train es para entrenar
X_valid_dg=cargaImágenes(validationdigits,28,28)
y_valid_dg=cargaClases(validationdigitslabels)
#valid es para ajustar hiperparametros
X_test_dg=cargaImágenes(testdigits,28,28)
y_test_dg=cargaClases(testdigitslabels)


def reduccion_conjunto(X, y,a):

    tam_muestra = int(len(X) * a)
    # Generar una lista de índices únicos
    indices = list(range(len(X)))
    
    # Realizar un muestreo aleatorio sin reemplazo para seleccionar la muestra
    indices_muestra = rd.sample(indices, tam_muestra)
    
    # Obtener los datos y etiquetas correspondientes a los índices de la muestra
    X_muestra = X[indices_muestra]
    y_muestra = y[indices_muestra]
    
    return X_muestra, y_muestra

# Aplicar el muestreo aleatorio al conjunto de entrenamiento, validación y prueba
# utilizamos solamente el 30% de los datos para reducir el tiempo de procesamiento
X_train_dg, y_train_dg = reduccion_conjunto(X_train_dg, y_train_dg,0.3)
X_valid_dg, y_valid_dg = reduccion_conjunto(X_valid_dg, y_valid_dg,0.3)
X_test_dg, y_test_dg = reduccion_conjunto(X_test_dg, y_test_dg,0.3)


im= cargaImágenes(trainingdigits,28,28)

clases_imagenes = np.unique(y_train_dg)
model_digit=RL_OvR(clases_imagenes ,rate=0.1,batch_tam=20)

model_digit.entrena(X_train_dg,y_train_dg,n_epochs=10)

digit_rendimiento_train =rendimiento(model_digit,X_train_dg, y_train_dg)
print("la precisión en el entrenamiento es: ", digit_rendimiento_train)

digit_rendimiento_valid =rendimiento(model_digit,X_valid_dg, y_valid_dg)
print("la precisión en la validación es: ", digit_rendimiento_valid)

digit_rendimiento_test =rendimiento(model_digit,X_test_dg, y_test_dg)
print("la precisión en el test es: ", digit_rendimiento_test)

