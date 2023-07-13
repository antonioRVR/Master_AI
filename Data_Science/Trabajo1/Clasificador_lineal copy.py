#Nueva terminal, boton derecho en la carpeta, abrir terminal integrada y escribir >python pruebas.py para ejecutarlo en powershell

print("Hola olvidona")

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
# Usar por ejemplo la siguiente secuencia de insStrucciones, para extraer los textos:

import numpy as np
import random as rd
from sklearn.datasets import load_files
import time

#reviews_train = load_files("data/aclImdb/train/")
#muestra_entr=rd.sample(list(zip(reviews_train.data,
#                                reviews_train.target)),k=2000)
#text_train=[d[0] for d in muestra_entr]
#text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
#yimdb_train=np.array([d[1] for d in muestra_entr])
#reviews_test = load_files("data/aclImdb/test/")
#muestra_test=rd.sample(list(zip(reviews_test.data,
#                                    reviews_test.target)),k=400)
#text_test=[d[0] for d in muestra_test]
#text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
#yimdb_test=np.array([d[1] for d in muestra_test])

# Ahora restaría vectorizar los textos. Puesto que la versión NaiveBayes que
# se ha pedido implementar es la categórica (es decir, no es la multinomial),
# a la hora de vectorizar los textos lo haremos simplemente indicando en cada
# componente del vector si el correspondiente término del vocabulario ocurre
# (1) o no ocurre (0). Para ello, usar CountVectorizer de Scikit Learn, con la
# opción binary=True. Para reducir el número de características (es decir,
# reducir el vocabulario), usar "stop words" y min_df=50. Se puede ver cómo
# hacer esto en el ejercicio del tema de modelos probabilísticos.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  

# ---------------------------------------------
# II.1) Implementación de un clasificador lineal
# ---------------------------------------------

# En esta sección se pide implementar un clasificador BINARIO basado en
# regresión logística, con algoritmo de entrenamiento de descenso por el
# gradiente mini-batch (para minimizar la entropía cruzada).

inicio=time.time()

class RegresionLogisticaMiniBatch():

    #asiganamos los valores iniciales y por defecto a las variables dentro de la clase
    def __init__(self,clases=[0,1],normalizacion=False, rate=0.1,rate_decay=False,batch_tam=64):
        #asignamos los valores de los argumentos a los atributos correspondientes para poder reutilizarlos en los metodos de la clase
        self.clases=clases
        self.normalizacion=normalizacion
        self.rate=rate
        self.rate_decay=rate_decay
        self.batch_tam=batch_tam
        self.rate_decay=rate_decay
    
    def sigmoide(self, x):
        from scipy.special import expit
        return expit(x)
        
    def entrena(self,X,y,n_epochs,reiniciar_pesos=False,pesos_iniciales=None):
        self.X=np.array(X)
        self.y=np.array(y)
        self.n_epochs=n_epochs
        self.reiniciar_pesos=reiniciar_pesos
        self.pesos_iniciales=pesos_iniciales
        indices=list(range(len(X)))
        self.indices=indices
        rango_pesos=np.random.uniform(-0.1,0.1,len(X[0]))
        self.rango_pesos=rango_pesos
        #Ajustamos los pesos iniciales en función de si se han proporcionado anteriormente o no
        if self.pesos_iniciales==None:
            self.pesos_iniciales=rango_pesos #inicializamos los pesos de forma aleatoria
        else:
            self.pesos_iniciales=pesos_iniciales

        
        #hacemos la normalización si es necesaria para el entrenamiento
        if self.normalizacion==True:
            #normalizamos los datos
            #especificamos que la normalización se hace por columnas (axis=0) para que se normalicen los atributos y no las instancias
            self.media=np.mean(self.X,axis=0)
            self.std=np.std(self.X,axis=0)
            self.X=(self.X-self.media)/self.std
            print("normalización realizada")



        #comprobamos si hay que reiniciar los pesos. Si hay que reiniciarlos se vuelven a generar aleatoriamente
        #en caso contrario, se mantienen los anteriiores
        if self.reiniciar_pesos==True:
            pesos_iniciales=rango_pesos
        else:
            pesos_iniciales=self.pesos_iniciales


        #Hacemos una copia de los pesos iniciales   
        pesos=pesos_iniciales.copy()
        self.pesos=pesos

        for n in range(self.n_epochs):
            #actualizamos el rate en cada epoch si es necesario
            if self.rate_decay==False:
                self.rate=self.rate
            else:
                self.rate=(self.rate/(1+n*self.rate_decay))

            rd.shuffle(indices) #permutamos los indices aleatoriamente
            for i in range(0,len(self.X),self.batch_tam):

                X_batch = self.X[indices[i:i+self.batch_tam]]
                y_minibatch = self.y[indices[i:i+self.batch_tam]]

                predicciones=self.sigmoide(np.dot(X_batch,self.pesos))
                self.pesos+=self.rate*np.dot(X_batch.T,(y_minibatch-predicciones))

                for k in range(len(pesos)):
                    
                    suma=0
                    for j in range(len(X_batch)):
                        #Calculo de la sigmoide para cada instancia del batch, el resultaado es la sigmoide del ejemplo j
                        calculo_sigmoide=self.sigmoide(np.dot(pesos.T,X_batch[j]))
                        #Calculo de la correcion para cada instancia del batch, el resultado es la correcion del ejemplo j
                        correcion=(y_minibatch[j]-calculo_sigmoide)
                        #ahora hacemos el sumatorio de todas las correcciones, para cada ejemplo del batch
                        suma+=correcion*X_batch[j][k]
                    self.pesos[k]=self.pesos[k]+self.rate*suma
  
        print("los pesos inicialmente son: ", pesos_iniciales)    
        print("los pesos finales son: ", self.pesos)
 


    def clasifica_prob(self,ejemplo):
        #escribimos clasificador no entrenado
        if np.all(self.pesos) == None:
            return "ClasificadorNoEntrenado"
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
        if self.normalizacion==True:
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
    



# ----------------------------------------------------------------

def rendimiento(clasificador, X,y):
    # Si el clasificador esta entrenado , habrá que ver para todos los valores si hay una clasificacion correcta
    d_y= len(y) # Numero de ejemplos
    y_p=[] # Prediccion ejemplos. Lista vacia y vamos añadiendo la prediccion a cada ejemplo
    for t in range(d_y):
        #print("iteración del bucle(instancia): ", t)
        y_p.append(clasificador.clasifica(X[t]))




    # Comparar y_p co(n y
    aciertos = sum(y_p == y)


    
    ## Accuracy = aciertos / n_ejemplos 
    accuracy= aciertos / d_y
    return accuracy
# Ejemplo de uso, con los datos del cáncer de mama, que se puede cargar desde
# Scikit Learn:

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
X_cancer,y_cancer=cancer.data,cancer.target

#unimos x_cancer e y_cancer para poder hacer la permutación de forma conjunta y que no se desordenen los datos
# esto es con el fin de que no se desordenen los datos y se mantenga la correspondencia entre los datos y sus clases
# y para hacer la división en entrenamiento y prueba
cancer=np.column_stack((X_cancer,y_cancer))
print("forma de cancer: ", cancer.shape)
#hay que hacer el shuffle de cancer
rd.shuffle(cancer)
#volvemos a separar cancer en X_cancer e y_cancer
X_cancer=cancer[:,:-1]
y_cancer=cancer[:,-1]
print("forma de X_cancer: ", X_cancer.shape)
print("forma de y_cancer: ", y_cancer.shape)

#dividimos los datos en entrenamiento y prueba
porcentaje_entrenamiento=0.7
Xe_cancer=X_cancer[:int(len(X_cancer)*porcentaje_entrenamiento)]
Xt_cancer=X_cancer[int(len(X_cancer)*(porcentaje_entrenamiento)):]
ye_cancer=y_cancer[:int(len(y_cancer)*porcentaje_entrenamiento)]
yt_cancer=y_cancer[int(len(y_cancer)*(porcentaje_entrenamiento)):]
print("la longutd del entrenamiento es: ", len(Xe_cancer))
print("la longutd de la prueba es: ", len(Xt_cancer))


lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True,normalizacion=True)
lr_cancer.entrena(Xe_cancer,ye_cancer,500)

precision_entrenamiento=rendimiento(lr_cancer,Xe_cancer,ye_cancer)
print("la precisión en el entrenamiento es: ", precision_entrenamiento)
# 0.9906103286384976
precision_test=rendimiento(lr_cancer,Xt_cancer,yt_cancer)
print("la precisión en el test es: ", precision_test)
# 0.9720279720279724
final=time.time()
tiempo=final-inicio
print("tiempo de calculo:", tiempo)
# -----------------------------------------------------------------



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

print("Adios olvidona")