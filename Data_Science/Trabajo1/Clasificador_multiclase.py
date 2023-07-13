class ClasificadorNoEntrenado(Exception): 
    pass

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
        self.entrenamiento = False # Variable lógica (flag) para indicar si se ha hecho el entrenamiento
    
    #definimos la función sigmoide
    def sigmoide(self, x):
        from scipy.special import expit
        return expit(x)
        
    #definimos la función de entrenamiento
    def entrena(self,X,y,n_epochs,reiniciar_pesos=False,pesos_iniciales=None):
        self.X=X 
        self.y=y
        self.n_epochs=n_epochs
        self.reiniciar_pesos=reiniciar_pesos
        self.pesos_iniciales=pesos_iniciales
        indices=list(range(len(X))) #creamos una lista con los indices de los datos para permutarlos aleatoriamente al hacer el entrenamiento

        if self.reiniciar_pesos:
            #Si se deben reiniciar los pesos, se generarán aleatoriamente al inicio de cada bucle de entrenamiento
            self.pesos_iniciales=np.random.uniform(-1,1,len(X[0]))
        else:
            if np.array_equal(self.pesos_iniciales, None): #and not self.entrenamient
                
                #generamos los pesos por defecto en el caso de no asignar pesos iniciales
                #estos pesos estaran aleatoriamente distribuidos entre -1 y 1 y tendran la misma longitud que el numero de atributos
                rango_pesos=np.random.uniform(-1,1,len(X[0]))
                self.rango_pesos=rango_pesos
                #Seleccionamos los pesos iniciales en función de si se han proporcionado anteriormente o en caso contrario, usamos
                #los pesos generados aleatoriamente por defecto
                self.pesos_iniciales=rango_pesos #inicializamos los pesos de forma aleatoria
                
        #en el caso de tener pesos iniciales detectados, ya sean estos aportados por el usuario o
        #por el entrenamiento anterior, se continuará con ellos y no se generaran nuevos.
        # Si se han proporcionado pesos iniciales, se usaran estos para el entrenamiento
        # Idependientemente de en cual de los dos casos nos hayemos, se asignan los pesos iniciales a la variable pesos
        # que será la que se irá actualizando en el entrenamiento
        pesos=self.pesos_iniciales
        self.pesos=pesos


        #hacemos la normalización si es necesaria para el entrenamiento
        if self.normalizacion:
            #normalizamos los datos
            #especificamos que la normalización se hace por columnas (axis=0) para que se normalicen los atributos y no las instancias
            #para que los atributos se hallen en el rango [0,1]
            self.media=np.mean(self.X,axis=0)
            self.std=np.std(self.X,axis=0)
            self.X=(self.X-self.media)/self.std
            print("normalización realizada")

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
                
            #if n<10: print("el rate es:" ,self.rate)

    #igualamos la variable de pesos iniciales a los pesos finales obtenidos tras el entrenamiento ( para poder utilizarlo la siguiente vez)
        self.pesos_iniciales=self.pesos
        self.entrenamiento = True

        
    def clasifica_prob(self,ejemplo):
        #escribimos clasificador no entrenado
        if np.all(self.pesos) == None:
            #return "ClasificadorNoEntrenado"
            raise ClasificadorNoEntrenado(' No se ha entrenado el modelo')
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


# ===================================
# PARTE III: CLASIFICACIÓN MULTICLASE
# ===================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica
# de "One vs Rest" (OvR)

# ------------------------------------
# III.1) Implementación de One vs Rest
# ------------------------------------
print("hola olvidona")

#  En concreto, se pide implementar una clase python RL_OvR con la siguiente
#  estructura, y que implemente un clasificador OvR usando como base el
#  clasificador binario del apartado anterior.

import numpy as np
import random as rd
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split


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
        # Entrenamos el clasificador binario para la clase seleccionada
        # Almacenamos el clasificador en el diccionario de clasificadores
        # Cada clave del diccionario será una clase y su valor será el clasificador binario entrenado
        for clase in self.clases:
            # Creamos un objeto de la clase regresion logistica. Uno para cada clase 
            self.clf_log = RegresionLogisticaMiniBatch(clases=[clase], rate=self.rate, rate_decay=self.rate_decay,
                                              batch_tam=self.batch_tam)
            self.clasificadores[clase] = self.clf_log
           
            # Formamos un vector con 1 en las posiciones correspondientes a la clase que estudia el clasificador y 0 para las demas
            yc= np.where(y == clase , 1, 0)
            self.clf_log.entrena(X, yc, n_epochs)
            


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

#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, excepto que ahora "clases" puede ser una lista con más de dos
#  elementos. 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
from sklearn.datasets import load_iris
iris=load_iris()
X_iris=iris.data
y_iris=iris.target
y_iris == 0
np.where(y_iris == 0, 1, 0)
Xe_iris,Xt_iris,ye_iris,yt_iris=train_test_split(X_iris,y_iris)


rl_iris=RL_OvR([0,1,2],rate=0.001,batch_tam=20)

rl_iris.entrena(Xe_iris,ye_iris,n_epochs=1000)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
rl_iris_rendimiento=rendimiento(rl_iris,Xe_iris,ye_iris)
print("rendimiento iris entrenamiento", rl_iris_rendimiento )
# 0.9732142857142857



# >>> rendimiento(rl_iris,Xt_iris,yt_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------


#------------------------------------------------------------
print("")
print("Clasificación de digitos")
print("")

# ------------------------------------------------------------
# III.2) Clasificación de imágenes de dígitos escritos a mano
# ------------------------------------------------------------


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

import random as rd

def reduccion_conjunto(X, y,a):
    # Obtener el tamaño de la muestra deseada
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
X_train_dg, y_train_dg = reduccion_conjunto(X_train_dg, y_train_dg,0.3)
X_valid_dg, y_valid_dg = reduccion_conjunto(X_valid_dg, y_valid_dg,0.3)
X_test_dg, y_test_dg = reduccion_conjunto(X_test_dg, y_test_dg,0.3)


im= cargaImágenes(trainingdigits,28,28)

clases_imagenes = np.unique(y_train_dg)
model_digit=RL_OvR(clases_imagenes ,rate=0.1,batch_tam=20)

model_digit.entrena(X_train_dg,y_train_dg,n_epochs=10)
print("batch: ", model_digit.batch_tam)
print("rate: ", model_digit.rate)
print("epochs" , model_digit.n_epochs)
digit_rendimiento_train =rendimiento(model_digit,X_train_dg, y_train_dg)
print("rendimiento digitos entrenamiento: ", digit_rendimiento_train)

digit_rendimiento_valid =rendimiento(model_digit,X_valid_dg, y_valid_dg)
print("rendimiento digitos validación: ", digit_rendimiento_valid)

digit_rendimiento_test =rendimiento(model_digit,X_test_dg, y_test_dg)
print("rendimiento digitos test: ", digit_rendimiento_test)



#test para medir los resultados


# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 

print("adios olvidona")














































