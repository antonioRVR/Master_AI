## Vectores: Todos los elementos de la misma naturaleza
## * Creación:
##   - c(dato1, dato2, ...)
##   - inicio:fin siendo inicio y fin dos números cualesquiera
## * Selección:
##   - Por los índices de las posiciones: vector[vector.índices]
##   - Por exclusión de las posiciones: vector[-vector.índices]
##   - Por condición: vector[vector.lógico]
## Matrices
## * Creación:
##   matrix(vector.x, ...) organiza los elementos de vector.x en una
##   matriz (consultar la ayuda para más detalles)
##   matrix(c("E", "C", "C", "C", "A", "E"), nrow = 2, byrow = TRUE,
##          dimnames = list(c("f1", "f2"), c("c1", "c2", "c3"))) 
##      c1  c2  c3 
##   f1 "E" "C" "C"
##   f2 "C" "A" "E"
## * Acceso:
##   - Por el índice de la columna: matriz[, índice.col]
##   - Por el nombre de la fila: matriz[nom.fila, ]

## Algunas funciones de interés

## unique(vector.x) devuelve un vector con una única copia de cada
##   elemento repetido en el vector.x
##   unique(c("E", "C", "C", "C", "A", "E")) ==> "E" "C" "A"

## count(vector.x, k, alphabet = vector.a) cuenta el número de veces
##   que aparecen en vector.x los k-meros (palabras de tamaño k) que
##   se pueden formar con los símbolos del alfabeto vector.a
##   count(c("E", "C", "C", "C", "A", "E"), 2,
##         alphabet = c("E", "C", "A")) ==>
##   AA AC AE CA CC CE EA EC EE 
##    0  0  1  1  2  0  0  1  0

## sort(vector.x) ordena los datos de vector.x de menor a mayor
##   sort(c("E", "C", "A")) ==> "A" "C" "E"

## length(vector.x) cantidad de datos que contiene vector.x
##   length(c("E", "C", "C", "C", "A", "E")) ==> 6

## sum(vector.x) suma los elementos que contiene vector.x

######################################################################
## (1) Descargar el genoma del bacterio fago lambda, identificado con
## NC_001416.1 de la base de datos NCBI, y cargar la función
## local.composition definida en la práctica anterior (se encuentra en
## el fichero HMM.RData).
######################################################################


library("seqinr")

bfl <- read.fasta("sequence.fasta")

genoma <- getSequence(bfl)[[1]]
load("HMM.Rdata")


######################################################################
## (2) Crear un gráfico para comparar el contenido global y local en
## GC del genoma del bacterio fago lambda usando una ventana de
## longitud 5000 y un desplazamiento de 500.
######################################################################

GC_local <- local.composition(genoma,GC,5000,500)
GC_local_positions <- GC_local[[2]]
GC_local_results <- GC_local[[1]]
######################################################################
## (3) Escribir una función para el entrenamiento de un HMM (es decir,
## que estime las probabilidades iniciales, matriz de transición y
## matriz de emisión. Como argumentos debe recibir una secuencia de
## símbolos y otra (de la misma longitud) con los correspondientes
## estados.
######################################################################

#Definimos la función para el entrenamiento del modelo oculto de Markov

estimate.hmm <- function (sec.simb, sec.est) {
##     ## Paso 1: Definir las siguientes variables
##     ## alf.e: val(E)
##     ## m: |val(E)|
##     ## alf.x: val(X)
##     ## n: |val(X)|
  
  #creamos los vectores que continene los valores de la secuencia de estados y
  #de simbolos
  alf.e <- unique(sec.est)
  alf.x <- unique(sec.simb)
  
  m <- length(alf.e)
  n <- length(alf.x)
  

##     ## Probabilidades iniciales
##     ## Paso 2: Calcular el número de veces que aparece cada estado,
##     ## alf.e, en la secuencia de estados.

  
#para esto utilizamos la función table
repeticiones <- table(sec.est)
##     ## Paso 3: Definir pi como el vector con las probabilidades
##     ## iniciales de la variable E.

#Para esto simplemente se divide los valores repetidos entre el total 


#pi <- repeticiones / sum(repeticiones)
pi <- as.vector(repeticiones / sum(repeticiones))
names(pi) <- alf.e

##     ## Probabilidades de transición
##     ## Paso 4: Calcular el número de veces que aparece cada posible
##     ## pareja de estados en la secuencia de estados.


# Obtener los estados únicos
alf.e <- unique(sec.est)
m <- length(alf.e)


##     ## Paso 5: Definir A como la matriz mXm adecuada que contiene
##     ## las frecuencias anteriores (nombrar adecuadamente sus filas y
##     ## columnas).

#esto se trata de ver la cantidad de veces que sale una pareja de estados,
# es decir un estado y su siguiente, esto serian las parejas (11) (12) (22) (21)

# Se crea una matriz del tamaño de la cantidad de estados unicos
A <- matrix(0, nrow = m, ncol = m, dimnames = list(alf.e, alf.e))

# añadimos las frecuencias de transición entre estados a la matriz A

for (i in 1:(length(sec.est)-1)) {
  A[sec.est[i], sec.est[i+1]] <- A[sec.est[i], sec.est[i+1]] + 1
}

#apply(indices, 1, function(i) {
#  estado_actual <- sec.est[i]
#  estado_siguiente <- sec.est[i + 1]
#  A[estado_actual, estado_siguiente] <- A[estado_actual, estado_siguiente] + 1
#})


##     ## Paso 6: Normalizar cada fila de la matriz A.

A <- t(t(A) / rowSums(A))


##     ## Probabilidades de emisión
##     ## Paso 7: Definir B como una matriz mXn (nombrar adecuadamente
##     ## sus filas y columnas).

B <- matrix(0, nrow = m, ncol = n, dimnames = list(alf.e, alf.x))


##     ## Paso 8: Para cada elemento e de alf.e
##     ##   Paso 8.1: Almacenar en la fila e de B el número de veces que
##     ##   aparece cada símbolo, alf.x, en las posiciones de la
##     ##   secuencia de símbolos que se corresponde con las apariciones
##     ##   de e en la secuencia de estados.

for (i in 1:m) {
  emission_counts <- table(sec.simb[sec.est == alf.e[i]])
  for (j in 1:n) {
    B[i, j] <- emission_counts[alf.x[j]]
  }
}

##     ##   Paso 8.2: Normalizar la fila e de la matriz B

B <- t(t(B) / rowSums(B))

##     ## Devolver una lista con pi, A y B.

return(list(pi=pi, A=A, B=B))


 }

## 
dna.seq <- c("a", "c", "t", "a", "a", "t", "g", "t", "c", "t", "g", "a", 
             "c", "t", "a", "t", "c", "a", "g", "c", "g", "g", "c", "a", 
             "g", "c", "a", "g", "c", "t", "g", "a", "c", "c", "g", "t")


hidden.seq <- c(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)

estimate.hmm(dna.seq, hidden.seq)

#se observa que se obtienen los resultados correctos

## > estimate.hmm(dna.seq, hidden.seq)
## $pi
##   1   2 
## 0.5 0.5 
## $A
##           1          2
## 1 0.9444444 0.05555556
## 2 0.0000000 1.00000000
## $B
##           a         c         g         t
## 1 0.3333333 0.2222222 0.1111111 0.3333333
## 2 0.1666667 0.3333333 0.3888889 0.1111111

######################################################################
## La siguiente función estima la sucesión de estados más probable
## dada una secuencia símbolos y el hmm correspondiente.

viterbiLog <- function (obs, mom) {
  l <- length(obs)
  estados <- names(mom$pi)
  pr <- matrix(nrow = l-1, ncol = length(mom$pi),
               dimnames = list(2:l, estados))
  B <- log(mom$B)
  A <- log(mom$A)
  pi <- log(mom$pi)
  nu <- B[,obs[1]] + pi 
  for (k in 2:l) {
    aux <- nu + A
    nu <- B[,obs[k]] + apply(aux, 2, max)
    pr[as.character(k),] <- apply(aux, 2, which.max)
  }
  q <- c()
  q[l] <- estados[which.max(nu)]
  for (k in (l-1):1) {
    q[k] <- estados[pr[as.character(k+1),q[k+1]]]
  }
  q
}

######################################################################
## (4) Identificar utilizando una cadena de Markov oculta las regiones
## con alto y bajo contenido en GC. Utiliza lambda.initial.hmm como
## modelo (se encuentra definido en el archivo HMM.RData).
######################################################################


estados_probables_1 <- viterbiLog(genoma, lambda.initial.hmm)
estados_probables_1


######################################################################
## (5) Representa en un panel dos gráficos: uno con el gráfico
## anterior con el contenido local en GC y otro gráfico que represente
## la estimación de la secuencia oculta. Comenta estos gráficos.
######################################################################


# donde aparece h2 y h1 nos dice si el contenido es alto o bajo en GC
#segun las cadenas de Markov. Vamos a cambiar esos valores por
# unos arbitrarios que podamos graficar

estados_1 <- ifelse(estados_probables_1 == "h2", 1, 0)
print(estados_1)
length((estados_1))

#Creamos un vector para dotar a los estados problables de posiciones
#Lo necesitaremos más adelante para crear su representación
posiciones_estados_1 <- seq(1:48502)

#Ahora gradicamos la primera grafica


plot(GC_local_positions,GC_local_results, type="l")


#Si queremos poner en un mismo panel los dos graficos debemos utilizar
#las siguientes librerias

library(ggplot2)
library(cowplot)

# Para el uso de graficas con estas librerias debemos utilizar dataframes
# Por lo tanto creamos dataframes con los datos que queremos representar

data_local <- data.frame(GC_local$positions, GC_local_results)
# Gráfico de contenido local en GC
plot_GC <- ggplot(data_local, aes(x = GC_local$positions, y = GC_local_results)) +
  geom_line() +
  xlab("Posición") +
  ylab("Contenido local en GC") +
  ggtitle("Contenido local en GC ")




data_oculta_1 <- data.frame(posiciones_estados_1,estados_1)

# Gráfico de estimación de la secuencia oculta
plot_secuencia <- ggplot(data_oculta_1, aes(x = posiciones_estados_1, y = estados_1)) +
  geom_line() +
  xlab("Posición") +
  ylab("Secuencia Oculta") +
  ggtitle("Secuencia oculta")

# Combinar los gráficos en un panel
panel <- plot_grid(plot_GC, plot_secuencia, ncol = 1, align = "v")
# Mostrar el panel
print(panel)


#Se puede observar que donde el contenido Local en GC tiene un decrecimiento
#considerable, la secuencia oculta estima un estado "h1". Lo opuesto para los
#crecimientos notables. Cuando la variación es pequeña, no se aprecia sin embargo
#un cambio en el estado probable


######################################################################
## Tarea:

## En general se desconocen exactamente los parámetros pi, A y B que
## definen un HMM. Sin embargo, dada una secuencia de símbolos y una
## aproximación inicial pi0, A0 y B0 se puede seguir un proceso
## iterativo de esperanza-maximización o algoritmo EM
## (Expectation-Maximization Algorithm, EM Algorithm) para obtener un
## modelo más verosímil.

## Este tipo de algoritmos se organizan en dos fases diferentes:

## Entrada: Una secuencia de símbolos observable y un primer HMM
## asociado a la misma.

## * Fase 1, Paso M o Paso de máxima verosimilitud:
##   Dadas una secuencia de símbolos observable y un HMM calcular la
##   secuencia de estados oculta de máxima verosimilitud.

## * Fase 2, Paso E o Paso de esperanza:
##   Dadas una secuencia de símbolos observable y una secuencia oculta
##   estimar el HMM más probable.

## Estas dos fases se repiten un número de iteraciones produciendo en
## cada vuelta del bucle una mejor aproximación del modelo

## Salida: La estimación del HMM de la última iteración y la secuencia
## oculta más probable.

## (a) Implementar el algoritmo EM descrito. Debe devolver el modelo
## HMM estimado y la secuencia de estados oculta más probable


#definimos el algoritmo que nos da el valor de HMM en 
#función del numero de itereaciones

algoritmo_EM <- function(observables, hmm_inicial, iteraciones) {
  hmm<- hmm_inicial
  
  for (i in 1:iteraciones) {
    
    #Paso M: usamos el primer hmm para calcular la secuencia oculta
            #mediante la función viterbilog
    
    secuencia_oculta <- viterbiLog(observables, hmm)
    
    #Paso E: como ya tenemos la secuencica oculta podemos calcular el 
            #hmm estimado mediante la misma
    
    
    hmm_estimado <- estimate.hmm(observables, secuencia_oculta)
    
    hmm <- hmm_estimado
  }

  return(list(hmm = hmm, secuencia_oculta= secuencia_oculta))
}


## (b) Mejorar el modelo proporcionado en el apartado 4 y comparar los
## resultados del apartado 5 con los que se obtengan con el nuevo
## modelo.

hmm_initial <- estimate.hmm(genoma, estados_probables_1)



estados_probables_2 <-algoritmo_EM(genoma, lambda.initial.hmm, 2)
estados_probables_2$secuencia_oculta

estados_2 <- ifelse(estados_probables_2$secuencia_oculta == "h2", 1, 0)
print(estados_2)
length((estados_2))
posiciones_estados_2 <- seq(1:48502)


data_oculta_2 <- data.frame(posiciones_estados_2,estados_2)

# Gráfico de estimación de la secuencia oculta
plot_secuencia_2 <- ggplot(data_oculta_2, aes(x = posiciones_estados_2, y = estados_2)) +
  geom_line() +
  xlab("Posición") +
  ylab("Secuencia Oculta") +
  ggtitle("Secuencia oculta 2")

# Combinar los gráficos en un panel
panel2 <- plot_grid(plot_GC, plot_secuencia_2, ncol = 1, align = "v")
# Mostrar el panel
print(panel2)

## (c) Estudiar la hidrofobicidad local y global de proteínas
## transmembranales.

## Las proteínas transmembranales son aquellas que se encuentra
## insertadas en las membranas atravesando una o varias veces la
## bicapa lipídica. De esta forma en las secuencias de aminoácidos que
## forman estas proteínas se pueden distinguir diferentes
## segmentos. Por una parte, los dominios insertados en la bicapa
## lipídica son altamente hidrofóbicos. Mientras que las regiones
## citosólicas y extracitosólicas son altamente hidrofílicas. Las
## siguientes funciones nos permitirán detectar de forma aproximada
## los dominios hidrofílicos e hidrofóbicos de proteínas
## transmembranales.


## (c.1) Descargar de la web de Uniprot las secuencias de la proteínas
## transmembranales correspondientes a los receptores olfativos del
## Homo sapiens: olfactory receptor 14C36, olfactory receptor 2J3 y
## olfactory receptor 1F1. Lee estas secuencias en formato fasta en R
## siguiendo la convención de usar letras mayúsculas para representar
## aminoácidos.

receptor1 <- read.fasta("receptor14C36.fasta")
receptor2 <- read.fasta("receptor2J3.fasta")
receptor3 <- read.fasta("receptor1F1.fasta")



secuencia1 <- toupper(getSequence(receptor1)[[1]])
secuencia2 <- toupper(getSequence(receptor2)[[1]])
secuencia3 <- toupper(getSequence(receptor3)[[1]])


## El vector hydrophobic.values (definido en el archivo HMM.RData)
## contiene los valores hidrofóbicos de cada aminoácido.

## (c.2) Implementar una función, hydrophobicity, que reciba como
## entrada una secuencia de aminoácidos y devuelva su hidrofobicidad
## global. Esta hidrofobicidad se calcula como la suma de la
## hidrofibicidad de cada aminoácido dividida por la longitud de la
## secuencia de aminoácidos.

hydrophobicity <- function(secuencia) {

  hidrophobicidad <- hydrophobic.values
  
  # Tenemos que cambiar las letras por su valor hidrofobico
  for (i in 1:length(hydrophobic.values)) {
    secuencia <- gsub(names(hydrophobic.values)[i], hydrophobic.values[i], secuencia)
  }
  
  secuencia <- as.numeric(secuencia)
  
  
  # Calcular la hidrofobicidad para cada aminoácido en la secuencia
  hidrophobia_total <- sum(secuencia)
  
  # Calcular la hidrofobicidad global dividida por la longitud de la secuencia
  hidrophobia_global <- hidrophobia_total / length(secuencia)
  
  return(hidrophobia_global)
}

## (c.3) Calcular la hidrofobicidad local y global de los receptores
## olfativos del apartado (c.1). Usa una ventana de longitud 20 con un
## desplazamiento de 1 para el calculo de la hidrofobicidad
## local. Representar en tres gráficos de líneas diferentes la
## hidrofobicidad global y local de cada proteína. ¿Cuántas regiones
## hidrofóbicas se distinguen en cada proteína? ¿Cuántas veces
## atraviesan estas proteínas la capa lipídica?


#calculamos la hidrophobicidad global
hidrofo_1 <- hydrophobicity(secuencia1)
hidrofo_2 <- hydrophobicity(secuencia2)
hidrofo_3 <- hydrophobicity(secuencia3)

#calculamos la hidrophobicidad local


hidrofo_local1 <- local.composition(secuencia1,hydrophobicity , 20, 1)
hidrofo_local2 <- local.composition(secuencia2,hydrophobicity , 20, 1)
hidrofo_local3 <- local.composition(secuencia3,hydrophobicity , 20, 1)



plot( hidrofo_local1$positions, hidrofo_local1$results, type = "l", col = "blue",
     xlab = "Posicion", ylab = "Hidrofobicidad Local",
     main = "Hidrofobicidad Local de Receptor 1")
abline(h=hidrofo_1)

plot( hidrofo_local2$positions, hidrofo_local2$results, type = "l", col = "red",
      xlab = "Posicion", ylab = "Hidrofobicidad Local",
      main = "Hidrofobicidad Local de Receptor 2")
abline(h=hidrofo_2)


plot( hidrofo_local3$positions, hidrofo_local3$results, type = "l", col = "green",
      xlab = "Posicion", ylab = "Hidrofobicidad Local",
      main = "Hidrofobicidad Local de Receptor 3")
abline(h=hidrofo_3)


#1º PROTEINA: se distinguen 10 regiones 

#2º PROTEINA: se distinguen 10 regiones

#3º PROTEINA: se distinguen 13 regiones



## (c.4) Usa la función del apartado (a) para realizar un aprendizaje
## no supervisado con los tres receptores olfativos tomando como
## modelo inicial hidro.initial.hmm y 10 iteraciones.

hidrophia_no_supervisada1 <- algoritmo_EM(secuencia1,hidro.initial.hmm,10)
hidrophia_no_supervisada2 <- algoritmo_EM(secuencia1,hidro.initial.hmm,10)
hidrophia_no_supervisada3 <- algoritmo_EM(secuencia1,hidro.initial.hmm,10)

## (c.5) Muestra gráficamente los resultados obtenidos. Comenta los
## resultados. ¿Qué propones para mejorar la eficiencia de la cadena
## de Markov oculta utilizada?
