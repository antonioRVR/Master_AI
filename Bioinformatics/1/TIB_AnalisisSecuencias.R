## Carga de bibliotecas adicionales
## library("biblioteca") o a través del menú Packages

## Variables y asignación 
## variable <- valor

## Funciones
## función(datos, ..., arg1 = valor1, ...)

## Vectores: Todos los elementos de la misma naturaleza
## * Creación:
##   - c(dato1, dato2, ...)
##   - inicio:fin siendo inicio y fin dos números cualesquiera
## * Selección:
##   - Por los índices de las posiciones: vector[vector.índices]
##   - Por exclusión de las posiciones: vector[-vector.índices]

## Matrices
## Acceso:
## - Por el índice de la columna: matriz[, índice.col]

## Listas: Cada elemento puede tener una naturaleza diferente.
## Acceso:
## - Por el índice del elemento: lista[[índice]]
## - Por el nombre del elemento (no siempre disponible): lista$nombre

## Gráficos: n.color, n.línea y n.punto son un entero indicando el
## color, el tipo de línea y el tipo de punto, respectivamente (por
## defecto 1 en ambos casos, negro, continua respectivamente).
## * En general: main, xlab, ylab (título y etiquetas de los ejes x e
##   y).
## * De líneas:
##   plot(coord.x, coord.y, type = "l", col = n.color, lty = n.línea)
## * Histograma (de densidad):
##   hist(datos, freq = FALSE)
## * Diagrama de caja y bigote
##   boxplot(datos1, datos2, ..., col = vector.colores,
##           names = vector.nombres)

## Añadidos a un gráfico:
## * Líneas verticales a distancias vector.x del origen
##   abline(v = vector.x, col = vector.color, lty = vector.línea)
## * Líneas horizontales a distancias vector.y del origen
##   abline(h = vector.y, col = vector.color, lty = vector.línea)
## * Líneas
##   lines(x.coord, y.coord, col = n.color, lty = n.línea)
## * Puntos
##   points(x.coord, y.coord, col = n.color, pch = n.punto)
## * Leyenda para un gráfico de líneas
##   legend(x.pos, y.pos, vector.nombres, col = vector.colores, 
##          lty = vector.líneas)

## plot(1:25, pch = 1:25, col = rep(1:8, length.out = 25))
## abline(h = 1:6, col = 1:6, lty = 1:6)

######################################################################
## Haemophilus influenzae (NC_000907) es una bacteria gram-negativa
## que fue erróneamente considerada la causa de la gripe hasta que en
## 1933 se descubrió la causa viral de la misma.

## El genoma de H. influenzae fue el primer genoma en secuenciarse de
## un organismo de vida libre por el grupo de Craig Venter en 1995.
######################################################################

######################################################################
## (1) Descargar su genoma de la base de datos de NCBI (como un
## fichero fasta) y, utilizando la función read.fasta de la biblioteca
## seqinr, guardar dicha información como hi.


#primero añadimos el paquete "seqinr" mediante la pestaña "packages" y
#clickando en "update". Tras instalarlo, lo activamos haciendo click en
#el mismo


#tambien podemos instalarlo y activarlo de la siguiente manera:
#install.packages("seqinr")
library("seqinr")

#ya podemos usarlo

hi <- read.fasta("sequence.fasta")
######################################################################

######################################################################
## (2) Utilizar la función getSequence para obtener, del primer
## elemento (y único) de hi (un objeto de la clase SeqFastadna
## anterior), la secuencia de ADN. Guardar dicha información como
## genomaHi

genomaHi <- getSequence(hi)[[1]]

######################################################################

######################################################################
## La siguiente función calcula (para una secuencia, seq) el valor
## local de una función, func, utilizando una ventada de longitud
## window.length y un desplazamiento offset.

local.composition <- function(seq, func, window.length, offset) {

    ## Paso 1: Inicialización de variables antes del bucle:
    ## low.extrems ← serie aritmética de razón offset con inicio en 1
    ## y con (tamaño de seq) - window.length + 1 como tope superior
    ## results ← matriz vacía (numérico)
    low.extrems <- seq(1, length(seq) - window.length + 1, by = offset)
    results <- numeric()
    #ha sido necesario modificar length(seq) por nchar(seq) o de lo contrario 
    #se obtenian
    
    
    ## Paso 2: Para cada uno de los elementos de low.extrems hace:
    for (i in low.extrems) {
        
        ## Paso 2.1: Añadir a results una nueva fila con los valores
        ## de func sobre el trozo de seq entre i e (i + window.length
        ## - 1)
        results <- rbind(results, func(seq[i:(i+window.length-1)]))
    }
    
    ## Paso 3: Devolver una lista con los valores de results y
    ## low.extrems como los elementos results y positions
    ## (respectivamente).
    list(results = results, positions = low.extrems)
}

######################################################################

######################################################################
## (3) Calcular el contenido en GC, global y local utilizando una
## ventana de longitud 20000 y un desplazamiento de 2000. Representar, 
## utilizando un gráfico de líneas, los resultados obtenidos.
## Nota: Utilizar la función GC de la biblioteca seqinr.

#Utilizamos la función anterior para calcular el valor local del contenido en GC
#una vez calculado guardamos las posiciones y sus resultados. Tras esto llamamos
#a la función de graficos y finalmente incorporamos lineas.

GC_local <- local.composition(genomaHi,GC,20000,2000)
GC_local_positions <- GC_local[[2]]
GC_local_results <- GC_local[[1]]

plot(GC_local_positions,GC_local_results, cex=0.5)
lines(GC_local_positions,GC_local_results)


#Para calcular el valor global de contenido en GC llamamos a la función GC
#que nos dará la proporción de GC que hay entre todas las bases distintas
GC_global <- GC(genomaHi)
GC_global

#Podemos representar el valor GC_global en nuestro grafico para ver como varia
#el valor local en torno al mismo y vemos que se mantiene en la media

abline(h= GC_global) 


#la media y el valor global tambien se puede calcular con mean()

mean(GC_local_results)
######################################################################

######################################################################
## El contenido en GC de un genoma, la proporción de los nucleótidos g
## y c, es una característica muy específica de cada organismo en
## concreto.

## La transferencia horizontal de genes hace referencia a la
## transmisión de genes entre organismos vivos (comúnmente de
## diferentes especies) sin que uno sea el progenitor del otro.

## Debido a que el contenido en GC es una de las características más
## específicas de cada especie, la identificación de zonas en el
## genoma cuyo contenido en GC diverge significativamente del
## contenido GC global puede indicar la existencia de un evento de
## transferencia horizontal de genes.

## ¿Se observa en el gráfico anterior una desviación significativa?
## ¿Puede esto indicar transferencia horizontal de genes?


######################################################################

######################################################################
## (4) Identificar los puntos en los que se observa la mayor
## desviación

#Si, se observan diversos picos significativos donde la proporción de G y C
# varía significativamente

desvios_pos <- c(GC_local_positions[GC_local_results >0.41])
desvios_pos


desvios_neg <- c(GC_local_positions[GC_local_results <0.35])
desvios_neg

#para visualizar las posiciones de nuestro intervalo local donde se hallan los
#mayores desvios hacemos lo siguiente:


desvios_pos_local <- which(GC_local_results >0.41)
desvios_neg_local <-which(GC_local_results <0.35)
desvios_pos_local
desvios_neg_local

######################################################################

######################################################################
## (5) Comprobar, visualizando los datos y con un test de
## Shapiro-Wilk, si el contenido en GC local en los distintos tramos
## sigue una distribución normal.

#Tras observar donde se hallan los mayores desvios, guardamos estos
#intervalos para realizar la comparación. Uniendo en un solo tramo 
# los valores que no deben tener transferencia horizontal de genes
# GC_1

GC_1 <- c(GC_local_results[1:454],GC_local_results[462:777],GC_local_results[794:900])
GC_2 <- c(GC_local_results[455:461])
GC_3 <- c(GC_local_results[778:793])
GC_4 <- c(GC_local_results[901:906])

shapiro.test(GC_1)
shapiro.test(GC_2)
shapiro.test(GC_3)
shapiro.test(GC_4)




plotn <- function(x,main="Histograma de frecuencias \ny distribución normal",
                  xlab="X",ylab="Densidad") {
  min <- min(x)
  max <- max(x)
  media <- mean(x)
  dt <- sd(x)
  hist(x,freq=F,main=main,xlab=xlab,ylab=ylab)
  curve(dnorm(x,media,dt), min, max,add = T,col="blue")
}

plotn(GC_1,main="GC1")
plotn(GC_2,main="GC2")
plotn(GC_3,main="GC3")
plotn(GC_4,main="GC4")



#Solo en los tramos 2,3  y 4 se tiene una distribución normal al tener
# p > 0.05. Curiosamente estos son los tramos que se pueden corresponder a
#una transferencia horizontal de genes

######################################################################

######################################################################
## (6) Determinar, utilizando un test adecuado, si la diferencia entre
## los tramos es significativa.

#Para determinar si la diferencia entre los tramos es significativa,
#utilizamos el test t de Student, cuya utilidad radica en la determinación
# de diferencias entre dos varianzas muestrales para cosntruir un intervalo de
#confianza. Si el resultado es <0.05, los tramos tienen una diferencia significativa 

# Este test no es el más adecaudo de utilizar cuando las distribuciones no son normales,
#sin embargo solamente los tramos 1 y 5 no son normales pero al tener un gran numero de
#muestras se puede utilizar 

t.test(GC_1,GC_2)
t.test(GC_1,GC_3)
t.test(GC_1,GC_4)


#se obtiene p valor muy reducido por lo que se determina que hay una diferencia significativa
#entre los tramos y por lo tanto deben corresponder a transferencia horizontal de genes

######################################################################

######################################################################
## (7) Visualizar, utilizando diagramas de caja y bigote, la
## significancia del resultado obtenido en el apartado anterior.


## - Un asterisco p-valor < 0.05
## - Dos asteriscos p-valor < 0.01
## - Tres asteriscos p-valor < 0.001
t.test12 <- t.test(GC_1, GC_2)
t.test13 <- t.test(GC_1, GC_3)
t.test14 <- t.test(GC_1, GC_4)
t.test23 <- t.test(GC_2, GC_3)
t.test24 <- t.test(GC_2, GC_4)
t.test34 <- t.test(GC_3, GC_4)

t.test12
t.test13
t.test14
t.test23
t.test24
t.test34

comparar <- list(GC_1, GC_2, GC_3, GC_4)

comparar <- list(GC_1, GC_2, GC_3, GC_4)
boxplot(comparar, main="Diagrama de caja", xlab="Grupos", ylab="Valores")


# Añadir asteriscos en función del valor p. Las comparaciones del fondo se
#realizaran con asteriscos rojos. Las comparaciones de los picos 3 y 4 con el
#pico 2 se realizaran en color verde y finalmente las comparación del pico
# 3 y 4 se realizará en color azul 

text(2,0.4,"***", col = "red")
text(3,0.4,"***", col = "red")
text(4,0.4,"***", col = "red")
text(3,0.39,"***", col = "green")
text(4,0.39,"***", col = "green")
text(4,0.38,"***", col = "blue")





#una vez graficado se observa como los diagramas de cajas no se superponen
# se evidencia unas normales y distribución totalmente independientes entre si

######################################################################

######################################################################
## Tarea:
## Realizar un análisis parecido para el genoma de la bacteria
## Methanocaldococcus jannaschii (NC_000909). Se trata de una archaea
## metanógena termofílica que habita ventanas hidrotermales creciendo
## usando como fuente de energía dióxido de carbono e hidrógeno y
## produciendo metano como producto secundario de su metabolismo.

## El grupo de Craig Venter fue el primero en secuenciar su genoma en
## 1996 que constituyó el primer genoma de archaea en secuenciarse
## completamente. La secuenciación de su genoma produjo evidencias
## claves para la existencia de los tres dominios de la vida (Archaea,
## Bacteria y Eukarya).

## Ref: "Compositional Biases of Bacterial Genomes and Evolutionary
## Implications", S. Karlin, J. Mrázek, A.M. Campbell, Journal of
## Bacteriology, June 1997, p. 3899–3913


######################################################################


Mj <- read.fasta("Methanocaldococcus_jannaschii.fasta")

genomaMj <- getSequence(Mj)[[1]]


######################################################################

######################################################################
## (3) Calcular el contenido en GC, global y local utilizando una
## ventana de longitud 20000 y un desplazamiento de 2000. Representar, 
## utilizando un gráfico de líneas, los resultados obtenidos.
## Nota: Utilizar la función GC de la biblioteca seqinr.

#Utilizamos la función anterior para calcular el valor local del contenido en GC
#una vez calculado guardamos las posiciones y sus resultados. Tras esto llamamos
#a la función de graficos y finalmente incorporamos lineas.

GC_local_Mj <- local.composition(genomaMj,GC,20000,2000)
GC_local_positions_Mj <- GC_local_Mj[[2]]
GC_local_results_Mj <- GC_local_Mj[[1]]

plot(GC_local_positions_Mj,GC_local_results_Mj, cex=0.5)
lines(GC_local_positions_Mj,GC_local_results_Mj)


#Para calcular el valor global de contenido en GC llamamos a la función GC
#que nos dará la proporción de GC que hay entre todas las bases distintas
GC_global_Mj <- GC(genomaMj)
GC_global_Mj

#Podemos representar el valor GC_global en nuestro grafico para ver como varia
#el valor local en torno al mismo y vemos que se mantiene en la media

abline(h= GC_global_Mj) 


#la media y el valor global tambien se puede calcular con mean()

mean(GC_local_results_Mj)


######################################################################

######################################################################
## El contenido en GC de un genoma, la proporción de los nucleótidos g
## y c, es una característica muy específica de cada organismo en
## concreto.

## La transferencia horizontal de genes hace referencia a la
## transmisión de genes entre organismos vivos (comúnmente de
## diferentes especies) sin que uno sea el progenitor del otro.

## Debido a que el contenido en GC es una de las características más
## específicas de cada especie, la identificación de zonas en el
## genoma cuyo contenido en GC diverge significativamente del
## contenido GC global puede indicar la existencia de un evento de
## transferencia horizontal de genes.

## ¿Se observa en el gráfico anterior una desviación significativa?
## ¿Puede esto indicar transferencia horizontal de genes?


######################################################################

######################################################################
## (4) Identificar los puntos en los que se observa la mayor
## desviación

#Si, se observan diversos picos significativos donde la proporción de G y C
# varía significativamente

desvios_pos_Mj <- c(GC_local_positions_Mj[GC_local_results_Mj >0.34])
desvios_pos_Mj


desvios_neg_Mj <- c(GC_local_positions_Mj[GC_local_results_Mj <0.27])
desvios_neg_Mj

#para visualizar las posiciones de nuestro intervalo local donde se hallan los
#mayores desvios hacemos lo siguiente:


desvios_pos_local_Mj <- which(GC_local_results_Mj >0.34)
desvios_neg_local_Mj <-which(GC_local_results_Mj <0.27)
desvios_pos_local_Mj
desvios_neg_local_Mj

######################################################################

######################################################################
## (5) Comprobar, visualizando los datos y con un test de
## Shapiro-Wilk, si el contenido en GC local en los distintos tramos
## sigue una distribución normal.

#En esta ocasión se identifican 2 picos, repetimos el procedimiento anterior

GC_1_Mj <- c(GC_local_results_Mj[1:68],GC_local_results_Mj[80:311],GC_local_results_Mj[323:823])
GC_2_Mj <- c(GC_local_results_Mj[69:79])
GC_3_Mj <- c(GC_local_results_Mj[312:322])



shapiro.test(GC_1_Mj)
shapiro.test(GC_2_Mj)
shapiro.test(GC_3_Mj)


plotn(GC_1_Mj,main="GC1_Mj")
plotn(GC_2_Mj,main="GC2_Mj")
plotn(GC_3_Mj,main="GC3_Mj")



#En este caso no se tiene distribución normal para ningun pico, además sus
#muestras son pequeñas por lo que el test t de student no es apropiado

######################################################################

######################################################################
## (6) Determinar, utilizando un test adecuado, si la diferencia entre
## los tramos es significativa.

#La prueba de Mann-Whitney U evalúa si las dos muestras provienen de 
#la misma población o si las medianas de las dos muestras son diferentes. 
#El resultado de la prueba es un valor U, que representa el número de veces 
#que se observa que los valores en una muestra son mayores que los valores en 
#la otra muestra. Si el valor U es pequeño, esto sugiere que hay una gran 
#diferencia entre las dos muestras.


# Comparación entre GC_1_Mj y GC_2_Mj
wilcox.test(GC_1_Mj, GC_2_Mj)

# Comparación entre GC_1_Mj y GC_3_Mj
wilcox.test(GC_1_Mj, GC_3_Mj)



#se obtiene p valor muy reducido por lo que se determina que hay una diferencia significativa
#entre los tramos y por lo tanto deben corresponder a transferencia horizontal de genes

######################################################################

######################################################################
## (7) Visualizar, utilizando diagramas de caja y bigote, la
## significancia del resultado obtenido en el apartado anterior.


## - Un asterisco p-valor < 0.05
## - Dos asteriscos p-valor < 0.01
## - Tres asteriscos p-valor < 0.001


comparar_Mj <- list(GC_1_Mj, GC_2_Mj, GC_3_Mj)
boxplot(comparar_Mj, main="Diagrama de caja", xlab="Grupos", ylab="Valores")
text(2,0.34,"***")
text(3,0.34,"***")


#una vez graficado se observa como los diagramas de cajas solamente se superponen
#ligeramente para las zonas con transición de genes. Sería interesante evaluar 
#si esta transición podria venir de la misma muestra