## Carga de bibliotecas adicionales
## library("biblioteca") o a través del menú packages

## Algunas funciones de interés

## daisy(dataframe) calcula todas las distancias entre observaciones
## del conjunto de datos.
## cutree(árbol, k) corta los datos provinientes del árbol en k
## grupos distintos.
## sapply(vector.x, function(y) ...) aplica la función function(y)
## sobre todos los elementos de x, siendo y el elemento en cuestión.

######################################################################
## El citocromo c es una proteína que funciona como transportador
## mitocondrial entre dos complejos respiratorios.
######################################################################

######################################################################
## (1) Cargar los datos de amino.acid.sequence.1972 (incluídos en la
## biblioteca cluster.datasets). Una vez cargados, visualizar las
## primeras entradas de la tabla para ver cómo están estructurados los
## datos.
######################################################################

library(cluster)
library(cluster.datasets)
data(amino.acid.sequence.1972)
head(amino.acid.sequence.1972)




######################################################################
## (2) Calcular la tabla de distancias entre los elementos de la
## tabla. Si se intenta hacer directamente, R devolverá una tabla
## llena de elementos NA. Esto es debido a que los valores tienen un
## valor alfanumérico. Para ello, previo al cálculo debemos
## transformar los valores de manera adecuada para que puedan ser
## tratados. Además, si uno se fija, se puede observar como en los
## datos hay alguna columna que no se precisa para la ejecución de
## cualquiera de los algoritmos estudiados.
######################################################################

# Eliminamos las columnas innecesarias, los nombres de los animales y las
#variables categoricas. Además, guardamos los nombres en otra variable porque 
#nos lo piden en otro apartadoS
names <- amino.acid.sequence.1972[,1]
data <- amino.acid.sequence.1972[, -c(1,2,3,6,7,8,9,10,12,14,15,17,21,22,23,24,25,26,27,28,31,32)]


#usando apply, que es más eficiente que los bucles para grandes volumenes de datos
#además de cómodo, modificamos los datos a númericos
data_numeric <- sapply(data, function(x) as.numeric(as.factor(x)))


head(data_numeric)


# Si queremos utilizar "daisy" para medir las distancias, primero debemos
# almacenar nuestros datos en un dataframe
data_df <- as.data.frame(data_numeric)
distances <- daisy(data_df)
distances


######################################################################
## (3) Para comenzar el análisis de estos datos, comenzaremos usando
## un algoritmo de clústering jerárquico, mediante la función hclust
## de la biblioteca estándar stats. Generar una gráfica que represente
## la jerarquía de clústeres del conjunto de datos. Haga que las
## etiquetas de cada rama sea la especie que representa.
######################################################################

library(stats)

cluster <- hclust(distances)

plot(cluster,labels= names)


######################################################################
## (4) Mirando la gráfica anterior se puede observar, en cierta
## manera, cómo están estructurados los datos de la tabla. Viendo la
## jerarquía generada se pueden diferenciar n grupos distintos, con
## los que podemos comenzar a trabajar. Vamos a «cortar» el árbol
## en el número de clústeres correspondientes, y a mostrar una tabla
## cuántos elementos hay en cada uno de los grupos.
######################################################################

grupos <- cutree(cluster, k = 4)

tabla_grupos <- table(grupos)
print(tabla_grupos)
######################################################################
## (5) De hecho, se pueden visualizar estos datos para distintas
## agrupaciones de una vez. Recuerda la función sapply para aplicar
## la misma función a cada uno de los elementos de una lista.
######################################################################


grupos_totales <- sapply(1:8, function(k) cutree(cluster, k = k))
tabla1 <- as.data.frame(grupos_totales)
rownames(tabla1) <- names
print(tabla1)

#De esta forma se puede ver cada animal (eje vertical) en que grupo ha sido
#clasificado tras hacer el corte en grupos

######################################################################
## (6) Ahora, intenta visualizar los individuos de cada uno de los
## grupos. Puedes hacerlo de uno en uno o puedes ver varios grupos
## usando la función sapply.
######################################################################

sapply(1:4, function(g) {
  animales <- names[grupos == g]
  cat("Grupo", g, ":", animales, "\n")
})

######################################################################
## (7) Prueba a visualizar la misma salida con distinto número de
## clústeres.
######################################################################

#para dar un numero distinto de cluster, hay que modificar el valor de k
#antes de utilizar sapply

k<-7
sapply(1:k, function(g) {
  grupos <- cutree(cluster, k =k)
  animales <- names[grupos == g]
  cat("Grupo", g, ":", animales, "\n")
})



######################################################################
## (8) Una vez tenemos el número de clústeres deseado para representar
## los grupos de individuos, pasamos a usar clústering no jerárquico.
## En este caso, usaremos la función pam de la biblioteca cluster.
## Una vez usado, muestre en una tabla los resultados obtenidos
## comparados con los grupos obtenidos en el ejercicio anterior.
## ¿Qué se puede concluir de esto? ¿Esperabas este resultado?
######################################################################

grupos <- cutree(cluster, k = 4)
cluster2 <- pam(data_numeric, k = 4)
cluster2_results <- cluster2$clustering

tabla_comparativa <- data.frame(Jerarquico = grupos, No_Jerarquico = cluster2_results)
rownames(tabla_comparativa) <- names
print(tabla_comparativa)


#Vemos que salvo para el caso del pato, el pollo y la serpiente de cascabel,
# los clusteres asignados a cada individuo para una selección de 4 clusteres 
#es la misma. Es previsible este resultado al estar muy proximos en el eje
#horizontal pero formando parte de otras ramificaciones verticales. Por lo que
#al deshacer el criterio jerargico esta asignación de grupos se ve modificada

######################################################################
## (9) Por último, muestre un gráfico de silueta del último clústering
## realizado. En el gráfico aparece la anchura media de la silueta,
## que indica la «naturalidad» de la estructura encontrada.
## De 0,71 a 1, se ha encontrado una estructura fuerte.
## De 0,51 a 0,7, se encuentra una estructura razonable.
## De 0,26 a 0,5, la estructura es débil y podría ser artificial.
## Si es menor que 0,25, no se ha encontrado una estructura
## sustancial.
## ¿Por qué crees que has obtenido los datos que has obtenido?
######################################################################


sil <- silhouette(cluster2, distances)


plot(sil, main = "Gráfico de Silueta", xlab = "Silueta", ylab = "Cluster")

#hemos encontrado estrucutras debiles ya que los grupos generados son muy extensos
# e incluyen especies que aunque pertenecientes al mismo tipo están muy alejadas 
#unas de otras por lo tanto las agrupaciones que forman son debiles

