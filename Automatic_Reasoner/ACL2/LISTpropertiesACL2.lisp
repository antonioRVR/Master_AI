;;; RA: inversa-concatena.lisp
;;; Ejemplo de definición y demostración de una propiedad.
;;; Departamento de Ciencias de la Computación e Inteligencia Artificial
;;; Universidad de Sevilla
;;;============================================================================

;Una lista l1 es un prefjo de otra lista l2, si todos los elementos de l1 aparecen en el
;mismo orden y consecutivos al comienzo de l2. Por ejemplo, la lista '(1 2) es prefijo de
;'(1 2 3 4), pero no es prefjo de '(2 3 1 4) ni de '(2 1 3 4). Este concepto se formaliza
;en Acl2 mediante la siguiente funcion:

(set-gag-mode nil)

(defun prefijo (l1 l2)
(cond ((endp l1) (equal l1 nil))
((endp l2) nil)
((equal (car l1) (car l2)) (prefijo (cdr l1) (cdr l2)))
(t nil)))
; 1: primer lista vacia, devuelve nulo y True si la lista es nulo
; 2: segundo lista vacia, devuelve nulo
; 3: si el primer elemento de ambas listas es el mismo,	devuelve True
;			y vuelve a llmar a la funcion para los demás elementos de la lista
; 4: si no se cumple nada, devuelve nulo


;Una lista l1 es un trozo de otra lista l2, si todos los elementos de l1 aparecen en el
;mismo orden y consecutivos en algun punto de l2. Por ejemplo, la lista '(2 3) es un
;trozo de '(1 2 3 4 5), pero no es prefijo de '(3 1 2 4)ni de '(1 3 2 4). Este concepto
;se formaliza en Acl2 mediante la siguiente funcion:

(defun trozo (l1 l2)
(cond ((endp l2) (equal l1 nil))
      (t (or (prefijo l1 l2) (trozo l1 (cdr l2))))))
; 1: si la secunda lista es vacia, devuelve true, entonces devuelve true
;			si L1 es igual a nil
; 2: si no se da la segunda condición devuelve o:
;			*el resultado de prefijo l1 l2
;			*el resultado de llamar recursivamente a trozo con l1 y 
;				el resto de l2


;En estas condiciones se tiene que la relacion trozo es transitiva y estable con respecto
;a la concatenacion:



(defthm prefijo-del-prefijo
	(implies (and(prefijo l1 l2)
						(prefijo l2 l3))
						(prefijo l1 l3))
)

(defthm l1-es-trozo-tambien-en-el-prefijo
  (implies (and (trozo l1 l2)
	     					(prefijo l2 l3))
	     			(trozo l1 l3)))

(defthm trozo-transitiva
(implies (and (trozo l1 l2)
(trozo l2 l3))
	 (trozo l1 l3)))
; trata de probar que si l1 es trozo de l2 y l2 es trozo de l3, l1 es trozo de l3

;Se obtiene un error que nos dice que:
; 1: Se comprueba si L3 es una lista no vacia
; 2: Se comprueba si L1 no es prefijo de L3
; 3: Se comprueba si L2 no es una sublista de el resto de L3
; 4: Se comprueba si L1 es una sublista de L2
; 5: se comprueba si L2 es prefijo de L3
; Finalmente plantea si L1 es una sublista del resto de L3
; por lo que dadas las primeras condiciones L1 tiene que ser una sublista de 
; L3.
; Para solucionar este problema se han escrito los teoremas encima de la relacion
; de transitividad con respecto a la concatenación



(defthm no-prefijo-auxiliar
    (implies (not(prefijo l1 (cons(car l2)(append (cdr l2)l3) )))
	     (not (prefijo l1 l2))))
  
(defthm trozo-append-1
(implies (trozo l1 l2)
	 (trozo l1 (append l2 l3))))


(defthm trozo-append-2
(implies (trozo l1 l2)
	 (trozo l1 (append l3 l2))))

;Tambien se tiene que si la lista l1 es un trozo de la lista l2, y l2 es un trozo de l1,
;entonces ambas listas son iguales:

(defthm trozo-equal
(implies (and (trozo l1 l2)
(trozo l2 l1))
(equal l1 l2))
  :rule-classes nil)

;La directiva :rule-classes nil indica al sistema Acl2 que la regla no debe usarse como
;regla de reescritura y no afecta para nada al proceso de demostracion de la propiedad.
;Se pide demostrar estos cuatro resultado en Acl2.


