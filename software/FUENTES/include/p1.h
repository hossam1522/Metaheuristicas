/**
 * @file p1.h
 * @author Hossam El Amraoui Leghzali
 * @brief 
 * Asignatura: Metahuristicas
 * 
 */

#ifndef P1_H
#define P1_H

#include <iostream>
#include <iomanip>
#include <armadillo>
#include <string>
#include "random.hpp"
#include "util.h"

using Random = effolkronium::random_static;


/************************************************************
************************************************************
CONSTANTES GLOBALES
************************************************************
************************************************************/

// Desviación típica para distribución normal
const float SIGMA = sqrt(0.3);

// Número máximo de iteraciones para el algoritmo de Búsqueda Local
const int MAX_ITER = 15000;

// Constante a multiplicar por el número de características
// para obtener el número máximo de vecinos a considerar 
const int CONST_MAX_VECINOS = 20;

/************************************************************
************************************************************
GREEDY RELIEF
************************************************************
************************************************************/

/**
 * @brief 
 * Función para calcular el vecino más cercano a un ejemplo con diferente clase
 * 
 * @param ejemplo Ejemplo a clasificar
 * @param datos Conjunto de datos de entrenamiento
 * @return arma::rowvec Vecino más cercano
 */
int enemigoMasCercano(const Dataset &ejemplo, const Dataset &datos);

/**
 * @brief 
 * Función para calcular el vecino más cercano a un ejemplo con la misma clase
 * 
 * @param ejemplo Ejemplo a clasificar
 * @param datos Conjunto de datos de entrenamiento
 * @return arma::rowvec Vecino más cercano
 */
int amigoMasCercano(const Dataset &ejemplo, const Dataset &datos);

/**
 * @brief 
 * Función para calcular el valor de los pesos de las características con Greedy Relief
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @return arma::rowvec Pesos de las características
 */
arma::rowvec greedy(const Dataset &datos);

/************************************************************
************************************************************
BUSQUEDA LOCAL
************************************************************
************************************************************/

/**
 * @brief 
 * Función para calcular el valor de los pesos de las características con Búsqueda Local 
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @return arma::rowvec Pesos de las características
 */
arma::rowvec busquedaLocal(const Dataset &datos);

/************************************************************
************************************************************
FUNCIONES PARA MOSTRAR RESULTADOS
************************************************************
************************************************************/

/**
 * @brief 
 * Función para mostrar los resultados de la tasa de clasificación y reducción
 * sin ponderaciones, con Greedy Relief y con Búsqueda Local
 * 
 * @param algoritmo Algoritmo a mostrar, 0 para sin ponderaciones, 
 *                  1 para Greedy Relief y 2 para Búsqueda Local
 */
void printResultados(int algoritmo);

#endif