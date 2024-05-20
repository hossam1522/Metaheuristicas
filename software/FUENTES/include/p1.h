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
 * @param v_pesos Vector de pesos inicial
 * @param iteraciones Número de iteraciones
 * @param max_vecinos Número máximo de vecinos a considerar
 * @param max_iter Número máximo de iteraciones
 * @return pair<arma::rowvec, double> Pesos de las características y fitness
 */
std::pair<arma::rowvec, double> busquedaLocal(const Dataset &datos, const arma::rowvec &v_pesos,
                                              int &iteraciones, const int max_vecinos, const int max_iter);


#endif