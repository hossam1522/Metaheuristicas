/**
 * @file p1.h
 * @author Hossam El Amraoui Leghzali
 * @brief 
 * Asignatura: Metahuristicas
 * 
 */

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
CLASIFICADOR 1-NN
************************************************************
************************************************************/

/**
 * @brief 
 * Función para clasificar un ejemplo con el clasificador 1-NN
 * 
 * @param ejemplo Ejemplo a clasificar
 * @param datos Conjunto de datos de entrenamiento
 * @param pesos Pesos para las características
 * @return std::string  Categoría a la que pertenece el ejemplo
 */
std::string clasificador1NN(const arma::rowvec &ejemplo, const Dataset &datos,
                            const arma::rowvec &pesos);


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
arma::rowvec enemigoMasCercano(const Dataset &ejemplo, const Dataset &datos);

/**
 * @brief 
 * Función para calcular el vecino más cercano a un ejemplo con la misma clase
 * 
 * @param ejemplo Ejemplo a clasificar
 * @param datos Conjunto de datos de entrenamiento
 * @return arma::rowvec Vecino más cercano
 */
arma::rowvec amigoMasCercano(const Dataset &ejemplo, const Dataset &datos);

/**
 * @brief 
 * Función para calcular el valor de los pesos de las características con Greedy Relief
 * 
 * @param datos Conjunto de datos de entrenamiento
 * @return arma::rowvec Pesos de las características
 */
arma::rowvec greedyRelief(const Dataset &datos);

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

/************************************************************
************************************************************
FUNCIONES DE EVALUACIÓN
************************************************************
************************************************************/

/**
 * @brief 
 * Función para calcular la tasa de clasificación de un conjunto de datos
 * 
 * @param test Conjunto de datos de test
 * @param entrenamiento Conjunto de datos de entrenamiento
 * @param pesos Pesos para las características
 * @return double Tasa de clasificación
 */
double tasa_clas(const Dataset &test, const Dataset &entrenamiento, const arma::rowvec &pesos);

/**
 * @brief 
 * Función para calcular la tasa de clasificación de un conjunto de datos
 * usando la técnica de Leave-One-Out
 * 
 * @param entrenamiento Conjunto de datos de entrenamiento
 * @param pesos Pesos para las características
 * @return double Tasa de clasificación
 */
double tasa_clas(const Dataset &entrenamiento, const arma::rowvec &pesos);


/**
 * @brief 
 * Función para calcular la tasa de reducción de un conjunto de datos
 * 
 * @param pesos Pesos para las características
 * @return double Tasa de reducción
 */
double tasa_red(const arma::rowvec &pesos);

/**
 * @brief 
 * Función para calcular la función objetivo
 * 
 * @param tasa_clas Tasa de clasificación
 * @param tasa_red Tasa de reducción
 * @return double Valor de la función objetivo
 */
double fitness(const double &tasa_clas, const double &tasa_red);
