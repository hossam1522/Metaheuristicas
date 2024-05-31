/**
 * @file p3.h
 * @author Hossam El Amraoui Leghzali
 * @brief 
 * Asignatura: Metahuristicas
 * 
 */

#ifndef P3_H
#define P3_H

#include <algorithm>
#include "p2.h"

/************************************************************
************************************************************
CONSTANTES GLOBALES
************************************************************
************************************************************/

// Constantes Enfriamiento Simulado
const double PHI = 0.3;
const double MU = 0.1;
const double Tf = 0.001;
const int MAX_VECINOS_ES = 10;
const double MAX_EXITOS = 0.1;
const int MAX_ITER_ES = 15000;

// Constantes Búsqueda Multiarranque Básica
const int SOL_INICIAL = 20;
const int MAX_ITER_BMB = 750;

// Constantes Búsqueda Local Reiterada
const int ITER_ILS = 20;
const double OPERADOR_MUTACION = 0.2;
const int MAX_ITER_ILS = 750;

/************************************************************
************************************************************
ESTRUCTURAS DE DATOS
************************************************************
************************************************************/

// Estructura para representar una solución
struct Solucion {
  arma::rowvec pesos;
  double fitness;

  // Constructor por defecto
  Solucion(){}

  // Constructor con parámetros
  Solucion(const arma::rowvec &pesos, const double &fitness) : pesos(pesos), fitness(fitness) {}

  // Constructor a partir de un pair de pesos y fitness
  Solucion(const std::pair<arma::rowvec, double> &sol) : pesos(sol.first), fitness(sol.second) {}

  // Operador de comparación
  bool operator<(const Solucion &sol) const {
      return fitness < sol.fitness;
  }
};

/************************************************************
************************************************************
FUNCIONES AUXILIARES
************************************************************
************************************************************/

/**
 * @brief 
 * Función para generar una solución aleatoria
 * 
 * @param datos Conjunto de datos
 * @return Solucion Solución generada
 */
Solucion solucion_aleatoria(const Dataset &datos);

/************************************************************
************************************************************
BÚSQUEDA LOCAL MULTIARRANQUE BÁSICA (BMB)
************************************************************
************************************************************/

/**
 * @brief 
 * Función para obtener la mejor solución de una busqueda local multiarranque básica
 * 
 * @param datos Conjunto de datos
 * @param num_ejecuciones Número de ejecuciones
 * @param maxIter Número máximo de iteraciones
 * @return arma::rowvec Pesos de las características de la mejor solución
 */
arma::rowvec BMB (const Dataset &datos, const int &num_ejecuciones, const int &maxIter);

/************************************************************
************************************************************
ENFRIAMIENTO SIMULADO (ES)
************************************************************
************************************************************/

/**
 * @brief 
 * Función para obtener la mejor solución de un enfriamiento simulado
 * 
 * @param datos Conjunto de datos
 * @param solucion Solución inicial
 * @param maxIter Número máximo de iteraciones
 * @return Solucion Mejor solución encontrada
 */
Solucion ES(const Dataset &datos, const Solucion &solucion, const int &maxIter);

/************************************************************
************************************************************
BÚSQUEDA LOCAL REITERADA (ILS)
************************************************************
************************************************************/

/**
 * @brief 
 * Función para mutar una solución reiniciando el valor de los pesos de las características
 * 
 * @param datos Conjunto de datos
 * @param solucion Solución a mutar
 * @param operadorMutacion Operador de mutación
 * @return Solucion Solución mutadas
 */
Solucion mutacion_ILS(const Dataset &datos, const Solucion &solucion, const double &operadorMutacion);

/**
 * @brief 
 * Función para obtener la mejor solución de una búsqueda local reiterada
 * 
 * @param datos Conjunto de datos
 * @param num_ejecuciones Número de ejecuciones
 * @param maxIter Número máximo de iteraciones
 * @return arma::rowvec Pesos de las características de la mejor solución
 */
arma::rowvec ILS(const Dataset &datos, const int &num_ejecuciones, const int &maxIter);

/**
 * @brief 
 * Función para obtener la mejor solución de una búsqueda local reiterada con enfriamiento simulado
 * 
 * @param datos Conjunto de datos
 * @param num_ejecuciones Número de ejecuciones
 * @param maxIter Número máximo de iteraciones
 * @return arma::rowvec Pesos de las características de la mejor solución
 */
arma::rowvec ILS_ES(const Dataset &datos, const int &num_ejecuciones, const int &maxIter);

/************************************************************
************************************************************
FUNCIONES PARA MOSTRAR RESULTADOS
************************************************************
************************************************************/

/**
 * @brief 
 * Función para mostrar los resultados de la tasa de clasificación y reducción
 * sin ponderaciones, con Greedy Relief, con Búsqueda Local, AGG-BLX, AGG-CA,
 * AGE-BLX, AGE-CA, AM-(10,1.0), AM-(10,0.1) y AM-(10,0.1mej), BMB, ES, ILS y
 * ILS con Enfriamiento Simulado
 * 
 * @param algoritmo Algoritmo a mostrar, 0 para sin ponderaciones, 
 *                  1 para Greedy Relief, 2 para Búsqueda Local,
 *                  3 para AGG-BLX, 4 para AGG-CA, 5 para AGE-BLX,
 *                  6 para AGE-CA, 7 para AM-(10,1.0), 8 para AM-(10,0.1),
 *                  9 para AM-(10,0.1mej), 10 para BMB, 11 para ES, 12 para ILS y
 *                  13 para ILS con Enfriamiento Simulado
 */
void printResultados(int algoritmo);


#endif