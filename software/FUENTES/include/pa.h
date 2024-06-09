/**
 * @file pa.h
 * @author Hossam El Amraoui Leghzali
 * @brief 
 * Asignatura: Metahuristicas
 * Práctica alternativa
 * 
 */

#ifndef PA_H
#define PA_H

#include <algorithm>
#include "p3.h"

/************************************************************
************************************************************
CONSTANTES GLOBALES
************************************************************
************************************************************/
// 1000 0.08
const double L = 0.08;
const int LH = 1000;
// 20
const int NUM_INDIVIDUOS_HGS = 20;
// 1000
const int MAX_ITER_HGS = 15000;

const int NUM_INDIVIDUOS_TSA = 20;
const int MAX_ITER_TSA = 15000;
const int NUM_DISTRITOS = 5;

// Probabilidad alianza
const double PROB_ALIANZA = 0.1;
// Probabilidad de traicion
const double PROB_TRAICION = 0.05;

/************************************************************
************************************************************
ESTRUCTURAS DE DATOS
************************************************************
************************************************************/

typedef std::vector<Solucion> Population;

struct CompararSoluciones {
  bool operator()(const Solucion &sol1, const Solucion &sol2) {
    return sol1.fitness > sol2.fitness;
  }
};

/************************************************************
************************************************************
FUNCIONES AUXILIARES
************************************************************
************************************************************/

/**
 * @brief 
 * Función para ordenar la población de soluciones por fitness y devolver
 * el individuo en la posición indicada
 * 
 * @param poblacion Población de soluciones
 * @param pos Posición del individuo a devolver
 * @return Solucion Individuo en la posición pos
 */
Solucion ordenar_poblacion(const Population &poblacion, int pos);

/************************************************************
************************************************************
HUNGER GAMES SEARCH (HGS)
************************************************************
************************************************************/

/**
 * @brief 
 * Función para calcular el sech de un número
 * 
 * @param x Número a calcular el sech
 * @return double Valor de sech(x)
 */
double sech(double x);

/**
 * @brief 
 * Función para obtener el mejor resultado aplicando Hunger Games Search
 * 
 * @param datos Conjunto de datos
 * @param tam_pob Tamaño de la población
 * @param maxIter Número máximo de iteraciones
 * @return arma::rowvec Mejor resultado obtenido
 */
arma::rowvec HGS(const Dataset &datos, const int &tam_pob, const int &maxIter);

/************************************************************
************************************************************
TRIBUTE SELECTION ALGORITHM (TSA) - Heurística propia
************************************************************
************************************************************/

/**
 * @brief 
 * Función para obtener el mejor resultado aplicando Tribute Selection Algorithm - Heurística propia
 * 
 * @param datos Conjunto de datos
 * @param tam_pob Tamaño de la población
 * @param maxIter Número máximo de iteraciones
 * @return arma::rowvec Mejor resultado obtenido
 */
arma::rowvec TSA(const Dataset &datos, const int &tam_pob, const int &maxIter);

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
 *                  9 para AM-(10,0.1mej), 10 para BMB, 11 para ES, 12 para ILS,
 *                  13 para ILS con Enfriamiento Simulado, 14 para HGS y 
 *                  15 para TSA
 */
void printResultados(int algoritmo);



#endif
